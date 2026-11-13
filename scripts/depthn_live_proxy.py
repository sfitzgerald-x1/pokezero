#!/usr/bin/env python3
"""One listening address in front of several depthn_live_server processes: each battle sticks to one backend; other traffic goes least busy.

Design. The panel's dispatch talks to ONE search server address (--server HOST:PORT), and the actor's client opens one connection per
request (phase_d_search_server framing: one request, one response). A server process answers one decision at a time (max_requests 1;
the constructor's prepare is serial per root), so several processes per GPU serve the live eval. This proxy forwards each accepted
connection, byte for byte in both directions, to the backend with the fewest connections in flight (ties: the next in round-robin
order; a backend whose connection setup failed is down, below), and closes both sides when either
closes. A backend's slot is reserved before its connection is opened, so connections accepted in the same instant spread out. It never
rewrites, retries or replays a request: connection setup alone is retried on the next backend (--connect-timeout each, before any byte
is forwarded); when no backend accepts, the client's connection is closed, which its client records as a transport failure
(connection_error, or server_error on a bare EOF) and plays greedy, as with a dead server.
- Battle affinity (sticky): a server keeps one constructor session per battle, so a battle whose decisions scatter over the pool
  folds from a stale or empty session on every switch (pre-flight r3: battles touched 8-12 of 12 servers, prepare p95 2.04 s against
  1.20 s on the same server, 28% timeouts). The proxy therefore reads each connection's first frame (the request: a 4-byte length
  and its JSON, within --first-frame-timeout) for its battle key (battle_id, owner: the two seats of one seed share a battle_id and
  are separate sessions on the server), sends those bytes on verbatim, and routes: a battle seen before goes
  to its home backend unless that backend is cooling down; a new battle (or a re-homed one) goes, the lead's rule, to the GPU with
  the fewest live battles, then the fewest requests in flight on it, then the GPU least recently given a new battle, and within
  it to the backend with the fewest live battles, then in flight, then --backend order (--backend HOST:PORT@GPU names each
  backend's GPU). Live: a decision within --sticky-idle seconds (default 60), until the battle's end frame. A worker sends one
  ({"schema": BATTLE_END_SCHEMA, "battle_id", "owner"}, pokezero.depthn_live_actor.Client.end) after its game's result is
  recorded; the proxy answers it (status ENDED, or UNKNOWN_BATTLE, logged and otherwise ignored), never forwards it, and the battle
  stops counting at once, so finished battles never steer a new one onto a busier GPU (r4a3 pre-flight runs 2 and 3: stale
  counts tied idle GPUs with a busy one and backend order put three battles on GPU 0; run 2: 7 timeouts). The idle window stays
  as the fallback for a lost end frame; an end frame is idempotent and keeps the battle's home. Concurrent pricing on one GPU
  costs real time (r3: price p95 1.07 s against the bench's 0.67 s), so battles spread across GPUs first. The home map is an LRU of at most --sticky-cap battles. A request without a string battle_id keeps the
  least-busy order below. Each connection's log row records its battle_id, owner, backend, GPU and whether it went home (home false
  with a battle_id: an assignment). Connections that send no request (a health check, a client that gave up) are counted apart
  (no_request), not as unserved.
- Startup identity (the lead's rule: a pool must not hide a wrong backend): --probe-requests PATH (JSONL, one request per line in the
  servers' wire format, e.g. built from a closed RAW cell's owning-wire capture as the panel's probe builds them) is sent to EVERY backend
  directly, each request under a fresh request_id and battle id, before the proxy listens. Per backend, requests are tried in order: a
  constructor, DepthN or root-class refusal moves to the next (as the panel's probe does); anything else that is not COMPLETE, or no
  COMPLETE at all, fails. The COMPLETE answer must carry the rung, archive_sha256, requested_worlds, root_operator and
  pricer.matmul_precision given by --expect-rung / --expect-archive-sha256 / --expect-worlds / --expect-root-operator /
  --expect-matmul-precision, its reply_prior record's reply_archive_sha256 ('none' without a reply model) given by
  --expect-reply-archive-sha256, its views mode given by --expect-reply-views, and whether it was built with peer view b
  (--expect-peer-view-b on/off), and all backends must agree on those, on tree_commit and on the whole pricer identity (evaluator, W, K,
  buckets, precision); otherwise the proxy exits non-zero. (The rung is the request's own, echoed; a backend serving another rung
  refuses the probe as a bad request, which fails it. requested_worlds is the W check.)
- Health (r5a: a preempted pod's two servers, ranked first again after every --cooldown because they held no battles, added two
  0.25 s connect timeouts to every new battle's first decision for an hour). A backend whose connection setup fails (refused, or no
  answer within --connect-timeout) is DOWN: out of every ranking, battle homes included, so it costs one connect timeout when it
  dies and none after. Every --health-interval seconds (default 1) each down backend is reconnected to; it ranks again only once it
  accepts a connection and, with --probe-requests, answers the identity probe with the identity the pool started with (else it is
  QUARANTINED: never retried until a reload names it again). Only when every backend is down are they all tried. HEALTH lines
  (DOWN, UP, QUARANTINED) go to stdout; POOL lines mark each backend down or not.
- --backends-file PATH (instead of --backend): one HOST:PORT[@GPU] per line, '#' comments. SIGHUP re-reads it and the proxy serves
  exactly its backends, without a restart (so no battle loses its home): a removed backend takes no new connection (those in flight
  finish; its battles re-home on their next decision); an added one, or a quarantined one still named, must accept a connection and
  pass the identity probe (with --probe-requests) before it joins. All or nothing: an unreadable or invalid file (the GPU rule
  below, the listen address, a joining backend that fails) changes nothing. RELOAD or RELOAD_REJECTED goes to stdout. Without
  --backends-file a SIGHUP is logged as rejected (it no longer kills the proxy).
- --wait-ready SECONDS (0 to 3600): before the identity probe, wait until every backend accepts a connection, else exit non-zero.
- --connection-log PATH (JSONL, appended, flushed): one row per client connection (backend, opened, closed, seconds, bytes each way,
  or 'unserved'), so latency breaks down by backend (the servers' own server_id in each response is the panel readout's breakdown). READY (one JSON line, with each backend's probe
  answer) and periodic POOL lines go to stdout; send stdout to a file or the container log.
"""
import argparse
import asyncio
from collections import Counter, OrderedDict
import json
import math
import signal
import struct
import sys
import time
import uuid

MAX_PACKET = 16 * 1024 * 1024
# A worker's end of one battle (pokezero.depthn_live_actor.Client.end, sent after the game's result is recorded): answered here,
# never forwarded. The battle stops counting toward its GPU's load at once instead of --sticky-idle seconds after its last decision.
BATTLE_END_SCHEMA = 'depthn-live.battle-end.v1'


def address(text):
    host, sep, port = text.rpartition(':')
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:raise argparse.ArgumentTypeError(f'HOST:PORT required, got {text!r}')
    return host, int(port)


def backend_spec(text):
    """HOST:PORT or HOST:PORT@GPU: the backend and its GPU label (None: its own)."""
    where, sep, gpu = text.partition('@');gpu = gpu.strip()
    if sep and not gpu:raise argparse.ArgumentTypeError(f'HOST:PORT@GPU needs a GPU label, got {text!r}')
    return address(where), gpu if sep else None


def bounded(lo, hi):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or not lo <= value <= hi:raise argparse.ArgumentTypeError(f'a number in [{lo}, {hi}] required, got {text!r}')
        return value
    return parse


class Pool:
    """Backends with their in-flight counts; order() is least busy first, recently failed backends last, ties round-robin.

    A backend whose connection setup fails is DOWN (fail()): out of every ranking until the health check reconnects to it (restore()),
    so a dead backend costs one connect timeout when it dies, never one per decision after (r5a: two dead backends re-ranked first
    after every cooldown added 0.5 s to every new battle's first decision). Only when every backend is down are they all tried. The
    backend set can change while serving (set_backends: a reload)."""

    def __init__(self, backends, *, cooldown=5., sticky_idle=60., sticky_cap=4096, gpus=None, on_event=None):
        self.gpu, self.active, self.served, self.refused, self.failed_at = {}, {}, {}, {}, {}
        self.backends = [];self._in = frozenset();self.down = {};self.on_event = on_event   # down: backend -> when it went down
        self.unserved = 0;self.no_request = 0;self.cooldown = cooldown
        self.homes = OrderedDict();self.sticky_idle = sticky_idle;self.sticky_cap = sticky_cap   # (battle_id, owner) -> [backend, last seen]
        self.placed_at = {};self.ended = 0;self.ended_unknown = 0   # GPU -> its latest new-battle placement; battle end frames
        self._turn = 0
        self.set_backends(backends, gpus)

    @staticmethod
    def check(backends, gpus):
        """The GPU labels of a valid backend set (ValueError otherwise)."""
        if not backends or len(set(backends)) != len(backends):raise ValueError('one or more distinct backends required')
        gpus = {} if gpus is None else dict(gpus)
        if set(gpus) - set(backends):raise ValueError('a GPU label for an unknown backend')
        if gpus and (set(gpus) != set(backends) or len(set(Counter(gpus.values()).values())) != 1):
            # GPU-first ranking spreads battles evenly only over equal GPUs: an unlabelled backend would compete as a whole GPU
            raise ValueError('label every backend with its GPU, the same number of backends on each GPU')
        return gpus

    def set_backends(self, backends, gpus=None):
        """Serve exactly `backends` from now on; ValueError, changing nothing, for an invalid set. A removed backend takes no new
        connection (those in flight finish; its battles re-home on their next decision) and stops counting toward any GPU; an added
        one (or one added back) starts up, with no failure on record. Returns (added, removed)."""
        backends = list(backends);gpus = self.check(backends, gpus)
        added = [b for b in backends if b not in self._in];removed = [b for b in self.backends if b not in set(backends)]
        for b in backends:
            self.gpu[b] = gpus.get(b) or f'{b[0]}:{b[1]}'   # unlabelled: its own group
            for counts in (self.active, self.served, self.refused):counts.setdefault(b, 0)
        for b in added:self.failed_at[b] = None;self.down.pop(b, None)
        for b in removed:self.down.pop(b, None)
        self.backends = backends;self._in = frozenset(backends)
        return added, removed

    def note(self, state, backend, **extra):
        if self.on_event is not None:
            self.on_event(dict(event='HEALTH', state=state, backend=f'{backend[0]}:{backend[1]}', gpu=self.gpu.get(backend), time=time.time(), **extra))

    def fail(self, backend, now=None):
        """A connection setup to `backend` failed: counted, cooling, and down until the health check reconnects."""
        now = time.monotonic() if now is None else now
        self.refused[backend] += 1;self.failed_at[backend] = now
        if backend in self._in and backend not in self.down:self.down[backend] = now;self.note('DOWN', backend)

    def restore(self, backend):
        """The health check reconnected to a down backend: it ranks again, with no failure on record."""
        if self.down.pop(backend, None) is not None:self.failed_at[backend] = None;self.note('UP', backend)

    def candidates(self):
        """The backends a connection may go to: every backend that is up, or all of them when none is."""
        return [b for b in self.backends if b not in self.down] or list(self.backends)

    def live(self, now):
        """Battles per backend seen within sticky_idle seconds (backends still served)."""
        return Counter(b for b, seen in self.homes.values() if now - seen < self.sticky_idle and b in self._in)

    def order(self, now=None, battle=None):
        now = time.monotonic() if now is None else now;pick = self.candidates()
        if battle is None:   # no battle to keep together: least busy, ties round-robin
            start = self._turn % len(pick);self._turn += 1;ring = pick[start:] + pick[:start]
            return sorted(ring, key=lambda b: (self.cooling(b, now), self.active[b]))
        # A new battle's GPU: fewest live battles, then fewest requests in flight there, then the least recently placed on (never
        # backend order: with stale ties that piled three battles on one GPU); within it, the backend with fewest live, then in flight.
        live = self.live(now);per_gpu = self.per_gpu(live);inflight = self.per_gpu(Counter({b: self.active[b] for b in self.backends}))
        rank = sorted(pick, key=lambda b: (self.cooling(b, now), per_gpu[self.gpu[b]], inflight[self.gpu[b]],
                                           self.placed_at.get(self.gpu[b], -math.inf), live[b], self.active[b], self.backends.index(b)))
        home = self.homes.get(battle)
        if home is not None and home[0] in rank and not self.cooling(home[0], now):rank.remove(home[0]);rank.insert(0, home[0])
        return rank

    def bind(self, battle, backend, now=None):
        """The battle's home is now `backend` (LRU, at most sticky_cap); True when it already was."""
        now = time.monotonic() if now is None else now;before = self.homes.pop(battle, None);self.homes[battle] = [backend, now]
        while len(self.homes) > self.sticky_cap:self.homes.popitem(last=False)
        if before is None or before[0] != backend:self.placed_at[self.gpu[backend]] = now   # a placement (new or re-homed), not a return home
        return before is not None and before[0] == backend

    def end(self, battle):
        """The battle's end frame: it no longer counts as live (its home is kept: a stray later request still goes home and counts
        again). Idempotent; False, counted and changing nothing, for a battle the pool does not know."""
        held = self.homes.get(battle)
        if held is None:self.ended_unknown += 1;return False
        held[1] = -math.inf;self.ended += 1;return True

    def cooling(self, backend, now=None):
        now = time.monotonic() if now is None else now
        return self.failed_at[backend] is not None and now - self.failed_at[backend] < self.cooldown

    def snapshot(self):
        live = self.live(time.monotonic())
        return dict(backends=[dict(backend=f'{h}:{p}', active=self.active[(h, p)], served=self.served[(h, p)], refused=self.refused[(h, p)],
            cooling=self.cooling((h, p)), down=(h, p) in self.down, live_battles=live[(h, p)], gpu=self.gpu[(h, p)]) for h, p in self.backends],
            unserved=self.unserved,
            no_request=self.no_request, battle_ends=self.ended, battle_ends_unknown=self.ended_unknown,
            battles=len(self.homes), live_battles_per_gpu=dict(sorted(self.per_gpu(live).items())))

    def per_gpu(self, live):
        """Live battles summed over each GPU's backends."""
        out = Counter()
        for b, n in live.items():out[self.gpu[b]] += n
        return out


async def pipe(reader, writer, counter):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:break
            counter[0] += len(chunk);writer.write(chunk);await writer.drain()
    except (OSError, asyncio.IncompleteReadError):pass
    finally:
        try:writer.write_eof() if writer.can_write_eof() else None
        except (OSError, RuntimeError):pass


async def _close(writer):
    writer.close()
    try:await writer.wait_closed()
    except OSError:pass


async def first_frame(reader, timeout):
    """The connection's first frame as read (its bytes, sent on verbatim, within one `timeout` in all), its battle key
    (battle_id, owner), None if the request is not JSON or carries no string battle_id, and whether it is a battle end frame."""
    deadline = time.monotonic() + timeout
    head = await asyncio.wait_for(reader.readexactly(4), timeout);n = struct.unpack('!I', head)[0]
    if not 0 < n <= MAX_PACKET:return head, None, False   # an unbounded length: forwarded as is, for the server to refuse
    body = await asyncio.wait_for(reader.readexactly(n), max(0., deadline - time.monotonic()))
    try:value = json.loads(body)
    except Exception:return head + body, None, False   # not JSON, or too deep to parse: least busy
    battle = value.get('battle_id') if isinstance(value, dict) else None;owner = value.get('owner') if isinstance(value, dict) else None
    key = (battle, owner if isinstance(owner, str) else None) if isinstance(battle, str) and battle else None
    return head + body, key, isinstance(value, dict) and value.get('schema') == BATTLE_END_SCHEMA


async def handle(pool, client_reader, client_writer, *, connect_timeout=.25, first_frame_timeout=2., log_row=None):
    opened, chosen, upstream = time.time(), None, None
    try:first, battle, end = await first_frame(client_reader, first_frame_timeout)
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):   # no request to route: nothing forwarded
        pool.no_request += 1
        if log_row is not None:log_row(dict(backend=None, reason='no request', opened=opened, closed=time.time(), seconds=time.time() - opened))
        await _close(client_writer);return
    if end:   # a battle end frame: answered here, never forwarded
        known = pool.end(battle) if battle is not None else False
        if battle is None:pool.ended_unknown += 1   # no string battle_id: unknown too
        out = json.dumps(dict(schema=BATTLE_END_SCHEMA, battle_id=None if battle is None else battle[0], owner=None if battle is None else battle[1],
                              status='ENDED' if known else 'UNKNOWN_BATTLE'), sort_keys=True).encode()
        try:client_writer.write(struct.pack('!I', len(out)) + out);await client_writer.drain()
        except OSError:pass
        if log_row is not None:log_row(dict(backend=None, reason='battle end' if known else 'battle end: unknown battle', opened=opened, closed=time.time(),
            seconds=time.time() - opened, battle_id=None if battle is None else battle[0], owner=None if battle is None else battle[1]))
        await _close(client_writer);return
    now = time.monotonic()
    for backend in pool.order(now, battle=battle):
        pool.active[backend] += 1   # reserved before connecting, so simultaneous clients spread out
        try:
            upstream = await asyncio.wait_for(asyncio.open_connection(*backend), connect_timeout);chosen = backend;break
        except (OSError, asyncio.TimeoutError):
            pool.active[backend] -= 1;pool.fail(backend)
        except BaseException:
            pool.active[backend] -= 1;raise
    if upstream is None:
        pool.unserved += 1
        if log_row is not None:log_row(dict(backend='unserved', battle_id=None if battle is None else battle[0], owner=None if battle is None else battle[1],
            opened=opened, closed=time.time(), seconds=time.time() - opened))
        await _close(client_writer);return
    up_reader, up_writer = upstream;pool.served[chosen] += 1;sent, received = [len(first)], [0]
    home = None if battle is None else pool.bind(battle, chosen, now)
    async def forward():
        try:up_writer.write(first);await up_writer.drain()
        except OSError:return
        await pipe(client_reader, up_writer, sent)
    try:await asyncio.gather(forward(), pipe(up_reader, client_writer, received))
    finally:
        pool.active[chosen] -= 1
        for w in (up_writer, client_writer):await _close(w)
        if log_row is not None:
            closed = time.time();log_row(dict(backend=f'{chosen[0]}:{chosen[1]}', opened=opened, closed=closed, seconds=closed - opened,
                bytes_to_backend=sent[0], bytes_to_client=received[0], battle_id=None if battle is None else battle[0],
                owner=None if battle is None else battle[1], home=home, gpu=pool.gpu[chosen]))


async def wait_ready(backends, seconds):
    deadline = time.monotonic() + seconds;pending = set(backends)
    while pending and time.monotonic() < deadline:
        for b in list(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:break
            try:
                _, w = await asyncio.wait_for(asyncio.open_connection(*b), min(1., remaining));pending.discard(b);await _close(w)
            except (OSError, asyncio.TimeoutError):pass
        if pending:await asyncio.sleep(min(.5, max(0., deadline - time.monotonic())))
    return sorted(pending)


async def ask(backend, request, *, timeout):
    """One framed request straight to one backend; its decoded answer."""
    reader, writer = await asyncio.wait_for(asyncio.open_connection(*backend), timeout)
    try:
        raw = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
        writer.write(struct.pack('!I', len(raw)) + raw);await asyncio.wait_for(writer.drain(), timeout)
        n = struct.unpack('!I', await asyncio.wait_for(reader.readexactly(4), timeout))[0]
        if not 0 < n <= MAX_PACKET:raise ValueError('bounded response required')
        return json.loads(await asyncio.wait_for(reader.readexactly(n), timeout))
    finally:await _close(writer)


def identity(answer):
    pricer = answer.get('pricer') if isinstance(answer.get('pricer'), dict) else {}
    reply = answer.get('reply_prior') if isinstance(answer.get('reply_prior'), dict) else {}
    return dict(rung=answer.get('rung'), archive_sha256=answer.get('archive_sha256'), requested_worlds=answer.get('requested_worlds'),
        root_operator=answer.get('root_operator'), matmul_precision=pricer.get('matmul_precision'), tree_commit=answer.get('tree_commit'),
        pricer=answer.get('pricer'), reply_archive_sha256=reply.get('reply_archive_sha256') or 'none', reply_views=reply.get('views') or 'none',
        peer_view_b='on' if 'peer_view_b' in answer else 'off')


NEXT_ROOT = ('constructor_refused', 'depthn_refused', 'class_refused')


async def check_identity(backends, requests, expect, *, timeout=120.):
    """Every backend directly, before listening (module docstring). Returns {backend: identity}; raises SystemExit naming what differs."""
    if not requests:raise SystemExit('one or more probe requests required')
    found = {}
    for host, port in backends:
        tried = []
        for request in requests:
            body = dict(request, request_id=uuid.uuid4().hex, battle_id=f'proxy-probe-{port}-{uuid.uuid4().hex}')
            try:answer = await ask((host, port), body, timeout=timeout)
            except (OSError, asyncio.TimeoutError, ValueError, struct.error, asyncio.IncompleteReadError) as error:
                raise SystemExit(f'backend {host}:{port}: identity probe failed: {type(error).__name__}: {error}')
            if type(answer) is not dict or answer.get('request_id') != body['request_id']:raise SystemExit(f'backend {host}:{port}: answer is not for the probe')
            tried.append((answer.get('status'), answer.get('cause')))
            if answer.get('status') == 'COMPLETE' or answer.get('cause') not in NEXT_ROOT:break
        if answer.get('status') != 'COMPLETE':
            raise SystemExit(f'backend {host}:{port}: no COMPLETE identity probe (tried {tried}; last refusal {answer.get("refusal")!r})')
        mine = identity(answer);wrong = {k: (mine[k], v) for k, v in expect.items() if mine[k] != v}
        if wrong:raise SystemExit(f'backend {host}:{port}: identity differs from the expected (found, expected): {wrong}')
        found[(host, port)] = mine
    if len({json.dumps(v, sort_keys=True) for v in found.values()}) != 1:
        raise SystemExit('backends disagree on their identity: ' + json.dumps({f'{h}:{p}': v for (h, p), v in found.items()}, sort_keys=True))
    return found


def read_backends_file(path):
    """--backends-file: one HOST:PORT or HOST:PORT@GPU per line ('#' comments, blank lines skipped) -> [(backend, gpu or None)]."""
    with open(path, encoding='utf-8') as f:
        return [backend_spec(line) for line in (raw.split('#', 1)[0].strip() for raw in f) if line]


async def run(listen, backends, *, wait_ready_seconds=0., report_seconds=60., log=print, started=None, connect_timeout=.25, cooldown=5.,
              first_frame_timeout=2., sticky_idle=60., sticky_cap=4096, gpus=None, probe_requests=None, expect=None, probe_timeout=120., log_row=None,
              health_interval=1., backends_file=None):
    if listen in backends:raise SystemExit('--listen must not be one of the backends')
    if type(sticky_cap) is not int or sticky_cap < 1:raise SystemExit('--sticky-cap must be a positive integer')
    if not (math.isfinite(health_interval) and health_interval > 0):raise SystemExit('--health-interval must be a positive number of seconds')
    def event(row):log(json.dumps(row))
    pool = Pool(backends, cooldown=cooldown, sticky_idle=sticky_idle, sticky_cap=sticky_cap, gpus=gpus, on_event=event)
    if wait_ready_seconds > 0:
        missing = await wait_ready(backends, wait_ready_seconds)
        if missing:raise SystemExit(f'backends not ready after {wait_ready_seconds} s: ' + ', '.join(f'{h}:{p}' for h, p in missing))
    identities = None if probe_requests is None else await check_identity(backends, probe_requests, expect or {}, timeout=probe_timeout)
    identity = None if identities is None else next(iter(identities.values()))
    quarantined = set()   # backends whose identity probe failed: never health-checked again, only a reload may add them back

    async def reachable(b):
        try:_, w = await asyncio.wait_for(asyncio.open_connection(*b), connect_timeout)
        except (OSError, asyncio.TimeoutError):return False
        await _close(w);return True

    async def admitted(b):
        """With --probe-requests: `b` answers the identity probe with the identity the pool started with (a backend coming back or
        joining is a backend the pool would otherwise hide); else quarantined. Without: True."""
        if probe_requests is None:return True
        try:found = (await check_identity([b], probe_requests, expect or {}, timeout=probe_timeout))[b]
        except SystemExit as error:found, why = None, str(error)
        else:why = None if json.dumps(found, sort_keys=True) == json.dumps(identity, sort_keys=True) else 'identity differs from the pool\'s'
        if why is None:return True
        quarantined.add(b);pool.note('QUARANTINED', b, reason=why[:500]);return False

    async def revive(b):
        if await reachable(b) and await admitted(b) and b in pool.down:pool.restore(b)

    async def health():
        """Every health_interval seconds, each down backend (not quarantined) is reconnected to; it ranks again only once it accepts
        a connection (and passes the identity probe, with --probe-requests). Requests never retry a down backend."""
        while True:
            await asyncio.sleep(health_interval)
            down = [b for b in pool.down if b in pool._in and b not in quarantined]
            if down:await asyncio.gather(*(revive(b) for b in down))

    lock = asyncio.Lock()
    async def reload():
        """Re-read --backends-file and serve exactly its backends (SIGHUP). All or nothing: an unreadable or invalid file, the listen
        address among the backends, or an added backend that does not accept a connection (or fails the identity probe) leaves the
        set unchanged (RELOAD_REJECTED). Removed backends take no new connection; added ones join up."""
        async with lock:
            try:
                if backends_file is None:raise ValueError('no --backends-file')
                specs = read_backends_file(backends_file);new = [b for b, _ in specs];new_gpus = {b: g for b, g in specs if g is not None}
                bound = server.sockets[0].getsockname()[1]   # the port listened on (--listen may name port 0)
                if listen in new or any(p == bound and h in {listen[0], '0.0.0.0', '127.0.0.1', 'localhost', '::', '::1'} for h, p in new):
                    raise ValueError('--listen must not be one of the backends')
                Pool.check(new, new_gpus)
                # a backend joining, or a quarantined one the file still names (the operator re-admits it), must accept a connection
                # and pass the identity probe first
                joining = [b for b in new if b not in pool._in or b in quarantined]
                for b in joining:
                    if not await reachable(b):raise ValueError(f'{b[0]}:{b[1]} does not accept a connection')
                    quarantined.discard(b)
                    if not await admitted(b):raise ValueError(f'{b[0]}:{b[1]} failed the identity probe')
                readmitted = [b for b in joining if b in pool._in];added, removed = pool.set_backends(new, new_gpus)
                for b in readmitted:pool.restore(b)
            except (OSError, ValueError, argparse.ArgumentTypeError) as error:
                log(json.dumps(dict(event='RELOAD_REJECTED', time=time.time(), reason=str(error)[:500])));return False
            quarantined.intersection_update(pool._in)
            log(json.dumps(dict(event='RELOAD', time=time.time(), added=[f'{h}:{p}' for h, p in added], removed=[f'{h}:{p}' for h, p in removed],
                                backends=[f'{h}:{p}' for h, p in pool.backends])))
            return True
    pool.reload = reload   # tests (and anything holding the pool) reload without a signal

    server = await asyncio.start_server(lambda r, w: handle(pool, r, w, connect_timeout=connect_timeout, first_frame_timeout=first_frame_timeout, log_row=log_row), *listen)
    log(json.dumps(dict(event='READY', listen=f'{listen[0]}:{server.sockets[0].getsockname()[1]}', backends=[f'{h}:{p}' for h, p in backends],
        identity=identity, probed=identities is not None)))
    if started is not None:started(server, pool)
    async def report():
        while True:
            await asyncio.sleep(report_seconds);log(json.dumps(dict(event='POOL', time=time.time(), **pool.snapshot())))
    tasks = [asyncio.create_task(health())] + ([asyncio.create_task(report())] if report_seconds > 0 else [])
    loop = asyncio.get_running_loop();pending = set();hup = False
    def on_hup():
        task = asyncio.ensure_future(reload());pending.add(task);task.add_done_callback(pending.discard)
    try:loop.add_signal_handler(signal.SIGHUP, on_hup);hup = True   # a stray SIGHUP no longer kills the proxy: it reloads (or logs why not)
    except (NotImplementedError, RuntimeError, ValueError):pass   # no signals here (not the main thread, or not Unix)
    try:
        async with server:await server.serve_forever()
    finally:
        if hup:loop.remove_signal_handler(signal.SIGHUP)
        for task in tasks + list(pending):task.cancel()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--listen', type=address, required=True, help='HOST:PORT to listen on')
    p.add_argument('--backend', type=backend_spec, action='append',
        help='a depthn_live_server HOST:PORT, or HOST:PORT@GPU naming its GPU (repeat per process)')
    p.add_argument('--backends-file', help='the backends instead, one HOST:PORT[@GPU] per line; re-read on SIGHUP (add or drop backends '
        'without a restart)')
    p.add_argument('--health-interval', type=bounded(.01, 600.), default=1.,
        help='seconds between reconnection attempts to each down backend (a backend whose connection setup failed)')
    p.add_argument('--wait-ready', type=bounded(0., 3600.), default=0., help='seconds to wait for every backend to accept a connection')
    p.add_argument('--connect-timeout', type=bounded(.01, 5.), default=.25, help='per-backend connection setup timeout (seconds)')
    p.add_argument('--cooldown', type=bounded(0., 600.), default=5., help='seconds a backend whose setup failed sorts last')
    p.add_argument('--first-frame-timeout', type=bounded(.01, 60.), default=2., help='seconds to read a connection\'s request (its battle_id)')
    p.add_argument('--sticky-idle', type=bounded(1., 86400.), default=60., help='a battle without a decision this long no longer counts as live')
    p.add_argument('--sticky-cap', type=int, default=4096, help='battle homes kept (LRU)')
    p.add_argument('--report-seconds', type=bounded(0., 86400.), default=60.)
    p.add_argument('--probe-requests', help='JSONL: requests in the servers\' wire format, tried on every backend before listening')
    p.add_argument('--probe-timeout', type=bounded(1., 3600.), default=120.)
    for name in ('rung', 'archive-sha256', 'root-operator', 'matmul-precision'):p.add_argument('--expect-' + name)
    p.add_argument('--expect-reply-archive-sha256', help="the served reply model's archive sha256 (br_fp), or 'none' for a server without one")
    p.add_argument('--expect-peer-view-b', choices=('on', 'off'), help='whether the COMPLETE probe answer was built with peer view b')
    p.add_argument('--expect-reply-views', choices=('eager', 'jit', 'check', 'none'), help="the reply model's views mode (br_fp), or 'none'")
    p.add_argument('--expect-worlds', type=int)
    p.add_argument('--connection-log', help='JSONL, appended: one row per client connection (backend, timings, bytes)')
    args = p.parse_args(argv)
    if (args.backend is None) == (args.backends_file is None):p.error('one of --backend (repeated) or --backends-file required')
    if args.backends_file is not None:
        try:args.backend = read_backends_file(args.backends_file)
        except (OSError, argparse.ArgumentTypeError) as error:p.error(f'--backends-file: {error}')
        if not args.backend:p.error('--backends-file: one or more backends required')
    expect = {k: v for k, v in dict(rung=args.expect_rung, archive_sha256=args.expect_archive_sha256, root_operator=args.expect_root_operator,
        matmul_precision=args.expect_matmul_precision, requested_worlds=args.expect_worlds, reply_archive_sha256=args.expect_reply_archive_sha256,
        peer_view_b=args.expect_peer_view_b, reply_views=args.expect_reply_views).items() if v is not None}
    if args.probe_requests is None and expect:p.error('--expect-* needs --probe-requests')
    probe = None
    if args.probe_requests is not None:
        with open(args.probe_requests, encoding='utf-8') as f:probe = [json.loads(line) for line in f if line.strip()]
        if not probe or any(type(q) is not dict for q in probe):p.error('--probe-requests: one or more request objects (JSONL) required')
    stream = open(args.connection_log, 'a', encoding='utf-8') if args.connection_log else None
    def log_row(row):stream.write(json.dumps(row, sort_keys=True) + '\n');stream.flush()
    def log(line):print(line, flush=True)
    backends = [b for b, _ in args.backend];gpus = {b: g for b, g in args.backend if g is not None}
    try:asyncio.run(run(args.listen, backends, gpus=gpus, wait_ready_seconds=args.wait_ready, report_seconds=args.report_seconds, log=log,
        connect_timeout=args.connect_timeout, cooldown=args.cooldown, first_frame_timeout=args.first_frame_timeout, sticky_idle=args.sticky_idle,
        sticky_cap=args.sticky_cap, probe_requests=probe, expect=expect, probe_timeout=args.probe_timeout,
        log_row=None if stream is None else log_row, health_interval=args.health_interval, backends_file=args.backends_file))
    except KeyboardInterrupt:return 0
    finally:
        if stream is not None:stream.close()
    return 0


if __name__ == '__main__':sys.exit(main())
