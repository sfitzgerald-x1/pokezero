#!/usr/bin/env python3
"""One listening address in front of several depthn_live_server processes: each battle sticks to one backend; other traffic goes least busy.

Design. The panel's dispatch talks to ONE search server address (--server HOST:PORT), and the actor's client opens one connection per
request (phase_d_search_server framing: one request, one response). A server process answers one decision at a time (max_requests 1;
the constructor's prepare is serial per root), so several processes per GPU serve the live eval. This proxy forwards each accepted
connection, byte for byte in both directions, to the backend with the fewest connections in flight (ties: the next in round-robin
order; a backend whose connection setup failed within the last --cooldown seconds sorts last), and closes both sides when either
closes. A backend's slot is reserved before its connection is opened, so connections accepted in the same instant spread out. It never
rewrites, retries or replays a request: connection setup alone is retried on the next backend (--connect-timeout each, before any byte
is forwarded); when no backend accepts, the client's connection is closed, which its client records as a transport failure
(connection_error, or server_error on a bare EOF) and plays greedy, as with a dead server.
- Battle affinity (sticky): a server keeps one constructor session per battle, so a battle whose decisions scatter over the pool
  folds from a stale or empty session on every switch (pre-flight r3: battles touched 8-12 of 12 servers, prepare p95 2.04 s against
  1.20 s on the same server, 28% timeouts). The proxy therefore reads each connection's first frame (the request: a 4-byte length
  and its JSON, within --first-frame-timeout) for its battle_id, sends those bytes on verbatim, and routes: a battle seen before goes
  to its home backend unless that backend is cooling down; a new battle (or a re-homed one) goes, the lead's rule, to the GPU with
  the fewest live battles, then to that GPU's backend with the fewest (--backend HOST:PORT@GPU names each backend's GPU; live: a
  decision within --sticky-idle seconds, default 60, so finished battles stop counting), ties by in-flight connections, then by
  --backend order. Concurrent pricing on one GPU costs real time (r3: price p95 1.07 s against the bench's 0.67 s), so battles spread
  across GPUs first. The home map is an LRU of at most --sticky-cap battles. A request without a string battle_id keeps the
  least-busy order below. Each connection's log row records its battle_id, backend, GPU and whether it went home (home false with a
  battle_id: an assignment).
- Startup identity (the lead's rule: a pool must not hide a wrong backend): --probe-requests PATH (JSONL, one request per line in the
  servers' wire format, e.g. built from a closed RAW cell's owning-wire capture as the panel's probe builds them) is sent to EVERY backend
  directly, each request under a fresh request_id and battle id, before the proxy listens. Per backend, requests are tried in order: a
  constructor, DepthN or root-class refusal moves to the next (as the panel's probe does); anything else that is not COMPLETE, or no
  COMPLETE at all, fails. The COMPLETE answer must carry the rung, archive_sha256, requested_worlds, root_operator and
  pricer.matmul_precision given by --expect-rung / --expect-archive-sha256 / --expect-worlds / --expect-root-operator /
  --expect-matmul-precision, and all backends must agree on those, on tree_commit and on the whole pricer identity (evaluator, W, K,
  buckets, precision); otherwise the proxy exits non-zero. (The rung is the request's own, echoed; a backend serving another rung
  refuses the probe as a bad request, which fails it. requested_worlds is the W check.)
- --wait-ready SECONDS (0 to 3600): before the identity probe, wait until every backend accepts a connection, else exit non-zero.
- --connection-log PATH (JSONL, appended, flushed): one row per client connection (backend, opened, closed, seconds, bytes each way,
  or 'unserved'), so latency breaks down by backend (the servers' own server_id in each response is the panel readout's breakdown). READY (one JSON line, with each backend's probe
  answer) and periodic POOL lines go to stdout; send stdout to a file or the container log.
"""
import argparse
import asyncio
from collections import Counter, OrderedDict
import itertools
import json
import math
import struct
import sys
import time
import uuid

MAX_PACKET = 16 * 1024 * 1024


def address(text):
    host, sep, port = text.rpartition(':')
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:raise argparse.ArgumentTypeError(f'HOST:PORT required, got {text!r}')
    return host, int(port)


def backend_spec(text):
    """HOST:PORT or HOST:PORT@GPU: the backend and its GPU label (None: its own)."""
    where, sep, gpu = text.partition('@')
    if sep and not gpu.strip():raise argparse.ArgumentTypeError(f'HOST:PORT@GPU needs a GPU label, got {text!r}')
    return address(where), gpu if sep else None


def bounded(lo, hi):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or not lo <= value <= hi:raise argparse.ArgumentTypeError(f'a number in [{lo}, {hi}] required, got {text!r}')
        return value
    return parse


class Pool:
    """Backends with their in-flight counts; order() is least busy first, recently failed backends last, ties round-robin."""

    def __init__(self, backends, *, cooldown=5., sticky_idle=60., sticky_cap=4096, gpus=None):
        if not backends or len(set(backends)) != len(backends):raise ValueError('one or more distinct backends required')
        gpus = {} if gpus is None else dict(gpus)
        if set(gpus) - set(backends):raise ValueError('a GPU label for an unknown backend')
        self.gpu = {b: gpus.get(b) or f'{b[0]}:{b[1]}' for b in backends}   # unlabelled: its own group
        self.backends = list(backends);self.active = {b: 0 for b in backends};self.served = {b: 0 for b in backends}
        self.refused = {b: 0 for b in backends};self.failed_at = {b: None for b in backends};self.unserved = 0;self.cooldown = cooldown
        self.homes = OrderedDict();self.sticky_idle = sticky_idle;self.sticky_cap = sticky_cap   # battle_id -> [backend, last seen]
        self._turn = itertools.cycle(range(len(backends)))

    def live(self, now):
        """Battles per backend seen within sticky_idle seconds."""
        return Counter(b for b, seen in self.homes.values() if now - seen < self.sticky_idle)

    def order(self, now=None, battle=None):
        now = time.monotonic() if now is None else now
        if battle is None:   # no battle to keep together: least busy, ties round-robin
            start = next(self._turn);ring = self.backends[start:] + self.backends[:start]
            return sorted(ring, key=lambda b: (self.cooling(b, now), self.active[b]))
        live = self.live(now);per_gpu = Counter()
        for b, n in live.items():per_gpu[self.gpu[b]] += n
        rank = sorted(self.backends, key=lambda b: (self.cooling(b, now), per_gpu[self.gpu[b]], live[b], self.active[b], self.backends.index(b)))
        home = self.homes.get(battle)
        if home is not None and not self.cooling(home[0], now):rank.remove(home[0]);rank.insert(0, home[0])
        return rank

    def bind(self, battle, backend, now=None):
        """The battle's home is now `backend` (LRU, at most sticky_cap); True when it already was."""
        now = time.monotonic() if now is None else now;before = self.homes.pop(battle, None);self.homes[battle] = [backend, now]
        while len(self.homes) > self.sticky_cap:self.homes.popitem(last=False)
        return before is not None and before[0] == backend

    def cooling(self, backend, now=None):
        now = time.monotonic() if now is None else now
        return self.failed_at[backend] is not None and now - self.failed_at[backend] < self.cooldown

    def snapshot(self):
        live = self.live(time.monotonic())
        return dict(backends=[dict(backend=f'{h}:{p}', active=self.active[(h, p)], served=self.served[(h, p)], refused=self.refused[(h, p)],
            cooling=self.cooling((h, p)), live_battles=live[(h, p)], gpu=self.gpu[(h, p)]) for h, p in self.backends], unserved=self.unserved,
            battles=len(self.homes), live_battles_per_gpu=dict(sorted(Counter({self.gpu[b]: n for b, n in live.items()}).items())))


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
    """The connection's first frame as read (its bytes, sent on verbatim) and its battle_id (None if absent or not JSON)."""
    head = await asyncio.wait_for(reader.readexactly(4), timeout);n = struct.unpack('!I', head)[0]
    if not 0 < n <= MAX_PACKET:return head, None   # an unbounded length: forwarded as is, for the server to refuse
    body = await asyncio.wait_for(reader.readexactly(n), timeout)
    try:value = json.loads(body)
    except (ValueError, UnicodeDecodeError):return head + body, None
    battle = value.get('battle_id') if isinstance(value, dict) else None
    return head + body, battle if isinstance(battle, str) and battle else None


async def handle(pool, client_reader, client_writer, *, connect_timeout=.25, first_frame_timeout=2., log_row=None):
    opened, chosen, upstream = time.time(), None, None
    try:first, battle = await first_frame(client_reader, first_frame_timeout)
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):   # no request to route: nothing forwarded
        pool.unserved += 1
        if log_row is not None:log_row(dict(backend='unserved', reason='no request', opened=opened, closed=time.time(), seconds=time.time() - opened))
        await _close(client_writer);return
    now = time.monotonic()
    for backend in pool.order(now, battle=battle):
        pool.active[backend] += 1   # reserved before connecting, so simultaneous clients spread out
        try:
            upstream = await asyncio.wait_for(asyncio.open_connection(*backend), connect_timeout);chosen = backend;break
        except (OSError, asyncio.TimeoutError):
            pool.active[backend] -= 1;pool.refused[backend] += 1;pool.failed_at[backend] = time.monotonic()
        except BaseException:
            pool.active[backend] -= 1;raise
    if upstream is None:
        pool.unserved += 1
        if log_row is not None:log_row(dict(backend='unserved', battle_id=battle, opened=opened, closed=time.time(), seconds=time.time() - opened))
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
                bytes_to_backend=sent[0], bytes_to_client=received[0], battle_id=battle, home=home, gpu=pool.gpu[chosen]))


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
    return dict(rung=answer.get('rung'), archive_sha256=answer.get('archive_sha256'), requested_worlds=answer.get('requested_worlds'),
        root_operator=answer.get('root_operator'), matmul_precision=pricer.get('matmul_precision'), tree_commit=answer.get('tree_commit'),
        pricer=answer.get('pricer'))


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


async def run(listen, backends, *, wait_ready_seconds=0., report_seconds=60., log=print, started=None, connect_timeout=.25, cooldown=5.,
              first_frame_timeout=2., sticky_idle=60., sticky_cap=4096, gpus=None, probe_requests=None, expect=None, probe_timeout=120., log_row=None):
    if listen in backends:raise SystemExit('--listen must not be one of the backends')
    if type(sticky_cap) is not int or sticky_cap < 1:raise SystemExit('--sticky-cap must be a positive integer')
    pool = Pool(backends, cooldown=cooldown, sticky_idle=sticky_idle, sticky_cap=sticky_cap, gpus=gpus)
    if wait_ready_seconds > 0:
        missing = await wait_ready(backends, wait_ready_seconds)
        if missing:raise SystemExit(f'backends not ready after {wait_ready_seconds} s: ' + ', '.join(f'{h}:{p}' for h, p in missing))
    identities = None if probe_requests is None else await check_identity(backends, probe_requests, expect or {}, timeout=probe_timeout)
    server = await asyncio.start_server(lambda r, w: handle(pool, r, w, connect_timeout=connect_timeout, first_frame_timeout=first_frame_timeout, log_row=log_row), *listen)
    log(json.dumps(dict(event='READY', listen=f'{listen[0]}:{server.sockets[0].getsockname()[1]}', backends=[f'{h}:{p}' for h, p in backends],
        identity=None if identities is None else next(iter(identities.values())), probed=identities is not None)))
    if started is not None:started(server, pool)
    async def report():
        while True:
            await asyncio.sleep(report_seconds);log(json.dumps(dict(event='POOL', time=time.time(), **pool.snapshot())))
    reporter = asyncio.create_task(report()) if report_seconds > 0 else None
    try:
        async with server:await server.serve_forever()
    finally:
        if reporter is not None:reporter.cancel()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--listen', type=address, required=True, help='HOST:PORT to listen on')
    p.add_argument('--backend', type=backend_spec, action='append', required=True,
        help='a depthn_live_server HOST:PORT, or HOST:PORT@GPU naming its GPU (repeat per process)')
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
    p.add_argument('--expect-worlds', type=int)
    p.add_argument('--connection-log', help='JSONL, appended: one row per client connection (backend, timings, bytes)')
    args = p.parse_args(argv)
    expect = {k: v for k, v in dict(rung=args.expect_rung, archive_sha256=args.expect_archive_sha256, root_operator=args.expect_root_operator,
        matmul_precision=args.expect_matmul_precision, requested_worlds=args.expect_worlds).items() if v is not None}
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
        log_row=None if stream is None else log_row))
    except KeyboardInterrupt:return 0
    finally:
        if stream is not None:stream.close()
    return 0


if __name__ == '__main__':sys.exit(main())
