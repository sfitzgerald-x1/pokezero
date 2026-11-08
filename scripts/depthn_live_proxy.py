#!/usr/bin/env python3
"""One listening address in front of several depthn_live_server processes: each incoming connection goes to the least-busy backend.

Design. The panel's dispatch talks to ONE search server address (--server HOST:PORT), and the actor's client opens one connection per
request (phase_d_search_server framing: one request, one response). A server process answers one decision at a time (max_requests 1;
the constructor's prepare is serial per root), so several processes per GPU serve the live eval. This proxy forwards each accepted
connection, byte for byte in both directions, to the backend with the fewest connections in flight (ties: the next in round-robin
order; a backend whose connection setup failed within the last --cooldown seconds sorts last), and closes both sides when either
closes. A backend's slot is reserved before its connection is opened, so connections accepted in the same instant spread out. It never
rewrites, retries or replays a request: connection setup alone is retried on the next backend (--connect-timeout each, before any byte
is forwarded); when no backend accepts, the client's connection is closed, which its client records as a transport failure
(connection_error, or server_error on a bare EOF) and plays greedy, as with a dead server.
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


def bounded(lo, hi):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or not lo <= value <= hi:raise argparse.ArgumentTypeError(f'a number in [{lo}, {hi}] required, got {text!r}')
        return value
    return parse


class Pool:
    """Backends with their in-flight counts; order() is least busy first, recently failed backends last, ties round-robin."""

    def __init__(self, backends, *, cooldown=5.):
        if not backends or len(set(backends)) != len(backends):raise ValueError('one or more distinct backends required')
        self.backends = list(backends);self.active = {b: 0 for b in backends};self.served = {b: 0 for b in backends}
        self.refused = {b: 0 for b in backends};self.failed_at = {b: None for b in backends};self.unserved = 0;self.cooldown = cooldown
        self._turn = itertools.cycle(range(len(backends)))

    def order(self, now=None):
        now = time.monotonic() if now is None else now;start = next(self._turn);ring = self.backends[start:] + self.backends[:start]
        return sorted(ring, key=lambda b: (self.cooling(b, now), self.active[b]))

    def cooling(self, backend, now=None):
        now = time.monotonic() if now is None else now
        return self.failed_at[backend] is not None and now - self.failed_at[backend] < self.cooldown

    def snapshot(self):
        return dict(backends=[dict(backend=f'{h}:{p}', active=self.active[(h, p)], served=self.served[(h, p)], refused=self.refused[(h, p)],
            cooling=self.cooling((h, p))) for h, p in self.backends], unserved=self.unserved)


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


async def handle(pool, client_reader, client_writer, *, connect_timeout=.25, log_row=None):
    opened, chosen, upstream = time.time(), None, None
    for backend in pool.order():
        pool.active[backend] += 1   # reserved before connecting, so simultaneous clients spread out
        try:
            upstream = await asyncio.wait_for(asyncio.open_connection(*backend), connect_timeout);chosen = backend;break
        except (OSError, asyncio.TimeoutError):
            pool.active[backend] -= 1;pool.refused[backend] += 1;pool.failed_at[backend] = time.monotonic()
        except BaseException:
            pool.active[backend] -= 1;raise
    if upstream is None:
        pool.unserved += 1
        if log_row is not None:log_row(dict(backend='unserved', opened=opened, closed=time.time(), seconds=time.time() - opened))
        await _close(client_writer);return
    up_reader, up_writer = upstream;pool.served[chosen] += 1;sent, received = [0], [0]
    try:await asyncio.gather(pipe(client_reader, up_writer, sent), pipe(up_reader, client_writer, received))
    finally:
        pool.active[chosen] -= 1
        for w in (up_writer, client_writer):await _close(w)
        if log_row is not None:
            closed = time.time();log_row(dict(backend=f'{chosen[0]}:{chosen[1]}', opened=opened, closed=closed, seconds=closed - opened,
                bytes_to_backend=sent[0], bytes_to_client=received[0]))


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
              probe_requests=None, expect=None, probe_timeout=120., log_row=None):
    if listen in backends:raise SystemExit('--listen must not be one of the backends')
    pool = Pool(backends, cooldown=cooldown)
    if wait_ready_seconds > 0:
        missing = await wait_ready(backends, wait_ready_seconds)
        if missing:raise SystemExit(f'backends not ready after {wait_ready_seconds} s: ' + ', '.join(f'{h}:{p}' for h, p in missing))
    identities = None if probe_requests is None else await check_identity(backends, probe_requests, expect or {}, timeout=probe_timeout)
    server = await asyncio.start_server(lambda r, w: handle(pool, r, w, connect_timeout=connect_timeout, log_row=log_row), *listen)
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
    p.add_argument('--backend', type=address, action='append', required=True, help='a depthn_live_server HOST:PORT (repeat per process)')
    p.add_argument('--wait-ready', type=bounded(0., 3600.), default=0., help='seconds to wait for every backend to accept a connection')
    p.add_argument('--connect-timeout', type=bounded(.01, 5.), default=.25, help='per-backend connection setup timeout (seconds)')
    p.add_argument('--cooldown', type=bounded(0., 600.), default=5., help='seconds a backend whose setup failed sorts last')
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
    try:asyncio.run(run(args.listen, args.backend, wait_ready_seconds=args.wait_ready, report_seconds=args.report_seconds, log=log,
        connect_timeout=args.connect_timeout, cooldown=args.cooldown, probe_requests=probe, expect=expect, probe_timeout=args.probe_timeout,
        log_row=None if stream is None else log_row))
    except KeyboardInterrupt:return 0
    finally:
        if stream is not None:stream.close()
    return 0


if __name__ == '__main__':sys.exit(main())
