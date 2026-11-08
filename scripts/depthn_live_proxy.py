#!/usr/bin/env python3
"""One listening address in front of several depthn_live_server processes: each incoming connection goes to the least-busy backend.

Design. The panel's dispatch talks to ONE search server address (--server HOST:PORT), and the actor's client opens one connection per
request (phase_d_search_server framing: one request, one response). A server process answers one decision at a time (max_requests 1;
the constructor's prepare is serial per root), so several processes per GPU serve the live eval. This proxy forwards each accepted
connection, byte for byte in both directions, to the backend with the fewest connections in flight (ties: the next in round-robin
order), and closes both sides when either closes. It never parses, rewrites, retries or replays a request, so the transport, its
framing and every timeout stay the client's and the servers': a request whose backend refuses the connection is offered to the next
least-busy backend (connection setup only, before any byte is forwarded); when none accepts, the client's connection is closed and
its client records a connection error and plays greedy, as with a dead server.
- --wait-ready SECONDS: before listening, wait until every backend accepts a probe connection (closed at once; the server's own READY
  probe is the same), else exit non-zero. The proxy reports READY (one JSON line: listen address, backends) only once listening.
- Telemetry (stdout, one JSON line per --report-seconds): connections per backend, in flight, refused setups, and client connections
  closed for want of a backend.
"""
import argparse
import asyncio
import itertools
import json
import sys
import time


def address(text):
    host, sep, port = text.rpartition(':')
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:raise argparse.ArgumentTypeError(f'HOST:PORT required, got {text!r}')
    return host, int(port)


class Pool:
    """Backends with their in-flight counts; choose() is the least busy, ties broken round-robin."""

    def __init__(self, backends):
        if not backends or len(set(backends)) != len(backends):raise ValueError('one or more distinct backends required')
        self.backends = list(backends);self.active = {b: 0 for b in backends};self.served = {b: 0 for b in backends}
        self.refused = {b: 0 for b in backends};self.unserved = 0;self._turn = itertools.cycle(range(len(backends)))

    def order(self):
        """The backends in choice order: fewest in flight first, ties in round-robin order from the next turn."""
        start = next(self._turn);ring = self.backends[start:] + self.backends[:start]
        return sorted(ring, key=lambda b: self.active[b])

    def snapshot(self):
        return dict(backends=[dict(backend=f'{h}:{p}', active=self.active[(h, p)], served=self.served[(h, p)], refused=self.refused[(h, p)])
            for h, p in self.backends], unserved=self.unserved)


async def pipe(reader, writer):
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:break
            writer.write(chunk);await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):pass
    finally:
        try:writer.write_eof() if writer.can_write_eof() else None
        except (OSError, RuntimeError):pass


async def handle(pool, client_reader, client_writer, *, connect_timeout=2.):
    upstream = None
    for backend in pool.order():
        try:
            upstream = await asyncio.wait_for(asyncio.open_connection(*backend), connect_timeout)
            chosen = backend;break
        except (OSError, asyncio.TimeoutError):pool.refused[backend] += 1
    if upstream is None:
        pool.unserved += 1;client_writer.close()
        try:await client_writer.wait_closed()
        except OSError:pass
        return
    up_reader, up_writer = upstream;pool.active[chosen] += 1;pool.served[chosen] += 1
    try:await asyncio.gather(pipe(client_reader, up_writer), pipe(up_reader, client_writer))
    finally:
        pool.active[chosen] -= 1
        for w in (up_writer, client_writer):
            w.close()
            try:await w.wait_closed()
            except OSError:pass


async def wait_ready(backends, seconds):
    deadline = time.monotonic() + seconds;pending = set(backends)
    while pending and time.monotonic() < deadline:
        for b in list(pending):
            try:
                _, w = await asyncio.wait_for(asyncio.open_connection(*b), 1.);w.close();pending.discard(b)
                try:await w.wait_closed()
                except OSError:pass
            except (OSError, asyncio.TimeoutError):pass
        if pending:await asyncio.sleep(.5)
    return sorted(pending)


async def run(listen, backends, *, wait_ready_seconds=0., report_seconds=60., log=print, started=None):
    pool = Pool(backends)
    if wait_ready_seconds > 0:
        missing = await wait_ready(backends, wait_ready_seconds)
        if missing:raise SystemExit(f'backends not ready after {wait_ready_seconds} s: ' + ', '.join(f'{h}:{p}' for h, p in missing))
    server = await asyncio.start_server(lambda r, w: handle(pool, r, w), *listen)
    log(json.dumps(dict(event='READY', listen=f'{listen[0]}:{server.sockets[0].getsockname()[1]}', backends=[f'{h}:{p}' for h, p in backends])))
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
    p.add_argument('--wait-ready', type=float, default=0., help='seconds to wait for every backend to accept a probe before listening')
    p.add_argument('--report-seconds', type=float, default=60.)
    args = p.parse_args(argv)
    def log(line):print(line, flush=True)
    try:asyncio.run(run(args.listen, args.backend, wait_ready_seconds=args.wait_ready, report_seconds=args.report_seconds, log=log))
    except KeyboardInterrupt:return 0
    return 0


if __name__ == '__main__':sys.exit(main())
