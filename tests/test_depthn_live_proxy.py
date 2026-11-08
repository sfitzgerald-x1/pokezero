import asyncio
import hashlib
import json
import socket
import struct
import unittest
from unittest import mock

from scripts import depthn_live_proxy as P


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0));return s.getsockname()[1]


async def framed_backend(name, delay, served):
    """A backend speaking the search-server framing: read one length-prefixed request, answer {backend, echo} after `delay`."""
    async def handle(reader, writer):
        try:
            n = struct.unpack('!I', await reader.readexactly(4))[0];body = json.loads(await reader.readexactly(n))
            served.append(name);await asyncio.sleep(delay)
            out = json.dumps(dict(backend=name, echo=body)).encode();writer.write(struct.pack('!I', len(out)) + out);await writer.drain()
        except asyncio.IncompleteReadError:pass
        finally:writer.close()
    return await asyncio.start_server(handle, '127.0.0.1', 0)


async def request(port, body):
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    raw = json.dumps(body).encode();writer.write(struct.pack('!I', len(raw)) + raw);await writer.drain()
    try:   # bounded: a proxy that never forwards EOF or truncates fails the test instead of hanging it
        n = struct.unpack('!I', await asyncio.wait_for(reader.readexactly(4), 10))[0];return json.loads(await asyncio.wait_for(reader.readexactly(n), 10))
    except asyncio.IncompleteReadError:return None
    finally:writer.close()


class ProxyTests(unittest.TestCase):
    def run_proxy(self, scenario, backends_spec, **kw):
        async def main():
            served, servers, backends = [], [], []
            for name, delay in backends_spec:
                if delay is None:backends.append(('127.0.0.1', free_port()));continue   # nothing listens there
                s = await framed_backend(name, delay, served);servers.append(s);backends.append(('127.0.0.1', s.sockets[0].getsockname()[1]))
            box, lines = {}, []
            task = asyncio.create_task(P.run(('127.0.0.1', 0), backends, log=lines.append, report_seconds=0,
                started=lambda server, pool: box.update(port=server.sockets[0].getsockname()[1], pool=pool), **kw))
            for _ in range(200):
                if 'port' in box or task.done():break
                await asyncio.sleep(.01)
            if task.done():task.result()
            try:return await scenario(box['port'], box['pool'], served, lines)
            finally:
                task.cancel()
                for s in servers:s.close()
        return asyncio.run(main())

    def test_forwards_byte_for_byte_and_reports_ready(self):
        async def scenario(port, pool, served, lines):
            out = await request(port, dict(request_id='r1', lines=['|request|{}']));return out, lines
        out, lines = self.run_proxy(scenario, [('a', 0.)])
        self.assertEqual(out, dict(backend='a', echo=dict(request_id='r1', lines=['|request|{}'])))
        self.assertEqual(json.loads(lines[0])['event'], 'READY')

    def test_concurrent_requests_spread_to_the_least_busy(self):
        async def scenario(port, pool, served, lines):
            outs = await asyncio.gather(*(request(port, dict(i=i)) for i in range(3)))
            for _ in range(100):   # the proxy sees each client's close just after the response: in-flight counts drain to 0
                if not any(pool.active.values()):break
                await asyncio.sleep(.01)
            return sorted(o['backend'] for o in outs), pool.snapshot()
        names, snap = self.run_proxy(scenario, [('a', .3), ('b', .3), ('c', .3)])
        self.assertEqual(names, ['a', 'b', 'c'])   # three in flight at once: one per backend
        self.assertEqual([b['active'] for b in snap['backends']], [0, 0, 0])
        self.assertEqual(sorted(b['served'] for b in snap['backends']), [1, 1, 1])

    def test_sequential_requests_rotate(self):
        async def scenario(port, pool, served, lines):
            return [(await request(port, dict(i=i)))['backend'] for i in range(4)]
        self.assertEqual(sorted(set(self.run_proxy(scenario, [('a', 0.), ('b', 0.)]))), ['a', 'b'])

    def test_a_refusing_backend_is_skipped_and_counted(self):
        async def scenario(port, pool, served, lines):
            outs = [await request(port, dict(i=i)) for i in range(3)];return [o['backend'] for o in outs], pool.snapshot()
        names, snap = self.run_proxy(scenario, [('a', 0.), ('dead', None)])
        self.assertEqual(names, ['a', 'a', 'a']);self.assertGreaterEqual(snap['backends'][1]['refused'], 1)
        self.assertEqual(snap['backends'][1]['active'], 0);self.assertTrue(snap['backends'][1]['cooling'])   # a failed setup releases its reservation

    def test_no_backend_closes_the_client_connection(self):
        async def scenario(port, pool, served, lines):
            return await request(port, dict(i=0)), pool.unserved
        out, unserved = self.run_proxy(scenario, [('dead1', None), ('dead2', None)])
        self.assertIsNone(out);self.assertEqual(unserved, 1)

    def test_wait_ready_refuses_when_a_backend_never_listens(self):
        async def main():
            dead = ('127.0.0.1', free_port())
            with self.assertRaises(SystemExit):await P.run(('127.0.0.1', 0), [dead], wait_ready_seconds=.6, report_seconds=0, log=lambda line: None)
        asyncio.run(main())

    def test_cli_validation(self):
        with self.assertRaises(SystemExit):P.main(['--listen', 'nohost', '--backend', '127.0.0.1:1'])
        with self.assertRaises(ValueError):P.Pool([('h', 1), ('h', 1)])
        for bad in (['--wait-ready', 'nan'], ['--wait-ready', '-1'], ['--wait-ready', '1e9'], ['--connect-timeout', '0'], ['--expect-rung', 'DN2W16']):
            with self.subTest(bad=bad), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                P.main(['--listen', '127.0.0.1:1', '--backend', '127.0.0.1:2'] + bad)

    def test_listen_must_not_be_a_backend(self):
        async def main():
            with self.assertRaises(SystemExit):await P.run(('127.0.0.1', 5), [('127.0.0.1', 5)], report_seconds=0, log=lambda line: None)
        asyncio.run(main())

    def test_simultaneous_clients_reserve_distinct_backends(self):
        # the review's case: 'a' busy, turns moved on, then two clients in the same instant whose rings both put 'b' first; only a
        # reservation taken before connecting sends the second to 'c'
        async def scenario(port, pool, served, lines):
            first = asyncio.create_task(request(port, dict(i='hold')))   # turn 0: a
            while sum(pool.active.values()) < 1:await asyncio.sleep(.005)
            for i in range(2):await request(port, dict(i=f's{i}'))   # turns 1, 2
            for _ in range(100):
                if sum(pool.active.values()) == 1:break
                await asyncio.sleep(.01)
            outs = await asyncio.gather(*(request(port, dict(i=i)) for i in range(2)));held = await first   # turns 0, 1
            return held['backend'], sorted(o['backend'] for o in outs)
        held, pair = self.run_proxy(scenario, [('a', .6), ('b', .15), ('c', .15)])
        self.assertEqual(held, 'a');self.assertEqual(pair, ['b', 'c'])

    def test_a_hung_backend_costs_one_connect_timeout_then_cools_down(self):
        real = asyncio.open_connection;hung = {}
        async def fake(host, port, *a, **k):
            if port == hung.get('port'):await asyncio.sleep(3600)
            return await real(host, port, *a, **k)
        async def scenario(port, pool, served, lines):
            loop = asyncio.get_running_loop();times = []
            for i in range(3):
                t = loop.time();out = await request(port, dict(i=i));times.append(loop.time() - t)
                self.assertEqual(out['backend'], 'a')
            return times, pool.snapshot()
        with mock.patch.object(P.asyncio, 'open_connection', fake):
            port = free_port();hung['port'] = port   # every connection setup to this backend hangs
            async def main():
                served = [];a = await framed_backend('a', 0., served);box = {}
                backends = [('127.0.0.1', port), ('127.0.0.1', a.sockets[0].getsockname()[1])]
                task = asyncio.create_task(P.run(('127.0.0.1', 0), backends, log=lambda line: None, report_seconds=0, connect_timeout=.2, cooldown=60.,
                    started=lambda server, pool: box.update(port=server.sockets[0].getsockname()[1], pool=pool)))
                while 'port' not in box:await asyncio.sleep(.01)
                try:return await scenario(box['port'], box['pool'], served, [])
                finally:task.cancel();a.close()
            times, snap = asyncio.run(main())
        self.assertLess(max(times), 1.)   # bounded by the connect timeout, never the client's budget
        self.assertEqual(snap['backends'][0]['refused'], 1)   # after one failure it sorts last: tried once, not per request

    def test_cooldown_sorts_a_failed_backend_last(self):
        pool = P.Pool([('h', 1), ('h', 2)], cooldown=5.);pool.failed_at[('h', 1)] = 100.
        self.assertEqual([pool.order(now=101.)[0] for _ in range(4)], [('h', 2)] * 4)
        self.assertEqual(sorted(pool.order(now=106.)[0] for _ in range(2)), [('h', 1), ('h', 2)])

    def test_connection_log_names_the_backend(self):
        rows = []
        async def scenario(port, pool, served, lines):
            for i in range(2):await request(port, dict(i=i))
            for _ in range(100):
                if len(rows) >= 2:break
                await asyncio.sleep(.01)
            return rows
        out = self.run_proxy(scenario, [('a', 0.)], log_row=rows.append)
        self.assertEqual(len(out), 2)
        for row in out:
            self.assertRegex(row['backend'], r'^127\.0\.0\.1:\d+$');self.assertGreater(row['bytes_to_backend'], 4)
            self.assertGreater(row['bytes_to_client'], 4);self.assertGreaterEqual(row['seconds'], 0.)
        dead = []
        self.run_proxy(lambda port, pool, served, lines: request(port, {}), [('dead', None)], log_row=dead.append)
        self.assertEqual([r['backend'] for r in dead], ['unserved'])


async def codex_like_backend(name, delay, served, *, answer=True):
    """Like phase_d_search_server.Server.handle: reads framed requests until EOF; never closes after a response."""
    async def handle(reader, writer):
        try:
            while True:
                n = struct.unpack('!I', await reader.readexactly(4))[0];raw = await reader.readexactly(n)
                served.append(name);await asyncio.sleep(delay)
                if not answer:break
                if n > 1000:out = json.dumps(dict(backend=name, sha=hashlib.sha256(raw).hexdigest(), pad='y' * (P.MAX_PACKET - 200))).encode()
                else:out = json.dumps(dict(backend=name, echo=json.loads(raw))).encode()
                writer.write(struct.pack('!I', len(out)) + out);await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):pass
        finally:writer.close()
    return await asyncio.start_server(handle, '127.0.0.1', 0)


class CodexLikeBackendTests(unittest.TestCase):
    def harness(self, scenario, specs):
        async def main():
            served, servers, backends = [], [], []
            for name, delay, answer in specs:
                s = await codex_like_backend(name, delay, served, answer=answer);servers.append(s);backends.append(('127.0.0.1', s.sockets[0].getsockname()[1]))
            box = {}
            task = asyncio.create_task(P.run(('127.0.0.1', 0), backends, log=lambda line: None, report_seconds=0,
                started=lambda server, pool: box.update(port=server.sockets[0].getsockname()[1], pool=pool)))
            while 'port' not in box:await asyncio.sleep(.01)
            try:return await scenario(box['port'], box['pool'], served)
            finally:
                task.cancel()
                for s in servers:s.close()
        return asyncio.run(main())

    def test_in_flight_drains_when_the_client_closes(self):
        async def scenario(port, pool, served):
            out = await request(port, dict(i=1))
            for _ in range(100):
                if not any(pool.active.values()):break
                await asyncio.sleep(.01)
            return out, list(pool.active.values())
        out, active = self.harness(scenario, [('a', 0., True)])
        self.assertEqual(out['backend'], 'a');self.assertEqual(active, [0])

    def test_16_mib_both_ways(self):
        async def scenario(port, pool, served):
            body = json.dumps(dict(pad='x' * (P.MAX_PACKET - 100))).encode()
            r, w = await asyncio.open_connection('127.0.0.1', port)
            w.write(struct.pack('!I', len(body)) + body);await asyncio.wait_for(w.drain(), 10)
            n = struct.unpack('!I', await asyncio.wait_for(r.readexactly(4), 10))[0];out = json.loads(await asyncio.wait_for(r.readexactly(n), 10));w.close()
            return out['sha'] == hashlib.sha256(body).hexdigest(), n
        same, n = self.harness(scenario, [('a', 0., True)])
        self.assertTrue(same);self.assertGreater(n, P.MAX_PACKET - 300)

    def test_a_request_is_never_replayed_when_its_backend_dies(self):
        async def scenario(port, pool, served):
            return [await request(port, dict(i=i)) for i in range(4)], list(served)
        outs, served = self.harness(scenario, [('crash', 0., False), ('ok', 0., True)])
        self.assertEqual(len(served), 4);self.assertIn('crash', served)
        self.assertEqual(sum(o is None for o in outs), served.count('crash'))   # the client sees EOF; no second backend gets it


def identity_backend(name, script, seen, *, tree='t1', archive='a1', worlds=16, precision='highest', K=8):
    """Answers each framed request with the next status in `script` (COMPLETE, or a refusal cause) and the identity fields."""
    steps = iter(script)
    async def handle(reader, writer):
        try:
            n = struct.unpack('!I', await reader.readexactly(4))[0];q = json.loads(await reader.readexactly(n));seen.append((name, q))
            step = next(steps, 'COMPLETE')
            out = json.dumps(dict(request_id=q['request_id'], status='COMPLETE' if step == 'COMPLETE' else 'REFUSED', cause=None if step == 'COMPLETE' else step,
                refusal=None, rung=q['rung'], archive_sha256=archive, requested_worlds=worlds, root_operator='ARGMAX_MEAN_COMPLETED_Q',
                tree_commit=tree, pricer=dict(matmul_precision=precision, K=K, W=worlds))).encode()
            writer.write(struct.pack('!I', len(out)) + out);await writer.drain()
        except asyncio.IncompleteReadError:pass
        finally:writer.close()
    return handle


class IdentityTests(unittest.TestCase):
    EXPECT = dict(rung='DN2W16', archive_sha256='a1', requested_worlds=16, matmul_precision='highest', root_operator='ARGMAX_MEAN_COMPLETED_Q')
    REQUESTS = [dict(rung='DN2W16', battle_id='captured', request_id='x', lines=['|request|{}']), dict(rung='DN2W16', battle_id='captured2', request_id='y', lines=[])]

    def go(self, specs, expect=None):
        async def main():
            seen, servers, backends, lines = [], [], [], []
            for name, script, kw in specs:
                s = await asyncio.start_server(identity_backend(name, script, seen, **kw), '127.0.0.1', 0)
                servers.append(s);backends.append(('127.0.0.1', s.sockets[0].getsockname()[1]))
            box = {}
            task = asyncio.create_task(P.run(('127.0.0.1', 0), backends, log=lines.append, report_seconds=0, probe_requests=self.REQUESTS,
                expect=self.EXPECT if expect is None else expect, probe_timeout=5., started=lambda server, pool: box.update(started=True)))
            try:
                for _ in range(300):
                    if box or task.done():break
                    await asyncio.sleep(.01)
                if task.done():task.result()
                return json.loads(lines[0]), seen
            finally:
                task.cancel()
                for s in servers:s.close()
        return asyncio.run(main())

    def test_every_backend_is_probed_directly_with_fresh_ids(self):
        ready, seen = self.go([('a', [], {}), ('b', ['constructor_refused'], {})])
        self.assertTrue(ready['probed']);self.assertEqual(ready['identity']['archive_sha256'], 'a1')
        self.assertEqual(sorted({n for n, _ in seen}), ['a', 'b']);self.assertEqual(len(seen), 3)   # b moved to its second root
        ids = [q['request_id'] for _, q in seen] + [q['battle_id'] for _, q in seen]
        self.assertEqual(len(set(ids)), 6);self.assertNotIn('captured', ids)

    def test_one_wrong_backend_refuses_to_start(self):
        for kw in (dict(archive='other'), dict(precision='default'), dict(worlds=8)):
            with self.subTest(kw=kw), self.assertRaisesRegex(SystemExit, 'differs from the expected'):self.go([('a', [], {}), ('b', [], kw)])

    def test_backends_that_disagree_refuse_to_start(self):
        for kw in (dict(tree='t2'), dict(K=4)):   # tree commit, or any pricer field
            with self.subTest(kw=kw), self.assertRaisesRegex(SystemExit, 'disagree'):self.go([('a', [], {}), ('b', [], kw)])

    def test_a_backend_without_a_complete_probe_refuses_to_start(self):
        with self.assertRaisesRegex(SystemExit, 'no COMPLETE'):self.go([('a', ['server_error'], {})])
        with self.assertRaisesRegex(SystemExit, 'no COMPLETE'):self.go([('a', ['depthn_refused', 'class_refused'], {})])
        with self.assertRaisesRegex(SystemExit, 'identity probe failed'):
            async def main():await P.check_identity([('127.0.0.1', free_port())], self.REQUESTS, {}, timeout=1.)
            asyncio.run(main())


if __name__ == '__main__':
    unittest.main()
