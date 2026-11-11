import asyncio
from collections import Counter
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


class StickyTests(unittest.TestCase):
    """Battle affinity: a battle's decisions stay on one backend (its constructor session); new battles go to the least-loaded GPU,
    ties by in-flight requests, then the least recently placed on, then --backend order."""
    def run_sticky(self, scenario, names, **kw):
        async def main():
            served, servers, backends, rows = [], {}, [], []
            for name in names:
                s = await framed_backend(name, 0., served);servers[name] = s;backends.append(('127.0.0.1', s.sockets[0].getsockname()[1]))
            box = {}
            task = asyncio.create_task(P.run(('127.0.0.1', 0), backends, log=lambda line: None, report_seconds=0, log_row=rows.append,
                started=lambda server, pool: box.update(port=server.sockets[0].getsockname()[1], pool=pool), **kw))
            while 'port' not in box:await asyncio.sleep(.01)
            try:return await scenario(box['port'], box['pool'], servers, rows)
            finally:
                task.cancel()
                for s in servers.values():s.close()
        return asyncio.run(main())

    def test_a_battle_sticks_to_its_home_and_new_battles_spread_to_the_least_recently_placed(self):
        async def scenario(port, pool, servers, rows):
            out = []
            for battle in ('x', 'y', 'x', 'z', 'y', 'x', 'w'):
                out.append((await request(port, dict(battle_id=battle, i=len(out))))['backend'])
            for _ in range(100):
                if len(rows) >= 7:break
                await asyncio.sleep(.01)
            return out, rows
        out, rows = self.run_sticky(scenario, ['a', 'b', 'c'])
        self.assertEqual(out, ['a', 'b', 'a', 'c', 'b', 'a', 'a'])   # w: one live battle each, none in flight: a was placed on longest ago
        self.assertEqual([(r['battle_id'], r['home']) for r in rows][:6], [('x', False), ('y', False), ('x', True), ('z', False), ('y', True), ('x', True)])
        self.assertTrue(all(r['gpu'] == r['backend'] for r in rows))   # unlabelled backends: each its own group

    def test_seat_twins_are_two_battles_with_their_own_homes(self):
        # the review's case: the two seats of one seed share a battle_id; the server keeps one session per (battle, owner)
        async def scenario(port, pool, servers, rows):
            rounds = []
            for k in range(3):
                outs = await asyncio.gather(request(port, dict(battle_id='s7', owner='p1', k=k)), request(port, dict(battle_id='s7', owner='p2', k=k)))
                rounds.append(tuple(o['backend'] for o in outs))
            return rounds, rows
        rounds, rows = self.run_sticky(scenario, ['a', 'b', 'c'])
        self.assertEqual(len(set(rounds)), 1); self.assertEqual(len(set(rounds[0])), 2)   # each seat keeps one backend, the two differ
        self.assertEqual(sorted({(r['battle_id'], r['owner']) for r in rows}), [('s7', 'p1'), ('s7', 'p2')])

    def test_concurrent_new_battles_spread_and_an_unbounded_length_is_forwarded(self):
        async def scenario(port, pool, servers, rows):
            outs = await asyncio.gather(request(port, dict(battle_id='x')), request(port, dict(battle_id='y')))
            return sorted(o['backend'] for o in outs)
        self.assertEqual(self.run_sticky(scenario, ['a', 'b']), ['a', 'b'])   # in-flight counts separate two new battles
        got = []
        async def main():
            async def handle(r, w): got.append(await r.read(8)); w.close()
            s = await asyncio.start_server(handle, '127.0.0.1', 0); box = {}
            task = asyncio.create_task(P.run(('127.0.0.1', 0), [('127.0.0.1', s.sockets[0].getsockname()[1])], log=lambda line: None, report_seconds=0,
                started=lambda server, pool: box.update(port=server.sockets[0].getsockname()[1])))
            while 'port' not in box: await asyncio.sleep(.01)
            r, w = await asyncio.open_connection('127.0.0.1', box['port']); w.write(struct.pack('!I', P.MAX_PACKET + 1)); await w.drain()
            await asyncio.wait_for(r.read(), 5); w.close(); task.cancel(); s.close()
        asyncio.run(main())
        self.assertEqual(got, [struct.pack('!I', P.MAX_PACKET + 1)])   # the header goes on, for the server to refuse

    def test_a_dead_home_rehomes_its_battle(self):
        async def scenario(port, pool, servers, rows):
            first = (await request(port, dict(battle_id='x')))['backend']
            servers[first].close();await servers[first].wait_closed()
            second = (await request(port, dict(battle_id='x')))['backend'];third = (await request(port, dict(battle_id='x')))['backend']
            return first, second, third
        first, second, third = self.run_sticky(scenario, ['a', 'b'])
        self.assertEqual((first, second, third), ('a', 'b', 'b'))   # setup to a fails: b, and x now lives on b

    def test_a_client_that_sends_no_request_is_unserved_and_nothing_is_forwarded(self):
        async def scenario(port, pool, servers, rows):
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            closed = await asyncio.wait_for(reader.read(), 5);writer.close()
            for _ in range(100):
                if rows:break
                await asyncio.sleep(.01)
            return closed, (pool.no_request, pool.unserved), rows, sum(pool.served.values())
        closed, counts, rows, served = self.run_sticky(scenario, ['a'], first_frame_timeout=.2)
        self.assertEqual((closed, counts, served, rows[0]['reason']), (b'', (1, 0), 0, 'no request'))   # apart from unserved

    def test_a_request_without_a_battle_id_is_still_forwarded_byte_for_byte(self):
        async def scenario(port, pool, servers, rows):
            return await request(port, dict(i=1)), await request(port, [1, 2])
        one, two = self.run_sticky(scenario, ['a', 'b'])
        self.assertEqual((one['echo'], two['echo']), (dict(i=1), [1, 2]))

    def test_new_battles_go_to_the_least_loaded_gpu_first(self):
        h = lambda port: ('h', port);gpus = {h(1): 'g0', h(2): 'g0', h(3): 'g1', h(4): 'g1'}
        pool = P.Pool([h(1), h(2), h(3), h(4)], gpus=gpus);picks = []
        for k, battle in enumerate('xyzw'):
            pick = pool.order(now=float(k), battle=battle)[0];pool.bind(battle, pick, now=float(k));picks.append(pick)
        self.assertEqual(picks, [h(1), h(3), h(2), h(4)])   # x on g0, y on g1 (fewer there), z on g0's free backend, w on g1's
        self.assertEqual(pool.order(now=5., battle='v')[0], h(1))   # 2 and 2, none in flight: g0 was placed on less recently (z before w)
        self.assertEqual(pool.order(now=100., battle='v')[0], h(1))   # all idle past 60 s: nothing counts
        self.assertEqual(pool.snapshot()['backends'][0]['gpu'], 'g0')
        with self.assertRaisesRegex(ValueError, 'unknown backend'):P.Pool([h(1)], gpus={h(9): 'g0'})
        for bad in ({h(1): 'g0', h(2): 'g0', h(3): 'g1'}, {h(1): 'g0', h(2): 'g1'}):   # 2+1 per GPU; a 4th backend unlabelled
            with self.subTest(bad), self.assertRaisesRegex(ValueError, 'same number of backends'):P.Pool([h(1), h(2), h(3), h(4)], gpus=bad)
        self.assertEqual(P.backend_spec('127.0.0.1:7001@ 3 '), (('127.0.0.1', 7001), '3'))
        pool = P.Pool([h(1), h(2), h(3), h(4)], gpus=gpus)   # the GPU first, then the backend (the review's discriminating case)
        for k, (battle, home) in enumerate((('a', 1), ('b', 1), ('c', 1), ('d', 3), ('e', 4))):pool.bind(battle, h(home), now=float(k))
        self.assertEqual(pool.order(now=5., battle='new')[0], h(3))   # g0 holds 3 (h2 empty), g1 holds 2: g1, then h3 by order
        self.assertEqual(pool.snapshot()['live_battles_per_gpu'], {})   # all idle by the wall clock: snapshot reads time.monotonic()
        live = Counter({h(1): 2, h(2): 3});self.assertEqual(pool.per_gpu(live), {'g0': 5})
        self.assertEqual(P.backend_spec('127.0.0.1:7001@3'), (('127.0.0.1', 7001), '3'))
        self.assertEqual(P.backend_spec('127.0.0.1:7001'), (('127.0.0.1', 7001), None))
        for bad in ('127.0.0.1:7001@', 'nohost@1'):
            with self.subTest(bad), self.assertRaises(Exception):P.backend_spec(bad)

    def test_an_ended_battle_stops_counting_at_once_and_ties_go_to_the_quieter_then_least_recently_placed_gpu(self):
        h = lambda port: ('h', port);gpus = {h(1): 'g0', h(2): 'g0', h(3): 'g1', h(4): 'g1'};pool = P.Pool([h(1), h(2), h(3), h(4)], gpus=gpus)
        for k, (battle, home) in enumerate((('x', 1), ('y', 3), ('z', 2))):pool.bind(battle, h(home), now=float(k))
        self.assertEqual(pool.gpu[pool.order(now=3., battle='new')[0]], 'g1')   # g0 runs x and z, g1 runs y
        self.assertEqual((pool.end('x'), pool.end('z'), pool.end('x')), (True, True, True))   # idempotent
        self.assertEqual(pool.gpu[pool.order(now=3., battle='new')[0]], 'g0')   # both ended: g0 holds 0 against g1's 1, within 60 s
        self.assertEqual(pool.order(now=3., battle='x')[0], h(1))   # an ended battle keeps its home: a stray request still goes there
        self.assertFalse(pool.end('never'));self.assertEqual((pool.ended, pool.ended_unknown), (3, 1))   # unknown: counted, nothing else
        pool.end('y');pool.active[h(1)] = 1   # all ended: a tie on live battles; g0 has a request in flight
        self.assertEqual(pool.order(now=3., battle='new')[0], h(3))   # the GPU with fewer in flight
        pool.active[h(1)] = 0   # in-flight tie too: g1 was given its battle (y, at 1) before g0's latest (z, at 2)
        self.assertEqual(pool.order(now=3., battle='new')[0], h(3))
        pool.bind('w', h(4), now=4.);pool.end('w')   # now g1 was placed on last
        self.assertEqual(pool.order(now=5., battle='new')[0], h(1))
        self.assertTrue(pool.bind('x', h(1), now=6.))   # a return home is not a placement
        self.assertEqual(pool.placed_at, {'g0': 2., 'g1': 4.})
        snap = pool.snapshot();self.assertEqual((snap['battle_ends'], snap['battle_ends_unknown']), (5, 1))

    def test_an_end_frame_is_answered_never_forwarded_and_logged(self):
        async def scenario(port, pool, servers, rows):
            first = (await request(port, dict(battle_id='x', owner='p1')))['backend']
            ended = await request(port, dict(schema=P.BATTLE_END_SCHEMA, battle_id='x', owner='p1'))
            unknown = await request(port, dict(schema=P.BATTLE_END_SCHEMA, battle_id='nope', owner='p2'))
            nameless = await request(port, dict(schema=P.BATTLE_END_SCHEMA, owner='p2'))   # no battle_id: unknown, counted, no crash
            assert nameless['status'] == 'UNKNOWN_BATTLE' and pool.ended_unknown == 2
            for _ in range(100):
                if len(rows) >= 4:break
                await asyncio.sleep(.01)
            return first, ended, unknown, sum(pool.served.values()), pool.live(__import__('time').monotonic()), rows
        first, ended, unknown, served, live, rows = self.run_sticky(scenario, ['a', 'b'])
        self.assertEqual(ended, dict(schema=P.BATTLE_END_SCHEMA, battle_id='x', owner='p1', status='ENDED'))
        self.assertEqual(unknown['status'], 'UNKNOWN_BATTLE');self.assertEqual(served, 1)   # only the request reached a backend
        self.assertEqual(sum(live.values()), 0)   # x stopped counting at once, not 60 s after its last decision
        self.assertEqual(sorted((str(r.get('reason')), r['backend'] is None) for r in rows),
                         [('None', False), ('battle end', True), ('battle end: unknown battle', True), ('battle end: unknown battle', True)])

    def test_the_end_schema_is_the_clients(self):
        from pokezero.depthn_live_actor import BATTLE_END_SCHEMA
        self.assertEqual(P.BATTLE_END_SCHEMA, BATTLE_END_SCHEMA)

    def test_the_first_frame_has_one_deadline_and_survives_deep_json(self):
        async def main():
            frame = lambda body: struct.pack('!I', len(body)) + body
            deep = b'['*100000 + b']'*100000
            r = asyncio.StreamReader();r.feed_data(frame(deep));r.feed_eof()
            first, battle, end = await P.first_frame(r, 1.)   # RecursionError in json: no battle, bytes intact
            assert (first, battle, end) == (frame(deep), None, False)
            r = asyncio.StreamReader();body = json.dumps(dict(battle_id='x', owner='p1')).encode();head = frame(body)[:4]
            loop = asyncio.get_running_loop()
            loop.call_later(.15, r.feed_data, head);loop.call_later(.3, r.feed_data, body)   # each part within .2, the whole not
            try:await P.first_frame(r, .2)
            except asyncio.TimeoutError:return 'timeout'
            return 'read'
        self.assertEqual(asyncio.run(main()), 'timeout')

    def test_live_battles_expire_and_homes_are_bounded(self):
        pool = P.Pool([('h', 1), ('h', 2)], sticky_idle=10., sticky_cap=2)
        self.assertFalse(pool.bind('x', ('h', 1), now=0.));self.assertTrue(pool.bind('x', ('h', 1), now=1.))
        self.assertEqual(pool.order(now=2., battle='y')[0], ('h', 2))   # x is live on h:1
        self.assertEqual(pool.order(now=20., battle='y')[0], ('h', 2))   # x idle: both empty, none in flight: h:2 never given a battle
        pool.bind('y', ('h', 2), now=21.);pool.bind('z', ('h', 1), now=22.)
        self.assertEqual(list(pool.homes), ['y', 'z'])   # LRU: x evicted
        self.assertEqual(pool.order(now=23., battle='z')[0], ('h', 1))   # z's home, though h:2 holds as many
        pool.failed_at[('h', 1)] = 23.;self.assertEqual(pool.order(now=24., battle='z')[0], ('h', 2))   # a cooling home is skipped
        pool.failed_at[('h', 1)] = None
        pool.bind('y', ('h', 2), now=24.5);pool.bind('w', ('h', 1), now=24.6)
        self.assertEqual(list(pool.homes), ['y', 'w'])   # a rebind refreshes: z (older) goes, not y


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


def identity_backend(name, script, seen, *, tree='t1', archive='a1', worlds=16, precision='highest', K=8, reply=None, view_b=False, views='jit'):
    """Answers each framed request with the next status in `script` (COMPLETE, or a refusal cause) and the identity fields."""
    steps = iter(script)
    async def handle(reader, writer):
        try:
            n = struct.unpack('!I', await reader.readexactly(4))[0];q = json.loads(await reader.readexactly(n));seen.append((name, q))
            step = next(steps, 'COMPLETE')
            out = json.dumps(dict(request_id=q['request_id'], status='COMPLETE' if step == 'COMPLETE' else 'REFUSED', cause=None if step == 'COMPLETE' else step,
                refusal=None, rung=q['rung'], archive_sha256=archive, requested_worlds=worlds, root_operator='ARGMAX_MEAN_COMPLETED_Q',
                tree_commit=tree, pricer=dict(matmul_precision=precision, K=K, W=worlds),
                **({} if reply is None else dict(reply_prior=dict(operator='br_fp', views=views, reply_archive_sha256=reply))),
                **(dict(peer_view_b=dict(enabled=True)) if view_b else {}))).encode()
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

    def test_the_reply_model_and_peer_view_b_are_pinned(self):
        brfp = dict(reply='r' * 64, view_b=True)
        ready, _ = self.go([('a', [], brfp), ('b', [], brfp)], expect=dict(self.EXPECT, reply_archive_sha256='r' * 64, peer_view_b='on', reply_views='jit'))
        self.assertEqual((ready['identity']['reply_archive_sha256'], ready['identity']['peer_view_b']), ('r' * 64, 'on'))
        ready, _ = self.go([('a', [], {})], expect=dict(self.EXPECT, reply_archive_sha256='none', peer_view_b='off'))   # r3/r4b: no reply model
        self.assertEqual(ready['identity']['reply_archive_sha256'], 'none')
        for kw in ({}, dict(reply='s' * 64, view_b=True), dict(reply='r' * 64)):   # a server without the reply model, another one, or view b off
            with self.subTest(kw=kw), self.assertRaisesRegex(SystemExit, 'differs from the expected'):
                self.go([('a', [], kw)], expect=dict(self.EXPECT, reply_archive_sha256='r' * 64, peer_view_b='on'))
        with self.assertRaisesRegex(SystemExit, 'disagree'):self.go([('a', [], brfp), ('b', [], {})])   # unpinned: still must agree
        with self.assertRaisesRegex(SystemExit, 'disagree'):self.go([('a', [], brfp), ('b', [], dict(brfp, views='eager'))])   # mixed views
        with self.assertRaisesRegex(SystemExit, 'differs from the expected'):
            self.go([('a', [], dict(brfp, views='eager'))], expect=dict(self.EXPECT, reply_archive_sha256='r' * 64, reply_views='jit'))

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
