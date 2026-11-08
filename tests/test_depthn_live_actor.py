"""Inert DepthN live actor controls: fake socket server and fake raw policy only."""
import asyncio
import json
import os
from pathlib import Path
import random
import socket
import struct
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from pokezero import depthn_live_actor as live
from pokezero import phase_d_search_server as wire
from pokezero.live_protocol_actor import ActorContract, JaxBackend, OwnedProvider, RawPolicy
from pokezero.live_protocol_bridge import OwningActorFactory
from pokezero.policy import PolicyDecision
from tests import test_phase_d_prerequisites as fixture

REQUEST_KEYS = {'schema','request_id','battle_id','owner','stream_id','decision_index','generation',
                'prefix_sha256','lines','own_request_line','rung','budget_seconds'}
ROW_KEYS = {'schema','arm','policy_id','cell','battle_id','owner','stream_id','generation','turn','decision_index','request_kind',
            'prefix_sha256','archive_sha256','archive_source','deadline_seconds','reserve_seconds','encode_seconds','greedy','search','played_action_index','source','cause',
            'search_differs','select_seconds'}


def decode_edge():
    """Deepest '{"zz":[[...]]}' json.loads accepts here; the row holds it one level deeper."""
    low, high = 1, 200000
    while low < high:
        middle = (low+high+1)//2
        try: json.loads('{"zz":'+'['*middle+']'*middle+'}'); low = middle
        except RecursionError: high = middle-1
    return low


def nested(depth):
    value = []
    for _ in range(depth-1): value = [value]
    return value


def complete(request, *, action=4, q=.25, **extra):
    values = [None]*9; values[0] = -.1; values[4] = q
    return dict(schema=wire.SCHEMA, request_id=request['request_id'], rung=request['rung'], archive_sha256='a'*64,
        status='COMPLETE', action_index=action, q_values=values, refusal=None, cause=None, requested_worlds=8,
        constructed_worlds=8, decision_seconds=.5, root_operator='ARGMAX_MEAN_COMPLETED_Q', policy_target=[0.]*9,
        root_value=.2, timings=dict(queue_seconds=.01, prepare_seconds=.2, build_seconds=.05, price_seconds=.2,
        reduce_seconds=.01), **extra)


def refused(request, cause):
    return dict(schema=wire.SCHEMA, request_id=request['request_id'], rung=request['rung'], archive_sha256='a'*64,
        status='REFUSED', action_index=None, q_values=None, refusal='fixture '+cause, cause=cause,
        requested_worlds=8, constructed_worlds=None, decision_seconds=None)


def exact(conn, n):
    output = bytearray()
    while len(output) < n:
        piece = conn.recv(n-len(output))
        if not piece: raise ConnectionError('fixture client closed')
        output.extend(piece)
    return bytes(output)


class FakeServer:
    """One accept loop thread; reply(request, conn) writes or withholds a response."""
    def __init__(self, reply):
        self.reply, self.requests, self.connections, self.stop = reply, [], 0, threading.Event()
        self.listener = socket.create_server(('127.0.0.1', 0)); self.listener.settimeout(.05)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.serve, daemon=True); self.thread.start()

    def serve(self):
        while not self.stop.is_set():
            try: conn, _ = self.listener.accept()
            except (TimeoutError, OSError): continue
            self.connections += 1
            with conn:
                conn.settimeout(5)
                try:
                    request = json.loads(exact(conn, struct.unpack('!I', exact(conn, 4))[0]))
                    self.requests.append(request); self.reply(request, conn)
                except OSError: pass

    def close(self):
        self.stop.set(); self.thread.join(10); self.listener.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def send(builder):
    return lambda request, conn: conn.sendall(wire.encode(builder(request)))


def raw_reply(text_builder):
    """Send a literal body (bypassing encode) built from the canonical COMPLETE body text."""
    def reply(request, conn):
        body = text_builder(wire.encode(complete(request))[4:].decode()).encode()
        conn.sendall(struct.pack('!I', len(body))+body)
    return reply


def snapshot():
    line = fixture.request()
    return NS(battle_id='literal', owner='p1', stream_id=1, generation=2, prefix_sha256='c'*64,
              lines=('|turn|1', line), own_request_line=line)


def request_line(*, members=2, force=False, trapped=False):
    packet = json.loads(fixture.request()[len('|request|'):]); side = packet['side']; key = next(k for k in side if k != 'id')
    base = side[key][1]
    side[key] = side[key][:1]+[dict(base, ident=base['ident'][:-1]+str(i)) for i in range(1, members)]
    if force: packet.pop('active'); packet['forceSwitch'] = [True]
    if trapped: packet['active'][0]['trapped'] = True
    return '|request|'+json.dumps(packet)


class ClientTests(unittest.TestCase):
    def call(self, server, *, deadline=3., reserve=.15, started=None):
        client = live.Client('127.0.0.1', server.port, rung='DN2W8', deadline_seconds=deadline, reserve_seconds=reserve)
        return client.request(snapshot(), decision_index=3, started=live.clock() if started is None else started)

    def test_complete_carries_contract_request_and_client_time(self):
        with FakeServer(send(complete)) as server:
            response = self.call(server)
        self.assertEqual(response['status'], 'COMPLETE'); self.assertIsNone(response['cause'])
        self.assertFalse(response['client_refused']); self.assertGreaterEqual(response['client_wall_seconds'], 0)
        request = server.requests[0]
        self.assertEqual(set(request), REQUEST_KEYS); self.assertEqual(server.connections, 1)
        self.assertEqual((request['rung'], request['decision_index'], request['schema']), ('DN2W8', 3, wire.SCHEMA))
        self.assertEqual(request['lines'], list(snapshot().lines)); self.assertEqual(request['request_id'], response['request_id'])
        self.assertTrue(0 < request['budget_seconds'] <= 2.85); self.assertEqual(response['budget_seconds'], request['budget_seconds'])

    def test_every_server_refusal_cause_passes_through(self):
        for cause in live.SERVER_CAUSES:
            with FakeServer(send(lambda r, c=cause: refused(r, c))) as server:
                response = self.call(server)
            self.assertEqual((response['status'], response['cause'], response['client_refused']), ('REFUSED', cause, False))
            self.assertEqual(server.connections, 1)

    def test_slow_response_times_out_before_the_deadline_without_retry(self):
        release = threading.Event()
        def slow(request, conn):
            release.wait(5); conn.sendall(wire.encode(complete(request)))
        with FakeServer(slow) as server:
            started = live.clock(); response = self.call(server, deadline=1., reserve=.3, started=started)
            elapsed = live.clock()-started; release.set()
            self.assertEqual((response['status'], response['cause']), ('REFUSED', 'timeout'))
            self.assertLess(elapsed, 1.); self.assertEqual(server.connections, 1)

    def test_slow_drip_response_times_out_before_the_deadline(self):
        release = threading.Event()
        def drip(request, conn):
            blob = wire.encode(complete(request)); conn.sendall(blob[:6])
            for byte in blob[6:]:
                if release.wait(.05): return
                conn.sendall(bytes([byte]))
        with FakeServer(drip) as server:
            started = live.clock(); response = self.call(server, deadline=1., reserve=.3, started=started)
            elapsed = live.clock()-started; release.set()
        self.assertEqual((response['status'], response['cause']), ('REFUSED', 'timeout')); self.assertLess(elapsed, 1.)

    def test_connection_refused(self):
        probe = socket.create_server(('127.0.0.1', 0)); port = probe.getsockname()[1]; probe.close()
        client = live.Client('127.0.0.1', port, rung='DN2W8')
        response = client.request(snapshot(), decision_index=0, started=live.clock())
        self.assertEqual((response['status'], response['cause'], response['client_refused']), ('REFUSED', 'connection_error', True))

    def test_wrong_binding_or_malformed_response_is_server_error(self):
        bad = dict(rung=lambda r: dict(complete(r), rung='DN2W16'),
                   request_id=lambda r: dict(complete(r), request_id='0'*32),
                   schema=lambda r: dict(complete(r), schema='other'),
                   status=lambda r: dict(complete(r), status='DONE'),
                   cause=lambda r: dict(refused(r, 'timeout')),
                   complete_cause=lambda r: dict(complete(r), cause='server_error'),
                   not_dict=lambda r: [complete(r)])
        for name, builder in bad.items():
            with self.subTest(name), FakeServer(send(builder)) as server:
                response = self.call(server)
                self.assertEqual((response['status'], response['cause']), ('REFUSED', 'server_error'))
                self.assertEqual(server.connections, 1)

    def test_nonfinite_numbers_in_a_schema_valid_response_are_server_errors(self):
        edits = dict(nan_root=lambda t: t.replace('"root_value":0.2', '"root_value":NaN'),
                     infinity_q=lambda t: t.replace('0.25', 'Infinity'),
                     huge_q=lambda t: t.replace('0.25', '1e999'),
                     huge_timing=lambda t: t.replace('"prepare_seconds":0.2', '"prepare_seconds":-1e999'))
        with FakeServer(raw_reply(lambda t: t)) as server:
            self.assertEqual(self.call(server)['status'], 'COMPLETE')
        for name, edit in edits.items():
            with self.subTest(name), FakeServer(raw_reply(edit)) as server:
                response = self.call(server)
                self.assertEqual((response['status'], response['cause']), ('REFUSED', 'server_error'))
                live.encoded(response)

    def test_closed_garbage_and_deep_responses_are_server_errors(self):
        deep = b'['*200000+b']'*200000
        replies = dict(closed=lambda r, c: None, zero_length=lambda r, c: c.sendall(struct.pack('!I', 0)),
                       truncated=lambda r, c: c.sendall(struct.pack('!I', 100)+b'{}'),
                       not_json=lambda r, c: c.sendall(struct.pack('!I', 3)+b'abc'),
                       deep=lambda r, c: c.sendall(struct.pack('!I', len(deep))+deep))
        for name, reply in replies.items():
            with self.subTest(name), FakeServer(reply) as server:
                response = self.call(server)
                self.assertEqual((response['status'], response['cause']), ('REFUSED', 'server_error'))
                live.encoded(response)

    def test_no_budget_left_never_connects(self):
        with FakeServer(send(complete)) as server:
            response = self.call(server, started=live.clock()-3.)
        self.assertEqual((response['status'], response['cause']), ('REFUSED', 'timeout')); self.assertEqual(server.connections, 0)
        self.assertLessEqual(response['budget_seconds'], 0)

    def test_budget_arithmetic(self):
        client = live.Client('127.0.0.1', 1, rung='DN2W8')
        self.assertEqual((client.deadline, client.reserve), (3., .15))
        with patch.object(live, 'clock', return_value=100.4):
            self.assertAlmostEqual(client.budget(100.), 2.45, places=12)

    def test_constructor_refuses_codex_rungs_and_unbounded_deadlines(self):
        for kwargs in (dict(rung='R1'), dict(rung='RAW'), dict(rung='DN2W8', deadline_seconds=0),
                       dict(rung='DN2W8', deadline_seconds=float('inf')), dict(rung='DN2W8', reserve_seconds=3.)):
            with self.assertRaises(ValueError): live.Client('127.0.0.1', 1, **kwargs)

    def test_host_is_resolved_once_at_construction(self):
        real = socket.getaddrinfo; calls = []
        def counted(*a, **k): calls.append(a); return real(*a, **k)
        with FakeServer(send(complete)) as server:
            with patch.object(live.socket, 'getaddrinfo', counted):
                client = live.Client('localhost', server.port, rung='DN2W8')
            self.assertEqual(len(calls), 1)
            with patch.object(live.socket, 'getaddrinfo', side_effect=AssertionError('no lookup per request')):
                responses = [client.request(snapshot(), decision_index=i, started=live.clock()) for i in range(2)]
        self.assertEqual([r['status'] for r in responses], ['COMPLETE']*2)


class FakeSocket:
    """Deterministic socket on a fake clock: records every timeout; recv drips one byte per 1/8 s."""
    made = []
    def __init__(self, family, kind, now, *, connect_error=None):
        self.now, self.connect_error, self.timeouts, self.reply = now, connect_error, [], b''
        FakeSocket.made.append(self)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def settimeout(self, value): self.timeouts.append(value)
    def connect(self, address):
        self.address, self.connect_timeout = address, self.timeouts[-1]
        if self.connect_error: raise self.connect_error
    def sendall(self, data): self.reply = wire.encode(complete(json.loads(data[4:])))
    def recv(self, n):
        self.now[0] += .125; piece, self.reply = self.reply[:1], self.reply[1:]
        return piece


class DeterministicDeadlineTests(unittest.TestCase):
    def setUp(self): FakeSocket.made.clear(); self.now = [100.]

    def request(self, client, *, started, **socket_kwargs):
        with patch.object(live, 'clock', lambda: self.now[0]), \
                patch.object(live.socket, 'socket', lambda f, k: FakeSocket(f, k, self.now, **socket_kwargs)):
            return client.request(snapshot(), decision_index=0, started=started)

    def test_every_read_gets_the_remaining_budget(self):
        client = live.Client('127.0.0.1', 7000, rung='DN2W8', deadline_seconds=1., reserve_seconds=.5)
        response = self.request(client, started=100.)
        self.assertEqual((response['status'], response['cause']), ('REFUSED', 'timeout'))
        stream, = FakeSocket.made; reads = stream.timeouts[2:]
        self.assertEqual(stream.timeouts[:2], [.5, .5]); self.assertEqual(reads, [.5, .375, .25, .125])
        self.assertEqual(self.now[0], 100.5)

    def test_connect_gets_the_remaining_budget_not_the_deadline(self):
        client = live.Client('127.0.0.1', 7000, rung='DN2W8')
        self.now[0] = 102.
        response = self.request(client, started=100., connect_error=TimeoutError('fixture connect timeout'))
        self.assertEqual(response['cause'], 'timeout')
        stream, = FakeSocket.made; self.assertAlmostEqual(stream.connect_timeout, .85, places=9)

    def test_only_the_first_resolved_address_is_ever_tried(self):
        def two(host, port, *a, **k):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.0.2.1', port)),
                    (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.0.2.2', port))]
        with patch.object(live.socket, 'getaddrinfo', two):
            client = live.Client('svc', 7000, rung='DN2W8', deadline_seconds=1., reserve_seconds=.5)
        response = self.request(client, started=100., connect_error=TimeoutError('fixture blackhole'))
        self.assertEqual(response['cause'], 'timeout'); stream, = FakeSocket.made
        self.assertEqual(stream.address, ('192.0.2.1', 7000))

    def test_the_first_ipv4_address_is_preferred(self):
        def both(host, port, *a, **k):
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('::1', port, 0, 0)),
                    (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', port))]
        with patch.object(live.socket, 'getaddrinfo', both): client = live.Client('localhost', 7000, rung='DN2W8')
        self.assertEqual((client.family, client.address), (socket.AF_INET, ('127.0.0.1', 7000)))
        with patch.object(live.socket, 'getaddrinfo', lambda *a, **k: both(*a, **k)[:1]): client = live.Client('h', 7000, rung='DN2W8')
        self.assertEqual((client.family, client.address), (socket.AF_INET6, ('::1', 7000, 0, 0)))


class Backend(JaxBackend):
    def __init__(self, c, **kw): OwnedProvider.__init__(self, c)
    ingest = fixture.Backend.ingest
    encode = fixture.Backend.encode
    close = OwnedProvider.close


def raw_factory(calls, *, action=0, before=None, backend=Backend):
    def select(observation, *, rng):
        if before is not None: before()
        calls.append('greedy')
        legal = observation.legal_action_mask
        return PolicyDecision(action if legal[action] else legal.index(True), 'champion-raw', action_probability=1., value_estimate=.1)
    return OwningActorFactory('jax-owning', ActorContract.champion(), lambda c, **kw: backend(c),
        lambda c, **kw: RawPolicy(c, select, lambda: None))


def fake_client(calls, responder, rung='DN2W8'):
    client = object.__new__(live.Client); client.rung = rung; client.deadline = 3.; client.reserve = .15
    def request(snapshot, *, decision_index, started):
        calls.append('search')
        response = responder(dict(request_id='f'*32, rung=rung, decision_index=decision_index))
        return dict(response, budget_seconds=client.budget(started), client_wall_seconds=.01)
    client.request = request
    return client


CELL = dict(ordinal=1, pair_ordinal=0, offset=0, seat=0, arm='DN2W8', seed=10000, fp_seed=20000)


def play(value, turns=1, lines=None):
    a = value.actor((None, None, None)); a.begin_game(battle_id='literal', owner='p1', seed=9, stream_id=1)
    decisions = []
    for turn, line in enumerate(lines or [fixture.request()]*turns, 1):
        fixture.feed(a, (f'|turn|{turn}', line)); s = a.snapshot_boundary()
        decisions.append(a.decide(s, rng=random.Random(4)))
    a.close_game()
    return decisions


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.path = Path(self.directory.name)/'telemetry'/'1.jsonl'
        self.patch = patch.object(live, 'require_actor_factory', side_effect=lambda c, f: f); self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.directory.cleanup()

    def rows(self):
        return [json.loads(line) for line in self.path.read_bytes().splitlines()]

    def run_arm(self, responder, *, arm='DN2W8', turns=1, greedy=0, lines=None):
        calls = []
        client = None if arm == 'RAW' else fake_client(calls, responder)
        value = live.factory(NS(), raw_factory=raw_factory(calls, action=greedy), client=client,
                             telemetry_path=self.path, arm=arm, cell=CELL)
        return play(value, turns, lines), calls

    def fresh(self, name):
        self.path = Path(self.directory.name)/'cases'/f'{name}.jsonl'

    def test_the_search_request_leaves_before_greedy_and_a_complete_legal_search_is_played(self):
        sent = threading.Event()
        def responder(r): sent.set(); return complete(r, action=4, q=.25)
        calls = []
        value = live.factory(NS(), raw_factory=raw_factory(calls, before=lambda: self.assertTrue(sent.wait(5))), client=fake_client(calls, responder),
                             telemetry_path=self.path, arm='DN2W8', cell=CELL)
        (decision,) = play(value)   # greedy waits for the request: a serial greedy-then-search actor never gets here
        self.assertEqual(calls, ['search', 'greedy'])
        self.assertEqual((decision.action_index, decision.policy_id, decision.value_estimate), (4, 'champion-depthn-DN2W8', .25))
        row, = self.rows()
        self.assertEqual((row['source'], row['cause'], row['played_action_index'], row['search_differs']), ('search', None, 4, True))
        self.assertEqual(row['greedy']['action_index'], 0); self.assertEqual(row['request_kind'], 'move')
        self.assertGreaterEqual(row['encode_seconds'], 0)

    def test_every_refusal_cause_plays_the_greedy_decision(self):
        for cause in live.SERVER_CAUSES+('timeout', 'connection_error'):
            with self.subTest(cause):
                self.fresh(cause)
                (decision,), calls = self.run_arm(lambda r, c=cause: refused(r, c))
                self.assertEqual(sorted(calls), ['greedy', 'search']); self.assertEqual(decision.action_index, 0)
                self.assertEqual(decision.policy_id, 'champion-raw')
                self.assertEqual(decision.metadata['depthn_live'], dict(arm='DN2W8', source='greedy_fallback', cause=cause, request_id='f'*32))
                row, = self.rows()
                self.assertEqual((row['source'], row['cause'], row['played_action_index'], row['search_differs']),
                                 ('greedy_fallback', cause, 0, None))

    def test_illegal_or_out_of_range_searched_action_falls_back(self):
        for action in (1, 2, 8, 9, -1, True, None, 4.):
            with self.subTest(action):
                self.fresh(f'illegal-{action}')
                (decision,), _ = self.run_arm(lambda r, a=action: complete(r, action=a))
                self.assertEqual((decision.action_index, self.rows()[0]['cause']), (0, 'illegal_action'))

    def test_forced_wins_and_losses_are_played(self):
        for q in (1., -1., 1+4*2**-53, -(1+4*2**-53), 1+1e-9, 0):
            with self.subTest(q):
                self.fresh(f'q{q!r}')
                (decision,), _ = self.run_arm(lambda r, v=q: complete(r, q=v))
                self.assertEqual((decision.action_index, decision.value_estimate, self.rows()[0]['source']), (4, float(q), 'search'))

    def test_bad_q_falls_back(self):
        def short(r): return dict(complete(r), q_values=[.1]*8)
        def missing(r): return dict(complete(r), q_values=None)
        cases = dict(none=lambda r: complete(r, q=None), over=lambda r: complete(r, q=1.001), under=lambda r: complete(r, q=-1.001),
                     edge=lambda r: complete(r, q=1+1e-8), text=lambda r: complete(r, q='0.1'), boolean=lambda r: complete(r, q=True),
                     short=short, missing=missing)
        for name, builder in cases.items():
            with self.subTest(name):
                self.fresh(f'bad-{name}')
                (decision,), _ = self.run_arm(builder)
                self.assertEqual((decision.action_index, self.rows()[0]['cause']), (0, 'bad_q'))
        observation = NS(legal_action_mask=(True,)+(False,)*8)
        for value in (float('nan'), float('inf'), -float('inf'), 1.001):
            q = [value]+[None]*8
            self.assertEqual(live.searched_action(dict(status='COMPLETE', action_index=0, q_values=q), observation), (None, 'bad_q'))
        for value in (1., -1., 1+4*2**-53, 1, -1, 0):
            q = [value]+[None]*8
            self.assertEqual(live.searched_action(dict(status='COMPLETE', action_index=0, q_values=q), observation), (0, None))
        for value in (10**400, -10**400, 2, True, False):
            q = [value]+[None]*8
            self.assertEqual(live.searched_action(dict(status='COMPLETE', action_index=0, q_values=q), observation), (None, 'bad_q'))

    def test_huge_json_integers_never_raise(self):
        cases = dict(q=(lambda t: t.replace('0.25', '1'+'0'*400), (0, 'greedy_fallback', 'bad_q')),
                     root_value=(lambda t: t.replace('"root_value":0.2', '"root_value":1'+'0'*400), (4, 'search', None)))
        for name, (edit, expected) in cases.items():
            with self.subTest(name), FakeServer(raw_reply(edit)) as server:
                self.fresh(f'bigint-{name}')
                client = live.Client('127.0.0.1', server.port, rung='DN2W8')
                value = live.factory(NS(), raw_factory=raw_factory([]), client=client, telemetry_path=self.path, arm='DN2W8')
                decision, = play(value); row = self.rows()[0]
                self.assertEqual((decision.action_index, row['source'], row['cause']), expected)

    def test_a_response_at_the_row_depth_edge_is_refused_by_the_client(self):
        edge = decode_edge(); results = {}
        for depth in range(edge-8, edge+3):
            with FakeServer(raw_reply(lambda t, d=depth: t[:-1]+',"zz":'+'['*d+']'*d+'}')) as server:
                client = live.Client('127.0.0.1', server.port, rung='DN2W8')
                response = client.request(snapshot(), decision_index=0, started=live.clock()); results[depth] = response
            if response['status'] == 'COMPLETE': live.encoded(dict(search=response))
        self.assertTrue(any(r['cause'] == 'server_error' and r['client_refused'] for r in results.values()))
        self.assertTrue(any(r['status'] == 'COMPLETE' for r in results.values()))

    def test_select_never_raises_at_the_depth_edge(self):
        edge = decode_edge(); sources = set()
        for depth in range(edge-8, edge+3):
            with FakeServer(raw_reply(lambda t, d=depth: t[:-1]+',"zz":'+'['*d+']'*d+'}')) as server:
                self.fresh(f'edge-{depth}')
                client = live.Client('127.0.0.1', server.port, rung='DN2W8')
                value = live.factory(NS(), raw_factory=raw_factory([]), client=client, telemetry_path=self.path, arm='DN2W8')
                decision, = play(value); row = self.rows()[0]
                sources.add((decision.action_index, row['source'], row['cause']))
        self.assertTrue(sources <= {(4, 'search', None), (0, 'greedy_fallback', 'server_error')}); self.assertEqual(len(sources), 2)

    def test_a_row_that_cannot_be_written_records_server_error_and_plays_greedy(self):
        deep = nested(30000)
        (decision,), calls = self.run_arm(lambda r: dict(complete(r), zz=deep))
        row, = self.rows()
        self.assertEqual(sorted(calls), ['greedy', 'search']); self.assertEqual(decision.action_index, 0)
        self.assertEqual((row['source'], row['cause'], row['played_action_index'], row['search_differs']), ('greedy_fallback', 'server_error', 0, None))
        self.assertEqual((row['search']['status'], row['search']['cause'], row['search']['request_id']), ('REFUSED', 'server_error', 'f'*32))
        self.assertTrue(row['search']['refusal'].startswith('telemetry row: RecursionError'))
        self.assertEqual(row['search']['client_wall_seconds'], .01); self.assertEqual(row['archive_source'], 'pinned')
        self.assertEqual(decision.metadata['depthn_live'], dict(arm='DN2W8', source='greedy_fallback', cause='server_error', request_id='f'*32))

    def test_class_refused_is_a_counted_greedy_fallback(self):
        self.assertIn('class_refused', live.SERVER_CAUSES)
        (decision,), calls = self.run_arm(lambda r: refused(r, 'class_refused'))
        row, = self.rows()
        self.assertEqual((decision.action_index, row['source'], row['cause'], sorted(calls)), (0, 'greedy_fallback', 'class_refused', ['greedy', 'search']))

    def test_rows_carry_turn_and_champion_archive(self):
        with FakeServer(send(lambda r: dict(complete(r), archive_sha256='a'*64))) as server:
            client = live.Client('127.0.0.1', server.port, rung='DN2W8')
            value = live.factory(NS(), raw_factory=raw_factory([]), client=client, telemetry_path=self.path, arm='DN2W8',
                                 archive_sha256='b'*64)
            play(value, lines=[fixture.request(), request_line(trapped=True), fixture.request()])
        rows = self.rows()
        self.assertEqual([(r['turn'], r['decision_index'], r['source'], r['archive_sha256'], r['archive_source']) for r in rows],
                         [(1, 0, 'search', 'a'*64, 'server'), (2, 1, 'single_legal', 'b'*64, 'pinned'), (3, 2, 'search', 'a'*64, 'server')])
        self.fresh('raw-archive')
        self.run_arm(None, arm='RAW'); row, = self.rows()
        self.assertEqual((row['archive_sha256'], row['archive_source']), (live.CHAMPION_ARCHIVE_SHA256, 'pinned'))
        self.fresh('timeout-archive')
        self.run_arm(lambda r: dict(refused(r, 'timeout'), client_refused=True)); row, = self.rows()
        self.assertEqual((row['archive_sha256'], row['archive_source']), (live.CHAMPION_ARCHIVE_SHA256, 'pinned'))
        for bad in ('B'*64, 'b'*63, None):
            with self.assertRaises(ValueError):
                live.factory(NS(), raw_factory=raw_factory([]), client=None, telemetry_path=Path(self.directory.name)/'x.jsonl', arm='RAW',
                             archive_sha256=bad)

    def test_battle_turn_from_the_owner_lines(self):
        self.assertEqual(live.battle_turn(('|start', '|turn|1', '|move|x', '|turn|12', '|request|{}')), 12)
        for lines in ((), ('|start', '|request|{}'), ('|turn|x',), ('|turn|²',), ('|turn|',)):
            self.assertIsNone(live.battle_turn(lines))

    def test_single_legal_decision_is_never_searched(self):
        lines = [request_line(trapped=True), fixture.request(), request_line(members=2, force=True)]
        decisions, calls = self.run_arm(lambda r: complete(r), lines=lines)
        self.assertEqual(calls[:1] + sorted(calls[1:3]) + calls[3:], ['greedy', 'greedy', 'search', 'greedy'])   # one decision at a time; within one, either order
        rows = self.rows()
        self.assertEqual([r['decision_index'] for r in rows], [0, 1, 2]); self.assertEqual([d.action_index for d in decisions], [0, 4, 4])
        self.assertEqual([(r['source'], r['request_kind'], r['search'] is None, r['cause']) for r in rows],
                         [('single_legal', 'forced', True, None), ('search', 'move', False, None), ('single_legal', 'forced', True, None)])
        self.assertEqual(decisions[0].metadata['depthn_live'], dict(arm='DN2W8', source='single_legal', cause=None, request_id=None))

    def test_raw_arm_never_contacts_a_client_and_keeps_the_row_shape(self):
        with patch.object(live.socket, 'socket', side_effect=AssertionError('RAW arm must not connect')):
            decisions, calls = self.run_arm(None, arm='RAW', lines=[fixture.request(), request_line(trapped=True)])
        self.assertEqual(calls, ['greedy', 'greedy']); self.assertEqual([d.action_index for d in decisions], [0, 0])
        rows = self.rows(); self.assertEqual([r['decision_index'] for r in rows], [0, 1])
        for row in rows:
            self.assertEqual((row['arm'], row['policy_id'], row['source'], row['cause'], row['search'], row['search_differs']),
                             ('RAW', 'champion-raw-ours', 'raw_arm', None, None, None))
        self.assertEqual([r['request_kind'] for r in rows], ['move', 'forced'])
        self.assertEqual({(r['deadline_seconds'], r['reserve_seconds']) for r in rows}, {(None, None)})
        searched = Path(self.directory.name)/'searched.jsonl'; self.path, raw_path = searched, self.path
        self.run_arm(lambda r: complete(r))
        self.assertEqual(set(json.loads(raw_path.read_bytes().splitlines()[0])), set(self.rows()[0]))

    def test_arm_client_binding_refusals(self):
        calls = []; raw = raw_factory(calls)
        for arm, client in (('RAW', fake_client(calls, complete)), ('DN2W8', None), ('DN2W16', fake_client(calls, complete)),
                            ('R1', None), ('DN2W8', object())):
            with self.subTest(arm), self.assertRaises(ValueError):
                live.factory(NS(), raw_factory=raw, client=client, telemetry_path=self.path, arm=arm)
        with self.assertRaises(ValueError):
            live.factory(NS(), raw_factory=object(), client=None, telemetry_path=self.path, arm='RAW')

    def test_telemetry_rows_are_complete_fsynced_and_fresh(self):
        with patch.object(live.os, 'fsync', wraps=os.fsync) as fsync:
            decisions, _ = self.run_arm(lambda r: complete(r, action=0, q=-.3), turns=3)
        self.assertEqual(fsync.call_count, 3)
        rows = self.rows(); self.assertEqual([r['decision_index'] for r in rows], [0, 1, 2])
        for row in rows:
            self.assertEqual(set(row), ROW_KEYS)
            self.assertEqual((row['schema'], row['cell'], row['battle_id'], row['owner']), (live.ROW_SCHEMA, CELL, 'literal', 'p1'))
            self.assertEqual(set(row['greedy']), {'action_index','policy_id','action_probability','value_estimate','seconds'})
            self.assertEqual(len(row['prefix_sha256']), 64); self.assertFalse(row['search_differs'])
            self.assertEqual((row['archive_sha256'], row['archive_source']), (live.CHAMPION_ARCHIVE_SHA256, 'pinned'))
        self.assertEqual([r['turn'] for r in rows], [1, 2, 3])
        self.assertEqual({(r['deadline_seconds'], r['reserve_seconds']) for r in rows}, {(3., .15)})
        for row in rows:
            self.assertEqual(row['search']['timings']['prepare_seconds'], .2)
            self.assertGreaterEqual(row['select_seconds'], row['greedy']['seconds']+row['encode_seconds']-1e-9)
        with self.assertRaises(FileExistsError):
            live.factory(NS(), raw_factory=raw_factory([]), client=None, telemetry_path=self.path, arm='RAW')

    def test_budget_starts_at_encode_and_counts_the_reserve_but_not_greedy(self):
        now = [100.]
        class SlowBackend(Backend):
            def encode(self, snapshot): now[0] += .3; return Backend.encode(self, snapshot)
        def greedy_cost():   # greedy runs while the request is out: its 0.4 s is not taken from search's budget
            deadline = time.monotonic() + 5
            while not server.requests and time.monotonic() < deadline: time.sleep(.001)
            now[0] += .4
        with FakeServer(send(complete)) as server, patch.object(live, 'clock', lambda: now[0]):
            client = live.Client('127.0.0.1', server.port, rung='DN2W8')
            value = live.factory(NS(), raw_factory=raw_factory([], before=greedy_cost, backend=SlowBackend), client=client,
                                 telemetry_path=self.path, arm='DN2W8', cell=CELL)
            decision, = play(value)
        self.assertAlmostEqual(server.requests[0]['budget_seconds'], 2.55, places=12)
        row, = self.rows(); self.assertAlmostEqual(row['encode_seconds'], .3, places=12)
        self.assertAlmostEqual(row['greedy']['seconds'], .4, places=12); self.assertAlmostEqual(row['select_seconds'], .7, places=12)
        self.assertEqual((decision.action_index, row['source']), (4, 'search'))

    def test_real_client_timeout_returns_greedy_before_the_deadline(self):
        release = threading.Event()
        with FakeServer(lambda request, conn: release.wait(5)) as server:
            client = live.Client('127.0.0.1', server.port, rung='DN2W8', deadline_seconds=1., reserve_seconds=.3)
            value = live.factory(NS(), raw_factory=raw_factory([]), client=client, telemetry_path=self.path, arm='DN2W8')
            decision, = play(value); release.set()
        row, = self.rows()
        self.assertEqual((decision.action_index, row['cause']), (0, 'timeout')); self.assertLess(row['select_seconds'], 1.)
        self.assertIsNone(row['cell']); self.assertEqual((row['deadline_seconds'], row['reserve_seconds']), (1., .3))

    def test_select_never_raises_on_server_content(self):
        deep = b'['*200000+b']'*200000
        replies = dict(deep=lambda r, c: c.sendall(struct.pack('!I', len(deep))+deep),
                       huge_root_value=raw_reply(lambda t: t.replace('"root_value":0.2', '"root_value":1e999')),
                       nan_q=raw_reply(lambda t: t.replace('0.25', 'NaN')))
        for name, reply in replies.items():
            with self.subTest(name), FakeServer(reply) as server:
                self.fresh(f'content-{name}')
                client = live.Client('127.0.0.1', server.port, rung='DN2W8')
                value = live.factory(NS(), raw_factory=raw_factory([]), client=client, telemetry_path=self.path, arm='DN2W8')
                decision, = play(value)
                self.assertEqual((decision.action_index, self.rows()[0]['cause']), (0, 'server_error'))

    def test_stale_missing_or_foreign_snapshot_is_refused(self):
        calls = []
        value = live.factory(NS(), raw_factory=raw_factory(calls), client=None, telemetry_path=self.path, arm='RAW')
        a = value.actor((None, None, None)); a.begin_game(battle_id='literal', owner='p1', seed=9, stream_id=1)
        fixture.feed(a, ('|turn|1', fixture.request())); s = a.snapshot_boundary()
        policy = a._policy; mask = (True,)+(False,)*3+(True,)+(False,)*4
        with self.assertRaisesRegex(ValueError, 'owning root'):
            policy.select_action(NS(metadata={'owner_prefix_sha256': s.prefix_sha256}, legal_action_mask=mask), rng=random.Random(1))
        a._backend.encode(s)
        with self.assertRaisesRegex(ValueError, 'owning root'):
            policy.select_action(NS(metadata={'owner_prefix_sha256': 'e'*64}, legal_action_mask=mask), rng=random.Random(1))
        self.assertEqual(calls, []); self.assertEqual(self.path.read_bytes(), b'')


class RequestKindTests(unittest.TestCase):
    def test_kinds_from_the_legal_mask(self):
        mask = lambda *legal: tuple(i in legal for i in range(9))
        self.assertEqual(live.request_kind(mask(0, 4)), 'move'); self.assertEqual(live.request_kind(mask(0, 1)), 'move')
        self.assertEqual(live.request_kind(mask(0)), 'forced'); self.assertEqual(live.request_kind(mask(5)), 'forced')
        self.assertEqual(live.request_kind(mask(4, 5)), 'switch')
        for bad in (mask(), (True,)*8):
            with self.assertRaises(ValueError): live.request_kind(bad)

    def test_kinds_from_actual_owning_requests(self):
        with TemporaryDirectory() as directory, patch.object(live, 'require_actor_factory', side_effect=lambda c, f: f):
            path = Path(directory)/'t.jsonl'
            value = live.factory(NS(), raw_factory=raw_factory([], action=4), client=None, telemetry_path=path, arm='RAW')
            play(value, lines=[fixture.request(), request_line(trapped=True), request_line(members=3, force=True),
                               request_line(members=2, force=True)])
            rows = [json.loads(x) for x in path.read_bytes().splitlines()]
        self.assertEqual([r['request_kind'] for r in rows], ['move', 'forced', 'switch', 'forced'])


class RuntimeTests(unittest.TestCase):
    def test_runtime_runs_one_declared_cell_through_the_original_bridge(self):
        from pokezero import foulplay_bridge as bridge
        calls = []; raw = raw_factory(calls)
        original = NS(runtime=('model','result','env','spec','vocab','dex','masks','d'*64), prepared=[])
        original.prepare = lambda config: (original.prepared.append(config), raw)[1]
        cell = dict(CELL, seat=1)
        config = NS(games=1, seed_start=10000, foulplay_random_seed=20000, pokezero_player='p2',
                    max_decision_rounds=500, format_id='gen3randombattle')
        with TemporaryDirectory() as directory, patch.object(live, 'require_actor_factory', side_effect=lambda c, f: f),\
                patch.object(bridge, '_build_policy', return_value='policy') as build,\
                patch.object(bridge, '_run_controlled_foulplay_games', new=AsyncMock(return_value='result')) as run:
            path = Path(directory)/'t.jsonl'
            runtime = live.Runtime(client=None, telemetry_path=path, arm='RAW', cell=cell, original=original)
            self.assertEqual(asyncio.run(runtime.run(config)), 'result')
            kwargs = run.call_args.kwargs
            self.assertIs(kwargs['owning_actor_factory'], runtime.factory); self.assertEqual(kwargs['policy_id'], 'champion-raw-ours')
            self.assertEqual(kwargs['actor_runtime'], ('model', 'result', 'env')); self.assertEqual(kwargs['checkpoint_sha256'], 'd'*64)
            self.assertEqual(build.call_args.kwargs['policy_id'], 'champion-raw-ours'); self.assertTrue(path.exists())
            with self.assertRaisesRegex(ValueError, 'one game'): asyncio.run(runtime.run(config))
            for change in (dict(seed_start=10001), dict(foulplay_random_seed=1), dict(pokezero_player='p1'), dict(games=2)):
                fresh = live.Runtime(client=None, telemetry_path=Path(directory)/'u.jsonl', arm='RAW', cell=cell, original=original)
                with self.subTest(change), self.assertRaisesRegex(ValueError, 'declared cell'):
                    asyncio.run(fresh.run(NS(**{**vars(config), **change})))
        for kwargs in (dict(client=None, arm='DN2W8'), dict(client=object(), arm='RAW'), dict(client=None, arm='R1')):
            with self.assertRaises(ValueError): live.Runtime(telemetry_path='x', cell=cell, original=original, **kwargs)
        with self.assertRaises(ValueError): live.Runtime(client=None, telemetry_path='x', arm='RAW', cell=dict(cell, seat=True), original=original)

    def test_actor_module_does_not_import_the_gpu_stack(self):
        import subprocess
        code = ('import sys,pokezero.depthn_live_actor;'
                'print(any(k=="gpu_engine" or k.startswith("gpu_engine.") or k in ("jax","torch") for k in sys.modules))')
        out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=True,
                             env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
        self.assertEqual(out.stdout.strip(), 'False')


if __name__ == '__main__': unittest.main()
