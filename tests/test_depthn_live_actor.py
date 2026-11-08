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
        self.stop.set(); self.thread.join(5); self.listener.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def send(builder):
    return lambda request, conn: conn.sendall(wire.encode(builder(request)))


def snapshot():
    line = fixture.request()
    return NS(battle_id='literal', owner='p1', stream_id=1, generation=2, prefix_sha256='c'*64,
              lines=('|turn|1', line), own_request_line=line)


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
        def slow(request, conn):
            time.sleep(1.5)
            conn.sendall(wire.encode(complete(request)))
        with FakeServer(slow) as server:
            started = live.clock(); response = self.call(server, deadline=.6, reserve=.1, started=started)
            elapsed = live.clock()-started
            self.assertEqual((response['status'], response['cause']), ('REFUSED', 'timeout'))
            self.assertLess(elapsed, .6); self.assertGreater(elapsed, .4)
            self.assertEqual(server.connections, 1)

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

    def test_closed_garbage_and_nonfinite_responses_are_server_errors(self):
        replies = dict(closed=lambda r, c: None, zero_length=lambda r, c: c.sendall(struct.pack('!I', 0)),
                       truncated=lambda r, c: c.sendall(struct.pack('!I', 100)+b'{}'),
                       nan=lambda r, c: c.sendall(struct.pack('!I', 9)+b'{"q":NaN}'),
                       not_json=lambda r, c: c.sendall(struct.pack('!I', 3)+b'abc'))
        for name, reply in replies.items():
            with self.subTest(name), FakeServer(reply) as server:
                response = self.call(server)
                self.assertEqual((response['status'], response['cause']), ('REFUSED', 'server_error'))

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


class Backend(JaxBackend):
    def __init__(self, c, **kw): OwnedProvider.__init__(self, c)
    ingest = fixture.Backend.ingest
    encode = fixture.Backend.encode
    close = OwnedProvider.close


def raw_factory(calls, *, action=0, before=None):
    def select(observation, *, rng):
        if before is not None: before()
        calls.append('greedy')
        return PolicyDecision(action, 'champion-raw', action_probability=1., value_estimate=.1)
    return OwningActorFactory('jax-owning', ActorContract.champion(), lambda c, **kw: Backend(c),
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


def play(value, turns=1):
    a = value.actor((None, None, None)); a.begin_game(battle_id='literal', owner='p1', seed=9, stream_id=1)
    decisions = []
    for turn in range(1, turns+1):
        fixture.feed(a, (f'|turn|{turn}', fixture.request())); s = a.snapshot_boundary()
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

    def run_arm(self, responder, *, arm='DN2W8', turns=1, greedy=0):
        calls = []
        client = None if arm == 'RAW' else fake_client(calls, responder)
        value = live.factory(NS(), raw_factory=raw_factory(calls, action=greedy), client=client,
                             telemetry_path=self.path, arm=arm, cell=CELL)
        return play(value, turns), calls

    def test_greedy_first_then_complete_legal_search_is_played(self):
        (decision,), calls = self.run_arm(lambda r: complete(r, action=4, q=.25))
        self.assertEqual(calls, ['greedy', 'search'])
        self.assertEqual((decision.action_index, decision.policy_id, decision.value_estimate), (4, 'champion-depthn-DN2W8', .25))
        row, = self.rows()
        self.assertEqual((row['source'], row['cause'], row['played_action_index'], row['search_differs']), ('search', None, 4, True))
        self.assertEqual(row['greedy']['action_index'], 0); self.assertEqual(row['request_kind'], 'move')

    def test_every_refusal_cause_plays_the_greedy_decision(self):
        for cause in live.SERVER_CAUSES+('timeout', 'connection_error'):
            with self.subTest(cause):
                self.path = Path(self.directory.name)/'causes'/f'{cause}.jsonl'
                (decision,), calls = self.run_arm(lambda r, c=cause: refused(r, c))
                self.assertEqual(calls, ['greedy', 'search']); self.assertEqual(decision.action_index, 0)
                self.assertEqual(decision.policy_id, 'champion-raw')
                self.assertEqual(decision.metadata['depthn_live'], dict(arm='DN2W8', source='greedy_fallback', cause=cause, request_id='f'*32))
                row, = self.rows()
                self.assertEqual((row['source'], row['cause'], row['played_action_index'], row['search_differs']),
                                 ('greedy_fallback', cause, 0, None))

    def test_illegal_or_out_of_range_searched_action_falls_back(self):
        for action in (1, 2, 8, 9, -1, True, None, 4.):
            with self.subTest(action):
                self.path = Path(self.directory.name)/'illegal'/f'{action}.jsonl'
                (decision,), _ = self.run_arm(lambda r, a=action: complete(r, action=a))
                self.assertEqual((decision.action_index, self.rows()[0]['cause']), (0, 'illegal_action'))

    def test_bad_q_falls_back(self):
        def short(r): return dict(complete(r), q_values=[.1]*8)
        def missing(r): return dict(complete(r), q_values=None)
        cases = dict(none=lambda r: complete(r, q=None), one=lambda r: complete(r, q=1.), minus_one=lambda r: complete(r, q=-1.),
                     big=lambda r: complete(r, q=1.5), text=lambda r: complete(r, q='0.1'), boolean=lambda r: complete(r, q=True),
                     short=short, missing=missing)
        for name, builder in cases.items():
            with self.subTest(name):
                self.path = Path(self.directory.name)/'bad'/f'{name}.jsonl'
                (decision,), _ = self.run_arm(builder)
                self.assertEqual((decision.action_index, self.rows()[0]['cause']), (0, 'bad_q'))
        observation = NS(legal_action_mask=(True,)+(False,)*8)
        for value in (float('nan'), float('inf'), -float('inf')):
            q = [value]+[None]*8
            self.assertEqual(live.searched_action(dict(status='COMPLETE', action_index=0, q_values=q), observation), (None, 'bad_q'))
        self.assertEqual(live.searched_action(dict(status='COMPLETE', action_index=0, q_values=[.9]+[None]*8), observation), (0, None))

    def test_raw_arm_never_contacts_a_client_and_keeps_the_row_shape(self):
        with patch.object(live.socket, 'create_connection', side_effect=AssertionError('RAW arm must not connect')):
            decisions, calls = self.run_arm(None, arm='RAW', turns=2)
        self.assertEqual(calls, ['greedy', 'greedy']); self.assertEqual([d.action_index for d in decisions], [0, 0])
        rows = self.rows(); self.assertEqual([r['decision_index'] for r in rows], [0, 1])
        for row in rows:
            self.assertEqual((row['arm'], row['policy_id'], row['source'], row['cause'], row['search'], row['search_differs']),
                             ('RAW', 'champion-raw-ours', 'raw_arm', None, None, None))
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
            self.assertEqual(set(row), {'schema','arm','policy_id','cell','battle_id','owner','stream_id','generation','decision_index',
                'request_kind','prefix_sha256','greedy','search','played_action_index','source','cause','search_differs','select_seconds'})
            self.assertEqual((row['schema'], row['cell'], row['battle_id'], row['owner']), (live.ROW_SCHEMA, CELL, 'literal', 'p1'))
            self.assertEqual(set(row['greedy']), {'action_index','policy_id','action_probability','value_estimate','policy_distribution','seconds'})
            self.assertEqual(len(row['prefix_sha256']), 64); self.assertFalse(row['search_differs'])
            self.assertEqual(row['search']['timings']['prepare_seconds'], .2); self.assertGreaterEqual(row['select_seconds'], row['greedy']['seconds'])
        with self.assertRaises(FileExistsError):
            live.factory(NS(), raw_factory=raw_factory([]), client=None, telemetry_path=self.path, arm='RAW')

    def test_budget_is_three_seconds_from_select_entry_less_greedy_and_reserve(self):
        now = [100.]
        def greedy_cost(): now[0] += .4
        with FakeServer(send(complete)) as server, patch.object(live, 'clock', lambda: now[0]):
            client = live.Client('127.0.0.1', server.port, rung='DN2W8')
            value = live.factory(NS(), raw_factory=raw_factory([], before=greedy_cost), client=client,
                                 telemetry_path=self.path, arm='DN2W8', cell=CELL)
            decision, = play(value)
        self.assertAlmostEqual(server.requests[0]['budget_seconds'], 2.45, places=12)
        row, = self.rows(); self.assertAlmostEqual(row['greedy']['seconds'], .4, places=12)
        self.assertEqual((decision.action_index, row['source']), (4, 'search'))

    def test_real_client_timeout_returns_greedy_before_the_deadline(self):
        def slow(request, conn): time.sleep(1.5)
        with FakeServer(slow) as server:
            client = live.Client('127.0.0.1', server.port, rung='DN2W8', deadline_seconds=.5, reserve_seconds=.05)
            value = live.factory(NS(), raw_factory=raw_factory([]), client=client, telemetry_path=self.path, arm='DN2W8')
            started = time.monotonic(); decision, = play(value); elapsed = time.monotonic()-started
        row, = self.rows()
        self.assertEqual((decision.action_index, row['cause']), (0, 'timeout')); self.assertLess(row['select_seconds'], .5)
        self.assertLess(elapsed, 1.); self.assertIsNone(row['cell'])

    def test_stale_or_missing_snapshot_is_refused(self):
        calls = []
        value = live.factory(NS(), raw_factory=raw_factory(calls), client=None, telemetry_path=self.path, arm='RAW')
        a = value.actor((None, None, None)); a.begin_game(battle_id='literal', owner='p1', seed=9, stream_id=1)
        fixture.feed(a, ('|turn|1', fixture.request())); s = a.snapshot_boundary()
        policy = a._policy
        with self.assertRaisesRegex(ValueError, 'owning root'):
            policy.select_action(NS(metadata={'owner_prefix_sha256': s.prefix_sha256}), rng=random.Random(1))
        self.assertEqual(calls, [])


class RequestKindTests(unittest.TestCase):
    def line(self, *, members=2, force=False, trapped=False):
        packet = json.loads(fixture.request()[len('|request|'):]); side = packet['side']; key = next(iter(k for k in side if k != 'id'))
        base = side[key][1]
        side[key] = side[key][:1]+[dict(base, ident=base['ident'][:-1]+str(i)) for i in range(1, members)]
        if force: packet.pop('active'); packet['forceSwitch'] = [True]
        if trapped: packet['active'][0]['trapped'] = True
        return '|request|'+json.dumps(packet)

    def test_kinds_from_own_request_line(self):
        self.assertEqual(live.request_kind(self.line(), 'p1'), 'move')
        self.assertEqual(live.request_kind(self.line(trapped=True), 'p1'), 'forced')
        self.assertEqual(live.request_kind(self.line(members=3, force=True), 'p1'), 'switch')
        self.assertEqual(live.request_kind(self.line(members=2, force=True), 'p1'), 'forced')
        with self.assertRaises(ValueError): live.request_kind(self.line(), 'p2')


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
