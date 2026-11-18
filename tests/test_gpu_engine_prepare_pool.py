"""gpu_engine.prepare_pool on the fake engine: shipping (pack/unpack), and real worker processes running Codex's real
ProtocolConstructor.prepare (its heavy collaborators faked, as tests/test_gpu_engine_depthn_live_search's Codex): battle
affinity, results equal to serial prepare, eviction, a crash, a hung prepare, an error, a failed start. CPU only; each worker
is a small process (the fake fixture), a few at a time."""
from dataclasses import replace
import os
import pickle
import time
import types
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import constructor_worlds as cw, depthn_live_search as live, prepare_pool as pp
from gpu_engine import device_public_world_state as mapping
from gpu_engine.phase_d_live_worlds import Refusal
from tests import test_gpu_engine_constructor_worlds as CW
from tests import test_gpu_engine_depthn_live_search as L

FACTORY = 'tests.test_gpu_engine_prepare_pool:fake_constructor'


def outcomes_of(fixture, payload):
    """The Codex harness's outcomes from a payload's picklable specs: ('outcome', i, keywords) -> fixture.made(i, ...), ('refusal',
    kind) -> Refusal(kind)."""
    W = payload['W'];out = {}
    for key, spec in payload['outcomes']:
        if spec[0] == 'outcome':
            out[tuple(key)] = fixture.made(spec[1], 0, fixture.modes(spec[1], W), **dict(spec[2]))
        else:
            out[tuple(key)] = Refusal(spec[1])
    return out


def fake_constructor(payload):
    """A prepare worker's constructor on the fake engine (FACTORY): Codex's ProtocolConstructor with the Codex harness's fakes
    held for the worker's life and the carry completion kept as built (LiveSearchTests.setUp's). payload: W, outcomes (specs),
    and per (battle, decision) `slow` seconds, `crash` (the process exits) or `fail` (RuntimeError) for testing; `init_fail`."""
    if payload.get('init_fail'):
        raise RuntimeError('this worker cannot start')
    CW.ConstructorWorldsTests.setUpClass();fixture = CW.ConstructorWorldsTests('test_both_root_moves_are_reported_and_can_differ')
    L.Codex(outcomes_of(fixture, payload)).__enter__()
    patch.object(cw, 'world_completer', return_value=lambda world, owner, **kw: world).start()
    constructor = fixture.constructor(payload['W'], saved_root_admission=False, rung=f'DN2W{payload["W"]}')
    slow = {tuple(k): s for k, s in payload.get('slow', ())};crash = {tuple(k) for k in payload.get('crash', ())}
    fail = {tuple(k) for k in payload.get('fail', ())};prepare = constructor.prepare

    def wrapped(request, index):
        key = (request.get('battle_id'), request.get('decision_index'))
        if key in crash:
            os._exit(3)
        if key in fail:
            raise RuntimeError('boom')
        if key in slow:
            time.sleep(slow[key])
        return prepare(request, index)
    constructor.prepare = wrapped
    return constructor


def protocol(request):
    return {k: request[k] for k in live.PROTOCOL_KEYS}


class Fixture:
    """The fake fixture in this process (serial prepare for the comparisons)."""
    @classmethod
    def setUpClass(cls):
        CW.ConstructorWorldsTests.setUpClass();cls.cw = CW.ConstructorWorldsTests('test_both_root_moves_are_reported_and_can_differ')

    def setUp(self):
        keep = patch.object(cw, 'world_completer', return_value=lambda world, owner, **kw: world);keep.start();self.addCleanup(keep.stop)

    def serial(self, payload, requests):
        """prepare_roots in this process, one request at a time in order, on one constructor (the serial server's way), and
        each item's classification."""
        constructor = self.cw.constructor(payload['W'], saved_root_admission=False, rung=f'DN2W{payload["W"]}');out = []
        with L.Codex(outcomes_of(self.cw, payload)):
            for r in requests:
                p = protocol(r);item = cw.prepare_roots(constructor, [p])[0]
                out.append((item, live.classify_prepared(constructor, p, item)))
        return out


def labels(x):
    return {k: v for k, v in x.items() if not k.endswith('_seconds')}


class ShippingTests(Fixture, unittest.TestCase):
    def test_a_prepared_root_ships_exactly(self):
        payload = dict(W=8, outcomes=[(('a', 0), ('outcome', 0, ())), (('a', 1), ('outcome', 1, (('refused', (2,)),))),
                                      (('b', 0), ('refusal', 'PUBLIC_ROOT_REFUSED'))])
        items = [item for item, _ in self.serial(payload, [L.request('a', 0), L.request('a', 1), L.request('b', 0)])]
        self.assertEqual([type(x).__name__ for x in items], ['Prepared', 'Prepared', 'RootRefusal'])
        cpu = jax.devices('cpu')[0]
        for item in items:
            back = pp.unpack(pp.pack(item), cpu)
            self.assertEqual(pp.digest(back), pp.digest(item))
            if type(item) is cw.Prepared:
                self.assertIs(back.made.root, back.detached)   # build's identity check holds after shipping
                self.assertEqual([w.state is None for w in back.made.worlds], [w.state is None for w in item.made.worlds])
        # The digest sees a changed byte, dtype, weak type or commitment.
        item = items[0];w0 = item.made.worlds[0];leaves, tree = jax.tree_util.tree_flatten((w0.state, w0.public))
        changed = [x for x in leaves];changed[0] = changed[0] + 1
        state, public = jax.tree_util.tree_unflatten(tree, changed)
        other = replace(item, made=replace(item.made, worlds=(replace(w0, state=state, public=public),) + item.made.worlds[1:]))
        self.assertNotEqual(pp.digest(other), pp.digest(item))

    def test_world_leaves_keep_kind_dtype_weak_type_and_commitment(self):
        cpu = jax.devices('cpu')[0]
        weak = j.asarray(1.5);self.assertTrue(weak.aval.weak_type)   # a Python scalar's array: weak-typed
        committed = jax.device_put(np.arange(3, dtype=np.int32), cpu)
        state = dict(a=weak, b=committed, c=np.arange(4, dtype=np.int16), d=7, e=j.zeros((2, 2), j.bfloat16))
        worlds = (mapping.World(0, state, (j.ones(3, j.int32),), (), 11, False), mapping.World(1, None, None, ('refused',), 12, False),
                  mapping.World(2, dict(state, d=8), (j.zeros(3, j.int32),), (), 13, False))
        shells, packed = pp.pack_worlds(worlds)
        self.assertEqual(len(packed), 1)   # worlds 0 and 2 share one structure: one stacked buffer per leaf
        back = pp.unpack_worlds(*pickle.loads(pickle.dumps((shells, packed))), cpu)
        self.assertEqual(pp.digest(back), pp.digest(worlds))
        s = back[0].state
        self.assertTrue(s['a'].aval.weak_type);self.assertFalse(s['b'].aval.weak_type)
        self.assertTrue(s['b'].committed);self.assertFalse(s['e'].committed);self.assertEqual(s['e'].dtype, j.bfloat16)
        self.assertIs(type(s['c']), np.ndarray);self.assertEqual((s['d'], back[2].state['d']), (7, 8));self.assertIsNone(back[1].state)

    def test_frozen_views_and_read_only_arrays_survive(self):
        frozen = np.arange(3);frozen.flags.writeable = False
        value = dict(view=types.MappingProxyType(dict(x=frozen, y=types.MappingProxyType({'z': 1}))))
        back = pickle.loads(pp.dumps(value))
        self.assertIs(type(back['view']), types.MappingProxyType);self.assertIs(type(back['view']['y']), types.MappingProxyType)
        self.assertFalse(back['view']['x'].flags.writeable);self.assertEqual(pp.digest(back), pp.digest(value))
        with self.assertRaises(TypeError):pickle.dumps(value)   # why prepare_pool has its own pickler


class PoolTests(Fixture, unittest.TestCase):
    def pool(self, workers, payload, **kw):
        kw.setdefault('startup_seconds', 300.)
        pool = pp.PreparePool(FACTORY, payload, workers=workers, **kw).start();self.addCleanup(pool.close)
        return pool

    def test_battle_affinity_matches_serial_prepare_in_order(self):
        keys = [('b0', 0), ('b1', 0), ('b0', 1), ('b2', 0), ('b1', 1), ('b0', 2), ('b2', 1)]
        payload = dict(W=8, outcomes=[(k, ('outcome', i, ())) for i, k in enumerate(keys)])
        requests = [L.request(b, d) for b, d in keys] + [L.request('b0', 1)]   # the last: an address already attempted
        pool = self.pool(2, payload)
        results = pool.prepare_many([protocol(r) for r in requests])
        self.assertEqual([r.kind for r in results], ['prepared'] * len(requests))
        self.assertEqual(pool.affinity, {'b0': 0, 'b1': 1, 'b2': 0})   # first sight: the worker with the fewest battles, lowest index
        self.assertEqual([r.worker for r in results], [0, 1, 0, 0, 1, 0, 0, 0])
        serial = self.serial(payload, requests)
        for r, (item, lab) in zip(results, serial):
            self.assertEqual(pp.digest(r.item), pp.digest(item));self.assertEqual(labels(r.labels), labels(lab))
        self.assertEqual(type(results[-1].item).__name__, 'RootRefusal')   # the no-retry rule: the worker holds the battle's attempts
        self.assertEqual(results[-1].item.kind, 'SEARCH_REQUEST_ALREADY_ATTEMPTED')
        # Eviction: the battle's sessions and attempts go (on its worker, after its pending prepares), so the address prepares again.
        pool.evict({'b0'});again = pool.prepare_many([protocol(L.request('b0', 1))])[0]
        self.assertEqual(type(again.item).__name__, 'Prepared');self.assertEqual(pool.affinity['b0'], 0)
        # reset forgets everything, everywhere.
        pool.reset();self.assertEqual(pool.affinity, {})
        self.assertEqual(type(pool.prepare_many([protocol(L.request('b1', 0))])[0].item).__name__, 'Prepared')

    def test_a_crash_a_hang_and_an_error_fail_only_their_requests(self):
        keys = [('ok', 0), ('crash', 0), ('crash', 1), ('slow', 0), ('fail', 0), ('ok', 1)]
        payload = dict(W=8, outcomes=[(k, ('outcome', i, ())) for i, k in enumerate(keys)], crash=[('crash', 0)], slow=[(('slow', 0), 60.)],
                       fail=[('fail', 0)])
        pool = self.pool(2, payload, timeout_seconds=3.)
        started = time.monotonic();results = pool.prepare_many([protocol(L.request(b, d)) for b, d in keys]);took = time.monotonic() - started
        kinds = {k: r.kind for k, r in zip(keys, results)}
        self.assertEqual(kinds, {('ok', 0): 'prepared', ('crash', 0): 'died', ('crash', 1): 'prepared', ('slow', 0): 'died',
                                 ('fail', 0): 'error', ('ok', 1): 'prepared'})
        texts = {k: r.text for k, r in zip(keys, results)}
        self.assertRegex(texts[('crash', 0)], r'^RuntimeError: prepare worker \d exited \(code 3\)$')
        self.assertRegex(texts[('slow', 0)], r'prepare worker \d exited \(code -9\) after its task ran past the prepare timeout')
        self.assertEqual(texts[('fail', 0)], 'RuntimeError: boom')
        self.assertLess(took, 40.)   # the hung prepare was killed at its timeout, never waited out
        # The crashed battle's next decision prepared (from a fresh session: the evicted path; moved to a live worker rather than
        # waiting for the restart's warm-in), and the crashed worker came back.
        self.assertGreaterEqual(pool.workers[results[1].worker].generation, 2)
        deadline = time.monotonic() + 120
        while not all(w.state == 'ready' for w in pool.workers) and time.monotonic() < deadline:
            for event in iter(lambda: pool._get(.2, []), None):pool.handle(event)
        self.assertTrue(all(w.state == 'ready' for w in pool.workers))

    def test_a_restart_that_cannot_spawn_answers_the_dead_workers_tasks(self):
        keys = [('crash', 0), ('ok', 0)]
        payload = dict(W=8, outcomes=[(k, ('outcome', i, ())) for i, k in enumerate(keys)], crash=[('crash', 0)])
        pool = self.pool(1, payload)
        def cannot(w):
            raise OSError('no process')
        pool._spawn = cannot   # the restart after the crash fails: no live worker is left for the queued task
        results = pool.prepare_many([protocol(L.request(b, d)) for b, d in keys])
        self.assertEqual([r.kind for r in results], ['died', 'died']);self.assertIn('no live prepare worker', results[1].text)
        self.assertEqual(pool.workers[0].state, 'failed');self.assertTrue(pool.workers[0].ident.startswith('restart failed: OSError'))

    def test_a_spawn_that_starts_no_process_closes_both_socket_ends(self):
        pool = pp.PreparePool(FACTORY, dict(W=8, outcomes=[]), workers=1, python='/nonexistent/python');made = [];real = pp.socket.socketpair
        def pair(*a):
            ends = real(*a);made.extend(ends);return ends
        with patch.object(pp.socket, 'socketpair', pair), self.assertRaises(OSError):
            pool._spawn(pool.workers[0])
        self.assertEqual([end.fileno() for end in made], [-1, -1])   # closed: no descriptor leaks per failed restart

    def test_a_dead_process_is_reported_once_even_without_its_readers_word(self):
        payload = dict(W=8, outcomes=[(('ok', 0), ('outcome', 0, ()))])
        pool = self.pool(2, payload);w = pool.workers[0];generation = w.generation
        w.process.kill();w.process.wait()
        pool.watchdog()   # the watchdog's own 'gone' for the exited process (the reader's may come too: one is handled)
        deadline = time.monotonic() + 120
        while (w.generation == generation or w.state != 'ready') and time.monotonic() < deadline:
            for event in iter(lambda: pool._get(.2, []), None):pool.handle(event)
        for event in iter(lambda: pool._get(1., []), None):pool.handle(event)   # any late duplicate 'gone'
        self.assertEqual((w.generation, len(w.restarts), w.state), (generation + 1, 1, 'ready'))

    def test_the_warm_in_runs_before_ready_and_forgets_its_sessions(self):
        keys = [('w', 0), ('w', 1), ('v', 3)]
        payload = dict(W=8, outcomes=[(k, ('outcome', i, ())) for i, k in enumerate(keys)])
        pool = self.pool(1, payload, warm_in=[protocol(L.request(b, d)) for b, d in keys])
        self.assertEqual({k: v for k, v in pool.workers[0].ident['warm_in'].items() if k in ('requests', 'kinds')}, dict(requests=3, kinds={'prepared': 3}))
        again = pool.prepare_many([protocol(L.request('w', 0))])[0]   # forgotten: the address is not an attempted one
        self.assertEqual(type(again.item).__name__, 'Prepared')

    def test_the_in_process_slot_prepares_here_and_a_hung_one_fails_closed(self):
        payload = dict(W=8, outcomes=[(k, ('outcome', i, ())) for i, k in enumerate([('h', 0), ('h', 1), ('q', 0)])])
        pool = self.pool(1, payload, timeout_seconds=2.)
        constructor = self.cw.constructor(8, saved_root_admission=False, rung='DN2W8');prepare = constructor.prepare
        def slow(request, index):
            if request['battle_id'] == 'h' and request['decision_index'] == 0:
                time.sleep(4)
            return prepare(request, index)
        constructor.prepare = slow;local = pool.attach_local(constructor)
        with L.Codex(outcomes_of(self.cw, payload)):
            pool.submit(('t', 'q'), protocol(L.request('q', 0)), 'q', prefer_local=True)
            here = pool._run([('t', 'q')])[0]   # homed in process: prepared by this process's constructor, nothing shipped
            self.assertEqual((here.kind, here.worker, here.bytes, pool.affinity['q']), ('prepared', local.index, 0, local.index))
            self.assertEqual(pp.digest(here.item), pp.digest(self.serial(payload, [L.request('q', 0)])[0][0]))
            pool.submit(('t', 0), protocol(L.request('h', 0)), 'h', prefer_local=True)
            hung = pool._run([('t', 0)])[0]
            self.assertEqual((hung.kind, hung.text), ('died', 'RuntimeError: the in-process prepare ran past the prepare timeout'))
            self.assertEqual(local.state, 'failed');self.assertNotIn('h', pool.affinity);self.assertNotIn('q', pool.affinity)
            pool.submit(('t', 1), protocol(L.request('h', 1)), 'h', prefer_local=True)   # the slot is given up: a worker, from scratch
            moved = pool._run([('t', 1)])[0]
            self.assertEqual((moved.kind, moved.worker, pool.affinity['h']), ('prepared', 0, 0))
            time.sleep(3)   # the hung thread finishes: its late result is ignored
            self.assertEqual(pool.handle(pool.events.get(timeout=5)), [])

    def test_a_worker_that_cannot_start_fails_the_start(self):
        pool = pp.PreparePool(FACTORY, dict(W=8, outcomes=[], init_fail=True), workers=1, startup_seconds=120.);self.addCleanup(pool.close)
        with self.assertRaisesRegex(RuntimeError, 'failed to start: 0: RuntimeError: this worker cannot start'):
            pool.start()


if __name__ == '__main__':
    unittest.main()
