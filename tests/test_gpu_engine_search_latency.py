"""d3_search_latency on stand-in pricers, prep and solvers (no bank, no device): contiguous stage accounting, cold-call
exclusion, the 800 ms selection, nominal leaves (and the real engine's slot widths), W=8 sequential and packed composition,
the --solver plug-in through the real backups, the header's constructor path / commit / note, isolation argv and arguments."""
import contextlib
import dataclasses
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import types
from typing import NamedTuple
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from gpu_engine import d3_depth2_gate as gate, d3_search_latency as tool
from gpu_engine import device_depthn_pricer as dn, device_public_worlds as real_worlds
from pokezero import v5_contract as contract, v5_gumbel as original


class FakeTime:
    """A deterministic perf_counter: callables advance it; `extra` adds seconds to successive price() calls."""
    def __init__(self):self.now, self.extra = 100., []

    def __call__(self):return self.now

    def advance(self, seconds):self.now += seconds


CLOCK = FakeTime()
RECIPES = [original.Recipe(seed) for seed in gate.world_block_seeds(1779000000, 4)]   # the blocks' recipes (block 0 the declared one)


def check_starts(requests, recipe):
    """device_public_world_state._materialize's check: every start's seed is the priced recipe's world seed for its address."""
    if any(s.seed != recipe.world_seed(r.plans.root.address, s.world) for r in requests for s in r.plans.starts):
        raise ValueError('same declared source/root/world/recipe required')


class FakeDepthN:
    """DepthNPricer's contract as the tool reads it: price() -> frame.timing (world_prepare_seconds measured inside the call),
    per-call `compiles`, slot widths k_own/k_opp (searched: ply 2 derived by searched_widths' rule). Each call advances CLOCK by
    .01 s materialization, 2 s compile on an instance's first call, .002 s per request and the next CLOCK.extra entry."""
    built = []

    def __init__(self, executor, model, *, capacity, depth, k_own, k_opp, selection='prior', **kw):
        widths = lambda k: tuple(k) if isinstance(k, tuple) else (k,) * (depth - 1)
        self.k_own, self.k_opp = widths(k_own), widths(k_opp)
        if selection == 'searched':
            ko, kp = self.k_own[0], self.k_opp[0];self.k_own = (min(ko + 1, 9),) + self.k_own[1:];self.k_opp = (min(max(kp + 1, ko + 2), 9),) + self.k_opp[1:]
        self.depth, self.selection, self.capacity, self.calls, self.log = depth, selection, capacity, 0, [];FakeDepthN.built.append(self)

    def price(self, requests, materializer, recipe, params, *, selection_maps=None):
        requests = list(requests)
        if len(requests) > self.capacity:raise ValueError('one frame exceeds declared root capacity')
        if (self.selection == 'searched') != (selection_maps is not None):raise ValueError('maps belong to searched')
        check_starts(requests, recipe);self.log.append((len(requests), RECIPES.index(recipe), [r.block for r in requests]))
        CLOCK.advance(.01 + (2. if not self.calls else 0.) + .002 * len(requests) + (CLOCK.extra.pop(0) if CLOCK.extra else 0.))
        self.compiles = (('P1', .5, 1.5),) if not self.calls else ();self.calls += 1
        leaves = 81 * 2 * len(requests) * math.prod(a * b for a, b in zip(self.k_own, self.k_opp))
        return SimpleNamespace(depth=self.depth, pre=np.zeros((len(requests), 2)), timing=dict(world_prepare_seconds=.01, device_seconds=.001 * len(requests),
            ply_loop_seconds=.0008, host_pack_seconds=.0001, frame_wall_seconds=.012, enumerated_leaves=leaves, evaluated_leaf_lanes=leaves))


def fake_maps(frame, k_own, k_opp):
    CLOCK.advance(.005);R = frame.pre.shape[0]
    return np.zeros((R, 2, 9, 9, k_own + 1), np.int32), np.zeros((R, 2, 9, 9, k_opp + 2), np.int32), np.zeros(R, bool)


MODULE = SimpleNamespace(DepthNPricer=FakeDepthN, as_depth2_frame=lambda f: f, searched_maps=fake_maps)


class FakeDepth2:
    """The merged Depth2Pricer's contract as the tool reads it (its constructor signature, d5150eac): price() -> frame.timing with
    world_prepare_seconds, this call's lower/compile seconds (0 once its kernel table holds the capacity) and turn-2 leaf counts.
    Each call advances CLOCK by .01 s materialization, 2 s compile on an instance's first call and .003 s per request."""
    built = []

    def __init__(self, executor, model, *, capacity=2048, actor_capacity=33, k_own=9, k_opp=9, k1_own=9, k1_opp=9, child_block=27, digest=False, forward=None,
            fixed_capacity=False):
        self.capacity, self.k_own, self.k_opp, self.child_block, self.digest, self.fixed_capacity = capacity, k_own, k_opp, child_block, digest, fixed_capacity
        self.given = dict(capacity=capacity, actor_capacity=actor_capacity, k_own=k_own, k_opp=k_opp, k1_own=k1_own, k1_opp=k1_opp, digest=digest, fixed_capacity=fixed_capacity)
        self.kernels, self.log = {}, [];FakeDepth2.built.append(self)

    def price(self, requests, materializer, recipe, params):
        requests = list(requests)
        if len(requests) > self.capacity:raise ValueError('one frame exceeds declared root capacity')
        check_starts(requests, recipe);self.log.append((len(requests), RECIPES.index(recipe), [r.block for r in requests]))
        cold = self.capacity not in self.kernels;self.kernels[self.capacity] = True
        CLOCK.advance(.01 + (2. if cold else 0.) + .003 * len(requests))
        leaves = 81 * 2 * len(requests) * self.k_own * self.k_opp
        return SimpleNamespace(timing=dict(world_prepare_seconds=.01, device_seconds=.002 * len(requests), lower_seconds=.5 if cold else 0., compile_seconds=1.5 if cold else 0.,
            root_capacity=self.capacity, child_block=self.child_block, frame_wall_seconds=.02, turn2_enumerated_leaves=leaves, turn2_evaluated_leaf_lanes=leaves))


class OldDepth2:
    def __init__(self, executor, model, *, capacity=2048, actor_capacity=33, k_own=9, k_opp=9, k1_own=9, k1_opp=9, child_block=27, digest=False, forward=None):pass


DEPTH2 = SimpleNamespace(Depth2Pricer=FakeDepth2)


class FakeSolver:
    spec = 'fake'

    def __init__(self):self.views = []

    def solve(self, views, requests, clock, pool=None):
        self.views.append(list(views));CLOCK.advance(.0001 * len(requests) * len(views));clock.lap('solve', 'backup_and_reduce')
        return [dict(status='LABELED', policy_target=(0., 1.) + (0.,) * 7) for _ in requests]

    def moves(self, results, requests):
        return [tool.move_of(r) for r in results]


def address(i):
    return contract.DecisionAddress(100 + i, 7, i % 2)


def request(i, block):
    """A request planned under block `block`'s recipe: its two starts carry that recipe's world seeds, as plan_starts writes them."""
    starts = tuple(SimpleNamespace(world=w, seed=RECIPES[block].world_seed(address(i), w)) for w in (0, 1))
    return SimpleNamespace(root=i, block=block, root_logits=(0.,) * 9, root_legal=(True,) + (False,) * 8,
        plans=SimpleNamespace(root=SimpleNamespace(address=address(i)), starts=starts))


def fake_ctx(blocks=4):
    entered = []
    def bypass(b):
        @contextlib.contextmanager
        def context():entered.append(b);yield
        return context
    ctx = SimpleNamespace(provider=SimpleNamespace(executor=None, materializer='materializer'), plan=SimpleNamespace(model=None), params=('params',),
        actor_capacity=1, blocks=[(RECIPES[b], bypass(b) if b else contextlib.nullcontext) for b in range(blocks)], entered=entered, checked=[], fence=lambda: None)
    def prep(roots):
        CLOCK.advance(.003);return [request(i, 0) for i in roots.bound], dict(detach=.002, plan_starts=.0005, assemble=.0005)
    def replan(r, recipe):
        CLOCK.advance(.001);return request(r.root, RECIPES.index(recipe))
    ctx.prep, ctx.replan = prep, replan
    ctx.check = lambda requests, roots: ctx.checked.append(len(requests))
    ctx.roots = lambda capacity: SimpleNamespace(bound=list(range(capacity)), slots=[(0, i, 0) for i in range(capacity)])
    return ctx


def pessimistic(grid, pi_own, pi_opp):
    """A solver plug-in for the test: the worst own slot's opponent-averaged value."""
    return min(math.fsum(p * v for p, v in zip(pi_opp, row)) for row in grid)


def two_args(grid, pi_own):
    return 0.


NOT_CALLABLE = 3


def depth2_frame(values, errored=()):
    """A real N=2 DepthNFrame: R roots, both worlds, only turn-1 cell (0,0) expanded with own/opponent slots on actions (0, 1) at
    equal priors, the root opponent waiting; leaf values values[r] [own slot][opponent slot]; (root, world) in errored gets a leaf error."""
    R = len(values);Z = lambda shape, d=np.float32: np.zeros((R, 2) + shape, d)
    own_actions, opp_actions = np.full((R, 2, 9, 9, 2), -1, np.int32), np.full((R, 2, 9, 9, 2), -1, np.int32)
    frame = dn.DepthNFrame(depth=2, k_own=(2,), k_opp=(2,), pre=Z((), np.int32), root_peer_requested=Z((), bool), root_peer_legal=Z((9,), bool), root_peer_logits=Z((9,)),
        t1_status=Z((9, 9), np.int8), t1_value=Z((9, 9)), t1_error=Z((9, 9), np.int8), t1_own_requested=Z((9, 9), bool), t1_own_legal=Z((9, 9, 9), bool),
        t1_own_logits=Z((9, 9, 9)), t1_peer_requested=Z((9, 9), bool), t1_peer_legal=Z((9, 9, 9), bool), t1_peer_logits=Z((9, 9, 9)), t1_own_actions=own_actions,
        t1_opp_actions=opp_actions, nodes=(), leaf_status=Z((81, 2, 2), np.int8), leaf_value=Z((81, 2, 2)), leaf_error=Z((81, 2, 2), np.int8), world_reports=(), timing={})
    for name in ('t1_own_requested', 't1_peer_requested'):getattr(frame, name)[:, :, 0, 0] = True
    frame.t1_status[:, :, 0, 0] = 2;frame.t1_own_legal[:, :, 0, 0, :2] = True;frame.t1_peer_legal[:, :, 0, 0, :2] = True
    own_actions[:, :, 0, 0] = [0, 1];opp_actions[:, :, 0, 0] = [0, 1];frame.leaf_status[:, :, 0] = 4
    for r, v in enumerate(values):frame.leaf_value[r, :, 0] = v
    for r, w in errored:frame.leaf_status[r, w, 0, 1, 1] = 3
    return frame


ROOT = SimpleNamespace(root_logits=(0.,) * 9, root_legal=(True,) + (False,) * 8)
ROOTS = [request(i, 0) for i in range(2)]


def depth3_frame(leaves):
    """A real N=3 DepthNFrame: R roots, both worlds; turn-1 cell (0,0) expanded with slots on actions (0, 1); its four ply-2 children
    (node ids 0..3) expanded with slots on actions (0, 1) at equal priors; leaves[r] [node][own slot][opponent slot]."""
    R, M = len(leaves), 81 * 4;Z = lambda shape, d=np.float32: np.zeros((R, 2) + shape, d);ids = np.arange(M)
    nodes = dn.PlyNodes(ply=2, parent=ids // 4, parent_own_slot=((ids % 4) // 2).astype(np.int8), parent_opp_slot=(ids % 2).astype(np.int8), status=Z((M,), np.int8),
        value=Z((M,)), error=Z((M,), np.int8), own_requested=Z((M,), bool), opp_requested=Z((M,), bool), own_legal=Z((M, 9), bool), opp_legal=Z((M, 9), bool),
        own_logits=Z((M, 9)), opp_logits=Z((M, 9)), own_actions=np.full((R, 2, M, 2), -1, np.int32), opp_actions=np.full((R, 2, M, 2), -1, np.int32))
    nodes.status[:, :, :4] = 2;nodes.own_actions[:, :, :4] = [0, 1];nodes.opp_actions[:, :, :4] = [0, 1]
    frame = dataclasses.replace(depth2_frame([[[0., 0.], [0., 0.]]] * R), depth=3, k_own=(2, 2), k_opp=(2, 2), nodes=(nodes,), leaf_status=Z((M, 2, 2), np.int8),
        leaf_value=Z((M, 2, 2)), leaf_error=Z((M, 2, 2), np.int8))
    frame.leaf_status[:, :, :4] = 4
    for r, v in enumerate(leaves):frame.leaf_value[r, :, :4] = v
    return frame


def stub_hedge(path=None):
    """A stand-in for search_hedge (the real one is PR #1829's and is not on this branch): its names and signatures with simple
    exact maths (matrix_game = the pure maximin; bayes_hedge = the legal action with the best mean worst case) and a call log."""
    hedge = types.ModuleType('gpu_engine.search_hedge');hedge.calls = []
    hedge.DEPTH2_FIELDS = ('pre', 'root_peer_requested', 'root_peer_logits', 't1_status', 't1_value', 't2_status', 't2_value')
    def matrix_game(values):
        m = np.asarray(values, np.float64);hedge.calls.append(('matrix_game', m.shape));x = np.zeros(len(m));x[int(np.argmax(m.min(axis=1)))] = 1.
        return float(m.min(axis=1).max()), x
    def depth2_hedged_root(fields):
        hedge.calls.append(('depth2_hedged_root', tuple(sorted(fields))));out = []
        for w in range(len(fields['pre'])):
            if int(fields['pre'][w]):out.append(None);continue
            status, requested = fields['t1_status'][w], bool(fields['root_peer_requested'][w])
            def cell(a, r):
                if status[a, r] == 1:return float(fields['t1_value'][w, a, r])
                t2 = fields['t2_status'][w, a, r]
                if status[a, r] != 2 or (t2 == 3).any():return None
                return matrix_game(np.asarray(fields['t2_value'][w, a, r], np.float64)[np.ix_(np.flatnonzero(t2.any(1)), np.flatnonzero(t2.any(0)))])[0]
            out.append(original.WorldPrices(requested, tuple(bool(status[:, r].any()) for r in range(9)) if requested else (False,) * 9,
                tuple(map(float, fields['root_peer_logits'][w])), [[cell(a, r) if status[a, r] else None for r in range(9)] for a in range(9)]))
        return tuple(out)
    def bayes_hedge(worlds, root_legal, *, lam=0.):
        worlds = list(worlds);hedge.calls.append(('bayes_hedge', len(worlds), lam))
        if not worlds or any(w is None for w in worlds):return None
        worst = {}
        for a in (a for a in range(9) if root_legal[a]):
            rows = [[w.joint_values[a][b] for b in ([b for b in range(9) if w.opponent_legal[b]] if w.opponent_requested else [0])] for w in worlds]
            if any(v is None for row in rows for v in row):return None
            worst[a] = sum(min(row) for row in rows) / len(rows)
        best = max(worst, key=lambda a: (worst[a], -a));return SimpleNamespace(strategy=tuple(float(a == best) for a in range(9)), value=worst[best])
    def search_uniform(seed, address):
        hedge.calls.append(('search_uniform', seed, (address.stream, address.frame, address.owner)));return .25
    def sample(strategy, u):
        hedge.calls.append(('sample', u));return next(a for a, p in enumerate(strategy) if p > 0)
    for f in (matrix_game, depth2_hedged_root, bayes_hedge, search_uniform, sample):setattr(hedge, f.__name__, f)
    if path is not None:hedge.__file__ = str(path)
    return hedge


class FrameDepthN(FakeDepthN):
    """FakeDepthN whose frames are real N=2 DepthNFrames (depth2_frame), so the real solvers run on them."""
    def price(self, requests, materializer, recipe, params, *, selection_maps=None):
        fake = super().price(requests, materializer, recipe, params, selection_maps=selection_maps)
        return dataclasses.replace(depth2_frame([[[.6, -.2], [.1, .1]]] * len(list(requests))), timing=fake.timing)


FRAMES = SimpleNamespace(DepthNPricer=FrameDepthN, as_depth2_frame=dn.as_depth2_frame, searched_maps=dn.searched_maps)


class FakePool:
    """pool.map's contract in process: every unit and every result goes through pickle (as the IPC does); the clock pays .001 s per unit."""
    def __init__(self):self.mapped = []

    def map(self, fn, units, chunksize=None):
        self.mapped.append((fn, len(units), chunksize));out = []
        for unit in units:CLOCK.advance(.001);out.append(pickle.loads(pickle.dumps(fn(pickle.loads(pickle.dumps(unit))))))
        return out


class FakeTables:
    pass


class FakeSource:
    pass


class FakeProvider:
    def __init__(self):self.materializer, self.source = FakeTables(), FakeSource()


def fake_numpy_worlds(requests, materializer, recipe):
    return ()


def fake_materialize_host(*a, **kw):
    return None


def fake_plan_starts(root, source, recipe):
    return None


def fake_detach_roots(*a, **kw):
    return ()


class RootContext(NamedTuple):
    streams: object
    frames: object


def chunk_frames(steps=2, rows=3):
    """Host frames as the bank's boundary gives them: [step, row(, owner)] arrays with distinct values per slot."""
    shape = (steps, rows, 2);index = np.arange(np.prod(shape)).reshape(shape)
    return SimpleNamespace(streams=np.arange(steps * rows).reshape(steps, rows) + 100, frames=np.arange(steps * rows).reshape(steps, rows) + 7,
        categorical_ids=index[..., None] * 10, numeric_features=index[..., None] * 1., token_type_ids=index[..., None], attention_mask=np.ones(shape + (1,), bool),
        raw_policy_logits=(index[..., None] + np.arange(9)).astype(np.float32), legal_action_mask=np.broadcast_to(np.arange(9) < 4, shape + (9,)).copy(),
        actor_indices=np.stack([index[..., 0] % 3, index[..., 1] % 5], -1))


class SearchLatencyTests(unittest.TestCase):
    def setUp(self):
        CLOCK.now, CLOCK.extra = 100., [];FakeDepthN.built = [];FakeDepth2.built = []

    def out(self):
        return str(Path(tempfile.gettempdir()) / f'search-latency-never-{uuid.uuid4().hex}')

    def hedge_file(self, content):
        path = Path(tempfile.mkdtemp()) / 'search_hedge.py';path.write_bytes(content);return path

    def args(self, *extra):
        return tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', self.out(), *extra])

    def test_stage_accounting_sums_to_the_total(self):
        args, ctx, solver = self.args(), fake_ctx(), FakeSolver()
        search = tool.Search('depth3-searched-k3', 4, args, ctx, module=MODULE)
        record, requests = tool.run_call(search, ctx, ctx.roots(4), 2, 'sequential', solver=solver, timer=CLOCK)
        s, p = record['stages'], record['parts'];v = record['variants']['fake@pool0']
        self.assertAlmostEqual(sum(s.values()), record['wall_seconds'], places=12)   # contiguous laps: every instant charged once
        self.assertAlmostEqual(v['total_seconds'], sum(s[k] for k in tool.STAGES), places=12)
        self.assertAlmostEqual(v['total_seconds'], record['wall_seconds'] - s['compile'], places=12)   # compile never in the total
        self.assertAlmostEqual(s['host_prep'], .003);self.assertAlmostEqual(s['compile'], 4.)   # both pricers' first-call compiles
        self.assertAlmostEqual(s['world_construction'], .02);self.assertAlmostEqual(p['world_construction.materialize'], .02)   # the pass and depth N each materialize
        self.assertAlmostEqual(p['pricer.depth2_pass'], .008);self.assertAlmostEqual(p['pricer.searched_maps_host'], .005);self.assertAlmostEqual(p['pricer.depthn'], .008)
        self.assertAlmostEqual(s['pricer'], p['pricer.depth2_pass'] + p['pricer.searched_maps_host'] + p['pricer.depthn'], places=12)
        self.assertAlmostEqual(s['solve'], .0004);self.assertAlmostEqual(s['return'], 0.)
        self.assertAlmostEqual(p['host_prep.detach'], .002)   # a part measured inside host_prep: noted, never added to the stage
        self.assertEqual((v['labeled'], v['moves'], v['leftover_device_arrays']), (4, [1] * 4, 0))
        self.assertAlmostEqual(record['common_seconds'], s['host_prep'] + s['world_construction'] + s['pricer'], places=12)
        self.assertEqual(record['pricing'][0]['maps_widths'], [4, 5]);self.assertEqual(record['pricing'][0]['slot_widths'], [4, 5])
        self.assertAlmostEqual(record['pricing'][0]['depthn']['p1_seconds'], .004 - .0007)
        self.assertEqual(len(requests), 4)

    def test_cold_calls_are_excluded_from_warm_statistics(self):
        args, ctx = self.args(), fake_ctx();search = tool.Search('depth3-prior-k3', 4, args, ctx, module=MODULE)
        CLOCK.extra = [1., .1, .3, .2]   # the cold call is slow and compiles; warm calls vary
        memory = lambda: dict(process_peak_bytes_so_far=10, bytes_in_use=4)
        [series] = tool.measure_series(search, ctx, ctx.roots(4), 2, 'sequential', 3, solver=FakeSolver(), progress=None, memory=memory,
            baseline=dict(bytes_in_use=4), capacity=4, timer=CLOCK)
        warm = [.003 + .01 + .008 + x + .0004 for x in (.1, .3, .2)]
        self.assertEqual(series['warm_calls'], 3);self.assertEqual(series['warm']['total']['n'], 3)
        self.assertAlmostEqual(series['warm']['total']['median_ms'], 1e3 * np.median(warm), places=6)
        self.assertAlmostEqual(series['warm']['total']['p90_ms'], 1e3 * np.percentile(warm, 90), places=6)
        self.assertAlmostEqual(series['warm']['total']['max_ms'], 1e3 * max(warm), places=6)
        self.assertAlmostEqual(series['first_call']['compile_seconds'], 2.);self.assertAlmostEqual(series['first_call']['total_seconds'], .003 + .01 + 1.008 + .0004)
        self.assertEqual((series['warm_calls_with_compiles'], series['warm_compile_seconds']), (0, 0.))
        self.assertAlmostEqual(series['warm']['pricer']['median_ms'], 1e3 * .208, places=6)
        self.assertEqual(len(series['calls']), 4);self.assertEqual(ctx.checked, [4])   # the first call's requests are checked once
        self.assertEqual(series['memory']['peak_over_baseline_in_use'], 6)
        self.assertEqual(series['leaves']['nominal_per_world'], 6561);self.assertEqual(series['leaves']['measured_per_world'], 6561)

    def test_800ms_selection(self):
        self.assertEqual(tool.largest_within({1: 100., 4: 500., 16: 900.}), dict(largest_capacity=4, monotone=True))
        self.assertEqual(tool.largest_within({1: 100., 4: 800., 16: 800.1}), dict(largest_capacity=4, monotone=True))   # <= 800 fits
        self.assertEqual(tool.largest_within({1: 900., 4: 700.}), dict(largest_capacity=4, monotone=False))
        self.assertEqual(tool.largest_within({1: 801., 4: None}), dict(largest_capacity=None, monotone=True))
        total = lambda p90: dict(median_ms=p90 - 10, p90_ms=p90, max_ms=p90 + 5)
        series = [dict(config='a', worlds=w, mode=m, capacity=c, variant='v', warm=dict(total=total(p))) for w, m, c, p in (
            (2, 'sequential', 1, 100.), (2, 'sequential', 4, 300.), (2, 'sequential', 16, 900.), (8, 'sequential', 1, 700.), (8, 'sequential', 4, 1200.),
            (8, 'sequential', 16, 3000.), (8, 'packed', 1, 300.), (8, 'packed', 4, 600.), (8, 'packed', 16, 2000.))] + [
            dict(config='b', worlds=2, mode='sequential', capacity=c, variant='v', warm=dict(total=total(p))) for c, p in ((1, 900.), (4, 1500.))]
        selection = tool.select(series)
        a, b = selection['configs']['a']['v'], selection['configs']['b']['v']
        self.assertEqual(a['largest_capacity'], dict(capacity=4, worlds=2, mode='sequential'))
        self.assertEqual(a['largest_worlds'], dict(capacity=1, worlds=8, mode='sequential'))
        self.assertEqual(a['packed_proxy']['largest_worlds'], dict(capacity=4, worlds=8, mode='packed'))   # packed apart, never the headline
        self.assertEqual((b['largest_capacity'], b['largest_worlds'], b['packed_proxy']), (None, None, None))
        rows = {(r['config'], r['worlds'], r['mode']): r for r in selection['rows']};self.assertEqual({r['variant'] for r in selection['rows']}, {'v'})
        self.assertEqual(rows['a', 8, 'packed']['largest_capacity'], 4);self.assertEqual(rows['a', 2, 'sequential']['by_capacity']['4'], total(300.))
        self.assertEqual(selection['limit_ms'], 800.);self.assertIn('<= 800 ms', selection['rule'])

    def test_nominal_leaves_and_the_engines_slot_widths(self):
        self.assertEqual(tool.nominal_leaves((3, 3), (3, 3)), 6561)          # depth-3 prior k=3
        self.assertEqual(tool.nominal_leaves((4, 3), (5, 3)), 14580)         # depth-3 searched k=3: ply-2 widths (4,5), ply 3 (3,3)
        self.assertEqual(tool.nominal_leaves((9,), (9,)), 6561)              # depth-2 full width
        args, ctx = self.args(), fake_ctx()
        widths = {name: tool.Search(name, 4, args, ctx, module=dn).widths() for name in ('depth2-fw', 'depth3-prior-k3', 'depth3-searched-k3')}   # the REAL engine
        self.assertEqual(widths['depth2-fw'], dict(k_own=(9,), k_opp=(9,), nominal_leaves_per_world=6561))
        self.assertEqual(widths['depth3-prior-k3'], dict(k_own=(3, 3), k_opp=(3, 3), nominal_leaves_per_world=6561))
        searched = widths['depth3-searched-k3'];self.assertEqual((searched['k_own'], searched['k_opp'], searched['nominal_leaves_per_world']), ((4, 3), (5, 3), 14580))
        self.assertEqual(searched['depth2_pass']['nominal_leaves_per_world'], 6561)

    def test_real_pricers_share_one_program_table_across_capacities(self):
        args, ctx, programs = self.args(), fake_ctx(), {};ctx.params = (object(),)
        searches = [tool.Search(name, c, args, ctx, module=dn, programs=programs) for name, c in (('depth2-fw', 1), ('depth3-searched-k3', 4))]
        pricers = [p for s in searches for p in s.pricers()];self.assertEqual(len(pricers), 3)
        for pricer in pricers:
            pricer.bind(ctx.params);self.assertIs(pricer.programs, programs)   # price() re-binds the same evaluator: the table survives
            self.assertTrue(pricer.fixed_capacity);self.assertFalse(pricer.digest)
        self.assertEqual([p.capacity for p in pricers], [1, 4, 4])

    def test_w8_sequential_and_packed_composition(self):
        args, ctx, solver = self.args(), fake_ctx(), FakeSolver()
        search = tool.Search('depth2-fw', 2, args, ctx, module=MODULE)
        record, _ = tool.run_call(search, ctx, ctx.roots(2), 8, 'sequential', solver=solver, timer=CLOCK)
        self.assertEqual(search.pricer.log, [(2, b, [b, b]) for b in range(4)])   # K=4 passes of C roots, each block its own recipe
        self.assertEqual(ctx.entered, [1, 2, 3, 1, 2, 3])   # quarter-selection bypass around planning and pricing of blocks 1..3
        self.assertEqual([[o for _, o in views] for views in solver.views], [[0, 0, 0, 0]])
        self.assertAlmostEqual(record['parts']['world_construction.plan_starts'], 3 * 2 * .001)
        self.assertAlmostEqual(record['stages']['world_construction'], 3 * 2 * .001 + 4 * .01)
        self.assertEqual(record['enumerated_leaves'] / (2 * 8), 6561);self.assertEqual([p['block'] for p in record['pricing']], [0, 1, 2, 3])
        packed = tool.Search('depth2-fw', 8, args, ctx, module=MODULE);solver = FakeSolver()
        record, _ = tool.run_call(packed, ctx, ctx.roots(2), 8, 'packed', solver=solver, timer=CLOCK)
        # One pass at K*C of K copies of block 0 under block 0's recipe (FakeDepthN refuses any start of another recipe, as _materialize does).
        self.assertEqual(packed.pricer.log, [(8, 0, [0] * 8)]);self.assertAlmostEqual(record['parts']['world_construction.plan_starts'], 3 * 2 * .001)   # still planned
        with self.assertRaises(ValueError):packed.pricer.price([request(0, 1)], None, RECIPES[0], None)   # the shape the old packed call priced
        self.assertEqual([o for _, o in solver.views[0]], [0, 2, 4, 6]);self.assertEqual(len({id(f) for f, _ in solver.views[0]}), 1)
        self.assertEqual(record['enumerated_leaves'] / (2 * 8), 6561)
        with self.assertRaises(ValueError):tool.run_call(packed, ctx, ctx.roots(2), 2, 'packed', solver=solver, timer=CLOCK)
        # measure_config: one Search per (config, pricer capacity); packed at K*C; the first series of each capacity holds its cold call.
        FakeDepthN.built = [];args = self.args('--configs', 'depth2-fw@2/8', '--capacities', '1,4', '--repeats', '1')
        memory = lambda: dict(process_peak_bytes_so_far=10, bytes_in_use=4)
        series = tool.measure_config('depth2-fw', (2, 8), args, fake_ctx(), None, solver=FakeSolver(), module=MODULE, memory=memory, timer=CLOCK)
        self.assertEqual([(s['capacity'], s['worlds'], s['mode'], s['pricer_capacity']) for s in series],
            [(1, 2, 'sequential', 1), (1, 8, 'sequential', 1), (1, 8, 'packed', 4), (4, 2, 'sequential', 4), (4, 8, 'sequential', 4), (4, 8, 'packed', 16)])
        self.assertEqual([p.capacity for p in FakeDepthN.built], [1, 4, 16]);self.assertTrue(series[2]['timing_proxy'])   # C=1's packed pricer is C=4's
        self.assertEqual([s['first_call']['cold_of_config_capacity'] for s in series], [True, False, False, True, False, False])
        self.assertEqual([s['first_call']['compile_seconds'] for s in series], [2., 0., 2., 0., 0., 2.])

    def test_compose_is_the_equal_weight_mean_of_blocks(self):
        legal = (True, True) + (False,) * 7;logits = (0.,) * 9
        block = lambda q0, q1: SimpleNamespace(completed_q=(q0, q1) + (None,) * 7, policy_target=(.5, .5) + (0.,) * 7, root_value=0.)
        one = tool.compose([block(.2, -.1)], logits, legal, original);self.assertEqual(one['completed_q'][:2], (.2, -.1))
        two = tool.compose([block(.2, -.1), block(.4, .1)], logits, legal, original)
        self.assertAlmostEqual(two['completed_q'][0], .3);self.assertAlmostEqual(two['completed_q'][1], 0.);self.assertIsNone(two['completed_q'][2])
        p0 = math.exp(.3) / (math.exp(.3) + 1.);self.assertAlmostEqual(two['policy_target'][0], p0);self.assertAlmostEqual(two['root_value'], p0 * .3)
        self.assertEqual(tool.compose([block(.2, -.1), None], logits, legal, original)['status'], 'REFUSED')
        self.assertEqual(tool.move_of(two), 0);self.assertIsNone(tool.move_of(dict(policy_target=None)))
        self.assertEqual(tool.move_of(dict(policy_target=(.25, .25, .5))), 2);self.assertEqual(tool.move_of(dict(policy_target=(.5, .5))), 0)   # ties: lowest action

    def test_solver_plug_in_path_through_the_real_backups(self):
        values = ([[.6, -.2], [.1, .1]], [[-.4, 0.], [.3, .5]]);frame = depth2_frame(values);requests = [ROOT, ROOT]
        solve = lambda solver, views=((frame, 0),): (lambda clock: (solver.solve(list(views), requests, clock), clock))(tool.Clock(CLOCK))
        reference, clock = solve(tool.Solver.load('learner_max'))   # depth 2: depth2_backups on as_depth2_frame
        self.assertIn('solve.as_depth2_frame', clock.parts);self.assertEqual({'solve.slice', 'solve.worlds', 'solve.root_reduce'} - set(clock.parts), set())
        self.assertAlmostEqual(reference[0]['completed_q'][0], .2, places=6);self.assertAlmostEqual(reference[1]['completed_q'][0], .4, places=6)
        baseline, clock = solve(tool.Solver.load('gpu_engine.depthn_backups:learner_max'))   # the plug-in path's like-for-like baseline
        self.assertNotIn('solve.as_depth2_frame', clock.parts)
        for a, b in zip(reference, baseline):self.assertEqual(a['completed_q'], b['completed_q'])
        plugged = tool.Solver.load('tests.test_gpu_engine_search_latency:pessimistic');self.assertIs(plugged.rule, pessimistic)
        low, _ = solve(plugged);self.assertAlmostEqual(low[0]['completed_q'][0], .1, places=6);self.assertAlmostEqual(low[1]['completed_q'][0], -.2, places=6)
        calls = []
        spy = tool.Solver('spy', lambda grid, pi_own, pi_opp: calls.append((len(grid), len(grid[0]), tuple(pi_opp))) or 0.)
        solve(spy);self.assertEqual(calls, [(2, 2, (.5, .5))] * 4)   # 2 roots x 2 worlds x the one expanded cell
        self.assertEqual(plugged.describe()['rule'], 'tests.test_gpu_engine_search_latency.pessimistic');self.assertTrue(tool.Solver().describe()['stand_in'])
        # W = 4 from two blocks (frames): the equal-weight mean of their completed-Q; an errored leaf in any block refuses the root.
        other = depth2_frame(([[0., 0.], [.4, .4]], [[.1, .1], [.1, .1]]), errored=((1, 1),))
        mixed, _ = solve(tool.Solver(), ((frame, 0), (other, 0)))
        self.assertAlmostEqual(mixed[0]['completed_q'][0], (.2 + .4) / 2, places=6);self.assertEqual(mixed[1]['status'], 'REFUSED')
        self.assertIsNone(tool.move_of(mixed[1]));self.assertEqual(tool.move_of(mixed[0]), 0)
        for spec, error in (('tests.test_gpu_engine_search_latency:two_args', 'rule(grid, pi_own, pi_opp)'), ('tests.test_gpu_engine_search_latency:NOT_CALLABLE', 'not a callable')):
            with self.assertRaises(ValueError) as raised:tool.Solver.load(spec)
            self.assertIn(error, str(raised.exception))

    def test_header_carries_constructor_path_commit_and_note(self):
        fake = SimpleNamespace(batch=SimpleNamespace(_numpy_worlds=fake_numpy_worlds, mapping=SimpleNamespace(materialize_host=fake_materialize_host)), DepthNPricer=FakeDepthN)
        here = 'tests.test_gpu_engine_search_latency.'
        path = tool.constructor_path(FakeProvider(), fake, SimpleNamespace(detach_roots=fake_detach_roots, plan_starts=fake_plan_starts))
        self.assertEqual((path['materializer'], path['source'], path['world_builder'], path['world_materialize'], path['world_planner']),
            (here + 'FakeTables', here + 'FakeSource', here + 'fake_numpy_worlds', here + 'fake_materialize_host', here + 'fake_plan_starts'))
        self.assertFalse(path['uses_phase_d_constructor']);self.assertIn(here + 'fake_numpy_worlds', path['timed'])
        real = tool.constructor_path(FakeProvider(), dn, real_worlds)   # the tree's own path, resolved from the pricer module at run time
        self.assertEqual((real['world_builder'], real['world_materialize'], real['world_planner'], real['root_detach'], real['pricer']),
            ('gpu_engine.device_public_teacher_batch._numpy_worlds', 'gpu_engine.device_public_world_state.materialize_host', 'gpu_engine.device_public_worlds.plan_starts',
             'gpu_engine.device_public_worlds.detach_roots', 'gpu_engine.device_depthn_pricer.DepthNPricer'))
        self.assertFalse(real['uses_phase_d_constructor'])
        from gpu_engine import device_depth2_pricer as d2
        merged = tool.constructor_path(FakeProvider(), dn, real_worlds, d2)
        self.assertEqual((merged['merged_pricer'], merged['merged_world_builder'], merged['merged_same_world_builder']),
            ('gpu_engine.device_depth2_pricer.Depth2Pricer', 'gpu_engine.device_public_teacher_batch._numpy_worlds', True))
        def phase_d_plan(*a):return None
        phase_d_plan.__module__ = 'gpu_engine.phase_d_live_worlds'
        self.assertTrue(tool.constructor_path(FakeProvider(), fake, SimpleNamespace(detach_roots=fake_detach_roots, plan_starts=phase_d_plan))['uses_phase_d_constructor'])
        # The commit: git rev-parse HEAD when the code directory holds a .git, else --source-commit, else UNKNOWN.
        root = Path(tempfile.mkdtemp());(root / '.git').write_text('gitdir: elsewhere\n');seen = []
        def run(command, **kw):
            seen.append(command[3:]);return SimpleNamespace(stdout='abc123def\n' if command[3] == 'rev-parse' else ' M x.py\n')
        commit = tool.source_commit(root, 'abc123', run=run)
        self.assertEqual((commit['commit'], commit['origin'], commit['dirty'], commit['matches_given']), ('abc123def', 'git rev-parse HEAD', True, True))
        self.assertEqual(seen, [['rev-parse', 'HEAD'], ['status', '--porcelain', '--untracked-files=no']])
        bare = Path(tempfile.mkdtemp())
        self.assertEqual(tool.source_commit(bare, 'f00d', run=run)['origin'], '--source-commit');self.assertEqual(tool.source_commit(bare, 'f00d', run=run)['commit'], 'f00d')
        self.assertEqual(tool.source_commit(bare, None, run=run)['commit'], 'UNKNOWN')
        def broken(command, **kw):raise OSError('no git')
        self.assertEqual(tool.source_commit(root, 'f00d', run=broken)['origin'], '--source-commit')
        # The report: evaluator first, then the header with the path, the commit and the note; the markdown names all three.
        args = self.args('--configs', 'depth2-fw@2,depth3-prior-k3@2', '--capacities', '1,2', '--repeats', '1');memory = lambda: {}
        sections = {name: tool.measure_config(name, w, args, fake_ctx(), None, solver=FakeSolver(), module=MODULE, memory=memory, timer=CLOCK)
            for name, w in args.config_worlds.items()}
        hedge = stub_hedge(self.hedge_file(b'stub hedge'))
        context = dict(evaluator=dict(override=True), root_values=[], roots_available=1, source={}, devices=['cpu'], block_seeds=[7], compile_cache=None,
            world_construction_path=path, solvers=[tool.Solver().describe(), tool.Solver('hedge', hedge=hedge, hedge_name=tool.HEDGE_MODULE).describe()],
            solver_pools=[dict(workers=16, startup_seconds=3.5)], hedge=tool.hedge_header(hedge, tool.HEDGE_COMMIT), variants=['fake@pool0'])
        report = tool.report_of(args, sections, context, commit, isolated_run=False)
        self.assertEqual(list(report)[:2], ['evaluator', 'header'])
        self.assertEqual((report['header']['world_construction_path'], report['header']['source_commit']['commit'], report['header']['constructor_note']),
            (path, 'abc123def', tool.CONSTRUCTOR_NOTE))
        self.assertIn('Codex\'s constructor batch 2 (own PP carry, removed items, sleep clocks)', tool.CONSTRUCTOR_NOTE)
        md = report['summary_markdown']
        for text in (path['timed'], 'abc123def', tool.CONSTRUCTOR_NOTE, '<= 800 ms', 'learner_max, hedge', 'fake_numpy_worlds', '| depth3-prior-k3 | 2 | sequential | fake | 0 |',
                'pool 16: startup 3.5 s (excluded)', hashlib.sha256(b'stub hedge').hexdigest(), 'commit 583555d2', 'seed 2026110500'):
            self.assertIn(text, md)
        self.assertEqual((report['header']['hedge']['sha256'], report['header']['solver_pools'][0]['workers']), (hashlib.sha256(b'stub hedge').hexdigest(), 16))
        self.assertTrue(report['cycled']);json.dumps(report, allow_nan=False, default=str)

    def test_host_prep_rebuilds_exactly_c_roots_with_one_detach_per_step(self):
        from gpu_engine.device_public_teacher_batch import Request
        frames = chunk_frames();detached, planned = [], []
        def detach_roots(context, source, *, rows, owners, enable_device_worlds):
            detached.append((int(context.streams[0]), tuple(rows), tuple(owners), source, enable_device_worlds))
            return tuple(SimpleNamespace(address=SimpleNamespace(stream=int(context.streams[r]), frame=int(context.frames[r]), owner=o)) for r, o in zip(rows, owners))
        def plan_starts(root, source, recipe):
            planned.append(recipe);return SimpleNamespace(root=root, selected=True, starts=())
        device_roots = RootContext(streams=frames.streams.copy(), frames=frames.frames.copy())
        ctx = SimpleNamespace(chunk=SimpleNamespace(roots=device_roots, frames=frames), provider=SimpleNamespace(source='source', recipe='declared'), head_override=False)
        slots = [(1, 2, 0), (0, 1, 1), (1, 0, 1), (1, 2, 0)]   # cycled: the first root repeats
        bound = [SimpleNamespace(root_logits=(9.,) * 9, learner_actor=0, opponent_actor=0)] * 4
        with patch.object(real_worlds, 'detach_roots', detach_roots), patch.object(real_worlds, 'plan_starts', plan_starts):
            requests, parts = tool.host_prep(ctx, SimpleNamespace(slots=slots, bound=bound))
            # One batched pass per scan step, steps in first-seen order and pairs in root order (as _prepare_batched's `batch` dict).
            self.assertEqual(detached, [(103, (2, 0, 2), (0, 1, 0), 'source', True), (100, (1,), (1,), 'source', True)])
            self.assertEqual(planned, ['declared'] * 4);self.assertEqual(set(parts), {'detach', 'plan_starts', 'assemble'})
            self.assertTrue(all(type(r) is Request for r in requests))
            self.assertEqual([(r.plans.root.address.stream, r.plans.root.address.owner) for r in requests], [(105, 0), (101, 1), (103, 1), (105, 0)])
            first = requests[0];self.assertEqual(first.root_logits, tuple(float(x) for x in frames.raw_policy_logits[1, 2, 0]))
            self.assertEqual((first.learner_actor, first.opponent_actor), (int(frames.actor_indices[1, 2, 0]), int(frames.actor_indices[1, 2, 1])))
            self.assertEqual(first.root_legal, (True,) * 4 + (False,) * 5);self.assertEqual(sorted(first.root_inputs), sorted(('history_mask',) + tuple(
                ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask'))))
            self.assertEqual(first.root_inputs['categorical_ids'].shape[:2], (1, 1));self.assertTrue(first.root_inputs['history_mask'].all())
            ctx.head_override = True   # the gate's head override: the bound head prior and actors 0/0
            requests, _ = tool.host_prep(ctx, SimpleNamespace(slots=slots, bound=bound))
            self.assertEqual([(r.root_logits, r.learner_actor, r.opponent_actor) for r in requests], [((9.,) * 9, 0, 0)] * 4)
            moved = ctx.chunk.roots.streams;moved[1, 2] += 1   # the device context disagrees with the frames' address
            with self.assertRaises(ValueError):tool.host_prep(ctx, SimpleNamespace(slots=slots, bound=bound))

    def test_check_prepared_and_the_chunk_capture_hook(self):
        start = lambda seed: SimpleNamespace(world=0, seed=seed, refusals=(), revealed_source_indices=((0, 1),))
        make = lambda seed, logits=(0.,) * 9: SimpleNamespace(plans=SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=1, frame=2, owner=0)),
            selected=True, starts=(start(seed),)), root_logits=logits, root_legal=(True,) * 9, learner_actor=0, opponent_actor=0)
        tool.check_prepared([make(5)], [make(5)])
        for other in (make(6), make(5, (1.,) * 9)):
            with self.assertRaises(ValueError):tool.check_prepared([make(5)], [other])
        with self.assertRaises(ValueError):tool.check_prepared([make(5)], [])
        # The hook: gate._probe (after the bank's JAX import) wraps Provider._prepare_batched; both are restored afterwards.
        from gpu_engine.v5_device_teacher import Provider
        probe, given = gate._probe, []
        def stub(self, device_roots, frames, bank):given.append((device_roots, frames, bank));return 'records', 'grouped'
        with patch.object(Provider, '_prepare_batched', stub):
            captured = {}
            with tool.capture_chunk(captured):
                self.assertIsNot(gate._probe, probe)
                with patch.object(gate, '_probe', gate._probe), patch('gpu_engine.d3_depth2_gate.log'):
                    gate._probe(SimpleNamespace(_probe=lambda runtime: None), {}, None)
                self.assertEqual(Provider._prepare_batched(None, 'roots', 'frames', 'bank'), ('records', 'grouped'))
            self.assertEqual((captured['roots'], captured['frames'], captured['bank']), ('roots', 'frames', 'bank'));self.assertEqual(given, [('roots', 'frames', 'bank')])
            self.assertIs(gate._probe, probe);self.assertIs(Provider._prepare_batched, stub)
            with self.assertRaises(SystemExit), tool.capture_chunk({}):pass   # nothing captured: the gate's builder changed shape

    def test_depth2_merged_is_the_gates_depth2_pricer_with_fixed_capacity(self):
        args, ctx = self.args(), fake_ctx();ctx.actor_capacity = 3
        search = tool.Search('depth2-merged', 4, args, ctx, depth2_module=DEPTH2)
        self.assertEqual(search.pricer.given, dict(capacity=4, actor_capacity=3, k_own=9, k_opp=9, k1_own=9, k1_opp=9, digest=True, fixed_capacity=True))
        self.assertEqual(search.pricer.child_block, 27)   # never passed: the pricer's own default, as the gate's --pricer-only without --child-block
        self.assertEqual((search.options['engine'], search.options['child_block'], search.options['digest']), ('tests.test_gpu_engine_search_latency.FakeDepth2', 27, True))
        self.assertEqual(search.widths(), dict(k_own=(9,), k_opp=(9,), nominal_leaves_per_world=6561))
        record, _ = tool.run_call(search, ctx, ctx.roots(4), 2, 'sequential', solver=FakeSolver(), timer=CLOCK)
        s, p = record['stages'], record['parts']
        self.assertAlmostEqual(sum(s.values()), record['wall_seconds'], places=12);self.assertAlmostEqual(s['compile'], 2.)   # from the pricer's timing
        self.assertAlmostEqual(s['world_construction'], .01);self.assertAlmostEqual(p['pricer.depth2_merged'], .012);self.assertAlmostEqual(s['pricer'], .012)
        self.assertEqual(record['pricing'][0]['depthn']['compiles'], [dict(key="('depth2_kernel', 4, 27)", lower_seconds=.5, compile_seconds=1.5)])
        self.assertIsNone(record['pricing'][0]['depthn']['p1_seconds']);self.assertEqual(record['enumerated_leaves'] / (4 * 2), 6561)
        record, _ = tool.run_call(search, ctx, ctx.roots(4), 2, 'sequential', solver=FakeSolver(), timer=CLOCK);self.assertEqual(record['stages']['compile'], 0.)
        # W=8 sequential and packed, one pricer per (config, pricer capacity); the row records the engine and the solve it used.
        args = self.args('--configs', 'depth2-merged@8', '--capacities', '1,4', '--repeats', '2', '--solver', 'tests.test_gpu_engine_search_latency:pessimistic')
        plugged = FakeSolver();plugged.spec = args.solver   # a plug-in solver: the merged rows still record depth2_backups learner-max
        series = tool.measure_config('depth2-merged', (8,), args, fake_ctx(), None, solver=plugged, depth2_module=DEPTH2,
            memory=lambda: {}, timer=CLOCK)
        self.assertEqual([(s['capacity'], s['mode'], s['pricer_capacity']) for s in series], [(1, 'sequential', 1), (1, 'packed', 4), (4, 'sequential', 4), (4, 'packed', 16)])
        self.assertEqual([p.capacity for p in FakeDepth2.built[1:]], [1, 4, 16]);self.assertEqual(FakeDepth2.built[2].log[0], (4, 0, [0, 0, 0, 0]))
        self.assertEqual({s['solve'] for s in series}, {tool.MERGED_SOLVE});self.assertEqual(series[0]['leaves']['measured_per_world'], 6561)
        self.assertEqual([s['first_call']['compile_seconds'] for s in series], [2., 2., 0., 2.])
        with self.assertRaises(SystemExit) as raised:tool.Search('depth2-merged', 4, self.args(), fake_ctx(), depth2_module=SimpleNamespace(Depth2Pricer=OldDepth2))
        self.assertIn('fixed_capacity', str(raised.exception.code))
        # The real solve on a real Depth2Frame: depth2_backups learner-max, even with a plug-in rule (it has no per-node hook).
        values = ([[.6, -.2], [.1, .1]], [[-.4, 0.], [.3, .5]]);frame = dn.as_depth2_frame(depth2_frame(values))
        for solver in (tool.Solver(), tool.Solver.load('tests.test_gpu_engine_search_latency:pessimistic')):
            results = solver.solve([(frame, 0)], [ROOT, ROOT], tool.Clock(CLOCK))
            self.assertAlmostEqual(results[0]['completed_q'][0], .2, places=6);self.assertAlmostEqual(results[1]['completed_q'][0], .4, places=6)

    def test_hedge_solve_with_a_stub_search_hedge(self):
        hedge = stub_hedge()
        with patch.dict(sys.modules, {tool.HEDGE_MODULE: hedge}):
            solver = tool.Solver.load('hedge');self.assertIs(solver.hedge, hedge);self.assertEqual(solver.describe()['lam'], 0.)
            # DepthNPricer N=2: depthn_backups.backup with matrix_game on every expanded cell's full 2x2 child grid; bayes_hedge at the root.
            frame = depth2_frame(([[.6, -.2], [.1, .1]], [[-.4, 0.], [.3, .5]]));clock = tool.Clock(CLOCK)
            results = solver.solve([(frame, 0)], ROOTS, clock)
            self.assertEqual([c for c in hedge.calls if c[0] == 'matrix_game'], [('matrix_game', (2, 2))] * 4)   # 2 roots x 2 worlds x 1 cell
            self.assertEqual([c for c in hedge.calls if c[0] == 'bayes_hedge'], [('bayes_hedge', 2, 0.)] * 2)   # one LP per root over its W worlds, lam=0
            self.assertAlmostEqual(results[0]['root_value'], .1, places=6);self.assertAlmostEqual(results[1]['root_value'], .3, places=6)   # the maximin, not learner-max
            self.assertNotIn('solve.as_depth2_frame', clock.parts);self.assertIsNone(results[0].get('policy_target'))
            hedge.calls.clear();moves = solver.moves(results, ROOTS)
            self.assertEqual(moves, [0, 0]);self.assertEqual(hedge.calls[0], ('search_uniform', 2026110500, (100, 7, 0)))   # the pre-registered seed, the root's address
            # W = 4 (two blocks): ONE bayes_hedge over all four worlds; a refused world refuses the root.
            hedge.calls.clear();other = depth2_frame(([[.2, .2], [0., 0.]], [[.1, .1], [.1, .1]]), errored=((1, 0),))
            results = solver.solve([(frame, 0), (other, 0)], ROOTS, tool.Clock(CLOCK))
            self.assertEqual([c for c in hedge.calls if c[0] == 'bayes_hedge'], [('bayes_hedge', 4, 0.)] * 2)
            self.assertAlmostEqual(results[0]['root_value'], (.1 + .2) / 2, places=6);self.assertEqual(results[1]['status'], 'REFUSED')
            self.assertEqual(solver.moves(results, ROOTS)[1], None)
            # N=3: the rule at every node (4 ply-2 nodes + the turn-1 cell per world), through the per-root slices of every ply.
            hedge.calls.clear();deep = depth3_frame([[[[.5, .1], [.2, .3]]] * 4, [[[-.5, .4], [0., -.1]]] * 4])
            results = solver.solve([(deep, 0)], ROOTS, tool.Clock(CLOCK))
            self.assertEqual(len([c for c in hedge.calls if c[0] == 'matrix_game']), 2 * 2 * 5)
            self.assertAlmostEqual(results[0]['root_value'], .2, places=6);self.assertAlmostEqual(results[1]['root_value'], -.1, places=6)
            # The merged Depth2Frame: search_hedge.depth2_hedged_root on the root's DEPTH2_FIELDS slices, once per (root, block).
            hedge.calls.clear();merged = dn.as_depth2_frame(frame)
            results = solver.solve([(merged, 0)], ROOTS, tool.Clock(CLOCK))
            roots = [c for c in hedge.calls if c[0] == 'depth2_hedged_root'];self.assertEqual(roots, [('depth2_hedged_root', tuple(sorted(hedge.DEPTH2_FIELDS)))] * 2)
            self.assertAlmostEqual(results[0]['root_value'], .1, places=6);self.assertAlmostEqual(results[1]['root_value'], .3, places=6)
        with self.assertRaises(SystemExit) as raised:tool.Solver.load('hedge', hedge_module='gpu_engine.no_such_hedge_module')
        self.assertIn('PR #1829', str(raised.exception.code));self.assertIn('583555d2', str(raised.exception.code))

    def test_unit_slices_rebuild_the_frames_backups(self):
        from gpu_engine import depthn_backups
        deep = depth3_frame([[[[.5, .1], [.2, .3]], [[.0, .9], [.4, .4]], [[-.2, .6], [.1, -.3]], [[.7, .7], [-.7, .2]]], [[[-.5, .4], [0., -.1]]] * 4])
        for r in range(2):
            unit = dict(solver='learner_max', hedge=None, kind='depthn', rule=None, fields=tool.root_fields(deep, r, 'depthn'))
            for mine, theirs in zip(tool.world_prices(pickle.loads(pickle.dumps(unit))), depthn_backups.backup_learner_max(deep, r)):
                self.assertEqual(mine.joint_values, theirs.joint_values);self.assertEqual(mine.opponent_logits, theirs.opponent_logits)
        size = lambda frame: len(pickle.dumps(dict(solver='learner_max', hedge=None, kind='depthn', rule=None, fields=tool.root_fields(frame, 0, 'depthn'))))
        self.assertEqual(size(deep), size(depth3_frame([[[[.5, .1], [.2, .3]]] * 4] * 6)))   # one root's slices: independent of the frame's root count

    def test_pool_timing_accounting_on_shared_frames(self):
        hedge, pool = stub_hedge(), FakePool();args, ctx = self.args(), fake_ctx()
        with patch.dict(sys.modules, {tool.HEDGE_MODULE: hedge}):
            variants = tool.make_variants([tool.Solver(), tool.Solver.load('hedge')], {0: None, 4: pool})
            self.assertEqual([v.name for v in variants], ['learner_max@pool0', 'learner_max@pool4', 'hedge@pool0', 'hedge@pool4'])
            search = tool.Search('depth2-fw', 2, args, ctx, module=FRAMES)
            record, _ = tool.run_call(search, ctx, ctx.roots(2), 4, 'sequential', variants=variants, rotate=1, timer=CLOCK)
        s, v = record['stages'], record['variants'];common = s['host_prep'] + s['world_construction'] + s['pricer']
        self.assertEqual(len(search.pricer.log), 2)   # priced once (two blocks), solved by all four variants
        self.assertEqual(record['variant_order'], ['learner_max@pool4', 'hedge@pool0', 'hedge@pool4', 'learner_max@pool0'])   # rotated per call
        self.assertAlmostEqual(sum(s.values()), record['wall_seconds'], places=12);self.assertAlmostEqual(record['common_seconds'], common, places=12)
        self.assertAlmostEqual(s['solve'] + s['return'], sum(x['solve_seconds'] + x['return_seconds'] for x in v.values()), places=12)
        for name, x in v.items():
            self.assertAlmostEqual(x['total_seconds'], common + x['solve_seconds'] + x['return_seconds'], places=12)
            self.assertIn('solve.worlds_pool' if x['pool'] else 'solve.worlds', x['parts'])
        self.assertAlmostEqual(v['hedge@pool4']['parts']['solve.worlds_pool'], .001 * 2 * 2);self.assertAlmostEqual(v['hedge@pool0']['solve_seconds'], 0.)   # IPC is in solve
        self.assertEqual([(fn, n, chunk) for fn, n, chunk in pool.mapped], [(tool.world_prices, 4, 1)] * 2)   # (root, block) units, both solvers
        self.assertEqual(v['learner_max@pool0']['moves'], v['learner_max@pool4']['moves']);self.assertEqual(v['hedge@pool0']['moves'], v['hedge@pool4']['moves'])
        self.assertEqual((v['learner_max@pool0']['labeled'], v['hedge@pool4']['labeled']), (2, 2))
        # A series per variant from the same calls; each total is its own, the first call is excluded from every one.
        CLOCK.now = 100.;memory = lambda: {}
        with patch.dict(sys.modules, {tool.HEDGE_MODULE: hedge}):
            group = tool.measure_series(tool.Search('depth2-fw', 2, args, ctx, module=FRAMES), ctx, ctx.roots(2), 2, 'sequential', 2, variants=variants,
                progress=None, memory=memory, baseline={}, capacity=2, timer=CLOCK)
        self.assertEqual([(g['solver'], g['pool'], g['variant']) for g in group], [('learner_max', 0, 'learner_max@pool0'), ('learner_max', 4, 'learner_max@pool4'),
            ('hedge', 0, 'hedge@pool0'), ('hedge', 4, 'hedge@pool4')])
        for g in group:
            self.assertEqual(g['warm_calls'], 2)
            for c in g['calls']:self.assertAlmostEqual(c['total_seconds'], sum(c['stages'][k] for k in tool.STAGES), places=12)
        self.assertGreater(group[3]['warm']['solve']['median_ms'], group[2]['warm']['solve']['median_ms'])   # the pool's pickling/IPC shows in its solve
        self.assertEqual(group[2]['solve'], tool.HEDGE_SOLVE);self.assertEqual(group[0]['solve'], 'learner_max')
        rows = tool.select(group)['rows'];self.assertEqual(len(rows), 4);self.assertEqual({r['variant'] for r in rows}, {v.name for v in variants})

    def test_real_spawn_pool_matches_in_process(self):
        pool, info = tool.start_pool(2)
        try:
            self.assertEqual((info['workers'], info['answered'], info['start_method']), (2, 2, 'spawn'));self.assertGreater(info['startup_seconds'], 0.)
            deep = depth3_frame([[[[.5, .1], [.2, .3]]] * 4, [[[-.5, .4], [0., -.1]]] * 4]);flat = depth2_frame(([[.6, -.2], [.1, .1]], [[-.4, 0.], [.3, .5]]))
            for solver, views in ((tool.Solver(), [(deep, 0)]), (tool.Solver(), [(flat, 0), (flat, 0)]),
                    (tool.Solver.load('gpu_engine.depthn_backups:learner_max'), [(deep, 0)])):
                pooled, clock = solver.solve(views, ROOTS, tool.Clock(), pool=pool), tool.Clock()
                self.assertEqual(pooled, solver.solve(views, ROOTS, clock));self.assertEqual(pooled[0]['status'], 'LABELED')
        finally:pool.terminate();pool.join()

    def test_hedge_header_records_the_file_sha256_and_commit(self):
        path = self.hedge_file(b'not the 583555d2 file\n');hedge = stub_hedge(path)
        head = tool.hedge_header(hedge, tool.HEDGE_COMMIT)
        self.assertEqual((head['module'], head['file'], head['sha256'], head['commit'], head['reference_sha256'], head['matches_reference']),
            (tool.HEDGE_MODULE, str(path), hashlib.sha256(b'not the 583555d2 file\n').hexdigest(), '583555d2', tool.HEDGE_REFERENCE_SHA256, False))
        self.assertEqual((head['lam'], head['seed'], head['weights'], head['iterations']), (0., 2026110500, '1/W', 0))
        self.assertIsNone(tool.hedge_header(hedge, 'f00dcafe')['matches_reference'])   # another commit: 583555d2's sha256 does not apply
        self.assertIsNone(tool.hedge_header(stub_hedge(), tool.HEDGE_COMMIT)['sha256'])   # no file (the in-memory stub)

    def test_isolate_argv(self):
        args = self.args('--isolate', '--configs', 'depth2-fw@2,depth3-prior-k3@2');calls = []
        def call(argv):
            calls.append(argv);out = Path(argv[argv.index('--out') + 1]);out.mkdir(parents=True)
            (out / 'latency.json').write_text(json.dumps(dict(sections={argv[argv.index('--only') + 1]: []}, context=dict(evaluator={}))));return 0
        argv = ['--d3-root', 'r', '--chunk', '0', '--out', 'o', '--isolate', '--configs', 'depth2-fw@2,depth3-prior-k3@2']
        with patch('subprocess.call', side_effect=call):sections, contexts = tool.isolated(args, argv, Path(tempfile.mkdtemp()))
        self.assertEqual(list(sections), ['depth2-fw', 'depth3-prior-k3']);self.assertEqual(len(calls), 2)
        for c in calls:
            self.assertNotIn('--isolate', c);self.assertEqual(c.count('--out'), 1);self.assertIn('gpu_engine.d3_search_latency', c)
        self.assertEqual(calls[1][calls[1].index('--only') + 1], 'depth3-prior-k3')

    def test_argument_parsing_and_refusing_to_overwrite_out(self):
        args = self.args()
        self.assertEqual(args.config_worlds, {'depth2-fw': (2, 8), 'depth2-merged': (2, 8), 'depth3-prior-k3': (2,), 'depth3-searched-k3': (2,)})
        self.assertEqual(list(args.config_worlds)[:2], ['depth2-fw', 'depth2-merged'])   # the merged pricer's rows sit next to depth2-fw
        self.assertEqual((args.capacity_values, args.repeats, args.solver_specs, args.pool_sizes, args.packed, args.hedge_commit),
            ((1, 4, 16), 10, ('learner_max',), (0,), True, '583555d2'))
        args = self.args('--configs', 'depth3-prior-k2,depth2-k4@4/2', '--worlds', '2,8', '--capacities', '8,2,8', '--no-packed',
            '--solver', 'tests.test_gpu_engine_search_latency:pessimistic')
        self.assertEqual(args.config_worlds, {'depth3-prior-k2': (2, 8), 'depth2-k4': (2, 4)});self.assertEqual(args.capacity_values, (2, 8))
        self.assertEqual(args.solver_specs, ('tests.test_gpu_engine_search_latency:pessimistic',));self.assertFalse(args.packed)
        args = self.args('--solver', 'learner_max,hedge', '--solver-pool', '16,0', '--hedge-module', 'tests.test_gpu_engine_search_latency', '--hedge-commit', 'abc')
        self.assertEqual((args.solver_specs, args.pool_sizes, args.hedge_module, args.hedge_commit), (('learner_max', 'hedge'), (0, 16), 'tests.test_gpu_engine_search_latency', 'abc'))
        existing = tempfile.mkdtemp()
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            for extra in (['--configs', 'depth1'], ['--configs', 'depth2-searched-k3'], ['--configs', 'depth3-prior-k10'], ['--configs', 'depth2-fw@3'],
                    ['--configs', 'bogus'], ['--configs', 'depth2-fw,depth2-fw'], ['--capacities', '0,4'], ['--repeats', '0'], ['--solver', 'nocolon'],
                    ['--solver', 'no_such_module_xyz:f'], ['--only', 'depth3-prior-k2'], ['--isolate', '--only', 'depth2-fw'], ['--compile-cache', 'relative/dir'],
                    ['--head-archive', 'a'], ['--solver', 'hedge,hedge'], ['--solver', 'learner_max,'], ['--solver-pool', '-1'], ['--solver-pool', 'x'],
                    ['--solver', 'learner_max,hedge', '--hedge-module', 'gpu_engine.no_such_hedge_module']):
                with self.assertRaises(SystemExit) as raised:tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', self.out(), *extra])
                self.assertEqual(raised.exception.code, 2, extra)
            with self.assertRaises(SystemExit) as raised:tool.parse(['--d3-root', 'r', '--chunk', '0', '--out', existing])
            self.assertEqual(raised.exception.code, 2)
            with self.assertRaises(SystemExit) as raised:tool.main(['--d3-root', 'r', '--chunk', '0', '--out', existing])   # never overwritten
            self.assertEqual(raised.exception.code, 2)
        self.assertIn('never overwrites', stderr.getvalue());self.assertIn('PR #1829', stderr.getvalue())
        out = self.out()
        with patch('importlib.util.find_spec', return_value=None), self.assertRaises(SystemExit) as raised:tool.main(['--d3-root', 'r', '--chunk', '0', '--out', out])
        self.assertIn('device_depthn_pricer', str(raised.exception.code));self.assertFalse(Path(out).exists())


if __name__ == '__main__':unittest.main()
