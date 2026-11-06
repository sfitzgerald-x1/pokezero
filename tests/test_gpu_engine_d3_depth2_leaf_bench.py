"""The depth-2 leaf bench's CPU-testable parts (gpu_engine.d3_depth2_leaf_bench).

The generated CUDA module needs nvcc and a GPU; here the lane function runs through JAX, standing in
for it, so the bench's turn-1 stage, child selection, lane packing, generated lane layout (repeat,
flatten, reshape back) and comparator are checked against the depth-2 kernel's own turn-2 joint on
the inert two-ply worlds of tests.test_gpu_engine_depth2_pricer.
"""
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import d3_depth2_leaf_bench as bench, depth2_digest, v2e_modules
from tests.test_gpu_engine_depth2_pricer import executor, State, Public


def world(mode, owner, seed, stream):
    state = State(j.asarray([.25], j.float32), j.asarray([False]), j.asarray([0], j.int32), j.asarray([mode], j.int32),
        j.asarray([seed], j.uint32), j.asarray([owner], j.int32), j.asarray([0], j.int32), j.asarray([-1], j.int32))
    return state, Public(j.asarray(float(stream % 79), j.float32), j.asarray([0], j.int32))


def turn1_data(ex, modes, owners):
    """The bench's turn-1 input dict for inert worlds: expected root inputs and legal masks from the executor itself,
    the pricing order = the owner's legal actions ascending (full width)."""
    worlds = [world(m, o, 11 + i, 3 + i) for i, (m, o) in enumerate(zip(modes, owners))]
    stack = lambda values: jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *values)
    n = len(worlds);expected, legal = [], [];order = np.zeros((n, 9), np.int32);size = np.ones(n, np.int32)
    for i, (state, public) in enumerate(worlds):
        masks = ex.legal(state);own = ex.observe(state, public, j.asarray([owners[i]], j.int32), masks[:, owners[i]])
        expected.append(jax.device_get(own.inputs));mask = np.asarray(masks[0, owners[i]]);legal.append(mask)
        acts = tuple(int(a) for a in np.flatnonzero(mask));order[i] = acts + (acts[0],) * (9 - len(acts));size[i] = len(acts)
    return dict(states=stack([w[0] for w in worlds]), publics=stack([w[1] for w in worlds]), expected=stack(expected),
        owner=np.asarray(owners, np.int32), legal=np.asarray(legal), order=order, order_size=size)


# Modes (tests.test_gpu_engine_depth2_pricer): 0 plain, 1 terminal win at turn 1, 3 invalid turn-1 pair (1,1), 29 the
# opponent's child view errors (code 5), 32 all nine actions legal at the root, 20 waiting learner at the child,
# 21 waiting opponent at the child.
MODES = (0, 1, 3, 29, 32, 20, 21)
OWNERS = (0, 1, 0, 1, 0, 1, 0)


class LanePairTest(unittest.TestCase):
    def test_slots_follow_the_kernels_ranked_at_a_uniform_prior(self):
        own = np.array([[1, 1, 0, 1, 0, 0, 0, 0, 0]], bool);peer = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0]], bool)
        pairs = bench.lane_pairs(own, peer, [True], [True])
        self.assertEqual((pairs.shape, pairs.dtype), ((1, 81, 2), np.int32))
        grid = pairs[0].reshape(9, 9, 2)
        np.testing.assert_array_equal(grid[:, 0, 0], [0, 1, 3, 0, 0, 0, 0, 0, 0])   # own slots: legal ascending, dead slots repeat the first
        np.testing.assert_array_equal(grid[0, :, 1], [1, 2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(set(map(tuple, pairs[0].tolist())), {(a, r) for a in (0, 1, 3) for r in (1, 2)})
        np.testing.assert_array_equal(bench.lane_pairs(own, peer, [False], [True])[0, :, 0], 0)   # a waiting seat enumerates 0
        np.testing.assert_array_equal(bench.lane_pairs(own, peer, [True], [False])[0, :, 1], 0)
        np.testing.assert_array_equal(bench.lane_pairs(np.zeros((1, 9), bool), peer, [True], [True])[0, :, 0], 0)   # requested, nothing legal

    def test_matches_the_kernels_ranked_code_with_zero_logits(self):
        def ranked(mask, logits, k, requested):   # device_depth2_pricer.make_depth2_kernel's `ranked`, verbatim
            order = j.argsort(j.where(mask & j.isfinite(logits), -logits, j.inf), axis=-1, stable=True)
            count = j.where(requested, j.minimum(j.sum(mask, axis=-1), k), 1)
            slot = j.arange(k);live = slot < count[..., None]
            chosen = j.take_along_axis(order, j.where(live, slot, 0), axis=-1)
            return j.where(requested[..., None], chosen, 0), live
        rng = np.random.default_rng(7);n = 40
        own, peer = rng.random((n, 9)) < .4, rng.random((n, 9)) < .4;own_req, peer_req = rng.random(n) < .8, rng.random(n) < .8
        sel_own, _ = ranked(j.asarray(own), j.zeros((n, 9)), 9, j.asarray(own_req));sel_opp, _ = ranked(j.asarray(peer), j.zeros((n, 9)), 9, j.asarray(peer_req))
        want = np.stack((np.broadcast_to(np.asarray(sel_own)[:, :, None], (n, 9, 9)), np.broadcast_to(np.asarray(sel_opp)[:, None, :], (n, 9, 9))), axis=-1).reshape(n, 81, 2)
        np.testing.assert_array_equal(bench.lane_pairs(own, peer, own_req, peer_req), want)

    def test_geometry_is_checked(self):
        with self.assertRaises(ValueError):bench.lane_pairs(np.zeros((2, 9), bool), np.zeros((3, 9), bool), [True] * 2, [True] * 3)
        with self.assertRaises(ValueError):bench.lane_pairs(np.zeros((2, 8), bool), np.zeros((2, 8), bool), [True] * 2, [True] * 2)


class SelectChildrenTest(unittest.TestCase):
    def test_strided_when_enough_cycled_when_not(self):
        expanded = np.array([0, 1, 1, 0, 1, 1, 1], bool)
        indices, cycled = bench.select_children(expanded, 2);self.assertEqual((indices.tolist(), cycled), ([1, 4], False))
        indices, cycled = bench.select_children(expanded, 5);self.assertEqual((indices.tolist(), cycled), ([1, 2, 4, 5, 6], False))
        indices, cycled = bench.select_children(expanded, 7);self.assertEqual((indices.tolist(), cycled), ([1, 2, 4, 5, 6, 1, 2], True))
        with self.assertRaises(ValueError):bench.select_children(expanded, 0)
        with self.assertRaises(ValueError):bench.select_children(np.zeros(4, bool), 1)


class CompareTest(unittest.TestCase):
    def test_equal_differing_geometry_and_structure(self):
        a = dict(x=np.arange(6, dtype=np.float32).reshape(3, 2), y=dict(z=np.ones((3, 1), bool)), s=np.int32(4))
        self.assertEqual(bench.compare_trees(a, a), {})
        b = jax.tree.map(np.copy, a);b['x'][0, 0] = -0.;b['y']['z'][2, 0] = False
        report = bench.compare_trees(a, b)
        self.assertEqual(sorted(report.values()), [1, 1]);self.assertTrue(all('x' in k or 'z' in k for k in report))
        c = jax.tree.map(np.copy, a);c['s'] = np.int32(5)
        self.assertEqual(list(bench.compare_trees(a, c).values()), [1])
        self.assertIn('geometry', str(bench.compare_trees(a, dict(a, x=a['x'].astype(np.int32)))))
        self.assertIn('geometry', str(bench.compare_trees(a, dict(a, x=a['x'][:2]))))
        self.assertEqual(bench.compare_trees(a, dict(x=a['x'], s=a['s'])), {'': 'structure'})

    def test_nans_compare_by_bits_and_device_arrays_are_accepted(self):
        quiet = np.array([np.nan], np.float32);other = np.array([np.nan], np.float32).view(np.uint32) | np.uint32(1)
        self.assertEqual(bench.compare_trees(dict(v=quiet), dict(v=quiet.copy())), {})
        self.assertEqual(list(bench.compare_trees(dict(v=quiet), dict(v=other.view(np.float32))).values()), [1])
        self.assertEqual(bench.compare_trees(dict(v=j.asarray(quiet)), dict(v=quiet)), {})
        self.assertEqual(bench.compare_trees(dict(e=np.zeros((0, 3))), dict(e=np.zeros((0, 3)))), {})


class ParseTest(unittest.TestCase):
    def parse(self, *extra):
        with redirect_stderr(io.StringIO()):return bench.parse(['--d3-root', 'r', '--chunk', '0', '--out', 'o', *extra])

    def test_defaults_and_refusals(self):
        args = self.parse()
        self.assertEqual((args.lanes, args.repeats, args.max_worlds, args.max_roots), (list(bench.DEFAULT_LANES), 5, 256, None))
        self.assertFalse(args.check_state or args.xla_lanes or args.skip_cuda or args.memory_policy)
        self.assertEqual(self.parse('--lanes', '81', '162', '--repeats', '2', '--check-state').lanes, [81, 162])
        for bad in (('--lanes', '100'), ('--lanes', '81', '81'), ('--lanes', '0'), ('--repeats', '0'), ('--max-worlds', '0'), ('--max-roots', '0'),
                ('--chunk', '-1'), ('--build-workers', '0')):
            with self.assertRaises(SystemExit):self.parse(*bad)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):bench.parse(['--d3-root', 'r', '--chunk', '0'])
        self.assertTrue(all(L % 81 == 0 for L in bench.DEFAULT_LANES))

    def test_summarize(self):
        s = bench.summarize([.002, .001, .003], 81)
        self.assertEqual((s['min_ms'], s['median_ms'], round(s['lanes_per_second_min']), round(s['lanes_per_second_median'])), (1., 2., 81000, 40500))

    def test_module_report_merges_generator_stats_that_carry_mode_and_logs(self):
        from types import SimpleNamespace
        module = SimpleNamespace(name='depth2_leaf_0', arena_words=54922, inputs=[(np.dtype(np.int32), (37, 1, 6))], outputs=[None] * 9,
            stats=dict(trace_s=1., generate_s=2., build_s=3., equations=4, mode='thread', chunks=5))
        generated = SimpleNamespace(module=module, mode='thread', path='/tmp/lib.so')
        report = bench.module_report(generated)
        self.assertEqual((report['name'], report['mode'], report['arena_words'], report['lanes_traced'], report['chunks'], report['build_s']),
            ('depth2_leaf_0', 'thread', 54922, 37, 5, 3.))
        bench.log('MODULE', module='leaf', **dict(report, wall_seconds=1.))   # the keywords the bench logs do not collide


class LaneFunctionTest(unittest.TestCase):
    def test_lane_functions_cover_both_module_sets_and_are_cached_per_tables(self):
        fns = v2e_modules.pricer_lane_functions(*(None,) * 5)
        self.assertEqual(set(fns), set(v2e_modules.PRICER_MODES) | set(v2e_modules.DEPTH2_MODES))
        a, b = object(), object()
        with patch.object(v2e_modules, '_pricer_parts', side_effect=lambda e, o: (None,) * 5):
            index, cached = v2e_modules.pricer_functions(a, b)
            self.assertEqual((index, cached), v2e_modules.pricer_functions(a, b))
            other, fns2 = v2e_modules.pricer_functions(b, a)
        self.assertNotEqual(index, other);self.assertIsNot(cached, fns2)

    def test_dispatch_targets_keep_only_what_map_actions_reads(self):
        from gpu_engine.engine_device_observation import Observation
        full = Observation(*(j.zeros((1, 2)) for _ in range(8)), move_targets=j.zeros((1, 4), j.int32))
        kept = v2e_modules.dispatch_targets(full)
        self.assertEqual([name for name in kept._fields if getattr(kept, name) is not None], ['switch_targets', 'move_targets'])
        self.assertEqual(len(jax.tree.leaves(kept)), 2)


class StagesTest(unittest.TestCase):
    """Turn 1, child selection and both runners on inert worlds; the generated path through a JAX stand-in."""
    @classmethod
    def setUpClass(cls):
        cls.ex, _ = executor();cls.data = turn1_data(cls.ex, MODES, OWNERS)
        cls.out = jax.block_until_ready(bench.make_turn1(cls.ex)(jax.device_put(cls.data)))
        cls.host = {k: np.asarray(v) for k, v in cls.out.items() if k not in ('child_states', 'child_publics')}

    def per_world(self, name):
        return np.asarray(self.host[name]).reshape(len(MODES), 81).sum(axis=1).tolist()

    def test_turn1_codes_and_counts_follow_the_kernel(self):
        self.assertEqual(self.host['pre'].tolist(), [0] * len(MODES))
        self.assertEqual(self.per_world('enumerated'), [6, 6, 6, 6, 81, 6, 6])   # 3 own x 2 replies; mode 32: 9 x 9
        self.assertEqual(self.per_world('expanded'), [6, 0, 5, 0, 81, 6, 6])
        self.assertEqual(self.per_world('done'), [0, 6, 0, 0, 0, 0, 0])
        errors = self.host['error'].reshape(len(MODES), 81)
        self.assertEqual(errors[2].tolist()[:6], [0, 0, 0, 1, 0, 0])   # mode 3: pair (1,1) is ordinal 3 of (0,0),(0,1),(1,0),(1,1),...
        self.assertEqual(errors[3].tolist()[:6], [5] * 6);self.assertEqual(int(errors[(0, 1, 4, 5, 6), :].sum()), 0)
        np.testing.assert_array_equal(self.host['pair1'].reshape(len(MODES), 81, 2)[0, :6], [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(self.host['owner'].reshape(len(MODES), 81)[:, 0].tolist(), list(OWNERS))
        requested = self.host['requested'].reshape(len(MODES), 81, 2)
        self.assertEqual(requested[0, 0].tolist(), [True, True])
        self.assertEqual(requested[5, 0].tolist(), [True, False])    # mode 20, owner 1: the learner (seat 1) waits
        self.assertEqual(requested[6, 0].tolist(), [True, False])    # mode 21, owner 0: the opponent (seat 1) waits
        self.assertEqual((self.host['own_requested'].reshape(-1, 81)[5, 0], self.host['peer_requested'].reshape(-1, 81)[6, 0]), (False, False))
        self.assertEqual(jax.tree.leaves(self.out['child_states'])[0].shape[0], len(MODES) * 81)

    def runner_inputs(self):
        picked = [np.flatnonzero(self.host['expanded'] & (np.arange(len(MODES) * 81) // 81 == w))[i] for w, i in ((0, 0), (2, 4), (4, 40), (5, 0), (6, 3))]
        indices = np.asarray(picked)
        pick = lambda tree: jax.tree.map(lambda x: x[indices], tree)
        states, publics, owners, requested = pick(self.out['child_states']), pick(self.out['child_publics']), pick(self.out['owner']), pick(self.out['requested'])
        own, peer, chance = jax.block_until_ready(bench.make_child_views(self.ex)(states, publics, owners))
        pairs = bench.lane_pairs(self.host['own_legal'][indices], self.host['peer_legal'][indices], self.host['own_requested'][indices], self.host['peer_requested'][indices])
        kernel_args = (states, publics, own, peer, owners, requested, j.asarray(pairs), chance)
        lane_args = (states, publics, v2e_modules.dispatch_targets(own), v2e_modules.dispatch_targets(peer), owners, requested, j.asarray(pairs), chance)
        return kernel_args, lane_args, len(indices)

    def test_generated_layout_through_jax_equals_the_kernels_leaf_joint(self):
        kernel_args, lane_args, children = self.runner_inputs()
        fns = v2e_modules.pricer_lane_functions(self.ex.legal, self.ex.observe, self.ex.dispatch, self.ex.advance, self.ex.sample)
        want = jax.block_until_ready(bench.make_xla_runner(self.ex)(*kernel_args))
        got, word = jax.block_until_ready(bench.make_lane_runner(bench.jax_stand_in(fns['leaf']))(*lane_args))
        self.assertEqual(sorted(want), sorted(bench.OUTPUTS));self.assertEqual(want['invalid'].shape, (children, 81))
        self.assertEqual(want['inputs']['numeric_features'].shape, (children, 81, 1, 1, 1, 3))
        self.assertEqual(bench.compare_trees(want, got), {});self.assertEqual(word.tolist(), [0])
        digest = jax.jit(lambda x: depth2_digest.batched(x, axes=2))
        self.assertEqual(bench.compare_trees(dict(d=digest(want['inputs'])), dict(d=digest(got['inputs']))), {})
        # The lanes do differ from one another (the comparator is not trivially satisfied) and waiting seats are honoured:
        # the mode-20 child (learner waiting) repeats each opponent slot's result over every own slot.
        values = np.asarray(want['inputs']['numeric_features'])[:, :, 0, 0, 0, 0].reshape(children, 9, 9)
        self.assertGreater(len(np.unique(values[0])), 1);np.testing.assert_array_equal(values[3], np.broadcast_to(values[3][:1], (9, 9)))
        self.assertGreater(len(np.unique(values[3])), 1)

    def test_state_variant_adds_the_stepped_state_to_the_same_outputs(self):
        kernel_args, lane_args, children = self.runner_inputs()
        fns = v2e_modules.pricer_lane_functions(self.ex.legal, self.ex.observe, self.ex.dispatch, self.ex.advance, self.ex.sample)
        want = jax.block_until_ready(bench.make_xla_runner(self.ex)(*kernel_args))
        got, _ = jax.block_until_ready(bench.make_lane_runner(bench.jax_stand_in(fns['leaf_state']), names=bench.OUTPUTS + ('state',))(*lane_args))
        self.assertEqual(bench.compare_trees(want, {k: got[k] for k in bench.OUTPUTS}), {})
        self.assertEqual(got['state'].turn.shape, (children, 81, 1));self.assertTrue(bool(j.all(got['state'].turn == 2)))
        self.assertEqual(set(got['state']._fields), set(State._fields))

    def test_select_children_cycles_the_inert_expanded_set(self):
        indices, cycled = bench.select_children(self.host['expanded'], 200)
        self.assertTrue(cycled);self.assertEqual(len(indices), 200);self.assertTrue(self.host['expanded'][indices].all())


if __name__ == '__main__':
    unittest.main()
