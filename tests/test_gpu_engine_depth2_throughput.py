"""Depth-2 throughput CLI: argument parsing, the per-block and derived arithmetic, and the split seam (CPU)."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import d3_depth2_throughput as throughput
from gpu_engine.v5_model import Output

REQUIRED = ['--d3-root', 'bank', '--chunk', '0', '--out', 'new', '--capacity', '4']


class FakeDepth2:
    """Pricer timing as Depth2Pricer reports it: compile on the first call only, then a fixed device sequence."""
    def __init__(self, device, leaves=(1000, 2187)):
        self.device = list(device);self.leaves = leaves;self.calls = []

    def price(self, requests, materializer, recipe, params):
        self.calls.append((tuple(requests), recipe));first = len(self.calls) == 1
        timing = dict(lower_seconds=2. if first else 0., compile_seconds=3. if first else 0., device_seconds=self.device[len(self.calls) - 1],
            turn2_enumerated_leaves=self.leaves[0], turn2_evaluated_leaf_lanes=self.leaves[1], frame_wall_seconds=9.)
        return SimpleNamespace(timing=timing)


class FakeDepth1:
    def __init__(self, device):
        self.device = list(device);self.calls = 0;self.last_timing = None

    def price(self, requests, materializer, recipe, params):
        self.calls += 1
        self.last_timing = dict(lower_seconds=0., compile_seconds=1. if self.calls == 1 else 0., device_seconds=self.device[self.calls - 1],
            evaluated_joint_slots=162 * len(requests), frame_wall_seconds=2.)
        return [SimpleNamespace(world_reports=(dict(required_pairs=6), dict(required_pairs=None))) for _ in requests]


class ParseTests(unittest.TestCase):
    def test_defaults_and_required(self):
        args = throughput.parse(REQUIRED)
        self.assertEqual((args.capacity, args.repeats, args.worlds_blocks, args.k_own, args.child_block, args.d1, args.profile_split),
            (4, 3, 1, 9, None, False, False))
        args = throughput.parse(REQUIRED + ['--k-own', '3', '--child-block', '27', '--worlds-blocks', '4', '--repeats', '5', '--max-roots', '10',
            '--d1', '--profile-split', '--memory-policy'])
        self.assertEqual((args.k_own, args.child_block, args.worlds_blocks, args.repeats, args.max_roots, args.d1, args.profile_split, args.memory_policy),
            (3, 27, 4, 5, 10, True, True, True))

    def test_head_override_flags(self):
        head = ['--head-archive', '/shared/x/actor-99.npz', '--head-archive-sha256', '1' * 64, '--head-checkpoint-sha256', '2' * 64,
            '--head-model-sha256', '3' * 64, '--value-reference', 'B1 iter~95']
        args = throughput.parse(REQUIRED + head)
        self.assertEqual((args.head_archive, args.head_model_sha256, args.value_reference), ('/shared/x/actor-99.npz', '3' * 64, 'B1 iter~95'))
        self.assertIn('label only', throughput.WEIGHTS_NOTE)
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (REQUIRED + head[:2], REQUIRED + ['--head-category-sha256', 'c' * 64]):
                with self.assertRaises(SystemExit) as raised:throughput.parse(argv)
                self.assertEqual(raised.exception.code, 2, argv)

    def test_invalid_arguments_exit_2(self):
        bad = (REQUIRED[:-2], REQUIRED + ['--capacity', '0'], REQUIRED + ['--repeats', '0'], REQUIRED + ['--k-opp', '10'],
            REQUIRED + ['--worlds-blocks', '0'], REQUIRED + ['--child-block', '0'], REQUIRED + ['--max-roots', '0'],
            ['--d3-root', 'b', '--chunk', '-1', '--out', 'o', '--capacity', '2'])
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in bad:
                with self.assertRaises(SystemExit) as raised:throughput.parse(argv)
                self.assertEqual(raised.exception.code, 2, argv)

    def test_main_fails_before_loading_without_the_pricer(self):
        out = Path(tempfile.gettempdir()) / f'depth2-throughput-never-created-{uuid.uuid4().hex}'
        argv = ['--d3-root', 'b', '--chunk', '0', '--out', str(out), '--capacity', '4']
        with patch.object(importlib.util, 'find_spec', return_value=None), self.assertRaises(SystemExit) as raised:throughput.main(argv)
        self.assertIn('device_depth2_pricer', str(raised.exception.code));self.assertFalse(out.exists())


class ArithmeticTests(unittest.TestCase):
    def test_roots_are_taken_in_order_and_cycled(self):
        self.assertEqual(throughput.cycle_roots('abc', 7), list('abcabca'));self.assertEqual(throughput.cycle_roots('abcde', 2), ['a', 'b'])
        with self.assertRaises(ValueError):throughput.cycle_roots([], 3)

    def test_blocks_cold_then_warm_and_derived_rates(self):
        entered = []
        @contextlib.contextmanager
        def context(block):
            entered.append(block);yield
        blocks = [(0, 'recipe0', ['r'] * 4, lambda: context(0)), (1, 'recipe1', ['s'] * 4, lambda: context(1))]
        pricer = FakeDepth2([1.0, .3, .2, .4, .9, .5, .6, .4]);memory = iter([dict(peak_bytes_in_use=p, bytes_in_use=p // 2) for p in range(10, 90, 10)])
        lines = []
        result = throughput.measure('depth2', throughput.depth2_call, pricer, blocks, 'materializer', 'params', 3,
            memory=lambda: next(memory), progress=lambda m, **v: lines.append((m, v)))
        self.assertEqual(entered, [0] * 4 + [1] * 4);self.assertEqual([c[1] for c in pricer.calls], ['recipe0'] * 4 + ['recipe1'] * 4)
        first, second = result
        self.assertEqual((first['cold_compile_seconds'], first['cold_device_seconds'], first['warm_device_seconds']), (5., 1., [.3, .2, .4]))
        self.assertEqual((first['warm_device_seconds_min'], first['warm_device_seconds_median']), (.2, .3))
        self.assertEqual((second['cold_compile_seconds'], second['warm_device_seconds_median']), (0., .5))
        self.assertEqual((first['leaves_enumerated'], first['leaves_evaluated'], first['leaves_constant']), (1000, 2187, True))
        self.assertEqual((first['peak_bytes_in_use'], first['bytes_in_use_after'], second['peak_bytes_in_use']), (40, 20, 80))
        derived = throughput.derive(result, 4)
        self.assertAlmostEqual(derived['device_ms_per_root_median'], 1e3 * (.3 + .5) / 4)
        self.assertAlmostEqual(derived['device_ms_per_root_min'], 1e3 * (.2 + .4) / 4)
        self.assertAlmostEqual(derived['leaves_enumerated_per_second_median'], 2000 / .8)
        self.assertAlmostEqual(derived['leaves_evaluated_per_second_min'], 2 * 2187 / .6)
        self.assertEqual((derived['worlds'], derived['leaves_enumerated'], derived['roots_per_call']), (4, 2000, 4))
        starts = [v for m, v in lines if m == 'STAGE_START'];ends = [v for m, v in lines if m == 'STAGE_END']
        self.assertEqual(len(starts), 8);self.assertEqual([v['warm'] for v in ends[:4]], [False, True, True, True])
        self.assertEqual(ends[0]['compile_seconds'], 5.)

    def test_depth1_counts_required_pairs_and_evaluated_lanes(self):
        blocks = [(0, 'recipe', ['r'] * 3, contextlib.nullcontext)]
        result = throughput.measure('depth1', throughput.depth1_call, FakeDepth1([.5, .1, .2]), blocks, None, None, 2,
            memory=lambda: dict(peak_bytes_in_use=None, bytes_in_use=None), progress=None)[0]
        self.assertEqual((result['leaves_enumerated'], result['leaves_evaluated'], result['cold_compile_seconds']), (18, 486, 1.))
        self.assertAlmostEqual(result['warm_device_seconds_median'], .15);self.assertIsNone(result['peak_bytes_in_use'])
        derived = throughput.derive([result], 3)
        self.assertAlmostEqual(derived['device_ms_per_root_median'], 50.);self.assertAlmostEqual(derived['leaves_enumerated_per_second_median'], 120.)

    def test_split_estimate(self):
        self.assertEqual(throughput.split_estimate(1., .4, .1), dict(normal_seconds=1., constant_turn2_forward_seconds=.4, depth1_seconds=.1,
            turn2_forward_seconds_estimate=.6, turn2_step_observe_and_child_prior_seconds_estimate=.4 - .1, depth2_minus_depth1_seconds=.9))
        skipped = throughput.split_estimate(1., None, .1)
        self.assertNotIn('turn2_forward_seconds_estimate', skipped);self.assertEqual(skipped['depth2_minus_depth1_seconds'], .9)


class SplitSeamTests(unittest.TestCase):
    def test_turn2_batches_are_identifiable_unless_they_collide(self):
        rows = throughput.turn2_rows(27, 9, 9)
        self.assertEqual(rows, 2187);self.assertIsNone(throughput.split_check(rows, 64))
        self.assertIn(162 * 64, throughput.turn1_rows(64));self.assertIn(81 * 32, throughput.turn1_rows(64))
        collide = throughput.turn2_rows(32, 9, 9)   # 32 * 81: a 32-row archive group's turn-1 leaf batch
        self.assertIn('can equal', throughput.split_check(collide, 16))
        self.assertIsNone(throughput.split_check(rows, 64, census=[128, 128 * 81, rows, 64 * 81]))
        self.assertIn('not seen', throughput.split_check(rows, 64, census=[128, 128 * 81]))
        self.assertIn('unexpected', throughput.split_check(rows, 64, census=[128, rows, 77]))

    def test_constant_forward_zeroes_turn2_batches_only(self):
        def real(params, inputs):
            x = inputs['numeric_features'][:, 0, 0, :9] + params
            return Output(x, j.tanh(x[:, 0]), x[:, :3] * 2, j.ones(x.shape[0], bool))
        census = [];forward = jax.jit(throughput.constant_turn2_forward(real, 6, census))
        inputs = lambda n: dict(numeric_features=j.ones((n, 1, 2, 12), j.float32), categorical_ids=j.ones((n, 1, 2, 3), j.int32))
        zeroed, kept = forward(j.float32(.5), inputs(6)), forward(j.float32(.5), inputs(4))
        self.assertEqual(census, [6, 4])   # traced once per batch size
        expected = real(j.float32(.5), inputs(6))
        for got, want in zip(zeroed, expected):
            self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
        self.assertTrue(np.all(np.asarray(zeroed.policy_logits) == 0) and np.all(np.asarray(zeroed.value) == 0))
        self.assertTrue(np.all(np.asarray(zeroed.value_logits) == 0) and np.all(np.asarray(zeroed.valid)))
        for got, want in zip(kept, real(j.float32(.5), inputs(4))):np.testing.assert_array_equal(got, want)

    def test_pricer_needs_fixed_capacity_and_passes_the_seam(self):
        class Old:
            def __init__(self, executor, model, *, capacity=2048, child_block=27, forward=None):pass
        class New:
            def __init__(self, executor, model, *, capacity=2048, actor_capacity=33, k_own=9, k_opp=9, k1_own=9, k1_opp=9, child_block=27,
                    digest=False, forward=None, fixed_capacity=False):
                self.options = dict(capacity=capacity, fixed_capacity=fixed_capacity, child_block=child_block, forward=forward, digest=digest)
                self.child_block = child_block
        widths = dict(k_own=9, k_opp=9, k1_own=9, k1_opp=9)
        make = lambda cls, **kw: throughput.make_depth2('ex', 'model', capacity=8, widths=widths, actor_capacity=33, **kw)
        with patch.dict(sys.modules, {'gpu_engine.device_depth2_pricer': SimpleNamespace(Depth2Pricer=Old)}), self.assertRaises(SystemExit):
            make(Old, child_block=None)
        with patch.dict(sys.modules, {'gpu_engine.device_depth2_pricer': SimpleNamespace(Depth2Pricer=New)}):
            pricer, options = make(New, child_block=None)
            self.assertEqual((pricer.options['capacity'], pricer.options['fixed_capacity'], pricer.options['digest'], options['child_block']), (8, True, False, 27))
            seam = object();pricer, options = make(New, child_block=81, forward=seam)
            self.assertIs(pricer.options['forward'], seam);self.assertEqual(options['child_block'], 81);self.assertNotIn('forward', options)


if __name__ == '__main__':unittest.main()
