"""The frame pricer's generated-CUDA leaves: lane functions, flag plumbing and frame wiring (CPU only).

The generated modules themselves need nvcc and a GPU (gpu_engine.d3_cuda_leaves_equal checks them
bitwise on real D3 frames); here the lane functions run through JAX, standing in for the generated
kernels, so make_kernel's generated branch (lane layout, dispatch targets, reshapes, error words) is
checked byte for byte against its default branch on the inert worlds of the batch tests.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import device_public_teacher as serial, device_public_teacher_batch as batch, v2e_modules
from gpu_engine import device_public_world_state as mapping
from tests.test_gpu_engine_device_public_teacher_batch import FrameBatchTests, executor, forward


def jax_generated(ex):
    """The pricer's lane functions over the inert executor, with the Generated calling convention."""
    fns = v2e_modules.pricer_lane_functions(ex.legal, ex.observe, ex.dispatch, ex.advance, ex.sample)
    return {name: (lambda *args, fn=fn: (fn(*args), j.zeros(1, j.uint32))) for name, fn in fns.items()}


class LaneFunctionTest(unittest.TestCase):
    def test_dispatch_targets_keep_only_what_map_actions_reads(self):
        from gpu_engine.engine_device_observation import Observation
        full = Observation(*(j.zeros((1, 2)) for _ in range(8)), move_targets=j.zeros((1, 4), j.int32))
        kept = v2e_modules.dispatch_targets(full)
        self.assertEqual([name for name in kept._fields if getattr(kept, name) is not None], ['switch_targets', 'move_targets'])
        self.assertEqual(len(jax.tree.leaves(kept)), 2)

    def test_pricer_functions_are_cached_per_table_identity(self):
        a, b = object(), object()
        with patch.object(v2e_modules, '_pricer_parts', side_effect=lambda e, o: (None,) * 5):
            index, fns = v2e_modules.pricer_functions(a, b)
            self.assertEqual((index, fns), v2e_modules.pricer_functions(a, b))
            other, fns2 = v2e_modules.pricer_functions(b, a)
        self.assertNotEqual(index, other);self.assertIsNot(fns, fns2)
        self.assertEqual(set(fns), set(v2e_modules.PRICER_MODES))


class FlagTest(unittest.TestCase):
    def test_cuda_leaves_requires_vectorized_pricing_and_executor_tables(self):
        _, ex = executor()
        with self.assertRaises(ValueError):batch.FramePricer(ex, None, cuda_leaves=True)
        with self.assertRaises(ValueError):batch.FramePricer(ex, None, vectorized=True, cuda_leaves=True)   # inert executor has no tables
        with self.assertRaises(ValueError):batch.FramePricer(ex, None, vectorized=True, cuda_leaves=1)
        pricer = batch.FramePricer(replace(ex, tables=(object(), object())), None, vectorized=True, cuda_leaves=True)
        self.assertTrue(pricer.cuda_leaves);self.assertIsNone(pricer.generated)
        self.assertFalse(batch.FramePricer(ex, None, vectorized=True).cuda_leaves)

    def test_make_kernel_generated_requires_vectorized_and_the_three_modules(self):
        _, ex = executor()
        with self.assertRaises(ValueError):batch.make_kernel(ex, None, params=({},), capacity=2, block_rows=1, generated=dict(views=None, sample=None, joint=None))
        with self.assertRaises(ValueError):batch.make_kernel(ex, None, params=({},), capacity=2, block_rows=1, vectorized=True, generated=dict(views=None))


class Sliced:
    """A stand-in generated joint with an arena size, so make_kernel slices the frame over worlds."""
    def __init__(self, fn, arena_words):
        self.fn = fn;self.module = SimpleNamespace(arena_words=arena_words)
    def __call__(self, *args):return self.fn(*args)


class SliceTest(unittest.TestCase):
    def test_slice_is_the_largest_power_of_two_within_the_arena_budget(self):
        per_lane = 4 * 81
        with patch.object(batch, 'JOINT_ARENA_BUDGET_BYTES', per_lane * 6):   # 6 worlds fit
            self.assertEqual(batch.joint_slice_worlds(Sliced(None, 1), 8), 4)
            self.assertEqual(batch.joint_slice_worlds(Sliced(None, 1), 2), 2)
            self.assertEqual(batch.joint_slice_worlds(Sliced(None, 7), 8), 1)   # nothing fits: one world at a time
        self.assertEqual(batch.joint_slice_worlds(Sliced(None, 54922), 8192), 1024)
        self.assertEqual(batch.joint_slice_worlds(lambda *a: None, 8), 8)


class GeneratedBranchTest(FrameBatchTests):
    """make_kernel with generated lane functions (run through JAX) against its default branch."""
    slice_budget = None   # None: one joint call per frame; else JOINT_ARENA_BUDGET_BYTES during pricing

    def compare(self, modes=(0, 0), owners=(0, 1), missing=(), alter=None):
        requests = self.requests(modes, owners);ex, mapped = executor()
        if alter:requests = alter(requests)
        params = ({'offset': j.asarray(0., j.float32), 'logits': j.arange(9, dtype=j.float32) / 9},
            {'offset': j.asarray(.07, j.float32), 'logits': -j.arange(9, dtype=j.float32) / 9})
        options = dict(capacity=4, block_rows=2, actor_capacity=3, vectorized=True)
        plain = batch.FramePricer(mapped, object(), **options)
        cuda = batch.FramePricer(replace(mapped, tables=(object(), object())), object(), cuda_leaves=True, **options)
        cuda.generated = jax_generated(ex)   # stands in for _generate (no nvcc on CPU)
        if self.slice_budget is not None:cuda.generated['joint'] = Sliced(cuda.generated['joint'], 1)
        for pricer in (plain, cuda):pricer.capture_inputs = True
        with patch.object(batch, 'JOINT_ARENA_BUDGET_BYTES', self.slice_budget or batch.JOINT_ARENA_BUDGET_BYTES):
            return self._compare(plain, cuda, requests, params, modes, missing)

    def _compare(self, plain, cuda, requests, params, modes, missing):
        with patch.object(mapping, 'materialize', side_effect=self.materialize(modes, missing)), patch.object(batch.v5_model, 'forward', side_effect=forward):
            expected = plain.price(requests, self.tables, self.recipe, params)
            actual = cuda.price(requests, self.tables, self.recipe, params)
        if plain.last_output is None:
            self.assertIsNone(cuda.last_output);return actual
        self.assertEqual(set(cuda.last_output), set(plain.last_output) | {'generated_errors'})
        for key in plain.last_output:
            for (path, x), (_, y) in zip(jax.tree_util.tree_flatten_with_path(plain.last_output[key])[0], jax.tree_util.tree_flatten_with_path(cuda.last_output[key])[0]):
                x, y = np.asarray(x), np.asarray(y)
                self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), key)
                self.assertEqual(x.tobytes(), y.tobytes(), key + jax.tree_util.keystr(path))
        self.assertEqual(cuda.last_output['generated_errors'].tolist(), [0, 0, 0])
        self.assertTrue(cuda.last_timing['cuda_leaves']);self.assertFalse(plain.last_timing['cuda_leaves'])
        for a, e in zip(actual, expected):
            self.assertEqual((a.status, a.reason, a.missing_worlds, a.reduction), (e.status, e.reason, e.missing_worlds, e.reduction))
            self.assertEqual(a.world_reports, e.world_reports)
        return actual

    def test_generated_error_word_fails_the_frame(self):
        requests = self.requests();ex, mapped = executor()
        params = ({'offset': j.asarray(0., j.float32), 'logits': j.zeros(9)},)
        cuda = batch.FramePricer(replace(mapped, tables=(object(), object())), object(), capacity=4, block_rows=2, actor_capacity=2, vectorized=True, cuda_leaves=True)
        cuda.generated = dict(jax_generated(ex), sample=lambda states: (jax.vmap(ex.sample)(states), j.ones(1, j.uint32)))
        with patch.object(mapping, 'materialize', side_effect=self.materialize((0, 0))), patch.object(batch.v5_model, 'forward', side_effect=forward):
            with self.assertRaises(RuntimeError):cuda.price(tuple(replace(r, learner_actor=0, opponent_actor=0) for r in requests), self.tables, self.recipe, params)

    # The inherited single-request tests build their own pricers without the flag; keep only the comparisons.
    test_first_mismatching_field_is_serial_numeric_before_categorical = None
    test_evaluated_padding_substitution_and_pre_refusal_work_count_once_per_frame = None
    test_capacities_and_actor_bank_fail_closed = None


class SlicedJointTest(GeneratedBranchTest):
    """The same comparisons with the joint split into world slices of two (8 worlds per frame)."""
    slice_budget = 4 * 81 * 2
    test_generated_error_word_fails_the_frame = None


if __name__ == '__main__':
    unittest.main()
