"""Heavy anchor: a compiled source-witness advance equals the eager route the light modules use.

The light source-witness modules (sleep_powder_runtime and the modules converted with it) run their advances through
gpu_engine_eager_witness. There, the frame and witness closures run their own bodies, and execution.consume_frame_device
runs under a stable jit. This module is what ties those light tests to the compiled path.

It runs sleep_powder's construct and advance compiled and eager from the same inputs, and requires every frame output,
every witness output (construction's included) and the advance result to match field by field: the same structure, dtypes
and shapes, and identical bits (integers and bools exactly, every other dtype by bit pattern). A negative control flips one
bit of the eager witness outputs and requires the comparison to fail at that leaf.

Heavy tier: the compiled route builds the whole frame, witness and consume programs.
"""
import contextlib
import re
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as driver, transition
import gpu_engine_eager_witness as eager
import test_gpu_engine_sleep_powder_runtime as fixture


def _bits(x):
    """Integers and bools as they are; every other dtype (float32, bfloat16, ...) by its bit pattern."""
    x = np.asarray(x)
    if x.dtype == bool or np.issubdtype(x.dtype, np.integer):return x
    return x.view(np.dtype(f'u{x.dtype.itemsize}'))


def first_leaf_path(output):
    """The key path of the first non-empty array leaf, the one flip_first_bit changes."""
    return next(jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_flatten_with_path(output)[0]
                if hasattr(leaf, 'dtype') and np.size(leaf))


def flip_first_bit(output):
    """The output with the lowest bit of its first non-empty array leaf flipped (a bool is negated)."""
    leaves, tree = jax.tree.flatten(output)
    for index, leaf in enumerate(leaves):
        if hasattr(leaf, 'dtype') and np.size(leaf):
            changed = np.array(leaf, copy=True)
            flat = changed.reshape(-1)
            flat[0] = ~flat[0] if changed.dtype == bool else (_bits(flat[:1]) ^ 1).view(changed.dtype)[0]
            leaves[index] = j.asarray(changed)
            return jax.tree.unflatten(tree, leaves)
    raise ValueError('no array leaf to perturb')


class SourceWitnessCompiledAnchorTests(unittest.TestCase):
    construct = fixture.SleepPowderRuntimeTests.construct
    _routes = {}

    @classmethod
    def setUpClass(cls):
        fixture.SleepPowderRuntimeTests.setUpClass();f = fixture.SleepPowderRuntimeTests
        for name in ('base', 'move_id', 'metadata', 'names', 'prepared', 'runtime'):setattr(cls, name, getattr(f, name))

    def run_route(self, *, eager_route, perturb=None):
        """Construct, then one advance (Sleep Powder hits on both sides), recording every frame and witness output."""
        records = dict(frame=[], witness=[]);result = None
        draws = j.zeros((2, 2, 12), j.int32).at[:, :, 4].set(1)
        chance = transition.ChanceInputs(draws, *[j.zeros((2, 2), j.int32) for _ in range(4)], j.zeros(2, j.int32))
        actions = j.array([[0, 7], [0, 7]], j.int32)
        with contextlib.ExitStack() as stack:
            if eager_route:stack.enter_context(eager.eager_source_witness(self.prepared))
            for key, slot in (('frame', 'frame'), ('frame_witness_fast', 'witness')):
                inner = self.runtime[key]
                def record(*args, _inner=inner, _slot=slot, **kwargs):
                    output = _inner(*args, **kwargs)
                    if perturb is not None and _slot == 'witness':output = perturb(output)
                    records[_slot].append(output);return output
                self.runtime[key] = record
                stack.callback(self.runtime.__setitem__, key, inner)
            try:result = driver.advance(self.construct(), actions, chance_inputs=chance)
            except Exception:
                if perturb is None:raise   # a perturbed witness output may be refused downstream; its records are kept
        return records, result

    def route(self, name):
        if name not in self._routes:self._routes[name] = self.run_route(eager_route=name == 'eager')
        return self._routes[name]

    def assert_same(self, compiled, eager_output, label):
        self.assertEqual(jax.tree.structure(compiled), jax.tree.structure(eager_output), label)
        for (path, x), (_, y) in zip(jax.tree_util.tree_flatten_with_path(compiled)[0],
                                     jax.tree_util.tree_flatten_with_path(eager_output)[0], strict=True):
            name = label + jax.tree_util.keystr(path);x, y = np.asarray(x), np.asarray(y)
            self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), name)
            np.testing.assert_array_equal(_bits(x), _bits(y), err_msg=name)

    def test_frame_outputs_compiled_equal_eager(self):
        (compiled, _), (eager_records, _) = self.route('compiled'), self.route('eager')
        self.assertEqual(len(compiled['frame']), 1);self.assertEqual(len(eager_records['frame']), 1)
        self.assert_same(compiled['frame'][0], eager_records['frame'][0], 'frame')

    def test_witness_outputs_compiled_equal_eager(self):
        (compiled, _), (eager_records, _) = self.route('compiled'), self.route('eager')
        self.assertGreaterEqual(len(compiled['witness']), 1)
        self.assertEqual(len(compiled['witness']), len(eager_records['witness']))
        for index, (a, b) in enumerate(zip(compiled['witness'], eager_records['witness'])):
            self.assert_same(a, b, f'witness[{index}]')

    def test_advance_result_compiled_equal_eager(self):
        (_, compiled), (_, eager_result) = self.route('compiled'), self.route('eager')
        self.assertEqual(compiled.state.core.admitted.tolist(), [True, True])
        self.assert_same(compiled.state, eager_result.state, 'state')
        self.assert_same(compiled.owners, eager_result.owners, 'owners')

    def test_negative_control_one_flipped_eager_witness_bit_fails_the_comparison(self):
        compiled, _ = self.route('compiled')
        perturbed, _ = self.run_route(eager_route=True, perturb=flip_first_bit)
        self.assertGreaterEqual(len(perturbed['witness']), 1)
        leaf = 'witness[0]' + first_leaf_path(perturbed['witness'][0])
        with self.assertRaisesRegex(AssertionError, re.escape(leaf)):
            self.assert_same(compiled['witness'][0], perturbed['witness'][0], 'witness[0]')


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled


if __name__ == '__main__':
    unittest.main()
