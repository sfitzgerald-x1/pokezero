"""Generator constant folding and refusals (CPU only: generation, no CUDA build)."""
import unittest

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import v2e_gen, v2e_j2c


def folded(fn):
    """Run the generator over a lane-free function; return (Generator, output Vals)."""
    closed = jax.make_jaxpr(fn)()
    g = v2e_j2c.Generator('t')
    return g, g.run_with_consts(closed.jaxpr, [g.const(c) for c in closed.consts], [])


def value(v):
    if v.kind == 'lit':return np.broadcast_to(np.asarray(v.lit, v.dtype), v.shape)
    return np.asarray(v.arr).reshape(v.shape)


class ConstantFoldTest(unittest.TestCase):
    F = np.array([1.0, 2.0, 3.0, -0.0], np.float32)
    C = np.array([[3], [0], [5], [0]], np.int32)
    B = np.array([False, True, False, True])

    def check(self, fn, want=None):
        """fn folds completely, to want (default: fn evaluated by JAX; give IEEE numpy references for
        division, since XLA folds constant division with the host's IEEE arithmetic)."""
        g, outs = folded(fn)
        want = jax.tree.leaves(fn()) if want is None else [want]
        for v, w in zip(outs, want):
            self.assertIn(v.kind, ('lit', 'const'), f'not folded: {v.kind}')
            bytes_ = lambda a: np.ascontiguousarray(a).view(np.uint8)
            np.testing.assert_array_equal(bytes_(value(v)), bytes_(np.asarray(w)))

    def test_constant_views_carry_values(self):
        F, C, B = self.F, self.C, self.B
        self.check(lambda: j.copy(j.asarray(F)) != 2.0)
        self.check(lambda: j.sum(j.asarray(C), axis=1) == 0)
        self.check(lambda: ~j.copy(j.asarray(B)))
        self.check(lambda: j.copy(j.ones(4, j.float32)) * 2.0)
        self.check(lambda: j.asarray(C).reshape(2, 2) + 1)

    def test_data_movement_and_exact_division(self):
        E = np.random.default_rng(0).standard_normal(9).astype(np.float32)
        self.check(lambda: j.asarray(E)[1:] / j.asarray(E)[:-1], E[1:] / E[:-1])
        self.check(lambda: j.concatenate([j.asarray(E), j.asarray(E)[::-1]]) / j.float32(3.0), np.concatenate([E, E[::-1]]) / np.float32(3.0))
        self.check(lambda: j.pad(j.asarray(E), (2, 1)) * 2.0)

    def test_stack_folds(self):
        E = np.random.default_rng(1).standard_normal(5).astype(np.float32)
        self.check(lambda: j.stack([j.asarray(E), j.asarray(E) * 2.0]) / j.float32(3.0), np.stack([E, E * np.float32(2.0)]) / np.float32(3.0))

    def test_barrier_is_opaque(self):
        g, outs = folded(lambda: jax.lax.optimization_barrier(j.float32(1.0)) * 2.0)
        self.assertEqual(outs[0].kind, 'reg')
        self.assertFalse(outs[0].cderived)


class GenerationTest(unittest.TestCase):
    def generate(self, fn, *args):
        closed, avals, _, _ = v2e_gen.trace(fn, args)
        return v2e_j2c.generate('t', closed, avals)

    def test_copy_of_constant_reaches_identity_fold(self):
        x = j.zeros((5, 4), j.float32);ones = np.ones(4, np.float32)
        m = self.generate(lambda x: x * j.copy(j.asarray(ones)), x)
        self.assertNotIn('v2e::fmul', ''.join(m.files.values()))

    def test_lane_broadcast_constant_division_folds(self):
        # XLA sinks the broadcast and folds the division at the constant's own shape.
        x = j.zeros((5, 4), j.float32);C = np.arange(1, 5, dtype=np.float32)
        m = self.generate(lambda x: x + j.broadcast_to(j.asarray(C), x.shape) / j.float32(3.0), x)
        self.assertNotIn('v2e::fdiv', ''.join(m.files.values()))

    def test_iota_division_stays_a_runtime_division(self):
        # XLA's constant folding leaves iota to run time, so the division is div.full on the device.
        x = j.zeros((5, 4), j.float32)
        for fn in (lambda x: x + j.arange(4, dtype=j.float32) / 3.0, lambda x: x + j.arange(x.shape[0], dtype=j.float32)[:, None] / 3.0):
            self.assertIn('v2e::fdiv', ''.join(self.generate(fn, x).files.values()))

    def test_unfoldable_constant_reduction_is_refused(self):
        x = j.zeros((5, 4), j.float32);C = np.arange(1, 5, dtype=np.float32)
        with self.assertRaises(v2e_j2c.Unsupported):
            self.generate(lambda x: x + j.max(j.broadcast_to(j.asarray(C), x.shape), axis=1, keepdims=True), x)

    def test_signaling_nan_literal_keeps_its_bits(self):
        x = j.zeros((5, 4), j.float32);sn = np.array([0x7f800003], np.uint32).view(np.float32)[0]
        m = self.generate(lambda x: j.where(x > 0, x, sn), x)
        self.assertIn('0x7f800003u', ''.join(m.files.values()))

    def test_unused_constant_buffers_are_pruned(self):
        x = j.zeros((5, 4), j.float32);C = np.arange(1, 5, dtype=np.float32)
        m = self.generate(lambda x: x + (j.copy(j.asarray(C)) * 2.0)[::-1], x)
        self.assertEqual(len(m.consts), 1)

    def test_runtime_division_is_generated(self):
        x = j.zeros((5, 4), j.float32)
        m = self.generate(lambda x: x / j.float32(3.0), x)
        self.assertIn('v2e::fdiv', ''.join(m.files.values()))


if __name__ == '__main__':
    unittest.main()
