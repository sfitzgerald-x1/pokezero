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


class VersionPinTest(unittest.TestCase):
    def test_unpinned_versions_are_refused(self):
        import os
        from unittest import mock
        saved = dict(v2e_j2c.PINNED)
        try:
            v2e_j2c.PINNED['jax'] = '0.0.0-not-installed'
            with mock.patch.dict(os.environ, {'V2E_ALLOW_UNPINNED_JAX': ''}):
                with self.assertRaises(v2e_j2c.Unsupported):v2e_j2c.check_pinned_versions()
            with mock.patch.dict(os.environ, {'V2E_ALLOW_UNPINNED_JAX': '1'}):v2e_j2c.check_pinned_versions()
        finally:
            v2e_j2c.PINNED.clear();v2e_j2c.PINNED.update(saved)

    def test_installed_versions_are_pinned(self):
        found = v2e_j2c.check_pinned_versions()
        self.assertEqual((found['jax'], found['jaxlib']), (v2e_j2c.PINNED['jax'], v2e_j2c.PINNED['jaxlib']))

    def test_absent_cuda_plugins_are_accepted_but_other_plugin_versions_refused(self):
        import importlib.metadata as metadata
        import os
        from unittest import mock
        real = metadata.version
        def version(name, plugin=None):
            if name in v2e_j2c.CUDA_PLUGINS:
                if plugin is None:raise metadata.PackageNotFoundError(name)
                return plugin
            return real(name)
        with mock.patch.dict(os.environ, {'V2E_ALLOW_UNPINNED_JAX': ''}):
            with mock.patch.object(metadata, 'version', version):
                found = v2e_j2c.check_pinned_versions()
            self.assertEqual([found[name] for name in v2e_j2c.CUDA_PLUGINS], [None, None])
            with mock.patch.object(metadata, 'version', lambda name: version(name, '0.11.0')):
                with self.assertRaises(v2e_j2c.Unsupported):v2e_j2c.check_pinned_versions()


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

    def test_iota_division_stays_at_run_time(self):
        # XLA's constant folding leaves iota to run time (the division becomes a runtime reciprocal multiply).
        x = j.zeros((5, 4), j.float32)
        for fn in (lambda x: x + j.arange(4, dtype=j.float32) / 3.0, lambda x: x + j.arange(x.shape[0], dtype=j.float32)[:, None] / 3.0):
            self.assertIn('v2e::fmul', ''.join(self.generate(fn, x).files.values()))
        self.assertIn('v2e::fdiv', ''.join(self.generate(lambda x: x + 1.0 / (j.arange(4, dtype=j.float32) + 1.0), x).files.values()))

    def test_unfoldable_constant_reduction_is_refused(self):
        x = j.zeros((5, 4), j.float32);C = np.arange(1, 5, dtype=np.float32)
        with self.assertRaises(v2e_j2c.Unsupported):
            self.generate(lambda x: x + j.max(j.broadcast_to(j.asarray(C), x.shape), axis=1, keepdims=True), x)

    def test_outer_product_of_constant_broadcasts_stays_runtime(self):
        # XLA folds neither bcast(C1, dims A) / bcast(C2, dims B) nor ops on its result.
        x = j.zeros((5, 4, 3), j.float32);C1 = np.arange(1, 5, dtype=np.float32);C2 = np.arange(2, 5, dtype=np.float32)
        M = np.full((4, 3), 1.5, np.float32) + np.arange(12, dtype=np.float32).reshape(4, 3)
        for fn in (lambda x: x + j.asarray(C1)[:, None] / j.asarray(C2)[None, :], lambda x: x + (j.asarray(C1)[:, None] / j.asarray(C2)[None, :]) * j.asarray(M)):
            code = ''.join(self.generate(fn, x).files.values())
            self.assertIn('v2e::fmul', code)   # bcast(C1) * bcast(1 / C2) at run time, as XLA rewrites it
            self.assertNotIn('v2e::fdiv', code)

    def test_scalar_over_constant_array_folds_exactly(self):
        # XLA folds k / C and C / k while lowering, before any simplifier rewrite.
        x = j.zeros((5, 4), j.float32);C = np.array([3.0, 7.0, 0.3, 1.7], np.float32)
        for fn in (lambda x: x + 0.7 / j.asarray(C), lambda x: x + j.asarray(C) / 0.7):
            code = ''.join(self.generate(fn, x).files.values())
            self.assertNotIn('v2e::fdiv', code);self.assertNotIn('v2e::fmul', code)
        g, outs = folded(lambda: j.float32(0.7) / j.asarray(C))
        np.testing.assert_array_equal(np.asarray(value(outs[0])).view(np.uint32), (np.float32(0.7) / C).view(np.uint32))

    def test_scalar_over_lane_broadcast_uses_the_reciprocal(self):
        # k / bcast(C): the broadcast sinks, then A / C -> A * (1 / C) fires on the broadcast-scalar dividend.
        C = np.array([3.0, 7.0, 0.3, 1.7], np.float32)
        x = j.zeros((5, 4), j.float32)
        closed, avals, _, _ = v2e_gen.trace(lambda x: x + 0.7 / j.broadcast_to(j.asarray(C), x.shape), (x,))
        g = v2e_j2c.Generator('t');ins = [v2e_j2c.Val(np.dtype(a.dtype), tuple(a.shape), 'in', ptr=f'in{n}', stride=4) for n, a in enumerate(avals)]
        g.run_with_consts(closed.jaxpr, [g.const(c) for c in closed.consts], ins)
        want = np.float32(0.7) * (np.float32(1.0) / C)
        self.assertTrue(any(np.array_equal(np.asarray(c).view(np.uint32), want.view(np.uint32)) for c in g.k.consts))

    def test_uncertain_provenance_is_refused(self):
        x = j.zeros((5, 2, 3), j.float32);C1 = np.arange(1, 5, dtype=np.float32);C2 = np.arange(2, 5, dtype=np.float32)
        fn = lambda x: x + j.broadcast_to(j.asarray(C1)[:, None], (4, 3))[1:3] / j.broadcast_to(j.asarray(C2)[None, :], (2, 3))
        with self.assertRaises(v2e_j2c.Unsupported):self.generate(fn, x)

    def test_constant_and_broadcast_fold(self):
        x = j.zeros((5, 4, 3), j.float32);C2 = np.arange(2, 5, dtype=np.float32)
        M = np.full((4, 3), 1.5, np.float32) + np.arange(12, dtype=np.float32).reshape(4, 3)
        self.assertNotIn('v2e::fdiv', ''.join(self.generate(lambda x: x + j.asarray(M) / j.asarray(C2)[None, :], x).files.values()))

    def test_signaling_nan_literal_keeps_its_bits(self):
        x = j.zeros((5, 4), j.float32);sn = np.array([0x7f800003], np.uint32).view(np.float32)[0]
        m = self.generate(lambda x: j.where(x > 0, x, sn), x)
        self.assertIn('0x7f800003u', ''.join(m.files.values()))

    def test_unused_constant_buffers_are_pruned(self):
        x = j.zeros((5, 4), j.float32);C = np.arange(1, 5, dtype=np.float32)
        m = self.generate(lambda x: x + (j.copy(j.asarray(C)) * 2.0)[::-1], x)
        self.assertEqual(len(m.consts), 1)

    def test_division_by_a_constant_multiplies_by_the_reciprocal(self):
        # XLA's simplifier: A / C -> A * (1 / C) with the reciprocal computed on the host.
        x = j.zeros((5, 4), j.float32)
        code = ''.join(self.generate(lambda x: x / j.float32(3.0), x).files.values())
        self.assertNotIn('v2e::fdiv', code)
        self.assertIn('0x%08xu' % int(np.array([np.float32(1) / np.float32(3)], np.float32).view(np.uint32)[0]), code)

    def test_runtime_division_is_generated(self):
        x = j.zeros((5, 4), j.float32)
        m = self.generate(lambda x, y: x / y, x, x)
        self.assertIn('v2e::fdiv', ''.join(m.files.values()))


if __name__ == '__main__':
    unittest.main()
