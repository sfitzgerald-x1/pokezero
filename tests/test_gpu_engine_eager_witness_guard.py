"""gpu_engine_eager_witness's compile guard, on stand-in jitted closures (no engine runtime, no source pins).

Inside eager_source_witness a new compile of the frame or witness closure fails the scope. jax.clear_caches() resets a jitted
function's compile count, so inside the scope it checks first and then counts again; a compile followed by a clear can't hide.
"""
import unittest
from unittest import mock

import jax
import jax.numpy as jnp

import gpu_engine_eager_witness as eager
from gpu_engine import engine_execution_completeness as execution
from gpu_engine import engine_retained_encoder_driver as driver


def runtime():
    return {'frame': jax.jit(lambda x: x * 2), 'frame_witness_fast': jax.jit(lambda x: x - 1)}


class EagerWitnessGuardTests(unittest.TestCase):
    def scope(self, r):
        self.enterContext(mock.patch.object(driver, '_checked', lambda _prepared: r))
        return eager.eager_source_witness(object())

    def assert_restored(self, r, clear):
        self.assertIs(jax.clear_caches, clear)
        self.assertIs(execution.consume_frame_device, eager._CONSUME)
        self.assertTrue(all(hasattr(r[key], '__wrapped__') for key in ('frame', 'frame_witness_fast')))

    def test_eager_calls_and_a_clear_pass_and_everything_is_restored(self):
        r, clear = runtime(), jax.clear_caches
        with self.scope(r):
            self.assertEqual(r['frame'](jnp.ones(2)).tolist(), [2.0, 2.0])
            self.assertIsNot(jax.clear_caches, clear)
            jax.clear_caches()
            self.assertEqual(r['frame_witness_fast'](jnp.ones(2)).tolist(), [0.0, 0.0])
        self.assert_restored(r, clear)

    def test_a_clear_resets_the_compile_count_the_guard_reads(self):
        probe = jax.jit(lambda x: x + 1)
        probe(jnp.ones(3))
        self.assertEqual(probe._cache_size(), 1)
        jax.clear_caches()
        self.assertEqual(probe._cache_size(), 0)   # why the scope must check before a clear

    def test_a_compile_then_a_clear_inside_the_scope_fails_at_the_clear(self):
        r, clear = runtime(), jax.clear_caches
        compiled = r['frame']
        with self.assertRaisesRegex(AssertionError, 'the frame was compiled inside an eager scope'):
            with self.scope(r):
                compiled(jnp.ones(2))
                jax.clear_caches()
        self.assert_restored(r, clear)

    def test_a_compile_without_a_clear_fails_at_exit(self):
        r, clear = runtime(), jax.clear_caches
        compiled = r['frame_witness_fast']
        with self.assertRaisesRegex(AssertionError, 'the frame_witness_fast was compiled inside an eager scope'):
            with self.scope(r):
                compiled(jnp.ones(2))
        self.assert_restored(r, clear)

    def test_a_nested_scope_is_refused(self):
        r = runtime()
        with self.scope(r):
            with self.assertRaisesRegex(ValueError, 'already eager'):
                with eager.eager_source_witness(object()):
                    pass


def tearDownModule():
    eager.clear()


if __name__ == '__main__':
    unittest.main()
