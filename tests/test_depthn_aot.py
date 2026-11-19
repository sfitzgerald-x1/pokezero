import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpu_engine.depthn_aot import ExecutableCache, heap_stats, input_signature


def build():
    return jax.jit(lambda x: {'answer': x['vector'] * 3 + 1})


def fail_if_traced():
    raise AssertionError('cache hit attempted to build a JIT program')


class ExecutableCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.args = ({'vector': jnp.arange(3, dtype=jnp.int32)},)

    def cache(self, source='a' * 40, static=('setting', 1)):
        return ExecutableCache(self.directory, source=source, static=static, device=jax.devices()[0])

    def test_hit_does_not_trace_and_keeps_input_and_output_trees(self):
        writer = self.cache()
        original, _, _ = writer.acquire(('program', 1), build, self.args)
        self.assertEqual((writer.events['misses'], writer.events['writes']), (1, 1))
        reader = self.cache()
        changed_values = ({'vector': self.args[0]['vector'] + 4},)
        restored, lower, compile_time = reader.acquire(('program', 1), fail_if_traced, changed_values)
        self.assertEqual((reader.events['hits'], lower, compile_time), (1, 0., 0.))
        self.assertEqual(original(*changed_values)['answer'].tobytes(), restored(*changed_values)['answer'].tobytes())
        self.assertGreater(heap_stats()['gc_objects'], 0)

    def test_source_settings_program_and_shape_invalidate(self):
        self.cache().acquire(('program', 1), build, self.args)
        variants = (
            (self.cache(source='b' * 40), ('program', 1), self.args),
            (self.cache(static=('setting', 2)), ('program', 1), self.args),
            (self.cache(), ('program', 2), self.args),
            (self.cache(), ('program', 1), ({'vector': jnp.arange(4, dtype=jnp.int32)},)),
            (self.cache(), ('program', 1), ({'vector': jnp.arange(3, dtype=jnp.float32)},)),
        )
        for cache, program, args in variants:
            cache.acquire(program, build, args)
            self.assertEqual((cache.events['hits'], cache.events['misses'], cache.events['writes']), (0, 1, 1))

    def test_corrupt_entry_is_replaced_by_a_working_executable(self):
        self.cache().acquire('program', build, self.args)
        next(self.directory.glob('*.jax')).write_bytes(b'broken')
        repair = self.cache()
        with self.assertLogs('gpu_engine.depthn_aot', 'WARNING'):
            repaired, _, _ = repair.acquire('program', build, self.args)
        self.assertEqual((repair.events['read_errors'], repair.events['misses'], repair.events['writes']), (1, 1, 1))
        restored, _, _ = self.cache().acquire('program', fail_if_traced, self.args)
        np.testing.assert_array_equal(repaired(*self.args)['answer'], restored(*self.args)['answer'])

    def test_input_signature_distinguishes_weak_types_and_tree_shapes(self):
        strong = jnp.asarray(1, dtype=jnp.int32)
        weak = jnp.asarray(1)
        self.assertNotEqual(input_signature((weak,)), input_signature((strong,)))
        self.assertNotEqual(input_signature(({'a': strong},)), input_signature(({'b': strong},)))
