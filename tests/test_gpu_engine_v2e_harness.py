"""CUDA engine harness, capture and trace-substitution invariants (CPU only)."""
from pathlib import Path
import tempfile
import unittest

import jax.numpy as j
import numpy as np

from gpu_engine import lane_index, v2e_modules


class TraceSubstitutionTest(unittest.TestCase):
    def test_restored_after_exception(self):
        limit, check = lane_index.LIMIT, lane_index._check
        with self.assertRaises(RuntimeError):
            with v2e_modules.trace_substitutions():
                self.assertEqual(lane_index.LIMIT, 0)
                raise RuntimeError('inside the substitution')
        self.assertEqual(lane_index.LIMIT, limit)
        self.assertIs(lane_index._check, check)

    def test_plain_forms_match_select_forms(self):
        rng = np.random.default_rng(0);B = 9;rows = j.arange(B)
        x = j.asarray(rng.integers(-100, 100, (B, 2, 6)), j.int32)
        side = j.asarray(rng.integers(-3, 4, B), j.int32);slot = j.asarray(rng.integers(-8, 9, B), j.int32)
        y = j.asarray(rng.integers(-100, 100, B), j.int32)
        want = (lane_index.take(x, rows, side, slot), lane_index.put(x, rows, (side, slot), y),
            lane_index.update(x, rows, (side, slot), lambda v, d: v + d, y))
        with v2e_modules.trace_substitutions():
            got = (lane_index.take(x, rows, side, slot), lane_index.put(x, rows, (side, slot), y),
                lane_index.update(x, rows, (side, slot), lambda v, d: v + d, y))
        for g, w in zip(got, want):np.testing.assert_array_equal(np.asarray(g), np.asarray(w))

    def test_unsigned_32bit_index_refused_only_while_substituted(self):
        x = j.zeros((3, 6), j.int32);rows = j.arange(3);index = j.asarray([0, 1, 2], j.uint32)
        with v2e_modules.trace_substitutions():
            with self.assertRaises(TypeError):lane_index.take(x, rows, index)
            lane_index.take(x, rows, j.asarray([0, 1, 2], j.uint16))
        lane_index.take(x, rows, index)


class HarnessTest(unittest.TestCase):
    def test_shard_leaves_load_in_numeric_order(self):
        from gpu_engine.v2e_diff import load_shard
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'shard-000000.npz';order = (0, 2, 100, 999, 1000, 1001)
            np.savez_compressed(path, tick=np.int64(0), level=np.int64(0), **{f'l{i:03d}': np.full(1, i) for i in order})
            leaves, tick, level = load_shard(path)
        self.assertEqual([int(x[0]) for x in leaves], list(order))

    def test_comparing_nothing_is_not_equal(self):
        from gpu_engine import v2e_diff
        v2e_diff.MODULES['_noop'] = lambda: (None, None)
        try:
            with tempfile.TemporaryDirectory() as d:
                result = v2e_diff.run('_noop', [], (j.zeros(1),), {}, out=d)
        finally:
            del v2e_diff.MODULES['_noop']
        self.assertFalse(result['equal'])
        self.assertEqual(result['lane_steps'], 0)


class CaptureSinkTest(unittest.TestCase):
    def test_writes_whole_shards_and_refuses_a_used_directory(self):
        from gpu_engine.v2e_capture import Sink
        with tempfile.TemporaryDirectory() as d:
            sink = Sink(d, every=1, skip=0, limit=0)
            sink(np.int32(2), np.arange(6).reshape(2, 3), np.ones((2,), np.uint32))
            sink.close()
            files = sorted(p.name for p in Path(d).iterdir())
            self.assertEqual(files, ['shard-000000.npz'])
            with np.load(Path(d) / files[0]) as z:np.testing.assert_array_equal(z['l000'], np.arange(6).reshape(2, 3))
            with self.assertRaises(SystemExit):Sink(d, every=1, skip=0, limit=0)


if __name__ == '__main__':
    unittest.main()
