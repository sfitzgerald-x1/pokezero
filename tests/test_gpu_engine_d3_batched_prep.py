"""Batched D3 preparation pieces equal their one-root originals (CPU)."""
import unittest

import numpy as np

from gpu_engine import device_public_world_state as mapping
from gpu_engine.state import empty_state


class WorldKeysTest(unittest.TestCase):
    def test_world_keys_equal_empty_state_keys(self):
        rng = np.random.default_rng(0)
        seeds = [0, 1, 2, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1] + [int(x) for x in rng.integers(0, 2**32, 64, dtype=np.uint64)]
        keys = mapping.world_keys(seeds)
        self.assertEqual(keys.shape, (len(seeds), 1, 2))
        for seed, key in zip(seeds, keys):
            np.testing.assert_array_equal(key, np.asarray(empty_state(1, seed=seed).keys))

    def test_empty_template_matches_empty_state(self):
        arrays = mapping._empty_arrays()
        reference = empty_state(1, seed=12345)._asdict()
        self.assertEqual(set(arrays), set(reference))
        for name, value in reference.items():
            if name == 'keys':continue
            value = np.asarray(value)
            self.assertEqual((arrays[name].dtype, arrays[name].shape), (value.dtype, value.shape), name)
            np.testing.assert_array_equal(arrays[name], value)

    def test_template_copies_are_independent(self):
        a = mapping._empty_arrays();a['hp'][0, 0, 0] = 99
        self.assertEqual(int(mapping._empty_arrays()['hp'][0, 0, 0]), 0)


class GateCompareTest(unittest.TestCase):
    def test_structure_is_exact_and_floats_have_tolerance(self):
        from gpu_engine.d3_saved_root_gate import differences, FLOAT_TOLERANCE
        row = dict(status='MATCH', worlds=[1, 2], target=dict(delta=0.5, top=True))
        self.assertEqual(differences(row, dict(row, target=dict(delta=0.5 + FLOAT_TOLERANCE / 2, top=True))), [])
        self.assertEqual(differences(row, dict(row, target=dict(delta=0.5 + 1e-6, top=True))), ['.target.delta:float 1e-06'])
        self.assertEqual(differences(row, dict(row, status='MISMATCH')), ['.status:value'])
        self.assertEqual(differences(row, dict(row, worlds=[1, 2, 3])), ['.worlds:length'])
        self.assertEqual(differences(row, dict(row, worlds=[1, 2.0])), ['.worlds[1]:value'])
        self.assertEqual(differences(row, dict(row, extra=1)), [':keys'])
        self.assertEqual(differences(row, dict(row, target=dict(delta=0.5, top=1))), ['.target.top:value'])


if __name__ == '__main__':
    unittest.main()
