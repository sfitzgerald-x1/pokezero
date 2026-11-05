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


if __name__ == '__main__':
    unittest.main()
