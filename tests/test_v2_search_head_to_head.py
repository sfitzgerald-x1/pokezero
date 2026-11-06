"""Pure helpers of the search-seat head-to-head and the cell comparison; no engine, model or device run."""
import json
from pathlib import Path
import tempfile
from typing import NamedTuple
import unittest

import jax.numpy as j
import numpy as np

from gpu_engine import v2_search_head_to_head as h2h, v2s_compare_cells as compare


class Toy(NamedTuple):
    own: object      # [B, 2, 3]: an own-boundary field
    other: object    # [B, 2]: a field the boundary never reads
    turn: object     # [B]: a global field


class HelperTests(unittest.TestCase):
    def test_world_block_seeds_prefix_and_determinism(self):
        seeds = h2h.world_block_seeds(17, 4)
        self.assertEqual(seeds[0], 17);self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(h2h.world_block_seeds(17, 2), seeds[:2])
        with self.assertRaises(ValueError):h2h.world_block_seeds(17, 0)

    def test_turn_bands(self):
        self.assertEqual([h2h.band(t) for t in (0, 1, 3, 4, 9, 10, 19, 20, 500)],
            ['1-3', '1-3', '1-3', '4-9', '4-9', '10-19', '10-19', '20+', '20+'])
        with self.assertRaises(ValueError):h2h.band(-1)

    def test_counterfactual_keeps_only_own_cells(self):
        b = 4;state = Toy(j.arange(b * 2 * 3).reshape(b, 2, 3), j.arange(b * 2).reshape(b, 2) + 100, j.arange(b) + 1000)
        rows, owners, perm = [0, 2], [1, 0], np.roll(np.arange(b), 1)
        cf = h2h.counterfactual(state, rows, owners, perm, ('own',))
        own, swapped = np.asarray(state.own), np.asarray(state.own)[perm]
        for r, o in zip(rows, owners):
            np.testing.assert_array_equal(np.asarray(cf.own)[r, o], own[r, o])            # A's own cells kept
            np.testing.assert_array_equal(np.asarray(cf.own)[r, 1 - o], swapped[r, 1 - o])  # the foe's from another game
        np.testing.assert_array_equal(np.asarray(cf.other), np.asarray(state.other)[perm])
        np.testing.assert_array_equal(np.asarray(cf.turn), np.asarray(state.turn)[perm])
        self.assertFalse(h2h.leaves_equal(cf, state))
        self.assertTrue(h2h.leaves_equal(state, Toy(*state)))

    def test_leaves_equal_is_byte_and_dtype_exact(self):
        self.assertTrue(h2h.leaves_equal({'a': np.zeros(3, np.float32)}, {'a': np.zeros(3, np.float32)}))
        self.assertFalse(h2h.leaves_equal({'a': np.zeros(3, np.float32)}, {'a': np.zeros(3, np.float64)}))
        self.assertFalse(h2h.leaves_equal({'a': np.zeros(3, np.float32)}, {'a': -np.zeros(3, np.float32)}))   # -0.0 differs
        self.assertFalse(h2h.leaves_equal({'a': np.zeros(3)}, {'b': np.zeros(3)}))
        from types import MappingProxyType
        frozen = lambda: MappingProxyType({'x': np.arange(3), 'nested': MappingProxyType({'y': np.ones(2, np.int32)})})
        self.assertTrue(h2h.leaves_equal(frozen(), frozen()))   # nested read-only mappings compare by content
        changed = MappingProxyType({'x': np.arange(3), 'nested': MappingProxyType({'y': np.zeros(2, np.int32)})})
        self.assertFalse(h2h.leaves_equal(frozen(), changed))
        with self.assertRaises(TypeError):h2h.leaves_equal({'a': object()}, {'a': object()})


class CompareTests(unittest.TestCase):
    def write(self, root, label, scores, games, decisions):
        prefix = Path(root, label)
        Path(str(prefix) + '-pairs.jsonl').write_text(''.join(json.dumps(dict(stream=s, swapped=s - 10, score=v)) + '\n' for s, v in scores.items()))
        np.savez(str(prefix) + '-games.npz', **games)
        Path(str(prefix) + '-decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decisions))
        return prefix

    def test_paired_difference_and_identity(self):
        games = dict(streams=np.arange(4), status=np.ones(4, np.int32), winner=np.zeros(4, np.int32), terminal_frame=np.full(4, 7))
        decisions = [dict(stream=1, a_side=0, decisions=[dict(frame=0, raw=2, chosen=2), dict(frame=1, raw=0, chosen=3)])]
        with tempfile.TemporaryDirectory() as root:
            x = self.write(root, 'x', {1: 1., 2: .5, 3: None, 4: 0.}, games, decisions)
            y = self.write(root, 'y', {1: .5, 2: .5, 3: 1., 4: 0.}, games, decisions)
            result = compare.paired_difference(compare.pairs(x), compare.pairs(y))
            self.assertEqual(result['pairs'], 3);self.assertEqual(result['excluded'], 1)
            self.assertAlmostEqual(result['difference_points'], 100 * .5 / 3)
            self.assertEqual((result['pairs_x_better'], result['pairs_y_better']), (1, 0))
            self.assertTrue(compare.identical(x, y)['all_identical'])
            z = self.write(root, 'z', {1: 1., 2: .5, 3: None, 4: 0.}, games,
                [dict(stream=1, a_side=0, decisions=[dict(frame=0, raw=2, chosen=2), dict(frame=1, raw=0, chosen=0)])])
            self.assertEqual(compare.identical(x, z)['differing_decisions'], 1)
            self.assertFalse(compare.identical(x, z)['all_identical'])


if __name__ == '__main__':
    unittest.main()
