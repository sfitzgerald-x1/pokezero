import unittest

from pokezero.actions import (
    ACTION_COUNT,
    MOVE_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
    switch_action_index_for_team_index,
)


class ActionHelpersTest(unittest.TestCase):
    def test_action_slots_partition_moves_and_switches(self) -> None:
        self.assertEqual(ACTION_COUNT, 9)
        self.assertEqual(MOVE_ACTION_COUNT, 4)
        self.assertTrue(is_move_action(0))
        self.assertTrue(is_move_action(3))
        self.assertFalse(is_move_action(4))
        self.assertTrue(is_switch_action(4))
        self.assertTrue(is_switch_action(8))
        self.assertFalse(is_switch_action(9))

    def test_canonical_switch_action_map_excludes_active_in_team_order(self) -> None:
        self.assertEqual(canonical_switch_action_map(2), (0, 1, 3, 4, 5))
        self.assertEqual(switch_action_index_for_team_index(3, active_team_index=2), 6)

    def test_canonical_switch_action_map_rejects_invalid_indices(self) -> None:
        with self.assertRaises(ValueError):
            canonical_switch_action_map(6)
        with self.assertRaises(ValueError):
            switch_action_index_for_team_index(2, active_team_index=2)


if __name__ == "__main__":
    unittest.main()
