import unittest

from pokezero.actions import (
    ACTION_COUNT,
    ActionCandidate,
    MOVE_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
    move_action_candidates,
    switch_action_index_for_team_index,
    switch_action_candidates,
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

    def test_switch_slots_are_dense_action_candidates(self) -> None:
        legal_mask = (False, False, False, False, True, False, True, False, False)
        candidates = switch_action_candidates(active_team_index=2, legal_mask=legal_mask)

        self.assertEqual(
            candidates,
            (
                ActionCandidate(action_index=4, kind="switch", team_index=0, legal=True),
                ActionCandidate(action_index=5, kind="switch", team_index=1, legal=False),
                ActionCandidate(action_index=6, kind="switch", team_index=3, legal=True),
                ActionCandidate(action_index=7, kind="switch", team_index=4, legal=False),
                ActionCandidate(action_index=8, kind="switch", team_index=5, legal=False),
            ),
        )

    def test_move_slots_are_action_candidates(self) -> None:
        legal_mask = (True, False, True, False, False, False, False, False, False)
        candidates = move_action_candidates(legal_mask)

        self.assertEqual(
            candidates,
            (
                ActionCandidate(action_index=0, kind="move", move_slot=0, legal=True),
                ActionCandidate(action_index=1, kind="move", move_slot=1, legal=False),
                ActionCandidate(action_index=2, kind="move", move_slot=2, legal=True),
                ActionCandidate(action_index=3, kind="move", move_slot=3, legal=False),
            ),
        )

    def test_canonical_switch_action_map_rejects_invalid_indices(self) -> None:
        with self.assertRaises(ValueError):
            canonical_switch_action_map(6)
        with self.assertRaises(ValueError):
            switch_action_index_for_team_index(2, active_team_index=2)

    def test_action_candidate_rejects_mismatched_slots(self) -> None:
        with self.assertRaises(ValueError):
            ActionCandidate(action_index=4, kind="move", move_slot=0, legal=True)
        with self.assertRaises(ValueError):
            ActionCandidate(action_index=0, kind="switch", team_index=1, legal=True)


if __name__ == "__main__":
    unittest.main()
