from __future__ import annotations

import unittest

from pokezero.mcts_diagnostics import root_puct_fallback_category


class RootPUCTFallbackCategoryTests(unittest.TestCase):
    def test_classifies_missing_sampled_world(self) -> None:
        self.assertEqual(
            root_puct_fallback_category(
                "start override planner did not produce a sampled world: opponent Unown-Z"
            ),
            "missing_sampled_world",
        )

    def test_classifies_planner_side_rejections(self) -> None:
        examples = {
            "opponent_action_planner returned an illegal action for p2: 5": (
                "opponent_planner_illegal_action"
            ),
            "opponent_action_planner returned the acting player's action": (
                "opponent_planner_self_action"
            ),
            "missing opponent actions for p2": "opponent_planner_missing_actions",
            "unexpected opponent actions for p2": "opponent_planner_unexpected_actions",
        }

        for reason, category in examples.items():
            with self.subTest(reason=reason):
                self.assertEqual(root_puct_fallback_category(reason), category)

    def test_classifies_illegal_current_request_action(self) -> None:
        self.assertEqual(
            root_puct_fallback_category("p2: action_index 2 is not legal for the current request."),
            "illegal_action_for_current_request",
        )

    def test_classifies_mixed_replay_prefix_divergence(self) -> None:
        reason = (
            "all opponent action scenarios were replay-illegal: "
            "replay actions for decision round 12 do not match environment request "
            "(unexpected players: p2); "
            "start override does not reproduce recorded replay prefix observations "
            "for decision round 28: p1."
        )

        self.assertEqual(root_puct_fallback_category(reason), "mixed_replay_prefix_divergence")

    def test_classifies_mixed_replay_prefix_divergence_with_missing_world(self) -> None:
        reason = (
            "all opponent action scenarios were replay-illegal: "
            "start override source did not produce a sampled world.; "
            "start override does not reproduce recorded replay prefix observations "
            "for decision round 28: p1."
        )

        self.assertEqual(root_puct_fallback_category(reason), "mixed_replay_prefix_divergence")

    def test_classifies_composite_missing_sampled_world(self) -> None:
        reason = (
            "all opponent action scenarios were replay-illegal: "
            "start override source did not produce a sampled world."
        )

        self.assertEqual(root_puct_fallback_category(reason), "missing_sampled_world")

    def test_classifies_start_override_observation_mismatch(self) -> None:
        self.assertEqual(
            root_puct_fallback_category(
                "start override does not reproduce recorded replay prefix observations "
                "for decision round 3: p1."
            ),
            "start_override_observation_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
