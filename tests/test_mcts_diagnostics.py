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

    def test_classifies_mixed_replay_prefix_divergence(self) -> None:
        reason = (
            "all opponent action scenarios were replay-illegal: "
            "replay actions for decision round 12 do not match environment request "
            "(unexpected players: p2); "
            "start override does not reproduce recorded replay prefix observations "
            "for decision round 28: p1."
        )

        self.assertEqual(root_puct_fallback_category(reason), "mixed_replay_prefix_divergence")

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
