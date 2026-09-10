"""Integration contract for the two-seat forced-replacement prior replay."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    pokezero_search = None

from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT  # noqa: E402

from force_switch_prior_replay import run_replay  # noqa: E402


@unittest.skipUnless(
    pokezero_search is not None and hasattr(pokezero_search, "LeafEncoder"),
    "requires native LeafEncoder",
)
@unittest.skipUnless(
    (Path(DEFAULT_SHOWDOWN_ROOT) / "dist" / "sim" / "index.js").exists(),
    "requires built Pokemon Showdown checkout",
)
class ForceSwitchPriorReplayTest(unittest.TestCase):
    def test_both_seats_reach_switch_only_native_root_mapping(self) -> None:
        result = run_replay(showdown_root=Path(DEFAULT_SHOWDOWN_ROOT))

        self.assertEqual(result["result"], "PASS")
        self.assertEqual([seat["player_id"] for seat in result["seats"]], ["p1", "p2"])
        for seat in result["seats"]:
            self.assertTrue(seat["request_force_switch"])
            self.assertEqual(seat["payload_request_kind"], "force-switch")
            self.assertTrue(seat["acting_side_force_switch"])
            self.assertFalse(seat["opposing_side_force_switch"])
            self.assertEqual(seat["live_legal_action_indices"], seat["native_root_action_indices"])
            self.assertTrue(all(action_index >= 4 for action_index in seat["native_root_action_indices"]))


if __name__ == "__main__":
    unittest.main()
