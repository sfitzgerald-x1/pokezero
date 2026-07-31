"""Public contract for the C26 row-identity and replay-boundary readout."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
READOUT = REPO_ROOT / "reports/c26_damage_composition_tail_readout.json"

WHAT_IDENTITIES = {
    "2000261/31",
    "2000298/23",
    "2000431/32",
    "2000561/67",
    "2100079/7",
    "2400156/29",
    "2401127/54",
    "2500120/60",
    "2500576/7",
    "2600657/49",
    "2601196/46",
}


class C26DamageCompositionReadoutTest(unittest.TestCase):
    def test_current_main_replays_every_c15_identity(self) -> None:
        readout = json.loads(READOUT.read_text())

        self.assertEqual(readout["schema"], "c26-damage-composition-tail-readout/1")
        self.assertEqual(set(readout["source_population"]["identities"]), WHAT_IDENTITIES)
        self.assertEqual(readout["current_main_control"]["ref"], "origin/main")
        rows = readout["current_main_control"]["rows"]
        self.assertEqual({row["identity"] for row in rows}, WHAT_IDENTITIES)
        self.assertTrue(all(row["verdict"] == "matched" for row in rows))
        self.assertTrue(all(row["branches"] > 0 for row in rows))

    def test_readout_does_not_turn_scope_or_family_into_c26_ownership(self) -> None:
        readout = json.loads(READOUT.read_text())
        matrix = readout["ownership_matrix"]

        self.assertEqual(matrix["closed_by_pr_980_exact_identity_evidence"], [])
        self.assertEqual(set(matrix["closed_by_current_main"]), WHAT_IDENTITIES)
        self.assertEqual(matrix["active_matcher_poison_tail_exactly_cleared"], [])
        self.assertEqual(matrix["c27_rest"], [])
        self.assertEqual(matrix["genuinely_unresolved_or_refused"], [])
        self.assertTrue(readout["invariants"]["no_family_name_is_used_as_ownership_evidence"])
        self.assertTrue(readout["invariants"]["current_main_control_replays_every_c15_identity"])

    def test_historical_c26_targets_are_fail_closed_without_their_trace(self) -> None:
        readout = json.loads(READOUT.read_text())
        targets = {row["identity"]: row for row in readout["c26_historical_targets"]}

        for identity in ("2900889/126", "3400914/75"):
            target = targets[identity]
            self.assertEqual(target["current_replay_status"], "refused_missing_pinned_protocol")
            self.assertFalse(target["available_run"]["identity_observed"])
            self.assertLess(
                target["available_run"]["boundaries_full_round"],
                target["available_run"]["required_step"],
            )
        self.assertTrue(
            readout["invariants"][
                "no_c26_historical_target_is_claimed_matched_without_a_retained_identity_replay"
            ]
        )


if __name__ == "__main__":
    unittest.main()
