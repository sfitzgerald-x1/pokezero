"""Keep the Rest residual lane closed until retained replay evidence exists."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "c25_rest_residual_composition_prediction.json"
LIMIT = ROOT / "docs" / "rest_residual_composition_comparison_limit.md"
RUST_PIN = ROOT / "rust" / "pokezero-search" / "tests" / "gen3_rest_residual_composition.rs"


class RestResidualCompositionLimitTests(unittest.TestCase):
    def test_unreplayable_rows_are_a_refusal_not_an_attribution(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        closure = report["closure"]

        self.assertEqual(report["status"], "closed_comparison_limit")
        self.assertEqual(
            report["scope"]["retained_rows"],
            [[2901076, 41], [3000156, 47], [3500842, 79]],
        )
        self.assertEqual(closure["disposition"], "comparison_limit")
        self.assertFalse(closure["production_change_licensed"])
        self.assertFalse(closure["matcher_or_composition_issue_proven"])
        self.assertEqual(closure["residual_scheduler_change"], "refused")
        self.assertEqual(closure["row_replay"]["outcome"], "not_replayable")
        self.assertEqual(closure["build_provenance"]["status"], "not_attested")
        self.assertEqual(ROOT / closure["contract"], LIMIT)

    def test_limit_requires_both_scheduler_controls_and_reprovenance_to_reopen(self) -> None:
        contract = LIMIT.read_text(encoding="utf-8")
        rust_pin = RUST_PIN.read_text(encoding="utf-8")

        self.assertIn("No production change is licensed", contract)
        self.assertIn("must not be laundered", contract)
        self.assertIn("immutable raw reports", contract)
        self.assertIn("fn surviving_rest_turn_keeps_leftovers_and_toxic_tail", rust_pin)
        self.assertIn("fn terminal_rest_turn_never_readds_residual_tail", rust_pin)


if __name__ == "__main__":
    unittest.main()
