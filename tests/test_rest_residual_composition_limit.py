"""Keep the Rest residual lane closed until retained replay evidence exists."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "c25_rest_residual_composition_prediction.json"
LIMIT = ROOT / "docs" / "rest_residual_composition_comparison_limit.md"
RUST_PIN = ROOT / "rust" / "pokezero-search" / "tests" / "gen3_rest_residual_composition.rs"
SHOWDOWN_ORACLE = ROOT / "scripts" / "gen3_switch_differential.py"


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
        self.assertTrue(closure["comparison_limit"])
        self.assertEqual(closure["verification_status"], "limited_not_clearance")
        self.assertFalse(closure["production_change_licensed"])
        self.assertFalse(closure["production_scheduler_behavior_changed"])
        self.assertFalse(closure["matcher_issue_proven"])
        self.assertFalse(closure["composition_issue_proven"])
        self.assertFalse(closure["matcher_or_composition_issue_proven"])
        self.assertEqual(closure["residual_scheduler_change"], "refused")
        self.assertEqual(closure["row_replay"]["outcome"], "not_replayable")
        self.assertEqual(closure["build_provenance"]["status"], "not_attested")
        self.assertEqual(closure["build_provenance"]["clearance_effect"], "not_clearance")
        self.assertEqual(ROOT / closure["contract"], LIMIT)

    def test_all_planned_pins_have_results_and_are_engine_only_controls(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        closure = report["closure"]
        pins = report["verification_plan"]["focused_pins"]
        results = closure["focused_pin_results"]

        self.assertEqual(
            {pin["id"] for pin in pins},
            {
                "surviving_rest_leftovers_tail",
                "surviving_rest_leftovers_toxic_order",
                "terminal_rest_no_tail",
            },
        )
        self.assertEqual(set(results), {pin["id"] for pin in pins})
        for result in results.values():
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["scope"], "engine_only_control")

    def test_each_identity_records_opaque_negative_replay_evidence(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        rows = report["closure"]["negative_evidence"]["rows"]

        self.assertEqual(report["closure"]["negative_evidence"]["public_paths"], "opaque")
        self.assertEqual(
            [row["identity"] for row in rows],
            ["2901076/41", "3000156/47", "3500842/79"],
        )
        for row in rows:
            self.assertEqual(
                row["searched"]["scopes"],
                ["current checkout tracked content", "reachable local Git content"],
            )
            self.assertEqual(row["searched"]["refs"], ["HEAD", "--all"])
            self.assertEqual(row["searched"]["retained_input"], "absent")
            self.assertTrue(
                row["candidate_public_replay"]["candidate_url_label"].startswith(
                    "opaque-public-gen3randombattle-candidate-"
                )
            )
            self.assertEqual(row["candidate_public_replay"]["http_status"], 404)
            self.assertEqual(
                row["harness_import"]["status"], "blocked_missing_poke_engine"
            )
            self.assertEqual(
                row["harness_import"]["pokezero_search_status"],
                "not_reached_and_not_installed",
            )
            self.assertEqual(row["harness_import"]["run_status"], "not_run")
            self.assertEqual(row["final_disposition"], "not_replayable")

    def test_gate_results_are_structured_and_do_not_clear_the_limit(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        closure = report["closure"]
        required = report["verification_plan"]["required_gates"]
        results = closure["gate_results"]

        self.assertEqual({gate["id"] for gate in required}, set(results))
        fingerprint = results["engine_build_fingerprint"]
        self.assertEqual(fingerprint["status"], "not_attested")
        self.assertEqual(fingerprint["clearance_effect"], "not_clearance")
        self.assertIn("full wheel and crate rebuild/write", fingerprint["reason"])
        self.assertIn("Merely installing pokezero_search", fingerprint["reason"])
        for gate_id, result in results.items():
            if gate_id == "engine_build_fingerprint":
                continue
            self.assertEqual(result["status"], "passed")
            self.assertIn("not_production", result["clearance_effect"])
            self.assertTrue(result["clearance_effect"].endswith("clearance"))

        self.assertEqual(
            results["sibling_residual_rest_regressions"]["commands"],
            [
                "cargo test --test gen3_rest_fidelity",
                "cargo test --test gen3_rest_sleep_clause",
                "cargo test --test gen3_battle_end_residuals",
                "cargo test --test gen3_hazard_residual_fidelity",
                "cargo test --test gen3_residual_rounding_fidelity",
                "cargo test --test gen3_residual_speed_order",
            ],
        )
        self.assertNotIn("command", results["sibling_residual_rest_regressions"])
        self.assertEqual(
            results["showdown_rest_residual_oracle"]["command"],
            "PYTHONPATH=src python scripts/gen3_switch_differential.py "
            "--showdown-root <showdown-root> --only restresidualtail",
        )

    def test_limit_requires_scheduler_controls_and_reprovenance_to_reopen(self) -> None:
        contract = LIMIT.read_text(encoding="utf-8")
        rust_pin = RUST_PIN.read_text(encoding="utf-8")
        showdown_oracle = SHOWDOWN_ORACLE.read_text(encoding="utf-8")
        report = json.loads(REPORT.read_text(encoding="utf-8"))

        self.assertIn("No production change is licensed", contract)
        self.assertIn("must not be laundered", contract)
        self.assertIn("immutable raw reports", contract)
        self.assertIn("fn surviving_rest_turn_keeps_leftovers_tail", rust_pin)
        self.assertIn("fn surviving_rest_turn_orders_leftovers_before_toxic_tail", rust_pin)
        self.assertIn("fn terminal_rest_turn_never_readds_residual_tail", rust_pin)
        self.assertIn("leftovers < toxic", rust_pin)
        self.assertIn("pins the observable truncation", rust_pin)
        self.assertIn("not one particular internal battle-end guard", rust_pin)
        self.assertIn("unreplayable retained rows", rust_pin)
        self.assertIn("remains\n/// unknown", rust_pin)
        self.assertEqual(report["historical_claim"]["status"], "unverified")
        self.assertIn("prior unverified", report["historical_claim"]["summary"])
        self.assertIn("do not identify any retained-row mechanism", report["stop_rule"])
        self.assertIn('if name == "restresidualtail"', showdown_oracle)


if __name__ == "__main__":
    unittest.main()
