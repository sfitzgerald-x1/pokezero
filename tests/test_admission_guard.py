from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pokezero.admission_guard import AdmissionGuardConfig, validate_admission_guard
from pokezero.diversity_population import diversity_population_dashboard
from pokezero.evaluation_profiles import SMOKE_EVALUATION_PROFILE
from pokezero.refutation_cli import main as refutation_cli_main


class AdmissionGuardTest(unittest.TestCase):
    def test_rejects_vacuous_zero_floor_and_no_vectors(self) -> None:
        result = validate_admission_guard(
            {
                "admission": {
                    "min_win_rate": 0.0,
                    "vector_distance_threshold": 0.0,
                },
                "behavior_embedding": {"pairwise_distances": []},
            }
        )

        self.assertFalse(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertFalse(checks["strength_floor_positive"].passed)
        self.assertEqual(checks["strength_floor_positive"].observed, 0.0)
        self.assertFalse(checks["comparison_vectors_present"].passed)
        self.assertEqual(checks["comparison_vectors_present"].observed, 0)
        self.assertFalse(checks["vector_distance_threshold_positive"].passed)

    def test_accepts_positive_floor_and_pairwise_novelty_evidence(self) -> None:
        result = validate_admission_guard(
            {
                "config": {
                    "min_benchmark_win_rate": 0.20,
                    "opponent_min_win_rates": {
                        "max-damage": 0.20,
                    },
                },
                "thresholds": {
                    "behavior_cluster_distance": 0.05,
                },
                "behavior_embedding": {
                    "pairwise_distances": [
                        {"left": "candidate", "right": "obsv2-500k", "distance": 0.21},
                    ],
                },
            }
        )

        self.assertTrue(result.passed)
        payload = result.to_dict()
        self.assertEqual(payload["schema_version"], "pokezero.admission_guard.v1")
        sources = {check["name"]: check["source"] for check in payload["checks"]}
        self.assertEqual(sources["strength_floor_positive"], "config.min_benchmark_win_rate")
        self.assertEqual(sources["comparison_vectors_present"], "behavior_embedding.pairwise_distances")

    def test_rejects_zero_default_even_when_specific_opponent_threshold_is_positive(self) -> None:
        result = validate_admission_guard(
            {
                "gate": {
                    "min_benchmark_win_rate": 0.0,
                    "opponent_min_win_rates": {"max-damage": 0.2},
                },
                "behavior_embedding": {
                    "distance_threshold": 0.05,
                    "pairwise_distances": [{"left": "candidate", "right": "anchor", "distance": 0.2}],
                },
            }
        )

        self.assertFalse(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertFalse(checks["strength_floor_positive"].passed)
        self.assertEqual(checks["strength_floor_positive"].source, "gate.min_benchmark_win_rate")

    def test_rejects_distance_rows_that_do_not_clear_active_threshold(self) -> None:
        result = validate_admission_guard(
            {
                "min_benchmark_win_rate": 0.1,
                "behavior_embedding": {
                    "distance_threshold": 0.2,
                    "pairwise_distances": [{"left": "candidate", "right": "anchor", "distance": 0.0}],
                },
            }
        )

        self.assertFalse(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertFalse(checks["observed_vector_distance_meets_threshold"].passed)
        self.assertEqual(checks["observed_vector_distance_meets_threshold"].observed, 0.0)

    def test_rejects_public_smoke_profile_as_admission_gate(self) -> None:
        result = validate_admission_guard(
            {
                **SMOKE_EVALUATION_PROFILE.to_dict(),
                "behavior_embedding": {
                    "distance_threshold": 0.1,
                    "pairwise_distances": [{"left": "candidate", "right": "anchor", "distance": 0.2}],
                },
            }
        )

        self.assertFalse(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertEqual(checks["strength_floor_positive"].observed, 0.0)

    def test_accepts_real_dashboard_shape_with_payoff_rank_and_behavior_distance(self) -> None:
        dashboard = diversity_population_dashboard(
            [
                {
                    "label": "candidate",
                    "move_usage": {"Thunderbolt": 0.7, "Spikes": 0.3},
                    "move_class_usage": {"attack": {"rate": 0.7}, "hazard": {"rate": 0.3}},
                },
                {
                    "label": "anchor",
                    "move_usage": {"Thunderbolt": 1.0, "Spikes": 0.0},
                    "move_class_usage": {"attack": {"rate": 1.0}, "hazard": {"rate": 0.0}},
                },
            ],
            payoff_vectors={
                "candidate": {"anchor": 0.6},
                "anchor": {"candidate": 0.4},
            },
            thresholds={"behavior_cluster_distance": 0.01},
        )
        result = validate_admission_guard({"min_benchmark_win_rate": 0.1, **dashboard})

        self.assertTrue(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertEqual(checks["comparison_vectors_present"].source, "payoff_rank.member_count/opponent_count")
        self.assertTrue(
            str(checks["observed_vector_distance_meets_threshold"].source).startswith(
                "behavior_embedding.pairwise_distances"
            )
        )

    def test_can_require_multiple_comparison_vectors(self) -> None:
        result = validate_admission_guard(
            {
                "min_benchmark_win_rate": 0.1,
                "min_vector_distance": 0.05,
                "comparison_vectors": ["anchor-a"],
            },
            config=AdmissionGuardConfig(min_comparison_vectors=2),
        )

        self.assertFalse(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertFalse(checks["comparison_vectors_present"].passed)
        self.assertEqual(checks["comparison_vectors_present"].threshold, 2)

    def test_payoff_rank_self_only_does_not_count_as_comparison(self) -> None:
        result = validate_admission_guard(
            {
                "min_benchmark_win_rate": 0.1,
                "min_vector_distance": 0.05,
                "observed_vector_distance": 0.2,
                "payoff_rank": {
                    "member_count": 1,
                    "opponent_count": 1,
                },
            }
        )

        self.assertFalse(result.passed)
        checks = {check.name: check for check in result.checks}
        self.assertFalse(checks["comparison_vectors_present"].passed)
        self.assertEqual(checks["comparison_vectors_present"].observed, 0)

    def test_cli_returns_two_for_failed_guard_and_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "admission.json"
            output_path = Path(temp_dir) / "guard.json"
            input_path.write_text(
                json.dumps({"min_win_rate": 0.0, "comparison_vectors": []}),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "admission-guard",
                        "--input",
                        str(input_path),
                        "--out",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertFalse(json.loads(stdout.getvalue())["passed"])
            self.assertFalse(json.loads(output_path.read_text(encoding="utf-8"))["passed"])

    def test_cli_returns_zero_for_passing_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "admission.json"
            input_path.write_text(
                json.dumps(
                    {
                        "min_win_rate": 0.1,
                        "vector_distance_threshold": 0.2,
                        "reference_vectors": [{"member_id": "obsv2-500k"}],
                        "observed_vector_distance": 0.25,
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = refutation_cli_main(
                    [
                        "admission-guard",
                        "--input",
                        str(input_path),
                    ]
                )

            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
