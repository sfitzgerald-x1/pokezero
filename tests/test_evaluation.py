import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.bootstrap import TEACHER_BOOTSTRAP_SCHEMA_VERSION
from pokezero.eval_cli import main as eval_cli_main
from pokezero.evaluation import PromotionGateConfig, evaluate_promotion_gate
from pokezero.neural_selfplay import NEURAL_SELFPLAY_RUN_SCHEMA_VERSION
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION


class PromotionGateTest(unittest.TestCase):
    def test_selfplay_manifest_passes_with_benchmark_strength_and_low_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_manifest(run_dir / "manifest.json", selfplay_manifest())

            result = evaluate_promotion_gate(
                run_dir,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.60,
                    min_benchmark_games=20,
                    max_collection_capped_rate=0.20,
                    max_benchmark_capped_rate=0.10,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.candidate_policy_id, "linear-selfplay-test-iter-0001")
        self.assertEqual(result.source_iteration, 1)
        self.assertEqual(result.benchmark_win_rate, 0.65)
        self.assertEqual(result.collection_capped_rate, 0.1)
        self.assertEqual(result.benchmark_capped_rate, 0.05)
        self.assertEqual(result.benchmark_opponents[0].opponent_policy_id, "random-legal")
        self.assertEqual(result.benchmark_opponents[0].win_rate, 0.65)

    def test_gate_fails_when_benchmark_is_missing_by_default(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(manifest_path)

        self.assertFalse(result.passed)
        self.assertEqual(result.benchmark_games, 0)
        self.assertIn("benchmark_available", failed_check_names(result))

    def test_gate_can_allow_missing_benchmark_for_smoke_runs(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(require_benchmark=False),
            )

        self.assertTrue(result.passed)

    def test_gate_fails_per_opponent_even_when_pooled_win_rate_clears_floor(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 19, 1, 0),
                ("simple-legal", 16, 4, 0),
                ("scripted-teacher", 6, 14, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.55,
                    min_benchmark_games=20,
                    max_collection_capped_rate=0.20,
                ),
            )

        self.assertFalse(result.passed)
        self.assertGreater(result.benchmark_win_rate, 0.55)
        self.assertEqual(
            {opponent.opponent_policy_id: opponent.win_rate for opponent in result.benchmark_opponents},
            {"random-legal": 0.95, "simple-legal": 0.8, "scripted-teacher": 0.3},
        )
        self.assertIn("benchmark_win_rate:scripted-teacher", failed_check_names(result))

    def test_gate_enforces_minimum_games_per_benchmark_opponent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, selfplay_manifest())

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(min_benchmark_games=50),
            )

        self.assertFalse(result.passed)
        self.assertIn("benchmark_games:random-legal", failed_check_names(result))

    def test_gate_can_target_required_benchmark_opponents_and_thresholds(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 19, 1, 0),
                ("scripted-teacher", 9, 11, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_games=20,
                    required_benchmark_opponents=("scripted-teacher",),
                    opponent_min_win_rates={"scripted-teacher": 0.40},
                ),
            )

        self.assertTrue(result.passed)
        checked_names = {check.name for check in result.checks}
        self.assertIn("benchmark_win_rate:scripted-teacher", checked_names)
        self.assertNotIn("benchmark_win_rate:random-legal", checked_names)

    def test_gate_fails_when_required_benchmark_opponent_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, selfplay_manifest())

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_games=20,
                    required_benchmark_opponents=("scripted-teacher",),
                ),
            )

        self.assertFalse(result.passed)
        self.assertIn("benchmark_opponent:scripted-teacher", failed_check_names(result))

    def test_gate_can_require_incumbent_delta_without_applying_generic_floor_to_incumbent(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 13, 7, 0),
                ("linear-selfplay-test-iter-0000", 18, 2, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.60,
                    min_incumbent_win_rate=0.55,
                    min_benchmark_games=20,
                    min_incumbent_games=20,
                    max_collection_capped_rate=0.20,
                    incumbent_policy_id="linear-selfplay-test-iter-0000",
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.gate_mode, "absolute_floor+incumbent_delta")
        self.assertEqual(result.incumbent_policy_id, "linear-selfplay-test-iter-0000")
        self.assertEqual(result.incumbent_win_rate, 0.9)
        self.assertEqual(result.incumbent_games, 20)
        self.assertGreater(result.incumbent_win_rate_lower_bound or 0.0, 0.50)
        self.assertEqual(result.benchmark_win_rate, 0.65)
        self.assertEqual(result.benchmark_games, 20)
        self.assertEqual([opponent.opponent_policy_id for opponent in result.benchmark_opponents], ["random-legal"])
        checked_names = {check.name for check in result.checks}
        self.assertIn("incumbent_win_rate:linear-selfplay-test-iter-0000", checked_names)
        self.assertIn("incumbent_win_rate_lower_bound:linear-selfplay-test-iter-0000", checked_names)
        self.assertNotIn("benchmark_win_rate:linear-selfplay-test-iter-0000", checked_names)

    def test_gate_auto_derives_selfplay_incumbent_from_previous_iteration(self) -> None:
        manifest = selfplay_manifest()
        previous_iteration = json.loads(json.dumps(manifest["iterations"][0]))
        latest_iteration = json.loads(json.dumps(manifest["iterations"][0]))
        previous_iteration["iteration"] = 1
        previous_iteration["training"]["model"]["policy_id"] = "linear-selfplay-test-iter-0001"
        latest_iteration["iteration"] = 2
        latest_iteration["checkpoint_path"] = "run/iteration-0002/linear-policy.json"
        latest_iteration["training"]["model"]["policy_id"] = "linear-selfplay-test-iter-0002"
        latest_iteration["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0002",
            rows=(
                ("random-legal", 13, 7, 0),
                ("linear-selfplay-test-iter-0001", 18, 2, 0),
            ),
        )
        manifest["iterations"] = [previous_iteration, latest_iteration]
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.60,
                    min_incumbent_win_rate=0.55,
                    min_benchmark_games=20,
                    min_incumbent_games=20,
                    max_collection_capped_rate=0.20,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.incumbent_policy_id, "linear-selfplay-test-iter-0001")
        self.assertEqual(result.gate_mode, "absolute_floor+incumbent_delta")

    def test_gate_supports_neural_selfplay_manifest(self) -> None:
        manifest = neural_selfplay_manifest()
        previous_iteration = json.loads(json.dumps(manifest["iterations"][0]))
        latest_iteration = json.loads(json.dumps(manifest["iterations"][0]))
        previous_iteration["iteration"] = 1
        previous_iteration["training"]["model_config"]["policy_id"] = "entity-test-iter-0001"
        latest_iteration["iteration"] = 2
        latest_iteration["checkpoint_path"] = "run/iteration-0002/transformer-policy.pt"
        latest_iteration["current_policy_spec"] = previous_iteration["next_current_policy_spec"]
        latest_iteration["training"]["model_config"]["policy_id"] = "entity-test-iter-0002"
        latest_iteration["benchmark"] = benchmark_payload(
            policy_id="entity-test-iter-0002",
            rows=(
                ("random-legal", 13, 7, 0),
                ("entity-test-iter-0001", 18, 2, 0),
            ),
        )
        manifest["iterations"] = [previous_iteration, latest_iteration]
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.60,
                    min_incumbent_win_rate=0.55,
                    min_benchmark_games=20,
                    min_incumbent_games=20,
                    max_collection_capped_rate=0.20,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.source_type, NEURAL_SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(result.candidate_policy_id, "entity-test-iter-0002")
        self.assertEqual(result.checkpoint_path, "run/iteration-0002/transformer-policy.pt")
        self.assertEqual(result.source_iteration, 2)
        self.assertEqual(result.incumbent_policy_id, "entity-test-iter-0001")
        self.assertEqual(result.gate_mode, "absolute_floor+incumbent_delta")

    def test_gate_rejects_statistically_thin_incumbent_point_estimate(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 13, 7, 0),
                ("linear-selfplay-test-iter-0000", 11, 9, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.60,
                    min_incumbent_win_rate=0.55,
                    min_benchmark_games=20,
                    min_incumbent_games=20,
                    max_collection_capped_rate=0.20,
                    incumbent_policy_id="linear-selfplay-test-iter-0000",
                ),
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.incumbent_win_rate, 0.55)
        self.assertLess(result.incumbent_win_rate_lower_bound or 1.0, 0.50)
        self.assertIn(
            "incumbent_win_rate_lower_bound:linear-selfplay-test-iter-0000",
            failed_check_names(result),
        )

    def test_gate_checks_incumbent_capped_rate_directly(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 13, 7, 0),
                ("linear-selfplay-test-iter-0000", 18, 2, 4),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.60,
                    min_incumbent_win_rate=0.55,
                    min_benchmark_games=20,
                    min_incumbent_games=20,
                    max_collection_capped_rate=0.20,
                    max_incumbent_capped_rate=0.10,
                    incumbent_policy_id="linear-selfplay-test-iter-0000",
                ),
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.incumbent_capped_rate, 0.20)
        self.assertIn("incumbent_capped_rate:linear-selfplay-test-iter-0000", failed_check_names(result))

    def test_gate_fails_when_incumbent_benchmark_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, selfplay_manifest())

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_games=20,
                    max_collection_capped_rate=0.20,
                    incumbent_policy_id="linear-selfplay-test-iter-0000",
                ),
            )

        self.assertFalse(result.passed)
        self.assertIn(
            "incumbent_benchmark_opponent:linear-selfplay-test-iter-0000",
            failed_check_names(result),
        )

    def test_bootstrap_manifest_checks_teacher_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, bootstrap_manifest())

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.50,
                    min_benchmark_games=20,
                    max_teacher_degradation_rate=0.0,
                ),
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.candidate_policy_id, "linear-bootstrap")
        self.assertEqual(result.teacher_degradation_rate, 0.1)
        self.assertIn("teacher_degradation_rate", failed_check_names(result))

    def test_eval_cli_gate_returns_nonzero_for_failed_gate_and_prints_json(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["collection_metrics"]["capped_games"] = 5
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["gate", str(manifest_path), "--json", "--min-benchmark-games", "20"])

            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertIn("collection_capped_rate", failed_check_names_from_payload(payload))

    def test_eval_cli_gate_prints_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, selfplay_manifest())

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--min-benchmark-win-rate",
                        "0.60",
                        "--min-benchmark-games",
                        "20",
                        "--max-collection-capped-rate",
                        "0.20",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("status: PASS", stdout.getvalue())
        self.assertIn("pooled_benchmark_win_rate: 0.650", stdout.getvalue())
        self.assertIn("random-legal: win_rate=0.650", stdout.getvalue())

    def test_eval_cli_gate_wires_required_opponent_and_threshold_overrides(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 19, 1, 0),
                ("scripted-teacher", 9, 11, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--benchmark-opponent",
                        "scripted-teacher",
                        "--opponent-win-rate",
                        "scripted-teacher=0.40",
                        "--min-benchmark-games",
                        "20",
                    ]
                )

        self.assertEqual(exit_code, 0)

    def test_eval_cli_gate_wires_incumbent_policy(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            rows=(
                ("random-legal", 13, 7, 0),
                ("linear-selfplay-test-iter-0000", 18, 2, 0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--min-benchmark-win-rate",
                        "0.60",
                        "--min-incumbent-win-rate",
                        "0.55",
                        "--min-benchmark-games",
                        "20",
                        "--min-incumbent-games",
                        "20",
                        "--max-collection-capped-rate",
                        "0.20",
                        "--incumbent-policy",
                        "linear-selfplay-test-iter-0000",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("mode: absolute_floor+incumbent_delta", stdout.getvalue())
        self.assertIn("incumbent_win_rate: 0.900", stdout.getvalue())
        self.assertIn("incumbent_win_rate_lower_bound:", stdout.getvalue())

    def test_eval_cli_profiles_json_lists_named_profiles(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(["profiles", "--json"])
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {profile["name"] for profile in payload["profiles"]},
            {"default", "long-run", "smoke"},
        )
        long_run = next(profile for profile in payload["profiles"] if profile["name"] == "long-run")
        self.assertEqual(long_run["gate"]["min_benchmark_games"], 100)
        self.assertGreater(long_run["gate"]["min_benchmark_win_rate"], 0.55)
        self.assertLess(long_run["gate"]["max_benchmark_capped_rate"], 0.10)

    def test_eval_cli_gate_smoke_profile_allows_missing_benchmark(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["gate", str(manifest_path), "--profile", "smoke", "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])

    def test_eval_cli_gate_smoke_profile_can_still_require_benchmark(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--profile",
                        "smoke",
                        "--require-benchmark",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIn("benchmark_available", failed_check_names_from_payload(payload))

    def test_eval_cli_gate_smoke_profile_relaxes_numeric_thresholds(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["collection_metrics"] = collection_metrics(games=10, capped_games=8)
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            wins=1,
            losses=19,
            capped_games=12,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as default_stdout:
                default_exit = eval_cli_main(["gate", str(manifest_path), "--json"])
            default_payload = json.loads(default_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as smoke_stdout:
                smoke_exit = eval_cli_main(["gate", str(manifest_path), "--profile", "smoke", "--json"])
            smoke_payload = json.loads(smoke_stdout.getvalue())

        self.assertEqual(default_exit, 2)
        self.assertIn("benchmark_win_rate:random-legal", failed_check_names_from_payload(default_payload))
        self.assertIn("collection_capped_rate", failed_check_names_from_payload(default_payload))
        self.assertEqual(smoke_exit, 0)
        self.assertTrue(smoke_payload["passed"])

    def test_eval_cli_gate_long_run_profile_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, selfplay_manifest())

            with patch("sys.stdout", new_callable=io.StringIO) as strict_stdout:
                strict_exit = eval_cli_main(["gate", str(manifest_path), "--profile", "long-run", "--json"])
            strict_payload = json.loads(strict_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as override_stdout:
                override_exit = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--profile",
                        "long-run",
                        "--min-benchmark-games",
                        "20",
                        "--min-benchmark-win-rate",
                        "0.55",
                        "--json",
                    ]
                )
            override_payload = json.loads(override_stdout.getvalue())

        self.assertEqual(strict_exit, 2)
        self.assertIn("benchmark_games:random-legal", failed_check_names_from_payload(strict_payload))
        self.assertEqual(override_exit, 0)
        self.assertTrue(override_payload["passed"])

    def test_eval_cli_gate_long_run_profile_enforces_stricter_benchmark_capped_rate(self) -> None:
        manifest = selfplay_manifest()
        manifest["iterations"][0]["benchmark"] = benchmark_payload(
            policy_id="linear-selfplay-test-iter-0001",
            wins=18,
            losses=2,
            capped_games=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as default_stdout:
                default_exit = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--min-benchmark-games",
                        "20",
                        "--json",
                    ]
                )
            default_payload = json.loads(default_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as long_run_stdout:
                long_run_exit = eval_cli_main(
                    [
                        "gate",
                        str(manifest_path),
                        "--profile",
                        "long-run",
                        "--min-benchmark-games",
                        "20",
                        "--json",
                    ]
                )
            long_run_payload = json.loads(long_run_stdout.getvalue())

        self.assertEqual(default_exit, 0)
        self.assertTrue(default_payload["passed"])
        self.assertEqual(long_run_exit, 2)
        self.assertIn("benchmark_capped_rate", failed_check_names_from_payload(long_run_payload))


def selfplay_manifest() -> dict:
    return {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": "run",
        "latest_checkpoint_path": "run/iteration-0001/linear-policy.json",
        "iterations": [
            {
                "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
                "iteration": 1,
                "checkpoint_path": "run/iteration-0001/linear-policy.json",
                "collection_metrics": collection_metrics(games=10, capped_games=1),
                "training": {"model": {"policy_id": "linear-selfplay-test-iter-0001"}},
                "benchmark": benchmark_payload(
                    policy_id="linear-selfplay-test-iter-0001",
                    wins=13,
                    losses=7,
                    capped_games=1,
                ),
            }
        ],
    }


def neural_selfplay_manifest() -> dict:
    return {
        "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": "run",
        "latest_checkpoint_path": "run/iteration-0001/transformer-policy.pt",
        "current_policy_spec": "neural:run/iteration-0001/transformer-policy.pt",
        "latest_accepted_checkpoint_path": "run/iteration-0001/transformer-policy.pt",
        "iterations": [
            {
                "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
                "iteration": 1,
                "checkpoint_path": "run/iteration-0001/transformer-policy.pt",
                "checkpoint_policy_spec": "neural:run/iteration-0001/transformer-policy.pt",
                "current_policy_spec": "random-legal",
                "next_current_policy_spec": "neural:run/iteration-0001/transformer-policy.pt",
                "collection_metrics": collection_metrics(games=10, capped_games=1),
                "training": {"model_config": {"policy_id": "entity-test-iter-0001"}},
                "benchmark": benchmark_payload(
                    policy_id="entity-test-iter-0001",
                    wins=13,
                    losses=7,
                    capped_games=1,
                ),
            }
        ],
    }


def bootstrap_manifest() -> dict:
    return {
        "schema_version": TEACHER_BOOTSTRAP_SCHEMA_VERSION,
        "checkpoint_path": "run/linear-bootstrap.json",
        "train_collection_metrics": collection_metrics(games=20, capped_games=0),
        "training": {"model": {"policy_id": "linear-bootstrap"}},
        "teacher_decision_summary": {
            "total_decisions": 10,
            "scripted_teacher_decisions": 10,
            "unknown_move_decisions": 1,
            "fallback_decisions": 0,
        },
        "benchmark": benchmark_payload(
            policy_id="linear-bootstrap",
            wins=12,
            losses=8,
            capped_games=0,
        ),
    }


def benchmark_payload(
    *,
    policy_id: str,
    wins: int | None = None,
    losses: int | None = None,
    capped_games: int | None = None,
    rows: tuple[tuple[str, int, int, int], ...] | None = None,
) -> dict:
    if rows is None:
        if wins is None or losses is None or capped_games is None:
            raise ValueError("wins, losses, and capped_games are required when rows is not provided.")
        rows = (("random-legal", wins, losses, capped_games),)
    return {
        "format_id": "gen3randombattle",
        "max_decision_rounds": 250,
        "games_per_matchup": max(wins + losses for _, wins, losses, _ in rows),
        "head_to_heads": [
            benchmark_row(policy_id=policy_id, opponent_id=opponent_id, wins=wins, losses=losses, capped_games=capped_games)
            for opponent_id, wins, losses, capped_games in rows
        ],
        "matchups": [],
    }


def benchmark_row(*, policy_id: str, opponent_id: str, wins: int, losses: int, capped_games: int) -> dict:
    games = wins + losses
    return {
        "label": f"{policy_id} vs {opponent_id}",
        "first_policy_id": policy_id,
        "second_policy_id": opponent_id,
        "games": games,
        "first_policy_wins": wins,
        "second_policy_wins": losses,
        "ties": 0,
        "capped_games": capped_games,
        "first_policy_win_rate": wins / games,
        "second_policy_win_rate": losses / games,
    }


def collection_metrics(*, games: int, capped_games: int) -> dict:
    return {
        "games": games,
        "elapsed_seconds": 1.0,
        "total_decision_rounds": games,
        "total_simulator_turns": games,
        "p1_wins": games - capped_games,
        "p2_wins": 0,
        "ties": 0,
        "capped_games": capped_games,
    }


def failed_check_names(result) -> set[str]:
    return {check.name for check in result.checks if not check.passed}


def failed_check_names_from_payload(payload: dict) -> set[str]:
    return {check["name"] for check in payload["checks"] if not check["passed"]}


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
