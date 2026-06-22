import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.bootstrap import TEACHER_BOOTSTRAP_SCHEMA_VERSION
from pokezero.bootstrap_cli import build_arg_parser as build_bootstrap_arg_parser
from pokezero.eval_cli import main as eval_cli_main
from pokezero.eval_cli import build_arg_parser as build_eval_arg_parser
from pokezero.evaluation import PromotionGateConfig, evaluate_promotion_gate
from pokezero.neural_selfplay import NEURAL_SELFPLAY_RUN_SCHEMA_VERSION
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION
from pokezero.selfplay_cli import build_arg_parser as build_selfplay_arg_parser
import pokezero.source_metadata as source_metadata
from pokezero.source_metadata import collect_source_metadata


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

    def test_eval_cli_cpu_smoke_plan_prints_text_recipe(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--run-root",
                    "runs/local smoke",
                    "--python-binary",
                    "./.venv/bin/python",
                    "--showdown-root",
                    "/tmp/showdown root",
                    "--workers",
                    "2",
                ]
            )
        output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("cpu_smoke_plan:", output)
        self.assertIn("./.venv/bin/python -m pokezero.bootstrap_cli teacher", output)
        self.assertIn("--run-dir 'runs/local smoke/teacher-bootstrap'", output)
        self.assertIn("--showdown-root '/tmp/showdown root'", output)
        self.assertIn("./.venv/bin/python -m pokezero.selfplay_cli iterate", output)
        self.assertIn("--profile smoke", output)
        self.assertIn("--audit-profile smoke", output)
        self.assertIn("./.venv/bin/python -m pokezero.eval_cli audit-calibrate", output)
        self.assertIn("--compare-profile smoke", output)
        self.assertIn("--write-config 'runs/local smoke/smoke-audit-config.json'", output)
        self.assertIn("./.venv/bin/python -m pokezero.eval_cli audit 'runs/local smoke/selfplay'", output)
        self.assertIn("--audit-config 'runs/local smoke/smoke-audit-config.json'", output)

    def test_eval_cli_cpu_smoke_plan_prints_json_recipe(self) -> None:
        source = {
            "available": True,
            "repo_root": "/repo",
            "branch": "scott/source-metadata",
            "head": "abc123",
            "dirty": False,
        }
        with (
            patch("pokezero.eval_cli.collect_source_metadata", return_value=source),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--run-root",
                    "runs/smoke",
                    "--python-binary",
                    "./.venv/bin/python",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["run_root"], "runs/smoke")
        self.assertEqual(payload["python_binary"], "./.venv/bin/python")
        self.assertEqual(payload["showdown_root"], "/tmp/showdown")
        self.assertEqual(payload["audit_config_path"], "runs/smoke/smoke-audit-config.json")
        self.assertEqual(payload["source"], source)
        self.assertEqual([step["name"] for step in payload["steps"]], [
            "bootstrap teacher checkpoint",
            "run smoke self-play iteration loop",
            "inspect self-play report",
            "audit smoke run",
            "calibrate smoke audit config",
            "audit smoke run with calibrated config",
        ])
        self.assertIn("linear:runs/smoke/teacher-bootstrap/linear-bootstrap.json", payload["steps"][1]["argv"])
        self.assertIn("--fail-on-profile", payload["steps"][-2]["argv"])
        self.assertIn("--write-config", payload["steps"][-2]["argv"])
        self.assertIn("runs/smoke/smoke-audit-config.json", payload["steps"][-2]["argv"])
        self.assertIn("--audit-config", payload["steps"][-1]["argv"])
        self.assertIn("runs/smoke/smoke-audit-config.json", payload["steps"][-1]["argv"])

    def test_eval_cli_cpu_smoke_plan_can_insert_teacher_branch_preflight(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--run-root",
                    "runs/smoke",
                    "--python-binary",
                    "./.venv/bin/python",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--seed-start",
                    "7",
                    "--teacher-branch-preflight-games",
                    "3",
                    "--require-teacher-branch",
                    "status_pressure",
                    "--min-teacher-branch-count",
                    "status_pressure=1",
                    "--json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["teacher_branch_preflight_requested"])
        self.assertEqual(payload["teacher_branch_preflight_games"], 3)
        self.assertEqual(payload["teacher_branch_preflight_output_path"], "runs/smoke/teacher-branch-preflight.json")
        self.assertEqual(payload["required_teacher_branches"], ["status_pressure"])
        self.assertEqual(payload["min_teacher_branch_counts"], ["status_pressure=1"])
        self.assertEqual([step["name"] for step in payload["steps"]][:2], [
            "benchmark scripted teacher branch coverage",
            "bootstrap teacher checkpoint",
        ])
        self.assertEqual(len(payload["steps"]), 7)
        preflight_argv = payload["steps"][0]["argv"]
        self.assertEqual(
            preflight_argv[:4],
            ["./.venv/bin/python", "-m", "pokezero.bootstrap_cli", "teacher-benchmark"],
        )
        self.assertEqual(preflight_argv[preflight_argv.index("--games") + 1], "3")
        self.assertEqual(preflight_argv[preflight_argv.index("--seed-start") + 1], "3000007")
        self.assertIn("--json", preflight_argv)
        self.assertIn("--require-teacher-branch", preflight_argv)
        self.assertIn("status_pressure", preflight_argv)
        self.assertIn("--min-teacher-branch-count", preflight_argv)
        self.assertIn("status_pressure=1", preflight_argv)
        self.assertEqual(payload["steps"][0]["output_json_path"], "runs/smoke/teacher-branch-preflight.json")

        parsers = {
            "pokezero.bootstrap_cli": build_bootstrap_arg_parser(),
            "pokezero.selfplay_cli": build_selfplay_arg_parser(),
            "pokezero.eval_cli": build_eval_arg_parser(),
        }
        for step in payload["steps"]:
            argv = step["argv"]
            parser = parsers[argv[2]]
            with self.subTest(step=step["name"]):
                parser.parse_args(argv[3:])

    def test_eval_cli_cpu_smoke_plan_accepts_custom_audit_config_path(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--run-root",
                    "runs/smoke",
                    "--python-binary",
                    "./.venv/bin/python",
                    "--audit-config-path",
                    "runs/audit-configs/smoke.json",
                    "--json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audit_config_path"], "runs/audit-configs/smoke.json")
        self.assertIn("runs/audit-configs/smoke.json", payload["steps"][-2]["argv"])
        self.assertIn("runs/audit-configs/smoke.json", payload["steps"][-1]["argv"])

    def test_eval_cli_cpu_smoke_plan_omits_showdown_root_when_unset(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--run-root",
                    "runs/smoke",
                    "--python-binary",
                    "./.venv/bin/python",
                    "--json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertIsNone(payload["showdown_root"])
        self.assertNotIn("--showdown-root", payload["steps"][0]["argv"])
        self.assertNotIn("/path/to/pokemon-showdown", payload["steps"][0]["command"])

    def test_eval_cli_cpu_smoke_plan_commands_parse_with_target_clis(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--run-root",
                    "runs/smoke",
                    "--python-binary",
                    "./.venv/bin/python",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        parsers = {
            "pokezero.bootstrap_cli": build_bootstrap_arg_parser(),
            "pokezero.selfplay_cli": build_selfplay_arg_parser(),
            "pokezero.eval_cli": build_eval_arg_parser(),
        }
        for step in payload["steps"]:
            argv = step["argv"]
            self.assertEqual(argv[:2], ["./.venv/bin/python", "-m"])
            parser = parsers[argv[2]]
            with self.subTest(step=step["name"]):
                parser.parse_args(argv[3:])

    def test_git_source_metadata_is_optional_outside_git_checkout(self) -> None:
        with patch("pokezero.source_metadata._git_output", side_effect=OSError("git unavailable")):
            metadata = collect_source_metadata(Path("/tmp/not-a-repo"))

        self.assertFalse(metadata["available"])
        self.assertIsNone(metadata["repo_root"])
        self.assertIsNone(metadata["head"])
        self.assertIsNone(metadata["dirty"])
        self.assertIn("git unavailable", metadata["error"])

    def test_git_source_metadata_defaults_to_package_source_location(self) -> None:
        calls = []

        def fake_git_output(cwd: Path, *args: str) -> str:
            calls.append((cwd, args))
            outputs = {
                ("rev-parse", "--show-toplevel"): "/repo",
                ("rev-parse", "HEAD"): "abc123",
                ("branch", "--show-current"): "main",
                ("status", "--porcelain"): "",
            }
            return outputs[args]

        with patch("pokezero.source_metadata._git_output", side_effect=fake_git_output):
            metadata = collect_source_metadata()

        expected_cwd = Path(source_metadata.__file__).resolve().parent
        self.assertTrue(metadata["available"])
        self.assertTrue(calls)
        self.assertEqual({cwd for cwd, _args in calls}, {expected_cwd})

    def test_git_source_metadata_collects_branch_head_and_dirty_state(self) -> None:
        def fake_git_output(_cwd: Path, *args: str) -> str:
            outputs = {
                ("rev-parse", "--show-toplevel"): "/repo",
                ("rev-parse", "HEAD"): "abc123",
                ("branch", "--show-current"): "main",
                ("status", "--porcelain"): "?? uv.lock\n",
            }
            return outputs[args]

        with patch("pokezero.source_metadata._git_output", side_effect=fake_git_output):
            metadata = collect_source_metadata(Path("/repo"))

        self.assertTrue(metadata["available"])
        self.assertEqual(metadata["repo_root"], "/repo")
        self.assertEqual(metadata["branch"], "main")
        self.assertEqual(metadata["head"], "abc123")
        self.assertTrue(metadata["dirty"])

    def test_git_source_metadata_maps_detached_head_branch_to_none(self) -> None:
        def fake_git_output(_cwd: Path, *args: str) -> str:
            outputs = {
                ("rev-parse", "--show-toplevel"): "/repo",
                ("rev-parse", "HEAD"): "abc123",
                ("branch", "--show-current"): "",
                ("status", "--porcelain"): "",
            }
            return outputs[args]

        with patch("pokezero.source_metadata._git_output", side_effect=fake_git_output):
            metadata = collect_source_metadata(Path("/repo"))

        self.assertTrue(metadata["available"])
        self.assertIsNone(metadata["branch"])
        self.assertFalse(metadata["dirty"])

    def test_git_source_metadata_is_optional_on_timeout(self) -> None:
        timeout = subprocess.TimeoutExpired(cmd=("git", "status", "--porcelain"), timeout=5)
        with patch("pokezero.source_metadata._git_output", side_effect=timeout):
            metadata = collect_source_metadata(Path("/repo"))

        self.assertFalse(metadata["available"])
        self.assertIsNone(metadata["head"])
        self.assertIn("TimeoutExpired", metadata["error"])

    def test_eval_cli_cpu_smoke_plan_rejects_non_positive_counts(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = eval_cli_main(["cpu-smoke-plan", "--workers", "0"])

        self.assertEqual(exit_code, 1)
        self.assertIn("workers must be positive", stderr.getvalue())

    def test_eval_cli_cpu_smoke_plan_rejects_non_positive_teacher_branch_preflight_games(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-plan",
                    "--teacher-branch-preflight-games",
                    "0",
                    "--require-teacher-branch",
                    "status_pressure",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("teacher-branch-preflight-games must be positive", stderr.getvalue())

    def test_eval_cli_cpu_smoke_run_executes_recipe_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            showdown_root = Path(temp_dir) / "showdown"
            showdown_root.mkdir()
            run_root = Path(temp_dir) / "runs" / "smoke"
            source = {
                "available": True,
                "repo_root": "/repo",
                "branch": "scott/source-metadata",
                "head": "abc123",
                "dirty": True,
            }
            with (
                patch("pokezero.eval_cli.collect_source_metadata", return_value=source),
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[SimpleNamespace(returncode=0) for _ in range(6)],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                    ]
                )
            summary = json.loads((run_root / "cpu-smoke-run-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 6)
        first_argv = run.call_args_list[0].args[0]
        self.assertEqual(first_argv[:4], ["./.venv/bin/python", "-m", "pokezero.bootstrap_cli", "teacher"])
        second_argv = run.call_args_list[1].args[0]
        self.assertIn("--profile", second_argv)
        self.assertIn("smoke", second_argv)
        config_argv = run.call_args_list[4].args[0]
        self.assertIn("--write-config", config_argv)
        self.assertIn(str(run_root / "smoke-audit-config.json"), config_argv)
        audit_config_argv = run.call_args_list[5].args[0]
        self.assertIn("--audit-config", audit_config_argv)
        self.assertIn(str(run_root / "smoke-audit-config.json"), audit_config_argv)
        output = stdout.getvalue()
        self.assertIn("cpu_smoke_run:", output)
        self.assertIn("running_step: 1/6 bootstrap teacher checkpoint", output)
        self.assertIn("cpu_smoke_run: PASS", output)
        self.assertEqual(summary["schema_version"], "pokezero.cpu_smoke_run_summary.v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["failed_step"], None)
        self.assertEqual(summary["recipe"]["run_root"], str(run_root))
        self.assertEqual(summary["recipe"]["audit_config_path"], str(run_root / "smoke-audit-config.json"))
        self.assertEqual(summary["recipe"]["seed_start"], 1)
        self.assertEqual(summary["source"], source)
        self.assertEqual(summary["recipe"]["source"], source)
        self.assertEqual(len(summary["steps"]), 6)
        self.assertEqual([step["status"] for step in summary["steps"]], ["passed"] * 6)
        self.assertEqual([step["returncode"] for step in summary["steps"]], [0] * 6)
        self.assertIn("started_at", summary)
        self.assertIn("ended_at", summary)
        self.assertIsInstance(summary["duration_seconds"], float)

    def test_eval_cli_cpu_smoke_run_executes_teacher_branch_preflight_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            showdown_root = Path(temp_dir) / "showdown"
            showdown_root.mkdir()
            run_root = Path(temp_dir) / "runs" / "smoke"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[
                        SimpleNamespace(
                            returncode=0,
                            stdout='{"passed": true, "teacher_decision_summary": {"teacher_branch_counts": {"status_pressure": 3}}}\n',
                            stderr="",
                        ),
                        *[SimpleNamespace(returncode=0) for _ in range(6)],
                    ],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                        "--require-teacher-branch",
                        "status_pressure",
                        "--min-teacher-branch-count",
                        "status_pressure=1",
                    ]
                )
            summary = json.loads((run_root / "cpu-smoke-run-summary.json").read_text(encoding="utf-8"))
            preflight_artifact = json.loads((run_root / "teacher-branch-preflight.json").read_text(encoding="utf-8"))
            with patch("sys.stdout", new_callable=io.StringIO) as report_stdout:
                report_exit_code = eval_cli_main(["cpu-smoke-report", str(run_root)])
            report_output = report_stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 7)
        self.assertEqual(
            run.call_args_list[0].args[0][:4],
            ["./.venv/bin/python", "-m", "pokezero.bootstrap_cli", "teacher-benchmark"],
        )
        self.assertEqual(
            run.call_args_list[1].args[0][:4],
            ["./.venv/bin/python", "-m", "pokezero.bootstrap_cli", "teacher"],
        )
        self.assertTrue(summary["recipe"]["teacher_branch_preflight_requested"])
        self.assertEqual(
            summary["recipe"]["teacher_branch_preflight_output_path"],
            str(run_root / "teacher-branch-preflight.json"),
        )
        self.assertEqual(summary["recipe"]["required_teacher_branches"], ["status_pressure"])
        self.assertEqual(summary["recipe"]["min_teacher_branch_counts"], ["status_pressure=1"])
        self.assertEqual(len(summary["steps"]), 7)
        self.assertEqual(summary["steps"][0]["name"], "benchmark scripted teacher branch coverage")
        self.assertEqual(summary["steps"][0]["output_json_path"], str(run_root / "teacher-branch-preflight.json"))
        self.assertTrue(summary["steps"][0]["output_json_written"])
        self.assertTrue(summary["steps"][0]["output_json_valid"])
        self.assertTrue(preflight_artifact["passed"])
        self.assertEqual(
            preflight_artifact["teacher_decision_summary"]["teacher_branch_counts"]["status_pressure"],
            3,
        )
        self.assertIn("running_step: 1/7 benchmark scripted teacher branch coverage", stdout.getvalue())
        self.assertEqual(report_exit_code, 0)
        self.assertIn("teacher_branch_preflight: PASS", report_output)
        self.assertIn("- status_pressure: 3", report_output)

    def test_eval_cli_cpu_smoke_run_stops_on_failed_teacher_branch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            showdown_root = Path(temp_dir) / "showdown"
            showdown_root.mkdir()
            run_root = Path(temp_dir) / "runs" / "smoke"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=2,
                        stdout='{"passed": false, "checks": [{"name": "teacher_branch_present:status_pressure"}]}\n',
                        stderr="",
                    ),
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                        "--require-teacher-branch",
                        "status_pressure",
                    ]
                )
            summary = json.loads((run_root / "cpu-smoke-run-summary.json").read_text(encoding="utf-8"))
            preflight_artifact = json.loads((run_root / "teacher-branch-preflight.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["failed_step"],
            {"index": 1, "name": "benchmark scripted teacher branch coverage", "returncode": 2},
        )
        self.assertEqual(len(summary["steps"]), 1)
        self.assertEqual(summary["steps"][0]["output_json_written"], True)
        self.assertEqual(summary["steps"][0]["output_json_valid"], True)
        self.assertFalse(preflight_artifact["passed"])
        self.assertIn("step 1 failed with exit code 2", stderr.getvalue())

    def test_eval_cli_cpu_smoke_plan_can_offset_recipe_seeds(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exit_code = eval_cli_main(["cpu-smoke-plan", "--seed-start", "42", "--json"])
        recipe = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(recipe["seed_start"], 42)
        bootstrap_argv = recipe["steps"][0]["argv"]
        self.assertIn("--seed-start", bootstrap_argv)
        self.assertEqual(bootstrap_argv[bootstrap_argv.index("--seed-start") + 1], "42")
        self.assertEqual(bootstrap_argv[bootstrap_argv.index("--shuffle-seed") + 1], "42")
        selfplay_argv = recipe["steps"][1]["argv"]
        self.assertEqual(selfplay_argv[selfplay_argv.index("--seed-start") + 1], "4000042")
        self.assertEqual(selfplay_argv[selfplay_argv.index("--evaluation-seed-start") + 1], "5000042")

    def test_eval_cli_cpu_smoke_run_executes_real_audit_config_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "smoke"
            audit_config_path = run_root / "smoke-audit-config.json"

            def fake_run(argv):
                if argv[2:4] == ["pokezero.selfplay_cli", "iterate"]:
                    manifest = selfplay_manifest()
                    manifest["run_dir"] = str(run_root / "selfplay")
                    write_manifest(run_root / "selfplay" / "manifest.json", manifest)
                if argv[2:4] == ["pokezero.eval_cli", "audit"]:
                    with patch("sys.stdout", new_callable=io.StringIO):
                        return SimpleNamespace(returncode=eval_cli_main(argv[3:]))
                if argv[2:4] == ["pokezero.eval_cli", "audit-calibrate"]:
                    with patch("sys.stdout", new_callable=io.StringIO):
                        return SimpleNamespace(returncode=eval_cli_main(argv[3:]))
                return SimpleNamespace(returncode=0)

            with (
                patch("pokezero.eval_cli.subprocess.run", side_effect=fake_run),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                    ]
                )
            summary = json.loads((run_root / "cpu-smoke-run-summary.json").read_text(encoding="utf-8"))
            audit_config = json.loads(audit_config_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["steps"][3]["status"], "passed")
        self.assertEqual(summary["steps"][4]["status"], "passed")
        self.assertEqual(summary["steps"][5]["status"], "passed")
        self.assertEqual(audit_config["schema_version"], "pokezero.run_audit_config.v1")
        self.assertEqual(audit_config["calibration"]["run_count"], 1)

    def test_eval_cli_cpu_smoke_run_honors_custom_summary_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "smoke"
            summary_path = temp_path / "custom" / "nested" / "summary.json"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[SimpleNamespace(returncode=0) for _ in range(6)],
                ),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--summary-path",
                        str(summary_path),
                        "--showdown-root",
                        str(showdown_root),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            default_summary_exists = (run_root / "cpu-smoke-run-summary.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["summary_path"], str(summary_path))
        self.assertEqual(summary["status"], "passed")
        self.assertFalse(default_summary_exists)

    def test_eval_cli_cpu_smoke_run_writes_running_step_before_subprocess(self) -> None:
        observed_statuses = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "smoke"
            summary_path = run_root / "cpu-smoke-run-summary.json"

            def fake_run(_argv):
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                observed_statuses.append(payload["steps"][-1]["status"])
                return SimpleNamespace(returncode=0)

            with (
                patch("pokezero.eval_cli.subprocess.run", side_effect=fake_run),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--showdown-root",
                        str(showdown_root),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_statuses, ["running"] * 6)

    def test_eval_cli_cpu_smoke_run_warns_and_continues_after_summary_update_failure(self) -> None:
        write_count = 0

        def flaky_write(_path, _payload):
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise OSError("disk full")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "smoke"
            with (
                patch("pokezero.eval_cli._write_json_payload", side_effect=flaky_write),
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[SimpleNamespace(returncode=0) for _ in range(6)],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--showdown-root",
                        str(showdown_root),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 6)
        self.assertEqual(write_count, 2)
        self.assertIn("warning: failed to update cpu smoke summary", stderr.getvalue())

    def test_eval_cli_cpu_smoke_run_stops_on_failed_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            showdown_root = Path(temp_dir) / "showdown"
            showdown_root.mkdir()
            run_root = Path(temp_dir) / "runs" / "smoke"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=7)],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-smoke-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                    ]
                )
            summary = json.loads((run_root / "cpu-smoke-run-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 7)
        self.assertEqual(run.call_count, 2)
        self.assertIn("cpu smoke step 2 failed with exit code 7", stderr.getvalue())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["failed_step"],
            {"index": 2, "name": "run smoke self-play iteration loop", "returncode": 7},
        )
        self.assertEqual(len(summary["steps"]), 2)
        self.assertEqual([step["status"] for step in summary["steps"]], ["passed", "failed"])
        self.assertEqual([step["returncode"] for step in summary["steps"]], [0, 7])

    def test_eval_cli_cpu_smoke_run_rejects_missing_explicit_showdown_root(self) -> None:
        missing_root = "/tmp/pokezero-missing-showdown-root"
        with (
            patch("pokezero.eval_cli.subprocess.run") as run,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = eval_cli_main(
                [
                    "cpu-smoke-run",
                    "--run-root",
                    "runs/smoke",
                    "--showdown-root",
                    missing_root,
                ]
            )

        self.assertEqual(exit_code, 1)
        run.assert_not_called()
        self.assertIn(f"showdown-root does not exist: {missing_root}", stderr.getvalue())

    def test_eval_cli_cpu_smoke_report_prints_passed_summary_from_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            summary_path = run_root / "cpu-smoke-run-summary.json"
            write_json(summary_path, cpu_smoke_summary(status="passed"))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(run_root)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("cpu_smoke_report:", output)
        self.assertIn("status: PASS", output)
        self.assertIn("source_available: True", output)
        self.assertIn("source_branch: main", output)
        self.assertIn("source_head: abc123", output)
        self.assertIn("source_dirty: False", output)
        self.assertIn("failed_step: -", output)
        self.assertIn("- 1: PASS bootstrap teacher checkpoint returncode=0", output)

    def test_eval_cli_cpu_smoke_report_failed_summary_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, cpu_smoke_summary(status="failed", failed_step_index=2))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("status: FAIL", output)
        self.assertIn("failed_step: 2 run smoke self-play iteration loop returncode=7", output)
        self.assertIn("- 2: FAIL run smoke self-play iteration loop returncode=7", output)

    def test_eval_cli_cpu_smoke_report_running_summary_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, cpu_smoke_summary(status="running"))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path)])

        self.assertEqual(exit_code, 2)
        self.assertIn("status: RUNNING", stdout.getvalue())

    def test_eval_cli_cpu_smoke_report_unknown_status_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, cpu_smoke_summary(status="stale"))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path)])

        self.assertEqual(exit_code, 2)
        self.assertIn("status: STALE", stdout.getvalue())

    def test_eval_cli_cpu_smoke_report_json_includes_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, cpu_smoke_summary(status="passed"))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["summary_source_path"], str(summary_path))
        self.assertEqual(payload["status"], "passed")

    def test_eval_cli_cpu_smoke_report_prints_teacher_branch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            preflight_path = run_root / "teacher-branch-preflight.json"
            summary = cpu_smoke_summary(status="passed")
            summary["recipe"].update(
                {
                    "teacher_branch_preflight_requested": True,
                    "teacher_branch_preflight_output_path": str(preflight_path),
                    "required_teacher_branches": ["status_pressure"],
                    "min_teacher_branch_counts": ["status_pressure=1"],
                }
            )
            write_json(
                preflight_path,
                {
                    "schema_version": "pokezero.teacher_benchmark.v1",
                    "passed": True,
                    "checks": [
                        {
                            "name": "teacher_branch_present:status_pressure",
                            "passed": True,
                            "message": "status_pressure observed.",
                            "observed": 6,
                            "threshold": 1,
                        }
                    ],
                    "teacher_decision_summary": {
                        "teacher_branch_counts": {"status_pressure": 6, "damaging_move": 12}
                    },
                },
            )
            write_json(run_root / "cpu-smoke-run-summary.json", summary)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(run_root)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("teacher_branch_preflight: PASS", output)
        self.assertIn(f"teacher_branch_preflight_path: {preflight_path}", output)
        self.assertIn("teacher_branch_counts:", output)
        self.assertIn("- damaging_move: 12", output)
        self.assertIn("- status_pressure: 6", output)

    def test_eval_cli_cpu_smoke_report_json_includes_teacher_branch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            preflight_path = run_root / "teacher-branch-preflight.json"
            summary_path = run_root / "cpu-smoke-run-summary.json"
            summary = cpu_smoke_summary(status="failed", failed_step_index=1)
            summary["recipe"].update(
                {
                    "teacher_branch_preflight_requested": True,
                    "teacher_branch_preflight_output_path": str(preflight_path),
                    "required_teacher_branches": ["status_pressure"],
                    "min_teacher_branch_counts": ["status_pressure=5"],
                }
            )
            write_json(
                preflight_path,
                {
                    "schema_version": "pokezero.teacher_benchmark.v1",
                    "passed": False,
                    "checks": [
                        {
                            "name": "teacher_branch_count:status_pressure",
                            "passed": False,
                            "message": "status_pressure count below required minimum.",
                            "observed": 3,
                            "threshold": 5,
                        }
                    ],
                    "teacher_decision_summary": {"teacher_branch_counts": {"status_pressure": 3}},
                },
            )
            write_json(summary_path, summary)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path), "--json"])
            payload = json.loads(stdout.getvalue())

        report = payload["teacher_branch_preflight_report"]
        self.assertEqual(exit_code, 2)
        self.assertTrue(report["requested"])
        self.assertTrue(report["available"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["schema_version"], "pokezero.teacher_benchmark.v1")
        self.assertEqual(report["teacher_branch_counts"], {"status_pressure": 3})
        self.assertEqual(report["required_teacher_branches"], ["status_pressure"])
        self.assertEqual(report["min_teacher_branch_counts"], ["status_pressure=5"])
        self.assertEqual(
            report["failed_checks"],
            [
                {
                    "name": "teacher_branch_count:status_pressure",
                    "passed": False,
                    "message": "status_pressure count below required minimum.",
                    "observed": 3,
                    "threshold": 5,
                }
            ],
        )

    def test_eval_cli_cpu_smoke_report_marks_missing_teacher_branch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "run"
            preflight_path = run_root / "teacher-branch-preflight.json"
            summary_path = run_root / "cpu-smoke-run-summary.json"
            summary = cpu_smoke_summary(status="passed")
            summary["recipe"].update(
                {
                    "teacher_branch_preflight_requested": True,
                    "teacher_branch_preflight_output_path": str(preflight_path),
                }
            )
            write_json(summary_path, summary)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path), "--json"])
            payload = json.loads(stdout.getvalue())

        report = payload["teacher_branch_preflight_report"]
        self.assertEqual(exit_code, 2)
        self.assertTrue(report["requested"])
        self.assertFalse(report["available"])
        self.assertIsNone(report["passed"])
        self.assertEqual(report["path"], str(preflight_path))
        self.assertEqual(report["error"], "teacher branch preflight artifact not found")

    def test_eval_cli_cpu_smoke_report_finds_relocated_teacher_branch_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_root = Path(temp_dir) / "original" / "run"
            relocated_root = Path(temp_dir) / "archive" / "run"
            original_preflight_path = original_root / "teacher-branch-preflight.json"
            relocated_preflight_path = relocated_root / "teacher-branch-preflight.json"
            summary = cpu_smoke_summary(status="passed")
            summary["recipe"].update(
                {
                    "teacher_branch_preflight_requested": True,
                    "teacher_branch_preflight_output_path": str(original_preflight_path),
                }
            )
            write_json(
                relocated_preflight_path,
                {
                    "schema_version": "pokezero.teacher_benchmark.v1",
                    "passed": True,
                    "checks": [],
                    "teacher_decision_summary": {"teacher_branch_counts": {"status_pressure": 4}},
                },
            )
            write_json(relocated_root / "cpu-smoke-run-summary.json", summary)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-smoke-report", str(relocated_root), "--json"])
            payload = json.loads(stdout.getvalue())

        report = payload["teacher_branch_preflight_report"]
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["available"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["recorded_path"], str(original_preflight_path))
        self.assertEqual(report["path"], str(relocated_preflight_path))
        self.assertEqual(report["teacher_branch_counts"], {"status_pressure": 4})

    def test_eval_cli_cpu_smoke_report_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, {"schema_version": "old", "status": "passed"})

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["cpu-smoke-report", str(summary_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unsupported cpu smoke summary schema", stderr.getvalue())

    def test_eval_cli_cpu_smoke_report_missing_run_root_points_to_default_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "missing-run"

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["cpu-smoke-report", str(run_root)])

        self.assertEqual(exit_code, 1)
        self.assertIn(str(run_root / "cpu-smoke-run-summary.json"), stderr.getvalue())

    def test_eval_cli_cpu_pilot_plan_prints_seeded_pilot_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-plan",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                        "--seed-start",
                        "100",
                        "--seed-stride",
                        "25",
                        "--json",
                    ]
                )
            recipe = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(recipe["pilot_count"], 2)
        self.assertEqual(recipe["seed_start"], 100)
        self.assertEqual(recipe["seed_stride"], 25)
        self.assertEqual(recipe["manifest_glob"], str(run_root / "pilot-*" / "selfplay" / "manifest.json"))
        self.assertEqual(recipe["calibration_output_path"], str(run_root / "pilot-calibration-compare.json"))
        self.assertEqual(recipe["replay_output_path"], str(run_root / "pilot-audit-replay.json"))
        self.assertEqual(len(recipe["steps"]), 4)
        first_pilot_argv = recipe["steps"][0]["argv"]
        second_pilot_argv = recipe["steps"][1]["argv"]
        self.assertEqual(first_pilot_argv[first_pilot_argv.index("--run-root") + 1], str(run_root / "pilot-0001"))
        self.assertEqual(second_pilot_argv[second_pilot_argv.index("--run-root") + 1], str(run_root / "pilot-0002"))
        self.assertEqual(first_pilot_argv[first_pilot_argv.index("--seed-start") + 1], "100")
        self.assertEqual(second_pilot_argv[second_pilot_argv.index("--seed-start") + 1], "125")
        calibration_argv = recipe["steps"][2]["argv"]
        self.assertEqual(recipe["steps"][2]["output_json_path"], str(run_root / "pilot-calibration-compare.json"))
        self.assertIn("--write-audit-config", calibration_argv)
        self.assertIn(str(run_root / "pilot-audit-config.json"), calibration_argv)
        self.assertIn("--calibration-aggregate-mode", calibration_argv)
        self.assertEqual(calibration_argv[calibration_argv.index("--calibration-aggregate-mode") + 1], "envelope")
        self.assertIn("--calibration-require-run-count", calibration_argv)
        self.assertIn("2", calibration_argv)
        audit_argv = recipe["steps"][3]["argv"]
        self.assertEqual(recipe["steps"][3]["output_json_path"], str(run_root / "pilot-audit-replay.json"))
        self.assertIn("--audit-config", audit_argv)
        self.assertIn(str(run_root / "pilot-audit-config.json"), audit_argv)

    def test_eval_cli_cpu_pilot_plan_propagates_teacher_branch_preflight_to_smoke_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-plan",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                        "--teacher-branch-preflight-games",
                        "3",
                        "--require-teacher-branch",
                        "status_pressure",
                        "--min-teacher-branch-count",
                        "status_pressure=1",
                        "--json",
                    ]
                )
            recipe = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(recipe["teacher_branch_preflight_requested"])
        self.assertEqual(recipe["teacher_branch_preflight_games"], 3)
        self.assertEqual(recipe["required_teacher_branches"], ["status_pressure"])
        self.assertEqual(recipe["min_teacher_branch_counts"], ["status_pressure=1"])
        first_pilot_argv = recipe["steps"][0]["argv"]
        second_pilot_argv = recipe["steps"][1]["argv"]
        for argv in (first_pilot_argv, second_pilot_argv):
            self.assertIn("--teacher-branch-preflight-games", argv)
            self.assertEqual(argv[argv.index("--teacher-branch-preflight-games") + 1], "3")
            self.assertIn("--require-teacher-branch", argv)
            self.assertIn("status_pressure", argv)
            self.assertIn("--min-teacher-branch-count", argv)
            self.assertIn("status_pressure=1", argv)

    def test_eval_cli_cpu_pilot_run_executes_recipe_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"
            source = {
                "available": True,
                "repo_root": "/repo",
                "branch": "scott/pilots",
                "head": "abc123",
                "dirty": False,
            }
            with (
                patch("pokezero.eval_cli.collect_source_metadata", return_value=source),
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0, stdout='{"audit_calibration_sufficient": true}\n', stderr=""),
                        SimpleNamespace(returncode=0, stdout='{"audit_failed": false}\n', stderr=""),
                    ],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        "./.venv/bin/python",
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                        "--seed-start",
                        "200",
                        "--seed-stride",
                        "50",
                    ]
                )
            summary = json.loads((run_root / "cpu-pilot-suite-summary.json").read_text(encoding="utf-8"))
            calibration_artifact = json.loads((run_root / "pilot-calibration-compare.json").read_text(encoding="utf-8"))
            replay_artifact = json.loads((run_root / "pilot-audit-replay.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(summary["schema_version"], "pokezero.cpu_pilot_suite_summary.v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["failed_step"], None)
        self.assertEqual(summary["source"], source)
        self.assertEqual(summary["recipe"]["pilot_count"], 2)
        self.assertEqual(summary["recipe"]["seed_start"], 200)
        self.assertEqual(summary["recipe"]["calibration_output_path"], str(run_root / "pilot-calibration-compare.json"))
        self.assertEqual(summary["recipe"]["replay_output_path"], str(run_root / "pilot-audit-replay.json"))
        self.assertEqual(len(summary["steps"]), 4)
        self.assertEqual([step["status"] for step in summary["steps"]], ["passed"] * 4)
        self.assertEqual(summary["steps"][2]["output_json_path"], str(run_root / "pilot-calibration-compare.json"))
        self.assertEqual(summary["steps"][2]["output_json_written"], True)
        self.assertEqual(summary["steps"][2]["output_json_valid"], True)
        self.assertEqual(summary["steps"][3]["output_json_path"], str(run_root / "pilot-audit-replay.json"))
        self.assertEqual(summary["steps"][3]["output_json_written"], True)
        self.assertEqual(summary["steps"][3]["output_json_valid"], True)
        self.assertEqual(calibration_artifact["audit_calibration_sufficient"], True)
        self.assertEqual(replay_artifact["audit_failed"], False)
        first_pilot_argv = run.call_args_list[0].args[0]
        second_pilot_argv = run.call_args_list[1].args[0]
        self.assertEqual(first_pilot_argv[:4], ["./.venv/bin/python", "-m", "pokezero.eval_cli", "cpu-smoke-run"])
        self.assertEqual(first_pilot_argv[first_pilot_argv.index("--seed-start") + 1], "200")
        self.assertEqual(second_pilot_argv[second_pilot_argv.index("--seed-start") + 1], "250")
        calibration_argv = run.call_args_list[2].args[0]
        self.assertEqual(calibration_argv[:4], ["./.venv/bin/python", "-m", "pokezero.eval_cli", "compare"])
        self.assertEqual(calibration_argv[calibration_argv.index("--calibration-aggregate-mode") + 1], "envelope")
        self.assertIn("--write-audit-config", calibration_argv)
        output = stdout.getvalue()
        self.assertIn("cpu_pilot_run:", output)
        self.assertIn("running_step: 1/4 run CPU smoke pilot 1", output)
        self.assertIn("cpu_pilot_run: PASS", output)

    def test_eval_cli_cpu_pilot_run_composes_real_subprocesses_for_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            helper_path = temp_path / "pilot_child.py"
            helper_path.write_text(
                f"""#!{sys.executable}
import json
from pathlib import Path
import sys


def value_after(argv, flag, default=None):
    if flag not in argv:
        return default
    return argv[argv.index(flag) + 1]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collection_metrics():
    return {{
        "games": 4,
        "elapsed_seconds": 1.0,
        "total_decision_rounds": 4,
        "total_simulator_turns": 4,
        "p1_wins": 4,
        "p2_wins": 0,
        "ties": 0,
        "capped_games": 0,
    }}


def benchmark_payload(policy_id):
    wins = 1 if policy_id.endswith("300") else 4
    losses = 4 - wins
    return {{
        "format_id": "gen3randombattle",
        "max_decision_rounds": 250,
        "games_per_matchup": 4,
        "head_to_heads": [
            {{
                "label": f"{{policy_id}} vs random-legal",
                "first_policy_id": policy_id,
                "second_policy_id": "random-legal",
                "games": 4,
                "first_policy_wins": wins,
                "second_policy_wins": losses,
                "ties": 0,
                "capped_games": 0,
                "first_policy_win_rate": wins / 4,
                "second_policy_win_rate": losses / 4,
            }}
        ],
        "matchups": [],
    }}


def write_smoke_manifest(argv):
    run_root = Path(value_after(argv, "--run-root"))
    seed_start = value_after(argv, "--seed-start", "0")
    selfplay_dir = run_root / "selfplay"
    policy_id = f"pilot-child-{{seed_start}}"
    manifest = {{
        "schema_version": "pokezero.selfplay_run.v1",
        "run_dir": str(selfplay_dir),
        "latest_checkpoint_path": str(selfplay_dir / "iteration-0001" / "linear-policy.json"),
        "iterations": [
            {{
                "schema_version": "pokezero.selfplay_run.v1",
                "iteration": 1,
                "checkpoint_path": str(selfplay_dir / "iteration-0001" / "linear-policy.json"),
                "collection_metrics": collection_metrics(),
                "training": {{"model": {{"policy_id": policy_id}}}},
                "benchmark": benchmark_payload(policy_id),
            }}
        ],
    }}
    write_json(selfplay_dir / "manifest.json", manifest)
    write_json(
        run_root / "cpu-smoke-run-summary.json",
        {{
            "schema_version": "pokezero.cpu_smoke_run_summary.v1",
            "status": "passed",
            "summary_path": str(run_root / "cpu-smoke-run-summary.json"),
            "started_at": "2026-06-22T12:00:00.000Z",
            "ended_at": "2026-06-22T12:00:01.000Z",
            "duration_seconds": 1.0,
            "source": {{"available": False}},
            "recipe": {{"run_root": str(run_root), "seed_start": int(seed_start), "steps": []}},
            "steps": [],
            "failed_step": None,
        }},
    )
    return 0


def main(argv):
    if argv[:3] == ["-m", "pokezero.eval_cli", "cpu-smoke-run"]:
        return write_smoke_manifest(argv[3:])
    if argv[:3] == ["-m", "pokezero.eval_cli", "compare"]:
        from pokezero.eval_cli import main as eval_main
        return eval_main(argv[2:])
    print(f"unexpected argv: {{argv}}", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
""",
                encoding="utf-8",
            )
            helper_path.chmod(0o755)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"

            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-run",
                        "--run-root",
                        str(run_root),
                        "--python-binary",
                        str(helper_path),
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                        "--selfplay-iterations",
                        "1",
                        "--seed-start",
                        "300",
                        "--seed-stride",
                        "10",
                    ]
            )
            summary = json.loads((run_root / "cpu-pilot-suite-summary.json").read_text(encoding="utf-8"))
            audit_config = json.loads((run_root / "pilot-audit-config.json").read_text(encoding="utf-8"))
            calibration_artifact = json.loads((run_root / "pilot-calibration-compare.json").read_text(encoding="utf-8"))
            replay_artifact = json.loads((run_root / "pilot-audit-replay.json").read_text(encoding="utf-8"))
            pilot_1_manifest = json.loads((run_root / "pilot-0001" / "selfplay" / "manifest.json").read_text(encoding="utf-8"))
            pilot_2_manifest = json.loads((run_root / "pilot-0002" / "selfplay" / "manifest.json").read_text(encoding="utf-8"))
            pilot_1_manifest_exists = (run_root / "pilot-0001" / "selfplay" / "manifest.json").exists()
            pilot_2_manifest_exists = (run_root / "pilot-0002" / "selfplay" / "manifest.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual([step["status"] for step in summary["steps"]], ["passed"] * 4)
        self.assertEqual(summary["steps"][2]["output_json_valid"], True)
        self.assertEqual(summary["steps"][3]["output_json_valid"], True)
        self.assertEqual(summary["recipe"]["benchmark_iterations_required"], 2)
        self.assertEqual(calibration_artifact["audit_calibration_sufficient"], True)
        self.assertEqual(replay_artifact["audit_failed"], False)
        self.assertTrue(pilot_1_manifest_exists)
        self.assertTrue(pilot_2_manifest_exists)
        self.assertEqual(pilot_1_manifest["iterations"][0]["benchmark"]["head_to_heads"][0]["first_policy_win_rate"], 0.25)
        self.assertEqual(pilot_2_manifest["iterations"][0]["benchmark"]["head_to_heads"][0]["first_policy_win_rate"], 1.0)
        self.assertEqual(audit_config["schema_version"], "pokezero.run_audit_config.v1")
        self.assertEqual(audit_config["calibration"]["run_count"], 2)
        self.assertLessEqual(audit_config["config"]["min_latest_benchmark_win_rate"], 0.25)

    def test_eval_cli_cpu_pilot_run_stops_on_failed_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=9)],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-run",
                        "--run-root",
                        str(run_root),
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                    ]
                )
            summary = json.loads((run_root / "cpu-pilot-suite-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 9)
        self.assertEqual(run.call_count, 2)
        self.assertIn("cpu pilot step 2 failed with exit code 9", stderr.getvalue())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["failed_step"],
            {"index": 2, "name": "run CPU smoke pilot 2", "returncode": 9},
        )

    def test_eval_cli_cpu_pilot_run_fails_when_compare_emits_no_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0, stdout="", stderr=""),
                    ],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-run",
                        "--run-root",
                        str(run_root),
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                    ]
                )
            summary = json.loads((run_root / "cpu-pilot-suite-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 70)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["failed_step"],
            {"index": 3, "name": "compare pilots and write calibrated audit config", "returncode": 70},
        )
        self.assertEqual(summary["steps"][2]["output_json_written"], False)
        self.assertEqual(summary["steps"][2]["output_json_valid"], False)
        self.assertIn("expected JSON stdout for artifact step", stderr.getvalue())

    def test_eval_cli_cpu_pilot_run_persists_failed_replay_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(
                            returncode=0,
                            stdout=json.dumps(
                                {
                                    "audit_calibration_sufficient": True,
                                    "written_audit_config_path": str(run_root / "pilot-audit-config.json"),
                                }
                            ),
                            stderr="",
                        ),
                        SimpleNamespace(
                            returncode=2,
                            stdout=json.dumps(
                                {
                                    "audit_failed": True,
                                    "entries": [
                                        {"audit_failed_checks": ["latest_benchmark_win_rate", "latest_benchmark_games"]},
                                        {"audit_failed_checks": []},
                                    ],
                                }
                            ),
                            stderr="",
                        ),
                    ],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-run",
                        "--run-root",
                        str(run_root),
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                    ]
                )
            summary = json.loads((run_root / "cpu-pilot-suite-summary.json").read_text(encoding="utf-8"))
            replay_artifact = json.loads((run_root / "pilot-audit-replay.json").read_text(encoding="utf-8"))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                report_exit_code = eval_cli_main(["cpu-pilot-report", str(run_root)])

        self.assertEqual(exit_code, 2)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["failed_step"],
            {"index": 4, "name": "compare pilots with calibrated audit config", "returncode": 2},
        )
        self.assertEqual(summary["steps"][3]["output_json_written"], True)
        self.assertEqual(summary["steps"][3]["output_json_valid"], True)
        self.assertEqual(replay_artifact["audit_failed"], True)
        self.assertEqual(report_exit_code, 2)
        self.assertIn("replay_audit_failed: True", stdout.getvalue())
        self.assertIn("replay_failed_check_count: 2", stdout.getvalue())

    def test_eval_cli_cpu_pilot_run_fails_when_compare_emits_invalid_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            showdown_root = temp_path / "showdown"
            showdown_root.mkdir()
            run_root = temp_path / "runs" / "pilots"
            with (
                patch(
                    "pokezero.eval_cli.subprocess.run",
                    side_effect=[
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0),
                        SimpleNamespace(returncode=0, stdout="{not-json", stderr=""),
                    ],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                exit_code = eval_cli_main(
                    [
                        "cpu-pilot-run",
                        "--run-root",
                        str(run_root),
                        "--showdown-root",
                        str(showdown_root),
                        "--pilot-count",
                        "2",
                    ]
                )
            summary = json.loads((run_root / "cpu-pilot-suite-summary.json").read_text(encoding="utf-8"))
            artifact_text = (run_root / "pilot-calibration-compare.json").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 70)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["steps"][2]["output_json_written"], True)
        self.assertEqual(summary["steps"][2]["output_json_valid"], False)
        self.assertEqual(artifact_text, "{not-json")
        self.assertIn("expected valid JSON stdout for artifact step", stderr.getvalue())

    def test_eval_cli_cpu_pilot_run_rejects_missing_explicit_showdown_root(self) -> None:
        missing_root = "/tmp/pokezero-missing-showdown-root"
        with (
            patch("pokezero.eval_cli.subprocess.run") as run,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            exit_code = eval_cli_main(
                [
                    "cpu-pilot-run",
                    "--run-root",
                    "runs/pilots",
                    "--showdown-root",
                    missing_root,
                ]
            )

        self.assertEqual(exit_code, 1)
        run.assert_not_called()
        self.assertIn(f"showdown-root does not exist: {missing_root}", stderr.getvalue())

    def test_eval_cli_cpu_pilot_plan_rejects_seed_band_overlap(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = eval_cli_main(
                [
                    "cpu-pilot-plan",
                    "--pilot-count",
                    "101",
                    "--seed-stride",
                    "10000",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("pilot seed offsets must stay below the smoke seed-band spacing", stderr.getvalue())

    def test_eval_cli_cpu_pilot_report_prints_passed_summary_from_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "pilots"
            summary_path = run_root / "cpu-pilot-suite-summary.json"
            summary = cpu_pilot_summary(status="passed")
            summary["recipe"]["audit_config_path"] = str(run_root / "pilot-audit-config.json")
            summary["recipe"]["calibration_output_path"] = str(run_root / "pilot-calibration-compare.json")
            summary["recipe"]["replay_output_path"] = str(run_root / "pilot-audit-replay.json")
            write_json(summary_path, summary)
            write_json(
                run_root / "pilot-calibration-compare.json",
                {
                    "audit_calibration_sufficient": True,
                    "written_audit_config_path": str(run_root / "pilot-audit-config.json"),
                },
            )
            write_json(
                run_root / "pilot-audit-replay.json",
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-pilot-report", str(run_root)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("cpu_pilot_report:", output)
        self.assertIn("status: PASS", output)
        self.assertIn("pilot_count: 2", output)
        self.assertIn(f"calibration_output_path: {run_root / 'pilot-calibration-compare.json'}", output)
        self.assertIn(f"replay_output_path: {run_root / 'pilot-audit-replay.json'}", output)
        self.assertIn("calibration_sufficient: True", output)
        self.assertIn(f"calibration_written_audit_config_path: {run_root / 'pilot-audit-config.json'}", output)
        self.assertIn("calibration_audit_config_write_error: -", output)
        self.assertIn("replay_audit_failed: False", output)
        self.assertIn("replay_failed_check_count: 0", output)
        self.assertIn("audit_config_ready: yes", output)
        self.assertIn("failed_step: -", output)
        self.assertIn("- 1: PASS run CPU smoke pilot 1 returncode=0", output)

    def test_eval_cli_cpu_pilot_report_json_includes_artifact_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "pilots"
            summary_path = run_root / "cpu-pilot-suite-summary.json"
            summary = cpu_pilot_summary(status="passed")
            summary["recipe"]["audit_config_path"] = str(run_root / "pilot-audit-config.json")
            summary["recipe"]["calibration_output_path"] = str(run_root / "pilot-calibration-compare.json")
            summary["recipe"]["replay_output_path"] = str(run_root / "pilot-audit-replay.json")
            write_json(summary_path, summary)
            write_json(
                run_root / "pilot-calibration-compare.json",
                {
                    "audit_calibration_sufficient": True,
                    "written_audit_config_path": str(run_root / "pilot-audit-config.json"),
                },
            )
            write_json(
                run_root / "pilot-audit-replay.json",
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-pilot-report", str(run_root), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        artifact_report = payload["pilot_artifact_report"]
        self.assertEqual(artifact_report["audit_config_ready"], True)
        self.assertEqual(artifact_report["audit_config_ready_reasons"], [])
        self.assertEqual(artifact_report["calibration"]["available"], True)
        self.assertEqual(artifact_report["calibration"]["sufficient"], True)
        self.assertEqual(
            artifact_report["calibration"]["expected_audit_config_path"],
            str(run_root / "pilot-audit-config.json"),
        )
        self.assertEqual(
            artifact_report["calibration"]["written_audit_config_path"],
            str(run_root / "pilot-audit-config.json"),
        )
        self.assertEqual(artifact_report["replay"]["available"], True)
        self.assertEqual(artifact_report["replay"]["audit_failed"], False)
        self.assertEqual(artifact_report["replay"]["failed_check_count"], 0)

    def test_eval_cli_cpu_pilot_report_marks_missing_artifacts_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "pilots"
            summary_path = run_root / "cpu-pilot-suite-summary.json"
            summary = cpu_pilot_summary(status="passed")
            summary["recipe"]["calibration_output_path"] = str(run_root / "pilot-calibration-compare.json")
            summary["recipe"]["replay_output_path"] = str(run_root / "pilot-audit-replay.json")
            write_json(summary_path, summary)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-pilot-report", str(run_root)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("audit_config_ready: no", output)
        self.assertIn("- calibration_artifact_missing", output)
        self.assertIn("- replay_artifact_missing", output)

    def test_eval_cli_cpu_pilot_report_require_ready_fails_when_artifacts_are_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "pilots"
            summary_path = run_root / "cpu-pilot-suite-summary.json"
            summary = cpu_pilot_summary(status="passed")
            summary["recipe"]["calibration_output_path"] = str(run_root / "pilot-calibration-compare.json")
            summary["recipe"]["replay_output_path"] = str(run_root / "pilot-audit-replay.json")
            write_json(summary_path, summary)

            with patch("sys.stdout", new_callable=io.StringIO):
                default_exit_code = eval_cli_main(["cpu-pilot-report", str(run_root)])
            with patch("sys.stdout", new_callable=io.StringIO):
                required_exit_code = eval_cli_main(["cpu-pilot-report", str(run_root), "--require-ready"])

        self.assertEqual(default_exit_code, 0)
        self.assertEqual(required_exit_code, 2)

    def test_eval_cli_cpu_pilot_report_readiness_reasons_cover_artifact_failures(self) -> None:
        cases = [
            (
                "calibration_not_sufficient",
                "passed",
                {"audit_calibration_sufficient": False, "written_audit_config_path": "MATCH"},
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
                {"calibration_not_sufficient"},
            ),
            (
                "calibrated_audit_config_not_written",
                "passed",
                {"audit_calibration_sufficient": True},
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
                {"calibrated_audit_config_not_written"},
            ),
            (
                "calibrated_audit_config_write_error",
                "passed",
                {
                    "audit_calibration_sufficient": True,
                    "written_audit_config_path": "MATCH",
                    "audit_config_write_error": "write failed",
                },
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
                {"calibrated_audit_config_write_error"},
            ),
            (
                "calibrated_audit_config_path_mismatch",
                "passed",
                {"audit_calibration_sufficient": True, "written_audit_config_path": "/tmp/other-config.json"},
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
                {"calibrated_audit_config_path_mismatch"},
            ),
            (
                "replay_audit_failed",
                "passed",
                {"audit_calibration_sufficient": True, "written_audit_config_path": "MATCH"},
                {"audit_failed": True, "entries": [{"audit_failed_checks": ["latest_benchmark_win_rate"]}]},
                {"replay_audit_failed"},
            ),
            (
                "replay_failed_checks_present",
                "passed",
                {"audit_calibration_sufficient": True, "written_audit_config_path": "MATCH"},
                {"audit_failed": False, "entries": [{"audit_failed_checks": ["latest_benchmark_win_rate"]}]},
                {"replay_failed_checks_present"},
            ),
            (
                "suite_status_not_passed",
                "failed",
                {"audit_calibration_sufficient": True, "written_audit_config_path": "MATCH"},
                {"audit_failed": False, "entries": [{"audit_failed_checks": []}]},
                {"suite_status_not_passed"},
            ),
        ]
        for name, status, calibration, replay, expected_reasons in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                run_root = Path(temp_dir) / "pilots"
                summary_path = run_root / "cpu-pilot-suite-summary.json"
                summary = cpu_pilot_summary(status=status, failed_step_index=2 if status == "failed" else None)
                audit_config_path = run_root / "pilot-audit-config.json"
                summary["recipe"]["audit_config_path"] = str(audit_config_path)
                summary["recipe"]["calibration_output_path"] = str(run_root / "pilot-calibration-compare.json")
                summary["recipe"]["replay_output_path"] = str(run_root / "pilot-audit-replay.json")
                if calibration.get("written_audit_config_path") == "MATCH":
                    calibration = {**calibration, "written_audit_config_path": str(audit_config_path)}
                write_json(summary_path, summary)
                write_json(run_root / "pilot-calibration-compare.json", calibration)
                write_json(run_root / "pilot-audit-replay.json", replay)

                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = eval_cli_main(["cpu-pilot-report", str(run_root), "--json"])
                payload = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 2 if status == "failed" else 0)
            artifact_report = payload["pilot_artifact_report"]
            self.assertEqual(artifact_report["audit_config_ready"], False)
            self.assertTrue(expected_reasons.issubset(set(artifact_report["audit_config_ready_reasons"])))

    def test_eval_cli_cpu_pilot_report_json_allows_missing_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(
                summary_path,
                {
                    "schema_version": "pokezero.cpu_pilot_suite_summary.v1",
                    "status": "passed",
                    "steps": [],
                    "failed_step": None,
                },
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                default_exit_code = eval_cli_main(["cpu-pilot-report", str(summary_path), "--json"])
            with patch("sys.stdout", new_callable=io.StringIO):
                required_exit_code = eval_cli_main(["cpu-pilot-report", str(summary_path), "--json", "--require-ready"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(default_exit_code, 0)
        self.assertEqual(required_exit_code, 2)
        self.assertIsNone(payload["pilot_artifact_report"])

    def test_eval_cli_cpu_pilot_report_failed_summary_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, cpu_pilot_summary(status="failed", failed_step_index=2))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["cpu-pilot-report", str(summary_path)])

        self.assertEqual(exit_code, 2)
        self.assertIn("status: FAIL", stdout.getvalue())

    def test_eval_cli_cpu_pilot_report_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"
            write_json(summary_path, {"schema_version": "old", "status": "passed"})

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["cpu-pilot-report", str(summary_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unsupported cpu pilot summary schema", stderr.getvalue())

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


def cpu_smoke_summary(*, status: str, failed_step_index: int | None = None) -> dict:
    steps = [
        {
            "index": 1,
            "name": "bootstrap teacher checkpoint",
            "status": "passed",
            "returncode": 0,
            "duration_seconds": 1.25,
        },
        {
            "index": 2,
            "name": "run smoke self-play iteration loop",
            "status": "failed" if failed_step_index == 2 else "passed",
            "returncode": 7 if failed_step_index == 2 else 0,
            "duration_seconds": 2.5,
        },
    ]
    return {
        "schema_version": "pokezero.cpu_smoke_run_summary.v1",
        "status": status,
        "summary_path": "run/cpu-smoke-run-summary.json",
        "started_at": "2026-06-22T12:00:00.000Z",
        "ended_at": "2026-06-22T12:01:00.000Z",
        "duration_seconds": 60.0,
        "source": {
            "available": True,
            "repo_root": "/repo",
            "branch": "main",
            "head": "abc123",
            "dirty": False,
        },
        "recipe": {"run_root": "run", "steps": []},
        "steps": steps,
        "failed_step": (
            None
            if failed_step_index is None
            else {
                "index": failed_step_index,
                "name": steps[failed_step_index - 1]["name"],
                "returncode": steps[failed_step_index - 1]["returncode"],
            }
        ),
    }


def cpu_pilot_summary(*, status: str, failed_step_index: int | None = None) -> dict:
    steps = [
        {
            "index": 1,
            "name": "run CPU smoke pilot 1",
            "status": "passed",
            "returncode": 0,
            "duration_seconds": 10.0,
        },
        {
            "index": 2,
            "name": "run CPU smoke pilot 2",
            "status": "failed" if failed_step_index == 2 else "passed",
            "returncode": 7 if failed_step_index == 2 else 0,
            "duration_seconds": 11.0,
        },
    ]
    return {
        "schema_version": "pokezero.cpu_pilot_suite_summary.v1",
        "status": status,
        "summary_path": "run/cpu-pilot-suite-summary.json",
        "started_at": "2026-06-22T12:00:00.000Z",
        "ended_at": "2026-06-22T12:01:00.000Z",
        "duration_seconds": 60.0,
        "source": {
            "available": True,
            "repo_root": "/repo",
            "branch": "main",
            "head": "abc123",
            "dirty": False,
        },
        "recipe": {
            "run_root": "run",
            "pilot_count": 2,
            "manifest_glob": "run/pilot-*/selfplay/manifest.json",
            "audit_config_path": "run/pilot-audit-config.json",
            "calibration_output_path": "run/pilot-calibration-compare.json",
            "replay_output_path": "run/pilot-audit-replay.json",
            "steps": [],
        },
        "steps": steps,
        "failed_step": (
            None
            if failed_step_index is None
            else {
                "index": failed_step_index,
                "name": steps[failed_step_index - 1]["name"],
                "returncode": steps[failed_step_index - 1]["returncode"],
            }
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
