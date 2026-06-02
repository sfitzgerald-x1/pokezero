import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.bootstrap import TEACHER_BOOTSTRAP_SCHEMA_VERSION
from pokezero.eval_cli import main as eval_cli_main
from pokezero.evaluation import PromotionGateConfig, evaluate_promotion_gate
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

    def test_bootstrap_manifest_checks_teacher_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, bootstrap_manifest())

            result = evaluate_promotion_gate(
                manifest_path,
                config=PromotionGateConfig(
                    min_benchmark_win_rate=0.50,
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
                exit_code = eval_cli_main(["gate", str(manifest_path), "--json"])

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
                        "--max-collection-capped-rate",
                        "0.20",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("status: PASS", stdout.getvalue())
        self.assertIn("benchmark_win_rate: 0.650", stdout.getvalue())


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


def benchmark_payload(*, policy_id: str, wins: int, losses: int, capped_games: int) -> dict:
    return {
        "format_id": "gen3randombattle",
        "max_decision_rounds": 250,
        "games_per_matchup": wins + losses,
        "head_to_heads": [
            {
                "label": f"{policy_id} vs random-legal",
                "first_policy_id": policy_id,
                "second_policy_id": "random-legal",
                "games": wins + losses,
                "first_policy_wins": wins,
                "second_policy_wins": losses,
                "ties": 0,
                "capped_games": capped_games,
                "first_policy_win_rate": wins / (wins + losses),
                "second_policy_win_rate": losses / (wins + losses),
            }
        ],
        "matchups": [],
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
