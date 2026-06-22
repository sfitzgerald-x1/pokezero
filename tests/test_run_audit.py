import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.eval_cli import main as eval_cli_main
from pokezero.run_audit import RunAuditConfig, audit_run
from pokezero.evaluation import NEURAL_SELFPLAY_RUN_SCHEMA_VERSION
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION


class RunAuditTest(unittest.TestCase):
    def test_audit_passes_healthy_linear_run(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),
                selfplay_iteration(iteration=2, wins=14, losses=6, capped_games=1, promotion_recorded=True),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                    max_latest_collection_capped_rate=0.20,
                    max_latest_benchmark_capped_rate=0.10,
                    require_latest_promotion=True,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.source_type, "linear_selfplay")
        self.assertEqual(result.latest_iteration, 2)
        self.assertEqual(result.latest_benchmark_win_rate, 0.70)
        self.assertEqual(result.best_benchmark_win_rate, 0.70)
        self.assertEqual(result.consecutive_promotion_failures, 0)

    def test_audit_fails_latest_benchmark_regression_from_previous_best(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=18, losses=2, capped_games=0),
                selfplay_iteration(iteration=2, wins=13, losses=7, capped_games=0),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_benchmark_win_rate_drop=0.05,
                ),
            )

        self.assertFalse(result.passed)
        self.assertIn("benchmark_win_rate_drop_from_best", failed_check_names(result))

    def test_audit_fails_latest_capped_rate_and_trailing_promotion_failures(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0, promotion_recorded=True),
                selfplay_iteration(iteration=2, wins=13, losses=7, capped_games=4, promotion_recorded=False),
                selfplay_iteration(iteration=3, wins=13, losses=7, capped_games=4, promotion_recorded=False),
            )
        )
        manifest["iterations"][-1]["collection_metrics"]["capped_games"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_collection_capped_rate=0.10,
                    max_latest_benchmark_capped_rate=0.10,
                    max_consecutive_promotion_failures=1,
                ),
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.consecutive_promotion_failures, 2)
        failed = failed_check_names(result)
        self.assertIn("latest_collection_capped_rate", failed)
        self.assertIn("latest_benchmark_capped_rate", failed)
        self.assertIn("consecutive_promotion_failures", failed)

    def test_audit_supports_neural_selfplay_manifest_without_torch(self) -> None:
        manifest = neural_selfplay_manifest(
            iterations=(
                neural_iteration(iteration=1, wins=13, losses=7, capped_games=0),
                neural_iteration(iteration=2, wins=14, losses=6, capped_games=0),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(min_latest_benchmark_win_rate=0.60, min_latest_benchmark_games=20),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.source_type, "neural_selfplay")
        self.assertEqual(result.iterations[-1].policy_id, "entity-test-iter-0002")

    def test_eval_cli_audit_prints_json_and_returns_nonzero_on_failure(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=8, losses=12, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--json",
                        "--min-latest-benchmark-win-rate",
                        "0.55",
                        "--min-latest-benchmark-games",
                        "20",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertIn("latest_benchmark_win_rate", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_prints_text_summary(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--min-latest-benchmark-win-rate",
                        "0.60",
                        "--min-latest-benchmark-games",
                        "20",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("status: PASS", stdout.getvalue())
        self.assertIn("latest_benchmark_win_rate: 0.650", stdout.getvalue())


def selfplay_manifest(*, iterations: tuple[dict, ...]) -> dict:
    return {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": "run",
        "latest_checkpoint_path": iterations[-1]["checkpoint_path"],
        "iterations": list(iterations),
    }


def neural_selfplay_manifest(*, iterations: tuple[dict, ...]) -> dict:
    return {
        "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": "run",
        "latest_checkpoint_path": iterations[-1]["checkpoint_path"],
        "current_policy_spec": iterations[-1]["checkpoint_policy_spec"],
        "latest_accepted_checkpoint_path": iterations[-1]["checkpoint_path"],
        "iterations": list(iterations),
    }


def selfplay_iteration(
    *,
    iteration: int,
    wins: int,
    losses: int,
    capped_games: int,
    promotion_recorded: bool | None = None,
) -> dict:
    policy_id = f"linear-selfplay-test-iter-{iteration:04d}"
    payload = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": iteration,
        "checkpoint_path": f"run/iteration-{iteration:04d}/linear-policy.json",
        "collection_metrics": collection_metrics(games=10, capped_games=0),
        "training": {"model": {"policy_id": policy_id}},
        "benchmark": benchmark_payload(policy_id=policy_id, wins=wins, losses=losses, capped_games=capped_games),
    }
    if promotion_recorded is not None:
        payload["promotion"] = {"recorded": promotion_recorded}
    return payload


def neural_iteration(*, iteration: int, wins: int, losses: int, capped_games: int) -> dict:
    policy_id = f"entity-test-iter-{iteration:04d}"
    checkpoint_path = f"run/iteration-{iteration:04d}/transformer-policy.pt"
    return {
        "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": iteration,
        "checkpoint_path": checkpoint_path,
        "checkpoint_policy_spec": f"neural:{checkpoint_path}",
        "current_policy_spec": "random-legal" if iteration == 1 else f"neural:run/iteration-{iteration - 1:04d}/transformer-policy.pt",
        "next_current_policy_spec": f"neural:{checkpoint_path}",
        "collection_metrics": collection_metrics(games=10, capped_games=0),
        "training": {"model_config": {"policy_id": policy_id}},
        "benchmark": benchmark_payload(policy_id=policy_id, wins=wins, losses=losses, capped_games=capped_games),
        "advancement": {"advance_collector": True, "reason": "beat_incumbent"},
    }


def benchmark_payload(*, policy_id: str, wins: int, losses: int, capped_games: int) -> dict:
    games = wins + losses
    return {
        "format_id": "gen3randombattle",
        "max_decision_rounds": 250,
        "games_per_matchup": games,
        "head_to_heads": [
            {
                "label": f"{policy_id} vs random-legal",
                "first_policy_id": policy_id,
                "second_policy_id": "random-legal",
                "games": games,
                "first_policy_wins": wins,
                "second_policy_wins": losses,
                "ties": 0,
                "capped_games": capped_games,
                "first_policy_win_rate": wins / games,
                "second_policy_win_rate": losses / games,
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
