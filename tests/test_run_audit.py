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
        self.assertEqual(result.latest_average_decision_rounds, 10.0)
        self.assertEqual(result.consecutive_promotion_failures, 0)

    def test_audit_fails_latest_same_opponent_benchmark_regression_from_previous_best(self) -> None:
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
        self.assertIn("benchmark_win_rate_drop_by_opponent", failed_check_names(result))
        self.assertEqual(result.benchmark_regressions[0].opponent_policy_id, "random-legal")
        self.assertEqual(result.benchmark_regressions[0].drop, 0.25)

    def test_audit_does_not_treat_harder_new_opponent_as_pooled_regression(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(("random-legal", 18, 2, 0),),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(
                        ("random-legal", 19, 1, 0),
                        ("linear-selfplay-test-iter-0001", 11, 9, 0),
                    ),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=40,
                    max_benchmark_win_rate_drop=0.05,
                ),
            )

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.latest_benchmark_win_rate or 0.0, 0.75)
        self.assertEqual(
            [(regression.opponent_policy_id, regression.drop) for regression in result.benchmark_regressions],
            [("random-legal", 0.0)],
        )

    def test_audit_fails_when_latest_benchmark_disappears_after_prior_evidence(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),
                selfplay_iteration(iteration=2, benchmark=False),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    require_benchmark=False,
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                ),
            )

        self.assertFalse(result.passed)
        regression_check = next(check for check in result.checks if check.name == "benchmark_win_rate_drop_by_opponent")
        self.assertEqual(regression_check.observed, "missing_latest_benchmark")

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

    def test_audit_fails_latest_average_decision_rounds_threshold(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["average_decision_rounds"] = 225.0
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_average_decision_rounds=200.0,
                ),
            )

        self.assertFalse(result.passed)
        self.assertIn("latest_average_decision_rounds", failed_check_names(result))
        average_check = next(check for check in result.checks if check.name == "latest_average_decision_rounds")
        self.assertEqual(average_check.observed, 225.0)
        self.assertEqual(average_check.threshold, 200.0)

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
        self.assertEqual(payload["latest_average_decision_rounds"], 10.0)
        self.assertIn("latest_benchmark_win_rate", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_average_decision_rounds_threshold_flag_fails(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["average_decision_rounds"] = 225.0
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--json",
                        "--min-latest-benchmark-games",
                        "20",
                        "--max-latest-average-decision-rounds",
                        "200",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIn("latest_average_decision_rounds", failed_check_names_from_payload(payload))
        average_check = next(check for check in payload["checks"] if check["name"] == "latest_average_decision_rounds")
        self.assertEqual(average_check["observed"], 225.0)
        self.assertEqual(average_check["threshold"], 200.0)
        self.assertIn("exceed", average_check["message"])

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
        self.assertIn("latest_average_decision_rounds: 10.000", stdout.getvalue())


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
    wins: int = 13,
    losses: int = 7,
    capped_games: int = 0,
    rows: tuple[tuple[str, int, int, int], ...] | None = None,
    benchmark: bool = True,
    promotion_recorded: bool | None = None,
) -> dict:
    policy_id = f"linear-selfplay-test-iter-{iteration:04d}"
    payload = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": iteration,
        "checkpoint_path": f"run/iteration-{iteration:04d}/linear-policy.json",
        "collection_metrics": collection_metrics(games=10, capped_games=0),
        "training": {"model": {"policy_id": policy_id}},
        "benchmark": (
            benchmark_payload(
                policy_id=policy_id,
                wins=wins,
                losses=losses,
                capped_games=capped_games,
                rows=rows,
            )
            if benchmark
            else None
        ),
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


def benchmark_payload(
    *,
    policy_id: str,
    wins: int,
    losses: int,
    capped_games: int,
    rows: tuple[tuple[str, int, int, int], ...] | None = None,
) -> dict:
    if rows is None:
        rows = (("random-legal", wins, losses, capped_games),)
    games_per_matchup = max(row_wins + row_losses for _, row_wins, row_losses, _ in rows)
    return {
        "format_id": "gen3randombattle",
        "max_decision_rounds": 250,
        "games_per_matchup": games_per_matchup,
        "head_to_heads": [
            benchmark_row(
                policy_id=policy_id,
                opponent_id=opponent_id,
                wins=row_wins,
                losses=row_losses,
                capped_games=row_capped_games,
            )
            for opponent_id, row_wins, row_losses, row_capped_games in rows
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
        "average_decision_rounds": 10.0,
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
