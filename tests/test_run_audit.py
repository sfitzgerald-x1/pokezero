import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.cli_audit import (
    MIN_SELFPLAY_POST_ITERATION_BENCHMARK_MATCHUPS,
    post_iteration_audit_config_from_args,
    validate_post_iteration_audit_evaluation_games,
)
from pokezero.eval_cli import main as eval_cli_main
from pokezero.run_audit import (
    RUN_AUDIT_CHECK_NAMES,
    RUN_AUDIT_CONFIG_SCHEMA_VERSION,
    RunAuditConfig,
    RunAuditFailure,
    audit_run,
    calibrate_run_audit,
    calibrate_run_audits,
    enforce_run_audit,
    compare_run_manifests,
    load_run_audit_config,
    run_audit_config_payload,
)
from pokezero.evaluation import NEURAL_SELFPLAY_RUN_SCHEMA_VERSION
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION
from pokezero.selfplay_cli import build_arg_parser as build_selfplay_arg_parser


def post_iteration_config_from_flags(flags: tuple[str, ...]) -> RunAuditConfig | None:
    parser = build_selfplay_arg_parser()
    args = parser.parse_args(
        [
            "iterate",
            "--run-dir",
            "run",
            "--iterations",
            "1",
            "--games-per-iteration",
            "1",
            *flags,
        ]
    )
    return post_iteration_audit_config_from_args(args)


def selfplay_iterate_args_from_flags(flags: tuple[str, ...]):
    parser = build_selfplay_arg_parser()
    return parser.parse_args(
        [
            "iterate",
            "--run-dir",
            "run",
            "--iterations",
            "1",
            "--games-per-iteration",
            "1",
            *flags,
        ]
    )


def assert_valid_selfplay_post_iteration_command_flags(
    test_case: unittest.TestCase,
    *,
    flags: tuple[str, ...],
    config_payload: dict,
    expected_evaluation_games: int,
) -> None:
    test_case.assertEqual(flags[:2], ("--evaluation-games", str(expected_evaluation_games)))
    args = selfplay_iterate_args_from_flags(flags)
    command_config = post_iteration_audit_config_from_args(args)
    validate_post_iteration_audit_evaluation_games(
        command_config,
        evaluation_games=args.evaluation_games,
        minimum_benchmark_matchups=MIN_SELFPLAY_POST_ITERATION_BENCHMARK_MATCHUPS,
    )
    test_case.assertEqual(command_config, RunAuditConfig(**config_payload))


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
        self.assertEqual(result.latest_benchmark_average_decision_rounds, 12.0)
        self.assertEqual(result.consecutive_promotion_failures, 0)

    def test_warnable_check_name_registry_covers_produced_audit_checks(self) -> None:
        benchmarked = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        unbenchmarked = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0, benchmark=False),)
        )
        config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.60,
            min_latest_benchmark_games=20,
            max_latest_average_decision_rounds=200.0,
            max_latest_benchmark_average_decision_rounds=200.0,
            max_latest_process_peak_rss_mb=1024.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmarked_path = Path(temp_dir) / "benchmarked" / "manifest.json"
            unbenchmarked_path = Path(temp_dir) / "unbenchmarked" / "manifest.json"
            write_manifest(benchmarked_path, benchmarked)
            write_manifest(unbenchmarked_path, unbenchmarked)

            produced_names = {
                check.name
                for result in (
                    audit_run(benchmarked_path, config=config),
                    audit_run(unbenchmarked_path, config=config),
                )
                for check in result.checks
            }

        self.assertLessEqual(produced_names, set(RUN_AUDIT_CHECK_NAMES))

    def test_audit_validates_recorded_promoted_opponent_pool_requirement(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                        "linear:runs/promoted-2/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(
                invocation_config(required_pool_size=2, promoted_checkpoint_count=2),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertTrue(result.passed)
        self.assertTrue(check.passed)
        self.assertEqual(check.observed, 1.0)

    def test_audit_fails_recorded_undersized_promoted_opponent_pool_requirement(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(
                invocation_config(required_pool_size=2, promoted_checkpoint_count=1),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertEqual(check.observed, 0.5)
        self.assertIn("launch_selectable=1,required=2", check.message)
        self.assertIn("iteration_1:selected=1,required=2", check.message)

    def test_audit_applies_current_policy_exclusion_to_recorded_promoted_pool_requirement(self) -> None:
        current_policy = "linear:runs/promoted-2/linear-policy.json"
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    current_policy_spec=current_policy,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(
                invocation_config(
                    required_pool_size=2,
                    promoted_checkpoint_count=2,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertEqual(check.observed, 0.5)
        self.assertIn("launch_selectable=1,required=2", check.message)
        self.assertIn("iteration_1:selected=1,required=2", check.message)

    def test_audit_excludes_relative_current_policy_from_absolute_promoted_pool_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            previous_checkpoint = temp_path / "runs" / "promoted-1" / "linear-policy.json"
            current_checkpoint = temp_path / "runs" / "promoted-2" / "linear-policy.json"
            previous_spec = f"linear:{previous_checkpoint.resolve(strict=False)}"
            current_relative_spec = f"linear:{os.path.relpath(current_checkpoint, Path.cwd())}"
            config = invocation_config(required_pool_size=2, promoted_checkpoint_count=2)
            config["opponent_pool"]["promoted_checkpoint_policy_specs"] = [
                previous_spec,
                f"linear:{current_checkpoint.resolve(strict=False)}",
            ]
            manifest = selfplay_manifest(
                iterations=(
                    selfplay_iteration(
                        iteration=1,
                        wins=14,
                        losses=6,
                        capped_games=0,
                        current_policy_spec=current_relative_spec,
                        opponent_policy_specs=("random-legal", previous_spec),
                    ),
                ),
                invocation_configs=(config,),
            )
            manifest_path = temp_path / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertEqual(check.observed, 0.5)
        self.assertIn("launch_selectable=1,required=2", check.message)
        self.assertIn("iteration_1:selected=1,required=2", check.message)

    def test_audit_applies_historical_cap_to_recorded_promoted_pool_requirement(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-3/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(
                invocation_config(
                    required_pool_size=2,
                    promoted_checkpoint_count=3,
                    max_historical_opponents=1,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertEqual(check.observed, 0.5)
        self.assertIn("required=2,max_historical=1", check.message)
        self.assertIn("launch_selectable=1,required=2", check.message)
        self.assertIn("iteration_1:selected=1,required=2", check.message)

    def test_audit_fails_malformed_promoted_pool_requirement_without_crashing(self) -> None:
        config = invocation_config(required_pool_size=2, promoted_checkpoint_count=2)
        config["opponent_pool"]["required_promoted_opponent_pool_size"] = "two"
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                        "linear:runs/promoted-2/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(config,),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertIsNone(check.observed)
        self.assertIn("invalid_required_promoted_opponent_pool_size", check.message)

    def test_audit_fails_negative_promoted_pool_requirement(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=14, losses=6, capped_games=0),),
            invocation_configs=(
                invocation_config(required_pool_size=-1, promoted_checkpoint_count=2),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertIsNone(check.observed)
        self.assertIn("invalid_required_promoted_opponent_pool_size", check.message)

    def test_audit_fails_invalid_promoted_pool_cap_without_double_counting_cap(self) -> None:
        config = invocation_config(required_pool_size=2, promoted_checkpoint_count=2)
        config["opponent_pool"]["max_historical_opponents"] = "many"
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                        "linear:runs/promoted-2/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(config,),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertEqual(check.observed, 1.0)
        self.assertIn("missing_or_invalid_max_historical_opponents", check.message)
        self.assertNotIn("max_historical=0", check.message)
        self.assertNotIn("launch_selectable=0", check.message)

    def test_audit_attributes_promoted_pool_requirement_to_covered_invocation_iterations(self) -> None:
        first_config = invocation_config(required_pool_size=1, promoted_checkpoint_count=1)
        first_config["first_iteration"] = 1
        first_config["iterations_requested"] = 1
        second_config = invocation_config(required_pool_size=2, promoted_checkpoint_count=1)
        second_config["first_iteration"] = 2
        second_config["iterations_requested"] = 1
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    wins=14,
                    losses=6,
                    capped_games=0,
                    opponent_policy_specs=(
                        "random-legal",
                        "linear:runs/promoted-1/linear-policy.json",
                    ),
                ),
            ),
            invocation_configs=(first_config, second_config),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.60,
                    min_latest_benchmark_games=20,
                ),
            )

        check = next(check for check in result.checks if check.name == "promoted_opponent_pool_requirement")
        self.assertFalse(result.passed)
        self.assertFalse(check.passed)
        self.assertEqual(check.observed, 0.5)
        self.assertIn("invocation_2:launch_selectable=1,required=2", check.message)
        self.assertIn("invocation_2:iteration_2:selected=1,required=2", check.message)
        self.assertNotIn("invocation_1:iteration_2", check.message)

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

    def test_audit_fails_when_latest_benchmark_drops_prior_opponent(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(("random-legal", 19, 1, 0),),
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
                    min_latest_benchmark_games=20,
                    max_benchmark_win_rate_drop=0.05,
                ),
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.missing_latest_benchmark_opponents, ("simple-legal",))
        self.assertIn("latest_benchmark_opponent_coverage", failed_check_names(result))
        coverage_check = next(check for check in result.checks if check.name == "latest_benchmark_opponent_coverage")
        self.assertEqual(coverage_check.observed, "simple-legal")

    def test_audit_allows_missing_prior_benchmark_opponent_when_disabled(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(("random-legal", 19, 1, 0),),
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
                    min_latest_benchmark_games=20,
                    max_benchmark_win_rate_drop=0.05,
                    require_benchmark_opponent_coverage=False,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.missing_latest_benchmark_opponents, ("simple-legal",))
        coverage_check = next(check for check in result.checks if check.name == "latest_benchmark_opponent_coverage")
        self.assertTrue(coverage_check.passed)
        self.assertEqual(coverage_check.observed, "optional")

    def test_audit_allows_rotating_incumbent_benchmark_opponents(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                        ("linear-selfplay-test-iter-0001", 12, 8, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=3,
                    rows=(
                        ("random-legal", 19, 1, 0),
                        ("simple-legal", 15, 5, 0),
                        ("linear-selfplay-test-iter-0002", 12, 8, 0),
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
                    min_latest_benchmark_games=60,
                    max_benchmark_win_rate_drop=0.05,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.missing_latest_benchmark_opponents, ())
        coverage_check = next(check for check in result.checks if check.name == "latest_benchmark_opponent_coverage")
        self.assertTrue(coverage_check.passed)

    def test_audit_allows_slow_rotating_incumbent_benchmark_opponents(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                        ("linear-incumbent-a", 12, 8, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                        ("linear-incumbent-a", 13, 7, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=3,
                    rows=(
                        ("random-legal", 19, 1, 0),
                        ("simple-legal", 15, 5, 0),
                        ("linear-incumbent-b", 12, 8, 0),
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
                    min_latest_benchmark_games=60,
                    max_benchmark_win_rate_drop=0.05,
                ),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.missing_latest_benchmark_opponents, ())
        coverage_check = next(check for check in result.checks if check.name == "latest_benchmark_opponent_coverage")
        self.assertTrue(coverage_check.passed)

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

    def test_audit_warning_only_failed_check_does_not_fail_run(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["average_decision_rounds"] = 225.0
        config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.50,
            min_latest_benchmark_games=20,
            max_latest_average_decision_rounds=200.0,
            warning_check_names=("latest_average_decision_rounds",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(manifest_path, config=config)
            enforced = enforce_run_audit(manifest_path, config=config)

        average_check = next(check for check in result.checks if check.name == "latest_average_decision_rounds")
        self.assertTrue(result.passed)
        self.assertTrue(enforced.passed)
        self.assertFalse(average_check.passed)
        self.assertEqual(average_check.severity, "warning")
        self.assertEqual(result.blocking_failed_checks, ())
        self.assertEqual(tuple(check.name for check in result.warning_failed_checks), ("latest_average_decision_rounds",))
        self.assertEqual(result.to_dict()["failed_checks"], [])
        self.assertEqual(result.to_dict()["warning_checks"], ["latest_average_decision_rounds"])

    def test_audit_still_fails_when_any_blocking_check_fails_alongside_warning(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=9, losses=11, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["average_decision_rounds"] = 225.0
        config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.60,
            min_latest_benchmark_games=20,
            max_latest_average_decision_rounds=200.0,
            warning_check_names=("latest_average_decision_rounds",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(manifest_path, config=config)
            with self.assertRaises(RunAuditFailure) as raised:
                enforce_run_audit(manifest_path, config=config)

        self.assertFalse(result.passed)
        self.assertEqual(tuple(check.name for check in result.blocking_failed_checks), ("latest_benchmark_win_rate",))
        self.assertEqual(tuple(check.name for check in result.warning_failed_checks), ("latest_average_decision_rounds",))
        self.assertEqual(tuple(check.name for check in raised.exception.result.blocking_failed_checks), ("latest_benchmark_win_rate",))

    def test_audit_fails_latest_benchmark_average_decision_rounds_threshold(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["benchmark"]["average_decision_rounds"] = 225.0
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_benchmark_average_decision_rounds=200.0,
                ),
            )

        self.assertFalse(result.passed)
        self.assertIn("latest_benchmark_average_decision_rounds", failed_check_names(result))
        average_check = next(
            check for check in result.checks if check.name == "latest_benchmark_average_decision_rounds"
        )
        self.assertEqual(average_check.observed, 225.0)
        self.assertEqual(average_check.threshold, 200.0)
        self.assertIn("exceed", average_check.message)

    def test_audit_passes_latest_benchmark_average_decision_rounds_threshold(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_benchmark_average_decision_rounds=200.0,
                ),
            )

        self.assertTrue(result.passed)
        average_check = next(
            check for check in result.checks if check.name == "latest_benchmark_average_decision_rounds"
        )
        self.assertTrue(average_check.passed)
        self.assertEqual(average_check.observed, 12.0)
        self.assertEqual(average_check.threshold, 200.0)
        self.assertIn("within limit", average_check.message)

    def test_audit_fails_latest_process_peak_rss_threshold(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 512.25
        manifest["iterations"][0]["benchmark"]["peak_rss_mb"] = 640.5
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_process_peak_rss_mb=600.0,
                ),
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.latest_process_peak_rss_mb, 640.5)
        self.assertIn("latest_process_peak_rss_mb", failed_check_names(result))
        rss_check = next(check for check in result.checks if check.name == "latest_process_peak_rss_mb")
        self.assertEqual(rss_check.observed, 640.5)
        self.assertEqual(rss_check.threshold, 600.0)
        self.assertIn("exceed", rss_check.message)

    def test_audit_passes_latest_process_peak_rss_threshold(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 512.25
        manifest["iterations"][0]["benchmark"]["peak_rss_mb"] = 640.5
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_process_peak_rss_mb=700.0,
                ),
            )

        self.assertTrue(result.passed)
        rss_check = next(check for check in result.checks if check.name == "latest_process_peak_rss_mb")
        self.assertEqual(rss_check.observed, 640.5)
        self.assertEqual(rss_check.threshold, 700.0)
        self.assertIn("within limit", rss_check.message)

    def test_audit_skips_process_peak_rss_threshold_when_metric_is_unavailable(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.50,
                    min_latest_benchmark_games=20,
                    max_latest_process_peak_rss_mb=700.0,
                ),
            )

        self.assertTrue(result.passed)
        self.assertIsNone(result.latest_process_peak_rss_mb)
        rss_check = next(check for check in result.checks if check.name == "latest_process_peak_rss_mb")
        self.assertTrue(rss_check.passed)
        self.assertIsNone(rss_check.observed)
        self.assertIn("skipped", rss_check.message)

    def test_audit_allows_missing_optional_benchmark_with_benchmark_average_threshold(self) -> None:
        manifest = selfplay_manifest(iterations=(selfplay_iteration(iteration=1, benchmark=False),))
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(
                    require_benchmark=False,
                    max_latest_benchmark_average_decision_rounds=200.0,
                ),
            )

        self.assertTrue(result.passed)
        average_check = next(
            check for check in result.checks if check.name == "latest_benchmark_average_decision_rounds"
        )
        self.assertTrue(average_check.passed)
        self.assertIsNone(average_check.observed)
        self.assertIn("optional", average_check.message)

    def test_audit_derives_benchmark_average_decision_rounds_from_matchups(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        benchmark = manifest["iterations"][0]["benchmark"]
        benchmark.pop("average_decision_rounds")
        benchmark["matchups"] = [
            {"metrics": {"games": 2, "average_decision_rounds": 40.0}},
            {"metrics": {"games": 6, "average_decision_rounds": 20.0}},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(min_latest_benchmark_win_rate=0.50, min_latest_benchmark_games=20),
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.latest_benchmark_average_decision_rounds, 25.0)

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

    def test_audit_result_exposes_manifest_source_metadata(self) -> None:
        source = source_metadata()
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),),
            source=source,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = audit_run(
                manifest_path,
                config=RunAuditConfig(min_latest_benchmark_win_rate=0.50, min_latest_benchmark_games=20),
            )

        self.assertEqual(result.source_metadata, source)
        self.assertEqual(result.to_dict()["source_metadata"], source)

    def test_calibrate_run_audit_suggests_thresholds_from_observed_history(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, rows=(("random-legal", 18, 2, 0),), promotion_recorded=True),
                selfplay_iteration(
                    iteration=2,
                    rows=(("random-legal", 16, 4, 2),),
                    promotion_recorded=False,
                ),
                selfplay_iteration(
                    iteration=3,
                    rows=(("random-legal", 17, 3, 1),),
                    promotion_recorded=False,
                ),
            )
        )
        manifest["iterations"][1]["collection_metrics"]["capped_games"] = 2
        manifest["iterations"][1]["collection_metrics"]["average_decision_rounds"] = 30.0
        manifest["iterations"][1]["collection_metrics"]["peak_rss_mb"] = 512.0
        manifest["iterations"][2]["collection_metrics"]["average_decision_rounds"] = 20.0
        manifest["iterations"][2]["benchmark"]["average_decision_rounds"] = 40.0
        manifest["iterations"][2]["benchmark"]["peak_rss_mb"] = 640.0
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = calibrate_run_audit(manifest_path, margin=0.10)

        self.assertEqual(result.iteration_count, 3)
        self.assertEqual(result.benchmark_iteration_count, 3)
        self.assertTrue(result.require_benchmark)
        self.assertEqual(result.min_latest_benchmark_win_rate, 0.72)
        self.assertEqual(result.min_latest_benchmark_games, 20)
        self.assertEqual(result.max_latest_collection_capped_rate, 0.22)
        self.assertEqual(result.max_latest_benchmark_capped_rate, 0.11)
        self.assertEqual(result.max_latest_average_decision_rounds, 33.0)
        self.assertEqual(result.max_latest_benchmark_average_decision_rounds, 44.0)
        self.assertEqual(result.max_latest_process_peak_rss_mb, 704.0)
        self.assertEqual(result.max_benchmark_win_rate_drop, 0.11)
        self.assertEqual(result.max_consecutive_promotion_failures, 2)
        self.assertTrue(result.require_benchmark_opponent_coverage)
        self.assertIn("--max-latest-process-peak-rss-mb", result.suggested_cli_flags())
        self.assertIn("--max-latest-average-decision-rounds", result.suggested_cli_flags())
        self.assertIn("--audit-after-iteration", result.suggested_post_iteration_cli_flags())
        self.assertIn("--audit-max-latest-process-peak-rss-mb", result.suggested_post_iteration_cli_flags())
        self.assertEqual(
            post_iteration_config_from_flags(result.suggested_post_iteration_cli_flags()),
            RunAuditConfig(**result.suggested_config()),
        )

    def test_calibrated_thresholds_keep_clean_pilot_headroom_for_comparable_run(self) -> None:
        pilot = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, rows=(("random-legal", 14, 6, 0),), promotion_recorded=True),
                selfplay_iteration(iteration=2, rows=(("random-legal", 15, 5, 0),), promotion_recorded=True),
            )
        )
        comparable = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, rows=(("random-legal", 14, 6, 0),), promotion_recorded=True),
                selfplay_iteration(iteration=2, rows=(("random-legal", 15, 5, 0),), promotion_recorded=True),
                selfplay_iteration(iteration=3, rows=(("random-legal", 14, 6, 1),), promotion_recorded=False),
            )
        )
        comparable["iterations"][2]["collection_metrics"]["capped_games"] = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            pilot_path = Path(temp_dir) / "pilot.json"
            comparable_path = Path(temp_dir) / "comparable.json"
            write_manifest(pilot_path, pilot)
            write_manifest(comparable_path, comparable)

            calibration = calibrate_run_audit(pilot_path, margin=0.10)
            audit = audit_run(comparable_path, config=RunAuditConfig(**calibration.suggested_config()))

        self.assertEqual(calibration.max_latest_collection_capped_rate, 0.10)
        self.assertEqual(calibration.max_latest_benchmark_capped_rate, 0.10)
        self.assertEqual(calibration.max_benchmark_win_rate_drop, 0.05)
        self.assertEqual(calibration.max_consecutive_promotion_failures, 1)
        self.assertTrue(audit.passed)

    def test_post_iteration_calibration_flags_clear_nullable_threshold_defaults(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = calibrate_run_audit(manifest_path)

        expected_config = RunAuditConfig(**result.suggested_config())
        flags = result.suggested_post_iteration_cli_flags()

        self.assertIsNone(expected_config.max_latest_process_peak_rss_mb)
        self.assertIn("--audit-max-latest-process-peak-rss-mb", flags)
        self.assertIn("none", flags)
        with patch(
            "pokezero.cli_audit.DEFAULT_POST_ITERATION_AUDIT_CONFIG",
            RunAuditConfig(max_latest_process_peak_rss_mb=4096.0),
        ):
            parsed_config = post_iteration_config_from_flags(flags)

        self.assertEqual(parsed_config, expected_config)

    def test_calibrate_run_audits_aggregates_thresholds_across_pilot_runs(self) -> None:
        first = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, rows=(("random-legal", 14, 6, 0),)),)
        )
        second = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, rows=(("random-legal", 12, 8, 2),)),)
        )
        second["iterations"][0]["collection_metrics"]["capped_games"] = 2
        second["iterations"][0]["collection_metrics"]["average_decision_rounds"] = 25.0
        first["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 256.0
        second["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 512.0
        second["iterations"][0]["benchmark"]["average_decision_rounds"] = 30.0
        second["iterations"][0]["benchmark"]["peak_rss_mb"] = 768.0
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.json"
            second_path = Path(temp_dir) / "second.json"
            write_manifest(first_path, first)
            write_manifest(second_path, second)

            calibration = calibrate_run_audits((first_path, second_path), margin=0.10)
            envelope = calibrate_run_audits((first_path, second_path), margin=0.10, aggregate_mode="envelope")
            first_audit = audit_run(first_path, config=RunAuditConfig(**envelope.suggested_config()))
            second_audit = audit_run(second_path, config=RunAuditConfig(**envelope.suggested_config()))

        self.assertEqual(calibration.run_count, 2)
        self.assertEqual(calibration.aggregate_mode, "median")
        self.assertEqual(calibration.iteration_count, 2)
        self.assertEqual(calibration.benchmark_iteration_count, 2)
        self.assertEqual(calibration.source_type, "linear_selfplay")
        self.assertEqual(calibration.min_latest_benchmark_win_rate, 0.585)
        self.assertEqual(calibration.max_latest_collection_capped_rate, 0.16)
        self.assertEqual(calibration.max_latest_benchmark_capped_rate, 0.105)
        self.assertEqual(calibration.max_latest_average_decision_rounds, 19.25)
        self.assertEqual(calibration.max_latest_benchmark_average_decision_rounds, 23.1)
        self.assertEqual(calibration.max_latest_process_peak_rss_mb, 563.2)
        self.assertEqual(envelope.aggregate_mode, "envelope")
        self.assertEqual(envelope.min_latest_benchmark_win_rate, 0.54)
        self.assertEqual(envelope.max_latest_collection_capped_rate, 0.22)
        self.assertEqual(envelope.max_latest_process_peak_rss_mb, 844.8)
        self.assertTrue(first_audit.passed)
        self.assertTrue(second_audit.passed)
        self.assertIn("--min-latest-benchmark-win-rate", calibration.suggested_cli_flags())

    def test_calibrate_run_audits_allows_missing_benchmarks_if_any_pilot_lacks_them(self) -> None:
        benchmarked = selfplay_manifest(iterations=(selfplay_iteration(iteration=1),))
        unbenchmarked = selfplay_manifest(iterations=(selfplay_iteration(iteration=1, benchmark=False),))
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmarked_path = Path(temp_dir) / "benchmarked.json"
            unbenchmarked_path = Path(temp_dir) / "unbenchmarked.json"
            write_manifest(benchmarked_path, benchmarked)
            write_manifest(unbenchmarked_path, unbenchmarked)

            calibration = calibrate_run_audits((benchmarked_path, unbenchmarked_path))
            config = RunAuditConfig(**calibration.suggested_config())

        self.assertFalse(calibration.require_benchmark)
        self.assertFalse(calibration.require_benchmark_opponent_coverage)
        self.assertIn("--allow-missing-benchmark", calibration.suggested_cli_flags())
        self.assertIn("--allow-missing-benchmark-opponents", calibration.suggested_cli_flags())
        self.assertNotIn("--min-latest-benchmark-win-rate", calibration.suggested_cli_flags())
        self.assertIn("--audit-allow-missing-benchmark", calibration.suggested_post_iteration_cli_flags())
        self.assertEqual(
            post_iteration_config_from_flags(calibration.suggested_post_iteration_cli_flags()),
            config,
        )
        self.assertEqual(config.min_latest_benchmark_win_rate, 0.0)
        self.assertEqual(config.max_latest_benchmark_capped_rate, 1.0)
        self.assertIn("allows missing benchmarks", calibration.notes[1])

    def test_calibrate_run_audits_marks_mixed_source_types(self) -> None:
        linear = selfplay_manifest(iterations=(selfplay_iteration(iteration=1),))
        neural = neural_selfplay_manifest(iterations=(neural_iteration(iteration=1, wins=13, losses=7, capped_games=0),))
        with tempfile.TemporaryDirectory() as temp_dir:
            linear_path = Path(temp_dir) / "linear.json"
            neural_path = Path(temp_dir) / "neural.json"
            write_manifest(linear_path, linear)
            write_manifest(neural_path, neural)

            calibration = calibrate_run_audits((linear_path, neural_path))

        self.assertEqual(calibration.source_type, "mixed")
        self.assertEqual(calibration.run_count, 2)

    def test_calibrate_run_audits_single_path_still_returns_aggregate_result(self) -> None:
        manifest = selfplay_manifest(iterations=(selfplay_iteration(iteration=1),))
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            calibration = calibrate_run_audits((manifest_path,))

        self.assertEqual(calibration.run_count, 1)
        self.assertEqual(calibration.paths, (manifest_path,))
        self.assertEqual(calibration.aggregate_mode, "median")

    def test_calibrate_run_audit_allows_missing_benchmark_when_history_has_none(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, benchmark=False),
                selfplay_iteration(iteration=2, benchmark=False),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = calibrate_run_audit(manifest_path)

        self.assertFalse(result.require_benchmark)
        self.assertFalse(result.require_benchmark_opponent_coverage)
        self.assertEqual(result.min_latest_benchmark_games, 0)
        self.assertIsNone(result.min_latest_benchmark_win_rate)
        self.assertIn("--allow-missing-benchmark", result.suggested_cli_flags())
        self.assertIn("--allow-missing-benchmark-opponents", result.suggested_cli_flags())
        self.assertNotIn("--min-latest-benchmark-games", result.suggested_cli_flags())
        self.assertIn("--audit-allow-missing-benchmark", result.suggested_post_iteration_cli_flags())
        self.assertEqual(
            post_iteration_config_from_flags(result.suggested_post_iteration_cli_flags()),
            RunAuditConfig(**result.suggested_config()),
        )
        self.assertIn("No benchmark iterations", result.notes[0])

    def test_calibrate_run_audit_rejects_negative_margin(self) -> None:
        manifest = selfplay_manifest(iterations=(selfplay_iteration(iteration=1),))
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "margin must be non-negative"):
                calibrate_run_audit(manifest_path, margin=-0.1)

    def test_calibrate_run_audit_accepts_zero_margin(self) -> None:
        manifest = selfplay_manifest(iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=2),))
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = calibrate_run_audit(manifest_path, margin=0.0)

        self.assertEqual(result.margin, 0.0)
        self.assertEqual(result.min_latest_benchmark_win_rate, 0.65)
        self.assertEqual(result.max_latest_benchmark_capped_rate, 0.10)

    def test_calibrate_run_audit_supports_neural_selfplay_manifest(self) -> None:
        manifest = neural_selfplay_manifest(
            iterations=(
                neural_iteration(iteration=1, wins=13, losses=7, capped_games=0),
                neural_iteration(iteration=2, wins=14, losses=6, capped_games=0),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = calibrate_run_audit(manifest_path)

        self.assertEqual(result.source_type, "neural_selfplay")
        self.assertEqual(result.benchmark_iteration_count, 2)
        self.assertEqual(result.min_latest_benchmark_win_rate, 0.585)

    def test_compare_run_manifests_summarizes_linear_and_neural_runs(self) -> None:
        linear_manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=30, losses=20, capped_games=0),
                selfplay_iteration(iteration=2, wins=40, losses=10, capped_games=1, promotion_recorded=True),
            )
        )
        linear_manifest["iterations"][1]["collection_metrics"]["games_per_second"] = 2.5
        linear_manifest["iterations"][1]["collection_metrics"]["peak_rss_mb"] = 512.25
        linear_manifest["iterations"][1]["benchmark"]["games_per_second"] = 1.25
        linear_manifest["iterations"][1]["benchmark"]["peak_rss_mb"] = 640.5
        neural_manifest = neural_selfplay_manifest(
            iterations=(
                neural_iteration(iteration=1, wins=30, losses=20, capped_games=0),
                neural_iteration(iteration=2, wins=35, losses=15, capped_games=0),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            linear_path = temp_path / "linear-run" / "manifest.json"
            neural_path = temp_path / "neural-run" / "manifest.json"
            write_manifest(linear_path, linear_manifest)
            write_manifest(neural_path, neural_manifest)

            result = compare_run_manifests((linear_path, neural_path))

        self.assertEqual([entry.label for entry in result.entries], ["linear-run", "neural-run"])
        self.assertEqual(result.entries[0].source_type, "linear_selfplay")
        self.assertEqual(result.entries[1].source_type, "neural_selfplay")
        self.assertEqual(result.entries[0].latest_policy_id, "linear-selfplay-test-iter-0002")
        self.assertEqual(result.entries[1].latest_policy_id, "entity-test-iter-0002")
        self.assertAlmostEqual(result.entries[0].latest_benchmark_win_rate or 0.0, 0.8)
        self.assertAlmostEqual(result.entries[1].latest_benchmark_win_rate or 0.0, 0.7)
        self.assertEqual(result.best_latest_benchmark_entry.label if result.best_latest_benchmark_entry else None, "linear-run")
        self.assertEqual(
            result.best_historical_benchmark_entry.label if result.best_historical_benchmark_entry else None,
            "linear-run",
        )
        self.assertEqual(result.entries[0].latest_collection_games_per_hour, 9000.0)
        self.assertEqual(result.entries[0].latest_benchmark_games_per_hour, 4500.0)
        self.assertEqual(result.entries[0].latest_collection_peak_rss_mb, 512.25)
        self.assertEqual(result.entries[0].latest_benchmark_peak_rss_mb, 640.5)
        self.assertEqual(result.entries[0].latest_process_peak_rss_mb, 640.5)
        self.assertTrue(result.entries[0].latest_promotion_recorded)
        self.assertTrue(result.entries[1].latest_advancement_recorded)

    def test_compare_run_manifests_derives_benchmark_throughput_from_elapsed_seconds(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        manifest["iterations"][0]["benchmark"]["total_games"] = 50
        manifest["iterations"][0]["benchmark"]["elapsed_seconds"] = 10.0
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            result = compare_run_manifests((manifest_path,))

        self.assertEqual(result.entries[0].latest_benchmark_games_per_hour, 18000.0)

    def test_compare_run_manifests_best_labels_require_minimum_benchmark_games(self) -> None:
        solid_manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(("random-legal", 800, 200, 0),),
                ),
            )
        )
        tiny_manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(("random-legal", 1, 0, 0),),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            solid_path = temp_path / "solid-run" / "manifest.json"
            tiny_path = temp_path / "tiny-run" / "manifest.json"
            write_manifest(solid_path, solid_manifest)
            write_manifest(tiny_path, tiny_manifest)

            result = compare_run_manifests((solid_path, tiny_path))

        self.assertEqual(result.entries[0].latest_benchmark_games, 1000)
        self.assertEqual(result.entries[1].latest_benchmark_games, 1)
        self.assertAlmostEqual(result.entries[1].latest_benchmark_win_rate or 0.0, 1.0)
        self.assertEqual(result.min_benchmark_games, 50)
        self.assertEqual(result.best_latest_benchmark_entry.label if result.best_latest_benchmark_entry else None, "solid-run")
        self.assertEqual(result.best_historical_benchmark_entry.label if result.best_historical_benchmark_entry else None, "solid-run")

    def test_compare_run_manifests_preserves_healthy_entries_when_one_manifest_is_bad(self) -> None:
        healthy_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        bad_manifest = {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": "bad-run",
            "latest_checkpoint_path": None,
            "iterations": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            healthy_path = temp_path / "healthy-run" / "manifest.json"
            bad_path = temp_path / "bad-run" / "manifest.json"
            write_manifest(healthy_path, healthy_manifest)
            write_manifest(bad_path, bad_manifest)

            result = compare_run_manifests((healthy_path, bad_path))

        self.assertEqual([entry.label for entry in result.entries], ["healthy-run"])
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].label, "bad-run")
        self.assertIn("no iterations", result.errors[0].error)

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
        self.assertEqual(payload["latest_benchmark_average_decision_rounds"], 12.0)
        self.assertIn("latest_benchmark_win_rate", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_reports_missing_benchmark_opponents(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(("random-legal", 19, 1, 0),),
                ),
            )
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
                        "0.50",
                        "--min-latest-benchmark-games",
                        "20",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["missing_latest_benchmark_opponents"], ["simple-legal"])
        self.assertIn("latest_benchmark_opponent_coverage", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_json_reports_manifest_source_metadata(self) -> None:
        source = source_metadata(branch="scott/report-source", head="def456", dirty=True)
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),),
            source=source,
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
                        "0.50",
                        "--min-latest-benchmark-games",
                        "20",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["source_metadata"], source)

    def test_eval_cli_audit_allows_missing_benchmark_opponents_flag(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 18, 2, 0),
                        ("simple-legal", 14, 6, 0),
                    ),
                ),
                selfplay_iteration(
                    iteration=2,
                    rows=(("random-legal", 19, 1, 0),),
                ),
            )
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
                        "0.50",
                        "--min-latest-benchmark-games",
                        "20",
                        "--allow-missing-benchmark-opponents",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["missing_latest_benchmark_opponents"], ["simple-legal"])
        self.assertNotIn("latest_benchmark_opponent_coverage", failed_check_names_from_payload(payload))

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

    def test_eval_cli_audit_benchmark_average_decision_rounds_threshold_flag_fails(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["benchmark"]["average_decision_rounds"] = 225.0
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
                        "--max-latest-benchmark-average-decision-rounds",
                        "200",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIn("latest_benchmark_average_decision_rounds", failed_check_names_from_payload(payload))
        average_check = next(
            check for check in payload["checks"] if check["name"] == "latest_benchmark_average_decision_rounds"
        )
        self.assertEqual(average_check["observed"], 225.0)
        self.assertEqual(average_check["threshold"], 200.0)
        self.assertIn("exceed", average_check["message"])

    def test_eval_cli_audit_process_peak_rss_threshold_flag_fails(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 512.25
        manifest["iterations"][0]["benchmark"]["peak_rss_mb"] = 640.5
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
                        "--max-latest-process-peak-rss-mb",
                        "600",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["latest_process_peak_rss_mb"], 640.5)
        self.assertIn("latest_process_peak_rss_mb", failed_check_names_from_payload(payload))
        rss_check = next(check for check in payload["checks"] if check["name"] == "latest_process_peak_rss_mb")
        self.assertEqual(rss_check["observed"], 640.5)
        self.assertEqual(rss_check["threshold"], 600.0)

    def test_eval_cli_audit_process_peak_rss_threshold_flag_skips_missing_metric(self) -> None:
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
                        "--json",
                        "--min-latest-benchmark-games",
                        "20",
                        "--max-latest-process-peak-rss-mb",
                        "600",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertIsNone(payload["latest_process_peak_rss_mb"])
        rss_check = next(check for check in payload["checks"] if check["name"] == "latest_process_peak_rss_mb")
        self.assertTrue(rss_check["passed"])
        self.assertIsNone(rss_check["observed"])
        self.assertIn("skipped", rss_check["message"])

    def test_eval_cli_audit_prints_text_summary(self) -> None:
        source = source_metadata(branch="scott/audit-text", head="abc999", dirty=True)
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),),
            source=source,
        )
        manifest["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 512.25
        manifest["iterations"][0]["benchmark"]["peak_rss_mb"] = 640.5
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
        self.assertIn("latest_benchmark_average_decision_rounds: 12.000", stdout.getvalue())
        self.assertIn("latest_process_peak_rss_mb: 640.5", stdout.getvalue())
        self.assertIn("source_metadata:", stdout.getvalue())
        self.assertIn("available: yes", stdout.getvalue())
        self.assertIn("branch: scott/audit-text", stdout.getvalue())
        self.assertIn("head: abc999", stdout.getvalue())
        self.assertIn("dirty: yes", stdout.getvalue())
        self.assertIn("repo_root: /repo", stdout.getvalue())

    def test_eval_cli_audit_text_handles_missing_and_unavailable_source_metadata(self) -> None:
        missing_source_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        unavailable_source_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),),
            source={
                "available": False,
                "repo_root": None,
                "branch": None,
                "head": None,
                "dirty": None,
                "error": "RuntimeError: git unavailable",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing" / "manifest.json"
            unavailable_path = Path(temp_dir) / "unavailable" / "manifest.json"
            write_manifest(missing_path, missing_source_manifest)
            write_manifest(unavailable_path, unavailable_source_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as missing_stdout:
                missing_exit = eval_cli_main(
                    [
                        "audit",
                        str(missing_path),
                        "--min-latest-benchmark-win-rate",
                        "0.60",
                        "--min-latest-benchmark-games",
                        "20",
                    ]
                )
            with patch("sys.stdout", new_callable=io.StringIO) as unavailable_stdout:
                unavailable_exit = eval_cli_main(
                    [
                        "audit",
                        str(unavailable_path),
                        "--min-latest-benchmark-win-rate",
                        "0.60",
                        "--min-latest-benchmark-games",
                        "20",
                    ]
                )

        self.assertEqual(missing_exit, 0)
        self.assertIn("source_metadata: -", missing_stdout.getvalue())
        self.assertEqual(unavailable_exit, 0)
        self.assertIn("available: no", unavailable_stdout.getvalue())
        self.assertIn("error: RuntimeError: git unavailable", unavailable_stdout.getvalue())

    def test_eval_cli_audit_calibrate_prints_json(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-calibrate", str(manifest_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["suggested_config"]["min_latest_benchmark_win_rate"], 0.585)
        self.assertEqual(payload["suggested_config"]["max_benchmark_win_rate_drop"], 0.05)
        self.assertIn("No same-opponent regression history", payload["notes"][0])
        self.assertIn("--min-latest-benchmark-win-rate", payload["suggested_cli_flags"])
        self.assertIn("--audit-after-iteration", payload["suggested_post_iteration_cli_flags"])
        self.assertIn("--audit-min-latest-benchmark-win-rate", payload["suggested_post_iteration_cli_flags"])
        self.assertEqual(payload["minimum_evaluation_games"], 5)
        assert_valid_selfplay_post_iteration_command_flags(
            self,
            flags=tuple(payload["suggested_post_iteration_command_flags"]),
            config_payload=payload["suggested_config"],
            expected_evaluation_games=5,
        )

    def test_eval_cli_audit_calibrate_can_discover_manifest_globs(self) -> None:
        first = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, rows=(("random-legal", 14, 6, 0),)),)
        )
        second = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, rows=(("random-legal", 12, 8, 2),)),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "pilot-b" / "manifest.json"
            second_path = temp_path / "pilot-a" / "manifest.json"
            write_manifest(first_path, first)
            write_manifest(second_path, second)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        "--manifest-glob",
                        str(temp_path / "pilot-*" / "manifest.json"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["paths"], [str(second_path), str(first_path)])
        self.assertEqual(payload["aggregate_mode"], "median")

    def test_eval_cli_audit_calibrate_single_manifest_glob_uses_single_run_calibration(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "pilot-a" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        "--manifest-glob",
                        str(temp_path / "pilot-*" / "manifest.json"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertNotIn("run_count", payload)
        self.assertEqual(payload["manifest_path"], str(manifest_path))

    def test_eval_cli_audit_calibrate_rejects_missing_paths_and_empty_globs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch("sys.stderr", new_callable=io.StringIO) as missing_paths_stderr:
                missing_paths_exit = eval_cli_main(["audit-calibrate", "--json"])
            with patch("sys.stderr", new_callable=io.StringIO) as empty_glob_stderr:
                empty_glob_exit = eval_cli_main(
                    [
                        "audit-calibrate",
                        "--manifest-glob",
                        str(temp_path / "missing-*" / "manifest.json"),
                        "--json",
                    ]
                )

        self.assertEqual(missing_paths_exit, 1)
        self.assertIn("provide at least one path or --manifest-glob", missing_paths_stderr.getvalue())
        self.assertEqual(empty_glob_exit, 1)
        self.assertIn("--manifest-glob matched no paths", empty_glob_stderr.getvalue())

    def test_eval_cli_audit_calibrate_ignores_empty_glob_when_other_paths_match(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "pilot-a" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    exit_code = eval_cli_main(
                        [
                            "audit-calibrate",
                            str(manifest_path),
                            "--manifest-glob",
                            str(temp_path / "missing-*" / "manifest.json"),
                            "--json",
                        ]
                    )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["manifest_path"], str(manifest_path))
        self.assertIn("warning: --manifest-glob matched no paths", stderr.getvalue())

    def test_eval_cli_audit_calibrate_dedupes_positional_duplicates_for_sufficiency(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        str(manifest_path),
                        "--require-run-count",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertNotIn("run_count", payload)
        self.assertEqual(payload["manifest_path"], str(manifest_path))
        self.assertFalse(payload["calibration_sufficient"])
        self.assertEqual(
            payload["calibration_sufficiency_errors"],
            ["calibration_run_count 1 is below required 2"],
        )

    def test_eval_cli_audit_calibrate_prints_text_flags(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-calibrate", str(manifest_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("suggested_audit_flags:", stdout.getvalue())
        self.assertIn("suggested_post_iteration_flags:", stdout.getvalue())
        self.assertIn("minimum_evaluation_games: 5", stdout.getvalue())
        self.assertIn("suggested_post_iteration_command_flags:", stdout.getvalue())
        self.assertIn("--evaluation-games 5 --audit-after-iteration", stdout.getvalue())
        self.assertIn("--audit-after-iteration", stdout.getvalue())
        self.assertIn("--max-latest-benchmark-average-decision-rounds 13.2", stdout.getvalue())
        self.assertIn("--audit-max-latest-benchmark-average-decision-rounds 13.2", stdout.getvalue())

    def test_eval_cli_audit_calibrate_can_write_reusable_audit_config(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--require-run-count",
                        "1",
                        "--require-benchmark-iterations",
                        "1",
                        "--require-min-benchmark-games",
                        "20",
                        "--write-config",
                        str(config_path),
                    ]
                )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            config = load_run_audit_config(config_path)

        self.assertEqual(exit_code, 0)
        self.assertIn(f"written_config: {config_path}", stdout.getvalue())
        self.assertEqual(payload["schema_version"], RUN_AUDIT_CONFIG_SCHEMA_VERSION)
        self.assertEqual(payload["config"]["min_latest_benchmark_win_rate"], 0.585)
        self.assertEqual(payload["calibration"]["benchmark_iteration_count"], 1)
        self.assertEqual(config.min_latest_benchmark_win_rate, 0.585)

    def test_eval_cli_audit_calibrate_requires_sufficiency_gate_before_writing_config(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["audit-calibrate", str(manifest_path), "--write-config", str(config_path)])
            config_exists = config_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertFalse(config_exists)
        self.assertIn("--write-config requires at least one calibration sufficiency requirement", stderr.getvalue())

    def test_eval_cli_audit_calibrate_does_not_write_config_when_profile_fails(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--compare-profile",
                        "long-run",
                        "--require-run-count",
                        "1",
                        "--require-benchmark-iterations",
                        "1",
                        "--require-min-benchmark-games",
                        "20",
                        "--write-config",
                        str(config_path),
                    ]
                )
            config_exists = config_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertFalse(config_exists)
        self.assertIn("--write-config requires the selected profile audit to pass", stderr.getvalue())

    def test_eval_cli_audit_config_file_can_be_enforced_and_overridden(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        strict_config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.90,
            min_latest_benchmark_games=20,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, run_audit_config_payload(strict_config))

            with patch("sys.stdout", new_callable=io.StringIO) as strict_stdout:
                strict_exit = eval_cli_main(["audit", str(manifest_path), "--audit-config", str(config_path), "--json"])
            strict_payload = json.loads(strict_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as override_stdout:
                override_exit = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--audit-config",
                        str(config_path),
                        "--min-latest-benchmark-win-rate",
                        "0.60",
                        "--json",
                    ]
                )
            override_payload = json.loads(override_stdout.getvalue())

        self.assertEqual(strict_exit, 2)
        self.assertIn("latest_benchmark_win_rate", failed_check_names_from_payload(strict_payload))
        self.assertEqual(override_exit, 0)
        self.assertTrue(override_payload["passed"])

    def test_eval_cli_audit_warning_check_flag_reports_warning_without_failure(self) -> None:
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
                        "--min-latest-benchmark-win-rate",
                        "0.50",
                        "--min-latest-benchmark-games",
                        "20",
                        "--max-latest-average-decision-rounds",
                        "200",
                        "--warning-check",
                        "latest_average_decision_rounds",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["failed_checks"], [])
        self.assertEqual(payload["warning_checks"], ["latest_average_decision_rounds"])
        check = next(check for check in payload["checks"] if check["name"] == "latest_average_decision_rounds")
        self.assertFalse(check["passed"])
        self.assertEqual(check["severity"], "warning")

    def test_run_audit_config_round_trips_warning_check_names(self) -> None:
        config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.60,
            min_latest_benchmark_games=20,
            warning_check_names=("latest_average_decision_rounds", "benchmark_win_rate_drop_by_opponent"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(config_path, run_audit_config_payload(config))

            loaded = load_run_audit_config(config_path)

        self.assertEqual(loaded.warning_check_names, config.warning_check_names)

    def test_run_audit_config_rejects_unknown_warning_check_names(self) -> None:
        with self.assertRaises(ValueError):
            RunAuditConfig(warning_check_names=("not_a_real_check",))

    def test_post_iteration_audit_warning_check_flag_round_trips(self) -> None:
        config = post_iteration_config_from_flags(
            (
                "--audit-after-iteration",
                "--audit-min-latest-benchmark-win-rate",
                "0.50",
                "--audit-min-latest-benchmark-games",
                "20",
                "--audit-warning-check",
                "latest_average_decision_rounds",
            )
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.warning_check_names, ("latest_average_decision_rounds",))

    def test_eval_cli_audit_config_report_json_includes_metadata_and_preflight(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        config_payload = run_audit_config_payload(
            RunAuditConfig(min_latest_benchmark_win_rate=0.60, min_latest_benchmark_games=20),
            source={"branch": "main", "head": "abc123", "dirty": False},
            calibration={
                "source_type": "linear_selfplay",
                "run_count": 2,
                "benchmark_iteration_count": 2,
                "aggregate_mode": "envelope",
                "paths": ["runs/pilot-a/manifest.json", "runs/pilot-b/manifest.json"],
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, config_payload)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-config-report",
                        str(config_path),
                        str(manifest_path),
                        "--require-source",
                        "--require-calibration",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["audit_config_path"], str(config_path))
        self.assertEqual(payload["source"]["head"], "abc123")
        self.assertEqual(payload["calibration"]["run_count"], 2)
        self.assertTrue(payload["preflight_requested"])
        self.assertTrue(payload["preflight_passed"])
        self.assertEqual(payload["preflight_runs"][0]["manifest_path"], str(manifest_path))
        self.assertEqual(payload["preflight_runs"][0]["failed_checks"], [])
        self.assertEqual(payload["config"]["min_latest_benchmark_win_rate"], 0.60)
        self.assertEqual(payload["minimum_evaluation_games"], 5)
        self.assertIn("--audit-after-iteration", payload["post_iteration_flags"])
        self.assertIn("--audit-min-latest-benchmark-win-rate", payload["post_iteration_flags"])
        self.assertEqual(
            post_iteration_config_from_flags(tuple(payload["post_iteration_flags"])),
            RunAuditConfig(**payload["config"]),
        )
        self.assertEqual(
            payload["post_iteration_command_flags"][:2],
            ["--evaluation-games", "5"],
        )
        selfplay_args = selfplay_iterate_args_from_flags(tuple(payload["post_iteration_command_flags"]))
        command_config = post_iteration_audit_config_from_args(selfplay_args)
        validate_post_iteration_audit_evaluation_games(
            command_config,
            evaluation_games=selfplay_args.evaluation_games,
            minimum_benchmark_matchups=MIN_SELFPLAY_POST_ITERATION_BENCHMARK_MATCHUPS,
        )
        self.assertEqual(command_config, RunAuditConfig(**payload["config"]))
        self.assertEqual(payload["checks"][-1]["name"], "preflight_audit_passed")

    def test_eval_cli_audit_config_report_preserves_warning_checks(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        manifest["iterations"][0]["collection_metrics"]["average_decision_rounds"] = 225.0
        config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.50,
            min_latest_benchmark_games=20,
            max_latest_average_decision_rounds=200.0,
            warning_check_names=("latest_average_decision_rounds",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, run_audit_config_payload(config))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-config-report", str(config_path), str(manifest_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["preflight_passed"])
        self.assertEqual(payload["preflight_runs"][0]["failed_checks"], [])
        self.assertEqual(payload["preflight_runs"][0]["warning_checks"], ["latest_average_decision_rounds"])
        self.assertIn("--audit-warning-check", payload["post_iteration_flags"])
        self.assertIn("latest_average_decision_rounds", payload["post_iteration_flags"])
        self.assertIn("--audit-warning-check", payload["post_iteration_command_flags"])
        command_config = post_iteration_config_from_flags(tuple(payload["post_iteration_command_flags"][2:]))
        self.assertEqual(command_config, config)

    def test_eval_cli_audit_config_report_requires_calibration_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(config_path, run_audit_config_payload(RunAuditConfig(), source={"head": "abc123"}))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    ["audit-config-report", str(config_path), "--require-calibration", "--json"]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertIn("calibration_metadata_present", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_config_report_requires_source_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(config_path, run_audit_config_payload(RunAuditConfig(), calibration={"run_count": 1}))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-config-report", str(config_path), "--require-source", "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertIn("source_metadata_present", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_config_report_preflight_failure_returns_nonzero(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        strict_config = RunAuditConfig(min_latest_benchmark_win_rate=0.90, min_latest_benchmark_games=20)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, run_audit_config_payload(strict_config))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    ["audit-config-report", str(config_path), str(manifest_path), "--json"]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["preflight_passed"])
        self.assertIn("preflight_audit_passed", failed_check_names_from_payload(payload))
        self.assertIn("latest_benchmark_win_rate", payload["preflight_runs"][0]["failed_checks"])

    def test_eval_cli_audit_config_report_reports_bad_manifest_without_dropping_config(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            missing_manifest_path = Path(temp_dir) / "missing-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(
                config_path,
                run_audit_config_payload(
                    RunAuditConfig(min_latest_benchmark_win_rate=0.60, min_latest_benchmark_games=20),
                    source={"head": "abc123"},
                    calibration={"run_count": 1},
                ),
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    ["audit-config-report", str(config_path), str(manifest_path), str(missing_manifest_path), "--json"]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["config"]["min_latest_benchmark_win_rate"], 0.60)
        self.assertTrue(payload["preflight_runs"][0]["passed"])
        self.assertFalse(payload["preflight_runs"][1]["passed"])
        self.assertEqual(payload["preflight_runs"][1]["failed_checks"], ["manifest_error"])
        self.assertIn("Experiment manifest does not exist", payload["preflight_runs"][1]["error"])

    def test_eval_cli_audit_config_report_can_discover_manifest_globs(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first_manifest_path = Path(temp_dir) / "pilot-1" / "manifest.json"
            second_manifest_path = Path(temp_dir) / "pilot-2" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(first_manifest_path, manifest)
            write_manifest(second_manifest_path, manifest)
            write_manifest(
                config_path,
                run_audit_config_payload(
                    RunAuditConfig(min_latest_benchmark_win_rate=0.60, min_latest_benchmark_games=20),
                    calibration={"run_count": 2},
                ),
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-config-report",
                        str(config_path),
                        "--manifest-glob",
                        str(Path(temp_dir) / "pilot-*" / "manifest.json"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["preflight_passed"])
        self.assertEqual(
            [run["manifest_path"] for run in payload["preflight_runs"]],
            [str(first_manifest_path), str(second_manifest_path)],
        )

    def test_eval_cli_audit_config_report_text_prints_config_and_preflight(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(
                config_path,
                run_audit_config_payload(
                    RunAuditConfig(min_latest_benchmark_win_rate=0.60, min_latest_benchmark_games=20),
                    calibration={
                        "source_type": "linear_selfplay",
                        "run_count": 1,
                        "paths": [str(manifest_path)],
                    },
                ),
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-config-report", str(config_path), str(manifest_path)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("audit_config_report:", output)
        self.assertIn("passed: PASS", output)
        self.assertIn("calibration_metadata: present", output)
        self.assertIn("calibration_paths:", output)
        self.assertIn("- min_latest_benchmark_win_rate: 0.6", output)
        self.assertIn("post_iteration_flags:", output)
        self.assertIn("--audit-after-iteration", output)
        self.assertIn("--audit-min-latest-benchmark-win-rate 0.6", output)
        self.assertIn("minimum_evaluation_games: 5", output)
        self.assertIn("post_iteration_command_flags:", output)
        self.assertIn("--evaluation-games 5", output)
        self.assertIn("preflight: PASS", output)

    def test_eval_cli_audit_rejects_profile_with_audit_config(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, run_audit_config_payload(RunAuditConfig()))

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--profile",
                        "smoke",
                        "--audit-config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--profile cannot be combined with --audit-config", stderr.getvalue())

    def test_run_audit_config_loader_rejects_non_boolean_flags(self) -> None:
        config_payload = run_audit_config_payload(RunAuditConfig())
        config_payload["config"]["require_benchmark"] = "false"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(config_path, config_payload)

            with self.assertRaisesRegex(ValueError, "require_benchmark must be a boolean"):
                load_run_audit_config(config_path)

    def test_run_audit_config_loader_rejects_schema_mismatch(self) -> None:
        config_payload = run_audit_config_payload(RunAuditConfig())
        config_payload["schema_version"] = "pokezero.run_audit_config.v0"
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(config_path, config_payload)

            with self.assertRaisesRegex(ValueError, "Unsupported run audit config schema"):
                load_run_audit_config(config_path)

    def test_eval_cli_audit_calibrate_does_not_write_config_when_sufficiency_fails(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--require-run-count",
                        "2",
                        "--write-config",
                        str(config_path),
                    ]
                )
            config_exists = config_path.exists()

        self.assertEqual(exit_code, 1)
        self.assertFalse(config_exists)
        self.assertIn("--write-config requires calibration sufficiency checks to pass", stderr.getvalue())

    def test_eval_cli_audit_calibrate_json_can_compare_named_profile(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--compare-profile",
                        "long-run",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["profile_audit"]["profile"], "long-run")
        self.assertFalse(payload["profile_audit"]["passed"])
        self.assertEqual(len(payload["profile_audit"]["runs"]), 1)
        self.assertFalse(payload["profile_audit"]["runs"][0]["passed"])
        self.assertIn("latest_benchmark_games", payload["profile_audit"]["runs"][0]["failed_checks"])

    def test_eval_cli_audit_calibrate_can_fail_on_named_profile(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--compare-profile",
                        "long-run",
                        "--fail-on-profile",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["profile_audit"]["passed"])
        self.assertIn("suggested_config", payload)

    def test_eval_cli_audit_calibrate_fail_on_profile_passes_when_profile_passes(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 65, 35, 0),
                        ("simple-legal", 65, 35, 0),
                    ),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--compare-profile",
                        "long-run",
                        "--fail-on-profile",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["profile_audit"]["passed"])
        self.assertEqual(payload["profile_audit"]["runs"][0]["failed_checks"], [])

    def test_eval_cli_audit_calibrate_profile_comparison_aggregates_multiple_runs(self) -> None:
        passing = selfplay_manifest(
            iterations=(
                selfplay_iteration(
                    iteration=1,
                    rows=(
                        ("random-legal", 65, 35, 0),
                        ("simple-legal", 65, 35, 0),
                    ),
                ),
            )
        )
        failing = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            passing_path = Path(temp_dir) / "passing.json"
            failing_path = Path(temp_dir) / "failing.json"
            write_manifest(passing_path, passing)
            write_manifest(failing_path, failing)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(passing_path),
                        str(failing_path),
                        "--compare-profile",
                        "long-run",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertFalse(payload["profile_audit"]["passed"])
        self.assertEqual(len(payload["profile_audit"]["runs"]), 2)
        self.assertTrue(payload["profile_audit"]["runs"][0]["passed"])
        self.assertFalse(payload["profile_audit"]["runs"][1]["passed"])
        self.assertIn("latest_benchmark_games", payload["profile_audit"]["runs"][1]["failed_checks"])

    def test_eval_cli_audit_calibrate_rejects_fail_on_profile_without_profile(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["audit-calibrate", str(manifest_path), "--fail-on-profile"])

        self.assertEqual(exit_code, 1)
        self.assertIn("--fail-on-profile requires --compare-profile", stderr.getvalue())

    def test_eval_cli_audit_calibrate_text_can_compare_named_profile(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    ["audit-calibrate", str(manifest_path), "--compare-profile", "long-run"]
                )
            output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("profile_audit:", output)
        self.assertIn("profile: long-run", output)
        self.assertIn("status: FAIL", output)
        self.assertIn("failed_checks: latest_benchmark_games", output)

    def test_eval_cli_audit_calibrate_aggregates_multiple_runs_json(self) -> None:
        first = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, rows=(("random-legal", 14, 6, 0),)),)
        )
        second = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, rows=(("random-legal", 12, 8, 2),)),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.json"
            second_path = Path(temp_dir) / "second.json"
            write_manifest(first_path, first)
            write_manifest(second_path, second)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-calibrate", str(first_path), str(second_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["aggregate_mode"], "median")
        self.assertEqual(len(payload["sources"]), 2)
        self.assertEqual(payload["suggested_config"]["min_latest_benchmark_win_rate"], 0.585)
        self.assertIn("Aggregated from multiple audit calibrations", payload["notes"][0])

    def test_eval_cli_audit_calibrate_aggregates_multiple_runs_text(self) -> None:
        first = selfplay_manifest(iterations=(selfplay_iteration(iteration=1),))
        second = selfplay_manifest(iterations=(selfplay_iteration(iteration=1, benchmark=False),))
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.json"
            second_path = Path(temp_dir) / "second.json"
            write_manifest(first_path, first)
            write_manifest(second_path, second)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit-calibrate", str(first_path), str(second_path)])
            output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("runs: 2", output)
        self.assertIn("aggregate_mode: median", output)
        self.assertIn("manifests:", output)
        self.assertIn("--allow-missing-benchmark", output)

    def test_eval_cli_audit_calibrate_can_require_minimum_evidence(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--require-run-count",
                        "2",
                        "--require-benchmark-iterations",
                        "2",
                        "--require-min-benchmark-games",
                        "50",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["calibration_sufficient"])
        self.assertEqual(
            payload["calibration_sufficiency_errors"],
            [
                "calibration_run_count 1 is below required 2",
                "calibration_benchmark_iterations 1 is below required 2",
                "calibration_min_benchmark_games 20 is below required 50",
            ],
        )
        self.assertIn("suggested_config", payload)

    def test_eval_cli_audit_calibrate_fails_when_required_benchmark_evidence_is_missing_per_run(self) -> None:
        benchmarked = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),
                selfplay_iteration(iteration=2, wins=14, losses=6, capped_games=0),
            )
        )
        unbenchmarked = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, benchmark=False),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmarked_path = Path(temp_dir) / "benchmarked.json"
            unbenchmarked_path = Path(temp_dir) / "unbenchmarked.json"
            write_manifest(benchmarked_path, benchmarked)
            write_manifest(unbenchmarked_path, unbenchmarked)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(benchmarked_path),
                        str(unbenchmarked_path),
                        "--require-run-count",
                        "2",
                        "--require-benchmark-iterations",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["calibration_sufficient"])
        self.assertFalse(payload["suggested_config"]["require_benchmark"])
        self.assertEqual(
            payload["calibration_sufficiency_errors"],
            ["calibration includes at least one run without benchmark iterations"],
        )

    def test_eval_cli_audit_calibrate_text_prints_sufficiency_status(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit-calibrate",
                        str(manifest_path),
                        "--require-run-count",
                        "2",
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("calibration_sufficiency: FAIL", output)
        self.assertIn("calibration_sufficiency_errors:", output)
        self.assertIn("calibration_run_count 1 is below required 2", output)

    def test_eval_cli_compare_prints_json(self) -> None:
        source = source_metadata(branch="scott/linear", head="abc123", dirty=False)
        linear_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),),
            source=source,
        )
        neural_manifest = neural_selfplay_manifest(
            iterations=(neural_iteration(iteration=1, wins=35, losses=15, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            linear_path = temp_path / "linear-run" / "manifest.json"
            neural_path = temp_path / "neural-run" / "manifest.json"
            write_manifest(linear_path, linear_manifest)
            write_manifest(neural_path, neural_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(linear_path), str(neural_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["min_benchmark_games"], 50)
        self.assertEqual(payload["best_latest_benchmark_label"], "linear-run")
        self.assertEqual([entry["label"] for entry in payload["entries"]], ["linear-run", "neural-run"])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["entries"][0]["latest_benchmark_win_rate"], 0.8)
        self.assertEqual(payload["entries"][0]["latest_collection_games_per_hour"], 36000.0)
        self.assertEqual(payload["entries"][0]["source_metadata"], source)
        self.assertEqual(payload["entries"][1]["source_metadata"], {})
        self.assertIn("latest_process_peak_rss_mb", payload["entries"][0])
        self.assertIsNone(payload["audit_profile"])
        self.assertFalse(payload["audit_failed"])
        self.assertIsNone(payload["entries"][0]["audit_passed"])
        self.assertEqual(payload["entries"][0]["audit_failed_checks"], [])

    def test_eval_cli_compare_can_discover_manifest_globs_and_dedupe_paths(self) -> None:
        first_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        second_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=35, losses=15, capped_games=1),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "pilot-a" / "manifest.json"
            second_path = temp_path / "pilot-b" / "manifest.json"
            write_manifest(first_path, first_manifest)
            write_manifest(second_path, second_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(first_path),
                        "--manifest-glob",
                        str(temp_path / "pilot-*" / "manifest.json"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual([entry["manifest_path"] for entry in payload["entries"]], [str(first_path), str(second_path)])
        self.assertEqual(payload["errors"], [])

    def test_eval_cli_compare_ignores_empty_glob_when_other_paths_match(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "pilot-a" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    exit_code = eval_cli_main(
                        [
                            "compare",
                            str(manifest_path),
                            "--manifest-glob",
                            str(temp_path / "missing-*" / "manifest.json"),
                            "--json",
                        ]
                    )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual([entry["manifest_path"] for entry in payload["entries"]], [str(manifest_path)])
        self.assertIn("warning: --manifest-glob matched no paths", stderr.getvalue())

    def test_eval_cli_compare_dedupes_run_dir_against_manifest_glob(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "pilot-a"
            manifest_path = run_dir / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(run_dir),
                        "--manifest-glob",
                        str(temp_path / "pilot-*" / "manifest.json"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual([entry["manifest_path"] for entry in payload["entries"]], [str(manifest_path)])

    def test_eval_cli_compare_rejects_missing_paths_and_empty_globs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch("sys.stderr", new_callable=io.StringIO) as missing_paths_stderr:
                missing_paths_exit = eval_cli_main(["compare", "--json"])
            with patch("sys.stderr", new_callable=io.StringIO) as empty_glob_stderr:
                empty_glob_exit = eval_cli_main(
                    [
                        "compare",
                        "--manifest-glob",
                        str(temp_path / "missing-*" / "manifest.json"),
                        "--json",
                    ]
                )

        self.assertEqual(missing_paths_exit, 1)
        self.assertIn("provide at least one path or --manifest-glob", missing_paths_stderr.getvalue())
        self.assertEqual(empty_glob_exit, 1)
        self.assertIn("--manifest-glob matched no paths", empty_glob_stderr.getvalue())

    def test_eval_cli_compare_json_can_suggest_audit_calibration(self) -> None:
        first_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        second_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=35, losses=15, capped_games=1),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "pilot-a" / "manifest.json"
            second_path = temp_path / "pilot-b" / "manifest.json"
            write_manifest(first_path, first_manifest)
            write_manifest(second_path, second_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(first_path),
                        str(second_path),
                        "--suggest-audit-calibration",
                        "--calibration-margin",
                        "0.20",
                        "--calibration-aggregate-mode",
                        "envelope",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audit_calibration"]["run_count"], 2)
        self.assertEqual(payload["audit_calibration"]["aggregate_mode"], "envelope")
        self.assertEqual(payload["audit_calibration"]["source_type"], "linear_selfplay")
        self.assertEqual(payload["audit_calibration"]["margin"], 0.20)
        self.assertIn("--min-latest-benchmark-games", payload["audit_calibration"]["suggested_cli_flags"])
        self.assertEqual(payload["audit_calibration"]["minimum_evaluation_games"], 13)
        assert_valid_selfplay_post_iteration_command_flags(
            self,
            flags=tuple(payload["audit_calibration"]["suggested_post_iteration_command_flags"]),
            config_payload=payload["audit_calibration"]["suggested_config"],
            expected_evaluation_games=13,
        )
        self.assertIsNone(payload["audit_calibration_error"])

    def test_eval_cli_compare_calibration_can_require_minimum_evidence(self) -> None:
        first_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        second_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=35, losses=15, capped_games=1),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "pilot-a" / "manifest.json"
            second_path = temp_path / "pilot-b" / "manifest.json"
            write_manifest(first_path, first_manifest)
            write_manifest(second_path, second_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(first_path),
                        str(second_path),
                        "--suggest-audit-calibration",
                        "--calibration-require-run-count",
                        "2",
                        "--calibration-require-benchmark-iterations",
                        "2",
                        "--calibration-require-min-benchmark-games",
                        "50",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["audit_calibration_sufficient"])
        self.assertEqual(payload["audit_calibration_sufficiency_errors"], [])

    def test_eval_cli_compare_can_write_suggested_audit_config(self) -> None:
        first_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        second_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=35, losses=15, capped_games=1),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "pilot-a" / "manifest.json"
            second_path = temp_path / "pilot-b" / "manifest.json"
            config_path = temp_path / "audit-configs" / "pilot.json"
            write_manifest(first_path, first_manifest)
            write_manifest(second_path, second_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(first_path),
                        str(second_path),
                        "--suggest-audit-calibration",
                        "--calibration-require-run-count",
                        "2",
                        "--calibration-require-benchmark-iterations",
                        "2",
                        "--calibration-require-min-benchmark-games",
                        "50",
                        "--write-audit-config",
                        str(config_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            written_payload = json.loads(config_path.read_text(encoding="utf-8"))
            loaded_config = load_run_audit_config(config_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["written_audit_config_path"], str(config_path))
        self.assertEqual(written_payload["schema_version"], RUN_AUDIT_CONFIG_SCHEMA_VERSION)
        self.assertEqual(written_payload["calibration"]["run_count"], 2)
        self.assertEqual(written_payload["calibration"]["benchmark_iteration_count"], 2)
        self.assertEqual(written_payload["calibration"]["paths"], [str(first_path), str(second_path)])
        self.assertEqual(loaded_config.min_latest_benchmark_games, 50)

    def test_eval_cli_compare_calibration_requirement_failure_returns_nonzero_with_suggestions(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "pilot-a" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--suggest-audit-calibration",
                        "--calibration-require-run-count",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIsNotNone(payload["audit_calibration"])
        self.assertFalse(payload["audit_calibration_sufficient"])
        self.assertEqual(
            payload["audit_calibration_sufficiency_errors"],
            ["calibration_run_count 1 is below required 2"],
        )

    def test_eval_cli_compare_rejects_write_audit_config_without_suggestions(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "linear-run" / "manifest.json"
            config_path = temp_path / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["compare", str(manifest_path), "--write-audit-config", str(config_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("--write-audit-config requires --suggest-audit-calibration", stderr.getvalue())
        self.assertFalse(config_path.exists())

    def test_eval_cli_compare_rejects_write_audit_config_without_sufficiency_requirements(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "linear-run" / "manifest.json"
            config_path = temp_path / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--suggest-audit-calibration",
                        "--write-audit-config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--write-audit-config requires at least one calibration sufficiency requirement", stderr.getvalue())
        self.assertFalse(config_path.exists())

    def test_eval_cli_compare_rejects_write_audit_config_when_sufficiency_fails(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "linear-run" / "manifest.json"
            config_path = temp_path / "audit-config.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--suggest-audit-calibration",
                        "--calibration-require-run-count",
                        "2",
                        "--write-audit-config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--write-audit-config requires calibration sufficiency checks to pass", stderr.getvalue())
        self.assertFalse(config_path.exists())

    def test_eval_cli_compare_rejects_write_audit_config_when_any_compared_manifest_fails(self) -> None:
        healthy_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            healthy_path = temp_path / "healthy" / "manifest.json"
            bad_path = temp_path / "bad" / "manifest.json"
            config_path = temp_path / "audit-config.json"
            write_manifest(healthy_path, healthy_manifest)
            write_manifest(bad_path, {"schema_version": "unknown"})

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(healthy_path),
                        str(bad_path),
                        "--suggest-audit-calibration",
                        "--calibration-require-run-count",
                        "1",
                        "--write-audit-config",
                        str(config_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIn(
            "--write-audit-config requires every compared manifest to load successfully",
            payload["audit_config_write_error"],
        )
        self.assertFalse(config_path.exists())

    def test_eval_cli_compare_skips_write_audit_config_when_selected_audit_fails(self) -> None:
        first_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        second_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=35, losses=15, capped_games=1),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = temp_path / "pilot-a" / "manifest.json"
            second_path = temp_path / "pilot-b" / "manifest.json"
            strict_config_path = temp_path / "strict-audit-config.json"
            output_config_path = temp_path / "audit-config.json"
            write_manifest(first_path, first_manifest)
            write_manifest(second_path, second_manifest)
            write_manifest(
                strict_config_path,
                run_audit_config_payload(
                    RunAuditConfig(
                        min_latest_benchmark_win_rate=0.99,
                        min_latest_benchmark_games=20,
                        require_benchmark=True,
                    )
                ),
            )

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(first_path),
                        str(second_path),
                        "--audit-config",
                        str(strict_config_path),
                        "--fail-on-audit",
                        "--suggest-audit-calibration",
                        "--calibration-require-run-count",
                        "2",
                        "--write-audit-config",
                        str(output_config_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertTrue(payload["audit_failed"])
        self.assertEqual(
            payload["audit_config_write_error"],
            "--write-audit-config requires the selected audit to pass.",
        )
        self.assertFalse(output_config_path.exists())

    def test_eval_cli_compare_rejects_calibration_requirements_without_suggestions(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["compare", str(manifest_path), "--calibration-require-run-count", "2"])

        self.assertEqual(exit_code, 1)
        self.assertIn("calibration sufficiency requirements require --suggest-audit-calibration", stderr.getvalue())

    def test_eval_cli_compare_calibration_excludes_bad_manifest_errors(self) -> None:
        healthy_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        bad_manifest = {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": "bad-run",
            "latest_checkpoint_path": None,
            "iterations": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            healthy_path = temp_path / "healthy-run" / "manifest.json"
            bad_path = temp_path / "bad-run" / "manifest.json"
            write_manifest(healthy_path, healthy_manifest)
            write_manifest(bad_path, bad_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(healthy_path),
                        str(bad_path),
                        "--suggest-audit-calibration",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual([entry["label"] for entry in payload["entries"]], ["healthy-run"])
        self.assertEqual(payload["audit_calibration"]["manifest_path"], str(healthy_path))
        self.assertEqual(payload["errors"][0]["label"], "bad-run")
        self.assertNotIn("audit_calibration_excluded_errors", payload)
        self.assertIsNone(payload["audit_calibration_error"])

    def test_eval_cli_compare_calibration_reports_json_error_when_no_runs_are_valid(self) -> None:
        bad_manifest = {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": "bad-run",
            "latest_checkpoint_path": None,
            "iterations": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad-run" / "manifest.json"
            write_manifest(bad_path, bad_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(bad_path), "--suggest-audit-calibration", "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["entries"], [])
        self.assertEqual(payload["errors"][0]["label"], "bad-run")
        self.assertIsNone(payload["audit_calibration"])
        self.assertEqual(
            payload["audit_calibration_error"],
            "no valid compared runs were available for audit calibration",
        )

    def test_eval_cli_compare_json_can_overlay_audit_profile_status(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-profile",
                        "long-run",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audit_profile"], "long-run")
        self.assertTrue(payload["audit_failed"])
        self.assertIsNone(payload["best_latest_benchmark_label"])
        self.assertEqual(payload["entries"][0]["audit_profile"], "long-run")
        self.assertFalse(payload["entries"][0]["audit_passed"])
        self.assertEqual(payload["entries"][0]["audit_failed_checks"], ["latest_benchmark_games"])

    def test_eval_cli_compare_json_audit_profile_can_pass(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, benchmark=False),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-profile",
                        "smoke",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audit_profile"], "smoke")
        self.assertFalse(payload["audit_failed"])
        self.assertTrue(payload["entries"][0]["audit_passed"])
        self.assertEqual(payload["entries"][0]["audit_failed_checks"], [])

    def test_eval_cli_compare_json_can_overlay_audit_config_status(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        strict_config = RunAuditConfig(
            min_latest_benchmark_win_rate=0.90,
            min_latest_benchmark_games=20,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, run_audit_config_payload(strict_config))

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-config",
                        str(config_path),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audit_profile"], str(config_path))
        self.assertTrue(payload["audit_failed"])
        self.assertEqual(payload["entries"][0]["audit_profile"], str(config_path))
        self.assertFalse(payload["entries"][0]["audit_passed"])
        self.assertIn("latest_benchmark_win_rate", payload["entries"][0]["audit_failed_checks"])

    def test_eval_cli_compare_fail_on_audit_returns_nonzero(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-profile",
                        "long-run",
                        "--fail-on-audit",
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 2)

    def test_eval_cli_compare_rejects_fail_on_audit_without_profile(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(["compare", str(manifest_path), "--fail-on-audit"])

        self.assertEqual(exit_code, 1)
        self.assertIn("--fail-on-audit requires --audit-profile or --audit-config", stderr.getvalue())

    def test_eval_cli_compare_rejects_profile_with_audit_config(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            config_path = Path(temp_dir) / "audit-config.json"
            write_manifest(manifest_path, manifest)
            write_manifest(config_path, run_audit_config_payload(RunAuditConfig()))

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-profile",
                        "smoke",
                        "--audit-config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--audit-profile cannot be combined with --audit-config", stderr.getvalue())

    def test_eval_cli_compare_returns_nonzero_but_prints_json_when_a_manifest_fails(self) -> None:
        healthy_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        bad_manifest = {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": "bad-run",
            "latest_checkpoint_path": None,
            "iterations": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            healthy_path = temp_path / "healthy-run" / "manifest.json"
            bad_path = temp_path / "bad-run" / "manifest.json"
            write_manifest(healthy_path, healthy_manifest)
            write_manifest(bad_path, bad_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(healthy_path), str(bad_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual([entry["label"] for entry in payload["entries"]], ["healthy-run"])
        self.assertEqual(payload["errors"][0]["label"], "bad-run")

    def test_eval_cli_compare_prints_text_table(self) -> None:
        source = source_metadata(branch="scott/report-source", head="abcdef123456", dirty=True)
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),),
            source=source,
        )
        manifest["iterations"][0]["collection_metrics"]["games_per_second"] = 12.345
        manifest["iterations"][0]["collection_metrics"]["peak_rss_mb"] = 8192.0
        manifest["iterations"][0]["benchmark"]["games_per_second"] = 3.2
        manifest["iterations"][0]["benchmark"]["peak_rss_mb"] = 16384.0
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(manifest_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("best_latest_benchmark: linear-run", stdout.getvalue())
        self.assertIn("min_benchmark_games_for_best: 50", stdout.getvalue())
        self.assertIn("linear-run", stdout.getvalue())
        self.assertIn("bench_wr", stdout.getvalue())
        self.assertIn("coll_gph", stdout.getvalue())
        self.assertIn("bench_gph", stdout.getvalue())
        self.assertIn("rss_hi_mb", stdout.getvalue())
        self.assertIn("source_provenance:", stdout.getvalue())
        self.assertIn("branch=scott/report-source", stdout.getvalue())
        self.assertIn("head=abcdef123456", stdout.getvalue())
        self.assertIn("dirty=yes", stdout.getvalue())
        self.assertNotIn("audit_profile:", stdout.getvalue())
        self.assertNotIn(" audit ", stdout.getvalue())
        row = next(line for line in stdout.getvalue().splitlines() if line.startswith("linear-run"))
        self.assertIn("44442", row)
        self.assertIn("11520", row)
        self.assertIn("16384.0", row)

    def test_eval_cli_compare_text_handles_missing_and_unavailable_source_metadata(self) -> None:
        missing_source_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        unavailable_source_manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=35, losses=15, capped_games=0),),
            source={
                "available": False,
                "repo_root": None,
                "branch": None,
                "head": None,
                "dirty": None,
                "error": "RuntimeError: git unavailable",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing-source" / "manifest.json"
            unavailable_path = Path(temp_dir) / "unavailable-source" / "manifest.json"
            write_manifest(missing_path, missing_source_manifest)
            write_manifest(unavailable_path, unavailable_source_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(missing_path), str(unavailable_path)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("source_provenance:", output)
        self.assertIn("- missing-source: -", output)
        self.assertIn(
            "- unavailable-source: available=no branch=- head=- dirty=- error=RuntimeError: git unavailable",
            output,
        )

    def test_eval_cli_compare_text_can_suggest_audit_calibration(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(manifest_path), "--suggest-audit-calibration"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("audit_calibration_suggestion:", output)
        self.assertIn("suggested_audit_flags:", output)
        self.assertIn("suggested_post_iteration_flags:", output)
        self.assertIn("minimum_evaluation_games: 13", output)
        self.assertIn("suggested_post_iteration_command_flags:", output)
        self.assertIn("--evaluation-games 13 --audit-after-iteration", output)
        self.assertIn("--min-latest-benchmark-games", output)
        self.assertIn("--audit-min-latest-benchmark-games", output)
        self.assertNotIn("calibration_excluded_errors:", output)

    def test_eval_cli_compare_text_reports_unavailable_calibration_when_no_runs_are_valid(self) -> None:
        bad_manifest = {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": "bad-run",
            "latest_checkpoint_path": None,
            "iterations": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad-run" / "manifest.json"
            write_manifest(bad_path, bad_manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["compare", str(bad_path), "--suggest-audit-calibration"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("audit_calibration_suggestion:", output)
        self.assertIn("unavailable: no valid compared runs were available for audit calibration", output)
        self.assertIn("calibration_excluded_errors:", output)
        self.assertIn("bad-run", output)

    def test_eval_cli_compare_text_can_overlay_audit_profile_status(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=40, losses=10, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-profile",
                        "long-run",
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("audit_profile: long-run", output)
        self.assertIn(" audit ", output)
        self.assertIn("best_latest_benchmark: -", output)
        row = next(line for line in output.splitlines() if line.startswith("linear-run"))
        self.assertIn("    no ", row)
        self.assertIn("audit_failures:", output)
        self.assertIn("latest_benchmark_games", output)

    def test_eval_cli_compare_text_omits_audit_failures_when_profile_passes(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, benchmark=False),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "linear-run" / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "compare",
                        str(manifest_path),
                        "--audit-profile",
                        "smoke",
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("audit_profile: smoke", output)
        row = next(line for line in output.splitlines() if line.startswith("linear-run"))
        self.assertIn("   yes ", row)
        self.assertNotIn("audit_failures:", output)

    def test_eval_cli_audit_smoke_profile_allows_missing_benchmark(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, benchmark=False),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(["audit", str(manifest_path), "--profile", "smoke", "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])

    def test_eval_cli_audit_smoke_profile_can_still_require_benchmark(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, benchmark=False),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--profile",
                        "smoke",
                        "--require-benchmark",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertIn("latest_benchmark_available", failed_check_names_from_payload(payload))

    def test_eval_cli_audit_smoke_profile_relaxes_numeric_thresholds(self) -> None:
        manifest = selfplay_manifest(
            iterations=(
                selfplay_iteration(iteration=1, wins=1, losses=19, capped_games=12),
            )
        )
        manifest["iterations"][0]["collection_metrics"] = collection_metrics(games=10, capped_games=8)
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as default_stdout:
                default_exit = eval_cli_main(["audit", str(manifest_path), "--json"])
            default_payload = json.loads(default_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as smoke_stdout:
                smoke_exit = eval_cli_main(["audit", str(manifest_path), "--profile", "smoke", "--json"])
            smoke_payload = json.loads(smoke_stdout.getvalue())

        self.assertEqual(default_exit, 2)
        self.assertIn("latest_benchmark_win_rate", failed_check_names_from_payload(default_payload))
        self.assertIn("latest_collection_capped_rate", failed_check_names_from_payload(default_payload))
        self.assertEqual(smoke_exit, 0)
        self.assertTrue(smoke_payload["passed"])

    def test_eval_cli_audit_latest_promotion_requirement_can_be_required_or_allowed(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, promotion_recorded=False),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as require_stdout:
                require_exit = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--min-latest-benchmark-games",
                        "20",
                        "--require-latest-promotion",
                        "--json",
                    ]
                )
            require_payload = json.loads(require_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as allow_stdout:
                allow_exit = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--min-latest-benchmark-games",
                        "20",
                        "--allow-missing-latest-promotion",
                        "--json",
                    ]
                )
            allow_payload = json.loads(allow_stdout.getvalue())

        self.assertEqual(require_exit, 2)
        self.assertIn("latest_promotion_recorded", failed_check_names_from_payload(require_payload))
        self.assertEqual(allow_exit, 0)
        self.assertTrue(allow_payload["passed"])

    def test_eval_cli_audit_long_run_profile_can_be_overridden(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=13, losses=7, capped_games=0),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as strict_stdout:
                strict_exit = eval_cli_main(["audit", str(manifest_path), "--profile", "long-run", "--json"])
            strict_payload = json.loads(strict_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as override_stdout:
                override_exit = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--profile",
                        "long-run",
                        "--min-latest-benchmark-games",
                        "20",
                        "--min-latest-benchmark-win-rate",
                        "0.55",
                        "--json",
                    ]
                )
            override_payload = json.loads(override_stdout.getvalue())

        self.assertEqual(strict_exit, 2)
        self.assertIn("latest_benchmark_games", failed_check_names_from_payload(strict_payload))
        self.assertEqual(override_exit, 0)
        self.assertTrue(override_payload["passed"])

    def test_eval_cli_audit_long_run_profile_enforces_stricter_benchmark_capped_rate(self) -> None:
        manifest = selfplay_manifest(
            iterations=(selfplay_iteration(iteration=1, wins=18, losses=2, capped_games=2),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.json"
            write_manifest(manifest_path, manifest)

            with patch("sys.stdout", new_callable=io.StringIO) as default_stdout:
                default_exit = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--min-latest-benchmark-games",
                        "20",
                        "--json",
                    ]
                )
            default_payload = json.loads(default_stdout.getvalue())

            with patch("sys.stdout", new_callable=io.StringIO) as long_run_stdout:
                long_run_exit = eval_cli_main(
                    [
                        "audit",
                        str(manifest_path),
                        "--profile",
                        "long-run",
                        "--min-latest-benchmark-games",
                        "20",
                        "--json",
                    ]
                )
            long_run_payload = json.loads(long_run_stdout.getvalue())

        self.assertEqual(default_exit, 0)
        self.assertTrue(default_payload["passed"])
        self.assertEqual(long_run_exit, 2)
        self.assertIn("latest_benchmark_capped_rate", failed_check_names_from_payload(long_run_payload))


def selfplay_manifest(
    *,
    iterations: tuple[dict, ...],
    invocation_configs: tuple[dict, ...] = (),
    source: dict | None = None,
) -> dict:
    payload = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": "run",
        "latest_checkpoint_path": iterations[-1]["checkpoint_path"],
        "iterations": list(iterations),
    }
    if source is not None:
        payload["source"] = source
    if invocation_configs:
        payload["invocation_configs"] = list(invocation_configs)
    return payload


def source_metadata(
    *,
    branch: str = "main",
    head: str = "abc123",
    dirty: bool = False,
) -> dict:
    return {
        "available": True,
        "repo_root": "/repo",
        "branch": branch,
        "head": head,
        "dirty": dirty,
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


def invocation_config(
    *,
    required_pool_size: int | None,
    promoted_checkpoint_count: int,
    max_historical_opponents: int = 3,
) -> dict:
    return {
        "resume": False,
        "first_iteration": 1,
        "iterations_requested": 1,
        "games_per_iteration": 10,
        "seed_start_argument": 1,
        "first_iteration_seed_start": 1,
        "initial_policy_spec": "random-legal",
        "evaluation_games": 10,
        "evaluation_seed_start": 1_000_000,
        "worker_count": 1,
        "opponent_pool": {
            "fixed_opponent_policy_specs": ["random-legal", "simple-legal"],
            "max_historical_opponents": max_historical_opponents,
            "promotion_registry_path": "promotions.json",
            "promotion_pool_registry_path": "promotions.json",
            "required_promoted_opponent_pool_size": required_pool_size,
            "promoted_checkpoint_policy_specs": [
                f"linear:runs/promoted-{index}/linear-policy.json"
                for index in range(1, promoted_checkpoint_count + 1)
            ],
        },
        "auto_promotion": {
            "enabled": False,
            "registry_path": None,
            "artifact_dir": None,
            "label_prefix": None,
            "notes": None,
            "allow_duplicate": False,
        },
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
    current_policy_spec: str = "random-legal",
    opponent_policy_specs: tuple[str, ...] = ("random-legal",),
) -> dict:
    policy_id = f"linear-selfplay-test-iter-{iteration:04d}"
    payload = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": iteration,
        "checkpoint_path": f"run/iteration-{iteration:04d}/linear-policy.json",
        "current_policy_spec": current_policy_spec,
        "opponent_policy_specs": list(opponent_policy_specs),
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
        "average_decision_rounds": 12.0,
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
