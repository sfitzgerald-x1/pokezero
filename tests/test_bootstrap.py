import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote

from pokezero.bootstrap import (
    DEFAULT_BENCHMARK_GAMES,
    DEFAULT_PREFLIGHT_GAMES,
    TEACHER_BOOTSTRAP_SCHEMA_VERSION,
    TeacherBenchmarkResult,
    benchmark_teacher_policy,
    run_teacher_bootstrap,
)
from pokezero.bootstrap_cli import TEACHER_BENCHMARK_PREFLIGHT_SCHEMA_VERSION, main as bootstrap_cli_main
from pokezero.collection import BenchmarkMatchupResult, BenchmarkReport, CollectionMetrics, read_rollout_records
from pokezero.env import StepResult, TerminalState
from pokezero.linear_policy import LinearTrainingConfig, linear_feature_fingerprint
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.rollout import RolloutConfig


def observation(mask: tuple[bool, ...]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((index,) for index in range(spec.token_count)),
        numeric_features=tuple((float(index),) for index in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
        perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
    )


class OneTurnEnv:
    def __init__(self) -> None:
        self._observation = observation((True, False, False, False, False, False, False, False, False))
        self._requested = ("p1", "p2")
        self._terminal = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self._requested = ("p1", "p2")
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self._observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self._observation.legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested

    def step(self, actions: dict[str, int]) -> StepResult:
        self._requested = ()
        self._terminal = TerminalState(winner="p1", turn_count=1)
        return StepResult(
            observations={},
            rewards={"p1": 1.0, "p2": -1.0},
            terminal=self._terminal,
            requested_players=(),
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal

    def close(self) -> None:
        self.closed = True


class TeacherBootstrapTest(unittest.TestCase):
    def test_run_teacher_bootstrap_writes_manifest_checkpoint_and_current_only_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = {
                "available": True,
                "repo_root": "/repo",
                "branch": "scott/source-test",
                "head": "abc123",
                "dirty": False,
            }

            with patch("pokezero.bootstrap.collect_source_metadata", return_value=source):
                result = run_teacher_bootstrap(
                    run_dir=run_dir,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-bootstrap-test",
                    ),
                    train_games=2,
                    validation_games=1,
                    teacher_policy_spec="simple-legal",
                    opponent_policy_specs=("random-legal",),
                    seed_start=10,
                    validation_seed_start=100,
                    benchmark_games=0,
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            full_train_records = read_rollout_records(result.full_train_rollout_path)
            train_records = read_rollout_records(result.train_rollout_path)
            validation_records = read_rollout_records(result.validation_rollout_path)
            checkpoint_payload = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_exists = result.checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertEqual(manifest["schema_version"], TEACHER_BOOTSTRAP_SCHEMA_VERSION)
        self.assertEqual(manifest["source"], source)
        self.assertEqual(manifest["checkpoint_policy_spec"], f"linear:{result.checkpoint_path}")
        self.assertEqual(manifest["teacher_policy_spec"], "simple-legal")
        self.assertEqual(manifest["opponent_policy_specs"], ["random-legal"])
        self.assertEqual(manifest["preflight"]["metrics"]["games"], DEFAULT_PREFLIGHT_GAMES)
        self.assertEqual(manifest["train_collection_metrics"]["games"], 2)
        self.assertEqual(manifest["validation_collection_metrics"]["games"], 1)
        self.assertEqual(manifest["teacher_decision_summary"]["total_decisions"], 3)
        self.assertEqual(manifest["teacher_decision_summary"]["unknown_move_decisions"], 0)
        self.assertEqual(manifest["teacher_decision_summary"]["fallback_decisions"], 0)
        self.assertIsNotNone(manifest["training"]["validation_metrics"])
        self.assertGreater(manifest["training"]["validation_metrics"]["examples"], 0)
        self.assertEqual(manifest["training"]["model"]["feature_fingerprint"], linear_feature_fingerprint())
        self.assertEqual(checkpoint_payload["policy_id"], "linear-bootstrap-test")
        self.assertEqual(checkpoint_payload["feature_fingerprint"], linear_feature_fingerprint())
        self.assertEqual(
            [record.policy_ids for record in full_train_records],
            [
                {"p1": "simple-legal", "p2": "random-legal"},
                {"p1": "random-legal", "p2": "simple-legal"},
            ],
        )
        self.assertEqual(
            [record.policy_ids for record in train_records],
            [
                {"p1": "simple-legal"},
                {"p2": "simple-legal"},
            ],
        )
        self.assertEqual({step.player_id for record in train_records for step in record.trajectory.steps}, {"p1", "p2"})
        self.assertEqual(validation_records[0].policy_ids, {"p1": "simple-legal"})

    def test_run_teacher_bootstrap_defaults_include_teacher_mirror_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_teacher_bootstrap(
                run_dir=Path(temp_dir) / "run",
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-bootstrap-test",
                ),
                train_games=2,
                validation_games=1,
                teacher_policy_spec="simple-legal",
                seed_start=10,
                validation_seed_start=100,
                benchmark_games=0,
                preflight_games=0,
            )

            full_train_records = read_rollout_records(result.full_train_rollout_path)

        self.assertEqual(result.opponent_policy_specs, ("simple-legal", "random-legal"))
        self.assertEqual(
            [record.policy_ids for record in full_train_records],
            [
                {"p1": "simple-legal", "p2": "simple-legal"},
                {"p1": "random-legal", "p2": "simple-legal"},
            ],
        )

    def test_run_teacher_bootstrap_refuses_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest already exists"):
                run_teacher_bootstrap(
                    run_dir=run_dir,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(feature_count=32, epochs=1),
                    train_games=1,
                    validation_games=1,
                    teacher_policy_spec="simple-legal",
                    opponent_policy_specs=("random-legal",),
                    benchmark_games=0,
                )

    def test_run_teacher_bootstrap_optional_benchmark_uses_actual_teacher_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_teacher_bootstrap(
                run_dir=Path(temp_dir) / "run",
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-bootstrap-test",
                ),
                train_games=1,
                validation_games=1,
                teacher_policy_spec="simple-legal",
                opponent_policy_specs=("random-legal",),
                benchmark_games=1,
                preflight_games=0,
            )

        self.assertIsNotNone(result.benchmark)
        labels = [matchup.label for matchup in result.benchmark.matchups] if result.benchmark is not None else []
        self.assertIn("linear-bootstrap-test vs simple-legal", labels)
        self.assertNotIn("linear-bootstrap-test vs scripted-teacher", labels)

    def test_benchmark_teacher_policy_runs_teacher_against_baseline_in_both_seats(self) -> None:
        result = benchmark_teacher_policy(
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            teacher_policy_spec="simple-legal",
            baseline_policy_specs=("random-legal",),
            games=1,
            seed_start=10,
        )
        report = result.benchmark

        self.assertEqual(report.total_games, 2)
        self.assertEqual(
            [matchup.label for matchup in report.matchups],
            ["simple-legal vs random-legal", "random-legal vs simple-legal"],
        )
        self.assertEqual(len(report.head_to_head_results), 1)
        head_to_head = report.head_to_head_results[0]
        self.assertEqual(head_to_head.first_policy_id, "simple-legal")
        self.assertEqual(head_to_head.second_policy_id, "random-legal")
        self.assertEqual(head_to_head.first_policy_wins, 1)
        self.assertEqual(head_to_head.second_policy_wins, 1)
        self.assertEqual(result.teacher_decision_summary["total_decisions"], 4)
        self.assertEqual(result.teacher_decision_summary["scripted_teacher_decisions"], 0)

    def test_benchmark_teacher_policy_reports_scripted_teacher_fallbacks(self) -> None:
        result = benchmark_teacher_policy(
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            teacher_policy_spec="scripted-teacher?allow_fallback=true",
            baseline_policy_specs=("random-legal",),
            games=1,
            seed_start=10,
        )

        self.assertEqual(result.benchmark.total_games, 2)
        self.assertEqual(result.teacher_decision_summary["scripted_teacher_decisions"], 2)
        self.assertEqual(result.teacher_decision_summary["fallback_decisions"], 2)
        self.assertEqual(result.teacher_decision_summary["fallback_reasons"]["dex unavailable"], 2)

    def test_benchmark_teacher_policy_rejects_non_positive_games(self) -> None:
        with self.assertRaisesRegex(ValueError, "games must be positive"):
            benchmark_teacher_policy(
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                teacher_policy_spec="simple-legal",
                baseline_policy_specs=("random-legal",),
                games=0,
            )

    def test_benchmark_teacher_policy_rejects_empty_or_same_teacher_baselines(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one baseline"):
            benchmark_teacher_policy(
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                teacher_policy_spec="simple-legal",
                baseline_policy_specs=(),
            )
        with self.assertRaisesRegex(ValueError, "distinct from the teacher"):
            benchmark_teacher_policy(
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                teacher_policy_spec="simple-legal",
                baseline_policy_specs=("simple-legal",),
            )

    def test_run_teacher_bootstrap_refuses_existing_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "train-rollouts.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "output path"):
                run_teacher_bootstrap(
                    run_dir=run_dir,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(feature_count=32, epochs=1),
                    train_games=1,
                    validation_games=1,
                    teacher_policy_spec="simple-legal",
                    opponent_policy_specs=("random-legal",),
                    benchmark_games=0,
                )

    def test_run_teacher_bootstrap_rejects_overlapping_seed_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "seed range"):
                run_teacher_bootstrap(
                    run_dir=Path(temp_dir) / "run",
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(feature_count=32, epochs=1),
                    train_games=2,
                    validation_games=1,
                    teacher_policy_spec="simple-legal",
                    opponent_policy_specs=("random-legal",),
                    seed_start=10,
                    validation_seed_start=11,
                    benchmark_games=0,
                    preflight_games=0,
                )

    def test_bootstrap_cli_teacher_wires_arguments(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=1.0,
            total_decision_rounds=2,
            total_simulator_turns=2,
            p1_wins=1,
            p2_wins=1,
            ties=0,
            capped_games=0,
        )
        fake_epoch = SimpleNamespace(examples=4, loss=0.25, accuracy=0.75)
        fake_validation = SimpleNamespace(examples=2, loss=0.5, accuracy=0.5)
        fake_result = SimpleNamespace(
            run_dir=Path("run"),
            train_rollout_path=Path("run/train-rollouts.jsonl"),
            validation_rollout_path=Path("run/validation-rollouts.jsonl"),
            checkpoint_path=Path("run/linear-bootstrap.json"),
            train_metrics=fake_metrics,
            validation_metrics=fake_metrics,
            training=SimpleNamespace(final_metrics=fake_epoch, validation_metrics=fake_validation),
            preflight_metrics=fake_metrics,
            teacher_decision_summary={"unknown_move_decisions": 0, "fallback_decisions": 0},
            benchmark=None,
            manifest_path=Path("run/manifest.json"),
            to_dict=lambda: {"schema_version": TEACHER_BOOTSTRAP_SCHEMA_VERSION},
        )

        with patch("pokezero.bootstrap_cli.run_teacher_bootstrap", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = bootstrap_cli_main(
                    [
                        "teacher",
                        "--run-dir",
                        "run",
                        "--train-games",
                        "2",
                        "--validation-games",
                        "1",
                        "--workers",
                        "3",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--opponent-policy",
                        "scripted-teacher?allow_fallback=true",
                        "--max-decision-rounds",
                        "12",
                        "--feature-count",
                        "64",
                        "--window-size",
                        "4",
                        "--epochs",
                        "2",
                        "--opponent-action-loss-weight",
                        "0.25",
                        "--benchmark-games",
                        "5",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["run_dir"], Path("run"))
        self.assertEqual(kwargs["train_games"], 2)
        self.assertEqual(kwargs["validation_games"], 1)
        self.assertEqual(kwargs["worker_count"], 3)
        self.assertEqual(kwargs["benchmark_games"], 5)
        self.assertEqual(kwargs["preflight_games"], DEFAULT_PREFLIGHT_GAMES)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 12)
        self.assertEqual(kwargs["training_config"].objective, "behavior-cloning")
        self.assertEqual(kwargs["training_config"].feature_count, 64)
        self.assertEqual(kwargs["training_config"].window_size, 4)
        self.assertEqual(kwargs["training_config"].epochs, 2)
        self.assertEqual(kwargs["training_config"].opponent_action_loss_weight, 0.25)
        expected_showdown_root = f"showdown_root={quote(str(Path('/tmp/showdown').resolve()), safe='')}"
        self.assertIn(expected_showdown_root, kwargs["teacher_policy_spec"])
        self.assertEqual(len(kwargs["opponent_policy_specs"]), 1)
        self.assertIn(expected_showdown_root, kwargs["opponent_policy_specs"][0])
        self.assertIn("checkpoint", stdout.getvalue())
        self.assertIn("manifest", stdout.getvalue())

    def test_bootstrap_cli_teacher_uses_default_mirror_opponents_and_benchmark(self) -> None:
        fake_metrics = CollectionMetrics(
            games=1,
            elapsed_seconds=1.0,
            total_decision_rounds=1,
            total_simulator_turns=1,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        fake_epoch = SimpleNamespace(examples=1, loss=0.25, accuracy=1.0)
        fake_result = SimpleNamespace(
            run_dir=Path("run"),
            train_rollout_path=Path("run/train-rollouts.jsonl"),
            validation_rollout_path=Path("run/validation-rollouts.jsonl"),
            checkpoint_path=Path("run/linear-bootstrap.json"),
            train_metrics=fake_metrics,
            validation_metrics=fake_metrics,
            preflight_metrics=fake_metrics,
            training=SimpleNamespace(final_metrics=fake_epoch, validation_metrics=None),
            teacher_decision_summary={"unknown_move_decisions": 0, "fallback_decisions": 0},
            benchmark=None,
            manifest_path=Path("run/manifest.json"),
            to_dict=lambda: {"schema_version": TEACHER_BOOTSTRAP_SCHEMA_VERSION},
        )

        with patch("pokezero.bootstrap_cli.run_teacher_bootstrap", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = bootstrap_cli_main(
                    [
                        "teacher",
                        "--run-dir",
                        "run",
                        "--train-games",
                        "1",
                        "--validation-games",
                        "1",
                        "--showdown-root",
                        "/tmp/showdown",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = run.call_args.kwargs
        self.assertIsNone(kwargs["opponent_policy_specs"])
        self.assertEqual(kwargs["benchmark_games"], DEFAULT_BENCHMARK_GAMES)

    def test_bootstrap_cli_teacher_benchmark_wires_arguments_and_json(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=1.0,
            total_decision_rounds=4,
            total_simulator_turns=4,
            p1_wins=2,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=12,
            games_per_matchup=2,
            matchups=(
                BenchmarkMatchupResult(
                    label="scripted-teacher vs random-legal",
                    p1_policy_id="scripted-teacher",
                    p2_policy_id="random-legal",
                    seed_start=10,
                    metrics=fake_metrics,
                ),
            ),
        )
        fake_result = TeacherBenchmarkResult(
            benchmark=fake_report,
            teacher_decision_summary={
                "total_decisions": 4,
                "scripted_teacher_decisions": 4,
                "unknown_move_decisions": 0,
                "fallback_decisions": 0,
                "fallback_reasons": {},
            },
        )

        with patch("pokezero.bootstrap_cli.benchmark_teacher_policy", return_value=fake_result) as benchmark:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = bootstrap_cli_main(
                    [
                        "teacher-benchmark",
                        "--games",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--max-decision-rounds",
                        "12",
                        "--seed-start",
                        "10",
                        "--teacher-policy",
                        "scripted-teacher?allow_fallback=true",
                        "--baseline-policy",
                        "random-legal",
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = benchmark.call_args.kwargs
        self.assertEqual(kwargs["games"], 2)
        self.assertEqual(kwargs["seed_start"], 10)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 12)
        expected_showdown_root = f"showdown_root={quote(str(Path('/tmp/showdown').resolve()), safe='')}"
        self.assertIn(expected_showdown_root, kwargs["teacher_policy_spec"])
        self.assertEqual(len(kwargs["baseline_policy_specs"]), 1)
        self.assertEqual(kwargs["baseline_policy_specs"][0], "random-legal")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema_version"], TEACHER_BENCHMARK_PREFLIGHT_SCHEMA_VERSION)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["checks"], [])
        self.assertEqual(payload["benchmark"]["total_games"], 2)
        self.assertEqual(payload["benchmark"]["head_to_heads"][0]["first_policy_id"], "scripted-teacher")
        self.assertEqual(payload["teacher_decision_summary"]["fallback_decisions"], 0)

    def test_bootstrap_cli_teacher_benchmark_can_fail_preflight_and_write_report(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=1.0,
            total_decision_rounds=4,
            total_simulator_turns=4,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=1,
        )
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=12,
            games_per_matchup=2,
            matchups=(
                BenchmarkMatchupResult(
                    label="scripted-teacher vs random-legal",
                    p1_policy_id="scripted-teacher",
                    p2_policy_id="random-legal",
                    seed_start=10,
                    metrics=fake_metrics,
                ),
            ),
        )
        fake_result = TeacherBenchmarkResult(
            benchmark=fake_report,
            teacher_decision_summary={
                "total_decisions": 4,
                "scripted_teacher_decisions": 4,
                "unknown_move_decisions": 1,
                "fallback_decisions": 1,
                "fallback_reasons": {"fallback": 1},
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "teacher-benchmark.json"
            with patch("pokezero.bootstrap_cli.benchmark_teacher_policy", return_value=fake_result):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = bootstrap_cli_main(
                        [
                            "teacher-benchmark",
                            "--games",
                            "2",
                            "--showdown-root",
                            "/tmp/showdown",
                            "--baseline-policy",
                            "random-legal",
                            "--min-teacher-win-rate",
                            "0.75",
                            "--max-capped-rate",
                            "0.25",
                            "--fail-on-degraded-decisions",
                            "--out",
                            str(report_path),
                            "--json",
                        ]
                    )
                payload = json.loads(stdout.getvalue())
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["passed"])
        self.assertEqual(report_payload, payload)
        failed_checks = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertEqual(
            failed_checks,
            {
                "teacher_win_rate:random-legal",
                "capped_rate:random-legal",
                "teacher_degraded_decisions",
            },
        )

    def test_bootstrap_cli_teacher_benchmark_text_prints_preflight_status_and_report_path(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=1.0,
            total_decision_rounds=4,
            total_simulator_turns=4,
            p1_wins=2,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=12,
            games_per_matchup=2,
            matchups=(
                BenchmarkMatchupResult(
                    label="scripted-teacher vs random-legal",
                    p1_policy_id="scripted-teacher",
                    p2_policy_id="random-legal",
                    seed_start=10,
                    metrics=fake_metrics,
                ),
            ),
        )
        fake_result = TeacherBenchmarkResult(
            benchmark=fake_report,
            teacher_decision_summary={
                "total_decisions": 4,
                "scripted_teacher_decisions": 4,
                "unknown_move_decisions": 0,
                "fallback_decisions": 0,
                "fallback_reasons": {},
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "teacher-benchmark.json"
            with patch("pokezero.bootstrap_cli.benchmark_teacher_policy", return_value=fake_result):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = bootstrap_cli_main(
                        [
                            "teacher-benchmark",
                            "--games",
                            "2",
                            "--showdown-root",
                            "/tmp/showdown",
                            "--baseline-policy",
                            "random-legal",
                            "--min-teacher-win-rate",
                            "0.75",
                            "--out",
                            str(report_path),
                        ]
                    )
            output = stdout.getvalue()
            report_exists = report_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertIn("preflight: PASS", output)
        self.assertIn("PASS teacher_win_rate:random-legal", output)
        self.assertIn(f"report: {report_path}", output)
        self.assertTrue(report_exists)

    def test_bootstrap_cli_teacher_benchmark_rejects_invalid_thresholds(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = bootstrap_cli_main(
                [
                    "teacher-benchmark",
                    "--games",
                    "2",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--min-teacher-win-rate",
                    "1.5",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--min-teacher-win-rate must be between 0 and 1", stderr.getvalue())

    def test_bootstrap_cli_teacher_benchmark_rejects_non_finite_thresholds(self) -> None:
        for flag in ("--min-teacher-win-rate", "--max-capped-rate"):
            with self.subTest(flag=flag):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    exit_code = bootstrap_cli_main(
                        [
                            "teacher-benchmark",
                            "--games",
                            "2",
                            "--showdown-root",
                            "/tmp/showdown",
                            flag,
                            "nan",
                        ]
                    )

                self.assertEqual(exit_code, 1)
                self.assertIn(f"{flag} must be between 0 and 1", stderr.getvalue())

    def test_bootstrap_cli_teacher_benchmark_fails_vacuous_threshold_when_teacher_row_missing(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=1.0,
            total_decision_rounds=4,
            total_simulator_turns=4,
            p1_wins=2,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=12,
            games_per_matchup=2,
            matchups=(
                BenchmarkMatchupResult(
                    label="other-policy vs random-legal",
                    p1_policy_id="other-policy",
                    p2_policy_id="random-legal",
                    seed_start=10,
                    metrics=fake_metrics,
                ),
            ),
        )
        fake_result = TeacherBenchmarkResult(
            benchmark=fake_report,
            teacher_decision_summary={
                "total_decisions": 4,
                "scripted_teacher_decisions": 4,
                "unknown_move_decisions": 0,
                "fallback_decisions": 0,
                "fallback_reasons": {},
            },
        )

        with patch("pokezero.bootstrap_cli.benchmark_teacher_policy", return_value=fake_result):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = bootstrap_cli_main(
                    [
                        "teacher-benchmark",
                        "--games",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--teacher-policy",
                        "scripted-teacher",
                        "--min-teacher-win-rate",
                        "0.55",
                        "--json",
                    ]
                )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["teacher_policy_id"], "scripted-teacher")
        self.assertFalse(payload["passed"])
        self.assertEqual(
            [check["name"] for check in payload["checks"]],
            ["teacher_head_to_head_present"],
        )
