import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import CollectionMetrics, read_rollout_records, rollout_record_to_dict
from pokezero.env import StepResult, TerminalState
from pokezero.linear_policy import LinearPolicyModel, LinearTrainingConfig, save_linear_model
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.rollout import RolloutConfig
from pokezero.selfplay import SELFPLAY_RUN_SCHEMA_VERSION, collect_selfplay_rollouts, run_selfplay_iterations
from pokezero.selfplay_cli import main as selfplay_cli_main


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


class ResetFailingEnv:
    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        raise RuntimeError("boom")


class MultiActionEnv(OneTurnEnv):
    def __init__(self) -> None:
        super().__init__()
        self._observation = observation((True, True, True, True, True, True, False, False, False))


class SelfPlayTest(unittest.TestCase):
    def test_collect_selfplay_rollouts_alternates_current_policy_seat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            training_output_path = Path(temp_dir) / "training-rollouts.jsonl"

            metrics = collect_selfplay_rollouts(
                output_path=output_path,
                training_output_path=training_output_path,
                games=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=10,
                current_policy_spec="simple-legal",
                opponent_policy_specs=("random-legal",),
            )

            records = read_rollout_records(output_path)
            training_records = read_rollout_records(training_output_path)
        self.assertEqual(metrics.games, 2)
        self.assertEqual(records[0].policy_ids, {"p1": "simple-legal", "p2": "random-legal"})
        self.assertEqual(records[1].policy_ids, {"p1": "random-legal", "p2": "simple-legal"})
        self.assertEqual(training_records[0].policy_ids, {"p1": "simple-legal"})
        self.assertEqual(training_records[1].policy_ids, {"p2": "simple-legal"})
        self.assertEqual({step.player_id for record in training_records for step in record.trajectory.steps}, {"p1", "p2"})

    def test_collect_selfplay_rollouts_filters_current_seat_even_when_policy_ids_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            training_output_path = Path(temp_dir) / "training-rollouts.jsonl"

            collect_selfplay_rollouts(
                output_path=output_path,
                training_output_path=training_output_path,
                games=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=10,
                current_policy_spec="random-legal",
                opponent_policy_specs=("random-legal",),
            )

            training_records = read_rollout_records(training_output_path)

        self.assertEqual(training_records[0].policy_ids, {"p1": "random-legal"})
        self.assertEqual(training_records[1].policy_ids, {"p2": "random-legal"})

    def test_collect_selfplay_rollouts_parallel_preserves_order_and_training_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            training_output_path = Path(temp_dir) / "training-rollouts.jsonl"

            metrics = collect_selfplay_rollouts(
                output_path=output_path,
                training_output_path=training_output_path,
                games=4,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=10,
                current_policy_spec="simple-legal",
                opponent_policy_specs=("random-legal",),
                worker_count=2,
            )

            records = read_rollout_records(output_path)
            training_records = read_rollout_records(training_output_path)

        self.assertEqual(metrics.games, 4)
        self.assertEqual([record.seed for record in records], [10, 11, 12, 13])
        self.assertEqual([record.battle_id for record in records], ["selfplay-10", "selfplay-11", "selfplay-12", "selfplay-13"])
        self.assertEqual(
            [record.policy_ids for record in records],
            [
                {"p1": "simple-legal", "p2": "random-legal"},
                {"p1": "random-legal", "p2": "simple-legal"},
                {"p1": "simple-legal", "p2": "random-legal"},
                {"p1": "random-legal", "p2": "simple-legal"},
            ],
        )
        self.assertEqual(
            [record.policy_ids for record in training_records],
            [
                {"p1": "simple-legal"},
                {"p2": "simple-legal"},
                {"p1": "simple-legal"},
                {"p2": "simple-legal"},
            ],
        )

    def test_collect_selfplay_rollouts_parallel_matches_serial_for_rng_policies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            serial_rollouts = temp_path / "serial-rollouts.jsonl"
            serial_training = temp_path / "serial-training.jsonl"
            parallel_rollouts = temp_path / "parallel-rollouts.jsonl"
            parallel_training = temp_path / "parallel-training.jsonl"

            collect_selfplay_rollouts(
                output_path=serial_rollouts,
                training_output_path=serial_training,
                games=8,
                env_factory=MultiActionEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=100,
                current_policy_spec="simple-legal",
                opponent_policy_specs=("random-legal",),
                worker_count=1,
            )
            collect_selfplay_rollouts(
                output_path=parallel_rollouts,
                training_output_path=parallel_training,
                games=8,
                env_factory=MultiActionEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=100,
                current_policy_spec="simple-legal",
                opponent_policy_specs=("random-legal",),
                worker_count=4,
            )

            serial_payloads = normalized_record_payloads(serial_rollouts)
            parallel_payloads = normalized_record_payloads(parallel_rollouts)
            serial_training_payloads = normalized_record_payloads(serial_training)
            parallel_training_payloads = normalized_record_payloads(parallel_training)

        self.assertEqual(serial_payloads, parallel_payloads)
        self.assertEqual(serial_training_payloads, parallel_training_payloads)

    def test_run_selfplay_iterations_writes_checkpoint_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            result = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=3,
                games_per_iteration=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-selfplay-test",
                ),
                seed_start=20,
                fixed_opponent_policy_specs=("random-legal",),
            )

            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            checkpoint_exists = bool(result.latest_checkpoint_path and result.latest_checkpoint_path.exists())

        self.assertEqual(len(result.iterations), 3)
        self.assertTrue(checkpoint_exists)
        self.assertEqual(run_manifest["schema_version"], "pokezero.selfplay_run.v1")
        self.assertEqual(iteration_manifest["iteration"], 1)
        self.assertEqual(iteration_manifest["collection_metrics"]["games"], 2)
        self.assertEqual(iteration_manifest["training"]["model"]["policy_id"], "linear-selfplay-test-iter-0001")
        self.assertEqual(iteration_manifest["training"]["model"]["observation_schema_version"], "pokezero.observation.v0")
        self.assertEqual(iteration_manifest["training"]["model"]["action_schema_version"], "pokezero.action_space.v0")
        self.assertEqual(iteration_manifest["training"]["model"]["feature_schema_version"], "pokezero.linear_features.v1")
        self.assertEqual(len(iteration_manifest["training_rollout_paths"]), 1)
        self.assertTrue(iteration_manifest["training_rollout_path"].endswith("training-rollouts.jsonl"))
        self.assertEqual(iteration_manifest["worker_count"], 1)
        self.assertIn(result.iterations[0].checkpoint_policy_spec, result.iterations[2].opponent_policy_specs)
        self.assertEqual(len(result.iterations[2].training_rollout_paths), 3)

    def test_run_selfplay_iterations_records_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            validation_path = temp_path / "validation.jsonl"
            collect_selfplay_rollouts(
                output_path=validation_path,
                games=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=900,
                current_policy_spec="random-legal",
                opponent_policy_specs=("random-legal",),
            )

            run_selfplay_iterations(
                run_dir=temp_path / "run",
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-selfplay-test",
                ),
                seed_start=20,
                fixed_opponent_policy_specs=("random-legal",),
                validation_rollout_paths=(validation_path,),
            )

            iteration_manifest = json.loads((temp_path / "run" / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(iteration_manifest["validation_rollout_paths"], [str(validation_path)])
        self.assertIsNotNone(iteration_manifest["training"]["validation_metrics"])
        self.assertGreater(iteration_manifest["training"]["validation_metrics"]["examples"], 0)

    def test_run_selfplay_iterations_benchmarks_candidate_against_linear_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bootstrap_path = temp_path / "bootstrap-linear.json"
            save_linear_model(
                bootstrap_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="bootstrap-linear",
                ),
            )

            result = run_selfplay_iterations(
                run_dir=temp_path / "run",
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-selfplay-test",
                ),
                seed_start=20,
                initial_policy_spec=f"linear:{bootstrap_path}",
                fixed_opponent_policy_specs=("random-legal",),
                evaluation_games=1,
            )

            benchmark = result.iterations[0].benchmark
            iteration_manifest = json.loads((temp_path / "run" / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertIsNotNone(benchmark)
        head_to_heads = benchmark.head_to_head_results if benchmark is not None else ()
        self.assertIn(
            ("linear-selfplay-test-iter-0001", "bootstrap-linear"),
            {(row.first_policy_id, row.second_policy_id) for row in head_to_heads},
        )
        self.assertIn(
            ("linear-selfplay-test-iter-0001", "bootstrap-linear"),
            {
                (row["first_policy_id"], row["second_policy_id"])
                for row in iteration_manifest["benchmark"]["head_to_heads"]
            },
        )

    def test_run_selfplay_iterations_rejects_missing_validation_data_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            missing_path = Path(temp_dir) / "missing-validation.jsonl"

            with self.assertRaisesRegex(FileNotFoundError, "Validation rollout path"):
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=ResetFailingEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                    validation_rollout_paths=(missing_path,),
                )

            self.assertFalse((run_dir / "iteration-0001").exists())

    def test_run_selfplay_iterations_rejects_empty_validation_data_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            validation_path = Path(temp_dir) / "empty-validation.jsonl"
            validation_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=ResetFailingEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                    validation_rollout_paths=(validation_path,),
                )

            self.assertFalse((run_dir / "iteration-0001").exists())

    def test_run_selfplay_iterations_resume_preserves_validation_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            validation_path = temp_path / "validation.jsonl"
            config = LinearTrainingConfig(
                feature_count=32,
                epochs=1,
                shuffle_buffer_size=0,
                policy_id="linear-selfplay-test",
            )
            collect_selfplay_rollouts(
                output_path=validation_path,
                games=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=900,
                current_policy_spec="random-legal",
                opponent_policy_specs=("random-legal",),
            )
            run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                validation_rollout_paths=(validation_path,),
            )

            run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                resume=True,
            )

            iteration_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(iteration_manifest["validation_rollout_paths"], [str(validation_path)])
        self.assertIsNotNone(iteration_manifest["training"]["validation_metrics"])

    def test_run_selfplay_iterations_requires_resume_for_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-selfplay-test",
                ),
                seed_start=20,
                fixed_opponent_policy_specs=("random-legal",),
            )

            with self.assertRaisesRegex(ValueError, "resume=True"):
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                )

    def test_run_selfplay_iterations_resumes_from_latest_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            config = LinearTrainingConfig(
                feature_count=32,
                epochs=1,
                shuffle_buffer_size=0,
                policy_id="linear-selfplay-test",
            )
            first = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                seed_start=20,
                fixed_opponent_policy_specs=("random-legal",),
            )

            second = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=2,
                games_per_iteration=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                resume=True,
            )

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([iteration.iteration for iteration in second.iterations], [2, 3])
        self.assertEqual(second.prior_iteration_manifests[0]["checkpoint_path"], str(first.latest_checkpoint_path))
        self.assertEqual(second.iterations[0].current_policy_spec, first.iterations[0].checkpoint_policy_spec)
        self.assertEqual(second.iterations[0].seed_start, 22)
        self.assertEqual(len(second.iterations[0].training_rollout_paths), 2)
        self.assertEqual(len(manifest["iterations"]), 3)
        self.assertEqual(manifest["latest_checkpoint_path"], str(second.latest_checkpoint_path))

    def test_run_selfplay_iterations_writes_manifest_after_each_completed_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            config = LinearTrainingConfig(
                feature_count=32,
                epochs=1,
                shuffle_buffer_size=0,
                policy_id="linear-selfplay-test",
            )
            envs = iter((OneTurnEnv, ResetFailingEnv))
            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: next(envs)(),
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=config,
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                )

            manifest_after_crash = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            resumed = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                resume=True,
            )

            final_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(manifest_after_crash["iterations"]), 1)
        self.assertEqual(resumed.iterations[0].iteration, 2)
        self.assertEqual(resumed.iterations[0].seed_start, 21)
        self.assertEqual(len(final_manifest["iterations"]), 2)

    def test_run_selfplay_iterations_rejects_resume_config_mismatch_before_collecting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=LinearTrainingConfig(
                    feature_count=32,
                    epochs=1,
                    shuffle_buffer_size=0,
                    policy_id="linear-selfplay-test",
                ),
                seed_start=20,
                fixed_opponent_policy_specs=("random-legal",),
            )

            with self.assertRaisesRegex(ValueError, "feature_count"):
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=64,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                    resume=True,
                )

            self.assertFalse((run_dir / "iteration-0002").exists())

    def test_run_selfplay_iterations_refuses_orphaned_iteration_directory_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            (run_dir / "iteration-0001").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "iteration directories"):
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                )

    def test_selfplay_cli_iterate_wires_arguments(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=1.0,
            total_decision_rounds=4,
            total_simulator_turns=3,
            p1_wins=1,
            p2_wins=1,
            ties=0,
            capped_games=0,
        )
        fake_epoch = SimpleNamespace(loss=0.25, accuracy=0.75)
        fake_iteration = SimpleNamespace(
            iteration=1,
            metrics=fake_metrics,
            training=SimpleNamespace(final_metrics=fake_epoch),
            checkpoint_path=Path("run/iteration-0001/linear-policy.json"),
        )
        fake_result = SimpleNamespace(
            run_dir=Path("run"),
            iterations=(fake_iteration,),
            latest_checkpoint_path=Path("run/iteration-0001/linear-policy.json"),
        )
        with patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--resume",
                        "--games-per-iteration",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--opponent-policy",
                        "random-legal",
                        "--workers",
                        "2",
                        "--validation-data",
                        "heldout-a.jsonl",
                        "--validation-data",
                        "heldout-b.jsonl",
                        "--evaluation-games",
                        "3",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["iterations"], 1)
        self.assertTrue(kwargs["resume"])
        self.assertEqual(kwargs["games_per_iteration"], 2)
        self.assertEqual(kwargs["fixed_opponent_policy_specs"], ("random-legal",))
        self.assertEqual(kwargs["evaluation_games"], 3)
        self.assertEqual(kwargs["worker_count"], 2)
        self.assertEqual(kwargs["validation_rollout_paths"], (Path("heldout-a.jsonl"), Path("heldout-b.jsonl")))
        self.assertEqual(kwargs["training_config"].objective, "reward-weighted")
        self.assertEqual(kwargs["training_config"].capped_terminal_value, -0.25)
        self.assertIn("latest_checkpoint", stdout.getvalue())

    def test_selfplay_cli_report_prints_manifest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_report_manifest(run_dir)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("iterations: 1", output)
        self.assertIn("latest_checkpoint:", output)
        self.assertIn("linear-policy.json", output)
        self.assertIn("0.600", output)
        self.assertIn("0.125000", output)
        self.assertIn("0.8750", output)
        self.assertIn(" val ", output)
        self.assertIn("fit metrics measure imitation", output)
        self.assertNotIn("0.250000", output)

    def test_selfplay_cli_report_can_print_json_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_report_manifest(run_dir)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir), "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(payload["iterations"][0]["iteration"], 1)

    def test_selfplay_cli_report_reconstructs_from_iteration_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_report_manifest(run_dir, top_level=False)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("iterations: 1", output)
        self.assertIn("linear-policy.json", output)

    def test_selfplay_cli_report_warns_when_validation_paths_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_iteration = manifest["iterations"][0]
            first_iteration["validation_rollout_paths"] = ["validation-a.jsonl"]
            second_iteration = dict(first_iteration)
            second_iteration["iteration"] = 2
            second_iteration["validation_rollout_paths"] = ["validation-b.jsonl"]
            manifest["iterations"] = [first_iteration, second_iteration]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        self.assertEqual(exit_code, 0)
        self.assertIn("validation rollout paths changed", stdout.getvalue())


def normalized_record_payloads(path: Path) -> tuple[dict, ...]:
    payloads = []
    for record in read_rollout_records(path):
        payload = rollout_record_to_dict(record)
        payload["elapsed_seconds"] = 0.0
        payloads.append(payload)
    return tuple(payloads)


def write_report_manifest(run_dir: Path, *, top_level: bool = True) -> None:
    checkpoint_path = run_dir / "iteration-0001" / "linear-policy.json"
    iteration_dir = run_dir / "iteration-0001"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    iteration_manifest = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": 1,
        "rollout_path": str(iteration_dir / "rollouts.jsonl"),
        "training_rollout_path": str(iteration_dir / "training-rollouts.jsonl"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_policy_spec": f"linear:{checkpoint_path}",
        "current_policy_spec": "random-legal",
        "opponent_policy_specs": ["random-legal"],
        "training_rollout_paths": [str(iteration_dir / "training-rollouts.jsonl")],
        "validation_rollout_paths": [],
        "seed_start": 20,
        "worker_count": 2,
        "collection_metrics": {
            "games": 3,
            "elapsed_seconds": 2.0,
            "total_decision_rounds": 6,
            "total_simulator_turns": 5,
            "p1_wins": 2,
            "p2_wins": 0,
            "ties": 0,
            "capped_games": 1,
            "games_per_second": 1.5,
            "decisions_per_second": 3.0,
            "average_decision_rounds": 2.0,
            "average_simulator_turns": 1.67,
        },
        "training": {
            "config": {},
            "epochs": [
                {
                    "epoch": 1,
                    "examples": 6,
                    "loss": 0.25,
                    "accuracy": 0.75,
                    "elapsed_seconds": 0.5,
                }
            ],
            "validation_metrics": {
                "examples": 4,
                "loss": 0.125,
                "accuracy": 0.875,
                "elapsed_seconds": 0.25,
            },
            "model": {"policy_id": "linear-selfplay-test-iter-0001"},
        },
        "benchmark": {
            "format_id": "gen3randombattle",
            "max_decision_rounds": 250,
            "games_per_matchup": 10,
            "total_games": 20,
            "elapsed_seconds": 4.0,
            "games_per_second": 5.0,
            "decisions_per_second": 10.0,
            "matchups": [],
            "head_to_heads": [
                {
                    "label": "linear-selfplay-test-iter-0001 vs random-legal",
                    "first_policy_id": "linear-selfplay-test-iter-0001",
                    "second_policy_id": "random-legal",
                    "games": 20,
                    "first_policy_wins": 12,
                    "second_policy_wins": 8,
                    "ties": 0,
                    "capped_games": 1,
                    "first_policy_win_rate": 0.6,
                    "second_policy_win_rate": 0.4,
                }
            ],
        },
    }
    (iteration_dir / "manifest.json").write_text(json.dumps(iteration_manifest, indent=2), encoding="utf-8")
    if not top_level:
        return
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
                "run_dir": str(run_dir),
                "latest_checkpoint_path": str(checkpoint_path),
                "iterations": [iteration_manifest],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
