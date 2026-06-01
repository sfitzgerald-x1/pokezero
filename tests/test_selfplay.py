import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import CollectionMetrics, read_rollout_records
from pokezero.env import StepResult, TerminalState
from pokezero.linear_policy import LinearTrainingConfig
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.rollout import RolloutConfig
from pokezero.selfplay import collect_selfplay_rollouts, run_selfplay_iterations
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
        self.assertIn(result.iterations[0].checkpoint_policy_spec, result.iterations[2].opponent_policy_specs)
        self.assertEqual(len(result.iterations[2].training_rollout_paths), 3)

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
        self.assertEqual(kwargs["training_config"].objective, "reward-weighted")
        self.assertIn("latest_checkpoint", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
