import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import (
    CollectionMetrics,
    linear_policy_factory_from_model_spec,
    read_rollout_records,
    rollout_record_to_dict,
)
from pokezero.dataset import TrajectoryDatasetConfig, is_training_cache_path, iter_training_batches
from pokezero.env import StepResult, TerminalState
from pokezero.linear_policy import (
    LINEAR_FEATURE_SCHEMA_VERSION,
    LinearPolicyModel,
    LinearTrainingConfig,
    save_linear_model,
)
from pokezero.observation import OBSERVATION_SCHEMA_VERSION, ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.promotion import PROMOTION_REGISTRY_SCHEMA_VERSION
from pokezero.evaluation import PromotionGateConfig
from pokezero.run_audit import RunAuditConfig, RunAuditFailure, run_audit_config_payload
from pokezero.rollout import RolloutConfig
from pokezero.selfplay import (
    SELFPLAY_RUN_SCHEMA_VERSION,
    SelfPlayPromotionConfig,
    _bounded_ordered_map,
    _promoted_checkpoint_specs,
    collect_selfplay_rollouts,
    load_selfplay_run_manifest,
    run_selfplay_iterations,
)
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


def _rollout_action_signature(records) -> list[tuple]:
    return [
        (
            record.seed,
            tuple(sorted(record.policy_ids.items())),
            None if record.terminal is None else record.terminal.winner,
            tuple((step.player_id, step.action_index) for step in record.trajectory.steps),
        )
        for record in records
    ]


class SelfPlayTest(unittest.TestCase):
    def test_bounded_ordered_map_limits_submitted_work(self) -> None:
        observed_submissions_at_result: list[tuple[int, ...]] = []

        class FakeFuture:
            def __init__(self, value: int, executor: "FakeExecutor") -> None:
                self._value = value
                self._executor = executor
                self.cancelled = False

            def result(self) -> int:
                observed_submissions_at_result.append(tuple(self._executor.submitted))
                return self._value

            def cancel(self) -> None:
                self.cancelled = True

        class FakeExecutor:
            def __init__(self) -> None:
                self.submitted: list[int] = []

            def submit(self, fn, value: int) -> FakeFuture:
                self.submitted.append(value)
                return FakeFuture(fn(value), self)

        executor = FakeExecutor()
        results = _bounded_ordered_map(executor, lambda value: value, range(5), buffersize=2)

        self.assertEqual(next(results), 0)
        self.assertEqual(executor.submitted, [0, 1])
        self.assertEqual(observed_submissions_at_result[0], (0, 1))
        self.assertEqual(next(results), 1)
        self.assertEqual(executor.submitted, [0, 1, 2])
        self.assertEqual(list(results), [2, 3, 4])
        self.assertEqual(executor.submitted, [0, 1, 2, 3, 4])

    def test_bounded_ordered_map_rejects_non_positive_buffer(self) -> None:
        with self.assertRaisesRegex(ValueError, "buffersize must be positive"):
            list(_bounded_ordered_map(SimpleNamespace(submit=None), lambda value: value, [1], buffersize=0))

    def test_collect_selfplay_rollouts_alternates_current_policy_seat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            training_output_path = Path(temp_dir) / "training-rollouts.jsonl"

            with patch("pokezero.collection.current_peak_rss_mb", return_value=88.0):
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
        self.assertEqual(metrics.peak_rss_mb, 88.0)
        self.assertEqual(records[0].policy_ids, {"p1": "simple-legal", "p2": "random-legal"})
        self.assertEqual(records[1].policy_ids, {"p1": "random-legal", "p2": "simple-legal"})
        self.assertEqual(training_records[0].policy_ids, {"p1": "simple-legal"})
        self.assertEqual(training_records[1].policy_ids, {"p2": "simple-legal"})
        self.assertEqual({step.player_id for record in training_records for step in record.trajectory.steps}, {"p1", "p2"})

    def test_collect_selfplay_rollouts_can_write_training_cache_chunks_without_raw_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_root = temp_path / "training-cache"
            cache_paths: list[Path] = []
            dataset_config = TrajectoryDatasetConfig(window_size=1)

            metrics = collect_selfplay_rollouts(
                output_path=None,
                training_cache_output_path=cache_root,
                training_cache_chunk_games=1,
                training_cache_dataset_config=dataset_config,
                training_cache_paths_out=cache_paths,
                games=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=50,
                current_policy_spec="random-legal",
                opponent_policy_specs=("simple-legal",),
            )

            batches = list(iter_training_batches(cache_paths, batch_size=8, config=dataset_config))
            self.assertTrue(all(is_training_cache_path(path) for path in cache_paths))

        self.assertEqual(metrics.games, 2)
        self.assertEqual([path.name for path in cache_paths], ["cache-00001", "cache-00002"])
        self.assertEqual(sum(batch.batch_size for batch in batches), 2)

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

    def test_run_selfplay_iterations_reuses_loaded_current_model_during_collection(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=32,
            window_size=1,
            policy_id="linear-loaded-once",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patch("pokezero.linear_policy.load_linear_model", return_value=model) as load:
                result = run_selfplay_iterations(
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
                    initial_policy_spec="linear:/tmp/current-policy.json?sample=true",
                    fixed_opponent_policy_specs=("random-legal",),
                )

        self.assertEqual(load.call_count, 1)
        self.assertEqual(result.iterations[0].current_policy_spec, "linear:/tmp/current-policy.json?sample=true")

    def test_reused_current_model_collection_matches_reloaded_model_collection(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=32,
            window_size=1,
            policy_id="linear-equivalent",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "linear-policy.json"
            save_linear_model(checkpoint_path, model)
            spec = f"linear:{checkpoint_path}?sample=true"
            reloaded_output_path = temp_path / "reloaded-rollouts.jsonl"
            reused_output_path = temp_path / "reused-rollouts.jsonl"

            collect_selfplay_rollouts(
                output_path=reloaded_output_path,
                games=4,
                env_factory=MultiActionEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=123,
                current_policy_spec=spec,
                opponent_policy_specs=("random-legal",),
            )
            collect_selfplay_rollouts(
                output_path=reused_output_path,
                games=4,
                env_factory=MultiActionEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=123,
                current_policy_spec=spec,
                opponent_policy_specs=("random-legal",),
                policy_factory_overrides={spec: linear_policy_factory_from_model_spec(spec, model)},
            )

            reloaded_records = read_rollout_records(reloaded_output_path)
            reused_records = read_rollout_records(reused_output_path)

        self.assertEqual(_rollout_action_signature(reused_records), _rollout_action_signature(reloaded_records))

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
            source = {
                "available": True,
                "repo_root": "/repo",
                "branch": "scott/source-test",
                "head": "abc123",
                "dirty": True,
            }

            with patch("pokezero.selfplay.collect_source_metadata", return_value=source):
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
        self.assertEqual(run_manifest["source"], source)
        self.assertEqual(iteration_manifest["iteration"], 1)
        self.assertEqual(iteration_manifest["source"], source)
        self.assertEqual(iteration_manifest["invocation_config"]["source"], source)
        self.assertEqual(run_manifest["invocation_configs"][0]["source"], source)
        self.assertEqual(iteration_manifest["collection_metrics"]["games"], 2)
        self.assertEqual(iteration_manifest["training"]["model"]["policy_id"], "linear-selfplay-test-iter-0001")
        self.assertEqual(iteration_manifest["training"]["model"]["observation_schema_version"], OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(iteration_manifest["training"]["model"]["action_schema_version"], "pokezero.action_space.v0")
        self.assertEqual(iteration_manifest["training"]["model"]["feature_schema_version"], LINEAR_FEATURE_SCHEMA_VERSION)
        self.assertRegex(iteration_manifest["training"]["model"]["feature_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(iteration_manifest["training_rollout_paths"]), 1)
        self.assertTrue(iteration_manifest["training_rollout_path"].endswith("training-rollouts.jsonl"))
        self.assertEqual(iteration_manifest["worker_count"], 1)
        self.assertIn(result.iterations[0].checkpoint_policy_spec, result.iterations[2].opponent_policy_specs)
        self.assertEqual(len(result.iterations[2].training_rollout_paths), 3)

    def test_load_selfplay_run_manifest_reconstructs_source_from_iteration_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = {"available": True, "repo_root": "/repo", "branch": "main", "head": "abc123", "dirty": False}
            with patch("pokezero.selfplay.collect_source_metadata", return_value=source):
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
            (run_dir / "manifest.json").unlink()

            manifest = json.loads(json.dumps(load_selfplay_run_manifest(run_dir)))

        self.assertEqual(manifest["source"], source)

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

    def test_run_selfplay_iterations_records_process_peak_rss_phase_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"
            with patch(
                "pokezero.selfplay.current_peak_rss_mb",
                side_effect=(
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                ),
            ), patch(
                "pokezero.collection.current_peak_rss_mb",
                return_value=19.0,
            ):
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
                    evaluation_games=1,
                    auto_promotion_config=SelfPlayPromotionConfig(
                        registry_path=registry_path,
                        gate_config=passing_promotion_gate_config(),
                    ),
                )

            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        expected = {
            "iteration_start": 10.0,
            "after_collection": 30.0,
            "after_training": 40.0,
            "after_checkpoint_save": 50.0,
            "after_benchmark": 60.0,
            "after_auto_promotion": 70.0,
        }
        expected_collection = {
            "collection_start": 11.0,
            "after_policy_factories": 12.0,
            "after_output_setup": 13.0,
            "after_first_record": 14.0,
            "after_half_records": 15.0,
            "after_all_records": 16.0,
            "after_record_collection": 17.0,
            "after_output_commit": 18.0,
            "after_summary": 20.0,
        }
        self.assertEqual(iteration_manifest["process_peak_rss_mb_by_phase"], expected)
        self.assertEqual(run_manifest["iterations"][0]["process_peak_rss_mb_by_phase"], expected)
        self.assertEqual(iteration_manifest["collection_metrics"]["peak_rss_mb"], 19.0)
        self.assertEqual(iteration_manifest["collection_metrics"]["peak_rss_mb_by_phase"], expected_collection)
        self.assertEqual(
            run_manifest["iterations"][0]["collection_metrics"]["peak_rss_mb_by_phase"],
            expected_collection,
        )

    def test_run_selfplay_iterations_rejects_neural_initial_policy_before_collecting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with self.assertRaisesRegex(ValueError, "linear checkpoints.*neural: initial policies"):
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
                    initial_policy_spec="neural:/tmp/model.pt?deterministic=true",
                    fixed_opponent_policy_specs=("random-legal",),
                )

            self.assertFalse((run_dir / "iteration-0001").exists())

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
        self.assertEqual(iteration_manifest["benchmark_reference_policy_specs"], [f"linear:{bootstrap_path}"])

    def test_run_selfplay_iterations_retains_static_linear_initial_benchmark_reference(self) -> None:
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
                iterations=2,
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

            second_manifest = json.loads((temp_path / "run" / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))
            benchmark = result.iterations[1].benchmark

        self.assertIsNotNone(benchmark)
        head_to_head_pairs = {
            (row.first_policy_id, row.second_policy_id)
            for row in (benchmark.head_to_head_results if benchmark is not None else ())
        }
        self.assertIn(("linear-selfplay-test-iter-0002", "linear-selfplay-test-iter-0001"), head_to_head_pairs)
        self.assertIn(("linear-selfplay-test-iter-0002", "bootstrap-linear"), head_to_head_pairs)
        self.assertEqual(
            result.iterations[1].benchmark_reference_policy_specs,
            (f"linear:{bootstrap_path}",),
        )
        self.assertEqual(second_manifest["benchmark_reference_policy_specs"], [f"linear:{bootstrap_path}"])
        self.assertIn(
            ("linear-selfplay-test-iter-0002", "bootstrap-linear"),
            {
                (row["first_policy_id"], row["second_policy_id"])
                for row in second_manifest["benchmark"]["head_to_heads"]
            },
        )

    def test_run_selfplay_iterations_resume_preserves_static_benchmark_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            bootstrap_path = temp_path / "bootstrap-linear.json"
            save_linear_model(
                bootstrap_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="bootstrap-linear",
                ),
            )
            config = LinearTrainingConfig(
                feature_count=32,
                epochs=1,
                shuffle_buffer_size=0,
                policy_id="linear-selfplay-test",
            )
            first = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                seed_start=20,
                initial_policy_spec=f"linear:{bootstrap_path}",
                fixed_opponent_policy_specs=("random-legal",),
                evaluation_games=1,
            )

            second = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                evaluation_games=1,
                resume=True,
            )

            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(first.iterations[0].benchmark_reference_policy_specs, (f"linear:{bootstrap_path}",))
        self.assertEqual(second.iterations[0].benchmark_reference_policy_specs, (f"linear:{bootstrap_path}",))
        self.assertIn(
            ("linear-selfplay-test-iter-0002", "bootstrap-linear"),
            {
                (row["first_policy_id"], row["second_policy_id"])
                for row in second_manifest["benchmark"]["head_to_heads"]
            },
        )

    def test_run_selfplay_iterations_resume_derives_static_reference_from_older_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            bootstrap_path = temp_path / "bootstrap-linear.json"
            save_linear_model(
                bootstrap_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="bootstrap-linear",
                ),
            )
            config = LinearTrainingConfig(
                feature_count=32,
                epochs=1,
                shuffle_buffer_size=0,
                policy_id="linear-selfplay-test",
            )
            run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                seed_start=20,
                initial_policy_spec=f"linear:{bootstrap_path}",
                fixed_opponent_policy_specs=("random-legal",),
                evaluation_games=1,
            )
            run_manifest_path = run_dir / "manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            for iteration in run_manifest["iterations"]:
                iteration.pop("benchmark_reference_policy_specs", None)
            run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
            iteration_manifest_path = run_dir / "iteration-0001" / "manifest.json"
            iteration_manifest = json.loads(iteration_manifest_path.read_text(encoding="utf-8"))
            iteration_manifest.pop("benchmark_reference_policy_specs", None)
            iteration_manifest_path.write_text(json.dumps(iteration_manifest), encoding="utf-8")

            resumed = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                evaluation_games=1,
                resume=True,
            )

            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(resumed.iterations[0].benchmark_reference_policy_specs, (f"linear:{bootstrap_path}",))
        self.assertIn(
            ("linear-selfplay-test-iter-0002", "bootstrap-linear"),
            {
                (row["first_policy_id"], row["second_policy_id"])
                for row in second_manifest["benchmark"]["head_to_heads"]
            },
        )

    def test_run_selfplay_iterations_uses_promotion_registry_for_historical_opponents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            promoted_checkpoint_path = temp_path / "promoted-linear.json"
            registry_path = temp_path / "promotions.json"
            save_linear_model(
                promoted_checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-promoted",
                ),
            )
            write_promotion_registry(
                registry_path,
                checkpoint_paths=(promoted_checkpoint_path,),
                policy_ids=("linear-promoted",),
            )

            result = run_selfplay_iterations(
                run_dir=temp_path / "run",
                iterations=2,
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
                max_historical_opponents=1,
                promotion_registry_path=registry_path,
                required_promoted_opponent_pool_size=1,
            )
            run_manifest = json.loads((temp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
            iteration_manifest = json.loads((temp_path / "run" / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        promoted_spec = f"linear:{promoted_checkpoint_path.resolve(strict=False)}"
        expected_pool_config = {
            "fixed_opponent_policy_specs": ["random-legal"],
            "max_historical_opponents": 1,
            "historical_opponent_selection": "recent",
            "promotion_registry_path": str(registry_path),
            "promotion_pool_registry_path": str(registry_path),
            "required_promoted_opponent_pool_size": 1,
            "promoted_checkpoint_policy_specs": [promoted_spec],
        }
        self.assertEqual(result.iterations[0].opponent_policy_specs, ("random-legal", promoted_spec))
        self.assertEqual(result.iterations[1].opponent_policy_specs, ("random-legal", promoted_spec))
        self.assertNotIn(result.iterations[0].checkpoint_policy_spec, result.iterations[1].opponent_policy_specs)
        self.assertEqual(run_manifest["invocation_configs"][0]["opponent_pool"], expected_pool_config)
        self.assertEqual(iteration_manifest["opponent_pool_config"], expected_pool_config)
        self.assertEqual(iteration_manifest["invocation_config"]["opponent_pool"], expected_pool_config)

    def test_run_selfplay_iterations_can_require_promoted_opponent_pool_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            promoted_checkpoint_path = temp_path / "promoted-linear.json"
            registry_path = temp_path / "promotions.json"
            save_linear_model(
                promoted_checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-promoted",
                ),
            )
            write_promotion_registry(
                registry_path,
                checkpoint_paths=(promoted_checkpoint_path,),
                policy_ids=("linear-promoted",),
            )

            with self.assertRaisesRegex(ValueError, "promoted opponent pool has 1 selectable opponents.*required 2"):
                run_selfplay_iterations(
                    run_dir=temp_path / "run",
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: self.fail("collection should not start when promoted pool is undersized"),
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                    max_historical_opponents=2,
                    promotion_registry_path=registry_path,
                    required_promoted_opponent_pool_size=2,
                )

    def test_run_selfplay_iterations_rejects_required_promoted_pool_without_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requires a promotion registry"):
                run_selfplay_iterations(
                    run_dir=Path(temp_dir) / "run",
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: self.fail("collection should not start without required registry"),
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                    required_promoted_opponent_pool_size=1,
                )

    def test_run_selfplay_iterations_rejects_required_promoted_pool_above_historical_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            promoted_checkpoint_path = temp_path / "promoted-linear.json"
            registry_path = temp_path / "promotions.json"
            save_linear_model(
                promoted_checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-promoted",
                ),
            )
            write_promotion_registry(
                registry_path,
                checkpoint_paths=(promoted_checkpoint_path,),
                policy_ids=("linear-promoted",),
            )

            with self.assertRaisesRegex(ValueError, "cannot exceed max_historical_opponents"):
                run_selfplay_iterations(
                    run_dir=temp_path / "run",
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: self.fail("collection should not start when requirement exceeds cap"),
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    training_config=LinearTrainingConfig(
                        feature_count=32,
                        epochs=1,
                        shuffle_buffer_size=0,
                        policy_id="linear-selfplay-test",
                    ),
                    fixed_opponent_policy_specs=("random-legal",),
                    max_historical_opponents=1,
                    promotion_registry_path=registry_path,
                    required_promoted_opponent_pool_size=2,
                )

    def test_promoted_checkpoint_specs_verify_registry_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            write_promotion_registry(registry_path, checkpoint_paths=(temp_path / "missing-linear.json",))

            with self.assertRaisesRegex(ValueError, "promotion registry verification failed"):
                _promoted_checkpoint_specs(registry_path)

    def test_promoted_checkpoint_specs_verify_loadable_registry_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "bad-linear.json"
            checkpoint_path.write_text("{}", encoding="utf-8")
            registry_path = temp_path / "promotions.json"
            write_promotion_registry(registry_path, checkpoint_paths=(checkpoint_path,))

            with self.assertRaisesRegex(ValueError, "checkpoint_policy_loadable"):
                _promoted_checkpoint_specs(registry_path)

    def test_promoted_checkpoint_specs_resolves_relative_registry_specs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "runs" / "promotions.json"
            checkpoint_path = registry_path.parent / "promoted" / "linear-policy.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-promoted-1",
                ),
            )
            write_promotion_registry(
                registry_path,
                checkpoint_paths=(Path("promoted/linear-policy.json"),),
                policy_ids=("linear-promoted-1",),
            )

            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                specs = _promoted_checkpoint_specs(Path("runs/promotions.json"))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(specs, (f"linear:{checkpoint_path.resolve(strict=False)}",))

    def test_run_selfplay_iterations_excludes_relative_current_policy_from_promoted_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            previous_checkpoint_path = temp_path / "previous-linear.json"
            current_checkpoint_path = temp_path / "current-linear.json"
            registry_path = temp_path / "promotions.json"
            save_linear_model(
                previous_checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-promoted-previous",
                ),
            )
            save_linear_model(
                current_checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-promoted-current",
                ),
            )
            write_promotion_registry(
                registry_path,
                checkpoint_paths=(previous_checkpoint_path, current_checkpoint_path),
                policy_ids=("linear-promoted-previous", "linear-promoted-current"),
            )
            current_relative_spec = f"linear:{os.path.relpath(current_checkpoint_path, Path.cwd())}"

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
                initial_policy_spec=current_relative_spec,
                fixed_opponent_policy_specs=("random-legal",),
                max_historical_opponents=2,
                promotion_registry_path=registry_path,
                required_promoted_opponent_pool_size=1,
            )

        previous_spec = f"linear:{previous_checkpoint_path.resolve(strict=False)}"
        current_registry_spec = f"linear:{current_checkpoint_path.resolve(strict=False)}"
        self.assertEqual(result.iterations[0].current_policy_spec, current_relative_spec)
        self.assertEqual(result.iterations[0].opponent_policy_specs, ("random-legal", previous_spec))
        self.assertNotIn(current_registry_spec, result.iterations[0].opponent_policy_specs)

    def test_run_selfplay_iterations_auto_promotes_and_feeds_next_opponent_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"

            result = run_selfplay_iterations(
                run_dir=temp_path / "run",
                iterations=2,
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
                max_historical_opponents=2,
                evaluation_games=1,
                promotion_registry_path=registry_path,
                auto_promotion_config=SelfPlayPromotionConfig(
                    registry_path=registry_path,
                    artifact_dir=artifact_dir,
                    gate_config=passing_promotion_gate_config(),
                    label_prefix="candidate",
                ),
            )

            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            iteration_one_manifest = json.loads((temp_path / "run" / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            iteration_two_manifest = json.loads((temp_path / "run" / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        first_promotion = result.iterations[0].promotion
        second_promotion = result.iterations[1].promotion
        first_promoted_spec = f"linear:{Path(registry_payload['entries'][0]['checkpoint_path']).resolve(strict=False)}"
        self.assertTrue(first_promotion.recorded if first_promotion else False)
        self.assertTrue(second_promotion.recorded if second_promotion else False)
        self.assertEqual(len(registry_payload["entries"]), 2)
        self.assertEqual(registry_payload["entries"][0]["label"], "candidate-0001")
        self.assertEqual(registry_payload["entries"][1]["label"], "candidate-0002")
        self.assertEqual(Path(registry_payload["entries"][0]["checkpoint_path"]).parent, artifact_dir)
        self.assertEqual(result.iterations[1].current_policy_spec, first_promoted_spec)
        self.assertNotIn(first_promoted_spec, result.iterations[1].opponent_policy_specs)
        self.assertEqual(iteration_one_manifest["promotion"]["recorded"], True)
        self.assertEqual(iteration_two_manifest["promotion"]["recorded"], True)
        self.assertEqual(iteration_two_manifest["current_policy_spec"], first_promoted_spec)
        self.assertEqual(iteration_two_manifest["promotion"]["gate_result"]["incumbent_policy_id"], "linear-selfplay-test-iter-0001")

    def test_run_selfplay_iterations_resume_uses_promoted_artifact_as_current_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"
            config = LinearTrainingConfig(
                feature_count=32,
                epochs=1,
                shuffle_buffer_size=0,
                policy_id="linear-selfplay-test",
            )

            first = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                seed_start=20,
                fixed_opponent_policy_specs=("random-legal",),
                max_historical_opponents=2,
                evaluation_games=1,
                promotion_registry_path=registry_path,
                auto_promotion_config=SelfPlayPromotionConfig(
                    registry_path=registry_path,
                    artifact_dir=artifact_dir,
                    gate_config=passing_promotion_gate_config(),
                    label_prefix="candidate",
                ),
            )
            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            promoted_spec = f"linear:{Path(registry_payload['entries'][0]['checkpoint_path']).resolve(strict=False)}"

            second = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                training_config=config,
                fixed_opponent_policy_specs=("random-legal",),
                max_historical_opponents=2,
                promotion_registry_path=registry_path,
                resume=True,
            )

        self.assertEqual(first.iterations[0].promotion.recorded if first.iterations[0].promotion else False, True)
        self.assertEqual(second.iterations[0].current_policy_spec, promoted_spec)
        self.assertNotIn(promoted_spec, second.iterations[0].opponent_policy_specs)

    def test_run_selfplay_iterations_static_reference_does_not_gate_auto_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bootstrap_path = temp_path / "bootstrap-linear.json"
            registry_path = temp_path / "promotions.json"
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
                iterations=2,
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
                auto_promotion_config=SelfPlayPromotionConfig(
                    registry_path=registry_path,
                    gate_config=PromotionGateConfig(
                        min_benchmark_win_rate=1.0,
                        min_incumbent_win_rate=0.0,
                        min_benchmark_games=0,
                        min_incumbent_games=0,
                        max_collection_capped_rate=1.0,
                        max_benchmark_capped_rate=1.0,
                        max_incumbent_capped_rate=1.0,
                        min_incumbent_win_rate_lower_bound=0.0,
                        opponent_min_win_rates={
                            "random-legal": 0.0,
                            "simple-legal": 0.0,
                        },
                    ),
                    label_prefix="candidate",
                ),
            )

        second_promotion = result.iterations[1].promotion
        self.assertTrue(second_promotion.recorded if second_promotion else False)
        check_names = {
            check["name"]
            for check in second_promotion.gate_result.to_dict()["checks"]
        } if second_promotion else set()
        self.assertNotIn("benchmark_games:bootstrap-linear", check_names)
        self.assertNotIn("benchmark_win_rate:bootstrap-linear", check_names)
        self.assertIn(
            "bootstrap-linear",
            {
                opponent.opponent_policy_id
                for opponent in second_promotion.gate_result.benchmark_opponents
            } if second_promotion else set(),
        )

    def test_run_selfplay_iterations_auto_promotion_benchmarks_frozen_registry_champion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            champion_checkpoint_path = temp_path / "champion-linear.json"
            registry_path = temp_path / "promotions.json"
            save_linear_model(
                champion_checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=32,
                    window_size=1,
                    policy_id="linear-champion",
                ),
            )
            write_promotion_registry(
                registry_path,
                checkpoint_paths=(champion_checkpoint_path,),
                policy_ids=("linear-champion",),
            )

            result = run_selfplay_iterations(
                run_dir=temp_path / "run",
                iterations=2,
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
                max_historical_opponents=2,
                evaluation_games=1,
                promotion_registry_path=registry_path,
                auto_promotion_config=SelfPlayPromotionConfig(
                    registry_path=registry_path,
                    gate_config=PromotionGateConfig(
                        min_benchmark_win_rate=0.0,
                        min_incumbent_win_rate=0.0,
                        min_benchmark_games=0,
                        min_incumbent_games=0,
                        max_collection_capped_rate=1.0,
                        max_benchmark_capped_rate=1.0,
                        max_incumbent_capped_rate=1.0,
                        min_incumbent_win_rate_lower_bound=0.0,
                        required_benchmark_opponents=("missing-opponent",),
                    ),
                ),
            )

        second_promotion = result.iterations[1].promotion
        failed_checks = {
            check["name"]
            for check in second_promotion.gate_result.to_dict()["checks"]
            if not check["passed"]
        } if second_promotion else set()
        self.assertFalse(second_promotion.recorded if second_promotion else True)
        self.assertEqual(second_promotion.gate_result.incumbent_policy_id if second_promotion else None, "linear-champion")
        self.assertGreater(second_promotion.gate_result.incumbent_games if second_promotion else 0, 0)
        self.assertNotIn("incumbent_benchmark_opponent:linear-champion", failed_checks)
        self.assertIn("benchmark_opponent:missing-opponent", failed_checks)

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
        self.assertEqual(len(manifest["invocation_configs"]), 2)
        self.assertFalse(manifest["invocation_configs"][0]["resume"])
        self.assertEqual(manifest["invocation_configs"][0]["seed_start_argument"], 20)
        self.assertEqual(manifest["invocation_configs"][0]["first_iteration_seed_start"], 20)
        self.assertTrue(manifest["invocation_configs"][1]["resume"])
        self.assertEqual(manifest["invocation_configs"][1]["seed_start_argument"], 1)
        self.assertEqual(manifest["invocation_configs"][1]["first_iteration_seed_start"], 22)
        self.assertEqual(second.iterations[0].to_manifest_dict()["invocation_config"], manifest["invocation_configs"][1])

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

    def test_run_selfplay_iterations_post_iteration_audit_stops_before_next_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with self.assertRaisesRegex(RunAuditFailure, "latest_benchmark_available") as raised:
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
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
                    post_iteration_audit_config=RunAuditConfig(require_benchmark=True),
                )

            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(raised.exception.result.passed)
        self.assertEqual(len(run_manifest["iterations"]), 1)
        self.assertFalse((run_dir / "iteration-0002").exists())

    def test_runtime_health_audit_failure_mode_still_stops_on_runtime_health_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with self.assertRaisesRegex(RunAuditFailure, "latest_benchmark_available") as raised:
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
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
                    post_iteration_audit_config=RunAuditConfig(require_benchmark=True),
                    post_iteration_audit_failure_mode="runtime-health",
                )

            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(raised.exception.result.passed)
        self.assertEqual(len(run_manifest["iterations"]), 1)
        self.assertFalse((run_dir / "iteration-0002").exists())

    def test_runtime_health_audit_failure_mode_stops_when_mixed_with_promotion_strength_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with self.assertRaises(RunAuditFailure) as raised:
                run_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
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
                    post_iteration_audit_config=RunAuditConfig(
                        require_benchmark=True,
                        require_latest_promotion=True,
                    ),
                    post_iteration_audit_failure_mode="runtime-health",
                )

            failed_names = {check.name for check in raised.exception.result.blocking_failed_checks}
            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertIn("latest_benchmark_available", failed_names)
        self.assertIn("latest_promotion_recorded", failed_names)
        self.assertEqual(len(run_manifest["iterations"]), 1)
        self.assertFalse((run_dir / "iteration-0002").exists())

    def test_post_iteration_audit_failure_prevents_auto_promotion_when_latest_promotion_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"

            with self.assertRaisesRegex(RunAuditFailure, "latest_process_peak_rss_mb") as raised:
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
                    evaluation_games=1,
                    promotion_registry_path=registry_path,
                    auto_promotion_config=SelfPlayPromotionConfig(
                        registry_path=registry_path,
                        artifact_dir=artifact_dir,
                        gate_config=passing_promotion_gate_config(),
                        label_prefix="candidate",
                    ),
                    post_iteration_audit_config=RunAuditConfig(
                        min_latest_benchmark_win_rate=0.0,
                        min_latest_benchmark_games=0,
                        max_latest_benchmark_capped_rate=1.0,
                        max_latest_process_peak_rss_mb=0.0,
                        max_benchmark_win_rate_drop=1.0,
                        require_benchmark=True,
                    ),
                )

            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(raised.exception.result.passed)
        self.assertFalse(registry_path.exists())
        self.assertEqual(list(artifact_dir.glob("*.json")), [])
        self.assertIsNone(iteration_manifest["promotion"])
        self.assertIsNone(run_manifest["iterations"][0]["promotion"])

    def test_post_iteration_audit_still_checks_promotion_failures_after_auto_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"

            with self.assertRaisesRegex(RunAuditFailure, "consecutive_promotion_failures") as raised:
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
                    evaluation_games=1,
                    promotion_registry_path=registry_path,
                    auto_promotion_config=SelfPlayPromotionConfig(
                        registry_path=registry_path,
                        gate_config=PromotionGateConfig(min_benchmark_win_rate=1.0, min_benchmark_games=0),
                        label_prefix="candidate",
                    ),
                    post_iteration_audit_config=RunAuditConfig(
                        min_latest_benchmark_win_rate=0.0,
                        min_latest_benchmark_games=0,
                        max_latest_benchmark_capped_rate=1.0,
                        max_benchmark_win_rate_drop=1.0,
                        max_consecutive_promotion_failures=0,
                        require_benchmark=True,
                    ),
                )

            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(raised.exception.result.passed)
        self.assertFalse(registry_path.exists())
        self.assertEqual(iteration_manifest["promotion"]["recorded"], False)
        self.assertEqual(run_manifest["iterations"][0]["promotion"]["recorded"], False)

    def test_runtime_health_audit_failure_mode_continues_on_promotion_strength_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                result = run_selfplay_iterations(
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
                    evaluation_games=1,
                    promotion_registry_path=registry_path,
                    auto_promotion_config=SelfPlayPromotionConfig(
                        registry_path=registry_path,
                        gate_config=PromotionGateConfig(min_benchmark_win_rate=1.0, min_benchmark_games=0),
                        label_prefix="candidate",
                    ),
                    post_iteration_audit_config=RunAuditConfig(
                        min_latest_benchmark_win_rate=0.0,
                        min_latest_benchmark_games=0,
                        max_latest_benchmark_capped_rate=1.0,
                        max_benchmark_win_rate_drop=1.0,
                        max_consecutive_promotion_failures=0,
                        require_benchmark=True,
                    ),
                    post_iteration_audit_failure_mode="runtime-health",
                )

            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(iteration_manifest["promotion"]["recorded"], False)
        self.assertEqual(run_manifest["iterations"][0]["promotion"]["recorded"], False)
        self.assertIn("audit_nonblocking_failed_checks: consecutive_promotion_failures", stderr.getvalue())

    def test_run_selfplay_iterations_post_iteration_audit_passes_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            result = run_selfplay_iterations(
                run_dir=run_dir,
                iterations=2,
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
                evaluation_games=1,
                post_iteration_audit_config=RunAuditConfig(
                    min_latest_benchmark_win_rate=0.0,
                    min_latest_benchmark_games=0,
                    max_latest_benchmark_capped_rate=1.0,
                    max_benchmark_win_rate_drop=1.0,
                    require_benchmark=True,
                ),
            )
            run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            second_manifest_exists = (run_dir / "iteration-0002" / "manifest.json").exists()

        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(len(run_manifest["iterations"]), 2)
        self.assertTrue(second_manifest_exists)

    def test_run_selfplay_iterations_reports_warning_only_post_iteration_audit_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                result = run_selfplay_iterations(
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
                    evaluation_games=1,
                    post_iteration_audit_config=RunAuditConfig(
                        min_latest_benchmark_win_rate=0.0,
                        min_latest_benchmark_games=0,
                        max_latest_average_decision_rounds=0.5,
                        max_latest_benchmark_capped_rate=1.0,
                        max_benchmark_win_rate_drop=1.0,
                        require_benchmark=True,
                        warning_check_names=("latest_average_decision_rounds",),
                    ),
                )

        self.assertEqual(len(result.iterations), 1)
        self.assertIn("audit_warning_checks: latest_average_decision_rounds", stderr.getvalue())

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
                        "--benchmark-reference-policy",
                        "linear:bootstrap.json",
                        "--workers",
                        "2",
                        "--opponent-action-loss-weight",
                        "0.3",
                        "--auto-promote",
                        "--promotion-registry",
                        "promotions.json",
                        "--require-promoted-opponent-pool-size",
                        "2",
                        "--promotion-artifact-dir",
                        "promoted-checkpoints",
                        "--promotion-label-prefix",
                        "candidate",
                        "--min-benchmark-win-rate",
                        "0.0",
                        "--min-benchmark-games",
                        "0",
                        "--max-collection-capped-rate",
                        "1.0",
                        "--validation-data",
                        "heldout-a.jsonl",
                        "--validation-data",
                        "heldout-b.jsonl",
                        "--evaluation-games",
                        "3",
                        "--audit-after-iteration",
                        "--audit-min-latest-benchmark-games",
                        "2",
                        "--audit-max-latest-average-decision-rounds",
                        "200",
                        "--audit-max-latest-benchmark-average-decision-rounds",
                        "210",
                        "--audit-max-latest-process-peak-rss-mb",
                        "2048",
                        "--audit-allow-missing-benchmark",
                        "--audit-allow-missing-benchmark-opponents",
                        "--audit-require-latest-promotion",
                        "--audit-failure-mode",
                        "runtime-health",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["iterations"], 1)
        self.assertTrue(kwargs["resume"])
        self.assertEqual(kwargs["games_per_iteration"], 2)
        self.assertEqual(kwargs["fixed_opponent_policy_specs"], ("random-legal",))
        self.assertEqual(kwargs["benchmark_reference_policy_specs"], ("linear:bootstrap.json",))
        self.assertEqual(kwargs["evaluation_games"], 3)
        self.assertEqual(kwargs["worker_count"], 2)
        self.assertEqual(kwargs["promotion_registry_path"], Path("promotions.json"))
        self.assertEqual(kwargs["required_promoted_opponent_pool_size"], 2)
        self.assertEqual(kwargs["auto_promotion_config"].registry_path, Path("promotions.json"))
        self.assertEqual(kwargs["auto_promotion_config"].artifact_dir, Path("promoted-checkpoints"))
        self.assertEqual(kwargs["auto_promotion_config"].label_prefix, "candidate")
        self.assertEqual(kwargs["auto_promotion_config"].gate_config.min_benchmark_win_rate, 0.0)
        self.assertEqual(kwargs["post_iteration_audit_config"].min_latest_benchmark_games, 2)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_latest_average_decision_rounds, 200.0)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_latest_benchmark_average_decision_rounds, 210.0)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_latest_process_peak_rss_mb, 2048.0)
        self.assertFalse(kwargs["post_iteration_audit_config"].require_benchmark)
        self.assertFalse(kwargs["post_iteration_audit_config"].require_benchmark_opponent_coverage)
        self.assertTrue(kwargs["post_iteration_audit_config"].require_latest_promotion)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_consecutive_promotion_failures, 3)
        self.assertEqual(kwargs["post_iteration_audit_config"].max_benchmark_win_rate_drop, 0.15)
        self.assertEqual(kwargs["post_iteration_audit_failure_mode"], "runtime-health")
        self.assertEqual(kwargs["validation_rollout_paths"], (Path("heldout-a.jsonl"), Path("heldout-b.jsonl")))
        self.assertEqual(kwargs["training_config"].objective, "reward-weighted")
        self.assertEqual(kwargs["training_config"].opponent_action_loss_weight, 0.3)
        self.assertEqual(kwargs["training_config"].capped_terminal_value, -0.25)
        self.assertIn("latest_checkpoint", stdout.getvalue())

    def test_selfplay_cli_iterate_uses_named_promotion_gate_profile(self) -> None:
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
        fake_iteration = SimpleNamespace(
            iteration=1,
            metrics=fake_metrics,
            training=SimpleNamespace(final_metrics=SimpleNamespace(loss=0.25, accuracy=0.75)),
            checkpoint_path=Path("run/iteration-0001/linear-policy.json"),
        )
        fake_result = SimpleNamespace(
            run_dir=Path("run"),
            iterations=(fake_iteration,),
            latest_checkpoint_path=Path("run/iteration-0001/linear-policy.json"),
        )
        with patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--auto-promote",
                        "--promotion-registry",
                        "promotions.json",
                        "--evaluation-games",
                        "1",
                        "--profile",
                        "smoke",
                    ]
                )

        self.assertEqual(exit_code, 0)
        gate_config = run.call_args.kwargs["auto_promotion_config"].gate_config
        self.assertEqual(gate_config.min_benchmark_games, 0)
        self.assertEqual(gate_config.min_benchmark_win_rate, 0.0)
        self.assertFalse(gate_config.require_benchmark)

    def test_selfplay_cli_iterate_uses_named_post_iteration_audit_profile(self) -> None:
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
        with (
            patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            exit_code = selfplay_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--evaluation-games",
                    "3",
                    "--audit-after-iteration",
                    "--audit-profile",
                    "long-run",
                    "--audit-min-latest-benchmark-games",
                    "7",
                    "--audit-allow-missing-benchmark",
                    "--audit-require-latest-promotion",
                ]
            )

        audit_config = run.call_args.kwargs["post_iteration_audit_config"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(audit_config.min_latest_benchmark_win_rate, 0.60)
        self.assertEqual(audit_config.min_latest_benchmark_games, 7)
        self.assertEqual(audit_config.max_latest_benchmark_capped_rate, 0.05)
        self.assertEqual(audit_config.max_benchmark_win_rate_drop, 0.03)
        self.assertFalse(audit_config.require_benchmark)
        self.assertTrue(audit_config.require_latest_promotion)
        self.assertTrue(audit_config.require_benchmark_opponent_coverage)

    def test_selfplay_cli_iterate_uses_post_iteration_audit_config_file(self) -> None:
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
        audit_defaults = RunAuditConfig(
            min_latest_benchmark_win_rate=0.72,
            min_latest_benchmark_games=8,
            require_benchmark=True,
            require_benchmark_opponent_coverage=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            config_path.write_text(json.dumps(run_audit_config_payload(audit_defaults), indent=2), encoding="utf-8")
            with (
                patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--evaluation-games",
                        "2",
                        "--audit-after-iteration",
                        "--audit-config",
                        str(config_path),
                        "--audit-min-latest-benchmark-games",
                        "4",
                        "--audit-allow-missing-benchmark-opponents",
                    ]
                )

        audit_config = run.call_args.kwargs["post_iteration_audit_config"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(audit_config.min_latest_benchmark_win_rate, 0.72)
        self.assertEqual(audit_config.min_latest_benchmark_games, 4)
        self.assertTrue(audit_config.require_benchmark)
        self.assertFalse(audit_config.require_benchmark_opponent_coverage)

    def test_selfplay_cli_iterate_rejects_post_iteration_audit_profile_with_config_file(self) -> None:
        audit_defaults = RunAuditConfig(
            min_latest_benchmark_win_rate=0.72,
            min_latest_benchmark_games=8,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            config_path.write_text(json.dumps(run_audit_config_payload(audit_defaults), indent=2), encoding="utf-8")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--evaluation-games",
                        "2",
                        "--audit-after-iteration",
                        "--audit-profile",
                        "smoke",
                        "--audit-config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--audit-profile cannot be combined with --audit-config", stderr.getvalue())

    def test_selfplay_cli_iterate_rejects_audit_config_without_post_iteration_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "audit-config.json"
            config_path.write_text(json.dumps(run_audit_config_payload(RunAuditConfig()), indent=2), encoding="utf-8")
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--audit-config",
                        str(config_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertIn("--audit-config requires --audit-after-iteration", stderr.getvalue())

    def test_selfplay_cli_iterate_rejects_audit_failure_mode_without_post_iteration_audit(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = selfplay_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--audit-failure-mode",
                    "runtime-health",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--audit-failure-mode requires --audit-after-iteration", stderr.getvalue())

    def test_selfplay_cli_iterate_profile_boolean_overrides(self) -> None:
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
        with (
            patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            exit_code = selfplay_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--evaluation-games",
                    "1",
                    "--audit-after-iteration",
                    "--audit-profile",
                    "smoke",
                    "--audit-require-benchmark",
                    "--audit-require-benchmark-opponents",
                    "--audit-allow-missing-latest-promotion",
                ]
            )

        audit_config = run.call_args.kwargs["post_iteration_audit_config"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(audit_config.min_latest_benchmark_games, 0)
        self.assertEqual(audit_config.max_benchmark_win_rate_drop, 1.0)
        self.assertTrue(audit_config.require_benchmark)
        self.assertTrue(audit_config.require_benchmark_opponent_coverage)
        self.assertFalse(audit_config.require_latest_promotion)

    def test_selfplay_cli_iterate_rejects_audit_profile_with_too_few_evaluation_games(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = selfplay_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--evaluation-games",
                    "3",
                    "--audit-after-iteration",
                    "--audit-profile",
                    "long-run",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("requires enough --evaluation-games", stderr.getvalue())

    def test_selfplay_cli_iterate_rejects_conflicting_post_iteration_audit_booleans(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as raised:
                selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--audit-after-iteration",
                        "--audit-require-benchmark",
                        "--audit-allow-missing-benchmark",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_selfplay_cli_iterate_rejects_audit_requiring_missing_benchmark(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = selfplay_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--audit-after-iteration",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--audit-after-iteration requires --evaluation-games", stderr.getvalue())

    def test_selfplay_cli_auto_promote_requires_evaluation_games_by_default(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = selfplay_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--auto-promote",
                    "--promotion-registry",
                    "promotions.json",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--auto-promote requires --evaluation-games", stderr.getvalue())

    def test_selfplay_cli_auto_promote_allows_missing_benchmark_without_evaluation_games(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--auto-promote",
                        "--promotion-registry",
                        "promotions.json",
                        "--allow-missing-benchmark",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertFalse(run.call_args.kwargs["auto_promotion_config"].gate_config.require_benchmark)

    def test_selfplay_cli_auto_promote_respects_smoke_profile_missing_benchmark_default(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.selfplay_cli.run_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = selfplay_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--auto-promote",
                        "--promotion-registry",
                        "promotions.json",
                        "--profile",
                        "smoke",
                    ]
                )

        self.assertEqual(exit_code, 0)
        gate_config = run.call_args.kwargs["auto_promotion_config"].gate_config
        self.assertFalse(gate_config.require_benchmark)
        self.assertEqual(gate_config.min_benchmark_games, 0)

    def test_selfplay_cli_report_prints_manifest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = report_source_metadata(branch="scott/report", head="abc123", dirty=True)
            write_report_manifest(run_dir, source=source)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("iterations: 1", output)
        self.assertIn("source_metadata:", output)
        self.assertIn("available: yes", output)
        self.assertIn("branch: scott/report", output)
        self.assertIn("head: abc123", output)
        self.assertIn("dirty: yes", output)
        self.assertIn("repo_root: /repo", output)
        self.assertIn("invocations: 1", output)
        self.assertIn("invocation=1", output)
        self.assertIn("resume=no", output)
        self.assertIn("first_iter=1", output)
        self.assertIn("requested_iters=1", output)
        self.assertIn("games_per_iter=3", output)
        self.assertIn("workers=2", output)
        self.assertIn("first_seed=20", output)
        self.assertIn("initial=random-legal", output)
        self.assertIn("eval_games=10", output)
        self.assertIn("fixed_opponents=1", output)
        self.assertIn("pool_registry=promotions.json", output)
        self.assertIn("required_pool=1", output)
        self.assertIn("promoted_available=2", output)
        self.assertIn("auto_promote=no", output)
        self.assertIn("latest_checkpoint:", output)
        self.assertIn("linear-policy.json", output)
        self.assertIn("0.600", output)
        self.assertIn("0.125000", output)
        self.assertIn("0.8750", output)
        self.assertIn("avg_dec", output)
        self.assertIn("peak_mb", output)
        self.assertIn("2.000", output)
        self.assertIn("77.500", output)
        self.assertIn(" val ", output)
        self.assertIn("fit metrics measure imitation", output)
        self.assertNotIn("0.250000", output)

    def test_selfplay_cli_report_can_print_json_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = report_source_metadata(branch="scott/json-report", head="abc777", dirty=False)
            write_report_manifest(run_dir, source=source)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir), "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(payload["source"], source)
        self.assertEqual(payload["iterations"][0]["iteration"], 1)
        self.assertNotIn("source_metadata:", stdout.getvalue())

    def test_selfplay_cli_report_reconstructs_from_iteration_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = report_source_metadata(branch="scott/reconstructed", head="def456", dirty=False)
            write_report_manifest(run_dir, top_level=False, source=source)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("iterations: 1", output)
        self.assertIn("branch: scott/reconstructed", output)
        self.assertIn("head: def456", output)
        self.assertIn("dirty: no", output)
        self.assertIn("invocations: 1", output)
        self.assertIn("pool_registry=promotions.json", output)
        self.assertIn("linear-policy.json", output)

    def test_selfplay_cli_report_reconstructs_without_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_report_manifest(run_dir, top_level=False)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        self.assertEqual(exit_code, 0)
        self.assertIn("iterations: 1", stdout.getvalue())
        self.assertIn("source_metadata: -", stdout.getvalue())

    def test_selfplay_cli_report_handles_missing_and_unavailable_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_run_dir = Path(temp_dir) / "missing"
            unavailable_run_dir = Path(temp_dir) / "unavailable"
            write_report_manifest(missing_run_dir)
            write_report_manifest(
                unavailable_run_dir,
                source={
                    "available": False,
                    "repo_root": None,
                    "branch": None,
                    "head": None,
                    "dirty": None,
                    "error": "RuntimeError: git unavailable",
                },
            )

            with patch("sys.stdout", new_callable=io.StringIO) as missing_stdout:
                missing_exit = selfplay_cli_main(["report", "--run-dir", str(missing_run_dir)])
            with patch("sys.stdout", new_callable=io.StringIO) as unavailable_stdout:
                unavailable_exit = selfplay_cli_main(["report", "--run-dir", str(unavailable_run_dir)])

        self.assertEqual(missing_exit, 0)
        self.assertIn("source_metadata: -", missing_stdout.getvalue())
        self.assertEqual(unavailable_exit, 0)
        self.assertIn("available: no", unavailable_stdout.getvalue())
        self.assertIn("error: RuntimeError: git unavailable", unavailable_stdout.getvalue())

    def test_selfplay_cli_report_prints_multiple_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            second_invocation = dict(manifest["invocation_configs"][0])
            second_invocation["resume"] = True
            second_invocation["first_iteration"] = 2
            second_invocation["iterations_requested"] = 2
            second_invocation["seed_start_argument"] = 1
            second_invocation["first_iteration_seed_start"] = 23
            manifest["invocation_configs"].append(second_invocation)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = selfplay_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("invocations: 2", output)
        self.assertIn("invocation=1 resume=no first_iter=1", output)
        self.assertIn("invocation=2 resume=yes first_iter=2", output)
        self.assertIn("first_seed=23", output)

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
        self.assertIn("promo", stdout.getvalue())


def normalized_record_payloads(path: Path) -> tuple[dict, ...]:
    payloads = []
    for record in read_rollout_records(path):
        payload = rollout_record_to_dict(record)
        payload["elapsed_seconds"] = 0.0
        payloads.append(payload)
    return tuple(payloads)


def write_promotion_registry(
    path: Path,
    *,
    checkpoint_paths: tuple[Path, ...],
    policy_ids: tuple[str, ...] | None = None,
) -> None:
    entry_policy_ids = policy_ids or tuple(f"linear-promoted-{index}" for index in range(1, len(checkpoint_paths) + 1))
    entries = [
        {
            "sequence": index,
            "policy_id": entry_policy_ids[index - 1],
            "checkpoint_path": str(checkpoint_path),
            "manifest_path": f"runs/promoted-{index}/manifest.json",
            "source_type": SELFPLAY_RUN_SCHEMA_VERSION,
            "source_iteration": index,
            "promoted_at": "2026-06-02T00:00:00Z",
            "label": None,
            "notes": None,
            "gate_result": {"passed": True},
        }
        for index, checkpoint_path in enumerate(checkpoint_paths, start=1)
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
                "registry_path": str(path),
                "latest_policy_id": entries[-1]["policy_id"] if entries else None,
                "latest_checkpoint_path": entries[-1]["checkpoint_path"] if entries else None,
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def passing_promotion_gate_config() -> PromotionGateConfig:
    return PromotionGateConfig(
        min_benchmark_win_rate=0.0,
        min_incumbent_win_rate=0.0,
        min_benchmark_games=0,
        min_incumbent_games=0,
        max_collection_capped_rate=1.0,
        max_benchmark_capped_rate=1.0,
        max_incumbent_capped_rate=1.0,
        min_incumbent_win_rate_lower_bound=0.0,
    )


def write_report_manifest(run_dir: Path, *, top_level: bool = True, source: dict | None = None) -> None:
    checkpoint_path = run_dir / "iteration-0001" / "linear-policy.json"
    iteration_dir = run_dir / "iteration-0001"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    invocation_config = {
        "resume": False,
        "first_iteration": 1,
        "iterations_requested": 1,
        "games_per_iteration": 3,
        "seed_start_argument": 20,
        "first_iteration_seed_start": 20,
        "initial_policy_spec": "random-legal",
        "evaluation_games": 10,
        "evaluation_seed_start": 1000,
        "worker_count": 2,
        "validation_rollout_paths": [],
        "benchmark_reference_policy_specs": [],
        "opponent_pool": {
            "fixed_opponent_policy_specs": ["random-legal"],
            "max_historical_opponents": 3,
            "historical_opponent_selection": "recent",
            "promotion_registry_path": "promotions.json",
            "promotion_pool_registry_path": "promotions.json",
            "required_promoted_opponent_pool_size": 1,
            "promoted_checkpoint_policy_specs": [
                "linear:runs/promoted-a/linear-policy.json",
                "linear:runs/promoted-b/linear-policy.json",
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
    iteration_manifest = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": 1,
        "rollout_path": str(iteration_dir / "rollouts.jsonl"),
        "training_rollout_path": str(iteration_dir / "training-rollouts.jsonl"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_policy_spec": f"linear:{checkpoint_path}",
        "current_policy_spec": "random-legal",
        "opponent_policy_specs": ["random-legal"],
        "opponent_pool_config": invocation_config["opponent_pool"],
        "invocation_config": invocation_config,
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
            "peak_rss_mb": 77.5,
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
    if source is not None:
        iteration_manifest["source"] = source
    (iteration_dir / "manifest.json").write_text(json.dumps(iteration_manifest, indent=2), encoding="utf-8")
    if not top_level:
        return
    run_manifest = {
        "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "invocation_configs": [invocation_config],
        "latest_checkpoint_path": str(checkpoint_path),
        "iterations": [iteration_manifest],
    }
    if source is not None:
        run_manifest["source"] = source
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def report_source_metadata(
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


if __name__ == "__main__":
    unittest.main()
