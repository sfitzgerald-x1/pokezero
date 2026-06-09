import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import BenchmarkReport, CollectionMetrics
from pokezero.neural_policy import (
    NEURAL_INSTALL_MESSAGE,
    TorchUnavailableError,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    torch_available,
)
from pokezero.neural_selfplay import (
    NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
from pokezero.rollout import RolloutConfig


class NeuralSelfPlayTest(unittest.TestCase):
    def test_run_neural_selfplay_iterations_requires_torch_before_collecting(self) -> None:
        if torch_available():
            self.skipTest("PyTorch is installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with self.assertRaises(TorchUnavailableError) as raised:
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=TransformerPolicyConfig(),
                    training_config=TransformerTrainingConfig(),
                )

            self.assertIn(NEURAL_INSTALL_MESSAGE, str(raised.exception))
            self.assertFalse((run_dir / "iteration-0001").exists())

    def test_run_neural_selfplay_iterations_writes_manifests_and_accumulates_training_data(self) -> None:
        collected = []
        trained_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(collected=collected, trained_paths=trained_paths):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=3,
                )

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(run_manifest["schema_version"], NEURAL_SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(first_manifest["checkpoint_policy_spec"], f"neural:{run_dir / 'iteration-0001' / 'transformer-policy.pt'}")
        self.assertEqual(second_manifest["current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(second_manifest["training_rollout_paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])
        self.assertEqual(run_manifest["latest_checkpoint_path"], str(run_dir / "iteration-0002" / "transformer-policy.pt"))
        self.assertEqual([call["seed_start"] for call in collected], [20, 22])
        self.assertEqual([call["worker_count"] for call in collected], [3, 3])
        self.assertEqual([tuple(path.name for path in paths) for paths in trained_paths], [
            ("training-rollouts.jsonl",),
            ("training-rollouts.jsonl", "training-rollouts.jsonl"),
        ])

    def test_run_neural_selfplay_iterations_benchmarks_checkpoint(self) -> None:
        captured_benchmark = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(captured_benchmark=captured_benchmark):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    evaluation_games=2,
                    evaluation_seed_start=100,
                )

        matchups = captured_benchmark["matchups"]
        self.assertEqual(captured_benchmark["games"], 2)
        self.assertEqual(captured_benchmark["seed_start"], 100)
        self.assertEqual([matchup.label for matchup in matchups], [
            "entity-test-iter-0001 vs random-legal",
            "random-legal vs entity-test-iter-0001",
            "entity-test-iter-0001 vs simple-legal",
            "simple-legal vs entity-test-iter-0001",
        ])


def patched_neural_selfplay_dependencies(
    *,
    collected: list | None = None,
    trained_paths: list | None = None,
    captured_benchmark: dict | None = None,
):
    collected = collected if collected is not None else []
    trained_paths = trained_paths if trained_paths is not None else []
    captured_benchmark = captured_benchmark if captured_benchmark is not None else {}

    def fake_collect_selfplay_rollouts(**kwargs):
        output_path = kwargs["output_path"]
        training_output_path = kwargs["training_output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        training_output_path.write_text("", encoding="utf-8")
        collected.append(kwargs)
        return CollectionMetrics(
            games=kwargs["games"],
            elapsed_seconds=1.0,
            total_decision_rounds=kwargs["games"],
            total_simulator_turns=kwargs["games"],
            p1_wins=kwargs["games"],
            p2_wins=0,
            ties=0,
            capped_games=0,
        )

    def fake_train_transformer_policy(paths, *, model_config, training_config):
        trained_paths.append(tuple(Path(path) for path in paths))
        result = TransformerTrainingResult(
            model_config=model_config,
            training_config=training_config,
            epochs=(
                TransformerEpochMetrics(
                    epoch=1,
                    examples=4,
                    loss=0.25,
                    policy_loss=0.2,
                    policy_accuracy=0.75,
                    value_loss=0.1,
                    opponent_loss=0.05,
                    opponent_accuracy=0.5,
                ),
            ),
        )
        return object(), result

    def fake_save_transformer_checkpoint(path, model, *, result):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")

    class FakePolicy:
        policy_id = "entity-test-iter-0001"

    def fake_benchmark_rollouts(**kwargs):
        captured_benchmark.update(kwargs)
        return BenchmarkReport(
            format_id=kwargs["rollout_config"].format_id,
            max_decision_rounds=kwargs["rollout_config"].max_decision_rounds,
            games_per_matchup=kwargs["games"],
            matchups=(),
        )

    return patch.multiple(
        "pokezero.neural_selfplay",
        require_torch=lambda: object(),
        collect_selfplay_rollouts=fake_collect_selfplay_rollouts,
        train_transformer_policy=fake_train_transformer_policy,
        save_transformer_checkpoint=fake_save_transformer_checkpoint,
        load_transformer_policy=lambda *args, **kwargs: FakePolicy(),
        benchmark_rollouts=fake_benchmark_rollouts,
    )


if __name__ == "__main__":
    unittest.main()
