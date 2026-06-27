import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import BenchmarkMatchupResult, BenchmarkReport, CollectionMetrics
from pokezero.env import StepResult, TerminalState
from pokezero.neural_policy import (
    NEURAL_INSTALL_MESSAGE,
    TorchUnavailableError,
    TransformerEpochMetrics,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    torch_available,
)
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.neural_selfplay import (
    NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
    NeuralSelfPlayPromotionConfig,
    NeuralValueCalibrationConfig,
    NeuralValueSelectionConfig,
    _promoted_checkpoint_specs,
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
from pokezero.neural_cli import _print_iterate_summary, main as neural_cli_main
from pokezero.evaluation import PromotionGateConfig
from pokezero.promotion import PROMOTION_REGISTRY_SCHEMA_VERSION, load_promotion_registry
from pokezero.run_audit import RunAuditConfig, RunAuditFailure
from pokezero.rollout import RolloutConfig


def _entity_test_model_config(**overrides):
    """Small compact-vocab model config for self-play tests (legacy hash embedding retired)."""
    params = dict(policy_id="entity-test", embedding_dim=16, attention_heads=4)
    params.update(overrides)
    return TransformerPolicyConfig.compact_category(
        category_vocab=tuple(range(1, 65)), category_oov_buckets=8, **params
    )


class NeuralSelfPlayTest(unittest.TestCase):
    def setUp(self) -> None:
        # Self-play builds the string->row CategoryVocabulary from --showdown-root; stub it so
        # CLI tests stay fast and do not need a real Showdown checkout.
        from pokezero.category_vocab import build_category_vocabulary

        fake_vocab = build_category_vocabulary(["species:a", "species:b", "move:c"], oov_buckets=16)
        vocab_patch = patch("pokezero.randbat_vocab.gen3_category_vocabulary", return_value=fake_vocab)
        vocab_patch.start()
        self.addCleanup(vocab_patch.stop)

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
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(),
                )

            self.assertIn(NEURAL_INSTALL_MESSAGE, str(raised.exception))
            self.assertFalse((run_dir / "iteration-0001").exists())

    def test_run_neural_selfplay_iterations_rejects_blind_multi_iteration_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_neural_selfplay_dependencies():
                with self.assertRaisesRegex(ValueError, "evaluation_games"):
                    run_neural_selfplay_iterations(
                        run_dir=Path(temp_dir) / "run",
                        iterations=2,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    )

    def test_neural_cli_auto_promote_requires_evaluation_games_by_default(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--initial-policy",
                    "random-legal",
                    "--auto-promote",
                    "--promotion-registry",
                    "promotions.json",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--auto-promote requires --evaluation-games", stderr.getvalue())

    def test_neural_cli_auto_promote_allows_missing_benchmark_without_evaluation_games(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = neural_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--initial-policy",
                        "random-legal",
                        "--auto-promote",
                        "--promotion-registry",
                        "promotions.json",
                        "--allow-missing-benchmark",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertFalse(run.call_args.kwargs["auto_promotion_config"].gate_config.require_benchmark)

    def test_neural_cli_iterate_wires_post_iteration_audit_failure_mode(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = neural_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--initial-policy",
                        "random-legal",
                        "--audit-after-iteration",
                        "--audit-allow-missing-benchmark",
                        "--audit-allow-missing-benchmark-opponents",
                        "--audit-failure-mode",
                        "runtime-health",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_args.kwargs["post_iteration_audit_failure_mode"], "runtime-health")

    def test_neural_cli_iterate_wires_collector_advancement_mode(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = neural_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--initial-policy",
                        "random-legal",
                        "--collector-advancement-mode",
                        "always",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_args.kwargs["collector_advancement_mode"], "always")

    def test_neural_cli_iterate_rejects_always_advance_with_auto_promote(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = neural_cli_main(
                [
                    "iterate",
                    "--run-dir",
                    "run",
                    "--iterations",
                    "1",
                    "--games-per-iteration",
                    "2",
                    "--showdown-root",
                    "/tmp/showdown",
                    "--initial-policy",
                    "random-legal",
                    "--auto-promote",
                    "--promotion-registry",
                    "promotions.json",
                    "--collector-advancement-mode",
                    "always",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("cannot be combined with --auto-promote", stderr.getvalue())

    def test_neural_cli_iterate_wires_value_selection_config(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = neural_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--initial-policy",
                        "random-legal",
                        "--value-selection",
                        "--value-selection-scope",
                        "history",
                        "--value-selection-metric",
                        "expected_calibration_error",
                        "--value-selection-batch-size",
                        "9",
                        "--value-selection-bins",
                        "6",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("not held-out validation", stderr.getvalue())
        self.assertIn("can become expensive", stderr.getvalue())
        self.assertEqual(
            run.call_args.kwargs["value_selection_config"],
            NeuralValueSelectionConfig(
                scope="history",
                metric="expected_calibration_error",
                batch_size=9,
                bins=6,
            ),
        )

    def test_neural_cli_iterate_wires_heldout_value_selection_config(self) -> None:
        fake_result = SimpleNamespace(run_dir=Path("run"), iterations=(), latest_checkpoint_path=None)
        with patch("pokezero.neural_cli.run_neural_selfplay_iterations", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                exit_code = neural_cli_main(
                    [
                        "iterate",
                        "--run-dir",
                        "run",
                        "--iterations",
                        "1",
                        "--games-per-iteration",
                        "2",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--initial-policy",
                        "random-legal",
                        "--value-selection-heldout-games",
                        "4",
                        "--value-selection-seed-start",
                        "3000",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("implies --value-selection", stderr.getvalue())
        self.assertNotIn("not held-out validation", stderr.getvalue())
        self.assertEqual(
            run.call_args.kwargs["value_selection_config"],
            NeuralValueSelectionConfig(
                heldout_games_per_iteration=4,
                heldout_seed_start=3000,
            ),
        )

    def test_run_neural_selfplay_iterations_writes_manifests_and_accumulates_supervised_training_data(self) -> None:
        collected = []
        trained_paths = []
        trained_initial_models = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = {
                "available": True,
                "repo_root": "/repo",
                "branch": "scott/source-test",
                "head": "abc123",
                "dirty": False,
            }

            with (
                patched_neural_selfplay_dependencies(
                    collected=collected,
                    trained_paths=trained_paths,
                    trained_initial_models=trained_initial_models,
                ),
                patch("pokezero.neural_selfplay.collect_source_metadata", return_value=source),
            ):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=3,
                    evaluation_games=1,
                )

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(run_manifest["schema_version"], NEURAL_SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(run_manifest["source"], source)
        self.assertEqual(first_manifest["checkpoint_policy_spec"], f"neural:{run_dir / 'iteration-0001' / 'transformer-policy.pt'}")
        self.assertEqual(first_manifest["source"], source)
        self.assertEqual(first_manifest["invocation_config"]["source"], source)
        self.assertEqual(run_manifest["invocation_configs"][0]["source"], source)
        self.assertEqual(first_manifest["advancement"]["reason"], "beat_incumbent")
        self.assertEqual(first_manifest["next_current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(second_manifest["current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(second_manifest["advancement"]["incumbent_policy_id"], "entity-test-iter-0001")
        self.assertEqual(second_manifest["training_rollout_paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])
        self.assertEqual(second_manifest["training_input_paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])
        self.assertEqual(run_manifest["latest_checkpoint_path"], str(run_dir / "iteration-0002" / "transformer-policy.pt"))
        self.assertEqual(run_manifest["current_policy_spec"], second_manifest["checkpoint_policy_spec"])
        self.assertEqual(run_manifest["latest_accepted_checkpoint_path"], str(run_dir / "iteration-0002" / "transformer-policy.pt"))
        self.assertEqual([call["seed_start"] for call in collected], [20, 22])
        self.assertEqual([call["worker_count"] for call in collected], [3, 3])
        self.assertEqual([tuple(path.name for path in paths) for paths in trained_paths], [
            ("training-rollouts.jsonl",),
            ("training-rollouts.jsonl", "training-rollouts.jsonl"),
        ])
        self.assertEqual(trained_initial_models, [None, "entity-test-iter-0001"])

    def test_run_neural_selfplay_iterations_uses_iteration_only_training_data_for_ppo(self) -> None:
        trained_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(trained_paths=trained_paths):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(
                        window_size=4,
                        epochs=1,
                        batch_size=2,
                        objective="ppo",
                    ),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=3,
                    evaluation_games=1,
                )

            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(
            [[path.parent.name for path in paths] for paths in trained_paths],
            [["iteration-0001"], ["iteration-0002"]],
        )
        self.assertEqual(second_manifest["training_rollout_paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])
        self.assertEqual(second_manifest["training_input_paths"], [
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])

    def test_run_neural_selfplay_iterations_keeps_ppo_training_iteration_only_with_value_selection(self) -> None:
        trained_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(trained_paths=trained_paths):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(
                        window_size=4,
                        epochs=1,
                        batch_size=2,
                        objective="ppo",
                    ),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=3,
                    evaluation_games=1,
                    value_selection_config=NeuralValueSelectionConfig(scope="history"),
                )

            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(
            [[path.parent.name for path in paths] for paths in trained_paths],
            [["iteration-0001"], ["iteration-0002"]],
        )
        self.assertEqual(second_manifest["training_input_paths"], [
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])
        self.assertEqual(result.iterations[1].value_selection["paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])

    def test_run_neural_selfplay_iterations_records_value_calibration(self) -> None:
        captured_calibrations = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(captured_calibrations=captured_calibrations):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2, device="cpu"),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=1,
                    value_calibration_config=NeuralValueCalibrationConfig(scope="iteration", batch_size=7, bins=5),
                )

            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertEqual(len(captured_calibrations), 1)
        self.assertEqual(captured_calibrations[0]["paths"], (run_dir / "iteration-0001" / "training-rollouts.jsonl",))
        self.assertEqual(captured_calibrations[0]["batch_size"], 7)
        self.assertEqual(captured_calibrations[0]["bins"], 5)
        self.assertEqual(captured_calibrations[0]["device"], "cpu")
        calibration = result.iterations[0].value_calibration
        self.assertIsNotNone(calibration)
        self.assertEqual(calibration["scope"], "iteration")
        self.assertEqual(calibration["paths"], [str(run_dir / "iteration-0001" / "training-rollouts.jsonl")])
        self.assertEqual(calibration["report"]["sign_accuracy"], 0.75)
        self.assertEqual(iteration_manifest["value_calibration"], calibration)
        self.assertEqual(run_manifest["iterations"][0]["value_calibration"], calibration)
        self.assertEqual(iteration_manifest["invocation_config"]["value_calibration"]["scope"], "iteration")

    def test_run_neural_selfplay_iterations_selects_best_value_epoch(self) -> None:
        class FakeReport:
            def __init__(self, *, mae: float) -> None:
                self.examples = 4
                self.mse = mae * mae
                self.mae = mae
                self.bias = 0.0
                self.sign_accuracy = 0.5
                self.expected_calibration_error = mae / 2.0

            def to_dict(self) -> dict:
                return {
                    "examples": self.examples,
                    "mse": self.mse,
                    "mae": self.mae,
                    "bias": self.bias,
                    "sign_accuracy": self.sign_accuracy,
                    "expected_calibration_error": self.expected_calibration_error,
                    "bins": [],
                    "slices": [],
                }

        reports = [FakeReport(mae=0.4), FakeReport(mae=0.2), FakeReport(mae=0.3)]

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with (
                patched_neural_selfplay_dependencies(),
                patch("pokezero.neural_selfplay.evaluate_value_calibration", side_effect=reports) as evaluate,
            ):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=3, batch_size=2, device="cpu"),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=1,
                    value_selection_config=NeuralValueSelectionConfig(
                        scope="iteration",
                        metric="mae",
                        batch_size=7,
                        bins=5,
                    ),
                )

            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            sidecar_path = run_dir / "iteration-0001" / "value-selection.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(evaluate.call_count, 3)
        for call in evaluate.call_args_list:
            self.assertEqual(call.kwargs["paths"], (run_dir / "iteration-0001" / "training-rollouts.jsonl",))
            self.assertEqual(call.kwargs["batch_size"], 7)
            self.assertEqual(call.kwargs["bins"], 5)
            self.assertEqual(call.kwargs["device"], "cpu")
        training = result.iterations[0].training
        self.assertEqual(training.training_config.epochs, 2)
        self.assertEqual(training.final_metrics.epoch, 2)
        selection = result.iterations[0].value_selection
        self.assertIsNotNone(selection)
        self.assertEqual(selection["scope"], "iteration")
        self.assertEqual(selection["paths"], [str(run_dir / "iteration-0001" / "training-rollouts.jsonl")])
        self.assertEqual(selection["data_role"], "training_rollouts")
        self.assertIn("not held-out validation", selection["data_note"])
        self.assertEqual(selection["metric"], "mae")
        self.assertEqual(selection["metric_direction"], "min")
        self.assertEqual(selection["selected_epoch"], 2)
        self.assertEqual(selection["selected_metric_value"], 0.2)
        self.assertEqual(selection["artifact_path"], str(sidecar_path))
        self.assertEqual(sidecar["selected_epoch"], 2)
        self.assertEqual(sidecar["data_role"], "training_rollouts")
        self.assertEqual(len(sidecar["epochs"]), 3)
        self.assertEqual(sidecar["epochs"][1]["metric_value"], 0.2)
        self.assertEqual(iteration_manifest["value_selection"], selection)
        self.assertEqual(run_manifest["iterations"][0]["value_selection"], selection)
        self.assertEqual(iteration_manifest["training"]["config"]["epochs"], 2)
        self.assertEqual(iteration_manifest["invocation_config"]["value_selection"]["scope"], "iteration")

    def test_run_neural_selfplay_iterations_uses_heldout_value_selection_rollouts(self) -> None:
        class FakeReport:
            examples = 4
            mse = 0.04
            mae = 0.2
            bias = 0.0
            sign_accuracy = 0.75
            expected_calibration_error = 0.1

            def to_dict(self) -> dict:
                return {
                    "examples": self.examples,
                    "mse": self.mse,
                    "mae": self.mae,
                    "bias": self.bias,
                    "sign_accuracy": self.sign_accuracy,
                    "expected_calibration_error": self.expected_calibration_error,
                    "bins": [],
                    "slices": [],
                }

        collected = []
        trained_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with (
                patched_neural_selfplay_dependencies(collected=collected, trained_paths=trained_paths),
                patch("pokezero.neural_selfplay.evaluate_value_calibration", return_value=FakeReport()) as evaluate,
            ):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2, device="cpu"),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    worker_count=1,
                    value_selection_config=NeuralValueSelectionConfig(
                        heldout_games_per_iteration=3,
                        heldout_seed_start=9000,
                    ),
                )

            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            sidecar = json.loads((run_dir / "iteration-0001" / "value-selection.json").read_text(encoding="utf-8"))

        self.assertEqual([call["output_path"].name for call in collected], [
            "rollouts.jsonl",
            "value-selection-rollouts.jsonl",
        ])
        self.assertEqual(collected[1]["training_output_path"].name, "value-selection-training-rollouts.jsonl")
        self.assertEqual(collected[1]["games"], 3)
        self.assertEqual(collected[1]["seed_start"], 9000)
        self.assertEqual(trained_paths, [(run_dir / "iteration-0001" / "training-rollouts.jsonl",)])
        self.assertEqual(evaluate.call_args.kwargs["paths"], (
            run_dir / "iteration-0001" / "value-selection-training-rollouts.jsonl",
        ))
        selection = result.iterations[0].value_selection
        self.assertIsNotNone(selection)
        self.assertEqual(selection["data_role"], "heldout_selfplay_rollouts")
        self.assertIn("held-out self-play rollouts", selection["data_note"])
        self.assertEqual(selection["paths"], [
            str(run_dir / "iteration-0001" / "value-selection-training-rollouts.jsonl")
        ])
        self.assertEqual(sidecar["data_role"], "heldout_selfplay_rollouts")
        self.assertEqual(iteration_manifest["value_selection_collection_metrics"]["games"], 3)
        self.assertEqual(iteration_manifest["value_selection_seed_start"], 9000)
        self.assertEqual(iteration_manifest["value_selection_next_seed_start"], 9003)
        self.assertEqual(
            iteration_manifest["value_selection_rollout_path"],
            str(run_dir / "iteration-0001" / "value-selection-rollouts.jsonl"),
        )
        self.assertEqual(
            iteration_manifest["value_selection_training_rollout_paths"],
            [str(run_dir / "iteration-0001" / "value-selection-training-rollouts.jsonl")],
        )
        self.assertEqual(run_manifest["iterations"][0]["value_selection"], selection)

    def test_heldout_value_selection_history_scope_accumulates_and_resumes_seed_cursor(self) -> None:
        class FakeReport:
            examples = 4
            mse = 0.04
            mae = 0.2
            bias = 0.0
            sign_accuracy = 0.75
            expected_calibration_error = 0.1

            def to_dict(self) -> dict:
                return {
                    "examples": self.examples,
                    "mse": self.mse,
                    "mae": self.mae,
                    "bias": self.bias,
                    "sign_accuracy": self.sign_accuracy,
                    "expected_calibration_error": self.expected_calibration_error,
                    "bins": [],
                    "slices": [],
                }

        collected = []
        trained_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with (
                patched_neural_selfplay_dependencies(collected=collected, trained_paths=trained_paths),
                patch(
                    "pokezero.neural_selfplay.evaluate_value_calibration",
                    side_effect=[FakeReport(), FakeReport(), FakeReport()],
                ) as evaluate,
            ):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2, device="cpu"),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    value_selection_config=NeuralValueSelectionConfig(
                        scope="history",
                        heldout_games_per_iteration=2,
                        heldout_seed_start=5000,
                    ),
                )
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2, device="cpu"),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    value_selection_config=NeuralValueSelectionConfig(
                        scope="history",
                        heldout_games_per_iteration=1,
                        heldout_seed_start=5000,
                    ),
                    resume=True,
                )

            third_manifest = json.loads((run_dir / "iteration-0003" / "manifest.json").read_text(encoding="utf-8"))

        heldout_calls = [call for call in collected if call["output_path"].name == "value-selection-rollouts.jsonl"]
        self.assertEqual([call["seed_start"] for call in heldout_calls], [5000, 5002, 5004])
        self.assertEqual(evaluate.call_args_list[0].kwargs["paths"], (
            run_dir / "iteration-0001" / "value-selection-training-rollouts.jsonl",
        ))
        self.assertEqual(evaluate.call_args_list[1].kwargs["paths"], (
            run_dir / "iteration-0001" / "value-selection-training-rollouts.jsonl",
            run_dir / "iteration-0002" / "value-selection-training-rollouts.jsonl",
        ))
        self.assertEqual(evaluate.call_args_list[2].kwargs["paths"], (
            run_dir / "iteration-0001" / "value-selection-training-rollouts.jsonl",
            run_dir / "iteration-0002" / "value-selection-training-rollouts.jsonl",
            run_dir / "iteration-0003" / "value-selection-training-rollouts.jsonl",
        ))
        self.assertEqual(trained_paths[-1], (
            run_dir / "iteration-0001" / "training-rollouts.jsonl",
            run_dir / "iteration-0002" / "training-rollouts.jsonl",
            run_dir / "iteration-0003" / "training-rollouts.jsonl",
        ))
        self.assertNotIn("value-selection-training-rollouts.jsonl", {path.name for path in trained_paths[-1]})
        self.assertEqual(third_manifest["value_selection_seed_start"], 5004)
        self.assertEqual(third_manifest["value_selection_next_seed_start"], 5005)

    def test_run_neural_selfplay_iterations_value_selection_history_scope_uses_accumulated_paths(self) -> None:
        class FakeReport:
            def __init__(self, *, sign_accuracy: float) -> None:
                self.examples = 4
                self.mse = 0.25
                self.mae = 0.5
                self.bias = 0.0
                self.sign_accuracy = sign_accuracy
                self.expected_calibration_error = 0.2

            def to_dict(self) -> dict:
                return {
                    "examples": self.examples,
                    "mse": self.mse,
                    "mae": self.mae,
                    "bias": self.bias,
                    "sign_accuracy": self.sign_accuracy,
                    "expected_calibration_error": self.expected_calibration_error,
                    "bins": [],
                    "slices": [],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with (
                patched_neural_selfplay_dependencies(),
                patch(
                    "pokezero.neural_selfplay.evaluate_value_calibration",
                    side_effect=[FakeReport(sign_accuracy=0.4), FakeReport(sign_accuracy=0.8)],
                ) as evaluate,
            ):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2, device="cpu"),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    value_selection_config=NeuralValueSelectionConfig(
                        scope="history",
                        metric="sign_accuracy",
                        batch_size=3,
                        bins=4,
                    ),
                )

            second_sidecar = json.loads((run_dir / "iteration-0002" / "value-selection.json").read_text(encoding="utf-8"))

        self.assertEqual(evaluate.call_args_list[0].kwargs["paths"], (run_dir / "iteration-0001" / "training-rollouts.jsonl",))
        self.assertEqual(evaluate.call_args_list[1].kwargs["paths"], (
            run_dir / "iteration-0001" / "training-rollouts.jsonl",
            run_dir / "iteration-0002" / "training-rollouts.jsonl",
        ))
        second_selection = result.iterations[1].value_selection
        self.assertIsNotNone(second_selection)
        self.assertEqual(second_selection["scope"], "history")
        self.assertEqual(second_selection["metric"], "sign_accuracy")
        self.assertEqual(second_selection["metric_direction"], "max")
        self.assertEqual(second_selection["selected_metric_value"], 0.8)
        self.assertEqual(second_sidecar["paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])

    def test_run_neural_selfplay_iterations_benchmarks_checkpoint(self) -> None:
        captured_benchmarks = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(captured_benchmarks=captured_benchmarks):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    evaluation_games=2,
                    evaluation_seed_start=100,
                )

        first_matchups = captured_benchmarks[0]["matchups"]
        second_matchups = captured_benchmarks[1]["matchups"]
        self.assertEqual(captured_benchmarks[0]["games"], 2)
        self.assertEqual(captured_benchmarks[0]["seed_start"], 100)
        self.assertEqual([matchup.label for matchup in first_matchups], [
            "entity-test-iter-0001 vs random-legal",
            "random-legal vs entity-test-iter-0001",
            "entity-test-iter-0001 vs simple-legal",
            "simple-legal vs entity-test-iter-0001",
        ])
        self.assertIn("entity-test-iter-0002 vs entity-test-iter-0001", [matchup.label for matchup in second_matchups])
        self.assertIn("entity-test-iter-0001 vs entity-test-iter-0002", [matchup.label for matchup in second_matchups])

    def test_run_neural_selfplay_iterations_benchmarks_eval_only_reference(self) -> None:
        captured_benchmarks: list = []
        collected: list = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            reference_spec = f"neural:{Path(temp_dir) / 'reference.pt'}"

            with patched_neural_selfplay_dependencies(
                collected=collected, captured_benchmarks=captured_benchmarks
            ):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    # Duplicate reference must collapse to a single spec.
                    benchmark_reference_policy_specs=(reference_spec, reference_spec),
                    evaluation_games=2,
                    evaluation_seed_start=100,
                )

        labels = [matchup.label for matchup in captured_benchmarks[0]["matchups"]]
        # The eval-only reference is benchmarked in both orientations (and only once).
        self.assertEqual(labels.count("entity-test-iter-0001 vs entity-test"), 1)
        self.assertEqual(labels.count("entity-test vs entity-test-iter-0001"), 1)
        # ...but it never enters rollout collection as a training opponent.
        self.assertNotIn(reference_spec, collected[0]["opponent_policy_specs"])
        # ...and is recorded TOP-LEVEL in the iteration manifest (deduped) so the promotion
        # gate can identify and exclude it; invocation_config carries it too.
        iteration_manifest = result.iterations[0].to_manifest_dict()
        self.assertEqual(iteration_manifest["benchmark_reference_policy_specs"], [reference_spec])
        self.assertEqual(
            result.invocation_config["benchmark_reference_policy_specs"], [reference_spec]
        )
        self.assertEqual(result.iterations[0].benchmark.games_per_matchup, 2)

    def test_with_collection_temperature_injects_only_for_checkpoint_specs(self) -> None:
        from urllib.parse import parse_qsl

        from pokezero.collection import policy_factory_from_spec
        from pokezero.neural_selfplay import _with_collection_temperature

        # No-op at temperature 1.0.
        self.assertEqual(_with_collection_temperature("neural:/m.pt", 1.0), "neural:/m.pt")
        # Non-checkpoint specs are unchanged (temperature is meaningless there).
        self.assertEqual(_with_collection_temperature("simple-legal", 1.5), "simple-legal")
        # Neural spec gets a sampling temperature and is set to sample.
        spec = _with_collection_temperature("neural:/m.pt", 1.5)
        body, _, query = spec.partition("?")
        params = dict(parse_qsl(query))
        self.assertEqual(body, "neural:/m.pt")
        self.assertEqual(float(params["temperature"]), 1.5)
        self.assertEqual(params["sample"], "true")
        self.assertNotIn("deterministic", params)

    def test_with_collection_temperature_normalizes_and_round_trips(self) -> None:
        from pokezero.collection import _split_policy_spec_options, policy_factory_from_spec
        from pokezero.neural_selfplay import _with_collection_temperature

        # A pre-existing deterministic option (any case) must be removed, not left to collide with
        # the injected sample=true; other options (epsilon) are preserved.
        spec = _with_collection_temperature("neural:/tmp/m.pt?Deterministic=true&epsilon=0.1", 1.25)
        from pokezero.collection import _split_policy_spec_options as _canonical_split
        _, options = _canonical_split(spec)
        self.assertNotIn("deterministic", options)
        self.assertEqual(options["sample"], "true")
        self.assertEqual(float(options["temperature"]), 1.25)
        self.assertEqual(options["epsilon"], "0.1")
        # Must round-trip through the real resolver without a sample/deterministic conflict
        # (the prior implementation raised here). neural specs build the factory lazily, so no
        # checkpoint file is needed to validate option parsing.
        self.assertTrue(callable(policy_factory_from_spec(spec)))
        # Duplicate normalized option keys are rejected (same as the canonical resolver), rather
        # than silently collapsed.
        with self.assertRaises(ValueError):
            _with_collection_temperature("neural:/m.pt?Sample=true&sample=false", 1.5)

    def test_collection_temperature_keeps_canonical_spec_clean_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patched_neural_selfplay_dependencies():
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    initial_policy_spec="neural:/tmp/bootstrap.pt",
                    fixed_opponent_policy_specs=("simple-legal",),
                    collection_temperature=2.0,
                )
        # The temperature is collection-only: the canonical recorded specs stay clean.
        manifest = result.iterations[0].to_manifest_dict()
        self.assertNotIn("temperature", manifest["current_policy_spec"])
        self.assertNotIn("temperature", manifest["next_current_policy_spec"])

    def test_collection_temperature_applies_to_collector_spec(self) -> None:
        collected: list = []
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patched_neural_selfplay_dependencies(collected=collected):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    initial_policy_spec="neural:/tmp/bootstrap.pt",
                    fixed_opponent_policy_specs=("simple-legal",),
                    collection_temperature=1.5,
                )
        # The collector spec passed to collection carries the exploration temperature.
        self.assertIn("temperature=1.5", collected[0]["current_policy_spec"])

    def test_collection_temperature_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_neural_selfplay_dependencies():
                with self.assertRaises(ValueError):
                    run_neural_selfplay_iterations(
                        run_dir=Path(temp_dir) / "run",
                        iterations=1,
                        games_per_iteration=2,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                        collection_temperature=0.0,
                    )

    def test_mirror_match_adds_current_policy_to_collection_opponents(self) -> None:
        collected: list = []
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patched_neural_selfplay_dependencies(collected=collected):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    initial_policy_spec="neural:/tmp/bootstrap.pt",
                    fixed_opponent_policy_specs=("simple-legal",),
                    mirror_match=True,
                )
        # Iteration 1 collection includes the current policy as an opponent (mirror match),
        # so self-play happens from the start rather than only after a promotion.
        self.assertIn("neural:/tmp/bootstrap.pt", collected[0]["opponent_policy_specs"])

    def test_spread_historical_selection_uses_older_and_recent_checkpoints(self) -> None:
        collected: list = []
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patched_neural_selfplay_dependencies(collected=collected):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=6,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    initial_policy_spec="neural:/tmp/bootstrap.pt",
                    fixed_opponent_policy_specs=("simple-legal",),
                    max_historical_opponents=2,
                    historical_opponent_selection="spread",
                    evaluation_games=1,
                    collector_advancement_mode="always",
                )
            iteration_manifest = json.loads((run_dir / "iteration-0006" / "manifest.json").read_text(encoding="utf-8"))

            first_checkpoint = f"neural:{run_dir / 'iteration-0001' / 'transformer-policy.pt'}"
            fourth_checkpoint = f"neural:{run_dir / 'iteration-0004' / 'transformer-policy.pt'}"
            fifth_checkpoint = f"neural:{run_dir / 'iteration-0005' / 'transformer-policy.pt'}"
            self.assertEqual(
                collected[5]["opponent_policy_specs"],
                ("simple-legal", first_checkpoint, fourth_checkpoint),
            )
            self.assertNotIn(fifth_checkpoint, collected[5]["opponent_policy_specs"])
            self.assertEqual(iteration_manifest["opponent_pool_config"]["historical_opponent_selection"], "spread")

    def test_tensorboard_scalars_flattens_training_and_benchmark(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_selfplay import _tensorboard_scalars

        candidate = "cand-iter-0002"
        epoch = TransformerEpochMetrics(
            epoch=1,
            examples=10,
            loss=0.5,
            policy_loss=0.4,
            policy_accuracy=0.6,
            value_loss=0.1,
            opponent_loss=0.05,
            opponent_accuracy=0.5,
            ppo_valid_examples=8,
            ppo_valid_fraction=0.8,
            ppo_advantage_mean=0.2,
            ppo_advantage_std=0.4,
            ppo_ratio_mean=1.1,
            ppo_clip_fraction=0.25,
            ppo_entropy=1.7,
        )

        def matchup(label, p1, p2, p1_wins, games=10):
            return BenchmarkMatchupResult(
                label=label,
                p1_policy_id=p1,
                p2_policy_id=p2,
                seed_start=1,
                metrics=CollectionMetrics(
                    games=games,
                    elapsed_seconds=1.0,
                    total_decision_rounds=games,
                    total_simulator_turns=games,
                    p1_wins=p1_wins,
                    p2_wins=games - p1_wins,
                    ties=0,
                    capped_games=0,
                ),
            )

        benchmark = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=5,
            games_per_matchup=10,
            matchups=(
                # candidate wins 2/10 as p1 and 3/10 as p2 -> combined 5/20 = 0.25
                matchup(f"{candidate} vs max-damage", candidate, "max-damage", 2),
                matchup(f"max-damage vs {candidate}", "max-damage", candidate, 7),
            ),
        )
        scalars = _tensorboard_scalars(
            candidate_policy_id=candidate,
            training=SimpleNamespace(epochs=(epoch,)),
            benchmark=benchmark,
            advancement=SimpleNamespace(advance_collector=True),
        )
        self.assertEqual(scalars["train/loss"], 0.5)
        self.assertEqual(scalars["train/policy_accuracy"], 0.6)
        self.assertEqual(scalars["train/value_loss"], 0.1)
        self.assertEqual(scalars["ppo/valid_fraction"], 0.8)
        self.assertEqual(scalars["ppo/advantage_mean"], 0.2)
        self.assertEqual(scalars["ppo/advantage_std"], 0.4)
        self.assertEqual(scalars["ppo/ratio_mean"], 1.1)
        self.assertEqual(scalars["ppo/clip_fraction"], 0.25)
        self.assertEqual(scalars["ppo/entropy"], 1.7)
        self.assertAlmostEqual(scalars["winrate/max-damage"], 0.25)
        self.assertEqual(scalars["train/advanced"], 1.0)

    def test_tensorboard_logger_closed_when_iteration_raises(self) -> None:
        closed: list[bool] = []

        class FakeLogger:
            def __init__(self, log_dir):
                self.open = True
                self._instance = self

            def log(self, scalars, *, step):
                pass

            def close(self):
                self.open = False
                closed.append(True)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patched_neural_selfplay_dependencies():
                with patch("pokezero.neural_selfplay._TensorBoardLogger", FakeLogger), patch(
                    "pokezero.neural_selfplay.collect_selfplay_rollouts",
                    side_effect=RuntimeError("boom"),
                ):
                    with self.assertRaises(RuntimeError):
                        run_neural_selfplay_iterations(
                            run_dir=run_dir,
                            iterations=1,
                            games_per_iteration=1,
                            env_factory=lambda: None,  # type: ignore[return-value]
                            rollout_config=RolloutConfig(max_decision_rounds=5),
                            model_config=_entity_test_model_config(),
                            training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                            tensorboard_log_dir=run_dir / "tb",
                        )

        # The SummaryWriter must be closed even though the iteration raised.
        self.assertEqual(closed, [True])

    def test_run_neural_selfplay_iterations_does_not_advance_failed_candidate(self) -> None:
        collected = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(collected=collected, candidate_beats_incumbent=False):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                )

            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertEqual([call["current_policy_spec"] for call in collected], ["random-legal", "random-legal"])
        self.assertFalse(first_manifest["advancement"]["advance_collector"])
        self.assertEqual(first_manifest["advancement"]["reason"], "failed_to_beat_incumbent")
        self.assertEqual(first_manifest["next_current_policy_spec"], "random-legal")
        self.assertEqual(second_manifest["current_policy_spec"], "random-legal")
        self.assertEqual(run_manifest["current_policy_spec"], "random-legal")
        self.assertIsNone(run_manifest["latest_accepted_checkpoint_path"])

    def test_run_neural_selfplay_iterations_always_mode_advances_failed_candidate(self) -> None:
        collected = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(collected=collected, candidate_beats_incumbent=False):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    collector_advancement_mode="always",
                )

            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertEqual(
            [call["current_policy_spec"] for call in collected],
            ["random-legal", first_manifest["checkpoint_policy_spec"]],
        )
        self.assertTrue(first_manifest["advancement"]["advance_collector"])
        self.assertEqual(first_manifest["advancement"]["reason"], "collector_advancement_mode_always")
        self.assertEqual(first_manifest["advancement"]["candidate_win_rate"], 0.0)
        self.assertEqual(first_manifest["advancement"]["incumbent_win_rate"], 1.0)
        self.assertEqual(first_manifest["next_current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(second_manifest["current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(run_manifest["current_policy_spec"], second_manifest["checkpoint_policy_spec"])
        self.assertIsNone(run_manifest["latest_accepted_checkpoint_path"])

    def test_run_neural_selfplay_iterations_always_mode_preserves_gate_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(candidate_beats_incumbent=True):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    collector_advancement_mode="always",
                )

            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertTrue(first_manifest["advancement"]["advance_collector"])
        self.assertEqual(first_manifest["advancement"]["reason"], "beat_incumbent")
        self.assertEqual(run_manifest["latest_accepted_checkpoint_path"], str(run_dir / "iteration-0001" / "transformer-policy.pt"))

    def test_run_neural_selfplay_iterations_always_mode_preserves_initial_neural_accepted_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            initial_checkpoint = Path(temp_dir) / "initial-policy.pt"

            with patched_neural_selfplay_dependencies(candidate_beats_incumbent=False):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    initial_policy_spec=f"neural:{initial_checkpoint}",
                    evaluation_games=1,
                    collector_advancement_mode="always",
                )

            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertEqual(first_manifest["next_current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(run_manifest["current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(run_manifest["latest_accepted_checkpoint_path"], str(initial_checkpoint))

    def test_run_neural_selfplay_iterations_always_mode_advances_without_benchmark_for_single_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies():
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=0,
                    collector_advancement_mode="always",
                )

            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertTrue(first_manifest["advancement"]["advance_collector"])
        self.assertEqual(first_manifest["advancement"]["reason"], "collector_advancement_mode_always")
        self.assertIsNone(first_manifest["advancement"]["candidate_win_rate"])
        self.assertEqual(first_manifest["next_current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertEqual(run_manifest["current_policy_spec"], first_manifest["checkpoint_policy_spec"])
        self.assertIsNone(run_manifest["latest_accepted_checkpoint_path"])

    def test_run_neural_selfplay_iterations_rejects_always_mode_with_auto_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patched_neural_selfplay_dependencies():
                with self.assertRaisesRegex(ValueError, "cannot be combined with auto promotion"):
                    run_neural_selfplay_iterations(
                        run_dir=Path(temp_dir) / "run",
                        iterations=1,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                        auto_promotion_config=NeuralSelfPlayPromotionConfig(
                            registry_path=Path(temp_dir) / "promotions.json",
                            gate_config=PromotionGateConfig(require_benchmark=False),
                        ),
                        collector_advancement_mode="always",
                    )

    def test_run_neural_selfplay_iterations_post_iteration_audit_stops_before_next_iteration(self) -> None:
        collected = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(collected=collected):
                with self.assertRaisesRegex(RunAuditFailure, "latest_promotion_recorded") as raised:
                    run_neural_selfplay_iterations(
                        run_dir=run_dir,
                        iterations=2,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                        fixed_opponent_policy_specs=("random-legal",),
                        evaluation_games=1,
                        post_iteration_audit_config=RunAuditConfig(
                            min_latest_benchmark_games=0,
                            require_latest_promotion=True,
                        ),
                    )

            run_manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertFalse(raised.exception.result.passed)
        self.assertEqual(len(collected), 1)
        self.assertEqual(len(run_manifest["iterations"]), 1)
        self.assertFalse((run_dir / "iteration-0002").exists())

    def test_post_iteration_audit_failure_prevents_auto_promotion_when_latest_promotion_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"

            with patched_neural_selfplay_dependencies():
                with self.assertRaisesRegex(RunAuditFailure, "latest_average_decision_rounds") as raised:
                    run_neural_selfplay_iterations(
                        run_dir=run_dir,
                        iterations=1,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                        fixed_opponent_policy_specs=("random-legal",),
                        evaluation_games=1,
                        promotion_registry_path=registry_path,
                        auto_promotion_config=NeuralSelfPlayPromotionConfig(
                            registry_path=registry_path,
                            artifact_dir=artifact_dir,
                            gate_config=passing_promotion_gate_config(),
                            label_prefix="neural-candidate",
                        ),
                        post_iteration_audit_config=RunAuditConfig(
                            min_latest_benchmark_win_rate=0.0,
                            min_latest_benchmark_games=0,
                            max_latest_average_decision_rounds=0.5,
                            max_latest_benchmark_capped_rate=1.0,
                            max_benchmark_win_rate_drop=1.0,
                            require_benchmark=True,
                        ),
                    )

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(raised.exception.result.passed)
        self.assertFalse(registry_path.exists())
        self.assertEqual(list(artifact_dir.glob("*.pt")), [])
        self.assertIsNone(iteration_manifest["promotion"])
        self.assertIsNone(run_manifest["iterations"][0]["promotion"])

    def test_post_iteration_audit_still_checks_promotion_failures_after_auto_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"

            with patched_neural_selfplay_dependencies(candidate_beats_incumbent=False):
                with self.assertRaisesRegex(RunAuditFailure, "consecutive_promotion_failures") as raised:
                    run_neural_selfplay_iterations(
                        run_dir=run_dir,
                        iterations=1,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                        fixed_opponent_policy_specs=("random-legal",),
                        evaluation_games=1,
                        promotion_registry_path=registry_path,
                        auto_promotion_config=NeuralSelfPlayPromotionConfig(
                            registry_path=registry_path,
                            gate_config=PromotionGateConfig(
                                min_benchmark_win_rate=0.5,
                                min_benchmark_games=0,
                                max_collection_capped_rate=1.0,
                                max_benchmark_capped_rate=1.0,
                            ),
                            label_prefix="neural-candidate",
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

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(raised.exception.result.passed)
        self.assertFalse(registry_path.exists())
        self.assertEqual(iteration_manifest["promotion"]["recorded"], False)
        self.assertEqual(run_manifest["iterations"][0]["promotion"]["recorded"], False)

    def test_runtime_health_audit_failure_mode_continues_on_neural_promotion_strength_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"

            with patched_neural_selfplay_dependencies(candidate_beats_incumbent=False):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    result = run_neural_selfplay_iterations(
                        run_dir=run_dir,
                        iterations=1,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                        fixed_opponent_policy_specs=("random-legal",),
                        evaluation_games=1,
                        promotion_registry_path=registry_path,
                        auto_promotion_config=NeuralSelfPlayPromotionConfig(
                            registry_path=registry_path,
                            gate_config=PromotionGateConfig(
                                min_benchmark_win_rate=0.5,
                                min_benchmark_games=0,
                                max_collection_capped_rate=1.0,
                                max_benchmark_capped_rate=1.0,
                            ),
                            label_prefix="neural-candidate",
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

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            iteration_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(iteration_manifest["promotion"]["recorded"], False)
        self.assertEqual(run_manifest["iterations"][0]["promotion"]["recorded"], False)
        self.assertIn("audit_nonblocking_failed_checks: consecutive_promotion_failures", stderr.getvalue())

    def test_run_neural_selfplay_iterations_post_iteration_audit_passes_and_continues(self) -> None:
        collected = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(collected=collected):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
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

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            second_manifest_exists = (run_dir / "iteration-0002" / "manifest.json").exists()

        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(len(collected), 2)
        self.assertEqual(len(run_manifest["iterations"]), 2)
        self.assertTrue(second_manifest_exists)

    def test_run_neural_selfplay_iterations_reports_warning_only_post_iteration_audit_checks(self) -> None:
        collected = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(collected=collected):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    result = run_neural_selfplay_iterations(
                        run_dir=run_dir,
                        iterations=1,
                        games_per_iteration=1,
                        env_factory=lambda: None,  # type: ignore[return-value]
                        rollout_config=RolloutConfig(max_decision_rounds=5),
                        model_config=_entity_test_model_config(),
                        training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
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
        self.assertEqual(len(collected), 1)
        self.assertIn("audit_warning_checks: latest_average_decision_rounds", stderr.getvalue())

    def test_run_neural_selfplay_iterations_resumes_from_manifest(self) -> None:
        collected = []
        trained_paths = []
        trained_initial_models = []

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            with patched_neural_selfplay_dependencies(
                collected=collected,
                trained_paths=trained_paths,
                trained_initial_models=trained_initial_models,
            ):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                )
                resumed = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=2,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    resume=True,
                )

            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(resumed.iterations[0].iteration, 2)
        self.assertEqual(len(run_manifest["iterations"]), 2)
        self.assertEqual(collected[1]["seed_start"], 22)
        self.assertEqual(collected[1]["current_policy_spec"], f"neural:{run_dir / 'iteration-0001' / 'transformer-policy.pt'}")
        self.assertEqual(second_manifest["training_rollout_paths"], [
            str(run_dir / "iteration-0001" / "training-rollouts.jsonl"),
            str(run_dir / "iteration-0002" / "training-rollouts.jsonl"),
        ])
        self.assertEqual(trained_initial_models[-1], "entity-test-iter-0001")
        self.assertEqual(len(run_manifest["invocation_configs"]), 2)
        self.assertFalse(run_manifest["invocation_configs"][0]["resume"])
        self.assertEqual(run_manifest["invocation_configs"][0]["seed_start_argument"], 20)
        self.assertEqual(run_manifest["invocation_configs"][0]["first_iteration_seed_start"], 20)
        self.assertTrue(run_manifest["invocation_configs"][1]["resume"])
        self.assertEqual(run_manifest["invocation_configs"][1]["seed_start_argument"], 1)
        self.assertEqual(run_manifest["invocation_configs"][1]["first_iteration_seed_start"], 22)
        self.assertEqual(second_manifest["invocation_config"], run_manifest["invocation_configs"][1])

    def test_load_neural_selfplay_run_manifest_reconstructs_source_from_iteration_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = {"available": True, "repo_root": "/repo", "branch": "main", "head": "abc123", "dirty": False}
            with (
                patched_neural_selfplay_dependencies(),
                patch("pokezero.neural_selfplay.collect_source_metadata", return_value=source),
            ):
                run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=1,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    seed_start=20,
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                )
            (run_dir / "manifest.json").unlink()

            manifest = load_neural_selfplay_run_manifest(run_dir)

        self.assertEqual(manifest["source"], source)
        self.assertEqual(len(manifest["iterations"]), 1)
        self.assertEqual(manifest["invocation_configs"][0]["source"], source)

    def test_run_neural_selfplay_iterations_auto_promotes_managed_checkpoint(self) -> None:
        collected = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"
            artifact_dir = temp_path / "promoted-checkpoints"

            with patched_neural_selfplay_dependencies(collected=collected):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    max_historical_opponents=2,
                    evaluation_games=1,
                    promotion_registry_path=registry_path,
                    auto_promotion_config=NeuralSelfPlayPromotionConfig(
                        registry_path=registry_path,
                        artifact_dir=artifact_dir,
                        gate_config=passing_promotion_gate_config(),
                        label_prefix="neural-candidate",
                    ),
                )

            registry = load_promotion_registry(registry_path)
            run_manifest = load_neural_selfplay_run_manifest(run_dir)
            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))
            first_selection_spec = registry.selection_checkpoint_policy_spec_for_entry(registry.entries[0])

        expected_pool_config = {
            "fixed_opponent_policy_specs": ["random-legal"],
            "max_historical_opponents": 2,
            "historical_opponent_selection": "recent",
            "promotion_registry_path": str(registry_path),
            "promotion_pool_registry_path": str(registry_path),
            "required_promoted_opponent_pool_size": None,
            "promoted_checkpoint_policy_specs": [],
        }
        self.assertEqual(len(registry.entries), 2)
        self.assertEqual(registry.entries[0].source_type, NEURAL_SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(registry.entries[0].label, "neural-candidate-0001")
        self.assertTrue(registry.entries[0].checkpoint_path)
        self.assertEqual(Path(registry.entries[0].checkpoint_path or "").parent, artifact_dir)
        self.assertEqual(registry.entries[0].checkpoint_policy_spec, f"neural:{registry.entries[0].checkpoint_path}")
        self.assertEqual(run_manifest["invocation_configs"][0]["opponent_pool"], expected_pool_config)
        self.assertEqual(run_manifest["invocation_configs"][0]["auto_promotion"]["artifact_dir"], str(artifact_dir))
        self.assertEqual(run_manifest["invocation_configs"][0]["auto_promotion"]["label_prefix"], "neural-candidate")
        self.assertEqual(first_manifest["opponent_pool_config"], expected_pool_config)
        self.assertEqual(first_manifest["invocation_config"]["opponent_pool"], expected_pool_config)
        self.assertEqual(first_manifest["promotion"]["recorded"], True)
        self.assertEqual(first_manifest["advancement"]["reason"], "promotion_recorded")
        self.assertEqual(first_manifest["next_current_policy_spec"], first_selection_spec)
        self.assertEqual(second_manifest["current_policy_spec"], first_selection_spec)
        self.assertEqual(collected[1]["current_policy_spec"], first_selection_spec)

    def test_promoted_checkpoint_specs_verify_registry_before_neural_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "promotions.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
                        "registry_path": str(registry_path),
                        "latest_policy_id": "entity-test-iter-0001",
                        "latest_checkpoint_path": str(temp_path / "missing-transformer.pt"),
                        "entries": [
                            {
                                "sequence": 1,
                                "policy_id": "entity-test-iter-0001",
                                "checkpoint_path": str(temp_path / "missing-transformer.pt"),
                                "manifest_path": "runs/neural/manifest.json",
                                "source_type": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
                                "source_iteration": 1,
                                "promoted_at": "2026-06-02T00:00:00Z",
                                "label": None,
                                "notes": None,
                                "gate_result": {"passed": True},
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "promotion registry verification failed"):
                _promoted_checkpoint_specs(registry_path)

    def test_run_neural_selfplay_iterations_can_promote_after_initial_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            registry_path = temp_path / "promotions.json"

            with patched_neural_selfplay_dependencies(candidate_beats_incumbent=(False, True)):
                result = run_neural_selfplay_iterations(
                    run_dir=run_dir,
                    iterations=2,
                    games_per_iteration=1,
                    env_factory=lambda: None,  # type: ignore[return-value]
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    model_config=_entity_test_model_config(),
                    training_config=TransformerTrainingConfig(window_size=4, epochs=1, batch_size=2),
                    fixed_opponent_policy_specs=("random-legal",),
                    evaluation_games=1,
                    promotion_registry_path=registry_path,
                    auto_promotion_config=NeuralSelfPlayPromotionConfig(
                        registry_path=registry_path,
                        gate_config=PromotionGateConfig(
                            min_benchmark_win_rate=0.5,
                            min_benchmark_games=0,
                            max_collection_capped_rate=1.0,
                            max_benchmark_capped_rate=1.0,
                        ),
                    ),
                )

            registry = load_promotion_registry(registry_path)
            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertFalse(result.iterations[0].promotion.recorded if result.iterations[0].promotion else True)
        self.assertTrue(result.iterations[1].promotion.recorded if result.iterations[1].promotion else False)
        self.assertEqual(len(registry.entries), 1)
        self.assertEqual(first_manifest["advancement"]["reason"], "promotion_gate_failed")
        self.assertEqual(first_manifest["next_current_policy_spec"], "random-legal")
        self.assertEqual(second_manifest["promotion"]["recorded"], True)
        self.assertIsNone(second_manifest["promotion"]["gate_result"]["incumbent_policy_id"])
        self.assertNotIn(
            "incumbent_benchmark_opponent:entity-test-iter-0001",
            {
                check["name"]
                for check in second_manifest["promotion"]["gate_result"]["checks"]
            },
        )

    def test_neural_cli_report_prints_manifest_summary_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = neural_report_source_metadata(branch="scott/neural-report", head="abc123", dirty=True)
            write_neural_report_manifest(run_dir, source=source)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("current_policy: neural:", output)
        self.assertIn("latest_checkpoint:", output)
        self.assertIn("latest_accepted_checkpoint:", output)
        self.assertIn("source_metadata:", output)
        self.assertIn("available: yes", output)
        self.assertIn("branch: scott/neural-report", output)
        self.assertIn("head: abc123", output)
        self.assertIn("dirty: yes", output)
        self.assertIn("repo_root: /repo", output)
        self.assertIn("iterations: 1", output)
        self.assertIn("bench_wr", output)
        self.assertIn("inc_wr", output)
        self.assertIn("val_sign", output)
        self.assertIn("val_ece", output)
        self.assertIn("ppo_cov", output)
        self.assertIn("ppo_clip", output)
        self.assertIn("ppo_ent", output)
        self.assertIn("0.617", output)
        self.assertIn("0.600", output)
        self.assertIn("0.250000", output)
        self.assertIn("0.7500", output)
        self.assertIn("benchmark_opponent_curves:", output)
        self.assertIn("note: fixed yardsticks only; rates are candidate wins / total games.", output)
        self.assertIn("- random-legal: 1:0.600/20g,cap=1", output)
        self.assertIn("- simple-legal: 1:1.000/20g", output)
        self.assertIn("- max-damage: 1:0.250/20g,cap=2", output)
        self.assertIn("foundation_readiness:", output)
        self.assertIn(
            "note: presence/sample-size only; inspect value quality and strength separately.",
            output,
        )
        self.assertIn("- value_calibration: present examples=6 sign=0.7200 ece=0.180000", output)
        self.assertIn(
            "- max_damage_yardstick: iter=1 win_rate=0.250 games=20 cap=2 sample=below_milestone(20/300)",
            output,
        )
        self.assertIn("- foundation_evidence_status: incomplete", output)
        self.assertIn("reasons: max_damage_sample_below_milestone", output)
        self.assertIn("0.100000", output)
        self.assertIn("0.7200", output)
        self.assertIn("0.180000", output)
        self.assertIn("0.5000", output)
        self.assertIn("0.875", output)
        self.assertIn("0.125", output)
        self.assertIn("1.750", output)

    def test_neural_cli_report_omits_incumbent_from_yardstick_curves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_neural_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["iterations"][0]["benchmark"]["head_to_heads"].append(
                {
                    "label": "entity-test-iter-0001 vs entity-test-iter-0000",
                    "first_policy_id": "entity-test-iter-0001",
                    "second_policy_id": "entity-test-iter-0000",
                    "games": 20,
                    "first_policy_wins": 11,
                    "second_policy_wins": 9,
                    "ties": 0,
                    "capped_games": 0,
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("benchmark_opponent_curves:", output)
        self.assertIn("- random-legal: 1:0.600/20g,cap=1", output)
        self.assertIn("- max-damage: 1:0.250/20g,cap=2", output)
        self.assertNotIn("entity-test-iter-0000", output)

    def test_neural_cli_report_yardstick_curves_fall_back_to_legacy_matchups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_neural_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            benchmark = manifest["iterations"][0]["benchmark"]
            benchmark["head_to_heads"] = []
            benchmark["matchups"] = [
                {
                    "label": "entity-test-iter-0001 vs random-legal",
                    "p1_policy_id": "entity-test-iter-0001",
                    "p2_policy_id": "random-legal",
                    "seed_start": 1,
                    "metrics": {
                        "games": 10,
                        "elapsed_seconds": 1.0,
                        "total_decision_rounds": 20,
                        "total_simulator_turns": 20,
                        "p1_wins": 6,
                        "p2_wins": 4,
                        "ties": 0,
                        "capped_games": 1,
                    },
                },
                {
                    "label": "random-legal vs entity-test-iter-0001",
                    "p1_policy_id": "random-legal",
                    "p2_policy_id": "entity-test-iter-0001",
                    "seed_start": 11,
                    "metrics": {
                        "games": 10,
                        "elapsed_seconds": 1.0,
                        "total_decision_rounds": 20,
                        "total_simulator_turns": 20,
                        "p1_wins": 5,
                        "p2_wins": 5,
                        "ties": 0,
                        "capped_games": 2,
                    },
                },
                {
                    "label": "entity-test-iter-0001 vs entity-test-iter-0000",
                    "p1_policy_id": "entity-test-iter-0001",
                    "p2_policy_id": "entity-test-iter-0000",
                    "seed_start": 21,
                    "metrics": {
                        "games": 10,
                        "elapsed_seconds": 1.0,
                        "total_decision_rounds": 20,
                        "total_simulator_turns": 20,
                        "p1_wins": 7,
                        "p2_wins": 3,
                        "ties": 0,
                        "capped_games": 0,
                    },
                },
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("benchmark_opponent_curves:", output)
        self.assertIn("- random-legal: 1:0.550/20g,cap=3", output)
        self.assertNotIn("entity-test-iter-0000", output)

    def test_neural_cli_report_marks_foundation_readable_at_milestone_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_neural_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            max_damage = next(
                result
                for result in manifest["iterations"][0]["benchmark"]["head_to_heads"]
                if result["second_policy_id"] == "max-damage"
            )
            max_damage["games"] = 300
            max_damage["first_policy_wins"] = 120
            max_damage["second_policy_wins"] = 180
            max_damage["capped_games"] = 0
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("- max_damage_yardstick: iter=1 win_rate=0.400 games=300 cap=0 sample=milestone", output)
        self.assertIn("- foundation_evidence_status: present_and_sample_sized", output)
        self.assertNotIn("reasons:", output)

    def test_neural_cli_report_foundation_readiness_requires_latest_yardstick(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_neural_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_iteration = manifest["iterations"][0]
            max_damage = next(
                result
                for result in first_iteration["benchmark"]["head_to_heads"]
                if result["second_policy_id"] == "max-damage"
            )
            max_damage["games"] = 300
            max_damage["first_policy_wins"] = 120
            max_damage["second_policy_wins"] = 180
            second_iteration = json.loads(json.dumps(first_iteration))
            second_iteration["iteration"] = 2
            second_iteration["benchmark"]["head_to_heads"] = [
                result
                for result in second_iteration["benchmark"]["head_to_heads"]
                if result["second_policy_id"] != "max-damage"
            ]
            manifest["iterations"].append(second_iteration)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("- max_damage_yardstick: missing", output)
        self.assertIn("- foundation_evidence_status: incomplete", output)
        self.assertIn("reasons: max_damage_yardstick_missing", output)

    def test_neural_cli_iterate_summary_prints_ppo_diagnostics_when_present(self) -> None:
        result = SimpleNamespace(
            run_dir=Path("/tmp/run"),
            latest_checkpoint_path=Path("/tmp/run/iteration-0001/transformer-policy.pt"),
            iterations=[
                SimpleNamespace(
                    iteration=1,
                    metrics=SimpleNamespace(games=3),
                    checkpoint_path=Path("/tmp/run/iteration-0001/transformer-policy.pt"),
                    training=SimpleNamespace(
                        final_metrics=TransformerEpochMetrics(
                            epoch=1,
                            examples=6,
                            loss=0.25,
                            policy_loss=0.2,
                            policy_accuracy=0.75,
                            ppo_valid_fraction=0.875,
                            ppo_clip_fraction=0.125,
                            ppo_entropy=1.75,
                        )
                    ),
                    benchmark=None,
                    promotion=None,
                )
            ],
        )

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            _print_iterate_summary(result)

        output = stdout.getvalue()
        self.assertIn("ppo_cov=0.875", output)
        self.assertIn("ppo_clip=0.125", output)
        self.assertIn("ppo_ent=1.750", output)

    def test_neural_cli_report_renders_missing_ppo_diagnostics_as_dash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            write_neural_report_manifest(run_dir)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            epoch = manifest["iterations"][0]["training"]["epochs"][0]
            for key in (
                "ppo_valid_examples",
                "ppo_valid_fraction",
                "ppo_advantage_mean",
                "ppo_advantage_std",
                "ppo_ratio_mean",
                "ppo_clip_fraction",
                "ppo_entropy",
            ):
                epoch.pop(key, None)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        self.assertEqual(exit_code, 0)
        lines = stdout.getvalue().splitlines()
        header = next(line for line in lines if line.strip().startswith("iter "))
        data_line = next(line for line in lines if line.split()[:1] == ["1"])
        columns = dict(zip(header.split(), data_line.split()))
        self.assertEqual(columns["ppo_cov"], "-")
        self.assertEqual(columns["ppo_clip"], "-")
        self.assertEqual(columns["ppo_ent"], "-")

    def test_neural_cli_report_can_print_json_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = neural_report_source_metadata(branch="scott/json-report", head="def456", dirty=False)
            write_neural_report_manifest(run_dir, source=source)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir), "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], NEURAL_SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(payload["source"], source)
        self.assertEqual(payload["iterations"][0]["iteration"], 1)
        self.assertNotIn("source_metadata:", stdout.getvalue())

    def test_neural_cli_report_reconstructs_from_iteration_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            source = neural_report_source_metadata(branch="scott/reconstructed", head="ghi789", dirty=False)
            write_neural_report_manifest(run_dir, top_level=False, source=source)

            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = neural_cli_main(["report", "--run-dir", str(run_dir)])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("iterations: 1", output)
        self.assertIn("branch: scott/reconstructed", output)
        self.assertIn("head: ghi789", output)
        self.assertIn("dirty: no", output)

    def test_neural_cli_report_handles_missing_and_unavailable_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_run_dir = Path(temp_dir) / "missing"
            unavailable_run_dir = Path(temp_dir) / "unavailable"
            write_neural_report_manifest(missing_run_dir)
            write_neural_report_manifest(
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
                missing_exit = neural_cli_main(["report", "--run-dir", str(missing_run_dir)])
            with patch("sys.stdout", new_callable=io.StringIO) as unavailable_stdout:
                unavailable_exit = neural_cli_main(["report", "--run-dir", str(unavailable_run_dir)])

        self.assertEqual(missing_exit, 0)
        self.assertIn("source_metadata: -", missing_stdout.getvalue())
        self.assertEqual(unavailable_exit, 0)
        self.assertIn("available: no", unavailable_stdout.getvalue())
        self.assertIn("error: RuntimeError: git unavailable", unavailable_stdout.getvalue())

    def test_torch_smoke_runs_train_save_load_benchmark_chain(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"

            result = run_neural_selfplay_iterations(
                run_dir=run_dir,
                iterations=1,
                games_per_iteration=1,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                model_config=TransformerPolicyConfig.compact_category(
                    category_vocab=tuple(range(1, 17)),
                    category_oov_buckets=4,
                    policy_id="entity-smoke",
                    window_size=2,
                    token_type_vocab_size=8,
                    categorical_feature_count=1,
                    numeric_feature_count=1,
                    embedding_dim=16,
                    transformer_layers=1,
                    attention_heads=4,
                    feedforward_dim=32,
                    dropout=0.0,
                ),
                training_config=TransformerTrainingConfig(
                    window_size=2,
                    epochs=1,
                    batch_size=2,
                    max_batches=1,
                    device="cpu",
                ),
                fixed_opponent_policy_specs=("random-legal",),
                evaluation_games=1,
                evaluation_seed_start=100,
            )

            # Assert while the TemporaryDirectory is still open — the saved checkpoint lives under
            # it, so checking .exists() after the `with` exits would always fail (the dir is gone).
            self.assertTrue(result.latest_checkpoint_path and result.latest_checkpoint_path.exists())
            self.assertIsNotNone(result.iterations[0].benchmark)


def write_neural_report_manifest(run_dir: Path, *, top_level: bool = True, source: dict | None = None) -> None:
    checkpoint_path = run_dir / "iteration-0001" / "transformer-policy.pt"
    iteration_dir = run_dir / "iteration-0001"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    iteration_manifest = {
        "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
        "iteration": 1,
        "rollout_path": str(iteration_dir / "rollouts.jsonl"),
        "training_rollout_path": str(iteration_dir / "training-rollouts.jsonl"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_policy_spec": f"neural:{checkpoint_path}",
        "current_policy_spec": "random-legal",
        "opponent_policy_specs": ["random-legal"],
        "opponent_pool_config": {},
        "invocation_config": {"benchmark_reference_policy_specs": ["max-damage"]},
        "benchmark_reference_policy_specs": ["max-damage"],
        "training_rollout_paths": [str(iteration_dir / "training-rollouts.jsonl")],
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
        },
        "training": {
            "model_config": {"policy_id": "entity-test-iter-0001"},
            "config": {},
            "epochs": [
                {
                    "epoch": 1,
                    "examples": 6,
                    "loss": 0.25,
                    "policy_loss": 0.2,
                    "policy_accuracy": 0.75,
                    "value_loss": 0.1,
                    "opponent_loss": 0.05,
                    "opponent_accuracy": 0.5,
                    "ppo_valid_examples": 7,
                    "ppo_valid_fraction": 0.875,
                    "ppo_advantage_mean": 0.2,
                    "ppo_advantage_std": 0.4,
                    "ppo_ratio_mean": 1.1,
                    "ppo_clip_fraction": 0.125,
                    "ppo_entropy": 1.75,
                }
            ],
        },
        "value_calibration": {
            "scope": "iteration",
            "paths": [str(iteration_dir / "training-rollouts.jsonl")],
            "batch_size": 128,
            "bins": 10,
            "report": {
                "examples": 6,
                "mse": 0.3,
                "mae": 0.4,
                "bias": -0.1,
                "sign_accuracy": 0.72,
                "expected_calibration_error": 0.18,
                "bins": [],
                "slices": [],
            },
        },
        "benchmark": {
            "format_id": "gen3randombattle",
            "max_decision_rounds": 250,
            "games_per_matchup": 10,
            "total_games": 60,
            "elapsed_seconds": 4.0,
            "matchups": [],
            "head_to_heads": [
                {
                    "label": "entity-test-iter-0001 vs random-legal",
                    "first_policy_id": "entity-test-iter-0001",
                    "second_policy_id": "random-legal",
                    "games": 20,
                    "first_policy_wins": 12,
                    "second_policy_wins": 8,
                    "ties": 0,
                    "capped_games": 1,
                },
                {
                    "label": "entity-test-iter-0001 vs simple-legal",
                    "first_policy_id": "entity-test-iter-0001",
                    "second_policy_id": "simple-legal",
                    "games": 20,
                    "first_policy_wins": 20,
                    "second_policy_wins": 0,
                    "ties": 0,
                    "capped_games": 0,
                },
                {
                    "label": "entity-test-iter-0001 vs max-damage",
                    "first_policy_id": "entity-test-iter-0001",
                    "second_policy_id": "max-damage",
                    "games": 20,
                    "first_policy_wins": 5,
                    "second_policy_wins": 15,
                    "ties": 0,
                    "capped_games": 2,
                }
            ],
        },
        "advancement": {
            "advance_collector": True,
            "reason": "beat_incumbent",
            "candidate_policy_id": "entity-test-iter-0001",
            "incumbent_policy_id": "random-legal",
            "candidate_win_rate": 0.6,
            "incumbent_win_rate": 0.4,
            "games": 20,
        },
        "promotion": {"recorded": False},
        "next_current_policy_spec": f"neural:{checkpoint_path}",
    }
    if source is not None:
        iteration_manifest["source"] = source
    (iteration_dir / "manifest.json").write_text(json.dumps(iteration_manifest, indent=2), encoding="utf-8")
    if not top_level:
        return
    run_manifest = {
        "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "invocation_configs": [],
        "latest_checkpoint_path": str(checkpoint_path),
        "current_policy_spec": f"neural:{checkpoint_path}",
        "latest_accepted_checkpoint_path": str(checkpoint_path),
        "iterations": [iteration_manifest],
    }
    if source is not None:
        run_manifest["source"] = source
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def neural_report_source_metadata(
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


def patched_neural_selfplay_dependencies(
    *,
    collected: list | None = None,
    trained_paths: list | None = None,
    trained_initial_models: list | None = None,
    captured_benchmarks: list | None = None,
    captured_calibrations: list | None = None,
    candidate_beats_incumbent: bool | tuple[bool, ...] = True,
):
    collected = collected if collected is not None else []
    trained_paths = trained_paths if trained_paths is not None else []
    trained_initial_models = trained_initial_models if trained_initial_models is not None else []
    captured_benchmarks = captured_benchmarks if captured_benchmarks is not None else []
    captured_calibrations = captured_calibrations if captured_calibrations is not None else []

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

    def fake_train_transformer_policy(paths, *, model_config, training_config, initial_model=None, epoch_callback=None):
        trained_paths.append(tuple(Path(path) for path in paths))
        trained_initial_models.append(getattr(initial_model, "policy_id", None))
        model = FakeModel(model_config.policy_id)
        metrics = []
        for epoch in range(1, training_config.epochs + 1):
            model.weight = epoch
            metrics.append(
                TransformerEpochMetrics(
                    epoch=epoch,
                    examples=4,
                    loss=float(epoch) * 0.25,
                    policy_loss=0.2,
                    policy_accuracy=0.75,
                    value_loss=0.1,
                    opponent_loss=0.05,
                    opponent_accuracy=0.5,
                )
            )
            if epoch_callback is not None:
                epoch_callback(
                    model,
                    TransformerTrainingResult(
                        model_config=model_config,
                        training_config=training_config,
                        epochs=tuple(metrics),
                    ),
                )
        result = TransformerTrainingResult(
            model_config=model_config,
            training_config=training_config,
            epochs=tuple(metrics),
        )
        return model, result

    def fake_save_transformer_checkpoint(path, model, *, result):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")

    class FakeModel:
        def __init__(self, policy_id: str) -> None:
            self.policy_id = policy_id
            self.weight = 0
            self.loaded_state = None

        def state_dict(self) -> dict[str, object]:
            return {"policy_id": self.policy_id, "weight": self.weight}

        def load_state_dict(self, state: Mapping[str, object]) -> None:
            self.loaded_state = dict(state)
            self.weight = int(state["weight"])

    class FakePolicy:
        def __init__(self, policy_id: str) -> None:
            self.policy_id = policy_id

    def fake_load_transformer_policy(path, *args, **kwargs):
        return FakePolicy(_policy_id_from_fake_checkpoint_path(Path(path)))

    def fake_load_transformer_checkpoint(path, *args, **kwargs):
        return FakeModel(_policy_id_from_fake_checkpoint_path(Path(path))), None

    def fake_benchmark_rollouts(**kwargs):
        call_index = len(captured_benchmarks)
        if isinstance(candidate_beats_incumbent, tuple):
            candidate_wins = candidate_beats_incumbent[min(call_index, len(candidate_beats_incumbent) - 1)]
        else:
            candidate_wins = candidate_beats_incumbent
        captured_benchmarks.append(kwargs)
        matchup_results = []
        games = kwargs["games"]
        for matchup in kwargs["matchups"]:
            p1_is_candidate = str(matchup.p1_policy.policy_id).startswith("entity-test-iter-")
            p2_is_candidate = str(matchup.p2_policy.policy_id).startswith("entity-test-iter-")
            if p1_is_candidate and p2_is_candidate:
                candidate_number = int(str(matchup.p1_policy.policy_id).rsplit("-", maxsplit=1)[-1])
                p1_is_candidate = candidate_number == max(
                    int(str(matchup.p1_policy.policy_id).rsplit("-", maxsplit=1)[-1]),
                    int(str(matchup.p2_policy.policy_id).rsplit("-", maxsplit=1)[-1]),
                )
                p2_is_candidate = not p1_is_candidate
            p1_wins = games if (p1_is_candidate == candidate_wins) else 0
            p2_wins = games - p1_wins
            matchup_results.append(
                BenchmarkMatchupResult(
                    label=matchup.label,
                    p1_policy_id=str(matchup.p1_policy.policy_id),
                    p2_policy_id=str(matchup.p2_policy.policy_id),
                    seed_start=kwargs["seed_start"],
                    metrics=CollectionMetrics(
                        games=games,
                        elapsed_seconds=1.0,
                        total_decision_rounds=games,
                        total_simulator_turns=games,
                        p1_wins=p1_wins,
                        p2_wins=p2_wins,
                        ties=0,
                        capped_games=0,
                    ),
                )
            )
        return BenchmarkReport(
            format_id=kwargs["rollout_config"].format_id,
            max_decision_rounds=kwargs["rollout_config"].max_decision_rounds,
            games_per_matchup=kwargs["games"],
            matchups=tuple(matchup_results),
        )

    class FakeValueCalibrationReport:
        mse = 0.25
        mae = 0.5
        bias = -0.1
        sign_accuracy = 0.75
        expected_calibration_error = 0.2

        def to_dict(self) -> dict:
            return {
                "examples": 4,
                "mse": self.mse,
                "mae": self.mae,
                "bias": self.bias,
                "sign_accuracy": self.sign_accuracy,
                "expected_calibration_error": self.expected_calibration_error,
                "bins": [],
                "slices": [],
            }

    def fake_evaluate_value_calibration(**kwargs):
        captured_calibrations.append(kwargs)
        return FakeValueCalibrationReport()

    return patch.multiple(
        "pokezero.neural_selfplay",
        require_torch=lambda: object(),
        collect_selfplay_rollouts=fake_collect_selfplay_rollouts,
        train_transformer_policy=fake_train_transformer_policy,
        save_transformer_checkpoint=fake_save_transformer_checkpoint,
        load_transformer_checkpoint=fake_load_transformer_checkpoint,
        load_transformer_policy=fake_load_transformer_policy,
        benchmark_rollouts=fake_benchmark_rollouts,
        evaluate_value_calibration=fake_evaluate_value_calibration,
    )


def _policy_id_from_fake_checkpoint_path(path: Path) -> str:
    if path.parent.name.startswith("iteration-"):
        iteration = path.parent.name.rsplit("-", maxsplit=1)[-1]
        return f"entity-test-iter-{iteration}"
    marker = "entity-test-iter-"
    if marker in path.stem:
        return f"{marker}{path.stem.rsplit(marker, maxsplit=1)[-1]}"
    return "entity-test"


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


def observation() -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((1,) for _ in range(spec.token_count)),
        numeric_features=tuple((1.0,) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=(True, False, False, False, False, False, False, False, False),
        perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
    )


class OneTurnEnv:
    def __init__(self) -> None:
        self._observation = observation()
        self._requested = ("p1", "p2")
        self._terminal = None

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
        pass


if __name__ == "__main__":
    unittest.main()
