import io
import json
from pathlib import Path
from types import SimpleNamespace
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
    _promoted_checkpoint_specs,
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
from pokezero.neural_cli import main as neural_cli_main
from pokezero.evaluation import PromotionGateConfig
from pokezero.promotion import PROMOTION_REGISTRY_SCHEMA_VERSION, load_promotion_registry
from pokezero.run_audit import RunAuditConfig, RunAuditFailure
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
                        model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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

    def test_run_neural_selfplay_iterations_writes_manifests_and_accumulates_training_data(self) -> None:
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                        model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                        model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
                    model_config=TransformerPolicyConfig(policy_id="entity-test", embedding_dim=16, attention_heads=4),
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
        self.assertIn("0.800", output)
        self.assertIn("0.600", output)
        self.assertIn("0.250000", output)
        self.assertIn("0.7500", output)
        self.assertIn("0.100000", output)
        self.assertIn("0.5000", output)

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
                model_config=TransformerPolicyConfig(
                    policy_id="entity-smoke",
                    window_size=2,
                    categorical_vocab_size=32,
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
        "invocation_config": {},
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
                }
            ],
        },
        "benchmark": {
            "format_id": "gen3randombattle",
            "max_decision_rounds": 250,
            "games_per_matchup": 10,
            "total_games": 20,
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
    candidate_beats_incumbent: bool | tuple[bool, ...] = True,
):
    collected = collected if collected is not None else []
    trained_paths = trained_paths if trained_paths is not None else []
    trained_initial_models = trained_initial_models if trained_initial_models is not None else []
    captured_benchmarks = captured_benchmarks if captured_benchmarks is not None else []

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

    def fake_train_transformer_policy(paths, *, model_config, training_config, initial_model=None):
        trained_paths.append(tuple(Path(path) for path in paths))
        trained_initial_models.append(getattr(initial_model, "policy_id", None))
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
        return FakeModel(model_config.policy_id), result

    def fake_save_transformer_checkpoint(path, model, *, result):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")

    class FakeModel:
        def __init__(self, policy_id: str) -> None:
            self.policy_id = policy_id

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

    return patch.multiple(
        "pokezero.neural_selfplay",
        require_torch=lambda: object(),
        collect_selfplay_rollouts=fake_collect_selfplay_rollouts,
        train_transformer_policy=fake_train_transformer_policy,
        save_transformer_checkpoint=fake_save_transformer_checkpoint,
        load_transformer_checkpoint=fake_load_transformer_checkpoint,
        load_transformer_policy=fake_load_transformer_policy,
        benchmark_rollouts=fake_benchmark_rollouts,
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
