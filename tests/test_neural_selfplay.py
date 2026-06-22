import json
from pathlib import Path
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
    load_neural_selfplay_run_manifest,
    run_neural_selfplay_iterations,
)
from pokezero.evaluation import PromotionGateConfig
from pokezero.promotion import load_promotion_registry
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

    def test_run_neural_selfplay_iterations_writes_manifests_and_accumulates_training_data(self) -> None:
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
        self.assertEqual(first_manifest["checkpoint_policy_spec"], f"neural:{run_dir / 'iteration-0001' / 'transformer-policy.pt'}")
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
            first_manifest = json.loads((run_dir / "iteration-0001" / "manifest.json").read_text(encoding="utf-8"))
            second_manifest = json.loads((run_dir / "iteration-0002" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(registry.entries), 2)
        self.assertEqual(registry.entries[0].source_type, NEURAL_SELFPLAY_RUN_SCHEMA_VERSION)
        self.assertEqual(registry.entries[0].label, "neural-candidate-0001")
        self.assertTrue(registry.entries[0].checkpoint_path)
        self.assertEqual(Path(registry.entries[0].checkpoint_path or "").parent, artifact_dir)
        self.assertEqual(registry.entries[0].checkpoint_policy_spec, f"neural:{registry.entries[0].checkpoint_path}")
        self.assertEqual(first_manifest["promotion"]["recorded"], True)
        self.assertEqual(first_manifest["advancement"]["reason"], "promotion_recorded")
        self.assertEqual(first_manifest["next_current_policy_spec"], registry.entries[0].checkpoint_policy_spec)
        self.assertEqual(second_manifest["current_policy_spec"], registry.entries[0].checkpoint_policy_spec)
        self.assertEqual(collected[1]["current_policy_spec"], registry.entries[0].checkpoint_policy_spec)

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


def patched_neural_selfplay_dependencies(
    *,
    collected: list | None = None,
    trained_paths: list | None = None,
    trained_initial_models: list | None = None,
    captured_benchmarks: list | None = None,
    candidate_beats_incumbent: bool = True,
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
            p1_wins = games if (p1_is_candidate == candidate_beats_incumbent) else 0
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
