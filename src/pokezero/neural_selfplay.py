"""Neural self-play iteration harness.

This module is the neural counterpart to the dependency-free linear self-play
loop. It still trains from collected rollout records rather than PPO updates,
but it closes the first transformer iteration loop: collect with the current
policy, train a neural checkpoint, benchmark it, and write auditable manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .collection import (
    BenchmarkMatchup,
    BenchmarkReport,
    CollectionMetrics,
    benchmark_rollouts,
)
from .env import PokeZeroEnv
from .neural_policy import (
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    load_transformer_policy,
    require_torch,
    save_transformer_checkpoint,
    train_transformer_policy,
)
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig
from .selfplay import collect_selfplay_rollouts


NEURAL_SELFPLAY_RUN_SCHEMA_VERSION = "pokezero.neural_selfplay_run.v1"


@dataclass(frozen=True)
class NeuralSelfPlayIterationResult:
    iteration: int
    rollout_path: Path
    training_rollout_path: Path
    checkpoint_path: Path
    manifest_path: Path
    current_policy_spec: str
    opponent_policy_specs: tuple[str, ...]
    training_rollout_paths: tuple[Path, ...]
    seed_start: int
    worker_count: int
    metrics: CollectionMetrics
    training: TransformerTrainingResult
    benchmark: BenchmarkReport | None = None

    @property
    def checkpoint_policy_spec(self) -> str:
        return f"neural:{self.checkpoint_path}"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
            "iteration": self.iteration,
            "rollout_path": str(self.rollout_path),
            "training_rollout_path": str(self.training_rollout_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_policy_spec": self.checkpoint_policy_spec,
            "current_policy_spec": self.current_policy_spec,
            "opponent_policy_specs": list(self.opponent_policy_specs),
            "training_rollout_paths": [str(path) for path in self.training_rollout_paths],
            "seed_start": self.seed_start,
            "worker_count": self.worker_count,
            "collection_metrics": self.metrics.to_dict(),
            "training": _training_result_to_dict(self.training),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
        }


@dataclass(frozen=True)
class NeuralSelfPlayRunResult:
    run_dir: Path
    iterations: tuple[NeuralSelfPlayIterationResult, ...]

    @property
    def latest_checkpoint_path(self) -> Path | None:
        if not self.iterations:
            return None
        return self.iterations[-1].checkpoint_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "iterations": [iteration.to_manifest_dict() for iteration in self.iterations],
            "latest_checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path else None,
        }


def load_neural_selfplay_run_manifest(run_dir: Path) -> Mapping[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Neural self-play run manifest does not exist: {manifest_path}")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported neural self-play run schema: {manifest.get('schema_version')!r}.")
    _sequence(manifest.get("iterations", ()))
    return manifest


def run_neural_selfplay_iterations(
    *,
    run_dir: Path,
    iterations: int,
    games_per_iteration: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    model_config: TransformerPolicyConfig,
    training_config: TransformerTrainingConfig,
    seed_start: int = 1,
    initial_policy_spec: str = "random-legal",
    fixed_opponent_policy_specs: Iterable[str] = ("random-legal", "simple-legal"),
    max_historical_opponents: int = 3,
    evaluation_games: int = 0,
    evaluation_seed_start: int = 1_000_000,
    worker_count: int = 1,
) -> NeuralSelfPlayRunResult:
    require_torch()
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive.")
    if max_historical_opponents < 0:
        raise ValueError("max_historical_opponents must be non-negative.")
    if evaluation_games < 0:
        raise ValueError("evaluation_games must be non-negative.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    fixed_opponents = tuple(fixed_opponent_policy_specs)
    if not fixed_opponents:
        raise ValueError("at least one fixed opponent policy spec is required.")
    if model_config.window_size != training_config.window_size:
        raise ValueError("model_config window_size must match training_config window_size.")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("neural self-play run manifest already exists; choose a new run_dir.")
    run_dir.mkdir(parents=True, exist_ok=True)

    current_policy_spec = initial_policy_spec
    checkpoint_history: list[str] = []
    training_rollout_history: list[Path] = []
    results: list[NeuralSelfPlayIterationResult] = []

    for offset in range(iterations):
        iteration = offset + 1
        iteration_dir = run_dir / f"iteration-{iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = iteration_dir / "rollouts.jsonl"
        training_rollout_path = iteration_dir / "training-rollouts.jsonl"
        checkpoint_path = iteration_dir / "transformer-policy.pt"
        iteration_manifest_path = iteration_dir / "manifest.json"
        iteration_seed_start = seed_start + (offset * games_per_iteration)
        opponent_policy_specs = _opponent_pool(
            fixed_policy_specs=fixed_opponents,
            checkpoint_history=checkpoint_history,
            current_policy_spec=current_policy_spec,
            max_historical_opponents=max_historical_opponents,
        )

        metrics = collect_selfplay_rollouts(
            output_path=rollout_path,
            training_output_path=training_rollout_path,
            games=games_per_iteration,
            env_factory=env_factory,
            rollout_config=rollout_config,
            seed_start=iteration_seed_start,
            current_policy_spec=current_policy_spec,
            opponent_policy_specs=opponent_policy_specs,
            worker_count=worker_count,
        )
        training_rollout_history.append(training_rollout_path)
        iteration_model_config = replace(
            model_config,
            policy_id=f"{model_config.policy_id}-iter-{iteration:04d}",
        )
        model, training = train_transformer_policy(
            tuple(training_rollout_history),
            model_config=iteration_model_config,
            training_config=training_config,
        )
        save_transformer_checkpoint(checkpoint_path, model, result=training)
        benchmark = None
        if evaluation_games:
            benchmark = _benchmark_checkpoint(
                checkpoint_path=checkpoint_path,
                env_factory=env_factory,
                rollout_config=rollout_config,
                games=evaluation_games,
                seed_start=evaluation_seed_start + (offset * evaluation_games),
                device=training_config.device,
            )

        result = NeuralSelfPlayIterationResult(
            iteration=iteration,
            rollout_path=rollout_path,
            training_rollout_path=training_rollout_path,
            checkpoint_path=checkpoint_path,
            manifest_path=iteration_manifest_path,
            current_policy_spec=current_policy_spec,
            opponent_policy_specs=opponent_policy_specs,
            training_rollout_paths=tuple(training_rollout_history),
            seed_start=iteration_seed_start,
            worker_count=worker_count,
            metrics=metrics,
            training=training,
            benchmark=benchmark,
        )
        _write_json(iteration_manifest_path, result.to_manifest_dict())
        results.append(result)
        _write_json(run_dir / "manifest.json", NeuralSelfPlayRunResult(run_dir=run_dir, iterations=tuple(results)).to_dict())
        checkpoint_history.append(result.checkpoint_policy_spec)
        current_policy_spec = result.checkpoint_policy_spec

    return NeuralSelfPlayRunResult(run_dir=run_dir, iterations=tuple(results))


def _benchmark_checkpoint(
    *,
    checkpoint_path: Path,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    games: int,
    seed_start: int,
    device: str | None,
) -> BenchmarkReport:
    model_policy = load_transformer_policy(checkpoint_path, deterministic=True, device=device)
    policy_id = str(model_policy.policy_id)
    return benchmark_rollouts(
        games=games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        matchups=(
            BenchmarkMatchup(f"{policy_id} vs random-legal", model_policy, RandomLegalPolicy()),
            BenchmarkMatchup(f"random-legal vs {policy_id}", RandomLegalPolicy(), model_policy),
            BenchmarkMatchup(f"{policy_id} vs simple-legal", model_policy, SimpleLegalPolicy()),
            BenchmarkMatchup(f"simple-legal vs {policy_id}", SimpleLegalPolicy(), model_policy),
        ),
    )


def _opponent_pool(
    *,
    fixed_policy_specs: tuple[str, ...],
    checkpoint_history: Iterable[str],
    current_policy_spec: str,
    max_historical_opponents: int,
) -> tuple[str, ...]:
    historical = [spec for spec in checkpoint_history if spec != current_policy_spec]
    if max_historical_opponents:
        historical = historical[-max_historical_opponents:]
    else:
        historical = []
    return fixed_policy_specs + tuple(historical)


def _training_result_to_dict(result: TransformerTrainingResult) -> dict[str, Any]:
    return {
        "model_config": result.model_config.to_dict(),
        "config": result.training_config.to_dict(),
        "epochs": [metrics.to_dict() for metrics in result.epochs],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise ValueError("expected JSON array payload.")
    return tuple(value)
