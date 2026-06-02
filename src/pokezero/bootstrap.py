"""Bootstrap data and checkpoint workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .collection import BenchmarkMatchup, BenchmarkReport, CollectionMetrics, benchmark_rollouts, policy_from_spec
from .env import PokeZeroEnv
from .linear_policy import LinearSoftmaxPolicy, LinearTrainingConfig, LinearTrainingResult, save_linear_model, train_linear_policy
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig
from .selfplay import collect_selfplay_rollouts


TEACHER_BOOTSTRAP_SCHEMA_VERSION = "pokezero.teacher_bootstrap.v1"


@dataclass(frozen=True)
class TeacherBootstrapResult:
    run_dir: Path
    manifest_path: Path
    full_train_rollout_path: Path
    train_rollout_path: Path
    full_validation_rollout_path: Path
    validation_rollout_path: Path
    checkpoint_path: Path
    teacher_policy_spec: str
    opponent_policy_specs: tuple[str, ...]
    train_metrics: CollectionMetrics
    validation_metrics: CollectionMetrics
    training: LinearTrainingResult
    benchmark: BenchmarkReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TEACHER_BOOTSTRAP_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "full_train_rollout_path": str(self.full_train_rollout_path),
            "train_rollout_path": str(self.train_rollout_path),
            "full_validation_rollout_path": str(self.full_validation_rollout_path),
            "validation_rollout_path": str(self.validation_rollout_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_policy_spec": f"linear:{self.checkpoint_path}",
            "teacher_policy_spec": self.teacher_policy_spec,
            "opponent_policy_specs": list(self.opponent_policy_specs),
            "train_collection_metrics": self.train_metrics.to_dict(),
            "validation_collection_metrics": self.validation_metrics.to_dict(),
            "training": _training_result_to_dict(self.training),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
        }


def run_teacher_bootstrap(
    *,
    run_dir: Path,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    training_config: LinearTrainingConfig,
    train_games: int,
    validation_games: int,
    teacher_policy_spec: str = "scripted-teacher",
    opponent_policy_specs: Iterable[str] = ("simple-legal", "random-legal"),
    seed_start: int = 1,
    validation_seed_start: int = 1_000_000,
    benchmark_games: int = 0,
    benchmark_seed_start: int = 2_000_000,
    worker_count: int = 1,
) -> TeacherBootstrapResult:
    if train_games <= 0:
        raise ValueError("train_games must be positive.")
    if validation_games <= 0:
        raise ValueError("validation_games must be positive.")
    if benchmark_games < 0:
        raise ValueError("benchmark_games must be non-negative.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    opponents = tuple(opponent_policy_specs)
    if not opponents:
        raise ValueError("at least one opponent policy spec is required.")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise ValueError("bootstrap run manifest already exists; choose a new run_dir.")

    full_train_rollout_path = run_dir / "train-full-rollouts.jsonl"
    train_rollout_path = run_dir / "train-rollouts.jsonl"
    full_validation_rollout_path = run_dir / "validation-full-rollouts.jsonl"
    validation_rollout_path = run_dir / "validation-rollouts.jsonl"
    checkpoint_path = run_dir / "linear-bootstrap.json"
    _require_outputs_absent(
        full_train_rollout_path,
        train_rollout_path,
        full_validation_rollout_path,
        validation_rollout_path,
        checkpoint_path,
    )

    train_metrics = collect_selfplay_rollouts(
        output_path=full_train_rollout_path,
        training_output_path=train_rollout_path,
        games=train_games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        current_policy_spec=teacher_policy_spec,
        opponent_policy_specs=opponents,
        worker_count=worker_count,
    )
    validation_metrics = collect_selfplay_rollouts(
        output_path=full_validation_rollout_path,
        training_output_path=validation_rollout_path,
        games=validation_games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=validation_seed_start,
        current_policy_spec=teacher_policy_spec,
        opponent_policy_specs=opponents,
        worker_count=worker_count,
    )
    training = train_linear_policy(
        train_rollout_path,
        config=training_config,
        validation_paths=validation_rollout_path,
    )
    save_linear_model(checkpoint_path, training.model)
    benchmark = None
    if benchmark_games:
        benchmark = _benchmark_bootstrap_checkpoint(
            model_policy=LinearSoftmaxPolicy(model=training.model),
            teacher_policy_spec=teacher_policy_spec,
            env_factory=env_factory,
            rollout_config=rollout_config,
            games=benchmark_games,
            seed_start=benchmark_seed_start,
        )

    result = TeacherBootstrapResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        full_train_rollout_path=full_train_rollout_path,
        train_rollout_path=train_rollout_path,
        full_validation_rollout_path=full_validation_rollout_path,
        validation_rollout_path=validation_rollout_path,
        checkpoint_path=checkpoint_path,
        teacher_policy_spec=teacher_policy_spec,
        opponent_policy_specs=opponents,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        training=training,
        benchmark=benchmark,
    )
    _write_json(manifest_path, result.to_dict())
    return result


def _benchmark_bootstrap_checkpoint(
    *,
    model_policy: LinearSoftmaxPolicy,
    teacher_policy_spec: str,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    games: int,
    seed_start: int,
) -> BenchmarkReport:
    policy_id = str(model_policy.policy_id)
    teacher_policy_as_p1 = policy_from_spec(teacher_policy_spec)
    teacher_policy_as_p2 = policy_from_spec(teacher_policy_spec)
    teacher_policy_id = teacher_policy_as_p1.policy_id
    return benchmark_rollouts(
        games=games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        matchups=(
            BenchmarkMatchup(f"{policy_id} vs random-legal", LinearSoftmaxPolicy(model=model_policy.model), RandomLegalPolicy()),
            BenchmarkMatchup(f"random-legal vs {policy_id}", RandomLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
            BenchmarkMatchup(f"{policy_id} vs simple-legal", LinearSoftmaxPolicy(model=model_policy.model), SimpleLegalPolicy()),
            BenchmarkMatchup(f"simple-legal vs {policy_id}", SimpleLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
            BenchmarkMatchup(f"{policy_id} vs {teacher_policy_id}", LinearSoftmaxPolicy(model=model_policy.model), teacher_policy_as_p2),
            BenchmarkMatchup(f"{teacher_policy_id} vs {policy_id}", teacher_policy_as_p1, LinearSoftmaxPolicy(model=model_policy.model)),
        ),
    )


def _training_result_to_dict(result: LinearTrainingResult) -> dict[str, Any]:
    return {
        "config": {
            "feature_count": result.config.feature_count,
            "window_size": result.config.window_size,
            "discount": result.config.discount,
            "capped_terminal_value": result.config.capped_terminal_value,
            "objective": result.config.objective,
            "epochs": result.config.epochs,
            "learning_rate": result.config.learning_rate,
            "l2": result.config.l2,
            "shuffle_buffer_size": result.config.shuffle_buffer_size,
            "shuffle_seed": result.config.shuffle_seed,
            "max_examples": result.config.max_examples,
            "policy_id": result.config.policy_id,
        },
        "epochs": [metrics.to_dict() for metrics in result.epochs],
        "validation_metrics": result.validation_metrics.to_dict() if result.validation_metrics is not None else None,
        "model": {
            "policy_id": result.model.policy_id,
            "action_schema_version": result.model.action_schema_version,
            "observation_schema_version": result.model.observation_schema_version,
            "feature_schema_version": result.model.feature_schema_version,
            "feature_count": result.model.feature_count,
            "window_size": result.model.window_size,
        },
    }


def _require_outputs_absent(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(f"bootstrap output path(s) already exist: {', '.join(existing)}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)
