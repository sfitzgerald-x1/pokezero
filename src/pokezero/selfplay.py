"""Self-play iteration harness for dependency-free policy experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from .collection import (
    BenchmarkMatchup,
    BenchmarkReport,
    CollectionMetrics,
    RolloutRecord,
    benchmark_rollouts,
    iter_rollout_records,
    policy_from_spec,
    run_rollout_record,
    summarize_records,
    write_rollout_record,
)
from .env import PokeZeroEnv
from .linear_policy import (
    LinearPolicyModel,
    LinearSoftmaxPolicy,
    LinearTrainingConfig,
    LinearTrainingResult,
    load_linear_model,
    save_linear_model,
    train_linear_policy,
)
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .rollout import RolloutConfig
from .trajectory import BattleTrajectory

SELFPLAY_RUN_SCHEMA_VERSION = "pokezero.selfplay_run.v1"


@dataclass(frozen=True)
class SelfPlayIterationResult:
    iteration: int
    rollout_path: Path
    training_rollout_path: Path
    checkpoint_path: Path
    manifest_path: Path
    current_policy_spec: str
    opponent_policy_specs: tuple[str, ...]
    training_rollout_paths: tuple[Path, ...]
    seed_start: int
    metrics: CollectionMetrics
    training: LinearTrainingResult
    benchmark: BenchmarkReport | None = None

    @property
    def checkpoint_policy_spec(self) -> str:
        return f"linear:{self.checkpoint_path}"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "iteration": self.iteration,
            "rollout_path": str(self.rollout_path),
            "training_rollout_path": str(self.training_rollout_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_policy_spec": self.checkpoint_policy_spec,
            "current_policy_spec": self.current_policy_spec,
            "opponent_policy_specs": list(self.opponent_policy_specs),
            "training_rollout_paths": [str(path) for path in self.training_rollout_paths],
            "seed_start": self.seed_start,
            "collection_metrics": self.metrics.to_dict(),
            "training": _training_result_to_dict(self.training),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
        }


@dataclass(frozen=True)
class SelfPlayRunResult:
    run_dir: Path
    iterations: tuple[SelfPlayIterationResult, ...]
    prior_iteration_manifests: tuple[Mapping[str, Any], ...] = ()

    @property
    def latest_checkpoint_path(self) -> Path | None:
        if not self.iterations:
            if self.prior_iteration_manifests:
                checkpoint_path = self.prior_iteration_manifests[-1].get("checkpoint_path")
                return Path(str(checkpoint_path)) if checkpoint_path is not None else None
            return None
        return self.iterations[-1].checkpoint_path

    def to_dict(self) -> dict[str, Any]:
        iteration_manifests = [dict(iteration) for iteration in self.prior_iteration_manifests]
        iteration_manifests.extend(iteration.to_manifest_dict() for iteration in self.iterations)
        return {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "iterations": iteration_manifests,
            "latest_checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path else None,
        }


def run_selfplay_iterations(
    *,
    run_dir: Path,
    iterations: int,
    games_per_iteration: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    training_config: LinearTrainingConfig,
    seed_start: int = 1,
    initial_policy_spec: str = "random-legal",
    fixed_opponent_policy_specs: Iterable[str] = ("random-legal", "simple-legal"),
    max_historical_opponents: int = 3,
    evaluation_games: int = 0,
    evaluation_seed_start: int = 1_000_000,
    resume: bool = False,
) -> SelfPlayRunResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive.")
    if max_historical_opponents < 0:
        raise ValueError("max_historical_opponents must be non-negative.")
    if evaluation_games < 0:
        raise ValueError("evaluation_games must be non-negative.")

    run_dir.mkdir(parents=True, exist_ok=True)
    fixed_opponents = tuple(fixed_opponent_policy_specs)
    if not fixed_opponents:
        raise ValueError("at least one fixed opponent policy spec is required.")

    prior_iteration_manifests = _load_prior_iteration_manifests(run_dir, resume=resume)
    current_policy_spec = initial_policy_spec
    current_model = _initial_model_from_policy_spec(initial_policy_spec)
    checkpoint_history: list[str] = []
    training_rollout_history: list[Path] = []
    first_iteration = 1
    next_seed_start = seed_start
    if prior_iteration_manifests:
        last_iteration = prior_iteration_manifests[-1]
        current_policy_spec = str(last_iteration["checkpoint_policy_spec"])
        current_model = load_linear_model(Path(str(last_iteration["checkpoint_path"])))
        checkpoint_history = [str(iteration["checkpoint_policy_spec"]) for iteration in prior_iteration_manifests]
        training_rollout_history = [
            Path(str(path))
            for path in _sequence(last_iteration.get("training_rollout_paths", ()))
        ]
        first_iteration = int(last_iteration["iteration"]) + 1
        next_seed_start = int(last_iteration["seed_start"]) + int(last_iteration["collection_metrics"]["games"])
    results: list[SelfPlayIterationResult] = []

    for offset in range(iterations):
        iteration = first_iteration + offset
        iteration_dir = run_dir / f"iteration-{iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = iteration_dir / "rollouts.jsonl"
        training_rollout_path = iteration_dir / "training-rollouts.jsonl"
        checkpoint_path = iteration_dir / "linear-policy.json"
        manifest_path = iteration_dir / "manifest.json"
        iteration_seed_start = next_seed_start + (offset * games_per_iteration)
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
        )
        iteration_training_config = replace(
            training_config,
            policy_id=f"{training_config.policy_id}-iter-{iteration:04d}",
        )
        training_rollout_history.append(training_rollout_path)
        training = train_linear_policy(
            tuple(training_rollout_history),
            config=iteration_training_config,
            initial_model=current_model,
        )
        save_linear_model(checkpoint_path, training.model)
        benchmark = None
        if evaluation_games:
            benchmark = _benchmark_checkpoint(
                model_policy=LinearSoftmaxPolicy(model=training.model),
                env_factory=env_factory,
                rollout_config=rollout_config,
                games=evaluation_games,
                seed_start=evaluation_seed_start + ((iteration - 1) * evaluation_games),
            )

        result = SelfPlayIterationResult(
            iteration=iteration,
            rollout_path=rollout_path,
            training_rollout_path=training_rollout_path,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            current_policy_spec=current_policy_spec,
            opponent_policy_specs=opponent_policy_specs,
            training_rollout_paths=tuple(training_rollout_history),
            seed_start=iteration_seed_start,
            metrics=metrics,
            training=training,
            benchmark=benchmark,
        )
        _write_json(manifest_path, result.to_manifest_dict())
        results.append(result)
        checkpoint_history.append(result.checkpoint_policy_spec)
        current_policy_spec = result.checkpoint_policy_spec
        current_model = training.model

    run_result = SelfPlayRunResult(
        run_dir=run_dir,
        iterations=tuple(results),
        prior_iteration_manifests=tuple(prior_iteration_manifests),
    )
    _write_json(run_dir / "manifest.json", run_result.to_dict())
    return run_result


def collect_selfplay_rollouts(
    *,
    output_path: Path,
    training_output_path: Path | None = None,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    seed_start: int,
    current_policy_spec: str,
    opponent_policy_specs: Iterable[str],
) -> CollectionMetrics:
    if games <= 0:
        raise ValueError("games must be positive.")
    opponent_specs = tuple(opponent_policy_specs)
    if not opponent_specs:
        raise ValueError("at least one opponent policy spec is required.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_path = output_path.with_name(f".{output_path.name}.tmp")
    training_write_path = None
    if training_output_path is not None:
        training_output_path.parent.mkdir(parents=True, exist_ok=True)
        training_write_path = training_output_path.with_name(f".{training_output_path.name}.tmp")
    collection_start = perf_counter()
    try:
        with write_path.open("w", encoding="utf-8") as handle:
            training_handle = training_write_path.open("w", encoding="utf-8") if training_write_path is not None else None
            try:
                _collect_selfplay_records(
                    handle=handle,
                    training_handle=training_handle,
                    games=games,
                    env_factory=env_factory,
                    rollout_config=rollout_config,
                    seed_start=seed_start,
                    current_policy_spec=current_policy_spec,
                    opponent_specs=opponent_specs,
                )
            finally:
                if training_handle is not None:
                    training_handle.close()
        write_path.replace(output_path)
        if training_write_path is not None and training_output_path is not None:
            training_write_path.replace(training_output_path)
    except Exception:
        write_path.unlink(missing_ok=True)
        if training_write_path is not None:
            training_write_path.unlink(missing_ok=True)
        raise
    return summarize_records(
        iter_rollout_records(output_path),
        elapsed_seconds=perf_counter() - collection_start,
    )


def _collect_selfplay_records(
    *,
    handle,
    training_handle,
    games: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    seed_start: int,
    current_policy_spec: str,
    opponent_specs: tuple[str, ...],
) -> None:
    for game_index in range(games):
        seed = seed_start + game_index
        opponent_spec = opponent_specs[game_index % len(opponent_specs)]
        p1_spec, p2_spec = _seat_policy_specs(
            current_policy_spec=current_policy_spec,
            opponent_policy_spec=opponent_spec,
            game_index=game_index,
        )
        current_player = "p1" if game_index % 2 == 0 else "p2"
        record = run_rollout_record(
            env_factory=env_factory,
            policies={
                "p1": policy_from_spec(p1_spec),
                "p2": policy_from_spec(p2_spec),
            },
            rollout_config=rollout_config,
            seed=seed,
            battle_id=f"selfplay-{seed}",
        )
        write_rollout_record(handle, record)
        if training_handle is not None:
            write_rollout_record(training_handle, _record_for_player(record, current_player))


def _record_for_player(record: RolloutRecord, player_id: str) -> RolloutRecord:
    trajectory = BattleTrajectory(
        battle_id=record.trajectory.battle_id,
        format_id=record.trajectory.format_id,
        seed=record.trajectory.seed,
        metadata=dict(record.trajectory.metadata),
    )
    for step in record.trajectory.steps:
        if step.player_id == player_id:
            trajectory.append(step)
    if record.trajectory.terminal is not None:
        trajectory.record_terminal(record.trajectory.terminal)
    return RolloutRecord(
        battle_id=record.battle_id,
        seed=record.seed,
        format_id=record.format_id,
        policy_ids={player_id: str(record.policy_ids[player_id])},
        decision_round_count=len(trajectory.steps),
        elapsed_seconds=record.elapsed_seconds,
        terminal=record.terminal,
        trajectory=trajectory,
    )


def _opponent_pool(
    *,
    fixed_policy_specs: tuple[str, ...],
    checkpoint_history: list[str],
    current_policy_spec: str,
    max_historical_opponents: int,
) -> tuple[str, ...]:
    historical = [spec for spec in checkpoint_history if spec != current_policy_spec]
    if max_historical_opponents:
        historical = historical[-max_historical_opponents:]
    else:
        historical = []
    return fixed_policy_specs + tuple(historical)


def _initial_model_from_policy_spec(policy_spec: str) -> LinearPolicyModel | None:
    policy = policy_from_spec(policy_spec)
    if isinstance(policy, LinearSoftmaxPolicy):
        return policy.model
    return None


def _load_prior_iteration_manifests(
    run_dir: Path,
    *,
    resume: bool,
) -> tuple[Mapping[str, Any], ...]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        if resume:
            raise ValueError("cannot resume: run manifest does not exist.")
        return ()
    if not resume:
        raise ValueError("run_dir already contains a manifest; pass resume=True to continue it.")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != SELFPLAY_RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported self-play run schema: {manifest.get('schema_version')!r}.")
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        raise ValueError("cannot resume: run manifest contains no iterations.")
    return iterations


def _seat_policy_specs(
    *,
    current_policy_spec: str,
    opponent_policy_spec: str,
    game_index: int,
) -> tuple[str, str]:
    if game_index % 2 == 0:
        return current_policy_spec, opponent_policy_spec
    return opponent_policy_spec, current_policy_spec


def _benchmark_checkpoint(
    *,
    model_policy: LinearSoftmaxPolicy,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    games: int,
    seed_start: int,
) -> BenchmarkReport:
    policy_id = str(model_policy.policy_id)
    return benchmark_rollouts(
        games=games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        matchups=(
            BenchmarkMatchup(f"{policy_id} vs random-legal", model_policy, RandomLegalPolicy()),
            BenchmarkMatchup(f"random-legal vs {policy_id}", RandomLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
            BenchmarkMatchup(f"{policy_id} vs simple-legal", LinearSoftmaxPolicy(model=model_policy.model), SimpleLegalPolicy()),
            BenchmarkMatchup(f"simple-legal vs {policy_id}", SimpleLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
        ),
    )


def _training_result_to_dict(result: LinearTrainingResult) -> dict[str, Any]:
    return {
        "config": {
            "feature_count": result.config.feature_count,
            "window_size": result.config.window_size,
            "discount": result.config.discount,
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
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("expected JSON array payload.")
    return tuple(value)
