"""Self-play iteration harness for dependency-free policy experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from .collection import (
    BenchmarkMatchup,
    BenchmarkReport,
    CollectionMetrics,
    NEURAL_POLICY_SPEC_PREFIX,
    RolloutRecord,
    benchmark_rollouts,
    iter_rollout_records,
    policy_factory_from_spec,
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
from .opponents import opponent_pool_policy_specs, require_historical_opponent_pool_size
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .run_manifest import auto_promotion_config_dict, opponent_pool_config_dict
from .rollout import RolloutConfig
from .trajectory import BattleTrajectory

if TYPE_CHECKING:
    from .evaluation import PromotionGateConfig
    from .promotion import PromotionRecordResult
    from .run_audit import RunAuditConfig, RunAuditResult

SELFPLAY_RUN_SCHEMA_VERSION = "pokezero.selfplay_run.v1"


@dataclass(frozen=True)
class SelfPlayPromotionConfig:
    registry_path: Path
    gate_config: "PromotionGateConfig"
    artifact_dir: Path | None = None
    label_prefix: str | None = "selfplay"
    notes: str | None = None
    allow_duplicate: bool = False


@dataclass(frozen=True)
class SelfPlayIterationResult:
    iteration: int
    rollout_path: Path
    training_rollout_path: Path
    checkpoint_path: Path
    manifest_path: Path
    current_policy_spec: str
    opponent_policy_specs: tuple[str, ...]
    benchmark_reference_policy_specs: tuple[str, ...]
    training_rollout_paths: tuple[Path, ...]
    validation_rollout_paths: tuple[Path, ...]
    seed_start: int
    worker_count: int
    metrics: CollectionMetrics
    training: LinearTrainingResult
    benchmark: BenchmarkReport | None = None
    promotion: "PromotionRecordResult | None" = None
    opponent_pool_config: Mapping[str, Any] = field(default_factory=dict)
    invocation_config: Mapping[str, Any] = field(default_factory=dict)

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
            "opponent_pool_config": dict(self.opponent_pool_config),
            "invocation_config": dict(self.invocation_config),
            "benchmark_reference_policy_specs": list(self.benchmark_reference_policy_specs),
            "training_rollout_paths": [str(path) for path in self.training_rollout_paths],
            "validation_rollout_paths": [str(path) for path in self.validation_rollout_paths],
            "seed_start": self.seed_start,
            "worker_count": self.worker_count,
            "collection_metrics": self.metrics.to_dict(),
            "training": _training_result_to_dict(self.training),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
            "promotion": self.promotion.to_dict() if self.promotion is not None else None,
        }


@dataclass(frozen=True)
class SelfPlayRunResult:
    run_dir: Path
    iterations: tuple[SelfPlayIterationResult, ...]
    prior_iteration_manifests: tuple[Mapping[str, Any], ...] = ()
    invocation_config: Mapping[str, Any] = field(default_factory=dict)
    prior_invocation_configs: tuple[Mapping[str, Any], ...] = ()

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
        invocation_configs = [dict(config) for config in self.prior_invocation_configs]
        if self.invocation_config:
            invocation_configs.append(dict(self.invocation_config))
        return {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "invocation_configs": invocation_configs,
            "iterations": iteration_manifests,
            "latest_checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path else None,
        }


def load_selfplay_run_manifest(run_dir: Path) -> Mapping[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        iteration_manifests = _load_iteration_manifests(run_dir)
        if not iteration_manifests:
            raise FileNotFoundError(f"Self-play run manifest does not exist: {manifest_path}")
        return {
            "schema_version": SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "iterations": list(iteration_manifests),
            "latest_checkpoint_path": str(iteration_manifests[-1].get("checkpoint_path")),
        }
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != SELFPLAY_RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported self-play run schema: {manifest.get('schema_version')!r}.")
    _sequence(manifest.get("iterations", ()))
    return manifest


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
    benchmark_reference_policy_specs: Iterable[str] | None = None,
    max_historical_opponents: int = 3,
    evaluation_games: int = 0,
    evaluation_seed_start: int = 1_000_000,
    validation_rollout_paths: Iterable[Path] | None = None,
    promotion_registry_path: Path | None = None,
    required_promoted_opponent_pool_size: int | None = None,
    auto_promotion_config: SelfPlayPromotionConfig | None = None,
    post_iteration_audit_config: "RunAuditConfig | None" = None,
    resume: bool = False,
    worker_count: int = 1,
) -> SelfPlayRunResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive.")
    if max_historical_opponents < 0:
        raise ValueError("max_historical_opponents must be non-negative.")
    if evaluation_games < 0:
        raise ValueError("evaluation_games must be non-negative.")
    if required_promoted_opponent_pool_size is not None and required_promoted_opponent_pool_size < 0:
        raise ValueError("required_promoted_opponent_pool_size must be non-negative.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")

    run_dir.mkdir(parents=True, exist_ok=True)
    fixed_opponents = tuple(fixed_opponent_policy_specs)
    if not fixed_opponents:
        raise ValueError("at least one fixed opponent policy spec is required.")
    explicit_benchmark_references = (
        ()
        if benchmark_reference_policy_specs is None
        else tuple(dict.fromkeys(str(spec) for spec in benchmark_reference_policy_specs))
    )
    validation_paths = tuple(Path(path) for path in (validation_rollout_paths or ()))
    promotion_pool_registry_path = promotion_registry_path or (
        auto_promotion_config.registry_path if auto_promotion_config is not None else None
    )
    promoted_checkpoint_specs = list(_promoted_checkpoint_specs(promotion_pool_registry_path))

    checkpoint_history: list[str] = []
    training_rollout_history: list[Path] = []
    first_iteration = 1
    next_seed_start = seed_start
    prior_iteration_manifests = _load_prior_iteration_manifests(run_dir, resume=resume)
    prior_invocation_configs = _load_prior_invocation_configs(run_dir) if prior_iteration_manifests else ()
    if prior_iteration_manifests:
        last_iteration = prior_iteration_manifests[-1]
        current_policy_spec = str(last_iteration["checkpoint_policy_spec"])
        benchmark_references = _dedupe_policy_specs(
            (
                *_benchmark_reference_policy_specs_from_manifest_history(prior_iteration_manifests),
                *explicit_benchmark_references,
            )
        )
        current_model = load_linear_model(Path(str(last_iteration["checkpoint_path"])))
        _validate_training_config_matches_model(training_config, current_model)
        checkpoint_history = [str(iteration["checkpoint_policy_spec"]) for iteration in prior_iteration_manifests]
        training_rollout_history = [
            Path(str(path))
            for path in _sequence(last_iteration.get("training_rollout_paths", ()))
        ]
        if not validation_paths:
            validation_paths = tuple(
                Path(str(path))
                for path in _sequence(last_iteration.get("validation_rollout_paths", ()))
            )
        first_iteration = int(last_iteration["iteration"]) + 1
        next_seed_start = int(last_iteration["seed_start"]) + int(last_iteration["collection_metrics"]["games"])
    else:
        if _is_neural_policy_spec(initial_policy_spec):
            raise ValueError(
                "self-play iterate currently trains linear checkpoints; neural: initial policies are not "
                "supported until a neural self-play training path exists. Use neural_cli benchmark for "
                "neural checkpoint evaluation."
            )
        current_policy_spec = initial_policy_spec
        benchmark_references = _dedupe_policy_specs(
            (
                *_default_benchmark_reference_policy_specs(
                    initial_policy_spec=initial_policy_spec,
                ),
                *explicit_benchmark_references,
            )
        )
        current_model = _initial_model_from_policy_spec(initial_policy_spec)
        if current_model is not None:
            _validate_training_config_matches_model(training_config, current_model)
    _validate_validation_rollout_paths(validation_paths)
    _require_promoted_opponent_pool(
        promoted_checkpoint_specs,
        promotion_pool_registry_path=promotion_pool_registry_path,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        required_size=required_promoted_opponent_pool_size,
    )
    opponent_pool_manifest_config = opponent_pool_config_dict(
        fixed_opponent_policy_specs=fixed_opponents,
        max_historical_opponents=max_historical_opponents,
        promotion_registry_path=promotion_registry_path,
        promotion_pool_registry_path=promotion_pool_registry_path,
        required_promoted_opponent_pool_size=required_promoted_opponent_pool_size,
    )
    invocation_config = {
        "resume": bool(prior_iteration_manifests),
        "first_iteration": first_iteration,
        "iterations_requested": iterations,
        "games_per_iteration": games_per_iteration,
        "seed_start_argument": seed_start,
        "first_iteration_seed_start": next_seed_start,
        "initial_policy_spec": initial_policy_spec,
        "evaluation_games": evaluation_games,
        "evaluation_seed_start": evaluation_seed_start,
        "worker_count": worker_count,
        "validation_rollout_paths": [str(path) for path in validation_paths],
        "benchmark_reference_policy_specs": list(benchmark_references),
        "opponent_pool": opponent_pool_manifest_config,
        "auto_promotion": auto_promotion_config_dict(
            enabled=auto_promotion_config is not None,
            registry_path=auto_promotion_config.registry_path if auto_promotion_config is not None else None,
            artifact_dir=auto_promotion_config.artifact_dir if auto_promotion_config is not None else None,
            label_prefix=auto_promotion_config.label_prefix if auto_promotion_config is not None else None,
            notes=auto_promotion_config.notes if auto_promotion_config is not None else None,
            allow_duplicate=auto_promotion_config.allow_duplicate if auto_promotion_config is not None else False,
        ),
    }
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
            checkpoint_history=promoted_checkpoint_specs if promotion_pool_registry_path is not None else checkpoint_history,
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
        iteration_training_config = replace(
            training_config,
            policy_id=f"{training_config.policy_id}-iter-{iteration:04d}",
        )
        training_rollout_history.append(training_rollout_path)
        training = train_linear_policy(
            tuple(training_rollout_history),
            config=iteration_training_config,
            initial_model=current_model,
            validation_paths=validation_paths or None,
        )
        save_linear_model(checkpoint_path, training.model)
        benchmark = None
        if evaluation_games:
            benchmark_incumbent_policy_spec = _benchmark_incumbent_policy_spec(
                fallback_policy_spec=current_policy_spec,
                promotion_config=auto_promotion_config,
            )
            benchmark = _benchmark_checkpoint(
                model_policy=LinearSoftmaxPolicy(model=training.model),
                incumbent_policy_spec=benchmark_incumbent_policy_spec,
                reference_policy_specs=benchmark_references,
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
            benchmark_reference_policy_specs=benchmark_references,
            training_rollout_paths=tuple(training_rollout_history),
            validation_rollout_paths=validation_paths,
            seed_start=iteration_seed_start,
            worker_count=worker_count,
            metrics=metrics,
            training=training,
            benchmark=benchmark,
            opponent_pool_config=opponent_pool_manifest_config,
            invocation_config=invocation_config,
        )
        _write_json(manifest_path, result.to_manifest_dict())
        results.append(result)
        run_manifest_path = run_dir / "manifest.json"
        # The promotion gate consumes the top-level run-manifest shape, not the
        # per-iteration manifest, so write it before evaluating auto-promotion.
        _write_json(
            run_manifest_path,
            SelfPlayRunResult(
                run_dir=run_dir,
                iterations=tuple(results),
                prior_iteration_manifests=tuple(prior_iteration_manifests),
                invocation_config=invocation_config,
                prior_invocation_configs=prior_invocation_configs,
            ).to_dict(),
        )
        if auto_promotion_config is not None:
            promotion = _record_auto_promotion(
                manifest_path=run_manifest_path,
                promotion_config=auto_promotion_config,
                iteration=iteration,
            )
            result = replace(result, promotion=promotion)
            results[-1] = result
            _write_json(manifest_path, result.to_manifest_dict())
            if promotion.recorded and promotion_pool_registry_path == auto_promotion_config.registry_path:
                promoted_checkpoint_specs = list(_promoted_checkpoint_specs(promotion_pool_registry_path))
        checkpoint_history.append(result.checkpoint_policy_spec)
        current_policy_spec = result.checkpoint_policy_spec
        current_model = training.model
        _write_json(
            run_dir / "manifest.json",
            SelfPlayRunResult(
                run_dir=run_dir,
                iterations=tuple(results),
                prior_iteration_manifests=tuple(prior_iteration_manifests),
                invocation_config=invocation_config,
                prior_invocation_configs=prior_invocation_configs,
            ).to_dict(),
        )
        _enforce_post_iteration_audit(run_manifest_path, post_iteration_audit_config)

    run_result = SelfPlayRunResult(
        run_dir=run_dir,
        iterations=tuple(results),
        prior_iteration_manifests=tuple(prior_iteration_manifests),
        invocation_config=invocation_config,
        prior_invocation_configs=prior_invocation_configs,
    )
    _write_json(run_dir / "manifest.json", run_result.to_dict())
    return run_result


def _enforce_post_iteration_audit(
    manifest_path: Path,
    config: "RunAuditConfig | None",
) -> "RunAuditResult | None":
    if config is None:
        return None
    from .run_audit import enforce_run_audit

    return enforce_run_audit(manifest_path, config=config)


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
    worker_count: int = 1,
) -> CollectionMetrics:
    if games <= 0:
        raise ValueError("games must be positive.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    opponent_specs = tuple(opponent_policy_specs)
    if not opponent_specs:
        raise ValueError("at least one opponent policy spec is required.")
    policy_factories = _policy_factories_for_specs((current_policy_spec, *opponent_specs))
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
                    policy_factories=policy_factories,
                    worker_count=worker_count,
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
    policy_factories: Mapping[str, Callable[[], Any]],
    worker_count: int,
) -> None:
    if worker_count == 1:
        results = (
            _run_selfplay_game_record(
                game_index=game_index,
                seed_start=seed_start,
                env_factory=env_factory,
                rollout_config=rollout_config,
                current_policy_spec=current_policy_spec,
                opponent_specs=opponent_specs,
                policy_factories=policy_factories,
            )
            for game_index in range(games)
        )
        _write_selfplay_game_results(handle=handle, training_handle=training_handle, results=results)
        return

    max_workers = min(worker_count, games)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pokezero-selfplay") as executor:
        results = executor.map(
            lambda game_index: _run_selfplay_game_record(
                game_index=game_index,
                seed_start=seed_start,
                env_factory=env_factory,
                rollout_config=rollout_config,
                current_policy_spec=current_policy_spec,
                opponent_specs=opponent_specs,
                policy_factories=policy_factories,
            ),
            range(games),
        )
        _write_selfplay_game_results(handle=handle, training_handle=training_handle, results=results)


def _write_selfplay_game_results(
    *,
    handle,
    training_handle,
    results: Iterable[tuple[RolloutRecord, RolloutRecord]],
) -> None:
    for record, training_record in results:
        write_rollout_record(handle, record)
        if training_handle is not None:
            write_rollout_record(training_handle, training_record)


def _run_selfplay_game_record(
    *,
    game_index: int,
    seed_start: int,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    current_policy_spec: str,
    opponent_specs: tuple[str, ...],
    policy_factories: Mapping[str, Callable[[], Any]],
) -> tuple[RolloutRecord, RolloutRecord]:
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
            "p1": policy_factories[p1_spec](),
            "p2": policy_factories[p2_spec](),
        },
        rollout_config=rollout_config,
        seed=seed,
        battle_id=f"selfplay-{seed}",
    )
    return record, _record_for_player(record, current_player)


def _policy_factories_for_specs(specs: Iterable[str]) -> Mapping[str, Callable[[], Any]]:
    return {spec: policy_factory_from_spec(spec) for spec in dict.fromkeys(specs)}


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
    checkpoint_history: Iterable[str],
    current_policy_spec: str,
    max_historical_opponents: int,
) -> tuple[str, ...]:
    return opponent_pool_policy_specs(
        fixed_policy_specs=fixed_policy_specs,
        checkpoint_history=checkpoint_history,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
    )


def _promoted_checkpoint_specs(promotion_registry_path: Path | None) -> tuple[str, ...]:
    if promotion_registry_path is None:
        return ()
    from .promotion import load_promotion_registry, verify_promotion_registry

    verification = verify_promotion_registry(promotion_registry_path, verify_loadable=True)
    if not verification.passed:
        failed = ", ".join(check.name for check in verification.checks if not check.passed)
        raise ValueError(f"promotion registry verification failed before selection: {failed}")

    return load_promotion_registry(promotion_registry_path).selection_checkpoint_policy_specs()


def _require_promoted_opponent_pool(
    promoted_checkpoint_specs: Iterable[str],
    *,
    promotion_pool_registry_path: Path | None,
    current_policy_spec: str,
    max_historical_opponents: int,
    required_size: int | None,
) -> None:
    if required_size is None:
        return
    if promotion_pool_registry_path is None:
        raise ValueError("required promoted opponent pool size requires a promotion registry.")
    require_historical_opponent_pool_size(
        promoted_checkpoint_specs,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        required_size=required_size,
        pool_label="promoted opponent pool",
    )


def _default_benchmark_reference_policy_specs(
    *,
    initial_policy_spec: str,
) -> tuple[str, ...]:
    if not _is_linear_policy_spec(initial_policy_spec):
        return ()
    return (initial_policy_spec,)


def _dedupe_policy_specs(policy_specs: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(spec) for spec in policy_specs))


def _benchmark_reference_policy_specs_from_manifest_history(
    manifests: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    for manifest in manifests:
        references = tuple(str(spec) for spec in _sequence(manifest.get("benchmark_reference_policy_specs", ())))
        if references:
            return references
    first = manifests[0]
    initial_policy_spec = str(first.get("current_policy_spec", ""))
    return _default_benchmark_reference_policy_specs(
        initial_policy_spec=initial_policy_spec,
    )


def _benchmark_incumbent_policy_spec(
    *,
    fallback_policy_spec: str,
    promotion_config: SelfPlayPromotionConfig | None,
) -> str:
    if promotion_config is None:
        return fallback_policy_spec
    from .promotion import load_promotion_registry

    registry = load_promotion_registry(promotion_config.registry_path)
    entry = _promotion_incumbent_entry_from_registry(registry, promotion_config)
    if entry is None or not entry.checkpoint_path:
        return fallback_policy_spec
    return registry.selection_checkpoint_policy_spec_for_entry(entry) or fallback_policy_spec


def _record_auto_promotion(
    *,
    manifest_path: Path,
    promotion_config: SelfPlayPromotionConfig,
    iteration: int,
) -> "PromotionRecordResult":
    from .promotion import record_promotion

    gate_config = promotion_config.gate_config
    if gate_config.incumbent_policy_id is None:
        latest = _promotion_incumbent_entry(promotion_config)
        if latest is not None and latest.policy_id:
            gate_config = replace(gate_config, incumbent_policy_id=latest.policy_id)
    label = (
        f"{promotion_config.label_prefix}-{iteration:04d}"
        if promotion_config.label_prefix
        else None
    )
    return record_promotion(
        manifest_path,
        registry_path=promotion_config.registry_path,
        config=gate_config,
        label=label,
        notes=promotion_config.notes,
        artifact_dir=promotion_config.artifact_dir,
        allow_duplicate=promotion_config.allow_duplicate,
    )


def _promotion_incumbent_entry(promotion_config: SelfPlayPromotionConfig):
    from .promotion import load_promotion_registry

    registry = load_promotion_registry(promotion_config.registry_path)
    return _promotion_incumbent_entry_from_registry(registry, promotion_config)


def _promotion_incumbent_entry_from_registry(registry, promotion_config: SelfPlayPromotionConfig):
    incumbent_policy_id = promotion_config.gate_config.incumbent_policy_id
    if incumbent_policy_id is None:
        return registry.latest
    for entry in reversed(registry.entries):
        if entry.policy_id == incumbent_policy_id:
            return entry
    return None


def _initial_model_from_policy_spec(policy_spec: str) -> LinearPolicyModel | None:
    policy = policy_from_spec(policy_spec)
    if isinstance(policy, LinearSoftmaxPolicy):
        return policy.model
    return None


def _is_neural_policy_spec(policy_spec: str) -> bool:
    policy_body = policy_spec.strip().partition("?")[0].strip().lower()
    return policy_body.startswith(NEURAL_POLICY_SPEC_PREFIX)


def _is_linear_policy_spec(policy_spec: str) -> bool:
    policy_body = policy_spec.strip().partition("?")[0].strip().lower()
    return policy_body.startswith("linear:")


def _load_prior_iteration_manifests(
    run_dir: Path,
    *,
    resume: bool,
) -> tuple[Mapping[str, Any], ...]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        iteration_manifests = _load_iteration_manifests(run_dir)
        if iteration_manifests:
            if not resume:
                raise ValueError("run_dir already contains iteration manifests; pass resume=True to continue it.")
            return iteration_manifests
        if list(run_dir.glob("iteration-*")):
            if not resume:
                raise ValueError("run_dir already contains iteration directories; pass resume=True to inspect or continue it.")
            raise ValueError("cannot resume: run directory contains no completed iteration manifests.")
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


def _load_iteration_manifests(run_dir: Path) -> tuple[Mapping[str, Any], ...]:
    manifests: list[Mapping[str, Any]] = []
    for manifest_path in sorted(run_dir.glob("iteration-*/manifest.json")):
        manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.get("schema_version") != SELFPLAY_RUN_SCHEMA_VERSION:
            raise ValueError(f"Unsupported self-play iteration schema: {manifest.get('schema_version')!r}.")
        manifests.append(manifest)
    return tuple(manifests)


def _load_prior_invocation_configs(run_dir: Path) -> tuple[Mapping[str, Any], ...]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
        configs = manifest.get("invocation_configs")
        if configs is not None:
            return tuple(_mapping(config) for config in _sequence(configs))
        legacy_config = manifest.get("run_config")
        if legacy_config is not None:
            return (_mapping(legacy_config),)
    configs_by_fingerprint: dict[str, Mapping[str, Any]] = {}
    for iteration in _load_iteration_manifests(run_dir):
        config = iteration.get("invocation_config")
        if config is None:
            continue
        mapped = _mapping(config)
        configs_by_fingerprint.setdefault(json.dumps(mapped, sort_keys=True), mapped)
    return tuple(configs_by_fingerprint.values())


def _validate_training_config_matches_model(
    training_config: LinearTrainingConfig,
    model: LinearPolicyModel,
) -> None:
    if training_config.feature_count != model.feature_count:
        raise ValueError("training_config feature_count must match the resumed checkpoint.")
    if training_config.window_size != model.window_size:
        raise ValueError("training_config window_size must match the resumed checkpoint.")


def _validate_validation_rollout_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Validation rollout path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Validation rollout path must be a file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Validation rollout path is empty: {path}")


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
    incumbent_policy_spec: str | None = None,
    reference_policy_specs: Iterable[str] = (),
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    games: int,
    seed_start: int,
) -> BenchmarkReport:
    policy_id = str(model_policy.policy_id)
    incumbent_matchups = _incumbent_benchmark_matchups(
        model_policy=model_policy,
        incumbent_policy_spec=incumbent_policy_spec,
    )
    excluded_reference_policy_ids = {
        policy_id,
        "random-legal",
        "simple-legal",
        *(
            policy_id
            for matchup in incumbent_matchups
            for policy_id in (matchup.p1_policy.policy_id, matchup.p2_policy.policy_id)
        ),
    }
    matchups = (
        BenchmarkMatchup(f"{policy_id} vs random-legal", model_policy, RandomLegalPolicy()),
        BenchmarkMatchup(f"random-legal vs {policy_id}", RandomLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
        BenchmarkMatchup(f"{policy_id} vs simple-legal", LinearSoftmaxPolicy(model=model_policy.model), SimpleLegalPolicy()),
        BenchmarkMatchup(f"simple-legal vs {policy_id}", SimpleLegalPolicy(), LinearSoftmaxPolicy(model=model_policy.model)),
        *incumbent_matchups,
        *_reference_benchmark_matchups(
            model_policy=model_policy,
            reference_policy_specs=reference_policy_specs,
            excluded_policy_ids=excluded_reference_policy_ids,
        ),
    )
    return benchmark_rollouts(
        games=games,
        env_factory=env_factory,
        rollout_config=rollout_config,
        seed_start=seed_start,
        matchups=matchups,
    )


def _incumbent_benchmark_matchups(
    *,
    model_policy: LinearSoftmaxPolicy,
    incumbent_policy_spec: str | None,
) -> tuple[BenchmarkMatchup, ...]:
    if incumbent_policy_spec is None:
        return ()
    policy_id = str(model_policy.policy_id)
    incumbent_factory = policy_factory_from_spec(incumbent_policy_spec)
    incumbent_policy = incumbent_factory()
    incumbent_policy_id = str(incumbent_policy.policy_id)
    if incumbent_policy_id in {policy_id, "random-legal", "simple-legal"}:
        return ()
    return (
        BenchmarkMatchup(
            f"{policy_id} vs {incumbent_policy_id}",
            LinearSoftmaxPolicy(model=model_policy.model),
            incumbent_policy,
        ),
        BenchmarkMatchup(
            f"{incumbent_policy_id} vs {policy_id}",
            incumbent_factory(),
            LinearSoftmaxPolicy(model=model_policy.model),
        ),
    )


def _reference_benchmark_matchups(
    *,
    model_policy: LinearSoftmaxPolicy,
    reference_policy_specs: Iterable[str],
    excluded_policy_ids: Iterable[str],
) -> tuple[BenchmarkMatchup, ...]:
    policy_id = str(model_policy.policy_id)
    seen_policy_ids = set(excluded_policy_ids)
    matchups: list[BenchmarkMatchup] = []
    for reference_policy_spec in reference_policy_specs:
        reference_factory = policy_factory_from_spec(reference_policy_spec)
        reference_policy = reference_factory()
        reference_policy_id = str(reference_policy.policy_id)
        if reference_policy_id in seen_policy_ids:
            continue
        seen_policy_ids.add(reference_policy_id)
        matchups.extend(
            (
                BenchmarkMatchup(
                    f"{policy_id} vs {reference_policy_id}",
                    LinearSoftmaxPolicy(model=model_policy.model),
                    reference_policy,
                ),
                BenchmarkMatchup(
                    f"{reference_policy_id} vs {policy_id}",
                    reference_factory(),
                    LinearSoftmaxPolicy(model=model_policy.model),
                ),
            )
        )
    return tuple(matchups)


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
            "feature_fingerprint": result.model.feature_fingerprint,
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
