"""Neural self-play iteration harness.

This module is the neural counterpart to the dependency-free linear self-play
loop: collect with the current policy, train a neural checkpoint (behavior
cloning or PPO, per the training config objective), benchmark it (including
eval-only references such as max-damage), and write auditable manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from .collection import (
    BenchmarkMatchup,
    BenchmarkReport,
    CollectionMetrics,
    benchmark_rollouts,
    policy_factory_from_spec,
)
from .env import PokeZeroEnv
from .neural_policy import (
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    load_transformer_checkpoint,
    load_transformer_policy,
    require_torch,
    save_transformer_checkpoint,
    _validate_initial_model_config,
    train_transformer_policy,
)
from .opponents import opponent_pool_policy_specs, require_historical_opponent_pool_size
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .run_manifest import auto_promotion_config_dict, opponent_pool_config_dict
from .rollout import RolloutConfig
from .selfplay import POST_ITERATION_AUDIT_FAILURE_MODES, _report_post_iteration_audit_warnings, collect_selfplay_rollouts
from .source_metadata import collect_source_metadata

if TYPE_CHECKING:
    from .evaluation import PromotionGateConfig
    from .promotion import PromotionRecordResult, PromotionRegistryEntry
    from .run_audit import RunAuditConfig, RunAuditResult


NEURAL_SELFPLAY_RUN_SCHEMA_VERSION = "pokezero.neural_selfplay_run.v1"


@dataclass(frozen=True)
class NeuralSelfPlayPromotionConfig:
    registry_path: Path
    gate_config: "PromotionGateConfig"
    artifact_dir: Path | None = None
    label_prefix: str | None = "neural-selfplay"
    notes: str | None = None
    allow_duplicate: bool = False


@dataclass(frozen=True)
class NeuralAdvancementDecision:
    advance_collector: bool
    reason: str
    candidate_policy_id: str
    incumbent_policy_id: str | None = None
    incumbent_policy_spec: str | None = None
    candidate_win_rate: float | None = None
    incumbent_win_rate: float | None = None
    games: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "advance_collector": self.advance_collector,
            "reason": self.reason,
            "candidate_policy_id": self.candidate_policy_id,
            "incumbent_policy_id": self.incumbent_policy_id,
            "incumbent_policy_spec": self.incumbent_policy_spec,
            "candidate_win_rate": self.candidate_win_rate,
            "incumbent_win_rate": self.incumbent_win_rate,
            "games": self.games,
        }


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
    advancement: NeuralAdvancementDecision | None = None
    promotion: "PromotionRecordResult | None" = None
    accepted_policy_spec: str | None = None
    opponent_pool_config: Mapping[str, Any] = field(default_factory=dict)
    invocation_config: Mapping[str, Any] = field(default_factory=dict)
    benchmark_reference_policy_specs: tuple[str, ...] = ()
    source: Mapping[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_policy_spec(self) -> str:
        return f"neural:{self.checkpoint_path}"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
            "iteration": self.iteration,
            "source": dict(self.source),
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
            "seed_start": self.seed_start,
            "worker_count": self.worker_count,
            "collection_metrics": self.metrics.to_dict(),
            "training": _training_result_to_dict(self.training),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
            "advancement": self.advancement.to_dict() if self.advancement is not None else None,
            "promotion": self.promotion.to_dict() if self.promotion is not None else None,
            "next_current_policy_spec": (
                self.accepted_policy_spec or self.checkpoint_policy_spec
                if self.advancement is not None and self.advancement.advance_collector
                else self.current_policy_spec
            ),
        }


@dataclass(frozen=True)
class NeuralSelfPlayRunResult:
    run_dir: Path
    iterations: tuple[NeuralSelfPlayIterationResult, ...]
    prior_iteration_manifests: tuple[Mapping[str, Any], ...] = ()
    invocation_config: Mapping[str, Any] = field(default_factory=dict)
    prior_invocation_configs: tuple[Mapping[str, Any], ...] = ()
    source: Mapping[str, Any] = field(default_factory=dict)

    @property
    def latest_checkpoint_path(self) -> Path | None:
        if not self.iterations:
            if self.prior_iteration_manifests:
                checkpoint_path = self.prior_iteration_manifests[-1].get("checkpoint_path")
                return Path(str(checkpoint_path)) if checkpoint_path is not None else None
            return None
        return self.iterations[-1].checkpoint_path

    @property
    def current_policy_spec(self) -> str | None:
        if self.iterations:
            latest = self.iterations[-1].to_manifest_dict()
            return str(latest["next_current_policy_spec"])
        if self.prior_iteration_manifests:
            latest = self.prior_iteration_manifests[-1]
            current = latest.get("next_current_policy_spec")
            return str(current) if current is not None else str(latest.get("checkpoint_policy_spec"))
        return None

    @property
    def latest_accepted_checkpoint_path(self) -> Path | None:
        current_policy_spec = self.current_policy_spec
        if current_policy_spec is None:
            return None
        body = current_policy_spec.partition("?")[0].strip()
        if not body.lower().startswith("neural:"):
            return None
        checkpoint = body[len("neural:") :].strip()
        return Path(checkpoint) if checkpoint else None

    def to_dict(self) -> dict[str, Any]:
        iteration_manifests = [dict(iteration) for iteration in self.prior_iteration_manifests]
        iteration_manifests.extend(iteration.to_manifest_dict() for iteration in self.iterations)
        invocation_configs = [dict(config) for config in self.prior_invocation_configs]
        if self.invocation_config:
            invocation_configs.append(dict(self.invocation_config))
        return {
            "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
            "run_dir": str(self.run_dir),
            "source": dict(self.source),
            "invocation_configs": invocation_configs,
            "iterations": iteration_manifests,
            "latest_checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path else None,
            "current_policy_spec": self.current_policy_spec,
            "latest_accepted_checkpoint_path": (
                str(self.latest_accepted_checkpoint_path) if self.latest_accepted_checkpoint_path else None
            ),
        }


def load_neural_selfplay_run_manifest(run_dir: Path) -> Mapping[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        iteration_manifests = _load_iteration_manifests(run_dir)
        if not iteration_manifests:
            raise FileNotFoundError(f"Neural self-play run manifest does not exist: {manifest_path}")
        run_result = NeuralSelfPlayRunResult(
            run_dir=run_dir,
            iterations=(),
            prior_iteration_manifests=iteration_manifests,
            prior_invocation_configs=_load_prior_invocation_configs(run_dir),
            source=_mapping(iteration_manifests[-1].get("source", {})),
        )
        return run_result.to_dict()
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
    benchmark_reference_policy_specs: Iterable[str] = (),
    max_historical_opponents: int = 3,
    evaluation_games: int = 0,
    evaluation_seed_start: int = 1_000_000,
    worker_count: int = 1,
    promotion_registry_path: Path | None = None,
    required_promoted_opponent_pool_size: int | None = None,
    auto_promotion_config: NeuralSelfPlayPromotionConfig | None = None,
    post_iteration_audit_config: "RunAuditConfig | None" = None,
    post_iteration_audit_failure_mode: str = "strict",
    tensorboard_log_dir: Path | str | None = None,
    resume: bool = False,
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
    if required_promoted_opponent_pool_size is not None and required_promoted_opponent_pool_size < 0:
        raise ValueError("required_promoted_opponent_pool_size must be non-negative.")
    if iterations > 1 and evaluation_games <= 0:
        raise ValueError("evaluation_games must be positive for multi-iteration neural self-play advancement.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    if post_iteration_audit_failure_mode not in POST_ITERATION_AUDIT_FAILURE_MODES:
        choices = ", ".join(POST_ITERATION_AUDIT_FAILURE_MODES)
        raise ValueError(f"post_iteration_audit_failure_mode must be one of: {choices}.")
    fixed_opponents = tuple(fixed_opponent_policy_specs)
    if not fixed_opponents:
        raise ValueError("at least one fixed opponent policy spec is required.")
    # Eval-only references (e.g. max-damage) are benchmarked against the candidate each
    # iteration but never enter rollout collection or the training opponent pool.
    benchmark_references = tuple(dict.fromkeys(str(spec) for spec in benchmark_reference_policy_specs))
    if model_config.window_size != training_config.window_size:
        raise ValueError("model_config window_size must match training_config window_size.")
    promotion_pool_registry_path = promotion_registry_path or (
        auto_promotion_config.registry_path if auto_promotion_config is not None else None
    )
    source_metadata = collect_source_metadata()
    promoted_checkpoint_specs = list(_promoted_checkpoint_specs(promotion_pool_registry_path))

    run_dir.mkdir(parents=True, exist_ok=True)

    prior_iteration_manifests = _load_prior_iteration_manifests(run_dir, resume=resume)
    prior_invocation_configs = _load_prior_invocation_configs(run_dir) if prior_iteration_manifests else ()
    if prior_iteration_manifests:
        last_iteration = prior_iteration_manifests[-1]
        current_policy_spec = str(last_iteration.get("next_current_policy_spec") or last_iteration["checkpoint_policy_spec"])
        current_model = _initial_neural_model_from_policy_spec(current_policy_spec, device=training_config.device)
        checkpoint_history = [
            str(iteration.get("next_current_policy_spec") or iteration["checkpoint_policy_spec"])
            for iteration in prior_iteration_manifests
            if _advancement_from_manifest(iteration).get("advance_collector")
        ]
        training_rollout_history = [
            Path(str(path))
            for path in _sequence(last_iteration.get("training_rollout_paths", ()))
        ]
        first_iteration = int(last_iteration["iteration"]) + 1
        next_seed_start = int(last_iteration["seed_start"]) + int(last_iteration["collection_metrics"]["games"])
    else:
        current_policy_spec = initial_policy_spec
        current_model = _initial_neural_model_from_policy_spec(current_policy_spec, device=training_config.device)
        checkpoint_history = []
        training_rollout_history = []
        first_iteration = 1
        next_seed_start = seed_start
    if current_model is not None:
        # Fail fast on a warm-start/resume embedding mismatch (e.g. resuming a compact
        # randbat-dex run without re-passing the flag) before collecting any rollouts.
        _validate_initial_model_config(current_model, model_config)
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
        promoted_checkpoint_policy_specs=promoted_checkpoint_specs,
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
        "source": source_metadata,
        "post_iteration_audit_failure_mode": post_iteration_audit_failure_mode,
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
    results: list[NeuralSelfPlayIterationResult] = []
    tensorboard_logger = (
        _TensorBoardLogger(tensorboard_log_dir) if tensorboard_log_dir is not None else None
    )

    try:
        for offset in range(iterations):
            iteration = first_iteration + offset
            iteration_dir = run_dir / f"iteration-{iteration:04d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            rollout_path = iteration_dir / "rollouts.jsonl"
            training_rollout_path = iteration_dir / "training-rollouts.jsonl"
            checkpoint_path = iteration_dir / "transformer-policy.pt"
            iteration_manifest_path = iteration_dir / "manifest.json"
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
            training_rollout_history.append(training_rollout_path)
            iteration_model_config = replace(
                model_config,
                policy_id=f"{model_config.policy_id}-iter-{iteration:04d}",
            )
            model, training = train_transformer_policy(
                tuple(training_rollout_history),
                model_config=iteration_model_config,
                training_config=training_config,
                initial_model=current_model,
            )
            save_transformer_checkpoint(checkpoint_path, model, result=training)
            benchmark = None
            if evaluation_games:
                benchmark_incumbent_policy_spec = _benchmark_incumbent_policy_spec(
                    fallback_policy_spec=current_policy_spec,
                    promotion_config=auto_promotion_config,
                )
                benchmark = _benchmark_checkpoint(
                    checkpoint_path=checkpoint_path,
                    incumbent_policy_spec=benchmark_incumbent_policy_spec,
                    env_factory=env_factory,
                    rollout_config=rollout_config,
                    games=evaluation_games,
                    seed_start=evaluation_seed_start + (offset * evaluation_games),
                    device=training_config.device,
                    benchmark_reference_policy_specs=benchmark_references,
                )
            if auto_promotion_config is None:
                advancement = _advancement_decision(
                    benchmark=benchmark,
                    candidate_policy_id=training.model_config.policy_id,
                    incumbent_policy_spec=current_policy_spec,
                )
            else:
                advancement = NeuralAdvancementDecision(
                    advance_collector=False,
                    reason="pending_promotion_gate",
                    candidate_policy_id=training.model_config.policy_id,
                    incumbent_policy_spec=_benchmark_incumbent_policy_spec(
                        fallback_policy_spec=current_policy_spec,
                        promotion_config=auto_promotion_config,
                    ),
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
                advancement=advancement,
                opponent_pool_config=opponent_pool_manifest_config,
                invocation_config=invocation_config,
                benchmark_reference_policy_specs=benchmark_references,
                source=source_metadata,
            )
            _write_json(iteration_manifest_path, result.to_manifest_dict())
            results.append(result)
            run_manifest_path = run_dir / "manifest.json"
            _write_json(
                run_manifest_path,
                NeuralSelfPlayRunResult(
                    run_dir=run_dir,
                    iterations=tuple(results),
                    prior_iteration_manifests=tuple(prior_iteration_manifests),
                    invocation_config=invocation_config,
                    prior_invocation_configs=prior_invocation_configs,
                    source=source_metadata,
                ).to_dict(),
            )
            post_iteration_audit_result = None
            if (
                post_iteration_audit_config is not None
                and not post_iteration_audit_config.require_latest_promotion
            ):
                post_iteration_audit_result = _enforce_post_iteration_audit(
                    run_manifest_path,
                    post_iteration_audit_config,
                    failure_mode=post_iteration_audit_failure_mode,
                )
            if auto_promotion_config is not None:
                promotion = _record_auto_promotion(
                    manifest_path=run_manifest_path,
                    promotion_config=auto_promotion_config,
                    iteration=iteration,
                )
                accepted_policy_spec = (
                    promotion.registry.selection_checkpoint_policy_spec_for_entry(promotion.entry)
                    if promotion.recorded and promotion.entry is not None
                    else None
                )
                advancement = _promotion_advancement_decision(
                    promotion=promotion,
                    candidate_policy_id=training.model_config.policy_id,
                    incumbent_policy_spec=_benchmark_incumbent_policy_spec(
                        fallback_policy_spec=current_policy_spec,
                        promotion_config=auto_promotion_config,
                    ),
                )
                result = replace(
                    result,
                    promotion=promotion,
                    advancement=advancement,
                    accepted_policy_spec=accepted_policy_spec,
                )
                results[-1] = result
                _write_json(iteration_manifest_path, result.to_manifest_dict())
                if promotion.recorded and promotion_pool_registry_path == auto_promotion_config.registry_path:
                    promoted_checkpoint_specs = list(promotion.registry.selection_checkpoint_policy_specs())
            if advancement.advance_collector:
                next_policy_spec = result.to_manifest_dict()["next_current_policy_spec"]
                checkpoint_history.append(str(next_policy_spec))
                current_policy_spec = str(next_policy_spec)
                current_model = model
            if tensorboard_logger is not None:
                tensorboard_logger.log(
                    _tensorboard_scalars(
                        candidate_policy_id=training.model_config.policy_id,
                        training=training,
                        benchmark=benchmark,
                        advancement=advancement,
                    ),
                    step=iteration,
                )
            _write_json(
                run_dir / "manifest.json",
                NeuralSelfPlayRunResult(
                    run_dir=run_dir,
                    iterations=tuple(results),
                    prior_iteration_manifests=tuple(prior_iteration_manifests),
                    invocation_config=invocation_config,
                    prior_invocation_configs=prior_invocation_configs,
                    source=source_metadata,
                ).to_dict(),
            )
            if (
                post_iteration_audit_config is not None
                and (
                    post_iteration_audit_config.require_latest_promotion
                    or auto_promotion_config is not None
                )
            ):
                post_iteration_audit_result = _enforce_post_iteration_audit(
                    run_manifest_path,
                    post_iteration_audit_config,
                    failure_mode=post_iteration_audit_failure_mode,
                )
            _report_post_iteration_audit_warnings(
                post_iteration_audit_result,
                failure_mode=post_iteration_audit_failure_mode,
            )

    finally:
        if tensorboard_logger is not None:
            tensorboard_logger.close()

    return NeuralSelfPlayRunResult(
        run_dir=run_dir,
        iterations=tuple(results),
        prior_iteration_manifests=tuple(prior_iteration_manifests),
        invocation_config=invocation_config,
        prior_invocation_configs=prior_invocation_configs,
        source=source_metadata,
    )


def _enforce_post_iteration_audit(
    manifest_path: Path,
    config: "RunAuditConfig | None",
    *,
    failure_mode: str = "strict",
) -> "RunAuditResult | None":
    if config is None:
        return None
    from .run_audit import RunAuditFailure, audit_run, runtime_health_failed_check_names

    result = audit_run(manifest_path, config=config)
    if result.passed:
        return result
    if failure_mode == "runtime-health":
        failed_names = tuple(check.name for check in result.blocking_failed_checks)
        if not runtime_health_failed_check_names(failed_names):
            return result
    raise RunAuditFailure(result)


def _tensorboard_scalars(
    *,
    candidate_policy_id: str,
    training: TransformerTrainingResult,
    benchmark: "BenchmarkReport | None",
    advancement: "NeuralAdvancementDecision | None",
) -> dict[str, float]:
    """Flatten one iteration's training + benchmark metrics into TensorBoard scalars.

    Pure and torch-free so it can be unit tested without a SummaryWriter.
    """
    scalars: dict[str, float] = {}
    if training.epochs:
        last = training.epochs[-1]
        if last.loss is not None:
            scalars["train/loss"] = float(last.loss)
        if last.policy_accuracy is not None:
            scalars["train/policy_accuracy"] = float(last.policy_accuracy)
        if last.value_loss is not None:
            scalars["train/value_loss"] = float(last.value_loss)
    if benchmark is not None:
        for result in benchmark.head_to_head_results:
            if result.first_policy_id == candidate_policy_id:
                opponent, win_rate = result.second_policy_id, result.first_policy_win_rate
            elif result.second_policy_id == candidate_policy_id:
                opponent, win_rate = result.first_policy_id, result.second_policy_win_rate
            else:
                continue
            scalars[f"winrate/{opponent}"] = float(win_rate)
    if advancement is not None:
        scalars["train/advanced"] = 1.0 if advancement.advance_collector else 0.0
    return scalars


class _TensorBoardLogger:
    """Thin lazy wrapper around torch's SummaryWriter (imported on demand)."""

    def __init__(self, log_dir: Path | str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, scalars: Mapping[str, float], *, step: int) -> None:
        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _benchmark_checkpoint(
    *,
    checkpoint_path: Path,
    incumbent_policy_spec: str,
    env_factory: Callable[[], PokeZeroEnv],
    rollout_config: RolloutConfig,
    games: int,
    seed_start: int,
    device: str | None,
    benchmark_reference_policy_specs: tuple[str, ...] = (),
) -> BenchmarkReport:
    model_policy = load_transformer_policy(checkpoint_path, deterministic=True, device=device)
    policy_id = str(model_policy.policy_id)
    incumbent_policy = _policy_from_spec_for_evaluation(incumbent_policy_spec, device=device)
    incumbent_policy_id = str(incumbent_policy.policy_id)
    incumbent_matchups: tuple[BenchmarkMatchup, ...] = ()
    if incumbent_policy_id not in {policy_id, "random-legal", "simple-legal"}:
        incumbent_matchups = (
            BenchmarkMatchup(f"{policy_id} vs {incumbent_policy_id}", model_policy, incumbent_policy),
            BenchmarkMatchup(
                f"{incumbent_policy_id} vs {policy_id}",
                _policy_from_spec_for_evaluation(incumbent_policy_spec, device=device),
                model_policy,
            ),
        )
    # Eval-only reference matchups (e.g. max-damage). Resolved fresh per orientation so each
    # side gets its own policy instance, and skipped if the reference id collides with the
    # candidate or a built-in baseline already covered above.
    covered_ids = {policy_id, "random-legal", "simple-legal", incumbent_policy_id}
    reference_matchups: list[BenchmarkMatchup] = []
    for reference_spec in benchmark_reference_policy_specs:
        reference_policy = _policy_from_spec_for_evaluation(reference_spec, device=device)
        reference_id = str(reference_policy.policy_id)
        if reference_id in covered_ids:
            continue
        covered_ids.add(reference_id)
        reference_matchups.append(
            BenchmarkMatchup(f"{policy_id} vs {reference_id}", model_policy, reference_policy)
        )
        reference_matchups.append(
            BenchmarkMatchup(
                f"{reference_id} vs {policy_id}",
                _policy_from_spec_for_evaluation(reference_spec, device=device),
                model_policy,
            )
        )
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
            *incumbent_matchups,
            *reference_matchups,
        ),
    )


def _promoted_checkpoint_specs(promotion_registry_path: Path | None) -> tuple[str, ...]:
    if promotion_registry_path is None:
        return ()
    from .promotion import load_promotion_registry, verify_promotion_registry

    verification = verify_promotion_registry(promotion_registry_path)
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


def _benchmark_incumbent_policy_spec(
    *,
    fallback_policy_spec: str,
    promotion_config: NeuralSelfPlayPromotionConfig | None,
) -> str:
    if promotion_config is None:
        return fallback_policy_spec
    from .promotion import load_promotion_registry

    registry = load_promotion_registry(promotion_config.registry_path)
    entry = _promotion_incumbent_entry_from_registry(registry, promotion_config)
    if entry is None or entry.checkpoint_policy_spec is None:
        return fallback_policy_spec
    return registry.selection_checkpoint_policy_spec_for_entry(entry) or fallback_policy_spec


def _record_auto_promotion(
    *,
    manifest_path: Path,
    promotion_config: NeuralSelfPlayPromotionConfig,
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


def _promotion_incumbent_entry(
    promotion_config: NeuralSelfPlayPromotionConfig,
) -> "PromotionRegistryEntry | None":
    from .promotion import load_promotion_registry

    registry = load_promotion_registry(promotion_config.registry_path)
    return _promotion_incumbent_entry_from_registry(registry, promotion_config)


def _promotion_incumbent_entry_from_registry(
    registry,
    promotion_config: NeuralSelfPlayPromotionConfig,
) -> "PromotionRegistryEntry | None":
    incumbent_policy_id = promotion_config.gate_config.incumbent_policy_id
    if incumbent_policy_id is None:
        return registry.latest
    for entry in reversed(registry.entries):
        if entry.policy_id == incumbent_policy_id:
            return entry
    return None


def _promotion_advancement_decision(
    *,
    promotion: "PromotionRecordResult",
    candidate_policy_id: str,
    incumbent_policy_spec: str,
) -> NeuralAdvancementDecision:
    gate_result = promotion.gate_result
    return NeuralAdvancementDecision(
        advance_collector=promotion.recorded,
        reason="promotion_recorded" if promotion.recorded else "promotion_gate_failed",
        candidate_policy_id=candidate_policy_id,
        incumbent_policy_id=gate_result.incumbent_policy_id,
        incumbent_policy_spec=incumbent_policy_spec,
        candidate_win_rate=gate_result.incumbent_win_rate,
        incumbent_win_rate=None,
        games=gate_result.incumbent_games,
    )


def _advancement_decision(
    *,
    benchmark: BenchmarkReport | None,
    candidate_policy_id: str,
    incumbent_policy_spec: str,
) -> NeuralAdvancementDecision:
    if benchmark is None:
        return NeuralAdvancementDecision(
            advance_collector=False,
            reason="not_evaluated",
            candidate_policy_id=candidate_policy_id,
            incumbent_policy_spec=incumbent_policy_spec,
        )
    incumbent_policy_id = str(_policy_from_spec_for_evaluation(incumbent_policy_spec, device=None).policy_id)
    for result in benchmark.head_to_head_results:
        ids = {result.first_policy_id, result.second_policy_id}
        if ids != {candidate_policy_id, incumbent_policy_id}:
            continue
        if result.first_policy_id == candidate_policy_id:
            candidate_win_rate = result.first_policy_win_rate
            incumbent_win_rate = result.second_policy_win_rate
        else:
            candidate_win_rate = result.second_policy_win_rate
            incumbent_win_rate = result.first_policy_win_rate
        advance = candidate_win_rate > incumbent_win_rate
        return NeuralAdvancementDecision(
            advance_collector=advance,
            reason="beat_incumbent" if advance else "failed_to_beat_incumbent",
            candidate_policy_id=candidate_policy_id,
            incumbent_policy_id=incumbent_policy_id,
            incumbent_policy_spec=incumbent_policy_spec,
            candidate_win_rate=candidate_win_rate,
            incumbent_win_rate=incumbent_win_rate,
            games=result.games,
        )
    return NeuralAdvancementDecision(
        advance_collector=False,
        reason="missing_incumbent_benchmark",
        candidate_policy_id=candidate_policy_id,
        incumbent_policy_id=incumbent_policy_id,
        incumbent_policy_spec=incumbent_policy_spec,
    )


def _initial_neural_model_from_policy_spec(policy_spec: str, *, device: str | None):
    policy_body = policy_spec.strip().partition("?")[0].strip()
    if not policy_body.lower().startswith("neural:"):
        return None
    checkpoint = policy_body[len("neural:") :].strip()
    if not checkpoint:
        return None
    model, _ = load_transformer_checkpoint(Path(checkpoint), map_location=device)
    return model


def _policy_from_spec_for_evaluation(policy_spec: str, *, device: str | None):
    body, options = _split_policy_spec_options(policy_spec)
    if body.lower().startswith("neural:"):
        checkpoint = body[len("neural:") :].strip()
        if not checkpoint:
            raise ValueError("neural policy spec must include a checkpoint path after 'neural:'.")
        return load_transformer_policy(
            Path(checkpoint),
            deterministic=True,
            exploration_epsilon=0.0,
            device=options.get("device") or device,
        )
    return policy_factory_from_spec(_deterministic_policy_spec(policy_spec))()


def _deterministic_policy_spec(policy_spec: str) -> str:
    body, options = _split_policy_spec_options(policy_spec)
    lowered = body.lower()
    if not (lowered.startswith("linear:") or lowered.startswith("neural:")):
        return policy_spec
    options.pop("sample", None)
    options["deterministic"] = "true"
    options["epsilon"] = "0.0"
    from urllib.parse import urlencode

    return f"{body}?{urlencode(options)}"


def _split_policy_spec_options(policy_spec: str) -> tuple[str, dict[str, str]]:
    body, separator, query = policy_spec.strip().partition("?")
    if not separator:
        return body, {}
    from urllib.parse import parse_qsl

    return body, {key: value for key, value in parse_qsl(query, keep_blank_values=True)}


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
                raise ValueError("run_dir already contains neural iteration manifests; pass resume=True to continue it.")
            return iteration_manifests
        if list(run_dir.glob("iteration-*")):
            if not resume:
                raise ValueError("run_dir already contains iteration directories; pass resume=True to inspect or continue it.")
            raise ValueError("cannot resume: run directory contains no completed neural iteration manifest.")
        if resume:
            raise ValueError("cannot resume: neural self-play run manifest does not exist.")
        return ()
    if not resume:
        raise ValueError("neural self-play run manifest already exists; pass resume=True to continue it.")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported neural self-play run schema: {manifest.get('schema_version')!r}.")
    iterations = tuple(_mapping(iteration) for iteration in _sequence(manifest.get("iterations", ())))
    if not iterations:
        raise ValueError("cannot resume: neural self-play run manifest contains no iterations.")
    return iterations


def _load_iteration_manifests(run_dir: Path) -> tuple[Mapping[str, Any], ...]:
    manifests: list[Mapping[str, Any]] = []
    for manifest_path in sorted(run_dir.glob("iteration-*/manifest.json")):
        manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.get("schema_version") != NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
            raise ValueError(f"Unsupported neural self-play iteration schema: {manifest.get('schema_version')!r}.")
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
    for manifest_path in sorted(run_dir.glob("iteration-*/manifest.json")):
        manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.get("schema_version") != NEURAL_SELFPLAY_RUN_SCHEMA_VERSION:
            raise ValueError(f"Unsupported neural self-play iteration schema: {manifest.get('schema_version')!r}.")
        config = manifest.get("invocation_config")
        if config is None:
            continue
        mapped = _mapping(config)
        configs_by_fingerprint.setdefault(json.dumps(mapped, sort_keys=True), mapped)
    return tuple(configs_by_fingerprint.values())


def _advancement_from_manifest(iteration: Mapping[str, Any]) -> Mapping[str, Any]:
    advancement = iteration.get("advancement")
    if advancement is None:
        return {}
    return _mapping(advancement)


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
