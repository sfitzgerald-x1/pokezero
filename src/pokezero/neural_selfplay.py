"""Neural self-play iteration harness.

This module is the neural counterpart to the dependency-free linear self-play
loop: collect with the current policy, train a neural checkpoint (behavior
cloning or PPO, per the training config objective), benchmark it (including
eval-only references such as max-damage), and write auditable manifests.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from .collection import (
    BenchmarkMatchup,
    BenchmarkReport,
    CollectionMetrics,
    benchmark_rollouts,
    policy_factory_from_spec,
    _split_policy_spec_options,
)
from .env import PokeZeroEnv
from .neural_policy import (
    CONSTANT_LEARNING_RATE_SCHEDULE,
    TransformerPolicyConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    load_transformer_checkpoint,
    load_transformer_policy,
    require_torch,
    resolve_torch_device,
    save_transformer_checkpoint,
    _validate_initial_model_config,
    train_transformer_policy,
)
from .dataset import (
    MAX_ACTIVE_TRAINING_CACHE_BYTES,
    TrajectoryDatasetConfig,
    delete_training_cache_path,
    training_cache_paths_byte_size,
)
from .opponents import (
    HISTORICAL_OPPONENT_SELECTION_MODES,
    opponent_pool_policy_specs,
    require_historical_opponent_pool_size,
)
from .policy import RandomLegalPolicy, SimpleLegalPolicy
from .run_manifest import auto_promotion_config_dict, opponent_pool_config_dict
from .rollout import RolloutConfig
from .selfplay import POST_ITERATION_AUDIT_FAILURE_MODES, _report_post_iteration_audit_warnings, collect_selfplay_rollouts
from .source_metadata import collect_source_metadata
from .value_calibration import (
    VALUE_SELECTION_METRICS,
    evaluate_value_calibration,
    value_selection_metric_direction,
    value_selection_metric_value,
    value_selection_score,
)

if TYPE_CHECKING:
    from .evaluation import PromotionGateConfig
    from .promotion import PromotionRecordResult, PromotionRegistryEntry
    from .run_audit import RunAuditConfig, RunAuditResult


NEURAL_SELFPLAY_RUN_SCHEMA_VERSION = "pokezero.neural_selfplay_run.v1"
COLLECTOR_ADVANCEMENT_MODES = ("incumbent-gate", "always", "yardstick-gate")
ACCEPTED_ADVANCEMENT_REASONS = frozenset(
    {
        "beat_incumbent",
        "promotion_recorded",
        "beat_yardstick_best",
        "yardstick_baseline_initialized",
    }
)
DEFAULT_COLLECTOR_YARDSTICK_POLICY_ID = "max-damage"


@dataclass(frozen=True)
class NeuralSelfPlayPromotionConfig:
    registry_path: Path
    gate_config: "PromotionGateConfig"
    artifact_dir: Path | None = None
    label_prefix: str | None = "neural-selfplay"
    notes: str | None = None
    allow_duplicate: bool = False


@dataclass(frozen=True)
class NeuralValueCalibrationConfig:
    scope: str = "iteration"
    batch_size: int = 128
    bins: int = 10

    def __post_init__(self) -> None:
        if self.scope not in {"iteration", "history"}:
            raise ValueError("value calibration scope must be 'iteration' or 'history'.")
        if self.batch_size <= 0:
            raise ValueError("value calibration batch_size must be positive.")
        if self.bins <= 0:
            raise ValueError("value calibration bins must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "batch_size": self.batch_size,
            "bins": self.bins,
        }


@dataclass(frozen=True)
class NeuralValueSelectionConfig:
    scope: str = "iteration"
    metric: str = "mae"
    batch_size: int = 128
    bins: int = 10
    heldout_games_per_iteration: int = 0
    heldout_seed_start: int = 2_000_000

    def __post_init__(self) -> None:
        if self.scope not in {"iteration", "history"}:
            raise ValueError("value selection scope must be 'iteration' or 'history'.")
        if self.metric not in VALUE_SELECTION_METRICS:
            choices = ", ".join(VALUE_SELECTION_METRICS)
            raise ValueError(f"value selection metric must be one of: {choices}.")
        if self.batch_size <= 0:
            raise ValueError("value selection batch_size must be positive.")
        if self.bins <= 0:
            raise ValueError("value selection bins must be positive.")
        if self.heldout_games_per_iteration < 0:
            raise ValueError("value selection heldout_games_per_iteration must be non-negative.")
        if self.heldout_seed_start < 0:
            raise ValueError("value selection heldout_seed_start must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "metric": self.metric,
            "batch_size": self.batch_size,
            "bins": self.bins,
            "heldout_games_per_iteration": self.heldout_games_per_iteration,
            "heldout_seed_start": self.heldout_seed_start,
        }


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
    yardstick_policy_id: str | None = None
    yardstick_win_rate: float | None = None
    previous_best_yardstick_win_rate: float | None = None

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
            "yardstick_policy_id": self.yardstick_policy_id,
            "yardstick_win_rate": self.yardstick_win_rate,
            "previous_best_yardstick_win_rate": self.previous_best_yardstick_win_rate,
        }


@dataclass(frozen=True)
class NeuralSelfPlayIterationResult:
    iteration: int
    rollout_path: Path | None
    training_rollout_path: Path | None
    value_selection_rollout_path: Path | None
    value_selection_training_rollout_path: Path | None
    value_selection_seed_start: int | None
    value_selection_next_seed_start: int | None
    checkpoint_path: Path
    manifest_path: Path
    current_policy_spec: str
    opponent_policy_specs: tuple[str, ...]
    training_rollout_paths: tuple[Path, ...]
    training_input_paths: tuple[Path, ...]
    value_selection_training_rollout_paths: tuple[Path, ...]
    seed_start: int
    worker_count: int
    metrics: CollectionMetrics
    value_selection_metrics: CollectionMetrics | None
    training: TransformerTrainingResult
    training_elapsed_seconds: float
    training_input_bytes: int | None = None
    checkpoint_bytes: int | None = None
    benchmark: BenchmarkReport | None = None
    advancement: NeuralAdvancementDecision | None = None
    promotion: "PromotionRecordResult | None" = None
    accepted_policy_spec: str | None = None
    opponent_pool_config: Mapping[str, Any] = field(default_factory=dict)
    invocation_config: Mapping[str, Any] = field(default_factory=dict)
    benchmark_reference_policy_specs: tuple[str, ...] = ()
    training_cache_paths: tuple[Path, ...] = ()
    training_cache_deleted_after_train: bool = False
    training_cache_deleted_bytes: int = 0
    value_calibration: Mapping[str, Any] | None = None
    value_selection: Mapping[str, Any] | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_policy_spec(self) -> str:
        return f"neural:{self.checkpoint_path}"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEURAL_SELFPLAY_RUN_SCHEMA_VERSION,
            "iteration": self.iteration,
            "source": dict(self.source),
            "rollout_path": str(self.rollout_path) if self.rollout_path is not None else None,
            "training_rollout_path": (
                str(self.training_rollout_path) if self.training_rollout_path is not None else None
            ),
            "value_selection_rollout_path": (
                str(self.value_selection_rollout_path) if self.value_selection_rollout_path is not None else None
            ),
            "value_selection_training_rollout_path": (
                str(self.value_selection_training_rollout_path)
                if self.value_selection_training_rollout_path is not None
                else None
            ),
            "value_selection_seed_start": self.value_selection_seed_start,
            "value_selection_next_seed_start": self.value_selection_next_seed_start,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_policy_spec": self.checkpoint_policy_spec,
            "current_policy_spec": self.current_policy_spec,
            "opponent_policy_specs": list(self.opponent_policy_specs),
            "opponent_pool_config": dict(self.opponent_pool_config),
            "invocation_config": dict(self.invocation_config),
            "benchmark_reference_policy_specs": list(self.benchmark_reference_policy_specs),
            "training_rollout_paths": [str(path) for path in self.training_rollout_paths],
            "training_input_paths": [str(path) for path in self.training_input_paths],
            "value_selection_training_rollout_paths": [
                str(path) for path in self.value_selection_training_rollout_paths
            ],
            "training_cache_paths": [str(path) for path in self.training_cache_paths],
            "training_cache_deleted_after_train": self.training_cache_deleted_after_train,
            "training_cache_deleted_bytes": self.training_cache_deleted_bytes,
            "training_input_bytes": self.training_input_bytes,
            "training_elapsed_seconds": self.training_elapsed_seconds,
            "checkpoint_bytes": self.checkpoint_bytes,
            "seed_start": self.seed_start,
            "worker_count": self.worker_count,
            "collection_metrics": self.metrics.to_dict(),
            "value_selection_collection_metrics": (
                self.value_selection_metrics.to_dict() if self.value_selection_metrics is not None else None
            ),
            "training": _training_result_to_dict(self.training),
            "value_calibration": dict(self.value_calibration) if self.value_calibration is not None else None,
            "value_selection": dict(self.value_selection) if self.value_selection is not None else None,
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
        iteration_manifests = [dict(iteration) for iteration in self.prior_iteration_manifests]
        iteration_manifests.extend(iteration.to_manifest_dict() for iteration in self.iterations)
        for iteration in reversed(iteration_manifests):
            advancement = _optional_mapping(iteration.get("advancement"))
            if not advancement.get("advance_collector"):
                continue
            if advancement.get("reason") not in ACCEPTED_ADVANCEMENT_REASONS:
                continue
            policy_spec = str(iteration.get("next_current_policy_spec") or iteration.get("checkpoint_policy_spec"))
            checkpoint = _neural_checkpoint_path_from_policy_spec(policy_spec)
            if checkpoint is not None:
                return checkpoint
        for config in (*self.prior_invocation_configs, self.invocation_config):
            checkpoint = _neural_checkpoint_path_from_policy_spec(str(config.get("initial_policy_spec", "")))
            if checkpoint is not None:
                return checkpoint
        return None

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
    mirror_match: bool = False,
    collection_temperature: float = 1.0,
    max_historical_opponents: int = 3,
    historical_opponent_selection: str = "recent",
    evaluation_games: int = 0,
    evaluation_interval_games: int | None = None,
    evaluation_seed_start: int = 1_000_000,
    worker_count: int = 1,
    promotion_registry_path: Path | None = None,
    required_promoted_opponent_pool_size: int | None = None,
    auto_promotion_config: NeuralSelfPlayPromotionConfig | None = None,
    post_iteration_audit_config: "RunAuditConfig | None" = None,
    post_iteration_audit_failure_mode: str = "strict",
    value_calibration_config: NeuralValueCalibrationConfig | None = None,
    value_selection_config: NeuralValueSelectionConfig | None = None,
    collector_advancement_mode: str = "incumbent-gate",
    experiment_preset: str = "none",
    tensorboard_log_dir: Path | str | None = None,
    training_cache_root: Path | None = None,
    training_cache_chunk_games: int | None = None,
    training_cache_max_root_bytes: int | None = MAX_ACTIVE_TRAINING_CACHE_BYTES,
    delete_training_cache_after_train: bool = True,
    write_rollout_jsonl: bool = True,
    learning_rate_schedule_completed_games: int | None = None,
    resume: bool = False,
) -> NeuralSelfPlayRunResult:
    require_torch()
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive.")
    if max_historical_opponents < 0:
        raise ValueError("max_historical_opponents must be non-negative.")
    if historical_opponent_selection not in HISTORICAL_OPPONENT_SELECTION_MODES:
        choices = ", ".join(HISTORICAL_OPPONENT_SELECTION_MODES)
        raise ValueError(f"historical_opponent_selection must be one of: {choices}.")
    if evaluation_games < 0:
        raise ValueError("evaluation_games must be non-negative.")
    if evaluation_interval_games is not None and evaluation_interval_games <= 0:
        raise ValueError("evaluation_interval_games must be positive when set.")
    if required_promoted_opponent_pool_size is not None and required_promoted_opponent_pool_size < 0:
        raise ValueError("required_promoted_opponent_pool_size must be non-negative.")
    if iterations > 1 and evaluation_games <= 0:
        raise ValueError("evaluation_games must be positive for multi-iteration neural self-play advancement.")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    if collection_temperature <= 0.0:
        raise ValueError("collection_temperature must be positive.")
    if post_iteration_audit_failure_mode not in POST_ITERATION_AUDIT_FAILURE_MODES:
        choices = ", ".join(POST_ITERATION_AUDIT_FAILURE_MODES)
        raise ValueError(f"post_iteration_audit_failure_mode must be one of: {choices}.")
    if collector_advancement_mode not in COLLECTOR_ADVANCEMENT_MODES:
        choices = ", ".join(COLLECTOR_ADVANCEMENT_MODES)
        raise ValueError(f"collector_advancement_mode must be one of: {choices}.")
    if collector_advancement_mode != "incumbent-gate" and auto_promotion_config is not None:
        raise ValueError(f"collector_advancement_mode={collector_advancement_mode!r} cannot be combined with auto promotion.")
    if (
        evaluation_interval_games is not None
        and evaluation_interval_games > games_per_iteration
        and collector_advancement_mode != "always"
    ):
        raise ValueError(
            "evaluation_interval_games can skip iteration benchmarks only when "
            "collector_advancement_mode='always'."
        )
    if training_cache_chunk_games is not None and training_cache_chunk_games <= 0:
        raise ValueError("training_cache_chunk_games must be positive when set.")
    if not write_rollout_jsonl and training_cache_root is None:
        raise ValueError("write_rollout_jsonl=False requires training_cache_root.")
    if training_cache_root is not None and training_cache_max_root_bytes is not None and training_cache_max_root_bytes <= 0:
        raise ValueError("training_cache_max_root_bytes must be positive when set.")
    if learning_rate_schedule_completed_games is not None and learning_rate_schedule_completed_games < 0:
        raise ValueError("learning_rate_schedule_completed_games must be non-negative.")
    if (
        training_cache_root is not None
        and training_config.objective != "ppo"
        and delete_training_cache_after_train
    ):
        raise ValueError(
            "delete_training_cache_after_train=True with training_cache_root requires objective='ppo' "
            "because non-PPO objectives replay historical training data. Use objective='ppo' or "
            "delete_training_cache_after_train=False."
        )
    if (
        training_cache_root is not None
        and delete_training_cache_after_train
        and value_calibration_config is not None
        and value_calibration_config.scope == "history"
    ):
        raise ValueError(
            "history-scoped value calibration cannot delete per-iteration training caches after each train step."
        )
    if (
        training_cache_root is not None
        and delete_training_cache_after_train
        and value_selection_config is not None
        and value_selection_config.scope == "history"
        and value_selection_config.heldout_games_per_iteration == 0
    ):
        raise ValueError(
            "history-scoped value selection on training data cannot delete per-iteration training caches after each train step."
        )
    fixed_opponents = tuple(fixed_opponent_policy_specs)
    if not fixed_opponents and not mirror_match:
        raise ValueError("at least one fixed opponent policy spec is required unless mirror_match=True.")
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
    learning_rate_schedule_completed_games = _resolve_learning_rate_schedule_completed_games(
        learning_rate_schedule_completed_games,
        prior_invocation_configs=prior_invocation_configs,
    )
    _validate_learning_rate_schedule_window(
        training_config,
        completed_games_offset=learning_rate_schedule_completed_games,
        requested_games=iterations * games_per_iteration,
    )
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
        value_selection_rollout_history = [
            Path(str(path))
            for path in _sequence(last_iteration.get("value_selection_training_rollout_paths", ()))
        ]
        next_value_selection_seed_start = _next_value_selection_seed_start(
            last_iteration,
            default_seed_start=(
                value_selection_config.heldout_seed_start
                if value_selection_config is not None
                else 2_000_000
            ),
        )
        first_iteration = int(last_iteration["iteration"]) + 1
        next_seed_start = int(last_iteration["seed_start"]) + int(last_iteration["collection_metrics"]["games"])
        next_evaluation_seed_start = _next_evaluation_seed_start(
            prior_iteration_manifests,
            default_seed_start=evaluation_seed_start,
        )
    else:
        current_policy_spec = initial_policy_spec
        current_model = _initial_neural_model_from_policy_spec(current_policy_spec, device=training_config.device)
        checkpoint_history = []
        training_rollout_history = []
        value_selection_rollout_history = []
        next_value_selection_seed_start = (
            value_selection_config.heldout_seed_start
            if value_selection_config is not None
            else 2_000_000
        )
        first_iteration = 1
        next_seed_start = seed_start
        next_evaluation_seed_start = evaluation_seed_start
    training_input_path_byte_sizes: dict[Path, int | None] = {}
    if current_model is not None:
        # Fail fast on a warm-start/resume embedding mismatch (e.g. resuming a compact
        # randbat-dex run without re-passing the flag) before collecting any rollouts.
        _validate_initial_model_config(current_model, model_config)
    _require_promoted_opponent_pool(
        promoted_checkpoint_specs,
        promotion_pool_registry_path=promotion_pool_registry_path,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        historical_opponent_selection=historical_opponent_selection,
        required_size=required_promoted_opponent_pool_size,
    )
    opponent_pool_manifest_config = opponent_pool_config_dict(
        fixed_opponent_policy_specs=fixed_opponents,
        max_historical_opponents=max_historical_opponents,
        historical_opponent_selection=historical_opponent_selection,
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
        "evaluation_interval_games": evaluation_interval_games,
        "evaluation_seed_start": evaluation_seed_start,
        "first_iteration_evaluation_seed_start": next_evaluation_seed_start,
        "worker_count": worker_count,
        "source": source_metadata,
        "post_iteration_audit_failure_mode": post_iteration_audit_failure_mode,
        "benchmark_reference_policy_specs": list(benchmark_references),
        "mirror_match": mirror_match,
        "collection_temperature": collection_temperature,
        "collector_advancement_mode": collector_advancement_mode,
        "experiment_preset": experiment_preset,
        "training_config": training_config.to_dict(),
        "training_cache": {
            "root": str(training_cache_root) if training_cache_root is not None else None,
            "chunk_games": training_cache_chunk_games,
            "max_root_bytes": training_cache_max_root_bytes,
            "delete_after_train": delete_training_cache_after_train,
            "write_rollout_jsonl": write_rollout_jsonl,
        },
        "learning_rate_schedule_completed_games": learning_rate_schedule_completed_games,
        "value_calibration": value_calibration_config.to_dict() if value_calibration_config is not None else None,
        "value_selection": value_selection_config.to_dict() if value_selection_config is not None else None,
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
    next_benchmark_seed_start = next_evaluation_seed_start
    tensorboard_logger = (
        _TensorBoardLogger(tensorboard_log_dir) if tensorboard_log_dir is not None else None
    )

    try:
        for offset in range(iterations):
            iteration = first_iteration + offset
            iteration_dir = run_dir / f"iteration-{iteration:04d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            rollout_path = iteration_dir / "rollouts.jsonl" if write_rollout_jsonl else None
            training_rollout_path = None if training_cache_root is not None else iteration_dir / "training-rollouts.jsonl"
            value_selection_rollout_path = None
            value_selection_training_rollout_path = None
            value_selection_seed_start = None
            value_selection_next_seed_start = None
            checkpoint_path = iteration_dir / "transformer-policy.pt"
            iteration_manifest_path = iteration_dir / "manifest.json"
            iteration_seed_start = next_seed_start + (offset * games_per_iteration)
            # Higher-temperature spec used only for collection so the collector explores; the
            # canonical current_policy_spec stays clean for benchmark/advancement/manifest. The
            # mirror opponent (built from this spec) inherits the same exploration temperature.
            collection_current_policy_spec = _with_collection_temperature(
                current_policy_spec, collection_temperature
            )
            opponent_policy_specs = _opponent_pool(
                fixed_policy_specs=fixed_opponents,
                checkpoint_history=promoted_checkpoint_specs if promotion_pool_registry_path is not None else checkpoint_history,
                current_policy_spec=collection_current_policy_spec,
                max_historical_opponents=max_historical_opponents,
                historical_opponent_selection=historical_opponent_selection,
                include_current_policy=mirror_match,
            )
            iteration_training_paths: tuple[Path, ...]
            training_cache_paths: tuple[Path, ...] = ()
            training_cache_paths_out: list[Path] | None = [] if training_cache_root is not None else None
            training_cache_output_path = (
                training_cache_root / f"iteration-{iteration:04d}"
                if training_cache_root is not None
                else None
            )

            metrics = collect_selfplay_rollouts(
                output_path=rollout_path,
                training_output_path=training_rollout_path,
                training_cache_output_path=training_cache_output_path,
                training_cache_chunk_games=training_cache_chunk_games,
                training_cache_dataset_config=_dataset_config_from_training_config(training_config),
                training_cache_max_root_bytes=training_cache_max_root_bytes,
                training_cache_root=training_cache_root,
                training_cache_paths_out=training_cache_paths_out,
                games=games_per_iteration,
                env_factory=env_factory,
                rollout_config=rollout_config,
                seed_start=iteration_seed_start,
                current_policy_spec=collection_current_policy_spec,
                opponent_policy_specs=opponent_policy_specs,
                worker_count=worker_count,
            )
            if training_cache_paths_out is not None:
                training_cache_paths = tuple(training_cache_paths_out)
                if not training_cache_paths:
                    raise ValueError("training cache collection produced no cache paths.")
                iteration_training_paths = training_cache_paths
            elif training_rollout_path is not None:
                iteration_training_paths = (training_rollout_path,)
            else:
                raise ValueError("self-play collection produced no training input paths.")
            training_rollout_history.extend(iteration_training_paths)
            training_input_paths = _training_input_paths_for_objective(
                objective=training_config.objective,
                iteration_training_rollout_paths=iteration_training_paths,
                training_rollout_history=tuple(training_rollout_history),
            )
            value_selection_metrics = None
            if value_selection_config is not None and value_selection_config.heldout_games_per_iteration:
                value_selection_rollout_path = iteration_dir / "value-selection-rollouts.jsonl"
                value_selection_training_rollout_path = iteration_dir / "value-selection-training-rollouts.jsonl"
                value_selection_seed_start = next_value_selection_seed_start
                value_selection_metrics = collect_selfplay_rollouts(
                    output_path=value_selection_rollout_path,
                    training_output_path=value_selection_training_rollout_path,
                    games=value_selection_config.heldout_games_per_iteration,
                    env_factory=env_factory,
                    rollout_config=rollout_config,
                    seed_start=value_selection_seed_start,
                    current_policy_spec=collection_current_policy_spec,
                    opponent_policy_specs=opponent_policy_specs,
                    worker_count=worker_count,
                )
                next_value_selection_seed_start += value_selection_config.heldout_games_per_iteration
                value_selection_next_seed_start = next_value_selection_seed_start
                value_selection_rollout_history.append(value_selection_training_rollout_path)
            iteration_model_config = replace(
                model_config,
                policy_id=f"{model_config.policy_id}-iter-{iteration:04d}",
            )
            value_selection = None
            selection_paths: tuple[Path, ...] | None = None
            selection_data_role = "training_rollouts"
            selection_data_note = "Selection data comes from self-play training rollouts, not held-out validation."
            if value_selection_config is not None and value_selection_config.heldout_games_per_iteration:
                selection_paths = (
                    (value_selection_training_rollout_path,)
                    if value_selection_config.scope == "iteration"
                    else tuple(value_selection_rollout_history)
                )
                selection_data_role = "heldout_selfplay_rollouts"
                selection_data_note = (
                    "Selection data comes from held-out self-play rollouts collected for epoch selection."
                )
            elif value_selection_config is not None:
                selection_paths = (
                    iteration_training_paths
                    if value_selection_config.scope == "iteration"
                    else tuple(training_rollout_history)
                )
            iteration_training_config = _training_config_for_iteration_learning_rate_schedule(
                training_config,
                iteration=iteration,
                games_per_iteration=games_per_iteration,
                total_scheduled_iterations=first_iteration + iterations - 1,
                completed_games_offset=learning_rate_schedule_completed_games,
            )
            training_input_bytes = _paths_byte_size_best_effort(
                training_input_paths,
                known_sizes=training_input_path_byte_sizes,
            )
            training_started = perf_counter()
            if value_selection_config is None:
                model, training = train_transformer_policy(
                    training_input_paths,
                    model_config=iteration_model_config,
                    training_config=iteration_training_config,
                    initial_model=current_model,
                )
            else:
                model, training, value_selection = _train_with_iteration_value_selection(
                    paths=training_input_paths,
                    model_config=iteration_model_config,
                    training_config=iteration_training_config,
                    initial_model=current_model,
                    selection_paths=selection_paths or (),
                    config=value_selection_config,
                    data_role=selection_data_role,
                    data_note=selection_data_note,
                    artifact_path=iteration_dir / "value-selection.json",
                )
            training_elapsed_seconds = perf_counter() - training_started
            save_transformer_checkpoint(checkpoint_path, model, result=training)
            checkpoint_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else None
            value_calibration = _evaluate_iteration_value_calibration(
                model=model,
                training=training,
                iteration_training_rollout_paths=iteration_training_paths,
                training_rollout_history=tuple(training_rollout_history),
                config=value_calibration_config,
            )
            training_cache_deleted = False
            training_cache_deleted_bytes = 0
            if training_cache_paths and delete_training_cache_after_train:
                training_cache_deleted_bytes = training_cache_paths_byte_size(training_cache_paths)
                for path in training_cache_paths:
                    delete_training_cache_path(path)
                training_cache_deleted = True
            benchmark = None
            should_benchmark = _should_benchmark_iteration(
                iteration=iteration,
                games_per_iteration=games_per_iteration,
                completed_games_offset=learning_rate_schedule_completed_games,
                evaluation_games=evaluation_games,
                evaluation_interval_games=evaluation_interval_games,
                final_iteration=first_iteration + iterations - 1,
            )
            if should_benchmark:
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
                    seed_start=next_benchmark_seed_start,
                    device=training_config.device,
                    benchmark_reference_policy_specs=benchmark_references,
                )
                next_benchmark_seed_start += evaluation_games
            if auto_promotion_config is None:
                advancement = _advancement_decision(
                    benchmark=benchmark,
                    candidate_policy_id=training.model_config.policy_id,
                    incumbent_policy_spec=current_policy_spec,
                )
                if collector_advancement_mode == "always":
                    advancement = _always_advance_collector_decision(advancement)
                elif collector_advancement_mode == "yardstick-gate":
                    advancement = _yardstick_advancement_decision(
                        benchmark=benchmark,
                        candidate_policy_id=training.model_config.policy_id,
                        yardstick_policy_id=DEFAULT_COLLECTOR_YARDSTICK_POLICY_ID,
                        prior_iteration_manifests=(
                            *prior_iteration_manifests,
                            *(prior.to_manifest_dict() for prior in results),
                        ),
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
                value_selection_rollout_path=value_selection_rollout_path,
                value_selection_training_rollout_path=value_selection_training_rollout_path,
                value_selection_seed_start=value_selection_seed_start,
                value_selection_next_seed_start=value_selection_next_seed_start,
                checkpoint_path=checkpoint_path,
                manifest_path=iteration_manifest_path,
                current_policy_spec=current_policy_spec,
                opponent_policy_specs=opponent_policy_specs,
                training_rollout_paths=tuple(training_rollout_history),
                training_input_paths=training_input_paths,
                value_selection_training_rollout_paths=tuple(value_selection_rollout_history),
                seed_start=iteration_seed_start,
                worker_count=worker_count,
                metrics=metrics,
                value_selection_metrics=value_selection_metrics,
                training=training,
                training_elapsed_seconds=training_elapsed_seconds,
                training_input_bytes=training_input_bytes,
                checkpoint_bytes=checkpoint_bytes,
                benchmark=benchmark,
                advancement=advancement,
                opponent_pool_config=opponent_pool_manifest_config,
                invocation_config=invocation_config,
                benchmark_reference_policy_specs=benchmark_references,
                training_cache_paths=training_cache_paths,
                training_cache_deleted_after_train=training_cache_deleted,
                training_cache_deleted_bytes=training_cache_deleted_bytes,
                value_calibration=value_calibration,
                value_selection=value_selection,
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


def _training_input_paths_for_objective(
    *,
    objective: str,
    iteration_training_rollout_paths: tuple[Path, ...],
    training_rollout_history: tuple[Path, ...],
) -> tuple[Path, ...]:
    # PPO is on-policy enough that replaying older iteration shards as policy-gradient data makes
    # the behavior-policy ratio stale. If future on-policy objectives are added, route them here too.
    if objective == "ppo":
        return iteration_training_rollout_paths
    return training_rollout_history


def _should_benchmark_iteration(
    *,
    iteration: int,
    games_per_iteration: int,
    completed_games_offset: int,
    evaluation_games: int,
    evaluation_interval_games: int | None,
    final_iteration: int | None = None,
) -> bool:
    if evaluation_games <= 0:
        return False
    if evaluation_interval_games is None:
        return True
    if final_iteration is not None and iteration == final_iteration:
        return True
    completed_before = completed_games_offset + ((iteration - 1) * games_per_iteration)
    completed_after = completed_games_offset + (iteration * games_per_iteration)
    return completed_before // evaluation_interval_games < completed_after // evaluation_interval_games


def _paths_byte_size_best_effort(
    paths: Iterable[Path],
    *,
    known_sizes: dict[Path, int | None] | None = None,
) -> int | None:
    total = 0
    for path in paths:
        if known_sizes is not None and path not in known_sizes:
            known_sizes[path] = _path_byte_size_best_effort(path)
        byte_size = known_sizes[path] if known_sizes is not None else _path_byte_size_best_effort(path)
        if byte_size is None:
            return None
        total += byte_size
    return total


def _path_byte_size_best_effort(path: Path) -> int | None:
    try:
        if path.is_file() or path.is_symlink():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                if child.is_file() or child.is_symlink():
                    total += child.stat().st_size
            return total
    except OSError:
        return None
    return None


def _dataset_config_from_training_config(config: TransformerTrainingConfig) -> TrajectoryDatasetConfig:
    return TrajectoryDatasetConfig(
        window_size=config.window_size,
        discount=config.discount,
        capped_terminal_value=config.capped_terminal_value,
        hp_delta_return_weight=config.hp_delta_return_weight,
        faint_delta_return_weight=config.faint_delta_return_weight,
        turn_penalty_after=config.turn_penalty_after,
        turn_penalty=config.turn_penalty,
        ppo_target_mode=config.ppo_target_mode,
        gae_lambda=config.gae_lambda,
    )


def _train_with_iteration_value_selection(
    *,
    paths: tuple[Path, ...],
    model_config: TransformerPolicyConfig,
    training_config: TransformerTrainingConfig,
    initial_model: object | None,
    selection_paths: tuple[Path, ...],
    config: NeuralValueSelectionConfig,
    data_role: str,
    data_note: str,
    artifact_path: Path,
) -> tuple[object, TransformerTrainingResult, Mapping[str, Any]]:
    if not selection_paths:
        raise ValueError("value selection requires at least one selection rollout path.")
    selection_reports: list[dict[str, Any]] = []
    best_state = None
    best_epoch = None
    best_metric_value = None
    best_score = None
    device = resolve_torch_device(training_config.device)

    def evaluate_epoch(model: object, epoch_result: TransformerTrainingResult) -> None:
        nonlocal best_epoch, best_metric_value, best_score, best_state
        epoch_metric = epoch_result.final_metrics
        report = evaluate_value_calibration(
            model=model,
            training_result=epoch_result,
            paths=selection_paths,
            batch_size=config.batch_size,
            bins=config.bins,
            device=device,
        )
        try:
            metric_value = value_selection_metric_value(report, config.metric)
            score = value_selection_score(metric_value, config.metric)
            metric_unavailable_reason = None
        except ValueError as exc:
            if config.metric != "pearson_correlation":
                raise
            metric_value = None
            score = None
            metric_unavailable_reason = str(exc)
        selection_entry: dict[str, object] = {
            "epoch": epoch_metric.epoch,
            "metric_value": metric_value,
            "training_metrics": epoch_metric.to_dict(),
            "report": report.to_dict(),
        }
        if metric_unavailable_reason is not None:
            selection_entry["metric_unavailable_reason"] = metric_unavailable_reason
        selection_reports.append(selection_entry)
        if score is None:
            return
        if best_score is None or score > best_score:
            best_score = score
            best_metric_value = metric_value
            best_epoch = epoch_metric.epoch
            best_state = copy.deepcopy(model.state_dict())

    model, full_result = train_transformer_policy(
        paths,
        model_config=model_config,
        training_config=training_config,
        initial_model=initial_model,
        epoch_callback=evaluate_epoch,
    )
    if best_state is None or best_epoch is None or best_metric_value is None:
        raise ValueError("value selection produced no selectable epoch reports.")
    model.load_state_dict(best_state)
    selected_result = TransformerTrainingResult(
        model_config=model_config,
        training_config=replace(training_config, epochs=best_epoch),
        epochs=tuple(full_result.epochs[:best_epoch]),
    )
    artifact_payload = {
        "scope": config.scope,
        "paths": [str(path) for path in selection_paths],
        "data_role": data_role,
        "data_note": data_note,
        "batch_size": config.batch_size,
        "bins": config.bins,
        "metric": config.metric,
        "metric_direction": value_selection_metric_direction(config.metric),
        "selected_epoch": best_epoch,
        "selected_metric_value": best_metric_value,
        "epochs": selection_reports,
    }
    _write_json(artifact_path, artifact_payload)
    return model, selected_result, {
        "scope": config.scope,
        "paths": [str(path) for path in selection_paths],
        "data_role": artifact_payload["data_role"],
        "data_note": artifact_payload["data_note"],
        "batch_size": config.batch_size,
        "bins": config.bins,
        "metric": config.metric,
        "metric_direction": artifact_payload["metric_direction"],
        "selected_epoch": best_epoch,
        "selected_metric_value": best_metric_value,
        "artifact_path": str(artifact_path),
    }


def _evaluate_iteration_value_calibration(
    *,
    model: object,
    training: TransformerTrainingResult,
    iteration_training_rollout_paths: tuple[Path, ...],
    training_rollout_history: tuple[Path, ...],
    config: NeuralValueCalibrationConfig | None,
) -> Mapping[str, Any] | None:
    if config is None:
        return None
    paths = iteration_training_rollout_paths if config.scope == "iteration" else training_rollout_history
    report = evaluate_value_calibration(
        model=model,
        training_result=training,
        paths=paths,
        batch_size=config.batch_size,
        bins=config.bins,
        device=resolve_torch_device(training.training_config.device),
    )
    return {
        "scope": config.scope,
        "paths": [str(path) for path in paths],
        "batch_size": config.batch_size,
        "bins": config.bins,
        "report": report.to_dict(),
    }


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
        if last.ppo_valid_fraction is not None:
            scalars["ppo/valid_fraction"] = float(last.ppo_valid_fraction)
        if last.ppo_advantage_mean is not None:
            scalars["ppo/advantage_mean"] = float(last.ppo_advantage_mean)
        if last.ppo_advantage_std is not None:
            scalars["ppo/advantage_std"] = float(last.ppo_advantage_std)
        if last.ppo_ratio_mean is not None:
            scalars["ppo/ratio_mean"] = float(last.ppo_ratio_mean)
        if last.ppo_clip_fraction is not None:
            scalars["ppo/clip_fraction"] = float(last.ppo_clip_fraction)
        if last.ppo_entropy is not None:
            scalars["ppo/entropy"] = float(last.ppo_entropy)
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
    historical_opponent_selection: str = "recent",
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
        selection_mode=historical_opponent_selection,
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


def _always_advance_collector_decision(decision: NeuralAdvancementDecision) -> NeuralAdvancementDecision:
    if decision.advance_collector:
        return decision
    return replace(decision, advance_collector=True, reason="collector_advancement_mode_always")


def _yardstick_advancement_decision(
    *,
    benchmark: BenchmarkReport | None,
    candidate_policy_id: str,
    yardstick_policy_id: str,
    prior_iteration_manifests: Iterable[Mapping[str, Any]],
) -> NeuralAdvancementDecision:
    result = _candidate_yardstick_result(
        benchmark=benchmark,
        candidate_policy_id=candidate_policy_id,
        yardstick_policy_id=yardstick_policy_id,
    )
    if result is None:
        return NeuralAdvancementDecision(
            advance_collector=False,
            reason="missing_yardstick_benchmark",
            candidate_policy_id=candidate_policy_id,
            incumbent_policy_id=yardstick_policy_id,
            yardstick_policy_id=yardstick_policy_id,
        )
    candidate_win_rate, yardstick_win_rate, games = result
    previous_best = _best_accepted_yardstick_win_rate(
        prior_iteration_manifests,
        yardstick_policy_id=yardstick_policy_id,
    )
    if previous_best is None:
        return NeuralAdvancementDecision(
            advance_collector=True,
            reason="yardstick_baseline_initialized",
            candidate_policy_id=candidate_policy_id,
            incumbent_policy_id=yardstick_policy_id,
            candidate_win_rate=candidate_win_rate,
            incumbent_win_rate=yardstick_win_rate,
            games=games,
            yardstick_policy_id=yardstick_policy_id,
            yardstick_win_rate=candidate_win_rate,
            previous_best_yardstick_win_rate=None,
        )
    advance = candidate_win_rate > previous_best
    return NeuralAdvancementDecision(
        advance_collector=advance,
        reason="beat_yardstick_best" if advance else "failed_to_beat_yardstick_best",
        candidate_policy_id=candidate_policy_id,
        incumbent_policy_id=yardstick_policy_id,
        candidate_win_rate=candidate_win_rate,
        incumbent_win_rate=previous_best,
        games=games,
        yardstick_policy_id=yardstick_policy_id,
        yardstick_win_rate=candidate_win_rate,
        previous_best_yardstick_win_rate=previous_best,
    )


def _candidate_yardstick_result(
    *,
    benchmark: BenchmarkReport | None,
    candidate_policy_id: str,
    yardstick_policy_id: str,
) -> tuple[float, float, int] | None:
    if benchmark is None:
        return None
    for result in benchmark.head_to_head_results:
        parsed = _candidate_yardstick_rates(
            result,
            candidate_policy_id=candidate_policy_id,
            yardstick_policy_id=yardstick_policy_id,
        )
        if parsed is not None:
            return parsed
    return None


def _best_accepted_yardstick_win_rate(
    iteration_manifests: Iterable[Mapping[str, Any]],
    *,
    yardstick_policy_id: str,
) -> float | None:
    best: float | None = None
    for iteration in iteration_manifests:
        advancement = _optional_mapping(iteration.get("advancement"))
        if not advancement.get("advance_collector"):
            continue
        if advancement.get("reason") not in ACCEPTED_ADVANCEMENT_REASONS:
            continue
        candidate_policy_id = _accepted_candidate_policy_id(iteration, advancement)
        if not candidate_policy_id:
            continue
        benchmark = _optional_mapping(iteration.get("benchmark"))
        head_to_heads = benchmark.get("head_to_heads")
        if not isinstance(head_to_heads, Iterable) or isinstance(head_to_heads, (str, bytes, Mapping)):
            continue
        for result in _sequence(head_to_heads):
            parsed = _candidate_yardstick_rates(
                _mapping(result),
                candidate_policy_id=candidate_policy_id,
                yardstick_policy_id=yardstick_policy_id,
            )
            if parsed is None:
                continue
            candidate_win_rate, _, _ = parsed
            best = candidate_win_rate if best is None else max(best, candidate_win_rate)
    return best


def _accepted_candidate_policy_id(iteration: Mapping[str, Any], advancement: Mapping[str, Any]) -> str | None:
    policy_id = advancement.get("candidate_policy_id")
    if isinstance(policy_id, str) and policy_id:
        return policy_id
    training = _optional_mapping(iteration.get("training"))
    model_config = _optional_mapping(training.get("model_config"))
    policy_id = model_config.get("policy_id")
    return policy_id if isinstance(policy_id, str) and policy_id else None


def _candidate_yardstick_rates(
    result: Any,
    *,
    candidate_policy_id: str,
    yardstick_policy_id: str,
) -> tuple[float, float, int] | None:
    first_policy_id = str(_benchmark_result_value(result, "first_policy_id", ""))
    second_policy_id = str(_benchmark_result_value(result, "second_policy_id", ""))
    if {first_policy_id, second_policy_id} != {candidate_policy_id, yardstick_policy_id}:
        return None
    first_rate = float(_benchmark_result_value(result, "first_policy_win_rate", 0.0))
    second_rate = float(_benchmark_result_value(result, "second_policy_win_rate", 0.0))
    games = int(_benchmark_result_value(result, "games", 0))
    if first_policy_id == candidate_policy_id:
        return first_rate, second_rate, games
    return second_rate, first_rate, games


def _benchmark_result_value(result: Any, name: str, default: Any) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


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
    checkpoint = _neural_checkpoint_path_from_policy_spec(policy_spec)
    if checkpoint is None:
        return None
    model, _ = load_transformer_checkpoint(checkpoint, map_location=device)
    return model


def _neural_checkpoint_path_from_policy_spec(policy_spec: str) -> Path | None:
    body = policy_spec.strip().partition("?")[0].strip()
    if not body.lower().startswith("neural:"):
        return None
    checkpoint = body[len("neural:") :].strip()
    return Path(checkpoint) if checkpoint else None


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


def _with_collection_temperature(policy_spec: str, temperature: float) -> str:
    """Return the spec with a sampling temperature for self-play *collection* only.

    Temperature only applies to learnable checkpoint policies (neural/linear); for any other
    spec it is meaningless and the spec is returned unchanged. The canonical
    ``current_policy_spec`` is left alone (benchmark/advancement/manifest use the clean spec);
    this derived spec is used only for rollout collection so the collector explores.
    """
    if temperature == 1.0:
        return policy_spec
    from urllib.parse import urlencode

    # Parse with the same semantics the resolver uses (lowercased keys, duplicate rejection) so
    # the injected spec always round-trips through policy_factory_from_spec.
    body, options = _split_policy_spec_options(policy_spec)
    lowered = body.strip().lower()
    if not (lowered.startswith("neural:") or lowered.startswith("linear:")):
        return policy_spec
    options = dict(options)
    options.pop("deterministic", None)  # sampling is required for temperature to have any effect
    options["sample"] = "true"
    options["temperature"] = repr(float(temperature))
    return f"{body}?{urlencode(options)}"


def _opponent_pool(
    *,
    fixed_policy_specs: tuple[str, ...],
    checkpoint_history: Iterable[str],
    current_policy_spec: str,
    max_historical_opponents: int,
    historical_opponent_selection: str,
    include_current_policy: bool = False,
) -> tuple[str, ...]:
    return opponent_pool_policy_specs(
        fixed_policy_specs=fixed_policy_specs,
        checkpoint_history=checkpoint_history,
        current_policy_spec=current_policy_spec,
        max_historical_opponents=max_historical_opponents,
        include_current_policy=include_current_policy,
        historical_selection_mode=historical_opponent_selection,
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


def _resolve_learning_rate_schedule_completed_games(
    requested: int | None,
    *,
    prior_invocation_configs: tuple[Mapping[str, Any], ...],
) -> int:
    prior = _prior_learning_rate_schedule_completed_games(prior_invocation_configs)
    if requested is None:
        return prior or 0
    if requested < 0:
        raise ValueError("learning_rate_schedule_completed_games must be non-negative.")
    if prior is not None and requested != prior:
        raise ValueError(
            "learning_rate_schedule_completed_games must match the existing run on resume "
            f"({prior}); use a fresh run directory to change the external progress offset."
        )
    return requested


def _prior_learning_rate_schedule_completed_games(
    prior_invocation_configs: tuple[Mapping[str, Any], ...],
) -> int | None:
    for config in reversed(prior_invocation_configs):
        value = config.get("learning_rate_schedule_completed_games")
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("prior learning_rate_schedule_completed_games must be an integer.") from exc
        if parsed < 0:
            raise ValueError("prior learning_rate_schedule_completed_games must be non-negative.")
        return parsed
    return None


def _validate_learning_rate_schedule_window(
    training_config: TransformerTrainingConfig,
    *,
    completed_games_offset: int,
    requested_games: int,
) -> None:
    if training_config.learning_rate_schedule == CONSTANT_LEARNING_RATE_SCHEDULE:
        return
    if requested_games <= 0:
        raise ValueError("requested_games must be positive.")
    if completed_games_offset < 0:
        raise ValueError("completed_games_offset must be non-negative.")
    total_games = training_config.learning_rate_schedule_total_games
    if total_games is None:
        return
    if completed_games_offset >= total_games:
        raise ValueError(
            "completed_games_offset must be less than the learning-rate schedule total games; "
            f"got offset={completed_games_offset} total={total_games}. "
            "Increase learning_rate_schedule_total_games to the new global total when continuing a run."
        )
    if completed_games_offset + requested_games > total_games:
        raise ValueError(
            "learning-rate schedule total games must cover the full requested run window; "
            f"got offset={completed_games_offset} requested={requested_games} total={total_games}. "
            "Set learning_rate_schedule_total_games to at least completed games plus requested games."
        )


def _advancement_from_manifest(iteration: Mapping[str, Any]) -> Mapping[str, Any]:
    advancement = iteration.get("advancement")
    if advancement is None:
        return {}
    return _mapping(advancement)


def _next_value_selection_seed_start(iteration: Mapping[str, Any], *, default_seed_start: int) -> int:
    explicit_next = iteration.get("value_selection_next_seed_start")
    if explicit_next is not None:
        return int(explicit_next)
    seed_start = iteration.get("value_selection_seed_start")
    metrics = iteration.get("value_selection_collection_metrics")
    if seed_start is not None and isinstance(metrics, Mapping):
        return int(seed_start) + int(metrics.get("games", 0))
    return int(default_seed_start)


def _next_evaluation_seed_start(iterations: Iterable[Mapping[str, Any]], *, default_seed_start: int) -> int:
    next_seed: int | None = None
    for iteration in iterations:
        benchmark = iteration.get("benchmark")
        if not isinstance(benchmark, Mapping):
            continue
        for matchup in _sequence(benchmark.get("matchups", ())):
            matchup_payload = _mapping(matchup)
            seed_start = matchup_payload.get("seed_start")
            metrics = matchup_payload.get("metrics")
            if seed_start is None or not isinstance(metrics, Mapping):
                continue
            if "games" not in metrics:
                raise ValueError("benchmark matchup metrics missing games; cannot derive next evaluation seed.")
            games = int(metrics["games"])
            if games <= 0:
                raise ValueError("benchmark matchup games must be positive to derive next evaluation seed.")
            matchup_next_seed = int(seed_start) + games
            next_seed = matchup_next_seed if next_seed is None else max(next_seed, matchup_next_seed)
    return next_seed if next_seed is not None else int(default_seed_start)


def _training_result_to_dict(result: TransformerTrainingResult) -> dict[str, Any]:
    payload = {
        "model_config": result.model_config.to_dict(),
        "config": result.training_config.to_dict(),
        "epochs": [metrics.to_dict() for metrics in result.epochs],
    }
    if result.value_calibration_transform is not None:
        payload["value_calibration_transform"] = result.value_calibration_transform.to_dict()
    return payload


def _training_config_for_iteration_learning_rate_schedule(
    training_config: TransformerTrainingConfig,
    *,
    iteration: int,
    games_per_iteration: int,
    total_scheduled_iterations: int,
    completed_games_offset: int = 0,
) -> TransformerTrainingConfig:
    if training_config.learning_rate_schedule == CONSTANT_LEARNING_RATE_SCHEDULE:
        return training_config
    if iteration <= 0:
        raise ValueError("iteration must be positive.")
    if games_per_iteration <= 0:
        raise ValueError("games_per_iteration must be positive.")
    if completed_games_offset < 0:
        raise ValueError("completed_games_offset must be non-negative.")
    if training_config.learning_rate_schedule_total_games is None and total_scheduled_iterations < iteration:
        raise ValueError("total_scheduled_iterations must include the current iteration.")
    total_scheduled_games = (
        training_config.learning_rate_schedule_total_games
        if training_config.learning_rate_schedule_total_games is not None
        else completed_games_offset + (total_scheduled_iterations * games_per_iteration)
    )
    if completed_games_offset >= total_scheduled_games:
        raise ValueError(
            "completed_games_offset must be less than the learning-rate schedule total games; "
            f"got offset={completed_games_offset} total={total_scheduled_games}. "
            "Increase learning_rate_schedule_total_games to the new global total when continuing a run."
        )
    completed_games_before = completed_games_offset + ((iteration - 1) * games_per_iteration)
    completed_games_after = completed_games_offset + (iteration * games_per_iteration)
    return replace(
        training_config,
        learning_rate_progress_start=min(1.0, completed_games_before / total_scheduled_games),
        learning_rate_progress_end=min(1.0, completed_games_after / total_scheduled_games),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
        raise ValueError("expected JSON array payload.")
    return tuple(value)
