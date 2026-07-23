"""Streaming trajectory dataset helpers for early training experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .actions import ACTION_COUNT
from .collection import RolloutRecord, iter_rollout_records
from .observation import require_current_observation_schema
from .padding import zeros_like as _zeros_like
from .shaping import ShapingConfig, shaping_rewards_by_step_index
from .trajectory import TrajectoryStep

_LOGGER = logging.getLogger(__name__)

MISSING_ACTION_INDEX = -1
# v2: cache arrays carry observation-spec-v2 tensors. Bumped with the observation break so a
# pre-break cache refuses at the metadata guard instead of failing shape-wise mid-training.
TRAINING_CACHE_SCHEMA_VERSION = "pokezero.training_cache.v2"
MAX_ACTIVE_TRAINING_CACHE_GB = 50.0
MAX_ACTIVE_TRAINING_CACHE_BYTES = int(MAX_ACTIVE_TRAINING_CACHE_GB * 1024 * 1024 * 1024)

# Env escape hatch: skip the storage-cap check entirely (local smokes / benchmarks where the
# output parent is an unrelated scratch dir and the cap is irrelevant). Never set in production.
_SKIP_CACHE_ROOT_CAP_ENV = "POKEZERO_SKIP_CACHE_ROOT_CAP"

PathInput = str | PathLike[str] | Path


class CacheRootByteBudget:
    """Amortized storage-cap tracker for repeated cache writes into one root.

    The cap check must sum the cache root's on-disk bytes, but ``_directory_byte_size`` is an
    ``rglob + stat`` over the whole root — O(files), and on NFS a per-flush getattr storm. A
    chunked/self-play run calls ``write`` once per chunk, so re-walking every flush makes cap cost
    grow with the run (the dominant collect-wall cost measured on metamon-M). This tracker re-walks
    only every ``revalidate_writes`` writes or ``revalidate_seconds`` — whichever first — and adds
    each write's estimated bytes locally in between. That keeps GLOBAL visibility (concurrent
    collector pods + the trainer GC-deleting consumed caches, both invisible to a pure per-process
    counter) with bounded staleness, at O(1) amortized cost. Shared across a writer's lifetime; the
    per-chunk builder is recreated each flush, so the budget must live on the caller (chunk writer).
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        revalidate_writes: int = 20,
        revalidate_seconds: float = 60.0,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self.max_bytes = max_bytes
        self._revalidate_writes = max(1, revalidate_writes)
        self._revalidate_seconds = revalidate_seconds
        self._walked_bytes: int | None = None
        self._walked_at = 0.0
        self._writes_since_walk = 0
        self._pending_bytes = 0

    def reserve(self, root: Path, estimated_bytes: int) -> None:
        """Raise if this write would exceed the cap; otherwise reserve its bytes.

        Re-walks ``root`` when stale (first call, >= revalidate_writes since, or >=
        revalidate_seconds since), else uses the last walk plus locally-accumulated reservations.
        """
        now = perf_counter()
        stale = (
            self._walked_bytes is None
            or self._writes_since_walk >= self._revalidate_writes
            or (now - self._walked_at) >= self._revalidate_seconds
        )
        if stale:
            self._walked_bytes = _directory_byte_size(root) if root.exists() else 0
            self._walked_at = now
            self._writes_since_walk = 0
            self._pending_bytes = 0
        current_bytes = (self._walked_bytes or 0) + self._pending_bytes
        if current_bytes + estimated_bytes > self.max_bytes:
            raise ValueError(
                f"training cache write would exceed storage cap: existing~={current_bytes} bytes "
                f"estimated_new={estimated_bytes} bytes limit={self.max_bytes} bytes "
                f"(amortized; last walk {self._writes_since_walk} write(s) ago)."
            )
        self._pending_bytes += estimated_bytes
        self._writes_since_walk += 1

_OPPONENT_POOL_METADATA_KEYS = (
    "opponent_policy_spec",
    "opponent_pool_checkpoint_hash",
    "opponent_pool_member_id",
    "opponent_pool_weight",
)


@dataclass(frozen=True)
class TrajectoryDatasetConfig:
    """Controls how serialized rollout steps become training examples.

    Discounting is applied per recorded decision for each player. Terminal
    returns are derived from the battle result rather than sparse per-step
    rewards, so asymmetric final rounds still label both players' histories.
    Optional shaping terms are player-relative and use only metadata already
    present in that player's observation. Final return targets are clipped to
    [-1, 1] to stay compatible with bounded value heads.
    """

    window_size: int = 1
    discount: float = 1.0
    capped_terminal_value: float = 0.0
    hp_delta_return_weight: float = 0.0
    faint_delta_return_weight: float = 0.0
    turn_penalty_after: int | None = None
    turn_penalty: float = 0.0
    ppo_target_mode: str = "returns"
    gae_lambda: float = 0.95
    # Dense potential-based shaping (pokezero.shaping; WS-E arm 1). None = unshaped and the
    # key is OMITTED from to_dict()/cache metadata so shaping-off caches stay byte-identical
    # to pre-shaping collection; from_dict defaults payloads lacking the field to unshaped.
    # The shaping gamma is this config's `discount`.
    potential_shaping: ShapingConfig | None = None

    def __post_init__(self) -> None:
        if self.potential_shaping is not None and not isinstance(self.potential_shaping, ShapingConfig):
            object.__setattr__(self, "potential_shaping", ShapingConfig.from_dict(self.potential_shaping))
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between 0 and 1.")
        if not -1.0 <= self.capped_terminal_value <= 0.0:
            raise ValueError("capped_terminal_value must be between -1 and 0.")
        if self.hp_delta_return_weight < 0.0:
            raise ValueError("hp_delta_return_weight must be non-negative.")
        if self.faint_delta_return_weight < 0.0:
            raise ValueError("faint_delta_return_weight must be non-negative.")
        if self.turn_penalty_after is not None and self.turn_penalty_after < 0:
            raise ValueError("turn_penalty_after must be non-negative when set.")
        if self.turn_penalty < 0.0:
            raise ValueError("turn_penalty must be non-negative.")
        if self.turn_penalty > 0.0 and self.turn_penalty_after is None:
            raise ValueError("turn_penalty_after must be set when turn_penalty is positive.")
        if self.ppo_target_mode not in {"returns", "gae"}:
            raise ValueError("ppo_target_mode must be 'returns' or 'gae'.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "discount": self.discount,
            "capped_terminal_value": self.capped_terminal_value,
            "hp_delta_return_weight": self.hp_delta_return_weight,
            "faint_delta_return_weight": self.faint_delta_return_weight,
            "turn_penalty_after": self.turn_penalty_after,
            "turn_penalty": self.turn_penalty,
            "ppo_target_mode": self.ppo_target_mode,
            "gae_lambda": self.gae_lambda,
            # Omitted entirely when unshaped: keeps shaping-off cache metadata (and the
            # cache-vs-train config equality check against legacy caches) byte-identical.
            **(
                {"potential_shaping": self.potential_shaping.to_dict()}
                if self.potential_shaping is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryDatasetConfig":
        return cls(
            window_size=int(payload.get("window_size", 1)),
            discount=float(payload.get("discount", 1.0)),
            capped_terminal_value=float(payload.get("capped_terminal_value", 0.0)),
            hp_delta_return_weight=float(payload.get("hp_delta_return_weight", 0.0)),
            faint_delta_return_weight=float(payload.get("faint_delta_return_weight", 0.0)),
            turn_penalty_after=(
                None if payload.get("turn_penalty_after") is None else int(payload["turn_penalty_after"])
            ),
            turn_penalty=float(payload.get("turn_penalty", 0.0)),
            ppo_target_mode=str(payload.get("ppo_target_mode", "returns")),
            gae_lambda=float(payload.get("gae_lambda", 0.95)),
            # Payloads lacking the field (all pre-shaping caches) are definitively unshaped.
            potential_shaping=(
                ShapingConfig.from_dict(payload["potential_shaping"])
                if payload.get("potential_shaping") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TrajectoryExample:
    battle_id: str
    seed: int
    format_id: str
    player_id: str
    turn_index: int
    categorical_ids: tuple[Any, ...]
    numeric_features: tuple[Any, ...]
    token_type_ids: tuple[Any, ...]
    attention_mask: tuple[Any, ...]
    history_mask: tuple[bool, ...]
    legal_action_mask: tuple[bool, ...]
    action_index: int
    reward: float
    return_value: float
    value_estimate: float | None = None
    ppo_advantage: float | None = None
    ppo_value_target: float | None = None
    opponent_action_index: int | None = None
    action_probability: float | None = None
    step_metadata: Mapping[str, Any] | None = None
    terminal_capped: bool = False
    # Dense potential-based shaping component for this decision (None when shaping is
    # off). Stored separately from `reward` (raw env reward, unchanged) and already
    # folded into return_value / PPO targets when the dataset config enables shaping.
    shaping_reward: float | None = None
    # Generic per-example training emphasis. Defaults to neutral weight; refutation
    # caches can raise this for certified high-surprise rows without changing the
    # primary rollout data contract.
    training_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.training_weight)) or float(self.training_weight) <= 0.0:
            raise ValueError("training_weight must be positive and finite.")

    @property
    def window_size(self) -> int:
        return len(self.history_mask)


@dataclass(frozen=True)
class TrainingBatch:
    categorical_ids: tuple[Any, ...]
    numeric_features: tuple[Any, ...]
    token_type_ids: tuple[Any, ...]
    attention_mask: tuple[Any, ...]
    history_mask: tuple[tuple[bool, ...], ...]
    legal_action_mask: tuple[tuple[bool, ...], ...]
    action_indices: tuple[int, ...]
    rewards: tuple[float, ...]
    returns: tuple[float, ...]
    value_estimates: tuple[float, ...]
    value_estimate_mask: tuple[bool, ...]
    ppo_advantages: tuple[float, ...]
    ppo_advantage_mask: tuple[bool, ...]
    ppo_value_targets: tuple[float, ...]
    ppo_value_target_mask: tuple[bool, ...]
    opponent_action_indices: tuple[int, ...]
    opponent_action_mask: tuple[bool, ...]
    action_probabilities: tuple[float, ...]
    action_probability_mask: tuple[bool, ...]
    training_weights: tuple[float, ...]
    battle_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    format_ids: tuple[str, ...]
    player_ids: tuple[str, ...]
    turn_indices: tuple[int, ...]
    terminal_capped: tuple[bool, ...]
    step_metadata: tuple[Mapping[str, Any], ...]
    row_categorical_ids: Any | None = None
    row_numeric_features: Any | None = None
    row_token_type_ids: Any | None = None
    row_attention_mask: Any | None = None
    window_row_indices: Any | None = None

    def __post_init__(self) -> None:
        batch_size = len(self.action_indices)
        if batch_size == 0:
            raise ValueError("TrainingBatch must contain at least one example.")
        row_indexed = self.window_row_indices is not None
        for name, values in (
            ("history_mask", self.history_mask),
            ("legal_action_mask", self.legal_action_mask),
            ("rewards", self.rewards),
            ("returns", self.returns),
            ("value_estimates", self.value_estimates),
            ("value_estimate_mask", self.value_estimate_mask),
            ("ppo_advantages", self.ppo_advantages),
            ("ppo_advantage_mask", self.ppo_advantage_mask),
            ("ppo_value_targets", self.ppo_value_targets),
            ("ppo_value_target_mask", self.ppo_value_target_mask),
            ("opponent_action_indices", self.opponent_action_indices),
            ("opponent_action_mask", self.opponent_action_mask),
            ("action_probabilities", self.action_probabilities),
            ("action_probability_mask", self.action_probability_mask),
            ("training_weights", self.training_weights),
            ("battle_ids", self.battle_ids),
            ("seeds", self.seeds),
            ("format_ids", self.format_ids),
            ("player_ids", self.player_ids),
            ("turn_indices", self.turn_indices),
            ("terminal_capped", self.terminal_capped),
            ("step_metadata", self.step_metadata),
        ):
            if len(values) != batch_size:
                raise ValueError(f"{name} must contain {batch_size} values.")
        if row_indexed:
            for name, values in (
                ("row_categorical_ids", self.row_categorical_ids),
                ("row_numeric_features", self.row_numeric_features),
                ("row_token_type_ids", self.row_token_type_ids),
                ("row_attention_mask", self.row_attention_mask),
            ):
                if values is None:
                    raise ValueError(f"{name} is required when window_row_indices is set.")
            if len(self.window_row_indices) != batch_size:
                raise ValueError(f"window_row_indices must contain {batch_size} values.")
        else:
            for name, values in (
                ("categorical_ids", self.categorical_ids),
                ("numeric_features", self.numeric_features),
                ("token_type_ids", self.token_type_ids),
                ("attention_mask", self.attention_mask),
            ):
                if len(values) != batch_size:
                    raise ValueError(f"{name} must contain {batch_size} values.")

    @property
    def batch_size(self) -> int:
        return len(self.action_indices)

    @property
    def window_size(self) -> int:
        return len(self.history_mask[0])


@dataclass(frozen=True)
class TrainingCacheSummary:
    path: Path
    record_count: int
    example_count: int
    byte_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "record_count": self.record_count,
            "example_count": self.example_count,
            "byte_size": self.byte_size,
        }


def _feature_masks_payload(feature_masks) -> dict | None:
    """JSON payload for cache metadata from an ObservationFeatureMasks (or mapping)."""
    if feature_masks is None:
        return None
    if is_dataclass(feature_masks):
        payload = asdict(feature_masks)
        # Wire-format compat: the serialized key stays "stats_block" (frozen in existing
        # checkpoint cache metadata + the golden corpus) even though the attribute was
        # renamed to opponent_tendency_stats_block. Remap on the way out until the corpus
        # regen lands and the wire key can move with it.
        if "opponent_tendency_stats_block" in payload:
            payload = {
                ("stats_block" if key == "opponent_tendency_stats_block" else key): value
                for key, value in payload.items()
            }
        return payload
    return dict(feature_masks)


def _opponent_pool_provenance_from_record(record: RolloutRecord) -> dict[str, Any] | None:
    metadata = record.trajectory.metadata
    if not any(key in metadata for key in _OPPONENT_POOL_METADATA_KEYS):
        return None
    provenance: dict[str, Any] = {
        "battle_id": record.trajectory.battle_id,
        "seed": record.trajectory.seed,
        "format_id": record.trajectory.format_id,
    }
    if record.trajectory.steps:
        provenance["player_id"] = record.trajectory.steps[0].player_id
    for key in _OPPONENT_POOL_METADATA_KEYS:
        if key in metadata:
            provenance[key] = metadata[key]
    return provenance


class TrainingCacheBuilder:
    """Build a compact, array-backed training cache from rollout records.

    The raw rollout JSONL stores every public observation as nested JSON. The cache stores each
    observation once, uses small numeric dtypes, and represents history windows as integer row
    references. This keeps shard storage bounded while letting training batch with memmapped arrays.
    """

    def __init__(
        self,
        *,
        config: TrajectoryDatasetConfig | None = None,
        feature_masks=None,
        observation_schema: str | None = None,
    ) -> None:
        self.config = config or TrajectoryDatasetConfig()
        self._record_count = 0
        self._categorical_rows: list[Any] = []
        self._numeric_rows: list[Any] = []
        self._token_type_rows: list[Any] = []
        self._attention_rows: list[Any] = []
        self._window_indices: list[tuple[int, ...]] = []
        self._legal_action_masks: list[Any] = []
        self._action_indices: list[int] = []
        self._rewards: list[float] = []
        self._returns: list[float] = []
        self._value_estimates: list[float] = []
        self._value_estimate_masks: list[bool] = []
        self._ppo_advantages: list[float] = []
        self._ppo_advantage_masks: list[bool] = []
        self._ppo_value_targets: list[float] = []
        self._ppo_value_target_masks: list[bool] = []
        self._opponent_action_indices: list[int] = []
        self._opponent_action_masks: list[bool] = []
        self._action_probabilities: list[float] = []
        self._action_probability_masks: list[bool] = []
        self._training_weights: list[float] = []
        self._seeds: list[int] = []
        self._turn_indices: list[int] = []
        self._terminal_capped: list[bool] = []
        # Dense shaping components, populated only when the config enables potential
        # shaping (written as an OPTIONAL shaping_rewards.npy array; readers must
        # tolerate its absence — legacy caches never carry it).
        self._shaping_rewards: list[float] = []
        # Belief provenance of ingested records; written to cache metadata as a single hash when
        # unanimous, else null (mixed/legacy) with a mixed flag for diagnostics.
        self._belief_set_source_hashes: set[str | None] = set()
        self._opponent_pool_provenance: list[dict[str, Any]] = []
        self._feature_masks_payload = _feature_masks_payload(feature_masks)
        # Observation schema the collecting env encoded under (None for legacy caches);
        # the trainer hard-fails on a cache-vs-model schema mismatch — the schema-axis
        # twin of the feature-mask cross-check (v2.2 fresh-selection latch).
        self._observation_schema = observation_schema

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def example_count(self) -> int:
        return len(self._action_indices)

    def add_record(self, record: RolloutRecord) -> None:
        history_by_player: dict[str, list[int]] = {}
        for example in examples_from_record(record, config=self.config):
            row_index = len(self._categorical_rows) + 1
            history = history_by_player.setdefault(example.player_id, [])
            history.append(row_index)
            window = tuple(history[-self.config.window_size :])
            padding_count = self.config.window_size - len(window)
            self._window_indices.append(tuple(0 for _ in range(padding_count)) + window)

            self._categorical_rows.append(example.categorical_ids[-1])
            self._numeric_rows.append(example.numeric_features[-1])
            self._token_type_rows.append(example.token_type_ids[-1])
            self._attention_rows.append(example.attention_mask[-1])
            self._legal_action_masks.append(example.legal_action_mask)
            self._action_indices.append(example.action_index)
            self._rewards.append(example.reward)
            self._returns.append(example.return_value)
            self._value_estimates.append(_optional_float(example.value_estimate))
            self._value_estimate_masks.append(example.value_estimate is not None)
            self._ppo_advantages.append(_optional_float(example.ppo_advantage))
            self._ppo_advantage_masks.append(example.ppo_advantage is not None)
            self._ppo_value_targets.append(_optional_float(example.ppo_value_target))
            self._ppo_value_target_masks.append(example.ppo_value_target is not None)
            self._opponent_action_indices.append(_optional_action_index(example.opponent_action_index))
            self._opponent_action_masks.append(example.opponent_action_index is not None)
            self._action_probabilities.append(_optional_float(example.action_probability))
            self._action_probability_masks.append(example.action_probability is not None)
            self._training_weights.append(float(example.training_weight))
            self._seeds.append(example.seed)
            self._turn_indices.append(example.turn_index)
            self._terminal_capped.append(example.terminal_capped)
            if self.config.potential_shaping is not None:
                self._shaping_rewards.append(_optional_float(example.shaping_reward))
        opponent_pool_provenance = _opponent_pool_provenance_from_record(record)
        if opponent_pool_provenance is not None:
            self._opponent_pool_provenance.append(opponent_pool_provenance)
        self._belief_set_source_hashes.add(record.belief_set_source_hash)
        self._record_count += 1

    def add_example(self, example: TrajectoryExample) -> None:
        """Add one already-materialized example to a cache.

        Normal rollout ingestion stores one observation row per decision and
        reconstructs windows from per-player history. Refutation examples are
        sparse counterfactual targets, so their full observation window is
        already materialized and must be written directly.
        """

        if example.window_size != self.config.window_size:
            raise ValueError("example window_size must match the training cache config.")
        window_indices: list[int] = []
        for valid, categorical, numeric, token_type, attention in zip(
            example.history_mask,
            example.categorical_ids,
            example.numeric_features,
            example.token_type_ids,
            example.attention_mask,
            strict=True,
        ):
            if not valid:
                window_indices.append(0)
                continue
            row_index = len(self._categorical_rows) + 1
            self._categorical_rows.append(categorical)
            self._numeric_rows.append(numeric)
            self._token_type_rows.append(token_type)
            self._attention_rows.append(attention)
            window_indices.append(row_index)
        self._window_indices.append(tuple(window_indices))
        self._append_example_targets(example)
        self._belief_set_source_hashes.add(None)
        self._record_count += 1

    def _append_example_targets(self, example: TrajectoryExample) -> None:
        self._legal_action_masks.append(example.legal_action_mask)
        self._action_indices.append(example.action_index)
        self._rewards.append(example.reward)
        self._returns.append(example.return_value)
        self._value_estimates.append(_optional_float(example.value_estimate))
        self._value_estimate_masks.append(example.value_estimate is not None)
        self._ppo_advantages.append(_optional_float(example.ppo_advantage))
        self._ppo_advantage_masks.append(example.ppo_advantage is not None)
        self._ppo_value_targets.append(_optional_float(example.ppo_value_target))
        self._ppo_value_target_masks.append(example.ppo_value_target is not None)
        self._opponent_action_indices.append(_optional_action_index(example.opponent_action_index))
        self._opponent_action_masks.append(example.opponent_action_index is not None)
        self._action_probabilities.append(_optional_float(example.action_probability))
        self._action_probability_masks.append(example.action_probability is not None)
        self._training_weights.append(float(example.training_weight))
        self._seeds.append(example.seed)
        self._turn_indices.append(example.turn_index)
        self._terminal_capped.append(example.terminal_capped)
        if self.config.potential_shaping is not None:
            self._shaping_rewards.append(_optional_float(example.shaping_reward))

    def write(
        self,
        path: PathInput,
        *,
        overwrite: bool = False,
        max_cache_root_bytes: int | None = MAX_ACTIVE_TRAINING_CACHE_BYTES,
        cache_root: PathInput | None = None,
        root_byte_budget: "CacheRootByteBudget | None" = None,
    ) -> TrainingCacheSummary:
        if self.example_count == 0:
            raise ValueError("training cache cannot be written with zero examples.")
        numpy = _require_numpy()
        output_path = Path(path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"training cache already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional write()-stage timing (off by default; POKEZERO_ARRAYS_TIMING=1) to localize the
        # cost between array materialization, the storage-cap directory walk (rglob over the whole
        # output parent — a real cost on a large/NFS cache root), and the numpy.save disk loop.
        _wtimed = os.environ.get("POKEZERO_ARRAYS_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}
        _wt: dict[str, float] = {}
        _wstart = perf_counter()
        arrays = self._arrays(numpy)
        if _wtimed:
            _wt["arrays_total"] = round(perf_counter() - _wstart, 4)
        _skip_cap = os.environ.get(_SKIP_CACHE_ROOT_CAP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        if max_cache_root_bytes is not None and not _skip_cap:
            if max_cache_root_bytes <= 0:
                raise ValueError("max_cache_root_bytes must be positive.")
            # Root default is the cache dir ITSELF, not its parent: walking the parent sweeps
            # unrelated siblings (a local smoke walked all of /tmp — 15s). Callers that intend a
            # shared multi-cache root pass cache_root explicitly (they already do).
            root = Path(cache_root) if cache_root is not None else output_path
            estimated_bytes = _estimated_training_cache_byte_size(arrays)
            _wstart = perf_counter()
            if root_byte_budget is not None:
                # Amortized: re-walks root only every N writes / T seconds (see CacheRootByteBudget).
                root_byte_budget.reserve(root, estimated_bytes)
            else:
                current_bytes = _directory_byte_size(root) if root.exists() else 0
                if current_bytes + estimated_bytes > max_cache_root_bytes:
                    raise ValueError(
                        f"training cache write would exceed storage cap: existing={current_bytes} bytes "
                        f"estimated_new={estimated_bytes} bytes limit={max_cache_root_bytes} bytes."
                    )
            if _wtimed:
                _wt["cap_dir_walk"] = round(perf_counter() - _wstart, 4)
        temp_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
        try:
            _wstart = perf_counter()
            for name, value in arrays.items():
                numpy.save(temp_path / f"{name}.npy", value, allow_pickle=False)
            if _wtimed:
                _wt["save_loop"] = round(perf_counter() - _wstart, 4)
                _LOGGER.warning("training-cache write() stage timing (s): %s", _wt)
            metadata = {
                "schema_version": TRAINING_CACHE_SCHEMA_VERSION,
                "dataset_config": self.config.to_dict(),
                "record_count": self.record_count,
                "example_count": self.example_count,
                "observation_shapes": {
                    "categorical_ids": list(arrays["categorical_ids"].shape[1:]),
                    "numeric_features": list(arrays["numeric_features"].shape[1:]),
                    "token_type_ids": list(arrays["token_type_ids"].shape[1:]),
                    "attention_mask": list(arrays["attention_mask"].shape[1:]),
                    "legal_action_mask": [ACTION_COUNT],
                    "window_size": self.config.window_size,
                },
                "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
                "belief_set_source_hash": (
                    next(iter(self._belief_set_source_hashes))
                    if len(self._belief_set_source_hashes) == 1
                    else None
                ),
                "belief_set_source_mixed": len(self._belief_set_source_hashes) > 1,
                "opponent_pool_provenance": self._opponent_pool_provenance,
                "opponent_pool_provenance_count": len(self._opponent_pool_provenance),
                "opponent_pool_provenance_mixed": 0 < len(self._opponent_pool_provenance) < self.record_count,
                # Encode-time feature masks the collecting env observed under (None for
                # legacy caches); the trainer hard-fails on a cache-vs-model mask
                # mismatch — the mask-axis twin of the belief-provenance hash above.
                "feature_masks": self._feature_masks_payload,
                # Observation schema the collecting env encoded under (absent/None on
                # legacy caches, which by definition predate v2.2 recording).
                "observation_schema": self._observation_schema,
                "format": "directory-of-npy-arrays",
                "padding_row": 0,
                "categorical_storage": {
                    "mode": "compact-nonzero",
                    "original_feature_count": int(len(self._categorical_rows[0][0])),
                    "stored_feature_count": int(arrays["categorical_ids"].shape[2]),
                    "semantic": "summed category embeddings are identical to dense zero-padded rows",
                },
            }
            (temp_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            temp_path.rename(output_path)
        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise
        return TrainingCacheSummary(
            path=output_path,
            record_count=self.record_count,
            example_count=self.example_count,
            byte_size=_directory_byte_size(output_path),
        )

    def _arrays(self, numpy: Any) -> dict[str, Any]:
        # Optional per-line timing (off by default; POKEZERO_ARRAYS_TIMING=1). The dominant cost of
        # write() is here — converting the builder's accumulated nested-Python-list rows into numpy
        # (esp. the float16 numeric asarray of ~tokens*features*examples objects) and the
        # categorical compaction — NOT the numpy.save disk loop. This breaks out which line pays.
        _timed = os.environ.get("POKEZERO_ARRAYS_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}
        _breakdown: dict[str, float] = {}

        def _timeit(label: str, make: "Callable[[], Any]") -> Any:
            if not _timed:
                return make()
            _start = perf_counter()
            value = make()
            _breakdown[label] = round(perf_counter() - _start, 4)
            return value

        categorical_raw = _timeit("categorical_asarray", lambda: numpy.asarray(self._categorical_rows))
        numeric = _timeit("numeric_asarray_float16", lambda: numpy.asarray(self._numeric_rows, dtype=numpy.float16))
        token_type_raw = _timeit("token_type_asarray", lambda: numpy.asarray(self._token_type_rows))
        if _array_min(numpy, categorical_raw) < 0 or _array_max(numpy, categorical_raw) > int(numpy.iinfo(numpy.uint16).max):
            raise ValueError("categorical ids exceed uint16 training-cache range.")
        if _array_min(numpy, token_type_raw) < 0 or _array_max(numpy, token_type_raw) > int(numpy.iinfo(numpy.uint8).max):
            raise ValueError("token type ids exceed uint8 training-cache range.")
        categorical = _timeit(
            "compact_categorical",
            lambda: _compact_categorical_rows(numpy, categorical_raw.astype(numpy.uint16, copy=False)),
        )
        token_type = token_type_raw.astype(numpy.uint8, copy=False)
        if _timed:
            _LOGGER.warning("training-cache _arrays per-line timing (s): %s", _breakdown)
        return {
            "categorical_ids": _prepend_zero_row(numpy, categorical),
            "numeric_features": _prepend_zero_row(numpy, numeric),
            "token_type_ids": _prepend_zero_row(numpy, token_type),
            "attention_mask": _prepend_zero_row(numpy, numpy.asarray(self._attention_rows, dtype=numpy.bool_)),
            "window_indices": numpy.asarray(self._window_indices, dtype=numpy.uint32),
            "legal_action_mask": numpy.asarray(self._legal_action_masks, dtype=numpy.bool_),
            "action_indices": numpy.asarray(self._action_indices, dtype=numpy.int16),
            "rewards": numpy.asarray(self._rewards, dtype=numpy.float32),
            "returns": numpy.asarray(self._returns, dtype=numpy.float32),
            "value_estimates": numpy.asarray(self._value_estimates, dtype=numpy.float32),
            "value_estimate_mask": numpy.asarray(self._value_estimate_masks, dtype=numpy.bool_),
            "ppo_advantages": numpy.asarray(self._ppo_advantages, dtype=numpy.float32),
            "ppo_advantage_mask": numpy.asarray(self._ppo_advantage_masks, dtype=numpy.bool_),
            "ppo_value_targets": numpy.asarray(self._ppo_value_targets, dtype=numpy.float32),
            "ppo_value_target_mask": numpy.asarray(self._ppo_value_target_masks, dtype=numpy.bool_),
            "opponent_action_indices": numpy.asarray(self._opponent_action_indices, dtype=numpy.int16),
            "opponent_action_mask": numpy.asarray(self._opponent_action_masks, dtype=numpy.bool_),
            "action_probabilities": numpy.asarray(self._action_probabilities, dtype=numpy.float32),
            "action_probability_mask": numpy.asarray(self._action_probability_masks, dtype=numpy.bool_),
            "training_weights": numpy.asarray(self._training_weights, dtype=numpy.float32),
            "seeds": numpy.asarray(self._seeds, dtype=numpy.int64),
            "turn_indices": numpy.asarray(self._turn_indices, dtype=numpy.int32),
            "terminal_capped": numpy.asarray(self._terminal_capped, dtype=numpy.bool_),
            # Optional array: present only for shaping-enabled caches. `rewards` above
            # stays the raw env reward; the shaping component is stored separately.
            **(
                {"shaping_rewards": numpy.asarray(self._shaping_rewards, dtype=numpy.float32)}
                if self.config.potential_shaping is not None
                else {}
            ),
        }


# The four big per-token arrays carry the prepended zero padding row (metadata
# "padding_row": 0); every other array is per-example with no reserved row.
_ZERO_ROW_ARRAYS = ("categorical_ids", "numeric_features", "token_type_ids", "attention_mask")
# Metadata fields that must match exactly for two caches to be concatenable.
_CONCAT_COMPAT_FIELDS = ("schema_version", "dataset_config", "observation_schema", "feature_masks")


def concat_training_caches(
    paths: Sequence[PathInput],
    output_path: PathInput,
    *,
    overwrite: bool = False,
) -> TrainingCacheSummary:
    """Concatenate training cache directories into one cache directory.

    Output is byte-identical to a single cache written over the same records in
    the same order (the shard fan-in invariant): subsequent caches' zero padding
    rows are dropped, their nonzero ``window_indices`` are offset by the running
    example count (0 entries stay 0 — window padding), and categorical arrays
    are zero-padded to the widest compaction width (documented as semantically
    identity: "summed category embeddings are identical to dense zero-padded
    rows"). Optional arrays (e.g. ``shaping_rewards``) must be present in all
    parts or none. Fails closed on any config/schema/mask mismatch. The output
    is assembled in a temp dir and atomically renamed into place.
    """
    numpy = _require_numpy()
    if len(paths) == 0:
        raise ValueError("concat_training_caches requires at least one input cache.")
    normalized = [Path(path) for path in paths]
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"training cache already exists: {output_path}")

    metadatas = []
    for path in normalized:
        if not is_training_cache_path(path):
            raise ValueError(f"not a training cache directory: {path}")
        metadatas.append(json.loads((path / "metadata.json").read_text(encoding="utf-8")))
    base = metadatas[0]
    for path, meta in zip(normalized[1:], metadatas[1:], strict=True):
        for field in _CONCAT_COMPAT_FIELDS:
            if meta.get(field) != base.get(field):
                raise ValueError(
                    f"cache {path} is not concatenable: {field} mismatch "
                    f"({meta.get(field)!r} != {base.get(field)!r})."
                )
        if meta.get("categorical_storage", {}).get("mode") != base.get("categorical_storage", {}).get("mode"):
            raise ValueError(f"cache {path} is not concatenable: categorical_storage mode mismatch.")
        if meta.get("categorical_storage", {}).get("original_feature_count") != base.get(
            "categorical_storage", {}
        ).get("original_feature_count"):
            raise ValueError(f"cache {path} is not concatenable: categorical original_feature_count mismatch.")

    array_names = sorted(p.stem for p in (normalized[0]).glob("*.npy"))
    for path in normalized[1:]:
        if sorted(p.stem for p in path.glob("*.npy")) != array_names:
            raise ValueError(f"cache {path} is not concatenable: array set differs from {normalized[0]}.")

    loaded: dict[str, list[Any]] = {name: [] for name in array_names}
    for path in normalized:
        for name in array_names:
            loaded[name].append(numpy.load(path / f"{name}.npy", allow_pickle=False))

    # Categorical compaction width: pad every part to the widest with zeros.
    cat_parts = loaded["categorical_ids"]
    max_width = max(part.shape[2] for part in cat_parts)
    for index, part in enumerate(cat_parts):
        if part.shape[2] < max_width:
            padded = numpy.zeros((part.shape[0], part.shape[1], max_width), dtype=part.dtype)
            padded[:, :, : part.shape[2]] = part
            cat_parts[index] = padded

    example_counts = [int(meta["example_count"]) for meta in metadatas]
    merged: dict[str, Any] = {}
    for name in array_names:
        parts = loaded[name]
        if name in _ZERO_ROW_ARRAYS:
            # Keep the first cache's zero row; drop the others'.
            merged[name] = numpy.concatenate([parts[0], *[part[1:] for part in parts[1:]]], axis=0)
        elif name == "window_indices":
            offset = 0
            shifted = []
            for part, count in zip(parts, example_counts, strict=True):
                adjusted = part.astype(numpy.uint32, copy=True)
                adjusted[adjusted != 0] += offset
                shifted.append(adjusted)
                offset += count
            merged[name] = numpy.concatenate(shifted, axis=0)
        else:
            merged[name] = numpy.concatenate(parts, axis=0)

    # Belief-source merge mirrors the builder: single common hash survives,
    # anything else (or an already-mixed part) reports mixed.
    hashes: set[Any] = set()
    mixed = False
    for meta in metadatas:
        if meta.get("belief_set_source_mixed"):
            mixed = True
        hashes.add(meta.get("belief_set_source_hash"))
    mixed = mixed or len(hashes) > 1
    provenance: list[Any] = []
    for meta in metadatas:
        provenance.extend(meta.get("opponent_pool_provenance") or ())
    record_count = sum(int(meta["record_count"]) for meta in metadatas)

    metadata = dict(base)
    metadata["record_count"] = record_count
    metadata["example_count"] = sum(example_counts)
    metadata["array_dtypes"] = {name: str(value.dtype) for name, value in merged.items()}
    metadata["observation_shapes"] = dict(base["observation_shapes"])
    metadata["observation_shapes"]["categorical_ids"] = list(merged["categorical_ids"].shape[1:])
    metadata["belief_set_source_hash"] = next(iter(hashes)) if not mixed and len(hashes) == 1 else None
    metadata["belief_set_source_mixed"] = mixed
    metadata["opponent_pool_provenance"] = provenance
    metadata["opponent_pool_provenance_count"] = len(provenance)
    metadata["opponent_pool_provenance_mixed"] = 0 < len(provenance) < record_count
    metadata["categorical_storage"] = dict(base["categorical_storage"])
    metadata["categorical_storage"]["stored_feature_count"] = int(merged["categorical_ids"].shape[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    try:
        for name, value in merged.items():
            numpy.save(temp_path / f"{name}.npy", value, allow_pickle=False)
        (temp_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        temp_path.rename(output_path)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return TrainingCacheSummary(
        path=output_path,
        record_count=record_count,
        example_count=int(metadata["example_count"]),
        byte_size=_directory_byte_size(output_path),
    )


def iter_training_examples(
    paths: PathInput | Iterable[PathInput],
    *,
    config: TrajectoryDatasetConfig | None = None,
) -> Iterator[TrajectoryExample]:
    dataset_config = config or TrajectoryDatasetConfig()
    for path in _normalize_paths(paths):
        for record in iter_rollout_records(path):
            yield from examples_from_record(record, config=dataset_config)


def examples_from_record(
    record: RolloutRecord,
    *,
    config: TrajectoryDatasetConfig | None = None,
) -> Iterator[TrajectoryExample]:
    dataset_config = config or TrajectoryDatasetConfig()
    returns_by_step_index = _discounted_returns_by_step_index(
        record,
        config=dataset_config,
    )
    ppo_targets_by_step_index = _ppo_targets_by_step_index(
        record,
        config=dataset_config,
    )
    potential_terms_by_step_index = _potential_shaping_terms(record, config=dataset_config)
    history_by_player: dict[str, list[TrajectoryStep]] = {}

    # Data-side one-way door: refuse legacy/unversioned observations HERE, with the clean
    # pinned-tag message, instead of letting 44-wide v1 rows die as a bare matmul shape error
    # inside the model. One check per record (all steps share a battle's encoding).
    if record.trajectory.steps:
        require_current_observation_schema(
            record.trajectory.steps[0].observation.schema_version,
            context=f"rollout record {record.trajectory.battle_id!r}",
        )

    for step_index, step in enumerate(record.trajectory.steps):
        player_history = history_by_player.setdefault(step.player_id, [])
        player_history.append(step)
        window_steps = tuple(player_history[-dataset_config.window_size :])
        yield _example_from_window(
            record=record,
            step=step,
            window_steps=window_steps,
            return_value=returns_by_step_index[step_index],
            ppo_target=ppo_targets_by_step_index.get(step_index),
            window_size=dataset_config.window_size,
            shaping_reward=(
                potential_terms_by_step_index.get(step_index, 0.0)
                if dataset_config.potential_shaping is not None
                else None
            ),
        )


def batch_training_examples(
    examples: Iterable[TrajectoryExample],
    *,
    batch_size: int,
) -> Iterator[TrainingBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    buffer: list[TrajectoryExample] = []
    for example in examples:
        buffer.append(example)
        if len(buffer) == batch_size:
            yield training_batch_from_examples(buffer)
            buffer = []
    if buffer:
        yield training_batch_from_examples(buffer)


def iter_training_batches(
    paths: PathInput | Iterable[PathInput],
    *,
    batch_size: int,
    config: TrajectoryDatasetConfig | None = None,
    consumed_cache_callback: Callable[[Path], None] | None = None,
    defer_cache_window_expansion: bool = False,
) -> Iterator[TrainingBatch]:
    normalized_paths = _normalize_paths(paths)
    cache_flags = tuple(is_training_cache_path(path) for path in normalized_paths)
    if any(cache_flags):
        if not all(cache_flags):
            raise ValueError("training cache directories cannot be mixed with rollout JSONL paths.")
        yield from _iter_coalesced_training_cache_batches(
            normalized_paths,
            batch_size=batch_size,
            config=config,
            consumed_cache_callback=consumed_cache_callback,
            defer_window_expansion=defer_cache_window_expansion,
        )
        return
    if consumed_cache_callback is not None:
        raise ValueError("consumed_cache_callback requires training cache directories.")
    yield from batch_training_examples(
        iter_training_examples(normalized_paths, config=config),
        batch_size=batch_size,
    )


def iter_training_batches_with_capped_auxiliary(
    primary_paths: PathInput | Iterable[PathInput],
    *,
    auxiliary_paths: PathInput | Iterable[PathInput],
    auxiliary_max_fraction: float,
    batch_size: int,
    config: TrajectoryDatasetConfig | None = None,
    consumed_cache_callback: Callable[[Path], None] | None = None,
    defer_cache_window_expansion: bool = False,
) -> Iterator[TrainingBatch]:
    """Stream primary batches plus a capped auxiliary-example mix.

    ``auxiliary_max_fraction`` bounds auxiliary examples as a fraction of the
    emitted total, i.e. auxiliary / (primary + auxiliary). This is intended for
    sparse corrective caches such as certified refutation examples, where the
    caller wants the data to influence training without dominating an epoch.
    """

    if not 0.0 < auxiliary_max_fraction < 1.0:
        raise ValueError("auxiliary_max_fraction must be greater than 0 and less than 1.")
    primary_seen = 0
    auxiliary_seen = 0
    auxiliary_ratio = auxiliary_max_fraction / (1.0 - auxiliary_max_fraction)
    auxiliary_iterator = iter_training_batches(
        auxiliary_paths,
        batch_size=batch_size,
        config=config,
        consumed_cache_callback=None,
        defer_cache_window_expansion=defer_cache_window_expansion,
    )
    pending_auxiliary: TrainingBatch | None = None
    auxiliary_exhausted = False

    for primary_batch in iter_training_batches(
        primary_paths,
        batch_size=batch_size,
        config=config,
        consumed_cache_callback=consumed_cache_callback,
        defer_cache_window_expansion=defer_cache_window_expansion,
    ):
        yield primary_batch
        primary_seen += primary_batch.batch_size
        allowed_auxiliary = int(primary_seen * auxiliary_ratio) - auxiliary_seen
        while allowed_auxiliary > 0 and not auxiliary_exhausted:
            if pending_auxiliary is None:
                try:
                    pending_auxiliary = next(auxiliary_iterator)
                except StopIteration:
                    auxiliary_exhausted = True
                    break
            emit_count = min(allowed_auxiliary, pending_auxiliary.batch_size)
            emitted = slice_training_batch(pending_auxiliary, 0, emit_count)
            yield emitted
            auxiliary_seen += emit_count
            allowed_auxiliary -= emit_count
            if emit_count == pending_auxiliary.batch_size:
                pending_auxiliary = None
            else:
                pending_auxiliary = slice_training_batch(
                    pending_auxiliary,
                    emit_count,
                    pending_auxiliary.batch_size,
                )


def write_training_cache_from_rollouts(
    paths: PathInput | Iterable[PathInput],
    output_path: PathInput,
    *,
    config: TrajectoryDatasetConfig | None = None,
    overwrite: bool = False,
    max_cache_root_bytes: int | None = MAX_ACTIVE_TRAINING_CACHE_BYTES,
    cache_root: PathInput | None = None,
) -> TrainingCacheSummary:
    return write_training_cache_streaming(
        paths,
        output_path,
        config=config,
        overwrite=overwrite,
        max_cache_root_bytes=max_cache_root_bytes,
        cache_root=cache_root,
    )


def write_training_cache_streaming(
    paths: PathInput | Iterable[PathInput],
    output_path: PathInput,
    *,
    config: TrajectoryDatasetConfig | None = None,
    overwrite: bool = False,
    max_cache_root_bytes: int | None = MAX_ACTIVE_TRAINING_CACHE_BYTES,
    cache_root: PathInput | None = None,
    flush_rows: int = 20000,
) -> TrainingCacheSummary:
    """Stream rollout JSONL into a training cache with bounded memory.

    The in-memory builder materialises every example's per-token rows before writing, which is
    O(corpus) RAM and OOMs on large obsv2 corpora. This makes two passes over the rollouts:
      pass 1 counts examples, captures row shapes + the global categorical compaction width, and
             accumulates only the small per-example arrays (window indices, masks, scalars);
      pass 2 fills the four big per-token arrays into on-disk .npy memmaps in fixed-size chunks.
    Output is byte-identical to ``TrainingCacheBuilder.write`` (same array names, dtypes, zero-row
    padding, and categorical compaction), so cache readers are unaffected. Peak memory is the small
    arrays plus one ``flush_rows`` chunk, not the whole corpus.
    """
    numpy = _require_numpy()
    cfg = config or TrajectoryDatasetConfig()
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"training cache already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized = list(_normalize_paths(paths))

    uint16_max = int(numpy.iinfo(numpy.uint16).max)
    uint8_max = int(numpy.iinfo(numpy.uint8).max)

    # ---- pass 1: counts, shapes, compaction width, small arrays, window indices ----
    n = 0
    record_count = 0
    token_count: int | None = None
    cat_width: int | None = None
    numeric_width: int | None = None
    max_nonzero = 0
    window_indices: list[tuple[int, ...]] = []
    legal_masks: list[Any] = []
    small: dict[str, list[Any]] = {k: [] for k in (
        "action_indices", "rewards", "returns", "value_estimates", "value_estimate_mask",
        "ppo_advantages", "ppo_advantage_mask", "ppo_value_targets", "ppo_value_target_mask",
        "opponent_action_indices", "opponent_action_mask", "action_probabilities",
        "action_probability_mask", "training_weights", "seeds", "turn_indices", "terminal_capped",
    )}
    shaping_rewards: list[Any] = []  # optional; populated only when potential_shaping is enabled
    belief_set_source_hashes: set[str | None] = set()
    opponent_pool_provenance: list[dict[str, Any]] = []
    for path in normalized:
        for record in iter_rollout_records(path):
            history_by_player: dict[str, list[int]] = {}
            for example in examples_from_record(record, config=cfg):
                row_index = n + 1
                history = history_by_player.setdefault(example.player_id, [])
                history.append(row_index)
                window = tuple(history[-cfg.window_size:])
                padding_count = cfg.window_size - len(window)
                window_indices.append(tuple(0 for _ in range(padding_count)) + window)
                cat_row = numpy.asarray(example.categorical_ids[-1])
                if token_count is None:
                    token_count = int(cat_row.shape[0])
                    cat_width = int(cat_row.shape[1])
                    numeric_width = int(numpy.asarray(example.numeric_features[-1]).shape[1])
                if cat_row.size:
                    if int(cat_row.min()) < 0 or int(cat_row.max()) > uint16_max:
                        raise ValueError("categorical ids exceed uint16 training-cache range.")
                    max_nonzero = max(max_nonzero, int((cat_row != 0).sum(axis=1).max()))
                tt_row = numpy.asarray(example.token_type_ids[-1])
                if tt_row.size:
                    if int(tt_row.min()) < 0 or int(tt_row.max()) > uint8_max:
                        raise ValueError("token type ids exceed uint8 training-cache range.")
                legal_masks.append(example.legal_action_mask)
                small["action_indices"].append(example.action_index)
                small["rewards"].append(example.reward)
                small["returns"].append(example.return_value)
                small["value_estimates"].append(_optional_float(example.value_estimate))
                small["value_estimate_mask"].append(example.value_estimate is not None)
                small["ppo_advantages"].append(_optional_float(example.ppo_advantage))
                small["ppo_advantage_mask"].append(example.ppo_advantage is not None)
                small["ppo_value_targets"].append(_optional_float(example.ppo_value_target))
                small["ppo_value_target_mask"].append(example.ppo_value_target is not None)
                small["opponent_action_indices"].append(_optional_action_index(example.opponent_action_index))
                small["opponent_action_mask"].append(example.opponent_action_index is not None)
                small["action_probabilities"].append(_optional_float(example.action_probability))
                small["action_probability_mask"].append(example.action_probability is not None)
                small["training_weights"].append(float(example.training_weight))
                small["seeds"].append(example.seed)
                small["turn_indices"].append(example.turn_index)
                small["terminal_capped"].append(example.terminal_capped)
                if cfg.potential_shaping is not None:
                    shaping_rewards.append(_optional_float(example.shaping_reward))
                n += 1
            provenance = _opponent_pool_provenance_from_record(record)
            if provenance is not None:
                opponent_pool_provenance.append(provenance)
            belief_set_source_hashes.add(record.belief_set_source_hash)
            record_count += 1
    if n == 0:
        raise ValueError("training cache cannot be written with zero examples.")
    assert token_count is not None and cat_width is not None and numeric_width is not None
    # Stored categorical width = what _compact_categorical_rows would produce (full if no gain).
    stored_cat_width = cat_width if cat_width <= 1 else min(max(1, max_nonzero), cat_width)
    rows = n + 1

    if max_cache_root_bytes is not None:
        if max_cache_root_bytes <= 0:
            raise ValueError("max_cache_root_bytes must be positive.")
        root = Path(cache_root) if cache_root is not None else out.parent
        current_bytes = _directory_byte_size(root) if root.exists() else 0
        est = (
            rows * token_count * stored_cat_width * 2   # categorical uint16
            + rows * token_count * numeric_width * 2     # numeric float16
            + rows * token_count                         # token_type uint8
            + rows * token_count                         # attention bool
            + n * cfg.window_size * 4                    # window uint32
            + n * ACTION_COUNT                           # legal_action_mask bool
            + n * 48                                     # ~scalar arrays (overestimate)
        )
        est += max(16 * 1024 * 1024, est // 100)
        if current_bytes + est > max_cache_root_bytes:
            raise ValueError(
                f"training cache write would exceed storage cap: existing={current_bytes} bytes "
                f"estimated_new={est} bytes limit={max_cache_root_bytes} bytes."
            )

    temp_path = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        cat_mm = numpy.lib.format.open_memmap(temp_path / "categorical_ids.npy", mode="w+", dtype=numpy.uint16, shape=(rows, token_count, stored_cat_width))
        num_mm = numpy.lib.format.open_memmap(temp_path / "numeric_features.npy", mode="w+", dtype=numpy.float16, shape=(rows, token_count, numeric_width))
        tt_mm = numpy.lib.format.open_memmap(temp_path / "token_type_ids.npy", mode="w+", dtype=numpy.uint8, shape=(rows, token_count))
        att_mm = numpy.lib.format.open_memmap(temp_path / "attention_mask.npy", mode="w+", dtype=numpy.bool_, shape=(rows, token_count))
        cat_mm[0] = 0
        num_mm[0] = 0
        tt_mm[0] = 0
        att_mm[0] = False
        buf_cat: list[Any] = []
        buf_num: list[Any] = []
        buf_tt: list[Any] = []
        buf_att: list[Any] = []
        pos = 1

        def _flush() -> None:
            nonlocal pos
            if not buf_cat:
                return
            chunk = _compact_categorical_rows(numpy, numpy.asarray(buf_cat).astype(numpy.uint16, copy=False), stored_cat_width)
            b = chunk.shape[0]
            cat_mm[pos:pos + b] = chunk
            num_mm[pos:pos + b] = numpy.asarray(buf_num, dtype=numpy.float16)
            tt_mm[pos:pos + b] = numpy.asarray(buf_tt, dtype=numpy.uint8)
            att_mm[pos:pos + b] = numpy.asarray(buf_att, dtype=numpy.bool_)
            pos += b
            buf_cat.clear(); buf_num.clear(); buf_tt.clear(); buf_att.clear()

        for path in normalized:
            for record in iter_rollout_records(path):
                for example in examples_from_record(record, config=cfg):
                    buf_cat.append(example.categorical_ids[-1])
                    buf_num.append(example.numeric_features[-1])
                    buf_tt.append(example.token_type_ids[-1])
                    buf_att.append(example.attention_mask[-1])
                    if len(buf_cat) >= flush_rows:
                        _flush()
        _flush()
        for mm in (cat_mm, num_mm, tt_mm, att_mm):
            mm.flush()
        del cat_mm, num_mm, tt_mm, att_mm

        numpy.save(temp_path / "window_indices.npy", numpy.asarray(window_indices, dtype=numpy.uint32))
        numpy.save(temp_path / "legal_action_mask.npy", numpy.asarray(legal_masks, dtype=numpy.bool_))
        _scalar_dtypes = {
            "action_indices": numpy.int16, "rewards": numpy.float32, "returns": numpy.float32,
            "value_estimates": numpy.float32, "value_estimate_mask": numpy.bool_,
            "ppo_advantages": numpy.float32, "ppo_advantage_mask": numpy.bool_,
            "ppo_value_targets": numpy.float32, "ppo_value_target_mask": numpy.bool_,
            "opponent_action_indices": numpy.int16, "opponent_action_mask": numpy.bool_,
            "action_probabilities": numpy.float32, "action_probability_mask": numpy.bool_,
            "training_weights": numpy.float32,
            "seeds": numpy.int64, "turn_indices": numpy.int32, "terminal_capped": numpy.bool_,
        }
        for name, dtype in _scalar_dtypes.items():
            numpy.save(temp_path / f"{name}.npy", numpy.asarray(small[name], dtype=dtype))
        # Optional shaping array: only for shaping-enabled caches (raw env reward stays in `rewards`).
        if cfg.potential_shaping is not None:
            numpy.save(temp_path / "shaping_rewards.npy", numpy.asarray(shaping_rewards, dtype=numpy.float32))

        metadata = {
            "schema_version": TRAINING_CACHE_SCHEMA_VERSION,
            "dataset_config": cfg.to_dict(),
            "record_count": record_count,
            "example_count": n,
            "observation_shapes": {
                "categorical_ids": [token_count, stored_cat_width],
                "numeric_features": [token_count, numeric_width],
                "token_type_ids": [token_count],
                "attention_mask": [token_count],
                "legal_action_mask": [ACTION_COUNT],
                "window_size": cfg.window_size,
            },
            "array_dtypes": {
                "categorical_ids": "uint16", "numeric_features": "float16",
                "token_type_ids": "uint8", "attention_mask": "bool",
                "window_indices": "uint32", "legal_action_mask": "bool",
                **{name: str(numpy.dtype(dtype)) for name, dtype in _scalar_dtypes.items()},
                **({"shaping_rewards": "float32"} if cfg.potential_shaping is not None else {}),
            },
            "belief_set_source_hash": (
                next(iter(belief_set_source_hashes))
                if len(belief_set_source_hashes) == 1
                else None
            ),
            "belief_set_source_mixed": len(belief_set_source_hashes) > 1,
            "opponent_pool_provenance": opponent_pool_provenance,
            "opponent_pool_provenance_count": len(opponent_pool_provenance),
            "opponent_pool_provenance_mixed": 0 < len(opponent_pool_provenance) < record_count,
            # feature_masks / observation_schema mirror the from-rollouts builder, which is
            # constructed without them on this path (legacy None); a future caller threading
            # them through should set them here identically.
            "feature_masks": _feature_masks_payload(None),
            "observation_schema": None,
            "format": "directory-of-npy-arrays",
            "padding_row": 0,
            "categorical_storage": {
                "mode": "compact-nonzero",
                "original_feature_count": int(cat_width),
                "stored_feature_count": int(stored_cat_width),
                "semantic": "summed category embeddings are identical to dense zero-padded rows",
            },
        }
        (temp_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
        temp_path.rename(out)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return TrainingCacheSummary(path=out, record_count=record_count, example_count=n, byte_size=_directory_byte_size(out))


def write_training_cache_from_examples(
    examples: Iterable[TrajectoryExample],
    output_path: PathInput,
    *,
    config: TrajectoryDatasetConfig | None = None,
    overwrite: bool = False,
    max_cache_root_bytes: int | None = MAX_ACTIVE_TRAINING_CACHE_BYTES,
    cache_root: PathInput | None = None,
) -> TrainingCacheSummary:
    builder = TrainingCacheBuilder(config=config)
    for example in examples:
        builder.add_example(example)
    return builder.write(
        output_path,
        overwrite=overwrite,
        max_cache_root_bytes=max_cache_root_bytes,
        cache_root=cache_root,
    )


def _iter_coalesced_training_cache_batches(
    paths: Sequence[Path],
    *,
    batch_size: int,
    config: TrajectoryDatasetConfig | None,
    consumed_cache_callback: Callable[[Path], None] | None,
    defer_window_expansion: bool,
) -> Iterator[TrainingBatch]:
    pending: list[TrainingBatch] = []
    pending_size = 0
    for path in paths:
        for batch in iter_training_cache_batches(
            path,
            batch_size=batch_size,
            config=config,
            defer_window_expansion=defer_window_expansion,
        ):
            remainder: TrainingBatch | None = batch
            while remainder is not None:
                available = batch_size - pending_size
                if remainder.batch_size <= available:
                    pending.append(remainder)
                    pending_size += remainder.batch_size
                    remainder = None
                else:
                    pending.append(slice_training_batch(remainder, 0, available))
                    pending_size += available
                    remainder = slice_training_batch(remainder, available, remainder.batch_size)
                if pending_size == batch_size:
                    yield _combine_training_batches(pending)
                    pending = []
                    pending_size = 0
        if consumed_cache_callback is not None:
            consumed_cache_callback(path)
    if pending:
        yield _combine_training_batches(pending)


def _combine_training_batches(batches: Sequence[TrainingBatch]) -> TrainingBatch:
    if not batches:
        raise ValueError("cannot combine zero training batches.")
    if len(batches) == 1:
        return batches[0]
    if any(batch.window_row_indices is not None for batch in batches):
        return _combine_row_indexed_training_batches(batches)
    return TrainingBatch(
        categorical_ids=_concat_categorical_batch_field(tuple(batch.categorical_ids for batch in batches)),
        numeric_features=_concat_batch_field(tuple(batch.numeric_features for batch in batches)),
        token_type_ids=_concat_batch_field(tuple(batch.token_type_ids for batch in batches)),
        attention_mask=_concat_batch_field(tuple(batch.attention_mask for batch in batches)),
        history_mask=_concat_batch_field(tuple(batch.history_mask for batch in batches)),
        legal_action_mask=_concat_batch_field(tuple(batch.legal_action_mask for batch in batches)),
        action_indices=_concat_batch_field(tuple(batch.action_indices for batch in batches)),
        rewards=_concat_batch_field(tuple(batch.rewards for batch in batches)),
        returns=_concat_batch_field(tuple(batch.returns for batch in batches)),
        value_estimates=_concat_batch_field(tuple(batch.value_estimates for batch in batches)),
        value_estimate_mask=_concat_batch_field(tuple(batch.value_estimate_mask for batch in batches)),
        ppo_advantages=_concat_batch_field(tuple(batch.ppo_advantages for batch in batches)),
        ppo_advantage_mask=_concat_batch_field(tuple(batch.ppo_advantage_mask for batch in batches)),
        ppo_value_targets=_concat_batch_field(tuple(batch.ppo_value_targets for batch in batches)),
        ppo_value_target_mask=_concat_batch_field(tuple(batch.ppo_value_target_mask for batch in batches)),
        opponent_action_indices=_concat_batch_field(tuple(batch.opponent_action_indices for batch in batches)),
        opponent_action_mask=_concat_batch_field(tuple(batch.opponent_action_mask for batch in batches)),
        action_probabilities=_concat_batch_field(tuple(batch.action_probabilities for batch in batches)),
        action_probability_mask=_concat_batch_field(tuple(batch.action_probability_mask for batch in batches)),
        training_weights=_concat_batch_field(tuple(batch.training_weights for batch in batches)),
        battle_ids=_concat_batch_field(tuple(batch.battle_ids for batch in batches)),
        seeds=_concat_batch_field(tuple(batch.seeds for batch in batches)),
        format_ids=_concat_batch_field(tuple(batch.format_ids for batch in batches)),
        player_ids=_concat_batch_field(tuple(batch.player_ids for batch in batches)),
        turn_indices=_concat_batch_field(tuple(batch.turn_indices for batch in batches)),
        terminal_capped=_concat_batch_field(tuple(batch.terminal_capped for batch in batches)),
        step_metadata=_concat_batch_field(tuple(batch.step_metadata for batch in batches)),
    )


def _combine_row_indexed_training_batches(batches: Sequence[TrainingBatch]) -> TrainingBatch:
    if not all(batch.window_row_indices is not None for batch in batches):
        raise ValueError("cannot combine row-indexed and expanded training batches.")
    row_offsets: list[int] = []
    next_offset = 0
    for batch in batches:
        row_offsets.append(next_offset)
        next_offset += len(batch.row_categorical_ids)
    adjusted_window_indices = []
    for batch, offset in zip(batches, row_offsets, strict=True):
        adjusted_window_indices.append(batch.window_row_indices + offset)
    return TrainingBatch(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        history_mask=_concat_batch_field(tuple(batch.history_mask for batch in batches)),
        legal_action_mask=_concat_batch_field(tuple(batch.legal_action_mask for batch in batches)),
        action_indices=_concat_batch_field(tuple(batch.action_indices for batch in batches)),
        rewards=_concat_batch_field(tuple(batch.rewards for batch in batches)),
        returns=_concat_batch_field(tuple(batch.returns for batch in batches)),
        value_estimates=_concat_batch_field(tuple(batch.value_estimates for batch in batches)),
        value_estimate_mask=_concat_batch_field(tuple(batch.value_estimate_mask for batch in batches)),
        ppo_advantages=_concat_batch_field(tuple(batch.ppo_advantages for batch in batches)),
        ppo_advantage_mask=_concat_batch_field(tuple(batch.ppo_advantage_mask for batch in batches)),
        ppo_value_targets=_concat_batch_field(tuple(batch.ppo_value_targets for batch in batches)),
        ppo_value_target_mask=_concat_batch_field(tuple(batch.ppo_value_target_mask for batch in batches)),
        opponent_action_indices=_concat_batch_field(tuple(batch.opponent_action_indices for batch in batches)),
        opponent_action_mask=_concat_batch_field(tuple(batch.opponent_action_mask for batch in batches)),
        action_probabilities=_concat_batch_field(tuple(batch.action_probabilities for batch in batches)),
        action_probability_mask=_concat_batch_field(tuple(batch.action_probability_mask for batch in batches)),
        training_weights=_concat_batch_field(tuple(batch.training_weights for batch in batches)),
        battle_ids=_concat_batch_field(tuple(batch.battle_ids for batch in batches)),
        seeds=_concat_batch_field(tuple(batch.seeds for batch in batches)),
        format_ids=_concat_batch_field(tuple(batch.format_ids for batch in batches)),
        player_ids=_concat_batch_field(tuple(batch.player_ids for batch in batches)),
        turn_indices=_concat_batch_field(tuple(batch.turn_indices for batch in batches)),
        terminal_capped=_concat_batch_field(tuple(batch.terminal_capped for batch in batches)),
        step_metadata=_concat_batch_field(tuple(batch.step_metadata for batch in batches)),
        row_categorical_ids=_concat_categorical_batch_field(tuple(batch.row_categorical_ids for batch in batches)),
        row_numeric_features=_concat_batch_field(tuple(batch.row_numeric_features for batch in batches)),
        row_token_type_ids=_concat_batch_field(tuple(batch.row_token_type_ids for batch in batches)),
        row_attention_mask=_concat_batch_field(tuple(batch.row_attention_mask for batch in batches)),
        window_row_indices=_concat_batch_field(tuple(adjusted_window_indices)),
    )


def _concat_categorical_batch_field(values: Sequence[Any]) -> Any:
    if not values:
        return ()
    if len(values) == 1:
        return values[0]
    first = values[0]
    if not hasattr(first, "shape"):
        return _concat_batch_field(values)
    numpy = _require_numpy()
    max_width = max(int(value.shape[-1]) for value in values)
    padded_values = []
    for value in values:
        width = int(value.shape[-1])
        if width == max_width:
            padded_values.append(value)
            continue
        padding = [(0, 0) for _ in range(len(value.shape))]
        padding[-1] = (0, max_width - width)
        padded_values.append(numpy.pad(value, padding, mode="constant"))
    return numpy.concatenate(padded_values, axis=0)


def _concat_batch_field(values: Sequence[Any]) -> Any:
    if not values:
        return ()
    if len(values) == 1:
        return values[0]
    first = values[0]
    if hasattr(first, "shape"):
        numpy = _require_numpy()
        return numpy.concatenate(values, axis=0)
    combined: list[Any] = []
    for value in values:
        combined.extend(tuple(value))
    return tuple(combined)


def slice_training_batch(batch: TrainingBatch, start: int, stop: int) -> TrainingBatch:
    """Return a contiguous example slice while retaining only its referenced cache rows.

    Distributed training uses this to split an already deterministic global batch into
    contiguous rank-local views. The cache-row remap keeps the deferred window path
    memory-bounded instead of materializing the full batch on every rank.
    """
    if start < 0 or stop < start or stop > batch.batch_size:
        raise ValueError("invalid training batch slice.")
    row_categorical_ids = batch.row_categorical_ids
    row_numeric_features = batch.row_numeric_features
    row_token_type_ids = batch.row_token_type_ids
    row_attention_mask = batch.row_attention_mask
    window_row_indices = None
    if batch.window_row_indices is not None:
        numpy = _require_numpy()
        sliced_window_indices = _slice_batch_field(batch.window_row_indices, start, stop)
        unique_row_indices, inverse_indices = numpy.unique(sliced_window_indices, return_inverse=True)
        window_row_indices = _owned_array(
            inverse_indices.reshape(sliced_window_indices.shape).astype(sliced_window_indices.dtype, copy=False)
        )
        row_categorical_ids = _owned_array(batch.row_categorical_ids[unique_row_indices])
        row_numeric_features = _owned_array(batch.row_numeric_features[unique_row_indices])
        row_token_type_ids = _owned_array(batch.row_token_type_ids[unique_row_indices])
        row_attention_mask = _owned_array(batch.row_attention_mask[unique_row_indices])
    return TrainingBatch(
        categorical_ids=_slice_batch_field(batch.categorical_ids, start, stop),
        numeric_features=_slice_batch_field(batch.numeric_features, start, stop),
        token_type_ids=_slice_batch_field(batch.token_type_ids, start, stop),
        attention_mask=_slice_batch_field(batch.attention_mask, start, stop),
        history_mask=_slice_batch_field(batch.history_mask, start, stop),
        legal_action_mask=_slice_batch_field(batch.legal_action_mask, start, stop),
        action_indices=_slice_batch_field(batch.action_indices, start, stop),
        rewards=_slice_batch_field(batch.rewards, start, stop),
        returns=_slice_batch_field(batch.returns, start, stop),
        value_estimates=_slice_batch_field(batch.value_estimates, start, stop),
        value_estimate_mask=_slice_batch_field(batch.value_estimate_mask, start, stop),
        ppo_advantages=_slice_batch_field(batch.ppo_advantages, start, stop),
        ppo_advantage_mask=_slice_batch_field(batch.ppo_advantage_mask, start, stop),
        ppo_value_targets=_slice_batch_field(batch.ppo_value_targets, start, stop),
        ppo_value_target_mask=_slice_batch_field(batch.ppo_value_target_mask, start, stop),
        opponent_action_indices=_slice_batch_field(batch.opponent_action_indices, start, stop),
        opponent_action_mask=_slice_batch_field(batch.opponent_action_mask, start, stop),
        action_probabilities=_slice_batch_field(batch.action_probabilities, start, stop),
        action_probability_mask=_slice_batch_field(batch.action_probability_mask, start, stop),
        training_weights=_slice_batch_field(batch.training_weights, start, stop),
        battle_ids=_slice_batch_field(batch.battle_ids, start, stop),
        seeds=_slice_batch_field(batch.seeds, start, stop),
        format_ids=_slice_batch_field(batch.format_ids, start, stop),
        player_ids=_slice_batch_field(batch.player_ids, start, stop),
        turn_indices=_slice_batch_field(batch.turn_indices, start, stop),
        terminal_capped=_slice_batch_field(batch.terminal_capped, start, stop),
        step_metadata=_slice_batch_field(batch.step_metadata, start, stop),
        row_categorical_ids=row_categorical_ids,
        row_numeric_features=row_numeric_features,
        row_token_type_ids=row_token_type_ids,
        row_attention_mask=row_attention_mask,
        window_row_indices=window_row_indices,
    )


# Private compatibility alias for callers that predate the distributed trainer.
_slice_training_batch = slice_training_batch


def _slice_batch_field(value: Any, start: int, stop: int) -> Any:
    return value[start:stop]


def _compact_categorical_rows(numpy: Any, categorical: Any, compact_width: int | None = None) -> Any:
    """Drop per-token zero padding from cache categorical rows.

    The neural model sums category embeddings across the final categorical-feature dimension, so
    zero padding and feature order are not semantically meaningful. Keeping only nonzero category
    ids preserves the exact summed embedding while reducing both cache size and CPU embedding work
    during cache-backed training.

    ``compact_width`` pins the packed width (used by the streaming writer, which computes the
    global maximum nonzero count in a first pass so every chunk packs to the same width); when
    ``None`` it is derived from ``categorical`` as before.
    """

    if len(categorical.shape) != 3:
        raise ValueError("categorical training-cache rows must be rank 3.")
    original_width = int(categorical.shape[2])
    if original_width <= 1:
        return categorical
    if compact_width is None:
        nonzero_mask = categorical != 0
        compact_width = max(1, int(nonzero_mask.sum(axis=2).max()))
    if compact_width >= original_width:
        return categorical

    compacted = numpy.zeros(
        (*categorical.shape[:2], compact_width),
        dtype=categorical.dtype,
    )
    slot_count_dtype = numpy.uint16 if original_width > int(numpy.iinfo(numpy.uint8).max) else numpy.uint8
    slot_counts = numpy.zeros(categorical.shape[:2], dtype=slot_count_dtype)
    for feature_index in range(original_width):
        values = categorical[:, :, feature_index]
        row_indices, token_indices = numpy.nonzero(values)
        if not len(row_indices):
            continue
        slot_indices = slot_counts[row_indices, token_indices]
        compacted[row_indices, token_indices, slot_indices] = values[row_indices, token_indices]
        slot_counts[row_indices, token_indices] += 1
    return compacted


def is_training_cache_path(path: PathInput) -> bool:
    resolved = Path(path)
    return resolved.is_dir() and (resolved / "metadata.json").is_file()


def training_cache_byte_size(path: PathInput) -> int:
    resolved = Path(path)
    if not is_training_cache_path(resolved):
        raise ValueError(f"not a training cache directory: {resolved}")
    return _directory_byte_size(resolved)


def training_cache_paths_byte_size(paths: PathInput | Iterable[PathInput]) -> int:
    return sum(training_cache_byte_size(path) for path in _normalize_paths(paths))


def training_cache_root_byte_size(path: PathInput) -> int:
    resolved = Path(path)
    if not resolved.exists():
        return 0
    if not resolved.is_dir():
        raise ValueError(f"training cache root is not a directory: {resolved}")
    return _directory_byte_size(resolved)


def delete_training_cache_path(path: PathInput) -> None:
    resolved = Path(path)
    if not is_training_cache_path(resolved):
        raise ValueError(f"not a training cache directory: {resolved}")
    shutil.rmtree(resolved)


def iter_training_cache_batches(
    path: PathInput,
    *,
    batch_size: int,
    config: TrajectoryDatasetConfig | None = None,
    defer_window_expansion: bool = False,
) -> Iterator[TrainingBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    numpy = _require_numpy()
    cache_path = Path(path)
    metadata = _read_training_cache_metadata(cache_path)
    cache_config = TrajectoryDatasetConfig.from_dict(_mapping(metadata["dataset_config"]))
    if config is not None and cache_config.to_dict() != config.to_dict():
        raise ValueError("training cache dataset config does not match requested training config.")
    example_count = int(metadata["example_count"])
    arrays = _load_training_cache_arrays(cache_path, numpy)
    for start in range(0, example_count, batch_size):
        stop = min(example_count, start + batch_size)
        example_slice = slice(start, stop)
        window_indices = _owned_array(arrays["window_indices"][example_slice])
        batch_len = stop - start
        if defer_window_expansion:
            unique_row_indices, inverse_indices = numpy.unique(window_indices, return_inverse=True)
            window_row_indices = _owned_array(inverse_indices.reshape(window_indices.shape).astype(numpy.uint32, copy=False))
            yield TrainingBatch(
                categorical_ids=(),
                numeric_features=(),
                token_type_ids=(),
                attention_mask=(),
                history_mask=_owned_array(window_indices != 0),
                legal_action_mask=_owned_array(arrays["legal_action_mask"][example_slice]),
                action_indices=_owned_array(arrays["action_indices"][example_slice]),
                rewards=_owned_array(arrays["rewards"][example_slice]),
                returns=_owned_array(arrays["returns"][example_slice]),
                value_estimates=_owned_array(arrays["value_estimates"][example_slice]),
                value_estimate_mask=_owned_array(arrays["value_estimate_mask"][example_slice]),
                ppo_advantages=_owned_array(arrays["ppo_advantages"][example_slice]),
                ppo_advantage_mask=_owned_array(arrays["ppo_advantage_mask"][example_slice]),
                ppo_value_targets=_owned_array(arrays["ppo_value_targets"][example_slice]),
                ppo_value_target_mask=_owned_array(arrays["ppo_value_target_mask"][example_slice]),
                opponent_action_indices=_owned_array(arrays["opponent_action_indices"][example_slice]),
                opponent_action_mask=_owned_array(arrays["opponent_action_mask"][example_slice]),
                action_probabilities=_owned_array(arrays["action_probabilities"][example_slice]),
                action_probability_mask=_owned_array(arrays["action_probability_mask"][example_slice]),
                training_weights=_owned_array(arrays["training_weights"][example_slice]),
                battle_ids=tuple("" for _ in range(batch_len)),
                seeds=_owned_array(arrays["seeds"][example_slice]),
                format_ids=tuple("" for _ in range(batch_len)),
                player_ids=tuple("" for _ in range(batch_len)),
                turn_indices=_owned_array(arrays["turn_indices"][example_slice]),
                terminal_capped=_owned_array(arrays["terminal_capped"][example_slice]),
                step_metadata=tuple({} for _ in range(batch_len)),
                row_categorical_ids=_owned_array(arrays["categorical_ids"][unique_row_indices]),
                row_numeric_features=_owned_array(arrays["numeric_features"][unique_row_indices]),
                row_token_type_ids=_owned_array(arrays["token_type_ids"][unique_row_indices]),
                row_attention_mask=_owned_array(arrays["attention_mask"][unique_row_indices]),
                window_row_indices=window_row_indices,
            )
            continue
        yield TrainingBatch(
            categorical_ids=_owned_array(arrays["categorical_ids"][window_indices]),
            numeric_features=_owned_array(arrays["numeric_features"][window_indices]),
            token_type_ids=_owned_array(arrays["token_type_ids"][window_indices]),
            attention_mask=_owned_array(arrays["attention_mask"][window_indices]),
            history_mask=_owned_array(window_indices != 0),
            legal_action_mask=_owned_array(arrays["legal_action_mask"][example_slice]),
            action_indices=_owned_array(arrays["action_indices"][example_slice]),
            rewards=_owned_array(arrays["rewards"][example_slice]),
            returns=_owned_array(arrays["returns"][example_slice]),
            value_estimates=_owned_array(arrays["value_estimates"][example_slice]),
            value_estimate_mask=_owned_array(arrays["value_estimate_mask"][example_slice]),
            ppo_advantages=_owned_array(arrays["ppo_advantages"][example_slice]),
            ppo_advantage_mask=_owned_array(arrays["ppo_advantage_mask"][example_slice]),
            ppo_value_targets=_owned_array(arrays["ppo_value_targets"][example_slice]),
            ppo_value_target_mask=_owned_array(arrays["ppo_value_target_mask"][example_slice]),
            opponent_action_indices=_owned_array(arrays["opponent_action_indices"][example_slice]),
            opponent_action_mask=_owned_array(arrays["opponent_action_mask"][example_slice]),
            action_probabilities=_owned_array(arrays["action_probabilities"][example_slice]),
            action_probability_mask=_owned_array(arrays["action_probability_mask"][example_slice]),
            training_weights=_owned_array(arrays["training_weights"][example_slice]),
            battle_ids=tuple("" for _ in range(batch_len)),
            seeds=_owned_array(arrays["seeds"][example_slice]),
            format_ids=tuple("" for _ in range(batch_len)),
            player_ids=tuple("" for _ in range(batch_len)),
            turn_indices=_owned_array(arrays["turn_indices"][example_slice]),
            terminal_capped=_owned_array(arrays["terminal_capped"][example_slice]),
            step_metadata=tuple({} for _ in range(batch_len)),
        )


def training_batch_from_examples(examples: Sequence[TrajectoryExample]) -> TrainingBatch:
    if not examples:
        raise ValueError("examples must contain at least one item.")
    window_size = examples[0].window_size
    for example in examples:
        if example.window_size != window_size:
            raise ValueError("all examples in a batch must have the same window_size.")

    return TrainingBatch(
        categorical_ids=tuple(example.categorical_ids for example in examples),
        numeric_features=tuple(example.numeric_features for example in examples),
        token_type_ids=tuple(example.token_type_ids for example in examples),
        attention_mask=tuple(example.attention_mask for example in examples),
        history_mask=tuple(example.history_mask for example in examples),
        legal_action_mask=tuple(example.legal_action_mask for example in examples),
        action_indices=tuple(example.action_index for example in examples),
        rewards=tuple(example.reward for example in examples),
        returns=tuple(example.return_value for example in examples),
        value_estimates=tuple(_optional_float(example.value_estimate) for example in examples),
        value_estimate_mask=tuple(example.value_estimate is not None for example in examples),
        ppo_advantages=tuple(_optional_float(example.ppo_advantage) for example in examples),
        ppo_advantage_mask=tuple(example.ppo_advantage is not None for example in examples),
        ppo_value_targets=tuple(_optional_float(example.ppo_value_target) for example in examples),
        ppo_value_target_mask=tuple(example.ppo_value_target is not None for example in examples),
        opponent_action_indices=tuple(_optional_action_index(example.opponent_action_index) for example in examples),
        opponent_action_mask=tuple(example.opponent_action_index is not None for example in examples),
        action_probabilities=tuple(_optional_float(example.action_probability) for example in examples),
        action_probability_mask=tuple(example.action_probability is not None for example in examples),
        training_weights=tuple(float(example.training_weight) for example in examples),
        battle_ids=tuple(example.battle_id for example in examples),
        seeds=tuple(example.seed for example in examples),
        format_ids=tuple(example.format_id for example in examples),
        player_ids=tuple(example.player_id for example in examples),
        turn_indices=tuple(example.turn_index for example in examples),
        terminal_capped=tuple(example.terminal_capped for example in examples),
        step_metadata=tuple(dict(example.step_metadata or {}) for example in examples),
    )


def _example_from_window(
    *,
    record: RolloutRecord,
    step: TrajectoryStep,
    window_steps: tuple[TrajectoryStep, ...],
    return_value: float,
    ppo_target: "_PPOTarget | None",
    window_size: int,
    shaping_reward: float | None = None,
) -> TrajectoryExample:
    if len(window_steps) > window_size:
        raise ValueError("window_steps cannot exceed window_size.")

    observation = step.observation
    padding_count = window_size - len(window_steps)
    categorical_padding = _zeros_like(observation.categorical_ids)
    numeric_padding = _zeros_like(observation.numeric_features)
    token_type_padding = _zeros_like(observation.token_type_ids)
    attention_padding = _zeros_like(observation.attention_mask)

    return TrajectoryExample(
        battle_id=record.battle_id,
        seed=record.seed,
        format_id=record.format_id,
        player_id=step.player_id,
        turn_index=step.turn_index,
        categorical_ids=tuple([categorical_padding] * padding_count)
        + tuple(history_step.observation.categorical_ids for history_step in window_steps),
        numeric_features=tuple([numeric_padding] * padding_count)
        + tuple(history_step.observation.numeric_features for history_step in window_steps),
        token_type_ids=tuple([token_type_padding] * padding_count)
        + tuple(history_step.observation.token_type_ids for history_step in window_steps),
        attention_mask=tuple([attention_padding] * padding_count)
        + tuple(history_step.observation.attention_mask for history_step in window_steps),
        history_mask=tuple(False for _ in range(padding_count)) + tuple(True for _ in window_steps),
        legal_action_mask=tuple(step.legal_action_mask),
        action_index=step.action_index,
        reward=float(step.reward),
        return_value=return_value,
        value_estimate=step.value_estimate,
        ppo_advantage=ppo_target.advantage if ppo_target is not None else None,
        ppo_value_target=ppo_target.value_target if ppo_target is not None else None,
        opponent_action_index=step.opponent_action_index,
        action_probability=step.action_probability,
        step_metadata=dict(step.metadata),
        terminal_capped=bool((record.terminal or record.trajectory.terminal) and (record.terminal or record.trajectory.terminal).capped),
        shaping_reward=shaping_reward,
    )


def _discounted_returns_by_step_index(
    record: RolloutRecord,
    *,
    config: TrajectoryDatasetConfig,
) -> dict[int, float]:
    step_indices_by_player: dict[str, list[int]] = {}
    for step_index, step in enumerate(record.trajectory.steps):
        step_indices_by_player.setdefault(step.player_id, []).append(step_index)

    returns_by_step_index: dict[int, float] = {}
    for player_id, step_indices in step_indices_by_player.items():
        shaping_rewards = _shaping_rewards_by_step_index(record, step_indices=step_indices, config=config)
        running_return = _terminal_value_for_player(
            record,
            player_id,
            capped_terminal_value=config.capped_terminal_value,
        )
        for step_index in reversed(step_indices):
            shaped_return = _clip_return_value(running_return + shaping_rewards.get(step_index, 0.0))
            returns_by_step_index[step_index] = shaped_return
            running_return = shaped_return * config.discount
    return returns_by_step_index


@dataclass(frozen=True)
class _PPOTarget:
    advantage: float
    value_target: float


def _ppo_targets_by_step_index(
    record: RolloutRecord,
    *,
    config: TrajectoryDatasetConfig,
) -> dict[int, _PPOTarget]:
    if config.ppo_target_mode != "gae":
        return {}
    step_indices_by_player: dict[str, list[int]] = {}
    for step_index, step in enumerate(record.trajectory.steps):
        step_indices_by_player.setdefault(step.player_id, []).append(step_index)

    targets_by_step_index: dict[int, _PPOTarget] = {}
    for player_id, step_indices in step_indices_by_player.items():
        steps = [record.trajectory.steps[step_index] for step_index in step_indices]
        if not steps or any(step.value_estimate is None for step in steps):
            continue
        shaping_rewards = _shaping_rewards_by_step_index(record, step_indices=step_indices, config=config)
        terminal_value = _terminal_value_for_player(
            record,
            player_id,
            capped_terminal_value=config.capped_terminal_value,
        )
        running_advantage = 0.0
        for position in range(len(step_indices) - 1, -1, -1):
            step_index = step_indices[position]
            step = record.trajectory.steps[step_index]
            value_estimate = float(step.value_estimate)
            is_last_player_step = position == len(step_indices) - 1
            reward = shaping_rewards.get(step_index, 0.0)
            if is_last_player_step:
                reward += terminal_value
                next_value_estimate = 0.0
            else:
                next_value_estimate = float(record.trajectory.steps[step_indices[position + 1]].value_estimate)
            delta = reward + (config.discount * next_value_estimate) - value_estimate
            running_advantage = delta + (config.discount * config.gae_lambda * running_advantage)
            targets_by_step_index[step_index] = _PPOTarget(
                advantage=running_advantage,
                # Value loss remains clipped to the training target range; policy advantage stays unclipped.
                value_target=_clip_return_value(value_estimate + running_advantage),
            )
    return targets_by_step_index


def _clip_return_value(value: float) -> float:
    return min(1.0, max(-1.0, value))


def _shaping_rewards_by_step_index(
    record: RolloutRecord,
    *,
    step_indices: Sequence[int],
    config: TrajectoryDatasetConfig,
) -> dict[int, float]:
    potential_terms = _potential_shaping_terms(record, config=config)
    if (
        config.hp_delta_return_weight == 0.0
        and config.faint_delta_return_weight == 0.0
        and (config.turn_penalty_after is None or config.turn_penalty == 0.0)
    ):
        return {step_index: potential_terms[step_index] for step_index in step_indices} if potential_terms else {}

    rewards: dict[int, float] = {}
    previous_snapshot: _VisibleTeamSnapshot | None = None
    for step_index in step_indices:
        step = record.trajectory.steps[step_index]
        snapshot = _visible_team_snapshot(step.observation.metadata)
        reward = 0.0
        if previous_snapshot is not None:
            hp_delta, faint_delta = _visible_differential_delta(previous_snapshot, snapshot)
            reward += config.hp_delta_return_weight * hp_delta
            reward += config.faint_delta_return_weight * faint_delta
        if (
            config.turn_penalty_after is not None
            and config.turn_penalty > 0.0
            and step.turn_index >= config.turn_penalty_after
        ):
            reward -= config.turn_penalty
        if potential_terms:
            reward += potential_terms.get(step_index, 0.0)
        rewards[step_index] = reward
        previous_snapshot = snapshot
    return rewards


def _potential_shaping_terms(
    record: RolloutRecord,
    *,
    config: TrajectoryDatasetConfig,
) -> dict[int, float]:
    """Per-step dense shaping terms (empty when shaping is off).

    Always recomputed from the record's ground-truth metadata and selected-action
    metadata via the pure functions in ``pokezero.shaping`` (the single source of
    truth); step-level ``shaping_reward`` annotations, when present, are provenance
    only. Gamma is the training discount for potential-based terms.
    """
    if config.potential_shaping is None or config.potential_shaping.is_zero():
        return {}
    return shaping_rewards_by_step_index(
        record,
        config=config.potential_shaping,
        gamma=config.discount,
    )


@dataclass(frozen=True)
class _VisiblePokemonSnapshot:
    hp_fraction: float
    fainted: bool


@dataclass(frozen=True)
class _VisibleTeamSnapshot:
    self_team: Mapping[str, _VisiblePokemonSnapshot]
    opponent_team: Mapping[str, _VisiblePokemonSnapshot]


def _visible_team_snapshot(metadata: Mapping[str, Any] | None) -> _VisibleTeamSnapshot:
    payload = metadata if isinstance(metadata, Mapping) else {}
    return _VisibleTeamSnapshot(
        self_team=_pokemon_snapshots_by_visible_key(payload.get("self_team")),
        opponent_team=_pokemon_snapshots_by_visible_key(payload.get("opponent_team")),
    )


def _pokemon_snapshots_by_visible_key(value: Any) -> dict[str, _VisiblePokemonSnapshot]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return {}
    snapshots: dict[str, _VisiblePokemonSnapshot] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        key = _visible_pokemon_key(item, fallback=f"slot-{index}")
        if key is None:
            continue
        snapshots[key] = _VisiblePokemonSnapshot(
            hp_fraction=_visible_hp_fraction(item),
            fainted=bool(item.get("fainted", False)),
        )
    return snapshots


def _visible_pokemon_key(item: Mapping[str, Any], *, fallback: str) -> str | None:
    species = item.get("species")
    if isinstance(species, str) and species:
        return species
    ident = item.get("ident")
    if isinstance(ident, str) and ident:
        return ident
    return fallback


def _visible_hp_fraction(item: Mapping[str, Any]) -> float:
    raw = item.get("hp_fraction")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0 if bool(item.get("fainted", False)) else 1.0
    return min(1.0, max(0.0, value))


def _visible_differential_delta(
    previous: _VisibleTeamSnapshot,
    current: _VisibleTeamSnapshot,
) -> tuple[float, float]:
    self_hp_delta, self_faint_delta = _team_delta(previous.self_team, current.self_team)
    opponent_hp_delta, opponent_faint_delta = _team_delta(previous.opponent_team, current.opponent_team)
    # Positive values should mean the player-relative position improved.
    hp_delta = opponent_hp_delta * -1.0 + self_hp_delta
    faint_delta = opponent_faint_delta - self_faint_delta
    return hp_delta, faint_delta


def _team_delta(
    previous: Mapping[str, _VisiblePokemonSnapshot],
    current: Mapping[str, _VisiblePokemonSnapshot],
) -> tuple[float, float]:
    shared_keys = set(previous) & set(current)
    if not shared_keys:
        return 0.0, 0.0
    hp_delta = 0.0
    faint_delta = 0.0
    for key in shared_keys:
        before = previous[key]
        after = current[key]
        hp_delta += after.hp_fraction - before.hp_fraction
        faint_delta += float(after.fainted) - float(before.fainted)
    # Normalize against a full singles team. This keeps scale stable even when
    # only a subset of the opponent team has been publicly revealed.
    return hp_delta / 6.0, faint_delta / 6.0


def _terminal_value_for_player(record: RolloutRecord, player_id: str, *, capped_terminal_value: float) -> float:
    terminal = record.terminal or record.trajectory.terminal
    if terminal is None:
        return 0.0
    if terminal.capped:
        return capped_terminal_value
    if terminal.winner is None:
        return 0.0
    if terminal.winner == player_id:
        return 1.0
    return -1.0


def _normalize_paths(paths: PathInput | Iterable[PathInput]) -> tuple[Path, ...]:
    if isinstance(paths, (str, PathLike)):
        return (Path(paths),)
    return tuple(Path(path) for path in paths)


def _require_numpy() -> Any:
    try:
        import numpy
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for training caches. Install with `pip install -e .[neural]`.") from exc
    return numpy


def _prepend_zero_row(numpy: Any, array: Any) -> Any:
    zero = numpy.zeros((1, *array.shape[1:]), dtype=array.dtype)
    return numpy.concatenate((zero, array), axis=0)


def _read_training_cache_metadata(path: Path) -> Mapping[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"not a training cache directory: {path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid training cache metadata: {metadata_path}")
    if payload.get("schema_version") != TRAINING_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported training cache schema: {payload.get('schema_version')!r}.")
    return payload


def _load_training_cache_arrays(path: Path, numpy: Any) -> dict[str, Any]:
    names = (
        "categorical_ids",
        "numeric_features",
        "token_type_ids",
        "attention_mask",
        "window_indices",
        "legal_action_mask",
        "action_indices",
        "rewards",
        "returns",
        "value_estimates",
        "value_estimate_mask",
        "ppo_advantages",
        "ppo_advantage_mask",
        "ppo_value_targets",
        "ppo_value_target_mask",
        "opponent_action_indices",
        "opponent_action_mask",
        "action_probabilities",
        "action_probability_mask",
        "training_weights",
        "seeds",
        "turn_indices",
        "terminal_capped",
    )
    arrays = {}
    for name in names:
        array_path = path / f"{name}.npy"
        if array_path.exists():
            arrays[name] = numpy.load(array_path, mmap_mode="c")
    missing_required = [
        name
        for name in names
        if name not in arrays and name not in {"value_estimates", "value_estimate_mask", "training_weights"}
    ]
    if missing_required:
        raise FileNotFoundError(f"training cache is missing required arrays: {missing_required}")
    if "value_estimates" not in arrays:
        arrays["value_estimates"] = numpy.zeros_like(arrays["returns"], dtype=numpy.float32)
    if "value_estimate_mask" not in arrays:
        arrays["value_estimate_mask"] = numpy.zeros_like(arrays["returns"], dtype=numpy.bool_)
    if "training_weights" not in arrays:
        arrays["training_weights"] = numpy.ones_like(arrays["returns"], dtype=numpy.float32)
    return arrays


def _owned_array(value: Any) -> Any:
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value


def _directory_byte_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _estimated_training_cache_byte_size(arrays: Mapping[str, Any]) -> int:
    array_bytes = sum(int(getattr(value, "nbytes", 0)) for value in arrays.values())
    # NPY headers and metadata are small, but overestimate so write-time caps fail before
    # the active cache root can cross the requested storage ceiling.
    return array_bytes + max(16 * 1024 * 1024, array_bytes // 100)


def _array_min(numpy: Any, value: Any) -> int:
    if int(getattr(value, "size", 0)) == 0:
        return 0
    return int(numpy.min(value))


def _array_max(numpy: Any, value: Any) -> int:
    if int(getattr(value, "size", 0)) == 0:
        return 0
    return int(numpy.max(value))


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected mapping payload.")
    return value


def _optional_action_index(value: int | None) -> int:
    return MISSING_ACTION_INDEX if value is None else int(value)


def _optional_float(value: float | None) -> float:
    return 0.0 if value is None else float(value)
