"""Streaming trajectory dataset helpers for early training experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .actions import ACTION_COUNT
from .collection import RolloutRecord, iter_rollout_records
from .trajectory import TrajectoryStep

MISSING_ACTION_INDEX = -1
TRAINING_CACHE_SCHEMA_VERSION = "pokezero.training_cache.v1"
MAX_ACTIVE_TRAINING_CACHE_GB = 50.0
MAX_ACTIVE_TRAINING_CACHE_BYTES = int(MAX_ACTIVE_TRAINING_CACHE_GB * 1024 * 1024 * 1024)

PathInput = str | PathLike[str] | Path


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

    def __post_init__(self) -> None:
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
    ppo_advantages: tuple[float, ...]
    ppo_advantage_mask: tuple[bool, ...]
    ppo_value_targets: tuple[float, ...]
    ppo_value_target_mask: tuple[bool, ...]
    opponent_action_indices: tuple[int, ...]
    opponent_action_mask: tuple[bool, ...]
    action_probabilities: tuple[float, ...]
    action_probability_mask: tuple[bool, ...]
    battle_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    format_ids: tuple[str, ...]
    player_ids: tuple[str, ...]
    turn_indices: tuple[int, ...]
    terminal_capped: tuple[bool, ...]
    step_metadata: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        batch_size = len(self.action_indices)
        if batch_size == 0:
            raise ValueError("TrainingBatch must contain at least one example.")
        for name, values in (
            ("categorical_ids", self.categorical_ids),
            ("numeric_features", self.numeric_features),
            ("token_type_ids", self.token_type_ids),
            ("attention_mask", self.attention_mask),
            ("history_mask", self.history_mask),
            ("legal_action_mask", self.legal_action_mask),
            ("rewards", self.rewards),
            ("returns", self.returns),
            ("ppo_advantages", self.ppo_advantages),
            ("ppo_advantage_mask", self.ppo_advantage_mask),
            ("ppo_value_targets", self.ppo_value_targets),
            ("ppo_value_target_mask", self.ppo_value_target_mask),
            ("opponent_action_indices", self.opponent_action_indices),
            ("opponent_action_mask", self.opponent_action_mask),
            ("action_probabilities", self.action_probabilities),
            ("action_probability_mask", self.action_probability_mask),
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


class TrainingCacheBuilder:
    """Build a compact, array-backed training cache from rollout records.

    The raw rollout JSONL stores every public observation as nested JSON. The cache stores each
    observation once, uses small numeric dtypes, and represents history windows as integer row
    references. This keeps shard storage bounded while letting training batch with memmapped arrays.
    """

    def __init__(self, *, config: TrajectoryDatasetConfig | None = None) -> None:
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
        self._ppo_advantages: list[float] = []
        self._ppo_advantage_masks: list[bool] = []
        self._ppo_value_targets: list[float] = []
        self._ppo_value_target_masks: list[bool] = []
        self._opponent_action_indices: list[int] = []
        self._opponent_action_masks: list[bool] = []
        self._action_probabilities: list[float] = []
        self._action_probability_masks: list[bool] = []
        self._seeds: list[int] = []
        self._turn_indices: list[int] = []
        self._terminal_capped: list[bool] = []

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
            self._ppo_advantages.append(_optional_float(example.ppo_advantage))
            self._ppo_advantage_masks.append(example.ppo_advantage is not None)
            self._ppo_value_targets.append(_optional_float(example.ppo_value_target))
            self._ppo_value_target_masks.append(example.ppo_value_target is not None)
            self._opponent_action_indices.append(_optional_action_index(example.opponent_action_index))
            self._opponent_action_masks.append(example.opponent_action_index is not None)
            self._action_probabilities.append(_optional_float(example.action_probability))
            self._action_probability_masks.append(example.action_probability is not None)
            self._seeds.append(example.seed)
            self._turn_indices.append(example.turn_index)
            self._terminal_capped.append(example.terminal_capped)
        self._record_count += 1

    def write(
        self,
        path: PathInput,
        *,
        overwrite: bool = False,
        max_cache_root_bytes: int | None = MAX_ACTIVE_TRAINING_CACHE_BYTES,
        cache_root: PathInput | None = None,
    ) -> TrainingCacheSummary:
        if self.example_count == 0:
            raise ValueError("training cache cannot be written with zero examples.")
        numpy = _require_numpy()
        output_path = Path(path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"training cache already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._arrays(numpy)
        if max_cache_root_bytes is not None:
            if max_cache_root_bytes <= 0:
                raise ValueError("max_cache_root_bytes must be positive.")
            root = Path(cache_root) if cache_root is not None else output_path.parent
            current_bytes = _directory_byte_size(root) if root.exists() else 0
            estimated_bytes = _estimated_training_cache_byte_size(arrays)
            if current_bytes + estimated_bytes > max_cache_root_bytes:
                raise ValueError(
                    f"training cache write would exceed storage cap: existing={current_bytes} bytes "
                    f"estimated_new={estimated_bytes} bytes limit={max_cache_root_bytes} bytes."
                )
        temp_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
        try:
            for name, value in arrays.items():
                numpy.save(temp_path / f"{name}.npy", value, allow_pickle=False)
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
        categorical_raw = numpy.asarray(self._categorical_rows)
        numeric = numpy.asarray(self._numeric_rows, dtype=numpy.float16)
        token_type_raw = numpy.asarray(self._token_type_rows)
        if _array_min(numpy, categorical_raw) < 0 or _array_max(numpy, categorical_raw) > int(numpy.iinfo(numpy.uint16).max):
            raise ValueError("categorical ids exceed uint16 training-cache range.")
        if _array_min(numpy, token_type_raw) < 0 or _array_max(numpy, token_type_raw) > int(numpy.iinfo(numpy.uint8).max):
            raise ValueError("token type ids exceed uint8 training-cache range.")
        categorical = _compact_categorical_rows(
            numpy,
            categorical_raw.astype(numpy.uint16, copy=False),
        )
        token_type = token_type_raw.astype(numpy.uint8, copy=False)
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
            "ppo_advantages": numpy.asarray(self._ppo_advantages, dtype=numpy.float32),
            "ppo_advantage_mask": numpy.asarray(self._ppo_advantage_masks, dtype=numpy.bool_),
            "ppo_value_targets": numpy.asarray(self._ppo_value_targets, dtype=numpy.float32),
            "ppo_value_target_mask": numpy.asarray(self._ppo_value_target_masks, dtype=numpy.bool_),
            "opponent_action_indices": numpy.asarray(self._opponent_action_indices, dtype=numpy.int16),
            "opponent_action_mask": numpy.asarray(self._opponent_action_masks, dtype=numpy.bool_),
            "action_probabilities": numpy.asarray(self._action_probabilities, dtype=numpy.float32),
            "action_probability_mask": numpy.asarray(self._action_probability_masks, dtype=numpy.bool_),
            "seeds": numpy.asarray(self._seeds, dtype=numpy.int64),
            "turn_indices": numpy.asarray(self._turn_indices, dtype=numpy.int32),
            "terminal_capped": numpy.asarray(self._terminal_capped, dtype=numpy.bool_),
        }


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
    history_by_player: dict[str, list[TrajectoryStep]] = {}

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
        )
        return
    if consumed_cache_callback is not None:
        raise ValueError("consumed_cache_callback requires training cache directories.")
    yield from batch_training_examples(
        iter_training_examples(normalized_paths, config=config),
        batch_size=batch_size,
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
    builder = TrainingCacheBuilder(config=config)
    for path in _normalize_paths(paths):
        for record in iter_rollout_records(path):
            builder.add_record(record)
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
) -> Iterator[TrainingBatch]:
    pending: list[TrainingBatch] = []
    pending_size = 0
    for path in paths:
        for batch in iter_training_cache_batches(path, batch_size=batch_size, config=config):
            remainder: TrainingBatch | None = batch
            while remainder is not None:
                available = batch_size - pending_size
                if remainder.batch_size <= available:
                    pending.append(remainder)
                    pending_size += remainder.batch_size
                    remainder = None
                else:
                    pending.append(_slice_training_batch(remainder, 0, available))
                    pending_size += available
                    remainder = _slice_training_batch(remainder, available, remainder.batch_size)
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
        ppo_advantages=_concat_batch_field(tuple(batch.ppo_advantages for batch in batches)),
        ppo_advantage_mask=_concat_batch_field(tuple(batch.ppo_advantage_mask for batch in batches)),
        ppo_value_targets=_concat_batch_field(tuple(batch.ppo_value_targets for batch in batches)),
        ppo_value_target_mask=_concat_batch_field(tuple(batch.ppo_value_target_mask for batch in batches)),
        opponent_action_indices=_concat_batch_field(tuple(batch.opponent_action_indices for batch in batches)),
        opponent_action_mask=_concat_batch_field(tuple(batch.opponent_action_mask for batch in batches)),
        action_probabilities=_concat_batch_field(tuple(batch.action_probabilities for batch in batches)),
        action_probability_mask=_concat_batch_field(tuple(batch.action_probability_mask for batch in batches)),
        battle_ids=_concat_batch_field(tuple(batch.battle_ids for batch in batches)),
        seeds=_concat_batch_field(tuple(batch.seeds for batch in batches)),
        format_ids=_concat_batch_field(tuple(batch.format_ids for batch in batches)),
        player_ids=_concat_batch_field(tuple(batch.player_ids for batch in batches)),
        turn_indices=_concat_batch_field(tuple(batch.turn_indices for batch in batches)),
        terminal_capped=_concat_batch_field(tuple(batch.terminal_capped for batch in batches)),
        step_metadata=_concat_batch_field(tuple(batch.step_metadata for batch in batches)),
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


def _slice_training_batch(batch: TrainingBatch, start: int, stop: int) -> TrainingBatch:
    if start < 0 or stop < start or stop > batch.batch_size:
        raise ValueError("invalid training batch slice.")
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
        ppo_advantages=_slice_batch_field(batch.ppo_advantages, start, stop),
        ppo_advantage_mask=_slice_batch_field(batch.ppo_advantage_mask, start, stop),
        ppo_value_targets=_slice_batch_field(batch.ppo_value_targets, start, stop),
        ppo_value_target_mask=_slice_batch_field(batch.ppo_value_target_mask, start, stop),
        opponent_action_indices=_slice_batch_field(batch.opponent_action_indices, start, stop),
        opponent_action_mask=_slice_batch_field(batch.opponent_action_mask, start, stop),
        action_probabilities=_slice_batch_field(batch.action_probabilities, start, stop),
        action_probability_mask=_slice_batch_field(batch.action_probability_mask, start, stop),
        battle_ids=_slice_batch_field(batch.battle_ids, start, stop),
        seeds=_slice_batch_field(batch.seeds, start, stop),
        format_ids=_slice_batch_field(batch.format_ids, start, stop),
        player_ids=_slice_batch_field(batch.player_ids, start, stop),
        turn_indices=_slice_batch_field(batch.turn_indices, start, stop),
        terminal_capped=_slice_batch_field(batch.terminal_capped, start, stop),
        step_metadata=_slice_batch_field(batch.step_metadata, start, stop),
    )


def _slice_batch_field(value: Any, start: int, stop: int) -> Any:
    return value[start:stop]


def _compact_categorical_rows(numpy: Any, categorical: Any) -> Any:
    """Drop per-token zero padding from cache categorical rows.

    The neural model sums category embeddings across the final categorical-feature dimension, so
    zero padding and feature order are not semantically meaningful. Keeping only nonzero category
    ids preserves the exact summed embedding while reducing both cache size and CPU embedding work
    during cache-backed training.
    """

    if len(categorical.shape) != 3:
        raise ValueError("categorical training-cache rows must be rank 3.")
    original_width = int(categorical.shape[2])
    if original_width <= 1:
        return categorical
    nonzero_mask = categorical != 0
    max_nonzero = int(nonzero_mask.sum(axis=2).max())
    compact_width = max(1, max_nonzero)
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
            ppo_advantages=_owned_array(arrays["ppo_advantages"][example_slice]),
            ppo_advantage_mask=_owned_array(arrays["ppo_advantage_mask"][example_slice]),
            ppo_value_targets=_owned_array(arrays["ppo_value_targets"][example_slice]),
            ppo_value_target_mask=_owned_array(arrays["ppo_value_target_mask"][example_slice]),
            opponent_action_indices=_owned_array(arrays["opponent_action_indices"][example_slice]),
            opponent_action_mask=_owned_array(arrays["opponent_action_mask"][example_slice]),
            action_probabilities=_owned_array(arrays["action_probabilities"][example_slice]),
            action_probability_mask=_owned_array(arrays["action_probability_mask"][example_slice]),
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
        ppo_advantages=tuple(_optional_float(example.ppo_advantage) for example in examples),
        ppo_advantage_mask=tuple(example.ppo_advantage is not None for example in examples),
        ppo_value_targets=tuple(_optional_float(example.ppo_value_target) for example in examples),
        ppo_value_target_mask=tuple(example.ppo_value_target is not None for example in examples),
        opponent_action_indices=tuple(_optional_action_index(example.opponent_action_index) for example in examples),
        opponent_action_mask=tuple(example.opponent_action_index is not None for example in examples),
        action_probabilities=tuple(_optional_float(example.action_probability) for example in examples),
        action_probability_mask=tuple(example.action_probability is not None for example in examples),
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
    if (
        config.hp_delta_return_weight == 0.0
        and config.faint_delta_return_weight == 0.0
        and (config.turn_penalty_after is None or config.turn_penalty == 0.0)
    ):
        return {}

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
        rewards[step_index] = reward
        previous_snapshot = snapshot
    return rewards


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
        "ppo_advantages",
        "ppo_advantage_mask",
        "ppo_value_targets",
        "ppo_value_target_mask",
        "opponent_action_indices",
        "opponent_action_mask",
        "action_probabilities",
        "action_probability_mask",
        "seeds",
        "turn_indices",
        "terminal_capped",
    )
    return {name: numpy.load(path / f"{name}.npy", mmap_mode="c") for name in names}


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


def _zeros_like(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    return tuple(_zeros_like(item) for item in value)
