"""Streaming trajectory dataset helpers for early training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .actions import ACTION_COUNT
from .collection import RolloutRecord, iter_rollout_records
from .trajectory import TrajectoryStep

MISSING_ACTION_INDEX = -1

PathInput = str | PathLike[str] | Path


@dataclass(frozen=True)
class TrajectoryDatasetConfig:
    """Controls how serialized rollout steps become training examples.

    Discounting is applied per recorded decision for each player. Terminal
    returns are derived from the battle result rather than sparse per-step
    rewards, so asymmetric final rounds still label both players' histories.
    """

    window_size: int = 1
    discount: float = 1.0
    capped_terminal_value: float = 0.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between 0 and 1.")
        if not -1.0 <= self.capped_terminal_value <= 0.0:
            raise ValueError("capped_terminal_value must be between -1 and 0.")


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
        discount=dataset_config.discount,
        capped_terminal_value=dataset_config.capped_terminal_value,
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
) -> Iterator[TrainingBatch]:
    yield from batch_training_examples(
        iter_training_examples(paths, config=config),
        batch_size=batch_size,
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
        opponent_action_index=step.opponent_action_index,
        action_probability=step.action_probability,
        step_metadata=dict(step.metadata),
        terminal_capped=bool((record.terminal or record.trajectory.terminal) and (record.terminal or record.trajectory.terminal).capped),
    )


def _discounted_returns_by_step_index(
    record: RolloutRecord,
    *,
    discount: float,
    capped_terminal_value: float,
) -> dict[int, float]:
    step_indices_by_player: dict[str, list[int]] = {}
    for step_index, step in enumerate(record.trajectory.steps):
        step_indices_by_player.setdefault(step.player_id, []).append(step_index)

    returns_by_step_index: dict[int, float] = {}
    for player_id, step_indices in step_indices_by_player.items():
        running_return = _terminal_value_for_player(record, player_id, capped_terminal_value=capped_terminal_value)
        for step_index in reversed(step_indices):
            returns_by_step_index[step_index] = running_return
            running_return *= discount
    return returns_by_step_index


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
