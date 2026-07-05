"""Trajectory containers for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional

from .actions import ACTION_COUNT
from .env import TerminalState
from .observation import UNVERSIONED_OBSERVATION_SCHEMA, ObservationPerspective, PokeZeroObservationV0


@dataclass(frozen=True)
class TrajectoryStep:
    player_id: str
    turn_index: int
    observation: PokeZeroObservationV0
    legal_action_mask: tuple[bool, ...]
    action_index: int
    reward: float = 0.0
    opponent_action_index: Optional[int] = None
    action_probability: Optional[float] = None
    value_estimate: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Dense potential-based shaping component for this decision (see pokezero.shaping),
    # recorded SEPARATELY from the raw env reward. None = unshaped collection (the
    # serialized key is omitted entirely, keeping shaping-off records byte-identical).
    shaping_reward: Optional[float] = None

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative.")
        if len(self.legal_action_mask) != ACTION_COUNT:
            raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
        if tuple(self.observation.legal_action_mask) != self.legal_action_mask:
            raise ValueError("legal_action_mask must match observation.legal_action_mask.")
        if self.action_index < 0 or self.action_index >= ACTION_COUNT:
            raise ValueError(f"action_index must be between 0 and {ACTION_COUNT - 1}.")
        if not self.legal_action_mask[self.action_index]:
            raise ValueError("action_index must be legal for the recorded observation.")
        if self.opponent_action_index is not None and not 0 <= self.opponent_action_index < ACTION_COUNT:
            raise ValueError(f"opponent_action_index must be between 0 and {ACTION_COUNT - 1}.")
        if self.action_probability is not None and not 0.0 <= self.action_probability <= 1.0:
            raise ValueError("action_probability must be between 0 and 1 when set.")
        if self.value_estimate is not None and not math.isfinite(float(self.value_estimate)):
            raise ValueError("value_estimate must be finite when set.")
        if self.shaping_reward is not None and not math.isfinite(float(self.shaping_reward)):
            raise ValueError("shaping_reward must be finite when set.")


@dataclass
class BattleTrajectory:
    battle_id: str
    format_id: str
    seed: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    terminal: Optional[TerminalState] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def append(self, step: TrajectoryStep) -> None:
        if self.terminal is not None:
            raise ValueError("cannot append trajectory steps after terminal state is recorded.")
        self.steps.append(step)

    def record_terminal(self, terminal: TerminalState) -> None:
        self.terminal = terminal

    @property
    def capped(self) -> bool:
        return bool(self.terminal and self.terminal.capped)

    def players(self) -> tuple[str, ...]:
        seen: set[str] = set()
        players: list[str] = []
        for step in self.steps:
            if step.player_id in seen:
                continue
            seen.add(step.player_id)
            players.append(step.player_id)
        return tuple(players)

    def total_reward(self, player_id: str) -> float:
        return sum(step.reward for step in self.steps if step.player_id == player_id)

    def steps_for_player(self, player_id: str) -> tuple[TrajectoryStep, ...]:
        return tuple(step for step in self.steps if step.player_id == player_id)

    def steps_for_turn(self, turn_index: int) -> tuple[TrajectoryStep, ...]:
        return tuple(step for step in self.steps if step.turn_index == turn_index)


def trajectory_to_dict(trajectory: BattleTrajectory) -> dict[str, Any]:
    return {
        "battle_id": trajectory.battle_id,
        "format_id": trajectory.format_id,
        "seed": trajectory.seed,
        "metadata": dict(trajectory.metadata),
        "terminal": _terminal_to_dict(trajectory.terminal),
        "steps": [_step_to_dict(step) for step in trajectory.steps],
    }


def trajectory_from_dict(payload: Mapping[str, Any]) -> BattleTrajectory:
    terminal_payload = payload.get("terminal")
    trajectory = BattleTrajectory(
        battle_id=str(payload["battle_id"]),
        format_id=str(payload["format_id"]),
        seed=int(payload["seed"]),
        metadata=_mapping(payload.get("metadata", {})),
    )
    for step_payload in _sequence(payload.get("steps")):
        trajectory.append(_step_from_dict(_mapping(step_payload)))
    if terminal_payload is not None:
        trajectory.record_terminal(_terminal_from_dict(_mapping(terminal_payload)))
    return trajectory


def _step_to_dict(step: TrajectoryStep) -> dict[str, Any]:
    return {
        "player_id": step.player_id,
        "turn_index": step.turn_index,
        "observation": _observation_to_dict(step.observation),
        "legal_action_mask": list(step.legal_action_mask),
        "action_index": step.action_index,
        "reward": step.reward,
        "opponent_action_index": step.opponent_action_index,
        "action_probability": step.action_probability,
        "value_estimate": step.value_estimate,
        "metadata": dict(step.metadata),
        # Optional shaping component: key omitted when absent so unshaped records stay
        # byte-identical to pre-shaping collection and old readers never see the field.
        **({"shaping_reward": step.shaping_reward} if step.shaping_reward is not None else {}),
    }


def _step_from_dict(payload: Mapping[str, Any]) -> TrajectoryStep:
    return TrajectoryStep(
        player_id=str(payload["player_id"]),
        turn_index=int(payload["turn_index"]),
        observation=_observation_from_dict(_mapping(payload["observation"])),
        legal_action_mask=tuple(bool(value) for value in _sequence(payload["legal_action_mask"])),
        action_index=int(payload["action_index"]),
        reward=float(payload.get("reward", 0.0)),
        opponent_action_index=_optional_int(payload.get("opponent_action_index")),
        action_probability=_optional_float(payload.get("action_probability")),
        value_estimate=_optional_float(payload.get("value_estimate")),
        metadata=_mapping(payload.get("metadata", {})),
        shaping_reward=_optional_float(payload.get("shaping_reward")),
    )


def _observation_to_dict(observation: PokeZeroObservationV0) -> dict[str, Any]:
    return {
        "schema_version": observation.schema_version,
        "categorical_ids": observation.categorical_ids,
        "numeric_features": observation.numeric_features,
        "token_type_ids": observation.token_type_ids,
        "attention_mask": observation.attention_mask,
        "legal_action_mask": observation.legal_action_mask,
        "perspective": _perspective_to_dict(observation.perspective),
        "metadata": dict(observation.metadata),
    }


def _observation_from_dict(payload: Mapping[str, Any]) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(int(value) for value in row) for row in _sequence(payload["categorical_ids"])),
        numeric_features=tuple(tuple(float(value) for value in row) for row in _sequence(payload["numeric_features"])),
        token_type_ids=tuple(int(value) for value in _sequence(payload["token_type_ids"])),
        attention_mask=tuple(bool(value) for value in _sequence(payload["attention_mask"])),
        legal_action_mask=tuple(bool(value) for value in _sequence(payload["legal_action_mask"])),
        perspective=_perspective_from_dict(payload.get("perspective")),
        metadata=_mapping(payload.get("metadata", {})),
        # One-way-door posture: a payload with NO schema version is an unknown/legacy artifact
        # and must be refused downstream — never silently coerced to the current spec.
        schema_version=str(payload.get("schema_version") or UNVERSIONED_OBSERVATION_SCHEMA),
    )


def _terminal_to_dict(terminal: Optional[TerminalState]) -> dict[str, Any] | None:
    if terminal is None:
        return None
    return {
        "winner": terminal.winner,
        "turn_count": terminal.turn_count,
        "capped": terminal.capped,
    }


def _terminal_from_dict(payload: Mapping[str, Any]) -> TerminalState:
    winner = payload.get("winner")
    return TerminalState(
        winner=str(winner) if winner is not None else None,
        turn_count=int(payload["turn_count"]),
        capped=bool(payload.get("capped", False)),
    )


def _perspective_to_dict(perspective: ObservationPerspective | None) -> dict[str, Any] | None:
    if perspective is None:
        return None
    return {
        "player_id": perspective.player_id,
        "showdown_slot": perspective.showdown_slot,
        "opponent_showdown_slot": perspective.opponent_showdown_slot,
    }


def _perspective_from_dict(payload: Any) -> ObservationPerspective | None:
    if payload is None:
        return None
    data = _mapping(payload)
    return ObservationPerspective(
        player_id=str(data["player_id"]),
        showdown_slot=str(data["showdown_slot"]),
        opponent_showdown_slot=str(data["opponent_showdown_slot"]),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object payload.")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("expected JSON array payload.")
    return tuple(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
