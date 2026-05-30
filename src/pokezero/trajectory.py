"""Trajectory containers for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .actions import ACTION_COUNT
from .env import TerminalState
from .observation import PokeZeroObservationV0


@dataclass(frozen=True)
class TrajectoryStep:
    player_id: str
    observation: PokeZeroObservationV0
    legal_action_mask: tuple[bool, ...]
    action_index: int
    reward: float = 0.0
    opponent_action_index: Optional[int] = None
    action_probability: Optional[float] = None
    value_estimate: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.legal_action_mask) != ACTION_COUNT:
            raise ValueError(f"legal_action_mask must contain {ACTION_COUNT} values.")
        if self.action_index < 0 or self.action_index >= ACTION_COUNT:
            raise ValueError(f"action_index must be between 0 and {ACTION_COUNT - 1}.")
        if not self.legal_action_mask[self.action_index]:
            raise ValueError("action_index must be legal for the recorded observation.")
        if self.opponent_action_index is not None and not 0 <= self.opponent_action_index < ACTION_COUNT:
            raise ValueError(f"opponent_action_index must be between 0 and {ACTION_COUNT - 1}.")
        if self.action_probability is not None and not 0.0 <= self.action_probability <= 1.0:
            raise ValueError("action_probability must be between 0 and 1 when set.")


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
