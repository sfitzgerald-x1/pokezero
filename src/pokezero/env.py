"""Environment protocol for PokeZero rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, runtime_checkable

from .observation import PokeZeroObservationV0

PlayerId = str
BattleFormat = str


@dataclass(frozen=True)
class TerminalState:
    winner: Optional[PlayerId]
    turn_count: int
    capped: bool = False


@dataclass(frozen=True)
class StepResult:
    observations: Mapping[PlayerId, PokeZeroObservationV0]
    rewards: Mapping[PlayerId, float]
    terminal: Optional[TerminalState]


@runtime_checkable
class PokeZeroEnv(Protocol):
    def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        ...

    def observe(self, player: PlayerId) -> PokeZeroObservationV0:
        ...

    def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        ...

    def step(self, actions: Mapping[PlayerId, int]) -> StepResult:
        ...

    def terminal(self) -> Optional[TerminalState]:
        ...
