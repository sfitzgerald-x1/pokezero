"""Environment protocols for PokeZero rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, runtime_checkable

from .observation import PokeZeroObservationV0

PlayerId = str
BattleFormat = str
DEFAULT_BATTLE_START_OVERRIDE_FORMAT: BattleFormat = "gen3customgame"


@dataclass(frozen=True)
class BattleStartOverride:
    """Optional explicit start-state materialization for replay/search branches."""

    player_teams: Mapping[PlayerId, str]
    format_id: BattleFormat = DEFAULT_BATTLE_START_OVERRIDE_FORMAT

    def __post_init__(self) -> None:
        format_id = str(self.format_id)
        if format_id != DEFAULT_BATTLE_START_OVERRIDE_FORMAT:
            raise ValueError(
                "BattleStartOverride currently requires "
                f"{DEFAULT_BATTLE_START_OVERRIDE_FORMAT!r} so packed teams are honored."
            )
        normalized: dict[PlayerId, str] = {}
        for player, team in self.player_teams.items():
            player_id = str(player)
            if player_id not in {"p1", "p2"}:
                raise ValueError("BattleStartOverride player_teams keys must be p1 or p2.")
            team_text = str(team)
            if not team_text:
                raise ValueError("BattleStartOverride player team strings must be non-empty.")
            normalized[player_id] = team_text
        missing = sorted({"p1", "p2"} - set(normalized))
        if missing:
            raise ValueError(
                "BattleStartOverride must provide complete p1 and p2 packed teams; "
                f"missing: {', '.join(missing)}."
            )
        object.__setattr__(self, "player_teams", normalized)
        object.__setattr__(self, "format_id", format_id)


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
    requested_players: tuple[PlayerId, ...] = ()


@runtime_checkable
class PokeZeroEnv(Protocol):
    def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        ...

    def observe(self, player: PlayerId) -> PokeZeroObservationV0:
        ...

    def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        ...

    def requested_players(self) -> tuple[PlayerId, ...]:
        ...

    def step(self, actions: Mapping[PlayerId, int]) -> StepResult:
        """Submit actions for the currently requested players.

        Standard turns request both players. Forced-switch and other asymmetric
        sub-requests may request exactly one player.
        """
        ...

    def terminal(self) -> Optional[TerminalState]:
        ...


@runtime_checkable
class AsyncPokeZeroEnv(Protocol):
    async def reset(self, *, seed: int, format_id: BattleFormat = "gen3randombattle") -> None:
        ...

    async def observe(self, player: PlayerId) -> PokeZeroObservationV0:
        ...

    async def legal_actions(self, player: PlayerId) -> tuple[bool, ...]:
        ...

    async def requested_players(self) -> tuple[PlayerId, ...]:
        ...

    async def step(self, actions: Mapping[PlayerId, int]) -> StepResult:
        ...

    async def terminal(self) -> Optional[TerminalState]:
        ...
