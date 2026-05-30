"""Core interfaces for PokeZero self-play training."""

from .actions import (
    ACTION_COUNT,
    MOVE_ACTION_COUNT,
    SWITCH_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
    switch_action_index_for_team_index,
)
from .env import PokeZeroEnv, StepResult, TerminalState
from .observation import ObservationSpec, PokeZeroObservationV0

__all__ = [
    "ACTION_COUNT",
    "MOVE_ACTION_COUNT",
    "ObservationSpec",
    "PokeZeroEnv",
    "PokeZeroObservationV0",
    "SWITCH_ACTION_COUNT",
    "StepResult",
    "TerminalState",
    "canonical_switch_action_map",
    "is_move_action",
    "is_switch_action",
    "switch_action_index_for_team_index",
]
