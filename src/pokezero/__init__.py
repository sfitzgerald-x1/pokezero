"""Core interfaces for PokeZero self-play training."""

from .actions import (
    ACTION_COUNT,
    ActionCandidate,
    MOVE_ACTION_COUNT,
    SWITCH_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
    move_action_candidates,
    switch_action_index_for_team_index,
    switch_action_candidates,
)
from .env import AsyncPokeZeroEnv, PokeZeroEnv, StepResult, TerminalState
from .observation import ObservationSpec, PokeZeroObservationV0
from .showdown import (
    PlayerRelativeBattleState,
    ShowdownPokemon,
    ShowdownReplayState,
    ShowdownSubmission,
    detect_showdown_slot,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
    showdown_submission_for_action,
)

__all__ = [
    "ACTION_COUNT",
    "ActionCandidate",
    "AsyncPokeZeroEnv",
    "MOVE_ACTION_COUNT",
    "ObservationSpec",
    "PokeZeroEnv",
    "PokeZeroObservationV0",
    "SWITCH_ACTION_COUNT",
    "PlayerRelativeBattleState",
    "ShowdownPokemon",
    "ShowdownReplayState",
    "ShowdownSubmission",
    "StepResult",
    "TerminalState",
    "canonical_switch_action_map",
    "detect_showdown_slot",
    "is_move_action",
    "is_switch_action",
    "move_action_candidates",
    "normalize_for_player",
    "observation_from_player_state",
    "parse_showdown_replay",
    "showdown_choice_for_action",
    "showdown_submission_for_action",
    "switch_action_index_for_team_index",
    "switch_action_candidates",
]
