"""Fixed Gen 3 singles action-slot helpers."""

from __future__ import annotations

ACTION_COUNT = 9
MOVE_ACTION_COUNT = 4
SWITCH_ACTION_COUNT = 5
TEAM_SIZE = 6


def is_move_action(action_index: int) -> bool:
    return 0 <= action_index < MOVE_ACTION_COUNT


def is_switch_action(action_index: int) -> bool:
    return MOVE_ACTION_COUNT <= action_index < ACTION_COUNT


def canonical_switch_action_map(active_team_index: int, *, team_size: int = TEAM_SIZE) -> tuple[int, ...]:
    """Return switch target team indices in stable team order."""
    if team_size < 2:
        raise ValueError("team_size must include at least two Pokemon.")
    if active_team_index < 0 or active_team_index >= team_size:
        raise ValueError("active_team_index must be inside the team.")
    return tuple(index for index in range(team_size) if index != active_team_index)


def switch_action_index_for_team_index(
    team_index: int,
    active_team_index: int,
    *,
    team_size: int = TEAM_SIZE,
) -> int:
    switch_targets = canonical_switch_action_map(active_team_index, team_size=team_size)
    try:
        switch_slot = switch_targets.index(team_index)
    except ValueError as exc:
        raise ValueError("team_index is not a valid switch target.") from exc
    return MOVE_ACTION_COUNT + switch_slot
