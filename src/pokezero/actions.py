"""Fixed Gen 3 singles action-candidate helpers.

The 9 policy logits score the current action candidates, not persistent team
identities. Move slots follow the current Showdown request. Switch slots are a
dense decode convention over the non-active team members in team order; the
corresponding action-candidate token carries the actual switch target identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ACTION_COUNT = 9
ACTION_SCHEMA_VERSION = "pokezero.action_space.v0"
MOVE_ACTION_COUNT = 4
SWITCH_ACTION_COUNT = 5
TEAM_SIZE = 6

ActionKind = Literal["move", "switch"]


@dataclass(frozen=True)
class ActionCandidate:
    action_index: int
    kind: ActionKind
    legal: bool
    move_slot: Optional[int] = None
    team_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "move":
            if self.move_slot is None or not is_move_action(self.action_index):
                raise ValueError("Move candidates must use action slots 0..3 and set move_slot.")
            if self.team_index is not None:
                raise ValueError("Move candidates must not set team_index.")
            return
        if self.kind == "switch":
            if self.team_index is None or not is_switch_action(self.action_index):
                raise ValueError("Switch candidates must use action slots 4..8 and set team_index.")
            if self.move_slot is not None:
                raise ValueError("Switch candidates must not set move_slot.")
            return
        raise ValueError(f"Unsupported action kind: {self.kind!r}.")


def is_move_action(action_index: int) -> bool:
    return 0 <= action_index < MOVE_ACTION_COUNT


def is_switch_action(action_index: int) -> bool:
    return MOVE_ACTION_COUNT <= action_index < ACTION_COUNT


def canonical_switch_action_map(active_team_index: int, *, team_size: int = TEAM_SIZE) -> tuple[int, ...]:
    """Return dense switch-candidate target team indices in stable team order."""
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


def move_action_candidates(legal_mask: tuple[bool, ...]) -> tuple[ActionCandidate, ...]:
    if len(legal_mask) != ACTION_COUNT:
        raise ValueError(f"legal_mask must contain {ACTION_COUNT} values.")
    return tuple(
        ActionCandidate(
            action_index=action_index,
            kind="move",
            move_slot=action_index,
            legal=legal_mask[action_index],
        )
        for action_index in range(MOVE_ACTION_COUNT)
    )


def switch_action_candidates(
    active_team_index: int,
    legal_mask: tuple[bool, ...],
    *,
    team_size: int = TEAM_SIZE,
) -> tuple[ActionCandidate, ...]:
    if len(legal_mask) != ACTION_COUNT:
        raise ValueError(f"legal_mask must contain {ACTION_COUNT} values.")
    return tuple(
        ActionCandidate(
            action_index=MOVE_ACTION_COUNT + switch_slot,
            kind="switch",
            team_index=team_index,
            legal=legal_mask[MOVE_ACTION_COUNT + switch_slot],
        )
        for switch_slot, team_index in enumerate(canonical_switch_action_map(active_team_index, team_size=team_size))
    )
