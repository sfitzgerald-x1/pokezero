"""Recover a teacher's action index from its Showdown ``/choose`` string, for behavior cloning.

We run ``foul-play`` as a separate process (keeping the GPL/MIT boundary intact — no foul-play
code is imported here) and capture, per decision, the battle protocol it received plus the choice
string it submitted (``move 3``, ``switch 2``, ``move surf``, ...). This module maps that choice
string back to our 0–8 action index, given the decision state (``PlayerRelativeBattleState``),
so the captured games can be written as RolloutRecords and trained with ``--objective
behavior-cloning``.

Design note: mapping the *choice string* is more robust than inferring the action from the public
``|move|``/``|switch|`` lines, because the choice is exactly what the player submitted (no
ambiguity from U-turn/Baton Pass follow-up switches, Pursuit, drags, or failed moves). We still
validate every decoded index against the request's legal-action mask and fall back to name
matching when the teacher submits a move/switch by name rather than slot.
"""

from __future__ import annotations

from typing import Any, Mapping

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT, switch_action_index_for_team_index


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _active_team_index(self_team) -> int | None:
    for index, mon in enumerate(self_team):
        if getattr(mon, "active", False):
            return index
    return None


def _request_move_ids(state) -> list[str]:
    """Normalized move ids/names in the active request's slot order (index == move action)."""
    request = getattr(state, "request", None)
    if not isinstance(request, Mapping):
        return []
    active = request.get("active")
    first = active[0] if isinstance(active, (list, tuple)) and active else None
    moves = first.get("moves") if isinstance(first, Mapping) else None
    if not isinstance(moves, (list, tuple)):
        return []
    return [_norm(m.get("id") or m.get("move")) if isinstance(m, Mapping) else "" for m in moves]


def _move_index_by_name(state, name: str) -> int | None:
    target = _norm(name)
    if not target:
        return None
    mask = state.legal_action_mask
    for slot, move_id in enumerate(_request_move_ids(state)):
        if move_id == target and slot < MOVE_ACTION_COUNT and slot < len(mask) and mask[slot]:
            return slot
    return None


def _switch_index_by_species(state, species: str) -> int | None:
    target = _norm(species)
    active = _active_team_index(state.self_team)
    if active is None or not target:
        return None
    mask = state.legal_action_mask
    for team_index, mon in enumerate(state.self_team):
        if team_index == active or _norm(getattr(mon, "species", "")) != target:
            continue
        try:
            action_index = switch_action_index_for_team_index(
                team_index, active, team_size=len(state.self_team)
            )
        except ValueError:
            return None
        if action_index < len(mask) and mask[action_index]:
            return action_index
    return None


def action_index_from_choice_string(state, choice: str) -> int | None:
    """Map a Showdown ``/choose`` body to our 0–8 action index, or None if undecodable/illegal.

    ``move N`` — N is 1-based into the request's active-move list, so the action index is ``N-1``.
    ``move <name>`` — matched against the request's move ids.
    ``switch N`` — N is 1-based into the team; mapped to the dense switch action index for the
    current active mon. ``switch <species>`` — matched against the bench.
    ``default``/``pass``/``team ...`` and anything illegal for the current request return None.
    """
    if not choice:
        return None
    tokens = choice.strip().lower().split()
    if tokens and tokens[0].startswith("/"):  # tolerate a leading "/choose"
        tokens = tokens[1:]
    if len(tokens) < 2:
        return None
    kind, arg = tokens[0], tokens[1]
    mask = state.legal_action_mask

    if kind == "move":
        if arg.isdigit():
            slot = int(arg) - 1
            if 0 <= slot < MOVE_ACTION_COUNT and slot < len(mask) and mask[slot]:
                return slot
            return None
        return _move_index_by_name(state, " ".join(tokens[1:]))

    if kind == "switch":
        if arg.isdigit():
            team_index = int(arg) - 1
            active = _active_team_index(state.self_team)
            if active is None or not (0 <= team_index < len(state.self_team)):
                return None
            try:
                action_index = switch_action_index_for_team_index(
                    team_index, active, team_size=len(state.self_team)
                )
            except ValueError:
                return None
            if action_index < len(mask) and mask[action_index]:
                return action_index
            return None
        return _switch_index_by_species(state, arg)

    return None  # default / pass / team-preview: not a gen3 mid-battle action we clone
