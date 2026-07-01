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

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .actions import ACTION_COUNT, MOVE_ACTION_COUNT, switch_action_index_for_team_index


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _active_team_index(self_team) -> int | None:
    for index, mon in enumerate(self_team):
        if getattr(mon, "active", False):
            return index
    return None


def _request_move_candidates(state) -> list[frozenset[str]]:
    """Per active-move slot (index == move action), the normalized names it can be chosen by:
    both the request ``id`` and the display ``move``. Hidden Power is the reason both are needed —
    foul-play submits ``move hiddenpowerflying70`` (the typed display) while the request ``id`` is
    just ``hiddenpower``; matching only the id would drop every Hidden Power decision."""
    request = getattr(state, "request", None)
    if not isinstance(request, Mapping):
        return []
    active = request.get("active")
    first = active[0] if isinstance(active, (list, tuple)) and active else None
    moves = first.get("moves") if isinstance(first, Mapping) else None
    if not isinstance(moves, (list, tuple)):
        return []
    candidates = []
    for move in moves:
        if isinstance(move, Mapping):
            candidates.append(frozenset(c for c in (_norm(move.get("id")), _norm(move.get("move"))) if c))
        else:
            candidates.append(frozenset())
    return candidates


def _move_index_by_name(state, name: str) -> int | None:
    target = _norm(name)
    if not target:
        return None
    mask = state.legal_action_mask
    for slot, names in enumerate(_request_move_candidates(state)):
        if target in names and slot < MOVE_ACTION_COUNT and slot < len(mask) and mask[slot]:
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


@dataclass
class CaptureDecision:
    room: str
    protocol_lines: tuple[str, ...]  # room protocol accumulated up to this decision
    choice: str  # the /choose body, e.g. "move icebeam" / "switch 3"


@dataclass
class CaptureGame:
    room: str
    decisions: list[CaptureDecision] = field(default_factory=list)
    winner: str | None = None
    final_lines: tuple[str, ...] = ()


def parse_capture_transcript(path: str) -> list[CaptureGame]:
    """Parse a foul-play capture transcript (JSONL of {"t":"recv"|"send","msg":...}) into games.

    Received messages are Showdown protocol blocks led by a ``>room`` header; we accumulate each
    room's ``|`` lines. Each outgoing ``room|/choose <body>|<rqid>`` becomes a decision snapshotted
    against the room's protocol so far. The room's ``|win|<name>`` line gives the terminal winner.
    """
    games: dict[str, CaptureGame] = {}
    rooms: dict[str, list[str]] = {}

    def game(room: str) -> CaptureGame:
        return games.setdefault(room, CaptureGame(room=room))

    for raw in open(path):
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        message = row.get("msg", "")
        if row.get("t") == "recv":
            current = None
            for line in message.split("\n"):
                if line.startswith(">"):
                    current = line[1:].strip()
                    rooms.setdefault(current, [])
                elif current is not None and line.startswith("|"):
                    rooms[current].append(line)
                    if line.startswith("|win|"):
                        game(current).winner = line[len("|win|"):].strip() or None
        elif row.get("t") == "send" and "/choose" in message:
            parts = message.split("|")
            room = parts[0]
            body = next((p for p in parts if p.startswith("/choose")), "")
            choice = body[len("/choose "):].strip()
            if room in rooms and choice:
                game(room).decisions.append(
                    CaptureDecision(room=room, protocol_lines=tuple(rooms[room]), choice=choice)
                )

    for room, entry in games.items():
        entry.final_lines = tuple(rooms.get(room, ()))
    return list(games.values())
