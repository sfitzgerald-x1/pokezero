"""Public protocol action capture shared by live rollouts and replay consumers."""

from __future__ import annotations

from typing import Mapping, Sequence

from .public_decision_corpus import PublicActionIdentifier, PublicResolvedActionRound
from .trajectory import BattleTrajectory


_UNRESOLVED_PUBLIC_EVENT_ID = "unresolved-public-event"


def public_action_round_from_protocol_lines(
    lines: Sequence[str],
    *,
    turn_index: int,
    requested_players: Sequence[str],
) -> PublicResolvedActionRound:
    """Capture one completed decision round from public protocol lines only.

    The result deliberately stores move and switched-species identifiers rather
    than request-local action indexes. A sampled belief world resolves those
    identifiers against its own legal request during replay.
    """

    actions = public_action_identifiers_from_protocol_lines(lines)
    for player_id in requested_players:
        actions.setdefault(
            str(player_id),
            PublicActionIdentifier(kind="event", event_id=_UNRESOLVED_PUBLIC_EVENT_ID),
        )
    if not actions:
        actions = {"p1": PublicActionIdentifier(kind="event", event_id=_UNRESOLVED_PUBLIC_EVENT_ID)}
    return PublicResolvedActionRound(turn_index=turn_index, actions=actions)


def public_action_identifiers_from_protocol_lines(
    lines: Sequence[str],
) -> dict[str, PublicActionIdentifier]:
    """Project public move, switch, and no-effect events to stable identifiers."""

    actions: dict[str, PublicActionIdentifier] = {}
    for line in lines:
        parts = str(line).split("|")
        if len(parts) < 3:
            continue
        event_type = parts[1]
        player_id = _protocol_player_id(parts[2])
        if player_id is None:
            continue
        if event_type == "move" and len(parts) >= 4:
            if _called_move_line(parts):
                continue
            existing = actions.get(player_id)
            if existing is not None and existing.kind != "event":
                continue
            move_id = _protocol_identifier(parts[3])
            if move_id:
                actions[player_id] = PublicActionIdentifier(kind="move", move_id=move_id)
        elif event_type == "switch" and len(parts) >= 4:
            existing = actions.get(player_id)
            if existing is not None and existing.kind != "event":
                continue
            species = _protocol_identifier(parts[3].split(",", 1)[0])
            if species:
                actions[player_id] = PublicActionIdentifier(kind="switch", switched_species=species)
        elif event_type == "cant" and len(parts) >= 4:
            if player_id in actions:
                continue
            reason = _protocol_identifier(parts[3])
            actions[player_id] = PublicActionIdentifier(
                kind="event",
                event_id=f"cant:{reason or 'unknown'}",
            )
    return actions


def append_public_action_round(
    trajectory: BattleTrajectory,
    action_round: PublicResolvedActionRound,
) -> None:
    """Persist a completed public round before the next policy decision."""

    payload = trajectory.metadata.get("public_resolved_action_rounds")
    existing = list(payload) if isinstance(payload, list) else []
    if any(_matches_turn_index(row, action_round.turn_index) for row in existing):
        raise ValueError(f"public action round {action_round.turn_index} was captured more than once.")
    existing.append(action_round.to_dict())
    trajectory.metadata = {
        **dict(trajectory.metadata),
        "public_resolved_action_rounds": existing,
    }


def public_action_rounds_from_trajectory_metadata(
    trajectory: BattleTrajectory,
) -> Mapping[int, PublicResolvedActionRound]:
    """Decode persisted public action IDs without inspecting request-local data."""

    payload = trajectory.metadata.get("public_resolved_action_rounds")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return {}
    rounds: dict[int, PublicResolvedActionRound] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            return {}
        action_round = PublicResolvedActionRound.from_dict(row)
        if action_round.turn_index in rounds:
            return {}
        rounds[action_round.turn_index] = action_round
    return rounds


def _protocol_player_id(value: str) -> str | None:
    prefix = str(value).strip().split(":", 1)[0]
    if prefix.startswith("p1"):
        return "p1"
    if prefix.startswith("p2"):
        return "p2"
    return None


def _protocol_identifier(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _called_move_line(parts: Sequence[str]) -> bool:
    for token in parts[4:]:
        text = str(token).strip()
        if not text.startswith("[from]"):
            continue
        if "lockedmove" not in _protocol_identifier(text):
            return True
    return False


def _matches_turn_index(payload: object, turn_index: int) -> bool:
    if not isinstance(payload, Mapping):
        return False
    try:
        return int(payload.get("turn_index", -1)) == turn_index
    except (TypeError, ValueError):
        return False
