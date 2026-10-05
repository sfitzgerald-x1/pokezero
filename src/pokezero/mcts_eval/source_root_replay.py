"""Fail-closed source-only repair for captured public replay prefixes.

Some root-audit captures use ``unresolved-public-event`` when the capture
adapter cannot derive a public action for the player whose record it is
building.  A sampled-world replay must never treat that placeholder as an
arbitrary legal action: doing so can manufacture a different root.  This
module permits source-owned earlier canonical actions, or a publicly witnessed
faint-before-action cancellation. The latter does not recover the opponent's
selected move, and must be reproduced by the sampled-world replay. Every
substitution is recorded in a serializable ledger.

The returned action rounds remain replay input only.  The repair ledger and
the source record's action candidate are not policy context or trajectory
metadata for MCTS.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..public_decision_corpus import (
    PublicActionIdentifier,
    PublicDecisionRecord,
    PublicResolvedActionRound,
)


_UNRESOLVED_EVENT_ID = "unresolved-public-event"


class SourceRootReplayError(ValueError):
    """A source-bound prefix cannot be repaired without crossing its boundary."""


@dataclass(frozen=True)
class SourceRootReplayRepair:
    """One placeholder replaced by a source action or witnessed cancellation."""

    turn_index: int
    player_id: str
    original_event_id: str
    source_decision_id: str
    source_action: PublicActionIdentifier
    public_cancellation_witness: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "player_id": self.player_id,
            "original_event_id": self.original_event_id,
            "source_decision_id": self.source_decision_id,
            "source_action": self.source_action.to_dict(),
            **({"public_cancellation_witness": dict(self.public_cancellation_witness)}
               if self.public_cancellation_witness is not None else {}),
        }


@dataclass(frozen=True)
class SourceRootReplayPrefix:
    """Repaired replay-only rounds and their complete source repair ledger."""

    public_action_rounds: tuple[PublicResolvedActionRound, ...]
    repairs: tuple[SourceRootReplayRepair, ...]


def source_bound_replay_prefix(
    record: PublicDecisionRecord,
    *,
    source_records: Sequence[PublicDecisionRecord],
) -> SourceRootReplayPrefix:
    """Repair source-owned actions or a witnessed public cancellation.

    ``source_records`` may contain records from a larger capture, but every
    record actually used must bind exactly to the target seed, battle, format,
    and source seat. An unresolved opposing action without a public
    cancellation witness, an ambiguous prior record, and any noncanonical
    source action are hard failures.
    """

    by_turn: dict[int, PublicDecisionRecord] = {}
    for candidate in source_records:
        if candidate.turn_index >= record.turn_index:
            continue
        if not _same_source(candidate, record):
            continue
        if candidate.turn_index in by_turn:
            raise SourceRootReplayError("ambiguous_source_record_for_turn")
        by_turn[candidate.turn_index] = candidate

    repaired_rounds: list[PublicResolvedActionRound] = []
    repairs: list[SourceRootReplayRepair] = []
    target_history = {entry.turn_index: entry.observation for entry in record.history}
    for action_round in record.public_resolved_action_rounds:
        actions = dict(action_round.actions)
        for player_id, identifier in action_round.actions.items():
            if identifier.kind != "event" or identifier.event_id != _UNRESOLVED_EVENT_ID:
                continue
            if player_id != record.acting_player:
                witness = _public_cancellation_witness(record, action_round.turn_index, player_id)
                if witness is None:
                    raise SourceRootReplayError("unresolved_public_event_for_non_source_player")
                source_action = PublicActionIdentifier(kind="event", event_id="faint-before-action")
                actions[player_id] = source_action
                repairs.append(SourceRootReplayRepair(action_round.turn_index, player_id,
                    identifier.event_id, record.decision_id, source_action, witness))
                continue
            source_record = by_turn.get(action_round.turn_index)
            if source_record is None:
                raise SourceRootReplayError("missing_source_record_for_unresolved_public_event")
            _validate_source_witness(source_record, record, target_history=target_history)
            source_action = _source_recorded_action(source_record)
            actions[player_id] = source_action
            repairs.append(
                SourceRootReplayRepair(
                    turn_index=action_round.turn_index,
                    player_id=player_id,
                    original_event_id=_UNRESOLVED_EVENT_ID,
                    source_decision_id=source_record.decision_id,
                    source_action=source_action,
                )
            )
        repaired_rounds.append(replace(action_round, actions=actions))
    return SourceRootReplayPrefix(public_action_rounds=tuple(repaired_rounds), repairs=tuple(repairs))


def _public_cancellation_witness(record: PublicDecisionRecord, turn: int, player: str) -> dict[str, Any] | None:
    """Use a retained actor-public event-window delta, never opposing requests.

    A later observed forced replacement can share the rolling window. Truncate
    at the faint so that replacement is not mistaken for the cancelled action.
    Require an overlapping exact prefix and the declared other actor's move.
    Live replay subsequently certifies that this cancellation actually occurs.
    """
    from ..public_action_capture import public_action_identifiers_from_protocol_lines
    views = {entry.turn_index: entry.observation.acting_player_state for entry in record.history}
    views[record.turn_index] = record.observation.acting_player_state
    before = views.get(turn)
    following = sorted(index for index in views if index > turn)
    if before is None or not following:
        return None
    after = views[following[0]]
    left, right = before.get("recent_public_events"), after.get("recent_public_events")
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or not left:
        return None
    overlap = next((size for size in range(min(len(left), len(right)), 0, -1)
                    if list(left[-size:]) == list(right[:size])), 0)
    if not overlap:
        return None
    # A separator-only overlap cannot identify a round. Require the latest
    # public turn marker to be retained in the exact overlapping window.
    markers = [line for line in left if isinstance(line, str) and line.startswith("|turn|")]
    if not markers or markers[-1] not in left[-overlap:]:
        return None
    opponent = "p2" if record.acting_player == "p1" else "p1"
    def canonical(line: str) -> str:
        return line.replace("|selfa:", f"|{record.acting_player}a:").replace(
            "|opponenta:", f"|{opponent}a:")
    delta = [canonical(str(line)) for line in right[overlap:]]
    active = before.get("opponent_active", {})
    species = active.get("species") if isinstance(active, Mapping) else None
    if not isinstance(species, str) or not species:
        return None
    faint = f"|faint|{player}a: {species}"
    if faint not in delta:
        return None
    lines = delta[:delta.index(faint) + 1]
    if any(line.startswith("|turn|") for line in lines):
        return None
    identifiers = public_action_identifiers_from_protocol_lines(lines, cancellation_players=(player,))
    event = identifiers.get(player)
    expected = record.public_resolved_action_rounds[turn].actions.get(record.acting_player)
    if (event is None or event.event_id != "faint-before-action"
            or expected is None or expected.kind != "move"
            or identifiers.get(record.acting_player) != expected):
        return None
    return {"source_decision_id": record.decision_id, "before_turn_index": turn,
            "after_turn_index": following[0], "public_lines": lines,
            "historical_selected_action_recovered": False}


def _same_source(candidate: PublicDecisionRecord, record: PublicDecisionRecord) -> bool:
    return (
        candidate.seed == record.seed
        and candidate.battle_id == record.battle_id
        and candidate.format_id == record.format_id
        and candidate.acting_player == record.acting_player
    )


def _validate_source_witness(
    source_record: PublicDecisionRecord,
    target_record: PublicDecisionRecord,
    *,
    target_history: Mapping[int, Any],
) -> None:
    """Require an exact retained target prefix, not merely reused labels."""

    if target_history.get(source_record.turn_index) != source_record.observation:
        raise SourceRootReplayError("source_record_observation_does_not_match_target_history")
    for entry in source_record.history:
        if target_history.get(entry.turn_index) != entry.observation:
            raise SourceRootReplayError("source_record_history_does_not_match_target_history")
    if source_record.public_resolved_action_rounds != target_record.public_resolved_action_rounds[: source_record.turn_index]:
        raise SourceRootReplayError("source_record_public_prefix_does_not_match_target_prefix")


def _source_recorded_action(record: PublicDecisionRecord) -> PublicActionIdentifier:
    candidates = record.observation.acting_player_state.get("action_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise SourceRootReplayError("source_record_missing_action_candidates")
    selected = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("action_index") == record.recorded_action_index
    ]
    if len(selected) != 1:
        raise SourceRootReplayError("source_record_action_candidate_not_unique")
    candidate: Mapping[str, Any] = selected[0]
    if candidate.get("legal") is False:
        raise SourceRootReplayError("source_recorded_action_not_legal")
    if candidate.get("kind") == "move":
        move_id = candidate.get("move_id")
        if isinstance(move_id, str) and move_id:
            return PublicActionIdentifier(kind="move", move_id=move_id)
    if candidate.get("kind") == "switch":
        species = candidate.get("switched_species")
        if not isinstance(species, str) or not species:
            pokemon = candidate.get("pokemon")
            species = pokemon.get("species") if isinstance(pokemon, Mapping) else None
        if isinstance(species, str) and species:
            return PublicActionIdentifier(kind="switch", switched_species=species)
    raise SourceRootReplayError("source_recorded_action_not_canonical_move_or_switch")
