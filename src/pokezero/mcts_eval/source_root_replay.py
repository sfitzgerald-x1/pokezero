"""Fail-closed source-only repair for captured public replay prefixes.

Some root-audit captures use ``unresolved-public-event`` when the capture
adapter cannot derive a public action for the player whose record it is
building.  A sampled-world replay must never treat that placeholder as an
arbitrary legal action: doing so can manufacture a different root.  This
module permits one deliberately narrow repair before replay.  It substitutes
only the source actor's *own* earlier, recorded canonical move or switch and
records every substitution in a serializable ledger.

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
    """One source-owned placeholder replaced by its prior canonical action."""

    turn_index: int
    player_id: str
    original_event_id: str
    source_action: PublicActionIdentifier

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "player_id": self.player_id,
            "original_event_id": self.original_event_id,
            "source_action": self.source_action.to_dict(),
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
    """Repair only ``record.acting_player``'s own unresolved prior actions.

    ``source_records`` may contain records from a larger capture, but every
    record actually used must bind exactly to the target seed, battle, format,
    and source seat.  An unresolved action for the other player, an ambiguous
    prior record, and any noncanonical source action are hard failures.
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
    for action_round in record.public_resolved_action_rounds:
        actions = dict(action_round.actions)
        for player_id, identifier in action_round.actions.items():
            if identifier.kind != "event" or identifier.event_id != _UNRESOLVED_EVENT_ID:
                continue
            if player_id != record.acting_player:
                raise SourceRootReplayError("unresolved_public_event_for_non_source_player")
            source_record = by_turn.get(action_round.turn_index)
            if source_record is None:
                raise SourceRootReplayError("missing_source_record_for_unresolved_public_event")
            source_action = _source_recorded_action(source_record)
            actions[player_id] = source_action
            repairs.append(
                SourceRootReplayRepair(
                    turn_index=action_round.turn_index,
                    player_id=player_id,
                    original_event_id=_UNRESOLVED_EVENT_ID,
                    source_action=source_action,
                )
            )
        repaired_rounds.append(replace(action_round, actions=actions))
    return SourceRootReplayPrefix(public_action_rounds=tuple(repaired_rounds), repairs=tuple(repairs))


def _same_source(candidate: PublicDecisionRecord, record: PublicDecisionRecord) -> bool:
    return (
        candidate.seed == record.seed
        and candidate.battle_id == record.battle_id
        and candidate.format_id == record.format_id
        and candidate.acting_player == record.acting_player
    )


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
