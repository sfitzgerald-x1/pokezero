"""Fail-closed source-only repair for captured public replay prefixes.

Some root-audit captures use ``unresolved-public-event`` when the capture
adapter cannot derive a public action for the player whose record it is
building.  A sampled-world replay must never treat that placeholder as an
arbitrary legal action: doing so can manufacture a different root.  This
module permits one deliberately narrow repair before replay.  It substitutes
only a same-battle, same-player, same-turn recorded canonical move or switch,
and only after every already-resolved public action in the two prefixes agrees.
Every substitution is recorded in a serializable ledger.

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
    """One source-bound placeholder replaced by its prior canonical action."""

    turn_index: int
    player_id: str
    original_event_id: str
    source_decision_id: str
    source_action: PublicActionIdentifier

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "player_id": self.player_id,
            "original_event_id": self.original_event_id,
            "source_decision_id": self.source_decision_id,
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
    """Repair a retained public placeholder only from its recorded actor.

    ``source_records`` may contain records from a larger capture, but every
    record actually used must bind exactly to the target seed, battle, format,
    placeholder player, and placeholder turn.  The two public prefixes must
    agree wherever both records resolve an action.  An ambiguous prior record,
    incompatible prefix, and any noncanonical source action are hard failures.
    """

    by_player_turn: dict[tuple[str, int], PublicDecisionRecord] = {}
    for candidate in source_records:
        if candidate.turn_index >= record.turn_index:
            continue
        if not _same_battle(candidate, record):
            continue
        key = (candidate.acting_player, candidate.turn_index)
        if key in by_player_turn:
            raise SourceRootReplayError("ambiguous_source_record_for_turn")
        by_player_turn[key] = candidate

    repaired_rounds: list[PublicResolvedActionRound] = []
    repairs: list[SourceRootReplayRepair] = []
    target_history = {entry.turn_index: entry.observation for entry in record.history}
    for action_round in record.public_resolved_action_rounds:
        actions = dict(action_round.actions)
        for player_id, identifier in action_round.actions.items():
            if identifier.kind != "event" or identifier.event_id != _UNRESOLVED_EVENT_ID:
                continue
            source_record = by_player_turn.get((player_id, action_round.turn_index))
            if source_record is None:
                raise SourceRootReplayError("missing_source_record_for_unresolved_public_event")
            _validate_source_witness(
                source_record,
                record,
                target_history=target_history,
                placeholder_player=player_id,
                placeholder_turn=action_round.turn_index,
            )
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


def _same_battle(candidate: PublicDecisionRecord, record: PublicDecisionRecord) -> bool:
    return (
        candidate.seed == record.seed
        and candidate.battle_id == record.battle_id
        and candidate.format_id == record.format_id
    )


def _validate_source_witness(
    source_record: PublicDecisionRecord,
    target_record: PublicDecisionRecord,
    *,
    target_history: Mapping[int, Any],
    placeholder_player: str,
    placeholder_turn: int,
) -> None:
    """Require an exact source actor and a compatible retained public prefix."""

    if source_record.acting_player != placeholder_player:
        raise SourceRootReplayError("source_record_actor_does_not_match_unresolved_public_event")
    if source_record.turn_index != placeholder_turn:
        raise SourceRootReplayError("source_record_turn_does_not_match_unresolved_public_event")
    if source_record.acting_player == target_record.acting_player:
        if target_history.get(source_record.turn_index) != source_record.observation:
            raise SourceRootReplayError("source_record_observation_does_not_match_target_history")
        for entry in source_record.history:
            if target_history.get(entry.turn_index) != entry.observation:
                raise SourceRootReplayError("source_record_history_does_not_match_target_history")
    _require_compatible_public_prefix(source_record, target_record)


def _require_compatible_public_prefix(
    source_record: PublicDecisionRecord, target_record: PublicDecisionRecord
) -> None:
    """Reject cross-seat repair unless their concrete public history agrees."""

    source_rounds = source_record.public_resolved_action_rounds
    target_rounds = target_record.public_resolved_action_rounds[: source_record.turn_index]
    if len(source_rounds) != source_record.turn_index or len(target_rounds) != source_record.turn_index:
        raise SourceRootReplayError("source_record_public_prefix_length_does_not_match_turn")
    for source_round, target_round in zip(source_rounds, target_rounds, strict=True):
        if source_round.turn_index != target_round.turn_index:
            raise SourceRootReplayError("source_record_public_round_index_does_not_match_target_prefix")
        if set(source_round.actions) != set(target_round.actions):
            raise SourceRootReplayError("source_record_public_round_players_do_not_match_target_prefix")
        for player_id in source_round.actions:
            source_action = source_round.actions[player_id]
            target_action = target_round.actions[player_id]
            if _is_unresolved_public_event(source_action) or _is_unresolved_public_event(target_action):
                continue
            if source_action != target_action:
                raise SourceRootReplayError("source_record_public_action_does_not_match_target_prefix")


def _is_unresolved_public_event(identifier: PublicActionIdentifier) -> bool:
    return identifier.kind == "event" and identifier.event_id == _UNRESOLVED_EVENT_ID


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
