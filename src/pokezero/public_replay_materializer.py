"""Sampled-world materialization for canonical public replay identifiers.

Persisted public prefixes never contain request-local opponent action indexes.
This module resolves public move/species IDs after a belief world is sampled.
It also handles a small, explicit set of no-effect public events (``|cant|``
and bridge-marked unresolved actions), using a deterministic sampled-world legal
representative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .public_decision_corpus import PublicActionIdentifier, PublicResolvedActionRound
from .replay_branching import replay_action_rounds


_SUPPORTED_CANT_REASONS = frozenset({"slp", "frz", "par", "flinch", "recharge", "truant"})
_SUPPORTED_UNRESOLVED_EVENTS = frozenset({"unresolved-public-event", "unresolved-public-action"})


class PublicReplayError(ValueError):
    """A named public-only replay failure suitable for corpus skip accounting."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PublicEventCanonicalization:
    """Public proof that a no-effect event used a sampled-world representative."""

    turn_index: int
    player_id: str
    event_id: str
    resolution: str = "sampled-world-lowest-legal-action"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "turn_index": self.turn_index,
            "player_id": self.player_id,
            "event_id": self.event_id,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class PublicReplayMaterialization:
    """Ephemeral raw replay actions plus serializable public canonicalizations."""

    terminal: Any
    requested_players: tuple[str, ...]
    replay_actions: Mapping[int, Mapping[str, int]]
    event_canonicalizations: tuple[PublicEventCanonicalization, ...]


def replay_public_action_rounds(
    env: Any,
    *,
    seed: int,
    format_id: str,
    public_action_rounds: tuple[PublicResolvedActionRound, ...],
    start_override: Any,
) -> PublicReplayMaterialization:
    """Replay public rounds, resolving all request-local indexes in this world only."""

    replay_action_rounds(
        env,
        seed=seed,
        format_id=format_id,
        action_rounds=(),
        start_override=start_override,
        check_prefix_observations=False,
    )
    replay_actions: dict[int, dict[str, int]] = {}
    canonicalizations: list[PublicEventCanonicalization] = []
    for expected_turn, action_round in enumerate(public_action_rounds):
        if action_round.turn_index != expected_turn:
            raise PublicReplayError("noncontiguous_public_action_round")
        requested_players = tuple(str(player) for player in env.requested_players())
        if set(requested_players) != set(action_round.actions):
            raise PublicReplayError("sampled_world_request_shape_mismatch")
        actions: dict[str, int] = {}
        for player, identifier in action_round.actions.items():
            action_index, canonicalization = resolve_public_action_identifier(
                env.observe(player),
                identifier,
                turn_index=action_round.turn_index,
                player_id=player,
            )
            actions[player] = action_index
            if canonicalization is not None:
                canonicalizations.append(canonicalization)
        env.step(actions)
        replay_actions[action_round.turn_index] = actions
    return PublicReplayMaterialization(
        terminal=env.terminal(),
        requested_players=tuple(str(player) for player in env.requested_players()),
        replay_actions=replay_actions,
        event_canonicalizations=tuple(canonicalizations),
    )


def resolve_public_action_identifier(
    observation: Any,
    identifier: PublicActionIdentifier,
    *,
    turn_index: int,
    player_id: str,
) -> tuple[int, PublicEventCanonicalization | None]:
    """Resolve one public identifier without consulting a source-battle request."""

    legal_actions = tuple(
        index for index, legal in enumerate(observation.legal_action_mask) if bool(legal)
    )
    if identifier.kind == "event":
        return _canonicalize_public_event(
            identifier,
            legal_actions=legal_actions,
            turn_index=turn_index,
            player_id=player_id,
        )
    candidates = observation.metadata.get("action_candidates")
    if not isinstance(candidates, Sequence):
        raise PublicReplayError("sampled_world_missing_action_candidates")
    matches: list[int] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        action_index = candidate.get("action_index")
        if not isinstance(action_index, int) or action_index not in legal_actions:
            continue
        if identifier.kind == "move" and candidate.get("kind") == "move":
            if _normalized_identifier(candidate.get("move_id")) == _normalized_identifier(identifier.move_id):
                matches.append(action_index)
        elif identifier.kind == "switch" and candidate.get("kind") == "switch":
            species = candidate.get("switched_species")
            pokemon = candidate.get("pokemon")
            if species is None and isinstance(pokemon, Mapping):
                species = pokemon.get("species")
            if _normalized_identifier(species) == _normalized_identifier(identifier.switched_species):
                matches.append(action_index)
    if not matches:
        kind = "move" if identifier.kind == "move" else "switch"
        raise PublicReplayError(f"public_{kind}_identifier_unavailable")
    return min(matches), None


def public_event_prefix_summary(public_action_rounds: Sequence[PublicResolvedActionRound]) -> dict[str, Any]:
    """Summarize persisted event identifiers without deriving any action index."""

    event_ids = tuple(
        str(identifier.event_id)
        for action_round in public_action_rounds
        for identifier in action_round.actions.values()
        if identifier.kind == "event"
    )
    unsupported = tuple(event_id for event_id in event_ids if not _is_supported_no_effect_event(event_id))
    return {
        "public_event_count": len(event_ids),
        "public_event_ids": list(event_ids),
        "unsupported_public_event_count": len(unsupported),
        "unsupported_public_event_ids": list(unsupported),
    }


def _canonicalize_public_event(
    identifier: PublicActionIdentifier,
    *,
    legal_actions: Sequence[int],
    turn_index: int,
    player_id: str,
) -> tuple[int, PublicEventCanonicalization]:
    event_id = str(identifier.event_id)
    if not _is_supported_no_effect_event(event_id):
        raise PublicReplayError(f"unsupported_public_event:{event_id}")
    if not legal_actions:
        raise PublicReplayError("public_event_canonicalization_no_legal_action")
    return (
        min(legal_actions),
        PublicEventCanonicalization(turn_index=turn_index, player_id=player_id, event_id=event_id),
    )


def _is_supported_no_effect_event(event_id: str) -> bool:
    if event_id in _SUPPORTED_UNRESOLVED_EVENTS:
        # The bridge emits this only when the requested player has no public
        # move/switch/cant event for the completed round. The hidden action did
        # not resolve publicly, so a sampled-world legal representative is the
        # only replay-safe public choice and exposes no request-local slot.
        return True
    prefix, separator, reason = event_id.partition(":")
    return prefix == "cant" and bool(separator) and reason in _SUPPORTED_CANT_REASONS


def _normalized_identifier(value: object) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())
