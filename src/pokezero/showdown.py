"""Minimal Showdown replay normalization helpers.

This module is intentionally small: it is a testable boundary between raw
Showdown protocol seats (`p1`/`p2`) and PokeZero's player-relative model input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping, Optional, Sequence

from .actions import (
    ACTION_COUNT,
    MOVE_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
)
from .belief import PlayerBeliefView, PokemonSetSource, PublicBattleBeliefEngine, RevealedPokemonBelief
from .observation import (
    ACTION_CANDIDATE_TOKEN_COUNT,
    FIELD_TOKEN_COUNT,
    OPPONENT_POKEMON_TOKEN_COUNT,
    ObservationPerspective,
    ObservationSpec,
    PokeZeroObservationV0,
    SELF_POKEMON_TOKEN_COUNT,
    opponent_showdown_slot,
)

BELIEF_ABILITY_BUCKET_COUNT = 8
BELIEF_ITEM_BUCKET_COUNT = 8
BELIEF_MOVE_BUCKET_COUNT = 64
BELIEF_FACT_BUCKET_COUNT = BELIEF_ABILITY_BUCKET_COUNT + BELIEF_ITEM_BUCKET_COUNT + BELIEF_MOVE_BUCKET_COUNT
DEFAULT_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=4 + BELIEF_FACT_BUCKET_COUNT,
    numeric_feature_count=12,
)
CATEGORY_ID_BUCKETS = 1_000_000
CATEGORY_PRIMARY = 0
CATEGORY_SECONDARY = 1
CATEGORY_ROLE = 2
CATEGORY_SLOT = 3
CATEGORY_BELIEF_ABILITY_OFFSET = 4
CATEGORY_BELIEF_ITEM_OFFSET = CATEGORY_BELIEF_ABILITY_OFFSET + BELIEF_ABILITY_BUCKET_COUNT
CATEGORY_BELIEF_MOVE_OFFSET = CATEGORY_BELIEF_ITEM_OFFSET + BELIEF_ITEM_BUCKET_COUNT
NUMERIC_HP_FRACTION = 0
NUMERIC_ACTIVE = 1
NUMERIC_LEGAL = 2
NUMERIC_PRESENT = 3
NUMERIC_REVEALED_MOVE_COUNT = 4
NUMERIC_CANDIDATE_SET_COUNT = 5
NUMERIC_UNCERTAINTY = 6
NUMERIC_POSSIBLE_ABILITY_COUNT = 7
NUMERIC_POSSIBLE_ITEM_COUNT = 8
NUMERIC_POSSIBLE_MOVE_COUNT = 9
NUMERIC_REVEALED_ABILITY = 10
NUMERIC_REVEALED_ITEM = 11

FIELD_TOKEN_OFFSET = 0
SELF_POKEMON_TOKEN_OFFSET = FIELD_TOKEN_OFFSET + FIELD_TOKEN_COUNT
OPPONENT_POKEMON_TOKEN_OFFSET = SELF_POKEMON_TOKEN_OFFSET + SELF_POKEMON_TOKEN_COUNT
ACTION_CANDIDATE_TOKEN_OFFSET = OPPONENT_POKEMON_TOKEN_OFFSET + OPPONENT_POKEMON_TOKEN_COUNT
RECENT_EVENT_TOKEN_OFFSET = ACTION_CANDIDATE_TOKEN_OFFSET + ACTION_CANDIDATE_TOKEN_COUNT


@dataclass(frozen=True)
class ShowdownPokemon:
    ident: str
    showdown_slot: str
    species: str
    condition: Optional[str] = None
    active: bool = False
    details: Optional[str] = None


@dataclass(frozen=True)
class ShowdownReplayState:
    battle_id: str
    players: Mapping[str, str]
    requests: Mapping[str, Mapping[str, Any]]
    public_active: Mapping[str, ShowdownPokemon]
    public_revealed: Mapping[str, tuple[ShowdownPokemon, ...]]
    side_conditions: Mapping[str, tuple[str, ...]]
    side_condition_counts: Mapping[str, Mapping[str, int]]
    public_events: tuple["ShowdownPublicEvent", ...]
    public_lines: tuple[str, ...]
    winner: Optional[str] = None


@dataclass(frozen=True)
class ShowdownPublicEvent:
    event_type: str
    raw_line: str
    actor_slot: Optional[str] = None
    actor_ident: Optional[str] = None
    target_slot: Optional[str] = None
    target_ident: Optional[str] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None


@dataclass(frozen=True)
class PlayerRelativePublicEvent:
    event_type: str
    raw_line: str
    actor_role: str = "none"
    target_role: str = "none"
    primary: Optional[str] = None
    secondary: Optional[str] = None
    relative_line: Optional[str] = None


@dataclass(frozen=True)
class ShowdownSubmission:
    showdown_slot: str
    choice: str


@dataclass(frozen=True)
class PlayerRelativeBattleState:
    battle_id: str
    player_id: str
    perspective: ObservationPerspective
    request: Mapping[str, Any] | None
    request_kind: str
    self_team: tuple[ShowdownPokemon, ...]
    opponent_team: tuple[ShowdownPokemon, ...]
    self_side_conditions: tuple[str, ...]
    opponent_side_conditions: tuple[str, ...]
    self_side_condition_counts: Mapping[str, int]
    opponent_side_condition_counts: Mapping[str, int]
    belief_view: PlayerBeliefView
    legal_action_mask: tuple[bool, ...]
    recent_events: tuple[PlayerRelativePublicEvent, ...]
    recent_public_events: tuple[str, ...]
    winner: Optional[str] = None

    @property
    def self_active(self) -> ShowdownPokemon | None:
        return next((pokemon for pokemon in self.self_team if pokemon.active), None)

    @property
    def opponent_active(self) -> ShowdownPokemon | None:
        return next((pokemon for pokemon in self.opponent_team if pokemon.active), None)


def parse_showdown_replay(lines: Sequence[str], *, battle_id: str = "replay") -> ShowdownReplayState:
    """Parse compact Showdown protocol lines into transport-level state."""
    players: dict[str, str] = {}
    requests: dict[str, Mapping[str, Any]] = {}
    public_active: dict[str, ShowdownPokemon] = {}
    public_revealed: dict[str, list[ShowdownPokemon]] = {}
    side_condition_counts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
    public_events: list[ShowdownPublicEvent] = []
    public_lines: list[str] = []
    winner: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            continue
        parts = line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""
        if event_type == "player" and len(parts) >= 4:
            showdown_slot = parts[2]
            if showdown_slot in {"p1", "p2"}:
                players[showdown_slot] = parts[3]
            public_events.append(_public_event_from_line(line))
            public_lines.append(line)
            continue
        if event_type == "request" and len(parts) >= 3:
            payload = _decode_request_payload(line)
            side = payload.get("side") if isinstance(payload.get("side"), Mapping) else {}
            showdown_slot = side.get("id") if isinstance(side, Mapping) else None
            if showdown_slot in {"p1", "p2"}:
                requests[showdown_slot] = payload
            continue
        if event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
            pokemon = _pokemon_from_public_line(parts)
            if pokemon is not None:
                public_active[pokemon.showdown_slot] = pokemon
                _record_public_reveal(public_revealed, pokemon)
            public_events.append(_public_event_from_line(line))
            public_lines.append(line)
            continue
        if event_type == "win" and len(parts) >= 3:
            winner = parts[2]
            public_events.append(_public_event_from_line(line))
            public_lines.append(line)
            continue
        _update_side_conditions(parts, side_condition_counts)
        public_events.append(_public_event_from_line(line))
        public_lines.append(line)

    return ShowdownReplayState(
        battle_id=battle_id,
        players=players,
        requests=requests,
        public_active=public_active,
        public_revealed={slot: tuple(pokemon) for slot, pokemon in public_revealed.items()},
        side_conditions={slot: tuple(sorted(conditions)) for slot, conditions in _side_conditions_from_counts(side_condition_counts).items()},
        side_condition_counts={
            slot: dict(sorted(conditions.items()))
            for slot, conditions in side_condition_counts.items()
        },
        public_events=tuple(public_events),
        public_lines=tuple(public_lines),
        winner=winner,
    )


def detect_showdown_slot(
    replay: ShowdownReplayState,
    *,
    player_name: str | None = None,
    configured_showdown_slot: str | None = None,
) -> str:
    """Resolve the actual Showdown side for a player.

    Player name from public battle state wins over a stale configured default.
    """
    normalized_name = _normalize_name(player_name)
    if normalized_name:
        for showdown_slot, name in replay.players.items():
            if _normalize_name(name) == normalized_name:
                return showdown_slot
        for showdown_slot, request in replay.requests.items():
            side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
            side_name = side.get("name") if isinstance(side, Mapping) else None
            if _normalize_name(side_name) == normalized_name:
                return showdown_slot
    if configured_showdown_slot in {"p1", "p2"}:
        return configured_showdown_slot
    raise ValueError("Unable to detect Showdown slot from player_name or configured_showdown_slot.")


def normalize_for_player(
    replay: ShowdownReplayState,
    *,
    player_id: str,
    player_name: str | None = None,
    configured_showdown_slot: str | None = None,
    format_id: str | None = None,
    set_source: PokemonSetSource | None = None,
    recent_event_limit: int = 24,
) -> PlayerRelativeBattleState:
    """Build a player-relative state view from raw Showdown transport state."""
    showdown_slot = detect_showdown_slot(
        replay,
        player_name=player_name,
        configured_showdown_slot=configured_showdown_slot,
    )
    opponent_slot = opponent_showdown_slot(showdown_slot)
    perspective = ObservationPerspective(
        player_id=player_id,
        showdown_slot=showdown_slot,
        opponent_showdown_slot=opponent_slot,
    )
    request = replay.requests.get(showdown_slot)
    self_team = _self_team_from_request(request, showdown_slot)
    opponent_team = _opponent_team_from_public_state(replay, opponent_slot)
    belief_engine = PublicBattleBeliefEngine.from_events(
        replay.public_events,
        format_id=format_id,
        set_source=set_source,
    )
    belief_engine.resolve_pending_switches_at_boundary()
    belief_view = belief_engine.snapshot().for_player(showdown_slot)
    recent_events = tuple(
        _relative_public_event(event, self_slot=showdown_slot, opponent_slot=opponent_slot)
        for event in replay.public_events[-recent_event_limit:]
    )
    return PlayerRelativeBattleState(
        battle_id=replay.battle_id,
        player_id=player_id,
        perspective=perspective,
        request=request,
        request_kind=_request_kind(request),
        self_team=self_team,
        opponent_team=opponent_team,
        self_side_conditions=tuple(replay.side_conditions.get(showdown_slot, ())),
        opponent_side_conditions=tuple(replay.side_conditions.get(opponent_slot, ())),
        self_side_condition_counts=dict(replay.side_condition_counts.get(showdown_slot, {})),
        opponent_side_condition_counts=dict(replay.side_condition_counts.get(opponent_slot, {})),
        belief_view=belief_view,
        legal_action_mask=_legal_action_mask(request),
        recent_events=recent_events,
        recent_public_events=tuple(event.relative_line or event.raw_line for event in recent_events),
        winner=replay.winner,
    )


def observation_from_player_state(
    state: PlayerRelativeBattleState,
    *,
    spec: ObservationSpec = DEFAULT_REPLAY_OBSERVATION_SPEC,
) -> PokeZeroObservationV0:
    """Encode normalized replay state into fixed-shape observation rows."""
    categorical_ids = _blank_categorical_rows(spec)
    numeric_features = _blank_numeric_rows(spec)
    _encode_field_token(categorical_ids, numeric_features, state)
    _encode_pokemon_tokens(
        categorical_ids,
        numeric_features,
        SELF_POKEMON_TOKEN_OFFSET,
        state.self_team,
        role="self",
        limit=SELF_POKEMON_TOKEN_COUNT,
    )
    opponent_beliefs = state.belief_view.opponent_by_species()
    _encode_pokemon_tokens(
        categorical_ids,
        numeric_features,
        OPPONENT_POKEMON_TOKEN_OFFSET,
        state.opponent_team,
        role="opponent",
        limit=OPPONENT_POKEMON_TOKEN_COUNT,
        beliefs_by_species=opponent_beliefs,
    )
    _encode_action_tokens(categorical_ids, numeric_features, state)
    _encode_recent_event_tokens(categorical_ids, numeric_features, state, spec)
    token_type_ids = _token_type_ids(spec)
    attention_mask = _attention_mask(state, spec)
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(row) for row in categorical_ids),
        numeric_features=tuple(tuple(row) for row in numeric_features),
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        legal_action_mask=state.legal_action_mask,
        perspective=state.perspective,
        metadata=_observation_metadata(state),
    )


def stable_category_id(value: str, *, buckets: int = CATEGORY_ID_BUCKETS) -> int:
    """Map a category string to a deterministic positive id.

    This is a stable hash-bucket encoder for early experiments. Explicit
    vocabularies can replace it once the observation vocabulary is finalized.
    """
    normalized = str(value or "").strip().lower()
    if not normalized:
        return 0
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % buckets) + 1


def showdown_choice_for_action(state: PlayerRelativeBattleState, action_index: int) -> str:
    """Translate a 0-8 policy action index to a Showdown choice string."""
    if action_index < 0 or action_index >= ACTION_COUNT:
        raise ValueError(f"action_index must be between 0 and {ACTION_COUNT - 1}.")
    if not state.legal_action_mask[action_index]:
        raise ValueError(f"action_index {action_index} is not legal for the current request.")
    if is_move_action(action_index):
        return f"move {action_index + 1}"
    if is_switch_action(action_index):
        active_team_index = _active_team_index(state.self_team)
        if active_team_index is None:
            raise ValueError("Cannot translate switch action without an active self Pokemon.")
        switch_slot = action_index - MOVE_ACTION_COUNT
        switch_targets = canonical_switch_action_map(active_team_index, team_size=len(state.self_team))
        if switch_slot >= len(switch_targets):
            raise ValueError(f"action_index {action_index} is outside the current switch target map.")
        return f"switch {switch_targets[switch_slot] + 1}"
    raise ValueError(f"Unsupported action_index: {action_index}.")


def showdown_submission_for_action(state: PlayerRelativeBattleState, action_index: int) -> ShowdownSubmission:
    """Translate a policy action into the protocol side and choice string."""
    return ShowdownSubmission(
        showdown_slot=state.perspective.showdown_slot,
        choice=showdown_choice_for_action(state, action_index),
    )


def _decode_request_payload(line: str) -> Mapping[str, Any]:
    payload_text = line[len("|request|") :]
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Showdown request payload: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Showdown request payload must be a JSON object.")
    return payload


def _pokemon_from_public_line(parts: Sequence[str]) -> ShowdownPokemon | None:
    ident = parts[2]
    showdown_slot = _slot_from_ident(ident)
    if showdown_slot is None:
        return None
    details = parts[3] if len(parts) > 3 else ""
    return ShowdownPokemon(
        ident=ident,
        showdown_slot=showdown_slot,
        species=_species_from_details(details) or _species_from_ident(ident),
        condition=parts[4] if len(parts) > 4 else None,
        active=True,
        details=details,
    )


def _side_conditions_from_counts(side_condition_counts: Mapping[str, Mapping[str, int]]) -> dict[str, set[str]]:
    return {
        slot: {condition for condition, count in conditions.items() if count > 0}
        for slot, conditions in side_condition_counts.items()
    }


def _update_side_conditions(parts: Sequence[str], side_conditions: dict[str, dict[str, int]]) -> None:
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type not in {"-sidestart", "-sideend"} or len(parts) < 4:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in side_conditions:
        return
    condition = _side_condition_identifier(parts[3])
    if not condition:
        return
    if event_type == "-sidestart":
        side_conditions[slot][condition] = min(
            _side_condition_max_layers(condition),
            side_conditions[slot].get(condition, 0) + 1,
        )
    else:
        side_conditions[slot].pop(condition, None)


def _side_condition_max_layers(condition: str) -> int:
    if condition == "spikes":
        return 3
    if condition == "toxicspikes":
        return 2
    return 1


def _side_condition_identifier(raw_condition: str) -> str:
    condition = raw_condition.strip()
    if condition.lower().startswith("move:"):
        condition = condition.split(":", 1)[1].strip()
    return _normalize_identifier(condition)


def _public_event_from_line(line: str) -> ShowdownPublicEvent:
    parts = line.split("|")
    event_type = parts[1] if len(parts) > 1 and parts[1] else "unknown"
    actor_ident: Optional[str] = None
    actor_slot: Optional[str] = None
    target_ident: Optional[str] = None
    target_slot: Optional[str] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None

    if event_type == "player" and len(parts) >= 4:
        actor_slot = parts[2] if parts[2] in {"p1", "p2"} else None
        primary = parts[3]
    elif event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
        actor_ident = parts[2]
        actor_slot = _slot_from_ident(actor_ident)
        primary = _species_from_details(parts[3]) or _species_from_ident(actor_ident)
        secondary = parts[4] if len(parts) > 4 else None
    elif event_type == "move" and len(parts) >= 4:
        actor_ident = parts[2]
        actor_slot = _slot_from_ident(actor_ident)
        primary = parts[3]
        if len(parts) > 4:
            target_ident = parts[4]
            target_slot = _slot_from_ident(target_ident)
    elif event_type in {
        "-ability",
        "ability",
        "-activate",
        "-boost",
        "-curestatus",
        "-damage",
        "-heal",
        "-item",
        "-sideend",
        "-sidestart",
        "-status",
        "-unboost",
        "faint",
    } and len(parts) >= 3:
        target_ident = parts[2]
        target_slot = _slot_from_ident(target_ident)
        primary = parts[3] if len(parts) > 3 else None
        secondary = parts[4] if len(parts) > 4 else None
    elif event_type == "win" and len(parts) >= 3:
        primary = parts[2]
    else:
        actor_ident = parts[2] if len(parts) > 2 and _slot_from_ident(parts[2]) else None
        actor_slot = _slot_from_ident(actor_ident or "")
        primary = parts[3] if len(parts) > 3 else None
        secondary = parts[4] if len(parts) > 4 else None

    return ShowdownPublicEvent(
        event_type=event_type,
        raw_line=line,
        actor_slot=actor_slot,
        actor_ident=actor_ident,
        target_slot=target_slot,
        target_ident=target_ident,
        primary=primary,
        secondary=secondary,
    )


def _relative_public_event(
    event: ShowdownPublicEvent,
    *,
    self_slot: str,
    opponent_slot: str,
) -> PlayerRelativePublicEvent:
    return PlayerRelativePublicEvent(
        event_type=event.event_type,
        raw_line=event.raw_line,
        actor_role=_relative_role(event.actor_slot, self_slot=self_slot, opponent_slot=opponent_slot),
        target_role=_relative_role(event.target_slot, self_slot=self_slot, opponent_slot=opponent_slot),
        primary=event.primary,
        secondary=event.secondary,
        relative_line=_relative_public_line(event, self_slot=self_slot, opponent_slot=opponent_slot),
    )


def _relative_role(slot: str | None, *, self_slot: str, opponent_slot: str) -> str:
    if slot == self_slot:
        return "self"
    if slot == opponent_slot:
        return "opponent"
    return "none"


def _relative_public_line(
    event: ShowdownPublicEvent,
    *,
    self_slot: str,
    opponent_slot: str,
) -> str:
    parts = event.raw_line.split("|")
    if len(parts) < 3:
        return event.raw_line
    normalized = [
        _normalize_public_field(field, self_slot=self_slot, opponent_slot=opponent_slot)
        for field in parts
    ]
    return "|".join(normalized)


def _self_team_from_request(request: Mapping[str, Any] | None, showdown_slot: str) -> tuple[ShowdownPokemon, ...]:
    side = request.get("side") if isinstance(request, Mapping) and isinstance(request.get("side"), Mapping) else {}
    pokemon_rows = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon_rows, list):
        return ()
    team: list[ShowdownPokemon] = []
    for row in pokemon_rows:
        if not isinstance(row, Mapping):
            continue
        ident = str(row.get("ident") or "")
        team.append(
            ShowdownPokemon(
                ident=ident,
                showdown_slot=_slot_from_ident(ident) or showdown_slot,
                species=_species_from_request_pokemon(row),
                condition=str(row.get("condition")) if row.get("condition") is not None else None,
                active=bool(row.get("active")),
                details=str(row.get("details")) if row.get("details") is not None else None,
            )
        )
    return tuple(team)


def _opponent_team_from_public_state(
    replay: ShowdownReplayState,
    opponent_slot: str,
) -> tuple[ShowdownPokemon, ...]:
    return tuple(replay.public_revealed.get(opponent_slot, ()))


def _blank_categorical_rows(spec: ObservationSpec) -> list[list[int]]:
    return [[0] * spec.categorical_feature_count for _ in range(spec.token_count)]


def _blank_numeric_rows(spec: ObservationSpec) -> list[list[float]]:
    return [[0.0] * spec.numeric_feature_count for _ in range(spec.token_count)]


def _encode_field_token(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
) -> None:
    _set_category(categorical_ids[FIELD_TOKEN_OFFSET], CATEGORY_PRIMARY, f"request_kind:{state.request_kind}")
    # Winner identity is deliberately NOT encoded: it is constant ("none") at every decision
    # point (the rollout records observations only while the game is live) and would otherwise
    # be the game outcome leaking into the model input. The SECONDARY slot stays padding.
    _set_category(categorical_ids[FIELD_TOKEN_OFFSET], CATEGORY_ROLE, "field")
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_PRESENT, 1.0)


def _encode_pokemon_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    offset: int,
    pokemon: Sequence[ShowdownPokemon],
    *,
    role: str,
    limit: int,
    beliefs_by_species: Mapping[str, RevealedPokemonBelief] | None = None,
) -> None:
    for slot_index, candidate in enumerate(pokemon[:limit]):
        token_index = offset + slot_index
        belief = _belief_for_species(beliefs_by_species, candidate.species)
        condition = _condition_features(belief.condition if belief is not None else candidate.condition)
        revealed_moves = belief.revealed_moves if belief is not None else ()
        revealed_ability = belief.revealed_ability if belief is not None else None
        revealed_item = belief.revealed_item if belief is not None else None
        possible_abilities = belief.possible_abilities if belief is not None else ()
        possible_items = belief.possible_items if belief is not None else ()
        possible_moves = belief.possible_moves if belief is not None else ()
        ability_feature_values = _known_or_possible_values(revealed_ability, possible_abilities)
        item_feature_values = _known_or_possible_values(revealed_item, possible_items)
        candidate_set_count = belief.candidate_set_count if belief is not None else None
        uncertainty = belief.uncertainty if belief is not None else 1.0
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"species:{candidate.species}")
        status = belief.status if belief is not None and belief.status is not None else condition.status
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, f"status:{status}")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, f"pokemon:{role}")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"{role}_slot:{slot_index}")
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_ability", ability_feature_values)
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_item", item_feature_values)
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_move", possible_moves)
        _set_numeric(numeric_features[token_index], NUMERIC_HP_FRACTION, condition.hp_fraction or 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_ACTIVE, 1.0 if candidate.active else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_LEGAL, 0.0 if condition.fainted else 1.0)
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0)
        _set_numeric(numeric_features[token_index], NUMERIC_REVEALED_MOVE_COUNT, float(len(revealed_moves)))
        _set_numeric(numeric_features[token_index], NUMERIC_CANDIDATE_SET_COUNT, float(candidate_set_count or 0))
        _set_numeric(numeric_features[token_index], NUMERIC_UNCERTAINTY, uncertainty)
        _set_numeric(numeric_features[token_index], NUMERIC_POSSIBLE_ABILITY_COUNT, float(len(ability_feature_values)))
        _set_numeric(numeric_features[token_index], NUMERIC_POSSIBLE_ITEM_COUNT, float(len(item_feature_values)))
        _set_numeric(numeric_features[token_index], NUMERIC_POSSIBLE_MOVE_COUNT, float(len(possible_moves)))
        _set_numeric(numeric_features[token_index], NUMERIC_REVEALED_ABILITY, 1.0 if revealed_ability else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_REVEALED_ITEM, 1.0 if revealed_item else 0.0)


def _encode_action_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
) -> None:
    active_request = _active_request(state.request)
    moves = active_request.get("moves") if isinstance(active_request, Mapping) else None
    for move_index in range(MOVE_ACTION_COUNT):
        token_index = ACTION_CANDIDATE_TOKEN_OFFSET + move_index
        move = moves[move_index] if isinstance(moves, list) and move_index < len(moves) else None
        move_name = _request_move_name(move) if isinstance(move, Mapping) else f"slot:{move_index + 1}"
        disabled = bool(move.get("disabled")) if isinstance(move, Mapping) else True
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"move:{move_name}")
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, "action:move")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, "action")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"move_slot:{move_index + 1}")
        _set_numeric(numeric_features[token_index], NUMERIC_LEGAL, 1.0 if state.legal_action_mask[move_index] else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0 if isinstance(move, Mapping) else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_ACTIVE, 0.0 if disabled else 1.0)

    active_team_index = _active_team_index(state.self_team)
    switch_targets = (
        canonical_switch_action_map(active_team_index, team_size=len(state.self_team))
        if active_team_index is not None and len(state.self_team) >= 2
        else ()
    )
    for switch_slot in range(ACTION_CANDIDATE_TOKEN_COUNT - MOVE_ACTION_COUNT):
        action_index = MOVE_ACTION_COUNT + switch_slot
        token_index = ACTION_CANDIDATE_TOKEN_OFFSET + action_index
        team_index = switch_targets[switch_slot] if switch_slot < len(switch_targets) else None
        pokemon = state.self_team[team_index] if team_index is not None and team_index < len(state.self_team) else None
        condition = _condition_features(pokemon.condition if pokemon is not None else None)
        species = pokemon.species if pokemon is not None else f"slot:{switch_slot + 1}"
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"species:{species}")
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, "action:switch")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, "action")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"switch_slot:{switch_slot + 1}")
        _set_numeric(numeric_features[token_index], NUMERIC_HP_FRACTION, condition.hp_fraction or 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_ACTIVE, 1.0 if pokemon is not None and pokemon.active else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_LEGAL, 1.0 if state.legal_action_mask[action_index] else 0.0)
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0 if pokemon is not None else 0.0)


def _encode_recent_event_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    spec: ObservationSpec,
) -> None:
    for event_index, event in enumerate(state.recent_events[: spec.recent_event_token_count]):
        token_index = RECENT_EVENT_TOKEN_OFFSET + event_index
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"event:{event.event_type}")
        event_detail = _event_detail_category(event)
        if event_detail is not None:
            _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, event_detail)
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, f"event_actor:{event.actor_role}")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"event_target:{event.target_role}")
        _set_numeric(numeric_features[token_index], NUMERIC_PRESENT, 1.0)


def _observation_metadata(state: PlayerRelativeBattleState) -> dict[str, Any]:
    return {
        "battle_id": state.battle_id,
        "player_id": state.player_id,
        "request_kind": state.request_kind,
        "showdown_slot": state.perspective.showdown_slot,
        "opponent_showdown_slot": state.perspective.opponent_showdown_slot,
        "self_side_conditions": list(state.self_side_conditions),
        "opponent_side_conditions": list(state.opponent_side_conditions),
        "self_side_condition_counts": dict(state.self_side_condition_counts),
        "opponent_side_condition_counts": dict(state.opponent_side_condition_counts),
        "self_active": _pokemon_metadata(state.self_active),
        "opponent_active": _pokemon_metadata(state.opponent_active),
        "self_team": [_pokemon_metadata(pokemon) for pokemon in state.self_team],
        "opponent_team": [_pokemon_metadata(pokemon) for pokemon in state.opponent_team],
        "action_candidates": _action_candidate_metadata(state),
        "recent_public_events": list(state.recent_public_events),
    }


def _action_candidate_metadata(state: PlayerRelativeBattleState) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    active_request = _active_request(state.request)
    moves = active_request.get("moves") if isinstance(active_request, Mapping) else None
    for move_index in range(MOVE_ACTION_COUNT):
        move = moves[move_index] if isinstance(moves, list) and move_index < len(moves) else None
        move_name = _request_move_name(move) if isinstance(move, Mapping) else f"slot:{move_index + 1}"
        candidates.append(
            {
                "action_index": move_index,
                "kind": "move",
                "legal": bool(state.legal_action_mask[move_index]),
                "move_slot": move_index + 1,
                "move_id": _normalize_identifier(move_name),
                "move_name": move_name,
                "disabled": bool(move.get("disabled")) if isinstance(move, Mapping) else True,
                "target_species": state.opponent_active.species if state.opponent_active is not None else None,
            }
        )

    active_team_index = _active_team_index(state.self_team)
    switch_targets = (
        canonical_switch_action_map(active_team_index, team_size=len(state.self_team))
        if active_team_index is not None and len(state.self_team) >= 2
        else ()
    )
    for switch_slot in range(ACTION_CANDIDATE_TOKEN_COUNT - MOVE_ACTION_COUNT):
        action_index = MOVE_ACTION_COUNT + switch_slot
        team_index = switch_targets[switch_slot] if switch_slot < len(switch_targets) else None
        pokemon = state.self_team[team_index] if team_index is not None and team_index < len(state.self_team) else None
        candidates.append(
            {
                "action_index": action_index,
                "kind": "switch",
                "legal": bool(state.legal_action_mask[action_index]),
                "switch_slot": switch_slot + 1,
                "team_index": team_index,
                "pokemon": _pokemon_metadata(pokemon),
            }
        )
    return candidates


def _pokemon_metadata(pokemon: ShowdownPokemon | None) -> dict[str, Any] | None:
    if pokemon is None:
        return None
    condition = _condition_features(pokemon.condition)
    return {
        "ident": pokemon.ident,
        "showdown_slot": pokemon.showdown_slot,
        "species": pokemon.species,
        "condition": pokemon.condition,
        "hp_fraction": condition.hp_fraction,
        "status": condition.status,
        "fainted": condition.fainted,
        "active": pokemon.active,
        "details": pokemon.details,
    }


@dataclass(frozen=True)
class _ConditionFeatures:
    hp_fraction: Optional[float]
    status: str
    fainted: bool


def _condition_features(condition: str | None) -> _ConditionFeatures:
    parts = str(condition or "").split()
    hp_fraction: Optional[float] = None
    if parts and "/" in parts[0]:
        numerator, _, denominator = parts[0].partition("/")
        try:
            hp_fraction = max(0.0, min(1.0, float(numerator) / float(denominator)))
        except (TypeError, ValueError, ZeroDivisionError):
            hp_fraction = None
    elif parts and parts[0] == "0":
        hp_fraction = 0.0
    fainted = "fnt" in parts
    status = next((part for part in parts[1:] if part != "fnt"), "none")
    return _ConditionFeatures(hp_fraction=hp_fraction, status=status, fainted=fainted)


def _set_category(row: list[int], index: int, value: str) -> None:
    if index < len(row):
        row[index] = stable_category_id(value)


def _set_numeric(row: list[float], index: int, value: float) -> None:
    if index < len(row):
        row[index] = float(value)


def _known_or_possible_values(known: str | None, possible: Sequence[str]) -> tuple[str, ...]:
    if known:
        return (known,)
    return _compact_belief_values(possible)


def _encode_belief_fact_categories(row: list[int], fact_kind: str, values: Sequence[str]) -> None:
    offset, bucket_count = _belief_bucket_range(fact_kind)
    for value in _compact_belief_values(values):
        normalized = _normalize_identifier(value)
        bucket = stable_category_id(f"belief_bucket:{fact_kind}:{normalized}", buckets=bucket_count) - 1
        column = offset + bucket
        if column >= len(row) or row[column]:
            continue
        row[column] = stable_category_id(f"belief:{fact_kind}:{normalized}")


def _belief_bucket_range(fact_kind: str) -> tuple[int, int]:
    if fact_kind == "possible_ability":
        return CATEGORY_BELIEF_ABILITY_OFFSET, BELIEF_ABILITY_BUCKET_COUNT
    if fact_kind == "possible_item":
        return CATEGORY_BELIEF_ITEM_OFFSET, BELIEF_ITEM_BUCKET_COUNT
    if fact_kind == "possible_move":
        return CATEGORY_BELIEF_MOVE_OFFSET, BELIEF_MOVE_BUCKET_COUNT
    raise ValueError(f"unsupported belief fact kind: {fact_kind!r}")


def _compact_belief_values(values: Sequence[str], *, limit: int | None = None) -> tuple[str, ...]:
    compact_by_key: dict[str, str] = {}
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        key = _normalize_identifier(value)
        if not key or key in compact_by_key:
            continue
        compact_by_key[key] = value
    compact = tuple(value for _, value in sorted(compact_by_key.items()))
    if limit is None:
        return compact
    return compact[:limit]


def _belief_for_species(
    beliefs_by_species: Mapping[str, RevealedPokemonBelief] | None,
    species: str,
) -> RevealedPokemonBelief | None:
    if not beliefs_by_species:
        return None
    return beliefs_by_species.get(_normalize_identifier(species))


def _legal_action_mask(request: Mapping[str, Any] | None) -> tuple[bool, ...]:
    mask = [False] * ACTION_COUNT
    if not isinstance(request, Mapping) or request.get("wait"):
        return tuple(mask)

    force_switch = request.get("forceSwitch")
    force_switch_requested = isinstance(force_switch, list) and any(bool(slot) for slot in force_switch)
    if not force_switch_requested:
        active_rows = request.get("active")
        active = active_rows[0] if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping) else None
        moves = active.get("moves") if isinstance(active, Mapping) else None
        if isinstance(moves, list):
            for move_index, move in enumerate(moves[:MOVE_ACTION_COUNT]):
                if isinstance(move, Mapping) and not move.get("disabled", False):
                    mask[move_index] = True

    if force_switch_requested or _switching_allowed(request):
        active_team_index = _active_team_index(_self_team_from_request(request, _request_side_id(request) or "p1"))
        team_size = _team_size_from_request(request)
        if active_team_index is not None and team_size >= 2:
            for switch_slot, team_index in enumerate(canonical_switch_action_map(active_team_index, team_size=team_size)):
                pokemon = _request_pokemon_at(request, team_index)
                if pokemon is not None and _can_switch_to(pokemon):
                    mask[MOVE_ACTION_COUNT + switch_slot] = True
    return tuple(mask)


def _request_kind(request: Mapping[str, Any] | None) -> str:
    if not isinstance(request, Mapping):
        return "none"
    if request.get("wait"):
        return "wait"
    if request.get("teamPreview"):
        return "team_preview"
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(slot) for slot in force_switch):
        return "force_switch"
    if request.get("active"):
        return "move"
    return "unknown"


def _switching_allowed(request: Mapping[str, Any]) -> bool:
    active_rows = request.get("active")
    active = active_rows[0] if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping) else None
    if isinstance(active, Mapping) and (active.get("trapped") is True or active.get("maybeTrapped") is True):
        return False
    return _request_kind(request) == "move"


def _request_pokemon_at(request: Mapping[str, Any], team_index: int) -> Mapping[str, Any] | None:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    if not isinstance(pokemon, list) or team_index < 0 or team_index >= len(pokemon):
        return None
    candidate = pokemon[team_index]
    return candidate if isinstance(candidate, Mapping) else None


def _active_request(request: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    active_rows = request.get("active") if isinstance(request, Mapping) else None
    if isinstance(active_rows, list) and active_rows and isinstance(active_rows[0], Mapping):
        return active_rows[0]
    return None


def _request_move_name(move: Mapping[str, Any]) -> str:
    for key in ("id", "move"):
        value = move.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _event_detail_category(event: PlayerRelativePublicEvent) -> str | None:
    """Enumerable detail token for a recent event, or None to leave the slot as padding.

    Only emits closed-vocabulary tokens (move / species / status, which reuse the same ids as
    the action and pokemon tokens). The previously-emitted dynamic strings are dropped, since
    they are unactionable in randbats and were the only things landing in the OOV block:
      - -damage/-heal carried the raw HP string ("234/267 tox") -> HP is already numeric
        (NUMERIC_HP_FRACTION) and any status is conveyed by -status events;
      - player/win carried usernames, which are meaningless in random battles;
      - the fallback carried free-form payloads (noise).
    The event *type* is still encoded in CATEGORY_PRIMARY, so no actionable signal is lost.
    """
    primary = event.primary or "none"
    if event.event_type == "move":
        return f"move:{primary}"
    if event.event_type in {"switch", "drag", "replace"}:
        return f"species:{primary}"
    if event.event_type in {"-status", "-curestatus"}:
        return f"status:{primary}"
    return None


def _request_side_id(request: Mapping[str, Any]) -> str | None:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    side_id = side.get("id") if isinstance(side, Mapping) else None
    return side_id if side_id in {"p1", "p2"} else None


def _team_size_from_request(request: Mapping[str, Any]) -> int:
    side = request.get("side") if isinstance(request.get("side"), Mapping) else {}
    pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
    return len(pokemon) if isinstance(pokemon, list) else 0


def _can_switch_to(pokemon: Mapping[str, Any]) -> bool:
    if pokemon.get("active"):
        return False
    condition = str(pokemon.get("condition") or "")
    return not condition.startswith("0 ")


def _active_team_index(team: Sequence[ShowdownPokemon]) -> int | None:
    for index, pokemon in enumerate(team):
        if pokemon.active:
            return index
    return None


def _species_from_request_pokemon(row: Mapping[str, Any]) -> str:
    details = row.get("details")
    ident = row.get("ident")
    if isinstance(details, str) and details.strip():
        return _species_from_details(details)
    if isinstance(ident, str):
        return _species_from_ident(ident)
    return "unknown"


def _species_from_details(details: str) -> str:
    return details.split(",", 1)[0].strip()


def _species_from_ident(ident: str) -> str:
    return ident.split(":", 1)[-1].strip() or "unknown"


def _slot_from_ident(ident: str) -> str | None:
    match = re.match(r"^(p[12])", ident.strip())
    return match.group(1) if match else None


def _normalize_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_public_field(field: str, *, self_slot: str, opponent_slot: str) -> str:
    field = re.sub(rf"^{self_slot}([a-z]?):", r"self\1:", field)
    return re.sub(rf"^{opponent_slot}([a-z]?):", r"opponent\1:", field)


def _record_public_reveal(
    public_revealed: dict[str, list[ShowdownPokemon]],
    pokemon: ShowdownPokemon,
) -> None:
    current = public_revealed.setdefault(pokemon.showdown_slot, [])
    next_revealed: list[ShowdownPokemon] = []
    matched = False
    for existing in current:
        if _same_public_pokemon(existing, pokemon):
            next_revealed.append(pokemon)
            matched = True
        else:
            next_revealed.append(replace(existing, active=False))
    if not matched:
        next_revealed.append(pokemon)
    public_revealed[pokemon.showdown_slot] = next_revealed


def _same_public_pokemon(left: ShowdownPokemon, right: ShowdownPokemon) -> bool:
    return left.showdown_slot == right.showdown_slot and left.species == right.species


def _token_type_ids(spec: ObservationSpec) -> tuple[int, ...]:
    token_types: list[int] = []
    token_types.extend([0])
    token_types.extend([1] * 6)
    token_types.extend([2] * 6)
    token_types.extend([3] * ACTION_COUNT)
    token_types.extend([4] * spec.recent_event_token_count)
    return tuple(token_types)


def _attention_mask(state: PlayerRelativeBattleState, spec: ObservationSpec) -> tuple[bool, ...]:
    mask: list[bool] = []
    mask.extend([True])
    mask.extend(index < len(state.self_team) for index in range(6))
    mask.extend(index < len(state.opponent_team) for index in range(6))
    mask.extend([True] * ACTION_COUNT)
    mask.extend(index < len(state.recent_public_events) for index in range(spec.recent_event_token_count))
    return tuple(mask)
