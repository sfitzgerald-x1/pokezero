"""Minimal Showdown replay normalization helpers.

This module is intentionally small: it is a testable boundary between raw
Showdown protocol seats (`p1`/`p2`) and PokeZero's player-relative model input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .category_vocab import CategoryVocabulary
    from .dex import ShowdownDex

from .actions import (
    ACTION_COUNT,
    MOVE_ACTION_COUNT,
    canonical_switch_action_map,
    is_move_action,
    is_switch_action,
)
from .belief import PlayerBeliefView, PokemonSetSource, PublicBattleBeliefEngine, RevealedPokemonBelief
from .dex import resolve_move_base_power, resolve_move_effect
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

# Belief-fact columns are sized to the Gen 3 closed universe's max distinct values per species
# (measured from the randbat set universe): at most 2 abilities, 5 items, 14 possible moves. The
# values are placed positionally (sorted) into these columns — exact and collision-free.
BELIEF_ABILITY_BUCKET_COUNT = 2
BELIEF_ITEM_BUCKET_COUNT = 6
BELIEF_MOVE_BUCKET_COUNT = 16
BELIEF_FACT_BUCKET_COUNT = BELIEF_ABILITY_BUCKET_COUNT + BELIEF_ITEM_BUCKET_COUNT + BELIEF_MOVE_BUCKET_COUNT
# Fixed categorical columns (0-8), then belief-fact buckets, then active-mon volatile-status
# columns. Volatiles (confusion / leech seed / substitute / taunt / ...) are placed positionally
# like belief facts; 6 columns cover any realistic simultaneous set on one mon.
CATEGORY_FIXED_COUNT = 9
VOLATILE_BUCKET_COUNT = 6
DEFAULT_REPLAY_OBSERVATION_SPEC = ObservationSpec(
    categorical_feature_count=CATEGORY_FIXED_COUNT + BELIEF_FACT_BUCKET_COUNT + VOLATILE_BUCKET_COUNT,
    numeric_feature_count=44,
)
CATEGORY_ID_BUCKETS = 1_000_000
CATEGORY_PRIMARY = 0
CATEGORY_SECONDARY = 1
CATEGORY_ROLE = 2
CATEGORY_SLOT = 3
# Raw mechanical type facts (dex-derived). For pokemon/switch tokens: the mon's two types
# (TYPE_2 padding if mono-type). For move tokens: the move's type in TYPE_1, its damage class
# (physical/special/status) in MOVE_CATEGORY. These let the type chart + effectiveness emerge
# in the embedding space rather than being hand-computed.
CATEGORY_TYPE_1 = 4
CATEGORY_TYPE_2 = 5
CATEGORY_MOVE_CATEGORY = 6
# Move-effect TYPE (move tokens): move_effect:<id> — the move's primary OR secondary effect as
# one label: a status (brn/par/frz/...), a volatile (substitute/leechseed/flinch/...), or a
# target-explicit, magnitude-enumerated stat change (lower_foe_def_sharply / raise_self_atk /
# raise_self_all / lower_self_atkdef / ...). NUMERIC_EFFECT_CHANCE carries its probability
# (1.0 = guaranteed), so the model can tell e.g. a 10% freeze from a guaranteed setup, and a
# foe-debuff from a self-drawback. NUMERIC_SELF_HP_COST carries the move's upfront HP price.
CATEGORY_MOVE_EFFECT = 7
# Move priority bracket (move tokens): move_priority:<n> for the integer priority (e.g. +1 Quick
# Attack, -3 Focus Punch). Priority is a discrete turn-order bracket — a higher bracket always
# moves first regardless of speed — so a per-bracket embedding captures it better than the scalar
# NUMERIC_PRIORITY (kept for ordinal grounding).
CATEGORY_MOVE_PRIORITY = 8
CATEGORY_BELIEF_ABILITY_OFFSET = CATEGORY_FIXED_COUNT
CATEGORY_BELIEF_ITEM_OFFSET = CATEGORY_BELIEF_ABILITY_OFFSET + BELIEF_ABILITY_BUCKET_COUNT
CATEGORY_BELIEF_MOVE_OFFSET = CATEGORY_BELIEF_ITEM_OFFSET + BELIEF_ITEM_BUCKET_COUNT
# Active-mon volatile-status columns follow the belief blocks (volatile:<name>, positional).
CATEGORY_VOLATILE_OFFSET = CATEGORY_BELIEF_MOVE_OFFSET + BELIEF_MOVE_BUCKET_COUNT
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
# Raw move mechanics (dex-derived), populated on move action tokens.
NUMERIC_BASE_POWER = 12  # normalized base power (bp/200, clamped)
NUMERIC_PRIORITY = 13  # move priority bracket (normalized)
NUMERIC_ACCURACY = 14  # accuracy [0,1]; 1.0 for never-miss
# Phase 2 — dynamic decision-critical state.
NUMERIC_LEVEL = 15  # per pokemon/switch token: level/100
# Species base stats (dex-derived, public, consistent scale stat/200) on every pokemon/switch
# token. With NUMERIC_LEVEL the model can reason about damage and turn order (speed).
NUMERIC_BASE_HP = 16
NUMERIC_BASE_ATK = 17
NUMERIC_BASE_DEF = 18
NUMERIC_BASE_SPA = 19
NUMERIC_BASE_SPD = 20
NUMERIC_BASE_SPE = 21
# Field token (global), player-relative: hazard layers + screen counts.
NUMERIC_SELF_HAZARDS = 22  # self-side entry-hazard layers (e.g. spikes) / 3
NUMERIC_OPP_HAZARDS = 23
NUMERIC_SELF_SCREENS = 24  # self-side screens active (reflect/lightscreen) / 2
NUMERIC_OPP_SCREENS = 25
# Current stat-boost stages (stage/6 in [-1, 1]) on the ACTIVE mon — the setup-sweep signal.
# Populated only on the active self/opponent pokemon token (boosts reset on switch).
NUMERIC_BOOST_ATK = 26
NUMERIC_BOOST_DEF = 27
NUMERIC_BOOST_SPA = 28
NUMERIC_BOOST_SPD = 29
NUMERIC_BOOST_SPE = 30
# Weather is encoded categorically on the field token's SECONDARY slot (weather:<id>).
# Per-move dynamic/mechanical facts on move action tokens (raw, not judgments).
NUMERIC_MOVE_PP_FRACTION = 31  # remaining PP / max PP from the request (1.0 = full; low = scarce)
NUMERIC_EFFECT_CHANCE = 32  # move-effect probability [0,1]; pairs with move_effect (1.0 = guaranteed)
NUMERIC_TURN_COUNT = 33  # field token: battle turn number / 1000 (clamped) — tempo / stall signal
# Move tokens: fraction of user max HP the move spends upfront (Belly Drum 0.5, Substitute 0.25,
# Explosion 1.0) — a deterrent the model weighs against the effect.
NUMERIC_SELF_HP_COST = 34
# Field token: a pending delayed attack (Future Sight / Doom Desire) landing on each side, as
# turns-remaining / 2. SELF = incoming to the player (a hit to brace/switch around); OPP = the
# player's own outgoing attack landing on the foe.
NUMERIC_SELF_FUTURE_SIGHT = 35
NUMERIC_OPP_FUTURE_SIGHT = 36
# Active mon token: badly-poisoned (tox) ramp stage / 15 — the escalating 1/16, 2/16, ... damage
# (0 if not badly poisoned). Distinct from the status:tox categorical, which only marks the type.
NUMERIC_TOXIC_STAGE = 37
# Actual computed stats (stat / 714, the Gen 3 max, so nothing saturates) on every self mon +
# switch token — free, exact knowledge from the request (EVs/nature/IVs baked in), unlike the
# species BASE stats which are all the model gets for the opponent. Left padding (0) for opponent
# mons, whose actual stats are hidden. HP is the actual max HP (from the request condition).
NUMERIC_ACTUAL_HP = 38
NUMERIC_ACTUAL_ATK = 39
NUMERIC_ACTUAL_DEF = 40
NUMERIC_ACTUAL_SPA = 41
NUMERIC_ACTUAL_SPD = 42
NUMERIC_ACTUAL_SPE = 43

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
    # Actual computed stats {hp, atk, def, spa, spd, spe} — known only for the player's own team
    # (from the request); None for opponent mons, whose actual stats are hidden.
    stats: Optional[Mapping[str, int]] = None


@dataclass(frozen=True)
class ShowdownReplayState:
    battle_id: str
    players: Mapping[str, str]
    requests: Mapping[str, Mapping[str, Any]]
    public_active: Mapping[str, ShowdownPokemon]
    public_revealed: Mapping[str, tuple[ShowdownPokemon, ...]]
    side_conditions: Mapping[str, tuple[str, ...]]
    side_condition_counts: Mapping[str, Mapping[str, int]]
    boosts: Mapping[str, Mapping[str, int]]
    volatiles: Mapping[str, tuple[str, ...]]
    future_sight: Mapping[str, int]
    toxic_stage: Mapping[str, int]
    public_events: tuple["ShowdownPublicEvent", ...]
    public_lines: tuple[str, ...]
    weather: Optional[str] = None
    turn_number: int = 0
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
    self_active_boosts: Mapping[str, int]
    opponent_active_boosts: Mapping[str, int]
    self_active_volatiles: tuple[str, ...]
    opponent_active_volatiles: tuple[str, ...]
    self_toxic_stage: int
    opponent_toxic_stage: int
    belief_view: PlayerBeliefView
    legal_action_mask: tuple[bool, ...]
    recent_events: tuple[PlayerRelativePublicEvent, ...]
    recent_public_events: tuple[str, ...]
    weather: Optional[str] = None
    turn_number: int = 0
    self_future_sight_turns: int = 0  # turns until a delayed attack lands on the player's side
    opponent_future_sight_turns: int = 0  # turns until the player's own delayed attack lands
    winner: Optional[str] = None

    @property
    def self_active(self) -> ShowdownPokemon | None:
        return next((pokemon for pokemon in self.self_team if pokemon.active), None)

    @property
    def opponent_active(self) -> ShowdownPokemon | None:
        return next((pokemon for pokemon in self.opponent_team if pokemon.active), None)


class _ReplayParser:
    """Incremental fold of Showdown protocol lines into transport-level replay state.

    ``parse_showdown_replay`` is a thin batch wrapper around this. The local sim env keeps a
    persistent instance and ``feed()``s only newly-arrived lines, so each line is parsed once
    (O(n) per game) instead of the whole accumulated log being re-parsed on every observation
    (O(n^2)). ``snapshot()`` returns an immutable :class:`ShowdownReplayState` and copies the
    mutable accumulators, so a snapshot is unaffected by later ``feed()`` calls.
    """

    def __init__(self, battle_id: str = "replay") -> None:
        self.battle_id = battle_id
        self.players: dict[str, str] = {}
        self.requests: dict[str, Mapping[str, Any]] = {}
        self.public_active: dict[str, ShowdownPokemon] = {}
        self.public_revealed: dict[str, list[ShowdownPokemon]] = {}
        self.side_condition_counts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.boosts: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.volatiles: dict[str, set[str]] = {"p1": set(), "p2": set()}
        self.future_sight: dict[str, int] = {}
        self.toxic_stage: dict[str, int] = {"p1": 0, "p2": 0}
        self.pending_baton_pass: set[str] = set()
        self.public_events: list[ShowdownPublicEvent] = []
        self.public_lines: list[str] = []
        self.weather: Optional[str] = None
        self.turn_number: int = 0
        self.winner: Optional[str] = None

    def feed(self, lines: Sequence[str]) -> None:
        for raw_line in lines:
            self._feed_line(raw_line)

    def _feed_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        if line.startswith(">"):
            return
        parts = line.split("|")
        event_type = parts[1] if len(parts) > 1 else ""
        # BattleStream emits wall-clock timestamp lines (``|t:|...``). They are useful for raw
        # protocol debugging but are not battle state and would make replay-from-root observations
        # differ across otherwise identical deterministic simulations.
        if event_type == "t:":
            return
        if event_type == "player" and len(parts) >= 4:
            showdown_slot = parts[2]
            if showdown_slot in {"p1", "p2"}:
                self.players[showdown_slot] = parts[3]
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "request" and len(parts) >= 3:
            payload = _decode_request_payload(line)
            side = payload.get("side") if isinstance(payload.get("side"), Mapping) else {}
            showdown_slot = side.get("id") if isinstance(side, Mapping) else None
            if showdown_slot in {"p1", "p2"}:
                self.requests[showdown_slot] = payload
            return
        if event_type in {"switch", "drag", "replace"} and len(parts) >= 4:
            pokemon = _pokemon_from_public_line(parts)
            if pokemon is not None:
                self.public_active[pokemon.showdown_slot] = pokemon
                _record_public_reveal(self.public_revealed, pokemon)
                # A new mon takes the slot with fresh (zero) stat-boost stages — UNLESS it came
                # in via Baton Pass, which carries the passer's boosts to the incoming mon. Only
                # a true |switch| can be a Baton Pass; a |drag| (Roar/Whirlwind) never is. We
                # detect it from the preceding |move|...|Baton Pass (the flag) or a "[from] Baton
                # Pass" tag on the switch line itself.
                is_baton_pass = event_type == "switch" and (
                    pokemon.showdown_slot in self.pending_baton_pass or _line_mentions_baton_pass(parts)
                )
                self.pending_baton_pass.discard(pokemon.showdown_slot)
                if not is_baton_pass:
                    self.boosts[pokemon.showdown_slot] = {}
                # Volatile statuses are tied to the mon on the field, so a new mon clears them
                # (Baton Pass passes some volatiles, but conservatively resetting is the simple,
                # rarely-wrong choice and keeps the volatile set honest about the current mon).
                self.volatiles[pokemon.showdown_slot] = set()
                # Gen 3 resets the toxic counter when a mon leaves the field.
                self.toxic_stage[pokemon.showdown_slot] = 0
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "win" and len(parts) >= 3:
            self.winner = parts[2]
            self.public_events.append(_public_event_from_line(line))
            self.public_lines.append(line)
            return
        if event_type == "turn" and len(parts) >= 3:
            try:
                self.turn_number = int(parts[2])
            except (TypeError, ValueError):
                pass
            # Each turn a badly-poisoned mon stays in, its toxic damage escalates (1/16, 2/16, ...).
            for slot, stage in self.toxic_stage.items():
                if stage:
                    self.toxic_stage[slot] = min(15, stage + 1)
        _update_side_conditions(parts, self.side_condition_counts)
        self.weather = _update_weather(parts, self.weather)
        _update_boosts(parts, self.boosts)
        _update_volatiles(parts, self.volatiles)
        _update_future_sight(parts, self.future_sight, self.turn_number)
        _update_toxic_stage(parts, self.toxic_stage)
        _flag_baton_pass(parts, self.pending_baton_pass)
        self.public_events.append(_public_event_from_line(line))
        self.public_lines.append(line)

    def snapshot(self) -> ShowdownReplayState:
        return ShowdownReplayState(
            battle_id=self.battle_id,
            players=dict(self.players),
            requests=dict(self.requests),
            public_active=dict(self.public_active),
            public_revealed={slot: tuple(pokemon) for slot, pokemon in self.public_revealed.items()},
            side_conditions={slot: tuple(sorted(conditions)) for slot, conditions in _side_conditions_from_counts(self.side_condition_counts).items()},
            side_condition_counts={
                slot: dict(sorted(conditions.items()))
                for slot, conditions in self.side_condition_counts.items()
            },
            boosts={slot: dict(sorted(stages.items())) for slot, stages in self.boosts.items()},
            volatiles={slot: tuple(sorted(names)) for slot, names in self.volatiles.items()},
            future_sight=dict(self.future_sight),
            toxic_stage=dict(self.toxic_stage),
            public_events=tuple(self.public_events),
            public_lines=tuple(self.public_lines),
            weather=self.weather,
            turn_number=self.turn_number,
            winner=self.winner,
        )


def parse_showdown_replay(lines: Sequence[str], *, battle_id: str = "replay") -> ShowdownReplayState:
    """Parse compact Showdown protocol lines into transport-level state."""
    parser = _ReplayParser(battle_id=battle_id)
    parser.feed(lines)
    return parser.snapshot()


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
    belief_engine: "PublicBattleBeliefEngine | None" = None,
) -> PlayerRelativeBattleState:
    """Build a player-relative state view from raw Showdown transport state.

    ``belief_engine`` lets a caller pass a persistent engine fed incrementally (the local sim
    env), avoiding a from-scratch rebuild from ``replay.public_events`` on every call. When
    omitted, the engine is built batch-style from the replay (unchanged behavior).
    """
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
    if belief_engine is None:
        belief_engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id=format_id,
            set_source=set_source,
        )
        belief_engine.resolve_pending_switches_at_boundary()
        belief_view = belief_engine.snapshot().for_player(showdown_slot)
    else:
        # Persistent engine fed incrementally: resolve+snapshot on a copy so its pending-switch
        # state survives for the next ingested event.
        belief_view = belief_engine.resolved_player_view(showdown_slot)
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
        self_active_boosts=dict(replay.boosts.get(showdown_slot, {})),
        opponent_active_boosts=dict(replay.boosts.get(opponent_slot, {})),
        self_active_volatiles=tuple(replay.volatiles.get(showdown_slot, ())),
        opponent_active_volatiles=tuple(replay.volatiles.get(opponent_slot, ())),
        self_toxic_stage=int(replay.toxic_stage.get(showdown_slot, 0)),
        opponent_toxic_stage=int(replay.toxic_stage.get(opponent_slot, 0)),
        belief_view=belief_view,
        legal_action_mask=_legal_action_mask(request),
        recent_events=recent_events,
        recent_public_events=tuple(event.relative_line or event.raw_line for event in recent_events),
        weather=replay.weather,
        turn_number=replay.turn_number,
        self_future_sight_turns=_future_sight_turns_remaining(replay, showdown_slot),
        opponent_future_sight_turns=_future_sight_turns_remaining(replay, opponent_slot),
        winner=replay.winner,
    )


def observation_from_player_state(
    state: PlayerRelativeBattleState,
    *,
    category_vocab: "CategoryVocabulary",
    spec: ObservationSpec = DEFAULT_REPLAY_OBSERVATION_SPEC,
    dex: "ShowdownDex | None" = None,
) -> PokeZeroObservationV0:
    """Encode normalized replay state into fixed-shape observation rows.

    Categorical slots are encoded as raw token strings and converted to compact embedding rows
    via ``category_vocab`` (required) in a single pass. When ``dex`` is supplied, raw mechanical
    facts (Pokemon types; move type / damage class / base power / priority / accuracy) are
    populated into the type/mechanic feature slots; without it those slots stay padding.
    """
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
        active_boosts=state.self_active_boosts,
        active_volatiles=state.self_active_volatiles,
        active_toxic_stage=state.self_toxic_stage,
        dex=dex,
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
        active_boosts=state.opponent_active_boosts,
        active_volatiles=state.opponent_active_volatiles,
        active_toxic_stage=state.opponent_toxic_stage,
        dex=dex,
    )
    _encode_action_tokens(categorical_ids, numeric_features, state, dex=dex)
    _encode_recent_event_tokens(categorical_ids, numeric_features, state, spec)
    # Convert the raw category strings to compact embedding rows in one pass.
    categorical_rows = [[category_vocab.encode(value) for value in row] for row in categorical_ids]
    token_type_ids = _token_type_ids(spec)
    attention_mask = _attention_mask(state, spec)
    return PokeZeroObservationV0(
        categorical_ids=tuple(tuple(row) for row in categorical_rows),
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


# The volatile statuses we surface (normalized ids). `-start`/`-end` carry many payloads (ability
# procs, type changes, internal markers); we track only this closed, decision-relevant set so every
# emitted volatile:<id> token has an enumerated vocab row (no OOV) and is a genuine status. This is
# the single source of truth — randbat_vocab enumerates volatile:<id> from it.
TRACKED_VOLATILES = frozenset({
    "confusion", "leechseed", "substitute", "taunt", "encore", "disable", "torment", "attract",
    "nightmare", "curse", "ingrain", "foresight", "lockon", "mindreader", "destinybond", "grudge",
    "focusenergy", "charge", "yawn", "stockpile", "bide", "uproar", "imprison", "magiccoat",
    "snatch", "mudsport", "watersport", "defensecurl", "minimize", "rage", "partiallytrapped",
    "perishsong", "perish0", "perish1", "perish2", "perish3", "flashfire",
})


def _update_volatiles(parts: Sequence[str], volatiles: dict[str, set[str]]) -> None:
    """Track active-mon volatile statuses from |-start| / |-end| lines (per Showdown slot).

    Only names in TRACKED_VOLATILES are recorded; other `-start` payloads (ability procs, type
    changes, internal markers) are ignored, so every emitted token has an enumerated vocab row.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type not in {"-start", "-end"} or len(parts) < 4:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in volatiles:
        return
    name = _side_condition_identifier(parts[3])  # strips move:/ability:/item: prefix + normalizes
    if name not in TRACKED_VOLATILES:
        return
    if event_type == "-start":
        volatiles[slot].add(name)
    else:
        volatiles[slot].discard(name)


# Delayed-damage moves (Future Sight / Doom Desire): used on one turn, they land on the target's
# side ~2 turns later. Tracked as a per-side landing turn so the model sees an incoming/outgoing hit.
_FUTURE_MOVES = frozenset({"futuresight", "doomdesire"})
_FUTURE_SIGHT_DELAY = 2


def _update_future_sight(parts: Sequence[str], future_sight: dict[str, int], turn_number: int) -> None:
    """Track pending delayed attacks per side from |-start| (use) / |-end| (land) lines.

    Showdown puts the |-start| on the USER and the |-end| on the side that takes the hit, so a use
    schedules a landing on the user's OPPONENT side; the landing |-end| clears it.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type not in {"-start", "-end"} or len(parts) < 4:
        return
    if _side_condition_identifier(parts[3]) not in _FUTURE_MOVES:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in {"p1", "p2"}:
        return
    if event_type == "-start":
        target_side = "p2" if slot == "p1" else "p1"
        future_sight[target_side] = turn_number + _FUTURE_SIGHT_DELAY
    else:
        future_sight.pop(slot, None)


def _update_toxic_stage(parts: Sequence[str], toxic_stage: dict[str, int]) -> None:
    """Track the badly-poisoned (tox) ramp stage per side from |-status| / |-curestatus| lines.

    A `tox` status starts the counter at 1 (per-turn escalation is applied on |turn|); any cured
    status clears it. The counter is also reset on switch (Gen 3 behavior) in the parse loop.
    """
    event_type = parts[1] if len(parts) > 1 else ""
    if len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in toxic_stage:
        return
    if event_type == "-status" and len(parts) >= 4 and _normalize_identifier(parts[3]) == "tox":
        toxic_stage[slot] = 1
    elif event_type == "-curestatus":
        toxic_stage[slot] = 0


def _future_sight_turns_remaining(replay: "ShowdownReplayState", slot: str) -> int:
    """Turns until a pending delayed attack lands on ``slot``'s side (0 if none/overdue)."""
    landing = replay.future_sight.get(slot)
    if landing is None:
        return 0
    return max(0, landing - replay.turn_number)


def _update_weather(parts: Sequence[str], weather: Optional[str]) -> Optional[str]:
    """Track the active weather from |-weather| lines ('none'/absent clears it)."""
    if (parts[1] if len(parts) > 1 else "") != "-weather":
        return weather
    raw = parts[2].strip() if len(parts) > 2 else ""
    identifier = _normalize_identifier(raw)
    if not identifier or identifier == "none":
        return None
    return identifier


def _flag_baton_pass(parts: Sequence[str], pending_baton_pass: set[str]) -> None:
    """Track whether a side is mid-Baton-Pass so the next switch-in inherits its boosts.

    A |move|...|Baton Pass sets the flag; any *other* move by that side clears a stale flag (so a
    failed/interrupted Baton Pass that never produced a switch can't carry boosts into a later
    unrelated switch). The flag is otherwise consumed by the following switch.
    """
    if (parts[1] if len(parts) > 1 else "") != "move" or len(parts) < 4:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in {"p1", "p2"}:
        return
    if _normalize_identifier(parts[3]) == "batonpass":
        pending_baton_pass.add(slot)
    else:
        pending_baton_pass.discard(slot)


def _line_mentions_baton_pass(parts: Sequence[str]) -> bool:
    """True if a switch line carries a '[from] Baton Pass' tag (trailing protocol fields)."""
    return any("baton pass" in part.lower() for part in parts[4:])


_BOOST_STAGE_LIMIT = 6


def _update_boosts(parts: Sequence[str], boosts: dict[str, dict[str, int]]) -> None:
    """Accumulate per-active-slot stat-boost stages from boost protocol lines."""
    event_type = parts[1] if len(parts) > 1 else ""
    if event_type == "-clearallboost":
        for slot in boosts:
            boosts[slot].clear()
        return
    if event_type == "-copyboost" and len(parts) >= 4:
        # Psych Up: SOURCE (parts[2]) copies the boost stages of TARGET (parts[3]).
        source = _slot_from_ident(parts[2])
        target = _slot_from_ident(parts[3])
        if source in boosts and target in boosts:
            boosts[source] = dict(boosts[target])
        return
    if event_type not in {
        "-boost", "-unboost", "-setboost", "-clearboost",
        "-clearpositiveboost", "-clearnegativeboost", "-restoreboost",
    } or len(parts) < 3:
        return
    slot = _slot_from_ident(parts[2])
    if slot not in boosts:
        return
    stages = boosts[slot]
    if event_type == "-clearboost" or event_type == "-restoreboost":
        stages.clear()
        return
    if event_type == "-clearpositiveboost":
        for stat in [s for s, stage in stages.items() if stage > 0]:
            stages.pop(stat, None)
        return
    if event_type == "-clearnegativeboost":
        for stat in [s for s, stage in stages.items() if stage < 0]:
            stages.pop(stat, None)
        return
    if len(parts) < 5:
        return
    stat = parts[3].strip()
    try:
        amount = int(parts[4])
    except (TypeError, ValueError):
        return
    if event_type == "-setboost":
        new_stage = amount
    elif event_type == "-unboost":
        new_stage = stages.get(stat, 0) - amount
    else:  # -boost
        new_stage = stages.get(stat, 0) + amount
    new_stage = max(-_BOOST_STAGE_LIMIT, min(_BOOST_STAGE_LIMIT, new_stage))
    if new_stage == 0:
        stages.pop(stat, None)
    else:
        stages[stat] = new_stage


def _side_condition_max_layers(condition: str) -> int:
    # Spikes is the only multi-layer side condition in Gen 3 (max 3 layers).
    if condition == "spikes":
        return 3
    return 1


def _side_condition_identifier(raw_condition: str) -> str:
    # Strip the source prefix Showdown attaches to some effects (e.g. "move: Leech Seed",
    # "ability: Flash Fire", "item: ...") so the normalized id is the bare effect name.
    condition = raw_condition.strip()
    if ":" in condition and condition.split(":", 1)[0].strip().lower() in {"move", "ability", "item"}:
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
        condition = str(row.get("condition")) if row.get("condition") is not None else None
        team.append(
            ShowdownPokemon(
                ident=ident,
                showdown_slot=_slot_from_ident(ident) or showdown_slot,
                species=_species_from_request_pokemon(row),
                condition=condition,
                active=bool(row.get("active")),
                details=str(row.get("details")) if row.get("details") is not None else None,
                stats=_actual_stats_from_request_row(row, condition),
            )
        )
    return tuple(team)


def _actual_stats_from_request_row(row: Mapping[str, Any], condition: str | None) -> dict[str, int] | None:
    """The player mon's actual computed stats from a request row: the 5 battle stats plus max HP.

    The request's ``stats`` object holds atk/def/spa/spd/spe; max HP is the denominator of the
    condition (e.g. "250/250"). Returns None when no stats are present (e.g. simplified payloads).
    """
    raw = row.get("stats")
    stats: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key in ("atk", "def", "spa", "spd", "spe"):
            value = raw.get(key)
            if isinstance(value, int):
                stats[key] = value
    max_hp = _max_hp_from_condition(condition)
    if max_hp is not None:
        stats["hp"] = max_hp
    return stats or None


def _max_hp_from_condition(condition: str | None) -> int | None:
    """Max HP (the denominator) from a request condition like '180/250'; None for '0 fnt'/absent."""
    if not condition:
        return None
    head = condition.split()[0]
    if "/" not in head:
        return None
    _, _, denominator = head.partition("/")
    return int(denominator) if denominator.isdigit() and int(denominator) > 0 else None


def _opponent_team_from_public_state(
    replay: ShowdownReplayState,
    opponent_slot: str,
) -> tuple[ShowdownPokemon, ...]:
    return tuple(replay.public_revealed.get(opponent_slot, ()))


def _blank_categorical_rows(spec: ObservationSpec) -> list[list[str]]:
    # Categorical slots hold the raw token *strings* during encoding; observation_from_player_
    # state converts them to compact embedding rows via the CategoryVocabulary in one pass.
    return [[""] * spec.categorical_feature_count for _ in range(spec.token_count)]


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
    if state.weather:
        _set_category(categorical_ids[FIELD_TOKEN_OFFSET], CATEGORY_SECONDARY, f"weather:{state.weather}")
    self_haz, self_scr = _side_condition_features(state.self_side_condition_counts)
    opp_haz, opp_scr = _side_condition_features(state.opponent_side_condition_counts)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SELF_HAZARDS, self_haz)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_OPP_HAZARDS, opp_haz)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SELF_SCREENS, self_scr)
    _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_OPP_SCREENS, opp_scr)
    if state.turn_number:
        _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_TURN_COUNT, min(1.0, state.turn_number / 1000.0))
    if state.self_future_sight_turns:
        _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_SELF_FUTURE_SIGHT, min(1.0, state.self_future_sight_turns / 2.0))
    if state.opponent_future_sight_turns:
        _set_numeric(numeric_features[FIELD_TOKEN_OFFSET], NUMERIC_OPP_FUTURE_SIGHT, min(1.0, state.opponent_future_sight_turns / 2.0))


# Gen 3 has a single entry hazard (Spikes, max 3 layers); Toxic Spikes / Stealth Rock are
# Gen 4+. Screens are Reflect + Light Screen.
_HAZARD_CONDITIONS = ("spikes",)
_SCREEN_CONDITIONS = ("reflect", "lightscreen")
# Boost stats encoded on the active mon, in (Showdown stat key, numeric slot) order.
_BOOST_STAT_SLOTS = (
    ("atk", NUMERIC_BOOST_ATK),
    ("def", NUMERIC_BOOST_DEF),
    ("spa", NUMERIC_BOOST_SPA),
    ("spd", NUMERIC_BOOST_SPD),
    ("spe", NUMERIC_BOOST_SPE),
)


def _side_condition_features(counts: Mapping[str, int]) -> tuple[float, float]:
    """(hazard layers /3, screens active /2) for one side's condition counts."""
    hazards = sum(int(counts.get(name, 0)) for name in _HAZARD_CONDITIONS)
    screens = sum(1 for name in _SCREEN_CONDITIONS if counts.get(name))
    return min(1.0, hazards / 3.0), min(1.0, screens / 2.0)


def _encode_active_boosts(num_row: list[float], boosts: Mapping[str, int] | None) -> None:
    """Set the five stat-boost-stage slots (stage/6, clamped to [-1, 1]) for an active mon."""
    if not boosts:
        return
    for stat_key, slot in _BOOST_STAT_SLOTS:
        stage = boosts.get(stat_key)
        if stage:
            _set_numeric(num_row, slot, max(-1.0, min(1.0, float(stage) / 6.0)))


def _encode_active_volatiles(cat_row: list[str], volatiles: Sequence[str]) -> None:
    """Place active-mon volatile statuses (sorted) positionally into the volatile columns."""
    for index, name in enumerate(sorted(set(volatiles))[:VOLATILE_BUCKET_COUNT]):
        column = CATEGORY_VOLATILE_OFFSET + index
        if column >= len(cat_row):
            break
        cat_row[column] = f"volatile:{_normalize_identifier(name)}"


def _encode_species_type_categories(row: list[int], dex: "ShowdownDex | None", species: str | None) -> None:
    """Set the two type slots for a Pokemon token from the dex (no-op without a dex)."""
    if dex is None or not species:
        return
    info = dex.species_info(species)
    if info is None:
        return
    if len(info.types) >= 1:
        _set_category(row, CATEGORY_TYPE_1, f"type:{info.types[0]}")
    if len(info.types) >= 2:
        _set_category(row, CATEGORY_TYPE_2, f"type:{info.types[1]}")


def _level_from_details(details: str | None) -> int | None:
    """Extract the level from a details string like 'Charizard, L83, M' (None if absent)."""
    if not details:
        return None
    for part in details.split(","):
        token = part.strip()
        if token.startswith("L") and token[1:].isdigit():
            return int(token[1:])
    return None


_BASE_STAT_SLOTS = (
    ("hp", NUMERIC_BASE_HP),
    ("atk", NUMERIC_BASE_ATK),
    ("def", NUMERIC_BASE_DEF),
    ("spa", NUMERIC_BASE_SPA),
    ("spd", NUMERIC_BASE_SPD),
    ("spe", NUMERIC_BASE_SPE),
)


_ACTUAL_STAT_SLOTS = (
    ("hp", NUMERIC_ACTUAL_HP),
    ("atk", NUMERIC_ACTUAL_ATK),
    ("def", NUMERIC_ACTUAL_DEF),
    ("spa", NUMERIC_ACTUAL_SPA),
    ("spd", NUMERIC_ACTUAL_SPD),
    ("spe", NUMERIC_ACTUAL_SPE),
)
# Gen 3 maximum possible stat (Blissey HP at level 100); normalizing by it keeps every actual
# stat in [0, 1] with no saturation.
_ACTUAL_STAT_DIVISOR = 714.0


def _encode_pokemon_stats(
    num_row: list[float], dex: "ShowdownDex | None", species: str | None, details: str | None
) -> None:
    """Set level + species base stats (dex-derived, public) for a pokemon/switch token."""
    level = _level_from_details(details)
    if level is not None:
        _set_numeric(num_row, NUMERIC_LEVEL, min(1.0, level / 100.0))
    if dex is None or not species:
        return
    info = dex.species_info(species)
    if info is None:
        return
    for stat_key, slot in _BASE_STAT_SLOTS:
        value = info.base_stats.get(stat_key)
        if value:
            _set_numeric(num_row, slot, min(1.0, float(value) / 200.0))


def _encode_actual_stats(num_row: list[float], stats: Mapping[str, int] | None) -> None:
    """Set the player mon's actual computed stats (known only for the self team; no-op otherwise)."""
    if not stats:
        return
    for stat_key, slot in _ACTUAL_STAT_SLOTS:
        value = stats.get(stat_key)
        if value:
            _set_numeric(num_row, slot, min(1.0, float(value) / _ACTUAL_STAT_DIVISOR))


def _encode_move_mechanics(
    cat_row: list[int],
    num_row: list[float],
    dex: "ShowdownDex | None",
    move_name: str,
    user_types: Sequence[str] = (),
    user_hp_fraction: float | None = None,
) -> None:
    """Set move type / damage class (categorical) + base power / priority / accuracy + effect.

    ``user_types`` and ``user_hp_fraction`` are the acting (self active) mon's types and current HP
    fraction, used to resolve type-dependent effects (Curse) and HP-variable base power
    (Reversal / Flail / Eruption / Water Spout) at encode time.
    """
    if dex is None:
        return
    move = dex.move_info(move_name)
    if move is None:
        return
    base_power = resolve_move_base_power(move, user_hp_fraction)
    _set_category(cat_row, CATEGORY_TYPE_1, f"type:{move.type}")
    _set_category(cat_row, CATEGORY_MOVE_CATEGORY, f"move_category:{move.gen3_category}")
    _set_category(cat_row, CATEGORY_MOVE_PRIORITY, f"move_priority:{move.priority}")
    _set_numeric(num_row, NUMERIC_BASE_POWER, min(1.0, float(base_power) / 200.0))
    _set_numeric(num_row, NUMERIC_PRIORITY, max(-1.0, min(1.0, float(move.priority) / 5.0)))
    _set_numeric(num_row, NUMERIC_ACCURACY, (float(move.accuracy) / 100.0) if move.accuracy else 1.0)
    effect_label, effect_chance, self_hp_cost = resolve_move_effect(move, user_types)
    if effect_label:
        _set_category(cat_row, CATEGORY_MOVE_EFFECT, f"move_effect:{effect_label}")
    _set_numeric(num_row, NUMERIC_EFFECT_CHANCE, min(1.0, float(effect_chance) / 100.0))
    _set_numeric(num_row, NUMERIC_SELF_HP_COST, max(0.0, min(1.0, float(self_hp_cost))))


def _encode_pokemon_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    offset: int,
    pokemon: Sequence[ShowdownPokemon],
    *,
    role: str,
    limit: int,
    beliefs_by_species: Mapping[str, RevealedPokemonBelief] | None = None,
    active_boosts: Mapping[str, int] | None = None,
    active_volatiles: Sequence[str] = (),
    active_toxic_stage: int = 0,
    dex: "ShowdownDex | None" = None,
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
        _encode_species_type_categories(categorical_ids[token_index], dex, candidate.species)
        _encode_pokemon_stats(numeric_features[token_index], dex, candidate.species, candidate.details)
        _encode_actual_stats(numeric_features[token_index], candidate.stats)
        if candidate.active:
            _encode_active_boosts(numeric_features[token_index], active_boosts)
            _encode_active_volatiles(categorical_ids[token_index], active_volatiles)
            if active_toxic_stage:
                _set_numeric(numeric_features[token_index], NUMERIC_TOXIC_STAGE, min(1.0, active_toxic_stage / 15.0))
        status = belief.status if belief is not None and belief.status is not None else condition.status
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, f"status:{status}")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, f"pokemon:{role}")
        # The party-slot index (self_slot/opponent_slot) is intentionally NOT encoded: team order
        # is arbitrary in random battles, so the index carries no actionable signal, and the
        # token's position in the sequence + token_type already identify which team slot it is.
        # (The SLOT column stays in use on action tokens for move_slot/switch_slot.)
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_ability", ability_feature_values)
        _encode_belief_fact_categories(categorical_ids[token_index], "possible_item", item_feature_values)
        # Moves mirror ability/item: revealed moves are ground truth (protocol-observed, no belief
        # set source required) and must always be encoded; possible_moves from the set source
        # augment them when available. Union with revealed first — the encoder dedups/sorts and
        # truncates to the bucket count (Gen 3 max 14 moves <= 16 buckets, so no reveal is dropped).
        _encode_belief_fact_categories(
            categorical_ids[token_index],
            "possible_move",
            tuple(revealed_moves) + tuple(possible_moves),
        )
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


def _self_active_types(state: PlayerRelativeBattleState, dex: "ShowdownDex | None") -> tuple[str, ...]:
    """Types of the acting (self active) mon, for resolving type-dependent move effects."""
    if dex is None or state.self_active is None:
        return ()
    info = dex.species_info(state.self_active.species)
    return tuple(info.types) if info is not None else ()


def _self_active_hp_fraction(state: PlayerRelativeBattleState) -> float | None:
    """Current HP fraction of the acting mon, for resolving HP-variable base power."""
    if state.self_active is None:
        return None
    return _condition_features(state.self_active.condition).hp_fraction


def _encode_action_tokens(
    categorical_ids: list[list[int]],
    numeric_features: list[list[float]],
    state: PlayerRelativeBattleState,
    *,
    dex: "ShowdownDex | None" = None,
) -> None:
    active_request = _active_request(state.request)
    moves = active_request.get("moves") if isinstance(active_request, Mapping) else None
    # The acting mon's types + HP fraction, to resolve type-dependent effects (Curse) and
    # HP-variable base power (Reversal / Flail / Eruption / Water Spout) on its moves.
    user_types = _self_active_types(state, dex)
    user_hp_fraction = _self_active_hp_fraction(state)
    for move_index in range(MOVE_ACTION_COUNT):
        token_index = ACTION_CANDIDATE_TOKEN_OFFSET + move_index
        move = moves[move_index] if isinstance(moves, list) and move_index < len(moves) else None
        move_name = _request_move_name(move) if isinstance(move, Mapping) else f"slot:{move_index + 1}"
        disabled = bool(move.get("disabled")) if isinstance(move, Mapping) else True
        _set_category(categorical_ids[token_index], CATEGORY_PRIMARY, f"move:{move_name}")
        _set_category(categorical_ids[token_index], CATEGORY_SECONDARY, "action:move")
        _set_category(categorical_ids[token_index], CATEGORY_ROLE, "action")
        _set_category(categorical_ids[token_index], CATEGORY_SLOT, f"move_slot:{move_index + 1}")
        if isinstance(move, Mapping):
            _encode_move_mechanics(
                categorical_ids[token_index], numeric_features[token_index], dex, move_name,
                user_types, user_hp_fraction,
            )
            _set_numeric(numeric_features[token_index], NUMERIC_MOVE_PP_FRACTION, _move_pp_fraction(move))
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
        if pokemon is not None:
            _encode_species_type_categories(categorical_ids[token_index], dex, pokemon.species)
            _encode_pokemon_stats(numeric_features[token_index], dex, pokemon.species, pokemon.details)
            _encode_actual_stats(numeric_features[token_index], pokemon.stats)
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
        "weather": state.weather,
        "turn_number": state.turn_number,
        "self_active_boosts": dict(state.self_active_boosts),
        "opponent_active_boosts": dict(state.opponent_active_boosts),
        "self_active_volatiles": list(state.self_active_volatiles),
        "opponent_active_volatiles": list(state.opponent_active_volatiles),
        "self_future_sight_turns": state.self_future_sight_turns,
        "opponent_future_sight_turns": state.opponent_future_sight_turns,
        "self_toxic_stage": state.self_toxic_stage,
        "opponent_toxic_stage": state.opponent_toxic_stage,
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


def _set_category(row: list[str], index: int, value: str) -> None:
    if index < len(row):
        row[index] = value


def _set_numeric(row: list[float], index: int, value: float) -> None:
    if index < len(row):
        row[index] = float(value)


def _known_or_possible_values(known: str | None, possible: Sequence[str]) -> tuple[str, ...]:
    if known:
        return (known,)
    return _compact_belief_values(possible)


def _encode_belief_fact_categories(row: list[str], fact_kind: str, values: Sequence[str]) -> None:
    offset, bucket_count = _belief_bucket_range(fact_kind)
    # Place the (sorted, deduped) belief values positionally into this fact's columns. The bucket
    # counts are sized to the Gen 3 closed universe's per-species maxima (2 abilities / 5 items /
    # 14 moves), so positional placement is exact and collision-free — no hashing needed. The
    # stored value is the category string, converted to a vocab row later.
    for index, value in enumerate(_compact_belief_values(values, limit=bucket_count)):
        column = offset + index
        if column >= len(row):
            break
        row[column] = f"belief:{fact_kind}:{_normalize_identifier(value)}"


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


def _move_pp_fraction(move: Mapping[str, Any]) -> float:
    """Remaining PP as a fraction of max PP from a request move (1.0 if PP data is absent)."""
    pp = move.get("pp")
    maxpp = move.get("maxpp")
    if isinstance(pp, (int, float)) and isinstance(maxpp, (int, float)) and maxpp:
        return max(0.0, min(1.0, float(pp) / float(maxpp)))
    return 1.0


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
