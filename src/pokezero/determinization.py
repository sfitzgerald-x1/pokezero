"""Belief-backed start-state materialization for hidden-information search."""

from __future__ import annotations

from dataclasses import replace
import random
import re
from typing import Any, Mapping, Sequence

from .belief import (
    BeliefEvidence,
    PlayerBeliefView,
    RevealedPokemonBelief,
)
from .env import BattleStartOverride
from .observation import PokeZeroObservationV0
from .policy import PolicyContext
from .randbat import Gen3RandbatSource, Gen3RandbatVariant
from .search import StartOverrideSource
from .search_policy import OpponentActionScenario, StartOverridePlanner
from .showdown_fixture import FixturePokemon, pack_team


DEFAULT_RANDBAT_TEAM_SIZE = 6
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
_PINCH_BERRIES = {"salacberry", "petayaberry", "liechiberry"}
_HIDDEN_POWER_IVS: Mapping[str, Mapping[str, int]] = {
    "bug": {"atk": 30, "def": 30, "spd": 30},
    "dark": {},
    "dragon": {"atk": 30},
    "electric": {"spa": 30},
    "fighting": {"def": 30, "spa": 30, "spd": 30, "spe": 30},
    "fire": {"atk": 30, "spa": 30, "spe": 30},
    "flying": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30},
    "ghost": {"def": 30, "spd": 30},
    "grass": {"atk": 30, "spa": 30},
    "ground": {"spa": 30, "spd": 30},
    "ice": {"atk": 30, "def": 30},
    "poison": {"def": 30, "spa": 30, "spd": 30},
    "psychic": {"atk": 30, "spe": 30},
    "rock": {"def": 30, "spd": 30, "spe": 30},
    "steel": {"spd": 30},
    "water": {"atk": 30, "def": 30, "spa": 30},
}


def gen3_randbat_belief_start_override_planner(
    set_source: Gen3RandbatSource,
    *,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
) -> StartOverridePlanner:
    """Create a root-PUCT start-override planner from player-relative public belief.

    The returned planner is hidden-info safe: it reads the acting player's observation metadata,
    which contains the player's own request-known team plus public belief about the opponent. It
    does not inspect the opponent's private observation or legal-action mask. Each scenario gets one
    sampled world so all candidate root actions are scored against the same materialized battle.
    """

    if team_size <= 0:
        raise ValueError("team_size must be positive.")

    def planner(
        context: PolicyContext,
        scenario: OpponentActionScenario,
        scenario_index: int,
        rng: random.Random,
    ) -> StartOverrideSource:
        del scenario, scenario_index
        if not set_source.supports(context.format_id):
            return None
        sampled_override = gen3_randbat_belief_start_override(
            context=context,
            set_source=set_source,
            rng=rng,
            team_size=team_size,
        )
        if sampled_override is None:
            return None

        def sample_override() -> BattleStartOverride | None:
            return sampled_override

        sample_override.start_override_id = "gen3-randbat-belief"  # type: ignore[attr-defined]
        return sample_override

    planner.planner_id = "gen3-randbat-belief"  # type: ignore[attr-defined]
    return planner


def gen3_randbat_belief_start_override(
    *,
    context: PolicyContext,
    set_source: Gen3RandbatSource,
    rng: random.Random,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
) -> BattleStartOverride | None:
    """Sample one complete custom-game start override from the acting player's belief.

    Returns ``None`` when the current observation lacks enough request-known self-team data to build
    a faithful packed team. Replay consistency checks then keep bad sampled worlds from being scored.
    """

    if team_size <= 0:
        raise ValueError("team_size must be positive.")
    if not set_source.supports(context.format_id):
        return None
    metadata = context.observation.metadata
    if not isinstance(metadata, Mapping):
        return None
    view = player_belief_view_from_payload(metadata.get("belief_view"))
    if view is None:
        return None
    self_team = _self_team_from_metadata(
        _root_self_team_payload(context, team_size=team_size) or metadata.get("self_team"),
        team_size=team_size,
        set_source=set_source,
    )
    if self_team is None:
        return None
    opponent_team = _opponent_team_from_belief(
        view,
        set_source=set_source,
        format_id=context.format_id,
        rng=rng,
        team_size=team_size,
        move_slot_constraints=_public_opponent_move_slot_constraints(context, view.opponent_slot),
    )
    if opponent_team is None:
        return None
    if view.self_slot not in {"p1", "p2"} or view.opponent_slot not in {"p1", "p2"}:
        return None
    return BattleStartOverride(
        player_teams={
            view.self_slot: pack_team(self_team),
            view.opponent_slot: pack_team(opponent_team),
        },
        observation_format_id=context.format_id,
    )


def _root_self_team_payload(context: PolicyContext, *, team_size: int) -> Any:
    """Return the earliest request-known self-team snapshot for root replay materialization."""

    for step in context.trajectory.steps:
        if step.player_id != context.player_id:
            continue
        metadata = step.observation.metadata
        if not isinstance(metadata, Mapping):
            continue
        rows = _as_sequence(metadata.get("self_team"))
        if len(rows) == team_size:
            return rows
    return None


def _public_opponent_move_slot_constraints(
    context: PolicyContext,
    opponent_slot: str,
) -> dict[str, dict[int, str]]:
    """Map public opponent moves back to recorded move slots for replay-prefix fidelity.

    Prefix replay must resubmit historic opponent choices by move slot. The slot index is part of the
    recorded trajectory, but the move name is taken only from the acting player's public event log.
    This avoids reading opponent-private request moves while still forcing already-revealed moves into
    the slots needed to reproduce the public history.
    """

    own_observations = _own_observations_by_decision_round(context)
    active_species_by_turn = {
        turn_index: species
        for turn_index, observation in own_observations.items()
        if (species := _public_opponent_active_species(observation)) is not None
    }
    constraints: dict[str, dict[int, str]] = {}
    for step in context.trajectory.steps:
        if step.player_id != opponent_slot or not 0 <= step.action_index < 4:
            continue
        species = active_species_by_turn.get(step.turn_index)
        if species is None:
            continue
        move = _public_move_after_decision_round(
            own_observations,
            opponent_slot=opponent_slot,
            self_slot=context.player_id,
            species=species,
            turn_index=step.turn_index,
        )
        if move is None:
            continue
        species_key = _normalize_id(species)
        slot_constraints = constraints.setdefault(species_key, {})
        existing = slot_constraints.get(step.action_index)
        if existing is not None and _normalize_id(existing) != _normalize_id(move):
            continue
        slot_constraints[step.action_index] = move
    return constraints


def _own_observations_by_decision_round(context: PolicyContext) -> dict[int, PokeZeroObservationV0]:
    observations: dict[int, PokeZeroObservationV0] = {
        context.decision_round_index: context.observation,
    }
    for step in context.trajectory.steps:
        if step.player_id == context.player_id:
            observations.setdefault(step.turn_index, step.observation)
    return observations


def _public_opponent_active_species(observation: PokeZeroObservationV0) -> str | None:
    metadata = observation.metadata
    if not isinstance(metadata, Mapping):
        return None
    active = metadata.get("opponent_active")
    if not isinstance(active, Mapping):
        return None
    return _optional_text(active.get("species"))


def _public_move_after_decision_round(
    observations_by_turn: Mapping[int, PokeZeroObservationV0],
    *,
    opponent_slot: str,
    self_slot: str,
    species: str,
    turn_index: int,
) -> str | None:
    next_observation = observations_by_turn.get(turn_index + 1)
    if next_observation is None:
        return None
    # The public-event window is rolling, so the same active mon's older move can still be present.
    # Walk backward through the next decision observation and take the newest matching move line.
    for line in reversed(_recent_public_events(next_observation)):
        move = _move_from_public_event_line(
            line,
            opponent_slot=opponent_slot,
            self_slot=self_slot,
            species=species,
        )
        if move is not None:
            return move
    return None


def _recent_public_events(observation: PokeZeroObservationV0) -> tuple[str, ...]:
    metadata = observation.metadata
    if not isinstance(metadata, Mapping):
        return ()
    return tuple(str(line) for line in _as_sequence(metadata.get("recent_public_events")))


def _move_from_public_event_line(
    line: str,
    *,
    opponent_slot: str,
    self_slot: str,
    species: str,
) -> str | None:
    parts = str(line).split("|")
    if len(parts) < 4 or parts[1] != "move":
        return None
    if _called_move_line(parts):
        return None
    actor = parts[2]
    if not _public_actor_matches_slot(actor, slot=opponent_slot, self_slot=self_slot):
        return None
    actor_species = _species_from_public_actor(actor)
    if actor_species is not None and _normalize_id(actor_species) != _normalize_id(species):
        return None
    return _optional_text(parts[3])


def _called_move_line(parts: Sequence[str]) -> bool:
    for token in parts[4:]:
        text = str(token).strip()
        if not text.startswith("[from]"):
            continue
        normalized = _normalize_id(text)
        if "lockedmove" in normalized:
            continue
        return True
    return False


def _public_actor_matches_slot(actor: str, *, slot: str, self_slot: str) -> bool:
    normalized = str(actor).strip().lower()
    if normalized.startswith(slot.lower()):
        return True
    if normalized.startswith("opponent"):
        return slot != self_slot
    if normalized.startswith("self"):
        return slot == self_slot
    return False


def _species_from_public_actor(actor: str) -> str | None:
    if ":" not in actor:
        return None
    return _optional_text(actor.split(":", 1)[1])


def player_belief_view_from_payload(payload: Any) -> PlayerBeliefView | None:
    if not isinstance(payload, Mapping):
        return None
    self_slot = _optional_slot(payload.get("self_slot"))
    opponent_slot = _optional_slot(payload.get("opponent_slot"))
    if self_slot is None or opponent_slot is None or self_slot == opponent_slot:
        return None
    return PlayerBeliefView(
        self_slot=self_slot,
        opponent_slot=opponent_slot,
        self_pokemon=tuple(
            pokemon
            for raw in _as_sequence(payload.get("self_pokemon"))
            if (pokemon := _revealed_pokemon_from_payload(raw)) is not None
        ),
        opponent_pokemon=tuple(
            pokemon
            for raw in _as_sequence(payload.get("opponent_pokemon"))
            if (pokemon := _revealed_pokemon_from_payload(raw)) is not None
        ),
    )


def _self_team_from_metadata(
    payload: Any,
    *,
    team_size: int,
    set_source: Gen3RandbatSource,
) -> tuple[FixturePokemon, ...] | None:
    rows = _as_sequence(payload)
    if len(rows) != team_size:
        return None
    team: list[FixturePokemon] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        species = _optional_text(row.get("species"))
        moves = _moves_from_payload(row.get("moves"))
        if species is None or not moves:
            return None
        level = _level_from_details(_optional_text(row.get("details"))) or 100
        spread = _gen3_randbat_fixture_spread(
            row,
            species=species,
            moves=moves,
            item=_optional_text(row.get("item")),
            level=level,
            set_source=set_source,
        )
        if spread is None:
            return None
        team.append(
            FixturePokemon(
                species=species,
                moves=moves,
                ability=_optional_text(row.get("ability")),
                item=_optional_text(row.get("item")),
                level=level,
                evs=spread["evs"],
                ivs=spread["ivs"],
            )
        )
    return tuple(team)


def _gen3_randbat_fixture_spread(
    row: Mapping[str, Any],
    *,
    species: str,
    moves: tuple[str, ...],
    item: str | None,
    level: int,
    set_source: Gen3RandbatSource,
) -> dict[str, Mapping[str, int]] | None:
    """Mirror Showdown's Gen 3 randbat EV/IV recipe for request-known self Pokemon.

    If the request includes actual stats, fail closed unless the reconstructed spread reproduces
    them exactly. Simplified fixture rows without stats keep the default randbat spread.
    """

    evs = {stat: 85 for stat in _STAT_ORDER}
    ivs = {stat: 31 for stat in _STAT_ORDER}
    normalized_moves = tuple(_normalize_id(move) for move in moves)
    hidden_power_type = _hidden_power_type(normalized_moves)
    if hidden_power_type is not None:
        for stat, value in _HIDDEN_POWER_IVS.get(hidden_power_type, {}).items():
            ivs[stat] = value

    base_stats = _base_stats_for_species(set_source, species)
    observed_stats = _stats_from_payload(row.get("stats"))
    if observed_stats and base_stats is None:
        return None

    if base_stats is not None:
        _adjust_hp_evs(
            evs=evs,
            ivs=ivs,
            base_hp=base_stats["hp"],
            level=level,
            moves=normalized_moves,
            item=item,
        )

    if _should_minimize_confusion_damage(normalized_moves, set_source.move_metadata):
        evs["atk"] = 0
        ivs["atk"] = (ivs["atk"] or 31) - 28 if hidden_power_type is not None else 0

    if base_stats is not None:
        _adjust_post_attack_hp_evs(
            evs=evs,
            ivs=ivs,
            base_hp=base_stats["hp"],
            level=level,
            moves=normalized_moves,
            item=item,
        )

    if observed_stats:
        computed = _computed_stats(base_stats, evs=evs, ivs=ivs, level=level) if base_stats is not None else None
        if computed is None or any(computed.get(stat) != value for stat, value in observed_stats.items()):
            return None
    return {"evs": evs, "ivs": ivs}


def _opponent_team_from_belief(
    view: PlayerBeliefView,
    *,
    set_source: Gen3RandbatSource,
    format_id: str,
    rng: random.Random,
    team_size: int,
    move_slot_constraints: Mapping[str, Mapping[int, str]] | None = None,
) -> tuple[FixturePokemon, ...] | None:
    team: list[FixturePokemon] = []
    used_species: set[str] = set()
    for belief in view.opponent_pokemon:
        fixture = _sample_revealed_opponent_fixture(
            belief,
            set_source=set_source,
            format_id=format_id,
            rng=rng,
            move_slot_constraints=move_slot_constraints,
        )
        if fixture is None:
            return None
        team.append(fixture)
        used_species.add(_normalize_id(fixture.species))
    hidden_needed = team_size - len(team)
    if hidden_needed < 0:
        return None
    hidden = _sample_hidden_backline(
        set_source,
        used_species=used_species,
        count=hidden_needed,
        rng=rng,
    )
    if hidden is None:
        return None
    return tuple(team + list(hidden))


def _sample_revealed_opponent_fixture(
    pokemon: RevealedPokemonBelief,
    *,
    set_source: Gen3RandbatSource,
    format_id: str,
    rng: random.Random,
    move_slot_constraints: Mapping[str, Mapping[int, str]] | None = None,
) -> FixturePokemon | None:
    variants = pokemon.candidate_variants
    if not variants:
        summary = set_source.summarize(
            format_id=format_id,
            species=pokemon.species,
            revealed_moves=pokemon.revealed_moves,
            revealed_ability=pokemon.revealed_ability,
            revealed_item=pokemon.revealed_item,
            ruled_out_abilities=pokemon.ruled_out_abilities,
        )
        variants = tuple(summary.candidate_variants) if summary is not None else ()
    if not variants:
        return None
    public_max_hp = _condition_max_hp(pokemon.condition)
    if public_max_hp is not None:
        hp_matched = tuple(
            variant
            for variant in variants
            if _variant_public_max_hp(
                variant,
                fallback_species=pokemon.species,
                set_source=set_source,
            )
            == public_max_hp
        )
        if not hp_matched:
            return None
        variants = hp_matched
    species_constraints = (move_slot_constraints or {}).get(_normalize_id(pokemon.species), {})
    if species_constraints:
        slot_matched = tuple(
            variant
            for variant in variants
            if _fixture_from_variant_payload(
                variant,
                fallback_species=pokemon.species,
                set_source=set_source,
                move_slot_constraints=species_constraints,
            )
            is not None
        )
        if not slot_matched:
            return None
        variants = slot_matched
    variant = variants[rng.randrange(len(variants))]
    return _fixture_from_variant_payload(
        variant,
        fallback_species=pokemon.species,
        set_source=set_source,
        move_slot_constraints=species_constraints,
    )


def _condition_max_hp(condition: str | None) -> int | None:
    if not condition:
        return None
    match = re.match(r"^\s*\d+\s*/\s*(\d+)\s*(?:\s|$)", condition)
    if match is None:
        return None
    try:
        max_hp = int(match.group(1))
    except ValueError:
        return None
    # Opponent public conditions are percentages in Showdown's request/public state
    # (e.g. "70/100"). Only request-known absolute HP denominators can safely filter variants.
    if max_hp <= 100:
        return None
    return max_hp


def _variant_public_max_hp(
    payload: Mapping[str, Any],
    *,
    fallback_species: str,
    set_source: Gen3RandbatSource,
) -> int | None:
    fixture = _fixture_from_variant_payload(
        payload,
        fallback_species=fallback_species,
        set_source=set_source,
    )
    if fixture is None:
        return None
    base_stats = _base_stats_for_species(set_source, fixture.species)
    if base_stats is None:
        return None
    ivs = fixture.ivs or {}
    evs = fixture.evs or {}
    return _hp_stat(
        base_hp=base_stats["hp"],
        iv=int(ivs.get("hp", 31)),
        ev=int(evs.get("hp", 0)),
        level=fixture.level,
    )


def _sample_hidden_backline(
    set_source: Gen3RandbatSource,
    *,
    used_species: set[str],
    count: int,
    rng: random.Random,
) -> tuple[FixturePokemon, ...] | None:
    candidates = [
        universe
        for universe in set_source.universes.values()
        if universe.variants and _normalize_id(universe.species) not in used_species
    ]
    team: list[FixturePokemon] = []
    for _ in range(count):
        if not candidates:
            return None
        universe_index = rng.randrange(len(candidates))
        universe = candidates.pop(universe_index)
        variant = universe.variants[rng.randrange(len(universe.variants))]
        used_species.add(_normalize_id(universe.species))
        fixture = _fixture_from_variant(variant, set_source=set_source)
        if fixture is None:
            return None
        team.append(fixture)
    return tuple(team)


def _hidden_power_type(normalized_moves: Sequence[str]) -> str | None:
    for move in normalized_moves:
        if move.startswith("hiddenpower") and len(move) > len("hiddenpower"):
            return move[len("hiddenpower") :]
    return None


def _base_stats_for_species(
    set_source: Gen3RandbatSource,
    species: str,
) -> dict[str, int] | None:
    raw = set_source.species_metadata.get(_normalize_id(species))
    if not isinstance(raw, Mapping):
        return None
    base_stats = raw.get("baseStats")
    if not isinstance(base_stats, Mapping):
        return None
    stats: dict[str, int] = {}
    for stat in _STAT_ORDER:
        value = base_stats.get(stat)
        if not isinstance(value, int):
            return None
        stats[stat] = value
    return stats


def _stats_from_payload(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, Mapping):
        return None
    stats: dict[str, int] = {}
    for stat in _STAT_ORDER:
        value = payload.get(stat)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            stats[stat] = value
    return stats or None


def _adjust_hp_evs(
    *,
    evs: dict[str, int],
    ivs: Mapping[str, int],
    base_hp: int,
    level: int,
    moves: Sequence[str],
    item: str | None,
) -> None:
    while evs["hp"] > 1:
        hp = _hp_stat(base_hp=base_hp, iv=ivs["hp"], ev=evs["hp"], level=level)
        if "substitute" in moves and any(move in moves for move in ("flail", "reversal")):
            if hp % 4 > 0:
                break
        elif "substitute" in moves and _normalize_id(item or "") in _PINCH_BERRIES:
            if hp % 4 == 0:
                break
        elif "bellydrum" in moves:
            if hp % 2 > 0:
                break
        else:
            break
        evs["hp"] -= 4


def _adjust_post_attack_hp_evs(
    *,
    evs: dict[str, int],
    ivs: Mapping[str, int],
    base_hp: int,
    level: int,
    moves: Sequence[str],
    item: str | None,
) -> None:
    hp = _hp_stat(base_hp=base_hp, iv=ivs["hp"], ev=evs["hp"], level=level)
    if "substitute" in moves and any(move in moves for move in ("endeavor", "flail", "reversal")):
        if hp % 4 == 0:
            evs["hp"] -= 4
    elif "substitute" in moves and _normalize_id(item or "") in _PINCH_BERRIES:
        while evs["hp"] > 1 and hp % 4 > 0:
            evs["hp"] -= 4
            hp = _hp_stat(base_hp=base_hp, iv=ivs["hp"], ev=evs["hp"], level=level)


def _should_minimize_confusion_damage(
    normalized_moves: Sequence[str],
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    if "transform" in normalized_moves:
        return False
    for move in normalized_moves:
        metadata = move_metadata.get(_normalize_id(move), {})
        # Showdown's random-team counter treats Seismic Toss/Night Shade/etc. as fixed-damage
        # moves, not physical attacks, when deciding whether to minimize confusion damage.
        if metadata.get("damage") or metadata.get("damageCallback"):
            continue
        if _gen3_move_category(move, move_metadata) == "Physical":
            return False
    return True


def _gen3_move_category(
    move: str,
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    metadata = move_metadata.get(_normalize_id(move), {})
    if str(metadata.get("category") or "") == "Status":
        return "Status"
    normalized_move = _normalize_id(move)
    if normalized_move.startswith("hiddenpower") and len(normalized_move) > len("hiddenpower"):
        raw_type = normalized_move[len("hiddenpower") :]
        move_type = raw_type[:1].upper() + raw_type[1:]
    else:
        move_type = str(metadata.get("type") or "")
    if move_type in {"Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel"}:
        return "Physical"
    if move_type in {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}:
        return "Special"
    return str(metadata.get("category") or "Status")


def _computed_stats(
    base_stats: Mapping[str, int],
    *,
    evs: Mapping[str, int],
    ivs: Mapping[str, int],
    level: int,
) -> dict[str, int]:
    stats = {
        "hp": _hp_stat(base_hp=base_stats["hp"], iv=ivs["hp"], ev=evs["hp"], level=level),
    }
    for stat in ("atk", "def", "spa", "spd", "spe"):
        stats[stat] = _battle_stat(
            base=base_stats[stat],
            iv=ivs[stat],
            ev=evs[stat],
            level=level,
        )
    return stats


def _hp_stat(*, base_hp: int, iv: int, ev: int, level: int) -> int:
    return ((2 * base_hp + iv + ev // 4 + 100) * level) // 100 + 10


def _battle_stat(*, base: int, iv: int, ev: int, level: int) -> int:
    return ((2 * base + iv + ev // 4) * level) // 100 + 5


def _fixture_from_determinized(pokemon: Any, *, set_source: Gen3RandbatSource) -> FixturePokemon | None:
    moves = tuple(str(move) for move in getattr(pokemon, "moves", ()) if str(move))
    species = _optional_text(getattr(pokemon, "species", None))
    if species is None or not moves:
        return None
    level = int(getattr(pokemon, "level", None) or 100)
    item = _optional_text(getattr(pokemon, "item", None))
    spread = _gen3_randbat_fixture_spread(
        {},
        species=species,
        moves=moves,
        item=item,
        level=level,
        set_source=set_source,
    )
    if spread is None:
        return None
    return FixturePokemon(
        species=species,
        moves=moves,
        ability=_optional_text(getattr(pokemon, "ability", None)),
        item=item,
        level=level,
        evs=spread["evs"],
        ivs=spread["ivs"],
    )


def _fixture_from_variant_payload(
    payload: Mapping[str, Any],
    *,
    fallback_species: str,
    set_source: Gen3RandbatSource,
    move_slot_constraints: Mapping[int, str] | None = None,
) -> FixturePokemon | None:
    moves = _moves_from_payload(payload.get("moves"))
    if not moves:
        return None
    species = _optional_text(payload.get("species")) or fallback_species
    level = payload.get("level")
    resolved_level = int(level) if isinstance(level, int) else 100
    item = _optional_text(payload.get("item"))
    spread = _gen3_randbat_fixture_spread(
        {},
        species=species,
        moves=moves,
        item=item,
        level=resolved_level,
        set_source=set_source,
    )
    if spread is None:
        return None
    fixture = FixturePokemon(
        species=species,
        moves=moves,
        ability=_optional_text(payload.get("ability")),
        item=item,
        level=resolved_level,
        evs=spread["evs"],
        ivs=spread["ivs"],
    )
    if move_slot_constraints:
        fixture = _fixture_with_move_slot_constraints(fixture, move_slot_constraints)
    return fixture


def _fixture_with_move_slot_constraints(
    fixture: FixturePokemon,
    move_slot_constraints: Mapping[int, str],
) -> FixturePokemon | None:
    moves = list(fixture.moves)
    if not moves:
        return None
    assigned: list[str | None] = [None] * len(moves)
    used: set[str] = set()
    for slot, move in sorted(move_slot_constraints.items()):
        if slot < 0 or slot >= len(assigned):
            return None
        actual_move = _fixture_move_for_public_constraint(moves, move)
        if actual_move is None:
            return None
        normalized = _normalize_id(actual_move)
        if normalized in used:
            return None
        assigned[slot] = actual_move
        used.add(normalized)
    remaining = [move for move in moves if _normalize_id(move) not in used]
    for index, value in enumerate(assigned):
        if value is None:
            assigned[index] = remaining.pop(0)
    return replace(fixture, moves=tuple(move for move in assigned if move is not None))


def _fixture_move_for_public_constraint(moves: Sequence[str], public_move: str) -> str | None:
    public_id = _normalize_id(public_move)
    # Some request/public labels include the Gen 3 Hidden Power base power, e.g. Hidden Power Ice 70.
    if public_id.endswith("70"):
        public_id = public_id[:-2]
    for move in moves:
        move_id = _normalize_id(move)
        if move_id == public_id:
            return move
        if public_id == "hiddenpower" and move_id.startswith("hiddenpower"):
            return move
    return None


def _fixture_from_variant(
    variant: Gen3RandbatVariant,
    *,
    set_source: Gen3RandbatSource,
) -> FixturePokemon | None:
    spread = _gen3_randbat_fixture_spread(
        {},
        species=variant.species,
        moves=variant.moves,
        item=variant.item,
        level=variant.level,
        set_source=set_source,
    )
    if spread is None:
        return None
    return FixturePokemon(
        species=variant.species,
        moves=variant.moves,
        ability=variant.ability,
        item=variant.item,
        level=variant.level,
        evs=spread["evs"],
        ivs=spread["ivs"],
    )


def _revealed_pokemon_from_payload(payload: Any) -> RevealedPokemonBelief | None:
    if not isinstance(payload, Mapping):
        return None
    showdown_slot = _optional_text(payload.get("showdown_slot"))
    species = _optional_text(payload.get("species"))
    if showdown_slot is None or species is None:
        return None
    return RevealedPokemonBelief(
        showdown_slot=showdown_slot,
        species=species,
        condition=_optional_text(payload.get("condition")),
        status=_optional_text(payload.get("status")),
        active=bool(payload.get("active")),
        revealed_moves=_moves_from_payload(payload.get("revealed_moves")),
        revealed_ability=_optional_text(payload.get("revealed_ability")),
        revealed_item=_optional_text(payload.get("revealed_item")),
        ruled_out_abilities=_moves_from_payload(payload.get("ruled_out_abilities")),
        candidate_set_count=_optional_int(payload.get("candidate_set_count")),
        uncertainty=_optional_float(payload.get("uncertainty"), default=1.0),
        possible_abilities=_moves_from_payload(payload.get("possible_abilities")),
        possible_items=_moves_from_payload(payload.get("possible_items")),
        possible_moves=_moves_from_payload(payload.get("possible_moves")),
        candidate_variants=tuple(
            dict(variant)
            for variant in _as_sequence(payload.get("candidate_variants"))
            if isinstance(variant, Mapping)
        ),
        source_metadata=dict(payload["source_metadata"])
        if isinstance(payload.get("source_metadata"), Mapping)
        else None,
        evidence=tuple(
            BeliefEvidence(
                kind=str(item.get("kind") or ""),
                detail=str(item.get("detail") or ""),
                source_line=_optional_text(item.get("source_line")),
            )
            for item in _as_sequence(payload.get("evidence"))
            if isinstance(item, Mapping)
        ),
        transformed=bool(payload.get("transformed")),
        transform_species=_optional_text(payload.get("transform_species")),
    )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _moves_from_payload(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in _as_sequence(value) if str(item).strip())


def _optional_slot(value: Any) -> str | None:
    text = _optional_text(value)
    return text if text in {"p1", "p2"} else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_float(value: Any, *, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _level_from_details(details: str | None) -> int | None:
    if not details:
        return None
    for part in details.split(","):
        token = part.strip()
        if token.startswith("L") and token[1:].isdigit():
            return int(token[1:])
    return None


def _normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
