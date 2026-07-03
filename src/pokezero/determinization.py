"""Belief-backed start-state materialization for hidden-information search."""

from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from .belief import (
    BeliefEvidence,
    PlayerBeliefView,
    RevealedPokemonBelief,
    sample_opponent_determinizations,
)
from .env import BattleStartOverride
from .policy import PolicyContext
from .randbat import Gen3RandbatSource, Gen3RandbatVariant
from .search import StartOverrideSource
from .search_policy import OpponentActionScenario, StartOverridePlanner
from .showdown_fixture import FixturePokemon, pack_team


DEFAULT_RANDBAT_TEAM_SIZE = 6


def gen3_randbat_belief_start_override_planner(
    set_source: Gen3RandbatSource,
    *,
    team_size: int = DEFAULT_RANDBAT_TEAM_SIZE,
) -> StartOverridePlanner:
    """Create a root-PUCT start-override planner from player-relative public belief.

    The returned planner is hidden-info safe: it reads the acting player's observation metadata,
    which contains the player's own request-known team plus public belief about the opponent. It
    does not inspect the opponent's private observation or legal-action mask. Each branch visit can
    call the returned source to sample a fresh complete world.
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

        def sample_override() -> BattleStartOverride | None:
            return gen3_randbat_belief_start_override(
                context=context,
                set_source=set_source,
                rng=rng,
                team_size=team_size,
            )

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
    self_team = _self_team_from_metadata(metadata.get("self_team"), team_size=team_size)
    if self_team is None:
        return None
    opponent_team = _opponent_team_from_belief(
        view,
        set_source=set_source,
        format_id=context.format_id,
        rng=rng,
        team_size=team_size,
    )
    if opponent_team is None:
        return None
    if view.self_slot not in {"p1", "p2"} or view.opponent_slot not in {"p1", "p2"}:
        return None
    return BattleStartOverride(
        player_teams={
            view.self_slot: pack_team(self_team),
            view.opponent_slot: pack_team(opponent_team),
        }
    )


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


def _self_team_from_metadata(payload: Any, *, team_size: int) -> tuple[FixturePokemon, ...] | None:
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
        team.append(
            FixturePokemon(
                species=species,
                moves=moves,
                ability=_optional_text(row.get("ability")),
                item=_optional_text(row.get("item")),
                level=_level_from_details(_optional_text(row.get("details"))) or 100,
            )
        )
    return tuple(team)


def _opponent_team_from_belief(
    view: PlayerBeliefView,
    *,
    set_source: Gen3RandbatSource,
    format_id: str,
    rng: random.Random,
    team_size: int,
) -> tuple[FixturePokemon, ...] | None:
    sampled = sample_opponent_determinizations(view, sample_count=1, rng=rng)[0]
    team: list[FixturePokemon] = []
    used_species: set[str] = set()
    for belief, pokemon in zip(view.opponent_pokemon, sampled.opponent_pokemon, strict=True):
        fixture = (
            _fixture_from_determinized(pokemon)
            if pokemon.resolved
            else _sample_revealed_opponent_fixture(
                belief,
                set_source=set_source,
                format_id=format_id,
                rng=rng,
            )
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
    variant = variants[rng.randrange(len(variants))]
    return _fixture_from_variant_payload(variant, fallback_species=pokemon.species)


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
        team.append(_fixture_from_variant(variant))
    return tuple(team)


def _fixture_from_determinized(pokemon: Any) -> FixturePokemon | None:
    moves = tuple(str(move) for move in getattr(pokemon, "moves", ()) if str(move))
    species = _optional_text(getattr(pokemon, "species", None))
    if species is None or not moves:
        return None
    return FixturePokemon(
        species=species,
        moves=moves,
        ability=_optional_text(getattr(pokemon, "ability", None)),
        item=_optional_text(getattr(pokemon, "item", None)),
        level=int(getattr(pokemon, "level", None) or 100),
    )


def _fixture_from_variant_payload(
    payload: Mapping[str, Any],
    *,
    fallback_species: str,
) -> FixturePokemon | None:
    moves = _moves_from_payload(payload.get("moves"))
    if not moves:
        return None
    level = payload.get("level")
    return FixturePokemon(
        species=_optional_text(payload.get("species")) or fallback_species,
        moves=moves,
        ability=_optional_text(payload.get("ability")),
        item=_optional_text(payload.get("item")),
        level=int(level) if isinstance(level, int) else 100,
    )


def _fixture_from_variant(variant: Gen3RandbatVariant) -> FixturePokemon:
    return FixturePokemon(
        species=variant.species,
        moves=variant.moves,
        ability=variant.ability,
        item=variant.item,
        level=variant.level,
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
