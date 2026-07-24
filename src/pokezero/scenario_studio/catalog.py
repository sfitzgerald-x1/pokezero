"""Source-backed set catalog and scenario validation for the local studio."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..dex import ShowdownDex, load_showdown_dex_cached, normalize_id
from ..env import BattleStartOverride
from ..randbat import Gen3RandbatSource, Gen3RandbatVariant, load_gen3_randbat_source_cached
from ..showdown_fixture import pack_team
from .domain import (
    EndgameScenario,
    ScenarioMove,
    ScenarioPokemon,
    ScenarioSide,
    ScenarioValidationError,
)


class ScenarioCatalog:
    """The immutable Gen 3 randbats set universe pinned to one Showdown source hash."""

    def __init__(self, *, showdown_root: Path | str) -> None:
        self.showdown_root = Path(showdown_root).expanduser().resolve()
        self.source: Gen3RandbatSource = load_gen3_randbat_source_cached(self.showdown_root)
        self.dex: ShowdownDex = load_showdown_dex_cached(self.showdown_root)
        self._variants = {
            variant.variant_id: variant
            for universe in self.source.universes.values()
            for variant in universe.variants
        }

    @property
    def source_hash(self) -> str:
        return self.source.metadata.source_hash

    def variant(self, variant_id: str) -> Gen3RandbatVariant:
        variant = self._variants.get(variant_id)
        if variant is None:
            raise ScenarioValidationError("does not exist in the pinned randbats source", path="/variant_id")
        return variant

    def pokemon_for_variant(
        self,
        variant_id: str,
        *,
        nature: str = "",
        gender: str | None = None,
        evs: Mapping[str, int] | None = None,
        ivs: Mapping[str, int] | None = None,
    ) -> ScenarioPokemon:
        variant = self.variant(variant_id)
        normalized_evs = dict(evs or {})
        normalized_ivs = dict(ivs or {})
        max_hp = self.max_hp(
            species=variant.species,
            level=variant.level,
            evs=normalized_evs,
            ivs=normalized_ivs,
        )
        moves = tuple(
            ScenarioMove(move_id=move_id, pp=self.max_pp(move_id), max_pp=self.max_pp(move_id))
            for move_id in _materializable_variant_move_ids(variant)
        )
        if not moves or any(move.max_pp <= 0 for move in moves):
            raise ScenarioValidationError(
                "variant contains a move without Gen 3 PP metadata", path="/variant_id"
            )
        return ScenarioPokemon(
            variant_id=variant.variant_id,
            species=variant.species,
            level=variant.level,
            ability=variant.ability,
            item=variant.item,
            moves=moves,
            current_hp=max_hp,
            max_hp=max_hp,
            nature=nature,
            gender=gender,
            evs=normalized_evs,
            ivs=normalized_ivs,
        )

    def max_pp(self, move: str) -> int:
        info = self.dex.move_info(move)
        return info.max_pp if info is not None else 0

    def max_hp(
        self,
        *,
        species: str,
        level: int,
        evs: Mapping[str, int],
        ivs: Mapping[str, int],
    ) -> int:
        info = self.dex.species_info(species)
        if info is None:
            raise ScenarioValidationError("does not exist in the Gen 3 dex", path="/species")
        base_hp = info.base_stats.get("hp")
        if not isinstance(base_hp, int) or base_hp < 1:
            raise ScenarioValidationError("has no usable HP stat", path="/species")
        iv = int(ivs.get("hp", 31))
        ev = int(evs.get("hp", 0))
        if iv < 0 or iv > 31 or ev < 0 or ev > 255:
            raise ScenarioValidationError("has invalid HP IV or EV", path="/evs/hp")
        return ((2 * base_hp + iv + (ev // 4)) * level) // 100 + level + 10

    def side_from_generated_team(self, seed: int, team: Sequence[Mapping[str, Any]]) -> ScenarioSide:
        if len(team) != 6:
            raise ScenarioValidationError("generated Showdown team must contain six Pokemon")
        pokemon = tuple(self.pokemon_from_generated_set(row) for row in team)
        return ScenarioSide(
            construction_mode="generated",
            generated_team_seed=seed,
            active_slot=0,
            pokemon=pokemon,
        )

    def pokemon_from_generated_set(self, payload: Mapping[str, Any]) -> ScenarioPokemon:
        species = str(payload.get("species") or "")
        moves = payload.get("moves")
        ability = str(payload.get("ability") or "")
        item = str(payload.get("item") or "")
        level = payload.get("level")
        if not species or not isinstance(moves, list) or not isinstance(level, int):
            raise ScenarioValidationError("bridge returned malformed generated Pokemon")
        candidates = self.source.universe_for(species)
        if candidates is None:
            raise ScenarioValidationError("generated species is not in the pinned randbats source")
        normalized_moves = {normalize_id(move) for move in moves}
        if not normalized_moves or len(normalized_moves) != len(moves):
            raise ScenarioValidationError("generated Pokemon has missing or duplicate moves")
        matching = [
            variant
            for variant in candidates.variants
            if variant.level == level
            and normalize_id(variant.ability) == normalize_id(ability)
            and normalize_id(variant.item) == normalize_id(item)
            and normalized_moves.issubset({normalize_id(move) for move in variant.moves})
        ]
        if not matching:
            raise ScenarioValidationError(
                f"generated {species} set is absent from the pinned variant catalog"
            )
        # The Showdown generator may cull one of a source set's mutually-exclusive moves (most
        # visibly Unown's Hidden Power choices). The emitted team is still the source of truth for
        # a generated side, so preserve its concrete move list while retaining the narrowest
        # source candidate for provenance. Source-composed sides remain exact-set-only below.
        variant = min(
            matching,
            key=lambda candidate: (
                len(candidate.moves) - len(normalized_moves),
                candidate.variant_id,
            ),
        )
        evs = _int_mapping(payload.get("evs"))
        ivs = _int_mapping(payload.get("ivs"))
        result = self.pokemon_for_variant(
            variant.variant_id,
            nature=str(payload.get("nature") or ""),
            gender=_optional_gender(payload.get("gender")),
            evs=evs,
            ivs=ivs,
        )
        # The generator controls both move selection and ordering. Preserve the emitted list: an
        # all-Hidden-Power source set can collapse to one concrete Hidden Power at generation time.
        reordered = tuple(
            ScenarioMove(
                move_id=normalize_id(generated_move),
                pp=self.max_pp(generated_move),
                max_pp=self.max_pp(generated_move),
            )
            for generated_move in moves
        )
        return replace(result, moves=reordered)

    def payload(self) -> dict[str, Any]:
        species = []
        for universe in sorted(self.source.universes.values(), key=lambda item: item.species.casefold()):
            variants = []
            for variant in universe.variants:
                starter = self.pokemon_for_variant(variant.variant_id)
                variants.append(
                    {
                        **variant.to_summary(),
                        "species": variant.species,
                        "max_hp": starter.max_hp,
                        "moves": [
                            {"id": move.move_id, "name": _display_move_name(move.move_id), "max_pp": move.max_pp}
                            for move in starter.moves
                        ],
                    }
                )
            species.append({"id": normalize_id(universe.species), "name": universe.species, "variants": variants})
        return {
            "format_id": self.source.metadata.format_id,
            "source_hash": self.source_hash,
            "species": species,
        }


def validate_scenario(scenario: EndgameScenario, catalog: ScenarioCatalog) -> EndgameScenario:
    """Fail closed unless a saved scenario remains source-valid and state-consistent."""

    if scenario.randbat_source_hash != catalog.source_hash:
        raise ScenarioValidationError(
            f"source hash {scenario.randbat_source_hash!r} does not match current {catalog.source_hash!r}",
            path="/provenance/randbat_source_hash",
        )
    for side_id in ("p1", "p2"):
        side = scenario.side(side_id)
        path = f"/teams/{side_id}"
        if not 1 <= len(side.pokemon) <= 6:
            raise ScenarioValidationError("must contain one to six Pokemon", path=f"{path}/pokemon")
        if not 0 <= side.active_slot < len(side.pokemon):
            raise ScenarioValidationError("must reference a team slot", path=f"{path}/active_slot")
        if side.construction_mode == "generated":
            if len(side.pokemon) != 6 or side.generated_team_seed is None:
                raise ScenarioValidationError(
                    "generated teams must retain all six Pokemon and their seed", path=path
                )
        elif side.generated_team_seed is not None:
            raise ScenarioValidationError(
                "source-composed teams cannot claim a generated team seed", path=f"{path}/generated_team_seed"
            )
        seen_species: set[str] = set()
        for index, pokemon in enumerate(side.pokemon):
            pokemon_path = f"{path}/pokemon/{index}"
            variant = catalog.variant(pokemon.variant_id)
            if normalize_id(pokemon.species) != normalize_id(variant.species):
                raise ScenarioValidationError("does not match its variant species", path=f"{pokemon_path}/species")
            canonical_species = normalize_id(pokemon.species)
            if canonical_species in seen_species:
                raise ScenarioValidationError("duplicates a species on this side", path=f"{pokemon_path}/species")
            seen_species.add(canonical_species)
            if pokemon.level != variant.level:
                raise ScenarioValidationError("does not match its variant level", path=f"{pokemon_path}/level")
            if normalize_id(pokemon.ability) != normalize_id(variant.ability):
                raise ScenarioValidationError("does not match its variant ability", path=f"{pokemon_path}/ability")
            if normalize_id(pokemon.item) != normalize_id(variant.item):
                raise ScenarioValidationError("does not match its variant item", path=f"{pokemon_path}/item")
            source_move_ids = tuple(normalize_id(move) for move in variant.moves)
            expected_move_ids = _materializable_variant_move_ids(variant)
            actual_move_ids = tuple(move.move_id for move in pokemon.moves)
            if len(set(actual_move_ids)) != len(actual_move_ids):
                raise ScenarioValidationError("contains duplicate moves", path=f"{pokemon_path}/moves")
            if side.construction_mode == "generated":
                if not actual_move_ids or not set(actual_move_ids).issubset(source_move_ids):
                    raise ScenarioValidationError(
                        "does not match a source-backed generated move subset",
                        path=f"{pokemon_path}/moves",
                    )
            elif len(actual_move_ids) != len(expected_move_ids) or set(actual_move_ids) != set(expected_move_ids):
                raise ScenarioValidationError("does not match its variant moves", path=f"{pokemon_path}/moves")
            expected_max_hp = catalog.max_hp(
                species=pokemon.species,
                level=pokemon.level,
                evs=pokemon.evs,
                ivs=pokemon.ivs,
            )
            if pokemon.max_hp != expected_max_hp:
                raise ScenarioValidationError(
                    f"must equal derived max HP {expected_max_hp}", path=f"{pokemon_path}/max_hp"
                )
            if not 0 <= pokemon.current_hp <= pokemon.max_hp:
                raise ScenarioValidationError("must be between zero and max HP", path=f"{pokemon_path}/current_hp")
            for move in pokemon.moves:
                expected_max_pp = catalog.max_pp(move.move_id)
                if move.max_pp != expected_max_pp:
                    raise ScenarioValidationError(
                        f"must equal derived max PP {expected_max_pp}", path=f"{pokemon_path}/moves"
                    )
                if not 0 <= move.pp <= move.max_pp:
                    raise ScenarioValidationError("must be between zero and max PP", path=f"{pokemon_path}/moves")
        active = side.pokemon[side.active_slot]
        if active.current_hp == 0:
            raise ScenarioValidationError("must select a living active Pokemon", path=f"{path}/active_slot")
        if not any(pokemon.current_hp > 0 for pokemon in side.pokemon):
            raise ScenarioValidationError("must have one living Pokemon", path=f"{path}/pokemon")
    return scenario


def scenario_start_override(scenario: EndgameScenario) -> BattleStartOverride:
    return BattleStartOverride(
        player_teams={
            side_id: pack_team(tuple(pokemon.to_fixture() for pokemon in scenario.side(side_id).pokemon))
            for side_id in ("p1", "p2")
        }
    )


def scenario_bridge_patch(scenario: EndgameScenario) -> dict[str, Any]:
    return {
        "sides": {
            side_id: {
                "activeSlot": scenario.side(side_id).active_slot,
                "pokemon": [
                    {
                        "slot": index,
                        "hp": pokemon.current_hp,
                        "moves": [{"id": move.move_id, "pp": move.pp} for move in pokemon.moves],
                    }
                    for index, pokemon in enumerate(scenario.side(side_id).pokemon)
                ],
            }
            for side_id in ("p1", "p2")
        }
    }


def _materializable_variant_move_ids(variant: Gen3RandbatVariant) -> tuple[str, ...]:
    """Return the concrete move slots Showdown can construct from a source variant.

    Gen 3's random-team generator culls mutually exclusive Hidden Power choices before
    building its move slots. The source catalog keeps the original choice pool, whereas a
    Custom Game team has only one actual ``hiddenpower`` slot. Manual composition therefore
    uses the first deterministic choice; generated sides preserve their concrete generated
    choice in ``pokemon_from_generated_set``.
    """

    move_ids = tuple(normalize_id(move) for move in variant.moves)
    hidden_power_ids = tuple(move_id for move_id in move_ids if move_id.startswith("hiddenpower"))
    if len(hidden_power_ids) <= 1:
        return move_ids
    return tuple(move_id for move_id in move_ids if not move_id.startswith("hiddenpower")) + (
        hidden_power_ids[0],
    )


def _display_move_name(move_id: str) -> str:
    if move_id.startswith("hiddenpower") and move_id != "hiddenpower":
        return f"Hidden Power {move_id.removeprefix('hiddenpower').title()}"
    return move_id.replace("-", " ").title()


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if not isinstance(item, bool) and isinstance(item, int)
    }


def _optional_gender(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value in {"M", "F", "N"} else None
