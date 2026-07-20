"""Gen 3 random battle set sources and belief-universe helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Iterable, Mapping, Optional, Sequence

from .belief import CandidateSetSummary


GEN3_RANDBAT_FORMATS = {"gen3randombattle", "[Gen 3] Random Battle"}
# v4: relaxed move-combo STAB / setup-incompatibility narrowing to match Showdown's actual
# gen3 generator (see ``_required_stab_types``). The universe CONTENT changed, and the disk
# cache is keyed on this schema (folded into ``_source_hash``), so the bump is REQUIRED — it
# forces a rebuild and stops stale v3 caches (the old over-pruned universe) from shadowing the
# fix in training/collection, where ``load_gen3_randbat_source_cached`` reads ``use_cache=True``.
_SOURCE_CACHE_SCHEMA = "gen3-randbat-source-v4"
_UNOWN_COSMETIC_FORM_SUFFIXES = frozenset("abcdefghijklmnopqrstuvwxyz") | {"exclamation", "question"}
PHYSICAL_TYPES = {"Normal", "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel"}
SPECIAL_TYPES = {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}
STATUS_INFLICTING_MOVES = {"stunspore", "thunderwave", "toxic", "willowisp", "yawn"}
RECOVERY_MOVES = {
    "milkdrink",
    "moonlight",
    "morningsun",
    "recover",
    "slackoff",
    "softboiled",
    "synthesis",
}
SETUP_MOVES = {
    "acidarmor",
    "agility",
    "bellydrum",
    "bulkup",
    "calmmind",
    "curse",
    "dragondance",
    "growth",
    "howl",
    "irondefense",
    "meditate",
    "raindance",
    "sunnyday",
    "swordsdance",
    "tailglow",
}
# Moves that shouldn't count as STAB (gen3 teams.ts NO_STAB). A move of a species' type only
# satisfies/enforces STAB if it is a damaging move NOT on this list — mirrors Showdown's
# ``!this.noStab.includes(moveid) && (move.basePower || move.basePowerCallback)`` gate.
NO_STAB = frozenset(
    {
        "eruption",
        "explosion",
        "fakeout",
        "focuspunch",
        "futuresight",
        "icywind",
        "knockoff",
        "machpunch",
        "pursuit",
        "quickattack",
        "rapidspin",
        "selfdestruct",
        "skyattack",
        "waterspout",
    }
)
# Only these species enforce a Bug STAB in the gen3 generator (moveEnforcementCheckers.Bug in
# teams.ts). Every other Bug-type species (Scyther, Scizor, Venomoth, Beedrill, ...) does NOT
# hard-require a Bug move, which is why the old blanket per-type STAB rule over-pruned them.
BUG_ENFORCE_SPECIES = frozenset({"armaldo", "heracross", "parasect"})
# The exact set of types that have a moveEnforcementChecker in gen3 teams.ts. A species type NOT
# in this set (GRASS and DRAGON are the two gen3 types with no checker) is never STAB-enforced
# through the per-type checker loop.
ENFORCED_STAB_TYPES = frozenset(
    {
        "Bug",
        "Dark",
        "Electric",
        "Fighting",
        "Fire",
        "Flying",
        "Ghost",
        "Ground",
        "Ice",
        "Normal",
        "Poison",
        "Psychic",
        "Rock",
        "Steel",
        "Water",
    }
)
MOVE_PAIRS = (
    ("sleeptalk", "rest"),
    ("protect", "wish"),
    ("leechseed", "substitute"),
    ("focuspunch", "substitute"),
    ("batonpass", "spiderweb"),
)
# NOTE: the SETUP_MOVES x {knockoff,rapidspin,toxic} pair was REMOVED here. In Showdown that
# incompatibility is applied via ``incompatibleMoves``, a SOFT cull that only trims the future
# movePool under size pressure and never guarantees exclusion — the generator produces
# setup+Toxic (and setup+Knock Off / setup+Rapid Spin) freely. Modeling it as a HARD reject
# pruned real sets (Cradily/Claydol/Girafarig CalmMind/SwordsDance + Toxic). The remaining pairs
# below are kept because no Showdown-generated set in the 36k-set gold-standard sample violates
# them (they are also soft in Showdown, but do not cause observed drift).
INCOMPATIBLE_MOVE_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("rest",), ("protect", "substitute")),
    (("selfdestruct", "explosion"), ("destinybond", "painsplit", "rest")),
    (("surf",), ("hydropump",)),
    (("bodyslam", "return"), ("bodyslam", "doubleedge")),
    (("fireblast",), ("flamethrower",)),
    (("bulkup",), ("overheat",)),
    (("endure",), ("substitute",)),
)


@dataclass(frozen=True)
class RandbatSourceMetadata:
    format_id: str
    generation: int
    showdown_root: Optional[str]
    sets_path: Optional[str]
    generator_path: Optional[str]
    source_hash: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Gen3RandbatVariant:
    variant_id: str
    source_set_id: str
    species: str
    role: str
    level: int
    moves: tuple[str, ...]
    ability: str
    item: str

    def matches(
        self,
        *,
        revealed_moves: Sequence[str] = (),
        revealed_ability: Optional[str] = None,
        revealed_item: Optional[str] = None,
        ruled_out_abilities: Sequence[str] = (),
        ruled_out_items: Sequence[str] = (),
    ) -> bool:
        normalized_moves = {_normalize_move(move) for move in self.moves}
        if any(not _revealed_move_matches_variant(move, normalized_moves) for move in revealed_moves):
            return False
        if _normalize_id(self.ability) in {_normalize_id(ability) for ability in ruled_out_abilities}:
            return False
        if revealed_ability and _normalize_id(self.ability) != _normalize_id(revealed_ability):
            return False
        if revealed_item and _normalize_id(self.item) != _normalize_id(revealed_item):
            return False
        if _normalize_id(self.item) in {_normalize_id(item) for item in ruled_out_items}:
            return False
        return True

    def to_summary(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "source_set_id": self.source_set_id,
            "role": self.role,
            "level": self.level,
            "moves": list(self.moves),
            "ability": self.ability,
            "item": self.item,
        }


@dataclass(frozen=True)
class Gen3RandbatSpeciesUniverse:
    species: str
    level: int
    variants: tuple[Gen3RandbatVariant, ...]

    def filter_variants(
        self,
        *,
        revealed_moves: Sequence[str] = (),
        revealed_ability: Optional[str] = None,
        revealed_item: Optional[str] = None,
        ruled_out_abilities: Sequence[str] = (),
        ruled_out_items: Sequence[str] = (),
    ) -> tuple[Gen3RandbatVariant, ...]:
        return tuple(
            variant
            for variant in self.variants
            if variant.matches(
                revealed_moves=revealed_moves,
                revealed_ability=revealed_ability,
                revealed_item=revealed_item,
                ruled_out_abilities=ruled_out_abilities,
                ruled_out_items=ruled_out_items,
            )
        )


@dataclass(frozen=True)
class Gen3RandbatSource:
    metadata: RandbatSourceMetadata
    universes: Mapping[str, Gen3RandbatSpeciesUniverse]
    move_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    species_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_showdown_root(
        cls,
        showdown_root: Path | str,
        *,
        cache_dir: Path | str | None = None,
        use_cache: bool = True,
    ) -> "Gen3RandbatSource":
        root = Path(showdown_root).expanduser().resolve()
        sets_path = root / "data" / "random-battles" / "gen3" / "sets.json"
        source_generator_path = root / "data" / "random-battles" / "gen3" / "teams.ts"
        dist_generator_path = root / "dist" / "data" / "random-battles" / "gen3" / "teams.js"
        if not sets_path.exists():
            raise FileNotFoundError(f"Gen 3 random battle sets were not found at {sets_path}.")
        if not dist_generator_path.exists():
            raise FileNotFoundError(
                "Built Pokemon Showdown Gen 3 random battle generator was not found at "
                f"{dist_generator_path}. Run `node build` in the Showdown checkout, or point "
                "`--showdown-root` at a built checkout."
            )
        source_hash = _source_hash(path for path in (sets_path, source_generator_path, dist_generator_path) if path.exists())
        metadata = RandbatSourceMetadata(
            format_id="gen3randombattle",
            generation=3,
            showdown_root=str(root),
            sets_path=str(sets_path),
            generator_path=str(dist_generator_path),
            source_hash=source_hash,
        )
        resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "pokezero"
        cache_path = resolved_cache_dir / f"gen3randbat-{source_hash}.json"
        if use_cache and cache_path.exists():
            cached = cls.from_payload(json.loads(cache_path.read_text(encoding="utf-8")))
            if cached.move_metadata and cached.species_metadata:
                return cached

        data = json.loads(sets_path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"Gen 3 random battle sets at {sets_path} must be a JSON object.")
        move_metadata, species_metadata = _load_showdown_metadata(root)
        source = cls.from_data(
            data,
            metadata=metadata,
            move_metadata=move_metadata,
            species_metadata=species_metadata,
        )
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(source.to_payload(), indent=2) + "\n", encoding="utf-8")
        return source

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        metadata: RandbatSourceMetadata | None = None,
        move_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        species_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "Gen3RandbatSource":
        source_metadata = metadata or RandbatSourceMetadata(
            format_id="gen3randombattle",
            generation=3,
            showdown_root=None,
            sets_path=None,
            generator_path=None,
            source_hash=_payload_hash(data),
        )
        universes: dict[str, Gen3RandbatSpeciesUniverse] = {}
        for raw_species, raw_entry in data.items():
            if not isinstance(raw_entry, Mapping):
                continue
            raw_sets = raw_entry.get("sets")
            if not isinstance(raw_sets, list):
                continue
            level = raw_entry.get("level")
            if not isinstance(level, int):
                level = 100
            species = _display_species_name(str(raw_species))
            variants = _build_variants_for_species(
                species=species,
                level=level,
                raw_sets=raw_sets,
                move_metadata=move_metadata or {},
                species_metadata=species_metadata or {},
            )
            universes[_normalize_species(species)] = Gen3RandbatSpeciesUniverse(
                species=species,
                level=level,
                variants=variants,
            )
        return cls(
            metadata=source_metadata,
            universes=universes,
            move_metadata=move_metadata or {},
            species_metadata=species_metadata or {},
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Gen3RandbatSource":
        metadata_payload = payload.get("metadata")
        if not isinstance(metadata_payload, Mapping):
            raise ValueError("Cached Gen 3 randbat payload is missing metadata.")
        metadata = RandbatSourceMetadata(
            format_id=str(metadata_payload.get("format_id") or "gen3randombattle"),
            generation=int(metadata_payload.get("generation") or 3),
            showdown_root=_optional_string(metadata_payload.get("showdown_root")),
            sets_path=_optional_string(metadata_payload.get("sets_path")),
            generator_path=_optional_string(metadata_payload.get("generator_path")),
            source_hash=str(metadata_payload.get("source_hash") or ""),
        )
        universes: dict[str, Gen3RandbatSpeciesUniverse] = {}
        raw_universes = payload.get("universes")
        if not isinstance(raw_universes, Mapping):
            raise ValueError("Cached Gen 3 randbat payload is missing universes.")
        for species_key, raw_universe in raw_universes.items():
            if not isinstance(raw_universe, Mapping):
                continue
            variants = tuple(
                Gen3RandbatVariant(
                    variant_id=str(raw_variant.get("variant_id")),
                    source_set_id=str(raw_variant.get("source_set_id")),
                    species=str(raw_universe.get("species") or species_key),
                    role=str(raw_variant.get("role") or "Unknown"),
                    level=int(raw_variant.get("level") or raw_universe.get("level") or 100),
                    moves=tuple(str(move) for move in raw_variant.get("moves", []) if str(move)),
                    ability=str(raw_variant.get("ability") or ""),
                    item=str(raw_variant.get("item") or ""),
                )
                for raw_variant in raw_universe.get("variants", [])
                if isinstance(raw_variant, Mapping)
            )
            species = str(raw_universe.get("species") or species_key)
            universes[_normalize_species(species)] = Gen3RandbatSpeciesUniverse(
                species=species,
                level=int(raw_universe.get("level") or 100),
                variants=variants,
            )
        move_metadata = payload.get("move_metadata")
        species_metadata = payload.get("species_metadata")
        return cls(
            metadata=metadata,
            universes=universes,
            move_metadata=move_metadata if isinstance(move_metadata, Mapping) else {},
            species_metadata=species_metadata if isinstance(species_metadata, Mapping) else {},
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_payload(),
            "universes": {
                key: {
                    "species": universe.species,
                    "level": universe.level,
                    "variants": [variant.to_summary() for variant in universe.variants],
                }
                for key, universe in self.universes.items()
            },
            "move_metadata": dict(self.move_metadata),
            "species_metadata": dict(self.species_metadata),
        }

    def supports(self, format_id: Optional[str]) -> bool:
        if not format_id:
            return False
        normalized = _normalize_id(format_id)
        return normalized == _normalize_id(self.metadata.format_id) or format_id in GEN3_RANDBAT_FORMATS

    def universe_for(self, species: str) -> Gen3RandbatSpeciesUniverse | None:
        return self.universes.get(_normalize_species(species))

    def summarize(
        self,
        *,
        format_id: Optional[str],
        species: str,
        revealed_moves: tuple[str, ...],
        revealed_ability: Optional[str] = None,
        revealed_item: Optional[str] = None,
        ruled_out_abilities: tuple[str, ...] = (),
        ruled_out_items: tuple[str, ...] = (),
    ) -> CandidateSetSummary | None:
        if not self.supports(format_id):
            return None
        universe = self.universe_for(species)
        if universe is None:
            return None
        surviving = universe.filter_variants(
            revealed_moves=revealed_moves,
            revealed_ability=revealed_ability,
            revealed_item=revealed_item,
            ruled_out_abilities=ruled_out_abilities,
            ruled_out_items=ruled_out_items,
        )
        total = max(1, len(universe.variants))
        notes: list[str] = []
        if revealed_moves:
            notes.append(f"Filtered by revealed moves: {', '.join(revealed_moves)}.")
        if revealed_ability:
            notes.append(f"Filtered by revealed ability: {revealed_ability}.")
        if revealed_item:
            notes.append(f"Filtered by revealed item: {revealed_item}.")
        if ruled_out_abilities:
            notes.append(f"Ruled out abilities: {', '.join(ruled_out_abilities)}.")

        # Off-script: the reveals are consistent with NO known set. This happens if Showdown's
        # randbats sets drift from our snapshot, or an unfiltered called/copied move slipped in.
        # Degrade to the unconstrained species pool (assume anything in the universe is possible)
        # rather than returning an empty, uncertainty-0.0 state that reads as "fully certain".
        inconsistent = not surviving and (
            bool(revealed_moves)
            or bool(revealed_ability)
            or bool(revealed_item)
            or bool(ruled_out_abilities)
            or bool(ruled_out_items)
        )
        if inconsistent:
            surviving = universe.variants
            notes.append("Reveals matched no known set (off-script); fell back to the full species pool.")

        possible_abilities = _stable_unique(variant.ability for variant in surviving if variant.ability)
        possible_items = _stable_unique(variant.item for variant in surviving if variant.item)
        possible_moves = _stable_unique(move for variant in surviving for move in variant.moves)
        count = len(surviving)
        return CandidateSetSummary(
            species=universe.species,
            candidate_count=count,
            # Off-script means maximally uncertain, not certain — force uncertainty high.
            uncertainty=1.0 if inconsistent else count / total,
            notes=tuple(notes),
            possible_abilities=tuple(possible_abilities),
            possible_items=tuple(possible_items),
            possible_moves=tuple(possible_moves),
            candidate_variants=tuple(variant.to_summary() for variant in surviving),
            source_metadata=self.metadata.to_payload(),
            inconsistent=inconsistent,
        )


def _build_variants_for_species(
    *,
    species: str,
    level: int,
    raw_sets: Sequence[Any],
    move_metadata: Mapping[str, Mapping[str, Any]],
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[Gen3RandbatVariant, ...]:
    species_id = _normalize_species(species)
    variants: list[Gen3RandbatVariant] = []
    seen: set[tuple[str, tuple[str, ...], str, str, str]] = set()
    for set_index, raw_set in enumerate(raw_sets, start=1):
        if not isinstance(raw_set, Mapping):
            continue
        role = str(raw_set.get("role") or f"Set {set_index}")
        raw_movepool = raw_set.get("movepool")
        raw_abilities = raw_set.get("abilities")
        if not isinstance(raw_movepool, list) or not isinstance(raw_abilities, list):
            continue
        movepool = tuple(str(move) for move in raw_movepool if str(move))
        abilities = tuple(str(ability) for ability in raw_abilities if str(ability))
        raw_preferred = raw_set.get("preferredTypes")
        preferred_types = (
            tuple(str(t) for t in raw_preferred if str(t)) if isinstance(raw_preferred, list) else ()
        )
        source_set_id = f"{species_id}-{set_index}"
        for moves in _enumerate_move_sets(
            movepool,
            role=role,
            species=species,
            abilities=abilities,
            preferred_types=preferred_types,
            move_metadata=move_metadata,
            species_metadata=species_metadata,
        ):
            counters = _move_counters(moves, move_metadata)
            possible_abilities = _possible_abilities(
                species=species,
                moves=moves,
                abilities=abilities,
                counters=counters,
            )
            for ability in possible_abilities:
                for item in _possible_items(
                    species=species,
                    role=role,
                    moves=moves,
                    ability=ability,
                    counters=counters,
                    species_metadata=species_metadata,
                ):
                    key = (source_set_id, tuple(sorted(moves)), ability, item, role)
                    if key in seen:
                        continue
                    seen.add(key)
                    variant_id = f"{source_set_id}-variant-{len(variants) + 1}"
                    variants.append(
                        Gen3RandbatVariant(
                            variant_id=variant_id,
                            source_set_id=source_set_id,
                            species=species,
                            role=role,
                            level=level,
                            moves=tuple(moves),
                            ability=ability,
                            item=item,
                        )
                    )
    return tuple(variants)


def _enumerate_move_sets(
    movepool: Sequence[str],
    *,
    role: str,
    species: str,
    abilities: Sequence[str],
    preferred_types: Sequence[str] = (),
    move_metadata: Mapping[str, Mapping[str, Any]],
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    unique_pool = tuple(_stable_unique(movepool))
    if len(unique_pool) <= 4:
        return (unique_pool,)
    candidates = [
        combo
        for combo in itertools.combinations(unique_pool, 4)
        if _valid_gen3_move_combo(
            combo,
            role=role,
            species=species,
            abilities=abilities,
            preferred_types=preferred_types,
            full_movepool=unique_pool,
            move_metadata=move_metadata,
            species_metadata=species_metadata,
        )
    ]
    if not candidates:
        candidates = list(itertools.combinations(unique_pool, 4))
    return tuple(candidates)


def _valid_gen3_move_combo(
    moves: Sequence[str],
    *,
    role: str,
    species: str,
    abilities: Sequence[str],
    preferred_types: Sequence[str] = (),
    full_movepool: Sequence[str],
    move_metadata: Mapping[str, Mapping[str, Any]],
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    normalized_moves = {_normalize_move(move) for move in moves}
    if sum(1 for move in normalized_moves if move.startswith("hiddenpower")) > 1:
        return False
    if role != "Staller" and len(normalized_moves.intersection(STATUS_INFLICTING_MOVES)) > 1:
        return False
    for move_a, move_b in MOVE_PAIRS:
        if move_a in full_movepool and move_b in full_movepool:
            if (move_a in normalized_moves) != (move_b in normalized_moves):
                return False
    for group_a, group_b in INCOMPATIBLE_MOVE_PAIRS:
        has_a = normalized_moves.intersection(group_a)
        has_b = normalized_moves.intersection(group_b)
        if has_a and has_b and has_a != has_b:
            return False
    # STAB requirement, mirroring Showdown's ACTUAL gen3 generator rather than a blanket
    # "STAB for every species type". Showdown only forces a *STAB-eligible* (damaging,
    # non-noStab) move for (a) every preferred type and (b) species types whose
    # moveEnforcementChecker fires (see ``_required_stab_types``). A combo is rejected only if
    # it omits a STAB-eligible move of a type Showdown would have enforced given this movepool.
    combo_has_hidden_power = any(move.startswith("hiddenpower") for move in normalized_moves)
    for required_type in _required_stab_types(
        species=species,
        preferred_types=preferred_types,
        full_movepool=full_movepool,
        move_metadata=move_metadata,
        species_metadata=species_metadata,
    ):
        if _moves_have_stab_eligible(normalized_moves, required_type, move_metadata):
            continue
        # Showdown culls all but one Hidden Power (cullMovePool). A type whose ONLY STAB-eligible
        # move is a Hidden Power is therefore not guaranteed when this set already spent its single
        # HP slot on a different-typed Hidden Power — the type's HP was culled and could not be
        # enforced. Allow the omission in exactly that case.
        if combo_has_hidden_power and _stab_only_via_hidden_power(
            full_movepool, required_type, move_metadata
        ):
            continue
        return False
    # Zero-STAB fallback (teams.ts 264-279): if the set ends up with NO STAB-eligible move of any
    # species type, the generator force-adds one whenever the movepool can still supply it. So a
    # zero-STAB combo is only reachable when the movepool cannot supply a species-type STAB at that
    # point — i.e. there is no non-Hidden-Power species STAB, and any Hidden-Power species STAB was
    # culled by a different-typed Hidden Power already in this combo. This is what makes e.g.
    # ``hypno: firepunch/protect/toxic/wish`` impossible (no checker forces Psychic at base SpA
    # < 100, but the fallback still forces the movepool's Psychic STAB).
    species_types = _species_types(species, species_metadata)
    if species_types and not any(
        _moves_have_stab_eligible(normalized_moves, species_type, move_metadata)
        for species_type in species_types
    ):
        if _movepool_species_stab_eligible(
            full_movepool, species_types, move_metadata, hidden_power=False
        ):
            return False
        if not combo_has_hidden_power and _movepool_species_stab_eligible(
            full_movepool, species_types, move_metadata, hidden_power=True
        ):
            return False
    return True


def _required_stab_types(
    *,
    species: str,
    preferred_types: Sequence[str],
    full_movepool: Sequence[str],
    move_metadata: Mapping[str, Mapping[str, Any]],
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Types for which this movepool forces a STAB-eligible move, per Showdown's gen3 generator.

    Two enforcement mechanisms are mirrored (both add a damaging, non-noStab move):
      * preferred-type enforcement (``randomMoveset`` "Enforce moves of all Preferred Types"),
        which is deterministic — every generated set carries a STAB-eligible move of each
        preferred type the movepool can supply; and
      * per-species-type STAB enforcement via ``moveEnforcementCheckers`` — which is NOT applied
        to every type: Bug fires only for {armaldo, heracross, parasect}; Flying skips Crobat;
        Steel skips Forretress; Psychic requires base SpA >= 100; Poison is skipped when a Bug
        STAB is already forced (its checker also gates on ``!counter.get('Bug')``); all other
        types fire whenever the movepool supplies a STAB-eligible move of that type.

    A type is only enforceable when the movepool actually contains a STAB-eligible move of it
    (otherwise the generator's ``while (checker) { if (!stabMoves.length) break; ... }`` can add
    nothing), so we gate every requirement on that.
    """
    species_id = _normalize_species(species)
    required: set[str] = set()

    def pool_has(type_name: str) -> bool:
        return _movepool_has_stab_eligible(full_movepool, type_name, move_metadata)

    # Preferred types: always enforced (when the movepool can supply one).
    for type_name in preferred_types:
        if pool_has(type_name):
            required.add(type_name)

    species_types = _species_types(species, species_metadata)
    # A Bug STAB is forced (making Poison enforcement skip) iff Bug is preferred or the species
    # is one of the three that enforce Bug — matching the checker's ``!counter.get('Bug')`` gate,
    # since Bug is never otherwise force-added before Poison enforcement in gen3.
    bug_forced = pool_has("Bug") and (
        "Bug" in preferred_types or species_id in BUG_ENFORCE_SPECIES
    )
    spa = _base_stat(species, "spa", species_metadata)
    for type_name in species_types:
        # Only types with a moveEnforcementChecker are enforced here. Notably GRASS and DRAGON
        # have NO checker in gen3 teams.ts, so a Grass/Dragon STAB is never hard-required (unless
        # it is a preferred type, handled above, or the zero-STAB fallback fires) — Showdown
        # freely makes Cradily/Celebi/Venusaur without a Grass move and Flygon/Latios without a
        # Dragon move.
        if type_name not in ENFORCED_STAB_TYPES or not pool_has(type_name):
            continue
        if type_name == "Bug":
            if species_id in BUG_ENFORCE_SPECIES:
                required.add(type_name)
        elif type_name == "Flying":
            if species_id != "crobat":
                required.add(type_name)
        elif type_name == "Steel":
            if species_id != "forretress":
                required.add(type_name)
        elif type_name == "Psychic":
            if spa >= 100:
                required.add(type_name)
        elif type_name == "Poison":
            if not bug_forced:
                required.add(type_name)
        else:
            required.add(type_name)
    return required


def _possible_abilities(
    *,
    species: str,
    moves: Sequence[str],
    abilities: Sequence[str],
    counters: Mapping[str, int],
) -> tuple[str, ...]:
    if not abilities:
        return ("",)
    if len(abilities) == 1:
        return (abilities[0],)
    if _normalize_species(species) == "yanma":
        return ("Compound Eyes",) if counters.get("inaccurate", 0) else ("Speed Boost",)
    possible: list[str] = []
    normalized_moves = {_normalize_move(move) for move in moves}
    for ability in abilities:
        if ability == "Rock Head" and not counters.get("recoil", 0):
            continue
        if ability == "Chlorophyll" and "sunnyday" not in normalized_moves:
            possible.append(ability)
            continue
        if ability == "Swift Swim" and "raindance" not in normalized_moves:
            possible.append(ability)
            continue
        possible.append(ability)
    return tuple(possible or abilities)


def _possible_items(
    *,
    species: str,
    role: str,
    moves: Sequence[str],
    ability: str,
    counters: Mapping[str, int],
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    species_id = _normalize_species(species)
    normalized_moves = {_normalize_move(move) for move in moves}
    speed = _base_stat(species, "spe", species_metadata)
    if species_id in {"latias", "latios"}:
        return ("Soul Dew",)
    if species_id == "linoone" and role == "Setup Sweeper":
        return ("Silk Scarf",)
    if species_id == "marowak":
        return ("Thick Club",)
    if species_id == "pikachu":
        return ("Light Ball",)
    if species_id == "shedinja":
        return ("Lum Berry",)
    if species_id == "unown":
        return ("Choice Band",) if counters.get("Physical", 0) else ("Twisted Spoon",)
    if species_id in {"deoxys", "deoxysattack"}:
        return ("White Herb",)
    if species_id == "farfetchd":
        return ("Stick",)
    if "trick" in normalized_moves:
        return ("Choice Band",)
    if counters.get("Physical", 0) >= 4:
        return ("Choice Band",)
    if counters.get("Physical", 0) >= 3 and ("batonpass" in normalized_moves or (role == "Wallbreaker" and counters.get("Special", 0))):
        return ("Choice Band",)
    if "dragondance" in normalized_moves and ability != "Natural Cure" and "healbell" not in normalized_moves and "substitute" not in normalized_moves:
        return ("Lum Berry",)
    if "bellydrum" in normalized_moves:
        return ("Salac Berry",) if "substitute" in normalized_moves else ("Lum Berry",)
    if "raindance" in normalized_moves and counters.get("Special", 0) >= 3:
        return ("Petaya Berry",)
    if role == "Berry Sweeper":
        if "endure" in normalized_moves:
            return ("Salac Berry",)
        if "flail" in normalized_moves or "reversal" in normalized_moves:
            return ("Liechi Berry",) if speed >= 90 else ("Salac Berry",)
        if "substitute" in normalized_moves and counters.get("Physical", 0) >= 3:
            return ("Liechi Berry",)
        if "substitute" in normalized_moves and counters.get("Special", 0) >= 3:
            return ("Petaya Berry",)
    salac_reqs = 60 <= speed <= 100 and not counters.get("priority", 0)
    if "bulkup" in normalized_moves and "substitute" in normalized_moves and counters.get("Status", 0) == 2 and salac_reqs:
        return ("Salac Berry",)
    if "swordsdance" in normalized_moves and "substitute" in normalized_moves and counters.get("Status", 0) == 2:
        if salac_reqs:
            return ("Salac Berry",)
        if speed > 100 and counters.get("Physical", 0) >= 2:
            return ("Liechi Berry",)
    if "swordsdance" in normalized_moves and counters.get("Status", 0) == 1:
        if salac_reqs:
            return ("Salac Berry",)
        if speed > 100:
            items = ["Lum Berry"]
            if counters.get("Physical", 0) >= 3:
                items.append("Liechi Berry")
            return tuple(items)
    return ("Leftovers",)


def _move_counters(
    moves: Sequence[str],
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    counters = {"Physical": 0, "Special": 0, "Status": 0}
    for raw_move in moves:
        move = _normalize_move(raw_move)
        metadata = move_metadata.get(move, {})
        move_type = _move_type(move, metadata)
        if metadata.get("damage") or metadata.get("damageCallback"):
            counters["damage"] = counters.get("damage", 0) + 1
        else:
            category = _gen3_move_category(move, metadata)
            counters[category] = counters.get(category, 0) + 1
        if metadata.get("recoil") or move in {"doubleedge", "submission", "volttackle"}:
            counters["recoil"] = counters.get("recoil", 0) + 1
        if int(metadata.get("priority") or 0) > 0:
            counters["priority"] = counters.get("priority", 0) + 1
        accuracy = metadata.get("accuracy")
        # Showdown counts a move as inaccurate only when ``accuracy && accuracy !== true &&
        # accuracy < 90``. A never-miss move serializes as boolean ``true``; ``bool`` is an ``int``
        # subclass in Python (``True < 90``), so it must be excluded explicitly — otherwise Aerial
        # Ace et al. are miscounted as inaccurate, which (via ``_possible_abilities``) wrongly pins
        # every Yanma set to Compound Eyes and hides Speed Boost from the candidate universe.
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and accuracy < 90:
            counters["inaccurate"] = counters.get("inaccurate", 0) + 1
        if move in RECOVERY_MOVES:
            counters["recovery"] = counters.get("recovery", 0) + 1
        if move in SETUP_MOVES:
            counters["setup"] = counters.get("setup", 0) + 1
        if move_type:
            counters[move_type] = counters.get(move_type, 0) + 1
    return counters


def _gen3_move_category(move: str, metadata: Mapping[str, Any]) -> str:
    if str(metadata.get("category") or "") == "Status":
        return "Status"
    move_type = _move_type(move, metadata)
    if move_type in PHYSICAL_TYPES:
        return "Physical"
    if move_type in SPECIAL_TYPES:
        return "Special"
    return str(metadata.get("category") or "Status")


def _move_type(move: str, metadata: Mapping[str, Any]) -> str:
    if move.startswith("hiddenpower") and len(move) > len("hiddenpower"):
        return _hidden_power_type_name(move[len("hiddenpower") :])
    return str(metadata.get("type") or "")


def _is_stab_eligible(move_id: str, species_type: str, move_metadata: Mapping[str, Mapping[str, Any]]) -> bool:
    """A move counts toward STAB for ``species_type`` exactly when Showdown's generator would.

    Mirrors ``!this.noStab.includes(moveid) && (move.basePower || move.basePowerCallback)`` with
    the move typed as ``species_type``: it must be a DAMAGING move (fixed base power or a
    base-power callback such as Return/Reversal/Flail; fixed-*damage* moves like Seismic Toss have
    only ``damageCallback`` and are excluded) and NOT on the NO_STAB list (Explosion, Pursuit,
    Quick Attack, Rapid Spin, ...). Status moves (Destiny Bond, Will-O-Wisp) never count — this is
    the fix for the old rule that treated a status move of a species' type as satisfying STAB."""
    metadata = move_metadata.get(move_id, {})
    if _move_type(move_id, metadata) != species_type:
        return False
    if move_id in NO_STAB:
        return False
    if move_id.startswith("hiddenpower"):
        return True
    base_power = metadata.get("basePower")
    if isinstance(base_power, (int, float)) and base_power > 0:
        return True
    return bool(metadata.get("basePowerCallback"))


def _moves_have_stab_eligible(
    normalized_moves: Iterable[str],
    species_type: str,
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    return any(_is_stab_eligible(move, species_type, move_metadata) for move in normalized_moves)


def _movepool_has_stab_eligible(
    movepool: Sequence[str],
    species_type: str,
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    return any(
        _is_stab_eligible(_normalize_move(move), species_type, move_metadata) for move in movepool
    )


def _movepool_species_stab_eligible(
    movepool: Sequence[str],
    species_types: Sequence[str],
    move_metadata: Mapping[str, Mapping[str, Any]],
    *,
    hidden_power: bool,
) -> bool:
    """Whether the movepool has a STAB-eligible move for ANY species type, restricted to Hidden
    Power moves (``hidden_power=True``) or non-Hidden-Power moves (``hidden_power=False``)."""
    for move in movepool:
        move_id = _normalize_move(move)
        if move_id.startswith("hiddenpower") != hidden_power:
            continue
        if any(_is_stab_eligible(move_id, species_type, move_metadata) for species_type in species_types):
            return True
    return False


def _stab_only_via_hidden_power(
    movepool: Sequence[str],
    species_type: str,
    move_metadata: Mapping[str, Mapping[str, Any]],
) -> bool:
    """True when every STAB-eligible ``species_type`` move in the movepool is a Hidden Power.

    Such a type cannot be guaranteed: Showdown keeps at most one Hidden Power per set, so if the
    set's HP slot is used by a different-typed Hidden Power, this type's HP is culled away."""
    eligible = [
        move_id
        for move in movepool
        if _is_stab_eligible((move_id := _normalize_move(move)), species_type, move_metadata)
    ]
    return bool(eligible) and all(move_id.startswith("hiddenpower") for move_id in eligible)


def _revealed_move_matches_variant(revealed_move: str, normalized_variant_moves: set[str]) -> bool:
    normalized = _normalize_move(revealed_move)
    if normalized == "hiddenpower":
        return any(move.startswith("hiddenpower") for move in normalized_variant_moves)
    return normalized in normalized_variant_moves


def _species_types(
    species: str,
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    metadata = species_metadata.get(_normalize_species(species), {})
    raw_types = metadata.get("types")
    if isinstance(raw_types, list):
        return tuple(str(item) for item in raw_types)
    return ()


def _base_stat(
    species: str,
    stat_id: str,
    species_metadata: Mapping[str, Mapping[str, Any]],
) -> int:
    metadata = species_metadata.get(_normalize_species(species), {})
    base_stats = metadata.get("baseStats")
    if isinstance(base_stats, Mapping):
        value = base_stats.get(stat_id)
        if isinstance(value, int):
            return value
    return 0


def _load_showdown_metadata(root: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    sim_index_path = root / "dist" / "sim" / "index.js"
    if not sim_index_path.exists():
        return {}, {}
    script = """
const root = process.argv[1];
const {Dex} = require(root + '/dist/sim/index.js');
const dex = Dex.mod('gen3');
const out = {moves: {}, species: {}};
for (const move of dex.moves.all()) {
  out.moves[move.id] = {
    name: move.name,
    type: move.type,
    category: move.category,
    basePower: move.basePower || 0,
    basePowerCallback: Boolean(move.basePowerCallback),
    accuracy: move.accuracy,
    priority: move.priority || 0,
    recoil: Boolean(move.recoil || move.hasCrashDamage),
    damage: move.damage || 0,
    damageCallback: Boolean(move.damageCallback)
  };
}
for (const species of dex.species.all()) {
  out.species[species.id] = {name: species.name, types: species.types || [], baseStats: species.baseStats || {}};
}
console.log(JSON.stringify(out));
"""
    try:
        result = subprocess.run(
            ["node", "-e", script, str(root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {}, {}
    payload = json.loads(result.stdout)
    moves = payload.get("moves") if isinstance(payload, Mapping) else {}
    species = payload.get("species") if isinstance(payload, Mapping) else {}
    return (
        moves if isinstance(moves, dict) else {},
        species if isinstance(species, dict) else {},
    )


def _source_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_CACHE_SCHEMA.encode("utf-8"))
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _display_species_name(species_id: str) -> str:
    compact = str(species_id).replace("-", " ").replace("_", " ").strip()
    if not compact:
        return species_id
    return " ".join(word[:1].upper() + word[1:] for word in compact.split())


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_id(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(str(value))
    return result


def _optional_string(value: Any) -> Optional[str]:
    return str(value) if value not in {None, ""} else None


def canonical_gen3_randbat_species_id(value: str) -> str:
    """Return the Gen 3 randbat source id for a possibly cosmetic public species name."""

    normalized = _normalize_id(value)
    if normalized.startswith("unown"):
        suffix = normalized[len("unown") :]
        if suffix in _UNOWN_COSMETIC_FORM_SUFFIXES:
            return "unown"
    return normalized


def _normalize_species(value: str) -> str:
    return canonical_gen3_randbat_species_id(value)


def _normalize_move(value: str) -> str:
    return _normalize_id(value)


def _hidden_power_type_name(value: str) -> str:
    normalized = _normalize_id(value)
    return normalized[:1].upper() + normalized[1:]


def _normalize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


_SOURCE_CACHE: dict[Path, "Gen3RandbatSource"] = {}
_SOURCE_CACHE_LOCK = threading.Lock()


def load_gen3_randbat_source_cached(showdown_root: Path | str) -> "Gen3RandbatSource":
    """Process-wide cached ``Gen3RandbatSource``.

    The source is immutable and heavy to build (it enumerates every species' candidate set
    universe), and belief engines share it read-only across battles (see ``resolved_player_view``).
    A collector creates one env per battle, so without caching each would rebuild the universe;
    this mirrors ``load_showdown_dex_cached`` so the cost is paid once per (root, process)."""
    root = Path(showdown_root).expanduser().resolve()
    with _SOURCE_CACHE_LOCK:
        cached = _SOURCE_CACHE.get(root)
        if cached is not None:
            return cached
    loaded = Gen3RandbatSource.from_showdown_root(root, use_cache=True)
    with _SOURCE_CACHE_LOCK:
        return _SOURCE_CACHE.setdefault(root, loaded)
