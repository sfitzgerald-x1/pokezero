"""Deterministic Gen 3 randbat catalog coverage planning.

The deep-line audit samples long, stateful games.  This module supplies its
complement: a source-derived fixture plan which visits every reachable species,
ability, move, and item through the production encoder.  The planner uses only
variants from :class:`Gen3RandbatSource`; custom-game starts are merely the
transport for materializing those otherwise legal randbat worlds deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .dex import normalize_id
from .env import BattleStartOverride
from .randbat import Gen3RandbatSource, Gen3RandbatVariant
from .showdown_fixture import FixturePokemon, pack_team


PLAN_SCHEMA_VERSION = "gen3-randbat-coverage-plan-v2"
OBSERVATION_FORMAT_ID = "gen3randombattle"


def normalize_coverage_move(move: str) -> str:
    """Normalize only dynamic battle aliases, preserving typed Hidden Power atoms.

    The source census distinguishes ``hiddenpowerfire`` from
    ``hiddenpowergrass``.  They share a request-action spelling in Showdown, but
    remain distinct source/belief atoms and must each be covered by the ledger.
    """

    normalized = normalize_id(move)
    if normalized.startswith("return"):
        return "return"
    if normalized.startswith("frustration"):
        return "frustration"
    return normalized


@dataclass(frozen=True)
class CoverageSelection:
    """One valid randbat variant materialized on one side of a fixture game."""

    species: str
    ability: str
    item: str
    level: int
    moves: tuple[str, ...]
    variant_id: str
    source_set_id: str
    pass_name: str
    targets: tuple[str, ...] = ()

    @classmethod
    def from_variant(
        cls,
        variant: Gen3RandbatVariant,
        *,
        pass_name: str,
        targets: Sequence[str] = (),
    ) -> "CoverageSelection":
        return cls(
            species=variant.species,
            ability=variant.ability,
            item=variant.item,
            level=variant.level,
            moves=tuple(variant.moves),
            variant_id=variant.variant_id,
            source_set_id=variant.source_set_id,
            pass_name=pass_name,
            targets=tuple(targets),
        )

    @property
    def species_id(self) -> str:
        return normalize_id(self.species)

    @property
    def ability_id(self) -> str:
        return normalize_id(self.ability)

    @property
    def item_id(self) -> str:
        return normalize_id(self.item)

    def to_fixture(self) -> FixturePokemon:
        return FixturePokemon(
            species=self.species,
            moves=self.moves,
            ability=self.ability,
            item=self.item,
            level=self.level,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "species_id": self.species_id,
            "ability": self.ability,
            "ability_id": self.ability_id,
            "item": self.item,
            "item_id": self.item_id,
            "level": self.level,
            "moves": list(self.moves),
            "variant_id": self.variant_id,
            "source_set_id": self.source_set_id,
            "pass": self.pass_name,
            "targets": list(self.targets),
        }


@dataclass(frozen=True)
class CoverageGame:
    """A deterministic, source-valid 1v1 custom-game audit fixture."""

    game_id: str
    seed: int
    pass_name: str
    purpose: str
    p1: CoverageSelection
    p2: CoverageSelection

    def start_override(self) -> BattleStartOverride:
        return BattleStartOverride(
            player_teams={"p1": pack_team((self.p1.to_fixture(),)), "p2": pack_team((self.p2.to_fixture(),))},
            observation_format_id=OBSERVATION_FORMAT_ID,
        )

    def selections(self) -> tuple[CoverageSelection, CoverageSelection]:
        return self.p1, self.p2

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "pass": self.pass_name,
            "purpose": self.purpose,
            "p1": self.p1.to_json_dict(),
            "p2": self.p2.to_json_dict(),
        }


@dataclass(frozen=True)
class CoveragePlan:
    """The complete fixture plan and its closed source-universe expectations."""

    source_metadata: Mapping[str, Any]
    expected_species: tuple[str, ...]
    expected_ability_pairs: tuple[tuple[str, str], ...]
    expected_moves: tuple[str, ...]
    expected_items: Mapping[str, str]
    expected_variants: tuple[str, ...]
    games: tuple[CoverageGame, ...]

    def games_for_shard(self, *, shard_index: int, shard_count: int) -> tuple[CoverageGame, ...]:
        if shard_count < 1:
            raise ValueError("shard_count must be positive")
        if not 0 <= shard_index < shard_count:
            raise ValueError(f"shard_index must be in [0, {shard_count}); got {shard_index}")
        return tuple(game for index, game in enumerate(self.games) if index % shard_count == shard_index)

    def coverage_ledger(self, games: Iterable[CoverageGame] | None = None) -> dict[str, Any]:
        """Return an exact ledger for all or a selected shard of planned games."""

        selected = tuple(self.games if games is None else games)
        species_first: dict[str, str] = {}
        ability_first: dict[tuple[str, str], str] = {}
        move_first: dict[str, str] = {}
        item_first: dict[str, str] = {}
        variant_first: dict[str, str] = {}
        for game in selected:
            for selection in game.selections():
                species_first.setdefault(selection.species_id, game.game_id)
                ability_first.setdefault((selection.species_id, selection.ability_id), game.game_id)
                variant_first.setdefault(selection.variant_id, game.game_id)
                for move in selection.moves:
                    move_first.setdefault(normalize_coverage_move(move), game.game_id)
                item_first.setdefault(selection.item_id, game.game_id)

        expected_item_ids = tuple(sorted(self.expected_items))
        missing_species = sorted(set(self.expected_species) - set(species_first))
        missing_abilities = sorted(
            f"{species}:{ability}"
            for species, ability in set(self.expected_ability_pairs) - set(ability_first)
        )
        missing_moves = sorted(set(self.expected_moves) - set(move_first))
        missing_items = sorted(set(expected_item_ids) - set(item_first))
        missing_variants = sorted(set(self.expected_variants) - set(variant_first))
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "source_metadata": dict(self.source_metadata),
            "games_selected": len(selected),
            "games_total": len(self.games),
            "expected": {
                "species": list(self.expected_species),
                "ability_pairs": [
                    {"species": species, "ability": ability}
                    for species, ability in self.expected_ability_pairs
                ],
                "moves": list(self.expected_moves),
                "items": [self.expected_items[item_id] for item_id in expected_item_ids],
                "variants": list(self.expected_variants),
            },
            "first_coverage": {
                "species": dict(sorted(species_first.items())),
                "ability_pairs": {
                    f"{species}:{ability}": game_id
                    for (species, ability), game_id in sorted(ability_first.items())
                },
                "moves": dict(sorted(move_first.items())),
                "items": {
                    self.expected_items.get(item_id, item_id): game_id
                    for item_id, game_id in sorted(item_first.items())
                },
                "variants": dict(sorted(variant_first.items())),
            },
            "uncovered": {
                "species": missing_species,
                "ability_pairs": missing_abilities,
                "moves": missing_moves,
                "items": [self.expected_items[item_id] for item_id in missing_items],
                "variants": missing_variants,
            },
            "complete": not (missing_species or missing_abilities or missing_moves or missing_items or missing_variants),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "source_metadata": dict(self.source_metadata),
            "games": [game.to_json_dict() for game in self.games],
            "coverage": self.coverage_ledger(),
        }


def build_coverage_plan(
    source: Gen3RandbatSource,
    *,
    source_species: Sequence[str] | None = None,
    source_moves: Sequence[str] | None = None,
    source_items: Sequence[str] | None = None,
    passes: Sequence[str] = ("A", "B"),
    gap_fill: bool = True,
    exact_variants: bool = False,
    seed_start: int = 9_300_000,
) -> CoveragePlan:
    """Build a deterministic complete-coverage fixture plan.

    ``source_*`` arguments are provided by ``gen3_randbat_entities`` in live
    runs.  Optional fallbacks make the planner directly unit-testable with small
    source fixtures.  A complete plan cannot be created if an expected source
    atom has no variant carrier; that mismatch is itself audit evidence.
    """

    normalized_passes = _normalize_passes(passes)
    variants_by_species = {
        species_id: tuple(sorted(universe.variants, key=lambda variant: variant.variant_id))
        for species_id, universe in sorted(source.universes.items())
        if universe.variants
    }
    if not variants_by_species:
        raise ValueError("Gen 3 randbat source contains no materializable variants")

    expected_species = tuple(sorted(normalize_id(species) for species in (source_species or variants_by_species)))
    missing_species = sorted(set(expected_species) - set(variants_by_species))
    if missing_species:
        raise ValueError(f"source species have no materializable variants: {', '.join(missing_species)}")

    expected_moves = tuple(sorted({normalize_coverage_move(move) for move in (source_moves or _all_moves(variants_by_species))}))
    expected_items = _expected_item_labels(source_items or _all_items(variants_by_species))
    _validate_carriers(
        variants_by_species,
        expected_moves=expected_moves,
        expected_items=expected_items,
    )

    all_variants = tuple(
        variant
        for species_id in expected_species
        for variant in variants_by_species[species_id]
    )
    expected_abilities = _expected_ability_pairs(
        variants_by_species,
        expected_species,
        ("A", "B") if exact_variants else normalized_passes,
    )
    uncovered_moves = set(expected_moves)
    uncovered_items = set(expected_items)
    if exact_variants:
        draft = [
            CoverageSelection.from_variant(variant, pass_name="exact-variant")
            for variant in all_variants
        ]
        for variant in all_variants:
            _consume_variant(variant, uncovered_moves, uncovered_items)
    else:
        draft = []
        for pass_name in normalized_passes:
            ability_slot = 0 if pass_name == "A" else 1
            for species_id in expected_species:
                variants = variants_by_species[species_id]
                abilities = sorted({variant.ability for variant in variants}, key=normalize_id)
                target_ability = abilities[min(ability_slot, len(abilities) - 1)]
                candidates = tuple(variant for variant in variants if variant.ability == target_ability)
                selected = _best_variant(candidates, uncovered_moves, uncovered_items)
                draft.append(CoverageSelection.from_variant(selected, pass_name=pass_name))
                _consume_variant(selected, uncovered_moves, uncovered_items)

    games: list[CoverageGame] = []
    for pair_index in range(0, len(draft), 2):
        p1 = draft[pair_index]
        p2 = draft[pair_index + 1] if pair_index + 1 < len(draft) else draft[0]
        game_index = len(games) + 1
        games.append(
            CoverageGame(
                game_id=(
                    f"variant-{game_index:04d}"
                    if exact_variants
                    else f"draft-{p1.pass_name.lower()}-{game_index:03d}"
                ),
                seed=seed_start + game_index - 1,
                pass_name=p1.pass_name,
                purpose="exact-variant" if exact_variants else "draft",
                p1=p1,
                p2=p2,
            )
        )

    if gap_fill and not exact_variants:
        anchor = CoverageSelection.from_variant(all_variants[0], pass_name="anchor")
        while uncovered_moves or uncovered_items:
            selected = _best_variant(all_variants, uncovered_moves, uncovered_items)
            targets = tuple(
                [f"move:{move}" for move in sorted(set(selected.moves) & uncovered_moves)]
                + (
                    [f"item:{normalize_id(selected.item)}"]
                    if normalize_id(selected.item) in uncovered_items
                    else []
                )
            )
            if not targets:
                raise ValueError(
                    "coverage gap-fill stalled despite prevalidated carriers; "
                    f"missing moves={sorted(uncovered_moves)} items={sorted(uncovered_items)}"
                )
            target = CoverageSelection.from_variant(
                selected,
                pass_name="gap-fill",
                targets=targets,
            )
            game_index = len(games) + 1
            games.append(
                CoverageGame(
                    game_id=f"gap-fill-{game_index:03d}",
                    seed=seed_start + game_index - 1,
                    pass_name="gap-fill",
                    purpose="gap-fill",
                    p1=target,
                    p2=anchor,
                )
            )
            _consume_variant(selected, uncovered_moves, uncovered_items)
            _consume_variant(all_variants[0], uncovered_moves, uncovered_items)

    plan = CoveragePlan(
        source_metadata=_source_provenance(source),
        expected_species=expected_species,
        expected_ability_pairs=expected_abilities,
        expected_moves=expected_moves,
        expected_items=expected_items,
        expected_variants=tuple(variant.variant_id for variant in all_variants) if exact_variants else (),
        games=tuple(games),
    )
    if (gap_fill or exact_variants) and not plan.coverage_ledger()["complete"]:
        raise AssertionError("coverage plan is incomplete")
    return plan


def _source_provenance(source: Gen3RandbatSource) -> dict[str, Any]:
    """Keep reproducible source identity without publishing machine-local paths."""

    metadata = source.metadata
    return {
        "format_id": metadata.format_id,
        "generation": metadata.generation,
        "source_hash": metadata.source_hash,
    }


def merge_coverage_ledgers(ledgers: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge completed shard ledgers into one exact source-coverage verdict.

    Shards are disjoint by game id, but an atom can intentionally appear in
    more than one fixture.  Keeping the lexically first fixture id makes the
    merged ``first_coverage`` map deterministic and equal to the unsharded
    plan's ordering.
    """

    payloads = tuple(ledgers)
    if not payloads:
        raise ValueError("at least one coverage ledger is required to merge")
    reference = payloads[0]
    schema_version = reference.get("schema_version")
    source_metadata = reference.get("source_metadata")
    expected = reference.get("expected")
    if not isinstance(schema_version, str) or not isinstance(source_metadata, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("coverage ledger is missing schema_version, source_metadata, or expected universe")

    expected_variants = expected.get("variants", ())
    if not isinstance(expected_variants, (list, tuple)):
        raise ValueError("coverage ledger expected variants must be a sequence")
    coverage_kinds = ("species", "ability_pairs", "moves", "items", "variants")
    first_coverage: dict[str, dict[str, str]] = {kind: {} for kind in coverage_kinds}
    games_selected = 0
    games_total = 0
    for payload in payloads:
        if payload.get("schema_version") != schema_version:
            raise ValueError("cannot merge coverage ledgers from different schemas")
        if payload.get("source_metadata") != source_metadata:
            raise ValueError("cannot merge coverage ledgers from different source versions")
        if payload.get("expected") != expected:
            raise ValueError("cannot merge coverage ledgers with different expected universes")
        games_selected += int(payload.get("games_selected") or 0)
        games_total = max(games_total, int(payload.get("games_total") or 0))
        raw_first = payload.get("first_coverage")
        if not isinstance(raw_first, Mapping):
            raise ValueError("coverage ledger is missing first_coverage")
        for kind, output in first_coverage.items():
            values = raw_first.get(kind)
            # A v1-only merge has no optional exact-variant lane. Mixed v1/v2
            # merges intentionally fail the schema guard above.
            if kind == "variants" and "variants" not in expected and values is None:
                continue
            if not isinstance(values, Mapping):
                raise ValueError(f"coverage ledger first_coverage is missing {kind}")
            for atom, game_id in values.items():
                atom_key = str(atom)
                candidate = str(game_id)
                prior = output.get(atom_key)
                if prior is None or candidate < prior:
                    output[atom_key] = candidate

    expected_species = {str(species) for species in expected.get("species", ())}
    raw_pairs = expected.get("ability_pairs", ())
    expected_pairs = {
        f"{str(pair.get('species'))}:{str(pair.get('ability'))}"
        for pair in raw_pairs
        if isinstance(pair, Mapping)
    }
    expected_moves = {str(move) for move in expected.get("moves", ())}
    item_labels = {normalize_id(str(item)): str(item) for item in expected.get("items", ())}
    expected_variant_ids = {str(variant_id) for variant_id in expected_variants}
    missing_species = sorted(expected_species - set(first_coverage["species"]))
    missing_pairs = sorted(expected_pairs - set(first_coverage["ability_pairs"]))
    missing_moves = sorted(expected_moves - set(first_coverage["moves"]))
    missing_items = sorted(set(item_labels) - {normalize_id(item) for item in first_coverage["items"]})
    missing_variants = sorted(expected_variant_ids - set(first_coverage["variants"]))
    # Exact-variant mode promises every source tuple was materialized. Atom-only
    # coverage can legitimately finish once every required atom is observed.
    requires_every_fixture = bool(expected_variant_ids)
    return {
        "schema_version": schema_version,
        "source_metadata": dict(source_metadata),
        "games_selected": games_selected,
        "games_total": games_total,
        "expected": dict(expected),
        "first_coverage": {kind: dict(sorted(values.items())) for kind, values in first_coverage.items()},
        "uncovered": {
            "species": missing_species,
            "ability_pairs": missing_pairs,
            "moves": missing_moves,
            "items": [item_labels[item_id] for item_id in missing_items],
            "variants": missing_variants,
        },
        "complete": (
            (not requires_every_fixture or games_selected == games_total)
            and not (missing_species or missing_pairs or missing_moves or missing_items or missing_variants)
        ),
    }


def _normalize_passes(passes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(pass_name).upper() for pass_name in passes)
    if not normalized or any(pass_name not in {"A", "B"} for pass_name in normalized):
        raise ValueError("passes must be a non-empty subset of ('A', 'B')")
    if len(set(normalized)) != len(normalized):
        raise ValueError("passes must not contain duplicates")
    return normalized


def _expected_ability_pairs(
    variants_by_species: Mapping[str, Sequence[Gen3RandbatVariant]],
    species: Sequence[str],
    passes: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for species_id in species:
        abilities = sorted({normalize_id(variant.ability) for variant in variants_by_species[species_id]})
        if not abilities:
            raise ValueError(f"species {species_id} has no reachable randbat ability")
        for pass_name in passes:
            pairs.append((species_id, abilities[min(0 if pass_name == "A" else 1, len(abilities) - 1)]))
    return tuple(dict.fromkeys(pairs))


def _expected_item_labels(items: Sequence[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in sorted(items, key=normalize_id):
        item_id = normalize_id(item)
        if not item_id:
            continue
        prior = labels.setdefault(item_id, str(item))
        if prior != str(item):
            raise ValueError(f"item normalization collision: {prior!r} and {item!r}")
    if not labels:
        raise ValueError("source item universe is empty")
    return labels


def _validate_carriers(
    variants_by_species: Mapping[str, Sequence[Gen3RandbatVariant]],
    *,
    expected_moves: Sequence[str],
    expected_items: Mapping[str, str],
) -> None:
    all_variants = tuple(variant for variants in variants_by_species.values() for variant in variants)
    carrier_moves = {normalize_coverage_move(move) for variant in all_variants for move in variant.moves}
    carrier_items = {normalize_id(variant.item) for variant in all_variants}
    missing_moves = sorted(set(expected_moves) - carrier_moves)
    missing_items = sorted(set(expected_items) - carrier_items)
    if missing_moves or missing_items:
        fragments: list[str] = []
        if missing_moves:
            fragments.append(f"moves={', '.join(missing_moves)}")
        if missing_items:
            fragments.append(
                "items=" + ", ".join(expected_items[item_id] for item_id in missing_items)
            )
        raise ValueError("source atoms lack a materializable source-variant carrier: " + "; ".join(fragments))


def _best_variant(
    variants: Sequence[Gen3RandbatVariant],
    uncovered_moves: set[str],
    uncovered_items: set[str],
) -> Gen3RandbatVariant:
    if not variants:
        raise ValueError("cannot select from an empty variant sequence")

    def key(variant: Gen3RandbatVariant) -> tuple[int, int, str]:
        new_moves = len({normalize_coverage_move(move) for move in variant.moves} & uncovered_moves)
        new_items = int(normalize_id(variant.item) in uncovered_items)
        return (-new_moves, -new_items, variant.variant_id)

    return min(variants, key=key)


def _consume_variant(
    variant: Gen3RandbatVariant,
    uncovered_moves: set[str],
    uncovered_items: set[str],
) -> None:
    uncovered_moves.difference_update(normalize_coverage_move(move) for move in variant.moves)
    uncovered_items.discard(normalize_id(variant.item))


def _all_moves(variants_by_species: Mapping[str, Sequence[Gen3RandbatVariant]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                normalize_coverage_move(move)
                for variants in variants_by_species.values()
                for variant in variants
                for move in variant.moves
            }
        )
    )


def _all_items(variants_by_species: Mapping[str, Sequence[Gen3RandbatVariant]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                variant.item
                for variants in variants_by_species.values()
                for variant in variants
                if variant.item
            },
            key=normalize_id,
        )
    )
