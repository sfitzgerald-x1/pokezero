"""Per-trajectory exact-server hidden-set draws for the opt-in reference.

Known species use the pinned server's randomSet with ten rejection attempts,
then force known original-set traits into the last draw. Unknown species/sets
come from fresh server-generated parties with species-clause filtering. The
paper does not specify team-context conditioning: known draws use empty team
context, unknown draws preserve a generated party's context, and forced sets
do not promise generator compatibility. Receipts make these choices visible.

Inputs are explicit PUBLIC original-set traits, not a live opponent request.
Mapping mutated current items/abilities or transformed moves to these traits
is the root factory's responsibility, not permission to copy private fields.
This sampler alone does not qualify a root, a study, or playing strength.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
import re
from typing import Any, Mapping, Protocol, Sequence

from .paper_reference import ReferenceRefusal
from ..determinization import _gen3_randbat_fixture_spread, _hp_stat
from ..randbat import Gen3RandbatSource, canonical_gen3_randbat_species_id
from ..showdown_fixture import FixturePokemon, pack_team
from ..tier2 import canonical_move_id


def _id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


@dataclass(frozen=True)
class KnownSetTraits:
    species: str
    moves: tuple[str, ...] = ()
    ability: str | None = None
    # None means UNKNOWN; an empty string means publicly known itemless.
    item: str | None = None
    level: int | None = None
    gender: str | None = None
    max_hp: int | None = None
    ruled_out_abilities: tuple[str, ...] = ()
    ruled_out_items: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.species, str) or not _id(self.species):
            raise ReferenceRefusal("known set has no public species")
        if (not isinstance(self.moves, tuple) or len(self.moves) > 4
                or any(not isinstance(m, str) or not _id(m) for m in self.moves)
                or len({canonical_move_id(m) for m in self.moves}) != len(self.moves)):
            raise ReferenceRefusal("known set has invalid or duplicate public moves")
        if any(v is not None and not isinstance(v, str) for v in (self.ability, self.item)):
            raise ReferenceRefusal("known item and ability must be public text or unknown")
        if self.ability is not None and not _id(self.ability):
            raise ReferenceRefusal("known ability cannot be empty")
        if self.level is not None and (type(self.level) is not int or not 1 <= self.level <= 100):
            raise ReferenceRefusal("known level is invalid")
        if self.gender not in (None, "M", "F", "N"):
            raise ReferenceRefusal("known gender is invalid")
        if self.max_hp is not None and (type(self.max_hp) is not int or not 1 <= self.max_hp <= 1000):
            raise ReferenceRefusal("known maximum HP is invalid")
        for excluded, positive in ((self.ruled_out_abilities, self.ability),
                                   (self.ruled_out_items, self.item)):
            if (not isinstance(excluded, tuple)
                    or any(not isinstance(v, str) or not _id(v) for v in excluded)
                    or len({_id(v) for v in excluded}) != len(excluded)
                    or (positive is not None and _id(positive) in {_id(v) for v in excluded})):
                raise ReferenceRefusal("invalid or contradictory public original-set exclusions")


class ServerGenerator(Protocol):
    def generate_reference_set(self, *, seed: int, species: str) -> Mapping[str, Any]: ...
    def generate_scenario_team(self, *, seed: int) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class KnownDrawReceipt:
    species: str
    seeds: tuple[int, ...]
    forced: bool
    packed_set_sha256: str
    gender_assigned_after_server_draw: bool = False


@dataclass(frozen=True)
class HiddenTeamDraw:
    team: tuple[FixturePokemon, ...]
    known: tuple[KnownDrawReceipt, ...]
    unknown_party_seeds: tuple[int, ...]
    packed_team_sha256: str

    @property
    def forced_sets(self) -> int:
        return sum(row.forced for row in self.known)


def _fixture(row: Mapping[str, Any]) -> FixturePokemon:
    if not isinstance(row, Mapping):
        raise ReferenceRefusal("server returned a malformed set")
    fields = ("species", "moves", "ability", "item", "level", "nature", "gender", "evs", "ivs")
    try:
        # Copy JSON fields so callers cannot mutate generator evidence afterward.
        values = json.loads(json.dumps({key: row[key] for key in fields}, allow_nan=False))
        moves = values["moves"]
        if (not isinstance(moves, list) or not 1 <= len(moves) <= 4
                or any(not isinstance(move, str) or not _id(move) for move in moves)):
            raise ValueError("invalid move list")
        values["moves"] = tuple(canonical_move_id(move) for move in moves)
        if len(set(values["moves"])) != len(values["moves"]):
            raise ValueError("duplicate moves")
        KnownSetTraits(values["species"], values["moves"], values["ability"], values["item"],
            values["level"], values["gender"] or None)
        for spread, maximum in (("evs", 255), ("ivs", 31)):
            if (not isinstance(values[spread], dict)
                    or set(values[spread]) != {"hp", "atk", "def", "spa", "spd", "spe"}
                    or any(type(v) is not int or not 0 <= v <= maximum for v in values[spread].values())):
                raise ValueError("invalid stat spread")
        return FixturePokemon(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceRefusal("server returned invalid reference set evidence") from exc


def _matches(candidate: FixturePokemon, traits: KnownSetTraits, source: Gen3RandbatSource) -> bool:
    return (canonical_gen3_randbat_species_id(candidate.species)
            == canonical_gen3_randbat_species_id(traits.species)
        and all(_move_known_matches(canonical_move_id(m), candidate.moves) for m in traits.moves)
        and (traits.ability is None or _id(candidate.ability or "") == _id(traits.ability))
        and (traits.item is None or _id(candidate.item or "") == _id(traits.item))
        and _id(candidate.ability or "") not in {_id(v) for v in traits.ruled_out_abilities}
        and _id(candidate.item or "") not in {_id(v) for v in traits.ruled_out_items}
        and (traits.level is None or candidate.level == traits.level)
        and (traits.gender is None or (candidate.gender or "N") == traits.gender)
        and (traits.max_hp is None or _maximum_hp(candidate, source) == traits.max_hp))


def _maximum_hp(candidate: FixturePokemon, source: Gen3RandbatSource) -> int:
    species = canonical_gen3_randbat_species_id(candidate.species)
    if species == "shedinja":
        return 1
    base = source.species_metadata.get(species, {}).get("baseStats", {}).get("hp")
    if type(base) is not int:
        raise ReferenceRefusal("maximum-HP conditioning needs pinned species base stats")
    return _hp_stat(base_hp=base, iv=candidate.ivs["hp"], ev=candidate.evs["hp"], level=candidate.level)


def _move_known_matches(required: str, moves: Sequence[str]) -> bool:
    # The public move line can reveal Hidden Power without revealing its type.
    # Do not reject a server's typed set, or force it to an arbitrary dark IVs.
    return required in moves or (required == "hiddenpower"
        and any(move.startswith("hiddenpower") for move in moves))


class PaperHiddenTeamSampler:
    """One NEW team per call, with no fixed-world cache and no private inputs.

    The set source supplies only Gen 3 Dex/stat rules for forced-set spreads;
    its candidate catalog is never queried for draws. Unknown-party rejection
    has a safety cap: exhausting it refuses, rather than inventing a team.
    """

    def __init__(self, generator: ServerGenerator, *, set_source: Gen3RandbatSource) -> None:
        self.generator, self.set_source = generator, set_source

    def draw(self, known: tuple[KnownSetTraits, ...], rng: random.Random) -> HiddenTeamDraw:
        if (not isinstance(known, tuple) or len(known) > 6
                or any(not isinstance(mon, KnownSetTraits) for mon in known)):
            raise ReferenceRefusal("reference requires at most six explicit public species traits")
        seen = {canonical_gen3_randbat_species_id(mon.species) for mon in known}
        if len(seen) != len(known):
            raise ReferenceRefusal("public known species violate the random-battle species clause")
        team: list[FixturePokemon] = []
        receipts = []
        for traits in known:
            seeds = []
            assigned_gender = False
            for _ in range(10):
                seed = rng.getrandbits(32)
                seeds.append(seed)
                candidate = _fixture(self.generator.generate_reference_set(seed=seed, species=traits.species))
                if canonical_gen3_randbat_species_id(candidate.species) != canonical_gen3_randbat_species_id(traits.species):
                    raise ReferenceRefusal("known-species server draw changed species")
                if traits.gender is not None and not candidate.gender:
                    # Gen 3 randomSet leaves gender to Battle's independent
                    # Pokemon constructor. Conditioning on an already-public
                    # gender must not make every otherwise-valid set reject.
                    metadata = self.set_source.species_metadata.get(
                        canonical_gen3_randbat_species_id(traits.species), {})
                    fixed = metadata.get("gender")
                    ratio = metadata.get("genderRatio", {})
                    if ((fixed and fixed != traits.gender)
                            or (not fixed and ratio and ratio.get(traits.gender, 0) <= 0)):
                        raise ReferenceRefusal("public gender is incompatible with sampled species")
                    candidate = replace(candidate, gender=traits.gender)
                    assigned_gender = True
                if _matches(candidate, traits, self.set_source):
                    break
            forced = not _matches(candidate, traits, self.set_source)
            if forced:
                required = tuple(next((m for m in candidate.moves if m.startswith("hiddenpower")), "hiddenpower")
                    if canonical_move_id(move) == "hiddenpower" else canonical_move_id(move)
                    for move in traits.moves)
                moves = (required + tuple(m for m in candidate.moves if m not in required))[:4]
                candidate = replace(candidate, moves=moves,
                    ability=candidate.ability if traits.ability is None else traits.ability,
                    item=candidate.item if traits.item is None else traits.item,
                    level=candidate.level if traits.level is None else traits.level,
                    gender=candidate.gender if traits.gender is None else traits.gender)
                # Gen 3 Hidden Power and pinch-item HP spreads must track the
                # forced traits. Do not retain incompatible IVs from the draw.
                spread = _gen3_randbat_fixture_spread({}, species=candidate.species,
                    moves=tuple(candidate.moves), item=candidate.item, level=candidate.level,
                    set_source=self.set_source)
                if spread is None:
                    raise ReferenceRefusal("forced known-set spread cannot be reconstructed")
                candidate = replace(candidate, evs=spread["evs"], ivs=spread["ivs"])
                if not _matches(candidate, traits, self.set_source):
                    # Negative facts cannot be invented away or silently
                    # discarded. No eleventh draw or catalog fallback.
                    raise ReferenceRefusal("tenth-draw forced completion violates a public original-set trait/exclusion")
            team.append(candidate)
            receipts.append(KnownDrawReceipt(traits.species, tuple(seeds), forced,
                hashlib.sha256(pack_team((candidate,)).encode()).hexdigest(), assigned_gender))
        party_seeds = []
        while len(team) < 6:
            if len(party_seeds) >= 10:
                raise ReferenceRefusal("unknown-species fresh-party rejection safety cap exceeded")
            seed = rng.getrandbits(32)
            party_seeds.append(seed)
            rows = self.generator.generate_scenario_team(seed=seed)
            if len(rows) != 6:
                raise ReferenceRefusal("unknown-species generator returned a partial party")
            for row in rows:
                candidate = _fixture(row)
                species = canonical_gen3_randbat_species_id(candidate.species)
                if species not in seen:
                    seen.add(species)
                    team.append(candidate)
                if len(team) == 6:
                    break
        return HiddenTeamDraw(tuple(team), tuple(receipts), tuple(party_seeds),
            hashlib.sha256(pack_team(tuple(team)).encode()).hexdigest())
