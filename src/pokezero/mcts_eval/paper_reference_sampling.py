"""Per-trajectory exact-server hidden-set draws for the opt-in reference.

Known species use the pinned server's randomSet with ten rejection attempts,
then force known original-set traits into the last draw. Unknown species/sets
come from fresh server-generated parties with species-clause filtering. The
paper does not specify team-context conditioning: known draws use empty team
context, unknown draws preserve a generated party's context, and forced sets
do not promise generator compatibility. Receipts make these choices visible.

An explicitly registered opt-in variant may project earlier templates from
the SAME ten draws when the tenth violates public HP or other hard facts.
It selects the latest fully compatible projection without additional draws,
catalog proposals, invented EVs, or silent relaxation. This is an adaptation,
not the paper's exact tenth-template rule or a conditional posterior sampler.

An additional explicit repair may extend ONLY an exhausted incompatible
completion, up to a registered total draw limit. Each extra server template
must match all public facts either directly or after the same disclosed trait
projection. It never invents HP/EVs or relaxes exclusions. This changes the
completion proposal, not the source position, and is not an exact posterior.

Inputs are explicit PUBLIC original-set traits, not a live opponent request.
Mapping mutated current items/abilities or transformed moves to these traits
is the root factory's responsibility, not permission to copy private fields.
This sampler alone does not qualify a root, a study, or playing strength.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    completion_template_attempt: int | None = None
    completion_projection_evidence: tuple[dict[str, Any], ...] = ()
    max_known_set_draws: int = 10
    extended_conditioning_used: bool = False


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


def _force_known_traits(candidate: FixturePokemon, traits: KnownSetTraits,
                        source: Gen3RandbatSource) -> FixturePokemon:
    required = tuple(next((m for m in candidate.moves if m.startswith("hiddenpower")), "hiddenpower")
        if canonical_move_id(move) == "hiddenpower" else canonical_move_id(move)
        for move in traits.moves)
    moves = (required + tuple(m for m in candidate.moves if m not in required))[:4]
    candidate = replace(candidate, moves=moves,
        ability=candidate.ability if traits.ability is None else traits.ability,
        item=candidate.item if traits.item is None else traits.item,
        level=candidate.level if traits.level is None else traits.level,
        gender=candidate.gender if traits.gender is None else traits.gender)
    spread = _gen3_randbat_fixture_spread({}, species=candidate.species,
        moves=tuple(candidate.moves), item=candidate.item, level=candidate.level,
        set_source=source)
    if spread is None:
        raise ReferenceRefusal("forced known-set spread cannot be reconstructed")
    return replace(candidate, evs=spread["evs"], ivs=spread["ivs"])


def _projection_evidence(candidate: FixturePokemon, traits: KnownSetTraits,
                         source: Gen3RandbatSource, attempt: int, seed: int) -> dict[str, Any]:
    mismatches = {}
    if canonical_gen3_randbat_species_id(candidate.species) != canonical_gen3_randbat_species_id(traits.species):
        mismatches["species"] = {"actual": candidate.species, "required": traits.species}
    missing = [m for m in traits.moves if not _move_known_matches(canonical_move_id(m), candidate.moves)]
    if missing:
        mismatches["moves"] = {"actual": candidate.moves, "required": traits.moves}
    for field, excluded in (("ability", traits.ruled_out_abilities), ("item", traits.ruled_out_items)):
        actual, required = getattr(candidate, field), getattr(traits, field)
        if required is not None and _id(actual or "") != _id(required):
            mismatches[field] = {"actual": actual, "required": required}
        if _id(actual or "") in {_id(v) for v in excluded}:
            mismatches[field+"_exclusion"] = {"actual": actual, "excluded": excluded}
    for field in ("level", "gender"):
        actual, required = getattr(candidate, field), getattr(traits, field)
        if field == "gender":
            actual = actual or "N"
        if required is not None and actual != required:
            mismatches[field] = {"actual": actual, "required": required}
    if traits.max_hp is not None:
        actual_hp = _maximum_hp(candidate, source)
        if actual_hp != traits.max_hp:
            mismatches["max_hp"] = {"actual": actual_hp, "required": traits.max_hp}
    return {"attempt": attempt, "seed": seed, "mismatches": mismatches,
        "packed_set_sha256": hashlib.sha256(pack_team((candidate,)).encode()).hexdigest()}


class PaperHiddenTeamSampler:
    """One NEW team per call, with no fixed-world cache and no private inputs.

    The set source supplies only Gen 3 Dex/stat rules for forced-set spreads;
    its candidate catalog is never queried for draws. Unknown-party rejection
    has a safety cap: exhausting it refuses, rather than inventing a team.
    """

    def __init__(self, generator: ServerGenerator, *, set_source: Gen3RandbatSource,
                 allow_earlier_compatible_template: bool = False,
                 max_known_set_draws: int = 10) -> None:
        if type(allow_earlier_compatible_template) is not bool:
            raise ReferenceRefusal("completion adaptation flag must be explicit boolean")
        if type(max_known_set_draws) is not int or not 10 <= max_known_set_draws <= 256:
            raise ReferenceRefusal("known-set draw limit must be an explicit integer in [10, 256]")
        self.generator, self.set_source = generator, set_source
        # Opt-in adaptation, NOT the paper's exact tenth-template rule. Preserve
        # that rule unless the caller explicitly registers the bounded variant.
        self.allow_earlier_compatible_template = allow_earlier_compatible_template
        self.max_known_set_draws = max_known_set_draws

    def draw(self, known: tuple[KnownSetTraits, ...], rng: random.Random) -> HiddenTeamDraw:
        self._validate_known(known)
        team, receipts = self._draw_known(known, rng)
        unknown, party_seeds = self._draw_unknown(known, rng)
        team.extend(unknown)
        return HiddenTeamDraw(tuple(team), tuple(receipts), tuple(party_seeds),
            hashlib.sha256(pack_team(tuple(team)).encode()).hexdigest())

    @staticmethod
    def _validate_known(known):
        if (not isinstance(known, tuple) or len(known) > 6
                or any(not isinstance(mon, KnownSetTraits) for mon in known)):
            raise ReferenceRefusal("reference requires at most six explicit public species traits")
        seen = {canonical_gen3_randbat_species_id(mon.species) for mon in known}
        if len(seen) != len(known):
            raise ReferenceRefusal("public known species violate the random-battle species clause")

    def _draw_known(self, known, rng):
        team: list[FixturePokemon] = []
        receipts = []
        for traits in known:
            seeds = []
            candidates = []
            genders_assigned = []
            def draw_candidate():
                assigned_gender = False
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
                candidates.append(candidate)
                genders_assigned.append(assigned_gender)
                return candidate, assigned_gender

            for _ in range(10):
                candidate, assigned_gender = draw_candidate()
                if _matches(candidate, traits, self.set_source):
                    break
            forced = not _matches(candidate, traits, self.set_source)
            projection_evidence = []
            selected_attempt = None
            if forced:
                attempts = range(len(candidates)-1, -1, -1) if self.allow_earlier_compatible_template else (len(candidates)-1,)
                for index in attempts:
                    projected = _force_known_traits(candidates[index], traits, self.set_source)
                    evidence = _projection_evidence(projected, traits, self.set_source, index+1, seeds[index])
                    projection_evidence.append(evidence)
                    if _matches(projected, traits, self.set_source):
                        candidate, selected_attempt = projected, index+1
                        assigned_gender = genders_assigned[index]
                        break
                # Preserve the original ten-draw behavior unless every allowed
                # projection is incompatible. Fresh proposals remain pinned to
                # this same public species; no private set/catalog is consulted.
                while selected_attempt is None and len(seeds) < self.max_known_set_draws:
                    candidate, assigned_gender = draw_candidate()
                    if _matches(candidate, traits, self.set_source):
                        forced = False
                        break
                    projected = _force_known_traits(candidate, traits, self.set_source)
                    evidence = _projection_evidence(projected, traits, self.set_source, len(seeds), seeds[-1])
                    projection_evidence.append(evidence)
                    if _matches(projected, traits, self.set_source):
                        candidate, selected_attempt = projected, len(seeds)
                        break
                if forced and selected_attempt is None:
                    error = ReferenceRefusal("bounded forced completion violates public traits/exclusions: "
                        + json.dumps(projection_evidence[-1]["mismatches"], sort_keys=True))
                    error.sampling_diagnostic = {"known_public_traits": asdict(traits), "draw_seeds": seeds,
                        "allow_earlier_compatible_template": self.allow_earlier_compatible_template,
                        "max_known_set_draws": self.max_known_set_draws,
                        "projection_evidence": projection_evidence}
                    raise error
            team.append(candidate)
            receipts.append(KnownDrawReceipt(traits.species, tuple(seeds), forced,
                hashlib.sha256(pack_team((candidate,)).encode()).hexdigest(), assigned_gender,
                selected_attempt, tuple(projection_evidence), self.max_known_set_draws, len(seeds) > 10))
        return team, receipts

    def _draw_unknown(self, known, rng):
        # Unknown completion depends on known SPECIES, never their sampled sets.
        seen = {canonical_gen3_randbat_species_id(mon.species) for mon in known}
        team = []
        party_seeds = []
        while len(team) + len(known) < 6:
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
                if len(team) + len(known) == 6:
                    break
        return team, party_seeds

    def draw_membership_first(self, known, rng, *, required, check, receipt):
        """Same accepted joint law, without drawing known sets for rejected parties.

        Conditional on fixed public known traits, original draws factor as
        K(known sets) U(unknown completion | known species). The necessary
        membership predicate M depends only on species in U plus fixed known
        species, so p(K,U | M) = p(K) p(U | M). Reordering independent original
        seed draws changes the deterministic stream coupling, not this target.
        No later trait is forced and no source-generated party is modified.
        Known-set completion errors and native errors still propagate; the
        original bounded completion remains unchanged for every retained party.
        """
        self._validate_known(known)
        if (not isinstance(required, frozenset) or not required or len(required) > 6
                or any(not isinstance(s, str) or not s for s in required)):
            raise ReferenceRefusal('invalid necessary membership-first species predicate')
        fixed = {canonical_gen3_randbat_species_id(mon.species) for mon in known}
        for _ in range(2048):
            check()
            unknown, seeds = self._draw_unknown(known, rng)
            receipt['complete_proposals'] += 1
            check()
            if not required <= fixed | {canonical_gen3_randbat_species_id(mon.species) for mon in unknown}:
                receipt['membership_rejections'] += 1
                continue
            receipt['matches'] += 1
            check()
            team, known_receipts = self._draw_known(known, rng)
            check()
            team.extend(unknown)
            receipt['known_completions_materialized'] += 1
            return HiddenTeamDraw(tuple(team), tuple(known_receipts), tuple(seeds),
                hashlib.sha256(pack_team(tuple(team)).encode()).hexdigest())
        raise ReferenceRefusal('particle necessary membership exhausted fixed original-proposal cap')
