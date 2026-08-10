"""A ``-transform`` ident is a NICKNAME, and a randbats forme is nicknamed after its base.

THE DEFECT. ``data/random-battles/gen3/teams.ts:619`` sets ``name: species.baseSpecies``,
so a generated ``Deoxys-Defense`` is *named* ``Deoxys``. ``sim/pokemon.ts:1345`` emits
``-transform`` with idents (``Pokemon.toString()`` -> ``fullname`` -> the nickname), so the
line reads ``|-transform|p2a: Ditto|p1a: Deoxys`` while the ``|switch|`` DETAILS that the
belief is built from read ``Deoxys-Defense, L74``.

``belief.ingest_event`` then read the two species of one event from two different sources:
the ACTOR's from the tracked belief, the TARGET's straight off the ident. That asymmetry --
not the sampler, not the renderer -- is the whole bug. It recorded
``transform_species='Deoxys'``; ``engine_world._apply_transform`` could not match ``deoxys``
against the party id ``deoxysdefense``, failed closed with ``transform_unexpressible``, and
the decision fell back on ``no_worlds_constructed``.

Measured: **24 truth-rejected decisions** on the census control block
``runE-ctrl512-s4-r1000`` (battle ``tdc-c0-9900352``, seed 9900352, round 85 / turn 77),
every one of them with ``truth_worlds_constructed == 0``. The refusal rejected the TRUE
world, which is a defect by construction.

AND THE MESSAGE WAS FALSE. It said the copied species was *"absent from the sampled
opposing party"* while the same record's ``truth_packed_teams`` showed the party holding
``Deoxys-Defense``. It pointed every reader at the sampler, which was correct, instead of
at the two ids, which did not match. ``TheRefusalMessageTests`` pins that the message now
states both sides of its own comparison, so it cannot make an unfalsifiable claim again.

WHY THE FIX IS ON THE BELIEF SIDE. Matching on BASE species inside ``_apply_transform``
would also clear this case, and it is the wrong fix twice over:

  * Deoxys-Attack is 180/20 where Deoxys-Defense is 70/160. A base match against a party
    holding more than one forme picks an arbitrary one and searches a world that never
    existed -- trading a loud refusal for a silent wrong answer.
  * ``transform_species`` has five other consumers (``showdown.py``'s encoder species,
    ``deep_line_audit``, ``public_projection``, ``determinization``, ``engine_search``).
    A local relaxation in ``engine_world`` leaves all of them holding the wrong species.

So the species is corrected where it is known, and the guard downstream stays EXACT.
``ASiblingFormeIsNotADonorTests`` is the pin on that: it holds the guard to the
strictly-more-conservative behaviour a base match would have destroyed.

AND THE UPGRADE IS BOUNDED. Substituting the tracked species for the ident is only sound
where the ident could be the base-name spelling of that same mon, so the upgrade requires
an equal BASE species and keeps the ident otherwise. A belief that disagrees with the
protocol at base level is desynchronised, and inventing a species from it would be exactly
the silent wrongness this guard exists to prevent. ``ADesynchronisedBeliefIsNotTrustedTests``
tests that NEW failure direction -- the one the fix opens, not the one it closes.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.belief import PublicBattleBeliefEngine, _base_species_id  # noqa: E402
from pokezero.dex import MoveInfo, ShowdownDex, SpeciesInfo, normalize_id  # noqa: E402
from pokezero.engine_world import (  # noqa: E402
    EngineWorldUnsupported,
    _engine_species_id,
    battle_spec_from_payload,
)
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.gen3_damage import gen3_hp_stat  # noqa: E402
from pokezero.showdown import _public_event_from_line  # noqa: E402
from pokezero.showdown_fixture import FixturePokemon, pack_team  # noqa: E402

from _showdown_root import requires_showdown, showdown_root, showdown_root_str  # noqa: E402


# ---------------------------------------------------------------------------
# The captured battle, verbatim from runE-ctrl512-s4-r1000 / tdc-c0-9900352.
# ---------------------------------------------------------------------------
CAPTURED_P1_PARTY = (
    "Charizard", "Piloswine", "Porygon2", "Deoxys-Defense", "Hariyama", "Stantler",
)
CAPTURED_P2_PARTY = ("Granbull", "Ditto", "Hypno", "Plusle", "Aggron", "Torkoal")

# The protocol shape. The `switch` DETAILS carry the forme; the `-transform` ident does not.
SWITCH_DEOXYS = "|switch|p1a: Deoxys|Deoxys-Defense, L74|196/196"
SWITCH_DITTO = "|switch|p2a: Ditto|Ditto, L100|258/258"
TRANSFORM_LINE = "|-transform|p2a: Ditto|p1a: Deoxys"


def _belief(*lines: str) -> PublicBattleBeliefEngine:
    return PublicBattleBeliefEngine.from_events(
        [_public_event_from_line(line) for line in lines], format_id="gen3randombattle"
    )


def _active(engine: PublicBattleBeliefEngine, slot: str):
    return next(mon for mon in engine.snapshot().side(slot) if mon.active)


# ---------------------------------------------------------------------------
# Construction fixtures. Real Deoxys base stats, because the point of the donor
# lookup is WHICH forme it finds: a sibling forme is not a survivable substitute.
# ---------------------------------------------------------------------------
_EVS = {stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")}

_DEOXYS_STATS = {
    "deoxys": {"hp": 50, "atk": 150, "def": 50, "spa": 150, "spd": 50, "spe": 150},
    "deoxysattack": {"hp": 50, "atk": 180, "def": 20, "spa": 180, "spd": 20, "spe": 150},
    "deoxysdefense": {"hp": 50, "atk": 70, "def": 160, "spa": 70, "spd": 160, "spe": 90},
}


def _move(move_id: str, pp: int) -> MoveInfo:
    return MoveInfo(
        id=move_id, name=move_id, type="normal", category="physical",
        gen3_category="physical", base_power=50, accuracy=100.0, priority=0,
        recoil=False, drain=False, heal=False, status=None, boosts={},
        target="normal", selfdestruct=False, pp=pp,
    )


def _species(species_id: str, name: str, types, base, weight: float) -> SpeciesInfo:
    return SpeciesInfo(
        id=species_id, name=name, types=types, base_stats=base, weight_kg=weight
    )


def _dex() -> ShowdownDex:
    species = {
        "ditto": _species("ditto", "Ditto", ("normal",),
                          {"hp": 48, "atk": 48, "def": 48, "spa": 48, "spd": 48, "spe": 48}, 4.0),
        "swampert": _species("swampert", "Swampert", ("water", "ground"),
                             {"hp": 100, "atk": 110, "def": 90, "spa": 85, "spd": 90, "spe": 60}, 81.9),
    }
    for species_id, name in (
        ("deoxys", "Deoxys"),
        ("deoxysattack", "Deoxys-Attack"),
        ("deoxysdefense", "Deoxys-Defense"),
    ):
        species[species_id] = _species(
            species_id, name, ("psychic",), dict(_DEOXYS_STATS[species_id]), 60.8
        )
    return ShowdownDex(
        moves={
            "transform": _move("transform", 10),
            "recover": _move("recover", 10),
            "toxic": _move("toxic", 10),
            "seismictoss": _move("seismictoss", 20),
            "spikes": _move("spikes", 20),
            "earthquake": _move("earthquake", 10),
            "surf": _move("surf", 15),
        },
        type_chart={},
        species=species,
    )


_DITTO = FixturePokemon(
    species="Ditto", moves=("transform",), ability="Limber", item="Leftovers",
    level=100, evs=dict(_EVS),
)
_SWAMPERT = FixturePokemon(
    species="Swampert", moves=("earthquake", "surf"), ability="Torrent",
    item="Leftovers", level=84, evs=dict(_EVS),
)


def _deoxys(forme: str) -> FixturePokemon:
    return FixturePokemon(
        species=forme, moves=("recover", "toxic", "seismictoss", "spikes"),
        ability="Pressure", item="Leftovers", level=74, evs=dict(_EVS),
    )


def _active_spec(world, slot: str):
    """The active ``PokemonSpec`` on ``slot`` of a built world."""
    side = getattr(world.spec, world.slot_sides[slot])
    return side.pokemon[side.active_index]


def _payload_donor_benched(dex: ShowdownDex, donor: FixturePokemon):
    """The donor has SWITCHED OUT since the copy. Transform outlives it.

    The engine still needs the donor's stats to express the transformed active, and the
    donor is still in the sampled party -- just not on the field. A donor lookup narrowed
    to the opposing ACTIVE would refuse this world, which is why it is pinned.
    """
    payload = _payload(dex, donor)
    p1 = payload["sides"]["p1"]
    swampert_hp = _maxhp(_SWAMPERT, dex)
    # Swampert leads; the donor is alive on the bench behind it.
    p1["pokemon"] = [
        {
            "species": "Swampert",
            "condition": f"{swampert_hp}/{swampert_hp}",
            "active": True,
            "moves": [
                {"id": "earthquake", "pp": 10, "maxpp": 16, "disabled": False},
                {"id": "surf", "pp": 15, "maxpp": 24, "disabled": False},
            ],
        },
        dict(p1["pokemon"][0], active=False),
    ]
    p1["lastUsedMove"] = "earthquake"
    return payload


def _maxhp(mon: FixturePokemon, dex: ShowdownDex) -> int:
    info = dex.species_info(mon.species)
    return gen3_hp_stat(
        int(info.base_stats["hp"]), 31, int((mon.evs or {}).get("hp", 0)), mon.level
    )


def _override(donor: FixturePokemon) -> BattleStartOverride:
    """We are p2 (the Ditto); p1 holds the donor. The captured seat exactly."""
    return BattleStartOverride(
        player_teams={
            "p1": pack_team((donor, _SWAMPERT)),
            "p2": pack_team((_DITTO, _SWAMPERT)),
        },
    )


def _payload(dex: ShowdownDex, donor: FixturePokemon):
    ditto_hp = _maxhp(_DITTO, dex)
    donor_hp = _maxhp(donor, dex)
    active_rows = [
        {"id": "recover", "pp": 10, "maxpp": 16, "disabled": False},
        {"id": "toxic", "pp": 10, "maxpp": 16, "disabled": False},
        {"id": "seismictoss", "pp": 20, "maxpp": 32, "disabled": False},
        {"id": "spikes", "pp": 20, "maxpp": 32, "disabled": False},
    ]
    return {
        "turn": 77,
        "weather": None,
        "weatherSetTurn": None,
        "weatherFromAbility": False,
        "futureSight": {"p1": 0, "p2": 0},
        "wishSetTurns": {},
        "leechSeedSourceSides": {},
        "pendingBatonPassSides": [],
        "deferredOpponentActions": {},
        "deferredOpponentActionPriors": {},
        "selfPlayer": "p2",
        "selfRequestKind": "move",
        "selfTeamOrder": ["Ditto", "Swampert"],
        "selfActiveRequestState": {
            "trapped": False, "maybeTrapped": False,
            "maybeDisabled": False, "maybeLocked": False,
        },
        "selfBenchedMoveHistory": False,
        "selfActiveMoves": list(active_rows),
        "sides": {
            "p1": {
                "pokemon": [
                    {
                        "species": donor.species,
                        "condition": f"{donor_hp}/{donor_hp}",
                        "active": True,
                        "moves": list(active_rows),
                    },
                    {"species": "Swampert", "condition": "0 fnt", "active": False, "moves": []},
                ],
                "boosts": {},
                "volatiles": [],
                "lastUsedMove": "recover",
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
            "p2": {
                "pokemon": [
                    {
                        "species": "Ditto",
                        "condition": f"{ditto_hp}/{ditto_hp}",
                        "active": True,
                        "moves": [
                            {"id": "transform", "pp": 15, "maxpp": 16, "disabled": False},
                        ],
                    },
                    {"species": "Swampert", "condition": "0 fnt", "active": False, "moves": []},
                ],
                "boosts": {},
                "volatiles": [],
                "lastUsedMove": "transform",
                "materializationBlockers": [],
                "toxicStage": 0,
                "sideConditions": {},
                "sideConditionSetTurns": {},
            },
        },
    }


class TheProtocolReallyNamesTheFormeAfterItsBaseTests(unittest.TestCase):
    """The producer fact, read from the vendored Showdown source, not restated.

    If upstream ever stops nicknaming a randbats forme after its base species, the
    correction below becomes dead code and this class says so.
    """

    @requires_showdown("needs a pokemon-showdown checkout to read teams.ts")
    def test_gen3_randbats_names_a_forme_after_its_base_species(self) -> None:
        teams = (showdown_root() / "data" / "random-battles" / "gen3" / "teams.ts").read_text()
        self.assertIn(
            "name: species.baseSpecies,",
            teams,
            "gen3 randbats no longer nicknames a forme after its base species; "
            "the ident seam this module exists for may be gone",
        )

    @requires_showdown("needs a pokemon-showdown checkout to read sim/pokemon.ts")
    def test_the_transform_line_is_emitted_with_idents(self) -> None:
        pokemon_ts = (showdown_root() / "sim" / "pokemon.ts").read_text()
        self.assertIn(
            "this.battle.add('-transform', this, pokemon);",
            pokemon_ts,
            "-transform no longer passes Pokemon objects (which serialize as idents)",
        )

    @requires_showdown("needs a pokemon-showdown checkout to load the gen3 randbat pool")
    def test_the_reachable_population_is_bounded_and_non_empty(self) -> None:
        """Anti-vacuity, and a staleness alarm on the two figures the PR reports.

        Counted SEPARATELY and never summed: the predicate needs a Transform carrier on
        one side AND a base-nicknamed forme on the OTHER side of the same battle.
        """
        import re

        from pokezero.randbat import load_gen3_randbat_source_cached

        # `baseSpecies` read from the producer, `data/pokedex.ts`, rather than guessed from
        # the spelling: the randbat pool stores `Deoxysdefense`, which no hyphen rule finds.
        dex_text = (showdown_root() / "data" / "pokedex.ts").read_text()
        base_of: dict[str, tuple[str, str]] = {}
        for entry in re.finditer(r"^\t(\w+): \{(.*?)^\t\},", dex_text, re.S | re.M):
            body = entry.group(2)
            name = re.search(r'^\t\tname: "(.*?)",', body, re.M)
            base = re.search(r'^\t\tbaseSpecies: "(.*?)",', body, re.M)
            if name is not None:
                base_of[entry.group(1)] = (
                    name.group(1), base.group(1) if base is not None else name.group(1)
                )

        source = load_gen3_randbat_source_cached(showdown_root_str())
        variants = [v for uni in source.universes.values() for v in uni.variants]
        self.assertEqual(len(variants), 1682, "the gen3 randbat pool moved")
        self.assertEqual(
            [v.species for v in variants if normalize_id(v.species) not in base_of], [],
            "a randbat species is not in pokedex.ts; the derivation below is incomplete",
        )
        formes = [
            v for v in variants
            if normalize_id(base_of[normalize_id(v.species)][1])
            != normalize_id(base_of[normalize_id(v.species)][0])
        ]
        transformers = [
            v for v in variants
            if "transform" in {m.lower().replace(" ", "") for m in v.moves}
        ]
        self.assertEqual(len(formes), 13, "forme-carrier count moved")
        self.assertEqual(len(transformers), 7, "Transform-carrier count moved")
        self.assertEqual(
            sorted({v.species for v in transformers}), ["Ditto", "Mew"]
        )
        self.assertEqual(
            sorted({v.species for v in formes}),
            ["Deoxysattack", "Deoxysdefense", "Deoxysspeed"],
            "Deoxys is no longer the only base-nicknamed forme family in gen3 randbats",
        )


class TheIdentSeamIsRealTests(unittest.TestCase):
    """Fixture guards. Without these the fix could be asserted against a fake input."""

    def test_the_switch_details_carry_the_forme_but_the_ident_does_not(self) -> None:
        switch = _public_event_from_line(SWITCH_DEOXYS)
        transform = _public_event_from_line(TRANSFORM_LINE)
        self.assertEqual(switch.primary, "Deoxys-Defense")
        self.assertEqual(switch.actor_ident, "p1a: Deoxys")
        # This is the input the old code read the target species from.
        self.assertEqual(transform.primary, "p1a: Deoxys")
        self.assertEqual(transform.actor_slot, "p2")

    def test_the_two_spellings_are_different_engine_ids(self) -> None:
        """Without this the donor lookup could 'work' for the wrong reason."""
        self.assertNotEqual(
            _engine_species_id(normalize_id("Deoxys")),
            _engine_species_id(normalize_id("Deoxys-Defense")),
        )

    def test_the_captured_party_really_contained_the_copied_mon(self) -> None:
        """The claim that made the old message self-disproving."""
        self.assertIn("Deoxys-Defense", CAPTURED_P1_PARTY)
        self.assertIn("Ditto", CAPTURED_P2_PARTY)
        # Species Clause on BASE species, which is the claim that distinguishes this from
        # the known precedent artifact (four Deoxys FORMES on one team, whose idents
        # Showdown collapsed). Exactly one Deoxys forme exists in the whole battle, and it
        # is on the OPPOSING side -- so no base match could have been ambiguous here, and
        # the refusal was not the sampler producing an impossible party.
        self.assertEqual(
            len({_base_species_id(s) for s in CAPTURED_P1_PARTY}), 6,
            "the captured p1 party is six distinct BASE species",
        )
        self.assertEqual(
            [s for s in CAPTURED_P1_PARTY + CAPTURED_P2_PARTY if _base_species_id(s) == "deoxys"],
            ["Deoxys-Defense"],
            "exactly one Deoxys forme in the whole battle; not the four-forme artifact",
        )


class BeliefResolvesTheTransformTargetFormeTests(unittest.TestCase):
    """RED before the fix: ``transform_species`` was ``'Deoxys'``."""

    def test_a_base_named_ident_resolves_to_the_tracked_forme(self) -> None:
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        ditto = _active(engine, "p2")
        self.assertTrue(ditto.transformed)
        self.assertEqual(ditto.transform_species, "Deoxys-Defense")

    def test_the_actor_and_the_target_species_come_from_the_same_source(self) -> None:
        """The defect stated as an invariant, not as a value.

        Both halves of a ``-transform`` are the belief's tracked species for their slot.
        A future edit that re-reads either one off the ident reddens here.
        """
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        self.assertEqual(_active(engine, "p2").species, _active(engine, "p2").species)
        self.assertEqual(_active(engine, "p2").transform_species, _active(engine, "p1").species)

    def test_the_correction_is_counted(self) -> None:
        """A refusal that stops firing does not say the replacing path ever ran."""
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        self.assertEqual(
            dict(engine.transform_forme_corrections), {"Deoxys->Deoxys-Defense": 1}
        )

    def test_an_exact_ident_is_not_counted_as_a_correction(self) -> None:
        """The counter must be a signal, not a tally of every Transform."""
        engine = _belief(
            "|switch|p1a: Blissey|Blissey, L79|300/300",
            SWITCH_DITTO,
            "|-transform|p2a: Ditto|p1a: Blissey",
        )
        self.assertEqual(_active(engine, "p2").transform_species, "Blissey")
        self.assertEqual(dict(engine.transform_forme_corrections), {})

    def test_repeated_corrections_accumulate(self) -> None:
        engine = _belief(
            SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE,
            "|switch|p2a: Ditto|Ditto, L100|258/258",
            TRANSFORM_LINE,
        )
        self.assertEqual(
            dict(engine.transform_forme_corrections), {"Deoxys->Deoxys-Defense": 2}
        )

    def test_the_counter_survives_a_clone(self) -> None:
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        twin = engine.clone()
        self.assertEqual(
            dict(twin.transform_forme_corrections), {"Deoxys->Deoxys-Defense": 1}
        )
        # ... and is a copy, not the same dict.
        engine.ingest_event(_public_event_from_line(TRANSFORM_LINE))
        self.assertEqual(dict(twin.transform_forme_corrections)["Deoxys->Deoxys-Defense"], 1)

    def test_the_evidence_line_names_the_forme(self) -> None:
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        details = [e.detail for e in _active(engine, "p2").evidence if e.kind == "transform"]
        self.assertEqual(len(details), 1)
        self.assertIn("Deoxys-Defense", details[0])

    def test_the_copied_ability_still_comes_from_the_target_belief(self) -> None:
        """Regression guard: the fix moved where ``target_belief`` is fetched."""
        engine = _belief(
            SWITCH_DEOXYS,
            "|-ability|p1a: Deoxys|Pressure",
            SWITCH_DITTO,
            TRANSFORM_LINE,
        )
        ditto = _active(engine, "p2")
        self.assertEqual(ditto.transform_species, "Deoxys-Defense")
        self.assertEqual(
            _active(engine, "p1").revealed_ability, "Pressure",
            "fixture guard: the target must actually have a revealed ability",
        )
        # The ability is still copied off the TARGET's belief, which is the object the
        # fix hoisted above the species read.
        self.assertEqual(engine._running_ability.get(ditto.key), "Pressure")

    def test_the_overlay_payload_key_engine_search_reads_carries_the_forme(self) -> None:
        """The exact hop: ``engine_search`` builds ``transformed_slots`` from this key."""
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        row = _active(engine, "p2").to_overlay_payload()
        self.assertTrue(row["transformed"])
        self.assertEqual(row["transform_species"], "Deoxys-Defense")


class ADesynchronisedBeliefIsNotTrustedTests(unittest.TestCase):
    """THE NEW FAILURE DIRECTION. The fix relaxes nothing it cannot bound.

    Substituting the tracked species for the ident is sound only where the ident could be
    the base-name spelling of that same mon. Where the belief and the protocol disagree at
    BASE level the belief is desynchronised, and using it would put a species nobody named
    into a searched world. The ident is kept and the guard downstream still fails closed --
    which is what the code did before the fix existed.
    """

    def test_a_different_base_species_is_never_substituted(self) -> None:
        engine = _belief(
            "|switch|p1a: Blissey|Blissey, L79|300/300",
            SWITCH_DITTO,
            # The protocol names a mon the belief does not have on the field.
            "|-transform|p2a: Ditto|p1a: Deoxys",
        )
        self.assertEqual(
            _active(engine, "p2").transform_species, "Deoxys",
            "the ident must survive a belief that disagrees at base level",
        )
        self.assertEqual(dict(engine.transform_forme_corrections), {})

    def test_an_untracked_target_slot_keeps_the_ident(self) -> None:
        """No belief at all -- the pre-fix path, unchanged."""
        engine = _belief(SWITCH_DITTO, TRANSFORM_LINE)
        self.assertEqual(_active(engine, "p2").transform_species, "Deoxys")
        self.assertEqual(dict(engine.transform_forme_corrections), {})

    def test_a_hyphenated_base_species_is_not_mistaken_for_a_forme(self) -> None:
        """``Ho-Oh`` is a base species, not ``Ho`` plus a forme suffix."""
        engine = _belief(
            "|switch|p1a: Ho-Oh|Ho-Oh, L73|300/300",
            SWITCH_DITTO,
            "|-transform|p2a: Ditto|p1a: Ho-Oh",
        )
        self.assertEqual(_active(engine, "p2").transform_species, "Ho-Oh")
        self.assertEqual(dict(engine.transform_forme_corrections), {})


class TheFormeDonorIsFoundAndCopiedTests(unittest.TestCase):
    """Direction 1: the true world CONSTRUCTS -- and holds the right forme's stats."""

    def setUp(self) -> None:
        self.dex = _dex()

    def _build(self, *, donor: FixturePokemon, target: str):
        return battle_spec_from_payload(
            _payload(self.dex, donor),
            _override(donor),
            dex=self.dex,
            transformed_slots={"p2": target},
        )

    def test_the_base_name_is_what_used_to_refuse(self) -> None:
        """The pre-fix input, held red on purpose.

        Without this the green below could come from a fixture that never had the
        problem. This is the exact ``transformed_slots`` value the old belief produced.
        """
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(donor=_deoxys("Deoxys-Defense"), target="Deoxys")
        self.assertEqual(caught.exception.reason, "transform_unexpressible")

    def test_the_forme_name_constructs(self) -> None:
        world = self._build(donor=_deoxys("Deoxys-Defense"), target="Deoxys-Defense")
        self.assertEqual(world.slot_sides, {"p1": "side_one", "p2": "side_two"})
        self.assertEqual(_active_spec(world, "p2").id, "deoxysdefense")

    def test_the_copied_stats_are_the_formes_own(self) -> None:
        """Direction 2, locally: constructing is not enough, it must be the RIGHT mon.

        Deoxys-Defense is 70/160 where Deoxys-Attack is 180/20, so a donor resolved by
        base species instead of forme is visible here as a factor-of-two stat error.
        """
        world = self._build(donor=_deoxys("Deoxys-Defense"), target="Deoxys-Defense")
        ditto = _active_spec(world, "p2")
        donor = _active_spec(world, "p1")
        self.assertEqual(ditto.attack, donor.attack)
        self.assertEqual(ditto.defense, donor.defense)
        self.assertEqual(ditto.speed, donor.speed)
        self.assertEqual(ditto.types, ("psychic",))
        # And they really are the Defense forme's, not some other Deoxys'.
        other = _active_spec(
            self._build(donor=_deoxys("Deoxys-Attack"), target="Deoxys-Attack"), "p2"
        )
        self.assertNotEqual(ditto.attack, other.attack)
        self.assertNotEqual(ditto.defense, other.defense)

    def test_a_benched_donor_is_still_a_donor(self) -> None:
        """The copy outlives the donor switching out.

        Found by a SAFER-direction mutant: narrowing the donor lookup to the opposing
        ACTIVE survived the whole battery, meaning nothing pinned this. It does now.
        """
        donor = _deoxys("Deoxys-Defense")
        world = battle_spec_from_payload(
            _payload_donor_benched(self.dex, donor),
            _override(donor),
            dex=self.dex,
            transformed_slots={"p2": "Deoxys-Defense"},
        )
        donor_side = getattr(world.spec, world.slot_sides["p1"])
        self.assertNotEqual(
            donor_side.pokemon[donor_side.active_index].id, "deoxysdefense",
            "fixture guard: the donor must actually be OFF the field for this to pin anything",
        )
        self.assertEqual(_active_spec(world, "p2").id, "deoxysdefense")

    def test_the_transformer_keeps_its_own_base_identity(self) -> None:
        """Unchanged behaviour, re-pinned on the forme path."""
        ditto = _active_spec(
            self._build(donor=_deoxys("Deoxys-Defense"), target="Deoxys-Defense"), "p2"
        )
        self.assertEqual(ditto.base_types, ("normal",))
        self.assertIsNotNone(ditto.pre_transform)
        self.assertEqual(ditto.pre_transform.id, "ditto")


class ASiblingFormeIsNotADonorTests(unittest.TestCase):
    """THE SAFER-DIRECTION PIN. The guard stays EXACT; it did not become a base match.

    A base-species match in ``_apply_transform`` would have cleared the census defect
    without touching ``belief.py``. It is refused, and this class is what refuses it: a
    strictly-more-conservative guard must SURVIVE these rows. If a later edit relaxes the
    donor lookup to base species, every test here goes red.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def _build(self, *, donor: FixturePokemon, target: str):
        return battle_spec_from_payload(
            _payload(self.dex, donor),
            _override(donor),
            dex=self.dex,
            transformed_slots={"p2": target},
        )

    def test_a_defense_copy_does_not_settle_for_an_attack_forme(self) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(donor=_deoxys("Deoxys-Attack"), target="Deoxys-Defense")
        self.assertEqual(caught.exception.reason, "transform_unexpressible")

    def test_a_normal_forme_copy_does_not_settle_for_a_defense_forme(self) -> None:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(donor=_deoxys("Deoxys-Defense"), target="Deoxys")
        self.assertEqual(caught.exception.reason, "transform_unexpressible")

    def test_an_unrelated_species_still_refuses(self) -> None:
        # `Ditto` is the TRANSFORMER's own species and is on the other side, so it is
        # genuinely absent from the donor party -- unlike Swampert, which is in it.
        with self.assertRaises(EngineWorldUnsupported) as caught:
            self._build(donor=_deoxys("Deoxys-Defense"), target="Ditto")
        self.assertEqual(caught.exception.reason, "transform_unexpressible")


class TheRefusalMessageTests(unittest.TestCase):
    """The message was FALSE, and a false refusal message costs debugging time twice.

    It asserted the copied species was "absent from the sampled opposing party" for 24
    decisions whose own record showed the party holding it. The rule taken from that: a
    refusal message states the INPUTS of the comparison it failed, never a conclusion
    about them.
    """

    def setUp(self) -> None:
        self.dex = _dex()

    def _refusal(self, *, donor: FixturePokemon, target: str) -> str:
        with self.assertRaises(EngineWorldUnsupported) as caught:
            battle_spec_from_payload(
                _payload(self.dex, donor),
                _override(donor),
                dex=self.dex,
                transformed_slots={"p2": target},
            )
        return str(caught.exception)

    def test_the_message_names_both_sides_of_the_comparison(self) -> None:
        message = self._refusal(donor=_deoxys("Deoxys-Defense"), target="Deoxys")
        self.assertIn("deoxys", message)
        self.assertIn("deoxysdefense", message)

    def test_the_message_makes_no_unfalsifiable_absence_claim(self) -> None:
        message = self._refusal(donor=_deoxys("Deoxys-Defense"), target="Deoxys")
        self.assertNotIn(
            "absent from the sampled opposing party", message,
            "the retired claim was false on the very record that surfaced it",
        )

    def test_the_message_is_self_checking(self) -> None:
        """Its own listed party must really lack its own listed target id.

        This is the property the old message did not have: a reader could not confirm or
        refute it from the message alone, and it happened to be wrong.
        """
        message = self._refusal(donor=_deoxys("Deoxys-Defense"), target="Deoxys")
        target_id = _engine_species_id(normalize_id("Deoxys"))
        listed = message.split("opposing party", 1)[1]
        self.assertIn(repr("deoxysdefense"), listed)
        self.assertNotIn(repr(target_id), listed)


class TheWholeChainOnTheCapturedDecisionTests(unittest.TestCase):
    """Parser -> belief -> the overlay key -> construction, on the captured shape.

    Each class above pins one seam. This one is the census decision itself: the reason
    the four separately-correct seams produced a refusal was the hand-off between them.
    """

    def test_the_captured_decision_now_constructs(self) -> None:
        engine = _belief(SWITCH_DEOXYS, SWITCH_DITTO, TRANSFORM_LINE)
        target = _active(engine, "p2").to_overlay_payload()["transform_species"]
        dex = _dex()
        donor = _deoxys("Deoxys-Defense")
        world = battle_spec_from_payload(
            _payload(dex, donor), _override(donor), dex=dex,
            transformed_slots={"p2": str(target)},
        )
        self.assertEqual(_active_spec(world, "p2").id, "deoxysdefense")
        self.assertEqual(dict(engine.transform_forme_corrections), {"Deoxys->Deoxys-Defense": 1})


if __name__ == "__main__":
    unittest.main()
