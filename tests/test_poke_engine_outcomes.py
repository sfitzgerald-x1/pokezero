from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
import unittest

from pokezero.dex import showdown_dex_from_payload
from pokezero.poke_engine_adapter import PokemonSpec, SideSpec
from pokezero.poke_engine_outcomes import (
    ActiveStateComparison,
    ActiveStateSummary,
    DirectCalculateDamageDiagnostic,
    EngineActiveStateSummary,
    EngineBranchDamageSummary,
    EngineBranchOutcome,
    ENGINE_FIELD_UNREAD,
    ENGINE_STATE_COMPARED_FIELDS,
    FieldMismatch,
    ObservedDamage,
    OneTurnDamageDiagnostic,
    OutcomeComparison,
    _mismatch_surface_and_notes,
    active_state_summary_from_side,
    build_battle_spec_from_result,
    build_one_turn_damage_diagnostic,
    compare_engine_active_state,
    compare_outcomes,
    direct_calculate_damage_diagnostic,
    engine_active_state_summary,
    engine_move_for_choice,
    enumerate_engine_outcomes,
    observed_damage_from_result,
    observed_final_active_hp,
    opening_active_hp,
    parse_condition,
    parse_details,
    pokemon_spec_from_request_member,
    side_spec_from_request,
)
from pokezero.poke_engine_backend import POKE_ENGINE_SUPPORTED_VERSION, probe_poke_engine
from pokezero.showdown_fixture import OneTurnFixtureResult


# A minimal Gen 3 dex carrying just the types the extraction tests need; built
# without node/Showdown so these tests never touch the native toolchain.
def minimal_dex():
    return showdown_dex_from_payload(
        {
            "moves": {},
            "species": {
                "charmander": {"id": "charmander", "name": "Charmander", "types": ["Fire"], "baseStats": {}},
                "squirtle": {"id": "squirtle", "name": "Squirtle", "types": ["Water"], "baseStats": {}},
            },
            "typeChart": {},
        }
    )


# The real opening requests + damage protocol captured from the seeded
# Charmander/Ember vs. Squirtle/Water Gun fixture (seed 7). Showdown's final active
# HP for this seeded turn is Charmander 127/219 and Squirtle 209/229.
def charmander_squirtle_result():
    p1_request = {
        "active": [
            {
                "moves": [
                    {"move": "Ember", "id": "ember", "pp": 40, "maxpp": 40, "target": "normal", "disabled": False},
                    {"move": "Tackle", "id": "tackle", "pp": 56, "maxpp": 56, "target": "normal", "disabled": False},
                ]
            }
        ],
        "side": {
            "name": "PokeZero p1",
            "id": "p1",
            "pokemon": [
                {
                    "ident": "p1: Charmander",
                    "details": "Charmander, M",
                    "condition": "219/219",
                    "active": True,
                    "stats": {"atk": 140, "def": 122, "spa": 156, "spd": 136, "spe": 166},
                    "moves": ["ember", "tackle"],
                    "baseAbility": "blaze",
                    "item": "",
                    "pokeball": "pokeball",
                }
            ],
        },
    }
    p2_request = {
        "active": [
            {
                "moves": [
                    {"move": "Water Gun", "id": "watergun", "pp": 40, "maxpp": 40, "target": "normal", "disabled": False},
                    {"move": "Tackle", "id": "tackle", "pp": 56, "maxpp": 56, "target": "normal", "disabled": False},
                ]
            }
        ],
        "side": {
            "name": "PokeZero p2",
            "id": "p2",
            "pokemon": [
                {
                    "ident": "p2: Squirtle",
                    "details": "Squirtle, F",
                    "condition": "229/229",
                    "active": True,
                    "stats": {"atk": 132, "def": 166, "spa": 136, "spd": 164, "spe": 122},
                    "moves": ["watergun", "tackle"],
                    "baseAbility": "torrent",
                    "item": "",
                    "pokeball": "pokeball",
                }
            ],
        },
    }
    protocol_lines = (
        "|switch|p1a: Charmander|Charmander, M|219/219",
        "|switch|p2a: Squirtle|Squirtle, F|229/229",
        "|turn|1",
        "|move|p1a: Charmander|Ember|p2a: Squirtle",
        "|-resisted|p2a: Squirtle",
        "|-damage|p2a: Squirtle|209/229",
        "|move|p2a: Squirtle|Water Gun|p1a: Charmander",
        "|-supereffective|p1a: Charmander",
        "|-damage|p1a: Charmander|127/219",
    )
    return OneTurnFixtureResult(
        format_id="gen3customgame",
        seed=7,
        choices={"p1": "move 1", "p2": "move 1"},
        protocol_lines=protocol_lines,
        p1_request=p1_request,
        p2_request=p2_request,
        terminal=False,
    )


# ---- details / condition parsing -----------------------------------------


class ParseDetailsConditionTest(unittest.TestCase):
    def test_parse_details_defaults_level_100_without_marker(self) -> None:
        self.assertEqual(parse_details("Charmander, M"), ("charmander", 100))

    def test_parse_details_reads_level_marker(self) -> None:
        self.assertEqual(parse_details("Charizard, L78"), ("charizard", 78))
        self.assertEqual(parse_details("Pikachu, L50, M"), ("pikachu", 50))

    def test_parse_details_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_details("")

    def test_parse_condition_full_and_partial(self) -> None:
        self.assertEqual(parse_condition("219/219"), (219, 219))
        self.assertEqual(parse_condition("127/219 par"), (127, 219))

    def test_parse_condition_fainted_has_no_max(self) -> None:
        self.assertEqual(parse_condition("0 fnt"), (0, None))
        self.assertEqual(parse_condition("0"), (0, None))


# ---- request -> BattleSpec / stat extraction ------------------------------


class RequestToBattleSpecTest(unittest.TestCase):
    def test_pokemon_spec_pulls_stats_hp_types_ability_from_request(self) -> None:
        result = charmander_squirtle_result()
        member = result.p1_request["side"]["pokemon"][0]
        active_row = result.p1_request["active"][0]

        spec = pokemon_spec_from_request_member(member, minimal_dex(), active_row=active_row)

        self.assertIsInstance(spec, PokemonSpec)
        self.assertEqual(spec.id, "charmander")
        self.assertEqual(spec.level, 100)
        self.assertEqual(spec.types, ("fire",))
        self.assertEqual((spec.hp, spec.maxhp), (219, 219))
        # Real computed stats from the opening request, mapped onto poke-engine fields.
        self.assertEqual(spec.attack, 140)
        self.assertEqual(spec.defense, 122)
        self.assertEqual(spec.special_attack, 156)
        self.assertEqual(spec.special_defense, 136)
        self.assertEqual(spec.speed, 166)
        self.assertEqual(spec.ability, "blaze")
        # Empty item string normalizes to "no item" (None).
        self.assertIsNone(spec.item)
        self.assertEqual([m.id for m in spec.moves], ["ember", "tackle"])
        # PP is carried from the active row.
        self.assertEqual(spec.moves[0].pp, 40)

    def test_side_spec_marks_active_index(self) -> None:
        result = charmander_squirtle_result()
        side = side_spec_from_request(result.p2_request, minimal_dex())
        self.assertIsInstance(side, SideSpec)
        self.assertEqual(side.active_index, 0)
        self.assertEqual([p.id for p in side.pokemon], ["squirtle"])
        self.assertEqual(side.pokemon[0].special_attack, 136)

    def test_build_battle_spec_from_result_builds_both_sides(self) -> None:
        spec = build_battle_spec_from_result(charmander_squirtle_result(), minimal_dex())
        self.assertEqual(spec.side_one.pokemon[0].id, "charmander")
        self.assertEqual(spec.side_two.pokemon[0].id, "squirtle")

    def test_missing_stats_block_is_rejected(self) -> None:
        result = charmander_squirtle_result()
        member = dict(result.p1_request["side"]["pokemon"][0])
        member.pop("stats")
        with self.assertRaises(ValueError):
            pokemon_spec_from_request_member(member, minimal_dex())

    def test_unknown_species_types_rejected(self) -> None:
        result = charmander_squirtle_result()
        member = dict(result.p1_request["side"]["pokemon"][0])
        member["details"] = "Missingno, M"
        with self.assertRaises(ValueError):
            pokemon_spec_from_request_member(member, minimal_dex())


# ---- choice -> engine move id --------------------------------------------


class EngineMoveForChoiceTest(unittest.TestCase):
    def test_numeric_choice_indexes_active_moves(self) -> None:
        request = charmander_squirtle_result().p1_request
        self.assertEqual(engine_move_for_choice(request, "move 1"), "ember")
        self.assertEqual(engine_move_for_choice(request, "move 2"), "tackle")

    def test_named_choice_normalizes(self) -> None:
        self.assertEqual(engine_move_for_choice(None, "move Water Gun"), "watergun")

    def test_named_choice_rejects_move_absent_from_request(self) -> None:
        request = charmander_squirtle_result().p1_request
        with self.assertRaises(ValueError):
            engine_move_for_choice(request, "move Thunderbolt")

    def test_switch_choice_rejected(self) -> None:
        with self.assertRaises(ValueError):
            engine_move_for_choice(None, "switch 2")

    def test_out_of_range_numeric_rejected(self) -> None:
        request = charmander_squirtle_result().p1_request
        with self.assertRaises(ValueError):
            engine_move_for_choice(request, "move 9")


# ---- observed Showdown HP -------------------------------------------------


class ObservedFinalActiveHpTest(unittest.TestCase):
    def test_reads_switch_hp_from_fourth_protocol_payload_field(self) -> None:
        result = OneTurnFixtureResult(
            format_id="gen3customgame",
            seed=7,
            choices={},
            protocol_lines=(
                "|switch|p1a: Charmander|Charmander, M|219/219",
                "|switch|p2a: Squirtle|Squirtle, F|229/229",
            ),
            p1_request=None,
            p2_request=None,
            terminal=False,
        )
        self.assertEqual(observed_final_active_hp(result), (219, 229))

    def test_replace_updates_hp_from_fourth_protocol_payload_field(self) -> None:
        result = OneTurnFixtureResult(
            format_id="gen3customgame",
            seed=7,
            choices={},
            protocol_lines=(
                "|switch|p1a: Charmander|Charmander, M|219/219",
                "|switch|p2a: Squirtle|Squirtle, F|229/229",
                "|replace|p2a: Squirtle|Squirtle, F|200/229",
            ),
            p1_request=None,
            p2_request=None,
            terminal=False,
        )
        self.assertEqual(observed_final_active_hp(result), (219, 200))

    def test_reads_last_damage_for_each_seat(self) -> None:
        self.assertEqual(observed_final_active_hp(charmander_squirtle_result()), (127, 209))

    def test_faint_sets_hp_zero(self) -> None:
        result = charmander_squirtle_result()
        lines = result.protocol_lines + ("|faint|p1a: Charmander",)
        result = OneTurnFixtureResult(
            format_id=result.format_id,
            seed=result.seed,
            choices=result.choices,
            protocol_lines=lines,
            p1_request=result.p1_request,
            p2_request=result.p2_request,
            terminal=result.terminal,
        )
        self.assertEqual(observed_final_active_hp(result), (0, 209))

    def test_missing_seat_hp_raises(self) -> None:
        result = OneTurnFixtureResult(
            format_id="gen3customgame",
            seed=7,
            choices={},
            protocol_lines=("|switch|p1a: Charmander|Charmander, M|219/219",),
            p1_request=None,
            p2_request=None,
            terminal=False,
        )
        with self.assertRaises(ValueError):
            observed_final_active_hp(result)


# ---- engine outcome enumeration (fake engine) -----------------------------


def fake_branch(p1_hp: int, p2_hp: int, percentage: float, instructions: str) -> SimpleNamespace:
    return SimpleNamespace(percentage=percentage, instruction_list=instructions, _hp=(p1_hp, p2_hp))


class FakeState:
    """A navigable engine-style state whose apply_instructions sets active HP."""

    def __init__(self, p1_hp: int = 100, p2_hp: int = 100) -> None:
        self.side_one = SimpleNamespace(active_index=0, pokemon=[SimpleNamespace(hp=p1_hp)])
        self.side_two = SimpleNamespace(active_index=0, pokemon=[SimpleNamespace(hp=p2_hp)])

    def apply_instructions(self, branch: SimpleNamespace) -> "FakeState":
        p1_hp, p2_hp = branch._hp
        return FakeState(p1_hp=p1_hp, p2_hp=p2_hp)


def fake_engine_module(branches):
    module = ModuleType("poke_engine_fake_outcomes")
    module.generate_instructions = lambda state, m1, m2: list(branches)
    return module


class EnumerateEngineOutcomesTest(unittest.TestCase):
    def test_applies_each_branch_and_reads_final_hp(self) -> None:
        branches = [
            fake_branch(122, 207, 79.1, "Damage A"),
            fake_branch(25, 207, 5.27, "Crit A"),
        ]
        engine = fake_engine_module(branches)
        outcomes = enumerate_engine_outcomes(engine, FakeState(), "ember", "watergun")

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0].final_hp, (122, 207))
        self.assertEqual(outcomes[0].percentage, 79.1)
        self.assertEqual(outcomes[0].description, "Damage A")
        self.assertEqual(outcomes[1].final_hp, (25, 207))


# ---- comparison matching --------------------------------------------------


class CompareOutcomesTest(unittest.TestCase):
    def test_match_when_observed_tuple_is_an_engine_branch(self) -> None:
        outcomes = [
            EngineBranchOutcome(79.1, (127, 209), "a"),
            EngineBranchOutcome(5.0, (25, 209), "b"),
        ]
        result = compare_outcomes((127, 209), outcomes, p1_move="ember", p2_move="watergun")
        self.assertTrue(result.supported)
        self.assertTrue(result.matched)
        self.assertEqual(result.showdown_final_hp, (127, 209))
        self.assertEqual(result.p1_move, "ember")

    def test_mismatch_when_observed_tuple_absent(self) -> None:
        outcomes = [
            EngineBranchOutcome(79.1, (122, 207), "a"),
            EngineBranchOutcome(5.0, (25, 207), "b"),
        ]
        result = compare_outcomes((127, 209), outcomes)
        self.assertTrue(result.supported)
        self.assertFalse(result.matched)
        self.assertEqual(result.engine_final_hp_tuples(), ((122, 207), (25, 207)))

    def test_to_dict_is_serializable(self) -> None:
        import json

        outcomes = [EngineBranchOutcome(79.1, (122, 207), "Damage")]
        result = compare_outcomes((127, 209), outcomes, p1_move="ember", p2_move="watergun")
        payload = result.to_dict()
        # Round-trips through JSON without custom encoders.
        json.dumps(payload)
        self.assertFalse(payload["matched"])
        self.assertEqual(payload["showdown_final_hp"], [127, 209])
        self.assertEqual(payload["engine_final_hp_outcomes"][0]["final_hp"], [122, 207])

    def test_unsupported_factory_shape(self) -> None:
        result = OutcomeComparison(
            supported=False,
            matched=False,
            showdown_final_hp=None,
            engine_final_hp_outcomes=(),
            reason="engine unavailable",
        )
        self.assertFalse(result.supported)
        self.assertIn("UNSUPPORTED", result.summary())


# ---- damage diagnostic: observed deltas -----------------------------------


class ObservedDamageTest(unittest.TestCase):
    def test_observed_deltas_from_opening_request_hp(self) -> None:
        observed = observed_damage_from_result(charmander_squirtle_result())
        self.assertIsInstance(observed, ObservedDamage)
        self.assertEqual(observed.opening_hp, (219, 229))
        self.assertEqual(observed.final_hp, (127, 209))
        # 219-127 = 92 on Charmander; 229-209 = 20 on Squirtle.
        self.assertEqual(observed.deltas, (92, 20))

    def test_opening_active_hp_reads_request_condition(self) -> None:
        self.assertEqual(opening_active_hp(charmander_squirtle_result()), (219, 229))

    def test_observed_damage_to_dict_is_serializable(self) -> None:
        import json

        payload = observed_damage_from_result(charmander_squirtle_result()).to_dict()
        json.dumps(payload)
        self.assertEqual(payload["deltas"], [92, 20])


# ---- damage diagnostic: request-derived active state ----------------------


class ActiveStateSummaryTest(unittest.TestCase):
    def test_summary_is_request_derived_no_engine_required(self) -> None:
        spec = build_battle_spec_from_result(charmander_squirtle_result(), minimal_dex())
        summary = active_state_summary_from_side(spec.side_one)

        self.assertIsInstance(summary, ActiveStateSummary)
        self.assertEqual(summary.species, "charmander")
        self.assertEqual(summary.level, 100)
        self.assertEqual((summary.hp, summary.maxhp), (219, 219))
        self.assertEqual(summary.types, ("fire",))
        self.assertEqual(summary.ability, "blaze")
        self.assertIsNone(summary.item)
        self.assertEqual(summary.attack, 140)
        self.assertEqual(summary.defense, 122)
        self.assertEqual(summary.special_attack, 156)
        self.assertEqual(summary.special_defense, 136)
        self.assertEqual(summary.speed, 166)
        self.assertEqual(summary.moves, (("ember", 40), ("tackle", 56)))

    def test_summary_to_dict_is_serializable(self) -> None:
        import json

        spec = build_battle_spec_from_result(charmander_squirtle_result(), minimal_dex())
        payload = active_state_summary_from_side(spec.side_two).to_dict()
        json.dumps(payload)
        self.assertEqual(payload["species"], "squirtle")
        self.assertEqual(payload["moves"][0], {"id": "watergun", "pp": 40})


# ---- damage diagnostic: engine-state inspection ---------------------------


def fake_engine_mon(**overrides):
    """A fake built-engine Pokemon mirroring the real binding's readable surface.

    Uses mixed-case ids and a Gen 3 ``typeless`` padding slot on purpose so the
    extractor's normalization (lower-casing, dropping ``typeless``) is exercised.
    """

    base = dict(
        id="Charmander",
        level=100,
        hp=219,
        maxhp=219,
        types=("Fire", "typeless"),
        ability="Blaze",
        item="none",
        attack=140,
        defense=122,
        special_attack=156,
        special_defense=136,
        speed=166,
        moves=[SimpleNamespace(id="Ember", pp=40), SimpleNamespace(id="Tackle", pp=56)],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def fake_engine_state(side_one_mon, side_two_mon):
    return SimpleNamespace(
        # active_index as a string on one side, int on the other: both must coerce.
        side_one=SimpleNamespace(active_index="0", pokemon=[side_one_mon]),
        side_two=SimpleNamespace(active_index=0, pokemon=[side_two_mon]),
    )


def request_active_summary(seat: str = "side_one") -> ActiveStateSummary:
    spec = build_battle_spec_from_result(charmander_squirtle_result(), minimal_dex())
    side = spec.side_one if seat == "side_one" else spec.side_two
    return active_state_summary_from_side(side)


def matching_engine_summary(req: ActiveStateSummary) -> EngineActiveStateSummary:
    return EngineActiveStateSummary(
        species=req.species,
        level=req.level,
        hp=req.hp,
        maxhp=req.maxhp,
        types=tuple(req.types),
        ability=req.ability,
        item=req.item,
        attack=req.attack,
        defense=req.defense,
        special_attack=req.special_attack,
        special_defense=req.special_defense,
        speed=req.speed,
        moves=tuple(req.moves),
    )


class EngineActiveStateSummaryTest(unittest.TestCase):
    def test_extracts_and_normalizes_fields(self) -> None:
        state = fake_engine_state(fake_engine_mon(), fake_engine_mon(id="Squirtle"))
        summary = engine_active_state_summary(state, "side_one")

        self.assertIsInstance(summary, EngineActiveStateSummary)
        self.assertEqual(summary.missing_fields, ())
        # ids lower-cased, typeless dropped from types.
        self.assertEqual(summary.species, "charmander")
        self.assertEqual(summary.types, ("fire",))
        self.assertEqual(summary.ability, "blaze")
        # "none" item normalizes to None (no item), not the literal string.
        self.assertIsNone(summary.item)
        self.assertEqual((summary.hp, summary.maxhp), (219, 219))
        self.assertEqual(summary.attack, 140)
        self.assertEqual(summary.speed, 166)
        self.assertEqual(summary.moves, (("ember", 40), ("tackle", 56)))

    def test_other_side_uses_int_active_index(self) -> None:
        state = fake_engine_state(fake_engine_mon(), fake_engine_mon(id="Squirtle"))
        summary = engine_active_state_summary(state, "side_two")
        self.assertEqual(summary.species, "squirtle")

    def test_missing_attribute_is_recorded_not_raised(self) -> None:
        # A Pokemon lacking 'speed' must not crash extraction; the field is recorded.
        mon = fake_engine_mon()
        del mon.speed
        summary = engine_active_state_summary(fake_engine_state(mon, fake_engine_mon()), "side_one")
        self.assertIn("speed", summary.missing_fields)
        self.assertIsNone(summary.speed)
        # Other fields still read fine.
        self.assertEqual(summary.attack, 140)

    def test_uncoercible_stat_is_recorded_missing(self) -> None:
        mon = fake_engine_mon(attack="not-an-int")
        summary = engine_active_state_summary(fake_engine_state(mon, fake_engine_mon()), "side_one")
        self.assertIn("attack", summary.missing_fields)
        self.assertIsNone(summary.attack)

    def test_uninspectable_state_marks_every_field_missing(self) -> None:
        # An opaque state object can't be navigated; every field is reported missing
        # rather than raising.
        summary = engine_active_state_summary(object(), "side_one")
        self.assertEqual(set(summary.missing_fields), set(ENGINE_STATE_COMPARED_FIELDS))
        self.assertIsNone(summary.species)
        self.assertIsNone(summary.moves)

    def test_to_dict_is_serializable(self) -> None:
        import json

        state = fake_engine_state(fake_engine_mon(), fake_engine_mon(id="Squirtle"))
        payload = engine_active_state_summary(state, "side_one").to_dict()
        json.dumps(payload)
        self.assertEqual(payload["types"], ["fire"])
        self.assertEqual(payload["moves"][0], {"id": "ember", "pp": 40})
        self.assertEqual(payload["missing_fields"], [])


class CompareEngineActiveStateTest(unittest.TestCase):
    def test_identical_summaries_match(self) -> None:
        req = request_active_summary("side_one")
        comparison = compare_engine_active_state(req, matching_engine_summary(req), "side_one")
        self.assertIsInstance(comparison, ActiveStateComparison)
        self.assertTrue(comparison.matched)
        self.assertEqual(comparison.mismatches, ())
        self.assertIsNone(comparison.reason)
        self.assertIn("MATCH", comparison.summary())

    def test_request_values_are_normalized_before_comparison(self) -> None:
        from dataclasses import replace

        req = replace(
            request_active_summary("side_one"),
            species="Charmander",
            types=("Fire", "typeless"),
            ability="BLAZE",
            item="None",
            moves=(("Ember", "40"), ("TACKLE", 56)),
        )
        engine = replace(matching_engine_summary(request_active_summary("side_one")), item=None)
        comparison = compare_engine_active_state(req, engine, "side_one")
        self.assertTrue(comparison.matched)

    def test_field_value_difference_is_reported(self) -> None:
        req = request_active_summary("side_one")
        from dataclasses import replace

        engine = replace(matching_engine_summary(req), special_attack=999, speed=1)
        comparison = compare_engine_active_state(req, engine, "side_one")
        self.assertFalse(comparison.matched)
        fields = {m.field: m for m in comparison.mismatches}
        self.assertEqual(set(fields), {"special_attack", "speed"})
        self.assertEqual((fields["special_attack"].request_value, fields["special_attack"].engine_value), (156, 999))
        self.assertIn("special_attack", comparison.summary())

    def test_types_difference_is_reported(self) -> None:
        req = request_active_summary("side_one")
        from dataclasses import replace

        engine = replace(matching_engine_summary(req), types=("water",))
        comparison = compare_engine_active_state(req, engine, "side_one")
        self.assertFalse(comparison.matched)
        self.assertEqual([m.field for m in comparison.mismatches], ["types"])

    def test_unreadable_field_reported_with_marker_and_reason(self) -> None:
        req = request_active_summary("side_one")
        from dataclasses import replace

        engine = replace(matching_engine_summary(req), speed=None, missing_fields=("speed",))
        comparison = compare_engine_active_state(req, engine, "side_one")
        self.assertFalse(comparison.matched)
        self.assertEqual([m.field for m in comparison.mismatches], ["speed"])
        self.assertEqual(comparison.mismatches[0].engine_value, ENGINE_FIELD_UNREAD)
        self.assertIsNotNone(comparison.reason)
        self.assertIn("speed", comparison.reason)

    def test_to_dict_is_serializable(self) -> None:
        import json

        req = request_active_summary("side_one")
        from dataclasses import replace

        engine = replace(matching_engine_summary(req), attack=1)
        payload = compare_engine_active_state(req, engine, "side_one").to_dict()
        json.dumps(payload)
        self.assertFalse(payload["matched"])
        self.assertEqual(payload["mismatches"][0]["field"], "attack")
        self.assertEqual(payload["side"], "side_one")


class MismatchSurfaceSelectionTest(unittest.TestCase):
    """Surface selection over the engine-state comparisons (no engine required)."""

    def _observed(self) -> ObservedDamage:
        return ObservedDamage(opening_hp=(219, 229), final_hp=(127, 209), deltas=(92, 20))

    def _branches(self):
        return (EngineBranchDamageSummary(100.0, (122, 207), (97, 22), "d"),)

    def _matched_comparison(self, side: str) -> ActiveStateComparison:
        req = request_active_summary(side)
        return compare_engine_active_state(req, matching_engine_summary(req), side)

    def _mismatched_comparison(self, side: str) -> ActiveStateComparison:
        req = request_active_summary(side)
        from dataclasses import replace

        return compare_engine_active_state(req, replace(matching_engine_summary(req), attack=1), side)

    def test_damage_match_reports_no_surface(self) -> None:
        surface, notes = _mismatch_surface_and_notes(
            matched=True,
            observed=self._observed(),
            engine_branches=self._branches(),
            side_one_comparison=None,
            side_two_comparison=None,
        )
        self.assertEqual(surface, "none")

    def test_both_engine_states_match_narrows_surface(self) -> None:
        surface, notes = _mismatch_surface_and_notes(
            matched=False,
            observed=self._observed(),
            engine_branches=self._branches(),
            side_one_comparison=self._matched_comparison("side_one"),
            side_two_comparison=self._matched_comparison("side_two"),
        )
        self.assertEqual(surface, "engine damage/data path")
        self.assertTrue(any("inspected exposed/stored field" in note for note in notes))
        self.assertTrue(any("UNRESOLVED" in note for note in notes))

    def test_one_engine_state_mismatch_keeps_broad_surface(self) -> None:
        surface, notes = _mismatch_surface_and_notes(
            matched=False,
            observed=self._observed(),
            engine_branches=self._branches(),
            side_one_comparison=self._mismatched_comparison("side_one"),
            side_two_comparison=self._matched_comparison("side_two"),
        )
        self.assertEqual(surface, "engine damage/data or state-translation path")
        self.assertTrue(any("did not clear" in note for note in notes))
        self.assertTrue(any("attack" in note for note in notes))
        self.assertTrue(any("UNRESOLVED" in note for note in notes))

    def test_uninspectable_comparison_keeps_broad_surface(self) -> None:
        surface, notes = _mismatch_surface_and_notes(
            matched=False,
            observed=self._observed(),
            engine_branches=self._branches(),
            side_one_comparison=self._matched_comparison("side_one"),
            side_two_comparison=None,
        )
        self.assertEqual(surface, "engine damage/data or state-translation path")
        self.assertTrue(any("could not be inspected" in note for note in notes))
        self.assertTrue(any("UNRESOLVED" in note for note in notes))


# ---- damage diagnostic: direct calculate_damage ---------------------------


def calc_module(impl):
    module = ModuleType("poke_engine_fake_calc")
    module.calculate_damage = impl
    return module


class DirectCalculateDamageDiagnosticTest(unittest.TestCase):
    def test_simple_shape_is_captured_for_both_turn_orders(self) -> None:
        calls = []

        def impl(state, m1, m2, side_one_first):
            calls.append((m1, m2, side_one_first))
            return [10.0, 4.0] if side_one_first else [9.0, 5.0]

        engine = calc_module(impl)
        diag = direct_calculate_damage_diagnostic(engine, object(), "ember", "watergun")

        self.assertTrue(diag.supported)
        self.assertEqual(diag.output_side_one_first, [10.0, 4.0])
        self.assertEqual(diag.output_side_two_first, [9.0, 5.0])
        self.assertIsNone(diag.reason)
        # Probed both orderings with the resolved move ids.
        self.assertEqual(calls, [("ember", "watergun", True), ("ember", "watergun", False)])

    def test_to_dict_is_serializable(self) -> None:
        import json

        engine = calc_module(lambda s, m1, m2, first: {"attacker": 10, "defender": 4})
        diag = direct_calculate_damage_diagnostic(engine, object(), "ember", "watergun")
        payload = diag.to_dict()
        json.dumps(payload)
        self.assertTrue(payload["supported"])
        self.assertEqual(payload["side_one_moves_first"], {"attacker": 10, "defender": 4})

    def test_missing_function_is_unsupported_not_raising(self) -> None:
        engine = ModuleType("poke_engine_no_calc")
        diag = direct_calculate_damage_diagnostic(engine, object(), "ember", "watergun")
        self.assertFalse(diag.supported)
        self.assertIsNotNone(diag.reason)
        self.assertIn("calculate_damage", diag.reason)

    def test_raising_function_is_unsupported_not_raising(self) -> None:
        def boom(*args):
            raise RuntimeError("native panic")

        diag = direct_calculate_damage_diagnostic(calc_module(boom), object(), "ember", "watergun")
        self.assertFalse(diag.supported)
        self.assertIn("raised", diag.reason)
        self.assertIn("native panic", diag.reason)

    def test_unknown_shape_is_unsupported_not_raising(self) -> None:
        engine = calc_module(lambda s, m1, m2, first: SimpleNamespace(opaque=True))
        diag = direct_calculate_damage_diagnostic(engine, object(), "ember", "watergun")
        self.assertFalse(diag.supported)
        self.assertIn("unrecognized", diag.reason)

    def test_non_finite_float_payload_is_unsupported(self) -> None:
        # NaN/inf are not strict-JSON-safe, so a payload carrying them must be
        # rejected rather than emitted (it would only survive allow_nan=True).
        engine = calc_module(lambda s, m1, m2, first: [float("nan"), 4.0])
        diag = direct_calculate_damage_diagnostic(engine, object(), "ember", "watergun")
        self.assertFalse(diag.supported)
        self.assertIn("unrecognized", diag.reason)

    def test_supported_payload_is_strict_json_safe(self) -> None:
        import json

        engine = calc_module(lambda s, m1, m2, first: [10.0, 4.0])
        diag = direct_calculate_damage_diagnostic(engine, object(), "ember", "watergun")
        self.assertTrue(diag.supported)
        # Strict JSON (no NaN/inf) round-trips the coerced payload.
        json.dumps(diag.to_dict(), allow_nan=False)


# ---- damage diagnostic assembler (fake engine, no wheel/Showdown) ---------


def fake_damage_engine_module(branches, *, calc=None, corrupt=None):
    """A fake engine module sufficient to run build_one_turn_damage_diagnostic.

    Provides the State/Side/Pokemon/Move construction API that
    build_poke_engine_state needs, plus generate_instructions/apply_instructions,
    so the assembler runs end-to-end without the native wheel or a built Showdown.
    Each branch carries the final active HP tuple its apply_instructions sets.

    ``corrupt`` is an optional callable invoked with each built Pokemon's kwargs dict
    so a test can mutate the *built engine state* away from the request-derived spec
    (modelling an unfaithful spec->engine translation) before construction.
    """

    module = ModuleType("poke_engine_fake_damage")

    class _State:
        def __init__(self, side_one, side_two, **kwargs):
            self.side_one = side_one
            self.side_two = side_two

        def apply_instructions(self, branch):
            p1_hp, p2_hp = branch._hp
            return _State(
                side_one=SimpleNamespace(active_index=0, pokemon=[SimpleNamespace(hp=p1_hp)]),
                side_two=SimpleNamespace(active_index=0, pokemon=[SimpleNamespace(hp=p2_hp)]),
            )

    def _pokemon(**kwargs):
        # Mirror the real poke-engine Pokemon, which always exposes ability/item as a
        # string ("none" when absent). The adapter only passes them when set, so default
        # the rest here -- otherwise the built engine state would look unfaithful purely
        # because a fake omitted an attribute the real binding always carries.
        kwargs.setdefault("ability", "none")
        kwargs.setdefault("item", "none")
        if corrupt is not None:
            corrupt(kwargs)
        return SimpleNamespace(**kwargs)

    module.State = _State
    module.Side = lambda **kwargs: SimpleNamespace(**kwargs)
    module.Pokemon = _pokemon
    module.Move = lambda **kwargs: SimpleNamespace(**kwargs)
    module.generate_instructions = lambda state, m1, m2: list(branches)
    if calc is not None:
        module.calculate_damage = calc
    return module


class BuildOneTurnDamageDiagnosticTest(unittest.TestCase):
    def test_mismatch_with_matching_engine_state_narrows_surface(self) -> None:
        import json

        # No engine branch reproduces Showdown's (127, 209) final HP, but the built
        # engine state faithfully mirrors the request-derived spec on both sides.
        engine = fake_damage_engine_module(
            [
                fake_branch(122, 207, 79.1, "Damage A"),
                fake_branch(25, 207, 5.27, "Crit A"),
            ]
        )
        diag = build_one_turn_damage_diagnostic(charmander_squirtle_result(), minimal_dex(), module=engine)

        self.assertIsInstance(diag, OneTurnDamageDiagnostic)
        self.assertTrue(diag.supported)
        self.assertFalse(diag.matched)
        self.assertEqual((diag.p1_move, diag.p2_move), ("ember", "watergun"))

        # Observed deltas come from the opening request HP (219/229) and final (127/209).
        self.assertEqual(diag.observed.opening_hp, (219, 229))
        self.assertEqual(diag.observed.final_hp, (127, 209))
        self.assertEqual(diag.observed.deltas, (92, 20))

        # Engine-branch deltas are computed against opening HP: 219-122=97, 229-207=22.
        self.assertEqual(diag.engine_branches[0].deltas, (97, 22))
        self.assertEqual(diag.engine_branches[1].deltas, (194, 22))
        self.assertEqual(diag.engine_delta_tuples(), ((97, 22), (194, 22)))

        # The built engine state was read back and matches the request spec on both
        # sides, so the surface narrows to the engine's damage/data path -- the mismatch
        # is NOT made to look like a pass (matched stays False).
        self.assertTrue(diag.side_one_comparison.matched)
        self.assertEqual(diag.side_one_comparison.mismatches, ())
        self.assertTrue(diag.side_two_comparison.matched)
        self.assertEqual(diag.likely_mismatch_surface, "engine damage/data path")
        self.assertTrue(any("dex-derived" in note for note in diag.notes))
        self.assertTrue(any("UNRESOLVED" in note for note in diag.notes))

        # Engine-state summaries are read back off the built state for both seats.
        self.assertEqual(diag.side_one_engine_state.species, "charmander")
        self.assertEqual(diag.side_one_engine_state.types, ("fire",))
        self.assertEqual(diag.side_one_engine_state.moves, (("ember", 40), ("tackle", 56)))
        self.assertIsNone(diag.side_one_engine_state.item)
        self.assertEqual(diag.side_two_engine_state.species, "squirtle")

        # Request-derived active-state summaries are still present for both seats.
        self.assertEqual(diag.side_one_state.species, "charmander")
        self.assertEqual(diag.side_one_state.types, ("fire",))
        self.assertEqual(diag.side_two_state.species, "squirtle")

        # No calculate_damage on this fake module -> explicit unsupported, not a raise.
        self.assertFalse(diag.direct_calculate_damage.supported)
        self.assertIn("calculate_damage", diag.direct_calculate_damage.reason)

        # Fully serializable (strict JSON, no NaN/inf).
        json.dumps(diag.to_dict(), allow_nan=False)
        self.assertIn("MISMATCH", diag.summary())
        self.assertIn("engine-state matches request spec", diag.summary())

    def test_mismatch_with_unfaithful_engine_state_keeps_broad_surface(self) -> None:
        import json

        # The built engine state mis-translates a stat (here Charmander's special_attack
        # comes back wrong), so the spec->engine translation path cannot be cleared and
        # the broad surface must stand with an explicit, named field mismatch.
        engine = fake_damage_engine_module(
            [fake_branch(122, 207, 100.0, "Damage A")],
            corrupt=lambda kwargs: kwargs.update(special_attack=999)
            if kwargs.get("id") == "charmander"
            else None,
        )
        diag = build_one_turn_damage_diagnostic(charmander_squirtle_result(), minimal_dex(), module=engine)

        self.assertFalse(diag.matched)
        self.assertEqual(diag.likely_mismatch_surface, "engine damage/data or state-translation path")

        # The mismatch is explicit and field-level, not a vague note.
        self.assertFalse(diag.side_one_comparison.matched)
        fields = [m.field for m in diag.side_one_comparison.mismatches]
        self.assertEqual(fields, ["special_attack"])
        mismatch = diag.side_one_comparison.mismatches[0]
        self.assertEqual(mismatch.request_value, 156)
        self.assertEqual(mismatch.engine_value, 999)
        # The other side translated faithfully.
        self.assertTrue(diag.side_two_comparison.matched)

        # Notes explain WHY the broad surface stands and name the offending field.
        self.assertTrue(any("did not clear" in note for note in diag.notes))
        self.assertTrue(any("special_attack" in note for note in diag.notes))
        self.assertTrue(any("UNRESOLVED" in note for note in diag.notes))

        json.dumps(diag.to_dict(), allow_nan=False)
        self.assertIn("engine-state mismatch on side_one", diag.summary())

    def test_match_branch_reports_no_surface(self) -> None:
        # One engine branch reproduces Showdown's exact (127, 209) final HP.
        engine = fake_damage_engine_module(
            [
                fake_branch(127, 209, 79.1, "Exact match"),
                fake_branch(25, 207, 5.27, "Crit A"),
            ]
        )
        diag = build_one_turn_damage_diagnostic(charmander_squirtle_result(), minimal_dex(), module=engine)

        self.assertTrue(diag.matched)
        self.assertEqual(diag.likely_mismatch_surface, "none")
        self.assertEqual(diag.engine_branches[0].deltas, (92, 20))
        self.assertIn("MATCH", diag.summary())

    def test_direct_calculate_damage_payload_propagates(self) -> None:
        def impl(state, m1, m2, side_one_first):
            return [10.0, 4.0] if side_one_first else [9.0, 5.0]

        engine = fake_damage_engine_module(
            [fake_branch(122, 207, 100.0, "Damage A")],
            calc=impl,
        )
        diag = build_one_turn_damage_diagnostic(charmander_squirtle_result(), minimal_dex(), module=engine)

        direct = diag.direct_calculate_damage
        self.assertTrue(direct.supported)
        self.assertEqual(direct.output_side_one_first, [10.0, 4.0])
        self.assertEqual(direct.output_side_two_first, [9.0, 5.0])

    def test_assembler_does_not_import_real_engine(self) -> None:
        had_real = "poke_engine" in sys.modules
        engine = fake_damage_engine_module([fake_branch(122, 207, 100.0, "Damage A")])
        build_one_turn_damage_diagnostic(charmander_squirtle_result(), minimal_dex(), module=engine)
        if not had_real:
            self.assertNotIn(
                "poke_engine",
                sys.modules,
                "damage diagnostic imported real poke_engine despite a fake module",
            )


# ---- isolation ------------------------------------------------------------


class FakeModuleIsolationTest(unittest.TestCase):
    def test_enumerate_does_not_import_real_engine(self) -> None:
        had_real = "poke_engine" in sys.modules
        engine = fake_engine_module([fake_branch(122, 207, 100.0, "Damage")])
        enumerate_engine_outcomes(engine, FakeState(), "ember", "watergun")
        if not had_real:
            self.assertNotIn(
                "poke_engine",
                sys.modules,
                "outcome enumeration imported real poke_engine despite a fake module",
            )


# ---- optional real integration --------------------------------------------


def integration_config():
    """Only run when a built Showdown checkout + node are available."""

    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=20.0)


@unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
class RealOutcomeComparisonIntegrationTest(unittest.TestCase):
    def test_engine_runs_and_reports_known_mismatch_on_0_0_47(self) -> None:
        probe = probe_poke_engine()
        if not probe.ready:
            self.skipTest("poke-engine is not installed/ready")
        if probe.version != POKE_ENGINE_SUPPORTED_VERSION:
            self.skipTest(
                f"known mismatch assertion is pinned to poke-engine {POKE_ENGINE_SUPPORTED_VERSION}, "
                f"found {probe.version or 'unknown'}"
            )

        from pokezero.poke_engine_outcomes import run_charmander_squirtle_outcome_comparison

        config = integration_config()
        assert config is not None
        result = run_charmander_squirtle_outcome_comparison(config=config)

        # The comparator must actually run end-to-end (engine enumerated branches).
        self.assertTrue(result.supported)
        self.assertTrue(result.engine_final_hp_outcomes)
        # Showdown's seeded outcome for this fixture is fixed.
        self.assertEqual(result.showdown_final_hp, (127, 209))
        # poke-engine 0.0.47 does NOT reproduce it: honest mismatch, not a fake pass.
        self.assertFalse(result.matched)
        self.assertNotIn((127, 209), result.engine_final_hp_tuples())

    def test_damage_diagnostic_runs_and_records_known_mismatch(self) -> None:
        import json

        probe = probe_poke_engine()
        if not probe.ready:
            self.skipTest("poke-engine is not installed/ready")
        if probe.version != POKE_ENGINE_SUPPORTED_VERSION:
            self.skipTest(
                f"known mismatch assertion is pinned to poke-engine {POKE_ENGINE_SUPPORTED_VERSION}, "
                f"found {probe.version or 'unknown'}"
            )

        from pokezero.poke_engine_outcomes import run_charmander_squirtle_damage_diagnostic

        config = integration_config()
        assert config is not None
        diag = run_charmander_squirtle_damage_diagnostic(config=config)

        # Diagnostic runs end-to-end and is fully serializable.
        self.assertTrue(diag.supported)
        json.dumps(diag.to_dict())

        # It records the known mismatch honestly (not a faked pass).
        self.assertFalse(diag.matched)
        self.assertIsNotNone(diag.observed)
        self.assertEqual(diag.observed.final_hp, (127, 209))
        self.assertEqual(diag.observed.deltas, (92, 20))
        self.assertTrue(diag.engine_branches)

        # Request-derived active state is present for both seats.
        self.assertEqual(diag.side_one_state.species, "charmander")
        self.assertEqual(diag.side_two_state.species, "squirtle")

        # The built engine state is inspected and compared field-by-field against the
        # request-derived spec. The surface narrows to the engine's damage/data path
        # ONLY if both sides' states match; otherwise the broad surface stands with
        # explicit field-level mismatches. Either way the damage mismatch is real
        # (matched stays False) -- this never fakes a pass.
        self.assertIsNotNone(diag.side_one_comparison)
        self.assertIsNotNone(diag.side_two_comparison)
        self.assertEqual(diag.side_one_engine_state.species, "charmander")
        self.assertEqual(diag.side_two_engine_state.species, "squirtle")
        both_match = diag.side_one_comparison.matched and diag.side_two_comparison.matched
        if both_match:
            self.assertEqual(diag.likely_mismatch_surface, "engine damage/data path")
            self.assertTrue(any("inspected exposed/stored field" in note for note in diag.notes))
        else:
            self.assertEqual(
                diag.likely_mismatch_surface, "engine damage/data or state-translation path"
            )
            # A broad surface here must be backed by explicit, named field mismatches
            # (or an inability to inspect), not left vague.
            explicit = [
                m.field
                for comparison in (diag.side_one_comparison, diag.side_two_comparison)
                for m in comparison.mismatches
            ]
            self.assertTrue(explicit, "broad surface must name the mismatching engine-state fields")
        self.assertTrue(any("UNRESOLVED" in note for note in diag.notes))

        # The local poke-engine 0.0.47 binding exposes a usable, serializable
        # calculate_damage output in this environment, so demand it: an unsupported
        # result here is a real regression, not an acceptable fallback.
        direct = diag.direct_calculate_damage
        self.assertIsNotNone(direct)
        self.assertTrue(
            direct.supported,
            f"poke-engine {POKE_ENGINE_SUPPORTED_VERSION} direct calculate_damage is "
            f"expected to be supported here, got unsupported: {direct.reason}",
        )
        self.assertIsNotNone(direct.output_side_one_first)
        self.assertIsNotNone(direct.output_side_two_first)
        # Strict JSON (no NaN/inf) round-trips the coerced output.
        json.dumps([direct.output_side_one_first, direct.output_side_two_first], allow_nan=False)

        print("\nREAL DAMAGE DIAGNOSTIC:", diag.summary())
        print("side_one engine-state:", diag.side_one_comparison.summary())
        print("side_two engine-state:", diag.side_two_comparison.summary())
        print("direct calculate_damage:", direct.to_dict())


if __name__ == "__main__":
    unittest.main()
