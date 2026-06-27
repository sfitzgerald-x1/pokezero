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
    EngineBranchOutcome,
    OutcomeComparison,
    build_battle_spec_from_result,
    compare_outcomes,
    engine_move_for_choice,
    enumerate_engine_outcomes,
    observed_final_active_hp,
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


if __name__ == "__main__":
    unittest.main()
