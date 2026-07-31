from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pokezero.dex import normalize_id, showdown_dex_from_payload
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
from pokezero.poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from scripts import attest_damage_arithmetic_tail as attestation
from scripts.attest_damage_arithmetic_tail import (
    DirectHit,
    EXACT_SUPPORTED_MOVE_SPECS,
    _basic_oracle,
    _branch_direct_damage,
    _branch_has_status,
    _branch_report,
    _candidate_report,
    _classify_branch_verdict,
    _native_rolls,
    _showdown_dependency_paths,
    _showdown_source_provenance,
    _validated_slot_sides,
    observed_direct_hit,
)


class DamageArithmeticTailAttestationTests(unittest.TestCase):
    def _state(
        self,
        *,
        ability: str = "",
        item: str = "",
        defender_ability: str = "",
        defender_item: str = "",
        weather: str = "",
        attacker_types: tuple[str, ...] = ("Poison",),
        defender_types: tuple[str, ...] = ("Normal",),
        volatiles: tuple[str, ...] = (),
        attack: int = 200,
        special_attack: int = 150,
        defense: int = 150,
        special_defense: int = 150,
    ) -> SimpleNamespace:
        attacker = SimpleNamespace(
            id="Attacker", level=80, hp=300, maxhp=300, attack=attack, defense=150,
            special_attack=special_attack, special_defense=150, speed=100,
            ability=ability, item=item, status="", types=attacker_types,
        )
        defender = SimpleNamespace(
            id="Defender", level=80, hp=400, maxhp=400, attack=150, defense=defense,
            special_attack=150, special_defense=special_defense, speed=80,
            ability=defender_ability, item=defender_item, status="", types=defender_types,
        )

        def side(member: SimpleNamespace) -> SimpleNamespace:
            return SimpleNamespace(
                pokemon=(member,), active_index=0, attack_boost=0, defense_boost=0,
                special_attack_boost=0, special_defense_boost=0, speed_boost=0,
                side_conditions=SimpleNamespace(reflect=0, light_screen=0), volatile_statuses=volatiles,
            )
        return SimpleNamespace(weather=weather, side_one=side(attacker), side_two=side(defender))

    @staticmethod
    def _dex():
        return showdown_dex_from_payload(
            {
                "moves": {
                    "sludgebomb": {
                        "name": "Sludge Bomb",
                        "type": "Poison",
                        "category": "Special",
                        "basePower": 90,
                        "accuracy": 100,
                        "priority": 0,
                    },
                    "fireblast": {
                        "name": "Fire Blast",
                        "type": "Fire",
                        "category": "Special",
                        "basePower": 120,
                        "accuracy": 85,
                        "priority": 0,
                    },
                },
                "species": {},
                "typeChart": {
                    "Normal": {},
                    "Grass": {"Poison": 1, "Fire": 1},
                    "Poison": {"Poison": 2},
                    "Water": {"Fire": 2},
                },
            }
        )

    @staticmethod
    def _native_state(
        *,
        side_one_moves: tuple[str, ...],
        side_two_moves: tuple[str, ...],
        side_one_types: tuple[str, ...] = ("poison",),
        side_two_types: tuple[str, ...] = ("normal",),
        side_one_speed: int = 200,
        side_two_speed: int = 100,
        side_one_hp: int = 500,
        side_two_hp: int = 500,
        side_one_maxhp: int = 500,
        side_two_maxhp: int = 500,
        side_one_item: str | None = None,
        side_two_item: str | None = None,
        weather: str = "none",
    ) -> str:
        native_engine, _, native_reason = attestation._native_modules()
        if native_reason is not None:
            raise unittest.SkipTest(native_reason)

        def pokemon(
            species: str,
            moves: tuple[str, ...],
            types: tuple[str, ...],
            speed: int,
            hp: int,
            maxhp: int,
            item: str | None,
        ) -> PokemonSpec:
            return PokemonSpec(
                id=species,
                level=80,
                types=types,
                hp=hp,
                maxhp=maxhp,
                attack=200,
                defense=150,
                special_attack=150,
                special_defense=150,
                speed=speed,
                item=item,
                moves=tuple(MoveSpec(id=move, pp=32) for move in moves),
            )

        return build_poke_engine_state(
            BattleSpec(
                side_one=SideSpec(
                    pokemon=(
                        pokemon(
                            "arbok", side_one_moves, side_one_types,
                            side_one_speed, side_one_hp, side_one_maxhp, side_one_item,
                        ),
                    )
                ),
                side_two=SideSpec(
                    pokemon=(
                        pokemon(
                            "snorlax", side_two_moves, side_two_types,
                            side_two_speed, side_two_hp, side_two_maxhp, side_two_item,
                        ),
                    )
                ),
                weather=weather,
            ),
            module=native_engine,
        ).to_string()

    def test_switch_seeds_direct_damage_from_the_incoming_mon(self) -> None:
        row = {
            "pre_features": {"p1_hp": 300, "p2_hp": 200},
            "protocol": [
                "|switch|p1a: Raichu|Raichu, L83|141/235",
                "|move|p2a: Qwilfish|Sludge Bomb|p1a: Raichu",
                "|-damage|p1a: Raichu|19/235",
                "|-status|p1a: Raichu|psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.actor, "p2")
        self.assertEqual(hit.target, "p1")
        self.assertEqual(hit.move, "sludgebomb")
        self.assertEqual(hit.damage, 122)
        self.assertEqual(hit.secondary_status, "psn")

    def test_residual_damage_is_not_mistaken_for_the_move(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Surf|p2a: B",
                "|-damage|p2a: B|70/100",
                "|-damage|p2a: B|64/100|[from] psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.damage, 30)
        self.assertIsNone(hit.secondary_status)

    def test_first_direct_hit_is_not_replaced_or_later_status_mutated(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Surf|p2a: B",
                "|-damage|p2a: B|70/100",
                "|move|p2a: B|Sludge Bomb|p1a: A",
                "|-damage|p1a: A|80/100",
                "|-status|p1a: A|psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual((hit.actor, hit.target, hit.move, hit.damage), ("p1", "p2", "surf", 30))
        self.assertIsNone(hit.secondary_status)

    def test_status_binds_to_the_selected_first_direct_hit(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Sludge Bomb|p2a: B",
                "|-damage|p2a: B|70/100",
                "|-supereffective|p2a: B",
                "|-status|p2a: B|psn",
            ],
        }

        hit = observed_direct_hit(row)

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.secondary_status, "psn")

    def test_status_attribution_ignores_from_sources(self) -> None:
        row = {
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
            "protocol": [
                "|move|p1a: A|Sludge Bomb|p2a: B",
                "|-damage|p2a: B|70/100",
                "|-status|p2a: B|psn|[from] ability: Poison Point",
            ],
        }
        hit = observed_direct_hit(row)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertIsNone(hit.secondary_status)
        self.assertFalse(_branch_has_status(row["protocol"], "p2", "psn"))

    def test_branch_damage_seeds_pre_hit_hp_and_switch_overrides_it(self) -> None:
        self.assertEqual(
            _branch_direct_damage(["|-damage|p2a: B|70/100"], "p2", pre_hit_hp=100), 30
        )
        self.assertEqual(
            _branch_direct_damage(
                ["|switch|p2a: B|B, L50|90/100", "|-damage|p2a: B|70/100"],
                "p2", pre_hit_hp=100,
            ),
            20,
        )

    def test_complex_oracle_context_fails_closed(self) -> None:
        state = self._state(ability="Guts")

        context, rolls, limit = _basic_oracle(
            state=state,
            direct=DirectHit("p1", "p2", "sludgebomb", 100, False, None),
            dex=self._dex(),
        )

        self.assertIsNone(rolls)
        self.assertEqual(limit, "attacker_ability_not_classified:guts")
        self.assertEqual(
            context["modifier_classification"]["attacker_ability"],
            "not_classified",
        )

    def test_oracle_rejects_every_named_untranscribed_class(self) -> None:
        direct = DirectHit("p1", "p2", "sludgebomb", 10, False, None)
        for move in (
            "hiddenpowerice", "return", "frustration", "rollout", "furycutter",
            "weatherball", "explosion", "selfdestruct", "charge", "mudsport", "watersport",
            "helpinghand", "armthrust", "dragonrage", "magnitude",
        ):
            _, rolls, limit = _basic_oracle(
                state=self._state(), direct=DirectHit("p1", "p2", move, 10, False, None), dex=self._dex()
            )
            self.assertIsNone(rolls, move)
            self.assertIsNotNone(limit, move)
        for state in (
            self._state(item="Deep Sea Tooth"), self._state(item="Deep Sea Scale"),
            self._state(item="Metal Powder"), self._state(item="Sea Incense"),
            self._state(weather="Hail"),
            self._state(volatiles=("charge",)),
        ):
            _, rolls, limit = _basic_oracle(state=state, direct=direct, dex=self._dex())
            self.assertIsNone(rolls)
            self.assertIsNotNone(limit)

    def test_real_oracle_models_physical_choice_band_stab_and_effectiveness(self) -> None:
        choice_band = self._state(
            ability="Poison Point",
            item="Choice Band",
            defender_ability="Effect Spore",
            defender_item="Leftovers",
            attacker_types=("Poison",),
            defender_types=("Grass",),
        )
        unbanded = self._state(
            ability="Poison Point",
            item="Leftovers",
            defender_ability="Effect Spore",
            defender_item="Leftovers",
            attacker_types=("Poison",),
            defender_types=("Grass",),
        )
        direct = DirectHit("p1", "p2", "sludgebomb", 10, False, None)

        context, rolls, limit = _basic_oracle(
            state=choice_band, direct=direct, dex=self._dex()
        )
        _, unbanded_rolls, unbanded_limit = _basic_oracle(
            state=unbanded, direct=direct, dex=self._dex()
        )

        self.assertIsNone(limit)
        self.assertIsNone(unbanded_limit)
        self.assertEqual(context["category"], "Physical")
        self.assertTrue(context["stab"])
        self.assertEqual(context["effectiveness"], 2.0)
        self.assertEqual(context["attack_mods"], [[1.5, 1]])
        self.assertEqual(
            context["modifier_classification"]["attacker_item"],
            "modeled_choice_band_attack",
        )
        assert rolls is not None and unbanded_rolls is not None
        self.assertGreater(max(rolls), max(unbanded_rolls))

    def test_real_oracle_models_special_fire_weather_and_ignores_choice_band(self) -> None:
        states = {
            weather: self._state(
                ability="Levitate",
                item="Choice Band",
                defender_ability="Pure Power",
                defender_item="Leftovers",
                weather=weather,
                attacker_types=("Fire",),
                defender_types=("Grass",),
            )
            for weather in ("", "Sunny Day", "Rain Dance")
        }
        direct = DirectHit("p1", "p2", "fireblast", 10, False, None)
        evidence = {
            weather: _basic_oracle(state=state, direct=direct, dex=self._dex())
            for weather, state in states.items()
        }
        unbanded = self._state(
            ability="Levitate",
            item="Leftovers",
            defender_ability="Pure Power",
            defender_item="Leftovers",
            attacker_types=("Fire",),
            defender_types=("Grass",),
        )
        _, unbanded_rolls, unbanded_limit = _basic_oracle(
            state=unbanded, direct=direct, dex=self._dex()
        )

        for context, rolls, limit in evidence.values():
            self.assertIsNone(limit)
            self.assertIsNotNone(rolls)
            self.assertEqual(context["category"], "Special")
            self.assertTrue(context["stab"])
            self.assertEqual(context["effectiveness"], 2.0)
            self.assertEqual(context["attack_mods"], [])
            self.assertEqual(
                context["modifier_classification"]["attacker_item"],
                "proven_irrelevant",
            )
        self.assertIsNone(unbanded_limit)
        normal_rolls = evidence[""][1]
        sun_rolls = evidence["Sunny Day"][1]
        rain_rolls = evidence["Rain Dance"][1]
        assert (
            normal_rolls is not None
            and sun_rolls is not None
            and rain_rolls is not None
            and unbanded_rolls is not None
        )
        self.assertEqual(normal_rolls, unbanded_rolls)
        self.assertGreater(max(sun_rolls), max(normal_rolls))
        self.assertGreater(max(normal_rolls), max(rain_rolls))

    def test_all_five_documented_direct_contexts_reach_the_exact_oracle(self) -> None:
        cases = (
            (
                "sludgebomb",
                self._state(
                    ability="Poison Point",
                    item="Salac Berry",
                    defender_ability="Static",
                    defender_item="Leftovers",
                    attacker_types=("Water", "Poison"),
                    defender_types=("Electric",),
                    attack=203,
                    defense=138,
                ),
            ),
            (
                "sludgebomb",
                self._state(
                    ability="Liquid Ooze",
                    item="Leftovers",
                    defender_ability="Effect Spore",
                    defender_item="Leftovers",
                    attacker_types=("Water", "Poison"),
                    defender_types=("Grass", "Fighting"),
                    attack=156,
                    defense=182,
                ),
            ),
            (
                "sludgebomb",
                self._state(
                    ability="Shield Dust",
                    item="Leftovers",
                    defender_ability="Rough Skin",
                    defender_item="Choice Band",
                    attacker_types=("Bug", "Poison"),
                    defender_types=("Water", "Dark"),
                    attack=162,
                    defense=116,
                ),
            ),
            (
                "fireblast",
                self._state(
                    ability="Levitate",
                    item="Leftovers",
                    defender_ability="Pure Power",
                    defender_item="Leftovers",
                    attacker_types=("Poison",),
                    defender_types=("Fighting", "Psychic"),
                    special_attack=184,
                    special_defense=168,
                ),
            ),
            (
                "sludgebomb",
                self._state(
                    ability="Poison Point",
                    item="Choice Band",
                    defender_ability="Keen Eye",
                    defender_item="Choice Band",
                    weather="Sand",
                    attacker_types=("Poison", "Ground"),
                    defender_types=("Normal", "Flying"),
                    attack=198,
                    defense=152,
                ),
            ),
        )

        for move, state in cases:
            context, rolls, limit = _basic_oracle(
                state=state,
                direct=DirectHit("p1", "p2", move, 1, False, None),
                dex=self._dex(),
            )
            self.assertIsNone(limit, context)
            self.assertIsNotNone(rolls, context)
            self.assertTrue(rolls, context)
            self.assertNotIn(
                "not_classified", context["modifier_classification"].values()
            )

    def test_branch_verdicts_fail_closed_and_require_secondary_coupling_evidence(self) -> None:
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=None, oracle_limit="attacker_ability:guts", native_max=100,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "oracle_unavailable:attacker_ability:guts"),
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 100), oracle_limit=None, native_max=None,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "native_damage_binding_unavailable"),
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 90, 100), oracle_limit=None, native_max=99,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            )[0],
            "native_arithmetic_disagreement",
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 90, 100), oracle_limit=None, native_max=100,
                observed_damage=90, nonterminal_damage=(92,), secondary_status=None,
                secondary_branch_has_observed_damage=False,
            )[0],
            "no_arithmetic_disagreement",
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 90, 100), oracle_limit=None, native_max=100,
                observed_damage=90, nonterminal_damage=(92,), secondary_status="psn",
                secondary_branch_has_observed_damage=False,
            )[0],
            "fixed_single_roll_composition",
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(), oracle_limit=None, native_max=100, observed_damage=90,
                nonterminal_damage=(), secondary_status=None, secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "oracle_empty_roll_support"),
        )
        self.assertEqual(
            _classify_branch_verdict(
                oracle_rolls=(85, 100), oracle_limit=None, native_max=99, observed_damage=100,
                observed_ko_clamped=True, nonterminal_damage=(), secondary_status=None,
                secondary_branch_has_observed_damage=False,
            ),
            ("comparison_limit", "observed_damage_ko_clamped"),
        )

    def test_native_controls_probe_both_orders_and_fail_closed(self) -> None:
        class Engine:
            def calculate_damage(self, _state: object, _one: str, _two: str, first: bool) -> tuple[tuple[int, int], tuple[int, int]]:
                return ((10, 20), (30 if first else 31, 40 if first else 41))

        rolls, reason = _native_rolls(
            native_engine=Engine(), state=object(), side_one_choice="one", side_two_choice="two",
            actor_side="side_two",
        )
        self.assertIsNone(reason)
        self.assertEqual(rolls, {True: (30, 40), False: (31, 41)})

        class PanicEngine:
            def calculate_damage(self, *_: object) -> object:
                raise BaseException("native panic")

        rolls, reason = _native_rolls(
            native_engine=PanicEngine(), state=object(), side_one_choice="one", side_two_choice="two",
            actor_side="side_one",
        )
        self.assertIsNone(rolls)
        self.assertEqual(reason, "native_damage_call_failed:BaseException")

    def test_native_state_access_baseexception_limits_the_branch(self) -> None:
        class State:
            @staticmethod
            def from_string(_text: str) -> object:
                return object()

        class Engine:
            @staticmethod
            def calculate_damage(*_args: object) -> object:
                raise AssertionError("oracle extraction must fail first")
        Engine.State = State

        class Search:
            @staticmethod
            def branch_events(*_args: object) -> str:
                return json.dumps({
                    "branches": [{
                        "percentage": 100,
                        "events": ["|-damage|p2a: Defender|70/100"],
                        "lossy": [],
                    }]
                })

        class PanicDex:
            @staticmethod
            def move_info(_move: str) -> object:
                raise BaseException("native-backed dex panic")

        report = _candidate_report(
            candidate_index=0,
            candidate_state="state",
            native_engine=Engine(),
            native_search=Search(),
            side_one_choice="sludgebomb",
            side_two_choice="splash",
            mapper_context="{}",
            direct=DirectHit("p1", "p2", "sludgebomb", 30, False, None),
            dex=PanicDex(),
            slot_sides={"p1": "side_one", "p2": "side_two"},
            pre_hit_hp=100,
        )
        self.assertEqual(report["verdict"], "comparison_limit")
        self.assertEqual(report["branch_population"]["unsupported"], 1)
        self.assertEqual(
            report["branches"][0]["unsupported_reason"],
            "oracle_context_failed:BaseException",
        )

    def test_slot_sides_are_required_and_inverted_orientation_reaches_native_correctly(self) -> None:
        self.assertEqual(_validated_slot_sides({}), (None, "missing_slot_sides"))
        self.assertEqual(
            _validated_slot_sides({"slot_sides": {"p1": "side_one", "p2": "side_one"}}),
            (None, "invalid_slot_sides"),
        )

        calls: list[tuple[object, ...]] = []
        class State:
            @staticmethod
            def from_string(_text: str) -> object:
                return object()
        class Engine:
            def calculate_damage(self, _state: object, one: str, two: str, first: bool) -> tuple[tuple[int, int], tuple[int, int]]:
                calls.append((one, two, first))
                return ((10, 20), (20, 30))
        Engine.State = State
        class Search:
            def branch_events(self, _state: str, one: str, two: str, context: str, *_: object) -> str:
                calls.append((one, two, json.loads(context)))
                return json.dumps({"branches": [{"events": ["|-damage|p1a: B|70/100"], "legal_roll_state": "state"}]})
        row = {
            "engine_state": "state", "engine_states": ["state"], "kind": "transition_diverged",
            "choices": {"p1": "left", "p2": "right"},
            "slot_sides": {"p1": "side_two", "p2": "side_one"},
            "party_display": {"p1": ["Left"], "p2": ["Right"]}, "turn": 7,
            "pre_features": {"p1_hp": 100, "p2_hp": 100},
        }
        with (
            patch.object(attestation, "_native_modules", return_value=(Engine(), Search(), None)),
            patch.object(attestation, "_basic_oracle", return_value=({}, (17, 20), None)),
        ):
            result = _branch_report(
                row, DirectHit("p1", "p2", "sludgebomb", 20, False, None), self._dex()
            )
        self.assertEqual(result["verdict"], "no_arithmetic_disagreement")
        self.assertNotIn("native_representative_damage", result)
        self.assertIn(("right", "left", {"p1": ["Right"], "p2": ["Left"], "turn": 7}), calls)
        self.assertTrue(all(call[:2] == ("right", "left") for call in calls if len(call) == 3 and isinstance(call[2], bool)))

    def test_candidate_aggregation_requires_identical_contexts(self) -> None:
        row = {
            "engine_state": "first", "engine_states": ["first", "second"],
            "choices": {"p1": "one", "p2": "two"},
            "slot_sides": {"p1": "side_one", "p2": "side_two"},
            "party_display": {"p1": [], "p2": []}, "pre_features": {"p1_hp": 100, "p2_hp": 100},
        }
        with (
            patch.object(attestation, "_native_modules", return_value=(object(), object(), None)),
            patch.object(attestation, "_candidate_report", side_effect=[
                {"candidate_index": 0, "verdict": "no_arithmetic_disagreement", "reason": None, "native_max": 100},
                {"candidate_index": 1, "verdict": "native_arithmetic_disagreement", "reason": None, "native_max": 101},
            ]),
        ):
            result = _branch_report(
                row, DirectHit("p1", "p2", "sludgebomb", 90, False, None), self._dex()
            )
        self.assertEqual(result["verdict"], "comparison_limit")
        self.assertEqual(result["verdict_reason"], "candidate_contexts_or_results_differ")
        self.assertNotIn("native_representative_damage", result)

    def test_real_native_absent_legal_state_uses_full_crit_partition(self) -> None:
        native_engine, native_search, native_reason = attestation._native_modules()
        if native_reason is not None:
            self.skipTest(native_reason)
        state = self._native_state(
            side_one_moves=("sludgebomb",),
            side_two_moves=("splash",),
        )
        common = {
            "candidate_index": 0,
            "candidate_state": state,
            "native_engine": native_engine,
            "native_search": native_search,
            "side_one_choice": "sludgebomb",
            "side_two_choice": "splash",
            "mapper_context": json.dumps(
                {"p1": ["Attacker"], "p2": ["Defender"], "turn": 1}
            ),
            "dex": self._dex(),
            "slot_sides": {"p1": "side_one", "p2": "side_two"},
            "pre_hit_hp": 500,
        }
        noncritical = _candidate_report(
            **common,
            direct=DirectHit("p1", "p2", "sludgebomb", 114, False, None),
        )
        critical = _candidate_report(
            **common,
            direct=DirectHit("p1", "p2", "sludgebomb", 229, True, None),
        )

        for report, observed_partition in (
            (noncritical, "noncritical"),
            (critical, "critical"),
        ):
            self.assertEqual(report["verdict"], "no_arithmetic_disagreement")
            self.assertEqual(report["observed_criticality_partition"], observed_partition)
            self.assertEqual(report["branch_population"], {
                "total_rendered": 4,
                "reported": 4,
                "dropped": 0,
                "damage_bearing": 4,
                "no_damage": 0,
                "damage_bearing_unsupported": 0,
                "observed_target_direct_damage": 4,
                "without_observed_target_direct_damage": 0,
                "comparable_observed_criticality": 2,
                "excluded_criticality_mismatch": 2,
                "unsupported": 0,
                "state_source_candidate_prestate": 4,
                "state_source_branch_local": 0,
                "criticality": {"critical": 2, "noncritical": 2, "unknown": 0},
            })
            self.assertEqual(
                report["rendered_direct_damages_by_criticality"],
                {"critical": [230, 230], "noncritical": [114, 114], "unknown": []},
            )
            self.assertEqual(len(report["branches"]), 4)
            self.assertNotIn("native_representative_damage", report)

    def test_real_native_present_legal_state_and_missing_state_fail_closed(self) -> None:
        native_engine, native_search, native_reason = attestation._native_modules()
        if native_reason is not None:
            self.skipTest(native_reason)
        state = self._native_state(
            side_one_moves=("swordsdance",),
            side_two_moves=("sludgebomb",),
            side_one_types=("normal",),
            side_two_types=("poison",),
        )
        kwargs = {
            "candidate_index": 0,
            "candidate_state": state,
            "native_engine": native_engine,
            "side_one_choice": "swordsdance",
            "side_two_choice": "sludgebomb",
            "mapper_context": json.dumps(
                {"p1": ["Defender"], "p2": ["Attacker"], "turn": 1}
            ),
            "direct": DirectHit("p2", "p1", "sludgebomb", 114, False, None),
            "dex": self._dex(),
            "slot_sides": {"p1": "side_one", "p2": "side_two"},
            "pre_hit_hp": 500,
        }
        present = _candidate_report(native_search=native_search, **kwargs)
        self.assertEqual(present["verdict"], "no_arithmetic_disagreement")
        self.assertEqual(present["branch_population"]["total_rendered"], 4)
        self.assertEqual(present["branch_population"]["state_source_branch_local"], 4)
        self.assertEqual(present["branch_population"]["unsupported"], 0)

        class StateStrippingSearch:
            @staticmethod
            def branch_events(*args: object) -> str:
                rendered = json.loads(native_search.branch_events(*args))
                for branch in rendered["branches"]:
                    branch.pop("legal_roll_state", None)
                return json.dumps(rendered)

        absent = _candidate_report(native_search=StateStrippingSearch(), **kwargs)
        self.assertEqual(absent["verdict"], "comparison_limit")
        self.assertEqual(absent["reason"], "unsupported_rendered_branch_population")
        self.assertEqual(absent["branch_population"]["total_rendered"], 4)
        self.assertEqual(absent["branch_population"]["damage_bearing"], 4)
        self.assertEqual(absent["branch_population"]["unsupported"], 4)
        self.assertEqual(
            {branch.get("unsupported_reason") for branch in absent["branches"]},
            {"missing_required_legal_roll_state"},
        )
        self.assertEqual(
            absent["rendered_direct_damages_by_criticality"],
            {"critical": [230, 230], "noncritical": [114, 114], "unknown": []},
        )

    def test_real_native_modifier_contexts_reach_exact_physical_and_special_paths(self) -> None:
        native_engine, native_search, native_reason = attestation._native_modules()
        if native_reason is not None:
            self.skipTest(native_reason)
        cases = (
            (
                self._native_state(
                    side_one_moves=("sludgebomb",),
                    side_two_moves=("splash",),
                    side_one_types=("poison",),
                    side_two_types=("grass",),
                    side_one_item="choiceband",
                    side_two_hp=1000,
                    side_two_maxhp=1000,
                ),
                "sludgebomb",
                "Physical",
                [[1.5, 1]],
                None,
            ),
            (
                self._native_state(
                    side_one_moves=("fireblast",),
                    side_two_moves=("splash",),
                    side_one_types=("fire",),
                    side_two_types=("grass",),
                    side_one_item="choiceband",
                    side_two_hp=1000,
                    side_two_maxhp=1000,
                    weather="sun",
                ),
                "fireblast",
                "Special",
                [],
                [1.5, 1],
            ),
        )
        for state_text, move, category, attack_mods, weather_mod in cases:
            state = native_engine.State.from_string(state_text)
            context, rolls, limit = _basic_oracle(
                state=state,
                direct=DirectHit("p1", "p2", move, 0, False, None),
                dex=self._dex(),
                slot_sides={"p1": "side_one", "p2": "side_two"},
            )
            self.assertIsNone(limit)
            assert rolls is not None
            report = _candidate_report(
                candidate_index=0,
                candidate_state=state_text,
                native_engine=native_engine,
                native_search=native_search,
                side_one_choice=move,
                side_two_choice="splash",
                mapper_context=json.dumps(
                    {"p1": ["Attacker"], "p2": ["Defender"], "turn": 1}
                ),
                direct=DirectHit("p1", "p2", move, rolls[7], False, None),
                dex=self._dex(),
                slot_sides={"p1": "side_one", "p2": "side_two"},
                pre_hit_hp=1000,
            )
            self.assertEqual(report["verdict"], "no_arithmetic_disagreement")
            self.assertEqual(report["branch_population"]["unsupported"], 0)
            self.assertEqual(context["category"], category)
            self.assertTrue(context["stab"])
            self.assertEqual(context["effectiveness"], 2.0)
            self.assertEqual(context["attack_mods"], attack_mods)
            self.assertEqual(context["weather_mod"], weather_mod)
            self.assertEqual(report["oracle_context"], context)

    def test_damage_bearing_branch_without_selected_target_is_not_dropped(self) -> None:
        native_engine, native_search, native_reason = attestation._native_modules()
        if native_reason is not None:
            self.skipTest(native_reason)
        state = self._native_state(
            side_one_moves=("sludgebomb",),
            side_two_moves=("splash",),
        )

        class ForeignDamageSearch:
            @staticmethod
            def branch_events(*args: object) -> str:
                rendered = json.loads(native_search.branch_events(*args))
                rendered["branches"].append({
                    "percentage": 0.0,
                    "events": ["|-damage|p1a: Attacker|490/500"],
                    "lossy": [],
                })
                return json.dumps(rendered)

        report = _candidate_report(
            candidate_index=0,
            candidate_state=state,
            native_engine=native_engine,
            native_search=ForeignDamageSearch(),
            side_one_choice="sludgebomb",
            side_two_choice="splash",
            mapper_context=json.dumps(
                {"p1": ["Attacker"], "p2": ["Defender"], "turn": 1}
            ),
            direct=DirectHit("p1", "p2", "sludgebomb", 114, False, None),
            dex=self._dex(),
            slot_sides={"p1": "side_one", "p2": "side_two"},
            pre_hit_hp=500,
        )
        self.assertEqual(report["verdict"], "comparison_limit")
        self.assertEqual(report["branch_population"]["total_rendered"], 5)
        self.assertEqual(report["branch_population"]["damage_bearing"], 5)
        self.assertEqual(
            report["branch_population"]["without_observed_target_direct_damage"], 1
        )
        self.assertEqual(report["branch_population"]["unsupported"], 1)
        self.assertEqual(
            report["branches"][-1]["unsupported_reason"],
            "damage_bearing_branch_without_observed_target_direct_damage",
        )

    def test_real_native_ko_clamped_unknown_crit_is_a_population_limit(self) -> None:
        native_engine, native_search, native_reason = attestation._native_modules()
        if native_reason is not None:
            self.skipTest(native_reason)
        state = self._native_state(
            side_one_moves=("sludgebomb",),
            side_two_moves=("splash",),
            side_two_hp=10,
        )
        report = _candidate_report(
            candidate_index=0,
            candidate_state=state,
            native_engine=native_engine,
            native_search=native_search,
            side_one_choice="sludgebomb",
            side_two_choice="splash",
            mapper_context=json.dumps(
                {"p1": ["Attacker"], "p2": ["Defender"], "turn": 1}
            ),
            direct=DirectHit("p1", "p2", "sludgebomb", 10, False, None, True),
            dex=self._dex(),
            slot_sides={"p1": "side_one", "p2": "side_two"},
            pre_hit_hp=10,
        )

        self.assertEqual(report["verdict"], "comparison_limit")
        self.assertEqual(report["reason"], "unsupported_rendered_branch_population")
        self.assertEqual(report["branch_population"]["total_rendered"], 1)
        self.assertEqual(report["branch_population"]["unsupported"], 1)
        self.assertEqual(
            report["branches"][0]["unsupported_reason"],
            "unknown_or_unlabeled_criticality",
        )
        self.assertEqual(
            report["rendered_direct_damages_by_criticality"]["unknown"], [10]
        )

    def test_supported_move_allowlist_is_audited_against_randbat_universe(self) -> None:
        showdown_root = Path(
            os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT
        )
        sets_path = showdown_root / "data" / "random-battles" / "gen3" / "sets.json"
        if not sets_path.is_file():
            self.skipTest(f"Gen 3 randbat source unavailable: {sets_path}")
        payload = json.loads(sets_path.read_text(encoding="utf-8"))
        move_universe = {
            normalize_id(move)
            for species in payload.values()
            for candidate in species["sets"]
            for move in candidate["movepool"]
        }

        self.assertEqual(len(move_universe), 125)
        self.assertEqual(
            set(EXACT_SUPPORTED_MOVE_SPECS), {"sludgebomb", "fireblast"}
        )
        self.assertLessEqual(set(EXACT_SUPPORTED_MOVE_SPECS), move_universe)
        self.assertTrue({
            "bonemerang",
            "dragonrage",
            "frustration",
            "hiddenpowerice",
            "magnitude",
            "return",
            "rollout",
            "weatherball",
        }.isdisjoint(EXACT_SUPPORTED_MOVE_SPECS))

    def test_target_selection_requires_the_transition_diverged_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(
                json.dumps({
                    "repros": [
                        {"seed": 7, "step": 9, "kind": "engine_error"},
                        {"seed": 7, "step": 9, "kind": "transition_diverged"},
                        {"seed": 8, "step": 9, "kind": "transition_diverged"},
                    ]
                }),
                encoding="utf-8",
            )
            rows = attestation._rows([report], {(7, 9)})

        self.assertEqual(
            [(row["seed"], row["step"], row["kind"]) for row in rows],
            [(7, 9, "transition_diverged")],
        )

    def test_pure_helpers_collect_without_a_native_wheel(self) -> None:
        self.assertFalse(hasattr(attestation, "poke_engine"))
        with patch.object(attestation.importlib, "import_module", side_effect=ModuleNotFoundError("no wheel")):
            self.assertEqual(attestation._native_modules()[2], "native_modules_unavailable:ModuleNotFoundError")

    def test_stale_or_missing_provenance_refuses_to_emit_attestation(self) -> None:
        with patch.object(attestation, "assert_fresh", side_effect=SystemExit("stale build")):
            with self.assertRaisesRegex(SystemExit, "stale build"):
                attestation.main([
                    "--report", "missing.json", "--target", "1/2", "--showdown-root", "showdown",
                ])
        with patch.object(attestation.subprocess, "run", side_effect=OSError("git unavailable")):
            with self.assertRaisesRegex(SystemExit, "cannot record source provenance"):
                attestation._source_provenance()

    def test_showdown_provenance_hashes_resolution_inputs_and_rejects_dirty_tree(self) -> None:
        required = (
            "dist/sim/index.js",
            "dist/sim/dex.js",
            "dist/sim/dex-data.js",
            "dist/data/moves.js",
            "dist/data/pokedex.js",
            "dist/data/typechart.js",
            "dist/data/abilities.js",
            "dist/data/items.js",
            "dist/data/mods/gen3/moves.js",
            "dist/data/mods/gen3/scripts.js",
            "dist/data/mods/gen3/abilities.js",
            "dist/data/mods/gen3/items.js",
            "data/random-battles/gen3/sets.json",
            "dist/data/random-battles/gen3/teams.js",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in required + (
                "dist/lib/utils.js",
                "dist/data/conditions.js",
                "dist/data/mods/gen3/conditions.js",
                "dist/data/mods/gen4/scripts.js",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"// {relative}\n", encoding="utf-8")
            for command in (
                ["git", "init", "-q"],
                ["git", "config", "user.name", "Attestation Test"],
                ["git", "config", "user.email", "attestation@example.invalid"],
                ["git", "add", "."],
                ["git", "commit", "-q", "-m", "fixture"],
            ):
                subprocess.run(command, cwd=root, check=True)

            paths = {
                str(path.relative_to(root)) for path in _showdown_dependency_paths(root)
            }
            provenance = _showdown_source_provenance(str(root))

            self.assertLessEqual(set(required), paths)
            self.assertIn("dist/lib/utils.js", paths)
            self.assertIn("dist/data/conditions.js", paths)
            self.assertIn("dist/data/mods/gen3/conditions.js", paths)
            self.assertIn("dist/data/mods/gen4/scripts.js", paths)
            self.assertEqual(
                {entry["path"] for entry in provenance["inputs"]}, paths
            )
            self.assertEqual(len(provenance["git_commit"]), 40)
            self.assertTrue(provenance["git_clean"])
            self.assertEqual(len(provenance["content_sha256"]), 64)

            (root / "dist/data/mods/gen3/moves.js").write_text(
                "// dirty resolution input\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "Showdown checkout is dirty"):
                _showdown_source_provenance(str(root))

    def test_emitted_json_has_deterministic_provenance_and_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "input.json"
            report.write_text(json.dumps({"repros": []}), encoding="utf-8")
            output = Path(directory) / "attestation.json"
            report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
            capture = io.StringIO()
            with (
                patch.object(attestation, "assert_fresh"),
                patch.object(attestation, "_source_provenance", return_value={"commit": "a" * 40, "tree_clean": True, "producer": {}}),
                patch.object(attestation, "_showdown_source_provenance", return_value={"content_sha256": "c" * 64, "inputs": [], "git_commit": None, "git_clean": None}),
                patch.object(attestation, "compute_fingerprint", return_value={"fingerprint": "b" * 64}),
                patch.object(attestation, "load_showdown_dex", return_value=object()),
                redirect_stdout(capture),
            ):
                self.assertEqual(
                    attestation.main([
                        "--report", str(report), "--target", "2/3", "--target", "1/4",
                        "--showdown-root", "showdown", "--json", str(output),
                    ]),
                    0,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "pokezero.damage-arithmetic-tail-attestation/v4")
        report_label = attestation._path_label(report)
        self.assertEqual(payload["input_reports"], [{"path": report_label, "sha256": report_hash}])
        self.assertEqual(payload["command"], [
            "python", "scripts/attest_damage_arithmetic_tail.py", "--report", report_label,
            "--target", "1/4", "--target", "2/3", "--showdown-root", "showdown",
        ])
        self.assertEqual([row["verdict"] for row in payload["targets"]], ["comparison_limit", "comparison_limit"])
        self.assertEqual(payload["showdown_source"]["content_sha256"], "c" * 64)
