from __future__ import annotations

from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_stat_attestation import (
    attest_battle_spec_transport,
    attest_battle_spec_transport_variants,
)
from pokezero.poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from pokezero.poke_engine_backend import probe_poke_engine


def _member(*, identifier: str = "marill", level: int = 50, pp: int = 32) -> PokemonSpec:
    return PokemonSpec(
        id=identifier,
        level=level,
        types=("water", "typeless"),
        hp=120,
        maxhp=120,
        attack=100,
        defense=110,
        special_attack=90,
        special_defense=100,
        speed=80,
        ability="hugepower",
        item="choiceband",
        status="burn",
        weight_kg=28.0,
        moves=(MoveSpec(id="tackle", pp=pp),),
    )


def _native_member(member: PokemonSpec, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": f"PokemonId.{member.id.upper()}",
        "level": member.level,
        "hp": member.hp,
        "maxhp": member.maxhp,
        "attack": member.attack,
        "defense": member.defense,
        "special_attack": member.special_attack,
        "special_defense": member.special_defense,
        "speed": member.speed,
        "ability": "Ability.HUGEPOWER",
        "base_ability": "Ability.HUGEPOWER",
        "item": "Item.CHOICEBAND",
        "status": "Status.BURN",
        "types": ("Type.WATER", "Type.TYPELESS"),
        "base_types": ("Type.WATER", "Type.TYPELESS"),
        "moves": (SimpleNamespace(id="MoveName.TACKLE", pp=member.moves[0].pp, disabled=False),),
        "weight_kg": member.weight_kg,
        "rest_turns": member.rest_turns,
        "sleep_turns": member.sleep_turns,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _native_side(
    member: PokemonSpec,
    *,
    active_index: str = "0",
    side_conditions: object | None = None,
    side_overrides: dict[str, object] | None = None,
    **member_overrides: object,
) -> SimpleNamespace:
    values: dict[str, object] = {
        "pokemon": (_native_member(member, **member_overrides),),
        "active_index": active_index,
        "attack_boost": 2,
        "defense_boost": -1,
        "special_attack_boost": 0,
        "special_defense_boost": 0,
        "speed_boost": 1,
        "side_conditions": side_conditions or SimpleNamespace(),
        "volatile_statuses": set(),
        "substitute_health": 0,
        "force_switch": False,
        "wish": (0, 0),
        "baton_passing": False,
        "slow_uturn_move": False,
        "switch_out_move_second_saved_move": "none",
        "last_used_move": "move:none",
        "volatile_status_durations": SimpleNamespace(),
    }
    values.update(side_overrides or {})
    return SimpleNamespace(**values)


class BattleSpecTransportAttestationTests(unittest.TestCase):
    def _spec(self) -> BattleSpec:
        member = _member()
        side = SideSpec(
            pokemon=(member,),
            boosts={"attack": 2, "defense": -1, "speed": 1},
        )
        return BattleSpec(side_one=side, side_two=side)

    def _fake_state(self, spec: BattleSpec, **side_one_overrides: object) -> SimpleNamespace:
        side_one = _native_side(spec.side_one.pokemon[0], **side_one_overrides)
        return SimpleNamespace(
            side_one=side_one,
            side_two=_native_side(spec.side_two.pokemon[0]),
            weather="Weather.NONE",
            weather_turns_remaining=0,
            terrain="Terrain.NONE",
            trick_room=False,
        )

    def test_normalizes_native_enum_identifiers_using_adapter_ids(self) -> None:
        spec = self._spec()
        result = attest_battle_spec_transport(spec, self._fake_state(spec))

        self.assertTrue(result.matches)
        self.assertEqual(result.actual["side_one.pokemon[0]"]["id"], "marill")
        self.assertEqual(result.actual["side_one.pokemon[0]"]["moves"], (("tackle", 32, False),))

    def test_reports_level_move_and_active_index_mismatches(self) -> None:
        spec = self._spec()
        state = self._fake_state(
            spec,
            active_index="1",
            level=49,
            moves=(SimpleNamespace(id="MoveName.TACKLE", pp=31, disabled=False),),
        )

        result = attest_battle_spec_transport(spec, state)

        self.assertFalse(result.matches)
        self.assertIn("side_one.transport.active_index: expected 0, got 1", result.mismatches)
        self.assertIn("side_one.pokemon[0].level: expected 50, got 49", result.mismatches)
        self.assertIn(
            "side_one.pokemon[0].moves: expected (('tackle', 32, False),), got (('tackle', 31, False),)",
            result.mismatches,
        )

    def test_reports_spurious_native_side_condition(self) -> None:
        spec = self._spec()
        state = self._fake_state(spec, side_conditions=SimpleNamespace(stealth_rock=1))

        result = attest_battle_spec_transport(spec, state)

        self.assertFalse(result.matches)
        self.assertIn(
            "side_one.transport.side_conditions: expected {}, got {'stealth_rock': 1}",
            result.mismatches,
        )

    def test_compares_direct_hp_transition_side_fields(self) -> None:
        member = _member()
        side = SideSpec(
            pokemon=(member,),
            boosts={"attack": 2, "defense": -1, "speed": 1},
            volatile_statuses=("substitute",),
            substitute_health=30,
            wish=(1, 45),
            baton_passing=True,
            slow_uturn_move=True,
            switch_out_move_second_saved_move="tackle",
            last_used_move="move:0",
            volatile_status_durations={"encore": 1},
        )
        spec = BattleSpec(side_one=side, side_two=side)
        side_values = {
            "volatile_statuses": {"VolatileStatus.SUBSTITUTE"},
            "substitute_health": 30,
            "wish": (1, 45),
            "baton_passing": True,
            "slow_uturn_move": True,
            "switch_out_move_second_saved_move": "MoveName.TACKLE",
            "last_used_move": "move:0",
            "volatile_status_durations": SimpleNamespace(encore=1),
        }
        state = SimpleNamespace(
            side_one=_native_side(member, side_overrides=side_values),
            side_two=_native_side(member, side_overrides=side_values),
            weather="Weather.NONE",
            weather_turns_remaining=0,
            terrain="Terrain.NONE",
            trick_room=False,
        )

        self.assertTrue(attest_battle_spec_transport(spec, state).matches)
        state.side_one.substitute_health = 29
        self.assertIn(
            "side_one.transport.substitute_health: expected 30, got 29",
            attest_battle_spec_transport(spec, state).mismatches,
        )

    def test_dropped_variant_is_a_structured_non_clearance(self) -> None:
        spec = self._spec()

        result = attest_battle_spec_transport_variants(
            (spec,),
            (),
            variant_construction=(
                {"variant_index": 0, "status": "dropped", "error_type": "ValueError"},
            ),
        )

        self.assertEqual(result["status"], "dropped_variant_construction")
        self.assertEqual(result["comparison_states"], 0)
        self.assertEqual(result["hidden_counter_variants"], 0)

    def test_real_adapter_positive_and_negative_controls(self) -> None:
        if not probe_poke_engine().ready:
            self.skipTest("poke-engine is not installed/ready")
        bench = _member(identifier="azumarill")
        active = _member()
        spec = BattleSpec(
            side_one=SideSpec(
                pokemon=(bench, active),
                active_index=1,
                side_conditions={"spikes": 2},
                boosts={"attack": 2},
            ),
            side_two=SideSpec(pokemon=(active,)),
        )
        state = build_poke_engine_state(spec)
        self.assertTrue(attest_battle_spec_transport(spec, state).matches)

        bad_level = replace(spec, side_one=replace(spec.side_one, pokemon=(bench, replace(active, level=49))))
        bad_move = replace(
            spec,
            side_one=replace(spec.side_one, pokemon=(bench, replace(active, moves=(MoveSpec("tackle", 31),)))),
        )
        bad_active = replace(spec, side_one=replace(spec.side_one, active_index=0))
        for label, corrupted in (("level", bad_level), ("move", bad_move), ("active", bad_active)):
            with self.subTest(label=label):
                self.assertFalse(attest_battle_spec_transport(corrupted, state).matches)


class TransportAttestationScriptTests(unittest.TestCase):
    def test_json_result_carries_reproducible_command_and_provenance(self) -> None:
        script_path = Path(__file__).parents[1] / "scripts" / "attest_materialized_damage_stats.py"
        module_spec = importlib.util.spec_from_file_location("attest_materialized_damage_stats_test", script_path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        class FakeEnv:
            def close(self) -> None:
                pass

        target = {"status": "target_diverged_transport_attested", "seed": 7, "step": 3}
        output = io.StringIO()
        with (
            patch.object(module, "assert_fresh"),
            patch.object(module, "load_showdown_dex", return_value=object()),
            patch.object(module, "LocalShowdownConfig", return_value=object()),
            patch.object(module, "LocalShowdownEnv", return_value=FakeEnv()),
            patch.object(module, "EngineMctsPolicy", return_value=object()),
            patch.object(module, "Gen3RandbatSource", SimpleNamespace(from_showdown_root=lambda _root: object())),
            patch.object(module, "attest_target", return_value=target),
            patch.object(
                module,
                "_checkpoint_provenance",
                return_value={
                    "source_commit": "a" * 40,
                    "engine_fingerprint": "b" * 64,
                    "image_commit": "c" * 40,
                },
            ),
            patch("sys.stdout", output),
        ):
            self.assertEqual(module.main(["--target", "7/3"]), 0)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema_version"], "pokezero.battle_spec_transport_attestation.v2")
        self.assertIn("--target 7/3", payload["command"])
        self.assertEqual(payload["provenance"]["engine_fingerprint"], "b" * 64)
        self.assertEqual(payload["targets"], [target])
