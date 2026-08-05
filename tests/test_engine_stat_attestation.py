from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.engine_stat_attestation import (
    attest_battle_spec_transport,
    attest_battle_spec_transport_variants,
    build_and_attest_battle_spec_transport,
)
from pokezero.poke_engine_adapter import (
    BattleSpec,
    MoveSpec,
    PokemonSpec,
    SideSpec,
    build_poke_engine_state,
)
from pokezero.poke_engine_backend import probe_poke_engine


def _load_attestation_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "attest_materialized_damage_stats.py"
    module_spec = importlib.util.spec_from_file_location(
        "attest_materialized_damage_stats_test",
        script_path,
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


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
        "nature": f"Nature.{(member.nature or 'serious').upper()}",
        "gender": {
            None: "Gender.NONE",
            "M": "Gender.MALE",
            "F": "Gender.FEMALE",
            "N": "Gender.NONE",
        }[member.gender],
        "status": "Status.BURN",
        "types": ("Type.WATER", "Type.TYPELESS"),
        "base_types": ("Type.WATER", "Type.TYPELESS"),
        "moves": (SimpleNamespace(id="MoveName.TACKLE", pp=member.moves[0].pp, disabled=False),),
        "weight_kg": member.weight_kg,
        "rest_turns": member.rest_turns,
        "sleep_turns": member.sleep_turns,
        "pre_transform": "",
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
        "accuracy_boost": 3,
        "evasion_boost": -2,
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


def _corrupting_constructor_module(
    *,
    target_kind: str | None = None,
    mutate: Callable[[dict[str, object]], None] | None = None,
) -> ModuleType:
    """Production-shaped constructor fake with one optional targeted corruption."""

    module = ModuleType("corrupting_poke_engine")
    corrupted = False

    def maybe_corrupt(kind: str, values: dict[str, object]) -> None:
        nonlocal corrupted
        if corrupted or kind != target_kind or mutate is None:
            return
        pokemon = values.get("pokemon")
        if kind == "Pokemon":
            eligible = values.get("id") == "marill"
        elif kind == "Move":
            eligible = values.get("id") == "tackle"
        elif kind == "Side":
            eligible = bool(pokemon) and getattr(pokemon[0], "id", None) == "marill"
        elif kind == "State":
            side = values.get("side_one")
            party = getattr(side, "pokemon", ())
            eligible = bool(party) and getattr(party[0], "id", None) == "marill"
        else:
            eligible = True
        if eligible:
            mutate(values)
            corrupted = True

    def move(**kwargs: object) -> SimpleNamespace:
        values = {"id": "none", "pp": 0, "disabled": False, **kwargs}
        maybe_corrupt("Move", values)
        return SimpleNamespace(**values)

    def pokemon(**kwargs: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "id": "pikachu",
            "level": 50,
            "types": ("normal", "typeless"),
            "base_types": ("normal", "typeless"),
            "hp": 100,
            "maxhp": 100,
            "attack": 100,
            "defense": 100,
            "special_attack": 100,
            "special_defense": 100,
            "speed": 100,
            "ability": "none",
            "base_ability": "",
            "item": "none",
            "nature": "serious",
            "gender": "none",
            "status": "none",
            "moves": (),
            "weight_kg": 0.0,
            "rest_turns": 0,
            "sleep_turns": 0,
            "pre_transform": "",
            **kwargs,
        }
        if not values["base_ability"]:
            values["base_ability"] = values["ability"]
        maybe_corrupt("Pokemon", values)
        return SimpleNamespace(**values)

    def side_conditions(**kwargs: object) -> SimpleNamespace:
        values = dict(kwargs)
        maybe_corrupt("SideConditions", values)
        return SimpleNamespace(**values)

    def durations(**kwargs: object) -> SimpleNamespace:
        values = dict(kwargs)
        maybe_corrupt("VolatileStatusDurations", values)
        return SimpleNamespace(**values)

    def side(**kwargs: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "pokemon": (),
            "active_index": "0",
            "attack_boost": 0,
            "defense_boost": 0,
            "special_attack_boost": 0,
            "special_defense_boost": 0,
            "speed_boost": 0,
            "accuracy_boost": 0,
            "evasion_boost": 0,
            "side_conditions": SimpleNamespace(),
            "volatile_statuses": set(),
            "substitute_health": 0,
            "force_switch": False,
            "wish": (0, 0),
            "baton_passing": False,
            "slow_uturn_move": False,
            "switch_out_move_second_saved_move": "none",
            "last_used_move": "move:none",
            "volatile_status_durations": SimpleNamespace(),
            **kwargs,
        }
        maybe_corrupt("Side", values)
        return SimpleNamespace(**values)

    class State:
        def __init__(self, **kwargs: object) -> None:
            values: dict[str, object] = {
                "side_one": side(),
                "side_two": side(),
                "weather": "none",
                "weather_turns_remaining": 0,
                "terrain": "none",
                "trick_room": False,
                **kwargs,
            }
            maybe_corrupt("State", values)
            self.__dict__.update(values)
            self._serialized: str | None = None

        def to_string(self) -> str:
            if self._serialized is not None:
                return self._serialized
            records = [
                (
                    f"{getattr(member, 'pre_transform', '')}:"
                    f"rest={getattr(member, 'rest_turns', 0)}"
                )
                for native_side in (self.side_one, self.side_two)
                for member in getattr(native_side, "pokemon", ())
            ]
            return "state|" + "|".join(records)

        @classmethod
        def from_string(cls, value: str) -> "State":
            restored = cls.__new__(cls)
            restored._serialized = value
            return restored

    module.Move = move
    module.Pokemon = pokemon
    module.SideConditions = side_conditions
    module.VolatileStatusDurations = durations
    module.Side = side
    module.State = State
    return module


class BattleSpecTransportAttestationTests(unittest.TestCase):
    def _spec(self) -> BattleSpec:
        member = _member()
        side = SideSpec(
            pokemon=(member,),
            boosts={
                "attack": 2,
                "defense": -1,
                "speed": 1,
                "accuracy": 3,
                "evasion": -2,
            },
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

    def _rich_spec(self) -> BattleSpec:
        base = PokemonSpec(
            id="ditto",
            level=77,
            types=("normal",),
            hp=90,
            maxhp=90,
            attack=48,
            defense=49,
            special_attack=50,
            special_defense=51,
            speed=52,
            moves=(MoveSpec("transform", 9),),
        )
        member = PokemonSpec(
            id="marill",
            level=50,
            types=("water",),
            base_types=("normal",),
            hp=119,
            maxhp=120,
            attack=100,
            defense=110,
            special_attack=90,
            special_defense=101,
            speed=80,
            ability="hugepower",
            base_ability="thickfat",
            item="choiceband",
            nature="adamant",
            gender="F",
            status="burn",
            rest_turns=1,
            sleep_turns=2,
            weight_kg=28.3,
            moves=(
                MoveSpec(id="tackle", pp=31, disabled=True),
                MoveSpec(id="surf", pp=23),
            ),
            pre_transform=base,
        )
        side = SideSpec(
            pokemon=(member,),
            active_index=0,
            side_conditions={"spikes": 2},
            boosts={
                "attack": 2,
                "defense": -1,
                "special_attack": 1,
                "special_defense": -2,
                "speed": 3,
                "accuracy": -3,
                "evasion": 1,
            },
            volatile_statuses=("substitute",),
            substitute_health=30,
            force_switch=True,
            wish=(1, 45),
            baton_passing=True,
            slow_uturn_move=True,
            switch_out_move_second_saved_move="tackle",
            last_used_move="move:0",
            volatile_status_durations={"encore": 1},
        )
        return BattleSpec(
            side_one=side,
            side_two=side,
            weather="rain",
            weather_turns_remaining=4,
            terrain="none",
            trick_room=True,
        )

    def test_normalizes_native_enum_identifiers_using_adapter_ids(self) -> None:
        spec = self._spec()
        result = attest_battle_spec_transport(spec, self._fake_state(spec))

        self.assertTrue(result.matches)
        self.assertEqual(result.actual["side_one.pokemon[0]"]["id"], "marill")
        self.assertEqual(result.actual["side_one.pokemon[0]"]["moves"], (("tackle", 32, False),))

    def test_accuracy_and_evasion_boosts_have_independent_negative_controls(self) -> None:
        spec = self._spec()
        for field, corrupted in (("accuracy_boost", 2), ("evasion_boost", -1)):
            with self.subTest(field=field):
                state = self._fake_state(
                    spec,
                    side_overrides={field: corrupted},
                )
                result = attest_battle_spec_transport(spec, state)
                self.assertFalse(result.matches)
                self.assertTrue(
                    any(
                        mismatch.startswith(f"side_one.active_boosts.{field}:")
                        for mismatch in result.mismatches
                    )
                )

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

    def test_malformed_unexpected_pre_transform_remains_visible(self) -> None:
        spec = self._spec()
        state = self._fake_state(spec, pre_transform="not-a-restoration-record")

        result = attest_battle_spec_transport(spec, state)

        self.assertFalse(result.matches)
        self.assertEqual(
            result.actual["side_one.pokemon[0]"]["pre_transform"],
            {"invalid_wire": "not-a-restoration-record"},
        )

    def test_compares_direct_hp_transition_side_fields(self) -> None:
        member = _member()
        side = SideSpec(
            pokemon=(member,),
            boosts={
                "attack": 2,
                "defense": -1,
                "speed": 1,
                "accuracy": 3,
                "evasion": -2,
            },
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

    def test_corrupting_native_constructor_is_detected_across_forwarded_fields(self) -> None:
        spec = self._rich_spec()

        def set_value(field: str, value: object):
            return lambda values: values.__setitem__(field, value)

        cases = [
            ("field.weather", "State", set_value("weather", "sun")),
            (
                "field.weather_turns_remaining",
                "State",
                set_value("weather_turns_remaining", 3),
            ),
            ("field.terrain", "State", set_value("terrain", "electric")),
            ("field.trick_room", "State", set_value("trick_room", False)),
            ("side.active_index", "Side", set_value("active_index", "1")),
            ("side.attack", "Side", set_value("attack_boost", 1)),
            ("side.defense", "Side", set_value("defense_boost", 0)),
            ("side.special_attack", "Side", set_value("special_attack_boost", 0)),
            ("side.special_defense", "Side", set_value("special_defense_boost", 0)),
            ("side.speed", "Side", set_value("speed_boost", 2)),
            ("side.accuracy", "Side", set_value("accuracy_boost", -2)),
            ("side.evasion", "Side", set_value("evasion_boost", 0)),
            (
                "side.side_conditions",
                "SideConditions",
                set_value("spikes", 1),
            ),
            (
                "side.volatile_statuses",
                "Side",
                set_value("volatile_statuses", set()),
            ),
            (
                "side.substitute_health",
                "Side",
                set_value("substitute_health", 29),
            ),
            ("side.force_switch", "Side", set_value("force_switch", False)),
            ("side.wish", "Side", set_value("wish", (1, 44))),
            ("side.baton_passing", "Side", set_value("baton_passing", False)),
            ("side.slow_uturn", "Side", set_value("slow_uturn_move", False)),
            (
                "side.saved_move",
                "Side",
                set_value("switch_out_move_second_saved_move", "surf"),
            ),
            ("side.last_move", "Side", set_value("last_used_move", "move:1")),
            (
                "side.volatile_duration",
                "VolatileStatusDurations",
                set_value("encore", 2),
            ),
            ("member.id", "Pokemon", set_value("id", "azumarill")),
            ("member.level", "Pokemon", set_value("level", 49)),
            ("member.types", "Pokemon", set_value("types", ("ice", "typeless"))),
            (
                "member.base_types",
                "Pokemon",
                set_value("base_types", ("water", "typeless")),
            ),
            ("member.hp", "Pokemon", set_value("hp", 118)),
            ("member.maxhp", "Pokemon", set_value("maxhp", 121)),
            ("member.attack", "Pokemon", set_value("attack", 99)),
            ("member.defense", "Pokemon", set_value("defense", 109)),
            (
                "member.special_attack",
                "Pokemon",
                set_value("special_attack", 89),
            ),
            (
                "member.special_defense",
                "Pokemon",
                set_value("special_defense", 100),
            ),
            ("member.speed", "Pokemon", set_value("speed", 79)),
            ("member.ability", "Pokemon", set_value("ability", "thickfat")),
            (
                "member.base_ability",
                "Pokemon",
                set_value("base_ability", "hugepower"),
            ),
            ("member.item", "Pokemon", set_value("item", "leftovers")),
            ("member.nature", "Pokemon", set_value("nature", "serious")),
            ("member.gender", "Pokemon", set_value("gender", "male")),
            ("member.status", "Pokemon", set_value("status", "paralysis")),
            ("member.weight", "Pokemon", set_value("weight_kg", 27.3)),
            ("member.rest_turns", "Pokemon", set_value("rest_turns", 0)),
            # Pins that the ACTUAL side reads the native object, not the spec.
            # Without an entry here, changing `native_member` to `member` in the
            # actual dict makes this field a tautology -- spec compared against
            # itself, always agreeing, a real binding drop invisible. Same shape as
            # the vacuous probe control, one instrument up. Found by review of #1113.
            (
                "member.rest_sleep_pending_refund",
                "Pokemon",
                set_value("rest_sleep_pending_refund", 1),
            ),
            ("member.sleep_turns", "Pokemon", set_value("sleep_turns", 1)),
            (
                "member.pre_transform",
                "Pokemon",
                set_value(
                    "pre_transform",
                    "ditto;47;49;50;51;52;transform:9;none:0;none:0;none:0",
                ),
            ),
            (
                "member.pre_transform.invalid_shape",
                "Pokemon",
                set_value("pre_transform", "ditto;48"),
            ),
            (
                "member.pre_transform.invalid_pp",
                "Pokemon",
                set_value(
                    "pre_transform",
                    "ditto;48;49;50;51;52;transform:nope;none:0;none:0;none:0",
                ),
            ),
            ("move.id", "Move", set_value("id", "surf")),
            ("move.pp", "Move", set_value("pp", 30)),
            ("move.disabled", "Move", set_value("disabled", False)),
        ]
        for label, target_kind, mutate in cases:
            with self.subTest(label=label):
                _, result = build_and_attest_battle_spec_transport(
                    spec,
                    module=_corrupting_constructor_module(
                        target_kind=target_kind,
                        mutate=mutate,
                    ),
                )
                self.assertFalse(result.matches, label)

    def test_production_shaped_constructor_fake_has_a_clean_positive_control(self) -> None:
        _, result = build_and_attest_battle_spec_transport(
            self._rich_spec(),
            module=_corrupting_constructor_module(),
        )

        self.assertTrue(result.matches, result.mismatches)

    def test_member_coverage_is_locked_to_the_pokemon_spec_surface(self) -> None:
        _, result = build_and_attest_battle_spec_transport(
            self._rich_spec(),
            module=_corrupting_constructor_module(),
        )

        self.assertEqual(
            set(result.expected["side_one.pokemon[0]"]),
            set(PokemonSpec.__dataclass_fields__),
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
        base = PokemonSpec(
            id="ditto",
            level=50,
            types=("normal",),
            hp=100,
            maxhp=100,
            attack=48,
            defense=48,
            special_attack=48,
            special_defense=48,
            speed=48,
            moves=(MoveSpec("transform", 10),),
        )
        active = replace(
            _member(),
            nature="adamant",
            gender="F",
            weight_kg=28.3,
            pre_transform=base,
        )
        spec = BattleSpec(
            side_one=SideSpec(
                pokemon=(bench, active),
                active_index=1,
                side_conditions={"spikes": 2},
                boosts={"attack": 2, "accuracy": -3, "evasion": 1},
            ),
            side_two=SideSpec(pokemon=(active,)),
        )
        state = build_poke_engine_state(spec)
        self.assertTrue(attest_battle_spec_transport(spec, state).matches)
        self.assertEqual(state.side_one.accuracy_boost, -3)
        self.assertEqual(state.side_one.evasion_boost, 1)
        self.assertNotEqual(state.side_one.pokemon[1].weight_kg, 28.3)

        bad_level = replace(spec, side_one=replace(spec.side_one, pokemon=(bench, replace(active, level=49))))
        bad_move = replace(
            spec,
            side_one=replace(spec.side_one, pokemon=(bench, replace(active, moves=(MoveSpec("tackle", 31),)))),
        )
        bad_active = replace(spec, side_one=replace(spec.side_one, active_index=0))
        bad_accuracy = replace(
            spec,
            side_one=replace(
                spec.side_one,
                boosts={"attack": 2, "accuracy": -2, "evasion": 1},
            ),
        )
        bad_evasion = replace(
            spec,
            side_one=replace(
                spec.side_one,
                boosts={"attack": 2, "accuracy": -3, "evasion": 0},
            ),
        )
        bad_nature = replace(
            spec,
            side_one=replace(
                spec.side_one,
                pokemon=(bench, replace(active, nature="serious")),
            ),
        )
        bad_gender = replace(
            spec,
            side_one=replace(
                spec.side_one,
                pokemon=(bench, replace(active, gender="M")),
            ),
        )
        bad_pre_transform = replace(
            spec,
            side_one=replace(
                spec.side_one,
                pokemon=(
                    bench,
                    replace(
                        active,
                        pre_transform=replace(base, attack=47),
                    ),
                ),
            ),
        )
        for label, corrupted in (
            ("level", bad_level),
            ("move", bad_move),
            ("active", bad_active),
            ("accuracy", bad_accuracy),
            ("evasion", bad_evasion),
            ("nature", bad_nature),
            ("gender", bad_gender),
            ("pre_transform", bad_pre_transform),
        ):
            with self.subTest(label=label):
                self.assertFalse(attest_battle_spec_transport(corrupted, state).matches)


class TransportAttestationScriptTests(unittest.TestCase):
    def test_clean_public_source_is_hashed_and_dirty_or_mismatched_source_is_rejected(
        self,
    ) -> None:
        module = _load_attestation_script()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subprocess.run(("git", "init", "-q", str(root)), check=True)
            subprocess.run(
                ("git", "-C", str(root), "config", "user.email", "test@example.com"),
                check=True,
            )
            subprocess.run(
                ("git", "-C", str(root), "config", "user.name", "Test"),
                check=True,
            )
            source = root / "source.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(("git", "-C", str(root), "add", "source.py"), check=True)
            subprocess.run(
                ("git", "-C", str(root), "commit", "-qm", "fixture"),
                check=True,
            )
            head = subprocess.run(
                ("git", "-C", str(root), "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            provenance = module._public_source_provenance(
                root,
                source_commit=head,
            )
            self.assertEqual(
                provenance["public_source_tree_status"],
                "clean_tracked_checkout",
            )
            self.assertEqual(provenance["public_source_checkout_head"], head)
            self.assertEqual(len(provenance["public_source_tree_sha256"]), 64)

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                module._public_source_provenance(
                    root,
                    source_commit="a" * 40,
                )

            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "dirty tracked public source tree",
            ) as raised:
                module._public_source_provenance(
                    root,
                    source_commit=head,
                )
            self.assertIn("source.py", str(raised.exception))

    def test_public_source_without_git_gets_an_explicit_content_hash(self) -> None:
        module = _load_attestation_script()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / "src" / "pokezero"
            scripts = root / "scripts"
            package.mkdir(parents=True)
            scripts.mkdir()
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (scripts / "audit.py").write_text("VALUE = 2\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

            provenance = module._public_source_provenance(
                root,
                source_commit="a" * 40,
            )
            (package / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
            changed = module._public_source_provenance(
                root,
                source_commit="a" * 40,
            )

        self.assertEqual(
            provenance["public_source_tree_status"],
            "explicit_hash_without_git",
        )
        self.assertEqual(len(provenance["public_source_tree_sha256"]), 64)
        self.assertNotEqual(
            provenance["public_source_tree_sha256"],
            changed["public_source_tree_sha256"],
        )

    def test_resolved_showdown_source_requires_and_records_content_hash(self) -> None:
        module = _load_attestation_script()
        metadata = {
            "format_id": "gen3randombattle",
            "generation": 3,
            "showdown_root": "/showdown",
            "sets_path": "/showdown/sets.json",
            "generator_path": "/showdown/teams.js",
            "source_hash": "a1b2c3d4",
        }
        source = SimpleNamespace(
            metadata=SimpleNamespace(to_payload=lambda: dict(metadata))
        )

        self.assertEqual(
            module._resolved_showdown_provenance(source),
            {
                "showdown_randbat_source_hash": "a1b2c3d4",
                "showdown_randbat_source": metadata,
            },
        )
        source.metadata = SimpleNamespace(
            to_payload=lambda: {**metadata, "source_hash": ""}
        )
        with self.assertRaisesRegex(RuntimeError, "source hash"):
            module._resolved_showdown_provenance(source)

    def test_real_replay_materializes_and_attests_transport_with_source_hash(self) -> None:
        from pokezero.dex import load_showdown_dex
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.env import BattleStartOverride
        from pokezero.local_showdown import (
            DEFAULT_SHOWDOWN_ROOT,
            LocalShowdownConfig,
            LocalShowdownEnv,
        )
        from pokezero.randbat import Gen3RandbatSource

        root = Path(DEFAULT_SHOWDOWN_ROOT)
        if not (root / "dist" / "sim" / "index.js").is_file() or not shutil.which("node"):
            self.skipTest("built local Pokemon Showdown checkout is unavailable")
        if not probe_poke_engine().ready:
            self.skipTest("poke-engine is not installed/ready")

        module = _load_attestation_script()
        dex = load_showdown_dex(root)
        source = Gen3RandbatSource.from_showdown_root(root)
        policy = EngineMctsPolicy(
            dex=dex,
            set_source=source,
            config=EngineMctsConfig(worlds=1, search_time_ms=1),
        )
        env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=root, set_belief_source=True)
        )
        try:
            env.reset(seed=987654, format_id="gen3randombattle")
            true_teams = module._true_teams_from_bridge_snapshot(
                env.snapshot().bridge_snapshot
            )
            packed = {slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}
            override = BattleStartOverride(player_teams=packed)
            teams = {
                slot: module.unpack_team(packed[slot])
                for slot in ("p1", "p2")
            }
            cumulative = list(env.protocol_lines)
            prepared = None
            replayed_steps = 0
            for _ in range(4):
                requested = tuple(env.requested_players())
                actions = {
                    player: next(
                        index
                        for index, allowed in enumerate(env.legal_actions(player))
                        if allowed
                    )
                    for player in requested
                }
                if replayed_steps and set(requested) == {"p1", "p2"}:
                    prepared = module._prepare_boundary(
                        env=env,
                        flags_policy=policy,
                        override=override,
                        teams=teams,
                        dex=dex,
                        actions=actions,
                        cumulative=cumulative,
                        counts=Counter(),
                        approximate_sleep=False,
                        hidden_counter_support=True,
                    )
                    if prepared is not None:
                        break
                env.step(actions)
                replayed_steps += 1
                cumulative = list(env.protocol_lines)

            self.assertIsNotNone(prepared, "real replay produced no materializable boundary")
            assert prepared is not None
            transport = attest_battle_spec_transport_variants(
                prepared["specs"],
                prepared["states"],
                variant_construction=prepared.get("variant_construction") or (),
            )
            self.assertEqual(transport["status"], "transport_attested", transport)
            self.assertGreater(transport["comparison_states"], 0)
            provenance = module._resolved_showdown_provenance(source)
            self.assertEqual(
                provenance["showdown_randbat_source_hash"],
                source.metadata.source_hash,
            )
            self.assertTrue(provenance["showdown_randbat_source_hash"])
        finally:
            env.close()

    def test_json_result_carries_reproducible_command_and_provenance(self) -> None:
        module = _load_attestation_script()

        class FakeEnv:
            def close(self) -> None:
                pass

        target = {"status": "target_diverged_transport_attested", "seed": 7, "step": 3}
        source = SimpleNamespace(
            metadata=SimpleNamespace(
                to_payload=lambda: {
                    "format_id": "gen3randombattle",
                    "generation": 3,
                    "showdown_root": "/showdown",
                    "sets_path": "/showdown/sets.json",
                    "generator_path": "/showdown/teams.js",
                    "source_hash": "d" * 16,
                }
            )
        )
        output = io.StringIO()
        public_source = {
            "public_source_tree_status": "clean_tracked_checkout",
            "public_source_tree_sha256": "e" * 64,
            "public_source_tree_hash_scope": "git_ls_files",
            "public_source_checkout_head": "a" * 40,
        }
        with (
            patch.object(module, "assert_fresh"),
            patch.object(
                module,
                "_public_source_provenance",
                return_value=public_source,
            ) as source_provenance,
            patch.object(module, "load_showdown_dex", return_value=object()),
            patch.object(module, "LocalShowdownConfig", return_value=object()),
            patch.object(module, "LocalShowdownEnv", return_value=FakeEnv()),
            patch.object(module, "EngineMctsPolicy", return_value=object()),
            patch.object(
                module,
                "Gen3RandbatSource",
                SimpleNamespace(from_showdown_root=lambda _root: source),
            ),
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
        self.assertEqual(payload["schema_version"], "pokezero.battle_spec_transport_attestation.v3")
        self.assertIn("--target 7/3", payload["command"])
        self.assertEqual(payload["provenance"]["engine_fingerprint"], "b" * 64)
        self.assertEqual(payload["provenance"]["showdown_randbat_source_hash"], "d" * 16)
        self.assertEqual(payload["provenance"]["public_source_tree_sha256"], "e" * 64)
        self.assertEqual(payload["targets"], [target])
        source_provenance.assert_called_once_with(
            module.REPO_ROOT,
            source_commit="a" * 40,
        )
