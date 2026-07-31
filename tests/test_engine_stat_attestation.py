from __future__ import annotations

from types import SimpleNamespace
import unittest

from pokezero.engine_stat_attestation import attest_damage_stat_inputs
from pokezero.poke_engine_adapter import BattleSpec, MoveSpec, PokemonSpec, SideSpec


def _member(*, attack: int = 100) -> PokemonSpec:
    return PokemonSpec(
        id="marill",
        level=50,
        types=("water", "typeless"),
        hp=120,
        maxhp=120,
        attack=attack,
        defense=110,
        special_attack=90,
        special_defense=100,
        speed=80,
        ability="hugepower",
        item="choiceband",
        status="burn",
        moves=(MoveSpec(id="tackle", pp=32),),
    )


def _native_member(member: PokemonSpec, *, attack: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=member.id,
        attack=member.attack if attack is None else attack,
        defense=member.defense,
        special_attack=member.special_attack,
        special_defense=member.special_defense,
        speed=member.speed,
        ability=member.ability,
        item=member.item,
        status=member.status,
        types=member.types,
    )


def _native_side(member: PokemonSpec, *, attack: int | None = None, attack_boost: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        pokemon=(_native_member(member, attack=attack),),
        attack_boost=attack_boost,
        defense_boost=-1,
        special_attack_boost=0,
        special_defense_boost=0,
        speed_boost=1,
    )


class DamageStatAttestationTests(unittest.TestCase):
    def _spec(self) -> BattleSpec:
        member = _member()
        side = SideSpec(
            pokemon=(member,),
            boosts={"attack": 2, "defense": -1, "speed": 1},
        )
        return BattleSpec(side_one=side, side_two=side)

    def test_accepts_identical_base_stats_and_damage_modifiers(self) -> None:
        spec = self._spec()
        state = SimpleNamespace(
            side_one=_native_side(spec.side_one.pokemon[0]),
            side_two=_native_side(spec.side_two.pokemon[0]),
        )

        result = attest_damage_stat_inputs(spec, state)

        self.assertTrue(result.matches)
        self.assertEqual(result.mismatches, ())
        self.assertEqual(result.actual["side_one.active_boosts"]["attack_boost"], 2)

    def test_reports_base_stat_and_stage_corruption_by_exact_path(self) -> None:
        spec = self._spec()
        state = SimpleNamespace(
            side_one=_native_side(spec.side_one.pokemon[0], attack=99, attack_boost=1),
            side_two=_native_side(spec.side_two.pokemon[0]),
        )

        result = attest_damage_stat_inputs(spec, state)

        self.assertFalse(result.matches)
        self.assertIn("side_one.pokemon[0].attack: expected 100, got 99", result.mismatches)
        self.assertIn("side_one.attack_boost: expected 2, got 1", result.mismatches)
