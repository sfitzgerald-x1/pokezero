from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.dex import load_showdown_dex_cached, showdown_dex_from_payload


def _move_effect(move_id: str, **fields):
    """Build a one-move dex from a raw move payload and return its derived MoveInfo."""
    payload = {"id": move_id, "name": move_id, "type": "Normal", "category": "Status", **fields}
    dex = showdown_dex_from_payload({"moves": {move_id: payload}, "species": {}, "typeChart": {}})
    return dex.move_info(move_id)


class MoveEffectLabelTest(unittest.TestCase):
    """The unified move-effect label: target-explicit, magnitude-enumerated, primary or secondary."""

    def test_self_stat_raise_magnitude_enumeration(self) -> None:
        howl = _move_effect("howl", topBoosts={"atk": 1}, target="self")
        swords = _move_effect("swordsdance", topBoosts={"atk": 2}, target="self")
        self.assertEqual((howl.effect_label, howl.effect_chance), ("raise_self_atk", 100))
        self.assertEqual((swords.effect_label, swords.effect_chance), ("raise_self_atk_sharply", 100))

    def test_compound_and_omniboost_labels(self) -> None:
        calm = _move_effect("calmmind", topBoosts={"spa": 1, "spd": 1}, target="self")
        self.assertEqual(calm.effect_label, "raise_self_spaspd")
        omni = _move_effect(
            "ancientpower",
            category="Special",
            secondaries=[{"chance": 10, "selfBoosts": {"atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1}}],
        )
        self.assertEqual((omni.effect_label, omni.effect_chance), ("raise_self_all", 10))
        superpower = _move_effect("superpower", category="Physical", selfBoosts={"atk": -1, "def": -1})
        self.assertEqual(superpower.effect_label, "lower_self_atkdef")

    def test_target_direction_foe_vs_self(self) -> None:
        # Screech drops the foe's Defense by 2 ("halving" it); Overheat drops the user's SpA by 2.
        screech = _move_effect("screech", topBoosts={"def": -2}, target="normal")
        overheat = _move_effect("overheat", category="Special", selfBoosts={"spa": -2})
        self.assertEqual(screech.effect_label, "lower_foe_def_sharply")
        self.assertEqual(overheat.effect_label, "lower_self_spa_sharply")

    def test_primary_status_and_volatile_effects(self) -> None:
        twave = _move_effect("thunderwave", topStatus="par", target="normal")
        self.assertEqual((twave.effect_label, twave.effect_chance), ("par", 100))
        seed = _move_effect("leechseed", topVolatile="leechseed", target="normal")
        self.assertEqual(seed.effect_label, "leechseed")

    def test_secondary_chance_preserved(self) -> None:
        icebeam = _move_effect("icebeam", category="Special", secondaries=[{"chance": 10, "status": "frz"}])
        self.assertEqual((icebeam.effect_label, icebeam.effect_chance), ("frz", 10))

    def test_curse_static_label_suppressed_resolved_by_user_type(self) -> None:
        # Curse is type-dependent, so its static label is empty; resolve_move_effect picks the
        # real label from the user's types at encode time (stable within a battle).
        from pokezero.dex import resolve_move_effect

        curse = _move_effect("curse", topVolatile="curse", target="normal")
        self.assertEqual((curse.effect_label, curse.effect_chance, curse.self_hp_cost), ("", 0, 0.0))
        # Ghost user: lays a foe curse for 50% of its own HP.
        self.assertEqual(resolve_move_effect(curse, ("Ghost",)), ("curse", 100, 0.5))
        self.assertEqual(resolve_move_effect(curse, ("Ghost", "Poison")), ("curse", 100, 0.5))
        # Non-Ghost user: +Atk/+Def/-Spe self setup, no HP cost.
        self.assertEqual(resolve_move_effect(curse, ("Normal",)), ("curse_setup", 100, 0.0))
        self.assertEqual(resolve_move_effect(curse, ()), ("curse_setup", 100, 0.0))

    def test_hp_variable_base_power(self) -> None:
        from pokezero.dex import resolve_move_base_power

        reversal = _move_effect("reversal", category="Physical")  # static base power 0
        self.assertEqual(resolve_move_base_power(reversal, 0.01), 200)  # ~1/48 HP
        self.assertEqual(resolve_move_base_power(reversal, 0.05), 150)
        self.assertEqual(resolve_move_base_power(reversal, 0.5), 40)
        self.assertEqual(resolve_move_base_power(reversal, 1.0), 20)
        self.assertEqual(resolve_move_base_power(reversal, None), 0)  # unknown HP -> static
        # Gen 3 buckets floor(48*HP/maxHP); values just above a boundary must floor down, not up.
        self.assertEqual(resolve_move_base_power(reversal, 3 / 96), 200)  # 48*=1.5 -> floor 1
        self.assertEqual(resolve_move_base_power(reversal, 9 / 96), 150)  # 48*=4.5 -> floor 4
        self.assertEqual(resolve_move_base_power(reversal, 19 / 96), 100)  # 48*=9.5 -> floor 9
        eruption = _move_effect("eruption", category="Special", basePower=150)
        self.assertEqual(resolve_move_base_power(eruption, 1.0), 150)  # direct scaling
        self.assertEqual(resolve_move_base_power(eruption, 0.5), 75)
        flamethrower = _move_effect("flamethrower", category="Special", basePower=95)
        self.assertEqual(resolve_move_base_power(flamethrower, 0.2), 95)  # normal move unaffected

    def test_resolve_move_effect_passes_through_non_dynamic_moves(self) -> None:
        from pokezero.dex import resolve_move_effect

        screech = _move_effect("screech", topBoosts={"def": -2}, target="normal")
        self.assertEqual(
            resolve_move_effect(screech, ("Ghost",)),
            ("lower_foe_def_sharply", 100, 0.0),  # user type is irrelevant for non-dynamic moves
        )

    def test_strongest_secondary_wins_over_primary(self) -> None:
        # A labeled secondary (even at 100%) is the move's notable effect and takes the slot over a
        # co-present primary; the highest-chance secondary wins among several.
        move = _move_effect(
            "mix",
            category="Special",
            topBoosts={"atk": 1},  # a (hypothetical) primary self-boost
            target="self",
            secondaries=[{"chance": 30, "status": "brn"}, {"chance": 100, "boosts": {"spe": -1}}],
        )
        self.assertEqual((move.effect_label, move.effect_chance), ("lower_foe_spe", 100))

    def test_self_hp_cost_from_selfdestruct_and_overrides(self) -> None:
        boom = _move_effect("explosion", category="Physical", selfdestruct="always")
        self.assertEqual(boom.self_hp_cost, 1.0)
        # Custom-onHit moves: effect/cost come from the override table.
        belly = _move_effect("bellydrum", target="self")
        self.assertEqual((belly.effect_label, belly.self_hp_cost), ("raise_self_atk_max", 0.5))
        sub = _move_effect("substitute", topVolatile="substitute", target="self")
        self.assertEqual((sub.effect_label, sub.self_hp_cost), ("substitute", 0.25))


class DexTest(unittest.TestCase):
    def test_showdown_dex_payload_normalizes_perfect_accuracy(self) -> None:
        dex = showdown_dex_from_payload(
            {
                "moves": {
                    "swift": {
                        "id": "swift",
                        "name": "Swift",
                        "type": "Normal",
                        "category": "Special",
                        "basePower": 60,
                        "accuracy": True,
                        "priority": 0,
                    }
                },
                "species": {},
                "typeChart": {},
            }
        )

        assert dex.move_info("swift") is not None
        self.assertEqual(dex.move_info("swift").accuracy, 100.0)

    def test_showdown_dex_cached_loads_once_per_root(self) -> None:
        first = showdown_dex_from_payload({"moves": {}, "species": {}, "typeChart": {}})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("pokezero.dex.load_showdown_dex", return_value=first) as load:
                first_result = load_showdown_dex_cached(root)
                second_result = load_showdown_dex_cached(root)

        self.assertIs(first_result, first)
        self.assertIs(second_result, first)
        self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
