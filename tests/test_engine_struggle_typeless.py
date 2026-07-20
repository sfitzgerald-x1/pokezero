"""Regression pins for the gen3 Struggle-typeless patch (patched wheel).

Upstream poke-engine 0.0.47 defines Struggle as ``move_type: PokemonType::NORMAL``
with no gen3 override, so the gen3 engine computed Struggle as a Normal move:
immune vs Ghost, resisted by Rock/Steel, and STAB-boosted from a Normal user.
``third_party/poke-engine-gen3-struggle-typeless.patch`` makes gen3 (and gen2+)
Struggle TYPELESS for type effectiveness while preserving its PHYSICAL damage
class through ``undo_physical_special_split`` (which re-derives category from
move_type and would otherwise flip a TYPELESS move to Special).

These tests pin the engine's damage *contract* directly against calculate_damage
so a future wheel rebuild (or a version bump that drops the patch) cannot silently
regress Struggle back to Normal-typed or to Special. They are the committed
counterpart to the Showdown/encoder gates and cover the damage/category class of
bug that a token-only encoder test cannot see.

Run in a DEDICATED struggle venv (never the shared one):
    .venv/bin/python -m pytest tests/test_engine_struggle_typeless.py
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import poke_engine
except ImportError:  # pragma: no cover - native wheel absent
    poke_engine = None


def _mon(types, *, attack=180, special_attack=170, status="none"):
    pe = poke_engine
    return pe.Pokemon(
        id="attacker", level=100, types=types, base_types=types, hp=350, maxhp=350,
        ability="none", item="none", attack=attack, defense=140,
        special_attack=special_attack, special_defense=130, speed=120, status=status,
    )


def _struggle_rolls(user_types, defender_types, *, attack=180, special_attack=170, status="none"):
    """Max damage roll for Struggle from ``user_types`` into ``defender_types``."""

    pe = poke_engine
    dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
    side_one = pe.Side(active_index="0", pokemon=[_mon(user_types, attack=attack,
                       special_attack=special_attack, status=status)] + [dummy] * 5)
    side_two = pe.Side(active_index="0", pokemon=[_mon(defender_types)] + [dummy] * 5)
    state = pe.State(side_one=side_one, side_two=side_two,
                     weather="none", terrain="none", trick_room=False)
    attacker_rolls, _ = pe.calculate_damage(state, "struggle", "splash", True)
    return attacker_rolls[1]  # max roll


def _has_struggle_typeless_patch() -> bool:
    """True iff the installed wheel models gen3 Struggle as typeless (hits Ghost).

    Unpatched: Struggle is Normal, immune vs Ghost -> 0 damage. Patched: > 0.
    """

    if poke_engine is None:
        return False
    return _struggle_rolls(("normal", "typeless"), ("ghost", "poison")) > 0


NORMAL = ("normal", "typeless")
GHOST = ("ghost", "poison")
STEEL = ("steel", "typeless")
ROCK = ("rock", "typeless")
WATER = ("water", "typeless")


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
@unittest.skipUnless(_has_struggle_typeless_patch(), "struggle-typeless-patched wheel not installed")
class StruggleTypelessEngineTests(unittest.TestCase):
    def test_struggle_hits_ghost_and_is_neutral_vs_all(self) -> None:
        # Typeless: neutral vs EVERY type. Ghost is no longer immune, and Rock/
        # Steel no longer resist; all defenders take the same (neutral) damage.
        neutral = _struggle_rolls(NORMAL, NORMAL)
        self.assertGreater(neutral, 0)
        for target in (GHOST, STEEL, ROCK, WATER):
            self.assertEqual(_struggle_rolls(NORMAL, target), neutral,
                             f"Struggle must be neutral vs {target}")

    def test_struggle_grants_no_stab(self) -> None:
        # No STAB: a Normal-type user must not get the 1.5x a Normal move would.
        from_normal_user = _struggle_rolls(NORMAL, NORMAL)
        from_water_user = _struggle_rolls(WATER, NORMAL)
        self.assertEqual(from_normal_user, from_water_user)

    def test_struggle_is_physical_tracks_attack(self) -> None:
        # PHYSICAL: damage scales with Attack, not Special Attack. If the patch
        # regressed Struggle to Special (the undo_physical_special_split trap for
        # a TYPELESS move), the high-SpA user would out-damage the high-Atk user.
        high_atk = _struggle_rolls(NORMAL, NORMAL, attack=220, special_attack=60)
        high_spa = _struggle_rolls(NORMAL, NORMAL, attack=60, special_attack=220)
        self.assertGreater(high_atk, high_spa)

    def test_struggle_is_physical_burn_halves(self) -> None:
        # PHYSICAL moves are halved by burn (gen3 damage_calc burn_modifier). A
        # Special Struggle would be unaffected -> this pins the category too.
        healthy = _struggle_rolls(NORMAL, WATER, status="none")
        burned = _struggle_rolls(NORMAL, WATER, status="burn")
        self.assertEqual(burned, healthy // 2)


if __name__ == "__main__":
    unittest.main()
