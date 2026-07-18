"""Regression pins for the gen3 residual-order patch (requires the patched wheel).

The Showdown-vs-engine differential cannot yet reach mid-battle HP states in a
one-turn fresh battle, so the berry-threshold timing is pinned directly against
the engine: residual order must be Leftovers/Shed Skin (5) -> Leech Seed (8) ->
status damage (9/10) -> threshold berries / Rain Dish (10+).
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


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class ResidualOrderTests(unittest.TestCase):
    MAXHP = 165

    def _end_of_turn(self, *, item: str, hp: int, status: str, ability: str = "naturalcure"):
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        holder = pe.Pokemon(
            id="starmie", level=57, types=("water", "psychic"), hp=hp, maxhp=self.MAXHP,
            ability=ability, item=item, attack=100, defense=120, special_attack=140,
            special_defense=120, speed=150, status=status,
            moves=[pe.Move(id="calmmind", pp=16)],
        )
        other = pe.Pokemon(
            id="swampert", level=80, types=("water", "ground"), hp=291, maxhp=291,
            ability="torrent", item="none", attack=200, defense=190, special_attack=180,
            special_defense=190, speed=140, moves=[pe.Move(id="curse", pp=16)],
        )
        state = pe.State(
            side_one=pe.Side(active_index="0", pokemon=[other] + [dummy] * 5),
            side_two=pe.Side(active_index="0", pokemon=[holder] + [dummy] * 5),
            weather="none", terrain="none", trick_room=False,
        )
        branch = pe.generate_instructions(state, "curse", "calmmind")[0]
        applied = state.apply_instructions(branch)
        return applied.side_two, [str(i) for i in branch.instruction_list]

    def test_sitrus_fires_after_status_damage_crosses_threshold(self) -> None:
        # 85/165 poisoned: poison -20 -> 65 <= 82 -> Sitrus +41 -> 106.
        side, instructions = self._end_of_turn(item="sitrusberry", hp=85, status="poison")
        self.assertEqual(side.pokemon[0].hp, 106)
        damage_pos = next(i for i, s in enumerate(instructions) if s == "Damage SideTwo: 20")
        heal_pos = next(i for i, s in enumerate(instructions) if s == "Heal SideTwo: 41")
        self.assertLess(damage_pos, heal_pos)

    def test_pinch_berry_boost_fires_after_status_damage(self) -> None:
        # 45/165 poisoned: poison -20 -> 25 <= 41 -> Liechi +1 Atk.
        side, instructions = self._end_of_turn(item="liechiberry", hp=45, status="poison")
        self.assertEqual(side.pokemon[0].hp, 25)
        self.assertEqual(side.attack_boost, 1)
        damage_pos = next(i for i, s in enumerate(instructions) if s == "Damage SideTwo: 20")
        boost_pos = next(i for i, s in enumerate(instructions) if "Boost SideTwo Attack" in s)
        self.assertLess(damage_pos, boost_pos)

    def test_leftovers_heals_before_status_damage(self) -> None:
        # Full-HP toxic holder: Leftovers no-ops at full, toxic nets -165/16.
        side, instructions = self._end_of_turn(item="leftovers", hp=165, status="toxic")
        self.assertEqual(side.pokemon[0].hp, 155)
        self.assertNotIn("Heal SideTwo: 10", instructions)
        # Burned at 100: Leftovers +10 BEFORE burn -20.
        side, instructions = self._end_of_turn(item="leftovers", hp=100, status="burn")
        self.assertEqual(side.pokemon[0].hp, 90)
        heal_pos = next(i for i, s in enumerate(instructions) if s == "Heal SideTwo: 10")
        damage_pos = next(i for i, s in enumerate(instructions) if s == "Damage SideTwo: 20")
        self.assertLess(heal_pos, damage_pos)

    def test_shed_skin_cures_before_status_damage(self) -> None:
        # Shed Skin (order 5.3) removes the status before it deals damage; the
        # engine does not branch on its 1/3 chance, so the cure is deterministic.
        side, instructions = self._end_of_turn(
            item="none", hp=100, status="poison", ability="shedskin"
        )
        self.assertEqual(side.pokemon[0].hp, 100)  # no poison damage after cure
        self.assertEqual(str(side.pokemon[0].status).upper(), "NONE")


@unittest.skipIf(poke_engine is None, "poke-engine wheel not installed")
class EncoreLockPinTests(unittest.TestCase):
    """Pin the engine semantics the encore construction relies on."""

    def test_encore_volatile_with_last_used_move_locks_the_side(self) -> None:
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)

        def mk(species, moves, speed):
            return pe.Pokemon(
                id=species, level=80, types=("normal", "typeless"), hp=300, maxhp=300,
                ability="innerfocus", item="leftovers", attack=180, defense=180,
                special_attack=180, special_defense=180, speed=speed,
                moves=[pe.Move(id=m, pp=16) for m in moves],
            )

        locked = pe.Side(
            active_index="0",
            pokemon=[mk("snorlax", ["bodyslam", "growl", "curse", "rest"], 90)] + [dummy] * 5,
            volatile_statuses={"ENCORE"},
            last_used_move="move:1",
            volatile_status_durations=pe.VolatileStatusDurations(encore=1),
        )
        free = pe.Side(
            active_index="0",
            pokemon=[mk("wobbuffet", ["counter", "encore"], 100)] + [dummy] * 5,
        )
        state = pe.State(side_one=locked, side_two=free, weather="none", terrain="none", trick_room=False)
        result = pe.monte_carlo_tree_search(state, 30, threads=1)
        choices = {entry.move_choice for entry in result.side_one}
        self.assertEqual(choices, {"growl"})


if __name__ == "__main__":
    unittest.main()