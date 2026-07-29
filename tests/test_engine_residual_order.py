"""Regression pins for the gen3 residual ORDER patches (requires the patched wheel).

Resolved through `Dex.mod('gen3')` (`scripts/gen3_dex_resolve.py`'s rule), gen3
keeps the pre-gen5 residual numbering and everything here is ONE order class:

    abilities 10.3 -> items 10.4 -> Leech Seed 10.5 -> status damage 10.6

so within one Pokemon the ability heal, then EVERY item (Leftovers and the
threshold berries alike), then Leech Seed, then the status tick. The 5.x values in
`data/{items,abilities}.ts` are the gen5+ table and do not apply to gen3.

This file previously asserted the opposite for the berries -- "status damage
(9/10) -> threshold berries / Rain Dish (10+)" -- and said so was necessary
because "the Showdown-vs-engine differential cannot yet reach mid-battle HP states
in a one-turn fresh battle". That was a pin on the ENGINE'S OWN behaviour, and it
was backwards. A MULTI-turn fixture reaches the state fine:
`scripts/gen3_switch_differential.py::residualberrybeforestatus` chips a burned
Sitrus holder under half and reads the sim's answer directly --

    |-enditem|p2a: Aipom|Sitrus Berry|[eat]
    |-heal|p2a: Aipom|150/251 brn|[from] item: Sitrus Berry
    |-damage|p2a: Aipom|119/251 brn|[from] brn

-- so the timing is now re-derived from Showdown on every differential run, and
these pins only guard that the engine still agrees with it.

Note the SIDE of the block these tests exercise: they build a fresh state where
the berry holder is the FASTER Pokemon, so its whole 10.x set resolves first.
Cross-side interleaving is pinned separately, in
`rust/pokezero-search/tests/gen3_residual_speed_order.rs`.
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

    def test_sitrus_fires_before_status_damage_at_item_suborder(self) -> None:
        """Sitrus is 10.4, the status tick is 10.6 -> the berry heals FIRST.

        The threshold is therefore read against the PRE-tick HP: 82/165 is already
        at or under half, so the berry eats, and the poison tick lands on the
        healed total. Ground truth `residualberrybeforestatus`.
        """
        # 82/165 poisoned: Sitrus +41 -> 123, then poison -20 -> 103.
        side, instructions = self._end_of_turn(item="sitrusberry", hp=82, status="poison")
        self.assertEqual(side.pokemon[0].hp, 103)
        heal_pos = next(i for i, s in enumerate(instructions) if s == "Heal SideTwo: 41")
        damage_pos = next(i for i, s in enumerate(instructions) if s == "Damage SideTwo: 20")
        self.assertLess(heal_pos, damage_pos)

    def test_sitrus_does_not_fire_on_hp_the_status_tick_would_have_reached(self) -> None:
        """The other side of the same rule, and the reason it is observable.

        85/165 is ABOVE half, so at 10.4 the berry does not qualify; the 10.6 tick
        then drops it to 65, which is under half, but the berry's slot has already
        passed and it does not eat this turn. Under the old (backwards) ordering
        the same fixture ate the berry.
        """
        side, instructions = self._end_of_turn(item="sitrusberry", hp=85, status="poison")
        self.assertEqual(side.pokemon[0].hp, 65)
        self.assertNotIn("Heal SideTwo: 41", instructions)

    def test_pinch_berry_boost_fires_before_status_damage(self) -> None:
        """Liechi is 10.4 as well -- the boost lands ahead of the 10.6 tick."""
        # 41/165 poisoned: Liechi +1 Atk at 41 <= 41, then poison -20 -> 21.
        side, instructions = self._end_of_turn(item="liechiberry", hp=41, status="poison")
        self.assertEqual(side.pokemon[0].hp, 21)
        self.assertEqual(side.attack_boost, 1)
        boost_pos = next(i for i, s in enumerate(instructions) if "Boost SideTwo Attack" in s)
        damage_pos = next(i for i, s in enumerate(instructions) if s == "Damage SideTwo: 20")
        self.assertLess(boost_pos, damage_pos)

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

    def test_shed_skin_branches_at_showdown_rate_before_status_damage(self) -> None:
        pe = poke_engine
        dummy = pe.Pokemon(id="pikachu", level=1, hp=0)
        holder = pe.Pokemon(
            id="seviper", level=80, types=("poison", "typeless"), hp=100, maxhp=self.MAXHP,
            ability="shedskin", item="none", attack=100, defense=120, special_attack=140,
            special_defense=120, speed=150, status="poison",
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

        outcomes: dict[tuple[int, str], float] = {}
        for branch in pe.generate_instructions(state, "curse", "calmmind"):
            applied = state.apply_instructions(branch)
            key = (applied.side_two.pokemon[0].hp, str(applied.side_two.pokemon[0].status).upper())
            outcomes[key] = outcomes.get(key, 0.0) + float(branch.percentage)

        self.assertAlmostEqual(outcomes[(100, "NONE")], 33.0, places=4)
        self.assertAlmostEqual(outcomes[(80, "POISON")], 67.0, places=4)


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
