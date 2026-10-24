"""Actual native status/PP law, anchored by the independent Showdown fixture.

These finite authored cases do not qualify whole futures, real-team roots,
coverage or strength. Both ordinary and enumerated damage profiles must retain
the same status/PP masses; no native flag is enabled by importing this module.
"""
from collections import defaultdict
import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec("poke_engine"), "requires actual native Gen3 binding")
class NativeFreezeGateTests(unittest.TestCase):
    def law(self, *, reverse=False, frozen=False, lum=False):
        import poke_engine as engine
        from pokezero.poke_engine_adapter import (
            BattleSpec, MoveSpec, PokemonSpec, SideSpec, build_poke_engine_state,
        )
        sides = []
        for seat in range(2):
            actor = seat == int(reverse)
            mon = PokemonSpec(id="mew", level=100, types=("water",), hp=1000, maxhp=1000,
                attack=200, defense=200, special_attack=200, special_defense=200,
                speed=301 if actor else 199, ability="none",
                status="freeze" if frozen and not actor else "none",
                item="lumberry" if lum and not actor else "none",
                moves=(MoveSpec("icebeam" if actor else "splash", 8),))
            sides.append(SideSpec(pokemon=(mon,)*6))
        state = build_poke_engine_state(BattleSpec(*sides), module=engine)
        before = state.to_string()
        moves = ("splash", "icebeam") if reverse else ("icebeam", "splash")
        branches = engine.generate_instructions(state, *moves)
        law = defaultdict(float)
        for branch in branches:
            mass = float(branch.percentage) / 100
            self.assertTrue(math.isfinite(mass) and mass >= 0)
            if mass == 0:
                continue
            after = state.apply_instructions(branch)
            target = getattr(after, "side_one" if reverse else "side_two").pokemon[0]
            law[(str(target.status).lower(), int(target.moves[0].pp),
                 str(target.item).lower())] += mass
        self.assertEqual(state.to_string(), before, "native generation mutated parent")
        self.assertAlmostEqual(math.fsum(law.values()), 1., places=5)
        return law

    def test_new_freeze_has_eight_percent_stopped_and_two_percent_thawed_mass_both_seats(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                law = self.law(reverse=reverse)
                self.assertEqual(set(law), {("none", 7, "none"), ("freeze", 8, "none")})
                self.assertAlmostEqual(law[("freeze", 8, "none")], .08, places=6)
                self.assertAlmostEqual(law[("none", 7, "none")], .92, places=6)

    def test_existing_freeze_keeps_ordinary_twenty_percent_thaw_both_seats(self):
        for reverse in (False, True):
            law = self.law(reverse=reverse, frozen=True)
            self.assertEqual(set(law), {("none", 7, "none"), ("freeze", 8, "none")})
            self.assertAlmostEqual(law[("freeze", 8, "none")], .8, places=6)
            self.assertAlmostEqual(law[("none", 7, "none")], .2, places=6)

    def test_new_freeze_cured_by_lum_does_not_cancel_queued_move_both_seats(self):
        for reverse in (False, True):
            law = self.law(reverse=reverse, lum=True)
            self.assertEqual(set(law), {("none", 7, "none"), ("none", 7, "lumberry")})
            self.assertAlmostEqual(law[("none", 7, "none")], .1, places=6)
            self.assertAlmostEqual(law[("none", 7, "lumberry")], .9, places=6)


if __name__ == "__main__":
    unittest.main()
