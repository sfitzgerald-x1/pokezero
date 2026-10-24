"""Authored native entry-age controls; not source-root or whole-game credit."""
from dataclasses import replace
import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec('poke_engine'),'requires actual native Gen3 binding')
class NativeSpeedBoostEntryTests(unittest.TestCase):
    def state(self,seat,*,initial=False,paralyzed=False,cap=0,forced=False,baton=False):
        import poke_engine as engine
        from pokezero.poke_engine_adapter import (
            BattleSpec,MoveSpec,PokemonSpec,SideSpec,build_poke_engine_state,
        )
        mon=PokemonSpec(id='mew',level=100,types=('normal',),hp=1000,maxhp=1000,
            attack=200,defense=200,special_attack=200,special_defense=200,speed=200,
            ability='none',moves=(MoveSpec('splash',8),))
        entrant=replace(mon,id='ninjask',ability='speedboost')
        sides=[]
        for s in range(2):
            active=replace(mon,ability='speedboost' if initial and s==seat else 'none',
                status='paralyze' if paralyzed and s==seat else 'none',
                hp=0 if forced and s==seat else 1000)
            sides.append(SideSpec(pokemon=(active,entrant,mon,mon,mon,mon),
                boosts={'speed':cap if s==seat else 0},force_switch=(forced or baton) and s==seat,
                baton_passing=baton and s==seat))
        return build_poke_engine_state(BattleSpec(*sides),module=engine)

    def successors(self,state,moves):
        import poke_engine as engine
        before=state.to_string();out=[]
        branches=engine.generate_instructions(state,*moves)
        self.assertAlmostEqual(math.fsum(float(b.percentage)/100 for b in branches),1,places=6)
        for branch in branches:
            mass=float(branch.percentage)/100
            self.assertTrue(math.isfinite(mass) and mass>=0)
            if not mass:continue
            after=state.apply_instructions(branch)
            self.assertEqual(after.reverse_instructions(branch).to_string(),before)
            out.append(after)
        self.assertEqual(state.to_string(),before,'native generation mutated raw parent')
        self.assertTrue(out)
        return out

    def test_ordinary_switch_skips_entry_bonus_then_next_turn_boosts_both_seats(self):
        for seat in range(2):
            moves=['splash','splash'];moves[seat]='ninjask'
            side=('side_one','side_two')[seat]
            for after in self.successors(self.state(seat),moves):
                self.assertEqual(int(getattr(after,side).speed_boost),0)
                for next_state in self.successors(after,['splash','splash']):
                    self.assertEqual(int(getattr(next_state,side).speed_boost),1)

    def test_initial_holder_boosts_despite_paralysis_and_caps_both_seats(self):
        for seat in range(2):
            for cap in (0,5,6):
                for after in self.successors(self.state(seat,initial=True,paralyzed=True,cap=cap),['splash','splash']):
                    self.assertEqual(int(getattr(after,('side_one','side_two')[seat]).speed_boost),min(6,cap+1))

    def test_other_seat_switch_does_not_suppress_established_holder(self):
        for seat in range(2):
            moves=['splash','splash'];moves[1-seat]='ninjask'
            for after in self.successors(self.state(seat,initial=True),moves):
                self.assertEqual(int(getattr(after,('side_one','side_two')[seat]).speed_boost),1)
                self.assertEqual(int(getattr(after,('side_one','side_two')[1-seat]).speed_boost),0)

    def test_forced_replacement_has_no_entry_or_residual_bonus_both_seats(self):
        for seat in range(2):
            moves=['none','none'];moves[seat]='ninjask'
            for after in self.successors(self.state(seat,forced=True),moves):
                self.assertEqual(int(getattr(after,('side_one','side_two')[seat]).speed_boost),0)
            # Completed Baton carries stages, but entry must not add another.
            for after in self.successors(self.state(seat,baton=True,cap=3),moves):
                self.assertEqual(int(getattr(after,('side_one','side_two')[seat]).speed_boost),3)


if __name__=='__main__':unittest.main()
