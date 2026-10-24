"""Authored major-status/self-effect fixtures; no full-game fidelity claim."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated spike JAX")
class EffectTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import mechanics as m
        from gpu_engine import effects,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.effects,self.t,self.empty = jax,j,m,effects,transition,empty_state
        self.names = ["_none","tackle","splash","struggle","toxic","thunderwave",
            "stunspore","willowisp","sleeppowder","rest","recover","calmmind",
            "swordsdance","bulkup","haze","dragondance","tailglow"]
        self.types = ["_none","normal","fighting","flying","poison","ground","rock",
            "bug","ghost","steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        self.meta = {"vocabulary":{"moves":self.names,"types":self.types}}
        n = len(self.names)
        self.table = transition.MoveTable(j.array([0,40,0,50]+[0]*(n-4)),
            j.array([0,1,1,0,4,13,12,10,12,14,1,14,1,2,15,16,7]),
            j.array([False,True,False,True]+[False]*(n-4)),
            j.full((n,),100),j.ones((n,),bool),j.zeros((n,),j.int32),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3),
            effects.from_metadata(self.meta))

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,101),max_hp=j.full_like(s.max_hp,101),
            pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,2),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def run_move(self,s,move):
        s = s._replace(moves=s.moves.at[:,0,0,0].set(self.names.index(move)))
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),self.table).state

    def test_toxic_applies_then_uses_first_stage_residual(self):
        r = self.run_move(self.live(),"toxic")
        self.assertEqual(r.status[0,1,0].item(),self.m.TOX)
        self.assertEqual(r.status_counter[0,1,0].item(),1)
        self.assertEqual(r.hp[0,1,0].item(),95)
        self.assertEqual(r.pp[0,0,0,0].item(),9)

    def test_status_never_overwrites_existing_status_or_penetrates_substitute(self):
        s = self.live()
        s = s._replace(status=s.status.at[0,1,0].set(self.m.BRN),
                       status_counter=s.status_counter.at[0,1,0].set(7))
        r = self.run_move(s,"toxic")
        self.assertEqual(r.status[0,1,0].item(),self.m.BRN)
        self.assertEqual(r.status_counter[0,1,0].item(),7)
        s = self.live()._replace(substitute_hp=self.live().substitute_hp.at[0,1,0].set(25))
        self.assertEqual(self.run_move(s,"toxic").status[0,1,0].item(),self.m.NONE)

    def test_gen3_type_immunities_and_electric_can_be_paralyzed(self):
        for move,typ,expected in (("toxic",4,self.m.NONE),("toxic",9,self.m.NONE),
            ("willowisp",10,self.m.NONE),("thunderwave",5,self.m.NONE),
            ("stunspore",5,self.m.PAR),("thunderwave",13,self.m.PAR)):
            with self.subTest(move=move,type=typ):
                s = self.live()._replace(types=self.live().types.at[0,1,0,0].set(typ))
                self.assertEqual(self.run_move(s,move).status[0,1,0].item(),expected)

    def test_sleep_counter_is_two_to_five_then_decrements_before_slower_move(self):
        r = self.run_move(self.live(64),"sleeppowder")
        self.assertTrue(bool(self.j.all(r.status[:,1,0] == self.m.SLP)))
        self.assertTrue(bool(self.j.all((r.status_counter[:,1,0] >= 1) & (r.status_counter[:,1,0] <= 4))))
        self.assertTrue(bool(self.j.all(r.pp[:,1,0,0] == 10)))

    def test_rest_heals_replaces_burn_and_sets_exact_counter(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,0,0].set(17),status=s.status.at[0,0,0].set(self.m.BRN))
        r = self.run_move(s,"rest")
        self.assertEqual((r.hp[0,0,0].item(),r.status[0,0,0].item(),
                          r.status_counter[0,0,0].item()),(101,self.m.SLP,3))
        self.assertEqual(r.pp[0,0,0,0].item(),9)

    def test_rest_fails_at_full_hp_and_cannot_act_while_asleep(self):
        s = self.live()._replace(status=self.live().status.at[0,0,0].set(self.m.PSN))
        r = self.run_move(s,"rest")
        self.assertEqual((r.status[0,0,0].item(),r.hp[0,0,0].item()),(self.m.PSN,89))
        s = self.live()
        s = s._replace(hp=s.hp.at[0,0,0].set(17),status=s.status.at[0,0,0].set(self.m.SLP),
                       status_counter=s.status_counter.at[0,0,0].set(3))
        r = self.run_move(s,"rest")
        self.assertEqual((r.hp[0,0,0].item(),r.status_counter[0,0,0].item(),
                          r.pp[0,0,0,0].item()),(17,2,10))

    def test_recover_floors_half_max_hp_heals_actor_and_caps(self):
        for start,expected in ((17,67),(90,101)):
            s = self.live()._replace(hp=self.live().hp.at[0,0,0].set(start))
            r = self.run_move(s,"recover")
            self.assertEqual(r.hp[0,:,0].tolist(),[expected,101])

    def test_boosts_apply_to_actor_and_clip_at_six(self):
        r = self.run_move(self.live(),"calmmind")
        self.assertEqual(r.boosts[0,0,0].tolist(),[0,0,1,1,0,0,0])
        s = self.live()._replace(boosts=self.live().boosts.at[0,0,0,0].set(5))
        r = self.run_move(s,"swordsdance")
        self.assertEqual(r.boosts[0,0,0,0].item(),6)
        self.assertEqual(self.run_move(self.live(),"bulkup").boosts[0,0,0].tolist(),[1,1,0,0,0,0,0])
        self.assertTrue(bool(self.j.all(r.boosts[0,1,0] == 0)))

    def test_dragon_dance_applies_and_clamps_attack_speed_on_either_seat(self):
        # Authored regression motivated by fixed-panel slot20, not a replay or
        # a new independent fidelity measurement. Other boosts/party stay put.
        for actor in (0,1):
            for attack,speed in ((0,0),(5,6),(6,5),(6,6)):
                with self.subTest(actor=actor,attack=attack,speed=speed):
                    state = self.live()._replace(moves=self.live().moves.at[0,actor,0,0].set(
                        self.names.index("dragondance")), boosts=self.live().boosts
                        .at[0,actor,0,0].set(attack).at[0,actor,0,4].set(speed))
                    result = self.t.step(state,self.j.zeros((1,2),self.j.int32),self.table)
                    expected = state.boosts.at[0,actor,0,0].set(min(6,attack+1))
                    expected = expected.at[0,actor,0,4].set(min(6,speed+1))
                    self.assertTrue(bool(self.j.array_equal(result.state.boosts,expected)))
                    self.assertTrue(bool(self.j.array_equal(result.state.hp,state.hp)))
                    self.assertEqual(int(result.state.pp[0,actor,0,0]),9)
                    self.assertFalse(bool(self.j.any(result.invalid_actions)))

    def test_gen3_tail_glow_adds_two_special_attack_stages_in_both_seats(self):
        # Motivated by saved slot1057, not a replay or fidelity credit. Gen3
        # inherits the Gen4 +2 rule, not modern Tail Glow's +3 stages.
        starts=[-6,-1,0,5,6]*2
        actors=[0]*5+[1]*5
        state=self.live(10)
        for row,(actor,stage) in enumerate(zip(actors,starts)):
            state=state._replace(moves=state.moves.at[row,actor,0,0].set(self.names.index('tailglow')),
                boosts=state.boosts.at[row,actor,0,2].set(stage).at[row,actor,0,0].set(1))
        actions=self.j.zeros((10,2),self.j.int32)
        eager=self.t.step(state,actions,self.table)
        compiled=self.jax.jit(self.t.step)(state,actions,self.table)
        self.jax.block_until_ready(compiled)
        expected=state.boosts
        for row,(actor,stage) in enumerate(zip(actors,starts)):
            expected=expected.at[row,actor,0,2].set(min(6,stage+2))
            self.assertEqual(int(compiled.state.pp[row,actor,0,0]),9)
        self.assertTrue(bool(self.j.array_equal(compiled.state.boosts,expected)))
        self.assertTrue(bool(self.j.array_equal(compiled.state.hp,state.hp)))
        self.assertFalse(bool(self.j.any(compiled.invalid_actions)))
        for a,b in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_haze_clears_both_actives_through_sub_but_not_bench(self):
        s = self.live()
        s = s._replace(boosts=s.boosts.at[0,0,0].set(6).at[0,1,0].set(-6).at[0,0,1].set(3),
                       substitute_hp=s.substitute_hp.at[0,1,0].set(25))
        r = self.run_move(s,"haze")
        self.assertTrue(bool(self.j.all(r.boosts[0,:,0] == 0)))
        self.assertTrue(bool(self.j.all(r.boosts[0,0,1] == 3)))

    def test_effect_dispatch_matches_jit(self):
        s = self.live()
        s = s._replace(moves=s.moves.at[0,0,0,0].set(self.names.index("calmmind")))
        actions = self.j.zeros((1,2),self.j.int32)
        eager = self.t.step(s,actions,self.table)
        compiled = self.jax.jit(self.t.step)(s,actions,self.table)
        compiled.state.hp.block_until_ready()
        for a,b in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(self.j.array_equal(a,b)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Gen 3 catalog")
    def test_effect_mappings_match_actual_pinned_gen3_catalog(self):
        names = ["toxic","poisonpowder","thunderwave","stunspore","willowisp","sleeppowder",
            "hypnosis","spore","sing","recover","softboiled","milkdrink","slackoff","rest","haze",
            "swordsdance","calmmind","bulkup","agility","amnesia","irondefense","barrier",
            "meditate","howl","doubleteam","dragondance","tailglow"]
        helper = Path(__file__).with_name("gpu_move_effect_oracle.cjs")
        rows = json.loads(subprocess.run([NODE,str(helper),SHOWDOWN],input=json.dumps(names),
            text=True,capture_output=True,check=True).stdout)
        effects = self.effects.from_metadata({"vocabulary":{"moves":names,"types":self.types}})
        statuses = {"":0,"par":self.m.PAR,"brn":self.m.BRN,"psn":self.m.PSN,"tox":self.m.TOX,"slp":self.m.SLP}
        stats = ["atk","def","spa","spd","spe","accuracy","evasion"]
        for i,row in enumerate(rows):
            self.assertEqual(int(effects.status[i]),statuses[row["status"]])
            self.assertEqual(effects.self_boosts[i].tolist(),[row["boosts"].get(stat,0) for stat in stats])
            self.assertEqual(bool(effects.heal_half[i]),row["heal"] == [1,2])
            if bool(self.j.any(effects.self_boosts[i])) or bool(effects.heal_half[i]):
                self.assertEqual(row["target"],"self")


if __name__ == "__main__":
    unittest.main()
