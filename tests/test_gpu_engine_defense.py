"""Authored Substitute/Protect/Endure checks, not full-game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class DefenseTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,mechanics,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.e,self.m,self.t,self.empty = jax,j,effects,mechanics,transition,empty_state
        self.names = ["_none","tackle","splash","struggle","substitute","protect","endure","toxic","haze"]
        self.types = ["_none","normal","fighting","flying","poison","ground","rock",
            "bug","ghost","steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        rows = [None]+[{"flags":{"protect":1} if name in {"tackle","struggle","toxic"} else {}}
                       for name in self.names[1:]]
        meta = {"vocabulary":{"moves":self.names,"types":self.types},"moves":rows}
        self.table = transition.MoveTable(j.array([0,40,0,50,0,0,0,0,0]),
            j.array([0,1,1,0,1,1,1,4,15]),j.array([False,True,False,True]+[False]*5),
            j.full((9,),100),j.ones((9,),bool),j.array([0,0,0,0,0,3,4,0,0]),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3),effects.from_metadata(meta))

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,101),max_hp=j.full_like(s.hp,101),
            pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,2),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,own="splash",foe="splash",table=None):
        s = s._replace(moves=s.moves.at[:,0,0,0].set(self.names.index(own))
                                    .at[:,1,0,0].set(self.names.index(foe)))
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),table or self.table).state

    def test_substitute_costs_floor_quarter_and_cannot_replace_existing(self):
        r = self.play(self.live(),"substitute")
        self.assertEqual((int(r.hp[0,0,0]),int(r.substitute_hp[0,0,0]),int(r.pp[0,0,0,0])),(76,25,9))
        r = self.play(r,"substitute")
        self.assertEqual((int(r.hp[0,0,0]),int(r.substitute_hp[0,0,0]),int(r.pp[0,0,0,0])),(76,25,8))

    def test_substitute_fails_at_quarter_hp_or_shedinja_max_hp(self):
        for hp,max_hp in ((25,101),(25,100),(1,1)):
            s = self.live()
            s = s._replace(hp=s.hp.at[0,0,0].set(hp),max_hp=s.max_hp.at[0,0,0].set(max_hp))
            r = self.play(s,"substitute")
            self.assertEqual((int(r.hp[0,0,0]),int(r.substitute_hp[0,0,0])),(hp,0))
            self.assertEqual(int(r.pp[0,0,0,0]),9)
        s = self.live()._replace(hp=self.live().hp.at[0,0,0].set(26))
        self.assertEqual(int(self.play(s,"substitute").hp[0,0,0]),1)

    def test_substitute_absorbs_damage_without_spill_or_move_cancellation(self):
        table = self.table._replace(power=self.table.power.at[1].set(1000))
        s = self.live()._replace(substitute_hp=self.live().substitute_hp.at[0,1,0].set(25))
        r = self.play(s,"tackle","splash",table)
        self.assertEqual(r.hp[0,:,0].tolist(),[101,101])
        self.assertEqual(int(r.substitute_hp[0,1,0]),0)
        self.assertEqual(r.pp[0,:,0,0].tolist(),[9,9])

    def test_substitute_is_created_before_slower_attack_and_blocks_status(self):
        r = self.play(self.live(),"substitute","tackle")
        self.assertEqual(int(r.hp[0,0,0]),76)
        self.assertEqual(int(r.substitute_hp[0,0,0]),0)
        r = self.play(self.live(),"substitute","toxic")
        self.assertEqual((int(r.hp[0,0,0]),int(r.status[0,0,0]),int(r.substitute_hp[0,0,0])),(76,0,25))

    def test_struggle_recoil_uses_actual_substitute_damage(self):
        s = self.live()._replace(pp=self.live().pp.at[:,0].set(0),
            substitute_hp=self.live().substitute_hp.at[0,1,0].set(7))
        r = self.play(s,"tackle")
        self.assertEqual(r.hp[0,:,0].tolist(),[100,101])
        self.assertEqual(int(r.substitute_hp[0,1,0]),0)

    def test_substitute_survives_small_hits_and_is_unchanged_by_miss(self):
        s = self.live()._replace(substitute_hp=self.live().substitute_hp.at[0,1,0].set(25))
        table = self.table._replace(power=self.table.power.at[1].set(1))
        r = self.play(s,"tackle",table=table)
        self.assertEqual(int(r.hp[0,1,0]),101)
        self.assertTrue(0 < int(r.substitute_hp[0,1,0]) < 25)
        table = table._replace(accuracy=table.accuracy.at[1].set(0),
                               always_hit=table.always_hit.at[1].set(False))
        r = self.play(s,"tackle",table=table)
        self.assertEqual((int(r.hp[0,1,0]),int(r.substitute_hp[0,1,0])),(101,25))

    def test_actual_move_flags_control_protect_and_substitute_bypass(self):
        s = self.live()._replace(substitute_hp=self.live().substitute_hp.at[0,1,0].set(25))
        effects = self.table.effects._replace(bypass_sub=self.table.effects.bypass_sub.at[1].set(True))
        r = self.play(s,"tackle",table=self.table._replace(effects=effects))
        self.assertLess(int(r.hp[0,1,0]),101)
        self.assertEqual(int(r.substitute_hp[0,1,0]),25)
        effects = self.table.effects._replace(protectable=self.table.effects.protectable.at[1].set(False))
        r = self.play(self.live(),"protect","tackle",self.table._replace(effects=effects))
        self.assertLess(int(r.hp[0,0,0]),101)

    def test_protect_blocks_attack_and_toxic_but_not_residual_damage(self):
        for foe in ("tackle","toxic"):
            r = self.play(self.live(),"protect",foe)
            self.assertEqual(r.hp[0,:,0].tolist(),[101,101])
            self.assertEqual(int(r.status[0,0,0]),0)
            self.assertEqual(r.pp[0,:,0,0].tolist(),[9,9])
            self.assertEqual(int(r.protect_counter[0,0,0]),2)
        s = self.live()._replace(status=self.live().status.at[0,0,0].set(self.m.PSN))
        self.assertEqual(int(self.play(s,"protect","tackle").hp[0,0,0]),89)

    def test_protect_does_not_block_foe_self_effect_or_haze(self):
        s = self.live()._replace(boosts=self.live().boosts.at[0,0,0].set(4))
        r = self.play(s,"protect","haze")
        self.assertTrue(bool(self.j.all(r.boosts[0,0,0] == 0)))
        r = self.play(self.live(),"protect","substitute")
        self.assertEqual((int(r.hp[0,1,0]),int(r.substitute_hp[0,1,0])),(76,25))

    def test_stall_denominators_have_exact_gen3_eighth_grid(self):
        for counter,successes in ((0,8),(2,4),(4,2),(8,1)):
            self.assertEqual(int(self.j.sum(self.e.stall_success(counter,self.j.arange(8)))),successes)

    def test_stall_counter_advances_caps_and_failed_uses_expire(self):
        s = self.live(128)
        for count,next_count in ((2,4),(4,8),(8,8)):
            s = s._replace(protect_counter=s.protect_counter.at[:,0,0].set(count))
            r = self.play(s,"protect","tackle")
            success = r.hp[:,0,0] == 101
            self.assertTrue(bool(self.j.any(success) & self.j.any(~success)))
            self.assertTrue(bool(self.j.all(r.protect_counter[:,0,0] == self.j.where(success,next_count,0))))
        r = self.play(s,"splash","tackle")
        self.assertTrue(bool(self.j.all(r.protect_counter[:,0,0] == 0)))

    def test_protect_expires_after_turn_and_after_inability_to_move(self):
        r = self.play(self.live(),"protect","tackle")
        self.assertLess(int(self.play(r,"splash","tackle").hp[0,0,0]),101)
        s = self.live()
        s = s._replace(status=s.status.at[0,0,0].set(self.m.SLP),
            status_counter=s.status_counter.at[0,0,0].set(3),protect_counter=s.protect_counter.at[0,0,0].set(8))
        r = self.play(s,"protect","tackle")
        self.assertEqual((int(r.protect_counter[0,0,0]),int(r.pp[0,0,0,0])),(0,10))

    def test_protect_fails_as_last_queued_move_or_after_foe_switch(self):
        r = self.play(self.live(),"protect","protect")
        self.assertEqual(r.protect_counter[0,:,0].tolist(),[2,0])
        s = self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(5))
        r = self.t.step(s,self.j.array([[0,4]]),self.table).state
        self.assertEqual(int(r.protect_counter[0,0,0]),0)

    def test_endure_leaves_one_hp_but_does_not_stop_status_or_residuals(self):
        table = self.table._replace(power=self.table.power.at[1].set(1000))
        self.assertEqual(int(self.play(self.live(),"endure","tackle",table).hp[0,0,0]),1)
        s = self.live()._replace(hp=self.live().hp.at[0,0,0].set(2),
            status=self.live().status.at[0,0,0].set(self.m.BRN))
        self.assertEqual(int(self.play(s,"endure","tackle",table).hp[0,0,0]),0)
        r = self.play(self.live(),"endure","toxic")
        self.assertEqual(int(r.status[0,0,0]),self.m.TOX)

    def test_forced_replacement_preserves_other_stall_counter(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,1,0].set(0),protect_counter=s.protect_counter.at[0,0,0].set(4))
        r = self.t.step(s,self.j.array([[0,4]]),self.table).state
        self.assertEqual(int(r.protect_counter[0,0,0]),4)

    def test_substitute_protect_dispatch_matches_jit(self):
        s = self.live(2)
        s = s._replace(moves=s.moves.at[:,0,0,0].set(self.j.array([4,5])).at[:,1,0,0].set(1))
        actions = self.j.zeros((2,2),self.j.int32)
        eager = self.t.step(s,actions,self.table)
        compiled = self.jax.jit(self.t.step)(s,actions,self.table)
        self.jax.block_until_ready(compiled)
        for a,b in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(self.j.array_equal(a,b)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Gen 3 defense source")
    def test_defense_rules_match_pinned_source_handlers(self):
        helper = Path(__file__).with_name("gpu_defense_oracle.cjs")
        result = json.loads(subprocess.run([NODE,str(helper),SHOWDOWN],text=True,
            capture_output=True,check=True).stdout)
        self.assertEqual(result["stall"],{"start":2,"after_restart":[4,8,8],"successes":[4,2,1]})
        self.assertEqual(result["substitute"],{"cost":25,"sub_hp":25,"remaining_sub_hp":0,"damage":25})
        self.assertEqual(result["priorities"],{"protect":3,"detect":3,"endure":4})


    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Gen3 stall source")
    def test_stall_failures_pay_pp_and_called_moves_only_charge_caller(self):
        helper=Path(__file__).with_name('gpu_stall_pp_oracle.cjs')
        result=json.loads(subprocess.run([NODE,str(helper),SHOWDOWN],text=True,
            capture_output=True,check=True).stdout)
        self.assertEqual(result['cases'],28)
        for row in result['rows']:
            if row.get('called_protect'):
                self.assertEqual(row['pp_after'],[row['pp_before'][0],row['pp_before'][1]-1])
            else:self.assertEqual(row['pp_after'],row['pp_before']-1)


if __name__ == "__main__":
    unittest.main()
