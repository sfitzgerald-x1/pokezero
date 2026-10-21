"""Selected Speed Boost timing and residual fixtures; no gameplay qualification."""
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
class SpeedBoostTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import entry,items,mechanics,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.tr,self.empty = jax,j,mechanics,transition,empty_state
        meta = {"vocabulary":{"moves":["_none","splash","struggle"],"abilities":["_none","speedboost"],
            "items":["_none","leftovers"],"species":["_none","mew"],
            "types":["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
                "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]}}
        self.table = transition.MoveTable(j.array([0,0,50]),j.array([0,1,0]),j.array([False,False,True]),
            j.full(3,100),j.ones(3,bool),j.zeros(3,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),
            j.array(2),items=items.from_metadata(meta),entry=entry.from_metadata(meta))

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        return s._replace(stats=j.full_like(s.stats,200),level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),pp=j.full_like(s.pp,10),
            moves=j.ones_like(s.moves),types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),self.table)

    def test_initial_entry_boosts_on_first_ordinary_turn_and_clamps_at_six(self):
        s = self.live()._replace(ability=self.live().ability.at[0,0,0].set(1),
            entry_pending=self.j.ones((1,2),bool))
        for turn in range(1,9):
            s = self.play(s).state
            self.assertEqual((int(s.boosts[0,0,0,4]),int(s.active_turns[0,0])),(min(6,turn),turn))
        self.assertEqual(s.boosts[0,0,0,[0,1,2,3,5,6]].tolist(),[0]*6)

    def test_ordinary_switch_entry_skips_bonus_then_next_turn_applies(self):
        s = self.live()._replace(ability=self.live().ability.at[0,0,1].set(1))
        switched = self.play(s,[[4,0]]).state
        self.assertEqual((int(switched.boosts[0,0,1,4]),int(switched.active_turns[0,0])),(0,0))
        after = self.play(switched).state
        self.assertEqual((int(after.boosts[0,0,1,4]),int(after.active_turns[0,0])),(1,1))
        away = self.play(after,[[4,0]]).state
        self.assertEqual((int(away.boosts[0,0,1,4]),int(away.active_turns[0,0])),(0,0))

    def test_forced_replacement_has_no_age_tick_or_residual_then_first_turn_boosts(self):
        s = self.live()._replace(ability=self.live().ability.at[0,0,1].set(1))
        s = s._replace(hp=s.hp.at[0,0,0].set(0),active_turns=self.j.array([[8,9]]))
        replaced = self.play(s,[[4,0]]).state
        self.assertEqual(replaced.active_turns.tolist(),[[0,9]])
        self.assertEqual((int(replaced.turn[0]),int(replaced.boosts[0,0,1,4])),(0,0))
        self.assertEqual(int(self.play(replaced).state.boosts[0,0,1,4]),1)

    def test_inability_to_move_does_not_stop_speed_boost(self):
        s = self.live()._replace(ability=self.live().ability.at[0,0,0].set(1))
        s = s._replace(status=s.status.at[0,0,0].set(self.m.SLP),status_counter=s.status_counter.at[0,0,0].set(10))
        r = self.play(s).state
        self.assertEqual((int(r.boosts[0,0,0,4]),int(r.pp[0,0,0,0])),(1,10))

    def test_fainted_or_bench_holder_does_not_boost(self):
        s = self.live()._replace(ability=self.live().ability.at[0,0,1].set(1))
        self.assertEqual(int(self.play(s).state.boosts[0,0,1,4]),0)
        s = s._replace(ability=s.ability.at[0,0,0].set(1),hp=s.hp.at[0,0,0].set(0))
        r = self.play(s,[[4,0]]).state
        self.assertEqual(int(r.boosts[0,0,0,4]),0)

    def test_speed_priority_precedes_suborder_and_party_ko_cancels_later_boost(self):
        j = self.j
        for speed,expected in ((300,0),(200,1)):
            s = self.live()._replace(hp=j.zeros_like(self.live().hp).at[0,0,0].set(1).at[0,1,0].set(100))
            s = s._replace(max_hp=j.full_like(s.hp,100),status=s.status.at[0,0,0].set(self.m.PSN),
                ability=s.ability.at[0,1,0].set(1),stats=s.stats.at[0,0,0,5].set(speed))
            r = self.play(s).state
            self.assertTrue(bool(r.done[0]))
            self.assertEqual(int(r.boosts[0,1,0,4]),expected)

    def test_speed_boost_does_not_resort_remaining_leftovers_and_status(self):
        j = self.j
        s = self.live()._replace(hp=j.zeros_like(self.live().hp).at[0,0,0].set(50).at[0,1,0].set(1))
        s = s._replace(max_hp=j.full_like(s.hp,100),ability=s.ability.at[0,1,0].set(1),
            item=s.item.at[0,0,0].set(1),status=s.status.at[0,1,0].set(self.m.PSN))
        r = self.play(s).state
        self.assertTrue(bool(r.done[0]))
        self.assertEqual(int(r.hp[0,0,0]),56)

    def test_jit_batch_prefix_and_invalid_terminal_state_age_identity(self):
        j = self.j
        s = self.live(4)._replace(ability=self.live(4).ability.at[:,0,0].set(1))
        s = s._replace(done=s.done.at[3].set(True),active_turns=j.full((4,2),7))
        actions = j.array([[0,0],[0,0],[9,0],[0,0]])
        compiled = self.jax.jit(self.tr.step)(s,actions,self.table); compiled.state.hp.block_until_ready()
        eager = self.tr.step(s,actions,self.table)
        for a,b in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(eager)):
            self.assertTrue(bool(j.array_equal(a,b)))
        smaller = self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for a,b in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(smaller)):
            self.assertTrue(bool(j.array_equal(a[:2],b)))
        for a,b in zip(s,compiled.state): self.assertTrue(bool(j.array_equal(a[2:],b[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_gen3_initial_switch_forced_and_residual_timing(self):
        result = subprocess.run([NODE,str(Path(__file__).with_name("gpu_speed_boost_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),{"stages":[0,1,0,1],"forced_stages":[0,1],
            "equal_speed_heal":56,"residual_order":[10,3]})


if __name__ == "__main__": unittest.main()
