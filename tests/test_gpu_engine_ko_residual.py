"""KO-before-residual request phases; not complete turn/game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get("NODE") or shutil.which("node")
SHOWDOWN=os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class KOResidualTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import damage_moves,entry,items,mechanics,sustain,transition,weather
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.tr,self.empty=jax,j,mechanics,transition,empty_state
        names=["_none","seismictoss","splash","struggle"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost","steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta={"vocabulary":{"moves":names,"types":types,"abilities":["_none","drizzle","speedboost"],"items":["_none","leftovers"],"species":["_none"]}}
        self.table=transition.MoveTable(j.array([0,0,0,50]),j.array([0,2,1,0]),j.ones(4,bool),
            j.full(4,100),j.ones(4,bool),j.zeros(4,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3),
            entry=entry.from_metadata(meta),items=items.from_metadata(meta),weather=weather.from_metadata(meta),
            sustain=sustain.from_metadata(meta),damage_moves=damage_moves.from_metadata(meta))

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=71)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),level=j.full_like(s.level,73),
            hp=j.full_like(s.hp,1000).at[:,0,0].set(500).at[:,1,0].set(1),max_hp=j.full_like(s.hp,1000),
            moves=j.full_like(s.moves,2).at[:,0,0,0].set(1),pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),
            done=j.zeros_like(s.done),wish_counter=s.wish_counter.at[:,0].set(1))

    def play(self,s,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),self.table)

    def test_wish_waits_for_replacement_and_original_turn_accounting_is_once(self):
        j=self.j;s=self.live();opening=self.play(s).state
        self.assertEqual((int(opening.hp[0,0,0]),int(opening.wish_counter[0,0]),bool(opening.pending_residual[0])),(500,1,True))
        self.assertEqual((int(opening.turn[0]),opening.active_turns[0].tolist(),opening.pp[0,:,0,0].tolist()),(1,[1,1],[9,10]))
        self.assertTrue(bool(j.any(opening.residual_draws)))
        result=self.play(opening,[[0,4]]).state
        self.assertEqual((int(result.hp[0,0,0]),int(result.wish_counter[0,0]),bool(result.pending_residual[0])),(1000,0,False))
        self.assertEqual((int(result.turn[0]),result.active_turns[0].tolist(),result.pp[0,:,0,0].tolist()),(1,[1,0],[9,10]))
        self.assertTrue(bool(j.array_equal(result.keys,opening.keys)))
        self.assertFalse(bool(j.any(result.residual_draws)))

    def test_wish_on_fainted_slot_heals_recipient_not_dead_outgoing(self):
        s=self.live();s=s._replace(wish_counter=self.j.array([[0,1]]),hp=s.hp.at[0,1,1].set(600))
        opening=self.play(s).state
        self.assertEqual((int(opening.hp[0,1,0]),int(opening.wish_counter[0,1])),(0,1))
        result=self.play(opening,[[0,4]]).state
        self.assertEqual((int(result.hp[0,1,0]),int(result.hp[0,1,1]),int(result.wish_counter[0,1])),(0,1000,0))

    def test_repeated_hazard_ko_defers_clock_and_wish_until_last_living_entry(self):
        j=self.j;s=self.live();s=s._replace(hp=s.hp.at[0,1,1].set(1).at[0,1,2].set(600),
            spikes=s.spikes.at[0,1].set(3),ability=s.ability.at[0,1,1].set(1),weather=j.array([2]),weather_counter=j.array([4]))
        opening=self.play(s).state;middle=self.play(opening,[[0,4]]).state
        self.assertEqual((int(middle.hp[0,1,1]),bool(middle.pending_residual[0]),int(middle.replacement_count[0])),(0,True,1))
        self.assertEqual((int(middle.hp[0,0,0]),int(middle.wish_counter[0,0]),int(middle.weather[0]),int(middle.weather_counter[0])),(500,1,2,4))
        self.assertTrue(bool(j.array_equal(middle.keys,opening.keys)));self.assertTrue(bool(j.array_equal(middle.residual_draws,opening.residual_draws)))
        result=self.play(middle,[[0,5]]).state
        self.assertEqual((int(result.hp[0,1,2]),int(result.hp[0,0,0]),int(result.weather_counter[0])),(350,1000,3))
        self.assertEqual((int(result.turn[0]),bool(result.pending_residual[0]),int(result.replacement_count[0])),(1,False,0))
        self.assertTrue(bool(j.array_equal(result.keys,opening.keys)))

    def test_terminal_hazard_exhaustion_clears_pending_without_residual(self):
        s=self.live();s=s._replace(hp=s.hp.at[0,1,1:].set(0).at[0,1,1].set(1),spikes=s.spikes.at[0,1].set(3))
        opening=self.play(s).state;result=self.play(opening,[[0,4]]).state
        self.assertEqual((bool(result.done[0]),int(result.winner[0]),bool(result.pending_residual[0])),(True,0,False))
        self.assertEqual((int(result.hp[0,0,0]),int(result.wish_counter[0,0])),(500,1))
        self.assertFalse(bool(self.j.any(result.residual_draws)))

    def test_double_recoil_ko_requires_both_entries_before_residual(self):
        s=self.live();s=s._replace(hp=s.hp.at[0,0,0].set(1).at[0,0,1].set(100).at[0,1,1].set(200),
            pp=s.pp.at[0,0,0].set(0),wish_counter=self.j.array([[1,1]]))
        opening=self.play(s).state
        self.assertEqual((opening.hp[0,:,0].tolist(),opening.forced_switch.tolist(),bool(opening.pending_residual[0])),([0,0],[[True,True]],True))
        result=self.play(opening,[[4,4]]).state
        self.assertEqual((result.hp[0,:,1].tolist(),result.wish_counter.tolist(),int(result.turn[0])),([600,700],[[0,0]],1))

    def test_residual_ko_replacement_is_not_another_residual_turn(self):
        s=self.live();s=s._replace(hp=s.hp.at[0,0,0].set(1).at[0,1,0].set(1000).at[0,0,1].set(700),
            status=s.status.at[0,:,0].set(self.m.PSN),wish_counter=self.j.zeros_like(s.wish_counter))
        opening=self.play(s).state
        self.assertEqual((int(opening.hp[0,0,0]),int(opening.hp[0,1,0]),bool(opening.pending_residual[0])),(0,802,False))
        result=self.play(opening,[[4,0]]).state
        self.assertEqual((int(result.hp[0,0,1]),int(result.hp[0,1,0]),int(result.turn[0])),(700,802,1))

    def test_residual_speed_is_snapshotted_after_final_replacement(self):
        s=self.live();s=s._replace(hp=s.hp.at[0,0,0].set(1).at[0,1,1:].set(0).at[0,1,1].set(1),
            item=s.item.at[0,0,0].set(1),status=s.status.at[0,1,1].set(self.m.PSN),
            stats=s.stats.at[0,1,1,5].set(400),wish_counter=self.j.zeros_like(s.wish_counter))
        opening=self.play(s).state;result=self.play(opening,[[0,4]]).state
        self.assertEqual((int(result.hp[0,0,0]),int(result.hp[0,1,1]),bool(result.done[0]),int(result.winner[0])),(1,0,True,0))

    def test_jit_batch_prefix_and_invalid_pending_identity(self):
        j=self.j;s=self.live(4)._replace(done=self.live(4).done.at[3].set(True))
        opening=self.play(s).state;actions=j.array([[0,4],[0,4],[0,9],[0,4]])
        a=self.jax.jit(self.tr.step)(opening,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(opening,actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],opening),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(opening,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_request_and_residual_order(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_ko_residual_oracle.cjs")),str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        data=json.loads(p.stdout)
        before=dict(request="switch",hp=500,wish=1,weatherDuration=0,turn=1)
        after=dict(request="move",hp=1000,wish=0,weatherDuration=0,turn=2)
        self.assertEqual(data["source"],dict(before=before,middle=after,after=after,recipient=1000))
        self.assertEqual(data["hazards"],dict(before={**before,"weatherDuration":4},middle={**before,"weatherDuration":4},
            after={**after,"weatherDuration":3},recipient=350))
        self.assertEqual(data["recipient"],dict(before=dict(request="switch",wish=1,oldHP=0),after=dict(hp=1000,wish=False)))
        self.assertEqual(data["suppressor"],dict(
            before=dict(request="switch",hp=1000,wish=0,weatherDuration=4,turn=1),
            after=dict(request="move",hp=938,wish=0,weatherDuration=3,turn=2)))


if __name__=="__main__":unittest.main()
