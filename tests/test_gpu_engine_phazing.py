"""Direct Gen 3 drag-switch fixtures; not a full-game divergence result."""
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
class PhazingTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton,effects,entry,items,mechanics,phazing,sustain,transition,weather
        from gpu_engine.state import empty_state,ROOTED
        self.jax,self.j,self.tr,self.ph,self.empty,self.m=jax,j,transition,phazing,empty_state,mechanics
        self.rooted=ROOTED
        self.names=["_none","roar","whirlwind","splash","struggle","batonpass","protect"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost","steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        rows=[None]+[{"flags":{"protect":1,"bypasssub":1}} if name in {"roar","whirlwind"}
            else {"flags":{}} for name in self.names[1:]]
        meta={"moves":rows,"vocabulary":{"moves":self.names,"types":types,
            "abilities":["_none","suctioncups","soundproof","naturalcure","drizzle","intimidate","levitate"],
            "items":["_none","lumberry"],"species":["_none"]}}
        n=len(self.names)
        self.table=transition.MoveTable(j.array([0,0,0,0,50,0,0]),j.array([0,1,1,1,0,1,1]),
            j.zeros(n,bool),j.full(n,100),j.array([True,False,False,True,True,True,True]),
            j.array([0,-6,-6,0,0,0,3]),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(4),
            effects=effects.from_metadata(meta),entry=entry.from_metadata(meta),items=items.from_metadata(meta),
            weather=weather.from_metadata(meta),sustain=sustain.from_metadata(meta),
            baton=baton.from_metadata(meta),phazing=phazing.from_metadata(meta))

    def live(self,batch=1,own="roar",foe="splash"):
        j=self.j;s=self.empty(batch,seed=71)
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),level=j.full_like(s.level,73),
            pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),done=j.zeros_like(s.done),
            moves=j.full_like(s.moves,3).at[:,0,0,0].set(self.names.index(own)).at[:,1,0,0].set(self.names.index(foe)))

    def play(self,s,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),self.table)

    def only_first_bench(self,s,side=1):
        return s._replace(hp=s.hp.at[:,side,2:].set(0))

    def test_uniform_bench_choice_for_all_possible_counts_and_nonzero_active(self):
        j=self.j;roll=j.arange(60,dtype=j.int32)
        for count in range(1,6):
            hp=j.zeros((60,6),j.int32).at[:,0:count+1].set(100)
            targets,exists=self.ph.choose_bench(hp,j.zeros(60,j.int32),roll)
            self.assertTrue(bool(j.all(exists)))
            self.assertEqual(j.bincount(targets,length=6).tolist(),[0]+[60//count]*count+[0]*(5-count))
        hp=j.zeros((60,6),j.int32).at[:,j.array([0,2,3,5])].set(100)
        target,exists=self.ph.choose_bench(hp,j.full(60,3),roll)
        self.assertEqual(j.bincount(target,length=6).tolist(),[20,0,20,0,0,20])
        _,exists=self.ph.choose_bench(j.zeros((60,6),j.int32),j.zeros(60,j.int32),roll)
        self.assertFalse(bool(j.any(exists)))

    def test_roar_and_whirlwind_bypass_substitute_clear_outgoing_and_spend_pp(self):
        for move in ("roar","whirlwind"):
            s=self.only_first_bench(self.live(own=move))
            s=s._replace(boosts=s.boosts.at[0,1,0,0].set(2),substitute_hp=s.substitute_hp.at[0,1,0].set(100),
                choice_move=s.choice_move.at[0,1,0].set(3),volatile=s.volatile.at[0,1,0].set(4))
            r=self.play(s).state
            self.assertEqual((int(r.active[0,1]),int(r.pp[0,0,0,0]),int(r.turn[0])),(1,9,1))
            self.assertEqual((int(r.boosts[0,1,0,0]),int(r.substitute_hp[0,1,0]),int(r.choice_move[0,1,0]),int(r.volatile[0,1,0])),(0,0,0,0))
            self.assertEqual((int(r.hp[0,1,0]),int(r.hp[0,1,1]),int(r.active_turns[0,1])),(1000,1000,0))

    def test_first_phazer_cancels_displaced_queued_action_not_incoming_moves(self):
        s=self.only_first_bench(self.live(foe="whirlwind"));r=self.play(s).state
        self.assertEqual((r.active.tolist(),r.pp[0,:,0,0].tolist()),([[0,1]],[9,10]))
        self.assertFalse(bool(r.pending_residual[0]))

    def test_protect_accuracy_suction_soundproof_root_and_no_bench(self):
        for ability,move,expected in ((1,"roar",0),(1,"whirlwind",0),(2,"roar",0),(2,"whirlwind",1)):
            s=self.only_first_bench(self.live(own=move));s=s._replace(ability=s.ability.at[0,1,0].set(ability))
            self.assertEqual(int(self.play(s).state.active[0,1]),expected)
        s=self.only_first_bench(self.live(foe="protect"))
        self.assertEqual(int(self.play(s).state.active[0,1]),0)
        s=self.only_first_bench(self.live());s=s._replace(volatile=s.volatile.at[0,1,0].set(self.rooted))
        self.assertEqual(int(self.play(s).state.active[0,1]),0)
        self.assertFalse(bool(self.tr.action_mask(s,self.table)[0,1,4]))
        s=self.live();s=s._replace(hp=s.hp.at[0,1,1:].set(0));r=self.play(s).state
        self.assertEqual((int(r.active[0,1]),int(r.pp[0,0,0,0])),(0,9))
        # A zero-accuracy authored table isolates miss handling without seeds.
        table=self.table._replace(accuracy=self.table.accuracy.at[1].set(0))
        self.assertEqual(int(self.tr.step(self.live(),self.j.array([[0,0]]),table).state.active[0,1]),0)

    def test_natural_cure_toxic_counter_and_recipient_lum_are_resolved_at_entry(self):
        for ability,status in ((3,0),(0,self.m.TOX)):
            s=self.only_first_bench(self.live());s=s._replace(status=s.status.at[0,1,0].set(self.m.TOX),
                status_counter=s.status_counter.at[0,1,0].set(3),ability=s.ability.at[0,1,0].set(ability),
                item=s.item.at[0,1,1].set(1))
            s=s._replace(status=s.status.at[0,1,1].set(self.m.PSN))
            r=self.play(s).state
            self.assertEqual((int(r.status[0,1,0]),int(r.status_counter[0,1,0])),(status,0))
            self.assertEqual((int(r.status[0,1,1]),int(r.item[0,1,1]),int(r.hp[0,1,1])),(0,0,1000))

    def test_hazards_weather_and_intimidate_run_before_residuals(self):
        s=self.only_first_bench(self.live());s=s._replace(spikes=s.spikes.at[0,1].set(3),
            ability=s.ability.at[0,1,1].set(4),weather=self.j.array([2]),weather_counter=self.j.array([4]))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,1,1]),int(r.weather[0]),int(r.weather_counter[0])),(750,1,0))
        s=self.only_first_bench(self.live());s=s._replace(ability=s.ability.at[0,1,1].set(5))
        self.assertEqual(int(self.play(s).state.boosts[0,0,0,0]),-1)
        s=self.only_first_bench(self.live());s=s._replace(spikes=s.spikes.at[0,1].set(3),ability=s.ability.at[0,1,1].set(6))
        self.assertEqual(int(self.play(s).state.hp[0,1,1]),1000)

    def test_entry_ko_pauses_wish_and_weather_then_replacement_finishes_turn(self):
        s=self.live();s=s._replace(hp=s.hp.at[0,0,0].set(500).at[0,1,1].set(1).at[0,1,2:].set(0),
            spikes=s.spikes.at[0,1].set(3),wish_counter=self.j.array([[1,0]]),weather=self.j.array([2]),weather_counter=self.j.array([4]))
        # Exactly one living bench isolates ordering from bench-choice RNG.
        paused=self.play(s).state
        self.assertEqual((int(paused.hp[0,1,1]),int(paused.hp[0,0,0]),int(paused.wish_counter[0,0]),int(paused.weather_counter[0])),(0,500,1,4))
        self.assertTrue(bool(paused.pending_residual[0]))
        # The original living outgoing slot is a legal recovery entrant.
        last=self.play(paused,[[0,4]]).state
        self.assertEqual((int(last.active[0,1]),int(last.hp[0,1,0]),int(last.hp[0,0,0]),int(last.weather_counter[0]),int(last.turn[0])),(0,750,1000,3,1))
        self.assertTrue(bool(self.j.array_equal(last.keys,paused.keys)))

    def test_phazing_after_baton_uses_saved_roll_and_original_action_once(self):
        s=self.live(own="batonpass",foe="roar");paused=self.play(s).state
        expected,_=self.ph.choose_bench(paused.hp[:,0],self.j.array([1]),paused.phazing_draws[:,1])
        r=self.play(paused,[[4,3]]).state
        self.assertEqual((int(r.active[0,0]),int(r.pp[0,1,0,0]),int(r.turn[0])),(int(expected[0]),9,1))
        self.assertFalse(bool(r.pending_residual[0]));self.assertFalse(bool(self.j.any(r.phazing_draws)))
        self.assertTrue(bool(self.j.array_equal(r.keys,paused.keys)))

    def test_jit_batch_prefix_invalid_and_terminal_identity(self):
        j=self.j;s=self.live(3);actions=j.zeros((3,2),j.int32)
        eager=self.tr.step(s,actions,self.table);compiled=self.jax.jit(self.tr.step)(s,actions,self.table)
        one=self.tr.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for a,b,c in zip(eager.state,compiled.state,one.state):
            self.assertTrue(bool(j.array_equal(a,b)));self.assertTrue(bool(j.array_equal(a[:1],c)))
        for frozen,actions in ((s,j.full((3,2),99,j.int32)),(s._replace(done=j.ones(3,bool)),actions)):
            result=self.tr.step(frozen,actions,self.table)
            for a,b in zip(frozen,result.state):self.assertTrue(bool(j.array_equal(a,b)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned source oracle")
    def test_pinned_gen3_direct_drag_and_hazard_request_order(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_phazing_oracle.cjs")),str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);d=json.loads(p.stdout)
        normal=dict(active="ditto",oldStatus="",oldBoost=0,oldSub=False,newHP=1000,request="move",turn=2)
        blocked={**normal,"active":"mew","oldBoost":2,"oldSub":True}
        for name in ("normal","natural","soundWind"):self.assertEqual(d[name],normal)
        for name in ("suction","soundRoar","rooted","protect"):self.assertEqual(d[name],blocked)
        self.assertEqual(d["cancelled"],dict(active=["mew","ditto"],oldPP=32,request="move"))
        self.assertEqual(d["hazards"],dict(before=dict(request="switch",oldHP=1000,newHP=0,sourceHP=500,wish=1,weather=4),
            after=dict(request="move",sourceHP=1000,wish=False,weather=3,active="chansey",hp=750)))


if __name__=="__main__":unittest.main()
