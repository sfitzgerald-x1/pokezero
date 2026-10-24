"""Authored Pursuit/switch boundaries, not complete game qualification."""
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
class PursuitTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import entry,items,mechanics,pressure,pursuit,sustain,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.p,self.tr,self.empty=jax,j,mechanics,pursuit,transition,empty_state
        self.names=["_none","pursuit","splash","struggle"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta={"vocabulary":{"moves":self.names,"types":types,"abilities":["_none","pressure","drizzle"],
            "items":["_none","choiceband","lightball"],"species":["_none","pikachu"]}}
        num=j.ones((18,18),j.int32).at[17,14].set(2)
        self.table=transition.MoveTable(j.array([0,40,0,50]),j.array([0,17,1,0]),
            j.array([False,False,False,True]),j.full(4,100),j.ones(4,bool),j.zeros(4,j.int32),
            num,j.ones_like(num),j.array(3),items=items.from_metadata(meta),
            pressure=pressure.from_metadata(meta),entry=entry.from_metadata(meta),
            sustain=sustain.from_metadata(meta),pursuit=pursuit.from_metadata(meta))

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=71)
        stats=j.full_like(s.stats,200).at[...,1].set(77).at[...,3].set(333).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,73),hp=j.full_like(s.hp,1000),
            max_hp=j.full_like(s.hp,1000),pp=j.full_like(s.pp,10),
            moves=j.full_like(s.moves,2).at[:,0,0,0].set(1),done=j.zeros_like(s.done),
            types=s.types.at[:,0,:,0].set(17).at[:,1,0,0].set(14).at[:,1,1,0].set(11))

    def play(self,s,actions=None,table=None):
        j=self.j
        return self.tr.step(s,j.tile(j.array([[0,4]]),(s.hp.shape[0],1)) if actions is None else actions,
            self.table if table is None else table)

    def test_controlled_special_damage_matches_native_boundaries(self):
        j=self.j;s=self.live(6)
        s=s._replace(item=s.item.at[:,0,0].set(j.array([0,1,2,0,0,0])),species=s.species.at[:,0,0].set(1),
            boosts=s.boosts.at[:,0,0,2].set(j.array([0,0,0,-2,-2,2])).at[:,1,0,3].set(2),
            hp=s.hp.at[:,1,0].set(10000),max_hp=s.max_hp.at[:,1,0].set(10000))
        chance=j.zeros((6,12),j.int32).at[:,4].set(j.array([1,1,1,1,0,0])).at[:,5].set(15)
        r,used,fainted=self.p.intercept(s,0,j.zeros(6,j.int32),j.ones(6,j.int32),chance,self.table,
            eligible=True,contact_roll=j.zeros(6,j.int32))
        self.assertEqual((10000-r.hp[:,1,0]).tolist(),[128,128,252,66,504,1002])
        self.assertTrue(bool(j.all(used)));self.assertFalse(bool(j.any(fainted)))

    def test_switch_hit_is_outgoing_always_hit_and_consumed_once(self):
        j=self.j;s=self.live(8)
        t=self.table._replace(accuracy=self.table.accuracy.at[1].set(0),always_hit=self.table.always_hit.at[1].set(False))
        r=self.play(s,table=t)
        self.assertTrue(bool(j.all(r.state.hp[:,1,0]<1000)))
        self.assertEqual(r.state.hp[:,1,1].tolist(),[1000]*8)
        self.assertEqual(r.state.pp[:,0,0,0].tolist(),[9]*8)
        self.assertEqual(r.placeholder_moves.tolist(),[1]*8)

    def test_outgoing_sub_pressure_and_successful_choice_lock(self):
        s=self.live();s=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(13),
            ability=s.ability.at[0,1,0].set(1),item=s.item.at[0,0,0].set(1))
        r=self.play(s).state
        self.assertEqual((r.hp[0,1].tolist(),int(r.pp[0,0,0,0]),int(r.choice_move[0,0,0])),([1000]*6,8,1))
        self.assertEqual(int(r.substitute_hp[0,1,0]),0)  # cleared by switch, not spilled

    def test_interception_bypasses_paralysis_confusion_flinch_without_counter_tick(self):
        j=self.j;s=self.live(16);base=self.play(s).state
        s=s._replace(status=s.status.at[:,0,0].set(self.m.PAR),
            confusion_counter=s.confusion_counter.at[:,0,0].set(3),volatile=s.volatile.at[:,0,0].set(1))
        r=self.play(s).state
        self.assertEqual(r.hp[:,1,0].tolist(),base.hp[:,1,0].tolist())
        self.assertEqual(r.hp[:,0,0].tolist(),[1000]*16)
        self.assertEqual(r.confusion_counter[:,0,0].tolist(),[3]*16)
        self.assertEqual(r.pp[:,0,0,0].tolist(),[9]*16)

    def test_sleep_and_freeze_registration_do_not_intercept_even_if_ordinary_gate_clears(self):
        j=self.j;s=self.live(16)
        for status in (self.m.SLP,self.m.FRZ):
            state=s._replace(status=s.status.at[:,0,0].set(status),status_counter=s.status_counter.at[:,0,0].set(1))
            r=self.play(state).state
            self.assertEqual(r.hp[:,1,0].tolist(),[1000]*16)
            if status==self.m.SLP:
                self.assertTrue(bool(j.all(r.hp[:,1,1]<1000)))
                self.assertEqual(r.status_counter[:,0,0].tolist(),[0]*16)
            else:self.assertTrue(bool(j.any(r.hp[:,1,1]==1000)))

    def test_pursuit_ko_keeps_chosen_entry_hazards_weather_wish_and_single_turn_age(self):
        s=self.live();s=s._replace(hp=s.hp.at[0,1,0].set(1).at[0,1,1].set(600),
            ability=s.ability.at[0,1,0].set(1).at[0,1,1].set(2),spikes=s.spikes.at[0,1].set(3),
            wish_counter=s.wish_counter.at[0,1].set(1),item=s.item.at[0,0,0].set(1))
        r=self.play(s)
        self.assertEqual((int(r.state.hp[0,1,0]),int(r.state.active[0,1]),int(r.state.hp[0,1,1])),(0,1,850))
        self.assertEqual((int(r.state.weather[0]),int(r.state.wish_counter[0,1]),int(r.state.turn[0])),(1,0,1))
        self.assertEqual((r.state.active_turns[0].tolist(),int(r.state.pp[0,0,0,0]),int(r.placeholder_moves[0])),([1,0],8,1))

    def test_forced_pseudoturn_and_exhausted_struggle_do_not_intercept(self):
        s=self.live();forced=s._replace(hp=s.hp.at[0,1,0].set(0),wish_counter=s.wish_counter.at[0,1].set(1))
        r=self.play(forced).state
        self.assertEqual((int(r.hp[0,1,1]),int(r.pp[0,0,0,0]),int(r.turn[0]),int(r.wish_counter[0,1])),(1000,10,0,1))
        exhausted=s._replace(pp=s.pp.at[0,0,0].set(0))
        r=self.play(exhausted).state
        self.assertEqual(int(r.hp[0,1,0]),1000)
        self.assertLess(int(r.hp[0,1,1]),1000);self.assertEqual(int(r.pp[0,0,0,0]),0)

    def test_non_switch_path_unchanged_and_second_seat_intercepts(self):
        j=self.j;s=self.live(8);actions=j.zeros((8,2),j.int32)
        a,b=self.play(s,actions),self.play(s,actions,self.table._replace(pursuit=None))
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x,y)))
        mirrored=self.jax.tree.map(lambda x:x[:,::-1] if x.ndim>1 and x.shape[1]==2 else x,s)
        # keys are battle-level [B,2], not seat state; retain the original stream.
        mirrored=mirrored._replace(keys=s.keys)
        r=self.play(mirrored,j.tile(j.array([[4,0]]),(8,1))).state
        self.assertTrue(bool(j.all(r.hp[:,0,0]<1000)));self.assertEqual(r.hp[:,0,1].tolist(),[1000]*8)

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4)._replace(done=self.live(4).done.at[3].set(True))
        actions=j.array([[0,4],[0,4],[9,4],[0,4]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(s,actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_interception_events(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_pursuit_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        expected=dict(oldHP=748,incomingChosen=True,incomingHP=750,weather="raindance",usedPP=2,confusion=0,lock="",wish=False)
        self.assertEqual(json.loads(p.stdout),dict(ordinary=expected,
            sub={**expected,"oldHP":1000,"lock":"pursuit"},
            bypass={**expected,"confusion":3,"lock":"pursuit"},
            sleep={**expected,"oldHP":1000,"incomingHP":686,"usedPP":1},
            ko={**expected,"oldHP":0,"incomingHP":850,"lock":"pursuit"},damage=[128,128,252,66,504,1002]))


if __name__=="__main__":unittest.main()
