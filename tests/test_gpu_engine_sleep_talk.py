"""Gen3 Sleep Talk integration, not general called-move qualification."""
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
class SleepTalkTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton,effects,entry,items,mechanics,phazing,pressure,pursuit,sleep_talk,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.talk,self.tr,self.empty=jax,j,mechanics,sleep_talk,transition,empty_state
        self.names=["_none","sleeptalk","tackle","solarbeam","focuspunch","splash","struggle",
            "batonpass","roar","protect","pursuit","rest","fly"]
        self.types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        self.abilities=["_none","earlybird","pressure","naturalcure"]
        meta={"vocabulary":{"moves":self.names,"types":self.types,"abilities":self.abilities,
            "species":["_none"],"items":["_none","choiceband","lumberry"]},
            "moves":[None]+[{"id":name,"source_type":"Normal","category":"Physical",
                "flags":{"protect":1,"nosleeptalk":name in {"sleeptalk","focuspunch","struggle"},
                    "charge":name in {"solarbeam","fly"}}} for name in self.names[1:]]}
        self.profile=sleep_talk.from_metadata(meta);n=len(self.names)
        power=j.array([50 if name in {"tackle","solarbeam","focuspunch","struggle","pursuit","fly"} else 0 for name in self.names])
        priority=j.zeros(n,j.int32).at[self.id("roar")].set(-6).at[self.id("protect")].set(4)
        self.table=transition.MoveTable(power,j.ones(n,j.int32),j.ones(n,bool),j.full(n,100),
            j.ones(n,bool),priority,j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(self.id("struggle")),
            effects=effects.from_metadata(meta),entry=entry.from_metadata(meta),items=items.from_metadata(meta),
            pressure=pressure.from_metadata(meta),pursuit=pursuit.from_metadata(meta),baton=baton.from_metadata(meta),
            phazing=phazing.from_metadata(meta),sleep_talk=self.profile)

    def id(self,name):return self.names.index(name)

    def live(self,batch=1,*,time=5):
        j=self.j;s=self.empty(batch,seed=137)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),level=j.full_like(s.level,100),
            moves=j.full_like(s.moves,self.id("splash")).at[:,0,0,:].set(
                j.array([self.id(x) for x in ("sleeptalk","tackle","solarbeam","focuspunch")])),
            pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),
            status=s.status.at[:,0,0].set(self.m.SLP),status_counter=s.status_counter.at[:,0,0].set(time),
            done=j.zeros_like(s.done))

    def play(self,s,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else actions,self.table)

    def call_only(self,s,name):
        return s._replace(moves=s.moves.at[:,0,0,:].set(
            self.j.array([self.id(x) for x in ("sleeptalk",name,"solarbeam","focuspunch")])))

    def test_uniform_slot_choice_includes_zero_pp_then_fails_without_reroll(self):
        j=self.j
        for count in (1,2,3):
            moves=j.tile(j.array([self.id("sleeptalk")]+[self.id(x) for x in ("tackle","rest","protect")]),(60,1))
            profile=self.profile._replace(callable=self.profile.callable.at[self.id("rest")].set(count>=2)
                .at[self.id("protect")].set(count>=3))
            pp=j.full((60,4),7).at[:,1].set(0)
            move,ok=self.talk.choose(profile,moves,pp,j.arange(60))
            for chosen in ("tackle","rest","protect")[:count]:
                self.assertEqual(int((move==self.id(chosen)).sum()),60//count)
            self.assertEqual(int((~ok).sum()),60//count)

    def test_normal_call_spends_only_sleep_talk_pp_and_sleeps_once(self):
        for ability in (0,2):
            s=self.live();s=s._replace(pp=s.pp.at[0,0,0,1].set(7),ability=s.ability.at[0,1,0].set(ability),
                disabled=s.disabled.at[0,0,0,1].set(True))
            r=self.play(s).state
            self.assertLess(int(r.hp[0,1,0]),1000)
            self.assertEqual(r.pp[0,0,0,:2].tolist(),[9,7])
            self.assertEqual((int(r.status_counter[0,0,0]),int(r.sleep_skipped[0,0,0])),(4,1))

    def test_selected_zero_pp_and_no_callable_slots_still_charge_caller(self):
        s=self.live();s=s._replace(pp=s.pp.at[0,0,0,1].set(0))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,1,0]),int(r.pp[0,0,0,0]),int(r.pp[0,0,0,1])),(1000,9,0))
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0,1].set(self.id("fly")))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,1,0]),int(r.pp[0,0,0,0])),(1000,9))

    def test_waking_and_awake_uses_fail_after_pp_and_early_bird_ticks_twice(self):
        for time in (0,1):
            s=self.live(time=time)
            if time==0:s=s._replace(status=s.status.at[0,0,0].set(0))
            r=self.play(s).state
            self.assertEqual((int(r.hp[0,1,0]),int(r.pp[0,0,0,0]),int(r.status[0,0,0])),(1000,9,0))
            self.assertEqual(int(r.sleep_skipped[0,0,0]),0)
        s=self.live();s=s._replace(ability=s.ability.at[0,0,0].set(1))
        r=self.play(s).state
        self.assertEqual((int(r.status_counter[0,0,0]),int(r.sleep_skipped[0,0,0])),(3,1))

    def test_switch_in_restores_only_consecutive_sleep_talk_decrements(self):
        for reset,expected in ((False,5),(True,3)):
            s=self.play(self.live()).state
            s=self.play(s,self.j.array([[1 if reset else 0,0]])).state
            self.assertEqual((int(s.status_counter[0,0,0]),int(s.sleep_skipped[0,0,0])),(3,0 if reset else 2))
            s=self.play(s,self.j.array([[4,0]])).state
            s=self.play(s,self.j.array([[4,0]])).state
            self.assertEqual((int(s.active[0,0]),int(s.status_counter[0,0,0]),int(s.sleep_skipped[0,0,0])),(0,expected,0))

    def test_flinch_still_decrements_sleep_and_records_skipped_turn_without_pp(self):
        s=self.live();s=s._replace(volatile=s.volatile.at[0,0,0].set(1))
        r=self.play(s).state
        self.assertEqual((int(r.status_counter[0,0,0]),int(r.sleep_skipped[0,0,0]),
            int(r.pp[0,0,0,0]),int(r.hp[0,1,0])),(4,1,10,1000))

    def test_caller_choice_lock_and_lum_natural_cure_reset(self):
        s=self.live();s=s._replace(item=s.item.at[0,0,0].set(1))
        r=self.play(s).state
        self.assertEqual(int(r.choice_move[0,0,0]),self.id("sleeptalk"))
        s=self.live();s=s._replace(item=s.item.at[0,0,0].set(2),sleep_skipped=s.sleep_skipped.at[0,0,0].set(2))
        r=self.play(s).state
        self.assertEqual((int(r.item[0,0,0]),int(r.status[0,0,0]),int(r.sleep_skipped[0,0,0])),(0,0,0))
        s=self.play(self.live()).state;s=s._replace(ability=s.ability.at[0,0,0].set(3))
        r=self.play(s,self.j.array([[4,0]])).state
        self.assertEqual((int(r.status[0,0,0]),int(r.status_counter[0,0,0]),int(r.sleep_skipped[0,0,0])),(0,0,0))

    def test_hazard_ko_still_restores_sleep_before_faint_cleanup(self):
        s=self.live(time=4);s=s._replace(hp=s.hp.at[0,0,0].set(1),
            sleep_skipped=s.sleep_skipped.at[0,0,0].set(1),spikes=s.spikes.at[0,0].set(1))
        s=self.play(s,self.j.array([[4,0]])).state
        r=self.play(s,self.j.array([[4,0]])).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.status_counter[0,0,0]),int(r.sleep_skipped[0,0,0])),(0,5,0))
        self.assertTrue(bool(r.forced_switch[0,0]))

    def test_called_protect_and_roar_use_original_sleep_talk_priority(self):
        s=self.call_only(self.live(),"protect")
        s=s._replace(moves=s.moves.at[0,1,0,0].set(self.id("tackle")))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.pp[0,0,0,0]),int(r.pp[0,0,0,1])),(1000,9,10))
        s=self.call_only(self.live(),"roar")
        s=s._replace(moves=s.moves.at[0,1,0,0].set(self.id("tackle")))
        r=self.play(s).state
        self.assertNotEqual(int(r.active[0,1]),0)
        self.assertEqual((int(r.pp[0,1,0,0]),int(r.hp[0,0,0])),(10,1000))

    def test_called_pursuit_hits_recipient_at_ordinary_power_not_outgoing(self):
        s=self.call_only(self.live(),"pursuit")
        r=self.play(s,self.j.array([[0,4]])).state
        control=self.live();control=control._replace(status=self.j.zeros_like(control.status),
            moves=control.moves.at[0,0,0,0].set(self.id("pursuit")))
        regular=self.play(control).state
        self.assertEqual(int(r.hp[0,1,0]),1000)
        self.assertEqual(int(r.hp[0,1,1]),int(regular.hp[0,1,0]))
        self.assertEqual(int(r.pp[0,0,0,1]),10)

    def test_called_baton_pauses_and_does_not_pass_sleep_clock(self):
        s=self.call_only(self.live(),"batonpass")
        s=s._replace(moves=s.moves.at[0,1,0,0].set(self.id("tackle")))
        paused=self.play(s).state
        self.assertTrue(bool(paused.baton_pass[0,0]))
        self.assertEqual((int(paused.pp[0,0,0,0]),int(paused.pp[0,0,0,1]),int(paused.pp[0,1,0,0])),(9,10,10))
        r=self.play(paused,self.j.array([[4,3]])).state
        self.assertEqual((int(r.status[0,0,1]),int(r.sleep_skipped[0,0,1]),int(r.sleep_skipped[0,0,0])),(0,0,1))
        self.assertLess(int(r.hp[0,0,1]),1000)
        self.assertEqual(int(r.pp[0,1,0,0]),9)
        self.assertTrue(bool(self.j.array_equal(r.keys,paused.keys)))

    def test_queued_sleep_talk_uses_saved_slot_roll_after_opposing_baton(self):
        j=self.j;s=self.live(16)
        s=s._replace(status=s.status.at[:,0,0].set(0).at[:,1,0].set(self.m.SLP),
            status_counter=s.status_counter.at[:,0,0].set(0).at[:,1,0].set(5),
            moves=s.moves.at[:,0,0,0].set(self.id("batonpass")).at[:,1,0,:].set(
                j.array([self.id(x) for x in ("sleeptalk","tackle","rest","focuspunch")])))
        paused=self.play(s).state
        self.assertEqual(paused.status_counter[:,1,0].tolist(),[5]*16)
        selected=(paused.sleep_talk_draws[:,1]%2)==0
        self.assertTrue(bool(j.any(selected)));self.assertTrue(bool(j.any(~selected)))
        r=self.play(paused,j.tile(j.array([[4,3]]),(16,1))).state
        self.assertEqual((r.hp[:,0,1]<1000).tolist(),selected.tolist())
        self.assertEqual(r.pp[:,1,0,:2].tolist(),[[9,10]]*16)
        self.assertEqual(r.status_counter[:,1,0].tolist(),[4]*16)
        self.assertTrue(bool(j.array_equal(r.keys,paused.keys)))
        self.assertTrue(bool(j.all(r.sleep_talk_draws==0)))

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(done=s.done.at[3].set(True))
        actions=j.array([[0,0],[0,0],[9,0],[0,0]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(s,actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_gen3_selection_pp_and_sleep_clock(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_sleep_talk_oracle.cjs")),str(Path(SHOWDOWN).resolve())],
            capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);data=json.loads(p.stdout)
        for name in ("normal","pressure","choice"):
            row=data[name]
            self.assertEqual((row["time"],row["skipped"],row["pp"][:2],row["damaged"],row["choices"]),(4,1,[15,7],True,[["tackle"]]))
        self.assertEqual(data["choice"]["lock"],"sleeptalk")
        self.assertTrue(data["zero"]["nopp"]);self.assertFalse(data["zero"]["damaged"])
        for name in ("awake","wake","no_eligible"):self.assertFalse(data[name]["damaged"])
        self.assertEqual(data["early"]["time"],3)
        self.assertEqual((data["switch_back"]["before"]["time"],data["switch_back"]["before"]["skipped"],
            data["switch_back"]["after"]["time"],data["switch_back"]["after"]["skipped"]),(3,2,5,0))
        self.assertEqual((data["reset_switch"]["after"]["time"],data["reset_switch"]["after"]["skipped"]),(3,0))
        self.assertEqual((data["hazard"]["hp"],data["hazard"]["time"],data["hazard"]["skipped"]),(0,5,0))


if __name__=="__main__":unittest.main()
