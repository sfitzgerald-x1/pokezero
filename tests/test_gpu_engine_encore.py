"""Selected Gen3 Encore fixtures, not full move-lock or game qualification."""
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
class EncoreTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton,effects,encore,items,mechanics,pressure,pursuit,sleep_talk,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.enc,self.tr,self.empty=jax,j,mechanics,encore,transition,empty_state
        self.names=["_none","encore","tackle","splash","protect","struggle","sleeptalk",
            "solarbeam","focuspunch","batonpass","pursuit"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta={"vocabulary":{"moves":self.names,"types":types,"abilities":["_none","pressure"],
            "items":["_none","choiceband"],"species":["_none"]},
            "moves":[None]+[{"flags":{"protect":1,"bypasssub":name=="encore",
                "failencore":name in {"encore","struggle"},"charge":name=="solarbeam",
                "nosleeptalk":name in {"sleeptalk","focuspunch"}}} for name in self.names[1:]]}
        self.profile=encore.from_metadata(meta);n=len(self.names)
        self.table=transition.MoveTable(j.array([0,0,35,0,0,50,0,120,150,0,40]),
            j.ones(n,j.int32).at[10].set(17),j.ones(n,bool).at[10].set(False),j.full(n,100),
            j.ones(n,bool),j.zeros(n,j.int32).at[4].set(3),j.ones((18,18),j.int32),
            j.ones((18,18),j.int32),j.array(5),effects=effects.from_metadata(meta),
            items=items.from_metadata(meta),pressure=pressure.from_metadata(meta),pursuit=pursuit.from_metadata(meta),
            baton=baton.from_metadata(meta),sleep_talk=sleep_talk.from_metadata(meta),encore=self.profile)

    def live(self,batch=1,*,slow=False):
        j=self.j;s=self.empty(batch,seed=191)
        moves=j.full_like(s.moves,3).at[:,0,0,0].set(1).at[:,1,0,:].set(j.array([2,3,4,6]))
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(50 if slow else 300),
            level=j.full_like(s.level,100),hp=j.full_like(s.hp,10000),max_hp=j.full_like(s.hp,10000),
            moves=moves,pp=j.full_like(s.pp,10),last_move=s.last_move.at[:,1,0].set(2),
            last_action_kind=s.last_action_kind.at[:,1].set(2),
            last_action_index=s.last_action_index.at[:,1].set(0),
            done=j.zeros_like(s.done))

    def play(self,s,actions=None):
        a=self.j.tile(self.j.array([[0,1]]),(s.hp.shape[0],1)) if actions is None else actions
        return self.tr.step(s,a,self.table)

    def duration_roll(self,s,side=0):
        keys=self.jax.vmap(lambda k:self.jax.random.split(k,2))(s.keys)
        draws=self.jax.vmap(lambda k:self.jax.random.randint(k,(2,12),0,
            self.j.array([2,100,100,100,16,16,100,4,8,2,2,2])))(keys[:,1])
        return draws[:,side,7]

    def test_all_four_duration_rolls_and_fast_vs_slow_target_queue(self):
        j=self.j;s=self.live(4)
        for queued,expected in ((True,[3,4,5,6]),(False,[4,5,6,7])):
            a=self.enc.apply(s,j.ones(4,j.int32),j.ones(4,j.int32),j.arange(4),self.profile,
                eligible=j.ones(4,bool),queued=j.full(4,queued))
            self.assertEqual(a.encore_counter[:,1,0].tolist(),expected)
        fast=self.play(s).state
        self.assertEqual(fast.encore_counter[:,1,0].tolist(),(2+self.duration_roll(s)).tolist())
        self.assertEqual(fast.encore_move[:,1,0].tolist(),[2]*4)
        self.assertEqual(fast.pp[:,1,0,:2].tolist(),[[9,10]]*4)
        slow=self.live(4,slow=True);a=self.play(slow).state
        self.assertEqual(a.encore_counter[:,1,0].tolist(),(3+self.duration_roll(slow)).tolist())
        self.assertEqual(a.encore_move[:,1,0].tolist(),[3]*4)
        self.assertEqual(a.pp[:,1,0,:2].tolist(),[[10,9]]*4)

    def test_protect_blocks_but_substitute_and_ghost_do_not(self):
        s=self.live();j=self.j
        a=self.play(s,j.array([[0,2]])).state
        self.assertEqual(int(a.encore_counter[0,1,0]),0)
        for target in (s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(19)),
                s._replace(types=s.types.at[0,1,0,0].set(8))):
            self.assertEqual(int(self.play(target).state.encore_move[0,1,0]),2)

    def test_none_switch_struggle_invalid_excluded_and_zero_pp_history_fail(self):
        s=self.live()
        for kind,index in ((0,-1),(1,0),(3,-1),(2,4)):
            a=s._replace(last_action_kind=s.last_action_kind.at[0,1].set(kind),
                last_action_index=s.last_action_index.at[0,1].set(index))
            self.assertEqual(int(self.play(a).state.encore_counter[0,1,0]),0)
        for move in (0,1,5):
            a=s._replace(moves=s.moves.at[0,1,0,0].set(move))
            self.assertEqual(int(self.play(a).state.encore_counter[0,1,0]),0)
        a=s._replace(pp=s.pp.at[0,1,0,0].set(0))
        self.assertEqual(int(self.play(a).state.encore_counter[0,1,0]),0)

    def test_no_refresh_and_pp_exhaustion_ends_at_residual(self):
        s=self.live();s=s._replace(encore_move=s.encore_move.at[0,1,0].set(2),
            encore_counter=s.encore_counter.at[0,1,0].set(2))
        a=self.play(s,self.j.array([[0,0]])).state
        self.assertEqual(int(a.encore_counter[0,1,0]),1)
        s=s._replace(pp=s.pp.at[0,1,0,0].set(1))
        a=self.play(s,self.j.array([[0,0]])).state
        self.assertEqual((int(a.encore_counter[0,1,0]),int(a.encore_move[0,1,0])),(0,0))
        self.assertTrue(bool(self.tr.action_mask(a,self.table)[0,1,1]))

    def test_encore_mask_disables_other_slots_but_keeps_switches_and_struggle(self):
        s=self.live();s=s._replace(encore_move=s.encore_move.at[0,1,0].set(2),
            encore_counter=s.encore_counter.at[0,1,0].set(2))
        mask=self.tr.action_mask(s,self.table)[0,1]
        self.assertEqual(mask.tolist(),[True,False,False,False]+[True]*5)
        s=s._replace(pp=s.pp.at[0,1,0,0].set(0))
        self.assertEqual(self.tr.action_mask(s,self.table)[0,1,:4].tolist(),[True,False,False,False])

    def test_forced_protect_retains_original_splash_queue_priority(self):
        s=self.live();s=s._replace(last_move=s.last_move.at[0,1,0].set(4),
            last_action_index=s.last_action_index.at[0,1].set(2))
        a=self.play(s).state
        self.assertEqual(int(a.encore_move[0,1,0]),4)
        self.assertEqual(int(a.pp[0,1,0,2]),9)
        self.assertEqual(int(a.protect_counter[0,1,0]),0) # Already last queued action.

    def test_step_dispatches_struggle_when_encore_slot_is_unusable(self):
        s=self.live();s=s._replace(encore_move=s.encore_move.at[0,1,0].set(2),
            encore_counter=s.encore_counter.at[0,1,0].set(2))
        variants=(s._replace(pp=s.pp.at[0,1,0,0].set(0)),
            s._replace(disabled=s.disabled.at[0,1,0,0].set(True)),
            s._replace(choice_move=s.choice_move.at[0,1,0].set(3)))
        for state in variants:
            a=self.play(state,self.j.array([[1,0]])).state
            self.assertEqual(int(a.last_move[0,1,0]),5)
            self.assertEqual(a.pp[0,1,0].tolist(),state.pp[0,1,0].tolist())
            self.assertLess(int(a.hp[0,0,0]),10000)
            self.assertLess(int(a.hp[0,1,0]),10000)

    def test_sleep_talk_records_and_forces_caller_not_called_move(self):
        s=self.live();s=s._replace(last_move=s.last_move.at[0,1,0].set(6),
            last_action_index=s.last_action_index.at[0,1].set(3),
            status=s.status.at[0,1,0].set(self.m.SLP),status_counter=s.status_counter.at[0,1,0].set(5))
        a=self.play(s).state
        self.assertEqual((int(a.encore_move[0,1,0]),int(a.last_move[0,1,0])),(6,6))
        self.assertEqual(a.pp[0,1,0].tolist(),[10,10,10,9])
        self.assertEqual(a.hp[0,:,0].tolist(),[10000,10000])

    def test_first_choice_sleep_talk_calls_but_locked_second_use_does_not(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0].set(self.j.array([6,2,7,8])),
            item=s.item.at[0,0,0].set(1),status=s.status.at[0,0,0].set(self.m.SLP),
            status_counter=s.status_counter.at[0,0,0].set(5))
        a=self.play(s).state
        self.assertLess(int(a.hp[0,1,0]),10000);self.assertEqual(int(a.choice_move[0,0,0]),6)
        b=self.play(a).state
        self.assertEqual(int(b.hp[0,1,0]),int(a.hp[0,1,0]))
        self.assertEqual(b.pp[0,0,0].tolist(),[8,10,10,10])

    def test_before_move_failure_preserves_last_and_does_not_spend_pp(self):
        s=self.live();s=s._replace(status=s.status.at[0,0,0].set(self.m.SLP),
            status_counter=s.status_counter.at[0,0,0].set(5),last_move=s.last_move.at[0,0,0].set(2))
        a=self.play(s).state
        self.assertEqual((int(a.last_move[0,0,0]),int(a.pp[0,0,0,0])),(2,10))

    def test_switch_and_baton_clear_nocopy_fields_and_faint_clears_tracking(self):
        from gpu_engine import baton
        s=self.live();s=s._replace(encore_move=s.encore_move.at[0,1,0].set(2),
            encore_counter=s.encore_counter.at[0,1,0].set(4))
        a=self.play(s,self.j.array([[1,4]])).state
        self.assertEqual((int(a.last_move[0,1,0]),int(a.encore_counter[0,1,0])),(0,0))
        copied=baton.copy_to_recipient(s,self.j.array([1]),self.j.array([1]),eligible=self.j.array([True]))
        self.assertEqual((int(copied.last_move[0,1,1]),int(copied.encore_counter[0,1,1])),(0,0))
        low=s._replace(hp=s.hp.at[0,1,0].set(1),moves=s.moves.at[0,0,0,0].set(2))
        a=self.play(low,self.j.array([[0,0]])).state
        self.assertEqual((int(a.last_move[0,1,0]),int(a.encore_counter[0,1,0])),(0,0))

    def test_pursuit_ko_installs_chosen_switch_and_ticks_encore(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0,0].set(10),
            last_action_kind=s.last_action_kind.at[0,0].set(2),
            last_action_index=s.last_action_index.at[0,0].set(0),
            encore_move=s.encore_move.at[0,0,0].set(10),encore_counter=s.encore_counter.at[0,0,0].set(3),
            hp=s.hp.at[0,1,0].set(1))
        a=self.play(s,self.j.array([[0,4]])).state
        self.assertEqual(int(a.last_move[0,0,0]),10)
        self.assertEqual(int(a.active[0,1]),1)
        self.assertFalse(bool(a.pending_residual[0]));self.assertEqual(int(a.encore_counter[0,0,0]),2)

    def test_ordinary_ko_pause_retains_clock_until_replacement(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0,0].set(2),
            last_action_kind=s.last_action_kind.at[0,0].set(2),
            last_action_index=s.last_action_index.at[0,0].set(0),
            encore_move=s.encore_move.at[0,0,0].set(2),encore_counter=s.encore_counter.at[0,0,0].set(3),
            hp=s.hp.at[0,1,0].set(1))
        a=self.play(s,self.j.array([[0,1]])).state
        self.assertEqual(int(a.last_move[0,0,0]),2)
        self.assertTrue(bool(a.pending_residual[0]));self.assertEqual(int(a.encore_counter[0,0,0]),3)
        b=self.play(a,self.j.array([[0,4]])).state
        self.assertEqual(int(b.encore_counter[0,0,0]),2)

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(done=s.done.at[3].set(True))
        actions=j.array([[0,1],[0,1],[9,0],[0,0]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(s,actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_gen3_override_clock_pp_and_nocopy(self):
        r=subprocess.run([NODE,str(Path(__file__).with_name("gpu_encore_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        cases=json.loads(r.stdout)
        self.assertEqual(cases["durations"],[2,3,4,5])
        self.assertEqual((cases["fast"]["initial"]["move"],cases["fast"]["initial"]["duration"]),("tackle",2))
        self.assertEqual((cases["slow"]["initial"]["move"],cases["slow"]["initial"]["duration"]),("splash",3))
        for name in ("none","zero","missing","excluded","protect"):
            self.assertEqual(cases[name]["initial"]["duration"],0)
        for name in ("sub","ghost"):self.assertEqual(cases[name]["initial"]["move"],"tackle")
        self.assertEqual(cases["repeat"]["after"]["duration"],1)
        self.assertEqual((cases["switch"]["after"]["duration"],cases["switch"]["after"]["last"]),(0,""))
        self.assertEqual(cases["asleep"]["initial"]["last"],"sleeptalk")
        self.assertFalse(cases["asleep"]["called"]);self.assertEqual(cases["asleep"]["hpLoss"],0)
        self.assertEqual(cases["forced_protect"]["initial"]["move"],"protect")
        self.assertEqual(cases["fallback"]["initial"]["last"],"struggle")
        self.assertEqual(cases["fallback"]["initial"]["pp"],[0,64,16,16])
        self.assertEqual(cases["fallback"]["initial"]["duration"],0)
        self.assertEqual(cases["choice"],dict(firstHit=True,secondHit=False,pp=14,lock="sleeptalk"))


if __name__=="__main__":unittest.main()
