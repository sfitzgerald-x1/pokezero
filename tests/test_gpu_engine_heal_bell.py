"""Gen3 Heal Bell components, not complete game or move qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class HealBellTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,mechanics,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.e,self.m,self.t,self.empty=jax,j,effects,mechanics,transition,empty_state
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta={'vocabulary':{'moves':['_none','healbell','splash','protect','struggle'],
            'types':types,'abilities':['_none','soundproof']},
            'moves':[None]+[{'flags':{'protect':n=='splash'}} for n in ['healbell','splash','protect','struggle']]}
        self.table=transition.MoveTable(j.zeros(5,j.int32),j.ones(5,j.int32),j.zeros(5,bool),
            j.full(5,100),j.ones(5,bool),j.array([0,0,0,3,0]),j.ones((18,18),j.int32),
            j.ones((18,18),j.int32),j.array(4),effects=effects.from_metadata(meta))

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=199)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            moves=j.full_like(s.moves,2).at[:,0,0,0].set(1),pp=j.full_like(s.pp,8),
            done=j.zeros_like(s.done))

    def play(self,s):return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),self.table)

    def test_cures_living_own_party_and_sleep_clock_not_confusion_or_foe(self):
        j=self.j;s=self.live();s=s._replace(status=s.status.at[0,0].set(j.array([2,5,4,1,6,3]))
            .at[0,1].set(3),status_counter=j.full_like(s.status_counter,5),
            sleep_skipped=j.full_like(s.sleep_skipped,2),confusion_counter=j.full_like(s.confusion_counter,4))
        a=self.e.cure_team(s,j.array([0]),self.table.effects,eligible=j.array([True]))
        self.assertEqual(a.status[0,0].tolist(),[0]*6)
        self.assertEqual(a.status_counter[0,0].tolist(),[0]*6)
        self.assertEqual(a.sleep_skipped[0,0].tolist(),[0]*6)
        self.assertTrue(bool(j.array_equal(a.confusion_counter,s.confusion_counter)))
        self.assertTrue(bool(j.array_equal(a.status[0,1],s.status[0,1])))

    def test_soundproof_includes_user_and_fainted_members_stay_unchanged(self):
        j=self.j;s=self.live();s=s._replace(status=j.full_like(s.status,3),
            ability=s.ability.at[0,0,0].set(1).at[0,0,2].set(1),hp=s.hp.at[0,0,5].set(0))
        a=self.e.cure_team(s,j.array([0]),self.table.effects,eligible=j.array([True]))
        self.assertEqual(a.status[0,0].tolist(),[3,0,3,0,0,3])

    def test_dispatch_cures_before_residual_spends_pp_and_ignores_foe_protect_sub(self):
        j=self.j
        for foe in (2,3):
            s=self.live();s=s._replace(status=s.status.at[0,0].set(2),
                moves=s.moves.at[0,1,0,0].set(foe),substitute_hp=s.substitute_hp.at[0,1,0].set(19))
            a=self.play(s).state
            self.assertEqual(a.status[0,0].tolist(),[0]*6)
            self.assertEqual((int(a.hp[0,0,0]),int(a.pp[0,0,0,0])),(1000,7))

    def test_no_status_still_spends_pp_sleep_prevents_direct_use(self):
        a=self.play(self.live()).state;self.assertEqual(int(a.pp[0,0,0,0]),7)
        s=self.live();s=s._replace(status=s.status.at[0,0,0].set(self.m.SLP),
            status_counter=s.status_counter.at[0,0,0].set(3))
        a=self.play(s).state
        self.assertEqual((int(a.status[0,0,0]),int(a.pp[0,0,0,0])),(self.m.SLP,8))

    def test_both_sides_and_batched_ineligible_identity(self):
        j=self.j;s=self.live(3);s=s._replace(status=j.full_like(s.status,3))
        a=self.e.cure_team(s,j.array([0,1,0]),self.table.effects,eligible=j.array([True,True,False]))
        self.assertEqual(a.status[:,0,0].tolist(),[0,3,3]);self.assertEqual(a.status[:,1,0].tolist(),[3,0,3])

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(2);s=s._replace(status=s.status.at[:,0].set(2));actions=j.zeros((2,2),j.int32)
        a=self.play(s);b=self.jax.jit(self.t.step)(s,actions,self.table)
        one=self.t.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for x,y,z in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b),self.jax.tree.leaves(one)):
            self.assertTrue(bool(j.array_equal(x,y)));self.assertTrue(bool(j.array_equal(x[:1],z)))
        s=s._replace(done=s.done.at[1].set(True));actions=actions.at[0,0].set(99)
        r=self.t.step(s,actions,self.table)
        for x,y in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r.state)):
            self.assertTrue(bool(j.array_equal(x,y)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_pinned_gen3_party_status_and_soundproof(self):
        cases=json.loads(subprocess.check_output([NODE,str(Path(__file__).with_name('gpu_heal_bell_oracle.cjs')),SHOWDOWN],text=True))
        for name,c in cases.items():
            self.assertEqual(c['status'],['brn' if name=='user_sound' else '','','tox','','','psn'],name)
            self.assertEqual((c['foe'],c['confused'],c['pp']),('psn',True,7),name)
