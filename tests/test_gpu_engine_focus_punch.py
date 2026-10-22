"""Gen3 Focus Punch components, not complete move/game qualification."""
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
class FocusPunchTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton,damage_moves,effects,encore,focus_punch,items,mechanics,pressure,secondary,transition
        from gpu_engine.state import empty_state,FOCUS_PUNCH,FOCUS_LOST,FLINCH
        self.jax,self.j,self.f,self.t,self.m,self.empty=jax,j,focus_punch,transition,mechanics,empty_state
        self.charge,self.lost,self.flinch=FOCUS_PUNCH,FOCUS_LOST,FLINCH
        self.names=['_none','focuspunch','tackle','splash','protect','endure','struggle','encore','batonpass','rockslide','toxic','seismictoss']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta={'vocabulary':{'moves':self.names,'types':types,'abilities':['_none','pressure'],
            'items':['_none','choiceband'],'species':['_none']},
            'moves':[None]+[{'category':'Physical' if name in {'focuspunch','tackle','struggle','rockslide','seismictoss'} else 'Status',
                'flags':{'protect':name in {'focuspunch','tackle','rockslide','encore','toxic'},
                    'bypasssub':name=='encore','failencore':name in {'encore','struggle'}}} for name in self.names[1:]]}
        n=len(self.names);num=j.ones((18,18),j.int32).at[1,8].set(0).at[2,8].set(0)
        self.table=transition.MoveTable(j.array([0,150,35,0,0,0,50,0,0,75,0,0]),
            j.array([0,2,1,1,1,1,0,1,1,6,4,2]),j.ones(n,bool),j.full(n,100),j.ones(n,bool),
            j.array([0,-3,0,0,3,3,0,0,0,0,0,0]),num,j.ones((18,18),j.int32),j.array(6),
            effects=effects.from_metadata(meta),items=items.from_metadata(meta),pressure=pressure.from_metadata(meta),
            baton=baton.from_metadata(meta),encore=encore.from_metadata(meta),secondary=secondary.from_metadata(meta),
            focus_punch=focus_punch.from_metadata(meta),damage_moves=damage_moves.from_metadata(meta))

    def live(self,batch=1,foe=3):
        j=self.j;s=self.empty(batch,seed=223)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),moves=j.full_like(s.moves,3)
                .at[:,0,0,:].set(j.array([1,2,3,3])).at[:,1,0,0].set(foe),
            pp=j.full_like(s.pp,8),last_move=s.last_move.at[:,0,0].set(2),done=j.zeros_like(s.done))

    def play(self,s,actions=None,table=None):
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else actions,
            self.table if table is None else table)

    def test_normal_damage_and_flags_expire_after_ordinary_turn(self):
        a=self.play(self.live()).state
        self.assertLess(int(a.hp[0,1,0]),1000)
        self.assertEqual((int(a.pp[0,0,0,0]),int(a.last_move[0,0,0])),(7,1))
        self.assertEqual(int(a.volatile[0,0,0])&(self.charge|self.lost),0)

    def test_direct_hit_breaks_focus_after_pp_last_and_choice_pressure(self):
        s=self.live(foe=2);s=s._replace(item=s.item.at[0,0,0].set(1),ability=s.ability.at[0,1,0].set(1))
        a=self.play(s).state
        self.assertLess(int(a.hp[0,0,0]),1000);self.assertEqual(int(a.hp[0,1,0]),1000)
        self.assertEqual((int(a.pp[0,0,0,0]),int(a.last_move[0,0,0]),int(a.choice_move[0,0,0])),(6,1,1))

    def test_substitute_blocks_loss_even_when_broken(self):
        for hp in (200,1):
            s=self.live(foe=2);s=s._replace(substitute_hp=s.substitute_hp.at[0,0,0].set(hp))
            a=self.play(s).state
            self.assertEqual(int(a.hp[0,0,0]),1000);self.assertLess(int(a.hp[0,1,0]),1000)
            self.assertLess(int(a.substitute_hp[0,0,0]),hp)

    def test_fixed_level_damage_breaks_despite_zero_catalog_power(self):
        a=self.play(self.live(foe=11)).state
        self.assertEqual((int(a.hp[0,0,0]),int(a.hp[0,1,0]),int(a.pp[0,0,0,0])),(900,1000,7))

    def test_miss_immunity_and_status_do_not_break_focus(self):
        s=self.live(foe=2)
        miss=self.table._replace(accuracy=self.table.accuracy.at[2].set(0),always_hit=self.table.always_hit.at[2].set(False))
        immune=s._replace(types=s.types.at[0,0,0,0].set(8))
        for state,table in ((s,miss),(immune,self.table),(self.live(foe=10),self.table)):
            a=self.play(state,table=table).state;self.assertLess(int(a.hp[0,1,0]),1000)

    def test_zero_actual_damage_hit_still_breaks_and_nonstatus_category_not_power(self):
        j=self.j;s=self.live();s=self.f.prepare(s,j.array([[1,3]]),self.table.focus_punch,
            ordinary=j.array([True]),move_requests=j.array([[True,True]]))
        a=self.f.hit(s,j.array([1]),j.array([2]),self.table.focus_punch,eligible=j.array([True]))
        self.assertTrue(bool(self.f.failed(a,j.array([0]),j.array([1]),self.table.focus_punch)[0]))
        self.assertEqual(int(a.hp[0,0,0]),1000) # No positive-damage precondition.
        modified=self.table.focus_punch._replace(damaging=self.table.focus_punch.damaging.at[2].set(False))
        a=self.f.hit(s,j.array([1]),j.array([2]),modified,eligible=j.array([True]))
        self.assertFalse(bool(self.f.failed(a,j.array([0]),j.array([1]),modified)[0]))

    def test_preparation_vetoes_added_flinch(self):
        from gpu_engine import secondary
        j=self.j;s=self.live();s=self.f.prepare(s,j.array([[1,3]]),self.table.focus_punch,
            ordinary=j.array([True]),move_requests=j.array([[True,True]]))
        a=secondary.apply(s,j.array([1]),j.array([9]),j.array([0]),self.table.secondary,eligible=j.array([True]))
        self.assertEqual(int(a.volatile[0,0,0])&self.flinch,0)
        ordinary=secondary.apply(self.live(),j.array([1]),j.array([9]),j.array([0]),self.table.secondary,eligible=j.array([True]))
        self.assertEqual(int(ordinary.volatile[0,0,0])&self.flinch,self.flinch)

    def test_sleep_is_before_pp_and_encore_override_uses_original_queue(self):
        s=self.live();s=s._replace(status=s.status.at[0,0,0].set(self.m.SLP),status_counter=s.status_counter.at[0,0,0].set(5))
        a=self.play(s).state
        self.assertEqual((int(a.pp[0,0,0,0]),int(a.last_move[0,0,0])),(8,2))
        a=self.play(self.live(foe=7)).state
        self.assertEqual((int(a.pp[0,0,0,0]),int(a.pp[0,0,0,1]),int(a.last_move[0,0,0])),(8,7,2))
        self.assertLess(int(a.hp[0,1,0]),1000)

    def test_baton_pause_retains_preparation_then_resumes_without_second_pp(self):
        j=self.j;s=self.live(foe=8);a=self.play(s).state
        self.assertTrue(bool(a.pending_residual[0]));self.assertEqual(int(a.pp[0,0,0,0]),8)
        self.assertEqual(int(a.volatile[0,0,0])&self.charge,self.charge)
        b=self.play(a,j.array([[0,4]])).state
        self.assertEqual(int(b.pp[0,0,0,0]),7);self.assertLess(int(b.hp[0,1,1]),1000)
        self.assertEqual(int(b.volatile[0,0,0])&(self.charge|self.lost),0)

    def test_jit_prefix_invalid_and_terminal_identity(self):
        j=self.j;s=self.live(2);actions=j.zeros((2,2),j.int32)
        a=self.play(s);b=self.jax.jit(self.t.step)(s,actions,self.table)
        one=self.t.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for x,y,z in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b),self.jax.tree.leaves(one)):
            self.assertTrue(bool(j.array_equal(x,y)));self.assertTrue(bool(j.array_equal(x[:1],z)))
        s=s._replace(done=s.done.at[1].set(True));actions=actions.at[0,0].set(99)
        r=self.t.step(s,actions,self.table)
        for x,y in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r.state)):
            self.assertTrue(bool(j.array_equal(x,y)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_pinned_gen3_hit_sub_pp_and_encore(self):
        cases=json.loads(subprocess.check_output([NODE,str(Path(__file__).with_name('gpu_focus_punch_oracle.cjs')),SHOWDOWN],text=True))
        for name in ('hit','hit_band','flinch','zero_damage','fixed_damage'):
            self.assertTrue(cases[name]['failed'],name);self.assertEqual(cases[name]['pp'],31,name)
        for name in ('normal','miss','immune','sub','broken_sub','status'):
            self.assertFalse(cases[name]['failed'],name);self.assertLess(cases[name]['hp'][1],1000,name)
        self.assertEqual(cases['hit_band']['lock'],'focuspunch')
        self.assertEqual((cases['sleep']['pp'],cases['sleep']['last']),(32,'tackle'))
        self.assertEqual((cases['encore']['pp'],cases['encore']['tacklePP'],cases['encore']['last']),(32,55,'tackle'))
        self.assertTrue(all(c['focused'] for c in cases.values()))
