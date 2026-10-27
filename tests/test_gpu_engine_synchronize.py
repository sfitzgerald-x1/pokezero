"""Gen3 Synchronize across the implemented primary, secondary and contact writers."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')

@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class SynchronizeTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import baton, effects, items, mechanics, secondary, transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.m,self.sec,self.tr,self.empty=jax,j,mechanics,secondary,transition,empty_state
        self.names=['_none','thunderbolt','firepunch','sludgebomb','icebeam','toxic','willowisp',
            'thunderwave','sleeppowder','bodyslam','splash','batonpass','struggle']
        self.abilities=['_none','synchronize','static','limber','waterveil','immunity','shielddust']
        self.types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        self.meta={'vocabulary':{'moves':self.names,'types':self.types,'abilities':self.abilities,
            'items':['_none','lumberry'],'species':['_none']},'moves':[None]+[
                dict(id=n,source_type='Electric' if n=='thunderbolt' else 'Normal',category='Physical',
                    flags={'protect':1,**({'contact':1} if n=='bodyslam' else {})}) for n in self.names[1:]]}
        self.profile=secondary.from_metadata(self.meta);n=len(self.names)
        self.table=transition.MoveTable(j.array([0,50,50,50,50,0,0,0,0,50,0,0,50]),
            j.array([0,13,10,4,15,4,10,13,11,1,1,1,1]),j.ones(n,bool),j.full(n,100),
            j.ones(n,bool),j.zeros(n,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),
            j.array(self.id('struggle')),effects=effects.from_metadata(self.meta),
            items=items.from_metadata(self.meta),secondary=self.profile,baton=baton.from_metadata(self.meta))

    def id(self,n):return self.names.index(n)
    def ability(self,n):return self.abilities.index(n)
    def live(self,batch=100,side=0,move='thunderbolt'):
        j=self.j;s=self.empty(batch,seed=139)
        abilities=s.ability.at[:,1-side,0].set(self.ability('synchronize'))
        return s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),
            stats=j.full_like(s.stats,200).at[:,side,:,5].set(300),level=j.full_like(s.level,100),
            types=s.types.at[...,0].set(11),ability=abilities,base_ability=abilities,
            moves=j.full_like(s.moves,self.id('splash')).at[:,side,0,0].set(self.id(move)),
            pp=j.full_like(s.pp,10),done=j.zeros_like(s.done))
    def play(self,s,side=0,actions=None,jit=False):
        j=self.j;c=self.tr.sample_chance_inputs(s)
        c=c._replace(secondary=j.zeros_like(c.secondary).at[:,side].set(j.arange(s.hp.shape[0])%100))
        fn=self.jax.jit(self.tr.step_with_chance) if jit else self.tr.step_with_chance
        return fn(s,j.zeros(s.active.shape,j.int32) if actions is None else actions,self.table,c)
    def direct(self,s,move,side=0,roll=0):
        return self.sec.apply(s,side,self.j.full(s.hp.shape[0],self.id(move)),roll,self.profile,eligible=True)

    def test_secondary_exact_ten_percent_in_both_seats(self):
        for side in (0,1):
            s=self.live(side=side);r=self.direct(s,'thunderbolt',side,self.j.arange(100))
            self.assertEqual(r.status[:,side,0].tolist(),[self.m.PAR]*10+[0]*90)
            self.assertEqual(r.status[:,1-side,0].tolist(),[self.m.PAR]*10+[0]*90)
            self.assertTrue(bool(self.j.array_equal(r.keys,s.keys)))

    def test_current_ability_and_fresh_reflected_counters(self):
        s=self.live(1);s=s._replace(ability=s.ability.at[0,1,0].set(0))
        self.assertEqual(int(self.direct(s,'thunderbolt').status[0,0,0]),0)
        s=self.live(1);s=s._replace(status_counter=s.status_counter.at[0,0,0].set(6),
            sleep_skipped=s.sleep_skipped.at[0,0,0].set(2),rest_origin=s.rest_origin.at[0,0,0].set(True))
        r=self.direct(s,'sludgebomb')
        self.assertEqual((int(r.status[0,0,0]),int(r.status_counter[0,0,0]),
            int(r.sleep_skipped[0,0,0]),bool(r.rest_origin[0,0,0])),(self.m.PSN,0,0,False))

    def test_primary_toxic_burn_and_paralysis_both_seats(self):
        for side in (0,1):
            for move,got,reflected in [('toxic',self.m.TOX,self.m.PSN),
                    ('willowisp',self.m.BRN,self.m.BRN),('thunderwave',self.m.PAR,self.m.PAR)]:
                r=self.play(self.live(1,side,move),side).state
                self.assertEqual(int(r.status[0,1-side,0]),got)
                self.assertEqual(int(r.status[0,side,0]),reflected)

    def test_static_contact_reflects_in_both_seats(self):
        for side in (0,1):
            s=self.live(60,side,'bodyslam');a=s.ability.at[:,side,0].set(self.ability('synchronize')).at[:,1-side,0].set(self.ability('static'))
            s=s._replace(ability=a,base_ability=a)
            r=self.sec.apply_contact(s,side,self.j.full(60,self.id('bodyslam')),self.j.arange(60),self.profile,eligible=True)
            self.assertEqual(r.status[:,side,0].tolist(),[self.m.PAR]*20+[0]*40)
            self.assertEqual(r.status[:,1-side,0].tolist(),[self.m.PAR]*20+[0]*40)

    def test_sleep_freeze_existing_failed_status_and_fainted_source(self):
        r=self.direct(self.live(1),'icebeam');self.assertEqual(int(r.status[0,0,0]),0)
        r=self.play(self.live(1,move='sleeppowder')).state;self.assertEqual(int(r.status[0,0,0]),0)
        for field,value in [('status',self.m.BRN),('hp',0)]:
            s=self.live(1);s=s._replace(**{field:getattr(s,field).at[0,0,0].set(value)})
            r=self.direct(s,'thunderbolt');self.assertEqual(int(r.status[0,0,0]),self.m.BRN if field=='status' else 0)
        s=self.live(1);s=s._replace(status=s.status.at[0,1,0].set(self.m.BRN))
        self.assertEqual(int(self.direct(s,'thunderbolt').status[0,0,0]),0)

    def test_reflected_type_and_ability_immunity_but_not_ground_or_substitute(self):
        for move,ability,typ in [('thunderbolt','limber',11),('firepunch','waterveil',11),
                ('sludgebomb','immunity',11),('firepunch','_none',10),('sludgebomb','_none',9)]:
            s=self.live(1);s=s._replace(ability=s.ability.at[0,0,0].set(self.ability(ability)),
                types=s.types.at[0,0,0,0].set(typ))
            self.assertEqual(int(self.direct(s,move).status[0,0,0]),0,(move,ability,typ))
        s=self.live(1);s=s._replace(types=s.types.at[0,0,0,0].set(5),substitute_hp=s.substitute_hp.at[0,0,0].set(100))
        self.assertEqual(int(self.direct(s,'thunderbolt').status[0,0,0]),self.m.PAR)
        r=self.sec.apply(self.live(1),0,self.j.array([self.id('thunderbolt')]),0,self.profile,eligible=True,substitute_hit=True)
        self.assertEqual(r.status[0,:,0].tolist(),[0,0])

    def test_both_lum_cures_follow_reflection(self):
        for source_lum,target_lum in [(True,False),(False,True),(True,True)]:
            s=self.live(1);s=s._replace(item=s.item.at[0,0,0].set(int(source_lum)).at[0,1,0].set(int(target_lum)))
            r=self.play(s).state
            self.assertEqual(r.status[0,:,0].tolist(),[0 if source_lum else self.m.PAR,0 if target_lum else self.m.PAR])
            self.assertEqual(r.item[0,:,0].tolist(),[0,0])

    def test_secondary_lum_then_static_then_synchronize_can_status_target_again(self):
        for side in (0,1):
            s=self.live(1,side,'bodyslam')
            a=s.ability.at[:,side,0].set(self.ability('synchronize')).at[:,1-side,0].set(self.ability('static'))
            s=s._replace(ability=a,base_ability=a,item=s.item.at[:,1-side,0].set(1))
            c=self.tr.sample_chance_inputs(s)
            c=c._replace(secondary=self.j.zeros_like(c.secondary),phazing=self.j.zeros_like(c.phazing))
            r=self.tr.step_with_chance(s,self.j.zeros(s.active.shape,self.j.int32),self.table,c).state
            self.assertEqual(r.status[0,:,0].tolist(),[self.m.PAR,self.m.PAR])
            self.assertEqual(r.item[0,1-side,0],0)

    def test_switched_in_holder_and_baton_carry_use_active_recipient(self):
        j=self.j
        for side in (0,1):
            s=self.live(10,side);a=s.ability.at[:,1-side,0].set(0).at[:,1-side,1].set(self.ability('synchronize'))
            s=s._replace(ability=a,base_ability=a)
            actions=j.zeros(s.active.shape,j.int32).at[:,1-side].set(4)
            r=self.play(s,side,actions).state
            self.assertEqual(r.active[:,1-side].tolist(),[1]*10)
            self.assertEqual(r.status[:,side,0].tolist(),[self.m.PAR]*10)
            self.assertEqual(r.status[:,1-side,1].tolist(),[self.m.PAR]*10)
            self.assertEqual(r.status[:,1-side,0].tolist(),[0]*10)
            s=s._replace(moves=s.moves.at[:,1-side,0,0].set(self.id('batonpass')),
                stats=s.stats.at[:,1-side,0,5].set(500))
            paused=self.play(s,side).state;self.assertTrue(bool(j.all(paused.pending_residual)))
            r=self.tr.step(paused,j.zeros(s.active.shape,j.int32).at[:,1-side].set(4),self.table).state
            self.assertEqual(r.status[:,side,0].tolist(),[self.m.PAR]*10)
            self.assertEqual(r.status[:,1-side,1].tolist(),[self.m.PAR]*10)
            self.assertTrue(bool(j.array_equal(r.keys,paused.keys)))

    def test_jit_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(done=s.done.at[3].set(True))
        actions=j.array([[0,0],[0,0],[99,0],[0,0]])
        eager=self.play(s,actions=actions);compiled=self.play(s,actions=actions,jit=True)
        for a,b in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(j.array_equal(a,b)))
        small=self.play(self.jax.tree.map(lambda a:a[:2],s),jit=True)
        for a,b in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(a[:2],b)))
        for a,b in zip(s,eager.state):self.assertTrue(bool(j.array_equal(a[2:],b[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown and Node')
    def test_pinned_gen3_status_events_and_lum_order(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name('gpu_synchronize_oracle.cjs')),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);result=json.loads(p.stdout)
        self.assertEqual(result['catalog']['thunderboltChance'],10)
        frozen=json.loads((Path(__file__).parent/'data/gpu_synchronize_catalog.json').read_text())
        self.assertEqual(result['metadata'],frozen['metadata'])
        self.assertEqual(len(result['cases']),23)
        for name,case in result['cases'].items():
            self.assertEqual(case['actual'],case['expected'],name)
