"""Authored omissions from saved counterexamples; no roster replay or coverage credit."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')
MOVES=('silverwind','ancientpower','mudshot','sacredfire')
EXTRAS=({}, {'sub':999}, {'sub':1}, {'ko':True}, {'endure':True}, {'protect':True},
        {'miss':True}, {'target':'shielddust'}, {'target':'clearbody'}, {'target':'whitesmoke'},
        {'source':'serenegrace','roll':19}, {'source':'serenegrace','roll':20},
        {'roll':9}, {'roll':10}, {'roll':49}, {'roll':50}, {'stage':5}, {'stage':6},
        {'targetStage':-6}, {'target':'waterveil'}, {'type':'Fire'}, {'type':'Flying'})
CASES=[dict(move=move,side=side,**extra) for move in MOVES for side in (0,1) for extra in EXTRAS]


def components(*,real_accuracy=False):
    import jax
    import jax.numpy as j
    from gpu_engine import absorption,effects,secondary,transition
    names=['_none','splash','struggle','protect','endure',*MOVES,'brickbreak']
    types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
           'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
    abilities=['_none','serenegrace','shielddust','clearbody','whitesmoke','waterveil','levitate','static']
    move_types={'silverwind':'Bug','ancientpower':'Rock','mudshot':'Ground','sacredfire':'Fire','brickbreak':'Fighting'}
    meta={'vocabulary':{'moves':names,'types':types,'abilities':abilities,
        'species':['_none','ditto','pikachu','blissey','snorlax','skarmory','slaking'],
        'items':['_none','lumberry']},'moves':[None]+[
        {'id':n,'source_type':move_types.get(n,'Normal'),'category':'Physical' if n in ('silverwind','ancientpower','mudshot','struggle','brickbreak') else 'Special' if n=='sacredfire' else 'Status',
         'flags':{'protect':int(n in MOVES or n=='brickbreak'),'contact':int(n in ('ancientpower','brickbreak')),'defrost':int(n=='sacredfire')}} for n in names[1:]]}
    n=len(names);num=j.ones((18,18),j.int32).at[5,3].set(0).at[2,1].set(2);den=j.ones_like(num)
    power=j.array([0,0,50,0,0,60,60,55,100,75]);accuracy=j.full(n,100)
    if real_accuracy:accuracy=accuracy.at[7:9].set(95)
    table=transition.MoveTable(power,j.array([0,1,0,1,1,7,6,5,10,2]),
        j.array([False,False,True,False,False,True,True,True,False,True]),accuracy,j.zeros(n,bool),
        j.array([0,0,0,3,4,0,0,0,0,0]),num,den,j.array(2),effects=effects.from_metadata(meta),
        secondary=secondary.from_metadata(meta),absorption=absorption.from_metadata(meta))
    return jax,j,meta,table


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class SecondaryMoveEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import secondary,transition
        from gpu_engine.state import empty_state
        cls.jax,cls.j,cls.meta,cls.table=components();cls.secondary=secondary;cls.tr=transition
        cls.empty=staticmethod(empty_state);cls.step=staticmethod(cls.jax.jit(transition.step_with_chance))

    def fixture(self,case,batch=1):
        j=self.j;side=case['side'];foe=1-side;names=self.meta['vocabulary']['moves'];abilities=self.meta['vocabulary']['abilities'];s=self.empty(batch,seed=157)
        s=s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),
            stats=j.full_like(s.stats,200).at[...,0].set(1000).at[:,side,:,5].set(301),
            level=j.full_like(s.level,100),types=s.types.at[...,0].set(1),
            species=j.broadcast_to(j.arange(1,7),(batch,2,6)),moves=j.ones_like(s.moves),pp=j.full_like(s.pp,8),done=j.zeros(batch,bool))
        target_move='protect' if case.get('protect') else 'endure' if case.get('endure') else 'splash'
        s=s._replace(moves=s.moves.at[:,side,0,0].set(names.index(case['move'])).at[:,foe,0,0].set(names.index(target_move)),
            ability=s.ability.at[:,side,0].set(abilities.index(case.get('source','_none'))).at[:,foe,0].set(abilities.index(case.get('target','_none'))),
            base_ability=s.base_ability.at[:,side,0].set(abilities.index(case.get('source','_none'))).at[:,foe,0].set(abilities.index(case.get('target','_none'))),
            types=s.types.at[:,foe,0,0].set(self.meta['vocabulary']['types'].index(case.get('type','Normal').lower())),
            boosts=s.boosts.at[:,side,0,:5].set(case.get('stage',0)).at[:,foe,0,4].set(case.get('targetStage',0)),
            hp=s.hp.at[:,foe,0].set(1 if case.get('ko') or case.get('endure') else 1000),
            substitute_hp=s.substitute_hp.at[:,foe,0].set(case.get('sub',0)))
        draws=j.zeros((batch,2,12),j.int32).at[:,:,1:4].set(99).at[:,:,4:7].set(15)
        chance=self.tr.ChanceInputs(draws,j.zeros((batch,2),j.int32),j.full((batch,2),case.get('roll',0)),
            j.zeros((batch,2),j.int32),j.zeros((batch,2),j.int32),j.zeros(batch,j.int32))
        table=self.table
        if case.get('miss'):table=table._replace(accuracy=table.accuracy.at[names.index(case['move'])].set(0))
        return s,j.zeros((batch,2),j.int32),table,chance

    @staticmethod
    def expected(case):
        move=case['move'];self_effect=move in ('silverwind','ancientpower');chance=10 if self_effect else 100 if move=='mudshot' else 50
        proc=case.get('roll',0)<chance*(2 if case.get('source')=='serenegrace' else 1)
        hit=not any(case.get(k) for k in ('miss','protect')) and not(move=='mudshot' and case.get('type')=='Flying')
        own=[min(6,case.get('stage',0)+1)]*5 if self_effect and hit and proc else [case.get('stage',0)]*5
        target=hit and proc and not any(case.get(k) for k in ('sub','ko')) and case.get('target')!='shielddust'
        speed=max(-6,case.get('targetStage',0)-1) if target and move=='mudshot' and case.get('target') not in ('clearbody','whitesmoke') else case.get('targetStage',0)
        status='brn' if target and move=='sacredfire' and case.get('target')!='waterveil' and case.get('type')!='Fire' else ''
        return own,speed,status

    def test_all_hundred_rolls_and_grace_use_the_existing_secondary_coordinate(self):
        j=self.j
        for name in MOVES:
            for grace in (False,True):
                for side in (0,1):
                    s,a,table,chance=self.fixture(dict(move=name,side=side,source='serenegrace' if grace else '_none'),100)
                    roll=j.arange(100);r=self.secondary.apply(s,side,j.full(100,self.meta['vocabulary']['moves'].index(name)),roll,table.secondary,eligible=True)
                    changed=j.any(r.boosts[:,side,0,:5]!=s.boosts[:,side,0,:5],axis=1)|j.any(r.boosts[:,1-side,0]!=s.boosts[:,1-side,0],axis=1)|(r.status[:,1-side,0]!=s.status[:,1-side,0])
                    threshold=(10 if name in MOVES[:2] else 100 if name=='mudshot' else 50)*(2 if grace else 1)
                    self.assertEqual(int(changed.sum()),min(100,threshold),(name,side,grace))

    def test_full_step_effect_boundaries_both_seats(self):
        for case in CASES:
            with self.subTest(case=case):
                s,a,t,chance=self.fixture(case);out=self.step(s,a,t,chance);self.assertFalse(bool(out.invalid_actions[0]));r=out.state
                own,speed,status=self.expected(case);side=case['side'];foe=1-side
                self.assertEqual(r.boosts[0,side,0,:5].tolist(),own)
                self.assertEqual(int(r.boosts[0,foe,0,4]),speed)
                self.assertEqual(int(r.status[0,foe,0]),2 if status=='brn' else 0)
                self.assertEqual(int(r.pp[0,side,0,0]),7)

    def test_invalid_terminal_and_ignored_replacement_requests_cannot_proc(self):
        for side in (0,1):
            s,a,t,chance=self.fixture(dict(move='silverwind',side=side));j=self.j
            for state,actions in ((s,a.at[0,side].set(9)),(s._replace(done=j.ones_like(s.done)),a)):
                r=self.step(state,actions,t,chance).state
                for x,y in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(r)):self.assertTrue(bool(j.array_equal(x,y)))
            state=s._replace(hp=s.hp.at[0,1-side,0].set(0));r=self.step(state,a.at[0,1-side].set(4),t,chance).state
            self.assertEqual(r.boosts[0,side,0].tolist(),s.boosts[0,side,0].tolist())

    def test_jit_batch_prefix_keeps_accuracy_evasion_and_bench_boosts(self):
        j=self.j;fixtures=[self.fixture(dict(move='silverwind',side=0,sub=1)),
            self.fixture(dict(move='ancientpower',side=1,roll=10))]
        state=self.jax.tree.map(lambda *v:j.concatenate(v),*(v[0] for v in fixtures))
        state=state._replace(boosts=state.boosts.at[...,5].set(-2).at[...,6].set(3)
            .at[:,:,1:,0].set(4))
        actions=j.concatenate([v[1] for v in fixtures]);chance=self.jax.tree.map(lambda *v:j.concatenate(v),*(v[3] for v in fixtures))
        out=self.step(state,actions,self.table,chance).state
        self.assertTrue(bool(j.array_equal(out.boosts[...,5:],state.boosts[...,5:])))
        self.assertTrue(bool(j.array_equal(out.boosts[:,:,1:],state.boosts[:,:,1:])))
        for i in range(2):
            one=self.step(self.jax.tree.map(lambda x:x[i:i+1],state),actions[i:i+1],self.table,
                self.jax.tree.map(lambda x:x[i:i+1],chance)).state
            for x,y in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(one)):
                self.assertTrue(bool(j.array_equal(x[i:i+1],y)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown and Node')
    def test_pinned_source_catalog_and_both_seat_boundaries(self):
        run=subprocess.run([NODE,str(Path(__file__).with_name('gpu_secondary_move_effects_oracle.cjs')),SHOWDOWN],input=json.dumps(CASES),text=True,capture_output=True)
        self.assertEqual(run.returncode,0,run.stderr);data=json.loads(run.stdout);self.assertEqual(len(data['cases']),len(CASES))
        omni={'atk':1,'def':1,'spa':1,'spd':1,'spe':1}
        self.assertEqual(data['catalog'],[
            {'id':'silverwind','secondaries':[{'chance':10,'self':{'boosts':omni}}]},
            {'id':'ancientpower','secondaries':[{'chance':10,'self':{'boosts':omni}}]},
            {'id':'mudshot','secondaries':[{'chance':100,'boosts':{'spe':-1}}]},
            {'id':'sacredfire','secondaries':[{'chance':50,'status':'brn'}]}])
        for case,row in zip(CASES,data['cases'],strict=True):
            own,speed,status=self.expected(case)
            self.assertEqual([row['ownBoosts'][k] for k in ('atk','def','spa','spd','spe')],own,case)
            self.assertEqual(row['foeBoosts']['spe'],speed,case);self.assertEqual(row['foeStatus'],status,case)
