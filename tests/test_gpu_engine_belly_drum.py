"""Measured Belly Drum omission: authored Gen3 repair, not coverage credit."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')
CASES=[
    dict(hp=261,maxhp=261,atk=0,after_hp=131,after_atk=6),
    dict(hp=261,maxhp=261,atk=-6,after_hp=131,after_atk=6),
    dict(hp=261,maxhp=261,atk=5,after_hp=131,after_atk=6),
    dict(hp=261,maxhp=261,atk=6,after_hp=261,after_atk=6),
    dict(hp=131,maxhp=261,atk=0,after_hp=1,after_atk=6),
    dict(hp=130,maxhp=261,atk=0,after_hp=130,after_atk=0),
    dict(hp=130,maxhp=260,atk=-1,after_hp=130,after_atk=-1),
    dict(hp=131,maxhp=260,atk=0,after_hp=1,after_atk=6),
    dict(hp=1,maxhp=1,atk=0,after_hp=1,after_atk=0),
    dict(hp=2,maxhp=2,atk=0,after_hp=1,after_atk=6),
    dict(hp=2,maxhp=3,atk=-6,after_hp=1,after_atk=6),
    dict(hp=261,maxhp=261,atk=0,sub=40,after_hp=131,after_atk=6),
    dict(hp=261,maxhp=261,atk=0,protect=True,after_hp=131,after_atk=6),
    dict(hp=200,maxhp=261,atk=0,prior_damage=True,after_hp=100,after_atk=0),
    dict(hp=250,maxhp=261,atk=0,prior_damage=True,after_hp=20,after_atk=6),
]


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class BellyDrumTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import damage_moves,effects,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.t,self.empty=jax,j,transition,empty_state
        names=['_none','splash','bellydrum','struggle','protect','seismictoss']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
               'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta=dict(vocabulary=dict(moves=names,types=types,abilities=['_none']),
                  moves=[None]+[dict(flags={'protect':1} if n=='seismictoss' else {}) for n in names[1:]])
        self.table=transition.MoveTable(j.array([0,0,0,50,0,0]),j.array([0,1,1,0,1,2]),
            j.array([False,False,False,True,False,True]),j.full(6,100),j.ones(6,bool),
            j.array([0,0,0,0,3,0]),j.ones((18,18),j.int32),j.ones((18,18),j.int32),
            j.array(3),effects=effects.from_metadata(meta),damage_moves=damage_moves.from_metadata(meta))

    def fixture(self,case,side=0):
        j=self.j;foe=1-side;s=self.empty(1,seed=117)
        s=s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),
            level=j.full_like(s.level,100),stats=j.full_like(s.stats,200),
            types=s.types.at[...,0].set(1),moves=j.ones_like(s.moves),pp=j.full_like(s.pp,8),
            done=j.zeros_like(s.done))
        s=s._replace(hp=s.hp.at[0,side,0].set(case['hp']),
            max_hp=s.max_hp.at[0,side,0].set(case['maxhp']),
            boosts=s.boosts.at[0,side,0,0].set(case['atk']),
            stats=s.stats.at[0,side,0,5].set(199 if case.get('prior_damage') else 301),
            substitute_hp=s.substitute_hp.at[0,side,0].set(case.get('sub',0)),
            moves=s.moves.at[0,side,0,0].set(2).at[0,foe,0,0].set(
                4 if case.get('protect') else 5 if case.get('prior_damage') else 1))
        return s,j.zeros((1,2),j.int32)

    def test_hp_cost_attack_cap_threshold_and_live_hp_in_both_seats(self):
        for case in CASES:
            for side in (0,1):
                with self.subTest(case=case,side=side):
                    s,a=self.fixture(case,side);result=self.t.step(s,a,self.table)
                    self.assertFalse(bool(result.invalid_actions[0]))
                    out=result.state
                    self.assertEqual(int(out.hp[0,side,0]),case['after_hp'])
                    self.assertEqual(int(out.boosts[0,side,0,0]),case['after_atk'])
                    self.assertEqual(int(out.pp[0,side,0,0]),7)
                    self.assertEqual(int(out.substitute_hp[0,side,0]),case.get('sub',0))
                    self.assertEqual(out.hp[0,side,1:].tolist(),s.hp[0,side,1:].tolist())
                    self.assertEqual(out.hp[0,1-side].tolist(),s.hp[0,1-side].tolist())

    def test_jit_batch_singleton_parity_and_unrelated_boosts(self):
        fixtures=[self.fixture(c,side) for side,c in enumerate((CASES[0],CASES[13]))]
        fixtures[0]=(fixtures[0][0]._replace(boosts=fixtures[0][0].boosts.at[0,0,0,1:].set(
            self.j.array([1,-2,3,0,-1,2]))),fixtures[0][1])
        state=self.jax.tree.map(lambda *v:self.j.concatenate(v,axis=0),*(s for s,a in fixtures))
        actions=self.j.concatenate([a for s,a in fixtures],axis=0)
        batch=self.jax.jit(self.t.step)(state,actions,self.table)
        self.jax.block_until_ready(batch)
        for i,(s,a) in enumerate(fixtures):
            single=self.t.step(s,a,self.table)
            for x,y in zip(self.jax.tree.leaves(batch),self.jax.tree.leaves(single)):
                self.assertTrue(bool(self.j.array_equal(x[i:i+1],y)))
        self.assertEqual(batch.state.boosts[0,0,0,1:].tolist(),[1,-2,3,0,-1,2])
        self.assertEqual(int(batch.state.hp[0,0,0]),131)
        self.assertEqual(int(batch.state.boosts[0,0,0,0]),6)

    def test_invalid_done_and_forced_replacement_do_not_drum(self):
        s,a=self.fixture(CASES[0]);j=self.j
        for state,actions in ((s,a.at[0,0].set(9)),(s._replace(done=j.ones_like(s.done)),a)):
            out=self.t.step(state,actions,self.table)
            for x,y in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(out.state)):
                self.assertTrue(bool(j.array_equal(x,y)))
        s=s._replace(hp=s.hp.at[0,1,0].set(0))
        out=self.t.step(s,a.at[0,1].set(4),self.table).state
        self.assertEqual(int(out.hp[0,0,0]),261)
        self.assertEqual(int(out.boosts[0,0,0,0]),0)
        self.assertEqual(int(out.pp[0,0,0,0]),8)

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Gen3 Showdown')
    def test_independent_pinned_gen3_battles(self):
        specs=[dict(c,side=side) for c in CASES for side in (0,1)]
        run=subprocess.run([NODE,str(Path(__file__).with_name('gpu_belly_drum_oracle.cjs')),SHOWDOWN],
            input=json.dumps(specs),text=True,capture_output=True,check=True)
        rows=json.loads(run.stdout);self.assertEqual(len(rows),len(specs))
        for spec,row in zip(specs,rows,strict=True):
            self.assertEqual(row,dict(hp=spec['after_hp'],atk=spec['after_atk'],pp=7,sub=spec.get('sub',0)),spec)


if __name__=='__main__':unittest.main()
