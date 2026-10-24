"""Observed Overheat omission: authored hit-dependent self drops, not fidelity credit."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get('NODE') or shutil.which('node')
SHOWDOWN=os.environ.get('POKEZERO_SHOWDOWN_ROOT')
CASES=[dict(move=move,**case) for move in ('overheat','psychoboost') for case in (
    {},dict(stage=-5),dict(stage=-6),dict(stage=6),dict(sub=999),dict(sub=1),
    dict(ko=True),dict(endure=True),dict(miss=True),dict(protect=True))]
CASES += [dict(move='psychoboost',dark=True),dict(move='overheat',absorb=True)]


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PostHitSelfDropTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import absorption,effects,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.t,self.empty=jax,j,transition,empty_state
        self.names=['_none','splash','overheat','psychoboost','protect','struggle','endure']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
               'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta=dict(vocabulary=dict(moves=self.names,types=types,abilities=['_none','flashfire']),
                  moves=[None]+[dict(source_type='Fire' if n=='overheat' else 'Psychic' if n=='psychoboost' else 'Normal',
                                    category='Special' if n in ('overheat','psychoboost') else 'Status',
                                    flags={'protect':1} if n in ('overheat','psychoboost') else {})
                                for n in self.names[1:]])
        num=j.ones((18,18),j.int32).at[14,17].set(0)
        self.table=transition.MoveTable(j.array([0,0,140,140,0,50,0]),j.array([0,1,10,14,1,0,1]),
            j.array([False,False,False,False,False,True,False]),j.full(7,100),j.ones(7,bool),
            j.array([0,0,0,0,3,0,3]),num,j.ones_like(num),j.array(5),
            effects=effects.from_metadata(meta),absorption=absorption.from_metadata(meta))

    @staticmethod
    def expected(case):
        stage=case.get('stage',0)
        blocked=any(case.get(k) for k in ('miss','protect','dark','absorb'))
        return stage if blocked else max(-6,stage-2)

    def fixture(self,case,side):
        j=self.j;foe=1-side;s=self.empty(1,seed=101)
        s=s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.max_hp,1000),
            level=j.full_like(s.level,100),stats=j.full_like(s.stats,200),
            types=s.types.at[...,0].set(1),moves=j.ones_like(s.moves),pp=j.full_like(s.pp,8),
            done=j.zeros_like(s.done))
        s=s._replace(moves=s.moves.at[0,side,0,0].set(self.names.index(case['move']))
                .at[0,foe,0,0].set(4 if case.get('protect') else 6 if case.get('endure') else 1),
            boosts=s.boosts.at[0,side,0,2].set(case.get('stage',0)),
            stats=s.stats.at[0,side,0,5].set(301),
            types=s.types.at[0,foe,0,0].set(17 if case.get('dark') else 1),
            ability=s.ability.at[0,foe,0].set(1 if case.get('absorb') else 0),
            substitute_hp=s.substitute_hp.at[0,foe,0].set(case.get('sub',0)),
            hp=s.hp.at[0,foe,0].set(1 if case.get('ko') or case.get('endure') else 1000))
        table=self.table
        if case.get('miss'):
            move=self.names.index(case['move'])
            table=table._replace(accuracy=table.accuracy.at[move].set(0),
                                 always_hit=table.always_hit.at[move].set(False))
        return s,j.zeros((1,2),j.int32),table

    def test_hits_substitutes_kos_caps_and_failure_gates_both_seats(self):
        for case in CASES:
            for side in (0,1):
                with self.subTest(case=case,side=side):
                    s,a,table=self.fixture(case,side)
                    out=self.jax.jit(self.t.step)(s,a,table)
                    self.assertFalse(bool(out.invalid_actions[0]))
                    self.assertEqual(int(out.state.boosts[0,side,0,2]),self.expected(case))
                    self.assertEqual(int(out.state.pp[0,side,0,0]),7)
                    self.assertEqual(out.state.boosts[0,side,1:].tolist(),s.boosts[0,side,1:].tolist())
                    if case.get('sub'):
                        self.assertLess(int(out.state.substitute_hp[0,1-side,0]),case['sub'])
                        self.assertEqual(int(out.state.hp[0,1-side,0]),1000)
                    elif case.get('endure'):
                        self.assertEqual(int(out.state.hp[0,1-side,0]),1)
                    elif not any(case.get(k) for k in ('miss','protect','dark','absorb')):
                        self.assertLess(int(out.state.hp[0,1-side,0]),int(s.hp[0,1-side,0]))
                    else:
                        self.assertEqual(int(out.state.hp[0,1-side,0]),1000)

    def test_jit_batch_prefix_parity_unrelated_boosts_and_nonselected_moves(self):
        fixtures=[self.fixture(dict(move='overheat',stage=-5),0),
                  self.fixture(dict(move='psychoboost',sub=1),1)]
        state=self.jax.tree.map(lambda *v:self.j.concatenate(v),*(s for s,a,t in fixtures))
        state=state._replace(boosts=state.boosts.at[:,:,0,0].set(3))
        actions=self.j.concatenate([a for s,a,t in fixtures])
        out=self.jax.jit(self.t.step)(state,actions,self.table)
        self.jax.block_until_ready(out)
        for i in range(2):
            single=self.t.step(self.jax.tree.map(lambda x:x[i:i+1],state),actions[i:i+1],self.table)
            for x,y in zip(self.jax.tree.leaves(out),self.jax.tree.leaves(single)):
                self.assertTrue(bool(self.j.array_equal(x[i:i+1],y)))
        self.assertEqual(out.state.boosts[:,:,0,0].tolist(),[[3,3],[3,3]])
        self.assertEqual(self.table.effects.hit_spa_drop.tolist(),[False,False,True,True,False,False,False])

    def test_invalid_terminal_and_replacement_do_not_apply(self):
        j=self.j;s,a,table=self.fixture(dict(move='overheat'),0)
        for state,actions in ((s,a.at[0,0].set(9)),(s._replace(done=j.ones_like(s.done)),a)):
            out=self.t.step(state,actions,table)
            for x,y in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(out.state)):
                self.assertTrue(bool(j.array_equal(x,y)))
        state=s._replace(hp=s.hp.at[0,1,0].set(0))
        out=self.t.step(state,a.at[0,1].set(4),table).state
        self.assertEqual(int(out.boosts[0,0,0,2]),0)
        self.assertEqual(int(out.pp[0,0,0,0]),8)

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Gen3 Showdown')
    def test_pinned_gen3_oracle(self):
        specs=[dict(case,side=side) for case in CASES for side in (0,1)]
        run=subprocess.run([NODE,str(Path(__file__).with_name('gpu_post_hit_self_drop_oracle.cjs')),SHOWDOWN],
            input=json.dumps(specs),text=True,capture_output=True)
        self.assertEqual(run.returncode,0,run.stderr)
        rows=json.loads(run.stdout);self.assertEqual(len(rows),len(specs))
        for case,row in zip(specs,rows,strict=True):
            self.assertEqual(row['stage'],self.expected(case),case)
            self.assertEqual(row['pp'],7,case)
            if case.get('sub'):
                self.assertLess(row['sub'],case['sub'],case)
                self.assertFalse(row['damaged'],case)
            elif case.get('endure'):
                self.assertEqual(row['hp'],1,case)
                self.assertFalse(row['damaged'],case)
            elif not any(case.get(k) for k in ('miss','protect','dark','absorb')):
                self.assertTrue(row['damaged'],case)
            else:self.assertFalse(row['damaged'],case)


if __name__=='__main__':unittest.main()
