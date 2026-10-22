"""Gen 3 weather healing components, not game/coverage qualification."""
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
class WeatherHealingTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,transition,weather
        from gpu_engine.state import empty_state
        self.jax,self.j,self.e,self.t,self.w,self.empty=jax,j,effects,transition,weather,empty_state
        self.names=['_none','splash','struggle','synthesis','moonlight','morningsun','sunnyday','raindance']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost',
            'steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta={'vocabulary':{'moves':self.names,'types':types,
            'abilities':['_none','airlock','cloudnine','sandveil']}}
        n=len(self.names)
        self.table=transition.MoveTable(j.zeros(n,j.int32),j.ones(n,j.int32),j.zeros(n,bool),
            j.full(n,100),j.ones(n,bool),j.zeros(n,j.int32),j.ones((18,18),j.int32),
            j.ones((18,18),j.int32),j.array(2),effects=effects.from_metadata(meta),weather=weather.from_metadata(meta))

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=197)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            hp=j.full_like(s.hp,101).at[:,0,0].set(1),max_hp=j.full_like(s.hp,101),
            moves=j.full_like(s.moves,1).at[:,0,0,0].set(3),pp=j.full_like(s.pp,8),
            types=s.types.at[...,0].set(6).at[...,1].set(15),done=j.zeros_like(s.done))

    def play(self,s):return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),self.table)

    def test_exact_fractions_and_minimum_not_modern_q12_rounding(self):
        j=self.j;hp=j.array([1,2,3,5,101,301])
        self.assertEqual(self.e.weather_healing(hp,j.full(6,2)).tolist(),[1,1,2,3,67,200])
        self.assertEqual(self.e.weather_healing(hp,j.full(6,1)).tolist(),[1,1,1,1,25,75])
        self.assertEqual(self.e.weather_healing(hp,j.zeros(6,j.int32)).tolist(),[1,1,1,2,50,150])

    def test_three_moves_all_five_weather_states(self):
        j=self.j;s=self.live(15)
        s=s._replace(moves=s.moves.at[:,0,0,0].set(j.repeat(j.array([3,4,5]),5)),
            weather=j.tile(j.arange(5),3))
        r=self.play(s).state
        self.assertEqual(r.hp[:,0,0].tolist(),[51,26,68,26,26]*3)
        self.assertEqual(r.pp[:,0,0,0].tolist(),[7]*15)

    def test_suppressors_on_either_live_active_but_not_fainted_holder(self):
        j=self.j;s=self.live(5);s=s._replace(weather=j.full(5,2),
            ability=s.ability.at[0,0,0].set(1).at[1,1,0].set(2).at[2,1,1].set(1)
                .at[3,1,0].set(1).at[4,0,0].set(2),
            hp=s.hp.at[3,1,0].set(0),forced_switch=s.forced_switch.at[3,1].set(True))
        # Direct healing weather surface: bench and fainted holders cannot suppress.
        rows=j.arange(5)[:,None];sides=j.arange(2)[None,:]
        effective=self.w.effective(self.table.weather,s.weather,s.ability[rows,sides,s.active],s.hp[rows,sides,s.active])
        self.assertEqual(effective.tolist(),[0,0,2,2,0])
        r=self.play(s).state
        self.assertEqual(r.hp[j.array([0,1,2,4]),0,0].tolist(),[51,51,68,51])

    def test_heal_precedes_weather_expiry_and_later_weather_move(self):
        j=self.j;s=self.live();s=s._replace(weather=j.array([2]),weather_counter=j.array([1]))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.weather[0])),(68,0))
        s=self.live()._replace(moves=self.live().moves.at[0,1,0,0].set(6))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.weather[0])),(51,2))
        s=s._replace(stats=s.stats.at[0,0,:,5].set(50))
        self.assertEqual(int(self.play(s).state.hp[0,0,0]),68)

    def test_full_hp_attempt_spends_pp_and_sleep_prevents_healing(self):
        s=self.live();j=self.j
        s=s._replace(hp=s.hp.at[0,0,0].set(101))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.pp[0,0,0,0])),(101,7))
        s=self.live();s=s._replace(status=s.status.at[0,0,0].set(5),status_counter=s.status_counter.at[0,0,0].set(3))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.pp[0,0,0,0])),(1,8))

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(2);a=j.zeros((2,2),j.int32)
        eager=self.t.step(s,a,self.table);compiled=self.jax.jit(self.t.step)(s,a,self.table)
        for x,y in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(j.array_equal(x,y)))
        one=self.jax.tree.map(lambda x:x[:1],s);single=self.t.step(one,a[:1],self.table)
        for x,y in zip(self.jax.tree.leaves(single),self.jax.tree.leaves(eager)):
            self.assertTrue(bool(j.array_equal(x,y[:1])))
        s=s._replace(done=s.done.at[1].set(True));a=a.at[0,0].set(99)
        r=self.t.step(s,a,self.table)
        for x,y in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r.state)):
            self.assertTrue(bool(j.array_equal(x,y)))

    @unittest.skipUnless(NODE and SHOWDOWN,'requires pinned Showdown')
    def test_pinned_gen3_oracle(self):
        cases=json.loads(subprocess.check_output([NODE,str(Path(__file__).with_name('gpu_weather_healing_oracle.cjs')),SHOWDOWN],text=True))
        ids={'':0,'raindance':1,'sunnyday':2,'sandstorm':3,'hail':4}
        for c in cases:
            weather=ids[c['weather']] if c['ability']=='Run Away' else 0
            amount=int(self.e.weather_healing(self.j.array(c['maxhp']),self.j.array(weather)))
            self.assertEqual(c['hp'],min(c['maxhp'],c['start']+amount),c)
            self.assertEqual(c['pp'],7,c)
