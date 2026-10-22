"""Solar Beam charge/weather component; no complete move/game qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated spike JAX')
class SolarBeamTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,mechanics,pressure,solar_beam,transition,weather
        from gpu_engine.state import empty_state
        self.jax,self.j,self.t,self.c,self.m,self.w,self.empty=jax,j,transition,solar_beam,mechanics,weather,empty_state
        self.names=['_none','solarbeam','splash','protect','struggle']
        types=['_none','normal','fighting','flying','poison','ground','rock','bug','ghost','steel','fire','water','grass','electric','psychic','ice','dragon','dark']
        meta={'vocabulary':{'moves':self.names,'types':types,'abilities':['_none','cloudnine','pressure']},
            'moves':[None]+[{'flags':{'charge':name=='solarbeam','nosleeptalk':name=='solarbeam',
                'protect':name=='solarbeam'}} for name in self.names[1:]]}
        self.table=transition.MoveTable(j.array([0,120,0,0,50]),j.array([0,12,1,1,0]),
            j.array([False,False,False,False,True]),j.full(5,100),j.ones(5,bool),j.array([0,0,0,3,0]),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(4),
            effects=effects.from_metadata(meta),weather=weather.from_metadata(meta),
            pressure=pressure.from_metadata(meta),solar_beam=solar_beam.from_metadata(meta))

    def live(self,batch=1,weather=0):
        j=self.j;s=self.empty(batch,seed=719)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            level=j.full_like(s.level,100),hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            types=s.types.at[...,0].set(14),moves=j.full_like(s.moves,2).at[:,0,0,0].set(1),
            pp=j.full_like(s.pp,8),weather=j.full_like(s.weather,weather),done=j.zeros_like(s.done))

    def play(self,s,actions=None):
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else actions,self.table)

    def test_clear_charges_then_releases_without_second_pp_cost(self):
        a=self.play(self.live()).state
        self.assertEqual((int(a.hp[0,1,0]),int(a.pp[0,0,0,0]),int(a.charge_move[0,0,0]),int(a.charge_counter[0,0,0])),(1000,7,1,1))
        b=self.play(a).state
        self.assertLess(int(b.hp[0,1,0]),1000)
        self.assertEqual((int(b.pp[0,0,0,0]),int(b.charge_move[0,0,0])),(7,0))

    def test_sun_skips_charge_and_live_suppressor_restores_charge(self):
        a=self.play(self.live(weather=self.w.SUN)).state
        self.assertLess(int(a.hp[0,1,0]),1000);self.assertEqual(int(a.charge_move[0,0,0]),0)
        s=self.live(weather=self.w.SUN);s=s._replace(ability=s.ability.at[0,1,0].set(1))
        a=self.play(s).state;self.assertEqual((int(a.hp[0,1,0]),int(a.charge_move[0,0,0])),(1000,1))

    def test_release_request_ignores_zero_pp_disable_and_other_locks(self):
        s=self.live();s=s._replace(pp=s.pp.at[0,0,0,0].set(1))
        a=self.play(s).state
        a=a._replace(disabled=a.disabled.at[0,0,0,0].set(True),choice_move=a.choice_move.at[0,0,0].set(2),
            encore_move=a.encore_move.at[0,0,0].set(2),encore_counter=a.encore_counter.at[0,0,0].set(3))
        mask=self.t.action_mask(a,self.table)
        self.assertEqual(mask[0,0].tolist(),[True,False,False,False,False,False,False,False,False])
        # The request is hard locked; Encore's dispatch override is a separate event.
        a=a._replace(encore_counter=self.j.zeros_like(a.encore_counter))
        b=self.play(a).state
        self.assertLess(int(b.hp[0,1,0]),1000);self.assertEqual(int(b.pp[0,0,0,0]),0)

    def test_weather_base_power_and_release_use_current_weather(self):
        j=self.j
        for weather in (self.w.NONE,self.w.SUN,self.w.RAIN,self.w.SAND,self.w.HAIL):
            power=self.c.base_power(self.table.solar_beam,j.array([1]),j.array([120]),j.array([weather]))
            self.assertEqual(int(power[0]),60 if weather in (self.w.RAIN,self.w.SAND,self.w.HAIL) else 120)
        a=self.play(self.live()).state
        rain=a._replace(weather=j.full_like(a.weather,self.w.RAIN))
        clear=self.play(a).state;wet=self.play(rain).state
        self.assertGreater(int(wet.hp[0,1,0]),int(clear.hp[0,1,0]))

    def test_sun_expiry_charges_on_the_next_ordinary_turn(self):
        s=self.live(weather=self.w.SUN);s=s._replace(weather_counter=self.j.ones_like(s.weather_counter))
        a=self.play(s).state;hp=int(a.hp[0,1,0]);self.assertLess(hp,1000)
        b=self.play(a).state;self.assertEqual((int(b.hp[0,1,0]),int(b.charge_move[0,0,0])),(hp,1))

    def test_before_move_sleep_aborts_charge_without_pp(self):
        a=self.play(self.live()).state
        a=a._replace(status=a.status.at[0,0,0].set(self.m.SLP),status_counter=a.status_counter.at[0,0,0].set(5))
        b=self.play(a).state
        self.assertEqual((int(b.hp[0,1,0]),int(b.pp[0,0,0,0]),int(b.charge_move[0,0,0])),(1000,7,0))

    def test_protect_does_not_block_preparation_but_blocks_release(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,1,0,0].set(3))
        a=self.play(s).state;self.assertEqual(int(a.charge_move[0,0,0]),1)
        # Reset the opponent's stall counter to guarantee another Protect.
        a=a._replace(protect_counter=self.j.zeros_like(a.protect_counter))
        b=self.play(a).state;self.assertEqual((int(b.hp[0,1,0]),int(b.charge_move[0,0,0])),(1000,0))

    def test_pressure_only_costs_pp_on_the_preparation(self):
        s=self.live();s=s._replace(ability=s.ability.at[0,1,0].set(2))
        a=self.play(s).state;b=self.play(a).state
        self.assertEqual((int(a.pp[0,0,0,0]),int(b.pp[0,0,0,0])),(6,6))

    def test_charge_clock_ignores_pseudo_requests_and_departure_clears(self):
        from gpu_engine import entry,baton
        j=self.j;a=self.play(self.live()).state
        held=self.c.residual(a,eligible=j.array([False]))
        self.assertEqual(int(held.charge_counter[0,0,0]),1)
        copied=baton.copy_to_recipient(a,j.array([0]),j.array([1]),eligible=j.array([True]))
        self.assertEqual(int(copied.charge_move[0,0,1]),0)
        departed=entry.depart(a,j.array([0]),None,eligible=j.array([True]))
        self.assertEqual((int(departed.charge_move[0,0,0]),int(departed.charge_counter[0,0,0])),(0,0))

    def test_native_damage_arithmetic_for_clear_and_halved_rain_power(self):
        j=self.j
        for power,expected in ((120,102),(60,52)):
            value=self.m.gen3_damage(level=j.array([100]),power=j.array([power]),attack=j.array([200]),
                defense=j.array([200]),physical=False,roll=j.array([100]))
            self.assertEqual(int(value[0]),expected)

    def test_invalid_done_prefix_and_jit(self):
        j=self.j;s=self.live(2);compiled=self.jax.jit(self.t.step)
        a=compiled(s,j.zeros((2,2),j.int32),self.table);b=compiled(a.state,j.zeros((2,2),j.int32),self.table)
        self.assertTrue(bool(j.all(b.state.hp[:,1,0]<1000)))
        one=self.play(self.live()).state
        self.assertEqual(int(a.state.charge_move[0,0,0]),int(one.charge_move[0,0,0]))
        bad=self.play(a.state,j.full((2,2),8,j.int32));self.assertTrue(bool(j.all(bad.invalid_actions)))
        for old,new in zip(a.state,bad.state):self.assertTrue(bool(j.array_equal(old,new)))
        done=s._replace(done=j.ones_like(s.done));out=self.play(done).state
        for old,new in zip(done,out):self.assertTrue(bool(j.array_equal(old,new)))

    @unittest.skipUnless((os.environ.get('NODE') or shutil.which('node')) and os.environ.get('POKEZERO_SHOWDOWN_ROOT'),'requires pinned Showdown checkout')
    def test_pinned_native_charge_pp_weather_boundaries(self):
        root=os.environ['POKEZERO_SHOWDOWN_ROOT'];node=os.environ.get('NODE') or shutil.which('node')
        actual=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()
        self.assertEqual(actual,'f76228a1354b5d0f307ca2d16101294ad3a2308b')
        result=json.loads(subprocess.check_output([node,str(Path(__file__).with_name('gpu_solar_beam_oracle.cjs')),root],text=True))
        for name in ('clear','rain','sand','hail','cloud','sleep','paralysis','onePP','protect','pressure'):
            self.assertTrue(result[name]['first']['charge'],name)
            self.assertEqual(result[name]['first']['locked'],'solarbeam',name)
        self.assertLess(result['clear']['second']['hp'],1000)
        self.assertEqual((result['clear']['second']['hp'],result['rain']['second']['hp']),(898,948))
        self.assertEqual((result['clear']['first']['pp'],result['clear']['second']['pp']),(15,15))
        self.assertEqual(result['onePP']['second']['pp'],0)
        self.assertFalse(result['sun']['first']['charge'])
        self.assertTrue(result['expire']['second']['charge'])
        for name in ('sleep','paralysis','protect'):
            self.assertEqual(result[name]['second']['hp'],1000,name)
            self.assertFalse(result[name]['second']['charge'],name)
        self.assertEqual(result['pressure']['first']['pp'],14)
