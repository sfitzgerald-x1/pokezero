"""Weather-move component fixtures, not full-game fidelity or move coverage."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class WeatherTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,items,transition,weather
        from gpu_engine.state import empty_state
        self.jax,self.j,self.t,self.w,self.empty = jax,j,transition,weather,empty_state
        self.names = ["_none","splash","struggle","raindance","sunnyday","sandstorm","hail",
            "surf","flamethrower","thunder"]
        types = ["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta = {"vocabulary":{"moves":self.names,"types":types,"abilities":["_none","airlock","cloudnine","sandveil"],
            "items":["_none","leftovers"],"species":["_none","ditto"]}}
        self.wt = weather.from_metadata(meta)
        n = len(self.names)
        self.table = transition.MoveTable(j.array([0,0,50,0,0,0,0,95,95,120]),
            j.array([0,1,0,11,10,6,15,11,10,13]),j.array([False,False,True]+[False]*(n-3)),
            j.array([100]*9+[70]),j.array([True]*9+[False]),j.zeros(n,j.int32),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2),
            effects.from_metadata(meta),items.from_metadata(meta),self.wt)

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,100),hp=j.full_like(s.hp,101),
            max_hp=j.full_like(s.hp,101),pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,1),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,own="splash",foe="splash"):
        s = s._replace(moves=s.moves.at[:,0,0,0].set(self.names.index(own)).at[:,1,0,0].set(self.names.index(foe)))
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32),self.table).state

    def test_different_move_weather_sets_five_and_same_weather_does_not_refresh(self):
        w,c = self.w.set_move(self.j.array([0,1,1,3,0]),self.j.array([0,2,0,0,0]),
            self.j.array([1,1,1,2,0]))
        self.assertEqual(w.tolist(),[1,1,1,2,0])
        self.assertEqual(c.tolist(),[5,2,0,5,0])

    def test_temporary_expires_before_upkeep_and_permanent_does_not_tick(self):
        w,c = self.w.residual_clock(self.j.array([3,3,3,3]),self.j.array([1,2,0,1]),
            eligible=self.j.array([True,True,True,False]))
        self.assertEqual((w.tolist(),c.tolist()),([0,3,3,3],[0,1,0,1]))

    def test_move_setting_and_five_turn_expiry_integration(self):
        r = self.play(self.live(),"sandstorm")
        self.assertEqual((int(r.weather[0]),int(r.weather_counter[0]),int(r.hp[0,0,0])),(3,4,95))
        for _ in range(3):
            r = self.play(r)
        self.assertEqual((int(r.weather_counter[0]),int(r.hp[0,0,0])),(1,77))
        r = self.play(r,"sandstorm")
        self.assertEqual((int(r.weather[0]),int(r.weather_counter[0]),int(r.hp[0,0,0])),(0,0,77))

    def test_all_four_weather_moves_and_permanent_counter(self):
        for name,value in (("raindance",1),("sunnyday",2),("sandstorm",3),("hail",4)):
            r = self.play(self.live(),name)
            self.assertEqual((int(r.weather[0]),int(r.weather_counter[0])),(value,4))
        s = self.live()._replace(weather=self.j.array([3]),weather_counter=self.j.array([0]))
        r = self.play(s,"sandstorm")
        self.assertEqual((int(r.weather[0]),int(r.weather_counter[0]),int(r.hp[0,0,0])),(3,0,95))

    def test_weather_move_affects_slower_same_turn_attack(self):
        s = self.live(32)
        s = s._replace(hp=self.j.full_like(s.hp,1000),max_hp=self.j.full_like(s.hp,1000))
        base = self.play(s,foe="surf")
        rain = self.play(s,"raindance","surf")
        sun = self.play(s,"sunnyday","surf")
        self.assertTrue(bool(self.j.all(rain.hp[:,0,0] < base.hp[:,0,0])))
        self.assertTrue(bool(self.j.all(sun.hp[:,0,0] > base.hp[:,0,0])))
        base = self.play(s,foe="flamethrower")
        rain = self.play(s,"raindance","flamethrower")
        sun = self.play(s,"sunnyday","flamethrower")
        self.assertTrue(bool(self.j.all(rain.hp[:,0,0] > base.hp[:,0,0])))
        self.assertTrue(bool(self.j.all(sun.hp[:,0,0] < base.hp[:,0,0])))

    def test_sand_hail_immunity_and_minimum_chip(self):
        j = self.j
        types = j.array([[6,0],[5,0],[9,0],[15,0],[1,0],[1,0]])
        hp = j.full((6,),101)
        sand = self.w.chip(self.wt,j.full((6,),3),types,j.array([0,0,0,0,3,0]),hp,hp)
        hail = self.w.chip(self.wt,j.full((6,),4),types,j.zeros(6,j.int32),hp,hp)
        self.assertEqual(sand.tolist(),[101,101,101,95,101,95])
        self.assertEqual(hail.tolist(),[95,95,95,101,95,95])
        self.assertEqual(int(self.w.chip(self.wt,j.array(3),j.array([1,0]),j.array(0),j.array(1),j.array(2))),0)

    def test_airlock_cloudnine_suppress_but_do_not_erase_or_pause_timer(self):
        for ability in (1,2):
            s = self.live()._replace(weather=self.j.array([3]),weather_counter=self.j.array([3]))
            s = s._replace(ability=s.ability.at[0,1,0].set(ability))
            r = self.play(s)
            self.assertEqual((int(r.weather[0]),int(r.weather_counter[0]),r.hp[0,:,0].tolist()),(3,2,[101,101]))
            r = self.t.step(r,self.j.array([[0,4]]),self.table).state
            self.assertEqual(r.hp[0,:,0].tolist(),[95,101])
            self.assertEqual(int(r.hp[0,1,1]),95)

    def test_suppressor_faint_restores_weather_and_bench_does_not_suppress(self):
        s = self.live()._replace(weather=self.j.array([3]))
        s = s._replace(ability=s.ability.at[0,1,0].set(1),hp=s.hp.at[0,1,0].set(1))
        r = self.play(s,"surf")
        self.assertEqual(int(r.hp[0,0,0]),95)
        s = self.live()._replace(weather=self.j.array([3]))
        s = s._replace(ability=s.ability.at[0,1,1].set(1))
        self.assertEqual(int(self.play(s).hp[0,0,0]),95)

    def test_thunder_rain_is_always_hit_and_sun_is_fifty_before_stages(self):
        j = self.j
        a,h = self.w.thunder_accuracy(self.wt,j.array([1,2,0,1]),j.array([9,9,9,7]),
            j.array([70,70,70,100]),j.zeros(4,bool))
        self.assertEqual((a.tolist(),h.tolist()),([70,50,70,100],[True,False,False,False]))
        s = self.live(64)._replace(weather=j.full((64,),1))
        s = s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            boosts=s.boosts.at[:,0,0,5].set(-6).at[:,1,0,6].set(6))
        r = self.play(s,"thunder")
        self.assertTrue(bool(j.all(r.hp[:,1,0] < 1000)))

    def test_weather_chip_precedes_leftovers_and_double_last_mon_ko_is_tie(self):
        s = self.live()._replace(weather=self.j.array([3]))
        s = s._replace(hp=s.hp.at[:,:,1:].set(0).at[:,:,0].set(1),item=s.item.at[:,:,0].set(1))
        r = self.play(s)
        self.assertTrue(bool(r.done[0]))
        self.assertEqual((int(r.winner[0]),r.hp[0,:,0].tolist()),(2,[0,0]))

    def test_forced_replacement_does_not_tick_or_chip(self):
        s = self.live()._replace(weather=self.j.array([3]),weather_counter=self.j.array([2]))
        s = s._replace(hp=s.hp.at[0,1,0].set(0))
        r = self.t.step(s,self.j.array([[0,4]]),self.table).state
        self.assertEqual((int(r.weather_counter[0]),int(r.hp[0,0,0]),int(r.hp[0,1,1])),(2,101,101))

    def test_invalid_and_terminal_states_remain_identical(self):
        for s,actions in ((self.live()._replace(done=self.j.array([True])),self.j.array([[0,0]])),
            (self.live(),self.j.array([[9,0]]))):
            s = s._replace(weather=self.j.array([3]),weather_counter=self.j.array([2]))
            r = self.t.step(s,actions,self.table).state
            for a,b in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r)):
                self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_jit_and_batch_invariant_weather_paths(self):
        s = self.live(2)._replace(weather=self.j.array([3,1]),weather_counter=self.j.array([2,0]))
        actions = self.j.zeros((2,2),self.j.int32)
        eager = self.t.step(s,actions,self.table)
        compiled = self.jax.jit(self.t.step)(s,actions,self.table)
        self.jax.block_until_ready(compiled)
        small = self.t.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for a,b,c in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled),self.jax.tree.leaves(small)):
            self.assertTrue(bool(self.j.array_equal(a,b)))
            self.assertTrue(bool(self.j.array_equal(a[:1],c)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned weather source")
    def test_weather_rules_match_pinned_source_handlers(self):
        helper = Path(__file__).with_name("gpu_weather_oracle.cjs")
        actual = json.loads(subprocess.run([NODE,str(helper),SHOWDOWN],text=True,capture_output=True,check=True).stdout)
        self.assertEqual(actual["orders"],[8,8,8,8])
        self.assertEqual(actual["durations"],[5,5,5,5])
        self.assertEqual(actual["ability_durations"],[0,0,0,0])
        self.assertEqual(actual["same_weather_move"],[False,False,False,False])
        self.assertEqual(actual["rain_modifiers"],[1.5,.5,None])
        self.assertEqual(actual["sun_modifiers"],[.5,1.5,None])
        self.assertEqual(actual["thunder_accuracy"],[True,50,70])
        self.assertEqual(actual["expiry_callback_counts"],[0,1])
        self.assertFalse(actual["rock_spdef_handler"])


if __name__ == "__main__":
    unittest.main()
