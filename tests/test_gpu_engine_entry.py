"""Authored switch/entry fixtures; not a gameplay coverage qualification."""
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
class EntryTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,entry,items,transition,weather
        from gpu_engine.state import empty_state
        self.jax,self.j,self.e,self.t,self.w,self.empty = jax,j,entry,transition,weather,empty_state
        self.names = ["_none","splash","struggle","tackle","spikes","earthquake"]
        self.abilities = ["_none","drizzle","drought","sandstream","intimidate","naturalcure",
            "levitate","clearbody","swiftswim","chlorophyll","airlock","whitesmoke","hypercutter"]
        types = ["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta = {"vocabulary":{"moves":self.names,"abilities":self.abilities,"types":types,
            "items":["_none","leftovers"],"species":["_none","ditto"]}}
        self.et = entry.from_metadata(meta)
        self.table = transition.MoveTable(j.array([0,0,50,40,0,100]),j.array([0,1,0,1,5,5]),
            j.array([False,False,True,True,False,True]),j.full(6,100),j.ones(6,bool),j.zeros(6,j.int32),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2),effects.from_metadata(meta),
            items.from_metadata(meta),weather.from_metadata(meta),self.et)

    def live(self,batch=1):
        j = self.j
        s = self.empty(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,100),hp=j.full_like(s.hp,101),
            max_hp=j.full_like(s.hp,101),pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,1),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,actions=None):
        return self.t.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),self.table).state

    def test_initial_weather_entries_are_speed_ordered_and_only_start_once(self):
        for faster,expected in ((0,self.w.SUN),(1,self.w.RAIN)):
            s = self.live()._replace(entry_pending=self.j.ones((1,2),bool))
            s = s._replace(ability=s.ability.at[0,0,0].set(1).at[0,1,0].set(2),
                stats=s.stats.at[0,faster,0,5].set(400))
            r = self.play(s)
            self.assertEqual((int(r.weather[0]),int(r.weather_counter[0])),(expected,0))
            self.assertEqual(r.entry_pending.tolist(),[[False,False]])
            r = r._replace(weather=self.j.array([self.w.HAIL]),weather_counter=self.j.array([3]))
            self.assertEqual(int(self.play(r).weather[0]),self.w.HAIL)

    def test_explicit_initialization_precedes_observation_and_matches_step_fallback(self):
        s = self.live(2)._replace(entry_pending=self.j.ones((2,2),bool))
        s = s._replace(ability=s.ability.at[:,0,0].set(3))
        initial = self.t.initialize(s,self.table)
        self.assertEqual(initial.weather.tolist(),[self.w.SAND]*2)
        self.assertEqual(initial.turn.tolist(),[0,0])
        self.assertTrue(bool(self.j.array_equal(initial.hp,s.hp)))
        self.assertTrue(bool(self.j.array_equal(initial.keys,s.keys)))
        direct,explicit = self.play(s),self.play(initial)
        for a,b in zip(self.jax.tree.leaves(direct),self.jax.tree.leaves(explicit)):
            self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_weather_ability_converts_temporary_same_weather_to_permanent(self):
        for ability,weather in ((1,self.w.RAIN),(2,self.w.SUN),(3,self.w.SAND)):
            s = self.live()._replace(weather=self.j.array([weather]),weather_counter=self.j.array([2]))
            s = s._replace(ability=s.ability.at[0,0,1].set(ability))
            r = self.play(s,[[4,0]])
            self.assertEqual((int(r.weather[0]),int(r.weather_counter[0])),(weather,0))
            r = self.play(r,[[4,0]])
            self.assertEqual((int(r.weather[0]),int(r.weather_counter[0])),(weather,0))

    def test_spikes_ko_prevents_weather_start_and_cancels_foe_move(self):
        s = self.live()._replace(weather=self.j.array([self.w.SUN]))
        s = s._replace(ability=s.ability.at[0,0,1].set(1),spikes=s.spikes.at[0,0].set(1),
            hp=s.hp.at[0,0,1].set(1),moves=s.moves.at[0,1,0,0].set(3))
        r = self.play(s,[[4,0]])
        self.assertEqual((int(r.hp[0,0,1]),int(r.weather[0]),int(r.pp[0,1,0,0])),(0,self.w.SUN,10))
        self.assertTrue(bool(r.forced_switch[0,0]))

    def test_intimidate_applies_before_incoming_foe_attack(self):
        s = self.live(64)
        s = s._replace(hp=self.j.full_like(s.hp,1000),max_hp=self.j.full_like(s.hp,1000),
            moves=s.moves.at[:,1,0,0].set(3))
        base = self.play(s,self.j.tile(self.j.array([[4,0]]),(64,1)))
        s = s._replace(ability=s.ability.at[:,0,1].set(4))
        r = self.play(s,self.j.tile(self.j.array([[4,0]]),(64,1)))
        self.assertTrue(bool(self.j.all(r.boosts[:,1,0,0] == -1)))
        self.assertTrue(bool(self.j.all(r.hp[:,0,1] >= base.hp[:,0,1])))
        self.assertTrue(bool(self.j.any(r.hp[:,0,1] > base.hp[:,0,1])))

    def test_double_switch_entry_uses_outgoing_speed_not_incoming_speed(self):
        for faster in (0,1):
            s = self.live()
            s = s._replace(ability=s.ability.at[:,:,1].set(4),
                stats=s.stats.at[0,faster,0,5].set(400).at[0,1-faster,1,5].set(900))
            r = self.play(s,[[4,4]])
            self.assertEqual(int(r.boosts[0,faster,1,0]),-1)
            self.assertEqual(int(r.boosts[0,1-faster,1,0]),0)
            self.assertEqual(r.boosts[0,:,0,0].tolist(),[0,0])

    def test_intimidate_respects_substitute_ability_immunity_and_floor(self):
        for ability,sub,initial,expected in ((0,0,0,-1),(0,25,0,0),(7,0,0,0),
            (11,0,0,0),(12,0,0,0),(0,0,-6,-6)):
            s = self.live()
            s = s._replace(ability=s.ability.at[0,0,1].set(4).at[0,1,0].set(ability),
                substitute_hp=s.substitute_hp.at[0,1,0].set(sub),boosts=s.boosts.at[0,1,0,0].set(initial))
            self.assertEqual(int(self.play(s,[[4,0]]).boosts[0,1,0,0]),expected)

    def test_natural_cure_clears_departing_status_and_all_volatile_counters(self):
        s = self.live(6)
        s = s._replace(ability=s.ability.at[:,0,0].set(5),status=s.status.at[:,0,0].set(self.j.arange(1,7)),
            status_counter=s.status_counter.at[:,0,0].set(4),confusion_counter=s.confusion_counter.at[:,0,0].set(3),
            substitute_hp=s.substitute_hp.at[:,0,0].set(25),boosts=s.boosts.at[:,0,0].set(2))
        r = self.play(s,self.j.tile(self.j.array([[4,0]]),(6,1)))
        for field in ("status","status_counter","confusion_counter","substitute_hp"):
            self.assertTrue(bool(self.j.all(getattr(r,field)[:,0,0] == 0)))
        self.assertTrue(bool(self.j.all(r.boosts[:,0,0] == 0)))

    def test_non_cure_switch_retains_status_and_resets_only_toxic_escalation(self):
        from gpu_engine import mechanics as m
        for status,counter in ((m.TOX,0),(m.SLP,4),(m.PAR,4)):
            s = self.live()._replace(status=self.live().status.at[0,0,0].set(status),
                status_counter=self.live().status_counter.at[0,0,0].set(4))
            r = self.play(s,[[4,0]])
            self.assertEqual((int(r.status[0,0,0]),int(r.status_counter[0,0,0])),(status,counter))

    def test_spikes_layers_floor_and_only_damage_the_incoming_grounded_mon(self):
        for layers,damage in ((1,12),(2,16),(3,25)):
            for ability,typ,immune in ((0,1,False),(6,1,True),(0,3,True)):
                s = self.live()
                s = s._replace(spikes=s.spikes.at[0,0].set(layers),ability=s.ability.at[0,0,1].set(ability),
                    types=s.types.at[0,0,1,0].set(typ))
                r = self.play(s,[[4,0]])
                self.assertEqual(int(r.hp[0,0,1]),101-(0 if immune else damage))
                self.assertEqual(int(r.hp[0,0,0]),101)

    def test_spikes_move_adds_foe_layers_up_to_three_through_substitute(self):
        s = self.live()
        s = s._replace(moves=s.moves.at[0,0,0,0].set(4),substitute_hp=s.substitute_hp.at[0,1,0].set(25))
        for expected in (1,2,3,3):
            s = self.play(s)
            self.assertEqual(s.spikes.tolist(),[[0,expected]])

    def test_hazard_faint_cancels_the_slower_ordinary_switch_before_mutation(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,0,1].set(1),spikes=s.spikes.at[0,0].set(1),
            ability=s.ability.at[0,1,1].set(2),boosts=s.boosts.at[0,1,0,0].set(3))
        r = self.play(s,[[4,4]])
        self.assertEqual(r.active.tolist(),[[1,0]])
        self.assertEqual(int(r.boosts[0,1,0,0]),3)
        self.assertEqual(int(r.weather[0]),self.w.NONE)

    def test_forced_replacement_runs_entry_but_no_ordinary_turn_or_residual(self):
        s = self.live()._replace(weather=self.j.array([self.w.SAND]),weather_counter=self.j.array([2]))
        s = s._replace(hp=s.hp.at[0,0,0].set(0),spikes=s.spikes.at[0,0].set(1),
            ability=s.ability.at[0,0,1].set(4))
        r = self.play(s,[[4,0]])
        self.assertEqual((int(r.hp[0,0,1]),int(r.hp[0,1,0]),int(r.weather_counter[0]),int(r.turn[0])),(89,101,2,0))
        self.assertEqual(int(r.boosts[0,1,0,0]),-1)

    def test_both_forced_entrants_exist_before_either_intimidate(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[:,:,0].set(0),ability=s.ability.at[:,:,1].set(4))
        r = self.play(s,[[4,4]])
        self.assertEqual(r.active.tolist(),[[1,1]])
        self.assertEqual(r.boosts[0,:,1,0].tolist(),[-1,-1])
        self.assertEqual(int(r.turn[0]),0)

    def test_forced_weather_order_uses_incoming_not_fainted_outgoing_speed(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[:,:,0].set(0),ability=s.ability.at[0,0,1].set(1).at[0,1,1].set(2),
            stats=s.stats.at[0,0,0,5].set(900).at[0,1,0,5].set(100).at[0,0,1,5].set(100).at[0,1,1,5].set(400))
        r = self.play(s,[[4,4]])
        self.assertEqual((int(r.weather[0]),int(r.weather_counter[0])),(self.w.RAIN,0))

    def test_forced_hazard_ko_cancels_later_callback_not_already_installed_entrant(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[:,:,0].set(0).at[0,0,1].set(1),spikes=s.spikes.at[0].set(1),
            ability=s.ability.at[0,1,1].set(2))
        r = self.play(s,[[4,4]])
        self.assertEqual(r.active.tolist(),[[1,1]])
        self.assertEqual(r.hp[0,:,1].tolist(),[0,101])
        self.assertEqual(int(r.weather[0]),self.w.NONE)

    def test_last_mon_hazard_ko_ends_before_foe_residual(self):
        s = self.live()
        s = s._replace(hp=s.hp.at[0,0].set(0).at[0,0,1].set(1).at[0,1,0].set(50),
            spikes=s.spikes.at[0,0].set(1),item=s.item.at[0,1,0].set(1))
        r = self.play(s,[[4,0]])
        self.assertEqual((bool(r.done[0]),int(r.winner[0]),int(r.hp[0,1,0])),(True,1,50))

    def test_levitate_blocks_ground_damage_but_not_other_types(self):
        for move,expected in ((5,101),(3,None)):
            s = self.live()._replace(ability=self.live().ability.at[0,1,0].set(6),
                moves=self.live().moves.at[0,0,0,0].set(move))
            hp = int(self.play(s).hp[0,1,0])
            self.assertEqual(hp,expected) if expected is not None else self.assertLess(hp,101)

    def test_weather_speed_multiplier_precedes_paralysis_and_obeys_suppression(self):
        from gpu_engine import mechanics as m
        for ability,weather in ((8,self.w.RAIN),(9,self.w.SUN)):
            s = self.live()._replace(weather=self.j.array([weather]))
            s = s._replace(ability=s.ability.at[0,0,0].set(ability),stats=s.stats.at[0,0,0,5].set(103),
                status=s.status.at[0,0,0].set(m.PAR))
            self.assertEqual(int(self.t._speeds(s,self.table)[0,0]),51)
            s = s._replace(ability=s.ability.at[0,1,0].set(10))
            self.assertEqual(int(self.t._speeds(s,self.table)[0,0]),26)

    def test_residual_speed_is_frozen_before_weather_expiry(self):
        from gpu_engine import mechanics as m
        s = self.live()._replace(weather=self.j.array([self.w.RAIN]),weather_counter=self.j.array([1]))
        s = s._replace(hp=s.hp.at[:,:,1:].set(0).at[0,0,0].set(1).at[0,1,0].set(50),
            stats=s.stats.at[0,0,0,5].set(100).at[0,1,0,5].set(150),
            ability=s.ability.at[0,0,0].set(8),status=s.status.at[0,0,0].set(m.PSN),
            item=s.item.at[0,1,0].set(1))
        r = self.play(s)
        self.assertEqual((int(r.weather[0]),int(r.hp[0,0,0]),int(r.hp[0,1,0])),(0,0,50))

    def test_jit_and_batch_invariance_for_initial_and_ordinary_entries(self):
        s = self.live(2)
        s = s._replace(entry_pending=s.entry_pending.at[0].set(True),
            ability=s.ability.at[:,0,0].set(3).at[:,0,1].set(4),spikes=s.spikes.at[1,0].set(2))
        actions = self.j.array([[0,0],[4,0]])
        a = self.t.step(s,actions,self.table)
        b = self.jax.jit(self.t.step)(s,actions,self.table)
        self.jax.block_until_ready(b)
        c = self.t.step(self.jax.tree.map(lambda x:x[:1],s),actions[:1],self.table)
        for x,y,z in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b),self.jax.tree.leaves(c)):
            self.assertTrue(bool(self.j.array_equal(x,y)))
            self.assertTrue(bool(self.j.array_equal(x[:1],z)))

    def test_pending_entry_does_not_mutate_invalid_or_terminal_battle(self):
        for done,action in ((True,0),(False,9)):
            s = self.live()._replace(done=self.j.array([done]),entry_pending=self.j.ones((1,2),bool))
            s = s._replace(ability=s.ability.at[0,0,0].set(3))
            r = self.play(s,[[action,0]])
            for x,y in zip(self.jax.tree.leaves(s),self.jax.tree.leaves(r)):
                self.assertTrue(bool(self.j.array_equal(x,y)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned entry source")
    def test_pinned_authored_switch_and_handler_oracle(self):
        actual = json.loads(subprocess.run([NODE,str(Path(__file__).with_name("gpu_entry_oracle.cjs")),SHOWDOWN],
            text=True,capture_output=True,check=True).stdout)
        self.assertEqual(actual["double_switch_attack_stages"],[-1,0])
        self.assertEqual(actual["weather_start"],"sunnyday")
        self.assertEqual(actual["spikes_damage"],[12,16,25])
        self.assertEqual(actual["natural_cure"],{"status":"","counter":0})
        self.assertEqual(actual["intimidate_substitute_drop_calls"],0)
        self.assertEqual(actual["forced_attack_stages"],[-1,-1])
        self.assertEqual(actual["forced_hazard_hp"],[0,101])


if __name__ == "__main__":
    unittest.main()
