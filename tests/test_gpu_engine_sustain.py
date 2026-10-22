"""Selected seed/Wish event fixtures, not completed-game fidelity evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE=os.environ.get("NODE") or shutil.which("node")
SHOWDOWN=os.environ.get("POKEZERO_SHOWDOWN_ROOT")


@unittest.skipUnless(importlib.util.find_spec("jax"),"requires isolated spike JAX")
class SustainTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,entry,items,mechanics,sustain,transition,weather
        from gpu_engine.state import LEECH_SEED,empty_state
        self.jax,self.j,self.m,self.su,self.tr,self.w=jax,j,mechanics,sustain,transition,weather
        self.empty,self.seed=empty_state,LEECH_SEED
        self.names=["_none","leechseed","wish","splash","struggle","protect","substitute","tackle"]
        self.meta={"vocabulary":{"moves":self.names,"items":["_none","leftovers"],
            "abilities":["_none","speedboost"],"species":["_none","mew"],
            "types":["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
                "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]},
            "moves":[None]+[{"flags":{"protect":1} if name in ("leechseed","tackle") else {}}
                for name in self.names[1:]]}
        self.sustain=sustain.from_metadata(self.meta)
        self.table=transition.MoveTable(j.array([0,0,0,0,50,0,0,40]),
            j.array([0,12,1,1,0,1,1,1]),j.array([False]*4+[True,False,False,True]),
            j.full(8,100),j.ones(8,bool),j.array([0]*5+[3,0,0]),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(4),
            effects=effects.from_metadata(self.meta),items=items.from_metadata(self.meta),
            entry=entry.from_metadata(self.meta),weather=weather.from_metadata(self.meta),sustain=self.sustain)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=71)
        return s._replace(stats=j.full_like(s.stats,200),level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,80),max_hp=j.full_like(s.hp,101),pp=j.full_like(s.pp,10),
            moves=j.full_like(s.moves,3),types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,actions=None,table=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),
            self.table if table is None else table)

    def test_seed_grass_substitute_dead_and_ineligible_targets_preserve_flags(self):
        j=self.j
        for typ,hp,sub,eligible,expected in ((1,80,0,True,6),(12,80,0,True,2),
                (1,80,25,True,2),(1,0,0,True,2),(1,80,0,False,2)):
            r=self.su.seed_target(self.sustain,j.array([1]),j.array([[typ,0]]),j.array([hp]),
                j.array([sub]),j.array([2],j.uint32),eligible=j.array([eligible]))
            self.assertEqual(r.tolist(),[expected])

    def test_leech_uses_actual_damage_and_requires_living_not_injured_recipient(self):
        j=self.j
        r=self.su.leech(j.array([80,3,80,80]),j.full(4,101),j.full(4,self.seed,j.uint32),
            j.array([101,50,0,50]),j.full(4,101),eligible=j.array([True,True,True,False]))
        self.assertEqual((r.hp.tolist(),r.recipient_hp.tolist()),([68,0,80,80],[101,53,0,50]))

    def test_wish_clock_expiry_uses_recipient_hp_and_clears_on_failed_heal(self):
        j=self.j
        r=self.su.wish_residual(j.array([10,0,101,10,10]),j.full(5,101),j.array([1,1,1,2,1]),
            eligible=j.array([True,True,True,True,False]))
        self.assertEqual((r.hp.tolist(),r.counter.tolist()),([60,0,101,10,10],[0,0,0,1,1]))

    def test_seed_actual_move_drains_and_blocks_on_miss_protect_substitute_grass(self):
        j=self.j;s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(1))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.hp[0,1,0])),(92,68))
        self.assertEqual(int(r.volatile[0,1,0]),self.seed)
        cases=[(s,self.table._replace(accuracy=self.table.accuracy.at[1].set(0),
            always_hit=self.table.always_hit.at[1].set(False))),
            (s._replace(moves=s.moves.at[0,1,0,0].set(5)),self.table),
            (s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(25)),self.table),
            (s._replace(types=s.types.at[0,1,0,0].set(12)),self.table)]
        for state,table in cases:
            r=self.play(state,table=table).state
            self.assertEqual((int(r.hp[0,1,0]),int(r.volatile[0,1,0])&self.seed),(80,0))
            self.assertEqual(int(r.pp[0,0,0,0]),9)

    def test_seed_clears_on_target_switch_but_follows_source_slot_occupant(self):
        j=self.j;s=self.live()._replace(volatile=self.live().volatile.at[0,1,0].set(self.seed))
        switched=self.play(s,[[0,4]]).state
        self.assertEqual((int(switched.volatile[0,1,0]),int(switched.hp[0,1,1])),(0,80))
        source=s._replace(hp=s.hp.at[0,0,0].set(50).at[0,0,1].set(10))
        r=self.play(source,[[4,0]]).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.hp[0,0,1]),int(r.hp[0,1,0])),(50,22,68))

    def test_forced_replacement_does_not_tick_wish_or_leech(self):
        s=self.live()._replace(volatile=self.live().volatile.at[0,1,0].set(self.seed),
            wish_counter=self.j.array([[1,1]]))
        s=s._replace(hp=s.hp.at[0,0,0].set(0))
        r=self.play(s,[[4,0]]).state
        self.assertEqual(r.wish_counter.tolist(),[[1,1]])
        self.assertEqual((int(r.hp[0,0,1]),int(r.hp[0,1,0]),int(r.turn[0])),(80,80,0))

    def test_seed_ko_heals_before_win_and_skips_later_status(self):
        j=self.j;s=self.live()
        s=s._replace(hp=j.zeros_like(s.hp).at[0,0,0].set(50).at[0,1,0].set(3),
            status=s.status.at[0,0,0].set(self.m.PSN),volatile=s.volatile.at[0,1,0].set(self.seed))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.hp[0,1,0]),bool(r.done[0]),int(r.winner[0])),(53,0,True,0))

    def test_speed_precedes_seed_suborder_but_equal_speed_leftovers_precedes_seed(self):
        for speed,expected in ((200,(7,38)),(300,(0,50))):
            s=self.live()._replace(hp=self.live().hp.at[0,0,0].set(1).at[0,1,0].set(50),
                volatile=self.live().volatile.at[0,1,0].set(self.seed),item=self.live().item.at[0,0,0].set(1),
                status=self.live().status.at[0,0,0].set(self.m.PSN))
            s=s._replace(stats=s.stats.at[0,0,0,5].set(speed))
            r=self.play(s).state
            self.assertEqual((int(r.hp[0,0,0]),int(r.hp[0,1,0])),expected)

    def test_wish_creation_turn_then_expiry_without_refresh(self):
        s=self.live()._replace(hp=self.live().hp.at[0,0,0].set(10),
            moves=self.live().moves.at[0,0,0,0].set(2))
        first=self.play(s).state
        self.assertEqual((int(first.hp[0,0,0]),int(first.wish_counter[0,0])),(10,1))
        second=self.play(first).state
        self.assertEqual((int(second.hp[0,0,0]),int(second.wish_counter[0,0])),(60,0))

    def test_wish_follows_slot_and_uses_recipient_not_wisher_max_hp(self):
        s=self.live()._replace(wish_counter=self.j.array([[1,0]]))
        s=s._replace(max_hp=s.max_hp.at[0,0,0].set(301),hp=s.hp.at[0,0,1].set(10))
        r=self.play(s,[[4,0]]).state
        self.assertEqual((int(r.hp[0,0,1]),int(r.wish_counter[0,0])),(60,0))

    def test_move_ko_defers_wish_until_replacement_recipient_is_installed(self):
        s=self.live()._replace(wish_counter=self.j.array([[1,0]]))
        s=s._replace(hp=s.hp.at[0,0,0].set(1),moves=s.moves.at[0,1,0,0].set(7),
            stats=s.stats.at[0,1,0,5].set(300))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.wish_counter[0,0]),bool(r.pending_residual[0])),(0,1,True))
        replaced=self.play(r,[[4,0]]).state
        self.assertEqual((int(replaced.hp[0,0,1]),int(replaced.wish_counter[0,0])),(101,0))

    def test_wish_precedes_potentially_lethal_weather_chip(self):
        s=self.live()._replace(wish_counter=self.j.array([[1,0]]),weather=self.j.array([self.w.SAND]))
        s=s._replace(hp=s.hp.at[0,0,0].set(1))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,0,0]),int(r.wish_counter[0,0])),(45,0))

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4)._replace(volatile=self.live(4).volatile.at[:,1,0].set(self.seed),
            wish_counter=j.ones((4,2),j.int32))
        s=s._replace(done=s.done.at[3].set(True));actions=j.array([[0,0],[0,0],[9,0],[0,0]])
        compiled=self.jax.jit(self.tr.step)(s,actions,self.table);compiled.state.hp.block_until_ready()
        eager=self.tr.step(s,actions,self.table)
        for a,b in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(eager)):
            self.assertTrue(bool(j.array_equal(a,b)))
        smaller=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for a,b in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(smaller)):
            self.assertTrue(bool(j.array_equal(a[:2],b)))
        for a,b in zip(s,compiled.state):self.assertTrue(bool(j.array_equal(a[2:],b[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_seed_and_wish_events(self):
        result=subprocess.run([NODE,str(Path(__file__).with_name("gpu_sustain_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(seed=dict(full=[101,68],dead=80,
            switched=[50,22,68],ko=[53,0,True],blocked=dict(grass=False,substitute=False,protect=False,miss=False),
            ordered=[[7,38],[0,50]]),
            wish=dict(initial=2,first=[10,1],retry=1,second=[60,False],switched=60,dead=[0,False],
                before_weather=[45,False]),seed_order=[10,5],wish_order=7))


if __name__ == "__main__":unittest.main()
