"""Pressure use boundaries and pinned target facts; not gameplay qualification."""
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
class PressureTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import effects,pressure,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.p,self.tr,self.empty=jax,j,pressure,transition,empty_state
        self.names=["_none","tackle","toxic","recover","haze","raindance","spikes",
            "reflect","healbell","curse","splash","struggle","seismictoss","protect"]
        self.types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        self.meta={"vocabulary":{"moves":self.names,"types":self.types,"abilities":["_none","pressure"]},
            "moves":[None]+[{"flags":{"protect":1}} for _ in self.names[1:]]}
        self.pressure=pressure.from_metadata(self.meta)
        n=len(self.names)
        self.table=transition.MoveTable(j.array([0,40]+[0]*9+[50,0,0]),j.ones(n,j.int32),
            j.array([False,True]+[False]*(n-2)),j.full(n,100),j.ones(n,bool),
            j.array([0]*(n-1)+[3]),j.ones((18,18),j.int32),j.ones((18,18),j.int32),
            j.array(11),effects=effects.from_metadata(self.meta),pressure=self.pressure)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=71)
        return s._replace(stats=j.full_like(s.stats,200),level=j.full_like(s.level,100),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),pp=j.full_like(s.pp,10),
            moves=j.full_like(s.moves,10),types=s.types.at[...,0].set(1),done=j.zeros_like(s.done),
            ability=s.ability.at[:,1,0].set(1))

    def test_targets_include_field_fixed_damage_and_spikes_not_self_or_ally(self):
        j=self.j
        costs=self.p.cost(self.pressure,j.arange(len(self.names)),j.tile(j.array([[1,0]]),(len(self.names),1)),
            j.ones(len(self.names),j.int32),j.ones(len(self.names),j.int32))
        self.assertEqual(costs.tolist(),[1,2,2,1,2,2,2,1,1,1,1,2,2,1])
        self.assertEqual(int(self.p.cost(self.pressure,j.array(9),j.array([8,0]),j.array(1),j.array(1))),2)

    def test_current_living_foe_only(self):
        j=self.j
        self.assertEqual(self.p.cost(self.pressure,j.ones(3,j.int32),j.array([[1,0]]*3),
            j.array([1,0,1]),j.array([1,1,0])).tolist(),[2,1,1])

    def test_actual_use_deducts_before_miss_protect_and_substitute(self):
        j=self.j;s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(1))
        cases=[(s,self.table),(s,self.table._replace(accuracy=self.table.accuracy.at[1].set(0),
            always_hit=self.table.always_hit.at[1].set(False))),
            (s._replace(moves=s.moves.at[0,1,0,0].set(13)),self.table),
            (s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(100)),self.table)]
        for state,table in cases:
            self.assertEqual(int(self.tr.step(state,j.array([[0,0]]),table).state.pp[0,0,0,0]),8)

    def test_inability_skips_cost_and_last_pp_clamps_without_struggle_charge(self):
        from gpu_engine import mechanics as m
        j=self.j;s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(1))
        asleep=s._replace(status=s.status.at[0,0,0].set(m.SLP),status_counter=s.status_counter.at[0,0,0].set(10))
        self.assertEqual(int(self.tr.step(asleep,j.array([[0,0]]),self.table).state.pp[0,0,0,0]),10)
        low=s._replace(pp=s.pp.at[0,0,0,0].set(1))
        self.assertEqual(int(self.tr.step(low,j.array([[0,0]]),self.table).state.pp[0,0,0,0]),0)
        empty=s._replace(pp=s.pp.at[0,0,0].set(0))
        self.assertEqual(self.tr.step(empty,j.array([[0,0]]),self.table).state.pp[0,0,0].tolist(),[0]*4)

    def test_switch_in_pressure_applies_but_forced_pseudoturn_does_not_cost(self):
        j=self.j;s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(1))
        s=s._replace(ability=s.ability.at[0,1,0].set(0).at[0,1,1].set(1))
        self.assertEqual(int(self.tr.step(s,j.array([[0,4]]),self.table).state.pp[0,0,0,0]),8)
        forced=s._replace(hp=s.hp.at[0,1,0].set(0))
        self.assertTrue(bool(j.array_equal(self.tr.step(forced,j.array([[0,4]]),self.table).state.pp,s.pp)))

    def test_jit_batch_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4)._replace(moves=self.live(4).moves.at[:,0,0,0].set(1))
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
    def test_pinned_target_catalog_and_authored_pp_events(self):
        result=subprocess.run([NODE,str(Path(__file__).with_name("gpu_pressure_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        actual=json.loads(result.stdout)
        self.assertEqual(actual["non_foe"],sorted(self.p.NON_FOE_MOVES))
        self.assertEqual(actual["costs"],dict(tackle=2,toxic=2,seismictoss=2,recover=1,haze=2,
            raindance=2,spikes=2,reflect=1,healbell=1,curse_normal=1,curse_ghost=2,
            protected=2,asleep=0,one_pp=1,incoming_pressure=2))


if __name__ == "__main__":unittest.main()
