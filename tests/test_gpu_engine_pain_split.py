"""Authored Pain Split HP assignments, not a source panel or coverage credit."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

NODE = os.environ.get("NODE") or shutil.which("node")
SHOWDOWN = os.environ.get("POKEZERO_SHOWDOWN_ROOT")

CASES = [
    dict(own=101, foe=400, own_max=500, foe_max=800, expected=[250,250]),
    dict(own=271, foe=341, own_max=271, foe_max=341, expected=[271,306]),
    dict(own=401, foe=100, own_max=1000, foe_max=150, expected=[250,150]),
    dict(own=100, foe=401, own_max=150, foe_max=1000, expected=[150,250]),
    dict(own=1, foe=2, own_max=500, foe_max=800, expected=[1,1]),
    dict(own=101, foe=400, own_max=500, foe_max=800, target_sub=20, expected=[101,400]),
    dict(own=101, foe=400, own_max=500, foe_max=800, protect=True, expected=[101,400]),
    dict(own=101, foe=400, own_max=500, foe_max=800, ghost=True, expected=[250,250]),
    dict(own=101, foe=400, own_max=500, foe_max=800, own_sub=20, expected=[250,250]),
    dict(own=401, foe=200, own_max=1000, foe_max=1000, prior_damage=True, expected=[250,250]),
    dict(own=401, foe=200, own_max=1000, foe_max=1000, switch=True, expected=[600,600]),
]


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires isolated JAX")
class PainSplitTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import damage_moves, effects, mechanics, transition
        from gpu_engine.state import empty_state
        self.jax, self.j, self.m, self.t, self.empty = jax, j, mechanics, transition, empty_state
        self.names = ["_none", "splash", "painsplit", "struggle", "protect", "seismictoss"]
        self.types = ["_none", "normal", "fighting", "flying", "poison", "ground", "rock",
                      "bug", "ghost", "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark"]
        rows = [None]+[dict(flags={"protect":1} if name in {"painsplit", "seismictoss"} else {})
                      for name in self.names[1:]]
        meta = dict(vocabulary=dict(moves=self.names, types=self.types, abilities=["_none"]), moves=rows)
        self.table = transition.MoveTable(j.array([0,0,0,50,0,0]), j.array([0,1,1,0,1,2]),
            j.array([False,False,False,True,False,True]), j.full(6,100), j.ones(6,bool),
            j.array([0,0,0,0,3,0]), j.ones((18,18),j.int32), j.ones((18,18),j.int32),
            j.array(3), effects=effects.from_metadata(meta), damage_moves=damage_moves.from_metadata(meta))

    def fixture(self, case, side=0):
        j = self.j
        state = self.empty(1,seed=117)
        foe = 1-side
        state = state._replace(hp=j.full_like(state.hp,1000), max_hp=j.full_like(state.max_hp,1000),
            level=j.full_like(state.level,100), stats=j.full_like(state.stats,200),
            types=state.types.at[...,0].set(1), moves=j.ones_like(state.moves),
            pp=j.full_like(state.pp,8), done=j.zeros_like(state.done))
        state = state._replace(hp=state.hp.at[0,side,0].set(case["own"]).at[0,foe,0].set(case["foe"]),
            max_hp=state.max_hp.at[0,side,0].set(case["own_max"]).at[0,foe,0].set(case["foe_max"]),
            stats=state.stats.at[0,side,0,5].set(199 if case.get("prior_damage") else 301),
            moves=state.moves.at[0,side,0,0].set(2).at[0,foe,0,0].set(
                4 if case.get("protect") else 5 if case.get("prior_damage") else 1),
            substitute_hp=state.substitute_hp.at[0,foe,0].set(case.get("target_sub",0)).at[0,side,0].set(case.get("own_sub",0)),
            types=state.types.at[0,foe,0,0].set(8 if case.get("ghost") else 1))
        actions=j.zeros((1,2),j.int32)
        if case.get("switch"):
            state=state._replace(hp=state.hp.at[0,foe,1].set(800))
            actions=actions.at[0,foe].set(4)
        return state, actions

    def check_case(self, case, side, *, compiled=False):
        state, actions=self.fixture(case,side)
        step=self.jax.jit(self.t.step) if compiled else self.t.step
        result=step(state,actions,self.table)
        self.jax.block_until_ready(result)
        self.assertFalse(bool(result.invalid_actions[0]))
        after=result.state;foe=1-side;target=1 if case.get("switch") else 0
        self.assertEqual([int(after.hp[0,side,0]),int(after.hp[0,foe,target])],case["expected"])
        self.assertEqual(int(after.pp[0,side,0,0]),7)
        self.assertEqual(int(after.substitute_hp[0,side,0]),case.get("own_sub",0))
        self.assertEqual(int(after.substitute_hp[0,foe,target]),case.get("target_sub",0))
        self.assertEqual(after.hp[0,side,1:].tolist(),state.hp[0,side,1:].tolist())
        if case.get("switch"):
            self.assertEqual(int(after.hp[0,foe,0]),case["foe"])
        else:
            self.assertEqual(after.hp[0,foe,1:].tolist(),state.hp[0,foe,1:].tolist())
            self.assertEqual(int(after.pp[0,foe,0,0]),7)
        self.assertEqual(after.status.tolist(),state.status.tolist())
        self.assertEqual(after.item.tolist(),state.item.tolist())
        return state,actions,result

    def test_live_hp_average_floor_caps_guards_and_switch_target_in_both_seats(self):
        for case in CASES:
            for side in (0,1):
                with self.subTest(case=case,side=side):self.check_case(case,side)

    def test_jit_matches_eager_and_invalid_terminal_battles_are_identity(self):
        state,actions,eager=self.check_case(CASES[0],0)
        compiled=self.jax.jit(self.t.step)(state,actions,self.table)
        self.jax.block_until_ready(compiled)
        for a,b in zip(self.jax.tree.leaves(eager),self.jax.tree.leaves(compiled)):
            self.assertTrue(bool(self.j.array_equal(a,b)))
        for state,actions in ((state,actions.at[0,0].set(9)),
                              (state._replace(done=self.j.ones_like(state.done)),actions)):
            result=self.t.step(state,actions,self.table)
            for a,b in zip(self.jax.tree.leaves(state),self.jax.tree.leaves(result.state)):
                self.assertTrue(bool(self.j.array_equal(a,b)))

    def test_batched_rows_are_independent_and_forced_replacement_does_not_split(self):
        j=self.j
        fixtures=[self.fixture(case) for case in (CASES[0],CASES[3],CASES[-1])]
        state=self.jax.tree.map(lambda *rows:j.concatenate(rows,axis=0),*(s for s,a in fixtures))
        actions=j.concatenate([a for s,a in fixtures],axis=0)
        compiled=self.jax.jit(self.t.step)(state,actions,self.table)
        self.jax.block_until_ready(compiled)
        for i,(s,a) in enumerate(fixtures):
            single=self.t.step(s,a,self.table)
            for x,y in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(single)):
                self.assertTrue(bool(j.array_equal(x[i:i+1],y)))
        state,actions=self.fixture(CASES[0])
        state=state._replace(hp=state.hp.at[0,1,0].set(0))
        result=self.t.step(state,actions.at[0,1].set(4),self.table)
        self.assertEqual(result.state.hp[0,0].tolist(),state.hp[0,0].tolist())
        self.assertEqual(int(result.state.hp[0,1,0]),0)
        self.assertEqual(int(result.state.hp[0,1,1]),1000)
        self.assertEqual(int(result.state.pp[0,0,0,0]),8)

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Gen3 Showdown")
    def test_independent_pinned_gen3_battle_matches_every_authored_case(self):
        helper=Path(__file__).with_name("gpu_pain_split_oracle.cjs")
        specs=[dict(case,side=side) for case in CASES for side in (0,1)]
        run=subprocess.run([NODE,str(helper),SHOWDOWN],input=json.dumps(specs),
            text=True,capture_output=True,check=True)
        rows=json.loads(run.stdout)
        self.assertEqual(len(rows),len(specs))
        for spec,row in zip(specs,rows,strict=True):
            self.assertEqual(row["hp"],spec["expected"],spec)
            self.assertEqual(row["pp"],7,spec)
            self.assertEqual(row["target_sub"],spec.get("target_sub",0),spec)
            self.assertEqual(row["own_sub"],spec.get("own_sub",0),spec)
            self.assertEqual(row["active"],1 if spec.get("switch") else 0,spec)


if __name__=="__main__":unittest.main()
