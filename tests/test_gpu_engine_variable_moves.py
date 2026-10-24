"""Per-Pokemon move fixtures; no complete move or game qualification."""
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
class VariableMoveTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import absorption,combat,damage_moves,effects,transition,variable_moves
        from gpu_engine.state import empty_state
        self.jax,self.j,self.tr,self.v,self.empty=jax,j,transition,variable_moves,empty_state
        self.names=["_none","hiddenpower","hiddenpowerfire","hiddenpowerfighting",
            "return","frustration","splash","struggle","reversal","flail","seismictoss","protect"]
        self.types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        self.meta={"vocabulary":{"moves":self.names,"types":self.types,
            "abilities":["_none","flashfire","thickfat","guts"],"species":["_none"],"items":["_none"]},
            "moves":[None]+[{"source_type":"Fire" if name=="hiddenpowerfire" else "Normal",
                "category":"Special" if name=="hiddenpowerfire" else "Physical","flags":{"protect":1}}
                for name in self.names[1:]]}
        self.profile=variable_moves.from_metadata(self.meta)
        num=j.ones((18,18),j.int32).at[2,8].set(0).at[2,6].set(2).at[10,12].set(2)
        den=j.ones_like(num).at[10,11].set(2)
        self.table=transition.MoveTable(j.array([0,0,70,70,0,0,0,50,0,0,0,0]),
            j.array([0,1,10,2,1,1,1,0,2,1,2,1]),j.ones(12,bool),j.full(12,100),j.ones(12,bool),
            j.zeros(12,j.int32).at[11].set(3),num,den,j.array(7),
            absorption=absorption.from_metadata(self.meta),combat=combat.from_metadata(self.meta),
            damage_moves=damage_moves.from_metadata(self.meta),effects=effects.from_metadata(self.meta),
            variable_moves=self.profile)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=71)
        stats=j.full_like(s.stats,200).at[:,0,:,1].set(77).at[:,0,:,3].set(333).at[:,0,:,5].set(300)
        return s._replace(stats=stats,level=j.full_like(s.level,73),hp=j.full_like(s.hp,1000),
            max_hp=j.full_like(s.hp,1000),moves=j.full_like(s.moves,6),pp=j.full_like(s.pp,10),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))

    def play(self,s,table=None,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else actions,
            self.table if table is None else table)

    def test_happiness_endpoints_and_hidden_power_defaults(self):
        j=self.j
        power,typ,physical=self.v.resolve(self.profile,j.array([1,2,3,4,4,5,5]),
            j.zeros(7,j.int32),j.ones(7,j.int32),j.ones(7,bool),
            j.array([0,10,2,0,0,0,0]),j.array([0,33,40,0,0,0,0]),j.array([255,255,255,0,255,0,255]),
            j.full(7,1000),j.full(7,1000))
        self.assertEqual(power.tolist(),[70,33,40,1,102,102,1])
        self.assertEqual(typ[:3].tolist(),[17,10,2])
        self.assertEqual(physical[:3].tolist(),[False,False,True])

    def test_all_hidden_power_aliases_use_pokemon_type_power_before_modifiers(self):
        j=self.j
        for move,typ,power,burn,ability,foe_ability in (
                (1,10,33,True,0,2),(2,2,40,True,3,0),(3,10,70,False,0,0)):
            s=self.live(8);s=s._replace(moves=s.moves.at[:,0,0,0].set(move),
                hidden_power_type=s.hidden_power_type.at[:,0,0].set(typ),
                hidden_power_power=s.hidden_power_power.at[:,0,0].set(power),
                status=s.status.at[:,0,0].set(2 if burn else 0),
                ability=s.ability.at[:,0,0].set(ability).at[:,1,0].set(foe_ability),
                types=s.types.at[:,0,0,0].set(typ).at[:,1,0,0].set(6 if typ==2 else 12))
            reference=self.table._replace(variable_moves=None,power=self.table.power.at[move].set(power),
                type=self.table.type.at[move].set(typ),
                physical=self.table.physical.at[move].set(not bool(self.profile.special_type[typ])))
            a,b=self.play(s).state,self.play(s,reference).state
            self.assertEqual(a.hp.tolist(),b.hp.tolist())
            self.assertTrue(bool(j.all(a.hp[:,1,0] < 1000)))

    def test_dynamic_fire_absorption_uses_resolved_power_for_canonical_zero_base(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0,0].set(1),
            hidden_power_type=s.hidden_power_type.at[0,0,0].set(10),
            ability=s.ability.at[0,1,0].set(1))
        r=self.play(s).state
        self.assertEqual((int(r.hp[0,1,0]),int(r.volatile[0,1,0]),int(r.pp[0,0,0,0])),(1000,2,9))

    def test_dynamic_fighting_immunity_and_return_frustration_default_damage(self):
        s=self.live();s=s._replace(moves=s.moves.at[0,0,0,0].set(2),
            hidden_power_type=s.hidden_power_type.at[0,0,0].set(2),types=s.types.at[0,1,0,0].set(8))
        self.assertEqual(int(self.play(s).state.hp[0,1,0]),1000)
        damages=[]
        for move,happiness,power in ((4,255,102),(4,0,1),(5,255,1),(5,0,102)):
            s=self.live(8);s=s._replace(moves=s.moves.at[:,0,0,0].set(move),
                happiness=s.happiness.at[:,0,0].set(happiness))
            reference=self.table._replace(variable_moves=None,power=self.table.power.at[move].set(power))
            result=self.play(s).state
            self.assertEqual(result.hp.tolist(),self.play(s,reference).state.hp.tolist())
            damages.append(result.hp[:,1,0].tolist())
        self.assertEqual(damages[0],damages[3]);self.assertEqual(damages[1],damages[2])
        self.assertTrue(all(a<b for a,b in zip(damages[0],damages[1])))

    def test_pool_keeps_existing_derived_fields_and_default_happiness(self):
        import numpy as np
        pool={name:np.array(getattr(self.live(),name)[0],copy=True) for name in
            ("species","level","types","stats","moves","pp","item","ability","hidden_power_type","hidden_power_power")}
        pool["hidden_power_type"][:]=10;pool["hidden_power_power"][:]=33
        s=self.tr.state_from_pool(pool,np.array([[0,1]]))
        self.assertEqual(s.hidden_power_type.tolist(),[[[10]*6]*2])
        self.assertEqual(s.hidden_power_power.tolist(),[[[33]*6]*2])
        self.assertEqual(s.happiness.tolist(),[[[255]*6]*2])
        pool["happiness"]=np.zeros((2,6),np.int32)
        self.assertEqual(self.tr.state_from_pool(pool,np.array([[0,1]])).happiness.tolist(),[[[0]*6]*2])
        del pool["hidden_power_type"]
        with self.assertRaises(KeyError):self.tr.state_from_pool(pool,np.array([[0,1]]))

    def test_reversal_flail_hp_bands_reach_damage_in_both_seats(self):
        j=self.j;rows=j.arange(4);side=j.array([0,1,0,1]);move=j.array([8,8,9,9])
        for hp,power in ((1,200),(50,150),(125,100),(250,80),(500,40),(1000,20)):
            with self.subTest(hp=hp,power=power):
                s=self.live(4)
                s=s._replace(moves=s.moves.at[rows,side,0,0].set(move),
                    hp=s.hp.at[rows,side,0].set(hp),
                    status=s.status.at[rows,side,0].set(2),  # physical burn modifier still applies
                    types=s.types.at[rows,side,0,0].set(self.table.type[move]))
                reference=self.table._replace(variable_moves=None,
                    power=self.table.power.at[8:10].set(power))
                actual,expected=self.play(s).state,self.play(s,reference).state
                self.assertEqual(actual.hp.tolist(),expected.hp.tolist())
                self.assertTrue(bool(j.all(actual.hp[rows,1-side,0]<1000)))
                self.assertEqual(actual.pp[rows,side,0,0].tolist(),[9]*4)
                compiled=self.jax.jit(self.tr.step)(s,j.zeros((4,2),j.int32),self.table).state
                for a,b in zip(compiled,actual):self.assertTrue(bool(j.array_equal(a,b)))

    def test_reversal_flail_resolve_live_hp_after_earlier_damage(self):
        j=self.j
        for side,move in ((0,8),(1,9)):
            s=self.live(4);foe=1-side
            s=s._replace(moves=s.moves.at[:,side,0,0].set(move).at[:,foe,0,0].set(10),
                hp=s.hp.at[:,side,0].set(400),stats=s.stats.at[:,foe,0,5].set(500))
            # Faster level-73 Seismic Toss changes 400/1000 (BP40) to327/1000 (BP80).
            reference=self.table._replace(variable_moves=None,power=self.table.power.at[move].set(80))
            actual,expected=self.play(s).state,self.play(s,reference).state
            self.assertEqual(actual.hp.tolist(),expected.hp.tolist())
            self.assertEqual(actual.hp[:,side,0].tolist(),[327]*4)
            self.assertTrue(bool(j.all(actual.hp[:,foe,0]<1000)))

    def test_reversal_flail_keep_immunity_substitute_and_protect(self):
        j=self.j
        for move in (8,9):
            s=self.live(3);s=s._replace(hp=s.hp.at[:,0,0].set(1),types=s.types.at[0,1,0,0].set(8),
                substitute_hp=s.substitute_hp.at[1,1,0].set(19),
                moves=s.moves.at[:,0,0,0].set(move).at[2,1,0,0].set(11))
            reference=self.table._replace(variable_moves=None,power=self.table.power.at[move].set(200),
                type_num=self.table.type_num.at[1,8].set(0))
            table=self.table._replace(type_num=reference.type_num)
            actual,expected=self.play(s,table).state,self.play(s,reference).state
            self.assertEqual(actual.hp.tolist(),expected.hp.tolist())
            self.assertEqual(actual.substitute_hp.tolist(),expected.substitute_hp.tolist())
            self.assertEqual(actual.hp[:,1,0].tolist(),[1000]*3)
            self.assertEqual(actual.substitute_hp[:,1,0].tolist(),[0,0,0])

    def test_jit_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(moves=s.moves.at[:,0,0,0].set(1),
            hidden_power_type=s.hidden_power_type.at[:,0,0].set(10),done=s.done.at[3].set(True))
        actions=j.array([[0,0],[0,0],[9,0],[0,0]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(s,actions=actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_canonical_power_happiness_and_frozen_flash_fire(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_variable_moves_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        result=json.loads(p.stdout);bands=result.pop('hpBands')
        self.assertEqual(result,dict(defaults=[255,102,1],endpoints=[1,102],
            hidden=[["Fire",33,True],["Fighting",40,True]],
            frozen=dict(damage=True,status="frz",flash=False),awake=dict(damage=False,status="",flash=True)))
        j=self.j;move=j.array([self.names.index(r[0]) for r in bands]);n=len(bands)
        power,typ,physical=self.v.resolve(self.profile,move,j.zeros(n,j.int32),
            self.table.type[move],j.ones(n,bool),j.zeros(n,j.int32),j.zeros(n,j.int32),
            j.full(n,255),j.array([r[1] for r in bands]),j.array([r[2] for r in bands]))
        self.assertEqual(power.tolist(),[r[3] for r in bands])
        self.assertEqual(typ.tolist(),self.table.type[move].tolist())
        self.assertTrue(bool(j.all(physical)))


if __name__=="__main__":unittest.main()
