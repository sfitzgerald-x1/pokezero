"""Selected fixed-hit/recoil fixtures, not complete move/game qualification."""
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
class DamageMoveTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import damage_moves,effects,mechanics,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.dm,self.m,self.tr,self.empty=jax,j,damage_moves,mechanics,transition,empty_state
        self.names=["_none","seismictoss","nightshade","sonicboom","dragonrage","struggle",
            "doubleedge","splash","protect","endure","takedown","submission","volttackle"]
        types=["_none","normal","fighting","flying","poison","ground","rock","bug","ghost",
            "steel","fire","water","grass","electric","psychic","ice","dragon","dark"]
        meta={"vocabulary":{"moves":self.names,"types":types,"abilities":["_none","rockhead"]},
            "moves":[None]+[{"flags":{"protect":1}} for _ in self.names[1:]]}
        self.profile=damage_moves.from_metadata(meta)
        num=j.ones((18,18),j.int32).at[1,8].set(0).at[2,8].set(0).at[8,1].set(0)
        num=num.at[2,6].set(2).at[8,14].set(2)
        den=j.ones_like(num).at[8,17].set(2).at[16,9].set(2)
        self.table=transition.MoveTable(j.array([0,0,0,0,0,50,120,0,0,0,90,80,120]),
            j.array([0,2,8,1,16,0,1,1,1,1,1,2,13]),j.ones(13,bool),
            j.full(13,100),j.ones(13,bool),j.array([0]*8+[3,3,0,0,0]),num,den,j.array(5),
            effects=effects.from_metadata(meta),damage_moves=self.profile)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=71)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            level=j.full_like(s.level,73),hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,7),types=s.types.at[...,0].set(1),
            done=j.zeros_like(s.done))

    def play(self,s,table=None,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else self.j.array(actions),
            self.table if table is None else table)

    def test_level_and_constant_damage_ignore_ordinary_damage_except_immunity(self):
        j=self.j
        r=self.dm.resolve_fixed(self.profile,j.array([1,2,3,4,6]),j.array([73]*5),
            j.array([999]*5),immune=j.array([False,True,False,False,False]))
        self.assertEqual(r.tolist(),[73,0,20,40,999])

    def test_recoil_floors_nondivisible_actual_damage_and_rock_head_excludes_struggle(self):
        j=self.j;amount=j.array([1,3,6,7,11,19])
        for move,expected in ((5,[1,1,1,1,2,4]),(6,[1,1,2,2,3,6])):
            self.assertEqual(self.dm.recoil_damage(self.profile,j.full(6,move),j.zeros(6,j.int32),amount).tolist(),expected)
        self.assertEqual(self.dm.recoil_damage(self.profile,j.array([5,6,6]),j.ones(3,j.int32),j.array([19,19,0])).tolist(),[4,0,0])

    def test_fixed_hit_keeps_immunity_but_not_resistance_stab_boosts_burn_or_roll(self):
        j=self.j
        for move,typ,expected in ((1,6,73),(1,8,0),(2,14,73),(2,17,73),(2,1,0),(3,8,0),(4,9,40)):
            s=self.live(8)._replace(moves=self.live(8).moves.at[:,0,0,0].set(move))
            s=s._replace(types=s.types.at[:,0,0,0].set(2).at[:,1,0,0].set(typ),
                boosts=s.boosts.at[:,0,0,0].set(6).at[:,1,0,1].set(6),status=s.status.at[:,0,0].set(self.m.BRN))
            r=self.play(s).state
            self.assertEqual((1000-r.hp[:,1,0]).tolist(),[expected]*8)

    def test_fixed_damage_uses_substitute_endure_protect_and_accuracy_boundaries(self):
        s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(1))
        sub=s._replace(substitute_hp=s.substitute_hp.at[0,1,0].set(19))
        r=self.play(sub).state
        self.assertEqual((int(r.hp[0,1,0]),int(r.substitute_hp[0,1,0])),(1000,0))
        endure=s._replace(moves=s.moves.at[0,1,0,0].set(9),hp=s.hp.at[0,1,0].set(10))
        self.assertEqual(int(self.play(endure).state.hp[0,1,0]),1)
        protects=s._replace(moves=s.moves.at[0,1,0,0].set(8))
        miss=self.table._replace(accuracy=self.table.accuracy.at[1].set(0),always_hit=self.table.always_hit.at[1].set(False))
        for state,table in ((protects,self.table),(s,miss)):
            r=self.play(state,table).state
            self.assertEqual((int(r.hp[0,1,0]),int(r.pp[0,0,0,0])),(1000,9))

    def test_actual_substitute_damage_controls_recoil_with_and_without_rock_head(self):
        for move,ability,expected in ((5,0,46),(5,1,46),(6,0,44),(6,1,50)):
            s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(move))
            s=s._replace(hp=s.hp.at[0,0,0].set(50),ability=s.ability.at[0,0,0].set(ability),
                substitute_hp=s.substitute_hp.at[0,1,0].set(19))
            if move==5:s=s._replace(pp=s.pp.at[0,0,0].set(0))
            r=self.play(s).state
            self.assertEqual((int(r.hp[0,0,0]),int(r.hp[0,1,0]),int(r.substitute_hp[0,1,0])),(expected,1000,0))

    def test_zero_actual_damage_causes_no_recoil_on_immunity_protect_or_miss(self):
        s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(6),hp=self.live().hp.at[0,0,0].set(50))
        immune=s._replace(types=s.types.at[0,1,0,0].set(8))
        protected=s._replace(moves=s.moves.at[0,1,0,0].set(8))
        endure=s._replace(moves=s.moves.at[0,1,0,0].set(9),hp=s.hp.at[0,1,0].set(1))
        miss=self.table._replace(accuracy=self.table.accuracy.at[6].set(0),always_hit=self.table.always_hit.at[6].set(False))
        for state,table,foe_hp in ((immune,self.table,1000),(protected,self.table,1000),(s,miss,1000),(endure,self.table,1)):
            r=self.play(state,table).state
            self.assertEqual((int(r.hp[0,0,0]),int(r.hp[0,1,0])),(50,foe_hp))

    def test_recoil_double_ko_finishes_before_foe_action(self):
        j=self.j;s=self.live()._replace(moves=self.live().moves.at[0,0,0,0].set(6))
        s=s._replace(hp=j.zeros_like(s.hp).at[0,0,0].set(1).at[0,1,0].set(1))
        r=self.play(s).state
        self.assertEqual((r.hp[0,:,0].tolist(),int(r.winner[0]),int(r.pp[0,1,0,0])),([0,0],2,10))

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4)._replace(moves=self.live(4).moves.at[:,0,0,0].set(6))
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
    def test_pinned_fixed_and_recoil_events(self):
        r=subprocess.run([NODE,str(Path(__file__).with_name("gpu_damage_moves_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout),dict(amounts=[1,3,6,7,11,19],recoil=dict(
            struggle=[1,1,1,1,2,4],doubleedge=[1,1,2,2,3,6],takedown=[1,1,1,1,2,4],
            submission=[1,1,1,1,2,4],volttackle=[1,1,2,2,3,6]),fixed=dict(toss_rock=73,toss_ghost=0,
            shade_psychic=73,shade_dark=73,shade_normal=0,sonic_ghost=0,sonic_normal=20,dragon_steel=40),
            sub=dict(struggle=[46,1000,0],struggle_rockhead=[46,1000,0],
                doubleedge=[44,1000,0],doubleedge_rockhead=[50,1000,0])))


if __name__ == "__main__":unittest.main()
