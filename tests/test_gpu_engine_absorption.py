"""Authored absorption fixtures; no move/game coverage qualification."""
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
class AbsorptionTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import absorption as a,mechanics as m
        self.jax,self.j,self.a,self.m = jax,j,a,m
        self.moves = ["_none","surf","thunderbolt","thunderwave","flamethrower","willowisp","hiddenpowerfire","raindance",
            "splash","struggle","protect"]
        self.abilities = ["_none","waterabsorb","voltabsorb","flashfire"]
        self.types = ["_none","normal","fire","water","electric","fighting","flying","poison",
            "ground","rock","bug","ghost","steel","grass","psychic","ice","dragon","dark"]
        self.meta = {"vocabulary":{"moves":self.moves,"abilities":self.abilities,"types":self.types},
            "moves":[None]+[{"source_type":typ,"category":cat} for typ,cat in
                [("Water","Special"),("Electric","Special"),("Electric","Status"),
                 ("Fire","Special"),("Fire","Status"),("Normal","Special"),("Water","Status"),
                 ("Normal","Status"),("???","Physical"),("Normal","Status")]]}
        self.table = a.from_metadata(self.meta)

    def hit(self,ability,move,typ,*,status=0,hp=50,max_hp=101,sub=0,volatile=0,eligible=True,types=(1,0)):
        j = self.j
        return self.a.try_hit(self.table,j.array([ability]),j.array([move]),j.array([typ]),
            j.array([status]),j.array([types]),j.array([hp]),j.array([max_hp]),j.array([sub]),
            j.array([volatile],j.uint32),eligible=j.array([eligible]))

    def test_water_and_volt_heal_floor_quarter_even_behind_substitute(self):
        for ability,move,typ in ((1,1,3),(2,2,4)):
            r = self.hit(ability,move,typ,sub=25)
            self.assertEqual((r.blocked.tolist(),r.hp.tolist()),([True],[75]))
            self.assertEqual(self.hit(ability,move,typ,hp=101).blocked.tolist(),[True])
            self.assertEqual(self.hit(ability,move,typ,hp=1,max_hp=1).hp.tolist(),[1])

    def test_volt_absorb_does_not_block_thunderwave_in_gen3(self):
        r = self.hit(2,3,4)
        self.assertEqual((r.blocked.tolist(),r.hp.tolist()),([False],[50]))

    def test_flash_fire_does_not_activate_while_frozen_or_on_failed_wisp(self):
        for status,sub,types in ((self.m.FRZ,0,(1,0)),(self.m.PAR,0,(1,0)),(0,25,(1,0)),(0,0,(2,0))):
            r = self.hit(3,5,2,status=status,sub=sub,types=types)
            self.assertEqual((r.blocked.tolist(),r.volatile.tolist()),([False],[0]))
        self.assertEqual(self.hit(3,4,2,status=self.m.FRZ).blocked.tolist(),[False])

    def test_flash_fire_applies_to_eligible_wisp_and_dynamic_fire(self):
        for move in (4,5,6):
            r = self.hit(3,move,2)
            self.assertEqual((r.blocked.tolist(),r.hp.tolist(),r.volatile.tolist()),([True],[50],[2]))
        r = self.hit(3,4,2,volatile=3)
        self.assertEqual((r.blocked.tolist(),r.volatile.tolist()),([True],[3]))

    def test_ineligible_dead_and_wrong_type_do_not_activate(self):
        for args in (dict(eligible=False),dict(hp=0)):
            r = self.hit(1,1,3,**args)
            self.assertEqual((r.blocked.tolist(),r.volatile.tolist()),([False],[0]))
        self.assertEqual(self.hit(1,2,4).blocked.tolist(),[False])

    def test_flash_multiplier_and_original_fire_thaw_catalog_are_separate(self):
        n,d = self.a.damage_phase1(self.table,self.j.array([2,2,3]),self.j.array([0,2,2],self.j.uint32))
        self.assertEqual((n.tolist(),d.tolist()),([1,3,1],[1,2,1]))
        self.assertEqual(self.table.fire_thaws.tolist(),[False,False,False,False,True,False,False,False,False,False,False])

    def battle(self,batch=1):
        from gpu_engine import effects,transition
        from gpu_engine.state import empty_state
        j = self.j
        meta = {**self.meta,"moves":[None]+[{**row,"flags":{"protect":1}}
            for row in self.meta["moves"][1:]]}
        size = len(self.moves)
        table = transition.MoveTable(j.array([0,95,95,0,95,0,70,0,0,50,0]),
            j.array([0,3,4,4,2,2,2,3,1,0,1]),j.array([False]*9+[True,False]),
            j.full(size,100),j.ones(size,bool),j.array([0]*10+[3]),
            j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(9),
            effects=effects.from_metadata(meta),absorption=self.table)
        s = empty_state(batch,seed=71)
        stats = j.full_like(s.stats,200).at[...,0].set(101).at[:,0,:,5].set(300)
        s = s._replace(stats=stats,level=j.full_like(s.level,100),hp=j.full_like(s.hp,101),
            max_hp=j.full_like(s.hp,101),pp=j.full_like(s.pp,10),moves=j.full_like(s.moves,8),
            types=s.types.at[...,0].set(1),done=j.zeros_like(s.done))
        return s,table,transition

    def test_step_absorbs_before_substitute_but_after_protect_or_miss(self):
        j = self.j
        s,t,tr = self.battle()
        s = s._replace(moves=s.moves.at[0,0,0,0].set(1),ability=s.ability.at[0,1,0].set(1),
            hp=s.hp.at[0,1,0].set(50),substitute_hp=s.substitute_hp.at[0,1,0].set(25))
        r = tr.step(s,j.array([[0,0]]),t).state
        self.assertEqual((int(r.hp[0,1,0]),int(r.substitute_hp[0,1,0]),int(r.pp[0,0,0,0])),(75,25,9))
        protects = s._replace(moves=s.moves.at[0,1,0,0].set(10))
        self.assertEqual(int(tr.step(protects,j.array([[0,0]]),t).state.hp[0,1,0]),50)
        miss = t._replace(accuracy=t.accuracy.at[1].set(0),always_hit=t.always_hit.at[1].set(False))
        self.assertEqual(int(tr.step(s,j.array([[0,0]]),miss).state.hp[0,1,0]),50)

    def test_step_thunderwave_bypasses_volt_absorb_and_wisp_triggers_flash_fire(self):
        j = self.j
        for move,ability,status,volatile in ((3,2,self.m.PAR,0),(5,3,self.m.NONE,2)):
            s,t,tr = self.battle()
            s = s._replace(moves=s.moves.at[0,0,0,0].set(move),ability=s.ability.at[0,1,0].set(ability))
            r = tr.step(s,j.array([[0,0]]),t).state
            self.assertEqual((int(r.status[0,1,0]),int(r.volatile[0,1,0]),int(r.hp[0,1,0])),(status,volatile,101))

    def test_field_water_move_does_not_target_absorber(self):
        s,t,tr = self.battle()
        s = s._replace(moves=s.moves.at[0,0,0,0].set(7),ability=s.ability.at[0,1,0].set(1),hp=s.hp.at[0,1,0].set(50))
        self.assertEqual(int(tr.step(s,self.j.array([[0,0]]),t).state.hp[0,1,0]),50)

    def test_flash_boost_persists_between_turns_and_clears_on_switch(self):
        j = self.j
        s,t,tr = self.battle(8)
        s = s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            moves=s.moves.at[:,0,0,0].set(4),ability=s.ability.at[:,0,0].set(3))
        actions = j.zeros((8,2),j.int32)
        base = tr.step(s,actions,t).state
        boosted = s._replace(volatile=s.volatile.at[:,0,0].set(2))
        r = tr.step(boosted,actions,t).state
        self.assertTrue(bool(j.all(r.hp[:,1,0] < base.hp[:,1,0])))
        self.assertEqual(r.volatile[:,0,0].tolist(),[2]*8)
        switched = tr.step(boosted,j.tile(j.array([[4,0]]),(8,1)),t).state
        self.assertEqual(switched.volatile[:,0,0].tolist(),[0]*8)

    def test_original_fire_thaws_frozen_flash_holder_but_hidden_power_does_not(self):
        j = self.j
        s,t,tr = self.battle(64)
        s = s._replace(hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            ability=s.ability.at[:,1,0].set(3),status=s.status.at[:,1,0].set(self.m.FRZ))
        split = self.jax.vmap(lambda k:self.jax.random.split(k,2))(s.keys)
        draws = self.jax.vmap(lambda k:self.jax.random.randint(k,(2,12),0,
            j.array([2,100,100,100,16,16,100,4,8,2,2,2])))(split[:,1])
        no_natural_thaw = draws[:,1,1] >= 20
        self.assertTrue(bool(j.any(no_natural_thaw)))
        for move,expected in ((4,self.m.NONE),(6,self.m.FRZ)):
            state = s._replace(moves=s.moves.at[:,0,0,0].set(move))
            r = tr.step(state,j.zeros((64,2),j.int32),t).state
            self.assertTrue(bool(j.all(r.hp[:,1,0] < 1000)))
            self.assertTrue(bool(j.all(r.status[:,1,0][no_natural_thaw] == expected)))
            self.assertTrue(bool(j.all(r.volatile[:,1,0] == 0)))

    def test_step_jit_batch_prefix_and_invalid_terminal_identity(self):
        j = self.j
        s,t,tr = self.battle(4)
        s = s._replace(moves=s.moves.at[:,0,0,0].set(1),ability=s.ability.at[:,1,0].set(1),
            hp=s.hp.at[:,1,0].set(50),done=s.done.at[3].set(True))
        actions = j.array([[0,0],[0,0],[9,0],[0,0]])
        fn = self.jax.jit(tr.step)
        compiled = fn(s,actions,t); compiled.state.hp.block_until_ready()
        eager = tr.step(s,actions,t)
        for left,right in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(eager)):
            self.assertTrue(bool(j.array_equal(left,right)))
        small = fn(self.jax.tree.map(lambda x:x[:2],s),actions[:2],t)
        for left,right in zip(self.jax.tree.leaves(compiled),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(left[:2],right)))
        for old,new in zip(s,compiled.state): self.assertTrue(bool(j.array_equal(old[2:],new[2:])))

    def test_jit_and_batch_prefix_match_eager(self):
        j = self.j
        def fn(hp):
            b = len(hp)
            return self.a.try_hit(self.table,j.ones(b,j.int32),j.ones(b,j.int32),j.full(b,3),
                j.zeros(b,j.int32),j.tile(j.array([[1,0]]),(b,1)),hp,j.full(b,101),
                j.full(b,25),j.zeros(b,j.uint32),eligible=j.ones(b,bool))
        compiled = self.jax.jit(fn)(j.arange(8)); compiled.hp.block_until_ready()
        eager = fn(j.arange(8))
        for left,right in zip(compiled,eager): self.assertTrue(bool(j.array_equal(left,right)))
        for left,right in zip(compiled,fn(j.arange(3))): self.assertTrue(bool(j.array_equal(left[:3],right)))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_gen3_absorption_handlers(self):
        rows = []
        expected = []
        for ability,move,typ,status,sub,types in (
            (1,1,3,0,25,(1,0)),(2,2,4,0,0,(1,0)),(2,3,4,0,0,(1,0)),
            (3,4,2,0,0,(1,0)),(3,4,2,self.m.FRZ,0,(1,0)),
            (3,5,2,0,0,(1,0)),(3,5,2,self.m.PAR,0,(1,0)),
            (3,5,2,0,25,(1,0)),(3,5,2,0,0,(2,0)),(3,6,2,0,0,(1,0))):
            r = self.hit(ability,move,typ,status=status,sub=sub,types=types)
            expected.append({"blocked":bool(r.blocked[0]),"hp":int(r.hp[0]),"flash":bool(r.volatile[0]&2)})
            rows.append({"ability":self.abilities[ability],"move":self.moves[move],
                "type":self.types[typ].capitalize(),"status":["","par","brn","psn","tox","slp","frz"][status],
                "sub":sub,"types":[self.types[t].capitalize() for t in types],"hp":50,"max_hp":101})
        result = subprocess.run([NODE,str(Path(__file__).with_name("gpu_absorption_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],input=json.dumps(rows),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        doc = json.loads(result.stdout)
        self.assertEqual(doc["results"],expected)
        self.assertEqual(doc["phase1"],[1.5,1])


if __name__ == "__main__": unittest.main()
