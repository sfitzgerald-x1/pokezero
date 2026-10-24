"""Stage zero/one and armor fixtures; complete gameplay remains unqualified."""
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
class CriticalTests(unittest.TestCase):
    def setUp(self):
        import jax
        import jax.numpy as j
        from gpu_engine import critical,pursuit,transition
        from gpu_engine.state import empty_state
        self.jax,self.j,self.c,self.tr,self.empty=jax,j,critical,transition,empty_state
        meta={"vocabulary":{"moves":["_none","slash","splash","struggle","pursuit"],
            "abilities":["_none","battlearmor","shellarmor"]}}
        self.profile=critical.from_metadata(meta)
        self.table=transition.MoveTable(j.array([0,70,0,50,40]),j.zeros(5,j.int32),
            j.array([True,True,True,True,False]),j.full(5,100),j.ones(5,bool),
            j.zeros(5,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(3),
            pursuit=pursuit.from_metadata(meta),critical=self.profile)

    def live(self,batch=1):
        j=self.j;s=self.empty(batch,seed=83)
        return s._replace(stats=j.full_like(s.stats,200).at[:,0,:,5].set(300),
            hp=j.full_like(s.hp,1000),max_hp=j.full_like(s.hp,1000),
            moves=j.full_like(s.moves,2).at[:,0,0,0].set(1),pp=j.full_like(s.pp,10),
            level=j.full_like(s.level,100),done=j.zeros_like(s.done))

    def play(self,s,actions=None):
        return self.tr.step(s,self.j.zeros(s.active.shape,self.j.int32) if actions is None else actions,self.table)

    def test_all_sixteen_rolls_stage_zero_one_and_armor(self):
        j=self.j;roll=j.arange(16)
        for move,expected in ((2,1),(1,2),(4,1)):
            self.assertEqual(int(self.c.is_critical(self.profile,move,0,roll).sum()),expected)
            for armor in (1,2):
                self.assertFalse(bool(j.any(self.c.is_critical(self.profile,move,armor,roll))))

    def test_transition_uses_same_chance_column_and_preserves_keys(self):
        from gpu_engine import mechanics as m
        j=self.j;s=self.live(64)
        keys=self.jax.vmap(lambda k:self.jax.random.split(k,2))(s.keys)
        draws=self.jax.vmap(lambda k:self.jax.random.randint(k,(2,12),0,
            j.array([2,100,100,100,16,16,100,4,8,2,2,2])))(keys[:,1])
        self.assertTrue(bool(j.any(draws[:,0,4]==1)))
        r=self.play(s).state
        damage=m.gen3_damage(level=100,power=70,attack=200,defense=200,
            critical=draws[:,0,4]<2,typeless=True,roll=85+draws[:,0,5])
        self.assertEqual(r.hp[:,1,0].tolist(),(1000-damage).tolist())
        self.assertTrue(bool(j.array_equal(r.keys,keys[:,0])))
        for armor in (1,2):
            a=s._replace(ability=s.ability.at[:,1,0].set(armor))
            expected=m.gen3_damage(level=100,power=70,attack=200,defense=200,
                critical=False,typeless=True,roll=85+draws[:,0,5])
            self.assertEqual(self.play(a).state.hp[:,1,0].tolist(),(1000-expected).tolist())

    def test_pursuit_interception_checks_outgoing_armor_not_recipient(self):
        from gpu_engine import pursuit
        j=self.j;s=self.live(16)
        chances=j.zeros((16,12),j.int32).at[:,4].set(j.arange(16))
        # Interception precedes switching, so only the outgoing holder vetoes.
        armed=s._replace(ability=s.ability.at[:,1,0].set(2))
        r,used,_=pursuit.intercept(armed,0,j.zeros(16,j.int32),j.full(16,4),chances,
            self.table,eligible=j.ones(16,bool),contact_roll=j.full(16,59,j.int32))
        self.assertTrue(bool(j.all(used)))
        self.assertEqual(len(set(r.hp[:,1,0].tolist())),1)
        bench=s._replace(ability=s.ability.at[:,1,1].set(2))
        r,_,_=pursuit.intercept(bench,0,j.zeros(16,j.int32),j.full(16,4),chances,
            self.table,eligible=j.ones(16,bool),contact_roll=j.full(16,59,j.int32))
        self.assertLess(int(r.hp[0,1,0]),int(r.hp[1,1,0]))

    def test_jit_batch_prefix_and_invalid_terminal_identity(self):
        j=self.j;s=self.live(4);s=s._replace(done=s.done.at[3].set(True))
        actions=j.array([[0,0],[0,0],[9,0],[0,0]])
        a=self.jax.jit(self.tr.step)(s,actions,self.table);a.state.hp.block_until_ready()
        b=self.play(s,actions)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(b)):
            self.assertTrue(bool(j.array_equal(x,y)))
        small=self.jax.jit(self.tr.step)(self.jax.tree.map(lambda x:x[:2],s),actions[:2],self.table)
        for x,y in zip(self.jax.tree.leaves(a),self.jax.tree.leaves(small)):
            self.assertTrue(bool(j.array_equal(x[:2],y)))
        for x,y in zip(s,a.state):self.assertTrue(bool(j.array_equal(x[2:],y[2:])))

    @unittest.skipUnless(NODE and SHOWDOWN,"requires pinned Showdown and Node")
    def test_pinned_catalog_chance_and_armor_veto(self):
        p=subprocess.run([NODE,str(Path(__file__).with_name("gpu_critical_oracle.cjs")),
            str(Path(SHOWDOWN).resolve())],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        result=json.loads(p.stdout)
        self.assertEqual(result["high_ids"],sorted(self.c.HIGH_CRIT))
        self.assertEqual(result["ratios"],[2]*12)
        catalog=self.c.from_metadata({"vocabulary":{"moves":["_none"]+result["high_ids"]+["tackle"],
            "abilities":["_none"]}})
        self.assertEqual(catalog.high.tolist(),[False]+[True]*12+[False])
        for name,den,crit in (("ordinary",16,True),("high",8,True),("battle",8,False),("shell",16,False)):
            self.assertEqual(result[name]["draws"],[[1,den]])
            self.assertEqual(result[name]["crit"],crit)
        self.assertGreater(result["high"]["damage"],result["battle"]["damage"])


if __name__=="__main__":unittest.main()
