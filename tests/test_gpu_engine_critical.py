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

    def test_stick_ratio_threshold_and_species(self):
        """Stick (critical.ratio): 1 + high-crit + 2 for a Farfetch'd holding it; thresholds 1, 2, 4 on the [0,16) draw
        (ratio 4, 1/3, reads 5: documented as not exact and refused by the planners; ratio 5 reads 8, exact)."""
        j=self.j
        meta={"vocabulary":{"moves":["_none","return","slash"],"abilities":["_none","battlearmor"],
            "items":["_none","stick","leftovers"],"species":["_none","farfetchd","mew"]}}
        t=self.c.from_metadata(meta);roll=j.arange(16)
        self.assertEqual(self.c.CRIT_THRESHOLD,(0,1,2,4,5,8))
        for move,item,species,ratio,count in ((1,0,1,1,1),(1,1,1,3,4),(1,1,2,1,1),(1,2,1,1,1),(2,0,1,2,2),(2,1,1,4,5),
                                               (2,1,2,2,2)):
            self.assertEqual(int(self.c.ratio(t,move,item=j.asarray(item),species=j.asarray(species))),ratio)
            hits=self.c.is_critical(t,move,0,roll,item=j.asarray(item),species=j.asarray(species))
            self.assertEqual(int(hits.sum()),count,(move,item,species))
            self.assertEqual(hits.tolist(),[r<count for r in range(16)])
            self.assertFalse(bool(j.any(self.c.is_critical(t,move,1,roll,item=j.asarray(item),species=j.asarray(species)))))
        # Without the attacker's item (Pursuit's intercept) there is no Stick; a table without the vocabularies has none.
        self.assertEqual(int(self.c.is_critical(t,1,0,roll).sum()),1)
        bare=self.c.from_metadata({"vocabulary":{"moves":["_none","return"],"abilities":["_none"]}})
        self.assertEqual(bare.stick.tolist(),[False]);self.assertEqual(bare.farfetchd.tolist(),[False])

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
        # Stick: Farfetch'd's Return at critRatio 3 (1/4), Slash at 4 (1/3); another holder and a bare Farfetch'd at 1.
        for name,den in (("stick",4),("stick_high",3),("stick_other",16),("farfetchd",16)):
            self.assertEqual(result[name]["draws"],[[1,den]],name)


@unittest.skipUnless(SHOWDOWN,"requires the pinned Showdown checkout")
class StickReachTests(unittest.TestCase):
    """No Gen 3 randbats state reaches critRatio 4 or 5, which the [0,16) draw cannot hold exactly (ratio 4) or which
    the planners refuse: a pool change that makes one reachable trips this. Pure Python over the pinned sources."""
    def test_no_pool_set_reaches_ratio_four(self):
        import re
        from gpu_engine import critical as c
        root=Path(SHOWDOWN)
        sets=json.loads((root/"data/random-battles/gen3/sets.json").read_text())
        teams=(root/"data/random-battles/gen3/teams.ts").read_text()
        # Only getItem's Farfetch'd branch hands out Stick, and no crit-ratio item at all otherwise.
        self.assertEqual(re.findall(r".*'Stick'.*",teams),["\t\tif (species.id === 'farfetchd') return 'Stick';"])
        for item in ("Scope Lens","Razor Claw","Lucky Punch","Leek"):self.assertNotIn(item,teams)
        # Farfetch'd's own moves: no high-crit move and nothing that calls or copies one (Sleep Talk, Metronome, Mimic,
        # Mirror Move, Assist, Nature Power, Sketch; Encore only repeats a move it has).
        callers={"sleeptalk","metronome","mimic","mirrormove","assist","naturepower","sketch","transform"}
        moves={m for s in sets["farfetchd"]["sets"] for m in s["movepool"]}
        self.assertFalse(moves&set(c.HIGH_CRIT));self.assertFalse(moves&callers)
        # +2 from Focus Energy (its own, or Baton Passed by any holder) or a copied focusenergy volatile: no pool set has it.
        everything={m for v in sets.values() for s in v["sets"] for m in s["movepool"]}
        self.assertFalse(everything&{"focusenergy"})
        # A Transformed Ditto is species farfetchd but keeps Ditto's own item, never Stick (getItem above).
        self.assertIn("farfetchd",sets);self.assertEqual(len(sets["farfetchd"]["sets"]),1)


if __name__=="__main__":unittest.main()
