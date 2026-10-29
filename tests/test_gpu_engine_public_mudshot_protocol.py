import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class MudShotProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_item_context import PublicItemContextTests
        from gpu_engine import public_mudshot_protocol as protocol,public_mudshot
        PublicItemContextTests.setUpClass();cls.f=PublicItemContextTests();cls.p,cls.m=protocol,public_mudshot
        cls.j,cls.jax=cls.f.j,cls.f.jax;cls.before,_=cls.f.project(cls.f.prefix)

    def test_called_moves_clean_damage_and_other_unboost_match_source_controls(self):
        cases=(('|move|p1a: OLD|Mud Shot|p2a: Absol',self.m.RESOLVE,True,False),
            ('|move|p1a: OLD|Mud Shot|p2a: Absol|[from] move: Sleep Talk',self.m.RESOLVE,True,False),
            ('|move|p2a: Absol|Surf|p1a: Snorlax',self.m.RESOLVE,False,False),
            ('|-damage|p2a: OLD|80/100',self.m.DAMAGE,False,False),
            ('|-damage|p2a: OLD|80/100|[from] psn',self.m.DAMAGE,False,True),
            ('|-unboost|p2a: OLD|def|1',self.m.UNBOOST,False,False),
            ('|faint|p2a: OLD',self.m.FAINT,False,False),('|upkeep',self.m.RESOLVE,False,False),
            ('|-heal|p2a: Absol|100/100',self.p.NOTE,False,False))
        for line,op,queue,tagged in cases:
            projection=self.p.project_line(line,self.before)
            facts,ok=self.jax.jit(self.p.unpack)(self.j.asarray([projection.payload],self.j.int32))
            self.assertTrue(bool(ok.all()),line);self.assertFalse(projection.full_line_consumers_complete)
            self.assertEqual((int(facts.operation[0]),bool(facts.queue[0]),bool(facts.tagged[0])),(op,queue,tagged))
            if queue or op in (self.m.DAMAGE,self.m.UNBOOST,self.m.FAINT):
                self.assertEqual((int(facts.side[0]),int(facts.slot[0]),int(facts.encounter[0])),(1,0,1))
        with self.assertRaisesRegex(ValueError,'private'):self.p.project_line('|request|{}',self.before)
        with self.assertRaisesRegex(ValueError,'canonical'):self.p.project_line('|upkeep|extra',self.before)

    def test_absent_target_and_heterogeneous_typed_refusals(self):
        from dataclasses import replace
        missing=replace(self.before,live=(True,False))
        projection=self.p.project_line('|move|p1a: Snorlax|Mud Shot|p2a: Absol',missing)
        facts,ok=self.p.unpack(self.j.asarray([projection.payload],self.j.int32))
        self.assertEqual(ok.tolist(),[True]);self.assertEqual(facts.side.tolist(),[-1]);self.assertEqual(facts.queue.tolist(),[True])
        good=self.j.asarray([self.p.project_line('|-damage|p2a: Absol|80/100',self.before).payload],self.j.int32)
        rows=self.j.repeat(good,8,axis=0).at[0,0].set(0).at[1,4].set(1).at[2,6].set(1).at[3,2].set(-1)
        _,ok=self.jax.jit(self.p.unpack)(rows);self.assertEqual(ok.tolist(),[False]*4+[True]*4)
        with self.assertRaisesRegex(ValueError,'int32'):self.p.unpack(rows.astype(self.j.float32))
