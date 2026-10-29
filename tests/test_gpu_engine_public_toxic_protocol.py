"""Raw Toxic transport retains parser grammar and device rollback."""
import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicToxicProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_toxic as fixtures
        fixtures.PublicToxicTests.setUpClass.__func__(cls)
        from gpu_engine import public_toxic_protocol as p
        cls.p=p

    initial=__import__('test_gpu_engine_public_toxic').PublicToxicTests.initial
    compare=__import__('test_gpu_engine_public_toxic').PublicToxicTests.compare

    def consume(self,state,line):
        projection=self.p.project_line(line,identity_ids=self.ids)
        fact,ok=self.jax.jit(self.p.unpack)(self.j.array([projection.side],self.j.int32),
            self.j.array([projection.payload],self.j.int32))
        self.assertEqual(ok.tolist(),[True])
        state,admitted=self.t.apply(state,self.tables,fact,producer_binding=self.tables.binding)
        self.assertTrue(bool(admitted.all()));return state,projection

    def test_all_raw_prefixes_preserve_actual_parser_proofs_and_residuals(self):
        lines=[*self.prefix,'|turn|1','|-status|p1a: Snorlax|tox',
            '|-damage|p1a: Snorlax|94/100 tox|[from] psn','|faint|p1a: Snorlax','|upkeep',
            '|request|'+json.dumps({'side':{'id':'p1'}}),
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox','|turn|2',
            '|-damage|p1a: Gardevoir|94/100 tox|[from] psn','|t:|123','|',
            '|upkeep\r','|turn','|turn|3','|-cureteam|p1a: Gardevoir']
        state=self.initial(visibility=(2,2));parser=self.reference._ReplayParser(complete_prefix=True,
            hp_visibility={'p1':'percentage','p2':'percentage'})
        for line in lines:
            state,p=self.consume(state,line);parser._feed_line(line);self.compare(state,parser)
            self.assertEqual(p.raw_line,line);self.assertFalse(p.full_line_consumers_complete)
            if line in ('|upkeep\r','|turn'):self.assertFalse(p.shared_event)
            if line=='|t:|123':self.assertTrue(p.timestamp)

    def test_forged_words_and_reserved_bits_refuse_in_mixed_batches(self):
        p=self.p.project_line('|turn|1',identity_ids=self.ids)
        good=self.j.array([p.payload,p.payload],self.j.int32);side=self.j.array([-1,-1],self.j.int32)
        fn=self.jax.jit(self.p.unpack)
        for column,value in ((0,99),(1,7),(2,-1),(3,-2),(4,-1),(5,128),(6,1),(7,1)):
            _,ok=fn(side,good.at[0,column].set(value));self.assertEqual(ok.tolist(),[False,True])
        _,ok=fn(side,good.at[0,5].set(int(p.payload[5])&~(1<<5)))
        self.assertEqual(ok.tolist(),[False,True])
        with self.assertRaises(ValueError):self.p.unpack(side,good.astype(self.j.float32))


if __name__=='__main__':unittest.main()
