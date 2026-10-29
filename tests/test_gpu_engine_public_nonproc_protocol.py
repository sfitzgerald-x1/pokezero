import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class NonprocProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_gpu_engine_public_item_context import PublicItemContextTests
        from gpu_engine import public_nonproc_protocol as protocol
        PublicItemContextTests.setUpClass();cls.f=PublicItemContextTests();cls.p=protocol;cls.j=cls.f.j;cls.jax=cls.f.jax
        cls.before,_=cls.f.project(cls.f.prefix)

    def test_lexical_hp_proc_phase_and_upkeep_codec(self):
        from pokezero.belief import _hp_snapshot_action
        cases=('|','| ','|turn|3','|move|p1a: Snorlax|Surf|p2a: Absol',
            '|-damage|p1a: Snorlax|20/100','|-damage|p1a: Snorlax|20/100|[from] ability: Liquid Ooze',
            '|-heal|p1a: Snorlax|70/100|[silent]','|-heal|p1a: Snorlax|90/100|[from] item: Leftovers',
            '|-enditem|p1a: Snorlax|Sitrus Berry|[eat]','|upkeep')
        names={'keep':0,'update':1,'discard':2}
        for line in cases:
            p=self.p.project_line(line,self.before)
            facts,ok=self.jax.jit(self.p.unpack)(self.j.asarray([p.payload],self.j.int32))
            self.assertTrue(bool(ok.all()),line);self.assertFalse(p.full_line_consumers_complete)
            if line.startswith(('|-damage|','|-heal|')):
                self.assertEqual(int(facts.action_phase_hp[0]),names[_hp_snapshot_action(line,in_residual_phase=False)])
                self.assertEqual(int(facts.residual_phase_hp[0]),names[_hp_snapshot_action(line,in_residual_phase=True)])
            if '[eat]' in line:self.assertTrue(bool(facts.berry_eat[0]))
        line='|switch|p1a: Gardevoir|Gardevoir, L80|100/100'
        after,_=self.f.project([line],state=self.before)
        p=self.p.project_line(line,self.before,after);self.assertEqual(p.payload[:4],(1,0,1,2))
        with self.assertRaisesRegex(ValueError,'candidate'):self.p.project_line(line,self.before)
        with self.assertRaisesRegex(ValueError,'private'):self.p.project_line('|request|{}',self.before)

    def test_typed_unknown_bits_targets_and_batch_isolation(self):
        projection=self.p.project_line('|-heal|p1a: Snorlax|90/100|[from] item: Leftovers',self.before)
        payload=self.j.repeat(self.j.asarray([projection.payload],self.j.int32),8,axis=0)
        bad=payload.at[0,0].set(7).at[1,6].set(64).at[2,2].set(-1).at[3,7].set(2)
        _,ok=self.jax.jit(self.p.unpack)(bad);self.assertEqual(ok.tolist(),[False]*4+[True]*4)
        with self.assertRaisesRegex(ValueError,'int32'):self.p.unpack(payload.astype(self.j.float32))
        with self.assertRaisesRegex(ValueError,'B by8'):self.p.unpack(self.j.zeros((8,7),self.j.int32))
        with self.assertRaisesRegex(ValueError,'exceeds int32'):
            self.p.project_line('|-damage|p1a: Snorlax|1/2147483648',self.before)

    def test_exact_registered_arrival_candidate_and_opcode_phase_coherence(self):
        from dataclasses import replace
        line='|switch|p1a: Gardevoir|Gardevoir, L80|100/100'
        after,_=self.f.project([line],state=self.before)
        for raw,candidate in ((line,self.before),(line.replace('p1a: Gardevoir','p1a: OTHER'),after),
                (line.replace('|Gardevoir,','|Snorlax,'),after),
                (line,replace(after,species_binding=tuple(v^1 for v in after.species_binding)))):
            with self.assertRaisesRegex(ValueError,'candidate'):self.p.project_line(raw,self.before,candidate)
        good=self.p.project_line(line,self.before,after)
        for phase in (0,2):
            payload=list(good.payload);payload[6]=phase
            _,ok=self.p.unpack(self.j.asarray([payload],self.j.int32));self.assertEqual(ok.tolist(),[False])
        hp=self.p.project_line('|-damage|p1a: Snorlax|20/100',self.before)
        for phase in (1,2):
            payload=list(hp.payload);payload[6]=(payload[6]&~3)|phase
            _,ok=self.p.unpack(self.j.asarray([payload],self.j.int32));self.assertEqual(ok.tolist(),[False])
        berry=list(self.p.project_line('|-enditem|p1a: Snorlax|Sitrus Berry|[eat]',self.before).payload);berry[6]=1
        _,ok=self.p.unpack(self.j.asarray([berry],self.j.int32));self.assertEqual(ok.tolist(),[False])
