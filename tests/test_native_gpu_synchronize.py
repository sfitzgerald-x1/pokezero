"""Complete authored Synchronize laws against the retained Rust engine."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires isolated JAX and native oracle')
class NativeSynchronizeTests(unittest.TestCase):
    def setUp(self):
        from tests.test_gpu_engine_synchronize import SynchronizeTests
        f=SynchronizeTests();f.setUp();self.f=f
        import json
        from pathlib import Path
        from gpu_engine.transition import tables_from_metadata
        self.meta=json.loads((Path(__file__).parent/'data/gpu_synchronize_catalog.json').read_text())['metadata']
        self.table=tables_from_metadata(self.meta)

    def compare(self,side,move,*,compound=False,source_lum=False,target_lum=False,
                switching=False,wrong_table=False):
        f=self.f;j=f.j;s=f.live(1,side,move)
        s=s._replace(moves=j.broadcast_to(j.array([f.id(n) for n in ('splash','firepunch','icebeam','thunderwave')]),s.moves.shape).at[0,side,0,0].set(f.id(move)),species=j.broadcast_to(j.arange(1,7),(1,2,6)),types=s.types.at[...,0].set(1),
            stats=s.stats.at[...,0].set(1000),item=s.item.at[:,side,0].set(int(source_lum)).at[:,1-side,0].set(int(target_lum)))
        if compound:
            a=s.ability.at[:,side,0].set(f.ability('synchronize')).at[:,1-side,0].set(f.ability('static'))
            s=s._replace(ability=a,base_ability=a)
        actions=[0,0]
        if switching:
            a=s.ability.at[:,1-side,0].set(0).at[:,1-side,1].set(f.ability('synchronize'))
            s=s._replace(ability=a,base_ability=a);actions[1-side]=4
        table=self.table
        if wrong_table:table=table._replace(secondary=table.secondary._replace(
            synchronize=j.zeros_like(table.secondary.synchronize)))
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        plan=c.plan_for_state(s,0,tuple(actions),table,self.meta,cap=131072)
        self.assertIsInstance(plan,c.ConsumerPlan,str(getattr(plan,'gaps',None)))
        r=compare_slot(plan,JaxCaseExecutor(s,tuple(actions),table,plan,batch_size=64),
            lambda:native_executor(plan),batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(r['status'],'DIVERGE' if wrong_table else 'AGREE',
            {k:v for k,v in r.items() if k not in ('outcomes','consumer_certificate')})
        return r

    def test_complete_secondary_laws_and_both_lum_paths(self):
        for side in (0,1):
            for source_lum,target_lum in [(False,False),(True,False),(False,True)]:
                with self.subTest(side=side,source_lum=source_lum,target_lum=target_lum):
                    self.compare(side,'thunderbolt',source_lum=source_lum,target_lum=target_lum)

    def test_primary_toxic_and_compound_contact_laws(self):
        for side in (0,1):
            with self.subTest(side=side,move='toxic'):self.compare(side,'toxic')
            with self.subTest(side=side,move='bodyslam'):self.compare(side,'bodyslam',compound=True,target_lum=True)

    def test_switched_in_synchronize_complete_ten_percent_law_both_seats(self):
        for side in (0,1):
            r=self.compare(side,'thunderbolt',switching=True)
            self.assertEqual(r['identified_mass'],'1')

    def test_wrong_synchronize_table_is_a_full_law_divergence(self):
        r=self.compare(0,'thunderbolt',switching=True,wrong_table=True)
        self.assertAlmostEqual(float(r['half_l1']),0.1)
