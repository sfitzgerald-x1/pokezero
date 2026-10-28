"""Complete authored Hustle laws; pinned-source boundary controls are separate."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires isolated JAX and native oracle')
class NativeHustleTests(unittest.TestCase):
    def compare(self,side,move,*,band=False,switching=False,wrong_table=False):
        from tests.test_gpu_engine_hustle import HustleTests
        f=HustleTests();f.setUpClass();j=f.j;s=f.live(side,move,batch=1)
        spare=[n for n in ('splash','toxic','doubleedge','aerialace') if n!=move][:3]
        moves=j.broadcast_to(j.array([f.id(n) for n in ('splash','toxic','doubleedge','aerialace')]),s.moves.shape)
        moves=moves.at[:,side,0].set(j.array([f.id(n) for n in (move,*spare)]))
        s=s._replace(item=s.item.at[:,side,0].set(int(band)),moves=moves)
        actions=[0,0]
        if switching:actions[1-side]=4
        table=f.table
        if wrong_table:table=table._replace(combat=table.combat._replace(hustle=j.zeros_like(table.combat.hustle)))
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        plan=c.plan_for_state(s,0,tuple(actions),table,f.metadata,cap=131072)
        self.assertIsInstance(plan,c.ConsumerPlan,str(getattr(plan,'gaps',None)))
        out=compare_slot(plan,JaxCaseExecutor(s,tuple(actions),table,plan,batch_size=64),
            lambda:native_executor(plan),batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(out['status'],'DIVERGE' if wrong_table else 'AGREE',
            {k:v for k,v in out.items() if k not in ('outcomes','consumer_certificate')})
        return out

    def test_complete_physical_hidden_power_choice_band_both_seats(self):
        for side in (0,1):
            for band in (False,True):
                with self.subTest(side=side,band=band):self.compare(side,'hiddenpowerground',band=band)

    def test_special_hidden_power_both_seats(self):
        for side in (0,1):
            with self.subTest(side=side):self.compare(side,'hiddenpowerice',band=True)

    def test_always_hit_full_law_bypasses_numeric_events_both_seats(self):
        for side in (0,1):
            self.compare(side,'aerialace',band=True)

    def test_switched_target_and_wrong_hustle_table(self):
        for side in (0,1):self.compare(side,'hiddenpowerground',band=True,switching=True)
        out=self.compare(0,'hiddenpowerground',band=True,wrong_table=True)
        self.assertGreater(float(out['half_l1']),0)
