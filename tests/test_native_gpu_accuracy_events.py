"""Complete authored shared-accuracy laws against current native bindings."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires isolated JAX and native oracle')
class NativeAccuracyEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_gpu_engine_accuracy_events import AccuracyEventTests
        AccuracyEventTests.setUpClass();cls.fixture=AccuracyEventTests();cls.f=cls.fixture.f

    def compare(self,side,move,*,source='_none',target='_none',weather=0,
                accuracy=0,evasion=0,switching=False,wrong_table=False):
        f=self.f;j=f.j
        s=self.fixture.live(side,move,self.fixture.ability(source),self.fixture.ability(target),weather)
        s=f.jax.tree.map(lambda x:x[:1],s)
        spare=[n for n in ('splash','toxic','doubleedge','aerialace') if n!=move][:3]
        moves=j.broadcast_to(j.array([f.id(n) for n in ('splash','toxic','doubleedge','aerialace')]),s.moves.shape)
        moves=moves.at[:,side,0].set(j.array([f.id(n) for n in (move,*spare)]))
        s=s._replace(moves=moves,boosts=s.boosts.at[0,side,0,5].set(accuracy).at[0,1-side,0,6].set(evasion))
        actions=[0,0]
        if switching:actions[1-side]=4
        table=f.table
        if wrong_table:table=table._replace(combat=table.combat._replace(compound_eyes=j.zeros_like(table.combat.compound_eyes)))
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        plan=c.plan_for_state(s,0,tuple(actions),table,f.metadata,cap=131072)
        self.assertIsInstance(plan,c.ConsumerPlan,getattr(plan,'gaps',()))
        out=compare_slot(plan,JaxCaseExecutor(s,tuple(actions),table,plan,batch_size=64),
            lambda:native_executor(plan),batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
        self.assertEqual(out['status'],'DIVERGE' if wrong_table else 'AGREE',
            {k:v for k,v in out.items()if k not in ('outcomes','consumer_certificate')})
        return out

    def test_complete_type_status_and_composed_modifier_laws_both_seats(self):
        for side in (0,1):
            for source,target,weather in [('hustle','_none',0),('compoundeyes','_none',0),
                    ('_none','sandveil',3),('hustle','sandveil',3),('compoundeyes','sandveil',3)]:
                with self.subTest(side=side,source=source,target=target):
                    self.compare(side,'toxic',source=source,target=target,weather=weather)

    def test_staged_hidden_power_and_plain_ieee_boundary_laws(self):
        for side in (0,1):
            for move,ability,a,e,weather in [('hiddenpowerground','hustle',-4,0,0),
                    ('hiddenpowerice','hustle',-4,0,0),('hydropump','_none',2,5,0),
                    ('thunder','_none',-6,6,1)]:
                with self.subTest(side=side,move=move):
                    self.compare(side,move,source=ability,accuracy=a,evasion=e,weather=weather)

    def test_switching_pursuit_and_ordinary_accuracy_both_seats(self):
        for side in (0,1):
            for switching in (False,True):
                with self.subTest(side=side,switching=switching):
                    self.compare(side,'pursuit',target='sandveil',weather=3,
                        accuracy=-6,evasion=6,switching=switching)

    def test_wrong_compound_eyes_table_is_a_complete_law_divergence(self):
        out=self.compare(0,'toxic',source='compoundeyes',wrong_table=True)
        self.assertAlmostEqual(float(out['half_l1']),.15,places=6)
