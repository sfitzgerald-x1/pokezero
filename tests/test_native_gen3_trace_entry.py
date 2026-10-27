"""Complete authored ordinary successor comparisons for Trace entry."""
import importlib.util
import unittest


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('jax','poke_engine')),
    'requires actual native binding and isolated JAX')
class NativeTraceEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_gpu_engine_trace_entry import TraceEntryTests
        TraceEntryTests.setUpClass();cls.helper=TraceEntryTests('test_trace_copies_without_copied_start_and_preserves_originals')

    def test_trace_static_contact_full_law_uses_the_original_auxiliary_axis(self):
        from gpu_engine import chance_consumers as c
        from gpu_engine.fidelity_panel import JaxCaseExecutor,compare_slot,native_executor
        h=self.helper
        for side in (0,1):
            state=h.fixture(side,'static')
            state=state._replace(moves=state.moves.at[:,1-side,0,0].set(3))
            actions=(4,0) if side==0 else (0,4)
            plan=c.plan_for_state(state,0,actions,h.table,h.meta,cap=4096)
            self.assertIsInstance(plan,c.ConsumerPlan,plan)
            self.assertTrue(next(a.observations for a in plan.axes if a.coordinate==f'phazing.{1-side}'))
            result=compare_slot(plan,JaxCaseExecutor(state,actions,h.table,plan,batch_size=64),
                lambda:native_executor(plan),batch_size=64,mass_tolerance=1e-5,probability_tolerance=1e-5)
            self.assertEqual(result['status'],'AGREE',{k:v for k,v in result.items() if k!='outcomes'})

    def test_complete_ordinary_successors_both_seats(self):
        import poke_engine
        from gpu_engine import future_state as f
        h=self.helper;j=h.j
        chance=h.t.ChanceInputs(j.zeros((1,2,12),j.int32),j.full((1,2),59),
            j.full((1,2),99),j.full((1,2),59),j.array([[0,1]]),j.array([0]))
        cases=[('static',0),('intimidate',0),('drizzle',0),('drought',0),('sandstream',0),
            ('truant',0),('waterveil',2),('magmaarmor',6),('limber',1),('immunity',3),('immunity',4)]
        for side in (0,1):
            for foe,status in cases:
                with self.subTest(side=side,foe=foe,status=status):
                    s=h.fixture(side,foe,status)
                    native=f.native_input_from_jax(s,0,h.meta,module=poke_engine)
                    self.assertIsInstance(native,f.NativeInput,native)
                    before=native.state.to_string()
                    moves=['splash','splash'];moves[side]='porygon2'
                    branches=poke_engine.generate_instructions(native.state,*moves)
                    self.assertEqual(len(branches),1)
                    self.assertAlmostEqual(branches[0].percentage,100)
                    after=native.state.apply_instructions(branches[0])
                    self.assertEqual(after.reverse_instructions(branches[0]).to_string(),before)
                    expected=f.canonical_native_future(after,native.certificate)
                    actions=j.array([[4,0] if side==0 else [0,4]])
                    actual=h.jax.jit(h.t.step_with_chance)(s,actions,h.table,chance).state
                    mapped=f.canonical_jax_future(actual,0,h.meta,certificate=native.certificate)
                    self.assertNotIsInstance(expected,f.Refused,expected)
                    self.assertNotIsInstance(mapped,f.Refused,mapped)
                    self.assertEqual(f.field_differences(mapped,expected),())
                    self.assertEqual(native.state.to_string(),before)


if __name__=='__main__':unittest.main()
