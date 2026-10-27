"""Existing Rust history regression rules exercised through the actual binding."""
from fractions import Fraction
import importlib.util
import unittest

AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','poke_engine'))


@unittest.skipUnless(AVAILABLE,'requires isolated JAX and native oracle')
class NativeActionHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import poke_engine
        from gpu_engine import future_state
        from tests import test_gpu_engine_action_history as cases
        cases.ActionHistoryTests.setUpClass()
        cls.helper=cases.ActionHistoryTests()
        cls.engine,cls.f=poke_engine,future_state

    def fixture(self):
        h=self.helper;s=h.live()
        # Native public names select the first matching slot. Keep active names
        # unique here; duplicate-slot dispatch is tested in the JAX/mapper suite.
        s=s._replace(moves=s.moves.at[:,:,0,:].set(h.j.array([2,3,6,1])))
        return s

    def construct(self,s):
        out=self.f.native_input_from_jax(s,0,self.helper.meta,module=self.engine)
        self.assertIsInstance(out,self.f.NativeInput,type(out).__name__)
        return out

    def assert_full_future(self,s,successor,certificate):
        left=self.f.canonical_jax_future(s,0,self.helper.meta,certificate=certificate)
        right=self.f.canonical_native_future(successor,certificate)
        self.assertIsInstance(left,self.f.CanonicalFuture,type(left).__name__)
        self.assertIsInstance(right,self.f.CanonicalFuture,type(right).__name__)
        self.assertEqual(self.f.field_differences(left,right),())

    def test_real_binding_records_nonzero_slot_and_pp_in_complete_both_seat_future(self):
        h=self.helper;s=self.fixture();n=self.construct(s)
        self.assertTrue(n.certificate.conditional_last)
        self.assertEqual(n.state.side_one.last_used_move,'move:none')
        branches=self.engine.generate_instructions(n.state,'splash','splash')
        self.assertEqual(len(branches),1)
        r=h.play(s,h.j.array([[3,3]]))
        self.assert_full_future(r,n.state.apply_instructions(branches[0]),n.certificate)

    def test_real_before_move_paralysis_law_preserves_none_or_records_exact_slot(self):
        h=self.helper
        for side in range(2):
            s=self.fixture();s=s._replace(status=s.status.at[0,side,0].set(h.m.PAR))
            n=self.construct(s);branches=self.engine.generate_instructions(n.state,'splash','splash')
            actual={}
            for b in branches:
                successor=n.state.apply_instructions(b)
                field=successor.side_one if side==0 else successor.side_two
                actual[field.last_used_move]=actual.get(field.last_used_move,Fraction())+Fraction(str(b.percentage))/100
                blocked=field.last_used_move=='move:none'
                c=h.chance();c=c._replace(draws=c.draws.at[0,side,2].set(0 if blocked else 99))
                self.assert_full_future(h.play(s,h.j.array([[3,3]]),c),successor,n.certificate)
            self.assertEqual(actual,{'move:none':Fraction(1,4),'move:3':Fraction(3,4)})

    def test_real_switch_future_and_hazard_ko_history_preserve_admission_guards(self):
        h=self.helper
        for side in range(2):
            for hazard in (False,True):
                s=self.fixture();s=s._replace(spikes=s.spikes.at[0,side].set(3 if hazard else 0),
                    hp=s.hp.at[0,side,1].set(250))
                n=self.construct(s)
                choices=['splash','splash'];choices[side]='pikachu'
                branches=self.engine.generate_instructions(n.state,*choices)
                self.assertEqual(len(branches),1)
                actions=h.j.array([[3,3]]).at[0,side].set(4)
                r=h.play(s,actions);successor=n.state.apply_instructions(branches[0])
                if not hazard:
                    self.assert_full_future(r,successor,n.certificate)
                else:
                    # An ordinary switch hazard-KO is outside the existing
                    # admitted suspension interface; retain that whole-state
                    # refusal while comparing the actual recorded history.
                    left=self.f.canonical_jax_future(r,0,h.meta,certificate=n.certificate)
                    self.assertIsInstance(left,self.f.Refused)
                    self.assertIn('LIVE_SUSPENSION_INTERFACE',{g.code for g in left.gaps})
                    field=successor.side_one if side==0 else successor.side_two
                    self.assertEqual(field.last_used_move,'switch:1')
                    h.assert_history(r,side,h.h.SWITCH,1)

    def test_conditional_tracking_disabled_keeps_native_inert_history(self):
        h=self.helper;s=self.fixture();s=s._replace(moves=s.moves.at[:,:,0,2].set(2))
        n=self.construct(s);self.assertFalse(n.certificate.conditional_last)
        branches=self.engine.generate_instructions(n.state,'splash','splash')
        self.assert_full_future(h.play(s,h.j.array([[3,3]])),n.state.apply_instructions(branches[0]),n.certificate)


if __name__=='__main__':unittest.main()
