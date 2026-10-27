"""Actual native Wish state/ordering guide; authored cases, no panel credit."""
import importlib.util
import json
import unittest

AVAILABLE=all(importlib.util.find_spec(n) for n in ('jax','poke_engine','pokezero_search'))
if AVAILABLE:
    import pokezero_search
    AVAILABLE=bool(getattr(pokezero_search,'CONDITIONED_ORACLE_ENABLED',False))


@unittest.skipUnless(AVAILABLE,'requires isolated JAX and both conditioned native bindings')
class NativeWishHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import poke_engine
        from gpu_engine import future_state
        from tests import test_gpu_engine_wish_history as cases
        cases.WishHistoryTests.setUpClass()
        cls.h=cases.WishHistoryTests()
        cls.engine,cls.f=poke_engine,future_state

    def construct(self,s):
        out=self.f.native_input_from_jax(s,0,self.h.meta,module=self.engine)
        self.assertIsInstance(out,self.f.NativeInput,type(out).__name__)
        return out

    def assert_agree(self,s,native,certificate):
        left=self.f.canonical_jax_future(s,0,self.h.meta,certificate=certificate)
        right=self.f.canonical_native_future(native,certificate)
        self.assertIsInstance(left,self.f.CanonicalFuture,type(left).__name__)
        self.assertIsInstance(right,self.f.CanonicalFuture,type(right).__name__)
        self.assertEqual(self.f.field_differences(left,right),())

    def advance(self,s,native,certificate,side,action,choice):
        choices=['splash','splash'];choices[side]=choice
        branches=self.engine.generate_instructions(native,*choices)
        self.assertEqual(len(branches),1)
        before=native.to_string()
        result=native.apply_instructions(branches[0])
        self.assertEqual(native.to_string(),before)
        # Public native names select the first matching slot. The opposing
        # Splash is slot0; match that exact request rather than another copy.
        actions=self.h.actions(side,action).at[0,1-side].set(0)
        out=self.h.play(s,actions)
        self.assert_agree(out,result,certificate)
        return out,result

    def test_full_creation_unequal_recipient_expiry_and_recast_both_seats(self):
        for side in range(2):
            s=self.h.caster(side);n=self.construct(s);native=n.state
            for action,choice,counter,amount in ((0,'wish',1,150),(4,'pikachu',0,150),(0,'wish',1,50)):
                s,native=self.advance(s,native,n.certificate,side,action,choice)
                self.h.assert_wish(s,side,counter,amount)
                field=native.side_one if side==0 else native.side_two
                self.assertEqual(field.wish,(counter,amount))

    def test_full_pending_recast_retains_amount_and_recipient_healing(self):
        for side in range(2):
            for counter,amount in ((2,77),(1,77),(1,0)):
                s=self.h.caster(side);s=s._replace(wish_counter=s.wish_counter.at[0,side].set(counter),
                    wish_amount=s.wish_amount.at[0,side].set(amount))
                n=self.construct(s)
                out,_=self.advance(s,n.state,n.certificate,side,0,'wish')
                self.h.assert_wish(out,side,counter-1,amount)
                self.assertEqual(int(out.pp[0,side,0,0]),9)

    def test_full_hp_expiry_and_simultaneous_asymmetric_wishes(self):
        for side in range(2):
            s=self.h.caster(side);s=s._replace(hp=s.hp.at[0,side,0].set(301),
                wish_counter=s.wish_counter.at[0,side].set(1),wish_amount=s.wish_amount.at[0,side].set(150))
            n=self.construct(s);out,_=self.advance(s,n.state,n.certificate,side,1,'splash')
            self.h.assert_wish(out,side,0,150)
        s=self.h.caster(0);s=s._replace(wish_counter=self.h.j.array([[1,1]]),
            wish_amount=self.h.j.array([[150,50]]))
        n=self.construct(s);out,_=self.advance(s,n.state,n.certificate,0,1,'splash')
        self.h.assert_wish(out,0,0,150);self.h.assert_wish(out,1,0,50)

    def test_expired_native_wire_amount_is_retained_as_an_exact_difference(self):
        keys=[]
        for amount in (150,151):
            s=self.h.caster(0);s=s._replace(wish_amount=s.wish_amount.at[0,0].set(amount))
            n=self.construct(s)
            self.assertEqual(n.state.side_one.wish,(0,amount))
            right=self.f.canonical_native_future(n.state,n.certificate)
            self.assertIsInstance(right,self.f.CanonicalFuture,type(right).__name__)
            keys.append(right.key)
        self.assertNotEqual(*keys)

    def test_conditional_guard_still_refuses_pending_and_expired_nonzero_wish(self):
        import pokezero_search
        from tests import test_conditioned_native_resume as cases
        cases.ConditionedResumeTests.setUpClass()
        helper=cases.ConditionedResumeTests()
        for side in range(2):
            _,native,_,capsule,_=helper.fixture(side,99)
            for counter,amount in ((1,0),(0,150)):
                pieces=native.to_string().split('/')
                fields=pieces[side].split('=')
                fields[18:20]=[str(counter),str(amount)]
                pieces[side]='='.join(fields)
                raw='/'.join(pieces);before=json.dumps(capsule,sort_keys=True)
                with self.assertRaisesRegex(ValueError,'UNSUPPORTED_SIDE_CONTEXT'):
                    pokezero_search.conditioned_resume(raw,json.dumps(capsule),1,[0,1])
                self.assertEqual(json.dumps(capsule,sort_keys=True),before)


if __name__=='__main__':unittest.main()
