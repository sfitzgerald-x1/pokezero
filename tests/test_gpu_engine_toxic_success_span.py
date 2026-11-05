"""Exact two-receipt primary/attempt capture, masks and stamps; no game/forward."""
import unittest
import jax.numpy as j
from gpu_engine import engine_status_callbacks as status,engine_toxic_callbacks as toxic
from gpu_engine import engine_execution_completeness as execution
import test_gpu_engine_toxic_guards as base

class ToxicSuccessSpanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):base.ToxicGuardTests.setUpClass();cls.f=base.ToxicGuardTests

    def test_missing_wrong_primary_counter_and_stamps_never_capture(self):
        f=self.f();t,ticket,side,actor,target,move=f.tape();a=f.attempt(accuracy_hit=True,hit=True,
            applied=True,secondary_active=True,new_status=4);ctx=f.runtime['toxic_context'];yes=j.ones(8,bool)
        primary=status.changed(t,1-side,target,a.old_status,a.new_status,status.PRIMARY,eligible=yes)
        raw=toxic.record_device(primary,side,actor,target,move,a,ctx,eligible=yes)
        good=execution.capture_toxic_attempt(raw,ticket,side,actor,target,move,a,requested=yes)
        self.assertEqual(good.execution.ledger.stage[:,0].tolist(),[2]*8)
        self.assertEqual(good.execution.ledger.claimed[:,:3].tolist(),[[0,1,1]]*8)
        faults=[raw._replace(words=raw.words.at[0,1,4].set(4)),
            raw._replace(words=raw.words.at[0,1,6].set(3)),
            raw._replace(words=raw.words.at[0,1,7].set(status.THAW)),
            raw._replace(event_generation=raw.event_generation.at[0,1,0].add(1)),
            raw._replace(words=raw.words.at[0,0,5].set(1)),
            raw._replace(words=raw.words.at[0,2,9].set(1))]
        for bad in faults:
            out=execution.capture_toxic_attempt(bad,ticket,side,actor,target,move,a,requested=yes)
            self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[1]+[2]*7)
        missing=toxic.record_device(t,side,actor,target,move,a,ctx,eligible=yes)
        out=execution.capture_toxic_attempt(missing,ticket,side,actor,target,move,a,requested=yes)
        self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[1]*8)
        inactive=j.array([True]+[False]*7)
        out=execution.capture_toxic_attempt(raw,ticket,side,actor,target,move,a,requested=inactive)
        self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2]+[1]*7)
        diagnostic=toxic.record_device(primary,side,actor,target,move,a,toxic.Context(ctx.data),eligible=yes)
        self.assertTrue(bool(((diagnostic.refusal&256)!=0).all()))
        prior=toxic.record_device(primary._replace(refusal=primary.refusal|128),side,actor,target,move,a,ctx,eligible=yes)
        self.assertTrue(bool(((prior.refusal&128)!=0).all()))
