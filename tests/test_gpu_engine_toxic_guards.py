"""Literal guard/counter/span faults, no model/native/scientific root."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_toxic_callbacks as toxic,engine_lifecycle_callbacks as life
from gpu_engine import engine_execution_completeness as execution,engine_public_events as public
from gpu_engine import engine_kernel_frame_risk as risks
import test_gpu_engine_toxic_runtime as fixtures

class ToxicGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ToxicRuntimeTests.setUpClass();cls.f=fixtures.ToxicRuntimeTests
        cls.data=toxic.check_compiled(cls.f.toxic);cls.runtime=cls.f.runtime

    def attempt(self,b=8,**updates):
        flags=[False,False,False,False,False,True,False,False,False,False,False]
        a=toxic.Attempt(*[j.full(b,x,bool) for x in flags],*[j.zeros(b,j.int32) for _ in range(4)])
        return a._replace(**{k:j.full(b,v,bool if k in a._fields[:11] else j.int32) for k,v in updates.items()})

    def tape(self,b=8):
        producer=self.runtime['request']['lifecycle'];cursor=life.register(j.arange(401,401+b,dtype=j.int32),catalog=producer,
            expected_binding=producer.binding,expected_engine_binding=np.array(producer.engine_binding,np.uint32))
        mapping=cursor.physical_to_public.at[:,0,3].set(0).at[:,1,5].set(0)
        cursor=cursor._replace(turn=j.ones(b,j.int32),initialized=j.ones(b,bool),physical_active=j.tile(j.array([3,5],j.int32),(b,1)),
            public_slot=j.zeros((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),
            revealed_species_id=j.tile(j.array([1,2],j.int32),(b,1)),physical_to_public=mapping,next_slot=j.ones((b,2),j.int32))
        tape=life._tape(cursor,producer,64);account=self.runtime['execution_compiled']
        ledger=execution.initialize(cursor.stream_id,compiled=account,capacity=16)
        tape=execution.begin_on_tape(tape,ledger,account.device,active=j.ones(b,bool))
        side=j.zeros(b,j.int32);actor=j.full(b,3,j.int32);target=j.full(b,5,j.int32);move=j.full(b,int(self.data.move_id),j.int32)
        tape=public.move(tape,side,actor,target,move,eligible=j.ones(b,bool))
        tape,ticket=execution.open_attempt(tape,execution.Site.TOXIC_ATTEMPT,1-side,active=j.ones(b,bool))
        return tape,ticket,side,actor,target,move

    def test_literal_miss_type_immunity_and_all_unsupported_outcomes_stay_distinct(self):
        tape,ticket,side,actor,target,move=self.tape();ctx=self.runtime['toxic_context']
        cases=[(self.attempt(),toxic.MISS,True),
            (self.attempt(accuracy_hit=True,hit=True,type_immune=True,secondary_active=True),toxic.PLAIN_TYPE_IMMUNITY,True),
            (self.attempt(accuracy_hit=True,hit=True,applied=True,secondary_active=True,new_status=4),toxic.STATUS_WRITE,False),
            (self.attempt(accuracy_hit=True,hit=True,substitute_block=True,secondary_active=True),toxic.SUBSTITUTE_BLOCK,False),
            (self.attempt(accuracy_hit=True,hit=True,ability_immune=True,secondary_active=True),toxic.ABILITY_BLOCK,False),
            (self.attempt(accuracy_hit=True,hit=True,secondary_active=True,old_status=3,new_status=3),toxic.EXISTING_STATUS,False),
            (self.attempt(accuracy_hit=True,protected=True),toxic.OTHER,False)]
        for attempt,code,supported in cases:
            out=toxic.record_device(tape,side,actor,target,move,attempt,ctx,eligible=j.ones(8,bool))
            raw=out.words[:,1];self.assertEqual(toxic.outcome_device(raw).tolist(),[code]*8)
            self.assertEqual(toxic.supported_raw(raw).tolist(),[supported]*8)
            out=execution.capture_toxic_attempt(out,ticket,side,actor,target,move,attempt,requested=j.ones(8,bool))
            self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2 if supported else 1]*8)
            self.assertEqual(((out.refusal&256)!=0).tolist(),[not supported]*8)

    def test_no_counter_alias_or_contradictory_guard_can_capture_and_b8_inactive_freezes(self):
        t,ticket,side,actor,target,move=self.tape();ctx=self.runtime['toxic_context']
        #17 and1 would alias when naively packing four-bit BEFORE/AFTER values.
        invalid=self.attempt(old_counter=17,new_counter=1)
        out=toxic.record_device(t,side,actor,target,move,invalid,ctx,eligible=j.ones(8,bool))
        self.assertEqual(out.count.tolist(),[1]*8);self.assertTrue(bool(((out.refusal&256)!=0).all()))
        contradiction=self.attempt(accuracy_hit=False,hit=True,secondary_active=True)
        out=toxic.record_device(t,side,actor,target,move,contradiction,ctx,eligible=j.ones(8,bool))
        self.assertEqual(out.count.tolist(),[1]*8)
        mask=j.array([True,True,False,False,False,False,False,False])
        out=toxic.record_device(t,side,actor,target.at[1].set(4),move,self.attempt(),ctx,eligible=mask)
        self.assertEqual(out.count.tolist(),[2]+[1]*7);self.assertTrue(bool(out.refusal[1]&256))
        self.assertEqual(out.refusal[2:].tolist(),t.refusal[2:].tolist())
        np.testing.assert_array_equal(out.words[2:],t.words[2:])
        prior=toxic.record_device(t._replace(refusal=t.refusal|128),side,actor,target,move,self.attempt(),ctx,eligible=j.ones(8,bool))
        self.assertTrue(bool(((prior.refusal&128)!=0).all()))
        full=toxic.record_device(t._replace(count=j.full(8,64,j.int32)),side,actor,target,move,self.attempt(),ctx,eligible=j.ones(8,bool))
        self.assertTrue(bool(((full.refusal&public.OVERFLOW)!=0).all()))

    def test_prefix_stamp_current_target_claim_and_lifetime_risk_faults(self):
        t,ticket,side,actor,target,move=self.tape();a=self.attempt();ctx=self.runtime['toxic_context'];ones=j.ones(8,bool)
        t=toxic.record_device(t,side,actor,target,move,a,ctx,eligible=ones)
        good=execution.capture_toxic_attempt(t,ticket,side,actor,target,move,a,requested=ones)
        tables=self.runtime['kernel_risk_compiled'].device
        carry=risks.initialize_device(t.stream_id,tables);carry=risks.begin_frame_device(carry,good,tables,active=ones)
        captured=risks.CapturedToxic(ctx,ticket,side,actor,target,move,a)
        zero=risks.observe_primary_status_device(carry,good,ones,tables,active=ones,captured=captured)
        self.assertEqual(zero.risk.tolist(),[0]*8)
        prior=carry._replace(risk=j.full(8,risks.FIELD_VOLATILE|risks.SELF_EFFECT,j.int32))
        retained=risks.observe_primary_status_device(prior,good,ones,tables,active=ones,captured=captured)
        self.assertEqual(retained.risk.tolist(),[risks.FIELD_VOLATILE|risks.SELF_EFFECT]*8)
        for bad in [t._replace(words=t.words.at[0,0,5].set(1)),t._replace(event_generation=t.event_generation.at[0,1,1].add(1)),
                    t._replace(cursor=t.cursor._replace(physical_active=t.cursor.physical_active.at[0,1].set(4)))]:
            failed=execution.capture_toxic_attempt(bad,ticket,side,actor,target,move,a,requested=ones)
            self.assertEqual(failed.execution.ledger.stage[:,0].tolist(),[1]+[2]*7)
        for name in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
            bad=ctx.data._replace(**{name:j.tile(getattr(ctx.data,name),(8,1))})
            with self.assertRaises(ValueError):toxic.record_device(t,side,actor,target,move,a,ctx._replace(data=bad),eligible=ones)
        with self.assertRaises(ValueError):jax.jit(lambda dummy:toxic.check_compiled(self.f.toxic))(j.array(0))
        with self.assertRaises(ValueError):jax.jit(lambda dummy:life.trace_step(None,None,None,cursor=None,catalog=None,
            expected_binding=None,expected_engine_binding=None,toxic_compiled=self.f.toxic))(j.array(0))
