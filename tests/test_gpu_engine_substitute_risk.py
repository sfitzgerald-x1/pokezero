"""Private exact Site11 risk joins; pure component fixtures, no model/kernel."""
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import engine_substitute_callbacks as sub,engine_execution_completeness as execution
from gpu_engine import engine_kernel_frame_risk as risk
import test_gpu_engine_substitute_guards as guards

class SubstituteRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guards.SubstituteGuardTests.setUpClass();cls.g=guards.SubstituteGuardTests();cls.f=cls.g.f
        cls.compiled=risk.compile_risks(cls.f.producer,cls.f.account,substitute_compiled=cls.f.compiled)

    def captured(self):
        t,ticket,side,phys,move=self.g.tape();w=self.g.write()
        ctx=sub.RegisteredContext(self.f.compiled.device,self.f.account.device)
        t=sub.record_device(t,side,phys,move,w,ctx,eligible=j.ones(8,bool))
        t=execution.capture_substitute_write(t,ticket,side,phys,move,w,requested=j.ones(8,bool))
        c=risk.initialize_device(t.stream_id,self.compiled.device)
        c=risk.begin_frame_device(c,t,self.compiled.device,active=j.ones(8,bool))
        return c,t,risk.CapturedSubstitute(ctx,ticket,side,phys,move,w)

    def test_original_static_candidate_only_and_exact_capture_with_lifetime_sticky_bits(self):
        plain=risk.compile_risks(self.f.producer,self.f.account)
        move=int(self.f.compiled.device.move_id)
        self.assertTrue(int(plain.device.selected_risk[move])&risk.SELF_EFFECT)
        self.assertTrue(int(plain.device.selected_risk[move])&risk.NON_WEATHER)
        self.assertEqual(int(self.compiled.device.selected_risk[move]),0)
        c,t,captured=self.captured()
        good=risk.observe_substitute_device(c,t,j.ones(8,bool),self.compiled.device,active=j.ones(8,bool),captured=captured)
        self.assertEqual(good.risk.tolist(),[0]*8)
        prior=c._replace(risk=j.full(8,risk.SELF_EFFECT|risk.FIELD_VOLATILE,j.int32))
        sticky=risk.observe_substitute_device(prior,t,j.ones(8,bool),self.compiled.device,active=j.ones(8,bool),captured=captured)
        self.assertEqual(sticky.risk.tolist(),[risk.SELF_EFFECT|risk.FIELD_VOLATILE]*8)

    def test_missing_ticket_claim_span_data_and_inactive_rows_never_become_supported(self):
        c,t,x=self.captured();ones=j.ones(8,bool)
        cases=[(t,None),(t,x._replace(ticket=x.ticket._replace(frame=x.ticket.frame+1))),
            (t._replace(words=t.words.at[0,1,6].add(1)),x),
            (t._replace(execution=t.execution._replace(ledger=t.execution.ledger._replace(
                claimed=t.execution.ledger.claimed.at[0,1].set(0)))),x),
            (t,x._replace(context=x.context._replace(data=x.context.data._replace(binding=x.context.data.binding+1))))]
        for tape,captured in cases:
            out=risk.observe_substitute_device(c,tape,ones,self.compiled.device,active=ones,captured=captured)
            self.assertTrue(bool(out.risk[0]&risk.SELF_EFFECT))
        out=risk.observe_substitute_device(c,t,ones,self.compiled.device,active=j.zeros(8,bool),captured=None)
        for before,after in zip(c,out):np.testing.assert_array_equal(before,after)
        for field in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
            bad=x.context.data._replace(**{field:j.tile(getattr(x.context.data,field),(8,1))})
            with self.subTest(field=field),self.assertRaises(ValueError):
                risk.observe_substitute_device(c,t,ones,self.compiled.device,active=ones,
                    captured=x._replace(context=x.context._replace(data=bad)))
