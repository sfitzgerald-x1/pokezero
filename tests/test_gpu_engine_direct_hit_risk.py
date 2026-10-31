"""Declared direct-hit risk capability requires its actual captured source span."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_kernel_frame_risk as r, engine_direct_hit_controls as h
from gpu_engine import engine_execution_completeness as e
import test_gpu_engine_direct_hit_accounting as fixture


class DirectHitRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DirectHitAccountingTests.setUpClass();cls.base=fixture.DirectHitAccountingTests
        cls.original=cls.base.f
        cls.compiled=r.compile_risks(cls.original.f.lifecycle,cls.base.account,
            direct_hit_compiled=cls.original.compiled)
        cls.weather_only=r.compile_risks(cls.original.f.lifecycle,cls.base.account)

    def source(self,*,critical=True,num=2,den=1):
        span=self.base().span(critical=critical,num=num,den=den)
        tape=self.base().capture(span)
        _,ticket,actor,physical,target,move,write,active=span
        captured=r.CapturedDirectHit(h.RegisteredContext(self.original.data,self.base.account.device),
            ticket,actor,physical,target,move,write)
        outcome=r.Outcome(write.power,write.critical,write.hit,write.immune,write.numerator,write.denominator,
            j.zeros(2,bool),write.substitute,write.protected,write.enduring,j.zeros(2,j.int32),j.zeros(2,bool))
        carry=r.initialize_device(tape.stream_id,self.compiled.device)
        carry=r.begin_frame_device(carry,tape,self.compiled.device,active=active)
        return tape,carry,captured,outcome,active

    def observe(self,tape,carry,captured,outcome,active):
        return r.observe_outcome_device(carry,tape,outcome,self.compiled.device,
            eligible=active,active=active,captured_direct=captured)

    def test_registered_candidate_and_literal_source_outcome_not_retrospective_risk_clear(self):
        self.assertEqual(int(self.compiled.device.selected_risk[self.original.move]),0)
        self.assertTrue(int(self.weather_only.device.selected_risk[self.original.move])&r.NON_WEATHER)
        self.assertIs(r._COMPILED[self.compiled][5],self.original.compiled)
        for critical,num,den in ((True,2,1),(True,1,2),(False,2,2),(False,1,1)):
            tape,carry,captured,outcome,active=self.source(critical=critical,num=num,den=den)
            got=jax.jit(self.observe)(tape,carry,captured,outcome,active)
            self.assertEqual(got.risk.tolist(),[0,0])
            prior=carry._replace(risk=j.full(2,r.NON_WEATHER|r.SELF_HP,j.int32))
            self.assertEqual(self.observe(tape,prior,captured,outcome,active).risk.tolist(),[r.NON_WEATHER|r.SELF_HP]*2)
            # This is a source-cutpoint diagnostic only, never frame coverage.
            self.assertFalse(hasattr(got,'complete'));self.assertFalse(hasattr(got,'admitted'))

    def test_wrong_or_missing_source_cannot_admit_even_neutral_noncritical_outcome(self):
        tape,carry,captured,outcome,active=self.source(critical=False,num=1,den=1)
        bads=[None,captured._replace(ticket=captured.ticket._replace(serial=captured.ticket.serial+1)),
            captured._replace(move=captured.move+1),captured._replace(actor=1-captured.actor),
            captured._replace(write=captured.write._replace(old_hp=captured.write.old_hp+1)),
            captured._replace(context=captured.context._replace(data=captured.context.data._replace(
                binding=captured.context.data.binding.at[0].add(j.uint32(1)))))]
        for bad in bads:
            with self.subTest(captured=bad is None):
                self.assertTrue(bool((self.observe(tape,carry,bad,outcome,active).risk&r.TARGET_OUTCOME).all()))
        # The same source ticket cannot explain a different actual kernel outcome.
        self.assertTrue(bool((self.observe(tape,carry,captured,outcome._replace(critical=j.ones(2,bool)),active).risk&r.TARGET_OUTCOME).all()))
        for change in (dict(absorbed=j.ones(2,bool)),dict(power=j.zeros(2,j.int32))):
            self.assertTrue(bool((self.observe(tape,carry,captured,outcome._replace(**change),active).risk&r.TARGET_OUTCOME).all()))
        for change in (dict(recoil=j.ones(2,j.int32)),dict(explodes=j.ones(2,bool))):
            self.assertTrue(bool((self.observe(tape,carry,captured,outcome._replace(**change),active).risk&r.SELF_HP).all()))
        altered=tape._replace(words=tape.words.at[0,0,5].add(1))
        result=self.observe(altered,carry,captured,outcome,active)
        self.assertTrue(int(result.risk[0])&r.TARGET_OUTCOME);self.assertEqual(int(result.risk[1]),0)

    def test_original_compiler_host_guard_and_default_zero_binding(self):
        np.testing.assert_array_equal(self.weather_only.device.direct_hit_binding,j.zeros(8,j.uint32))
        with self.assertRaises(ValueError):r.compile_risks(self.original.f.lifecycle,self.base.account,
            direct_hit_compiled=replace(self.original.compiled))
        with patch.object(h,'check_compiled',side_effect=ValueError('changed original direct compiler')):
            with self.assertRaisesRegex(ValueError,'changed original direct compiler'):r.check_compiled(self.compiled)
        with self.assertRaises(ValueError):jax.jit(lambda:r.compile_risks(self.original.f.lifecycle,
            self.base.account,direct_hit_compiled=self.original.compiled).device.binding)()


if __name__=='__main__':unittest.main()
