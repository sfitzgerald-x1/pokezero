"""Explicit diagnostic caller mode; no public/status-origin qualification."""
from dataclasses import replace
import gc
import json
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as driver
from gpu_engine import engine_status_residual_callbacks as status
from gpu_engine import mechanics,transition
import test_gpu_engine_execution_runtime_plumbing as fixtures


class StatusResidualRuntimeTests(unittest.TestCase):
    def setUp(self):
        # Setup builds several prepared variants; do not retain their compiled
        # caches or a previous case's executables into this runtime case.
        jax.clear_caches()
        gc.collect()

    construct=fixtures.ExecutionRuntimePlumbingTests.construct
    chance=fixtures.ExecutionRuntimePlumbingTests.chance
    same=fixtures.ExecutionRuntimePlumbingTests.same

    @classmethod
    def setUpClass(cls):
        fixtures.ExecutionRuntimePlumbingTests.setUpClass();base=fixtures.ExecutionRuntimePlumbingTests
        cls.f=base.f;cls.plain=base.prepared;r=driver._checked(cls.plain)
        cls.helper=status.compile_status_residual(json.loads(r['request']['metadata']),
            engine_catalog=r['request']['lifecycle'])
        cls.args=(r['request_compiled'],r['own_compiled'])
        cls.kwargs=dict(public_mode='retained-full',leftovers_compiled=r['leftovers_compiled'],
            leftovers_mode='retained',account_source_sites=True,status_residual_compiled=cls.helper,
            fake_forward=lambda _:driver.FakeScores(j.ones(9),j.array(1,j.int32)))
        cls.prepared=driver.prepare(*cls.args,**cls.kwargs)

    def test_original_host_guards_and_default_actual_frame_remain_intact(self):
        with self.assertRaises(ValueError):driver.prepare(*self.args,**dict(self.kwargs,public_mode='basic'))
        with self.assertRaises(ValueError):driver.prepare(*self.args,**dict(self.kwargs,status_residual_compiled=replace(self.helper)))
        for call in (lambda:jax.jit(lambda:driver.prepare(*self.args,**self.kwargs).binding)(),
                     lambda:jax.jit(lambda dummy:driver.prepare(*self.args,**self.kwargs).binding)(j.array(0))):
            with self.assertRaisesRegex(ValueError,'outside outer JIT'):call()
        diagnostic=self.construct();plain=self.construct(prepared=self.plain)
        self.same(diagnostic.state.core.engine,plain.state.core.engine)
        r=driver._checked(self.prepared)
        frame=r['frame'](diagnostic.state.core,j.full((2,2),8,j.int32),self.chance())
        ordinary=driver._checked(self.plain)['frame'](plain.state.core,j.full((2,2),8,j.int32),self.chance())
        self.same(frame,ordinary)
        # Revalidate even after the compiled closures have been warmed.
        with patch.object(status,'check_compiled',side_effect=ValueError('status metadata changed')):
            with self.assertRaisesRegex(ValueError,'status metadata changed'):
                driver.advance(diagnostic,j.full((2,2),8,j.int32),chance_inputs=self.chance())

    def test_internal_actual_status_write_stays_explicitly_unknown(self):
        session=self.construct();core=session.state.core;r=driver._checked(self.prepared)
        # Explicit INTERNAL fault fixture, not a sealed owning/public history:
        # it tests the real kernel's literal diagnostic write, never admission.
        engine=core.engine._replace(status=core.engine.status.at[:,0,0].set(mechanics.BRN))
        core=core._replace(engine=engine)
        actions=j.tile(j.array([[0,8]],j.int32),(2,1))
        frame=r['frame'](core,actions,self.chance())
        legacy=transition.step_with_chance(engine,actions,r['request']['engine_table'],self.chance())
        self.same(frame.state,legacy.state)
        for row in range(2):
            raw=np.asarray(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
            writes=raw[raw[:,0]==status.STATUS_RESIDUAL]
            self.assertEqual(writes.shape,(1,10));write=writes[0]
            self.assertEqual(write[1:5].tolist(),[0,0,0,0])
            self.assertEqual(int(write[6]),int(frame.state.hp[row,0,0]))
            self.assertEqual(int(write[7]),int(engine.max_hp[row,0,0]))
            self.assertEqual(write[8:].tolist(),[mechanics.BRN,status.BURN])
            self.assertGreater(int(write[5]),int(write[6]))
            self.assertTrue(int(frame.callbacks.refusal[row])&status.UNKNOWN_STATUS_RESIDUAL)
            self.assertTrue(int(frame.callbacks.refusal[row])&512)
            self.assertNotEqual(int(frame.callbacks.execution.ledger.refusal[row]),0)
        for owner in session.result.owners:
            self.assertFalse(bool(owner.scored.any()));self.assertFalse(bool(owner.B4_qualified.any()))

    def test_explicit_kernel_event_preparation_requires_original_helper_accounting_and_fresh_host_guards(self):
        from gpu_engine import engine_status_residual_callbacks as callbacks
        options=dict(self.kwargs,status_residual_mode='kernel-event')
        prepared=driver.prepare(*self.args,**options);runtime=driver._checked(prepared)
        self.assertIs(type(runtime['status_residual_context']),callbacks.RegisteredStatusResidualContext)
        self.assertIs(runtime['status_residual_context'].data,self.helper.device)
        self.assertIs(runtime['status_residual_context'].weather_profile_data,runtime['speed_profile'].device)
        self.assertNotEqual(prepared.binding,self.prepared.binding)
        for change in (dict(account_source_sites=False),dict(status_residual_compiled=None),
                       dict(public_mode='basic'),dict(status_residual_mode='implicit')):
            with self.assertRaises(ValueError):driver.prepare(*self.args,**dict(options,**change))
        for call in (lambda:jax.jit(lambda:driver.prepare(*self.args,**options).binding)(),
                     lambda:jax.jit(lambda dummy:driver.prepare(*self.args,**options).binding)(j.array(0))):
            with self.assertRaisesRegex(ValueError,'outside outer JIT'):call()
        with patch.object(status,'check_compiled',side_effect=ValueError('status original changed')):
            with self.assertRaisesRegex(ValueError,'status original changed'):driver._checked(prepared)


if __name__=='__main__':unittest.main()
