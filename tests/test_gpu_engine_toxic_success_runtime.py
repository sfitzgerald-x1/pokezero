"""Actual Toxic write+nonlethal residual through both genuine owners, no model."""
import unittest
import jax.numpy as j
from gpu_engine import engine_retained_encoder_driver as driver,engine_status_residual_callbacks as residual
from gpu_engine import engine_execution_completeness as accounting,engine_toxic_callbacks as toxic
import test_gpu_engine_toxic_runtime as original

class ToxicSuccessRuntimeTests(unittest.TestCase):
    same=original.ToxicRuntimeTests.same
    construct=original.ToxicRuntimeTests.construct
    advance=original.ToxicRuntimeTests.advance

    @classmethod
    def setUpClass(cls):
        original.ToxicRuntimeTests.setUpClass();f=original.ToxicRuntimeTests
        for name in ('base','move_id','metadata','names','catalog','table','toxic','request','facts'):
            setattr(cls,name,getattr(f,name))
        r=f.runtime
        helper=residual.compile_status_residual(cls.metadata,engine_catalog=r['request']['lifecycle'])
        cls.prepared=driver.prepare(cls.request,cls.facts,public_mode='retained-full',
            leftovers_compiled=r['leftovers_compiled'],leftovers_mode='retained',
            account_source_sites=True,track_kernel_risks=True,
            toxic_compiled=cls.toxic,toxic_mode='kernel-event',
            status_residual_compiled=helper,status_residual_mode='kernel-event',frame_input_mode='source-witness',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)

    def test_actual_primary_and_toxic_residual_both_owners_drain_without_duplicate_status(self):
        before,result,frame=self.advance(self.construct(),0,foe_action=7)
        raw=frame.callbacks.words;present=j.arange(raw.shape[1])[None,:]<frame.callbacks.count[:,None]
        self.assertEqual(result.state.core.admitted.tolist(),[True,True],msg=str(dict(
            risk=result.state.core.risks.risk.tolist(),tape=frame.callbacks.refusal.tolist(),
            accounting=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),
            owners=[o.reason.tolist() for o in result.owners])))
        for kind in (9,toxic.TOXIC_ATTEMPT,residual.STATUS_RESIDUAL):
            self.assertEqual(j.sum(present&(raw[:,:,0]==kind),axis=1).tolist(),[1,1])
        at=j.argmax(present&(raw[:,:,0]==26),axis=1);rows=j.arange(2)
        self.assertEqual(raw[rows,at-1,0].tolist(),[9,9]);self.assertEqual(raw[rows,at,7:10].tolist(),[[0,4,0]]*2)
        ledger=result.state.core.execution
        for site in (accounting.Site.TOXIC_ATTEMPT,accounting.Site.STATUS_RESIDUAL):
            selected=(ledger.site==int(site))&(ledger.serial>0)
            self.assertEqual(selected.sum(axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(selected,ledger.stage==3,True))))
        selected=ledger.site==int(accounting.Site.TOXIC_ATTEMPT)
        self.assertTrue(bool(j.all(j.where(selected,ledger.end-ledger.start==2,True))))
        self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
        for owner in result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True])
            self.assertFalse(bool(owner.B4_qualified.any()))
