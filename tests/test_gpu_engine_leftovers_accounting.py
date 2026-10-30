"""Registered-context transport drain; no exhaustive/source-origin claim."""
import unittest
import jax.numpy as j
from gpu_engine import engine_execution_completeness as accounting
import test_gpu_engine_leftovers_public as fixture


class LeftoversAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.LeftoversPublicTests.setUpClass();cls.f=fixture.LeftoversPublicTests
        cls.compiled=accounting.compile_accounting(cls.f.f.compiled)

    def test_exact_heal_spans_require_original_full_consumer_with_typed_contexts(self):
        f=self.f();before,_,initial,record=f.prepared()
        frame,_,_=f.frame(initial);tape=frame.callbacks
        tables=self.compiled.device;active=j.ones(2,bool)
        ledger=accounting.initialize(tape.stream_id,compiled=self.compiled)
        prefix=tape._replace(count=j.zeros(2,j.int32),cursor=initial.cursor,turn=initial.cursor.turn)
        ledger=accounting.begin_frame(ledger,prefix,frame=j.ones(2,j.int32),active=active,tables=tables)
        # Source-seam fixture: two actual-style distinct heal spans. This
        # checks transport only, not the pending actual semantic site mapping.
        for side,start in ((0,3),(1,4)):
            prefix=prefix._replace(count=j.full(2,start,j.int32))
            ledger,ticket=accounting.enter(ledger,prefix,j.full(2,int(accounting.Site.LEFTOVERS),j.int32),
                j.full(2,side,j.int32),active=active,tables=tables)
            ledger=accounting.emitted(ledger,ticket,prefix._replace(count=prefix.count+1),active=active,tables=tables)
        kwargs=dict(active=active,tables=tables,adapter_tables=self.f.f.compiled.device,catalog=self.f.f.catalog,
            weather_hp=(record,self.f.profile.device),leftovers=self.f.heal.device)
        good,consumed=accounting.consume_frame_device(ledger,frame,before.projection.state,before.public,**kwargs)
        self.assertEqual(consumed.projection.error.tolist(),[0,0])
        self.assertEqual(accounting.report(good,tables=tables).scoped_drained.tolist(),[True,True])
        self.assertFalse(bool(consumed.projection.full_encoder_admitted.any()))
        self.assertFalse(bool(consumed.projection.B4_qualified.any()))
        # Losing the actual typed registration cannot be replaced by a
        # claimed span or consumed-prefix count; the genuine consumer refuses.
        kwargs['leftovers']=None
        bad,refused=accounting.consume_frame_device(ledger,frame,before.projection.state,before.public,**kwargs)
        self.assertTrue(bool((refused.projection.error!=0).all()))
        self.assertFalse(bool(accounting.report(bad,tables=tables).scoped_drained.any()))
        self.f.f().same(refused.public,before.public)


if __name__=='__main__':unittest.main()
