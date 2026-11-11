"""Literal source-span mutations; component fixtures, no owning coverage claim."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_direct_hit_controls as hit, engine_execution_completeness as e
from gpu_engine import engine_full_v4_event_device as full, engine_public_event_device as base
from gpu_engine import engine_public_events as public
import test_gpu_engine_direct_hit_controls as fixture


class DirectHitAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DirectHitControlsTests.setUpClass();cls.f=fixture.DirectHitControlsTests
        projector=base.compile_projector(cls.f.f.lifecycle,expected_binding=cls.f.f.lifecycle.binding,
            expected_engine_binding=np.asarray(cls.f.f.lifecycle.engine_binding,np.uint32))
        adapter=full.compile_adapter(cls.f.f.metadata,projector,catalog=cls.f.f.catalog)
        cls.account=e.compile_accounting(adapter)

    def span(self,b=2,active=None,*,critical=True,num=2,den=1):
        f=self.f();tape=f.tape(b=b,capacity=8)
        active=j.ones(b,bool) if active is None else active
        ledger=e.initialize(tape.stream_id,compiled=self.account,capacity=4,receipt_capacity=8)
        tape=e.begin_on_tape(tape,ledger,self.account.device,active=active)
        actor=j.zeros(b,j.int32);physical=j.full(b,2,j.int32);target=j.full(b,4,j.int32)
        move=j.full(b,self.f.move,j.int32)
        write=f.write(b,numerator=num,denominator=den)._replace(critical=critical)
        tape=public.move(tape,actor,physical,target,move,eligible=active)
        tape,ticket=e.open_attempt(tape,e.Site.DIRECT_HIT,1-actor,active=active)
        tape=hit.record_registered_direct_hit_controls_device(tape,actor,physical,target,move,write,
            hit.RegisteredContext(self.f.data,self.account.device),eligible=active)
        tape=public.condition(tape,1-actor,target,write.new_hp,write.maximum,j.zeros(b,j.int32),eligible=active)
        return tape,ticket,actor,physical,target,move,write,active

    def capture(self,span):
        tape,ticket,*values,active=span
        return e.capture_direct_hit_write(tape,ticket,*values,requested=active)

    def test_exact_controls_or_plain_hp_capture_and_b8_inactive_globals(self):
        for critical,num,den,length in ((True,2,1,3),(True,1,2,3),(True,2,2,2),(False,1,1,1)):
            span=self.span(critical=critical,num=num,den=den)
            result=jax.jit(lambda t,k,a,p,v,m,w,x:e.capture_direct_hit_write(t,k,a,p,v,m,w,requested=x))(*span)
            self.assertEqual(result.execution.ledger.stage[:,0].tolist(),[2,2])
            self.assertEqual(result.execution.ledger.end[:,0].tolist(),[1+length]*2)
            self.assertEqual(result.refusal.tolist(),[0,0])
        active=j.array([True,False]*4);span=self.span(8,active);result=self.capture(span)
        self.assertEqual(result.execution.ledger.count.tolist(),active.astype(j.int32).tolist())
        np.testing.assert_array_equal(result.execution.ledger.inventory_binding,self.account.device.inventory_binding)
        self.assertEqual(result.execution.ledger.stage[:,0].tolist(),[2,0]*4)

    def test_source_span_mutations_reused_ticket_and_no_output_refuse(self):
        span=self.span();tape=span[0]
        for index,column in ((0,5),(1,5),(1,6),(2,7),(3,4),(3,5),(3,6)):
            bad=tape._replace(words=tape.words.at[0,index,column].add(1))
            result=self.capture((bad,*span[1:]))
            self.assertNotEqual(int(result.execution.ledger.refusal[0]),0)
            self.assertEqual(int(result.execution.ledger.refusal[1]),0)
            self.assertEqual(result.execution.ledger.stage[:,0].tolist(),[1,2])
        swapped=tape._replace(words=tape.words.at[0,1].set(tape.words[0,2]).at[0,2].set(tape.words[0,1]))
        self.assertNotEqual(int(self.capture((swapped,*span[1:])).execution.ledger.refusal[0]),0)
        captured=self.capture(span)
        self.assertNotEqual(int(self.capture((captured,*span[1:])).execution.ledger.refusal[0]),0)
        empty=tape._replace(count=j.zeros(2,j.int32))
        self.assertTrue(bool((self.capture((empty,*span[1:])).execution.ledger.refusal!=0).all()))
        changed=tape._replace(event_turn=tape.event_turn.at[0,2].add(1))
        self.assertNotEqual(int(self.capture((changed,*span[1:])).execution.ledger.refusal[0]),0)


if __name__=='__main__':unittest.main()
