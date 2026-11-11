"""Malformed/failed/inactive self-write spans; no kernel/native/model calls."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_substitute_callbacks as sub,engine_lifecycle_callbacks as life
from gpu_engine import engine_execution_completeness as execution,engine_public_events as events
import test_gpu_engine_substitute_owning as fixtures

class SubstituteGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.SubstituteOwningTests.setUpClass();cls.f=fixtures.SubstituteOwningTests

    def tape(self,b=8):
        f=self.f;c=life.register(j.arange(301,301+b,dtype=j.int32),**f.kw)
        mapping=c.physical_to_public.at[:,0,3].set(0).at[:,1,5].set(0)
        c=c._replace(turn=j.ones(b,j.int32),initialized=j.ones(b,bool),physical_active=j.tile(j.array([3,5],j.int32),(b,1)),
            public_slot=j.zeros((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),
            revealed_species_id=j.tile(j.array([1,2],j.int32),(b,1)),physical_to_public=mapping,next_slot=j.ones((b,2),j.int32))
        t=life._tape(c,f.producer,64)
        ledger=execution.initialize(c.stream_id,compiled=f.account,capacity=16)
        t=execution.begin_on_tape(t,ledger,f.account.device,active=j.ones(b,bool))
        side=j.zeros(b,j.int32);phys=j.full(b,3,j.int32);move=j.ones(b,j.int32)
        t=events.move(t,side,phys,j.full(b,5,j.int32),move,eligible=j.ones(b,bool))
        t,ticket=execution.open_attempt(t,execution.Site.SUBSTITUTE_SELF,side,active=j.ones(b,bool))
        return t,ticket,side,phys,move

    def write(self,b=8):
        return sub.Write(*[j.full(b,n,j.int32) for n in (101,76,101,0,0,25)])

    def test_invalid_noop_weak_existing_lethal_and_wrongmove_never_claim(self):
        t,ticket,side,phys,move=self.tape();write=self.write();ctx=sub.RegisteredContext(self.f.compiled.device,self.f.account.device)
        cases=[write._replace(new_hp=j.full(8,101,j.int32),new_sub_hp=j.zeros(8,j.int32)),
            write._replace(old_sub_hp=j.ones(8,j.int32)),write._replace(new_hp=j.zeros(8,j.int32)),
            write._replace(new_sub_hp=j.full(8,24,j.int32)),write._replace(status=j.full(8,7,j.int32))]
        for w in cases:
            out=sub.record_device(t,side,phys,move,w,ctx,eligible=j.ones(8,bool))
            out=execution.capture_substitute_write(out,ticket,side,phys,move,w,requested=j.ones(8,bool))
            self.assertEqual(out.count.tolist(),[1]*8)
            self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[1]*8)
            self.assertTrue(bool(((out.refusal&256)!=0).all()));self.assertTrue(bool(((out.refusal&512)!=0).all()))
        wrong=sub.record_device(t,side,phys,j.full(8,2,j.int32),write,ctx,eligible=j.ones(8,bool))
        self.assertTrue(bool(((wrong.refusal&256)!=0).all()));self.assertEqual(wrong.count.tolist(),[1]*8)

    def test_exact_span_prefix_currentphysical_stamps_and_prior_flags(self):
        t,ticket,side,phys,move=self.tape();w=self.write();ctx=sub.RegisteredContext(self.f.compiled.device,self.f.account.device)
        out=sub.record_device(t,side,phys,move,w,ctx,eligible=j.ones(8,bool))
        good=execution.capture_substitute_write(out,ticket,side,phys,move,w,requested=j.ones(8,bool))
        self.assertEqual(good.execution.ledger.stage[:,0].tolist(),[2]*8)
        for bad in [out._replace(words=out.words.at[0,0,5].set(2)),
                    out._replace(event_turn=out.event_turn.at[0,1].add(1)),
                    out._replace(cursor=out.cursor._replace(physical_active=out.cursor.physical_active.at[0,0].set(4)))]:
            failed=execution.capture_substitute_write(bad,ticket,side,phys,move,w,requested=j.ones(8,bool))
            self.assertEqual(failed.execution.ledger.stage[:,0].tolist(),[1]+[2]*7)
            self.assertNotEqual(int(failed.execution.ledger.refusal[0]),0)
        prior=t._replace(refusal=t.refusal|256)
        retained=sub.record_device(prior,side,phys,move,w,ctx,eligible=j.ones(8,bool))
        self.assertTrue(bool(((retained.refusal&256)!=0).all()))

    def test_inactive_mixed_overflow_namespace_and_closed_host_jit(self):
        t,ticket,side,phys,move=self.tape();w=self.write();ctx=sub.RegisteredContext(self.f.compiled.device,self.f.account.device)
        mask=j.array([True,True,False,False,False,False,False,False])
        out=sub.record_device(t,side,phys.at[1].set(4),move,w,ctx,eligible=mask)
        self.assertEqual(out.count.tolist(),[2]+[1]*7)
        self.assertTrue(bool(out.refusal[1]&256))
        for a,b in zip(jax.tree.leaves(out),jax.tree.leaves(t)):
            if getattr(a,'ndim',0)>0 and a.shape[0]==8 and a.shape!=(8,):np.testing.assert_array_equal(a[2:],b[2:])
        full=t._replace(count=j.full(8,64,j.int32));overflow=sub.record_device(full,side,phys,move,w,ctx,eligible=j.ones(8,bool))
        self.assertTrue(bool(((overflow.refusal&events.OVERFLOW)!=0).all()));self.assertTrue(bool(((overflow.refusal&256)!=0).all()))
        for field in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
            data=ctx.data._replace(**{field:j.tile(getattr(ctx.data,field),(8,1))})
            with self.subTest(field=field),self.assertRaises(ValueError):
                sub.record_device(t,side,phys,move,w,ctx._replace(data=data),eligible=j.ones(8,bool))
        with self.assertRaises(ValueError):jax.jit(lambda dummy:sub.check_compiled(self.f.compiled))(j.array(0))
        with self.assertRaises(ValueError):jax.jit(lambda dummy:life.trace_step(None,None,None,cursor=None,
            catalog=None,expected_binding=None,expected_engine_binding=None,substitute_compiled=self.f.compiled))(j.array(0))
