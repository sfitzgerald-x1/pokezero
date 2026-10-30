"""Partial PRIVATE transport accounting; no totality/source-semantic claim."""
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_execution_completeness as a
from gpu_engine import engine_public_events as event
from gpu_engine import engine_lifecycle_callbacks as life
from gpu_engine import engine_full_v4_event_device as full
from test_gpu_engine_engine_full_v4_event_device import FullV4EventDeviceTests as Fixture


class ExecutionAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.fixture=Fixture()
        cls.adapter=Fixture.compiled
        cls.compiled=a.compile_accounting(cls.adapter)
        cls.tables=cls.compiled.device
        cls.frame=cls.fixture.initial()

    def same(self,left,right):
        for x,y in zip(jax.tree.leaves(left),jax.tree.leaves(right)):
            np.testing.assert_array_equal(x,y)

    def ledger(self,b=2,capacity=4):
        return a.initialize(j.arange(1,b+1,dtype=j.int32),compiled=self.compiled,capacity=capacity)

    def tape(self,b=2):
        t=self.frame.callbacks
        # Explicit trusted receipt fixtures, not actual owning session proof.
        def repeat(x):return j.repeat(x[:1],b,axis=0)
        cursor=jax.tree.map(repeat,t.cursor)._replace(stream_id=j.arange(1,b+1,dtype=j.int32),
            turn=j.ones(b,j.int32),phase=j.zeros(b,j.int32),generation=j.zeros((b,2),j.int32))
        return t._replace(words=j.zeros((b,64,10),j.int32),count=j.zeros(b,j.int32),
            refusal=j.ones(b,j.int32),turn=cursor.turn,stream_id=cursor.stream_id,cursor=cursor,
            event_turn=j.zeros((b,64),j.int32),event_phase=j.zeros((b,64),j.int32),
            event_generation=j.zeros((b,64,2),j.int32),switch_details=j.zeros((b,64,6),j.int32))

    def begin(self,b=2,capacity=4,active=None):
        active=j.ones(b,bool) if active is None else active
        l=self.ledger(b,capacity);t=self.tape(b)
        return a.begin_frame(l,t,frame=j.ones(b,j.int32),active=active,tables=self.tables),t

    def enter(self,l,t,active=None,site=a.Site.LEFTOVERS):
        b=l.stream_id.shape[0];active=j.ones(b,bool) if active is None else active
        return a.enter(l,t,j.full(b,int(site),j.int32),j.zeros(b,j.int32),active=active,tables=self.tables)

    def append(self,t,active=None):
        b=t.count.shape[0];active=j.ones(b,bool) if active is None else active
        return event.condition(t,j.zeros(b,j.int32),j.zeros(b,j.int32),
            j.full(b,90,j.int32),j.full(b,100,j.int32),j.zeros(b,j.int32),eligible=active)

    def capture(self,l,t,active=None):
        l,key=self.enter(l,t,active);t=self.append(t,active)
        l=a.emitted(l,key,t,active=j.ones(l.stream_id.shape[0],bool) if active is None else active,tables=self.tables)
        return l,key,t

    def test_unresolved_no_write_and_transient_unknown_never_clear(self):
        for site in (a.Site.WISH,a.Site.SPEED_BOOST,a.Site.PINCH_ITEM,a.Site.STATUS_RESIDUAL,
                     a.Site.LEECH_DRAIN,a.Site.LEECH_HEAL,a.Site.ENTRY_HAZARD,a.Site.INTIMIDATE_ATTEMPT):
            with self.subTest(site=site):
                l,t=self.begin();l,key=self.enter(l,t,site=site)
                # Actual blocked/no-write attempt is still entered; net-zero
                # or later reversal has no API that erases its obligation.
                self.assertEqual(a.report(l,tables=self.tables).unresolved.tolist(),[1,1])
                refused=a.unhandled(l,key,active=j.ones(2,bool),tables=self.tables)
                self.assertTrue(bool(j.all(refused.refusal & a.UNKNOWN_SITE)))
                later,key2,t=self.capture(refused,t)
                self.assertTrue(bool(j.all(later.refusal & a.UNKNOWN_SITE)))
                self.assertFalse(bool(a.report(later,tables=self.tables).scoped_drained.any()))
                # An empty span cannot masquerade as a source no-output proof.
                no_output=a.emitted(l,key,t._replace(count=j.zeros(2,j.int32)),active=j.ones(2,bool),tables=self.tables)
                self.assertTrue(bool(j.all(no_output.refusal & a.RECEIPT)))

    def test_exact_span_stamps_overlap_and_ticket_replay(self):
        l,t=self.begin();l,key,t=self.capture(l,t)
        self.assertEqual(l.stage[:,0].tolist(),[2,2])
        for bad in (t._replace(event_turn=t.event_turn.at[0,0].add(1)),
                    t._replace(event_phase=t.event_phase.at[0,0].add(1)),
                    t._replace(event_generation=t.event_generation.at[0,0,1].add(1)),
                    t._replace(words=t.words.at[0,0,0].set(0))):
            original,blank=self.begin();original,k=self.enter(original,blank)
            result=a.emitted(original,k,bad,active=j.ones(2,bool),tables=self.tables)
            self.assertTrue(int(result.refusal[0]) & a.RECEIPT)
            self.assertEqual(int(result.refusal[1]),0)
        replay=a.emitted(l,key,t,active=j.ones(2,bool),tables=self.tables)
        self.assertTrue(bool(j.all(replay.refusal & a.RECEIPT)))
        # Two attempts entered at the same count cannot spend the same receipt.
        original,blank=self.begin();original,k1=self.enter(original,blank);original,k2=self.enter(original,blank)
        appended=self.append(blank)
        first=a.emitted(original,k1,appended,active=j.ones(2,bool),tables=self.tables)
        second=a.emitted(first,k2,appended,active=j.ones(2,bool),tables=self.tables)
        self.assertTrue(bool(j.all(second.refusal & a.RECEIPT)))

    def test_overflow_serial_freshness_and_no_reset_of_history(self):
        l,t=self.begin(capacity=1);l,key=self.enter(l,t)
        overflow,_=self.enter(l,t)
        self.assertTrue(bool(j.all(overflow.refusal & a.OVERFLOW)))
        serial=l._replace(next_serial=j.full(2,2**31-1,j.int32),count=j.zeros(2,j.int32))
        serial,_=self.enter(serial,t)
        self.assertTrue(bool(j.all(serial.refusal & a.OVERFLOW)))
        # Fresh-frame reset preserves lifetime serial/refusal; forged/old ticket
        # must not resolve a later occurrence at the same slot.
        closed=l._replace(opened=j.zeros(2,bool),refusal=j.full(2,a.UNKNOWN_SITE,j.int32))
        fresh=a.begin_frame(closed,t,frame=j.full(2,2,j.int32),active=j.ones(2,bool),tables=self.tables)
        self.assertEqual(fresh.refusal.tolist(),[a.UNKNOWN_SITE]*2)
        self.assertEqual(fresh.next_serial.tolist(),l.next_serial.tolist())
        reused=a.emitted(fresh,key,self.append(t),active=j.ones(2,bool),tables=self.tables)
        self.assertTrue(bool(j.all(reused.refusal & a.RECEIPT)))
        for frame in (j.ones(2,j.int32),j.full(2,4,j.int32)):
            stale=a.begin_frame(closed,t,frame=frame,active=j.ones(2,bool),tables=self.tables)
            self.assertTrue(bool(j.all(stale.refusal & a.ORDER)))

    def test_b8_global_namespaces_and_bitwise_inactive_rows(self):
        mask=j.array([True,False,True,False,True,False,True,False])
        old=self.ledger(8);t=self.tape(8)
        l=a.begin_frame(old,t,frame=j.ones(8,j.int32),active=mask,tables=self.tables)
        l,key,t=self.capture(l,t,mask)
        for name in l._fields:
            if name=='inventory_binding':
                self.same(getattr(l,name),getattr(old,name));continue
            self.same(getattr(l,name)[~mask],getattr(old,name)[~mask])
        # GLOBAL word7 affects every active row; B8 must not turn it into row7.
        bad=t._replace(source_binding=t.source_binding.at[7].add(j.uint32(1)))
        result=a.emitted(l,key,bad,active=mask,tables=self.tables)
        self.assertEqual((result.refusal & a.NAMESPACE).tolist(),[a.NAMESPACE if x else 0 for x in mask.tolist()])
        poisoned=t._replace(words=t.words.at[1,0].set(999),count=t.count.at[1].set(-1),
            refusal=t.refusal.at[1].set(1024))
        untouched=a.emitted(l,key,poisoned,active=j.zeros(8,bool),tables=self.tables)
        self.same(l,untouched)
        with self.assertRaises(ValueError):
            a.report(l,tables=self.tables._replace(source_binding=self.tables.source_binding.astype(j.int32)))

    def test_cached_device_entry_capture_b8_exact_and_no_host_callbacks(self):
        mask=j.array([True,False,True,False,True,False,True,False])
        old=self.ledger(8);t=self.tape(8)
        def path(ledger,tape,active):
            ledger=a.begin_frame(ledger,tape,frame=j.ones(8,j.int32),active=active,tables=self.tables)
            ledger,key=a.enter(ledger,tape,j.full(8,int(a.Site.LEFTOVERS),j.int32),j.zeros(8,j.int32),
                active=active,tables=self.tables)
            tape=event.condition(tape,j.zeros(8,j.int32),j.zeros(8,j.int32),j.full(8,90,j.int32),
                j.full(8,100,j.int32),j.zeros(8,j.int32),eligible=active)
            return a.emitted(ledger,key,tape,active=active,tables=self.tables)
        expected=path(old,t,mask)
        fast=jax.jit(path)
        self.same(fast(old,t,mask),expected)
        self.same(fast(old,t,mask),expected)
        text=str(jax.make_jaxpr(path)(old,t,mask))
        for callback in ('io_callback','pure_callback','debug_callback'):self.assertNotIn(callback,text)

    def transport_fixture(self):
        # Capture actual immutable initialization receipts0..1, followed by
        # actual TURN1 at2. This tests exact transport draining ONLY: the test's
        # site label is synthetic and does not register initial reveals as a
        # hazard or prove the missing semantic site/emitter mapping.
        frame=self.frame;t=frame.callbacks
        before=t._replace(count=j.zeros(2,j.int32),cursor=t.cursor._replace(turn=j.zeros(2,j.int32)))
        l=a.initialize(frame.cursor.stream_id,compiled=self.compiled,capacity=4)
        l=a.begin_frame(l,before,frame=j.ones(2,j.int32),active=j.ones(2,bool),tables=self.tables)
        l,key=self.enter(l,before,site=a.Site.ENTRY_HAZARD)
        prefix=t._replace(count=j.full(2,2,j.int32),cursor=before.cursor)
        l=a.emitted(l,key,prefix,active=j.ones(2,bool),tables=self.tables)
        stream,public=self.fixture.observer(frame,j.array([0,1],j.int32))
        return l,frame,stream,public

    def test_actual_full_consumption_required_and_atomic_rollback(self):
        l,frame,stream,public=self.transport_fixture()
        fast=jax.jit(lambda ledger,fr,st,pub,active:a.consume_frame_device(ledger,fr,st,pub,
            active=active,tables=self.tables,adapter_tables=self.adapter.device,catalog=self.adapter.catalog))
        result,consumed=fast(l,frame,stream,public,j.ones(2,bool))
        self.assertEqual(a.report(result,tables=self.tables).scoped_drained.tolist(),[True,True])
        self.assertTrue(bool(j.all(result.upstream_refusal & event.UNINSTRUMENTED)))
        self.assertFalse(bool(consumed.projection.full_encoder_admitted.any()))
        # Corrupt exact captured content AFTER emission: whole bad row rolls
        # back even if full source parser would otherwise accept the changed HP.
        bad=l._replace(words=l.words.at[0,0,5].add(1))
        rejected,out=fast(bad,frame,stream,public,j.ones(2,bool))
        self.assertTrue(int(rejected.refusal[0]) & a.RECEIPT)
        self.assertEqual(int(rejected.refusal[1]),0)
        self.same(jax.tree.map(lambda x:x[0],stream),jax.tree.map(lambda x:x[0],out.projection.state))
        self.assertEqual(int(out.public.cursor.sequence[0]),int(public.cursor.sequence[0]))
        self.assertGreater(int(out.public.cursor.sequence[1]),int(public.cursor.sequence[1]))
        # Consuming a valid later bundle cannot erase an earlier unhandled
        # transient/no-write attempt or commit that row's consumer state.
        sticky=l._replace(refusal=j.array([a.UNKNOWN_SITE,0],j.int32))
        later,out=fast(sticky,frame,stream,public,j.ones(2,bool))
        self.assertTrue(int(later.refusal[0]) & a.UNKNOWN_SITE)
        self.assertEqual(out.projection.component_admitted.tolist(),[False,True])
        self.assertEqual(int(out.public.cursor.sequence[0]),int(public.cursor.sequence[0]))
        # Inactive rows never advance or lose private state, even during a
        # genuine full fold of the independent active peer.
        inactive,out=fast(l,frame,stream,public,j.array([False,True]))
        for name in l._fields:
            self.same(getattr(inactive,name) if name=='inventory_binding' else getattr(inactive,name)[0],
                      getattr(l,name) if name=='inventory_binding' else getattr(l,name)[0])
        self.assertEqual(int(out.public.cursor.sequence[0]),int(public.cursor.sequence[0]))
        # Replaying consumption cannot spend this frame a second time.
        replay,out=fast(result,frame,stream,public,j.ones(2,bool))
        self.assertTrue(bool(j.all(replay.refusal & a.CONSUMPTION)))
        self.assertFalse(bool(out.projection.component_admitted.any()))

    def test_missing_consumer_drain_and_unresolved_obligation_refuse(self):
        l,frame,stream,public=self.transport_fixture()
        genuine=full.project_and_fold(stream,frame,self.adapter.device,public,self.adapter.catalog)
        # Independently fault the ACTUAL consumer result, never pass an
        # accepted boolean into the accounting API.
        for broken in (genuine._replace(public=public),
                       genuine._replace(projection=genuine.projection._replace(error=j.array([7,0],j.int32))),
                       genuine._replace(projection=genuine.projection._replace(frames=genuine.projection.frames._replace(
                           valid=j.zeros_like(genuine.projection.frames.valid))))):
            with patch.object(full,'project_and_fold',return_value=broken):
                result,out=a.consume_frame(l,frame,stream,public,active=j.ones(2,bool),compiled=self.compiled)
            self.assertTrue(int(result.refusal[0]) & a.CONSUMPTION)
            self.assertFalse(bool(a.report(result,tables=self.tables).scoped_drained[0]))
        unresolved=l._replace(stage=l.stage.at[0,0].set(1))
        result,out=a.consume_frame(unresolved,frame,stream,public,active=j.ones(2,bool),compiled=self.compiled)
        self.assertTrue(int(result.refusal[0]) & a.UNRESOLVED)
        self.assertEqual(int(out.public.cursor.sequence[0]),int(public.cursor.sequence[0]))
        self.assertEqual(int(result.refusal[1]),0)

    def test_original_compiler_and_no_exhaustiveness_or_no_output_escape(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):a.initialize(j.array([1,2],j.int32),compiled=replace(self.compiled))
        with self.assertRaises(ValueError):a.initialize(j.array([1,1],j.int32),compiled=self.compiled)
        self.assertIn('original-source-hooks-outside-engine-mutations',a.INTEGRATION_REQUIRED)
        self.assertNotIn('full_encoder_admitted',a.Report._fields)
        self.assertNotIn('B4_qualified',a.Report._fields)
        self.assertFalse(hasattr(a,'resolve_no_output'))
        self.assertFalse(hasattr(a,'resolve'))
        l=self.ledger()
        self.assertEqual(a.report(l,tables=self.tables).scoped_drained.tolist(),[False,False])


if __name__=='__main__':unittest.main()
