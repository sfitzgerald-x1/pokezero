"""Private partial hook-history transport; literal arrays, no games/models."""
from dataclasses import replace
import gc
import unittest
from unittest.mock import patch
import weakref

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_status_hook_history as h, engine_status_hook_source as hooks
from gpu_engine.state import empty_state
import test_gpu_engine_status_hook_source as source_fixture


class StatusHookHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_fixture.StatusHookSourceTests.setUpClass();cls.f=source_fixture.StatusHookSourceTests
        cls.prepared=h.prepare_history(cls.f.compiled);cls.tables=h.check_prepared(cls.prepared)
        cls.v=cls.f.metadata['vocabulary']

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def state(self,b=2):
        state=empty_state(b,seed=19)
        return state._replace(active=j.broadcast_to(j.array([2,4],j.int32),(b,2)),
            species=j.ones((b,2,6),j.int32),ability=j.full((b,2,6),self.v['abilities'].index('pressure'),j.int32),
            item=j.zeros((b,2,6),j.int32))

    def streams(self,b=2):return j.arange(701,701+b,dtype=j.int32)

    def initialize(self,state=None):
        state=self.state() if state is None else state
        return h.initialize_device(state,self.streams(len(state.active)),self.tables)

    def retain(self,ledger,state=None,*,frame=0,event=1,code=h.Opportunity.ENTRY,eligible=True,streams=None,**kwargs):
        b=len(ledger.stream_id);state=self.state(b) if state is None else state
        return h.retain_device(ledger,state,self.streams(b) if streams is None else streams,
            frame,event,code,self.tables,eligible=eligible,**kwargs)

    def test_initialize_observes_both_nonphysical_actives_only_not_bench_execution(self):
        state=self.state(3);lum=self.v['items'].index('lumberry');shed=self.v['abilities'].index('shedskin')
        state=state._replace(item=state.item.at[0,1,4].set(lum).at[2,0,5].set(lum),
            ability=state.ability.at[1,0,2].set(shed))
        result=self.initialize(state);report=h.report_device(result,self.tables)
        self.assertEqual(report.transport_consistent.tolist(),[True]*3)
        self.assertTrue(int(result.observed_relevant_flags[0,1])&hooks.UPDATE)
        self.assertTrue(int(result.observed_relevant_flags[1,0])&hooks.RESIDUAL)
        self.assertEqual(result.observed_relevant_flags[2].tolist(),[0,0])
        self.assertTrue(bool(result.seen_items[0,1,lum]));self.assertFalse(bool(result.seen_items[2,:,lum].any()))
        self.assertTrue(bool(result.seen_abilities[1,0,shed]))
        self.assertEqual(result.attempt_count.tolist(),[0]*3)
        self.assertFalse(bool(result.opportunity_counts.any()))
        self.assertEqual(report.pending.tolist(),[h.PENDING_MASK]*3)
        for absent in ('admitted','full_source','full_hooks','b4_qualified'):self.assertFalse(hasattr(report,absent))

    def test_transient_lum_consumption_reversion_and_leftovers_classes_are_sticky_bitsets(self):
        state=self.state();ledger=self.initialize(state)
        lum=self.v['items'].index('lumberry');left=self.v['items'].index('leftovers')
        before=state._replace(item=state.item.at[0,1,4].set(lum))
        first=self.retain(ledger,before,code=h.Opportunity.UPDATE)
        second=self.retain(first,state,event=2,code=h.Opportunity.POST_WRITE)
        third=self.retain(second,state._replace(item=state.item.at[0,1,4].set(left)),event=3,code=h.Opportunity.RESIDUAL)
        for value in (first,second,third):
            self.assertTrue(bool(value.seen_items[0,1,lum]))
            self.assertTrue(int(value.observed_classes[0,1])&(1<<hooks.RELEVANT_UNREVIEWED))
            self.assertTrue(int(value.observed_relevant_flags[0,1])&hooks.UPDATE)
        self.assertTrue(int(third.observed_classes[0,1])&(1<<hooks.LEFTOVERS_TRANSPORT_CANDIDATE))
        self.assertTrue(bool(third.seen_items[0,1,left]))
        self.assertEqual(third.opportunity_counts[0].tolist(),[0,0,0,1,1,0,1,0])
        self.assertEqual(third.attempt_count.tolist(),[3,3])
        self.assertTrue(bool(h.report_device(third,self.tables).transport_consistent.all()))
        self.assertTrue(bool((h.report_device(third,self.tables).pending!=0).all()))

    def test_all_party_namespace_membership_padding_and_unknown_never_becomes_absence(self):
        state=self.state(4)
        bad=state._replace(item=state.item.at[0,1,5].set(999),ability=state.ability.at[1,0,5].set(-1),
            species=state.species.at[2,1,5].set(999),active=state.active.at[3,0].set(6))
        result=self.initialize(bad)
        self.assertEqual(result.sticky_refusal.tolist(),[h.MEMBERSHIP]*4)
        self.assertEqual(h.report_device(result,self.tables).transport_consistent.tolist(),[False]*4)
        repaired=self.retain(result,state)
        self.assertEqual(repaired.sticky_refusal.tolist(),[h.MEMBERSHIP]*4)
        self.assertFalse(bool(h.report_device(repaired,self.tables).transport_consistent.any()))
        padding=state._replace(species=state.species.at[:,:,5].set(0),ability=state.ability.at[:,:,5].set(0))
        self.assertEqual(self.initialize(padding).sticky_refusal.tolist(),[0]*4)
        invalid=padding._replace(item=padding.item.at[0,0,5].set(self.v['items'].index('leftovers')))
        self.assertTrue(int(self.initialize(invalid).sticky_refusal[0])&h.MEMBERSHIP)

    def test_serial_replay_gaps_foreign_stream_and_bad_opportunity_latch_without_advancing(self):
        ledger=self.initialize();first=self.retain(ledger)
        second=self.retain(first,event=2,code=h.Opportunity.STATUS)
        third=self.retain(second,frame=1,event=1,code=h.Opportunity.UPDATE)
        self.assertEqual(third.frame_serial.tolist(),[1,1]);self.assertEqual(third.event_serial.tolist(),[1,1])
        cases=(dict(frame=0,event=1),dict(frame=0,event=3),dict(frame=2,event=1),dict(frame=-1,event=2),
            dict(event=0),dict(event=True),dict(event=np.array([2**32+1]*2,np.uint64)),dict(code=9))
        for kwargs in cases:
            with self.subTest(kwargs=str(kwargs)):
                value=self.retain(first,**kwargs)
                self.assertTrue(bool((value.sticky_refusal!=0).all()))
                self.same(value.frame_serial,first.frame_serial);self.same(value.event_serial,first.event_serial)
                self.assertEqual(value.attempt_count.tolist(),[2,2])
        wrong=self.retain(first,event=2,streams=self.streams().at[0].set(999))
        self.assertEqual(wrong.sticky_refusal.tolist(),[h.NAMESPACE,0])
        self.assertEqual(wrong.event_serial.tolist(),[1,2])
        self.assertEqual(self.retain(wrong,event=j.array([2,3],j.int32)).sticky_refusal.tolist(),[h.NAMESPACE,0])

    def test_unique_positive_streams_device_refusal_and_checked_boundary_rejection(self):
        state=self.state()
        for streams in (j.array([1,1],j.int32),j.array([0,2],j.int32),j.array([-1,2],j.int32)):
            with self.subTest(streams=str(streams)):
                result=h.initialize_device(state,streams,self.tables)
                self.assertTrue(bool((result.sticky_refusal&h.NAMESPACE).any()))
                with self.assertRaises(ValueError):h.initialize(state,streams,prepared=self.prepared)
        h.initialize(state,self.streams(),prepared=self.prepared)
        ledger=self.initialize()
        duplicate=self.retain(ledger,streams=j.array([701,701],j.int32))
        self.assertEqual(duplicate.sticky_refusal.tolist(),[0,h.NAMESPACE])

    def test_move_declaration_uses_only_explicit_current_selection_and_actor(self):
        state=self.state();ledger=self.initialize(state)
        move=self.v['moves'].index('toxic');side=j.array([0,1],j.int32)
        first=self.retain(ledger,code=h.Opportunity.MOVE_DECLARE,selected_move=move,actor_side=side)
        self.assertEqual(first.sticky_refusal.tolist(),[0,0])
        self.assertTrue(bool(first.seen_moves[0,0,move]));self.assertTrue(bool(first.seen_moves[1,1,move]))
        self.assertFalse(bool(first.seen_moves[0,1].any()))
        self.assertEqual(first.opportunity_counts[:,7].tolist(),[1,1])
        # last_move can change arbitrarily; it is never a chosen move receipt.
        changed=state._replace(last_move=j.full_like(state.last_move,move))
        self.same(self.retain(ledger,state),self.retain(ledger,changed))
        for kwargs in ({},{'selected_move':0,'actor_side':0},{'selected_move':999,'actor_side':0},
                {'selected_move':move,'actor_side':2},{'selected_move':True,'actor_side':0}):
            value=self.retain(ledger,code=h.Opportunity.MOVE_DECLARE,**kwargs)
            self.assertEqual(value.sticky_refusal.tolist(),[h.SELECTED_MOVE]*2)
            self.assertEqual(value.event_serial.tolist(),[0,0])
            self.assertFalse(bool(value.seen_moves.any()))
            self.assertEqual(value.opportunity_counts[:,7].tolist(),[1,1])

    def test_b8_inactive_rows_and_all_global8_bindings_are_preserved_hot_jit(self):
        state=self.state(8);ledger=self.initialize(state);active=j.array([True,False]*4)
        state=state._replace(item=state.item.at[0,1,4].set(self.v['items'].index('lumberry')))
        call=lambda value,actual:self.retain(value,actual,eligible=active)
        expected=call(ledger,state);fast=jax.jit(call)
        for _ in range(2):self.same(fast(ledger,state),expected)
        self.assertEqual(fast._cache_size(),1)
        for name in h.GLOBALS:self.assertIs(getattr(expected,name),getattr(ledger,name))
        for name in h.Ledger._fields[5:]:self.same(getattr(expected,name)[~active],getattr(ledger,name)[~active])
        bad_inactive_streams=self.streams(8).at[1].set(701).at[3].set(-1)
        self.same(self.retain(ledger,state,eligible=active,streams=bad_inactive_streams),expected)
        drift=ledger._replace(engine_binding=ledger.engine_binding.at[7].add(1))
        refused=call(drift,state)
        self.assertEqual(refused.sticky_refusal.tolist(),[h.NAMESPACE,0]*4)
        for name in h.GLOBALS:self.same(getattr(refused,name),getattr(drift,name))
        # Walk structured jaxpr primitives, including nested call/control-flow.
        names=[];visited=set()
        def walk(value):
            if id(value) in visited:return
            visited.add(id(value))
            if hasattr(value,'eqns'):
                for eq in value.eqns:
                    names.append(eq.primitive.name)
                    for v in eq.params.values():walk(v)
            elif hasattr(value,'jaxpr'):walk(value.jaxpr)
            elif isinstance(value,(tuple,list)):
                for v in value:walk(v)
            elif isinstance(value,dict):
                for v in value.values():walk(v)
        walk(jax.make_jaxpr(call)(ledger,state))
        self.assertFalse(set(names)&{'io_callback','pure_callback','debug_callback'})

    def test_overflow_malformed_ledger_and_prior_bits_never_clear(self):
        ledger=self.initialize()
        overflow=ledger._replace(attempt_count=j.full(2,h.MAX_SERIAL,j.int32),
            opportunity_counts=ledger.opportunity_counts.at[:,0].set(h.MAX_SERIAL),
            sticky_refusal=j.array([128,64],j.int32),last_opportunity=j.ones(2,j.int32))
        value=self.retain(overflow)
        self.assertEqual(value.attempt_count.tolist(),[h.MAX_SERIAL]*2)
        self.assertEqual(value.opportunity_counts[:,0].tolist(),[h.MAX_SERIAL]*2)
        self.assertEqual(value.sticky_refusal.tolist(),[128|h.OVERFLOW,64|h.OVERFLOW])
        self.assertEqual(value.event_serial.tolist(),[0,0])
        capped=ledger._replace(event_serial=j.full(2,h.MAX_SERIAL,j.int32))
        value=self.retain(capped,event=h.MAX_SERIAL)
        self.assertTrue(bool((value.sticky_refusal&h.OVERFLOW).all()))
        for bad in (ledger._replace(attempt_count=j.full(2,-1,j.int32)),
                ledger._replace(opportunity_counts=ledger.opportunity_counts.at[:,0].set(-1)),
                ledger._replace(observed_classes=j.full((2,2),32,j.int32)),
                ledger._replace(last_opportunity=j.full(2,9,j.int32)),
                ledger._replace(attempt_count=j.ones(2,j.int32),last_opportunity=j.ones(2,j.int32),
                    opportunity_counts=j.ones((2,8),j.int32))):
            self.assertFalse(bool(h.report_device(bad,self.tables).transport_consistent.any()))
            self.assertTrue(bool((self.retain(bad).sticky_refusal&h.MALFORMED).all()))

    def test_original_preparation_edges_all_closed_guards_and_no_owning_copy_claim(self):
        with self.assertRaises(ValueError):h.check_prepared(replace(self.prepared))
        with self.assertRaises(ValueError):h.prepare_history(replace(self.f.compiled))
        old=self.prepared.device
        try:
            object.__setattr__(self.prepared,'device',old._replace(binding=j.array(old.binding,copy=True)))
            with self.assertRaises(ValueError):h.check_prepared(self.prepared)
        finally:object.__setattr__(self.prepared,'device',old)
        state=self.state();ledger=self.initialize(state)
        calls=(lambda:h.prepare_history(self.f.compiled),lambda:h.check_prepared(self.prepared),
            lambda:h.initialize(state,self.streams(),prepared=self.prepared),
            lambda:h.retain(ledger,state,self.streams(),0,1,h.Opportunity.ENTRY,prepared=self.prepared,eligible=True),
            lambda:h.report(ledger,prepared=self.prepared))
        for call in calls:
            for dummy in (False,True):
                fn=(lambda _:call()) if dummy else (lambda:call())
                with patch.object(hooks,'check_compiled',side_effect=AssertionError('late source guard')):
                    with self.assertRaisesRegex(ValueError,'cannot be traced'):
                        jax.jit(fn)(j.array(1)) if dummy else jax.jit(fn)()
        # Pure lower carry can be copied, by design. No origin/once claim is
        # manufactured by a Python object identity check inside cached JIT.
        copied=ledger._replace()
        self.same(self.retain(copied),self.retain(ledger))
        self.assertTrue(bool((h.report_device(copied,self.tables).pending!=0).all()))
        temporary=h.prepare_history(self.f.compiled);weak=weakref.ref(temporary)
        del temporary;gc.collect();self.assertIsNone(weak())


if __name__=='__main__':unittest.main()
