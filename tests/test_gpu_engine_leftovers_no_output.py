"""Literal actual-kernel full-HP path; no original-hook/physics admission."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_leftovers_no_output as n, engine_execution_completeness as a
from gpu_engine import engine_lifecycle_callbacks as life, engine_weather_speed_profile as speed
from gpu_engine import items
import test_gpu_engine_leftovers_public as fixture


class LeftoversNoOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.LeftoversPublicTests.setUpClass();cls.f=fixture.LeftoversPublicTests
        cls.accounting=a.compile_accounting(cls.f.f.compiled)
        cls.compiled=n.compile_no_output(cls.f.heal,profile=cls.f.profile,accounting_compiled=cls.accounting)
        cls.tables=n.check_compiled(cls.compiled)
        cls.initial=cls.f.f().initial('hail',arrays=cls.f.f().arrays('pressure','pressure'))

    def same(self,x,y):
        for left,right in zip(jax.tree.leaves(x),jax.tree.leaves(y)):np.testing.assert_array_equal(left,right)

    def boundary(self,b=2,status=0,side=None,count=0):
        # Explicit lower-seam arrays, not a constructor-origin attestation.
        tile=lambda x:j.repeat(x[:1],b,axis=0)
        state=jax.tree.map(tile,self.initial.state)
        streams=j.arange(1101,1101+b,dtype=j.int32)
        state=state._replace(hp=state.max_hp,status=j.full_like(state.status,status),
            turn=j.zeros(b,j.int32),done=j.zeros(b,bool),active=j.broadcast_to(j.array([2,4],j.int32),(b,2)))
        record=speed.qualify_birth_device(state,streams,self.tables.profile)
        cursor=jax.tree.map(tile,self.initial.cursor)._replace(stream_id=streams,
            physical_active=state.active,turn=j.ones(b,j.int32),phase=j.zeros(b,j.int32))
        rows=j.arange(b)[:,None];sides=j.arange(2)[None,:]
        mapping=j.full((b,2,6),-1,j.int32).at[rows,sides,state.active].set(0)
        cursor=cursor._replace(physical_to_public=mapping,public_slot=j.zeros((b,2),j.int32),
            encounter=j.ones((b,2),j.int32),next_slot=j.ones((b,2),j.int32),
            revealed_species_id=state.species[rows,sides,state.active])
        # Ordinary residual executes with engine.turn still0 while the causal
        # public cursor is1; the kernel increments engine.turn only at return.
        tape=life._tape(cursor,self.f.f.producer,64)._replace(count=j.full(b,count,j.int32))
        ledger=a.initialize(streams,compiled=self.accounting)
        ledger=a.begin_frame(ledger,tape,frame=j.ones(b,j.int32),active=j.ones(b,bool),tables=self.accounting.device)
        side=j.arange(b,dtype=j.int32)%2 if side is None else j.full(b,side,j.int32)
        tape=tape._replace(execution=a.Context(ledger,self.accounting.device))
        tape,ticket=a.open_attempt(tape,a.Site.LEFTOVERS,side,active=j.ones(b,bool))
        return tape,ticket,state,record,2+side

    def capture(self,tape,ticket,state,record,event,requested=True):
        return n.capture_before_device(tape,ticket,state,event,record,self.tables,requested=requested)

    def actual_write(self,state,event):
        b=len(state.active);rows=j.arange(b);side=event%2;p=state.active[rows,side]
        hp=state.hp[rows,side,p];maximum=state.max_hp[rows,side,p];item=state.item[rows,side,p]
        # ORIGINAL numerical leaf + SAME literal HP assignment; no local amount
        # formula or inference that an arbitrary unchanged HP implies no output.
        healed=items.leftovers_heal(self.f.f.table.items,item,hp,maximum,eligible=j.ones(b,bool))
        return state._replace(hp=state.hp.at[rows,side,p].set(healed))

    def test_exact_actual_kernel_full_hp_both_sides_all_statuses_no_public_writes(self):
        for status in range(7):
            tape,ticket,state,record,event=self.boundary(status=status)
            before=self.capture(tape,ticket,state,record,event)
            self.assertEqual(before.reason.tolist(),[0,0])
            after=self.actual_write(state,event)
            receipt=n.finish_device(before,tape,after,record,self.tables)
            self.assertEqual(receipt.reason.tolist(),[0,0])
            self.assertEqual(receipt.before.side.tolist(),[0,1]);self.assertEqual(receipt.after_physical.tolist(),[2,4])
            self.same(receipt.before.hp,receipt.after_hp);self.same(state,after)
            self.assertEqual(receipt.after_count.tolist(),[0,0])
            self.same(receipt.upstream_refusal,tape.refusal)
            self.assertFalse(hasattr(receipt,'admitted'));self.assertFalse(hasattr(receipt,'accepted'))
            # Helper NEVER resolves accounting or changes the public tape.
            self.assertEqual(tape.execution.ledger.stage[:,0].tolist(),[1,1])
            self.assertEqual(a.report(tape.execution.ledger,tables=self.accounting.device).unresolved.tolist(),[1,1])

    def test_not_generic_noop_nonfull_dead_noncanonical_item_or_other_event_refuse(self):
        tape,ticket,state,record,event=self.boundary();rows=j.arange(2);side=event%2;p=state.active[rows,side]
        states=(state._replace(hp=state.hp.at[rows,side,p].add(-1)),
            state._replace(hp=state.hp.at[rows,side,p].set(0)),
            state._replace(item=state.item.at[rows,side,p].set(0)),
            state._replace(status=state.status.at[rows,side,p].set(7)))
        for bad in states:
            before=self.capture(tape,ticket,bad,record,event)
            result=n.finish_device(before,tape,bad,record,self.tables)
            self.assertTrue(bool((result.reason&n.BEFORE).all()))
        for wrong in (0,1,4,9,True):
            before=self.capture(tape,ticket,state,record,wrong)
            self.assertTrue(bool((before.reason&n.BRANCH).all()))
        inactive=self.capture(tape,ticket,state,record,event,requested=j.array([False,True]))
        self.assertEqual(inactive.reason.tolist(),[n.INACTIVE,0])

    def test_ticket_site_side_serial_stamps_active_and_profile_fail_closed(self):
        tape,ticket,state,record,event=self.boundary()
        for field,value in (('serial',ticket.serial.at[0].add(1)),('frame',ticket.frame.at[0].add(1)),
                ('stream_id',ticket.stream_id.at[0].add(1)),('slot',ticket.slot.at[0].set(-1))):
            before=self.capture(tape,ticket._replace(**{field:value}),state,record,event)
            self.assertTrue(int(before.reason[0])&n.ATTEMPT);self.assertEqual(int(before.reason[1]),0)
        ledger=tape.execution.ledger
        for field,value in (('site',ledger.site.at[0,0].set(int(a.Site.WISH))),
                ('side',ledger.side.at[0,0].set(1)),('stage',ledger.stage.at[0,0].set(2)),
                ('turn',ledger.turn.at[0,0].add(1)),('generation',ledger.generation.at[0,0,0].add(1))):
            bad=tape._replace(execution=tape.execution._replace(ledger=ledger._replace(**{field:value})))
            before=self.capture(bad,ticket,state,record,event)
            self.assertTrue(int(before.reason[0])&n.ATTEMPT);self.assertEqual(int(before.reason[1]),0)
        wrong=state._replace(active=state.active.at[0,0].set(3))
        self.assertTrue(int(self.capture(tape,ticket,wrong,record,event).reason[0])&n.BEFORE)
        before=self.capture(tape,ticket,state,record._replace(qualified_birth=j.array([False,True])),event)
        self.assertEqual(before.reason.tolist(),[n.PROFILE,0])
        namespace=tape._replace(engine_binding=tape.engine_binding.at[7].add(1))
        self.assertTrue(bool((self.capture(namespace,ticket,state,record,event).reason&n.NAMESPACE).all()))
        namespace=tape._replace(move_binding=tape.move_binding.at[7].add(1))
        self.assertTrue(bool((self.capture(namespace,ticket,state,record,event).reason&n.NAMESPACE).all()))

    def test_after_same_write_and_zero_output_are_literal_not_delta_only(self):
        tape,ticket,state,record,event=self.boundary();before=self.capture(tape,ticket,state,record,event)
        rows=j.arange(2);side=event%2;p=state.active[rows,side]
        for field,array in (('item',state.item.at[0,0,2].set(0)),('status',state.status.at[0,0,2].set(2)),
                ('max_hp',state.max_hp.at[0,0,2].add(1)),('hp',state.hp.at[0,0,2].add(-1)),
                ('active',state.active.at[0,0].set(3)),('turn',state.turn.at[0].add(1))):
            result=n.finish_device(before,tape,state._replace(**{field:array}),record,self.tables)
            self.assertTrue(int(result.reason[0])&n.AFTER);self.assertEqual(int(result.reason[1]),0)
        for bad in (tape._replace(count=tape.count.at[0].add(1)),
                tape._replace(words=tape.words.at[0,63,0].set(17)),
                tape._replace(event_turn=tape.event_turn.at[0,63].add(1)),
                tape._replace(event_phase=tape.event_phase.at[0,0].add(1)),
                tape._replace(event_generation=tape.event_generation.at[0,0,1].add(1)),
                tape._replace(switch_details=tape.switch_details.at[0,0,1].add(1)),
                tape._replace(refusal=tape.refusal.at[0].set(0))):
            result=n.finish_device(before,bad,state,record,self.tables)
            self.assertTrue(int(result.reason[0])&n.OUTPUT);self.assertEqual(int(result.reason[1]),0)
        spent=tape.execution.ledger._replace(stage=tape.execution.ledger.stage.at[0,0].set(2))
        result=n.finish_device(before,tape._replace(execution=tape.execution._replace(ledger=spent)),state,record,self.tables)
        self.assertTrue(int(result.reason[0])&n.ATTEMPT)
        # Corrupting the snapshot's branch facts is revalidated, not trusted as
        # reason=0 or an external 'accepted' boolean.
        forged=before._replace(hp=j.zeros(2,j.int32),maximum=j.zeros(2,j.int32))
        self.assertTrue(bool((n.finish_device(forged,tape,state,record,self.tables).reason&n.BEFORE).all()))
        # Literal integers beyond float precision remain exact. If the ORIGINAL
        # numerical leaf itself wraps near int32 max, a full-HP input alone is
        # insufficient: AFTER equality refuses that actual changed result.
        for maximum in (16777221,2**31-1):
            literal=state._replace(hp=j.full_like(state.hp,maximum),max_hp=j.full_like(state.max_hp,maximum))
            snapshot=self.capture(tape,ticket,literal,record,event)
            actual=self.actual_write(literal,event)
            result=n.finish_device(snapshot,tape,actual,record,self.tables)
            if maximum==16777221:
                self.assertEqual(result.after_hp.tolist(),[maximum]*2);self.assertEqual(result.reason.tolist(),[0,0])
            else:self.assertTrue(bool((result.reason&n.AFTER).all()))

    def test_b8_global_namespace_hot_jit_capacity_prior_bits_and_no_callbacks(self):
        tape,ticket,state,record,event=self.boundary(8,count=64)
        tape=tape._replace(refusal=j.full(8,1|256,j.int32))
        call=lambda t,s:n.finish_device(self.capture(t,ticket,s,record,event),t,self.actual_write(s,event),record,self.tables)
        expected=call(tape,state);fast=jax.jit(call)
        for _ in range(2):self.same(fast(tape,state),expected)
        self.assertEqual(fast._cache_size(),1);self.assertEqual(expected.reason.tolist(),[0]*8)
        self.assertEqual(expected.upstream_refusal.tolist(),[257]*8)
        bad=tape._replace(source_binding=tape.source_binding.at[7].add(1))
        self.assertTrue(bool((call(bad,state).reason&n.NAMESPACE).all()))
        primitives=[];visited=set()
        def walk(x):
            if id(x) in visited:return
            visited.add(id(x))
            if hasattr(x,'eqns'):
                for eq in x.eqns:
                    primitives.append(eq.primitive.name)
                    for value in eq.params.values():walk(value)
            elif hasattr(x,'jaxpr'):walk(x.jaxpr)
            elif isinstance(x,(list,tuple)):
                for value in x:walk(value)
            elif isinstance(x,dict):
                for value in x.values():walk(value)
        walk(jax.make_jaxpr(call)(tape,state))
        self.assertFalse(set(primitives)&{'pure_callback','io_callback','debug_callback'})

    def test_original_host_holders_edges_and_all_closed_apis(self):
        with self.assertRaises(ValueError):n.check_compiled(replace(self.compiled))
        with self.assertRaises(ValueError):n.compile_no_output(replace(self.f.heal),profile=self.f.profile,accounting_compiled=self.accounting)
        old=self.compiled.device
        try:
            object.__setattr__(self.compiled,'device',old._replace(binding=j.array(old.binding,copy=True)))
            with self.assertRaises(ValueError):n.check_compiled(self.compiled)
        finally:object.__setattr__(self.compiled,'device',old)
        tape,ticket,state,record,event=self.boundary();before=self.capture(tape,ticket,state,record,event)
        calls=(lambda:n.compile_no_output(self.f.heal,profile=self.f.profile,accounting_compiled=self.accounting),
            lambda:n.check_compiled(self.compiled),
            lambda:n.capture_before(tape,ticket,state,event,record,compiled=self.compiled,requested=True),
            lambda:n.finish(before,tape,state,record,compiled=self.compiled))
        for call in calls:
            for dummy in (False,True):
                fn=(lambda _:call()) if dummy else (lambda:call())
                with patch.object(n.leftovers,'check_compiled',side_effect=AssertionError('late guard')):
                    with self.assertRaisesRegex(ValueError,'cannot be traced'):
                        jax.jit(fn)(j.array(1)) if dummy else jax.jit(fn)()


if __name__=='__main__':unittest.main()
