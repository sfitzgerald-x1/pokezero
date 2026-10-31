"""Literal Leftovers write/namespace controls; no games/models/public port."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_leftovers_callbacks as c, engine_public_events as public
from gpu_engine import engine_lifecycle_callbacks as life, items, mechanics
import test_gpu_engine_engine_own_facts as fixtures
import test_gpu_engine_engine_item_callbacks as tape_fixtures


class LeftoversCallbacksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeviceOwnFactsTests.setUpClass();cls.f=fixtures.DeviceOwnFactsTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.compiled=c.compile_leftovers(cls.metadata,engine_catalog=cls.f.lifecycle)
        cls.data=c.check_compiled(cls.compiled)
        cls.item=cls.metadata['vocabulary']['items'].index('leftovers')

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def tape(self,batch=1,capacity=16,prior=0):
        state=tape_fixtures.state(batch)._replace(active=j.broadcast_to(j.array([2,4],j.int32),(batch,2)))
        cursor=tape_fixtures.tape(state).cursor._replace(
            engine_binding=j.broadcast_to(self.data.engine_binding,(batch,8)))
        return life._tape(cursor,self.f.lifecycle,capacity)._replace(refusal=j.full(batch,prior,j.int32))

    def heal(self,tape,*,old=95,new=101,maximum=101,status=0,item=None,side=0,physical=2,eligible=True):
        return c.record_leftovers_heal_device(tape,side,physical,self.item if item is None else item,
            old,new,maximum,status,self.data,eligible=eligible)

    def test_literal_item_identity_all_statuses_and_source_stamps(self):
        tape=self.tape(7,prior=128)
        tape=tape._replace(cursor=tape.cursor._replace(generation=j.full((7,2),3,j.int32)))
        result=self.heal(tape,status=j.arange(7,dtype=j.int32))
        self.assertEqual(result.count.tolist(),[1]*7)
        self.assertEqual(result.words[:,0,8].tolist(),list(range(7)))
        self.assertEqual(result.words[0,0].tolist(),[17,-1,-1,0,2,95,101,101,0,self.item])
        self.assertEqual(result.refusal.tolist(),[128|256]*7)
        self.assertEqual(result.event_turn[:,0].tolist(),tape.cursor.turn.tolist())
        self.assertEqual(result.event_phase[:,0].tolist(),tape.cursor.phase.tolist())
        self.assertEqual(result.event_generation[:,0].tolist(),[[3,3]]*7)
        self.assertFalse(bool((result.words[:,:,0]==public.DIRECT_FAINT).any()))

    def test_actual_numerical_write_mask_noop_dead_other_item_and_ineligible(self):
        old=j.array([95,101,0,95,95],j.int32);maximum=j.full(5,101,j.int32)
        held=j.array([self.item,self.item,self.item,2,self.item],j.int32)
        eligible=j.array([True,True,True,True,False])
        new=items.leftovers_heal(self.f.table.items,held,old,maximum,eligible=eligible)
        self.assertEqual(new.tolist(),[101,101,0,95,95])
        result=self.heal(self.tape(5),old=old,new=new,maximum=maximum,item=held,eligible=eligible&(new>old))
        self.assertEqual(result.count.tolist(),[1,0,0,0,0])
        self.assertEqual(result.refusal.tolist(),[256,0,0,0,0])
        self.assertFalse(bool(result.words[1:].any()))

    def test_helper_does_not_recompute_magnitude_or_infer_final_item(self):
        # A literal delta remains exact; future caller/source proof owns amount.
        result=self.heal(self.tape(),old=16777217,new=16777218,maximum=16777219,status=mechanics.FRZ)
        self.assertEqual(result.words[0,0,5:10].tolist(),[16777217,16777218,16777219,mechanics.FRZ,self.item])
        self.assertEqual(result.refusal.tolist(),[256])
        # No BattleState/final held item is accepted by this actual-write API.
        invalid=self.heal(self.tape(),item=2)
        self.assertEqual(invalid.count.tolist(),[0]);self.assertEqual(invalid.refusal.tolist(),[256])
        self.assertFalse(bool(invalid.words.any()))

    def test_invalid_hp_status_item_type_shape_and_noop_claim_are_unknown_only(self):
        for values in (dict(old=0,new=1),dict(old=101,new=101),dict(new=94),dict(new=-1),
                dict(maximum=100),dict(status=-1),dict(status=7),dict(item=0),dict(item=2**31-1),
                dict(old=True),dict(new=j.array([100.5],j.float32)),dict(item=np.array([2**32+1],np.int64)),
                dict(old=j.ones(2,j.int32))):
            with self.subTest(values=str(values)):
                result=self.heal(self.tape(prior=128),**values)
                self.assertEqual(result.count.tolist(),[0]);self.assertEqual(result.refusal.tolist(),[128|256])
                self.assertFalse(bool(result.words.any()))
        old=self.tape(prior=128)
        self.same(self.heal(old,old=None,new=None,item=None,eligible=False),old)

    def test_b8_mixed_current_identity_namespace_eligibility_and_hot_jit(self):
        tape=self.tape(8,prior=128)
        tape=tape._replace(stream_id=tape.stream_id.at[2].set(999),
            cursor=tape.cursor._replace(engine_binding=tape.cursor.engine_binding.at[3,0].add(1),
                initialized=tape.cursor.initialized.at[4].set(False)))
        side=j.arange(8,dtype=j.int32)%2;physical=j.where(side==0,2,4).at[1].set(3)
        held=j.full(8,self.item,j.int32).at[5].set(2)
        eligible=j.array([True]*6+[False,True])
        call=lambda value:c.record_leftovers_heal_device(value,side,physical,held,95,101,101,
            j.arange(8,dtype=j.int32)%7,self.data,eligible=eligible)
        expected=call(tape);fast=jax.jit(call)
        for _ in range(2):self.same(jax.block_until_ready(fast(tape)),expected)
        self.assertEqual(fast._cache_size(),1)
        self.assertEqual(expected.count.tolist(),[1,0,0,0,0,0,0,1])
        self.assertEqual(expected.refusal.tolist(),[128|256]*6+[128,128|256])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            self.assertEqual(getattr(expected,name).shape,(8,));self.same(getattr(expected,name),getattr(tape,name))
            damaged=tape._replace(**{name:getattr(tape,name).at[0].add(1)})
            self.assertFalse(bool(call(damaged).words.any()))
        self.assertEqual(expected.cursor.engine_binding.shape,(8,8))

    def test_clock_metadata_capacity_and_existing_bits_are_never_cleared(self):
        tape=self.tape(prior=128)
        for damaged in (tape._replace(turn=tape.turn+1),
                tape._replace(cursor=tape.cursor._replace(phase=j.ones(1,j.int32))),
                tape._replace(switch_details=tape.switch_details.at[0,0,0].set(1)),
                tape._replace(count=j.array([-1],j.int32))):
            result=self.heal(damaged);self.assertEqual(result.count.tolist(),damaged.count.tolist())
            self.same(result.words,damaged.words);self.assertEqual(result.refusal.tolist(),[128|256])
        initial=self.tape(capacity=1,prior=128)
        first=self.heal(initial);full=self.heal(first)
        self.assertEqual(full.count.tolist(),[1]);self.same(full.words,first.words)
        self.assertEqual(full.refusal.tolist(),[128|256|public.OVERFLOW])
        with patch.object(c,'_unknown',side_effect=lambda tape,eligible:tape):
            result=self.heal(initial)
        self.assertEqual(result.refusal.tolist(),[128])

    def test_original_compiler_metadata_catalog_and_every_data_edge_guard(self):
        with self.assertRaises(ValueError):c.check_compiled(replace(self.compiled))
        with self.assertRaises(ValueError):c.compile_leftovers(self.metadata,engine_catalog=replace(self.f.lifecycle))
        changed=deepcopy(self.metadata);changed['vocabulary']['items'][self.item]='machobrace'
        with self.assertRaises(ValueError):c.compile_leftovers(changed,engine_catalog=self.f.lifecycle)
        original=self.compiled.device
        for name in original._fields:
            value=getattr(original,name)
            for replacement in (np.array(value,copy=True),j.array(value,copy=True)):
                try:
                    object.__setattr__(self.compiled,'device',original._replace(**{name:replacement}))
                    with self.assertRaises(ValueError):c.check_compiled(self.compiled)
                finally:object.__setattr__(self.compiled,'device',original)
        old=self.metadata['schema'];self.metadata['schema']='drift'
        try:
            with self.assertRaises(ValueError):c.check_compiled(self.compiled)
        finally:self.metadata['schema']=old
        self.assertIs(c.check_compiled(self.compiled),original)

    def test_compiler_rejects_original_but_noncanonical_item_flags(self):
        altered=self.f.table._replace(items=self.f.table.items._replace(
            leftovers=self.f.table.items.leftovers.at[self.item].set(False).at[2].set(True)))
        catalog=public.compile_catalog(self.metadata,altered,public_moves=self.f.catalog.moves,dex=self.f.catalog.dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        with self.assertRaisesRegex(ValueError,'canonical Leftovers'):
            c.compile_leftovers(self.metadata,engine_catalog=catalog)

    def test_host_entry_and_strict_structures_and_legacy_abi(self):
        tape=self.tape()
        result=c.record_leftovers_heal(tape,0,2,self.item,95,101,101,0,compiled=self.compiled,eligible=True)
        self.assertEqual(result.count.tolist(),[1]);self.assertEqual(result.refusal.tolist(),[256])
        with self.assertRaises(ValueError):self.heal(tape,eligible=j.ones(2,bool))
        with self.assertRaises(ValueError):self.heal(tape._replace(count=j.zeros(1,j.float32)))
        with self.assertRaises(ValueError):self.heal(tape._replace(engine_binding=j.zeros((1,8),j.uint32)))
        with self.assertRaises(ValueError):c.record_leftovers_heal_device(tape,0,2,self.item,95,101,101,0,True,eligible=True)
        old=public.CallbackTape(*tape[:8])
        self.same(c.record_leftovers_heal_device(old,None,None,None,None,None,None,None,None,eligible='legacy'),old)
        self.assertEqual(public.CallbackTape._fields,('words','count','refusal','turn','engine_binding',
            'source_binding','move_binding','producer_binding'))


if __name__=='__main__':unittest.main()
