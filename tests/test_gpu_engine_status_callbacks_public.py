"""Literal actual-core status writes into existing FULL consumers, no games.

Light replacement coverage for engine_status_callbacks' tests (its fixture API is tests/gpu_engine_status_callbacks_fixture.py): the same
tests and assertions, with compare's causal projection and reference folds on one compiled program each
(tests/gpu_engine_shared_public_jits.py, test-local routing) instead of eager scans. The hot-jit test is
test_gpu_engine_status_callbacks_hot_jit.
"""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import jax
import jax.numpy as j

from pokezero.dex import ShowdownDex
from pokezero.category_vocab import CategoryVocabulary
from gpu_engine import transition as t, engine_lifecycle_callbacks as life, engine_public_events as events
from gpu_engine import engine_public_event_device as base, engine_full_v4_event_device as full
from gpu_engine import engine_status_callbacks as status, public_wire, public_inputs as inputs, v4_encoder as v4
import gpu_engine_shared_public_jits as shared
import gpu_engine_status_callbacks_fixture as fixture


class StatusCallbacksPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StatusCallbacksTests.setUpClass();base=fixture.StatusCallbacksTests
        for attr in ('f','metadata','ids','catalog','table','producer','compiled'):setattr(cls,attr,getattr(base,attr))
        cls.fast=staticmethod(base.fast)   # the fixture's one compiled project_and_fold (class access unwraps it)

    kwargs=fixture.StatusCallbacksTests.kwargs
    initial=fixture.StatusCallbacksTests.initial
    chance=fixture.StatusCallbacksTests.chance
    step=fixture.StatusCallbacksTests.step
    same=fixture.StatusCallbacksTests.same
    start=fixture.StatusCallbacksTests.start
    compare=shared.compare(fixture)

    def test_actual_primary_status_exact_full_words_both_observers_and_legacy_rng(self):
        frame=self.initial()
        for side in (0,1):
            state,current=self.start(frame,j.full(2,side,j.int32))
            result,wire=self.compare(state,frame,current,[public_wire.WireState()]*2)
            actual=self.step(frame,[[0,0],[0,0]])
            self.assertEqual(actual.callbacks.words[0,:int(actual.callbacks.count[0]),0].tolist(),[1,9,7,5,4])
            self.assertEqual(actual.callbacks.words[0,1].tolist(),[9,-1,-1,1,0,0,1,0,0,0])
            self.assertEqual(actual.callbacks.refusal.tolist(),[events.UNINSTRUMENTED]*2)
            result,_=self.compare(result.projection.state,actual,result.public,wire)
            self.assertEqual(result.public.mons.status[:,1,0].tolist(),[2,2])
            self.assertEqual(result.public.context.sleep.status[:,1,0].tolist(),[2,2])

    def test_actual_sleep_then_wake_has_old_status_hp_before_cure_then_move(self):
        frame=self.initial('sleeppowder')
        state,current=self.start(frame,j.array([0,1],j.int32))
        result,wire=self.compare(state,frame,current,[public_wire.WireState()]*2)
        first=self.step(frame,[[0,0],[0,0]])
        result,wire=self.compare(result.projection.state,first,result.public,wire)
        self.assertEqual(result.public.mons.status[:,1,0].tolist(),[1,1])
        second=self.step(first,[[1,0],[1,0]])
        kinds=second.callbacks.words[0,:int(second.callbacks.count[0]),0].tolist()
        self.assertIn(status.STATUS_CURED,kinds)
        cure=kinds.index(status.STATUS_CURED)
        self.assertEqual(second.callbacks.words[0,cure,5:8].tolist(),[5,0,status.WAKE])
        prior_hp=kinds.index(events.MAIN_HP)
        self.assertLess(prior_hp,cure)
        self.assertEqual(second.callbacks.words[0,prior_hp,7].item(),5)
        self.assertEqual(kinds[cure+1],events.MOVE)
        result,_=self.compare(result.projection.state,second,result.public,wire)
        self.assertEqual(result.public.mons.status[:,1,0].tolist(),[0,0])

    def test_valid_enum_wrong_old_status_cure_rolls_back_only_affected_row(self):
        frame=self.initial('sleeppowder');state,p=self.start(frame,j.array([0,1],j.int32))
        initial=self.fast(state,frame,p)
        first=self.step(frame,[[0,0],[0,0]])
        asleep=self.fast(initial.projection.state,first,initial.public)
        second=self.step(first,[[1,0],[1,0]])
        kinds=np.array(second.callbacks.words[0,:,0]);index=int(np.flatnonzero(kinds==status.STATUS_CURED)[0])
        wrong=second.callbacks.words.at[0,index,5].set(6).at[0,index,7].set(status.THAW)
        tampered=second._replace(callbacks=second.callbacks._replace(words=wrong))
        # This deliberately passes the enum/reason grammar; only prior causal
        # public status distinguishes the false freeze cure from actual wake.
        lower=shared.lower(self.compiled.causal.device)(asleep.projection.state.causal,tampered)
        self.assertEqual(lower.error.tolist(),[0,0])
        result=self.fast(asleep.projection.state,tampered,asleep.public)
        self.assertEqual(result.projection.error.tolist(),[int(inputs.Error.IDENTITY),0])
        self.assertFalse(bool(result.projection.chunk.valid[0].any()))
        self.assertTrue(bool(result.projection.chunk.valid[1].any()))
        self.assertEqual(result.public.cursor.sequence[0].item(),asleep.public.cursor.sequence[0].item())
        self.assertEqual(result.public.mons.status[:,1,0].tolist(),[1,0])

    def test_unproved_lum_and_nonzero_entry_prefix_refuse_entire_rows(self):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['item'][1,0]=self.metadata['vocabulary']['items'].index('lumberry')
        frame=self.initial(arrays=arrays)
        actual=self.step(frame,[[0,0],[0,0]])
        self.assertFalse(bool((actual.callbacks.refusal & status.UNKNOWN_STATUS).any()))
        self.assertEqual(actual.state.status[:,1,0].tolist(),[0,0])
        state,current=self.start(frame,j.array([0,1],j.int32))
        initial=self.fast(state,frame,current)
        refused=self.fast(initial.projection.state,actual,initial.public)
        self.assertFalse(bool(refused.projection.chunk.valid.any()))
        self.same(refused.public,initial.public)
        unsupported=self.initial(status_value=3)  # real Immunity entry cure
        self.assertTrue(bool((unsupported.callbacks.refusal & status.UNKNOWN_STATUS).all()))
        registered,public=self.start(unsupported,j.array([0,1],j.int32))
        result=self.fast(registered,unsupported,public)
        self.assertFalse(bool(result.projection.chunk.words.any()))

    def test_private_counterfactuals_and_mixed_unknown_do_not_supply_public_facts(self):
        baseline=self.initial()
        baseline_step=self.step(baseline,[[0,0],[0,0]])
        for owner in (0,1):
            arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}; peer=1-owner
            arrays['stats'][peer,:,0]+=37
            arrays['stats'][peer,:,1:5]+=23  # preserve speed and causal order
            arrays['item'][peer]=0
            arrays['ability'][peer]=np.roll(arrays['ability'][peer],1)
            arrays['hidden_power_type'][peer]=9;arrays['hidden_power_power'][peer]=31
            changed=self.initial(arrays=arrays)
            changed_step=self.step(changed,[[0,0],[0,0]])
            owners=j.full(2,owner,j.int32)
            a,p=self.start(baseline,owners);b,q=self.start(changed,owners)
            left=self.fast(a,baseline,p);right=self.fast(b,changed,q)
            self.same(left.projection.chunk,right.projection.chunk);self.same(left.public,right.public)
            left=self.fast(left.projection.state,baseline_step,left.public)
            right=self.fast(right.projection.state,changed_step,right.public)
            self.assertEqual(left.projection.error.tolist(),[0,0]);self.assertEqual(right.projection.error.tolist(),[0,0])
            self.same(left.projection.chunk,right.projection.chunk);self.same(left.public,right.public)
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['item'][1,0]=self.metadata['vocabulary']['items'].index('lumberry')
        frame=self.initial(arrays=arrays)
        state,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(state,frame,p)
        mixed=self.step(frame,[[0,0],[1,0]])
        self.assertEqual((mixed.callbacks.refusal & status.UNKNOWN_STATUS).tolist(),[0,0])
        result=self.fast(initial.projection.state,mixed,initial.public)
        self.assertNotEqual(result.projection.error[0].item(),0)
        self.assertEqual(result.projection.error[1].item(),0)
        self.assertFalse(bool(result.projection.chunk.valid[0].any()))
        self.assertTrue(bool(result.projection.chunk.valid[1].any()))
        self.assertEqual(result.public.cursor.sequence[0].item(),initial.public.cursor.sequence[0].item())
        self.assertEqual(self.fast._cache_size(),1)

    def test_actual_nested_reflection_is_unknown_before_private_ability_attribution(self):
        arrays={k:np.array(v,copy=True) for k,v in self.f.arrays.items()}
        arrays['ability'][1,0]=self.metadata['vocabulary']['abilities'].index('synchronize')
        frame=self.initial(arrays=arrays)
        state,p=self.start(frame,j.array([0,1],j.int32));initial=self.fast(state,frame,p)
        actual=self.step(frame,[[0,0],[0,0]])
        self.assertEqual(actual.state.status[:,:,0].tolist(),[[1,1],[1,1]])
        self.assertTrue(bool((actual.callbacks.refusal & status.UNKNOWN_STATUS).all()))
        result=self.fast(initial.projection.state,actual,initial.public)
        self.assertFalse(bool(result.projection.chunk.words.any()))
        self.same(result.public,initial.public)


if __name__=='__main__':unittest.main()
