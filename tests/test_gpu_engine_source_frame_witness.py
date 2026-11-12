"""Bounded actual source frames/two original folds; no model/game/qualification."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_source_frame_witness as w
from gpu_engine import engine_retained_encoder_driver as d, engine_request_source as source
from gpu_engine import engine_lifecycle_callbacks as life, engine_full_v4_event_device as full
from gpu_engine import engine_kernel_frame_risk as risks, source_provenance, engine_private_actions as actions
from gpu_engine import engine_weather_speed_profile as speed, v4_encoder
import test_gpu_engine_kernel_risk_runtime as fixtures


class SourceFrameWitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.KernelRiskDriverPreparationTests.setUpClass()
        cls.prepared=fixtures.KernelRiskDriverPreparationTests.prepared
        cls.runtime=d._checked(cls.prepared)
        cls.compiled=w.compile_witness(cls.prepared);cls.tables=w.check_compiled(cls.compiled)
        cls.fixture=fixtures.KernelRiskDriverPreparationTests.base.f
        cls.fast=staticmethod(jax.jit(lambda carry,core,frame,streams,publics,active:
            w.consume_pair_device(carry,core,frame,streams,publics,cls.tables,cls.runtime['catalog'],
                active=active)))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def actual(self,b=2):
        r=self.runtime
        arrays=self.fixture().arrays('pressure','pressure')
        arrays['item'][:]=0
        arrays['moves'][0,0,0]=self.fixture.ids['raindance']
        arrays['moves'][1,0,0]=self.fixture.ids['sunnyday']
        with TemporaryDirectory() as directory:
            p=Path(directory)/'literal.npz'
            np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.fixture.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(p)
            built=source.construct_from_pool(pool,np.tile([[0,1]],(b,1)).astype(np.int64),
                compiled=r['request_compiled'],stream_ids=j.arange(7101,7101+b,dtype=j.int32),seed=37,
                _execution_compiled=r['execution_compiled'],_kernel_risk_compiled=r['kernel_risk_compiled'])
        initial=life.register(built.cursor.stream_id,catalog=r['request']['lifecycle'],
            expected_binding=r['request']['lifecycle'].binding,
            expected_engine_binding=r['request']['rules'].device.engine_binding)
        profile=speed.qualify_birth(built.birth_state,built.cursor.stream_id,profile=r['speed_profile'])
        profile=r['retain_speed'](profile,built.state,built.cursor.stream_id)
        core=d.Core(built.state,built.cursor,built.orders,tuple(x.original for x in built.registrations),
            tuple(actions._constructor(x.constructor) for x in built.registrations),
            tuple(x.unsafe_origin for x in built.registrations),built.requests,built.cursor.admitted,
            j.zeros(b,j.int32),profile,built.callbacks.execution.ledger,built.callbacks.execution.risks.carry)
        frame=life.LifecycleResult(built.state,built.cursor,built.callbacks,life.snapshot(initial),
            built.registry,j.zeros(b,bool),j.zeros(b,j.int32),profile)
        streams=tuple(full.register(built.cursor.stream_id,j.full(b,owner,j.int32),compiled=r['public']) for owner in (0,1))
        publics=tuple(v4_encoder.initialize(r['catalog'],stream_ids=built.cursor.stream_id,
            owner_sides=j.full(b,owner,j.int32)) for owner in (0,1))
        return built,core,frame,streams,publics

    def call(self,carry,core,frame,streams,publics,active=None,tables=None):
        return w.consume_pair_device(carry,core,frame,streams,publics,self.tables if tables is None else tables,
            self.runtime['catalog'],active=j.ones(len(carry.stream_id),bool) if active is None else active)

    def test_genuine_constructor_and_actual_weather_frame_complete_then_replay_refuses(self):
        built,core,frame,streams,publics=self.actual()
        carry=w.initialize_from_constructed(self.compiled,built)
        self.assertIs(w.check_host_origin(self.compiled,constructed=built),self.tables)
        fast=lambda *args:self.fast(*args,j.ones(2,bool))
        initial=fast(carry,core,frame,streams,publics)
        self.assertEqual(initial.report.frame_inputs_complete.tolist(),[True,True])
        self.assertEqual(initial.carry.frame.tolist(),[1,1])
        self.assertTrue(bool((initial.report.owner_words>0).all()))
        self.same(initial.report.raw_words,frame.callbacks.words)
        for owner,result in enumerate(initial.owners):
            self.assertEqual(result.projection.state.causal.owner.tolist(),[owner]*2)
            self.assertFalse(bool(result.projection.full_line_consumers_complete.any()))
        # Actual SAME owning kernel function, no synthetic event or accepted fold.
        core=core._replace(execution=initial.ledger)
        chance=fixtures.KernelRiskDriverPreparationTests.base().chance()
        actual=self.runtime['frame'](core,j.zeros((2,2),j.int32),chance,j.ones(2,bool))
        after=core._replace(engine=actual.state,cursor=actual.cursor,execution=actual.callbacks.execution.ledger,
            risks=actual.callbacks.execution.risks.carry,frame_number=j.ones(2,j.int32),
            weather_profile=actual.weather_profile)
        next_streams=tuple(x.projection.state for x in initial.owners)
        next_publics=tuple(x.public for x in initial.owners)
        step=fast(initial.carry,after,actual,next_streams,next_publics)
        self.assertEqual(step.report.frame_inputs_complete.tolist(),[True,True],msg=json.dumps(dict(
            witness_refusal=step.report.refusal.tolist(),tape_refusal=actual.callbacks.refusal.tolist(),
            risk=after.risks.risk.tolist(),expected=step.report.expected_cutpoints.tolist(),
            observed=step.report.observed_cutpoints.tolist(),ledger_refusal=step.ledger.refusal.tolist(),
            raw_count=actual.callbacks.count.tolist(),kinds=actual.callbacks.words[0,:int(actual.callbacks.count[0]),0].tolist())))
        self.assertEqual(step.carry.frame.tolist(),[2,2])
        self.assertEqual(step.report.observed_cutpoints[:,risks.Branch.ENCORE-1].tolist(),[16,16])
        self.assertEqual(step.report.observed_cutpoints[:,risks.Branch.SECONDARY_STATS-1].tolist(),[6,6])
        self.assertEqual(step.report.observed_cutpoints[:,risks.Branch.SECONDARY_FLINCH-1].tolist(),[2,2])
        no_dispatch=self.tables._replace(secondary_weather=j.zeros_like(self.tables.secondary_weather))
        self.assertFalse(bool(w._raw_complete(actual.callbacks,no_dispatch).any()))
        replay=fast(step.carry,after,actual,tuple(x.projection.state for x in step.owners),tuple(x.public for x in step.owners))
        self.assertEqual(replay.report.frame_inputs_complete.tolist(),[False,False])
        self.assertTrue(bool((replay.carry.refusal&w.ORDER).all()))
        self.assertEqual(self.fast._cache_size(),1)

    def test_missing_guard_counts_same_owner_and_second_owner_fault_rollback_both(self):
        built,core,frame,streams,publics=self.actual()
        carry=w.initialize_from_constructed(self.compiled,built)
        risk=frame.callbacks.execution.risks.carry
        bad=risk._replace(counts=risk.counts.at[0,risks.Branch.ENTRY_CONFUSION_CURE-1].add(-1))
        trace=frame.callbacks._replace(execution=frame.callbacks.execution._replace(
            risks=frame.callbacks.execution.risks._replace(carry=bad)))
        fault=frame._replace(callbacks=trace)
        fast=lambda *args:self.fast(*args,j.ones(2,bool))
        result=fast(carry,core._replace(risks=bad),fault,streams,publics)
        self.assertEqual(result.report.frame_inputs_complete.tolist(),[False,True])
        self.assertTrue(int(result.carry.refusal[0])&w.CUTPOINTS)
        for before,after in zip(streams,result.owners):self.same(before.sequence[0],after.projection.state.sequence[0])

        # The actual tape cursor must be exactly the owning frame cursor,
        # including generation even when no selected span mentions it.
        cursor=frame.callbacks.cursor._replace(generation=frame.callbacks.cursor.generation.at[0,0].add(1))
        changed=frame._replace(callbacks=frame.callbacks._replace(cursor=cursor))
        drift=fast(carry,core,changed,streams,publics)
        self.assertEqual(drift.report.frame_inputs_complete.tolist(),[False,True])
        self.assertTrue(int(drift.report.refusal[0])&w.SOURCE)
        for before,after in zip(streams,drift.owners):self.same(before.sequence[0],after.projection.state.sequence[0])
        result=fast(carry,core,frame,(streams[0],streams[0]),publics)
        self.assertFalse(bool(result.report.frame_inputs_complete.any()))
        self.assertTrue(bool((result.carry.refusal&w.DRAIN).all()))
        second=streams[1]._replace(sequence=streams[1].sequence.at[0].add(1))
        result=fast(carry,core,frame,(streams[0],second),publics)
        self.assertEqual(result.report.frame_inputs_complete.tolist(),[False,True])
        self.same(result.owners[0].projection.state.sequence[0],streams[0].sequence[0])
        self.same(result.owners[1].public.cursor.sequence[0],publics[1].cursor.sequence[0])
        inactive=self.fast(carry,core,frame,streams,publics,j.zeros(2,bool))
        self.same(inactive.carry,carry);self.same(inactive.ledger,core.execution)
        for before,after in zip(publics,inactive.owners):self.same(before,after.public)

    def test_b8_global_binding_commit_inactive_and_unclaimed_actual_hp_refuses(self):
        # Small pure transport controls; they do not mint a constructor origin.
        b=8;before=w.Carry(self.tables.binding,j.arange(1,9,dtype=j.int32),j.zeros(b,j.int32),
            j.zeros((b,2),j.int32),j.zeros(b,j.int32))
        active=j.array([True,False]*4)
        after=before._replace(binding=before.binding.at[7].add(1),frame=j.ones(b,j.int32),
            refusal=j.ones(b,j.int32))
        result=jax.jit(w.commit_rows)(active,before,after)
        self.same(result.binding,before.binding);self.same(result.stream_id,before.stream_id)
        self.assertEqual(result.frame.tolist(),[0]*8)
        self.assertEqual(result.refusal.tolist(),[1,0]*4)
        built,core,frame,_,_=self.actual()
        tape=frame.callbacks
        # An unclaimed raw HP row is not complete merely because scoped count0.
        forged=tape._replace(words=tape.words.at[:,0,0].set(2))
        self.assertFalse(bool(w._raw_complete(forged,self.tables).any()))
        self.assertTrue(bool(w._raw_complete(tape,self.tables).all()))

    def test_original_source_host_guards_copies_and_no_new_global_qualification(self):
        built,_,_,_,_=self.actual()
        with self.assertRaises(ValueError):w.check_compiled(replace(self.compiled))
        with self.assertRaises(ValueError):w.check_host_origin(self.compiled,constructed=replace(built))
        with self.assertRaises(ValueError):w.check_host_origin(self.compiled)
        for fn in (lambda:w.compile_witness(self.prepared),lambda:w.check_compiled(self.compiled),
                lambda:w.check_host_origin(self.compiled,constructed=built),
                lambda:w.initialize_from_constructed(self.compiled,built)):
            for dummy in (False,True):
                with patch.object(w,'_source',side_effect=AssertionError('late source read')):
                    with self.assertRaisesRegex(ValueError,'cannot be traced'):
                        jax.jit(lambda _:fn())(j.array(1)) if dummy else jax.jit(lambda:fn())()
        with patch.dict(w.SOURCE_PINS,{'transition.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'actual source inventory changed'):w.check_compiled(self.compiled)


if __name__=='__main__':unittest.main()
