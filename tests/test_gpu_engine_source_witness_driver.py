"""Actual owning integration of conditional frame input coverage; fake model only."""
from dataclasses import replace
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_retained_encoder_driver as d, engine_source_frame_witness as w
from gpu_engine import source_provenance
import test_gpu_engine_kernel_risk_runtime as fixtures


class SourceWitnessDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.KernelRiskDriverPreparationTests.setUpClass()
        cls.base=fixtures.KernelRiskDriverPreparationTests
        cls.options=dict(cls.base.options,frame_input_mode='source-witness',
            fake_forward=lambda _:d.FakeScores(j.zeros(9),j.array(1,j.int32)))
        cls.prepared=d.prepare(*cls.base.args,**cls.options)
        cls.fixture=cls.base.base.f

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
            np.testing.assert_array_equal(x,y)

    def construct(self,b=2):
        arrays=self.fixture().arrays('pressure','pressure');arrays['item'][:]=0
        arrays['moves'][0,0,0]=self.fixture.ids['raindance']
        arrays['moves'][1,0,0]=self.fixture.ids['sunnyday']
        with TemporaryDirectory() as directory:
            p=Path(directory)/'literal.npz'
            np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(
                json.dumps(self.fixture.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(p)
            return d.construct(pool,np.tile([[0,1]],(b,1)).astype(np.int64),prepared=self.prepared,
                stream_ids=j.arange(8101,8101+b,dtype=j.int32),seed=37)

    def test_original_mode_guards_no_prepared_cycle_and_default_stays_closed(self):
        r=d._checked(self.prepared);self.assertIsNotNone(r['frame_witness_compiled'])
        self.assertIsNone(d._checked(self.base.prepared)['frame_witness_compiled'])
        for options in (dict(frame_input_mode='yes'),dict(frame_input_mode=True),
                dict(track_kernel_risks=False),dict(account_source_sites=False),dict(public_mode='basic')):
            with self.assertRaises(ValueError):d.prepare(*self.base.args,**dict(self.options,**options))
        with self.assertRaises(ValueError):d._checked(replace(self.prepared))
        with patch.object(w,'check_compiled',side_effect=ValueError('witness static drift')):
            # Base-only check must not recurse through the witness.
            self.assertIs(d._checked(self.prepared),r)
            with self.assertRaisesRegex(ValueError,'witness static drift'):d._checked_witness(r)
        ephemeral=d.prepare(*self.base.args,**self.options)
        reference=weakref.ref(ephemeral);compiled=d._checked(ephemeral)['frame_witness_compiled']
        del ephemeral;gc.collect();self.assertIsNone(reference())
        with self.assertRaises(ValueError):w.check_compiled(compiled)

    def test_actual_constructor_and_weather_step_score_only_after_both_folds(self):
        session=self.construct();r=d._checked(self.prepared)
        self.assertEqual(session.state.core.frame_inputs.frame.tolist(),[1,1])
        self.assertEqual(session.state.core.frame_input_report.frame_inputs_complete.tolist(),[True,True])
        for owner in session.result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(owner.policy_admitted.tolist(),[True,True])
            self.assertEqual(owner.fake_calls.tolist(),[1,1])
            self.assertFalse(bool(owner.full_encoder_admitted.any()))
            self.assertFalse(bool(owner.B4_qualified.any()))
        encode=r['encode_fast'];seen=[]
        def verify_before_encode(core,observer):
            self.assertEqual(core.frame_inputs.frame.tolist(),[2,1])
            self.assertEqual(core.frame_input_report.frame_inputs_complete.tolist(),[True,True])
            occupied=j.arange(core.execution.stage.shape[1])<core.execution.count[0]
            self.assertTrue(bool(((core.execution.stage[0]==3) | ~occupied).all()))
            seen.append(int(observer.owner[0]));return encode(core,observer)
        # Inspect the joined original ledger at the real encode boundary.
        r['encode_fast']=verify_before_encode
        try:
            result=d.advance(session,j.zeros((2,2),j.int32),active_rows=j.array([True,False]),
                chance_inputs=self.base.base().chance())
        finally:r['encode_fast']=encode
        self.assertEqual(seen,[0,1])
        self.assertEqual(result.state.core.frame_inputs.frame.tolist(),[2,1])
        self.assertEqual(result.state.core.frame_input_report.refusal.tolist(),[0,0])
        for owner in result.owners:
            self.assertEqual(owner.policy_admitted.tolist(),[True,True])
            self.assertEqual(owner.fake_calls.tolist(),[2,1])
            self.assertFalse(bool(owner.B4_qualified.any()))

    def test_session_and_timing_static_mutation_fail_before_next_phase(self):
        session=self.construct()
        with self.assertRaises(ValueError):d.advance(replace(session),j.zeros((2,2),j.int32))
        r=d._checked(self.prepared);compiled=r['frame_witness_compiled']
        def mutate(name,milliseconds):
            if name=='engine':object.__setattr__(compiled,'device',compiled.device._replace(
                binding=compiled.device.binding.at[7].add(j.uint32(1))))
        before=compiled.device
        try:
            with self.assertRaises(ValueError):d.advance(session,j.zeros((2,2),j.int32),
                chance_inputs=self.base.base().chance(),timing_callback=mutate)
        finally:object.__setattr__(compiled,'device',before)

    def test_core_witness_transport_rejects_broadcasts_and_keeps_b8_identity(self):
        binding=d._checked(self.prepared)['frame_witness_compiled'].device.binding
        before=w.Carry(binding,j.arange(1,9,dtype=j.int32),j.zeros(8,j.int32),
            j.zeros((8,2),j.int32),j.arange(8,dtype=j.int32))
        active=j.array([True,False]*4)
        returned=before._replace(frame=j.ones(8,j.int32),refusal=j.zeros(8,j.int32),
            binding=binding.at[7].add(j.uint32(1)))
        core=d.Core(j.zeros(8,j.int32),j.zeros(8,j.int32),(),(),(),(),(),
            j.ones(8,bool),j.zeros(8,j.int32),frame_inputs=before)
        selected=jax.jit(d._core_rows)(active,core,core._replace(frame_inputs=returned))
        self.same(selected.frame_inputs.binding,binding)
        self.same(selected.frame_inputs.stream_id,before.stream_id)
        self.same(selected.frame_inputs.frame,before.frame)
        self.same(selected.frame_inputs.refusal,before.refusal|j.where(active,w.SOURCE,0))
        malformed=(dict(frame=j.array(1,j.int32)),dict(refusal=j.array(0,j.int32)),
            dict(stream_id=j.array(1,j.int32)),dict(sequence=j.zeros(2,j.int32)),
            dict(binding=binding[None,:]),dict(frame=j.zeros(8,j.float32)),
            dict(sequence=j.zeros((8,2),j.float32)))
        for bad in malformed:
            for old,new in ((before._replace(**bad),before),(before,before._replace(**bad))):
                with self.subTest(bad=tuple(bad)),self.assertRaises(ValueError):
                    w.commit_rows(active,old,new)
        with self.assertRaises(ValueError):w.commit_rows(j.array(True),before,before)


if __name__=='__main__':unittest.main()
