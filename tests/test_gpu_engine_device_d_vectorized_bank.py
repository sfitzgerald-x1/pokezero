"""Inert bank clocks/counts: no actual engine, model, forward or optimizer."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from gpu_engine import device_d_vectorized_bank as bank,v5_device_backend,v5_device_teacher,v5_model
from pokezero import v5_run as runs
from tests import test_v5_run as fixture

class Backend(fixture.FakeBackend):
    def __init__(self):
        super().__init__();self.teacher=object.__new__(v5_device_teacher.Provider);self.teacher.vectorized=True
        self.teacher_queue_chunks=2;self.measure_runtime=True;self.closed=False
    def collect(self,prepared,store,on_end,check_progress):
        result=super().collect(prepared,store,on_end,check_progress)
        call=dict(frames=[0],frame_wall_seconds=7.,world_prepare_seconds=4.,device_seconds=1.,lower_seconds=1.,compile_seconds=1.,padded_leaf_count=82944)
        store.emit('DEVICE_TEACHER_FRAME_COUNTS',by_frame={'0':dict(requested=2048,selected=512,labeled=480,refused=32,unselected=1536)},frame_calls=[call])
        store.emit('DEVICE_RUNTIME_TIMING',phases=dict(collector_lower_seconds=2.,collector_compile_seconds=3.,initializer_lower_seconds=.1,initializer_compile_seconds=.2),teacher_queue=dict(capacity=2))
        return result
    def close(self):self.closed=True

class BankTests(unittest.TestCase):
    def test_enablement_and_wrong_cohort_refuse_before_any_setup(self):
        with self.assertRaises(ValueError):bank.measure(None,None,backend=None)
        with self.assertRaises(ValueError):bank.measure(fixture.plan('D'),None,backend=None,enable_benchmark=True)
    def run_bank(self,seconds):
        template=fixture.plan('D');games=tuple(runs.Game(i+1,'d3-fixture-'+str(i),100+i,(2*i,2*i+1))for i in range(2048))
        plan=replace(template,iterations=((runs.Batch(games,101,201),),));backend=Backend();training=fixture.FakeTraining()
        def export(checkpoint,out,**kwargs):
            Path(out).write_bytes(b'inert-actor');return dict(weights_archive_sha256=runs._file_sha(out))
        with tempfile.TemporaryDirectory()as directory:
            with patch.object(v5_device_backend,'RuntimeBackend',Backend),patch.object(runs.training,'create',side_effect=training.create),patch.object(runs.training,'save_checkpoint',side_effect=training.save),patch.object(v5_model,'export',side_effect=export),patch.object(runs.training,'train_caches',side_effect=AssertionError('bank must never train')),patch.object(bank.time,'perf_counter',side_effect=[0.,1.,2.,seconds+2.]):
                result=bank.measure(plan,Path(directory)/'run',backend=backend,enable_benchmark=True)
            self.assertTrue(backend.closed);self.assertEqual(training.latest.steps,0)
            self.assertEqual(result['bank_wall_seconds'],seconds);self.assertEqual(result['optimizer_updates'],0)
            self.assertEqual(result['frames']['0']['refused_share_selected'],32/512)
            self.assertEqual(result['frames']['0']['padded_leaf_counts_per_call'],[82944])
            self.assertEqual(result['compile']['collector_compile_seconds'],3.)
            self.assertEqual(result['actor_setup_seconds'],1.)
            return result
    def test_full_bank_clock_includes_compile_and_cache_drain_at_boundary(self):
        result=self.run_bank(2048.)
        self.assertEqual(result['admission'],'PASS');self.assertEqual(result['declared_games_per_second'],1.)
        self.assertEqual(result['dispositions']['COMPLETE'],2048)
    def test_distant_tail_cannot_be_removed_for_a_faster_gate(self):
        result=self.run_bank(2049.)
        self.assertEqual(result['admission'],'FAIL');self.assertLess(result['declared_games_per_second'],1.)

if __name__=='__main__':unittest.main()
