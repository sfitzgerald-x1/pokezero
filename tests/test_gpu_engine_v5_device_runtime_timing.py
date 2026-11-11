"""Literal counters and cached-call fake scopes; no actual engine/model/game."""
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
from gpu_engine import v5_device_timing as timing,v5_device_backend as backend,v5_device_rollout as rollout,v5_model
from tests import test_gpu_engine_v5_device_backend as fixtures
from tests import test_gpu_engine_v5_device_actor_bank as actors


class DeviceRuntimeTimingTests(unittest.TestCase):
    def test_literal_occupied_lanes_denominators_and_unknown_rates(self):
        count=timing.Counts(2)
        requested=np.array([[[1,1],[0,0]],[[0,1],[1,0]],[[0,0],[1,1]]],bool)
        count.add(requested)
        r=count.report(np.array([rollout.COMPLETE,rollout.CAPPED]),kernel_seconds=2.,collection_seconds=8.)
        self.assertEqual((r['physical_game_lane_steps'],r['occupied_game_lane_steps'],r['requested_seat_views']),(6,4,6))
        self.assertEqual(r['completed_per_kernel_second'],.5);self.assertEqual(r['lane_occupancy_corrected_completed_per_kernel_second'],.75)
        self.assertEqual(r['completed_per_collection_wall_second'],.125);self.assertEqual(r['completed_game_request_steps'],dict(mean=2.,p99=2.))
        empty=timing.Counts(2).report(np.array([rollout.REFUSED,rollout.RUNNING]),kernel_seconds=0.,collection_seconds=1.)
        self.assertIsNone(empty['completed_per_kernel_second']);self.assertIsNone(empty['lane_occupancy_corrected_completed_per_kernel_second'])
        self.assertIsNone(empty['completed_game_request_steps'])
        for a in (requested.astype(np.int32),requested[:,:,:1]):
            with self.assertRaises(ValueError):count.add(a)

    def test_aggregate_uses_raw_counts_not_mean_of_bank_rates(self):
        first=timing.Counts(2);first.add(np.ones((2,2,2),bool))
        second=timing.Counts(1);second.add(np.ones((1,1,2),bool))
        a=first.report(np.array([1,2]),kernel_seconds=1.,collection_seconds=2.)
        b=second.report(np.array([1]),kernel_seconds=3.,collection_seconds=4.)
        report=timing.aggregate([a,b],[dict(status='COMPLETE',requests=2),dict(status='CAPPED',requests=2000),dict(status='COMPLETE',requests=10)])
        self.assertEqual(report['complete'],2);self.assertEqual(report['completed_per_kernel_second'],.5)
        self.assertEqual(report['completed_engine_request_steps']['mean'],6.)
        self.assertEqual(report['completed_engine_request_steps']['p99'],9.92)
        self.assertEqual(report['physical_batch_rows'],[2,1]);self.assertEqual(report['physical_game_lane_steps'],5)
        self.assertEqual(timing.aggregate([],[])['status'],'UNMEASURED_NO_RUNTIME_REPORTS')

    def test_explicit_compile_once_then_actual_chunk_only_and_cache_scope(self):
        with tempfile.TemporaryDirectory()as root:
            plan,worker,store,snapshots,assignments,carry=fixtures.DeviceBackendTests().setup_backend(root)
            with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),\
                    patch.object(worker,'_actor',side_effect=lambda snap,store:worker.actors.setdefault(snap['archive_sha256'],({'inert':np.array(0)},{}))[0]),\
                    patch.object(rollout,'make_collector',return_value=None):
                prepared=worker.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
            raw,_=actors.DeviceActorBankTests().run_inert(prepared['bank'],bindings=prepared['bindings']);calls=[]
            class Lowered:
                def compile(self):
                    calls.append('compile')
                    def execute(carry,params):calls.append('execute');return raw
                    return execute
            class Collector:
                def lower(self,carry,params):calls.append('lower');return Lowered()
            prepared['collector']=Collector();worker.measure_runtime=True
            with patch.object(worker,'_construct',return_value=carry),patch.object(rollout,'_complete_gae',side_effect=lambda v,o,l,discount:(v*0,v*0)):
                worker.collect(prepared,store,lambda *a,**kw:None,lambda:None)
            self.assertEqual(calls,['lower','compile','execute'])
            report=store.matching('DEVICE_RUNTIME_TIMING')[0]
            self.assertEqual(report['physical_game_lane_steps'],6)
            self.assertEqual(report['occupied_game_lane_steps'],int(np.any(raw.frames.requested,axis=2).sum()))
            self.assertEqual(report['complete'],2);self.assertEqual(report['phases']['collector_compile_calls'],1)
            for name in ('collector_lower_seconds','collector_compile_seconds','kernel_blocked_seconds','spool_write_wall_seconds','game_cache_wall_seconds','cache_merge_wall_seconds'):
                self.assertGreaterEqual(report['phases'][name],0.)
            self.assertIn('compiled_collector',prepared)

    def test_original_actor_source_guard_accepts_exact_current_contract(self):
        v5_model._sources()  # Only file identity, never model loading.

if __name__=='__main__':unittest.main()
