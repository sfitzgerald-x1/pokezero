"""Inert original-cache byte/order and owned data-worker failure controls."""
from concurrent.futures import Future,ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from gpu_engine import v5_device_rollout as d,v5_device_spool as s,v5_cache_pipeline as p
from gpu_engine.v5_cache_worker import Assembler
from pokezero import v5_data,v5_run as runs
from tests.test_gpu_engine_v5_device_spool import DeviceSpoolTests
from tests.test_gpu_engine_v5_device_teacher import inert_d,records
from tests import test_v5_run as rf
from pokezero.v5_loss import V5LossConfig


class LocalWorkers:
    """Same bootstrap adapter, in-process for inert GAE seam only."""
    def __init__(self):self.assembler=Assembler();self.requests=[]
    def submit(self,request):
        self.requests.append(request);future=Future()
        try:future.set_result(self.assembler.assemble(request))
        except Exception as error:future.set_exception(error)
        return future
    def close(self):pass


class CachePipelineTests(unittest.TestCase):
    def test_raw_sampling_is_predeclared_global_bank_order(self):
        self.assertEqual([i for i in range(100)if s.retain_raw_bank(i,50)],[49,99])
        self.assertTrue(all(s.retain_raw_bank(i,1)for i in range(100)))
        for ordinal,every in ((True,50),(-1,50),(0,True),(0,0)):
            with self.assertRaises(ValueError):s.retain_raw_bank(ordinal,every)

    def compare_bytes(self,left,right):
        self.assertEqual(sorted(x.name for x in left.iterdir()),sorted(x.name for x in right.iterdir()))
        for path in left.iterdir():self.assertEqual(path.read_bytes(),(right/path.name).read_bytes(),path.name)

    def test_actual_cpu_subprocess_capped_rows_original_cache_bytes_and_cleanup(self):
        with tempfile.TemporaryDirectory()as folder:
            plan,store,pending,raw,games=DeviceSpoolTests().example(folder)
            raw=raw._replace(carry=raw.carry._replace(status=np.full(2,d.CAPPED,np.int32)))
            serial=s.Spool(store,pending['bank'],bank_sha256='a'*64);serial.append(raw)
            parallel=s.Spool(store,pending['bank'],bank_sha256='a'*64,transient_raw_root=folder);parallel.append(raw)
            temporary=Path(parallel._raw_directory.name)
            workers=p.CpuWorkers(1)
            try:
                assembly=p.BankAssembly(workers,parallel,bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
                for row in (1,0):
                    serial.finalize_game(row,games[row],bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
                    assembly.submit(row,games[row])
                actual=assembly.finish(Path(folder)/'parallel')
                self.assertGreater(actual['worker_seconds'],0);self.assertEqual(len(workers.processes),1)
                self.assertGreater(actual['assembly_span_seconds'],0)
                serial.merge(Path(folder)/'serial');self.compare_bytes(Path(folder)/'serial',Path(folder)/'parallel')
            finally:workers.close()
            parallel.release_raw()
            self.assertFalse(temporary.exists());self.assertTrue((parallel.root/'games/0/metadata.json').is_file())
            recovered=s.Spool.__new__(s.Spool);recovered.parts=list(parallel.parts);recovered.bank=parallel.bank
            recovered.merge(Path(folder)/'recovered');self.compare_bytes(Path(folder)/'serial',Path(folder)/'recovered')
            self.assertTrue(all(x.poll()is not None for x in workers.processes))

    def test_device_backend_defers_only_cache_work_and_owner_publishes_timing(self):
        from tests.test_gpu_engine_v5_device_backend import DeviceBackendTests
        from tests.test_gpu_engine_v5_device_actor_bank import DeviceActorBankTests
        from gpu_engine import v5_device_backend as backend_module
        with tempfile.TemporaryDirectory()as folder:
            plan,backend,store,snapshots,assignments,carry=DeviceBackendTests().setup_backend(folder)
            with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),\
                    patch.object(backend,'_actor',side_effect=lambda snap,store:backend.actors.setdefault(snap['archive_sha256'],({'inert':np.array(0)},{}))[0]),\
                    patch.object(d,'make_collector',return_value=None):
                prepared=backend.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
            raw,_=DeviceActorBankTests().run_inert(prepared['bank'],bindings=prepared['bindings'])
            class Collector:
                def lower(self,*args):return self
                def compile(self):return lambda *args:raw
            prepared['collector']=Collector();backend.measure_runtime=True;backend.cache_workers=2
            backend.raw_retention_every=50;backend.transient_raw_root=folder
            backend._cache_pool=LocalWorkers();backend._cache_merger=ThreadPoolExecutor(max_workers=1)
            ended=[];owner=threading.get_ident();emitter=store.emit;threads=[]
            def emit(*a,**kw):threads.append(threading.get_ident());return emitter(*a,**kw)
            try:
                with patch.object(backend,'_construct',return_value=carry),patch.object(store,'emit',side_effect=emit),\
                        patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*0,v*0+o)):
                    pending=backend.collect(prepared,store,lambda *a,**kw:ended.append(a),lambda:None)
                    self.assertIs(type(pending),runs.PendingCache)
                    self.assertFalse(store.matching('DEVICE_RUNTIME_TIMING'))
                    path,digest=pending.result()
                self.assertEqual(len(ended),2);self.assertEqual(runs._file_sha(store.path/path/'metadata.json'),digest)
                timing=store.matching('DEVICE_RUNTIME_TIMING')[0]
                self.assertEqual(timing['bank_streams'],[101,202]);self.assertEqual(timing['cache_pipeline']['workers'],2)
                self.assertEqual(timing['phases']['game_cache_wall_seconds'],timing['cache_pipeline']['assembly_elapsed_span_seconds'])
                self.assertTrue(all(x==owner for x in threads));self.assertEqual(timing['complete'],2)
                self.assertFalse(timing['cache_pipeline']['raw_storage']['durable_raw_sample'])
                self.assertEqual(timing['cache_pipeline']['raw_storage']['global_bank_ordinal'],0)
                self.assertFalse(backend._raw_spools);self.assertFalse(list(Path(folder).glob('v5-raw-*')))
            finally:backend.close()

    def test_delayed_timing_joins_original_bank_instead_of_latest_kernel_bank(self):
        from pokezero.v5_phasec_throughput import collector_banks
        def start(index,streams):return dict(kind='BATCH_STARTED',payload=dict(iteration=0,batch=index,streams=streams))
        def timing(streams):return dict(kind='DEVICE_RUNTIME_TIMING',payload=dict(bank_streams=streams))
        events=[start(0,[1,2]),start(1,[3,4]),timing([1,2]),timing([3,4])]
        self.assertEqual(collector_banks(events),{(0,0):dict(bank_streams=[1,2]),(0,1):dict(bank_streams=[3,4])})
        for bad in (events+[timing([1,2])],events+[timing([5,6])],events+[start(2,[1,2])]):
            with self.assertRaises(ValueError):collector_banks(bad)

    def test_early_completion_prefix_and_out_of_order_worker_results(self):
        with tempfile.TemporaryDirectory()as folder:
            plan,store,pending,raw,games=DeviceSpoolTests().example(folder)
            serial=s.Spool(store,pending['bank'],bank_sha256='a'*64)
            parallel=s.Spool(store,pending['bank'],bank_sha256='a'*64)
            tasks=[]
            class Queued:
                def submit(self,request):
                    future=Future();tasks.append((request,future));return future
            assembly=p.BankAssembly(Queued(),parallel,bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
            def piece(start,stop):return raw._replace(frames=d.Frames(*(None if x is None else x[start:stop]for x in raw.frames)))
            with patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*.2+o,v*.4-o)):
                for row,(start,stop)in enumerate(((0,1),(1,3))):
                    serial.append(piece(start,stop));parallel.append(piece(start,stop))
                    serial.finalize_game(row,games[row],bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
                    assembly.submit(row,games[row])
                adapter=Assembler()
                for request,future in reversed(tasks):future.set_result(adapter.assemble(request))
                actual=assembly.finish(Path(folder)/'parallel');self.assertIsNotNone(actual['metadata'])
                serial.merge(Path(folder)/'serial');self.compare_bytes(Path(folder)/'serial',Path(folder)/'parallel')
            self.assertEqual([x[0]['prefix']for x in tasks],[1,2])
            with self.assertRaisesRegex(ValueError,'twice'):assembly.submit(0,games[0])

    def test_d_teacher_labels_and_refusals_preserve_all_cache_bytes(self):
        with tempfile.TemporaryDirectory()as folder:
            plan,store,_,_,games=DeviceSpoolTests().example(folder)
            raw,bank=inert_d();recipe,labels=records(raw,bank);loss=V5LossConfig(arm='D')
            games=tuple(replace(g,stream_id=bank.streams[i])for i,g in enumerate(games))
            serial=s.Spool(store,bank,bank_sha256='a'*64,teacher_recipe=recipe)
            parallel=s.Spool(store,bank,bank_sha256='a'*64,teacher_recipe=recipe,transient_raw_root=folder)
            serial.append(raw,teacher_records=labels);parallel.append(raw,teacher_records=labels)
            assembly=p.BankAssembly(LocalWorkers(),parallel,bindings=bank.bindings,loss_config=loss,discount=plan.discount)
            with patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*.2+o,v*.4-o)):
                for row in (1,0):
                    serial.finalize_game(row,games[row],bindings=bank.bindings,loss_config=loss,discount=plan.discount)
                    assembly.submit(row,games[row])
                assembly.finish(Path(folder)/'parallel');serial.merge(Path(folder)/'serial')
            self.compare_bytes(Path(folder)/'serial',Path(folder)/'parallel')
            parallel.release_raw()
            self.assertTrue((parallel.root/'0/teacher.json').is_file())
            self.assertTrue((parallel.root/'0/metadata.json').is_file())

    def test_worker_failure_and_raw_or_metadata_mutation_never_publish_merge(self):
        for mutation in ('raw','transient_raw','metadata','worker'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as folder:
                plan,store,pending,raw,games=DeviceSpoolTests().example(folder)
                spool=s.Spool(store,pending['bank'],bank_sha256='a'*64,
                    **(dict(transient_raw_root=folder)if mutation=='transient_raw'else{}));spool.append(raw);tasks=[]
                class Queue:
                    def submit(self,request):
                        future=Future();tasks.append((request,future));return future
                assembly=p.BankAssembly(Queue(),spool,bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
                assembly.submit(0,games[0]);request,future=tasks[0]
                if mutation=='worker':future.set_exception(RuntimeError('owned worker failed'))
                else:
                    file=(store.path/spool.records[0]['frames']['numeric_features']['path']if mutation in ('raw','transient_raw')
                        else spool.root/'0'/'metadata.json')
                    file.chmod(0o644)
                    if mutation=='metadata':file.write_text('{}')
                    try:Assembler().assemble(request)
                    except Exception as error:future.set_exception(error)
                    else:self.fail('mutation accepted')
                with self.assertRaises(Exception):assembly.check()
                with self.assertRaises(Exception):assembly.finish(Path(folder)/'must-not-publish')
                self.assertFalse((Path(folder)/'must-not-publish').exists())
                spool.release_raw()

    def test_pending_bank_overlaps_next_bank_but_never_training(self):
        entered=threading.Event();release=threading.Event();events=[];executor=ThreadPoolExecutor(max_workers=1)
        class Backend(rf.FakeBackend):
            def collect(self,prepared,store,on_end,check_progress):
                result=super().collect(prepared,store,on_end,check_progress);events.append(('collect',prepared.batch.games[0].stream_id))
                if len(events)==1:
                    def assemble():entered.set();release.wait(10);return result
                    return runs.PendingCache(executor.submit(assemble),lambda x:x)
                self.assert_started=entered.wait(5);release.set();return result
            def close(self):release.set();executor.shutdown(wait=True)
        class Trainer(rf.FakeTraining):
            def train(self,runtime,paths,**kw):
                if len(events)!=2 or not release.is_set():raise AssertionError('trained before pending bank finished')
                return super().train(runtime,paths,**kw)
        recipe=rf.plan();recipe=replace(recipe,iterations=((recipe.iterations[0][0],recipe.iterations[1][0]),))
        backend=Backend()
        with tempfile.TemporaryDirectory()as folder:
            result=rf.V5RunTests().execute(Path(folder)/'run',recipe,backend,Trainer())
            self.assertEqual(result['status'],'COMPLETE_DECLARATION');self.assertTrue(backend.assert_started)
            caches=runs.Store(Path(folder)/'run',recipe,resume=True).matching('CACHE')
            self.assertEqual(sorted(x['batch']for x in caches),[0,1])

    def test_failed_pending_bank_stops_before_training_and_closes_backend(self):
        class Backend(rf.FakeBackend):
            closed=False
            def collect(self,*args):
                super().collect(*args);future=Future();future.set_exception(RuntimeError('cache worker failed'))
                return runs.PendingCache(future,lambda x:x)
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory()as folder:
            backend=Backend();trainer=rf.FakeTraining()
            result=rf.V5RunTests().execute(Path(folder)/'run',rf.plan(),backend,trainer)
            self.assertEqual(result['status'],'STOPPED');self.assertIn('CACHE',result['reason'])
            self.assertTrue(backend.closed);self.assertEqual(trainer.traces,[])

    def test_unresponsive_pending_worker_rechecks_original_stall_guard_and_stops(self):
        checks=[]
        class Backend(rf.FakeBackend):
            closed=False
            def collect(self,*args):
                super().collect(*args)
                return runs.PendingCache(Future(),lambda x:x)
            def close(self):self.closed=True
        def check(progress):
            checks.append(1)
            if len(checks)>=3:raise TimeoutError('original no-completed-game stall window')
        with tempfile.TemporaryDirectory()as folder,patch.object(runs.Progress,'check',check):
            backend=Backend();trainer=rf.FakeTraining()
            result=rf.V5RunTests().execute(Path(folder)/'run',rf.plan(),backend,trainer)
            self.assertEqual(result['status'],'STOPPED');self.assertIn('stall window',result['reason'])
            self.assertEqual(len(checks),3);self.assertTrue(backend.closed);self.assertEqual(trainer.traces,[])

    def test_finished_worker_timeout_error_is_propagated_without_retrying(self):
        future=Future();future.set_exception(TimeoutError('worker failed'))
        with self.assertRaisesRegex(TimeoutError,'worker failed'):
            runs.PendingCache(future,lambda x:self.fail('published failed worker')).result()


if __name__=='__main__':unittest.main()
