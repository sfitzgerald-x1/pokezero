"""Finite inert pipelines: no model forward, game, dataset or allocation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from pokezero import v5_run as r
from test_v5_run import FakeBackend,FakeTraining,plan


def execute(path,recipe,backend,trainer,*,resume=False,lag=1,interrupt=None):
    from gpu_engine import v5_model
    def export(checkpoint,out,**kwargs):
        assert kwargs['expected_checkpoint_sha256']==r._file_sha(checkpoint)
        Path(out).write_bytes(b'synthetic-actor:'+Path(checkpoint).read_bytes())
        return dict(weights_archive_sha256=r._file_sha(out))
    original=r.Store.checkpoint
    def checkpoint(store,runtime,*,iteration):
        result=original(store,runtime,iteration=iteration)
        if interrupt and interrupt(iteration,runtime):raise RuntimeError('inert durable interruption')
        return result
    with ExitStack()as stack:
        for name,method in (('create',trainer.create),('save_checkpoint',trainer.save),('load_checkpoint',trainer.load),('train_caches',trainer.train)):
            stack.enter_context(patch.object(r.training,name,side_effect=method))
        stack.enter_context(patch.object(v5_model,'export',side_effect=export))
        stack.enter_context(patch.object(r.Store,'checkpoint',checkpoint))
        return r.run(recipe,path,backend=backend,resume=resume,enable_v5=True,iteration_policy_lag=lag,measure_timings=True)


class Backend(FakeBackend):
    def __init__(self,**kwargs):super().__init__(**kwargs);self.acting=[];self.closed=False
    def require_teacher(self):pass
    def prepare_batch(self,recipe,batch,current,assignments,store):
        self.acting.append(current.iteration)
        return super().prepare_batch(recipe,batch,current,assignments,store)
    def close(self):self.closed=True


class IterationLagTests(unittest.TestCase):
    def test_each_arm_retains_fixed_order_budgets_and_lags_exactly_one(self):
        for arm,replicate in (('A',0),('B',0),('B',1),('D',0)):
            with self.subTest(arm=arm,replicate=replicate),tempfile.TemporaryDirectory()as directory:
                base=plan(arm,replicate)
                extra=tuple(r.Game(i+1,'game'+str(i),91+i,(2*i,2*i+1))for i in range(4,6))
                recipe=replace(base,iterations=base.iterations+((r.Batch(extra,103,203),),))
                backend=Backend();trainer=FakeTraining();path=Path(directory)/'run'
                result=execute(path,recipe,backend,trainer)
                self.assertEqual(result['status'],'COMPLETE_DECLARATION')
                self.assertEqual((result['steps'],result['rows']),(9,36))
                self.assertEqual(backend.collect_calls,[1,2,3,4,5,6])
                self.assertEqual(backend.acting,[-1,-1,0]);self.assertTrue(backend.closed)
                store=r.Store(path,recipe,resume=True)
                self.assertEqual(len(store.matching('ITERATION_COMPLETE')),3)
                self.assertEqual(len(store.matching('POLICY_LAG_WINDOW_COMPLETE')),2)
                self.assertEqual(len(store.matching('COLLECTION_ITERATION_COMPLETE')),3)
                self.assertEqual([b['current']['iteration']for b in store.matching('ITERATION_ACTOR_BINDING')],[-1,-1,0])
                for event in store.matching('POLICY_LAG_WINDOW_COMPLETE'):
                    self.assertGreaterEqual(event['wall_seconds'],0)
                    self.assertFalse(event['includes_interruption'])

    def test_next_collection_and_prior_training_actually_overlap(self):
        collecting=threading.Event();training=threading.Event()
        class ConcurrentBackend(Backend):
            def collect(self,prepared,*args,**kwargs):
                if prepared.batch.games[0].stream_id==3:
                    collecting.set()
                    if not training.wait(5):raise RuntimeError('training was not concurrent')
                return super().collect(prepared,*args,**kwargs)
        class ConcurrentTraining(FakeTraining):
            def train(self,*args,**kwargs):
                if not collecting.wait(5):raise RuntimeError('collection was not concurrent')
                training.set();return super().train(*args,**kwargs)
        with tempfile.TemporaryDirectory()as directory:
            result=execute(Path(directory)/'run',plan(),ConcurrentBackend(),ConcurrentTraining())
            self.assertEqual(result['status'],'COMPLETE_DECLARATION')
            self.assertTrue(collecting.is_set()and training.is_set())

    def test_checkpoint_resume_preserves_optimizer_rng_and_never_replays_prefetch(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);recipe=plan();full=FakeTraining();resumed=FakeTraining()
            execute(root/'full',recipe,Backend(),full)
            first=Backend()
            with self.assertRaisesRegex(RuntimeError,'durable interruption'):
                execute(root/'resume',recipe,first,resumed,interrupt=lambda i,t:i==0 and t.steps==1)
            second=Backend();result=execute(root/'resume',recipe,second,resumed,resume=True)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION')
            self.assertEqual(first.collect_calls+second.collect_calls,[1,2,3,4])
            self.assertEqual(full.traces,resumed.traces)
            self.assertEqual((full.latest.rng,full.latest.weight,full.latest.steps),
                (resumed.latest.rng,resumed.latest.weight,resumed.latest.steps))

    def test_training_failure_cancels_background_before_backend_close(self):
        collecting=threading.Event();stopped=threading.Event()
        class Cancellable(Backend):
            def collect(self,prepared,store,on_end,check_progress):
                if prepared.batch.games[0].stream_id==3:
                    collecting.set()
                    try:
                        while True:check_progress();stopped.wait(.001)
                    finally:stopped.set()
                return super().collect(prepared,store,on_end,check_progress)
            def close(self):
                assert stopped.is_set();super().close()
        class Failed(FakeTraining):
            def train(self,*args,**kwargs):
                assert collecting.wait(5)
                raise RuntimeError('original trainer failed')
        with tempfile.TemporaryDirectory()as directory:
            backend=Cancellable();path=Path(directory)/'run'
            with self.assertRaisesRegex(RuntimeError,'original trainer failed'):
                execute(path,plan(),backend,Failed())
            self.assertTrue(stopped.is_set()and backend.closed)
            self.assertFalse(r.Store(path,plan(),resume=True).matching('SNAPSHOT',iteration=0))

    def test_interrupted_prefetch_keeps_lost_cell_unknown_and_resumes_only_untouched(self):
        base=plan();second=base.iterations[1][0]
        recipe=replace(base,iterations=(base.iterations[0],tuple(r.Batch((g,),102+i,202+i)for i,g in enumerate(second.games))))
        class Interrupted(Backend):
            def collect(self,prepared,*args,**kwargs):
                if prepared.batch.games[0].stream_id==3:
                    self.collect_calls.append(3);raise KeyboardInterrupt('inert lost session')
                return super().collect(prepared,*args,**kwargs)
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'run';first=Interrupted();trainer=FakeTraining()
            with self.assertRaises(KeyboardInterrupt):execute(path,recipe,first,trainer)
            next_backend=Backend();result=execute(path,recipe,next_backend,trainer,resume=True)
            self.assertEqual(first.collect_calls,[1,2,3]);self.assertEqual(next_backend.collect_calls,[4])
            self.assertEqual(result['statuses'],{1:'COMPLETE',2:'COMPLETE',3:'FAILED',4:'COMPLETE'})
            self.assertEqual(result['status'],'COMPLETE_DECLARATION')

    def test_concurrent_journal_is_one_original_valid_hash_chain(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'run';recipe=plan();store=r.Store(path,recipe)
            with ThreadPoolExecutor(max_workers=4)as executor:
                list(executor.map(lambda i:store.emit('INERT_CONCURRENT',cell=i),range(40)))
            checked=r.Store(path,recipe,resume=True)
            self.assertEqual(sorted(x['cell']for x in checked.matching('INERT_CONCURRENT')),list(range(40)))

    def test_lag_mode_cannot_silently_change_on_resume(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'run';recipe=plan();execute(path,recipe,Backend(),FakeTraining())
            with self.assertRaisesRegex(ValueError,'policy cannot change'):
                execute(path,recipe,Backend(),FakeTraining(),resume=True,lag=0)


if __name__=='__main__':unittest.main()
