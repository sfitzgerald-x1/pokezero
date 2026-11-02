"""Fake checkpoint/optimizer/engine iterations; no actual model or game."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import v5_run as r
from pokezero.v5_model import V5ModelConfig
from pokezero.v5_loss import V5LossConfig
from pokezero.v5_training import TrainConfig
import test_v5_contract as fixtures


def plan(arm='B',replicate=0):
    model=V5ModelConfig(128,'b'*64,fixtures.vocabulary(),arm=arm,replicate=replicate)
    games=tuple(r.Game(i+1,'game'+str(i),91+i,(2*i,2*i+1)) for i in range(4))
    return r.Plan('source-test',model,V5LossConfig(arm=arm),TrainConfig(11+replicate,batch_size=2),
        ((r.Batch(games[:2],101,201),),(r.Batch(games[2:],102,202),)), 'a'*64,301,
        0. if arm=='A' else 1.,3,20,r.Watchdog(10,backstop_seconds=1000),checkpoint_updates=1)


class FakeBackend:
    def __init__(self,*,fail_prepare=False,fail_after_start=False):
        self.prepare_calls=[];self.collect_calls=[];self.assignments=[]
        self.fail_prepare=fail_prepare;self.fail_after_start=fail_after_start
    def check_plan(self,plan):pass
    def prepare_batch(self,plan,batch,current,assignments,store):
        self.prepare_calls.append(tuple(g.stream_id for g in batch.games))
        if self.fail_prepare:raise ValueError('synthetic preparation failure')
        return NS(plan=plan,batch=batch,current=current,assignments=assignments)
    def collect(self,prepared,store,on_end,check_progress):
        self.collect_calls.extend(g.stream_id for g in prepared.batch.games);self.assignments.extend(prepared.assignments)
        if self.fail_after_start:raise ValueError('synthetic owned constructor failure')
        for game in prepared.batch.games:on_end(game.stream_id,'COMPLETE',duration_seconds=.1,source_sha256='c'*64)
        path=f'cache-{prepared.batch.games[0].stream_id}';directory=store.path/path;directory.mkdir()
        (directory/'metadata.json').write_text(json.dumps({'source_fake_only':True,'streams':[g.stream_id for g in prepared.batch.games]}))
        return path,r._file_sha(directory/'metadata.json')


class FakeTraining:
    def __init__(self):self.latest=None;self.traces=[]
    def create(self,model,loss,train,**kwargs):
        self.latest=NS(model_config=model,loss_config=loss,train_config=train,steps=0,rows=0,progress=None,rng=train.seed,weight=0.)
        return self.latest
    def save(self,path,runtime):
        payload={name:getattr(runtime,name) for name in ('steps','rows','progress','rng','weight')}
        with Path(path).open('x') as stream:json.dump(payload,stream,sort_keys=True)
        return dict(sha256=r._file_sha(path))
    def load(self,path,**kwargs):
        payload=json.loads(Path(path).read_text())
        self.latest=NS(model_config=kwargs['expected_model_config'],loss_config=kwargs['expected_loss_config'],
            train_config=kwargs['expected_train_config'],**payload)
        return self.latest
    def train(self,runtime,paths,*,max_updates):
        manifest=[r._file_sha(Path(path)/'metadata.json') for path in paths]
        if runtime.progress is None:runtime.progress=dict(manifest=manifest,batch=0,complete=False)
        if runtime.progress['manifest']!=manifest:raise ValueError('fake original finite cache cursor mismatch')
        for _ in range(max_updates):
            if runtime.progress['complete']:break
            runtime.rng=(runtime.rng*1103515245+12345)%2**31;runtime.weight+=runtime.rng/2**31
            runtime.steps+=1;runtime.rows+=4;runtime.progress['batch']+=1
            runtime.progress['complete']=runtime.progress['batch']==3
            self.traces.append((runtime.steps,runtime.rng,runtime.weight))
        return dict(complete=runtime.progress['complete'])


class V5RunTests(unittest.TestCase):
    def execute(self,path,plan,backend,trainer,*,resume=False,interrupt=None):
        from contextlib import ExitStack
        from gpu_engine import v5_model
        def export(checkpoint,out,**kwargs):
            self.assertEqual(kwargs['expected_checkpoint_sha256'],r._file_sha(checkpoint))
            with Path(out).open('xb') as stream:stream.write(b'synthetic-actor:'+Path(checkpoint).read_bytes())
            return {'weights_archive_sha256':r._file_sha(out)}
        original=r.Store.checkpoint
        def checkpoint(store,runtime,*,iteration):
            result=original(store,runtime,iteration=iteration)
            if interrupt and interrupt(iteration,runtime):raise RuntimeError('synthetic interruption after durable checkpoint')
            return result
        with ExitStack() as stack:
            for name,method in (('create',trainer.create),('save_checkpoint',trainer.save),('load_checkpoint',trainer.load),('train_caches',trainer.train)):
                stack.enter_context(patch.object(r.training,name,side_effect=method))
            stack.enter_context(patch.object(v5_model,'export',side_effect=export))
            stack.enter_context(patch.object(r.Store,'checkpoint',checkpoint))
            return r.run(plan,path,backend=backend,resume=resume,enable_v5=True)

    def test_two_real_api_phases_are_ordered_scratch_history_and_independent_B_seed(self):
        for arm,replicate in (('A',0),('B',0),('B',1)):
            with self.subTest(arm=arm,replicate=replicate),tempfile.TemporaryDirectory() as directory:
                recipe=plan(arm,replicate);backend=FakeBackend();trainer=FakeTraining()
                result=self.execute(Path(directory)/'run',recipe,backend,trainer)
                self.assertEqual(result['status'],'COMPLETE_DECLARATION');self.assertEqual(result['steps'],6)
                self.assertEqual(backend.collect_calls,[1,2,3,4])
                self.assertEqual([a['opponent_kind'] for a in backend.assignments[:2]],['SELFPLAY']*2)
                self.assertEqual([a['opponent_kind'] for a in backend.assignments[2:]],
                    ['SELFPLAY' if arm=='A' else 'HISTORICAL']*2)
                if arm=='B':self.assertTrue(all(a['opponent']['iteration']==-1 for a in backend.assignments[2:]))
                store=r.Store(Path(directory)/'run',recipe,resume=True)
                self.assertEqual(len(store.matching('SNAPSHOT')),3)
                self.assertEqual(len(store.matching('DATASET_TRANSITION')),1)

    def test_resume_preserves_optimizer_rng_cursor_without_replaying_games(self):
        recipe=plan();full=FakeTraining();full_backend=FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.execute(root/'full',recipe,full_backend,full)
            resumed=FakeTraining();first=FakeBackend()
            with self.assertRaisesRegex(RuntimeError,'durable checkpoint'):
                self.execute(root/'resume',recipe,first,resumed,interrupt=lambda iteration,runtime:iteration==0 and runtime.steps==1)
            second=FakeBackend();self.execute(root/'resume',recipe,second,resumed,resume=True)
            self.assertEqual(first.collect_calls,[1,2]);self.assertEqual(second.collect_calls,[3,4])
            self.assertEqual((full.latest.rng,full.latest.weight,full.latest.steps),
                (resumed.latest.rng,resumed.latest.weight,resumed.latest.steps))
            self.assertEqual(full.traces,resumed.traces)

    def test_completed_cursor_crash_before_export_does_not_train_again(self):
        recipe=replace(plan(),iterations=plan().iterations[:1]);trainer=FakeTraining()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run';first=FakeBackend()
            with self.assertRaises(RuntimeError):self.execute(path,recipe,first,trainer,
                interrupt=lambda iteration,runtime:iteration==0 and runtime.steps==3)
            self.execute(path,recipe,FakeBackend(),trainer,resume=True)
            self.assertEqual(trainer.latest.steps,3);self.assertEqual(len(trainer.traces),3)

    def test_inflight_original_session_is_not_replayed_and_prior_cache_still_trains(self):
        original=plan();games=original.games
        recipe=replace(original,iterations=((r.Batch(games[:1],101,201),r.Batch(games[1:2],102,202)),original.iterations[1]))
        class Interrupted(FakeBackend):
            def collect(self,prepared,*args,**kwargs):
                if prepared.batch.games[0].stream_id==2:
                    self.collect_calls.append(2);raise KeyboardInterrupt('synthetic process interruption')
                return super().collect(prepared,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run';trainer=FakeTraining();first=Interrupted()
            with self.assertRaises(KeyboardInterrupt):self.execute(path,recipe,first,trainer)
            second=FakeBackend();out=self.execute(path,recipe,second,trainer,resume=True)
            self.assertEqual(first.collect_calls,[1,2]);self.assertEqual(second.collect_calls,[3,4])
            self.assertEqual([out['statuses'][i] for i in (1,2,3,4)],['COMPLETE','FAILED','COMPLETE','COMPLETE'])
            self.assertEqual(out['steps'],6)

    def test_preparation_failure_is_not_attempted_and_started_failure_is_not_retried(self):
        recipe=plan()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prepare';trainer=FakeTraining()
            out=self.execute(path,recipe,FakeBackend(fail_prepare=True),trainer)
            self.assertEqual(set(out['statuses'].values()),{'NOT_ATTEMPTED'})
            backend=FakeBackend();self.execute(path,recipe,backend,trainer,resume=True)
            self.assertEqual(backend.collect_calls,[1,2,3,4])
            path=Path(directory)/'attempt';trainer=FakeTraining()
            out=self.execute(path,recipe,FakeBackend(fail_after_start=True),trainer)
            self.assertEqual([out['statuses'][i] for i in (1,2,3,4)],['FAILED','FAILED','NOT_ATTEMPTED','NOT_ATTEMPTED'])
            backend=FakeBackend();out=self.execute(path,recipe,backend,trainer,resume=True)
            self.assertEqual(backend.collect_calls,[])  # no actual cache exists for iteration0; stop without refill
            self.assertEqual(out['reason'],'NO_ACTUAL_TRAINING_ROWS_NO_REFILL')

    def test_D_missing_teacher_refuses_before_files_or_model_and_changed_declaration_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'D'
            with self.assertRaises(r.MissingDependency):self.execute(path,plan('D'),FakeBackend(),FakeTraining())
            self.assertFalse(path.exists())
            path=Path(directory)/'B';recipe=plan();self.execute(path,recipe,FakeBackend(),FakeTraining())
            with self.assertRaises(ValueError):r.Store(path,replace(recipe,selection_seed=999),resume=True)
            store=r.Store(path,recipe,resume=True);snapshot=store.matching('SNAPSHOT')[0]
            archive=store.path/snapshot['archive'];archive.chmod(0o644);archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact'):
                self.execute(path,recipe,FakeBackend(),FakeTraining(),resume=True)

    def test_treatment_recipe_is_explicit_and_not_silently_disabled(self):
        recipe=plan()
        for name in ('entropy_coefficient','uniform_kl_coefficient','auxiliary_weight'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                replace(recipe,loss=replace(recipe.loss,**{name:0.}))
        with self.assertRaises(ValueError):replace(recipe,model=replace(recipe.model,value_sigma_ratio=.5))
        self.assertEqual(plan('A').historical_fraction,0.)

    def test_assignment_stall_and_outer_owned_process_backstop(self):
        recipe=plan();game=recipe.games[2]
        current=r.Snapshot(2,'checkpoint','a'*64,'actor','b'*64,'c'*64)
        past=tuple(r.Snapshot(i,'checkpoint','a'*64,'actor',str(i+2)*64,'c'*64) for i in range(-1,2))
        self.assertEqual(r.assignment(recipe,game,2,current,past),r.assignment(recipe,game,2,current,past))
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):current.iteration=9
        clock=[0.]
        with patch.object(r.time,'monotonic',side_effect=lambda:clock[0]):
            guard=r.Progress(recipe.watchdog);clock[0]=5;guard.completed_game(4.)
            clock[0]=24;guard.check();clock[0]=26
            with self.assertRaisesRegex(TimeoutError,'completed game'):guard.check()
        class Child:
            pid=17;returncode=None
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def wait(self,timeout=None):return self.returncode
            def kill(self):self.returncode=-9
        child=Child();clock=[0.]
        with tempfile.TemporaryDirectory() as directory,patch.object(r.time,'monotonic',side_effect=lambda:clock[0]),\
                patch.object(r,'_stop_owned_group',side_effect=lambda child,grace,**kwargs:child.terminate()):
            def sleep(seconds):clock[0]+=seconds
            out=r.supervise_owned(['owned-synthetic-worker'],plan=recipe,path=directory,popen=lambda cmd,**kwargs:child,sleep=sleep)
            self.assertEqual(out['exit_code'],-15);self.assertEqual(out['owned_pid'],17)
            self.assertEqual(out['stop_reason'],'NO_DURABLE_WORK_PROGRESS')

    def test_progress_journal_fault_is_reported_and_cannot_be_overridden_by_stall(self):
        class Child:
            pid=18;returncode=None
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def wait(self,timeout=None):return self.returncode
        with tempfile.TemporaryDirectory() as directory,\
                patch.object(r,'_stop_owned_group',side_effect=lambda child,grace,**kwargs:child.terminate()):
            path=Path(directory);(path/'journal').mkdir()
            (path/'journal'/'000000000.json').write_text(json.dumps(dict(sequence=0,previous=None,
                kind='GAME_END',payload={},time=1,sha256='wrong')))
            out=r.supervise_owned(['owned-synthetic-worker'],plan=plan(),path=path,popen=lambda cmd,**kwargs:Child(),sleep=lambda seconds:None)
            self.assertEqual(out['exit_code'],-15);self.assertEqual(out['stop_reason'],'PROGRESS_JOURNAL_CHANGED')

    def test_owned_artifact_is_create_only_namespace_bound_and_checked_on_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store=r.Store(Path(directory)/'run',plan())
            receipt=store.artifact('history/one.json',b'{"canonical":true}',kind='OWNING_HISTORY',owner=0)
            self.assertEqual(store.resolve(receipt['path'],receipt['sha256']).read_bytes(),b'{"canonical":true}')
            with self.assertRaises(FileExistsError):store.artifact('history/one.json',b'changed',kind='OWNING_HISTORY')
            with self.assertRaises(ValueError):store.artifact('../foreign.json',b'foreign',kind='OWNING_HISTORY')




class SupervisorRepairTests(unittest.TestCase):
    """Fake owned process groups only; never signal a real PID in unit tests."""
    def invoke(self,*,journal=None,finish=False,orphan=False,persistent=False):
        clock=[0.];signals=[]
        class Child:
            pid=17003;returncode=None;polls=0;group=True;waits=[];killed=False
            def poll(self):
                self.polls+=1
                if finish and self.polls>=3:
                    self.returncode=0
                    if not orphan:self.group=False
                return self.returncode
            def wait(self,timeout=None):self.waits.append(timeout);return self.returncode
            def kill(self):self.killed=True;self.returncode=-9
        child=Child();kwargs=[]
        def popen(command,**options):kwargs.append(options);return child
        def killpg(pid,sig):
            self.assertEqual(pid,child.pid)
            if not child.group:raise ProcessLookupError()
            if sig==0:return
            signals.append(sig)
            if child.returncode is None:child.returncode=-15 if sig==r.signal.SIGTERM else -9
            if sig==r.signal.SIGKILL and not persistent:child.group=False
            elif sig==r.signal.SIGTERM and not orphan and not persistent:child.group=False
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'os',NS(killpg=killpg)),\
                patch.object(r,'time',NS(monotonic=lambda:clock[0],time=lambda:2000.)):
            path=Path(directory)
            if journal is not None:
                (path/'journal').mkdir();(path/'journal'/'000000000.json').write_text(journal)
            out=r.supervise_owned(['fake-owned'],plan=plan(),path=path,popen=popen,
                sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds))
        self.assertEqual(kwargs,[dict(start_new_session=True)])
        self.assertTrue(all(x is not None for x in child.waits))
        return out,child,signals,clock[0]

    def test_new_resume_uses_its_own_backstop_not_old_run_age(self):
        event=dict(sequence=0,previous=None,time=1.,kind='RUN_STARTED',payload={});event['sha256']=r._sha(event)
        out,child,_,_=self.invoke(journal=json.dumps(event),finish=True)
        self.assertEqual(out['exit_code'],0);self.assertIsNone(out['stop_reason'])

    def test_invalid_journal_reports_error_and_reaps_owned_worker(self):
        out,child,signals,_=self.invoke(journal='{')
        self.assertEqual(out['stop_reason'],'PROGRESS_JOURNAL_CHANGED')
        self.assertIn('JSONDecodeError',out['observer_error']);self.assertFalse(child.group)
        self.assertIn(r.signal.SIGTERM,signals)

    def test_leader_exit_does_not_prove_its_ignoring_descendant_is_gone(self):
        out,child,signals,_=self.invoke(finish=True,orphan=True)
        self.assertEqual(out['exit_code'],0);self.assertFalse(child.group)
        self.assertIn(r.signal.SIGTERM,signals);self.assertIn(r.signal.SIGKILL,signals)

    def test_still_present_group_refuses_after_finite_cleanup(self):
        with self.assertRaisesRegex(RuntimeError,'process group remains'):
            self.invoke(finish=True,orphan=True,persistent=True)

if __name__=='__main__':unittest.main()
