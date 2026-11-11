"""Fixed milestone dispatch with fake trainer/exporter/collector; no real panel."""
from contextlib import ExitStack
from dataclasses import FrozenInstanceError,asdict,replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_run as r
import test_v5_run as fixture


class MilestoneEvaluationTests(unittest.TestCase):
    def recipe(self):
        return replace(fixture.plan(),evaluation=r.EvaluationProtocol('e'*64))

    def execute(self,path,recipe,backend,trainer,hook,*,resume=False):
        from gpu_engine import v5_model
        def export(checkpoint,out,**kwargs):
            self.assertEqual(kwargs['expected_checkpoint_sha256'],r._file_sha(checkpoint))
            Path(out).write_bytes(b'fake-archive:'+Path(checkpoint).read_bytes())
            return dict(weights_archive_sha256=r._file_sha(out))
        with ExitStack() as stack:
            for name,method in [('create',trainer.create),('save_checkpoint',trainer.save),
                    ('load_checkpoint',trainer.load),('train_caches',trainer.train)]:
                stack.enter_context(patch.object(r.training,name,side_effect=method))
            stack.enter_context(patch.object(v5_model,'export',side_effect=export))
            return r.run(recipe,path,backend=backend,resume=resume,enable_v5=True,evaluation_hook=hook)

    def result(self,request):
        report='panel-'+str(request.declared_games)+'.json'
        r._publish(request.run_directory/report,b'{"fake_only":true,"qualification":false}')
        return r.EvaluationOutcome('RESULT',report,r._file_sha(request.run_directory/report))

    def test_every_500k_boundary_requires_exact_declared_iteration_end(self):
        self.assertEqual(r.evaluation_milestones([1250]*1600),{399:500000,799:1000000,1199:1500000,1599:2000000})
        self.assertEqual(r.evaluation_milestones([200000,300000,499999,1,500000,500000]),
            {1:500000,3:1000000,4:1500000,5:2000000})
        for counts in ([500001],[250000,250001],[500000,500001],[1000000],[0],[True]):
            with self.subTest(counts=counts),self.assertRaises(ValueError):r.evaluation_milestones(counts)
        self.assertEqual(r.evaluation_milestones([499999]),{})

    def test_default_manifest_and_absent_hook_leave_legacy_calls_unchanged(self):
        recipe=fixture.plan()
        self.assertNotIn('evaluation',recipe.manifest())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run';trainer=fixture.FakeTraining()
            result=self.execute(path,recipe,fixture.FakeBackend(),trainer,None)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION')
            self.assertFalse(any(e['kind'].startswith('EVALUATION_') for e in r.Store(path,recipe,resume=True).events))

    def test_missing_or_undeclared_hook_and_crossing_refuse_before_allocation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r.training,'create') as create:
            backend=fixture.FakeBackend();backend.check_plan=lambda p:self.fail('backend reached before preflight')
            with self.assertRaisesRegex(ValueError,'hook'):
                r.run(self.recipe(),Path(temp)/'missing',backend=backend,enable_v5=True)
            with self.assertRaisesRegex(ValueError,'declaration'):
                r.run(fixture.plan(),Path(temp)/'undeclared',backend=backend,enable_v5=True,evaluation_hook=self.result)
            with patch.object(r,'EVALUATION_INTERVAL_GAMES',3):
                #4 declared games end at2/4, so3 cannot be evaluated on time.
                with self.assertRaisesRegex(ValueError,'crosses'):self.recipe()
                recipe=replace(fixture.plan(),evaluation=None)
                object.__setattr__(recipe,'evaluation',r.EvaluationProtocol('e'*64))
                with self.assertRaisesRegex(ValueError,'crosses'):
                    r.run(recipe,Path(temp)/'crossing',backend=backend,enable_v5=True,evaluation_hook=self.result)
            create.assert_not_called();self.assertEqual(list(Path(temp).iterdir()),[])

    def test_exact_snapshot_and_start_are_published_before_hook_and_result_is_bound(self):
        with patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory() as temp:
            recipe=self.recipe();path=Path(temp)/'run';trainer=fixture.FakeTraining();seen=[]
            def hook(request):
                self.assertIs(type(request),r.EvaluationRequest);self.assertIs(type(request.snapshot),r.Snapshot)
                store=r.Store(path,recipe,resume=True)
                self.assertEqual(store.events[-1]['kind'],'EVALUATION_STARTED')
                original=store.matching('SNAPSHOT',iteration=request.snapshot.iteration)[0]
                self.assertEqual(asdict(request.snapshot),original)
                self.assertEqual(request.checkpoint_path,store.resolve(original['checkpoint'],original['checkpoint_sha256']))
                self.assertEqual(request.archive_path,store.resolve(original['archive'],original['archive_sha256']))
                self.assertEqual(request.declaration_sha256,store.matching('RUN_STARTED')[0]['declaration_sha256'])
                self.assertEqual((request.protocol_sha256,request.declared_games),('e'*64,2*(request.snapshot.iteration+1)))
                self.assertTrue(json.loads(request.checkpoint_path.read_text())['progress']['complete'])
                with self.assertRaises(FrozenInstanceError):request.snapshot.iteration=99
                seen.append(request);return self.result(request)
            result=self.execute(path,recipe,fixture.FakeBackend(),trainer,hook)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION');self.assertFalse(result['scientific_qualification'])
            self.assertEqual([x.declared_games for x in seen],[2,4])
            store=r.Store(path,recipe,resume=True)
            for start,end in zip(store.matching('EVALUATION_STARTED'),store.matching('EVALUATION_RESULT')):
                self.assertEqual(start,{key:value for key,value in end.items() if key!='outcome'})
                store.resolve(end['outcome']['report'],end['outcome']['report_sha256'])
            self.execute(path,recipe,fixture.FakeBackend(),fixture.FakeTraining(),
                lambda request:self.fail('completed panel replayed'),resume=True)

    def test_exception_stops_invocation_then_resume_reenters_original_panel_before_next_iteration(self):
        with patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory() as temp:
            recipe=self.recipe();path=Path(temp)/'run';trainer=fixture.FakeTraining();backend=fixture.FakeBackend()
            def fail(request):raise RuntimeError('fake-panel-failure')
            stopped=self.execute(path,recipe,backend,trainer,fail)
            self.assertEqual((stopped['status'],trainer.latest.steps,backend.collect_calls),('STOPPED',3,[1,2]))
            store=r.Store(path,recipe,resume=True);self.assertEqual(len(store.matching('EVALUATION_UNKNOWN')),1)
            self.assertIn('fake-panel-failure',store.matching('EVALUATION_UNKNOWN')[0]['outcome']['reason'])
            self.assertEqual(len(store.matching('SNAPSHOT',iteration=0)),1)
            resumed=[];second=fixture.FakeBackend()
            def hook(request):resumed.append(request.declared_games);return self.result(request)
            result=self.execute(path,recipe,second,trainer,hook,resume=True)
            self.assertEqual((result['status'],resumed,second.collect_calls),('COMPLETE_DECLARATION',[2,4],[3,4]))
            self.assertEqual(len(r.Store(path,recipe,resume=True).matching('EVALUATION_STARTED')),2)
            self.assertEqual(len(r.Store(path,recipe,resume=True).matching('EVALUATION_RESUMED')),1)

    def test_interrupted_start_and_result_before_iteration_commit_never_replay(self):
        for after_result in (False,True):
            with self.subTest(after_result=after_result),patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory() as temp:
                recipe=self.recipe();path=Path(temp)/'run';trainer=fixture.FakeTraining();original=r.Store.emit
                def emit(store,kind,**payload):
                    out=original(store,kind,**payload)
                    if kind==('EVALUATION_RESULT' if after_result else 'EVALUATION_STARTED'):
                        raise KeyboardInterrupt('fake-process-loss')
                    return out
                with patch.object(r.Store,'emit',emit),self.assertRaises(KeyboardInterrupt):
                    self.execute(path,recipe,fixture.FakeBackend(),trainer,self.result)
                requests=[]
                def hook(request):requests.append(request.declared_games);return self.result(request)
                result=self.execute(path,recipe,fixture.FakeBackend(),trainer,hook,resume=True)
                self.assertEqual((result['status'],requests,trainer.latest.steps),('COMPLETE_DECLARATION',[4]if after_result else[2,4],6))
                store=r.Store(path,recipe,resume=True)
                self.assertEqual(len(store.matching('EVALUATION_UNKNOWN')),0)
                self.assertEqual(len(store.matching('EVALUATION_RESUMED')),0 if after_result else 1)

    def test_explicit_unknown_bad_outcomes_and_mutated_inputs_never_grant_result(self):
        for mode in ('unknown','wrongtype','foreign-report','mutated-checkpoint'):
            with self.subTest(mode=mode),patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory() as temp:
                recipe=self.recipe();path=Path(temp)/'run'
                def hook(request):
                    if mode=='unknown':return r.EvaluationOutcome('UNKNOWN',reason='fake-incomplete-panel')
                    if mode=='wrongtype':return {'status':'RESULT'}
                    if mode=='foreign-report':return r.EvaluationOutcome('RESULT','../foreign.json','f'*64)
                    request.checkpoint_path.chmod(0o644);request.checkpoint_path.write_text('changed')
                    return self.result(request)
                result=self.execute(path,recipe,fixture.FakeBackend(),fixture.FakeTraining(),hook)
                store=r.Store(path,recipe,resume=True)
                self.assertEqual(result['status'],'STOPPED')
                self.assertEqual(len(store.matching('EVALUATION_RESULT')),0)
                self.assertEqual(len(store.matching('EVALUATION_UNKNOWN')),1)

    def test_journal_protocol_snapshot_report_and_optin_drift_refuse_before_trainer(self):
        with patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory() as temp:
            root=Path(temp);recipe=self.recipe()
            for mutation in ('snapshot','protocol','report','missing-start','missing-evaluation'):
                with self.subTest(mutation=mutation):
                    path=root/mutation;self.execute(path,recipe,fixture.FakeBackend(),fixture.FakeTraining(),self.result)
                    store=r.Store(path,recipe,resume=True)
                    if mutation=='report':
                        out=store.matching('EVALUATION_RESULT')[0]['outcome'];p=path/out['report'];p.chmod(0o644);p.write_bytes(b'changed')
                    else:
                        events=store.events
                        if mutation in ('snapshot','protocol'):
                            payload=next(e['payload'] for e in events if e['kind']=='EVALUATION_STARTED')
                            if mutation=='snapshot':payload['snapshot']['archive_sha256']='f'*64
                            else:payload['protocol_sha256']='f'*64
                        elif mutation=='missing-start':events=[e for e in events if e['kind']!='EVALUATION_STARTED']
                        else:events=[e for e in events if not e['kind'].startswith('EVALUATION_')]
                        previous=None
                        for index,event in enumerate(events):
                            event=dict(event);event.pop('sha256');event.update(sequence=index,previous=previous)
                            event['sha256']=r._sha(event);previous=event['sha256']
                            p=path/'journal'/f'{index:09d}.json';p.chmod(0o644);p.write_text(r._json(event))
                        #Fake-only mutation of this temporary owned journal.
                        for p in sorted((path/'journal').glob('*.json'))[len(events):]:p.unlink()
                    with patch.object(r.training,'create') as create,patch.object(r.training,'load_checkpoint') as load:
                        with self.assertRaises(ValueError):r.run(recipe,path,backend=fixture.FakeBackend(),resume=True,
                            enable_v5=True,evaluation_hook=self.result)
                        create.assert_not_called();load.assert_not_called()
            path=root/'optin';self.execute(path,recipe,fixture.FakeBackend(),fixture.FakeTraining(),self.result)
            with self.assertRaisesRegex(ValueError,'declaration'):
                r.Store(path,replace(recipe,evaluation=r.EvaluationProtocol('f'*64)),resume=True)
            with self.assertRaisesRegex(ValueError,'declaration'):r.Store(path,replace(recipe,evaluation=None),resume=True)


if __name__=='__main__':unittest.main()
