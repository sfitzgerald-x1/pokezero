"""Frozen CPU dispatch with inert trainer/panel controls; no real games/models."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_run as run,v5_evaluation as evaluation,v5_deferred_evaluation as deferred
import test_v5_run as fixture
import test_v5_milestone_evaluation as milestone
import test_v5_evaluation as panel_fixture


class DeferredEvaluationTests(unittest.TestCase):
    def recipe(self,root):
        recipe=fixture.plan();path=Path(root)/'protocol.json'
        path.write_text(run._json({'model_sha256':run._sha(recipe.model.metadata())}))
        reference=evaluation.Artifact(str(path),run._file_sha(path))
        return replace(recipe,evaluation=run.EvaluationProtocol(reference.sha256)),reference

    def execute(self,path,recipe,hook,*,resume=False):
        return milestone.MilestoneEvaluationTests().execute(path,recipe,fixture.FakeBackend(),fixture.FakeTraining(),hook,resume=resume)

    def test_result_reader_recomputes_known_scores_and_keeps_all1000_unknown_denominators(self):
        with tempfile.TemporaryDirectory()as folder:
            protocol_ref,protocol,_,request=panel_fixture.EvaluationTests().fixture(Path(folder))
            dispatch=deferred.publish_dispatch(request,protocol_ref,request.declaration_sha256)
            ref=evaluation.Artifact(str(request.run_directory/dispatch.request),dispatch.request_sha256)
            _,panels=evaluation._protocol(protocol_ref,parse_configs=True,checkpoint=request.checkpoint_path)
            configs=panels[request.declared_games];rows=evaluation._fixed_rows(configs)
            for row in rows:row.update(status='FAILED',reason='inert unknown, never played')
            game=panel_fixture.game(configs[0],rows[0]['seed'])
            rows[0].update(status='COMPLETE',score=1.,duration_seconds=2.,source_result_sha256=run._sha(game),source_game=game)
            directory=Path(ref.path).parent;report=request.run_directory/'inert-report.json'
            document=dict(status='RESULT',request=evaluation._request_document(request),rows=rows,
                known_completed=1,unknown_cells=999,completed_only_score=1.,score_bounds=[.001,1.])
            deferred._publish_equal(report,document)
            outcome=run.EvaluationOutcome('RESULT',str(report.relative_to(request.run_directory)),run._file_sha(report))
            started=dict(schema=deferred.ATTEMPT,dispatch_sha256=ref.sha256,attempt=0)
            deferred._publish_equal(directory/'attempts/000000.json',started)
            target=directory/'outcomes/000000.json'
            deferred._publish_equal(target,dict(started,outcome=outcome.__dict__))
            actual=deferred.read_dispatched_result(ref)
            self.assertEqual((actual['known_completed'],actual['unknown_cells'],actual['score_bounds']),(1,999,[.001,1.]))
            for change in ('score','mean','pending'):
                broken=json.loads(run._json(document))
                if change=='score':broken['rows'][0]['score']=0.
                elif change=='mean':broken['completed_only_score']=0.
                else:broken['rows'][-1]['status']='NOT_ATTEMPTED'
                report.chmod(0o644);report.write_text(run._json(broken))
                forged=dict(started,outcome=dict(outcome.__dict__,report_sha256=run._file_sha(report)))
                target.chmod(0o644);target.write_text(run._json(forged))
                with self.subTest(change=change),self.assertRaises(ValueError):deferred.read_dispatched_result(ref)

    def test_dispatch_continues_both_iterations_without_creating_any_result_or_process(self):
        with patch.object(run,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory()as folder:
            recipe,protocol=self.recipe(folder);path=Path(folder)/'run';requests=[]
            def hook(request):
                requests.append(request)
                return deferred.publish_dispatch(request,protocol,request.declaration_sha256)
            with patch.object(evaluation,'EvaluationHook',side_effect=AssertionError('CPU panel launched on GPU')):
                result=self.execute(path,recipe,hook)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION');self.assertEqual(result['steps'],6)
            store=run.Store(path,recipe,resume=True)
            self.assertEqual(len(store.matching('ITERATION_COMPLETE')),2)
            self.assertEqual(len(store.matching('EVALUATION_DISPATCHED')),2)
            self.assertFalse(store.matching('EVALUATION_RESULT'));self.assertFalse(store.matching('EVALUATION_UNKNOWN'))
            self.assertEqual(run._evaluation_history(store),{})
            for request,payload in zip(requests,store.matching('EVALUATION_DISPATCHED')):
                dispatch=run.EvaluationDispatch(**payload['dispatch'])
                ref=evaluation.Artifact(str(store.resolve(dispatch.request,dispatch.request_sha256)),dispatch.request_sha256)
                self.assertEqual(deferred.read_dispatched_result(ref)['status'],'PENDING')
                self.assertEqual(json.loads(Path(ref.path).read_text())['request'],evaluation._request_document(request))
            self.execute(path,recipe,lambda _:self.fail('dispatch replayed on resume'),resume=True)

    def test_crash_after_dispatch_never_requeues_the_original_panel(self):
        with patch.object(run,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory()as folder:
            recipe,protocol=self.recipe(folder);path=Path(folder)/'run';original=run.Store.emit
            def hook(request):return deferred.publish_dispatch(request,protocol,request.declaration_sha256)
            def emit(store,kind,**payload):
                out=original(store,kind,**payload)
                if kind=='EVALUATION_DISPATCHED':raise KeyboardInterrupt('inert interruption')
                return out
            with patch.object(run.Store,'emit',emit),self.assertRaises(KeyboardInterrupt):self.execute(path,recipe,hook)
            called=[]
            def resumed(request):called.append(request.declared_games);return hook(request)
            result=self.execute(path,recipe,resumed,resume=True)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION');self.assertEqual(called,[4])
            store=run.Store(path,recipe,resume=True)
            self.assertEqual(len(store.matching('EVALUATION_DISPATCHED')),2)
            self.assertFalse(store.matching('EVALUATION_RESUMED'))

    def test_mutated_dispatch_refuses_before_resume_trainer_and_foreign_requests_stop(self):
        with patch.object(run,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory()as folder:
            recipe,protocol=self.recipe(folder);path=Path(folder)/'run'
            self.execute(path,recipe,lambda q:deferred.publish_dispatch(q,protocol,q.declaration_sha256))
            store=run.Store(path,recipe,resume=True);item=store.matching('EVALUATION_DISPATCHED')[0]['dispatch']
            target=store.resolve(item['request'],item['request_sha256']);target.chmod(0o644)
            value=json.loads(target.read_text());value['request']['declared_games']+=1;target.write_text(run._json(value))
            with patch.object(run.training,'create')as create,patch.object(run.training,'load_checkpoint')as load:
                with self.assertRaises(ValueError):self.execute(path,recipe,lambda _:None,resume=True)
                create.assert_not_called();load.assert_not_called()
        with patch.object(run,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory()as folder:
            recipe,_=self.recipe(folder);path=Path(folder)/'run'
            result=self.execute(path,recipe,lambda _:run.EvaluationDispatch('../foreign.json','f'*64))
            self.assertEqual(result['status'],'STOPPED')
            self.assertFalse(run.Store(path,recipe,resume=True).matching('EVALUATION_DISPATCHED'))

    def test_queue_publication_is_idempotent_and_cpu_controller_never_writes_training_journal(self):
        with patch.object(run,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory()as folder:
            recipe,protocol=self.recipe(folder);path=Path(folder)/'run';requests=[]
            def hook(request):
                requests.append(request)
                first=deferred.publish_dispatch(request,protocol,request.declaration_sha256)
                self.assertEqual(first,deferred.publish_dispatch(request,protocol,request.declaration_sha256))
                return first
            self.execute(path,recipe,hook);store=run.Store(path,recipe,resume=True)
            item=store.matching('EVALUATION_DISPATCHED')[0]['dispatch']
            ref=evaluation.Artifact(str(store.resolve(item['request'],item['request_sha256'])),item['request_sha256'])
            journal={p.name:p.read_bytes()for p in (path/'journal').glob('*.json')}
            calls=[]
            class OriginalHook:
                def __init__(self,*args):pass
                def __call__(self,request):calls.append(request);return run.EvaluationOutcome('UNKNOWN',reason='inert lost worker')
            with patch.dict('os.environ',{'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':''}),patch.object(evaluation,'EvaluationHook',OriginalHook):
                value=deferred.execute_dispatch(ref,enable_cpu_panel=True);self.assertEqual(value.status,'UNKNOWN')
                with self.assertRaisesRegex(ValueError,'explicit original CPU panel resume'):
                    deferred.execute_dispatch(ref,enable_cpu_panel=True)
                deferred.execute_dispatch(ref,resume=True,enable_cpu_panel=True)
            self.assertEqual(len(calls),2)
            self.assertEqual(journal,{p.name:p.read_bytes()for p in (path/'journal').glob('*.json')})
            self.assertEqual(deferred.read_dispatched_result(ref)['status'],'UNKNOWN')
            with self.assertRaisesRegex(ValueError,'explicit separate CPU panel'):
                deferred.execute_dispatch(ref)


def dispatch(root):
    reference,protocol,_,request=panel_fixture.EvaluationTests().fixture(root)
    value=dict(schema=deferred.SCHEMA,protocol=asdict(reference),expected_run_declaration_sha256=request.declaration_sha256,
        request=evaluation._request_document(request),prior_completions=[])
    directory=request.run_directory/'artifacts/cpu-panels'/str(request.declared_games)
    ref=deferred._publish_equal(directory/'request.json',value)
    return ref,reference,protocol,request,directory

class IndependentDeferredPanels(unittest.TestCase):
    def test_latest_started_attempt_without_outcome_is_pending_not_prior_unknown(self):
        with tempfile.TemporaryDirectory()as temporary:
            ref,_,_,_,directory=dispatch(Path(temporary))
            first=dict(schema=deferred.ATTEMPT,dispatch_sha256=ref.sha256,attempt=0)
            deferred._publish_equal(directory/'attempts/000000.json',first)
            deferred._publish_equal(directory/'outcomes/000000.json',dict(first,outcome=asdict(run.EvaluationOutcome('UNKNOWN',reason='old stopped attempt'))))
            deferred._publish_equal(directory/'attempts/000001.json',dict(first,attempt=1))
            actual=deferred.read_dispatched_result(ref)
            self.assertEqual(actual['status'],'PENDING')

    def test_final_report_cannot_erase_durable_cpu_completion(self):
        with tempfile.TemporaryDirectory()as temporary:
            ref,protocol_ref,protocol,request,directory=dispatch(Path(temporary))
            configs=evaluation._pairs(protocol,parse_configs=True,checkpoint=request.checkpoint_path)[request.declared_games]
            rows=evaluation._fixed_rows(configs)
            for row in rows:row.update(status='FAILED',reason='inert unknown')
            game=panel_fixture.game(configs[0],rows[0]['seed'])
            rows[0].update(status='COMPLETE',score=1.,duration_seconds=2.,source_game=game,source_result_sha256=run._sha(game))
            receipt=evaluation._completed_receipts(rows,evaluation._fixed_rows(configs),configs,protocol_path=protocol_ref.path)[0]
            deferred._publish_equal(directory/'completions/0.json',asdict(receipt))
            report=request.run_directory/'inert-panel-report.json'
            value=dict(status='RESULT',request=evaluation._request_document(request),rows=rows,
                known_completed=1,unknown_cells=999,completed_only_score=1.,score_bounds=[.001,1.])
            deferred._publish_equal(report,value)
            first=dict(schema=deferred.ATTEMPT,dispatch_sha256=ref.sha256,attempt=0)
            deferred._publish_equal(directory/'attempts/000000.json',first)
            outcome=run.EvaluationOutcome('RESULT',str(report.relative_to(request.run_directory)),run._file_sha(report))
            target=directory/'outcomes/000000.json';deferred._publish_equal(target,dict(first,outcome=asdict(outcome)))
            self.assertEqual(deferred.read_dispatched_result(ref)['known_completed'],1)
            # Coherent report+summary+outcome digest edit cannot erase an
            # already persisted original completion at the CPU boundary.
            value['rows'][0].update(status='FAILED',score=None,source_game=None,source_result_sha256=None,duration_seconds=None)
            value.update(known_completed=0,unknown_cells=1000,completed_only_score=None,score_bounds=[0.,1.])
            report.chmod(0o644);report.write_text(run._json(value));outcome=run.EvaluationOutcome('RESULT',outcome.report,run._file_sha(report))
            target.chmod(0o644);target.write_text(run._json(dict(first,outcome=asdict(outcome))))
            with self.assertRaises(ValueError):deferred.read_dispatched_result(ref)


if __name__=='__main__':unittest.main()
