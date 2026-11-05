"""Completed-report repair retains clocks, partial capture and missing counters."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_phasec_report as report, v5_phasec_throughput as ct


class SavedThroughputReportTests(unittest.TestCase):
    def evidence(self,root):
        snapshots=[]
        for i in range(2):
            value=dict(iteration=i)
            for name in ('checkpoint','archive'):
                path=root/f'{name}-{i}.bin';path.write_bytes(f'original-{name}-{i}'.encode())
                value.update({name:path.name,name+'_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            snapshots.append(dict(iteration=i,snapshot=value))
        smoke=dict(status='COMPLETE_DECLARATION',declared_games=40000,iteration_policy_lag=1,pipeline_iterations=2,
            model_arm='B',jax_reload='VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED',steps=10,rows=20,
            original_run_wall_seconds=3000.,collection_runtime={},original_phase_wall_seconds={'TRAIN_CACHES_INCLUSIVE':100.},
            pipeline_windows=[dict(training_iteration=0,collection_iteration=1,wall_seconds=2000.,includes_interruption=False)],
            complete_iterations=snapshots,collection_iterations=[dict(iteration=i)for i in (0,1)],
            acting_snapshots=[dict(current=dict(iteration=-1))for _ in (0,1)],
            cohort_training_budgets=[dict(iteration=i,start_step=i*5,end_step=(i+1)*5,row_visits=10)for i in (0,1)],
            dispositions=dict(COMPLETE=39999,CAPPED=1,FAILED=0,REFUSED=0,RUNNING=0,NOT_ATTEMPTED=0),
            checkpoint_sha256=snapshots[-1]['snapshot']['checkpoint_sha256'],archive_sha256=snapshots[-1]['snapshot']['archive_sha256'])
        declaration=dict(model=dict(arm='B'),iterations=[[dict(games=[dict(stream_id=i)for i in range(20000)])],
            [dict(games=[dict(stream_id=i)for i in range(20000,40000)])]])
        events=[dict(kind='GAME_END',payload=dict(stream_id=i,status='COMPLETE'if i else'CAPPED'))for i in range(40000)]
        events.append(dict(kind='SMOKE_END_TO_END',sequence=len(events),sha256='saved-smoke',payload=smoke))
        scope=dict(ph='X',cat='user_annotation',name='pokezero.learner_update',pid=1,tid=10,ts=0,dur=50)
        trace=dict(traceEvents=[scope,dict(scope,cat='gpu_user_annotation',pid=0),
            dict(ph='X',cat='cuda_runtime',pid=1,tid=10,ts=1,dur=1,args={'correlation':9}),
            dict(ph='X',cat='kernel',dur=20,args={'correlation':9}),
            dict(ph='X',cat='kernel',dur=999,args={'correlation':90})])
        (root/'ct-training-trace.json').write_text(json.dumps(trace))
        owner=root/'owner-result.json';owner.write_text(json.dumps(dict(exit_code=1,stop_reason=None,observer_error=None)))
        banks={(i,j):dict(phases={k:0. for k in ct.COLLECTOR_PHASES})for i in (0,1)for j in range(10)}
        return events,declaration,owner,banks

    def test_report_only_repair_preserves_original_clocks_and_unavailable_components(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);events,declaration,owner,banks=self.evidence(root)
            raw_trace=(root/'ct-training-trace.json').read_bytes();raw_owner=owner.read_bytes()
            with patch.object(report,'saved_events',return_value=(events,declaration)),\
                    patch.object(ct,'collector_banks',return_value=banks),\
                    patch.object(ct,'execute',side_effect=AssertionError('no new computation')):
                actual=report.repair_completed_report(root,owner)
            self.assertEqual(actual['declared_games_per_second'],10.)
            self.assertEqual(actual['iteration_wall_seconds'],2000.)
            self.assertEqual(actual['full_study_wall_seconds'],3000.)
            self.assertEqual(actual['threshold'],'RECOVERED_REPORT_ONLY_NO_AUTOMATIC_ADMISSION')
            self.assertIsNone(actual['training']['data_seconds_per_update'])
            self.assertIsNone(actual['training']['updates'])
            self.assertIsNone(actual['training']['prior_durable_updates'])
            self.assertEqual(actual['training']['original_completed_updates'],10)
            self.assertEqual(actual['training']['original_completed_row_visits'],20)
            self.assertIsNone(actual['training']['gpu_kernel_seconds_per_sampled_update'])
            self.assertEqual(actual['training']['concurrent_gpu_attribution']['missing_cpu_scopes'],3)
            self.assertEqual(actual['report_repair']['original_owner']['exit_code'],1)
            self.assertEqual((root/'ct-training-trace.json').read_bytes(),raw_trace)
            self.assertEqual(owner.read_bytes(),raw_owner)
            self.assertFalse((root/'ct-result.json').exists())

    def test_component_measurements_require_exact_durable_cohort_rows(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);events,declaration,owner,banks=self.evidence(root)
            timing=dict(updates=10,prior_durable_updates=0,cache_prepare_seconds=1.,batch_slice_seconds=1.,
                tensor_conversion_seconds=1.,update_wall_seconds=2.,training_datasets=[
                    dict(invocation_dataset_ordinal=i,updates=5,start_step=i*5,end_step=(i+1)*5,
                        start_rows=i*10,end_rows=(i+1)*10,row_visits=10,
                        cache_prepare_seconds=.5,batch_slice_seconds=.5,tensor_conversion_seconds=.5,update_wall_seconds=1.)for i in (0,1)])
            counter=root/'ct-training-counters.json'
            with patch.object(report,'saved_events',return_value=(events,declaration)),patch.object(ct,'collector_banks',return_value=banks):
                for change in ('row_visits','end_rows','updates','total'):
                    bad=copy.deepcopy(timing)
                    if change=='total':bad['batch_slice_seconds']=1000.
                    else:bad['training_datasets'][0][change]=1000
                    counter.write_text(json.dumps(dict(schema='pokezero.ct-training-counters.v1',counters=bad)))
                    with self.assertRaises(ValueError):report.repair_completed_report(root,owner)
                    self.assertFalse((root/'ct-result-report-repair.json').exists())
                counter.write_text(json.dumps(dict(schema='pokezero.ct-training-counters.v1',counters=timing)))
                actual=report.repair_completed_report(root,owner)
            self.assertEqual(actual['training']['updates'],10)
            self.assertEqual(actual['training']['data_seconds_per_update'],.2)
            self.assertEqual([x['row_visits']for x in actual['training']['training_datasets']],[10,10])

    def test_incomplete_reload_interruption_roster_or_artifact_never_becomes_completion(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);events,declaration,owner,banks=self.evidence(root)
            for change in ('reload','window','roster','digest','numerical_resume'):
                altered=copy.deepcopy(events)
                if change=='reload':altered[-1]['payload']['jax_reload']='UNAVAILABLE'
                elif change=='window':altered[-1]['payload']['pipeline_windows'][0]['includes_interruption']=True
                elif change=='roster':altered[0]['payload']['stream_id']=1
                elif change=='digest':altered[-1]['payload']['complete_iterations'][0]['snapshot']['checkpoint_sha256']='0'*64
                else:altered.insert(0,dict(kind='RUNTIME_PHASE',payload=dict(scope='TRAINER_RESUME')))
                with patch.object(report,'saved_events',return_value=(altered,declaration)),patch.object(ct,'collector_banks',return_value=banks):
                    with self.assertRaises(ValueError):report.repair_completed_report(root,owner)
                self.assertFalse((root/'ct-result-report-repair.json').exists())

    def test_journal_hash_chain_and_declaration_are_read_only(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);(root/'journal').mkdir();declaration={'declared':'inert'}
            (root/'declaration.json').write_text(json.dumps(declaration))
            event=dict(sequence=0,previous=None,time=1.,kind='RUN_STARTED',payload=dict(declaration_sha256=report._sha(declaration)))
            event['sha256']=report._sha(event);path=root/'journal/000000000.json';path.write_text(json.dumps(event))
            before=path.read_bytes();actual,manifest=report.saved_events(root)
            self.assertEqual(actual,[event]);self.assertEqual(manifest,declaration);self.assertEqual(path.read_bytes(),before)
            event['payload']['declaration_sha256']='changed';path.write_text(json.dumps(event))
            with self.assertRaises(ValueError):report.saved_events(root)


if __name__=='__main__':unittest.main()
