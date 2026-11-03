"""C-T denominator, missing cells and actual GPU trace classification controls."""
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from pokezero import v5_phasec_throughput as ct, v5_training as training


def result(wall):
    return dict(declared_games=20000,status='COMPLETE_DECLARATION',
        jax_reload='VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED',steps=20100,
        original_run_wall_seconds=wall,collection_runtime={'kernel_blocked_seconds':33.0},
        original_phase_wall_seconds={'TRAIN_CACHES_INCLUSIVE':10.0},
        dispositions={'COMPLETE':19980,'CAPPED':20,'FAILED':0,'REFUSED':0,'RUNNING':0,'NOT_ATTEMPTED':0})


class PhaseCThroughputTests(unittest.TestCase):
    def test_concurrent_instrumentation_keeps_each_cohort_data_and_update_budget(self):
        import json
        from types import SimpleNamespace
        from pokezero import dataset
        def update(runtime,batch):runtime.steps+=1;runtime.rows+=10;return 'original'
        def train(runtime,paths,**kwargs):
            self.assertEqual(kwargs['resident_cache_bytes'],1024)
            training._cache_reader(None)
            for _ in range(2):
                dataset.slice_training_batch(None);training._training_tensors(None);training._update(runtime,None)
        with tempfile.TemporaryDirectory()as folder:
            scopes=[dict(ph='X',name='pokezero.learner_update',pid=1,tid=1,ts=i*100,dur=50)for i in range(4)]
            (Path(folder)/'ct-training-trace.json').write_text(json.dumps(dict(traceEvents=scopes)))
            runtime=SimpleNamespace(steps=64,rows=640)
            with patch.object(training,'_update',side_effect=update),patch.object(training,'train_caches',side_effect=train),\
                    patch.object(training,'_cache_reader'),patch.object(training,'_training_tensors'),patch.object(dataset,'slice_training_batch'):
                with ct.instrument_training(folder,1024,concurrent_collection=True)as state:
                    training.train_caches(runtime,['cohort0']);training.train_caches(runtime,['cohort0'])
                    training.train_caches(runtime,['cohort1'])
            self.assertEqual(state['updates'],6)
            self.assertEqual([x['updates']for x in state['training_datasets']],[4,2])
            self.assertEqual([x['row_visits']for x in state['training_datasets']],[40,20])
            for x in state['training_datasets']:
                self.assertGreaterEqual(x['data_seconds_per_update'],0)
                self.assertEqual(x['end_step']-x['start_step'],x['updates'])

    def test_fixed_pipeline_window_never_hides_fill_drain_or_changes_numerator(self):
        smoke=dict(result(3000),declared_games=40000,iteration_policy_lag=1,pipeline_iterations=2,
            pipeline_windows=[dict(training_iteration=0,collection_iteration=1,wall_seconds=2000.,includes_interruption=False)],
            complete_iterations=[dict(iteration=i)for i in (0,1)],collection_iterations=[dict(iteration=i)for i in (0,1)],
            acting_snapshots=[dict(current=dict(iteration=-1))for _ in (0,1)],
            dispositions=dict(COMPLETE=39980,CAPPED=20))
        smoke['cohort_training_budgets']=[dict(iteration=i,start_step=i*10050,end_step=(i+1)*10050,row_visits=50000)for i in (0,1)]
        timing=dict(updates=20100,training_datasets=[dict(iteration_ordinal=i,updates=10050,
            start_step=i*10050,end_step=(i+1)*10050,row_visits=50000)for i in (0,1)])
        report=ct.gate_report(smoke,timing)
        self.assertEqual(report['declared_games_per_second'],10.)
        self.assertEqual(report['gate_numerator_games'],20000)
        self.assertEqual(report['full_study_declared_games_per_second'],40000/3000)
        self.assertEqual(report['natural_complete_per_second'],39980/3000)
        self.assertEqual(report['declared_games'],40000)
        for bad in (dict(smoke,declared_games=20000),dict(smoke,pipeline_windows=smoke['pipeline_windows']*2),
                dict(smoke,complete_iterations=[dict(iteration=0)])):
            with self.assertRaises(ValueError):ct.gate_report(bad,timing)
        with self.assertRaises(ValueError):ct.gate_report(smoke,dict(updates=20100))

        resumed=dict(timing,updates=10050,prior_durable_updates=10050,
            training_datasets=[dict(timing['training_datasets'][1],iteration_ordinal=0)])
        recovered=ct.gate_report(smoke,resumed)
        self.assertEqual(recovered['threshold'],'RECOVERED_DIAGNOSTIC_NO_AUTOMATIC_ADMISSION')
        self.assertEqual(recovered['training']['training_datasets'][0]['original_iteration'],1)
        self.assertEqual(recovered['training']['prefix_component_times'],'UNAVAILABLE_NOT_ZERO')
        self.assertEqual(recovered['training']['cohort_training_budgets'],smoke['cohort_training_budgets'])
        finished=ct.gate_report(smoke,dict(updates=0,prior_durable_updates=20100,training_datasets=[]))
        self.assertEqual(finished['threshold'],'RECOVERED_DIAGNOSTIC_NO_AUTOMATIC_ADMISSION')
        self.assertEqual(finished['training']['training_datasets'],[])

    def test_gpu_profile_joins_main_and_autograd_kernels_but_excludes_overlapping_jax(self):
        events=[]
        for i in range(4):
            start=1000*i
            events.extend([
                dict(ph='X',cat='user_annotation',name='pokezero.learner_update',pid=1,tid=10,ts=start,dur=500),
                dict(ph='X',cat='cuda_runtime',pid=1,tid=10,ts=start+10,dur=1,args={'correlation':i}),
                dict(ph='X',cat='cpu_op',pid=1,tid=20,ts=start+20,dur=1,args={'External id':100+i}),
                dict(ph='X',cat='kernel',dur=20,args={'correlation':i}),
                dict(ph='X',cat='kernel',dur=30,args={'correlation':50+i,'External id':100+i}),
                dict(ph='X',cat='cuda_runtime',pid=1,tid=11,ts=start+25,dur=1,args={'correlation':90+i}),
                dict(ph='X',cat='kernel',dur=999,args={'correlation':90+i}),
            ])
        report=ct.learner_trace_categories(dict(traceEvents=events))
        self.assertEqual(report['learner']['kernel']['events'],8)
        self.assertAlmostEqual(report['learner']['kernel']['seconds'],200e-6)
        self.assertEqual(report['other_or_unattributed']['kernel']['events'],4)
        with self.assertRaises(ValueError):ct.learner_trace_categories(dict(traceEvents=[]))

    def test_all_collector_write_phases_survive_without_becoming_a_new_denominator(self):
        banks=[{'phases':{name:float(i+1)for name in ct.COLLECTOR_PHASES}}for i in range(10)]
        actual=ct.collector_phase_totals(banks)
        self.assertEqual(actual['banks'],10)
        for key in ('spool_write_wall_seconds','game_cache_wall_seconds','cache_merge_wall_seconds'):
            self.assertEqual(actual['wall_seconds'][key],55.0)
        with self.assertRaises(ValueError):ct.collector_phase_totals(banks[:-1])
        broken=copy.deepcopy(banks);del broken[4]['phases']['game_cache_wall_seconds']
        with self.assertRaises(ValueError):ct.collector_phase_totals(broken)
        broken=copy.deepcopy(banks);broken[0]['phases']['game_cache_wall_seconds']=float('nan')
        with self.assertRaises(ValueError):ct.collector_phase_totals(broken)

    def test_actual_compile_metadata_is_separate_from_host_walls_and_counts(self):
        phases={name:1. for name in ct.COLLECTOR_PHASES+ct.COMPILE_SECONDS}
        phases.update({name:1 for name in ct.COMPILE_CALLS})
        actual=ct.collector_phase_totals([dict(phases=phases)]*10)
        self.assertEqual(actual['wall_seconds']['game_cache_wall_seconds'],10.)
        self.assertEqual(actual['lower_compile_seconds']['collector_compile_seconds'],10.)
        self.assertEqual(actual['compile_calls']['collector_compile_calls'],10)
        recovered=ct.collector_phase_totals([dict(phases=phases)]*9,interrupted_banks=1)
        self.assertEqual(recovered['unavailable_interrupted_banks'],1)
        for bad in (dict(phases,unexpected=1.),dict(phases,collector_compile_calls=1.5)):
            with self.assertRaises(ValueError):ct.collector_phase_totals([dict(phases=bad)]*10)

    def test_pretraining_continuation_uses_original_run_and_retains_original_start(self):
        from types import SimpleNamespace
        from pokezero import v5_run as runs
        class Store:
            events=[dict(kind='RUN_STARTED',time=10.)]
            def matching(self,kind):return [dict(steps=0)]if kind=='CHECKPOINT'else []
        original=lambda plan,path,**kw:dict(original_resume=kw['resume'])
        with patch.object(runs,'run',original),patch.object(runs,'Store',return_value=Store()),                patch.object(ct,'recover_published_parts'),patch.object(ct.time,'time',return_value=20.):
            with ct.continue_pretraining_iteration('/tmp/inert')as state:
                self.assertEqual(runs.run(object(),'/tmp/inert'),dict(original_resume=True))
                self.assertEqual(state['original_run_started_epoch'],10.)
                self.assertEqual(state['run_finished_epoch'],20.)
            self.assertIs(runs.run,original)
        store=Store();store.matching=lambda kind:[dict(steps=1)]if kind=='CHECKPOINT'else []
        with patch.object(runs,'Store',return_value=store),patch.object(ct,'recover_published_parts')as recover:
            with ct.continue_pretraining_iteration('/tmp/inert'),self.assertRaisesRegex(ValueError,'prior update timings'):
                runs.run(object(),'/tmp/inert')
            recover.assert_not_called()

    def test_trained_prefix_is_explicit_and_checkpoint_loss_stays_visible(self):
        from pokezero import v5_run as runs
        class Store:
            events=[dict(kind='RUN_STARTED',time=10.)]
            def matching(self,kind):
                return {'CHECKPOINT':[dict(steps=64,rows=128)],'TRAIN_PROGRESS':[dict(steps=96)],'CT_INTERRUPTED_BANK_RECOVERED':[dict(iteration=0,batch=1)]}.get(kind,[])
        original=lambda plan,path,**kw:dict(original_resume=kw['resume'])
        with patch.object(runs,'run',original),patch.object(runs,'Store',return_value=Store()),patch.object(ct,'recover_published_parts'),patch.object(ct.time,'time',return_value=20.):
            with ct.continue_pretraining_iteration('/tmp/inert',allow_trained_prefix=True)as state:
                self.assertEqual(runs.run(object(),'/tmp/inert'),dict(original_resume=True))
                self.assertEqual(state['prior_durable_updates'],64);self.assertEqual(state['prior_reported_updates'],96)
                self.assertEqual(state['uncheckpointed_reported_updates'],32)
                self.assertEqual(state['recovered_banks'],[dict(iteration=0,batch=1)])
            self.assertIs(runs.run,original)

    def test_resume_reuses_original_profile_without_claiming_prefix_data_times(self):
        import json
        from types import SimpleNamespace
        def update(runtime,batch):runtime.steps+=1;return 'original'
        with tempfile.TemporaryDirectory()as folder:
            trace=Path(folder)/'ct-training-trace.json'
            original=json.dumps({'traceEvents':[{'ph':'X','cat':'kernel','dur':40}]}).encode();trace.write_bytes(original)
            runtime=SimpleNamespace(steps=64)
            with patch.object(training,'_update',side_effect=update),patch.object(training,'_torch',side_effect=AssertionError('no new profile')):
                with ct.instrument_training(folder,1024)as state:
                    self.assertEqual(training._update(runtime,object()),'original')
                    self.assertEqual(training._update(runtime,object()),'original')
            self.assertEqual(trace.read_bytes(),original);self.assertEqual(state['updates'],2)
            self.assertEqual(state['prior_durable_updates'],64);self.assertEqual(state['profile_updates'],[])
            self.assertEqual(state['gpu_sample_update_ordinals'],[8,9,10,11])
            self.assertTrue(state['gpu_profile_scope'].startswith('REUSED_ORIGINAL_PREFIX'))
            r=ct.gate_report(dict(result(1000),steps=66),state)
            self.assertEqual(r['threshold'],'RECOVERED_TRAINED_PREFIX_NO_AUTOMATIC_ADMISSION')
            with self.assertRaises(ValueError):ct.gate_report(dict(result(1000),steps=67),state)

    def test_copied_controller_is_compatible_outside_original_checkout(self):
        import json,runpy
        from types import SimpleNamespace
        module=runpy.run_path(ct.__file__,run_name='standalone_ct_controller')
        with tempfile.TemporaryDirectory()as folder:
            (Path(folder)/'ct-training-trace.json').write_text(json.dumps({'traceEvents':[]}))
            runtime=SimpleNamespace(steps=64)
            def update(r,b):r.steps+=1;return 'original'
            with patch.object(training,'_update',side_effect=update):
                with module['instrument_training'](folder,1024)as state:training._update(runtime,object())
            self.assertEqual(state['prior_durable_updates'],64);self.assertEqual(state['updates'],1)

    def test_empty_publication_prefix_recovers_zero_rows_but_contradictions_refuse(self):
        import json
        from types import SimpleNamespace as NS
        from pokezero import v5_run as runs
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);game=NS(stream_id=41,battle_id='declared',seed=12,team_indices=(0,1))
            batch=NS(games=(game,),constructor_seed=21,sampling_seed=31)
            plan=NS(iterations=((batch,),),manifest=lambda:{'declared':'inert'})
            body=dict(streams=[41],declaration_sha256=runs._sha(plan.manifest()),games=[dict(stream=41,battle_id='declared',seed=12,team_indices=[0,1])],constructor_seed=21,sampling_seed=31)
            (root/'bank.json').write_text(json.dumps(body))
            class Store:
                path=root
                def __init__(self,bank,*,ended=False,orphan=False,duplicate=False):
                    self.bank,self.ended,self.orphan,self.duplicate=bank,ended,orphan,duplicate;self.emitted=[]
                def matching(self,kind,**kw):
                    if kind=='BATCH_STARTED':return [dict(iteration=0,batch=0)]*(2 if self.duplicate else 1)
                    if kind=='DEVICE_ACTOR_BANK'and self.bank:return [dict(path='bank.json',sha256='inert')]
                    if kind=='DEVICE_ROLLOUT_CHUNK'and self.orphan:return [dict(bank_sha256='unbound')]
                    if kind=='GAME_END'and self.ended:return [dict(stream_id=41,status='COMPLETE')]
                    return []
                def resolve(self,path,digest):return root/path
                def emit(self,kind,**payload):self.emitted.append((kind,payload))
            for bank in (False,True):
                store=Store(bank);records=ct.recover_published_parts(store,plan)
                self.assertEqual(records[0]['published_game_parts'],0);self.assertIsNone(records[0]['cache'])
                self.assertEqual(store.emitted[0][0],'CT_INTERRUPTED_BANK_RECOVERED')
                with self.assertRaises(ValueError):ct.recover_published_parts(Store(bank,ended=True),plan)
            for store in (Store(False,orphan=True),Store(True,duplicate=True)):
                with self.assertRaises(ValueError):ct.recover_published_parts(store,plan)

    def test_saved_game_parts_recover_without_engine_and_missing_rows_stay_explicit(self):
        import json
        import numpy as np
        from pokezero import v5_run as runs
        from gpu_engine import v5_device_rollout as rollout,v5_device_spool as spool_module
        from tests.test_gpu_engine_v5_device_spool import DeviceSpoolTests
        with tempfile.TemporaryDirectory()as temporary:
            plan,store,pending,raw,_=DeviceSpoolTests().example(temporary)
            batch=plan.iterations[0][0];bank=pending['bank'];bindings=pending['bindings']
            raw=raw._replace(carry=raw.carry._replace(status=np.array([rollout.COMPLETE,rollout.COMPLETE],np.int32)))
            body=dict(streams=list(bank.streams),declaration_sha256=runs._sha(plan.manifest()),
                games=[dict(stream=g.stream_id,battle_id=g.battle_id,seed=g.seed,team_indices=list(g.team_indices))for g in batch.games],
                constructor_seed=batch.constructor_seed,sampling_seed=batch.sampling_seed,assignment_sha256=bank.assignment_sha256,
                actor_archives=bank.actor_archives,actor_indices=bank.actor_indices)
            store.emit('BATCH_STARTED',iteration=0,batch=0,streams=list(bank.streams))
            reference=store.artifact('rollouts/inert-bank.json',runs._json(body).encode(),kind='DEVICE_ACTOR_BANK')
            spool=spool_module.Spool(store,bank,bank_sha256=reference['sha256']);spool.append(raw)
            for game in batch.games:store.emit('GAME_END',iteration=0,batch=0,stream_id=game.stream_id,status='COMPLETE')
            with patch.object(rollout,'_complete_gae',side_effect=lambda v,o,l,d:(v*0,v*0)):
                spool.finalize_game(0,batch.games[0],bindings=bindings,loss_config=plan.loss,discount=plan.discount)
            spool.merge(Path(temporary)/'expected')
            original_emit=store.emit
            def interrupted(kind,**payload):
                if kind=='CACHE':raise KeyboardInterrupt('inert crash after durable recovery')
                return original_emit(kind,**payload)
            with patch.object(store,'emit',interrupted),self.assertRaises(KeyboardInterrupt):
                ct.recover_published_parts(store,plan)
            with patch.object(spool_module.Spool,'merge',side_effect=AssertionError('replayed recovered merge')):
                records=ct.recover_published_parts(store,plan)
            self.assertEqual(records[0]['completed_games_without_published_training_rows'],[batch.games[1].stream_id])
            self.assertEqual(records[0]['published_game_parts'],1)
            target=store.path/records[0]['cache']
            self.assertEqual((target/'metadata.json').read_bytes(),(Path(temporary)/'expected/metadata.json').read_bytes())
            self.assertEqual(ct.recover_published_parts(store,plan),[])
            self.assertEqual(len(store.matching('CACHE')),1)

    def test_full_iteration_wall_is_the_denominator_and_unknowns_are_retained(self):
        for wall,status in ((1000,'TARGET20_MET'),(4000,'FLOOR5_MET_TARGET20_MISSED'),(4001,'BELOW_BINDING5')):
            original=result(wall);saved=copy.deepcopy(original)
            report=ct.gate_report(original,{'updates':20100})
            self.assertEqual(report['threshold'],status)
            self.assertEqual(report['declared_games_per_second'],20000/wall)
            self.assertEqual(report['natural_complete_per_second'],19980/wall)
            self.assertEqual(report['dispositions']['CAPPED'],20)
            self.assertEqual(original,saved)

    def test_distinct_shape_and_d_slower_admission_keep_full_iteration_wall(self):
        for wall,status in ((1000,'D_GE5'),(4000,'D_GE5'),(4001,'D_GE1_LT5'),(20000,'D_GE1_LT5'),(20001,'D_LT1_HOLD')):
            r=ct.gate_report(dict(result(wall),model_arm='D'),{'updates':20100})
            self.assertEqual(r['threshold'],status);self.assertEqual(r['model_arm'],'D')
            self.assertEqual(r['target20_met'],wall<=1000)
        self.assertEqual(ct.gate_report(dict(result(4000),model_arm='A'),{'updates':20100})['threshold'],'FLOOR5_MET_TARGET20_MISSED')

    def test_partial_iterations_or_bad_wall_never_receive_an_admission(self):
        for change in ({'declared_games':19999},{'status':'STOPPED'}, {'steps':0},
                {'jax_reload':'UNAVAILABLE'},*({'original_run_wall_seconds':v}for v in (0,-1,True,float('nan'),float('inf')))):
            with self.subTest(change=change),self.assertRaises(ValueError):
                ct.gate_report(dict(result(1000),**change),{'updates':20100})
        with self.assertRaises(ValueError):ct.gate_report(result(1000),{'updates':20099})

    def test_only_actual_kernel_and_memory_categories_count_not_host_annotations(self):
        report=ct.gpu_trace_categories({'traceEvents':[
            {'ph':'X','cat':'kernel','dur':30},
            {'ph':'X','cat':'kernel','dur':50},
            {'ph':'X','cat':'gpu_memcpy','dur':7},
            {'ph':'X','cat':'gpu_memset','dur':3},
            {'ph':'X','cat':'user_annotation','dur':9000},
            {'ph':'X','cat':'cpu_op','dur':10000},
            {'ph':'i','cat':'kernel','dur':2000}]})
        self.assertEqual(report['kernel']['events'],2)
        self.assertAlmostEqual(report['kernel']['seconds'],80/1e6)
        self.assertAlmostEqual(report['gpu_memcpy']['seconds'],7/1e6)
        for duration in (True,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                ct.gpu_trace_categories({'traceEvents':[{'ph':'X','cat':'kernel','dur':duration}]})

    def test_disabled_execution_and_failed_instrumentation_restore_original_calls(self):
        with self.assertRaisesRegex(ValueError,'explicit finite'):
            ct.execute({},'/tmp/inert',resident_cache_bytes=1)
        original=training.train_caches;update=training._update
        with self.assertRaisesRegex(RuntimeError,'inert failure'):
            with ct.instrument_training('/tmp/inert',1024):
                self.assertIsNot(training.train_caches,original)
                raise RuntimeError('inert failure')
        self.assertIs(training.train_caches,original)
        self.assertIs(training._update,update)
        for budget in (True,0,-1):
            with self.assertRaises(ValueError):
                with ct.instrument_training('/tmp/inert',budget):pass


if __name__=='__main__':unittest.main()
