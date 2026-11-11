"""Fake panels/providers and tiny authored saved rows only; no actual inference."""
import asyncio
from contextlib import ExitStack,redirect_stdout
from dataclasses import asdict,replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,Mock

import numpy as np
from pokezero import v5_evaluation as e,v5_run as r,v5_foulplay as fp,v5_worker as worker
from pokezero import foulplay_bridge as old,v5_data as data
from pokezero.v5_model import V5ModelConfig
import test_v5_data as fixture
import test_v5_run as run_fixture


def artifact(path,value):
    blob=value if isinstance(value,bytes) else r._json(value).encode();path.write_bytes(blob)
    return asdict(e.Artifact(str(path),hashlib.sha256(blob).hexdigest()))


def game(config,seed):
    return dict(seed=seed,capped=False,tied=False,winner=config.pokezero_username,pokezero_won=True,
        terminal_witness=dict(status='PUBLIC_WIN',winner_player=config.pokezero_player,
            public_terminal_line='|win|'+config.pokezero_username))


def row(index,category,*,complete=True):
    value=fixture.row(index,status='COMPLETE' if complete else 'CAPPED')
    original=value.step.observation;numeric=[list(x) for x in original.numeric_features];numeric[0][0]=index/10
    original=replace(original,numeric_features=tuple(tuple(x) for x in numeric))
    obs=replace(value.observation,numeric_features=tuple(tuple(x) for x in numeric)+value.observation.numeric_features[23:],
        category_vocabulary_sha256=category)
    return replace(value,step=replace(value.step,observation=original),observation=obs,
        source_frame_sha256=hashlib.sha256(str(index).encode()).hexdigest())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        # Explicit UNIT harness only. No such resolver is present in production.
        unit=NS(require_original_capture=lambda p:None,
            resolve_original_capture=lambda p,r:e._mechanism_data(p),
            evaluate_original_capture=lambda p,f,d:e._mechanism_arithmetic(p,f,d))
        self.capture_patch=patch.object(e,'_capture_resolver',return_value=unit)
        self.capture_patch.start();self.addCleanup(self.capture_patch.stop)

    def fixture(self,root):
        root=root.resolve()
        for name in ('show','fp'): (root/name).mkdir()
        python=artifact(root/'python',b'fake-python-never-executed');node=artifact(root/'node',b'fake-node-never-executed')
        vocab=dict(tokens=['unit'+str(i) for i in range(111)],oov_buckets=16,aliases={})
        category=r._sha(vocab);vocab_ref=artifact(root/'vocab.json',vocab)
        source=row(0,category);unknown=row(20,category,complete=False)
        config=V5ModelConfig(128,category,source.observation.label_vocabulary,arm='B')
        header=dict(schema=e.POSITIONS,split='PHASE_C_FIXED_HELDOUT',rows=2,
            category_vocabulary_sha256=category,label_vocabulary_sha256=config.label_vocabulary.sha256)
        positions=root/'positions.jsonl';positions.write_text('\n'.join(r._json(x) for x in (header,data.row_to_dict(source),data.row_to_dict(unknown)))+'\n')
        identity=list(e._identity(source));group=dict(root_identity=identity,world_id='actual-fixture-world',chance_id='actual-fixture-chance',
            opponent_action=0,selection='ALL_LEGAL_ACTIONS_BEFORE_VALUES',source_receipt_sha256='a'*64,branches=[])
        for action,legal in enumerate(source.observation.legal_action_mask):
            child=row(action+1,category);address=replace(source.observation.address,frame=source.observation.address.frame+1)
            child=replace(child,observation=replace(child.observation,address=address),labels=replace(child.labels,address=address))
            if legal:group['branches'].append(dict(action=action,root_identity=identity,world_id=group['world_id'],chance_id=group['chance_id'],
                opponent_action=0,terminal=False,terminal_outcome=None,terminal_source_sha256=None,
                decision=data.row_to_dict(child),child_state_sha256=str(action+1)*64))
        siblings=root/'siblings.jsonl';siblings.write_text(r._json(dict(header,schema=e.SIBLINGS,rows=1))+'\n'+r._json(group)+'\n')
        runtime=dict(schema=worker.SCHEMA,category_vocabulary=vocab_ref,
            showdown=dict(root=str(root/'show'),node=node),memory_policy=dict(environment=dict(zip(worker.ENV_KEYS,('false',None,None))),torch_fraction=None),
            precision_policy=dict(inference='bfloat16'))
        runtime_ref=artifact(root/'runtime.json',runtime)
        configs=[]
        for seat in ('p1','p2'):
            cfg=old.ControlledFoulPlayConfig(checkpoint=Path('/unused-template-never-opened'),showdown_root=root/'show',node_binary=node['path'],
                foulplay_root=root/'fp',foulplay_python=Path(python['path']),games=500,seed_start=100,
                foulplay_random_seed=1000,policy_mode='raw',device='cpu',encoder_kind='jax-owning',belief_set_source=True,
                allow_search_fallback=False,opponent_crash_retries=0,pokezero_player=seat,owning_decision_timeout_seconds=60.)
            configs.append(dict(json.loads(json.dumps(asdict(cfg),default=str)),checkpoint=None))
        protocol=dict(schema=e.SCHEMA,model_sha256=r._sha(config.metadata()),label_vocabulary_sha256=config.label_vocabulary.sha256,
            runtime=runtime_ref,worker_python=python,foulplay_python=python,foulplay_commit='f'*40,
            foulplay_source=artifact(root/'foulplay-component.json',{'unit_metadata_only':True}),
            positions=asdict(e.Artifact(str(positions),r._file_sha(positions))),siblings=asdict(e.Artifact(str(siblings),r._file_sha(siblings))),
            panels=[dict(declared_games=500000,seeds=list(range(100,600)),configs=configs)],
            actor_cpus=[0],foulplay_cpus=[1,2,3,4],watchdog=asdict(r.Watchdog(10,backstop_seconds=1000)),bins=5,batch_size=4,deterministic=True)
        reference=e.Artifact(**artifact(root/'protocol.json',protocol));run_root=root/'run';run_root.mkdir()
        checkpoint=run_root/'checkpoint.pt';checkpoint.write_bytes(b'fake-checkpoint');archive=run_root/'archive.npz';archive.write_bytes(b'fake-archive')
        declaration=dict(model=config.metadata(),evaluation=r.EvaluationProtocol(reference.sha256).metadata())
        (run_root/'declaration.json').write_text(r._json(declaration));decl=r._sha(declaration)
        snapshot=r.Snapshot(399,'checkpoint.pt',r._file_sha(checkpoint),'archive.npz',r._file_sha(archive),r._sha(config.metadata()))
        request=r.EvaluationRequest(500000,decl,reference.sha256,snapshot,run_root,checkpoint,archive,lambda receipt:None)
        return reference,protocol,config,request

    def test_fixed_protocol_and_missing_or_foreign_dependencies_refuse_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root)
            with patch.object(fp,'supervise_panel') as launch:
                hook=e.EvaluationHook(ref,request.declaration_sha256)
                with self.assertRaises(r.MissingDependency):hook(replace(request,progress_callback=None))
                for key in ('positions','siblings','runtime'):
                    changed=json.loads(r._json(protocol));changed[key]['sha256']='f'*64
                    bound=e.Artifact(**artifact(root/(key+'-bad.json'),changed))
                    with self.subTest(key=key),self.assertRaises(ValueError):e.EvaluationHook(bound,request.declaration_sha256)
                for change in ('games','cores','seats'):
                    changed=json.loads(r._json(protocol))
                    if change=='games':changed['panels'][0]['configs'][0]['games']=499
                    if change=='cores':changed['foulplay_cpus']=[1,1]
                    if change=='seats':changed['panels'][0]['configs'][1]['pokezero_player']='p1'
                    bound=e.Artifact(**artifact(root/(change+'-bad.json'),changed))
                    with self.subTest(change=change),self.assertRaises(ValueError):e.EvaluationHook(bound,request.declaration_sha256)
                launch.assert_not_called()

    def test_missing_actual_capture_integration_refuses_even_complete_numeric_fixtures(self):
        with tempfile.TemporaryDirectory() as temp:
            ref,protocol,_,request=self.fixture(Path(temp))
            with patch.object(e,'_capture_resolver',side_effect=r.MissingDependency('genuine capture not integrated')),patch.object(fp,'supervise_panel') as launch:
                with self.assertRaises(r.MissingDependency):e.EvaluationHook(ref,request.declaration_sha256)
                with self.assertRaises(r.MissingDependency):e.mechanism(protocol,object(),e._mechanism_data(protocol))
                launch.assert_not_called()

    def test_genuine_data_contract_rejects_adjacent_context_missing_siblings_and_private_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,protocol,_,_=self.fixture(root);positions,groups=e._mechanism_data(protocol)
            self.assertEqual((len(positions),len(groups)),(2,1))
            original=Path(protocol['siblings']['path']).read_text().splitlines();group=json.loads(original[1])
            for mutation in ('missing','reply','owner','private','adjacent'):
                changed=json.loads(r._json(group))
                if mutation=='missing':changed['branches'].pop()
                if mutation=='reply':changed['branches'][0]['opponent_action']=1
                if mutation=='owner':changed['branches'][0]['decision']['observation']['address']['owner']=1
                if mutation=='private':changed['branches'][0]['decision']['observation']['view']='PRIVILEGED'
                if mutation=='adjacent':
                    changed['branches'][0]['decision']['observation']['address']['frame']+=1
                    changed['branches'][0]['decision']['labels']['address']['frame']+=1
                path=root/(mutation+'.jsonl');path.write_text(original[0]+'\n'+r._json(changed)+'\n')
                candidate=dict(protocol,siblings=asdict(e.Artifact(str(path),r._file_sha(path))))
                with self.subTest(mutation=mutation),self.assertRaises((ValueError,r.MissingDependency)):e._mechanism_data(candidate)

    def test_document_and_stream_hash_bind_the_exact_bytes_parsed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,_,_=self.fixture(root)
            with patch.object(Path,'read_bytes',return_value=b'{"foreign":true}'):
                with self.assertRaisesRegex(ValueError,'digest'):ref.document()
            original=Path(protocol['positions']['path']);stream=e._records(e.Artifact(**protocol['positions']),e.POSITIONS)
            next(stream)
            with original.open('ab') as output:output.write(b'\n')
            with self.assertRaises(ValueError):list(stream)

    def test_mechanism_uses_only_public_inputs_actual_outcomes_and_value_top_two(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,protocol,config,_=self.fixture(root);captured=e._mechanism_data(protocol);inputs=[]
            def prepare(columns,cfg):
                self.assertEqual(set(columns),set(('categorical_ids','numeric_features','token_type_ids','attention_mask','history_mask',
                    'schema','observation_schema','category_vocabulary_sha256','label_vocabulary_sha256','view')))
                inputs.append(columns);return columns
            def forward(columns):return NS(value=columns['numeric_features'][:,0,0,0],valid=np.ones(len(columns['history_mask']),bool))
            fake_model=NS(prepare_inputs=prepare);fake_live=NS(check_factory=lambda f:f)
            import gpu_engine
            with patch.dict(sys.modules,{'gpu_engine.v5_model':fake_model,'gpu_engine.v5_live_protocol':fake_live}),\
                    patch.object(gpu_engine,'v5_model',fake_model,create=True):
                result=e.mechanism(protocol,NS(config=config,forward=forward),captured)
            self.assertEqual((result['heldout_positions'],result['known_outcomes'],result['unknown_outcomes']),(2,1,1))
            self.assertEqual(result['calibration']['mse'],1.)
            self.assertEqual(result['top_two_rows'][0]['top_two_actions'],[1,0])
            self.assertEqual(result['share_gap_exceeds_outcome_rmse'],0.)
            self.assertIn('UNMEASURED',result['actual_ranking_accuracy']);self.assertEqual(result['fitted_models'],0)
            with self.assertRaisesRegex(ValueError,'namespace'):e._columns([next(iter(captured[0].values()))],replace(config,category_vocabulary_sha256='f'*64))

    def test_selected_offset_preserves_original_schedule_and_default_recipe(self):
        config=old.ControlledFoulPlayConfig(checkpoint=Path('/fake'),showdown_root=Path('/fake-source'),games=5,seed_start=10,
            foulplay_random_seed=100,policy_mode='raw',device='cpu',encoder_kind='jax-owning',belief_set_source=True,
            allow_search_fallback=False,opponent_crash_retries=0)
        factory=NS(contract=NS(checkpoint_sha256='c'*64,feature_masks=tuple(dict(exact_state=False,feature_pack_last_move=False,investment_belief_narrowing=False,
            item_belief_narrowing=False,stats_block=False,tier2_investment=False,tier2_residuals=False,transition_token_budget=0).items())),
            catalog=NS(vocabulary=object(),dex=object()))
        seen=[]
        async def source(cfg,**kwargs):
            offset=kwargs['game_offsets'][0];seen.append(offset);value=game(cfg,cfg.seed_start+offset)
            return NS(games=(NS(seed=value['seed'],capped=False,to_dict=lambda:value),),foulplay_random_seed_schedule=(100+offset,))
        with patch.object(fp,'_recipe',side_effect=lambda *a:{'original':True,'archive_sha256':'a'*64}),patch.object(old,'_run_controlled_foulplay_games',side_effect=source):
            subset=asyncio.run(fp.run_panel(config,factory,enable_v5=True,game_offsets=(3,)))
            default=asyncio.run(fp.run_panel(config,factory,enable_v5=True))
        self.assertEqual(subset['rows'][0]['offset'],3);self.assertEqual(subset['rows'][0]['seed'],13)
        self.assertEqual(subset['rows'][0]['fp_seed'],103);self.assertEqual(len(subset['rows']),1)
        self.assertNotIn('game_offsets',default['recipe']);self.assertEqual(seen,[3,0,1,2,3,4])

    def test_all1000_fixed_cells_concurrent_at_most4_with_ordinal_core_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root);_,panels=e._protocol(ref,parse_configs=True,checkpoint=request.checkpoint_path)
            state={};started=[];active=0;maximum=0
            async def fake(config,factory,*,game_offsets,progress_callback,enable_v5):
                nonlocal active,maximum
                offset=game_offsets[0];index=2*offset+(config.pokezero_player=='p2');started.append(index)
                self.assertEqual(config.foulplay_cpu_affinity,protocol['foulplay_cpus'][index%4]);active+=1;maximum=max(maximum,active)
                initial=dict(offset=offset,seed=config.seed_start+offset,fp_seed=1000+offset,status='NOT_ATTEMPTED',score=None,
                    source_game=None,source_result_sha256=None,duration_seconds=None,reason=None)
                progress_callback(dict(rows=[initial]));await asyncio.sleep(0)
                source=game(config,initial['seed']);final=dict(initial,status='COMPLETE',score=1.,source_game=source,
                    source_result_sha256=r._sha(source),duration_seconds=.1)
                if index%7==0:final.update(status='REFUSED',score=None,source_game=None,source_result_sha256=None,reason='fake-wire-unknown')
                progress_callback(dict(rows=[final]));active-=1;return dict(rows=[final])
            target=root/'progress.json'
            def atomic(path,value):state['value']=value
            def read(path):
                if path==target:return r._json(state['value']).encode()
                return saved_read(path)
            saved_read=Path.read_bytes
            with patch.object(fp,'run_panel',side_effect=fake),patch.object(e,'_atomic',side_effect=atomic),patch.object(Path,'read_bytes',read):
                result=asyncio.run(e._concurrent_panel(panels[500000],object(),protocol,target,'n'*64))
            self.assertEqual(started,list(range(1000)));self.assertEqual(maximum,4)
            self.assertEqual(len(result['rows']),1000)
            self.assertEqual(sum(row['status']=='REFUSED' for row in result['rows']),143)
            self.assertEqual(sum(row['status']=='COMPLETE' for row in result['rows']),857)

    def test_owned_hook_checks_raw_source_and_emits_completed_progress_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root);completed=[];request=replace(request,progress_callback=completed.append)
            hook=e.EvaluationHook(ref,request.declaration_sha256)
            def supervisor(command,*,progress_path,**kwargs):
                self.assertEqual(command[1:3],['-m','pokezero.v5_evaluation']);child_path=Path(command[command.index('--request')+1])
                payload=json.loads(child_path.read_bytes());namespace=r._sha(payload)
                configs=e._pairs(protocol,parse_configs=True,checkpoint=request.checkpoint_path)[500000];rows=e._fixed_rows(configs)
                for row in rows:row.update(status='REFUSED',reason='fake-unsupported-cell')
                source=game(configs[0],rows[0]['seed']);rows[0].update(status='COMPLETE',score=1.,duration_seconds=.2,
                    source_game=source,source_result_sha256=r._sha(source))
                e._atomic(progress_path,dict(schema=fp.SCHEMA,namespace=namespace,rows=rows))
                value=dict(schema=e.SCHEMA,status='RESULT',namespace=namespace,request=payload['request'],rows=rows,
                    scientific_qualification=False,known_completed=1,unknown_cells=999,score_bounds=[.001,1.])
                r._publish(child_path.parent/'worker-report.json',r._json(value).encode());kwargs['sleep'](0)
                return dict(exit_code=0,stop_reason=None)
            with patch.object(fp,'supervise_panel',side_effect=supervisor):outcome=hook(request)
            self.assertEqual(outcome.status,'RESULT');self.assertEqual(len(completed),1)
            self.assertIs(type(completed[0]),r.EvaluationCompletion)
            report=json.loads((request.run_directory/outcome.report).read_bytes())
            self.assertFalse(report['scientific_qualification']);self.assertEqual(report['unknown_cells'],999)
            self.assertEqual(report['score_bounds'],[.001,1.]);self.assertEqual(r._file_sha(request.run_directory/outcome.report),outcome.report_sha256)

    def test_continuation_dispatches_only_untouched_suffix_and_never_lost_inflight(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root)
            configs=e._pairs(protocol,parse_configs=True,checkpoint=request.checkpoint_path)[500000]
            rows=e._fixed_rows(configs);source=game(configs[0],rows[0]['seed'])
            rows[0].update(status='COMPLETE',score=1.,duration_seconds=.2,source_game=source,source_result_sha256=r._sha(source))
            rows[1].update(status='REFUSED',reason='original refusal')
            rows[2].update(status='RUNNING',invocation_id='lost-original-process')
            original=json.loads(r._json(rows));started=[]
            async def fake(config,factory,*,game_offsets,progress_callback,enable_v5):
                index=2*game_offsets[0]+(config.pokezero_player=='p2');started.append(index)
                self.assertEqual(config.foulplay_cpu_affinity,protocol['foulplay_cpus'][index%4])
                await asyncio.sleep(0)
                actual={k:rows[index][k]for k in ('offset','seed','fp_seed','score','source_game','source_result_sha256','duration_seconds')}
                actual.update(status='REFUSED',reason='fake unsupported')
                result=dict(rows=[actual]);progress_callback(result);return result
            with patch.object(fp,'run_panel',side_effect=fake):
                result=asyncio.run(e._concurrent_panel(configs,object(),protocol,root/'continued.json','n'*64,initial_rows=rows))
            self.assertEqual(started,list(range(3,1000)))
            self.assertEqual(result['rows'][:2],original[:2])
            self.assertEqual(result['rows'][2],dict(original[2],status='FAILED',reason='interrupted-inflight-result-unknown-not-replayed'))
            self.assertEqual(rows,original);self.assertEqual(len(result['rows']),1000)

    def test_hook_keeps_immutable_attempts_and_joins_prior_completed_rows(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root)
            configs=e._pairs(protocol,parse_configs=True,checkpoint=request.checkpoint_path)[500000]
            attempts=[];delivered=[];request=replace(request,progress_callback=delivered.append)
            hook=e.EvaluationHook(ref,request.declaration_sha256)
            def supervisor(command,*,progress_path,**kwargs):
                child=Path(command[command.index('--request')+1]);payload=json.loads(child.read_bytes())
                attempts.append(child.parent)
                if len(attempts)==1:
                    self.assertIsNone(payload['resume_progress']);rows=e._fixed_rows(configs)
                    rows[1].update(status='RUNNING',invocation_id='original-dispatch')
                    index=0;status='UNKNOWN';code=1
                else:
                    previous=e.Artifact(**payload['resume_progress']).document()
                    rows=e._lost_inflight(previous['rows']);self.assertEqual(rows[1]['status'],'FAILED')
                    for value in rows[2:]:value.update(status='REFUSED',reason='fake-remaining-unknown')
                    index=2;status='RESULT';code=0
                source=game(configs[index%2],rows[index]['seed'])
                rows[index].update(status='COMPLETE',score=1.,duration_seconds=.2,source_game=source,source_result_sha256=r._sha(source))
                namespace=r._sha(payload);e._atomic(progress_path,dict(schema=fp.SCHEMA,namespace=namespace,rows=rows))
                r._publish(child.parent/'worker-report.json',r._json(dict(schema=e.SCHEMA,status=status,namespace=namespace,
                    request=payload['request'],rows=rows,reason='fake-interruption'if code else None)).encode())
                kwargs['sleep'](0);return dict(exit_code=code,stop_reason=None)
            with patch.object(fp,'supervise_panel',side_effect=supervisor):
                first=hook(request);saved={p:p.read_bytes()for p in attempts[0].iterdir()if p.is_file()}
                second=hook(request)
            self.assertEqual((first.status,second.status),('UNKNOWN','RESULT'))
            self.assertNotEqual(first.report,second.report)
            self.assertTrue(all(p.read_bytes()==blob for p,blob in saved.items()))
            self.assertEqual([x.cell_ordinal for x in delivered],[0,0,2])
            final=json.loads((request.run_directory/second.report).read_bytes())
            self.assertEqual((final['known_completed'],final['unknown_cells'],final['disposition_counts']['FAILED']),(2,998,1))
            # A completed cell rewritten in a later continuation cannot be accepted.
            latest=attempts[-1]/'progress.json';changed=json.loads(latest.read_bytes());changed['rows'][0]['reason']='mutated'
            e._atomic(latest,changed)
            with patch.object(fp,'supervise_panel')as launch,self.assertRaisesRegex(ValueError,'changed'):
                hook(request)
            launch.assert_not_called()

    def test_interruption_before_report_cannot_drop_durable_completion_from_progress(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root)
            configs=e._pairs(protocol,parse_configs=True,checkpoint=request.checkpoint_path)[500000]
            panel=request.run_directory/'artifacts'/('evaluation-500000-'+r._sha(e._request_document(request)))
            directory=panel/'attempt-000000';directory.mkdir(parents=True)
            payload=dict(protocol=asdict(ref),expected_run_declaration_sha256=request.declaration_sha256,
                request=e._request_document(request),directory=str(directory),resume_progress=None)
            r._publish(directory/'request.json',r._json(payload).encode())
            expected=e._fixed_rows(configs);rows=json.loads(r._json(expected));source=game(configs[0],rows[0]['seed'])
            rows[0].update(status='COMPLETE',score=1.,duration_seconds=.2,source_game=source,source_result_sha256=r._sha(source))
            receipt=e._completed_receipts(rows,expected,configs,protocol_path=ref.path)[0]
            e._atomic(directory/'progress.json',dict(namespace=r._sha(payload),rows=expected))
            with self.assertRaisesRegex(ValueError,'durable run-journal completion'):
                e._panel_predecessor(panel,e._request_document(request),ref,configs,prior_completions=(receipt,))

    def test_owned_worker_failure_retains_full_unknown_bank_without_zero_score_or_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,_,request=self.fixture(Path(temp));ref=e.Artifact(str(Path(temp).resolve()/'protocol.json'),request.protocol_sha256)
            hook=e.EvaluationHook(ref,request.declaration_sha256);completions=[];request=replace(request,progress_callback=completions.append)
            with patch.object(fp,'supervise_panel',return_value=dict(exit_code=1,stop_reason='LEARNER_TIMEOUT')):
                result=hook(request)
            self.assertEqual(result.status,'UNKNOWN');self.assertEqual(completions,[])
            report=json.loads((request.run_directory/result.report).read_bytes())
            self.assertEqual((len(report['rows']),report['known_completed'],report['unknown_cells']),(1000,0,1000))
            self.assertIsNone(report['completed_only_score']);self.assertEqual(report['score_bounds'],[0.,1.])
            self.assertEqual(report['supervisor']['stop_reason'],'LEARNER_TIMEOUT')

    def test_forged_terminal_or_foreign_cell_cannot_produce_a_watchdog_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            ref,protocol,_,request=self.fixture(Path(temp));configs=e._pairs(protocol,parse_configs=True,checkpoint=request.checkpoint_path)[500000]
            expected=e._fixed_rows(configs)
            for mutation in ('hash','terminal','seed','score','order'):
                rows=json.loads(r._json(expected));source=game(configs[0],rows[0]['seed'])
                rows[0].update(status='COMPLETE',score=1.,duration_seconds=.2,source_game=source,source_result_sha256=r._sha(source))
                if mutation=='hash':rows[0]['source_result_sha256']='f'*64
                if mutation=='terminal':rows[0]['source_game']['terminal_witness']['public_terminal_line']='|win|Unrelated'
                if mutation=='seed':rows[0]['seed']+=1
                if mutation=='score':rows[0]['score']=0.
                if mutation=='order':rows[0],rows[1]=rows[1],rows[0]
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):e._completed_receipts(rows,expected,configs)

    def test_worker_cli_constructs_only_explicit_bound_hook_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,_,_,request=self.fixture(root)
            plan=NS(evaluation=r.EvaluationProtocol(ref.sha256),manifest=lambda:json.loads((request.run_directory/'declaration.json').read_bytes()))
            inputs=NS(plan=plan,declaration_sha256='d'*64)
            argv=['--enable-v5','--arm','B0','--declaration','/fake','--expected-declaration-sha256','d'*64,
                '--runtime','/fake-runtime','--expected-runtime-sha256','e'*64,'--run-root',str(request.run_directory),
                '--evaluation-protocol',ref.path,'--expected-evaluation-protocol-sha256',ref.sha256]
            with patch.object(worker,'load_inputs',return_value=inputs),patch.object(worker,'execute',return_value=dict(status='COMPLETE_DECLARATION')) as execute,redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(argv),0)
                self.assertIs(type(execute.call_args.kwargs['evaluation_hook']),e.EvaluationHook)
                with self.assertRaisesRegex(ValueError,'override'):worker.main(argv,evaluation_hook=lambda _:None)

    def test_original_run_journals_only_valid_completed_receipts_once_and_resume_validates(self):
        import test_v5_milestone_evaluation as milestone
        harness=milestone.MilestoneEvaluationTests();config=NS(pokezero_username='Unit',foulplay_username='Peer',pokezero_player='p1')
        with patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run';base=run_fixture.plan()
            protocol=dict(schema=e.SCHEMA,model_sha256=r._sha(base.model.metadata()),panels=[dict(declared_games=count,
                seeds=list(range(101,601)),configs=[dict(seed_start=101,foulplay_random_seed=1000,pokezero_player=seat,pokezero_username='Unit',foulplay_username='Peer')
                    for seat in ('p1','p2')]) for count in (2,4)])
            ref=artifact(Path(temp)/'unit-progress-protocol.json',protocol);recipe=replace(base,evaluation=r.EvaluationProtocol(ref['sha256']))
            source=game(config,101);receipt=r.EvaluationCompletion(0,0,101,'p1',3.,r._json(source),r._sha(source),'Unit','Peer',ref['path'],1000)
            def hook(request):
                request.progress_callback(receipt)
                with self.assertRaisesRegex(ValueError,'replayed'):request.progress_callback(receipt)
                capped=dict(source,capped=True)
                with self.assertRaisesRegex(ValueError,'capped'):
                    request.progress_callback(replace(receipt,cell_ordinal=2,offset=1,source_game_json=r._json(capped),source_result_sha256=r._sha(capped)))
                return harness.result(request)
            result=harness.execute(path,recipe,run_fixture.FakeBackend(),run_fixture.FakeTraining(),hook)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION')
            store=r.Store(path,recipe,resume=True);events=store.matching('EVALUATION_GAME_END')
            self.assertEqual(len(events),2);self.assertTrue(all(x['status']=='COMPLETE' for x in events))
            self.assertTrue(all('progress_callback' not in x for x in events));r._evaluation_history(store)
            expected={k:v for k,v in events[0].items() if k not in ('completion','status','duration_seconds')}
            for change in ('seed','fp-seed','protocol','snapshot'):
                payload=json.loads(r._json(events[0]));value=r.EvaluationCompletion(**payload['completion'])
                if change=='seed':
                    wrong=json.loads(value.source_game_json);wrong['seed']=102
                    value=replace(value,seed=102,source_game_json=r._json(wrong),source_result_sha256=r._sha(wrong))
                if change=='fp-seed':value=replace(value,foulplay_seed=999)
                if change=='protocol':value=replace(value,protocol_path=str(Path(temp)/'missing-protocol.json'))
                if change=='snapshot':payload['snapshot']['model_sha256']='f'*64
                payload['completion']=asdict(value)
                with self.subTest(change=change),self.assertRaises((ValueError,OSError)):
                    r._checked_evaluation_progress(payload,expected,set())
            # A duplicate even on a valid rehashed journal cannot become new progress.
            original=next(event for event in store.events if event['kind']=='EVALUATION_GAME_END')
            store.emit('EVALUATION_GAME_END',**original['payload'])
            with self.assertRaisesRegex(ValueError,'open original attempt|replayed'):r._evaluation_history(store)

    def test_process_loss_redelivers_completed_source_without_new_game_progress(self):
        import test_v5_milestone_evaluation as milestone
        harness=milestone.MilestoneEvaluationTests()
        with patch.object(r,'EVALUATION_INTERVAL_GAMES',2),tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'run';base=run_fixture.plan()
            protocol=dict(schema=e.SCHEMA,model_sha256=r._sha(base.model.metadata()),panels=[dict(declared_games=count,
                seeds=list(range(101,601)),configs=[dict(seed_start=101,foulplay_random_seed=1000,pokezero_player=seat,
                    pokezero_username='Unit',foulplay_username='Peer')for seat in ('p1','p2')])for count in (2,4)])
            ref=artifact(Path(temp)/'protocol.json',protocol);recipe=replace(base,evaluation=r.EvaluationProtocol(ref['sha256']))
            cfg=NS(pokezero_username='Unit',foulplay_username='Peer',pokezero_player='p1')
            def receipt(offset):
                source=game(cfg,101+offset)
                return r.EvaluationCompletion(2*offset,offset,101+offset,'p1',3.,r._json(source),r._sha(source),
                    'Unit','Peer',ref['path'],1000+offset)
            def interrupted(request):request.progress_callback(receipt(0));raise KeyboardInterrupt('lost parent')
            with self.assertRaises(KeyboardInterrupt):
                harness.execute(path,recipe,run_fixture.FakeBackend(),run_fixture.FakeTraining(),interrupted)
            def continued(request):
                request.progress_callback(receipt(0))
                with self.assertRaisesRegex(ValueError,'replayed'):request.progress_callback(receipt(0))
                request.progress_callback(receipt(1));return harness.result(request)
            result=harness.execute(path,recipe,run_fixture.FakeBackend(),run_fixture.FakeTraining(),continued,resume=True)
            self.assertEqual(result['status'],'COMPLETE_DECLARATION')
            store=r.Store(path,recipe,resume=True);r._evaluation_history(store)
            self.assertEqual(len(store.matching('EVALUATION_RESUMED')),1)
            self.assertEqual(len(store.matching('EVALUATION_GAME_END',declared_games=2)),2)
            self.assertEqual(len(store.matching('EVALUATION_GAME_END',declared_games=4)),2)

    def test_outer_supervisor_advances_only_on_actual_complete_evaluation_event(self):
        for status in ('COMPLETE','REFUSED'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);base=run_fixture.plan()
                protocol=dict(schema=e.SCHEMA,model_sha256=r._sha(base.model.metadata()),panels=[dict(declared_games=2,
                    seeds=list(range(101,601)),configs=[dict(seed_start=101,foulplay_random_seed=1000,pokezero_player=seat,pokezero_username='Unit',foulplay_username='Peer')
                        for seat in ('p1','p2')])])
                ref=artifact(root/'unit-progress-protocol.json',protocol);recipe=replace(base,evaluation=r.EvaluationProtocol(ref['sha256']))
                store=r.Store(root/'run',recipe);clock=[0.];published=False
                cp=store.artifact('checkpoint.pt',b'unit-only-checkpoint',kind='CHECKPOINT',iteration=0,steps=3,rows=12)
                arc=store.artifact('archive.npz',b'unit-only-archive',kind='UNIT_ARCHIVE')
                snapshot=r.Snapshot(0,cp['path'],cp['sha256'],arc['path'],arc['sha256'],r._sha(recipe.model.metadata()))
                store.emit('SNAPSHOT',**asdict(snapshot))
                with patch.object(r,'EVALUATION_INTERVAL_GAMES',2):binding=r._evaluation_binding(store,snapshot)
                expected=dict(binding,evaluation_sha256=r._sha(binding));store.emit('EVALUATION_STARTED',**expected)
                cfg=NS(pokezero_username='Unit',foulplay_username='Peer',pokezero_player='p1');source=game(cfg,101)
                receipt=r.EvaluationCompletion(0,0,101,'p1',3.,r._json(source),r._sha(source),'Unit','Peer',ref['path'],1000)
                class Child:
                    pid=17901;returncode=None
                    def poll(self):
                        if clock[0]>=16:self.returncode=0
                        return self.returncode
                child=Child()
                def sleep(seconds):
                    nonlocal published
                    clock[0]+=seconds
                    if not published and clock[0]>=2:
                        store.emit('EVALUATION_GAME_END',**expected,status=status,duration_seconds=3.,completion=asdict(receipt));published=True
                def stop(process,*args,**kwargs):
                    if process.returncode is None:process.returncode=-15
                with patch.object(r,'EVALUATION_INTERVAL_GAMES',2),patch.object(r.time,'monotonic',side_effect=lambda:clock[0]),patch.object(r,'_stop_owned_group',side_effect=stop):
                    report=r.supervise_owned(['fake-never-spawned'],plan=recipe,path=store.path,popen=lambda *a,**k:child,sleep=sleep)
                if status=='COMPLETE':self.assertEqual((report['exit_code'],report['stop_reason']),(0,None))
                else:self.assertIn(report['stop_reason'],('PROGRESS_JOURNAL_CHANGED','NO_DURABLE_WORK_PROGRESS'))

    def test_affinity_covers_preexisting_native_threads_with_no_real_placement(self):
        masks={11:{0,1,2},12:{0,1,2}};changes=[]
        def get(tid):return {0,1,2} if tid==0 else masks[tid]
        def put(tid,cpus):masks[tid]=set(cpus);changes.append(tid)
        with patch.object(e.os,'sched_getaffinity',side_effect=get,create=True),patch.object(e.os,'sched_setaffinity',side_effect=put,create=True),\
                patch.object(e.Path,'iterdir',return_value=iter((Path('/proc/self/task/11'),Path('/proc/self/task/12')))):
            #A fresh iterator is needed for the independent postcheck.
            with patch.object(e.Path,'iterdir',side_effect=lambda:iter((Path('/proc/self/task/11'),Path('/proc/self/task/12')))):
                e._pin_actor(dict(actor_cpus=[0],foulplay_cpus=[1,2]))
        self.assertEqual(changes,[11,12]);self.assertEqual(masks,{11:{0},12:{0}})

    def test_owned_child_binds_policy_before_typed_runtime_and_never_uses_unreviewed_executor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,_,request=self.fixture(root)
            directory=request.run_directory/'artifacts'/('evaluation-500000-'+r._sha(e._request_document(request)))/'attempt-000000'
            directory.mkdir(parents=True)
            payload=dict(protocol=asdict(ref),expected_run_declaration_sha256=request.declaration_sha256,
                request=e._request_document(request),directory=str(directory))
            child_ref=e.Artifact(**artifact(directory/'request.json',payload));order=[]
            original=e._protocol
            def parse(*a,**k):order.append('typed-config');return original(*a,**k)
            async def fake(configs,*args,**kwargs):
                rows=e._fixed_rows(configs)
                for value in rows:value.update(status='REFUSED',reason='fake-unknown')
                return dict(rows=rows)
            with patch.object(e,'_pin_actor',side_effect=lambda _:order.append('affinity')),\
                    patch.object(worker,'_memory',side_effect=lambda *a,**kw:order.append('memory') if kw.get('apply') else None),\
                    patch.object(worker,'_source'),patch.object(e,'_protocol',side_effect=parse),\
                    patch.object(e,'_factory',return_value=object()) as factory,patch.object(e,'mechanism',return_value={'fake_only':True}),\
                    patch.object(e,'_concurrent_panel',side_effect=fake),\
                    patch.object(e,'_require_foulplay_source',side_effect=lambda *a:order.append('fp-source')) as fp_source:
                self.assertEqual(e.child(child_ref),0)
            self.assertEqual(order[:3],['affinity','memory','typed-config']);factory.assert_called_once()
            fp_source.assert_called_once()
            report=json.loads((directory/'worker-report.json').read_bytes())
            self.assertEqual((report['known_completed'],report['unknown_cells'],report['score_bounds']),(0,1000,[0.,1.]))
            self.assertFalse(report['scientific_qualification'])


if __name__=='__main__':unittest.main()
