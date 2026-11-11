"""Fake endpoint source/terminal records only; no model, game, bank or process."""
import asyncio
from dataclasses import asdict,replace
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock,Mock,patch

from pokezero import v5_endpoint as endpoint,v5_evaluation as evaluation,v5_foulplay as fp,v5_run as run,v5_worker as worker
from pokezero import foulplay_bridge as old
from pokezero.live_protocol_actor import ActorContract
from pokezero.v5_model import V5ModelConfig
from test_v5_contract import vocabulary


def artifact(path,value):
    path=path.resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else run._json(value).encode())
    return dict(path=str(path),sha256=run._file_sha(path))


def fixture(root):
    root=root.resolve();(root/'show').mkdir();(root/'fp').mkdir()
    interpreter=artifact(root/'python',b'UNIT-NOT-EXECUTABLE');node=artifact(root/'node',b'UNIT-NOT-EXECUTABLE')
    vocab=dict(tokens=['unit'+str(i) for i in range(111)],oov_buckets=16,aliases={});category=run._sha(vocab)
    vocab_ref=artifact(root/'vocabulary.json',vocab)
    runtime=dict(schema=worker.SCHEMA,category_vocabulary=vocab_ref,
        memory_policy=dict(environment=dict(zip(worker.ENV_KEYS,('false',None,None))),torch_fraction=None),
        precision_policy=dict(inference='bfloat16'),showdown=dict(root=str(root/'show'),node=node))
    runtime_ref=artifact(root/'runtime.json',runtime);arms=[]
    for name in endpoint.ARMS:
        directory=root/name;directory.mkdir();model=V5ModelConfig(128,category,vocabulary(),arm=name[0],replicate=int(name[1]))
        declaration=dict(schema=run.SCHEMA,model=model.metadata(),iterations=[[dict(games=[dict(seed=1)])]])
        declaration_ref=artifact(directory/'declaration.json',declaration)
        checkpoint=artifact(directory/'artifacts'/'final.pt',b'UNIT-NOT-A-CHECKPOINT-'+name.encode())
        archive=artifact(directory/'artifacts'/'final.npz',b'UNIT-NOT-AN-ACTOR-'+name.encode())
        snapshot=run.Snapshot(0,'artifacts/final.pt',checkpoint['sha256'],'artifacts/final.npz',archive['sha256'],run._sha(model.metadata()))
        previous=None;final_events={}
        for index,(kind,payload) in enumerate((('RUN_STARTED',dict(declaration_sha256=run._sha(declaration))),
                ('CHECKPOINT',dict(iteration=0,path=snapshot.checkpoint,sha256=snapshot.checkpoint_sha256,steps=1,rows=1)),
                ('SNAPSHOT',asdict(snapshot)),('ITERATION_COMPLETE',dict(iteration=0,snapshot=asdict(snapshot),
                    counts=dict(NOT_ATTEMPTED=0,RUNNING=0,COMPLETE=1,CAPPED=0,REFUSED=0,FAILED=0))))):
            event=dict(sequence=index,previous=previous,time=1.,kind=kind,payload=payload);event['sha256']=run._sha(event)
            reference=artifact(directory/'journal'/f'{index:09d}.json',event);previous=event['sha256']
            if kind!='RUN_STARTED':final_events[kind.lower()]=reference
        configs=[]
        for seat in ('p1','p2'):
            config=old.ControlledFoulPlayConfig(checkpoint=Path(checkpoint['path']),showdown_root=root/'show',node_binary=node['path'],
                foulplay_root=root/'fp',foulplay_python=Path(interpreter['path']),games=1000,seed_start=100,
                foulplay_random_seed=5000,policy_mode='raw',device='cpu',encoder_kind='jax-owning',belief_set_source=True,
                allow_search_fallback=False,opponent_crash_retries=0,pokezero_player=seat,owning_decision_timeout_seconds=60.)
            configs.append(json.loads(json.dumps(asdict(config),default=fp._json_default)))
        arms.append(dict(name=name,run_directory=str(directory),declaration=declaration_ref,final_events=final_events,
            snapshot=asdict(snapshot),runtime=runtime_ref,configs=configs))
    protocol=dict(schema=endpoint.SCHEMA,seeds=list(range(100,1100)),arms=arms,worker_python=interpreter,
        foulplay_python=interpreter,foulplay_commit='f'*40,
        foulplay_source=artifact(root/'foulplay-component.json',{'unit_metadata_only':True}),
        actor_cpus=[0],foulplay_cpus=[1,2,3,4],
        watchdog=asdict(run.Watchdog(10,backstop_seconds=1000)),bootstrap=dict(method=endpoint.METHOD,replicates=10000,seed=31,confidence=.9),
        deterministic=True,core_schedule=endpoint.CORE_SCHEDULE)
    reference=evaluation.Artifact(**artifact(root/'recipe.json',protocol))
    return reference,*endpoint._protocol(reference,parse_configs=True)


def source_result(row,protocol,configs,score=1.,*,status='COMPLETE',reason=None):
    config=endpoint._config_for(row,configs);document=json.loads(json.dumps(asdict(config),default=fp._json_default))
    arm=next(a for a in protocol['arms'] if a['name']==row['arm']);snapshot=arm['snapshot']
    model=configs.models[row['arm']]
    contract=replace(ActorContract.champion(),checkpoint_sha256=snapshot['checkpoint_sha256'],model_config_sha256=snapshot['model_sha256'])
    recipe=dict(schema=fp.SCHEMA,actor_version='v5',config=document,actor_namespace=contract.namespace,
        archive_sha256=snapshot['archive_sha256'],checkpoint_sha256=snapshot['checkpoint_sha256'],
        category_vocabulary_sha256=model.category_vocabulary_sha256,label_vocabulary_sha256=model.label_vocabulary.sha256,
        deterministic=True,inference_precision='bfloat16',learner_seconds=60.,game_offsets=[row['offset']])
    source=None
    if status in ('COMPLETE','CAPPED'):
        tied=score==.5;won=score==1.;winner=None if tied else config.pokezero_username if won else config.foulplay_username
        witness=(dict(status='PUBLIC_TIE',winner_player=None,public_terminal_line='|tie|') if tied else
            dict(status='PUBLIC_WIN',winner_player=config.pokezero_player if won else ('p2' if config.pokezero_player=='p1' else 'p1'),public_terminal_line='|win|'+winner))
        source=dict(seed=row['seed'],capped=status=='CAPPED',tied=tied,winner=winner,pokezero_won=won,terminal_witness=witness)
    actual=dict(offset=row['offset'],seed=row['seed'],fp_seed=row['fp_seed'],status=status,score=score if status=='COMPLETE' else None,
        duration_seconds=.2,reason=reason,source_result_sha256=None if source is None else run._sha(source),source_game=source,invocation_id='unit-inner')
    return dict(row=actual,recipe=recipe,namespace=fp._sha(recipe),invocation_id='unit-inner')


def complete_rows(protocol,configs,score):
    rows=endpoint._rows(protocol,configs)
    for row in rows:
        value=score(row);row.update(status='COMPLETE',result=source_result(row,protocol,configs,value),invocation_id='unit-outer')
    return rows


class EndpointTests(unittest.TestCase):
    def setUp(self):
        # UNIT-only tiny fake manifests, not a smaller production budget or any actual training.
        self.assertEqual(endpoint.TRAINING_GAMES,2_000_000)
        budget=patch.object(endpoint,'TRAINING_GAMES',1);budget.start();self.addCleanup(budget.stop)

    def test_exact_four_final_snapshots_source_and_separate_2000_cell_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);rows=endpoint._rows(protocol,configs)
            self.assertEqual(len(rows),8000)
            self.assertEqual([(r['arm'],r['seat'],r['offset']) for r in rows[:8]],[(a,s,0) for s in ('p1','p2') for a in endpoint.ARMS])
            self.assertEqual([sum(r['arm']==a for r in rows) for a in endpoint.ARMS],[2000]*4)
            for i in range(0,8000,4):
                self.assertEqual(len({(r['seed'],r['fp_seed'],r['seat']) for r in rows[i:i+4]}),1)
                self.assertEqual({r['slot'] for r in rows[i:i+4]},{0,1,2,3})
            self.assertEqual([r['slot'] for r in rows[8:12]],[1,2,3,0])
            three=endpoint._rows(dict(protocol,foulplay_cpus=[1,2,3]),configs)
            self.assertEqual([three[i*8]['slot'] for i in range(3)],[0,1,2])
            for kind in ('seeds','order','snapshot','head','configuration','budget','source'):
                changed=json.loads(run._json(protocol))
                if kind=='seeds':changed['seeds'].pop()
                if kind=='order':changed['arms'].reverse()
                if kind=='snapshot':changed['arms'][0]['snapshot']['iteration']=1
                if kind=='head':changed['arms'][1]['final_events']['snapshot']['sha256']='a'*64
                if kind=='configuration':changed['arms'][1]['configs'][0]['max_decision_rounds']+=1
                if kind=='budget':changed['arms'][0]['configs'][0]['owning_decision_timeout_seconds']=61.
                if kind=='source':changed['arms'][0]['snapshot']['archive_sha256']='b'*64
                bad=evaluation.Artifact(**artifact(root/(kind+'.json'),changed))
                with self.subTest(kind=kind),self.assertRaises((ValueError,run.MissingDependency)):endpoint._protocol(bad)
            with patch.object(evaluation.Artifact,'document') as access:
                with self.assertRaises(ValueError):endpoint.run_endpoint(ref,root/'disabled')
                access.assert_not_called()
            copied=NS(document=Mock(return_value=protocol))
            with self.assertRaises(ValueError):endpoint._protocol(copied)
            copied.document.assert_not_called()
            # Original milestone remains exactly500 seeds ×two seats.
            self.assertEqual(len(evaluation._fixed_rows(tuple(replace(c,games=500) for c in configs['A0']))),1000)

    def test_three_publication_receipts_reject_nonterminal_counts_and_cache_original_model_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);row=endpoint._rows(protocol,configs)[0]
            result=source_result(row,protocol,configs)
            # Per-cell validation must not reparse the multi-million-game Plan.
            with patch.object(evaluation.Artifact,'document',side_effect=AssertionError('per-cell declaration read')):
                endpoint._checked_result(row,result,configs,protocol)
            with self.assertRaises(TypeError):configs.models['A0']=configs.models['B0']
            changed=json.loads(run._json(protocol));event_ref=changed['arms'][0]['final_events']['iteration_complete']
            event=evaluation.Artifact(**event_ref).document();event['payload']['counts'].update(COMPLETE=0,RUNNING=1)
            event.pop('sha256');event['sha256']=run._sha(event)
            changed['arms'][0]['final_events']['iteration_complete']=artifact(Path(event_ref['path']),event)
            with self.assertRaisesRegex(ValueError,'terminal2M'):endpoint._protocol(evaluation.Artifact(**artifact(root/'running-recipe.json',changed)))

    def test_durable_original_attempt_and_terminal_survive_resume_without_replay_or_drop(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);directory=root/'endpoint'
            with endpoint._ownership(directory):
                store=endpoint.Store(directory,ref,protocol,configs)
                with self.assertRaises(ValueError):store.emit('START',1,'first')
                store.emit('START',0,'first');store.emit('END',0,'first',result=source_result(store.rows[0],protocol,configs,.5))
                store.emit('START',1,'first') # Crash after dispatch, no result exists.
            resumed=endpoint.Store(directory,ref,protocol,configs,resume=True)
            resumed.interrupt_older('second')
            self.assertEqual([r['status'] for r in resumed.rows[:3]],['COMPLETE','INTERRUPTED','NOT_ATTEMPTED'])
            self.assertEqual(resumed.rows[0]['result']['row']['score'],.5)
            with self.assertRaises(ValueError):resumed.emit('START',0,'second')
            resumed.emit('START',2,'second')
            journal=directory/'journal'/'000000001.json';blob=journal.read_bytes();journal.chmod(0o644)
            changed=json.loads(blob);changed['result']['row']['score']=0.;journal.write_text(run._json(changed))
            with self.assertRaises(ValueError):resumed.refresh()

    def test_literal_unknown_caps_namespace_and_false_terminal_do_not_become_zero_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);row=endpoint._rows(protocol,configs)[0]
            for score in (0.,.5,1.):self.assertEqual(endpoint._checked_result(row,source_result(row,protocol,configs,score),configs,protocol)['score'],score)
            capped=source_result(row,protocol,configs,.5,status='CAPPED')
            self.assertIsNone(endpoint._checked_result(row,capped,configs,protocol)['score'])
            for mutation in ('caps','winner','hash','archive','namespace','core','seed'):
                result=source_result(row,protocol,configs,1.)
                if mutation=='caps':result['row']['source_game']['capped']=True
                if mutation=='winner':result['row']['source_game']['terminal_witness']['winner_player']='p2'
                if mutation=='hash':result['row']['source_result_sha256']='f'*64
                if mutation=='archive':result['recipe']['archive_sha256']='f'*64
                if mutation=='namespace':result['recipe']['actor_namespace']='f'*64
                if mutation=='core':result['recipe']['config']['foulplay_cpu_affinity']=99
                if mutation=='seed':result['row']['seed']+=1
                result['namespace']=fp._sha(result['recipe'])
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):endpoint._checked_result(row,result,configs,protocol)

    def test_fake_original_runner_fixed_first_cells_cores_and_timeout_stops_without_refill(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);store=endpoint.Store(root/'endpoint',ref,protocol,configs)
            factories={a:object() for a in endpoint.ARMS};calls=[];active=set();peak=0
            async def run_panel(config,factory,**kwargs):
                nonlocal peak
                arm=next(a for a,f in factories.items() if f is factory);offset=kwargs['game_offsets'][0]
                row=next(r for r in store.rows if r['arm']==arm and r['offset']==offset and r['seat']==config.pokezero_player)
                self.assertEqual(row['status'],'RUNNING') # Durable before original dispatch.
                self.assertNotIn(config.foulplay_cpu_affinity,active);active.add(config.foulplay_cpu_affinity);peak=max(peak,len(active))
                calls.append(row['ordinal']);await asyncio.sleep(0)
                result=source_result(row,protocol,configs,status='FAILED',reason='TimeoutError: unit-owned-worker')
                value=dict(rows=[result['row']],recipe=result['recipe'],namespace=result['namespace'],invocation_id=result['invocation_id'])
                kwargs['progress_callback'](value);active.remove(config.foulplay_cpu_affinity);return value
            with patch.object(fp,'_recipe',return_value={}),patch.object(fp,'run_panel',side_effect=run_panel):
                report=asyncio.run(endpoint.execute_cells(store,factories,invocation='first'))
            self.assertEqual(calls,[0,1,2,3]);self.assertEqual(peak,4)
            self.assertEqual(report['counts']['FAILED'],4);self.assertEqual(report['counts']['NOT_ATTEMPTED'],7996)
            self.assertTrue(all(r['result'] is None for r in report['rows'][4:]))

    def test_seed_blocks_keep_both_seats_and_b0_is_shared_not_pooled_with_replica(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,protocol,configs=fixture(root)
            def score(row):
                if row['arm']=='A0':return .5
                if row['arm']=='B0':return 1. if row['seat']=='p1' else 0.
                if row['arm']=='D0':return 1.
                return .5
            result=endpoint.reduce_endpoint(protocol,complete_rows(protocol,configs,score),configs)
            self.assertEqual(result['contrasts']['B0-A0']['paired_seed_bootstrap_90'],[0.,0.])
            self.assertEqual(result['contrasts']['D0-B0']['paired_seed_bootstrap_90'],[.5,.5])
            self.assertEqual(result['contrasts']['B1-B0']['paired_seed_bootstrap_90'],[0.,0.])
            self.assertTrue(result['contrasts']['D0-B0']['supported_positive_primary'])
            self.assertFalse(result['contrasts']['B0-A0']['equivalence_established'])
            self.assertEqual(result['arms']['B0']['known_completed'],2000)
            self.assertEqual(result['declared_cells'],8000)

    def test_unknown_full_roster_bounds_cannot_promote_complete_case_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,protocol,configs=fixture(root);rows=endpoint._rows(protocol,configs)
            # Positive observed cells, but most original cells remain unknown.
            for row in rows:
                if row['offset']==0:
                    score=1. if row['arm'] in ('B0','D0') else 0.
                    row.update(status='COMPLETE',result=source_result(row,protocol,configs,score),invocation_id='unit')
            result=endpoint.reduce_endpoint(protocol,rows,configs);contrast=result['contrasts']['B0-A0']
            self.assertEqual(contrast['complete_seed_point'],1.)
            self.assertEqual(contrast['full_roster_identification_bounds'],[-.998,1.])
            self.assertEqual((contrast['complete_seed_blocks'],contrast['unknown_cells_across_arms']),(1,3996))
            self.assertFalse(contrast['supported_positive_primary']);self.assertLess(contrast['paired_seed_bootstrap_90'][0],0)
            self.assertEqual(result['arms']['A0']['unknown_cells'],1998)
            self.assertIn('not confidence intervals',result['interpretation'])

    def test_parent_verified_progress_counts_only_literal_completion_and_uses_original_supervisor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);directory=root/'endpoint'
            def supervisor(command,**kwargs):
                self.assertIn('pokezero.v5_endpoint',command)
                args=json.loads(Path(command[command.index('--request')+1]).read_bytes())
                store=endpoint.Store(directory,ref,protocol,configs,resume=True)
                store.emit('START',0,args['invocation']);store.emit('END',0,args['invocation'],result=source_result(store.rows[0],protocol,configs))
                kwargs['sleep'](0)
                verified=json.loads(Path(kwargs['progress_path']).read_bytes())
                self.assertEqual(sum(r['status']=='COMPLETE' for r in verified['rows']),1)
                return dict(exit_code=1,stop_reason='NO_COMPLETED_GAME_WITHIN_ADAPTIVE_WINDOW')
            with patch.object(evaluation,'_pin_actor'),patch.object(fp,'supervise_panel',side_effect=supervisor),patch.object(endpoint,'reduce_endpoint',return_value={'UNIT':'no metric execution'}):
                result=endpoint.run_endpoint(ref,directory,enable_v5_endpoint=True)
            self.assertEqual(result['status'],'UNKNOWN')
            report=json.loads(Path(result['report']).read_bytes())
            self.assertEqual(report['counts']['COMPLETE'],1);self.assertEqual(report['counts']['NOT_ATTEMPTED'],7999)
            self.assertEqual(report['supervisor']['stop_reason'],'NO_COMPLETED_GAME_WITHIN_ADAPTIVE_WINDOW')

    def test_fake_sigterm_waits_for_all_four_runner_finalizers_and_restores_prior_async_handler(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);store=endpoint.Store(root/'endpoint',ref,protocol,configs)
            factories={a:object() for a in endpoint.ARMS};started=[];closed=[];handlers={};prior=Mock();restored=[]
            async def exercise():
                loop=asyncio.get_running_loop()
                async def panel(config,factory,**kwargs):
                    arm=next(a for a,f in factories.items() if f is factory);started.append(arm)
                    try:
                        if len(started)==4:
                            handlers[signal.SIGTERM]()
                            handlers[signal.SIGTERM]() # Repeated TERM cannot cancel cleanup a second time.
                        await asyncio.Event().wait()
                    finally:
                        await asyncio.sleep(0);closed.append(arm)
                def add(sig,callback,*args):
                    handlers[sig]=callback
                    if callback is prior:restored.append(args)
                existing=getattr(loop,'_signal_handlers',{})
                with patch.dict(existing,{signal.SIGTERM:NS(_callback=prior,_args=('original',))}),\
                        patch.object(loop,'add_signal_handler',side_effect=add),patch.object(loop,'remove_signal_handler'),\
                        patch.object(fp,'_recipe',return_value={}),patch.object(fp,'run_panel',side_effect=panel):
                    with self.assertRaises(asyncio.CancelledError):
                        await endpoint.execute_cells(store,factories,invocation='cancelled')
                    self.assertEqual(len(closed),4)
            asyncio.run(exercise())
            self.assertEqual(started,list(endpoint.ARMS));self.assertEqual(set(closed),set(endpoint.ARMS));self.assertEqual(restored,[('original',)])
            resumed=endpoint.Store(root/'endpoint',ref,protocol,configs,resume=True);resumed.interrupt_older('next')
            self.assertEqual([r['status'] for r in resumed.rows[:5]],['INTERRUPTED']*4+['NOT_ATTEMPTED'])

    def test_fake_parent_sigterm_runs_original_owned_group_cleanup_and_restores_handler(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);handlers={};previous=object();alive=[True];signals=[]
            class Process:
                pid=987654321;returncode=None
                def poll(self):return self.returncode
                def wait(self,timeout):return self.returncode
                def kill(self):self.returncode=-9;alive[0]=False
            process=Process();original=fp.supervise_panel
            def install(sig,handler):handlers[sig]=handler
            def send(group,sig):signals.append((group,sig));alive[0]=False;process.returncode=-15
            def supervise(command,**kwargs):
                def interrupted_sleep(seconds):handlers[signal.SIGTERM](signal.SIGTERM,None)
                return original(command,**dict(kwargs,sleep=interrupted_sleep),popen=lambda *a,**k:process,
                    group_exists=lambda group:alive[0],signal_group=send)
            with patch.object(endpoint.signal,'getsignal',side_effect=lambda sig:handlers.get(sig,previous)),patch.object(endpoint.signal,'signal',side_effect=install),\
                    patch.object(evaluation,'_pin_actor'),patch.object(fp,'supervise_panel',side_effect=supervise):
                with self.assertRaises(KeyboardInterrupt):endpoint.run_endpoint(ref,root/'endpoint',enable_v5_endpoint=True)
            self.assertFalse(alive[0]);self.assertEqual(signals,[(process.pid,signal.SIGTERM)])
            self.assertIs(handlers[signal.SIGTERM],previous)

    def test_sync_owner_does_not_replace_a_newer_parent_handler(self):
        previous=object();newer=object();handlers={signal.SIGTERM:previous}
        def install(sig,handler):handlers[sig]=handler
        def body(*args,**kwargs):
            handlers[signal.SIGTERM]=newer
            return 'done'
        with patch.object(endpoint.signal,'getsignal',side_effect=lambda sig:handlers[sig]), \
                patch.object(endpoint.signal,'signal',side_effect=install),patch.object(endpoint,'_run_endpoint',side_effect=body):
            self.assertEqual(endpoint.run_endpoint(None,None,enable_v5_endpoint=True),'done')
        self.assertIs(handlers[signal.SIGTERM],newer)

    def test_cancelled_original_probe_kills_owned_group_even_after_leader_exit_and_reraises(self):
        process=NS(pid=987654322,returncode=0,communicate=AsyncMock(side_effect=asyncio.CancelledError),
            wait=AsyncMock(return_value=0),kill=Mock());alive=[True];sent=[]
        def killpg(group,sig):
            self.assertEqual(group,process.pid);sent.append(sig)
            if sig==signal.SIGKILL:alive[0]=False
            elif not alive[0]:raise ProcessLookupError
        config=NS(resolved_foulplay_python='/unit/not-executed')
        with patch.object(old.asyncio,'create_subprocess_exec',new_callable=AsyncMock,return_value=process),\
                patch.object(old.os,'killpg',side_effect=killpg),patch.object(old.os,'getpgid',side_effect=AssertionError('exited leader lookup')):
            with self.assertRaises(asyncio.CancelledError):asyncio.run(old._probe_foulplay_start_method(config))
        self.assertFalse(alive[0]);self.assertEqual(sent,[signal.SIGKILL,0]);process.wait.assert_awaited_once()

    def test_normal_probe_unchanged_and_persistent_cancelled_group_is_a_hard_failure(self):
        config=NS(resolved_foulplay_python='/unit/not-executed')
        process=NS(pid=987654323,returncode=0,communicate=AsyncMock(return_value=(b'POKEZERO_START_METHOD=fork\nPOKEZERO_PYTHON=3.13\n',b'')))
        with patch.object(old.asyncio,'create_subprocess_exec',new_callable=AsyncMock,return_value=process),patch.object(old.os,'killpg') as kill:
            result=asyncio.run(old._probe_foulplay_start_method(config))
            self.assertEqual(result['status'],'probed');self.assertTrue(result['iterations_observable']);kill.assert_not_called()
        process.communicate=AsyncMock(side_effect=asyncio.CancelledError);process.wait=AsyncMock(return_value=0);process.kill=Mock()
        # Patch this MODULE's clock object; never replace asyncio's global clock.
        clock=NS(monotonic=Mock(side_effect=[0.,2.]))
        with patch.object(old.asyncio,'create_subprocess_exec',new_callable=AsyncMock,return_value=process),\
                patch.object(old.os,'killpg'),patch.object(old,'time',clock):
            with self.assertRaisesRegex(RuntimeError,str(process.pid)):asyncio.run(old._probe_foulplay_start_method(config))

    def test_cell_cleanup_failure_propagates_instead_of_claiming_successful_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,protocol,configs=fixture(root);store=endpoint.Store(root/'endpoint',ref,protocol,configs)
            factories={a:object() for a in endpoint.ARMS};handlers={};count=[0]
            async def exercise():
                loop=asyncio.get_running_loop()
                async def panel(*args,**kwargs):
                    count[0]+=1
                    try:
                        if count[0]==4:handlers[signal.SIGTERM]()
                        await asyncio.Event().wait()
                    finally:raise RuntimeError('UNIT owned group still present')
                with patch.object(loop,'add_signal_handler',side_effect=lambda sig,fn,*a:handlers.update({sig:fn})),\
                        patch.object(loop,'remove_signal_handler'),patch.object(endpoint.signal,'signal'),\
                        patch.object(fp,'_recipe',return_value={}),patch.object(fp,'run_panel',side_effect=panel):
                    with self.assertRaisesRegex(RuntimeError,'cell cleanup failed'):
                        await endpoint.execute_cells(store,factories,invocation='cancelled')
            asyncio.run(exercise());self.assertEqual(count[0],4)


if __name__=='__main__':unittest.main()
