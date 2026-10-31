"""Fake orchestration and literal tiny trajectory fixtures; no model/game calls."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import b4_fidelity_canary as c,neural_policy as neural,dataset


def training():
    return neural.TransformerTrainingConfig(batch_size=512,epochs=5,learning_rate=5.9e-5,
        learning_rate_schedule='mit-thesis',learning_rate_schedule_total_games=15000000,
        learning_rate_progress_start=.999893333333,learning_rate_progress_end=1.,discount=.9999,
        capped_terminal_value=-.25,value_loss_weight=.4375,value_clip_range=.0184,objective='ppo',
        clip_epsilon=.0829,entropy_coef=.0147,ppo_target_mode='gae',gae_lambda=.754,max_grad_norm=.543,
        amp='bf16',batch_replay=True)


def model_config():
    return neural.TransformerPolicyConfig(policy_id='unit-original9375',window_size=1,categorical_vocab_size=916,
        token_type_vocab_size=16,categorical_feature_count=41,numeric_feature_count=132,token_count=23,
        embedding_dim=512,transformer_layers=3,attention_heads=8,feedforward_dim=2048,dropout=.1,
        observation_schema_version='pokezero.observation.v4',value_activation='tanh',transition_token_budget=0,
        category_vocab=tuple('unit%04d'%i for i in range(899)),category_oov_buckets=16)


def plan():
    batches=[]
    for offset in (0,2):
        games=tuple(c.Game(i+1,'unit-'+str(i),100+i,(2*i,2*i+1)) for i in range(offset,offset+2))
        batches.append(c.Batch(games,1000+offset,2000+offset,3000+offset,4000,tuple(5000+i for i in range(offset,offset+2))))
    configs=tuple(dict(checkpoint='/untouched',showdown_root='/unit-showdown',games=2,seed_start=6000,
        foulplay_random_seed=7000,pokezero_player=side,policy_mode='raw',search_time_ms=1000,
        pokezero_username='Learner',foulplay_username='Peer') for side in ('p1','p2'))
    return c.Plan(tuple((b,) for b in batches),'a'*64,'b'*40,'c'*64,1024,configs)


class CanaryTests(unittest.TestCase):
    def setUp(self):
        self.a=patch.object(c,'DECLARED_GAMES',4);self.b=patch.object(c,'PANEL_SEEDS',2)
        self.a.start();self.b.start();self.addCleanup(self.a.stop);self.addCleanup(self.b.stop)

    def test_original_control_config_and_fresh_declared_schedule_only(self):
        original=training();actual=c.controls(original.to_dict(),start=0,end=2,device='cpu')
        allowed={'learning_rate_schedule_total_games','learning_rate_progress_start','learning_rate_progress_end','device'}
        differences={k for k,v in original.to_dict().items() if actual.to_dict().get(k)!=v}
        self.assertEqual(differences,allowed)
        self.assertEqual((actual.learning_rate_progress_start,actual.learning_rate_progress_end),(0.,.5))
        self.assertEqual(c.dataset_controls(actual).capped_terminal_value,-.25)
        for key,value in [('gae_lambda',.95),('value_clip_range',None),('entropy_coef',.0441),('discount',1.),
                          ('turn_penalty',.01),('opponent_action_loss_weight',0.)]:
            candidate=original.to_dict();candidate[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.controls(candidate,start=0,end=4,device='cpu')

    def test_fixed_original_banks_no_refill_and_actual_digest_journal(self):
        recipe=plan()
        with self.assertRaisesRegex(ValueError,'team reuse'):replace(recipe,iterations=((recipe.iterations[0][0],),(replace(recipe.iterations[1][0],
            games=(replace(recipe.iterations[1][0].games[0],team_indices=(0,1)),recipe.iterations[1][0].games[1])),)))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';book=c.Journal(root,recipe);book.emit('UNIT',actual=1)
            self.assertEqual(c.Journal(root,recipe,resume=True).matching('UNIT'),[{'actual':1}])
            (root/'journal.jsonl').write_text((root/'journal.jsonl').read_text().replace('"actual":1','"actual":2'))
            with self.assertRaisesRegex(ValueError,'journal'):c.Journal(root,recipe,resume=True)

    def test_full_declaration_loader_keeps_original_cells_and_rejects_extra_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json';recipe=plan();path.write_bytes(c.encoded(recipe.manifest()))
            actual=c.load_plan(path,expected_sha256=c.file_sha(path))
            self.assertEqual(actual.manifest(),recipe.manifest())
            extra=recipe.manifest();extra['silent_refill']=True;path.write_bytes(c.encoded(extra))
            with self.assertRaisesRegex(ValueError,'controls'):c.load_plan(path,expected_sha256=c.file_sha(path))

    def test_stall_clock_counts_completed_games_not_refusals_or_output(self):
        clock=[0.]
        with patch.object(c.time,'monotonic',side_effect=lambda:clock[0]):
            progress=c.Progress(plan());clock[0]=1801.
            with self.assertRaises(TimeoutError):progress.check()
            progress.completed(600.);clock[0]=4800.;progress.check();clock[0]=4802.
            with self.assertRaises(TimeoutError):progress.check()

    def _run(self,root,recipe,*,resume=False,collect_calls=None,train_calls=None):
        import gpu_engine.b4_canary_collection as engine
        import gpu_engine.champion as champion
        calls=[] if collect_calls is None else collect_calls;updates=[] if train_calls is None else train_calls
        cfg=model_config();payload=dict(schema_version=neural.neural_policy_schema_version_for_model_config(cfg),
            model_config=cfg.to_dict(),training_config=training().to_dict(),state_dict={'literal':'original-state'})
        class FakeModel:
            def __init__(self,config):self.config=config
            def load_state_dict(self,state,strict):
                if strict is not True or state!={'literal':'original-state'}:raise AssertionError('strict original weights')
        class Backend:
            def __init__(self,*args,**kwargs):pass
            def collect(self,batch,current,path,**kwargs):
                calls.append(tuple(g.stream_id for g in batch.games));path.mkdir();cache=path/'cache';cache.mkdir();(cache/'metadata.json').write_text('{}')
                for game in batch.games:kwargs['on_end'](game.stream_id,'COMPLETE',2.,'d'*64)
                return dict(cache=str(cache),cache_metadata_sha256=c.file_sha(cache/'metadata.json'),games=len(batch.games),complete=len(batch.games),unknown=0)
        def train(paths,**kwargs):
            updates.append(kwargs['training_config']);result=NS(epochs=(NS(),),model_config=cfg)
            kwargs['epoch_callback'](None,result);return kwargs['initial_model'],result
        real_sha=c.file_sha
        def sha(path):return c.CHAMPION_SHA if str(path)=='/untouched' else real_sha(path)
        def save(path,*args,**kwargs):Path(path).write_bytes(b'unit-checkpoint-not-a-model')
        def export(cp,path):Path(path).write_bytes(b'unit-archive-not-weights')
        with patch.object(c,'file_sha',side_effect=sha),patch.object(c,'checkpoint_payload',return_value=payload),\
             patch.object(neural,'EntityTokenTransformerPolicy',FakeModel),patch.object(neural,'train_transformer_policy',side_effect=train),\
             patch.object(neural,'save_transformer_checkpoint',side_effect=save),patch.object(champion,'export',side_effect=export),patch.object(engine,'Backend',Backend):
            result=c.run(recipe,root,checkpoint='/untouched',prepared=object(),pool=object(),gender_policy=object(),device='cpu',resume=resume,enable_canary=True)
        return result,calls,updates

    def test_original_trainer_strict_weights_final_only_schedule_and_resume_no_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run';recipe=plan();result,calls,updates=self._run(root,recipe)
            self.assertEqual(calls,[(1,2),(3,4)])
            self.assertEqual([(x.learning_rate_progress_start,x.learning_rate_progress_end) for x in updates],[(0.,.5),(.5,1.)])
            self.assertEqual(result['status'],'FINAL_CHECKPOINT_NOT_EVALUATED')
            result,again,trained=self._run(root,recipe,resume=True)
            self.assertEqual((again,trained),([],[]))
            self.assertEqual(len(c.Journal(root,recipe,resume=True).matching('CANARY_TRAINING_END')),1)

    def test_interrupted_collection_never_reconstructs_attempted_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe=plan();root=Path(temp)/'run';book=c.Journal(root,recipe);book.emit('COLLECT_START',iteration=0,batch=0,streams=[1,2])
            result,calls,trained=self._run(root,recipe,resume=True)
            self.assertEqual((result['status'],calls,trained),('UNKNOWN_NO_COMPLETE_CACHE',[],[]))
            self.assertEqual(c.Journal(root,recipe,resume=True).matching('COLLECT_END')[0]['status'],'INTERRUPTED_UNKNOWN')

    def test_interrupted_legacy_training_does_not_silently_reset_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe=plan();root=Path(temp)/'run';book=c.Journal(root,recipe);book.emit('TRAIN_START',iteration=0)
            with self.assertRaisesRegex(RuntimeError,'optimizer attempt UNKNOWN'):self._run(root,recipe,resume=True)

    def test_final_fp_fixed_order_both_seats_untouched_control_no_replay(self):
        from pokezero import foulplay_bridge as fp
        with tempfile.TemporaryDirectory() as temp:
            recipe=plan();root=Path(temp)/'run';result,_,_=self._run(root,recipe);calls=[];real=c.file_sha
            async def fake(config):
                self.assertIsInstance(config.showdown_root,Path)
                calls.append((str(config.checkpoint),config.seed_start,config.foulplay_random_seed,config.pokezero_player))
                win=str(config.checkpoint)!='/untouched';name='Learner' if win else 'Peer';side=config.pokezero_player if win else ('p2' if config.pokezero_player=='p1' else 'p1')
                item=dict(seed=config.seed_start,capped=False,tied=False,winner=name,pokezero_won=win,
                    terminal_witness=dict(status='PUBLIC_WIN',winner_player=side,public_terminal_line='|win|'+name))
                return NS(games=[NS(to_dict=lambda:item)])
            with patch.object(c,'file_sha',side_effect=lambda p:c.CHAMPION_SHA if str(p)=='/untouched' else real(p)),\
                 patch.object(fp,'run_controlled_foulplay_benchmark',side_effect=fake):
                report=asyncio.run(c.evaluate(recipe,root,champion_checkpoint='/untouched',final_checkpoint=result['checkpoint'],enable_canary=True))
                asyncio.run(c.evaluate(recipe,root,champion_checkpoint='/untouched',final_checkpoint=result['checkpoint'],enable_canary=True))
            self.assertEqual(len(calls),8)
            self.assertEqual([(s,f,seat) for _,s,f,seat in calls],[(6000+i,7000+i,seat) for i in (0,1) for seat in ('p1','p2') for _ in range(2)])
            self.assertEqual((report['known_pairs'],report['unknown_pairs'],report['paired_mean']),(4,0,1.))
            self.assertEqual(report['paired_seed_means'],[1.,1.]);self.assertFalse(report['automatic_fidelity_pass'])

    def test_original_gae_complete_terminal_is_not_replaced_by_capped_value(self):
        import numpy as np
        from gpu_engine.b4_canary_collection import complete_record
        from pokezero.observation import PokeZeroObservationV0
        from pokezero.trajectory import TrajectoryStep
        legal=(True,)+(False,)*8
        observation=PokeZeroObservationV0(tuple((0,)*41 for _ in range(23)),tuple((0.,)*132 for _ in range(23)),
            (0,)*23,(True,)*23,legal,schema_version='pokezero.observation.v4')
        steps=[TrajectoryStep('p1',i+1,observation,legal,0,action_probability=.75,value_estimate=v) for i,v in enumerate((.2,.4))]
        record=complete_record(c.Game(1,'unit',100,(0,1)),steps,winner=0,turn=2,duration=1.,source_hash='unit-source',checkpoint_sha='d'*64)
        examples=list(dataset.examples_from_record(record,config=c.dataset_controls(training())))
        self.assertEqual([x.return_value for x in examples],[.9999,1.])
        self.assertAlmostEqual(examples[0].ppo_advantage,.19996+.9999*.754*.6)
        self.assertAlmostEqual(examples[-1].ppo_value_target,1.)
        self.assertTrue(all(not x.terminal_capped for x in examples))

    def test_collector_keeps_real23_prefix_behavior_and_unknown_out_of_original_cache(self):
        import numpy as np
        from gpu_engine import b4_canary_collection as engine
        from pokezero.collection import read_rollout_records
        b=2;legal=np.zeros((b,9),bool);legal[:,0]=True
        observation=NS(categorical_ids=np.zeros((b,23,41),np.int32),numeric_features=np.full((b,23,132),.125,np.float32),
            token_type_ids=np.zeros((b,23),np.int32),attention_mask=np.ones((b,23),bool),legal_action_mask=legal)
        owners=tuple(NS(encoded=NS(observation=observation),policy_admitted=np.ones(b,bool),basic_error=np.zeros(b,np.int32),retained_error=np.zeros(b,np.int32)) for _ in range(2))
        core=NS(engine=NS(done=np.zeros(b,bool),winner=np.asarray([0,1])),frame_number=np.zeros(b,np.int32),
            cursor=NS(turn=np.ones(b,np.int32),stream_id=np.asarray([1,2])),admitted=np.ones(b,bool))
        state=NS(core=core,observers=(NS(retained_healthy=np.ones(b,bool)),NS(retained_healthy=np.ones(b,bool))))
        session=NS(state=state,result=NS(owners=owners,state=state))
        diag=NS(active_rows=np.ones(b,bool),executed_rows=np.ones(b,bool),invalid_actions=np.zeros(b,bool))
        config=NS();params={};runtime=object();endings=[]
        backend=engine.Backend.__new__(engine.Backend);backend.prepared=object();backend.pool=object();backend.gender_policy=object()
        backend.plan=NS(max_requests=2);backend.model_config=model_config();backend.source_hash='unit-source'
        def advance(actual,actions,**kwargs):
            self.assertIs(actual,session);self.assertEqual(kwargs['active_rows'].tolist(),[True,True])
            core.frame_number+=1;core.engine.done[:]=True;core.admitted[1]=False
            return session.result
        class Archive:
            def __call__(self,*args):return {'checkpoint_sha256':'e'*64}
        with tempfile.TemporaryDirectory() as temp,patch.object(engine.champion,'export',side_effect=Archive()),\
             patch.object(engine.champion,'load',return_value=(params,config,{'weights_archive_sha256':'f'*64})),\
             patch.object(engine.champion,'config_from_checkpoint',return_value=config),patch.object(engine,'file_sha',return_value='e'*64),\
             patch.object(engine.original,'_bind',return_value=runtime),patch.object(engine.original,'_checked',return_value={'config':config}),\
             patch.object(engine.original.driver,'construct',return_value=session) as construct,patch.object(engine.original.driver,'advance',side_effect=advance),\
             patch.object(engine.original.driver,'latest_diagnostics',return_value=diag),patch.object(engine.jax,'block_until_ready',side_effect=lambda x:x),\
             patch.object(engine.original,'_admission',return_value=(np.ones((b,2),bool),np.zeros(b,bool),None)),\
             patch.object(engine,'decide',return_value=(np.zeros((b,2),np.int32),np.zeros(b,bool),np.zeros((b,2),np.int32),
                 np.full((b,2),.75,np.float32),np.full((b,2),.25,np.float32))):
            path=Path(temp)/'collection';result=backend.collect(plan().iterations[0][0],'/unit-checkpoint',path,
                dataset_config=c.dataset_controls(training()),on_end=lambda *args:endings.append(args),check_progress=lambda:None)
            records=read_rollout_records(path/'complete.jsonl');self.assertEqual(len(records),1)
            self.assertEqual((records[0].battle_id,records[0].terminal.winner),('unit-0','p1'))
            self.assertEqual([s.action_probability for s in records[0].trajectory.steps],[.75,.75])
            self.assertEqual(records[0].trajectory.steps[0].observation.numeric_features[0][0],.125)
            batches=list(dataset.iter_training_cache_batches(path/'cache',config=c.dataset_controls(training()),batch_size=512))
            self.assertEqual(batches[0].returns.tolist(),[1.,-1.]);self.assertEqual(batches[0].opponent_action_indices.tolist(),[0,0])
            self.assertEqual((result['complete'],result['unknown']),(1,1));construct.assert_called_once()
        self.assertEqual([(x[0],x[1]) for x in endings],[(2,'REFUSED'),(1,'COMPLETE')])

    def test_original_network_keys_and_router_retain_literal_sample_probability(self):
        import numpy as np
        from gpu_engine import b4_canary_collection as engine
        legal=np.zeros((1,9),bool);legal[:,:2]=True
        obs=NS(numeric_features=np.zeros((1,23,132),np.float32),categorical_ids=np.zeros((1,23,41),np.int32),
            token_type_ids=np.zeros((1,23),np.int32),attention_mask=np.ones((1,23),bool),legal_action_mask=legal)
        core=NS(cursor=NS(stream_id=np.asarray([1])),frame_number=np.asarray([0]),requests=(NS(actions=object()),NS(actions=object())),orders=(object(),object()))
        result=NS(state=NS(core=core),owners=(NS(encoded=NS(observation=obs)),NS(encoded=NS(observation=obs))))
        heads=engine.champion.Output(np.zeros((2,9),np.float32),np.asarray([.25,-.25],np.float32),np.zeros((2,9),np.float32))
        route=lambda request,order,choice:NS(admitted=np.asarray([True]),engine_action=choice)
        with patch.object(engine.original,'_network',return_value=(heads,0.,0.)) as network,\
             patch.object(engine.original,'_keys',return_value=np.zeros((2,2),np.uint32)) as keys,\
             patch.object(engine.original,'sample_actions',return_value=np.asarray([1,0])),\
             patch.object(engine.original.engine_private_actions,'route_choice',side_effect=route) as router:
            _,refused,choices,probabilities,values=engine.decide({},result,np.ones((1,2),bool),123)
        self.assertEqual(choices.tolist(),[[1,0]]);self.assertEqual(probabilities.tolist(),[[.5,.5]])
        self.assertEqual(values.tolist(),[[.25,-.25]]);self.assertEqual(refused.tolist(),[False])
        network.assert_called_once();keys.assert_called_once();self.assertEqual(router.call_count,2)


if __name__=='__main__':unittest.main()
