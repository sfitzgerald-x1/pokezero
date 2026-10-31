"""Pure configuration/template checks; never allocate a model or game bank."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero import v5_phasec as phase
from pokezero.live_protocol_actor import ActorContract
from test_v5_contract import vocabulary
from test_v5_run import plan


def original_metadata():
    # Compact synthetic metadata with the DIRECT9375 source controls. No
    # checkpoint or model tensor is read by this test.
    return dict(checkpoint_sha256=ActorContract.champion().checkpoint_sha256,
        metadata=dict(model_config=dict(policy_id='foundation-midscale-iter-9375',
            categorical_vocab_size=916,dropout=.1),training_config=dict(
                objective='ppo',ppo_target_mode='gae',batch_size=512,epochs=5,
                learning_rate=5.9e-5,learning_rate_schedule='mit-thesis',
                learning_rate_progress_start=.999893333333,learning_rate_progress_end=1.,
                weight_decay=0.,discount=.9999,clip_epsilon=.0829,entropy_coef=.0147,
                value_loss_weight=.4375,value_clip_range=.0184,gae_lambda=.754,
                max_grad_norm=.543,normalize_advantage=True,opponent_action_loss_weight=.1)))


def profile():
    blob=json.dumps(original_metadata(),sort_keys=True)
    return phase.ChampionProfile(hashlib.sha256(blob.encode()).hexdigest(),blob)


class PhaseCPreparationTests(unittest.TestCase):
    def test_direct_metadata_is_external_digest_bound_not_historical_profile(self):
        data=profile();self.assertEqual(data.metadata['training_config']['entropy_coef'],.0147)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'inventory.json';path.write_text(data.payload)
            self.assertEqual(phase.ChampionProfile.read(path,expected_sha256=data.inventory_sha256),data)
            path.write_text(data.payload+' ')
            with self.assertRaisesRegex(ValueError,'differs'):
                phase.ChampionProfile.read(path,expected_sha256=data.inventory_sha256)
        wrong=original_metadata();wrong['checkpoint_sha256']='a'*64
        blob=json.dumps(wrong)
        with self.assertRaisesRegex(ValueError,'selected9375'):
            phase.ChampionProfile(hashlib.sha256(blob.encode()).hexdigest(),blob)

    def test_all_four_arms_have_exact_current_controls_and_intended_treatment_only(self):
        data=profile();configs=[data.configurations(vocabulary(),arm=arm,replicate=rep) for arm,rep in phase.ARMS]
        self.assertEqual([x['role'] for x in configs],['A0','B0','D0','B1'])
        for config in configs:
            model=config['model'];loss=config['loss'];train=config['training']
            self.assertEqual((model.width,model.layers,model.attention_heads,model.expert_width,model.dropout),(512,3,8,512,.1))
            self.assertEqual((train['batch_size'],train['epochs'],train['learning_rate'],train['learning_rate_schedule']),(512,5,5.9e-5,'mit-thesis'))
            self.assertEqual((loss.clip_epsilon,loss.value_weight,loss.entropy_coefficient,loss.policy_lambda),(.0829,.4375,.0147,.754))
            self.assertEqual(config['discount'],.9999);self.assertEqual(config['games'],2_000_000)
            self.assertEqual(train['max_grad_norm'],.543)
            self.assertNotIn('learning_rate_progress_start',train)
            self.assertNotIn('opponent_action_loss_weight',train)
            if model.arm=='A':
                self.assertEqual((loss.value_lambda,loss.value_clip,loss.auxiliary_weight,loss.uniform_kl_coefficient,config['historical_fraction']),(.754,.0184,0.,0.,0.))
            else:
                self.assertEqual((loss.value_lambda,loss.value_clip,loss.auxiliary_weight,loss.uniform_kl_coefficient,config['historical_fraction']),(.95,None,.1,.01,.5))
            self.assertEqual(loss.search_policy_weight,1. if model.arm=='D' else 0.)
        self.assertEqual([x['training']['seed'] for x in configs],[0,0,0,1])
        with self.assertRaises(ValueError):data.configurations(vocabulary(),arm='C')

    def test_preparation_is_honest_about_unbound_panels_and_runtime(self):
        doc=phase.preparation(profile(),vocabulary())
        self.assertEqual(doc['status'],'PREPARATION_ONLY_NO_LAUNCH')
        self.assertEqual(doc['panels']['at'],[500000,1000000,1500000,2000000])
        self.assertIn('NO_EVALUATION_HOOK',doc['panels']['execution'])
        self.assertEqual(doc['endpoint']['paired_games_per_contrast'],2000)
        self.assertEqual(doc['endpoint']['contrasts'],[['B0','A0'],['D0','B0']])
        self.assertEqual(doc['teacher']['outcome_root_mix'],[.5,.5])
        self.assertEqual(doc['teacher']['worlds'],2)
        self.assertEqual(doc['learning_rate']['reset_progress'],[0.,1.])
        self.assertTrue(any('fixed game banks' in x for x in doc['missing_bindings']))
        self.assertFalse(doc['initial_privileged_critic'])
        json.dumps(doc,allow_nan=False)

    def test_bind_requires_full_declared_bank_and_real_scheduled_trainer(self):
        fixture=plan()
        kwargs=dict(run_id='prep',arm='B',replicate=0,iterations=fixture.iterations,
            pool_sha256=fixture.pool_sha256,selection_seed=fixture.selection_seed)
        with self.assertRaisesRegex(ValueError,'complete predeclared2M'):
            phase.bind_plan(profile(),vocabulary(),**kwargs)
        # Tiny declared objects only: patch the production LENGTH requirement,
        # never instantiate/iterate a pretend2M bank just to test configuration.
        with patch.object(phase,'GAMES_PER_ARM',len(fixture.games)):
            if 'learning_rate_schedule' not in {x.name for x in fields(phase.training.TrainConfig)}:
                with self.assertRaisesRegex(phase.runs.MissingDependency,'scheduled'):
                    phase.bind_plan(profile(),vocabulary(),**kwargs)
            else:
                bound=phase.bind_plan(profile(),vocabulary(),**kwargs)
                self.assertEqual(bound.discount,.9999)
                self.assertEqual(bound.train.learning_rate_schedule,'mit-thesis')
                self.assertEqual(bound.iterations,fixture.iterations)
                self.assertEqual((bound.watchdog.initial_window_seconds,bound.watchdog.longest_multiplier,
                    bound.watchdog.backstop_seconds),(1800,5.,30*24*3600.))

    def test_backend_wires_original_provider_only_for_D_without_running_it(self):
        from gpu_engine import v5_selfplay,v5_depth1_provider
        prepared=object();pool=object();parser=object();contract=object();sentinel=object()
        for arm in ('A','B','D'):
            config=plan(arm)
            with patch.object(v5_selfplay,'EngineBackend',return_value=sentinel) as backend,patch.object(v5_depth1_provider,'Provider') as provider:
                self.assertIs(phase.build_backend(config,prepared,pool,public_config=parser,actor_contract=contract),sentinel)
                if arm=='D':
                    self.assertEqual(provider.call_args.args[:3],(prepared,parser,contract))
                    self.assertEqual(provider.call_args.args[3].seed,config.selection_seed)
                    self.assertIs(backend.call_args.kwargs['teacher'],provider.return_value)
                else:
                    provider.assert_not_called();self.assertIsNone(backend.call_args.kwargs['teacher'])
        with self.assertRaises(phase.runs.MissingDependency):phase.build_backend(plan('D'),prepared,pool)

    def test_four_suspended_single_gpu_manifests_share_node_not_seeds_or_argv_defaults(self):
        doc=phase.preparation(profile(),vocabulary())
        bindings=dict(name='unit-phasec',namespace='test',image='unit@sha256:'+'a'*64,
            worker_command=['python','unit_worker.py'],volumes=[],volume_mounts=[],cpu=4,memory='8Gi',private_manifest_root='/unit/private')
        jobs=phase.job_templates(doc,**bindings)
        self.assertEqual(len(jobs),4)
        self.assertEqual([j['metadata']['labels']['pokezero-arm'] for j in jobs],['A0','B0','D0','B1'])
        for job in jobs:
            self.assertTrue(job['spec']['suspend']);self.assertEqual(job['spec']['backoffLimit'],0)
            spec=job['spec']['template']['spec'];container=spec['containers'][0]
            self.assertEqual(container['resources']['requests']['nvidia.com/gpu'],'1')
            self.assertEqual(container['resources']['limits']['nvidia.com/gpu'],'1')
            self.assertEqual(spec['affinity']['podAffinity']['requiredDuringSchedulingIgnoredDuringExecution'][0]['topologyKey'],'kubernetes.io/hostname')
            self.assertEqual(container['args'][-1],'/unit/private/preparation.json')
        for change in ({'image':'unit:latest'},{'worker_command':'python unit.py'},{'cpu':True}):
            with self.assertRaises(ValueError):phase.job_templates(doc,**dict(bindings,**change))
        malformed=deepcopy(doc);malformed['arms'].pop()
        with self.assertRaisesRegex(ValueError,'four'):phase.job_templates(malformed,**bindings)


if __name__=='__main__':unittest.main()
