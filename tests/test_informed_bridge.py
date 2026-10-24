"""P2 opt-in registration and accounting; no games or strength claims."""
from dataclasses import replace
from contextlib import ExitStack
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import pokezero.foulplay_bridge as bridge


class InformedBridgeTests(unittest.TestCase):
    def config(self,**kwargs):
        values=dict(checkpoint=Path('checkpoint.pt'),showdown_root=Path('/showdown'),
            policy_mode='informed-one-ply',belief_set_source=True,allow_search_fallback=False,
            engine_model_path=Path('model.pt'),engine_tables_path=Path('tables.json'),
            one_ply_trace_path=Path('candidate-p1.decisions.jsonl'),one_ply_seconds=60.,
            device='cuda')
        values.update(kwargs)
        return bridge.ControlledFoulPlayConfig(**values)

    def test_explicit_cli_and_effective_config_roundtrip(self):
        args=bridge.build_arg_parser().parse_args([
            '--checkpoint','checkpoint.pt','--showdown-root','/showdown',
            '--policy-mode','informed-one-ply','--belief-set-source','on',
            '--no-search-fallback','--engine-model-path','model.pt',
            '--engine-tables-path','tables.json','--one-ply-seconds','60',
            '--one-ply-trace-path','candidate-p2.decisions.jsonl',
            '--device','cpu','--one-ply-model-device','cuda',
            '--one-ply-batch-size','1024','--one-ply-bulk-encode-workers','8'])
        cfg=bridge._config_from_args(args)
        self.assertEqual(cfg.policy_mode,'informed-one-ply')
        self.assertEqual(cfg.one_ply_trace_path,Path('candidate-p2.decisions.jsonl'))
        self.assertEqual((cfg.device,cfg.one_ply_model_device),('cpu','cuda'))
        effective=bridge._informed_policy_config(cfg)
        self.assertEqual((effective.worlds,effective.batch_size,effective.bulk_encode_workers),(4,1024,8))

    def test_invalid_or_ignored_flags_are_refused(self):
        for kwargs in (dict(one_ply_trace_path=None),dict(one_ply_seconds=None),
                       dict(engine_model_path=None),dict(engine_tables_path=None),
                       dict(allow_search_fallback=True),dict(belief_set_source=False),
                       dict(engine_worlds=3),dict(value_checkpoint=Path('other.pt')),
                       dict(temperature=2),dict(search_time_ms=760),
                       dict(foulplay_mcts_iterations=128),dict(one_ply_seconds=0),
                       dict(one_ply_bulk_encode_workers=33),dict(policy_mode='raw')):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.config(**kwargs)

    def test_raw_opponent_clears_candidate_only_options(self):
        cfg=self.config(opponent_policy_mode='raw')
        other=bridge._opponent_seat_config(cfg)
        self.assertEqual(other.policy_mode,'raw')
        for name in bridge._ONE_PLY_FIELDS:self.assertIsNone(getattr(other,name))
        self.assertEqual(other.checkpoint,cfg.checkpoint)

    def test_candidate_exact_chance_never_changes_foulplay_environment(self):
        cfg=self.config()
        with patch.dict(os.environ,POKEZERO_ENUMERATE_ROLLS='1'):
            for policy in (cfg,replace(cfg,policy_mode='raw',**{n:None for n in bridge._ONE_PLY_FIELDS})):
                self.assertEqual(bridge._foulplay_env(policy)['POKEZERO_ENUMERATE_ROLLS'],'0')
                self.assertEqual(os.environ['POKEZERO_ENUMERATE_ROLLS'],'1')

    def test_factory_uses_exact_policy_and_declared_artifacts_not_mcts(self):
        cfg=self.config()
        with patch.object(bridge,'load_showdown_dex_cached',return_value='dex'), \
             patch.object(bridge,'load_gen3_randbat_source_cached',return_value='source'), \
             patch('pokezero.informed_policy.InformedRootPolicy') as factory:
            bridge._build_policy(config=cfg,model=object(),result=object(),value_model=object(),
                value_result=object(),env_config=object(),rollout_config=object(),policy_id='candidate')
        args=factory.call_args.kwargs
        self.assertEqual(args['policy_id'],'candidate+informed-one-ply')
        self.assertEqual(args['model_path'],'model.pt')
        self.assertEqual(args['trace_path'],cfg.one_ply_trace_path)
        self.assertEqual(args['config'].worlds,4)
        self.assertEqual(args['device'],'cuda')

    def test_summary_is_distinct_and_refusal_is_not_a_clean_partial(self):
        cfg=self.config()
        result=bridge.ControlledFoulPlayBenchmarkResult(config=cfg,policy_id='candidate',
            games=(),policy_stats=dict(decisions=1,completed_decisions=0,refusals=1))
        doc=result.to_dict()
        self.assertIsNone(doc['engine_mcts'])
        self.assertEqual(doc['informed_root']['information_model'],'informed_adversary_root_minimax')
        self.assertEqual(doc['informed_root']['status'],'REFUSED')
        self.assertEqual(doc['informed_root']['support_floor'],'NONE')
        self.assertEqual(doc['informed_root']['config']['worlds'],4)
        self.assertEqual(doc['informed_root']['policy_stats']['refusals'],1)
        policy=SimpleNamespace(stats=SimpleNamespace(to_dict=lambda:{'decisions':1}))
        self.assertEqual(bridge._engine_policy_stats(policy,'informed-one-ply'),{'decisions':1})
        self.assertIsNone(bridge._engine_policy_stats(policy,'raw'))

    def test_actual_benchmark_owner_preserves_prefix_and_closes_on_failure(self):
        from pokezero.informed_policy import InformedPolicyStats
        cfg=self.config()
        for already_recorded in (False,True):
            with self.subTest(already_recorded=already_recorded),ExitStack() as stack:
                stats=InformedPolicyStats(refusals=int(already_recorded))
                def refusal(*args):stats.refusals+=1
                policy=SimpleNamespace(policy_id='candidate',stats=stats,
                    close=Mock(),record_failure=Mock(side_effect=refusal))
                result=SimpleNamespace(model_config=SimpleNamespace(policy_id='candidate'))
                values={'_validate_external_paths':None,'_checkpoint_sha256':'f'*64,
                    'load_transformer_checkpoint':(object(),result),
                    '_warn_on_belief_provenance_mismatch':None,
                    'observation_spec_from_model_config':object(),
                    'category_vocab_from_model_config':object(),
                    'load_showdown_dex_cached':object(),
                    'feature_masks_from_model_config':object(),
                    '_build_policy':policy}
                for name,value in values.items():
                    stack.enter_context(patch.object(bridge,name,return_value=value))
                marker=object()
                async def fail(config,**kwargs):
                    kwargs['progress_callback'](bridge.ControlledFoulPlayBenchmarkResult(
                        config=config,policy_id='candidate',games=(marker,),policy_stats={'refusals':0}))
                    raise ValueError('actual authored owner failure')
                stack.enter_context(patch.object(bridge,'_run_controlled_foulplay_games',side_effect=fail))
                snapshots=[]
                with self.assertRaisesRegex(ValueError,'actual authored owner failure'):
                    asyncio.run(bridge.run_controlled_foulplay_benchmark(cfg,progress_callback=snapshots.append))
                self.assertEqual(snapshots[-1].games,(marker,))
                self.assertEqual(snapshots[-1].policy_stats['refusals'],1)
                policy.close.assert_called_once()
                self.assertEqual(policy.record_failure.call_count,0 if already_recorded else 1)

    def test_completed_game_survives_foulplay_cleanup_failure_before_progress(self):
        from pokezero.informed_policy import InformedPolicyStats
        cfg=self.config();stats=InformedPolicyStats()
        def refusal(*args):stats.refusals+=1
        policy=SimpleNamespace(stats=stats,record_failure=Mock(side_effect=refusal))
        game=bridge.ControlledFoulPlayGameResult(battle_id='battle-1',seed=cfg.seed_start,
            winner=cfg.pokezero_username,pokezero_won=True,decision_rounds=1,
            pokezero_decisions=1,root_puct_searches=0,root_puct_fallbacks=0)
        endpoints=[]
        class Endpoint:
            def __init__(self,**kwargs):
                self.uri='ws://authored';self.closed=False;endpoints.append(self)
            async def start(self):pass
            async def close(self):self.closed=True
        snapshots=[]
        with ExitStack() as stack:
            for name in ('_FoulPlayWebsocketServer','_BattleBridge'):
                stack.enter_context(patch.object(bridge,name,Endpoint))
            for name,value in (
                ('_probe_foulplay_start_method',{}),
                ('_spawn_foulplay',SimpleNamespace(stdout=None,stderr=None)),
                ('_wait_for_foulplay_challenge_or_exit',None),('_run_single_game',game)):
                stack.enter_context(patch.object(bridge,name,new=AsyncMock(return_value=value)))
            stack.enter_context(patch.object(bridge,'_stop_foulplay_process',
                new=AsyncMock(side_effect=ValueError('cleanup after actual terminal'))))
            with self.assertRaisesRegex(ValueError,'cleanup after actual terminal'):
                asyncio.run(bridge._run_controlled_foulplay_games(cfg,policy=policy,
                    policy_id='candidate',vocab=object(),dex=object(),observation_spec=object(),
                    feature_masks=object(),checkpoint_sha256='f'*64,
                    progress_callback=snapshots.append))
        self.assertIs(snapshots[-1].games[0],game)
        self.assertEqual(snapshots[-1].completed_games,1)
        self.assertEqual(snapshots[-1].to_dict()['informed_root']['status'],'REFUSED')
        self.assertTrue(all(endpoint.closed for endpoint in endpoints))
        policy.record_failure.assert_called_once()


if __name__=='__main__':unittest.main()
