"""Fake FP panels and source-only setup checks; no JAX/model/game execution."""
import ast
import asyncio
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pokezero import v5_foulplay as fp, v5_endpoint as endpoint, v5_evaluation as evaluation
from pokezero import foulplay_bridge as old, owning_worker_runtime as runtime
from pokezero.live_protocol_actor import ActorContract
import test_v5_endpoint as endpoint_fixture
import test_v5_evaluation as milestone_fixture


class PersistentPanelCompilationTests(unittest.TestCase):
    def test_same_factory_all_cells_before_after_counters_and_full_unknowns(self):
        class Function:
            signatures=0
            def _cache_size(self):return self.signatures
        class Factory:pass
        factory=Factory();function=Function()
        factory.contract=ActorContract.champion();factory.catalog=NS(vocabulary='UNIT',dex='UNIT')
        runtime.register_factory(factory,lambda:dict(fold=function))
        config=old.ControlledFoulPlayConfig(checkpoint=Path('/unit'),showdown_root=Path('/unit'),
            encoder_kind='jax-owning',owning_decision_timeout_seconds=60.,policy_mode='raw',
            device='cpu',belief_set_source=True,allow_search_fallback=False,opponent_crash_retries=0,
            games=3,seed_start=7,foulplay_random_seed=101)
        calls=[];progress=[]
        async def game(cfg,**kwargs):
            self.assertIs(kwargs['actor_runtime'],factory)
            offset=kwargs['game_offsets'][0];calls.append(offset);function.signatures=1
            if offset==1:raise ValueError('unit unsupported input')
            source=dict(seed=cfg.seed_start+offset,capped=False,tied=False,winner=cfg.pokezero_username,
                pokezero_won=True,terminal_witness=dict(status='PUBLIC_WIN',winner_player=cfg.pokezero_player,
                    public_terminal_line='|win|'+cfg.pokezero_username))
            return NS(games=(NS(seed=source['seed'],capped=False,to_dict=lambda:deepcopy(source)),),
                foulplay_random_seed_schedule=(101+offset,))
        with patch.object(fp,'_recipe',return_value={'archive_sha256':'a'*64}), \
             patch.object(old,'_run_controlled_foulplay_games',side_effect=game):
            result=asyncio.run(fp.run_panel(config,factory,enable_v5=True,progress_callback=lambda v:progress.append(deepcopy(v))))
        self.assertEqual(calls,[0,1,2]);self.assertEqual(result['unknown_cells'],1)
        self.assertEqual(result['score_bounds'],[2/3,1.])
        self.assertEqual([x['status'] for x in result['rows']],['COMPLETE','REFUSED','COMPLETE'])
        self.assertEqual(result['rows'][0]['worker_compilation']['before']['functions']['fold']['resident_signatures'],0)
        self.assertTrue(all(x['worker_compilation']['after']['functions']['fold']['resident_signatures']==1 for x in result['rows']))
        self.assertEqual(result['rows'][2]['worker_compilation']['before']['functions']['fold']['resident_signatures'],1)
        started=next(v for v in progress if v['rows'][0]['status']=='RUNNING')
        self.assertIn('before',started['rows'][0]['worker_compilation'])
        self.assertNotIn('after',started['rows'][0]['worker_compilation'])

    def test_endpoint_accepts_only_new_diagnostic_key_without_loosening_source_outcomes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(endpoint,'TRAINING_GAMES',1):
            ref,protocol,configs=endpoint_fixture.fixture(Path(temp))
            row=endpoint._rows(protocol,configs)[0]
            result=endpoint_fixture.source_result(row,protocol,configs,.5)
            old_result=deepcopy(result)
            expected=endpoint._checked_result(row,result,configs,protocol)
            result['row']['worker_compilation']=dict(before={'unit':True},after={'unit':True})
            self.assertEqual(endpoint._checked_result(row,result,configs,protocol)['score'],expected['score'])
            self.assertEqual(result['row']['source_result_sha256'],old_result['row']['source_result_sha256'])
            bad=deepcopy(result);bad['row']['score']=1.
            with self.assertRaises(ValueError):endpoint._checked_result(row,bad,configs,protocol)
            bad=deepcopy(result);bad['row']['anything_else']=True
            with self.assertRaises(ValueError):endpoint._checked_result(row,bad,configs,protocol)

    def test_actual_child_path_configures_disk_cache_before_factory_without_initializing_jax(self):
        case=milestone_fixture.EvaluationTests('test_owned_child_binds_policy_before_typed_runtime_and_never_uses_unreviewed_executor')
        case.setUp();self.addCleanup(case.doCleanups)
        with patch.object(runtime,'configure_compilation_cache') as configure:
            case.test_owned_child_binds_policy_before_typed_runtime_and_never_uses_unreviewed_executor()
            configure.assert_called_once()
            self.assertEqual(configure.call_args.args[0].name,'jax-compilation-cache')
        for module,target in ((evaluation,'_factory'),(endpoint,'_factories')):
            body=ast.parse(inspect.getsource(module.child))
            calls=[(n.lineno,n.func.id) for n in ast.walk(body) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            cache=min(line for line,name in calls if name=='configure_compilation_cache')
            factory=min(line for line,name in calls if name==target)
            self.assertLess(cache,factory)


if __name__=='__main__':unittest.main()
