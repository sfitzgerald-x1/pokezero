"""Inert factory/dispatch/cache controls. No model, JAX device or game runs."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pokezero import owning_encoder_continuation as c, owning_encoder_study as s
from pokezero import owning_worker_runtime as runtime
from pokezero import owning_encoder_worker as worker
from tests.test_owning_encoder_continuation import ContinuationTests


class WorkerRuntimeTests(unittest.TestCase):
    def test_persistent_cache_is_explicit_pinned_and_has_no_device_initialization(self):
        updates=[];jax=SimpleNamespace(config=SimpleNamespace(update=lambda *x:updates.append(x)))
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',jax=jax),patch.object(runtime,'_CACHE',None):
            expected=runtime.configure_compilation_cache(directory)
            self.assertEqual(expected['directory'],str(Path(directory).resolve()))
            self.assertEqual(updates,[('jax_enable_compilation_cache',True),('jax_compilation_cache_dir',str(Path(directory).resolve())),
                ('jax_persistent_cache_min_compile_time_secs',0.)])
            with self.assertRaises(ValueError):runtime.configure_compilation_cache(directory+'/other')
            with self.assertRaises(ValueError):runtime.configure_compilation_cache('relative')

    def test_compile_counters_are_host_only_and_keep_same_functions(self):
        class Factory:pass
        class Function:
            count=0
            def _cache_size(self):return self.count
            def __call__(self,*args):raise AssertionError('no execution')
        factory=Factory();function=Function()
        runtime.register_factory(factory,lambda:dict(fold=function))
        before=runtime.compilation_snapshot(factory);function.count=1
        after=runtime.compilation_snapshot(factory)
        self.assertEqual(before['functions']['fold']['resident_signatures'],0)
        self.assertEqual(after['functions']['fold']['resident_signatures'],1)
        self.assertEqual(before['worker_pid'],after['worker_pid'])
        self.assertEqual(runtime.compilation_snapshot(None)['functions'],{})

    def test_original_raw_runtime_loaded_once_guards_every_cell_and_fresh_game_wrapper(self):
        from pokezero import foulplay_bridge as bridge
        from pokezero.live_protocol_actor import ActorContract
        config=bridge.ControlledFoulPlayConfig(checkpoint=Path('/original-champion'),showdown_root=Path('/original-source'),
            policy_mode='raw',device='cpu',belief_set_source=True,allow_search_fallback=False,
            opponent_crash_retries=0,encoder_kind='jax-owning',games=1)
        model=object();result=SimpleNamespace(model_config=SimpleNamespace(policy_id='raw-original'))
        factory=SimpleNamespace(contract=ActorContract.champion())
        wrappers=[];calls=[]
        def policy(**kwargs):
            value=SimpleNamespace(policy_id='raw-original',history=[]);wrappers.append(value);return value
        async def game(cfg,**kwargs):
            calls.append((cfg,kwargs));return object()
        bound=runtime.ChampionWorkerRuntime()
        with patch.object(bridge,'_validate_external_paths') as paths, \
             patch('pokezero.live_protocol_runtime._require_source') as source, \
             patch('pokezero.live_protocol_runtime._checkpoint_sha',return_value=factory.contract.checkpoint_sha256) as sha, \
             patch.object(bridge,'_checkpoint_sha256',return_value=factory.contract.checkpoint_sha256), \
             patch.object(bridge,'load_transformer_checkpoint',return_value=(model,result)) as load, \
             patch.object(bridge,'_warn_on_belief_provenance_mismatch'), \
             patch.object(bridge,'observation_spec_from_model_config',return_value='original-spec'), \
             patch.object(bridge,'category_vocab_from_model_config',return_value='original-vocab'), \
             patch.object(bridge,'load_showdown_dex_cached',return_value='original-dex'), \
             patch.object(bridge,'feature_masks_from_model_config',return_value='original-masks'), \
             patch.object(bridge,'LocalShowdownConfig',side_effect=lambda **kw:SimpleNamespace(**kw)), \
             patch('pokezero.live_protocol_runtime.champion_actor_factory',return_value=factory) as make, \
             patch('pokezero.live_protocol_bridge.require_actor_factory'), \
             patch.object(bridge,'_build_policy',side_effect=policy), \
             patch.object(bridge,'_run_controlled_foulplay_games',side_effect=game):
            async def two():
                await bound.run(config)
                await bound.run(replace(config,seed_start=config.seed_start+1,pokezero_player='p2'))
            asyncio.run(two())
            self.assertEqual((load.call_count,make.call_count),(1,1))
            self.assertEqual((paths.call_count,source.call_count,sha.call_count),(2,2,2))
            self.assertIs(calls[0][1]['actor_runtime'][0],model)
            self.assertIs(calls[0][1]['actor_runtime'][2],calls[1][1]['actor_runtime'][2])
            self.assertIsNot(wrappers[0],wrappers[1]);self.assertIsNot(wrappers[0].history,wrappers[1].history)
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                bound.prepare(replace(config,max_decision_rounds=config.max_decision_rounds+1))
            with patch('pokezero.live_protocol_runtime._checkpoint_sha',return_value='0'*64),self.assertRaisesRegex(ValueError,'bytes changed'):
                bound.prepare(config)


class PersistentDispatchTests(unittest.TestCase):
    def setUp(self):
        self.f=ContinuationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_explicit_plan_and_stop_marker_drain_without_next_attempt(self):
        original=self.f.plan()
        plan=c.make_plan(**original['references'],cpu_pairs=original['cpu_pairs'],
            model_factory_lifecycle='PERSISTENT_PER_CORE_WORKER',compilation_cache_directory=str(self.f.root/'cache'))
        c.validate_plan(plan)
        self.assertNotIn('compilation_cache_directory',original)
        output=self.f.root/'drained';calls=[]
        async def fake(index,pair):
            calls.append(index)
            self.assertEqual(json.loads((output/'ledger.json').read_bytes())['cells'][index]['status'],'ATTEMPTED')
            if index==326:
                s.atomic_json(output/'stop-dispatch.json',dict(schema='pokezero.B3-stop-dispatch.v1',
                    plan_sha256=s.prespec_file_sha256(plan),stop_new_dispatch=True))
            await asyncio.sleep(0)
            receipt=self.f.receipt(index,pair);receipt['worker_compilation']=dict(before={'cold':True},after={'cold':False})
            return receipt
        result=asyncio.run(c.run_continuation(plan,output,worker=fake))
        self.assertEqual(calls,[326,327])
        self.assertEqual([result['cells'][i]['status'] for i in (326,327,328)],['COMPLETE','COMPLETE','NOT_ATTEMPTED'])
        self.assertEqual(result['failure']['error_type'],'DispatchDrainRequested')
        self.assertEqual(result['cells'][326]['worker_compilation']['after'],{'cold':False})
        self.assertEqual(result['cells'][:325],self.f.ledger['cells'][:325])

    def test_serve_reuses_one_runtime_claims_before_game_and_stops_after_failure(self):
        original=self.f.plan()
        plan=c.make_plan(**original['references'],cpu_pairs=original['cpu_pairs'],
            model_factory_lifecycle='PERSISTENT_PER_CORE_WORKER',compilation_cache_directory=str(self.f.root/'cache'))
        calls=[];claim=[];stub=SimpleNamespace(factories={'python-owning':None})
        async def run(config):
            index=config.seed_start-self.f.spec['seed_start'];calls.append(index)
            self.assertEqual(claim[-1],index)
            if index==328:raise RuntimeError('unit failure')
            return SimpleNamespace(to_dict=lambda:self.f.receipt(index,[0,1])['document'])
        stub.run=run
        commands=io.StringIO(''.join(json.dumps(dict(index=i))+'\n' for i in (326,327,328,329)))
        with patch.object(worker.continuation,'cpu_allocation',return_value={}), \
             patch.object(worker.os,'sched_setaffinity',create=True),patch.object(worker.os,'sched_getaffinity',return_value={0},create=True), \
             patch.object(worker.study,'_configure_precision',return_value={'unit':True}), \
             patch.object(worker,'configure_compilation_cache',return_value={'unit':True}), \
             patch.object(worker,'ChampionWorkerRuntime',return_value=stub) as constructor, \
             patch.object(worker.continuation,'claim_cell',side_effect=lambda p,i,pair,path:claim.append(i)), \
             patch.object(worker.sys,'stdin',commands):
            self.assertEqual(asyncio.run(worker.serve(plan,self.f.root,[0,1])),1)
        self.assertEqual(calls,[326,327,328]);constructor.assert_called_once()
        self.assertEqual(json.loads((self.f.root/'cell-327.json').read_bytes())['worker_compilation']['game_ordinal'],1)
        self.assertEqual(json.loads((self.f.root/'cell-328.json').read_bytes())['status'],'FAILED')
        self.assertFalse((self.f.root/'cell-329.json').exists())

    def test_pool_retains_pair_process_and_never_resends_existing_cell(self):
        plan=self.f.plan();output=self.f.root
        writes=[];process=SimpleNamespace(returncode=None,pid=999)
        class Input:
            def write(inner,line):
                index=json.loads(line)['index'];writes.append(index)
                s.atomic_json(output/f'cell-{index}.json',self.f.receipt(index,[0,1]))
            async def drain(inner):pass
            def close(inner):pass
        process.stdin=Input()
        async def wait():process.returncode=0;return 0
        process.wait=wait
        async def body():
            pool=worker.Pool(output/'plan.json',plan,output)
            await pool.invoke(326,[0,1]);await pool.invoke(328,[0,1])
            with self.assertRaisesRegex(ValueError,'resent'):await pool.invoke(326,[0,1])
            await pool.close();await pool.close()
        with patch.object(worker.asyncio,'create_subprocess_exec',new_callable=AsyncMock,return_value=process) as spawn, \
             patch.object(worker.continuation,'_stop_owned_group',new_callable=AsyncMock) as cleanup:
            asyncio.run(body())
        self.assertEqual(writes,[326,328]);spawn.assert_awaited_once();cleanup.assert_awaited_once_with(process)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertEqual(spawn.call_args.args[-2:],('0','1'))


if __name__=='__main__':unittest.main()
