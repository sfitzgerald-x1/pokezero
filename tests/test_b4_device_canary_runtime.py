"""Inert endpoint holder/process/interval controls; no model or game calls."""
import asyncio
from dataclasses import replace
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

from pokezero import b4_device_canary_runtime as runtime, b4_canary_endpoint as endpoint
from pokezero import live_protocol_actor as actor, live_protocol_runtime as original
from pokezero import foulplay_bridge as fp, live_protocol_bridge as bridge


def config(path):
    return fp.ControlledFoulPlayConfig(checkpoint=Path(path), showdown_root=Path('/inert-showdown'), games=1,
        seed_start=71, foulplay_random_seed=91, pokezero_player='p1', policy_mode='raw', device='cpu',
        capture_driver='checkpoint', encoder_kind='jax-owning', belief_set_source=True,
        opponent_legal_mask_mode='hidden', owning_decision_timeout_seconds=60., allow_search_fallback=False,
        opponent_crash_retries=0)


class DeviceCanaryRuntimeTests(unittest.TestCase):
    def test_device_backend_seam_preserves_original_trainer_schedule_and_all_dispositions(self):
        from tests import test_b4_fidelity_canary as fixture
        from pokezero import b4_fidelity_canary as c
        from gpu_engine import b4_canary_collection as archived
        with patch.object(c,'DECLARED_GAMES',4),patch.object(c,'PANEL_SEEDS',2),tempfile.TemporaryDirectory() as folder:
            recipe=fixture.plan()
            configs=tuple(dict(x,encoder_kind='jax-owning',device='cpu',belief_set_source=True,
                owning_decision_timeout_seconds=60.,allow_search_fallback=False,opponent_crash_retries=0) for x in recipe.fp_configs)
            recipe=replace(recipe,fp_configs=configs)
            with patch.object(archived,'Backend',side_effect=AssertionError('capture collector must not run')):
                result,calls,train=fixture.CanaryTests()._run(Path(folder)/'run',recipe,device_path=True)
            self.assertEqual(calls,[(1,2),(3,4)]);self.assertEqual(result['collection_path'],'DEVICE_PUBLIC_V4_NO_CAPTURE')
            self.assertEqual(result['dispositions']['COMPLETE'],4);self.assertEqual(sum(result['dispositions'].values()),4)
            self.assertEqual([(x.learning_rate_progress_start,x.learning_rate_progress_end)for x in train],[(0.,.5),(.5,1.)])

    def test_bound_factory_compiles_once_current_snapshot_and_original_champion_route_unchanged(self):
        from pokezero import dex, randbat
        from gpu_engine import public_belief, v4_encoder
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'snapshot.pt';path.write_bytes(b'not-a-model; mocked original loader')
            cfg=config(path);digest=original._checkpoint_sha(path)
            masks=NS(**{('opponent_tendency_stats_block' if k=='stats_block' else k):(0 if k=='transition_token_budget' else False) for k in actor.MASK_NAMES})
            vocab=NS(tokens=('', 'field'),size=2)
            mc=NS(to_dict=lambda:{'fixture':'same original descriptor'},policy_id='unit')
            result=NS(model_config=mc)
            contract=actor.ActorContract('0'*64,actor._vocab_sha(vocab),'unit-source',actor._sha(actor._canonical(mc.to_dict())),
                actor._mask_values(masks),'a'*40,categorical_vocab_size=2)
            holder=NS(step=object(),encode_step=object(),require_catalog=lambda c:None)
            catalog=object();source=NS(metadata=NS(source_hash='unit-source'))
            backend_calls=[]
            def backend(c,**kw):
                item=NS(contract=c,**kw);backend_calls.append(item);return item
            with ExitStack() as stack:
                stack.enter_context(patch.object(actor.ActorContract,'champion',return_value=contract))
                load=stack.enter_context(patch.object(fp,'load_transformer_checkpoint',return_value=(object(),result)))
                for name,value in [('_validate_external_paths',None),('_warn_on_belief_provenance_mismatch',None),
                        ('observation_spec_from_model_config',object()),('category_vocab_from_model_config',vocab),
                        ('feature_masks_from_model_config',masks)]:
                    stack.enter_context(patch.object(fp,name,return_value=value))
                source_check=stack.enter_context(patch.object(original,'_require_source'))
                stack.enter_context(patch.object(fp,'LocalShowdownConfig',side_effect=lambda **kw:NS(**kw)))
                stack.enter_context(patch.object(randbat,'load_gen3_randbat_source_cached',return_value=source))
                stack.enter_context(patch.object(dex,'load_showdown_dex_cached',return_value=object()))
                stack.enter_context(patch.object(public_belief,'compile_source',return_value=object()))
                compile=stack.enter_context(patch.object(v4_encoder,'compile_encoder',return_value=catalog))
                functions=stack.enter_context(patch.object(original,'CompiledEncoderFunctions',return_value=holder))
                stack.enter_context(patch.object(actor,'JaxBackend',side_effect=backend))
                worker=runtime.Runtime();first=worker.prepare(cfg,expected_checkpoint_sha256=digest)
                again=worker.prepare(replace(cfg,seed_start=72,foulplay_random_seed=92,pokezero_player='p2'),expected_checkpoint_sha256=digest)
                self.assertIs(first,again);self.assertEqual((load.call_count,compile.call_count,functions.call_count),(1,1,1))
                self.assertEqual(source_check.call_count,2)
                factory=first['factory'];bound=first['contract']
                a=factory.backend_factory(bound,env_config=first['env'],battle_id='a',owner='p1',stream_id=1)
                b=factory.backend_factory(bound,env_config=first['env'],battle_id='b',owner='p1',stream_id=2)
                self.assertIsNot(a,b);self.assertIs(a.compiled_functions,b.compiled_functions)
                self.assertIs(bridge.require_actor_factory(cfg,factory,actor_version='canary-v4'),factory)
                with self.assertRaises(ValueError):bridge.require_actor_factory(cfg,factory) # Original SHA9375 route stays closed.
                forged=runtime.Factory('jax-owning',bound,factory.backend_factory,factory.policy_factory)
                with self.assertRaises(ValueError):runtime.check_factory(cfg,forged)
                path.write_bytes(b'changed snapshot')
                with self.assertRaisesRegex(ValueError,'checkpoint bytes'):worker.prepare(cfg,expected_checkpoint_sha256=digest)

    def test_one_owned_process_per_core_pair_reused_and_closed_without_a_game(self):
        class Input:
            def __init__(self):self.lines=[]
            def write(self,line):self.lines.append(line)
            async def drain(self):pass
        class Output:
            async def readline(self):return b'{"exit": 0}\n'
        process=NS(returncode=None,stdin=Input(),stdout=Output())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'fp-results').mkdir();(root/'fp-requests').mkdir()
            paths=[]
            for i in (0,4):
                p=root/'fp-requests'/f'{i}.json';p.write_text('{"cell":{"slot":0}}');paths.append(p)
                endpoint._write(root/'fp-results'/p.name,{'ordinal':i})
            pool=endpoint.WorkerPool(root,NS(fp_cpu_pairs=((1,2),)))
            async def run():
                values=[await pool.launch(p)for p in paths];await pool.close();return values
            with patch.object(endpoint,'_spawn_owned',new=AsyncMock(return_value=process)) as spawn,\
                    patch.object(endpoint,'_finish_owned_cleanup',new=AsyncMock()) as close:
                self.assertEqual(asyncio.run(run()),[{'ordinal':0},{'ordinal':4}])
            self.assertEqual(spawn.await_count,1);self.assertEqual(close.await_count,1)
            self.assertEqual(len(process.stdin.lines),2);self.assertEqual(pool.workers,{})

    def test_paired_90_failure_only_wholly_negative_and_missing_cells_never_pass(self):
        def rows(delta):
            return [dict(ordinal=4*i+2*s+a,offset=i,seat=s,arm=arm,status='COMPLETE',
                score=float((arm=='CANARY') if delta>0 else (arm=='CONTROL')) if delta else .5)
                for i in range(3)for s in (0,1)for a,arm in enumerate(('CONTROL','CANARY'))]
        for delta in (-1,0,1):
            report=endpoint.paired_interval(rows(delta),seed=7,blocks=3,replicates=100)
            self.assertEqual(report['paired_seed_bootstrap_90'],[float(delta)]*2)
            self.assertEqual(report['canary_failed'],delta<0);self.assertFalse(report['equivalence_established'])
            self.assertNotIn('canary_passed',report)
        missing=rows(1)
        for r in missing:r.update(status='UNKNOWN',score=None)
        report=endpoint.paired_interval(missing,seed=7,blocks=3,replicates=100)
        self.assertEqual(report['paired_seed_bootstrap_90'],[-1.,1.]);self.assertEqual(report['unknown_cells_across_models'],12)
        self.assertIsNone(report['matched_complete_cell_mean']);self.assertFalse(report['canary_failed'])
        with self.assertRaises(ValueError):endpoint.paired_interval(rows(1)[::-1],seed=7,blocks=3,replicates=100)

if __name__=='__main__':unittest.main()
