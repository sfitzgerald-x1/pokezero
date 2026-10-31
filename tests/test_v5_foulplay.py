"""Explicit v5 bridge dispatch with fake games/processes, never Native/model."""
import ast
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock,patch

import jax

from pokezero import foulplay_bridge as old, live_protocol_bridge as bridge, v5_foulplay as v5
from pokezero.policy import PolicyDecision
from test_live_protocol_bridge import MemoryBridge,MemoryServer
from test_gpu_engine_v5_live_protocol import V5LiveProtocolTests as fixture


class V5FoulPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.setUpClass();cls.factory=fixture.factory;cls.f=fixture.f

    def config(self,**updates):
        fields=dict(checkpoint=Path('/synthetic-checkpoint'),showdown_root=Path('/synthetic-source'),
            policy_mode='raw',device='cpu',belief_set_source=True,allow_search_fallback=False,
            opponent_crash_retries=0,encoder_kind='jax-owning',owning_decision_timeout_seconds=60.,
            seed_start=7,games=3,foulplay_random_seed=101)
        return old.ControlledFoulPlayConfig(**dict(fields,**updates))

    def game(self,config,seed,*,capped=False,tie=False):
        winner=None if capped or tie else config.pokezero_username
        witness=dict(status='ROUND_CAP' if capped else 'PUBLIC_TIE' if tie else 'PUBLIC_WIN',
            public_terminal_line=None if capped else '|tie' if tie else '|win|'+winner,
            winner_player=None if capped or tie else config.pokezero_player)
        return old.ControlledFoulPlayGameResult(f'{old.DEFAULT_BATTLE_ID_PREFIX}-{seed}',seed,winner,not(capped or tie),
            10,4,0,0,capped=capped,tied=tie,terminal_witness=witness,encoder_kind='jax-owning',
            encoder_provenance=dict(actor_version='v5'))

    def preparation(self,stack):
        stack.enter_context(patch.object(old,'_validate_external_paths'))
        stack.enter_context(patch('pokezero.live_protocol_runtime._require_source'))
        stack.enter_context(patch('pokezero.live_protocol_runtime._checkpoint_sha',return_value='a'*64))
        legacy=stack.enter_context(patch.object(old,'load_transformer_checkpoint',side_effect=AssertionError('no old loader')))
        return legacy

    def test_explicit_real_factory_dispatch_preserves_bank_and_unknowns_without_old_loader(self):
        config=self.config();seen=[]
        async def run(cfg,**kwargs):
            offset=kwargs['game_offsets'][0];seen.append((cfg,kwargs))
            if offset==1:raise ValueError('unsupported actual wire kind')
            return NS(games=(self.game(cfg,cfg.seed_start+offset,capped=offset==2),),
                foulplay_random_seed_schedule=(101+offset,))
        with ExitStack() as stack,jax.default_matmul_precision('highest'):
            legacy=self.preparation(stack);stack.enter_context(patch.object(old,'_run_controlled_foulplay_games',side_effect=run))
            result=asyncio.run(v5.run_panel(config,self.factory,enable_v5=True))
            json.loads(json.dumps(result))  # ready for the ordinary progress writer
            self.assertEqual([r['seed'] for r in result['rows']],[7,8,9]);self.assertEqual([r['fp_seed'] for r in result['rows']],[101,102,103])
            self.assertEqual([r['status'] for r in result['rows']],['COMPLETE','REFUSED','CAPPED'])
            self.assertEqual([r['score'] for r in result['rows']],[1.,None,None])
            self.assertEqual(result['score_bounds'],[1/3,1.]);self.assertEqual(result['unknown_cells'],2)
            self.assertEqual([kw['game_offsets'] for _,kw in seen],[(0,),(1,),(2,)])
            for cfg,kwargs in seen:
                self.assertIs(cfg,config);self.assertIs(kwargs['actor_runtime'],self.factory)
                self.assertIs(kwargs['owning_actor_factory'],self.factory);self.assertEqual(kwargs['actor_version'],'v5')
                with self.assertRaises(RuntimeError):kwargs['policy'].select_action(None)
            legacy.assert_not_called()

    def test_timeout_stops_then_digest_bound_resume_only_attempts_next_original_cells(self):
        config=self.config();seen=[]
        async def fail(cfg,**kwargs):seen.append(kwargs['game_offsets']);raise TimeoutError('blocked learner')
        async def resume(cfg,**kwargs):
            offset=kwargs['game_offsets'][0];seen.append((offset,));return NS(games=(self.game(cfg,cfg.seed_start+offset,tie=True),),
                foulplay_random_seed_schedule=(101+offset,))
        with ExitStack() as stack,jax.default_matmul_precision('highest'):
            self.preparation(stack)
            with patch.object(old,'_run_controlled_foulplay_games',side_effect=fail):
                first=asyncio.run(v5.run_panel(config,self.factory,enable_v5=True))
            self.assertEqual([r['status'] for r in first['rows']],['FAILED','NOT_ATTEMPTED','NOT_ATTEMPTED'])
            with patch.object(old,'_run_controlled_foulplay_games',side_effect=resume):
                with self.assertRaises(ValueError):asyncio.run(v5.run_panel(config,self.factory,previous=first,enable_v5=True))
                result=asyncio.run(v5.run_panel(config,self.factory,previous=first,expected_previous_sha256=v5._sha(first),enable_v5=True))
            self.assertEqual(seen,[(0,),(1,),(2,)]);self.assertEqual([r['score'] for r in result['rows']],[None,.5,.5])

    def test_original_lowlevel_selected_offset_maps_original_seed_and_fp_seed(self):
        config=self.config();game=self.game(config,9)
        class Server:
            uri='ws://synthetic-only'
            start=AsyncMock();close=AsyncMock()
        class Battle:
            start=AsyncMock();close=AsyncMock()
        with ExitStack() as stack,jax.default_matmul_precision('highest'):
            stack.enter_context(patch.object(old,'_probe_foulplay_start_method',new_callable=AsyncMock,return_value={}))
            stack.enter_context(patch.object(old,'_FoulPlayWebsocketServer',return_value=Server()))
            stack.enter_context(patch.object(old,'_BattleBridge',return_value=Battle()))
            stack.enter_context(patch.object(old,'_spawn_foulplay',new_callable=AsyncMock,return_value=NS(stdout=None,stderr=None)))
            stack.enter_context(patch.object(old,'_drain_process_stream',new_callable=AsyncMock))
            stack.enter_context(patch.object(old,'_wait_for_foulplay_challenge_or_exit',new_callable=AsyncMock))
            stack.enter_context(patch.object(old,'_stop_foulplay_process',new_callable=AsyncMock))
            run=stack.enter_context(patch.object(old,'_run_single_game',new_callable=AsyncMock,return_value=game))
            result=asyncio.run(old._run_controlled_foulplay_games(config,policy=v5._TransportOnly(),policy_id='v5-source-fixture',
                vocab=self.factory.catalog.vocabulary,dex=self.factory.catalog.dex,observation_spec=object(),
                feature_masks=old.DEFAULT_OBSERVATION_FEATURE_MASKS,checkpoint_sha256='a'*64,
                owning_actor_factory=self.factory,actor_runtime=self.factory,actor_version='v5',game_offsets=(2,)))
            self.assertEqual(run.call_args.kwargs['seed'],9);self.assertEqual(run.call_args.kwargs['config'].foulplay_random_seed,103)
            self.assertEqual(result.foulplay_random_seed_schedule,(103,))

    def test_actual_original_memory_game_uses_v5_actor_and_literal_owner_terminal(self):
        config=self.config(games=1);battle=f'{old.DEFAULT_BATTLE_ID_PREFIX}-7'
        stream=lambda seat,lines:dict(type='stream',battleId=battle,stream=seat,lines=list(lines))
        own=tuple(self.f.fixture.public.prefix)+('|turn|1','|request|'+json.dumps(self.f.fixture.request(0)))
        peer='|request|'+json.dumps(self.f.fixture.request(1));terminal='|win|'+config.pokezero_username
        events=[stream('p1',own),stream('p2',[peer]),dict(type='started',battleId=battle,formatid=config.format_id,seed=old.showdown_seed_from_int(7)),
            dict(type='ready',battleId=battle,requested=['p1','p2']),stream('omniscient',[terminal]),
            dict(type='terminal',battleId=battle),stream('p2',[terminal]),stream('p1',[terminal])]
        memory=MemoryBridge(events);trajectories=[]
        with ExitStack() as stack,jax.default_matmul_precision('highest'):
            stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no process')))
            stack.enter_context(patch('subprocess.run',side_effect=AssertionError('no runtime')))
            stack.enter_context(patch.object(old,'_resolved_belief_set_source',return_value=None))
            stack.enter_context(patch.object(old,'_player_state',return_value=NS()))
            stack.enter_context(patch.object(old,'_select_live_foulplay_decision',new_callable=AsyncMock,
                return_value=('move 1',PolicyDecision(0,'external-fake-FP'))))
            for name in ('_context_for_seat','observation_from_player_state','showdown_choice_for_action','_select_policy_decision','_capture_resolved_public_action_round'):
                stack.enter_context(patch.object(old,name,side_effect=AssertionError('no legacy own '+name)))
            result=asyncio.run(old._run_single_game(config=config,bridge=memory,server=MemoryServer(),policy=v5._TransportOnly(),
                vocab=self.factory.catalog.vocabulary,dex=None,observation_spec=object(),seed=7,
                foulplay_process=None,foulplay_logs=old._ProcessLogBuffer(),trajectory_callback=trajectories.append,
                owning_actor_factory=self.factory,actor_runtime=self.factory,actor_version='v5'))
            self.assertTrue(result.encoder_provenance['actual_owner_terminal_seen'])
            self.assertEqual(result.encoder_provenance['actor_version'],'v5')
            self.assertEqual(result.terminal_witness['public_terminal_line'],terminal)
            self.assertEqual(len(trajectories[0].steps),1)
            self.assertEqual(trajectories[0].steps[0].metadata['archive_sha256'],'b'*64)

    def test_legacy_config_and_provenance_shapes_unchanged_and_default_cannot_admit_v5(self):
        before=ast.parse(subprocess.check_output(['git','show','5f9845ae:src/pokezero/foulplay_bridge.py'],text=True))
        after=ast.parse(Path(old.__file__).read_text())
        find=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ControlledFoulPlayConfig')
        self.assertEqual(ast.dump(find(before),include_attributes=False),ast.dump(find(after),include_attributes=False))
        config=self.config();self.assertNotIn('actor_version',asdict(config))
        with self.assertRaises(ValueError):bridge.require_actor_factory(config,self.factory)
        with jax.default_matmul_precision('highest'):
            self.assertIs(bridge.require_actor_factory(config,self.factory,actor_version='v5'),self.factory)
        with patch.object(old,'_run_controlled_foulplay_games') as runner:
            with self.assertRaises(ValueError):asyncio.run(v5.run_panel(config,self.factory))
            runner.assert_not_called()



class V5PanelSupervisionTests(unittest.TestCase):
    def test_owned_group_watchdog_drains_descendants_after_leader_exit_and_on_parse_failure(self):
        class Child:
            pid=12345;returncode=0
            def poll(self):return self.returncode
            def wait(self,timeout=None):return self.returncode
        for bad in (False,True):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as directory:
                child=Child();groups={12345};signals=[];kwargs=[]
                def send(group,signal):signals.append((group,signal));groups.discard(group);child.returncode=-15 if bad else 0
                if bad:
                    child.returncode=None;(Path(directory)/'progress.json').write_text('broken-json')
                def spawn(command,**options):kwargs.append(options);return child
                arguments=dict(progress_path=Path(directory)/'progress.json',initial_window_seconds=10,popen=spawn,
                    group_exists=lambda group:group in groups,signal_group=send,sleep=lambda _:None)
                if bad:
                    with self.assertRaises(ValueError):v5.supervise_panel(['owned-fake-worker'],**arguments)
                else:
                    result=v5.supervise_panel(['owned-fake-worker'],**arguments)
                    self.assertTrue(result['cleanup_confirmed']);self.assertEqual(result['exit_code'],0)
                self.assertEqual(kwargs,[{'start_new_session':True}]);self.assertFalse(groups);self.assertTrue(signals)

    def test_watchdog_uses_completed_game_duration_and_invocation_local_backstop(self):
        import time
        class Child:
            pid=12345;returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout=None):return self.returncode
        for ancient in (False,True):
            with self.subTest(ancient=ancient),tempfile.TemporaryDirectory() as directory:
                clock=[0.];child=Child();groups={12345};path=Path(directory)/'progress.json'
                def send(group,signal):groups.discard(group);child.returncode=-15
                def sleep(seconds):
                    clock[0]+=seconds
                    if not path.exists() and clock[0]>=5:
                        path.write_text(json.dumps(dict(schema=v5.SCHEMA,namespace='literal',started_unix=time.time(),
                            rows=[dict(status='COMPLETE',duration_seconds=4.)])))
                if ancient:path.write_text(json.dumps(dict(schema=v5.SCHEMA,namespace='literal',started_unix=time.time()-1001,rows=[])))
                with patch.object(v5.time,'monotonic',side_effect=lambda:clock[0]):
                    result=v5.supervise_panel(['owned-fake-worker'],progress_path=path,initial_window_seconds=10,
                        backstop_seconds=1000,popen=lambda *a,**kw:child,group_exists=lambda group:group in groups,
                        signal_group=send,sleep=sleep)
                self.assertTrue(result['cleanup_confirmed']);self.assertEqual(result['exit_code'],-15)
                self.assertEqual(result['stop_reason'],'NO_COMPLETED_GAME_WITHIN_ADAPTIVE_WINDOW')
                self.assertGreater(clock[0],10. if ancient else 25.)


if __name__=='__main__':unittest.main()
