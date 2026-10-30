"""Async in-memory wiring controls; no model/corpus/simulator/process access."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import pokezero.foulplay_bridge as bridge_module
from pokezero.foulplay_bridge import ControlledFoulPlayConfig, _ProcessLogBuffer, _run_single_game
from pokezero.live_protocol_actor import ActorContract, JaxBackend, MASK_NAMES, PythonBackend, RawPolicy, _vocab_sha
from pokezero.live_protocol_bridge import OwningActorFactory, OwningWireSession, encoder_provenance, require_actor_factory
from pokezero.policy import PolicyDecision
from test_live_protocol_actor import FixtureBackend, request


class FixturePythonBackend(PythonBackend, FixtureBackend):
    def __init__(self, contract, **identity):
        FixtureBackend.__init__(self,contract)
        self.identity=identity
    ingest=FixtureBackend.ingest
    encode=FixtureBackend.encode
    close=FixtureBackend.close


class FixtureJaxBackend(JaxBackend, FixtureBackend):
    def __init__(self, contract, **identity):FixtureBackend.__init__(self,contract)
    ingest=FixtureBackend.ingest
    encode=FixtureBackend.encode
    close=FixtureBackend.close


class MemoryServer:
    def __init__(self):self.sent=[]
    async def send_room_lines(self,*args):self.sent.append(args)


class MemoryBridge:
    def __init__(self,events,*,stall=False,auto_ack=True):
        self.events=list(events); self.sent=[]; self.stall=stall; self.auto_ack=auto_ack
    async def next_event(self):
        if self.events:return self.events.pop(0)
        if self.stall:await asyncio.Event().wait()
        raise RuntimeError('authored bridge exhausted without owning terminal')
    async def send(self,value):
        self.sent.append(value)
        if self.auto_ack and value.get('type')=='choices':
            acks=[dict(type='choice_ack',battleId=value['battleId'],player=seat,choice=value['choices'][seat],
                **({'submissionId':value['submissionId']} if 'submissionId' in value else {}))
                for seat in ('p1','p2') if seat in value['choices']]
            self.events[:0]=acks


class OwningBridgeTests(unittest.TestCase):
    def setUp(self):
        self.contract=ActorContract.champion()
        self.backends=[]; self.observations=[]; self.trajectories=[]; self.resets=[]
        self.action=0; self.fail_line=None
        self.addCleanup(patch.stopall)
        patch('subprocess.Popen',side_effect=AssertionError('no external process')).start()
        patch('subprocess.run',side_effect=AssertionError('no external runtime')).start()
        patch('pokezero.foulplay_bridge._resolved_belief_set_source',return_value=None).start()
        self.legacy_players=[]
        def opponent_state(state,player,**kwargs):
            self.assertNotEqual(player,self.current_config.pokezero_player)
            self.legacy_players.append(player)
            return SimpleNamespace()
        patch('pokezero.foulplay_bridge._player_state',side_effect=opponent_state).start()
        for name in ('_context_for_seat','observation_from_player_state','showdown_choice_for_action',
                     '_select_policy_decision','_capture_resolved_public_action_round'):
            patch('pokezero.foulplay_bridge.'+name,side_effect=AssertionError('no legacy own '+name)).start()
        self.fp=patch('pokezero.foulplay_bridge._select_live_foulplay_decision',new_callable=AsyncMock).start()
        self.fp.return_value=('move 1',PolicyDecision(0,'external-foulplay'))

    def config(self, *, owner='p1',kind='python-owning',**updates):
        values=dict(checkpoint=Path('/authored-champion'),showdown_root=Path('/authored-source'),
            policy_mode='raw',device='cpu',belief_set_source=True,allow_search_fallback=False,
            opponent_crash_retries=0,encoder_kind=kind,pokezero_player=owner,
            seed_start=7,games=2)
        values.update(updates)
        self.current_config=ControlledFoulPlayConfig(**values)
        return self.current_config

    def factory(self, *, kind='python-owning', backend_factory=None):
        def backend(c,**kwargs):
            value=FixturePythonBackend(c,**kwargs)
            value.fail_line=self.fail_line
            self.backends.append(value)
            return value
        def policy(c,**kwargs):
            def select(observation,*,rng):
                self.observations.append(observation)
                return PolicyDecision(self.action,'authored-raw')
            return RawPolicy(c,select,lambda:self.resets.append(True))
        return OwningActorFactory(kind,self.contract,backend_factory or backend,policy)

    def events(self,config,*,seed=7,owner_lines=None,opponent_lines=None,requested=None):
        battle=f'{bridge_module.DEFAULT_BATTLE_ID_PREFIX}-{seed}'
        owner=config.pokezero_player; peer=config.foulplay_player
        stream=lambda seat,lines:dict(type='stream',battleId=battle,stream=seat,lines=list(lines))
        return [stream('omniscient',['|turn|999','|-message|coordinator only']),
                stream(owner,owner_lines or [request(owner)]),
                stream(peer,opponent_lines or [request(peer)]),
                self.started(config,seed),
                dict(type='ready',battleId=battle,requested=requested or ['p1','p2']),
                stream('omniscient',['|win|'+config.pokezero_username]),
                dict(type='terminal',battleId=battle),
                stream(peer,['|win|'+config.pokezero_username]),
                stream(owner,['|win|'+config.pokezero_username])]

    def started(self,config,seed=7):
        return dict(type='started',battleId=f'{bridge_module.DEFAULT_BATTLE_ID_PREFIX}-{seed}',
            formatid=config.format_id,seed=bridge_module.showdown_seed_from_int(seed))

    async def game(self,config,events,*,factory=None,seed=7,stall=False,runtime=(None,None,None),memory=None):
        memory=memory or MemoryBridge(events,stall=stall); server=MemoryServer()
        result=await _run_single_game(config=config,bridge=memory,server=server,
            policy=object(),vocab=object(),dex=None,observation_spec=object(),seed=seed,
            foulplay_process=None,foulplay_logs=_ProcessLogBuffer(),
            trajectory_callback=self.trajectories.append,
            owning_actor_factory=factory or self.factory(kind=config.encoder_kind),actor_runtime=runtime)
        return result,memory,server

    def test_both_seats_actual_owner_only_exact_observation_choice_and_terminal(self):
        for owner in ('p1','p2'):
            config=self.config(owner=owner)
            result,memory,_=asyncio.run(self.game(config,self.events(config)))
            self.assertEqual(self.backends[-1].lines,[request(owner),'|win|'+config.pokezero_username])
            trajectory=self.trajectories[-1]
            self.assertEqual(len(trajectory.steps),1)
            self.assertIs(trajectory.steps[0].observation,self.observations[-1])
            self.assertEqual(trajectory.steps[0].player_id,owner)
            self.assertEqual(memory.sent[1]['choices'][owner],'move 1')
            self.assertEqual(result.pokezero_decision_players,(owner,))
            self.assertEqual(result.pokezero_submitted_choice_players,(owner,))
            self.assertEqual(result.encoder_kind,'python-owning')
            self.assertTrue(result.encoder_provenance['actual_owner_terminal_seen'])
            self.assertFalse(result.encoder_provenance['public_wire_qualified'])
            self.assertEqual(trajectory.metadata['encoder_provenance']['actor_receipt']['status'],
                             'TERMINAL_UNQUALIFIED')
            self.assertEqual(result.terminal_witness['public_terminal_line'],'|win|'+config.pokezero_username)
            self.assertEqual(self.backends[-1].closed,1)
        self.assertEqual(self.legacy_players,['p2','p1'])

    def test_peer_and_omniscient_variants_never_change_owner_policy_input(self):
        config=self.config()
        events=self.events(config)
        asyncio.run(self.game(config,events))
        changed=self.events(config,opponent_lines=[request('p2',active_index=0)])
        changed[0]['lines']=['|turn|12345','|switch|p2a: Hidden|Hidden|999/999']
        asyncio.run(self.game(config,changed))
        self.assertEqual(self.observations[0],self.observations[1])
        self.assertEqual(self.backends[0].lines,self.backends[1].lines)

    def test_wait_asymmetry_force_switch_every_request_kept_and_no_extra_selection(self):
        config=self.config(); self.action=4
        battle=f'{bridge_module.DEFAULT_BATTLE_ID_PREFIX}-7'
        stream=lambda seat,*lines:dict(type='stream',battleId=battle,stream=seat,lines=list(lines))
        events=[stream('p1',request('p1',wait=True)),stream('p2',request('p2')),
            self.started(config),
            dict(type='ready',battleId=battle,requested=['p2']),
            stream('p1','|-message|own update',request('p1',forced=True)),
            stream('p2',request('p2',wait=True)),
            dict(type='ready',battleId=battle,requested=['p1']),
            stream('p1','|win|'+config.pokezero_username),
            stream('omniscient','|win|'+config.pokezero_username),
            dict(type='terminal',battleId=battle),stream('p2','|win|'+config.pokezero_username)]
        result,memory,_=asyncio.run(self.game(config,events))
        self.assertEqual(len(self.observations),1); self.assertEqual(self.fp.await_count,1)
        self.assertEqual(memory.sent[1]['choices'],{'p2':'move 1'})
        self.assertEqual(memory.sent[2]['choices'],{'p1':'switch 1'})
        self.assertEqual(result.encoder_provenance['actor_receipt']['owner_requests'],2)
        self.assertEqual(result.pokezero_decisions,1)

    def test_bad_battle_role_lines_ready_and_own_request_refuse_close_without_callback(self):
        for mutation in ('battle','role','coerce','empty','ready-type','duplicate-ready','wrong-ready',
                         'foreign-request','foreign-terminal','malformed-terminal','unknown-event',
                         'started-seed','duplicate-started','missing-started','duplicate-terminal'):
            config=self.config(); events=self.events(config)
            if mutation=='battle':events[1]['battleId']='foreign'
            if mutation=='role':events[1]['stream']='spectator'
            if mutation=='coerce':events[1]['lines']=[42]
            if mutation=='empty':events[1]['lines']=[]
            if mutation=='ready-type':events[4]['requested']=('p1','p2')
            if mutation=='duplicate-ready':events[4]['requested']=['p1','p1','p2']
            if mutation=='wrong-ready':events[4]['requested']=['p2']
            if mutation=='foreign-request':events[1]['lines']=[request('p2')]
            if mutation=='foreign-terminal':events[-1]['lines']=['|win|Foreign']
            if mutation=='malformed-terminal':events[-1]['lines']=['|tie|extra']
            if mutation=='unknown-event':events[3]['type']='mystery'
            if mutation=='started-seed':events[3]['seed']='foreign-seed'
            if mutation=='duplicate-started':events.insert(4,dict(events[3]))
            if mutation=='missing-started':events.pop(3)
            if mutation=='duplicate-terminal':events.insert(-2,dict(events[-3]))
            before=len(self.trajectories)
            with self.assertRaises(ValueError,msg=mutation):asyncio.run(self.game(config,events))
            self.assertEqual(self.backends[-1].closed,1,mutation)
            self.assertEqual(len(self.trajectories),before)

    def test_partial_ingest_and_send_failures_keep_actor_closed_and_fail_receipt(self):
        config=self.config(); self.fail_line='|-message|fail'
        events=self.events(config,owner_lines=[request(),'|-message|fail'])
        with self.assertRaisesRegex(ValueError,'late line') as caught:asyncio.run(self.game(config,events))
        receipt=caught.exception.owning_encoder_receipt
        self.assertEqual(receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(receipt['actor_receipt']['owner_lines'],0)
        self.assertEqual(self.backends[-1].closed,1); self.assertEqual(self.trajectories,[])
        self.fail_line=None
        with patch.object(MemoryBridge,'send',new_callable=AsyncMock,side_effect=OSError('authored send fail')):
            with self.assertRaises(OSError) as caught:asyncio.run(self.game(config,self.events(config)))
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(self.backends[-1].closed,1)

    def test_trajectory_callback_failure_happens_after_closed_actor_and_has_receipt(self):
        config=self.config()
        class FailingSink:
            def append(inner,trajectory):
                self.assertEqual(self.backends[-1].closed,1)
                raise OSError('authored report sink failure')
        self.trajectories=FailingSink()
        with self.assertRaisesRegex(OSError,'report sink') as caught:
            asyncio.run(self.game(config,self.events(config)))
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(self.backends[-1].closed,1)

    def test_only_omniscient_terminal_cannot_complete_and_owner_drain_is_bounded(self):
        config=self.config(owning_terminal_timeout_seconds=.005)
        events=self.events(config)[:-2]
        with self.assertRaises(TimeoutError) as caught:
            asyncio.run(self.game(config,events,stall=True))
        self.assertFalse(caught.exception.owning_encoder_receipt['actual_owner_terminal_seen'])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(self.trajectories,[]); self.assertEqual(self.backends[-1].closed,1)

    def test_fresh_pergame_backend_history_and_reset(self):
        config=self.config(); factory=self.factory()
        memory=MemoryBridge(self.events(config,seed=7)+self.events(config,seed=8))
        for seed in (7,8):
            asyncio.run(self.game(config,[],factory=factory,seed=seed,memory=memory))
        self.assertEqual(memory.events,[])
        self.assertIsNot(self.backends[0],self.backends[1])
        self.assertEqual([b.identity['stream_id'] for b in self.backends],[1,2])
        self.assertEqual(len(self.resets),2)
        self.assertNotEqual(self.observations[0].metadata['battle_id'],self.observations[1].metadata['battle_id'])

    def test_owner_terminal_first_drains_same_battle_and_inconsistent_terminal_refuses(self):
        config=self.config()
        events=self.events(config)
        events=events[:5]+[events[-1],events[-3],events[-2],events[-4]]
        result,memory,_=asyncio.run(self.game(config,events))
        self.assertEqual(memory.events,[])
        self.assertEqual(result.encoder_provenance['completed_terminal_streams'],['omniscient','p1','p2'])
        bad=self.events(config); bad[-2]['lines']=['|win|'+config.foulplay_username]
        with self.assertRaisesRegex(ValueError,'terminal streams disagree'):
            asyncio.run(self.game(config,bad))

    def test_missing_peer_terminal_refuses_even_when_owner_terminal_is_present(self):
        config=self.config(owning_terminal_timeout_seconds=.005)
        events=self.events(config); events.pop(-2)
        with self.assertRaises(TimeoutError) as caught:asyncio.run(self.game(config,events,stall=True))
        self.assertTrue(caught.exception.owning_encoder_receipt['actual_owner_terminal_seen'])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(self.trajectories,[])

    def test_final_peer_forwarding_uses_remaining_deadline_and_cancels_blocked_write(self):
        for owner in ('p1','p2'):
            for blocked in (False,True):
                config=self.config(owner=owner,owning_terminal_timeout_seconds=.005)
                events=self.events(config)
                events=events[:5]+[events[5],events[6],events[8],events[7]]
                original=MemoryServer.send_room_lines
                entered=[]; cancelled=[]; finished=[]
                async def terminal_write(server,*args):
                    if any(line.startswith('|win|') for line in args[-1]):
                        entered.append(True)
                        try:
                            if blocked:await asyncio.Event().wait()
                            else:await asyncio.sleep(.04)
                        except asyncio.CancelledError:
                            cancelled.append(True)
                            raise
                        finally:finished.append(True)
                    return await original(server,*args)
                before=len(self.trajectories)
                started=time.monotonic()
                with patch.object(MemoryServer,'send_room_lines',terminal_write):
                    with self.assertRaises(TimeoutError) as caught:
                        asyncio.run(self.game(config,events))
                self.assertLess(time.monotonic()-started,.1)
                self.assertEqual((entered,cancelled,finished),([True],[True],[True]))
                self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
                self.assertEqual(len(self.trajectories),before)
                self.assertEqual(self.backends[-1].closed,1)

    def test_expired_terminal_io_never_constructs_or_starts_operation(self):
        config=self.config(); session=OwningWireSession(config,self.factory(),(None,None,None),
            seed=7,battle_id='authored-unstarted',start_seed=bridge_module.showdown_seed_from_int(7))
        session.terminal_deadline=time.monotonic()-1
        operation=AsyncMock()
        with self.assertRaises(TimeoutError):asyncio.run(session.terminal_io(operation))
        operation.assert_not_called()

    def test_terminal_write_swallowing_cancellation_cannot_publish_success(self):
        config=self.config(owning_terminal_timeout_seconds=.005)
        events=self.events(config); events=events[:5]+[events[5],events[6],events[8],events[7]]
        original=MemoryServer.send_room_lines; cancelled=[]
        async def terminal_write(server,*args):
            if any(line.startswith('|win|') for line in args[-1]):
                try:await asyncio.Event().wait()
                except asyncio.CancelledError:cancelled.append(True)
            return await original(server,*args)
        with patch.object(MemoryServer,'send_room_lines',terminal_write):
            with self.assertRaises(TimeoutError) as caught:asyncio.run(self.game(config,events))
        self.assertEqual(cancelled,[True]); self.assertEqual(self.trajectories,[])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(self.backends[-1].closed,1)

    def test_final_notification_uses_same_deadline(self):
        config=self.config(owning_terminal_timeout_seconds=.005)
        entered=[]; cancelled=[]
        async def blocked_notification(**kwargs):
            entered.append(True)
            try:await asyncio.Event().wait()
            finally:cancelled.append(True)
        with patch.object(bridge_module,'_notify_foulplay_terminal',blocked_notification):
            with self.assertRaises(TimeoutError) as caught:asyncio.run(self.game(config,self.events(config)))
        self.assertEqual((entered,cancelled),([True],[True]))
        self.assertEqual(self.trajectories,[]); self.assertEqual(self.backends[-1].closed,1)
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')

    def test_sync_closure_expiry_keeps_closed_actor_and_never_publishes(self):
        config=self.config(); original=OwningWireSession.close
        def expired_close(session,*,failed):
            session.terminal_deadline=time.monotonic()-1
            return original(session,failed=failed)
        with patch.object(OwningWireSession,'close',expired_close):
            with self.assertRaises(TimeoutError) as caught:asyncio.run(self.game(config,self.events(config)))
        self.assertEqual(self.backends[-1].closed,1); self.assertEqual(self.trajectories,[])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')

    def test_postclose_witness_expiry_refuses_before_trajectory_publication(self):
        config=self.config(); original=OwningWireSession.close; witness=bridge_module._controlled_terminal_witness
        closed=[]
        def close(session,*,failed):
            result=original(session,failed=failed); closed.append(session); return result
        def expire_witness(*args,**kwargs):
            result=witness(*args,**kwargs)
            if closed:closed[-1].terminal_deadline=time.monotonic()-1
            return result
        with patch.object(OwningWireSession,'close',close), \
             patch.object(bridge_module,'_controlled_terminal_witness',expire_witness):
            with self.assertRaises(TimeoutError) as caught:asyncio.run(self.game(config,self.events(config)))
        self.assertEqual(self.backends[-1].closed,1); self.assertEqual(self.trajectories,[])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')
        self.assertEqual(closed[-1].trajectory.metadata['encoder_provenance']['status'],'FAILED_CLOSED_UNQUALIFIED')

    def test_primary_forward_error_survives_cleanup_deadline_expiry(self):
        config=self.config(); original=OwningWireSession.close; failure=OSError('primary peer write failure')
        def close(session,*,failed):
            session.terminal_deadline=time.monotonic()-1
            return original(session,failed=failed)
        original_write=MemoryServer.send_room_lines
        async def fail_write(server,*args):
            if any(line.startswith('|win|') for line in args[-1]):raise failure
            return await original_write(server,*args)
        with patch.object(OwningWireSession,'close',close), patch.object(MemoryServer,'send_room_lines',fail_write):
            with self.assertRaises(OSError) as caught:asyncio.run(self.game(config,self.events(config)))
        self.assertIs(caught.exception,failure)
        self.assertEqual(self.backends[-1].closed,1); self.assertEqual(self.trajectories,[])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')

    def test_first_owner_terminal_ingestion_is_inside_deadline(self):
        config=self.config(); original=OwningWireSession.wait_for_owner_terminal
        sessions=[]
        def start_deadline(session):
            original(session); sessions.append(session)
        ingest=FixturePythonBackend.ingest
        def expire_ingest(backend,lines):
            result=ingest(backend,lines)
            if any(line.startswith('|win|') for line in lines):
                self.assertTrue(sessions)
                sessions[-1].terminal_deadline=time.monotonic()-1
            return result
        events=self.events(config); events=events[:5]+[events[8],events[5],events[6],events[7]]
        with patch.object(OwningWireSession,'wait_for_owner_terminal',start_deadline), \
             patch.object(FixturePythonBackend,'ingest',expire_ingest):
            with self.assertRaises(TimeoutError) as caught:asyncio.run(self.game(config,events))
        self.assertEqual(self.backends[-1].closed,1); self.assertEqual(self.trajectories,[])
        self.assertEqual(caught.exception.owning_encoder_receipt['status'],'FAILED_CLOSED_UNQUALIFIED')

    def test_missing_mismatched_factory_or_backend_cannot_fall_back(self):
        config=self.config()
        with self.assertRaisesRegex(ValueError,'matching exact'):
            asyncio.run(bridge_module.run_controlled_foulplay_benchmark(config))
        wrong=self.factory(kind='jax-owning')
        with self.assertRaises(ValueError):require_actor_factory(config,wrong)
        bad=self.factory(backend_factory=lambda c,**kw:FixtureJaxBackend(c,**kw))
        with self.assertRaisesRegex(ValueError,'backend differs'):
            asyncio.run(self.game(config,self.events(config),factory=bad))
        legacy=self.config(kind='legacy-mixed')
        with self.assertRaises(ValueError):require_actor_factory(legacy,self.factory())

    def test_encoder_axis_forbids_other_policies_budgets_oracle_probes_and_fallback(self):
        config=self.config()
        variants=[dict(policy_mode='root-puct'),dict(capture_driver='random-legal'),
            dict(opponent_policy_mode='raw'),dict(search_time_ms=999),dict(device='cuda'),
            dict(temperature=2.),dict(engine_oracle_belief=True),dict(live_continuation_smoke=True),
            dict(live_continuation_oracle=True),dict(opponent_crash_retries=1),
            dict(allow_search_fallback=True),dict(belief_set_source=None),dict(encoder_kind='jax'),
            dict(owning_terminal_timeout_seconds=True)]
        for values in variants:
            with self.assertRaises(ValueError,msg=str(values)):replace(config,**values)
        with patch.dict('os.environ',{'POKEZERO_TRAIT_CAPTURE':'1'}):
            with self.assertRaises(ValueError):require_actor_factory(config,self.factory())

    def test_legacy_wrapper_delegation_and_mixed_label_preserve_default_route(self):
        config=ControlledFoulPlayConfig(checkpoint=Path('/fixture'),showdown_root=Path('/fixture'),policy_mode='raw')
        sentinel=object()
        with patch('pokezero.foulplay_bridge._run_single_game_body',new_callable=AsyncMock,return_value=sentinel) as body:
            self.assertIs(asyncio.run(_run_single_game(config=config,seed=7)),sentinel)
        body.assert_awaited_once_with(config=config,seed=7)
        self.assertEqual(encoder_provenance(config.encoder_kind)['input_source'],
                         'LEGACY_MIXED_OMNISCIENT_AND_PRIVATE_REQUESTS')


    def test_real_python_owner_stream_to_exact_trajectory_observation(self):
        from test_saved_player_projection import ProductionHistoryTests, leads, request as own_request
        f=ProductionHistoryTests(); f.setUp(); self.addCleanup(f.doCleanups)
        f.source.metadata=SimpleNamespace(source_hash='b'*16)
        self.contract=replace(self.contract,belief_set_source_hash='b'*16,
            category_vocab_sha256=_vocab_sha(f.config.category_vocab),categorical_vocab_size=f.config.category_vocab.size)
        config=self.config(pokezero_username='Authored one',foulplay_username='Authored two')
        def backend(c,env_config,**kw):
            b=PythonBackend(c,config=env_config,**kw); self.backends.append(b); return b
        factory=self.factory(backend_factory=backend)
        lines=list(leads())
        for i,line in enumerate(lines):
            if line.startswith('|switch|p2'):lines[i]=line.rsplit('|',1)[0]+'|67/100'
        lines += [own_request('p1',wait=True),'|-message|literal owner update',own_request('p1',move='thunderbolt')]
        events=self.events(config,owner_lines=lines,opponent_lines=[own_request('p2',wait=True)],requested=['p1'])
        with patch('pokezero.live_protocol_bridge.ActorContract.champion',return_value=self.contract):
            result,_,_=asyncio.run(self.game(config,events,factory=factory,runtime=(None,None,f.config)))
        self.assertIs(self.trajectories[-1].steps[0].observation,self.observations[-1])
        self.assertEqual(result.encoder_kind,'python-owning')
        self.assertTrue(result.encoder_provenance['actual_owner_terminal_seen'])

    def test_real_jax_route_and_canonical_host_terminal_controls_no_device_event(self):
        import jax
        from test_gpu_engine_v4_encoder import V4EncoderTests
        precision=jax.config.jax_default_matmul_precision
        x64=jax.config.jax_enable_x64
        jax.config.update('jax_default_matmul_precision','highest')
        jax.config.update('jax_enable_x64',False)
        self.addCleanup(jax.config.update,'jax_default_matmul_precision',precision)
        self.addCleanup(jax.config.update,'jax_enable_x64',x64)
        V4EncoderTests.setUpClass(); f=V4EncoderTests()
        self.contract=ActorContract('a'*64,_vocab_sha(f.catalog.vocabulary),f.catalog.source.source_hash,
            'd'*64,tuple((k,f.masks[k]) for k in MASK_NAMES),
            'e'*40,categorical_vocab_size=f.catalog.vocabulary.size)
        config=self.config(kind='jax-owning')
        def backend(c,env_config,**kw):
            b=JaxBackend(c,catalog=f.catalog,**kw)
            b.delivered_frames=[]
            feed=b.adapter.feed
            def recorded(lines):b.delivered_frames.append(lines); return feed(lines)
            b.adapter.feed=recorded
            self.backends.append(b)
            return b
        factory=self.factory(kind='jax-owning',backend_factory=backend)
        own='|request|'+json.dumps(f.request(0))
        events=self.events(config,owner_lines=[*f.public.prefix,own],
            opponent_lines=[request('p2',wait=True)],requested=['p1'])
        with patch('pokezero.live_protocol_bridge.ActorContract.champion',return_value=self.contract):
            result,_,_=asyncio.run(self.game(config,events,factory=factory))
        value=self.backends[-1]
        self.assertEqual(len(value.delivered_frames),1)
        self.assertEqual(value.delivered_frames[0],tuple(f.public.prefix))
        self.assertEqual(value.terminal_lifecycle_line,'|win|'+config.pokezero_username)
        self.assertIs(self.trajectories[-1].steps[0].observation,self.observations[-1])
        self.assertEqual(result.encoder_provenance['actor_receipt']['terminal_lifecycle']['kind'],'win')
        self.assertEqual(result.encoder_provenance['actor_receipt']['terminal_lifecycle']['scope'],
                         'OWNING_HOST_LIFECYCLE_NOT_A_DEVICE_FEATURE')
        self.assertFalse(result.encoder_provenance['B3_qualified'])
        # The same real JAX backend also consumes canonical tie with no device
        # frame/feature change, and refuses later encode or malformed controls.
        actor=factory.actor((None,None,None)); actor.begin_game(battle_id='tie-fixture',owner='p1',seed=9,stream_id=3)
        value=self.backends[-1]; initial=value.adapter.state
        actor.accept_owner_stream(battle_id='tie-fixture',owner='p1',generation=0,
            prefix_sha256=actor.prefix_sha256,lines=('|tie','|'))
        self.assertIs(value.adapter.state,initial); self.assertEqual(value.delivered_frames,[])
        self.assertEqual(value.terminal_lifecycle_line,'|tie')
        with self.assertRaises(ValueError):value.encode(actor.snapshot_boundary())
        receipt=actor.close_game()
        self.assertEqual(receipt['status'],'TERMINAL_UNQUALIFIED')
        self.assertTrue(receipt['terminal_lifecycle']['consumed'])
        for index,lines in enumerate((('|win',),('|win|',),('|win| PokeZeroBot',),
                ('|win|PokeZeroBot|extra',),('|tie|',),('|tie|extra',),
                ('|tie','|turn|2'),('|tie',own),('|tie','|tie'))):
            actor=factory.actor((None,None,None))
            actor.begin_game(battle_id=f'bad-terminal-{index}',owner='p1',seed=10+index,stream_id=4+index)
            value=self.backends[-1]; initial=value.adapter.state
            with self.assertRaises(ValueError,msg=str(lines)):
                actor.accept_owner_stream(battle_id=actor.battle_id,owner='p1',generation=0,
                    prefix_sha256=actor.prefix_sha256,lines=lines)
            self.assertIs(value.adapter.state,initial); self.assertEqual(value.delivered_frames,[])
            self.assertEqual(actor.close_game()['status'],'FAILED_CLOSED')
