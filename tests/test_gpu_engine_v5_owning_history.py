"""Tiny original FULL writer/private parser fixtures; no constructor or game."""
from contextlib import ExitStack
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import v5_owning_history as h, private_actions, public_wire, v4_encoder
from gpu_engine import engine_retained_encoder_driver as d


class OwningHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_v4_encoder as f
        f.V4EncoderTests.setUpClass();cls.f=f.V4EncoderTests();cls.catalog=cls.f.catalog

    def runtime(self, lines):
        _, projections=public_wire.project_lines(public_wire.WireState(),lines,**v4_encoder.wire_kwargs(self.catalog))
        words=np.asarray([event.words() for p in projections for event in p.events],np.int32)
        capacity=((len(words)+63)//64)*64
        packet=np.zeros((1,2,capacity,16),np.int32);valid=np.zeros((1,2,capacity),bool)
        packet[0,0,:len(words)]=words;packet[0,1]=777777;valid[0,0,:len(words)]=True
        packet=packet.reshape(1,2,-1,64,16);valid=valid.reshape(1,2,-1,64)
        more=np.zeros(valid.shape[:3],bool)
        for i in range(more.shape[2]):more[0,0,i]=valid[0,0,i+1:].any()
        request=self.f.request(0);request['side']['pokemon'].reverse()
        facts=private_actions.parse_action_facts(request,owner_side=0,vocabulary=self.catalog.vocabulary,catalog=self.catalog.actions)
        batched=jax.tree.map(lambda x:j.asarray(x)[None],facts)
        moves=tuple(self.catalog.actions.names[int(i)] for i in np.asarray(facts.mechanics) if i)
        metadata={'vocabulary':{'moves':['']+list(moves)}}
        original=NS(moves=j.asarray([[[1,2,0,0],[1,2,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]],j.int32))
        core=NS(frame_number=j.asarray([0],j.int32),cursor=NS(stream_id=j.asarray([31],j.int32)),
            requests=(NS(actions=NS(facts=batched,admitted=j.asarray([True]))),object()),
            orders=(NS(request_to_physical=j.asarray([[1,0,-1,-1,-1,-1]],j.int32)),object()),
            originals=(original,object()),frame_input_report=NS(owner_full_words=j.asarray(packet),
                owner_full_valid=j.asarray(valid),owner_full_more=j.asarray(more),
                frame_inputs_complete=j.asarray([True]),refusal=j.asarray([0],j.int32)))
        core.constructors=(NS(binding=j.zeros((1,8),j.uint32),species=j.ones((1,6),j.int32),
            moves=original.moves,max_pp=j.ones((1,6,4),j.int32)*24),object())
        state=NS(core=core);session=d.Session(object(),state,object());record=dict(state=state,constructed=object())
        runtime=dict(catalog=self.catalog,request={'metadata':json.dumps(metadata)})
        return session,record,runtime,words

    def checked(self,stack,session,record,runtime):
        stack.enter_context(patch.object(d,'_session',side_effect=lambda actual:record if actual is session else None))
        stack.enter_context(patch.object(d,'_checked',return_value=runtime))
        stack.enter_context(patch.object(d,'_checked_witness'))

    def test_same_call_owner_only_frames_and_original_own_request_roundtrip(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80, M|120/240',
            '|switch|p2a: Absol|Absol, L80, F|100/100','|turn|1']
        session,record,runtime,words=self.runtime(lines)
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime);history=h.capture(session,row=0,owner=0)
        frame=history.frames[0]
        self.assertFalse(frame.refusal, frame.refusal)
        self.assertFalse(hasattr(history,'session'))
        self.assertEqual(dict(history.own_constructor)['max_pp'][0],(24,24,24,24))
        self.assertNotIn('777777',str(frame.words))
        np.testing.assert_array_equal(np.asarray(frame.words)[np.asarray(frame.valid)],words)
        request=json.loads(frame.request_json)
        self.assertEqual(request['side']['id'],'p1')
        self.assertEqual(request['active'][0]['moves'][0]['pp'],10)
        self.assertEqual(request['active'][0]['moves'][1]['pp'],12)
        self.assertEqual(request['side']['pokemon'][0]['moves'],['bodyslam','surf'])
        decoded=h.decode(history)
        self.assertTrue(decoded.complete,decoded.refusal_counts)
        self.assertEqual(decoded.required_raw_groups,3)
        self.assertEqual(decoded.lines[:-1],tuple(lines))

    def test_history_replay_gap_and_late_start_fail_before_copy(self):
        session,record,runtime,_=self.runtime(['|switch|p1a: Snorlax|Snorlax, L80, M|120/240'])
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime);history=h.capture(session,row=0,owner=0)
            with self.assertRaisesRegex(ValueError,'consecutive'):h.capture(session,row=0,owner=0,history=history)
            session.state.core.frame_number=j.asarray([2],j.int32)
            with self.assertRaisesRegex(ValueError,'frame0'):h.capture(session,row=0,owner=0)

    def test_unsupported_group_and_source_refusal_remain_nonfatal_and_counted(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80, M|120/240',
            '|switch|p2a: Absol|Absol, L80, F|100/100','|-start|p2a: Absol|confusion']
        session,record,runtime,_=self.runtime(lines)
        session.state.core.frame_input_report.frame_inputs_complete=j.asarray([False])
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime);history=h.capture(session,row=0,owner=0)
        decoded=h.decode(history)
        self.assertFalse(decoded.complete)
        self.assertEqual(decoded.required_raw_groups,3)
        self.assertGreaterEqual(decoded.unsupported_raw_groups,1)
        self.assertIn(('source-frame-refused',1),decoded.refusal_counts)

    def test_multichunk_retains_every_word_and_detects_more_flag_loss(self):
        lines=['|switch|p1a: Snorlax|Snorlax, L80, M|120/240',
            '|switch|p2a: Absol|Absol, L80, F|100/100']+['|']*80
        session,record,runtime,words=self.runtime(lines)
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime);history=h.capture(session,row=0,owner=0)
            self.assertGreater(len(history.frames[0].words),1)
            self.assertEqual(sum(sum(mask) for mask in history.frames[0].valid),len(words))
            report=session.state.core.frame_input_report
            report.owner_full_more=j.zeros_like(report.owner_full_more)
            with self.assertRaisesRegex(ValueError,'more flags'):h.capture(session,row=0,owner=0)
        self.assertEqual(h.decode(history).required_raw_groups,len(lines))

    def test_missing_request_inverse_never_fabricates_source_packet(self):
        session,record,runtime,_=self.runtime(['|switch|p1a: Snorlax|Snorlax, L80, M|120/240'])
        packet=session.state.core.requests[0].actions
        packet.facts=packet.facts._replace(request=packet.facts.request._replace(owner_side=j.asarray([1],j.int32)))
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime);history=h.capture(session,row=0,owner=0)
        self.assertIsNone(history.frames[0].request_json)
        self.assertIn(('own-request-owner-mismatch',1),history.frames[0].refusal)
        self.assertFalse(h.decode(history).complete)

    def test_continuation_preserves_initial_request_and_actual_frame_word_order(self):
        initial=['|switch|p1a: Snorlax|Snorlax, L80, M|120/240',
            '|switch|p2a: Absol|Absol, L80, F|100/100','|turn|1']
        extra=['|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|90/100','|','|upkeep','|turn|2']
        session,record,runtime,firstwords=self.runtime(initial)
        _,_,_,allwords=self.runtime(initial+extra)
        words=allwords[len(firstwords):]
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime);history=h.capture(session,row=0,owner=0)
            core=session.state.core;core.frame_number=j.asarray([1],j.int32)
            padded=np.zeros((1,2,1,64,16),np.int32);padded[0,0,0,:len(words)]=words
            valid=np.zeros((1,2,1,64),bool);valid[0,0,0,:len(words)]=True
            report=core.frame_input_report
            report.owner_full_words=j.asarray(padded);report.owner_full_valid=j.asarray(valid)
            report.owner_full_more=j.zeros((1,2,1),bool)
            packet=core.requests[0].actions
            packet.facts=packet.facts._replace(request=packet.facts.request._replace(pp=packet.facts.request.pp.at[0,0].set(9)))
            history=h.capture(session,row=0,owner=0,history=history)
        self.assertEqual([frame.number for frame in history.frames],[0,1])
        self.assertEqual(json.loads(history.frames[0].request_json)['active'][0]['moves'][0]['pp'],10)
        self.assertEqual(json.loads(history.frames[1].request_json)['active'][0]['moves'][0]['pp'],9)
        decoded=h.decode(history)
        self.assertTrue(decoded.complete,decoded.refusal_counts)
        self.assertEqual(decoded.required_raw_groups,len(initial)+len(extra))
        self.assertEqual(sum(line.startswith('|request|') for line in decoded.lines),2)

    def test_actual_process_free_public_root_join_without_invented_start_lines(self):
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
        from pokezero.live_protocol_actor import ActorContract,_vocab_sha
        source=self.f.public.source
        masks=dict(self.f.masks);masks['opponent_tendency_stats_block']=masks.pop('stats_block')
        config=LocalShowdownConfig(showdown_root='/authored-source-only',set_belief_source=True,
            category_vocab=self.catalog.vocabulary,observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=ObservationFeatureMasks(**masks))
        contract=replace(ActorContract.champion(),category_vocab_sha256=_vocab_sha(config.category_vocab),
            categorical_vocab_size=config.category_vocab.size,belief_set_source_hash=source.metadata.source_hash)
        session,record,runtime,_=self.runtime(['|switch|p1a: Snorlax|Snorlax, L80, M|120/240',
            '|switch|p2a: Absol|Absol, L80, F|100/100','|turn|1'])
        with ExitStack() as stack:
            self.checked(stack,session,record,runtime)
            stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no simulator')))
            stack.enter_context(patch('subprocess.run',side_effect=AssertionError('no runtime')))
            stack.enter_context(patch('pokezero.local_showdown.load_gen3_randbat_source_cached',return_value=source))
            stack.enter_context(patch('pokezero.local_showdown.load_showdown_dex_cached',return_value=self.catalog.dex))
            history=h.capture(session,row=0,owner=0)
            result=h.public_root(history,config=config,contract=contract,battle_id='authored',seed=11)
        self.assertTrue(result.decoded.complete,result.decoded.refusal_counts)
        self.assertIsNotNone(result.root)
        root=result.root;state=root.context.public_materialization_state
        self.assertEqual(state.replay.requests,{})
        self.assertEqual(state.self_request['side']['id'],'p1')
        self.assertEqual(root.address.stream,31)
        self.assertFalse(root.device_branch_ready)
        self.assertFalse(any(line=='|start' for line in result.decoded.lines))

    def test_literal_original_device_produced_request_inverse_both_owners(self):
        # Existing two-row literal component fixture only; no generated team,
        # scientific root address, model, Native simulator or game is used.
        import test_gpu_engine_engine_own_facts as fixture
        fixture.DeviceOwnFactsTests.setUpClass();f=fixture.DeviceOwnFactsTests
        snapshot=f.start
        core=NS(requests=snapshot.requests,orders=snapshot.orders,
            originals=tuple(reg.original for reg in snapshot.registrations))
        for owner in (0,1):
            for row in (0,1):
                packet=json.loads(h._request(core,row,owner,f.catalog,f.metadata))
                self.assertEqual(packet['side']['id'],f'p{owner+1}')
                self.assertEqual(len(packet['side']['pokemon']),6)
                self.assertEqual(packet['active'][0]['moves'][0]['pp'],24)
                self.assertEqual(packet['active'][0]['moves'][1]['pp'],16)


if __name__=='__main__':unittest.main()
