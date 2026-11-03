"""Literal owning wire + original device encoder, with a fake tiny raw head.

No checkpoint/model/simulator/game is loaded or run. Only the archive loader and
policy head are faked; original public fold/own request/encoder/append execute.
"""
from dataclasses import replace
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import v5_live_protocol as p
from pokezero import live_protocol_actor as live, v5_contract as contract
from pokezero.neural_policy import _sample_action


class V5LiveProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_v5_observation as fixtures
        fixtures.V5ObservationTests.setUpClass();cls.f=fixtures.V5ObservationTests
        cls.catalog=cls.f.catalog;cls.config=cls.f.adapter.model_config
        cls.contract=live.ActorContract('a'*64,live._vocab_sha(cls.catalog.vocabulary),cls.catalog.source.source_hash,
            contract._sha(cls.config.metadata()),tuple((k,cls.f.fixture.masks[k]) for k in live.MASK_NAMES),
            'e'*40,categorical_vocab_size=cls.catalog.vocabulary.size)
        cls.seen=[]
        def forward(params,inputs,config,**kwargs):
            cls.seen.append(set(inputs));count=inputs['history_mask'].shape[0]
            return p.model.Output(j.zeros((count,9),j.float32),j.full(count,.2,j.float32),None,inputs['history_mask'][:,0])
        cls.head=patch.object(p.model,'forward',side_effect=forward);cls.head.start();cls.addClassCleanup(cls.head.stop)
        with jax.default_matmul_precision('highest'):cls.factory=cls.make_factory(deterministic=False)

    @classmethod
    def make_factory(cls,**kwargs):
        with patch.object(p.model,'load',return_value=({},cls.config,{'weights_archive_sha256':'b'*64})) as load:
            value=p.load_factory('/synthetic-only',catalog=cls.catalog,contract=cls.contract,
                expected_archive_sha256='b'*64,expected_checkpoint_sha256='a'*64,
                expected_category_vocabulary_sha256=cls.config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=cls.config.label_vocabulary.sha256,enable_v5=True,**kwargs)
            assert load.call_args.kwargs['expected_archive_sha256']=='b'*64
            assert load.call_args.kwargs['expected_category_vocabulary_sha256']==cls.config.category_vocabulary_sha256
            return value

    def actor(self,*,owner=0,stream=1,battle='literal',factory=None):
        actor=(factory or self.factory).create_actor()
        actor.begin_game(battle_id=battle,owner=f'p{owner+1}',seed=42,stream_id=stream)
        return actor

    def feed(self,actor,*,request=None,lines=None):
        if lines is None:
            request=self.f.fixture.request(int(actor.owner[1])-1) if request is None else request
            lines=tuple(self.f.fixture.public.prefix)+('|turn|1','|request|'+json.dumps(request))
        actor.accept_owner_stream(battle_id=actor.battle_id,owner=actor.owner,generation=actor.generation,
            prefix_sha256=actor.prefix_sha256,lines=tuple(lines))

    def test_same_actual_encode_step_v4_prefix_v5_public_tail_and_original_seeded_raw_rng(self):
        with jax.default_matmul_precision('highest'):
            results=[]
            for owner in (0,1):
                actor=self.actor(owner=owner,stream=owner+1,battle='owner'+str(owner));self.feed(actor)
                before=actor._backend.adapter.state;snapshot=actor.snapshot_boundary();rng=random.Random(13)
                decision=actor.decide(snapshot,rng=rng);routing=actor.decision_observation(snapshot)
                backend=actor._backend;inputs=backend.last_inputs
                self.assertEqual(backend._compiled_functions.calls,1)
                self.assertIs(before,backend.adapter.state)  # boundary copies do not commit belief counters
                self.assertIsNone(backend.captured);self.assertIsNone(backend.pending)
                self.assertEqual(inputs['numeric_features'].shape,(1,1,29,132))
                for name in ('categorical_ids','numeric_features','token_type_ids','attention_mask'):
                    np.testing.assert_array_equal(np.asarray(inputs[name])[0,0,:23],getattr(routing,name))
                np.testing.assert_array_equal(np.asarray(inputs['token_type_ids'])[0,0,23:],[7]*6)
                self.assertTrue(np.asarray(inputs['attention_mask'])[0,0,23:].all())
                self.assertEqual(routing.schema_version,'pokezero.observation.v4')
                self.assertEqual(routing.perspective.player_id,actor.owner)
                reference=random.Random(13)
                legal=tuple(np.flatnonzero(routing.legal_action_mask).tolist())
                expected=_sample_action(decision.metadata['policy_distribution'],legal,reference)
                self.assertEqual(expected,decision.action_index);self.assertEqual(rng.getstate(),reference.getstate())
                self.assertAlmostEqual(decision.value_estimate,.2,places=6)
                self.assertIn(actor.choice(snapshot).split()[0],('move','switch'))
                self.assertEqual(set(inputs),set(p.model.MODEL_INPUTS))
                results.append((decision.action_index,decision.action_probability,decision.value_estimate))
                self.assertFalse(actor.close_game()['B3_qualified'])
            repeat=self.actor(owner=0,stream=7,battle='repeat');self.feed(repeat)
            decision=repeat.decide(repeat.snapshot_boundary(),rng=random.Random(13))
            self.assertEqual(results[0],(decision.action_index,decision.action_probability,decision.value_estimate))
            self.assertIs(repeat._backend.adapter.step,self.factory.encoder.step)
            self.assertTrue(all(keys==set(p.model.MODEL_INPUTS) for keys in self.seen))

    def test_foreign_request_peer_exact_hp_and_split_fail_before_forward(self):
        with jax.default_matmul_precision('highest'):
            cases=[('|request|'+json.dumps(self.f.fixture.request(1)),),
                ('|-damage|p2a: Absol|90/301',),('|split|p1',),
                ('|-sethp|p1a: Snorlax|20/240|p2a: Absol|10/301',)]
            for index,lines in enumerate(cases):
                actor=self.actor(stream=20+index,battle='bad'+str(index));old=len(self.seen)
                with self.assertRaises(ValueError):self.feed(actor,lines=lines)
                self.assertEqual(actor.status,'POISONED');self.assertEqual(len(self.seen),old)

    def test_wait_terminal_and_same_request_replay_never_call_raw_policy(self):
        with jax.default_matmul_precision('highest'):
            actor=self.actor(stream=31,battle='wait');request=self.f.fixture.request();request.pop('active');request['wait']=True
            self.feed(actor,request=request);snapshot=actor.snapshot_boundary()
            with self.assertRaises(live.NoAction):actor.decide(snapshot,rng=random.Random(1))
            self.assertEqual(actor._backend._compiled_functions.calls,0)
            self.feed(actor,lines=('|request|'+json.dumps(self.f.fixture.request()),));snapshot=actor.snapshot_boundary()
            actor.decide(snapshot,rng=random.Random(1))
            with self.assertRaises(ValueError):actor.decide(snapshot,rng=random.Random(1))
            self.assertEqual(actor._backend._compiled_functions.calls,1)
            terminal=self.actor(stream=32,battle='terminal');self.feed(terminal,lines=('|win|Player1',))
            with self.assertRaises(live.NoAction):terminal.decide(terminal.snapshot_boundary(),rng=random.Random(1))

    def test_deadline_includes_owner_fold_encoding_and_late_forward_cannot_publish_choice(self):
        with jax.default_matmul_precision('highest'):
            actor=self.actor(stream=41,battle='deadline');clock=[10.]
            with patch.object(p.time,'monotonic',side_effect=lambda:clock[0]):
                self.feed(actor);clock[0]=71.
                with self.assertRaises(TimeoutError):actor.decide(actor.snapshot_boundary(),rng=random.Random(1))
                self.assertEqual(actor._backend._compiled_functions.calls,0)
            actor=self.actor(stream=42,battle='late');clock=[10.]
            with patch.object(p.time,'monotonic',side_effect=lambda:clock[0]):
                self.feed(actor)
                original=jax.block_until_ready
                def slow(value):
                    result=original(value)
                    if isinstance(value,p.model.Output):clock[0]=71.
                    return result
                with patch.object(p.jax,'block_until_ready',side_effect=slow):
                    with self.assertRaises(TimeoutError):actor.decide(actor.snapshot_boundary(),rng=random.Random(1))
                self.assertEqual(actor.status,'POISONED');self.assertIsNone(actor._choice)
            actor=self.actor(stream=43,battle='single-lines');clock=[10.]
            original=p._Backend.ingest
            def charged(backend,lines):
                result=original(backend,lines);clock[0]+=1 if any(x.startswith('|request|') for x in lines) else 40
                return result
            with patch.object(p.time,'monotonic',side_effect=lambda:clock[0]),patch.object(p._Backend,'ingest',charged):
                self.feed(actor,lines=tuple(self.f.fixture.public.prefix)+('|turn|1',))
                self.feed(actor,lines=('|request|'+json.dumps(self.f.fixture.request()),));clock[0]=71.
                with self.assertRaises(TimeoutError):actor.decide(actor.snapshot_boundary(),rng=random.Random(1))
                self.assertEqual(actor._backend._compiled_functions.calls,0)

    def test_explicit_archive_namespace_and_factory_identity_no_default_enablement(self):
        with patch.object(p.model,'load') as load:
            with self.assertRaises(ValueError):p.load_factory('/never-open',catalog=None,contract=None,
                expected_archive_sha256='b'*64,expected_checkpoint_sha256='a'*64,
                expected_category_vocabulary_sha256='c'*64,expected_label_vocabulary_sha256='c'*64)
            load.assert_not_called()
        with jax.default_matmul_precision('highest'):
            with self.assertRaises(ValueError):p._checked(replace(self.factory))
            with self.assertRaises(ValueError):self.make_factory(learner_seconds=61)
            with self.assertRaises(ValueError):jax.jit(lambda dummy:p._checked(self.factory) or dummy)(j.array(0))
            with patch.object(p.model,'load',return_value=({},replace(self.config,replicate=1),{'weights_archive_sha256':'b'*64})):
                with self.assertRaisesRegex(ValueError,'namespace'):p.load_factory('/synthetic-only',catalog=self.catalog,contract=self.contract,
                    expected_archive_sha256='b'*64,expected_checkpoint_sha256='a'*64,
                    expected_category_vocabulary_sha256=self.config.category_vocabulary_sha256,
                    expected_label_vocabulary_sha256=self.config.label_vocabulary.sha256,enable_v5=True)

    def test_explicit_greedy_raw_mode_preserves_rng_and_bound_callback_cannot_be_replaced(self):
        # This checks selection and callback integrity, not CPU compilation
        # latency. The separate deadline test advances its clock past60s and
        # checks fold/forward refusal. Keep the real factory limit unchanged.
        with jax.default_matmul_precision('highest'),patch.object(p,'time',SimpleNamespace(monotonic=lambda:100.)):
            factory=self.make_factory(deterministic=True,compute_dtype='bfloat16')
            actor=self.actor(stream=51,battle='greedy',factory=factory);self.feed(actor)
            rng=random.Random(12);before=rng.getstate()
            decision=actor.decide(actor.snapshot_boundary(),rng=rng)
            self.assertEqual(decision.action_index,0);self.assertEqual(decision.action_probability,1.)
            self.assertEqual(rng.getstate(),before);self.assertEqual(decision.metadata['inference_precision'],'bfloat16')
            actor=self.actor(stream=52,battle='changed');self.feed(actor)
            with patch.object(actor._policy,'_select_action',return_value=decision):
                with self.assertRaisesRegex(ValueError,'callback'):actor.decide(actor.snapshot_boundary(),rng=random.Random(12))
            actor=self.actor(stream=53,battle='changed-validator');self.feed(actor)
            with patch.object(actor._policy,'_validate',lambda:None):
                with self.assertRaisesRegex(ValueError,'callback'):actor.decide(actor.snapshot_boundary(),rng=random.Random(12))


if __name__=='__main__':unittest.main()
