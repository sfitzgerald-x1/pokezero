"""Synthetic factory/state checks; no checkpoint, corpus, model forward or game."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero.category_vocab import CategoryVocabulary
from pokezero.foulplay_bridge import ControlledFoulPlayConfig
from pokezero.live_protocol_actor import ActorContract, _mask_values, _vocab_sha
from pokezero.live_protocol_runtime import CompiledEncoderFunctions, _require_source, champion_actor_factory
import test_live_protocol_actor as actor_fixtures


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('subprocess.Popen', side_effect=AssertionError('no process')).start()
        patch('subprocess.run', side_effect=AssertionError('no runtime')).start()

    def bind(self, vocab, *, kind, env, source_hash):
        contract, policy, torch = actor_fixtures.RawTransformerBindingTests().fixture()
        cfg = policy.result.model_config
        cfg.policy_id = 'authored-raw'
        cfg.category_vocab = tuple(vocab.tokens)
        cfg.categorical_vocab_size = vocab.size
        contract = replace(contract, category_vocab_sha256=_vocab_sha(vocab),
            categorical_vocab_size=vocab.size, belief_set_source_hash=source_hash,
            feature_masks=_mask_values(env.feature_masks))
        for key, value in contract.feature_masks:
            setattr(cfg, {'exact_state':'exact_state_enabled', 'stats_block':'stats_block_enabled'}.get(key,key), value)
        model = policy.model
        model.eval = lambda: model
        model.to = lambda device: model
        patch.object(ActorContract, 'champion', return_value=contract).start()
        patch('pokezero.neural_policy.require_torch', return_value=torch).start()
        patch('pokezero.live_protocol_runtime._checkpoint_sha', return_value=contract.checkpoint_sha256).start()
        patch('pokezero.live_protocol_runtime._require_source').start()
        config = ControlledFoulPlayConfig(checkpoint=Path('/authored-champion'),
            showdown_root=Path('/authored-source'), policy_mode='raw', device='cpu',
            belief_set_source=True, allow_search_fallback=False, opponent_crash_retries=0,
            encoder_kind=kind, games=2)
        env = replace(env, showdown_root=config.showdown_root, node_binary=config.node_binary,
                      category_vocab=vocab, set_belief_source=None)
        return champion_actor_factory(config), (model, policy.result, env), contract

    def test_python_factory_fresh_parsers_requests_providers_and_raw_histories(self):
        from test_saved_player_projection import ProductionHistoryTests, leads, request
        fixture = ProductionHistoryTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.source.metadata = SimpleNamespace(source_hash='b'*16)
        factory, runtime, contract = self.bind(fixture.config.category_vocab, kind='python-owning',
            env=fixture.config, source_hash='b'*16)
        actors = []
        for i, owner in enumerate(('p1', 'p2', 'p1', 'p2')):
            actor = factory.actor(runtime)
            actor.begin_game(battle_id=f'authored-{i}', owner=owner, seed=i, stream_id=i+1)
            if i < 2:
                actor._backend.ingest(tuple(leads()) + (request(owner),))
                actor._policy._transformer_policy._history_by_player[owner] = ['authored old history']
            else:
                self.assertEqual(actor._backend.env.protocol_lines, ())
                self.assertEqual(actor._backend.env._latest_requests, {})
                self.assertEqual(actor._policy._transformer_policy._history_by_player, {})
            self.assertIs(actor._policy._transformer_policy.model, runtime[0])
            self.assertIs(actor._policy._transformer_policy.result, runtime[1])
            self.assertEqual(actor._policy._transformer_policy.weights_sha256, contract.checkpoint_sha256)
            actors.append(actor)
        self.assertEqual(len({id(a._backend) for a in actors}), 4)
        self.assertEqual(len({id(a._backend.env._parser) for a in actors}), 4)
        self.assertEqual(len({id(a._policy) for a in actors}), 4)
        self.assertEqual(len({id(a._policy._history) for a in actors}), 4)
        for actor in actors:
            self.assertFalse(actor.close_game()['B3_qualified'])

    def test_runtime_drift_refused_before_new_policy_can_restore_eval(self):
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.showdown import DEFAULT_OBSERVATION_FEATURE_MASKS, V4_REPLAY_OBSERVATION_SPEC
        env = LocalShowdownConfig(observation_spec=V4_REPLAY_OBSERVATION_SPEC,
            feature_masks=replace(DEFAULT_OBSERVATION_FEATURE_MASKS, transition_token_budget=0,
                                  investment_belief_narrowing=False))
        factory, runtime, contract = self.bind(CategoryVocabulary(('authored',)), kind='python-owning',
            env=env, source_hash='b'*16)
        first = factory.policy_factory(contract, model=runtime[0], result=runtime[1])
        self.assertIsNotNone(first)
        with self.assertRaisesRegex(ValueError, 'identity drift'):
            factory.policy_factory(contract, model=object(), result=runtime[1])
        runtime[0].training = True
        with self.assertRaisesRegex(ValueError, 'CPU eval FP32'):
            factory.policy_factory(contract, model=runtime[0], result=runtime[1])
        self.assertTrue(runtime[0].training)
        with self.assertRaisesRegex(ValueError, 'namespace drift'):
            factory.backend_factory(contract, env_config=replace(runtime[2], category_vocab=CategoryVocabulary(('wrong',))),
                battle_id='wrong', owner='p1', stream_id=1)

    def test_explicit_axis_and_source_revision_dirty_and_node_guards(self):
        with self.assertRaisesRegex(ValueError, 'explicit owning axis'):
            champion_actor_factory(SimpleNamespace(encoder_kind='legacy-mixed'))
        contract = ActorContract.champion()
        with tempfile.TemporaryDirectory() as root:
            config = SimpleNamespace(showdown_root=root, node_binary='node')
            def result(text): return SimpleNamespace(stdout=text)
            for revision, dirty, valid in ((contract.showdown_revision, '', True),
                    ('f'*40, '', False), (contract.showdown_revision, ' M data/moves.ts', False)):
                with patch('pokezero.live_protocol_runtime.subprocess.run', side_effect=[result(revision), result(dirty)]), \
                     patch('pokezero.live_protocol_runtime.shutil.which', return_value='/authored/node'):
                    if valid: _require_source(config, contract)
                    else:
                        with self.assertRaisesRegex(ValueError, 'pinned Showdown'):
                            _require_source(config, contract)
            with patch('pokezero.live_protocol_runtime.subprocess.run', side_effect=[result(contract.showdown_revision), result('')]), \
                 patch('pokezero.live_protocol_runtime.shutil.which', side_effect=['/one/node', '/other/node']):
                with self.assertRaisesRegex(ValueError, 'Node differs'):
                    _require_source(config, contract)

    def test_jax_factory_reuses_pure_functions_with_fresh_every_battle_state(self):
        import jax
        from pokezero.local_showdown import LocalShowdownConfig
        from pokezero.showdown import DEFAULT_OBSERVATION_FEATURE_MASKS, V4_REPLAY_OBSERVATION_SPEC
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from gpu_engine import v4_encoder
        V4EncoderTests.setUpClass(); fixture = V4EncoderTests()
        masks = replace(DEFAULT_OBSERVATION_FEATURE_MASKS, exact_state=True,
            feature_pack_last_move=True, opponent_tendency_stats_block=True, transition_token_budget=0)
        env = LocalShowdownConfig(feature_masks=masks, observation_spec=V4_REPLAY_OBSERVATION_SPEC)
        with jax.default_matmul_precision('highest'), jax.enable_x64(False):
            factory, runtime, contract = self.bind(fixture.vocab, kind='jax-owning', env=env,
                source_hash=fixture.catalog.source.source_hash)
            patch('pokezero.randbat.load_gen3_randbat_source_cached', return_value=fixture.public.source).start()
            patch('pokezero.dex.load_showdown_dex_cached', return_value=fixture.inputdex).start()
            compiled = patch('gpu_engine.v4_encoder.compile_encoder', wraps=v4_encoder.compile_encoder).start()
            actors = []
            for i, owner in enumerate(('p1', 'p2', 'p1', 'p2')):
                actor = factory.actor(runtime)
                actor.begin_game(battle_id=f'authored-jax-{i}', owner=owner, seed=i, stream_id=i+1)
                backend = actor._backend
                expected = v4_encoder.initialize(backend.catalog,
                    stream_ids=jax.numpy.array([i+1], jax.numpy.int32),
                    owner_sides=np.array([int(owner[1])-1], np.int32))
                for actual, fresh in zip(jax.tree_util.tree_leaves(backend.adapter.state),
                                         jax.tree_util.tree_leaves(expected)):
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(fresh))
                self.assertIsNone(backend.adapter.bindings)
                self.assertIsNone(backend.terminal_lifecycle_line)
                self.assertEqual(backend.adapter.registered, {i+1:f'authored-jax-{i}'})
                if i < 2:
                    packet = fixture.request(int(owner[1])-1)
                    backend.ingest(tuple(fixture.public.prefix) + ('|turn|1',
                        '|-boost|p1a: Snorlax|atk|2', '|request|'+json.dumps(packet)))
                    backend.encode(SimpleNamespace(own_request_line='|request|'+json.dumps(packet)))
                    backend.ingest(('|tie',))
                actors.append(actor)
            self.assertEqual(compiled.call_count, 1)
            adapters = [actor._backend.adapter for actor in actors]
            self.assertEqual(len({id(adapter) for adapter in adapters}), 4)
            self.assertEqual(len({id(adapter.host) for adapter in adapters}), 4)
            self.assertEqual(len({id(adapter.registered) for adapter in adapters}), 4)
            self.assertEqual(len({id(adapter.step) for adapter in adapters}), 1)
            self.assertEqual(len({id(adapter.encode_step) for adapter in adapters}), 1)
            holder = adapters[0].compiled_functions
            holder.require_catalog(holder.catalog)
            with self.assertRaises(FrozenInstanceError): holder.step = lambda *_: None
            with self.assertRaisesRegex(ValueError, 'namespace drift'):
                holder.require_catalog(replace(holder.catalog))
            original_step = holder.step
            object.__setattr__(holder, 'step', lambda *_: None)
            with self.assertRaisesRegex(ValueError, 'namespace drift'):
                holder.require_catalog(holder.catalog)
            object.__setattr__(holder, 'step', original_step)
            another_holder = CompiledEncoderFunctions(holder.catalog)
            adapters[3].compiled_functions = another_holder
            with self.assertRaisesRegex(ValueError, 'owner/catalog/function drift'):
                actors[3]._backend.ingest(('|turn|1',))
            adapters[3].compiled_functions = holder
            with jax.default_matmul_precision('default'):
                with self.assertRaisesRegex(ValueError, 'precision drift'):
                    actors[2]._backend.ingest(('|turn|1',))
            actors[2]._backend.owner = 'p2'
            with self.assertRaisesRegex(ValueError, 'owner/catalog/function drift'):
                actors[2]._backend.ingest(('|turn|1',))
            adapters[3].step = lambda *_: None
            with self.assertRaisesRegex(ValueError, 'function binding drift'):
                actors[3]._backend.ingest(('|turn|1',))
            for actor in actors: actor.close_game()


if __name__ == '__main__':
    unittest.main()
