"""V2 self_moves, the served checkpoint's declaration end to end (review B1).

- off identities unchanged: the champion contract's namespace, a catalog's encoder_identity and the cache payload carry no
  self_moves key when off;
- a served on-contract encodes on WITHOUT the env (and despite OBS_SELF_MOVES=off): its encoder masks declare it;
- every contract check refuses a catalog/config/archive whose self_moves differs from the contract's;
- neural_cli train builds an on model config from --self-moves (the training run itself patched out).
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from _showdown_root import requires_showdown, showdown_root_str


class ContractIdentityTests(unittest.TestCase):
    def test_off_namespace_is_the_pre_self_moves_identity(self):
        from dataclasses import replace
        from pokezero.live_protocol_actor import ActorContract, _canonical, _sha
        champion = ActorContract.champion()
        self.assertIs(champion.self_moves, False)
        legacy = {k: v for k, v in champion.__dict__.items() if k != 'self_moves'}
        self.assertEqual(champion.namespace, _sha(_canonical(legacy)))
        on = replace(champion, self_moves=True)
        self.assertNotEqual(on.namespace, champion.namespace)
        # identity(): THE contract serialization (the namespace and the B3 study prespec): off = the legacy document.
        self.assertEqual(champion.identity(), legacy)
        self.assertEqual(on.identity(), dict(legacy, self_moves=True))
        self.assertEqual(on.namespace, _sha(_canonical(on.identity())))
        self.assertEqual(champion.encoder_masks(), dict(champion.feature_masks, self_moves=False))
        self.assertEqual(on.encoder_masks()['self_moves'], True)
        with self.assertRaises(ValueError):replace(champion, self_moves=1)

    def test_declared_on_encodes_on_without_and_despite_the_env(self):
        from dataclasses import replace
        from pokezero.live_protocol_actor import ActorContract
        from pokezero.observation import ObservationFeatureMasks
        on = replace(ActorContract.champion(), self_moves=True)
        masks = on.encoder_masks(); masks['opponent_tendency_stats_block'] = masks.pop('stats_block')
        for value in ('', 'off'):
            with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': value}):
                self.assertIs(ObservationFeatureMasks(**masks).self_moves, True)
        off = ActorContract.champion().encoder_masks(); off['opponent_tendency_stats_block'] = off.pop('stats_block')
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'on'}):
            self.assertIs(ObservationFeatureMasks(**off).self_moves, False)   # a declared-off contract is not flipped

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
    def test_b2_masks_follow_the_contract(self):
        from dataclasses import replace
        from gpu_engine import v4_encoder
        from pokezero.live_protocol_actor import ActorContract
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'off'}):
            self.assertTrue(v4_encoder.Masks.from_checkpoint(replace(ActorContract.champion(), self_moves=True).encoder_masks()).self_moves)
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'on'}):
            self.assertFalse(v4_encoder.Masks.from_checkpoint(ActorContract.champion().encoder_masks()).self_moves)


class ContractCheckTests(unittest.TestCase):
    def test_python_backend_refuses_a_mismatch(self):
        from dataclasses import replace
        from pokezero.live_protocol_actor import ActorContract, PythonBackend
        from pokezero.observation import ObservationFeatureMasks
        base = ActorContract.champion()
        masks = base.encoder_masks(); masks['opponent_tendency_stats_block'] = masks.pop('stats_block')
        def config(on):
            return SimpleNamespace(category_vocab=SimpleNamespace(tokens=(), size=base.categorical_vocab_size),
                observation_spec=SimpleNamespace(schema_version='pokezero.observation.v4'),
                feature_masks=ObservationFeatureMasks(**dict(masks, self_moves=on)), set_belief_source=True)
        with mock.patch('pokezero.live_protocol_actor._vocab_sha', return_value=base.category_vocab_sha256):
            for contract_on, config_on in ((False, True), (True, False)):
                with self.assertRaisesRegex(Exception, 'Python encoder contract differs'):
                    PythonBackend(replace(base, self_moves=contract_on), config=config(config_on), battle_id='b', owner='p1', stream_id=1)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
    def test_champion_evaluator_binding_refuses_a_mismatch(self):
        from gpu_engine import champion_evaluator
        from pokezero.live_protocol_actor import ActorContract, MASK_NAMES
        contract = ActorContract.champion()
        masks = SimpleNamespace(**{k: v for k, v in contract.feature_masks}, self_moves=True)
        catalog = SimpleNamespace(checkpoint_sha256=contract.checkpoint_sha256, vocabulary=SimpleNamespace(tokens=(), size=7), masks=masks)
        for meta, catalog_on, ok in (({}, True, False), ({'self_moves': True}, True, True), ({'self_moves': True}, False, False),
                                     ({}, False, True)):
            masks.self_moves = catalog_on
            evaluator = SimpleNamespace(checkpoint_sha256=contract.checkpoint_sha256, config=SimpleNamespace(vocabulary=7), meta=meta)
            with mock.patch('pokezero.live_protocol_actor._vocab_sha', return_value=contract.category_vocab_sha256):
                if ok:self.assertTrue(champion_evaluator.check_binding(evaluator, catalog))
                else:
                    with self.assertRaisesRegex(ValueError, 'self_moves'):champion_evaluator.check_binding(evaluator, catalog)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
    def test_jax_backend_refuses_a_mismatch(self):
        from dataclasses import replace
        import test_self_moves_encoders as fx
        import jax
        from pokezero.live_protocol_actor import ActorContract, JaxBackend, MASK_NAMES, _vocab_sha
        jax.config.update('jax_enable_x64', False); jax.config.update('jax_default_matmul_precision', 'highest')
        fx.B2SelfMovesTests.setUpClass()
        for on in (False, True):
            catalog = fx.B2SelfMovesTests.catalogs[on]
            contract = ActorContract(catalog.checkpoint_sha256, _vocab_sha(catalog.vocabulary), catalog.source.source_hash, 'b' * 64,
                tuple((k, getattr(catalog.masks, k)) for k in MASK_NAMES), 'f' * 40, categorical_vocab_size=catalog.vocabulary.size,
                self_moves=on)
            JaxBackend(contract, catalog=catalog, battle_id='ok', owner='p1', stream_id=1)
            with self.assertRaisesRegex(Exception, 'JAX encoder contract differs'):
                JaxBackend(replace(contract, self_moves=not on), catalog=catalog, battle_id='bad', owner='p1', stream_id=2)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
    def test_v5_load_factory_refuses_an_archive_declaration_mismatch(self):
        """V5 serving (review B1): the archive's meta['self_moves'], the contract and the catalog must agree. Agreeing
        cases pass the check and stop at the next binding check (checkpoint b*64 is not the contract's)."""
        from dataclasses import replace
        from gpu_engine import v5_live_protocol as v5
        from pokezero.live_protocol_actor import ActorContract
        off = ActorContract.champion(); on = replace(off, self_moves=True)
        kwargs = dict(expected_archive_sha256='a' * 64, expected_checkpoint_sha256='b' * 64,
                      expected_category_vocabulary_sha256='c' * 64, expected_label_vocabulary_sha256='d' * 64, enable_v5=True)
        for meta, contract, catalog_on, ok in (({}, off, False, True), ({'self_moves': True}, on, True, True),
                                               ({'self_moves': True}, off, False, False), ({}, on, True, False),
                                               ({'self_moves': True}, on, False, False), ({}, off, True, False)):
            catalog = SimpleNamespace(masks=SimpleNamespace(self_moves=catalog_on))
            with (mock.patch.object(v5, 'CompiledEncoderFunctions'),
                  mock.patch.object(v5.model, 'load', return_value=({}, None, meta)),
                  self.assertRaisesRegex(ValueError, 'same explicit v5 archive' if ok else 'self_moves declaration differs')):
                v5.load_factory('archive.npz', catalog=catalog, contract=contract, **kwargs)

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
    def test_reply_prior_refuses_an_on_reply_model_over_off_tables(self):
        from gpu_engine.depthn_reply_prior import ReplyPrior
        on_model = SimpleNamespace(meta={'self_moves': True}, params={})
        with self.assertRaisesRegex(ValueError, 'self_moves observation tables'):
            ReplyPrior(SimpleNamespace(), on_model, views=lambda *a: None)
        ReplyPrior(SimpleNamespace(observe_self_moves_off=object()), on_model, views=lambda *a: None)   # on tables: accepted
        ReplyPrior(SimpleNamespace(), SimpleNamespace(meta={}, params={}), views=lambda *a: None)       # off model: accepted

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
    def test_search_serving_fails_closed_until_threaded(self):
        from gpu_engine.private_self_moves import refuse_search_serving
        refuse_search_serving(catalog=SimpleNamespace(masks=SimpleNamespace(self_moves=False)), meta={}, context='t')
        for kwargs in (dict(catalog=SimpleNamespace(masks=SimpleNamespace(self_moves=True))), dict(meta={'self_moves': True})):
            with self.assertRaisesRegex(ValueError, 'not threaded yet'):refuse_search_serving(context='t', **kwargs)

    def test_serving_entry_points_refuse_a_contradicting_override(self):
        from pokezero.observation import require_declared_self_moves
        with mock.patch.dict(os.environ, {'OBS_SELF_MOVES': 'on'}):
            with self.assertRaisesRegex(ValueError, 'serving refuses'):require_declared_self_moves(False, context='server')


@requires_showdown('neural_cli builds its category vocabulary from a pokemon-showdown checkout')
@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class TrainerFlagTests(unittest.TestCase):
    def test_neural_cli_train_builds_an_on_config(self):
        from pokezero.neural_cli import main as neural_cli_main
        from pokezero.neural_policy import TransformerEpochMetrics, TransformerTrainingResult

        def fake_train(paths, *, model_config, training_config, initial_model=None):
            return object(), TransformerTrainingResult(model_config=model_config, training_config=training_config,
                epochs=(TransformerEpochMetrics(epoch=1, examples=4, loss=0.25, policy_loss=0.2, policy_accuracy=0.75),))

        for flag, want in (('--self-moves', True), ('--no-self-moves', False), (None, False)):
            with tempfile.TemporaryDirectory() as temp_dir:
                argv = ['train', '--data', 'train-rollouts.jsonl', '--out', str(Path(temp_dir) / 'c.pt'),
                        '--showdown-root', showdown_root_str(), '--observation-schema', 'v2.1'] + ([flag] if flag else [])
                with (mock.patch('pokezero.neural_cli.train_transformer_policy', side_effect=fake_train) as train,
                      mock.patch('pokezero.neural_cli.save_transformer_checkpoint'),
                      contextlib.redirect_stdout(io.StringIO())):
                    self.assertEqual(neural_cli_main(argv), 0)
                config = train.call_args.kwargs['model_config']
                self.assertIs(config.self_moves, want, flag)
                self.assertEqual('self_moves' in config.to_dict(), want, flag)


if __name__ == '__main__':
    unittest.main()
