"""V2 self_moves through a V5/V2 run: the plan declares it and the run follows it.

- Plan.self_moves: off manifests carry no key (declarations, resume and identity unchanged); an on declaration loads as an
  on plan; a present false or non-bool value is refused (no default or discarded field);
- prepare_device_runtime compiles the plan's declaration into the catalog and so into the device tables (on tables for an
  on plan, off for an off plan or a bare pool binding);
- the Python routes, the device backend's plan check, actor loads and the unthreaded gate tools fail closed on a mismatch.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from _showdown_root import requires_showdown


class PlanDeclarationTests(unittest.TestCase):
    def test_declaration_round_trip_and_off_identity(self):
        import test_v5_worker as fx
        authored = fx.WorkerTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe, contract, document, runtime, _ = authored.fixture(root)
            self.assertIs(recipe.self_moves, False)
            self.assertNotIn('self_moves', recipe.manifest())
            self.assertIs(authored.load(document, runtime, contract).plan.self_moves, False)
            on = replace(recipe, self_moves=True)
            self.assertIs(on.manifest()['self_moves'], True)
            self.assertEqual({k: v for k, v in on.manifest().items() if k != 'self_moves'}, recipe.manifest())
            loaded = authored.load(fx.record(root / 'plan-on.json', on.manifest()), runtime, contract).plan
            self.assertEqual(loaded, on)
            for value, message in ((False, 'full original plan differs'), ('yes', 'explicit bool self_moves')):
                bad = fx.record(root / f'plan-{value}.json', dict(recipe.manifest(), self_moves=value))
                with self.assertRaisesRegex(ValueError, message):authored.load(bad, runtime, contract)
            with self.assertRaisesRegex(ValueError, 'explicit bool self_moves'):replace(recipe, self_moves=1)

    def test_python_routes_fail_closed_when_on(self):
        from pokezero import v5_worker as w, v5_python_backend as python
        inputs = SimpleNamespace(plan=SimpleNamespace(self_moves=True, pool_sha256='p' * 64))
        with self.assertRaisesRegex(ValueError, 'device route only'):w.prepare_runtime(inputs)
        with self.assertRaisesRegex(ValueError, 'device route only'):python.prepare(inputs)


@requires_showdown('the synthetic fixture compiles its catalog from a pokemon-showdown checkout')
@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class DeviceRuntimeTests(unittest.TestCase):
    """prepare_device_runtime over the synthetic seven-species fixture: real compile_source / compile_encoder, the
    champion contract (its masks, exact_state on) rebound to the fixture source, the pool and backend preparation
    patched; the prepared device tables are compiled from the catalog it builds."""

    @classmethod
    def setUpClass(cls):
        import test_v2_offline_view_parity as fx
        cls.f = fx.fixture()

    def prepare(self, plan):
        from pokezero import v5_worker as w
        from pokezero.live_protocol_actor import ActorContract
        from gpu_engine import engine_device_observation as encoder
        f = self.f
        vocabulary = f.catalog.vocabulary
        champion = replace(ActorContract.champion(), belief_set_source_hash=f.source.metadata.source_hash)
        pool = SimpleNamespace(metadata=lambda: dict(compiled_tree=dict(sha256='d' * 64)))
        captured = {}

        def prepare(catalog, source, pool, *, source_commit):
            captured.update(catalog=catalog, tables=encoder.compile_tables(f.metadata, catalog=catalog))
            return 'prepared'
        inputs = SimpleNamespace(runtime=dict(showdown=dict(root='/unit/showdown', compiled_tree_sha256='d' * 64),
            constructor_pool=dict(path='/unit/pool'), source_commit='a' * 40), plan=plan,
            vocabulary=dict(tokens=list(vocabulary.tokens), oov_buckets=vocabulary.oov_buckets, aliases=dict(vocabulary.aliases)))
        with (mock.patch.object(ActorContract, 'champion', return_value=champion),
              mock.patch('pokezero.randbat.load_gen3_randbat_source_cached', return_value=f.source),
              mock.patch('pokezero.dex.load_showdown_dex_cached', return_value=f.catalog.dex),
              mock.patch('gpu_engine.source_provenance.load_constructor_pool', return_value=pool),
              mock.patch('gpu_engine.source_provenance._pool', return_value=dict(pool_sha256='p' * 64)),
              mock.patch('gpu_engine.v5_device_backend.prepare', side_effect=prepare)):
            self.assertEqual(w.prepare_device_runtime(inputs), ('prepared', pool, None, None))
        return captured

    def test_an_on_plan_compiles_on_tables(self):
        for plan, want in ((SimpleNamespace(pool_sha256='p' * 64, self_moves=True), True),
                           (SimpleNamespace(pool_sha256='p' * 64, self_moves=False), False),
                           (SimpleNamespace(pool_sha256='p' * 64), False)):          # a gate script's bare pool binding
            got = self.prepare(plan)
            self.assertIs(got['catalog'].masks.self_moves, want)
            self.assertEqual(got['tables'].belief.self_moves is not None, want)


@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class RunChecksTests(unittest.TestCase):
    def test_device_backend_refuses_a_plan_that_differs_from_its_tables(self):
        from gpu_engine.v5_device_backend import Prepared, RuntimeBackend
        for catalog_on, plan_on in ((False, True), (True, False)):
            prepared = Prepared(SimpleNamespace(masks=SimpleNamespace(self_moves=catalog_on)), None, None,
                SimpleNamespace(belief=None), 'p' * 64, '{}', '{}', 'c' * 64, 7)
            backend = RuntimeBackend(prepared, None)
            plan = SimpleNamespace(model=SimpleNamespace(arm='B'), pool_sha256='p' * 64, self_moves=plan_on)
            with self.assertRaisesRegex(ValueError, 'differs from the plan declaration'):backend.check_plan(plan)

    def test_run_actor_declarations(self):
        """An on archive in an off run is refused; an on run's own exports are on, and only its initial snapshot (the
        checkpoint it starts from) may be off: the plan's declaration wins."""
        from gpu_engine.private_self_moves import check_run_actor
        on, off = {'self_moves': True}, {}
        for meta, plan_on, iteration, ok in ((off, False, 0, True), (on, True, 3, True), (off, True, -1, True), (on, True, -1, True),
                                             (off, True, 0, False), (on, False, -1, False), (on, False, 2, False)):
            if ok:check_run_actor(meta, plan_self_moves=plan_on, iteration=iteration, context='t')
            else:
                with self.assertRaisesRegex(ValueError, 'differs from the plan'):
                    check_run_actor(meta, plan_self_moves=plan_on, iteration=iteration, context='t')

    def test_fused_actor_load_follows_the_plan(self):
        from gpu_engine import v2_fused
        model = object()
        for meta, plan_on, iteration, ok in (({}, False, 0, True), ({'self_moves': True}, True, 0, True),
                                             ({}, True, -1, True), ({'self_moves': True}, False, -1, False), ({}, True, 0, False)):
            snapshot = SimpleNamespace(archive='a.npz', archive_sha256='a' * 64, checkpoint_sha256='b' * 64, iteration=iteration)
            plan = SimpleNamespace(model=SimpleNamespace(category_vocabulary_sha256='c' * 64,
                label_vocabulary=SimpleNamespace(sha256='d' * 64)), self_moves=plan_on)
            with mock.patch('gpu_engine.v5_model.load', return_value=(model, plan.model, meta)):
                if ok:self.assertIs(v2_fused.load_actor(plan, snapshot, '/unit'), model)
                else:
                    with self.assertRaisesRegex(ValueError, 'differs from the plan'):v2_fused.load_actor(plan, snapshot, '/unit')

    def test_unthreaded_tools_refuse_an_on_archive(self):
        from gpu_engine.private_self_moves import refuse_unthreaded
        refuse_unthreaded({}, context='t');refuse_unthreaded(None, context='t');refuse_unthreaded({'self_moves': False}, context='t')
        with self.assertRaisesRegex(ValueError, 'not threaded through this tool'):refuse_unthreaded({'self_moves': True}, context='t')


@requires_showdown('the synthetic fixture compiles its catalog from a pokemon-showdown checkout')
@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class ServingEndpointTests(unittest.TestCase):
    """V5 serving and evaluation take the archive's declaration (load_factory -> take_declaration): an on archive is served
    and evaluated on, an off archive off. The archive load, the encoder functions and the V5 observation adapter are
    patched; the catalog is the synthetic fixture's (real compile_encoder)."""

    @classmethod
    def setUpClass(cls):
        import test_v2_offline_view_parity as fx
        cls.f = fx.fixture()

    def contract(self, **changes):
        from pokezero.live_protocol_actor import ActorContract, _vocab_sha
        return replace(ActorContract.champion(), belief_set_source_hash=self.f.source.metadata.source_hash,
                       category_vocab_sha256=_vocab_sha(self.f.catalog.vocabulary), **changes)

    def catalog(self, contract):
        from gpu_engine import v4_encoder
        f = self.f
        return v4_encoder.compile_encoder(f.catalog.source, f.catalog.dex, f.catalog.vocabulary,
            checkpoint_sha256=contract.checkpoint_sha256, feature_masks=contract.encoder_masks())

    def loading(self, meta, *, model_sha256, category_sha256, archive_sha256, size):
        """Patches under which load_factory reaches a real Factory: the archive load returns `meta` and a config whose
        bindings the caller's contract and expectations match."""
        from contextlib import ExitStack
        from gpu_engine import v5_live_protocol as v5
        config = SimpleNamespace(metadata=lambda: {'model': model_sha256}, category_vocabulary_sha256=category_sha256,
                                 category_vocab_size=size)
        stack = ExitStack()
        stack.enter_context(mock.patch.object(v5, 'CompiledEncoderFunctions'))
        stack.enter_context(mock.patch.object(v5.model, 'load', return_value=({}, config, dict(meta, weights_archive_sha256=archive_sha256))))
        stack.enter_context(mock.patch.object(v5.observation, 'compile_adapter', side_effect=lambda catalog, model_config: SimpleNamespace(catalog=catalog)))
        stack.enter_context(mock.patch.object(v5.observation, 'check_adapter', side_effect=lambda adapter: adapter.catalog))
        return stack, config

    def test_take_declaration(self):
        from gpu_engine.v5_live_protocol import take_declaration
        contract = self.contract(checkpoint_sha256='a' * 64);catalog = self.catalog(contract)
        self.assertEqual(take_declaration({}, contract=contract, catalog=catalog), (contract, catalog))
        taken, on = take_declaration({'self_moves': True}, contract=contract, catalog=catalog)
        self.assertIs(taken.self_moves, True);self.assertIs(on.masks.self_moves, True);self.assertIsNotNone(on.self_moves)
        self.assertTrue(bool((on.binding == catalog.binding).all()))      # encode-only: the fold binding is unchanged
        self.assertEqual(take_declaration({'self_moves': True}, contract=taken, catalog=on), (taken, on))
        back, off = take_declaration({}, contract=taken, catalog=on)
        self.assertIs(back.self_moves, False);self.assertIs(off.masks.self_moves, False)
        with self.assertRaisesRegex(ValueError, 'must be a bool'):take_declaration({'self_moves': 'on'}, contract=contract, catalog=catalog)

    def test_load_factory_serves_the_archive_as_declared(self):
        from gpu_engine import v5_live_protocol as v5
        from pokezero import v5_contract
        model_sha = 'm' * 8
        contract = self.contract(checkpoint_sha256='a' * 64, model_config_sha256=v5_contract._sha({'model': model_sha}))
        catalog = self.catalog(contract)
        for meta, want in (({'self_moves': True}, True), ({}, False)):
            stack, _ = self.loading(meta, model_sha256=model_sha, category_sha256='c' * 64, archive_sha256='w' * 64,
                                    size=contract.categorical_vocab_size)
            with stack:
                factory = v5.load_factory('archive.npz', catalog=catalog, contract=contract, expected_archive_sha256='w' * 64,
                    expected_checkpoint_sha256='a' * 64, expected_category_vocabulary_sha256='c' * 64,
                    expected_label_vocabulary_sha256='d' * 64, enable_v5=True)
            self.assertIs(factory.contract.self_moves, want)
            self.assertIs(factory.catalog.masks.self_moves, want)
            self.assertEqual(factory.contract.namespace == contract.namespace, not want)

    def test_evaluation_factory_evaluates_the_archive_as_declared(self):
        """v5_evaluation._factory (the owned and the detached evaluator child) through the real load_factory."""
        import hashlib, json
        from pokezero import v5_evaluation as evaluation, v5_run as run, v5_contract
        from pokezero.live_protocol_actor import ActorContract
        from pokezero.v5_phasec import category_binding
        f = self.f;vocabulary = f.catalog.vocabulary;model_sha = 'e' * 8
        champion = self.contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def artifact(name, value):
                blob = json.dumps(value).encode();(root / name).write_bytes(blob)
                return dict(path=str(root / name), sha256=hashlib.sha256(blob).hexdigest())
            category = artifact('vocabulary.json', dict(tokens=list(vocabulary.tokens), oov_buckets=vocabulary.oov_buckets,
                                                        aliases=dict(vocabulary.aliases)))
            runtime = artifact('runtime.json', dict(showdown=dict(root='/unit/showdown', node=dict(path='/unit/node'),
                compiled_tree_sha256='d' * 64), category_vocabulary=category))
            protocol = dict(runtime=runtime, label_vocabulary_sha256='e' * 64, deterministic=True,
                            model_sha256=run._sha({'model': model_sha}))
            request = SimpleNamespace(archive_path=str(root / 'actor.npz'), snapshot=SimpleNamespace(archive_sha256='w' * 64,
                checkpoint_sha256='b' * 64, model_sha256=v5_contract._sha({'model': model_sha})))
            for meta, want in (({'self_moves': True}, True), ({}, False)):
                stack, _ = self.loading(meta, model_sha256=model_sha, category_sha256=category_binding(vocabulary),
                                        archive_sha256='w' * 64, size=champion.categorical_vocab_size)
                with (stack, mock.patch.object(ActorContract, 'champion', return_value=champion),
                      mock.patch('pokezero.v5_worker._source'), mock.patch('pokezero.v5_worker._probe'),
                      mock.patch.object(evaluation.subprocess, 'check_output', return_value=json.dumps({'sha256': 'd' * 64})),
                      mock.patch('pokezero.randbat.load_gen3_randbat_source_cached', return_value=f.source),
                      mock.patch('pokezero.dex.load_showdown_dex_cached', return_value=f.catalog.dex)):
                    factory = evaluation._factory(protocol, request)
                self.assertIs(factory.contract.self_moves, want)
                self.assertIs(factory.catalog.masks.self_moves, want)


if __name__ == '__main__':
    unittest.main()
