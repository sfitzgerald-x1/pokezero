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


def write_archive(directory, name, meta):
    """A minimal npz carrying only metadata_json (what archive_self_moves reads); returns (path, sha256)."""
    import hashlib, io, json
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metadata_json=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8))
    path = Path(directory) / name;path.write_bytes(buffer.getvalue())
    return str(path), hashlib.sha256(buffer.getvalue()).hexdigest()


@requires_showdown('the synthetic fixture compiles its catalog from a pokemon-showdown checkout')
@unittest.skipUnless(importlib.util.find_spec('jax'), 'requires isolated JAX')
class ServingEndpointTests(unittest.TestCase):
    """The V5 Foul Play endpoints take the archive's declaration: an on archive is served (and evaluated) on, an off
    archive off. load_factory is patched to capture what it is handed (it re-checks archive == contract == catalog)."""

    @classmethod
    def setUpClass(cls):
        import test_v2_offline_view_parity as fx
        cls.f = fx.fixture()

    def contract(self):
        from pokezero.live_protocol_actor import ActorContract, _vocab_sha
        return replace(ActorContract.champion(), belief_set_source_hash=self.f.source.metadata.source_hash,
                       category_vocab_sha256=_vocab_sha(self.f.catalog.vocabulary))

    def test_archive_self_moves_reads_the_digest_bound_declaration(self):
        from gpu_engine.v5_live_protocol import archive_self_moves
        with tempfile.TemporaryDirectory() as tmp:
            for meta, want in (({}, False), ({'self_moves': True}, True), ({'self_moves': False}, False)):
                path, digest = write_archive(tmp, f'a-{want}-{len(meta)}.npz', dict(meta, model='x'))
                self.assertIs(archive_self_moves(path, expected_archive_sha256=digest), want)
            path, digest = write_archive(tmp, 'bad.npz', {'self_moves': 1})
            with self.assertRaisesRegex(ValueError, 'must be a bool'):archive_self_moves(path, expected_archive_sha256=digest)
            with self.assertRaisesRegex(ValueError, 'bytes differ'):archive_self_moves(path, expected_archive_sha256='0' * 64)

    def test_foulplay_load_and_run_serves_the_archive_as_declared(self):
        import asyncio
        from pokezero import v5_foulplay as fp
        from gpu_engine import v4_encoder
        f = self.f;contract = self.contract()
        catalog = v4_encoder.compile_encoder(f.catalog.source, f.catalog.dex, f.catalog.vocabulary,
            checkpoint_sha256=f.catalog.checkpoint_sha256, feature_masks=contract.encoder_masks())
        with tempfile.TemporaryDirectory() as tmp:
            for meta, want in (({'self_moves': True}, True), ({}, False)):
                path, digest = write_archive(tmp, f'a-{want}.npz', meta)
                with (mock.patch('gpu_engine.v5_live_protocol.load_factory', return_value='factory') as load,
                      mock.patch.object(fp, 'run_panel', new=mock.AsyncMock(return_value='report'))):
                    self.assertEqual(asyncio.run(fp.load_and_run('config', path, catalog=catalog, contract=contract,
                        expected_archive_sha256=digest, expected_checkpoint_sha256='b' * 64, expected_category_vocabulary_sha256='c' * 64,
                        expected_label_vocabulary_sha256='d' * 64, enable_v5=True)), 'report')
                served = load.call_args.kwargs
                self.assertIs(served['contract'].self_moves, want)
                self.assertIs(served['catalog'].masks.self_moves, want)
                self.assertEqual(served['catalog'].self_moves is not None, want)
                # encode-only: the fold binding is the off catalog's either way
                self.assertTrue(bool((served['catalog'].binding == catalog.binding).all()))

    def test_evaluation_factory_evaluates_the_archive_as_declared(self):
        import json
        from pokezero import v5_evaluation as evaluation, v5_run as run
        from pokezero.live_protocol_actor import ActorContract
        f = self.f;vocabulary = f.catalog.vocabulary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def artifact(name, value):
                import hashlib
                blob = json.dumps(value).encode();(root / name).write_bytes(blob)
                return dict(path=str(root / name), sha256=hashlib.sha256(blob).hexdigest())
            category = artifact('vocabulary.json', dict(tokens=list(vocabulary.tokens), oov_buckets=vocabulary.oov_buckets,
                                                        aliases=dict(vocabulary.aliases)))
            runtime = artifact('runtime.json', dict(showdown=dict(root='/unit/showdown', node=dict(path='/unit/node'),
                compiled_tree_sha256='d' * 64), category_vocabulary=category))
            config = SimpleNamespace(metadata=lambda: {'model': 'unit'})
            protocol = dict(runtime=runtime, label_vocabulary_sha256='e' * 64, deterministic=True, model_sha256=run._sha({'model': 'unit'}))
            for meta, want in (({'self_moves': True}, True), ({}, False)):
                path, digest = write_archive(tmp, f'a-{want}.npz', meta)
                request = SimpleNamespace(archive_path=path, snapshot=SimpleNamespace(archive_sha256=digest,
                    checkpoint_sha256='b' * 64, model_sha256='c' * 64))
                with (mock.patch.object(ActorContract, 'champion', return_value=self.contract()),
                      mock.patch('pokezero.v5_worker._source'), mock.patch('pokezero.v5_worker._probe'),
                      mock.patch.object(evaluation.subprocess, 'check_output', return_value=json.dumps({'sha256': 'd' * 64})),
                      mock.patch('pokezero.randbat.load_gen3_randbat_source_cached', return_value=f.source),
                      mock.patch('pokezero.dex.load_showdown_dex_cached', return_value=f.catalog.dex),
                      mock.patch('gpu_engine.v5_live_protocol.load_factory', return_value=SimpleNamespace(config=config)) as load):
                    evaluation._factory(protocol, request)
                served = load.call_args.kwargs
                self.assertIs(served['contract'].self_moves, want)
                self.assertIs(served['catalog'].masks.self_moves, want)


if __name__ == '__main__':
    unittest.main()
