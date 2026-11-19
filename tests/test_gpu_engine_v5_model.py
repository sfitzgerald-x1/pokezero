"""Tiny synthetic Torch/JAX actor parity; no checkpoint/model artifacts or game."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from gpu_engine import v5_model as actor
from pokezero import v5_contract as contract
from pokezero.v5_model import MODEL_INPUTS, TOKEN_ROLES, V5ModelConfig


TORCH_PYTHON = os.environ.get('POKEZERO_TEST_TORCH_PYTHON') or (
    sys.executable if importlib.util.find_spec('torch') else None)


def emit_fixture(directory, arm, replicate):
    """Original frozen Torch eval only, random tiny weights, two public rows."""
    import torch
    from pokezero.v5_model import build_v5_model
    torch.set_num_threads(1); torch.manual_seed(704 + replicate)
    vocabulary = contract.LabelVocabulary('a' * 64, ('surf', 'rest'), (('blissey', 3), ('swalot', 2)))
    config = V5ModelConfig(16, 'b' * 64, vocabulary, arm=arm, replicate=replicate,
        width=8, layers=2, attention_heads=2, expert_width=12, value_hidden=(10, 12), dropout=.4)
    model = build_v5_model(config, enable_v5=True).eval()
    with torch.no_grad():
        if config.treatment: model.value_query.normal_(mean=0, std=.5)
    rng = np.random.default_rng(19)
    arrays = dict(categorical_ids=rng.integers(0, 16, (2, 1, 29, 41), dtype=np.int64),
        numeric_features=rng.normal(0, .2, (2, 1, 29, 132)).astype(np.float32),
        token_type_ids=np.broadcast_to(np.asarray(TOKEN_ROLES, np.int64), (2, 1, 29)).copy(),
        attention_mask=np.ones((2, 1, 29), bool), history_mask=np.ones((2, 1), bool))
    arrays['attention_mask'][0, 0, [2, 8, 16]] = False
    arrays['attention_mask'][1, 0, [4, 10, 20]] = False
    tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
    for precision in ('float32', 'bfloat16'):
        context = torch.autocast('cpu', dtype=torch.bfloat16) if precision == 'bfloat16' else nullcontext()
        with torch.no_grad(), context:
            output = model(**tensors)
        for name in ('policy_logits', 'value', 'value_logits'):
            value = getattr(output, name)
            if value is not None: arrays[precision + '_' + name] = value.float().numpy()
    metadata = dict(schema=actor.CHECKPOINT_SCHEMA, training_schema=contract.TRAINING_CONTRACT_SCHEMA,
        observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5, model=config.metadata(),
        source_contract=contract.contract_descriptor(), source_contract_sha256=actor._sha(contract.contract_descriptor()),
        category_vocabulary_sha256=config.category_vocabulary_sha256,
        label_vocabulary_sha256=config.label_vocabulary.sha256)
    payload = dict(metadata=metadata, metadata_sha256=actor._sha(metadata), state_dict=model.state_dict())
    directory = Path(directory); checkpoint = directory / 'synthetic.pt'
    torch.save(payload, checkpoint); digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    from pokezero.checkpoint_self_moves import _writer_receipt,off_from_receipt
    proof=off_from_receipt(_writer_receipt(checkpoint,checkpoint_sha256=digest,
        metadata_sha256=payload['metadata_sha256'],self_moves=False))
    archive = directory / 'actor.npz'
    receipt = actor.export(checkpoint, archive, expected_checkpoint_sha256=digest, enable_v5=True,off_provenance=proof)
    (directory / 'export-receipt.json').write_text(json.dumps(receipt))
    arrays['checkpoint_sha256'] = np.frombuffer(digest.encode(), np.uint8)
    np.savez_compressed(directory / 'fixture.npz', **arrays)
    # Check export failures in the runtime that owns safe Torch deserialization.
    rejected = []
    for name in ('legacy', 'namespace', 'shape', 'nonfinite', 'aux_missing', 'value_centers', 'center_drift'):
        changed = dict(payload, state_dict=dict(payload['state_dict']))
        if name == 'legacy': changed = dict(model_config={}, state_dict={})
        elif name == 'namespace':
            changed['metadata'] = dict(metadata, category_vocabulary_sha256='f' * 64)
            changed['metadata_sha256'] = actor._sha(changed['metadata'])
        elif name == 'shape': changed['state_dict']['policy_head.weight'] = torch.zeros(2, 8)
        elif name == 'nonfinite': changed['state_dict']['policy_head.bias'] = torch.full((1,), float('nan'))
        elif name == 'aux_missing':
            if not config.treatment: continue
            del changed['state_dict']['hidden_variant_head.weight']
        elif name == 'value_centers': changed['state_dict']['value_centers'] = torch.zeros(config.value_bins)
        else: changed['state_dict']['value_centers'] = torch.nextafter(model.value_centers, torch.ones_like(model.value_centers))
        bad = directory / 'bad.pt'; torch.save(changed, bad)
        bad_digest=hashlib.sha256(bad.read_bytes()).hexdigest()
        # Preserve each original shape/namespace/nonfinite rejection's scope.
        bad_proof=None
        if name!='legacy':
            proof_path=directory/(name+'-writer.pt')
            bad_proof=off_from_receipt(_writer_receipt(proof_path,checkpoint_sha256=bad_digest,
                metadata_sha256=changed['metadata_sha256'],self_moves=False))
        try: actor.export(bad, directory / 'must-not-exist.npz',
            expected_checkpoint_sha256=bad_digest, enable_v5=True,off_provenance=bad_proof)
        except ValueError: rejected.append(name)
        else: raise AssertionError('export accepted ' + name)
    (directory / 'rejections.json').write_text(json.dumps(rejected))
    # All temporary artifacts are tiny untrained fixtures, never real weights.
    for path in directory.iterdir():
        if path.stat().st_size >= 100 * 1024: raise AssertionError('fixture exceeds100KiB: ' + str(path))
    checkpoint.unlink(); (directory / 'bad.pt').unlink()


@unittest.skipUnless(importlib.util.find_spec('jax') and TORCH_PYTHON, 'requires optional JAX and Torch runtimes')
class V5JaxActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls.jax = jax
        cls.temporary = tempfile.TemporaryDirectory(prefix='pokezero-v5-jax-synthetic-')
        cls.fixtures = []
        root = Path(cls.temporary.name)
        for arm, replicate in (('A', 0), ('B', 0), ('D', 0), ('B', 1)):
            directory = root / (arm + str(replicate)); directory.mkdir()
            result = subprocess.run([TORCH_PYTHON, str(Path(__file__).resolve()), '--fixture',
                str(directory), arm, str(replicate)], text=True, capture_output=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='src:tests:.'))
            if result.returncode: raise AssertionError(result.stdout + result.stderr)
            with np.load(directory / 'fixture.npz', allow_pickle=False) as archive:
                data = {name: archive[name] for name in archive.files}
            digest = data.pop('checkpoint_sha256').tobytes().decode()
            with np.load(directory / 'actor.npz', allow_pickle=False) as archive:
                metadata = json.loads(archive['metadata_json'].tobytes())
            receipt = json.loads((directory / 'export-receipt.json').read_text())
            params, config, loaded = actor.load(directory / 'actor.npz', enable_v5=True,
                expected_archive_sha256=receipt['weights_archive_sha256'],
                expected_checkpoint_sha256=digest, expected_category_vocabulary_sha256='b' * 64,
                expected_label_vocabulary_sha256=metadata['model']['label_vocabulary_sha256'])
            columns = {name: data[name] for name in MODEL_INPUTS}
            columns.update(schema=contract.TRAINING_CONTRACT_SCHEMA,
                observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5,
                category_vocabulary_sha256='b' * 64, label_vocabulary_sha256=config.label_vocabulary.sha256)
            cls.fixtures.append((directory, params, config, columns, data, loaded))

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_original_torch_fp32_policy_value_logits_and_argmax_all_initial_arms(self):
        for _, params, config, columns, data, _ in self.fixtures:
            with self.subTest(arm=config.arm, replicate=config.replicate):
                result = actor.forward(params, actor.prepare_inputs(columns, config), config)
                self.assertTrue(np.asarray(result.valid).all())
                for name in ('policy_logits', 'value', 'value_logits'):
                    actual = getattr(result, name)
                    if actual is None: self.assertEqual(config.arm, 'A'); continue
                    np.testing.assert_allclose(actual, data['float32_' + name], atol=3e-6, rtol=3e-6)
                np.testing.assert_array_equal(np.argmax(result.policy_logits, -1),
                    np.argmax(data['float32_policy_logits'], -1))

    def test_original_torch_explicit_bfloat16_policy_value_logits_and_argmax(self):
        for _, params, config, columns, data, _ in self.fixtures:
            with self.subTest(arm=config.arm, replicate=config.replicate):
                result = actor.forward(params, actor.prepare_inputs(columns, config), config, compute_dtype='bfloat16')
                for name in ('policy_logits', 'value', 'value_logits'):
                    actual = getattr(result, name)
                    if actual is not None:
                        tolerance = .0002 if name == 'value' else .004
                        np.testing.assert_allclose(actual, data['bfloat16_' + name], atol=tolerance, rtol=.004)
                np.testing.assert_array_equal(np.argmax(result.policy_logits, -1),
                    np.argmax(data['bfloat16_policy_logits'], -1))

    def test_jit_batch_separation_masked_tokens_and_dynamic_invalid_rows(self):
        import jax.numpy as j
        _, params, config, columns, _, _ = self.fixtures[2]
        inputs = actor.prepare_inputs(columns, config)
        compiled = self.jax.jit(lambda values: actor.forward(params, values, config))
        baseline = compiled(inputs)
        changed = dict(inputs)
        changed['numeric_features'] = inputs['numeric_features'].at[0, 0, 2].set(12345.)
        changed['categorical_ids'] = inputs['categorical_ids'].at[0, 0, 2].set(15)
        modified = compiled(changed)
        np.testing.assert_allclose(modified.policy_logits, baseline.policy_logits, atol=0, rtol=0)
        np.testing.assert_allclose(modified.value, baseline.value, atol=0, rtol=0)
        for key, update in (
                ('categorical_ids', lambda x: x.at[0, 0, 0, 0].set(config.category_vocab_size)),
                ('token_type_ids', lambda x: x.at[0, 0, 24].set(2)),
                ('numeric_features', lambda x: x.at[0, 0, 0, 0].set(j.nan)),
                ('attention_mask', lambda x: x.at[0, 0, 26].set(False)),
                ('history_mask', lambda x: x.at[0, 0].set(False))):
            broken = dict(inputs); broken[key] = update(inputs[key]); output = compiled(broken)
            np.testing.assert_array_equal(output.valid, [False, True])
            self.assertTrue(np.isnan(output.policy_logits[0]).all())
            self.assertTrue(np.isnan(output.value[0]))
            np.testing.assert_array_equal(output.policy_logits[1], baseline.policy_logits[1])
        one = {name: value[1:] for name, value in inputs.items()}
        output = actor.forward(params, one, config)
        np.testing.assert_allclose(output.policy_logits[0], baseline.policy_logits[1], atol=2e-6)
        bf16 = self.jax.jit(lambda values: actor.forward(params, values, config, compute_dtype='bfloat16'))(inputs)
        self.assertTrue(np.isfinite(bf16.policy_logits).all())

    def test_public_namespace_shape_dtype_roles_and_all_six_beliefs(self):
        _, params, config, columns, _, _ = self.fixtures[0]
        self.assertNotIn('opponent_semantic_head.weight', params)
        for key, value in (('category_vocabulary_sha256', 'f' * 64), ('label_vocabulary_sha256', 'f' * 64),
                ('schema', 'pokezero.training_contract.v4'), ('view', 'PRIVILEGED')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                actor.prepare_inputs(dict(columns, **{key: value}), config)
        for name, value in (('numeric_features', columns['numeric_features'].astype(np.float64)),
                ('categorical_ids', columns['categorical_ids'].astype(np.float32)),
                ('attention_mask', columns['attention_mask'].astype(np.int32)),
                ('history_mask', np.zeros((2, 1), bool)),
                ('numeric_features', columns['numeric_features'][:, :, :23])):
            with self.subTest(name=name), self.assertRaises(ValueError):
                actor.prepare_inputs(dict(columns, **{name: value}), config)
        for index in range(23, 29):
            mask = columns['attention_mask'].copy(); mask[1, 0, index] = False
            with self.assertRaises(ValueError):actor.prepare_inputs(dict(columns, attention_mask=mask), config)
        # Label/private sidecars are not actor inputs, including arbitrary poison.
        prepared = actor.prepare_inputs(dict(columns, privileged_numeric_features=object(),
            hidden_set_targets=object()), config)
        self.assertEqual(set(prepared), set(MODEL_INPUTS))
        with self.assertRaises(ValueError):actor.forward(params, prepared, config, compute_dtype='float16')

    def test_explicit_export_rejections_and_archive_namespaces_and_tensor_digests(self):
        for directory, params, config, _, _, metadata in self.fixtures:
            expected = {'legacy', 'namespace', 'shape', 'nonfinite', 'value_centers', 'center_drift'}
            if config.treatment: expected.add('aux_missing')
            self.assertEqual(set(json.loads((directory / 'rejections.json').read_text())), expected)
            self.assertFalse(any('head' in key and key.startswith(('hidden_', 'opponent_', 'survival', 'final_hp')) for key in params))
            kwargs = dict(enable_v5=True, expected_checkpoint_sha256=metadata['checkpoint_sha256'],
                expected_archive_sha256=metadata['weights_archive_sha256'],
                expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=config.label_vocabulary.sha256)
            for key in ('expected_archive_sha256', 'expected_checkpoint_sha256',
                    'expected_category_vocabulary_sha256', 'expected_label_vocabulary_sha256'):
                with self.assertRaises(ValueError):actor.load(directory / 'actor.npz', **dict(kwargs, **{key: 'f' * 64}))
            with self.assertRaises(ValueError):actor.load('/never-read', **dict(kwargs, enable_v5=False))
            with self.assertRaises(ValueError):actor.export('/never-read', '/never-written', expected_checkpoint_sha256='a' * 64)
        directory, _, _, _, _, metadata = self.fixtures[2]
        with np.load(directory / 'actor.npz', allow_pickle=False) as source:
            changed = {name: source[name] for name in source.files}
        changed['policy_head.bias'] = changed['policy_head.bias'] + 1
        bad = directory / 'bad-archive.npz'; np.savez_compressed(bad, **changed)
        with self.assertRaisesRegex(ValueError, 'archive SHA256'):
            actor.load(bad, enable_v5=True, expected_checkpoint_sha256=metadata['checkpoint_sha256'],
                expected_archive_sha256=metadata['weights_archive_sha256'],
                expected_category_vocabulary_sha256='b' * 64,
                expected_label_vocabulary_sha256=metadata['model']['label_vocabulary_sha256'])

    def test_recomputed_internal_tensor_hash_cannot_rebind_trusted_archive(self):
        directory, _, config, _, _, metadata = self.fixtures[2]
        with np.load(directory / 'actor.npz', allow_pickle=False) as archive:
            changed = {name: archive[name] for name in archive.files}
        inner = json.loads(changed['metadata_json'].tobytes())
        changed['policy_head.bias'] = changed['policy_head.bias'] + 1
        inner['tensor_sha256']['policy_head.bias'] = hashlib.sha256(changed['policy_head.bias'].tobytes()).hexdigest()
        changed['metadata_json'] = np.frombuffer(json.dumps(inner, sort_keys=True).encode(), np.uint8)
        path = directory / 'rehashed.npz'; np.savez_compressed(path, **changed)
        with self.assertRaisesRegex(ValueError, 'archive SHA256'):
            actor.load(path, enable_v5=True, expected_archive_sha256=metadata['weights_archive_sha256'],
                expected_checkpoint_sha256=metadata['checkpoint_sha256'],
                expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=config.label_vocabulary.sha256)


    def _archive_variant(self, producer, mutate=None, *, name='compat-variant.npz'):
        directory, _, config, _, _, receipt = self.fixtures[2]
        with np.load(directory / 'actor.npz', allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files if key != 'metadata_json'}
            metadata = json.loads(source['metadata_json'].tobytes())
        if producer is None:
            metadata.pop('torch_source_sha256')
        else:
            metadata['torch_source_sha256'] = producer
        if mutate is not None:
            mutate(metadata, arrays)
        arrays['metadata_json'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), np.uint8)
        path = directory / name
        np.savez_compressed(path, **arrays)
        return path, dict(enable_v5=True,
            expected_archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_checkpoint_sha256=receipt['checkpoint_sha256'],
            expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
            expected_label_vocabulary_sha256=config.label_vocabulary.sha256)

    def test_current_export_and_bounded_old_archive_provenance_are_distinct(self):
        directory, original, config, _, _, receipt = self.fixtures[2]
        self.assertEqual(receipt['torch_source_sha256'], actor.TORCH_SOURCE_SHA256)
        self.assertEqual(receipt['torch_source_sha256'],
            hashlib.sha256(Path(__import__('pokezero.v5_model', fromlist=['x']).__file__).read_bytes()).hexdigest())
        for producer in (actor.TORCH_SOURCE_SHA256, actor.LEGACY_ARCHIVE_TORCH_SOURCE_SHA256):
            with self.subTest(producer=producer):
                path, kwargs = self._archive_variant(producer)
                before = path.read_bytes()
                params, loaded_config, metadata = actor.load(path, **kwargs)
                self.assertEqual(loaded_config, config)
                self.assertEqual(metadata['torch_source_sha256'], producer)
                self.assertEqual(metadata['loader_torch_source_sha256'], actor.TORCH_SOURCE_SHA256)
                self.assertEqual(metadata['loader_contract_source_sha256'], actor.CONTRACT_SOURCE_SHA256)
                self.assertEqual(metadata['weights_archive_sha256'], kwargs['expected_archive_sha256'])
                self.assertEqual(path.read_bytes(), before)
                for name in params:
                    np.testing.assert_array_equal(params[name], original[name])
        for producer in (None, '0' * 64, [], {}):
            with self.subTest(rejected_producer=producer):
                path, kwargs = self._archive_variant(producer)
                with self.assertRaisesRegex(ValueError, 'public v5 actor archive'):
                    actor.load(path, **kwargs)

    def test_installed_source_guard_stays_strict_for_both_entrypoints(self):
        import pokezero.v5_model as original
        model_bytes = Path(original.__file__).read_bytes()
        contract_bytes = Path(contract.__file__).read_bytes()
        sha256 = hashlib.sha256
        kwargs = dict(enable_v5=True, expected_archive_sha256='a' * 64,
            expected_checkpoint_sha256='b' * 64, expected_category_vocabulary_sha256='c' * 64,
            expected_label_vocabulary_sha256='d' * 64)
        for changed_bytes, fake_digest in ((model_bytes, actor.LEGACY_ARCHIVE_TORCH_SOURCE_SHA256),
                (model_bytes, '0' * 64), (contract_bytes, '0' * 64)):
            def altered(value, raw=changed_bytes, digest=fake_digest):
                return SimpleNamespace(hexdigest=lambda: digest) if value == raw else sha256(value)
            with self.subTest(digest=fake_digest, model=changed_bytes == model_bytes), \
                    mock.patch.object(actor.hashlib, 'sha256', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'Torch/contract source differs'):
                    actor.load('/never-read', **kwargs)
                with self.assertRaisesRegex(ValueError, 'Torch/contract source differs'):
                    actor.export('/never-read', '/never-written', expected_checkpoint_sha256='b' * 64, enable_v5=True)

    def test_every_archive_invariant_rejects_for_both_compatible_producers(self):
        changes = [(key, lambda metadata, arrays, key=key, value=value: metadata.update({key:value}))
            for key, value in (('schema','unknown'), ('actor','unknown'), ('view','PRIVATE'),
                ('auxiliary_heads',True), ('evaluation',False), ('checkpoint_sha256','e'*64),
                ('contract_source_sha256','e'*64), ('source_contract_sha256','e'*64),
                ('parameter_count',-1), ('tensor_sha256',{}))]
        changes.extend([
            ('category', lambda m,a: m['model'].update(category_vocabulary_sha256='e'*64)),
            ('labels', lambda m,a: m['model'].update(label_vocabulary_sha256='e'*64)),
            ('missing_tensor', lambda m,a: a.pop('policy_head.bias')),
            ('extra_tensor', lambda m,a: a.update(unexpected=np.zeros(1,np.float32))),
            ('shape', lambda m,a: a.update({'policy_head.bias':np.zeros(2,np.float32)})),
            ('dtype', lambda m,a: a.update({'policy_head.bias':a['policy_head.bias'].astype(np.float64)})),
            ('nonfinite', lambda m,a: a.update({'policy_head.bias':np.full_like(a['policy_head.bias'],np.nan)})),
            ('value_centers', lambda m,a: a.update(value_centers=np.zeros_like(a['value_centers']))),
        ])
        for producer in (actor.TORCH_SOURCE_SHA256, actor.LEGACY_ARCHIVE_TORCH_SOURCE_SHA256):
            for name, change in changes:
                with self.subTest(producer=producer, invalid=name):
                    path, kwargs = self._archive_variant(producer, change)
                    # Trust the changed outer SHA here, so each inner invariant is exercised.
                    with self.assertRaises(ValueError):
                        actor.load(path, **kwargs)

    def test_real_production_and_historical_wrappers_keep_all_bindings(self):
        from gpu_engine import v2_fused, v5_device_backend
        _, _, config, _, _, _ = self.fixtures[2]
        plan = SimpleNamespace(model=config)
        backend = object.__new__(v5_device_backend.RuntimeBackend)
        backend.actors = {}; backend.model_sha256 = actor._sha(config.metadata())
        def resolve(name, digest):
            path = self.fixtures[2][0] / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            return path
        store = SimpleNamespace(plan=plan, resolve=resolve)
        for index, producer in enumerate((actor.TORCH_SOURCE_SHA256, actor.LEGACY_ARCHIVE_TORCH_SOURCE_SHA256)):
            path, kwargs = self._archive_variant(producer, name=f'wrapper-{index}.npz')
            snapshot = dict(archive=path.name, archive_sha256=kwargs['expected_archive_sha256'],
                checkpoint_sha256=kwargs['expected_checkpoint_sha256'], model_sha256=backend.model_sha256)
            fused = v2_fused.load_actor(plan, SimpleNamespace(**snapshot), path.parent)
            actual = backend._actor(snapshot, store)
            for name in actual:
                np.testing.assert_array_equal(actual[name], fused[name])
            with self.assertRaisesRegex(ValueError, 'cached actor checkpoint provenance'):
                backend._actor(dict(snapshot, checkpoint_sha256='e'*64), store)
            changed_plan = SimpleNamespace(model=replace(config, dropout=.3))
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                v2_fused.load_actor(changed_plan, SimpleNamespace(**snapshot), path.parent)
            other = object.__new__(v5_device_backend.RuntimeBackend)
            other.actors = {}; other.model_sha256 = actor._sha(changed_plan.model.metadata())
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                other._actor(dict(snapshot,model_sha256=other.model_sha256),
                    SimpleNamespace(plan=changed_plan,resolve=resolve))

    def test_fresh_load_without_torch_for_both_archive_producers(self):
        entries = []
        for index, producer in enumerate((actor.TORCH_SOURCE_SHA256, actor.LEGACY_ARCHIVE_TORCH_SOURCE_SHA256)):
            path, kwargs = self._archive_variant(producer, name=f'no-torch-{index}.npz')
            entries.append((str(path), kwargs, producer))
        program = """import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise ModuleNotFoundError('Torch disabled for load-only test',name='torch')
sys.meta_path.insert(0,NoTorch())
from gpu_engine import v5_model as actor
for path,kwargs,producer in json.loads(sys.argv[1]):
 params,config,metadata=actor.load(path,**kwargs)
 assert params and metadata['torch_source_sha256']==producer
 assert metadata['loader_torch_source_sha256']==actor.TORCH_SOURCE_SHA256
assert 'torch' not in sys.modules
print('LOAD_ONLY_WITHOUT_TORCH_PASS')
"""
        result = subprocess.run([sys.executable,'-c',program,json.dumps(entries)],capture_output=True,text=True,
            env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONPATH='src:tests:.'),timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('LOAD_ONLY_WITHOUT_TORCH_PASS',result.stdout)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fixture': emit_fixture(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else: unittest.main()
