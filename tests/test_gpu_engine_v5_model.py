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

import numpy as np

from gpu_engine import v5_model as actor
from pokezero import v5_contract as contract
from pokezero.v5_model import MODEL_INPUTS, TOKEN_ROLES, V5ModelConfig


_LOCAL_TORCH = '/Users/scott/workspace/agents/pokezero-confusion-before-substitute-clean/.venv/bin/python'
TORCH_PYTHON = os.environ.get('POKEZERO_TEST_TORCH_PYTHON') or (
    sys.executable if importlib.util.find_spec('torch') else _LOCAL_TORCH if Path(_LOCAL_TORCH).is_file() else None)


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
    archive = directory / 'actor.npz'
    actor.export(checkpoint, archive, expected_checkpoint_sha256=digest, enable_v5=True)
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
        try: actor.export(bad, directory / 'must-not-exist.npz',
            expected_checkpoint_sha256=hashlib.sha256(bad.read_bytes()).hexdigest(), enable_v5=True)
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
            params, config, loaded = actor.load(directory / 'actor.npz', enable_v5=True,
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
                expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=config.label_vocabulary.sha256)
            for key in ('expected_checkpoint_sha256', 'expected_category_vocabulary_sha256', 'expected_label_vocabulary_sha256'):
                with self.assertRaises(ValueError):actor.load(directory / 'actor.npz', **dict(kwargs, **{key: 'f' * 64}))
            with self.assertRaises(ValueError):actor.load('/never-read', **dict(kwargs, enable_v5=False))
            with self.assertRaises(ValueError):actor.export('/never-read', '/never-written', expected_checkpoint_sha256='a' * 64)
        directory, _, _, _, _, metadata = self.fixtures[2]
        with np.load(directory / 'actor.npz', allow_pickle=False) as source:
            changed = {name: source[name] for name in source.files}
        changed['policy_head.bias'] = changed['policy_head.bias'] + 1
        bad = directory / 'bad-archive.npz'; np.savez_compressed(bad, **changed)
        with self.assertRaisesRegex(ValueError, 'digest'):
            actor.load(bad, enable_v5=True, expected_checkpoint_sha256=metadata['checkpoint_sha256'],
                expected_category_vocabulary_sha256='b' * 64,
                expected_label_vocabulary_sha256=metadata['model']['label_vocabulary_sha256'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fixture': emit_fixture(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else: unittest.main()
