"""Explicit public v5 actor export and functional inference; no legacy dispatch.

This mirrors the frozen Torch actor in evaluation mode. Auxiliary/private heads
are never exported or evaluated. Host preparation validates public inputs;
device inference also returns a row validity mask (invalid rows contain NaNs).
It does not construct observations, load a champion, or qualify a game pipeline.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pokezero import v5_contract as contract
from pokezero.v5_model import EXPERT_ROLES, MODEL_INPUTS, TOKEN_ROLES, V5ModelConfig


ARCHIVE_SCHEMA = 'pokezero.gpu_actor.v5'
CHECKPOINT_SCHEMA = 'pokezero.checkpoint.v5'
TORCH_SOURCE_SHA256 = 'f593bc5f41fa72d12d78fa57c8ff94b9cadff824f3a3a52cc026743fbd6e3d9b'
CONTRACT_SOURCE_SHA256 = 'b6a823195350ff3fed68699ea70bd925c4e9df1ff4f65e222928098d75080170'


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _snapshot(path):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno()); blob = stream.read(); after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or len(blob) != after.st_size:
        raise ValueError('v5 weight file changed while reading')
    return blob, hashlib.sha256(blob).hexdigest()


def _sources():
    import pokezero.v5_model as original
    actual = (hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest())
    if actual != (TORCH_SOURCE_SHA256, CONTRACT_SOURCE_SHA256):
        raise ValueError('frozen v5 Torch/contract source differs')


def shapes(config, *, include_auxiliary=False):
    """Exact original Torch names, including its persisted value-center buffer."""
    if type(config) is not V5ModelConfig:
        raise ValueError('explicit v5 model configuration required')
    c = config; d = c.width
    result = {'category_embedding.weight': (c.category_vocab_size, d),
        'token_type_embedding.weight': (16, d), 'history_position_embedding.weight': (1, d),
        'numeric_projection.weight': (d, 132), 'numeric_projection.bias': (d,),
        'final_norm.weight': (d,), 'policy_head.weight': (1, d), 'policy_head.bias': (1,),
        'value_centers': (c.value_bins,)}
    if c.treatment:
        result['value_query'] = (d,)
        for i, (n, m) in enumerate(((d, c.value_hidden[0]), (c.value_hidden[0], c.value_hidden[1]),
                (c.value_hidden[1], c.value_bins))):
            result[f'value_head.{i * 2}.weight'] = (m, n)
            result[f'value_head.{i * 2}.bias'] = (m,)
        if include_auxiliary:
            for name, count in (('opponent_semantic_head', len(c.label_vocabulary.opponent_classes)),
                    ('hidden_species_head', len(c.label_vocabulary.species)), ('hidden_variant_head', 64),
                    ('survival_head', 1), ('final_hp_head', 1)):
                result[name + '.weight'] = (count, d); result[name + '.bias'] = (count,)
    else:
        result.update({'value_head.weight': (1, d), 'value_head.bias': (1,)})
    for i in range(c.layers):
        p = f'encoder.layers.{i}.'
        result.update({p + 'norm1.weight': (d,), p + 'norm2.weight': (d,),
            p + 'self_attn.in_proj_weight': (3 * d, d), p + 'self_attn.in_proj_bias': (3 * d,),
            p + 'self_attn.out_proj.weight': (d, d), p + 'self_attn.out_proj.bias': (d,)})
        for expert in range(3):
            q = p + f'role_experts.experts.{expert}.'
            result.update({q + '0.weight': (c.expert_width, d), q + '0.bias': (c.expert_width,),
                q + '2.weight': (d, c.expert_width), q + '2.bias': (d,)})
    return result


def validate_weights(params, config, *, include_auxiliary=False):
    expected = shapes(config, include_auxiliary=include_auxiliary)
    if set(params) != set(expected):
        raise ValueError('v5 tensor keys differ from declared architecture')
    for name, shape in expected.items():
        a = np.asarray(params[name])
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('finite float32 v5 tensor shape required: ' + name)
    # The buffer is part of the literal original HL-Gauss expected-value head.
    edges = np.linspace(-1, 1, config.value_bins + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    if not np.allclose(params['value_centers'], centers, rtol=0, atol=2e-7):
        raise ValueError('original v5 value-bin centers required')


def _config(metadata):
    config = V5ModelConfig.from_metadata(metadata)
    if _sha(config.metadata()) != _sha(metadata):
        raise ValueError('exact original v5 model metadata required')
    return config


def export(checkpoint, out, *, expected_checkpoint_sha256, enable_v5=False):
    """Export an explicit v5 checkpoint snapshot; Torch is imported only here."""
    if enable_v5 is not True:
        raise ValueError('explicit v5 actor enablement required')
    contract._binding(expected_checkpoint_sha256); _sources()
    blob, digest = _snapshot(checkpoint)
    if digest != expected_checkpoint_sha256:
        raise ValueError('original v5 checkpoint SHA256 differs')
    import torch
    payload = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    metadata = payload.get('metadata', {})
    if (metadata.get('schema') != CHECKPOINT_SCHEMA
            or payload.get('metadata_sha256') != _sha(metadata)
            or metadata.get('training_schema') != contract.TRAINING_CONTRACT_SCHEMA
            or metadata.get('observation_schema') != contract.OBSERVATION_SCHEMA_VERSION_V5
            or metadata.get('source_contract_sha256') != _sha(metadata.get('source_contract'))
            or metadata.get('source_contract_sha256') != _sha(contract.contract_descriptor())):
        raise ValueError('explicit intact same-source v5 checkpoint required')
    config = _config(metadata['model'])
    if (metadata.get('category_vocabulary_sha256') != config.category_vocabulary_sha256
            or metadata.get('label_vocabulary_sha256') != config.label_vocabulary.sha256):
        raise ValueError('v5 checkpoint category/label namespace differs')
    state = payload['state_dict']
    if any(not torch.is_tensor(value) or value.dtype != torch.float32 for value in state.values()):
        raise ValueError('original float32 v5 state_dict required')
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    validate_weights(arrays, config, include_auxiliary=True)
    edges = torch.linspace(-1, 1, config.value_bins + 1)
    if not torch.equal(state['value_centers'], (edges[:-1] + edges[1:]) / 2):
        raise ValueError('exact original Torch v5 value-bin centers required')
    params = {name: arrays[name] for name in shapes(config)}
    result = dict(schema=ARCHIVE_SCHEMA, model=config.metadata(), checkpoint_sha256=digest,
        checkpoint_metadata_sha256=payload['metadata_sha256'],
        source_contract_sha256=metadata['source_contract_sha256'],
        torch_source_sha256=TORCH_SOURCE_SHA256, contract_source_sha256=CONTRACT_SOURCE_SHA256,
        actor='RAW_POLICY', view='PUBLIC', auxiliary_heads=False, evaluation=True,
        parameter_count=sum(a.size for a in params.values()),
        tensor_sha256={name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in params.items()})
    encoded = np.frombuffer(json.dumps(result, sort_keys=True, allow_nan=False).encode(), np.uint8)
    archive = io.BytesIO()
    np.savez_compressed(archive, metadata_json=encoded, **params)
    blob = archive.getvalue()
    with Path(out).open('xb') as stream:
        stream.write(blob)
    # This digest belongs to the export receipt, not self-declared NPZ content.
    return dict(result, weights_archive_sha256=hashlib.sha256(blob).hexdigest())


def load(path, *, expected_archive_sha256, expected_checkpoint_sha256, expected_category_vocabulary_sha256,
        expected_label_vocabulary_sha256, enable_v5=False):
    """Load an opt-in actor archive with explicit caller-selected namespaces."""
    if enable_v5 is not True:
        raise ValueError('explicit v5 actor enablement required')
    for value in (expected_archive_sha256, expected_checkpoint_sha256,
            expected_category_vocabulary_sha256, expected_label_vocabulary_sha256):
        contract._binding(value)
    _sources(); blob, digest = _snapshot(path)
    if digest != expected_archive_sha256:
        raise ValueError('caller-trusted v5 actor archive SHA256 differs')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata_json'].tobytes())
        params = {name: archive[name] for name in archive.files if name != 'metadata_json'}
    if (metadata.get('schema') != ARCHIVE_SCHEMA or metadata.get('actor') != 'RAW_POLICY'
            or metadata.get('view') != 'PUBLIC' or metadata.get('auxiliary_heads') is not False
            or metadata.get('evaluation') is not True
            or metadata.get('checkpoint_sha256') != expected_checkpoint_sha256
            or metadata.get('torch_source_sha256') != TORCH_SOURCE_SHA256
            or metadata.get('contract_source_sha256') != CONTRACT_SOURCE_SHA256
            or metadata.get('source_contract_sha256') != _sha(contract.contract_descriptor())):
        raise ValueError('explicit original public v5 actor archive required')
    config = _config(metadata['model'])
    if (config.category_vocabulary_sha256 != expected_category_vocabulary_sha256
            or config.label_vocabulary.sha256 != expected_label_vocabulary_sha256):
        raise ValueError('v5 actor category/label namespace differs')
    validate_weights(params, config)
    if (metadata.get('parameter_count') != sum(a.size for a in params.values())
            or metadata.get('tensor_sha256') != {name: hashlib.sha256(a.tobytes()).hexdigest()
                for name, a in params.items()}):
        raise ValueError('v5 actor tensor digest/count differs')
    import jax.numpy as j
    return {name: j.asarray(value) for name, value in params.items()}, config, dict(metadata, weights_archive_sha256=digest)


def prepare_inputs(columns, config):
    """Validate host public columns, then select only the five actor inputs.

    Labels and optional training-only sidecars never become model arguments.
    This uses the same Bx1x29 layout and valid frame rules as original Torch.
    """
    from pokezero.v5_model import require_v5_bindings
    require_v5_bindings(columns, config)
    if columns.get('view', 'PUBLIC') != 'PUBLIC':
        raise ValueError('v5 actor requires the public view')
    arrays = {name: np.asarray(columns[name]) for name in MODEL_INPUTS}
    b = arrays['categorical_ids'].shape[0]
    _input_shapes(arrays, b)
    cats, numbers, roles, mask, history = (arrays[name] for name in MODEL_INPUTS)
    if (not np.isfinite(numbers).all() or (cats < 0).any() or (cats >= config.category_vocab_size).any()
            or not history.all() or not mask[:, :, 23:29].all() or not (roles == np.asarray(TOKEN_ROLES)).all()):
        raise ValueError('original finite public categories/roles and six attended beliefs required')
    import jax.numpy as j
    # JAX defaults to 32-bit indices. No clipping or category remapping occurs.
    return {name: j.asarray(value, dtype=j.int32 if name in ('categorical_ids', 'token_type_ids') else None)
        for name, value in arrays.items()}


def _input_shapes(inputs, batch):
    if set(inputs) != set(MODEL_INPUTS) or batch <= 0:
        raise ValueError('five typed public v5 actor tensors required')
    expected = ((batch, 1, 29, 41), (batch, 1, 29, 132), (batch, 1, 29), (batch, 1, 29), (batch, 1))
    for name, shape in zip(MODEL_INPUTS, expected):
        a = inputs[name]
        valid_dtype = (np.dtype(a.dtype) in (np.dtype('int32'), np.dtype('int64'))
            if name in ('categorical_ids', 'token_type_ids') else np.dtype(a.dtype) ==
            (np.dtype('float32') if name == 'numeric_features' else np.dtype('bool')))
        if a.shape != shape or not valid_dtype:
            raise ValueError('v5 public tensor shape/dtype differs: ' + name)


class Output(NamedTuple):
    policy_logits: object
    value: object
    value_logits: object  # None for A; raw HL-Gauss logits for B/D.
    valid: object


def forward(params, inputs, config, *, compute_dtype='float32'):
    """Jittable evaluation-only actor; precision is explicit and static.

    FP32 embeddings/residuals/RMS/softmax mirror Torch eval. BF16 rounds linear
    outputs and attention operands/results as Torch autocast does; neither path
    executes dropout or auxiliary heads. Dynamic invalid rows remain invalid
    under JIT instead of being silently clipped into another observation.
    """
    if compute_dtype not in ('float32', 'bfloat16') or type(config) is not V5ModelConfig:
        raise ValueError('explicit v5 config and float32/bfloat16 precision required')
    import jax
    import jax.numpy as j
    b = inputs['categorical_ids'].shape[0]; _input_shapes(inputs, b)
    if set(params) != set(shapes(config)):
        raise ValueError('inference-only v5 tensor keys required')
    for name, shape in shapes(config).items():
        if params[name].shape != shape or params[name].dtype != j.float32:
            raise ValueError('v5 actor parameter shape/dtype differs: ' + name)
    c = config; dtype = j.bfloat16 if compute_dtype == 'bfloat16' else j.float32
    cats, numeric, roles, attention, history = (inputs[name] for name in MODEL_INPUTS)
    valid = (j.all(j.isfinite(numeric), axis=(1, 2, 3))
        & j.all((cats >= 0) & (cats < c.category_vocab_size), axis=(1, 2, 3))
        & j.all(roles == j.asarray(TOKEN_ROLES), axis=(1, 2)) & history[:, 0]
        & j.all(attention[:, :, 23:29], axis=(1, 2)))
    mask = attention[:, 0]
    # Safe local indexing for invalid rows; outputs are poisoned below.
    safe_cats = j.clip(cats[:, 0], 0, c.category_vocab_size - 1)
    safe_roles = j.clip(roles[:, 0], 0, 15)
    safe_numeric = j.where(j.isfinite(numeric[:, 0]), numeric[:, 0], 0)
    safe_mask = mask.at[:, 23:29].set(True)

    def dot(a, w):
        return j.matmul(a.astype(dtype), w.astype(dtype), precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=j.float32)

    def linear(a, name):
        return (dot(a, params[name + '.weight'].T) + params[name + '.bias'].astype(dtype).astype(j.float32)).astype(dtype)

    def norm(a, name):
        full = a.astype(j.float32)
        return (full * jax.lax.rsqrt(j.mean(full * full, axis=-1, keepdims=True) + c.norm_epsilon)).astype(a.dtype) * params[name + '.weight'].astype(a.dtype)

    x = (j.sum(params['category_embedding.weight'][safe_cats], axis=-2)
        + linear(safe_numeric, 'numeric_projection').astype(j.float32)
        + params['token_type_embedding.weight'][safe_roles] + params['history_position_embedding.weight'][0])
    x = j.where(safe_mask[..., None], x, 0)
    for layer in range(c.layers):
        p = f'encoder.layers.{layer}.'; a = norm(x, p + 'norm1')
        packed = (dot(a, params[p + 'self_attn.in_proj_weight'].T)
            + params[p + 'self_attn.in_proj_bias'].astype(dtype).astype(j.float32)).astype(dtype)
        q, k, v = [part.reshape((b, 29, c.attention_heads, c.width // c.attention_heads)).transpose((0, 2, 1, 3))
            for part in j.split(packed, 3, axis=-1)]
        scores = dot(q, k.swapaxes(-1, -2)) / j.sqrt(j.asarray(c.width // c.attention_heads, j.float32))
        weights = jax.nn.softmax(j.where(safe_mask[:, None, None, :], scores, -j.inf), axis=-1)
        attended = dot(weights, v).astype(dtype).transpose((0, 2, 1, 3)).reshape((b, 29, c.width))
        x = x + linear(attended, p + 'self_attn.out_proj').astype(j.float32)
        a = norm(x, p + 'norm2'); update = j.zeros_like(x)
        for expert in range(3):
            indices = np.asarray([i for i, role in enumerate(EXPERT_ROLES) if role == expert], np.int32)
            prefix = p + f'role_experts.experts.{expert}.'
            z = jax.nn.gelu(linear(a[:, indices], prefix + '0'), approximate=False)
            update = update.at[:, indices].set(linear(z, prefix + '2').astype(j.float32))
        x = j.where(safe_mask[..., None], x + update, 0)
    encoded = j.where(safe_mask[..., None], norm(x, 'final_norm'), 0)
    mean = j.sum(encoded * (safe_mask.astype(j.float32) / j.sum(safe_mask, axis=-1, keepdims=True))[..., None], axis=1)
    policy = linear(encoded[:, 13:22], 'policy_head')[..., 0]
    if c.treatment:
        attention_scores = dot(encoded, params['value_query']).astype(dtype).astype(j.float32) / j.sqrt(j.asarray(c.width, j.float32))
        attention_weights = jax.nn.softmax(j.where(safe_mask, attention_scores, -j.finfo(j.float32).max), axis=-1)
        pooled = j.sum(encoded * attention_weights[..., None].astype(encoded.dtype), axis=1)
        logits = linear(jax.nn.gelu(linear(jax.nn.gelu(linear(pooled, 'value_head.0'), approximate=False),
            'value_head.2'), approximate=False), 'value_head.4')
        value = j.sum(jax.nn.softmax(logits.astype(j.float32), axis=-1) * params['value_centers'], axis=-1)
        logits = j.where(valid[:, None], logits.astype(j.float32), j.nan)
    else:
        logits = None
        value = j.tanh(linear(mean, 'value_head')[..., 0]).astype(j.float32)
    return Output(j.where(valid[:, None], policy.astype(j.float32), j.nan),
        j.where(valid, value, j.nan), logits, valid)
