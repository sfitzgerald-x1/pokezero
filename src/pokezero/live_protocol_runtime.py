"""Explicit champion owning-encoder assembly; no launcher or qualification.

Call ``champion_actor_factory(config)`` and pass its result as the existing
``owning_actor_factory`` argument to the controlled bridge. The bridge still
loads its ordinary checkpoint once. This module never loads weights, runs a
model, starts a game, or changes process-wide precision settings. B2 and live
B3 scientific admission remain separate gates.
"""
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import shutil
import subprocess

from .live_protocol_actor import (ActorContract, JaxBackend, PythonBackend,
    RawPolicy, _mask_values, _require, _vocab_sha)
from .live_protocol_bridge import OwningActorFactory, require_actor_factory


@dataclass(frozen=True, eq=False)
class CompiledEncoderFunctions:
    """One catalog's pure JAX functions; no owner, request, or battle state.

    There is deliberately no global cache. A factory retains its own holder,
    while each adapter allocates fresh host/device state and registration data.
    """
    catalog: object
    step: object = field(init=False)
    encode_step: object = field(init=False)
    _validate: object = field(init=False, repr=False)

    def __post_init__(self):
        import jax
        import numpy as np
        from gpu_engine import v4_encoder
        catalog = self.catalog
        step = jax.jit(lambda state, chunk: v4_encoder.fold_chunk(state, chunk, catalog))
        encode = jax.jit(lambda state, own, side:
            v4_encoder.encode(state, own, catalog, observer_sides=side))
        binding = np.asarray(catalog.binding).tobytes()
        def vocabulary_rows():
            vocab = catalog.vocabulary
            return (tuple(vocab.tokens), vocab.oov_buckets, tuple(sorted(vocab.aliases.items())),
                    tuple(sorted(vocab._index.items())), vocab._oov_offset)
        namespace = (catalog.checkpoint_sha256, catalog.source.source_hash,
                     vocabulary_rows(), catalog.masks)
        def validate(candidate):
            _require(candidate is catalog and self.catalog is catalog
                and self.step is step and self.encode_step is encode
                and np.asarray(catalog.binding).tobytes() == binding
                and (catalog.checkpoint_sha256, catalog.source.source_hash,
                     vocabulary_rows(), catalog.masks) == namespace,
                'compiled encoder catalog/function namespace drift')
            _require(not jax.config.jax_enable_x64
                and jax.config.jax_default_matmul_precision == 'highest', 'JAX precision drift')
        object.__setattr__(self, 'step', step)
        object.__setattr__(self, 'encode_step', encode)
        object.__setattr__(self, '_validate', validate)
        validate(catalog)

    def require_catalog(self, catalog):
        self._validate(catalog)


def _checkpoint_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _require_source(config, contract):
    """Read-only source checks; no simulator or model invocation."""
    root = Path(config.showdown_root).resolve(strict=True)
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
        '--untracked-files=no'], check=True, capture_output=True, text=True).stdout
    _require(revision == contract.showdown_revision and not dirty,
             'exact clean pinned Showdown source required')
    # Dex's metadata loader invokes `node` through PATH. Refuse a different
    # binary rather than silently binding it to another runtime.
    configured = shutil.which(config.node_binary)
    metadata = shutil.which('node')
    _require(configured is not None and metadata is not None
        and Path(configured).resolve() == Path(metadata).resolve(),
        'Dex metadata Node differs from configured runtime')


def champion_actor_factory(config):
    """Bind one explicit owning axis to the bridge's unchanged raw champion.

    The existing bridge supplies (model, result, env_config). A factory refuses
    later replacement of that runtime, and creates new provider leases and
    transformer histories on every begin_game. Precision must be configured by
    the caller before use; the existing RawPolicy guard checks it unchanged.
    """
    contract = ActorContract.champion()
    # Validate before touching paths or optional JAX dependencies.
    from .live_protocol_bridge import validate_encoder_config
    validate_encoder_config(config)
    _require(config.encoder_kind in ('python-owning', 'jax-owning'), 'explicit owning axis required')
    _require(_checkpoint_sha(config.checkpoint) == contract.checkpoint_sha256,
             'exact champion checkpoint required')
    _require_source(config, contract)
    checkpoint = str(Path(config.checkpoint).resolve())
    kind = config.encoder_kind
    bound = {}

    def backend(c, *, env_config, **identity):
        _require(c == contract, 'factory contract drift')
        _require(env_config.category_vocab is not None
            and _vocab_sha(env_config.category_vocab) == c.category_vocab_sha256
            and env_config.category_vocab.size == c.categorical_vocab_size
            and _mask_values(env_config.feature_masks) == c.feature_masks
            and env_config.observation_spec.schema_version == 'pokezero.observation.v4',
            'factory encoder namespace drift')
        if 'env' not in bound:
            _require(Path(env_config.resolved_showdown_root()) == Path(config.showdown_root).resolve()
                and env_config.node_binary == config.node_binary
                and env_config.set_belief_source in (None, True), 'owning source configuration differs')
            bound['env'] = env_config
            # The bridge pins belief_set_source=True; make the same choice
            # explicit on its process-free owning encoder config.
            bound['own_env'] = replace(env_config, set_belief_source=True)
        _require(env_config is bound['env'], 'factory encoder configuration identity drift')
        if kind == 'python-owning':
            return PythonBackend(c, config=bound['own_env'], **identity)
        if 'functions' not in bound:
            from .dex import load_showdown_dex_cached
            from .randbat import load_gen3_randbat_source_cached
            from gpu_engine import public_belief, v4_encoder
            source = load_gen3_randbat_source_cached(config.showdown_root)
            _require(source.metadata.source_hash == c.belief_set_source_hash, 'JAX source hash differs')
            catalog = v4_encoder.compile_encoder(public_belief.compile_source(source),
                load_showdown_dex_cached(config.showdown_root), env_config.category_vocab,
                checkpoint_sha256=c.checkpoint_sha256, feature_masks=dict(c.feature_masks))
            bound['functions'] = CompiledEncoderFunctions(catalog)
        functions = bound['functions']
        return JaxBackend(c, catalog=functions.catalog, compiled_functions=functions, **identity)

    def policy(c, *, model, result):
        from .neural_policy import TransformerSoftmaxPolicy, require_torch
        _require(c == contract, 'factory contract drift')
        if 'model' not in bound:
            bound['model'], bound['result'] = model, result
        _require(model is bound['model'] and result is bound['result'], 'factory model/result identity drift')
        # TransformerSoftmaxPolicy.__post_init__ calls eval()/to(). Refuse
        # drift first, so fresh wrappers cannot conceal an altered runtime.
        torch = require_torch()
        _require(not model.training and all(t.device.type == 'cpu'
            and (not t.is_floating_point() or t.dtype == torch.float32)
            for t in list(model.parameters()) + list(model.buffers())), 'CPU eval FP32 model required')
        raw = TransformerSoftmaxPolicy(model=model, result=result, deterministic=True,
            sampling_temperature=1., device='cpu', policy_id=result.model_config.policy_id,
            checkpoint_path=checkpoint, weights_sha256=contract.checkpoint_sha256)
        return RawPolicy.from_transformer(c, raw)

    factory = OwningActorFactory(kind, contract, backend, policy)
    return require_actor_factory(config, factory)
