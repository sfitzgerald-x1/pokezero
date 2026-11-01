"""Explicit reusable owning runtime. Import does not initialize JAX or a model.

Only pure compiled functions and immutable model/catalog inputs survive a game.
The original runner still creates/closes a new actor/provider/history per game.
"""
from dataclasses import asdict
import os
from pathlib import Path
import weakref


_ENCODERS = weakref.WeakKeyDictionary()
_CACHE = None


def configure_compilation_cache(directory):
    """Enable JAX's original persistent cache before this worker compiles."""
    global _CACHE
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError('explicit absolute worker compilation cache directory required')
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve(strict=True))
    if _CACHE is not None and _CACHE != resolved:
        raise ValueError('worker compilation cache directory changed')
    import jax
    jax.config.update('jax_enable_compilation_cache', True)
    jax.config.update('jax_compilation_cache_dir', resolved)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    _CACHE = resolved
    return dict(directory=resolved, enabled=True, min_compile_time_seconds=0.)


def register_factory(factory, functions):
    # functions is a lookup: the v4 catalog is lazy until its first owning game.
    _ENCODERS[factory] = functions


def compilation_snapshot(factory):
    """Host counters only; no compile/transfer/forward and no cache-hit claim."""
    try:
        lookup = _ENCODERS.get(factory)
    except TypeError:  # Inert legacy test doubles are not registered factories.
        lookup = None
    functions = {} if lookup is None else lookup()
    counters = {}
    for name, function in functions.items():
        size = getattr(function, '_cache_size', None)
        counters[name] = dict(available=callable(size),
            resident_signatures=int(size()) if callable(size) else None)
    return dict(worker_pid=os.getpid(), persistent_cache_directory=_CACHE,
        functions=counters)


class ChampionWorkerRuntime:
    """One worker's raw champion runtime; distinct factories for owning axes.

    All non-cell config values are bound, including precision, source, timeout,
    checkpoint and FP controls. This opt-in path never loads another checkpoint,
    resets a model, or reuses a live game object. Legacy benchmark is unchanged.
    """
    CELL_FIELDS = frozenset(('games', 'seed_start', 'foulplay_random_seed',
        'pokezero_player', 'encoder_kind'))

    def __init__(self):
        self.identity = None
        self.runtime = None
        self.factories = {}

    def _bind(self, config):
        from .live_protocol_bridge import validate_encoder_config
        validate_encoder_config(config)
        if config.encoder_kind not in ('python-owning', 'jax-owning') or config.games != 1:
            raise ValueError('one original owning cell required')
        identity = {k: v for k, v in asdict(config).items() if k not in self.CELL_FIELDS}
        if self.identity is not None and identity != self.identity:
            raise ValueError('worker source/checkpoint/FP/runtime configuration changed')
        return identity

    def prepare(self, config):
        """Original raw checkpoint/env assembly, once per dedicated worker."""
        from . import foulplay_bridge as bridge
        from .live_protocol_runtime import champion_actor_factory, _checkpoint_sha, _require_source
        from .live_protocol_actor import ActorContract
        identity = self._bind(config)
        bridge._validate_external_paths(config)  # Retained on EVERY game.
        contract = ActorContract.champion()
        _require_source(config, contract)
        if _checkpoint_sha(config.checkpoint) != contract.checkpoint_sha256:
            raise ValueError('original champion checkpoint bytes changed')
        if self.runtime is None:
            checkpoint_sha = bridge._checkpoint_sha256(config)
            model, result = bridge.load_transformer_checkpoint(config.checkpoint, map_location=config.device)
            bridge._warn_on_belief_provenance_mismatch(config, result)
            spec = bridge.observation_spec_from_model_config(result.model_config)
            vocab = bridge.category_vocab_from_model_config(result.model_config, config.showdown_root)
            dex = bridge.load_showdown_dex_cached(config.showdown_root)
            masks = bridge.feature_masks_from_model_config(result.model_config)
            env = bridge.LocalShowdownConfig(showdown_root=config.showdown_root,
                node_binary=config.node_binary, observation_spec=spec,
                category_vocab=vocab, feature_masks=masks)
            self.runtime = (model, result, env, spec, vocab, dex, masks, checkpoint_sha)
            self.identity = identity
        if config.encoder_kind not in self.factories:
            self.factories[config.encoder_kind] = champion_actor_factory(config)
        return self.factories[config.encoder_kind]

    async def run(self, config):
        from . import foulplay_bridge as bridge
        from .live_protocol_bridge import require_actor_factory
        factory = self.prepare(config)
        require_actor_factory(config, factory)
        model, result, env, spec, vocab, dex, masks, digest = self.runtime
        if digest != factory.contract.checkpoint_sha256:
            raise ValueError('same original champion checkpoint required')
        policy_id = str(result.model_config.policy_id)
        policy = bridge._build_policy(config=config, model=model, result=result,
            value_model=model, value_result=result, env_config=env,
            rollout_config=bridge.RolloutConfig(max_decision_rounds=config.max_decision_rounds,
                format_id=config.format_id), policy_id=policy_id)
        return await bridge._run_controlled_foulplay_games(config, policy=policy,
            policy_id=policy.policy_id if hasattr(policy, 'policy_id') else policy_id,
            vocab=vocab, dex=dex, observation_spec=spec, feature_masks=masks,
            checkpoint_sha256=digest, owning_actor_factory=factory,
            actor_runtime=(model, result, env))
