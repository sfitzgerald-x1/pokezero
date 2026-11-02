"""Snapshot-bound v4 B2 runtime for the two declared canary endpoints only.

The production9375 factory stays unchanged. This explicit actor version keeps
its source/category/model/mask contract and substitutes only the hash of one
of the endpoint's immutable checkpoints. Each worker retains two model/code
holders; every game still creates fresh original B2 providers and histories.
"""
import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
import contextlib
import json
import os
from pathlib import Path
import sys
import weakref

from .live_protocol_bridge import OwningActorFactory

_ORIGINS = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class Factory(OwningActorFactory):
    pass


def check_factory(config, factory):
    from .live_protocol_actor import ActorContract, _require
    from .live_protocol_runtime import _checkpoint_sha
    origin = _ORIGINS.get(factory) if type(factory) is Factory else None
    _require(origin is not None and config.encoder_kind == 'jax-owning' and factory.encoder_kind == 'jax-owning',
        'explicit canary B2 factory required')
    owner, checkpoint, contract, backend, policy = origin
    _require(owner() is factory and factory.contract == contract and factory.backend_factory is backend and factory.policy_factory is policy
        and str(Path(config.checkpoint).resolve()) == checkpoint and _checkpoint_sha(checkpoint) == contract.checkpoint_sha256
        and replace(contract, checkpoint_sha256=ActorContract.champion().checkpoint_sha256) == ActorContract.champion(),
        'same bound original-v4 endpoint snapshot/source required')


class Runtime:
    CELL_FIELDS = frozenset(('games', 'seed_start', 'foulplay_random_seed', 'pokezero_player', 'foulplay_cpu_affinity'))
    def __init__(self): self.bound = {}
    def prepare(self, config, *, expected_checkpoint_sha256):
        from . import foulplay_bridge as bridge
        from .live_protocol_actor import ActorContract, JaxBackend, RawPolicy, _sha, _canonical, _vocab_sha, _mask_values, _require
        from .live_protocol_bridge import validate_encoder_config
        from .live_protocol_runtime import _checkpoint_sha, _require_source, CompiledEncoderFunctions
        validate_encoder_config(config)
        _require(config.encoder_kind == 'jax-owning' and config.games == 1, 'one B2 canary endpoint cell required')
        bridge._validate_external_paths(config)
        contract = replace(ActorContract.champion(), checkpoint_sha256=expected_checkpoint_sha256)
        checkpoint = str(Path(config.checkpoint).resolve())
        _require(_checkpoint_sha(checkpoint) == expected_checkpoint_sha256, 'declared checkpoint bytes differ')
        _require_source(config, contract)
        identity = {k: v for k, v in asdict(config).items() if k not in self.CELL_FIELDS}
        if checkpoint in self.bound:
            entry = self.bound[checkpoint]
            _require(entry['identity'] == identity and entry['contract'] == contract, 'worker snapshot/runtime changed')
            check_factory(config, entry['factory']); return entry
        _require(len(self.bound)<2,'only CONTROL and final CANARY snapshots fit one worker')
        model, result = bridge.load_transformer_checkpoint(checkpoint, map_location='cpu')
        mc = result.model_config
        _require(_sha(_canonical(mc.to_dict())) == contract.model_config_sha256,
            'fine tuning cannot alter original9375 model configuration')
        bridge._warn_on_belief_provenance_mismatch(config, result)
        spec = bridge.observation_spec_from_model_config(mc)
        vocab = bridge.category_vocab_from_model_config(mc, config.showdown_root)
        masks = bridge.feature_masks_from_model_config(mc)
        _require(_vocab_sha(vocab) == contract.category_vocab_sha256 and _mask_values(masks) == contract.feature_masks,
            'original category and every feature mask required')
        env = bridge.LocalShowdownConfig(showdown_root=config.showdown_root, node_binary=config.node_binary,
            observation_spec=spec, category_vocab=vocab, feature_masks=masks, set_belief_source=True)
        from .dex import load_showdown_dex_cached
        from .randbat import load_gen3_randbat_source_cached
        from gpu_engine import public_belief, v4_encoder
        source = load_gen3_randbat_source_cached(config.showdown_root)
        _require(source.metadata.source_hash == contract.belief_set_source_hash, 'same original public source required')
        catalog = v4_encoder.compile_encoder(public_belief.compile_source(source), load_showdown_dex_cached(config.showdown_root),
            vocab, checkpoint_sha256=contract.checkpoint_sha256, feature_masks=dict(contract.feature_masks))
        functions = CompiledEncoderFunctions(catalog)
        def backend(c, *, env_config, **identity):
            _require(c == contract and env_config is env, 'fresh game requires same current B2 runtime')
            functions.require_catalog(catalog)
            return JaxBackend(c, catalog=catalog, compiled_functions=functions, **identity)
        def policy(c, *, model: object, result: object):
            _require(c == contract and model is bound_model and result is bound_result, 'same actual original model/result')
            from .neural_policy import TransformerSoftmaxPolicy, require_torch
            torch = require_torch()
            _require(not model.training and all(t.device.type == 'cpu' and
                (not t.is_floating_point() or t.dtype == torch.float32)
                for t in list(model.parameters()) + list(model.buffers())), 'current CPU eval FP32 model required')
            raw = TransformerSoftmaxPolicy(model=model, result=result, deterministic=True, sampling_temperature=1.,
                device='cpu', policy_id=result.model_config.policy_id, checkpoint_path=checkpoint,
                weights_sha256=contract.checkpoint_sha256)
            return RawPolicy.from_transformer(c, raw)
        bound_model, bound_result = model, result
        factory = Factory('jax-owning', contract, backend, policy)
        _ORIGINS[factory] = (weakref.ref(factory), checkpoint, contract, backend, policy)
        from .owning_worker_runtime import register_factory
        register_factory(factory, lambda: {'fold': functions.step, 'encode': functions.encode_step})
        entry = dict(identity=identity, contract=contract, model=model, result=result, env=env, spec=spec, vocab=vocab,
            dex=load_showdown_dex_cached(config.showdown_root), masks=masks, factory=factory)
        self.bound[checkpoint] = entry
        return entry
    async def run(self, config, *, expected_checkpoint_sha256):
        from . import foulplay_bridge as bridge
        entry = self.prepare(config, expected_checkpoint_sha256=expected_checkpoint_sha256)
        factory = entry['factory']; check_factory(config, factory)
        model, result, env = (entry[k] for k in ('model', 'result', 'env'))
        policy = bridge._build_policy(config=config, model=model, result=result, value_model=model, value_result=result,
            env_config=env, rollout_config=bridge.RolloutConfig(max_decision_rounds=config.max_decision_rounds,
                format_id=config.format_id), policy_id=result.model_config.policy_id)
        return await bridge._run_controlled_foulplay_games(config, policy=policy, policy_id=policy.policy_id,
            vocab=entry['vocab'], dex=entry['dex'], observation_spec=entry['spec'], feature_masks=entry['masks'],
            checkpoint_sha256=expected_checkpoint_sha256, owning_actor_factory=factory,
            actor_runtime=(model, result, env), actor_version='canary-v4')


async def serve(pair):
    from .b4_canary_endpoint import pin_owned_actor, worker
    pin_owned_actor(pair)  # Before numerical/model imports and worker threads.
    os.environ['JAX_PLATFORMS']='cpu'; os.environ['CUDA_VISIBLE_DEVICES']=''
    import jax, torch
    jax.config.update('jax_enable_x64', False); jax.config.update('jax_default_matmul_precision', 'highest')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest'); torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    runtime = Runtime(); output = sys.stdout
    for line in sys.stdin:
        request = json.loads(line)
        if set(request) != {'path', 'sha256'}: raise ValueError('bounded endpoint request required')
        with contextlib.redirect_stdout(sys.stderr):
            code = await worker(request['path'], request['sha256'], runtime=runtime, expected_pair=pair)
        output.write(json.dumps({'exit': code}) + '\n'); output.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--cpu-pair', nargs=2, type=int, required=True)
    args = parser.parse_args(); asyncio.run(serve(tuple(args.cpu_pair)))
