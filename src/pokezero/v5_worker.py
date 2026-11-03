"""Private opt-in Phase C worker binding, without job or bank generation.

Inputs are immutable caller-selected declarations. Import and inspection do not
construct an engine/model. The CLI has no default panel executor: production
execution requires the separately reviewed original evaluation hook.
"""
import argparse
from collections import Counter
from dataclasses import dataclass,fields,replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SCHEMA='pokezero.phasec.runtime.v1'
ENV_KEYS=('XLA_PYTHON_CLIENT_PREALLOCATE','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_ALLOCATOR')


def _keys(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('complete exact runtime declaration fields required')


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('explicit SHA256 required')


def _file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _absolute(path):
    if type(path) is not str or not Path(path).is_absolute():raise ValueError('explicit absolute artifact path required')
    return path


def _record(record,*,json_value=True):
    _keys(record,('path','sha256'));_absolute(record['path']);_digest(record['sha256'])
    if not json_value:
        if _file_sha(record['path'])!=record['sha256']:raise ValueError('bound artifact digest differs')
        return None
    blob=Path(record['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=record['sha256']:raise ValueError('bound artifact digest differs')
    return json.loads(blob)


@dataclass(frozen=True)
class Inputs:
    plan:object
    profile:object
    runtime_json:str
    vocabulary_json:str
    gender_policy_json:str
    declaration_sha256:str

    @property
    def runtime(self):return json.loads(self.runtime_json)
    @property
    def vocabulary(self):return json.loads(self.vocabulary_json)
    @property
    def gender_policy(self):return json.loads(self.gender_policy_json)


def _plan(document,profile,arm,category_vocabulary):
    from . import v5_run as run,v5_training as training,v5_phasec as phase
    from .v5_model import V5ModelConfig
    from .v5_loss import V5LossConfig
    model=V5ModelConfig.from_metadata(document['model'])
    if model.arm+str(model.replicate)!=arm:raise ValueError('selected arm differs from original declaration')
    loss=V5LossConfig(**{f.name:document['loss'][f.name] for f in fields(V5LossConfig)})
    # These controls are mandatory production choices, never dataclass defaults.
    _keys(document['train'],(f.name for f in fields(training.TrainConfig)))
    train=training.TrainConfig(**document['train'])
    iterations=[]
    for iteration in document['iterations']:
        batches=[]
        for batch in iteration:
            _keys(batch,(f.name for f in fields(run.Batch)))
            games=[]
            for game in batch['games']:
                _keys(game,(f.name for f in fields(run.Game)));games.append(run.Game(**game))
            batches.append(run.Batch(**dict(batch,games=tuple(games))))
        iterations.append(tuple(batches))
    extras={}
    if 'evaluation' in document:
        if not hasattr(run,'EvaluationProtocol'):raise run.MissingDependency('reviewed milestone source is not integrated')
        extras['evaluation']=run.EvaluationProtocol(document['evaluation']['protocol_sha256'])
    result=run.Plan(document['run_id'],model,loss,train,tuple(iterations),document['pool_sha256'],
        document['selection_seed'],document['historical_fraction'],document['historical_window'],
        document['max_requests'],run.Watchdog(**document['watchdog']),
        checkpoint_updates=document['checkpoint_updates'],discount=document['discount'],**extras)
    if run._json(result.manifest())!=run._json(document):raise ValueError('full original plan differs; no default or discarded fields')
    expected=phase.bind_plan(profile,model.label_vocabulary,category_vocabulary=category_vocabulary,run_id=result.run_id,arm=model.arm,
        replicate=model.replicate,iterations=result.iterations,pool_sha256=result.pool_sha256,selection_seed=result.selection_seed)
    if extras:expected=replace(expected,**extras)
    if expected!=result:raise ValueError('declared controls differ from Phase C direct9375 recipe')
    return result


def _runtime_metadata(runtime_document,*,expected_pool_sha256):
    """Shared original artifact validation, without a training Plan."""
    _keys(runtime_document,('schema','source_commit','control_inventory','category_vocabulary','constructor_pool',
        'gender_policy','showdown','versions','memory_policy','precision_policy'))
    if runtime_document['schema']!=SCHEMA or re.fullmatch('[0-9a-f]{40}',runtime_document['source_commit']) is None:
        raise ValueError('explicit original worker source commit required')
    from . import v5_phasec as phase
    inventory=runtime_document['control_inventory'];_record(inventory)
    profile=phase.ChampionProfile.read(inventory['path'],expected_sha256=inventory['sha256'])
    vocabulary=_record(runtime_document['category_vocabulary'])
    _record(runtime_document['constructor_pool'],json_value=False)
    if runtime_document['constructor_pool']['sha256']!=expected_pool_sha256:
        raise ValueError('original declared constructor pool differs')
    gender=None if runtime_document['gender_policy'] is None else _record(runtime_document['gender_policy'])
    show=runtime_document['showdown'];_keys(show,('root','commit','compiled_tree_sha256','node'))
    _absolute(show['root']);_digest(show['compiled_tree_sha256']);_record(show['node'],json_value=False)
    from .live_protocol_actor import ActorContract
    from .live_protocol_actor import _vocab_sha
    from .category_vocab import CategoryVocabulary
    contract=ActorContract.champion()
    if (show['commit']!=contract.showdown_revision or vocabulary['checkpoint_sha256']!=contract.checkpoint_sha256
            or vocabulary['category_vocab_sha256']!=contract.category_vocab_sha256
            or vocabulary['source_hash']!=contract.belief_set_source_hash):
        raise ValueError('original champion category/source provenance differs')
    resolved=CategoryVocabulary(tuple(vocabulary['tokens']),vocabulary['oov_buckets'],vocabulary['aliases'])
    if _vocab_sha(resolved)!=contract.category_vocab_sha256 or resolved.size!=contract.categorical_vocab_size:
        raise ValueError('literal original category vocabulary differs')
    _keys(runtime_document['versions'],('python','jax','numpy','torch','torch_cuda'))
    if any(type(v) is not str or not v for v in runtime_document['versions'].values()):raise ValueError('explicit runtime versions required')
    _memory(runtime_document['memory_policy'],apply=False)
    precision=runtime_document['precision_policy'];_keys(precision,('training','inference','jax_x64','jax_matmul','torch_matmul'))
    if (precision['training']!='bfloat16' or precision['inference']!='bfloat16'
            or precision['jax_x64'] is not False or precision['jax_matmul']!='highest'
            or precision['torch_matmul'] not in ('highest','high','medium')):
        raise ValueError('explicit common bf16 precision/runtime arithmetic policy required')
    return profile,vocabulary,gender,resolved


def load_inputs(*,declaration,expected_declaration_sha256,runtime,expected_runtime_sha256,arm,enable_v5=False):
    """Hash/read only; no source compiler, engine constructor or model allocation."""
    if enable_v5 is not True:raise ValueError('explicit v5 worker enablement required before artifact access')
    declaration_document=_record(dict(path=str(declaration),sha256=expected_declaration_sha256))
    runtime_document=_record(dict(path=str(runtime),sha256=expected_runtime_sha256))
    profile,vocabulary,gender,resolved=_runtime_metadata(runtime_document,expected_pool_sha256=declaration_document['pool_sha256'])
    plan=_plan(declaration_document,profile,arm,resolved)
    uses_gender=any(b.gender_seed is not None for iteration in plan.iterations for b in iteration)
    if uses_gender!=(gender is not None):raise ValueError('explicit gender policy must match declared gender recipe')
    return Inputs(plan,profile,json.dumps(runtime_document,sort_keys=True,allow_nan=False),
        json.dumps(vocabulary,sort_keys=True,allow_nan=False),json.dumps(gender,sort_keys=True,allow_nan=False),expected_declaration_sha256)


def _memory(policy,*,apply):
    """No guessed JAX/Torch sharing policy; bind literal environment and fraction."""
    _keys(policy,('environment','torch_fraction'));env=policy['environment'];_keys(env,ENV_KEYS)
    if env[ENV_KEYS[0]] not in ('true','false'):raise ValueError('explicit JAX preallocation choice required')
    fraction=env[ENV_KEYS[1]]
    if fraction is not None:
        if type(fraction) is not str or not math.isfinite(float(fraction)) or not 0<float(fraction)<=1:raise ValueError('explicit JAX memory fraction required')
    if env[ENV_KEYS[2]] not in (None,'default','platform'):raise ValueError('explicit supported JAX allocator choice required')
    tf=policy['torch_fraction']
    if tf is not None and (type(tf) not in (int,float) or not math.isfinite(tf) or not 0<tf<=1):raise ValueError('explicit Torch memory fraction required')
    if apply:
        if sys.modules.get('jax') is not None:raise ValueError('bound memory policy must precede JAX imports')
        # pokezero.__init__ legitimately imports Torch through neural_policy.
        # A cold module import allocates no CUDA state; initialized/unknown
        # device state cannot accept a late memory policy.
        torch_module=sys.modules.get('torch')
        if torch_module is not None:
            initialized=getattr(getattr(torch_module,'cuda',None),'is_initialized',None)
            if not callable(initialized) or initialized() is not False:
                raise ValueError('bound memory policy must precede Torch CUDA initialization')
        for key,value in env.items():
            if key in os.environ and os.environ[key]!=value:raise ValueError('runtime memory environment differs from declaration')
            if value is not None:os.environ[key]=value


def _probe(runtime):
    import jax,numpy as np,torch
    versions=dict(python='.'.join(map(str,sys.version_info[:3])),jax=jax.__version__,numpy=np.__version__,
        torch=torch.__version__,torch_cuda=torch.version.cuda)
    if versions!=runtime['versions']:raise ValueError('bound combined JAX/Torch runtime differs')
    devices=jax.devices()
    if (len(devices)!=1 or devices[0].platform!='gpu' or 'GB300' not in devices[0].device_kind
            or not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'GB300' not in torch.cuda.get_device_name(0)
            or not torch.cuda.is_bf16_supported()):
        raise ValueError('one same visible GB300 per worker required')
    precision=runtime['precision_policy']
    jax.config.update('jax_enable_x64',precision['jax_x64']);jax.config.update('jax_default_matmul_precision',precision['jax_matmul'])
    torch.set_float32_matmul_precision(precision['torch_matmul'])
    fraction=runtime['memory_policy']['torch_fraction']
    if fraction is not None:torch.cuda.set_per_process_memory_fraction(fraction,0)


def _source(runtime):
    root=Path(__file__).resolve().parents[2]
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if git('rev-parse','HEAD')!=runtime['source_commit'] or git('status','--porcelain','--','gpu_engine','src/pokezero'):
        raise ValueError('original clean worker/engine source differs')
    show=runtime['showdown']
    _record(show['node'],json_value=False)
    found=shutil.which('node')
    if found is None or Path(found).resolve()!=Path(show['node']['path']).resolve():
        raise ValueError('PATH Node must be the declared digest-bound executable')
    if subprocess.check_output(['git','-C',show['root'],'rev-parse','HEAD'],text=True).strip()!=show['commit']:
        raise ValueError('original Showdown source differs')


def prepare_runtime(inputs):
    """Original training wrapper; compilation is shared with nontraining capture."""
    return _prepare_runtime(inputs.runtime,inputs.vocabulary,expected_pool_sha256=inputs.plan.pool_sha256)


def prepare_device_runtime(inputs):
    """Static device preparation only; no retained-FULL driver or actor load."""
    from .category_vocab import CategoryVocabulary
    from .dex import load_showdown_dex_cached
    from .randbat import load_gen3_randbat_source_cached
    from .live_protocol_actor import ActorContract
    from gpu_engine import source_provenance,public_belief,v4_encoder,v5_device_backend
    runtime=inputs.runtime;v=inputs.vocabulary;show=runtime['showdown'];contract=ActorContract.champion()
    vocabulary=CategoryVocabulary(tuple(v['tokens']),v['oov_buckets'],v['aliases'])
    source=load_gen3_randbat_source_cached(show['root']);dex=load_showdown_dex_cached(show['root'])
    if source.metadata.source_hash!=contract.belief_set_source_hash:raise ValueError('original public belief source differs')
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),dex,vocabulary,
        checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    pool=source_provenance.load_constructor_pool(runtime['constructor_pool']['path']);metadata=pool.metadata()
    if (source_provenance._pool(pool,rehash=True)['pool_sha256']!=inputs.plan.pool_sha256
            or metadata['compiled_tree']['sha256']!=show['compiled_tree_sha256']):
        raise ValueError('original constructor source binding differs')
    prepared=v5_device_backend.prepare(catalog,source,pool,source_commit=runtime['source_commit'])
    return prepared,pool,None,None


def prepare_capture_runtime(bank_reference,*,runtime,expected_runtime_sha256,enable_v5=False):
    """Prepare the real NONTRAINING bank without a Plan, actor or constructor.

    bank_reference uses the original capture schema's path/SHA256 record.
    Return (prepared, pool, public_config, contract, gender_policy), ready for
    the original actor loader and collect_declared_bank. This call compiles
    original source; it does not load models or execute the declared games.
    """
    if enable_v5 is not True:raise ValueError('explicit v5 capture enablement required before artifact access')
    runtime_document=_record(dict(path=str(runtime),sha256=expected_runtime_sha256))
    from . import v5_evaluation_capture as capture
    bank=capture.load_bank(bank_reference)
    heldout=capture._artifact(bank.document['heldout_roster']).document()
    _,vocabulary,gender,_=_runtime_metadata(runtime_document,expected_pool_sha256=heldout['pool_sha256'])
    uses_gender=any(b.gender_seed is not None for b in bank.batches)
    if uses_gender!=(gender is not None):raise ValueError('explicit gender policy must match declared gender recipe')
    _source(runtime_document);_memory(runtime_document['memory_policy'],apply=True);_probe(runtime_document)
    prepared,pool,public,contract=_prepare_runtime(runtime_document,vocabulary,expected_pool_sha256=heldout['pool_sha256'])
    return prepared,pool,public,contract,gender


def _prepare_runtime(runtime,vocabulary,*,expected_pool_sha256):
    """Original checked compilation only. Never load champion weights or teams."""
    from .category_vocab import CategoryVocabulary
    from .dex import load_showdown_dex_cached
    from .randbat import load_gen3_randbat_source_cached
    from .live_protocol_actor import ActorContract
    from .observation import ObservationFeatureMasks
    from .showdown import V4_REPLAY_OBSERVATION_SPEC
    from .local_showdown import LocalShowdownConfig
    from gpu_engine import (source_provenance,public_belief,v4_encoder,transition,
        engine_public_events as events,engine_request_source as requests,engine_own_facts as own,
        engine_leftovers_callbacks as leftovers,engine_status_residual_callbacks as status,
        engine_direct_hit_controls as hit,engine_substitute_callbacks as substitute,engine_toxic_callbacks as toxic,
        engine_recoil_callbacks as recoil,engine_calm_mind_callbacks as calm,engine_swords_dance_callbacks as sword,
        engine_spikes_callbacks as spikes,engine_thunder_wave_callbacks as thunder,engine_sleep_powder_callbacks as powder,engine_retained_encoder_driver as driver)
    import jax.numpy as j
    from gpu_engine.mechanics import SLP
    show=runtime['showdown'];contract=ActorContract.champion();v=vocabulary
    vocabulary=CategoryVocabulary(tuple(v['tokens']),v['oov_buckets'],v['aliases'])
    belief=load_gen3_randbat_source_cached(show['root']);dex=load_showdown_dex_cached(show['root'])
    if belief.metadata.source_hash!=contract.belief_set_source_hash:raise ValueError('original public belief source differs')
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(belief),dex,vocabulary,
        checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    pool=source_provenance.load_constructor_pool(runtime['constructor_pool']['path'])
    metadata=pool.metadata()
    if (source_provenance._pool(pool,rehash=True)['pool_sha256']!=expected_pool_sha256
            or metadata['compiled_tree']['sha256']!=show['compiled_tree_sha256']):
        raise ValueError('original constructor source binding differs')
    tables=transition.tables_from_metadata(metadata)
    lifecycle=events.compile_catalog(metadata,tables,public_moves=catalog.moves,dex=dex,
        exposure=events.HealthExposure(requests.SHOWDOWN,'gen3randombattle',True,False))
    rules=requests.load_source_rules(show['root'],metadata,node=show['node']['path'])
    request=requests.compile_request_source(metadata,vocabulary=catalog.vocabulary,dex=dex,
        checkpoint_sha256=contract.checkpoint_sha256,rules=rules,lifecycle_catalog=lifecycle)
    facts=own.compile_own_facts(request,catalog)
    helpers=dict(leftovers_compiled=leftovers.compile_leftovers(metadata,engine_catalog=lifecycle),leftovers_mode='retained',
        status_residual_compiled=status.compile_status_residual(metadata,engine_catalog=lifecycle,source_rules=rules),status_residual_mode='kernel-event',
        direct_hit_compiled=hit.compile_direct_hit_controls(engine_catalog=lifecycle,metadata=metadata,source_rules=rules),direct_hit_mode='kernel-event',
        substitute_compiled=substitute.compile_substitute(engine_catalog=lifecycle,metadata=metadata,source_rules=rules),substitute_mode='kernel-event',
        toxic_compiled=toxic.compile_toxic(engine_catalog=lifecycle,metadata=metadata,source_rules=rules),toxic_mode='kernel-event')
    for name,move,module,function in (('recoil','doubleedge',recoil,'compile_recoil'),('calm_mind','calmmind',calm,'compile_calm_mind'),
            ('swords_dance','swordsdance',sword,'compile_swords_dance'),('spikes','spikes',spikes,'compile_spikes'),
            ('thunder_wave','thunderwave',thunder,'compile_thunder_wave')):
        helper=getattr(module,function)(engine_catalog=lifecycle,metadata=metadata,**({'source_rules':rules} if name in ('recoil','thunder_wave') else {})) if move in metadata['vocabulary']['moves'] else None
        helpers[name+'_compiled']=helper;helpers[name+'_mode']='kernel-event' if helper is not None else 'closed'
    # Explicit family compiler: actual numerical SLP setters determine whether
    # there is a capability to compile; verified source descriptors determine
    # its exact supported rows. Never fall back to the legacy one-hot helper.
    sleep_helper=None
    if tables.effects is not None and bool(j.any(tables.effects.status==SLP)):
        sleep_helper=powder.compile_primary_sleep(engine_catalog=lifecycle,metadata=metadata,source_rules=rules)
    helpers['sleep_powder_compiled']=sleep_helper
    helpers['sleep_powder_mode']='kernel-event' if sleep_helper is not None else 'closed'
    def source_spy(_):return driver.FakeScores(j.zeros(9,j.float32),j.array(0,j.int32))
    prepared=driver.prepare(request,facts,fake_forward=source_spy,public_mode='retained-full',account_source_sites=True,
        track_kernel_risks=True,frame_input_mode='source-witness',**helpers)
    masks=dict(catalog.masks.__dict__);masks['opponent_tendency_stats_block']=masks.pop('stats_block')
    public=LocalShowdownConfig(showdown_root=show['root'],node_binary=show['node']['path'],
        observation_spec=V4_REPLAY_OBSERVATION_SPEC,feature_masks=ObservationFeatureMasks(**masks),
        category_vocab=catalog.vocabulary,set_belief_source=True)
    return prepared,pool,public,contract


def execute(inputs,run_root,*,resume=False,evaluation_hook=None,device_rollout=False,device_teacher=False,cache_workers=0):
    from . import v5_run as run,v5_phasec as phase
    _absolute(str(run_root))
    if type(resume) is not bool:raise ValueError('explicit resume choice required')
    if type(device_rollout)is not bool:raise ValueError('explicit device rollout choice required')
    if type(device_teacher)is not bool or device_teacher and (not device_rollout or inputs.plan.model.arm!='D'):
        raise ValueError('explicit D device teacher route required')
    if device_rollout and inputs.plan.model.arm=='D' and not device_teacher:
        raise run.MissingDependency('device D teacher is default closed; no FULL fallback')
    # No fake/default executor, zero-result success or panel bypass.
    if getattr(inputs.plan,'evaluation',None) is None or not callable(evaluation_hook):
        raise run.MissingDependency('declared reviewed evaluation protocol and genuine panel executor required')
    _source(inputs.runtime);_memory(inputs.runtime['memory_policy'],apply=True);_probe(inputs.runtime)
    prepared,pool,public,contract=prepare_device_runtime(inputs) if device_rollout else prepare_runtime(inputs)
    options=dict(device_rollout=True) if device_rollout else {}
    if cache_workers:options['cache_workers']=cache_workers
    if device_teacher:options['device_teacher']=True
    backend=phase.build_backend(inputs.plan,prepared,pool,public_config=public,actor_contract=contract,gender_policy=inputs.gender_policy,**options)
    return run.run(inputs.plan,run_root,backend=backend,resume=resume,enable_v5=True,evaluation_hook=evaluation_hook)


def main(argv=None,*,evaluation_hook=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-v5',action='store_true');parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--arm',required=True,choices=('A0','B0','D0','B1'))
    for name in ('declaration','expected-declaration-sha256','runtime','expected-runtime-sha256','run-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--evaluation-protocol');parser.add_argument('--expected-evaluation-protocol-sha256')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--device-rollout',action='store_true');parser.add_argument('--device-teacher',action='store_true')
    parser.add_argument('--external-evaluation',action='store_true',help='Publish frozen CPU panel requests and continue training')
    parser.add_argument('--cache-workers',type=int,default=0)
    args=parser.parse_args(argv)
    inputs=load_inputs(declaration=args.declaration,expected_declaration_sha256=args.expected_declaration_sha256,
        runtime=args.runtime,expected_runtime_sha256=args.expected_runtime_sha256,arm=args.arm,enable_v5=args.enable_v5)
    if (args.evaluation_protocol is None)!=(args.expected_evaluation_protocol_sha256 is None):
        raise ValueError('explicit evaluation protocol path and digest required together')
    if args.external_evaluation and args.evaluation_protocol is None:
        raise ValueError('external CPU panels require the original declared evaluation protocol')
    if args.evaluation_protocol is not None:
        from .v5_evaluation import Artifact,EvaluationHook
        if (evaluation_hook is not None or inputs.plan.evaluation is None
                or inputs.plan.evaluation.protocol_sha256!=args.expected_evaluation_protocol_sha256):
            raise ValueError('original declared evaluation hook required; no executor override')
        from .v5_run import _sha
        hook_type=EvaluationHook
        if args.external_evaluation:
            from .v5_deferred_evaluation import DetachedEvaluationHook
            hook_type=DetachedEvaluationHook
        evaluation_hook=hook_type(Artifact(args.evaluation_protocol,args.expected_evaluation_protocol_sha256),_sha(inputs.plan.manifest()))
    if args.inspect_only:
        print(json.dumps(dict(status='BOUND_INPUTS_ONLY_NO_RUNTIME_OR_MODEL',arm=args.arm,declaration_sha256=inputs.declaration_sha256)))
        return 0
    options=dict(device_rollout=True) if args.device_rollout else {}
    if args.cache_workers:options['cache_workers']=args.cache_workers
    if args.device_teacher:options['device_teacher']=True
    result=execute(inputs,_absolute(args.run_root),resume=args.resume,evaluation_hook=evaluation_hook,**options)
    print(json.dumps(dict(status=result['status'],steps=result.get('steps'),rows=result.get('rows'),
        reason=result.get('reason'),disposition_counts=dict(Counter(result.get('statuses',{}).values())),
        scientific_qualification=False),allow_nan=False))
    return 0 if result['status']=='COMPLETE_DECLARATION' else 1


if __name__=='__main__':raise SystemExit(main())
