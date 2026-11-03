"""One separate 20k v5-ready measurement, using the original finite trainer.

No work occurs on import. This diagnostic Plan is not a truncated production
arm, never consumes production cells, and never runs a Foul Play panel. Actual
execution is opt-in, from hash-bound existing runtime inputs and a fixed bank.
Stop rule: one measurement, one producer repair if it fails, then report.
"""
import argparse
import json
import os
from pathlib import Path
import time

SCHEMA='pokezero.device-v5-ready-smoke.v1'
GAMES=20_000
BANK_FIELDS=('run_id','pool_sha256','games','batch_rows','model_seed','stream_start','identity_seed_start',
    'constructor_seed_start','sampling_seed_start','gender_seed','gender_root_start','gender_domain','selection_seed','max_requests')


def build_plan(bank,profile,labels,category):
    """Use genuine Plan types and the original B model/loss/optimizer recipe."""
    from . import v5_run as runs,v5_training as training
    from . import v5_phasec as phase
    if set(bank)!=set(BANK_FIELDS)or bank['games']!=GAMES or bank['batch_rows']!=2048:
        raise ValueError('exact separate20k declaration with2048 physical batch required')
    for name in BANK_FIELDS:
        if name not in ('run_id','pool_sha256'):
            runs._seed(bank[name])
    if bank['model_seed']in (0,1):raise ValueError('smoke initialization must differ from fixed production seeds')
    config=profile.configurations(labels,category_vocabulary=category,arm='B',replicate=0)
    train=training.TrainConfig(**dict(config['training'],seed=bank['model_seed']))
    batches=[]
    for index,start in enumerate(range(0,GAMES,bank['batch_rows'])):
        stop=min(start+bank['batch_rows'],GAMES)
        games=tuple(runs.Game(bank['stream_start']+i,f"{bank['run_id']}-{i:05d}",bank['identity_seed_start']+i,(2*i,2*i+1))for i in range(start,stop))
        batches.append(runs.Batch(games,bank['constructor_seed_start']+index,bank['sampling_seed_start']+index,
            bank['gender_seed'],bank['gender_domain'],tuple(bank['gender_root_start']+i for i in range(start,stop))))
    return runs.Plan(bank['run_id'],config['model'],config['loss'],train,(tuple(batches),),bank['pool_sha256'],
        bank['selection_seed'],config['historical_fraction'],config['historical_window'],bank['max_requests'],
        runs.Watchdog(1800),checkpoint_updates=32,discount=config['discount'])


def execute(config,run_root,*,enable_smoke=False,cache_workers=0):
    """Actual collect -> cache -> original train/checkpoint/export -> JAX load."""
    if enable_smoke is not True:raise ValueError('explicit finite smoke execution required')
    if set(config)!=set(('schema','runtime','labels','bank','persistent_cache'))or config['schema']!=SCHEMA:
        raise ValueError('literal finite smoke configuration required')
    cache=config['persistent_cache']
    if type(cache)is not str or not Path(cache).is_absolute():raise ValueError('explicit persistent CPU/GPU cache path required')
    if os.environ.get('JAX_COMPILATION_CACHE_DIR',cache)!=cache:raise ValueError('persistent cache environment differs')
    os.environ['JAX_COMPILATION_CACHE_DIR']=cache
    from . import v5_worker as worker,v5_run as runs,v5_contract as contract
    runtime=worker._record(config['runtime']);labels=contract.LabelVocabulary(**worker._record(config['labels']))
    profile,vocabulary,gender,category=worker._runtime_metadata(runtime,expected_pool_sha256=config['bank']['pool_sha256'])
    if gender is None:raise ValueError('original gender metadata for declared smoke pool required')
    plan=build_plan(config['bank'],profile,labels,category)
    # Same source/memory/precision ordering as the original production worker;
    # there is no fabricated production evaluation protocol or readiness object.
    worker._source(runtime);worker._memory(runtime['memory_policy'],apply=True);worker._probe(runtime)
    inputs=worker.Inputs(plan,profile,json.dumps(runtime),json.dumps(vocabulary),json.dumps(gender),runs._sha(plan.manifest()))
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine import v5_model
    begin=time.perf_counter();prepared,pool,_,_=worker.prepare_device_runtime(inputs)
    preparation=time.perf_counter()-begin
    backend=RuntimeBackend(prepared,pool,gender_policy=gender,measure_runtime=True,cache_workers=cache_workers)
    begin=time.perf_counter();result=runs.run(plan,run_root,backend=backend,enable_v5=True,measure_timings=True)
    pipeline=time.perf_counter()-begin
    report=dict(schema=SCHEMA,status=result['status'],declared_games=GAMES,steps=result.get('steps'),rows=result.get('rows'),
        dispositions={status:list(result.get('statuses',{}).values()).count(status)for status in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED')},
        preparation_cold_inclusive_seconds=preparation,original_run_wall_seconds=pipeline,
        purpose='V5_READY_20K_DIAGNOSTIC_NOT_PRODUCTION_ARM_OR_STRENGTH',stop_rule='one20k measurement; one repair if producer fails; then report',
        scientific_qualification=False,jax_reload='UNAVAILABLE_NO_COMPLETED_ITERATION')
    if result['status']=='COMPLETE_DECLARATION':
        if not isinstance(result.get('steps'),int)or result['steps']<=0 or not isinstance(result.get('rows'),int)or result['rows']<=0:
            raise ValueError('actual completed training updates/rows required')
        store=runs.Store(run_root,plan,resume=True);snapshot=store.matching('SNAPSHOT',iteration=0)
        if len(snapshot)!=1:raise ValueError('one original final trained export required')
        from gpu_engine.v5_device_timing import aggregate
        report['collection_runtime']=aggregate(store.matching('DEVICE_RUNTIME_TIMING'),store.matching('GAME_END'))
        report['original_phase_wall_seconds']={name:sum(x['wall_seconds']for x in store.matching('RUNTIME_PHASE')if x['scope']==name)
            for name in sorted(set(x['scope']for x in store.matching('RUNTIME_PHASE')))}
        report['training_wall_scope']='TRAIN_CACHES_INCLUSIVE includes original reader/data I/O and training; no pure GPU-training decomposition claimed'
        final=snapshot[0];path=store.resolve(final['archive'],final['archive_sha256'])
        begin=time.perf_counter();params,actual,metadata=v5_model.load(path,expected_archive_sha256=final['archive_sha256'],
            expected_checkpoint_sha256=final['checkpoint_sha256'],expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
            expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256,enable_v5=True)
        import jax
        jax.block_until_ready(params)
        if actual!=plan.model:raise ValueError('actual reloaded final B model differs')
        report.update(jax_reload='VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED',jax_load_blocked_seconds=time.perf_counter()-begin,
            checkpoint_sha256=final['checkpoint_sha256'],archive_sha256=final['archive_sha256'],actor_parameters=metadata['parameter_count'])
        store.emit('SMOKE_END_TO_END',**report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--run-root',required=True);parser.add_argument('--enable-smoke',action='store_true')
    args=parser.parse_args(argv)
    # Existing artifact reader, not a permissive alternative source loader.
    from . import v5_worker as worker
    config=worker._record(dict(path=args.config,sha256=args.expected_config_sha256))
    worker._absolute(args.run_root)
    result=execute(config,args.run_root,enable_smoke=args.enable_smoke)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status']=='COMPLETE_DECLARATION'and result['jax_reload']=='VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED'else 1


if __name__=='__main__':raise SystemExit(main())
