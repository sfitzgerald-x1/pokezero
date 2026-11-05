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


def build_plan(bank,profile,labels,category,*,arm='B',iterations=1):
    """Use genuine Plan types and the original B model/loss/optimizer recipe."""
    from . import v5_run as runs,v5_training as training
    from . import v5_phasec as phase
    if set(bank)!=set(BANK_FIELDS)or bank['games']!=GAMES or bank['batch_rows']!=2048:
        raise ValueError('exact separate20k declaration with2048 physical batch required')
    for name in BANK_FIELDS:
        if name not in ('run_id','pool_sha256'):
            runs._seed(bank[name])
    if bank['model_seed']in (0,1):raise ValueError('smoke initialization must differ from fixed production seeds')
    if arm not in ('A','B','D'):raise ValueError('explicit A/B/D model shape required')
    config=profile.configurations(labels,category_vocabulary=category,arm=arm,replicate=0)
    train=training.TrainConfig(**dict(config['training'],seed=bank['model_seed']))
    if type(iterations)is not int or iterations not in (1,2):raise ValueError('one iteration or fixed two-iteration pipeline required')
    declared=[];batch_index=0
    for iteration in range(iterations):
        batches=[]
        for start in range(iteration*GAMES,(iteration+1)*GAMES,bank['batch_rows']):
            stop=min(start+bank['batch_rows'],(iteration+1)*GAMES)
            games=tuple(runs.Game(bank['stream_start']+i,f"{bank['run_id']}-{i:05d}",bank['identity_seed_start']+i,(2*i,2*i+1))for i in range(start,stop))
            batches.append(runs.Batch(games,bank['constructor_seed_start']+batch_index,bank['sampling_seed_start']+batch_index,
                bank['gender_seed'],bank['gender_domain'],tuple(bank['gender_root_start']+i for i in range(start,stop))))
            batch_index+=1
        declared.append(tuple(batches))
    return runs.Plan(bank['run_id'],config['model'],config['loss'],train,tuple(declared),bank['pool_sha256'],
        bank['selection_seed'],config['historical_fraction'],config['historical_window'],bank['max_requests'],
        runs.Watchdog(21600 if arm=='D' else 1800),checkpoint_updates=32,discount=config['discount'])



def teacher_table(store,run_root):
    """All selected/refused decisions by original frame, not accepted labels."""
    rows={};strategies=set();timing={}
    for report in store.matching('DEVICE_TEACHER_FRAME_COUNTS'):
        strategies.add(report['strategy'])
        for frame,value in report['by_frame'].items():
            counts=rows.setdefault(int(frame),dict(requested=0,selected=0,labeled=0,refused=0,unselected=0))
            for key in counts:counts[key]+=value[key]
        for key,value in report['batch_timing_totals'].items():timing[key]=timing.get(key,0)+value
    if not rows:return None
    columns=('frame','requested','selected','labeled','refused','unselected','refused_share_selected')
    values=[]
    for frame,counts in sorted(rows.items()):
        if counts['selected']!=counts['labeled']+counts['refused']or counts['requested']!=counts['selected']+counts['unselected']:
            raise ValueError('teacher frame denominator differs')
        values.append([frame,*(counts[k]for k in columns[1:-1]),None if not counts['selected']else counts['refused']/counts['selected']])
    path=Path(run_root)/'teacher-frame-table.json'
    path.write_text(json.dumps(dict(columns=columns,rows=values),allow_nan=False)+'\n')
    from .v5_data import _file_sha
    totals={key:sum(v[key]for v in rows.values())for key in ('requested','selected','labeled','refused','unselected')}
    return dict(path=str(path),sha256=_file_sha(path),frames=len(rows),strategies=sorted(strategies),
        totals=totals,refused_share_selected=None if not totals['selected']else totals['refused']/totals['selected'],batch_timing_totals=timing)


def execute(config,run_root,*,enable_smoke=False,cache_workers=0,model_arm='B',teacher_frame_batch=False,raw_retention_every=1,transient_raw_root=None,teacher_queue_chunks=0,iteration_policy_lag=0):
    """Actual collect -> cache -> original train/checkpoint/export -> JAX load."""
    if enable_smoke is not True:raise ValueError('explicit finite smoke execution required')
    if type(raw_retention_every)is not int or raw_retention_every<=0:raise ValueError('positive fixed raw retention interval required')
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
    if type(iteration_policy_lag)is not int or iteration_policy_lag not in (0,1):raise ValueError('explicit zero or one iteration lag required')
    iterations=2 if iteration_policy_lag else 1
    plan=build_plan(config['bank'],profile,labels,category,arm=model_arm,iterations=iterations)
    if (type(teacher_queue_chunks)is not int or not 0<=teacher_queue_chunks<=4
            or teacher_queue_chunks and not teacher_frame_batch):raise ValueError('queue requires explicit D frame batching')
    if type(teacher_frame_batch)is not bool or teacher_frame_batch and model_arm!='D':raise ValueError('frame teacher batching is D-only')
    # Same source/memory/precision ordering as the original production worker;
    # there is no fabricated production evaluation protocol or readiness object.
    worker._source(runtime);worker._memory(runtime['memory_policy'],apply=True);worker._probe(runtime)
    inputs=worker.Inputs(plan,profile,json.dumps(runtime),json.dumps(vocabulary),json.dumps(gender),runs._sha(plan.manifest()))
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine import v5_model
    begin=time.perf_counter();prepared,pool,_,_=worker.prepare_device_runtime(inputs)
    preparation=time.perf_counter()-begin
    teacher=None
    if model_arm=='D':
        from gpu_engine.v5_device_teacher import Provider
        teacher=Provider(prepared,plan,frame_batch=teacher_frame_batch,**(dict(chunk_batch=True)if teacher_queue_chunks else{}))
    options={}if raw_retention_every==1 and transient_raw_root is None else dict(raw_retention_every=raw_retention_every,transient_raw_root=transient_raw_root)
    if teacher_queue_chunks:options['teacher_queue_chunks']=teacher_queue_chunks
    backend=RuntimeBackend(prepared,pool,gender_policy=gender,measure_runtime=True,cache_workers=cache_workers,teacher=teacher,**options)
    begin=time.perf_counter();result=runs.run(plan,run_root,backend=backend,enable_v5=True,measure_timings=True,**(dict(iteration_policy_lag=1)if iteration_policy_lag else{}))
    pipeline=time.perf_counter()-begin
    report=dict(schema=SCHEMA,status=result['status'],declared_games=GAMES*iterations,steps=result.get('steps'),rows=result.get('rows'),
        dispositions={status:list(result.get('statuses',{}).values()).count(status)for status in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED')},
        preparation_cold_inclusive_seconds=preparation,original_run_wall_seconds=pipeline,
        purpose='V5_READY_20K_DIAGNOSTIC_NOT_PRODUCTION_ARM_OR_STRENGTH',stop_rule='one20k measurement; one repair if producer fails; then report',
        scientific_qualification=False,jax_reload='UNAVAILABLE_NO_COMPLETED_ITERATION',model_arm=model_arm)
    if result['status']=='COMPLETE_DECLARATION':
        if not isinstance(result.get('steps'),int)or result['steps']<=0 or not isinstance(result.get('rows'),int)or result['rows']<=0:
            raise ValueError('actual completed training updates/rows required')
        store=runs.Store(run_root,plan,resume=True);snapshot=store.matching('SNAPSHOT',iteration=iterations-1)
        if model_arm=='D':report['teacher']=teacher_table(store,run_root)
        if iteration_policy_lag:
            cohort_timings=[];training_budgets=[];prior_steps=0;prior_rows=0
            for index in range(iterations):
                start=next(e['time']for e in store.events if e['kind']=='COLLECTION_ITERATION_STARTED'and e['payload']['iteration']==index)
                end=next(e['time']for e in store.events if e['kind']=='SNAPSHOT'and e['payload']['iteration']==index)
                cohort_timings.append(dict(iteration=index,collection_start_epoch=start,snapshot_epoch=end,
                    collection_to_own_trained_snapshot_seconds=end-start))
                final_snapshot=store.matching('SNAPSHOT',iteration=index)[0]
                checkpoints=[x for x in store.matching('CHECKPOINT',iteration=index)if x['sha256']==final_snapshot['checkpoint_sha256']]
                if len(checkpoints)!=1:raise ValueError('actual final checkpoint for each finite training cohort required')
                checkpoint=checkpoints[0]
                training_budgets.append(dict(iteration=index,start_step=prior_steps,end_step=checkpoint['steps'],
                    row_visits=checkpoint['rows']-prior_rows))
                prior_steps=checkpoint['steps'];prior_rows=checkpoint['rows']
            report.update(iteration_policy_lag=1,pipeline_iterations=iterations,
                pipeline_windows=store.matching('POLICY_LAG_WINDOW_COMPLETE'),
                collection_iterations=store.matching('COLLECTION_ITERATION_COMPLETE'),
                phase_iteration_walls=store.matching('RUNTIME_PHASE'),
                acting_snapshots=store.matching('ITERATION_ACTOR_BINDING'),
                complete_iterations=store.matching('ITERATION_COMPLETE'),cohort_latencies=cohort_timings,
                cohort_training_budgets=training_budgets)
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
