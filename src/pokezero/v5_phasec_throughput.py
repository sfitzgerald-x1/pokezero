"""Finite C-T measurement: the original B-shape 20k collection and five epochs.

Instrumentation and resident cache policy are process-local. No training math,
row order, optimizer, checkpoint cursor or collector cell is replaced. All
profiling overhead remains in the measured end-to-end wall time.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

SCHEMA = 'pokezero.phasec-throughput.v1'
PROFILE_UPDATES = (8, 9, 10, 11)
COLLECTOR_PHASES = ('constructor_wall_seconds','kernel_blocked_seconds','teacher_wall_seconds',
    'host_transfer_wall_seconds','spool_write_wall_seconds','game_cache_wall_seconds','cache_merge_wall_seconds')
COMPILE_SECONDS = ('initializer_lower_seconds','initializer_compile_seconds','collector_lower_seconds','collector_compile_seconds')
COMPILE_CALLS = ('initializer_compile_calls','collector_compile_calls')


def collector_phase_totals(reports, *, interrupted_banks=0):
    reports=tuple(reports)
    if type(interrupted_banks)is not int or not 0<=interrupted_banks<=10 or len(reports)+interrupted_banks!=10:
        raise ValueError('all ten original 20k collector banks required')
    totals={name:0.0 for name in COLLECTOR_PHASES}
    compilation={name:0.0 for name in COMPILE_SECONDS};calls={name:0 for name in COMPILE_CALLS}
    for report in reports:
        phases=report['phases']
        if not set(COLLECTOR_PHASES)<=set(phases) or set(phases)-set(COLLECTOR_PHASES+COMPILE_SECONDS+COMPILE_CALLS):
            raise ValueError('complete literal collector phase breakdown required')
        for name,value in phases.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('finite nonnegative collector phase wall required')
            if name in calls:
                if type(value)is not int:raise ValueError('literal compile call count required')
                calls[name]+=value
            elif name in compilation:compilation[name]+=value
            else:totals[name]+=value
    return dict(banks=len(reports),unavailable_interrupted_banks=interrupted_banks,
        wall_seconds=totals,lower_compile_seconds=compilation,compile_calls=calls,
        scope='Literal saved phase sums only; interrupted banks have unavailable phase times, never zero times. Compilation and overlapping scopes are not added to form the iteration denominator.')


def recover_published_parts(store, plan):
    """Merge only durable per-game caches; no constructor, actor or replay.

    A completion published just before its game cache is lost remains an
    outcome fact with explicitly unavailable training rows. No new terminal,
    value or policy is synthesized to replace it.
    """
    import uuid
    from types import SimpleNamespace
    from gpu_engine.v5_device_spool import Spool
    from pokezero import v5_data, v5_run as runs
    known_banks={p['sha256']for p in store.matching('DEVICE_ACTOR_BANK')}
    if any(p['bank_sha256']not in known_banks for p in store.matching('DEVICE_ROLLOUT_CHUNK')):
        raise ValueError('raw chunk lacks its original actor bank')
    recovered=[]
    def empty(iteration,index,endings):
        if any(p['status']!='FAILED'for p in endings.values()):
            raise ValueError('ended source game requires its original bank and raw chunk')
        record=dict(iteration=iteration,batch=index,published_game_parts=0,
            completed_games_without_published_training_rows=[],cache=None,cache_metadata_sha256=None,
            phase_times='UNAVAILABLE_INTERRUPTED_BANK',no_game_or_model_executed=True)
        store.emit('CT_INTERRUPTED_BANK_RECOVERED',**record);recovered.append(record)
    for iteration,batches in enumerate(plan.iterations):
        for index,batch in enumerate(batches):
            starts=store.matching('BATCH_STARTED',iteration=iteration,batch=index)
            if len(starts)>1:raise ValueError('original collector bank cannot replay')
            if not starts or store.matching('CACHE',iteration=iteration,batch=index):continue
            previous=store.matching('CT_INTERRUPTED_BANK_RECOVERED',iteration=iteration,batch=index)
            if previous:
                if len(previous)!=1:raise ValueError('one durable cache recovery required')
                record=previous[0]
                if record['cache']is not None:
                    store.resolve(record['cache'],record['cache_metadata_sha256'])
                    store.emit('CACHE',iteration=iteration,batch=index,path=record['cache'],sha256=record['cache_metadata_sha256'])
                recovered.append(record);continue
            streams=[g.stream_id for g in batch.games]
            endings={p['stream_id']:p for p in store.matching('GAME_END',iteration=iteration,batch=index)}
            banks=[]
            for reference in store.matching('DEVICE_ACTOR_BANK'):
                body=json.loads(store.resolve(reference['path'],reference['sha256']).read_bytes())
                if body['streams']==streams:banks.append((reference,body))
            if not banks:
                empty(iteration,index,endings);continue
            if len(banks)!=1:raise ValueError('one original interrupted actor bank required')
            reference,body=banks[0]
            if (body['declaration_sha256']!=runs._sha(plan.manifest()) or body['games']!=[
                    dict(stream=g.stream_id,battle_id=g.battle_id,seed=g.seed,team_indices=list(g.team_indices))for g in batch.games]
                    or body['constructor_seed']!=batch.constructor_seed or body['sampling_seed']!=batch.sampling_seed):
                raise ValueError('interrupted bank declaration differs')
            chunks=[r for r in store.matching('DEVICE_ROLLOUT_CHUNK')if r['bank_sha256']==reference['sha256']]
            if not chunks:
                empty(iteration,index,endings);continue
            roots={store.resolve(r['path'],r['sha256']).parent.parent for r in chunks}
            if len(roots)!=1:raise ValueError('one original interrupted spool required')
            spool=Spool.__new__(Spool);spool.store=store;spool.bank=SimpleNamespace(streams=streams);spool.parts=[]
            unavailable=[]
            folder=next(iter(roots))/'games'
            for row,stream in enumerate(streams):
                part=folder/str(row);metadata=part/'metadata.json'
                if not metadata.exists():
                    if endings.get(stream,{}).get('status')=='COMPLETE':unavailable.append(stream)
                    continue
                if stream not in endings:raise ValueError('unended game cannot enter recovered training cache')
                value=json.loads(metadata.read_bytes())
                if any(a['stream']!=stream for a in value['v5_rows']['addresses']):raise ValueError('saved game-cache address differs')
                static=value['v5_static']
                if (static['category_vocabulary_sha256']!=plan.model.category_vocabulary_sha256
                        or static['label_vocabulary_sha256']!=plan.model.label_vocabulary.sha256):
                    raise ValueError('saved game-cache namespace differs')
                for address,meta in zip(value['v5_rows']['addresses'],value['base_json']['step_metadata']):
                    actor,opponent=body['actor_indices'][row][address['owner']],body['actor_indices'][row][1-address['owner']]
                    if (meta['actor_bank_assignment_sha256']!=body['assignment_sha256']
                            or meta['actor_archive_sha256']!=body['actor_archives'][actor]
                            or meta['opponent_archive_sha256']!=body['actor_archives'][opponent]):
                        raise ValueError('saved game-cache actor assignment differs')
                spool.parts.append((row,part,v5_data._file_sha(metadata)))
            path=None;digest=None
            if spool.parts:
                (store.path/'caches').mkdir(exist_ok=True)
                relative='caches/recovered-'+uuid.uuid4().hex
                spool.merge(store.path/relative)  # Original writer verifies metadata and all array hashes.
                path=relative;digest=v5_data._file_sha(store.path/relative/'metadata.json')
            record=dict(iteration=iteration,batch=index,published_game_parts=len(spool.parts),
                completed_games_without_published_training_rows=unavailable,cache=path,
                cache_metadata_sha256=digest,
                phase_times='UNAVAILABLE_INTERRUPTED_BANK',no_game_or_model_executed=True)
            store.emit('CT_INTERRUPTED_BANK_RECOVERED',**record)
            if path is not None:store.emit('CACHE',iteration=iteration,batch=index,path=path,sha256=digest)
            recovered.append(record)
    return recovered


@contextmanager
def continue_pretraining_iteration(run_root,*,allow_trained_prefix=False):
    """Original resume; trained prefixes require explicit missing-timing scope.

    This controller may be copied outside that clean checkout. Its reporting
    compatibility never changes the engine/source binding of existing caches.
    """
    from pokezero import v5_run as runs
    if type(allow_trained_prefix)is not bool:raise ValueError('explicit trained-prefix reporting choice required')
    original=runs.run;state={}
    def continued(plan,path,**kwargs):
        if Path(path).resolve()!=Path(run_root).resolve() or 'resume'in kwargs:
            raise ValueError('one original C-T continuation path required')
        store=runs.Store(path,plan,resume=True)
        checkpoints=store.matching('CHECKPOINT');progress=store.matching('TRAIN_PROGRESS')
        if store.matching('ITERATION_COMPLETE'):raise ValueError('completed original iteration cannot replay')
        if not allow_trained_prefix and(progress or any(p['steps']!=0 for p in checkpoints)):
            raise ValueError('C-T pretraining recovery cannot erase prior update timings')
        checkpoint=checkpoints[-1]if checkpoints else dict(steps=0,rows=0)
        state['prior_durable_updates']=checkpoint['steps'];state['prior_durable_row_visits']=checkpoint.get('rows',0)
        state['prior_reported_updates']=max((p['steps']for p in progress),default=0)
        state['uncheckpointed_reported_updates']=max(0,state['prior_reported_updates']-checkpoint['steps'])
        state['prefix_component_timings']='UNAVAILABLE; durable original RUNTIME_PHASE walls retained'if checkpoint['steps']else'NO_TRAINED_PREFIX'
        first=store.events[0]
        if first['kind']!='RUN_STARTED' or not math.isfinite(first['time']):
            raise ValueError('original durable iteration start required')
        state['original_run_started_epoch']=first['time']
        # Previously recovered banks remain immutable on another pretraining
        # interruption; recover_published_parts skips their published caches.
        recover_published_parts(store,plan)
        state['recovered_banks']=store.matching('CT_INTERRUPTED_BANK_RECOVERED')
        result=original(plan,path,resume=True,**kwargs)
        state['run_finished_epoch']=time.time()
        return result
    runs.run=continued
    try:yield state
    finally:runs.run=original


def gpu_trace_categories(document):
    totals = {name:dict(events=0, seconds=0.0) for name in ('kernel','gpu_memcpy','gpu_memset')}
    for event in document['traceEvents']:
        category = event.get('cat')
        if event.get('ph') != 'X' or category not in totals:
            continue
        duration = event['dur']
        if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not math.isfinite(duration) or duration<0:
            raise ValueError('finite nonnegative GPU trace duration required')
        totals[category]['events'] += 1
        totals[category]['seconds'] += duration/1e6
    return totals


def gate_report(smoke, timing):
    wall = smoke['original_run_wall_seconds']
    if (smoke['declared_games'] != 20000 or smoke['status'] != 'COMPLETE_DECLARATION'
            or smoke['jax_reload'] != 'VERIFIED_ORIGINAL_TRAINED_ARCHIVE_LOADED'
            or type(smoke['steps']) is not int or smoke['steps']<=0
            or type(timing['updates'])is not int or timing['updates']<=0
            or type(timing.get('prior_durable_updates',0))is not int or timing.get('prior_durable_updates',0)<0
            or timing['updates']+timing.get('prior_durable_updates',0) != smoke['steps']
            or isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<=0):
        raise ValueError('one full original 20k collection/five-epoch training result required')
    rate = 20000/wall
    arm=smoke.get('model_arm','B')
    threshold=('D_GE5'if rate>=5 else'D_GE1_LT5'if rate>=1 else'D_LT1_HOLD')if arm=='D'else(
        'TARGET20_MET'if rate>=20 else'FLOOR5_MET_TARGET20_MISSED'if rate>=5 else'BELOW_BINDING5')
    if timing.get('prior_durable_updates',0):threshold='RECOVERED_TRAINED_PREFIX_NO_AUTOMATIC_ADMISSION'
    return dict(schema=SCHEMA,model_arm=arm,declared_games=20000,iteration_wall_seconds=wall,
        declared_games_per_second=rate,
        natural_complete_per_second=smoke['dispositions'].get('COMPLETE',0)/wall,
        dispositions=smoke['dispositions'],
        threshold=threshold,target20_met=rate>=20,
        kernel_and_collection=smoke['collection_runtime'],
        host_phase_wall_seconds=smoke['original_phase_wall_seconds'],training=timing,
        timing_scope='Original run wall, including collection compile, cache I/O, five epochs, checkpoint and export. Preparation and final strict JAX reload separately reported. Instrumentation overhead retained.',
        limitations='Nominal fixed20k numerator beside natural completions and all unknown dispositions; no strength claim. Host walls can overlap; sampled GPU kernel times are not extrapolated into a gate denominator.',
        original_pipeline=smoke)


@contextmanager
def instrument_training(run_root, resident_cache_bytes):
    """Wrap original calls; collect CPU data walls and a fixed four-update trace."""
    if type(resident_cache_bytes) is not int or resident_cache_bytes<=0:
        raise ValueError('positive explicit host residency budget required')
    from pokezero import v5_training as training, dataset, v5_data
    originals = []
    state = dict(updates=0,prior_durable_updates=None,cache_prepare_seconds=0.0,batch_slice_seconds=0.0,
        tensor_conversion_seconds=0.0,update_wall_seconds=0.0,profile_updates=[])
    profiler = None
    retained_trace=Path(run_root)/'ct-training-trace.json'
    from pokezero.v5_data import _file_sha
    retained_sha=_file_sha(retained_trace)if retained_trace.exists()else None
    # Never overwrite an already durable original profile on a continuation.
    import uuid
    trace_path=(Path(run_root)/('ct-training-trace-'+uuid.uuid4().hex+'.json')if retained_sha else retained_trace)

    def install(owner,name,callback):
        original = getattr(owner,name)
        originals.append((owner,name,original))
        setattr(owner,name,callback(original))

    def wall(field):
        def decorator(original):
            def measured(*args,**kwargs):
                start=time.perf_counter()
                try:return original(*args,**kwargs)
                finally:state[field]+=time.perf_counter()-start
            return measured
        return decorator

    install(training,'_cache_reader',wall('cache_prepare_seconds'))
    install(dataset,'slice_training_batch',wall('batch_slice_seconds'))
    if hasattr(v5_data,'_resident_slice_batch'):
        install(v5_data,'_resident_slice_batch',wall('batch_slice_seconds'))
        state['resident_validation']='ONCE_AT_LOAD; exact immutable private slices only'
    install(training,'_training_tensors',wall('tensor_conversion_seconds'))

    def resident(original):
        def measured(*args,**kwargs):
            if 'resident_cache_bytes' in kwargs:
                raise ValueError('C-T owns its declared residency budget')
            return original(*args,**kwargs,resident_cache_bytes=resident_cache_bytes)
        return measured
    install(training,'train_caches',resident)

    def update(original):
        def measured(runtime,batch):
            nonlocal profiler
            if state['prior_durable_updates']is None:state['prior_durable_updates']=runtime.steps
            if runtime.steps!=state['prior_durable_updates']+state['updates']:
                raise ValueError('original durable update cursor changed during measurement')
            ordinal=runtime.steps+1
            if ordinal==PROFILE_UPDATES[0]:
                torch=training._torch()
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
            start=time.perf_counter()
            result=original(runtime,batch)
            # The original update already synchronizes finite checks and its
            # scalar diagnostics. Profiling adds no alternative optimizer path.
            state['update_wall_seconds']+=time.perf_counter()-start
            state['updates']+=1
            if ordinal in PROFILE_UPDATES:
                state['profile_updates'].append(ordinal)
                profiler.step()
            if ordinal==PROFILE_UPDATES[-1]:
                profiler.__exit__(None,None,None)
                profiler.export_chrome_trace(str(trace_path))
                profiler=None
            return result
        return measured
    install(training,'_update',update)
    try:
        yield state
        prefix=state['prior_durable_updates']
        if prefix is None:raise ValueError('actual training updates required')
        if prefix>=PROFILE_UPDATES[-1]:
            if state['profile_updates']or retained_sha is None or _file_sha(retained_trace)!=retained_sha:
                raise ValueError('unchanged original prespecified prefix GPU trace required')
            trace_path=retained_trace;state['gpu_profile_scope']='REUSED_ORIGINAL_PREFIX_UPDATES; not a resumed-update sample'
        else:
            if tuple(state['profile_updates']) != PROFILE_UPDATES:
                raise ValueError('all prespecified GPU profile updates required')
            state['gpu_profile_scope']='CURRENT_INVOCATION_ORIGINAL_GLOBAL_UPDATES'
        state['gpu_trace_path']=str(trace_path);state['gpu_trace_sha256']=_file_sha(trace_path)
        state['gpu_sample_categories']=gpu_trace_categories(json.loads(trace_path.read_bytes()))
        state['gpu_sample_updates']=len(PROFILE_UPDATES)
        state['gpu_sample_update_ordinals']=list(PROFILE_UPDATES)
        state['gpu_kernel_seconds_per_sampled_update']=state['gpu_sample_categories']['kernel']['seconds']/len(PROFILE_UPDATES)
        state['data_seconds_per_update']=(state['batch_slice_seconds']+state['tensor_conversion_seconds'])/state['updates']
        state['data_scope']='CURRENT INVOCATION updates only: CPU slice plus tensor conversion wall, including blocking transfers. Prefix component times unavailable; one-time resident preparation separate.'
        state['gpu_scope']='Only actual kernel trace-category durations on fixed updates8..11; memcpy/memset separate, no CPU annotation spans. Diagnostic sample, not full-iteration GPU total.'
    finally:
        if profiler is not None:
            profiler.__exit__(None,None,None)
        for owner,name,original in reversed(originals):
            setattr(owner,name,original)


def collector_banks(events):
    """Join delayed data-writer timings by their original declared streams."""
    banks={};current=None;stream_banks={}
    for event in events:
        if event['kind']=='BATCH_STARTED':
            p=event['payload'];streams=tuple(p['streams'])
            if streams in stream_banks:raise ValueError('declared collector streams cannot repeat')
            stream_banks[streams]=(p['iteration'],p['batch'])
    for event in events:
        if event['kind']=='BATCH_STARTED':current=(event['payload']['iteration'],event['payload']['batch'])
        if event['kind']=='DEVICE_RUNTIME_TIMING':
            payload=event['payload'];key=stream_banks.get(tuple(payload['bank_streams']))if 'bank_streams'in payload else current
            if key is None or key in banks:raise ValueError('one literal timing per original collector bank required')
            banks[key]=payload
    return banks


def execute(config,run_root,*,resident_cache_bytes,enable_ct=False,resume=False,cache_workers=0,model_arm='B',teacher_frame_batch=False,resume_trained=False,
        raw_retention_every=1,transient_raw_root=None,teacher_queue_chunks=0):
    if enable_ct is not True:
        raise ValueError('explicit finite C-T execution required')
    if type(raw_retention_every)is not int or raw_retention_every<=0:raise ValueError('positive fixed raw retention interval required')
    if type(resume_trained)is not bool or resume_trained and not resume:raise ValueError('trained continuation requires original resume')
    if model_arm=='D'and teacher_frame_batch is not True:raise ValueError('D-T requires the batched teacher route')
    from contextlib import nullcontext
    from pokezero import v5_device_smoke
    with (continue_pretraining_iteration(run_root,allow_trained_prefix=resume_trained)if resume else nullcontext(None))as recovery:
        with instrument_training(run_root,resident_cache_bytes) as timing:
            options={}if cache_workers==0 else dict(cache_workers=cache_workers)
            if model_arm!='B':options['model_arm']=model_arm
            if teacher_frame_batch:options['teacher_frame_batch']=True
            if teacher_queue_chunks:options['teacher_queue_chunks']=teacher_queue_chunks
            if raw_retention_every!=1 or transient_raw_root is not None:
                options.update(raw_retention_every=raw_retention_every,transient_raw_root=transient_raw_root)
            original=v5_device_smoke.execute(config,run_root,enable_smoke=True,**options)
    if recovery is not None:
        if timing['prior_durable_updates']!=recovery['prior_durable_updates']:
            raise ValueError('timed continuation differs from original durable checkpoint')
        original['current_attempt_run_wall_seconds']=original['original_run_wall_seconds']
        original['original_run_wall_seconds']=recovery['run_finished_epoch']-recovery['original_run_started_epoch']
        if original['original_run_wall_seconds']<original['current_attempt_run_wall_seconds']:
            raise ValueError('original elapsed wall cannot omit the continuation')
    report=gate_report(original,timing)
    events=[json.loads(path.read_bytes())for path in sorted((Path(run_root)/'journal').glob('*.json'))]
    banks=collector_banks(events)
    missing=([(p['iteration'],p['batch'])for p in recovery['recovered_banks']if (p['iteration'],p['batch'])not in banks]
        if recovery else [])
    report['collector_phase_walls']=collector_phase_totals(banks.values(),interrupted_banks=len(missing))
    report['collector_phase_walls']['unavailable_bank_keys']=missing
    report['cache_pipeline']=[dict(bank=list(key),**p['cache_pipeline'])for key,p in sorted(banks.items())if 'cache_pipeline'in p]
    report['teacher_queue']=[dict(bank=list(key),**p['teacher_queue'])for key,p in sorted(banks.items())if p.get('teacher_queue')is not None]
    if recovery is not None:
        report['recovery']=recovery
        report['threshold_before_recovery_caveat']=report['threshold']
        report['threshold']='RECOVERED_DIAGNOSTIC_NO_AUTOMATIC_ADMISSION'
        report['timing_scope']='Elapsed wall from original durable RUN_STARTED to completed resumed run, including interruption and reinitialization; final reload separate. Missing interrupted-bank component timings remain unavailable. No clean-run speed comparison or automatic admission.'
    target=Path(run_root)/'ct-result.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    import argparse
    from pokezero import v5_worker
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--resident-cache-bytes',type=int,required=True)
    parser.add_argument('--enable-ct',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-trained',action='store_true')
    parser.add_argument('--cache-workers',type=int,default=0)
    parser.add_argument('--model-arm',choices=('A','B','D'),default='B')
    parser.add_argument('--teacher-frame-batch',action='store_true')
    parser.add_argument('--teacher-queue-chunks',type=int,default=0)
    parser.add_argument('--raw-retention-every',type=int,default=1)
    parser.add_argument('--transient-raw-root')
    args=parser.parse_args(argv)
    config=v5_worker._record(dict(path=args.config,sha256=args.expected_config_sha256))
    report=execute(config,args.run_root,resident_cache_bytes=args.resident_cache_bytes,enable_ct=args.enable_ct,resume=args.resume,cache_workers=args.cache_workers,model_arm=args.model_arm,teacher_frame_batch=args.teacher_frame_batch,teacher_queue_chunks=args.teacher_queue_chunks,resume_trained=args.resume_trained,
        raw_retention_every=args.raw_retention_every,transient_raw_root=args.transient_raw_root)
    print(json.dumps(dict(status=report['threshold'],result=str(Path(args.run_root)/'ct-result.json'))))
    return 0


if __name__=='__main__':raise SystemExit(main())
