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
def continue_pretraining_iteration(run_root):
    """Original resume on the original backend source, before any update.

    This controller may be copied outside that clean checkout. Its reporting
    compatibility never changes the engine/source binding of existing caches.
    """
    from pokezero import v5_run as runs
    original=runs.run;state={}
    def continued(plan,path,**kwargs):
        if Path(path).resolve()!=Path(run_root).resolve() or 'resume'in kwargs:
            raise ValueError('one original C-T continuation path required')
        store=runs.Store(path,plan,resume=True)
        if store.matching('TRAIN_PROGRESS') or store.matching('ITERATION_COMPLETE') or any(
                p['steps']!=0 for p in store.matching('CHECKPOINT')):
            raise ValueError('C-T pretraining recovery cannot erase prior update timings')
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
            or timing['updates'] != smoke['steps']
            or isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<=0):
        raise ValueError('one full original 20k collection/five-epoch training result required')
    rate = 20000/wall
    return dict(schema=SCHEMA,declared_games=20000,iteration_wall_seconds=wall,
        declared_games_per_second=rate,
        natural_complete_per_second=smoke['dispositions'].get('COMPLETE',0)/wall,
        dispositions=smoke['dispositions'],
        threshold='TARGET20_MET' if rate>=20 else 'FLOOR5_MET_TARGET20_MISSED' if rate>=5 else 'BELOW_BINDING5',
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
    from pokezero import v5_training as training, dataset
    originals = []
    state = dict(updates=0,cache_prepare_seconds=0.0,batch_slice_seconds=0.0,
        tensor_conversion_seconds=0.0,update_wall_seconds=0.0,profile_updates=[])
    profiler = None
    trace_path = Path(run_root)/'ct-training-trace.json'

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
            ordinal=state['updates']+1
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
        if tuple(state['profile_updates']) != PROFILE_UPDATES:
            raise ValueError('all prespecified GPU profile updates required')
        state['gpu_trace_path']=str(trace_path)
        state['gpu_sample_categories']=gpu_trace_categories(json.loads(trace_path.read_bytes()))
        state['gpu_sample_updates']=len(PROFILE_UPDATES)
        state['gpu_kernel_seconds_per_sampled_update']=state['gpu_sample_categories']['kernel']['seconds']/len(PROFILE_UPDATES)
        state['data_seconds_per_update']=(state['batch_slice_seconds']+state['tensor_conversion_seconds'])/state['updates']
        state['data_scope']='CPU slice plus tensor conversion wall, including blocking transfers; one-time resident preparation separately reported.'
        state['gpu_scope']='Only actual kernel trace-category durations on fixed updates8..11; memcpy/memset separate, no CPU annotation spans. Diagnostic sample, not full-iteration GPU total.'
    finally:
        if profiler is not None:
            profiler.__exit__(None,None,None)
        for owner,name,original in reversed(originals):
            setattr(owner,name,original)


def execute(config,run_root,*,resident_cache_bytes,enable_ct=False,resume=False):
    if enable_ct is not True:
        raise ValueError('explicit finite C-T execution required')
    from contextlib import nullcontext
    from pokezero import v5_device_smoke
    with (continue_pretraining_iteration(run_root)if resume else nullcontext(None))as recovery:
        with instrument_training(run_root,resident_cache_bytes) as timing:
            original=v5_device_smoke.execute(config,run_root,enable_smoke=True)
    if recovery is not None:
        original['current_attempt_run_wall_seconds']=original['original_run_wall_seconds']
        original['original_run_wall_seconds']=recovery['run_finished_epoch']-recovery['original_run_started_epoch']
        if original['original_run_wall_seconds']<original['current_attempt_run_wall_seconds']:
            raise ValueError('original elapsed wall cannot omit the continuation')
    report=gate_report(original,timing)
    banks={};current=None
    for path in sorted((Path(run_root)/'journal').glob('*.json')):
        event=json.loads(path.read_bytes())
        if event['kind']=='BATCH_STARTED':current=(event['payload']['iteration'],event['payload']['batch'])
        if event['kind']=='DEVICE_RUNTIME_TIMING':
            if current is None or current in banks:raise ValueError('one literal timing per original collector bank required')
            banks[current]=event['payload']
    missing=([(p['iteration'],p['batch'])for p in recovery['recovered_banks']if (p['iteration'],p['batch'])not in banks]
        if recovery else [])
    report['collector_phase_walls']=collector_phase_totals(banks.values(),interrupted_banks=len(missing))
    report['collector_phase_walls']['unavailable_bank_keys']=missing
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
    args=parser.parse_args(argv)
    config=v5_worker._record(dict(path=args.config,sha256=args.expected_config_sha256))
    report=execute(config,args.run_root,resident_cache_bytes=args.resident_cache_bytes,enable_ct=args.enable_ct,resume=args.resume)
    print(json.dumps(dict(status=report['threshold'],result=str(Path(args.run_root)/'ct-result.json'))))
    return 0


if __name__=='__main__':raise SystemExit(main())
