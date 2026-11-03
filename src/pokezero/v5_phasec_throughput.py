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


def collector_phase_totals(reports):
    reports=tuple(reports)
    if len(reports)!=10:
        raise ValueError('all ten original 20k collector banks required')
    totals={name:0.0 for name in COLLECTOR_PHASES}
    for report in reports:
        phases=report['phases']
        if set(phases)!=set(COLLECTOR_PHASES):
            raise ValueError('complete literal collector phase breakdown required')
        for name,value in phases.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('finite nonnegative collector phase wall required')
            totals[name]+=value
    return dict(banks=len(reports),wall_seconds=totals,
        scope='Sum each literal DEVICE_RUNTIME_TIMING phase over all ten fixed banks; overlapping scopes are not added to form the iteration denominator.')


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
    from . import v5_training as training, dataset
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


def execute(config,run_root,*,resident_cache_bytes,enable_ct=False):
    if enable_ct is not True:
        raise ValueError('explicit finite C-T execution required')
    from . import v5_device_smoke
    with instrument_training(run_root,resident_cache_bytes) as timing:
        original=v5_device_smoke.execute(config,run_root,enable_smoke=True)
    report=gate_report(original,timing)
    banks=[]
    for path in sorted((Path(run_root)/'journal').glob('*.json')):
        event=json.loads(path.read_bytes())
        if event['kind']=='DEVICE_RUNTIME_TIMING':banks.append(event['payload'])
    report['collector_phase_walls']=collector_phase_totals(banks)
    target=Path(run_root)/'ct-result.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    import argparse
    from . import v5_worker
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--resident-cache-bytes',type=int,required=True)
    parser.add_argument('--enable-ct',action='store_true')
    args=parser.parse_args(argv)
    config=v5_worker._record(dict(path=args.config,sha256=args.expected_config_sha256))
    report=execute(config,args.run_root,resident_cache_bytes=args.resident_cache_bytes,enable_ct=args.enable_ct)
    print(json.dumps(dict(status=report['threshold'],result=str(Path(args.run_root)/'ct-result.json'))))
    return 0


if __name__=='__main__':raise SystemExit(main())
