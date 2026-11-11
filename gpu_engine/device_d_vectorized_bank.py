"""One declared2048-game D3 collection/label/cache measurement, no training.

Cold actor setup is reported separately. The bank clock includes preparation,
all first-use compilation, queue drain and the genuine training-cache merge.
No warm-only replacement denominator or favorable-frame selection is used.
"""
from pathlib import Path
import json
import math
import time

GAMES=2048
THRESHOLD_SECONDS=2048.


def measure(plan,path,*,backend,enable_benchmark=False):
    from pokezero import v5_run as runs,v5_training as training
    from .v5_device_backend import RuntimeBackend
    from .v5_device_teacher import Provider
    if enable_benchmark is not True:raise ValueError('explicit one-bank D3 benchmark required')
    if (type(plan)is not runs.Plan or plan.model.arm!='D' or len(plan.iterations)!=1
            or len(plan.iterations[0])!=1 or len(plan.games)!=GAMES):
        raise ValueError('one fixed declared2048-game D bank required')
    if (type(backend)is not RuntimeBackend or type(backend.teacher)is not Provider
            or not backend.teacher.vectorized or backend.teacher_queue_chunks!=2
            or not backend.measure_runtime):
        raise ValueError('genuine vectorized D3 provider, reviewed queue2 and timings required')
    backend.check_plan(plan);store=runs.Store(path,plan,resume=False)
    setup=time.perf_counter()
    runtime=training.create(plan.model,plan.loss,plan.train,enable_v5=True)
    checkpoint=store.checkpoint(runtime,iteration=-1);snapshot=store.snapshot(checkpoint,iteration=-1)
    actor_setup=time.perf_counter()-setup
    # Initial parameters and actor export exist, but no optimizer update runs.
    if runtime.steps!=0 or runtime.rows!=0:raise ValueError('initial untouched D model required')
    batch=plan.iterations[0][0];progress=runs.Progress(plan.watchdog)
    assignments=tuple(runs.assignment(plan,g,i,snapshot,())for i,g in enumerate(batch.games))
    statuses={g.stream_id:'NOT_ATTEMPTED'for g in batch.games}
    begin=time.perf_counter()
    try:
        prepared=backend.prepare_batch(plan,batch,snapshot,assignments,store)
        store.emit('BATCH_STARTED',iteration=0,batch=0,streams=list(statuses),assignments=assignments)
        statuses={stream:'RUNNING'for stream in statuses}
        def ended(stream_id,status,**facts):
            if statuses.get(stream_id)!='RUNNING':raise ValueError('unknown/replayed D3 completion')
            if status not in ('COMPLETE','CAPPED','REFUSED','FAILED'):raise ValueError('literal D3 game disposition required')
            statuses[stream_id]=status
            if status=='COMPLETE':progress.completed_game(facts['duration_seconds'])
            store.emit('GAME_END',stream_id=stream_id,status=status,iteration=0,batch=0,**facts)
        result=backend.collect(prepared,store,ended,progress.check)
        if type(result)is runs.PendingCache:result=result.result(check_progress=progress.check)
        if result is None:raise ValueError('no actual labeled training cache; no refill')
        cache,digest=result;store.resolve(cache,digest)
        store.emit('CACHE',iteration=0,batch=0,path=cache,sha256=digest)
        wall=time.perf_counter()-begin
        if not math.isfinite(wall)or wall<=0:raise ValueError('finite positive original bank wall required')
    except BaseException as error:
        for stream,status in statuses.items():
            if status=='RUNNING':
                statuses[stream]='FAILED';store.emit('GAME_END',stream_id=stream,status='FAILED',reason='D3-interrupted-no-replay')
        store.emit('STOP',reason=type(error).__name__+': '+str(error));raise
    finally:backend.close()
    counts={s:list(statuses.values()).count(s)for s in ('COMPLETE','CAPPED','REFUSED','FAILED','RUNNING','NOT_ATTEMPTED')}
    frames={};calls=[];root_preparation={}
    for event in store.matching('DEVICE_TEACHER_FRAME_COUNTS'):
        for frame,values in event['by_frame'].items():
            row=frames.setdefault(int(frame),dict(requested=0,selected=0,labeled=0,refused=0,unselected=0))
            for k in row:row[k]+=values[k]
        calls.extend(event['frame_calls'])
        for frame,seconds in event.get('root_prepare_by_frame',{}).items():
            root_preparation[int(frame)]=root_preparation.get(int(frame),0.)+seconds
    for frame,row in frames.items():
        if row['requested']!=row['selected']+row['unselected']or row['selected']!=row['labeled']+row['refused']:
            raise ValueError('D3 frame denominator changed')
        row['refused_share_selected']=None if row['selected']==0 else row['refused']/row['selected']
        own=[x for x in calls if x['frames']==[frame]]
        row['root_detach_and_draw_seconds']=root_preparation.get(frame,0.)
        row['label_call_wall_seconds']=sum(x['frame_wall_seconds']for x in own)+row['root_detach_and_draw_seconds']
        row['world_materialize_seconds']=sum(x['world_prepare_seconds']for x in own)
        row['world_prepare_seconds']=row['world_materialize_seconds']+row['root_detach_and_draw_seconds']
        row['device_seconds']=sum(x['device_seconds']for x in own)
        row['lower_seconds']=sum(x['lower_seconds']for x in own)
        row['compile_seconds']=sum(x['compile_seconds']for x in own)
        row['padded_leaf_counts_per_call']=[x['padded_leaf_count']for x in own]
    timings=store.matching('DEVICE_RUNTIME_TIMING')
    if len(timings)!=1:raise ValueError('exactly one bank timing required')
    phases=timings[0]['phases']
    known=counts['COMPLETE']+counts['CAPPED']==GAMES
    report=dict(schema='pokezero.device-d3-bank.v1',declared_games=GAMES,dispositions=counts,
        actor_setup_seconds=actor_setup,bank_wall_seconds=wall,declared_games_per_second=GAMES/wall,
        natural_games_per_second=counts['COMPLETE']/wall,threshold_seconds=THRESHOLD_SECONDS,
        admission='PASS'if known and wall<=THRESHOLD_SECONDS else'FAIL',optimizer_updates=runtime.steps,
        phase_walls=phases,teacher_queue=timings[0]['teacher_queue'],frames={str(k):v for k,v in sorted(frames.items())},frame_calls=calls,
        compile=dict(collector_lower_seconds=phases['collector_lower_seconds'],collector_compile_seconds=phases['collector_compile_seconds'],
            initializer_lower_seconds=phases['initializer_lower_seconds'],initializer_compile_seconds=phases['initializer_compile_seconds'],
            teacher_lower_seconds=sum(x['lower_seconds']for x in calls),teacher_compile_seconds=sum(x['compile_seconds']for x in calls)),
        cache=dict(path=cache,sha256=digest),declaration_sha256=runs._sha(plan.manifest()),
        timing_scope='One original bank including cold preparation/compilation, collection, labels, queue and cache drain; setup/export separately reported. Phase spans overlap and are not additive.',
        stop_rule='One bank; if miss and host-world-preparation is largest, one batched-world-preparation change and one separate declared measurement, then existing hold rule. No other speed loop.',
        scientific_strength_qualification=False)
    if getattr(backend.teacher,'batched_prep',False):
        report['batched_preparation']=True
        report['root_prepare_timing_scope']='Per-root planning wall plus equal-per-pending-root share of chunk detach wall; not isolated per-frame detach timing.'
        report['compile']['batched_prepare_seconds']=None
        report['compile']['batched_prepare_scope']='First-use detach/readout compilation is included in root preparation and bank wall, not separately isolated.'
    store.emit('D3_BANK_END',**report)
    with (Path(path)/'d3-bank-result.json').open('x')as output:json.dump(report,output,allow_nan=False,indent=2)
    return report
