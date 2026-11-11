"""Fixed-order concurrent ORIGINAL-v4 canary endpoint; import starts nothing.

Finite owned cell processes, each with a bound actor/FP physical-core pair.
Physical parallelism is independent of original scientific cell slots. Only a
child changes its own affinity, before importing model libraries.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


def _write(path,value):
    from . import b4_fidelity_canary as c
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);blob=c.encoded(value)
    if path.exists():c.require(path.read_bytes()==blob,'immutable original endpoint artifact differs')
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.pending',delete=False) as stream:
            temporary=Path(stream.name);stream.write(blob);stream.flush();os.fsync(stream.fileno())
        try:
            os.link(temporary,path)  # Exclusive atomic publication, never partial JSON.
            c._sync_directory(path.parent);c._sync_directory(path.parent.parent)
        finally:temporary.unlink();c._sync_directory(path.parent)
    return hashlib.sha256(blob).hexdigest()


def _read_result(path):
    from . import b4_fidelity_canary as c
    blob=Path(path).read_bytes();value=json.loads(blob)
    c.require(c.encoded(value)==blob,'canonical immutable original FP result required')
    return value


def _cpu_pairs(plan, pairs=None):
    """Runtime placement/parallelism may move; declared cells never change."""
    if pairs is None:return plan.fp_cpu_pairs
    from . import b4_fidelity_canary as c
    c.require(type(pairs) in (tuple,list) and 1<=len(pairs)<=4*c.PANEL_SEEDS and
        all(type(pair) in (tuple,list) and len(pair)==2 and
            all(type(cpu) is int and cpu>=0 for cpu in pair) for pair in pairs) and
        len({cpu for pair in pairs for cpu in pair})==2*len(pairs),
        'finite runtime CPU pairs must use distinct literal cores')
    return tuple(tuple(pair) for pair in pairs)


def _execution_slot(cell,plan,pairs):
    """Keep old slot provenance; assign a separate fixed physical slot."""
    _checked_cell(cell,plan)
    return cell['slot'] if len(pairs)==len(plan.fp_cpu_pairs) else cell['ordinal']%len(pairs)


def _request_pairs(request, plan):
    """Old requests retain old placement; moved requests bind their own file."""
    from . import b4_fidelity_canary as c
    digest=request.get('endpoint_cpu_runtime_sha256')
    if 'endpoint_cpu_runtime_sha256' not in request:pairs=plan.fp_cpu_pairs
    else:
        c.require(type(digest) is str and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),
            'literal endpoint runtime identity required')
        value=_read_result(Path(request['run_directory'])/'fp-cpu-runtime'/(digest+'.json'))
        c.require(set(value)=={'schema','declaration_sha256','cpu_pairs','cpu_topology'} and
            value['schema']=='pokezero.b4.canary.endpoint-cpu-runtime.v1' and c.sha(value)==digest and
            value['declaration_sha256']==request['declaration_sha256'],'endpoint runtime/source binding differs')
        pairs=_cpu_pairs(plan,value['cpu_pairs'])
    c.require(request['cpu_pair']==list(pairs[_execution_slot(request['cell'],plan,pairs)]),'original ordinal/runtime core mapping differs')
    return pairs


def cells(plan):
    from . import b4_fidelity_canary as c
    return tuple(dict(ordinal=4*offset+2*seat+arm_index,offset=offset,seat=seat,arm=arm,
        slot=(4*offset+2*seat+arm_index+offset)%len(plan.fp_cpu_pairs))
        for offset in range(c.PANEL_SEEDS) for seat in (0,1) for arm_index,arm in enumerate(('CONTROL','CANARY')))


def _checked_cell(cell,plan):
    from . import b4_fidelity_canary as c
    c.require(type(cell) is dict and set(cell)=={'ordinal','offset','seat','arm','slot'} and
        all(type(cell[k]) is int for k in ('ordinal','offset','seat','slot')) and
        0<=cell['offset']<c.PANEL_SEEDS and cell['seat'] in (0,1) and cell['arm'] in ('CONTROL','CANARY') and
        cell['ordinal']==4*cell['offset']+2*cell['seat']+(cell['arm']=='CANARY') and
        cell['slot']==(cell['ordinal']+cell['offset'])%len(plan.fp_cpu_pairs),'exact typed original ordinal/slot required')


def configuration(plan,cell,checkpoint,*,cpu_pairs=None):
    from dataclasses import fields
    from . import foulplay_bridge as fp
    _checked_cell(cell,plan);raw=plan.fp_configs[cell['seat']];pairs=_cpu_pairs(plan,cpu_pairs)
    values={**raw,'checkpoint':Path(checkpoint),'games':1,'seed_start':raw['seed_start']+cell['offset'],
        'foulplay_random_seed':raw['foulplay_random_seed']+cell['offset'],
        'foulplay_cpu_affinity':pairs[_execution_slot(cell,plan,pairs)][1]}
    for field in fields(fp.ControlledFoulPlayConfig):
        if field.name in values and values[field.name] is not None and 'Path' in str(field.type):values[field.name]=Path(values[field.name])
    return fp.ControlledFoulPlayConfig(**values)


def _start(request,digest):
    return dict(schema='pokezero.b4.canary.fp-start.v1',request_sha256=digest,cell=request['cell'],
        declaration_sha256=request['declaration_sha256'],checkpoint=request['checkpoint'],
        checkpoint_sha256=request['checkpoint_sha256'],cpu_pair=request['cpu_pair'])


def checked_result(value,request,plan):
    from . import b4_fidelity_canary as c,v5_foulplay
    _checked_cell(value['cell'],plan);_checked_cell(request['cell'],plan)
    c.require(set(value)=={'schema','request_sha256','cell','cpu_pair','checkpoint_sha256','duration_seconds','status','score','error','source_game','benchmark'} and
        value['schema']=='pokezero.b4.canary.fp-cell.v1' and value['request_sha256']==c.sha(request)
        and value['cell']==request['cell'] and c.encoded(value['cpu_pair'])==c.encoded(request['cpu_pair']) and
        value['checkpoint_sha256']==request['checkpoint_sha256'],'original cell result binding differs')
    duration=value['duration_seconds'];c.require(type(duration) in (int,float) and math.isfinite(duration)
        and 0<=duration<=plan.backstop_seconds,'literal actual cell duration')
    config=configuration(plan,request['cell'],request['checkpoint'],cpu_pairs=_request_pairs(request,plan))
    if value['error'] is not None:
        c.require(type(value['error']) is str and value['status']=='UNKNOWN' and value['score'] is None,'failure cannot become a score')
    else:
        game=value['source_game'];benchmark=value['benchmark']
        c.require(benchmark['checkpoint_sha256']==request['checkpoint_sha256'] and benchmark['policy_mode']=='raw'
            and benchmark['seed_start']==config.seed_start and benchmark['pokezero_player']==config.pokezero_player
            and benchmark['foulplay_random_seed_schedule']['seeds']==[config.foulplay_random_seed]
            and benchmark['game_results']==[game] and game['seed']==config.seed_start,'actual original FP source result differs')
        if config.encoder_kind=='jax-owning':
            c.require(benchmark.get('encoder_kind')=='jax-owning','canary endpoint must retain actual B2 encoding')
        score=v5_foulplay._score(config,game)
        c.require(value['score']==score and value['status']==('COMPLETE' if score is not None else 'UNKNOWN'),
            'literal terminal score differs')
    return value


async def launch(request_path):
    """Fixed original worker only, never an arbitrary executor callback."""
    from . import b4_fidelity_canary as c
    request_path=Path(request_path);digest=c.file_sha(request_path)
    log=request_path.parent.parent/'fp-errors'/request_path.with_suffix('.log').name;log.parent.mkdir(exist_ok=True)
    with log.open('xb') as stderr:
        try:
            process=await _spawn_owned(sys.executable,'-B','-m','pokezero.b4_canary_endpoint',
                '--request',str(request_path),'--expected-request-sha256',digest,
                stdout=asyncio.subprocess.DEVNULL,stderr=stderr,start_new_session=True)
            try:code=await process.wait()
            finally:await _finish_owned_cleanup(process)
        finally:
            stderr.flush();os.fsync(stderr.fileno());c._sync_directory(log.parent)
    c.require(code==0,'owned FP worker exited unsuccessfully; no retry')
    result=request_path.parent.parent/'fp-results'/request_path.name
    c.require(result.is_file(),'owned FP worker missing durable result; no retry')
    return _read_result(result)


async def _spawn_owned(*args,**kwargs):
    """Cancellation cannot lose a process spawned before its transport returns."""
    spawning=asyncio.create_task(asyncio.create_subprocess_exec(*args,**kwargs))
    try:return await asyncio.shield(spawning)
    except asyncio.CancelledError:
        while not spawning.done():
            try:await asyncio.shield(spawning)
            except asyncio.CancelledError:pass
        process=spawning.result()
        await _finish_owned_cleanup(process)
        raise


async def _finish_owned_cleanup(process):
    from .owning_encoder_continuation import _stop_owned_group
    cleanup=asyncio.create_task(_stop_owned_group(process))
    try:await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        while not cleanup.done():
            try:await asyncio.shield(cleanup)
            except asyncio.CancelledError:pass
        cleanup.result()  # A failed reap remains visible rather than becoming cancellation.
        raise


async def evaluate_with_sigterm(plan,book,*,champion_checkpoint,final_checkpoint,device_path=False,endpoint_cpu_pairs=None):
    """Own TERM for this public invocation, then restore/defer caller handlers.

    A separate endpoint task makes repeated termination idempotent while its
    finally block gathers all independent process groups. Existing asyncio
    handlers retain their original Handle/context; Python handlers receive the
    deferred signal with frame=None, only after successful owned cleanup.
    """
    loop=asyncio.get_running_loop();registry=getattr(loop,'_signal_handlers',None)
    if not isinstance(registry,dict):raise RuntimeError('Unix asyncio signal-handler preservation required')
    prior_async=registry.get(signal.SIGTERM);prior_python=signal.getsignal(signal.SIGTERM)
    if prior_async is not None and prior_async.cancelled():raise RuntimeError('prior SIGTERM handler is cancelled')
    child=None;requested=False;received=False;cleanup_finished=False
    def cancel_once():
        nonlocal requested
        if child is not None and not requested:requested=True;child.cancel()
    def terminate():
        nonlocal received
        received=True;cancel_once()
    loop.add_signal_handler(signal.SIGTERM,terminate)
    installed=registry.get(signal.SIGTERM)
    try:
        options={} if endpoint_cpu_pairs is None else dict(endpoint_cpu_pairs=endpoint_cpu_pairs)
        child=asyncio.create_task(evaluate_declared(plan,book,champion_checkpoint=champion_checkpoint,final_checkpoint=final_checkpoint,device_path=device_path,**options))
        if received:cancel_once()
        try:
            value=await asyncio.shield(child);cleanup_finished=True;return value
        except asyncio.CancelledError:
            cancel_once()
            while not child.done():
                try:await asyncio.shield(child)
                except asyncio.CancelledError:pass
            if not child.cancelled():child.result()  # Propagate cleanup failure, never report a clean stop.
            cleanup_finished=True
            raise
    finally:
        # Do not replace a different handler installed by the parent meanwhile.
        current_registry=getattr(loop,'_signal_handlers',None)
        still_ours=current_registry is registry and (installed is None or registry.get(signal.SIGTERM) is installed)
        if still_ours:
            loop.remove_signal_handler(signal.SIGTERM)
            if prior_async is not None:
                loop.add_signal_handler(signal.SIGTERM,prior_async._run)
                # Re-arm the OS/wakeup fd through the public API, then retain
                # the exact original Handle, callback arguments and context.
                registry[signal.SIGTERM]=prior_async
            elif signal.getsignal(signal.SIGTERM)!=prior_python:signal.signal(signal.SIGTERM,prior_python)
            if received and cleanup_finished:
                if prior_async is not None:prior_async._run()
                elif callable(prior_python):prior_python(signal.SIGTERM,None)


def pin_owned_actor(pair):
    """Only this new child. FP's original wrapper moves its child to pair[1]."""
    if len(pair)!=2 or any(type(x) is not int or x<0 for x in pair) or pair[0]==pair[1]:raise ValueError('explicit disjoint actor/FP core pair')
    if not hasattr(os,'sched_setaffinity'):raise ValueError('Linux dedicated-core affinity required')
    allowed=os.sched_getaffinity(0)
    if pair[0] not in allowed:raise ValueError('declared actor core absent')
    # A controller intentionally excludes FP cores from its own inherited mask.
    # Probe permission in this fresh child; Linux cpusets remain authoritative.
    os.sched_setaffinity(0,{pair[1]})
    if os.sched_getaffinity(0)!={pair[1]}:raise ValueError('declared FP core unavailable to owned child')
    for tid in Path('/proc/self/task').iterdir():
        try:os.sched_setaffinity(int(tid.name),{pair[0]})
        except ProcessLookupError:pass
    for tid in Path('/proc/self/task').iterdir():
        try:
            if os.sched_getaffinity(int(tid.name))!={pair[0]}:raise ValueError('owned actor thread affinity differs')
        except ProcessLookupError:pass


def check_placement(plan,*,cpu_pairs=None):
    """Read-only parent check. Launcher taskset is an ordinary runtime action."""
    from .owning_encoder_continuation import cpu_allocation
    available=None
    for path in ('/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/cpuset/cpuset.cpus'):
        source=Path(path)
        if source.is_file():
            text=source.read_text().strip()
            if not text:continue
            available=set()
            for block in text.split(','):
                bounds=block.split('-');start=int(bounds[0]);end=int(bounds[-1])
                if len(bounds)>2 or start<0 or end<start:raise ValueError('invalid kernel cpuset')
                available.update(range(start,end+1))
            break
    if available is None:raise ValueError('Linux effective cpuset required before endpoint')
    pairs=_cpu_pairs(plan,cpu_pairs)
    topology=cpu_allocation(pairs,available=available)
    actors={pair[0] for pair in pairs}
    for tid in Path('/proc/self/task').iterdir():
        try:
            if not set(os.sched_getaffinity(int(tid.name)))<=actors:
                raise ValueError('launcher must bind controller threads to declared actor CPUs before endpoint')
        except ProcessLookupError:pass
    return topology


async def worker(request_path,expected_sha256,*,runtime=None,expected_pair=None):
    """One actual original benchmark, after durable START/claim and pinning."""
    # Standard library only until this child's CPU mask is installed.
    blob=Path(request_path).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected_sha256:raise ValueError('original cell request digest differs')
    request=json.loads(blob)
    if expected_pair is None:pin_owned_actor(tuple(request['cpu_pair']))
    elif tuple(request['cpu_pair'])!=tuple(expected_pair):raise ValueError('persistent worker core pair changed')
    from . import b4_fidelity_canary as c,foulplay_bridge as fp,v5_foulplay
    keys={'schema','run_directory','declaration_sha256','cell','checkpoint','checkpoint_sha256','cpu_pair'}
    c.require(set(request) in (keys,keys|{'endpoint_cpu_runtime_sha256'}) and
        c.encoded(request)==blob and request['schema']=='pokezero.b4.canary.fp-request.v1','original cell request schema')
    root=Path(request['run_directory']);plan=c.load_plan(root/'declaration.json',expected_sha256=request['declaration_sha256'])
    cell=request['cell'];_checked_cell(cell,plan)
    pairs=_request_pairs(request,plan)
    c.require(Path(request_path).resolve()==root/'fp-requests'/f"{cell['ordinal']}.json",'original owned request path')
    start_path=root/'fp-starts'/f"{cell['ordinal']}.json"
    c.require(start_path.read_bytes()==c.encoded(_start(request,expected_sha256)),'durable original start must precede attempt')
    binding=json.loads((root/'fp-snapshot.json').read_bytes())
    c.require(binding['declaration_sha256']==request['declaration_sha256'] and
        binding[cell['arm']]==dict(checkpoint=request['checkpoint'],sha256=request['checkpoint_sha256']), 'original frozen endpoint checkpoint')
    claim=root/'fp-claims'/f"{cell['ordinal']}.json"
    claim.parent.mkdir(exist_ok=True)
    with claim.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    c._sync_directory(claim.parent);c._sync_directory(root)
    c.require(c.file_sha(request['checkpoint'])==request['checkpoint_sha256'],'original selected model artifact differs')
    config=configuration(plan,cell,request['checkpoint'],cpu_pairs=pairs);started=time.monotonic();game=document=None
    try:
        result=(await fp.run_controlled_foulplay_benchmark(config) if runtime is None else
            await runtime.run(config,expected_checkpoint_sha256=request['checkpoint_sha256']))
        document=result.to_dict()
        if len(result.games)==1:game=result.games[0].to_dict()
        c.require(result.config==config and result.checkpoint_sha256==request['checkpoint_sha256'] and
            result.foulplay_random_seed_schedule==(config.foulplay_random_seed,) and len(result.games)==1,'original one-cell benchmark binding')
        score=v5_foulplay._score(config,game)
        value=dict(status='COMPLETE' if score is not None else 'UNKNOWN',score=score,error=None)
    except Exception as error:value=dict(status='UNKNOWN',score=None,error=type(error).__name__+': '+str(error))
    value.update(schema='pokezero.b4.canary.fp-cell.v1',request_sha256=expected_sha256,cell=cell,cpu_pair=request['cpu_pair'],
        checkpoint_sha256=request['checkpoint_sha256'],duration_seconds=time.monotonic()-started,source_game=game,benchmark=document)
    checked_result(value,request,plan)
    _write(root/'fp-results'/f"{cell['ordinal']}.json",value)
    return 0


async def evaluate_declared(plan,book,*,champion_checkpoint,final_checkpoint,device_path=False,endpoint_cpu_pairs=None):
    from . import b4_fidelity_canary as c
    pairs=_cpu_pairs(plan,endpoint_cpu_pairs)
    topology=check_placement(plan) if endpoint_cpu_pairs is None else check_placement(plan,cpu_pairs=pairs)
    root=book.path.resolve();declaration=c.file_sha(root/'declaration.json')
    if type(device_path)is not bool:raise ValueError('explicit endpoint path required')
    if device_path and not all(x.get('encoder_kind')=='jax-owning' for x in plan.fp_configs):
        raise ValueError('device canary requires original B2 owning endpoint')
    pool=(WorkerPool(root,plan) if endpoint_cpu_pairs is None else WorkerPool(root,plan,cpu_pairs=pairs)) if device_path else None
    snapshots={arm:dict(checkpoint=str(Path(path).resolve()),sha256=c.file_sha(path)) for arm,path in
        [('CONTROL',champion_checkpoint),('CANARY',final_checkpoint)]}
    frozen=dict(declaration_sha256=declaration,cpu_topology=topology,**snapshots)
    runtime=None
    if endpoint_cpu_pairs is not None:
        placement=dict(schema='pokezero.b4.canary.endpoint-cpu-runtime.v1',declaration_sha256=declaration,
            cpu_pairs=pairs,cpu_topology=topology)
        runtime=c.sha(placement);_write(root/'fp-cpu-runtime'/(runtime+'.json'),placement)
        if (root/'fp-snapshot.json').exists():
            previous=_read_result(root/'fp-snapshot.json')
            c.require({k:v for k,v in previous.items() if k!='cpu_topology'}==
                {k:v for k,v in frozen.items() if k!='cpu_topology'},'original frozen endpoint checkpoint differs')
            frozen=previous  # Retain first placement as provenance, never rewrite it.
    _write(root/'fp-snapshot.json',frozen)
    roster=cells(plan);rows=[dict(**cell,status='NOT_ATTEMPTED',score=None) for cell in roster]
    progress=c.Progress(plan);book.emit('BATCH_STARTED',stage='FP_ENDPOINT')
    recorded={}
    def request_for(cell):
        if cell['ordinal'] in recorded:return recorded[cell['ordinal']]
        snapshot=snapshots[cell['arm']]
        request=dict(schema='pokezero.b4.canary.fp-request.v1',run_directory=str(root),declaration_sha256=declaration,
            cell=cell,checkpoint=snapshot['checkpoint'],checkpoint_sha256=snapshot['sha256'],cpu_pair=list(pairs[_execution_slot(cell,plan,pairs)]))
        if runtime is not None:request['endpoint_cpu_runtime_sha256']=runtime
        return request
    def retain(cell,value,*,receipt=None):
        row=dict(**cell,status=value['status'],score=value['score'],source_game=value.get('source_game'),reason=value.get('error'),
            duration_seconds=value['duration_seconds'],result_sha256=c.sha(value) if receipt else None,result=receipt)
        key={k:cell[k] for k in ('arm','offset','seat')}
        if row['status']=='COMPLETE':
            completion=dict(phase='FP',**key,status='COMPLETE',duration_seconds=row['duration_seconds'],source_sha256=c.sha(value['source_game']))
            previous=book.matching('GAME_END',phase='FP',**key)
            if previous:c.require(previous==[completion],'original completed FP receipt changed')
            else:book.emit('GAME_END',**completion);progress.completed(row['duration_seconds'])
        book.emit('FP_END',**row);rows[cell['ordinal']]=row
    for cell in roster:
        key={k:cell[k] for k in ('arm','offset','seat')};ended=book.matching('FP_END',**key);started=book.matching('FP_START',**key)
        result=root/'fp-results'/f"{cell['ordinal']}.json"
        if started:
            request_path=root/'fp-requests'/f"{cell['ordinal']}.json";start_path=root/'fp-starts'/request_path.name
            request=request_for(cell)
            if endpoint_cpu_pairs is not None:
                saved=_read_result(request_path);ignored={'cpu_pair','endpoint_cpu_runtime_sha256'}
                c.require({k:v for k,v in saved.items() if k not in ignored}==
                    {k:v for k,v in request.items() if k not in ignored},'original saved endpoint request differs')
                _request_pairs(saved,plan);recorded[cell['ordinal']]=saved;request=saved
            c.require(len(started)==1 and all(started[0][k]==cell[k] for k in cell) and
                request_path.read_bytes()==c.encoded(request) and started[0]['request_sha256']==c.sha(request) and
                start_path.read_bytes()==c.encoded(_start(request,c.sha(request))) and
                started[0]['start_sha256']==c.file_sha(start_path) and started[0]['cpu_pair']==request['cpu_pair'],
                'original durable cell start/request/core mapping changed')
        if ended:
            c.require(started and len(ended)==1 and all(ended[0][k]==cell[k] for k in cell),'immutable original ended cell differs')
            row=ended[0]
            if row['result']:
                c.require(Path(row['result'])==result,'original owned result path differs')
                value=_read_result(result);checked_result(value,request_for(cell),plan)
                c.require(c.sha(value)==row['result_sha256'] and value['score']==row['score'] and value['status']==row['status']
                    and value['duration_seconds']==row['duration_seconds'] and value['source_game']==row['source_game']
                    and value['error']==row['reason'],'completed source changed')
            else:c.require(row['status']=='UNKNOWN' and row['score'] is None,'complete result needs original source receipt')
            rows[cell['ordinal']]=row
            if row['status']=='COMPLETE':progress.longest=max(progress.longest,row['duration_seconds'])
        elif started:
            if result.exists():retain(cell,checked_result(_read_result(result),request_for(cell),plan),receipt=str(result))
            else:retain(cell,dict(status='UNKNOWN',score=None,source_game=None,error='attempted interrupted cell; no replay',duration_seconds=0.))
        else:c.require(not result.exists() and not (root/'fp-claims'/f"{cell['ordinal']}.json").exists(),
                       'untracked attempted source cannot become a fresh cell')
    pending=[cell for cell in roster if rows[cell['ordinal']]['status']=='NOT_ATTEMPTED'];next_cell=0;active={};stop=False
    async def one(cell,request):
        launch_cell=launch if pool is None else pool.launch
        value=checked_result(await launch_cell(root/'fp-requests'/f"{cell['ordinal']}.json"),request,plan)
        retain(cell,value,receipt=str(root/'fp-results'/f"{cell['ordinal']}.json"))
        return value['error'] is not None
    try:
        while next_cell<len(pending) or active:
            progress.check()
            while next_cell<len(pending) and not stop:
                cell=pending[next_cell];slot=_execution_slot(cell,plan,pairs)
                if slot in active:break  # Never skip declared order for whichever outcome frees a core.
                request=request_for(cell);digest=_write(root/'fp-requests'/f"{cell['ordinal']}.json",request)
                start=_start(request,digest);start_sha=_write(root/'fp-starts'/f"{cell['ordinal']}.json",start)
                book.emit('FP_START',**cell,request_sha256=digest,start_sha256=start_sha,cpu_pair=request['cpu_pair'])
                active[slot]=(cell,asyncio.create_task(one(cell,request)));next_cell+=1
            if not active:break
            done,_=await asyncio.wait([task for _,task in active.values()],timeout=1.,return_when=asyncio.FIRST_COMPLETED)
            for slot,(cell,task) in list(active.items()):
                if task not in done:continue
                del active[slot]
                try:stop=task.result() or stop
                except Exception as error:
                    retain(cell,dict(status='UNKNOWN',score=None,source_game=None,error=type(error).__name__+': '+str(error),duration_seconds=0.))
                    raise  # Cleanup failure cannot release/reuse a physical slot.
    finally:
        for _,task in active.values():task.cancel()
        results=await asyncio.gather(*(task for _,task in active.values()),return_exceptions=True)
        if pool is not None:await pool.close()
        for result in results:
            if isinstance(result,Exception):raise result
    paired=[]
    for offset in range(c.PANEL_SEEDS):
        for seat in (0,1):
            pair={r['arm']:r for r in rows if r['offset']==offset and r['seat']==seat}
            paired.append(None if any(x['status']!='COMPLETE' for x in pair.values()) else pair['CANARY']['score']-pair['CONTROL']['score'])
    known=[x for x in paired if x is not None];unknown=len(paired)-len(known)
    report=dict(status='ENDPOINT_RECORDED',original_paired_cells=c.PANEL_SEEDS*2,known_pairs=len(known),unknown_pairs=unknown,
        paired_mean=sum(known)/len(known) if known else None,identification_bounds=[(sum(known)-unknown)/len(paired),(sum(known)+unknown)/len(paired)],
        automatic_fidelity_pass=False,uncertainty='No equivalence margin or significance pass inferred; paired interval/result review required',rows=rows,
        paired_seed_means=[None if paired[2*i] is None or paired[2*i+1] is None else (paired[2*i]+paired[2*i+1])/2 for i in range(c.PANEL_SEEDS)],
        interval_unit='Original paired seed block (both seats); no IID-cell significance claim',max_workers=len(pairs))
    report.update(paired_interval(rows,seed=plan.bootstrap_seed,blocks=c.PANEL_SEEDS))
    if runtime is not None:report['endpoint_cpu_runtime_sha256']=runtime
    book.emit('FP_ENDPOINT',report=report);return report


class WorkerPool:
    """One owned process/core pair, two immutable model holders, fresh games."""
    def __init__(self, root, plan, *, cpu_pairs=None):
        self.root, self.plan, self.workers = Path(root), plan, {}
        self.cpu_pairs=_cpu_pairs(plan,cpu_pairs)
    async def launch(self, request_path):
        from . import b4_fidelity_canary as c
        request = json.loads(Path(request_path).read_bytes())
        slot = _execution_slot(request['cell'],self.plan,self.cpu_pairs)
        if slot not in self.workers:
            folder = self.root / 'fp-worker-errors'; folder.mkdir(exist_ok=True)
            log = (folder / (str(slot) + '-' + str(time.time_ns()) + '.log')).open('xb')
            pair = self.cpu_pairs[slot]
            try:
                process = await _spawn_owned(sys.executable, '-B', '-m', 'pokezero.b4_device_canary_runtime',
                    '--cpu-pair', str(pair[0]), str(pair[1]), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=log, start_new_session=True)
            except BaseException:
                log.close(); raise
            self.workers[slot] = (process, log)
        process, _ = self.workers[slot]
        if process.returncode is not None:raise RuntimeError('owned canary worker stopped; no automatic retry')
        line = json.dumps(dict(path=str(request_path), sha256=c.file_sha(request_path))) + '\n'
        process.stdin.write(line.encode()); await process.stdin.drain()
        reply = await process.stdout.readline()
        if not reply or json.loads(reply) != {'exit': 0}:raise RuntimeError('owned B2 cell failed; no automatic retry')
        return _read_result(self.root / 'fp-results' / Path(request_path).name)
    async def close(self):
        errors = []
        for process, log in self.workers.values():
            try:await _finish_owned_cleanup(process)
            except BaseException as error:errors.append(error)
            finally:log.flush();os.fsync(log.fileno());log.close()
        self.workers.clear()
        if errors:raise errors[0]


def paired_interval(rows, *, seed, blocks=500, replicates=10000):
    """Original seed blocks keep both seats together and every missing cell."""
    from types import SimpleNamespace
    from .capstone_analysis import _percentile, _stratified_bootstrap_means
    if (type(blocks)is not int or blocks<=0 or type(seed)is not int or not 0<=seed<2**64
            or type(replicates)is not int or replicates<=0 or len(rows)!=blocks*4):
        raise ValueError('complete declared paired canary roster/bootstrap required')
    outcomes={}
    for ordinal,row in enumerate(rows):
        offset, seat, arm = row['offset'], row['seat'], row['arm']
        if (row['ordinal']!=ordinal or ordinal!=4*offset+2*seat+(arm=='CANARY')
                or not 0<=offset<blocks or seat not in (0,1) or arm not in ('CONTROL','CANARY')):
            raise ValueError('original fixed pair order required')
        if row['status'] not in ('COMPLETE','UNKNOWN','NOT_ATTEMPTED'):raise ValueError('literal endpoint disposition required')
        value=row['score'] if row['status']=='COMPLETE' else None
        if row['status']=='COMPLETE' and (type(value)not in (int,float) or value not in (0.,.5,1.)):
            raise ValueError('actual terminal score required')
        if row['status']!='COMPLETE' and row['score'] is not None:raise ValueError('unknown cannot become score')
        outcomes[offset,seat,arm]=value
    lower=[];upper=[];matched=[];whole=[];unknown=0
    for offset in range(blocks):
        lo=hi=0.;known=[]
        for seat in (0,1):
            a,b=outcomes[offset,seat,'CANARY'],outcomes[offset,seat,'CONTROL']
            lo+=(0. if a is None else a)-(1. if b is None else b)
            hi+=(1. if a is None else a)-(0. if b is None else b)
            unknown+=int(a is None)+int(b is None)
            if a is not None and b is not None:matched.append(a-b);known.append(a-b)
        lower.append(lo/2);upper.append(hi/2)
        if len(known)==2:whole.append(sum(known)/2)
    def draws(values):
        if len(set(values))==1:return (values[0],)*replicates
        return _stratified_bootstrap_means(tuple(SimpleNamespace(band='CANARY',delta=v) for v in values),
            replicates=replicates,seed=seed)
    interval=[_percentile(tuple(draws(lower)),.05),_percentile(tuple(draws(upper)),.95)]
    return dict(paired_seed_bootstrap_90=interval, bootstrap=dict(seed=seed,replicates=replicates,confidence=.9,
        unit='ORIGINAL_SEED_BLOCK_BOTH_SEATS',method='PAIRED_PERCENTILE_MISSING_CELL_ENVELOPE'),
        full_roster_identification_bounds=[sum(lower)/blocks,sum(upper)/blocks],
        matched_complete_cell_mean=None if not matched else sum(matched)/len(matched),
        complete_seed_point=None if not whole else sum(whole)/len(whole),unknown_cells_across_models=unknown,
        canary_failed=interval[1]<0, equivalence_established=False,
        failure_rule='FAIL_ONLY_IF_PAIRED_90_INTERVAL_ENTIRELY_BELOW_ZERO',
        stop_rule='one canary; at most one mismatch-driven tracker repair and one retry; never a third',
        uncertainty='Approximate paired seed-block percentile interval; missing-cell bounds are not a confidence interval')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True);args=parser.parse_args()
    raise SystemExit(asyncio.run(worker(args.request,args.expected_request_sha256)))
