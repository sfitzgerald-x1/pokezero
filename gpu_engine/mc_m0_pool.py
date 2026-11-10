"""One-pod handoff at a completed root, retaining every completed stage.

The only worker this controller can interrupt is the explicitly identified
serial mc_m0 process on the qualified output directory. It waits for both
stages of root zero, retains immutable results, then assigns unfinished roots
disjointly. No model, RNG, roster, K or environment count changes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from gpu_engine import mc_m0 as m0, mc_m0_partition as part, mc_m0_readout as readout


def process(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text();fields=stat[stat.rindex(')')+2:].split()
        if fields[0]=='Z':return None
        args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return dict(pid=pid,start=fields[19],ppid=int(fields[1]),args=[x.decode() for x in args if x],
                    rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                    cpu_seconds=sum(int(fields[n]) for n in (11,12,13,14))/os.sysconf('SC_CLK_TCK'))
    except FileNotFoundError:return None


def require_serial(pid, directory):
    info=process(pid)
    if info is None:raise ValueError('serial worker is terminal/missing; this controller does not retry it')
    args=info['args']
    if '-m' not in args or args[args.index('-m')+1]!='gpu_engine.mc_m0':raise ValueError('not the owned serial M0 worker')
    if '--out' not in args or Path(args[args.index('--out')+1]).resolve()!=Path(directory).resolve():
        raise ValueError('serial output identity differs')
    return info


def validate_stage(row, reference):
    i,k=row['index'],row['K']
    if type(i) is not int or i not in reference or k not in (64,256):raise ValueError('stage identity')
    if row['root']!=reference[i]['root'] or row['mode']!=('unbatched' if k==64 else 'batched'):raise ValueError('stage root/mode differs')
    if np.asarray(row['outcomes'],float).shape!=(len(row['root']['actions']),k):raise ValueError('stage outcome shape')
    for n in (64,256):
        if n<=k and row['pairs'][str(n)]!=m0.pair_stats(row['outcomes'],row['statuses'],n):raise ValueError('stage diagnostics differ')
    return i,k


def completed(directory, reference):
    stages={};incomplete=[]
    for path in sorted(Path(directory).glob('root-*.json')):
        try:row=json.loads(path.read_bytes())
        except json.JSONDecodeError:
            incomplete.append(dict(path=str(path),sha256=m0.sha(path)));continue
        key=validate_stage(row,reference)
        if key in stages:raise ValueError('duplicate original stage')
        stages[key]=dict(path=str(path),sha256=m0.sha(path))
    return stages,incomplete


def distribute(indices, workers):
    return [list(indices)[n::workers] for n in range(workers) if list(indices)[n::workers]]


def claim_handoff(directory,info,config_sha,out):
    path=Path(directory)/'handoff-claim.json'
    m0.write(path,dict(serial_pid=info['pid'],serial_start=info['start'],config_sha256=config_sha,
                      controller_pid=os.getpid(),pool_out=str(Path(out).resolve())))
    return m0.sha(path)


def capacity_guard(items,workers):
    limit_text=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if limit_text=='max':raise ValueError('bounded pod memory required')
    limit=int(limit_text);current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    serial_rss=sum(item['rss_bytes'] for item in items)
    projected=current+workers*serial_rss
    if projected>limit*.8:raise ValueError('insufficient memory headroom for the requested pool; original worker preserved')
    return dict(current_bytes=current,limit_bytes=limit,serial_rss_bytes=serial_rss,conservative_projected_bytes=projected,workers=workers)


def release_serial(info, directory, reference, accounting=None, workers=1):
    # Preserve a complete first-root measurement before any scheduling change.
    while True:
        current=require_serial(info['pid'],directory)
        if current['start']!=info['start']:raise ValueError('serial PID was recycled')
        if (Path(directory)/'failure-summary.json').exists():raise ValueError('serial failed; no automatic retry')
        stages,_=completed(directory,reference)
        if (0,64) in stages and (0,256) in stages:break
        time.sleep(1)
    children={};pending=[info['pid']]
    while pending:
        parent=pending.pop()
        task=Path(f'/proc/{parent}/task')
        if not task.exists():continue
        for thread in task.iterdir():
            try:pids=[int(x) for x in (thread/'children').read_text().split()]
            except FileNotFoundError:continue
            for pid in pids:
                child=process(pid)
                if child and pid not in children:children[pid]=child;pending.append(pid)
    current=require_serial(info['pid'],directory)
    if current['start']!=info['start']:raise ValueError('serial identity changed before handoff')
    capacity=capacity_guard([current,*children.values()],workers)
    if accounting is not None:accounting['capacity']=capacity
    observed_cpu=current['cpu_seconds']+sum(child['cpu_seconds'] for child in children.values())
    if accounting is not None:accounting['serial_cpu']=observed_cpu
    os.kill(info['pid'],signal.SIGTERM)
    for child in children.values():
        now=process(child['pid'])
        if now and now['start']==child['start']:os.kill(child['pid'],signal.SIGTERM)
    deadline=time.monotonic()+30
    def still_owned(item):
        now=process(item['pid']);return now is not None and now['start']==item['start']
    while any(still_owned(item) for item in [info,*children.values()]):
        if time.monotonic()>deadline:raise ValueError('serial process tree did not stop; no parallel dispatch')
        time.sleep(.1)
    if (Path(directory)/'failure-summary.json').exists():raise ValueError('serial failure evidence; no parallel dispatch')
    stages,incomplete=completed(directory,reference)
    return stages,incomplete,observed_cpu


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qualified-run',required=True)
    p.add_argument('--serial-pid',type=int,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    state=dict(created=False)
    try:return run(args,state)
    except Exception as error:
        if state['created']:
            children=state.get('children',[])
            if any(child.poll() is None for child in children):
                m0.write(Path(args.out)/'failure-pending-accounting.json',failure_accounting(state,error,False))
                print(json.dumps(dict(event='POOL_FAILED_WAITING_FOR_EXISTING_PARTITIONS',pids=[c.pid for c in children if c.poll() is None])),flush=True)
                for child in children:child.wait()
            m0.write(Path(args.out)/'failure-summary.json',failure_accounting(state,error,True))
        raise


def failure_accounting(state,error,terminal):
    return dict(status='FAILED',reason=type(error).__name__+': '+str(error),
                pool_process_tree_cpu_seconds=m0.process_cpu()-state['started_cpu'],
                pool_wall_seconds=time.perf_counter()-state['started_wall'],
                observed_serial_process_tree_cpu_seconds_before_handoff=state.get('serial_cpu'),
                terminal_accounting_complete=terminal,
                workers=[dict(pid=c.pid,exit_code=c.poll()) for c in state.get('children',[])])


def run(args,state):
    if not 1<=args.workers<=4:raise ValueError('one to four workers inside the existing CPU pod only')
    serial=Path(args.qualified_run).resolve();config,reference,manifest=part.qualify(serial)
    m0.validate(config);info=require_serial(args.serial_pid,serial)
    out=Path(args.out).resolve();out.mkdir(exist_ok=False,parents=True)
    state['created']=True;started_cpu=m0.process_cpu();started_wall=time.perf_counter()
    state.update(started_cpu=started_cpu,started_wall=started_wall)
    claim_sha=claim_handoff(serial,info,manifest['config_sha256'],out)
    m0.write(out/'intent.json',dict(serial=info,qualified_run=str(serial),workers=args.workers,config_sha256=manifest['config_sha256']))
    print(json.dumps(dict(event='WAITING_FOR_FIRST_COMPLETE_ROOT',pid=info['pid'])),flush=True)
    stages,incomplete,serial_cpu=release_serial(info,serial,reference,state,args.workers)
    state['serial_cpu']=serial_cpu
    remaining=[i for i in range(50) if (i,64) not in stages or (i,256) not in stages]
    groups=distribute(remaining,args.workers)
    plan=dict(qualified_run=str(serial),config_sha256=manifest['config_sha256'],core_sha256=manifest['source_sha256'],
              handoff_claim_sha256=claim_sha,
              adapter_sha256=m0.sha(part.__file__),pool_sha256=m0.sha(__file__),
              capacity=state['capacity'],
              groups=groups,reuse={f'{i}:{k}':v for (i,k),v in stages.items()},incomplete_original_files=incomplete,
              serial_release=dict(pid=info['pid'],start=info['start'],status='CONTROLLED_HANDOFF_AFTER_COMPLETE_ROOT',
                                  observed_process_tree_cpu_seconds=serial_cpu))
    m0.write(out/'plan.json',plan)
    children=[];state['children']=children
    worker_log=open(out/'workers.jsonl','x')
    for worker,indices in enumerate(groups):
        log=open(out/f'worker-{worker}.log','x')
        command=[sys.executable,'-B','-m','gpu_engine.mc_m0_partition','--qualified-run',str(serial),
                 '--indices',','.join(map(str,indices)),'--out',str(out/f'worker-{worker}'),'--plan',str(out/'plan.json'),'--worker',str(worker)]
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);log.close();children.append(child)
        worker_log.write(json.dumps(dict(worker=worker,pid=child.pid,indices=indices))+'\n');worker_log.flush()
    worker_log.close()
    codes=[child.wait() for child in children]
    m0.write(out/'worker-exits.json',codes)
    if any(codes):raise RuntimeError('partition failed; no retry or completion claim')
    collect(out,serial,reference,plan,m0.process_cpu()-started_cpu,time.perf_counter()-started_wall)
    print(json.dumps(dict(event='GLOBAL_M0_COMPLETE',roots=50)),flush=True)
    return 0


def collect(out,serial,reference,plan,pool_cpu,pool_wall):
    groups=plan['groups'];serial_cpu=plan['serial_release']['observed_process_tree_cpu_seconds']
    codes=json.loads((out/'worker-exits.json').read_bytes())
    if len(codes)!=len(groups) or any(type(code) is not int or code!=0 for code in codes):raise ValueError('successful worker exits required')
    collection=out/'collection';collection.mkdir()
    for name in ('config.json','manifest.json','prepared.json','runtime-smoke.json'):os.link(serial/name,collection/name)
    phases={}
    for key,value in plan['reuse'].items():
        if m0.sha(value['path'])!=value['sha256']:raise ValueError('retained stage changed')
        validate_stage(json.loads(Path(value['path']).read_bytes()),reference)
        i,k=map(int,key.split(':'));os.link(value['path'],collection/f'root-{i:03d}-{k}.json')
        phases[key]='original_single_worker'
    for worker,indices in enumerate(groups):
        directory=out/f'worker-{worker}'
        if (directory/'failure-summary.json').exists():raise ValueError('partition failure evidence')
        assignment=json.loads((directory/'assignment.json').read_bytes())
        if (assignment['indices']!=indices or assignment['config_sha256']!=plan['config_sha256']
                or assignment['core_sha256']!=plan['core_sha256'] or assignment['adapter_sha256']!=plan['adapter_sha256']
                or assignment['plan_sha256']!=m0.sha(out/'plan.json')):
            raise ValueError('partition source/assignment differs')
        summary=json.loads((directory/'partition-summary.json').read_bytes())
        if summary['status']!='PARTITION_COMPLETE' or summary['assigned']!=indices or summary['completed']!=indices:raise ValueError('partition coverage differs')
        for i in indices:
            for k in (64,256):
                source=directory/f'root-{i:03d}-{k}.json';validate_stage(json.loads(source.read_bytes()),reference)
                dest=collection/source.name
                if dest.exists():
                    if m0.sha(dest)!=m0.sha(source):raise ValueError('conflicting duplicate stage')
                else:os.link(source,dest);phases[f'{i}:{k}']='concurrent_partitions'
    m0.write(collection/'execution-phases.json',phases)
    interim=readout.readout(collection)
    if interim['completed_both_modes']!=50:raise ValueError('global 50-root coverage missing')
    m0.write(collection/'summary.json',dict(status='COMPLETE',completed=50,expected=50,
              execution='partitioned after controlled serial handoff; original exit remains recorded separately',
              pool_process_tree_cpu_seconds=pool_cpu,
              observed_serial_process_tree_cpu_seconds_before_handoff=serial_cpu,
              pool_wall_seconds=pool_wall))
    m0.write(out/'readout.json',readout.readout(collection))


if __name__=='__main__':raise SystemExit(main())
