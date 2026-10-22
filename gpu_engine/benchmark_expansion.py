"""81-joint-action cost shape, not a search or qualified battle engine."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import jax
import jax.numpy as j
import numpy as np

from . import transition
from .benchmark import load_bound_pool,sole_gpu


def joint_actions(nodes):
    if type(nodes)is not int or nodes<1:raise ValueError('positive node count required')
    ids=j.arange(81,dtype=j.int32)
    pairs=j.stack((ids//9,ids%9),axis=1)
    return j.broadcast_to(pairs,(nodes,81,2)).reshape((-1,2))


def expand(parents,table):
    """Replicate parents INSIDE the measured call; preserve a CRN per parent.

    Illegal joint requests are still computed slots, but frozen and explicitly
    flagged by the transition. They are not valid child-state transitions.
    """
    children=jax.tree.map(lambda a:j.repeat(a,81,axis=0),parents)
    return transition.step(children,joint_actions(parents.hp.shape[0]),table)


def render_markdown(result):
    lines=['# Joint action expansion cost shape','',
        f"Hardware: {result['device']}; JAX {result['jax']}; one GPU.",'',
        'Each parent expands all 81 action pairs; illegal requests return frozen states and are counted separately.',
        'Parent replication and complete output synchronization are timed. Chance is shared across siblings per parent.',
        'Unqualified mechanics; no actual search, game throughput, strength or go/no-go claim.','',
        '| Parent nodes | Mean ms | Node expansions/s | Child slots/s | Valid child transitions | Valid transitions/s |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result['rows']:
        lines.append(f"| {r['nodes']} | {r['mean_ms']:.3f} | {r['node_expansions_per_second']:.1f} | {r['child_slots_per_second']:.1f} | {r['valid_child_transitions']} | {r['valid_child_transitions_per_second']:.1f} |")
    lines.extend(['','Raw samples, source bindings and after-only device counters are in JSON. Counters are not mean utilization.'])
    return '\n'.join(lines)+'\n'


def measure(pool_path,expected_pool_sha,*,nodes=(256,1024,4096),warmup=5,repeats=100):
    if (not nodes or any(type(n)is not int or n<1 for n in nodes)
            or type(warmup)is not int or warmup<1 or type(repeats)is not int or repeats<2):
        raise ValueError('positive node counts/warmup and >=2 repeats required')
    device=sole_gpu();pool,meta,digest=load_bound_pool(pool_path)
    if (digest!=expected_pool_sha or len(pool['species'])<200000
            or meta.get('completed_teams')!=len(pool['species']) or max(nodes)*2>len(pool['species'])):
        raise ValueError('bound complete >=200000-team pool and distinct parent pairs required')
    root=Path(__file__).resolve().parents[1]
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'gpu_engine').glob('*.py'))}
    source=git('rev-parse','HEAD');bound_hashes=hashes()
    if git('status','--porcelain','--','gpu_engine'):raise RuntimeError('dirty GPU source')
    with jax.default_device(device):table=transition.tables_from_metadata(meta)
    compiled=jax.jit(expand)
    result=dict(schema='pokezero.gpu_joint_expansion.v1',source_commit=source,source_clean=True,
        package_sha256=bound_hashes,device=device.device_kind,jax=jax.__version__,pool_teams=len(pool['species']),
        team_pool_sha256=digest,warmup=warmup,repeats=repeats,rows=[],
        scope='All moves remain placeholders/unqualified. Engine cost shape only, not a search/game/strength/go-no-go result.',
        workload='Consecutive distinct prefix parent team pairs (2r,2r+1), same prefixes across sizes. All81 lexicographic joint actions per parent; invalid child slots freeze and are counted separately. Original per-parent chance key is copied to all siblings (CRN).',
        timing='Synchronized complete-output host walltime including parent replication and dispatch, excluding setup/compile/warmup.')
    for n in nodes:
        with jax.default_device(device):parents=transition.state_from_pool(pool,np.arange(2*n).reshape(n,2),seed=719,table=table)
        parents=jax.device_put(parents,device)
        for _ in range(warmup):out=jax.block_until_ready(compiled(parents,table))
        if any(x.devices()!={device} for x in jax.tree.leaves(out)):
            raise RuntimeError('expanded output not solely on recorded GPU')
        valid=int(np.sum(~np.asarray(out.invalid_actions)))
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter_ns();jax.block_until_ready(compiled(parents,table))
            samples.append((time.perf_counter_ns()-start)/1e6)
        ms=statistics.fmean(samples)
        row=dict(nodes=n,child_slots=81*n,valid_child_transitions=valid,invalid_frozen_slots=81*n-valid,
            mean_ms=ms,samples_ms=samples,node_expansions_per_second=n*1000/ms,
            child_slots_per_second=81*n*1000/ms,valid_child_transitions_per_second=valid*1000/ms)
        result['rows'].append(row);print(json.dumps(row),flush=True)
    if (source!=git('rev-parse','HEAD') or git('status','--porcelain','--','gpu_engine') or hashes()!=bound_hashes):
        raise RuntimeError('bound source changed during timing')
    try:result['device_counters_after']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):result['device_counters_after']=None
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--team-pool',required=True)
    p.add_argument('--expected-pool-sha',required=True);p.add_argument('--out',required=True)
    p.add_argument('--nodes',type=int,nargs='+',default=[256,1024,4096])
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args();paths=[Path(a.out),Path(a.out).with_suffix('.md')]
    if len(set(paths))!=2 or any(x.exists() for x in paths):raise FileExistsError('exclusive JSON/Markdown outputs required')
    result=measure(a.team_pool,a.expected_pool_sha,nodes=tuple(a.nodes),warmup=a.warmup,repeats=a.repeats)
    with paths[0].open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    with paths[1].open('x') as f:f.write(render_markdown(result))
