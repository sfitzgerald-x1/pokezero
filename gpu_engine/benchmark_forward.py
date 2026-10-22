"""Actual champion forward-only cost on fixed real-team placeholder inputs.

Not end-to-end steps, games/second, production encoder parity or a go/no-go.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import jax
import numpy as np

from . import champion,observation,transition
from .benchmark import load_bound_pool,sole_gpu


def root_indices(batch,team_count,*,fixture=False):
    if team_count<2 or (not fixture and (team_count<200000 or batch*2>team_count)):
        raise ValueError('accepted mode needs >=200000 teams and distinct prefix pairs; smaller pools require explicit fixture mode')
    return np.arange(batch*2,dtype=np.int64).reshape(batch,2)%team_count


def measure(pool_path,weights_path,expected_pool_sha,*,batches=(256,1024,2048,8192),warmup=5,repeats=100,fixture=False):
    if (not batches or any(type(b)is not int or b<=0 for b in batches)
            or type(warmup)is not int or warmup<1 or type(repeats)is not int or repeats<2):
        raise ValueError('positive batches/warmup and at least two repeats required')
    device=sole_gpu();pool,metadata,pool_sha=load_bound_pool(pool_path)
    if pool_sha!=expected_pool_sha:raise ValueError('accepted pool binding differs')
    root=Path(__file__).resolve().parents[1]
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    if git('status','--porcelain','--','gpu_engine'):
        raise RuntimeError('GPU source must be clean for a bound timing')
    source_commit=git('rev-parse','HEAD')
    package_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'gpu_engine').glob('*.py'))}
    with jax.default_device(device):params,c,weightmeta=champion.load(weights_path)
    params=jax.device_put(params,device)
    if (c.tokens,c.numeric,c.categorical)!=(23,132,41):raise ValueError('placeholder shape differs from weights')
    with jax.default_device(device):table=transition.tables_from_metadata(metadata)
    result={'schema':'pokezero.gpu_forward_only.v1','device':device.device_kind,'jax':jax.__version__,
        'source_commit':source_commit,'package_sha256':package_hashes,'source_clean':True,
        'pool_mode':'fixture diagnostic, not accepted-pool measurement' if fixture else 'accepted >=200000-team pool',
        'root_selection':'consecutive ordered pairs (2r,2r+1), modulo team_count only in explicit fixture mode',
        'team_pool_sha256':pool_sha,'pool_teams':len(pool['species']),'weights':weightmeta,'warmup':warmup,'repeats':repeats,
        'timing':'FP32 HIGHEST, both seats, all three raw heads, synchronized whole output; host dispatch included; encode/setup/compile/warmup/sampling excluded',
        'scope':'Fixed real-team roots and placeholder observations; no games or qualified move coverage; no end-to-end/go-no-go conclusion','rows':[]}
    compiled=jax.jit(lambda n,k,t,a:champion.forward(params,n,k,t,a,c))
    for batch in batches:
        indices=root_indices(batch,len(pool['species']),fixture=fixture)
        with jax.default_device(device):
            state=transition.state_from_pool(pool,indices,seed=719,table=table)
            obs=observation.encode(state,table)
        args=jax.device_put((obs.numeric_features,obs.categorical_ids,obs.token_type_ids,obs.attention_mask),device)
        jax.block_until_ready(args)
        for _ in range(warmup):out=jax.block_until_ready(compiled(*args))
        if any(x.devices()!={device} for x in out):raise RuntimeError('forward did not execute solely on the recorded GPU')
        if not all(np.isfinite(np.asarray(x)).all() for x in out):raise RuntimeError('nonfinite champion output')
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter_ns();jax.block_until_ready(compiled(*args))
            samples.append((time.perf_counter_ns()-start)/1e6)
        row={'batch':batch,'seat_rows':batch*2,'unique_ordered_roots':len(np.unique(indices,axis=0)),
            'repeated_roots':batch-len(np.unique(indices,axis=0)),'forward_ms':statistics.fmean(samples),
            'median_forward_ms':statistics.median(samples),'samples_ms':samples}
        result['rows'].append(row);print(json.dumps(row),flush=True)
    try:
        result['device_counters_after']=subprocess.check_output(['nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):result['device_counters_after']=None
    if (git('rev-parse','HEAD')!=source_commit or git('status','--porcelain','--','gpu_engine')
            or package_hashes!={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'gpu_engine').glob('*.py'))}):
        raise RuntimeError('bound source changed during timing')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--team-pool',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--expected-pool-sha',required=True)
    p.add_argument('--out',required=True);p.add_argument('--batches',type=int,nargs='+',default=[256,1024,2048,8192])
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--fixture',action='store_true',help='explicit smaller-pool/repeated-root diagnostic, not accepted measurement')
    a=p.parse_args();result=measure(a.team_pool,a.weights,a.expected_pool_sha,
        batches=tuple(a.batches),warmup=a.warmup,repeats=a.repeats,fixture=a.fixture)
    with Path(a.out).open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
