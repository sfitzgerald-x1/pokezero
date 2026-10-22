"""Actual champion forward-only cost on fixed real-team placeholder inputs.

Not end-to-end steps, games/second, production encoder parity or a go/no-go.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import jax
import numpy as np

from . import champion,observation,transition
from .benchmark import load_bound_pool,sole_gpu


def measure(pool_path,weights_path,expected_pool_sha,*,batches=(256,1024,2048,8192),warmup=5,repeats=100):
    if (not batches or any(type(b)is not int or b<=0 for b in batches)
            or type(warmup)is not int or warmup<1 or type(repeats)is not int or repeats<2):
        raise ValueError('positive batches/warmup and at least two repeats required')
    device=sole_gpu();pool,metadata,pool_sha=load_bound_pool(pool_path)
    if pool_sha!=expected_pool_sha:raise ValueError('accepted pool binding differs')
    params,c,weightmeta=champion.load(weights_path)
    if (c.tokens,c.numeric,c.categorical)!=(23,132,41):raise ValueError('placeholder shape differs from weights')
    table=transition.tables_from_metadata(metadata)
    result={'schema':'pokezero.gpu_forward_only.v1','device':device.device_kind,'jax':jax.__version__,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'team_pool_sha256':pool_sha,'weights':weightmeta,'warmup':warmup,'repeats':repeats,
        'timing':'FP32 HIGHEST, both seats, all three raw heads, synchronized whole output; host dispatch included; encode/setup/compile/warmup/sampling excluded',
        'scope':'Fixed real-team roots and placeholder observations; no games or qualified move coverage; no end-to-end/go-no-go conclusion','rows':[]}
    compiled=jax.jit(lambda n,k,t,a:champion.forward(params,n,k,t,a,c))
    for batch in batches:
        indices=np.arange(batch*2,dtype=np.int64).reshape(batch,2)%len(pool['species'])
        state=transition.state_from_pool(pool,indices,seed=719,table=table)
        obs=observation.encode(state,table)
        args=(obs.numeric_features,obs.categorical_ids,obs.token_type_ids,obs.attention_mask)
        jax.block_until_ready(args)
        for _ in range(warmup):out=jax.block_until_ready(compiled(*args))
        if not all(np.isfinite(np.asarray(x)).all() for x in out):raise RuntimeError('nonfinite champion output')
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter_ns();jax.block_until_ready(compiled(*args))
            samples.append((time.perf_counter_ns()-start)/1e6)
        row={'batch':batch,'seat_rows':batch*2,'forward_ms':statistics.fmean(samples),
            'median_forward_ms':statistics.median(samples),'samples_ms':samples}
        result['rows'].append(row);print(json.dumps(row),flush=True)
    try:
        result['device_counters_after']=subprocess.check_output(['nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):result['device_counters_after']=None
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--team-pool',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--expected-pool-sha',required=True)
    p.add_argument('--out',required=True);p.add_argument('--batches',type=int,nargs='+',default=[256,1024,2048,8192])
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args();result=measure(a.team_pool,a.weights,a.expected_pool_sha,
        batches=tuple(a.batches),warmup=a.warmup,repeats=a.repeats)
    with Path(a.out).open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
