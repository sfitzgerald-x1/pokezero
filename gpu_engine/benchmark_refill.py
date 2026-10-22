"""Bounded refill cost-shape diagnostic; not qualified self-play or strength.

Every game uses the next unused real-team pair. No wrap, redraw, or conversion
of pseudo-requests into games. Reset/gather and host accounting are timed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as j
import numpy as np

from . import champion, observation, transition
from .benchmark import load_bound_pool, sole_gpu
from .benchmark_games import sample_actions


def policy_keys(ids, requests):
    """Per-game/per-seat/per-request streams independent of slot and batch."""
    def game(i, r):
        key=jax.random.fold_in(jax.random.PRNGKey(9771), j.maximum(i, 0))
        key=jax.random.fold_in(key, r)
        return jax.vmap(lambda seat:jax.random.fold_in(key, seat))(j.arange(2))
    return jax.vmap(game)(ids, requests)


def refill(state, ids, requests, cursor, initial, *, max_requests):
    """Retire each assigned ID once; freeze empty slots after pool exhaustion."""
    active=ids>=0
    completed=active & state.done
    capped=active & ~state.done & (requests>=max_requests)
    retired=completed|capped
    available=retired|(ids<0)
    assigned=cursor+j.cumsum(available.astype(j.int32))-1
    fill=available & (assigned<initial.done.shape[0])
    safe=j.clip(assigned,0,initial.done.shape[0]-1)
    replacement=jax.tree.map(lambda x:x[safe], initial)
    state=jax.tree.map(lambda old,new:j.where(fill.reshape((len(ids),)+(1,)*(old.ndim-1)),new,old),state,replacement)
    new_ids=j.where(fill,assigned,j.where(retired,-1,ids))
    # A capped slot must be terminal/frozen if no replacement remains.
    state=state._replace(done=state.done | (new_ids<0))
    new_requests=j.where(fill,0,requests)
    new_cursor=cursor+j.sum(fill,dtype=j.int32)
    return state,new_ids,new_requests,new_cursor,j.where(retired,ids,-1),completed,capped


def driver(forward, table, *, max_requests):
    def advance(state, ids, requests, cursor, initial):
        obs=observation.encode(state,table)
        heads=forward(obs)
        actions=sample_actions(heads.policy_logits,obs.legal_action_mask,policy_keys(ids,requests))
        out=transition.step(state,j.maximum(0,actions),table)
        requests=requests+(ids>=0).astype(j.int32)
        turns=j.where((ids>=0)&out.state.done,out.state.turn,0)
        next_values=refill(out.state,ids,requests,cursor,initial,max_requests=max_requests)
        finite=j.all(j.stack([j.all(j.isfinite(x)) for x in heads]))
        invalid=j.any(out.invalid_actions)|~finite
        return next_values,turns,invalid,heads
    return jax.jit(advance)


def summarize(retired, completed, capped, *, seconds, requests, assigned, batch, turns):
    if seconds<=0 or requests<1 or len(set(retired))!=len(retired):
        raise ValueError('positive measured window and exactly-once retired IDs required')
    if completed+capped!=len(retired) or assigned<batch or not 0<=assigned-len(retired)<=batch:
        raise ValueError('completion/cap/in-flight accounting mismatch')
    return dict(completed_games=completed,capped_games=capped,assigned_games=assigned,
        in_flight_games=assigned-len(retired),elapsed_seconds=seconds,lockstep_requests=requests,
        games_per_second=completed/seconds,mean_request_ms=seconds*1000/requests,
        mean_completed_game_turns=turns/completed if completed else None)


def measure(pool_path,weights_path,expected_pool_sha,*,batches=(256,1024,2048,8192),
            warm_seconds=30,measure_seconds=60,max_requests=2000):
    if not batches or any(type(b)is not int or b<1 for b in batches):raise ValueError('positive batches required')
    if warm_seconds<=0 or measure_seconds<=0 or type(max_requests)is not int or max_requests<1:
        raise ValueError('positive windows and integer per-game request cap required')
    device=sole_gpu();pool,metadata,pool_sha=load_bound_pool(pool_path)
    games=100000
    if pool_sha!=expected_pool_sha or len(pool['species'])<2*games or max(batches)>games:
        raise ValueError('bound >=200000-team pool and batches <=100000 required')
    root=Path(__file__).resolve().parents[1]
    git=lambda *a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    source=git('rev-parse','HEAD')
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'gpu_engine').glob('*.py'))}
    package=hashes()
    if git('status','--porcelain','--','gpu_engine'):raise RuntimeError('dirty GPU package')
    with jax.default_device(device):
        params,config,weightmeta=champion.load(weights_path)
        table=transition.tables_from_metadata(metadata)
        initial=transition.state_from_pool(pool,np.arange(2*games).reshape(games,2),seed=719,table=table)
    initial,params=jax.block_until_ready(jax.device_put((initial,params),device))
    forward=lambda o:champion.forward(params,o.numeric_features,o.categorical_ids,o.token_type_ids,o.attention_mask,config)
    advance=driver(forward,table,max_requests=max_requests)
    result=dict(schema='pokezero.gpu_refill_cost_shape.v1',source_commit=source,source_clean=True,
        package_sha256=package,device=device.device_kind,jax=jax.__version__,weights=weightmeta,
        team_pool_sha256=pool_sha,pool_games=games,warm_seconds=warm_seconds,measure_seconds=measure_seconds,
        max_requests_per_game=max_requests,rows=[],
        scope='Unqualified mechanics, omniscient non-production placeholder observation, unchanged champion both seats/all3 heads; not strength, production self-play, fidelity or go/no-go.',
        roster='Fixed ordered pairs(2r,2r+1), no wrap/redraw. Same prefixes reused across batches, not independent cohorts.',
        timing='Synchronized host wall time includes refill, gather/reset, forward, sampling, transition and host accounting. Initial states/setup/compile excluded. Warmup advances; measured window starts from its in-flight state. Capped games retained separately, never completions.')
    for batch in batches:
        state=jax.tree.map(lambda x:x[:batch],initial)
        ids=j.arange(batch,dtype=j.int32);counts=j.zeros(batch,j.int32);cursor=j.asarray(batch,j.int32)
        # Compile without consuming any roster ID, then reset to the same initial state.
        probe=jax.block_until_ready(advance(state,ids,counts,cursor,initial))
        if any(x.devices()!={device} for x in probe[-1]):raise RuntimeError('forward not on sole GPU')
        all_retired=[];all_completed=0;all_capped=0;all_turns=0;status='WINDOW_COMPLETE'
        for phase,duration in (('warmup',warm_seconds),('measured',measure_seconds)):
            retired=[];completed=0;capped=0;turns=0;requests=0
            start=time.perf_counter()
            while time.perf_counter()-start<duration:
                values,ended_turns,invalid,heads=jax.block_until_ready(advance(state,ids,counts,cursor,initial))
                requests+=1
                if bool(np.asarray(invalid)):status='INVALID_ACTION_OR_NONFINITE';break
                state,ids,counts,cursor,ended,done,cap=values
                # These transfers/checks are inside the measured clock.
                ended=np.asarray(ended);done=np.asarray(done);cap=np.asarray(cap)
                retired.extend(ended[ended>=0].tolist());completed+=int(done.sum());capped+=int(cap.sum())
                turns+=int(np.asarray(ended_turns).sum())
                if int(np.asarray(cursor))>=games:status='POOL_EXHAUSTED';break
            elapsed=time.perf_counter()-start
            all_retired.extend(retired);all_completed+=completed;all_capped+=capped;all_turns+=turns
            if len(set(all_retired))!=len(all_retired):raise RuntimeError('duplicate retired game ID')
            record=dict(phase=phase,completed_games=completed,capped_games=capped,retired_ids=retired,
                elapsed_seconds=elapsed,lockstep_requests=requests,completed_turns=turns,
                cursor=int(np.asarray(cursor)))
            if phase=='warmup':warm=record
            else:measured=record
            print(json.dumps({k:v for k,v in record.items() if k!='retired_ids'}|dict(batch=batch,status=status)),flush=True)
            if status!='WINDOW_COMPLETE':break
        assigned=int(np.asarray(cursor))
        row=dict(batch=batch,status=status,warmup=warm,measured=measured if phase=='measured' else None,
            total=summarize(all_retired,all_completed,all_capped,seconds=warm['elapsed_seconds']+(measured['elapsed_seconds'] if phase=='measured' else 0),
                requests=warm['lockstep_requests']+(measured['lockstep_requests'] if phase=='measured' else 0),assigned=assigned,batch=batch,turns=all_turns),
            unstarted_games=games-assigned,final_ids=np.asarray(ids).tolist(),final_requests=np.asarray(counts).tolist())
        row['games_per_second']=measured['completed_games']/measured['elapsed_seconds'] if phase=='measured' and status=='WINDOW_COMPLETE' else None
        if status=='INVALID_ACTION_OR_NONFINITE':
            for field in ('games_per_second','mean_request_ms'):row['total'][field]=None
        result['rows'].append(row)
        if status!='WINDOW_COMPLETE':break
    result['protocol_completed']=len(result['rows'])==len(batches) and all(r['status']=='WINDOW_COMPLETE' for r in result['rows'])
    if git('rev-parse','HEAD')!=source or git('status','--porcelain','--','gpu_engine') or hashes()!=package:
        raise RuntimeError('bound GPU source changed')
    try:result['device_counters_after']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):result['device_counters_after']=None
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team-pool',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--expected-pool-sha',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[256,1024,2048,8192])
    p.add_argument('--warm-seconds',type=float,default=30);p.add_argument('--measure-seconds',type=float,default=60)
    p.add_argument('--max-requests',type=int,default=2000)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('exclusive output required')
    result=measure(a.team_pool,a.weights,a.expected_pool_sha,batches=tuple(a.batches),
        warm_seconds=a.warm_seconds,measure_seconds=a.measure_seconds,max_requests=a.max_requests)
    with out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    raise SystemExit(0 if result['protocol_completed'] else 1)
