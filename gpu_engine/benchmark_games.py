"""Advancing finite-cohort cost-shape diagnostic, not qualified gameplay.

Uses the real unchanged champion on the explicitly non-production placeholder
observation. Counts actual completions rather than dividing pseudo-requests by
ordinary turns. No refill/redraw; unfinished games remain censored at the cap.
"""
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

from . import observation,transition
from .benchmark import load_bound_pool,sole_gpu


def sample_actions(logits,legal,keys):
    if logits.shape!=legal.shape or keys.shape!=legal.shape[:-1]+(2,):
        raise ValueError('policy/mask/key shapes differ')
    live=j.any(legal,axis=-1)
    masked=j.where(legal,logits,-j.inf)
    safe=j.where(live[...,None],masked,0.)
    draws=jax.vmap(jax.random.categorical)(keys.reshape((-1,2)),safe.reshape((-1,9)))
    return j.where(live,draws.reshape(live.shape),-1)


def driver(forward,table):
    """Inject the actual forward (or a focused test double), no host callbacks."""
    def request(state,index,keys):
        obs=observation.encode(state,table)
        heads=forward(obs)
        drawn=jax.vmap(lambda key:jax.random.fold_in(key,index))(keys.reshape((-1,2))).reshape(keys.shape)
        actions=sample_actions(heads.policy_logits,obs.legal_action_mask,drawn)
        # No-legal terminal/wait seats have no action; the transition determines
        # which seats are actually requested and freezes terminal battles.
        result=transition.step(state,j.maximum(0,actions),table)
        finite=j.all(j.stack([j.all(j.isfinite(x)) for x in heads]))
        return result,heads,actions,finite
    return jax.jit(request)


def cohort_summary(done,turns,*,seconds,requests,live_requests,ordinary_turns):
    done=np.asarray(done,bool);turns=np.asarray(turns)
    if done.ndim!=1 or turns.shape!=done.shape or not len(done):
        raise ValueError('one completion/turn entry per fixed-roster game required')
    if not np.isfinite(seconds) or seconds<=0 or requests<1:
        raise ValueError('positive elapsed time and request count required')
    completed=int(done.sum())
    return dict(completed_games=completed,censored_games=int((~done).sum()),
        games_per_second=completed/seconds,elapsed_seconds=seconds,lockstep_requests=requests,
        mean_end_to_end_request_ms=seconds*1000/requests,
        live_game_requests=int(live_requests),live_game_requests_per_second=live_requests/seconds,
        ordinary_game_turns=int(ordinary_turns),ordinary_game_turns_per_second=ordinary_turns/seconds,
        mean_completed_game_turns=float(np.mean(turns[done])) if completed else None,
        final_turns=turns.tolist(),completed_mask=done.tolist())


def _samples(call,*,warmup,repeats):
    for _ in range(warmup):jax.block_until_ready(call())
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();jax.block_until_ready(call())
        values.append((time.perf_counter_ns()-start)/1e6)
    return dict(mean_ms=statistics.fmean(values),samples_ms=values)


def render_markdown(result):
    lines=['# Advancing GPU cost shape diagnostic','',
        f"Hardware: {result['device']}; JAX {result['jax']}; one GPU; FP32/HIGHEST.",'',
        'Real champion, both seats, non-production placeholder observation and unqualified mechanics.',
        'Actual completed cohort games divided by advancing wall time; no refill. Unfinished games remain censored.',
        'Completed-game turn means exclude censored games and are not used to project a completion rate.',
        'Component latencies are separate fixed-root samples, not summands of the advancing mean.','',
        '| Batch | Engine ms | Encode ms | Forward ms | Advancing request ms | Completed | Censored | Games/s | Completed mean turns |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result['rows']:
        mean='—' if r['mean_completed_game_turns'] is None else f"{r['mean_completed_game_turns']:.2f}"
        lines.append(f"| {r['batch']} | {r['engine']['mean_ms']:.3f} | {r['encode']['mean_ms']:.3f} | {r['forward']['mean_ms']:.3f} | {r['mean_end_to_end_request_ms']:.3f} | {r['completed_games']} | {r['censored_games']} | {r['games_per_second']:.3f} | {mean} |")
    lines.extend(['','Request caps, per-game final turns/completion masks, ordinary-turn/live-request rates and raw samples are in JSON.',
        'No production fidelity, strength or go/no-go conclusion follows. Device counters are after-only, not average utilization.'])
    return '\n'.join(lines)+'\n'


def measure(pool_path,weights_path,expected_pool_sha,*,batches=(256,1024,2048,8192),
            warmup=5,repeats=100,max_requests=1000):
    # Import only at execution: the champion port is a separately reviewed PR.
    from . import champion
    if (not batches or any(type(b)is not int or b<=0 for b in batches)
            or any(type(v)is not int or v<1 for v in (warmup,max_requests))
            or type(repeats)is not int or repeats<2):
        raise ValueError('positive batches/warmup/request cap and >=2 repeats required')
    device=sole_gpu();pool,metadata,pool_sha=load_bound_pool(pool_path)
    if (pool_sha!=expected_pool_sha or len(pool['species'])<200000
            or metadata.get('completed_teams')!=len(pool['species']) or 2*max(batches)>len(pool['species'])):
        raise ValueError('bound complete >=200000-team pool with distinct prefix pairs required')
    root=Path(__file__).resolve().parents[1]
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    source=git('rev-parse','HEAD')
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'gpu_engine').glob('*.py'))}
    package_hashes=hashes()
    if git('status','--porcelain','--','gpu_engine'):raise RuntimeError('dirty GPU source')
    with jax.default_device(device):
        params,config,weightmeta=champion.load(weights_path)
        table=transition.tables_from_metadata(metadata)
    params=jax.device_put(params,device)
    if (config.tokens,config.numeric,config.categorical)!=(23,132,41):raise ValueError('placeholder shape differs')
    forward=lambda o:champion.forward(params,o.numeric_features,o.categorical_ids,o.token_type_ids,o.attention_mask,config)
    encode=jax.jit(lambda s:observation.encode(s,table));network=jax.jit(forward)
    core=jax.jit(lambda s,a:transition.step(s,a,table));advance=driver(forward,table)
    result=dict(schema='pokezero.gpu_advancing_cost_shape.v1',source_commit=source,
        source_clean=True,package_sha256=package_hashes,device=device.device_kind,jax=jax.__version__,
        team_pool_sha256=pool_sha,pool_teams=len(pool['species']),weights=weightmeta,
        warmup=warmup,repeats=repeats,max_requests=max_requests,rows=[],
        root_selection='consecutive distinct prefix team pairs (2r,2r+1); same prefixes across batch sizes; no refill/redraw',
        scope='Unqualified mechanics and omniscient placeholder observation; all moves remain flagged placeholders. No strength/production/go-no-go claim.',
        accounting='Actual completed cohort games / advancing elapsed wall time; censored games excluded from completed-turn mean, not silently dropped from roster. Live requests and ordinary turns reported separately; no pseudo-request/turn-rate conversion.',
        timing='FP32 HIGHEST both seats/all three heads; synchronized host walltime. Setup/compile/warmup excluded; advancing timing includes sampling, invalid/nonfinite/done checks and per-request host synchronization. Fixed-root component samples are separate from advancing run.')
    for batch in batches:
        with jax.default_device(device):
            state=transition.state_from_pool(pool,np.arange(2*batch).reshape(batch,2),seed=719,table=table)
            keys=jax.vmap(lambda n:jax.random.fold_in(jax.random.PRNGKey(9771),n))(j.arange(2*batch)).reshape((batch,2,2))
        state,keys=jax.device_put((state,keys),device);obs=jax.block_until_ready(encode(state))
        heads=jax.block_until_ready(network(obs))
        if any(x.devices()!={device} for x in heads):raise RuntimeError('forward not on sole GPU')
        actions=j.maximum(0,sample_actions(heads.policy_logits,obs.legal_action_mask,keys))
        row=dict(batch=batch,engine=_samples(lambda:core(state,actions),warmup=warmup,repeats=repeats),
            encode=_samples(lambda:encode(state),warmup=warmup,repeats=repeats),
            forward=_samples(lambda:network(obs),warmup=warmup,repeats=repeats))
        for _ in range(warmup):jax.block_until_ready(advance(state,j.array(0,j.int32),keys))
        done=np.asarray(state.done);live_requests=0;requests=0;status='REQUEST_CAP'
        start=time.perf_counter_ns()
        for index in range(max_requests):
            old_done=done;live_requests+=int((~old_done).sum())
            out,heads,actions,finite=jax.block_until_ready(advance(state,j.asarray(index,j.int32),keys))
            state=out.state;done=np.asarray(state.done);requests=index+1
            if bool(np.asarray(j.any(out.invalid_actions))) or not bool(np.asarray(finite)):
                status='INVALID_ACTION_OR_NONFINITE';break
            if done.all():status='COMPLETE_COHORT';break
            if requests%50==0:print(json.dumps(dict(batch=batch,requests=requests,completed=int(done.sum()))),flush=True)
        elapsed=(time.perf_counter_ns()-start)/1e9
        turns=np.asarray(state.turn)
        row.update(status=status,**cohort_summary(done,turns,seconds=elapsed,requests=requests,
            live_requests=live_requests,ordinary_turns=int(turns.sum())))
        result['rows'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('engine','encode','forward','final_turns','completed_mask')}),flush=True)
        if status=='INVALID_ACTION_OR_NONFINITE':break
    result['protocol_completed']=len(result['rows'])==len(batches) and all(r['status']!='INVALID_ACTION_OR_NONFINITE' for r in result['rows'])
    if (git('rev-parse','HEAD')!=source or git('status','--porcelain','--','gpu_engine') or hashes()!=package_hashes):
        raise RuntimeError('bound source changed during measurement')
    try:result['device_counters_after']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):result['device_counters_after']=None
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team-pool',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--expected-pool-sha',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[256,1024,2048,8192])
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--max-requests',type=int,default=1000)
    a=p.parse_args();paths=[Path(a.out),Path(a.out).with_suffix('.md')]
    if len(set(paths))!=2 or any(path.exists() for path in paths):
        raise FileExistsError('exclusive distinct JSON/Markdown output paths required')
    result=measure(a.team_pool,a.weights,a.expected_pool_sha,batches=tuple(a.batches),
        warmup=a.warmup,repeats=a.repeats,max_requests=a.max_requests)
    with paths[0].open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    with paths[1].open('x') as stream:stream.write(render_markdown(result))
    raise SystemExit(0 if result['protocol_completed'] else 1)
