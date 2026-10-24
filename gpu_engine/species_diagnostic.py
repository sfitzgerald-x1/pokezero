"""Finite seat-reversed self-play smell test, not a strength experiment.

Species team scores are descriptive, not tested against 50 percent: species,
partners and opponents have genuinely different strengths. The paired seat
contrast instead has a symmetry baseline of zero under identical seat policies.
Intervals are game-pair clustered, marginal exploratory intervals, not a family-
wise significance test. Extreme rows nominate oracle investigations only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def roster(start, pairs, pool_size):
    if any(type(x) is not int for x in (start, pairs, pool_size)) or start < 0 or pairs < 1 or start+2*pairs > pool_size:
        raise ValueError('in-range nonempty fixed pair roster required')
    teams=np.arange(start,start+2*pairs).reshape(pairs,2)
    return np.stack((teams,teams[:,::-1]),axis=1).reshape(-1,2)


def clustered_ratio(numerator, denominator):
    """Delta-method ratio SE with independent original team pairs as clusters."""
    y=np.asarray(numerator,float);n=np.asarray(denominator,float)
    if y.shape != n.shape or y.ndim != 1 or np.any(n<0) or np.any(~np.isfinite(y+n)):
        raise ValueError('finite equal-length cluster totals required')
    used=n>0;clusters=int(used.sum());total=float(n.sum())
    if total==0:return dict(estimate=None,clusters=0,standard_error=None,interval95=None)
    rate=float(y.sum()/total)
    se=float(np.sqrt(clusters/(clusters-1)*np.sum((y[used]-rate*n[used])**2))/total) if clusters>1 else None
    interval=[rate-1.96*se,rate+1.96*se] if se is not None and clusters>=30 else None
    return dict(estimate=rate,clusters=clusters,standard_error=se,interval95=interval)


def summarize(species, winners, attempted, active_requests, move_requests, vocabulary):
    """Retain unstarted/capped cells; all arrays are in original game/seat order."""
    species=np.asarray(species);winners=np.asarray(winners);attempted=np.asarray(attempted,bool)
    active_requests=np.asarray(active_requests);move_requests=np.asarray(move_requests)
    games=len(winners)
    if games%2 or species.shape!=(games,2,6) or attempted.shape!=(games,) or active_requests.shape!=species.shape or move_requests.shape!=species.shape:
        raise ValueError('paired game/seat/party accounting shapes differ')
    if (np.any(~np.isin(winners,[-1,0,1,2])) or np.any((winners>=0)&~attempted)
            or np.any(active_requests<0) or np.any(move_requests<0) or np.any(move_requests>active_requests)
            or not np.array_equal(species[::2],species[1::2,::-1])):
        raise ValueError('invalid outcome, participation or seat-reversal accounting')
    known=winners>=0
    scores=np.where(winners[:,None]==2,.5,(winners[:,None]==np.arange(2)).astype(float))
    scores=np.where(known[:,None],scores,0.)
    rows=[]
    for sid in np.unique(species):
        if sid==0:continue
        member=species==sid;present=member.any(axis=2)
        # Sum both orientations and both original teams WITHIN each pair.
        exposure=present.reshape(-1,2,2).sum(axis=(1,2))
        scored=(present*known[:,None]).reshape(-1,2,2).sum(axis=(1,2))
        points=(present*scores).reshape(-1,2,2).sum(axis=(1,2))
        unknown=int(exposure.sum()-scored.sum())
        rate=clustered_ratio(points,scored)
        # Seat denominators match by construction on the FULL fixed roster.
        # A missing game prevents an identified seat contrast; do not impute it.
        seat_points=(present*scores).reshape(-1,2,2).sum(axis=1)
        seat_exposure=present.reshape(-1,2,2).sum(axis=1)
        complete=np.repeat((known.reshape(-1,2).all(axis=1)),2)
        cpoints=(present*scores*complete[:,None]).reshape(-1,2,2).sum(axis=1)
        cexposure=(present*complete[:,None]).reshape(-1,2,2).sum(axis=1)
        seat=clustered_ratio(cpoints[:,0]-cpoints[:,1],cexposure[:,0])
        side_unknown=(present*~known[:,None]).reshape(-1,2,2).sum(axis=1).sum(axis=0)
        denom=float(seat_exposure[:,0].sum())
        difference=float((seat_points[:,0]-seat_points[:,1]).sum())
        rows.append(dict(species_id=int(sid),species=vocabulary[int(sid)],
            original_teams=int(present[::2].sum()),presence_games=int(exposure.sum()),
            scored_presence_games=int(scored.sum()),unknown_presence_games=unknown,
            team_score=rate,fixed_roster_score_bounds=[float(points.sum()/exposure.sum()),float((points.sum()+unknown)/exposure.sum())],
            seat0_minus_seat1_complete_pairs=seat,
            fixed_roster_seat_difference_bounds=[(difference-side_unknown[1])/denom,(difference+side_unknown[0])/denom],
            active_games=int(((member*active_requests).sum(axis=2)>0).sum()),
            active_requests=int((member*active_requests).sum()),
            move_requests=int((member*move_requests).sum()),
            unstarted_presence_games=int((present*~attempted[:,None]).sum())))
    eligible=[r for r in rows if r['team_score']['clusters']>=30 and r['scored_presence_games']>=100]
    # These are rankings, not rejection rules or evidence of a bug by themselves.
    low=sorted(eligible,key=lambda r:(r['team_score']['estimate'],r['species']))[:10]
    high=sorted(eligible,key=lambda r:(-r['team_score']['estimate'],r['species']))[:10]
    seat_eligible=[r for r in rows if r['seat0_minus_seat1_complete_pairs']['clusters']>=30]
    seat=sorted(seat_eligible,key=lambda r:(-abs(r['seat0_minus_seat1_complete_pairs']['estimate']),r['species']))[:10]
    return dict(games=games,attempted_games=int(attempted.sum()),unstarted_games=int((~attempted).sum()),
        terminal_games=int(known.sum()),capped_or_interrupted_games=int((attempted&~known).sum()),
        seat0_wins=int((winners==0).sum()),seat1_wins=int((winners==1).sum()),draws=int((winners==2).sum()),
        complete_game_pairs=int(known.reshape(-1,2).all(axis=1).sum()),species=rows,
        exploratory_triage=dict(low_team_scores=[r['species'] for r in low],high_team_scores=[r['species'] for r in high],largest_absolute_seat_differences=[r['species'] for r in seat]),
        interpretation='Marginal normal intervals cluster both orientations/original teams within original team pair. Scored rates exclude unknown games; full-roster worst-case bounds retain them. No species-50% null, causal species strength, multiplicity-adjusted test, fidelity qualification or training authorization. Complete-pair seat contrasts are exploratory symmetry checks; missing-pair selection is disclosed, not assumed random.')


def accept_step(previous, candidate, *, within_cap=True):
    """A globally failed step cannot create a newly accepted terminal game."""
    if not within_cap or bool(np.asarray(candidate[-1])):return previous,False
    return candidate[:-1],True


def driver(forward, table):
    """Count live requested actors; a selected move is not necessarily executed."""
    import jax
    import jax.numpy as j
    from . import observation, transition
    from .benchmark_games import sample_actions
    from .benchmark_refill import policy_keys
    def advance(state,ids,request,active,moves):
        obs=observation.encode(state,table)
        heads=forward(obs)
        acts=sample_actions(heads.policy_logits,obs.legal_action_mask,policy_keys(ids,j.full_like(ids,request)))
        hp=j.take_along_axis(state.hp,state.active[...,None],axis=2)[...,0]
        forced=state.forced_switch|(hp<=0)
        requested=~j.any(forced,axis=1,keepdims=True)|forced
        live=(~state.done[:,None]) & (hp>0) & requested & j.any(obs.legal_action_mask,axis=2)
        slots=jax.nn.one_hot(state.active,6,dtype=j.int32)
        active=active+slots*live[...,None]
        moves=moves+slots*(live & (acts>=0)&(acts<4))[...,None]
        result=transition.step(state,j.maximum(0,acts),table)
        finite=j.all(j.stack([j.all(j.isfinite(x)) for x in heads]))
        return result.state,active,moves,j.any(result.invalid_actions)|~finite
    return jax.jit(advance)


def measure(pool_path, weights_path, expected_pool_sha, *, start=140000, pairs=8192,
            batch=1024, max_requests=2000, seconds=1800, seed=20261023, out_dir):
    import jax
    import jax.numpy as j
    from . import champion, transition
    from .benchmark import load_bound_pool, sole_gpu
    if type(batch)is not int or batch<2 or batch%2 or pairs*2%batch or type(max_requests)is not int or max_requests<1 or not 0<seconds<=1800:
        raise ValueError('even complete-pair batches dividing roster, positive requests and <=1800 seconds required')
    folder=Path(out_dir);folder.mkdir(exist_ok=False,parents=True)
    start_clock=time.monotonic();device=sole_gpu();pool,metadata,digest=load_bound_pool(pool_path)
    if digest!=expected_pool_sha:raise ValueError('team-pool binding differs')
    teams=roster(start,pairs,len(pool['species']));species=pool['species'][teams]
    root=Path(__file__).resolve().parents[1]
    git=lambda *a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    source=git('rev-parse','HEAD')
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'gpu_engine').glob('*.py'))}
    package=hashes()
    if git('status','--porcelain','--','gpu_engine'):raise RuntimeError('dirty GPU package')
    with jax.default_device(device):
        params,config,weightmeta=champion.load(weights_path);table=transition.tables_from_metadata(metadata)
    params=jax.device_put(params,device)
    # Persist the complete fixed roster and bindings BEFORE the first advance.
    # After an external kill, every absent chunk is still an unknown fixed cell.
    entry=dict(schema='pokezero.gpu_species_smell_test_entry.v1',source_commit=source,
        package_sha256=package,team_pool_sha256=digest,weights=weightmeta,seed=seed,
        team_start=start,pairs=pairs,batch=batch,max_requests=max_requests,wall_cap_seconds=seconds,
        unknown_recovery='Join durable chunk game IDs to roster.npz; every absent game remains unknown, never redrawn or inferred terminal.')
    with (folder/'entry.json').open('x') as stream:json.dump(entry,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    with (folder/'roster.npz').open('xb') as stream:
        np.savez_compressed(stream,game_ids=np.arange(len(teams)),team_indices=teams,species=species)
    with (folder/'started-chunks.jsonl').open('x'):pass
    forward=lambda obs:champion.forward(params,obs.numeric_features,obs.categorical_ids,obs.token_type_ids,obs.attention_mask,config)
    advance=driver(forward,table)
    winners=np.full(2*pairs,-1,np.int32);attempted=np.zeros(2*pairs,bool)
    active=np.zeros_like(species,np.int32);moves=np.zeros_like(species,np.int32)
    turns=np.zeros(2*pairs,np.int32);requests=np.zeros(2*pairs,np.int32)
    status='ROSTER_COMPLETE';chunk_receipts=[]
    for offset in range(0,len(teams),batch):
        if time.monotonic()-start_clock>=seconds:status='WALL_CAP';break
        attempted[offset:offset+batch]=True
        with (folder/'started-chunks.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(offset=offset,games=batch,elapsed_seconds=time.monotonic()-start_clock))+'\n')
            stream.flush();os.fsync(stream.fileno())
        print(json.dumps(dict(event='BATCH_START',offset=offset,games=batch)),flush=True)
        with jax.default_device(device):
            state=transition.state_from_pool(pool,teams[offset:offset+batch],seed=seed)
            ids=j.arange(offset,offset+batch,dtype=j.int32)
            state=state._replace(keys=jax.vmap(lambda n:jax.random.fold_in(jax.random.PRNGKey(seed),n))(ids))
            state=transition.initialize(state,table)
            ca=j.zeros((batch,2,6),j.int32);cm=j.zeros_like(ca)
        before=time.monotonic();done=np.asarray(state.done)
        for request in range(max_requests):
            if time.monotonic()-start_clock>=seconds:status='WALL_CAP';break
            candidate=jax.block_until_ready(advance(state,ids,j.asarray(request,j.int32),ca,cm))
            requests[offset:offset+batch]+=(~done).astype(np.int32)
            within_cap=time.monotonic()-start_clock<seconds
            # An invalid action or nonfinite head invalidates this whole NEW
            # step. Preserve earlier valid outcomes and discard every newly
            # generated winner/participation update, not just the culprit row.
            accepted,valid=accept_step((state,ca,cm),candidate,within_cap=within_cap)
            if not valid:
                status='WALL_CAP' if not within_cap else 'INVALID_ACTION_OR_NONFINITE'
                break
            state,ca,cm=accepted;done=np.asarray(state.done)
            if done.all():break
        winners[offset:offset+batch]=np.asarray(state.winner)
        active[offset:offset+batch]=np.asarray(ca);moves[offset:offset+batch]=np.asarray(cm)
        turns[offset:offset+batch]=np.asarray(state.turn)
        receipt=dict(offset=offset,games=batch,terminal=int(done.sum()),requests=int(requests[offset:offset+batch].sum()),elapsed_seconds=time.monotonic()-before,status=status)
        chunk_receipts.append(receipt)
        # One fixed append-free snapshot per completed/interrupted batch. A hard
        # process loss still leaves earlier game records, never fictitious exits.
        with (folder/f'chunk-{offset:05d}.npz').open('xb') as stream:
            np.savez_compressed(stream,game_ids=np.arange(offset,offset+batch),team_indices=teams[offset:offset+batch],winner=winners[offset:offset+batch],active_requests=active[offset:offset+batch],move_requests=moves[offset:offset+batch],turns=turns[offset:offset+batch],requests=requests[offset:offset+batch])
        print(json.dumps(receipt),flush=True)
        if status!='ROSTER_COMPLETE':break
    summary=summarize(species,winners,attempted,active,moves,metadata['vocabulary']['species'])
    if git('rev-parse','HEAD')!=source or git('status','--porcelain','--','gpu_engine') or hashes()!=package:
        status='SOURCE_DRIFT'
    result=dict(schema='pokezero.gpu_species_smell_test.v1',status=status,source_commit=source,
        package_sha256=package,team_pool_sha256=digest,weights=weightmeta,device=device.device_kind,jax=jax.__version__,
        seed=seed,team_start=start,pairs=pairs,batch=batch,max_requests=max_requests,wall_cap_seconds=seconds,
        elapsed_seconds=time.monotonic()-start_clock,scope='Unchanged champion both seats on omniscient non-production placeholder observations; sampled legal policy actions. Unqualified mechanics, diagnostic only, no strength or training claim.',
        chance='Independent per-game ID streams, stable across batching; reversal is a second stochastic game, not a chance-aligned replay.',
        chunks=chunk_receipts,**summary)
    with (folder/'result.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('team-pool','weights','expected-pool-sha','out-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--team-start',type=int,default=140000);p.add_argument('--pairs',type=int,default=8192)
    p.add_argument('--batch',type=int,default=1024);p.add_argument('--max-requests',type=int,default=2000)
    p.add_argument('--seconds',type=float,default=1800);p.add_argument('--seed',type=int,default=20261023)
    a=p.parse_args();r=measure(a.team_pool,a.weights,a.expected_pool_sha,start=a.team_start,pairs=a.pairs,batch=a.batch,max_requests=a.max_requests,seconds=a.seconds,seed=a.seed,out_dir=a.out_dir)
    print(json.dumps({k:r[k] for k in ('status','games','terminal_games','capped_or_interrupted_games','unstarted_games')}),flush=True)
    raise SystemExit(0 if r['status']=='ROSTER_COMPLETE' else 1)
