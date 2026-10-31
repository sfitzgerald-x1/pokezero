"""Original-v4 retained173/bf16 raw selfplay -> original trajectory/PPO cache.

No new observations, targets, physics or admission waiver. Per-root steps spool
to remote run storage so the complete declared corpus is never retained in RAM.
"""
from dataclasses import asdict,replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import jax
import jax.numpy as j
import numpy as np

from . import benchmark_retained_games as original,champion,source_provenance
from pokezero.b4_fidelity_canary import require,file_sha,encoded,sha
from pokezero.observation import PokeZeroObservationV0,ObservationPerspective
from pokezero.trajectory import TrajectoryStep,BattleTrajectory,trajectory_to_dict,trajectory_from_dict
from pokezero.collection import RolloutRecord,write_rollout_record,rollout_record_to_dict,rollout_record_from_dict
from pokezero.env import TerminalState


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def append_durable(path,blob):
    path=Path(path)
    with path.open('ab') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_tree(path):
    """Finish original cache publication before its durable batch receipt."""
    path=Path(path)
    for item in sorted(path.rglob('*')):
        if item.is_file():
            with item.open('rb') as stream:os.fsync(stream.fileno())
    for item in sorted((p for p in path.rglob('*') if p.is_dir()),reverse=True):sync_directory(item)
    sync_directory(path);sync_directory(path.parent)


def publish_completed(path,*,game,pool_sha256,checkpoint_sha256,source_hash,prepared_binding,source_commit,ending,record):
    """Exclusive full original trajectory+terminal, durable BEFORE callback."""
    directory=Path(path)/'completed';directory.mkdir(exist_ok=True);sync_directory(directory.parent)
    body=dict(schema='pokezero.b4.canary.completed-cell.v1',game=asdict(game),pool_sha256=pool_sha256,
        checkpoint_sha256=checkpoint_sha256,source_hash=source_hash,prepared_binding=prepared_binding,source_commit=source_commit,
        ending=ending,rollout=rollout_record_to_dict(record))
    blob=encoded(body);target=directory/f'{game.stream_id}.json';temporary=directory/f'.{game.stream_id}.pending'
    with temporary.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    try:
        os.link(temporary,target)  # Exclusive: never replace an observed completion.
        sync_directory(directory)
    finally:
        temporary.unlink();sync_directory(directory)
    return hashlib.sha256(blob).hexdigest()


def stored_completed(path,*,batch,pool_sha256,checkpoint_sha256,source_hash,prepared_binding,source_commit):
    """Restore only literal durable cells, never a simulator state or RNG."""
    directory=Path(path)/'completed';games={g.stream_id:g for g in batch.games};found={}
    if not directory.exists():return found
    require(not directory.is_symlink(),'original owned completed directory required')
    for name in sorted(directory.glob('*.json')):
        require(name.is_file() and not name.is_symlink(),'original owned completed record required')
        require(name.stem.isdigit() and int(name.stem) in games,'foreign completed source cell')
        stream=int(name.stem);blob=name.read_bytes();body=json.loads(blob)
        require(set(body)=={'schema','game','pool_sha256','checkpoint_sha256','source_hash','prepared_binding','source_commit','ending','rollout'} and
            encoded(body)==blob and body['schema']=='pokezero.b4.canary.completed-cell.v1' and
            sha(body['game'])==sha(asdict(games[stream])) and body['pool_sha256']==pool_sha256 and
            body['checkpoint_sha256']==checkpoint_sha256 and body['source_hash']==source_hash and
            body['prepared_binding']==prepared_binding and body['source_commit']==source_commit,'original completed cell binding differs')
        ending=body['ending'];duration=ending['duration_seconds']
        require(ending['stream_id']==stream and ending['status']=='COMPLETE' and type(duration) in (int,float) and
            math.isfinite(duration) and duration>=0 and type(ending['frame']) is int and ending['frame']>0 and
            type(ending['executed_steps']) is int and ending['executed_steps']==ending['frame'] and
            type(ending['winner']) is int and ending['winner'] in (0,1,2),'literal completed terminal evidence differs')
        record=rollout_record_from_dict(body['rollout']);game=games[stream]
        require(record.battle_id==game.battle_id and record.seed==game.seed and record.format_id=='gen3randombattle' and
            not record.terminal.capped and record.terminal.winner==(None if ending['winner']==2 else f"p{ending['winner']+1}") and
            record.belief_set_source_hash==source_hash and record.elapsed_seconds==duration and
            record.policy_ids=={'p1':checkpoint_sha256,'p2':checkpoint_sha256} and
            record.trajectory.metadata['checkpoint_sha256']==checkpoint_sha256 and
            record.trajectory.steps and all(step.observation.schema_version=='pokezero.observation.v4' for step in record.trajectory.steps),
            'original terminal/trajectory receipt differs')
        # Retain only small descriptors; never all completed trajectories in RAM.
        found[stream]=(name,ending,hashlib.sha256(blob).hexdigest())
    return found


def cache_completed(path,*,batch,pool_sha256,checkpoint_sha256,source_hash,prepared_binding,source_commit,dataset_config,cache_name):
    from pokezero import dataset
    path=Path(path);found=stored_completed(path,batch=batch,pool_sha256=pool_sha256,
        checkpoint_sha256=checkpoint_sha256,source_hash=source_hash,prepared_binding=prepared_binding,source_commit=source_commit)
    report=dict(games=len(batch.games),complete=len(found),unknown=len(batch.games)-len(found),cache=None,
                completed_sources={str(stream):value[2] for stream,value in found.items()})
    if found:
        records=path/(cache_name+'-complete.jsonl')
        with records.open('x') as handle:
            for game in batch.games:
                if game.stream_id in found:
                    blob=found[game.stream_id][0].read_bytes();body=json.loads(blob)
                    require(hashlib.sha256(blob).hexdigest()==found[game.stream_id][2],'completed trajectory changed during cache packing')
                    write_rollout_record(handle,rollout_record_from_dict(body['rollout']))
            handle.flush();os.fsync(handle.fileno())
        sync_directory(path)
        cache=path/cache_name;dataset.write_training_cache_streaming(records,cache,config=dataset_config,collection_env='engine',belief_fidelity='full')
        sync_tree(cache)
        report.update(cache=str(cache),cache_metadata_sha256=file_sha(cache/'metadata.json'),complete_rollouts_sha256=file_sha(records))
    return report,found


def decide(record,result,eligible,seed):
    """Same original bf16 network/keys/router, retaining literal behavior data."""
    b=len(eligible);flat=np.flatnonzero(eligible.reshape(-1));core=result.state.core
    require(len(flat)>0,'no model call without eligible owner')
    args=[]
    for field in ('numeric_features','categorical_ids','token_type_ids','attention_mask'):
        values=j.stack([getattr(r.encoded.observation,field) for r in result.owners],axis=1)
        args.append(values.reshape((2*b,)+values.shape[2:])[flat])
    heads,_,_=original._network(record,tuple(args))
    require(heads.policy_logits.shape==(len(flat),9) and heads.value.shape==(len(flat),) and
            heads.opponent_action_logits.shape==(len(flat),9),'original champion three heads')
    finite=np.logical_and.reduce([np.isfinite(np.asarray(x)).reshape(len(flat),-1).all(axis=1) for x in heads])
    owners=flat%2;rows=flat//2
    keys=original._keys(seed,np.asarray(core.cursor.stream_id)[rows],np.asarray(core.frame_number)[rows],owners)
    legal=j.stack([r.encoded.observation.legal_action_mask for r in result.owners],axis=1).reshape(2*b,9)[flat]
    choices=np.full((b,2),-1,np.int32);choices[rows,owners]=np.where(finite,np.asarray(original.sample_actions(heads.policy_logits,legal,keys)),-1)
    probabilities=np.zeros((b,2),np.float32);values=np.zeros((b,2),np.float32)
    p=np.asarray(jax.nn.softmax(j.where(legal,heads.policy_logits,-j.inf),axis=-1))
    for k,(row,owner) in enumerate(zip(rows,owners)):
        if finite[k]:probabilities[row,owner]=p[k,choices[row,owner]];values[row,owner]=np.asarray(heads.value)[k]
    refused=np.zeros(b,bool);refused[rows[~finite]]=True
    refused|=np.any(eligible&((probabilities<=0)|~np.isfinite(probabilities)),axis=1)
    routed=[]
    for side in (0,1):
        route=jax.block_until_ready(original.engine_private_actions.route_choice(core.requests[side].actions,core.orders[side],j.asarray(choices[:,side])))
        refused|=eligible[:,side]&~np.asarray(route.admitted)
        routed.append(j.where(route.admitted,route.engine_action,0))
    return j.stack(routed,axis=1),refused,choices,probabilities,values


def captured_step(result,*,row,owner,choice,opponent_choice,probability,value,archive_sha):
    obs=result.owners[owner].encoded.observation;core=result.state.core
    require(bool(np.asarray(result.owners[owner].policy_admitted)[row]),'actual original owner admission')
    arrays=[np.asarray(getattr(obs,key))[row].tolist() for key in
            ('categorical_ids','numeric_features','token_type_ids','attention_mask','legal_action_mask')]
    def freeze(x):return tuple(freeze(v) for v in x) if isinstance(x,list) else x
    observation=PokeZeroObservationV0(*(freeze(x) for x in arrays),
        perspective=ObservationPerspective.from_showdown_slot(f'p{owner+1}',f'p{owner+1}'),schema_version='pokezero.observation.v4')
    return TrajectoryStep(f'p{owner+1}',int(np.asarray(core.cursor.turn)[row]),observation,tuple(observation.legal_action_mask),
        int(choice),opponent_action_index=None if opponent_choice<0 else int(opponent_choice),
        action_probability=float(probability),value_estimate=float(value),
        metadata=dict(canary_source_frame=int(np.asarray(core.frame_number)[row]),canary_stream=int(np.asarray(core.cursor.stream_id)[row]),
                      original_weights_archive_sha256=archive_sha))


def complete_record(game,steps,*,winner,turn,duration,source_hash,checkpoint_sha):
    """Only called after actual physical execution and both public consumers."""
    require(type(winner) is int and winner in (0,1,2),'original terminal winner0/1/tie2')
    terminal=TerminalState(None if winner==2 else f'p{winner+1}',turn,False)
    steps=list(steps)
    for owner in ('p1','p2'):
        indices=[i for i,s in enumerate(steps) if s.player_id==owner]
        if indices:
            reward=0. if winner==2 else 1. if terminal.winner==owner else -1.
            steps[indices[-1]]=replace(steps[indices[-1]],reward=reward)
    trajectory=BattleTrajectory(game.battle_id,'gen3randombattle',game.seed,steps,terminal,
                                dict(source='original-retained173-bf16-canary',checkpoint_sha256=checkpoint_sha))
    return RolloutRecord(game.battle_id,game.seed,'gen3randombattle',{'p1':checkpoint_sha,'p2':checkpoint_sha},
                         max((s.turn_index for s in steps),default=0),duration,terminal,trajectory,source_hash)


class Backend:
    def __init__(self,prepared,pool,*,gender_policy,plan,model_config):
        self.prepared=prepared;self.pool=pool;self.gender_policy=gender_policy;self.plan=plan;self.model_config=model_config
        record=original.driver._checked(prepared);original.driver._checked_witness(record)
        require(record.get('frame_input_mode')=='source-witness','actual original both-FULL source path required')
        require(source_provenance._pool(pool,rehash=True)['pool_sha256']==plan.pool_sha256,'declared pool differs')
        self.source_hash=record['catalog'].source.source_hash
        from .benchmark_retained_games import _source_snapshot
        require(_source_snapshot()['commit']==plan.source_commit,'declared actual source freeze differs')
        require(prepared.binding==plan.prepared_binding,'declared original Prepared binding differs')

    def recover(self,batch,checkpoint,path,*,dataset_config,on_end):
        """Data-only recovery after an attempted batch; never call the engine."""
        import uuid
        path=Path(path)
        if not path.exists():
            return dict(status='INTERRUPTED_UNKNOWN',games=len(batch.games),complete=0,unknown=len(batch.games),cache=None,completed_sources={})
        report,found=cache_completed(path,batch=batch,pool_sha256=self.plan.pool_sha256,
            checkpoint_sha256=file_sha(checkpoint),source_hash=self.source_hash,prepared_binding=self.plan.prepared_binding,
            source_commit=self.plan.source_commit,dataset_config=dataset_config,
            cache_name='recovered-cache-'+uuid.uuid4().hex)
        for game in batch.games:
            if game.stream_id in found:
                _,ending,digest=found[game.stream_id]
                on_end(game.stream_id,'COMPLETE',ending['duration_seconds'],digest)
        report.update(status='RECOVERED_STORED_ATTEMPT',preserved_completed_streams=[g.stream_id for g in batch.games if g.stream_id in found],
            note='Only original durable complete trajectories recovered; every absent attempted cell remains UNKNOWN, no replay/refill')
        if (path/'endings.jsonl').exists():report.update(endings=str(path/'endings.jsonl'),endings_sha256=file_sha(path/'endings.jsonl'))
        return report

    def collect(self,batch,checkpoint,path,*,dataset_config,on_end,check_progress):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        sync_directory(path);sync_directory(path.parent)
        archive=path/'actor.npz';metadata=champion.export(checkpoint,archive)
        params,config,weights=champion.load(archive)
        require(config==champion.config_from_checkpoint(self.model_config.to_dict()),'original v4 shape changed')
        require(metadata['checkpoint_sha256']==file_sha(checkpoint),'export checkpoint binding')
        runtime=original._bind(self.prepared,params,config,weights);record=original._checked(runtime)
        indices=np.asarray([g.team_indices for g in batch.games],np.int32);streams=j.asarray([g.stream_id for g in batch.games],j.int32)
        start=time.monotonic();b=len(batch.games);status=['RUNNING']*b;steps=np.zeros(b,np.int64)
        session=original.driver.construct(self.pool,indices,prepared=self.prepared,stream_ids=streams,seed=batch.constructor_seed,
            gender_initialization=dict(policy=self.gender_policy,seed=batch.gender_seed,domain=batch.gender_domain,root_ids=batch.gender_root_ids))
        jax.block_until_ready(session.result)
        records=path/'complete.jsonl';complete=0
        def close(row,disposition,reason):
            nonlocal complete
            if status[row]!='RUNNING':return
            status[row]=disposition;duration=time.monotonic()-start
            ending=dict(stream_id=batch.games[row].stream_id,status=disposition,reason=reason,duration_seconds=duration,executed_steps=int(steps[row]),
                        frame=int(np.asarray(session.state.core.frame_number)[row]))
            digest=None
            if disposition=='COMPLETE':
                spool=path/f'row-{row}.jsonl';saved=[]
                if spool.exists():
                    with spool.open('rb') as handle:os.fsync(handle.fileno())
                    sync_directory(path)
                    for line in spool.read_text().splitlines():saved.extend(trajectory_from_dict(json.loads(line)).steps)
                core=session.state.core
                ending['winner']=int(np.asarray(core.engine.winner)[row])
                item=complete_record(batch.games[row],saved,winner=int(np.asarray(core.engine.winner)[row]),
                    turn=int(np.asarray(core.cursor.turn)[row]),duration=duration,source_hash=self.source_hash,checkpoint_sha=file_sha(checkpoint))
                digest=publish_completed(path,game=batch.games[row],pool_sha256=self.plan.pool_sha256,
                    checkpoint_sha256=file_sha(checkpoint),source_hash=self.source_hash,prepared_binding=self.plan.prepared_binding,
                    source_commit=self.plan.source_commit,ending=ending,record=item)
                append_durable(records,encoded(rollout_record_to_dict(item))+b'\n')
                complete+=1
            append_durable(path/'endings.jsonl',encoded(dict(ending,completed_source_sha256=digest))+b'\n')
            on_end(batch.games[row].stream_id,disposition,duration,digest if digest is not None else sha(ending))
        try:
            for _ in range(self.plan.max_requests):
                check_progress();original._checked(runtime,session=session)
                active=np.asarray([s=='RUNNING' for s in status]);initialdone=active&np.asarray(session.state.core.engine.done)
                for row in np.flatnonzero(initialdone):close(int(row),'REFUSED','terminal without observed transition')
                active=np.asarray([s=='RUNNING' for s in status])
                eligible,failed,_=original._admission(session.result,active,config)
                for row in np.flatnonzero(failed):close(int(row),'REFUSED','original retained/request admission')
                active&=~failed
                if not active.any():break
                actions,failed,choices,probabilities,values=decide(record,session.result,eligible,batch.sampling_seed)
                for row in np.flatnonzero(failed):close(int(row),'REFUSED','nonfinite head/unroutable choice')
                active&=~failed
                if not active.any():break
                for row in np.flatnonzero(active):
                    for owner in (0,1):
                        if not eligible[row,owner]:continue
                        step=captured_step(session.result,row=int(row),owner=owner,choice=choices[row,owner],
                            opponent_choice=choices[row,1-owner] if eligible[row,1-owner] else -1,
                            probability=probabilities[row,owner],value=values[row,owner],archive_sha=weights['weights_archive_sha256'])
                        one=BattleTrajectory(batch.games[row].battle_id,'gen3randombattle',batch.games[row].seed,[step])
                        with (path/f'row-{row}.jsonl').open('ab') as handle:handle.write(encoded(trajectory_to_dict(one))+b'\n')
                before=np.asarray(session.state.core.frame_number).copy()
                result=original.driver.advance(session,actions,active_rows=j.asarray(active));jax.block_until_ready(result)
                diagnostics=jax.block_until_ready(original.driver.latest_diagnostics(session))
                executed=np.asarray(diagnostics.executed_rows)
                require(np.array_equal(np.asarray(diagnostics.active_rows),active) and not np.any(executed&~active) and
                        np.array_equal(np.asarray(result.state.core.frame_number)-before,executed.astype(np.int32)), 'original exact physical execution set')
                steps+=executed
                public=np.logical_and.reduce([(np.asarray(r.basic_error)==0)&(np.asarray(r.retained_error)==0)&np.asarray(o.retained_healthy)
                        for r,o in zip(result.owners,result.state.observers)])
                bad=active&(~executed|np.asarray(diagnostics.invalid_actions)|~np.asarray(result.state.core.admitted)|~public)
                for row in np.flatnonzero(bad):close(int(row),'REFUSED','original engine/public source refusal')
                for row in np.flatnonzero(active&~bad&np.asarray(result.state.core.engine.done)):close(int(row),'COMPLETE',None)
        except BaseException:
            for row in range(b):close(row,'FAILED','collection interrupted/failed; no automatic retry')
            raise
        for row in range(b):close(row,'CAPPED','declared finite request cap; no outcome')
        report,_=cache_completed(path,batch=batch,pool_sha256=self.plan.pool_sha256,checkpoint_sha256=file_sha(checkpoint),
            source_hash=self.source_hash,prepared_binding=self.plan.prepared_binding,source_commit=self.plan.source_commit,
            dataset_config=dataset_config,cache_name='cache')
        report.update(status='COLLECTED',endings=str(path/'endings.jsonl'),endings_sha256=file_sha(path/'endings.jsonl'))
        return report
