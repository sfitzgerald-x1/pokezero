"""Bounded real-driver selfplay with explicit immutable per-seat opponents.

Only this NEW opt-in path constructs declared sessions. It reuses the original
collector's guards, router, public adapter and typed data producer. Historical
opponent rows are not relabeled as learner-policy behavior or trained as PPO.
No legacy/default runner is changed. D remains unavailable until an actual
current public-world materializer and teacher provider are integrated.
"""
from dataclasses import asdict, dataclass
import json
import time
import uuid

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract as c, v5_data
from pokezero.v5_run import MissingDependency, _file_sha
from . import v5_collection as base, v5_collection_data as data
from . import v5_owning_history as history_capture
from . import v5_public_world as public_world


@dataclass(frozen=True)
class TeacherContext:
    address:object
    source_frame_sha256:str
    public_observation:object
    own_packet:object
    retained_public_state:object
    raw_policy_logits:object
    raw_public_value:float
    actor_archive_sha256:str
    opponent_archive_sha256:str
    owning_packets:tuple
    # Encoded/request packets are NOT complete literal public protocol history.
    public_events_complete:bool=False


@dataclass(frozen=True)
class PairedCollection:
    decisions:tuple
    terminals:tuple
    ledger:dict
    owning_histories:tuple
    history_refusals:tuple
    own_statics:tuple=()
    own_static_refusals:tuple=()


def _archive(actor):return json.loads(actor.metadata)['weights_archive_sha256']


def _pair_inference(records,actors,inputs,legal,eligible,streams,frames,seed):
    b=len(streams);choices=np.full((b,2),-1,np.int32);prob=np.zeros((b,2),np.float32)
    values=np.zeros((b,2),np.float32);logits=np.full((b,2,9),np.nan,np.float32);failed=np.zeros(b,bool)
    unique=[]
    for pair in actors:
        for actor in pair:
            if all(actor is not previous for previous in unique):unique.append(actor)
    for actor in unique:
        mask=np.asarray([[selected is actor for selected in pair] for pair in actors])&eligible
        flat=np.flatnonzero(mask.reshape(-1))
        if not len(flat):continue
        # Power-of-two padding keeps a bounded executable set as roots finish.
        count=1<<(len(flat)-1).bit_length();selected={}
        for name,value in inputs.items():
            a=value[j.asarray(flat)];padding=j.zeros((count-len(flat),*a.shape[1:]),a.dtype)
            selected[name]=j.concatenate((a,padding),axis=0)
        output=jax.block_until_ready(records[actor]['forward'](selected))
        ls=np.asarray(output.policy_logits)[:len(flat)];vs=np.asarray(output.value)[:len(flat)]
        valid=np.asarray(output.valid)[:len(flat)]&np.isfinite(ls).all(-1)&np.isfinite(vs)&(np.abs(vs)<=1)
        for index,position in enumerate(flat):
            row,owner=divmod(int(position),2)
            if not valid[index] or not legal[row,owner].any():failed[row]=True;continue
            score=j.where(j.asarray(legal[row,owner]),j.asarray(ls[index]),-j.inf)
            key=jax.random.fold_in(jax.random.PRNGKey(seed),int(streams[row]))
            key=jax.random.fold_in(jax.random.fold_in(key,int(frames[row])),owner)
            choice=int(jax.random.categorical(key,score));p=float(jax.nn.softmax(score)[choice])
            if not np.isfinite(p) or not 0<p<=1:failed[row]=True;continue
            choices[row,owner]=choice;prob[row,owner]=p;values[row,owner]=vs[index];logits[row,owner]=ls[index]
    return choices,prob,values,logits,failed


def collect_paired(session,*,learner,actors,pool,team_indices,ledger,max_requests,sampling_seed,
        on_end,check_progress,started_at=None):
    """Collect one fixed original Session. No constructor/retry exists here."""
    if (type(max_requests) is not int or max_requests<=0 or type(sampling_seed) is not int
            or not 0<=sampling_seed<2**32 or not callable(on_end) or not callable(check_progress)):
        raise ValueError('finite declared paired collection recipe/callbacks required')
    if learner.config.arm=='D':raise MissingDependency('D current-root public-world materializer/teacher is not integrated')
    record,r,current=base._checked(learner,session)
    actors=tuple(tuple(pair) for pair in actors);streams=np.asarray(session.state.core.cursor.stream_id);b=len(streams)
    if len(actors)!=b or any(len(pair)!=2 for pair in actors):raise ValueError('two declared original actors per game required')
    records={}
    for pair in actors:
        if learner not in pair:raise ValueError('actual learner must own at least one seat')
        for actor in pair:
            actual,other,_=base._checked(actor,session)
            if actor.config!=learner.config or other['catalog'] is not r['catalog']:
                raise ValueError('same arm/replicate/source/category model namespace required')
            records[actor]=actual
    book=base._ledger(ledger);status=book['status'];detail=book['detail']
    by_stream={cell.stream_id:i for i,cell in enumerate(ledger.cells)}
    if len(set(map(int,streams)))!=b or any(int(x) not in by_stream for x in streams):raise ValueError('declared actual streams required')
    positions=[by_stream[int(x)] for x in streams]
    if positions!=[i for i,s in enumerate(status) if s=='NOT_ATTEMPTED'][:b]:
        raise ValueError('next original not-attempted prefix required')
    indices=np.asarray(team_indices)
    if indices.shape!=(b,2) or indices.dtype.kind not in 'iu':raise ValueError('declared integer original team pairs required')
    for row,pos in enumerate(positions):
        if tuple(map(int,indices[row]))!=ledger.cells[pos].team_indices:raise ValueError('declared team pair changed')
        status[pos]='RUNNING';detail[pos]={'decisions':0,'executed_frames':0}
    started=time.monotonic() if started_at is None else started_at
    decisions=[];terminals={};receipt={'pool_sha256':None};histories={};history_refusals={}
    own_statics={};own_static_refusals={}

    def retain_history(rows):
        for row in rows:
            for owner in (0,1):
                key=(int(row),owner)
                if (int(np.asarray(session.state.core.frame_number)[row])==0
                        and key not in own_statics and key not in own_static_refusals):
                    try:own_statics[key]=public_world.capture_own_static(session,row=int(row),owner=owner)
                    except (ValueError,KeyError,IndexError) as error:
                        # Missing private own happiness is a teacher refusal,
                        # independent of the retained public-history capture.
                        own_static_refusals[key]=type(error).__name__+': '+str(error)
                if key in history_refusals:continue
                try:histories[key]=history_capture.capture(session,row=int(row),owner=owner,history=histories.get(key))
                except (ValueError,KeyError,IndexError) as error:
                    # D history is an auxiliary source artifact. An unsupported
                    # capture never changes the A/B actor or game denominator.
                    history_refusals[key]=type(error).__name__+': '+str(error)

    def close(row,disposition,reason):
        pos=positions[row]
        if status[pos]!='RUNNING':return
        core=session.state.core;frame=int(np.asarray(core.frame_number)[row]);duration=time.monotonic()-started
        body=dict(stream=int(streams[row]),frame=frame,status=disposition,reason=reason,pool_sha256=receipt['pool_sha256'])
        if disposition=='COMPLETE':body.update(winner=int(np.asarray(core.engine.winner)[row]),
            hp=np.asarray(core.engine.hp)[row].tolist(),max_hp=np.asarray(core.engine.max_hp)[row].tolist())
        terminal=data.TerminalCapture(int(streams[row]),frame,disposition,data.sha(body),body.get('winner'),
            base._as_tuple(body.get('hp',())),base._as_tuple(body.get('max_hp',())))
        terminals[int(streams[row])]=terminal;status[pos]=disposition
        detail[pos].update(reason=reason,duration_seconds=duration,source_sha256=terminal.source_sha256)
        on_end(int(streams[row]),disposition,duration_seconds=duration,source_sha256=terminal.source_sha256,
            frame=frame,reason=reason)

    try:
        truth,receipt=base._pool_truth(pool,indices,session,r,current)
        retain_history(range(b))
        for _ in range(max_requests):
            check_progress()
            for actor in records:base._checked(actor,session)
            core=session.state.core;result=session.result
            active=np.asarray([status[p]=='RUNNING' for p in positions])
            for row in np.flatnonzero(active&np.asarray(core.engine.done)):close(int(row),'REFUSED','terminal-without-observed-transition')
            active=np.asarray([status[p]=='RUNNING' for p in positions])
            if not active.any():break
            appended,inputs,eligible=base._public_inputs(record,result,active)
            ready=np.stack([np.asarray(base.private_requests.action_inputs(request.actions.facts.request,
                observer_side=result.state.observers[owner].owner).ready) for owner,request in enumerate(core.requests)],axis=1)
            for row in np.flatnonzero(active&np.any(ready&~eligible,axis=1)):close(int(row),'REFUSED','current-owner-input-refused')
            active=np.asarray([status[p]=='RUNNING' for p in positions]);eligible&=active[:,None]
            legal=np.stack([np.asarray(report.encoded.observation.legal_action_mask) for report in result.owners],axis=1)
            choices,probabilities,values,raw_logits,failed=_pair_inference(records,actors,inputs,legal,eligible,streams,
                np.asarray(core.frame_number),sampling_seed)
            for row in np.flatnonzero(active&failed):close(int(row),'REFUSED','actor-nonfinite-or-no-legal-choice')
            routes=[]
            for owner in (0,1):
                route=base.actions.route_choice(core.requests[owner].actions,core.orders[owner],j.asarray(choices[:,owner]))
                jax.block_until_ready(route);routes.append(route)
                for row in np.flatnonzero(eligible[:,owner]&~np.asarray(route.admitted)):close(int(row),'REFUSED','actual-request-route-refused')
            active=np.asarray([status[p]=='RUNNING' for p in positions])
            if not active.any():break
            pending=[]
            for row in np.flatnonzero(active):
                row_pending=[]
                try:
                    for owner in (0,1):
                        if not eligible[row,owner] or actors[row][owner] is not learner:continue
                        address=c.DecisionAddress(int(streams[row]),int(np.asarray(core.frame_number)[row]),owner)
                        old,public=data.public_observation(result.owners[owner].encoded,appended[owner],row=int(row),owner=owner,
                            address=address,adapter=learner.adapter,
                            public_species=np.asarray(result.state.observers[owner].retained.beliefs.species)[row,owner])
                        physical=base._physical(core,int(row),owner);cell=ledger.cells[positions[row]]
                        hidden=tuple(None if p<0 else truth[row][1-owner][p] for p in physical[6:])
                        semantic=base._semantic(core,routes,choices,int(row),1-owner,r)
                        if semantic is not None:
                            try:semantic.index(learner.config.label_vocabulary)
                            except ValueError:semantic=None
                        peer_capture=None;privileged=None
                        if bool(np.asarray(appended[1-owner].admitted)[row]):
                            peer_old,_=data.public_observation(result.owners[1-owner].encoded,appended[1-owner],row=int(row),owner=1-owner,
                                address=c.DecisionAddress(address.stream,address.frame,1-owner),adapter=learner.adapter,
                                public_species=np.asarray(result.state.observers[1-owner].retained.beliefs.species)[row,1-owner])
                            peer_physical=base._physical(core,int(row),1-owner)[:6]
                            peer_capture=dict(address=dict(stream=address.stream,frame=address.frame,owner=1-owner),
                                category_vocabulary_sha256=learner.config.category_vocabulary_sha256,physical=peer_physical,
                                categorical_ids=peer_old.categorical_ids[1:7],numeric_features=peer_old.numeric_features[1:7],
                                token_type_ids=peer_old.token_type_ids[1:7],attention_mask=peer_old.attention_mask[1:7])
                            if len(set(physical[6:]))==len(set(peer_physical))==6 and -1 not in physical[6:] and -1 not in peer_physical:
                                privileged=c.privileged_view(public,peer_old,peer_address=c.DecisionAddress(address.stream,address.frame,1-owner),
                                    opponent_slot_map=tuple(peer_physical.index(p) for p in physical[6:]),
                                    peer_category_vocabulary_sha256=learner.config.category_vocabulary_sha256)
                        body=dict(stream=address.stream,frame=address.frame,owner=owner,turn=int(np.asarray(core.engine.turn)[row]),
                            pool_sha256=receipt['pool_sha256'],team_indices=cell.team_indices,physical=physical,
                            public_inputs=public.model_inputs(),request_binding=np.asarray(core.requests[owner].actions.facts.catalog_binding)[row].tolist(),
                            engine_action=int(np.asarray(routes[owner].engine_action)[row]),choice=int(choices[row,owner]),
                            behavior_probability=float(probabilities[row,owner]),value_estimate=float(values[row,owner]),
                            actor_archive_sha256=_archive(actors[row][owner]),opponent_archive_sha256=_archive(actors[row][1-owner]),
                            training_only_peer_own_capture=peer_capture)
                        row_pending.append((int(row),data.capture_decision(old,public,battle_id=cell.battle_id,seed=cell.seed,
                            turn=body['turn'],choice=int(choices[row,owner]),probability=float(probabilities[row,owner]),
                            value=float(values[row,owner]),physical=physical,hidden_sets=hidden,source_frame_sha256=data.sha(body),
                            capture_metadata={k:v for k,v in body.items() if k!='public_inputs'},opponent_action=semantic,privileged=privileged)))
                except ValueError as error:close(int(row),'REFUSED','source-row-capture: '+str(error))
                else:pending.extend(row_pending)
            active=np.asarray([status[p]=='RUNNING' for p in positions])
            if not active.any():break
            transport=j.stack([j.where(route.admitted,route.engine_action,0) for route in routes],axis=1)
            before=np.asarray(core.frame_number).copy()
            base.driver.advance(session,transport,active_rows=j.asarray(active));jax.block_until_ready(session.result)
            diagnostics=base.driver.latest_diagnostics(session);jax.block_until_ready(diagnostics)
            executed=np.asarray(diagnostics.executed_rows);invalid=np.asarray(diagnostics.invalid_actions)
            if (executed.shape!=(b,) or executed.dtype!=np.bool_ or np.any(executed&~active)
                    or not np.array_equal(np.asarray(diagnostics.active_rows),active)
                    or not np.array_equal(np.asarray(session.state.core.frame_number)-before,executed.astype(np.int32))):
                raise ValueError('actual driver executed a different frame/row set')
            # Nonexecuted requests have no new original frame. Executed refused
            # frames still retain the literal pre-rollback FULL report.
            retain_history(np.flatnonzero(executed))
            for row,captured in pending:
                if executed[row] and not invalid[row]:decisions.append(captured);detail[positions[row]]['decisions']+=1
            public_ok=np.logical_and.reduce([(np.asarray(report.basic_error)==0)&(np.asarray(report.retained_error)==0)
                &np.asarray(observer.retained_healthy) for report,observer in zip(session.result.owners,session.state.observers)])
            for row in np.flatnonzero(active):
                detail[positions[row]]['executed_frames']+=int(executed[row])
                if not executed[row] or invalid[row] or not bool(np.asarray(session.state.core.admitted)[row]) or not public_ok[row]:
                    close(int(row),'REFUSED','actual-engine-or-public-frame-refused')
                elif bool(np.asarray(session.state.core.engine.done)[row]):close(int(row),'COMPLETE','observed-admitted-terminal')
            check_progress()
        for row,pos in enumerate(positions):
            if status[pos]=='RUNNING':close(row,'CAPPED','finite-request-cap')
    except Exception as error:
        for row,pos in enumerate(positions):
            if status[pos]=='RUNNING':close(row,'FAILED',type(error).__name__+': '+str(error))
        if isinstance(error,TimeoutError):raise
    closed=tuple(terminals[int(stream)] for stream in streams)
    return PairedCollection(data.finish(decisions,closed),closed,ledger.report(),
        tuple((int(streams[row]),owner,value) for (row,owner),value in sorted(histories.items())),
        tuple((int(streams[row]),owner,reason) for (row,owner),reason in sorted(history_refusals.items())),
        tuple((int(streams[row]),owner,value) for (row,owner),value in sorted(own_statics.items())),
        tuple((int(streams[row]),owner,reason) for (row,owner),reason in sorted(own_static_refusals.items())))


class EngineBackend:
    """Real original constructor/collector/cache calls; creation executes none."""
    def __init__(self,prepared,pool,*,gender_policy=None,compute_dtype='bfloat16'):
        self.prepared=prepared;self.pool=pool;self.gender_policy=gender_policy;self.compute_dtype=compute_dtype;self.actors={}

    def require_teacher(self):
        raise MissingDependency('D current-root public-world materializer and ready provider are not integrated')

    def check_plan(self,plan):
        r=base.driver._checked(self.prepared);base.driver._checked_witness(r)
        if base.source_provenance._pool(self.pool,rehash=True)['pool_sha256']!=plan.pool_sha256:
            raise ValueError('original declared constructor pool differs')
        base.v5_observation.compile_adapter(r['catalog'],model_config=plan.model)

    def _actor(self,snapshot,store):
        digest=snapshot['archive_sha256']
        if digest not in self.actors:
            path=store.resolve(snapshot['archive'],digest)
            self.actors[digest]=base.load_actor(self.prepared,path,expected_archive_sha256=digest,
                expected_checkpoint_sha256=snapshot['checkpoint_sha256'],
                expected_category_vocabulary_sha256=store.plan.model.category_vocabulary_sha256,
                expected_label_vocabulary_sha256=store.plan.model.label_vocabulary.sha256,
                compute_dtype=self.compute_dtype,enable_v5=True)
        return self.actors[digest]

    def prepare_batch(self,plan,batch,current,assignments,store):
        self.check_plan(plan)
        if [a['stream_id'] for a in assignments]!=[g.stream_id for g in batch.games]:raise ValueError('declared assignment order differs')
        required={current['archive_sha256'],*(a['opponent']['archive_sha256'] for a in assignments)}
        self.actors={key:value for key,value in self.actors.items() if key in required}
        learner=self._actor(current,store);pairs=[]
        for assignment in assignments:
            if type(assignment['learner_side']) is not int or assignment['learner_side'] not in (0,1):raise ValueError('declared owning learner seat required')
            opponent=self._actor(assignment['opponent'],store)
            pairs.append((learner,opponent) if assignment['learner_side']==0 else (opponent,learner))
        cells=tuple(base.Cell(g.stream_id,g.battle_id,g.seed,g.team_indices) for g in batch.games)
        ledger=base.declare_cells(cells);indices=np.asarray([g.team_indices for g in batch.games],np.int32)
        streams=j.asarray([g.stream_id for g in batch.games],j.int32)
        gender=None
        if batch.gender_seed is not None:
            if self.gender_policy is None:raise ValueError('original declared gender policy required')
            gender=dict(policy=self.gender_policy,seed=batch.gender_seed,domain=batch.gender_domain,root_ids=batch.gender_root_ids)
        return dict(plan=plan,batch=batch,learner=learner,pairs=tuple(pairs),ledger=ledger,indices=indices,streams=streams,gender=gender)

    def collect(self,prepared,store,on_end,check_progress):
        check_progress()
        plan=prepared['plan'];batch=prepared['batch'];learner=prepared['learner'];pairs=prepared['pairs']
        ledger=prepared['ledger'];indices=prepared['indices'];streams=prepared['streams'];gender=prepared['gender']
        started=time.monotonic()
        try:session=base.driver.construct(self.pool,indices,prepared=self.prepared,stream_ids=streams,
            seed=batch.constructor_seed,gender_initialization=gender)
        except Exception as error:
            for game in batch.games:on_end(game.stream_id,'FAILED',reason=type(error).__name__+': '+str(error),duration_seconds=time.monotonic()-started)
            raise
        collection=collect_paired(session,learner=learner,actors=pairs,pool=self.pool,team_indices=indices,
            ledger=ledger,max_requests=plan.max_requests,sampling_seed=batch.sampling_seed,
            on_end=on_end,check_progress=check_progress,started_at=started)
        # Per-owner artifacts have no live Session/Core/catalog handle. Keep
        # complete canonical records and unknowns even when no training row
        # exists. A future teacher must use the original decoder, never relabel
        # this archive as literal simulator text or a ready device root.
        for stream,owner,history in collection.owning_histories:
            own_static=next((value for actual,seat,value in collection.own_statics if (actual,seat)==(stream,owner)),None)
            payload=dict(schema='pokezero.owning_canonical_history.v5',stream_id=stream,owner=owner,
                source_hash=history.catalog.source.source_hash,
                category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
                own_constructor=history.own_constructor,frames=[asdict(frame) for frame in history.frames],
                own_static=None if own_static is None else asdict(own_static),
                own_static_refusals=[reason for actual,seat,reason in collection.own_static_refusals if (actual,seat)==(stream,owner)],
                capture_refusals=[reason for actual,seat,reason in collection.history_refusals if (actual,seat)==(stream,owner)],
                device_branch_ready=False)
            store.artifact(f'histories/{stream}-p{owner+1}-{uuid.uuid4().hex}.json',
                json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),
                kind='OWNING_HISTORY',stream_id=stream,owner=owner)
        for stream,owner,reason in collection.history_refusals:
            store.emit('OWNING_HISTORY_REFUSED',stream_id=stream,owner=owner,reason=reason)
        for stream,owner,reason in collection.own_static_refusals:
            store.emit('OWN_STATIC_REFUSED',stream_id=stream,owner=owner,reason=reason)
        if not collection.decisions:return None
        relative=f'caches/cache-{uuid.uuid4().hex}'
        data.write_cache(store.path/relative,collection.decisions,loss_config=plan.loss,batch_size=plan.train.batch_size)
        return relative,_file_sha(store.path/relative/'metadata.json')
