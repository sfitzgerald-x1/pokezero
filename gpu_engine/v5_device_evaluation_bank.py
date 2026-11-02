"""One prospectively declared device heldout study; never a training collector.

No model, constructor or game runs on import. Ordinary games have no refill.
Counterfactual one-step siblings are a separate, explicitly declared branch
study: every legal action shares the same actual root, reply and chance capsule.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import uuid
import jax
import jax.numpy as j
import numpy as np
from pokezero import v5_device_evaluation_bank as schema, v5_run as run
from . import transition, engine_device_observation as encoder, engine_device_trackers as trackers
from . import v5_model, benchmark_device_selfplay as sampling, device_root_snapshot
from .v5_device_rollout import RootContext


def load_root(entry, index):
    if index['schema'] != schema.ROOT_INDEX or entry not in index['cost_roots']:
        raise ValueError('exact declared saved-root index entry required')
    return device_root_snapshot.load(entry, index)


def _host(value):
    if hasattr(value, '_fields'): return {k:_host(v) for k,v in value._asdict().items()}
    if type(value) is dict: return {k:_host(v) for k,v in value.items()}
    if type(value) in (tuple,list): return [_host(v) for v in value]
    if value is None or type(value) in (str,int,float,bool): return value
    return np.asarray(jax.block_until_ready(value)).tolist()


def _one(value, row): return jax.tree.map(lambda a:a[row:row+1], value)


def _one_view(value, row):
    # error_counts is the actual encoder's whole-call summary, not a batch
    # axis. Preserve it literally; every model/routing tensor is row-indexed.
    return value._replace(**{name:None if a is None else a if name=='error_counts' else a[row:row+1]
        for name,a in value._asdict().items()})


def _observation(value):
    return dict(schema=schema.OBSERVATION, **{name:_host(getattr(value,name)[0]) for name in
        ('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask')})


def _publish(path, value, *, mutable=False):
    path = Path(path); blob = (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if not mutable:
        run._publish(path,blob)
    else:
        temporary = path.with_name('.pending-'+uuid.uuid4().hex)
        with temporary.open('xb') as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)


@dataclass(frozen=True)
class Executor:
    query: object
    advance: object
    child_view: object


def make_executor(prepared, config, *, sampling_seed):
    """Actual original BF16 public actors, engine mapping and explicit chance."""
    tables = prepared.engine_tables; observations = prepared.observation_tables
    def views(state, public):
        mask = transition.action_mask(state,tables) & sampling.request_seats(state)[...,None]
        return tuple(encoder.encode(state,trackers.readout(public,j.full(len(state.done),side,j.int32)),observations,
            observer_sides=j.full(len(state.done),side,j.int32),legal_action_mask=mask[:,side]) for side in (0,1))
    def query(state, public, streams, frame, params):
        obs = views(state,public)
        out = [v5_model.forward(params[s],encoder.actor_inputs(obs[s]),config,compute_dtype='bfloat16') for s in (0,1)]
        logits = j.stack([o.policy_logits for o in out],axis=1)
        finite = j.stack([o.valid & j.all(j.isfinite(o.policy_logits),axis=-1) & j.isfinite(o.value)
            & (j.abs(o.value)<=1) & (j.all(j.isfinite(o.value_logits),axis=-1) if o.value_logits is not None else True) for o in out],axis=1)
        legal = j.stack([o.legal_action_mask for o in obs],axis=1)
        safe = j.where(legal,j.nan_to_num(logits,nan=0.,posinf=0.,neginf=0.),-j.inf)
        safe = j.where(j.any(legal,axis=-1)[...,None],safe,0.)
        keys = sampling.sample_keys(streams,frame,sampling_seed=sampling_seed)
        actions = jax.vmap(jax.random.categorical)(keys.reshape(-1,2),safe.reshape(-1,9)).reshape(-1,2).astype(j.int32)
        physical = j.stack([encoder.map_actions(obs[s],state,j.full(len(state.done),s,j.int32),actions[:,s]) for s in (0,1)],axis=1)
        requested = sampling.request_seats(state)
        valid = finite & (j.stack([o.error for o in obs],axis=1)==0) & j.any(legal,axis=-1)
        return dict(observations=obs,policy_logits=logits,policy_finite=finite,requested=requested,
            actions=actions,physical=physical,input_valid=j.all(~requested|valid,axis=1),
            chance=transition.sample_chance_inputs(state))
    def advance(state,public,actions,chance):
        return transition.step_with_public(state,actions,tables,public,
            belief_tables=observations.belief,chance_inputs=chance)
    return Executor(jax.jit(query),jax.jit(advance),jax.jit(views))


def _siblings(executor, state, public, query, row, selector):
    s = _one(state,row); p = _one(public,row); chance = _one(query['chance'],row)
    owner = selector['owner']; obs = _one_view(query['observations'][owner],row)
    world = schema.digest(_host(s)); root = schema.digest(dict(selector=selector,world=world,public=_host(p)))
    capsule = _host(chance); cid = schema.digest(capsule); reply = int(query['actions'][row,1-owner])
    group = dict(selector=selector,status='OBSERVED',root_id=root,world_id=world,chance_id=cid,chance=capsule,reply=reply,branches=[])
    legal = np.asarray(obs.legal_action_mask[0])
    for action in np.flatnonzero(legal):
        choices = query['actions'][row:row+1].at[0,owner].set(int(action))
        # Root action labels are canonical request coordinates, not physical
        # move slots or post-switch pointers. Map every branch at THIS root.
        physical = j.stack([encoder.map_actions(_one_view(query['observations'][side],row),s,
            j.asarray([side],j.int32),choices[:,side]) for side in (0,1)],axis=1)
        result = jax.block_until_ready(executor.advance(s,p,physical,chance))
        child = dict(action=int(action),root_id=root,world_id=world,chance_id=cid,reply=reply,
            status='REFUSED',observation=None,terminal=None,reason=None)
        if bool(result.step.invalid_actions[0]): child['reason']='ACTUAL_CHILD_ENGINE_REFUSAL'
        elif bool(result.step.state.done[0]):
            winner = int(result.step.state.winner[0])
            if winner not in (0,1,2): raise ValueError('actual child winner unavailable')
            child.update(status='COMPLETE',terminal=dict(done=True,winner=winner,state_id=schema.digest(_host(result.step.state))))
        else:
            view = jax.block_until_ready(executor.child_view(result.step.state,result.trackers))[owner]
            if int(view.error[0]): child['reason']='ACTUAL_CHILD_VIEW_REFUSAL'
            else: child.update(status='OBSERVED',observation=_observation(view))
        group['branches'].append(child)
    return root,world,_observation(obs),group


def _unknown(selector, status, reason):
    return dict(selector=selector,status=status,root_id=None,world_id=None,observation=None,terminal=None,reason=reason), \
        dict(selector=selector,status=status,root_id=None,world_id=None,branches=[],reason=reason)


def _collect_batch(executor, initialized, batch, owners, plan, params, directory, namespace, cost_entries, ordinal_start, save_index=lambda:None):
    """Private controller seam used by inert tests, never accepted as provenance."""
    state, public = initialized.state,initialized.trackers
    streams = j.asarray([g.stream_id for g in batch.games],j.int32); frames = j.zeros(len(streams),j.int32)
    live = np.ones(len(streams),bool); records = [{} for _ in live]; groups = [{} for _ in live]
    status = [None]*len(live); terminals = [None]*len(live)
    while live.any():
        state = state._replace(done=state.done|j.asarray(~live))
        query = jax.block_until_ready(executor.query(state,public,streams,frames,params))
        valid = np.asarray(query['input_valid']); requested = np.asarray(query['requested'])
        for row in np.flatnonzero(live):
            frame = int(frames[row]); owner = owners[row]
            if frame not in schema.FRAMES: continue
            selector = dict(stream=int(streams[row]),frame=frame,owner=owner)
            key = (selector['stream'],frame,owner)
            if not valid[row] or not requested[row,owner]:
                position, group = _unknown(selector,'REFUSED','ACTUAL_ROOT_NOT_REQUESTED_OR_VIEW_REFUSED')
                if key in cost_entries: cost_entries[key].update(status='REFUSED',reason=position['reason'])
            else:
                root,world,observation,group = _siblings(executor,state,public,query,row,selector)
                position = dict(selector=selector,status='OBSERVED',root_id=root,world_id=world,
                    observation=observation,terminal=None,reason=None)
                if key in cost_entries:
                    context = RootContext(_one(state,row),_one(public,row),streams[row:row+1],frames[row:row+1])
                    source_fields = {name:(_one(query[name],row) if name!='observations' else tuple(_one_view(o,row) for o in query[name])) for name in device_root_snapshot.SOURCE_FIELDS}
                    path = directory/'roots'/('%d-%d-%d.npz'%key)
                    artifact = device_root_snapshot.save(path,context,source_fields,namespace=namespace,selector=selector)
                    cost_entries[key].update(status='SAVED',artifact=artifact,reason=None)
            _publish(directory/'positions'/('%d-%d-%d.json'%key),dict(position=position,sibling=group))
            save_index()
            records[row][frame] = position; groups[row][frame] = group
        permitted = live & valid
        step_state = state._replace(done=state.done|j.asarray(~permitted))
        result = jax.block_until_ready(executor.advance(step_state,public,query['physical'],query['chance']))
        frames = frames+j.asarray(live,j.int32)
        refused = live & (~valid | np.asarray(result.step.invalid_actions))
        completed = live & ~refused & np.asarray(result.step.state.done)
        capped = live & ~refused & ~completed & (np.asarray(frames)>=plan['max_requests'])
        for row in np.flatnonzero(refused|completed|capped):
            disposition = 'REFUSED' if refused[row] else 'COMPLETE' if completed[row] else 'CAPPED'
            winner = int(result.step.state.winner[row]) if completed[row] else None
            if completed[row] and winner not in (0,1,2): raise ValueError('actual ordinary terminal winner required')
            terminal = dict(status=disposition,done=bool(completed[row]),winner=winner,frame=int(frames[row]),
                state_id=schema.digest(_host(_one(result.step.state,int(row)))))
            status[row] = disposition; terminals[row] = terminal
            positions = []; siblings = []
            for frame in schema.FRAMES:
                selector = dict(stream=int(streams[row]),frame=frame,owner=owners[row])
                position,group = (records[row][frame],groups[row][frame]) if frame in records[row] else _unknown(selector,'UNREACHED','GAME_ENDED_BEFORE_FIXED_SELECTOR')
                position = dict(position,terminal=terminal)
                positions.append(position); siblings.append(group)
            _publish(directory/'games'/('%03d.json'%(ordinal_start+int(row))),dict(game=asdict(batch.games[row]),terminal=terminal,positions=positions,siblings=siblings))
        live &= ~(refused|completed|capped); state,public = result.step.state,result.trackers
    return status


def collect(reference, prepared, pool, *, gender_policy, resume=False):
    """Only public producer entry: strict original archives and actual engine.

    Resume advances only whole batches never marked attempted. An interrupted
    attempted cell without its immutable completed record stays FAILED; it is
    never replayed. The one collection's declaration/output root stays fixed.
    """
    from .v5_device_backend import Prepared
    from .engine_request_source import _gender
    from . import source_provenance
    if type(resume) is not bool: raise ValueError('explicit no-replay resume mode required')
    plan,batches,games,labels = schema.declaration(reference)
    if type(prepared) is not Prepared or prepared.source_sha256!=plan['source_sha256'] or prepared.pool_sha256!=plan['pool_sha256']:
        raise ValueError('genuine same frozen nontraining Prepared/pool required')
    if (source_provenance._pool(pool,rehash=True)['pool_sha256']!=prepared.pool_sha256
            or pool.metadata()!=json.loads(prepared.metadata_json)):
        raise ValueError('actual constructor pool differs from prepared bank')
    binding = json.loads(prepared.source_binding_json)
    if (binding['source_commit']!=plan['source_commit'] or prepared.category_sha256!=plan['category_vocabulary_sha256']
            or prepared.category_size!=plan['category_count']):
        raise ValueError('bank source/category binding differs')
    from pokezero.v5_contract import source_variant_census
    if source_variant_census(prepared.source)['source_rows_sha256']!=labels.source_binding_sha256:
        raise ValueError('bank original source-label namespace differs')
    loaded = {}; parameters = []; config = None
    for actor in plan['actors']:
        key = (actor['archive']['sha256'],actor['checkpoint_sha256'])
        if key not in loaded:
            loaded[key] = v5_model.load(actor['archive']['path'],expected_archive_sha256=key[0],expected_checkpoint_sha256=key[1],
                expected_category_vocabulary_sha256=plan['category_vocabulary_sha256'],expected_label_vocabulary_sha256=labels.sha256,enable_v5=True)
        params,actual,_ = loaded[key]
        if config is not None and actual!=config: raise ValueError('same frozen actor geometry required')
        config = actual; parameters.append(params)
    parameters = tuple(jax.block_until_ready(parameters)); directory = Path(plan['output_directory'])
    namespace = dict(source_commit=plan['source_commit'],source_sha256=prepared.source_sha256,pool_sha256=prepared.pool_sha256,
        category_vocabulary_sha256=prepared.category_sha256,label_vocabulary_sha256=labels.sha256,
        pool_metadata_sha256=binding['pool_metadata_sha256'],public_source_hash=prepared.source.metadata.source_hash)
    journal_path = directory/'journal.json'
    if resume:
        journal = json.loads(journal_path.read_text())
        if journal['declaration']!=reference or journal['namespace']!=namespace: raise ValueError('same interrupted declaration/source required')
    else:
        directory.mkdir(parents=True,exist_ok=False); (directory/'games').mkdir(); (directory/'roots').mkdir(); (directory/'positions').mkdir()
        journal = dict(declaration=reference,namespace=namespace,status=['NOT_ATTEMPTED']*256)
        _publish(journal_path,journal,mutable=True)
    cost_entries = {tuple(s[k] for k in ('stream','frame','owner')):dict(selector=s,status='UNREACHED',artifact=None,reason='FIXED_SELECTOR_NOT_REACHED') for s in plan['cost_selectors']}
    root_index_path = directory/'cost-root-index.json'
    if resume and root_index_path.exists():
        old = json.loads(root_index_path.read_text())
        if (old['schema']!=schema.ROOT_INDEX or old['declaration']!=reference or old['namespace']!=namespace
                or old['cost_recipe_seed']!=plan['cost_recipe_seed']
                or [e['selector'] for e in old['cost_roots']]!=plan['cost_selectors']):
            raise ValueError('same declared saved-root index/source/order required')
        for entry in old['cost_roots']: cost_entries[tuple(entry['selector'][k] for k in ('stream','frame','owner'))]=entry
    def save_index():
        _publish(root_index_path,dict(schema=schema.ROOT_INDEX,declaration=reference,namespace=namespace,
            cost_recipe_seed=plan['cost_recipe_seed'],cost_roots=list(cost_entries.values())),mutable=True)
    # Recover an exclusively published snapshot even if the process stopped
    # between that write and its index update. No game is rerun for recovery.
    if resume:
        for key,entry in cost_entries.items():
            path = directory/'roots'/('%d-%d-%d.npz'%key)
            if entry['status']!='SAVED' and path.exists():
                candidate = dict(entry,status='SAVED',artifact=dict(path=str(path),sha256=run._file_sha(path)),reason=None)
                device_root_snapshot.load(candidate,dict(namespace=namespace))
                cost_entries[key]=candidate
            elif entry['status']!='SAVED':
                pending=directory/'positions'/('%d-%d-%d.json'%key)
                if pending.exists():
                    position=json.loads(pending.read_text())['position']
                    if position['selector']!=entry['selector']:raise ValueError('same pending cost selector required')
                    if position['status']=='REFUSED':entry.update(status='REFUSED',reason=position['reason'])
    save_index()
    initialize = jax.jit(lambda raw:transition.initialize_with_public(raw,prepared.engine_tables,belief_tables=prepared.observation_tables.belief))
    for batch_index,batch in enumerate(batches):
        start = batch_index*16; indices = np.asarray([g.team_indices for g in batch.games],np.int32)
        if any(s!='NOT_ATTEMPTED' for s in journal['status'][start:start+16]):
            for i in range(start,start+16):
                path = directory/'games'/('%03d.json'%i)
                if path.exists(): journal['status'][i]=json.loads(path.read_text())['terminal']['status']
                elif journal['status'][i]=='ATTEMPTED': journal['status'][i]='FAILED'
            continue
        journal['status'][start:start+16]=['ATTEMPTED']*16; _publish(journal_path,journal,mutable=True)
        raw = transition.state_from_pool(pool.arrays,indices,seed=batch.constructor_seed)
        if batch.gender_seed is None: raise ValueError('actual declared gender roots required')
        raw = _gender(raw,pool,indices,dict(policy=gender_policy,seed=batch.gender_seed,domain=batch.gender_domain,
            root_ids=np.asarray(batch.gender_root_ids,np.uint32)))
        initialized = jax.block_until_ready(initialize(raw))
        executor = make_executor(prepared,config,sampling_seed=batch.sampling_seed)
        status = _collect_batch(executor,initialized,batch,[(start+i)%2 for i in range(16)],plan,parameters,directory,namespace,cost_entries,start,save_index)
        journal['status'][start:start+16]=status; _publish(journal_path,journal,mutable=True)
        save_index()
    positions = []; groups = []
    for i,game in enumerate(games):
        path = directory/'games'/('%03d.json'%i)
        if path.exists():
            record = json.loads(path.read_text()); positions.extend(record['positions']); groups.extend(record['siblings'])
        else:
            for frame in schema.FRAMES:
                selector = dict(stream=game.stream_id,frame=frame,owner=i%2)
                pending = directory/'positions'/('%d-%d-%d.json'%(game.stream_id,frame,i%2))
                if pending.exists():
                    record=json.loads(pending.read_text());p,g=record['position'],record['sibling']
                else:p,g = _unknown(selector,journal['status'][i],'INTERRUPTED_ATTEMPT_NO_RETURNED_GAME')
                positions.append(p); groups.append(g)
    header = dict(split='PHASE_C_FIXED_HELDOUT',rows=1024,category_vocabulary_sha256=prepared.category_sha256,
        label_vocabulary_sha256=labels.sha256,category_count=prepared.category_size,device_bank=reference,device_source=plan['source_commit'])
    publication = directory/('publication-'+uuid.uuid4().hex); publication.mkdir()
    artifacts = {}
    for name,rows,kind in [('positions',positions,schema.POSITIONS),('siblings',groups,schema.SIBLINGS)]:
        path = publication/(name+'.jsonl')
        with path.open('x') as stream:
            for row in [dict(header,schema=kind),*rows]: stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        artifacts[name]=dict(path=str(path),sha256=run._file_sha(path))
    _publish(journal_path,journal,mutable=True)
    _publish(publication/'summary.json',dict(schema=schema.DECLARATION,declared_games=256,position_slots=1024,sibling_root_groups=1024,
        game_dispositions={s:journal['status'].count(s) for s in set(journal['status'])},artifacts=artifacts,
        actual_ranking_accuracy='UNMEASURED_NO_MATCHED_TERMINAL_CONTINUATIONS',stop_rule=schema.STOP_RULE))
    return artifacts
