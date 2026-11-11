"""Quarantined fixed-bank continuation producer, with zero work on import.

Invoke this FILE (not -m) with --original-source: original engine, observation,
constructor and archive-loader modules are loaded from the bank's frozen tree.
The new rules/reducer is loaded by file path, never by substituting a package in
that original namespace. All data and model files stay on the caller's remote
storage. This is not a training collector or replay of original outcomes.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

_RULES_PATH=Path(__file__).resolve().parents[1]/'src'/'pokezero'/'rollout_truth.py'

def rules():
    name='_pokezero_rollout_truth_rules'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,_RULES_PATH);mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]


def read(reference):
    blob=Path(reference['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=reference['sha256']:raise ValueError('immutable artifact bytes differ')
    return json.loads(blob)


def label_vocabulary_sha256(metadata):
    """The serialized vocabulary stores fields; its digest is a property."""
    from pokezero.v5_contract import LabelVocabulary
    return LabelVocabulary(**metadata).sha256


def publish(path,value):
    path=Path(path)
    with path.open('x')as out:json.dump(value,out,sort_keys=True,allow_nan=False,separators=(',',':'))
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bank_manifest(config,schema):
    """Validate original complete bank, before original runtime/model work."""
    decl,batches,games,labels=schema.declaration(config['declaration'])
    ph,positions=schema._records(config['positions'],schema.POSITIONS)
    sh,groups=schema._records(config['siblings'],schema.SIBLINGS)
    protocol=dict(positions=config['positions'],siblings=config['siblings'],label_vocabulary_sha256=labels.sha256)
    schema.resolve_device_bank(protocol,None) # Original full1024/all-child guard.
    if ph['device_bank']!=config['declaration']or sh['device_bank']!=config['declaration']:
        raise ValueError('literal bank declaration reference differs')
    index=read(config['root_index'])
    if index['declaration']!=config['declaration']or index['namespace']['source_commit']!=decl['source_commit']:
        raise ValueError('original saved-root index differs')
    observed=sum(p['status']=='OBSERVED'for p in positions);children=sum(len(g['branches'])for g in groups)
    if len(positions)!=1024 or len(games)!=256:
        raise ValueError('original declared bank denominator differs; no survivor replacement')
    dispositions={}
    for start in range(0,len(positions),4):
        terms=[p['terminal']for p in positions[start:start+4]]
        if any(x!=terms[0]for x in terms):raise ValueError('original per-game terminal provenance differs by frame')
        status='UNKNOWN'if terms[0]is None else terms[0]['status']
        dispositions[status]=dispositions.get(status,0)+1
    manifest=dict(schema='pokezero.rollout-truth.bank.v1',declaration=config['declaration'],positions=config['positions'],siblings=config['siblings'],root_index=config['root_index'],
        source_commit=decl['source_commit'],source_sha256=decl['source_sha256'],pool_sha256=decl['pool_sha256'],
        declared_games=len(games),declared_slots=len(positions),observed_slots=observed,declared_children=children,game_dispositions=dispositions,
        brief_comparison=dict(reported_observed=988,actual_minus_reported_observed=observed-988,reported_children=8378,actual_minus_reported_children=children-8378),
        declared_rollouts=64*children,stored_physical_root_entries=len(index['cost_roots']),
        original_actors=decl['actors'],
        original_outcome_scope='Original collector actors/outcomes, not outcomes of the evaluated head or raw9375 continuations; terminal only on root calibration; NEVER copied to alternate children',
        missing_root_policy='Quarantined exact prefix reconstruction or MISSING; no old terminal rerun, no refill',
        truth_policy_checkpoint_sha256=rules().CHAMPION)
    return decl,batches,games,labels,positions,groups,index,manifest


def _import_original(source):
    source=Path(source).resolve();sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source))
    from pokezero import v5_worker as worker,v5_device_evaluation_bank as schema
    for module in (worker,schema):
        if not Path(module.__file__).resolve().is_relative_to(source):raise ValueError('original source package already shadowed')
    return worker,schema


def prepare_original(worker,decl,runtime_ref):
    """Original source/pool/category factory, without any mock production Plan."""
    runtime=read(runtime_ref)
    if runtime['source_commit']!=decl['source_commit']:raise ValueError('bank original numerical source required')
    _,v,gender,_=worker._runtime_metadata(runtime,expected_pool_sha256=decl['pool_sha256'])
    worker._source(runtime);worker._memory(runtime['memory_policy'],apply=True);worker._probe(runtime)
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.live_protocol_actor import ActorContract
    from gpu_engine import source_provenance,public_belief,v4_encoder,v5_device_backend
    source=load_gen3_randbat_source_cached(runtime['showdown']['root']);dex=load_showdown_dex_cached(runtime['showdown']['root'])
    contract=ActorContract.champion();vocabulary=CategoryVocabulary(tuple(v['tokens']),v['oov_buckets'],v['aliases'])
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),dex,vocabulary,checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    pool=source_provenance.load_constructor_pool(runtime['constructor_pool']['path'])
    prepared=v5_device_backend.prepare(catalog,source,pool,source_commit=runtime['source_commit'])
    if prepared.source_sha256!=decl['source_sha256']or prepared.pool_sha256!=decl['pool_sha256']or prepared.category_sha256!=decl['category_vocabulary_sha256']:
        raise ValueError('original compiled source/pool/category differs')
    return prepared,pool,gender


def accept_root(state,public,view,selector,position,bank,schema):
    world=schema.digest(bank._host(state))
    root=schema.digest(dict(selector=selector,world=world,public=bank._host(public)))
    return world==position['world_id']and root==position['root_id']and bank._observation(view)==position['observation']


def reconstruct(prepared,pool,gender,decl,batches,positions,index,directory,bank,schema,snapshot):
    """ONE original prefix pass, never beyond frame24 or a terminal/refusal.

    Output every original selector. Digests and literal observation must ALL
    match before publishing any physical root. Mismatch stays missing with no
    source tweak/retry. Original terminal records and labels are never written.
    """
    import jax
    import jax.numpy as j
    import numpy as np
    from gpu_engine import transition,v5_model
    from gpu_engine.engine_request_source import _gender
    from gpu_engine.v5_device_rollout import RootContext
    params=[];config=None
    for actor in decl['actors']:
        p,c,_=v5_model.load(actor['archive']['path'],expected_archive_sha256=actor['archive']['sha256'],expected_checkpoint_sha256=actor['checkpoint_sha256'],expected_category_vocabulary_sha256=decl['category_vocabulary_sha256'],expected_label_vocabulary_sha256=label_vocabulary_sha256(decl['label_vocabulary']),enable_v5=True)
        if config is not None and config!=c:raise ValueError('original actor geometry differs')
        config=c;params.append(p)
    executor=bank.make_executor(prepared,config)
    initialize=jax.jit(lambda s:transition.initialize_with_public(s,prepared.engine_tables,belief_tables=prepared.observation_tables.belief))
    entries=[dict(selector=p['selector'],status='MISSING',artifact=None,reason='ORIGINAL_ROOT_UNRECORDED')for p in positions]
    for bi,batch in enumerate(batches):
        indices=np.asarray([g.team_indices for g in batch.games],np.int32)
        raw=transition.state_from_pool(pool.arrays,indices,seed=batch.constructor_seed)
        raw=_gender(raw,pool,indices,dict(policy=gender,seed=batch.gender_seed,domain=batch.gender_domain,root_ids=np.asarray(batch.gender_root_ids,np.uint32)))
        initial=jax.block_until_ready(initialize(raw));state,public=initial.state,initial.trackers
        streams=j.asarray([g.stream_id for g in batch.games],j.int32);frames=j.zeros(16,j.int32);live=np.ones(16,bool)
        last=np.asarray([max((p['selector']['frame']for p in positions[(bi*16+r)*4:(bi*16+r+1)*4]if p['status']=='OBSERVED'),default=-1)for r in range(16)])
        live &= last>=0
        for frame in range(25):
            if not live.any():break
            state=state._replace(done=state.done|j.asarray(~live))
            query=jax.block_until_ready(executor.query(state,public,streams,frames,tuple(params),j.asarray(batch.sampling_seed,j.uint32)))
            valid=np.asarray(query['input_valid']);requested=np.asarray(query['requested'])
            if frame in schema.FRAMES:
                for row in range(16):
                    ordinal=(bi*16+row)*4+schema.FRAMES.index(frame);position=positions[ordinal];entry=entries[ordinal];owner=position['selector']['owner']
                    if position['status']!='OBSERVED':entry['reason']='ORIGINAL_'+position['status'];continue
                    if not live[row]or not valid[row]or not requested[row,owner]:entry['reason']='PREFIX_ROOT_NOT_AVAILABLE';continue
                    s=bank._one(state,row);p=bank._one(public,row);v=bank._one_view(query['observations'][owner],row)
                    if not accept_root(s,p,v,entry['selector'],position,bank,schema):entry['reason']='PREFIX_DIGEST_OR_PUBLIC29_MISMATCH';continue
                    context=RootContext(s,p,streams[row:row+1],frames[row:row+1])
                    fields={name:(tuple(bank._one_view(v,row)for v in query[name])if name=='observations'else bank._one(query[name],row))for name in snapshot.SOURCE_FIELDS}
                    artifact=snapshot.save(directory/('root-%04d.npz'%ordinal),context,fields,namespace=index['namespace'],selector=entry['selector'])
                    entry.update(status='SAVED',artifact=artifact,reason=None)
            if frame==24:break # Never replay later original terminal/outcome.
            permitted=live&valid&(frame<last)
            out=jax.block_until_ready(executor.advance(state._replace(done=state.done|j.asarray(~permitted)),public,query['physical'],query['chance']))
            frames=frames+j.asarray(live,j.int32)
            live&=permitted&~(np.asarray(out.step.invalid_actions)|np.asarray(out.step.state.done))
            state,public=out.step.state,out.trackers
    return entries


class RawExecutor:
    """One original9375 raw policy, both seats; no head-dependent decisions."""
    def __init__(self,prepared,archive):
        import jax
        import jax.numpy as j
        from gpu_engine import champion,transition,engine_device_observation as obs,engine_device_trackers as public,benchmark_device_selfplay as sample
        params,config,meta=champion.load(archive['path'])
        if meta['weights_archive_sha256']!=archive['sha256']or meta['checkpoint_sha256']!=rules().CHAMPION or config.tokens!=23 or config.vocabulary!=prepared.category_size:
            raise ValueError('immutable original9375 champion23 archive required')
        self.prepared=prepared;self.meta=meta
        def query(s,p,keys):
            requested=sample.request_seats(s)
            mask=transition.action_mask(s,prepared.engine_tables)&requested[...,None]
            views=tuple(obs.encode(s,public.readout(p,j.full(len(s.done),side,j.int32)),prepared.observation_tables,observer_sides=j.full(len(s.done),side,j.int32),legal_action_mask=mask[:,side])for side in (0,1))
            pred=sample.run_actor(params,config,views,actor_kind='champion23')
            legal=j.stack([v.legal_action_mask for v in views],axis=1)
            finite=pred.finite&(j.stack([v.error for v in views],axis=1)==0)&j.any(legal,axis=-1)
            # Exact at-play RAW9375: legal greedy action; argmax chooses the
            # lower request index on ties. Replicates randomize engine chance,
            # never the policy. Absent seats retain the original -1 marker.
            actions=j.where(j.any(legal,axis=-1),j.argmax(j.where(legal,pred.policy_logits,-j.inf),axis=-1),-1).astype(j.int32)
            physical=j.stack([obs.map_actions(views[side],s,j.full(len(s.done),side,j.int32),j.maximum(actions[:,side],0))for side in (0,1)],axis=1)
            return views,pred.policy_logits,physical,j.all(~requested|finite,axis=1)
        self.query=jax.jit(query)
        self.chance=jax.jit(transition.sample_chance_inputs)
        self.policy_keys=jax.jit(lambda policy,step:jax.vmap(lambda k:jax.vmap(lambda side:jax.random.fold_in(jax.random.fold_in(k,step),side))(j.arange(2)))(policy))
        self.advance=jax.jit(lambda s,p,a,c:transition.step_with_public(s,a,prepared.engine_tables,p,belief_tables=prepared.observation_tables.belief,chance_inputs=c))
        self.view=jax.jit(lambda s,p:query(s,p,j.zeros((len(s.done),2,2),j.uint32))[0])

    def continuation_batch(self,children,recipe):
        """Fixed32×64 program, preserving unavailable/tail child lane positions.

        None slots are ignored padding, not a replacement child. Padding copies
        an actual verified state/public carry and is marked inactive.
        """
        import jax
        import jax.numpy as j
        import numpy as np
        from gpu_engine import transition
        if not children:return []
        if len(children)>32:raise ValueError('one fixed32-child physical batch required')
        count=len(children);slots=list(children)+[None]*(32-count);n=2048
        template=next((c for c in slots if c is not None),getattr(self,'padding',None))
        if template is None:
            # No verified physical state anywhere: no safe actor input exists.
            # Preserve all unknowns, never invent a padding battle.
            return [None]*count
        keys=[k for c in slots for k in (recipe.keys(c['selector'],c['action'],c['reply'],c['chance_id'])if c is not None else((0,0),)*64)]
        states=jax.tree.map(lambda *xs:j.concatenate(xs,axis=0),*((template if c is None else c)['state']for c in slots))
        publics=jax.tree.map(lambda *xs:j.concatenate(xs,axis=0),*((template if c is None else c)['public']for c in slots))
        s=jax.tree.map(lambda a:j.repeat(a,64,axis=0),states);p=jax.tree.map(lambda a:j.repeat(a,64,axis=0),publics)
        owners=np.repeat([0 if c is None else c['selector']['owner']for c in slots],64)
        scientific=np.repeat([c is not None for c in slots],64)
        s=s._replace(done=s.done|j.asarray(~scientific))
        s=s._replace(keys=jax.vmap(jax.random.PRNGKey)(j.asarray([x[0]for x in keys],j.uint32)))
        policy=jax.vmap(jax.random.PRNGKey)(j.asarray([x[1]for x in keys],j.uint32))
        rows=[dict(replicate=i%64,status='MISSING',outcome=None,requests=0,reason=None)for i in range(n)]
        live=~np.asarray(s.done)
        def terminal(i):
            winner=int(s.winner[i])
            if winner not in (0,1,2):raise ValueError('actual continuation terminal winner required')
            rows[i].update(status='COMPLETE',outcome=0 if winner==2 else 1 if winner==owners[i]else -1,reason=None)
        for i in np.flatnonzero(scientific&~live):terminal(i)
        for step in range(recipe.max_requests):
            if not live.any():break
            s=s._replace(done=s.done|j.asarray(~live))
            seatkeys=self.policy_keys(policy,j.asarray(step,j.uint32))
            _,_,physical,valid=jax.block_until_ready(self.query(s,p,seatkeys))
            valid=np.asarray(valid)
            out=jax.block_until_ready(self.advance(s._replace(done=s.done|j.asarray(~(live&valid))),p,physical,self.chance(s)))
            failed=live&(~valid|np.asarray(out.step.invalid_actions));s,p=out.step.state,out.trackers
            done=live&~failed&np.asarray(s.done)
            for i in np.flatnonzero(live):rows[i]['requests']+=1
            for i in np.flatnonzero(failed):rows[i].update(status='REFUSED',reason='ACTUAL_VIEW_ACTOR_OR_ENGINE_REFUSAL')
            for i in np.flatnonzero(done):terminal(i)
            live&=~(failed|done)
        for i in np.flatnonzero(live):rows[i].update(status='CAPPED',reason='FIXED_CONTINUATION_REQUEST_CAP')
        return [None if c is None else rows[64*i:64*(i+1)]for i,c in enumerate(children)]


def child_step(executor,root,fields,group,action,reply):
    import jax
    import jax.numpy as j
    from gpu_engine import transition,engine_device_observation as obs
    owner=group['selector']['owner'];s,p=root.state,root.public
    # The appended effect stream (fidelity R) may be absent from a capsule saved before it; the
    # transition then derives it from the keys exactly as sample_chance_inputs does.
    fields=set(transition.ChanceInputs._fields)
    if not fields-{'effect'}<=set(group['chance'])<=fields:raise ValueError('complete original chance capsule required')
    chance=transition.ChanceInputs(**{k:j.asarray(group['chance'][k],j.int32)for k in transition.ChanceInputs._fields if k in group['chance']})
    choices=fields['actions'].at[0,owner].set(action).at[0,1-owner].set(reply)
    physical=j.stack([obs.map_actions(fields['observations'][side],s,j.asarray([side],j.int32),choices[:,side])for side in (0,1)],axis=1)
    return jax.block_until_ready(executor.advance(s,p,physical,chance))


def missing(recipe,reason):
    return [dict(replicate=i,status='MISSING',outcome=None,requests=0,reason=reason)for i in range(recipe.rollouts)]


def grid(executor,root,fields,group,directory,ordinal):
    """Separate head-only full reply grid. Never adds truth rollout children."""
    import jax
    import jax.numpy as j
    import numpy as np
    owner=group['selector']['owner'];views,logits,_,valid=jax.block_until_ready(executor.query(root.state,root.public,j.zeros((1,2,2),j.uint32)))
    if not bool(valid[0]):return dict(status='REFUSED',reason='RAW_POLICY_ROOT_REFUSED',artifact=None)
    own=np.flatnonzero(np.asarray(views[owner].legal_action_mask[0]));peer=np.flatnonzero(np.asarray(views[1-owner].legal_action_mask[0]))
    requested=bool(fields['requested'][0,1-owner]);replies=peer.tolist()if requested else[0]
    if len(own)==0 or not replies:return dict(status='REFUSED',reason='NO_LEGAL_GRID_DENOMINATOR',artifact=None)
    lp=np.asarray(logits[0,1-owner],float)[replies]if requested else np.asarray([0.])
    weights=np.exp(lp-lp.max());weights/=weights.sum()
    arrays={k:[]for k in ('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask')}
    rows=[]
    for action in own:
        for reply,weight in zip(replies,weights):
            out=child_step(executor,root,fields,group,int(action),reply)
            row=dict(action=int(action),reply=reply,weight=float(weight),status='REFUSED',terminal_value=None)
            view=jax.block_until_ready(executor.view(out.step.state,out.trackers))[owner]
            if not bool(out.step.invalid_actions[0]):
                if bool(out.step.state.done[0]):
                    winner=int(out.step.state.winner[0])
                    if winner not in (0,1,2):raise ValueError('grid actual terminal winner required')
                    row.update(status='COMPLETE',terminal_value=0 if winner==2 else 1 if winner==owner else -1)
                elif int(view.error[0])==0:row['status']='OBSERVED'
            rows.append(row)
            for name in arrays:arrays[name].append(np.asarray(getattr(view,name)[0]))
    path=directory/('grid-%04d.npz'%ordinal)
    with path.open('xb')as stream:np.savez_compressed(stream,**{k:np.stack(v)for k,v in arrays.items()})
    return dict(status='OBSERVED'if all(r['status']!='REFUSED'for r in rows)else'REFUSED',
        reason=None if all(r['status']!='REFUSED'for r in rows)else'UNKNOWN_REQUIRED_REPLY_NO_RENORMALIZATION',
        rows=rows,own_actions=[int(a)for a in own],replies=replies,reply_weights=[float(w)for w in weights],artifact=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        scope='SEPARATE_HEAD_ONLY_GRID; original fixed chance/single physical world; raw9375 reply policy; NOT additional truth rollouts')


def collect(config,source,directory,*,enable_truth=False):
    if enable_truth is not True:raise ValueError('explicit fixed-bank truth enablement required')
    worker,schema=_import_original(source)
    decl,batches,games,labels,positions,groups,index,manifest=bank_manifest(config,schema)
    prespec=read(config['prespec'])
    if (prespec['bank_declaration']!=config['declaration']or prespec['seed']!=config['seed']or prespec['rollouts_per_child']!=64
            or prespec['max_requests']!=config['max_requests']or prespec['split_seed']!=config['split_seed']
            or prespec['policy']!='FROZEN_RAW_CHAMPION9375_DETERMINISTIC_LEGAL_ARGMAX_BOTH_SEATS_ALL_HEADS_SAME_TRUTH'):
        raise ValueError('same pre-outcome bank/seed/truth declaration required')
    recipe=rules().Recipe(config['seed'],max_requests=config['max_requests'])
    prepared,pool,gender=prepare_original(worker,decl,config['runtime'])
    from gpu_engine import v5_device_evaluation_bank as bank,device_root_snapshot as snapshot
    if any(not Path(m.__file__).resolve().is_relative_to(Path(source).resolve())for m in (bank,snapshot)):
        raise ValueError('original engine package already shadowed')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);(directory/'roots').mkdir();(directory/'grids').mkdir()
    manifest.update(producer_files={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()for p in (Path(__file__).resolve(),_RULES_PATH)},prespec=config['prespec'],runtime=config['runtime'],recipe=vars(recipe),raw_actor=config['raw_actor'],
        continuation_policy='DETERMINISTIC_LEGAL_ARGMAX_BOTH_SEATS_LOWER_INDEX_TIES',physical_continuation_batch=2048,padding='Inactive copies of verified state; unavailable/tail slots never truth; if no physical root exists all truth stays missing without execution',critic_split_seed=config['split_seed'],critic_folds=rules().folds([p['selector']for p in positions],seed=config['split_seed']),
        privileged_critic='NOT_TRAINED; actual B checkpoint required; frozen B trunk and original public-head warmstart, small steps, epoch0 eligible. Five game-clustered folds: test k, validation (k+1)%5, train other3. Every reported critic prediction OUT OF FOLD. Crossfit metric distinct from a final all-bank fitted critic. No extra games.')
    publish(directory/'manifest.json',manifest)
    entries=reconstruct(prepared,pool,gender,decl,batches,positions,index,directory/'roots',bank,schema,snapshot)
    root_index=dict(namespace=index['namespace'],cost_roots=entries)
    publish(directory/'reconstructed-root-index.json',root_index)
    executor=RawExecutor(prepared,config['raw_actor'])
    # One real verified snapshot supplies inactive padding even when a whole
    # chunk is missing. No private state is invented for that missing chunk.
    first=next((e for e in entries if e['status']=='SAVED'),None)
    if first is not None:
        padding,_=snapshot.load(first,root_index)
        executor.padding=dict(state=padding.state,public=padding.public)
    import jax
    children=0;known=0;unknown=0
    pending=[]
    def flush(truth):
        nonlocal children,known,unknown
        # Keep unavailable children in their ORIGINAL32-child lane slots.
        results=executor.continuation_batch([r.get('_continuation')for r in pending],recipe)
        for record,rows in zip(pending,results,strict=True):
            child=record.pop('_continuation',None)
            if child is not None:
                if rows is None:raise ValueError('available child lost its fixed-batch dispositions')
                record['rollouts']=rows
            summary=rules().summarize_rollouts(record['rollouts']);record['summary']=summary
            children+=1;known+=summary['known'];unknown+=summary['unknown']
            truth.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        truth.flush();pending.clear()
    with (directory/'truth.jsonl').open('x')as truth,(directory/'head-grids.jsonl').open('x')as grids:
        truth.write(json.dumps(manifest,sort_keys=True)+'\n')
        for ordinal,(entry,position,group)in enumerate(zip(entries,positions,groups,strict=True)):
            root=fields=None
            if entry['status']=='SAVED':root,fields=snapshot.load(entry,root_index)
            gridrow=dict(selector=position['selector'],status='MISSING',reason=entry['reason'],artifact=None)
            if root is not None:gridrow.update(grid(executor,root,fields,group,directory/'grids',ordinal))
            grids.write(json.dumps(gridrow,sort_keys=True,allow_nan=False)+'\n');grids.flush()
            for branch in group['branches']:
                rows=missing(recipe,entry['reason']or'ORIGINAL_BRANCH_NOT_AVAILABLE');continuation=None
                if root is not None:
                    if int(fields['actions'][0,1-group['selector']['owner']])!=group['reply']:raise ValueError('original fixed reply differs from saved source')
                    out=child_step(executor,root,fields,group,branch['action'],group['reply'])
                    if bool(out.step.invalid_actions[0]):rows=missing(recipe,'ACTUAL_CHILD_ENGINE_REFUSAL')
                    else:
                        owner=group['selector']['owner'];valid=False
                        if branch['status']=='COMPLETE':
                            actual=dict(done=bool(out.step.state.done[0]),winner=int(out.step.state.winner[0]),state_id=schema.digest(bank._host(out.step.state)))
                            valid=actual==branch['terminal']
                        elif branch['status']=='OBSERVED'and not bool(out.step.state.done[0]):
                            view=jax.block_until_ready(executor.view(out.step.state,out.trackers))[owner]
                            valid=int(view.error[0])==0 and bank._observation(view)==branch['observation']
                        if valid:continuation=dict(state=out.step.state,public=out.trackers,selector=group['selector'],action=branch['action'],reply=group['reply'],chance_id=group['chance_id'])
                        else:rows=missing(recipe,'ORIGINAL_CHILD_BYTES_OR_TERMINAL_MISMATCH')
                record=dict(selector=group['selector'],action=branch['action'],reply=group['reply'],chance_id=group['chance_id'],
                    root_id=group['root_id'],world_id=group['world_id'],rollouts=rows,_continuation=continuation)
                pending.append(record)
                if len(pending)==32:flush(truth)
        if pending:flush(truth)
    if children!=manifest['declared_children']or known+unknown!=64*children:raise ValueError('truth denominator dropped')
    result=dict(schema=rules().TRUTH,manifest=manifest,children=children,terminal_rollouts=known,unknown_rollouts=unknown,
        reconstructed_roots=sum(e['status']=='SAVED'for e in entries),missing_roots=sum(e['status']!='SAVED'for e in entries),
        truth= dict(path=str(directory/'truth.jsonl'),sha256=hashlib.sha256((directory/'truth.jsonl').read_bytes()).hexdigest()),
        head_grids=dict(path=str(directory/'head-grids.jsonl'),sha256=hashlib.sha256((directory/'head-grids.jsonl').read_bytes()).hexdigest()),
        scientific_qualification=False,scope='Original complete declared game/slot/child denominator retained;64 NEW frozen-policy continuations per actual declared child; brief differences explicit; missing stays unknown; no original terminal rerun/rescore')
    publish(directory/'summary.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--original-source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--enable-truth',action='store_true');args=parser.parse_args(argv)
    if not args.enable_truth:raise ValueError('explicit rollout-truth enablement required')
    config=read(dict(path=args.config,sha256=args.expected_config_sha256))
    collect(config,args.original_source,args.output,enable_truth=True)

if __name__=='__main__':main()
