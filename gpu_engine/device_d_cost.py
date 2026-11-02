"""One fixed saved-root Arm-D cost cohort; no games, actor actions or training.

Actual execution uses the unchanged genuine Provider on the saved-bank pool,
with a strict current-source D actor archive. Selectors precede all outcomes.
Unreached/refused slots stay; cold-first timing stays in the prescribed p95.
"""
from dataclasses import dataclass
import math
import time
import jax
import jax.numpy as j
import numpy as np
from pokezero import v5_contract as c,v5_gumbel as original,v5_run as runs
from . import v5_device_rollout as rollout,v5_device_teacher as teacher,v5_model
from . import engine_device_observation as encoder,engine_device_trackers as trackers
from . import transition
from .benchmark_device_selfplay import request_seats

SCHEMA='pokezero.device-d-saved-root-cost.v1'
FRAMES=(0,4,12,24)
SLOTS_PER_FRAME=10
THRESHOLD_SECONDS=30.


def select(plan):
    """First10 hash-selected unused streams per frame, declaration order only."""
    if type(plan)is not runs.Plan or plan.model.arm!='D':raise ValueError('genuine saved-bank D Plan required')
    recipe=original.Recipe(plan.selection_seed);used=set();result=[]
    for frame in FRAMES:
        group=[]
        for ordinal,game in enumerate(plan.games):
            address=c.DecisionAddress(game.stream_id,frame,ordinal%2)
            if game.stream_id not in used and recipe.selected(address):
                group.append(address);used.add(game.stream_id)
                if len(group)==SLOTS_PER_FRAME:break
        if len(group)!=SLOTS_PER_FRAME:raise ValueError('declared bank has insufficient unique hash-selected cost addresses')
        result.extend(group)
    return tuple(result)


@dataclass(frozen=True)
class Slot:
    address:c.DecisionAddress
    status:str
    context:object=None
    reason:str|None=None

    def __post_init__(self):
        if type(self.address)is not c.DecisionAddress or self.status not in ('AVAILABLE','UNREACHED','REFUSED'):
            raise ValueError('literal fixed saved-root slot required')
        if self.status=='AVAILABLE':
            if type(self.context)is not rollout.RootContext or self.reason is not None:
                raise ValueError('actual saved PRE RootContext required')
            if (self.context.streams.shape!=(1,)or self.context.frames.shape!=(1,)
                    or int(self.context.streams[0])!=self.address.stream or int(self.context.frames[0])!=self.address.frame):
                raise ValueError('saved PRE root address/geometry differs')
        elif self.context is not None or type(self.reason)is not str or not self.reason:
            raise ValueError('unavailable saved slot needs literal reason, no substituted state')


def actor(path,*,snapshot,plan):
    """Original strict archive loader; old contract metadata never waived."""
    if type(plan)is not runs.Plan or plan.model.arm!='D':raise ValueError('actual D Plan required')
    if type(snapshot)is not runs.Snapshot or snapshot.model_sha256!=runs._sha(plan.model.metadata()):
        raise ValueError('actual current-source D snapshot/model required')
    params,config,metadata=v5_model.load(path,expected_archive_sha256=snapshot.archive_sha256,
        expected_checkpoint_sha256=snapshot.checkpoint_sha256,expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256,enable_v5=True)
    if config!=plan.model:raise ValueError('actual D actor configuration differs')
    return params,metadata


def bank(provider,snapshot):
    """Genuine declared D-vs-D cost pair, not original training provenance."""
    if type(provider)is not teacher.Provider or type(snapshot)is not runs.Snapshot:
        raise ValueError('genuine saved-bank Provider and verified snapshot required')
    plan=provider.plan
    if snapshot.model_sha256!=runs._sha(plan.model.metadata()):raise ValueError('same D model required')
    bindings=rollout.Bindings(plan.model.category_vocabulary_sha256,plan.model.label_vocabulary,
        provider.prepared.source_sha256,(snapshot.archive_sha256,)*2,(True,True),runs._sha(plan.manifest()))
    assignments=tuple(dict(stream_id=game.stream_id,learner_side=ordinal%2,opponent=snapshot)
        for ordinal,game in enumerate(plan.games))
    import json
    recorded=json.loads(runs._json(assignments))
    return rollout.actor_bank_from_assignments(bindings,(snapshot.archive_sha256,),current_archive=snapshot.archive_sha256,
        assignments=recorded)


def _percentile(values):
    return None if not values else sorted(values)[math.ceil(.95*len(values))-1]


def summarize(records):
    """Observed attempt cost plus honest bounds on every fixed original slot."""
    if len(records)!=40:raise ValueError('all40 predeclared cost slots required')
    values=[];labeled=[];lower=[];upper=[]
    allowed=('LABELED','PROVIDER_REFUSED','UNREACHED','REFUSED','SNAPSHOT_REFUSED','INPUT_REFUSED','NOT_REQUESTED','PROVIDER_ERROR')
    for row in records:
        if row['status']not in allowed:raise ValueError('literal fixed cost disposition required')
        if row['status']in ('LABELED','PROVIDER_REFUSED'):
            value=row['label_seconds']
            if type(value)not in (int,float)or not math.isfinite(value)or value<0:raise ValueError('literal finite provider wall required')
            values.append(value);lower.append(value);upper.append(value)
            if row['status']=='LABELED':labeled.append(value)
        else:
            if row.get('label_seconds')is not None:raise ValueError('unpriced slot cannot claim completed label time')
            lower.append(0.);upper.append(math.inf)
    p95=_percentile(values);bound=_percentile(upper)
    return dict(fixed_slots=40,timed_provider_attempts=len(values),usable_labeled=len(labeled),unavailable_or_unpriced=40-len(values),
        p95_label_seconds=p95,p95_labeled_only_seconds=_percentile(labeled),
        warm_descriptive_p95_label_seconds=_percentile(values[1:]),cold_first_label_seconds=None if not values else values[0],
        warm_descriptive_scope='calls after first measured provider call; may include first-exercised branch JIT, not a pure-warm claim',
        full_roster_p95_lower_seconds=_percentile(lower),full_roster_p95_upper_seconds='unbounded'if math.isinf(bound)else bound,
        threshold_seconds=THRESHOLD_SECONDS,minimum20_timed=len(values)>=20,minimum20_usable_labels=len(labeled)>=20,
        cost_admission='INSUFFICIENT_TIMED_ROOTS'if len(values)<20 else 'INSUFFICIENT_USABLE_LABELS'if len(labeled)<20 else 'PASS'if p95<=THRESHOLD_SECONDS else 'FAIL',
        dispositions={status:sum(x['status']==status for x in records)for status in sorted(set(x['status']for x in records))},
        p95_definition='nearest rank ceil(.95*n); cold-first INCLUDED; provider refusals remain separately named timed paths',
        stop_rule='one fixed cost cohort; one repair only for invalid producer; no performance retries or replacement roots',
        scope='OFFLINE_SAVED_ROOT_LABEL_COST_ONLY_ZERO_NEW_GAMES_OR_OUTCOMES',scientific_strength_qualification=False)


def plan_from_bank(*,profile,labels,category,batches,pool_sha256,selection_seed,run_id,max_requests):
    """A genuine NONTRAINING cost Plan for the original saved-bank cells.

    This does not truncate/relabel a production D pool or invoke its roster.
    Batch construction/gender/action seeds stay exactly the bank declaration.
    """
    from pokezero import v5_training as training
    config=profile.configurations(labels,category_vocabulary=category,arm='D',replicate=0)
    return runs.Plan(run_id,config['model'],config['loss'],training.TrainConfig(**config['training']),
        (tuple(batches),),pool_sha256,selection_seed,config['historical_fraction'],config['historical_window'],
        max_requests,runs.Watchdog(1800),discount=config['discount'])


def _observer(provider):
    def observe(state,public):
        masks=transition.action_mask(state,provider.prepared.engine_tables)
        return tuple(encoder.encode(state,trackers.readout(public,j.asarray([owner],j.int32)),provider.prepared.observation_tables,
            observer_sides=j.asarray([owner],j.int32),legal_action_mask=masks[:,owner])for owner in (0,1))
    return jax.jit(observe)


def run(provider,snapshot,archive_path,index,*,enable_cost=False):
    """Actual once-only offline cohort; caller supplies genuine startup objects.

    The bank's canonical loader verifies primitive typed NPZ/source bindings.
    Both actual D-seat forwards are recomputed; bank-policy logits never become
    D logits. No transition runs except the unchanged Provider's offline worlds.
    """
    if enable_cost is not True or type(provider)is not teacher.Provider:
        raise ValueError('explicit genuine saved-bank D Provider required')
    if type(index)is not dict or index.get('schema')!='pokezero.device_heldout_saved_roots.v1':
        raise ValueError('original immutable saved-root index required')
    from . import v5_device_evaluation_bank as saved
    import json
    namespace=index['namespace'];binding_doc=json.loads(provider.prepared.source_binding_json)
    expected=dict(source_commit=binding_doc['source_commit'],source_sha256=provider.prepared.source_sha256,
        public_source_hash=provider.prepared.source.metadata.source_hash,pool_sha256=provider.plan.pool_sha256,
        pool_metadata_sha256=binding_doc['pool_metadata_sha256'],category_vocabulary_sha256=provider.plan.model.category_vocabulary_sha256,
        label_vocabulary_sha256=provider.plan.model.label_vocabulary.sha256)
    if namespace!=expected:raise ValueError('saved bank/actual Prepared source, pool and namespaces differ')
    declared=select(provider.plan);entries=index['cost_roots']
    if len(entries)!=40 or tuple(c.DecisionAddress(**x['selector'])for x in entries)!=declared:
        raise ValueError('exact40 predeclared Recipe-selected cost slots required')
    for entry in entries:
        if set(entry)!=set(('selector','status','artifact','reason'))or entry['status']not in ('SAVED','UNREACHED','REFUSED'):
            raise ValueError('literal saved-root entry fields/disposition required')
        if entry['status']=='SAVED':
            artifact=entry['artifact']
            if type(artifact)is not dict or set(artifact)!=set(('path','sha256'))or type(artifact['path'])is not str or not artifact['path']or entry['reason']is not None:
                raise ValueError('actual saved-root artifact required')
            c._binding(artifact['sha256'])
        elif entry['artifact']is not None or type(entry['reason'])is not str or not entry['reason']:
            raise ValueError('unavailable fixed slot requires literal reason, not substitute artifact')
    actor_clock=time.perf_counter();params,metadata=actor(archive_path,snapshot=snapshot,plan=provider.plan)
    jax.block_until_ready(params);actor_seconds=time.perf_counter()-actor_clock
    full_bank=bank(provider,snapshot);row_lookup={stream:row for row,stream in enumerate(full_bank.streams)}
    from dataclasses import replace
    observe=_observer(provider)
    forward=jax.jit(lambda p,inputs:v5_model.forward(p,inputs,provider.plan.model,compute_dtype='bfloat16'))
    records=[];integrity=True
    for ordinal,(address,entry)in enumerate(zip(declared,entries)):
        record=dict(ordinal=ordinal,address=dict(stream=address.stream,frame=address.frame,owner=address.owner),
            status='UNREACHED',reason=entry.get('reason'),label_seconds=None,root_input_seconds=None,
            provider_record=None,cold_first_label_call=False)
        records.append(record)
        if entry['status']!='SAVED':
            if entry['status']not in ('UNREACHED','REFUSED'):raise ValueError('literal saved-root disposition required')
            record['status']=entry['status'];continue
        try:
            context,_source_fields=saved.load_root(entry,index)
            slot=Slot(address,'AVAILABLE',context)
        except (ValueError,TypeError,KeyError,OSError)as error:
            record.update(status='SNAPSHOT_REFUSED',reason=type(error).__name__+': '+str(error));integrity=False;continue
        input_clock=time.perf_counter()
        try:
            # Same compiled observer/actor algebra and BOTH actual seat outputs.
            observations=observe(context.state,context.public)
            outputs=tuple(jax.block_until_ready(forward(params,encoder.actor_inputs(o)))for o in observations)
            requested=np.asarray(request_seats(context.state))
            legal=np.asarray(j.stack([o.legal_action_mask for o in observations],axis=1))
            errors=np.asarray(j.stack([o.error for o in observations],axis=1))
            finite=np.asarray(j.stack([o.valid&j.isfinite(o.value)&(j.abs(o.value)<=1)&j.all(j.isfinite(o.policy_logits),axis=-1)
                &(j.all(j.isfinite(o.value_logits),axis=-1)if o.value_logits is not None else True)for o in outputs],axis=1))
            valid=requested&finite&(errors==0)&legal.any(axis=-1)
            row=row_lookup[address.stream]
            root_bank=replace(full_bank,streams=(address.stream,),actor_indices=(full_bank.actor_indices[row],),learner_mask=(full_bank.learner_mask[row],))
            fields=dict((name,None)for name in rollout.Frames._fields)
            for name in ('numeric_features','categorical_ids','token_type_ids','attention_mask'):
                fields[name]=np.asarray(jax.block_until_ready(j.stack([getattr(o,name)for o in observations],axis=1)))[None]
            fields.update(legal_action_mask=legal[None],streams=np.asarray(context.streams)[None],frames=np.asarray(context.frames)[None],
                turns=np.asarray(context.state.turn)[None],requested=requested[None],policy_valid=valid[None],
                trainable=(valid&np.asarray(root_bank.learner_mask))[None],errors=errors[None],actor_finite=finite[None],
                values=np.asarray(j.stack([o.value for o in outputs],axis=1))[None],
                raw_policy_logits=np.asarray(j.stack([o.policy_logits for o in outputs],axis=1))[None],
                actor_indices=np.asarray(root_bank.actor_indices,np.int32)[None],learner_mask=np.asarray(root_bank.learner_mask,bool)[None])
            frames=rollout.Frames(**fields)
            logits=frames.raw_policy_logits[0,0,address.owner]
            record.update(root_logits=[float(x)if np.isfinite(x)else None for x in logits],
                root_logits_status='FINITE'if np.isfinite(logits).all()else'NONFINITE',
                root_legal=list(map(bool,legal[0,address.owner])),root_errors=list(map(int,errors[0])),
                root_actor_finite=list(map(bool,finite[0])))
            record['root_input_seconds']=time.perf_counter()-input_clock
            if not requested[0,address.owner]:
                record.update(status='NOT_REQUESTED',reason='selected owner has no actual PRE request');continue
        except (ValueError,RuntimeError,TypeError,KeyError,IndexError)as error:
            record.update(status='INPUT_REFUSED',reason=type(error).__name__+': '+str(error));integrity=False;continue
        cold=not any(x['label_seconds']is not None for x in records)
        clock=time.perf_counter()
        try:
            roots=jax.tree.map(lambda x:x[None],slot.context)
            labels=provider.label_chunk(roots,frames,root_bank,(params,))
            elapsed=time.perf_counter()-clock
            if len(labels)!=1 or labels[0]['address']!=record['address']:raise ValueError('one exact selected provider record required')
            label=labels[0]
            if label['status']not in ('LABELED','REFUSED'):raise ValueError('selected original Provider must label or refuse')
            record.update(status='LABELED'if label['status']=='LABELED'else'PROVIDER_REFUSED',reason=label['refusal'],
                provider_record=label,label_seconds=elapsed,cold_first_label_call=cold)
        except (ValueError,RuntimeError,TypeError,KeyError,IndexError)as error:
            record.update(status='PROVIDER_ERROR',reason=type(error).__name__+': '+str(error));integrity=False
    summary=summarize(records)
    if not integrity:summary['cost_admission']='INVALID_INPUT_OR_PROVIDER'
    return dict(schema=SCHEMA,summary=summary,records=records,namespace=expected,
        declaration_sha256=runs._sha(provider.plan.manifest()),snapshot=dict(checkpoint_sha256=snapshot.checkpoint_sha256,
            archive_sha256=snapshot.archive_sha256,model_sha256=snapshot.model_sha256,parameter_count=metadata['parameter_count']),
        startup_actor_load_blocked_seconds=actor_seconds,
        label_timing_scope='Provider.label_chunk entire synchronized public-world price, source sampling/materialization/reply/leaf, including first JIT cold; root re-encode/forward measured separately',
        preparation_scope='genuine Provider static preparation precedes run; no actual models/worlds/games executed during source preparation',
        cost_pair='DECLARED_D0_VS_D0_COST_PAIR_NOT_A_CLAIM_ABOUT_ORIGINAL_BANK_ACTING_POLICY',
        producer_integrity=integrity)
