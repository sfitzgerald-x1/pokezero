"""Opt-in chunked device rollout -> original v5 TrainingBatch/cache.

No Session, captured FULL, event tape, host fold, or per-step host transfer.
The caller owns a fixed declaration, actual constructor and two original actor
archives. One parameter tree per seat is shared across this cohort, never
broadcast into one 9M-weight copy per game. The declared B/D historical pool
assignment planner is NOT replaced here. D's actual teacher consumer is absent
and refuses explicitly; its root public-carry handoff is available separately.
"""
from dataclasses import asdict, dataclass
from typing import NamedTuple
import hashlib
import json

import jax
import jax.numpy as j
import numpy as np
from pokezero import v5_contract as contract
from pokezero.v5_model import V5ModelConfig
from . import transition, engine_device_trackers as trackers
from . import engine_device_observation as encoder, v5_model
from . import benchmark_device_selfplay as sampling

RUNNING, COMPLETE, CAPPED, REFUSED = range(4)
STATUS=('IN_PROGRESS','COMPLETE','CAPPED','REFUSED')
SCHEMA='pokezero.device_rollout.v5.1'


@dataclass(frozen=True)
class Bindings:
    category_sha256:str
    label_vocabulary:object
    source_sha256:str
    actor_archives:tuple
    learner_sides:tuple
    declaration_sha256:str
    def __post_init__(self):
        object.__setattr__(self,'actor_archives',tuple(self.actor_archives))
        object.__setattr__(self,'learner_sides',tuple(self.learner_sides))
        if type(self.label_vocabulary) is not contract.LabelVocabulary:raise ValueError('original full label vocabulary required')
        for value in (self.category_sha256,self.source_sha256,self.declaration_sha256,*self.actor_archives):contract._binding(value)
        if len(self.actor_archives)!=2 or len(self.learner_sides)!=2 or any(type(x)is not bool for x in self.learner_sides) or not any(self.learner_sides):
            raise ValueError('two actual actor archives and explicit learner seat assignment required')

class Labels(NamedTuple):
    move:object # engine move ID -> full original semantic class, -1 unknown
    species:object # engine species ID -> full original semantic species, -1

class ConstructorTruth(NamedTuple):
    probabilities:object # [B,2,6,64] source-matched original constructor rows
    species:object       # [B,2,6] zero-based label species
    known:object         # [B,2,6], missing source facts remain masked

class Carry(NamedTuple):
    state:object
    public:object
    streams:object
    frames:object
    status:object
    winner:object
    terminal_frame:object
    terminal_hp:object
    terminal_max_hp:object
    truth:object

class Frames(NamedTuple):
    numeric_features:object
    categorical_ids:object
    token_type_ids:object
    attention_mask:object
    legal_action_mask:object
    streams:object
    frames:object
    turns:object
    requested:object
    trainable:object
    policy_valid:object
    errors:object
    actions:object
    engine_actions:object
    probabilities:object
    values:object
    actor_finite:object
    advanced:object
    physical:object
    semantic_targets:object
    semantic_mask:object
    hidden_targets:object
    hidden_species:object
    hidden_mask:object

class Rollout(NamedTuple):
    carry:object
    frames:object

class RootContext(NamedTuple):
    state:object
    public:object
    streams:object
    frames:object


def compile_labels(metadata,vocabulary):
    if type(vocabulary)is not contract.LabelVocabulary:raise ValueError('full label namespace required')
    moves={name:i for i,name in enumerate(vocabulary.moves)}
    species={name:i for i,name in enumerate(vocabulary.species)}
    move=lambda name:'hiddenpower' if name.startswith('hiddenpower') else name
    return Labels(j.asarray([moves.get(move(n),-1) for n in metadata['vocabulary']['moves']],j.int32),
        j.asarray([species.get(n,-1) for n in metadata['vocabulary']['species']],j.int32))


def initialize(initialized,streams,*,truth=None):
    """Use the SAME actual initialize_with_public result, never re-poll a world.

    This is a batch-boundary constructor, not a per-step host check. Every
    supplied declared stream is attempted once; no refill/retry path exists.
    """
    state,public=initialized;ids=np.asarray(streams);b=state.hp.shape[0]
    if ids.shape!=(b,) or ids.dtype.kind not in 'iu' or np.any(ids<=0) or np.any(ids>=2**31) or len(set(map(int,ids)))!=b:
        raise ValueError('fixed unique original declared stream IDs required')
    if public.mon_features is None:raise ValueError('actual public-hook initialization required')
    if np.asarray(state.done).any():raise ValueError('fresh nonterminal declared constructor required')
    if truth is None:truth=ConstructorTruth(j.zeros((b,2,6,64),j.float32),j.full((b,2,6),-1,j.int32),j.zeros((b,2,6),bool))
    for value,shape,dtype in zip(truth,((b,2,6,64),(b,2,6),(b,2,6)),(j.float32,j.int32,j.bool_)):
        if value.shape!=shape or value.dtype!=dtype:raise ValueError('typed original constructor truth geometry required')
    z=j.zeros(b,j.int32)
    return Carry(state,public,j.asarray(ids,j.int32),z,z,j.full(b,-1,j.int32),z,
        j.zeros_like(state.hp),j.zeros_like(state.max_hp),truth)


def _physical(public):
    b=len(public.error);slots=j.arange(6)
    match=public.physical_to_public[...,None]==slots[None,None,None,:]
    inverse=j.where(j.any(match,axis=2),j.argmax(match,axis=2),-1).astype(j.int32)
    return j.stack([j.concatenate((public.own_order[:,s],inverse[:,1-s]),axis=1) for s in range(2)],axis=1)


def make_collector(engine_tables,observation_tables,model_config,labels,bindings,*,
        sampling_seed,chunk_steps=8,max_requests=4096):
    """Prepare once; returned JIT accepts (carry, (seat0params,seat1params)).

    No weight bank is broadcast per game. The caller must group exactly the
    authoritative declared actor assignments; this initial entry does not
    implement the B/D .5 historical/32-snapshot planner. D never falls back to
    the old captured-FULL provider.
    """
    if type(model_config)is not V5ModelConfig or type(bindings)is not Bindings:raise ValueError('explicit model/config bindings required')
    if model_config.arm=='D':raise ValueError('device D teacher consumer not integrated; no old FULL fallback')
    if model_config.category_vocabulary_sha256!=bindings.category_sha256 or model_config.label_vocabulary.sha256!=bindings.label_vocabulary.sha256:
        raise ValueError('same actual actor category/full-label namespace required')
    if any(type(x)is not int or x<=0 for x in (chunk_steps,max_requests)) or type(sampling_seed)is not int or not 0<=sampling_seed<2**32:
        raise ValueError('finite chunk/request bounds and original sampling seed required')
    # Static class offset without attaching Python metadata to device labels.
    move_classes=len(bindings.label_vocabulary.moves)
    def advance(c,params):
        state=c.state;b=len(c.streams);r=j.arange(b)[:,None];sides=j.arange(2)[None,:]
        live=c.status==RUNNING;state=state._replace(done=state.done|~live)
        requested=sampling.request_seats(state)&live[:,None]
        physical=_physical(c.public)
        masks=transition.action_mask(state,engine_tables)
        observations=[encoder.encode(state,trackers.readout(c.public,j.full(b,s,j.int32)),observation_tables,
            observer_sides=j.full(b,s,j.int32),legal_action_mask=masks[:,s]) for s in range(2)]
        outputs=[v5_model.forward(params[s],encoder.actor_inputs(observations[s]),model_config,compute_dtype='bfloat16') for s in range(2)]
        raw_logits=j.stack([o.policy_logits for o in outputs],axis=1);values=j.stack([o.value for o in outputs],axis=1)
        finite=j.stack([o.valid&j.isfinite(o.value)&(j.abs(o.value)<=1)&j.all(j.isfinite(o.policy_logits),axis=-1)
            & (j.all(j.isfinite(o.value_logits),axis=-1) if o.value_logits is not None else True) for o in outputs],axis=1)
        legal=j.stack([o.legal_action_mask for o in observations],axis=1)
        safe_logits=j.where(legal,j.nan_to_num(raw_logits,nan=0.,posinf=0.,neginf=0.),-j.inf)
        safe_logits=j.where(j.any(legal,axis=-1)[...,None],safe_logits,0.)
        keys=sampling.sample_keys(c.streams,c.frames,sampling_seed=sampling_seed)
        actions=jax.vmap(lambda key,logits:jax.random.categorical(key,logits))(
            keys.reshape(-1,2),safe_logits.reshape(-1,9)).reshape(b,2).astype(j.int32)
        probabilities=j.take_along_axis(jax.nn.softmax(safe_logits,axis=-1),actions[...,None],axis=-1)[...,0]
        engine_actions=j.stack([encoder.map_actions(observations[s],state,j.full(b,s,j.int32),actions[:,s]) for s in range(2)],axis=1)
        errors=j.stack([o.error for o in observations],axis=1)
        input_valid=finite&(errors==0)&j.any(legal,axis=-1)&j.isfinite(probabilities)&(probabilities>0)
        input_refused=live&j.any(requested&~input_valid,axis=1)
        # A training collector does not label fallback behavior as the named
        # raw actor. Bad input/actor ends that declared cell without retry.
        step_state=state._replace(done=state.done|input_refused)
        out=transition.step_with_public(step_state,engine_actions,engine_tables,c.public,belief_tables=observation_tables.belief)
        after=out.step.state;refused=input_refused|(live&out.step.invalid_actions)
        completed=live&~refused&after.done
        frame=c.frames+live.astype(j.int32)
        capped=live&~refused&~completed&(frame>=max_requests)
        ended=completed|capped|refused
        status=j.where(refused,REFUSED,j.where(completed,COMPLETE,j.where(capped,CAPPED,c.status)))
        carry=c._replace(state=after,public=out.trackers,frames=frame,status=status,
            winner=j.where(completed,after.winner,c.winner),terminal_frame=j.where(ended,frame,c.terminal_frame),
            terminal_hp=j.where(completed[:,None,None],after.hp,c.terminal_hp),
            terminal_max_hp=j.where(completed[:,None,None],after.max_hp,c.terminal_max_hp))
        # These training-only values are computed AFTER actor inference. None
        # of ConstructorTruth, peer choices, or terminal data enter encode.
        slot=j.clip(engine_actions,0,3);move=state.moves[r,sides,state.active,slot]
        from .state import usable_moves,bench_indices
        charge=state.charge_move[r,sides,state.active]
        # Hard charging requests name the actual current child, which may be
        # absent from installed slots (Sleep Talk) and have no remaining PP.
        move=j.where(charge>0,charge,j.where(j.any(usable_moves(state),axis=-1),move,engine_tables.struggle))
        move_label=labels.move[j.clip(move,0,len(labels.move)-1)]
        targets=j.take_along_axis(bench_indices(state.active),j.clip(engine_actions-4,0,4)[...,None],axis=-1)[...,0]
        species_label=labels.species[j.clip(state.species[r,sides,targets],0,len(labels.species)-1)]
        semantic=j.where(engine_actions<4,move_label,j.where(species_label>=0,move_classes+species_label,-1))
        opponent_targets=semantic[:,::-1]
        opponent_mask=requested&requested[:,::-1]&(opponent_targets>=0)&~refused[:,None]
        foe=1-sides;positions=j.clip(physical[:,:,6:],0,5)
        hidden=c.truth.probabilities[r[:,:,None],foe[:,:,None],positions]
        hs=c.truth.species[r[:,:,None],foe[:,:,None],positions]
        hm=c.truth.known[r[:,:,None],foe[:,:,None],positions]&(physical[:,:,6:]>=0)&requested[:,:,None]&~refused[:,None,None]
        valid=requested&input_valid&~refused[:,None]
        trainable=valid&j.asarray(bindings.learner_sides)[None,:]
        stack=lambda name:j.stack([getattr(o,name) for o in observations],axis=1)
        record=Frames(stack('numeric_features'),stack('categorical_ids'),stack('token_type_ids'),stack('attention_mask'),
            legal,c.streams,c.frames,state.turn,requested,trainable,valid,errors,actions,engine_actions,probabilities,values,finite,live&~refused,physical,
            opponent_targets,opponent_mask,hidden,hs,hm)
        return carry,record
    def collect(c,params):
        if len(params)!=2:raise ValueError('two explicit original seat parameter trees required')
        carry,frames=jax.lax.scan(lambda c,_:advance(c,params),c,None,length=chunk_steps)
        return Rollout(carry,frames)
    return jax.jit(collect)


def public_root_context(carry):
    """Actual root public handoff; no worlds/private opponent reconstruction."""
    return RootContext(carry.state,carry.public,carry.streams,carry.frames)


def leaf_from_roots(context,world_state,root_rows,tables,*,observer_sides,legal_action_mask):
    from . import device_public_leaf
    if type(context)is not RootContext:raise ValueError('actual rollout root context required')
    return device_public_leaf.from_roots(world_state,context.public,root_rows,tables,
        observer_sides=observer_sides,legal_action_mask=legal_action_mask,enable_device_leaf=True)


def boundary(rollout):
    """The ONLY device->host transfer in this collector; retain raw invalid rows."""
    return jax.device_get(rollout)


def constructor_truth(raw_state,pool,team_indices,*,source,vocabulary):
    """Match ORIGINAL pre-entry pool rows once, never later private-state guesses."""
    from . import v5_collection_data as original
    indices=np.asarray(team_indices);b=raw_state.hp.shape[0];meta=pool.metadata();v=meta['vocabulary']
    if indices.shape!=(b,2) or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(pool.arrays['species'])):
        raise ValueError('same actual constructor team indices required')
    if contract.source_variant_census(source)['source_rows_sha256']!=vocabulary.source_binding_sha256:
        raise ValueError('same complete original source label axis required')
    for actual,key in (('species','species'),('level','level'),('moves','moves'),('base_ability','ability'),('item','item'),('hidden_power_type','hidden_power_type')):
        if not np.array_equal(np.asarray(getattr(raw_state,actual)),pool.arrays[key][indices]):
            raise ValueError('pre-entry original constructor differs: '+actual)
    targets=np.zeros((b,2,6,64),np.float32);species=np.full((b,2,6),-1,np.int32);known=np.zeros((b,2,6),bool)
    for row in range(b):
        for side in (0,1):
            idx=indices[row,side]
            for p in range(6):
                get=lambda key:int(pool.arrays[key][idx,p])
                label=original.matching_hidden_set(v['species'][get('species')],level=get('level'),
                    moves=tuple(v['moves'][int(m)] for m in pool.arrays['moves'][idx,p] if m),
                    ability=v['abilities'][get('ability')],item=v['items'][get('item')],
                    hidden_power_type=v['types'][get('hidden_power_type')],source=source)
                if label is not None:
                    targets[row,side,p]=label.probabilities(vocabulary)
                    species[row,side,p]=vocabulary.species.index(label.species);known[row,side,p]=True
    return ConstructorTruth(j.asarray(targets),j.asarray(species),j.asarray(known))


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()



def _complete_gae(values,outcome,loss_config,discount):
    """Original trainer equations, lazy Torch dependency at the batch boundary."""
    import torch
    from pokezero.v5_loss import gae_targets
    v=torch.tensor(values,dtype=torch.float32);reward=torch.zeros_like(v);reward[-1]=outcome
    next_v=torch.cat((v[1:],torch.zeros(1)));terminated=torch.zeros(len(values),dtype=torch.bool);terminated[-1]=True
    a,target=gae_targets(reward,v,next_v,terminated,~terminated,discount=discount,
        policy_lambda=loss_config.policy_lambda,value_lambda=loss_config.value_lambda)
    return a.numpy(),target.numpy()

def training_batch(host_rollouts,bindings,*,battle_ids,seeds,loss_config,discount=1.):
    """Batch-boundary adapter into the ORIGINAL validated v5 cache/trainer API.

    Retains actual all29 inputs. Terminal unknowns never manufacture PPO,
    outcome or end-HP targets. Complete trajectories use original gae_targets;
    any missing/invalid learner decision disables PPO for that owner/stream.
    Raw requested/error rows remain in host_rollouts even when not trainable.
    This entry does not substitute for the configured historical assignment
    planner and does not produce D/private-teacher labels.
    """
    from types import MappingProxyType
    from pokezero.dataset import TrainingBatch
    from pokezero import v5_data
    from pokezero.v5_loss import V5LossConfig
    if type(bindings)is not Bindings or type(loss_config)is not V5LossConfig or loss_config.arm=='D':
        raise ValueError('bound actual A/B rollout and original loss recipe required; D consumer absent')
    if isinstance(discount,bool) or not isinstance(discount,(int,float)) or not np.isfinite(discount) or not 0<=discount<=1:
        raise ValueError('finite declared discount required')
    rolls=tuple(host_rollouts)
    if not rolls or any(isinstance(x,jax.Array) for x in jax.tree.leaves(rolls)):
        raise ValueError('explicit completed batch-boundary device_get required')
    carry=rolls[-1].carry;b=len(carry.streams)
    if len(battle_ids)!=b or len(seeds)!=b or len(set(battle_ids))!=b:raise ValueError('original declared battle/seed rows required')
    if any(not np.array_equal(x.carry.streams,carry.streams) for x in rolls):raise ValueError('same fixed cohort, no refill required')
    f=jax.tree.map(lambda *xs:np.concatenate(xs,axis=0),*(x.frames for x in rolls))
    addresses=np.argwhere(f.trainable)
    if not len(addresses):raise ValueError('no valid actual learner decisions; retain raw rollout and dispositions')
    n=len(addresses);take=lambda name:np.asarray(getattr(f,name))[tuple(addresses.T)]
    # Per-game fields do not contain an owner axis.
    step=addresses[:,0];rows=addresses[:,1];owners=addresses[:,2]
    streams=f.streams[step,rows];frames=f.frames[step,rows];turns=f.turns[step,rows]
    cats=take('categorical_ids');numbers=take('numeric_features');roles=take('token_type_ids');attention=take('attention_mask')
    legal=take('legal_action_mask');actions=take('actions');engine_actions=take('engine_actions')
    prob=take('probabilities');values=take('values');physical=take('physical')
    keys=[(int(s),int(frame),int(owner)) for s,frame,owner in zip(streams,frames,owners)]
    if len(set(keys))!=n:raise ValueError('duplicate actual decision address across chunks')
    if np.any(carry.status<0) or np.any(carry.status>=len(STATUS)):raise ValueError('literal disposition outside enum')
    statuses=tuple(STATUS[int(carry.status[row])] for row in rows)
    if any(s not in STATUS for s in statuses):raise ValueError('literal disposition required')
    source_hashes=[];terminal_hashes=[];metadata=[]
    outcome=np.zeros(n,np.float32);om=np.zeros(n,bool);end_hp=np.zeros((n,12),np.float32);end_mask=np.zeros((n,12),bool)
    for i,(row,owner) in enumerate(zip(rows,owners)):
        body={'schema':SCHEMA,'source_sha256':bindings.source_sha256,'declaration_sha256':bindings.declaration_sha256,
            'actor_archives':bindings.actor_archives,'stream':int(streams[i]),'frame':int(frames[i]),'owner':int(owner),
            'numeric':numbers[i].tolist(),'categorical':cats[i].tolist(),'roles':roles[i].tolist(),'attention':attention[i].tolist(),
            'legal':legal[i].tolist(),'action':int(actions[i]),'engine_action':int(engine_actions[i]),
            'probability':float(prob[i]),'value':float(values[i]),'physical':physical[i].tolist()}
        source_hashes.append(_sha(body))
        meta={'device_rollout_schema':SCHEMA,'source_frame_sha256':source_hashes[-1],'v5_return_known':statuses[i]=='COMPLETE',
            'actor_archive_sha256':bindings.actor_archives[owner],'opponent_archive_sha256':bindings.actor_archives[1-owner]}
        metadata.append(meta)
        terminal_hashes.append(None if statuses[i]=='IN_PROGRESS' else _sha({'stream':int(streams[i]),'frame':int(carry.terminal_frame[row]),
            'status':statuses[i],'winner':int(carry.winner[row]) if statuses[i]=='COMPLETE' else None,
            'hp':carry.terminal_hp[row].tolist() if statuses[i]=='COMPLETE' else [],
            'max_hp':carry.terminal_max_hp[row].tolist() if statuses[i]=='COMPLETE' else []}))
        if statuses[i]=='COMPLETE':
            winner=int(carry.winner[row])
            if winner not in (0,1,2):raise ValueError('actual terminal winner required')
            hp=carry.terminal_hp[row];maximum=carry.terminal_max_hp[row]
            if np.any(maximum<=0) or np.any(hp<0) or np.any(hp>maximum):raise ValueError('literal terminal HP geometry required')
            outcome[i]=0 if winner==2 else 1 if winner==owner else -1;om[i]=True
            for token,p in enumerate(physical[i]):
                if p>=0:
                    side=owner if token<6 else 1-owner
                    end_hp[i,token]=hp[side,p]/maximum[side,p];end_mask[i,token]=True
    advantage=np.zeros(n,np.float32);value_target=np.zeros(n,np.float32);ppo=np.zeros(n,bool);rewards=np.zeros(n,np.float32)
    groups={}
    for i,key in enumerate(keys):groups.setdefault((key[0],key[2]),[]).append(i)
    for (stream,owner),indices in groups.items():
        indices=sorted(indices,key=lambda i:int(frames[i]));row=int(rows[indices[0]])
        full=f.requested[:,row,owner]&np.asarray(bindings.learner_sides[owner])
        good=f.trainable[:,row,owner]
        if statuses[indices[0]]!='COMPLETE' or np.any(full&~good):continue
        a,target=_complete_gae(values[indices],float(outcome[indices[-1]]),loss_config,discount)
        advantage[indices]=a;value_target[indices]=target;ppo[indices]=True;rewards[indices[-1]]=outcome[indices[-1]]
    empty=np.zeros(n,bool);zero=np.zeros(n,np.float32)
    sidecar=dict(schema=contract.TRAINING_CONTRACT_SCHEMA,observation_schema=contract.OBSERVATION_SCHEMA_VERSION_V5,
        category_vocabulary_sha256=bindings.category_sha256,label_vocabulary_sha256=bindings.label_vocabulary.sha256,
        source_binding_sha256=bindings.label_vocabulary.source_binding_sha256,label_vocabulary=asdict(bindings.label_vocabulary),
        addresses=tuple(dict(stream=s,frame=frame,owner=owner) for s,frame,owner in keys),source_frame_sha256=tuple(source_hashes),
        terminal_source_sha256=tuple(terminal_hashes),terminal_status=statuses,search_status=('UNSELECTED',)*n,
        search_refusal=(None,)*n,root_improvement_records=(None,)*n,
        opponent_semantic_action_targets=take('semantic_targets'),opponent_semantic_action_target_mask=take('semantic_mask'),
        hidden_set_targets=take('hidden_targets'),hidden_set_species=take('hidden_species'),hidden_set_target_mask=take('hidden_mask'),
        survival_targets=(end_hp>0).astype(np.float32),final_hp_targets=end_hp,pokemon_end_target_mask=end_mask,
        outcome_targets=outcome,outcome_target_mask=om,privileged_view_mask=empty,
        privileged_categorical_ids=np.zeros((n,1,29,41),np.int32),privileged_numeric_features=np.zeros((n,1,29,132),np.float32),
        privileged_token_type_ids=np.zeros((n,1,29),np.int32),privileged_attention_mask=np.zeros((n,1,29),bool))
    batch=TrainingBatch(categorical_ids=cats[:,None],numeric_features=numbers[:,None],token_type_ids=roles[:,None],
        attention_mask=attention[:,None],history_mask=np.ones((n,1),bool),legal_action_mask=legal,action_indices=actions,
        rewards=rewards,returns=outcome,value_estimates=values,value_estimate_mask=np.ones(n,bool),
        ppo_advantages=advantage,ppo_advantage_mask=ppo,ppo_value_targets=value_target,ppo_value_target_mask=ppo,
        opponent_action_indices=np.full(n,-1,np.int32),opponent_action_mask=empty,action_probabilities=prob,
        action_probability_mask=np.ones(n,bool),training_weights=np.ones(n,np.float32),
        battle_ids=tuple(battle_ids[row] for row in rows),seeds=tuple(int(seeds[row]) for row in rows),format_ids=('gen3randombattle',)*n,
        player_ids=tuple(('p1','p2')[owner] for owner in owners),turn_indices=tuple(map(int,turns)),
        terminal_capped=tuple(s=='CAPPED' for s in statuses),step_metadata=tuple(metadata),
        search_policy_targets=np.zeros((n,9),np.float32),search_policy_target_mask=empty,
        search_value_targets=zero,search_value_target_mask=empty,v5_columns=MappingProxyType(sidecar))
    v5_data.validate_batch(batch)
    return batch


def write_cache(path,host_rollouts,bindings,**kwargs):
    """Existing exclusive v5 writer, called only after the explicit boundary."""
    from pokezero import v5_data
    return v5_data.write_cache(path,training_batch(host_rollouts,bindings,**kwargs))
