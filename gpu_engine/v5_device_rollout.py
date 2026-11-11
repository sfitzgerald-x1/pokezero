"""Opt-in chunked device rollout -> original v5 TrainingBatch/cache.

No Session, captured FULL, event tape, host fold, or per-step host transfer.
The caller owns a fixed declaration, actual constructor and original actor
archives. The default shares one parameter tree per seat across the cohort.
An explicit immutable actor bank gathers the original per-game assignments
without broadcasting a 9M-weight copy per game. The declared B/D historical
pool planner is not replaced. D requires an explicit original-recipe teacher;
its PRE public-carry handoff stays on device until selected at a chunk boundary.
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

@dataclass(frozen=True)
class ActorBank:
    """Immutable, already-decided actor assignment; never a new pool planner.

    Parameter trees are supplied once per archive to the prepared collector.
    Only row indices, not weights, are repeated for games sharing an actor.
    """
    bindings:Bindings
    actor_archives:tuple
    streams:tuple
    actor_indices:tuple
    learner_mask:tuple
    assignment_sha256:str
    def __post_init__(self):
        for name in ('actor_archives','streams'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        for name in ('actor_indices','learner_mask'):
            object.__setattr__(self,name,tuple(tuple(row) for row in getattr(self,name)))
        if type(self.bindings)is not Bindings:raise ValueError('original rollout bindings required')
        contract._binding(self.assignment_sha256)
        for digest in self.actor_archives:contract._binding(digest)
        if not self.actor_archives or len(set(self.actor_archives))!=len(self.actor_archives):
            raise ValueError('unique original actor archive bank required')
        b=len(self.streams)
        if not b or len(set(self.streams))!=b or any(type(x)is not int or not 0<x<2**31 for x in self.streams):
            raise ValueError('ordered unique declared streams required')
        if len(self.actor_indices)!=b or any(len(row)!=2 or any(type(x)is not int or not 0<=x<len(self.actor_archives) for x in row) for row in self.actor_indices):
            raise ValueError('exact per-game/per-seat actor indices required')
        if len(self.learner_mask)!=b or any(len(row)!=2 or any(type(x)is not bool for x in row) for row in self.learner_mask) or not any(any(row) for row in self.learner_mask):
            raise ValueError('explicit per-game/per-seat learner mask required')


def actor_bank_from_assignments(bindings,actor_archives,*,current_archive,assignments):
    """Consume the ORIGINAL planner output without sampling/reordering it.

    The historical probability and 32-snapshot retention remain decisions of
    that unchanged planner. This adapter uses its exact stream/learner_side/
    opponent archive fields, including a current-vs-current chosen opponent.
    """
    archives=tuple(actor_archives);lookup={digest:i for i,digest in enumerate(archives)}
    if current_archive not in lookup:raise ValueError('declared current actor missing from bank')
    rows=tuple(assignments);indices=[];learners=[];streams=[]
    for row in rows:
        side=row['learner_side'];opponent=row['opponent']['archive_sha256']
        if type(side)is not int or side not in (0,1) or opponent not in lookup:
            raise ValueError('original learner seat/opponent archive missing from bank')
        pair=[lookup[opponent],lookup[opponent]];pair[side]=lookup[current_archive]
        indices.append(tuple(pair));learners.append((side==0,side==1));streams.append(row['stream_id'])
    return ActorBank(bindings,archives,tuple(streams),tuple(indices),tuple(learners),_sha(rows))

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
    actor_indices:object=None # bank only; [B,2], same original assignment every frame
    learner_mask:object=None  # bank only; [B,2], historical seats never become PPO rows
    raw_policy_logits:object=None # D only: original PRE actor logits, never resampled

class Rollout(NamedTuple):
    carry:object
    frames:object
    roots:object=None # D only, one PRE RootContext per scan step; never bulk host copied

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
        sampling_seed,chunk_steps=8,max_requests=4096,actor_bank=None,retain_teacher_roots=False):
    """Prepare once; returned JIT accepts (carry, tuple(original_params)).

    Default: two original seat trees. With actor_bank: one tree per declared
    archive, grouped by its predeclared row indices before the device loop.
    No weights are broadcast per game. The original B/D .5 historical/
    32-snapshot planner remains authoritative. D never falls back to FULL.
    """
    if type(model_config)is not V5ModelConfig or type(bindings)is not Bindings:raise ValueError('explicit model/config bindings required')
    if type(retain_teacher_roots)is not bool or retain_teacher_roots!=(model_config.arm=='D'):
        raise ValueError('device D teacher consumer requires explicit PRE roots; no old FULL fallback')
    if retain_teacher_roots and actor_bank is None:raise ValueError('D requires the exact original actor bank')
    if model_config.category_vocabulary_sha256!=bindings.category_sha256 or model_config.label_vocabulary.sha256!=bindings.label_vocabulary.sha256:
        raise ValueError('same actual actor category/full-label namespace required')
    if any(type(x)is not int or x<=0 for x in (chunk_steps,max_requests)) or type(sampling_seed)is not int or not 0<=sampling_seed<2**32:
        raise ValueError('finite chunk/request bounds and original sampling seed required')
    if actor_bank is not None and (type(actor_bank)is not ActorBank or actor_bank.bindings!=bindings):
        raise ValueError('same immutable actor bank/bindings required')
    # Known indices are grouped ONCE before the device loop. Every real
    # (row,seat) appears in exactly one group. No padding or per-game weights.
    groups=() if actor_bank is None else tuple((actor,tuple(2*row+side for row,pair in enumerate(actor_bank.actor_indices)
        for side in (0,1) if pair[side]==actor)) for actor in range(len(actor_bank.actor_archives)))
    groups=tuple((actor,indices) for actor,indices in groups if indices)
    # Static class offset without attaching Python metadata to device labels.
    move_classes=len(bindings.label_vocabulary.moves)
    def advance(c,params):
        state=c.state;b=len(c.streams);r=j.arange(b)[:,None];sides=j.arange(2)[None,:]
        if actor_bank is not None and b!=len(actor_bank.streams):raise ValueError('same fixed declared actor-bank cohort required')
        live=c.status==RUNNING;state=state._replace(done=state.done|~live)
        requested=sampling.request_seats(state)&live[:,None]
        physical=_physical(c.public)
        masks=transition.action_mask(state,engine_tables)
        observations=[encoder.encode(state,trackers.readout(c.public,j.full(b,s,j.int32)),observation_tables,
            observer_sides=j.full(b,s,j.int32),legal_action_mask=masks[:,s]) for s in range(2)]
        def output_finite(o):
            return o.valid&j.isfinite(o.value)&(j.abs(o.value)<=1)&j.all(j.isfinite(o.policy_logits),axis=-1)\
                &(j.all(j.isfinite(o.value_logits),axis=-1) if o.value_logits is not None else True)
        if actor_bank is None:
            outputs=[v5_model.forward(params[s],encoder.actor_inputs(observations[s]),model_config,compute_dtype='bfloat16') for s in range(2)]
            raw_logits=j.stack([o.policy_logits for o in outputs],axis=1);values=j.stack([o.value for o in outputs],axis=1)
            finite=j.stack([output_finite(o) for o in outputs],axis=1)
        else:
            inputs=jax.tree.map(lambda *xs:j.stack(xs,axis=1).reshape((2*b,)+xs[0].shape[1:]),
                *(encoder.actor_inputs(o) for o in observations))
            raw_logits=j.zeros((2*b,9),j.float32);values=j.zeros(2*b,j.float32);finite=j.zeros(2*b,bool)
            for actor,indices in groups:
                at=j.asarray(indices,j.int32)
                out=v5_model.forward(params[actor],jax.tree.map(lambda x:x[at],inputs),model_config,compute_dtype='bfloat16')
                raw_logits=raw_logits.at[at].set(out.policy_logits);values=values.at[at].set(out.value)
                finite=finite.at[at].set(output_finite(out))
            raw_logits=raw_logits.reshape(b,2,9);values=values.reshape(b,2);finite=finite.reshape(b,2)
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
        if actor_bank is not None:input_refused=input_refused|(live&(c.streams!=j.asarray(actor_bank.streams,j.int32)))
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
        learner_mask=None if actor_bank is None else j.asarray(actor_bank.learner_mask,bool)
        actor_indices=None if actor_bank is None else j.asarray(actor_bank.actor_indices,j.int32)
        trainable=valid&(j.asarray(bindings.learner_sides)[None,:] if actor_bank is None else learner_mask)
        stack=lambda name:j.stack([getattr(o,name) for o in observations],axis=1)
        record=Frames(stack('numeric_features'),stack('categorical_ids'),stack('token_type_ids'),stack('attention_mask'),
            legal,c.streams,c.frames,state.turn,requested,trainable,valid,errors,actions,engine_actions,probabilities,values,finite,live&~refused,physical,
            opponent_targets,opponent_mask,hidden,hs,hm,actor_indices,learner_mask,raw_logits if retain_teacher_roots else None)
        return carry,(record,public_root_context(c)) if retain_teacher_roots else record
    def collect(c,params):
        if actor_bank is None and len(params)!=2:
            raise ValueError('two explicit original seat parameter trees required')
        if actor_bank is not None and len(params)!=len(actor_bank.actor_archives):
            raise ValueError('one explicit original parameter tree per declared actor required')
        carry,records=jax.lax.scan(lambda c,_:advance(c,params),c,None,length=chunk_steps)
        if retain_teacher_roots:
            frames,roots=records
            return Rollout(carry,frames,roots)
        return Rollout(carry,records)
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


def constructor_truth(raw_state,pool,team_indices,*,source,vocabulary,compiled_source=None):
    """Match ORIGINAL pre-entry pool rows once, never later private-state guesses."""
    from . import v5_collection_data as original
    indices=np.asarray(team_indices);b=raw_state.hp.shape[0];meta=pool.metadata();v=meta['vocabulary']
    if indices.shape!=(b,2) or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(pool.arrays['species'])):
        raise ValueError('same actual constructor team indices required')
    if contract.source_variant_census(source)['source_rows_sha256']!=vocabulary.source_binding_sha256:
        raise ValueError('same complete original source label axis required')
    # The full label census belongs to the original Gen3RandbatSource; the
    # constructor matcher uses its compiled source-order species/variant API.
    # These are distinct original types, never interchangeable placeholders.
    from . import public_belief
    if compiled_source is None:compiled_source=public_belief.compile_source(source)
    if type(compiled_source)is not public_belief.SourceTables or compiled_source.source_hash!=source.metadata.source_hash:
        raise ValueError('same original compiled constructor-label source required')
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
                    hidden_power_type=v['types'][get('hidden_power_type')],source=compiled_source)
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

def training_batch(host_rollouts,bindings,*,battle_ids,seeds,loss_config,discount=1.,actor_bank=None,teacher_records=None,teacher_recipe=None):
    """Batch-boundary adapter into the ORIGINAL validated v5 cache/trainer API.

    Retains actual all29 inputs. Terminal unknowns never manufacture PPO,
    outcome or end-HP targets. Complete trajectories use original gae_targets;
    any missing/invalid learner decision disables PPO for that owner/stream.
    Raw requested/error rows remain in host_rollouts even when not trainable.
    This entry does not substitute for the configured historical assignment
    planner. Explicit D records use the original public-leaf target algebra;
    no private critic or replacement trainer is introduced.
    """
    from types import MappingProxyType
    from pokezero.dataset import TrainingBatch
    from pokezero import v5_data
    from pokezero.v5_loss import V5LossConfig
    if type(bindings)is not Bindings or type(loss_config)is not V5LossConfig:
        raise ValueError('bound actual rollout and original loss recipe required')
    if loss_config.arm=='D':
        from .v5_device_teacher import validate_records
        if teacher_records is None or actor_bank is None:raise ValueError('D consumer absent: original teacher records/bank required')
    elif teacher_records is not None or teacher_recipe is not None:
        raise ValueError('teacher labels belong only to the explicit D treatment')
    if isinstance(discount,bool) or not isinstance(discount,(int,float)) or not np.isfinite(discount) or not 0<=discount<=1:
        raise ValueError('finite declared discount required')
    rolls=tuple(host_rollouts)
    if not rolls or any(isinstance(x,jax.Array) for x in jax.tree.leaves(rolls)):
        raise ValueError('explicit completed batch-boundary device_get required')
    carry=rolls[-1].carry;b=len(carry.streams)
    if len(battle_ids)!=b or len(seeds)!=b or len(set(battle_ids))!=b:raise ValueError('original declared battle/seed rows required')
    if any(not np.array_equal(x.carry.streams,carry.streams) for x in rolls):raise ValueError('same fixed cohort, no refill required')
    f=jax.tree.map(lambda *xs:np.concatenate(xs,axis=0),*(x.frames for x in rolls))
    if actor_bank is None:
        if f.actor_indices is not None or f.learner_mask is not None:raise ValueError('bank rollout requires its exact original assignment binding')
    else:
        if type(actor_bank)is not ActorBank or actor_bank.bindings!=bindings or tuple(map(int,carry.streams))!=actor_bank.streams:
            raise ValueError('same original actor-bank declaration required at boundary')
        expected_actors=np.broadcast_to(np.asarray(actor_bank.actor_indices,np.int32),f.policy_valid.shape)
        expected_learners=np.broadcast_to(np.asarray(actor_bank.learner_mask,bool),f.policy_valid.shape)
        if f.actor_indices is None or f.learner_mask is None or not np.array_equal(f.actor_indices,expected_actors) or not np.array_equal(f.learner_mask,expected_learners) or not np.array_equal(f.trainable,f.policy_valid&expected_learners):
            raise ValueError('literal row actor/learner assignments differ from declaration')
    teacher_by_address=None
    if loss_config.arm=='D':teacher_by_address=validate_records(f,actor_bank,teacher_recipe,teacher_records)
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
        archives=bindings.actor_archives if actor_bank is None else actor_bank.actor_archives
        actor=owner if actor_bank is None else int(f.actor_indices[step[i],row,owner])
        opponent=1-owner if actor_bank is None else int(f.actor_indices[step[i],row,1-owner])
        body={'schema':SCHEMA,'source_sha256':bindings.source_sha256,'declaration_sha256':bindings.declaration_sha256,
            'actor_archives':archives,'stream':int(streams[i]),'frame':int(frames[i]),'owner':int(owner),
            'numeric':numbers[i].tolist(),'categorical':cats[i].tolist(),'roles':roles[i].tolist(),'attention':attention[i].tolist(),
            'legal':legal[i].tolist(),'action':int(actions[i]),'engine_action':int(engine_actions[i]),
            'probability':float(prob[i]),'value':float(values[i]),'physical':physical[i].tolist()}
        if actor_bank is not None:body.update(actor_index=actor,opponent_actor_index=opponent,learner=True,assignment_sha256=actor_bank.assignment_sha256)
        source_hashes.append(_sha(body))
        meta={'device_rollout_schema':SCHEMA,'source_frame_sha256':source_hashes[-1],'v5_return_known':statuses[i]=='COMPLETE',
            'actor_archive_sha256':archives[actor],'opponent_archive_sha256':archives[opponent]}
        if actor_bank is not None:meta.update(actor_bank_assignment_sha256=actor_bank.assignment_sha256,actor_bank_index=actor,opponent_actor_bank_index=opponent,learner=True)
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
        full=f.requested[:,row,owner]&(np.asarray(bindings.learner_sides[owner]) if actor_bank is None else f.learner_mask[:,row,owner])
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
    search_policy=np.zeros((n,9),np.float32);search_value=np.zeros(n,np.float32)
    search_pm=np.zeros(n,bool);search_vm=np.zeros(n,bool)
    if teacher_by_address is not None:
        records=[teacher_by_address[key] for key in keys]
        sidecar.update(search_status=tuple(x['status']for x in records),
            search_refusal=tuple(x['refusal']for x in records),
            root_improvement_records=tuple(x['improvement']for x in records))
        for i,record in enumerate(records):
            if record['status']=='LABELED':
                target=contract.search_target_columns(contract.RootImprovement(**record['improvement']),
                    float(outcome[i]) if om[i] else None,tuple(map(bool,legal[i])))
                search_policy[i]=target['search_policy_target'];search_value[i]=target['search_value_target']
                search_pm[i]=target['search_policy_target_mask'];search_vm[i]=target['search_value_target_mask']
    batch=TrainingBatch(categorical_ids=cats[:,None],numeric_features=numbers[:,None],token_type_ids=roles[:,None],
        attention_mask=attention[:,None],history_mask=np.ones((n,1),bool),legal_action_mask=legal,action_indices=actions,
        rewards=rewards,returns=outcome,value_estimates=values,value_estimate_mask=np.ones(n,bool),
        ppo_advantages=advantage,ppo_advantage_mask=ppo,ppo_value_targets=value_target,ppo_value_target_mask=ppo,
        opponent_action_indices=np.full(n,-1,np.int32),opponent_action_mask=empty,action_probabilities=prob,
        action_probability_mask=np.ones(n,bool),training_weights=np.ones(n,np.float32),
        battle_ids=tuple(battle_ids[row] for row in rows),seeds=tuple(int(seeds[row]) for row in rows),format_ids=('gen3randombattle',)*n,
        player_ids=tuple(('p1','p2')[owner] for owner in owners),turn_indices=tuple(map(int,turns)),
        terminal_capped=tuple(s=='CAPPED' for s in statuses),step_metadata=tuple(metadata),
        search_policy_targets=search_policy,search_policy_target_mask=search_pm,
        search_value_targets=search_value,search_value_target_mask=search_vm,v5_columns=MappingProxyType(sidecar))
    v5_data.validate_batch(batch)
    return batch


def write_cache(path,host_rollouts,bindings,**kwargs):
    """Existing exclusive v5 writer, called only after the explicit boundary."""
    from pokezero import v5_data
    return v5_data.write_cache(path,training_batch(host_rollouts,bindings,**kwargs))
