"""Ordered public item non-proc facts and the source's upkeep exclusions.

Snapshots describe HP at the holder's item slot, not the end of the turn.
Unknown residual HP attribution discards the snapshot. The raw producer owns
identity lookup, stage ordering and checkpointing each returned callback.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp

from pokezero.belief import _hp_snapshot_action, _ACTION_PHASE_EVENT_TYPES
from . import public_mon_state as mon
from .public_moves import _bool

NAMES=('leftovers','lumberry','salacberry','petayaberry','liechiberry')
KEEP,UPDATE,DISCARD=range(3)
SAME_PHASE,ACTION_PHASE,RESIDUAL_PHASE=range(3)
NOTE,ARRIVE,HP=range(3)
SCHEMA='pokezero.public_item_nonproc.v1'


@dataclass(frozen=True)
class RawFacts:
    phase: int
    action_phase_hp: int
    residual_phase_hp: int
    leftovers_heal: bool
    berry_eat: bool
    arrival: bool
    hp_line: bool
    upkeep: bool


def lexical_facts(line):
    """Only lexical facts; no belief summaries, hidden state or HP proxy."""
    if not isinstance(line,str) or not line.startswith('|'):raise ValueError('public raw line required')
    parts=line.split('|');kind=parts[1]
    if kind=='request':raise ValueError('private request cannot enter non-proc facts')
    phase=RESIDUAL_PHASE if line.strip()=='|' else ACTION_PHASE if kind=='turn' or kind in _ACTION_PHASE_EVENT_TYPES else SAME_PHASE
    codes={'keep':KEEP,'update':UPDATE,'discard':DISCARD}
    return RawFacts(phase,codes[_hp_snapshot_action(line,in_residual_phase=False)],
        codes[_hp_snapshot_action(line,in_residual_phase=True)],
        kind=='-heal' and '[from] item: Leftovers' in line,
        kind=='-enditem' and '[eat]' in line,kind in ('switch','drag','replace'),
        kind in ('-damage','-heal','-sethp'),line=='|upkeep')


class NonprocState(NamedTuple):
    species_binding: jax.Array
    encounter: jax.Array
    residual: jax.Array
    hp: jax.Array
    maximum: jax.Array
    leftovers_healed: jax.Array
    berry_ate: jax.Array
    excluded: jax.Array           # B,2,6, five NAMES, persistent across stays


class SweepResult(NamedTuple):
    state: NonprocState
    stages: jax.Array             # B,2,3,2,6,5: cumulative excluded ledger
    callbacks: jax.Array          # B,2,3: Leftovers, Lum, pinch per source side
    delta: jax.Array              # B,2,3,5: newly excluded names per callback
    admitted: jax.Array


def initialize(batch,*,species_binding):
    if type(batch) is not int or batch<=0:raise ValueError('positive non-proc batch required')
    z=lambda:jnp.zeros((batch,2,6),jnp.int32)
    return NonprocState(mon.namespace_binding(batch,species_binding),z(),jnp.zeros(batch,bool),z(),z(),
        jnp.zeros((batch,2,6),bool),jnp.zeros((batch,2,6),bool),jnp.zeros((batch,2,6,5),bool))


def _validate(state,mons):
    batch=mons.encounter.shape[0]
    for owner,name,tail,dtype in ((state,'species_binding',(8,),jnp.uint32),
            (mons,'species_binding',(8,),jnp.uint32),(mons,'encounter',(2,6),jnp.int32),
            (state,'encounter',(2,6),jnp.int32),(state,'residual',(),jnp.bool_),
            (state,'hp',(2,6),jnp.int32),(state,'maximum',(2,6),jnp.int32),
            (state,'leftovers_healed',(2,6),jnp.bool_),(state,'berry_ate',(2,6),jnp.bool_),
            (state,'excluded',(2,6,5),jnp.bool_)):
        value=getattr(owner,name)
        if value.shape!=(batch,*tail) or value.dtype!=dtype:raise ValueError('non-proc shape/dtype: '+name)
    ok=jnp.all(state.species_binding==mons.species_binding,axis=-1)
    ok &= jnp.all((state.hp>=0)&(state.maximum>=state.hp)&(state.encounter>=0),axis=(1,2))
    empty=state.encounter==0
    ok &= ~jnp.any(empty&((state.maximum!=0)|state.leftovers_healed|state.berry_ate|jnp.any(state.excluded,axis=-1)),axis=(1,2))
    return ok


def _packet(value,batch,boolean=False):
    value=_bool(value) if boolean else mon._int32(value)
    if value.shape not in ((),(batch,)):raise ValueError('non-proc fact requires scalar or exact batch')
    return jnp.broadcast_to(value,(batch,))


def _choose(use,new,old):
    return jnp.where(use.reshape(use.shape+(1,)*(old.ndim-use.ndim)),new,old)


def record(state,mons,*,operation=NOTE,side=-1,slot=-1,encounter=0,hp=0,maximum=0,
        phase=SAME_PHASE,action_phase_hp=KEEP,residual_phase_hp=KEEP,
        leftovers_heal=False,berry_eat=False,valid=True):
    """Consume public HP and proc facts; no exclusions or summary callbacks yet."""
    ok=_validate(state,mons);batch=mons.encounter.shape[0]
    op,side,slot,enc,hp,maximum,phase,action,residual=(_packet(x,batch) for x in
        (operation,side,slot,encounter,hp,maximum,phase,action_phase_hp,residual_phase_hp))
    left,berry,valid=(_packet(x,batch,True) for x in (leftovers_heal,berry_eat,valid))
    row=jnp.arange(batch);seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5);address=(row,seat,target)
    targeted=(op!=NOTE)|left|berry
    identity=(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0)&(mons.encounter[address]==enc)
    ok &= jnp.where(targeted,identity,(side==-1)&(slot==-1)&(enc==0))
    ok &= (op>=NOTE)&(op<=HP)&(phase>=SAME_PHASE)&(phase<=RESIDUAL_PHASE)
    ok &= (action>=KEEP)&(action<=DISCARD)&(residual>=KEEP)&(residual<=DISCARD)
    ok &= jnp.where((op==HP)|(op==ARRIVE),(hp>=0)&(maximum>=hp),(hp==0)&(maximum==0))
    mask=(jnp.arange(2)[None,:,None]==seat[:,None,None])&(jnp.arange(6)[None,None,:]==target[:,None,None])
    same=state.encounter==mons.encounter
    same |= ((op==ARRIVE)[:,None,None]&mask&((state.encounter==0)|(state.encounter==mons.encounter)))
    ok &= jnp.all(same,axis=(1,2))
    in_residual=jnp.where(phase==RESIDUAL_PHASE,True,jnp.where(phase==ACTION_PHASE,False,state.residual))
    action=jnp.where(in_residual,residual,action)
    action=jnp.where(op==ARRIVE,UPDATE,action)
    update=((op==HP)|(op==ARRIVE))&(action!=KEEP)
    value=jnp.where(action==DISCARD,0,hp);denom=jnp.where(action==DISCARD,0,maximum)
    write=lambda arr,value,which:mon._write(arr,address,value,which)
    candidate=state._replace(encounter=write(state.encounter,enc,op==ARRIVE),residual=in_residual,
        hp=write(state.hp,value,update),maximum=write(state.maximum,denom,update),
        leftovers_healed=write(state.leftovers_healed,True,left),berry_ate=write(state.berry_ate,True,berry))
    use=ok&valid
    return jax.tree.map(lambda new,old:_choose(use,new,old),candidate,state),ok|~valid


def sweep(state,mons,items,sleep,*,item_tables,valid=True):
    """Source order p1 then p2, each Leftovers→Lum→pinch; then clear turn facts.

    Integer comparisons retain the Python fraction thresholds even when FP32
    would round a large numerator up to full HP. Repeated negatives do not
    schedule a callback. Each stage must be checkpointed by the raw producer.
    """
    ok=_validate(state,mons);batch=mons.encounter.shape[0];valid=_packet(valid,batch,True)
    ok &= jnp.all(state.encounter==mons.encounter,axis=(1,2))
    for owner,name,tail,dtype in ((mons,'hp_numerator',(2,6),jnp.int32),(mons,'hp_denominator',(2,6),jnp.int32),
            (mons,'hp_known',(2,6),jnp.bool_),(mons,'active',(2,),jnp.int32),
            (items,'binding',(8,),jnp.uint32),(items,'species_binding',(8,),jnp.uint32),
            (items,'mutated',(2,6),jnp.bool_),(items,'revealed',(2,6),jnp.int32),
            (sleep,'species_binding',(8,),jnp.uint32),(sleep,'encounter',(2,6),jnp.int32),
            (sleep,'active',(2,),jnp.int32),(sleep,'live',(2,),jnp.bool_),(sleep,'status',(2,6),jnp.int32)):
        value=getattr(owner,name)
        if value.shape!=(batch,*tail) or value.dtype!=dtype:raise ValueError('non-proc upkeep shape/dtype: '+name)
    if getattr(item_tables.binding,'shape',None)!=(8,) or item_tables.binding.dtype!=jnp.uint32:
        raise ValueError('non-proc item tables require eight uint32 namespace words')
    if item_tables.predicate.ndim!=1 or item_tables.predicate.dtype!=jnp.int32 or not item_tables.predicate.shape[0]:
        raise ValueError('non-proc item predicate table shape/dtype')
    ok &= jnp.all(items.binding==item_tables.binding,axis=-1)
    ok &= jnp.all((items.revealed>=0)&(items.revealed<item_tables.predicate.shape[0]),axis=(1,2))
    for binding in (items.species_binding,sleep.species_binding):ok &= jnp.all(binding==mons.species_binding,axis=-1)
    ok &= jnp.all(sleep.encounter==mons.encounter,axis=(1,2))
    ok &= jnp.all((sleep.active>=-1)&(sleep.active<6)&(sleep.active==mons.active),axis=-1)
    rows=jnp.arange(batch)[:,None];seats=jnp.arange(2)[None,:]
    active_encounter=mons.encounter[rows,seats,jnp.clip(sleep.active,0,5)]
    ok &= jnp.all(~sleep.live|((sleep.active>=0)&(active_encounter>0)),axis=-1)
    ok &= jnp.all((sleep.status>=mon.NONE)&(sleep.status<=mon.TOXIC),axis=(1,2))
    ok &= jnp.all((mons.hp_numerator>=0)&(mons.hp_denominator>=mons.hp_numerator),axis=(1,2))
    excluded=state.excluded;rows=jnp.arange(batch);stages=[];callbacks=[];deltas=[]
    for side in (0,1):
        slot=jnp.clip(sleep.active[:,side],0,5);address=(rows,side,slot)
        eligible=sleep.live[:,side]&(sleep.active[:,side]>=0)&(mons.encounter[address]>0)
        eligible &= ~items.mutated[address]&(items.revealed[address]==0)&mons.hp_known[address]&(mons.hp_numerator[address]>0)
        known=state.maximum[address]>0
        triggers=(known&(state.hp[address]<state.maximum[address])&~state.leftovers_healed[address],
            sleep.status[address]!=mon.NONE,known&(state.hp[address]<=state.maximum[address]//4)&~state.berry_ate[address])
        seat_stages=[];seat_callbacks=[];seat_deltas=[]
        for group,trigger in zip(((True,False,False,False,False),(False,True,False,False,False),(False,False,True,True,True)),triggers):
            delta=jnp.asarray(group,bool)[None,:]&~excluded[address]&(ok&valid&eligible&trigger)[:,None]
            excluded=excluded.at[address].set(excluded[address]|delta)
            seat_stages.append(excluded);seat_callbacks.append(jnp.any(delta,axis=-1));seat_deltas.append(delta)
        stages.append(jnp.stack(seat_stages,axis=1));callbacks.append(jnp.stack(seat_callbacks,axis=1));deltas.append(jnp.stack(seat_deltas,axis=1))
    candidate=state._replace(excluded=excluded,hp=jnp.zeros_like(state.hp),maximum=jnp.zeros_like(state.maximum),
        leftovers_healed=jnp.zeros_like(state.leftovers_healed),berry_ate=jnp.zeros_like(state.berry_ate))
    return SweepResult(jax.tree.map(lambda new,old:_choose(ok&valid,new,old),candidate,state),
        jnp.stack(stages,axis=1),jnp.stack(callbacks,axis=1),jnp.stack(deltas,axis=1),ok|~valid)


class NonprocTables(NamedTuple):
    predicates: jax.Array
    source_binding: jax.Array
    species_binding: jax.Array
    item_binding: jax.Array
    sleep_binding: jax.Array
    binding: jax.Array


def compile_nonproc(source,species,items,sleep_tables):
    """Bind the five fixed exclusion names to this frozen public stack."""
    import hashlib
    import json
    import numpy as np
    for left,right in ((source.device.binding,species.device.source_binding),
            (source.device.binding,items.device.source_binding),
            (species.device.binding,sleep_tables.species_binding)):
        if (left.shape!=(8,) or right.shape!=(8,) or left.dtype!=jnp.uint32 or right.dtype!=jnp.uint32
                or not np.array_equal(np.asarray(left),np.asarray(right))):
            raise ValueError('non-proc tables require one frozen source/species/item/sleep namespace')
    for value in (items.device.binding,sleep_tables.binding):
        if value.shape!=(8,) or value.dtype!=jnp.uint32:raise ValueError('non-proc table namespace requires eight uint32 words')
    predicates=[source.predicate_id('item',name) for name in NAMES]
    payload=dict(schema=SCHEMA+'.tables.v1',names=NAMES,predicates=predicates,
        source=np.asarray(source.device.binding).tolist(),species=np.asarray(species.device.binding).tolist(),
        items=np.asarray(items.device.binding).tolist(),sleep=np.asarray(sleep_tables.binding).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return NonprocTables(jnp.asarray(predicates,jnp.int32),source.device.binding,species.device.binding,
        items.device.binding,sleep_tables.binding,jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)))
