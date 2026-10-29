"""Pending public Mud Shot evidence, resolved at the next move or upkeep.

This ports the source inference on its cached possible abilities. The raw fold
must resolve before the next move's upsert, apply an authoritative original
claim and checkpoint, then queue the new target after that actor's upsert.
"""
import hashlib
import json
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokezero.dex import normalize_id
from . import public_mon_state as mon
from .public_moves import _bool

SCHEMA='pokezero.public_mudshot.v1'
QUEUE,DAMAGE,UNBOOST,FAINT,RESOLVE=range(5)


class Tables(NamedTuple):
    shield_dust: jax.Array
    known_ability: jax.Array
    shield_dust_id: jax.Array
    ability_count: jax.Array
    source_binding: jax.Array
    species_binding: jax.Array
    ability_binding: jax.Array
    binding: jax.Array


def compile_mudshot(source,species,abilities):
    for a,b in ((source.device.binding,species.device.source_binding),
            (source.device.binding,abilities.device.source_binding),
            (abilities.device.binding,species.device.ability_binding)):
        if not np.array_equal(np.asarray(a),np.asarray(b)):raise ValueError('Mud Shot requires one frozen source/species/ability namespace')
    masks=[np.zeros_like(np.asarray(source.device.pool)) for _ in range(2)]
    for index,variants in enumerate(source.variants):
        for number,variant in enumerate(variants):
            name=normalize_id(variant.ability or '')
            for mask,yes in zip(masks,(name=='shielddust',bool(name))):
                if yes:mask[index,number//32]|=np.uint32(1<<(number%32))
    ability_id=abilities.ids.get('shielddust',0)
    if ability_id<=0:raise ValueError('Mud Shot requires a semantic Shield Dust ability row')
    payload=dict(schema=SCHEMA,masks=[x.tolist() for x in masks],shield_dust_id=ability_id,
        ability_count=len(abilities.names),source=np.asarray(source.device.binding).tolist(),
        species=np.asarray(species.device.binding).tolist(),abilities=np.asarray(abilities.device.binding).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return Tables(*(jnp.asarray(x,jnp.uint32) for x in masks),jnp.asarray(ability_id,jnp.int32),
        jnp.asarray(len(abilities.names),jnp.int32),source.device.binding,species.device.binding,
        abilities.device.binding,jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)))


class State(NamedTuple):
    binding: jax.Array
    side: jax.Array
    slot: jax.Array
    encounter: jax.Array
    saw_damage: jax.Array
    cancelled: jax.Array


class Result(NamedTuple):
    state: State
    side: jax.Array
    slot: jax.Array
    encounter: jax.Array
    reveal: jax.Array
    admitted: jax.Array


def initialize(batch,tables):
    if type(batch) is not int or batch<=0:raise ValueError('positive Mud Shot batch required')
    return State(jnp.broadcast_to(tables.binding,(batch,8)),jnp.full(batch,-1,jnp.int32),jnp.full(batch,-1,jnp.int32),
        jnp.zeros(batch,jnp.int32),jnp.zeros(batch,bool),jnp.zeros(batch,bool))


def apply(state,mons,abilities,source_tables,tables,effective,live,*,operation,side=-1,slot=-1,encounter=0,
        tagged=False,producer_binding,valid=True):
    """QUEUE is an actual Mud Shot's opposing active, even while called/copied.

    Any unboost of the pending target cancels, regardless of stat. Only plain
    damage marks a hit; faint clears it. Switches and other lines leave it live.
    A missing target on QUEUE leaves no pending record; RESOLVE clears it even
    when the original reveal or cached candidate-ability guard suppresses a claim.
    """
    if state.side.ndim!=1 or not state.side.shape[0]:raise ValueError('nonempty Mud Shot batch required')
    b=state.side.shape[0]
    for value,shape,dtype in ((state.binding,(b,8),jnp.uint32),(state.side,(b,),jnp.int32),
            (state.slot,(b,),jnp.int32),(state.encounter,(b,),jnp.int32),(state.saw_damage,(b,),jnp.bool_),
            (state.cancelled,(b,),jnp.bool_),(mons.encounter,(b,2,6),jnp.int32),
            (mons.source_species,(b,2,6),jnp.int32),(mons.active,(b,2),jnp.int32),
            (mons.species_binding,(b,8),jnp.uint32),(abilities.binding,(b,8),jnp.uint32),
            (abilities.species_binding,(b,8),jnp.uint32),(abilities.original,(b,2,6),jnp.int32),
            (effective,(b,2,6,2),jnp.uint32),(live,(b,2),jnp.bool_)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):raise ValueError('Mud Shot state/snapshot shape/dtype')
    for value in (*tables[4:],producer_binding,source_tables.binding):
        if getattr(value,'shape',None)!=(8,) or value.dtype!=jnp.uint32:raise ValueError('Mud Shot requires eight uint32 namespace words')
    for value in tables[:2]:
        if value.shape!=source_tables.pool.shape or value.dtype!=jnp.uint32:raise ValueError('Mud Shot source mask shape/dtype')
    for value in tables[2:4]:
        if value.shape!=() or value.dtype!=jnp.int32:raise ValueError('Mud Shot semantic IDs require scalar int32')
    def packet(value,boolean=False):
        value=_bool(value) if boolean else mon._int32(value)
        if value.shape not in ((),(b,)):raise ValueError('Mud Shot fact requires scalar or exact batch')
        return jnp.broadcast_to(value,(b,))
    op,side,slot,enc=map(packet,(operation,side,slot,encounter));tagged=packet(tagged,True);valid=packet(valid,True)
    rows=jnp.arange(b);seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5)
    ps=jnp.clip(state.side,0,1);pt=jnp.clip(state.slot,0,5);pending=state.side>=0
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(producer_binding==tables.binding)
    ok &= jnp.all(source_tables.binding==tables.source_binding)&jnp.all(abilities.binding==tables.ability_binding,axis=-1)
    for value in (mons.species_binding,abilities.species_binding):ok &= jnp.all(value==tables.species_binding,axis=-1)
    ok &= (tables.ability_count>0)&(tables.shield_dust_id>0)&(tables.shield_dust_id<tables.ability_count)
    ok &= jnp.all((abilities.original>=0)&(abilities.original<tables.ability_count),axis=(1,2))
    ok &= jnp.where(pending,(state.side<2)&(state.slot>=0)&(state.slot<6)&(state.encounter>0)&
        (mons.encounter[rows,ps,pt]==state.encounter),(state.side==-1)&(state.slot==-1)&(state.encounter==0)&~state.saw_damage&~state.cancelled)
    ok &= (op>=QUEUE)&(op<=RESOLVE)&((op==DAMAGE)|~tagged)
    none=(side==-1)&(slot==-1)&(enc==0)
    identity=(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0)&(mons.encounter[rows,seat,target]==enc)
    ok &= jnp.where(op==RESOLVE,none,identity|((op==QUEUE)&none))
    ok &= jnp.all((mons.active>=-1)&(mons.active<6),axis=-1)
    ok &= jnp.all((mons.source_species>=0)&(mons.source_species<source_tables.pool.shape[0]),axis=(1,2))
    safe=jnp.clip(mons.source_species,0,source_tables.pool.shape[0]-1)
    absolute=effective[:,jnp.array([1,0])]
    ok &= ~jnp.any((absolute&~source_tables.pool[safe])!=0,axis=(1,2,3))
    active_seen=mons.encounter[rows[:,None],jnp.arange(2)[None,:],jnp.clip(mons.active,0,5)]>0
    ok &= jnp.all(~live|((mons.active>=0)&active_seen),axis=-1)
    ok &= (op!=QUEUE)|none|(live[rows,seat]&(mons.active[rows,seat]==slot))
    ok &= (op!=QUEUE)|~pending
    match=pending&(side==state.side)&(slot==state.slot)&(enc==state.encounter)
    species=jnp.clip(mons.source_species[rows,ps,pt],0,source_tables.pool.shape[0]-1)
    mask=effective[rows,1-ps,pt]
    dust=jnp.any((mask&tables.shield_dust[species])!=0,axis=-1)
    known=jnp.any((mask&tables.known_ability[species])!=0,axis=-1)
    reveal=(op==RESOLVE)&pending&state.saw_damage&~state.cancelled&(abilities.original[rows,ps,pt]==0)&(~known|dust)
    clear=(op==RESOLVE)|((op==FAINT)&match)
    queued=(op==QUEUE)&~none
    candidate=state._replace(side=jnp.where(clear,-1,jnp.where(queued,side,state.side)),
        slot=jnp.where(clear,-1,jnp.where(queued,slot,state.slot)),encounter=jnp.where(clear,0,jnp.where(queued,enc,state.encounter)),
        saw_damage=jnp.where(clear|queued,False,state.saw_damage|((op==DAMAGE)&match&~tagged&~state.cancelled)),
        cancelled=jnp.where(clear|queued,False,state.cancelled|((op==UNBOOST)&match)))
    use=valid&ok
    choose=lambda new,old:jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),new,old)
    return Result(jax.tree.map(choose,candidate,state),state.side,state.slot,state.encounter,use&reveal,ok|~valid)
