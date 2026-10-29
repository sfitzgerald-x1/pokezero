"""Pending Intimidate source evidence, resolved before each public raw event.

This ledger does not infer event order or refresh snapshots. The canonical fold
must resolve a pending arrival before the next event, apply the returned public
claim/exclusion and source checkpoint, and queue a new arrival only after its
final source snapshot. Raw candidate abilities intentionally ignore the separate
ruled-out ledger: source fallback can expose a previously excluded ability again.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id
from . import public_mon_state as mon, public_sleep
from .public_abilities import _bool

SCHEMA='pokezero.public_switch_evidence.v1'


class Tables(NamedTuple):
    intimidate: jax.Array
    other: jax.Array
    blockers: jax.Array
    intimidate_id: jax.Array
    blocker_ids: jax.Array
    ability_count: jax.Array
    public_species_count: jax.Array
    source_binding: jax.Array
    species_binding: jax.Array
    ability_binding: jax.Array
    binding: jax.Array


def compile_switch_evidence(source,species,abilities):
    for a,b in ((source.device.binding,species.device.source_binding),
            (source.device.binding,abilities.device.source_binding),
            (abilities.device.binding,species.device.ability_binding)):
        if not np.array_equal(np.asarray(a),np.asarray(b)):
            raise ValueError('switch evidence requires one source/species/ability namespace')
    blockers={'clearbody','hypercutter','whitesmoke'}
    masks=[np.zeros_like(np.asarray(source.device.pool)) for _ in range(3)]
    for index,variants in enumerate(source.variants):
        for number,variant in enumerate(variants):
            name=normalize_id(variant.ability or '')
            for mask,yes in zip(masks,(name=='intimidate',bool(name) and name!='intimidate',name in blockers)):
                if yes:mask[index,number//32]|=np.uint32(1<<(number%32))
    iid=abilities.ids.get('intimidate',0)
    block=[abilities.ids.get(name,0) for name in sorted(blockers)]
    payload=dict(schema=SCHEMA,public_species_count=len(species.names),ability_count=len(abilities.names),masks=[a.tolist() for a in masks],intimidate=iid,blockers=block,
        source=np.asarray(source.device.binding).tolist(),species=np.asarray(species.device.binding).tolist(),
        abilities=np.asarray(abilities.device.binding).tolist())
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return Tables(*(jnp.asarray(x) for x in masks),jnp.asarray(iid,jnp.int32),jnp.asarray(block,jnp.int32),jnp.asarray(len(abilities.names),jnp.int32),jnp.asarray(len(species.names),jnp.int32),
        source.device.binding,species.device.binding,abilities.device.binding,
        jnp.asarray(np.frombuffer(h,dtype='>u4').astype(np.uint32)))


class State(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    side: jax.Array
    slot: jax.Array
    encounter: jax.Array
    has_ident: jax.Array
    ruled_out: jax.Array


class Resolution(NamedTuple):
    state: State
    side: jax.Array
    slot: jax.Array
    encounter: jax.Array
    exclude: jax.Array
    reveal: jax.Array
    checkpoint: jax.Array
    admitted: jax.Array


def initialize(batch,tables):
    if type(batch) is not int or batch<=0:raise ValueError('positive switch evidence batch required')
    z=jnp.zeros(batch,jnp.int32);no=jnp.full(batch,-1,jnp.int32)
    return State(jnp.broadcast_to(tables.binding,(batch,8)),jnp.broadcast_to(tables.species_binding,(batch,8)),
        no,no,z,jnp.zeros(batch,bool),jnp.zeros((batch,2,6),bool))


def validate(state,tables,mons,abilities,source_tables,effective,live):
    if state.side.ndim!=1 or not state.side.shape[0]:raise ValueError('nonempty switch evidence batch required')
    b=state.side.shape[0]
    for value,shape,dtype in ((state.binding,(b,8),jnp.uint32),(state.species_binding,(b,8),jnp.uint32),
            (state.side,(b,),jnp.int32),(state.slot,(b,),jnp.int32),(state.encounter,(b,),jnp.int32),(state.has_ident,(b,),jnp.bool_),
            (state.ruled_out,(b,2,6),jnp.bool_),(mons.encounter,(b,2,6),jnp.int32),
            (mons.source_species,(b,2,6),jnp.int32),(mons.species,(b,2,6),jnp.int32),(mons.active,(b,2),jnp.int32),
            (mons.species_binding,(b,8),jnp.uint32),(abilities.original,(b,2,6),jnp.int32),
            (abilities.binding,(b,8),jnp.uint32),(abilities.species_binding,(b,8),jnp.uint32),
            (effective,(b,2,6,2),jnp.uint32),(live,(b,2),jnp.bool_)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('switch evidence state/snapshot shape/dtype')
    for value in tables[7:]:public_sleep._binding(value)
    for value in tables[:3]:
        if value.shape!=source_tables.pool.shape or value.dtype!=jnp.uint32:
            raise ValueError('switch evidence static source mask shape/dtype')
    if tables.intimidate_id.shape!=() or tables.intimidate_id.dtype!=jnp.int32 or tables.blocker_ids.shape!=(3,) or tables.blocker_ids.dtype!=jnp.int32:
        raise ValueError('switch evidence semantic ability ID shape/dtype')
    if tables.ability_count.shape!=() or tables.ability_count.dtype!=jnp.int32:
        raise ValueError('switch evidence ability count must be scalar int32')
    if tables.public_species_count.shape!=() or tables.public_species_count.dtype!=jnp.int32:
        raise ValueError('switch evidence public species count must be scalar int32')
    rows=jnp.arange(b);seat=jnp.clip(state.side,0,1);slot=jnp.clip(state.slot,0,5)
    pending=state.side>=0
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)&jnp.all(abilities.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(abilities.binding==tables.ability_binding,axis=-1)&jnp.all(source_tables.binding==tables.source_binding)
    ok &= jnp.where(pending,(state.side<2)&(state.slot>=0)&(state.slot<6)&(state.encounter>0)&
        (mons.encounter[rows,seat,slot]==state.encounter),(state.side==-1)&(state.slot==-1)&(state.encounter==0)&~state.has_ident)
    ok &= tables.public_species_count>0
    ok &= jnp.all((mons.species>=0)&(mons.species<tables.public_species_count),axis=(-1,-2))
    ok &= jnp.all((mons.encounter==0)|(mons.species>0),axis=(-1,-2))
    ok &= (tables.ability_count>0)&(tables.intimidate_id>=0)&(tables.intimidate_id<tables.ability_count)
    ok &= jnp.all((tables.blocker_ids>=0)&(tables.blocker_ids<tables.ability_count))
    ok &= jnp.all((abilities.original>=0)&(abilities.original<tables.ability_count),axis=(-1,-2))
    ok &= jnp.all((mons.active>=-1)&(mons.active<6),axis=-1)
    active=jnp.clip(mons.active,0,5)
    seen=mons.encounter[rows[:,None],jnp.arange(2)[None,:],active]>0
    ok &= jnp.all(~live|((mons.active>=0)&seen),axis=-1)
    ok &= jnp.all((mons.source_species>=0)&(mons.source_species<tables.intimidate.shape[0]),axis=(-1,-2))
    ok &= ~jnp.any(state.ruled_out&(mons.encounter==0),axis=(-1,-2))
    # Observer1-side holds the source snapshot of each absolute public side.
    absolute=effective[:,jnp.array([1,0])]
    species=mons.source_species
    safe=jnp.clip(species,0,source_tables.pool.shape[0]-1)
    ok &= ~jnp.any((absolute&~source_tables.pool[safe])!=0,axis=(-1,-2,-3))
    return ok


def _possibilities(effective,mons,tables,side,slot):
    rows=jnp.arange(effective.shape[0]);side=jnp.clip(side,0,1);slot=jnp.clip(slot,0,5)
    species=jnp.clip(mons.source_species[rows,side,slot],0,tables.intimidate.shape[0]-1)
    mask=effective[rows,1-side,slot]
    return tuple(jnp.any((mask&x[species])!=0,axis=-1) for x in tables[:3])


def resolve(state,tables,mons,abilities,source_tables,effective,live,*,arrival,boundary,
            confirmed_ability,confirmed_side=-1,confirmed_slot=-1,confirmed_encounter=0,
            exact_ident=False,species_ident=0,producer_binding,valid=True):
    """Resolve BEFORE common effects, including a FAINT or another arrival.

    exact_ident is normalized full-ident equality with the registered
    positioned arrival ident, not an alias-address match. species_ident is the raw ident name's
    normalized species ID, without cosmetic/base-forme stripping.
    """
    b=state.side.shape[0];rows=jnp.arange(b)
    ok=validate(state,tables,mons,abilities,source_tables,effective,live)
    public_sleep._binding(producer_binding);ok &= jnp.all(producer_binding==tables.binding)
    def word(x):
        x=mon._int32(x)
        if x.shape not in ((),(b,)):raise ValueError('switch evidence word must be scalar or exact batch')
        return jnp.broadcast_to(x,(b,))
    def flag(x):
        x=_bool(x)
        if x.shape not in ((),(b,)):raise ValueError('switch evidence flag must be scalar or exact batch')
        return jnp.broadcast_to(x,(b,))
    arrival,boundary,exact,valid=map(flag,(arrival,boundary,exact_ident,valid))
    claim,cs,ct,ce,ident=map(word,(confirmed_ability,confirmed_side,confirmed_slot,confirmed_encounter,species_ident))
    ok &= (claim>=0)&(claim<tables.ability_count)
    ok &= (cs>=-1)&(cs<2)&(ct>=-1)&(ct<6)&(ce>=0)&(ident>=0)&(ident<tables.public_species_count)
    ok &= ~exact|((claim>0)&(cs>=0)&(ct>=0)&(ce>0))
    side=jnp.clip(state.side,0,1);slot=jnp.clip(state.slot,0,5);pending=state.side>=0
    has_claim=claim>0
    literal=state.has_ident&exact&(cs==state.side)&(ct==state.slot)&(ce==state.encounter)
    # Source returns the full-ident comparison immediately when the pending
    # arrival has an ident. Species fallback exists ONLY without that ident;
    # its legacy predicate does not additionally compare the incoming side.
    species_match=~state.has_ident&(ident>0)&(ident==mons.species[rows,side,slot])
    reveal=pending&~arrival&has_claim&(claim==tables.intimidate_id)&(tables.intimidate_id>0)&(literal|species_match)
    clear=pending&(arrival|(~has_claim&boundary))
    intimidate,other,_=_possibilities(effective,mons,tables,side,slot)
    opposing=1-side;active=jnp.clip(mons.active[rows,opposing],0,5)
    _,_,possible_blocker=_possibilities(effective,mons,tables,opposing,active)
    known=abilities.original[rows,opposing,active]
    known_blocker=jnp.any((known[:,None]==tables.blocker_ids[None,:])&(tables.blocker_ids[None,:]>0),axis=-1)
    blocked=jnp.where(known>0,known_blocker,possible_blocker)
    exclude=clear&intimidate&other&live[rows,opposing]&(mons.active[rows,opposing]>=0)&~blocked
    use=ok&valid;clear=(clear|reveal)&use
    ruled=state.ruled_out.at[rows,side,slot].set(state.ruled_out[rows,side,slot]|(exclude&use))
    result=state._replace(side=jnp.where(clear,-1,state.side),slot=jnp.where(clear,-1,state.slot),
        encounter=jnp.where(clear,0,state.encounter),has_ident=jnp.where(clear,False,state.has_ident),ruled_out=ruled)
    return Resolution(result,state.side,state.slot,state.encounter,exclude&use,reveal&use,(exclude|reveal)&use,ok|~valid)


def queue_arrival(state,tables,mons,abilities,source_tables,effective,live,*,side,slot,encounter,producer_binding,has_ident=True,valid=True):
    """Queue only AFTER arrival's final source checkpoint and previous resolve.

    The source queues from cached possible abilities even if Intimidate has
    already been excluded or a contradictory original reveal is retained.
    """
    b=state.side.shape[0];rows=jnp.arange(b)
    ok=validate(state,tables,mons,abilities,source_tables,effective,live)
    public_sleep._binding(producer_binding);ok &= jnp.all(producer_binding==tables.binding)
    def word(value):
        value=mon._int32(value)
        if value.shape not in ((),(b,)):raise ValueError('switch arrival word must be scalar or exact batch')
        return jnp.broadcast_to(value,(b,))
    side,slot,enc=map(word,(side,slot,encounter));valid=_bool(valid)
    if valid.shape not in ((),(b,)):raise ValueError('switch arrival validity must be scalar or exact batch')
    valid=jnp.broadcast_to(valid,(b,));has_ident=_bool(has_ident)
    if has_ident.shape not in ((),(b,)):raise ValueError('switch arrival ident flag must be scalar or exact batch')
    has_ident=jnp.broadcast_to(has_ident,(b,));seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5)
    ok &= (state.side==-1)&(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0)
    ok &= (mons.encounter[rows,seat,target]==enc)&(mons.active[rows,seat]==slot)&live[rows,seat]
    intimidate,other,_=_possibilities(effective,mons,tables,seat,target)
    queue=intimidate&other&(tables.intimidate_id>0)&ok&valid
    result=state._replace(side=jnp.where(queue,side,state.side),slot=jnp.where(queue,slot,state.slot),
        encounter=jnp.where(queue,enc,state.encounter),has_ident=jnp.where(queue,has_ident,state.has_ident))
    return result,ok|~valid


def lexical_control(line):
    """Return only source event spelling; never inspect a posterior or queue."""
    if not isinstance(line,str):raise ValueError('switch source line must be text')
    p=line.split('|');kind=p[1] if len(p)>1 else ''
    arrival=kind in ('switch','drag','replace')
    boundary=kind not in ('-boost','-unboost','-damage','-heal','-status','-curestatus','-item')
    return arrival,boundary
