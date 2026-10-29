"""Public exit/cure evidence used by ordered source summary replacements.

The existing sleep consumer owns status, Rest counts and skippedTime. This
ledger carries only the additional source cure epochs and Shed Skin proc set
needed to infer original abilities at entry and at a one-turn Rest wake.
Raw framing and summary checkpoints must call it in source order.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import public_mon_state as mon,public_moves as moves,public_sleep as sleep_api

ARRIVE,MOVE,SHED_SKIN_ACTIVATE,CURE,CURE_TEAM,FAINT,UPKEEP=range(1,8)


class LifecycleTables(NamedTuple):
    natural_cure: jax.Array
    early_bird: jax.Array
    shed_skin: jax.Array
    cure_moves: jax.Array
    ability_count: jax.Array
    species_binding: jax.Array
    sleep_binding: jax.Array
    ability_binding: jax.Array
    move_binding: jax.Array
    binding: jax.Array


def _binding(x,shape=(8,)):
    if getattr(x,'shape',None)!=shape or x.dtype!=jnp.uint32:
        raise ValueError('public lifecycle requires exact uint32 namespace words')


def compile_lifecycle(species_catalog,abilities,move_catalog,sleep_tables):
    for x in (species_catalog.device.binding,abilities.device.binding,move_catalog.device.binding,
            abilities.device.source_binding,move_catalog.device.source_binding,sleep_tables.binding):_binding(x)
    if not np.array_equal(np.asarray(species_catalog.device.binding),np.asarray(sleep_tables.species_binding)):
        raise ValueError('lifecycle and sleep require the same public species namespace')
    if not np.array_equal(np.asarray(abilities.device.source_binding),np.asarray(move_catalog.device.source_binding)):
        raise ValueError('lifecycle move/ability catalogs require the same source')
    for actual,expected in ((species_catalog.device.source_binding,abilities.device.source_binding),
            (species_catalog.device.ability_binding,abilities.device.binding)):
        _binding(actual);_binding(expected)
        if not np.array_equal(np.asarray(actual),np.asarray(expected)):
            raise ValueError('lifecycle species and ability catalog namespaces differ')
    ids=[abilities.ids.get(name,0) for name in ('naturalcure','earlybird','shedskin')]
    if any(x<=0 for x in ids):raise ValueError('lifecycle requires semantic Natural Cure, Early Bird and Shed Skin ability rows')
    cures=[name in ('healbell','aromatherapy') for name in move_catalog.names]
    payload=dict(schema='pokezero.public_belief_lifecycle.v1',abilities=ids,cure_moves=cures,
        species=np.asarray(species_catalog.device.binding).tolist(),sleep=np.asarray(sleep_tables.binding).tolist(),
        ability=np.asarray(abilities.device.binding).tolist(),move=np.asarray(move_catalog.device.binding).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    return LifecycleTables(*(jnp.asarray(x,jnp.int32) for x in ids),jnp.asarray(cures,bool),
        jnp.asarray(len(abilities.names),jnp.int32),species_catalog.device.binding,sleep_tables.binding,
        abilities.device.binding,move_catalog.device.binding,jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)))


class LifecycleState(NamedTuple):
    binding: jax.Array
    cure_count: jax.Array          # B,2: source HealBell/Aromatherapy uses, even called
    exit_cure_count: jax.Array     # B,2,6: source epoch at most recent living exit
    exit_status: jax.Array         # explicit source status on exit; not suffix fallback
    shed_skin: jax.Array           # actual -activate proc seen since last upkeep


class LifecycleResult(NamedTuple):
    state: LifecycleState
    ability_claim: jax.Array       # 0=no new original-ability claim
    admitted: jax.Array


def initialize(batch,tables):
    if type(batch) is not int or batch<=0:raise ValueError('positive lifecycle batch required')
    _binding(tables.binding)
    return LifecycleState(jnp.broadcast_to(tables.binding,(batch,8)),jnp.zeros((batch,2),jnp.int32),
        jnp.full((batch,2,6),-1,jnp.int32),jnp.zeros((batch,2,6),bool),jnp.zeros((batch,2,6),bool))


def apply(state,sleep,abilities,mons,tables,*,operation,side=-1,slot=-1,encounter=0,
        move_id=0,ability_id=0,benched=False,producer_binding,valid=True):
    """Consume pre-stage sleep/original ability facts and post-arrival mon facts.

    Returned claims belong between the incoming upsert and switch-in refresh,
    or before the final cure replacement. Neither a claim nor this ledger
    independently materializes a belief summary.
    """
    if state.cure_count.ndim!=2 or state.cure_count.shape[1]!=2:raise ValueError('lifecycle B by2 cure counts required')
    batch=state.cure_count.shape[0]
    for name,tail,dtype in (('binding',(8,),jnp.uint32),('cure_count',(2,),jnp.int32),
            ('exit_cure_count',(2,6),jnp.int32),('exit_status',(2,6),jnp.bool_),('shed_skin',(2,6),jnp.bool_)):
        value=getattr(state,name)
        if value.shape!=(batch,*tail) or value.dtype!=dtype:raise ValueError('lifecycle state shape/dtype: '+name)
    for x in tables[5:]:_binding(x)
    _binding(producer_binding)
    for x in tables[:3]+(tables.ability_count,):
        if x.shape!=() or x.dtype!=jnp.int32:raise ValueError('lifecycle ability scalar requires int32')
    if tables.cure_moves.ndim!=1 or not tables.cure_moves.shape[0] or tables.cure_moves.dtype!=jnp.bool_:raise ValueError('lifecycle cure move rows require nonempty bool rows')
    for value in (sleep.binding,sleep.species_binding,abilities.binding,abilities.species_binding,mons.species_binding):_binding(value,(batch,8))
    for owner,name,dtype,tail in ((sleep,'active',jnp.int32,(2,)),(sleep,'live',jnp.bool_,(2,)),
            (sleep,'status',jnp.int32,(2,6)),(sleep,'condition_status',jnp.int32,(2,6)),
            (sleep,'literal_faint',jnp.bool_,(2,6)),(sleep,'rest',jnp.bool_,(2,6)),
            (sleep,'count',jnp.int32,(2,6)),(sleep,'encounter',jnp.int32,(2,6)),
            (abilities,'original',jnp.int32,(2,6)),(mons,'encounter',jnp.int32,(2,6)),
            (mons,'active',jnp.int32,(2,)),(mons,'status',jnp.int32,(2,6))):
        value=getattr(owner,name)
        if value.shape!=(batch,*tail) or value.dtype!=dtype:raise ValueError('lifecycle public ledger shape/dtype: '+name)
    def packet(x,dtype):
        value=moves._bool(x) if dtype==jnp.bool_ else mon._int32(x)
        if value.shape not in ((),(batch,)):raise ValueError('lifecycle fact must be scalar or exact batch')
        return jnp.broadcast_to(value,(batch,))
    op,side,slot,enc,mid,aid=(packet(x,jnp.int32) for x in (operation,side,slot,encounter,move_id,ability_id))
    benched,valid=(packet(x,jnp.bool_) for x in (benched,valid))
    rows=jnp.arange(batch);seat=jnp.clip(side,0,1);target=jnp.clip(slot,0,5);address=(rows,seat,target)
    global_fact=op==UPKEEP;team=op==CURE_TEAM;targeted=~global_fact&~team
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(sleep.binding==tables.sleep_binding,axis=-1)
    ok &= jnp.all(producer_binding==tables.binding)
    ok &= jnp.all(abilities.binding==tables.ability_binding,axis=-1)
    for binding in (sleep.species_binding,abilities.species_binding,mons.species_binding):ok &= jnp.all(binding==tables.species_binding,axis=-1)
    ok &= (op>=ARRIVE)&(op<=UPKEEP)&jnp.all(state.cure_count>=0,axis=-1)&jnp.all(state.exit_cure_count>=-1,axis=(1,2))
    ids=jnp.stack(tables[:3]);ok &= jnp.all((ids>0)&(ids<tables.ability_count))
    ok &= jnp.all((sleep.active>=-1)&(sleep.active<6),axis=-1)&jnp.all((mons.active>=-1)&(mons.active<6),axis=-1)
    ok &= jnp.all((abilities.original>=0)&(abilities.original<tables.ability_count),axis=(1,2))
    ok &= jnp.all((sleep.status>=mon.NONE)&(sleep.status<=mon.TOXIC)&(sleep.count>=0),axis=(1,2))
    ok &= jnp.all((sleep.condition_status>=mon.NONE)&(sleep.condition_status<=mon.TOXIC),axis=(1,2))
    ok &= jnp.all((mons.status>=mon.NONE)&(mons.status<=mon.TOXIC)&(mons.encounter>=0),axis=(1,2))
    target_mask=(jnp.arange(2)[None,:,None]==seat[:,None,None])&(jnp.arange(6)[None,None,:]==target[:,None,None])
    same=sleep.encounter==mons.encounter
    same |= (op==ARRIVE)[:,None,None]&target_mask
    ok &= jnp.all(same,axis=(1,2))
    active_same=(sleep.active==mons.active)|((op==ARRIVE)[:,None]&(jnp.arange(2)[None,:]==seat[:,None]))
    ok &= jnp.all(active_same,axis=-1)
    identity=(side>=0)&(side<2)&(slot>=0)&(slot<6)&(enc>0)&(mons.encounter[address]==enc)
    identity &= jnp.where(op==ARRIVE,(mons.active[rows,seat]==slot)&((sleep.encounter[address]==0)|(sleep.encounter[address]==enc)),
        (sleep.encounter[address]==enc)&(benched|(sleep.active[rows,seat]==slot)|~sleep.live[rows,seat]))
    ok &= jnp.where(global_fact,(side==-1)&(slot==-1)&(enc==0),
        jnp.where(team,(side>=0)&(side<2)&(slot==-1)&(enc==0),identity))
    ok &= ((op==CURE)|~benched)&jnp.where(op==MOVE,(mid>0)&(mid<tables.cure_moves.shape[0]),mid==0)
    ok &= jnp.where(op==SHED_SKIN_ACTIVATE,aid==tables.shed_skin,aid==0)
    safe_mid=jnp.clip(mid,0,tables.cure_moves.shape[0]-1)
    use_cure=(op==MOVE)&tables.cure_moves[safe_mid]
    ok &= ~use_cure|(state.cure_count[rows,seat]<2**31-1)
    exit_status=state.exit_status;exit_count=state.exit_cure_count
    outgoing=jnp.clip(sleep.active[rows,seat],0,5);old=(rows,seat,outgoing)
    leaving=(op==ARRIVE)&sleep.live[rows,seat]&~sleep.literal_faint[old]
    exit_status=exit_status.at[old].set(jnp.where(leaving,sleep.status[old]!=mon.NONE,exit_status[old]))
    exit_count=exit_count.at[old].set(jnp.where(leaving,state.cure_count[rows,seat],exit_count[old]))
    natural=(op==ARRIVE)&exit_status[address]&(mons.status[address]==mon.NONE)
    natural &= (exit_count[address]==state.cure_count[rows,seat])&(abilities.original[address]==0)
    early=(op==CURE)&~benched&(sleep.status[address]==mon.SLEEP)&sleep.rest[address]&(sleep.count[address]==1)
    early &= ~state.shed_skin[address]&(abilities.original[address]==0)
    claim=jnp.where(natural,tables.natural_cure,jnp.where(early,tables.early_bird,0))
    retire=(op==ARRIVE)|(op==FAINT)|((op==CURE)&benched)
    exit_status=exit_status.at[address].set(jnp.where(retire,False,exit_status[address]))
    exit_count=exit_count.at[address].set(jnp.where(retire,-1,exit_count[address]))
    team_mask=team[:,None,None]&(jnp.arange(2)[None,:,None]==seat[:,None,None])&~sleep.literal_faint
    team_mask &= (sleep.status!=mon.NONE)|(sleep.condition_status!=mon.NONE)
    exit_status=jnp.where(team_mask,False,exit_status);exit_count=jnp.where(team_mask,-1,exit_count)
    shed=state.shed_skin.at[address].set(jnp.where(op==SHED_SKIN_ACTIVATE,True,state.shed_skin[address]))
    shed=jnp.where(global_fact[:,None,None],False,shed)
    counts=state.cure_count.at[rows,seat].add(use_cure.astype(jnp.int32))
    candidate=state._replace(cure_count=counts,exit_cure_count=exit_count,exit_status=exit_status,shed_skin=shed)
    use=ok&valid
    choose=lambda new,old:jnp.where(use.reshape((batch,)+(1,)*(old.ndim-1)),new,old)
    return LifecycleResult(jax.tree.map(choose,candidate,state),jnp.where(use,claim,0),ok|~valid)
