"""Standing public survivor pins and source-visible device belief summaries.

Hard reveal/non-proc facts and standing survivor pins are different inputs.
Registering survivors never refreshes a visible summary. Only the caller's
actual source upsert/replace cutpoint does that, including unchanged-hard
refreshes. No engine/world, damage formula or private opponent fact is read.
"""
from typing import NamedTuple
import jax.numpy as j
from . import public_belief as original
from . import lane_index as _li


class State(NamedTuple):
    pin: object                   # [B,2,6,2] original-source variant identities
    pin_active: object            # [B,2,6]
    conflicts: object             # incoming disjoint pins + refresh conflicts
    visible_hard: object          # cached basis, not a replacement hard mask
    visible_pin: object
    visible_pin_active: object
    excluded_abilities: object     # packed engine-ID names, not variant bits
    excluded_items: object


def _ensure(trackers, tables):
    if trackers.visible_belief is not None:return trackers
    hard=trackers.belief_mask;shape=hard.shape[:-1]
    z=j.zeros_like(hard);no=j.zeros(shape,bool)
    carry=State(z,no,j.zeros(shape,j.int32),hard,z,no,
        j.zeros(shape+((tables.ability_mask.shape[1]+31)//32,),j.uint32),
        j.zeros(shape+((tables.item_mask.shape[1]+31)//32,),j.uint32))
    return trackers._replace(visible_belief=carry)


def _member(trackers, tables, side, physical, use):
    b=len(trackers.error);r=j.arange(b);physical=j.asarray(physical,j.int32)
    if physical.shape!=(b,):raise ValueError('public belief physical subject requires B rows')
    slot=_li.take(trackers.physical_to_public,r,side,j.clip(physical,0,5));at=j.clip(slot,0,5)
    species=_li.take(trackers.species,r,side,at);pools=tables.species_initial_mask
    valid=j.broadcast_to(j.asarray(use,bool),(b,))&(physical>=0)&(physical<6)&(slot>=0)&(slot<6)
    valid&=_li.take(trackers.present,r,side,at)&(species>0)&(species<pools.shape[0])
    pool=pools[j.clip(species,0,pools.shape[0]-1)]
    return r,at,species,pool,valid


def _state(trackers,r,side,at,species):
    carry=trackers.visible_belief;hard=_li.take(trackers.belief_mask,r,side,at)
    # All admitted subjects have an actual source encounter. Inconsistency is
    # determined by hard==0, as in the existing device projector; no fallback
    # is written into the raw hard mask or standing pin.
    return original.BeliefState(species,j.ones_like(species,bool),hard,
        _li.take(carry.pin,r,side,at),_li.take(carry.pin_active,r,side,at),j.ones_like(species,bool),_li.take(carry.conflicts,r,side,at))


class _Pool(NamedTuple):
    pool:object


def refresh(trackers,tables,side,physical,use):
    """Copy the basis only at an actual source upsert/replace for this subject."""
    if tables is None or trackers.belief_mask is None:return trackers
    trackers=_ensure(trackers,tables);carry=trackers.visible_belief
    r,at,species,pool,valid=_member(trackers,tables,side,physical,use)
    current=_state(trackers,r,side,at,species)
    materialized,_=original.materialize(current,_Pool(tables.species_initial_mask))
    def copy(old,new):
        before=_li.take(old,r,side,at)
        return _li.put(old,r,(side,at),j.where(valid.reshape(valid.shape+(1,)*(before.ndim-1)),new,before))
    return trackers._replace(visible_belief=carry._replace(
        visible_hard=copy(carry.visible_hard,current.hard),visible_pin=copy(carry.visible_pin,current.pin),
        visible_pin_active=copy(carry.visible_pin_active,current.pin_active),conflicts=copy(carry.conflicts,materialized.conflicts)))


def register_variant_survivors(trackers,tables,side,physical,incoming,use):
    """Original asymmetric public survivor registration, deliberately no refresh.

    A producer must supply source-order survivor identities from actual PUBLIC
    evidence. No producer is synthesized here. Empty/disjoint inputs preserve
    the standing pin; out-of-source bits are unavailable evidence.
    """
    b=len(trackers.error)
    if getattr(incoming,'shape',None)!=(b,2) or incoming.dtype!=j.uint32:
        raise ValueError('source survivor mask requires Bx2 uint32')
    if tables is None or trackers.belief_mask is None:return trackers,j.zeros(b,bool)
    trackers=_ensure(trackers,tables);carry=trackers.visible_belief
    r,at,species,pool,valid=_member(trackers,tables,side,physical,use)
    valid&=j.all((incoming&~pool)==0,axis=-1)
    current=_state(trackers,r,side,at,species)
    updated,changed=original.narrow_pin(current,incoming,valid=valid)
    return trackers._replace(visible_belief=carry._replace(
        pin=_li.put(carry.pin,r,(side,at),updated.pin),
        pin_active=_li.put(carry.pin_active,r,(side,at),updated.pin_active),
        conflicts=_li.put(carry.conflicts,r,(side,at),updated.conflicts))),changed


def exclude(trackers,tables,side,physical,facts,use,*,kind):
    """New ruled-out ability/item names are hard evidence + immediate refresh.

    ``facts`` is B or BxK engine IDs from supported deterministic public rules.
    Keeping the name ledger matters even when a source predicate is empty:
    Python records a newly excluded name and refreshes, but repeated names do
    neither. Several names in one rule refresh once, matching _rule_out_items.
    """
    if kind not in ('ability','item'):raise ValueError('public exclusion kind')
    if tables is None or trackers.belief_mask is None:return trackers
    trackers=_ensure(trackers,tables);carry=trackers.visible_belief;b=len(trackers.error)
    facts=j.asarray(facts,j.int32)
    if facts.ndim==1:facts=facts[:,None]
    if facts.ndim!=2 or facts.shape[0]!=b or facts.shape[1]<1:raise ValueError('public exclusion names require BxK')
    r,at,species,pool,valid=_member(trackers,tables,side,physical,use)
    table=getattr(tables,kind+'_mask');field='excluded_abilities' if kind=='ability' else 'excluded_items'
    ledger=getattr(carry,field);before=_li.take(ledger,r,side,at)
    valid&=j.all((facts>0)&(facts<table.shape[1]),axis=-1)
    bits=j.where(j.arange(before.shape[-1])[None,None,:]==facts[...,None]//32,
        j.uint32(1)<<((facts%32)[...,None].astype(j.uint32)),j.uint32(0))
    names=j.bitwise_or.reduce(bits,axis=1);fresh=j.any((names&~before)!=0,axis=-1)&valid
    excluded=table[j.clip(species,0,table.shape[0]-1)[:,None],j.clip(facts,0,table.shape[1]-1)]
    excluded=j.bitwise_or.reduce(excluded,axis=1)
    hard=trackers.belief_mask;old=_li.take(hard,r,side,at)
    hard=_li.put(hard,r,(side,at),j.where(fresh[:,None],old&~excluded,old))
    ledger=_li.put(ledger,r,(side,at),j.where(fresh[:,None],before|names,before))
    trackers=trackers._replace(belief_mask=hard,visible_belief=carry._replace(**{field:ledger}))
    return refresh(trackers,tables,side,physical,fresh)
