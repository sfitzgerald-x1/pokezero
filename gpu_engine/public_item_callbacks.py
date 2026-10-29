"""Intermediate public item states in the Python source's callback order.

This does not schedule raw events or infer callbacks from final bit changes.
The enclosing producer must perform source target lookups and checkpoint each
returned stage when its explicit mask is set. In particular a repeated removal
still has a mutation callback, and Trick credits the partner only afterwards.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from . import public_items as items,public_mon_state as mon
from .public_abilities import _bool

SCHEMA='pokezero.public_item_callbacks.v1'


class Stages(NamedTuple):
    mutation: items.ItemState
    reveal: items.ItemState
    partner: items.ItemState
    mutation_callback: jax.Array
    reveal_callback: jax.Array
    partner_callback: jax.Array
    berry_ate: jax.Array
    admitted: jax.Array


def _validate(state,mons,tables):
    if mons.encounter.ndim!=3 or mons.encounter.shape[1:]!=(2,6) or not mons.encounter.shape[0]:
        raise ValueError('item callback stages require a nonempty public encounter batch')
    b=mons.encounter.shape[0]
    shapes={'binding':((b,8),jnp.uint32),'species_binding':((b,8),jnp.uint32),
        'pending':((b,),jnp.bool_),'pending_side':((b,2),jnp.int32),'pending_slot':((b,2),jnp.int32),
        'pending_encounter':((b,2),jnp.int32),'pending_original_known':((b,2),jnp.bool_)}
    for name in ('revealed','original','current'):shapes[name]=((b,2,6),jnp.int32)
    for name in ('mutated','removed'):shapes[name]=((b,2,6),jnp.bool_)
    for name,(shape,dtype) in shapes.items():
        value=getattr(state,name)
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('item callback state shape/dtype: '+name)
    for value,shape,dtype in ((mons.encounter,(b,2,6),jnp.int32),(mons.species_binding,(b,8),jnp.uint32),
            (mons.active,(b,2),jnp.int32),(mons.fainted,(b,2,6),jnp.bool_)):
        if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('item callback public mon shape/dtype')
    if tables.binding.shape!=(8,) or tables.binding.dtype!=jnp.uint32:
        raise ValueError('item callback requires eight uint32 item namespace words')
    count=tables.predicate.shape[0]
    ok=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==mons.species_binding,axis=-1)
    for name in ('revealed','original','current'):
        value=getattr(state,name);ok &= jnp.all((value>=0)&(value<count),axis=(-1,-2))
    ok &= ~jnp.any(state.removed&(state.current>0),axis=(-1,-2))
    ok &= ~jnp.any((state.current>0)&~state.mutated,axis=(-1,-2))
    for name in ('revealed','original','current','mutated','removed'):
        ok &= ~jnp.any((mons.encounter==0)&(getattr(state,name)!=0),axis=(-1,-2))
    ok &= jnp.all((state.pending_side>=-1)&(state.pending_side<2)&(state.pending_slot>=-1)&(state.pending_slot<6)&
        (state.pending_encounter>=0),axis=-1)
    row=jnp.arange(b)[:,None];side=jnp.clip(state.pending_side,0,1);slot=jnp.clip(state.pending_slot,0,5)
    present=(state.pending_side>=0)&(state.pending_slot>=0)&(state.pending_encounter>0)&(
        mons.encounter[row,side,slot]==state.pending_encounter)
    ok &= jnp.where(state.pending,jnp.all(present,axis=-1)&(state.pending_side[:,0]!=state.pending_side[:,1]),
        jnp.all(state.pending_encounter==0,axis=-1)&~jnp.any(state.pending_original_known,axis=-1))
    return ok


def project(state,mons,tables,*,side,slot,encounter,item_id,operation,item_binding,is_eat=False,valid=True):
    """Return mutation→named reveal→partner credit, with source callback masks.

    DIRECT/RECEIVE are explicit -item callbacks even for an unchanged name.
    INLINE and -enditem raw reveals skip only the duplicate named reveal; an
    end-item mutation callback still occurs on every such raw line.
    """
    ok=_validate(state,mons,tables);b=mons.encounter.shape[0];rows=jnp.arange(b)
    def word(value):
        value=mon._int32(value)
        if value.shape not in ((),(b,)):raise ValueError('item callback word must be scalar or exact batch')
        return jnp.broadcast_to(value,(b,))
    def flag(value):
        value=_bool(value)
        if value.shape not in ((),(b,)):raise ValueError('item callback flag must be scalar or exact batch')
        return jnp.broadcast_to(value,(b,))
    side,slot,encounter,item_id,operation=map(word,(side,slot,encounter,item_id,operation))
    is_eat,valid=map(flag,(is_eat,valid))
    ok &= ~is_eat|(operation==items.CONSUME)
    complete,record_ok=items.record(state,mons,tables,side=side,slot=slot,encounter=encounter,
        item_id=item_id,operation=operation,item_binding=item_binding,valid=valid&ok)
    ok &= record_ok
    use=ok&valid
    raw_end=(operation==items.REMOVE)|(operation==items.CONSUME)|(operation==items.OTHER_END)
    explicit=(operation==items.DIRECT)|(operation==items.TRICK_RECEIVE)|(operation==items.OTHER_RECEIVE)
    old=state.revealed[rows,jnp.clip(side,0,1),jnp.clip(slot,0,5)]
    mutation_callback=use&raw_end
    reveal_callback=use&(explicit|(old!=item_id))
    partner_callback=(complete.original!=state.original)&(use&(operation==items.TRICK_RECEIVE))[:,None,None]
    # The source first replaces the subject, then resolves/credits its partner.
    subject=complete._replace(original=jnp.where(partner_callback,state.original,complete.original))
    def choose(mask,new,old):
        return jax.tree.map(lambda a,b:jnp.where(mask.reshape((mask.shape[0],)+(1,)*(a.ndim-1)),a,b),new,old)
    mutation=choose(mutation_callback,subject._replace(revealed=state.revealed),state)
    reveal=choose(reveal_callback,subject,mutation)
    return Stages(mutation,reveal,complete,mutation_callback,reveal_callback,partner_callback,
        use&is_eat,ok|~valid)
