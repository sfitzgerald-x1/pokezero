"""PRIVATE sticky observations at explicitly entered source opportunities.

This is partial instrumentation, NOT admission or exhaustive hook history. The
owning session must mint exactly one ledger before real constructor/Entry hooks,
carry it without reset, and call retain BEFORE each actual hook/setter and at
the AFTER-write boundary. A copied pure-device ledger is not an origin or
at-most-once attestation. Missing calls cannot be detected from final state.

Static NO_RELEVANT_HOOKS means listed-key absence only. Active carrier state,
dynamic durations/events, chosen-move conditions, semantic callback classes,
constructor origin and complete cutpoint wiring remain explicitly PENDING,
even if transport_consistent is true and every observed flag is zero. Relevant
presence is not callback execution. No source/full/encoder/B4 admission API is
provided; the separately registered source context must remain closed.
"""
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_status_hook_source as hooks, public_mon_state
from .state import BattleState

SCHEMA='pokezero.private.status_hook_history.partial.v1'


class Opportunity(IntEnum):
    ENTRY=1
    DAMAGE=2
    STATUS=3
    UPDATE=4
    RESIDUAL=5
    ITEM_USE=6
    POST_WRITE=7
    MOVE_DECLARE=8


NAMESPACE, MALFORMED, ORDER, OVERFLOW, MEMBERSHIP, SELECTED_MOVE = (1<<n for n in range(6))
PENDING=('actual-constructor/session-origin','exhaustive-pre-hook/setter-cutpoints',
    'active-volatile/side/field/slot-carriers','dynamic-duration/Battle.events',
    'selected-move/temporary-condition-history','semantic-callback-class-support')
PENDING_MASK=(1<<len(PENDING))-1
MAX_SERIAL=2**31-1
_PREPARED=weakref.WeakKeyDictionary()


class Tables(NamedTuple):
    binding: object
    hooks: hooks.Data


@dataclass(frozen=True,eq=False)
class Prepared:
    device: Tables


class Ledger(NamedTuple):
    # GLOBAL (8,) leaves. Never select these as B rows, including when B==8.
    history_binding: object
    hook_binding: object
    engine_binding: object
    rules_binding: object
    manifest_binding: object
    # Everything below has leading B.
    stream_id: object
    frame_serial: object
    event_serial: object
    initialized: object
    sticky_refusal: object
    observed_relevant_flags: object       # [B,2], static presence, not dispatch
    observed_classes: object              # [B,2], OR of 1<<classification
    seen_abilities: object                # [B,2,N], exact sticky source IDs
    seen_items: object
    seen_species: object
    seen_moves: object                    # explicit MOVE_DECLARE only
    opportunity_counts: object            # [B,8], entered attempts incl no-write
    attempt_count: object
    last_opportunity: object


GLOBALS=Ledger._fields[:5]


class Report(NamedTuple):
    transport_consistent: object
    sticky_refusal: object
    observed_relevant_flags: object
    observed_classes: object
    opportunity_counts: object
    attempt_count: object
    pending: object


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked hook history host boundary cannot be traced')


def prepare_history(hook_compiled):
    _host()
    data=hooks.check_compiled(hook_compiled)
    payload=dict(schema=SCHEMA,hook=np.asarray(data.binding).tolist(),
        opportunities={x.name:int(x) for x in Opportunity},pending=PENDING,
        producer=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    binding=hooks._words(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    tables=Tables(binding,data);value=Prepared(tables)
    _PREPARED[value]=(tables,tuple(jax.tree.leaves(tables)),hook_compiled)
    return value


def check_prepared(prepared):
    _host()
    record=_PREPARED.get(prepared) if type(prepared) is Prepared else None
    if record is None or prepared.device is not record[0]:
        raise ValueError('original prepared hook history required')
    leaves=tuple(jax.tree.leaves(prepared.device))
    if (len(leaves)!=len(record[1]) or any(a is not b for a,b in zip(leaves,record[1]))
            or any(not isinstance(a,jax.Array) or isinstance(a,jax.core.Tracer) for a in leaves)
            or hooks.check_compiled(record[2]) is not prepared.device.hooks):
        raise ValueError('original immutable hook history/source edges required')
    return prepared.device


def _typed(value,shape,dtype):
    if getattr(value,'shape',None)!=shape or getattr(value,'dtype',None)!=jnp.dtype(dtype):
        raise ValueError('typed private history storage required')


def _tables(tables):
    if type(tables) is not Tables or type(tables.hooks) is not hooks.Data:
        raise ValueError('original checked private history tables required')
    _typed(tables.binding,(8,),jnp.uint32)
    for name in hooks.Data._fields[:6]:_typed(getattr(tables.hooks,name),(8,),jnp.uint32)
    for kind in hooks.KINDS:
        rows=getattr(tables.hooks,kind)
        if type(rows) is not hooks.Rows or getattr(rows.manifest_known,'ndim',None)!=1:
            raise ValueError('typed hook rows required')
        n=rows.manifest_known.shape[0]
        if not n:raise ValueError('nonempty source namespace required')
        _typed(rows.manifest_known,(n,),jnp.bool_)
        _typed(rows.relevant_flags,(n,),jnp.int32)
        _typed(rows.classification,(n,),jnp.int32)
        _typed(rows.row_digest,(n,8),jnp.uint32)


def _state(state):
    if type(state) is not BattleState or getattr(state.active,'ndim',None)!=2:
        raise ValueError('actual typed boundary BattleState required')
    b=state.active.shape[0]
    if not b:raise ValueError('nonempty history batch required')
    _typed(state.active,(b,2),jnp.int32)
    for name in ('ability','item','species'):_typed(getattr(state,name),(b,2,6),jnp.int32)
    return b


def _ledger(ledger,tables):
    if type(ledger) is not Ledger or getattr(ledger.stream_id,'ndim',None)!=1:
        raise ValueError('typed private history ledger required')
    b=ledger.stream_id.shape[0]
    if not b:raise ValueError('nonempty private history ledger required')
    for name in GLOBALS:_typed(getattr(ledger,name),(8,),jnp.uint32)
    for name in ('stream_id','frame_serial','event_serial','sticky_refusal','attempt_count','last_opportunity'):
        _typed(getattr(ledger,name),(b,),jnp.int32)
    _typed(ledger.initialized,(b,),jnp.bool_)
    for name in ('observed_relevant_flags','observed_classes'):_typed(getattr(ledger,name),(b,2),jnp.int32)
    for name,kind in (('seen_abilities','abilities'),('seen_items','items'),('seen_species','species'),('seen_moves','moves')):
        _typed(getattr(ledger,name),(b,2,len(getattr(tables.hooks,kind).manifest_known)),jnp.bool_)
    _typed(ledger.opportunity_counts,(b,8),jnp.int32)
    return b


def _vector(value,b):
    if type(value) is Opportunity:value=int(value)
    value=public_mon_state._int32(value)
    if value.shape not in ((),(b,)):raise ValueError('scalar/B int32 history word required')
    return jnp.broadcast_to(value,(b,))


def _eligible(value,b):
    if type(value) is bool:return jnp.full((b,),value,jnp.bool_)
    if getattr(value,'shape',None) not in ((),(b,)) or getattr(value,'dtype',None)!=jnp.dtype(bool):
        raise ValueError('actual opportunity eligibility must be bool scalar/B')
    return jnp.broadcast_to(value,(b,))


def _streams(streams,b):
    try:value=_vector(streams,b)
    except (TypeError,ValueError):return jnp.zeros(b,jnp.int32),jnp.zeros(b,jnp.bool_)
    sorted_ids=jnp.sort(value)
    unique=jnp.all(sorted_ids[1:]!=sorted_ids[:-1])
    return value,(value>0)&unique


def _sane(ledger):
    _,streams=_streams(ledger.stream_id,len(ledger.stream_id))
    # Bound the SUM without int32 wrap, even for forged multi-column maxima.
    remaining=jnp.maximum(ledger.attempt_count,0);counts_fit=jnp.ones_like(streams)
    for index in range(8):
        value=jnp.maximum(ledger.opportunity_counts[:,index],0)
        counts_fit &= value<=remaining
        remaining=jnp.maximum(remaining-value,0)
    last=((ledger.attempt_count==0)&(ledger.last_opportunity==0))|((ledger.attempt_count>0)&
        (((ledger.last_opportunity>=1)&(ledger.last_opportunity<=8))|(ledger.last_opportunity==-1)))
    return (ledger.initialized&streams&(ledger.frame_serial>=0)&(ledger.event_serial>=0)
        &((ledger.frame_serial==0)|(ledger.event_serial>0))
        &(ledger.sticky_refusal>=0)&(ledger.attempt_count>=0)&counts_fit&last
        &(ledger.event_serial<=ledger.attempt_count)&(ledger.frame_serial<=ledger.attempt_count)
        &jnp.all((ledger.opportunity_counts>=0)&(ledger.opportunity_counts<=ledger.attempt_count[:,None]),axis=1)
        &jnp.all((ledger.observed_relevant_flags>=0)&(ledger.observed_relevant_flags<1<<len(hooks.FAMILIES)),axis=1)
        &jnp.all((ledger.observed_classes>=0)&(ledger.observed_classes<32),axis=1))


def _observe(state,data):
    """All-party namespace membership; only ACTIVE effects are observations."""
    b=state.active.shape[0];batch=jnp.arange(b)[:,None];sides=jnp.arange(2)[None,:]
    active=jnp.clip(state.active,0,5)
    valid=jnp.all((state.active>=0)&(state.active<6),axis=1)
    flags=jnp.zeros((b,2),jnp.int32);classes=jnp.zeros((b,2),jnp.int32);seen=[]
    for field,kind in (('ability','abilities'),('item','items'),('species','species')):
        values=getattr(state,field);rows=getattr(data,kind);n=len(rows.manifest_known)
        safe=jnp.clip(values,0,n-1)
        membership=(values>=0)&(values<n)&rows.manifest_known[safe]
        valid &= jnp.all(membership,axis=(1,2))
        ids=values[batch,sides,active];ids_safe=jnp.clip(ids,0,n-1)
        present=membership[batch,sides,active]
        if field=='species':valid &= jnp.all(ids>0,axis=1)
        flag=rows.relevant_flags[ids_safe];cls=rows.classification[ids_safe]
        # UNKNOWN IDs/types never enter the absence class through clipping.
        row_ok=present&(cls>=hooks.ABSENT)&(cls<=hooks.LEFTOVERS_TRANSPORT_CANDIDATE)&(flag>=0)&(flag<1<<len(hooks.FAMILIES))
        valid &= jnp.all(row_ok,axis=1)
        flags |= jnp.where(row_ok,flag,0)
        classes |= jnp.where(row_ok,1<<jnp.clip(cls,0,4),1<<hooks.UNSUPPORTED)
        seen.append((jnp.arange(n)[None,None,:]==ids[:,:,None])&present[:,:,None])
    # Padding is not a physical member carrying an ability or item.
    valid &= jnp.all((state.species>0)|((state.ability==0)&(state.item==0)),axis=(1,2))
    return valid,flags,classes,seen


def initialize_device(state,stream_ids,tables):
    """Trusted single owning-constructor boundary; NOT birth attestation.

    Bare reinitialization/copies cannot establish lifetime uniqueness; RootSession
    must own that contract. Duplicate/nonpositive streams are explicitly refused.
    """
    _tables(tables);b=_state(state);streams,stream_ok=_streams(stream_ids,b)
    member,flags,classes,seen=_observe(state,tables.hooks)
    refusal=jnp.where(stream_ok,0,NAMESPACE)|jnp.where(member,0,MEMBERSHIP)
    observed=stream_ok[:,None]
    blank=lambda n:jnp.zeros((b,2,n),jnp.bool_)
    return Ledger(tables.binding,tables.hooks.binding,tables.hooks.engine_binding,
        tables.hooks.rules_binding,tables.hooks.manifest_binding,streams,
        jnp.zeros(b,jnp.int32),jnp.zeros(b,jnp.int32),stream_ok,refusal,
        jnp.where(observed,flags,0),jnp.where(observed,classes,0),
        *(value&observed[:,:,None] for value in seen),blank(len(tables.hooks.moves.manifest_known)),
        jnp.zeros((b,8),jnp.int32),jnp.zeros(b,jnp.int32),jnp.zeros(b,jnp.int32))


def retain_device(ledger,state,stream_ids,frame_serial,event_serial,opportunity,tables,*,
                  eligible,selected_move=None,actor_side=None):
    """Observe an ACTUAL pre-hook/setter/post-write opportunity, including no-op.

    Sequence is same-frame event+1 or next-frame event1, per eligible row.
    Malformed attempts latch a refusal and bounded attempt/opportunity counters;
    they cannot advance stamps or replace prior observations. No last_move or
    end-state diff is used. MOVE_DECLARE requires explicit actual move/actor.
    """
    _tables(tables);b=_ledger(ledger,tables)
    if _state(state)!=b:raise ValueError('same physical history batch required')
    eligible=_eligible(eligible,b)
    try:
        streams=_vector(stream_ids,b);stream_ok=streams>0
    except (TypeError,ValueError):
        streams=jnp.zeros(b,jnp.int32);stream_ok=jnp.zeros(b,jnp.bool_)
    # The retained original IDs are unique (_sane). Exact row matching rejects
    # any eligible foreign duplicate without letting an inactive bad input
    # poison a valid neighbor. Two accepted rows therefore cannot alias.
    namespace=stream_ok&(streams==ledger.stream_id)
    for name,value in zip(GLOBALS,(tables.binding,tables.hooks.binding,tables.hooks.engine_binding,
            tables.hooks.rules_binding,tables.hooks.manifest_binding)):
        namespace &= jnp.all(getattr(ledger,name)==value)
    malformed=jnp.zeros(b,jnp.bool_)
    try:frame,event,code=(_vector(x,b) for x in (frame_serial,event_serial,opportunity))
    except (TypeError,ValueError):
        frame,event,code=(jnp.zeros(b,jnp.int32) for _ in range(3));malformed=jnp.ones(b,jnp.bool_)
    counts=ledger.opportunity_counts
    sane=_sane(ledger)
    malformed |= ~sane
    same=(frame==ledger.frame_serial)&(ledger.event_serial<MAX_SERIAL)&(event==ledger.event_serial+1)
    next_frame=(ledger.frame_serial<MAX_SERIAL)&(frame==ledger.frame_serial+1)&(event==1)
    order=(frame>=0)&(event>0)&(same|next_frame)
    code_ok=(code>=1)&(code<=8)
    slot=jnp.clip(code-1,0,7);rows=jnp.arange(b)
    overflow=(ledger.attempt_count==MAX_SERIAL)|(counts[rows,slot]==MAX_SERIAL)
    overflow |= (ledger.event_serial==MAX_SERIAL)&(frame==ledger.frame_serial)
    overflow |= (ledger.frame_serial==MAX_SERIAL)&(frame!=ledger.frame_serial)
    member,flags,classes,seen=_observe(state,tables.hooks)
    move_site=code==int(Opportunity.MOVE_DECLARE)
    move_bad=jnp.zeros(b,jnp.bool_);move_seen=jnp.zeros_like(ledger.seen_moves)
    if selected_move is None or actor_side is None:
        move_bad=move_site
    else:
        try:move,side=(_vector(x,b) for x in (selected_move,actor_side))
        except (TypeError,ValueError):move,side=jnp.zeros(b,jnp.int32),jnp.full(b,-1,jnp.int32)
        static=tables.hooks.moves;n=len(static.manifest_known);safe=jnp.clip(move,0,n-1)
        move_ok=(move>0)&(move<n)&(side>=0)&(side<2)&static.manifest_known[safe]
        cls=static.classification[safe];flag=static.relevant_flags[safe]
        move_ok &= (cls>=hooks.NO_RELEVANT_HOOKS)&(cls<=hooks.LEFTOVERS_TRANSPORT_CANDIDATE)&(flag>=0)&(flag<1<<len(hooks.FAMILIES))
        move_bad=move_site&~move_ok
        use=move_site&move_ok
        address=(rows,jnp.clip(side,0,1))
        flags=flags.at[address].set(flags[address]|jnp.where(use,flag,0))
        classes=classes.at[address].set(classes[address]|jnp.where(use,1<<jnp.clip(cls,0,4),0))
        move_seen=(jnp.arange(2)[None,:,None]==side[:,None,None])&(jnp.arange(n)[None,None,:]==move[:,None,None])&use[:,None,None]
    failure=(jnp.where(namespace,0,NAMESPACE)|jnp.where(malformed|~code_ok,MALFORMED,0)
        |jnp.where(order,0,ORDER)|jnp.where(overflow,OVERFLOW,0)
        |jnp.where(member,0,MEMBERSHIP)|jnp.where(move_bad,SELECTED_MOVE,0))
    # Invalid membership can still conservatively stick an observed known Lum;
    # foreign/misordered frames cannot attach observations to this owner.
    observe=eligible&namespace&~malformed&code_ok&order&~overflow
    advance=observe&member&~move_bad
    expand=lambda mask:mask[:,None]
    counters=counts.at[rows,slot].set(jnp.where(eligible&code_ok,
        jnp.minimum(counts[rows,slot],MAX_SERIAL-1)+1,counts[rows,slot]))
    fields=dict(sticky_refusal=ledger.sticky_refusal|jnp.where(eligible,failure,0),
        frame_serial=jnp.where(advance,frame,ledger.frame_serial),event_serial=jnp.where(advance,event,ledger.event_serial),
        observed_relevant_flags=ledger.observed_relevant_flags|jnp.where(expand(observe),flags,0),
        observed_classes=ledger.observed_classes|jnp.where(expand(observe),classes,0),
        opportunity_counts=counters,
        attempt_count=jnp.where(eligible,jnp.minimum(ledger.attempt_count,MAX_SERIAL-1)+1,ledger.attempt_count),
        last_opportunity=jnp.where(eligible,jnp.where(code_ok,code,-1),ledger.last_opportunity))
    for name,value in zip(('seen_abilities','seen_items','seen_species','seen_moves'),(*seen,move_seen)):
        fields[name]=getattr(ledger,name)|(value&observe[:,None,None])
    return ledger._replace(**fields)


def report_device(ledger,tables):
    """Only partial transport consistency/counts. pending is ALWAYS nonzero."""
    _tables(tables);b=_ledger(ledger,tables)
    same=jnp.ones(b,jnp.bool_)
    for name,value in zip(GLOBALS,(tables.binding,tables.hooks.binding,tables.hooks.engine_binding,
            tables.hooks.rules_binding,tables.hooks.manifest_binding)):
        same &= jnp.all(getattr(ledger,name)==value)
    return Report(_sane(ledger)&(ledger.sticky_refusal==0)&same,
        ledger.sticky_refusal,ledger.observed_relevant_flags,ledger.observed_classes,
        ledger.opportunity_counts,ledger.attempt_count,jnp.full(b,PENDING_MASK,jnp.int32))


def initialize(state,stream_ids,*,prepared):
    _host();tables=check_prepared(prepared)
    b=_state(state)
    streams=np.asarray(public_mon_state._int32(stream_ids))
    if streams.shape!=(b,) or np.any(streams<=0) or len(np.unique(streams))!=b:
        raise ValueError('unique positive physical stream IDs required')
    return initialize_device(state,stream_ids,tables)


def retain(ledger,state,stream_ids,frame_serial,event_serial,opportunity,*,prepared,
           eligible,selected_move=None,actor_side=None):
    _host()
    return retain_device(ledger,state,stream_ids,frame_serial,event_serial,opportunity,
        check_prepared(prepared),eligible=eligible,selected_move=selected_move,actor_side=actor_side)


def report(ledger,*,prepared):
    _host()
    return report_device(ledger,check_prepared(prepared))
