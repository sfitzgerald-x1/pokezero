"""Public item facts for the incremental observation input port.

Original assignment, last reveal and confirmed current possession are separate
ledgers. This ports explicit item/consumption/Trick facts, not non-proc inference,
Choice lock, residual item credit, or complete raw-line consumer coverage.
"""
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.belief import _TRICK_EFFECT
from pokezero.dex import normalize_id
from . import public_mon_state as mon
from .public_abilities import _bool
from .public_moves import _static_category


SCHEMA = 'pokezero.public_items.v1'
DIRECT, TRICK_RECEIVE, OTHER_RECEIVE, INLINE, CONSUME, REMOVE, OTHER_END = range(7)


class ItemTables(NamedTuple):
    predicate: jax.Array
    category: jax.Array
    source_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class ItemCatalog:
    names: tuple[str, ...]
    ids: Mapping[str, int]
    device: ItemTables

    def item_id(self, name):
        key = normalize_id(name)
        if not key or key not in self.ids:raise ValueError('item outside frozen public item namespace')
        return self.ids[key]


def compile_items(source, vocabulary, *, item_names=()):
    names = ('',) + tuple(sorted(({normalize_id(v.item) for rows in source.variants for v in rows if v.item}
                                 | {normalize_id(name) for name in item_names}) - {''}))
    predicate = [source.predicate_id('item', name) if name else 0 for name in names]
    category = [_static_category(vocabulary, 'belief:possible_item:'+name) if name else 0 for name in names]
    payload = dict(schema=SCHEMA,names=names,predicate=predicate,category=category,
        source_binding=np.asarray(source.device.binding).tolist(),tokens=vocabulary.tokens,
        aliases=dict(vocabulary.aliases),oov_buckets=vocabulary.oov_buckets)
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return ItemCatalog(names,MappingProxyType({name:i for i,name in enumerate(names) if i}),
        ItemTables(jnp.asarray(predicate,jnp.int32),jnp.asarray(category,jnp.int32),source.device.binding,binding))


class ItemState(NamedTuple):
    binding: jax.Array
    species_binding: jax.Array
    revealed: jax.Array              # [B,2,6], latest named item, possibly already gone
    original: jax.Array              # certain original assignment after a mutation
    mutated: jax.Array
    removed: jax.Array
    current: jax.Array               # positive only on audited Trick receipt
    pending: jax.Array               # [B], contiguous Trick transaction
    pending_side: jax.Array          # [B,2], source then target
    pending_slot: jax.Array
    pending_encounter: jax.Array
    pending_original_known: jax.Array


def initialize(batch,tables,*,species_binding=None):
    if type(batch) is not int or batch<=0:raise ValueError('positive item-state batch required')
    shape=(batch,2,6);z=lambda:jnp.zeros(shape,jnp.int32)
    return ItemState(jnp.broadcast_to(tables.binding,(batch,8)),mon.namespace_binding(batch,species_binding),
        z(),z(),jnp.zeros(shape,bool),jnp.zeros(shape,bool),z(),jnp.zeros(batch,bool),
        jnp.full((batch,2),-1,jnp.int32),jnp.full((batch,2),-1,jnp.int32),
        jnp.zeros((batch,2),jnp.int32),jnp.zeros((batch,2),bool))


def _address(state,mons,tables,side,slot,encounter):
    row,seat,target,ok=mon._address(mons,side,slot);encounter=mon._int32(encounter)
    ok &= (encounter>0)&(mons.encounter[row,seat,target]==encounter)
    ok &= jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==mons.species_binding,axis=-1)
    return (row,seat,target),ok


def begin_event(state,mons,tables,*,item_event=False,trick=False,
                actor_side=-1,actor_slot=-1,actor_encounter=0,
                target_side=-1,target_slot=-1,target_encounter=0,valid=True):
    """Call once for EVERY raw event before its item operations.

    Any non-item line clears a prior pairing. A supported Trick activation
    captures each live participant's pre-transfer original-assignment status.
    Missing live participants refuse; a raw adapter must not fabricate them.
    """
    item_event,trick,valid=(_bool(x) for x in (item_event,trick,valid))
    actor,aa=_address(state,mons,tables,actor_side,actor_slot,actor_encounter)
    target,ta=_address(state,mons,tables,target_side,target_slot,target_encounter)
    live_a=(mons.active[actor[:2]]==actor[2])&~mons.fainted[actor]
    live_t=(mons.active[target[:2]]==target[2])&~mons.fainted[target]
    bound=jnp.all(state.binding==tables.binding,axis=-1)&jnp.all(state.species_binding==mons.species_binding,axis=-1)
    admitted=bound&~(item_event&trick)&(~trick|(aa&ta&live_a&live_t&(actor[1]!=target[1])))
    apply=valid&admitted
    clear=apply&~item_event
    arm=apply&trick
    choose=lambda old,new:jnp.where(arm[:,None],new,jnp.where(clear[:,None],jnp.zeros_like(old),old))
    return state._replace(pending=jnp.where(clear,False,state.pending)|arm,
        pending_side=choose(state.pending_side,jnp.stack((actor[1],target[1]),axis=-1)),
        pending_slot=choose(state.pending_slot,jnp.stack((actor[2],target[2]),axis=-1)),
        pending_encounter=choose(state.pending_encounter,jnp.stack((mons.encounter[actor],mons.encounter[target]),axis=-1)),
        pending_original_known=choose(state.pending_original_known,jnp.stack((~state.mutated[actor],~state.mutated[target]),axis=-1))),admitted|~valid


def record(state,mons,tables,*,side,slot,encounter,item_id,item_binding,operation=DIRECT,valid=True):
    """Apply one lexical item fact; raw and explicit facts may share a line."""
    address,ok=_address(state,mons,tables,side,slot,encounter)
    if getattr(item_binding,'shape',None)!=(8,) or getattr(item_binding,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('item producer requires eight uint32 namespace words')
    ok &= jnp.all(item_binding==tables.binding)
    item_id,operation=(mon._int32(x) for x in (item_id,operation));valid=_bool(valid)
    ok &= (item_id>0)&(item_id<tables.predicate.shape[0])&(operation>=DIRECT)&(operation<=OTHER_END)
    apply=ok&valid
    mutation=(operation==TRICK_RECEIVE)|(operation==OTHER_RECEIVE)|(operation==REMOVE)|(operation==OTHER_END)
    remove=(operation==REMOVE)|(operation==CONSUME)
    reset=(operation==TRICK_RECEIVE)|(operation==OTHER_RECEIVE)|(operation==OTHER_END)
    current_change=remove|reset
    old_original=state.original[address]
    own_removal=(operation==REMOVE)&~state.mutated[address]&(old_original==0)
    result=state._replace(revealed=mon._write(state.revealed,address,item_id,apply),
        original=mon._write(state.original,address,item_id,apply&own_removal),
        mutated=mon._write(state.mutated,address,True,apply&mutation),
        removed=mon._write(state.removed,address,remove,apply&(remove|reset)),
        current=mon._write(state.current,address,jnp.where(operation==TRICK_RECEIVE,item_id,0),apply&current_change))
    row,seat,index=address
    subject=(state.pending_side==seat[:,None])&(state.pending_slot==index[:,None])&(state.pending_encounter==mon._int32(encounter)[...,None])
    partner_index=jnp.where(subject[:,0],1,0)
    partner_side=state.pending_side[row,partner_index];partner_slot=state.pending_slot[row,partner_index]
    partner_encounter=state.pending_encounter[row,partner_index]
    partner,pa=_address(result,mons,tables,partner_side,partner_slot,partner_encounter)
    original_known=state.pending_original_known[row,partner_index]
    credit=apply&(operation==TRICK_RECEIVE)&state.pending&jnp.any(subject,axis=-1)&pa&original_known&(result.original[partner]==0)
    return result._replace(original=mon._write(result.original,partner,item_id,credit)),ok|~valid


def matching_item(state,*,item_belief_narrowing=False):
    """The source's assignment-only key, separate from current possession."""
    narrow=_bool(item_belief_narrowing)
    return jnp.where(state.mutated,jnp.where(narrow,state.original,0),state.revealed)


@dataclass(frozen=True)
class ItemProjection:
    raw_line: str
    item_event: bool
    trick_idents: tuple[str,str] | None
    facts: tuple[tuple[str,int,int], ...]  # ident, operation, frozen item ID
    item_binding: tuple[int, ...]
    full_line_consumers_complete: bool = False


def project_line(line,*,items):
    """Decode source lexical item facts; [of] never owns inline item procs."""
    if not isinstance(line,str) or not line.startswith('|'):raise ValueError('public protocol line required')
    parts=line.split('|');kind=parts[1] if len(parts)>1 else ''
    if kind=='request':raise ValueError('private requests cannot enter public item inputs')
    event=reference._public_event_from_line(line);facts=[];trick=None
    if kind=='-activate' and _TRICK_EFFECT.search(line):
        actor=parts[2].strip() if len(parts)>2 else ''
        of_match=re.search(r'\[of\] ([^|\]]+)',line)
        target=of_match.group(1).strip() if of_match else ''
        if reference._slot_from_ident(actor) in ('p1','p2') and reference._slot_from_ident(target) in ('p1','p2') and reference._slot_from_ident(actor)!=reference._slot_from_ident(target):
            trick=(actor,target)
    raw_item=event.primary if kind=='-enditem' and event.primary else line.split('[from] item:',1)[1].split('|')[0].strip() if '[from] item:' in line else None
    holder=event.target_ident or event.actor_ident
    if raw_item and holder:
        operation=INLINE
        if kind=='-enditem':
            operation=REMOVE if '[from] move: Knock Off' in line or '[from] move: Trick' in line else OTHER_END if '[from] move:' in line else CONSUME
        facts.append((holder,operation,items.item_id(raw_item)))
    if kind=='-item' and event.primary and event.target_slot:
        operation=TRICK_RECEIVE if '[from] move: Trick' in line else OTHER_RECEIVE if '[from] move:' in line else DIRECT
        facts.append((event.target_ident,operation,items.item_id(event.primary)))
    return ItemProjection(line,kind in ('-item','-enditem'),trick,tuple(facts),
        tuple(int(x) for x in np.asarray(items.device.binding)))
