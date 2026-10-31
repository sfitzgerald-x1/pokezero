"""Prospective SAME-call positive Double-Edge target/recoil source pair.

A pending Site14 is entered after the captured target write and before recoil
risk observation. It is NOT emission/consumption credit: only the later literal
nonlethal actor HP write can close its span. Missing/blocked/Rock Head/KO or
interleaved callbacks stay unknown. No recoil amount or damage is reconstructed.
Device values are trusted owning-call inputs, not copied-carry attestation.
"""
from dataclasses import dataclass
import hashlib,json,weakref
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life
from . import engine_direct_hit_controls as direct
from .engine_private_team import engine_metadata_binding

RECOIL_HP=28
UNKNOWN_RECOIL=256
_COMPILED=weakref.WeakKeyDictionary()

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    rock_head:object

@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

class Context(NamedTuple):
    data:Data
    direct:direct.RegisteredContext

class Pending(NamedTuple):
    context:Context
    ticket:object
    captured:object
    recoil:object
    ability:object
    old_hp:object
    maximum:object
    status:object
    requested:object

class Before(NamedTuple):
    actor_physical:object
    target_physical:object
    ability:object
    old_hp:object
    maximum:object
    status:object

class Write(NamedTuple):
    actor_physical:object
    target_physical:object
    ability:object
    old_hp:object
    new_hp:object
    maximum:object
    status:object


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked recoil compiler must remain outside outer JIT')

def compile_recoil(*,engine_catalog,metadata):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('same original recoil metadata required')
    names=metadata['vocabulary']['moves'];ids=[i for i,n in enumerate(names) if normalize_id(n)=='doubleedge'];table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.damage_moves is None:raise ValueError('canonical Double-Edge source row required')
    move=ids[0];rock=np.array([normalize_id(n)=='rockhead' for n in metadata['vocabulary']['abilities']],bool)
    if (int(table.power[move])!=120 or int(table.damage_moves.recoil_den[move])!=3
        or bool(table.damage_moves.struggle[move]) or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1
        or np.asarray(table.damage_moves.rock_head).shape!=rock.shape or not np.array_equal(table.damage_moves.rock_head,rock)):
        raise ValueError('original Double-Edge120/recoil3/target/RockHead columns required')
    raw=dict(schema='pokezero.literal_doubleedge_pair.v1',move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding,rock_head=rock.tolist())
    words=lambda x:jnp.asarray(x,jnp.uint32)
    data=Data(words(np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32),jnp.array(rock))
    out=Compiled(data);_COMPILED[out]=(data,tuple(data),engine_catalog);return out

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original immutable recoil compiler required')
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,expected_engine_binding=compiled.device.engine_binding)
    return compiled.device

def _vector(x,b,dtype=jnp.int32):
    from . import engine_execution_completeness as execution
    execution._shape(x,(b,),dtype);return x

def _bound(tape,context):
    from . import engine_execution_completeness as execution
    if type(context) is not Context or type(context.data) is not Data or type(context.direct) is not direct.RegisteredContext:raise ValueError('original typed recoil/direct context required')
    data=context.data
    direct._check(tape,context.direct.data)
    for a in data[:5]:execution._shape(a,(8,),jnp.uint32)
    execution._shape(data.move_id,(),jnp.int32)
    execution._shape(data.rock_head,(len(data.rock_head),),jnp.bool_)
    if len(data.rock_head)<1:raise ValueError('nonempty actual ability namespace required')
    original=execution._context(tape)
    if original is None:return jnp.zeros_like(tape.count,bool)
    for tables in (context.direct.accounting,original.tables):
        if type(tables) is not execution.Tables:raise ValueError('original accounting tables required')
        for value in tables:execution._shape(value,(8,),jnp.uint32)
    good=execution._validate(original.ledger,context.direct.accounting)&execution._tape(original.ledger,tape,context.direct.accounting)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= jnp.all(getattr(data,n)==getattr(tape,n))&jnp.all(getattr(context.direct.data,n)==getattr(data,n))
    good &= jnp.all(original.tables.inventory_binding==context.direct.accounting.inventory_binding)
    return good

def matches_pending(tape,pending):
    """Read-only prospective obligation. No flag/risk is ever cleared."""
    from . import engine_execution_completeness as execution
    from . import engine_kernel_frame_risk as risks
    if type(pending) is not Pending or type(pending.captured) is not risks.CapturedDirectHit:raise ValueError('typed SAME-call pending recoil required')
    b=tape.count.shape[0];rows=jnp.arange(b);context=pending.context;data=context.data;capture=pending.captured
    for value in pending[3:8]:_vector(value,b)
    _vector(pending.requested,b,jnp.bool_)
    bound=_bound(tape,context);original=execution._context(tape)
    if original is None:return jnp.zeros(b,bool)
    ledger=original.ledger;slot,same=execution._ticket(ledger,pending.ticket)
    dslot,dsame=execution._ticket(ledger,capture.ticket)
    if type(capture.context) is not direct.RegisteredContext or type(capture.write) is not direct.HitWrite:raise ValueError('original captured direct write required')
    direct._check(tape,capture.context.data)
    for a,z in zip(capture.context.accounting,context.direct.accounting):
        execution._shape(a,(8,),jnp.uint32);same &= jnp.all(a==z)
    for n in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
        same &= jnp.all(getattr(capture.context.data,n)==getattr(context.direct.data,n))
    actor=_vector(capture.actor,b);ap=_vector(capture.actor_physical,b);target=_vector(capture.target_physical,b)
    move=_vector(capture.move,b);safe=jnp.clip(actor,0,1)
    ability=pending.ability;n=len(data.rock_head)
    good=(bound&same&dsame&pending.requested&ledger.opened&(ledger.refusal==0)
        &((tape.refusal&~public.UNINSTRUMENTED)==0)&(move==data.move_id)&(data.move_id>0)
        &(ledger.site[rows,slot]==int(execution.Site.RECOIL_HP))&(ledger.stage[rows,slot]==1)
        &(ledger.start[rows,slot]==tape.count)&(ledger.side[rows,slot]==actor)
        &(ledger.stage[rows,dslot]==2)&(ledger.end[rows,dslot]==tape.count)
        &(ledger.site[rows,dslot]==int(execution.Site.DIRECT_HIT))
        &(actor>=0)&(actor<2)&(ap==tape.cursor.physical_active[rows,safe])
        &(target==tape.cursor.physical_active[rows,1-safe])
        &(ability>=0)&(ability<n)&~data.rock_head[jnp.clip(ability,0,n-1)]
        &(pending.recoil>0)&(pending.old_hp>pending.recoil)&(pending.maximum>=pending.old_hp)&(pending.maximum<=1_000_000)
        &(pending.status>=0)&(pending.status<=6)
        &execution._direct_hit_span_matches(tape,capture.ticket,actor,ap,target,move,capture.write))
    start=ledger.start[rows,dslot];span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    good &= jnp.all(~span|((ledger.claimed==dslot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return good

def prepare_device(tape,captured,recoil,ability,hp,maximum,status,context,*,eligible):
    from . import engine_execution_completeness as execution
    b=tape.count.shape[0]
    for x in (recoil,ability,hp,maximum,status):_vector(x,b)
    _vector(eligible,b,jnp.bool_)
    requested=eligible&(captured.move==context.data.move_id)
    tape,ticket=execution.open_attempt(tape,execution.Site.RECOIL_HP,captured.actor,active=requested)
    pending=Pending(context,ticket,captured,recoil,ability,hp,maximum,status,requested)
    good=matches_pending(tape,pending)
    original=execution._context(tape)
    if original is not None:
        ledger=execution.unhandled(original.ledger,ticket,active=requested&~good,tables=original.tables)
        tape=execution._put_context(tape,ledger,original)
    return tape._replace(refusal=tape.refusal|jnp.where(requested&~good,UNKNOWN_RECOIL,0)),pending

def matches_before(tape,pending,write):
    if type(write) not in (Before,Write):raise ValueError('literal recoil BEFORE/AFTER write required')
    b=tape.count.shape[0]
    for x in write:_vector(x,b)
    c=pending.captured
    return (matches_pending(tape,pending)&(write.actor_physical==c.actor_physical)&(write.target_physical==c.target_physical)
        &(write.ability==pending.ability)&(write.old_hp==pending.old_hp)&(write.maximum==pending.maximum)&(write.status==pending.status))

def record_device(tape,pending,write):
    """After SAME actor assignment: capture actual words, never an HP formula."""
    from . import engine_execution_completeness as execution
    good=matches_before(tape,pending,write)&(write.old_hp>write.new_hp)&(write.new_hp>0)&(write.old_hp-write.new_hp==pending.recoil)
    c=pending.captured;b=tape.count.shape[0];rows=jnp.arange(b)
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);index=jnp.clip(tape.count,0,tape.words.shape[1]-1)
    good &= ~inside|jnp.all(tape.switch_details[rows,index]==0,axis=1)
    tape=public._append(tape,RECOIL_HP,c.actor,c.actor_physical,1-c.actor,c.target_physical,
        (c.move,write.old_hp,write.new_hp,write.maximum,write.status),pending.requested&good)
    original=execution._context(tape)
    ledger=execution.unhandled(original.ledger,pending.ticket,active=pending.requested&~good,tables=original.tables)
    ledger=execution.emitted(ledger,pending.ticket,tape,active=pending.requested&good,tables=original.tables)
    tape=execution._put_context(tape,ledger,original)
    return tape._replace(refusal=tape.refusal|jnp.where(pending.requested&(~good|~inside),UNKNOWN_RECOIL,0))
