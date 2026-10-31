"""Literal actual Toxic attempt guards, not reconstruction of old run causes.

Kind26: actor side/physical, target side/physical, canonical move, literal guard
bits, BEFORE/AFTER status, packed literal BEFORE/AFTER counter (four bits each).
A simple actual accuracy miss or plain type immunity emits its lexical line.
A successful primary write closes the exact existing kind9+kind26 pair without
duplicating the status line. Reflected/ability/Substitute/other outcomes remain
UNKNOWN, as do unrepresented later effects.
No damage/probability/status/counter is calculated from final outcomes.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
import weakref
import jax
import jax.numpy as jnp
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life,mechanics as m

TOXIC_ATTEMPT=26
UNKNOWN_TOXIC=256
ACCURACY_HIT,FINAL_HIT,TYPE_IMMUNE,ABILITY_IMMUNE,SUB_BLOCK,LIVE,APPLIED,PROTECTED,ABSORBED,DAMAGE_IMMUNE,SECONDARY=(1<<i for i in range(11))
MISS,PLAIN_TYPE_IMMUNITY,STATUS_WRITE,SUBSTITUTE_BLOCK,ABILITY_BLOCK,EXISTING_STATUS,OTHER=range(1,8)
_COMPILED=weakref.WeakKeyDictionary()

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
class Context(NamedTuple):
    data:Data
class RegisteredContext(NamedTuple):
    data:Data
    accounting:object
class Attempt(NamedTuple):
    accuracy_hit:object
    hit:object
    type_immune:object
    ability_immune:object
    substitute_block:object
    live:object
    applied:object
    protected:object
    absorbed:object
    damage_immune:object
    secondary_active:object
    old_status:object
    new_status:object
    old_counter:object
    new_counter:object
@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

def _shape(a,shape,dtype):
    if getattr(a,'shape',None)!=shape or getattr(a,'dtype',None)!=jnp.dtype(dtype):raise ValueError('Toxic exact shape/dtype required')
def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Toxic boundary cannot be traced')
def compile_toxic(*,engine_catalog,metadata):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    from .engine_private_team import engine_metadata_binding
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('original Toxic metadata required')
    ids=[i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='toxic']
    table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.effects is None:raise ValueError('canonical Toxic row required')
    move=ids[0];status=np.asarray(table.effects.status)
    if (status.shape!=(len(table.power),) or status[move]!=m.TOX or np.count_nonzero(status==m.TOX)!=1
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1
            or int(table.power[move])!=0 or float(table.accuracy[move])!=85 or bool(table.always_hit[move])):
        raise ValueError('effective Gen3 Toxic85/status/target mapping required')
    raw=dict(schema='pokezero.toxic_actual_attempt.v1',move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding)
    binding=np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)
    data=Data(jnp.asarray(binding),jnp.asarray(engine_catalog.engine_binding,jnp.uint32),jnp.asarray(engine_catalog.source_binding,jnp.uint32),
        jnp.asarray(engine_catalog.move_binding,jnp.uint32),jnp.asarray(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog);return result

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original Toxic compiler required')
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32));return compiled.device

def _check(tape,data):
    if type(tape) is not life.LifecycleTape or type(data) is not Data:raise ValueError('typed Toxic tape/Data required')
    for x in data[:5]:_shape(x,(8,),jnp.uint32)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):_shape(getattr(tape,n),(8,),jnp.uint32)
    _shape(data.move_id,(),jnp.int32)
    return (data.move_id>0)&jnp.all(data.engine_binding==tape.engine_binding)&jnp.all(data.source_binding==tape.source_binding)&jnp.all(data.move_binding==tape.move_binding)&jnp.all(data.producer_binding==tape.producer_binding)

def words_device(side,physical,target,move,attempt):
    if type(attempt) is not Attempt:raise ValueError('literal Toxic Attempt required')
    b=len(side)
    for value in (side,physical,target,move):_shape(value,(b,),jnp.int32)
    for value in attempt[:11]:_shape(value,(b,),jnp.bool_)
    for value in attempt[11:]:_shape(value,(b,),jnp.int32)
    flags=jnp.zeros(b,jnp.int32)
    for index,value in enumerate(attempt[:11]):flags|=value.astype(jnp.int32)<<index
    return jnp.stack((jnp.full_like(side,TOXIC_ATTEMPT),side,physical,1-side,target,move,flags,
        attempt.old_status,attempt.new_status,attempt.old_counter|(attempt.new_counter<<4)),axis=1)

def literal_valid(raw):
    flags=raw[:,6];old=raw[:,7];new=raw[:,8];before=raw[:,9]&15;after=raw[:,9]>>4
    applied=(flags&APPLIED)!=0;hit=(flags&FINAL_HIT)!=0;acc=(flags&ACCURACY_HIT)!=0
    secondary=(flags&SECONDARY)!=0
    return ((raw[:,0]==TOXIC_ATTEMPT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)
        &(flags>=0)&(flags<2048)&(old>=0)&(old<=6)&(new>=0)&(new<=6)&(raw[:,9]>=0)&(raw[:,9]<256)
        &(~hit|acc)&(secondary==(hit&((flags&DAMAGE_IMMUNE)==0)))
        &jnp.where(applied,(old==m.NONE)&(new==m.TOX)&(after==0)&hit,
            (old==new)&(before==after)))

def outcome_device(raw):
    f=raw[:,6];present=lambda bit:(f&bit)!=0
    miss=~present(ACCURACY_HIT)&~present(FINAL_HIT)&~present(APPLIED)&~present(ABSORBED)&present(LIVE)
    type_only=present(ACCURACY_HIT)&present(FINAL_HIT)&present(TYPE_IMMUNE)&~present(ABILITY_IMMUNE)
    type_only &= ~present(SUB_BLOCK)&~present(PROTECTED)&~present(ABSORBED)&~present(APPLIED)&present(LIVE)
    type_only &= (raw[:,7]==0)&(raw[:,8]==0)&(raw[:,9]==0)
    return jnp.where(miss,MISS,jnp.where(type_only,PLAIN_TYPE_IMMUNITY,jnp.where(present(APPLIED),STATUS_WRITE,
        jnp.where(present(SUB_BLOCK),SUBSTITUTE_BLOCK,jnp.where(present(ABILITY_IMMUNE),ABILITY_BLOCK,
            jnp.where(raw[:,7]!=0,EXISTING_STATUS,OTHER))))))

def supported_raw(raw):
    code=outcome_device(raw);f=raw[:,6]
    success=(code==STATUS_WRITE)&((f&(ACCURACY_HIT|FINAL_HIT|LIVE|APPLIED|SECONDARY))
        ==(ACCURACY_HIT|FINAL_HIT|LIVE|APPLIED|SECONDARY))
    success &= (f&(TYPE_IMMUNE|ABILITY_IMMUNE|SUB_BLOCK|PROTECTED|ABSORBED|DAMAGE_IMMUNE))==0
    success &= (raw[:,7]==m.NONE)&(raw[:,8]==m.TOX)&(raw[:,9]==0)
    return literal_valid(raw)&((code==MISS)|(code==PLAIN_TYPE_IMMUNITY)|success)

def record_device(tape,side,physical,target,move,attempt,context,*,eligible):
    registered=type(context) is RegisteredContext
    if type(context) not in (Context,RegisteredContext):raise ValueError('typed Toxic context required')
    b=len(tape.count);_shape(eligible,(b,),jnp.bool_);raw=words_device(side,physical,target,move,attempt)
    rows=jnp.arange(b);safe=jnp.clip(side,0,1)
    valid=literal_valid(raw)&(attempt.old_counter>=0)&(attempt.old_counter<=15)&(attempt.new_counter>=0)&(attempt.new_counter<=15)
    valid &= _check(tape,context.data)&life._valid(tape.cursor)
    valid &= (physical==tape.cursor.physical_active[rows,safe])&(target==tape.cursor.physical_active[rows,1-safe])&(move==context.data.move_id)&(tape.cursor.phase==0)
    good=supported_raw(raw)&valid
    if registered:
        from . import engine_execution_completeness as execution
        actual=execution._context(tape)
        if actual is None:good &= False
        else:
            good &= execution._validate(actual.ledger,context.accounting)&execution._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.refusal==0)
            for name in ('engine_binding','source_binding','producer_binding','inventory_binding'):good &= jnp.all(getattr(actual.tables,name)==getattr(context.accounting,name))
    room=tape.count<tape.words.shape[1]
    tape=public._append(tape,TOXIC_ATTEMPT,side,physical,1-side,target,tuple(raw[:,i] for i in range(5,10)),eligible&valid)
    return tape._replace(refusal=tape.refusal|jnp.where(eligible&(~good|~room|~jnp.asarray(registered)),UNKNOWN_TOXIC,0))
