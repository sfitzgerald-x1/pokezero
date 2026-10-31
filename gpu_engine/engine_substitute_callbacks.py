"""Literal successful Substitute self-write; failures remain unrepresented.

Kind25 is [kind, OLD_SUB_HP, NEW_SUB_HP, side, physical, OLD_HP, NEW_HP,
MAX_HP, OLD_STATUS, canonical engine move]. Values come from the SAME actual
actor write, not from a later state diff. Effective Gen3 inherits Substitute
condition.onStart ('-start') then move.onHit/directDamage ('-damage').

Diagnostic mode always retains UNKNOWN256. PRIVATE registered mode can omit
only its own newly arising latch for a valid open accounting context; the
actual Site11 ticket/span and BOTH original owner drains remain required.
There is no damage formula, original-physics or global coverage claim here.
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
from . import engine_public_events as public, engine_lifecycle_callbacks as life

SUBSTITUTE_CREATED=25
UNKNOWN_SUBSTITUTE=256
SCHEMA='pokezero.substitute_literal_write.v1'
_COMPILED=weakref.WeakKeyDictionary()

class Data(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    move_id: object

class Context(NamedTuple):
    data: Data

class RegisteredContext(NamedTuple):
    data: Data
    accounting: object

class Write(NamedTuple):
    old_hp: object
    new_hp: object
    maximum: object
    status: object
    old_sub_hp: object
    new_sub_hp: object

@dataclass(frozen=True,eq=False)
class Compiled:
    device: Data


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked Substitute boundary cannot be traced')


def compile_substitute(*,engine_catalog,metadata):
    _host()
    public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    from .engine_private_team import engine_metadata_binding
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):
        raise ValueError('original Substitute engine metadata required')
    ids=[i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='substitute']
    if len(ids)!=1 or ids[0]<=0 or engine_catalog.engine_table.effects is None:
        raise ValueError('one canonical kernel Substitute row required')
    move=ids[0]
    flags=np.asarray(engine_catalog.engine_table.effects.substitute)
    if flags.shape!=(len(engine_catalog.engine_table.power),) or not flags[move] or np.count_nonzero(flags)!=1:
        raise ValueError('exact canonical Substitute effect column required')
    if engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=0:
        raise ValueError('original public self-target Substitute mapping required')
    raw=dict(schema=SCHEMA,engine=engine_catalog.engine_binding,source=engine_catalog.source_binding,
        moves=engine_catalog.move_binding,producer=engine_catalog.binding,canonical_move=move)
    binding=np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)
    data=Data(jnp.asarray(binding),jnp.asarray(engine_catalog.engine_binding,jnp.uint32),
        jnp.asarray(engine_catalog.source_binding,jnp.uint32),jnp.asarray(engine_catalog.move_binding,jnp.uint32),
        jnp.asarray(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog);return result


def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):
        raise ValueError('original immutable Substitute compiler required')
    c=r[2];public.check_catalog(c,c.engine_table,expected_binding=c.binding,
        expected_engine_binding=np.asarray(c.engine_binding,np.uint32))
    return compiled.device


def _shape(a,shape,dtype):
    if getattr(a,'shape',None)!=shape or getattr(a,'dtype',None)!=jnp.dtype(dtype):
        raise ValueError('Substitute exact device shape/dtype required')


def validate_write_device(write,b):
    if type(write) is not Write:raise ValueError('literal Substitute Write required')
    for x in write:_shape(x,(b,),jnp.int32)
    return ((write.old_hp>write.new_hp)&(write.new_hp>0)&(write.maximum>=write.old_hp)
        &(write.maximum<=1_000_000)&(write.status>=0)&(write.status<=6)
        &(write.old_sub_hp==0)&(write.new_sub_hp>0)&(write.new_sub_hp==write.old_hp-write.new_hp))


def _check(tape,data):
    if type(tape) is not life.LifecycleTape or type(data) is not Data:
        raise ValueError('typed lifecycle tape and Substitute Data required')
    for x in data[:5]:_shape(x,(8,),jnp.uint32)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        _shape(getattr(tape,name),(8,),jnp.uint32)
    _shape(data.move_id,(),jnp.int32)
    return ((data.move_id>0)&jnp.all(data.engine_binding==tape.engine_binding)
        &jnp.all(data.source_binding==tape.source_binding)&jnp.all(data.move_binding==tape.move_binding)
        &jnp.all(data.producer_binding==tape.producer_binding))


def record_device(tape,side,physical,move,write,context,*,eligible):
    registered=type(context) is RegisteredContext
    if type(context) not in (Context,RegisteredContext):raise ValueError('typed Substitute context required')
    b=tape.count.shape[0]
    for x in (side,physical,move):_shape(x,(b,),jnp.int32)
    _shape(eligible,(b,),jnp.bool_)
    valid=validate_write_device(write,b)&_check(tape,context.data)&life._valid(tape.cursor)
    rows=jnp.arange(b);safe=jnp.clip(side,0,1)
    valid &= ((side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,safe])
        &(move==context.data.move_id)&(tape.cursor.phase==0))
    if registered:
        from . import engine_execution_completeness as execution
        actual=execution._context(tape)
        if actual is None:valid &= False
        else:
            valid &= execution._validate(actual.ledger,context.accounting)&execution._tape(actual.ledger,tape,context.accounting)
            valid &= actual.ledger.opened&(actual.ledger.refusal==0)
            for name in ('engine_binding','source_binding','producer_binding','inventory_binding'):
                valid &= jnp.all(getattr(actual.tables,name)==getattr(context.accounting,name))
    room=tape.count<tape.words.shape[1]
    tape=public._append(tape,SUBSTITUTE_CREATED,write.old_sub_hp,write.new_sub_hp,side,physical,
        (write.old_hp,write.new_hp,write.maximum,write.status,move),eligible&valid)
    return tape._replace(refusal=tape.refusal|jnp.where(eligible&(~valid|~room|~jnp.asarray(registered)),UNKNOWN_SUBSTITUTE,0))
