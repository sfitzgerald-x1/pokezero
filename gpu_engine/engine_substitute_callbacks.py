"""Literal Substitute writes and explicit source-bound no-write failures.

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
import subprocess
from pathlib import Path
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
    failure: object = None # explicit original rules, closed by default

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


def compile_substitute(*,engine_catalog,metadata,source_rules=None):
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
    descriptor=None
    if source_rules is not None:
        from . import engine_request_source as request
        origin=request._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original Substitute rules required')
        script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('substitute');
process.stdout.write(JSON.stringify({tryHit:crypto.createHash('sha256').update(m.onTryHit.toString()).digest('hex'),move:{id:m.id,target:m.target,power:m.basePower,accuracy:m.accuracy,category:m.category,volatile:m.volatileStatus},source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
        descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
        if descriptor['tryHit']!='4031f00170707b1b2744ab8781a7f3addf590d47357b4dc7c8e13db1a5825cbb' or descriptor['move']!=dict(id='substitute',target='self',power=0,accuracy=True,category='Status',volatile='substitute'):raise ValueError('effective original Gen3 Substitute failure source required')
        if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Substitute source changed')
        table=engine_catalog.engine_table
        if int(table.power[move])!=0 or not bool(table.always_hit[move]):raise ValueError('installed original Substitute control required')
    raw=dict(schema=SCHEMA,engine=engine_catalog.engine_binding,source=engine_catalog.source_binding,
        moves=engine_catalog.move_binding,producer=engine_catalog.binding,canonical_move=move,failure=descriptor)
    binding=np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)
    data=Data(jnp.asarray(binding),jnp.asarray(engine_catalog.engine_binding,jnp.uint32),
        jnp.asarray(engine_catalog.source_binding,jnp.uint32),jnp.asarray(engine_catalog.move_binding,jnp.uint32),
        jnp.asarray(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32),jnp.array(descriptor is not None,jnp.bool_))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog,source_rules);return result


def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):
        raise ValueError('original immutable Substitute compiler required')
    if r[3] is not None:
        from . import engine_request_source as request
        request._rules_origin(r[3])
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
    _shape(data.move_id,(),jnp.int32);_shape(data.failure,(),jnp.bool_)
    return ((data.move_id>0)&jnp.all(data.engine_binding==tape.engine_binding)
        &jnp.all(data.source_binding==tape.source_binding)&jnp.all(data.move_binding==tape.move_binding)
        &jnp.all(data.producer_binding==tape.producer_binding))


def record_device(tape,side,physical,move,write,context,*,eligible,caller_move=None):
    registered=type(context) is RegisteredContext
    if type(context) not in (Context,RegisteredContext):raise ValueError('typed Substitute context required')
    b=tape.count.shape[0]
    for x in (side,physical,move):_shape(x,(b,),jnp.int32)
    _shape(eligible,(b,),jnp.bool_)
    failure=jnp.zeros(b,bool)
    if caller_move is not None:
        _shape(caller_move,(b,),jnp.int32)
        failure=context.data.failure&failure_write_device(write,b)&(caller_move==move)
    valid=(validate_write_device(write,b)|failure)&_check(tape,context.data)&life._valid(tape.cursor)
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


def failure_write_device(write,b):
    if type(write) is not Write:raise ValueError('literal Substitute Write required')
    for x in write:_shape(x,(b,),jnp.int32)
    return ((write.old_hp>0)&(write.old_hp==write.new_hp)&(write.old_hp<=write.maximum)&(write.maximum<=1_000_000)
        &(write.status>=0)&(write.status<=6)&(write.old_sub_hp>=0)&(write.old_sub_hp<=write.maximum)
        &(write.old_sub_hp==write.new_sub_hp)&((write.old_sub_hp>0)|(write.old_hp*4<=write.maximum)|(write.maximum==1)))


def failure_raw(raw):
    write=Write(raw[:,5],raw[:,6],raw[:,7],raw[:,8],raw[:,1],raw[:,2])
    return (raw[:,0]==SUBSTITUTE_CREATED)&(raw[:,3]>=0)&(raw[:,3]<2)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,9]>0)&failure_write_device(write,len(raw))


def owned_failure(tape,index,*,stage=2):
    """Independent exact claimed Site11 failure after its original MOVE."""
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=jnp.arange(b);idx=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,))
    if c is None or c.risks is None:return jnp.zeros(b,bool)
    l=c.ledger;t=c.risks.tables;at=jnp.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    _shape(t.substitute_failure,(),jnp.bool_);_shape(t.substitute_binding,(8,),jnp.uint32)
    good=t.substitute_failure&jnp.any(t.substitute_binding!=0)&ex._validate(l,c.tables)&ex._tape(l,tape,c.tables)&l.opened&(l.frame>0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        _shape(getattr(tape,name),(8,),jnp.uint32);_shape(getattr(t,name),(8,),jnp.uint32);good &= jnp.all(getattr(tape,name)==getattr(t,name))
    claim=l.count if stage==1 else l.claimed[rows,at];slot=jnp.clip(claim-1,0,l.site.shape[1]-1)
    good &= failure_raw(raw)&(idx>0)&(idx<tape.count)&(claim>0)&(claim<=l.count)&(raw[:,9]<len(t.substitute_candidates))&t.substitute_candidates[jnp.clip(raw[:,9],0,len(t.substitute_candidates)-1)]
    good &= (l.site[rows,slot]==int(ex.Site.SUBSTITUTE_SELF))&(l.side[rows,slot]==raw[:,3])&(l.stage[rows,slot]==stage)&(l.start[rows,slot]==idx)&(l.no_output_kind[rows,slot]==0)&jnp.all(tape.switch_details[rows,at]==0,axis=1)
    good &= jnp.where(stage==1,tape.count==idx+1,l.end[rows,slot]==idx+1)
    prev=jnp.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prev]
    good &= (decl[:,0]==public.MOVE)&(decl[:,1]==raw[:,3])&(decl[:,2]==raw[:,4])&(decl[:,3]==1-raw[:,3])&(decl[:,5]==raw[:,9])&jnp.all(decl[:,6:]==0,axis=1)
    for address in (prev,at):good &= (tape.event_turn[rows,address]==l.turn[rows,slot])&(tape.event_phase[rows,address]==l.phase[rows,slot])&jnp.all(tape.event_generation[rows,address]==l.generation[rows,slot],axis=1)
    if stage==1:
        side=jnp.clip(raw[:,3],0,1)
        good &= life._valid(tape.cursor)&(tape.cursor.phase==0)&(l.turn[rows,slot]==tape.cursor.turn)&(l.phase[rows,slot]==tape.cursor.phase)&jnp.all(l.generation[rows,slot]==tape.cursor.generation,axis=1)&(raw[:,4]==tape.cursor.physical_active[rows,side])&(decl[:,4]==tape.cursor.physical_active[rows,1-side])
    else:good &= (l.claimed[rows,at]==claim)&jnp.all(l.words[rows,at]==raw,axis=1)&(l.event_turn[rows,at]==tape.event_turn[rows,at])&(l.event_phase[rows,at]==tape.event_phase[rows,at])&jnp.all(l.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good
