"""Actual canonical Thunder Wave NONE->PAR input; existing RAW9+26 only.

The status write is made by the original numerical kernel. This observer emits
its literal guard continuation, then requires the same current MOVE/status
span and both original consumers. Every non-success remains UNKNOWN.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
import weakref
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life,mechanics as m
from . import engine_toxic_callbacks as primary,engine_status_callbacks as status

_COMPILED=weakref.WeakKeyDictionary()
Attempt=primary.Attempt  # Existing RAW26 literal guard schema, not Toxic admission.
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
class Context(NamedTuple):
    data:Data
    accounting:object
@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

def _host():
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('checked Thunder Wave compiler cannot be traced')
def compile_thunder_wave(*,engine_catalog,metadata):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    from .engine_private_team import engine_metadata_binding
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('original Thunder Wave metadata required')
    ids=[i for i,name in enumerate(metadata['vocabulary']['moves']) if normalize_id(name)=='thunderwave'];table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.effects is None:raise ValueError('canonical Thunder Wave required')
    move=ids[0]
    if (int(table.power[move])!=0 or float(table.accuracy[move])!=100 or bool(table.always_hit[move])
            or int(table.effects.status[move])!=m.PAR or not bool(table.effects.thunderwave[move])
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1):
        raise ValueError('canonical Gen3 Thunder Wave100/primary PAR/target required')
    payload=dict(schema='pokezero.thunder-wave-actual-par.v1',move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding)
    words=lambda x:j.asarray(x,j.uint32)
    data=Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),j.array(move,j.int32))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog);return result

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original Thunder Wave compiler required')
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32));return compiled.device

def supported_raw(raw):
    # Successful primary application ONLY. Do not generalize the original
    # Toxic validator, or admit its miss/immunity branches under this source.
    flags=primary.ACCURACY_HIT|primary.FINAL_HIT|primary.LIVE|primary.APPLIED|primary.SECONDARY
    return ((raw[:,0]==primary.TOXIC_ATTEMPT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)
        &(raw[:,6]==flags)&(raw[:,7]==m.NONE)&(raw[:,8]==m.PAR)&(raw[:,9]==0))

def _bound(tape,context):
    from . import engine_execution_completeness as ex
    if type(tape) is not life.LifecycleTape or type(context) is not Context or type(context.data) is not Data:raise ValueError('typed Thunder Wave source context required')
    for word in context.data[:5]:primary._shape(word,(8,),j.uint32)
    primary._shape(context.data.move_id,(),j.int32)
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    good=ex._validate(actual.ledger,context.accounting)&ex._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.refusal==0)&life._valid(tape.cursor)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= j.all(getattr(context.data,name)==getattr(tape,name))
    for before,after in zip(actual.tables,context.accounting):good &= j.all(before==after)
    from . import engine_kernel_frame_risk as risk
    if type(actual.risks) is not risk.Context:return good&False
    rbound,_=risk._bound(actual.risks.carry,tape,actual.risks.tables)
    good &= rbound&j.any(context.data.binding!=0)&j.all(context.data.binding==actual.risks.tables.thunder_wave_binding)
    return (good&(actual.ledger.frame>0)&tape.cursor.initialized&(tape.cursor.phase==0)
        &(tape.turn==tape.cursor.turn)&(context.data.move_id>0))

def span_matches(tape,ticket,side,physical,target,move,attempt,context):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    ledger=actual.ledger;slot,same=ex._ticket(ledger,ticket);b=len(tape.count);rows=j.arange(b);safe=j.clip(side,0,1)
    raw=primary.words_device(side,physical,target,move,attempt);start=ledger.start[rows,slot]
    first=j.clip(start,0,tape.words.shape[1]-1);at=j.clip(start+1,0,tape.words.shape[1]-1);prefix=j.clip(start-1,0,tape.words.shape[1]-1)
    absent=j.full_like(side,-1);zero=j.zeros_like(side)
    status_word=j.stack((j.full_like(side,status.STATUS_APPLIED),absent,absent,1-side,target,
        attempt.old_status,attempt.new_status,j.full_like(side,status.PRIMARY),zero,zero),axis=1)
    decl=tape.words[rows,prefix]
    good=(_bound(tape,context)&same&supported_raw(raw)&(move==context.data.move_id)
        &(attempt.old_counter==0)&(attempt.new_counter==0)&(start>0)&(tape.count==start+2)
        &(ledger.site[rows,slot]==int(ex.Site.PRIMARY_PARALYSIS))&(ledger.side[rows,slot]==1-side)
        &(physical==tape.cursor.physical_active[rows,safe])&(target==tape.cursor.physical_active[rows,1-safe])
        &(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)
        &j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
        &j.all(tape.words[rows,first]==status_word,axis=1)&j.all(tape.words[rows,at]==raw,axis=1)
        &(decl[:,0]==public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)&(decl[:,3]==1-side)&(decl[:,4]==target)&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1))
    for index in (prefix,first,at):
        good &= ((tape.event_turn[rows,index]==ledger.turn[rows,slot])&(tape.event_phase[rows,index]==ledger.phase[rows,slot])
            &j.all(tape.event_generation[rows,index]==ledger.generation[rows,slot],axis=1))
    return good

def record_device(tape,ticket,side,physical,target,move,attempt,context,*,requested):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return tape
    primary._shape(requested,tape.count.shape,j.bool_)
    raw=primary.words_device(side,physical,target,move,attempt)
    valid=_bound(tape,context)&supported_raw(raw)&(move==context.data.move_id)&(attempt.old_counter==0)&(attempt.new_counter==0)
    tape=public._append(tape,primary.TOXIC_ATTEMPT,side,physical,1-side,target,tuple(raw[:,i] for i in range(5,10)),requested&valid)
    same=span_matches(tape,ticket,side,physical,target,move,attempt,context)
    actual=ex._context(tape);ledger=ex.unhandled(actual.ledger,ticket,active=requested&~same,tables=actual.tables)
    ledger=ex.emitted(ledger,ticket,tape,active=requested&same,tables=actual.tables)
    return ex._put_context(tape,ledger,actual)


def exclude_cure_device(tape,requested,applied,item,item_table):
    """This narrow admission excludes an actual primary-triggered Lum cure.

    Read the original current item/actual apply guard before AfterSetStatus;
    never infer a cure from final status or duplicate its numerical operation.
    The existing reflection guard remains in original after_status_event.
    """
    b=len(tape.count)
    primary._shape(requested,(b,),j.bool_);primary._shape(applied,(b,),j.bool_);primary._shape(item,(b,),j.int32)
    if item_table is None:return tape
    return status.unknown(tape,requested&applied&item_table.lum[item])
