"""Literal canonical Spikes increment at its actual owning hazard assignment.

RAW31 is ONE Site17 receipt. Registered mode only; capped/nonincremental,
foreign/blocked/malformed attempts remain UNKNOWN. No hazard-damage coverage,
no no-output exemption, no retroactive risk clearing or scientific qualification.
"""
from dataclasses import dataclass
import hashlib,json,weakref
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life
from .engine_private_team import engine_metadata_binding

SPIKES=31
UNKNOWN_SPIKES=256
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
    accounting:object

class Write(NamedTuple):
    target_side:object
    old_layers:object
    new_layers:object
    secondary_dispatch:object

@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data


def compile_spikes(*,engine_catalog,metadata):
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('Spikes compiler must remain outside outer JIT')
    public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('same original Spikes metadata required')
    ids=[i for i,n in enumerate(metadata['vocabulary']['moves']) if n=='spikes']
    if len(ids)!=1 or ids[0]<=0:raise ValueError('one canonical Spikes row required')
    move=ids[0];t=engine_catalog.engine_table
    if (t.entry is None or t.effects is None or not bool(t.entry.spikes_move[move])
            or int(t.power[move])!=0 or int(t.type[move])!=int(t.entry.ground_type) or bool(t.physical[move])
            or not bool(t.always_hit[move]) or int(t.priority[move])!=0
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1):
        raise ValueError('exact canonical Ground foeSide Spikes columns required')
    for name in ('heal_half','rest','haze','substitute','protect','endure','heal_bell','weather_heal','pain_split','belly_drum','hit_spa_drop','status','self_boosts','protectable'):
        if bool(np.any(np.asarray(getattr(t.effects,name)[move])!=0)):raise ValueError('Spikes mixed effect unsupported')
    raw=dict(schema='pokezero.spikes.literal.v1',producer=engine_catalog.binding,engine=engine_catalog.engine_binding,source=engine_catalog.source_binding,moves=engine_catalog.move_binding,move=move,raw_kind=SPIKES)
    digest=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest()
    data=Data(j.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),j.asarray(engine_catalog.engine_binding,j.uint32),j.asarray(engine_catalog.source_binding,j.uint32),j.asarray(engine_catalog.move_binding,j.uint32),j.asarray(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog);return result


def check_compiled(compiled):
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('checked Spikes must remain outside outer JIT')
    r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(x is not y for x,y in zip(compiled.device,r[1])):raise ValueError('original immutable Spikes compiler required')
    public.check_catalog(r[2],r[2].engine_table,expected_binding=r[2].binding,expected_engine_binding=np.asarray(r[2].engine_binding,np.uint32));return r[0]


def validate_write_device(write,b):
    if type(write) is not Write:raise ValueError('literal Spikes Write required')
    for x in write[:3]:life._typed(x,(b,),j.int32)
    life._typed(write.secondary_dispatch,(b,),j.bool_)
    return ((write.target_side>=0)&(write.target_side<2)&(write.old_layers>=0)&(write.old_layers<3)&(write.new_layers==write.old_layers+1))


def _bound(tape,context):
    from . import engine_execution_completeness as ex
    if type(context) is not Context or type(context.data) is not Data or type(tape) is not life.LifecycleTape:raise ValueError('original registered Spikes Context/tape required')
    b=tape.count.shape[0];data=context.data
    for x in data[:5]:life._typed(x,(8,),j.uint32)
    life._typed(data.move_id,(),j.int32)
    actual=ex._context(tape)
    if actual is None:return j.zeros(b,bool)
    from . import engine_kernel_frame_risk as risk
    if type(actual.risks) is not risk.Context:return j.zeros(b,bool)
    rbound,_=risk._bound(actual.risks.carry,tape,actual.risks.tables)
    same=rbound&j.any(actual.risks.tables.spikes_binding!=0)&j.all(context.data.binding==actual.risks.tables.spikes_binding)
    same &= ex._validate(actual.ledger,context.accounting)&ex._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.frame>0)
    for x,y in zip(context.accounting,actual.tables):same &= j.all(x==y)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):same &= j.all(getattr(data,name)==getattr(tape,name))
    return same&life._valid(tape.cursor)&tape.cursor.initialized&(tape.cursor.phase==0)&(tape.turn==tape.cursor.turn)&(data.move_id>0)


def span_matches(tape,ticket,side,physical,move,write,context):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=tape.count.shape[0];rows=j.arange(b)
    if actual is None:return j.zeros(b,bool)
    for x in (side,physical,move):life._typed(x,(b,),j.int32)
    ledger=actual.ledger;slot,same=ex._ticket(ledger,ticket);start=ledger.start[rows,slot];safe=j.clip(side,0,1)
    good=(_bound(tape,context)&same&validate_write_device(write,b)&(side>=0)&(side<2)&(move==context.data.move_id)
        &(write.target_side==1-side)&(physical==tape.cursor.physical_active[rows,safe])
        &(ledger.site[rows,slot]==int(ex.Site.SPIKES_WRITE))&(ledger.side[rows,slot]==side)&(start>0)&(tape.count==start+1))
    prefix=j.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (decl[:,0]==public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)&(decl[:,3]==1-side)&(decl[:,4]==tape.cursor.physical_active[rows,1-safe])&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1)
    expected=j.stack((j.full(b,SPIKES,j.int32),side,physical,move,write.target_side,write.old_layers,write.new_layers,write.secondary_dispatch.astype(j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32)),axis=1)
    at=j.clip(start,0,tape.words.shape[1]-1)
    return good&j.all(tape.words[rows,at]==expected,axis=1)&(tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)&j.all(tape.switch_details[rows,at]==0,axis=1)


def record_device(tape,ticket,side,physical,move,write,context,*,requested):
    from . import engine_execution_completeness as ex
    b=tape.count.shape[0]
    if type(side) is int:side=j.full(b,side,j.int32)
    for x in (side,physical,move):life._typed(x,(b,),j.int32)
    life._typed(requested,(b,),j.bool_)
    valid=_bound(tape,context)&validate_write_device(write,b)&(side>=0)&(side<2)&(move==context.data.move_id)&(write.target_side==1-side)
    valid &= physical==tape.cursor.physical_active[j.arange(b),j.clip(side,0,1)]
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);at=j.clip(tape.count,0,tape.words.shape[1]-1)
    valid &= ~inside|j.all(tape.switch_details[j.arange(b),at]==0,axis=1)
    tape=public._append(tape,SPIKES,side,physical,move,write.target_side,
        (write.old_layers,write.new_layers,write.secondary_dispatch.astype(j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32)),requested&valid)
    actual=ex._context(tape)
    if actual is not None:
        good=valid&inside&span_matches(tape,ticket,side,physical,move,write,context)
        ledger=ex.unhandled(actual.ledger,ticket,active=requested&~good,tables=actual.tables)
        ledger=ex.emitted(ledger,ticket,tape,active=requested&good,tables=actual.tables)
        tape=ex._put_context(tape,ledger,actual)
    return tape._replace(refusal=tape.refusal|j.where(requested&(~valid|~inside),UNKNOWN_SPIKES,0))
