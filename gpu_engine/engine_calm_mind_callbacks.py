"""Literal Calm Mind SPA then SPD +1 pair at the SAME actual boost assignment.

RAW30=[30,side,physical,move,stat2or3,old,new,actualSecondaryDispatch,0,0].
One Site16 owns BOTH consecutive receipts. No persistent boost inventory,
reconstructed stats, no-output exemption, copied-carry origin or qualification.
Only original explicit registered mode may emit; every unsupported attempt is
sticky UNKNOWN. Original owning host checks remain required before cached JIT.
"""
from dataclasses import dataclass
import hashlib,json,weakref
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life
from .engine_private_team import engine_metadata_binding

CALM_MIND=30
UNKNOWN_CALM_MIND=256
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
    old_spa:object
    new_spa:object
    old_spd:object
    new_spd:object
    secondary_dispatch:object       # literal actual acts&hit&~immune mask

@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Calm Mind must remain outside outer JIT')


def compile_calm_mind(*,engine_catalog,metadata):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('same original Calm Mind metadata required')
    ids=[i for i,n in enumerate(metadata['vocabulary']['moves']) if n=='calmmind']
    if len(ids)!=1 or ids[0]<=0:raise ValueError('one canonical Calm Mind row required')
    move=ids[0];t=engine_catalog.engine_table
    if (t.effects is None or not np.array_equal(np.asarray(t.effects.self_boosts[move]),[0,0,1,1,0,0,0])
            or int(t.power[move])!=0 or int(t.type[move])!=14 or bool(t.physical[move])
            or not bool(t.always_hit[move]) or int(t.priority[move])!=0
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=0):raise ValueError('exact canonical self-target Calm Mind columns required')
    # No adjacent self-effect source can share this receipt.
    for name in ('heal_half','rest','haze','substitute','protect','endure','heal_bell','weather_heal','pain_split','belly_drum','hit_spa_drop','status'):
        if bool(np.any(np.asarray(getattr(t.effects,name)[move])!=0)):raise ValueError('Calm Mind mixed self effect unsupported')
    raw=dict(schema='pokezero.calm_mind.literal.v1',producer=engine_catalog.binding,engine=engine_catalog.engine_binding,source=engine_catalog.source_binding,moves=engine_catalog.move_binding,move=move,raw_kind=CALM_MIND)
    binding=np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)
    data=Data(jnp.asarray(binding),jnp.asarray(engine_catalog.engine_binding,jnp.uint32),jnp.asarray(engine_catalog.source_binding,jnp.uint32),jnp.asarray(engine_catalog.move_binding,jnp.uint32),jnp.asarray(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.asarray(move,jnp.int32))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog);return result


def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(x is not y for x,y in zip(compiled.device,r[1])):raise ValueError('original immutable Calm Mind compiler required')
    public.check_catalog(r[2],r[2].engine_table,expected_binding=r[2].binding,expected_engine_binding=np.asarray(r[2].engine_binding,np.uint32));return r[0]


def validate_write_device(write,b):
    if type(write) is not Write:raise ValueError('literal Calm Mind Write required')
    for x in write[:4]:life._typed(x,(b,),jnp.int32)
    life._typed(write.secondary_dispatch,(b,),jnp.bool_)
    return ((write.old_spa>=-6)&(write.old_spa<6)&(write.new_spa==write.old_spa+1)
        &(write.old_spd>=-6)&(write.old_spd<6)&(write.new_spd==write.old_spd+1))


def _bound(tape,context):
    from . import engine_execution_completeness as ex
    if type(context) is not Context or type(context.data) is not Data:raise ValueError('original registered Calm Mind Context required')
    if type(tape) is not life.LifecycleTape:raise ValueError('original stamped Calm Mind lifecycle tape required')
    data=context.data;b=tape.count.shape[0]
    for x in data[:5]:life._typed(x,(8,),jnp.uint32)
    life._typed(data.move_id,(),jnp.int32)
    actual=ex._context(tape)
    if actual is None:return jnp.zeros(b,bool)
    same=ex._validate(actual.ledger,context.accounting)&ex._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.frame>0)
    for old,new in zip(context.accounting,actual.tables):same &= jnp.all(old==new)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):same &= jnp.all(getattr(data,name)==getattr(tape,name))
    if actual.risks is not None:same &= jnp.all(data.binding==actual.risks.tables.calm_mind_binding)
    return same&life._valid(tape.cursor)&(tape.cursor.phase==0)&(tape.turn==tape.cursor.turn)&(data.move_id>0)


def span_matches(tape,ticket,side,physical,move,write,context):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=tape.count.shape[0];rows=jnp.arange(b)
    if actual is None:return jnp.zeros(b,bool)
    for x in (side,physical,move):life._typed(x,(b,),jnp.int32)
    ledger=actual.ledger;slot,same=ex._ticket(ledger,ticket);start=ledger.start[rows,slot]
    good=_bound(tape,context)&same&validate_write_device(write,b)&(side>=0)&(side<2)&(move==context.data.move_id)
    safe=jnp.clip(side,0,1);good &= (physical==tape.cursor.physical_active[rows,safe])&(ledger.site[rows,slot]==int(ex.Site.CALM_MIND))&(ledger.side[rows,slot]==side)&(start>0)&(tape.count==start+2)
    prefix=jnp.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (decl[:,0]==public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)&(decl[:,3]==1-side)&(decl[:,4]==tape.cursor.physical_active[rows,1-safe])&(decl[:,5]==move)&jnp.all(decl[:,6:]==0,axis=1)
    for offset,stat,old,new in ((0,2,write.old_spa,write.new_spa),(1,3,write.old_spd,write.new_spd)):
        at=jnp.clip(start+offset,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
        expected=jnp.stack((jnp.full(b,CALM_MIND,jnp.int32),side,physical,move,jnp.full(b,stat,jnp.int32),old,new,write.secondary_dispatch.astype(jnp.int32),jnp.zeros(b,jnp.int32),jnp.zeros(b,jnp.int32)),axis=1)
        good &= jnp.all(raw==expected,axis=1)&(tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    good &= (tape.event_turn[rows,prefix]==ledger.turn[rows,slot])&(tape.event_phase[rows,prefix]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,prefix]==ledger.generation[rows,slot],axis=1)
    return good&(tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==ledger.phase[rows,slot])&jnp.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1)


def record_device(tape,ticket,side,physical,move,write,context,*,eligible):
    from . import engine_execution_completeness as ex
    b=tape.count.shape[0];life._typed(eligible,(b,),jnp.bool_)
    if type(side) is int:side=jnp.full(b,side,jnp.int32)
    for x in (side,physical,move):life._typed(x,(b,),jnp.int32)
    valid=_bound(tape,context)&validate_write_device(write,b)&(move==context.data.move_id)
    room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-2)
    overflow=eligible&valid&~room
    rows=jnp.arange(b)
    for offset in (0,1):
        address=tape.count+offset;inside=(address>=0)&(address<tape.words.shape[1]);at=jnp.clip(address,0,tape.words.shape[1]-1)
        valid &= ~inside|jnp.all(tape.switch_details[rows,at]==0,axis=1)

    z=jnp.zeros(b,jnp.int32);use=eligible&valid&room
    for stat,old,new in ((2,write.old_spa,write.new_spa),(3,write.old_spd,write.new_spd)):
        tape=public._append(tape,CALM_MIND,side,physical,move,jnp.full(b,stat,jnp.int32),(old,new,write.secondary_dispatch.astype(jnp.int32),z,z),use)
    good=valid&room&span_matches(tape,ticket,side,physical,move,write,context)
    actual=ex._context(tape)
    if actual is not None:
        ledger=ex.unhandled(actual.ledger,ticket,active=eligible&~good,tables=actual.tables);ledger=ex.emitted(ledger,ticket,tape,active=eligible&good,tables=actual.tables);tape=ex._put_context(tape,ledger,actual)
    return tape._replace(refusal=tape.refusal|jnp.where(eligible&~good,UNKNOWN_CALM_MIND,0)|jnp.where(overflow,public.OVERFLOW,0))
