"""SAME-write nonlethal entry-Spikes receipts, never a damage calculator.

RAW50: layers, ability, side, physical, oldHP, newHP, maxHP, engine status,
then literal type0 | type1<<5 | item<<10. All are private source operands.
Only the original '-damage ... [from] Spikes' line reaches public consumers.
No-output, lethal, unknown history and missing source context remain UNKNOWN.
"""
from typing import NamedTuple
import jax.numpy as j
from . import engine_public_events as public,engine_lifecycle_callbacks as life

ENTRY_SPIKES_HP=50
UNKNOWN_ENTRY_SPIKES=256


class Pending(NamedTuple):
    ticket:object
    side:object
    physical:object
    layers:object
    ability:object
    hp:object
    maximum:object
    status:object
    types:object
    item:object
    table:object
    requested:object


def namespaces(tape):
    from . import engine_execution_completeness as ex
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        ex._shape(getattr(tape,name),(8,),j.uint32)


def installation(tape,index,cursor):
    """RAW6 immediately before index, stamped BEFORE its generation increment."""
    b=len(tape.count);rows=j.arange(b);index=j.broadcast_to(j.asarray(index,j.int32),(b,))
    at=j.clip(index-1,0,tape.words.shape[1]-1);raw=tape.words[rows,at];side=raw[:,3];safe=j.clip(side,0,1)
    generation=tape.event_generation[rows,at].at[rows,safe].add(1)
    return (life._valid(cursor)&cursor.initialized&(cursor.turn>0)&(cursor.phase==0)
        &j.all(cursor.established,axis=1)&(tape.stream_id==cursor.stream_id)
        &(index>0)&(index<=tape.count)&(index<=tape.words.shape[1])
        &(raw[:,0]==life.SWITCH_INSTALLED)&(raw[:,1]==side)&(side>=0)&(side<2)
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,2]!=raw[:,4])
        &(raw[:,4]==cursor.physical_active[rows,safe])&((raw[:,5]==0)|(raw[:,5]==1))
        &j.all(raw[:,6:]==0,axis=1)&(tape.event_turn[rows,at]==cursor.turn)
        &(tape.event_phase[rows,at]==0)&j.all(generation==cursor.generation,axis=1))


def schema(raw):
    return ((raw[:,0]==ENTRY_SPIKES_HP)&(raw[:,1]>=1)&(raw[:,1]<=3)
        &(raw[:,2]>=0)&(raw[:,3]>=0)&(raw[:,3]<2)&(raw[:,4]>=0)&(raw[:,4]<6)
        &(raw[:,5]>raw[:,6])&(raw[:,6]>0)&(raw[:,7]>=raw[:,5])&(raw[:,7]<=1_000_000)
        &(raw[:,8]>=0)&(raw[:,8]<=6)&(raw[:,9]>=0))


def ground(raw,levitate,flying,type_count,item_count,boots):
    ability=raw[:,2];types=j.stack((raw[:,9]&31,(raw[:,9]>>5)&31),axis=1);item=raw[:,9]>>10
    return (schema(raw)&(type_count>0)&(type_count<=32)&(item_count>0)&(item_count<=2**21)
        &(flying>0)&(flying<type_count)&(ability<levitate.shape[0])
        &~levitate[j.clip(ability,0,levitate.shape[0]-1)]
        &(types[:,0]>0)&j.all((types>=0)&(types<type_count)&(types!=flying),axis=1)
        &(item>=0)&(item<item_count)&((boots==0)|(item!=boots)))


def owned(tape,index,cursor):
    b=len(tape.count);rows=j.arange(b);index=j.broadcast_to(j.asarray(index,j.int32),(b,))
    at=j.clip(index,0,tape.words.shape[1]-1);raw=tape.words[rows,at];side=raw[:,3];safe=j.clip(side,0,1)
    previous=tape.words[rows,j.clip(index-1,0,tape.words.shape[1]-1)]
    return (schema(raw)&installation(tape,index,cursor)&(index<tape.count)
        &(previous[:,3]==side)&(previous[:,4]==raw[:,4])
        &(raw[:,4]==cursor.physical_active[rows,safe])&j.all(tape.switch_details[rows,at]==0,axis=1)
        &(tape.event_turn[rows,at]==cursor.turn)&(tape.event_phase[rows,at]==0)
        &j.all(tape.event_generation[rows,at]==cursor.generation,axis=1))


def claimed(tape,index,cursor):
    """Only this exact owned positive Site8 payload may bridge an entry prefix."""
    from . import engine_execution_completeness as ex
    from . import engine_kernel_frame_risk as risks
    context=ex._context(tape);b=len(tape.count)
    if context is None:return j.zeros(b,bool)
    namespaces(tape);rows=j.arange(b);index=j.broadcast_to(j.asarray(index,j.int32),(b,))
    at=j.clip(index,0,tape.words.shape[1]-1);ledger=context.ledger;claim=ledger.claimed[rows,at]
    slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    bound=j.zeros(b,bool)
    if type(context.risks) is risks.Context:
        bound,_=risks._bound(context.risks.carry,tape,context.risks.tables)
        bound &= context.risks.carry.opened&(context.risks.carry.frame==ledger.frame)
    return (bound&owned(tape,index,cursor)&ex._validate(ledger,context.tables)&ledger.opened&(ledger.refusal==0)
        &(ledger.stream_id==tape.stream_id)&(claim>0)&(claim<=ledger.count)
        &(ledger.site[rows,slot]==int(ex.Site.ENTRY_HAZARD))&(ledger.side[rows,slot]==tape.words[rows,at,3])
        &(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==index)&(ledger.end[rows,slot]==index+1)
        &(ledger.turn[rows,slot]==cursor.turn)&(ledger.phase[rows,slot]==cursor.phase)
        &j.all(ledger.generation[rows,slot]==cursor.generation,axis=1)
        &j.all(ledger.words[rows,at]==tape.words[rows,at],axis=1)
        &(ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])
        &j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1))


def begin(tape,state,side,layers,table,*,requested):
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);side=j.broadcast_to(j.asarray(side,j.int32),(b,))
    ex._shape(requested,(b,),j.bool_);namespaces(tape)
    physical=state.active[rows,side]
    tape,ticket=ex.open_attempt(tape,ex.Site.ENTRY_HAZARD,side,active=requested)
    return tape,Pending(ticket,side,physical,layers,state.ability[rows,side,physical],
        state.hp[rows,side,physical],state.max_hp[rows,side,physical],state.status[rows,side,physical],
        state.types[rows,side,physical],state.item[rows,side,physical],table,requested)


def finish(tape,p,new_hp):
    from . import engine_execution_completeness as ex
    from . import engine_kernel_frame_risk as risks
    context=ex._context(tape)
    if context is None:return tape
    if type(p) is not Pending:raise ValueError('SAME-call entry Spikes pending write required')
    namespaces(tape);b=len(tape.count);rows=j.arange(b)
    for x in (p.side,p.physical,p.layers,p.ability,p.hp,new_hp,p.maximum,p.status,p.item):ex._shape(x,(b,),j.int32)
    ex._shape(p.types,(b,2),j.int32);ex._shape(p.requested,(b,),j.bool_)
    # Legacy EntryTable constructors keep the appended observer statics None.
    # Preserve their old requested-UNKNOWN behavior rather than failing an
    # unrelated entry with no hazard attempt.
    if any(getattr(p.table,name,None) is None for name in ('spikes_type_count','spikes_item_count','spikes_boots_item')):
        ledger=ex.unhandled(context.ledger,p.ticket,active=p.requested,tables=context.tables)
        tape=ex._put_context(tape,ledger,context)
        return tape._replace(refusal=tape.refusal|j.where(p.requested,UNKNOWN_ENTRY_SPIKES,0))
    for x in (p.table.flying_type,p.table.spikes_type_count,p.table.spikes_item_count,p.table.spikes_boots_item):ex._shape(x,(),j.int32)
    if p.table.levitate.ndim!=1 or not len(p.table.levitate):raise ValueError('original entry ability namespace required')
    ex._shape(p.table.levitate,p.table.levitate.shape,j.bool_)
    packed=p.types[:,0]|(p.types[:,1]<<5)|(p.item<<10)
    raw=j.stack((j.full(b,ENTRY_SPIKES_HP,j.int32),p.layers,p.ability,p.side,p.physical,p.hp,new_hp,p.maximum,p.status,packed),axis=1)
    ledger=context.ledger;slot,same=ex._ticket(ledger,p.ticket);bound=j.zeros(b,bool)
    if type(context.risks) is risks.Context:
        bound,_=risks._bound(context.risks.carry,tape,context.risks.tables)
        bound &= context.risks.carry.opened&(context.risks.carry.frame==ledger.frame)
    previous=tape.words[rows,j.clip(tape.count-1,0,tape.words.shape[1]-1)]
    valid=(bound&same&ex._validate(ledger,context.tables)&(ledger.refusal==0)
        &((tape.refusal&~public.UNINSTRUMENTED)==0)&installation(tape,tape.count,tape.cursor)
        &(previous[:,3]==p.side)&(previous[:,4]==p.physical)
        &(ledger.site[rows,slot]==int(ex.Site.ENTRY_HAZARD))&(ledger.side[rows,slot]==p.side)
        &(ledger.start[rows,slot]==tape.count)&(ledger.turn[rows,slot]==tape.cursor.turn)
        &(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
        &j.all((p.types>=0)&(p.types<32),axis=1)&(p.item>=0)&(p.item<2**21)
        &ground(raw,p.table.levitate,p.table.flying_type,p.table.spikes_type_count,p.table.spikes_item_count,p.table.spikes_boots_item))
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);at=j.clip(tape.count,0,tape.words.shape[1]-1)
    valid &= ~inside|j.all(tape.switch_details[rows,at]==0,axis=1)
    tape=public._append(tape,ENTRY_SPIKES_HP,p.layers,p.ability,p.side,p.physical,
        (p.hp,new_hp,p.maximum,p.status,packed),p.requested&valid)
    ledger=ex.unhandled(ledger,p.ticket,active=p.requested&~valid,tables=context.tables)
    ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&valid,tables=context.tables)
    tape=ex._put_context(tape,ledger,context)
    return tape._replace(refusal=tape.refusal|j.where(p.requested&(~valid|~inside),UNKNOWN_ENTRY_SPIKES,0))
