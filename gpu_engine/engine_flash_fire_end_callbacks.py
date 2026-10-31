"""Literal live Flash Fire departure, before the original switch installation.

Only the SAME depart call's bit2 ->0 write is represented. No Baton, faint,
copied ability, other volatile or absent source capability is inferred safe.
"""
from typing import NamedTuple
import jax.numpy as j
from . import absorption, engine_public_events as public, engine_lifecycle_callbacks as life

FLASH_FIRE_END=37

class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    ability:object
    volatile:object
    hp:object
    requested:object
    supported:object

def begin(tape,state,side,*,eligible,baton,context):
    from . import engine_direct_hit_controls as direct, engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);physical=state.active[rows,side]
    old=state.volatile[rows,side,physical];hp=state.hp[rows,side,physical]
    abi=state.ability[rows,side,physical]
    requested=eligible&((old&j.uint32(absorption.FLASH_FIRE))!=0)
    if context is None or type(context) is not direct.RegisteredContext:
        return direct._unknown(tape,requested),None
    data=context.data.no_damage
    good=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= j.all(getattr(data,name)==getattr(tape,name))
    good &= (abi>=0)&(abi<len(data.flash))&data.flash[j.clip(abi,0,len(data.flash)-1)]
    good &= (state.base_ability[rows,side,physical]==abi)&(hp>0)&~baton
    good &= (old==j.uint32(absorption.FLASH_FIRE))&tape.cursor.initialized
    good &= (physical==tape.cursor.physical_active[rows,side])&(tape.cursor.phase==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.FLASH_FIRE_END,side,active=requested)
    return tape,Pending(context,ticket,side,physical,abi,old,hp,requested,good)

def finish(tape,p,state):
    if p is None:return tape
    from . import engine_execution_completeness as ex,engine_direct_hit_controls as direct
    rows=j.arange(len(tape.count));new=state.volatile[rows,p.side,p.physical]
    good=p.supported&direct._registered_bound(tape,p.context)&(new==j.uint32(0))
    good &= (state.hp[rows,p.side,p.physical]==p.hp)&(state.active[rows,p.side]==p.physical)
    good &= (state.ability[rows,p.side,p.physical]==p.ability)
    absent=j.full_like(p.side,-1);zero=j.zeros_like(p.side)
    tape=public._append(tape,FLASH_FIRE_END,absent,absent,p.side,p.physical,
        (p.ability,p.volatile.astype(j.int32),new.astype(j.int32),p.hp,zero),p.requested&good)
    context=ex._context(tape)
    ledger=ex.emitted(context.ledger,p.ticket,tape,active=p.requested&good,tables=context.tables)
    ledger=ex.unhandled(ledger,p.ticket,active=p.requested&~good,tables=context.tables)
    return ex._put_context(tape,ledger,context)

def owned(tape,index,data):
    """Exact stage2 source span followed by its actual non-Baton switch."""
    from . import engine_execution_completeness as ex,engine_no_damage_callbacks as no_damage
    no_damage._shape(data);b=len(tape.count);rows=j.arange(b);context=ex._context(tape)
    if context is None:return j.zeros(b,bool)
    ledger=context.ledger;at=j.broadcast_to(j.asarray(index,j.int32),(b,))
    safe=j.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,safe]
    next_at=j.clip(at+1,0,tape.words.shape[1]-1);following=tape.words[rows,next_at]
    claim=ledger.claimed[rows,safe];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,context.tables)&ex._tape(ledger,tape,context.tables)&ledger.opened
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32)
        good &= j.all(getattr(data,name)==getattr(tape,name))
    good &= (at>=0)&(at+1<tape.count)&(claim>0)&(claim<=ledger.count)
    good &= (raw[:,0]==FLASH_FIRE_END)&j.all(raw[:,1:3]==-1,axis=1)
    good &= (raw[:,3]>=0)&(raw[:,3]<2)&(raw[:,4]>=0)&(raw[:,4]<6)
    abi=raw[:,5];good &= (abi>0)&(abi<len(data.flash))&data.flash[j.clip(abi,0,len(data.flash)-1)]
    good &= (raw[:,6]==absorption.FLASH_FIRE)&(raw[:,7]==0)&(raw[:,8]>0)&(raw[:,8]<=1_000_000)&(raw[:,9]==0)
    good &= (ledger.site[rows,slot]==int(ex.Site.FLASH_FIRE_END))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==2)
    good &= (ledger.start[rows,slot]==at)&(ledger.end[rows,slot]==at+1)
    good &= j.all(ledger.words[rows,safe]==raw,axis=1)
    good &= (following[:,0]==life.SWITCH_INSTALLED)&(following[:,1]==raw[:,3])&(following[:,2]==raw[:,4])&(following[:,3]==raw[:,3])
    good &= (following[:,4]>=0)&(following[:,4]<6)&(following[:,4]!=raw[:,4])
    good &= ((following[:,5]==0)|(following[:,5]==1))&j.all(following[:,6:]==0,axis=1)
    for where in (safe,next_at):
        good &= (tape.event_turn[rows,where]==ledger.turn[rows,slot])&(tape.event_phase[rows,where]==ledger.phase[rows,slot])
        good &= j.all(tape.event_generation[rows,where]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])
    good &= j.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    return good
