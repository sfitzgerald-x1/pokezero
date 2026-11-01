"""SAME-call sleeping Sleep Talk opener/selection, RAW57/Site42.

Only a positive-PP canonical normal-target damaging child is declared here.
The existing child write/outcome guards remain mandatory and unchanged. No
status, sleep clock, PP, choice, draw or damage is assigned by this observer.
"""
from typing import NamedTuple
import jax.numpy as j
from . import engine_public_events as public,engine_lifecycle_callbacks as life

SLEEP_TALK=57
CLOCK,CALL=0,1
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    target:object
    caller:object
    roll:object # literal same-call sleep_talk_draws before original choose
    slot:object
    moves:object
    pp:object
    counter:object
    skipped:object
    rest:object
    early:object
    hp:object
    maximum:object
    ability:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    raw:object
    before:object
    called:object
    roll:object
    selected_slot:object


def _bound(tape,context):
    from . import engine_self_move_callbacks as sm
    if context is None or context.data.self_move is None:return j.zeros(len(tape.count),bool)
    sm.shape(context.data.self_move)
    if context.data.self_move.talking is None:return j.zeros(len(tape.count),bool)
    return sm._bound(tape,context)


def begin(tape,state,side,caller,slot,context,table,*,choice_roll,eligible,sleep_usable,early_bird,loafing):
    from . import engine_execution_completeness as ex
    if context is None or context.data.self_move is None or context.data.self_move.talking is None:return tape,None
    b=len(tape.count);rows=j.arange(b)
    for x in(side,caller,slot,choice_roll):life._typed(x,(b,),j.int32)
    for x in(eligible,sleep_usable,early_bird,loafing):life._typed(x,(b,),j.bool_)
    d=context.data.self_move;good=_bound(tape,context);installed=d.talking
    if type(table)is not type(installed):raise ValueError('original installed Sleep Talk table required')
    for x,y in zip(table,installed,strict=True):
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Sleep Talk table geometry differs')
        good &= j.all(x==y)
    seat=j.clip(side,0,1);physical=state.active[rows,seat];at=(rows,seat,physical)
    hp=state.hp[at];maximum=state.max_hp[at];counter=state.status_counter[at];ability=state.ability[at]
    requested=eligible&sleep_usable&(state.status[at]==5)&(counter>1+early_bird.astype(j.int32))
    good &= (choice_roll>=0)&(choice_roll<60)&(side>=0)&(side<2)&(caller==d.move_ids[1])&(caller>0)&(slot>=0)&(slot<4)
    good &= (state.moves[at][rows,j.clip(slot,0,3)]==caller)&(state.pp[at][rows,j.clip(slot,0,3)]>0)
    good &= (physical==tape.cursor.physical_active[rows,seat])&(hp>0)&(hp<=maximum)&(maximum<=1_000_000)
    good &= (counter>=2)&(counter<=5)&(state.sleep_skipped[at]>=0)&(state.sleep_skipped[at]<=5)
    good &= (ability>=0)&(ability<len(installed.early_bird))&(early_bird==installed.early_bird[j.clip(ability,0,len(installed.early_bird)-1)])
    good &= j.all(installed.callable[state.moves[at]]==d.talking_source[state.moves[at]],axis=1)
    good &= ~loafing&(state.volatile[at]==0)&(state.confusion_counter[at]==0)&(state.encore_counter[at]==0)&(state.choice_move[at]==0)
    good &= (tape.refusal&~public.UNINSTRUMENTED)==0
    tape,ticket=ex.open_attempt(tape,ex.Site.SLEEP_TALK,side,active=requested)
    return tape,Pending(context,ticket,side,physical,state.active[rows,1-seat],caller,choice_roll,slot,state.moves[at],state.pp[at],counter,state.sleep_skipped[at],state.rest_origin[at],early_bird,hp,maximum,ability,requested,good)


def owned(tape,index,*,stage=2):
    """Original declaration span: CANT(SLP), caller MOVE, actual child RAW1."""
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None or c.risks is None:return j.zeros(b,bool)
    ledger=c.ledger;t=c.risks.tables
    claim=ledger.count if stage==1 else ledger.claimed[rows,at];entry=j.clip(claim-1,0,ledger.site.shape[1]-1)
    start=ledger.start[rows,entry];addresses=j.clip(start[:,None]+j.arange(3)[None,:],0,tape.words.shape[1]-1);parts=tape.words[rows[:,None],addresses];a,z,child=(parts[:,k]for k in range(3))
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)
    for name in('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(t,name))
    life._typed(t.self_move_binding,(8,),j.uint32);life._typed(t.self_move_ids,(3,),j.int32)
    good &= j.any(t.self_move_binding!=0)&(t.self_move_ids[1]>0)&(claim>0)&(claim<=ledger.count)&(idx>=start)&(idx<start+3)&(idx>=0)&(idx<tape.count)&(start>=0)&(start+3<=tape.count)
    good &= (a[:,0]==SLEEP_TALK)&(z[:,0]==SLEEP_TALK)&(a[:,3]==CLOCK)&(z[:,3]==CALL)&j.all(a[:,1:3]==z[:,1:3],axis=1)&(a[:,4]==t.self_move_ids[1])&(z[:,4]==a[:,4])
    good &= (a[:,1]>=0)&(a[:,1]<2)&(a[:,2]>=0)&(a[:,2]<6)&(a[:,5]>=2)&(a[:,5]<=5)&(a[:,6]>0)&(a[:,6]==a[:,5]-1-(a[:,9]&1))&(a[:,7]>=0)&(a[:,8]==a[:,7]+1)&(a[:,8]<=6)&(a[:,9]>=0)&(a[:,9]<=3)
    good &= (z[:,5]>0)&(z[:,5]<len(t.direct_candidates))&t.direct_candidates[j.clip(z[:,5],0,len(t.direct_candidates)-1)]&(z[:,6]>=0)&(z[:,6]<4)&(z[:,7]>0)&(z[:,8]>=0)&(z[:,8]<240)&(z[:,6]!=(z[:,8]%4))&(z[:,9]>=0)
    good &= (child[:,0]==public.MOVE)&j.all(child[:,1:3]==a[:,1:3],axis=1)&(child[:,3]==1-a[:,1])&(child[:,4]>=0)&(child[:,4]<6)&(child[:,5]==z[:,5])&j.all(child[:,6:]==0,axis=1)
    good &= (ledger.site[rows,entry]==int(ex.Site.SLEEP_TALK))&(ledger.side[rows,entry]==a[:,1])&(ledger.stage[rows,entry]==stage)&(ledger.no_output_kind[rows,entry]==0)
    for offset in range(3):
        pos=addresses[:,offset];good &= j.all(tape.switch_details[rows,pos]==0,axis=1)&(tape.event_turn[rows,pos]==ledger.turn[rows,entry])&(tape.event_phase[rows,pos]==ledger.phase[rows,entry])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,entry],axis=1)
        if stage==2:good &= (ledger.claimed[rows,pos]==claim)&j.all(ledger.words[rows,pos]==parts[:,offset],axis=1)&(ledger.event_turn[rows,pos]==tape.event_turn[rows,pos])&(ledger.event_phase[rows,pos]==tape.event_phase[rows,pos])&j.all(ledger.event_generation[rows,pos]==tape.event_generation[rows,pos],axis=1)
    if stage==1:
        good &= (tape.count==start+3)&(a[:,2]==tape.cursor.physical_active[rows,j.clip(a[:,1],0,1)])&(child[:,4]==tape.cursor.physical_active[rows,1-j.clip(a[:,1],0,1)])&(ledger.turn[rows,entry]==tape.cursor.turn)&(ledger.phase[rows,entry]==tape.cursor.phase)&j.all(ledger.generation[rows,entry]==tape.cursor.generation,axis=1)
    else:good &= ledger.end[rows,entry]==start+3
    return good


def finish(tape,p,state,before,called,usable,roll,*,acts,caller_acts):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex,mechanics as m
    if type(before)is not m.BeforeMove:raise ValueError('literal original BeforeMove required')
    b=len(tape.count);rows=j.arange(b);seat=j.clip(p.side,0,1);at=(rows,seat,p.physical);d=p.context.data.self_move;table=d.talking
    for x in(before.status,before.counter,before.confusion_counter,called,roll):life._typed(x,(b,),j.int32)
    for x in(before.can_act,before.confusion_self_hit):life._typed(x,(b,),j.bool_)
    for x in(usable,acts,caller_acts):life._typed(x,(b,),j.bool_)
    eligible=table.callable[p.moves];count=j.sum(eligible,axis=1,dtype=j.int32);ranks=j.cumsum(eligible,axis=1)-1
    selected=eligible&(ranks==(roll%j.maximum(1,count))[:,None]);slot=j.argmax(selected,axis=1).astype(j.int32);pp=state.pp[at];moves=state.moves[at]
    good=p.valid&_bound(tape,p.context)&usable&acts&caller_acts&before.can_act&~before.confusion_self_hit&(before.confusion_counter==0)&(before.status==5)&(before.counter==p.counter-1-p.early.astype(j.int32))&(before.counter>0)
    good &= (roll==p.roll)&(roll>=0)&(roll<60)&(count>0)&j.all(moves==p.moves,axis=1)&(called==p.moves[rows,slot])&(pp[rows,slot]>0)&(slot!=p.slot)
    good &= (called>0)&(called<len(tape.execution.risks.tables.direct_candidates))&tape.execution.risks.tables.direct_candidates[j.clip(called,0,len(tape.execution.risks.tables.direct_candidates)-1)]
    wanted=p.pp.at[rows,p.slot].set(j.maximum(0,p.pp[rows,p.slot]-1));good &= j.all(pp==wanted,axis=1)
    good &= (state.active[rows,seat]==p.physical)&(state.active[rows,1-seat]==p.target)&(state.status[at]==5)&(state.status_counter[at]==before.counter)&(state.sleep_skipped[at]==p.skipped+1)&(state.rest_origin[at]==p.rest)&(state.hp[at]==p.hp)&(state.max_hp[at]==p.maximum)&(state.ability[at]==p.ability)&(state.volatile[at]==0)&(state.confusion_counter[at]==0)
    c=ex._context(tape);entry,same=ex._ticket(c.ledger,p.ticket);good &= same&(c.ledger.refusal==0)&(c.ledger.start[rows,entry]==tape.count)&((tape.refusal&~public.UNINSTRUMENTED)==0)
    clock=j.stack((j.full(b,SLEEP_TALK,j.int32),p.side,p.physical,j.full(b,CLOCK,j.int32),p.caller,p.counter,before.counter,p.skipped,state.sleep_skipped[at],p.early.astype(j.int32)+2*p.rest.astype(j.int32)),axis=1)
    call=j.stack((j.full(b,SLEEP_TALK,j.int32),p.side,p.physical,j.full(b,CALL,j.int32),p.caller,called,slot,pp[rows,slot],4*roll+p.slot,pp[rows,p.slot]),axis=1)
    child=j.stack((j.full(b,public.MOVE,j.int32),p.side,p.physical,1-p.side,p.target,called,j.zeros(b,j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32)),axis=1)
    room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-3)
    for offset in range(3):
        pos=j.clip(tape.count+offset,0,tape.words.shape[1]-1);good &= ~room|j.all(tape.switch_details[rows,pos]==0,axis=1)
    use=p.requested&good&room
    for raw in(clock,call,child):tape=public._append(tape,raw[:,0],raw[:,1],raw[:,2],raw[:,3],raw[:,4],tuple(raw[:,i]for i in range(5,10)),use)
    good &= owned(tape,j.maximum(tape.count-1,0),stage=1)&room;c=ex._context(tape)
    ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&~room,public.OVERFLOW,0))
    return tape,Captured(p,j.stack((clock,call,child),axis=1),before,called,roll,slot)


def captured_matches(tape,captured):
    if captured is None:return j.zeros(len(tape.count),bool)
    from . import engine_execution_completeness as ex
    if type(captured)is not Captured or type(captured.pending)is not Pending:raise ValueError('SAME Sleep Talk capture required')
    p=captured.pending;c=ex._context(tape);b=len(tape.count);rows=j.arange(b);entry,same=ex._ticket(c.ledger,p.ticket);start=c.ledger.start[rows,entry]
    good=same&p.requested&p.valid&_bound(tape,p.context)&owned(tape,start)&(c.ledger.end[rows,entry]==tape.count)
    life._typed(p.roll,(b,),j.int32)
    eligible=p.context.data.self_move.talking.callable[p.moves];count=j.sum(eligible,axis=1,dtype=j.int32);ranks=j.cumsum(eligible,axis=1)-1
    selected_mask=eligible&(ranks==(p.roll%j.maximum(1,count))[:,None]);actual_slot=j.argmax(selected_mask,axis=1).astype(j.int32)
    good &= (p.roll>=0)&(p.roll<60)&(count>0)&(captured.roll==p.roll)&(captured.selected_slot==actual_slot)&(captured.called==p.moves[rows,actual_slot])
    z=j.zeros(b,j.int32);before=captured.before;selected=j.clip(captured.selected_slot,0,3)
    clock=j.stack((j.full(b,SLEEP_TALK,j.int32),p.side,p.physical,z,p.caller,p.counter,before.counter,p.skipped,p.skipped+1,p.early.astype(j.int32)+2*p.rest.astype(j.int32)),axis=1)
    call=j.stack((j.full(b,SLEEP_TALK,j.int32),p.side,p.physical,z+CALL,p.caller,captured.called,captured.selected_slot,p.pp[rows,selected],4*captured.roll+p.slot,j.maximum(0,p.pp[rows,p.slot]-1)),axis=1)
    child=j.stack((z+public.MOVE,p.side,p.physical,1-p.side,p.target,captured.called,z,z,z,z),axis=1)
    good &= j.all(captured.raw==j.stack((clock,call,child),axis=1),axis=(1,2))&before.can_act&(before.status==5)&(before.counter==p.counter-1-p.early.astype(j.int32))&(before.counter>0)&~before.confusion_self_hit&(before.confusion_counter==0)
    for k in range(3):good &= j.all(tape.words[rows,j.clip(start+k,0,tape.words.shape[1]-1)]==captured.raw[:,k],axis=1)
    good &= (c.ledger.turn[rows,entry]==tape.cursor.turn)&(c.ledger.phase[rows,entry]==tape.cursor.phase)&j.all(c.ledger.generation[rows,entry]==tape.cursor.generation,axis=1)&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])
    return good
