"""Private SAME-call direct/status KO receipts; no terminal/physics inference.

The owning source-witness driver opens this observer before its one transition.
The original direct HP0/faint Site10 receiver is unchanged. A separate optional
companion accepts only the source-owned Site5 status HP0/faint transaction.
The final numerical assignment supplies literal done/winner and all twelve HP
and maxima. This is not an event synthesized from a terminal state, a general
terminal capability, or birth/origin attestation. Both original owner consumers
must still drain the entire frame before Core may expose a terminal closeout.
"""
from typing import NamedTuple
import jax.numpy as jnp
from . import engine_lifecycle_callbacks as life

_GLOBAL = ('engine_binding','source_binding','move_binding','producer_binding',
           'inventory_binding','direct_binding')

class Context(NamedTuple):
    engine_binding: object
    source_binding: object
    move_binding: object
    producer_binding: object
    inventory_binding: object
    direct_binding: object
    stream_id: object
    frame: object
    turn: object
    phase: object
    generation: object
    enabled: object
    before_hp: object
    before_max_hp: object
    before_done: object
    before_winner: object
    candidates: object
    slot: object
    serial: object
    actor: object
    target_physical: object
    old_hp: object
    final_hp: object
    final_max_hp: object
    final_done: object
    final_winner: object
    finalized: object
    status: object = None


def _context(tape):
    context=getattr(getattr(tape,'execution',None),'terminal',None)
    if context is not None and type(context) is not Context:
        raise ValueError('original private terminal observer required')
    return context


def _typed(c,b):
    if type(c) is not Context:raise ValueError('typed terminal receipt required')
    for name in _GLOBAL:life._typed(getattr(c,name),(8,),jnp.uint32)
    for name in ('stream_id','frame','turn','phase','candidates','slot','serial','actor','target_physical','old_hp','before_winner','final_winner','finalized'):
        life._typed(getattr(c,name),(b,),jnp.int32)
    for name in ('enabled','before_done','final_done'):life._typed(getattr(c,name),(b,),jnp.bool_)
    life._typed(c.generation,(b,2),jnp.int32)
    for name in ('before_hp','before_max_hp','final_hp','final_max_hp'):
        life._typed(getattr(c,name),(b,2,6),jnp.int32)
    if c.status is not None:_typed_status(c.status,b)


def _bound(tape,c):
    from . import engine_execution_completeness as ex
    b=len(tape.count);_typed(c,b)
    actual=ex._context(tape)
    if actual is None:return jnp.zeros(b,bool)
    good=ex._validate(actual.ledger,actual.tables)&ex._tape(actual.ledger,tape,actual.tables)
    for name in _GLOBAL[:4]:
        life._typed(getattr(tape,name),(8,),jnp.uint32)
        good &= jnp.all(getattr(c,name)==getattr(tape,name))
    good &= jnp.all(c.inventory_binding==actual.tables.inventory_binding)
    risk=actual.risks
    if risk is None:return jnp.zeros(b,bool)
    good &= jnp.any(c.direct_binding!=0)&jnp.all(c.direct_binding==risk.tables.direct_hit_binding)
    good &= c.enabled&(c.stream_id==tape.stream_id)&(c.stream_id==tape.cursor.stream_id)
    good &= (c.frame==actual.ledger.frame)&actual.ledger.opened&(c.frame>0)
    good &= life._valid(tape.cursor)&tape.cursor.initialized
    good &= (c.turn==tape.cursor.turn)&(c.phase==tape.cursor.phase)&jnp.all(c.generation==tape.cursor.generation,axis=1)
    return good


def begin_on_tape(tape,state,context,*,active,status_data=None):
    """Owning pre-transition cutpoint only; no accepted/completed argument."""
    from . import engine_execution_completeness as ex, engine_direct_hit_controls as hit
    from .state import BattleState
    if type(state) is not BattleState or type(context) is not hit.RegisteredContext:
        raise ValueError('original state and registered direct context required')
    actual=ex._context(tape)
    if actual is None:raise ValueError('terminal observer requires actual open accounting')
    b=len(tape.count);life._typed(active,(b,),jnp.bool_)
    for value in (state.hp,state.max_hp):life._typed(value,(b,2,6),jnp.int32)
    life._typed(state.done,(b,),jnp.bool_);life._typed(state.winner,(b,),jnp.int32)
    z=jnp.zeros(b,jnp.int32);no=jnp.zeros(b,bool)
    c=Context(*(getattr(tape,n) for n in _GLOBAL[:4]),actual.tables.inventory_binding,context.data.binding,
        tape.stream_id,actual.ledger.frame,tape.cursor.turn,tape.cursor.phase,tape.cursor.generation,
        active&hit._registered_bound(tape,context),state.hp,state.max_hp,state.done,state.winner,
        z,jnp.full(b,-1,jnp.int32),z,jnp.full(b,-1,jnp.int32),jnp.full(b,-1,jnp.int32),z,
        jnp.zeros_like(state.hp),jnp.zeros_like(state.max_hp),no,jnp.full(b,-1,jnp.int32),z)
    if status_data is not None:
        c=c._replace(status=_initialize_status(status_data,c.enabled,state))
    return tape._replace(execution=actual._replace(terminal=c))


def allows_last_device(tape,p):
    c=_context(tape);b=len(tape.count)
    if c is None:return jnp.zeros(b,bool)
    rows=jnp.arange(b);side=jnp.clip(p.actor,0,1);target=jnp.clip(p.target_physical,0,5)
    good=_bound(tape,c)&~c.before_done&(c.before_winner==-1)&(c.candidates==0)
    good &= jnp.all(c.before_hp>=0,axis=(1,2))&jnp.all(c.before_max_hp>0,axis=(1,2))
    good &= jnp.all(c.before_hp<=c.before_max_hp,axis=(1,2))
    good &= jnp.all(jnp.any(c.before_hp>0,axis=2),axis=1)
    good &= jnp.all(p.target_party_before==c.before_hp[rows,1-side],axis=1)
    good &= (p.actor_hp_before==c.before_hp[rows,side,jnp.clip(p.actor_physical,0,5)])
    good &= (p.write.maximum==c.before_max_hp[rows,1-side,target])
    good &= (jnp.sum(p.target_party_before>0,axis=1)==1)&(p.target_party_before[rows,target]>0)
    return good


def capture_ko_on_tape(tape,p,good):
    """Called only immediately after existing exact HP0/faint span is stage2."""
    c=_context(tape)
    if c is None:return tape
    from . import engine_execution_completeness as ex
    ledger=tape.execution.ledger;rows=jnp.arange(len(tape.count))
    slot,same=ex._ticket(ledger,p.ticket)
    use=p.requested&good&allows_last_device(tape,p)&same&(ledger.stage[rows,slot]==2)
    return tape._replace(execution=tape.execution._replace(terminal=c._replace(
        candidates=c.candidates+use.astype(jnp.int32),slot=jnp.where(use,slot,c.slot),
        serial=jnp.where(use,p.ticket.serial,c.serial),actor=jnp.where(use,p.actor,c.actor),
        target_physical=jnp.where(use,p.target_physical,c.target_physical),old_hp=jnp.where(use,p.write.old_hp,c.old_hp))))


def _candidate(tape,c):
    from . import engine_execution_completeness as ex,engine_public_events as public
    b=len(tape.count);rows=jnp.arange(b);ledger=tape.execution.ledger
    slot=jnp.clip(c.slot,0,ledger.site.shape[1]-1);start=ledger.start[rows,slot];end=ledger.end[rows,slot]
    good=_bound(tape,c)&(c.candidates==1)&(c.slot>=0)&(c.slot<ledger.count)
    good &= (ledger.serial[rows,slot]==c.serial)&(ledger.site[rows,slot]==int(ex.Site.DIRECT_HIT))&(ledger.stage[rows,slot]==2)
    good &= (ledger.side[rows,slot]==1-c.actor)&(ledger.turn[rows,slot]==c.turn)&(ledger.phase[rows,slot]==c.phase)&jnp.all(ledger.generation[rows,slot]==c.generation,axis=1)
    good &= (end>=start+2)&(end<=tape.count)&(end<=tape.words.shape[1])
    hp=tape.words[rows,jnp.clip(end-2,0,tape.words.shape[1]-1)];faint=tape.words[rows,jnp.clip(end-1,0,tape.words.shape[1]-1)]
    good &= (hp[:,0]==public.MAIN_HP)&(hp[:,3]==1-c.actor)&(hp[:,4]==c.target_physical)&(hp[:,5]==0)&(hp[:,6]==c.before_max_hp[rows,1-jnp.clip(c.actor,0,1),jnp.clip(c.target_physical,0,5)])
    good &= (faint[:,0]==public.DIRECT_FAINT)&(faint[:,3]==1-c.actor)&(faint[:,4]==c.target_physical)&jnp.all(faint[:,5:]==0,axis=1)
    span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<end[:,None])
    good &= jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)&(tape.event_turn==c.turn[:,None])&(tape.event_phase==c.phase[:,None])&jnp.all(tape.event_generation==c.generation[:,None,:],axis=2)),axis=1)
    return good


def capture_final_on_tape(tape,state,*,active):
    """Immediately AFTER the original final done/winner assignment, no math."""
    c=_context(tape)
    if c is None:return tape
    b=len(tape.count);life._typed(active,(b,),jnp.bool_)
    for value in (state.hp,state.max_hp):life._typed(value,(b,2,6),jnp.int32)
    life._typed(state.done,(b,),jnp.bool_);life._typed(state.winner,(b,),jnp.int32)
    use=active&c.enabled
    updated=c._replace(final_hp=jnp.where(use[:,None,None],state.hp,c.final_hp),
        final_max_hp=jnp.where(use[:,None,None],state.max_hp,c.final_max_hp),
        final_done=jnp.where(use,state.done,c.final_done),final_winner=jnp.where(use,state.winner,c.final_winner),
        finalized=c.finalized+use.astype(jnp.int32))
    return tape._replace(execution=tape.execution._replace(terminal=updated))


def _matches_direct_final_device(tape,state):
    c=_context(tape);b=len(tape.count)
    if c is None:return jnp.zeros(b,bool)
    rows=jnp.arange(b);side=jnp.clip(c.actor,0,1);target=jnp.clip(c.target_physical,0,5)
    expected=c.before_hp.at[rows,1-side,target].set(0)
    good=_candidate(tape,c)&(c.finalized==1)&~c.before_done&(c.before_winner==-1)&c.final_done
    good &= (c.actor>=0)&(c.actor<2)&(c.target_physical>=0)&(c.target_physical<6)&(c.old_hp==c.before_hp[rows,1-side,target])&(c.old_hp>0)
    good &= jnp.all(c.final_hp==expected,axis=(1,2))&jnp.all(c.final_max_hp==c.before_max_hp,axis=(1,2))
    alive=jnp.any(c.final_hp>0,axis=2)
    good &= alive[rows,side]&~alive[rows,1-side]&(c.final_winner==c.actor)
    good &= state.done&(state.winner==c.final_winner)&jnp.all(state.hp==c.final_hp,axis=(1,2))&jnp.all(state.max_hp==c.final_max_hp,axis=(1,2))
    return good


def commit_rows(active,before,after):
    """No GLOBAL8 row-selection, including B8; inactive rows stay bitwise old."""
    b=len(active);life._typed(active,(b,),jnp.bool_);_typed(before,b);_typed(after,b)
    same=jnp.array(True)
    for name in _GLOBAL:same &= jnp.all(getattr(before,name)==getattr(after,name))
    for name in ('stream_id','frame','turn','phase','generation','enabled','before_hp','before_max_hp','before_done','before_winner'):
        old,new=getattr(before,name),getattr(after,name)
        same &= jnp.all((old==new).reshape((b,-1)),axis=1)
    use=active&same
    return before._replace(**{name:jnp.where(use.reshape((b,)+(1,)*(old.ndim-1)),new,old)
        for name,old,new in zip(before._fields,before,after) if name not in _GLOBAL and name!='status'},status=_commit_status_rows(use,before.status,after.status))


class StatusReceipt(NamedTuple):
    data: object
    enabled: object
    candidates: object
    slot: object
    serial: object
    side: object
    physical: object
    old_hp: object
    before_hp: object
    before_max_hp: object


def _typed_status(s,b):
    from . import engine_status_residual_callbacks as status
    if type(s)is not StatusReceipt or type(s.data)is not status.Data:raise ValueError('original residual terminal companion required')
    for x in s.data[:5]:life._typed(x,(8,),jnp.uint32)
    life._typed(s.data.reason_by_status,(7,),jnp.int32);life._typed(s.data.damage_tag_by_reason,(4,),jnp.int32)
    life._typed(s.data.singleton_order,(),jnp.bool_);life._typed(s.data.lethal_source,(),jnp.bool_)
    life._typed(s.enabled,(b,),jnp.bool_)
    for n in ('candidates','slot','serial','side','physical','old_hp'):life._typed(getattr(s,n),(b,),jnp.int32)
    for n in ('before_hp','before_max_hp'):life._typed(getattr(s,n),(b,2,6),jnp.int32)


def _initialize_status(data,enabled,state):
    b=len(enabled);z=jnp.zeros(b,jnp.int32);absent=jnp.full(b,-1,jnp.int32)
    s=StatusReceipt(data,enabled&data.lethal_source,z,absent,z,absent,absent,z,jnp.zeros_like(state.hp),jnp.zeros_like(state.max_hp));_typed_status(s,b);return s


def _commit_status_rows(active,before,after):
    if before is None and after is None:return None
    b=len(active);_typed_status(before,b);_typed_status(after,b)
    same=jnp.array(True)
    for a,z in zip(before.data,after.data):same &= jnp.all(a==z)
    use=active&same&(before.enabled==after.enabled)
    return before._replace(**{n:jnp.where(use.reshape((b,)+(1,)*(a.ndim-1)),z,a)for n,a,z in zip(before._fields,before,after)if n!='data'})


def capture_status_on_tape(tape,capture,good):
    """Only a just-claimed original Site5 HP0→faint can be a terminal candidate."""
    from . import engine_status_residual_callbacks as status,engine_execution_completeness as ex
    c=_context(tape)
    if c is None or c.status is None:return tape
    b=len(tape.count);rows=jnp.arange(b);s=c.status;_typed_status(s,b);p=capture.pending;ledger=tape.execution.ledger;slot,same=ex._ticket(ledger,p.ticket)
    victim=jnp.clip(p.side,0,1);physical=jnp.clip(p.physical,0,5)
    source=jnp.array(True)
    for a,z in zip(s.data,p.data):source &= jnp.all(a==z)
    last=(jnp.sum(p.hp[rows,victim]>0,axis=1)==1)&(p.hp[rows,victim,physical]>0)
    use=(good&source&_bound(tape,c)&s.enabled&same&(c.candidates==0)&(s.candidates==0)&last
        &status.owned_lethal(tape,ledger.start[rows,slot],s.data)&(ledger.end[rows,slot]==tape.count))
    updated=s._replace(candidates=s.candidates+use.astype(jnp.int32),slot=jnp.where(use,slot,s.slot),serial=jnp.where(use,p.ticket.serial,s.serial),side=jnp.where(use,p.side,s.side),physical=jnp.where(use,p.physical,s.physical),old_hp=jnp.where(use,p.hp[rows,victim,physical],s.old_hp),before_hp=jnp.where(use[:,None,None],p.hp,s.before_hp),before_max_hp=jnp.where(use[:,None,None],p.maximum,s.before_max_hp))
    return tape._replace(execution=tape.execution._replace(terminal=c._replace(status=updated)))


def _matches_status_final_device(tape,state):
    from . import engine_status_residual_callbacks as status,engine_execution_completeness as ex
    c=_context(tape);b=len(tape.count)
    if c is None or c.status is None:return jnp.zeros(b,bool)
    s=c.status;_typed_status(s,b);rows=jnp.arange(b);ledger=tape.execution.ledger;slot=jnp.clip(s.slot,0,ledger.site.shape[1]-1);victim=jnp.clip(s.side,0,1);physical=jnp.clip(s.physical,0,5)
    good=(_bound(tape,c)&s.enabled&(c.candidates==0)&(s.candidates==1)&(s.slot>=0)&(s.slot<ledger.count)&(ledger.serial[rows,slot]==s.serial)&(ledger.side[rows,slot]==s.side)&status.owned_lethal(tape,ledger.start[rows,slot],s.data))
    raw=tape.words[rows,jnp.clip(ledger.start[rows,slot],0,tape.words.shape[1]-1)]
    good &= ((raw[:,3]==s.side)&(raw[:,4]==s.physical)&(raw[:,5]==s.old_hp)
        &(raw[:,5]==s.before_hp[rows,victim,physical])
        &(raw[:,7]==s.before_max_hp[rows,victim,physical]))
    expected=s.before_hp.at[rows,victim,physical].set(0)
    good &= ((c.finalized==1)&~c.before_done&(c.before_winner==-1)&c.final_done
        &(s.side>=0)&(s.side<2)&(s.physical>=0)&(s.physical<6)&(s.old_hp>0)&(s.old_hp==s.before_hp[rows,victim,physical])
        &jnp.all((s.before_hp>=0)&(s.before_max_hp>0)&(s.before_hp<=s.before_max_hp),axis=(1,2))&jnp.all(jnp.any(s.before_hp>0,axis=2),axis=1)
        &jnp.all(c.final_hp==expected,axis=(1,2))&jnp.all(c.final_max_hp==s.before_max_hp,axis=(1,2)))
    alive=jnp.any(c.final_hp>0,axis=2)
    good &= ~alive[rows,victim]&alive[rows,1-victim]&(c.final_winner==1-s.side)
    good &= state.done&(state.winner==c.final_winner)&jnp.all(state.hp==c.final_hp,axis=(1,2))&jnp.all(state.max_hp==c.final_max_hp,axis=(1,2))
    return good


def matches_final_device(tape,state):
    """Disjoint original direct-KO or exact residual source transactions only."""
    return _matches_direct_final_device(tape,state)|_matches_status_final_device(tape,state)
