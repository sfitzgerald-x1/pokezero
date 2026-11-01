"""Prospective SAME-write direct KO, not generic lethal admission.

Existing RAW controls->MAIN_HP0->DIRECT_FAINT and Site10 are reused. The typed
pending source snapshots actual pre-write actor HP/target party HP and literal
HitWrite; it earns NO stage2 until the actual later faint callback completes
its exact contiguous span. Missing/interleaved/self/recoil/last-mon paths fail
closed by default. A private original-driver terminal observer additionally
binds a last-mon KO to its literal final write and both later full drains.
Original owning host guards remain required; device copies do not
attest origin. No damage, HP, status, RNG or winner formula is implemented.
"""
from typing import NamedTuple
import jax.numpy as jnp
from . import engine_direct_hit_controls as hit,engine_public_events as public,engine_lifecycle_callbacks as life

UNKNOWN_KO=256

class Pending(NamedTuple):
    context:hit.RegisteredContext
    ticket:object
    actor:object
    actor_physical:object
    target_physical:object
    move:object
    write:hit.HitWrite
    actor_hp_before:object
    target_party_before:object
    requested:object
    actor_happiness:object=None
    recoil_before:object=None


def _literal(p,b,terminal_last=None):
    if type(p) is not Pending or type(p.context) is not hit.RegisteredContext or type(p.write) is not hit.HitWrite:raise ValueError('typed literal direct KO pending required')
    for x in (p.actor,p.actor_physical,p.target_physical,p.move,p.actor_hp_before):life._typed(x,(b,),jnp.int32)
    life._typed(p.target_party_before,(b,6),jnp.int32);life._typed(p.requested,(b,),jnp.bool_)
    w=hit._typed_write(p.write,b);data=p.context.data;life._typed(data.ko_candidates,(int(data.ko_candidates.shape[0]),),jnp.bool_)
    rows=jnp.arange(b);target=jnp.clip(p.target_physical,0,5)
    other=jnp.arange(6)[None,:]!=target[:,None]
    life._typed(data.return_ko_move,(),jnp.int32)
    is_return=(data.return_ko_move>0)&(p.move==data.return_ko_move)
    return_power=jnp.zeros(b,bool)
    if p.actor_happiness is not None:
        life._typed(p.actor_happiness,(b,),jnp.int32)
        return_power=(p.actor_happiness>=0)&(p.actor_happiness<=255)&(w.power==jnp.maximum(1,p.actor_happiness*10//25))
    from . import engine_recoil_callbacks as recoil
    composite=recoil.ko_literal(p,b)
    return ((~is_return|return_power)&(w.power>0)&w.hit&~w.immune&~w.substitute&~w.protected&~w.enduring
        &hit._positive_ratio(w.numerator,w.denominator)&(w.old_hp>0)&(w.new_hp==0)&(w.maximum>=w.old_hp)
        &(p.actor_hp_before>0)&jnp.all(p.target_party_before>=0,axis=1)
        &(p.target_party_before[rows,target]==w.old_hp)&(jnp.any(other&(p.target_party_before>0),axis=1)
            |(jnp.zeros(b,bool) if terminal_last is None else terminal_last))
        &(p.move>0)&(p.move<data.move_count)&(data.ko_candidates[jnp.clip(p.move,0,len(data.ko_candidates)-1)]|composite))


def _prefix(tape,p):
    from . import engine_execution_completeness as ex
    b=tape.count.shape[0];rows=jnp.arange(b);context=ex._context(tape)
    if context is None:return jnp.zeros(b,bool),jnp.zeros(b,jnp.int32),jnp.zeros(b,jnp.int32)
    from . import engine_terminal_callbacks as terminal
    good=hit._registered_bound(tape,p.context)&_literal(p,b,terminal.allows_last_device(tape,p));ledger=context.ledger;slot,same=ex._ticket(ledger,p.ticket);start=ledger.start[rows,slot]
    if p.recoil_before is not None:
        from . import engine_recoil_callbacks as recoil
        good &= ~recoil.ko_selected(p)|recoil._bound(tape,p.recoil_before.context)
    good &= same&life._valid(tape.cursor)&tape.cursor.initialized&(p.actor>=0)&(p.actor<2)
    safe=jnp.clip(p.actor,0,1)
    good &= (p.actor_physical==tape.cursor.physical_active[rows,safe])&(p.target_physical==tape.cursor.physical_active[rows,1-safe])
    good &= (ledger.site[rows,slot]==int(ex.Site.DIRECT_HIT))&(ledger.side[rows,slot]==1-p.actor)&(ledger.stage[rows,slot]==1)
    good &= (start>0)&(tape.cursor.turn==ledger.turn[rows,slot])&(tape.cursor.phase==0)&(tape.cursor.phase==ledger.phase[rows,slot])&jnp.all(tape.cursor.generation==ledger.generation[rows,slot],axis=1)
    at=jnp.clip(start-1,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    good &= (raw[:,0]==public.MOVE)&(raw[:,1]==p.actor)&(raw[:,2]==p.actor_physical)&(raw[:,3]==1-p.actor)&(raw[:,4]==p.target_physical)&(raw[:,5]==p.move)&jnp.all(raw[:,6:]==0,axis=1)
    good &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    return good,slot,start


def prepare_controls_device(tape,ticket,actor,actor_physical,target_physical,move,write,context,actor_hp_before,target_party_before,*,eligible,actor_happiness=None,recoil_before=None):
    """AFTER SAME HP assignment, BEFORE HP callback/early outcome. No credit."""
    from . import engine_execution_completeness as ex
    b=tape.count.shape[0];life._typed(eligible,(b,),jnp.bool_)
    requested=eligible&(write.new_hp==0)
    p=Pending(context,ticket,actor,actor_physical,target_physical,move,write,actor_hp_before,target_party_before,requested,actor_happiness,recoil_before)
    good,_,start=_prefix(tape,p);good &= tape.count==start
    critical,effect=hit.annotation_flags_device(write,data=context.data,move=move)
    need=critical.astype(jnp.int32)+effect.astype(jnp.int32)
    room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-need-2);overflow=requested&good&~room
    rows=jnp.arange(b)
    for offset in range(4):
        address=tape.count+offset;inside=(address>=0)&(address<tape.words.shape[1]);at=jnp.clip(address,0,tape.words.shape[1]-1)
        good &= (offset>=need+2)|~inside|jnp.all(tape.switch_details[rows,at]==0,axis=1)
    use=requested&good&room
    facts=(move,write.old_hp,write.new_hp,write.numerator,write.denominator)
    for kind,flag in ((hit.CRITICAL,critical),(hit.SUPER_EFFECTIVE,effect&(write.numerator>write.denominator)),(hit.RESISTED,effect&(write.numerator<write.denominator))):
        tape=public._append(tape,kind,actor,actor_physical,1-actor,target_physical,facts,use&flag)
    actual=ex._context(tape)
    if actual is not None:
        ledger=ex.unhandled(actual.ledger,ticket,active=requested&~(good&room),tables=actual.tables);tape=ex._put_context(tape,ledger,actual)
    return tape._replace(refusal=tape.refusal|jnp.where(requested&~(good&room),UNKNOWN_KO,0)|jnp.where(overflow,public.OVERFLOW,0)),p


def _span(tape,p,*,with_faint):
    from . import engine_execution_completeness as ex
    good,slot,start=_prefix(tape,p);actual=ex._context(tape)
    if actual is None:return good
    ledger=actual.ledger;b=tape.count.shape[0];rows=jnp.arange(b);w=p.write
    critical,effect=hit.annotation_flags_device(w,data=p.context.data,move=p.move)
    from . import engine_recoil_callbacks as recoil
    composite=recoil.ko_selected(p)
    controls=critical.astype(jnp.int32)+effect.astype(jnp.int32);length=controls+1+int(with_faint)+jnp.where(with_faint,composite.astype(jnp.int32),0)
    good &= p.requested&(tape.count==start+length)
    expected=jnp.stack((p.actor,p.actor_physical,1-p.actor,p.target_physical,p.move,w.old_hp,w.new_hp,w.numerator,w.denominator),axis=1)
    for offset,flag,kind in ((jnp.zeros(b,jnp.int32),critical,jnp.full(b,hit.CRITICAL,jnp.int32)),(critical.astype(jnp.int32),effect,jnp.where(w.numerator>w.denominator,hit.SUPER_EFFECTIVE,hit.RESISTED))):
        raw=tape.words[rows,jnp.clip(start+offset,0,tape.words.shape[1]-1)]
        good &= ~flag|((raw[:,0]==kind)&jnp.all(raw[:,1:]==expected,axis=1))
    hp=tape.words[rows,jnp.clip(start+controls,0,tape.words.shape[1]-1)]
    good &= (hp[:,0]==public.MAIN_HP)&jnp.all(hp[:,1:3]==-1,axis=1)&(hp[:,3]==1-p.actor)&(hp[:,4]==p.target_physical)&(hp[:,5]==0)&(hp[:,6]==w.maximum)&(hp[:,7]>=0)&(hp[:,7]<=6)&jnp.all(hp[:,8:]==0,axis=1)
    if with_faint:
        raw=tape.words[rows,jnp.clip(start+controls+1+composite.astype(jnp.int32),0,tape.words.shape[1]-1)]
        if p.recoil_before is not None:good &= ~composite|recoil.ko_recoil_word(tape,p,start+controls+1)
        good &= (raw[:,0]==public.DIRECT_FAINT)&jnp.all(raw[:,1:3]==-1,axis=1)&(raw[:,3]==1-p.actor)&(raw[:,4]==p.target_physical)&jnp.all(raw[:,5:]==0,axis=1)
    span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    good &= jnp.all(~span|((tape.event_turn==ledger.turn[rows,slot,None])&(tape.event_phase==ledger.phase[rows,slot,None])&jnp.all(tape.event_generation==ledger.generation[rows,slot,None,:],axis=2)),axis=1)
    return good


def matches_outcome_device(tape,p,outcome,tables):
    b=tape.count.shape[0];good=_span(tape,p,with_faint=False)
    life._typed(tables.direct_ko_candidates,(len(tables.selected_risk),),jnp.bool_)
    good &= jnp.any(tables.direct_hit_binding!=0)&jnp.all(p.context.data.binding==tables.direct_hit_binding)&jnp.all(p.context.data.ko_candidates==tables.direct_ko_candidates)
    from . import engine_recoil_callbacks as recoil
    composite=recoil.ko_selected(p)
    amount=jnp.zeros(b,jnp.int32)if p.recoil_before is None else p.recoil_before.amount
    if p.recoil_before is not None:
        life._typed(tables.recoil_ko,(),jnp.bool_);life._typed(tables.recoil_binding,(8,),jnp.uint32)
        good &= ~composite|(tables.recoil_ko&jnp.all(tables.recoil_binding==p.recoil_before.context.data.binding))
    good &= (p.context.data.seismic_toss_move==tables.seismic_toss_move)&jnp.where(composite,(outcome.recoil==amount)&(amount>0)&(p.actor_hp_before>amount),(outcome.recoil==0))&~outcome.explodes&~outcome.absorbed
    for name in ('power','critical','hit','immune','numerator','denominator','substitute','protected','enduring'):good &= getattr(outcome,name)==getattr(p.write,name)
    return good


def finish_device(tape,p,actor_hp,target_party,actor_physical,target_physical,*,recoil_after=None):
    """Immediately AFTER original faint append, then exact stage2 or sticky fail."""
    from . import engine_execution_completeness as ex
    b=tape.count.shape[0];rows=jnp.arange(b)
    for x in (actor_hp,actor_physical,target_physical):life._typed(x,(b,),jnp.int32)
    life._typed(target_party,(b,6),jnp.int32)
    expected=p.target_party_before.at[rows,jnp.clip(p.target_physical,0,5)].set(0)
    from . import engine_recoil_callbacks as recoil
    composite=recoil.ko_selected(p)
    good=_span(tape,p,with_faint=True)&~composite&(actor_hp==p.actor_hp_before)&(actor_physical==p.actor_physical)&(target_physical==p.target_physical)&jnp.all(target_party==expected,axis=1)
    actual=ex._context(tape)
    if actual is not None:
        ledger=ex.unhandled(actual.ledger,p.ticket,active=p.requested&~composite&~good,tables=actual.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=actual.tables);tape=ex._put_context(tape,ledger,actual)
    tape=tape._replace(refusal=tape.refusal|jnp.where(p.requested&~composite&~good,UNKNOWN_KO,0))
    if p.recoil_before is not None:tape=recoil.finish_ko_on_tape(tape,p,actor_hp,target_party,actor_physical,target_physical,recoil_after)
    from . import engine_terminal_callbacks as terminal
    return terminal.capture_ko_on_tape(tape,p,good)


def validate_owned_control_device(tape,index,raw):
    """Exact zero-HP control suffix, only inside actual snapshotted Site10 KO."""
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=tape.count.shape[0];no=jnp.zeros(b,bool)
    if actual is None:return no
    ledger=actual.ledger;rows=jnp.arange(b);claim=ledger.claimed[:,index];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1)
    syntax=((raw[:,0]==hit.CRITICAL)|((raw[:,0]==hit.SUPER_EFFECTIVE)&(raw[:,8]>raw[:,9]))|((raw[:,0]==hit.RESISTED)&(raw[:,8]<raw[:,9])))&(raw[:,6]>0)&(raw[:,7]==0)&hit._positive_ratio(raw[:,8],raw[:,9])
    needs=(raw[:,0]==hit.CRITICAL)&(raw[:,8]!=raw[:,9]);hp_at=index+1+needs.astype(jnp.int32);faint_at=hp_at+1
    hp=tape.words[rows,jnp.clip(hp_at,0,tape.words.shape[1]-1)];faint=tape.words[rows,jnp.clip(faint_at,0,tape.words.shape[1]-1)]
    good=syntax&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.DIRECT_HIT))&(ledger.stage[rows,slot]==2)&(ledger.end[rows,slot]==faint_at+1)&(faint_at<tape.count)
    good &= (hp[:,0]==public.MAIN_HP)&(hp[:,3]==raw[:,3])&(hp[:,4]==raw[:,4])&(hp[:,5]==0)&(hp[:,6]>=raw[:,6])&(faint[:,0]==public.DIRECT_FAINT)&(faint[:,3]==raw[:,3])&(faint[:,4]==raw[:,4])
    span=(jnp.arange(tape.words.shape[1])[None,:]>=ledger.start[rows,slot,None])&(jnp.arange(tape.words.shape[1])[None,:]<ledger.end[rows,slot,None])
    return good&jnp.all(~span|((ledger.claimed==slot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)&(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)&jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
