"""Actual Natural Cure departure, RAW60; no numerical/status/RNG writes.

Ordinary cures own Site45. A Baton cure is the middle word of the SAME Site39
INSTALL/cure/SWITCH composite, never an overlapping second claim. All private
before/after values are actual entry.depart cutpoints. Missing facts stay closed.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
from . import engine_lifecycle_callbacks as life,engine_public_events as public

NATURAL_CURE=60
FIELDS=('hp','max_hp','status','status_counter','sleep_skipped','rest_origin','ability','base_ability',
 'types','base_types','species','level','gender','stats','moves','pp','item','hidden_power_type','hidden_power_power','happiness','boosts','volatile',
 'confusion_counter','protect_counter','substitute_hp','choice_move','last_move','encore_move',
 'encore_counter','charge_move','charge_counter','truant_loaf','trap_source','active','keys','baton_pass')
Snapshot=NamedTuple('Snapshot',[(n,object)for n in FIELDS])
class Pending(NamedTuple):
    ticket:object
    side:object
    physical:object
    incoming:object
    composite:object
    requested:object
    valid:object
    before:object
    start:object
    context:object
    entry_mask:object
    source_record:object
    eligible:object
class Receipt(NamedTuple):
    pending:object
    after:object
    index:object
    emitted:object

def snapshot(state):
    values=tuple(getattr(state,n,None)for n in FIELDS)
    if any(v is None for v in values):return None
    return Snapshot(*values)

def equal(a,b):
    if a is None or b is None:return False
    if type(a)is not Snapshot or type(b)is not Snapshot:raise ValueError('actual departure snapshots required')
    result=j.ones(a.active.shape[0],bool)
    for old,new in zip(a,b,strict=True):
        if old.shape!=new.shape or old.dtype!=new.dtype:raise ValueError('same original departure geometry required')
        result &= j.all((old==new).reshape((len(result),-1)),axis=1)
    return result

def source_status(ability,status,counter,data):
    cap=getattr(data,'natural_cure',None)
    if cap is None:return j.zeros_like(status,bool)
    life._typed(cap,data.quiet_abilities.shape,j.bool_)
    return (ability>0)&(ability<len(cap))&cap[j.clip(ability,0,len(cap)-1)]&(status>0)&(status<=6)&(counter>=0)&(counter<=15)

def expected(before,side,physical):
    # Validate only the already-existing entry.depart writes. No physical
    # function is rerun and no status/probability/HP fact is reconstructed.
    r=j.arange(len(side));s=j.clip(side,0,1);p=j.clip(physical,0,5);values={}
    for name in ('status','status_counter','sleep_skipped','rest_origin','boosts','volatile','confusion_counter',
                 'protect_counter','substitute_hp','choice_move','last_move','encore_move','encore_counter',
                 'charge_move','charge_counter','truant_loaf'):
        a=getattr(before,name);values[name]=a.at[r,s,p].set(j.zeros_like(a[r,s,p]))
    values['ability']=before.ability.at[r,s,p].set(before.base_ability[r,s,p])
    values['types']=before.types.at[r,s,p].set(before.base_types[r,s,p])
    return before._replace(**values)

def _private_valid(p,after,data):
    b=len(p.side)
    if p.before is None or after is None:return j.zeros(b,bool)
    old=p.before;r=j.arange(b);side=j.clip(p.side,0,1);mon=j.clip(p.physical,0,5);new=j.clip(p.incoming,0,5)
    good=(p.side>=0)&(p.side<2)&(p.physical>=0)&(p.physical<6)&(p.incoming>=0)&(p.incoming<6)&(p.incoming!=p.physical)
    good &= (old.active[r,side]==p.physical)&(old.hp[r,side,mon]>0)&(old.hp[r,side,mon]<=old.max_hp[r,side,mon])&(old.hp[r,side,new]>0)
    good &= source_status(old.ability[r,side,mon],old.status[r,side,mon],old.status_counter[r,side,mon],data)
    # clearStatus can separately emit Nightmare END. This source port records
    # that branch only when the actual represented volatile word is zero.
    good &= (old.volatile[r,side,mon]==0)&j.all(old.trap_source<0,axis=(1,2))
    good &= equal(expected(old,p.side,p.physical),after)
    return good

def _start_valid(tape,p):
    from . import engine_baton_pass_callbacks as bp,engine_direct_hit_controls as direct
    if p.before is None or p.context is None or p.context.data.baton_pass is None:return j.zeros_like(p.side,bool)
    d=p.context.data.baton_pass;b=len(p.side);r=j.arange(b);s=j.clip(p.side,0,1);old=j.clip(p.physical,0,5);new=j.clip(p.incoming,0,5)
    good=direct._registered_bound(tape,p.context)&bp._bound(tape,d)
    if d.natural_cure is None:return good&False
    life._typed(p.entry_mask,d.natural_cure.shape,j.bool_)
    good &= j.all(p.entry_mask==d.natural_cure)
    good &= p.eligible&(p.side>=0)&(p.side<2)&(p.physical>=0)&(p.physical<6)&(p.incoming>=0)&(p.incoming<6)&(p.incoming!=p.physical)
    good &= (p.before.active[r,s]==p.physical)&(p.before.hp[r,s,old]>0)&(p.before.hp[r,s,new]>0)&(p.composite==p.before.baton_pass[r,s])
    good &= source_status(p.before.ability[r,s,old],p.before.status[r,s,old],p.before.status_counter[r,s,old],d)
    if p.source_record is None:return good&~p.composite
    rec=p.source_record;h=rec.history
    copied_boosts=rec.boost_before.at[r,s,new].set(rec.boost_before[r,s,old])
    private=bp.Private(*(getattr(p.before,n)for n in bp.PRIVATE_FIELDS))
    copied=(rec.code==bp.INSTALL)&rec.known&(rec.side==p.side)&(rec.outgoing==p.physical)&(rec.incoming==p.incoming)&(rec.start+1==p.start)
    copied &= (rec.slot==p.ticket.slot)&(rec.serial==p.ticket.serial)&(rec.frame==p.ticket.frame)&h.ready&(h.frame+1==rec.frame)
    copied &= bp.equal(rec.before,h.private)&bp.equal(rec.after,bp.copied_private(rec.before,p.side,p.physical,p.incoming))&bp.equal(private,rec.after)
    copied &= j.all(copied_boosts==p.before.boosts,axis=(1,2,3))&j.all(copied_boosts==rec.boost_after,axis=(1,2,3))
    marker=tape.words[r,j.clip(p.start-1,0,tape.words.shape[1]-1)]
    copied &= (marker[:,0]==bp.BATON_PASS)&(marker[:,1]==p.side)&(marker[:,2]==p.physical)&(marker[:,3]==p.incoming)&(marker[:,4]==bp.INSTALL)
    return good&(~p.composite|copied)

def begin(tape,state,side,incoming,table,context,*,eligible,baton):
    from . import engine_execution_completeness as ex,engine_baton_pass_callbacks as bp,engine_direct_hit_controls as direct
    if table.entry is None or context is None or context.data.baton_pass is None or getattr(context.data.baton_pass,'natural_cure',None)is None:return tape,None
    b=len(side);r=j.arange(b);data=context.data.baton_pass;physical=state.active[r,side];before=snapshot(state)
    for a in (side,incoming):life._typed(a,(b,),j.int32)
    for a in (eligible,baton):life._typed(a,(b,),j.bool_)
    if before is None:return tape,None
    abi=before.ability[r,side,physical];status=before.status[r,side,physical]
    potential=eligible&(before.hp[r,side,physical]>0)&(status!=0)&table.entry.natural_cure[abi]
    good=direct._registered_bound(tape,context)&bp._bound(tape,data)&source_status(abi,status,before.status_counter[r,side,physical],data)
    if table.entry.natural_cure.shape!=data.natural_cure.shape:raise ValueError('same original Natural Cure ability namespace required')
    good &= j.all(table.entry.natural_cure==data.natural_cure)
    tape,ticket=ex.open_attempt(tape,ex.Site.NATURAL_CURE,side,active=potential&~baton)
    tr=tape.execution.baton_transaction
    if tr is None:
        good &= ~baton
    else:
        rec=tr.record;composite=rec.known&(rec.code==bp.INSTALL)&(rec.side==side)&(rec.outgoing==physical)&(rec.incoming==incoming)&(tape.count==rec.start+1)
        composite &= bp.equal(rec.after,bp.snapshot(state))&j.all(rec.boost_after==state.boosts,axis=(1,2,3))
        good &= ~baton|composite
        ticket=ex.Ticket(ticket.stream_id,ticket.inventory_binding,j.where(baton,rec.frame,ticket.frame),j.where(baton,rec.slot,ticket.slot),j.where(baton,rec.serial,ticket.serial))
    return tape,Pending(ticket,side,physical,incoming,baton,potential,good,before,tape.count,context,table.entry.natural_cure,None if tr is None else tr.record,eligible)

def prospective(p):
    if p is None:return False
    # The final same-write check remains mandatory, not a supplied acceptance.
    return p.requested&p.valid

def words(p):
    r=j.arange(len(p.side));side=j.clip(p.side,0,1);mon=j.clip(p.physical,0,5);old=p.before;z=j.zeros_like(p.side)
    return j.stack((z+NATURAL_CURE,p.side,p.physical,old.status[r,side,mon],old.status_counter[r,side,mon],
        old.sleep_skipped[r,side,mon],old.rest_origin[r,side,mon].astype(j.int32),old.ability[r,side,mon],old.hp[r,side,mon],old.max_hp[r,side,mon]),axis=1)

def finish(tape,p,state):
    if p is None:return tape
    from . import engine_execution_completeness as ex,engine_baton_pass_callbacks as bp
    if p.before is None or p.context is None or p.context.data.baton_pass is None:
        c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested,tables=c.tables)
        return tape._replace(refusal=tape.refusal|j.where(p.requested,128,0),execution=c._replace(ledger=ledger))
    data=p.context.data.baton_pass;after=snapshot(state);good=p.valid&_start_valid(tape,p)&_private_valid(p,after,data)
    slot,same=ex._ticket(tape.execution.ledger,p.ticket);r=j.arange(len(p.side));ledger=tape.execution.ledger
    good &= same&(tape.count==p.start)&(ledger.stage[r,slot]==1)&(ledger.site[r,slot]==j.where(p.composite,int(ex.Site.BATON_PASS),int(ex.Site.NATURAL_CURE)))&(ledger.start[r,slot]==p.start-p.composite.astype(j.int32))
    captures=tape.execution.natural_cure
    room=j.ones_like(good)if captures is None or len(captures)<2 else j.zeros_like(good)
    good &= room
    raw=words(p);use=p.requested&good
    tape=public._append(tape,NATURAL_CURE,p.side,p.physical,raw[:,3],raw[:,4],tuple(raw[:,i]for i in range(5,10)),use)
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables)
    ledger=ex.emitted(ledger,p.ticket,tape,active=use&~p.composite,tables=c.tables)
    receipt=Receipt(p,after,p.start,use)
    values=(receipt,)if captures is None else captures+(receipt,)if len(captures)<2 else captures
    return tape._replace(refusal=tape.refusal|j.where(p.requested&~good,128,0),execution=c._replace(ledger=ledger,natural_cure=values))

def _matches(tape,receipt,index,*,stage):
    from . import engine_execution_completeness as ex,engine_baton_pass_callbacks as bp
    p=receipt.pending;b=len(p.side);r=j.arange(b)
    if p.before is None or receipt.after is None or p.context is None or p.context.data.baton_pass is None:return j.zeros(b,bool)
    data=p.context.data.baton_pass;ledger=tape.execution.ledger;slot,same=ex._ticket(ledger,p.ticket)
    at=j.broadcast_to(j.asarray(index,j.int32),(b,));pos=j.clip(at,0,tape.words.shape[1]-1);raw=tape.words[r,pos]
    good=(p.requested&p.valid&receipt.emitted&_start_valid(tape,p)&_private_valid(p,receipt.after,data)&same&(at==receipt.index)&(at==p.start)&(at>=0)&(at<tape.count)&j.all(raw==words(p),axis=1)&j.all(tape.switch_details[r,pos]==0,axis=1))
    good &= (ledger.site[r,slot]==j.where(p.composite,int(ex.Site.BATON_PASS),int(ex.Site.NATURAL_CURE)))&(ledger.side[r,slot]==p.side)&(ledger.stage[r,slot]==stage)&(ledger.start[r,slot]==at-p.composite.astype(j.int32))
    good &= (tape.event_turn[r,pos]==ledger.turn[r,slot])&(tape.event_phase[r,pos]==ledger.phase[r,slot])&j.all(tape.event_generation[r,pos]==ledger.generation[r,slot],axis=1)
    if stage==2:
        good &= (ledger.end[r,slot]==at+1+p.composite.astype(j.int32))&(ledger.claimed[r,pos]==slot+1)&j.all(ledger.words[r,pos]==raw,axis=1)&(ledger.event_turn[r,pos]==tape.event_turn[r,pos])&(ledger.event_phase[r,pos]==tape.event_phase[r,pos])&j.all(ledger.event_generation[r,pos]==tape.event_generation[r,pos],axis=1)
        following=j.clip(at+1,0,tape.words.shape[1]-1);switch=tape.words[r,following]
        good &= (at+1<tape.count)&(switch[:,0]==life.SWITCH_INSTALLED)&(switch[:,1]==p.side)&(switch[:,2]==p.physical)&(switch[:,3]==p.side)&(switch[:,4]==p.incoming)&(switch[:,6]==p.composite.astype(j.int32))
    return good

def owned(tape,index,*,stage=2):
    if tape.execution is None or tape.execution.natural_cure is None:return j.zeros_like(tape.count,bool)
    good=j.zeros_like(tape.count,bool)
    for receipt in tape.execution.natural_cure:good |= _matches(tape,receipt,index,stage=stage)
    return good

def composite_at(tape,index):
    result=j.zeros_like(tape.count,bool)
    if tape.execution is not None and tape.execution.natural_cure is not None:
        for receipt in tape.execution.natural_cure:result |= receipt.pending.composite&(receipt.index==index)&receipt.pending.requested
    return result

def matches_baton(tape,record,*,stage):
    from . import engine_baton_pass_callbacks as bp
    good=j.zeros_like(tape.count,bool)
    if tape.execution is None or tape.execution.natural_cure is None:return good
    expected_private=bp.copied_private(record.before,record.side,record.outgoing,record.incoming)
    for receipt in tape.execution.natural_cure:
        p=receipt.pending
        if p.before is None:continue
        private=bp.Private(*(getattr(p.before,n)for n in bp.PRIVATE_FIELDS))
        r=j.arange(len(record.side));side=j.clip(record.side,0,1);old=j.clip(record.outgoing,0,5);new=j.clip(record.incoming,0,5)
        copied_boosts=record.boost_before.at[r,side,new].set(record.boost_before[r,side,old])
        same=bp.equal(private,expected_private)&j.all(p.before.boosts==copied_boosts,axis=(1,2,3))
        good |= p.composite&(p.ticket.slot==record.slot)&(p.ticket.serial==record.serial)&(p.ticket.frame==record.frame)&(p.side==record.side)&(p.physical==record.outgoing)&(p.incoming==record.incoming)&same&_matches(tape,receipt,record.start+1,stage=stage)
    return good

def commit(active,captures):
    if captures is None:return None
    return tuple(r._replace(pending=r.pending._replace(requested=r.pending.requested&active,valid=r.pending.valid&active),emitted=r.emitted&active)for r in captures)
