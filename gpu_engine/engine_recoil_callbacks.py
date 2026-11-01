"""Prospective SAME-call positive Double-Edge target/recoil source pair.

A pending Site14 is entered after the captured target write and before recoil
risk observation. It is NOT emission/consumption credit: only the later literal
nonlethal actor HP write can close its span. Missing/blocked/Rock Head/actor-KO or
interleaved callbacks stay unknown. No recoil amount or damage is reconstructed.
Device values are trusted owning-call inputs, not copied-carry attestation.
"""
from dataclasses import dataclass
import subprocess
import hashlib,json,weakref
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life
from . import engine_direct_hit_controls as direct
from .engine_private_team import engine_metadata_binding

RECOIL_HP=28
UNKNOWN_RECOIL=256
_COMPILED=weakref.WeakKeyDictionary()

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    rock_head:object
    ko_source:object

@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

class Context(NamedTuple):
    data:Data
    direct:direct.RegisteredContext

class Pending(NamedTuple):
    context:Context
    ticket:object
    captured:object
    recoil:object
    ability:object
    old_hp:object
    maximum:object
    status:object
    requested:object
    target_ko:object=None

class Before(NamedTuple):
    actor_physical:object
    target_physical:object
    ability:object
    old_hp:object
    maximum:object
    status:object

class Write(NamedTuple):
    actor_physical:object
    target_physical:object
    ability:object
    old_hp:object
    new_hp:object
    maximum:object
    status:object


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked recoil compiler must remain outside outer JIT')

def compile_recoil(*,engine_catalog,metadata,source_rules=None):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('same original recoil metadata required')
    names=metadata['vocabulary']['moves'];ids=[i for i,n in enumerate(names) if normalize_id(n)=='doubleedge'];table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.damage_moves is None:raise ValueError('canonical Double-Edge source row required')
    move=ids[0];rock=np.array([normalize_id(n)=='rockhead' for n in metadata['vocabulary']['abilities']],bool)
    if (int(table.power[move])!=120 or int(table.damage_moves.recoil_den[move])!=3
        or bool(table.damage_moves.struggle[move]) or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1
        or np.asarray(table.damage_moves.rock_head).shape!=rock.shape or not np.array_equal(table.damage_moves.rock_head,rock)):
        raise ValueError('original Double-Edge120/recoil3/target/RockHead columns required')
    ko_source=False;source=None
    if source_rules is not None:
        from . import engine_request_source as request
        origin=request._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original recoil rules required')
        script=r"""const path=require('path'),c=require('crypto'),d=require(path.join(process.argv[1],'dist/sim/dex.js')).Dex.mod('gen3'),{Battle}=require(path.join(process.argv[1],'dist/sim/battle.js')),h=x=>c.createHash('sha256').update(x.toString()).digest('hex');console.log(JSON.stringify({tryMoveHit:h(d.data.Scripts.actions.tryMoveHit),calcRecoilDamage:h(d.data.Scripts.actions.calcRecoilDamage),spreadDamage:h(Battle.prototype.spreadDamage),faintMessages:h(Battle.prototype.faintMessages),runAction:h(Battle.prototype.runAction),recoil:d.moves.get('doubleedge').recoil}));"""
        source=json.loads(subprocess.run([origin.node,'-e',script,origin.root],capture_output=True,text=True,check=True,timeout=30).stdout)
        if source!=KO_SOURCE:raise ValueError('original target KO/recoil/faint source changed')
        ko_source=True
    raw=dict(schema='pokezero.literal_doubleedge_pair.v2',ko_source=source,move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding,rock_head=rock.tolist())
    words=lambda x:jnp.asarray(x,jnp.uint32)
    data=Data(words(np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32),jnp.array(rock),jnp.array(ko_source,jnp.bool_))
    out=Compiled(data);_COMPILED[out]=(data,tuple(data),engine_catalog,source_rules);return out

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original immutable recoil compiler required')
    if r[3] is not None:
        from . import engine_request_source as request
        request._rules_origin(r[3])
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,expected_engine_binding=compiled.device.engine_binding)
    return compiled.device

def _vector(x,b,dtype=jnp.int32):
    from . import engine_execution_completeness as execution
    execution._shape(x,(b,),dtype);return x

def _bound(tape,context):
    from . import engine_execution_completeness as execution
    if type(context) is not Context or type(context.data) is not Data or type(context.direct) is not direct.RegisteredContext:raise ValueError('original typed recoil/direct context required')
    data=context.data
    direct._check(tape,context.direct.data)
    for a in data[:5]:execution._shape(a,(8,),jnp.uint32)
    execution._shape(data.move_id,(),jnp.int32)
    execution._shape(data.ko_source,(),jnp.bool_)
    execution._shape(data.rock_head,(len(data.rock_head),),jnp.bool_)
    if len(data.rock_head)<1:raise ValueError('nonempty actual ability namespace required')
    original=execution._context(tape)
    if original is None:return jnp.zeros_like(tape.count,bool)
    for tables in (context.direct.accounting,original.tables):
        if type(tables) is not execution.Tables:raise ValueError('original accounting tables required')
        for value in tables:execution._shape(value,(8,),jnp.uint32)
    good=execution._validate(original.ledger,context.direct.accounting)&execution._tape(original.ledger,tape,context.direct.accounting)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= jnp.all(getattr(data,n)==getattr(tape,n))&jnp.all(getattr(context.direct.data,n)==getattr(data,n))
    good &= jnp.all(original.tables.inventory_binding==context.direct.accounting.inventory_binding)
    return good

def matches_pending(tape,pending):
    """Read-only prospective obligation. No flag/risk is ever cleared."""
    from . import engine_execution_completeness as execution
    from . import engine_kernel_frame_risk as risks
    if type(pending) is not Pending or type(pending.captured) is not risks.CapturedDirectHit:raise ValueError('typed SAME-call pending recoil required')
    b=tape.count.shape[0];rows=jnp.arange(b);context=pending.context;data=context.data;capture=pending.captured
    for value in pending[3:8]:_vector(value,b)
    _vector(pending.requested,b,jnp.bool_)
    bound=_bound(tape,context);original=execution._context(tape)
    if original is None:return jnp.zeros(b,bool)
    ledger=original.ledger;slot,same=execution._ticket(ledger,pending.ticket)
    dslot,dsame=execution._ticket(ledger,capture.ticket)
    if type(capture.context) is not direct.RegisteredContext or type(capture.write) is not direct.HitWrite:raise ValueError('original captured direct write required')
    direct._check(tape,capture.context.data)
    for a,z in zip(capture.context.accounting,context.direct.accounting):
        execution._shape(a,(8,),jnp.uint32);same &= jnp.all(a==z)
    for n in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
        same &= jnp.all(getattr(capture.context.data,n)==getattr(context.direct.data,n))
    actor=_vector(capture.actor,b);ap=_vector(capture.actor_physical,b);target=_vector(capture.target_physical,b)
    move=_vector(capture.move,b);safe=jnp.clip(actor,0,1)
    ability=pending.ability;n=len(data.rock_head)
    good=(bound&same&dsame&pending.requested&ledger.opened&(ledger.refusal==0)
        &((tape.refusal&~public.UNINSTRUMENTED)==0)&(move==data.move_id)&(data.move_id>0)
        &(ledger.site[rows,slot]==int(execution.Site.RECOIL_HP))&(ledger.stage[rows,slot]==1)
        &(ledger.start[rows,slot]==tape.count)&(ledger.side[rows,slot]==actor)
        &(ledger.site[rows,dslot]==int(execution.Site.DIRECT_HIT))
        &(actor>=0)&(actor<2)&(ap==tape.cursor.physical_active[rows,safe])
        &(target==tape.cursor.physical_active[rows,1-safe])
        &(ability>=0)&(ability<n)&~data.rock_head[jnp.clip(ability,0,n-1)]
        &(pending.recoil>0)&(pending.old_hp>pending.recoil)&(pending.maximum>=pending.old_hp)&(pending.maximum<=1_000_000)
        &(pending.status>=0)&(pending.status<=6))
    ordinary=(ledger.stage[rows,dslot]==2)&(ledger.end[rows,dslot]==tape.count)&execution._direct_hit_span_matches(tape,capture.ticket,actor,ap,target,move,capture.write)
    composite=matches_ko_pending(tape,pending)
    good &= ordinary|composite
    start=ledger.start[rows,dslot];span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    good &= composite|jnp.all(~span|((ledger.claimed==dslot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return good

def prepare_device(tape,captured,recoil,ability,hp,maximum,status,context,*,eligible,target_ko=None):
    from . import engine_execution_completeness as execution
    b=tape.count.shape[0]
    for x in (recoil,ability,hp,maximum,status):_vector(x,b)
    _vector(eligible,b,jnp.bool_)
    requested=eligible&(captured.move==context.data.move_id)
    tape,ticket=execution.open_attempt(tape,execution.Site.RECOIL_HP,captured.actor,active=requested)
    pending=Pending(context,ticket,captured,recoil,ability,hp,maximum,status,requested,target_ko)
    good=matches_pending(tape,pending)
    original=execution._context(tape)
    if original is not None:
        ledger=execution.unhandled(original.ledger,ticket,active=requested&~good,tables=original.tables)
        tape=execution._put_context(tape,ledger,original)
    return tape._replace(refusal=tape.refusal|jnp.where(requested&~good,UNKNOWN_RECOIL,0)),pending

def matches_before(tape,pending,write):
    if type(write) not in (Before,Write):raise ValueError('literal recoil BEFORE/AFTER write required')
    b=tape.count.shape[0]
    for x in write:_vector(x,b)
    c=pending.captured
    return (matches_pending(tape,pending)&(write.actor_physical==c.actor_physical)&(write.target_physical==c.target_physical)
        &(write.ability==pending.ability)&(write.old_hp==pending.old_hp)&(write.maximum==pending.maximum)&(write.status==pending.status))

def record_device(tape,pending,write):
    """After SAME actor assignment: capture actual words, never an HP formula."""
    from . import engine_execution_completeness as execution
    good=matches_before(tape,pending,write)&(write.old_hp>write.new_hp)&(write.new_hp>0)&(write.old_hp-write.new_hp==pending.recoil)
    c=pending.captured;b=tape.count.shape[0];rows=jnp.arange(b)
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);index=jnp.clip(tape.count,0,tape.words.shape[1]-1)
    good &= ~inside|jnp.all(tape.switch_details[rows,index]==0,axis=1)
    tape=public._append(tape,RECOIL_HP,c.actor,c.actor_physical,1-c.actor,c.target_physical,
        (c.move,write.old_hp,write.new_hp,write.maximum,write.status),pending.requested&good)
    original=execution._context(tape)
    ledger=execution.unhandled(original.ledger,pending.ticket,active=pending.requested&~good,tables=original.tables)
    ledger=execution.emitted(ledger,pending.ticket,tape,active=pending.requested&good,tables=original.tables)
    tape=execution._put_context(tape,ledger,original)
    return tape._replace(refusal=tape.refusal|jnp.where(pending.requested&(~good|~inside),UNKNOWN_RECOIL,0))

# Separate source capability; legacy compiler calls never enable target-KO.
KO_SOURCE=dict(tryMoveHit='af529997d52b6d856301bc8f958ead5963f68abc58d9568c82df24f212b0a97f',calcRecoilDamage='5272ac5de33ad15ed00cb18b2e119d25294de30bab6383a9bd92517ed82a6238',spreadDamage='aeefec5d8ad3851ec46a5d1bf73bfffcd74924aa36eda154873de9129a29a7bb',faintMessages='62f5ba2782de2f0716c5071a3555ac238a430105b1e2e04e8b96b9cb6c9e5c4b',runAction='6c36c083886cd9f924699a934ba43e3aee101c4c435fdb00a811b476bc9f23df',recoil=[1,3])

class KOBefore(NamedTuple):
    context:Context
    caller:object
    ability:object
    maximum:object
    status:object
    amount:object


def ko_selected(p):
    before=p.recoil_before
    if before is None:return jnp.zeros_like(p.requested)
    if type(before)is not KOBefore:raise ValueError('same-write recoil KO before required')
    from . import engine_execution_completeness as ex
    ex._shape(before.context.data.ko_source,(),jnp.bool_)
    return before.context.data.ko_source&(p.move==before.context.data.move_id)&(p.write.new_hp==0)


def ko_literal(p,b):
    before=p.recoil_before
    if before is None:return jnp.zeros(b,bool)
    for x in before[1:]:_vector(x,b)
    data=before.context.data;_vector(p.target_party_before[:,0],b)
    rows=jnp.arange(b);other=jnp.arange(6)[None,:]!=p.target_physical[:,None]
    return (ko_selected(p)&(before.caller==p.move)&(p.write.power==120)
        &(before.ability>=0)&(before.ability<len(data.rock_head))&~data.rock_head[jnp.clip(before.ability,0,len(data.rock_head)-1)]
        &(p.actor_hp_before>0)&(p.actor_hp_before<=before.maximum)&(before.maximum<=1_000_000)&(before.status>=0)&(before.status<=6)
        &jnp.any(other&(p.target_party_before>0),axis=1))


def with_ko_amount(p,amount):
    if p.recoil_before is None:return p
    _vector(amount,len(p.requested))
    return p._replace(recoil_before=p.recoil_before._replace(amount=amount))


def matches_ko_pending(tape,pending):
    from . import engine_direct_ko_callbacks as ko
    p=pending.target_ko
    if p is None or p.recoil_before is None:return jnp.zeros(len(tape.count),bool)
    before=p.recoil_before;c=pending.captured
    good=ko._span(tape,p,with_faint=False)&ko_literal(p,len(tape.count))&_bound(tape,before.context)
    good &= (pending.recoil==before.amount)&(pending.recoil>0)&(pending.old_hp==p.actor_hp_before)&(pending.old_hp>pending.recoil)
    for a,b in((pending.maximum,before.maximum),(pending.status,before.status),(pending.ability,before.ability),(c.actor,p.actor),(c.actor_physical,p.actor_physical),(c.target_physical,p.target_physical),(c.move,p.move)):
        good &= a==b
    for a,b in zip(c.write,p.write):good &= a==b
    for name in('binding','engine_binding','source_binding','move_binding','producer_binding'):
        good &= jnp.all(getattr(c.context.data,name)==getattr(p.context.data,name))
    return good


def ko_recoil_word(tape,p,index):
    """Literal uniquely claimed Site14 word inside this exact Site10 span."""
    from . import engine_execution_completeness as ex
    rows=jnp.arange(len(tape.count));at=jnp.clip(index,0,tape.words.shape[1]-1);raw=tape.words[rows,at];ledger=tape.execution.ledger;claim=ledger.claimed[rows,at];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1);before=p.recoil_before
    expected=jnp.stack((jnp.full_like(p.actor,RECOIL_HP),p.actor,p.actor_physical,1-p.actor,p.target_physical,p.move,p.actor_hp_before,p.actor_hp_before-before.amount,before.maximum,before.status),axis=1)
    good=ko_literal(p,len(rows))&_bound(tape,before.context)&(before.amount>0)&(p.actor_hp_before>before.amount)&jnp.all(raw==expected,axis=1)
    good &= (claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.RECOIL_HP))&(ledger.side[rows,slot]==p.actor)&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==index)&(ledger.end[rows,slot]==index+1)&(ledger.no_output_kind[rows,slot]==0)
    good &= (ledger.turn[rows,slot]==tape.event_turn[rows,at])&(ledger.phase[rows,slot]==tape.event_phase[rows,at])&jnp.all(ledger.generation[rows,slot]==tape.event_generation[rows,at],axis=1)
    good &= jnp.all(ledger.words[rows,at]==raw,axis=1)&(ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&jnp.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good


def finish_ko_on_tape(tape,p,actor_hp,target_party,actor_physical,target_physical,write):
    """Only this exact composite partitions one span; generic emitted unchanged."""
    from . import engine_direct_ko_callbacks as ko,engine_execution_completeness as ex
    b=len(tape.count);rows=jnp.arange(b);selected=ko_selected(p);before=p.recoil_before
    if before is None:return tape
    good=ko._span(tape,p,with_faint=True)&_bound(tape,before.context)&ko_literal(p,b)
    if type(write)is not Write:raise ValueError('literal same-call actor final recoil write required')
    for x in write:_vector(x,b)
    expected_party=p.target_party_before.at[rows,jnp.clip(p.target_physical,0,5)].set(0)
    good &= (actor_hp==p.actor_hp_before-before.amount)&(actor_hp>0)&jnp.all(target_party==expected_party,axis=1)&(actor_physical==p.actor_physical)&(target_physical==p.target_physical)
    for a,z in((write.actor_physical,p.actor_physical),(write.target_physical,p.target_physical),(write.ability,before.ability),(write.old_hp,p.actor_hp_before),(write.new_hp,actor_hp),(write.maximum,before.maximum),(write.status,before.status)):good &= a==z
    c=ex._context(tape);ledger=c.ledger;slot,same=ex._ticket(ledger,p.ticket);good &= same&(ledger.refusal==0)&((tape.refusal&~public.UNINSTRUMENTED)==0)
    start=ledger.start[rows,slot];recoil_at=tape.count-2;span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None]);own=span&(jnp.arange(tape.words.shape[1])[None,:]!=recoil_at[:,None])
    good &= jnp.all(~own|(ledger.claimed==0),axis=1)
    use=p.requested&selected&good;mask=use[:,None]&own
    ledger=ex.unhandled(ledger,p.ticket,active=p.requested&selected&~good,tables=c.tables)
    ledger=ledger._replace(stage=ledger.stage.at[rows,slot].set(jnp.where(use,2,ledger.stage[rows,slot])),end=ledger.end.at[rows,slot].set(jnp.where(use,tape.count,ledger.end[rows,slot])),claimed=jnp.where(mask,slot[:,None]+1,ledger.claimed),words=jnp.where(mask[:,:,None],tape.words,ledger.words),event_turn=jnp.where(mask,tape.event_turn,ledger.event_turn),event_phase=jnp.where(mask,tape.event_phase,ledger.event_phase),event_generation=jnp.where(mask[:,:,None],tape.event_generation,ledger.event_generation),upstream_refusal=ledger.upstream_refusal|jnp.where(use,tape.refusal,0))
    tape=ex._put_context(tape,ledger,c)
    return tape._replace(refusal=tape.refusal|jnp.where(p.requested&selected&~good,UNKNOWN_RECOIL,0))


def validate_owned_ko_control(tape,index,raw):
    """Independent exact final two-ticket partition, no pending-object waiver."""
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=jnp.arange(b)
    if c is None or c.risks is None:return jnp.zeros(b,bool)
    t=c.risks.tables;ex._shape(t.recoil_ko,(),jnp.bool_)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        ex._shape(getattr(t,name),(8,),jnp.uint32);ex._shape(getattr(tape,name),(8,),jnp.uint32)
    ex._shape(t.recoil_binding,(8,),jnp.uint32);ex._shape(t.recoil_candidates,(len(t.selected_risk),),jnp.bool_)
    ledger=c.ledger;claim=ledger.claimed[:,index];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1)
    start=ledger.start[rows,slot];end=ledger.end[rows,slot];faint_at=end-1;recoil_at=end-2;hp_at=end-3
    get=lambda at:tape.words[rows,jnp.clip(at,0,tape.words.shape[1]-1)]
    hp=get(hp_at);r=get(recoil_at);f=get(faint_at);decl=get(start-1);rc=ledger.claimed[rows,jnp.clip(recoil_at,0,tape.words.shape[1]-1)];rs=jnp.clip(rc-1,0,ledger.site.shape[1]-1)
    syntax=((raw[:,0]==direct.CRITICAL)|((raw[:,0]==direct.SUPER_EFFECTIVE)&(raw[:,8]>raw[:,9]))|((raw[:,0]==direct.RESISTED)&(raw[:,8]<raw[:,9])))&(raw[:,6]>0)&(raw[:,7]==0)&direct._positive_ratio(raw[:,8],raw[:,9])
    critical=get(start)[:,0]==direct.CRITICAL;effect=(raw[:,8]!=raw[:,9]);controls=critical.astype(jnp.int32)+effect.astype(jnp.int32)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&(ledger.refusal==0)&((tape.refusal&~public.UNINSTRUMENTED)==0)&t.recoil_ko&jnp.any(t.recoil_binding!=0)&syntax&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.DIRECT_HIT))&(ledger.stage[rows,slot]==2)&(ledger.no_output_kind[rows,slot]==0)&(start>0)&(hp_at==start+controls)&(end<=tape.count)&(index>=start)&(index<hp_at)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= jnp.all(getattr(t,name)==getattr(tape,name))
    move=raw[:,5];good &= (move>0)&(move<len(t.recoil_candidates))&t.recoil_candidates[jnp.clip(move,0,len(t.recoil_candidates)-1)]
    good &= (decl[:,0]==public.MOVE)&jnp.all(decl[:,1:6]==raw[:,1:6],axis=1)&jnp.all(decl[:,6:]==0,axis=1)&(hp[:,0]==public.MAIN_HP)&jnp.all(hp[:,1:3]==-1,axis=1)&(hp[:,3]==raw[:,3])&(hp[:,4]==raw[:,4])&(hp[:,5]==0)&(hp[:,6]>=raw[:,6])&(hp[:,7]>=0)&(hp[:,7]<=6)&jnp.all(hp[:,8:]==0,axis=1)
    good &= (r[:,0]==RECOIL_HP)&jnp.all(r[:,1:6]==raw[:,1:6],axis=1)&(r[:,6]>r[:,7])&(r[:,7]>0)&(r[:,8]>=r[:,6])&(r[:,8]<=1_000_000)&(r[:,9]>=0)&(r[:,9]<=6)
    good &= (f[:,0]==public.DIRECT_FAINT)&jnp.all(f[:,1:3]==-1,axis=1)&jnp.all(f[:,3:5]==raw[:,3:5],axis=1)&jnp.all(f[:,5:]==0,axis=1)&(ledger.side[rows,slot]==raw[:,3])
    # Every optional annotation is the exact SAME literal HitWrite envelope.
    for offset,flag,kind in ((jnp.zeros(b,jnp.int32),critical,jnp.full(b,direct.CRITICAL,jnp.int32)),(critical.astype(jnp.int32),effect,jnp.where(raw[:,8]>raw[:,9],direct.SUPER_EFFECTIVE,direct.RESISTED))):
        control=get(start+offset)
        good &= ~flag|((control[:,0]==kind)&jnp.all(control[:,1:]==raw[:,1:],axis=1))
    good &= jnp.where(raw[:,0]==direct.CRITICAL,index==start,index==start+critical.astype(jnp.int32))
    good &= (tape.event_turn[rows,jnp.clip(start-1,0,tape.words.shape[1]-1)]==ledger.turn[rows,slot])&(tape.event_phase[rows,jnp.clip(start-1,0,tape.words.shape[1]-1)]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,jnp.clip(start-1,0,tape.words.shape[1]-1)]==ledger.generation[rows,slot],axis=1)
    good &= (rc>0)&(rc<=ledger.count)&(ledger.site[rows,rs]==int(ex.Site.RECOIL_HP))&(ledger.side[rows,rs]==raw[:,1])&(ledger.stage[rows,rs]==2)&(ledger.start[rows,rs]==recoil_at)&(ledger.end[rows,rs]==recoil_at+1)&(ledger.no_output_kind[rows,rs]==0)
    good &= (ledger.turn[rows,rs]==ledger.turn[rows,slot])&(ledger.phase[rows,rs]==ledger.phase[rows,slot])&jnp.all(ledger.generation[rows,rs]==ledger.generation[rows,slot],axis=1)
    indices=jnp.arange(tape.words.shape[1])[None,:];span=(indices>=start[:,None])&(indices<end[:,None]);expected_claim=jnp.where(indices==recoil_at[:,None],rc[:,None],claim[:,None])
    good &= jnp.all(~span|((ledger.claimed==expected_claim)&jnp.all(ledger.words==tape.words,axis=2)&(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)&jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    good &= jnp.all(~span|((tape.event_turn==ledger.turn[rows,slot,None])&(tape.event_phase==ledger.phase[rows,slot,None])&jnp.all(tape.event_generation==ledger.generation[rows,slot,None,:],axis=2)),axis=1)
    return good
