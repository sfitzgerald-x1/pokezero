"""Prospective SAME-call positive Double-Edge target/recoil source pair.

A pending Site14 is entered after the captured target write and before recoil
risk observation. It is NOT emission/consumption credit: only the later literal
nonlethal actor HP write can close its span. The separate verified RockHead
family captures a nonempty private receipt and emits no public words. Other
missing/blocked/actor-KO or interleaved callbacks remain unknown. No amount or
damage is reconstructed.
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
    rock_head_source:object
    immune_source:object=None

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
    caller_move:object=None
    no_damage:object=None
    no_damage_outcome:object=None
    substitute_damage:object=None # same actual RAW38 target write, never inferred HP

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
    ko_source=False;source=None;rock_head_source=False;rock_head_descriptor=None
    if source_rules is not None:
        from . import engine_request_source as request
        origin=request._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original recoil rules required')
        script=r"""const path=require('path'),c=require('crypto'),d=require(path.join(process.argv[1],'dist/sim/dex.js')).Dex.mod('gen3'),{Battle}=require(path.join(process.argv[1],'dist/sim/battle.js')),h=x=>c.createHash('sha256').update(x.toString()).digest('hex');console.log(JSON.stringify({tryMoveHit:h(d.data.Scripts.actions.tryMoveHit),calcRecoilDamage:h(d.data.Scripts.actions.calcRecoilDamage),spreadDamage:h(Battle.prototype.spreadDamage),faintMessages:h(Battle.prototype.faintMessages),runAction:h(Battle.prototype.runAction),recoil:d.moves.get('doubleedge').recoil}));"""
        source=json.loads(subprocess.run([origin.node,'-e',script,origin.root],capture_output=True,text=True,check=True,timeout=30).stdout)
        if source!=KO_SOURCE:raise ValueError('original target KO/recoil/faint source changed')
        ko_source=True
        rock_script=r"""const path=require('path'),c=require('crypto'),d=require(path.join(process.argv[1],'dist/sim/dex.js')).Dex.mod('gen3'),a=d.abilities.get('rockhead');console.log(JSON.stringify({id:a.id,exists:a.exists,gen:a.gen,callbacks:Object.keys(a).filter(k=>/^on/.test(k)&&a[k]!==undefined),onDamage:c.createHash('sha256').update(a.onDamage.toString()).digest('hex')}));"""
        rock_head_descriptor=json.loads(subprocess.run([origin.node,'-e',rock_script,origin.root],capture_output=True,text=True,check=True,timeout=30).stdout)
        if rock_head_descriptor!={'id':'rockhead','exists':True,'gen':3,'callbacks':['onDamage'],'onDamage':'93f86584c39c6ffe0a2a36792751f9e88f304762eb8e48befecd1ee630b9ef1d'}:raise ValueError('canonical original silent Rock Head recoil source required')
        rock_head_source=bool(np.count_nonzero(rock)==1)
    raw=dict(schema='pokezero.literal_doubleedge_pair.v3',immune_source=source is not None,ko_source=source,rock_head=rock_head_descriptor,move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding,rock_head_abilities=rock.tolist())
    words=lambda x:jnp.asarray(x,jnp.uint32)
    data=Data(words(np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32),jnp.array(rock),jnp.array(ko_source,jnp.bool_),jnp.array(rock_head_source,jnp.bool_),jnp.array(source is not None,jnp.bool_))
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
    execution._shape(data.immune_source,(),jnp.bool_)
    execution._shape(data.ko_source,(),jnp.bool_)
    execution._shape(data.rock_head_source,(),jnp.bool_)
    execution._shape(data.rock_head,(len(data.rock_head),),jnp.bool_)
    if len(data.rock_head)<1:raise ValueError('nonempty actual ability namespace required')
    original=execution._context(tape)
    if original is None:return jnp.zeros_like(tape.count,bool)
    for tables in (context.direct.accounting,original.tables):
        if type(tables) is not execution.Tables:raise ValueError('original accounting tables required')
        for value in tables:execution._shape(value,(8,),jnp.uint32)
    good=execution._validate(original.ledger,context.direct.accounting)&execution._tape(original.ledger,tape,context.direct.accounting)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):
        execution._shape(getattr(tape,n),(8,),jnp.uint32)
        good &= jnp.all(getattr(data,n)==getattr(tape,n))&jnp.all(getattr(context.direct.data,n)==getattr(data,n))
    good &= jnp.all(original.tables.inventory_binding==context.direct.accounting.inventory_binding)
    return good

def matches_substitute_pending(tape,pending,captured=None):
    """Only the source-bound Double-Edge Sub write may precede positive recoil.

    No damage/recoil amount is reconstructed. The actual Outcome and later
    actor HP write must separately match the original pending recoil amount.
    """
    from . import engine_substitute_damage_callbacks as sub
    from . import engine_execution_completeness as ex
    b=len(tape.count);no=jnp.zeros(b,bool)
    s=pending.substitute_damage
    if s is None:return no
    if type(s)is not sub.Captured:raise ValueError('same-call captured Substitute write required')
    p=s.pending;d=p.context.data.substitute_damage;w=s.write;c=pending.captured
    if d is None or d.double_edge_move is None:return no
    if pending.caller_move is None or p.caller_move is None:return no
    good=_bound(tape,pending.context)&sub.span_matches(tape,p,w)
    good &= pending.context.data.ko_source&(d.double_edge_move>0)&(p.move==d.double_edge_move)
    good &= (p.move==pending.context.data.move_id)&(p.caller_move==p.move)&(pending.caller_move==p.move)
    good &= (c.actor==p.actor)&(c.actor_physical==p.physical)&(c.target_physical==p.target)&(c.move==p.move)
    for name in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
        good &= jnp.all(getattr(p.context.data,name)==getattr(c.context.data,name))
    for name in ('power','critical','hit','immune','numerator','denominator','protected','enduring','old_hp','new_hp'):
        good &= getattr(c.write,name)==getattr(w,name)
    good &= c.write.substitute&(w.power==120)&(pending.recoil>0)&(pending.old_hp>pending.recoil)
    good &= ~pending.context.data.rock_head[jnp.clip(pending.ability,0,len(pending.context.data.rock_head)-1)]
    if captured is not None:
        if type(captured)is not sub.Captured:raise ValueError('literal observed Substitute capture required')
        for name in w._fields:good &= getattr(w,name)==getattr(captured.write,name)
        for name in ('actor','physical','target','move','requested','caller_move'):
            value=getattr(captured.pending,name)
            if value is None:return no
            good &= getattr(p,name)==value
        ledger=ex._context(tape).ledger
        slot,same=ex._ticket(ledger,p.ticket);other,matched=ex._ticket(ledger,captured.pending.ticket)
        good &= same&matched&(slot==other)
        for name in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
            good &= jnp.all(getattr(p.context.data,name)==getattr(captured.pending.context.data,name))
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
    sub_mode=jnp.zeros(b,bool)
    if pending.substitute_damage is not None:
        sp=pending.substitute_damage.pending
        sub_mode=sp.requested
        subslot,subsame=execution._ticket(ledger,sp.ticket)
        dslot=jnp.where(sub_mode,subslot,dslot);dsame=jnp.where(sub_mode,subsame,dsame)
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
        &(ledger.site[rows,dslot]==jnp.where(sub_mode,int(execution.Site.SUBSTITUTE_DAMAGE),int(execution.Site.DIRECT_HIT)))
        &(actor>=0)&(actor<2)&(ap==tape.cursor.physical_active[rows,safe])
        &(target==tape.cursor.physical_active[rows,1-safe])
        &(ability>=0)&(ability<n)&(pending.old_hp>0)&(pending.maximum>=pending.old_hp)&(pending.maximum<=1_000_000)
        &(pending.status>=0)&(pending.status<=6))
    ordinary=~sub_mode&(ledger.stage[rows,dslot]==2)&(ledger.end[rows,dslot]==tape.count)&execution._direct_hit_span_matches(tape,capture.ticket,actor,ap,target,move,capture.write)
    composite=matches_ko_pending(tape,pending)
    substitute=matches_substitute_pending(tape,pending)
    positive=~data.rock_head[jnp.clip(ability,0,n-1)]&(pending.recoil>0)&(pending.old_hp>pending.recoil)
    silent=silent_pending(pending)
    good &= (positive&(ordinary|composite|substitute))|(_rock_head_pending(pending)&ordinary)|matches_immune_pending(tape,pending)
    start=ledger.start[rows,dslot];span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<tape.count[:,None])
    good &= composite|jnp.all(~span|((ledger.claimed==dslot[:,None]+1)&jnp.all(ledger.words==tape.words,axis=2)
        &(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)
        &jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    return good

def prepare_device(tape,captured,recoil,ability,hp,maximum,status,context,*,eligible,target_ko=None,caller_move=None,no_damage=None,no_damage_outcome=None,substitute_damage=None):
    from . import engine_execution_completeness as execution
    b=tape.count.shape[0]
    for x in (recoil,ability,hp,maximum,status):_vector(x,b)
    _vector(eligible,b,jnp.bool_)
    requested=eligible&(captured.move==context.data.move_id)
    tape,ticket=execution.open_attempt(tape,execution.Site.RECOIL_HP,captured.actor,active=requested)
    pending=Pending(context,ticket,captured,recoil,ability,hp,maximum,status,requested,target_ko,caller_move,no_damage,no_damage_outcome,substitute_damage)
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
    silent=silent_pending(pending)
    good=matches_before(tape,pending,write)&(write.new_hp>0)&(write.old_hp-write.new_hp==pending.recoil)&jnp.where(silent,write.old_hp==write.new_hp,write.old_hp>write.new_hp)
    c=pending.captured;b=tape.count.shape[0];rows=jnp.arange(b)
    inside=(tape.count>=0)&(tape.count<tape.words.shape[1]);index=jnp.clip(tape.count,0,tape.words.shape[1]-1)
    good &= ~inside|jnp.all(tape.switch_details[rows,index]==0,axis=1)
    tape=public._append(tape,RECOIL_HP,c.actor,c.actor_physical,1-c.actor,c.target_physical,
        (c.move,write.old_hp,write.new_hp,write.maximum,write.status),pending.requested&good)
    original=execution._context(tape)
    ledger=execution.unhandled(original.ledger,pending.ticket,active=pending.requested&~good,tables=original.tables)
    ledger=execution.emitted(ledger,pending.ticket,tape,active=pending.requested&good,tables=original.tables)
    # A nonempty PRIVATE RAW28 is consumed, with no public output. The typed
    # ability/source receipt is separate from the generic empty-span protocol.
    slot,_=execution._ticket(ledger,pending.ticket);use=pending.requested&good&silent
    caller=pending.caller_move if pending.caller_move is not None else jnp.full_like(write.ability,-1)
    source_fact=jnp.where(_immune_literal(pending),caller,write.ability)
    facts=jnp.stack((source_fact,write.old_hp,write.maximum,write.status,c.write.old_hp,c.write.new_hp),axis=1)
    ledger=ledger._replace(no_output_facts=ledger.no_output_facts.at[rows,slot].set(jnp.where(use[:,None],facts,ledger.no_output_facts[rows,slot])),
        no_output_binding=ledger.no_output_binding.at[rows,slot].set(jnp.where(use[:,None],pending.context.data.binding[None,:],ledger.no_output_binding[rows,slot])))
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


def _rock_head_pending(pending):
    """Only current canonical RockHead +positive nonlethal SAME target write."""
    b=len(pending.requested);no=jnp.zeros(b,bool);data=pending.context.data
    from . import engine_execution_completeness as ex
    ex._shape(data.rock_head_source,(),jnp.bool_)
    if pending.caller_move is None:return no
    _vector(pending.caller_move,b)
    w=pending.captured.write;ability=pending.ability
    return (data.rock_head_source&(pending.caller_move==pending.captured.move)
        &(pending.captured.move==data.move_id)&(data.move_id>0)&(pending.recoil==0)
        &(ability>=0)&(ability<len(data.rock_head))&data.rock_head[jnp.clip(ability,0,len(data.rock_head)-1)]
        &(w.power==120)&w.hit&~w.immune&~w.substitute&~w.protected&~w.enduring
        &(w.old_hp>w.new_hp)&(w.new_hp>0)&(w.maximum>=w.old_hp))


def _owned_rock_head(tape,index,cursor=None):
    """Independent typed source receipt for RAW28 with exactly zero public words."""
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=jnp.arange(b);at=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,));safe=jnp.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,safe]
    if c is None or c.risks is None:return jnp.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger
    ex._shape(t.recoil_rock_head_source,(),jnp.bool_);ex._shape(t.recoil_rock_head,(len(t.recoil_rock_head),),jnp.bool_)
    ex._shape(t.recoil_binding,(8,),jnp.uint32);ex._shape(t.recoil_candidates,(len(t.selected_risk),),jnp.bool_)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&(ledger.refusal==0)&((tape.refusal&~public.UNINSTRUMENTED)==0)&t.recoil_rock_head_source&jnp.any(t.recoil_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        ex._shape(getattr(tape,name),(8,),jnp.uint32);ex._shape(getattr(t,name),(8,),jnp.uint32);good &= jnp.all(getattr(tape,name)==getattr(t,name))
    claim=ledger.claimed[rows,safe];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1);facts=ledger.no_output_facts[rows,slot];ability=facts[:,0];move=raw[:,5]
    good &= (raw[:,0]==RECOIL_HP)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]==1-raw[:,1])&(raw[:,4]>=0)&(raw[:,4]<6)
    good &= (raw[:,6]==raw[:,7])&(raw[:,6]>0)&(raw[:,8]>=raw[:,6])&(raw[:,8]<=1_000_000)&(raw[:,9]>=0)&(raw[:,9]<=6)&(at>0)&(at<tape.count)
    good &= (move>0)&(move<len(t.recoil_candidates))&t.recoil_candidates[jnp.clip(move,0,len(t.recoil_candidates)-1)]
    good &= (ability>=0)&(ability<len(t.recoil_rock_head))&t.recoil_rock_head[jnp.clip(ability,0,len(t.recoil_rock_head)-1)]&jnp.all(facts[:,1:4]==jnp.stack((raw[:,6],raw[:,8],raw[:,9]),axis=1),axis=1)&(facts[:,4]>facts[:,5])&(facts[:,5]>0)
    good &= (claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.RECOIL_HP))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==at)&(ledger.end[rows,slot]==at+1)&(ledger.no_output_kind[rows,slot]==0)&jnp.all(ledger.no_output_binding[rows,slot]==t.recoil_binding[None,:],axis=1)
    # The prior direct target transaction ends exactly at this source receipt.
    hp_at=jnp.clip(at-1,0,tape.words.shape[1]-1);hp=tape.words[rows,hp_at];direct_claim=ledger.claimed[rows,hp_at];ds=jnp.clip(direct_claim-1,0,ledger.site.shape[1]-1);start=ledger.start[rows,ds];decl_at=jnp.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[rows,decl_at]
    good &= (direct_claim>0)&(direct_claim<=ledger.count)&(ledger.site[rows,ds]==int(ex.Site.DIRECT_HIT))&(ledger.side[rows,ds]==raw[:,3])&(ledger.stage[rows,ds]==2)&(ledger.end[rows,ds]==at)&(ledger.no_output_kind[rows,ds]==0)&(start>0)
    good &= (hp[:,0]==public.MAIN_HP)&jnp.all(hp[:,1:3]==-1,axis=1)&jnp.all(hp[:,3:5]==raw[:,3:5],axis=1)&(hp[:,5]==facts[:,5])&(hp[:,6]>=facts[:,4])&(hp[:,7]>=0)&(hp[:,7]<=6)&jnp.all(hp[:,8:]==0,axis=1)
    good &= (decl[:,0]==public.MOVE)&jnp.all(decl[:,1:6]==raw[:,1:6],axis=1)&jnp.all(decl[:,6:]==0,axis=1)&jnp.all(tape.switch_details[rows,safe]==0,axis=1)
    span=(jnp.arange(tape.words.shape[1])[None,:]>=start[:,None])&(jnp.arange(tape.words.shape[1])[None,:]<at[:,None])
    good &= jnp.all(~span|((ledger.claimed==direct_claim[:,None])&jnp.all(ledger.words==tape.words,axis=2)&(ledger.event_turn==tape.event_turn)&(ledger.event_phase==tape.event_phase)&jnp.all(ledger.event_generation==tape.event_generation,axis=2)),axis=1)
    good &= (ledger.turn[rows,ds]==ledger.turn[rows,slot])&(ledger.phase[rows,ds]==ledger.phase[rows,slot])&jnp.all(ledger.generation[rows,ds]==ledger.generation[rows,slot],axis=1)
    for pos in (decl_at,hp_at):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    if cursor is not None:
        good &= (raw[:,2]==cursor.physical_active[rows,jnp.clip(raw[:,1],0,1)])&(raw[:,4]==cursor.physical_active[rows,jnp.clip(raw[:,3],0,1)])&(cursor.turn==ledger.turn[rows,slot])&(cursor.phase==ledger.phase[rows,slot])&jnp.all(cursor.generation==ledger.generation[rows,slot],axis=1)
    good &= jnp.all(ledger.words[rows,safe]==raw,axis=1)&(ledger.turn[rows,slot]==tape.event_turn[rows,safe])&(ledger.phase[rows,slot]==tape.event_phase[rows,safe])&jnp.all(ledger.generation[rows,slot]==tape.event_generation[rows,safe],axis=1)
    good &= (ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])&jnp.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    return good


def _immune_literal(p):
    """Literal ordinary type-immunity/zero-recoil envelope; no HP formula."""
    from . import engine_no_damage_callbacks as nd,engine_kernel_frame_risk as risks
    b=len(p.requested);no=jnp.zeros(b,bool)
    if type(p.no_damage)is not nd.Captured or type(p.no_damage_outcome)is not risks.Outcome or p.caller_move is None:return no
    _vector(p.caller_move,b);d=p.context.data;w=p.captured.write;c=p.no_damage;o=p.no_damage_outcome
    same=jnp.ones(b,bool)
    for x,y in zip(c.write,w,strict=True):same &= x==y
    for x,y in ((c.pending.actor,p.captured.actor),(c.pending.physical,p.captured.actor_physical),(c.pending.target,p.captured.target_physical),(c.pending.move,p.captured.move),(c.actor_hp,p.old_hp),(c.pending.before.actor_hp,p.old_hp)):
        same &= x==y
    return (same&d.immune_source&(p.caller_move==p.captured.move)&(p.captured.move==d.move_id)&(d.move_id>0)
        &(p.recoil==0)&(o.recoil==0)&(w.power==120)&w.hit&w.immune&(w.numerator==0)&((w.denominator==1)|(w.denominator==2))
        &~w.substitute&~w.protected&~w.enduring&~o.absorbed&~o.explodes&(w.old_hp==w.new_hp)&(w.new_hp>0))


def matches_immune_pending(tape,p):
    from . import engine_no_damage_callbacks as nd,engine_execution_completeness as ex
    good=_immune_literal(p);c=ex._context(tape)
    if c is None or c.risks is None or p.no_damage is None or p.no_damage_outcome is None:return good&False
    t=c.risks.tables;ex._shape(t.recoil_immune_source,(),jnp.bool_)
    good &= t.recoil_immune_source&jnp.all(p.context.data.binding==t.recoil_binding)
    good &= nd.captured_matches(tape,p.no_damage,p.no_damage_outcome)
    for name in ('stream_id','frame','slot','serial'):good &= getattr(p.no_damage.pending.ticket,name)==getattr(p.captured.ticket,name)
    good &= jnp.all(p.no_damage.pending.ticket.inventory_binding==p.captured.ticket.inventory_binding)
    for name in ('binding','engine_binding','source_binding','move_binding','producer_binding'):
        good &= jnp.all(getattr(p.no_damage.pending.context.data,name)==getattr(p.context.direct.data,name))
    return good


def silent_pending(pending):
    return _rock_head_pending(pending)|_immune_literal(pending)


def _owned_immune(tape,index,cursor=None):
    """Two separately claimed nonempty sites; original immune line only."""
    from . import engine_execution_completeness as ex,engine_no_damage_callbacks as nd
    c=ex._context(tape);b=len(tape.count);rows=jnp.arange(b);at=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,));safe=jnp.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,safe]
    if c is None or c.risks is None or type(c.secondary_status)is not direct.RegisteredContext:return jnp.zeros(b,bool)
    t=c.risks.tables;d=c.secondary_status.data.no_damage;nd._shape(d)
    ex._shape(t.recoil_immune_source,(),jnp.bool_);ex._shape(t.recoil_binding,(8,),jnp.uint32)
    good=direct._registered_bound(tape,c.secondary_status)&t.recoil_immune_source&d.recoil_immune_source&jnp.any(t.recoil_binding!=0)
    good &= jnp.all(c.secondary_status.data.binding==t.direct_hit_binding)&jnp.all(d.binding==t.no_damage_binding)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        ex._shape(getattr(tape,name),(8,),jnp.uint32);ex._shape(getattr(t,name),(8,),jnp.uint32);good &= jnp.all(getattr(tape,name)==getattr(t,name))
    pair,header,_=nd.owned_pair(tape,at-2,d);good &= pair&(header[:,6]==nd.TYPE_IMMUNE)&(header[:,5]==d.recoil_immune_move)&(d.recoil_immune_move>0)
    ledger=c.ledger;claim=ledger.claimed[rows,safe];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1);facts=ledger.no_output_facts[rows,slot]
    direct_at=jnp.clip(at-2,0,tape.words.shape[1]-1);dc=ledger.claimed[rows,direct_at];ds=jnp.clip(dc-1,0,ledger.site.shape[1]-1)
    good &= ledger.opened&(ledger.refusal==0)&((tape.refusal&~public.UNINSTRUMENTED)==0)&(at>=3)&(at<tape.count)
    good &= (raw[:,0]==RECOIL_HP)&jnp.all(raw[:,1:6]==header[:,1:6],axis=1)&(raw[:,6]==raw[:,7])&(raw[:,6]>0)&(raw[:,8]>=raw[:,6])&(raw[:,8]<=1_000_000)&(raw[:,9]>=0)&(raw[:,9]<=6)
    # Fact0 is the actual ordinary caller for this subtype, never an ability.
    expected=jnp.stack((raw[:,5],raw[:,6],raw[:,8],raw[:,9],header[:,7],header[:,8]),axis=1)
    good &= jnp.all(facts==expected,axis=1)&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.RECOIL_HP))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==at)&(ledger.end[rows,slot]==at+1)&(ledger.no_output_kind[rows,slot]==0)
    good &= jnp.all(ledger.no_output_binding[rows,slot]==t.recoil_binding[None,:],axis=1)&jnp.all(ledger.words[rows,safe]==raw,axis=1)&jnp.all(tape.switch_details[rows,safe]==0,axis=1)
    good &= (dc>0)&(dc<=ledger.count)&(dc!=claim)&(ledger.side[rows,ds]==raw[:,3])&(ledger.end[rows,ds]==at)
    good &= (ledger.turn[rows,ds]==ledger.turn[rows,slot])&(ledger.phase[rows,ds]==ledger.phase[rows,slot])&jnp.all(ledger.generation[rows,ds]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])&jnp.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    good &= (ledger.turn[rows,slot]==tape.event_turn[rows,safe])&(ledger.phase[rows,slot]==tape.event_phase[rows,safe])&jnp.all(ledger.generation[rows,slot]==tape.event_generation[rows,safe],axis=1)
    if cursor is not None:
        good &= (raw[:,2]==cursor.physical_active[rows,jnp.clip(raw[:,1],0,1)])&(raw[:,4]==cursor.physical_active[rows,jnp.clip(raw[:,3],0,1)])&(cursor.turn==ledger.turn[rows,slot])&(cursor.phase==ledger.phase[rows,slot])&jnp.all(cursor.generation==ledger.generation[rows,slot],axis=1)
    return good


def owned_silent(tape,index,cursor=None):
    return _owned_rock_head(tape,index,cursor)|_owned_immune(tape,index,cursor)
