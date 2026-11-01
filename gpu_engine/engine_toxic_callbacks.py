"""Literal actual Toxic attempt guards, not reconstruction of old run causes.

Kind26: actor side/physical, target side/physical, canonical move, literal guard
bits, BEFORE/AFTER status, packed literal BEFORE/AFTER counter (four bits each).
A simple actual accuracy miss or plain type immunity emits its lexical line.
A successful primary write closes the exact existing kind9+kind26 pair without
duplicating the status line. Explicit original rules additionally bind only the
exact no-write Toxic/Substitute failure and original target erasure. Reflected,
ability and all other Substitute/blocked outcomes remain UNKNOWN, as do
unrepresented later effects.
No damage/probability/status/counter is calculated from final outcomes.
"""
from dataclasses import dataclass
import hashlib
import json
import subprocess
from pathlib import Path
from typing import NamedTuple
import weakref
import jax
import jax.numpy as jnp
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life,mechanics as m

TOXIC_ATTEMPT=26
UNKNOWN_TOXIC=256
ACCURACY_HIT,FINAL_HIT,TYPE_IMMUNE,ABILITY_IMMUNE,SUB_BLOCK,LIVE,APPLIED,PROTECTED,ABSORBED,DAMAGE_IMMUNE,SECONDARY=(1<<i for i in range(11))
MISS,PLAIN_TYPE_IMMUNITY,STATUS_WRITE,SUBSTITUTE_BLOCK,ABILITY_BLOCK,EXISTING_STATUS,OTHER=range(1,8)
_COMPILED=weakref.WeakKeyDictionary()

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    substitute_failure:object=None # explicit original rules only, default closed
    existing_status:object=None # same-source no-write branch, default closed
class Context(NamedTuple):
    data:Data
class RegisteredContext(NamedTuple):
    data:Data
    accounting:object
class Attempt(NamedTuple):
    accuracy_hit:object
    hit:object
    type_immune:object
    ability_immune:object
    substitute_block:object
    live:object
    applied:object
    protected:object
    absorbed:object
    damage_immune:object
    secondary_active:object
    old_status:object
    new_status:object
    old_counter:object
    new_counter:object
@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

def _shape(a,shape,dtype):
    if getattr(a,'shape',None)!=shape or getattr(a,'dtype',None)!=jnp.dtype(dtype):raise ValueError('Toxic exact shape/dtype required')
def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):raise ValueError('checked Toxic boundary cannot be traced')
def compile_toxic(*,engine_catalog,metadata,source_rules=None):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    from .engine_private_team import engine_metadata_binding
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('original Toxic metadata required')
    ids=[i for i,n in enumerate(metadata['vocabulary']['moves']) if normalize_id(n)=='toxic']
    table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.effects is None:raise ValueError('canonical Toxic row required')
    move=ids[0];status=np.asarray(table.effects.status)
    if (status.shape!=(len(table.power),) or status[move]!=m.TOX or np.count_nonzero(status==m.TOX)!=1
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1
            or int(table.power[move])!=0 or float(table.accuracy[move])!=85 or bool(table.always_hit[move])):
        raise ValueError('effective Gen3 Toxic85/status/target mapping required')
    descriptor=None
    if source_rules is not None:
        if bool(table.effects.bypass_sub[move]) or not bool(table.effects.protectable[move]):raise ValueError('original Toxic Substitute/protection flags required')
        from . import engine_request_source as request
        origin=request._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original Toxic rules required')
        script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('toxic'),s=d.moves.get('substitute').condition,tryHit=require(path.join(root,'dist/data/mods/gen3/scripts.js')).Scripts.actions.tryMoveHit;
const {Pokemon}=require(path.join(root,'dist/sim/pokemon.js'));
const h=f=>crypto.createHash('sha256').update(f.toString()).digest('hex');
process.stdout.write(JSON.stringify({trySetStatus:h(Pokemon.prototype.trySetStatus),setStatus:h(Pokemon.prototype.setStatus),sub:h(s.onTryPrimaryHit),getDamage:h(require(path.join(root,'dist/sim/battle-actions.js')).BattleActions.prototype.getDamage),priority:s.onTryPrimaryHitPriority,tryMoveHit:h(tryHit),move:{id:m.id,category:m.category,target:m.target,status:m.status,accuracy:m.accuracy,bypass:!!m.flags.bypasssub},source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
        descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
        if descriptor['trySetStatus']!='11c0204459e16765256cfcb741932358f443878d4cd11aac4847e34ec527e850' or descriptor['setStatus']!='1078d45febffb2ca663eca5c04d18d154304aaa467b57bf22535d90715172ef0':raise ValueError('original existing-status early failure source changed')
        if descriptor['sub']!='050e1c1fbdb80ea3ab7bc63909eb72433cb45e9c49ff7889e16ec2c207528a24' or descriptor['getDamage']!='dfd031900da8c77e5e1a2721858d8d4a7804bc7def1b7a1d486220521a6a7f12' or descriptor['priority']!=-1 or descriptor['tryMoveHit']!='af529997d52b6d856301bc8f958ead5963f68abc58d9568c82df24f212b0a97f' or descriptor['move']!=dict(id='toxic',category='Status',target='normal',status='tox',accuracy=85,bypass=False):raise ValueError('original Gen3 Toxic/Substitute source changed')
        if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Toxic source tree changed')
    raw=dict(schema='pokezero.toxic_actual_attempt.v1',substitute_failure=descriptor,move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding)
    binding=np.frombuffer(hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)
    data=Data(jnp.asarray(binding),jnp.asarray(engine_catalog.engine_binding,jnp.uint32),jnp.asarray(engine_catalog.source_binding,jnp.uint32),
        jnp.asarray(engine_catalog.move_binding,jnp.uint32),jnp.asarray(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),jnp.array(move,jnp.int32),jnp.array(descriptor is not None,jnp.bool_),jnp.array(descriptor is not None,jnp.bool_))
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog,source_rules);return result

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original Toxic compiler required')
    if r[3] is not None:
        from . import engine_request_source as request
        request._rules_origin(r[3])
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32));return compiled.device

def _check(tape,data):
    if type(tape) is not life.LifecycleTape or type(data) is not Data:raise ValueError('typed Toxic tape/Data required')
    for x in data[:5]:_shape(x,(8,),jnp.uint32)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):_shape(getattr(tape,n),(8,),jnp.uint32)
    _shape(data.move_id,(),jnp.int32);_shape(data.substitute_failure,(),jnp.bool_);_shape(data.existing_status,(),jnp.bool_)
    return (data.move_id>0)&jnp.all(data.engine_binding==tape.engine_binding)&jnp.all(data.source_binding==tape.source_binding)&jnp.all(data.move_binding==tape.move_binding)&jnp.all(data.producer_binding==tape.producer_binding)

def words_device(side,physical,target,move,attempt):
    if type(attempt) is not Attempt:raise ValueError('literal Toxic Attempt required')
    b=len(side)
    for value in (side,physical,target,move):_shape(value,(b,),jnp.int32)
    for value in attempt[:11]:_shape(value,(b,),jnp.bool_)
    for value in attempt[11:]:_shape(value,(b,),jnp.int32)
    flags=jnp.zeros(b,jnp.int32)
    for index,value in enumerate(attempt[:11]):flags|=value.astype(jnp.int32)<<index
    return jnp.stack((jnp.full_like(side,TOXIC_ATTEMPT),side,physical,1-side,target,move,flags,
        attempt.old_status,attempt.new_status,attempt.old_counter|(attempt.new_counter<<4)),axis=1)

def literal_valid(raw):
    flags=raw[:,6];old=raw[:,7];new=raw[:,8];before=raw[:,9]&15;after=raw[:,9]>>4
    applied=(flags&APPLIED)!=0;hit=(flags&FINAL_HIT)!=0;acc=(flags&ACCURACY_HIT)!=0
    secondary=(flags&SECONDARY)!=0
    return ((raw[:,0]==TOXIC_ATTEMPT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)
        &(flags>=0)&(flags<2048)&(old>=0)&(old<=6)&(new>=0)&(new<=6)&(raw[:,9]>=0)&(raw[:,9]<256)
        &(~hit|acc)&(secondary==(hit&((flags&DAMAGE_IMMUNE)==0)))
        &jnp.where(applied,(old==m.NONE)&(new==m.TOX)&(after==0)&hit,
            (old==new)&(before==after)))

def outcome_device(raw):
    f=raw[:,6];present=lambda bit:(f&bit)!=0
    miss=~present(ACCURACY_HIT)&~present(FINAL_HIT)&~present(APPLIED)&~present(ABSORBED)&present(LIVE)
    type_only=present(ACCURACY_HIT)&present(FINAL_HIT)&present(TYPE_IMMUNE)&~present(ABILITY_IMMUNE)
    type_only &= ~present(SUB_BLOCK)&~present(PROTECTED)&~present(ABSORBED)&~present(APPLIED)&present(LIVE)
    type_only &= (raw[:,7]==0)&(raw[:,8]==0)&(raw[:,9]==0)
    return jnp.where(miss,MISS,jnp.where(type_only,PLAIN_TYPE_IMMUNITY,jnp.where(present(APPLIED),STATUS_WRITE,
        jnp.where(present(SUB_BLOCK),SUBSTITUTE_BLOCK,jnp.where(present(ABILITY_IMMUNE),ABILITY_BLOCK,
            jnp.where(raw[:,7]!=0,EXISTING_STATUS,OTHER))))))

def supported_raw(raw):
    code=outcome_device(raw);f=raw[:,6]
    success=(code==STATUS_WRITE)&((f&(ACCURACY_HIT|FINAL_HIT|LIVE|APPLIED|SECONDARY))
        ==(ACCURACY_HIT|FINAL_HIT|LIVE|APPLIED|SECONDARY))
    success &= (f&(TYPE_IMMUNE|ABILITY_IMMUNE|SUB_BLOCK|PROTECTED|ABSORBED|DAMAGE_IMMUNE))==0
    success &= (raw[:,7]==m.NONE)&(raw[:,8]==m.TOX)&(raw[:,9]==0)
    return literal_valid(raw)&((code==MISS)|(code==PLAIN_TYPE_IMMUNITY)|success)

def substitute_failure_raw(raw):
    """Exact observed no-write branch, not other blocked-primary outcomes."""
    flags=ACCURACY_HIT|FINAL_HIT|SUB_BLOCK|LIVE|SECONDARY
    return literal_valid(raw)&(raw[:,6]==flags)&(raw[:,7]==m.NONE)&(raw[:,8]==m.NONE)&(raw[:,9]==0)

def substitute_failure_source(tape,raw):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count)
    if c is None or c.risks is None:return jnp.zeros(b,bool)
    t=c.risks.tables;_shape(t.toxic_binding,(8,),jnp.uint32);_shape(t.toxic_substitute_failure,(),jnp.bool_)
    good=t.toxic_substitute_failure&jnp.any(t.toxic_binding!=0)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):
        _shape(getattr(tape,n),(8,),jnp.uint32);_shape(getattr(t,n),(8,),jnp.uint32);good &= jnp.all(getattr(tape,n)==getattr(t,n))
    move=raw[:,5];safe=jnp.clip(move,0,len(t.toxic_candidates)-1)
    return good&substitute_failure_raw(raw)&(move>0)&(move<len(t.toxic_candidates))&t.toxic_candidates[safe]

def owned_substitute_failure(tape,index):
    """Actual claimed Site12 span + canonical caller, independently rechecked."""
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=jnp.arange(b);idx=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,));at=jnp.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None:return jnp.zeros(b,bool)
    ledger=c.ledger;claim=ledger.claimed[rows,at];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1);prior=jnp.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prior]
    good=substitute_failure_source(tape,raw)&ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)
    good &= (idx>0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.TOXIC_ATTEMPT))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==idx)&(ledger.end[rows,slot]==idx+1)&(ledger.no_output_kind[rows,slot]==0)
    good &= jnp.all(ledger.words[rows,at]==raw,axis=1)&jnp.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (decl[:,0]==public.MOVE)&jnp.all(decl[:,1:6]==raw[:,1:6],axis=1)&jnp.all(decl[:,6:]==0,axis=1)
    for pos in (prior,at):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&jnp.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good

def record_device(tape,side,physical,target,move,attempt,context,*,eligible,caller_move=None):
    registered=type(context) is RegisteredContext
    if type(context) not in (Context,RegisteredContext):raise ValueError('typed Toxic context required')
    b=len(tape.count);_shape(eligible,(b,),jnp.bool_);raw=words_device(side,physical,target,move,attempt)
    rows=jnp.arange(b);safe=jnp.clip(side,0,1)
    valid=literal_valid(raw)&(attempt.old_counter>=0)&(attempt.old_counter<=15)&(attempt.new_counter>=0)&(attempt.new_counter<=15)
    valid &= _check(tape,context.data)&life._valid(tape.cursor)
    valid &= (physical==tape.cursor.physical_active[rows,safe])&(target==tape.cursor.physical_active[rows,1-safe])&(move==context.data.move_id)&(tape.cursor.phase==0)
    existing=jnp.zeros(b,bool)
    if caller_move is not None:
        _shape(caller_move,(b,),jnp.int32)
        existing=context.data.existing_status&existing_status_source(tape,raw)&(caller_move==move)
    good=(supported_raw(raw)|(context.data.substitute_failure&substitute_failure_source(tape,raw))|existing)&valid
    if registered:
        from . import engine_execution_completeness as execution
        actual=execution._context(tape)
        if actual is None:good &= False
        else:
            good &= execution._validate(actual.ledger,context.accounting)&execution._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.refusal==0)
            for name in ('engine_binding','source_binding','producer_binding','inventory_binding'):good &= jnp.all(getattr(actual.tables,name)==getattr(context.accounting,name))
    room=tape.count<tape.words.shape[1]
    tape=public._append(tape,TOXIC_ATTEMPT,side,physical,1-side,target,tuple(raw[:,i] for i in range(5,10)),eligible&valid&(~existing_status_raw(raw)|existing))
    return tape._replace(refusal=tape.refusal|jnp.where(eligible&(~good|~room|~jnp.asarray(registered)),UNKNOWN_TOXIC,0))


def existing_status_raw(raw):
    """Only literal successful TryHit with existing unchanged status/counter."""
    return literal_valid(raw)&(raw[:,6]==(ACCURACY_HIT|FINAL_HIT|LIVE|SECONDARY))&(raw[:,7]>0)&(raw[:,7]<=6)&(raw[:,7]==raw[:,8])


def existing_status_source(tape,raw):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count)
    if c is None or c.risks is None:return jnp.zeros(b,bool)
    t=c.risks.tables;_shape(t.toxic_binding,(8,),jnp.uint32);_shape(t.toxic_existing_status,(),jnp.bool_)
    good=t.toxic_existing_status&jnp.any(t.toxic_binding!=0)
    for n in ('engine_binding','source_binding','move_binding','producer_binding'):
        _shape(getattr(tape,n),(8,),jnp.uint32);_shape(getattr(t,n),(8,),jnp.uint32);good &= jnp.all(getattr(tape,n)==getattr(t,n))
    move=raw[:,5];return good&existing_status_raw(raw)&(move>0)&(move<len(t.toxic_candidates))&t.toxic_candidates[jnp.clip(move,0,len(t.toxic_candidates)-1)]


def owned_existing_status(tape,index):
    """Independent exact claimed Site12 and original current caller prefix."""
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=jnp.arange(b);idx=jnp.broadcast_to(jnp.asarray(index,jnp.int32),(b,));at=jnp.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None:return jnp.zeros(b,bool)
    ledger=c.ledger;claim=ledger.claimed[rows,at];slot=jnp.clip(claim-1,0,ledger.site.shape[1]-1);prior=jnp.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prior]
    good=existing_status_source(tape,raw)&ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)
    good &= (idx>0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.TOXIC_ATTEMPT))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==idx)&(ledger.end[rows,slot]==idx+1)&(ledger.no_output_kind[rows,slot]==0)
    good &= jnp.all(ledger.words[rows,at]==raw,axis=1)&jnp.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (decl[:,0]==public.MOVE)&jnp.all(decl[:,1:6]==raw[:,1:6],axis=1)&jnp.all(decl[:,6:]==0,axis=1)
    for pos in(prior,at):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&jnp.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&jnp.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good
