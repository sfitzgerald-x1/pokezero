"""Actual canonical Thunder Wave NONE->PAR input; existing RAW9+26 only.

The status write is made by the original numerical kernel. This observer emits
its literal guard continuation, then requires the same current MOVE/status
span and both original consumers. Explicit SAME source rules additionally bind
the actual no-write Substitute failure and prospective same-call natural Ground
immunity. Other non-successes stay UNKNOWN.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import NamedTuple
import weakref
import jax
import jax.numpy as j
import numpy as np
from pokezero.dex import normalize_id
from . import engine_public_events as public,engine_lifecycle_callbacks as life,mechanics as m
from . import engine_toxic_callbacks as primary,engine_status_callbacks as status

_COMPILED=weakref.WeakKeyDictionary()
Attempt=primary.Attempt  # Existing RAW26 literal guard schema, not Toxic admission.
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    substitute_failure:object=None # SAME original source rules; legacy closed
    ground_immunity:object=None # SAME source-only private current-fact capability
class GroundData(NamedTuple):
    electric:object
    ground:object
    abilities:object
    items:object
class Private(NamedTuple):
    hp:object
    max_hp:object
    status:object
    status_counter:object
    ability:object
    base_ability:object
    item:object
    types:object
    base_types:object
    moves:object
    pp:object
    substitute_hp:object
    volatile:object
    confusion_counter:object
    protect_counter:object
    choice_move:object
    encore_move:object
    encore_counter:object
    charge_move:object
    charge_counter:object
class GroundPending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    target:object
    move:object
    caller:object
    slot:object
    requested:object
    before:Private
    resolved_type:object
    power:object
    accuracy:object
    hit:object
    immune:object
    numerator:object
    denominator:object
    protected:object
    absorbed:object
class GroundCaptured(NamedTuple):
    pending:GroundPending
    after:Private
    attempt:object
class Context(NamedTuple):
    data:Data
    accounting:object
@dataclass(frozen=True,eq=False)
class Compiled:
    device:Data

def _host():
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('checked Thunder Wave compiler cannot be traced')
def compile_thunder_wave(*,engine_catalog,metadata,source_rules=None):
    _host();public.check_catalog(engine_catalog,engine_catalog.engine_table,expected_binding=engine_catalog.binding,
        expected_engine_binding=np.asarray(engine_catalog.engine_binding,np.uint32))
    from .engine_private_team import engine_metadata_binding
    if not np.array_equal(engine_metadata_binding(metadata),engine_catalog.engine_binding):raise ValueError('original Thunder Wave metadata required')
    ids=[i for i,name in enumerate(metadata['vocabulary']['moves']) if normalize_id(name)=='thunderwave'];table=engine_catalog.engine_table
    if len(ids)!=1 or ids[0]<=0 or table.effects is None:raise ValueError('canonical Thunder Wave required')
    move=ids[0]
    if (int(table.power[move])!=0 or float(table.accuracy[move])!=100 or bool(table.always_hit[move])
            or int(table.effects.status[move])!=m.PAR or not bool(table.effects.thunderwave[move])
            or engine_catalog.ids[move]<=0 or engine_catalog.targets[move]!=1):
        raise ValueError('canonical Gen3 Thunder Wave100/primary PAR/target required')
    descriptor=None;ground=None
    if source_rules is not None:
        from . import engine_request_source as request
        origin=request._rules_origin(source_rules)
        if not np.array_equal(source_rules.device.engine_binding,engine_catalog.engine_binding):raise ValueError('same original Thunder Wave rules required')
        if bool(table.effects.bypass_sub[move]) or not bool(table.effects.protectable[move]):raise ValueError('original Thunder Wave Substitute/protection flags required')
        script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('thunderwave'),s=d.moves.get('substitute').condition;
const h=f=>crypto.createHash('sha256').update(f.toString()).digest('hex');
const hooks=['onTryHit','onTryPrimaryHit','onTryMove','onPrepareHit','onModifyMove','onAnyModifyMove','onAllyModifyMove','onNegateImmunity','onAnyTryHit','onAllyTryHit','onFoeTryHit','onSourceTryHit','onDeductPP'];
const quiet=e=>e.exists&&e.gen<=3&&!hooks.some(k=>e[k]!==undefined);
process.stdout.write(JSON.stringify({sub:h(s.onTryPrimaryHit),getDamage:h(require(path.join(root,'dist/sim/battle-actions.js')).BattleActions.prototype.getDamage),runImmunity:h(require(path.join(root,'dist/sim/pokemon.js')).Pokemon.prototype.runImmunity),ground:{type:m.type,ignoreImmunity:m.ignoreImmunity,abilities:meta.vocabulary.abilities.map(n=>quiet(d.abilities.get(n))),items:meta.vocabulary.items.map((n,i)=>!i||quiet(d.items.get(n)))},priority:s.onTryPrimaryHitPriority,tryMoveHit:h(require(path.join(root,'dist/data/mods/gen3/scripts.js')).Scripts.actions.tryMoveHit),move:{id:m.id,category:m.category,target:m.target,status:m.status,accuracy:m.accuracy,bypass:!!m.flags.bypasssub},source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
        descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
        if (descriptor['sub']!='050e1c1fbdb80ea3ab7bc63909eb72433cb45e9c49ff7889e16ec2c207528a24'
                or descriptor['getDamage']!='dfd031900da8c77e5e1a2721858d8d4a7804bc7def1b7a1d486220521a6a7f12'
                or descriptor['priority']!=-1 or descriptor['tryMoveHit']!='af529997d52b6d856301bc8f958ead5963f68abc58d9568c82df24f212b0a97f'
                or descriptor['move']!=dict(id='thunderwave',category='Status',target='normal',status='par',accuracy=100,bypass=False)):
            raise ValueError('original Gen3 Thunder Wave/Substitute source changed')
        if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Thunder Wave source tree changed')
        g=descriptor['ground'];electric=metadata['vocabulary']['types'].index('electric');ground_type=metadata['vocabulary']['types'].index('ground')
        if (descriptor['runImmunity']!='4c7cbdb0b36ed635276037455e9ee78593e79268c4c97c70a1a892d6651c198d'
                or g['type']!='Electric' or g['ignoreImmunity']is not False or int(table.type[move])!=electric
                or int(table.effects.ground_type)!=ground_type or int(table.type_num[electric,ground_type])!=0):raise ValueError('canonical Thunder Wave natural Ground immunity required')
        ground=GroundData(j.array(electric,j.int32),j.array(ground_type,j.int32),j.array(g['abilities'],j.bool_),j.array(g['items'],j.bool_))
    payload=dict(schema='pokezero.thunder-wave-actual-par.v1',substitute_failure=descriptor,move=move,engine=engine_catalog.engine_binding,
        source=engine_catalog.source_binding,moves=engine_catalog.move_binding,producer=engine_catalog.binding)
    words=lambda x:j.asarray(x,j.uint32)
    data=Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),
        words(engine_catalog.engine_binding),words(engine_catalog.source_binding),words(engine_catalog.move_binding),
        words(np.frombuffer(bytes.fromhex(engine_catalog.binding),dtype='>u4').astype(np.uint32)),j.array(move,j.int32),j.array(descriptor is not None,j.bool_),ground)
    result=Compiled(data);_COMPILED[result]=(data,tuple(data),engine_catalog,source_rules);return result

def check_compiled(compiled):
    _host();r=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if r is None or compiled.device is not r[0] or any(a is not b for a,b in zip(compiled.device,r[1])):raise ValueError('original Thunder Wave compiler required')
    if r[3] is not None:
        from . import engine_request_source as request
        request._rules_origin(r[3])
    catalog=r[2];public.check_catalog(catalog,catalog.engine_table,expected_binding=catalog.binding,
        expected_engine_binding=np.asarray(catalog.engine_binding,np.uint32));return compiled.device

def supported_raw(raw):
    # Successful primary application ONLY. Do not generalize the original
    # Toxic validator, or admit its miss/immunity branches under this source.
    flags=primary.ACCURACY_HIT|primary.FINAL_HIT|primary.LIVE|primary.APPLIED|primary.SECONDARY
    return ((raw[:,0]==primary.TOXIC_ATTEMPT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])
        &(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)
        &(raw[:,6]==flags)&(raw[:,7]==m.NONE)&(raw[:,8]==m.PAR)&(raw[:,9]==0))


def ground_raw(raw):
    flags=primary.ACCURACY_HIT|primary.FINAL_HIT|primary.LIVE|primary.DAMAGE_IMMUNE
    return primary.literal_valid(raw)&(raw[:,6]==flags)&j.all(raw[:,7:]==0,axis=1)


def _private(state):
    b=len(state.active);rows=j.arange(b)[:,None];sides=j.arange(2)[None,:]
    life._typed(state.active,(b,2),j.int32)
    values=[]
    for name in Private._fields:
        x=getattr(state,name);tail=(2,)if name in ('types','base_types')else(4,)if name in ('moves','pp')else()
        life._typed(x,(b,2,6)+tail,j.uint32 if name=='volatile'else j.int32)
        values.append(x[rows,sides,j.clip(state.active,0,5)])
    return Private(*values)


def ground_shape(g):
    if type(g)is not GroundData:raise ValueError('typed original Ground immunity source required')
    for value in g[:2]:life._typed(value,(),j.int32)
    for value in g[2:]:
        if getattr(value,'ndim',None)!=1 or not len(value):raise ValueError('nonempty original Ground namespace required')
        life._typed(value,(len(value),),j.bool_)


def pack_ground(g):
    """Exact flat Risk.Tables representation; empty means no capability."""
    if g is None:return j.zeros((0,),j.int32)
    ground_shape(g)
    return j.concatenate((g.electric.reshape(1),g.ground.reshape(1),
        g.abilities.astype(j.int32),g.items.astype(j.int32)))


def _ground_source(tape,context):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);data=context.data;g=data.ground_immunity;b=len(tape.count)
    if g is None or actual is None or actual.risks is None:return j.zeros(b,bool)
    ground_shape(g)
    t=actual.risks.tables;other=t.thunder_wave_ground_immunity
    expected=pack_ground(g);life._typed(other,expected.shape,j.int32)
    good=ex._validate(actual.ledger,context.accounting)&ex._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened
    good &= j.any(data.binding!=0)&j.all(data.binding==t.thunder_wave_binding)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(data,name),(8,),j.uint32)
        good &= j.all(getattr(tape,name)==getattr(data,name))
    good &= j.all(other==expected)
    return good


def begin_ground(tape,state,ticket,side,move,caller,slot,context,*,requested,resolved_type,power,accuracy,hit,immune,numerator,denominator,protected,absorbed):
    if context.data.ground_immunity is None:return None
    return GroundPending(context,ticket,side,state.active[j.arange(len(side)),j.clip(side,0,1)],
        state.active[j.arange(len(side)),1-j.clip(side,0,1)],move,caller,slot,requested,_private(state),
        resolved_type,power,accuracy,hit,immune,numerator,denominator,protected,absorbed)


def _ground_valid(tape,p):
    b=len(tape.count);r=j.arange(b);side=j.clip(p.side,0,1);foe=1-side;g=p.context.data.ground_immunity
    if g is None:return j.zeros(b,bool)
    for name in ('side','physical','target','move','caller','slot','resolved_type','power','numerator','denominator'):life._typed(getattr(p,name),(b,),j.int32)
    for name in ('requested','accuracy','hit','immune','protected','absorbed'):life._typed(getattr(p,name),(b,),j.bool_)
    before=p.before
    for name,x in zip(Private._fields,before,strict=True):
        tail=(2,)if name in ('types','base_types')else(4,)if name in ('moves','pp')else()
        life._typed(x,(b,2)+tail,j.uint32 if name=='volatile'else j.int32)
    good=_ground_source(tape,p.context)&p.requested&(p.side>=0)&(p.side<2)&(p.physical>=0)&(p.physical<6)&(p.target>=0)&(p.target<6)
    good &= (p.move==p.context.data.move_id)&(p.caller==p.move)&(p.slot>=0)&(p.slot<4)&(before.moves[r,side,j.clip(p.slot,0,3)]==p.move)
    good &= (p.resolved_type==g.electric)&(p.power==0)&p.accuracy&p.hit&p.immune&(p.numerator==0)&(p.denominator==1)&~p.protected&~p.absorbed
    good &= j.any(before.types[r,foe]==g.ground,axis=1)&j.all((before.hp>0)&(before.hp<=before.max_hp)&(before.max_hp<=1_000_000),axis=1)
    good &= j.all((before.status==0)&(before.status_counter==0)&(before.substitute_hp==0),axis=1)&j.all(before.pp>=0,axis=(1,2))
    good &= j.all(before.ability==before.base_ability,axis=1)&j.all(before.types==before.base_types,axis=(1,2))
    for n,mask in [('ability',g.abilities),('item',g.items)]:
        value=getattr(before,n);good &= j.all((value>=0)&(value<len(mask))&mask[j.clip(value,0,len(mask)-1)],axis=1)
    for n in ('volatile','confusion_counter','protect_counter','choice_move','encore_move','encore_counter','charge_move','charge_counter'):good &= j.all(getattr(before,n)==0,axis=1)
    return good


def _ground_matches(tape,c,index,*,claimed):
    from . import engine_execution_completeness as ex
    p=c.pending;b=len(tape.count);r=j.arange(b);actual=ex._context(tape);ledger=actual.ledger
    slot,same=ex._ticket(ledger,p.ticket);start=ledger.start[r,slot];at=j.clip(j.broadcast_to(j.asarray(index,j.int32),(b,)),0,tape.words.shape[1]-1);prior=j.clip(at-1,0,tape.words.shape[1]-1)
    raw=primary.words_device(p.side,p.physical,p.target,p.move,c.attempt);decl=tape.words[r,prior]
    good=_ground_valid(tape,p)&same&ground_raw(raw)&j.all(tape.words[r,at]==raw,axis=1)&(start==index)&(start>0)
    for a,z in zip(p.before,c.after,strict=True):good &= j.all((a==z).reshape(b,-1),axis=1)
    good &= (ledger.site[r,slot]==int(ex.Site.PRIMARY_PARALYSIS))&(ledger.side[r,slot]==1-p.side)&(ledger.phase[r,slot]==0)
    good &= (p.physical==tape.cursor.physical_active[r,j.clip(p.side,0,1)])&(p.target==tape.cursor.physical_active[r,1-j.clip(p.side,0,1)])
    good &= j.all(ledger.generation[r,slot]==tape.cursor.generation,axis=1)&tape.cursor.initialized
    good &= (decl[:,0]==public.MOVE)&j.all(decl[:,1:6]==raw[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
    for pos in (prior,at):good &= (tape.event_turn[r,pos]==ledger.turn[r,slot])&(tape.event_phase[r,pos]==ledger.phase[r,slot])&j.all(tape.event_generation[r,pos]==ledger.generation[r,slot],axis=1)
    good &= j.all(tape.switch_details[r,at]==0,axis=1)
    if claimed:
        good &= (ledger.stage[r,slot]==2)&(ledger.end[r,slot]==start+1)&(at<tape.count)&(ledger.claimed[r,at]==slot+1)&(ledger.no_output_kind[r,slot]==0)
        good &= j.all(ledger.words[r,at]==raw,axis=1)&(ledger.event_turn[r,at]==tape.event_turn[r,at])&(ledger.event_phase[r,at]==tape.event_phase[r,at])&j.all(ledger.event_generation[r,at]==tape.event_generation[r,at],axis=1)
    else:good &= (ledger.stage[r,slot]==1)&(tape.count==start+1)
    return good


def owned_ground_immunity(tape,index):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);good=j.zeros(len(tape.count),bool)
    if actual is None or actual.thunder_wave_immunity is None:return good
    for capture in actual.thunder_wave_immunity:good |= _ground_matches(tape,capture,index,claimed=True)
    return good


def commit_ground(active,captures):
    """Retain actual frame receipts only on the driver's source-ready rows."""
    if captures is None:return None
    return tuple(c._replace(pending=c.pending._replace(requested=c.pending.requested&active)) for c in captures)


def substitute_failure_source(tape,raw):
    """Explicit checked source capability, never a generic no-write waiver."""
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=len(tape.count)
    if actual is None or actual.risks is None:return j.zeros(b,bool)
    tables=actual.risks.tables
    primary._shape(tables.thunder_wave_binding,(8,),j.uint32)
    primary._shape(tables.thunder_wave_substitute_failure,(),j.bool_)
    good=tables.thunder_wave_substitute_failure&j.any(tables.thunder_wave_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        primary._shape(getattr(tape,name),(8,),j.uint32);primary._shape(getattr(tables,name),(8,),j.uint32)
        good &= j.all(getattr(tape,name)==getattr(tables,name))
    move=raw[:,5];safe=j.clip(move,0,len(tables.thunder_wave_candidates)-1)
    return good&primary.substitute_failure_raw(raw)&(move>0)&(move<len(tables.thunder_wave_candidates))&tables.thunder_wave_candidates[safe]


def owned_substitute_failure(tape,index):
    """Recheck the unique current Site19 one-word transaction at consumption."""
    from . import engine_execution_completeness as ex
    actual=ex._context(tape);b=len(tape.count);rows=j.arange(b)
    if actual is None:return j.zeros(b,bool)
    idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1)
    raw=tape.words[rows,at];ledger=actual.ledger;claim=ledger.claimed[rows,at]
    slot=j.clip(claim-1,0,ledger.site.shape[1]-1);prior=j.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prior]
    good=substitute_failure_source(tape,raw)&ex._validate(ledger,actual.tables)&ex._tape(ledger,tape,actual.tables)&ledger.opened&(ledger.frame>0)&life._valid(tape.cursor)
    good &= (idx>0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&(ledger.site[rows,slot]==int(ex.Site.PRIMARY_PARALYSIS))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==2)&(ledger.start[rows,slot]==idx)&(ledger.end[rows,slot]==idx+1)&(ledger.no_output_kind[rows,slot]==0)
    good &= j.all(ledger.words[rows,at]==raw,axis=1)&j.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (decl[:,0]==public.MOVE)&j.all(decl[:,1:6]==raw[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
    actor=j.clip(raw[:,1],0,1);target=j.clip(raw[:,3],0,1)
    good &= (raw[:,2]==tape.cursor.physical_active[rows,actor])&(raw[:,4]==tape.cursor.physical_active[rows,target])
    # Consumption follows original upkeep/next-turn advancement. Its
    # exact clock is the current ledger/event snapshot, not the final cursor.
    good &= tape.cursor.initialized&(ledger.phase[rows,slot]==0)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    for pos in (prior,at):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    good &= (ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good

def _bound(tape,context):
    from . import engine_execution_completeness as ex
    if type(tape) is not life.LifecycleTape or type(context) is not Context or type(context.data) is not Data:raise ValueError('typed Thunder Wave source context required')
    for word in context.data[:5]:primary._shape(word,(8,),j.uint32)
    primary._shape(context.data.move_id,(),j.int32)
    primary._shape(context.data.substitute_failure,(),j.bool_)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        primary._shape(getattr(tape,name),(8,),j.uint32)
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    good=ex._validate(actual.ledger,context.accounting)&ex._tape(actual.ledger,tape,context.accounting)&actual.ledger.opened&(actual.ledger.refusal==0)&life._valid(tape.cursor)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        good &= j.all(getattr(context.data,name)==getattr(tape,name))
    for before,after in zip(actual.tables,context.accounting):good &= j.all(before==after)
    from . import engine_kernel_frame_risk as risk
    if type(actual.risks) is not risk.Context:return good&False
    rbound,_=risk._bound(actual.risks.carry,tape,actual.risks.tables)
    good &= rbound&j.any(context.data.binding!=0)&j.all(context.data.binding==actual.risks.tables.thunder_wave_binding)
    return (good&(actual.ledger.frame>0)&tape.cursor.initialized&(tape.cursor.phase==0)
        &(tape.turn==tape.cursor.turn)&(context.data.move_id>0))

def span_matches(tape,ticket,side,physical,target,move,attempt,context):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return j.zeros_like(tape.count,bool)
    ledger=actual.ledger;slot,same=ex._ticket(ledger,ticket);b=len(tape.count);rows=j.arange(b);safe=j.clip(side,0,1)
    raw=primary.words_device(side,physical,target,move,attempt);start=ledger.start[rows,slot]
    first=j.clip(start,0,tape.words.shape[1]-1);at=j.clip(start+1,0,tape.words.shape[1]-1);prefix=j.clip(start-1,0,tape.words.shape[1]-1)
    absent=j.full_like(side,-1);zero=j.zeros_like(side)
    status_word=j.stack((j.full_like(side,status.STATUS_APPLIED),absent,absent,1-side,target,
        attempt.old_status,attempt.new_status,j.full_like(side,status.PRIMARY),zero,zero),axis=1)
    decl=tape.words[rows,prefix]
    failed=(context.data.substitute_failure&substitute_failure_source(tape,raw))|owned_ground_immunity(tape,start)
    expected_end=start+j.where(failed,1,2)
    good=(_bound(tape,context)&same&(supported_raw(raw)|failed)&(move==context.data.move_id)
        &(attempt.old_counter==0)&(attempt.new_counter==0)&(start>0)&(tape.count==expected_end)
        &(ledger.site[rows,slot]==int(ex.Site.PRIMARY_PARALYSIS))&(ledger.side[rows,slot]==1-side)
        &(physical==tape.cursor.physical_active[rows,safe])&(target==tape.cursor.physical_active[rows,1-safe])
        &(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)
        &j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
        &j.where(failed,j.all(tape.words[rows,first]==raw,axis=1),j.all(tape.words[rows,first]==status_word,axis=1)&j.all(tape.words[rows,at]==raw,axis=1))
        &(decl[:,0]==public.MOVE)&(decl[:,1]==side)&(decl[:,2]==physical)&(decl[:,3]==1-side)&(decl[:,4]==target)&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1))
    for index,required in ((prefix,True),(first,True),(at,~failed)):
        stamp=((tape.event_turn[rows,index]==ledger.turn[rows,slot])&(tape.event_phase[rows,index]==ledger.phase[rows,slot])
            &j.all(tape.event_generation[rows,index]==ledger.generation[rows,slot],axis=1))
        good &= ~j.asarray(required)|stamp
    return good

def record_device(tape,ticket,side,physical,target,move,attempt,context,*,requested,ground_pending=None,state=None):
    from . import engine_execution_completeness as ex
    actual=ex._context(tape)
    if actual is None:return tape
    primary._shape(requested,tape.count.shape,j.bool_)
    raw=primary.words_device(side,physical,target,move,attempt)
    ground=j.zeros(len(tape.count),bool);capture=None
    if ground_pending is not None:
        if type(ground_pending)is not GroundPending or state is None:raise ValueError('actual Ground immunity before/after required')
        capture=GroundCaptured(ground_pending,_private(state),attempt)
        ground=_ground_valid(tape,ground_pending)&ground_raw(raw)
        for left,right in zip(ground_pending.ticket,ticket,strict=True):ground &= j.all(left==right)
        for left,right in zip((ground_pending.side,ground_pending.physical,ground_pending.target,ground_pending.move),(side,physical,target,move),strict=True):ground &= left==right
        if ground_pending.context is not context:ground &= False
        for a,z in zip(ground_pending.before,capture.after,strict=True):ground &= j.all((a==z).reshape(len(tape.count),-1),axis=1)
        previous=actual.thunder_wave_immunity or ()
        if len(previous)>=2:raise ValueError('only original two actor immunity records per frame')
        tape=tape._replace(execution=actual._replace(thunder_wave_immunity=previous+(capture,)))
    valid=_bound(tape,context)&(supported_raw(raw)|(context.data.substitute_failure&substitute_failure_source(tape,raw))|ground)&(move==context.data.move_id)&(attempt.old_counter==0)&(attempt.new_counter==0)
    tape=public._append(tape,primary.TOXIC_ATTEMPT,side,physical,1-side,target,tuple(raw[:,i] for i in range(5,10)),requested&valid)
    same=span_matches(tape,ticket,side,physical,target,move,attempt,context)
    if capture is not None:same |= _ground_matches(tape,capture,tape.execution.ledger.start[j.arange(len(tape.count)),j.clip(ticket.slot,0,tape.execution.ledger.site.shape[1]-1)],claimed=False)
    actual=ex._context(tape);ledger=ex.unhandled(actual.ledger,ticket,active=requested&~same,tables=actual.tables)
    ledger=ex.emitted(ledger,ticket,tape,active=requested&same,tables=actual.tables)
    return ex._put_context(tape,ledger,actual)


def exclude_cure_device(tape,requested,applied,item,item_table):
    """This narrow admission excludes an actual primary-triggered Lum cure.

    Read the original current item/actual apply guard before AfterSetStatus;
    never infer a cure from final status or duplicate its numerical operation.
    The existing reflection guard remains in original after_status_event.
    """
    b=len(tape.count)
    primary._shape(requested,(b,),j.bool_);primary._shape(applied,(b,),j.bool_);primary._shape(item,(b,),j.int32)
    if item_table is None:return tape
    return status.unknown(tape,requested&applied&item_table.lum[item])
