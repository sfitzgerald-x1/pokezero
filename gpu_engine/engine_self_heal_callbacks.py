"""Literal positive Recover/Soft-Boiled/Synthesis SAME-write input.

RAW45=[45,move,effectiveWeather,side,physical,oldHP,newHP,maxHP,status,
       literalSecondaryDispatch]. Site32 owns exactly one word. This observes
existing assignments; full/no-effect/called/unsupported branches stay UNKNOWN.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

SELF_HEAL=45
SYNTHESIS_SHA='f055a11ff7d7d6a15761908c36fef2c6d9d1ce23e696feb1b8be8309466ee75b'
HEAL_SHA='75d2b2e30f95f52a9adc9847fae763fe92145535416f69ebb0ec0782d9d65ddb'
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    installed:object
    weather:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    old_hp:object
    maximum:object
    status:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    raw:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original self-heal metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),{Battle}=require(path.join(root,'dist/sim/battle.js'));
const h=x=>crypto.createHash('sha256').update(x).digest('hex');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,exists:m.exists,gen:m.gen,name:m.name,
category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,priority:m.priority,heal:m.heal||null,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined),onHitSHA:m.onHit?h(m.onHit.toString()):null,
mixed:!!(m.self||m.boosts||m.status||m.secondary||m.secondaries||m.volatileStatus||m.condition||m.sideCondition||m.slotCondition||m.weather||m.terrain||m.pseudoWeather||m.recoil||m.drain||m.selfdestruct||m.forceSwitch||m.selfSwitch)}});
process.stdout.write(JSON.stringify({rows,healSHA:h(Battle.prototype.heal.toString()),source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    result=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(result.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json or descriptor['healSHA']!=HEAL_SHA:raise ValueError('original self-heal source changed')
    t=catalog.engine_table;e=t.effects
    if e is None:raise ValueError('actual original self-heal effects required')
    mask=np.zeros(len(t.power),bool)
    for i,row in enumerate(descriptor['rows']):
        half=row['id']in ('recover','softboiled')and row['heal']==[1,2]and not row['callbacks']
        weather=row['id']=='synthesis'and row['heal']is None and row['callbacks']==['onHit']and row['onHitSHA']==SYNTHESIS_SHA
        mask[i]=bool(i and(half or weather)and row['exists']and row['gen']<=3 and row['id']==metadata['vocabulary']['moves'][i]
            and row['category']=='Status'and row['target']=='self'and row['power']==0 and row['accuracy']is True and not row['mixed']
            and catalog.ids[i]>0 and catalog.targets[i]==0 and int(t.power[i])==0 and bool(t.always_hit[i])and int(t.priority[i])==row['priority']
            and bool(e.heal_half[i])==half and bool(e.weather_heal[i])==weather)
        for name in ('status','self_boosts','rest','haze','substitute','protect','endure','heal_bell','pain_split','belly_drum','hit_spa_drop','thunderwave','bypass_sub'):
            mask[i]&=not bool(np.any(np.asarray(getattr(e,name)[i])!=0))
    payload=dict(schema='pokezero.positive-original-self-heal.v1',descriptor=descriptor,candidates=mask.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,
        installed={k:np.asarray(v).tolist()for k,v in zip(e._fields,e)},weather=None if t.weather is None else {k:np.asarray(v).tolist()for k,v in zip(t.weather._fields,t.weather)})
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),e,t.weather)

def shape(d):
    if type(d)is not Data:raise ValueError('typed original self-heal data required')
    for x in d[:5]:life._typed(x,(8,),j.uint32)
    life._typed(d.candidates,(len(d.installed.heal_half),),j.bool_)

def selected(context,move):
    if context is None or context.data.self_heal is None:return j.zeros_like(move,bool)
    d=context.data.self_heal;shape(d)
    return(move>0)&(move<len(d.candidates))&d.candidates[j.clip(move,0,len(d.candidates)-1)]

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.self_heal;shape(d);good=direct._registered_bound(tape,context)
    r=tape.execution.risks
    if r is None:return j.zeros(len(tape.count),bool)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good&=j.all(getattr(d,name)==getattr(tape,name))
    life._typed(r.tables.self_heal_binding,(8,),j.uint32);life._typed(r.tables.self_heal_candidates,d.candidates.shape,j.bool_)
    return good&life._valid(tape.cursor)&tape.cursor.initialized&(tape.turn==tape.cursor.turn)&(tape.cursor.phase==0)&j.any(d.binding!=0)&j.all(r.tables.self_heal_binding==d.binding)&j.all(r.tables.self_heal_candidates==d.candidates)

def begin(tape,state,side,move,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.self_heal is None:return tape,None
    b=len(move);rows=j.arange(b);d=context.data.self_heal
    for x in (side,move):life._typed(x,(b,),j.int32)
    life._typed(eligible,(b,),j.bool_);good=_bound(tape,context)
    for actual,expected in ((table.effects,d.installed),(table.weather,d.weather)):
        if type(actual)is not type(expected):raise ValueError('original self-heal installed table required')
        if actual is not None:
            for x,y in zip(actual,expected,strict=True):
                if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('self-heal installed table geometry')
                good&=j.all(x==y)
    safe=j.clip(side,0,1);physical=state.active[rows,safe];hp=state.hp[rows,safe,physical];maximum=state.max_hp[rows,safe,physical];status=state.status[rows,safe,physical]
    good&=(side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,safe])&(hp>0)&(hp<maximum)&(maximum<=1000000)&(tape.cursor.phase==0)
    requested=eligible&selected(context,move);tape,ticket=ex.open_attempt(tape,ex.Site.SELF_HEAL,side,active=requested)
    return tape,Pending(context,ticket,side,physical,move,hp,maximum,status,requested,good)

def valid_raw(raw):
    return(raw[:,0]==SELF_HEAL)&(raw[:,1]>0)&(raw[:,2]>=0)&(raw[:,2]<=4)&(raw[:,3]>=0)&(raw[:,3]<2)&(raw[:,4]>=0)&(raw[:,4]<6)&(raw[:,5]>0)&(raw[:,6]>raw[:,5])&(raw[:,6]<=raw[:,7])&(raw[:,7]<=1000000)&(raw[:,8]>=0)&(raw[:,8]<=6)&((raw[:,9]==0)|(raw[:,9]==1))

def _owned(tape,index,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,));safe=j.clip(at,0,tape.words.shape[1]-1);raw=tape.words[rows,safe]
    if c is None or c.risks is None:return raw,j.zeros(b,bool)
    ledger=c.ledger;t=c.risks.tables
    life._typed(t.self_heal_binding,(8,),j.uint32);life._typed(t.self_heal_candidates,(len(t.selected_risk),),j.bool_)
    claim=ledger.count if stage==1 else ledger.claimed[rows,safe];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.self_heal_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good&=j.all(getattr(tape,name)==getattr(t,name))
    move=raw[:,1];good&=valid_raw(raw)&(at>0)&(at<tape.count)&(move<len(t.self_heal_candidates))&t.self_heal_candidates[j.clip(move,0,len(t.self_heal_candidates)-1)]&(claim>0)&(claim<=ledger.count)
    good&=(ledger.site[rows,slot]==int(ex.Site.SELF_HEAL))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==at)&(ledger.no_output_kind[rows,slot]==0)
    good&=j.where(stage==1,tape.count==at+1,ledger.end[rows,slot]==at+1)
    prefix=j.clip(at-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good&=(decl[:,0]==public.MOVE)&(decl[:,1]==raw[:,3])&(decl[:,2]==raw[:,4])&(decl[:,3]==1-raw[:,3])&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1)&j.all(tape.switch_details[rows,safe]==0,axis=1)
    for pos in (prefix,safe):good&=(tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    if stage==1:
        good&=(raw[:,4]==tape.cursor.physical_active[rows,j.clip(raw[:,3],0,1)])&(decl[:,4]==tape.cursor.physical_active[rows,j.clip(1-raw[:,3],0,1)])&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    else:
        good&=(ledger.claimed[rows,safe]==claim)&j.all(ledger.words[rows,safe]==raw,axis=1)&(ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])&j.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    return raw,good

def owned(tape,index):return _owned(tape,index)[1]

def finish(tape,p,state,weather,secondary):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b)
    life._typed(weather,(b,),j.int32);life._typed(secondary,(b,),j.bool_)
    safe=j.clip(p.side,0,1);new=state.hp[rows,safe,p.physical];maximum=state.max_hp[rows,safe,p.physical];status=state.status[rows,safe,p.physical]
    raw=j.stack((j.full_like(p.move,SELF_HEAL),p.move,weather,p.side,p.physical,p.old_hp,new,maximum,status,secondary.astype(j.int32)),axis=1)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw)&(maximum==p.maximum)&(status==p.status)&(state.active[rows,safe]==p.physical)
    room=(tape.count>=0)&(tape.count<tape.words.shape[1]);at=j.clip(tape.count,0,tape.words.shape[1]-1);valid&=j.all(tape.switch_details[rows,at]==0,axis=1)
    use=p.requested&valid&room;tape=public._append(tape,SELF_HEAL,p.move,weather,p.side,p.physical,tuple(raw[:,i]for i in range(5,10)),use)
    actual,good=_owned(tape,j.maximum(tape.count-1,0),stage=1);slot,same=ex._ticket(tape.execution.ledger,p.ticket);good&=valid&room&same&j.all(actual==raw,axis=1)
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables)
    tape=ex._put_context(tape,ledger,c);tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&valid&~room,public.OVERFLOW,0))
    return tape,Captured(p,raw)

def captured_matches(tape,captured):
    if captured is None:return j.zeros(len(tape.count),bool)
    if type(captured)is not Captured or type(captured.pending)is not Pending:raise ValueError('typed SAME self-heal capture required')
    p=captured.pending;from . import engine_execution_completeness as ex
    c=ex._context(tape);slot,same=ex._ticket(c.ledger,p.ticket);rows=j.arange(len(tape.count));raw,good=_owned(tape,j.maximum(tape.count-1,0))
    return good&same&p.requested&p.valid&_bound(tape,p.context)&(c.ledger.turn[rows,slot]==tape.cursor.turn)&(c.ledger.phase[rows,slot]==tape.cursor.phase)&j.all(c.ledger.generation[rows,slot]==tape.cursor.generation,axis=1)&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])&(c.ledger.end[rows,slot]==tape.count)&j.all(raw==captured.raw,axis=1)&(raw[:,1]==p.move)&(raw[:,3]==p.side)&(raw[:,4]==p.physical)&(raw[:,5]==p.old_hp)&(raw[:,7]==p.maximum)&(raw[:,8]==p.status)
