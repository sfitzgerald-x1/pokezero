"""Successful original plain primary NONE->PAR/BRN writes, RAW9+26 only.

No status/HP/RNG rewrite, generic no-output exemption or copied-carry origin.
The original host guards the DirectData compiler; both owners drain the span.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_lifecycle_callbacks as life,engine_toxic_callbacks as primary,engine_status_callbacks as status,mechanics as m

Attempt=primary.Attempt

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    status:object
    installed:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    target:object
    move:object
    old_status:object
    old_counter:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    attempt:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original plain-primary metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,name:m.name,exists:m.exists,gen:m.gen,
category:m.category,target:m.target,power:m.basePower,status:m.status||'',accuracy:m.accuracy,priority:m.priority,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined),mixed:!!(m.self||m.boosts||m.secondary||m.secondaries||m.volatileStatus||m.condition||m.sideCondition||m.slotCondition||m.weather||m.terrain||m.pseudoWeather||m.recoil||m.drain||m.heal||m.selfdestruct||m.forceSwitch||m.selfSwitch)}});
const conditions=['par','brn'].map(id=>({id,start:String(d.conditions.get(id).onStart)}));
process.stdout.write(JSON.stringify({rows,conditions,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    result=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(result.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original plain-primary source changed')
    t=catalog.engine_table;e=t.effects;mask=np.zeros(len(t.power),bool);codes=np.zeros(len(t.power),np.int32)
    if e is None or t.secondary is None:raise ValueError('original installed status tables required')
    for i,row in enumerate(descriptor['rows']):
        code={'par':m.PAR,'brn':m.BRN}.get(row['status'],0);codes[i]=code
        mask[i]=bool(i and row['exists']and row['gen']<=3 and row['id']==metadata['vocabulary']['moves'][i]
            and row['category']=='Status'and row['target']=='normal'and row['power']==0 and code and not row['mixed']and not row['callbacks']
            and catalog.ids[i]>0 and catalog.targets[i]==1 and int(t.power[i])==0 and int(e.status[i])==code and not bool(e.thunderwave[i])
            and int(t.priority[i])==row['priority']and float(t.accuracy[i])==float(row['accuracy']))
        for name in ('self_boosts','heal_half','rest','haze','substitute','protect','endure','heal_bell','weather_heal','pain_split','belly_drum','hit_spa_drop','bypass_sub'):
            mask[i] &= not bool(np.any(np.asarray(getattr(e,name)[i])!=0))
    payload=dict(schema='pokezero.literal-primary-plain-par-brn.v1',descriptor=descriptor,mask=mask.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(e._fields,e)})
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),j.asarray(codes),e)

def shape(d):
    if type(d)is not Data:raise ValueError('typed original plain-primary data required')
    for v in d[:5]:life._typed(v,(8,),j.uint32)
    life._typed(d.candidates,(len(d.installed.protect),),j.bool_);life._typed(d.status,d.candidates.shape,j.int32)

def selected(context,move):
    if context is None or context.data.primary_plain_status is None:return j.zeros_like(move,bool)
    d=context.data.primary_plain_status;shape(d);return (move>0)&(move<len(d.candidates))&d.candidates[j.clip(move,0,len(d.candidates)-1)]

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.primary_plain_status;shape(d);good=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(d,name)==getattr(tape,name))
    c=tape.execution.risks
    if c is None:return j.zeros(len(tape.count),bool)
    t=c.tables;life._typed(t.primary_plain_binding,(8,),j.uint32);life._typed(t.primary_plain_candidates,d.candidates.shape,j.bool_);life._typed(t.primary_plain_status,d.status.shape,j.int32)
    return good&j.any(d.binding!=0)&j.all(t.primary_plain_binding==d.binding)&j.all(t.primary_plain_candidates==d.candidates)&j.all(t.primary_plain_status==d.status)

def begin(tape,state,side,move,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.primary_plain_status is None:return tape,None
    b=len(move);rows=j.arange(b)
    for v in (side,move):life._typed(v,(b,),j.int32)
    life._typed(eligible,(b,),j.bool_);d=context.data.primary_plain_status;requested=eligible&selected(context,move);good=_bound(tape,context)
    if type(table.effects)is not type(d.installed):raise ValueError('typed original installed effect table required')
    for actual,expected in zip(table.effects,d.installed,strict=True):
        if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('plain-primary installed effect geometry differs')
        good &= j.all(actual==expected)
    s=context.data.secondary_status
    if s is None:raise ValueError('same original installed status-hook table required')
    if type(table.secondary)is not type(s.installed):raise ValueError('typed original installed status-hook table required')
    for actual,expected in zip(table.secondary,s.installed,strict=True):
        if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('plain-primary status-hook geometry differs')
        good &= j.all(actual==expected)
    physical=state.active[rows,side];target=state.active[rows,1-side];abi=state.ability[rows,1-side,target];item=state.item[rows,1-side,target]
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,side])&(target==tape.cursor.physical_active[rows,1-side])
    good &= (abi>=0)&(abi<len(s.synchronize))&~s.synchronize[j.clip(abi,0,len(s.synchronize)-1)]&(item>=0)&(item<len(s.lum))&~s.lum[j.clip(item,0,len(s.lum)-1)]
    good &= tape.cursor.initialized&(tape.cursor.phase==0)&(tape.turn==tape.cursor.turn)
    tape,ticket=ex.open_attempt(tape,ex.Site.PRIMARY_PLAIN_STATUS,1-side,active=requested)
    return tape,Pending(context,ticket,side,physical,target,move,state.status[rows,1-side,target],state.status_counter[rows,1-side,target],requested,good)

def valid_raw(raw,candidates,codes):
    move=raw[:,5];safe=j.clip(move,0,len(candidates)-1);f=raw[:,6]
    required=primary.ACCURACY_HIT|primary.FINAL_HIT|primary.LIVE|primary.APPLIED
    allowed=required|primary.DAMAGE_IMMUNE|primary.SECONDARY
    return ((raw[:,0]==primary.TOXIC_ATTEMPT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)
        &(move>0)&(move<len(candidates))&candidates[safe]&((f&required)==required)&((f&~allowed)==0)
        &(((f&primary.SECONDARY)!=0)==((f&primary.DAMAGE_IMMUNE)==0))&(raw[:,7]==m.NONE)&(raw[:,8]==codes[safe])&(raw[:,9]==0))

def owned_pair(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,));last=j.clip(at,0,tape.words.shape[1]-1);first=j.clip(at-1,0,tape.words.shape[1]-1)
    if c is None or c.risks is None:return j.zeros(b,bool)
    raw=tape.words[rows,last];s=tape.words[rows,first];t=c.risks.tables;ledger=c.ledger
    life._typed(t.primary_plain_binding,(8,),j.uint32);life._typed(t.primary_plain_candidates,(len(t.selected_risk),),j.bool_);life._typed(t.primary_plain_status,(len(t.selected_risk),),j.int32)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.primary_plain_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(t,name))
    claim=j.where(stage==1,ledger.count,ledger.claimed[rows,last]);slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good &= valid_raw(raw,t.primary_plain_candidates,t.primary_plain_status)&(at>1)&(at<tape.count)&(claim>0)&(claim<=ledger.count)
    good &= (s[:,0]==status.STATUS_APPLIED)&j.all(s[:,1:3]==-1,axis=1)&j.all(s[:,3:5]==raw[:,3:5],axis=1)&(s[:,5]==raw[:,7])&(s[:,6]==raw[:,8])&j.all(s[:,7:]==0,axis=1)
    good &= (ledger.site[rows,slot]==int(ex.Site.PRIMARY_PLAIN_STATUS))&(ledger.side[rows,slot]==raw[:,3])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==at-1)
    good &= j.where(stage==1,tape.count==at+1,ledger.end[rows,slot]==at+1)
    prefix=j.clip(at-2,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (decl[:,0]==1)&j.all(decl[:,1:6]==raw[:,1:6],axis=1)&j.all(decl[:,6:]==0,axis=1)
    for pos in (prefix,first,last):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    if stage==1:good &= (raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(raw[:,4]==tape.cursor.physical_active[rows,j.clip(raw[:,3],0,1)])
    if stage==2:
        for pos in (first,last):good &= (ledger.claimed[rows,pos]==claim)&j.all(ledger.words[rows,pos]==tape.words[rows,pos],axis=1)&(ledger.event_turn[rows,pos]==tape.event_turn[rows,pos])&(ledger.event_phase[rows,pos]==tape.event_phase[rows,pos])&j.all(ledger.event_generation[rows,pos]==tape.event_generation[rows,pos],axis=1)
    return good

def finish(tape,p,attempt):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex,engine_public_events as public
    if type(attempt)is not primary.Attempt:raise ValueError('literal primary setter operands required')
    raw=primary.words_device(p.side,p.physical,p.target,p.move,attempt);d=p.context.data.primary_plain_status
    good=p.valid&_bound(tape,p.context)&valid_raw(raw,d.candidates,d.status)&(attempt.old_status==p.old_status)&(attempt.old_counter==p.old_counter)&(attempt.old_counter==0)&(attempt.new_counter==0)
    tape=public._append(tape,primary.TOXIC_ATTEMPT,p.side,p.physical,1-p.side,p.target,tuple(raw[:,i]for i in range(5,10)),p.requested&good)
    good &= owned_pair(tape,j.maximum(tape.count-1,0),stage=1);rows=j.arange(len(tape.count));slot,same=ex._ticket(tape.execution.ledger,p.ticket);good &= same&(tape.execution.ledger.start[rows,slot]==tape.count-2)
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables)
    tape=ex._put_context(tape,ledger,c);tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0))
    return tape,Captured(p,attempt)

def captured_matches(tape,captured):
    if captured is None:return j.zeros(len(tape.count),bool)
    if type(captured)is not Captured or type(captured.pending)is not Pending:raise ValueError('typed same-primary setter capture required')
    from . import engine_execution_completeness as ex
    p,w=captured;rows=j.arange(len(tape.count));ledger=tape.execution.ledger;slot,same=ex._ticket(ledger,p.ticket);end=ledger.end[rows,slot]
    raw=tape.words[rows,j.clip(end-1,0,tape.words.shape[1]-1)];expected=primary.words_device(p.side,p.physical,p.target,p.move,w)
    good=owned_pair(tape,end-1)&same&p.requested&p.valid&_bound(tape,p.context)&j.all(raw==expected,axis=1)&(tape.count==end)&(w.old_status==p.old_status)&(w.old_counter==p.old_counter)&(w.old_counter==0)&(w.new_counter==0)
    good &= (ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    return good&(p.physical==tape.cursor.physical_active[rows,p.side])&(p.target==tape.cursor.physical_active[rows,1-p.side])
