"""Literal original ATK then DEF +1 self-boost pair, no numerical rewrite.

RAW41=[41,side,physical,move,stat0or1,old,new,secondaryDispatch,0,0].
A single Site27 owns both words. Device copies are trusted original-callsite
transport, not attestation; the owning host still checks original statics.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

BULK_UP=41
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    installed:object
class Write(NamedTuple):
    old_atk:object
    new_atk:object
    old_def:object
    new_def:object
    secondary_dispatch:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    old_atk:object
    old_def:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    write:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original self-pair metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,exists:m.exists,gen:m.gen,name:m.name,
category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,priority:m.priority,boosts:m.boosts||null,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined),
mixed:!!(m.self||m.status||m.secondary||m.secondaries||m.volatileStatus||m.condition||m.sideCondition||m.slotCondition||m.weather||m.terrain||m.pseudoWeather||m.recoil||m.drain||m.heal||m.selfdestruct||m.forceSwitch||m.selfSwitch)}});
process.stdout.write(JSON.stringify({rows,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    result=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(result.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original self-pair source changed')
    t=catalog.engine_table;e=t.effects
    if e is None:raise ValueError('original installed self-boost effects required')
    mask=np.zeros(len(t.power),bool)
    for i,row in enumerate(descriptor['rows']):
        mask[i]=bool(i and row['exists']and row['gen']<=3 and row['id']==metadata['vocabulary']['moves'][i]
            and row['category']=='Status'and row['target']=='self'and row['power']==0 and row['accuracy']is True
            and row['boosts']=={'atk':1,'def':1}and list(row['boosts'])==['atk','def']and not row['callbacks']and not row['mixed']
            and catalog.ids[i]>0 and catalog.targets[i]==0 and int(t.power[i])==0 and bool(t.always_hit[i])and int(t.priority[i])==row['priority']
            and np.array_equal(np.asarray(e.self_boosts[i]),[1,1,0,0,0,0,0]))
        for name in ('heal_half','rest','haze','substitute','protect','endure','heal_bell','weather_heal','pain_split','belly_drum','hit_spa_drop','status'):
            mask[i] &= not bool(np.any(np.asarray(getattr(e,name)[i])!=0))
    payload=dict(schema='pokezero.literal-self-atk-def-pair.v1',descriptor=descriptor,candidates=mask.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(e._fields,e)})
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),e)

def shape(data):
    if type(data)is not Data:raise ValueError('typed original self-pair data required')
    for v in data[:5]:life._typed(v,(8,),j.uint32)
    life._typed(data.candidates,(len(data.installed.protect),),j.bool_)

def selected(context,move):
    if context is None or context.data.bulk_up is None:return j.zeros_like(move,bool)
    d=context.data.bulk_up;shape(d)
    return (move>0)&(move<len(d.candidates))&d.candidates[j.clip(move,0,len(d.candidates)-1)]

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.bulk_up;shape(d);same=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):same &= j.all(getattr(d,name)==getattr(tape,name))
    r=tape.execution.risks
    if r is None:return j.zeros(len(tape.count),bool)
    life._typed(r.tables.bulk_up_binding,(8,),j.uint32);life._typed(r.tables.bulk_up_candidates,d.candidates.shape,j.bool_)
    return same&j.any(d.binding!=0)&j.all(r.tables.bulk_up_binding==d.binding)&j.all(r.tables.bulk_up_candidates==d.candidates)

def validate_write_device(w,b):
    if type(w)is not Write:raise ValueError('literal ATK/DEF Write required')
    for x in w[:4]:life._typed(x,(b,),j.int32)
    life._typed(w.secondary_dispatch,(b,),j.bool_)
    return (w.old_atk>=-6)&(w.old_atk<6)&(w.new_atk==w.old_atk+1)&(w.old_def>=-6)&(w.old_def<6)&(w.new_def==w.old_def+1)

def begin(tape,state,side,move,boosts,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.bulk_up is None:return tape,None
    b=len(move);rows=j.arange(b)
    for x in (side,move):life._typed(x,(b,),j.int32)
    life._typed(boosts,(b,7),j.int32);life._typed(eligible,(b,),j.bool_)
    requested=eligible&selected(context,move);d=context.data.bulk_up;good=_bound(tape,context)
    if type(table)is not type(d.installed):raise ValueError('original installed self-pair table required')
    for a,z in zip(table,d.installed,strict=True):
        if a.shape!=z.shape or a.dtype!=z.dtype:raise ValueError('self-pair table geometry differs')
        good &= j.all(a==z)
    safe=j.clip(side,0,1);physical=state.active[rows,safe]
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,safe])&(state.hp[rows,safe,physical]>0)&(tape.cursor.phase==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.BULK_UP,side,active=requested)
    return tape,Pending(context,ticket,side,physical,move,boosts[:,0],boosts[:,1],requested,good)

def _pair(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,))
    raw=tape.words[rows,j.clip(at,0,tape.words.shape[1]-1)];first_at=at-j.where(raw[:,4]==0,0,1)
    first=j.clip(first_at,0,tape.words.shape[1]-1);second=j.clip(first_at+1,0,tape.words.shape[1]-1)
    a=tape.words[rows,first];z=tape.words[rows,second]
    if c is None or c.risks is None:return a,z,j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger
    life._typed(t.bulk_up_binding,(8,),j.uint32);life._typed(t.bulk_up_candidates,(len(t.selected_risk),),j.bool_)
    claim=j.where(stage==1,ledger.count,ledger.claimed[rows,first]);slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.bulk_up_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32)
        good &= j.all(getattr(t,name)==getattr(tape,name))
    move=a[:,3];safe=j.clip(move,0,len(t.bulk_up_candidates)-1)
    good &= (first_at>0)&(first_at+1<tape.count)&(claim>0)&(claim<=ledger.count)
    good &= (a[:,0]==BULK_UP)&(z[:,0]==BULK_UP)&(a[:,4]==0)&(z[:,4]==1)&j.all(a[:,1:4]==z[:,1:4],axis=1)&(a[:,7]==z[:,7])&j.all(a[:,8:]==0,axis=1)&j.all(z[:,8:]==0,axis=1)
    good &= (a[:,1]>=0)&(a[:,1]<2)&(a[:,2]>=0)&(a[:,2]<6)&(move>0)&(move<len(t.bulk_up_candidates))&t.bulk_up_candidates[safe]
    good &= validate_write_device(Write(a[:,5],a[:,6],z[:,5],z[:,6],a[:,7]!=0),b)&((a[:,7]==0)|(a[:,7]==1))
    good &= (ledger.site[rows,slot]==int(ex.Site.BULK_UP))&(ledger.side[rows,slot]==a[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==first_at)
    good &= j.where(stage==1,tape.count==first_at+2,ledger.end[rows,slot]==first_at+2)
    prefix=j.clip(first_at-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (decl[:,0]==public.MOVE)&j.all(decl[:,1:3]==a[:,1:3],axis=1)&(decl[:,3]==1-a[:,1])&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1)
    for pos in (prefix,first,second):
        good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    if stage==1:
        good &= (a[:,2]==tape.cursor.physical_active[rows,j.clip(a[:,1],0,1)])&(decl[:,4]==tape.cursor.physical_active[rows,j.clip(1-a[:,1],0,1)])
    if stage==2:
        for pos in (first,second):good &= (ledger.claimed[rows,pos]==claim)&j.all(ledger.words[rows,pos]==tape.words[rows,pos],axis=1)&(ledger.event_turn[rows,pos]==tape.event_turn[rows,pos])&(ledger.event_phase[rows,pos]==tape.event_phase[rows,pos])&j.all(ledger.event_generation[rows,pos]==tape.event_generation[rows,pos],axis=1)
    return a,z,good

def owned_pair(tape,index):return _pair(tape,index)[2]

def finish(tape,p,boosts,secondary):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    b=len(tape.count);life._typed(boosts,(b,7),j.int32);w=Write(p.old_atk,boosts[:,0],p.old_def,boosts[:,1],secondary)
    valid=p.valid&_bound(tape,p.context)&validate_write_device(w,b);room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-2)
    overflow=p.requested&valid&~room;rows=j.arange(b)
    for offset in (0,1):
        address=tape.count+offset;inside=(address>=0)&(address<tape.words.shape[1]);at=j.clip(address,0,tape.words.shape[1]-1)
        valid &= ~inside|j.all(tape.switch_details[rows,at]==0,axis=1)
    z=j.zeros(b,j.int32);use=p.requested&valid&room
    for stat,old,new in ((0,w.old_atk,w.new_atk),(1,w.old_def,w.new_def)):
        tape=public._append(tape,BULK_UP,p.side,p.physical,p.move,z+stat,(old,new,secondary.astype(j.int32),z,z),use)
    a,z,good=_pair(tape,j.maximum(tape.count-2,0),stage=1);slot,same=ex._ticket(tape.execution.ledger,p.ticket)
    good &= valid&room&same&(tape.execution.ledger.start[rows,slot]==tape.count-2)
    for name,value in ((1,p.side),(2,p.physical),(3,p.move)):good &= a[:,name]==value
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables)
    tape=ex._put_context(tape,ledger,c);tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(overflow,public.OVERFLOW,0))
    return tape,Captured(p,w)

def captured_matches(tape,captured):
    if captured is None:return j.zeros(len(tape.count),bool)
    if type(captured)is not Captured or type(captured.pending)is not Pending:raise ValueError('typed SAME self-pair capture required')
    from . import engine_execution_completeness as ex
    p,w=captured;rows=j.arange(len(tape.count));slot,same=ex._ticket(tape.execution.ledger,p.ticket);start=tape.execution.ledger.start[rows,slot]
    a,z,good=_pair(tape,start);good &= same&p.requested&p.valid&_bound(tape,p.context)&validate_write_device(w,len(rows))
    for col,value in ((1,p.side),(2,p.physical),(3,p.move),(5,w.old_atk),(6,w.new_atk),(7,w.secondary_dispatch.astype(j.int32))):good &= a[:,col]==value
    good &= (z[:,5]==w.old_def)&(z[:,6]==w.new_def)&(tape.count==start+2)
    ledger=tape.execution.ledger;good &= (ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    return good&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])
