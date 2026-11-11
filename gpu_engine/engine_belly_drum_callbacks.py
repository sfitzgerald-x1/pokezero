"""Actual Belly Drum HP payment then absolute ATK6, one Site31/two RAW44.

Only original canonical successful writes are supported. Failure/called/capped
and interleaved output remain UNKNOWN. No final-state reconstruction or flags
establish caller origin; the owning host must retain the original context.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

BELLY_DRUM=44
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    installed:object
class Write(NamedTuple):
    old_hp:object
    new_hp:object
    maximum:object
    status:object
    old_atk:object
    new_atk:object
    secondary_dispatch:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    old_hp:object
    maximum:object
    status:object
    old_atk:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    write:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Belly Drum metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,exists:m.exists,gen:m.gen,name:m.name,
category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,priority:m.priority,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined),onHit:m.onHit?require('crypto').createHash('sha256').update(m.onHit.toString()).digest('hex'):null}});
process.stdout.write(JSON.stringify({rows,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Belly Drum source changed')
    t=catalog.engine_table;e=t.effects
    if e is None:raise ValueError('original installed Belly Drum effects required')
    mask=np.zeros(len(t.power),bool)
    for i,row in enumerate(descriptor['rows']):
        mask[i]=bool(i and row['id']=='bellydrum'and row['exists']and row['gen']<=3 and row['id']==metadata['vocabulary']['moves'][i]
            and row['category']=='Status'and row['target']=='self'and row['power']==0 and row['accuracy']is True and row['priority']==0
            and row['callbacks']==['onHit']and row['onHit']=='8f4ffc9cf90da1aabf0e729d70face6e71087b0072b19be174ecf3725c8a1059'
            and catalog.ids[i]>0 and catalog.targets[i]==0 and int(t.power[i])==0 and bool(t.always_hit[i])and int(t.priority[i])==0
            and bool(e.belly_drum[i])and not np.any(np.asarray(e.self_boosts[i])))
        for name in ('heal_half','rest','haze','substitute','protect','endure','heal_bell','weather_heal','pain_split','hit_spa_drop','status'):
            mask[i]&=not bool(np.any(np.asarray(getattr(e,name)[i])!=0))
    payload=dict(schema='pokezero.literal-belly-drum-hp-atk.v1',descriptor=descriptor,candidates=mask.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(e._fields,e)})
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),e)

def shape(data):
    if type(data)is not Data:raise ValueError('typed original Belly Drum data required')
    for v in data[:5]:life._typed(v,(8,),j.uint32)
    life._typed(data.candidates,(len(data.installed.belly_drum),),j.bool_)

def selected(context,move):
    if context is None or context.data.belly_drum is None:return j.zeros_like(move,bool)
    d=context.data.belly_drum;shape(d)
    return (move>0)&(move<len(d.candidates))&d.candidates[j.clip(move,0,len(d.candidates)-1)]

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.belly_drum;shape(d);same=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):same &= j.all(getattr(d,name)==getattr(tape,name))
    r=tape.execution.risks
    if r is None:return j.zeros(len(tape.count),bool)
    life._typed(r.tables.belly_drum_binding,(8,),j.uint32);life._typed(r.tables.belly_drum_candidates,d.candidates.shape,j.bool_)
    return same&j.any(d.binding!=0)&j.all(r.tables.belly_drum_binding==d.binding)&j.all(r.tables.belly_drum_candidates==d.candidates)

def validate_write(w,b):
    if type(w)is not Write:raise ValueError('literal Belly Drum Write required')
    for x in w[:6]:life._typed(x,(b,),j.int32)
    life._typed(w.secondary_dispatch,(b,),j.bool_)
    return ((w.maximum>1)&(w.maximum<=1_000_000)&(w.old_hp>w.maximum//2)&(w.old_hp<=w.maximum)
        &(w.new_hp==w.old_hp-w.maximum//2)&(w.new_hp>0)&(w.status>=0)&(w.status<=6)
        &(w.old_atk>=-6)&(w.old_atk<6)&(w.new_atk==6))

def begin(tape,state,side,move,hp,boosts,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.belly_drum is None:return tape,None
    b=len(move);rows=j.arange(b);life._typed(boosts,(b,7),j.int32)
    for x in (side,move,hp):life._typed(x,(b,),j.int32)
    life._typed(eligible,(b,),j.bool_);requested=eligible&selected(context,move);d=context.data.belly_drum;good=_bound(tape,context)
    if type(table)is not type(d.installed):raise ValueError('original installed Belly Drum table required')
    for a,z in zip(table,d.installed,strict=True):
        if a.shape!=z.shape or a.dtype!=z.dtype:raise ValueError('Belly Drum table geometry differs')
        good &= j.all(a==z)
    safe=j.clip(side,0,1);physical=state.active[rows,safe]
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,safe])&(hp==state.hp[rows,safe,physical])&(tape.cursor.phase==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.BELLY_DRUM,side,active=requested)
    return tape,Pending(context,ticket,side,physical,move,hp,state.max_hp[rows,safe,physical],state.status[rows,safe,physical],boosts[:,0],requested,good)

def pair(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,))
    raw=tape.words[rows,j.clip(at,0,tape.words.shape[1]-1)];start=at-raw[:,4]
    first=j.clip(start,0,tape.words.shape[1]-1);second=j.clip(start+1,0,tape.words.shape[1]-1);a=tape.words[rows,first];z=tape.words[rows,second]
    if c is None or c.risks is None:return a,z,j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger
    life._typed(t.belly_drum_binding,(8,),j.uint32);life._typed(t.belly_drum_candidates,(len(t.selected_risk),),j.bool_)
    claim=j.where(stage==1,ledger.count,ledger.claimed[rows,first]);slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.belly_drum_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(t,name)==getattr(tape,name))
    move=a[:,3];safe=j.clip(move,0,len(t.belly_drum_candidates)-1)
    good &= (raw[:,0]==BELLY_DRUM)&((raw[:,4]==0)|(raw[:,4]==1))
    good &= (start>0)&(start+1<tape.count)&(claim>0)&(claim<=ledger.count)
    good &= (a[:,0]==BELLY_DRUM)&(z[:,0]==BELLY_DRUM)&(a[:,4]==0)&(z[:,4]==1)&j.all(a[:,1:4]==z[:,1:4],axis=1)&(a[:,9]==z[:,7])&j.all(z[:,8:]==0,axis=1)
    good &= (a[:,1]>=0)&(a[:,1]<2)&(a[:,2]>=0)&(a[:,2]<6)&(move>0)&(move<len(t.belly_drum_candidates))&t.belly_drum_candidates[safe]
    good &= validate_write(Write(a[:,5],a[:,6],a[:,7],a[:,8],z[:,5],z[:,6],a[:,9]!=0),b)&((a[:,9]==0)|(a[:,9]==1))
    good &= (ledger.site[rows,slot]==int(ex.Site.BELLY_DRUM))&(ledger.side[rows,slot]==a[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==start)
    good &= j.where(stage==1,tape.count==start+2,ledger.end[rows,slot]==start+2)
    prefix=j.clip(start-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (decl[:,0]==public.MOVE)&j.all(decl[:,1:3]==a[:,1:3],axis=1)&(decl[:,3]==1-a[:,1])&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1)
    good &= j.all(tape.switch_details[rows,first]==0,axis=1)&j.all(tape.switch_details[rows,second]==0,axis=1)
    for pos in (prefix,first,second):good &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
    if stage==1:good &= (a[:,2]==tape.cursor.physical_active[rows,j.clip(a[:,1],0,1)])&(decl[:,4]==tape.cursor.physical_active[rows,j.clip(1-a[:,1],0,1)])
    if stage==2:
        for pos in (first,second):good &= (ledger.claimed[rows,pos]==claim)&j.all(ledger.words[rows,pos]==tape.words[rows,pos],axis=1)&(ledger.event_turn[rows,pos]==tape.event_turn[rows,pos])&(ledger.event_phase[rows,pos]==tape.event_phase[rows,pos])&j.all(ledger.event_generation[rows,pos]==tape.event_generation[rows,pos],axis=1)
    return a,z,good

def finish(tape,p,state,secondary):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);seat=j.clip(p.side,0,1);where=state.active[rows,seat]
    w=Write(p.old_hp,state.hp[rows,seat,where],p.maximum,p.status,p.old_atk,state.boosts[rows,seat,where,0],secondary)
    valid=p.valid&_bound(tape,p.context)&validate_write(w,b)&(where==p.physical)&(state.max_hp[rows,seat,where]==p.maximum)&(state.status[rows,seat,where]==p.status)
    room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-2);overflow=p.requested&valid&~room
    for offset in (0,1):
        address=tape.count+offset;inside=(address>=0)&(address<tape.words.shape[1]);at=j.clip(address,0,tape.words.shape[1]-1)
        valid &= ~inside|j.all(tape.switch_details[rows,at]==0,axis=1)
    zero=j.zeros(b,j.int32);use=p.requested&valid&room
    tape=public._append(tape,BELLY_DRUM,p.side,p.physical,p.move,zero,(w.old_hp,w.new_hp,w.maximum,w.status,secondary.astype(j.int32)),use)
    tape=public._append(tape,BELLY_DRUM,p.side,p.physical,p.move,zero+1,(w.old_atk,w.new_atk,secondary.astype(j.int32),zero,zero),use)
    a,z,good=pair(tape,j.maximum(tape.count-1,0),stage=1);good &= valid&room
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables)
    ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(overflow,public.OVERFLOW,0))
    return tape,Captured(p,w)

def captured_matches(tape,captured):
    if type(captured)is not Captured:return j.zeros(len(tape.count),bool)
    p=captured.pending;w=captured.write;b=len(tape.count);rows=j.arange(b)
    a,z,good=pair(tape,j.maximum(tape.count-1,0));good &= _bound(tape,p.context)&p.requested&p.valid&validate_write(w,b)
    for col,value in ((1,p.side),(2,p.physical),(3,p.move),(5,w.old_hp),(6,w.new_hp),(7,w.maximum),(8,w.status),(9,w.secondary_dispatch.astype(j.int32))):good &= a[:,col]==value
    good &= (z[:,5]==w.old_atk)&(z[:,6]==w.new_atk)&(w.old_hp==p.old_hp)&(w.maximum==p.maximum)&(w.status==p.status)&(w.old_atk==p.old_atk)
    from . import engine_execution_completeness as ex
    c=tape.execution;slot,same=ex._ticket(c.ledger,p.ticket)
    good &= same&(c.ledger.claimed[rows,j.clip(c.ledger.start[rows,slot],0,tape.words.shape[1]-1)]==slot+1)&(c.ledger.end[rows,slot]==tape.count)&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])&(c.ledger.turn[rows,slot]==tape.cursor.turn)&(c.ledger.phase[rows,slot]==tape.cursor.phase)&j.all(c.ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    return good
