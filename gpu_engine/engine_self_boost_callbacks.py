"""SAME-operation ordinary positive self boosts, no numerical implementation.

RAW53=[53,side,physical,move,stat,old,new,secondaryDispatch,ordinal,length].
Site38 owns one SPA/SPE+2 word or the ordered ATK+1,SPE+1 pair. Caps,
partial caps, no writes and called moves remain UNKNOWN. No empty receipt.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

SELF_BOOST_INPUT=53
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    stats:object
    deltas:object
    lengths:object
    installed:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    old:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    new:object
    secondary:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original self-boost metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,exists:m.exists,gen:m.gen,
category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,priority:m.priority,boosts:m.boosts||null,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined),
mixed:!!(m.self||m.status||m.secondary||m.secondaries||m.volatileStatus||m.condition||m.sideCondition||m.slotCondition||m.weather||m.terrain||m.pseudoWeather||m.recoil||m.drain||m.heal||m.selfdestruct||m.forceSwitch||m.selfSwitch)}});
process.stdout.write(JSON.stringify({rows,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    result=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(result.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original self-boost source changed')
    t=catalog.engine_table;e=t.effects
    if e is None:raise ValueError('original installed self-boost effects required')
    stats=np.full((len(t.power),2),-1,np.int32);deltas=np.zeros_like(stats);lengths=np.zeros(len(t.power),np.int32)
    families={'tailglow':([2],[2],{'spa':2}),'agility':([4],[2],{'spe':2}),'dragondance':([0,4],[1,1],{'atk':1,'spe':1})}
    for i,row in enumerate(descriptor['rows']):
        name=metadata['vocabulary']['moves'][i]
        if name not in families:continue
        ss,dd,source=families[name];vector=np.zeros(7,np.int32);vector[ss]=dd
        good=bool(i and row['exists']and row['gen']<=3 and row['id']==name
            and row['category']=='Status'and row['target']=='self'and row['power']==0 and row['accuracy']is True
            and row['boosts']==source and list(row['boosts'])==list(source)and not row['callbacks']and not row['mixed']
            and catalog.ids[i]>0 and catalog.targets[i]==0 and int(t.power[i])==0 and bool(t.always_hit[i])and int(t.priority[i])==row['priority']
            and np.array_equal(np.asarray(e.self_boosts[i]),vector))
        for field in ('heal_half','rest','haze','substitute','protect','endure','heal_bell','weather_heal','pain_split','belly_drum','hit_spa_drop','status'):
            good &= not bool(np.any(np.asarray(getattr(e,field)[i])!=0))
        if not good:raise ValueError('canonical original self-boost row differs: '+name)
        lengths[i]=len(ss);stats[i,:len(ss)]=ss;deltas[i,:len(ss)]=dd
    payload=dict(schema='pokezero.literal-ordinary-selfboost.v1',descriptor=descriptor,stats=stats.tolist(),deltas=deltas.tolist(),lengths=lengths.tolist(),engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(e._fields,e)})
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(stats),j.asarray(deltas),j.asarray(lengths),e)

def shape(d):
    if type(d)is not Data:raise ValueError('typed original self-boost data required')
    for v in d[:5]:life._typed(v,(8,),j.uint32)
    m=len(d.installed.protect)
    life._typed(d.stats,(m,2),j.int32);life._typed(d.deltas,(m,2),j.int32);life._typed(d.lengths,(m,),j.int32)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.self_boost;shape(d)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):life._typed(getattr(tape,name),(8,),j.uint32)
    same=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):same &= j.all(getattr(d,name)==getattr(tape,name))
    c=tape.execution
    if c is None or c.risks is None:return j.zeros(len(tape.count),bool)
    t=c.risks.tables
    for name,v in (('binding',d.binding),('stats',d.stats),('deltas',d.deltas),('lengths',d.lengths)):
        target=getattr(t,'self_boost_'+name);life._typed(target,v.shape,v.dtype);same &= j.all(target==v)
    return same&j.any(d.binding!=0)

def selected(context,move):
    if context is None or context.data.self_boost is None:return j.zeros_like(move,bool)
    d=context.data.self_boost;shape(d)
    return (move>0)&(move<len(d.lengths))&(d.lengths[j.clip(move,0,len(d.lengths)-1)]>0)

def begin(tape,state,side,move,caller_move,boosts,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.self_boost is None:return tape,None
    b=len(move);rows=j.arange(b)
    for x in (side,move,caller_move):life._typed(x,(b,),j.int32)
    life._typed(boosts,(b,7),j.int32);life._typed(eligible,(b,),j.bool_)
    requested=eligible&selected(context,move);d=context.data.self_boost;good=_bound(tape,context)&(caller_move==move)
    if type(table)is not type(d.installed):raise ValueError('original installed self-boost table required')
    for a,z in zip(table,d.installed,strict=True):
        if a.shape!=z.shape or a.dtype!=z.dtype:raise ValueError('self-boost table geometry differs')
        good &= j.all(a==z)
    safe=j.clip(side,0,1);physical=state.active[rows,safe]
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,safe])&(state.hp[rows,safe,physical]>0)&(tape.cursor.phase==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.SELF_BOOST_INPUT,side,active=requested)
    return tape,Pending(context,ticket,side,physical,move,boosts,requested,good)

def _write(p,new,secondary):
    b=len(p.move);life._typed(p.old,(b,7),j.int32);life._typed(new,(b,7),j.int32);life._typed(secondary,(b,),j.bool_)
    d=p.context.data.self_boost;move=j.clip(p.move,0,len(d.lengths)-1);expected=j.zeros_like(p.old);rows=j.arange(b)
    for n in range(2):
        use=n<d.lengths[move];stat=j.clip(d.stats[move,n],0,6)
        expected=expected.at[rows,stat].add(j.where(use,d.deltas[move,n],0))
    return j.all((p.old>=-6)&(p.old<=6)&(new>=-6)&(new<=6)&(new==p.old+expected),axis=1)

def _span(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);at=j.broadcast_to(j.asarray(index,j.int32),(b,));raw=tape.words[rows,j.clip(at,0,tape.words.shape[1]-1)]
    if c is None or c.risks is None:return raw,j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger
    m=len(t.selected_risk)
    for name,sh,dtype in (('binding',(8,),j.uint32),('stats',(m,2),j.int32),('deltas',(m,2),j.int32),('lengths',(m,),j.int32)):life._typed(getattr(t,'self_boost_'+name),sh,dtype)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.self_boost_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(t,name)==getattr(tape,name))
    move=raw[:,3];safe=j.clip(move,0,m-1);length=t.self_boost_lengths[safe];first_at=at-raw[:,8];first=j.clip(first_at,0,tape.words.shape[1]-1)
    claim=j.where(stage==1,ledger.count,ledger.claimed[rows,first]);slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good &= (move>0)&(move<m)&((length==1)|(length==2))&(raw[:,9]==length)&(raw[:,8]>=0)&(raw[:,8]<length)
    good &= (first_at>0)&(first_at+length<=tape.count)&(claim>0)&(claim<=ledger.count)
    good &= (ledger.site[rows,slot]==int(ex.Site.SELF_BOOST_INPUT))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==first_at)
    good &= j.where(stage==1,tape.count==first_at+length,ledger.end[rows,slot]==first_at+length)
    for n in range(2):
        pos=j.clip(first_at+n,0,tape.words.shape[1]-1);word=tape.words[rows,pos];used=n<length
        valid=(word[:,0]==SELF_BOOST_INPUT)&j.all(word[:,1:4]==raw[:,1:4],axis=1)&(word[:,4]==t.self_boost_stats[safe,n])&(word[:,5]>=-6)&(word[:,6]<=6)&(word[:,6]==word[:,5]+t.self_boost_deltas[safe,n])&(t.self_boost_deltas[safe,n]>0)&((word[:,7]==0)|(word[:,7]==1))&(word[:,7]==raw[:,7])&(word[:,8]==n)&(word[:,9]==length)
        valid &= (tape.event_turn[rows,pos]==ledger.turn[rows,slot])&(tape.event_phase[rows,pos]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,pos]==ledger.generation[rows,slot],axis=1)
        if stage==2:valid &= (ledger.claimed[rows,pos]==claim)&j.all(ledger.words[rows,pos]==word,axis=1)&(ledger.event_turn[rows,pos]==tape.event_turn[rows,pos])&(ledger.event_phase[rows,pos]==tape.event_phase[rows,pos])&j.all(ledger.event_generation[rows,pos]==tape.event_generation[rows,pos],axis=1)
        good &= ~used|valid
    prefix=j.clip(first_at-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prefix]
    good &= (raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(decl[:,0]==public.MOVE)&j.all(decl[:,1:3]==raw[:,1:3],axis=1)&(decl[:,3]==1-raw[:,1])&(decl[:,5]==move)&j.all(decl[:,6:]==0,axis=1)
    good &= (tape.event_turn[rows,prefix]==ledger.turn[rows,slot])&(tape.event_phase[rows,prefix]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,prefix]==ledger.generation[rows,slot],axis=1)
    if stage==1:good &= (raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(decl[:,4]==tape.cursor.physical_active[rows,j.clip(1-raw[:,1],0,1)])
    return raw,good

def owned(tape,index):return _span(tape,index)[1]

def finish(tape,p,new,secondary):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);d=p.context.data.self_boost;move=j.clip(p.move,0,len(d.lengths)-1);length=d.lengths[move]
    good=p.valid&_bound(tape,p.context)&_write(p,new,secondary);room=(tape.count>=0)&(tape.count<=tape.words.shape[1]-length)
    overflow=p.requested&good&~room
    for n in range(2):
        pos=j.clip(tape.count+n,0,tape.words.shape[1]-1);good &= (n>=length)|j.all(tape.switch_details[rows,pos]==0,axis=1)
    use=p.requested&good&room
    for n in range(2):
        stat=j.clip(d.stats[move,n],0,6);tape=public._append(tape,SELF_BOOST_INPUT,p.side,p.physical,p.move,d.stats[move,n],(p.old[rows,stat],new[rows,stat],secondary.astype(j.int32),j.full(b,n,j.int32),length),use&(n<length))
    _,valid=_span(tape,tape.count-length,stage=1);slot,same=ex._ticket(tape.execution.ledger,p.ticket);valid &= good&room&same
    c=ex._context(tape);ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~valid,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&valid,tables=c.tables)
    tape=ex._put_context(tape,ledger,c);tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~valid,256,0)|j.where(overflow,public.OVERFLOW,0))
    return tape,Captured(p,new,secondary)

def captured_matches(tape,c):
    if c is None:return j.zeros(len(tape.count),bool)
    if type(c)is not Captured or type(c.pending)is not Pending:raise ValueError('typed SAME self-boost capture required')
    from . import engine_execution_completeness as ex
    p=c.pending;b=len(tape.count);rows=j.arange(b);slot,same=ex._ticket(tape.execution.ledger,p.ticket);ledger=tape.execution.ledger;start=ledger.start[rows,slot]
    raw,good=_span(tape,start);good &= same&p.requested&p.valid&_bound(tape,p.context)&_write(p,c.new,c.secondary)
    for col,v in ((1,p.side),(2,p.physical),(3,p.move),(7,c.secondary.astype(j.int32))):good &= raw[:,col]==v
    d=p.context.data.self_boost;move=j.clip(p.move,0,len(d.lengths)-1)
    for n in range(2):
        word=tape.words[rows,j.clip(start+n,0,tape.words.shape[1]-1)];stat=j.clip(d.stats[move,n],0,6)
        good &= (n>=d.lengths[move])|((word[:,5]==p.old[rows,stat])&(word[:,6]==c.new[rows,stat]))
    good &= (tape.count==start+d.lengths[move])&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    return good&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])
