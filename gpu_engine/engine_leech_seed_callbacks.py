"""Original singles Leech Seed assignment and paired residual writes.

RAW49=[49,sourceSide,currentSourcePhysical,targetSide,targetPhysical,
        START0/DRAIN1/HEAL2,move/oldHP,oldVol/newHP,newVol/maxHP,sourceSlot/status].
SourceSlot is p1a/p2a's side, NOT a party index. Previous original public bags
or a fresh SAME-write START establish it; arbitrary private seed bits do not.
No public duration/hidden value, numerical rewrite, generic no-output waiver,
or copy-resistant attestation is supplied by these trusted device primitives.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life, sustain
from .state import LEECH_SEED

LEECH=49
START,DRAIN,HEAL=range(3)
HASHES={'onStart':'f52802c2b776c0bb78b571cbe448748d567959ce264afe08d52f3a2b3b9162eb','onResidual':'8c05994767627467fd6ef8cd8ce3db75384cad370936043ec3f8bf50adc1dc2e','onTryImmunity':'64f745bd193049175756b01a76b9b09d8c011c779f4032febfddc51c20fdbddd'}
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_id:object
    installed:object
class History(NamedTuple):
    binding:object                 # GLOBAL8, including B==8
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    stream_id:object
    turn:object
    generation:object
    physical:object
    seeded:object                 # literal original public presence per side
    source_slot:object            # actual singles source side; -1 unknown
class Pending(NamedTuple):
    context:object
    ticket:object
    source:object
    source_physical:object
    target:object
    target_physical:object
    old_hp:object
    maximum:object
    status:object
    recipient_hp:object
    recipient_max:object
    recipient_status:object
    old_volatile:object
    expected:object
    secondary:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    raw:object
    result:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Leech Seed metadata/rules required')
    names=metadata['vocabulary']['moves'];table=catalog.engine_table.sustain
    if table is None or 'leechseed'not in names:return None
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),m=d.moves.get('leechseed'),hashes={};
for(const[k,fn]of[['onStart',m.condition.onStart],['onResidual',m.condition.onResidual],['onTryImmunity',m.onTryImmunity]])hashes[k]=crypto.createHash('sha256').update(fn.toString()).digest('hex');
process.stdout.write(JSON.stringify({hashes,order:m.condition.onResidualOrder,suborder:m.condition.onResidualSubOrder,id:m.id,category:m.category,target:m.target,power:m.basePower,accuracy:m.accuracy,type:m.type,volatile:m.volatileStatus,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    d=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if d['hashes']!=HASHES or(d['order'],d['suborder'],d['id'],d['category'],d['target'],d['power'],d['accuracy'],d['type'],d['volatile'])!=(10,5,'leechseed','Status','normal',0,90,'Grass','leechseed')or json.dumps(d['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(d['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Gen3 Leech Seed source changed')
    move=names.index('leechseed');t=catalog.engine_table
    if not np.array_equal(table.seed,np.array([n=='leechseed'for n in names]))or int(table.grass_type)!=metadata['vocabulary']['types'].index('grass')or int(t.power[move])!=0 or int(t.accuracy[move])!=90 or bool(t.always_hit[move])or int(t.type[move])!=int(table.grass_type)or int(t.priority[move])!=0 or metadata['moves'][move].get('category')!='Status'or int(catalog.targets[move])!=1 or int(catalog.ids[move])<=0:raise ValueError('original installed seed row required')
    payload=dict(schema='pokezero.literal-single-seed-source-slot.v1',descriptor=d,engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(table._fields,table)})
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32),table)

def shape(data):
    if type(data)is not Data:raise ValueError('typed original Leech Seed data required')
    for x in data[:5]:life._typed(x,(8,),j.uint32)
    life._typed(data.move_id,(),j.int32)
    for x in(data.installed.seed,data.installed.wish,data.installed.spin):life._typed(x,data.installed.seed.shape,j.bool_)
    life._typed(data.installed.grass_type,(),j.int32)

def history_shape(h,b):
    if type(h)is not History:raise ValueError('typed original public seed history required')
    for x in h[:5]:life._typed(x,(8,),j.uint32)
    for x in(h.stream_id,h.turn):life._typed(x,(b,),j.int32)
    for x in(h.generation,h.physical,h.source_slot):life._typed(x,(b,2),j.int32)
    life._typed(h.seeded,(b,2),j.bool_)

def empty_history(cursor,data):
    shape(data);b=len(cursor.stream_id)
    return History(*data[:5],cursor.stream_id,cursor.turn,cursor.generation,cursor.physical_active,j.zeros((b,2),bool),j.full((b,2),-1,j.int32))

def history_from_public(engine,cursor,publics,*,data,active_tables,healthy):
    """SAME original owning public roots/session, never private seed inference.

    Canonical uncopied normal-target seed writes in this singles source always
    set the opposite side's slot. Unsupported transfers already refuse the
    ordinary source path; ambiguous public/root history remains absent here.
    """
    from . import public_active
    b=len(cursor.stream_id);life._typed(healthy,(b,),j.bool_)
    if type(publics)is not tuple or len(publics)!=2:raise ValueError('both owning public views required')
    h=empty_history(cursor,data);presence=[];good=healthy
    for owner,pub in enumerate(publics):
        old=pub.context.active;public_active._shape(old,active_tables,pub.mons);current,coherent=public_active._coherent(old,active_tables,pub.mons)
        good &= coherent&j.all(old.encounter==current,axis=1)&(pub.cursor.stream_id==cursor.stream_id)&(pub.cursor.turn==cursor.turn)
        good &= j.all(pub.cursor.source_binding==data.source_binding,axis=1)&j.all(pub.mons.active==cursor.public_slot,axis=1)&j.all(current==cursor.encounter,axis=1)
        good &= j.all(engine.active==cursor.physical_active,axis=1)
        presence.append(old.volatiles[:,:,public_active.NAMES.index('leechseed')])
    known=(presence[0]==presence[1])&good[:,None];seeded=presence[0]&known
    return h._replace(seeded=seeded,source_slot=j.where(seeded,j.broadcast_to(j.array([1,0],j.int32),(b,2)),-1))

def commit_history(active,before,after):
    if before is None:return after if after is None else after
    b=len(before.stream_id);history_shape(before,b);history_shape(after,b);life._typed(active,(b,),j.bool_)
    return History(*before[:5],*(j.where(active.reshape((b,)+(1,)*(x.ndim-1)),y,x)for x,y in zip(before[5:],after[5:],strict=True)))

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.leech_seed;shape(d);good=direct._registered_bound(tape,context)
    if tape.execution is None or tape.execution.risks is None:return j.zeros(len(tape.count),bool)
    for name in('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(d,name))
    t=tape.execution.risks.tables;life._typed(t.leech_binding,(8,),j.uint32);life._typed(t.leech_move,(),j.int32)
    return good&j.any(d.binding!=0)&j.all(t.leech_binding==d.binding)&(t.leech_move==d.move_id)&life._valid(tape.cursor)&tape.cursor.initialized

def _known(tape,data,target,physical):
    b=len(tape.count);rows=j.arange(b);h=tape.execution.leech_history
    if h is None:return j.zeros(b,bool)
    history_shape(h,b);seat=j.clip(target,0,1)
    good=j.all(h.binding==data.binding)&(h.stream_id==tape.stream_id)&(h.turn==tape.cursor.turn)&h.seeded[rows,seat]&(h.source_slot[rows,seat]==1-target)
    good &= (h.physical[rows,seat]==physical)&(h.generation[rows,seat]==tape.cursor.generation[rows,seat])
    for name in('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(h,name)==getattr(tape,name))
    return good

def _pending(tape,state,source,target,context,site,requested,valid,expected,secondary=None):
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);ss=j.clip(source,0,1);ts=j.clip(target,0,1);sp=state.active[rows,ss];tp=state.active[rows,ts];at=(rows,ts,tp);src=(rows,ss,sp)
    if secondary is None:secondary=j.zeros(b,bool)
    life._typed(secondary,(b,),j.bool_)
    valid &= (source>=0)&(source<2)&(target==1-source)&(state.hp[src]>0)&(sp==tape.cursor.physical_active[rows,ss])&(tp==tape.cursor.physical_active[rows,ts])
    tape,ticket=ex.open_attempt(tape,site,j.where(int(site)==35,source,j.where(int(site)==6,target,source)),active=requested)
    return tape,Pending(context,ticket,source,sp,target,tp,state.hp[at],state.max_hp[at],state.status[at],state.hp[src],state.max_hp[src],state.status[src],state.volatile[at],expected,secondary,requested,valid)

def begin_seed(tape,state,side,move,context,table,*,eligible,secondary):
    from . import engine_execution_completeness as ex
    if context is None or context.data.leech_seed is None:return tape,None
    b=len(tape.count);rows=j.arange(b);life._typed(side,(b,),j.int32);life._typed(move,(b,),j.int32);life._typed(eligible,(b,),j.bool_);d=context.data.leech_seed;good=_bound(tape,context)
    if type(table)is not type(d.installed):raise ValueError('original installed sustain table required')
    for x,y in zip(table,d.installed,strict=True):
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('seed table geometry differs')
        good &= j.all(x==y)
    target=1-side;ts=j.clip(target,0,1);tp=state.active[rows,ts];at=(rows,ts,tp)
    good &= (tape.cursor.phase==0)&(state.hp[at]>0)&(state.substitute_hp[at]==0)&~j.any(state.types[at]==table.grass_type,axis=1)&((state.volatile[at]&j.uint32(LEECH_SEED))==0)&(state.volatile[at]<j.uint32(2**31))
    requested=eligible&(move==d.move_id)&(d.move_id>0)
    return _pending(tape,state,side,target,context,ex.Site.LEECH_SEED,requested,good,move,secondary)

def begin_drain(tape,state,target,context,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.leech_seed is None:return tape,None
    b=len(tape.count);rows=j.arange(b);life._typed(target,(b,),j.int32);life._typed(eligible,(b,),j.bool_);ts=j.clip(target,0,1);tp=state.active[rows,ts];ss=j.clip(1-target,0,1);sp=state.active[rows,ss]
    requested=eligible&(state.hp[rows,ts,tp]>0)&((state.volatile[rows,ts,tp]&j.uint32(LEECH_SEED))!=0)
    good=_bound(tape,context)&_known(tape,context.data.leech_seed,target,tp)&(state.hp[rows,ss,sp]>0)
    return _pending(tape,state,1-target,target,context,ex.Site.LEECH_DRAIN,requested,good,j.zeros(b,j.int32))

def valid_raw(raw,move):
    start=raw[:,5]==START;drain=raw[:,5]==DRAIN;heal=raw[:,5]==HEAL
    address=(raw[:,0]==LEECH)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,3]==1-raw[:,1])&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,4]>=0)&(raw[:,4]<6)
    return address&((start&(raw[:,6]==move)&(move>0)&(raw[:,7]>=0)&((raw[:,7]&LEECH_SEED)==0)&(raw[:,8]==(raw[:,7]|LEECH_SEED))&(raw[:,9]>=0)&(raw[:,9]<4)&((raw[:,9]&1)==raw[:,1]))|((drain|heal)&(raw[:,6]>0)&(raw[:,6]<=raw[:,8])&(raw[:,8]<=1_000_000)&(raw[:,9]>=0)&(raw[:,9]<=6)&j.where(drain,(raw[:,7]>0)&(raw[:,7]<raw[:,6]),(raw[:,7]>=raw[:,6])&(raw[:,7]<=raw[:,8])&((raw[:,7]>raw[:,6])|(raw[:,6]==raw[:,8])))))

def owned(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None or c.risks is None:return j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger;life._typed(t.leech_binding,(8,),j.uint32);life._typed(t.leech_move,(),j.int32)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.leech_binding!=0)&(t.leech_move>0)
    for name in('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(t,name))
    claim=ledger.count if stage==1 else ledger.claimed[rows,at];slot=j.clip(claim-1,0,ledger.site.shape[1]-1);start=raw[:,5]==START;drain=raw[:,5]==DRAIN;heal=raw[:,5]==HEAL
    site=j.where(start,int(ex.Site.LEECH_SEED),j.where(drain,int(ex.Site.LEECH_DRAIN),int(ex.Site.LEECH_HEAL)));side=j.where(drain,raw[:,3],raw[:,1])
    good &= valid_raw(raw,t.leech_move)&(idx>=0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&j.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (ledger.site[rows,slot]==site)&(ledger.side[rows,slot]==side)&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==idx)&(ledger.no_output_kind[rows,slot]==0)&j.where(stage==1,tape.count==idx+1,ledger.end[rows,slot]==idx+1)
    good &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    prior=j.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prior]
    prefix=(idx>0)&(decl[:,0]==public.MOVE)&j.all(decl[:,1:5]==raw[:,1:5],axis=1)&(decl[:,5]==t.leech_move)&j.all(decl[:,6:]==0,axis=1)
    prefix &= (tape.event_turn[rows,prior]==ledger.turn[rows,slot])&(tape.event_phase[rows,prior]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,prior]==ledger.generation[rows,slot],axis=1)
    good &= ~start|prefix
    # HEAL belongs to the immediately preceding exact captured DRAIN, not any
    # generic HP line or a no-output permission flag.
    pc=ledger.claimed[rows,prior];ps=j.clip(pc-1,0,ledger.site.shape[1]-1)
    pair=(idx>0)&valid_raw(decl,t.leech_move)&(decl[:,0]==LEECH)&(decl[:,5]==DRAIN)&j.all(decl[:,1:5]==raw[:,1:5],axis=1)&(pc>0)&(pc<=ledger.count)&(ledger.site[rows,ps]==int(ex.Site.LEECH_DRAIN))&(ledger.stage[rows,ps]==2)&(ledger.start[rows,ps]==idx-1)&(ledger.end[rows,ps]==idx)
    pair &= j.all(ledger.words[rows,prior]==decl,axis=1)&(ledger.event_turn[rows,prior]==tape.event_turn[rows,prior])&(ledger.event_phase[rows,prior]==tape.event_phase[rows,prior])&j.all(ledger.event_generation[rows,prior]==tape.event_generation[rows,prior],axis=1)
    pair &= (tape.event_turn[rows,prior]==ledger.turn[rows,slot])&(tape.event_phase[rows,prior]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,prior]==ledger.generation[rows,slot],axis=1)&(raw[:,7]==j.minimum(raw[:,8],raw[:,6]+decl[:,6]-decl[:,7]))
    good &= ~heal|pair
    if stage==1:good &= (raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(raw[:,4]==tape.cursor.physical_active[rows,j.clip(raw[:,3],0,1)])&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    else:good &= (ledger.claimed[rows,at]==claim)&j.all(ledger.words[rows,at]==raw,axis=1)&(ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good

def _emit(tape,p,raw,result,valid):
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);room=(tape.count>=0)&(tape.count<tape.words.shape[1]);at=j.clip(tape.count,0,tape.words.shape[1]-1);valid &= ~room|j.all(tape.switch_details[rows,at]==0,axis=1)
    tape=public._append(tape,LEECH,p.source,p.source_physical,p.target,p.target_physical,tuple(raw[:,i]for i in range(5,10)),p.requested&valid&room)
    good=owned(tape,j.maximum(tape.count-1,0),stage=1)&valid&room;c=ex._context(tape);slot,same=ex._ticket(c.ledger,p.ticket);good &= same&(c.ledger.start[rows,slot]==tape.count-1)
    ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&valid&~room,public.OVERFLOW,0))
    return tape,Captured(p,raw,result),good

def finish_seed(tape,p,state):
    if p is None:return tape,None
    b=len(tape.count);rows=j.arange(b);at=(rows,j.clip(p.target,0,1),p.target_physical);new=state.volatile[at]
    raw=j.stack((j.full(b,LEECH,j.int32),p.source,p.source_physical,p.target,p.target_physical,j.full(b,START,j.int32),p.expected,p.old_volatile.astype(j.int32),new.astype(j.int32),p.source+2*p.secondary.astype(j.int32)),axis=1)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw,p.context.data.leech_seed.move_id)&(state.hp[at]==p.old_hp)&(state.status[at]==p.status)&(state.active[rows,j.clip(p.target,0,1)]==p.target_physical)&(state.active[rows,j.clip(p.source,0,1)]==p.source_physical)
    valid &= (state.hp[rows,j.clip(p.source,0,1),p.source_physical]==p.recipient_hp)&(state.max_hp[at]==p.maximum)&(state.max_hp[rows,j.clip(p.source,0,1),p.source_physical]==p.recipient_max)
    tape,captured,good=_emit(tape,p,raw,None,valid);context=tape.execution;h=context.leech_history
    if h is None:h=empty_history(tape.cursor,p.context.data.leech_seed)
    history_shape(h,b);target=j.clip(p.target,0,1)
    h=h._replace(seeded=h.seeded.at[rows,target].set(j.where(p.requested&good,True,h.seeded[rows,target])),source_slot=h.source_slot.at[rows,target].set(j.where(p.requested&good,p.source,h.source_slot[rows,target])),physical=h.physical.at[rows,target].set(j.where(p.requested&good,p.target_physical,h.physical[rows,target])),generation=h.generation.at[rows,target].set(j.where(p.requested&good,tape.cursor.generation[rows,target],h.generation[rows,target])))
    return tape._replace(execution=context._replace(leech_history=h)),captured

def finish_drain(tape,p,state,result):
    if p is None:return tape,None
    if type(result)is not sustain.LeechResult:raise ValueError('literal original LeechResult required')
    b=len(tape.count);rows=j.arange(b);at=(rows,j.clip(p.target,0,1),p.target_physical);src=(rows,j.clip(p.source,0,1),p.source_physical)
    for x in result:life._typed(x,(b,),j.int32)
    raw=j.stack((j.full(b,LEECH,j.int32),p.source,p.source_physical,p.target,p.target_physical,j.full(b,DRAIN,j.int32),p.old_hp,state.hp[at],p.maximum,p.status),axis=1)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw,p.context.data.leech_seed.move_id)&(state.hp[at]==result.hp)&(state.hp[src]==p.recipient_hp)&(state.status[at]==p.status)&(state.status[src]==p.recipient_status)&(state.volatile[at]==p.old_volatile)
    valid &= (state.active[rows,j.clip(p.target,0,1)]==p.target_physical)&(state.active[rows,j.clip(p.source,0,1)]==p.source_physical)&(state.max_hp[at]==p.maximum)&(state.max_hp[src]==p.recipient_max)
    tape,capture,_=_emit(tape,p,raw,result,valid);return tape,capture

def begin_heal(tape,state,capture):
    if capture is None:return tape,None
    from . import engine_execution_completeness as ex
    if type(capture)is not Captured or type(capture.pending)is not Pending:raise ValueError('same-call drain capture required')
    p=capture.pending;b=len(tape.count);requested=p.requested&(p.recipient_hp>0);good=captured_matches(tape,capture)&(capture.raw[:,5]==DRAIN)
    return _pending(tape,state,p.source,p.target,p.context,ex.Site.LEECH_HEAL,requested,good,capture.result.recipient_hp)

def finish_heal(tape,p,state):
    if p is None:return tape,None
    b=len(tape.count);rows=j.arange(b);src=(rows,j.clip(p.source,0,1),p.source_physical);at=(rows,j.clip(p.target,0,1),p.target_physical)
    raw=j.stack((j.full(b,LEECH,j.int32),p.source,p.source_physical,p.target,p.target_physical,j.full(b,HEAL,j.int32),p.recipient_hp,state.hp[src],p.recipient_max,p.recipient_status),axis=1)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw,p.context.data.leech_seed.move_id)&(state.hp[src]==p.expected)&(state.hp[at]==p.old_hp)&(state.status[src]==p.recipient_status)&(state.status[at]==p.status)
    valid &= (state.active[rows,j.clip(p.target,0,1)]==p.target_physical)&(state.active[rows,j.clip(p.source,0,1)]==p.source_physical)&(state.max_hp[at]==p.maximum)&(state.max_hp[src]==p.recipient_max)
    tape,capture,_=_emit(tape,p,raw,None,valid);return tape,capture

def captured_matches(tape,capture):
    if capture is None:return j.zeros(len(tape.count),bool)
    if type(capture)is not Captured or type(capture.pending)is not Pending:raise ValueError('typed same-write seed capture required')
    from . import engine_execution_completeness as ex
    p=capture.pending;rows=j.arange(len(tape.count));ledger=tape.execution.ledger;slot,same=ex._ticket(ledger,p.ticket);idx=ledger.start[rows,slot];raw=tape.words[rows,j.clip(idx,0,tape.words.shape[1]-1)]
    return owned(tape,idx)&same&p.requested&p.valid&_bound(tape,p.context)&j.all(raw==capture.raw,axis=1)&(p.source_physical==tape.cursor.physical_active[rows,j.clip(p.source,0,1)])&(p.target_physical==tape.cursor.physical_active[rows,j.clip(p.target,0,1)])&(ledger.end[rows,slot]==tape.count)&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
