"""Same-call RAW51 inputs: full-HP Rest fail, awake Sleep Talk fail, Wish start/tick.

This partial family does not reconstruct successful Rest, called moves, Wish
history/expiry or missing public output. Failed guards retain requested UNKNOWN.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life

SELF_MOVE_INPUT=51
REST_FAIL,SLEEP_FAIL,WISH_START,WISH_TICK=range(4)
SECONDARY=4
SOURCE={'rest':{'onTry':'d4b2d4552ebc64b4d569c934b81c348d0f9aac342f7368047cef17d730577be5','onHit':'05f8e437d8566c60900b0a4999dac0341fcfc18137a2abb9c8716964c0d7a179'},'sleeptalk':{'onTry':'aef2554ca54ab070bcb041558d16e30eeaa63310c528d71d0c7ab6b40d79542d','onHit':'203359a676bb58c1492dfdb7293174560abf06abb26b02cd993ac797df6f9483'},'wish':{'onEnd':'8f43feb1254f761b08f8e40b8d82d5de7874a9ddc7577292284ca23cc1b930df'}}
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    move_ids:object # Rest, Sleep Talk, Wish; zero absent, never key0 permission
    candidates:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    code:object
    hp:object
    maximum:object
    status:object
    counter:object
    sleep_skipped:object
    rest_origin:object
    wish_counter:object
    wish_amount:object
    secondary:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    raw:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original self-move metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),out={};
for(const id of ['rest','sleeptalk','wish']) {const m=d.moves.get(id),hashes={};for(const k of ['onTry','onHit'])if(m[k])hashes[k]=crypto.createHash('sha256').update(m[k].toString()).digest('hex');if(id==='wish')hashes.onEnd=crypto.createHash('sha256').update(m.condition.onEnd.toString()).digest('hex');out[id]={hashes,target:m.target,category:m.category,power:m.basePower,accuracy:m.accuracy,slot:m.slotCondition||'',sleepUsable:!!m.sleepUsable,duration:m.condition?.duration||0,order:m.condition?.onResidualOrder||0};}
process.stdout.write(JSON.stringify({out,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original source tree changed')
    for name,row in descriptor['out'].items():
        if row['hashes']!=SOURCE[name]or(row['target'],row['category'],row['power'],row['accuracy'])!=('self','Status',0,True):raise ValueError('original self-move callback changed')
    if descriptor['out']['wish']['slot']!='Wish'or(descriptor['out']['wish']['duration'],descriptor['out']['wish']['order'])!=(2,7)or not descriptor['out']['sleeptalk']['sleepUsable']:raise ValueError('original self-move timing changed')
    names=metadata['vocabulary']['moves'];table=catalog.engine_table
    installed=(None if table.effects is None else table.effects.rest,None if table.sleep_talk is None else table.sleep_talk.move,None if table.sustain is None else table.sustain.wish)
    ids=[];mask=np.zeros(len(names),bool)
    for name,column in zip(('rest','sleeptalk','wish'),installed):
        wanted=np.array([n==name for n in names]);indices=np.flatnonzero(wanted);index=int(indices[0])if len(indices)==1 else 0
        if column is not None and not np.array_equal(column,wanted):raise ValueError('original installed self-move column required')
        if index and column is not None:
            if int(catalog.ids[index])<=0 or int(catalog.targets[index])!=0 or int(table.power[index])!=0 or not bool(table.always_hit[index]):raise ValueError('canonical original self-move row required')
            mask[index]=True
        else:index=0
        ids.append(index)
    if not mask.any():return None
    payload=dict(schema='pokezero.literal-self-move.partial.v1',descriptor=descriptor,engine=np.asarray(catalog.engine_binding).tolist(),source=np.asarray(catalog.source_binding).tolist(),moves=np.asarray(catalog.move_binding).tolist(),producer=catalog.binding,ids=ids)
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(ids,j.int32),j.asarray(mask))

def shape(d):
    if type(d)is not Data:raise ValueError('typed original self-move data required')
    for x in d[:5]:life._typed(x,(8,),j.uint32)
    life._typed(d.move_ids,(3,),j.int32);life._typed(d.candidates,(len(d.candidates),),j.bool_)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.self_move;shape(d);good=direct._registered_bound(tape,context)
    if tape.execution is None or tape.execution.risks is None:return j.zeros(len(tape.count),bool)
    t=tape.execution.risks.tables
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(d,name)==getattr(tape,name))
    life._typed(t.self_move_binding,(8,),j.uint32);life._typed(t.self_move_ids,(3,),j.int32)
    return good&j.any(d.binding!=0)&j.all(t.self_move_binding==d.binding)&j.all(t.self_move_ids==d.move_ids)&life._valid(tape.cursor)&tape.cursor.initialized

def awake_failure(tape,state,side,caller,context,*,eligible,acts):
    """Only source-bound onTry(false) can remove the original unsupported flag."""
    if context is None or context.data.self_move is None:return j.zeros(len(tape.count),bool)
    b=len(tape.count);rows=j.arange(b);d=context.data.self_move;physical=state.active[rows,side]
    return _bound(tape,context)&eligible&~acts&(caller==d.move_ids[1])&(d.move_ids[1]>0)&(state.status[rows,side,physical]!=5)&(state.hp[rows,side,physical]>0)&(tape.cursor.phase==0)&(physical==tape.cursor.physical_active[rows,side])

def valid_raw(raw,ids):
    code=raw[:,3]&3;sec=(raw[:,3]&SECONDARY)!=0
    common=(raw[:,0]==SELF_MOVE_INPUT)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]>=0)&(raw[:,3]<8)&(raw[:,5]>0)&(raw[:,5]<=raw[:,6])&(raw[:,6]<=1_000_000)&(raw[:,7]>=0)&(raw[:,7]<=6)
    return common&j.where(code==REST_FAIL,(ids[0]>0)&(raw[:,4]==ids[0])&(raw[:,5]==raw[:,6])&(raw[:,7]!=5)&(raw[:,8]==raw[:,9]),j.where(code==SLEEP_FAIL,(ids[1]>0)&(raw[:,4]==ids[1])&(raw[:,7]!=5)&~sec&(raw[:,8]==raw[:,9]),(ids[2]>0)&(raw[:,4]==ids[2])&j.where(code==WISH_START,(raw[:,8]==0)&(raw[:,9]==2),(raw[:,8]==2)&(raw[:,9]==1)&~sec)))

def owned(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None or c.risks is None:return j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger;life._typed(t.self_move_binding,(8,),j.uint32);life._typed(t.self_move_ids,(3,),j.int32)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.self_move_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(t,name))
    claim=ledger.count if stage==1 else ledger.claimed[rows,at];slot=j.clip(claim-1,0,ledger.site.shape[1]-1);tick=(raw[:,3]&3)==WISH_TICK
    good &= valid_raw(raw,t.self_move_ids)&(idx>=0)&(idx<tape.count)&(claim>0)&(claim<=ledger.count)&j.all(tape.switch_details[rows,at]==0,axis=1)
    good &= (ledger.site[rows,slot]==j.where(tick,int(ex.Site.WISH),int(ex.Site.SELF_MOVE_INPUT)))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==idx)&(ledger.no_output_kind[rows,slot]==0)
    good &= j.where(stage==1,tape.count==idx+1,ledger.end[rows,slot]==idx+1)
    good &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    prev=j.clip(idx-1,0,tape.words.shape[1]-1);decl=tape.words[rows,prev]
    declaration=(idx>0)&(decl[:,0]==public.MOVE)&(decl[:,1]==raw[:,1])&(decl[:,2]==raw[:,2])&(decl[:,3]==1-raw[:,1])&(decl[:,5]==raw[:,4])&j.all(decl[:,6:]==0,axis=1)&(tape.event_turn[rows,prev]==ledger.turn[rows,slot])&(tape.event_phase[rows,prev]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,prev]==ledger.generation[rows,slot],axis=1)
    # A silent tick is only the same frame's exact fresh START, not private history.
    indices=j.arange(tape.words.shape[1]);prior=tape.words;claims=ledger.claimed;slots=j.clip(claims-1,0,ledger.site.shape[1]-1)
    start=(indices[None,:]<idx[:,None])&(prior[:,:,0]==SELF_MOVE_INPUT)&((prior[:,:,3]&3)==WISH_START)&(prior[:,:,1]==raw[:,1,None])&(prior[:,:,2]==raw[:,2,None])&(prior[:,:,4]==raw[:,4,None])&(prior[:,:,6]==raw[:,6,None])&(prior[:,:,8]==0)&(prior[:,:,9]==2)
    start &= (claims>0)&(claims<=ledger.count[:,None])&(j.take_along_axis(ledger.site,slots,axis=1)==int(ex.Site.SELF_MOVE_INPUT))&(j.take_along_axis(ledger.stage,slots,axis=1)==2)&(j.take_along_axis(ledger.start,slots,axis=1)==indices[None,:])&(j.take_along_axis(ledger.end,slots,axis=1)==indices[None,:]+1)
    start &= (tape.event_turn==ledger.turn[rows,slot,None])&j.all(tape.event_generation==ledger.generation[rows,slot,None,:],axis=2)&j.all(ledger.words==prior,axis=2)
    start &= valid_raw(prior.reshape((-1,10)),t.self_move_ids).reshape((b,-1))
    start &= (j.take_along_axis(ledger.side,slots,axis=1)==raw[:,1,None])&(j.take_along_axis(ledger.no_output_kind,slots,axis=1)==0)
    start &= (tape.event_turn==j.take_along_axis(ledger.turn,slots,axis=1))&(tape.event_phase==j.take_along_axis(ledger.phase,slots,axis=1))
    start &= j.all(tape.event_generation==j.take_along_axis(ledger.generation,slots[:,:,None],axis=1),axis=2)
    previous=j.clip(indices-1,0,tape.words.shape[1]-1);declarations=prior[:,previous]
    start &= (indices[None,:]>0)&(declarations[:,:,0]==public.MOVE)&(declarations[:,:,1]==prior[:,:,1])&(declarations[:,:,2]==prior[:,:,2])&(declarations[:,:,3]==1-prior[:,:,1])&(declarations[:,:,5]==prior[:,:,4])&j.all(declarations[:,:,6:]==0,axis=2)
    start &= (tape.event_turn[:,previous]==tape.event_turn)&(tape.event_phase[:,previous]==tape.event_phase)&j.all(tape.event_generation[:,previous]==tape.event_generation,axis=2)
    good &= j.where(tick,j.sum(start,axis=1)==1,declaration)
    if stage==1:good &= (raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)&(tick|(decl[:,4]==tape.cursor.physical_active[rows,j.clip(1-raw[:,1],0,1)]))
    else:good &= (ledger.claimed[rows,at]==claim)&j.all(ledger.words[rows,at]==raw,axis=1)&(ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good

def begin(tape,state,side,move,context,*,code,eligible,secondary=None):
    from . import engine_execution_completeness as ex
    if context is None or context.data.self_move is None:return tape,None
    if type(code)is not int or code not in(REST_FAIL,SLEEP_FAIL,WISH_START,WISH_TICK):raise ValueError('static original self-move cutpoint required')
    b=len(tape.count);rows=j.arange(b);side=j.broadcast_to(j.asarray(side,j.int32),(b,));move=j.broadcast_to(j.asarray(move,j.int32),(b,));life._typed(eligible,(b,),j.bool_)
    if secondary is None:secondary=j.zeros(b,bool)
    life._typed(secondary,(b,),j.bool_);d=context.data.self_move;good=_bound(tape,context);seat=j.clip(side,0,1);physical=state.active[rows,seat];at=(rows,seat,physical);hp=state.hp[at];maximum=state.max_hp[at];status=state.status[at];counter=state.wish_counter[rows,seat]
    requested=eligible&((counter>0)if code==WISH_TICK else(move==d.move_ids[min(code,2)])&(move>0))
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,seat])&(hp>0)&(maximum<=1_000_000)&(hp<=maximum)
    good &= ((hp==maximum)&(status!=5))if code==REST_FAIL else(status!=5)if code==SLEEP_FAIL else(counter==0)if code==WISH_START else(counter==2)
    good &= (tape.cursor.phase==0)if code!=WISH_TICK else j.ones(b,bool)
    tape,ticket=ex.open_attempt(tape,ex.Site.WISH if code==WISH_TICK else ex.Site.SELF_MOVE_INPUT,side,active=requested)
    return tape,Pending(context,ticket,side,physical,move,j.full(b,code,j.int32),hp,maximum,status,state.status_counter[at],state.sleep_skipped[at],state.rest_origin[at],counter,state.wish_amount[rows,seat],secondary,requested,good)

def finish(tape,p,state):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    b=len(tape.count);rows=j.arange(b);at=(rows,j.clip(p.side,0,1),p.physical);counter=state.wish_counter[rows,j.clip(p.side,0,1)]
    raw=j.stack((j.full(b,SELF_MOVE_INPUT,j.int32),p.side,p.physical,p.code+SECONDARY*p.secondary.astype(j.int32),p.move,p.hp,p.maximum,p.status,p.wish_counter,counter),axis=1)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw,p.context.data.self_move.move_ids)&(state.active[rows,j.clip(p.side,0,1)]==p.physical)
    for actual,old in((state.hp[at],p.hp),(state.max_hp[at],p.maximum),(state.status[at],p.status),(state.status_counter[at],p.counter),(state.sleep_skipped[at],p.sleep_skipped),(state.rest_origin[at],p.rest_origin)):valid &= actual==old
    amount=state.wish_amount[rows,j.clip(p.side,0,1)];valid &= amount==j.where(p.code==WISH_START,p.maximum//2,p.wish_amount)
    valid &= j.where(p.code==WISH_TICK,p.wish_amount==p.maximum//2,True)
    room=(tape.count>=0)&(tape.count<tape.words.shape[1]);address=j.clip(tape.count,0,tape.words.shape[1]-1);valid &= ~room|j.all(tape.switch_details[rows,address]==0,axis=1)
    tape=public._append(tape,SELF_MOVE_INPUT,p.side,p.physical,raw[:,3],p.move,tuple(raw[:,i]for i in range(5,10)),p.requested&valid&room)
    good=owned(tape,j.maximum(tape.count-1,0),stage=1)&valid&room;c=ex._context(tape);slot,same=ex._ticket(c.ledger,p.ticket);good &= same&(c.ledger.start[rows,slot]==tape.count-1)
    ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&valid&~room,public.OVERFLOW,0))
    return tape,Captured(p,raw)

def captured_matches(tape,c):
    if c is None:return j.zeros(len(tape.count),bool)
    if type(c)is not Captured or type(c.pending)is not Pending:raise ValueError('literal same-call self-move capture required')
    from . import engine_execution_completeness as ex
    p=c.pending;rows=j.arange(len(tape.count));ledger=tape.execution.ledger;slot,same=ex._ticket(ledger,p.ticket);idx=ledger.start[rows,slot];raw=tape.words[rows,j.clip(idx,0,tape.words.shape[1]-1)]
    return owned(tape,idx)&same&p.requested&p.valid&_bound(tape,p.context)&j.all(raw==c.raw,axis=1)&(ledger.end[rows,slot]==tape.count)&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)&(p.physical==tape.cursor.physical_active[rows,j.clip(p.side,0,1)])
