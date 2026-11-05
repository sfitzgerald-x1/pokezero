"""Original Protect-condition setter and target interception, no physics/RNG.

Private same-call holders never establish origin themselves. The owning host
checks the original compiler; both FULL consumers must drain these spans.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import engine_lifecycle_callbacks as life,engine_public_events as public

PROTECT_ATTEMPT,PROTECT_BLOCK=39,40
ELIGIBLE,FIRST,NOT_PENDING,FOE_MOVE,SUCCESS,SECONDARY=(1<<i for i in range(6))

class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    candidates:object
    block_candidates:object
    protectable:object
    installed:object

class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    count:object
    before:object
    roll:object
    first:object
    not_pending:object
    foe_move:object
    requested:object
    valid:object

class BlockPending(NamedTuple):
    context:object
    ticket:object
    actor:object
    physical:object
    target:object
    move:object
    protect_index:object
    old_hit:object
    requested:object
    valid:object

class CapturedBlock(NamedTuple):
    pending:object
    index:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding):raise ValueError('original Protect metadata required')
    if not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Protect rules required')
    script=r"""
const fs=require('fs'),path=require('path'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const checked=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id),c=d.conditions.get(m.volatileStatus||'');return {
id:m.id,exists:m.exists,gen:m.gen,name:m.name,target:m.target,power:m.basePower,category:m.category,priority:m.priority,
volatile:m.volatileStatus||'',stall:m.stallingMove===true,protectable:!!m.flags.protect,
callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined).sort(),prepare:String(m.onPrepareHit),hit:String(m.onHit),
condition:{duration:c.duration,start:String(c.onStart),tryHit:String(c.onTryHit)},
mixed:!!(m.self||m.status||m.boosts||m.secondary||m.secondaries||m.sideCondition||m.slotCondition||m.weather||m.terrain||m.pseudoWeather||m.recoil||m.drain||m.heal||m.selfdestruct||m.forceSwitch||m.selfSwitch)}});
process.stdout.write(JSON.stringify({rows,source_tree:checked.source_tree,compiled_tree:checked.compiled_tree}));"""
    result=subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30)
    descriptor=json.loads(result.stdout)
    if json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Protect source changed')
    t=catalog.engine_table;e=t.effects;n=len(t.power);mask=np.zeros(n,bool)
    if e is None:raise ValueError('original installed Protect effects required')
    for i,row in enumerate(descriptor['rows']):
        mask[i]=bool(i and row['exists']and row['gen']<=3 and row['target']=='self'and row['category']=='Status'
            and row['id']==metadata['vocabulary']['moves'][i]and catalog.targets[i]==0
            and row['power']==0 and row['volatile']=='protect'and row['stall']and not row['mixed']
            and row['callbacks']==['onHit','onPrepareHit']and row['condition']['duration']==1
            and bool(e.protect[i])and not bool(e.endure[i])and int(t.power[i])==0 and int(t.priority[i])==row['priority']and catalog.ids[i]>0)
    # Exact installed protectable columns, not category or move-name inference.
    protectable=np.asarray(e.protectable)
    original_protectable=np.asarray([row['exists']and row['gen']<=3 and row['protectable']for row in descriptor['rows']],bool)
    block=protectable&original_protectable&(np.asarray(catalog.ids)>0)&(np.asarray(catalog.targets)==1)
    if t.self_destruct is not None:block &= ~np.asarray(t.self_destruct.move)
    if t.damage_moves is not None:block &= np.asarray(t.damage_moves.recoil_den)==0
    payload=dict(schema='pokezero.literal-protect.v1',descriptor=descriptor,mask=mask.tolist(),block=block.tolist(),
        engine=catalog.engine_binding,source=catalog.source_binding,producer=catalog.binding,installed={k:np.asarray(v).tolist()for k,v in zip(e._fields,e)})
    words=lambda v:j.asarray(v,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(mask),j.asarray(block),j.asarray(protectable),e)

def shape(data):
    if type(data)is not Data:raise ValueError('original typed Protect data required')
    for v in data[:5]:life._typed(v,(8,),j.uint32)
    for v in data[5:8]:life._typed(v,(len(data.installed.protect),),j.bool_)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    data=context.data.protect;shape(data);good=direct._registered_bound(tape,context)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(data,name)==getattr(tape,name))
    risk=tape.execution.risks
    if risk is None:return j.zeros(len(tape.count),bool)
    good &= j.any(risk.tables.protect_binding!=0)&j.all(risk.tables.protect_binding==data.binding)
    life._typed(risk.tables.protect_candidates,data.candidates.shape,j.bool_)
    life._typed(risk.tables.protect_block_candidates,data.block_candidates.shape,j.bool_)
    good &= j.all(risk.tables.protect_candidates==data.candidates)&j.all(risk.tables.protect_block_candidates==data.block_candidates)
    return good

def selected(context,move):
    if context is None or context.data.protect is None:return j.zeros_like(move,bool)
    d=context.data.protect;shape(d)
    return (move>0)&(move<len(d.candidates))&d.candidates[j.clip(move,0,len(d.candidates)-1)]

def begin(tape,state,side,move,count,roll,first,not_pending,foe_move,context,table,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.protect is None:return tape,None
    d=context.data.protect;requested=eligible&selected(context,move);b=len(move);rows=j.arange(b)
    for v in (eligible,first,not_pending,foe_move):life._typed(v,(b,),j.bool_)
    for v in (side,move,count,roll):life._typed(v,(b,),j.int32)
    good=_bound(tape,context)
    if type(table)is not type(d.installed):raise ValueError('original installed Protect table required')
    for a,z in zip(table,d.installed,strict=True):
        if a.shape!=z.shape or a.dtype!=z.dtype:raise ValueError('installed Protect table shape/dtype differs')
        good &= j.all(a==z)
    physical=state.active[rows,side];before=state.protect_counter[rows,side,physical]
    good &= (state.hp[rows,side,physical]>0)&(before==0)&(tape.cursor.phase==0)
    good &= (count==0)|(count==2)|(count==4)|(count==8)
    good &= (roll>=0)&(roll<8)&(physical==tape.cursor.physical_active[rows,side])
    tape,ticket=ex.open_attempt(tape,ex.Site.PROTECT_ATTEMPT,side,active=requested)
    return tape,Pending(context,ticket,side,physical,move,count,before,roll,first,not_pending,foe_move,requested,good)

def finish(tape,p,state,stall,protected,secondary):
    if p is None:return tape
    from . import engine_execution_completeness as ex
    rows=j.arange(len(p.side));after=state.protect_counter[rows,p.side,p.physical]
    good=p.valid&_bound(tape,p.context)&(protected==stall)
    good &= j.where(stall,p.first&p.not_pending&p.foe_move&(after==j.minimum(8,j.maximum(2,p.count*2))),(after==0)&(~(p.first&p.not_pending&p.foe_move)|(p.count>0)))
    flags=j.ones_like(p.move)*ELIGIBLE
    for bit,value in ((FIRST,p.first),(NOT_PENDING,p.not_pending),(FOE_MOVE,p.foe_move),(SUCCESS,stall),(SECONDARY,secondary)):flags |= value.astype(j.int32)*bit
    tape=public._append(tape,PROTECT_ATTEMPT,p.side,p.physical,p.side,p.physical,(p.move,flags,p.count,after,p.roll),p.requested&good)
    c=ex._context(tape);ledger=ex.emitted(c.ledger,p.ticket,tape,active=p.requested&good,tables=c.tables)
    ledger=ex.unhandled(ledger,p.ticket,active=p.requested&~good,tables=c.tables)
    return ex._put_context(tape,ledger,c)

def _receipt(tape,index,site):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b)
    at=j.broadcast_to(j.asarray(index,j.int32),(b,));safe=j.clip(at,0,tape.words.shape[1]-1)
    raw=tape.words[rows,safe]
    if c is None or c.risks is None:return raw,j.zeros(b,bool)
    ledger=c.ledger;t=c.risks.tables
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32)
    life._typed(t.protect_binding,(8,),j.uint32)
    for mask in (t.protect_candidates,t.protect_block_candidates):life._typed(mask,(len(mask),),j.bool_)
    claim=ledger.claimed[rows,safe];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)
    good &= j.any(t.protect_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(t,name)==getattr(tape,name))
    good &= (at>0)&(at<tape.count)&(claim>0)&(claim<=ledger.count)
    good &= (ledger.site[rows,slot]==int(site))&(ledger.stage[rows,slot]==2)
    good &= (ledger.start[rows,slot]==at)&(ledger.end[rows,slot]==at+1)&j.all(ledger.words[rows,safe]==raw,axis=1)
    good &= (ledger.turn[rows,slot]==tape.event_turn[rows,safe])&(ledger.phase[rows,slot]==tape.event_phase[rows,safe])
    good &= j.all(ledger.generation[rows,slot]==tape.event_generation[rows,safe],axis=1)
    good &= (ledger.event_turn[rows,safe]==tape.event_turn[rows,safe])&(ledger.event_phase[rows,safe]==tape.event_phase[rows,safe])
    good &= j.all(ledger.event_generation[rows,safe]==tape.event_generation[rows,safe],axis=1)
    previous=tape.words[rows,j.maximum(at-1,0)]
    good &= (previous[:,0]==public.MOVE)&j.all(previous[:,1:3]==raw[:,1:3],axis=1)&(previous[:,5]==raw[:,5])&j.all(previous[:,6:]==0,axis=1)
    good &= (tape.event_turn[rows,j.maximum(at-1,0)]==tape.event_turn[rows,safe])&(tape.event_phase[rows,j.maximum(at-1,0)]==tape.event_phase[rows,safe])
    good &= j.all(tape.event_generation[rows,j.maximum(at-1,0)]==tape.event_generation[rows,safe],axis=1)
    good &= (raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)
    good &= ledger.side[rows,slot]==j.where(raw[:,0]==PROTECT_ATTEMPT,raw[:,1],raw[:,3])
    return raw,good

def owned_attempt(tape,index):
    from . import engine_execution_completeness as ex
    raw,good=_receipt(tape,index,ex.Site.PROTECT_ATTEMPT)
    if tape.execution is None or tape.execution.risks is None:return good
    mask=tape.execution.risks.tables.protect_candidates;move=raw[:,5]
    success=(raw[:,6]&SUCCESS)!=0;queue=(raw[:,6]&(FIRST|NOT_PENDING|FOE_MOVE))==(FIRST|NOT_PENDING|FOE_MOVE)
    good &= (raw[:,0]==PROTECT_ATTEMPT)&(raw[:,3]==raw[:,1])&(raw[:,4]==raw[:,2])
    good &= (move>0)&(move<len(mask))&mask[j.clip(move,0,len(mask)-1)]
    good &= ((raw[:,6]&ELIGIBLE)!=0)&((raw[:,6]&~63)==0)
    good &= (raw[:,7]==0)|(raw[:,7]==2)|(raw[:,7]==4)|(raw[:,7]==8)
    good &= (raw[:,9]>=0)&(raw[:,9]<8)
    good &= j.where(success,queue&(raw[:,8]==j.minimum(8,j.maximum(2,raw[:,7]*2))),(raw[:,8]==0)&(~queue|(raw[:,7]>0)))
    return good

def begin_block(tape,state,actor,move,old_hit,actual_protected,protectable,context,*,eligible):
    from . import engine_execution_completeness as ex
    if context is None or context.data.protect is None:return tape,None
    d=context.data.protect;b=len(move);rows=j.arange(b);physical=state.active[rows,actor];target=state.active[rows,1-actor]
    requested=eligible&old_hit&actual_protected&protectable
    positions=j.arange(tape.words.shape[1])[None,:]
    successes=(positions<tape.count[:,None])&(tape.words[:,:,0]==PROTECT_ATTEMPT)&((tape.words[:,:,6]&SUCCESS)!=0)
    successes &= (tape.words[:,:,1]==(1-actor)[:,None])&(tape.words[:,:,2]==target[:,None])
    index=j.max(j.where(successes,positions,-1),axis=1)
    good=_bound(tape,context)&owned_attempt(tape,index)&(index>=0)&old_hit
    good &= (move>0)&(move<len(d.block_candidates))&d.block_candidates[j.clip(move,0,len(d.block_candidates)-1)]
    good &= (state.hp[rows,actor,physical]>0)&(state.hp[rows,1-actor,target]>0)
    good &= (physical==tape.cursor.physical_active[rows,actor])&(target==tape.cursor.physical_active[rows,1-actor])
    good &= (tape.cursor.phase==0)
    tape,ticket=ex.open_attempt(tape,ex.Site.PROTECT_BLOCK,1-actor,active=requested)
    return tape,BlockPending(context,ticket,actor,physical,target,move,index,old_hit,requested,good)

def finish_block(tape,p,new_hit):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    good=p.valid&_bound(tape,p.context)&~new_hit
    start=tape.count
    tape=public._append(tape,PROTECT_BLOCK,p.actor,p.physical,1-p.actor,p.target,
        (p.move,p.protect_index,j.ones_like(p.move),p.old_hit.astype(j.int32),new_hit.astype(j.int32)),p.requested&good)
    c=ex._context(tape);ledger=ex.emitted(c.ledger,p.ticket,tape,active=p.requested&good,tables=c.tables)
    ledger=ex.unhandled(ledger,p.ticket,active=p.requested&~good,tables=c.tables)
    return ex._put_context(tape,ledger,c),CapturedBlock(p,start)

def owned_block(tape,index):
    from . import engine_execution_completeness as ex
    raw,good=_receipt(tape,index,ex.Site.PROTECT_BLOCK)
    if tape.execution is None or tape.execution.risks is None:return good
    mask=tape.execution.risks.tables.protect_block_candidates;move=raw[:,5];prior=raw[:,6]
    rows=j.arange(len(move));at=j.broadcast_to(j.asarray(index,j.int32),move.shape)
    source=tape.words[rows,j.clip(prior,0,tape.words.shape[1]-1)]
    good &= (raw[:,0]==PROTECT_BLOCK)&(raw[:,3]==1-raw[:,1])&(raw[:,4]>=0)&(raw[:,4]<6)
    good &= (move>0)&(move<len(mask))&mask[j.clip(move,0,len(mask)-1)]
    good &= (raw[:,7]==1)&(raw[:,8]==1)&(raw[:,9]==0)&(prior>0)&(prior<at-1)
    good &= owned_attempt(tape,prior)&((source[:,6]&SUCCESS)!=0)&(source[:,1]==raw[:,3])&(source[:,2]==raw[:,4])
    safe=j.clip(at,0,tape.words.shape[1]-1);earlier=j.clip(prior,0,tape.words.shape[1]-1)
    good &= (tape.event_turn[rows,earlier]==tape.event_turn[rows,safe])&(tape.event_phase[rows,earlier]==tape.event_phase[rows,safe])
    good &= j.all(tape.event_generation[rows,earlier]==tape.event_generation[rows,safe],axis=1)
    return good

def captured_block_matches(tape,captured,outcome=None):
    if captured is None:return j.zeros(len(tape.count),bool)
    if type(captured)is not CapturedBlock or type(captured.pending)is not BlockPending:raise ValueError('typed SAME-call Protect interception required')
    from . import engine_execution_completeness as ex
    p=captured.pending;rows=j.arange(len(tape.count));raw=tape.words[rows,j.clip(captured.index,0,tape.words.shape[1]-1)]
    good=owned_block(tape,captured.index)&p.requested&p.valid&_bound(tape,p.context)
    good &= (raw[:,1]==p.actor)&(raw[:,2]==p.physical)&(raw[:,4]==p.target)&(raw[:,5]==p.move)&(raw[:,6]==p.protect_index)
    slot,same=ex._ticket(tape.execution.ledger,p.ticket);good &= same&(tape.execution.ledger.start[rows,slot]==captured.index)
    safe=j.clip(captured.index,0,tape.words.shape[1]-1)
    good &= (tape.cursor.turn==tape.event_turn[rows,safe])&(tape.cursor.phase==tape.event_phase[rows,safe])
    good &= j.all(tape.cursor.generation==tape.event_generation[rows,safe],axis=1)
    good &= (tape.cursor.physical_active[rows,p.actor]==p.physical)&(tape.cursor.physical_active[rows,1-p.actor]==p.target)
    if outcome is not None:good &= outcome.protected&~outcome.hit&~outcome.absorbed&~outcome.enduring&~outcome.explodes&(outcome.recoil==0)
    return good
