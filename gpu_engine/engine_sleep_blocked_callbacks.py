"""Literal ordinary Gen3 sleep BeforeMove block, RAW47/Site33.

The pending receipt surrounds the actual before_move call and original status/
counter assignment. Rest/SleepTalk/skipped-time/mixed blockers remain UNKNOWN.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import mechanics as m,engine_public_events as public,engine_lifecycle_callbacks as life

SLEEP_BLOCKED=47
SOURCE_SHA='69a6ba02bb56466f43b8fd6422165dff5caca176790b2495be2a7a0e5793e672'
class Data(NamedTuple):
    binding:object
    engine_binding:object
    source_binding:object
    move_binding:object
    producer_binding:object
    moves:object
    installed:object
class Pending(NamedTuple):
    context:object
    ticket:object
    side:object
    physical:object
    move:object
    counter:object
    hp:object
    early_bird:object
    requested:object
    valid:object
class Captured(NamedTuple):
    pending:object
    raw:object

def compile_data(catalog,metadata,rules):
    from . import engine_request_source as request
    from .engine_private_team import engine_metadata_binding
    origin=request._rules_origin(rules)
    if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original sleep-block metadata/rules required')
    script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),c=d.conditions.get('slp');
const rows=meta.vocabulary.moves.map(id=>{const m=d.moves.get(id);return {id:m.id,exists:m.exists,gen:m.gen,sleepUsable:!!m.sleepUsable}});
process.stdout.write(JSON.stringify({rows,sha:crypto.createHash('sha256').update(c.onBeforeMove.toString()).digest('hex'),priority:c.onBeforeMovePriority,source_tree:verified.source_tree,compiled_tree:verified.compiled_tree}));"""
    descriptor=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
    if descriptor['sha']!=SOURCE_SHA or descriptor['priority']!=10 or json.dumps(descriptor['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(descriptor['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original Gen3 sleep callback changed')
    installed=catalog.engine_table.sleep_talk
    if installed is None:return None
    count=len(catalog.engine_table.power);names=metadata['vocabulary']['moves'];abilities=metadata['vocabulary']['abilities']
    if len(installed.move)!=count or not np.array_equal(installed.move,np.array([name=='sleeptalk'for name in names]))or not np.array_equal(installed.early_bird,np.array([name=='earlybird'for name in abilities])):raise ValueError('original installed sleep flags required')
    moves=np.array([bool(i and row['exists']and row['gen']<=3 and row['id']==names[i]and not row['sleepUsable']and catalog.ids[i]>0)for i,row in enumerate(descriptor['rows'])])
    payload=dict(schema='pokezero.literal-ordinary-sleep-block.v1',descriptor=descriptor,engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,candidates=moves.tolist(),installed={k:np.asarray(v).tolist()for k,v in zip(installed._fields,installed)})
    words=lambda x:j.asarray(x,j.uint32)
    return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(moves),installed)

def shape(d):
    if type(d)is not Data:raise ValueError('typed original sleep-block capability required')
    for x in d[:5]:life._typed(x,(8,),j.uint32)
    life._typed(d.moves,(len(d.installed.move),),j.bool_)

def _bound(tape,context):
    from . import engine_direct_hit_controls as direct
    d=context.data.sleep_blocked;shape(d);good=direct._registered_bound(tape,context)
    c=tape.execution
    if c is None or c.risks is None:return j.zeros(len(tape.count),bool)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);good &= j.all(getattr(d,name)==getattr(tape,name))
    t=c.risks.tables;life._typed(t.sleep_blocked_binding,(8,),j.uint32);life._typed(t.sleep_blocked_moves,d.moves.shape,j.bool_)
    return good&life._valid(tape.cursor)&tape.cursor.initialized&(tape.turn==tape.cursor.turn)&(tape.cursor.phase==0)&j.any(d.binding!=0)&j.all(t.sleep_blocked_binding==d.binding)&j.all(t.sleep_blocked_moves==d.moves)

def begin(tape,state,side,move,context,table,*,eligible,sleep_usable,early_bird,loafing):
    from . import engine_execution_completeness as ex
    if context is None or context.data.sleep_blocked is None:return tape,None
    b=len(move);rows=j.arange(b)
    for x in (side,move):life._typed(x,(b,),j.int32)
    for x in (eligible,sleep_usable,early_bird,loafing):life._typed(x,(b,),j.bool_)
    d=context.data.sleep_blocked;good=_bound(tape,context)
    if type(table)is not type(d.installed):raise ValueError('original installed sleep table required')
    for x,y in zip(table,d.installed,strict=True):
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('sleep table geometry differs')
        good &= j.all(x==y)
    seat=j.clip(side,0,1);physical=state.active[rows,seat];at=(rows,seat,physical);status=state.status[at];counter=state.status_counter[at];hp=state.hp[at];ability=state.ability[at]
    move_ok=(move>0)&(move<len(d.moves))&d.moves[j.clip(move,0,len(d.moves)-1)]
    ordinary=(status==m.SLP)&(counter>=2)&(counter<=5)&(counter>1+early_bird.astype(j.int32))&~sleep_usable&~state.rest_origin[at]&(state.sleep_skipped[at]==0)
    ordinary &= ~loafing&(state.confusion_counter[at]==0)&(state.volatile[at]==0)
    good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[rows,seat])&(hp>0)&(ability>=0)&(ability<len(d.installed.early_bird))&(early_bird==d.installed.early_bird[j.clip(ability,0,len(d.installed.early_bird)-1)])
    requested=eligible&move_ok&ordinary;tape,ticket=ex.open_attempt(tape,ex.Site.SLEEP_BLOCKED,side,active=requested)
    from . import engine_outcome_diagnostics as diagnostics
    tape=diagnostics.record_sleep_guard_on_tape(tape,state.rest_origin[at],state.sleep_skipped[at],
        state.volatile[at],requested,good,move_ok,eligible=eligible)
    return tape,Pending(context,ticket,side,physical,move,counter,hp,early_bird,requested,good)

def valid_raw(raw):
    return(raw[:,0]==SLEEP_BLOCKED)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]>0)&(raw[:,4]>=2)&(raw[:,4]<=5)&(raw[:,5]>0)&(raw[:,5]==raw[:,4]-1-raw[:,6])&((raw[:,6]==0)|(raw[:,6]==1))&j.all(raw[:,7:9]==0,axis=1)&(raw[:,9]==m.SLP)

def owned(tape,index,*,stage=2):
    from . import engine_execution_completeness as ex
    c=ex._context(tape);b=len(tape.count);rows=j.arange(b);index=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(index,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
    if c is None or c.risks is None:return j.zeros(b,bool)
    t=c.risks.tables;ledger=c.ledger;life._typed(t.sleep_blocked_binding,(8,),j.uint32);life._typed(t.sleep_blocked_moves,(len(t.selected_risk),),j.bool_)
    claim=ledger.count if stage==1 else ledger.claimed[rows,at];slot=j.clip(claim-1,0,ledger.site.shape[1]-1)
    good=ex._validate(ledger,c.tables)&ex._tape(ledger,tape,c.tables)&ledger.opened&(ledger.frame>0)&j.any(t.sleep_blocked_binding!=0)
    for name in ('engine_binding','source_binding','move_binding','producer_binding'):
        life._typed(getattr(tape,name),(8,),j.uint32);life._typed(getattr(t,name),(8,),j.uint32);good &= j.all(getattr(tape,name)==getattr(t,name))
    good &= valid_raw(raw)&(index>=0)&(index<tape.count)&(raw[:,3]<len(t.sleep_blocked_moves))&t.sleep_blocked_moves[j.clip(raw[:,3],0,len(t.sleep_blocked_moves)-1)]&(claim>0)&(claim<=ledger.count)
    good &= (ledger.site[rows,slot]==int(ex.Site.SLEEP_BLOCKED))&(ledger.side[rows,slot]==raw[:,1])&(ledger.stage[rows,slot]==stage)&(ledger.start[rows,slot]==index)&(ledger.no_output_kind[rows,slot]==0)&j.all(tape.switch_details[rows,at]==0,axis=1)
    good &= j.where(stage==1,tape.count==index+1,ledger.end[rows,slot]==index+1)
    good &= (tape.event_turn[rows,at]==ledger.turn[rows,slot])&(tape.event_phase[rows,at]==ledger.phase[rows,slot])&j.all(tape.event_generation[rows,at]==ledger.generation[rows,slot],axis=1)
    if stage==1:good &= (raw[:,2]==tape.cursor.physical_active[rows,j.clip(raw[:,1],0,1)])&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
    else:good &= (ledger.claimed[rows,at]==claim)&j.all(ledger.words[rows,at]==raw,axis=1)&(ledger.event_turn[rows,at]==tape.event_turn[rows,at])&(ledger.event_phase[rows,at]==tape.event_phase[rows,at])&j.all(ledger.event_generation[rows,at]==tape.event_generation[rows,at],axis=1)
    return good

def finish(tape,p,state,before):
    if p is None:return tape,None
    from . import engine_execution_completeness as ex
    if type(before)is not m.BeforeMove:raise ValueError('literal original BeforeMove required')
    b=len(tape.count);rows=j.arange(b);at=(rows,j.clip(p.side,0,1),p.physical)
    for x in (before.status,before.counter,before.confusion_counter):life._typed(x,(b,),j.int32)
    for x in (before.can_act,before.confusion_self_hit):life._typed(x,(b,),j.bool_)
    raw=j.stack((j.full(b,SLEEP_BLOCKED,j.int32),p.side,p.physical,p.move,p.counter,before.counter,p.early_bird.astype(j.int32),j.zeros(b,j.int32),j.zeros(b,j.int32),before.status),axis=1)
    valid=p.valid&_bound(tape,p.context)&valid_raw(raw)&~before.can_act&~before.confusion_self_hit&(before.confusion_counter==0)
    valid &= (state.active[rows,j.clip(p.side,0,1)]==p.physical)&(state.status[at]==before.status)&(state.status_counter[at]==before.counter)&(state.hp[at]==p.hp)&(state.sleep_skipped[at]==0)&~state.rest_origin[at]
    room=(tape.count>=0)&(tape.count<tape.words.shape[1]);index=j.clip(tape.count,0,tape.words.shape[1]-1);valid &= j.all(tape.switch_details[rows,index]==0,axis=1)
    tape=public._append(tape,SLEEP_BLOCKED,p.side,p.physical,p.move,p.counter,tuple(raw[:,i]for i in range(5,10)),p.requested&valid&room)
    good=owned(tape,j.maximum(tape.count-1,0),stage=1)&valid&room;c=ex._context(tape)
    ledger=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);ledger=ex.emitted(ledger,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
    tape=tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&valid&~room,public.OVERFLOW,0))
    return tape,Captured(p,raw)

def captured_matches(tape,captured,state,side,before):
    if captured is None:return j.zeros(len(tape.count),bool)
    if type(captured)is not Captured or type(captured.pending)is not Pending:raise ValueError('typed SAME sleep-block capture required')
    from . import engine_execution_completeness as ex
    p=captured.pending;c=ex._context(tape);rows=j.arange(len(tape.count));slot,same=ex._ticket(c.ledger,p.ticket);index=j.maximum(tape.count-1,0);raw=tape.words[rows,index];at=(rows,j.clip(side,0,1),p.physical)
    return (owned(tape,index)&same&p.requested&p.valid&_bound(tape,p.context)&(c.ledger.claimed[rows,index]==slot+1)&(c.ledger.end[rows,slot]==tape.count)
        &(p.side==side)&(p.physical==state.active[rows,j.clip(side,0,1)])&(p.physical==tape.cursor.physical_active[rows,j.clip(side,0,1)])&(state.status[at]==m.SLP)&(state.status_counter[at]==p.counter)&(state.hp[at]==p.hp)
        &(c.ledger.turn[rows,slot]==tape.cursor.turn)&(c.ledger.phase[rows,slot]==tape.cursor.phase)&j.all(c.ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
        &j.all(raw==captured.raw,axis=1)&(raw[:,3]==p.move)&(raw[:,4]==p.counter)&(raw[:,5]==before.counter)&(raw[:,9]==before.status)&~before.can_act&~before.confusion_self_hit)
