"""Ordinary exact PAR/FRZ cant at SAME original BeforeMove, no physics.

RAW56 [56,side,physical,move,status,thawRoll,paralysisRoll,confusionRoll,0,ability].
Competing blockers, defrost/sleep-usable/called moves and unknown source facts
remain UNKNOWN. No MOVE or PP spend is manufactured for a cant event.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax.numpy as j
import numpy as np
from . import mechanics as m,engine_public_events as public,engine_lifecycle_callbacks as life
BEFORE_MOVE_BLOCKED=56
class Data(NamedTuple):
 binding:object
 engine_binding:object
 source_binding:object
 move_binding:object
 producer_binding:object
 moves:object
 abilities:object
 items:object
class Pending(NamedTuple):
 context:object
 ticket:object
 side:object
 physical:object
 move:object
 caller:object
 status:object
 counter:object
 hp:object
 maximum:object
 ability:object
 item:object
 draws:object
 requested:object
 valid:object
class Captured(NamedTuple):
 pending:object
 raw:object

def compile_data(catalog,metadata,rules):
 from . import engine_request_source as request
 from .engine_private_team import engine_metadata_binding
 origin=request._rules_origin(rules)
 if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original PAR/FRZ metadata/rules required')
 script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8')),checked=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const hash=f=>crypto.createHash('sha256').update(f.toString()).digest('hex');
process.stdout.write(JSON.stringify({source_tree:checked.source_tree,compiled_tree:checked.compiled_tree,status:['par','frz'].map(n=>{const c=d.conditions.get(n);return {id:c.id,priority:c.onBeforeMovePriority,callback:hash(c.onBeforeMove)}}),moves:meta.vocabulary.moves.map(n=>{const m=d.moves.get(n);return {id:m.id,exists:m.exists,gen:m.gen,sleepUsable:!!m.sleepUsable,defrost:!!m.flags?.defrost}}),abilities:meta.vocabulary.abilities.map(n=>{const a=d.abilities.get(n);return a.exists&&a.gen<=3&&a.id!=='magicguard'&&a.onBeforeMove===undefined}),items:meta.vocabulary.items.map((n,i)=>{const a=d.items.get(n);return !i||(a.exists&&a.gen<=3&&a.onBeforeMove===undefined)})}));"""
 v=json.loads(subprocess.check_output([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata).encode(),timeout=30))
 if v['status']!=[dict(id='par',priority=1,callback='713407730186869cd1e77e250eec0ff6ceb40b018e884725ebcbff59cbbaf66f'),dict(id='frz',priority=10,callback='9c1e9aa02c50d3dffa13974c5f6fc778b529581804a06ff45b6f9192d9b14117')]:raise ValueError('exact original Gen3 cant callbacks required')
 if json.dumps(v['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(v['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original BeforeMove source drift')
 names=metadata['vocabulary']['moves'];ordinary=np.asarray([bool(i and x['id']==names[i]and x['exists']and x['gen']<=3 and not x['sleepUsable']and catalog.ids[i]>0)for i,x in enumerate(v['moves'])]);moves=np.stack((ordinary,ordinary&~np.asarray([x['defrost']for x in v['moves']])))
 payload=dict(schema='pokezero.before-move-par-frz.block.v1',source=v,moves=moves.tolist(),engine=catalog.engine_binding,source_binding=catalog.source_binding,move_binding=catalog.move_binding,producer=catalog.binding)
 words=lambda x:j.asarray(x,j.uint32)
 return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(moves),j.asarray(v['abilities']),j.asarray(v['items']))
def shape(d):
 if type(d)is not Data:raise ValueError('typed original PAR/FRZ capability required')
 for v in d[:5]:life._typed(v,(8,),j.uint32)
 life._typed(d.moves,(2,d.moves.shape[-1]),j.bool_)
 for v in d[6:]:life._typed(v,(len(v),),j.bool_)
def _bound(tape,context):
 from . import engine_direct_hit_controls as direct
 d=context.data.before_move_blocked;shape(d);good=direct._registered_bound(tape,context);c=tape.execution
 if c is None or c.risks is None:return j.zeros(len(tape.count),bool)
 for n in ('engine_binding','source_binding','move_binding','producer_binding'):
  life._typed(getattr(tape,n),(8,),j.uint32);good &= j.all(getattr(d,n)==getattr(tape,n))
 t=c.risks.tables
 for n,v in [('binding',d.binding),('moves',d.moves),('abilities',d.abilities),('items',d.items)]:
  actual=getattr(t,'before_move_blocked_'+n);life._typed(actual,v.shape,v.dtype);good &= j.all(actual==v)
 return good&life._valid(tape.cursor)&tape.cursor.initialized&(tape.turn==tape.cursor.turn)&(tape.cursor.phase==0)&j.any(d.binding!=0)
def valid_raw(raw):
 status=raw[:,4];rolls=j.all((raw[:,5:8]>=0)&(raw[:,5:8]<100),axis=1)
 return (raw[:,0]==BEFORE_MOVE_BLOCKED)&(raw[:,1]>=0)&(raw[:,1]<2)&(raw[:,2]>=0)&(raw[:,2]<6)&(raw[:,3]>0)&(((status==m.PAR)&(raw[:,6]<25))|((status==m.FRZ)&(raw[:,5]>=20)))&rolls&(raw[:,8]==0)&(raw[:,9]>0)
def begin(tape,state,side,move,caller,draws,context,*,eligible,sleep_usable,loafing):
 from . import engine_execution_completeness as ex
 if context is None or context.data.before_move_blocked is None:return tape,None
 b=len(move);r=j.arange(b);life._typed(draws,(b,3),j.int32)
 for x in (side,move,caller):life._typed(x,(b,),j.int32)
 for x in (eligible,sleep_usable,loafing):life._typed(x,(b,),j.bool_)
 d=context.data.before_move_blocked;good=_bound(tape,context);seat=j.clip(side,0,1);physical=state.active[r,seat];at=(r,seat,physical);status=state.status[at];counter=state.status_counter[at];hp=state.hp[at];maximum=state.max_hp[at];ability=state.ability[at];item=state.item[at]
 row=(status==m.FRZ).astype(j.int32);move_ok=(move>0)&(move<d.moves.shape[1])&d.moves[row,j.clip(move,0,d.moves.shape[1]-1)]
 raw=j.stack((j.full(b,BEFORE_MOVE_BLOCKED,j.int32),side,physical,move,status,draws[:,0],draws[:,1],draws[:,2],counter,ability),axis=1)
 ordinary=valid_raw(raw)&~loafing&~sleep_usable&(state.confusion_counter[at]==0)&(state.volatile[at]==0)&(caller==move)
 good &= (side>=0)&(side<2)&(physical==tape.cursor.physical_active[r,seat])&(hp>0)&(hp<=maximum)&(maximum<=1000000)&(ability>0)&(ability<len(d.abilities))&d.abilities[j.clip(ability,0,len(d.abilities)-1)]&(item>=0)&(item<len(d.items))&d.items[j.clip(item,0,len(d.items)-1)]
 requested=eligible&move_ok&ordinary;tape,ticket=ex.open_attempt(tape,ex.Site.BEFORE_MOVE_BLOCKED,side,active=requested)
 return tape,Pending(context,ticket,side,physical,move,caller,status,counter,hp,maximum,ability,item,draws,requested,good)
def owned(tape,index,*,stage=2):
 from . import engine_execution_completeness as ex
 c=ex._context(tape);b=len(tape.count);r=j.arange(b);index=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(index,0,tape.words.shape[1]-1);raw=tape.words[r,at]
 if c is None or c.risks is None:return j.zeros(b,bool)
 t=c.risks.tables;l=c.ledger;life._typed(t.before_move_blocked_binding,(8,),j.uint32);life._typed(t.before_move_blocked_moves,(2,len(t.selected_risk)),j.bool_);life._typed(t.before_move_blocked_abilities,(len(t.before_move_blocked_abilities),),j.bool_)
 claim=l.count if stage==1 else l.claimed[r,at];slot=j.clip(claim-1,0,l.site.shape[1]-1);row=(raw[:,4]==m.FRZ).astype(j.int32);ability=raw[:,9]
 good=ex._validate(l,c.tables)&ex._tape(l,tape,c.tables)&l.opened&(l.frame>0)&j.any(t.before_move_blocked_binding!=0)
 for n in ('engine_binding','source_binding','move_binding','producer_binding'):
  life._typed(getattr(tape,n),(8,),j.uint32);life._typed(getattr(t,n),(8,),j.uint32);good &= j.all(getattr(tape,n)==getattr(t,n))
 good &= valid_raw(raw)&(index>=0)&(index<tape.count)&(raw[:,3]<t.before_move_blocked_moves.shape[1])&t.before_move_blocked_moves[row,j.clip(raw[:,3],0,t.before_move_blocked_moves.shape[1]-1)]&(ability<len(t.before_move_blocked_abilities))&t.before_move_blocked_abilities[j.clip(ability,0,len(t.before_move_blocked_abilities)-1)]&(claim>0)&(claim<=l.count)
 good &= (l.site[r,slot]==int(ex.Site.BEFORE_MOVE_BLOCKED))&(l.side[r,slot]==raw[:,1])&(l.stage[r,slot]==stage)&(l.start[r,slot]==index)&(l.no_output_kind[r,slot]==0)&j.all(tape.switch_details[r,at]==0,axis=1)
 good &= j.where(stage==1,tape.count==index+1,l.end[r,slot]==index+1)&(tape.event_turn[r,at]==l.turn[r,slot])&(tape.event_phase[r,at]==l.phase[r,slot])&j.all(tape.event_generation[r,at]==l.generation[r,slot],axis=1)
 if stage==1:good &= (raw[:,2]==tape.cursor.physical_active[r,j.clip(raw[:,1],0,1)])&(l.turn[r,slot]==tape.cursor.turn)&(l.phase[r,slot]==tape.cursor.phase)&j.all(l.generation[r,slot]==tape.cursor.generation,axis=1)
 else:good &= (l.claimed[r,at]==claim)&j.all(l.words[r,at]==raw,axis=1)&(l.event_turn[r,at]==tape.event_turn[r,at])&(l.event_phase[r,at]==tape.event_phase[r,at])&j.all(l.event_generation[r,at]==tape.event_generation[r,at],axis=1)
 return good
def finish(tape,p,state,before):
 if p is None:return tape,None
 from . import engine_execution_completeness as ex
 if type(before)is not m.BeforeMove:raise ValueError('literal actual BeforeMove result required')
 b=len(tape.count);r=j.arange(b);at=(r,j.clip(p.side,0,1),p.physical)
 for v in (before.status,before.counter,before.confusion_counter):life._typed(v,(b,),j.int32)
 for v in (before.can_act,before.confusion_self_hit):life._typed(v,(b,),j.bool_)
 raw=j.stack((j.full(b,BEFORE_MOVE_BLOCKED,j.int32),p.side,p.physical,p.move,p.status,p.draws[:,0],p.draws[:,1],p.draws[:,2],p.counter,p.ability),axis=1)
 valid=p.valid&_bound(tape,p.context)&(p.caller==p.move)&valid_raw(raw)&~before.can_act&~before.confusion_self_hit&(before.confusion_counter==0)&(before.status==p.status)&(before.counter==p.counter)
 valid &= (state.active[r,j.clip(p.side,0,1)]==p.physical)&(state.status[at]==p.status)&(state.status_counter[at]==p.counter)&(state.hp[at]==p.hp)&(state.max_hp[at]==p.maximum)&(state.ability[at]==p.ability)&(state.item[at]==p.item)&(state.confusion_counter[at]==0)&(state.volatile[at]==0)
 room=(tape.count>=0)&(tape.count<tape.words.shape[1]);index=j.clip(tape.count,0,tape.words.shape[1]-1);valid &= j.all(tape.switch_details[r,index]==0,axis=1)
 tape=public._append(tape,BEFORE_MOVE_BLOCKED,p.side,p.physical,p.move,p.status,tuple(raw[:,i]for i in range(5,10)),p.requested&valid&room)
 good=owned(tape,j.maximum(tape.count-1,0),stage=1)&valid&room;c=ex._context(tape);l=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);l=ex.emitted(l,p.ticket,tape,active=p.requested&good,tables=c.tables);tape=ex._put_context(tape,l,c)
 return tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&valid&~room,public.OVERFLOW,0)),Captured(p,raw)
def captured_matches(tape,captured,state,side,before):
 if captured is None:return j.zeros(len(tape.count),bool)
 if type(captured)is not Captured or type(captured.pending)is not Pending:raise ValueError('typed SAME BeforeMove cant capture required')
 from . import engine_execution_completeness as ex
 p=captured.pending;c=ex._context(tape);r=j.arange(len(tape.count));slot,same=ex._ticket(c.ledger,p.ticket);index=j.maximum(tape.count-1,0);raw=tape.words[r,index];at=(r,j.clip(side,0,1),p.physical)
 expected=j.stack((j.full(len(tape.count),BEFORE_MOVE_BLOCKED,j.int32),p.side,p.physical,p.move,p.status,p.draws[:,0],p.draws[:,1],p.draws[:,2],p.counter,p.ability),axis=1)
 return owned(tape,index)&same&p.requested&p.valid&_bound(tape,p.context)&(c.ledger.claimed[r,index]==slot+1)&(c.ledger.end[r,slot]==tape.count)&(p.side==side)&(p.caller==p.move)&(p.physical==state.active[r,j.clip(side,0,1)])&(state.status[at]==p.status)&(state.status_counter[at]==p.counter)&(state.hp[at]==p.hp)&(state.max_hp[at]==p.maximum)&(state.ability[at]==p.ability)&(state.item[at]==p.item)&(state.confusion_counter[at]==0)&(state.volatile[at]==0)&(c.ledger.turn[r,slot]==tape.cursor.turn)&(c.ledger.phase[r,slot]==tape.cursor.phase)&j.all(c.ledger.generation[r,slot]==tape.cursor.generation,axis=1)&j.all(raw==captured.raw,axis=1)&j.all(raw==expected,axis=1)&(raw[:,3]==p.move)&(raw[:,4]==before.status)&(raw[:,8]==before.counter)&(before.confusion_counter==0)&~before.can_act&~before.confusion_self_hit
