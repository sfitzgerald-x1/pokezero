"""One original positive Whirlwind drag transaction; no general switch waiver.

The selected recipient and source party order are actual same-call facts. The
strict quiet entry subset owns a distinct DRAG, before entry handlers execute.
Blocked/no-output/called/missing/entry-hook cases keep their Site43 obligation.
"""
from typing import NamedTuple
from pathlib import Path
import hashlib,json,subprocess
import jax
import jax.numpy as j
import numpy as np
from . import engine_public_events as public,engine_lifecycle_callbacks as life
from . import engine_request_order as order

WHIRLWIND=58
SOURCE_HASHES={'dragIn':'3a522c4ea9cbce2274dd6863c3ca7aa2c22f8941e9004a81c22f085dc451642c','forceSwitch':'33b5539b7fcbede772fa1837b2c4fa5eaaddbcd099d65d050b2325b8d04241ca','switchIn':'b3a9271f92b7c4c5281b2773fd80b5512eb50765ff8d2d5cbad83411e250f500','clearVolatile':'bad93b384074efda34ac198ef3acabbed50f249f6464d9a030a4181a10839885'}
class Data(NamedTuple):
 binding:object
 engine_binding:object
 source_binding:object
 move_binding:object
 producer_binding:object
 move:object
 abilities:object # actor, outgoing, incoming source-hook masks
 items:object
 installed:object # exact original numerical PhazingTable
class Pending(NamedTuple):
 ticket:object
 requested:object
 side:object
 move:object
 caller:object
 slot:object
 roll:object
 hit:object
 immune:object
 protected:object
 absorbed:object
 before:object
 start:object
 turn:object
 generation:object
 stream:object
 source:object # current source vector broadcast as ROWS, never a global8
class Record(NamedTuple):
 pending:object
 after:object # actual post-install, pre-entry state
 final:object # actual post-entry state
 changed:object
 incoming:object
 known:object
 finalized:object

def compile_data(catalog,metadata,rules):
 from . import engine_request_source as request
 from .engine_private_team import engine_metadata_binding
 names=metadata['vocabulary']['moves']
 if 'whirlwind' not in names:return None
 origin=request._rules_origin(rules)
 if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Whirlwind source/metadata required')
 script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8')),verified=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),A=require(path.join(root,'dist/sim/battle-actions.js')).BattleActions,P=require(path.join(root,'dist/sim/pokemon.js')).Pokemon,m=d.moves.get('whirlwind');
const hashes=Object.fromEntries(['dragIn','forceSwitch','switchIn'].map(k=>[k,crypto.createHash('sha256').update(String(A.prototype[k])).digest('hex')]));hashes.clearVolatile=crypto.createHash('sha256').update(String(P.prototype.clearVolatile)).digest('hex');
const hooks=[['onTryMove','onPrepareHit','onModifyMove','onAnyModifyMove','onAllyModifyMove','onFoeTryHit','onSourceTryHit','onAnyTryHit','onAllyTryHit','onTryHit','onUpdate'],['onTryHit','onTryPrimaryHit','onDragOut','onEnd','onBeforeSwitchOut','onSwitchOut','onAnySwitchIn','onAllySwitchIn','onUpdate','onAnyTryHit'],['onStart','onSwitchIn','onAnySwitchIn','onAllySwitchIn','onUpdate','onEnd']];
const quiet=(e,hs)=>e.exists&&e.gen<=3&&!hs.some(k=>e[k]!==undefined);
process.stdout.write(JSON.stringify({source_tree:verified.source_tree,compiled_tree:verified.compiled_tree,hashes,move:{id:m.id,gen:m.gen,accuracy:m.accuracy,power:m.basePower,category:m.category,type:m.type,target:m.target,priority:m.priority,forceSwitch:m.forceSwitch,flags:m.flags,callbacks:Object.keys(m).filter(k=>/^on/.test(k)&&m[k]!==undefined)},abilities:hooks.map(hs=>meta.vocabulary.abilities.map(n=>quiet(d.abilities.get(n),hs))),items:meta.vocabulary.items.map((n,i)=>!i||quiet(d.items.get(n),[...new Set(hooks.flat())]))}));"""
 d=json.loads(subprocess.check_output([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata).encode(),timeout=30))
 if d['hashes']!=SOURCE_HASHES or d['move']!=dict(id='whirlwind',gen=1,accuracy=100,power=0,category='Status',type='Normal',target='normal',priority=-6,forceSwitch=True,flags=dict(protect=1,mirror=1,bypasssub=1,metronome=1),callbacks=[]):raise ValueError('canonical original Gen3 Whirlwind/drag source required')
 if json.dumps(d['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(d['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('Whirlwind source tree drift')
 move=names.index('whirlwind');t=catalog.engine_table
 if (move<=0 or names.count('whirlwind')!=1 or t.phazing is None or not bool(t.phazing.move[move])or bool(t.phazing.roar[move])or int(catalog.ids[move])<=0 or int(catalog.targets[move])!=1 or int(t.power[move])!=0 or int(t.accuracy[move])!=100 or bool(t.always_hit[move])or int(t.priority[move])!=-6 or int(t.type[move])!=metadata['vocabulary']['types'].index('normal')or bool(t.physical[move])):raise ValueError('canonical installed ordinary Whirlwind required')
 payload=dict(source=d,installed=[np.asarray(x).tolist()for x in t.phazing],engine=catalog.engine_binding,source_binding=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,move=move)
 binding=np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32);word=lambda x:j.asarray(x,j.uint32)
 return Data(word(binding),word(catalog.engine_binding),word(catalog.source_binding),word(catalog.move_binding),word(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32),j.asarray(d['abilities'],bool),j.asarray(d['items'],bool),t.phazing)

def vector(data):
 shape(data);header=j.asarray([int(data.abilities.shape[1]),len(data.items),*(len(x)for x in data.installed)],j.int32)
 return j.concatenate((data.move[None],header,data.abilities.reshape(-1).astype(j.int32),data.items.astype(j.int32),*(x.astype(j.int32)for x in data.installed)))
def shape(data):
 if type(data)is not Data:raise ValueError('original Whirlwind Data required')
 for x in data[:5]:life._typed(x,(8,),j.uint32)
 life._typed(data.move,(),j.int32);life._typed(data.abilities,(3,data.abilities.shape[1]),j.bool_);life._typed(data.items,(len(data.items),),j.bool_)
 from .phazing import PhazingTable
 if type(data.installed)is not PhazingTable:raise ValueError('original PhazingTable required')
 for x in data.installed:life._typed(x,(len(x),),j.bool_)
def equal(a,b):
 from .state import BattleState
 if type(a)is not BattleState or type(b)is not BattleState:raise ValueError('same actual private BattleState schema required')
 good=j.ones(a.hp.shape[0],bool)
 for x,y in zip(a,b,strict=True):
  if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('same private shape/dtype required')
  good &= j.all(x==y,axis=tuple(range(1,x.ndim)))
 return good

def _bound(tape,data=None):
 from . import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
 b=len(tape.count);c=ex._context(tape)
 for n in('engine_binding','source_binding','move_binding','producer_binding'):life._typed(getattr(tape,n),(8,),j.uint32)
 if c is None or type(c.risks)is not risk.Context:return j.zeros(b,bool)
 t=c.risks.tables
 life._typed(t.whirlwind_binding,(8,),j.uint32);life._typed(t.whirlwind_move,(),j.int32);life._typed(t.whirlwind_abilities,(3,t.whirlwind_abilities.shape[1]),j.bool_);life._typed(t.whirlwind_items,(len(t.whirlwind_items),),j.bool_);life._typed(t.whirlwind_source,(len(t.whirlwind_source),),j.int32)
 good,_=risk._bound(c.risks.carry,tape,t)
 good &= ex._validate(c.ledger,c.tables)&ex._tape(c.ledger,tape,c.tables)&c.ledger.opened&j.any(t.whirlwind_binding!=0)&(t.whirlwind_move>0)
 if data is not None:
  shape(data)
  for n in('engine_binding','source_binding','move_binding','producer_binding'):good &= j.all(getattr(data,n)==getattr(tape,n))
  if data.abilities.shape!=t.whirlwind_abilities.shape or data.items.shape!=t.whirlwind_items.shape or vector(data).shape!=t.whirlwind_source.shape:raise ValueError('same Whirlwind source namespace required')
  good &= j.all(data.binding==t.whirlwind_binding)&(data.move==t.whirlwind_move)&j.all(data.abilities==t.whirlwind_abilities)&j.all(data.items==t.whirlwind_items)&j.all(vector(data)==t.whirlwind_source)
 return good

def _order(tape,side,stop):
 # Same actual BEFORE-frame owner orders, but only installations BEFORE this
 # source operation. Never infer source party order from final active state.
 c=tape.execution;b=len(side);r=j.arange(b)
 if c.party_orders is None:return None
 if type(c.party_orders)is not tuple or len(c.party_orders)!=2 or any(type(x)is not order.Order for x in c.party_orders):raise ValueError('actual two owner orders required')
 a,z=c.party_orders;current=order.Order(*(j.where((side==0).reshape((b,)+(1,)*(x.ndim-1)),x,y)for x,y in zip(a,z,strict=True)))
 def step(o,index):
  raw=tape.words[:,index];present=index<stop;generation=j.where(present,tape.event_generation[:,index][r,side],o.generation)
  receipt=order.SwitchInstalled(tape.stream_id,side,j.broadcast_to(tape.engine_binding,(b,8)),generation,present&(raw[:,0]==life.SWITCH_INSTALLED)&(raw[:,1]==side),raw[:,2],raw[:,4])
  return order.install_switch(o,receipt),None
 return jax.lax.scan(step,current,j.arange(tape.words.shape[1]))[0]

def _select(before,side,roll,perm):
 r=j.arange(len(side));old=before.active[r,side];safe=j.clip(perm,0,5);alive=(perm>=0)&(perm<6)&(safe!=old[:,None])&(before.hp[r[:,None],side[:,None],safe]>0)
 count=j.sum(alive,axis=1,dtype=j.int32);rank=j.cumsum(alive,axis=1)-1;pos=j.argmax(alive&(rank==(roll%j.maximum(1,count))[:,None]),axis=1)
 return perm[r,pos],count>0

def _start_valid(tape,p):
 from . import engine_execution_completeness as ex
 b=len(tape.count);r=j.arange(b);s=p.before;side=j.clip(p.side,0,1);foe=1-side;actor=s.active[r,side];old=s.active[r,foe];t=tape.execution.risks.tables;l=tape.execution.ledger
 for n in('side','move','caller','slot','roll','start','turn','stream'):life._typed(getattr(p,n),(b,),j.int32)
 for n in('requested','hit','immune','protected','absorbed'):life._typed(getattr(p,n),(b,),j.bool_)
 life._typed(p.generation,(b,2),j.int32);life._typed(p.source,(b,len(t.whirlwind_source)),j.int32)
 slot,same=ex._ticket(l,p.ticket);good=_bound(tape)&same&(p.side>=0)&(p.side<2)&(p.move==t.whirlwind_move)&(p.caller==p.move)&p.hit&~p.immune&~p.protected&~p.absorbed&(p.roll>=0)&(p.roll<60)&(p.slot>=0)&(p.slot<4)
 good &= (p.stream==tape.stream_id)&(p.ticket.frame==l.frame)&(p.start==l.start[r,slot])&(l.site[r,slot]==43)&(l.side[r,slot]==p.side)&(p.turn==l.turn[r,slot])&(l.phase[r,slot]==0)&j.all(p.generation==l.generation[r,slot],axis=1)&j.all(p.source==t.whirlwind_source,axis=1)
 prefix=j.clip(p.start-1,0,tape.words.shape[1]-1);raw=tape.words[r,prefix];good &= (p.start>0)&(raw[:,0]==public.MOVE)&(raw[:,1]==p.side)&(raw[:,2]==actor)&(raw[:,3]==foe)&(raw[:,4]==old)&(raw[:,5]==p.move)&j.all(raw[:,6:]==0,axis=1)&(tape.event_turn[r,prefix]==p.turn)&(tape.event_phase[r,prefix]==0)&j.all(tape.event_generation[r,prefix]==p.generation,axis=1)
 good &= j.all(s.active>=0,axis=1)&j.all(s.active<6,axis=1)&~s.done&~s.pending_residual&~j.any(s.forced_switch|s.entry_pending|s.baton_pass,axis=1)&(s.turn+1==p.turn)&(s.moves[r,side,actor,j.clip(p.slot,0,3)]==p.move)&j.all(s.pp>=0,axis=(1,2,3))&j.all(s.hp>0,axis=(1,2))&j.all(s.hp<=s.max_hp,axis=(1,2))
 a=s.ability[r,side,actor];out=s.ability[r,foe,old];items=s.item
 good &= (a>=0)&(a<t.whirlwind_abilities.shape[1])&(out>=0)&(out<t.whirlwind_abilities.shape[1])&t.whirlwind_abilities[0,j.clip(a,0,t.whirlwind_abilities.shape[1]-1)]&t.whirlwind_abilities[1,j.clip(out,0,t.whirlwind_abilities.shape[1]-1)]
 good &= j.all((items>=0)&(items<len(t.whirlwind_items))&t.whirlwind_items[j.clip(items,0,len(t.whirlwind_items)-1)],axis=(1,2))&j.all(s.ability==s.base_ability,axis=(1,2))&j.all(s.types==s.base_types,axis=(1,2,3))
 good &= (s.hp[r,side,actor]>0)&(s.hp[r,foe,old]>0)&j.all(s.status==0,axis=(1,2))&j.all(s.status_counter==0,axis=(1,2))&j.all(s.volatile==0,axis=(1,2))&j.all(s.confusion_counter==0,axis=(1,2))&j.all(s.trap_source==-1,axis=(1,2))&~j.any(s.trapped,axis=1)&~j.any(s.spikes!=0,axis=1)&~j.any(s.sleep_skipped!=0,axis=(1,2))&~j.any(s.rest_origin,axis=(1,2))&~j.any(s.truant_loaf,axis=(1,2))
 current=_order(tape,foe,p.start)
 if current is None:return good&False,j.zeros(b,j.int32)
 good &= order._valid(current)&(current.owner_side==foe)&(current.stream_id==p.stream)&j.all(current.engine_binding==tape.engine_binding,axis=1)&(current.generation==p.generation[r,foe])&(current.request_to_physical[:,0]==old)&(current.team_size==6)
 incoming,exists=_select(s,foe,p.roll,current.request_to_physical);new=j.clip(incoming,0,5);abi=s.ability[r,foe,new]
 good &= exists&(incoming>=0)&(incoming<6)&(incoming!=old)&(abi>=0)&(abi<t.whirlwind_abilities.shape[1])&t.whirlwind_abilities[2,j.clip(abi,0,t.whirlwind_abilities.shape[1]-1)]&(s.species[r,foe,new]>0)&(s.level[r,foe,new]>0)&(s.level[r,foe,new]<=100)&(s.gender[r,foe,new]>=0)&(s.gender[r,foe,new]<=2)&(s.hp[r,foe,new]>0)&(s.max_hp[r,foe,new]<=1000000)
 good &= (l.refusal==0)&((l.upstream_refusal&~public.UNINSTRUMENTED)==0)&((tape.refusal&~public.UNINSTRUMENTED)==0)
 return good,incoming

def begin(tape,state,side,move,caller,slot,roll,context,table,*,eligible,hit,immune,protected,absorbed):
 from . import engine_execution_completeness as ex
 if context is None or context.data.whirlwind is None:return tape,None
 from . import engine_direct_hit_controls as direct
 d=context.data.whirlwind;bound=_bound(tape,d)&direct._registered_bound(tape,context)
 for actual,expected in zip(table.phazing,d.installed,strict=True):
  if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('current installed phazing namespace required')
  bound &= j.all(actual==expected)
 requested=eligible&(move==d.move);prior=tape;tape,ticket=ex.open_attempt(tape,ex.Site.WHIRLWIND,side,active=requested)
 p=Pending(ticket,requested,side,move,caller,slot,roll,hit,immune,protected,absorbed,state,prior.count,prior.cursor.turn,prior.cursor.generation,prior.stream_id,j.broadcast_to(vector(d),(len(side),len(vector(d)))))
 # A failed source/table join remains a requested unresolved operation. Keep
 # the observed source vector unequal, rather than manufacture a valid flag.
 p=p._replace(source=j.where(bound[:,None],p.source,j.full_like(p.source,-1)))
 return tape,p

def _departed(p,incoming):
 s=p.before;b=len(p.side);r=j.arange(b);side=1-j.clip(p.side,0,1);old=s.active[r,side]
 for n in('boosts','volatile','confusion_counter','protect_counter','substitute_hp','choice_move','last_move','encore_move','encore_counter','charge_move','charge_counter'):
  x=getattr(s,n);s=s._replace(**{n:x.at[r,side,old].set(0)})
 return s._replace(active=s.active.at[r,side].set(incoming),active_turns=s.active_turns.at[r,side].set(0),trapped=s.trapped.at[r,side].set(False),baton_pass=s.baton_pass.at[r,side].set(False),last_action_kind=s.last_action_kind.at[r,side].set(1),last_action_index=s.last_action_index.at[r,side].set(incoming))

def install(tape,p,state,changed):
 if p is None:return tape,None
 good,incoming=_start_valid(tape,p);good &= p.requested&changed&equal(_departed(p,j.clip(incoming,0,5)),state)
 r=j.arange(len(p.side));foe=1-p.side;old=p.before.active[r,foe];z=j.zeros_like(foe)
 tape=public._append(tape,WHIRLWIND,p.side,p.before.active[r,p.side],foe,old,(p.move,incoming,p.roll,z,z),good)
 # This is the actual physical installation BEFORE entry callbacks. Distinct
 # DRAG grammar is proved below, never relabel an ordinary SWITCH receipt.
 tape=life.installed(tape,state,foe,old,incoming,eligible=good,replacement=j.zeros_like(good),baton=j.zeros_like(good),drag=good)
 return tape,Record(p,state,state,changed,incoming,good,j.zeros_like(good))

def finish(tape,rec,state):
 if rec is None:return tape
 from . import engine_execution_completeness as ex
 p=rec.pending;good=rec.known&equal(rec.after,state)&(tape.count==p.start+2);rec=rec._replace(final=state,known=good,finalized=p.requested)
 c=ex._context(tape);l=ex.unhandled(c.ledger,p.ticket,active=p.requested&~good,tables=c.tables);l=ex.emitted(l,p.ticket,tape,active=p.requested&good,tables=c.tables)
 records=()if c.whirlwind is None else c.whirlwind
 if type(records)is not tuple or len(records)>=2:raise ValueError('at most two actual Whirlwind records per frame')
 return ex._put_context(tape,l,c._replace(whirlwind=records+(rec,)))

def owned(tape,index):
 from . import engine_execution_completeness as ex
 if tape.execution is None or tape.execution.whirlwind is None:return j.zeros(len(tape.count),bool)
 records=tape.execution.whirlwind
 if type(records)is not tuple or len(records)>2:raise ValueError('bounded original Whirlwind records required')
 b=len(tape.count);r=j.arange(b);result=j.zeros(b,bool);index=j.broadcast_to(index,(b,));l=tape.execution.ledger
 for rec in records:
  if type(rec)is not Record:raise ValueError('actual Whirlwind Record required')
  p=rec.pending;good,incoming=_start_valid(tape,p);slot,same=ex._ticket(l,p.ticket);at=j.clip(p.start,0,tape.words.shape[1]-1);next_at=j.clip(p.start+1,0,tape.words.shape[1]-1);foe=1-p.side;old=p.before.active[r,foe];z=j.zeros(b,j.int32)
  expected=j.stack((z+WHIRLWIND,p.side,p.before.active[r,p.side],foe,old,p.move,incoming,p.roll,z,z),axis=1);installed=j.stack((z+life.SWITCH_INSTALLED,foe,old,foe,incoming,z,z,z+1,z,z),axis=1)
  good &= rec.known&rec.finalized&rec.changed&p.requested&(rec.incoming==incoming)&equal(rec.after,_departed(p,j.clip(incoming,0,5)))&equal(rec.final,rec.after)&same&(l.stage[r,slot]==2)&(l.end[r,slot]==p.start+2)&(p.start+2<=tape.count)&j.all(tape.words[r,at]==expected,axis=1)&j.all(tape.words[r,next_at]==installed,axis=1)
  details=j.stack(tuple(getattr(rec.after,n)[r,foe,j.clip(incoming,0,5)]for n in('species','level','gender','hp','max_hp','status')),axis=1);good &= j.all(tape.switch_details[r,next_at]==details,axis=1)
  for pos in(at,next_at):good &= (tape.event_turn[r,pos]==p.turn)&(tape.event_phase[r,pos]==0)&j.all(tape.event_generation[r,pos]==p.generation,axis=1)&(l.claimed[r,pos]==slot+1)&j.all(l.words[r,pos]==tape.words[r,pos],axis=1)&(l.event_turn[r,pos]==tape.event_turn[r,pos])&(l.event_phase[r,pos]==tape.event_phase[r,pos])&j.all(l.event_generation[r,pos]==tape.event_generation[r,pos],axis=1)
  result |= good&((index==p.start)|(index==p.start+1))
 return result

def commit(active,records):
 if records is None:return None
 return tuple(rec._replace(known=rec.known&active)for rec in records)
