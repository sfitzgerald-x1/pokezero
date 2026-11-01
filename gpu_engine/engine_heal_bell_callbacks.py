"""Source-owned plain Heal Bell transaction around the EXISTING cure_team write.

RAW52/Site37: activation plus each original-order member's literal status and
provenance facts. Unknown party order/public bench identity stays UNKNOWN.
"""
import hashlib,json,subprocess
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
from . import engine_lifecycle_callbacks as life,engine_request_order as order
from . import engine_public_events as public

HEAL_BELL=52
HEADER,STATUS,COUNTERS,ORIGIN=range(4)
SOURCE={'onHit':'1af77538a51c4594653bcb8c7087d8dfa5f562c5c35c39f67a1881b1fb548d5b','cureStatus':'2bc9bbb3b7727c5faab3d0569c39ac959f3ea2d8b8a036ef3ae2e7b97dbdebf5','setStatus':'1078d45febffb2ca663eca5c04d18d154304aaa467b57bf22535d90715172ef0','getRandomTarget':'f8bc14a3934ea026686c58770fc7c5cc3e8b226613ccab21c5ac2e5ac78428a5','switchIn':'b3a9271f92b7c4c5281b2773fd80b5512eb50765ff8d2d5cbad83411e250f500'}
class Data(NamedTuple):
 binding:object
 engine_binding:object
 source_binding:object
 move_binding:object
 producer_binding:object
 move:object
 installed:object
class Pending(NamedTuple):
 context:object
 ticket:object
 side:object
 physical:object
 move:object
 before:object
 permutation:object
 requested:object
 valid:object
 secondary:object
class Captured(NamedTuple):
 pending:object
 after:object
 write_mask:object
 end:object

def compile_data(catalog,metadata,rules):
 from . import engine_request_source as request
 from .engine_private_team import engine_metadata_binding
 origin=request._rules_origin(rules);table=catalog.engine_table
 if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original Heal Bell metadata/rules required')
 if table.effects is None:return None
 wanted=np.array([name=='healbell'for name in metadata['vocabulary']['moves']]);found=np.flatnonzero(wanted)
 if not np.array_equal(table.effects.heal_bell,wanted):raise ValueError('exact original installed Heal Bell column required')
 if not len(found):return None
 if len(found)!=1 or not found[0]:raise ValueError('one canonical Heal Bell row required')
 move=int(found[0])
 script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8')),v=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3'),{Pokemon}=require(path.join(root,'dist/sim/pokemon.js')),{Battle}=require(path.join(root,'dist/sim/battle.js')),{BattleActions}=require(path.join(root,'dist/sim/battle-actions.js')),h=x=>crypto.createHash('sha256').update(x.toString()).digest('hex'),m=d.moves.get('healbell');
process.stdout.write(JSON.stringify({hashes:{onHit:h(m.onHit),cureStatus:h(Pokemon.prototype.cureStatus),setStatus:h(Pokemon.prototype.setStatus),getRandomTarget:h(Battle.prototype.getRandomTarget),switchIn:h(BattleActions.prototype.switchIn)},row:{id:m.id,gen:m.gen,target:m.target,category:m.category,power:m.basePower,accuracy:m.accuracy,priority:m.priority,flags:m.flags,callbacks:Object.keys(m).filter(k=>k.startsWith('on')&&m[k]!==undefined)},source_tree:v.source_tree,compiled_tree:v.compiled_tree}));"""
 d=json.loads(subprocess.run([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata,allow_nan=False),capture_output=True,text=True,check=True,timeout=30).stdout)
 if d['hashes']!=SOURCE or json.dumps(d['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(d['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original effective Gen3 Heal Bell source changed')
 row=d['row']
 if row!={'id':'healbell','gen':2,'target':'allyTeam','category':'Status','power':0,'accuracy':True,'priority':0,'flags':{'snatch':1,'sound':1,'metronome':1},'callbacks':['onHit']}:raise ValueError('original plain Heal Bell envelope changed')
 if catalog.ids[move]<=0 or catalog.targets[move]!=0 or int(table.power[move])!=0 or not bool(table.always_hit[move])or int(table.priority[move])!=0:raise ValueError('installed Heal Bell row/lexical target changed')
 soundproof=np.array([n=='soundproof'for n in metadata['vocabulary']['abilities']])
 if not np.array_equal(table.effects.soundproof,soundproof):raise ValueError('original current Soundproof column changed')
 payload=dict(schema='pokezero.actual-plain-heal-bell.v1',descriptor=d,installed={k:np.asarray(v).tolist()for k,v in zip(table.effects._fields,table.effects)},engine=catalog.engine_binding,source=catalog.source_binding,moves=catalog.move_binding,producer=catalog.binding,move=move)
 words=lambda x:j.asarray(x,j.uint32)
 return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(move,j.int32),table.effects)

def shape(d):
 if type(d)is not Data:raise ValueError('typed original Heal Bell data required')
 for x in d[:5]:life._typed(x,(8,),j.uint32)
 life._typed(d.move,(),j.int32)
 life._typed(d.installed.heal_bell,(len(d.installed.heal_bell),),j.bool_)
 life._typed(d.installed.soundproof,(len(d.installed.soundproof),),j.bool_)

def _bound(tape,context):
 from . import engine_direct_hit_controls as direct
 d=context.data.heal_bell;shape(d);ok=direct._registered_bound(tape,context)
 for n in ('engine_binding','source_binding','move_binding','producer_binding'):ok &= j.all(getattr(tape,n)==getattr(d,n))
 if tape.execution.risks is None:return j.zeros(len(tape.count),bool)
 t=tape.execution.risks.tables;life._typed(t.heal_bell_binding,(8,),j.uint32);life._typed(t.heal_bell_move,(),j.int32);life._typed(t.heal_bell_soundproof,d.installed.soundproof.shape,j.bool_)
 return ok&j.any(d.binding!=0)&j.all(t.heal_bell_binding==d.binding)&(t.heal_bell_move==d.move)&j.all(t.heal_bell_soundproof==d.installed.soundproof)

def _prefix_order(tape,side):
 """Advance actual BEFORE-frame Order only using actual installation receipts."""
 b=len(tape.count);rows=j.arange(b);c=tape.execution
 if c.party_orders is None:return None
 if type(c.party_orders)is not tuple or len(c.party_orders)!=2 or any(type(x)is not order.Order for x in c.party_orders):raise ValueError('two actual original owner orders required')
 a,z=c.party_orders
 selected=order.Order(*(j.where((side==0).reshape((b,)+(1,)*(x.ndim-1)),x,y)for x,y in zip(a,z)))
 def step(current,index):
  raw=tape.words[:,index];present=index<tape.count
  generation=j.where(present,tape.event_generation[:,index][rows,side],current.generation)
  receipt=order.SwitchInstalled(tape.stream_id,side,j.broadcast_to(tape.engine_binding,(b,8)),generation,
   present&(raw[:,0]==life.SWITCH_INSTALLED)&(raw[:,1]==side),raw[:,2],raw[:,4])
  return order.install_switch(current,receipt),None
 selected,_=jax.lax.scan(step,selected,j.arange(tape.words.shape[1]))
 return selected

def begin(tape,state,side,move,context,table,*,eligible,primary,secondary):
 from . import engine_execution_completeness as ex
 if context is None or context.data.heal_bell is None:return tape,None
 b=len(move);rows=j.arange(b);life._typed(side,(b,),j.int32);side_valid=(side>=0)&(side<2);side=j.clip(side,0,1);d=context.data.heal_bell;shape(d)
 for v in (eligible,primary,secondary):life._typed(v,(b,),j.bool_)
 requested=eligible&(move==d.move)&(d.move>0);good=_bound(tape,context)&primary&side_valid
 for actual,expected in zip(table.effects,d.installed,strict=True):
  if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise ValueError('original Heal Bell installed table shape')
  good &= j.all(actual==expected)
 actual_order=_prefix_order(tape,side)
 if actual_order is None:
  perm=j.full((b,6),-1,j.int32);good &= False
 else:
  perm=actual_order.request_to_physical
  good &= order._valid(actual_order)&(actual_order.owner_side==side)&(actual_order.stream_id==tape.stream_id)&j.all(actual_order.engine_binding==d.engine_binding,axis=1)
  good &= (actual_order.generation==tape.cursor.generation[rows,side])&(perm[:,0]==state.active[rows,side])
 physical=state.active[rows,side];at=j.clip(tape.count-1,0,tape.words.shape[1]-1);raw=tape.words[rows,at]
 good &= (side>=0)&(side<2)&j.all(state.active==tape.cursor.physical_active,axis=1)&(state.hp[rows,side,physical]>0)
 good &= life._valid(tape.cursor)&tape.cursor.initialized&(tape.cursor.phase==0)&(tape.count>0)&(raw[:,0]==public.MOVE)&(raw[:,1]==side)&(raw[:,2]==physical)&(raw[:,5]==move)&j.all(raw[:,6:]==0,axis=1)
 good &= (tape.event_turn[rows,at]==tape.cursor.turn)&(tape.event_phase[rows,at]==0)&j.all(tape.event_generation[rows,at]==tape.cursor.generation,axis=1)
 good &= ((tape.refusal&~public.UNINSTRUMENTED)==0)&(tape.execution.ledger.refusal==0)&((tape.execution.ledger.upstream_refusal&~public.UNINSTRUMENTED)==0)
 abi=state.ability[rows,side];hp=state.hp[rows,side];status=state.status[rows,side];safe=j.clip(abi,0,len(d.installed.soundproof)-1)
 cure=(hp>0)&(status!=0)&~d.installed.soundproof[safe]
 visible=cure|((j.arange(6)[None,:]==physical[:,None])&d.installed.soundproof[safe])
 good &= j.all((abi>=0)&(abi<len(d.installed.soundproof))&(status>=0)&(status<=6)&(hp>=0)&(state.max_hp[rows,side]>=hp),axis=1)
 good &= j.all(~visible|(tape.cursor.physical_to_public[rows,side]>=0),axis=1)
 good &= j.all((state.species[rows,side]>0)==j.any(perm[:,:,None]==j.arange(6)[None,None,:],axis=1),axis=1)
 good &= j.all((state.status_counter[rows,side]>=0)&(state.sleep_skipped[rows,side]>=0)&(state.rest_origin[rows,side]>=0)&(state.rest_origin[rows,side]<=1),axis=1)
 tape,ticket=ex.open_attempt(tape,ex.Site.HEAL_BELL,side,active=requested)
 return tape,Pending(context,ticket,side,physical,move,state,perm,requested,good,secondary)

def prospective(p):return j.asarray(False)if p is None else p.requested&p.valid

def finish(tape,p,after,write_mask):
 from . import engine_execution_completeness as ex
 if p is None:return tape,None
 b=len(tape.count);rows=j.arange(b);d=p.context.data.heal_bell;life._typed(write_mask,(b,6),j.bool_)
 before=p.before;side=p.side;abi=before.ability[rows,side];status=before.status[rows,side];hp=before.hp[rows,side]
 expected=p.requested[:,None]&(hp>0)&(status!=0)&~d.installed.soundproof[j.clip(abi,0,len(d.installed.soundproof)-1)]
 good=p.valid&_bound(tape,p.context)&j.all(write_mask==expected,axis=1)
 current_order=_prefix_order(tape,side)
 if current_order is None:good &= False
 else:good &= order._valid(current_order)&j.all(current_order.request_to_physical==p.permutation,axis=1)&(current_order.generation==tape.cursor.generation[rows,side])
 ledger=tape.execution.ledger;slot=j.clip(p.ticket.slot,0,ledger.stage.shape[1]-1)
 good &= (p.physical==before.active[rows,side])&(p.physical==tape.cursor.physical_active[rows,side])&(p.move==d.move)
 good &= (ledger.serial[rows,slot]==p.ticket.serial)&(ledger.side[rows,slot]==side)&(ledger.stage[rows,slot]==1)&(ledger.turn[rows,slot]==tape.cursor.turn)&(ledger.phase[rows,slot]==tape.cursor.phase)&j.all(ledger.generation[rows,slot]==tape.cursor.generation,axis=1)
 for name in before._fields:
  old=getattr(before,name);new=getattr(after,name)
  if old is None or new is None:
   good &= False
   continue
  if name in ('status','status_counter','sleep_skipped','rest_origin'):
   wanted=old.at[rows,side].set(j.where(write_mask,0,old[rows,side]))
  else:wanted=old
  good &= j.all((new==wanted).reshape((b,-1)),axis=1)
 good &= tape.count==ex._context(tape).ledger.start[rows,j.clip(p.ticket.slot,0,ex._context(tape).ledger.stage.shape[1]-1)]
 z=j.zeros(b,j.int32);size=j.sum(p.permutation>=0,axis=1,dtype=j.int32)
 room=(tape.count>=0)&(tape.count+1+3*size<=tape.words.shape[1])
 positions=j.arange(tape.words.shape[1])[None,:];future=(positions>=tape.count[:,None])&(positions<(tape.count+1+3*size)[:,None])
 good &= room&j.all(~future|j.all(tape.switch_details==0,axis=2),axis=1)
 use=p.requested&good
 tape=public._append(tape,HEAL_BELL,side,p.physical,side,p.physical,(z,p.move,size,p.secondary.astype(j.int32),z),use)
 for position in range(6):
  physical=p.permutation[:,position];safe=j.clip(physical,0,5);on=use&(physical>=0)
  def get(s,name):return getattr(s,name)[rows,side,safe]
  tape=public._append(tape,HEAL_BELL,side,p.physical,side,physical,(z+STATUS,get(before,'status'),get(after,'status'),get(before,'hp'),get(before,'ability')),on)
  tape=public._append(tape,HEAL_BELL,side,p.physical,side,physical,(z+COUNTERS,get(before,'status_counter'),get(after,'status_counter'),get(before,'sleep_skipped'),get(after,'sleep_skipped')),on)
  tape=public._append(tape,HEAL_BELL,side,p.physical,side,physical,(z+ORIGIN,get(before,'rest_origin'),get(after,'rest_origin'),get(before,'max_hp'),get(after,'max_hp')),on)
 c=ex._context(tape);ledger=ex.emitted(c.ledger,p.ticket,tape,active=use,tables=c.tables)
 ledger=ex.unhandled(ledger,p.ticket,active=p.requested&~good,tables=c.tables)
 tape=ex._put_context(tape._replace(refusal=tape.refusal|j.where(p.requested&~good,256,0)|j.where(p.requested&~room,public.OVERFLOW,0)),ledger,c)
 return tape,Captured(p,after,write_mask,tape.count)

def owned(tape,index,*,stage=2):
 """Independently validate every claimed original-order transaction word."""
 from . import engine_execution_completeness as ex
 c=ex._context(tape);b=len(tape.count);rows=j.arange(b);idx=j.broadcast_to(j.asarray(index,j.int32),(b,));at=j.clip(idx,0,tape.words.shape[1]-1)
 if c is None or c.risks is None:return j.zeros(b,bool)
 t=c.risks.tables;life._typed(t.heal_bell_binding,(8,),j.uint32);life._typed(t.heal_bell_move,(),j.int32);life._typed(t.heal_bell_soundproof,(len(t.heal_bell_soundproof),),j.bool_)
 l=c.ledger;claim=l.claimed[rows,at];slot=j.clip(claim-1,0,l.site.shape[1]-1);start=l.start[rows,slot];end=l.end[rows,slot];h=tape.words[rows,j.clip(start,0,tape.words.shape[1]-1)];size=h[:,7];side=h[:,1];actor=h[:,2]
 good=ex._validate(l,c.tables)&ex._tape(l,tape,c.tables)&l.opened&j.any(t.heal_bell_binding!=0)&(t.heal_bell_move>0)&(claim>0)&(claim<=l.count)&(l.site[rows,slot]==int(ex.Site.HEAL_BELL))&(l.stage[rows,slot]==stage)&(l.no_output_kind[rows,slot]==0)
 good &= (h[:,0]==HEAL_BELL)&(h[:,3]==side)&(h[:,4]==actor)&(h[:,5]==HEADER)&(h[:,6]==t.heal_bell_move)&(h[:,8]>=0)&(h[:,8]<=1)&(h[:,9]==0)&(size>=1)&(size<=6)&(side>=0)&(side<2)&(actor>=0)&(actor<6)
 good &= (start>0)&(end==start+1+3*size)&(end<=tape.count)&(idx>=start)&(idx<end)&(l.side[rows,slot]==side)
 caller=tape.words[rows,j.clip(start-1,0,tape.words.shape[1]-1)]
 good &= (caller[:,0]==public.MOVE)&(caller[:,1]==side)&(caller[:,2]==actor)&(caller[:,5]==h[:,6])&j.all(caller[:,6:]==0,axis=1)
 seen=j.zeros((b,6),bool)
 for position in range(6):
  base=start+1+position*3;on=position<size
  facts=tape.words[rows,j.clip(base,0,tape.words.shape[1]-1)];count=tape.words[rows,j.clip(base+1,0,tape.words.shape[1]-1)];origin=tape.words[rows,j.clip(base+2,0,tape.words.shape[1]-1)]
  physical=facts[:,4];safe=j.clip(physical,0,5);abi=facts[:,9];sp=t.heal_bell_soundproof[j.clip(abi,0,len(t.heal_bell_soundproof)-1)];cured=(facts[:,8]>0)&(facts[:,6]!=0)&~sp
  valid=(physical>=0)&(physical<6)&~seen[rows,safe]&(facts[:,5]==STATUS)&(count[:,5]==COUNTERS)&(origin[:,5]==ORIGIN)
  valid &= (facts[:,6]>=0)&(facts[:,6]<=6)&(facts[:,7]==j.where(cured,0,facts[:,6]))&(facts[:,8]>=0)&(abi>=0)&(abi<len(t.heal_bell_soundproof))
  valid &= (count[:,6]>=0)&(count[:,8]>=0)&(count[:,7]==j.where(cured,0,count[:,6]))&(count[:,9]==j.where(cured,0,count[:,8]))
  valid &= (origin[:,6]>=0)&(origin[:,6]<=1)&(origin[:,7]==j.where(cured,0,origin[:,6]))&(origin[:,8]>=facts[:,8])&(origin[:,9]==origin[:,8])
  valid &= j.where(position==0,physical==actor,True)
  for offset,raw in enumerate((facts,count,origin)):
   i=j.clip(base+offset,0,tape.words.shape[1]-1)
   valid &= (raw[:,0]==HEAL_BELL)&(raw[:,1]==side)&(raw[:,2]==actor)&(raw[:,3]==side)&(raw[:,4]==physical)&(l.claimed[rows,i]==claim)&j.all(l.words[rows,i]==raw,axis=1)&j.all(tape.switch_details[rows,i]==0,axis=1)
  good &= ~on|valid;seen=seen.at[rows,safe].set(seen[rows,safe]|on)
 positions=j.arange(tape.words.shape[1])[None,:];span=(positions>=start[:,None])&(positions<end[:,None])
 good &= j.all(~span|((l.claimed==claim[:,None])&j.all(l.words==tape.words,axis=2)&(tape.event_turn==l.turn[rows,slot,None])&(tape.event_phase==l.phase[rows,slot,None])&j.all(tape.event_generation==l.generation[rows,slot,None,:],axis=2)&j.all(tape.switch_details==0,axis=2)),axis=1)
 return good

def captured_matches(tape,captured):
 if captured is None:return j.zeros(len(tape.count),bool)
 p=captured.pending;c=tape.execution;rows=j.arange(len(tape.count));slot=j.clip(p.ticket.slot,0,c.ledger.site.shape[1]-1)
 return p.requested&p.valid&_bound(tape,p.context)&(captured.end==tape.count)&(c.ledger.serial[rows,slot]==p.ticket.serial)&owned(tape,c.ledger.start[rows,slot])
