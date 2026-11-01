"""Prospective ordinary Gen3 self-KO composite, no numerical assignments.

RAW55 [55,actor,physical,targetPhysical,move,oldActorHP,0,max,status,secondary].
Exact source span: RAW55, literal hit controls, target HP, actor faint, optional
 target faint. Both sides must retain a live bench; terminal outcomes stay closed.
"""
from pathlib import Path
from typing import NamedTuple
import hashlib,json,subprocess
import jax
import jax.numpy as j
import numpy as np
from . import engine_lifecycle_callbacks as life,engine_public_events as public
SELF_DESTRUCT=55
class Data(NamedTuple):
 binding:object
 engine_binding:object
 source_binding:object
 move_binding:object
 producer_binding:object
 candidates:object
 powers:object
 damp:object
 quiet_abilities:object
 quiet_items:object
class Before(NamedTuple):
 hp:object
 maximum:object
 status:object
 abilities:object
 items:object
 active:object
class Record(NamedTuple):
 requested:object
 self_ok:object
 target_ok:object
 finalized:object
 side:object
 move:object
 caller:object
 slot:object
 serial:object
 frame:object
 start:object
 before:object
 self_after:object
 write:object
 secondary:object

def compile_data(catalog,metadata,rules):
 from . import engine_request_source as request
 from .engine_private_team import engine_metadata_binding
 origin=request._rules_origin(rules)
 if not np.array_equal(engine_metadata_binding(metadata),catalog.engine_binding)or not np.array_equal(rules.device.engine_binding,catalog.engine_binding):raise ValueError('same original self-destruct metadata/rules required')
 script=r"""
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[1],meta=JSON.parse(fs.readFileSync(0,'utf8'));
const checked=require(process.argv[2]).build(root,meta),d=require(path.join(root,'dist/sim/dex.js')).Dex.mod('gen3');
const B=require(path.join(root,'dist/sim/battle.js')).Battle,P=require(path.join(root,'dist/sim/pokemon.js')).Pokemon,A=require(path.join(root,'dist/sim/battle-actions.js')).BattleActions;const hash=f=>crypto.createHash('sha256').update(f.toString()).digest('hex');
const callbacks={useMoveInner:hash(d.data.Scripts.actions.useMoveInner),faintMessages:hash(B.prototype.faintMessages),faint:hash(P.prototype.faint),getDamage:hash(d.data.Scripts.actions.getDamage||A.prototype.getDamage),tryMoveHit:hash(d.data.Scripts.actions.tryMoveHit||A.prototype.tryMoveHit)};
const quiet=e=>e.exists&&e.gen<=3&&!['onDamagingHit','onAfterDamage','onAfterMoveSecondary','onAfterMoveSecondarySelf','onUpdate','onAfterSubDamage','onEnd','onFaint','onBeforeFaint','onAfterFaint'].some(k=>e[k]!==undefined);
process.stdout.write(JSON.stringify({callbacks,source_tree:checked.source_tree,compiled_tree:checked.compiled_tree,moves:['explosion','selfdestruct'].map(n=>{const m=d.moves.get(n);return {id:m.id,gen:m.gen,power:m.basePower,type:m.type,category:m.category,target:m.target,accuracy:m.accuracy,priority:m.priority,selfdestruct:m.selfdestruct,callbacks:Object.keys(m).filter(k=>(/^on/.test(k)||/Callback$/.test(k))&&m[k]!==undefined),exotic:!!(m.multihit||m.recoil||m.drain||m.secondary||m.secondaries||m.self||m.boosts||m.status||m.volatileStatus||m.forceSwitch||m.selfSwitch)}}),abilities:meta.vocabulary.abilities.map(n=>quiet(d.abilities.get(n))),items:meta.vocabulary.items.map((n,i)=>!i||quiet(d.items.get(n)))}));"""
 v=json.loads(subprocess.check_output([origin.node,'-e',script,origin.root,str(Path(request.__file__).with_name('request_source_rules.cjs'))],input=json.dumps(metadata).encode(),timeout=30))
 if v['callbacks']!={'useMoveInner': '8c35e507bccb4bc25d13e24d582895097cd14ea6b24630b163e6ffafc7b4a160', 'faintMessages': '62f5ba2782de2f0716c5071a3555ac238a430105b1e2e04e8b96b9cb6c9e5c4b', 'faint': '2ba9a893c499af2d32dd02dcfc255a63ba03f9308cf6fa2f51f98940dc315a28', 'getDamage': 'dfd031900da8c77e5e1a2721858d8d4a7804bc7def1b7a1d486220521a6a7f12', 'tryMoveHit': 'af529997d52b6d856301bc8f958ead5963f68abc58d9568c82df24f212b0a97f'}:raise ValueError('exact original Gen3 self-KO/damage/faint callbacks required')
 if json.dumps(v['source_tree'],sort_keys=True)!=origin.source_tree_json or json.dumps(v['compiled_tree'],sort_keys=True)!=origin.compiled_tree_json:raise ValueError('original self-destruct source drift')
 names=metadata['vocabulary']['moves'];t=catalog.engine_table;c=np.zeros(len(names),bool);powers=np.zeros(len(names),np.int32)
 for row in v['moves']:
  expected=250 if row['id']=='explosion'else 200
  if (row['gen'],row['power'],row['type'],row['category'],row['target'],row['accuracy'],row['priority'],row['selfdestruct'],row['callbacks'],row['exotic'])!=(1,expected,'Normal','Physical','allAdjacent',100,0,'always',[],False):raise ValueError('original canonical Gen3 self-destruct source required')
  for move,name in enumerate(names):
   if name!=row['id']:continue
   if not(move>0 and int(catalog.ids[move])>0 and int(catalog.targets[move])==1 and t.self_destruct is not None and bool(t.self_destruct.move[move])and int(t.power[move])==expected and bool(t.physical[move])and int(t.type[move])==1 and int(t.accuracy[move])==100 and not bool(t.always_hit[move])):raise ValueError('canonical installed self-destruct row required')
   c[move]=True;powers[move]=expected
 damp=np.asarray([n=='damp'for n in metadata['vocabulary']['abilities']],bool)
 if t.self_destruct is None or not np.array_equal(t.self_destruct.damp,damp):raise ValueError('original Damp guard required')
 payload=dict(schema='pokezero.self-destruct.composite.v1',source=v,candidates=c.tolist(),powers=powers.tolist(),engine=catalog.engine_binding,moves=catalog.move_binding,producer=catalog.binding)
 words=lambda x:j.asarray(x,j.uint32)
 return Data(words(np.frombuffer(hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).digest(),dtype='>u4').astype(np.uint32)),words(catalog.engine_binding),words(catalog.source_binding),words(catalog.move_binding),words(np.frombuffer(bytes.fromhex(catalog.binding),dtype='>u4').astype(np.uint32)),j.asarray(c),j.asarray(powers),j.asarray(damp),j.asarray(v['abilities']),j.asarray(v['items']))
def shape(d):
 if type(d)is not Data:raise ValueError('typed self-destruct capability required')
 for v in d[:5]:life._typed(v,(8,),j.uint32)
 for v in (d.candidates,d.damp,d.quiet_abilities,d.quiet_items):life._typed(v,(len(v),),j.bool_)
 life._typed(d.powers,d.candidates.shape,j.int32)
def _data(tape):
 t=tape.execution.risks.tables
 return Data(t.self_destruct_binding,t.engine_binding,t.source_binding,t.move_binding,t.producer_binding,t.self_destruct_candidates,t.self_destruct_powers,t.self_destruct_damp,t.self_destruct_abilities,t.self_destruct_items)
def _bound(tape,d):
 from . import engine_execution_completeness as ex
 shape(d);ctx=ex._context(tape)
 if ctx is None or ctx.risks is None:return j.zeros(len(tape.count),bool)
 good=ex._validate(ctx.ledger,ctx.tables)&ex._tape(ctx.ledger,tape,ctx.tables)&j.any(d.binding!=0)
 for n in ('engine_binding','source_binding','move_binding','producer_binding'):
  life._typed(getattr(tape,n),(8,),j.uint32);good &= j.all(getattr(tape,n)==getattr(d,n))
 actual=_data(tape);shape(actual)
 for a,b in zip(actual,d,strict=True):good &= j.all(a==b)
 return good
def _put(tape,rec):
 old=tape.execution.self_destruct
 if old is not None:rec=jax.tree.map(lambda a,b:j.where(rec.requested.reshape((len(rec.requested),)+(1,)*(a.ndim-1)),b,a),old,rec)
 return tape._replace(execution=tape.execution._replace(self_destruct=rec))
def snapshot(s):
 r=j.arange(len(s.hp));sides=j.arange(2)[None,:]
 return Before(s.hp,s.max_hp,s.status,s.ability[r[:,None],sides,s.active],s.item[r[:,None],sides,s.active],s.active)
def equal(a,b):
 out=j.ones(len(a[0]),bool)
 for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):out &= j.all(x==y,axis=tuple(range(1,x.ndim)))if x.ndim>1 else x==y
 return out
def _literal(rec,b):
 from . import engine_direct_hit_controls as hit
 if type(rec)is not Record or type(rec.before)is not Before:raise ValueError('typed same-call self-destruct receipt required')
 for n in ('requested','self_ok','target_ok','finalized','secondary'):life._typed(getattr(rec,n),(b,),j.bool_)
 for n in ('side','move','caller','slot','serial','frame','start'):life._typed(getattr(rec,n),(b,),j.int32)
 for n in ('hp','maximum','status'):life._typed(getattr(rec.before,n),(b,2,6),j.int32)
 for n in ('abilities','items','active'):life._typed(getattr(rec.before,n),(b,2),j.int32)
 life._typed(rec.self_after,(b,2,6),j.int32);hit._typed_write(rec.write,b)
def begin(tape,state,side,move,caller,context,*,requested):
 from . import engine_execution_completeness as ex,engine_direct_hit_controls as hit
 if context is None or context.data.self_destruct is None:
  return ex.unhandled_attempt(tape,ex.Site.SELF_DESTRUCT,side,active=requested),None
 d=context.data.self_destruct;b=len(side);r=j.arange(b);seat=j.clip(side,0,1);before=snapshot(state);active=before.active;physical=active[r,seat];target=active[r,1-seat];old=before.hp[r,seat,physical]
 good=hit._registered_bound(tape,context)&_bound(tape,d)&(caller==move)&(move>0)&(move<len(d.candidates))&d.candidates[j.clip(move,0,len(d.candidates)-1)]&(side>=0)&(side<2)&(old>0)&(old<=before.maximum[r,seat,physical])
 good &= ~j.any(d.damp[j.clip(before.abilities,0,len(d.damp)-1)]&(before.hp[r[:,None],j.arange(2)[None,:],active]>0),axis=1)
 for s in (0,1):
  ability=before.abilities[:,s];item=before.items[:,s];good &= (ability>0)&(ability<len(d.quiet_abilities))&d.quiet_abilities[j.clip(ability,0,len(d.quiet_abilities)-1)]&(item>=0)&(item<len(d.quiet_items))&d.quiet_items[j.clip(item,0,len(d.quiet_items)-1)]
 good &= j.all(j.any((before.hp>0)&(j.arange(6)[None,None,:]!=active[:,:,None]),axis=2),axis=1)
 tape,ticket=ex.open_attempt(tape,ex.Site.SELF_DESTRUCT,side,active=requested);c=ex._context(tape);slot,_=ex._ticket(c.ledger,ticket);start=c.ledger.start[r,slot];at=j.clip(start-1,0,tape.words.shape[1]-1);raw=tape.words[r,at]
 good &= (start>0)&(raw[:,0]==public.MOVE)&(raw[:,1]==side)&(raw[:,2]==physical)&(raw[:,3]==1-side)&(raw[:,4]==target)&(raw[:,5]==move)&j.all(raw[:,6:]==0,axis=1)
 good &= (tape.event_turn[r,at]==c.ledger.turn[r,slot])&(tape.event_phase[r,at]==0)&j.all(tape.event_generation[r,at]==c.ledger.generation[r,slot],axis=1)
 z=j.zeros(b,j.int32);no=j.zeros(b,bool);w=hit.HitWrite(z,no,no,no,z,z,no,no,no,z,z,z)
 rec=Record(requested,requested&good,no,no,side,move,caller,slot,ticket.serial,ticket.frame,start,before,before.hp,w,no)
 return _put(tape,rec),rec
def self_write(tape,rec,state):
 if rec is None:return tape,None
 b=len(rec.side);r=j.arange(b);side=j.clip(rec.side,0,1);physical=rec.before.active[r,side];expected=rec.before.hp.at[r,side,physical].set(0)
 before=rec.before._replace(hp=expected);good=rec.self_ok&equal(before,snapshot(state));rec=rec._replace(self_ok=good,self_after=state.hp)
 facts=(rec.move,rec.before.hp[r,side,physical],j.zeros(b,j.int32),rec.before.maximum[r,side,physical],rec.before.status[r,side,physical],j.zeros(b,j.int32))
 tape=public._append(tape,SELF_DESTRUCT,rec.side,physical,rec.before.active[r,1-side],rec.move,(facts[1],facts[2],facts[3],facts[4],facts[5]),good)
 return _put(tape,rec),rec
def _before_valid(rec,d):
 b=len(rec.side);r=j.arange(b);side=j.clip(rec.side,0,1);active=rec.before.active;old=rec.before.hp[r,side,active[r,side]]
 good=(rec.caller==rec.move)&(rec.move>0)&(rec.move<len(d.candidates))&d.candidates[j.clip(rec.move,0,len(d.candidates)-1)]&(rec.side>=0)&(rec.side<2)&(old>0)&j.all((active>=0)&(active<6),axis=1)&j.all((rec.before.hp>=0)&(rec.before.hp<=rec.before.maximum)&(rec.before.maximum>0)&(rec.before.maximum<=1000000)&(rec.before.status>=0)&(rec.before.status<=6),axis=(1,2))
 good &= j.all(j.any((rec.before.hp>0)&(j.arange(6)[None,None,:]!=active[:,:,None]),axis=2),axis=1)
 for seat in (0,1):
  a=rec.before.abilities[:,seat];i=rec.before.items[:,seat];good &= (a>0)&(a<len(d.damp))&(a<len(d.quiet_abilities))&~d.damp[j.clip(a,0,len(d.damp)-1)]&d.quiet_abilities[j.clip(a,0,len(d.quiet_abilities)-1)]&(i>=0)&(i<len(d.quiet_items))&d.quiet_items[j.clip(i,0,len(d.quiet_items)-1)]
 expected=rec.before.hp.at[r,side,active[r,side]].set(0)
 return good&j.all(expected==rec.self_after,axis=(1,2))
def _source(tape,rec):
 from . import engine_execution_completeness as ex
 b=len(tape.count);r=j.arange(b);l=tape.execution.ledger;slot=j.clip(rec.slot,0,l.site.shape[1]-1);side=j.clip(rec.side,0,1);actor=rec.before.active[r,side];target=rec.before.active[r,1-side]
 ticket=ex.Ticket(tape.stream_id,tape.execution.tables.inventory_binding,rec.frame,rec.slot,rec.serial);_,same=ex._ticket(l,ticket)
 good=same&(rec.slot>=0)&(rec.slot<l.count)&(l.site[r,slot]==40)&(l.side[r,slot]==rec.side)&(l.start[r,slot]==rec.start)&life._valid(tape.cursor)&tape.cursor.initialized
 good &= (tape.cursor.turn==l.turn[r,slot])&(tape.cursor.phase==0)&(tape.cursor.phase==l.phase[r,slot])&j.all(tape.cursor.generation==l.generation[r,slot],axis=1)&j.all(tape.cursor.physical_active==rec.before.active,axis=1)
 at=j.clip(rec.start-1,0,tape.words.shape[1]-1);raw=tape.words[r,at]
 good &= (rec.start>0)&(raw[:,0]==public.MOVE)&(raw[:,1]==rec.side)&(raw[:,2]==actor)&(raw[:,3]==1-rec.side)&(raw[:,4]==target)&(raw[:,5]==rec.move)&j.all(raw[:,6:]==0,axis=1)
 good &= (tape.event_turn[r,at]==l.turn[r,slot])&(tape.event_phase[r,at]==l.phase[r,slot])&j.all(tape.event_generation[r,at]==l.generation[r,slot],axis=1)
 expected=j.stack((j.full(b,SELF_DESTRUCT,j.int32),rec.side,actor,target,rec.move,rec.before.hp[r,side,actor],j.zeros(b,j.int32),rec.before.maximum[r,side,actor],rec.before.status[r,side,actor],rec.secondary.astype(j.int32)),axis=1)
 return good&(tape.count>=rec.start+1)&j.all(tape.words[r,j.clip(rec.start,0,tape.words.shape[1]-1)]==expected,axis=1)
def self_matches(tape,rec):
 if rec is None:return j.zeros(len(tape.count),bool)
 _literal(rec,len(tape.count));return _bound(tape,_data(tape))&rec.self_ok&_before_valid(rec,_data(tape))&_source(tape,rec)

def _target_valid(rec,d):
 w=rec.write;b=len(rec.side);r=j.arange(b);side=j.clip(rec.side,0,1);target=rec.before.active[r,1-side]
 from . import engine_direct_hit_controls as hit
 return (w.power==d.powers[j.clip(rec.move,0,len(d.powers)-1)])&w.hit&~w.immune&~w.substitute&~w.protected&~w.enduring&hit._positive_ratio(w.numerator,w.denominator)&(w.old_hp>w.new_hp)&(w.new_hp>=0)&(w.old_hp==rec.before.hp[r,1-side,target])&(w.maximum==rec.before.maximum[r,1-side,target])
def target_write(tape,rec,write,*,secondary):
 from . import engine_direct_hit_controls as hit
 if rec is None:return tape
 b=len(rec.side);_literal(rec,b);hit._typed_write(write,b);d=_data(tape);r=j.arange(b);side=j.clip(rec.side,0,1);target=rec.before.active[r,1-side]
 good=self_matches(tape,rec)&(tape.count==rec.start+1)&(write.power==d.powers[j.clip(rec.move,0,len(d.powers)-1)])&write.hit&~write.immune&~write.substitute&~write.protected&~write.enduring&hit._positive_ratio(write.numerator,write.denominator)&(write.old_hp>write.new_hp)&(write.new_hp>=0)&(write.old_hp==rec.before.hp[r,1-side,target])&(write.maximum==rec.before.maximum[r,1-side,target])
 critical,effect=hit.annotation_flags_device(write,data=None)
 facts=(rec.move,write.old_hp,write.new_hp,write.numerator,write.denominator)
 for kind,flag in ((hit.CRITICAL,critical),(hit.SUPER_EFFECTIVE,effect&(write.numerator>write.denominator)),(hit.RESISTED,effect&(write.numerator<write.denominator))):tape=public._append(tape,kind,rec.side,rec.before.active[r,side],1-rec.side,target,facts,good&flag)
 rec=rec._replace(target_ok=good,write=write,secondary=secondary)
 # The original dispatch fact is captured literally in the already written header.
 at=j.clip(rec.start,0,tape.words.shape[1]-1);words=tape.words.at[r,at,9].set(j.where(good,secondary.astype(j.int32),tape.words[r,at,9]));tape=tape._replace(words=words)
 return _put(tape,rec)
def _span(tape,rec,*,complete):
 from . import engine_direct_hit_controls as hit,engine_execution_completeness as ex
 b=len(tape.count);_literal(rec,b);r=j.arange(b);l=tape.execution.ledger;slot=j.clip(rec.slot,0,l.site.shape[1]-1);side=j.clip(rec.side,0,1);actor=rec.before.active[r,side];target=rec.before.active[r,1-side];w=rec.write
 good=self_matches(tape,rec)&rec.target_ok&_target_valid(rec,_data(tape))&rec.requested&(rec.frame==l.frame)&(l.site[r,slot]==40)&(l.side[r,slot]==rec.side)&(l.serial[r,slot]==rec.serial)&(l.start[r,slot]==rec.start)
 critical,effect=hit.annotation_flags_device(w,data=None);controls=critical.astype(j.int32)+effect.astype(j.int32);ko=w.new_hp==0;length=2+controls+(1+ko.astype(j.int32)if complete else 0);good &= tape.count==rec.start+length if not complete else rec.start+length<=tape.count
 expected=j.stack((j.full(b,SELF_DESTRUCT,j.int32),rec.side,actor,target,rec.move,rec.before.hp[r,side,actor],j.zeros(b,j.int32),rec.before.maximum[r,side,actor],rec.before.status[r,side,actor],rec.secondary.astype(j.int32)),axis=1);good &= j.all(tape.words[r,j.clip(rec.start,0,tape.words.shape[1]-1)]==expected,axis=1)
 for offset,flag,kind in ((j.ones(b,j.int32),critical,j.full(b,hit.CRITICAL,j.int32)),(1+critical.astype(j.int32),effect,j.where(w.numerator>w.denominator,hit.SUPER_EFFECTIVE,hit.RESISTED))):
  raw=tape.words[r,j.clip(rec.start+offset,0,tape.words.shape[1]-1)];want=j.stack((kind,rec.side,actor,1-rec.side,target,rec.move,w.old_hp,w.new_hp,w.numerator,w.denominator),axis=1);good &= ~flag|j.all(raw==want,axis=1)
 hp=tape.words[r,j.clip(rec.start+1+controls,0,tape.words.shape[1]-1)];good &= (hp[:,0]==public.MAIN_HP)&j.all(hp[:,1:3]==-1,axis=1)&(hp[:,3]==1-rec.side)&(hp[:,4]==target)&(hp[:,5]==w.new_hp)&(hp[:,6]==w.maximum)&(hp[:,7]==rec.before.status[r,1-side,target])&j.all(hp[:,8:]==0,axis=1)
 if complete:
  for n,who,physical,use in ((0,rec.side,actor,j.ones(b,bool)),(1,1-rec.side,target,ko)):
   raw=tape.words[r,j.clip(rec.start+2+controls+n,0,tape.words.shape[1]-1)];good &= ~use|((raw[:,0]==public.DIRECT_FAINT)&j.all(raw[:,1:3]==-1,axis=1)&(raw[:,3]==who)&(raw[:,4]==physical)&j.all(raw[:,5:]==0,axis=1))
 span=(j.arange(tape.words.shape[1])[None,:]>=rec.start[:,None])&(j.arange(tape.words.shape[1])[None,:]<(rec.start+length)[:,None]);good &= j.all(~span|((tape.event_turn==l.turn[r,slot,None])&(tape.event_phase==l.phase[r,slot,None])&j.all(tape.event_generation==l.generation[r,slot,None,:],axis=2)),axis=1)
 return good,length

def matches_outcome(tape,o):
 rec=tape.execution.self_destruct if tape.execution is not None else None
 if rec is None:return j.zeros(len(tape.count),bool)
 good,_=_span(tape,rec,complete=False)
 for n in ('power','critical','hit','immune','numerator','denominator','substitute','protected','enduring'):good &= getattr(o,n)==getattr(rec.write,n)
 return good&o.explodes&~o.absorbed&(o.recoil==0)
def actor_faint(tape,state):
 rec=tape.execution.self_destruct
 if rec is None:return tape
 b=len(tape.count);r=j.arange(b);side=j.clip(rec.side,0,1);physical=rec.before.active[r,side]
 good,_=_span(tape,rec,complete=False)
 good &= (state.hp[r,side,physical]==0)&~rec.finalized
 return public.faint(tape,rec.side,physical,eligible=good)
def finish(tape,state):
 from . import engine_execution_completeness as ex
 rec=tape.execution.self_destruct
 if rec is None:return tape
 good,length=_span(tape,rec,complete=True);b=len(tape.count);r=j.arange(b);side=j.clip(rec.side,0,1);target=rec.before.active[r,1-side];expected=rec.self_after.at[r,1-side,target].set(rec.write.new_hp)
 good &= equal(rec.before._replace(hp=expected),snapshot(state))&(tape.count==rec.start+length)
 ticket=ex.Ticket(tape.stream_id,tape.execution.tables.inventory_binding,rec.frame,rec.slot,rec.serial);c=ex._context(tape);ledger=ex.unhandled(c.ledger,ticket,active=rec.requested&~rec.finalized&~good,tables=c.tables);ledger=ex.emitted(ledger,ticket,tape,active=rec.requested&~rec.finalized&good,tables=c.tables);tape=ex._put_context(tape,ledger,c)
 return tape._replace(execution=tape.execution._replace(self_destruct=rec._replace(finalized=rec.finalized|good)))
def owned(tape,index):
 if tape.execution is None or tape.execution.self_destruct is None:return j.zeros(len(tape.count),bool)
 rec=tape.execution.self_destruct;good,length=_span(tape,rec,complete=True);l=tape.execution.ledger;r=j.arange(len(tape.count));slot=j.clip(rec.slot,0,l.site.shape[1]-1);positions=j.arange(tape.words.shape[1])[None,:];span=(positions>=rec.start[:,None])&(positions<(rec.start+length)[:,None]);claim=rec.slot+1
 joined=(l.claimed==claim[:,None])&j.all(l.words==tape.words,axis=2)&(l.event_turn==tape.event_turn)&(l.event_phase==tape.event_phase)&j.all(l.event_generation==tape.event_generation,axis=2)
 return good&rec.finalized&(l.stage[r,slot]==2)&(l.end[r,slot]==rec.start+length)&(index>=rec.start)&(index<rec.start+length)&j.all(~span|joined,axis=1)
