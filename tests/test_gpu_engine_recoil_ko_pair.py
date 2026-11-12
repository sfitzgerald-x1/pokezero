"""Literal prospective helper transactions only; no engine/world/game execution."""
from _showdown_root import showdown_root_str
import ast
from pathlib import Path
from types import SimpleNamespace
import json,subprocess,unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_recoil_callbacks as recoil,engine_direct_ko_callbacks as ko
from gpu_engine import engine_direct_hit_controls as direct,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risks,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_events as events,engine_full_v4_event_device as full,engine_source_frame_witness as witness
from gpu_engine import engine_public_event_device as base
from gpu_engine import v4_encoder as v4,public_wire,engine_request_source as request
import test_gpu_engine_recoil_owning_event as fixture
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

ROOT=showdown_root_str()
class RecoilKOPairTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.RecoilOwningEventTests.setUpClass();cls.f=fixture.RecoilOwningEventTests()
  f=cls.f
  f.metadata['source_commit_label']='f76228a1354b5d0f307ca2d16101294ad3a2308b'
  f.metadata['compiled_tree']=json.loads(subprocess.check_output(['node','-e','console.log(JSON.stringify(require(process.argv[1]).compiledTree(process.argv[2])))',str(Path(request.__file__).with_name('gender_policy.cjs')),ROOT+'/dist'],text=True))
  f.producer=events.compile_catalog(f.metadata,f.table,public_moves=f.catalog.moves,dex=f.catalog.dex,exposure=fixture.fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(f.producer,expected_binding=f.producer.binding,expected_engine_binding=np.asarray(f.producer.engine_binding,np.uint32))
  f.adapter=full.compile_adapter(f.metadata,projector,catalog=f.catalog);f.account=ex.compile_accounting(f.adapter)
  f.direct=direct.compile_direct_hit_controls(engine_catalog=f.producer)
  f.helper=recoil.compile_recoil(engine_catalog=f.producer,metadata=f.metadata)
  f.context=recoil.Context(f.helper.device,direct.RegisteredContext(f.direct.device,f.account.device))
  f.kw=dict(catalog=f.producer,expected_binding=f.producer.binding,expected_engine_binding=np.asarray(f.producer.engine_binding,np.uint32))
  cls.rules=request.load_source_rules(ROOT,f.metadata)
  cls.helper=recoil.compile_recoil(engine_catalog=cls.f.producer,metadata=cls.f.metadata,source_rules=cls.rules)
  cls.context=recoil.Context(cls.helper.device,cls.f.context.direct)
  cls.risk=risks.compile_risks(cls.f.producer,cls.f.account,direct_hit_compiled=cls.f.direct,recoil_compiled=cls.helper)
 def initial(self,b=8,consume=False):
  f=self.f;s=f.state(b);rows=j.arange(b);side=rows%2;ap=s.active[rows,side];tp=s.active[rows,1-side]
  s=s._replace(hp=s.hp.at[rows,side,ap].set(200).at[rows,1-side,tp].set(60),max_hp=s.max_hp.at[rows,side,ap].set(300).at[rows,1-side,tp].set(100))
  c=life.register(j.arange(4801,4801+b,dtype=j.int32),**f.kw);before=life.snapshot(c);t=life._tape(c,f.producer,64)
  for owner in (0,1):t=life._reveal(t,s,j.full(b,owner,j.int32),j.full(b,-1,j.int32),s.active[:,owner],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool));consumers=[]
  if consume:
   frame=life.LifecycleResult(s,t.cursor,t,before,life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32))
   fold=jax.jit(lambda stream,p:full.project_and_fold(stream,frame,f.adapter.device,p,f.catalog))
   for owner in (0,1):
    stream=full.register(t.stream_id,j.full(b,owner,j.int32),compiled=f.adapter);p=v4.initialize(f.catalog,stream_ids=t.stream_id,owner_sides=j.full(b,owner,j.int32));got=fold(stream,p)
    self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  return s,t.cursor,tuple(consumers)
 def pending(self,b=8,*,state=None,cursor=None,change=None,context=None,mask=None,amount=None):
  f=self.f
  if state is None:state,cursor,_=self.initial(b)
  rows=j.arange(b);side=rows%2;ap=state.active[rows,side];tp=state.active[rows,1-side];move=j.full(b,f.move,j.int32);mask=j.ones(b,bool)if mask is None else mask;context=self.context if context is None else context
  t=life._tape(cursor,f.producer,64);t=ex.begin_on_tape(t,ex.initialize(cursor.stream_id,compiled=f.account),f.account.device,active=j.ones(b,bool))
  carry=risks.begin_frame_device(risks.initialize_device(t.stream_id,self.risk.device),t,self.risk.device,active=j.ones(b,bool));t=t._replace(execution=t.execution._replace(risks=risks.Context(carry,self.risk.device)))
  t=events.move(t,side,ap,tp,move,eligible=mask);t,ticket=ex.open_attempt(t,ex.Site.DIRECT_HIT,1-side,active=mask)
  w=direct.HitWrite(j.full(b,120,j.int32),j.ones(b,bool),j.ones(b,bool),j.zeros(b,bool),j.ones(b,j.int32),j.full(b,2,j.int32),j.zeros(b,bool),j.zeros(b,bool),j.zeros(b,bool),j.full(b,60,j.int32),j.zeros(b,j.int32),j.full(b,100,j.int32))
  before=recoil.KOBefore(context,move,j.zeros(b,j.int32),j.full(b,300,j.int32),j.zeros(b,j.int32),j.full(b,-1,j.int32))
  hp=state.hp[rows,side,ap];party=state.hp[rows,1-side]
  if change:w,before,hp,party=change(w,before,hp,party)
  t,p=ko.prepare_controls_device(t,ticket,side,ap,tp,move,w,context.direct,hp,party,eligible=mask,recoil_before=before)
  t=events.condition(t,1-side,tp,w.new_hp,w.maximum,j.zeros(b,j.int32),eligible=mask)
  amount=j.full(b,20,j.int32)if amount is None else amount;p=recoil.with_ko_amount(p,amount)
  captured=risks.CapturedDirectHit(context.direct,ticket,side,ap,tp,move,w)
  t,r=recoil.prepare_device(t,captured,amount,before.ability,hp,before.maximum,before.status,context,eligible=mask,target_ko=p)
  return t,p,r
 def after(self,t,p,r,*,write=None,foreign=False,faint=True):
  w=recoil.Write(p.actor_physical,p.target_physical,r.ability,r.old_hp,r.old_hp-r.recoil,r.maximum,r.status)if write is None else write
  out=recoil.record_device(t,r,w)
  if foreign:out=events.move(out,p.actor,p.actor_physical,p.target_physical,p.move,eligible=p.requested)
  if faint:out=events.faint(out,1-p.actor,p.target_physical,eligible=p.requested)
  party=p.target_party_before.at[j.arange(len(p.actor)),p.target_physical].set(0)
  return ko.finish_device(out,p,w.new_hp,party,w.actor_physical,w.target_physical,recoil_after=w)
 def test_literal_pending_final_partition_negative_guards_and_legacy_closed(self):
  t,p,r=self.pending();self.assertTrue(bool(recoil.matches_pending(t,r).all()));self.assertEqual(t.execution.ledger.stage[:,:2].tolist(),[[1,1]]*8)
  outcome=risks.Outcome(p.write.power,p.write.critical,p.write.hit,p.write.immune,p.write.numerator,p.write.denominator,j.zeros(8,bool),p.write.substitute,p.write.protected,p.write.enduring,r.recoil,j.zeros(8,bool))
  observed=risks.observe_outcome_device(t.execution.risks.carry,t,outcome,self.risk.device,eligible=j.ones(8,bool),active=j.ones(8,bool),captured_direct=r.captured,pending_recoil=r,pending_ko=p)
  self.assertEqual(observed.risk.tolist(),[0]*8)
  out=self.after(t,p,r);self.assertEqual(out.refusal.tolist(),[1]*8);self.assertEqual(out.execution.ledger.stage[:,:2].tolist(),[[2,2]]*8)
  self.assertEqual(out.words[0,:6,0].tolist(),[1,21,23,2,28,3]);self.assertEqual(out.execution.ledger.claimed[0,:6].tolist(),[0,1,1,1,2,1])
  self.assertTrue(bool(recoil.validate_owned_ko_control(out,1,out.words[:,1]).all()));self.assertTrue(bool(recoil.validate_owned_ko_control(out,2,out.words[:,2]).all()))
  for bad in (self.after(t,p,r,foreign=True),self.after(t,p,r,faint=False),self.after(t,p,r,write=recoil.Write(p.actor_physical,p.target_physical,r.ability+1,r.old_hp,r.old_hp-r.recoil,r.maximum,r.status)),self.after(t,p,r,write=recoil.Write(p.actor_physical,p.target_physical,r.ability,r.old_hp,r.old_hp-r.recoil,r.maximum,r.status+1))):
   self.assertTrue(bool(((bad.refusal&~1)!=0).all()));self.assertTrue(bool((bad.execution.ledger.refusal!=0).all()))
  for change in (lambda w,a,h,p:(w,a._replace(caller=a.caller-1),h,p),lambda w,a,h,p:(w,a._replace(ability=j.ones(8,j.int32)),h,p),lambda w,a,h,p:(w,a,h,j.zeros_like(p).at[j.arange(8),j.array([5,3]*4)].set(60)),lambda w,a,h,p:(w._replace(protected=j.ones(8,bool)),a,h,p)):
   bad,_,_=self.pending(change=change);self.assertTrue(bool(((bad.refusal&~1)!=0).all()))
  for amount in (j.zeros(8,j.int32),j.full(8,200,j.int32)):
   bad,_,_=self.pending(amount=amount);self.assertTrue(bool(((bad.refusal&~1)!=0).all()))
  legacy,_,_=self.pending(context=self.f.context);self.assertTrue(bool(((legacy.refusal&~1)!=0).all()));self.assertFalse(bool(self.f.helper.device.ko_source));self.assertTrue(bool(self.helper.device.ko_source))
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):recoil.validate_owned_ko_control(out._replace(**{name:j.tile(getattr(out,name),(8,1))}),1,out.words[:,1])
  forged=out._replace(words=out.words.at[0,4,7].set(0));self.assertFalse(bool(recoil.validate_owned_ko_control(forged,1,forged.words[:,1])[0]))
  # No earlier lifetime refusal can be cleared by this later finalization.
  sticky=t._replace(refusal=t.refusal|64);self.assertTrue(bool((self.after(sticky,p,r).refusal&64).all()))
 def test_both_original_full_consumers_exact_words_and_source_witness(self):
  b=2;s,c,consumers=self.initial(b,consume=True);t,p,r=self.pending(b,state=s,cursor=c);out=self.after(t,p,r)
  self.assertEqual(out.refusal.tolist(),[1,1]);ledger=out.execution.ledger
  wt=SimpleNamespace(risk=self.risk.device,direct_hit_binding=self.f.direct.device.binding,secondary_weather=j.zeros(len(self.risk.device.selected_risk),bool),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
  self.assertTrue(bool(witness._raw_complete(out,wt).all()))
  wrong=out._replace(execution=out.execution._replace(ledger=ledger._replace(claimed=ledger.claimed.at[:,4].set(1))))
  self.assertFalse(bool(witness._raw_complete(wrong,wt).any()))
  state=s._replace(hp=s.hp.at[j.arange(b),p.actor,p.actor_physical].set(180).at[j.arange(b),1-p.actor,p.target_physical].set(0))
  frame=life.LifecycleResult(state,out.cursor,out,life.snapshot(c),life.snapshot(out.cursor),j.zeros(b,bool),j.zeros(b,j.int32));closed=[]
  consume=jax.jit(lambda stream,pub:ex.consume_frame_device(ledger,frame,stream,pub,active=j.ones(b,bool),tables=self.f.account.device,adapter_tables=self.f.adapter.device,catalog=self.f.catalog))
  for owner,(stream,pub) in enumerate(consumers):
   end,result=consume(stream,pub);closed.append(end);self.assertEqual(result.projection.error.tolist(),[0]*b);self.assertTrue(bool(ex.report(end,tables=self.f.account.device).scoped_drained.all()))
   for row in range(b):np.testing.assert_array_equal(self.f.f.words(result.projection,row),self.reference_words(s,out,row,owner))
  _,good=ex.join_consumed_owners(*closed,tables=self.f.account.device);self.assertTrue(bool(good.all()))
 def reference_words(self,s,t,row,owner):
  ident={};names={};initial=[]
  for side in (0,1):
   species=int(self.f.adapter.device.causal.species[side+1]);names[side]=self.f.catalog.species.names[species];ident[side]=f'p{side+1}a: '+names[side]
   ap=int(s.active[row,side]);hp=int(s.hp[row,side,ap]);maximum=int(s.max_hp[row,side,ap]);health=f'{hp}/{maximum}'if side==owner else f'{min(99,(100*hp+maximum-1)//maximum)if hp<maximum else 100}/100'
   initial.append(f'|switch|{ident[side]}|{names[side]}, L80, M|{health}')
  wire,_=public_wire.project_lines(public_wire.WireState(),initial+['|turn|1'],**v4.wire_kwargs(self.f.catalog));lines=[]
  for raw in np.asarray(t.words[row,:int(t.count[row])]):
   kind=int(raw[0]);target=int(raw[3]);actor=int(raw[1])
   if kind==1:lines.append(f'|move|{ident[actor]}|Double-Edge|{ident[target]}')
   elif kind in (21,23):lines.append('|'+{21:'-crit',23:'-resisted'}[kind]+'|'+ident[target])
   elif kind==2:lines.append(f'|-damage|{ident[target]}|0 fnt')
   elif kind==28:
    hp=int(raw[7]);maximum=int(raw[8]);health=f'{hp}/{maximum}'if actor==owner else f'{min(99,(100*hp+maximum-1)//maximum)}/100'
    lines.append(f'|-damage|{ident[actor]}|{health}|[from] Recoil|[of] {ident[target]}')
   elif kind==3:lines.append('|faint|'+ident[target])
   else:self.fail('unexpected raw')
  _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.f.catalog))
  return np.asarray([e.words()for packet in packets for e in packet.events],np.int32)
 def test_default_source_and_capability_are_separate(self):
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  for file,names in [('engine_execution_completeness.py',('emitted','consume_frame_device')),('engine_direct_hit_controls.py',('validate_write_device',))]:
   old=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/'+file],text=True));new=ast.parse(Path('gpu_engine/'+file).read_text())
   for name in names:self.assertEqual(ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name)),ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)))

 def test_worker_forwards_same_rules_to_recoil_and_thunder_wave(self):
  tree=ast.parse(Path('src/pokezero/v5_worker.py').read_text())
  loop=next(n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['name','move','module','function'])
  calls=[]
  class Spy:
   def __getattr__(self,name):
    def call(**kwargs):calls.append((name,kwargs));return name
    return call
  original=object();catalog=object();metadata={'vocabulary':{'moves':['doubleedge','calmmind','swordsdance','spikes','thunderwave']}}
  env=dict(recoil=Spy(),calm=Spy(),sword=Spy(),spikes=Spy(),thunder=Spy(),lifecycle=catalog,metadata=metadata,rules=original,helpers={})
  exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'<original-worker-loop>','exec'),env)
  self.assertEqual(len(calls),5)
  for name,kw in calls:
   self.assertIs(kw['engine_catalog'],catalog);self.assertIs(kw['metadata'],metadata)
   self.assertEqual(set(kw),{'engine_catalog','metadata','source_rules'}if name in ('compile_recoil','compile_thunder_wave')else {'engine_catalog','metadata'})
   if name in ('compile_recoil','compile_thunder_wave'):self.assertIs(kw['source_rules'],original)
  calls.clear();metadata['vocabulary']['moves']=[]
  exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'<original-worker-loop>','exec'),env)
  self.assertEqual(calls,[])
 def test_original_writer_target_damage_recoil_then_faint_order(self):
  script=r"""
const path=require('path'),root=process.argv[1],{Dex}=require(path.join(root,'dist/sim/dex.js')),{Battle}=require(path.join(root,'dist/sim/battle.js'));
const dex=Dex.mod('gen3'),lines=[];
const battle={gen:3,gameType:'singles',dex,faintQueue:[],ended:false,clampIntRange:(x,min)=>Math.max(x,min),runEvent:(id,t,s,e,n)=>id==='Damage'?n:true,singleEvent:()=>{},add:(...xs)=>lines.push(xs.map(String).join('|')),queue:{cancelAction:()=>{}},checkWin:()=>false,getAllActive:()=>[]};
function mon(name,hp,max){return {hp,isActive:true,fainted:false,side:{pokemonLeft:6,totalFainted:0},toString:()=>name,get getHealth(){return this.hp?this.hp+'/'+max:'0 fnt'},damage(x,source,effect){const n=Math.min(x,this.hp);this.hp-=n;if(!this.hp)battle.faintQueue.push({target:this,source,effect});return n},getAbility:()=>({}),getItem:()=>({}),clearVolatile:()=>{}}}
const actor=mon('p1a: Snorlax',200,300),target=mon('p2a: Donphan',60,100),move=dex.moves.get('doubleedge');
Battle.prototype.spreadDamage.call(battle,[60],[target],actor,move);
const amount=dex.data.Scripts.actions.calcRecoilDamage.call({battle},60,move);
Battle.prototype.spreadDamage.call(battle,[amount],[actor],target,'recoil');
Battle.prototype.faintMessages.call(battle);
process.stdout.write(JSON.stringify({lines,actorHP:actor.hp,targetHP:target.hp}));
"""
  got=json.loads(subprocess.check_output(['node','-e',script,ROOT],text=True))
  self.assertEqual(got,{'lines':['-damage|p2a: Donphan|0 fnt','-damage|p1a: Snorlax|180/300|[from] Recoil|[of] p2a: Donphan','faint|p2a: Donphan'],'actorHP':180,'targetHP':0})
