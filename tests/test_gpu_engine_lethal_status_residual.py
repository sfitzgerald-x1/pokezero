"""Literal residual helper transactions; never execute a turn/game/model."""
import ast,json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_status_residual_callbacks as status,engine_terminal_callbacks as terminal
from gpu_engine import engine_direct_hit_controls as direct,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risks,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_events as events,engine_full_v4_event_device as full
from gpu_engine import engine_leftovers_callbacks as leftovers,engine_source_frame_witness as witness
from gpu_engine import engine_weather_speed_profile as profile,v4_encoder as v4,public_wire
from gpu_engine.state import empty_state
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
import test_gpu_engine_status_residual_singleton_queue as fixture

ROOT=fixture.ROOT
class LethalStatusResidualTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixture.SingletonStatusQueueTests.setUpClass();cls.f=fixture.SingletonStatusQueueTests()
  cls.direct=direct.compile_direct_hit_controls(engine_catalog=cls.f.f.producer)
  cls.context=direct.RegisteredContext(cls.direct.device,cls.f.account.device)
  cls.risk=risks.compile_risks(cls.f.f.producer,cls.f.account,direct_hit_compiled=cls.direct)
  cls.leftovers=leftovers.compile_leftovers(cls.f.f.metadata,engine_catalog=cls.f.f.producer)
 def initial(self,b=8,*,consume=False,last=True):
  f=self.f;s=empty_state(b);rows=j.arange(b);side=rows%2;ability=f.f.metadata['vocabulary']['abilities'].index('pressure');item=f.f.metadata['vocabulary']['items'].index('leftovers')
  s=s._replace(done=j.zeros(b,bool),active=j.broadcast_to(j.array([3,5],j.int32),(b,2)),species=j.ones_like(s.species).at[:,1].set(2),hp=j.full_like(s.hp,255),max_hp=j.full_like(s.max_hp,255),stats=j.full_like(s.stats,151),level=j.full_like(s.level,80),gender=j.ones_like(s.gender),ability=j.full_like(s.ability,ability),base_ability=j.full_like(s.base_ability,ability),item=j.full_like(s.item,item))
  streams=j.arange(6801,6801+b,dtype=j.int32);record=profile.qualify_birth(s,streams,profile=f.profile);self.assertTrue(bool(record.qualified_birth.all()))
  hp=s.hp.at[rows,side,s.active[rows,side]].set(16)
  if last:hp=j.where((j.arange(2)[None,:,None]==side[:,None,None])&(j.arange(6)[None,None,:]!=s.active[rows,side,None,None]),0,hp)
  s=s._replace(hp=hp,status=s.status.at[rows,side,s.active[rows,side]].set(4),status_counter=s.status_counter.at[rows,side,s.active[rows,side]].set(3));record=profile.retain_device(record,s,streams,f.profile.device)
  kw=dict(catalog=f.f.producer,expected_binding=f.f.producer.binding,expected_engine_binding=np.asarray(f.f.producer.engine_binding,np.uint32));c=life.register(streams,**kw);before=life.snapshot(c);t=life._tape(c,f.f.producer,64)
  for owner in (0,1):t=life._reveal(t,s,j.full(b,owner,j.int32),j.full(b,-1,j.int32),s.active[:,owner],eligible=j.ones(b,bool),replacement=j.zeros(b,bool),baton=j.zeros(b,bool),initial=True)
  t=t._replace(cursor=t.cursor._replace(initialized=j.ones(b,bool)));t=life._turn(t,j.ones(b,bool));consumers=[]
  if consume:
   frame=life.LifecycleResult(s,t.cursor,t,before,life.snapshot(t.cursor),j.zeros(b,bool),j.zeros(b,j.int32));fold=jax.jit(lambda stream,pub:full.project_and_fold(stream,frame,f.f.compiled.device,pub,f.f.catalog))
   for owner in (0,1):
    stream=full.register(streams,j.full(b,owner,j.int32),compiled=f.f.compiled);pub=v4.initialize(f.f.catalog,stream_ids=streams,owner_sides=j.full(b,owner,j.int32));got=fold(stream,pub);self.assertEqual(got.projection.error.tolist(),[0]*b);consumers.append((got.projection.state,got.public))
  return s,t.cursor,record,tuple(consumers)
 def pending(self,b=8,*,state=None,cursor=None,legacy=False,observer=True,prior=0,last=True):
  if state is None:state,cursor,_,_=self.initial(b,last=last)
  rows=j.arange(b);side=rows%2;phys=state.active[rows,side];data=self.f.legacy.device if legacy else self.f.helper.device
  t=life._tape(cursor,self.f.f.producer,64)._replace(refusal=j.full(b,1|prior,j.int32));t=ex.begin_on_tape(t,ex.initialize(cursor.stream_id,compiled=self.f.account),self.f.account.device,active=j.ones(b,bool));carry=risks.begin_frame_device(risks.initialize_device(t.stream_id,self.risk.device),t,self.risk.device,active=j.ones(b,bool));t=t._replace(execution=t.execution._replace(risks=risks.Context(carry,self.risk.device)))
  if observer:t=terminal.begin_on_tape(t,state,self.context,active=j.ones(b,bool),status_data=data)
  t=life.action_end(t,j.ones(b,bool));t,ticket=ex.open_attempt(t,ex.Site.LEFTOVERS,side,active=j.ones(b,bool))
  t=leftovers.record_leftovers_heal_device(t,side,phys,self.leftovers.device.leftovers_item,j.full(b,16,j.int32),j.full(b,31,j.int32),j.full(b,255,j.int32),j.full(b,4,j.int32),self.leftovers.device,eligible=True,_defer_own_unknown=True)
  t=ex.capture_leftovers_write(t,ticket,requested=j.ones(b,bool),positive_write=j.ones(b,bool))
  before=state._replace(hp=state.hp.at[rows,side,phys].set(31));t,ticket=ex.open_attempt(t,ex.Site.STATUS_RESIDUAL,side,active=j.ones(b,bool));p=status.begin_lethal_device(t,before,ticket,side,data,requested=j.ones(b,bool))
  after=before._replace(hp=before.hp.at[rows,side,phys].set(0),status_counter=before.status_counter.at[rows,side,phys].set(4))
  return t,status.LethalWrite(p,after,j.ones(b,bool))
 def emit(self,t,cap,*,finish=True):
  p=cap.pending;b=len(t.count);rows=j.arange(b);side=j.clip(p.side,0,1);phys=j.clip(p.physical,0,5)
  out=status.record_status_residual_device(t,p.side,p.physical,p.hp[rows,side,phys],cap.after.hp[rows,side,phys],p.maximum[rows,side,phys],p.status[rows,side,phys],p.counter[rows,side,phys],cap.after.status_counter[rows,side,phys],p.data,eligible=j.ones(b,bool),_defer_own_unknown=cap.eligible,_lethal_write=cap)
  return status.finish_lethal_device(out,cap)if finish else out
 def finalize(self,t,cap,*,last=True):
  b=len(t.count);state=cap.after._replace(done=j.full(b,last,bool),winner=1-cap.pending.side if last else j.full(b,-1,j.int32))
  return terminal.capture_final_on_tape(t,state,active=j.ones(b,bool)),state
 def test_same_write_terminal_and_nonterminal_guards_are_exact(self):
  t,cap=self.pending();self.assertTrue(bool(status.lethal_matches(t,cap).all()));out=self.emit(t,cap);self.assertEqual(out.refusal.tolist(),[1]*8);self.assertEqual(out.words[0,:4,0].tolist(),[7,17,20,3]);self.assertEqual(out.execution.ledger.stage[:,:2].tolist(),[[2,2]]*8)
  self.assertTrue(bool(status.owned_lethal(out,2,self.f.helper.device).all()));self.assertTrue(bool(status.owned_lethal(out,3,self.f.helper.device,faint=True).all()))
  final,state=self.finalize(out,cap);self.assertTrue(bool(terminal.matches_final_device(final,state).all()))
  # Frame-start16 and actual same-assignment31 are deliberately different.
  self.assertEqual(final.execution.terminal.before_hp[0,0,3].item(),16);self.assertEqual(final.execution.terminal.status.before_hp[0,0,3].item(),31)
  non,write=self.pending(last=False);non=self.emit(non,write);non,state=self.finalize(non,write,last=False);self.assertFalse(bool(terminal.matches_final_device(non,state).any()));self.assertEqual(non.refusal.tolist(),[1]*8)
  for field,value in [('hp',cap.after.hp.at[0,1,0].add(-1)),('max_hp',cap.after.max_hp.at[0,0,0].add(1)),('status',cap.after.status.at[0,0,3].set(0)),('status_counter',cap.after.status_counter.at[0,0,3].set(5)),('active',cap.after.active.at[0,0].set(0))]:
   changed=cap._replace(after=cap.after._replace(**{field:value}));self.assertFalse(bool(status.lethal_matches(t,changed)[0]),field)
  missing=self.emit(t,cap,finish=False);self.assertEqual(missing.execution.ledger.stage[:,1].tolist(),[1]*8);self.assertFalse(bool(terminal.matches_final_device(*self.finalize(missing,cap)).any()))
  foreign=events.move(missing,cap.pending.side,cap.pending.physical,j.full(8,5,j.int32),j.ones(8,j.int32),eligible=j.ones(8,bool));bad=status.finish_lethal_device(foreign,cap);self.assertTrue(bool((bad.refusal&256).all()))
  for field in ('engine_binding','source_binding','move_binding','producer_binding'):
   with self.assertRaises(ValueError):status.lethal_matches(t._replace(**{field:j.tile(getattr(t,field),(8,1))}),cap)
  legacy,lc=self.pending(legacy=True);self.assertTrue(bool((self.emit(legacy,lc).refusal&256).all()))
  sticky,sc=self.pending(prior=128);self.assertTrue(bool((self.emit(sticky,sc).refusal&128).all()))
  absent,ac=self.pending(observer=False);out,state=self.finalize(self.emit(absent,ac),ac);self.assertFalse(bool(terminal.matches_final_device(out,state).any()))
  wrong=final._replace(execution=final.execution._replace(terminal=final.execution.terminal._replace(final_winner=cap.pending.side)));self.assertFalse(bool(terminal.matches_final_device(wrong,state).any()))
 def test_original_both_full_all_words_and_site_routing(self):
  b=2;s,c,record,consumers=self.initial(b,consume=True);t,cap=self.pending(b,state=s,cursor=c);out,state=self.finalize(self.emit(t,cap),cap);record=profile.retain_device(record,state,c.stream_id,self.f.profile.device)
  frame=life.LifecycleResult(state,out.cursor,out,life.snapshot(c),life.snapshot(out.cursor),j.zeros(b,bool),j.zeros(b,j.int32));closed=[]
  wt=SimpleNamespace(risk=self.risk.device,direct_hit_binding=self.direct.device.binding,secondary_weather=j.zeros(len(self.risk.device.selected_risk),bool),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32),status=self.f.helper.device)
  self.assertTrue(bool(witness._raw_complete(out,wt).all()))
  consume=jax.jit(lambda stream,pub:ex.consume_frame_device(out.execution.ledger,frame,stream,pub,active=j.ones(b,bool),tables=self.f.account.device,adapter_tables=self.f.f.compiled.device,catalog=self.f.f.catalog,weather_hp=(record,self.f.profile.device),leftovers=self.leftovers.device,status_residual=self.f.helper.device))
  for owner,(stream,pub)in enumerate(consumers):
   ledger,result=consume(stream,pub);closed.append(ledger);self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertTrue(bool(ex.report(ledger,tables=self.f.account.device).scoped_drained.all()))
   for row in range(b):
    names={i:self.f.f.catalog.species.names[int(self.f.f.compiled.device.causal.species[i+1])]for i in(0,1)};lines=[]
    for i in(0,1):lines.append(f'|switch|p{i+1}a: {names[i]}|{names[i]}, L80, M|'+(('16/255'if owner==row else '7/100')if i==row else('255/255'if i==owner else '100/100'))+(' tox'if i==row else ''))
    wire,_=public_wire.project_lines(public_wire.WireState(),lines+['|turn|1'],**v4.wire_kwargs(self.f.f.catalog));hp='31/255'if owner==row else '13/100';_,packets=public_wire.project_lines(wire,['|',f'|-heal|p{row+1}a: {names[row]}|{hp} tox|[from] item: Leftovers',f'|-damage|p{row+1}a: {names[row]}|0 fnt|[from] psn',f'|faint|p{row+1}a: {names[row]}'],**v4.wire_kwargs(self.f.f.catalog));expected=np.asarray([e.words()for packet in packets for e in packet.events],np.int32);actual=np.asarray(result.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(result.projection.frames.valid).transpose(1,0,2)[row]];np.testing.assert_array_equal(actual,expected)
  _,good=ex.join_consumed_owners(*closed,tables=self.f.account.device);self.assertTrue(bool(good.all()))
 def test_original_direct_branch_default_math_and_optional_companion_rows(self):
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  before=ast.parse(subprocess.check_output(['git','show','1ea9:gpu_engine/engine_terminal_callbacks.py'],text=True));after=ast.parse(Path('gpu_engine/engine_terminal_callbacks.py').read_text());get=lambda tree,n:next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==n)
  for n in ('allows_last_device','capture_ko_on_tape','_candidate'):
   self.assertEqual(ast.dump(get(before,n)),ast.dump(get(after,n)))
  prior=get(before,'matches_final_device');current=get(after,'_matches_direct_final_device');current.name=prior.name;self.assertEqual(ast.dump(prior),ast.dump(current))
  t,cap=self.pending();out=self.emit(t,cap);a=t.execution.terminal;z=out.execution.terminal;mask=j.array([True,False]*4);joined=terminal.commit_rows(mask,a,z);self.assertEqual(joined.status.candidates.tolist(),[1,0]*4)
  self.assertTrue(bool((terminal.commit_rows(mask,a,z._replace(status=z.status._replace(data=z.status.data._replace(binding=z.status.data.binding.at[0].add(j.uint32(1)))))).status.candidates==0).all()))

 def test_original_status_writer_faint_order_and_legacy_emitter_default(self):
  script=r"""const path=require('path'),r=process.argv[1],{Battle}=require(path.join(r,'dist/sim/battle.js')),d=require(path.join(r,'dist/sim/dex.js')).Dex.mod('gen3'),lines=[];const p={hp:31,maxhp:255,baseMaxhp:255,isActive:true,fainted:false,side:{pokemonLeft:1,totalFainted:0},toString:()=> 'p2a: Victim',get getHealth(){return this.hp?this.hp+'/255 tox':'0 fnt'},damage(n,source,effect){const x=Math.min(this.hp,n);this.hp-=x;if(!this.hp)b.faintQueue.push({target:this,source,effect});return x},getAbility:()=>({}),getItem:()=>({}),clearVolatile:()=>{}};const e=d.conditions.get('tox'),b={gen:3,gameType:'singles',dex:d,effect:e,event:{target:p},effectState:{stage:3},faintQueue:[],ended:false,clampIntRange:(x,min)=>Math.max(Math.trunc(x),min),trunc:Math.trunc,runEvent:(name,t,s,e,n)=>name==='Damage'?n:true,singleEvent:()=>{},add:(...xs)=>lines.push(xs.map(String).join('|')),queue:{cancelAction:()=>{}},checkWin:()=>false,getAllActive:()=>[],damage:Battle.prototype.damage,spreadDamage:Battle.prototype.spreadDamage};e.onResidual.call(b,p);Battle.prototype.faintMessages.call(b);console.log(JSON.stringify({lines,hp:p.hp,stage:b.effectState.stage}));"""
  got=json.loads(subprocess.check_output(['node','-e',script,ROOT],text=True));self.assertEqual(got,dict(lines=['-damage|p2a: Victim|0 fnt|[from] psn','faint|p2a: Victim'],hp=0,stage=4))
  # Execute the unchanged old emitter AST as the reference for default calls.
  tree=ast.parse(subprocess.check_output(['git','show','1ea9:gpu_engine/engine_status_residual_callbacks.py'],text=True));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='record_status_residual_device');fn.name='original';env=dict(status.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<original-emitter>','exec'),env)
  t,cap=self.pending();p=cap.pending;b=len(t.count)
  for new in (0,12):
   args=(t,p.side,p.physical,j.full(b,31,j.int32),j.full(b,new,j.int32),j.full(b,255,j.int32),j.full(b,4,j.int32),j.full(b,3,j.int32),j.full(b,4,j.int32),p.data)
   old=env['original'](*args,eligible=True,_defer_own_unknown=True);current=status.record_status_residual_device(*args,eligible=True,_defer_own_unknown=True)
   for a,z in zip(jax.tree.leaves(old),jax.tree.leaves(current)):np.testing.assert_array_equal(a,z)
  # No optional status context alters original direct-only row handling.
  import test_gpu_engine_terminal_closeout as old_tests
  old_tests.TerminalTransportAndCountingTests().test_b8_global_and_original_snapshot_cannot_be_row_selected_or_mutated()
  subprocess.run([__import__('sys').executable,'-c','from gpu_engine import benchmark_capture_gap_census, engine_source_frame_witness, engine_full_v4_event_device'],check=True,capture_output=True,text=True)

 def test_final_raw_is_joined_to_captured_status_party_member(self):
  t,cap=self.pending();final,state=self.finalize(self.emit(t,cap),cap)
  self.assertEqual(terminal.matches_final_device(final,state).tolist(),[True]*8)
  for column,value in ((5,32),(7,256),(4,2),(3,1)):
   words=final.words.at[0,2,column].set(value)
   if column in (3,4):words=words.at[0,3,column].set(value)
   ledger=final.execution.ledger._replace(words=final.execution.ledger.words.at[0,2].set(words[0,2]).at[0,3].set(words[0,3]))
   changed=final._replace(words=words,execution=final.execution._replace(ledger=ledger))
   self.assertEqual(terminal.matches_final_device(changed,state).tolist(),[False]+[True]*7,column)
