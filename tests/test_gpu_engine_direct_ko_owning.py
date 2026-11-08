"""Prospective typed direct-KO components, no worlds/models/Native/games."""
import ast,subprocess,unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,engine_direct_ko_callbacks as ko,engine_direct_hit_controls as hit
from gpu_engine import engine_kernel_frame_risk as risk,engine_execution_completeness as ex,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_events as events,engine_public_event_device as base,engine_full_v4_event_device as full,v4_encoder as v4,public_wire
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

class DirectKOOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests();cls.metadata=deepcopy(cls.f.metadata)
  cls.metadata['vocabulary']['abilities']=['_none'];cls.metadata['vocabulary']['items']=['_none']
  for row in cls.metadata['moves'][1:]:row.update(accuracy=100,priority=0,flags={},source_type='Normal')
  chart=np.ones((17,17));chart[0,5]=.5;cls.metadata['type_chart']=chart.tolist();cls.catalog=cls.f.catalog;cls.table=tr.tables_from_metadata(cls.metadata)
  cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter);cls.helper=hit.compile_direct_hit_controls(engine_catalog=cls.producer);cls.context=hit.RegisteredContext(cls.helper.device,cls.account.device);cls.risks=risk.compile_risks(cls.producer,cls.account,direct_hit_compiled=cls.helper)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def initial(self,b=2,*,last=False,consume=True):
  s=self.f.state(b);hp=j.full_like(s.hp,100).at[:,1,0].set(10)
  if last:hp=hp.at[:,1,1:].set(0)
  s=s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),done=j.zeros(b,bool),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),item=j.zeros_like(s.item),moves=j.ones_like(s.moves),pp=j.full_like(s.pp,24),hp=hp,stats=j.full_like(s.stats,142).at[:,1,:,5].set(131))
  cursor=life.register(j.arange(4101,4101+b,dtype=j.int32),**self.kw);frame=life.initialize_with_lifecycle(s,self.table,cursor=cursor,**self.kw,tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risks)
  consumers=[]
  for owner in ((0,1) if consume else ()):
   stream=full.register(frame.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=j.full(b,owner,j.int32))
   result=full.project_and_fold(stream,frame,self.adapter.device,public,self.catalog);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  return frame.state,frame.cursor,consumers
 def chance(self,b=2):return fixtures.engine_fixture.EnginePublicEventsTests().chance(b)
 def play(self,state,cursor,*,prior=None,chance=None):
  b=len(state.active);tape=life._tape(cursor,self.producer,64)._replace(refusal=j.ones(b,j.int32) if prior is None else prior)
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  return tr.step_with_chance(state,j.zeros((b,2),j.int32),self.table,self.chance(b) if chance is None else chance,_public_trace=tape,_lifecycle_trace=True,_direct_hit_context=self.context)
 def test_source_scope_keeps_nonlethal_validator_and_default_math_strict(self):
  self.assertTrue(bool(self.helper.device.ko_candidates[1]));self.assertFalse(bool(self.helper.device.ko_candidates[int(self.table.struggle)]))
  write=hit.HitWrite(j.full(2,40,j.int32),j.zeros(2,bool),j.ones(2,bool),j.zeros(2,bool),j.ones(2,j.int32),j.ones(2,j.int32),j.zeros(2,bool),j.zeros(2,bool),j.zeros(2,bool),j.full(2,10,j.int32),j.zeros(2,j.int32),j.full(2,100,j.int32))
  self.assertFalse(bool(hit.validate_write_device(write,2,data=self.helper.device,move=j.ones(2,j.int32)).any()))
  before=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True));after=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(before);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
 def test_actual_hp0_faint_span_both_full_and_no_terminal_promotion(self):
  state,cursor,consumers=self.initial();actual=self.play(state,cursor);self.same(actual.step.state,tr.step_with_chance(state,j.zeros((2,2),j.int32),self.table,self.chance()).state)
  self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0]);self.assertEqual(actual.step.state.hp[:,1,0].tolist(),[0,0]);self.assertEqual(actual.step.state.done.tolist(),[False,False]);self.assertEqual(actual.step.state.pending_residual.tolist(),[True,True]);self.assertEqual(actual.step.state.forced_switch.tolist(),[[False,True]]*2)
  raw=np.array(actual.callbacks.words[0,:int(actual.callbacks.count[0])]);self.assertEqual(raw[:,0].tolist(),[1,2,3]);self.assertEqual(actual.callbacks.execution.ledger.stage[:,0].tolist(),[2,2])
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32));closed=[]
  for owner,(stream,public) in enumerate(consumers):
   ledger,result=ex.consume_frame_device(actual.callbacks.execution.ledger,frame,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog);closed.append(ledger)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,0].tolist(),[3,3]);self.assertFalse(bool(result.projection.B4_qualified.any()))
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),self.reference_words(actual.callbacks,row,owner))
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_critical_resisted_ko_controls_are_exact_owned_suffixes_both_full(self):
  state,cursor,consumers=self.initial();state=state._replace(types=state.types.at[:,1,0,0].set(6));chance=self.chance()._replace(draws=self.chance().draws.at[:,0,4].set(0));actual=self.play(state,cursor,chance=chance)
  self.same(actual.step.state,tr.step_with_chance(state,j.zeros((2,2),j.int32),self.table,chance).state);self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.words[0,:5,0].tolist(),[1,21,23,2,3])
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32));closed=[]
  for owner,(stream,public) in enumerate(consumers):
   ledger,result=ex.consume_frame_device(actual.callbacks.execution.ledger,frame,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog);closed.append(ledger);self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,0].tolist(),[3,3])
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),self.reference_words(actual.callbacks,row,owner))
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def pending(self,b=8,*,mask=None,write_change=None,last=False):
  state,cursor,_=self.initial(b,last=last,consume=False);mask=j.ones(b,bool) if mask is None else mask
  tape=life._tape(cursor,self.producer,64);tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  side=j.zeros(b,j.int32);physical=state.active[:,0];target=state.active[:,1];move=j.ones(b,j.int32)
  tape=events.move(tape,side,physical,target,move,eligible=mask);tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,1-side,active=mask)
  write=hit.HitWrite(j.full(b,40,j.int32),j.zeros(b,bool),j.ones(b,bool),j.zeros(b,bool),j.ones(b,j.int32),j.ones(b,j.int32),j.zeros(b,bool),j.zeros(b,bool),j.zeros(b,bool),j.full(b,10,j.int32),j.zeros(b,j.int32),j.full(b,100,j.int32))
  if write_change:write=write_change(write)
  tape,p=ko.prepare_controls_device(tape,ticket,side,physical,target,move,write,self.context,state.hp[:,0,0],state.hp[:,1],eligible=mask)
  tape=events.condition(tape,1-side,target,j.zeros(b,j.int32),j.full(b,100,j.int32),j.zeros(b,j.int32),eligible=mask)
  return tape,p
 def finish(self,tape,p,*,actor_hp=None,party=None):
  b=len(p.actor);party=p.target_party_before.at[j.arange(b),p.target_physical].set(0) if party is None else party
  return ko.finish_device(tape,p,p.actor_hp_before if actor_hp is None else actor_hp,party,p.actor_physical,p.target_physical)
 def test_pending_has_no_credit_missing_interleaved_self_terminal_and_prior_fail_closed(self):
  tape,p=self.pending(2);self.assertEqual(tape.execution.ledger.stage[:,0].tolist(),[1,1])
  outcome=risk.Outcome(p.write.power,p.write.critical,p.write.hit,p.write.immune,p.write.numerator,p.write.denominator,j.zeros(2,bool),p.write.substitute,p.write.protected,p.write.enduring,j.zeros(2,j.int32),j.zeros(2,bool))
  self.assertTrue(bool(ko.matches_outcome_device(tape,p,outcome,self.risks.device).all()))
  prior=tape.execution.risks.carry._replace(risk=j.full(2,risk.BOOST,j.int32));observed=risk.observe_outcome_device(prior,tape,outcome,self.risks.device,eligible=j.ones(2,bool),active=j.ones(2,bool),pending_ko=p);self.assertEqual(observed.risk.tolist(),[risk.BOOST]*2)
  self.assertTrue(bool((self.finish(tape,p).refusal&256).all()))
  complete=events.faint(tape,1-p.actor,p.target_physical,eligible=p.requested);good=self.finish(complete,p);self.assertEqual(good.execution.ledger.stage[:,0].tolist(),[2,2])
  changed=events.condition(tape,p.actor,p.actor_physical,j.full(2,99,j.int32),j.full(2,100,j.int32),j.zeros(2,j.int32),eligible=p.requested);changed=events.faint(changed,1-p.actor,p.target_physical,eligible=p.requested)
  self.assertTrue(bool((self.finish(changed,p).refusal&256).all()));self.assertTrue(bool((self.finish(complete,p,actor_hp=p.actor_hp_before-1).refusal&256).all()))
  for mutate in (lambda w:w._replace(power=j.zeros(8,j.int32)),lambda w:w._replace(protected=j.ones(8,bool)),lambda w:w._replace(enduring=j.ones(8,bool)),lambda w:w._replace(immune=j.ones(8,bool)),lambda w:w._replace(substitute=j.ones(8,bool)),lambda w:w._replace(hit=j.zeros(8,bool))):
   bad,_=self.pending(write_change=mutate);self.assertTrue(bool((bad.refusal&256).all()))
  terminal,_=self.pending(last=True);self.assertTrue(bool((terminal.refusal&256).all()))
  for changed in (complete._replace(event_generation=complete.event_generation.at[0,2,1].add(1)),complete._replace(words=complete.words.at[0,2,4].set(4))):
   result=self.finish(changed,p);self.assertTrue(int(result.refusal[0])&256);self.assertEqual(int(result.refusal[1]),int(complete.refusal[1]))
 def reference_words(self,tape,row,owner):
  names=tuple(self.catalog.species.names[int(self.adapter.device.causal.species[i])] for i in (1,2));ident={i:f'p{i+1}a: '+names[i] for i in (0,1)}
  wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[0]}|{names[0]}, L80, M|100/100',f'|switch|{ident[1]}|{names[1]}, L80, M|10/100','|turn|1'],**v4.wire_kwargs(self.catalog));lines=[]
  for w in np.array(tape.words[row,:int(tape.count[row])]):
   k=int(w[0])
   if k==1:lines.append(f'|move|{ident[int(w[1])]}|{self.catalog.moves.names[int(self.producer.ids[int(w[5])])]}|{ident[int(w[3])]}')
   elif k in (21,22,23):lines.append('|'+{21:'-crit',22:'-supereffective',23:'-resisted'}[k]+'|'+ident[int(w[3])])
   elif k==2:lines.append(f'|-damage|{ident[int(w[3])]}|0 fnt')
   elif k==3:lines.append('|faint|'+ident[int(w[3])])
   elif k==life.TURNLOOP_PREFIX:
    self.assertEqual(w.tolist(),[35,-1,-1,-1,-1,0,0,0,0,0]);lines.extend(['|','|t:|1'])
   else:self.fail('unrepresented KO reference '+str(k))
  _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));return np.array([e.words() for p in packets for e in p.events],np.int32)
