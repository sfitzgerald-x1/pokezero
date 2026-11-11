"""Literal Spikes owning write/FULL/field encoder; no worlds/models/Native/games."""
from _showdown_root import showdown_root_str
import ast,json,os,shutil,subprocess,unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,engine_spikes_callbacks as spike,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full,v4_encoder as v4,public_wire,public_field,private_requests
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

class SpikesOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests();cls.metadata=deepcopy(cls.f.metadata)
  vocab=cls.metadata['vocabulary'];vocab['abilities']=['_none'];vocab['items']=['_none'];old=cls.f.catalog;moves=dict(old.dex.moves);labels=set(old.vocabulary.tokens)
  for row in cls.metadata['moves'][1:]:row.update(accuracy=100,priority=0,flags={},source_type='Normal')
  cls.move=len(vocab['moves']);vocab['moves'].append('spikes');cls.metadata['moves'].append(dict(id='spikes',base_power=0,type=vocab['types'].index('ground'),source_type='Ground',category='Status',accuracy=True,priority=0,flags={'mustpressure':1}))
  moves['spikes']=replace(old.dex.moves['bodyslam'],id='spikes',name='Spikes',type='Ground',base_power=0,category='Status',gen3_category='Status',accuracy=True,priority=0,target='foeSide',recoil=False,drain=False,heal=False,status=None,boosts={},selfdestruct=False,effect_chance=0,effect_label='')
  labels|={'move:spikes','belief:possible_move:spikes'};dex=ShowdownDex(moves,old.dex.species,old.dex.type_chart)
  chart=np.ones((17,17));chart[vocab['types'].index('ground')-1,vocab['types'].index('flying')-1]=0;cls.metadata['type_chart']=chart.tolist()
  cls.catalog=v4.compile_encoder(old.source,dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=old.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter);cls.helper=spike.compile_spikes(engine_catalog=cls.producer,metadata=cls.metadata);cls.context=spike.Context(cls.helper.device,cls.account.device);cls.risks=risk.compile_risks(cls.producer,cls.account,spikes_compiled=cls.helper)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.asarray(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)
 def initial(self,b=2,consume=True):
  s=self.f.state(b);s=s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),done=j.zeros(b,bool),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),item=j.zeros_like(s.item),moves=j.full_like(s.moves,self.move),pp=j.full_like(s.pp,32),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131))
  cursor=life.register(j.arange(4101,4101+b,dtype=j.int32),**self.kw);f=life.initialize_with_lifecycle(s,self.table,cursor=cursor,**self.kw,tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risks);consumers=[]
  for owner in ((0,1) if consume else ()):
   stream=full.register(f.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=f.cursor.stream_id,owner_sides=j.full(b,owner,j.int32));result=full.project_and_fold(stream,f,self.adapter.device,public,self.catalog);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  return f.state,f.cursor,consumers
 def play(self,s,cursor,context=True,prior=None):
  b=len(s.active);tape=life._tape(cursor,self.producer,64)._replace(refusal=j.ones(b,j.int32) if prior is None else prior)
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  return tr.step_with_chance(s,j.zeros((b,2),j.int32),self.table,fixtures.engine_fixture.EnginePublicEventsTests().chance(b),_public_trace=tape,_lifecycle_trace=True,_spikes_context=self.context if context else None)
 def reference(self,t,row):
  names=[self.catalog.species.names[int(self.adapter.device.causal.species[i])] for i in (1,2)];ident=[f'p{i+1}a: '+n for i,n in enumerate(names)]
  wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[i]}|{names[i]}, L80, M|100/100' for i in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[]
  for index,w in enumerate(np.asarray(t.words[row,:int(t.count[row])])):
   k=int(w[0])
   if k==events.MOVE:lines.append(f'|move|{ident[int(w[1])]}|Spikes|{ident[1-int(w[1])]}')
   elif k==31:lines.append(f'|-sidestart|p{int(w[4])+1}: Player|Spikes')
   elif k==life.TURNLOOP_PREFIX:
    self.assertEqual(w.tolist(),[35,-1,-1,-1,-1,0,0,0,0,0])
    lines.extend(['|','|t:|1'])
   elif k==life.ACTION_END:lines.append('|')
   elif k==life.UPKEEP:lines.append('|upkeep')
   elif k==life.TURN:lines.append(f'|turn|{int(t.event_turn[row,index])}')
   else:self.fail('unexpected raw '+str(k))
  _,ps=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));return np.asarray([e.words() for p in ps for e in p.events],np.int32)
 def test_original_actual_increment_numeric_rng_both_full_and_field_encoder(self):
  s,cursor,consumers=self.initial();s=s._replace(types=s.types.at[1,1,0,0].set(self.metadata['vocabulary']['types'].index('flying')))
  actual=self.play(s,cursor);self.same(actual.step.state,tr.step_with_chance(s,j.zeros((2,2),j.int32),self.table,fixtures.engine_fixture.EnginePublicEventsTests().chance(2)).state)
  self.assertEqual(actual.step.state.spikes.tolist(),[[1,1],[1,1]]);self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0])
  from gpu_engine import engine_source_frame_witness as witness
  tables=SimpleNamespace(risk=self.risks.device,secondary_weather=j.zeros(len(self.table.power),bool),direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
  self.assertTrue(bool(witness._raw_complete(actual.callbacks,tables).all()));expected,_,ok=witness._geometry(SimpleNamespace(frame=j.ones(2,j.int32)),actual.callbacks);self.assertTrue(bool(ok.all()));np.testing.assert_array_equal(expected,actual.callbacks.execution.risks.carry.counts)
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32));closed=[]
  for owner,(stream,public) in enumerate(consumers):
   ledger,result=ex.consume_frame_device(actual.callbacks.execution.ledger,frame,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog);closed.append(ledger)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.site[:,0].tolist(),[int(ex.Site.TURNLOOP_PREFIX)]*2);self.assertEqual(ledger.stage[:,:3].tolist(),[[3,3,3],[3,3,3]])
   self.assertEqual(result.public.context.field.spikes.tolist(),[[1,1],[1,1]])
   request=private_requests.parse_action_request({'wait':True},owner_side=owner)
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),self.reference(actual.callbacks,row))
   patch=public_field.project(result.public.context.field,self.catalog.context.field,request,observer_side=owner,expected_vocabulary=self.catalog.context.field.vocabulary_binding)
   self.assertTrue(bool(patch.admitted.all()))
   for name in ('SELF_HAZARDS','OPP_HAZARDS'):np.testing.assert_allclose(np.asarray(patch.numeric[:,public_field.NUMERIC_COLUMNS[public_field.CORE.index(name)]]),1/3,rtol=0,atol=1e-7)
   self.assertFalse(bool(result.projection.B4_qualified.any()))
  _,ok=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(ok.all()))
 def test_closed_caps_blocked_inactive_and_prior_are_still_unknown(self):
  s,cursor,_=self.initial(8,False);s=s._replace(spikes=s.spikes.at[1,1].set(3).at[2,0].set(-1),status=s.status.at[3,0,0].set(tr.m.SLP),status_counter=s.status_counter.at[3,0,0].set(3),done=s.done.at[7].set(True))
  actual=self.play(s,cursor,prior=j.ones(8,j.int32).at[5].set(65));self.same(actual.step.state,tr.step_with_chance(s,j.zeros((8,2),j.int32),self.table,fixtures.engine_fixture.EnginePublicEventsTests().chance(8)).state)
  for row in (1,2,3,5):self.assertNotEqual(int(actual.callbacks.refusal[row])&~1,0)
  self.assertTrue(int(actual.callbacks.refusal[5])&64);self.assertEqual(int(actual.callbacks.count[7]),0)
  closed=self.play(s,cursor,False);self.assertFalse(bool(j.any(closed.callbacks.words[:,:,0]==31)));self.assertTrue(int(closed.callbacks.execution.risks.carry.risk[0])&risk.FIELD_VOLATILE)
 def pending(self,b=8,mask=None):
  s,cursor,_=self.initial(b,False);mask=j.ones(b,bool) if mask is None else mask
  t=life._tape(cursor,self.producer,64);t=ex.begin_on_tape(t,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,self.risks.device),t,self.risks.device,active=j.ones(b,bool));t=t._replace(execution=t.execution._replace(risks=risk.Context(carry,self.risks.device)))
  side=j.zeros(b,j.int32);physical=s.active[:,0];move=j.full(b,self.move,j.int32)
  t=events.move(t,side,physical,s.active[:,1],move,eligible=mask);t,k=ex.open_attempt(t,ex.Site.SPIKES_WRITE,side,active=mask)
  w=spike.Write(j.ones(b,j.int32),j.zeros(b,j.int32),j.ones(b,j.int32),j.ones(b,bool))
  return t,k,side,physical,move,w,mask
 def test_exact_old_new_source_span_capacity_bindings_and_b8_inactive(self):
  mask=j.array([True]*6+[False]*2);t,k,side,physical,move,w,mask=self.pending(mask=mask)
  w=w._replace(old_layers=j.array([0,1,2,3,-1,0,0,0],j.int32),new_layers=j.array([1,2,3,3,0,1,1,1],j.int32))
  t=t._replace(count=t.count.at[5].set(64),refusal=t.refusal.at[1].set(65))
  out=spike.record_device(t,k,side,physical,move,w,self.context,requested=mask)
  self.assertEqual(out.count.tolist(),[2,2,2,1,1,64,0,0]);self.assertTrue(int(out.refusal[1])&64);self.assertTrue(int(out.refusal[5])&events.OVERFLOW)
  for row in (3,4,5):self.assertTrue(int(out.refusal[row])&256)
  for a,b in zip(jax.tree.leaves(t),jax.tree.leaves(out),strict=True):
   if getattr(a,'ndim',0) and a.shape[0]==8:np.testing.assert_array_equal(a[6:],b[6:])
  for name in ('engine_binding','source_binding','move_binding','producer_binding'):np.testing.assert_array_equal(getattr(t,name),getattr(out,name))
  bad=self.context._replace(data=self.context.data._replace(binding=self.context.data.binding.at[0].add(j.uint32(1))))
  foreign=spike.record_device(t,k,side,physical,move,w,bad,requested=mask);self.assertEqual(foreign.count.tolist(),t.count.tolist());self.assertTrue(bool(((foreign.refusal[:6]&256)!=0).all()))
  for name in self.context.data._fields[:-1]:
   malformed=self.context._replace(data=self.context.data._replace(**{name:getattr(self.context.data,name)[None,:]}))
   with self.assertRaises(ValueError):spike.record_device(t,k,side,physical,move,w,malformed,requested=mask)
  t,k,side,physical,move,w,mask=self.pending(2);good=spike.record_device(t,k,side,physical,move,w,self.context,requested=mask)
  for changed in (good._replace(words=good.words.at[0,1,4].set(0)),good._replace(words=good.words.at[0,1,6].set(2)),good._replace(event_generation=good.event_generation.at[0,1,1].add(1))):
   self.assertEqual(spike.span_matches(changed,k,side,physical,move,w,self.context).tolist(),[False,True])
  poisoned=good._replace(execution=good.execution._replace(ledger=good.execution.ledger._replace(claimed=good.execution.ledger.claimed.at[0,1].set(0))))
  result=risk.spikes_on_tape(poisoned,j.ones(2,bool),active=j.ones(2,bool),captured=risk.CapturedSpikes(self.context,k,side,physical,move,w))
  self.assertTrue(int(result.execution.risks.carry.risk[0])&risk.FIELD_VOLATILE);self.assertEqual(int(result.execution.risks.carry.risk[1]),0)
 def test_current_and_preceding_move_stamps_must_match_same_ticket(self):
  t,k,side,physical,move,w,mask=self.pending(2);good=spike.record_device(t,k,side,physical,move,w,self.context,requested=mask)
  self.assertEqual(spike.span_matches(good,k,side,physical,move,w,self.context).tolist(),[True,True])
  slot,_=ex._ticket(good.execution.ledger,k)
  for name,changed in (
      ('current_generation',good._replace(cursor=good.cursor._replace(generation=good.cursor.generation.at[0,1].add(1)))),
      ('prefix_turn',good._replace(event_turn=good.event_turn.at[0,0].add(1))),
      ('prefix_phase',good._replace(event_phase=good.event_phase.at[0,0].set(1))),
      ('prefix_generation',good._replace(event_generation=good.event_generation.at[0,0,1].add(1))),
      ('ticket_turn',good._replace(execution=good.execution._replace(ledger=good.execution.ledger._replace(turn=good.execution.ledger.turn.at[0,slot[0]].add(1)))))):
   with self.subTest(name=name):
    self.assertEqual(spike.span_matches(changed,k,side,physical,move,w,self.context).tolist(),[False,True])
    refused=risk.spikes_on_tape(changed,j.ones(2,bool),active=j.ones(2,bool),captured=risk.CapturedSpikes(self.context,k,side,physical,move,w))
    self.assertTrue(int(refused.execution.risks.carry.risk[0])&risk.FIELD_VOLATILE)
    self.assertEqual(int(refused.execution.risks.carry.risk[1]),0)
 def test_default_ast_and_original_pinned_source_callbacks(self):
  self.assertEqual(int(self.risks.device.selected_risk[self.move]),0)
  self.assertTrue(int(risk.compile_risks(self.producer,self.account).device.selected_risk[self.move])&risk.FIELD_VOLATILE)
  with self.assertRaises(ValueError):spike.check_compiled(replace(self.helper))
  with self.assertRaises(ValueError):jax.jit(lambda unused:spike.check_compiled(self.helper))(j.array(0))
  moves=dict(self.catalog.dex.moves);moves['surf']=replace(moves['surf'],target='foeSide')
  other=events.compile_catalog(self.metadata,self.table,public_moves=self.catalog.moves,dex=ShowdownDex(moves,self.catalog.dex.species,self.catalog.dex.type_chart),exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  self.assertEqual(other.ids[self.metadata['vocabulary']['moves'].index('surf')],0);self.assertGreater(other.ids[self.move],0)
  # Re-certify the integrated device-hook program at the published PR base.
  # Keep the whole normalized AST and HP/RNG mutation controls exact.
  normalize=__import__('test_gpu_engine_v5_hypothetical_profile').normalized_hypothetical_default
  before=normalize(subprocess.check_output(['git','show','b8092a85:gpu_engine/transition.py'],text=True),as_tree=True);after=normalize(Path('gpu_engine/transition.py').read_text(),as_tree=True);baseline=dump_default_program(before);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  root=Path(showdown_root_str());self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),events.SHOWDOWN)
  node=shutil.which('node');self.assertIsNotNone(node)
  script="const {Dex}=require(process.argv[1]+'/dist/sim/dex.js');const {Battle}=require(process.argv[1]+'/dist/sim/battle.js');const m=Dex.mod('gen3').moves.get('spikes');const foe={toString:()=> 'p2a: Absol'};const p={side:{foe:{active:[foe]}}};const target=Battle.prototype.getRandomTarget.call({dex:Dex.mod('gen3'),gameType:'singles'},p,m);const rows=[];for(const n of [0,1,2,3]){const lines=[];const ctx={effectState:{layers:n},add:(...x)=>lines.push(x.map(String))};const side={toString:()=> 'p2: Player'};const ret=(n?m.condition.onSideRestart:m.condition.onSideStart).call(ctx,side);rows.push({lines,layers:ctx.effectState.layers,blocked:ret===false})}console.log(JSON.stringify({target:String(target),rows}));"
  d=json.loads(subprocess.check_output([node,'-e',script,str(root)],text=True));self.assertEqual(d['target'],'p2a: Absol');self.assertEqual([r['layers'] for r in d['rows']],[1,2,3,3]);self.assertEqual([len(r['lines']) for r in d['rows']],[1,1,1,0]);self.assertTrue(d['rows'][3]['blocked'])
