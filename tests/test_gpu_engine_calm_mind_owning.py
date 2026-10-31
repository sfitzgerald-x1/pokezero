"""Literal kernel self-boost/FULL tests; no models/Native/worlds/games."""
import ast,json,os,shutil,subprocess,unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,engine_calm_mind_callbacks as calm,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full,v4_encoder as v4,public_wire
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

class CalmMindOwningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests();cls.metadata=deepcopy(cls.f.metadata)
  vocab=cls.metadata['vocabulary'];vocab['abilities']=['_none'];vocab['items']=['_none'];cls.ids={};old=cls.f.catalog;moves=dict(old.dex.moves);labels=set(old.vocabulary.tokens)
  for row in cls.metadata['moves'][1:]:row.update(accuracy=100,priority=0,flags={},source_type='Normal')
  for name,label,typ,typid,boosts in (('calmmind','Calm Mind','Psychic',14,{'spa':1,'spd':1}),('swordsdance','Swords Dance','Normal',1,{'atk':2})):
   cls.ids[name]=len(vocab['moves']);vocab['moves'].append(name);cls.metadata['moves'].append(dict(id=name,base_power=0,type=typid,source_type=typ,category='Status',accuracy=True,priority=0,flags={'snatch':1},boosts=boosts))
   moves[name]=replace(old.dex.moves['bodyslam'],id=name,name=label,type=typ,base_power=0,category='Status',gen3_category='Status',accuracy=True,priority=0,target='self',recoil=False,drain=False,heal=False,status=None,boosts=boosts,selfdestruct=False,effect_chance=0,effect_label='')
   labels|={'move:'+name,'belief:possible_move:'+name}
  chart=np.ones((17,17));chart[13,16]=0;cls.metadata['type_chart']=chart.tolist();dex=ShowdownDex(moves,old.dex.species,old.dex.type_chart)
  cls.catalog=v4.compile_encoder(old.source,dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=old.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter);cls.helper=calm.compile_calm_mind(engine_catalog=cls.producer,metadata=cls.metadata);cls.context=calm.Context(cls.helper.device,cls.account.device);cls.risks=risk.compile_risks(cls.producer,cls.account,calm_mind_compiled=cls.helper)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def initial(self,b=2,*,consume=True):
  s=self.f.state(b);s=s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),done=j.zeros(b,bool),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),item=j.zeros_like(s.item),moves=j.full_like(s.moves,self.ids['calmmind']),pp=j.full_like(s.pp,24),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131))
  cursor=life.register(j.arange(3101,3101+b,dtype=j.int32),**self.kw);frame=life.initialize_with_lifecycle(s,self.table,cursor=cursor,**self.kw,tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risks)
  consumers=[]
  for owner in ((0,1) if consume else ()):
   stream=full.register(frame.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=j.full(b,owner,j.int32))
   result=full.project_and_fold(stream,frame,self.adapter.device,public,self.catalog);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  return frame.state,frame.cursor,consumers
 def play(self,state,cursor,*,context=True,prior=None):
  b=len(state.active);tape=life._tape(cursor,self.producer,64)._replace(refusal=j.ones(b,j.int32) if prior is None else prior)
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  return tr.step_with_chance(state,j.zeros((b,2),j.int32),self.table,fixtures.engine_fixture.EnginePublicEventsTests().chance(b),_public_trace=tape,_lifecycle_trace=True,_calm_mind_context=self.context if context else None)
 def test_canonical_static_capability_default_ast_and_original_source_order(self):
  from gpu_engine import engine_source_frame_witness as witness
  witness._source();self.assertIn('engine_calm_mind_callbacks.py',witness.SOURCE_PINS)
  move=self.ids['calmmind'];self.assertEqual(int(self.risks.device.selected_risk[move]),0)
  self.assertEqual(int(self.risks.device.selected_risk[self.ids['swordsdance']])&(risk.NON_WEATHER|risk.BOOST),risk.NON_WEATHER|risk.BOOST)
  closed=risk.compile_risks(self.producer,self.account);self.assertTrue(int(closed.device.selected_risk[move])&risk.BOOST)
  with self.assertRaises(ValueError):calm.check_compiled(replace(self.helper))
  with self.assertRaises(ValueError):jax.jit(lambda unused:calm.check_compiled(self.helper))(j.array(0))
  before=ast.parse(subprocess.check_output(['git','show','7bc45a72:gpu_engine/transition.py'],text=True));after=__import__('test_gpu_engine_v5_hypothetical_profile').normalized_hypothetical_default(Path('gpu_engine/transition.py').read_text(),as_tree=True);baseline=dump_default_program(before);self.assertEqual(baseline,dump_default_program(after));assert_hp_rng_mutations_detected(self,after,baseline)
  node=shutil.which('node')
  if not node:self.skipTest('node unavailable')
  root=Path(os.environ.get('POKEZERO_SHOWDOWN_ROOT','/Users/scott/workspace/agents/pokemon-showdown-timing-smoke-f76228a'));self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),events.SHOWDOWN)
  script="const root=process.argv[1];const {Dex}=require(root+'/dist/sim/dex.js');const {Battle}=require(root+'/dist/sim/battle.js');const m=Dex.mod('gen3').moves.get('calmmind');const lines=[];const p={hp:100,isActive:true,boosts:{spa:0,spd:0},getCappedBoost:x=>x,boostBy(x){const k=Object.keys(x)[0];this.boosts[k]+=x[k];return x[k]},toString:()=> 'p1a: Snorlax'};Battle.prototype.boost.call({gen:3,runEvent:(event,t,s,e,x)=>x,add:(...x)=>lines.push(x.map(String).join('|'))},m.boosts,p,p,m);process.stdout.write(JSON.stringify(lines));"
  self.assertEqual(json.loads(subprocess.check_output([node,'-e',script,str(root)],text=True)),['-boost|p1a: Snorlax|spa|1','-boost|p1a: Snorlax|spd|1'])
 def test_actual_pair_same_numeric_rng_and_original_both_full_consumers(self):
  state,cursor,consumers=self.initial();state=state._replace(types=state.types.at[1,1,0,0].set(17));actual=self.play(state,cursor);chance=fixtures.engine_fixture.EnginePublicEventsTests().chance(2);self.same(actual.step.state,tr.step_with_chance(state,j.zeros((2,2),j.int32),self.table,chance).state)
  self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0]);self.assertEqual(actual.step.state.boosts[:,:,:,2:4][:,:,0].tolist(),[[[1,1],[1,1]]]*2)
  for row in (0,1):
   raw=np.array(actual.callbacks.words[row,:int(actual.callbacks.count[row])]);pair=raw[raw[:,0]==30];self.assertEqual(pair[:,4].tolist(),[2,3,2,3]);self.assertEqual(pair[:,5:7].tolist(),[[0,1]]*4)
  from gpu_engine import engine_source_frame_witness as witness
  tables=SimpleNamespace(risk=self.risks.device,secondary_weather=j.zeros(len(self.table.power),bool),direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
  self.assertTrue(bool(witness._raw_complete(actual.callbacks,tables).all()))
  expected,_,geometry=witness._geometry(SimpleNamespace(frame=j.ones(2,j.int32)),actual.callbacks)
  self.assertTrue(bool(geometry.all()));np.testing.assert_array_equal(expected,actual.callbacks.execution.risks.carry.counts)
  self.assertEqual(expected[:,int(risk.Branch.SECONDARY_STATS)-1].tolist(),[6,3])
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32));closed=[]
  for owner,(stream,public) in enumerate(consumers):
   ledger,result=ex.consume_frame_device(actual.callbacks.execution.ledger,frame,stream,public,active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog);closed.append(ledger)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,:2].tolist(),[[3,3],[3,3]]);self.assertEqual(result.public.mons.boosts[:,:,2:4].tolist(),[[[1,1],[1,1]]]*2);self.assertFalse(bool(result.projection.B4_qualified.any()))
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),self.reference_words(actual.callbacks,row,owner))
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_caps_blocked_other_boosts_and_prior_refusal_stay_unknown(self):
  state,cursor,_=self.initial(8,consume=False)
  state=state._replace(boosts=state.boosts.at[1,0,0,2].set(6).at[2,0,0,3].set(6).at[6,0,0,2:4].set(6),
      status=state.status.at[3,0,0].set(tr.m.SLP),status_counter=state.status_counter.at[3,0,0].set(3),
      moves=state.moves.at[4,0,0,0].set(self.ids['swordsdance']),done=state.done.at[7].set(True))
  prior=j.ones(8,j.int32).at[5].set(65);actual=self.play(state,cursor,prior=prior)
  self.same(actual.step.state,tr.step_with_chance(state,j.zeros((8,2),j.int32),self.table,fixtures.engine_fixture.EnginePublicEventsTests().chance(8)).state)
  self.assertEqual(int(actual.callbacks.refusal[0]),1)
  for row in (1,2,3,4,5,6):self.assertNotEqual(int(actual.callbacks.refusal[row])&~1,0)
  self.assertTrue(int(actual.callbacks.refusal[5])&64);self.assertTrue(int(actual.callbacks.execution.risks.carry.risk[4])&risk.BOOST)
  self.assertEqual(int(actual.callbacks.count[7]),0)
  closed=self.play(state,cursor,context=False)
  self.assertFalse(bool(j.any(closed.callbacks.words[:,:,0]==30)));self.assertTrue(int(closed.callbacks.execution.risks.carry.risk[0])&risk.BOOST)
 def pending(self,b=8,*,mask=None):
  state,cursor,_=self.initial(b,consume=False);mask=j.ones(b,bool) if mask is None else mask
  tape=life._tape(cursor,self.producer,64);tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool))
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  side=j.zeros(b,j.int32);physical=state.active[:,0];move=j.full(b,self.ids['calmmind'],j.int32)
  tape=events.move(tape,side,physical,state.active[:,1],move,eligible=mask);tape,ticket=ex.open_attempt(tape,ex.Site.CALM_MIND,side,active=mask)
  write=calm.Write(j.zeros(b,j.int32),j.ones(b,j.int32),j.zeros(b,j.int32),j.ones(b,j.int32),j.ones(b,bool))
  return tape,ticket,side,physical,move,write,mask
 def test_b8_atomic_overflow_global_bindings_inactive_sticky_and_pair_mutants(self):
  mask=j.array([True]*4+[False]*4);tape,ticket,side,physical,move,write,mask=self.pending(mask=mask)
  tape=tape._replace(count=tape.count.at[2].set(64),switch_details=tape.switch_details.at[2,63,0].set(7),refusal=tape.refusal.at[1].set(64))
  out=calm.record_device(tape,ticket,side,physical,move,write,self.context,eligible=mask)
  self.assertEqual(out.count.tolist(),[3,3,64,3,0,0,0,0]);self.assertTrue(int(out.refusal[1])&64);self.assertTrue(int(out.refusal[2])&events.OVERFLOW)
  for a,b in zip(jax.tree.leaves(tape),jax.tree.leaves(out)):
   if getattr(a,'ndim',0) and a.shape[0]==8:np.testing.assert_array_equal(a[4:],b[4:])
  bad=self.context._replace(data=self.context.data._replace(binding=self.context.data.binding.at[0].add(j.uint32(1))))
  refused=calm.record_device(tape,ticket,side,physical,move,write,bad,eligible=mask);self.assertEqual(refused.count.tolist(),tape.count.tolist());self.assertTrue(bool((refused.refusal[:4]&256).all()))
  for field in self.context.data._fields[:-1]:
   malformed=self.context._replace(data=self.context.data._replace(**{field:getattr(self.context.data,field)[None,:]}))
   with self.assertRaises(ValueError):calm.record_device(tape,ticket,side,physical,move,write,malformed,eligible=mask)
  # Exact source-span read-only checks cannot accept a partial/migrated pair.
  clean=self.pending(2);t,k,s,p,m,w,active=clean;complete=calm.record_device(t,k,s,p,m,w,self.context,eligible=active)
  for changed in (complete._replace(words=complete.words.at[0,2,4].set(2)),complete._replace(count=complete.count.at[0].add(-1)),complete._replace(event_generation=complete.event_generation.at[0,2,1].add(1))):
   matches=calm.span_matches(changed,k,s,p,m,w,self.context);self.assertEqual(matches.tolist(),[False,True])
 def reference_words(self,tape,row,owner):
  names=tuple(self.catalog.species.names[int(self.adapter.device.causal.species[i])] for i in (1,2));ident={i:f'p{i+1}a: '+names[i] for i in (0,1)}
  wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[i]}|{names[i]}, L80, M|100/100' for i in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[]
  for index,w in enumerate(np.array(tape.words[row,:int(tape.count[row])])):
   k=int(w[0])
   if k==events.MOVE:lines.append(f'|move|{ident[int(w[1])]}|Calm Mind|{ident[int(w[1])]}')
   elif k==30:lines.append(f'|-boost|{ident[int(w[1])]}|'+('spa' if int(w[4])==2 else 'spd')+'|1')
   elif k==life.ACTION_END:lines.append('|')
   elif k==life.UPKEEP:lines.append('|upkeep')
   elif k==life.TURN:lines.append(f'|turn|{int(tape.event_turn[row,index])}')
   else:self.fail('unexpected raw '+str(k))
  _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));return np.array([e.words() for p in packets for e in p.events],np.int32)
