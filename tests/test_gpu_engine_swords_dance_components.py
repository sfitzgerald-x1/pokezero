"""Literal kernel self-boost/FULL tests; no models/Native/worlds/games.

Light replacement coverage for swords_dance_owning with #1819's current-kernel expectations (the turn-loop
prefix entry): the same fixtures and assertions, with the full public fold and the full consumer compiled once
per class (every input an argument) instead of called eagerly, and ChanceInputs built by keyword. Its static-capability/AST/source-pin
test (test_canonical_static_capability_default_ast_and_original_source_order) joins after #1819 re-pins its sources.
"""
from _showdown_root import showdown_root_str
import ast,json,os,shutil,subprocess,unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as tr,engine_swords_dance_callbacks as swords,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_public_events as events,engine_lifecycle_callbacks as life
from gpu_engine import engine_public_event_device as base,engine_full_v4_event_device as full,v4_encoder as v4,public_wire
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
import test_gpu_engine_pressure_entry_owning as fixtures
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected

def _chance(b):
 draws=j.zeros((b,2,12),j.int32).at[:,:,4].set(1);zeros=j.zeros((b,2),j.int32)
 return tr.ChanceInputs(draws=draws,phazing=zeros,secondary=zeros,sleep_talk=zeros,entry_ties=zeros,initial_tie=j.zeros(b,j.int32))

class SwordsDanceComponentTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  fixtures.PressureEntryOwningTests.setUpClass();cls.f=fixtures.PressureEntryOwningTests();cls.metadata=deepcopy(cls.f.metadata)
  vocab=cls.metadata['vocabulary'];vocab['abilities']=['_none'];vocab['items']=['_none'];cls.ids={};old=cls.f.catalog;moves=dict(old.dex.moves);labels=set(old.vocabulary.tokens)
  for row in cls.metadata['moves'][1:]:row.update(accuracy=100,priority=0,flags={},source_type='Normal')
  for name,label,typ,typid,boosts in (('swordsdance','Swords Dance','Normal',1,{'atk':2}),('calmmind','Calm Mind','Psychic',14,{'spa':1,'spd':1})):
   cls.ids[name]=len(vocab['moves']);vocab['moves'].append(name);cls.metadata['moves'].append(dict(id=name,base_power=0,type=typid,source_type=typ,category='Status',accuracy=True,priority=0,flags={'snatch':1},boosts=boosts))
   moves[name]=replace(old.dex.moves['bodyslam'],id=name,name=label,type=typ,base_power=0,category='Status',gen3_category='Status',accuracy=True,priority=0,target='self',recoil=False,drain=False,heal=False,status=None,boosts=boosts,selfdestruct=False,effect_chance=0,effect_label='')
   labels|={'move:'+name,'belief:possible_move:'+name}
  chart=np.ones((17,17));chart[0,7]=0;cls.metadata['type_chart']=chart.tolist();dex=ShowdownDex(moves,old.dex.species,old.dex.type_chart)
  cls.catalog=v4.compile_encoder(old.source,dex,CategoryVocabulary(tuple(sorted(labels))),checkpoint_sha256='a'*64,feature_masks=old.masks.__dict__)
  cls.table=tr.tables_from_metadata(cls.metadata);cls.producer=events.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.catalog.dex,exposure=fixtures.engine_fixture.EnginePublicEventsTests.exposure)
  projector=base.compile_projector(cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  cls.adapter=full.compile_adapter(cls.metadata,projector,catalog=cls.catalog);cls.account=ex.compile_accounting(cls.adapter);cls.helper=swords.compile_swords_dance(engine_catalog=cls.producer,metadata=cls.metadata);cls.context=swords.Context(cls.helper.device,cls.account.device);cls.risks=risk.compile_risks(cls.producer,cls.account,swords_dance_compiled=cls.helper)
  cls.kw=dict(catalog=cls.producer,expected_binding=cls.producer.binding,expected_engine_binding=np.array(cls.producer.engine_binding,np.uint32))
  # One compiled public fold and one compiled full consumer per class (eager calls recompile their scans).
  cls.seed=staticmethod(jax.jit(lambda stream,frame,public:full.project_and_fold(stream,frame,cls.adapter.device,public,cls.catalog)))
  cls.consume=staticmethod(jax.jit(lambda ledger,frame,stream,public,active:ex.consume_frame_device(ledger,frame,stream,public,
   active=active,tables=cls.account.device,adapter_tables=cls.adapter.device,catalog=cls.catalog)))
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def initial(self,b=2,*,consume=True):
  s=self.f.state(b);s=s._replace(active=j.zeros((b,2),j.int32),entry_pending=j.ones((b,2),bool),done=j.zeros(b,bool),ability=j.zeros_like(s.ability),base_ability=j.zeros_like(s.base_ability),item=j.zeros_like(s.item),moves=j.full_like(s.moves,self.ids['swordsdance']),pp=j.full_like(s.pp,24),stats=j.full_like(s.stats,142).at[:,1,:,5].set(131))
  cursor=life.register(j.arange(3101,3101+b,dtype=j.int32),**self.kw);frame=life.initialize_with_lifecycle(s,self.table,cursor=cursor,**self.kw,tie_coin=j.zeros(b,j.int32),execution_compiled=self.account,kernel_risk_compiled=self.risks)
  consumers=[]
  for owner in ((0,1) if consume else ()):
   stream=full.register(frame.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter);public=v4.initialize(self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=j.full(b,owner,j.int32))
   result=self.seed(stream,frame,public);self.assertEqual(result.projection.error.tolist(),[0]*b);consumers.append((result.projection.state,result.public))
  return frame.state,frame.cursor,consumers
 def play(self,state,cursor,*,context=True,prior=None):
  b=len(state.active);tape=life._tape(cursor,self.producer,64)._replace(refusal=j.ones(b,j.int32) if prior is None else prior)
  tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool));carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  return tr.step_with_chance(state,j.zeros((b,2),j.int32),self.table,_chance(b),_public_trace=tape,_lifecycle_trace=True,_swords_dance_context=self.context if context else None)
 def test_actual_uncapped_write_same_numeric_rng_and_original_both_full_consumers(self):
  state,cursor,consumers=self.initial();state=state._replace(types=state.types.at[1,1,0,0].set(8));actual=self.play(state,cursor);chance=_chance(2);self.same(actual.step.state,tr.step_with_chance(state,j.zeros((2,2),j.int32),self.table,chance).state)
  self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0]);self.assertEqual(actual.step.state.boosts[:,:,0,0].tolist(),[[2,2]]*2)
  for row in (0,1):
   raw=np.array(actual.callbacks.words[row,:int(actual.callbacks.count[row])]);literal=raw[raw[:,0]==32];self.assertEqual(literal[:,4:6].tolist(),[[0,2]]*2)
  from gpu_engine import engine_source_frame_witness as witness
  tables=SimpleNamespace(risk=self.risks.device,secondary_weather=j.zeros(len(self.table.power),bool),direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
  self.assertTrue(bool(witness._raw_complete(actual.callbacks,tables).all()))
  expected,_,geometry=witness._geometry(SimpleNamespace(frame=j.ones(2,j.int32)),actual.callbacks)
  self.assertTrue(bool(geometry.all()));np.testing.assert_array_equal(expected,actual.callbacks.execution.risks.carry.counts)
  self.assertEqual(expected[:,int(risk.Branch.SECONDARY_STATS)-1].tolist(),[6,3])
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),j.zeros(2,bool),j.zeros(2,j.int32));closed=[]
  for owner,(stream,public) in enumerate(consumers):
   ledger,result=self.consume(actual.callbacks.execution.ledger,frame,stream,public,j.ones(2,bool));closed.append(ledger)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.site[:,0].tolist(),[int(ex.Site.TURNLOOP_PREFIX)]*2);self.assertEqual(ledger.stage[:,:3].tolist(),[[3,3,3],[3,3,3]]);self.assertEqual(result.public.mons.boosts[:,:,0].tolist(),[[2,2]]*2);self.assertFalse(bool(result.projection.B4_qualified.any()))
   for row in (0,1):np.testing.assert_array_equal(self.f.words(result.projection,row),self.reference_words(actual.callbacks,row,owner))
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_caps_blocked_other_boosts_and_prior_refusal_stay_unknown(self):
  state,cursor,_=self.initial(8,consume=False)
  state=state._replace(boosts=state.boosts.at[1,0,0,0].set(6).at[2,0,0,0].set(5).at[6,0,0,0].set(4),
      status=state.status.at[3,0,0].set(tr.m.SLP),status_counter=state.status_counter.at[3,0,0].set(3),
      moves=state.moves.at[4,0,0,0].set(self.ids['calmmind']),done=state.done.at[7].set(True))
  prior=j.ones(8,j.int32).at[5].set(65);actual=self.play(state,cursor,prior=prior)
  self.same(actual.step.state,tr.step_with_chance(state,j.zeros((8,2),j.int32),self.table,_chance(8)).state)
  self.assertEqual(int(actual.callbacks.refusal[0]),1);self.assertEqual(int(actual.callbacks.refusal[6]),1)
  for row in (1,2,3,4,5):self.assertNotEqual(int(actual.callbacks.refusal[row])&~1,0)
  self.assertTrue(int(actual.callbacks.refusal[5])&64);self.assertTrue(int(actual.callbacks.execution.risks.carry.risk[4])&risk.BOOST)
  self.assertEqual(int(actual.callbacks.count[7]),0)
  closed=self.play(state,cursor,context=False)
  self.assertFalse(bool(j.any(closed.callbacks.words[:,:,0]==32)));self.assertTrue(int(closed.callbacks.execution.risks.carry.risk[0])&risk.BOOST)
 def pending(self,b=8,*,mask=None):
  state,cursor,_=self.initial(b,consume=False);mask=j.ones(b,bool) if mask is None else mask
  tape=life._tape(cursor,self.producer,64);tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=self.account),self.account.device,active=j.ones(b,bool))
  carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,self.risks.device),tape,self.risks.device,active=j.ones(b,bool));tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,self.risks.device)))
  side=j.zeros(b,j.int32);physical=state.active[:,0];move=j.full(b,self.ids['swordsdance'],j.int32)
  tape=events.move(tape,side,physical,state.active[:,1],move,eligible=mask);tape,ticket=ex.open_attempt(tape,ex.Site.SWORDS_DANCE,side,active=mask)
  write=swords.Write(j.zeros(b,j.int32),j.full(b,2,j.int32),j.ones(b,bool))
  return tape,ticket,side,physical,move,write,mask
 def test_b8_overflow_global_bindings_inactive_sticky_and_span_mutants(self):
  mask=j.array([True]*4+[False]*4);tape,ticket,side,physical,move,write,mask=self.pending(mask=mask)
  tape=tape._replace(count=tape.count.at[2].set(64),switch_details=tape.switch_details.at[2,63,0].set(7),refusal=tape.refusal.at[1].set(64))
  out=swords.record_device(tape,ticket,side,physical,move,write,self.context,eligible=mask)
  self.assertEqual(out.count.tolist(),[2,2,64,2,0,0,0,0]);self.assertTrue(int(out.refusal[1])&64);self.assertTrue(int(out.refusal[2])&events.OVERFLOW)
  for a,b in zip(jax.tree.leaves(tape),jax.tree.leaves(out)):
   if getattr(a,'ndim',0) and a.shape[0]==8:np.testing.assert_array_equal(a[4:],b[4:])
  bad=self.context._replace(data=self.context.data._replace(binding=self.context.data.binding.at[0].add(j.uint32(1))))
  refused=swords.record_device(tape,ticket,side,physical,move,write,bad,eligible=mask);self.assertEqual(refused.count.tolist(),tape.count.tolist());self.assertTrue(bool((refused.refusal[:4]&256).all()))
  for field in self.context.data._fields[:-1]:
   malformed=self.context._replace(data=self.context.data._replace(**{field:getattr(self.context.data,field)[None,:]}))
   with self.assertRaises(ValueError):swords.record_device(tape,ticket,side,physical,move,write,malformed,eligible=mask)
  # Exact source-span read-only checks cannot accept a partial/migrated receipt.
  clean=self.pending(2);t,k,s,p,m,w,active=clean;complete=swords.record_device(t,k,s,p,m,w,self.context,eligible=active)
  for changed in (complete._replace(words=complete.words.at[0,1,5].set(1)),complete._replace(count=complete.count.at[0].add(-1)),complete._replace(event_generation=complete.event_generation.at[0,1,1].add(1))):
   matches=swords.span_matches(changed,k,s,p,m,w,self.context);self.assertEqual(matches.tolist(),[False,True])
 def reference_words(self,tape,row,owner):
  names=tuple(self.catalog.species.names[int(self.adapter.device.causal.species[i])] for i in (1,2));ident={i:f'p{i+1}a: '+names[i] for i in (0,1)}
  wire,_=public_wire.project_lines(public_wire.WireState(),[f'|switch|{ident[i]}|{names[i]}, L80, M|100/100' for i in (0,1)]+['|turn|1'],**v4.wire_kwargs(self.catalog));lines=[]
  for index,w in enumerate(np.array(tape.words[row,:int(tape.count[row])])):
   k=int(w[0])
   if k==events.MOVE:lines.append(f'|move|{ident[int(w[1])]}|Swords Dance|{ident[int(w[1])]}')
   elif k==32:lines.append(f'|-boost|{ident[int(w[1])]}|atk|2')
   elif k==life.TURNLOOP_PREFIX:
    self.assertEqual(w.tolist(),[35,-1,-1,-1,-1,0,0,0,0,0])
    lines.extend(['|','|t:|1'])
   elif k==life.ACTION_END:lines.append('|')
   elif k==life.UPKEEP:lines.append('|upkeep')
   elif k==life.TURN:lines.append(f'|turn|{int(tape.event_turn[row,index])}')
   else:self.fail('unexpected raw '+str(k))
  _,packets=public_wire.project_lines(wire,lines,**v4.wire_kwargs(self.catalog));return np.array([e.words() for p in packets for e in p.events],np.int32)
