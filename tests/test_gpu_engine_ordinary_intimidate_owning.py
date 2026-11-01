"""Actual entry function/FULL components only; no step kernel, game or model."""
import ast
import json
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import entry,engine_lifecycle_callbacks as life,engine_public_events as events
from gpu_engine import engine_intimidate_callbacks as intimidate,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risks,engine_full_v4_event_device as full,v4_encoder as v4,public_wire
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine_default_observer_ast import dump_default_program
import test_gpu_engine_intimidate_owning_event as initial_fixture


class OrdinaryIntimidateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  initial_fixture.InitialIntimidateOwningTests.setUpClass();f=initial_fixture.InitialIntimidateOwningTests
  for name in ('f','metadata','catalog','table','producer','adapter','account','risk','kw'):setattr(cls,name,getattr(f,name))
 state=initial_fixture.InitialIntimidateOwningTests.state
 initial=initial_fixture.InitialIntimidateOwningTests.initial
 same=initial_fixture.InitialIntimidateOwningTests.same
 def before(self,b=2):
  s=self.state(b);species=j.full_like(s.species,2).at[:,:,3].set(1)
  ability=j.full_like(s.ability,2).at[:,:,3].set(1)
  s=s._replace(species=species,ability=ability,base_ability=ability)
  frame=self.initial(s);closed=[];observers=[]
  for owner in(0,1):
   stream=full.register(frame.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter)
   public=v4.initialize(self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=j.full(b,owner,j.int32))
   ledger,result=ex.consume_frame_device(frame.callbacks.execution.ledger,frame,stream,public,
    active=j.ones(b,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog)
   self.assertEqual(result.projection.error.tolist(),[0]*b);closed.append(ledger);observers.append(result)
  ledger,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
  tape=life._tape(frame.cursor,self.producer,64)
  tape=ex.begin_on_tape(tape,ledger,self.account.device,active=j.ones(b,bool))
  carry=risks.begin_frame_device(frame.callbacks.execution.risks.carry,tape,self.risk.device,active=j.ones(b,bool))
  tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,self.risk.device)))
  return frame.state,tape,observers
 def install(self,state,tape,side,*,active=None):
  b=len(tape.count);mask=j.ones(b,bool)if active is None else active;rows=j.arange(b)
  state=entry.depart(state,side,self.table.entry,eligible=mask)
  outgoing=state.active[rows,side];incoming=j.full(b,3,j.int32)
  state=state._replace(active=state.active.at[rows,side].set(j.where(mask,incoming,outgoing)))
  tape=life.installed(tape,state,side,outgoing,incoming,eligible=mask,replacement=j.zeros(b,bool),baton=j.zeros(b,bool))
  return state,tape
 def test_actual_mirrored_entry_write_both_full_exact_words_and_numeric_identity(self):
  # Both source/target seats; nonzero incoming physical and public slot.
  state,tape,observers=self.before();side=j.array([0,1],j.int32)
  state,tape=self.install(state,tape,side)
  actual,ko,out=entry.enter(state,side,self.table.entry,eligible=j.ones(2,bool),hazards=False,_weather_trace=tape)
  plain,plain_ko=entry.enter(state,side,self.table.entry,eligible=j.ones(2,bool),hazards=False)
  self.same(actual,plain);self.same(ko,plain_ko)
  self.assertEqual(out.count.tolist(),[2,2]);self.assertEqual(out.execution.ledger.site[:,0].tolist(),[9,9])
  self.assertEqual(out.execution.ledger.stage[:,0].tolist(),[2,2]);self.assertEqual(out.refusal.tolist(),[1,1])
  # Actual source-witness raw/site routing with the SAME original static risk
  # tables. This is its pure mapping component, not full-frame geometry proof.
  source_tables=SimpleNamespace(risk=self.risk.device,secondary_weather=j.zeros(len(self.table.power),bool),
    direct_hit_binding=j.zeros(8,j.uint32),substitute_binding=j.zeros(8,j.uint32),toxic_binding=j.zeros(8,j.uint32))
  self.assertEqual(witness._raw_complete(out,source_tables).tolist(),[True,True])
  wrong=out._replace(execution=out.execution._replace(ledger=out.execution.ledger._replace(
    site=out.execution.ledger.site.at[:,0].set(int(ex.Site.INTIMIDATE_INITIAL)))))
  self.assertEqual(witness._raw_complete(wrong,source_tables).tolist(),[False,False])
  frame=life.LifecycleResult(actual,out.cursor,out,life.snapshot(tape.cursor),life.snapshot(out.cursor),j.zeros(2,bool),j.zeros(2,j.int32))
  # BEFORE registry must precede the actual installation.
  frame=frame._replace(before_registry=life.snapshot(observers[0].projection.state.causal.cursor))
  closed=[];captures=[]
  for owner,prior in enumerate(observers):
   ledger,result=ex.consume_frame_device(out.execution.ledger,frame,prior.projection.state,prior.public,
    active=j.ones(2,bool),tables=self.account.device,adapter_tables=self.adapter.device,catalog=self.catalog)
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,0].tolist(),[3,3]);closed.append(ledger)
   for row,actor in enumerate((0,1)):
    lines=['|switch|p1a: Celebi|Celebi, L80, M|100/100','|switch|p2a: Celebi|Celebi, L80, M|100/100','|turn|1']
    wire,_=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
    _,packets=public_wire.project_lines(wire,[f'|switch|p{actor+1}a: Arbok|Arbok, L80, M|100/100',f'|-ability|p{actor+1}a: Arbok|Intimidate|boost',f'|-unboost|p{2-actor}a: Celebi|atk|1'],**v4.wire_kwargs(self.catalog))
    expected=np.array([event.words()for packet in packets for event in packet.events],np.int32)
    got=self.f.words(result.projection,row)
    captures.append(dict(owner=owner,row=row,actual=got.tolist(),expected=expected.tolist()))
    Path('/tmp/pokezero-ordinary-intimidate-full-words.json').write_text(json.dumps(captures)+'\n')
    np.testing.assert_array_equal(got,expected)
    self.assertEqual(int(result.public.mons.boosts[row,1-actor,0]),-1)
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_b8_current_prefix_and_write_guards_keep_requested_unknowns(self):
  state,tape,_=self.before(8);side=j.zeros(8,j.int32);state,tape=self.install(state,tape,side)
  # cap, immune, Substitute, absent foe, stale install stamp, foreign actor,
  # prior sticky refusal, inactive row. No empty-span exemption.
  state=state._replace(boosts=state.boosts.at[0,1,0,0].set(-6),ability=state.ability.at[1,1,0].set(3),
    substitute_hp=state.substitute_hp.at[2,1,0].set(10),hp=state.hp.at[3,1,0].set(0))
  tape=tape._replace(event_generation=tape.event_generation.at[4,0,0].add(1),
    words=tape.words.at[5,0,1].set(1),refusal=tape.refusal.at[6].set(257))
  mask=j.arange(8)<7
  plain,ko=entry.enter(state,side,self.table.entry,eligible=mask,hazards=False)
  actual,actual_ko,out=entry.enter(state,side,self.table.entry,eligible=mask,hazards=False,_weather_trace=tape)
  self.same(plain,actual);self.same(ko,actual_ko)
  self.assertEqual(out.execution.ledger.stage[:7,0].tolist(),[1]*7)
  self.assertTrue(bool((out.refusal[:7]&512).all()));self.assertEqual(out.count.tolist(),[1]*8)
  self.assertEqual(int(out.refusal[6])&257,257);self.assertEqual(int(out.execution.ledger.count[7]),0)
  self.same(tape.cursor.generation,out.cursor.generation)
 def test_original_default_ast_constructor_api_and_source_callback(self):
  path=Path('gpu_engine/entry.py');old=subprocess.check_output(['git','show','f3c9dbea:'+str(path)],text=True)
  self.assertEqual(dump_default_program(ast.parse(old)),dump_default_program(ast.parse(path.read_text())))
  for old_numeric,new_numeric in [('jnp.maximum(-6,old-1)','jnp.maximum(-5,old-1)'),
      ('jnp.maximum(-6,old-1)','jnp.maximum(-6,old-2)')]:
   self.assertIn(old_numeric,path.read_text())
   self.assertNotEqual(dump_default_program(ast.parse(old)),
    dump_default_program(ast.parse(path.read_text().replace(old_numeric,new_numeric))))
  original=ast.parse(subprocess.check_output(['git','show','f3c9dbea:gpu_engine/engine_intimidate_callbacks.py'],text=True))
  current=ast.parse(Path('gpu_engine/engine_intimidate_callbacks.py').read_text())
  for name in('_constructor','record_device'):
   self.assertEqual(ast.dump(next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name==name)),ast.dump(next(n for n in current.body if isinstance(n,ast.FunctionDef)and n.name==name)))
  initial_fixture.InitialIntimidateOwningTests.test_default_entry_ast_unchanged_and_original_gen3_callback_pair(self)
 def test_all_four_global_namespaces_require_exact_uint32_eight_before_context(self):
  names=('engine_binding','source_binding','move_binding','producer_binding')
  fields={name:j.arange(8,dtype=j.uint32)for name in names}
  fields['count']=j.zeros(8,j.int32)
  tape=SimpleNamespace(**fields);vector=j.zeros(8,j.int32)
  write=intimidate.Write(*(vector for _ in range(6)))
  def emit(value):
   return intimidate.record_ordinary_device(value,None,vector,vector,vector,write,
    self.table.entry,requested=j.ones(8,bool))
  self.assertIs(emit(tape),tape) # No original context: no receipt/admission.
  for name in names:
   for malformed in (j.zeros((1,8),j.uint32),j.zeros((8,8),j.uint32),
       j.zeros((0,),j.uint32),j.zeros((8,),j.int32),j.zeros((8,),j.float32)):
    with self.subTest(name=name,shape=malformed.shape,dtype=malformed.dtype):
     with self.assertRaises(ValueError):emit(SimpleNamespace(**dict(fields,**{name:malformed})))


if __name__=='__main__':unittest.main()
