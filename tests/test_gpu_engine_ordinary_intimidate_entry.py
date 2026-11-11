"""Ordinary (mid-battle) Intimidate entry through both FULL owners; no step kernel, game or model.

Light replacement coverage for ordinary_intimidate_owning, with the same fixtures and assertions. The full
consumer is compiled once (ledger, frame, stream, public and the active mask as arguments) instead of being
called eagerly six times, and the B=8 guard test builds its pre-entry tape without consumers: a fresh
accounting frame over the real initialized cursor and risk carry (the guards read neither prior consumption).
The FULL words are checked against Showdown's lines exactly as before.
"""
import ast
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


def default_entry_program(text):
 tree=ast.parse(text)
 fields=('spikes_type_count','spikes_item_count','spikes_boots_item')
 table=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='EntryTable')
 added=[n for n in table.body if isinstance(n,ast.AnnAssign)and n.target.id in fields]
 if added:
  assert tuple(n.target.id for n in added)==fields
  expected=ast.parse("x: jax.Array | None = None").body[0]
  assert all(ast.dump(n.annotation)==ast.dump(expected.annotation)and ast.dump(n.value)==ast.dump(expected.value)for n in added)
  table.body=[n for n in table.body if n not in added]
  factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='from_metadata')
  arguments=factory.body[-1].value.args
  approved=ast.parse("x(jnp.asarray(len(vocab['types']),jnp.int32),jnp.asarray(len(vocab.get('items',[])),jnp.int32),jnp.asarray(next((i for i,name in enumerate(vocab.get('items',[])) if name=='heavydutyboots'),0),jnp.int32))").body[0].value.args
  assert [ast.dump(n)for n in arguments[-3:]]==[ast.dump(n)for n in approved]
  del arguments[-3:]
 return dump_default_program(tree)


class OrdinaryIntimidateEntryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  initial_fixture.InitialIntimidateOwningTests.setUpClass();f=initial_fixture.InitialIntimidateOwningTests
  for name in ('f','metadata','catalog','table','producer','adapter','account','risk','kw'):setattr(cls,name,getattr(f,name))
  # One compiled full consumer for the module: every input is an argument, the tables are the module's.
  cls.consume=staticmethod(jax.jit(lambda ledger,frame,stream,public,active:ex.consume_frame_device(ledger,frame,stream,public,
   active=active,tables=cls.account.device,adapter_tables=cls.adapter.device,catalog=cls.catalog)))
 state=initial_fixture.InitialIntimidateOwningTests.state
 initial=initial_fixture.InitialIntimidateOwningTests.initial
 same=initial_fixture.InitialIntimidateOwningTests.same
 def initialized(self,b):
  s=self.state(b);species=j.full_like(s.species,2).at[:,:,3].set(1)
  ability=j.full_like(s.ability,2).at[:,:,3].set(1)
  return self.initial(s._replace(species=species,ability=ability,base_ability=ability))
 def before(self,b=2):
  frame=self.initialized(b);closed=[];observers=[]
  for owner in(0,1):
   stream=full.register(frame.cursor.stream_id,j.full(b,owner,j.int32),compiled=self.adapter)
   public=v4.initialize(self.catalog,stream_ids=frame.cursor.stream_id,owner_sides=j.full(b,owner,j.int32))
   ledger,result=self.consume(frame.callbacks.execution.ledger,frame,stream,public,j.ones(b,bool))
   self.assertEqual(result.projection.error.tolist(),[0]*b);closed.append(ledger);observers.append(result)
  ledger,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
  tape=life._tape(frame.cursor,self.producer,64)
  tape=ex.begin_on_tape(tape,ledger,self.account.device,active=j.ones(b,bool))
  carry=risks.begin_frame_device(frame.callbacks.execution.risks.carry,tape,self.risk.device,active=j.ones(b,bool))
  tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,self.risk.device)))
  return frame.state,tape,observers
 def before_tape(self,b):
  # The same pre-entry tape without the owner consumers: the real initialized state, cursor and risk
  # carry, with a fresh accounting ledger placed at the birth ledger's frame, so the entry opens frame 2
  # exactly as after the consumed birth and the risk carry opens with it. The entry guards then decide
  # each row on their own (a closed carry would refuse every row whatever the guards do).
  frame=self.initialized(b)
  tape=life._tape(frame.cursor,self.producer,64)
  ledger=ex.initialize(frame.cursor.stream_id,compiled=self.account)._replace(frame=frame.callbacks.execution.ledger.frame)
  tape=ex.begin_on_tape(tape,ledger,self.account.device,active=j.ones(b,bool))
  carry=risks.begin_frame_device(frame.callbacks.execution.risks.carry,tape,self.risk.device,active=j.ones(b,bool))
  self.assertEqual(carry.opened.tolist(),[True]*b);self.assertEqual(carry.risk.tolist(),[0]*b)
  self.assertEqual(tape.execution.ledger.frame.tolist(),[2]*b)
  tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,self.risk.device)))
  return frame.state,tape
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
  closed=[]
  for owner,prior in enumerate(observers):
   ledger,result=self.consume(out.execution.ledger,frame,prior.projection.state,prior.public,j.ones(2,bool))
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertEqual(ledger.stage[:,0].tolist(),[3,3]);closed.append(ledger)
   for row,actor in enumerate((0,1)):
    lines=['|switch|p1a: Celebi|Celebi, L80, M|100/100','|switch|p2a: Celebi|Celebi, L80, M|100/100','|turn|1']
    wire,_=public_wire.project_lines(public_wire.WireState(),lines,**v4.wire_kwargs(self.catalog))
    _,packets=public_wire.project_lines(wire,[f'|switch|p{actor+1}a: Arbok|Arbok, L80, M|100/100',f'|-ability|p{actor+1}a: Arbok|Intimidate|boost',f'|-unboost|p{2-actor}a: Celebi|atk|1'],**v4.wire_kwargs(self.catalog))
    expected=np.array([event.words()for packet in packets for event in packet.events],np.int32)
    got=self.f.words(result.projection,row)
    np.testing.assert_array_equal(got,expected)
    self.assertEqual(int(result.public.mons.boosts[row,1-actor,0]),-1)
  _,good=ex.join_consumed_owners(*closed,tables=self.account.device);self.assertTrue(bool(good.all()))
 def test_b8_current_prefix_and_write_guards_keep_requested_unknowns(self):
  state,tape=self.before_tape(8);side=j.zeros(8,j.int32);state,tape=self.install(state,tape,side)
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
  # Positive control on the same consumer-free tape: with no guard triggered every active row captures its
  # exact emission (stage 2) and the inactive row enters nothing, so each refusal above is its own guard's.
  clean_state,clean_tape=self.before_tape(8);clean_state,clean_tape=self.install(clean_state,clean_tape,side)
  _,_,clean=entry.enter(clean_state,side,self.table.entry,eligible=mask,hazards=False,_weather_trace=clean_tape)
  self.assertEqual(clean.execution.ledger.stage[:7,0].tolist(),[2]*7);self.assertEqual(int(clean.execution.ledger.count[7]),0)
 def test_original_default_ast_constructor_api_and_source_callback(self):
  path=Path('gpu_engine/entry.py');old=subprocess.check_output(['git','show','f3c9dbea:'+str(path)],text=True)
  self.assertEqual(default_entry_program(old),default_entry_program(path.read_text()))
  for old_numeric,new_numeric in [('jnp.maximum(-6,old-1)','jnp.maximum(-5,old-1)'),
      ('jnp.maximum(-6,old-1)','jnp.maximum(-6,old-2)')]:
   self.assertIn(old_numeric,path.read_text())
   self.assertNotEqual(default_entry_program(old),
    default_entry_program(path.read_text().replace(old_numeric,new_numeric)))
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
