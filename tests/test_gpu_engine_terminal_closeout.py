"""Literal final direct-KO source closeout; no games, models or scientific roots.

Light replacement coverage for terminal_source (#1819's version), with the same assertions. The birth
comes from direct_ko_components' compiled seeder and per-class cache; each instrumented step is built once
per (batch, enabled, prior) and reused read-only; the paired source-frame consumer is compiled once (every
input an argument, so each call keeps its own tables). The steps stay eager, as before.
"""
import ast
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_terminal_callbacks as terminal
from gpu_engine import engine_kernel_frame_risk as risk,engine_execution_completeness as ex
from gpu_engine import engine_lifecycle_callbacks as life,engine_retained_encoder_driver as driver
from gpu_engine import engine_source_frame_witness as witness,transition as tr
import test_gpu_engine_direct_ko_components as ko_fixture
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from gpu_engine_transition_certification import CURRENT_TRANSITION


class TerminalCloseoutTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  ko_fixture.DirectKOComponentTests.setUpClass();cls.f=ko_fixture.DirectKOComponentTests()
  catalog=cls.f.catalog
  cls.consume_pair=staticmethod(jax.jit(lambda carry,core,frame,streams,pubs,tables,active:witness.consume_pair_device(
   carry,core,frame,streams,pubs,tables,catalog,active=active)))
  cls.steps={}
 def same(self,a,b):
  for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 def actual(self,b=2,*,enabled=True,prior=0):
  key=(b,enabled,prior)
  if key not in self.steps:self.steps[key]=self.stepped(b,enabled=enabled,prior=prior)
  return self.steps[key]
 def stepped(self,b,*,enabled,prior):
  f=self.f;state,cursor,consumers=f.initial(b,last=True)
  tape=life._tape(cursor,f.producer,64)._replace(refusal=j.full(b,1|prior,j.int32))
  ledger=ex.initialize(cursor.stream_id,compiled=f.account)._replace(frame=j.ones(b,j.int32))
  tape=ex.begin_on_tape(tape,ledger,f.account.device,active=j.ones(b,bool))
  carry=risk.initialize_device(tape.stream_id,f.risks.device)._replace(frame=j.ones(b,j.int32))
  carry=risk.begin_frame_device(carry,tape,f.risks.device,active=j.ones(b,bool))
  tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,f.risks.device)))
  if enabled:tape=terminal.begin_on_tape(tape,state,f.context,active=j.ones(b,bool))
  actual=tr.step_with_chance(state,j.zeros((b,2),j.int32),f.table,f.chance(b),
      _public_trace=tape,_lifecycle_trace=True,_direct_hit_context=f.context)
  frame=life.LifecycleResult(actual.step.state,actual.callbacks.cursor,actual.callbacks,
      life.snapshot(cursor),life.snapshot(actual.callbacks.cursor),actual.step.invalid_actions,actual.step.placeholder_moves)
  return state,actual,frame,consumers
 def table(self,tape):
  f=self.f;z=j.zeros(8,j.uint32)
  return witness.Tables(j.arange(1,9,dtype=j.uint32),j.asarray(tape.engine_binding,j.uint32),j.asarray(tape.source_binding,j.uint32),
      j.asarray(tape.move_binding,j.uint32),j.asarray(tape.producer_binding,j.uint32),f.catalog.binding,f.account.device,f.risks.device,
      f.adapter.device,None,None,None,None,j.zeros_like(f.helper.device.ko_candidates),f.helper.device.binding,z,z)
 def pair(self,actual,frame,consumers,*,streams=None,active=None):
  b=len(frame.state.done);tables=self.table(actual.callbacks)
  streams=tuple(x[0] for x in consumers) if streams is None else streams
  pubs=tuple(x[1] for x in consumers)
  carry=witness.Carry(tables.binding,frame.cursor.stream_id,j.ones(b,j.int32),j.stack([x[0].sequence for x in consumers],axis=1),j.zeros(b,j.int32))
  core=driver.Core(frame.state,frame.cursor,None,None,None,None,None,j.ones(b,bool),j.ones(b,j.int32),
      execution=actual.callbacks.execution.ledger,risks=actual.callbacks.execution.risks.carry)
  return self.consume_pair(carry,core,frame,streams,pubs,tables,j.ones(b,bool) if active is None else active)
 def test_actual_last_hp_faint_literal_finalization_both_owners_and_default_identity(self):
  before,actual,frame,consumers=self.actual()
  self.same(actual.step.state,tr.step_with_chance(before,j.zeros((2,2),j.int32),self.f.table,self.f.chance()).state)
  self.assertEqual(frame.state.done.tolist(),[True,True]);self.assertEqual(frame.state.winner.tolist(),[0,0])
  self.assertEqual(actual.callbacks.refusal.tolist(),[1,1]);self.assertEqual(actual.callbacks.execution.risks.carry.risk.tolist(),[0,0])
  self.assertEqual(actual.callbacks.words[0,:int(actual.callbacks.count[0]),0].tolist(),[35,1,2,3])
  c=actual.callbacks.execution.terminal
  self.same(c.before_hp,before.hp);self.same(c.final_hp,frame.state.hp);self.same(c.final_max_hp,frame.state.max_hp)
  self.assertEqual(c.candidates.tolist(),[1,1]);self.assertEqual(c.finalized.tolist(),[1,1])
  self.assertEqual(terminal.matches_final_device(actual.callbacks,frame.state).tolist(),[True,True])
  paired=self.pair(actual,frame,consumers)
  self.assertEqual(paired.report.frame_inputs_complete.tolist(),[True,True],msg=str(dict(
      refusal=paired.report.refusal.tolist(),expected=paired.report.expected_cutpoints.tolist(),observed=paired.report.observed_cutpoints.tolist())))
  self.assertEqual(paired.terminal_closeout.tolist(),[True,True]);self.assertEqual(paired.ledger.count.tolist(),[2,2]);self.assertEqual(paired.ledger.site[:,:2].tolist(),[[int(ex.Site.TURNLOOP_PREFIX),int(ex.Site.DIRECT_HIT)]]*2);self.assertEqual(paired.ledger.stage[:,:2].tolist(),[[3,3]]*2)
  for owner,result in enumerate(paired.owners):
   self.assertEqual(result.projection.error.tolist(),[0,0]);self.assertFalse(bool(result.projection.B4_qualified.any()))
   for row in (0,1):np.testing.assert_array_equal(self.f.f.words(result.projection,row),self.f.reference_words(actual.callbacks,row,owner))
  core=SimpleNamespace(engine=frame.state,admitted=j.ones(2,bool),terminal_closeout=paired.terminal_closeout)
  self.assertEqual(driver.terminal_complete(core).tolist(),[True,True])
 def test_missing_wrongwinner_stale_or_other_risk_never_close(self):
  _,actual,frame,consumers=self.actual();tape=actual.callbacks;c=tape.execution.terminal
  for changed in (c._replace(final_winner=j.ones(2,j.int32)),c._replace(final_hp=c.final_hp.at[0,0,0].add(-1)),
      c._replace(generation=c.generation.at[0,0].add(1)),c._replace(candidates=j.zeros(2,j.int32)),
      c._replace(finalized=j.zeros(2,j.int32)),c._replace(frame=c.frame+1)):
   bad=tape._replace(execution=tape.execution._replace(terminal=changed))
   self.assertFalse(bool(terminal.matches_final_device(bad,frame.state)[0]))
  bad=tape._replace(execution=tape.execution._replace(terminal=None))
  self.assertFalse(bool(terminal.matches_final_device(bad,frame.state).any()))
  carry=tape.execution.risks.carry._replace(risk=j.full(2,risk.BOOST,j.int32))
  still=risk.observe_end_device(carry,tape,frame.state,self.f.risks.device,active=j.ones(2,bool))
  self.assertEqual(still.risk.tolist(),[risk.BOOST]*2)
  denied=risk.observe_end_device(carry,bad,frame.state,self.f.risks.device,active=j.ones(2,bool))
  self.assertEqual(denied.risk.tolist(),[risk.BOOST|risk.TERMINAL]*2)
  _,default,_,_=self.actual(enabled=False)
  self.assertTrue(bool((default.callbacks.refusal&256).all()));self.assertTrue(bool((default.callbacks.execution.risks.carry.risk&risk.TERMINAL).all()))
 def test_public_peer_fault_prior_unknown_and_inactive_rollback(self):
  _,actual,frame,consumers=self.actual();streams=tuple(x[0] for x in consumers)
  broken=streams[1]._replace(sequence=streams[1].sequence.at[0].add(1))
  bad=self.pair(actual,frame,consumers,streams=(streams[0],broken))
  self.assertEqual(bad.terminal_closeout.tolist(),[False,True])
  for owner in (0,1):self.same(bad.owners[owner].public.cursor.sequence[0],consumers[owner][1].cursor.sequence[0])
  inactive=self.pair(actual,frame,consumers,active=j.array([False,True]))
  self.assertEqual(inactive.terminal_closeout.tolist(),[False,True])
  self.same(inactive.owners[0].public.cursor.sequence[0],consumers[0][1].cursor.sequence[0])
  _,prior,priorframe,pubs=self.actual(prior=128)
  self.assertTrue(bool((prior.callbacks.refusal&128).all()))
  self.assertFalse(bool(self.pair(prior,priorframe,pubs).terminal_closeout.any()))
 def test_default_whole_program_and_nonlethal_validator_unchanged(self):
  base='185868792ac7c4e61f2df42273b025407b4ca5d4'
  before=ast.parse(subprocess.check_output(['git','show',base+':gpu_engine/transition.py'],text=True));after=ast.parse(subprocess.check_output(['git','show','1511dc02:gpu_engine/transition.py'],text=True))
  expected=dump_default_program(before);self.assertEqual(dump_default_program(after),expected);assert_hp_rng_mutations_detected(self,after,expected)
  # Recertify every current statement independently of the historical delta.
  current=ast.parse(Path('gpu_engine/transition.py').read_text())
  integrated=ast.parse(subprocess.check_output(['git','show',CURRENT_TRANSITION+':gpu_engine/transition.py'],text=True))
  self.assertEqual(ast.dump(current),ast.dump(integrated))
  assert_hp_rng_mutations_detected(self,current,dump_default_program(integrated))
  old=ast.parse(subprocess.check_output(['git','show',base+':gpu_engine/engine_direct_hit_controls.py'],text=True));new=ast.parse(Path('gpu_engine/engine_direct_hit_controls.py').read_text())
  for name in ('_valid_write','validate_write_device','validate_raw_device'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(a),ast.dump(b))


class TerminalTransportAndCountingTests(unittest.TestCase):
 def context(self,b=8):
  glob=j.arange(8,dtype=j.uint32);i=j.zeros(b,j.int32);no=j.zeros(b,bool)
  return terminal.Context(*(glob for _ in terminal._GLOBAL),j.arange(1,b+1,dtype=j.int32),j.ones(b,j.int32),i,i,j.zeros((b,2),j.int32),~no,
      j.ones((b,2,6),j.int32),j.ones((b,2,6),j.int32),no,j.full(b,-1,j.int32),i,i,i,i,i,i,
      j.zeros((b,2,6),j.int32),j.ones((b,2,6),j.int32),no,j.full(b,-1,j.int32),i)
 def test_b8_global_and_original_snapshot_cannot_be_row_selected_or_mutated(self):
  old=self.context();active=j.array([True,False]*4)
  new=old._replace(candidates=j.ones(8,j.int32),finalized=j.ones(8,j.int32))
  good=jax.jit(terminal.commit_rows)(active,old,new)
  self.assertEqual(good.candidates.tolist(),[1,0]*4)
  np.testing.assert_array_equal(good.engine_binding,old.engine_binding)
  changed=new._replace(before_hp=new.before_hp.at[0,0,0].set(9))
  self.assertEqual(terminal.commit_rows(active,old,changed).candidates.tolist(),[0,0,1,0,1,0,1,0])
  for name in terminal._GLOBAL:
   changed=new._replace(**{name:getattr(new,name).at[7].add(1)})
   self.assertEqual(terminal.commit_rows(active,old,changed).candidates.tolist(),[0]*8)
   for malformed in (j.zeros(1,j.uint32),j.zeros((8,1),j.uint32),j.zeros(8,j.int32)):
    with self.assertRaises(ValueError):terminal.commit_rows(active,old,new._replace(**{name:malformed}))
 def test_done_without_genuine_closeout_is_refused_without_another_forward(self):
  import test_gpu_engine_benchmark_retained_games as fixture
  from gpu_engine import champion
  first,config=fixture.result(batch=2)
  def network(_,args):return champion.Output(j.zeros((4,9)),j.zeros(4),j.zeros((4,9))),0.,1.
  def advance(session,actions,*,active_rows,timing_callback):
   out,_=fixture.result(batch=2,frames=[1,1],done=[True,True])
   out.state.core.terminal_closeout=j.array([False,True])
   session.state,session.result=out.state,out
   session.diagnostics=SimpleNamespace(active_rows=active_rows,executed_rows=active_rows,invalid_actions=j.zeros(2,bool),placeholder_moves=j.zeros(2,j.int32))
   for name in ('engine','requests','public_p1','public_p2','encode_p1','encode_p2','admission_p1','admission_p2'):timing_callback(name,.001)
   return out
  measured,model,stepped=fixture.RetainedBenchmarkTests().measure(first,config,network=network,advance=advance,cap=1)
  self.assertEqual((measured['completed_games'],measured['refused_games']), (1,1))
  self.assertEqual((model.call_count,stepped.call_count),(1,1))
  self.assertFalse(measured['full_B4_qualified']);self.assertFalse(measured['fidelity_qualified'])
 def test_original_collector_does_not_label_done_without_source_closeout(self):
  from contextlib import ExitStack
  import test_gpu_engine_v5_collection as fixture
  fixture.V5CollectionTests.setUpClass();f=fixture.V5CollectionTests()
  with ExitStack() as stack:
   session,actor,ledger,calls,indices=f.setup_runtime(stack)
   original_advance=f.c.driver.advance.side_effect
   def missing(*args,**kwargs):
    result=original_advance(*args,**kwargs)
    result.state.core.terminal_closeout=j.zeros(1,bool)
    return result
   f.c.driver.advance.side_effect=missing
   out=f.c.collect_session(session,actor=actor,pool=object(),team_indices=indices,ledger=ledger,max_requests=1,sampling_seed=97,enable_v5=True)
   self.assertEqual(out.ledger['counts']['COMPLETE'],0);self.assertEqual(out.ledger['counts']['REFUSED'],1)
   self.assertTrue(all(d.row.labels.outcome is None for d in out.decisions))

if __name__=='__main__':unittest.main()
