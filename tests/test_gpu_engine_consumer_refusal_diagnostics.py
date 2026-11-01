"""Private observations only: original consumers/fake packets, no engine turns."""
import ast
from collections import namedtuple
from pathlib import Path
import subprocess
import unittest
from unittest import mock
from types import SimpleNamespace as NS
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_full_v4_event_device as full
from gpu_engine import engine_execution_completeness as ex
from gpu_engine import engine_source_frame_witness as witness
from gpu_engine import benchmark_capture_gap_census as census

BASE='1ea9e1b0f4ebf9b5079107bb2c95151e285ed862'

def old_function(module,name):
 tree=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+Path(module.__file__).name],text=True))
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
 env=dict(vars(module));exec(compile(ast.Module(body=[node],type_ignores=[]),'immutable-original','exec'),env)
 return env[name]

def same(test,left,right):
 a=jax.tree.leaves(left);b=jax.tree.leaves(right);test.assertEqual(len(a),len(b))
 for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

Cursor=namedtuple('Cursor','sequence stream_id turn phase admitted')
Causal=namedtuple('Causal','cursor owner admitted')
Mons=namedtuple('Mons','species status hp_numerator hp_denominator')
Field=namedtuple('Field','weather weather_ability weather_set weather_upkeeps')
Context=namedtuple('Context','toxic field')
Toxic=namedtuple('Toxic','visibility')
Public=namedtuple('Public','cursor mons context')

def fixtures(b=4):
 seq=j.arange(b,dtype=j.int32)+10;z=j.zeros(b,j.int32);yes=j.ones(b,bool)
 cursor=Cursor(seq,j.arange(b,dtype=j.int32)+500,z,z,yes)
 species=j.ones((b,2,6),j.int32);status=j.zeros_like(species);hp=species*90;maximum=species*100
 owner=j.arange(b,dtype=j.int32)%2
 visibility=j.where(j.arange(2)[None,:]==owner[:,None],full.public_toxic.EXACT,full.public_toxic.PERCENTAGE)
 state=full.StreamState(Causal(cursor,owner,yes),seq,j.ones((b,2),bool),species,status,z,j.zeros(b,bool),z,hp,maximum)
 public=Public(cursor,Mons(species,status,hp,maximum),Context(Toxic(visibility),Field(z,j.zeros(b,bool),j.zeros(b,bool),z)))
 candidate=state._replace(sequence=seq+1,causal=state.causal._replace(cursor=cursor._replace(sequence=seq+1)))
 chunk=full.inputs.EventChunk(cursor.stream_id,j.zeros((b,1,16),j.int32),j.ones((b,1),bool),j.zeros(b,bool),*[j.zeros(8,j.uint32)]*6)
 projection=full.Projection(candidate,chunk,z,z,yes,j.zeros(b,bool),j.zeros(b,bool),j.zeros(b,bool),
   full.FrameChunks(j.zeros((1,b,1,16),j.int32),j.ones((1,b,1),bool),j.zeros((1,b),bool)))
 return state,public,projection

class ConsumerDiagnosticTests(unittest.TestCase):
 def test_full_errors_and_context_predicates_survive_exact_original_rollback(self):
  state,public,projected=fixtures();projected=projected._replace(error=j.array([0,6,0,0],j.int32))
  public=public._replace(mons=public.mons._replace(hp_numerator=public.mons.hp_numerator.at[3,0,0].add(1)))
  def fold(current,chunk,catalog):
   return current._replace(cursor=current.cursor._replace(sequence=current.cursor.sequence+j.sum(chunk.valid,axis=1,dtype=j.int32))),j.array([0,0,9,0],j.int32)
  with mock.patch.object(full,'project_frame',return_value=projected),mock.patch.object(full.v4_encoder,'fold_chunk',side_effect=fold),mock.patch.object(full,'_commit_public',side_effect=full._commit):
   old=old_function(full,'project_and_fold');expected=old(state,None,None,public,None);actual=full.project_and_fold(state,None,None,public,None)
  same(self,expected,actual._replace(projection=actual.projection._replace(fold_diagnostics=None)))
  d=dict(zip(full.FOLD_DIAGNOSTIC_FIELDS,np.array(actual.projection.fold_diagnostics).T.tolist()))
  self.assertEqual(d['projection_error'],[0,6,0,0]);self.assertEqual(d['fold_error'],[0,0,9,0])
  self.assertEqual(d['prior_context_equal'],[1,1,1,0]);self.assertEqual(d['prior_hp_equal'],[1,1,1,0])
  self.assertEqual(d['projected_word_count'],[1]*4)
  self.assertEqual(actual.projection.component_admitted.tolist(),[True,False,False,False])
  self.assertEqual(actual.projection.frames.valid[:,1:].any().item(),False)
  self.assertEqual(actual.public.cursor.sequence.tolist(),[11,11,12,13])
  # Every original context leaf has an independent literal equality column.
  for key in full.FOLD_DIAGNOSTIC_FIELDS[3:16]:self.assertTrue(all(v in(0,1)for v in d[key]))

 def test_accounting_drains_original_failures_identically_and_retains_unmasked_error(self):
  state,public,p=fixtures();b=4;z=j.zeros(b,j.int32);one=j.ones(b,j.int32);yes=j.ones(b,bool)
  table=ex.Tables(*[j.zeros(8,j.uint32)]*4)
  l=ex.Ledger(state.causal.cursor.stream_id,table.inventory_binding,one,j.array([True,False,True,True]),one,z,z,one,
    one[:,None],z[:,None],one[:,None],one[:,None]*2,z[:,None],one[:,None],z[:,None],z[:,None],j.zeros((b,1,2),j.int32),
    one[:,None],j.zeros((b,1,10),j.int32),z[:,None],z[:,None],j.zeros((b,1,2),j.int32),z[:,None],j.zeros((b,1,6),j.int32),j.zeros((b,1,8),j.uint32))
  tape=NS(words=l.words,count=one,event_turn=l.event_turn,event_phase=l.event_phase,event_generation=l.event_generation,refusal=z)
  frame=NS(callbacks=tape);p=p._replace(error=j.array([0,0,6,0],j.int32),component_admitted=j.array([True,True,False,True]))
  consumed_public=public._replace(cursor=public.cursor._replace(sequence=state.sequence+1+j.array([0,0,0,1],j.int32)))
  kw=dict(active=yes,tables=table,adapter_tables=None,catalog=None)
  with mock.patch.object(ex,'_validate',return_value=yes),mock.patch.object(ex,'_tape',return_value=yes),mock.patch.object(full,'project_and_fold',return_value=full.Consumed(p,consumed_public)),mock.patch.object(full,'_commit_public',side_effect=full._commit):
   old=old_function(ex,'consume_frame_device');expected=old(l,frame,state,public,**kw);actual=ex.consume_frame_device(l,frame,state,public,**kw)
  same(self,expected,(actual[0],actual[1]._replace(projection=actual[1].projection._replace(consumption_diagnostics=None))))
  d=dict(zip(ex.CONSUMPTION_DIAGNOSTIC_FIELDS,np.array(actual[1].projection.consumption_diagnostics).T.tolist()))
  self.assertEqual(d['ledger_opened'],[1,0,1,1]);self.assertEqual(d['projection_error'],[0,0,6,0])
  self.assertEqual(d['public_sequence_equal'],[1,1,1,0]);self.assertEqual(d['drained'],[1,1,0,0])
  self.assertEqual(actual[0].refusal.tolist(),[0,32,32,32]);self.assertEqual(actual[0].stage[:,0].tolist(),[3,2,2,2])
  self.assertEqual(actual[1].projection.component_admitted.tolist(),[True,False,False,False])

 def test_packet_copy_rows_missing_malformed_and_nonexclusive_counts(self):
  b=2048;fold=np.ones((b,2,len(full.FOLD_DIAGNOSTIC_FIELDS)),np.int32);fold[:,:,0:2]=0
  accounting=np.ones((b,2,len(ex.CONSUMPTION_DIAGNOSTIC_FIELDS)),np.int32);accounting[:,:,11]=0
  owner=np.ones((b,2,len(witness.OWNER_DRAIN_DIAGNOSTIC_FIELDS)),np.int32);drained=np.ones((b,2),bool);joint=np.ones(b,bool)
  fold[0,1,1]=6;fold[0,1,10]=0;accounting[0,1,9:11]=0;owner[0,1,-1]=0;drained[0,1]=False;joint[0]=False
  value=witness.ConsumerDiagnostics(fold,accounting,owner,drained,joint);selected=np.zeros(b,bool);selected[[0,2]]=True
  packets=[{}for _ in range(b)];copies={}
  def host(x):copies[id(x)]=copies.get(id(x),0)+1;return np.asarray(x)
  census._attach_consumer_diagnostics(packets,value,selected,host=host)
  self.assertEqual(list(copies.values()),[1]*5);self.assertEqual(packets[1],{})
  self.assertEqual(packets[0]['consumer_diagnostics']['owners'][1]['fold']['fold_error'],6)
  self.assertIn('owner1/FULL_prior_hp_equal',packets[0]['consumer_failures']);self.assertIn('BOTH_OWNER_JOIN',packets[0]['consumer_failures'])
  self.assertEqual(packets[2]['consumer_failures'],[])
  for malformed in (None,value._replace(fold=fold.astype(np.float32)),value._replace(joined=joint[:2]),value._replace(owner_checks=np.full_like(owner,2))):
   q=[{}for _ in range(b)];census._attach_consumer_diagnostics(q,malformed,selected,host=np.asarray)
   self.assertFalse(q[0]['consumer_diagnostics']['available']);self.assertEqual(q[0]['consumer_failures'],['UNRECORDED']);self.assertEqual(q[1],{})
  ledger=census.Ledger(census.Plan(),np.arange(1000,3048,dtype=np.int32))
  for step in(0,1):ledger.observe(step=step,active=selected,executed=selected,engine_failed=np.zeros(b,bool),done=np.zeros(b,bool),turns=np.ones(b,np.int32),packet=lambda r:packets[r],terminal=lambda r:{})
  table={v['kind']:v for v in ledger.summary()['unsupported']}
  self.assertEqual(table['consumer:BOTH_OWNER_JOIN']['occurrences'],2);self.assertEqual(table['consumer:BOTH_OWNER_JOIN']['games_affected'],1)

 def test_advance_actual_fake_packets_calls_and_fallback_remain_original(self):
  from test_gpu_engine_census_host_copy import HostCopyTests,Device
  fixture=HostCopyTests();b=8
  before,_,old_calls,active,scored=fixture.make_backend(b,False)
  after,tracked,new_calls,_,_=fixture.make_backend(b,False)
  def diagnostic(wrap):
   fold=np.ones((b,2,len(full.FOLD_DIAGNOSTIC_FIELDS)),np.int32);fold[:,:,:2]=0
   accounting=np.ones((b,2,len(ex.CONSUMPTION_DIAGNOSTIC_FIELDS)),np.int32);accounting[:,:,11]=0
   return witness.ConsumerDiagnostics(wrap(fold),wrap(accounting),wrap(np.ones((b,2,len(witness.OWNER_DRAIN_DIAGNOSTIC_FIELDS)),np.int32)),wrap(np.ones((b,2),bool)),wrap(np.ones(b,bool)))
  def wrap(x):v=Device(x);tracked.append(v);return v
  after.session.state.core.frame_input_report.consumer_diagnostics=diagnostic(wrap)
  tree=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py'],text=True))
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_Runtime');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='advance')
  env=dict(vars(census),_row_select=lambda mask,old,new:new);exec(compile(ast.Module(body=[fn],type_ignores=[]),'immutable-census','exec'),env)
  with mock.patch.object(census,'_row_select',side_effect=lambda mask,old,new:new),mock.patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):expected=env['advance'](before,np.zeros((b,2),np.int32),active,scored,'SAME_LITERAL_CHANCE')
  with mock.patch.object(census,'_row_select',side_effect=lambda mask,old,new:new),mock.patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):actual=after.advance(np.zeros((b,2),np.int32),active,scored,'SAME_LITERAL_CHANCE')
  stripped=[{k:v for k,v in p.items()if k not in('consumer_diagnostics','consumer_failures')}for p in after.last_packet]
  self.assertEqual(stripped,before.last_packet);self.assertEqual(old_calls,new_calls)
  for a,z in zip(expected[:2],actual[:2]):np.testing.assert_array_equal(a,z)
  self.assertEqual(expected[2],actual[2]);self.assertTrue(all(v.copies<=1 and v.slices==0 for v in tracked))
  for row in np.flatnonzero(active):
   d=after.last_packet[row]['consumer_diagnostics']
   self.assertEqual(d['available'],bool(scored[row]))
   if not scored[row]:self.assertEqual(d['reason'],'DIAGNOSTIC_LOCAL_NO_PUBLIC_CONSUMPTION')

 def test_original_decisions_exact_ast_after_only_diagnostic_data_removed(self):
  class Remove(ast.NodeTransformer):
   def visit_Assign(self,n):
    if any(isinstance(t,ast.Name)and t.id in('fold_diagnostics','consumption_diagnostics','owner_checks','registry_equal')for t in n.targets):return None
    return self.generic_visit(n)
   def visit_For(self,n):
    if any(isinstance(x,ast.AugAssign)and isinstance(x.target,ast.Name)and x.target.id=='registry_equal'for x in n.body):return None
    return self.generic_visit(n)
   def visit_Expr(self,n):
    if any(isinstance(x,ast.Name)and x.id=='owner_checks'for x in ast.walk(n)):return None
    return self.generic_visit(n)
   def visit_Call(self,n):
    n=self.generic_visit(n);n.keywords=[k for k in n.keywords if k.arg not in('fold_diagnostics','consumption_diagnostics')]
    if isinstance(n.func,ast.Name)and n.func.id=='Report':n.args=[a for a in n.args if not(isinstance(a,ast.Call)and isinstance(a.func,ast.Name)and a.func.id=='ConsumerDiagnostics')]
    return n
  for module,name in((full,'project_and_fold'),(ex,'consume_frame_device'),(witness,'consume_pair_device')):
   path=Path(module.__file__);old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/'+path.name],text=True));new=ast.parse(path.read_text())
   get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
   normalized=Remove().visit(get(new));self.assertEqual(ast.dump(get(old),include_attributes=False),ast.dump(normalized,include_attributes=False),name)
   # An actual admission/refusal edit remains detectable after normalization.
   import copy
   mutant=copy.deepcopy(normalized);target={'project_and_fold':'ok','consume_frame_device':'good','consume_pair_device':'complete'}[name]
   changed=0
   for node in ast.walk(mutant):
    if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id==target for t in node.targets):node.value=ast.Constant(True);changed+=1
   self.assertGreater(changed,0);self.assertNotEqual(ast.dump(get(old),include_attributes=False),ast.dump(mutant,include_attributes=False))
  # Physical assignments, chance, driver, sampler and generic join unchanged.
  for name in('transition.py','state.py','engine_retained_encoder_driver.py','engine_kernel_frame_risk.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]))
  text=Path(witness.__file__).read_text();pins=ast.literal_eval(next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE_PINS'for t in n.targets)).value)
  import hashlib
  for name,digest in pins.items():self.assertEqual(hashlib.sha256(Path('gpu_engine',name).read_bytes()).hexdigest(),digest,name)

if __name__=='__main__':unittest.main()
