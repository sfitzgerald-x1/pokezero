"""Fake device conversion counts and exact stored-data equivalence only."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import unittest
from unittest import mock
import numpy as np
from gpu_engine import benchmark_capture_gap_census as c,engine_outcome_diagnostics as d

BASE='ac13fdbd'
class Device:
 def __init__(self,value,root=None):
  self.value=np.asarray(value);self.shape=self.value.shape;self.dtype=self.value.dtype
  self.root=self if root is None else root
  if root is None:self.copies=0;self.slices=0
 def __array__(self,dtype=None,copy=None):self.root.copies+=1;return np.asarray(self.value,dtype=dtype)
 def __getitem__(self,index):self.root.slices+=1;return Device(self.value[index],self.root)

def original(path,name,env,cls=None):
 tree=ast.parse(subprocess.check_output(['git','show',BASE+':'+path],text=True))
 body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls).body
 node=next(n for n in body if isinstance(n,ast.FunctionDef)and n.name==name);node.name='original_'+name
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'immutable-original','exec'),env)
 return env[node.name]

def capture(b,wrap):
 words=np.arange(b*2*len(d.FIELDS),dtype=np.int32).reshape(b,2,len(d.FIELDS));valid=np.zeros((b,2),bool);valid[:,0]=True
 residual=np.arange(b*2*len(d.RESIDUAL_FIELDS),dtype=np.int32).reshape(b,2,len(d.RESIDUAL_FIELDS))
 overflow=np.arange(b)%3==0
 return d.Capture(*(wrap(x)for x in(words,valid,np.ones(b,np.int32),overflow,residual,valid.copy(),np.ones(b,np.int32),~overflow)))

class HostCopyTests(unittest.TestCase):
 def test_refused_serializer_exact_old_new_missing_malformed_overflow_and_copy_bound(self):
  old=original('gpu_engine/engine_outcome_diagnostics.py','refused_rows',dict(vars(d)))
  for b in(2,8,16):
   for mode in('normal','legacy','missing','malformed','empty'):
    with self.subTest(b=b,mode=mode):
     tracked=[]
     def wrap(x):v=Device(x);tracked.append(v);return v
     value=capture(b,wrap)
     if mode=='legacy':value=d.Capture(*value[:4])
     if mode=='missing':value=None
     if mode=='malformed':value=value._replace(words=wrap(np.zeros((b,2,len(d.FIELDS)),np.float32)))
     refused=np.arange(b)%2==1
     if mode=='empty':refused[:]=False
     expected=old(value,refused)
     for v in tracked:v.copies=v.slices=0
     actual=d.refused_rows(value,refused)
     self.assertEqual(actual,expected)
     self.assertTrue(all(v.copies<=1 and v.slices==0 for v in tracked))
     if mode=='empty':self.assertTrue(all(v.copies==0 for v in tracked))
     self.assertEqual([row['row']for row in actual['rows']],np.flatnonzero(refused).tolist())
 def test_only_data_serializer_and_owned_pin_change_outside_census(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_outcome_diagnostics.py'],text=True));new=ast.parse(Path(d.__file__).read_text())
  keep=lambda tree:ast.Module(body=[n for n in tree.body if not isinstance(n,ast.FunctionDef)or n.name!='refused_rows'],type_ignores=[])
  self.assertEqual(ast.dump(keep(old),include_attributes=False),ast.dump(keep(new),include_attributes=False))
  for path in('transition.py','engine_kernel_frame_risk.py'):
   self.assertEqual(Path('gpu_engine',path).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+path]))
  # Built-in timing's guarded driver delta is separately exact-AST checked
  # against c028 by test_gpu_engine_builtin_host_timing.
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_source_frame_witness.py'],text=True));new=ast.parse(Path('gpu_engine/engine_source_frame_witness.py').read_text())
  keep=lambda tree:ast.Module(body=[n for n in tree.body if not isinstance(n,ast.Assign)or not any(isinstance(t,ast.Name)and t.id=='SOURCE_PINS'for t in n.targets)],type_ignores=[])
  self.assertEqual(ast.dump(keep(old),include_attributes=False),ast.dump(keep(new),include_attributes=False))
 def make_backend(self,b,missing):
  tracked=[]
  def wrap(x):v=Device(x);tracked.append(v);return v
  zero=wrap(np.zeros(b,np.int32));yes=wrap(np.ones(b,bool))
  def ledger(site):return NS(count=wrap(np.ones(b,np.int32)),site=wrap(np.full((b,2),site,np.int32)),stage=wrap(np.where(np.arange(b)[:,None]%3==0,1,2).repeat(2,axis=1).astype(np.int32)),refusal=zero)
  words=np.zeros((b,4,10),np.int32);words[:,0,0]=35;words[:,1,0]=1;words[:,1,5]=2;words[:,2,0]=20
  count=wrap(np.full(b,3,np.int32));raw=wrap(words);good=wrap(np.ones((b,4),bool));risk=NS(risk=zero)
  original_capture=capture(b,wrap);diag_capture=capture(b,wrap)
  if missing:original_capture=None
  witness=NS(raw_words=raw,raw_count=count,raw_complete_words=None if missing else good,refusal=zero,outcome_diagnostics=original_capture)
  current=NS(engine=NS(fake='before'),cursor=NS(admitted=yes),weather_profile=None,frame_number=wrap(np.zeros(b,np.int32)),execution=ledger(10),risks=risk,frame_input_report=witness)
  after=NS(**vars(current));active=np.arange(b)%4!=0;scored=np.arange(b)%2==0;kept=active&scored
  after.frame_number=wrap(kept.astype(np.int32));after.engine=NS(fake='after')
  callbacks=NS(words=raw,count=count,refusal=zero,execution=NS(ledger=ledger(5),risks=NS(carry=risk),leech_history=None,diagnostics=diag_capture))
  if not missing:witness.outcome_diagnostics=original_capture._replace(count=current.execution.count)
  callbacks.execution.diagnostics=diag_capture._replace(count=callbacks.execution.ledger.count)
  backend=c._Runtime.__new__(c._Runtime);backend.j=NS(asarray=lambda x:x);backend.jax=NS(block_until_ready=lambda x:x)
  backend.engine=current.engine;backend.cursor=current.cursor;backend.profile=None;backend.move_names=('none','one','two');backend.session=NS(state=NS(core=current),result=None);calls=[]
  def advance(session,actions,**kw):
   calls.append(('original',actions.tolist(),kw['active_rows'].tolist(),kw['chance_inputs']));session.state=NS(core=after);session.result=NS(state=session.state)
   if kw.get('return_diagnostics',False):
    from gpu_engine.engine_retained_encoder_driver import AdvanceWithDiagnostics
    return AdvanceWithDiagnostics(session.result,diag)
   return session.result
  diag=NS(executed_rows=wrap(kept),active_rows=wrap(kept.copy()),invalid_actions=wrap(np.zeros(b,bool)),host_timings={'advance_wall_ms':1.25})
  backend.driver=NS(advance=advance,latest_diagnostics=lambda session:diag)
  bad=wrap(np.arange(b)%5==0)
  def executable(*args):calls.append(('fallback',args[3].tolist(),args[5].tolist(),args[4]));return NS(fake='combined'),NS(admitted=yes),None,callbacks,bad,None,None if missing else good,yes
  backend.diag_executable=executable
  return backend,tracked,calls,active,scored
 def test_mixed_advance_exact_packets_and_no_per_row_device_conversions(self):
  choose=lambda mask,prior,new:new
  old=original('gpu_engine/benchmark_capture_gap_census.py','advance',dict(vars(c),_row_select=choose),cls='_Runtime')
  for b in(8,16):
   for missing in(False,True):
    with self.subTest(b=b,missing=missing):
     before,old_arrays,old_calls,active,scored=self.make_backend(b,missing);after,new_arrays,new_calls,_,_=self.make_backend(b,missing)
     with mock.patch.object(c,'_row_select',side_effect=choose),mock.patch.object(c.time,'perf_counter_ns',side_effect=[1000000,2000000]):
      expected=old(before,np.zeros((b,2),np.int32),active,scored,'literal-same-chance')
     with mock.patch.object(c,'_row_select',side_effect=choose),mock.patch.object(c.time,'perf_counter_ns',side_effect=[1000000,2000000]):
      actual=after.advance(np.zeros((b,2),np.int32),active,scored,'literal-same-chance')
     self.assertEqual(before.last_packet,after.last_packet);self.assertEqual(old_calls,new_calls)
     for a,z in zip(actual[:2],expected[:2]):np.testing.assert_array_equal(a,z)
     self.assertEqual(actual[2],expected[2]);self.assertTrue(any(v.copies>1 for v in old_arrays))
     self.assertTrue(all(v.copies<=1 and v.slices==0 for v in new_arrays),[(v.shape,v.copies,v.slices)for v in new_arrays if v.copies>1 or v.slices])
     self.assertEqual([bool(p)for p in after.last_packet],active.tolist())

if __name__=='__main__':unittest.main()
