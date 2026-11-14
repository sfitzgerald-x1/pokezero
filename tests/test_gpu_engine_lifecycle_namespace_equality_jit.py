"""Current namespace equality code reuse only; no engine/FULL/model/game."""
import ast,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_lifecycle_event_device as cb
from gpu_engine import engine_source_frame_witness as witness
BASE='e481228626c9cfdfe2a2989871672392aa3e5859'

class NamespaceEqualityTests(unittest.TestCase):
 def pairs(self):
  return tuple((j.arange(8,dtype=j.uint32)+i,j.arange(8,dtype=j.uint32)+i)for i in range(6))
 def old_check(self,pairs):
  for a,b in pairs:
   if not np.array_equal(np.asarray(a),np.asarray(b)):raise ValueError('lifecycle source/public namespace drift')
 def outcome(self,fn,pairs):
  try:fn(pairs);return None
  except Exception as e:return type(e),str(e)
 def test_current_both_sides_and_executable_reuse(self):
  pairs=self.pairs();executable=cb._NAMESPACE_EQUAL_DEVICE
  self.assertIsNone(self.outcome(cb._check_namespace_pairs,pairs))
  for index in range(6):
   for side in (0,1):
    current=list(pairs);p=list(current[index]);p[side]=p[side].at[3].add(j.uint32(1));current[index]=tuple(p);current=tuple(current)
    self.assertEqual(self.outcome(cb._check_namespace_pairs,current),self.outcome(self.old_check,current))
    # Replacing BOTH holders with fresh equal values remains accepted; there
    # is no closed-over expected-array value or cached predicate result.
    current=list(current);current[index]=(p[side],p[side]);self.assertIsNone(self.outcome(cb._check_namespace_pairs,tuple(current)))
  self.assertIs(cb._NAMESPACE_EQUAL_DEVICE,executable)
  fresh=tuple((a+40,b+40)for a,b in pairs);self.assertIsNone(self.outcome(cb._check_namespace_pairs,fresh))
 def test_fallback_origin_errors_deleted_and_unsupported_signatures(self):
  pairs=self.pairs();bad=list(pairs);bad[0]=(bad[0][0]+1,bad[0][1]);bad=tuple(bad)
  # Stale/forged cached code must never execute. Its original eager CURRENT
  # semantics continue to decide valid and invalid inputs.
  called=[]
  with patch.object(cb,'_NAMESPACE_EQUAL_DEVICE',lambda *a:called.append(a)or j.ones(6,bool)):
   self.assertIsNone(self.outcome(cb._check_namespace_pairs,pairs));self.assertEqual(self.outcome(cb._check_namespace_pairs,bad),self.outcome(self.old_check,bad))
  self.assertEqual(called,[])
  with patch.object(cb,'_namespace_values_equal',lambda *a:called.append(a)or j.ones(6,bool)):
   self.assertEqual(self.outcome(cb._check_namespace_pairs,bad),self.outcome(self.old_check,bad))
  self.assertEqual(called,[])
  # Compile/service failure also retains original current eager behavior.
  def fail(*a):raise RuntimeError('unavailable executable')
  with patch.object(cb,'_NAMESPACE_EQUAL_DEVICE',fail),patch.object(cb,'_NAMESPACE_EQUAL_ORIGIN',(cb._namespace_values_equal,fail)):
   self.assertIsNone(self.outcome(cb._check_namespace_pairs,pairs));self.assertEqual(self.outcome(cb._check_namespace_pairs,bad),self.outcome(self.old_check,bad))
  for dtype in (j.int32,j.float32):
   changed=tuple((a.astype(dtype),b.astype(dtype))for a,b in pairs);self.assertEqual(self.outcome(cb._check_namespace_pairs,changed),self.outcome(self.old_check,changed))
  class Explodes:
   def __array__(self,*a,**k):raise RuntimeError('later value must not mask first namespace failure')
  first=((np.zeros(8,np.uint32),np.ones(8,np.uint32)),(Explodes(),np.zeros(8,np.uint32)))
  self.assertEqual(self.outcome(cb._check_namespace_pairs,first),self.outcome(self.old_check,first))
  deleted=j.ones(8,j.uint32);deleted.delete();current=list(pairs);current[1]=(deleted,pairs[1][1]);current=tuple(current)
  self.assertEqual(self.outcome(cb._check_namespace_pairs,current),self.outcome(self.old_check,current))
  current=list(current);current[0]=bad[0];current=tuple(current)
  self.assertEqual(self.outcome(cb._check_namespace_pairs,current),self.outcome(self.old_check,current))
 def test_exact_guard_function_overlay_source_and_physical_scope(self):
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_lifecycle_event_device.py'],text=True));new=ast.parse(Path(cb.__file__).read_text())
  oldf=next(x for x in old.body if isinstance(x,ast.FunctionDef)and x.name=='_host_inputs');newf=next(x for x in new.body if isinstance(x,ast.FunctionDef)and x.name=='_host_inputs')
  # Every original structural/base/context guard node remains exact and ordered.
  self.assertEqual(ast.dump(ast.Module(body=oldf.body[:-1],type_ignores=[])),ast.dump(ast.Module(body=newf.body[:-1],type_ignores=[])))
  self.assertEqual(ast.dump(oldf.body[-1].iter),ast.dump(newf.body[-1].value.args[0]))
  # Preserve the original reversal on its immutable feature source, and
  # separately bind every current integrated guard and helper statement.
  integrated=ast.parse(subprocess.check_output(['git','show','b8092a85:gpu_engine/engine_lifecycle_event_device.py'],text=True))
  self.assertEqual(ast.dump(new),ast.dump(integrated))
  current=ast.parse(subprocess.check_output(['git','show','eadf3359:gpu_engine/engine_lifecycle_event_device.py'],text=True));helpers={'_namespace_values_equal','_check_namespace_pairs'};bindings={'_NAMESPACE_EQUAL_DEVICE','_NAMESPACE_EQUAL_ORIGIN'}
  current.body=[n for n in current.body if not(isinstance(n,ast.FunctionDef)and n.name in helpers)and not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in bindings for t in n.targets))]
  current.body=[oldf if isinstance(n,ast.FunctionDef)and n.name=='_host_inputs'else n for n in current.body]
  self.assertEqual(ast.dump(current),ast.dump(old))
  # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
  for name in ('engine_retained_encoder_driver.py','engine_kernel_frame_risk.py','engine_execution_completeness.py','engine_full_v4_event_device.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+name]))
  witness._source()

class GenuineLifecycleGuardTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import test_gpu_engine_hidden_power_substitute_input as fixture
  fixture.HiddenPowerSubstituteInputTests.setUpClass();cls.f=fixture.HiddenPowerSubstituteInputTests
 def test_original_current_host_guard_and_unchanged_errors(self):
  f=self.f;lower=f.runtime['public'].causal
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_lifecycle_event_device.py'],text=True));node=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='_host_inputs');ns=dict(cb.__dict__);exec(compile(ast.Module(body=[node],type_ignores=[]),'<original-current-host-inputs>','exec'),ns);original=ns['_host_inputs']
  args=(lower.projector,lower.species_catalog,lower.source,lower.context_tables)
  self.assertIsNone(original(*args));self.assertIsNone(cb._host_inputs(*args));self.assertIs(cb.check_projector(lower),lower.device)
  # Exercise the ACTUAL guard argument tuple, not only the pure equality
  # helper: fresh equal holders keep old accepted semantics, current poisoned
  # values still fail the original error before later consumption.
  from dataclasses import replace
  for delta in (0,1):
   source=replace(lower.source,device=lower.source.device._replace(binding=lower.source.device.binding+j.uint32(delta)))
   current=(*args[:2],source,args[3]);results=[]
   for fn in (original,cb._host_inputs):
    try:fn(*current);results.append(None)
    except Exception as e:results.append((type(e),str(e)))
   self.assertEqual(results[0],results[1])
   if delta:self.assertEqual(results[0],(ValueError,'lifecycle source/public namespace drift'))
   else:self.assertIsNone(results[0])
  # The real checked API keeps its original outside-JIT rejection.
  def outcome(fn):
   try:jax.jit(lambda:fn(*args))();return None
   except Exception as e:return type(e),str(e)
  self.assertEqual(outcome(cb._host_inputs),outcome(original))
  # Existing earlier context mode failures cannot reach the new comparator.
  bad=lower.context_tables._replace(refresh=j.array(1,j.int32));badargs=(*args[:3],bad)
  for fn in (original,cb._host_inputs):
   with self.assertRaisesRegex(ValueError,'enhanced raw semantic/closure producers remain unknown'):fn(*badargs)
