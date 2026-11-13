"""Lean genuine-table JAX comparison regression; no FULL or engine turn."""
import ast,copy,json,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_thunder_wave_callbacks as cb,engine_kernel_frame_risk as risk
from gpu_engine import engine_execution_completeness as ex,engine_lifecycle_callbacks as life
from gpu_engine import engine_source_frame_witness as witness
import test_gpu_engine_thunder_wave_substitute_failure as fixture
BASE='ca3e39d2cdcb9947a334af33ed2d9868ae426db0'

def probe(source,guards=True):
 t=ast.parse(source);fn=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='consume_pair_device')
 loop=next(n for n in fn.body if isinstance(n,ast.For)and 'tape.execution.risks.tables' in ast.unparse(n.iter))
 before=[]
 if guards:
  start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='packed'for v in n.targets))
  stop=next(i for i,n in enumerate(fn.body)if n is loop);before=[n for n in fn.body[start:stop]if not isinstance(n,ast.For)]
 wrapper=ast.parse('def compare(actual,expected):\n tape=NS(execution=NS(risks=NS(tables=actual)))\n tables=NS(risk=expected)\n valid=jnp.ones(8,bool)\n return valid\n').body[0]
 wrapper.body[-1:-1]=copy.deepcopy(before)+[copy.deepcopy(loop)]
 ns=dict(NS=NS,jnp=j,execution=ex);exec(compile(ast.fix_missing_locations(ast.Module([wrapper],[])),'<literal consume_pair Risk comparison>','exec'),ns);return ns['compare'],loop

class FlatGroundRiskTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixture.ThunderWaveSubstituteTests.setUpClass.__func__(cls)
 def tape(self):
  r=j.arange(8,dtype=j.int32);cursor=life.register(r+331001,catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
  t=life._tape(cursor,self.producer,64);a=self.runtime['execution_compiled'];rt=self.runtime['kernel_risk_compiled'].device
  t=ex.begin_on_tape(t,ex.initialize(cursor.stream_id,compiled=a),a.device,active=j.ones(8,bool));carry=risk.begin_frame_device(risk.initialize_device(t.stream_id,rt),t,rt,active=j.ones(8,bool))
  return t._replace(execution=t.execution._replace(risks=risk.Context(carry,rt)))
 def test_real_compiler_exact_flat_values_and_strict_current_ground_join(self):
  g=self.thunder.device.ground_immunity;rt=self.runtime['kernel_risk_compiled'].device;packed=rt.thunder_wave_ground_immunity
  expected=np.concatenate(([int(g.electric),int(g.ground),len(g.abilities),len(g.items)],np.asarray(g.abilities,np.int32),np.asarray(g.items,np.int32))).astype(np.int32)
  np.testing.assert_array_equal(packed,expected);self.assertEqual(packed.shape,expected.shape);self.assertEqual(packed.dtype,j.int32)
  self.assertTrue(all(isinstance(v,jax.Array)for v in rt));risk.check_compiled(self.runtime['kernel_risk_compiled'])
  raw=json.loads(risk._COMPILED[self.runtime['kernel_risk_compiled']][4]);self.assertEqual(raw['thunder_wave_ground_immunity'],expected.tolist())
  t=self.tape();context=self.runtime['thunder_wave_context'];self.assertEqual(cb._ground_source(t,context).tolist(),[True]*8)
  for i in (0,1,2,3,4,len(packed)-1):
   bad=packed.at[i].add(1);current=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=rt._replace(thunder_wave_ground_immunity=bad))))
   self.assertEqual(cb._ground_source(current,context).tolist(),[False]*8)
  for bad in (packed[None,:],packed.astype(j.uint32),packed[:-1],None):
   current=t._replace(execution=t.execution._replace(risks=t.execution.risks._replace(tables=rt._replace(thunder_wave_ground_immunity=bad))))
   with self.assertRaises(ValueError):cb._ground_source(current,context)
  self.assertEqual(cb.pack_ground(None).shape,(0,));self.assertEqual(cb.pack_ground(None).dtype,j.int32)
  legacy=risk.compile_risks(self.producer,self.runtime['execution_compiled'],thunder_wave_compiled=self.legacy)
  risk.check_compiled(legacy);self.assertEqual(legacy.device.thunder_wave_ground_immunity.shape,(0,))
 def test_original_consume_pair_loop_jit_preserves_every_leaf_and_reproduces_old_error(self):
  old_source=subprocess.check_output(['git','show',BASE+':gpu_engine/engine_source_frame_witness.py'],text=True);current=Path(witness.__file__).read_text()
  old,old_loop=probe(old_source,False);check,loop=probe(current);self.assertEqual(ast.dump(old_loop),ast.dump(loop))
  rt=self.runtime['kernel_risk_compiled'].device;g=self.thunder.device.ground_immunity
  nested=rt._replace(thunder_wave_ground_immunity=g)
  with self.assertRaises(jax.errors.TracerBoolConversionError):jax.make_jaxpr(old)(nested,nested)
  fast=jax.jit(check);jax.make_jaxpr(check)(rt,rt);self.assertEqual(fast(rt,rt).tolist(),[True]*8)
  packed=rt.thunder_wave_ground_immunity;changed=rt._replace(thunder_wave_ground_immunity=packed.at[-1].add(1));self.assertEqual(fast(changed,rt).tolist(),[False]*8)
  self.assertEqual(fast(rt,rt).tolist(),[True]*8)
  for bad in (packed[None,:],packed.astype(j.uint32),packed[:-1],g):
   with self.assertRaises(ValueError):jax.make_jaxpr(check)(rt._replace(thunder_wave_ground_immunity=bad),rt)
 def test_empty_legacy_is_not_a_capability_and_unrelated_source_is_exact(self):
  t=ast.parse(Path(witness.__file__).read_text());node=next(n for n in ast.walk(t)if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='thunder_ticket'for v in n.targets))
  code=compile(ast.Module([node],[]),'<actual thunder_ticket capability>','exec');z=j.zeros((8,1),j.int32);ledger=NS(site=z+int(ex.Site.PRIMARY_PARALYSIS),stage=z+2,no_output_kind=z,end=z+1,start=z)
  for packed,want in ((cb.pack_ground(None),False),(None,False),(cb.pack_ground(self.thunder.device.ground_immunity),True)):
   ns=dict(entries=j.ones((8,1),bool),ledger=ledger,execution=ex,tables=NS(risk=NS(thunder_wave_substitute_failure=j.array(False),thunder_wave_ground_immunity=packed)));exec(code,ns);self.assertEqual(ns['thunder_ticket'].tolist(),[[want]]*8)
  for file in ('transition.py','engine_retained_encoder_driver.py','engine_lifecycle_event_device.py','engine_full_v4_event_device.py','engine_execution_completeness.py'):
   historical=subprocess.check_output(['git','show','62ccffa7:gpu_engine/'+file])
   self.assertEqual(historical,subprocess.check_output(['git','show',BASE+':gpu_engine/'+file]))
   # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components.
   if file!='transition.py':self.assertEqual(Path('gpu_engine',file).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+file]))
