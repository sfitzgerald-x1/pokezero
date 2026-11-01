"""Same-transaction diagnostic context only; transition is a literal fake."""
import ast,hashlib,subprocess,unittest
from pathlib import Path
from typing import NamedTuple
from types import SimpleNamespace as S
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import benchmark_capture_gap_census as census,engine_lifecycle_callbacks as life
from gpu_engine import engine_request_order as order,engine_execution_completeness as ex
from gpu_engine import engine_kernel_frame_risk as risk,engine_source_frame_witness as witness,transition
BASE='ab04b1e48396f7424490fb6086d50e3801bdfaa2'
class Engine(NamedTuple):
 done:object
 value:object
class CensusPartyOrderTests(unittest.TestCase):
 def inputs(self):
  b=8;r=j.arange(b);z=j.zeros((b,2),j.int32);blank=z-1;yes=j.ones(b,bool);stream=r+910001;binding=j.arange(8,dtype=j.uint32);mapping=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0)
  c=life.Cursor(stream,j.broadcast_to(binding,(b,8)),z,j.ones(b,j.int32),j.zeros(b,j.int32),yes,yes,z,z,j.ones((b,2),j.int32),j.ones((b,2),bool),z+1,mapping,j.ones((b,2),j.int32))
  orders=tuple(order.initialize_order(stream,j.full(b,s,j.int32),j.full(b,6,j.int32),engine_binding=c.engine_binding)for s in(0,1));base=S(engine_binding=binding,source_binding=binding,move_binding=binding,producer_binding=binding)
  rt=object.__new__(census._Runtime);rt.static={'public':S(device=S(causal=S(base=base))),'execution_compiled':S(device=object()),'kernel_risk_compiled':S(device=object()),'frame_witness_compiled':S(device=object()),'speed_profile':None}
  for key in ('leftovers_context','status_residual_context','direct_hit_context','substitute_context','toxic_context','recoil_context','calm_mind_context','swords_dance_context','spikes_context','thunder_wave_context','sleep_powder_context'):rt.static[key]=None
  rt.empty_ledger=object();rt.empty_risk=object();rt.table=object();return rt,c,orders,Engine(~yes,j.zeros(b,j.int32))
 def run_fake(self,party,active):
  rt,c,initial,engine=self.inputs();observed=[]
  def begin(tape,*args,**kw):return tape._replace(execution=ex.Context(ledger=object(),tables=object()))
  def step(masked,actions,table,chance,**kw):
   t=kw['_public_trace'];observed.append(t.execution.party_orders);b=len(active);r=j.arange(b)
   # A literal actual-installed receipt is provided by this fake callback,
   # including inactive row0 to prove row commit cannot rewrite its carry.
   gen=c.generation.at[:,0].set(1);phys=c.physical_active.at[:,0].set(3);slots=c.public_slot.at[:,0].set(1);mp=c.physical_to_public.at[:,0,3].set(1)
   after=c._replace(generation=gen,physical_active=phys,public_slot=slots,encounter=slots+1,physical_to_public=mp,next_slot=c.next_slot.at[:,0].set(2))
   words=t.words.at[:,0].set(j.array([6,0,0,0,3,0,0,0,0,0],j.int32));t=t._replace(words=words,count=j.ones(b,j.int32),cursor=after,event_turn=t.event_turn.at[:,0].set(1),event_generation=t.event_generation.at[:,0].set(c.generation))
   return S(step=S(state=engine._replace(value=j.ones(b,j.int32)),invalid_actions=j.zeros(b,bool),placeholder_moves=j.zeros(b,j.int32)),callbacks=t,weather_profile=None)
  with patch.object(ex,'begin_on_tape',side_effect=begin),patch.object(risk,'begin_frame_device',return_value=None),patch.object(transition,'step_with_chance',side_effect=step)as called,patch.object(witness,'_raw_complete_words',return_value=j.ones((8,64),bool)),patch.object(ex,'_validate',return_value=j.ones(8,bool)),patch.object(ex,'_tape',return_value=j.ones(8,bool)):
   out=rt._diagnostic_step(engine,c,None,j.zeros((8,2),j.int32),object(),active,party)
  self.assertEqual(called.call_count,1);return c,initial,observed,out
 def test_same_orders_transported_and_actual_receipts_only_with_inactive_rows(self):
  _,_,orders,_=self.inputs();active=j.ones(8,bool).at[0].set(False);c,initial,seen,out=self.run_fake(orders,active);self.assertIs(seen[0],orders)
  got=out[-1];self.assertEqual(got[0].request_to_physical[0].tolist(),[0,1,2,3,4,5]);self.assertEqual(got[0].request_to_physical[1].tolist(),[3,1,2,0,4,5]);self.assertEqual(got[0].generation.tolist(),[0]+[1]*7)
  for a,b in zip(jax.tree.leaves(got[1]),jax.tree.leaves(orders[1]),strict=True):np.testing.assert_array_equal(a,b)
  self.assertEqual(out[0].value.tolist(),[0]+[1]*7)
 def test_missing_invalid_and_constructor_session_lifetime_are_not_rebuilt(self):
  _,_,orders,_=self.inputs();_,_,seen,out=self.run_fake(None,j.ones(8,bool));self.assertIsNone(seen[0]);self.assertIsNone(out[-1])
  invalid=(orders[0]._replace(admitted=j.zeros(8,bool)),orders[1]);_,_,_,out=self.run_fake(invalid,j.ones(8,bool));self.assertFalse(bool(out[-1][0].admitted.any()))
  # Actual owning-source and fallback update expressions use the current row
  # facts, not stale Session orders on post-gap rows or copied fresh defaults.
  tree=ast.parse(Path(census.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_Runtime');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__');advance=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='advance')
  assigns=[n for n in ast.walk(init)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)and t.attr=='party_orders'for t in n.targets)];self.assertEqual(len(assigns),1);self.assertEqual(ast.unparse(assigns[0].value),'self.session.state.core.orders')
  assignment=next(n for n in ast.walk(advance)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)and t.attr=='party_orders'for t in n.targets));self.assertEqual(ast.unparse(assignment.value),'_row_select(kept, self.party_orders, result.state.core.orders)')
  for name in ('engine_retained_encoder_driver.py','engine_heal_bell_callbacks.py','engine_request_order.py','transition.py','state.py'):
   self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show',BASE+':gpu_engine/'+name]))
  # Later reviewed observers update their owned pins. Keep the CURRENT
  # witness body exact and validate every CURRENT pin against its actual file.
  old=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/engine_source_frame_witness.py'],text=True));new=ast.parse(Path('gpu_engine/engine_source_frame_witness.py').read_text())
  is_pins=lambda n:isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SOURCE_PINS'for t in n.targets)
  pins=ast.literal_eval(next(n.value for n in new.body if is_pins(n)))
  self.assertEqual(len(pins),85)
  for name,want in pins.items():self.assertEqual(hashlib.sha256(Path('gpu_engine',name).read_bytes()).hexdigest(),want,name)
  strip=lambda t:ast.Module(body=[n for n in t.body if not is_pins(n)],type_ignores=[])
  self.assertEqual(ast.dump(strip(old),include_attributes=False),ast.dump(strip(new),include_attributes=False))
 def test_current_parent_packets_choices_chance_and_timing_calls_equal(self):
  from test_gpu_engine_census_host_copy import HostCopyTests
  from unittest import mock
  tree=ast.parse(subprocess.check_output(['git','show',BASE+':gpu_engine/benchmark_capture_gap_census.py'],text=True));cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_Runtime');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef)and n.name=='advance');fn.name='parent_advance';env=dict(vars(census),_row_select=lambda mask,a,b:b);exec(compile(ast.fix_missing_locations(ast.Module([fn],[])),'frozen-parent','exec'),env)
  def backend(batch,missing):
   value,tracked,calls,active,scored=HostCopyTests().make_backend(batch,missing);value.party_orders=None;value.session.state.core.orders=None;advance=value.driver.advance;execute=value.diag_executable
   def adv(*a,**kw):
    got=advance(*a,**kw);result=got.result if hasattr(got,'diagnostics')else got;result.state.core.orders=None;return got
   def diag(*args):return execute(*args[:6])+((None,)if len(args)==7 else())
   value.driver.advance=adv;value.diag_executable=diag;return value,calls,active,scored
  for size in(8,16):
   for missing in(False,True):
    before,oldcalls,active,scored=backend(size,missing);after,calls,_,_=backend(size,missing)
    with mock.patch.object(census,'_row_select',side_effect=lambda mask,a,b:b),mock.patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):old=env['parent_advance'](before,np.zeros((size,2),np.int32),active,scored,'SAME_LITERAL_CHANCE')
    with mock.patch.object(census,'_row_select',side_effect=lambda mask,a,b:b),mock.patch.object(census.time,'perf_counter_ns',side_effect=[1000000,2000000]):new=after.advance(np.zeros((size,2),np.int32),active,scored,'SAME_LITERAL_CHANCE')
    self.assertEqual(before.last_packet,after.last_packet);self.assertEqual(calls,oldcalls);self.assertEqual(old[2],new[2])
    for a,b in zip(old[:2],new[:2]):np.testing.assert_array_equal(a,b)
if __name__=='__main__':unittest.main()
