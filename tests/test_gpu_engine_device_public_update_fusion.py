"""Synthetic exact-update controls; no engine execution/model/game/timing."""
import ast
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h,engine_device_trackers as tr,transition
from gpu_engine.engine_device_member_updates import set_member
from tests.test_gpu_engine_device_public_batch import fixture

BASE='03be68de16f4265f5f9c226ce6ada80b8dd8619d'


def exact(test,a,b):
 test.assertEqual(jax.tree.structure(a),jax.tree.structure(b))
 for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
  x,y=np.asarray(x),np.asarray(y);test.assertEqual(x.dtype,y.dtype);test.assertEqual(x.shape,y.shape)
  if x.dtype==np.float32:x=x.view(np.uint32);y=y.view(np.uint32)
  np.testing.assert_array_equal(x,y)


class DevicePublicUpdateFusionTests(unittest.TestCase):
 def test_single_member_update_exact_dtypes_bits_mixed_indices_and_no_scatter(self):
  b=8;r=j.arange(b);sides=j.array([0,1,-1,-2,2,-3,1,0],j.int32)
  slots=j.array([0,5,-1,-6,0,0,6,-7],j.int32)
  for dtype,tail in ((j.int32,()),(j.uint32,()),(j.float32,()),(j.bool_,()),(j.int32,(7,))):
   shape=(b,2,6)+tail;n=np.prod(shape)
   array=j.arange(n,dtype=j.int32).reshape(shape).astype(dtype)
   values=j.full((b,)+tail,16777217 if dtype!=j.bool_ else True,dtype)
   if dtype==j.float32:
    array=jax.lax.bitcast_convert_type(j.arange(n,dtype=j.uint32).reshape(shape)+j.uint32(0x7fc00000),j.float32)
    values=jax.lax.bitcast_convert_type(j.array([0x80000000,0x7fc12345]*4,j.uint32),j.float32)
   old=jax.jit(lambda a,s,p,v:a.at[r,s,p].set(v));new=jax.jit(set_member)
   exact(self,old(array,sides,slots,values),new(array,sides,slots,values))
   exact(self,old(array,1,slots,values),new(array,1,slots,values))
  old_ir=str(old.lower(array,sides,slots,values).compiler_ir());new_ir=str(new.lower(array,sides,slots,values).compiler_ir())
  self.assertIn('stablehlo.scatter',old_ir);self.assertNotIn('stablehlo.scatter',new_ir)
  with self.assertRaises(ValueError):set_member(j.zeros((3,2,6),j.int32),0,j.zeros(3,j.int32),j.zeros(3,j.float32))

 def test_mixed_hook_updates_and_both_readouts_match_default_exactly(self):
  state,_,public=fixture(3);r=j.arange(3);side=j.array([0,1,0],j.int32);use=j.array([True,False,True])
  # Include integers outside float32's exact range and signed-zero payloads.
  public=public._replace(mon_features={**public.mon_features,'turns_active':public.mon_features['turns_active']+j.int32(16777217)})
  def work(t,s,*,dense):
   c=h.Context(t,dense_updates=dense);p=s.active[r,side]
   for name in ('turns_active','turns_active_total','moves_shown'):
    h.feature(c,side,p,name,j.array([1,2,3],j.int32),use,add=True)
   h.feature(c,side,j.array([-1,9,0],j.int32),'rest_sleep',j.ones(3,j.int32),use)
   h.feature(c,side,p,'last_damage_taken',j.array([-0.,.25,.75],j.float32),use)
   s=s._replace(hp=s.hp.at[r,side,p].set(j.array([80,90,0],j.int32)),
    status=s.status.at[r,side,p].set(j.array([5,4,0],j.int32)))
   h.observe(c,s,use);h.move(c,s,side,j.array([1,1,1],j.int32),use,pp_use=True)
   incoming=s._replace(active=s.active.at[r,side].set(j.where(use,3,p)))
   h.install(c,incoming,side,use);h.baton_copy(c,incoming,side,p,use);h.withdraw(c,incoming,side,p,use)
   h.party_cure(c,incoming,incoming._replace(status=j.zeros_like(incoming.status)),side,use)
   h.turn_end(c,s,incoming,use)
   return c.trackers
  old=jax.jit(lambda t,s:work(t,s,dense=False))(public,state)
  new=jax.jit(lambda t,s:work(t,s,dense=True))(public,state)
  exact(self,old,new)
  for owner in (0,1):exact(self,tr.readout(old,j.full(3,owner,j.int32)),tr.readout(new,j.full(3,owner,j.int32)))
  with self.assertRaises(ValueError):h.Context(public,dense_updates=1)

 def test_observe_identity_current_masks_errors_and_unknown_members(self):
  state,_,public=fixture(3);r=j.arange(3);side=j.array([1,0,1],j.int32);physical=j.array([0,4,-1],j.int32)
  use=j.array([True,True,False]);state=state._replace(hp=state.hp.at[0,1,0].set(-1))
  for identity in (True,False):
   old=jax.jit(lambda t,s:tr._observe(t,s,side,physical,use,identity=identity))(public,state)
   new=jax.jit(lambda t,s:tr._observe(t,s,side,physical,use,identity=identity,dense_updates=True))(public,state)
   exact(self,old,new);self.assertNotEqual(int(new.error[0]),0)

 def test_actual_step_lowering_accepts_both_routes_without_execution(self):
  state,table,public=fixture(3);actions=j.zeros((3,2),j.int32);chance=transition.sample_chance_inputs(state)
  lower=lambda flag:jax.jit(lambda p:transition.step_with_public(state,actions,table,p,chance_inputs=chance,
   dense_public_updates=flag)).lower(public)
  old,new=lower(False),lower(True)
  self.assertEqual(jax.tree.structure(old.out_info),jax.tree.structure(new.out_info))
  for a,b in zip(jax.tree.leaves(old.out_info),jax.tree.leaves(new.out_info)):
   self.assertEqual(a.shape,b.shape);self.assertEqual(a.dtype,b.dtype)

 def test_all_other_functions_and_complete_numeric_transition_unchanged(self):
  changes={'gpu_engine/engine_device_public_hooks.py':{'Context','feature','observe'},
   'gpu_engine/engine_device_trackers.py':{'_observe'},'gpu_engine/transition.py':{'step_with_public'}}
  for path,names in changes.items():
   old=ast.parse(subprocess.check_output(['git','show',BASE+':'+path],text=True));new=ast.parse(Path(path).read_text())
   originals={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
   new.body=[originals[n.name] if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names else n for n in new.body]
   self.assertEqual(ast.dump(old),ast.dump(new),path)

if __name__=='__main__':unittest.main()
