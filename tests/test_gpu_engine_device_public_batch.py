"""Row addressing and lowering regressions; no engine/model execution."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import transition as t, entry
from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
from gpu_engine import benchmark_device_selfplay as bench, engine_device_observation as encoder
from gpu_engine.state import empty_state


def fixture(b=3):
 s=empty_state(b,seed=17);shape=(b,2,6)
 stats=j.full_like(s.stats,50).at[:,0,:,5].set(j.arange(b,dtype=j.int32)[:,None]+60)
 s=s._replace(species=j.broadcast_to(j.arange(1,13,dtype=j.int32).reshape(1,2,6),shape),
     stats=stats,level=j.full(shape,80,j.int32),gender=j.ones(shape,j.int32),
     hp=j.full(shape,100,j.int32),max_hp=j.full(shape,100,j.int32),moves=j.ones_like(s.moves),
     pp=j.full_like(s.pp,10),types=s.types.at[...,0].set(1),done=j.zeros(b,bool))
 table=t.MoveTable(j.array([0,40,50],j.int32),j.array([0,1,0],j.int32),j.array([False,True,True]),
     j.full(3,100,j.int32),j.ones(3,bool),j.zeros(3,j.int32),j.ones((18,18),j.int32),j.ones((18,18),j.int32),j.array(2,j.int32))
 meta={'vocabulary':{'abilities':['_none','naturalcure','trace','pressure','intimidate'],
     'types':['_none','normal','flying','ground'],'items':['_none'],'moves':['_none','tackle','struggle']}}
 table=table._replace(entry=entry.from_metadata(meta))
 return s,table,h.initialize(s).trackers

class PublicBatchTests(unittest.TestCase):
 def test_actual_public_step_lowers_batch_without_cross_row_shape(self):
  for b in (2,3):
   with self.subTest(batch=b):
    s,table,carry=fixture(b)
    actions=j.broadcast_to(j.asarray([4,0],j.int32),(b,2))
    chance=t.sample_chance_inputs(s)
    lowered=jax.jit(lambda state,actions,public,chance:t.step_with_public(state,actions,table,public,chance_inputs=chance)).lower(s,actions,carry,chance)
    self.assertEqual(lowered.out_info.step.state.boosts.shape,(b,2,6,7))
    self.assertEqual(lowered.out_info.trackers.boosts.shape,(b,2,6,7))
    self.assertEqual(lowered.out_info.trackers.physical_to_public.shape,(b,2,6))
 def test_mixed_row_sides_install_only_their_own_member(self):
  s,_,carry=fixture(3);sides=j.array([0,1,0],j.int32);use=j.array([True,False,True])
  s=s._replace(active=s.active.at[j.arange(3),sides].set(j.array([3,4,5],j.int32)))
  out=jax.jit(lambda public,state,side,eligible:tr._install(public,state,side,eligible))(carry,s,sides,use)
  self.assertEqual(out.active_physical.tolist(),[[3,0],[0,0],[5,0]])
  self.assertEqual(out.next_slot.tolist(),[[2,1],[1,1],[2,1]])
  self.assertEqual(out.species[0,0,:2].tolist(),[1,4]);self.assertEqual(out.species[2,0,:2].tolist(),[1,6])
  for old,new in zip(jax.tree.leaves(carry),jax.tree.leaves(out)):
   np.testing.assert_array_equal(old[1],new[1])
  # A uniform side vector and the old scalar form have identical results.
  scalar=jax.jit(lambda public:tr._install(public,s,0,use))(carry)
  vector=jax.jit(lambda public:tr._install(public,s,j.zeros(3,j.int32),use))(carry)
  for a,b in zip(jax.tree.leaves(scalar),jax.tree.leaves(vector)):np.testing.assert_array_equal(a,b)
 def test_both_driver_actor_shapes_lower_actual_public_step_without_model(self):
  s,table,carry=fixture(3);start=bench.initialize(s,carry,2)
  def encode(state,view,tables,*,observer_sides,legal_action_mask):
   b=len(observer_sides)
   return encoder.Observation(j.zeros((b,29,132),j.float32),j.zeros((b,29,41),j.int32),
       j.zeros((b,29),j.int32),j.ones((b,29),bool),legal_action_mask,
       j.broadcast_to(j.arange(1,6,dtype=j.int32),(b,5)),j.zeros(b,j.int32),
       j.zeros(len(encoder.ERROR_KINDS),j.int32),j.broadcast_to(j.arange(4,dtype=j.int32),(b,4)))
  kinds=[]
  def actor(params,config,observations,*,actor_kind):
   kinds.append(actor_kind);b=observations[0].numeric_features.shape[0]
   return bench.ActorResult(j.zeros((b,2,9)),j.zeros((b,2)),j.ones((b,2),bool))
  for kind in ('champion23','v5-public29'):
   with self.subTest(actor=kind),patch.object(encoder,'encode',encode),patch.object(bench,'run_actor',actor):
    driver=bench.make_driver(None,table,NS(belief=None),sampling_seed=9,chunk_steps=1,actor_kind=kind)
    lowered=driver.lower(start,s,carry,j.arange(3,dtype=j.uint32),{})
    self.assertEqual(lowered.out_info.trackers.boosts.shape,(2,2,6,7))
    self.assertEqual(lowered.out_info.state.hp.shape,(2,2,6))
  self.assertEqual(kinds,['champion23','v5-public29'])

if __name__=='__main__':unittest.main()
