"""Lean synthetic public-carry tests; no engine turn/model/forward/game."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_trackers as tracker
from gpu_engine.state import empty_state

INIT=jax.jit(tracker.initialize)
UPDATE=jax.jit(tracker.update)
READ=jax.jit(tracker.readout)
CLONE=jax.jit(tracker.clone_for_worlds)

class DeviceTrackersTests(unittest.TestCase):
 def state(self,b=3):
  species=j.broadcast_to(j.arange(1,13,dtype=j.int32).reshape(1,2,6),(b,2,6))
  s=empty_state(b)._replace(species=species,level=j.full((b,2,6),80,j.int32),gender=j.ones((b,2,6),j.int32),hp=j.full((b,2,6),513,j.int32),max_hp=j.full((b,2,6),513,j.int32),done=j.zeros(b,bool))
  return s
 def same(self,a,b,skip=()):
  for n in a._fields:
   if n not in skip:
    x,y=getattr(a,n),getattr(b,n)
    if x is None:self.assertIsNone(y,n)
    else:np.testing.assert_array_equal(x,y,err_msg=n)
 def test_initial_only_active_public_and_percent(self):
  s=self.state();s=s._replace(hp=s.hp.at[:,1,0].set(j.array([512,257,0],j.int32)))
  t=INIT(s);v=READ(t,j.zeros(3,j.int32))
  self.assertEqual(v.species.tolist(),[[7,0,0,0,0,0]]*3);self.assertEqual(v.present.tolist(),[[True,False,False,False,False,False]]*3)
  np.testing.assert_allclose(v.hp_fraction[:,0],np.array([.99,.51,0],np.float32));self.assertEqual(v.fainted[:,0].tolist(),[False,False,True])
  self.assertFalse(bool(v.ability.any()));self.assertFalse(bool(v.item.any()));self.assertFalse(bool(v.types.any()));self.assertIsNone(v.posterior_weights)
  self.assertEqual(v.error.tolist(),[0]*3)
  for name in ('observed_moves','pp_used','last_moves','tendencies','belief_mask'):self.assertIsNone(getattr(t,name))
  # Schema v3: the public request order starts as identity (the lead is public slot 0) and is never read out.
  self.assertEqual(tracker.SCHEMA,'pokezero.engine_device_public_trackers.v3')
  self.assertEqual(t.public_order.tolist(),[[list(range(6))]*2]*3);self.assertNotIn('public_order',tracker.PublicView._fields)
 def test_private_opponent_and_hidden_bench_noninterference(self):
  s=self.state();base=READ(INIT(s),j.zeros(3,j.int32))
  changed=s._replace(ability=s.ability.at[:,1].set(71),item=s.item.at[:,1].set(122),moves=s.moves.at[:,1].set(344),pp=s.pp.at[:,1].set(2),stats=s.stats.at[:,1].set(999),types=s.types.at[:,1].set(17),base_types=s.base_types.at[:,1].set(17),status_counter=s.status_counter.at[:,1].set(13),hidden_power_type=s.hidden_power_type.at[:,1].set(4),hidden_power_power=s.hidden_power_power.at[:,1].set(30),happiness=s.happiness.at[:,1].set(2))
  changed=changed._replace(species=changed.species.at[:,1,1:].set(99),level=changed.level.at[:,1,1:].set(1),gender=changed.gender.at[:,1,1:].set(2),hp=changed.hp.at[:,1,1:].set(1),status=changed.status.at[:,1,1:].set(4),boosts=changed.boosts.at[:,1,1:].set(6))
  self.same(base,READ(INIT(changed),j.zeros(3,j.int32)))
  # Updates likewise ignore hidden private values and unrelated bench state.
  t=INIT(s);a=j.zeros((3,2),j.int32);left=UPDATE(t,s,s,a);right=UPDATE(t,s,changed,a)
  self.same(READ(left,j.zeros(3,j.int32)),READ(right,j.zeros(3,j.int32)))
 def test_switch_identity_order_public_updates_and_inactive_rows(self):
  s=self.state();t=INIT(s);after=s._replace(active=s.active.at[:,1].set(3),hp=s.hp.at[:,1,0].set(200).at[:,1,3].set(300),status=s.status.at[:,1,3].set(2),boosts=s.boosts.at[:,1,3,2].set(2),weather=j.ones(3,j.int32),spikes=s.spikes.at[:,0].set(2),turn=j.ones(3,j.int32))
  active=j.array([True,False,True]);out=UPDATE(t,s,after,j.full((3,2),4,j.int32),active_rows=active);v=READ(out,j.zeros(3,j.int32))
  self.assertEqual(v.species[0].tolist(),[7,10,0,0,0,0]);self.assertEqual(v.active[0].tolist(),[False,True,False,False,False,False]);self.assertEqual(v.status[0,1],2);self.assertEqual(v.boosts[0,1,2],2)
  self.assertEqual(READ(out,j.ones(3,j.int32)).own_order[0].tolist(),[3,1,2,0,4,5]);self.assertEqual(out.public_order[:,1].tolist(),[[1,0,2,3,4,5],list(range(6)),[1,0,2,3,4,5]]);self.assertEqual(v.weather.tolist(),[1,0,1]);self.assertEqual(v.spikes.tolist(),[[2,0],[0,0],[2,0]])
  for name in t._fields:
   if getattr(t,name)is not None:np.testing.assert_array_equal(getattr(t,name)[1],getattr(out,name)[1],err_msg=name)
  back=after._replace(active=after.active.at[:,1].set(0));out2=UPDATE(out,after,back,j.zeros((3,2),j.int32),active_rows=active)
  self.assertEqual(out2.next_slot[:,1].tolist(),[2,1,2]);self.assertEqual(READ(out2,j.ones(3,j.int32)).own_order[0].tolist(),list(range(6)))
  self.assertEqual(out2.public_order[0,1].tolist(),list(range(6)))
  self.assertEqual(READ(out2,j.zeros(3,j.int32)).active[0].tolist(),[True,False,False,False,False,False])
 def test_root_public_clone_for_worlds_no_sampled_state_input(self):
  s=self.state(2);t=INIT(s);rows=j.array([1,1,0,0],j.int32);worlds=CLONE(t,rows)
  for name in t._fields:
   if getattr(t,name)is not None:np.testing.assert_array_equal(getattr(worlds,name),np.asarray(getattr(t,name))[np.array(rows)],err_msg=name)
  self.assertEqual(READ(worlds,j.array([0,1,0,1],j.int32)).species[:,0].tolist(),[7,1,7,1])
  bad=READ(CLONE(t,j.array([-1,2],j.int32)),j.zeros(2,j.int32));self.assertFalse(bool(bad.present.any()));self.assertEqual(bad.error.tolist(),[tracker.BAD_ROOT]*2)
 def test_no_reveal_from_replaced_before_active_or_invalid_observer(self):
  s=self.state();t=INIT(s);sample=s._replace(active=s.active.at[:,1].set(4));out=UPDATE(t,sample,sample,j.zeros((3,2),j.int32));self.same(t,out,skip=('error',));self.assertEqual(out.error.tolist(),[tracker.BEFORE_ACTIVE_MISMATCH]*3)
  self.assertEqual(READ(t,j.full(3,2,j.int32)).error.tolist(),[tracker.BAD_OBSERVER]*3)
  with self.assertRaises(ValueError):READ(t,j.zeros(3,j.int64)) if jax.config.x64_enabled else READ(t,j.zeros(3,j.float32))

if __name__=='__main__':unittest.main()
