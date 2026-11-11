"""Inert actor/transition controls: no actual model, engine turn or training."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import v5_device_rollout as d,transition,engine_device_observation as e,v5_model
from gpu_engine import engine_device_public_hooks as h
from pokezero.v5_model import TOKEN_ROLES
from pokezero.v5_loss import V5LossConfig
from tests.test_gpu_engine_v5_device_rollout import fixture


class DeviceActorBankTests(unittest.TestCase):
 def run_inert(self,bank,*,bindings=None,bad_actor=-1,stream_drift=False):
  s,carry,labels,config,original=fixture();bindings=original if bindings is None else bindings
  if stream_drift:carry=carry._replace(streams=carry.streams.at[0].set(999))
  calls=[]
  def encode(state,view,tables,*,observer_sides,legal_action_mask):
   b=len(observer_sides)
   numbers=j.zeros((b,29,132),j.float32).at[:,0,0].set(j.arange(b)*10+observer_sides)
   return e.Observation(numbers,j.zeros((b,29,41),j.int32),j.broadcast_to(j.array(TOKEN_ROLES,j.int32),(b,29)),
    j.ones((b,29),bool),legal_action_mask,j.broadcast_to(j.arange(1,6,dtype=j.int32),(b,5)),
    j.zeros(b,j.int32),j.zeros(len(e.ERROR_KINDS),j.int32),j.broadcast_to(j.arange(4,dtype=j.int32),(b,4)))
  def forward(params,inputs,config,*,compute_dtype):
   self.assertEqual(compute_dtype,'bfloat16');self.assertEqual(params['actor'].shape,())
   self.assertEqual(inputs['numeric_features'].shape[1:],(1,29,132));n=len(inputs['numeric_features']);calls.append(n)
   actor=params['actor'];row=inputs['numeric_features'][:,0,0,0]
   logits=j.full((n,9),-1000.,j.float32).at[:,actor].set(0.)
   values=j.where(actor==bad_actor,j.nan,actor*.1+row*.001)
   return v5_model.Output(logits,values,j.zeros((n,101),j.float32),j.ones(n,bool))
  def step(state,actions,table,public,*,belief_tables):
   active=~state.done;turn=state.turn+active.astype(j.int32);done=active&(turn>=2)
   after=state._replace(turn=turn,done=state.done|done,winner=j.where(done,j.arange(2),state.winner),
    forced_switch=j.zeros_like(state.forced_switch))
   return h.PublicStepResult(transition.StepResult(after,j.zeros(2,bool),j.zeros(2,j.int32)),public)
  with patch.object(e,'encode',encode),patch.object(v5_model,'forward',forward),\
   patch.object(transition,'step_with_public',step),patch.object(transition,'action_mask',lambda s,t:j.ones((2,2,9),bool)):
   collect=d.make_collector(NS(struggle=j.array(2,j.int32)),NS(belief=None),config,labels,bindings,
    sampling_seed=7,chunk_steps=3,max_requests=4,actor_bank=bank)
   params=tuple({'actor':j.array(i,j.int32)} for i in range(len(bank.actor_archives))) if bank is not None else ({'actor':j.array(0,j.int32)},)*2
   out=d.boundary(collect(carry,params))
  return out,calls

 def bank(self):
  *_,bindings=fixture()
  return d.ActorBank(bindings,('2'*64,'3'*64,'6'*64,'7'*64),(101,202),((0,1),(2,0)),((True,False),(False,True)),'8'*64)

 def test_one_model_exact_default_equivalence_and_every_row_retained(self):
  *_,bindings=fixture();bindings=replace(bindings,actor_archives=('2'*64,)*2)
  bank=d.ActorBank(bindings,('2'*64,),(101,202),((0,0),(0,0)),((True,False),)*2,'8'*64)
  old,old_calls=self.run_inert(None,bindings=bindings);new,calls=self.run_inert(bank,bindings=bindings)
  self.assertEqual(old_calls,[2,2]);self.assertEqual(calls,[4])
  for a,b in zip(jax.tree.leaves(old.carry),jax.tree.leaves(new.carry)):np.testing.assert_array_equal(a,b)
  for name in d.Frames._fields:
   if name not in ('actor_indices','learner_mask'):np.testing.assert_array_equal(getattr(old.frames,name),getattr(new.frames,name))
  self.assertEqual(new.frames.values.shape,(3,2,2));self.assertTrue((new.frames.actor_indices==0).all())

 def test_mixed_actor_groups_scatter_once_with_both_learner_seats_and_no_padding(self):
  bank=self.bank();out,calls=self.run_inert(bank);f=out.frames
  self.assertEqual(calls,[2,1,1]);self.assertEqual(sum(calls),4) # unused bank entry executes no placeholder rows
  self.assertEqual(f.actions[0].tolist(),[[0,1],[2,0]])
  np.testing.assert_allclose(f.values[0],[[0.,.101],[.210,.011]],rtol=0,atol=1e-7)
  np.testing.assert_array_equal(f.actor_indices,np.broadcast_to(bank.actor_indices,f.actor_indices.shape))
  np.testing.assert_array_equal(f.trainable,f.policy_valid&np.asarray(bank.learner_mask))
  self.assertEqual(f.trainable.sum(),4);self.assertTrue(f.trainable[:2,0,0].all());self.assertTrue(f.trainable[:2,1,1].all())
  self.assertFalse(f.trainable[:,0,1].any());self.assertFalse(f.trainable[:,1,0].any())
  self.assertEqual(out.carry.streams.tolist(),[101,202]);self.assertEqual(out.carry.status.tolist(),[d.COMPLETE,d.COMPLETE])

 def test_invalid_actor_and_changed_stream_are_refusals_without_substitution(self):
  bank=self.bank();out,_=self.run_inert(bank,bad_actor=1)
  self.assertEqual(out.carry.status.tolist(),[d.REFUSED,d.COMPLETE]);self.assertFalse(out.frames.trainable[:,0].any())
  self.assertFalse(out.frames.actor_finite[0,0,1]);self.assertFalse(out.frames.advanced[:,0].any())
  drift,_=self.run_inert(bank,stream_drift=True)
  self.assertEqual(drift.carry.status.tolist(),[d.REFUSED,d.COMPLETE]);self.assertFalse(drift.frames.trainable[:,0].any())

 def test_boundary_resolves_row_archives_and_masks_and_rejects_wrong_bank(self):
  bank=self.bank();out,_=self.run_inert(bank)
  with patch.object(d,'_complete_gae',side_effect=lambda v,o,c,discount:(np.zeros_like(v),np.full_like(v,o))) as gae:
   batch=d.training_batch((out,),bank.bindings,actor_bank=bank,battle_ids=('a','b'),seeds=(11,22),loss_config=V5LossConfig())
  self.assertEqual(gae.call_count,2);self.assertEqual(batch.player_ids,('p1','p2','p1','p2'))
  for meta,opponent in zip(batch.step_metadata,('3'*64,'6'*64)*2):
   self.assertEqual(meta['actor_archive_sha256'],'2'*64);self.assertEqual(meta['opponent_archive_sha256'],opponent)
   self.assertEqual(meta['actor_bank_assignment_sha256'],bank.assignment_sha256);self.assertTrue(meta['learner'])
  with self.assertRaisesRegex(ValueError,'bank rollout'):
   d.training_batch((out,),bank.bindings,battle_ids=('a','b'),seeds=(11,22),loss_config=V5LossConfig())
  forged=out._replace(frames=out.frames._replace(actor_indices=out.frames.actor_indices.copy()))
  forged.frames.actor_indices[0,0,0]=1
  with self.assertRaisesRegex(ValueError,'assignments differ'):
   d.training_batch((forged,),bank.bindings,actor_bank=bank,battle_ids=('a','b'),seeds=(11,22),loss_config=V5LossConfig())

 def test_original_planner_adapter_preserves_order_and_history_without_rng(self):
  _,carry,labels,config,bindings=fixture()
  default=d.make_collector(None,None,config,labels,bindings,sampling_seed=7)
  with self.assertRaisesRegex(ValueError,'two explicit original seat parameter trees required'):default(carry,())
  assignments=({'stream_id':202,'learner_side':1,'opponent':{'archive_sha256':'6'*64}},
   {'stream_id':101,'learner_side':0,'opponent':{'archive_sha256':'3'*64}})
  bank=d.actor_bank_from_assignments(bindings,('2'*64,'3'*64,'6'*64),current_archive='2'*64,assignments=assignments)
  self.assertEqual(bank.streams,(202,101));self.assertEqual(bank.actor_indices,((2,0),(0,1)))
  self.assertEqual(bank.learner_mask,((False,True),(True,False)));self.assertEqual(bank.assignment_sha256,d._sha(assignments))
  assignments[0]['opponent']['archive_sha256']='2'*64
  self.assertEqual(bank.actor_indices,((2,0),(0,1))) # detached immutable declaration
  for changes in (dict(actor_indices=((0,),(0,0))),dict(streams=(101,101)),dict(learner_mask=((1,0),(0,1)))):
   with self.assertRaises(ValueError):replace(bank,**changes)

if __name__=='__main__':unittest.main()
