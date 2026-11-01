"""Inert actor/transition controls; no real models/engine turns/games."""
from dataclasses import replace
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import tempfile
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import v5_device_rollout as d,transition,engine_device_observation as e,v5_model
from gpu_engine import engine_device_public_hooks as h
from gpu_engine.state import empty_state
from pokezero.v5_model import V5ModelConfig,TOKEN_ROLES
from pokezero.v5_contract import LabelVocabulary
from pokezero.v5_loss import V5LossConfig


def fixture():
 vocab=LabelVocabulary('0'*64,('tackle','struggle'),(('gardevoir',2),('snorlax',2)))
 config=V5ModelConfig(16,'1'*64,vocab,arm='B',width=16,layers=1,attention_heads=4,expert_width=16)
 bindings=d.Bindings('1'*64,vocab,'4'*64,('2'*64,'3'*64),(True,False),'5'*64)
 s=empty_state(2,seed=17);shape=(2,2,6)
 s=s._replace(species=j.broadcast_to(j.array([1,2],j.int32)[None,:,None],shape),level=j.full(shape,80,j.int32),
     gender=j.ones(shape,j.int32),hp=j.full(shape,100,j.int32),max_hp=j.full(shape,100,j.int32),
     moves=j.ones_like(s.moves),pp=j.full_like(s.pp,10),done=j.zeros(2,bool),
     forced_switch=s.forced_switch.at[1,1].set(True))
 public=h.initialize(s).trackers
 truth=d.ConstructorTruth(j.zeros((2,2,6,64),j.float32).at[...,0].set(1),
     j.broadcast_to(j.array([0,1],j.int32)[None,:,None],shape),j.ones(shape,bool))
 carry=d.initialize((s,public),np.array([101,202],np.int32),truth=truth)
 labels=d.compile_labels({'vocabulary':{'moves':['_none','tackle','struggle'],'species':['_none','gardevoir','snorlax']}},vocab)
 return s,carry,labels,config,bindings

class DeviceRolloutTests(unittest.TestCase):
 def run_fake(self,*,bad_actor=False):
  s,carry,labels,config,bindings=fixture();seen=[]
  def encode(state,view,tables,*,observer_sides,legal_action_mask):
   b=len(observer_sides);numeric=j.zeros((b,29,132),j.float32).at[:,0,0].set(state.turn.astype(j.float32))
   return e.Observation(numeric,j.zeros((b,29,41),j.int32),j.broadcast_to(j.array(TOKEN_ROLES,j.int32),(b,29)),
       j.ones((b,29),bool),legal_action_mask,j.broadcast_to(j.arange(1,6,dtype=j.int32),(b,5)),
       j.zeros(b,j.int32),j.zeros(len(e.ERROR_KINDS),j.int32),j.broadcast_to(j.arange(4,dtype=j.int32),(b,4)))
  def forward(params,inputs,config,*,compute_dtype):
   self.assertEqual(compute_dtype,'bfloat16');self.assertEqual(set(inputs),{'numeric_features','categorical_ids','token_type_ids','attention_mask','history_mask'})
   self.assertEqual(inputs['numeric_features'].shape,(2,1,29,132));self.assertEqual(params['seat'].shape,())
   seen.append('actor');b=inputs['numeric_features'].shape[0]
   return v5_model.Output(j.zeros((b,9)),j.full(b,j.nan if bad_actor else .2),j.zeros((b,101)),j.ones(b,bool))
  def step(state,actions,table,public,*,belief_tables):
   active=~state.done;turn=state.turn+active.astype(j.int32)
   ended=active&(j.arange(2)==0)&(turn>=2)
   after=state._replace(turn=turn,done=state.done|ended,winner=j.where(ended,0,state.winner),forced_switch=j.zeros_like(state.forced_switch))
   return h.PublicStepResult(transition.StepResult(after,j.zeros(2,bool),j.zeros(2,j.int32)),public)
  with patch.object(e,'encode',encode),patch.object(v5_model,'forward',forward),\
       patch.object(transition,'step_with_public',step),patch.object(transition,'action_mask',lambda s,t:j.ones((2,2,9),bool)):
   collector=d.make_collector(NS(struggle=j.array(2,j.int32)),NS(belief=None),config,labels,bindings,sampling_seed=7,chunk_steps=4,max_requests=3)
   out=collector(carry,({'seat':j.array(0)},{'seat':j.array(1)}))
   out=d.boundary(out)
  self.assertEqual(seen,['actor','actor'])
  return out,bindings
 def test_device_chunk_records_raw29_requests_fixed_cohort_and_literal_aux(self):
  out,bindings=self.run_fake();f=out.frames;c=out.carry
  self.assertEqual(f.numeric_features.shape,(4,2,2,29,132));self.assertEqual(c.frames.tolist(),[2,3])
  self.assertEqual(c.streams.tolist(),[101,202]);self.assertEqual(c.status.tolist(),[d.COMPLETE,d.CAPPED])
  self.assertEqual(f.requested[0].tolist(),[[True,True],[False,True]])
  self.assertFalse(f.requested[3].any());self.assertFalse(f.trainable[:,:,1].any())
  self.assertEqual(int(f.trainable.sum()),4)
  for t,row,owner in np.argwhere(f.trainable):
   self.assertTrue(f.legal_action_mask[t,row,owner,f.actions[t,row,owner]])
   self.assertGreater(f.probabilities[t,row,owner],0)
  # Exactly the originally public encountered foe is label-addressable.
  self.assertTrue(f.hidden_mask[0,0,0,0]);self.assertFalse(f.hidden_mask[0,0,0,1:].any())
  self.assertEqual(f.physical[0,0,0].tolist(),[0,1,2,3,4,5,0,-1,-1,-1,-1,-1])
  self.assertTrue(f.semantic_mask[0,0].all())
 def test_boundary_connects_original_v5_cache_and_masks_unknown_terminals(self):
  out,bindings=self.run_fake()
  # This local JAX-only environment has no Torch. Exercise row/cache mapping
  # with a TEST-ONLY inert seam; production still calls original lazy GAE.
  with patch.object(d,'_complete_gae',side_effect=lambda values,outcome,config,discount:(np.zeros_like(values),np.full_like(values,outcome))) as gae:
   batch=d.training_batch((out,),bindings,battle_ids=('game101','game202'),seeds=(11,22),loss_config=V5LossConfig())
  self.assertEqual(gae.call_count,1)
  from pokezero import v5_data
  v5_data.validate_batch(batch)
  self.assertEqual(batch.batch_size,4);self.assertEqual(batch.numeric_features.shape,(4,1,29,132))
  complete=np.asarray(batch.v5_columns['terminal_status'])=='COMPLETE'
  np.testing.assert_array_equal(batch.ppo_advantage_mask,complete);np.testing.assert_array_equal(batch.ppo_value_target_mask,complete)
  np.testing.assert_array_equal(batch.v5_columns['outcome_target_mask'],complete)
  self.assertFalse(np.asarray(batch.v5_columns['pokemon_end_target_mask'])[~complete].any())
  self.assertFalse(np.asarray(batch.search_policy_target_mask).any())
  with tempfile.TemporaryDirectory() as temp:
   meta=v5_data.write_cache(Path(temp)/'actual',batch)
   self.assertEqual(meta['example_count'],4)
   restored=tuple(v5_data.iter_cache_batches(Path(temp)/'actual',batch_size=4))[0]
   np.testing.assert_array_equal(restored.numeric_features,batch.numeric_features)
   np.testing.assert_array_equal(restored.v5_columns['hidden_set_targets'],batch.v5_columns['hidden_set_targets'])
 def test_bad_raw_actor_closes_fixed_cell_without_fallback_training_labels(self):
  out,_=self.run_fake(bad_actor=True)
  self.assertEqual(out.carry.status.tolist(),[d.REFUSED,d.REFUSED])
  self.assertEqual(out.carry.frames.tolist(),[1,1])
  self.assertFalse(out.frames.trainable.any());self.assertFalse(out.frames.advanced.any())
  self.assertFalse(out.frames.semantic_mask.any());self.assertFalse(out.frames.hidden_mask.any())
  self.assertFalse(out.carry.terminal_hp.any())
 def test_no_legacy_capture_and_d_has_explicit_public_root_handoff(self):
  s,c,labels,config,bindings=fixture()
  with self.assertRaisesRegex(ValueError,'D teacher'):
   d.make_collector(None,None,replace(config,arm='D'),labels,bindings,sampling_seed=7)
  context=d.public_root_context(c);self.assertIs(context.public,c.public)
  with patch('gpu_engine.device_public_leaf.from_roots',return_value='leaf') as leaf:
   self.assertEqual(d.leaf_from_roots(context,'world','rows','tables',observer_sides='owners',legal_action_mask='legal'),'leaf')
  self.assertIs(leaf.call_args.args[1],c.public)
  source=Path(d.__file__).read_text();tree=ast.parse(source)
  names={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
  self.assertFalse(any(n and any(x in n for x in ('retained_encoder_driver','owning_history','depth1_provider','full_v4')) for n in names))
  for name in ('device_get','asarray'):
   # No transfer or host numpy conversion appears inside the JIT factory.
   factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_collector')
   self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
       and ((n.func.value.id=='jax' and n.func.attr=='device_get') or n.func.value.id=='np') for n in ast.walk(factory)))

if __name__=='__main__':unittest.main()
