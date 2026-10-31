"""Tiny CPU AMP destination/source join, FP32 identity, and unit schedule."""
import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from pokezero import v5_model as model,v5_training as training,v5_data as data
from pokezero import v5_cuda_training_component as component
import test_v5_model_loss as model_fixture
import test_v5_training as train_fixture


class RoleExpertAmpTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)

 def strict_index_copy(self):
  original=torch.Tensor.index_copy
  def copy(destination,dim,index,source):
   # CUDA index_copy requires this exact join. CPU autocast may itself
   # promote, which would otherwise conceal the source/destination mismatch.
   self.assertEqual(destination.dtype,source.dtype,'expert output must join original residual dtype before index_copy')
   return original(destination,dim,index,source)
  return patch.object(torch.Tensor,'index_copy',copy)

 def test_cpu_bf16_role_routing_forward_backward_has_exact_destination_dtype(self):
  torch.manual_seed(91);cfg=model_fixture.config('A');net=model.build_v5_model(cfg,enable_v5=True)
  routed=net.encoder.layers[0].role_experts if hasattr(net.encoder,'layers') else net.encoder.blocks[0].role_experts
  x=torch.randn(2,29,cfg.width,requires_grad=True);observed=[]
  handles=[expert.register_forward_hook(lambda _m,_a,out:observed.append(out.dtype))for expert in routed.experts]
  try:
   with self.strict_index_copy(),torch.autocast('cpu',dtype=torch.bfloat16):actual=routed(x)
  finally:
   for handle in handles:handle.remove()
  self.assertEqual(actual.dtype,x.dtype);self.assertEqual(observed,[torch.bfloat16]*3)
  expected=torch.zeros_like(x)
  with torch.autocast('cpu',dtype=torch.bfloat16):
   for role,expert in enumerate(routed.experts):
    indices=(routed.roles==role).nonzero().flatten();piece=expert(x.index_select(1,indices)).to(x.dtype)
    expected=expected.index_copy(1,indices,piece)
  self.assertTrue(torch.equal(actual,expected));actual.square().sum().backward()
  self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in routed.parameters()))
  for role in range(3):self.assertEqual(int((routed.roles==role).sum()),(12,9,8)[role])

 def test_tiny_all_arms_bf16_actual_forward_loss_backward(self):
  for arm in ('A','B','D'):
   with self.subTest(arm=arm):
    torch.manual_seed(37);cfg=model_fixture.config(arm);net=model.build_v5_model(cfg,enable_v5=True)
    tensors=model_fixture.v5_batch_to_torch(model_fixture.batch_columns(arm),cfg,training=True)
    with self.strict_index_copy(),torch.autocast('cpu',dtype=torch.bfloat16):
     output=model.forward_v5(net,tensors,include_auxiliary=cfg.treatment)
     loss,_=model_fixture.v5_loss(output,tensors,cfg,model_fixture.V5LossConfig(arm=arm))
    self.assertTrue(torch.isfinite(loss));loss.backward()
    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()for p in net.parameters()))

 def test_fp32_original_forward_initialization_rng_and_gradients_unchanged(self):
  original=subprocess.check_output(['git','show','33b13066:src/pokezero/v5_model.py'],text=True)
  role=next(n for n in ast.walk(ast.parse(original))if isinstance(n,ast.ClassDef)and n.name=='RoleExperts')
  fn=next(n for n in role.body if isinstance(n,ast.FunctionDef)and n.name=='forward');scope={'torch':torch}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-fp32-role-experts>','exec'),scope)
  cfg=model_fixture.config('A');torch.manual_seed(42);actual=model.build_v5_model(cfg,enable_v5=True);rng=torch.get_rng_state().clone()
  torch.manual_seed(42);expected=model.build_v5_model(cfg,enable_v5=True);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  for a,b in zip(actual.state_dict().values(),expected.state_dict().values(),strict=True):self.assertTrue(torch.equal(a,b))
  blocks=expected.encoder.layers if hasattr(expected.encoder,'layers') else expected.encoder.blocks
  for block in blocks:block.role_experts.forward=types.MethodType(scope['forward'],block.role_experts)
  tensors=model_fixture.v5_batch_to_torch(model_fixture.batch_columns('A'),cfg,training=True)
  left=model.forward_v5(actual,tensors);right=model.forward_v5(expected,tensors)
  self.assertTrue(torch.equal(left.policy_logits,right.policy_logits));self.assertTrue(torch.equal(left.value,right.value))
  (left.policy_logits.square().sum()+left.value.square().sum()).backward();(right.policy_logits.square().sum()+right.value.square().sum()).backward()
  for a,b in zip(actual.parameters(),expected.parameters(),strict=True):self.assertTrue(torch.equal(a.grad,b.grad))

 def test_original_unit_zero_start_schedule_has_positive_rate_and_changes_tiny_weights(self):
  cfg,loss,train=train_fixture.configurations(arm='A',epochs=2,precision='bfloat16')
  train=replace(train,learning_rate_schedule='mit-thesis');window=training.DeclaredGameWindow(0,1,2000000)
  self.assertEqual(training._epoch_learning_rate(train,window,1),train.learning_rate)
  self.assertGreater(training._epoch_learning_rate(train,window,2),0)
  with tempfile.TemporaryDirectory()as directory:
   cache=Path(directory)/'NON_GAME_CACHE';data.write_cache(cache,train_fixture.batch('A'))
   runtime=training.create(cfg,loss,train,enable_v5=True);before=component.tensor_tree_sha(runtime.model.state_dict(),torch)
   with self.strict_index_copy():first=training.train_caches(runtime,[cache],max_updates=1,learning_rate_window=window)
   after=component.tensor_tree_sha(runtime.model.state_dict(),torch)
   self.assertEqual(first['steps'],1);self.assertEqual(first['rows'],1);self.assertGreater(first['updates'][0]['gradient_norm'],0)
   self.assertEqual(first['updates'][0]['learning_rate'],train.learning_rate);self.assertNotEqual(before,after)
