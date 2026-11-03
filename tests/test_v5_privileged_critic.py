import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from pokezero import v5_privileged_critic as c
from gpu_engine import device_privileged_rows as prep


def doc(offset=0):
 return dict(numeric_features=np.arange(29*132,dtype=np.float32).reshape(29,132)+offset,
  categorical_ids=np.arange(29*41,dtype=np.int32).reshape(29,41)+offset,
  token_type_ids=np.asarray((0,)+(1,)*6+(2,)*6+(3,)*9+(5,)+(7,)*6,np.int32),attention_mask=np.ones(29,bool))


def row(stream,kind='child',available=True,target=.5):
 selector=dict(stream=stream,frame=4,owner=stream%2)
 return dict(selector=selector,kind=kind,available=available,target=target)

class CriticTests(unittest.TestCase):
 def test_exact_source_swap_preserves_public_and_all_other_tokens(self):
  a,b=doc(),doc(10000);original=copy.deepcopy(a);mapping=(5,3,1,4,2,0)
  v=c.privileged_arrays(a,b,mapping)
  for k in ('numeric_features','categorical_ids','attention_mask'):
   np.testing.assert_array_equal(v[k][7:13],b[k][1+np.asarray(mapping)])
   np.testing.assert_array_equal(v[k][:7],a[k][:7]);np.testing.assert_array_equal(v[k][13:],a[k][13:])
  np.testing.assert_array_equal(v['token_type_ids'],a['token_type_ids'])
  for k in a:np.testing.assert_array_equal(a[k],original[k])
 def test_real_physical_join_known_slots_and_explicit_unseen_completion(self):
  # physical0->slot2, physical3->slot0. Remaining actual physical members
  # fill unused slots ascending, then map to the peer's own request order.
  got=c.physical_slot_map(np.asarray([2,-1,-1,0,-1,-1],np.int32),np.asarray([4,3,1,5,0,2],np.int32))
  self.assertEqual(got,(1,2,4,5,0,3))
  for p in ([0,0,-1,-1,-1,-1],[6,-1,-1,-1,-1,-1]):
   with self.assertRaises(ValueError):c.physical_slot_map(np.asarray(p,np.int32),np.arange(6,dtype=np.int32))
 def test_unknown_or_foreign_tensor_not_zero_filled(self):
  for k in c.KEYS:
   bad=doc();bad[k]=bad[k][:-1]
   with self.assertRaises(ValueError):c.privileged_arrays(doc(),bad,range(6))
  bad=doc();bad['numeric_features']=bad['numeric_features'].astype(np.float64)
  with self.assertRaises(ValueError):c.privileged_arrays(doc(),bad,range(6))
 def test_five_folds_no_game_or_sibling_leak_and_no_grid_fit(self):
  records=[row(i,k,available=a,target=t)for i in range(100)for k,a,t in [('root',True,None),('child',True,.5),('child',False,.5),('grid',True,None)]]
  counts=[0]*len(records)
  for fold in range(5):
   indices=c.fold_indices(records,fold=fold)
   for i in indices['train']+indices['validation']:self.assertEqual(records[i]['kind'],'child');self.assertTrue(records[i]['available'])
   test_games={records[i]['selector']['stream']for i in indices['test']}
   self.assertFalse(test_games&{records[i]['selector']['stream']for i in indices['train']+indices['validation']})
   for i in indices['test']:counts[i]+=1
  self.assertEqual(set(counts),{1})
 def test_epoch0_eligible_validation_only_and_earliest_tie(self):
  h=[dict(epoch=0,validation_mse=.1),dict(epoch=1,validation_mse=.2),dict(epoch=2,validation_mse=.1)]
  self.assertEqual(c.select_epoch(h),0)
  h.append(dict(epoch=3,validation_mse=.09));self.assertEqual(c.select_epoch(h),3)
  with self.assertRaises(ValueError):c.select_epoch(h[1:])
 def test_optins_precede_any_runtime_or_model_work(self):
  for name in ('value_query','value_head.0.weight','value_head.4.bias'):self.assertTrue(c.head_parameter(name))
  for name in ('policy_head.weight','encoder.layers.0.norm1.weight','value_centers','value_query.extra','hidden_species_head.weight'):self.assertFalse(c.head_parameter(name))
  with patch.object(c,'production_binding',side_effect=AssertionError('must not read')):
   with self.assertRaises(ValueError):c.crossfit(None,None,None)
  with self.assertRaises(ValueError):prep.prepare(None,None,None)
  for kw in (dict(learning_rate=.1),dict(epochs=500),dict(seed=9)):
   with self.assertRaises(ValueError):c.FitConfig(**kw)
 def test_nonproduction_smoke_cannot_supply_trunk(self):
  from pokezero import v5_run
  plan=dict(schema=v5_run.SCHEMA,model=dict(arm='B'),iterations=[[dict(games=[{}]*20000)]])
  def event(kind,payload,seq):
   e=dict(kind=kind,payload=payload,sequence=seq,previous=None);return dict(e,sha256=c.truth().digest(e))
  start=event('RUN_STARTED',dict(declaration_sha256=c.truth().digest(plan)),0)
  ckpt=event('CHECKPOINT',dict(steps=30,rows=300,sha256='a'*64),1)
  with patch.object(c,'artifact',side_effect=[plan,start,ckpt]):
   with self.assertRaises(ValueError):c.production_binding(dict(production_declaration={},run_started_event={},checkpoint_event={},checkpoint=dict(sha256='a'*64)))
 def test_full_oos_metrics_keep_missing_and_use_actual_reply_grid(self):
  def outcomes(v):return [dict(replicate=i,status='COMPLETE',outcome=v)for i in range(64)]
  selector=dict(stream=1,frame=4,owner=0)
  records=[dict(row_id='root',selector=selector,kind='root',outcome=1),dict(row_id='a',selector=selector,kind='child',action=0,rollouts=outcomes(1)),dict(row_id='b',selector=selector,kind='child',action=1,rollouts=outcomes(-1)),dict(row_id='g',selector=selector,kind='grid')]
  grid=dict(selector=selector,status='OBSERVED',own_actions=[0],replies=[0],reply_weights=[1.],rows=[dict(row_id='g',action=0,reply=0,weight=1.,status='OBSERVED',terminal_value=None)])
  m=c.score_predictions(dict(grids=[grid]),records,[0.,1.,-1.,1.]);self.assertEqual(m['decision'],'PASS')
  m=c.score_predictions(dict(grids=[grid]),records,[0.,None,-1.,None]);self.assertEqual(m['decision'],'UNKNOWN_INCOMPLETE_DENOMINATOR');self.assertEqual(m['child_unknown'],1)


class FitControllerTests(unittest.TestCase):
 def test_actual_controller_fits_train_selects_validation_restores_best(self):
  weights={'value':0};seen=[];losses=iter([.3,.4,.2,.2])
  def update(ids,epoch):
   self.assertEqual(ids,[1,2]);weights['value']=epoch;seen.append(epoch);return 2
  def validate(ids):self.assertEqual(ids,[3]);return next(losses)
  result=c.fit_fold([1,2],[3],3,train_epoch=update,validate=validate,snapshot=lambda:dict(weights),restore=lambda w:weights.update(w))
  self.assertEqual(result['selected_epoch'],2);self.assertEqual(weights['value'],2)
  self.assertEqual(result['optimizer_updates'],6);self.assertEqual(seen,[1,2,3])
 def test_missing_validation_never_trains_and_epoch0_remains(self):
  calls=[]
  result=c.fit_fold([1],[],3,train_epoch=lambda *a:calls.append(a),validate=lambda ids:None,snapshot=lambda:'public-head',restore=lambda value:self.assertEqual(value,'public-head'))
  self.assertEqual(result['selected_epoch'],0);self.assertEqual(result['optimizer_updates'],0);self.assertFalse(calls)

if __name__=='__main__':unittest.main()
