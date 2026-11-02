"""Injected runtime/archive/actor controls; no actual model/game/training."""
from dataclasses import replace
import ast
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import jax.numpy as j
import numpy as np
from gpu_engine import v5_device_backend as b,v5_device_rollout as d,public_belief as pb
from pokezero import v5_run as r,v5_contract as c,v5_phasec as phase,v5_worker as worker
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatVariant,Gen3RandbatSpeciesUniverse,RandbatSourceMetadata
from pokezero.v5_loss import V5LossConfig
from tests.test_gpu_engine_v5_device_rollout import fixture
from tests import test_gpu_engine_v5_device_actor_bank as bank_tests
from tests.test_v5_run import plan as original_plan


def inputs():
 _,carry,_,model,_=fixture();model=replace(model,width=512,layers=3,attention_heads=8,expert_width=512)
 games=(r.Game(101,'a',11,(0,1)),r.Game(202,'b',22,(2,3)))
 plan=replace(original_plan(),model=model,loss=V5LossConfig(),iterations=((r.Batch(games,17,7),),),max_requests=3)
 metadata={'vocabulary':{'moves':['_none','tackle','struggle'],'species':['_none','gardevoir','snorlax']}}
 prepared=b.Prepared(NS(source=object()),object(),object(),NS(belief=None),plan.pool_sha256,json.dumps(metadata),
  json.dumps({'source_commit':'a'*40}),'1'*64,16)
 return plan,prepared,carry


class DeviceBackendTests(unittest.TestCase):
 def setup_backend(self,root):
  plan,prepared,carry=inputs();store=r.Store(root,plan,resume=False);backend=b.RuntimeBackend(prepared,NS(arrays={}),chunk_steps=3)
  snapshots=[]
  for index in range(3):
   artifact=store.artifact(f'actors/{index}.npz',f'inert-{index}'.encode(),kind='INERT_TEST_ACTOR')
   snapshots.append(dict(archive=artifact['path'],archive_sha256=artifact['sha256'],checkpoint_sha256=str(index+2)*64,model_sha256=r._sha(plan.model.metadata())))
  assignments=({'stream_id':101,'learner_side':0,'opponent':snapshots[1]},
   {'stream_id':202,'learner_side':1,'opponent':snapshots[2]})
  return plan,backend,store,snapshots,assignments,carry

 def test_verified_unique_loads_exact_planner_bank_and_one_callable_per_batch(self):
  with tempfile.TemporaryDirectory()as root:
   plan,backend,store,snapshots,assignments,_=self.setup_backend(root);calls=[]
   def load(path,**kw):
    self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),kw['expected_archive_sha256'])
    self.assertEqual(kw['expected_category_vocabulary_sha256'],plan.model.category_vocabulary_sha256)
    self.assertEqual(kw['expected_label_vocabulary_sha256'],plan.model.label_vocabulary.sha256);self.assertIs(kw['enable_v5'],True)
    params={'inert':j.array(len(calls))};calls.append((params,kw));return params,plan.model,{'checkpoint_sha256':kw['expected_checkpoint_sha256']}
   with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),\
    patch.object(b.v5_model,'load',side_effect=load),patch.object(d,'make_collector',return_value='same-bank-collector')as factory:
    first=backend.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
    second=backend.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
    self.assertEqual(len(calls),3);self.assertEqual(factory.call_count,2)
    self.assertEqual(first['bank'].actor_indices,((0,1),(2,0)));self.assertEqual(first['bank'].learner_mask,((True,False),(False,True)))
    self.assertEqual(first['bank'].assignment_sha256,d._sha(assignments));self.assertEqual(first['bank'].streams,(101,202))
    self.assertTrue(all(x is y for x,y in zip(first['params'],second['params'])))
    self.assertEqual(len(first['params']),3);self.assertEqual(first['params'][0]['inert'].shape,())
    self.assertIs(first['collector'],'same-bank-collector')
    with self.assertRaisesRegex(ValueError,'checkpoint provenance'):
     backend._actor(dict(snapshots[0],checkpoint_sha256='f'*64),store)

 def test_original_planner_snapshot_objects_match_original_journal_binding(self):
  with tempfile.TemporaryDirectory()as root:
   plan,backend,store,snapshots,_,_=self.setup_backend(root)
   typed=tuple(r.Snapshot(i-1,f'checkpoints/{i}.pt',x['checkpoint_sha256'],x['archive'],x['archive_sha256'],x['model_sha256'])for i,x in enumerate(snapshots))
   assignments=tuple(r.assignment(plan,game,i,typed[0],typed[1:])for i,game in enumerate(plan.iterations[0][0].games))
   with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),\
    patch.object(b.v5_model,'load',side_effect=lambda path,**kw:({'inert':j.array(0)},plan.model,{'checkpoint_sha256':kw['expected_checkpoint_sha256']})),\
    patch.object(d,'make_collector',return_value=None):
    pending=backend.prepare_batch(plan,plan.iterations[0][0],typed[0],assignments,store)
   self.assertEqual(pending['bank'].assignment_sha256,r._sha(assignments));self.assertEqual(pending['bank'].streams,(101,202))
   for row,assignment in enumerate(assignments):
    peer=1-assignment['learner_side'];index=pending['bank'].actor_indices[row][peer]
    self.assertEqual(pending['bank'].actor_archives[index],assignment['opponent']['archive_sha256'])

 def test_actual_bank_to_original_cache_boundary_and_exact_dispositions(self):
  with tempfile.TemporaryDirectory()as root:
   plan,backend,store,snapshots,assignments,carry=self.setup_backend(root)
   with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),\
    patch.object(b.v5_model,'load',side_effect=lambda path,**kw:({'inert':j.array(0)},plan.model,{'checkpoint_sha256':kw['expected_checkpoint_sha256']})),\
    patch.object(d,'make_collector',return_value=None):
    pending=backend.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
   # Genuine reviewed collector code, with only inert actor/step seams.
   raw,_=bank_tests.DeviceActorBankTests().run_inert(pending['bank'],bindings=pending['bindings'])
   seen=[]
   pending['collector']=lambda actual,params:(seen.append((actual,params))or raw)
   ended=[]
   with patch.object(backend,'_construct',return_value=carry)as constructor,\
    patch.object(d,'_complete_gae',side_effect=lambda values,outcome,loss,discount:(np.zeros_like(values),np.full_like(values,outcome)))as gae:
    cache,digest=backend.collect(pending,store,lambda *a,**kw:ended.append((a,kw)),lambda:None)
   constructor.assert_called_once_with(plan.iterations[0][0]);self.assertEqual(len(seen),1)
   self.assertIs(seen[0][0],carry);self.assertIs(seen[0][1],pending['params']);self.assertEqual(gae.call_count,2)
   self.assertEqual([x[0]for x in ended],[(101,'COMPLETE'),(202,'COMPLETE')])
   self.assertEqual(r._file_sha(store.path/cache/'metadata.json'),digest)
   batch=next(b.v5_data.iter_cache_batches(store.path/cache,batch_size=4))
   self.assertEqual(tuple(batch.player_ids),('p1','p2','p1','p2'))
   self.assertTrue(all(meta['actor_archive_sha256']==snapshots[0]['archive_sha256']for meta in batch.step_metadata))
   self.assertEqual([m['opponent_archive_sha256']for m in batch.step_metadata],[snapshots[1]['archive_sha256'],snapshots[2]['archive_sha256']]*2)
   chunks=store.matching('DEVICE_ROLLOUT_CHUNK');self.assertEqual(len(chunks),1)
   raw=json.loads(store.resolve(chunks[0]['path'],chunks[0]['sha256']).read_text())
   array=lambda name:np.swapaxes(np.load(store.path/raw['frames'][name]['path'],allow_pickle=False),0,1)
   self.assertEqual(array('numeric_features').shape,(3,2,2,29,132));self.assertEqual(array('actor_indices')[0].tolist(),[[0,1],[2,0]])
   self.assertFalse(array('trainable')[:,0,1].any());self.assertFalse(array('trainable')[:,1,0].any())

 def test_constructor_uses_original_selection_seed_gender_and_fresh_initializer(self):
  plan,prepared,carry=inputs();batch=replace(plan.iterations[0][0],gender_seed=9,gender_domain=10,gender_root_ids=(31,32))
  backend=b.RuntimeBackend(prepared,NS(arrays={'gender':'retained-original'}),gender_policy={'genuine':'policy'});backend._plan=plan
  truth=object();raw=object();after_gender=object();initialized=object()
  with patch.object(b.transition,'state_from_pool',return_value=raw)as birth,\
   patch('gpu_engine.engine_request_source._gender',return_value=after_gender)as genders,\
   patch.object(d,'constructor_truth',return_value=truth)as labels,patch.object(backend,'initialize',return_value=initialized)as initialize,\
   patch.object(d,'initialize',return_value='fresh-carry')as register:
   self.assertEqual(backend._construct(batch),'fresh-carry')
  self.assertIs(birth.call_args.args[0],backend.pool.arrays);np.testing.assert_array_equal(birth.call_args.args[1],[[0,1],[2,3]])
  self.assertEqual(birth.call_args.kwargs,{'seed':17});self.assertIs(genders.call_args.args[0],raw)
  opts=genders.call_args.args[3];self.assertEqual((opts['seed'],opts['domain']),(9,10));np.testing.assert_array_equal(opts['root_ids'],[31,32])
  self.assertIs(labels.call_args.args[0],after_gender);self.assertIs(labels.call_args.kwargs['source'],prepared.source)
  self.assertIs(labels.call_args.kwargs['compiled_source'],prepared.catalog.source);initialize.assert_called_once_with(after_gender)
  self.assertIs(register.call_args.args[0],initialized);np.testing.assert_array_equal(register.call_args.args[1],[101,202])

 def test_real_raw_source_axis_and_compiled_constructor_matching_are_distinct(self):
  variant=Gen3RandbatVariant('a','set','Swalot','role',80,('Tackle','Surf','Protect','Rest'),'Sticky Hold','Leftovers')
  source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'c'*64),
   {'swalot':Gen3RandbatSpeciesUniverse('Swalot',80,(variant,replace(variant,variant_id='duplicate')))})
  compiled=pb.compile_source(source);vocabulary=c.LabelVocabulary(c.source_variant_census(source)['source_rows_sha256'],('tackle','surf','protect','rest'),(('swalot',2),))
  from gpu_engine.state import empty_state
  shape=(1,2,6);one=j.ones(shape,j.int32)
  state=empty_state(1)._replace(species=one,level=one*80,base_ability=one,item=one,
   moves=j.broadcast_to(j.arange(1,5,dtype=j.int32),(1,2,6,4)),hidden_power_type=one)
  arrays={key:np.asarray(getattr(state,field))[0]for field,key in [('species','species'),('level','level'),('moves','moves'),('base_ability','ability'),('item','item'),('hidden_power_type','hidden_power_type')]}
  metadata={'vocabulary':{'species':['_none','swalot'],'moves':['_none','tackle','surf','protect','rest'],
   'abilities':['_none','stickyhold'],'items':['_none','leftovers'],'types':['_none','normal']}}
  pool=NS(arrays=arrays,metadata=lambda:metadata)
  truth=d.constructor_truth(state,pool,np.array([[0,1]],np.int32),source=source,compiled_source=compiled,vocabulary=vocabulary)
  self.assertTrue(truth.known.all());np.testing.assert_array_equal(truth.probabilities[...,:2],np.full((1,2,6,2),.5,np.float32))
  with self.assertRaisesRegex(ValueError,'compiled constructor-label'):
   d.constructor_truth(state,pool,np.array([[0,1]],np.int32),source=source,compiled_source=replace(compiled,source_hash='d'*64),vocabulary=vocabulary)

 def test_explicit_factory_worker_route_and_D_refusal_keep_legacy_default(self):
  plan,prepared,_=inputs()
  with patch('gpu_engine.v5_device_backend.RuntimeBackend',return_value='device')as new,\
   patch('gpu_engine.v5_selfplay.EngineBackend',return_value='legacy')as old:
   self.assertEqual(phase.build_backend(plan,prepared,'pool',device_rollout=True),'device');old.assert_not_called()
   self.assertEqual(phase.build_backend(plan,'old-prepared','pool'),'legacy');self.assertEqual(new.call_count,1)
  dplan=replace(plan,model=replace(plan.model,arm='D'),loss=V5LossConfig(arm='D'))
  with self.assertRaisesRegex(r.MissingDependency,'no FULL fallback'):phase.build_backend(dplan,prepared,'pool',device_rollout=True)
  with self.assertRaisesRegex(r.MissingDependency,'no FULL fallback'):b.RuntimeBackend(prepared,'pool').check_plan(dplan)
  plan=replace(plan,evaluation=r.EvaluationProtocol('e'*64));input=NS(plan=plan,runtime={'memory_policy':{}},gender_policy=None)
  with patch.object(worker,'_source'),patch.object(worker,'_memory'),patch.object(worker,'_probe'),\
   patch.object(worker,'prepare_device_runtime',return_value=(prepared,'pool',None,None))as prep,patch.object(worker,'prepare_runtime')as legacy,\
   patch.object(phase,'build_backend',return_value='actual-device-backend')as factory,patch.object(r,'run',return_value='original-run')as run:
   hook=lambda request:None
   self.assertEqual(worker.execute(input,'/tmp/inert-source-test',device_rollout=True,evaluation_hook=hook),'original-run')
   prep.assert_called_once_with(input);legacy.assert_not_called();self.assertIs(factory.call_args.kwargs['device_rollout'],True)
   self.assertIs(run.call_args.args[0],plan);self.assertEqual(run.call_args.kwargs['backend'],'actual-device-backend')
  tree=ast.parse(Path(b.__file__).read_text())
  forbidden={'collect_paired','construct_from_pool','trace_step','load_actor','Session','EngineBackend','history_capture'}
  self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr in forbidden for n in ast.walk(tree)))

if __name__=='__main__':unittest.main()
