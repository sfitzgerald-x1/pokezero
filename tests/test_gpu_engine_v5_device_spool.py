"""Bounded synthetic disk finalization; no actual model/game/training."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from gpu_engine import v5_device_rollout as d,v5_device_spool as s
from pokezero import v5_data
from tests import test_gpu_engine_v5_device_actor_bank as banktests
from tests import test_gpu_engine_v5_device_backend as backendtests


class DeviceSpoolTests(unittest.TestCase):
 def example(self,folder):
  plan,backend,store,snapshots,assignments,_=backendtests.DeviceBackendTests().setup_backend(folder)
  with patch('pokezero.v5_contract.source_variant_census',return_value={'source_rows_sha256':'0'*64}),\
   patch.object(backend,'_actor',side_effect=lambda snap,store:backend.actors.setdefault(snap['archive_sha256'],({'inert':np.array(0)},{}))[0]),\
   patch.object(d,'make_collector',return_value=None):
   pending=backend.prepare_batch(plan,plan.iterations[0][0],snapshots[0],assignments,store)
  raw,_=banktests.DeviceActorBankTests().run_inert(pending['bank'],bindings=pending['bindings'])
  # Preserve one complete and one capped original disposition, not fake end labels.
  raw=raw._replace(carry=raw.carry._replace(status=np.array([d.COMPLETE,d.CAPPED],np.int32)))
  games=(replace(plan.iterations[0][0].games[0],battle_id='short'),replace(plan.iterations[0][0].games[1],battle_id='much-longer-game-identity'))
  return plan,store,pending,raw,games

 def test_spool_all_raw_rows_game_only_finalization_and_original_cache_byte_identity(self):
  with tempfile.TemporaryDirectory()as folder:
   plan,store,pending,raw,games=self.example(folder);spool=s.Spool(store,pending['bank'],bank_sha256='a'*64)
   for start,stop in ((0,1),(1,3)):
    frames=d.Frames(*(None if x is None else x[start:stop]for x in raw.frames))
    spool.append(raw._replace(frames=frames))
   self.assertEqual(len(spool.records),2)
   for row in (0,1):
    game=spool.game(row)
    for original,current in zip(raw.frames,game.frames):
     if original is not None:
      self.assertEqual(current.shape[1],1);np.testing.assert_array_equal(current,original[:,row:row+1])
   def gae(values,outcome,loss,discount):return values[::-1]*.3+outcome,values*.7-outcome
   seen=[];original=d.training_batch
   def checked(rolls,*args,**kwargs):
    rolls=tuple(rolls);self.assertTrue(all(x.frames.streams.shape[1]==1 for x in rolls));seen.append(rolls[0].frames.numeric_features.shape)
    return original(rolls,*args,**kwargs)
   with patch.object(d,'_complete_gae',side_effect=gae),patch.object(d,'training_batch',side_effect=checked):
    # Completion order differs deliberately; final order must remain frame,row,owner.
    for row in (1,0):spool.finalize_game(row,games[row],bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
   self.assertEqual(seen,[(3,1,2,29,132)]*2)
   merged=spool.merge(Path(folder)/'merged')
   with patch.object(d,'_complete_gae',side_effect=gae):
    baseline=d.training_batch((raw,),pending['bindings'],actor_bank=pending['bank'],battle_ids=tuple(g.battle_id for g in games),seeds=tuple(g.seed for g in games),loss_config=plan.loss,discount=plan.discount)
   expected=v5_data.write_cache(Path(folder)/'baseline',baseline)
   self.assertEqual(merged['metadata_sha256'],expected['metadata_sha256'])
   self.assertEqual((Path(folder)/'merged'/'metadata.json').read_bytes(),(Path(folder)/'baseline'/'metadata.json').read_bytes())
   batches=list(v5_data.iter_cache_batches(Path(folder)/'merged',batch_size=2))
   self.assertEqual(sum(x.batch_size for x in batches),4)
   self.assertFalse(np.asarray(batches[0].v5_columns['outcome_target_mask'])[1])
   self.assertFalse(np.asarray(batches[0].ppo_advantage_mask)[1])
   # Existing read-only raw storage is not discarded after final cache assembly.
   for record in spool.records:
    for desc in record['frames'].values():
     if desc is not None:self.assertTrue((store.path/desc['path']).is_file())

 def test_spool_detects_owned_file_mutation_and_keeps_no_dense_history_list(self):
  with tempfile.TemporaryDirectory()as folder:
   _,store,pending,raw,_=self.example(folder);spool=s.Spool(store,pending['bank'],bank_sha256='a'*64);spool.append(raw)
   self.assertFalse(any(isinstance(value,np.ndarray)and value.ndim>1 for value in vars(spool).values()))
   desc=spool.records[0]['frames']['numeric_features'];file=store.path/desc['path'];file.chmod(0o644)
   with self.assertRaisesRegex(ValueError,'spool file changed'):spool.game(0)

 def test_phase_capacity_depends_on_one_chunk_and_one_game_not_full_dense_history(self):
  with tempfile.TemporaryDirectory()as folder:
   _,_,_,raw,_=self.example(folder)
   per_step_row=sum(x.nbytes//(x.shape[0]*x.shape[1])for x in raw.frames if x is not None)
   numeric_cat=sum(getattr(raw.frames,name).nbytes//6 for name in ('numeric_features','categorical_ids'))
   self.assertEqual(numeric_cat*2048*1024,84171292672) # exact78.390625GiB old retained inputs
   self.assertLess(per_step_row*2048*8,1024**3) # one actual8-step host chunk, all Frames fields
   self.assertLess(per_step_row*1024,64*1024**2) # one1024-step game history, all Frames fields
   # This is a logical dense-data bound, not an RSS claim. Original JSON row
   # metadata remains O(training rows); final cohort tensors use disk memmaps.

 def test_finalized_cache_metadata_cannot_be_changed_before_merge(self):
  with tempfile.TemporaryDirectory()as folder:
   plan,store,pending,raw,games=self.example(folder);spool=s.Spool(store,pending['bank'],bank_sha256='a'*64);spool.append(raw)
   with patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*0,v*0)):
    spool.finalize_game(0,games[0],bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
   file=spool.parts[0][1]/'metadata.json';file.chmod(0o644);meta=json.loads(file.read_text());meta['v5_rows']['addresses'][0]['owner']=1
   file.write_text(json.dumps(meta))
   with self.assertRaisesRegex(ValueError,'cache metadata changed'):spool.merge(Path(folder)/'must-not-publish')

 def test_early_completed_game_finalized_before_later_chunks_keeps_exact_order(self):
  with tempfile.TemporaryDirectory()as folder:
   plan,store,pending,raw,games=self.example(folder);values={}
   for name in ('requested','trainable','policy_valid','semantic_mask','hidden_mask','advanced'):
    value=getattr(raw.frames,name).copy();value[1:,0]=False;values[name]=value
   for name in ('frames','turns'):
    value=getattr(raw.frames,name).copy();value[1:,0]=1;values[name]=value
   frames=raw.frames._replace(**values);final=raw.carry._replace(frames=np.array([1,2],np.int32),terminal_frame=np.array([1,2],np.int32))
   raw=raw._replace(frames=frames,carry=final);spool=s.Spool(store,pending['bank'],bank_sha256='a'*64)
   def piece(start,stop,carry):return raw._replace(frames=d.Frames(*(None if x is None else x[start:stop]for x in frames)),carry=carry)
   first=final._replace(status=np.array([d.COMPLETE,d.RUNNING],np.int32))
   with patch.object(d,'_complete_gae',side_effect=lambda v,o,l,discount:(v*.2+o,v*.4-o)):
    spool.append(piece(0,1,first));spool.finalize_game(0,games[0],bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
    spool.append(piece(1,3,final));spool.finalize_game(1,games[1],bindings=pending['bindings'],loss_config=plan.loss,discount=plan.discount)
    expected=d.training_batch((raw,),pending['bindings'],actor_bank=pending['bank'],battle_ids=tuple(g.battle_id for g in games),seeds=tuple(g.seed for g in games),loss_config=plan.loss,discount=plan.discount)
   v5_data.write_cache(Path(folder)/'expected',expected);spool.merge(Path(folder)/'merged')
   self.assertEqual((Path(folder)/'expected'/'metadata.json').read_bytes(),(Path(folder)/'merged'/'metadata.json').read_bytes())

if __name__=='__main__':unittest.main()
