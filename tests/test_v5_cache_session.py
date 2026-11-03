"""Finite cache order and mutation controls; no Torch, model or game needed."""
from dataclasses import fields,replace
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from pokezero import dataset, v5_data as data, v5_training as training
import test_v5_data as fixture


def normalized(value):
    if isinstance(value,np.ndarray):return normalized(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if hasattr(value,'items'):return {key:normalized(x) for key,x in value.items()}
    if isinstance(value,(list,tuple)):return [normalized(x) for x in value]
    return value


def payload(batch):
    return {field.name:normalized(getattr(batch,field.name)) for field in fields(batch)}


def cache(path,start,count):
    rows=[fixture.row(start+i,status='CAPPED' if i==1 else 'COMPLETE',search=i%2==0,
        privileged=i%3==0) for i in range(count)]
    batch=data.attach_batch(fixture.base_batch(rows),rows)
    # This fixture has no later public bootstrap capture for capped rows.
    batch=replace(batch,ppo_value_target_mask=tuple(not bool(x)for x in batch.terminal_capped))
    data.write_cache(path,batch)
    return batch


def runtime(epochs=2):
    return training.Runtime(None,None,SimpleNamespace(batch_size=2,epochs=epochs,
        learning_rate_schedule='constant'),None,None,{}, {})


def count_update(owner,batch):
    owner.steps+=1;owner.rows+=batch.batch_size
    return dict(addresses=normalized(batch.v5_columns['addresses']),rows=batch.batch_size)


class CacheSessionTests(unittest.TestCase):
    def test_resident_validates_once_and_private_slices_keep_all_bytes_and_guards(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);original=cache(root/'cache',0,5)
            with patch.object(data,'validate_batch',wraps=data.validate_batch)as validate,\
                    patch.object(training,'validate_bootstraps',wraps=training.validate_bootstraps)as bootstraps:
                reader=data.VerifiedCache(root/'cache',resident_bytes=10**8)
                pieces=list(reader.batches(batch_size=2))*2
                self.assertEqual((validate.call_count,bootstraps.call_count),(1,1))
            expected=[dataset.slice_training_batch(original,i,min(i+2,5))for i in range(0,5,2)]*2
            self.assertEqual([payload(x)for x in pieces],[payload(x)for x in expected])
            batch=pieces[0];self.assertTrue(data._validated_resident_slice(batch))
            with self.assertRaises(ValueError):batch.numeric_features.flags.writeable=True
            with self.assertRaises(TypeError):batch.v5_columns['addresses'][0]['owner']=1
            with self.assertRaises(TypeError):batch.v5_columns['label_vocabulary']['moves']=()
            # Cache output from a valid private slice has the original bytes,
            # including JSON metadata; freezing is not a new serialization.
            data.write_cache(root/'actual',batch);data.write_cache(root/'expected',expected[0])
            for file in (root/'actual').iterdir():self.assertEqual(file.read_bytes(),(root/'expected'/file.name).read_bytes())
            with patch.object(data,'validate_batch')as validate,patch.object(training,'validate_bootstraps')as bootstraps,\
                    patch.object(training,'require_v5_bindings'),patch.object(training,'_validate_optimizer'),\
                    patch.object(training,'_torch',side_effect=RuntimeError('original tensor path')):
                with self.assertRaisesRegex(RuntimeError,'original tensor path'):training._update(runtime(),batch)
                validate.assert_not_called();bootstraps.assert_not_called()
                arbitrary=replace(batch)  # Public constructor still validates.
                validate.reset_mock();bootstraps.reset_mock()
                with self.assertRaisesRegex(RuntimeError,'original tensor path'):training._update(runtime(),arbitrary)
                self.assertEqual((validate.call_count,bootstraps.call_count),(1,1))
            object.__setattr__(batch,'returns',np.array(batch.returns,copy=True))
            with self.assertRaisesRegex(ValueError,'resident slice'):data._validated_resident_slice(batch)

    def test_resident_missing_bootstrap_refuses_at_load_and_policy_is_scoped(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);row=fixture.row(status='CAPPED')
            invalid=data.attach_batch(fixture.base_batch([row]),[row]);data.write_cache(root/'invalid',invalid)
            with self.assertRaisesRegex(ValueError,'bootstrap'):data.VerifiedCache(root/'invalid',resident_bytes=10**8)
            cache(root/'valid',0,3);owner=runtime()
            with training.resident_cache_policy(10**8),patch.object(training,'update',side_effect=count_update):
                training.train_caches(owner,[root/'valid'],max_updates=1)
                self.assertIsNotNone(training._CACHE_READERS[owner].resident)
                with self.assertRaisesRegex(ValueError,'policy differs'):
                    training.train_caches(owner,[root/'valid'],max_updates=1,resident_cache_bytes=99)
            self.assertIsNone(training._RESIDENT_POLICY.get())

    def test_replaced_source_and_unregistered_reader_cannot_mint_trusted_slices(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'cache';cache(path,0,3)
            reader=data.VerifiedCache(path,resident_bytes=10**8)
            source=reader._batch
            invalid=replace(source,ppo_value_target_mask=np.frombuffer(np.ones(3,bool).tobytes(),dtype=bool))
            with self.assertRaisesRegex(ValueError,'bootstrap'):training.validate_bootstraps(invalid)
            reader._batch=invalid
            with self.assertRaisesRegex(ValueError,'resident source'):next(reader.batches(batch_size=2))
            reader._batch=source
            forged=data.VerifiedCache.__new__(data.VerifiedCache);forged._resident=True;forged._batch=source
            with self.assertRaisesRegex(ValueError,'registered validated'):data._resident_slice_batch(forged,0,2)
            object.__setattr__(source,'returns',np.frombuffer(np.zeros(3).tobytes(),dtype=np.float64))
            with self.assertRaisesRegex(ValueError,'resident source'):next(reader.batches(batch_size=2))
    def test_resident_iteration_keeps_exact_order_and_reuses_every_verified_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[root/'one',root/'two']
            originals=[cache(paths[0],0,5),cache(paths[1],5,3)]
            owner=runtime(3);mapped=runtime(3)
            with patch.object(training,'update',side_effect=count_update):
                expected=training.train_caches(mapped,paths)
            with patch.object(data,'_file_sha',wraps=data._file_sha) as digest,\
                    patch.object(training,'update',side_effect=count_update):
                updates=[]
                while owner.progress is None or not owner.progress['complete']:
                    updates+=training.train_caches(owner,paths,max_updates=1,
                        resident_cache_bytes=10**8)['updates']
                arrays=sum(len(json.loads((p/'metadata.json').read_text())['arrays'])for p in paths)
                self.assertEqual(digest.call_count,arrays)
            self.assertEqual(updates,expected['updates'])
            self.assertEqual((owner.steps,owner.rows,owner.progress),(mapped.steps,mapped.rows,mapped.progress))
            reader=training._CACHE_READERS[owner]
            self.assertEqual(len(reader.resident),2)
            for resident,original in zip(reader.resident,originals):
                self.assertEqual(payload(resident._batch),payload(original))
                array=resident._batch.numeric_features
                self.assertNotIsInstance(array,np.memmap)
                self.assertFalse(array.flags.writeable)

    def test_resident_budget_refuses_before_array_open_and_actual_headers_enforce_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache';cache(path,0,3)
            for budget in (True,0,-1,1):
                with self.subTest(budget=budget),patch.object(data,'VerifiedCache') as opened:
                    with self.assertRaisesRegex(ValueError,'byte budget'):
                        training.train_caches(runtime(),[path],max_updates=0,resident_cache_bytes=budget)
                    opened.assert_not_called()
            with self.assertRaisesRegex(ValueError,'byte budget'):
                data.VerifiedCache(path,resident_bytes=1)

    def test_resident_resume_keeps_original_cursor_and_mutated_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache';cache(path,0,5);owner=runtime(2)
            with patch.object(training,'update',side_effect=count_update):
                training.train_caches(owner,[path],max_updates=2,resident_cache_bytes=10**8)
                resumed=runtime(2);resumed.progress=json.loads(json.dumps(owner.progress))
                resumed.steps=owner.steps;resumed.rows=owner.rows
                a=training.train_caches(owner,[path],resident_cache_bytes=10**8)
                b=training.train_caches(resumed,[path],resident_cache_bytes=10**8)
                self.assertEqual(a,b)
            other=runtime(2)
            with patch.object(training,'update',side_effect=count_update) as update:
                training.train_caches(other,[path],max_updates=1,resident_cache_bytes=10**8)
                target=path/'base__returns.npy';target.write_bytes(target.read_bytes()+b'x')
                with self.assertRaisesRegex(ValueError,'files changed'):
                    training.train_caches(other,[path],max_updates=1,resident_cache_bytes=10**8)
                self.assertEqual(update.call_count,1)

    def test_direct_cursor_preserves_every_base_and_sidecar_column_without_slicing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache';original=cache(path,0,9)
            reader=data.VerifiedCache(path)
            for start in range(6):
                with self.subTest(start=start),patch.object(dataset,'slice_training_batch',
                        wraps=dataset.slice_training_batch) as sliced:
                    actual=list(reader.batches(batch_size=2,start_batch=start))
                    self.assertEqual(sliced.call_count,5-start)
                expected=[dataset.slice_training_batch(original,i,min(i+2,9)) for i in range(start*2,9,2)]
                self.assertEqual([payload(x) for x in actual],[payload(x) for x in expected])
            for cursor in (-1,6,True):
                with self.assertRaisesRegex(ValueError,'cursor'):list(reader.batches(batch_size=2,start_batch=cursor))

    def test_each_new_open_keeps_original_digest_checks_and_reuse_refuses_changes(self):
        for change in ('array_bytes','metadata_bytes','array_replace','array_remove','same_bytes_rewrite'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'cache';cache(path,0,3);reader=data.VerifiedCache(path)
                target=path/'base__returns.npy';blob=target.read_bytes()
                if change=='array_bytes':target.write_bytes(blob+b'x')
                elif change=='metadata_bytes':(path/'metadata.json').write_text('{}')
                elif change=='array_replace':
                    other=path/'replacement';other.write_bytes(blob);other.replace(target)
                elif change=='array_remove':target.unlink()
                else:target.write_bytes(blob)
                with self.assertRaisesRegex(ValueError,'files changed'):list(reader.batches(batch_size=2))
                if change=='array_bytes':
                    with self.assertRaisesRegex(ValueError,'array bytes changed'):data.VerifiedCache(path)

    def test_chunked_updates_exactly_equal_full_order_and_new_runtime_resumes_next_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[root/'one',root/'two']
            originals=[cache(paths[0],0,5),cache(paths[1],5,3)]
            expected=[normalized(dataset.slice_training_batch(b,i,min(i+2,b.batch_size)).v5_columns['addresses'])
                for _ in range(3) for b in originals for i in range(0,b.batch_size,2)]
            full=runtime(3)
            with patch.object(training,'update',side_effect=count_update):
                complete=training.train_caches(full,paths)
                owner=runtime(3);updates=[]
                for _ in range(4):updates+=training.train_caches(owner,paths,max_updates=1)['updates']
                # Only the literal checkpoint cursor/row/step data survive a
                # process restart. No session mapping or metadata is restored.
                resumed=runtime(3);resumed.steps=owner.steps;resumed.rows=owner.rows
                resumed.progress=json.loads(json.dumps(owner.progress))
                while not resumed.progress['complete']:
                    updates+=training.train_caches(resumed,paths,max_updates=2)['updates']
                self.assertEqual(updates,complete['updates'])
                self.assertEqual([x['addresses'] for x in updates],expected)
                self.assertEqual((resumed.steps,resumed.rows,resumed.progress),(full.steps,full.rows,full.progress))

    def test_one_manifest_per_runtime_and_one_active_cache_digest_per_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache';cache(path,0,9);owner=runtime(2)
            with patch.object(training,'_cache_manifest',wraps=training._cache_manifest) as manifest,\
                    patch.object(data,'_file_sha',wraps=data._file_sha) as digest,\
                    patch.object(dataset,'slice_training_batch',wraps=dataset.slice_training_batch) as sliced,\
                    patch.object(training,'update',side_effect=count_update):
                for _ in range(10):training.train_caches(owner,[path],max_updates=1)
                count=len(json.loads((path/'metadata.json').read_text())['arrays'])
                self.assertEqual((manifest.call_count,digest.call_count,sliced.call_count),(1,count,10))
            reader=training._CACHE_READERS[owner];reference=weakref.ref(owner)
            del owner;gc.collect();self.assertIsNone(reference())
            self.assertNotIn(reader,list(training._CACHE_READERS.values()))

    def test_active_eviction_revalidates_bytes_and_metadata_roster_cannot_alias_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[root/'one',root/'two']
            cache(paths[0],0,2);cache(paths[1],2,2);owner=runtime(2)
            with patch.object(training,'update',side_effect=count_update):
                training.train_caches(owner,paths,max_updates=1)
                first=weakref.ref(training._CACHE_READERS[owner].active)
                training.train_caches(owner,paths,max_updates=1)
                gc.collect();self.assertIsNone(first())
                target=paths[0]/'base__returns.npy';target.write_bytes(target.read_bytes()+b'x')
                with self.assertRaisesRegex(ValueError,'array bytes changed'):training.train_caches(owner,paths,max_updates=1)
            other=runtime(1)
            with patch.object(training,'update',side_effect=count_update):
                training.train_caches(other,[paths[1]],max_updates=0)
                other.progress['manifest'][0]['examples']+=1
                with self.assertRaisesRegex(ValueError,'ordered v5 caches'):training.train_caches(other,[paths[1]],max_updates=0)

    def test_metadata_change_between_chunks_fails_before_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache';cache(path,0,5);owner=runtime(1)
            with patch.object(training,'update',side_effect=count_update) as update:
                training.train_caches(owner,[path],max_updates=1)
                metadata=path/'metadata.json';metadata.write_text(metadata.read_text())
                with self.assertRaisesRegex(ValueError,'files changed'):training.train_caches(owner,[path],max_updates=1)
                self.assertEqual(update.call_count,1)


if __name__=='__main__':unittest.main()
