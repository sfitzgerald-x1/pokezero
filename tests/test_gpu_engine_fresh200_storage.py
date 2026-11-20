"""Fail-closed guards for value-free, independently bound capture storage."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from gpu_engine import fresh200_storage as storage


class StorageGuards(unittest.TestCase):
    def test_import_does_not_import_engine_or_random_runtime(self):
        source = '''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('jax','torch','pokezero'):
   raise AssertionError('runtime imported: '+fullname)
sys.meta_path.insert(0,Deny())
from gpu_engine import fresh200_storage
fresh200_storage._schema()
assert not any(n.split('.')[0] in ('jax','torch','pokezero') for n in sys.modules)
'''
        result = subprocess.run([sys.executable, '-B', '-c', source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_storage_leaf_cannot_supply_or_coerce_battle_values(self):
        spec = storage.ArraySpec((256, 2, 6), np.int32)
        with self.assertRaisesRegex(TypeError, 'no battle values'):np.asarray(spec)
        for shape, dtype in (((True,), 'int32'), ((-1,), 'int32'), ((2,), 'object')):
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ValueError):
                storage.ArraySpec(shape, np.dtype(dtype))
        with self.assertRaisesRegex(TypeError, 'trusted specification'):storage.scan_spec(np.zeros(2), 4)

    def test_changed_schema_is_rejected_before_shapes_can_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);document = storage._schema()
            document['records']['gpu_engine.state.BattleState'][0]['spec']['dtype'] = 'float32'
            (root/'fresh200_storage_schema.json').write_text(json.dumps(document))
            with patch.object(storage, '__file__', str(root/'fresh200_storage.py')):
                with self.assertRaisesRegex(ValueError, 'schema SHA'):storage._schema()

    def test_source_declarations_are_parsed_without_execution_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'gpu_engine/state.py';path.parent.mkdir()
            sentinel = root/'executed'
            source = "from pathlib import Path\nPath("+repr(str(sentinel))+").write_text('bad')\nclass BattleState:\n hp: UndefinedAnnotation\n"
            path.write_text(source)
            schema = dict(source_files={'gpu_engine/state.py':hashlib.sha256(source.encode()).hexdigest()},
                records={'gpu_engine.state.BattleState':[dict(name='hp')]})
            storage.verify_source(root, schema);self.assertFalse(sentinel.exists())
            path.write_text(source+'# changed\n')
            with self.assertRaisesRegex(ValueError, 'source differs'):storage.verify_source(root, schema)

    def test_pool_dimensions_bind_metadata_bytes_before_using_namespace_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pool.npz'
            blob = json.dumps(dict(vocabulary=dict(abilities=['']+['a']*32, items=['']+['i']*64))).encode()
            np.savez_compressed(path, metadata_utf8=np.frombuffer(blob, np.uint8))
            digest = storage.sha256_file(path);result = storage.pool_dimensions(path, digest)
            self.assertEqual((result['ability_words'], result['item_words']), (2, 3))
            self.assertEqual(result['metadata_sha256'], hashlib.sha256(blob).hexdigest())
            with self.assertRaisesRegex(ValueError, 'pool SHA'):storage.pool_dimensions(path, '0'*64)
            with self.assertRaisesRegex(ValueError, 'frozen original'):
                storage.SealedStorage(directory, path, expected_pool_sha256=digest)

    def test_ambiguous_or_object_pool_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pool.npz';array = io.BytesIO()
            np.save(array, np.zeros(1, np.uint8));value = array.getvalue()
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('metadata_utf8.npy', value)
                # A second spelling of the same member cannot select a namespace.
                with self.assertWarns(UserWarning):archive.writestr('metadata_utf8.npy', value)
            with self.assertRaisesRegex(ValueError, 'one bounded'):storage.pool_dimensions(path, storage.sha256_file(path))
            np.savez_compressed(path, metadata_utf8=np.array([dict(vocabulary={})], object))
            with self.assertRaises(ValueError):storage.pool_dimensions(path, storage.sha256_file(path))


if __name__ == '__main__':unittest.main()
