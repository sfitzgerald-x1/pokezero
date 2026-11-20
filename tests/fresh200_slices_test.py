"""Synthetic guards only; these fixtures are never corpus/reference evidence."""
from collections import namedtuple
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from gpu_engine import device_corpus_attestation as attestation
from gpu_engine import fresh200_slices as slices
from gpu_engine import fresh200_storage as storage
from gpu_engine.fresh200_boundary_gate import BoundaryAddress, BoundaryGate

SOURCE = dict(engine_commit=storage.INTEGRATION, producer_commit='1'*40,
    entry_sha256='2'*64, runtime_sha256='3'*64)


def layout():
    # Synthetic unit-fixture layout. Production SealedStorage verifies all
    # original source files and the resident pool before constructing layouts.
    value = storage.SealedStorage.__new__(storage.SealedStorage)
    value.rows = 3;value.schema = storage._schema();value.dimensions = dict(B=3,
        V=len(value.schema['volatile_names']), A=1, I=1);value.types = {}
    for identity, fields in value.schema['records'].items():
        module, name = identity.rsplit('.', 1)
        value.types[identity] = namedtuple(name, [x['name'] for x in fields], module=module)
    return value


def materialize(spec):
    if isinstance(spec, storage.ArraySpec):
        n = int(np.prod(spec.shape));data = np.arange(n, dtype=np.int64) % 17
        return data.astype(spec.dtype).reshape(spec.shape)
    if isinstance(spec, tuple) and hasattr(spec, '_fields'):
        return type(spec)(*(materialize(x) for x in spec))
    if isinstance(spec, dict):return {k:materialize(v) for k, v in spec.items()}
    if spec is None:return None
    raise AssertionError('unsupported synthetic fixture')


def seal_gate(address):
    gate = BoundaryGate(address)
    # A mock sealer tests the ordering API only, not production attestation.
    gate.freeze({}, lambda a, p:dict(address=a, field_count=73, source_registry_hash='4'*64,
        reference=dict(path='synthetic-native-test-only.npz', bytes=1, sha256='5'*64)))
    return gate


def reference(root, path):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(root)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class SliceGuards(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name);self.storage = layout()

    def simple(self, name='simple', array=None):
        if array is None:array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        template = dict(values=storage.ArraySpec(array.shape, array.dtype))
        ref = attestation.save_snapshot(self.root, name, dict(values=array), source=SOURCE)
        return ref, template, array

    def replay(self):
        tree = materialize(self.storage.replay_inputs(2));tree['streams'][:] = [101, 202, 303]
        ref = attestation.save_snapshot(self.root, 'replay', tree, source=SOURCE)
        return slices.SealedInputs(self.root, ref, self.storage, source=SOURCE,
            kind='replay', start_step=4, steps=2), tree

    def rewrite(self, ref, change):
        path = self.root/ref['path'];document = json.loads(path.read_bytes());change(document)
        path.write_text(json.dumps(document));return reference(self.root, path)

    def test_import_does_not_import_engine_or_random_runtime(self):
        script = '''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('jax','torch','pokezero'):
   raise AssertionError('runtime imported: '+fullname)
sys.meta_path.insert(0,Deny())
from gpu_engine import fresh200_slices
assert not any(n.split('.')[0] in ('jax','torch','pokezero') for n in sys.modules)
'''
        p = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)

    def test_replay_selectors_never_decode_expected_state_or_trackers(self):
        reader, tree = self.replay()
        with patch.object(reader._snapshot, '_read', wraps=reader._snapshot._read) as read:
            actual = reader.replay_inputs(1, 2)
        self.assertEqual(set(actual), {'actions','requested','streams','frames','status','actor_keys','chance'})
        self.assertTrue(all(call.args[0][0] in actual for call in read.call_args_list))
        self.assertTrue(all(call.args[1] == (1, 2) for call in read.call_args_list))
        self.assertEqual(set(actual['chance']), set(tree['chance']._fields))
        np.testing.assert_array_equal(actual['actions'], tree['actions'][1, 2])
        for name, value in actual['chance'].items():
            self.assertEqual(value.tobytes(), getattr(tree['chance'], name)[1, 2].tobytes())
        with self.assertRaisesRegex(ValueError, 'constructor'):reader.constructor(0)

    def test_constructor_decodes_only_literal_constructor_allowlist(self):
        tree = materialize(self.storage.constructor_input());tree['streams'][:] = [101, 202, 303]
        ref = attestation.save_snapshot(self.root, 'constructor', tree, source=SOURCE)
        reader = slices.SealedInputs(self.root, ref, self.storage, source=SOURCE, kind='constructor')
        with patch.object(reader._snapshot, '_read', wraps=reader._snapshot._read) as read:
            actual = reader.constructor(1)
        self.assertEqual(set(actual['raw']), set(slices.CONSTRUCTOR_FIELDS))
        self.assertEqual(int(actual['streams']), 202)
        self.assertNotIn('pending_residual', actual['raw'])
        self.assertEqual({x.args[0] for x in read.call_args_list},
            {('raw', n) for n in slices.CONSTRUCTOR_FIELDS}|{('initial_tie',),('streams',)})
        initialized = materialize(self.storage.initialized())
        initialized_ref = attestation.save_snapshot(self.root, 'initialized', initialized, source=SOURCE)
        expected = slices.ExpectedState(reader, initialized_reference=initialized_ref)
        gate = seal_gate(BoundaryAddress(202, 0, 'INITIALIZED'))
        result = expected.read(gate, row=1)
        self.assertEqual(len(result), 73)
        for name, value in result.items():self.assertEqual(value.tobytes(), getattr(initialized.state, name)[1].tobytes())

    def test_expected_fields_are_read_only_after_matching_native_freeze(self):
        reader, tree = self.replay();expected = slices.ExpectedState(reader)
        gate = BoundaryGate(BoundaryAddress(303, 5, 'POST'))
        with patch.object(reader._snapshot, '_read', wraps=reader._snapshot._read) as read:
            with self.assertRaisesRegex(ValueError, 'before Native'):expected.read(gate, row=2)
            self.assertEqual(read.call_count, 0)
            result = expected.read(seal_gate(gate.address), row=2)
        self.assertEqual(read.call_count, 74)
        self.assertEqual(read.call_args_list[0].args[0], ('streams',))
        self.assertEqual(len(result), 73)
        for name, value in result.items():
            self.assertEqual(value.tobytes(), getattr(tree['post_state'], name)[1, 2].tobytes())
            self.assertFalse(value.flags.writeable)

    def test_wrong_stream_role_or_global_step_never_decodes_physical_expected(self):
        reader, _ = self.replay();expected = slices.ExpectedState(reader)
        for address, row in [(BoundaryAddress(101, 4, 'PRE'), 1),
                (BoundaryAddress(101, 3, 'PRE'), 0), (BoundaryAddress(101, 6, 'POST'), 0),
                (BoundaryAddress(101, 4, 'INITIALIZED'), 0)]:
            with self.subTest(address=address), patch.object(reader._snapshot, '_read', wraps=reader._snapshot._read) as read:
                with self.assertRaises(ValueError):expected.read(seal_gate(address), row=row)
                self.assertTrue(all(x.args[0] == ('streams',) for x in read.call_args_list))

    def test_descriptor_layout_addresses_and_duplicate_metadata_fail_closed(self):
        for mode in ('dtype', 'address', 'hash'):
            ref, template, _ = self.simple(mode)
            def change(doc):
                leaf = doc['tree']['children'][0]
                if mode == 'dtype':leaf['dtype'] = '<i4'
                elif mode == 'address':leaf['array'] = 'a000001'
                else:leaf['value_sha256'] = 'bad'
            ref = self.rewrite(ref, change)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                slices._Snapshot(self.root, ref, template, source=SOURCE)
        ref, template, _ = self.simple('duplicate-json');path = self.root/ref['path']
        blob = path.read_text();path.write_text('{"kind":"snapshot",'+blob[1:]);ref = reference(self.root, path)
        with self.assertRaisesRegex(ValueError, 'duplicate snapshot'):
            slices._Snapshot(self.root, ref, template, source=SOURCE)

    def test_npz_roster_and_fortran_object_wrong_shape_headers_fail_before_decode(self):
        for mode in ('duplicate', 'extra', 'fortran', 'object', 'shape'):
            ref, template, array = self.simple(mode);document = json.loads((self.root/ref['path']).read_bytes())
            path = self.root/document['arrays']['path'];buffer = io.BytesIO()
            value = np.asfortranarray(array) if mode == 'fortran' else (array.astype(object) if mode == 'object'
                else array.reshape(3, 2, 4) if mode == 'shape' else array)
            np.save(buffer, value, allow_pickle=mode == 'object')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('a000000.npy', buffer.getvalue())
                    if mode in ('duplicate', 'extra'):
                        archive.writestr('a000000.npy' if mode == 'duplicate' else 'unexpected.npy', buffer.getvalue())
            ref = self.rewrite(ref, lambda doc:doc.update(arrays=reference(self.root, path)))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                slices._Snapshot(self.root, ref, template, source=SOURCE)

    def test_selected_span_preserves_float_bits_and_validates_all_coordinates(self):
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        array.view(np.uint32)[1, 2, :2] = [0x80000000, 0x7fc00001]
        ref, template, _ = self.simple(array=array);reader = slices._Snapshot(self.root, ref, template, source=SOURCE)
        value = reader._read(('values',), (1, 2))
        self.assertEqual(value.tobytes(), array[1, 2].tobytes());self.assertFalse(value.flags.writeable)
        self.assertEqual(reader._read(('values',), (1, 2, 1)).shape, ())
        for coordinates in ((True,), (-1,), (2,), (0.0,), (1, 3), (1, 2, 4), [1]):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):reader._read(('values',), coordinates)

    def test_full_leaf_hash_rejects_corruption_outside_the_selected_span(self):
        ref, template, array = self.simple();document = json.loads((self.root/ref['path']).read_bytes())
        path = self.root/document['arrays']['path'];array[0, 0, 0] = -99
        np.savez_compressed(path, a000000=array)
        ref = self.rewrite(ref, lambda doc:doc.update(arrays=reference(self.root, path)))
        reader = slices._Snapshot(self.root, ref, template, source=SOURCE)
        with self.assertRaisesRegex(ValueError, 'full leaf bits'):reader._read(('values',), (1, 2))

    def test_whole_artifact_sha_and_source_are_verified_before_value_access(self):
        ref, template, _ = self.simple();document = json.loads((self.root/ref['path']).read_bytes())
        path = self.root/document['arrays']['path'];blob = bytearray(path.read_bytes());blob[-1] ^= 1;path.write_bytes(blob)
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            slices._Snapshot(self.root, ref, template, source=SOURCE)
        bad_source = dict(SOURCE, engine_commit='6'*40)
        with self.assertRaisesRegex(ValueError, 'frozen f9'):
            slices._Snapshot(self.root, ref, template, source=bad_source)


if __name__ == '__main__':unittest.main()
