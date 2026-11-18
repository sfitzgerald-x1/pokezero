"""Integrity/coverage checks for transport; no game, model, or device execution."""
from collections import namedtuple
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gpu_engine import device_corpus_attestation as attestation

State = namedtuple('State', 'hp optional nested')
SOURCE = dict(engine_commit='1' * 40, producer_commit='2' * 40, entry_sha256='3' * 64, runtime_sha256='4' * 64)


class AttestationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state = State(np.array([[714, 0]], np.int32), None,
            {'pair': np.array([0x3f000000, 0x80000000], np.uint32).view(np.float32),
             'keys': (np.array([0, 2**32-1], np.uint32), [True, 17])})

    def save(self):
        return attestation.save_snapshot(self.root, 'state-0', self.state, source=SOURCE)

    def test_complete_roundtrip_preserves_types_none_and_all_bits(self):
        restored = attestation.load_snapshot(self.root, self.save(), self.state, expected_source=SOURCE)
        self.assertIs(type(restored), State);self.assertIsNone(restored.optional)
        self.assertIs(type(restored.nested['keys']), tuple)
        self.assertIs(type(restored.nested['keys'][1]), list)
        np.testing.assert_array_equal(restored.hp, self.state.hp)
        np.testing.assert_array_equal(restored.nested['pair'].view(np.uint32), self.state.nested['pair'].view(np.uint32))
        np.testing.assert_array_equal(restored.nested['keys'][0], self.state.nested['keys'][0])
        self.assertEqual(restored.nested['keys'][1], [True, 17])
        restored.hp[0, 0] = 1
        self.assertEqual(self.state.hp[0, 0], 714)

    def test_missing_changed_or_extra_template_fields_are_refused(self):
        reference = self.save()
        for changed in (self.state._replace(optional=np.zeros(1)),
                        self.state._replace(hp=self.state.hp.astype(np.int64)),
                        self.state._replace(hp=self.state.hp[:, :1]),
                        self.state._replace(nested={'pair': self.state.nested['pair']})):
            with self.subTest(tree=repr(changed)[:80]), self.assertRaisesRegex(ValueError, 'template'):
                attestation.load_snapshot(self.root, reference, changed, expected_source=SOURCE)

    def test_source_mismatch_and_unbound_producer_are_refused(self):
        reference = self.save();changed = dict(SOURCE, producer_commit='5' * 40)
        with self.assertRaisesRegex(ValueError, 'source binding'):
            attestation.load_snapshot(self.root, reference, self.state, expected_source=changed)
        with self.assertRaisesRegex(ValueError, 'source binding'):
            attestation.save_snapshot(self.root, 'missing', self.state, source={})

    def test_snapshot_bytes_cannot_change_under_a_sealed_reference(self):
        reference = self.save();metadata = json.loads(attestation.verify_artifact(self.root, reference))
        path = self.root / metadata['arrays']['path']
        data = bytearray(path.read_bytes());data[-1] ^= 1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            attestation.load_snapshot(self.root, reference, self.state, expected_source=SOURCE)

    def test_manifest_rejects_missing_duplicate_output_and_changed_parents(self):
        parent = self.save()
        reference = attestation.seal_manifest(self.root, 'selected', {'roots': 1}, parents=iter([parent]), source=SOURCE)
        value = json.loads(attestation.verify_artifact(self.root, reference))
        self.assertEqual(value['parents'], [parent]);self.assertEqual(value['payload'], {'roots': 1})
        with self.assertRaises(FileExistsError):
            attestation.seal_manifest(self.root, 'selected', {}, parents=[parent], source=SOURCE)
        wrong = copy.deepcopy(parent);wrong['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            attestation.seal_manifest(self.root, 'invalid', {}, parents=[wrong], source=SOURCE)
        self.assertFalse((self.root / 'invalid.json').exists())

    def test_even_resealed_missing_or_extra_npz_leaves_are_refused(self):
        reference = self.save();original = json.loads(attestation.verify_artifact(self.root, reference))
        with np.load(self.root / original['arrays']['path'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for index, defect in enumerate(('missing', 'extra')):
            changed = dict(arrays)
            if defect == 'missing':changed.pop(next(iter(changed)))
            else:changed['unexpected'] = np.zeros(1)
            data = self.root / ('changed-%d.npz' % index)
            np.savez_compressed(data, **changed)
            metadata = copy.deepcopy(original)
            metadata['arrays'] = dict(path=data.name, bytes=data.stat().st_size, sha256=attestation.sha256_file(data))
            path = self.root / ('changed-%d.json' % index);path.write_text(json.dumps(metadata))
            sealed = dict(path=path.name, bytes=path.stat().st_size, sha256=attestation.sha256_file(path))
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, 'missing or extra'):
                attestation.load_snapshot(self.root, sealed, self.state, expected_source=SOURCE)

    def test_paths_and_nonprimitive_arrays_are_refused(self):
        reference = dict(path='../escape.npz', sha256='0' * 64, bytes=0)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            attestation.verify_artifact(self.root, reference)
        with self.assertRaisesRegex(ValueError, 'artifact name'):
            attestation.save_snapshot(self.root, '../escape', self.state, source=SOURCE)
        with self.assertRaisesRegex(TypeError, 'dtype'):
            attestation.save_snapshot(self.root, 'object', np.array([object()], object), source=SOURCE)

    def test_existing_snapshot_is_preserved(self):
        reference = self.save()
        with self.assertRaises(FileExistsError):self.save()
        self.assertEqual(attestation.sha256_file(self.root / reference['path']), reference['sha256'])


if __name__ == '__main__':unittest.main()
