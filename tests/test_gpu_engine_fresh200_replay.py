"""Capsule/source/roster gates and complete templates, without replayed games."""
from collections import namedtuple
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import device_corpus_attestation as attestation, fresh200_capture as capture, fresh200_replay as replay

SOURCE = dict(engine_commit='1'*40, producer_commit='2'*40, entry_sha256='3'*64, runtime_sha256='4'*64)


class ReplayBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.snapshot = attestation.save_snapshot(self.root, 'literal', np.zeros((1, 2), np.int32), source=SOURCE)
        addresses = [dict(frame=8, original_ordinal=row, row=row, stream_id=51+row,
            step=8, live=True, requested=[True, True]) for row in range(200)]
        self.payload = dict(schema=capture.SCHEMA, status='CAPTURED_ONLY', passed=False, selection_error=None,
            selection_policy=capture.selection_policy(), selected=capture.select_roots(addresses), addresses=addresses,
            actual_horizon=9, chunks=[dict(start_step=start, steps=steps) for start, steps in ((0, 4), (4, 4), (8, 1))])
        self.sequence = 0

    def seal(self, payload=None, parents=None):
        self.sequence += 1
        return attestation.seal_manifest(self.root, 'capture-%d' % self.sequence,
            self.payload if payload is None else payload, parents=[self.snapshot] if parents is None else parents, source=SOURCE)

    def verify(self, reference):
        return replay.verify_capture(self.root, reference, source=SOURCE, capture=capture, attestation=attestation)

    def test_sealed_complete_source_roster_and_prefix_bounds(self):
        self.assertEqual(self.verify(self.seal()), self.payload)

    def test_altered_selected_owner_or_physical_address_fails(self):
        for field, value in (('owner', 1-self.payload['selected'][0]['owner']), ('frame', 9), ('root_index', 4)):
            changed = copy.deepcopy(self.payload);changed['selected'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'roster changed'):
                self.verify(self.seal(changed))

    def test_gap_changed_horizon_or_collection_policy_fails(self):
        cases = []
        changed = copy.deepcopy(self.payload);changed['chunks'][1]['start_step'] = 5;cases.append(changed)
        changed = copy.deepcopy(self.payload);changed['actual_horizon'] = 65;cases.append(changed)
        changed = copy.deepcopy(self.payload);changed['selection_policy']['refill'] = True;cases.append(changed)
        changed = copy.deepcopy(self.payload);changed['passed'] = True;cases.append(changed)
        changed = copy.deepcopy(self.payload);changed['status'] = 'INCOMPLETE';cases.append(changed)
        for changed in cases:
            with self.subTest(changed=repr(changed)[:80]), self.assertRaises(ValueError):self.verify(self.seal(changed))

    def test_changed_parent_source_is_fatal(self):
        other = dict(SOURCE, engine_commit='5'*40)
        snapshot = attestation.save_snapshot(self.root, 'other', np.zeros(1), source=other)
        with self.assertRaisesRegex(ValueError, 'source/schema'):self.verify(self.seal(parents=[snapshot]))

    def test_modified_parent_array_is_fatal_even_when_top_manifest_is_unchanged(self):
        reference = self.seal()
        metadata = json.loads(attestation.verify_artifact(self.root, self.snapshot))
        path = self.root/metadata['arrays']['path']
        data = bytearray(path.read_bytes());data[-1] ^= 1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'digest changed'):self.verify(reference)

    def test_runtime_templates_preserve_all_state_history_fields_and_none(self):
        State = namedtuple('State', 'hp keys optional')
        Trackers = namedtuple('Trackers', 'damage_history optional')
        Chance = namedtuple('Chance', 'draws phazing secondary sleep_talk entry_ties initial_tie effect')
        state = State(np.zeros((2, 2, 6), np.int32), np.zeros((2, 2), np.uint32), None)
        public = Trackers({'taken': np.zeros((2, 2, 2, 2), np.float32)}, None)
        template = replay.input_template(state, public, rows=2, chance_type=Chance)
        self.assertIs(template['pre_state'], state);self.assertIs(template['post_state'], state)
        self.assertIs(template['pre_public'], public);self.assertIsNone(template['pre_public'].optional)
        self.assertEqual(template['chance'].effect.shape, (2, 2, 5))
        self.assertEqual(template['chance'].initial_tie.dtype, np.dtype('int32'))
        self.assertEqual(template['actor_keys'].shape, (2, 2, 2))
        self.assertEqual(template['requested'].dtype, np.dtype('bool'))

    def test_cpu_cannot_substitute_for_actual_device_replay(self):
        fake = SimpleNamespace(devices=lambda: [SimpleNamespace(platform='cpu')])
        with patch.dict('sys.modules', {'jax': fake}), self.assertRaisesRegex(ValueError, 'actual GPU'):
            replay.replay(self.root, None, prepared=None, integration_source=SOURCE, candidate_source=SOURCE,
                capture=capture, attestation=attestation)


class JaxSnapshotOrderingTests(unittest.TestCase):
    def test_constructor_template_matches_actual_device_get_dictionary_order(self):
        import jax
        State = namedtuple('StorageState', 'hp keys optional')
        raw = State(np.zeros((2, 2, 6), np.int32), np.zeros((2, 2), np.uint32), None)
        actual = jax.device_get(dict(raw=raw, initial_tie=np.zeros(2, np.int32), streams=np.arange(2, dtype=np.int32)))
        with tempfile.TemporaryDirectory() as directory:
            reference = attestation.save_snapshot(directory, 'constructor', actual, source=SOURCE)
            wrong = dict(raw=raw, initial_tie=np.zeros(2, np.int32), streams=np.zeros(2, np.int32))
            with self.assertRaisesRegex(ValueError, 'template'):attestation.load_snapshot(directory, reference, wrong, expected_source=SOURCE)
            value = attestation.load_snapshot(directory, reference, replay.constructor_template(raw, rows=2), expected_source=SOURCE)
            np.testing.assert_array_equal(value['streams'], [0, 1]);self.assertIsNone(value['raw'].optional)

    def test_scan_axis_preserves_captured_outer_order_and_validates_every_leaf(self):
        import jax
        State = namedtuple('StorageState', 'hp keys optional')
        Trackers = namedtuple('StorageTrackers', 'damage_history optional')
        Chance = namedtuple('StorageChance', 'draws phazing secondary sleep_talk entry_ties initial_tie effect')
        state = State(np.zeros((2, 2, 6), np.int32), np.zeros((2, 2), np.uint32), None)
        public = Trackers({'taken':np.zeros((2, 2, 2, 2), np.float32)}, None)
        base = replay.input_template(state, public, rows=2, chance_type=Chance)
        # Producer created this mapping after device_get, retaining this order.
        actual = {key:jax.tree.map(lambda value:np.stack([np.asarray(value)]*4), value) for key,value in base.items()}
        with tempfile.TemporaryDirectory() as directory:
            reference = attestation.save_snapshot(directory, 'prefix', actual, source=SOURCE)
            wrong = jax.tree.map(lambda value:np.stack([np.asarray(value)]*4), base)
            with self.assertRaisesRegex(ValueError, 'template'):attestation.load_snapshot(directory, reference, wrong, expected_source=SOURCE)
            template = replay.scanned_template(base, steps=4)
            restored = attestation.load_snapshot(directory, reference, template, expected_source=SOURCE)
            capture.assert_tree_bits(actual, restored)
            template['actions'] = template['actions'].astype(np.int64)
            with self.assertRaisesRegex(ValueError, 'template'):attestation.load_snapshot(directory, reference, template, expected_source=SOURCE)


if __name__ == '__main__':unittest.main()
