from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from gpu_engine.fresh200_boundary_gate import BoundaryAddress
from gpu_engine.fresh200_native_compare import compare_boundary, compare_pre, freeze_linked_role, freeze_pre
from gpu_engine.fresh200_native_initial import _layout
from gpu_engine.fresh200_native_ledger import Frame, NativeHistory


class NativeFullBoundaryTest(unittest.TestCase):
    def test_first_physical_difference_preserves_coordinates_and_literal_bits(self):
        from gpu_engine.fresh200_native_compare import _difference

        actual = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype='<f4')
        expected = actual.copy()
        expected[0, 0] = -0.0
        expected[1, 1] = 4.0
        proof = _difference(actual, expected, 'bytes')
        self.assertEqual(proof['coordinate'], [0, 0])
        self.assertEqual(proof['native_bits'], '00000000')
        self.assertEqual(proof['expected_bits'], '00000080')
        self.assertEqual(_difference(actual[0, 0], expected[0, 0], 'bytes')['coordinate'], [])
        self.assertEqual(_difference(actual, expected[0], 'dtype_or_shape')['expected_shape'], [2])

    def test_initial_clear_does_not_require_member_cache_before_side_install(self):
        history = NativeHistory(battle_id='b', source_registry_hash='a'*64)
        incomplete = {'sides': [None, None]}
        frame = Frame(1, None, 'Pokemon.clearVolatile', 2, {'before': incomplete},
                      {'after': incomplete}, 3)
        history._clear_move_restrictions(frame, (0, 0))
        self.assertFalse(history.disable_causes)

    def test_comparison_requires_all_73_fields_and_the_unchanged_sealed_bytes(self):
        expected_calls = []
        source = dict(engine_commit='1'*40, producer_commit='2'*40, entry_sha256='3'*64, runtime_sha256='4'*64)
        values = {name: np.zeros((1, *spec['shape'][1:]), dtype=spec['dtype']) for name, spec in _layout().items()}

        class Expected:
            def read(self, gate, *, row):
                expected_calls.append((gate.address, row))
                return {name: data[0] for name, data in values.items()}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            bind = dict(recording_sha256='5'*64, registry='6'*64, source=source)
            with self.assertRaisesRegex(ValueError, 'complete Native projection'):
                freeze_pre(root, 3, 8, ({'turn': values['turn']},), **bind)
            self.assertFalse(list(root.iterdir()))
            gate = freeze_pre(root, 3, 8, (values,), **bind)
            values['turn'][0] = 7
            self.assertEqual(compare_pre(root, gate, Expected(), row=9, source=source), {'turn': 'bytes'})
            self.assertEqual(len(expected_calls), 1)
            (root / '3-pre-8-state.npz').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                compare_pre(root, gate, Expected(), row=9, source=source)
            self.assertEqual(len(expected_calls), 1)


class NativeLinkedRoleTest(unittest.TestCase):
    def test_post_uses_the_next_pre_and_has_its_own_seal(self):
        source = dict(engine_commit='1'*40, producer_commit='2'*40, entry_sha256='3'*64, runtime_sha256='4'*64)
        values = {name: np.zeros((1, *spec['shape'][1:]), dtype=spec['dtype']) for name, spec in _layout().items()}

        class Expected:
            def read(self, gate, *, row):
                self.address = gate.address
                return {name: data[0] for name, data in values.items()}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            bind = dict(recording_sha256='5'*64, registry='6'*64, source=source)
            pre0 = freeze_pre(root, 3, 0, (values,), **bind)
            pre1 = freeze_pre(root, 3, 1, (values,), **bind)
            address = BoundaryAddress(stream=3, step=0, role='POST')
            origin = dict(pre_root=root, pre_source=source, **bind)
            with self.assertRaisesRegex(ValueError, 'source-bound frozen Native boundary'):
                freeze_linked_role(root, address, pre_reference=pre0.receipt['reference'], **origin)
            post = freeze_linked_role(root, address, pre_reference=pre1.receipt['reference'], **origin)
            self.assertNotEqual(post.receipt['reference'], pre1.receipt['reference'])
            expected = Expected()
            self.assertEqual(compare_boundary(root, post, expected, row=0, source=source), {})
            self.assertEqual(expected.address, address)
