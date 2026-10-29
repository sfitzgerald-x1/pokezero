"""Authored tensors only. Pending dependency staging is read-only and pinned."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest

import numpy as np
import pokezero


def bootstrap_dependencies():
    # Temporary pre-merge harness: never copied into the package or runtime.
    if importlib.util.find_spec('pokezero.saved_player_projection') is None:
        stage_path = os.environ.get('POKEZERO_PROJECTED_TEST_DEPENDENCY_ROOT')
        if not stage_path:
            raise RuntimeError('projected player dependencies must be merged or explicitly staged for tests')
        stage = Path(stage_path)
        pins = {'saved_player_projection.py': 'bf51f6e4e5f5e7efd1925fc453bb355938c6f9f68acddb4af2fa5dfae5dede58',
                'recorded_player_encoder_runner.py': '80b4889ef293cbd3e956bcc4baa666e8fd3e554c51c2a5c9c733c641c160ff3e'}
        for name, expected in pins.items():
            if hashlib.sha256((stage/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('reviewed dependency staging changed: ' + name)
        pokezero.__path__.append(str(stage))


bootstrap_dependencies()
from pokezero.mirror_value_probe import canonical, digest
from pokezero.projected_encoder_parity import ProjectedParityTally, projected_pair_arrays
from pokezero.recorded_encoder_parity import SHAPES
from pokezero.recorded_player_encoder_runner import ProjectedEncoderPair


def example(index=0, seat='p1', battle='authored', marker=.25):
    actual = {name: np.zeros(shape, np.float32 if name == 'numeric_features' else np.bool_
              if name.endswith('mask') else np.int32) for name, shape in SHAPES.items()}
    actual['numeric_features'][:] = marker
    actual['attention_mask'][0] = True
    actual['legal_action_mask'][:2] = True
    reference = {key: value.tolist() for key, value in actual.items()}
    reference['turn_number'] = 3
    return ProjectedEncoderPair(digest([battle, index, seat]), battle, seat, index, 3,
        digest([battle, index]), canonical({'side': {'id': seat}, 'active': [{}]}),
        canonical(reference), 'd'*64, actual)


def forward(reverse=False, value=.5):
    return dict(policy_logits=np.array([1, 2] if reverse else [2, 1], np.float32).tolist() + [0]*7,
                value=np.float32(value))


def fp32(reverse=False, value=.5):
    out = forward(reverse, value)
    out['policy_logits'] = np.array(out['policy_logits'], np.float32)
    return out


class ProjectedParityTests(unittest.TestCase):
    def test_all173_columns_padding_other_tensors_and_regenerated_only(self):
        pair = example()
        pair.actual['numeric_features'][22, 131] += np.float32(2e-6)
        pair.actual['categorical_ids'][22, 40] = 3
        pair.actual['token_type_ids'][22] = 4
        pair.actual['attention_mask'][22] = True
        pair.actual['legal_action_mask'][8] = True
        tally = ProjectedParityTally(expected_views=2)
        tally.add(pair)
        tally.add(replace(example(1), original_reference_sha256='e'*64))
        report = tally.report()
        self.assertEqual((len(report['numeric_columns']), len(report['categorical_columns'])), (132, 41))
        self.assertEqual(report['numeric_columns'][131]['mismatched_cells'], 1)
        self.assertEqual(report['numeric_columns'][0]['cells'], 46)
        self.assertEqual(report['categorical_columns'][40]['mismatched_cells'], 1)
        self.assertEqual(report['other_tensor_mismatched_cells'],
                         dict(token_type_ids=1, attention_mask=1, legal_action_mask=1))
        self.assertFalse(report['original_reference_values_compared'])
        self.assertFalse(report['model_forward_complete'])
        self.assertFalse(report['minimum10000_views_met'])
        for key in ('captured_player_wire', 'owning_stream_bytes_reconstructed', 'public_wire_qualified',
                    'fullencoder', 'full_encoder_admitted'):
            self.assertIs(report[key], False)

    def test_invalid_scope_identity_owner_request_turn_and_json_refuse_atomically(self):
        tally = ProjectedParityTally(expected_views=2)
        pair = example()
        reference = json.loads(pair.regenerated_reference_json)
        bad = [replace(pair, visibility='TRUSTED_SOURCE_CONDITIONS_PLAYER_VISIBILITY_UNQUALIFIED'),
               replace(pair, actual_captured_wire=True), replace(pair, full_encoder_admitted=0),
               replace(pair, decision_id='bad'), replace(pair, battle_id=''), replace(pair, seat='p3'),
               replace(pair, decision_round_index=True), replace(pair, turn_number=True),
               replace(pair, source_prefix_sha256='bad'), replace(pair, original_reference_sha256='bad'),
               replace(pair, own_request_json=canonical({'side': {'id': 'p2'}, 'active': [{}]})),
               replace(pair, own_request_json=canonical({'side': {'id': 'p1'}, 'wait': True})),
               replace(pair, own_request_json=b'{"active":[{}],"side":{"id":"p2","id":"p1"}}'),
               replace(pair, regenerated_reference_json=canonical(dict(reference, turn_number=4))),
               replace(pair, regenerated_reference_json=canonical(dict(reference, extra=0))),
               replace(pair, regenerated_reference_json=pair.regenerated_reference_json+b' ')]
        before = tally.report()
        for value in bad:
            with self.subTest(value=value.decision_id, seat=value.seat), self.assertRaises(ValueError):
                tally.add(value)
            self.assertEqual(tally.report(), before)

    def test_dtype_domain_nonfinite_partial_and_forward_output_refuse_atomically(self):
        tally = ProjectedParityTally(expected_views=2)
        before = tally.report()
        for mode in ('nan', 'float64', 'cat_domain', 'type_domain', 'mask_dtype', 'empty_mask', 'partial',
                     'ref_domain', 'ref_bool_turn', 'one_forward', 'output64', 'output_nan'):
            pair = example()
            kwargs = {}
            if mode == 'nan': pair.actual['numeric_features'][0, 0] = np.nan
            elif mode == 'float64': pair.actual['numeric_features'] = pair.actual['numeric_features'].astype(np.float64)
            elif mode == 'cat_domain': pair.actual['categorical_ids'][0, 0] = 916
            elif mode == 'type_domain': pair.actual['token_type_ids'][0] = -1
            elif mode == 'mask_dtype': pair.actual['legal_action_mask'] = pair.actual['legal_action_mask'].astype(np.int32)
            elif mode == 'empty_mask': pair.actual['attention_mask'][:] = False
            elif mode == 'partial': pair.actual.pop('token_type_ids')
            elif mode.startswith('ref_'):
                reference = json.loads(pair.regenerated_reference_json)
                if mode == 'ref_domain': reference['categorical_ids'][0][0] = -1
                else: reference['turn_number'] = True
                pair = replace(pair, regenerated_reference_json=canonical(reference))
            else:
                kwargs = dict(actual_forward=fp32())
                if mode != 'one_forward':
                    kwargs['reference_forward'] = fp32()
                    if mode == 'output64': kwargs['actual_forward']['policy_logits'] = np.zeros(9, np.float64)
                    else: kwargs['actual_forward']['value'] = np.float32(np.nan)
            with self.subTest(mode=mode), self.assertRaises(ValueError): tally.add(pair, **kwargs)
            self.assertEqual(tally.report(), before)

    def test_canonical_duplicates_paired_prefix_turn_and_owner_chronology(self):
        tally = ProjectedParityTally(expected_views=4)
        pair = example()
        tally.add(pair)
        before = tally.report()
        for bad in (pair, replace(pair, decision_id='b'*64),
                    replace(example(seat='p2'), source_prefix_sha256='b'*64),
                    replace(example(seat='p2'), turn_number=4,
                        regenerated_reference_json=canonical(dict(json.loads(pair.regenerated_reference_json), turn_number=4)))):
            with self.assertRaises(ValueError): tally.add(bad)
            self.assertEqual(tally.report(), before)
        tally.add(example(seat='p2'))
        tally.add(example(2))
        before = tally.report()
        with self.assertRaises(ValueError): tally.add(example(1))
        self.assertEqual(tally.report(), before)

    def test_raw_and_legal_argmax_are_both_measured_and_value_strict(self):
        pair = example()
        pair.actual['legal_action_mask'][1] = False
        tally = ProjectedParityTally(expected_views=1)
        tally.add(pair, actual_forward=fp32(True, .5001), reference_forward=fp32(True))
        report = tally.report()
        self.assertEqual(report['policy_argmax_agreement'], 1.)
        self.assertEqual(report['legal_policy_argmax_agreement'], 0.)
        self.assertGreater(report['maximum_value_difference'], 1e-4)
        self.assertFalse(report['measured_value_criterion_met'])
        pair = example()
        pair.actual['legal_action_mask'][1] = False
        reference = json.loads(pair.regenerated_reference_json)
        reference['legal_action_mask'][1] = False
        pair = replace(pair, regenerated_reference_json=canonical(reference))
        tally = ProjectedParityTally(expected_views=1)
        tally.add(pair, actual_forward=fp32(True), reference_forward=fp32())
        self.assertEqual(tally.report()['policy_argmax_agreement'], 0.)
        self.assertEqual(tally.report()['legal_policy_argmax_agreement'], 1.)

    def test_10000_complete_all_forward_exact_threshold_and_lineage_only(self):
        tally = ProjectedParityTally(expected_views=10000)
        base = example()
        for index in range(9999):
            pair = replace(base, decision_id=digest(index), decision_round_index=index,
                           source_prefix_sha256=digest(['prefix', index]))
            tally.add(pair, actual_forward=fp32(index < 10), reference_forward=fp32())
        self.assertFalse(tally.report()['projected_comparison_numeric_criteria_met'])
        def clone(value):
            out = copy.copy(value)
            for key, field in vars(value).items():
                setattr(out, key, field.append([]) if key == 'lineage' else copy.deepcopy(field))
            return out
        eleven, missing = clone(tally), clone(tally)
        pair = replace(base, decision_id=digest(9999), decision_round_index=9999)
        tally.add(pair, actual_forward=fp32(), reference_forward=fp32())
        eleven.add(pair, actual_forward=fp32(True), reference_forward=fp32())
        missing.add(pair)
        report = tally.report()
        self.assertEqual(report['policy_argmax_agreement'], .999)
        self.assertTrue(report['projected_input_numeric_criteria_met'])
        self.assertTrue(report['projected_forward_numeric_criteria_met'])
        self.assertFalse(eleven.report()['projected_forward_numeric_criteria_met'])
        self.assertFalse(missing.report()['projected_forward_numeric_criteria_met'])
        self.assertFalse(report['full_encoder_admitted'])
        with self.assertRaises(ValueError): tally.add(example(10000))
        a, b = ProjectedParityTally(expected_views=1), ProjectedParityTally(expected_views=1)
        a.add(base); b.add(replace(base, original_reference_sha256='e'*64))
        self.assertNotEqual(a.report()['lineage_sha256'], b.report()['lineage_sha256'])
        self.assertEqual(a.report()['numeric_columns'], b.report()['numeric_columns'])
