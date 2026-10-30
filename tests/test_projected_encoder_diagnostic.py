"""Authored source fixtures; never open real model/data or start a simulator."""
from contextlib import ExitStack
import json
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import test_projected_encoder_collection as collection_fixtures
from test_projected_encoder_collection import FakeAdapter, FakeProjector, FakeEnvironment, fake_feed
from test_projected_encoder_parity import example, fp32
from pokezero.projected_encoder_diagnostic import (
    compare_diagnostic_collection, DiagnosticTally, refusal)


class DiagnosticsTests(unittest.TestCase):
    def run_fixture(self, *, adapter_type=FakeAdapter, feed=fake_feed, forward=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = collection_fixtures.ProjectedCollectionTests()
            kwargs = fixture.prepare(root)
            adapters = {seat: adapter_type(seat) for seat in ('p1', 'p2')}
            def factory(config, *, battle_id):
                return {seat: FakeEnvironment(battle_id, seat) for seat in ('p1', 'p2')}
            module = 'pokezero.projected_encoder_diagnostic.'
            import pokezero.projected_encoder_diagnostic as diagnostic
            original = diagnostic.compare_diagnostic_battle
            def battle(*a, **k):
                return original(*a, **k, projector=FakeProjector(battle_id=k['battle']['battle_id']),
                                reference_feed=feed)
            with ExitStack() as stack:
                for target in ('subprocess.run', 'subprocess.Popen', 'numpy.load'):
                    stack.enter_context(patch(target, side_effect=AssertionError('actual runtime forbidden')))
                stack.enter_context(patch(module+'new_player_reference_environments', side_effect=factory))
                stack.enter_context(patch(module+'compare_diagnostic_battle', side_effect=battle))
                result = compare_diagnostic_collection(root, **kwargs, adapters=adapters,
                    reference_config={}, with_forward=forward)
            return result, adapters

    def test_complete_positive_all_columns_and_paired_forward(self):
        result, adapters = self.run_fixture()
        m = result['metrics']
        self.assertEqual(m['processed_views'], m['expected_views'])
        self.assertEqual(m['views'], m['expected_views'])
        self.assertEqual(m['refused_input_views'], 0)
        self.assertEqual(m['policy_argmax_agreement'], 1.)
        self.assertEqual(m['maximum_value_difference'], 0.)
        self.assertEqual(len(m['numeric_columns']), 132)
        self.assertEqual(len(m['categorical_columns']), 41)
        self.assertEqual(m['policy_argmax_full_roster_identification_bounds'], [1., 1.])
        self.assertFalse(result['actual_captured_wire'])

    def test_wire_refusal_keeps_later_decisions_battles_and_other_owner(self):
        class Refuses(FakeAdapter):
            def reset(self, **kw):
                super().reset(**kw)
                self.failed = False
            def feed(self, lines):
                if self.seat == 'p1' and not self.failed:
                    self.failed = True
                    kind = lines[0].split('|')[1]
                    raise ValueError(f'wire refused kind={kind} offset=0 sha256=authored')
                super().feed(lines)
            def encode(self, request, **kw):
                if self.failed:
                    raise AssertionError('stale state must never be encoded')
                return super().encode(request, **kw)
        result, adapters = self.run_fixture(adapter_type=Refuses)
        m = result['metrics']
        self.assertGreater(m['views'], 0)
        self.assertGreater(m['refused_input_views'], 0)
        self.assertEqual(m['views']+m['refused_input_views'], m['expected_views'])
        self.assertTrue(any(r['propagated_prefix_refusal'] for r in m['refusal_records']))
        self.assertTrue(all(r['seat'] == 'p1' for r in m['refusal_records']))
        self.assertTrue(all(row['compared_cells']+row['refused_cells'] == row['expected_cells']
                            for row in m['numeric_columns']+m['categorical_columns']))
        self.assertFalse(m['model_forward_complete'])
        self.assertEqual(m['policy_argmax_full_roster_identification_bounds'][1], 1.)
        self.assertGreater(len(adapters['p2'].forwards), 0)

    def test_one_decision_encode_refusal_does_not_taint_next_decision(self):
        class Refuses(FakeAdapter):
            def reset(self, **kw):
                super().reset(**kw)
                self.encoded = 0
            def encode(self, *a, **k):
                self.encoded += 1
                if self.seat == 'p1' and self.encoded == 1:
                    raise ValueError('authored request refusal')
                return super().encode(*a, **k)
        result, _ = self.run_fixture(adapter_type=Refuses)
        m = result['metrics']
        self.assertGreater(m['owning_views']['p1'], 0)
        self.assertGreater(m['refused_input_views'], 0)
        self.assertEqual(m['views']+m['refused_input_views'], m['processed_views'])

    def test_forward_refusals_keep_input_mismatches_and_full_unknown_bound(self):
        class Refuses(FakeAdapter):
            def encode(self, *a, **k):
                result = {name: np.array(value, copy=True) for name, value in super().encode(*a, **k).items()}
                result['numeric_features'][22, 131] += np.float32(2e-6)
                result['categorical_ids'][22, 40] += 1
                return result
            def forward(self, observation):
                raise RuntimeError('authored forward refusal')
        result, _ = self.run_fixture(adapter_type=Refuses)
        m = result['metrics']
        self.assertEqual(m['views'], m['expected_views'])
        self.assertEqual(m['refused_input_views'], 0)
        self.assertEqual(m['refused_forward_views'], m['views'])
        self.assertEqual(m['model_forward_views'], 0)
        self.assertIsNone(m['policy_argmax_agreement'])
        self.assertEqual(m['policy_argmax_full_roster_identification_bounds'], [0., 1.])
        self.assertEqual(m['refusals_by_column']['value'], m['views'])
        self.assertEqual(m['numeric_columns'][131]['mismatched_cells'], m['views'])
        self.assertEqual(m['categorical_columns'][40]['mismatched_cells'], m['views'])
        self.assertGreater(m['numeric_columns'][131]['maximum_absolute_error'], 1e-6)

    def test_forward_disagreement_and_value_difference_remain_measured(self):
        class Disagrees(FakeAdapter):
            def forward(self, observation):
                super().forward(observation)
                reference = len(self.forwards) % 2 == 0
                return fp32(reverse=reference, value=.25 if reference else .5)
        result, _ = self.run_fixture(adapter_type=Disagrees)
        m = result['metrics']
        self.assertEqual(m['refused_input_views'], 0)
        self.assertEqual(m['refused_forward_views'], 0)
        self.assertEqual(m['policy_argmax_agreement'], 0.)
        self.assertEqual(m['policy_argmax_full_roster_identification_bounds'], [0., 0.])
        self.assertEqual(m['maximum_value_difference'], .25)
        self.assertEqual(m['mean_absolute_value_difference'], .25)

    def test_reference_refusal_does_not_stop_fixed_roster(self):
        def bad(*a, **k):
            raise ValueError('authored reference refusal')
        result, adapters = self.run_fixture(feed=bad)
        m = result['metrics']
        self.assertEqual(m['refused_input_views'], m['expected_views'])
        self.assertEqual(m['views'], 0)
        self.assertIsNone(m['numeric_columns'][0]['maximum_absolute_error'])
        self.assertEqual(m['policy_argmax_full_roster_identification_bounds'], [0., 1.])
        self.assertFalse(any(a.forwards for a in adapters.values()))

    def test_refusal_record_never_contains_raw_error_payload(self):
        e = ValueError('wire refused kind=damage offset=0 private=[1,2,3]')
        r = refusal(e, 'jax_feed', ('|damage|private-body',))
        self.assertEqual(r['line_kind'], 'damage')
        self.assertNotIn('private', json.dumps(r))

    def test_missing_or_duplicate_roster_is_not_a_complete_measurement(self):
        t = DiagnosticTally(2)
        t.visit(example())
        with self.assertRaises(ValueError):
            t.visit(example())
        with self.assertRaises(ValueError):
            t.report()

    def cli(self, output, result=None, error=None):
        from scripts import compare_projected_v4_diagnostic as cli
        with patch.object(cli, 'build_runtime', return_value=({}, {}, {'authored': True})), patch(
                'pokezero.projected_encoder_diagnostic.compare_diagnostic_collection',
                return_value=result, side_effect=error):
            return cli.main(['--checkpoint', 'fake', '--scratch-checkpoint', 'fake-copy',
                '--showdown', 'fake-showdown', '--node', 'fake-node', '--collection', 'fake-collection',
                '--collection-sha256', 'a'*64, '--cohort-sha256', 'b'*64, '--output', str(output)])

    def test_cli_publishes_173_rows_even_when_every_decision_refuses(self):
        def bad(*a, **k):
            raise ValueError('authored reference refusal')
        result, _ = self.run_fixture(feed=bad)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'output'
            self.assertEqual(self.cli(output, result), 0)
            with (output/'column-table.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 173)
            self.assertTrue(all(int(r['compared_cells']) == 0 and int(r['refused_cells']) > 0 for r in rows))
            self.assertEqual(json.loads((output/'attempt.json').read_text())['status'],
                             'COMPLETE_FIXED_ROSTER_WITH_REFUSALS')

    def test_cli_does_not_publish_a_table_after_integrity_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'output'
            self.assertEqual(self.cli(output, error=ValueError('source integrity failure')), 1)
            self.assertFalse((output/'column-table.csv').exists())
            self.assertFalse((output/'comparison.json').exists())


if __name__ == '__main__':
    unittest.main()
