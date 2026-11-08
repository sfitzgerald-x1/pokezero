"""Inert timing-denominator tests; no model, device, constructor or captured roots."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.phase_d_constructor_timing import (
    SCHEMA, ROSTER_SHA256, measure, p95, request_for, summarize, validate_config,
)
from pokezero.phase_d_saved_roots import address
from pokezero.phase_d_ladder_admission import timing_roster
from pokezero.phase_d_ladder_panel import encoded, write_new


def roster():
    return [dict(cell=dict(ordinal=i * 6), owner='p1', decision_index=i,
                 prefix_sha256=f'{i:064x}', turn=(1, 4, 10, 20)[i // 5])
            for i in range(20)]


def records(*, worlds=8, phase='warm'):
    return [dict(ordinal=i, address=list(address(row)), turn=row['turn'],
                 worlds=worlds, phase=phase, status='COMPLETE',
                 constructed_worlds=worlds, prepare_seconds=float(i + 1),
                 warm_context_qualified=phase == 'warm', refusals=[])
            for i, row in enumerate(roster())]


class ConstructorTimingTests(unittest.TestCase):
    def test_same_address_cannot_substitute_battle_stream_or_generation(self):
        meta = dict(roster()[0], battle_id='original-battle', stream_id=101,
                    generation=3, capture_path='/inert/capture', capture_row=7)
        captured = dict(meta, lines=['|turn|1', '|request|{}'], own_request_line='|request|{}')
        with patch('pokezero.phase_d_owning_capture.read_capture', return_value=iter([captured])):
            result = request_for(meta, worlds=8, phase='warm', ordinal=0)
        self.assertEqual((result['battle_id'], result['stream_id'], result['generation']),
                         ('original-battle', 101, 3))
        for key, changed in (('battle_id', 'different-battle'), ('stream_id', 202),
                             ('generation', 4)):
            row = dict(captured, **{key: changed})
            self.assertEqual(address(row), address(meta))
            with (self.subTest(key=key),
                  patch('pokezero.phase_d_owning_capture.read_capture', return_value=iter([row])),
                  self.assertRaisesRegex(ValueError, 'identity differs')):
                request_for(meta, worlds=8, phase='warm', ordinal=0)

    def run_inert_point(self, output, *, resume=False):
        selected = [dict(cell=dict(ordinal=i * 6), owner='p1', decision_index=i,
                    prefix_sha256=f'{i:064x}', turn=(1, 4, 10, 20)[i // 50])
                    for i in range(200)]
        chosen = timing_roster(selected)
        config = dict(schema=SCHEMA, rung='R7', worlds=32, seed=20261103,
            lanes=4096, branch_cap=262144, compute_dtype='float32',
            saved_root_admission=True, source_commit='a' * 40,
            source_directory='/inert/source', constructor_inputs={},
            roster=dict(path='/inert/roster', sha256=ROSTER_SHA256),
            warm_constructor=dict(worlds=[8, 16], passes=['warmup', 'warm'],
                roots_per_pass=20, before_full_request_timing=True, output=str(output)))
        args = SimpleNamespace(config='/inert/config', expected_config_sha256='b' * 64,
            producer_review='/inert/review', expected_producer_review_sha256='c' * 64,
            output=str(output))
        review = dict(status='SCOPED_CLEAR_INERT', source_commit=config['source_commit'],
                      config_sha256=args.expected_config_sha256)
        settings = dict(config_sha256=args.expected_config_sha256,
            producer_review_sha256=args.expected_producer_review_sha256,
            addresses=[list(address(row)) for row in chosen], worlds=[8, 16],
            passes=['warmup', 'warm'], seed=20261103)
        kept = None
        if resume:
            write_new(output / 'settings.json', settings)
            for i in (0, 1):
                write_new(output / '8/warmup/claims' / f'{i}.json',
                    dict(ordinal=i, address=list(address(chosen[i])),
                         settings_sha256=hashlib.sha256(encoded(settings)).hexdigest(),
                         invocation='prior-inert-invocation'))
            row = dict(ordinal=1, address=list(address(chosen[1])), turn=chosen[1]['turn'],
                worlds=8, phase='warmup', invocation='prior-inert-invocation',
                status='COMPLETE', constructed_worlds=8, prepare_seconds=1.,
                warm_context_qualified=False, refusals=[])
            write_new(output / '8/warmup/records/1.json', row)
            kept = (output / '8/warmup/records/1.json').read_bytes()
        calls, constructors, function_ids = [], [], []

        class InertConstructor:
            def __init__(self, **kwargs):
                self.count = kwargs['count']
                self.seen = set()
                constructors.append(self)
                function_ids.append(id(kwargs['functions']))
                assert kwargs['saved_root_admission'] is True
                assert kwargs['rung'] == 'R7' and kwargs['seed'] == 20261103

            def prepare(self, request, ordinal):
                assert ordinal not in self.seen
                self.seen.add(ordinal)
                calls.append((self.count, request['phase'], ordinal))
                return None, SimpleNamespace(fully_constructed=True,
                    worlds=[SimpleNamespace(state=object(), public=object(), refusals=())
                            for _ in range(self.count)]), ()

        def bound(path, digest):
            return {args.config: config, args.producer_review: review,
                    '/inert/roster': dict(selected=selected)}[path]

        modules = {
            'jax': SimpleNamespace(devices=lambda backend: ['inert-gpu'],
                                   block_until_ready=lambda values: None),
            'gpu_engine.phase_d_protocol_constructor': SimpleNamespace(
                ProtocolConstructor=InertConstructor,
                load_inputs=lambda args: ('catalog', 'source', {}, 'owning-config')),
            'pokezero.live_protocol_runtime': SimpleNamespace(
                CompiledEncoderFunctions=lambda catalog: SimpleNamespace(catalog=catalog)),
        }
        with (patch.dict(sys.modules, modules),
              patch('pokezero.phase_d_constructor_timing.read_bound', side_effect=bound),
              patch('pokezero.phase_d_constructor_timing.subprocess.check_output',
                    return_value='a' * 40 + '\n'),
              patch('pokezero.phase_d_constructor_timing.request_for',
                    side_effect=lambda meta, **kwargs: dict(phase=kwargs['phase'])),
              redirect_stdout(io.StringIO())):
            result = measure(args)
            before = len(calls)
            with self.assertRaisesRegex(ValueError, 'must not be rerun'):
                measure(args)
            self.assertEqual(len(calls), before)
        self.assertEqual(len(constructors), 4)
        self.assertEqual(len(set(function_ids)), 1)
        if kept is not None:
            self.assertEqual((output / '8/warmup/records/1.json').read_bytes(), kept)
        return result, calls

    def test_finite_two_budget_order_fresh_state_and_shared_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_inert_point(Path(tmp))
            self.assertEqual(calls, [(worlds, phase, i) for worlds in (8, 16)
                for phase in ('warmup', 'warm') for i in range(20)])
            self.assertEqual([x['qualified_attempt_times'] for x in result['summaries']], [20] * 4)

    def test_continuation_does_not_replay_or_claim_lost_warm_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_inert_point(Path(tmp), resume=True)
            self.assertEqual(len(calls), 78)
            self.assertNotIn((8, 'warmup', 0), calls)
            self.assertNotIn((8, 'warmup', 1), calls)
            self.assertEqual(result['summaries'][0]['dispositions']['INTERRUPTED_UNKNOWN'], 1)
            self.assertEqual(result['summaries'][1]['qualified_attempt_times'], 0)
            self.assertIsNone(result['summaries'][1]['full_roster_attempt_p95_seconds'])
            self.assertEqual(result['summaries'][3]['qualified_attempt_times'], 20)

    def test_full_twenty_nearest_rank_keeps_unknowns(self):
        self.assertEqual(p95(list(range(1, 21))), 19)
        self.assertEqual(p95(list(range(1, 20)) + [None]), 19)
        self.assertIsNone(p95(list(range(1, 19)) + [None, None]))
        self.assertIsNone(p95([None] * 20))
        for value in (True, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p95([value] + [1] * 19)

    def test_refused_attempt_time_is_not_full_construction_time(self):
        rows = records()
        for row in rows[:2]:
            row.update(status='REFUSED', constructed_worlds=0,
                       refusals=['BLOCKED: x', 'BLOCKED: another world'])
        result = summarize(roster(), rows, worlds=8, phase='warm')
        self.assertEqual(result['full_roster_attempt_p95_seconds'], 19)
        self.assertIsNone(result['full_roster_complete_p95_seconds'])
        self.assertTrue(result['full_roster_complete_p95_unavailable'])
        self.assertEqual(result['refusal_roots_by_kind'], {'BLOCKED': 2})
        self.assertEqual(result['by_turn_band']['0']['fully_constructed'], 3)
        self.assertEqual([v['requested'] for v in result['by_turn_band'].values()], [5] * 4)

    def test_interrupted_or_unconditioned_slots_do_not_gain_warm_times(self):
        rows = records()
        rows[0].update(status='INTERRUPTED_UNKNOWN', constructed_worlds=None,
                       prepare_seconds=None, warm_context_qualified=False,
                       refusals=['INTERRUPTED_ORIGINAL_ATTEMPT_UNKNOWN_NO_REPLAY'])
        rows[1]['warm_context_qualified'] = False
        result = summarize(roster(), rows, worlds=8, phase='warm')
        self.assertEqual(result['qualified_attempt_times'], 18)
        self.assertIsNone(result['full_roster_attempt_p95_seconds'])
        self.assertEqual(result['dispositions']['COMPLETE'], 19)
        self.assertEqual(result['dispositions']['INTERRUPTED_UNKNOWN'], 1)
        # A component timer cannot invent successful return data after loss.
        rows[0]['prepare_seconds'] = 0.
        with self.assertRaisesRegex(ValueError, 'Interrupted attempt'):
            summarize(roster(), rows, worlds=8, phase='warm')

    def test_missing_record_keeps_its_original_band_and_denominator(self):
        result = summarize(roster(), records()[:-2], worlds=8, phase='warm')
        self.assertEqual(result['requested_roots'], 20)
        self.assertEqual(result['dispositions']['UNAVAILABLE'], 2)
        self.assertEqual(result['by_turn_band']['3']['requested'], 5)
        self.assertEqual(result['by_turn_band']['3']['returned'], 3)
        self.assertIsNone(result['full_roster_complete_p95_seconds'])

    def test_foreign_duplicate_budget_and_time_corruption_refuse(self):
        original = records()
        for change in ('duplicate', 'worlds', 'phase', 'address', 'turn', 'time'):
            rows = deepcopy(original)
            if change == 'duplicate':
                rows.append(rows[0])
            elif change == 'worlds':
                rows[0]['worlds'] = 16
            elif change == 'phase':
                rows[0]['phase'] = 'warmup'
            elif change == 'address':
                rows[0]['address'][0] = 999
            elif change == 'turn':
                rows[0]['turn'] = 99
            else:
                rows[0]['prepare_seconds'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(roster(), rows, worlds=8, phase='warm')

    def test_frozen_protocol_rejects_budget_pass_order_or_roster_changes(self):
        config = dict(schema=SCHEMA, rung='R7', worlds=32, seed=20261103,
                      lanes=4096, branch_cap=262144, compute_dtype='float32',
                      saved_root_admission=True, roster=dict(sha256=ROSTER_SHA256),
                      warm_constructor=dict(worlds=[8, 16], passes=['warmup', 'warm'],
                          roots_per_pass=20, before_full_request_timing=True))
        self.assertIs(validate_config(config), config)
        for change in ('worlds', 'passes', 'roots', 'roster', 'seed', 'mode'):
            value = deepcopy(config)
            if change == 'worlds':
                value['warm_constructor']['worlds'] = [16, 32]
            elif change == 'passes':
                value['warm_constructor']['passes'].reverse()
            elif change == 'roots':
                value['warm_constructor']['roots_per_pass'] = 19
            elif change == 'roster':
                value['roster']['sha256'] = '0' * 64
            elif change == 'seed':
                value['seed'] = 0
            else:
                value['saved_root_admission'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(value)


if __name__ == '__main__':
    unittest.main()
