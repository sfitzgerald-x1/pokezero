"""Authored capture fixtures only: no corpus, models, simulator or JAX runtime."""
import copy
from contextlib import ExitStack
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pokezero.actual_saved_wire import (ActualSavedWireCursor, ActualSavedWirePair,
    ActualSavedWireTally, SavedWireJaxAdapter, WireEncoderRefusal, actual_pair_arrays,
    compare_saved_battle, compare_saved_collection, inventory_saved_battle, _rows)
from pokezero.mirror_value_probe import canonical, digest, file_sha
from pokezero.projected_encoder_diagnostic import DiagnosticTally
from pokezero.projected_encoder_parity import projected_pair_arrays
from pokezero.recorded_encoder_parity import SHAPES
from test_recorded_mirror_roots import boundary
from test_projected_encoder_parity import fp32


def fixture(*, capped=False, absent=False, battle_id='actual-authored'):
    battle = dict(index=0, battle_id=battle_id, seed=1234)
    source, owners, rows = [], {'p1': [], 'p2': []}, []
    for round_index in range(2):
        row = boundary(battle)
        row['decision_round_index'] = round_index
        turn = round_index + 3
        for ref in row['observations'].values():
            ref['turn_number'] = turn
        if round_index:
            row['requests']['p2']['wait'] = True
            row['requested_players'] = ['p1']
            row['observations'].pop('p2')
            row['actions'].pop('p2')
            row['paired'] = False
        intro = ['|player|p1|alpha|', '|player|p2|beta|'] if not round_index else []
        mixed = intro + [f'|turn|{turn}', '|-damage|p1a: Snorlax|133/299'] + [
            '|request|' + canonical(row['requests'][seat]).decode() for seat in ('p1', 'p2')]
        row.update(protocol_start=len(source), protocol_end=len(source) + len(mixed), protocol_delta=mixed)
        source.extend(mixed)
        row['public_prefix_sha256'] = digest(source)
        row['root_id'] = digest([battle_id, round_index, row['public_prefix_sha256'], row['requests']])
        row['owning_wire'] = {}
        for seat in owners:
            # Deliberately different from mixed provenance, and public evidence
            # AFTER the own request: a request-tail projector would be wrong.
            hp = '133/299' if seat == 'p1' else '45/100'
            lines = intro + [f'|turn|{turn}', f'|-damage|p1a: Snorlax|{hp}',
                '|request|' + canonical(row['requests'][seat]).decode(), '|', '|message|after-own-request']
            start = len(owners[seat])
            owners[seat].extend(lines)
            row['owning_wire'][seat] = dict(start=start, end=len(owners[seat]),
                delta=lines, prefix_sha256=digest(owners[seat]))
        rows.append(row)
    terminal = None if absent else dict(winner=None if capped else 'p1', turn_count=4, capped=capped)
    tail = ['|upkeep'] if capped or absent else ['|win|alpha', '|']
    end = dict(event='end', battle_id=battle_id, actual_terminal=terminal,
        protocol_start=len(source), protocol_end=len(source) + len(tail), protocol_delta=tail,
        owning_wire={})
    for seat in owners:
        start = len(owners[seat])
        owners[seat].extend(tail)
        end['owning_wire'][seat] = dict(start=start, end=len(owners[seat]), delta=tail,
            prefix_sha256=digest(owners[seat]))
    rows.append(end)
    battle.update(actual_terminal=terminal, paired_rows=2, identities=[dict(root_id=rows[0]['root_id'],
        seat=s, turn_number=3, decision_round_index=0) for s in ('p1', 'p2')])
    return battle, rows


class Adapter:
    def __init__(self, seat, *, refusal_stage=None):
        self.seat, self.refusal_stage = seat, refusal_stage
        self.ingested, self.encoded, self.forwards = [], [], []
        self.failed = False

    def reset(self, **identity):
        self.identity = identity
        self.failed = False

    def ingest(self, lines):
        self.ingested.append(lines)
        if self.refusal_stage == 'ingest' and self.seat == 'p1':
            self.failed = True
            raise WireEncoderRefusal(dict(stage='jax_ingest', line_kind='-damage',
                line_offset=3, line_sha256='a' * 64, error_type='ValueError'))

    def encode(self, view):
        if self.failed:
            raise AssertionError('stale state encoded')
        self.encoded.append(view)
        if self.refusal_stage == 'encode' and self.seat == 'p1':
            self.failed = True
            raise ValueError('unsupported encode')
        reference = json.loads(view.own_reference_json)
        return {key: np.asarray(reference[key], dtype=np.bool_ if key.endswith('mask')
            else np.float32 if key == 'numeric_features' else np.int32) for key in SHAPES}

    def forward(self, arrays):
        self.forwards.append(arrays)
        if self.refusal_stage == 'forward':
            raise ValueError('unsupported forward')
        return fp32()


class ActualSavedWireTests(unittest.TestCase):
    def run_fixture(self, *, refusal_stage=None, rows=None, battle=None):
        if rows is None:
            battle, rows = fixture()
        adapters = {s: Adapter(s, refusal_stage=refusal_stage) for s in ('p1', 'p2')}
        tally = DiagnosticTally(3, parity=ActualSavedWireTally(3))
        with patch('numpy.load', side_effect=AssertionError('NPZ forbidden')), patch(
                'subprocess.Popen', side_effect=AssertionError('simulator forbidden')):
            compare_saved_battle(rows, battle=battle, ordinal=0, adapters=adapters,
                tally=tally, with_forward=True)
        return tally.report(), adapters

    def test_actual_owner_lines_and_consumed_arrays_not_mixed_or_peer(self):
        metrics, adapters = self.run_fixture()
        self.assertEqual((metrics['views'], metrics['processed_views']), (3, 3))
        self.assertEqual(metrics['policy_argmax_agreement'], 1.)
        self.assertEqual(metrics['maximum_value_difference'], 0.)
        self.assertTrue(metrics['actual_captured_wire'])
        self.assertTrue(metrics['original_reference_values_compared'])
        self.assertFalse(metrics['raw_bridge_chunk_bytes_captured'])
        self.assertFalse(metrics['full_encoder_admitted'])
        self.assertFalse(metrics['minimum10000_views_met'])
        self.assertFalse(metrics['input_numeric_criteria_met'])
        self.assertFalse(any(k.startswith('projected_') for k in metrics))
        for seat, adapter in adapters.items():
            self.assertEqual(len(adapter.ingested), 2)
            for lines in adapter.ingested:
                self.assertIn('|', lines)
                self.assertEqual(lines[-1], '|message|after-own-request')
                self.assertFalse(any('|win|' in line for line in lines))
                packets = [json.loads(line[len('|request|'):]) for line in lines if line.startswith('|request|')]
                self.assertTrue(all(p['side']['id'] == seat for p in packets))
            expected_hp = '133/299' if seat == 'p1' else '45/100'
            self.assertIn(f'|-damage|p1a: Snorlax|{expected_hp}', adapter.ingested[0])
        self.assertEqual([len(a.encoded) for a in adapters.values()], [2, 1])

    def test_cursor_atomic_hash_and_identity_failures_never_become_refusals(self):
        battle, rows = fixture()
        mutations = (
            lambda row: row['owning_wire']['p1'].update(start=1),
            lambda row: row['owning_wire']['p2'].update(prefix_sha256='f' * 64),
            lambda row: row['owning_wire'].pop('p2'),
            lambda row: row['observations']['p1'].update(turn_number=2),
            lambda row: row['observations']['p1'].update(extra='not-a-tensor'),
            lambda row: row.update(root_id='f' * 64),
            lambda row: row['observations']['p1']['numeric_features'][22].__setitem__(131, float('nan')),
        )
        for mutate in mutations:
            cursor = ActualSavedWireCursor(battle)
            bad = copy.deepcopy(rows[0])
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, KeyError)):
                cursor.advance(bad)
            self.assertEqual(cursor.source.next_round, 0)
            self.assertTrue(all(p.count == 0 for p in cursor.prefixes.values()))
        bad = copy.deepcopy(rows)
        bad[0]['owning_wire']['p1']['delta'][3] = '|request|' + canonical(bad[0]['requests']['p2']).decode()
        bad[0]['owning_wire']['p1']['prefix_sha256'] = digest(bad[0]['owning_wire']['p1']['delta'])
        with self.assertRaisesRegex(ValueError, 'owning request identity'):
            self.run_fixture(battle=battle, rows=bad)

    def test_refusals_taint_only_owner_and_keep_every_requested_decision(self):
        for stage in ('ingest', 'encode'):
            with self.subTest(stage=stage):
                metrics, adapters = self.run_fixture(refusal_stage=stage)
                self.assertEqual((metrics['views'], metrics['refused_input_views']), (1, 2))
                self.assertEqual(metrics['policy_argmax_full_roster_identification_bounds'], [1/3, 1.])
                self.assertEqual([r['propagated_prefix_refusal'] for r in metrics['refusal_records']], [False, True])
                self.assertTrue(all(r['seat'] == 'p1' for r in metrics['refusal_records']))
                self.assertEqual(len(adapters['p1'].ingested), 1)
                self.assertEqual(len(adapters['p2'].ingested), 2)
                for row in metrics['numeric_columns'] + metrics['categorical_columns']:
                    self.assertEqual(row['compared_cells'] + row['refused_cells'], row['expected_cells'])

    def test_forward_unknowns_retain_input_table_and_full_bounds(self):
        metrics, _ = self.run_fixture(refusal_stage='forward')
        self.assertEqual((metrics['views'], metrics['refused_input_views'], metrics['refused_forward_views']), (3, 0, 3))
        self.assertEqual(metrics['policy_argmax_full_roster_identification_bounds'], [0., 1.])
        self.assertFalse(metrics['identification_bounds_are_confidence_intervals'])
        self.assertEqual(metrics['refusals_by_column'], {'policy_logits': 3, 'value': 3})

    def test_terminal_caps_absence_and_identity_inventory_are_honest(self):
        for kwargs in ({}, {'capped': True}, {'absent': True}):
            battle, rows = fixture(**kwargs)
            report = inventory_saved_battle(rows, battle)
            self.assertEqual(report['requested_views'], 3)
            self.assertEqual(report['asymmetric_boundaries'], 1)
            self.assertEqual(report['actual_terminal'], battle['actual_terminal'])
            self.run_fixture(battle=battle, rows=rows)
        battle, rows = fixture()
        for bad in (rows[:-1], rows + [rows[-1]]):
            with self.assertRaises(ValueError):
                inventory_saved_battle(bad, battle)
        for mutate in (lambda r: r[-1].update(actual_terminal=None),
            lambda r: r[-1]['owning_wire']['p1'].update(prefix_sha256='0' * 64),
            lambda r: r[-1]['owning_wire']['p2']['delta'].__setitem__(0, '|win|beta')):
            bad = copy.deepcopy(rows)
            mutate(bad)
            with self.assertRaises(ValueError):
                inventory_saved_battle(bad, battle)
        bad_battle = copy.deepcopy(battle)
        bad_battle['identities'][1]['seat'] = 'p1'
        with self.assertRaisesRegex(ValueError, 'identity roster'):
            inventory_saved_battle(rows, bad_battle)

    def test_canonical_actual_envelope_cannot_be_scored_as_projected(self):
        battle, rows = fixture()
        projection, _ = ActualSavedWireCursor(battle).advance(rows[0])
        view = projection.views[0]
        arrays = Adapter('p1').encode(view)
        pair = ActualSavedWirePair(view.decision_id, view.battle_id, view.seat, 0, view.turn_number,
            view.source_prefix_sha256, view.own_request_json, view.own_reference_json,
            digest(json.loads(view.own_reference_json)), arrays)
        actual_pair_arrays(pair)
        with self.assertRaises(ValueError):
            projected_pair_arrays(pair)
        for bad in (replace(pair, original_reference_sha256='0' * 64),
                    replace(pair, turn_number=7), replace(pair, own_reference_json=b'{}')):
            with self.assertRaises(ValueError):
                actual_pair_arrays(bad)

    def test_battle_checkpoint_durable_atomicity_and_no_refill(self):
        from scripts.compare_actual_saved_wire import atomic_json, column_table
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            battles, reports = [], []
            for i in range(2):
                battle, rows = fixture(battle_id=f'actual-authored-{i}')
                battle['index'] = i
                directory = root / f'battle-{i:04d}'
                directory.mkdir()
                path = directory / 'boundaries.jsonl.gz'
                with gzip.open(path, 'wt') as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + '\n')
                battle['files'] = {'boundaries.jsonl.gz': file_sha(path)}
                battles.append(battle)
                reports.append(inventory_saved_battle(rows, battle))
            inventory = dict(root=root, expected_views=6, battles=battles, selected=[], reports=reports,
                selection_sha256='a' * 64, selection_metadata_sha256='c' * 64, collection_sha256='b' * 64)
            (root / 'collection.json').write_text('{}')
            inventory['collection_sha256'] = file_sha(root / 'collection.json')
            checkpoints = []
            def checkpoint(value):
                atomic_json(root / 'progress.json', value)
                checkpoints.append(json.loads((root / 'progress.json').read_text()))
            adapters = {s: Adapter(s, refusal_stage='ingest') for s in ('p1', 'p2')}
            result = compare_saved_collection(inventory, adapters=adapters, checkpoint=checkpoint)
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(checkpoints[0]['metrics']['unprocessed_views'], 3)
            self.assertEqual(checkpoints[-1]['metrics'], result['metrics'])
            self.assertEqual(result['metrics']['processed_views'], 6)
            self.assertEqual(result['metrics']['refused_input_views'], 4)
            self.assertFalse(result['full_encoder_admitted'])
            previous = (root / 'progress.json').read_bytes()
            with self.assertRaises(ValueError):
                atomic_json(root / 'progress.json', {'bad': float('nan')})
            self.assertEqual((root / 'progress.json').read_bytes(), previous)
            with patch('scripts.compare_actual_saved_wire.os.replace', side_effect=OSError('authored write failure')):
                with self.assertRaises(OSError):
                    atomic_json(root / 'progress.json', {'changed': True})
            self.assertEqual((root / 'progress.json').read_bytes(), previous)
            self.assertFalse(list(root.glob('*.tmp')))
            column_table(root / 'columns.csv', result['metrics'])
            with (root / 'columns.csv').open() as stream:
                columns = list(csv.DictReader(stream))
            self.assertEqual(len(columns), 173)
            self.assertTrue(all(int(c['expected_cells']) == 6 * 23 for c in columns))

    def test_hashed_compressed_stream_rejects_changes(self):
        _, rows = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.gz'
            with gzip.open(path, 'wt') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            expected = file_sha(path)
            self.assertEqual(len(list(_rows(path, expected))), 3)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                list(_rows(path, '0' * 64))
            iterator = _rows(path, expected)
            next(iterator)
            with path.open('ab') as stream:
                stream.write(b'\0')
            with self.assertRaisesRegex(ValueError, 'changed during read'):
                list(iterator)

    def test_backend_ingest_retains_request_public_order_and_offset_evidence(self):
        from pokezero.live_protocol_actor import JaxBackend
        events = []
        class Encoder:
            def parse_own_requests(self, catalog, packets, *, owner_sides):
                events.append(('request', packets[0]['rqid']))
        class Backend:
            ingest = JaxBackend.ingest
            _check_namespace = lambda self: None
            terminal_lifecycle_line = None
            owner = 'p1'
            catalog = None
            adapter = SimpleNamespace(encoder=Encoder(), feed=lambda lines: events.append(('public', lines)))
        adapter = SavedWireJaxAdapter(None, 'p1')
        adapter.backend = Backend()
        request = lambda i: '|request|' + json.dumps({'side': {'id': 'p1'}, 'rqid': i})
        adapter.ingest(('|turn|1', request(1), '|', '|turn|2', request(2), '|upkeep'))
        self.assertEqual(events, [('public', ('|turn|1',)), ('request', 1),
            ('public', ('|', '|turn|2')), ('request', 2), ('public', ('|upkeep',))])
        def refused(lines):
            raise ValueError('wire refused kind=upkeep offset=1')
        adapter.backend.adapter.feed = refused
        with self.assertRaises(WireEncoderRefusal) as caught:
            adapter.ingest((request(3), '|', '|upkeep'))
        self.assertEqual(caught.exception.detail['line_kind'], 'upkeep')
        self.assertEqual(caught.exception.detail['line_offset'], 2)

    def test_metadata_selection_exact_training_roster_and_denominator(self):
        import pokezero.actual_saved_wire as actual
        from pokezero.mirror_value_probe import cohort
        with tempfile.TemporaryDirectory() as tmp:
            producer = Path(tmp).resolve()
            root = producer / 'collection'
            root.mkdir()
            (producer / 'repo').mkdir()
            (producer / 'repo/source.py').write_text('# authored source only\n')
            files = {'source.py': file_sha(producer / 'repo/source.py')}
            config = dict(root='metadata-authored', battles=344, champion_sha256=actual.CHAMPION_SHA,
                probe_contract='owning-wire-true-moves-champion-linear-v2')
            roster = cohort(config['root'], config['battles'])
            logical = digest(roster)
            def write(path, value):
                path.write_bytes(canonical(value))
                return file_sha(path)
            config_sha = write(producer / 'config.json', config)
            prespec_sha = write(producer / 'prespec.json', {'authored': True})
            files_sha = write(producer / 'scientific-hashes.json', files)
            cohort_sha = write(root / 'cohort.json', dict(schema='pokezero.mirror_cohort.v1',
                config=config, config_sha256=digest(config), battles=roster, roster_sha256=logical))
            manifest = dict(schema='pokezero.mirror_collection.v1', status='COMPLETE_FIXED_COHORT',
                config=config, config_sha256=digest(config), cohort_sha256=cohort_sha,
                battles=roster, no_refill=True, sibling_measurement=False,
                battle_receipt_hashes={f"battle-{row['index']:04d}/battle.json": 'a' * 64 for row in roster})
            collection_sha = write(root / 'collection.json', manifest)
            prespec = dict(schema='pokezero.B2-actual-owning-wire-selection-prespec.v1',
                producer_source=actual.PRODUCER_SOURCE, producer_config_sha256=config_sha,
                producer_prespec_sha256=prespec_sha, cohort_sha256=logical,
                selection_is_training_only=True, no_actual_tensors_or_outcomes_examined=True,
                minimum_view_gate=10000,
                selection='Exactly first256 TRAIN cohort battles sorted by numeric index, both seats and every requested decision; no tensor/outcome/mismatch/length-based selection.')
            selection_path = producer / 'selection.json'
            selection_sha = write(selection_path, prespec)
            selected = [r for r in roster if r['split'] == 'train'][:256]
            frozen = [dict(**{k: r[k] for k in ('index', 'battle_id', 'seed', 'split')},
                battle_receipt_sha256='a' * 64, source_file_sha256='b' * 64,
                boundaries=1, requested_views=2) for r in selected]
            metadata = dict(schema='pokezero.B2-fixed-actual-owning-selection-metadata.v1',
                collection_sha256=collection_sha, cohort_file_sha256=cohort_sha,
                canonical_roster_digest=logical, producer_source_commit=actual.PRODUCER_SOURCE,
                selected_battles=256, selected=frozen, expected_views=512,
                no_outcomes_tensors_or_actions_used=True, only_metadata_fields_examined=True,
                selection='First256TRAINcohortindicesnumeric/bothseats/allrequested; fixedbefore tensors/results')
            metadata_path = producer / 'metadata.json'
            def read(value):
                metadata_sha = write(metadata_path, value)
                return actual._selection(root, selection_path, selection_sha, collection_sha,
                    metadata_path, metadata_sha)
            with ExitStack() as stack:
                for name, value in dict(PRODUCER_CONFIG_SHA=config_sha, PRODUCER_PRESPEC_SHA=prespec_sha,
                    PRODUCER_HASHES_SHA=files_sha, PRODUCER_COHORT_FILE_SHA=cohort_sha,
                    PRODUCER_ROSTER_DIGEST=logical, PRODUCER_FILES=files).items():
                    stack.enter_context(patch.object(actual, name, value))
                _, got, counts = read(metadata)
                self.assertEqual(got, selected)
                self.assertEqual(counts['expected_views'], 512)
                self.assertNotEqual(logical, cohort_sha)
                for mutate in (lambda m: m.update(expected_views=513),
                    lambda m: m['selected'].reverse(),
                    lambda m: m['selected'][0].update(split='test'),
                    lambda m: m['selected'][0].update(battle_receipt_sha256='c' * 64),
                    lambda m: m.update(cohort_file_sha256=logical),
                    lambda m: m['selected'].pop(),
                    lambda m: m.update(no_outcomes_tensors_or_actions_used=False)):
                    bad = copy.deepcopy(metadata)
                    mutate(bad)
                    with self.assertRaises(ValueError):
                        read(bad)
                (producer / 'repo/source.py').write_text('# changed\n')
                with self.assertRaisesRegex(ValueError, 'source file mismatch'):
                    read(metadata)

    def test_cli_integrity_failure_precedes_runtime_and_creates_no_table(self):
        from scripts import compare_actual_saved_wire as cli
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'result'
            args = []
            for key, value in dict(checkpoint='fake', scratch_checkpoint='fake-copy', showdown='fake',
                node='fake', collection='fake', collection_sha256='a' * 64, selection='fake',
                selection_sha256='b' * 64, selection_metadata='fake',
                selection_metadata_sha256='c' * 64, output=str(output)).items():
                args.extend(['--' + key.replace('_', '-'), value])
            with patch('pokezero.actual_saved_wire.inventory_saved_collection',
                    side_effect=ValueError('private failure text')), patch.object(cli, 'build_runtime') as runtime:
                self.assertEqual(cli.main(args), 1)
            runtime.assert_not_called()
            self.assertFalse((output / 'comparison.json').exists())
            self.assertFalse((output / 'column-table.csv').exists())
            receipt = json.loads((output / 'attempt.json').read_bytes())
            self.assertFalse(receipt['actual_captured_wire'])
            self.assertFalse(receipt['original_reference_values_compared'])
            self.assertEqual(receipt['actual_exit'], 1)
            self.assertNotIn('private', json.dumps(receipt))

    def test_backend_requires_reviewed_owner_initialization_dependency(self):
        adapter = SavedWireJaxAdapter(SimpleNamespace(catalog='authored'), 'p1')
        with patch('pokezero.live_protocol_actor.JaxBackend',
                return_value=SimpleNamespace(adapter=SimpleNamespace(owner_side=None))):
            with self.assertRaisesRegex(ValueError, 'HP-provenance encoder fix required'):
                adapter.reset(stream_id=1, battle_id='fixture')


if __name__ == '__main__':
    unittest.main()
