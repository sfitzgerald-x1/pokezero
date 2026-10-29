import copy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import numpy as np

from pokezero.mirror_value_probe import digest
from pokezero.recorded_encoder_parity import SHAPES
from pokezero.recorded_player_encoder_runner import compare_player_battle, VISIBILITY
from pokezero.recorded_player_encoder_runner import SHOWDOWN_REVISION
from pokezero.observation import PokeZeroObservationV0, ObservationPerspective
import test_recorded_encoder_inventory as fixtures


def tensors(marker):
    return {name: np.full(shape, marker if name == 'numeric_features' else True
            if name.endswith('mask') else 0, dtype=np.float32
            if name == 'numeric_features' else np.bool_
            if name.endswith('mask') else np.int32)
            for name, shape in SHAPES.items()}


def observation(seat, **changes):
    other = 'p2' if seat == 'p1' else 'p1'
    value = PokeZeroObservationV0(**tensors(.5), schema_version='pokezero.observation.v4',
        perspective=ObservationPerspective.from_showdown_slot(seat, seat),
        metadata=dict(battle_id='two-player-fixture', player_id=seat, showdown_slot=seat,
                      opponent_showdown_slot=other, turn_number=3))
    return replace(value, **changes)


class Adapter:
    def __init__(self, seat):
        self.seat = seat
        self.feeds = []
        self.encodes = []
        self.output = tensors(.75)

    def reset(self, **kwargs):
        self.identity = kwargs

    def feed(self, lines):
        if any(line.startswith('|request|') for line in lines):
            raise AssertionError('private request leaked into projected public fold')
        self.feeds.append(lines)

    def encode(self, request, *, owner_side):
        if request['side']['id'] != self.seat or owner_side != int(self.seat[1])-1:
            raise AssertionError('peer request supplied')
        self.encodes.append(request)
        return self.output


class Projector:
    def __init__(self):
        self.calls = []
        self.prefix = []

    def project_nonrequest_delta(self, lines):
        self.calls.append(lines)
        start = len(self.prefix)
        self.prefix.extend(lines)
        return SimpleNamespace(for_seat=lambda seat: SimpleNamespace(
            seat=seat, lines=tuple(line + '|projected-for=' + seat for line in lines)),
            battle_id='two-player-fixture', source_nonrequest_start=start,
            source_nonrequest_end=len(self.prefix), source_nonrequest_prefix_sha256=digest(self.prefix),
            counts={'inputs': len(lines)}, showdown_revision=SHOWDOWN_REVISION,
            scope='synthetic projected source', struggle_warning_policy='explicit omission unqualified',
            captured_player_wire=False, owning_stream_bytes_reconstructed=False,
            public_wire_qualified=False, fullencoder=False)


class ProjectedRunnerTests(unittest.TestCase):
    def rows(self):
        battle = dict(battle_id='two-player-fixture', seed=123)
        first = fixtures.RecordedEncoderInventoryTests().rows(battle)[0]
        second = copy.deepcopy(first)
        second['decision_round_index'] = 1
        second['requests']['p2']['wait'] = True
        second['requested_players'] = ['p1']
        second['paired'] = False
        second['observations'].pop('p2')
        second['actions'].pop('p2')
        delta = ['|', '|request|' + json.dumps(second['requests']['p2'])]
        second.update(protocol_start=first['protocol_end'],
            protocol_end=first['protocol_end']+len(delta), protocol_delta=delta,
            public_prefix_sha256=digest(first['protocol_delta']+delta))
        second['root_id'] = digest([battle['battle_id'], 1,
                                   second['public_prefix_sha256'], second['requests']])
        end = dict(event='end', protocol_start=second['protocol_end'],
                   protocol_end=second['protocol_end']+1, protocol_delta=['|win|p1'])
        return battle, [first, second, end]

    def run_rows(self, rows, battle, *, reference_feed=None, on_view=None, adapters=None, projector=None):
        self.projector = projector or Projector()
        self.adapters = adapters or {seat: Adapter(seat) for seat in ('p1', 'p2')}
        self.requests = []
        self.pairs = []

        def reference(batch, *, own_request_deltas, requested_seats, request_tail_verified):
            self.assertIs(request_tail_verified, True)
            self.requests.append(own_request_deltas)
            for seat, lines in own_request_deltas.items():
                for line in lines:
                    self.assertEqual(json.loads(line[len('|request|'):])['side']['id'], seat)
            return {seat: observation(seat) for seat in requested_seats}

        return compare_player_battle(iter(rows), battle=battle, stream_id=10,
            projector=self.projector, reference_feed=reference_feed or reference,
            adapters=self.adapters, on_view=on_view or self.pairs.append)

    def test_both_histories_advance_on_asymmetric_wait_and_all_references_regenerate(self):
        battle, rows = self.rows()
        report = self.run_rows(rows, battle)
        self.assertEqual((report['boundaries'], report['views'], report['asymmetric_boundaries']), (2, 3, 1))
        self.assertEqual(report['owning_request_deltas'], {'p1': 1, 'p2': 2})
        self.assertEqual([len(self.adapters[s].feeds) for s in ('p1', 'p2')], [2, 2])
        self.assertEqual([len(self.adapters[s].encodes) for s in ('p1', 'p2')], [2, 1])
        self.assertEqual([self.adapters[s].identity['stream_id'] for s in ('p1', 'p2')], [20, 21])
        self.assertEqual(self.requests[1]['p1'], ())
        self.assertTrue(json.loads(self.requests[1]['p2'][0][len('|request|'):])['wait'])
        for pair in self.pairs:
            self.assertEqual(pair.visibility, VISIBILITY)
            self.assertFalse(pair.actual_captured_wire)
            self.assertFalse(pair.full_encoder_admitted)
            self.assertEqual(json.loads(pair.regenerated_reference_json)['numeric_features'][0][0], .5)
            self.assertEqual(pair.actual['numeric_features'][0, 0], .75)
        self.adapters['p1'].output['numeric_features'][:] = 9
        self.assertEqual(self.pairs[0].actual['numeric_features'][0, 0], .75)
        with self.assertRaises(ValueError):
            self.pairs[0].actual['numeric_features'][0, 0] = 9

    def test_interleaved_private_public_lines_refuse_without_reordering(self):
        battle, rows = self.rows()
        first = rows[0]
        first['protocol_delta'].append('|')
        first['protocol_end'] += 1
        first['public_prefix_sha256'] = digest(first['protocol_delta'])
        first['root_id'] = digest([battle['battle_id'], 0,
                                  first['public_prefix_sha256'], first['requests']])
        with self.assertRaisesRegex(ValueError, 'interleaved'):
            self.run_rows(rows[:1], battle)
        self.assertEqual(self.projector.calls, [])
        self.assertEqual(self.requests, [])

    def test_wrong_reference_roster_turn_partial_tensor_and_missing_terminal_abort(self):
        battle, rows = self.rows()
        for mode in ('roster', 'turn', 'shape', 'terminal'):
            def reference(batch, *, own_request_deltas, requested_seats, request_tail_verified):
                self.assertIs(request_tail_verified, True)
                out = {seat: observation(seat) for seat in requested_seats}
                if mode == 'roster':
                    out.pop('p2')
                elif mode == 'turn':
                    out['p1'].metadata['turn_number'] = 4
                elif mode == 'shape':
                    out['p1'] = replace(out['p1'], numeric_features=np.zeros((22, 132), np.float32))
                return out
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_rows(rows[:-1] if mode == 'terminal' else rows,
                              battle, reference_feed=reference)

    def test_shared_adapter_and_late_callback_failure_never_return_success(self):
        battle, rows = self.rows()
        adapter = Adapter('p1')
        with self.assertRaises(ValueError):
            self.run_rows(rows, battle, adapters={'p1': adapter, 'p2': adapter})
        calls = []
        def callback(pair):
            calls.append(pair.seat)
            if len(calls) == 3:
                raise RuntimeError('whole collection must discard partial tally')
        with self.assertRaisesRegex(RuntimeError, 'discard'):
            self.run_rows(rows, battle, on_view=callback)
        self.assertEqual(calls, ['p1', 'p2', 'p1'])

    def test_projected_private_line_refuses_before_any_reference_consumption(self):
        class PrivateProjector(Projector):
            def project_nonrequest_delta(self, lines):
                batch = super().project_nonrequest_delta(lines)
                batch.for_seat = lambda seat: SimpleNamespace(seat=seat, lines=(
                    '|request|' + json.dumps({'side': {'id': 'p2'}}),))
                return batch
        battle, rows = self.rows()
        with self.assertRaisesRegex(ValueError, 'nonrequest'):
            self.run_rows(rows, battle, projector=PrivateProjector())
        self.assertEqual(self.requests, [])
        self.assertFalse(any(adapter.feeds for adapter in self.adapters.values()))

    def test_dynamic_projector_is_read_once_and_both_consumers_use_same_snapshot(self):
        class DynamicProjector(Projector):
            def __init__(self):
                super().__init__()
                self.owner_calls = {'p1': 0, 'p2': 0}

            def project_nonrequest_delta(self, lines):
                batch = super().project_nonrequest_delta(lines)
                def for_seat(seat):
                    self.owner_calls[seat] += 1
                    return SimpleNamespace(seat=seat, lines=(
                        '|-hint|A' if self.owner_calls[seat] % 2 else '|-hint|B',))
                batch.for_seat = for_seat
                return batch
        seen = []
        def reference(batch, *, own_request_deltas, requested_seats, request_tail_verified):
            first = {s: batch.for_seat(s).lines for s in ('p1', 'p2')}
            self.assertEqual(first, {s: batch.for_seat(s).lines for s in ('p1', 'p2')})
            seen.append(first)
            return {seat: observation(seat) for seat in requested_seats}
        battle, rows = self.rows()
        projector = DynamicProjector()
        self.run_rows(rows, battle, projector=projector, reference_feed=reference)
        self.assertEqual(projector.owner_calls, {'p1': 2, 'p2': 2})
        for index, batch in enumerate(seen):
            for seat in ('p1', 'p2'):
                self.assertEqual(self.adapters[seat].feeds[index], batch[seat])

    def test_reference_schema_seat_battle_and_exact_turn_identity_refuse(self):
        battle, rows = self.rows()
        for mode in ('perspective', 'schema', 'battle', 'turn_bool', 'metadata_seat'):
            def reference(batch, *, own_request_deltas, requested_seats, request_tail_verified):
                out = {seat: observation(seat) for seat in requested_seats}
                if mode == 'perspective':
                    out['p1'] = replace(out['p1'], perspective=out['p2'].perspective)
                elif mode == 'schema':
                    out['p1'] = replace(out['p1'], schema_version='pokezero.observation.v3')
                else:
                    key, value = {'battle': ('battle_id', 'foreign'),
                                  'turn_bool': ('turn_number', True),
                                  'metadata_seat': ('showdown_slot', 'p2')}[mode]
                    out['p1'].metadata[key] = value
                return out
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'identity'):
                self.run_rows(rows, battle, reference_feed=reference)
            self.assertEqual(self.pairs, [])
            self.assertFalse(any(adapter.feeds for adapter in self.adapters.values()))
