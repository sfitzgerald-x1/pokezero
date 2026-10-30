"""Full fixed-roster parity with decision refusals retained as unknown cells.

A failed state update taints that owner's remaining prefix, rather than
forwarding stale state. Other owners and battles continue. File, identity and
terminal integrity checks remain fatal; encoder refusals do not end the run.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .mirror_value_probe import digest
from .projected_encoder_collection import _close_environments, _forward_input
from .projected_encoder_parity import ProjectedParityTally
from .recorded_encoder_corpus import RecordedEncoderCursor
from .recorded_encoder_inventory import inventory_collection, _inside, _json, _HashedInput, _sha
from .recorded_encoder_parity import SHAPES
from .recorded_player_encoder_runner import (
    ProjectedEncoderPair, _own_request_deltas, _reference, _snapshot_batch, VISIBILITY)
from .saved_player_projection import (
    SavedSourcePlayerProjector, new_player_reference_environments,
    feed_player_reference_boundary, PROJECTION_SCOPE, STRUGGLE_WARNING_POLICY)

SEATS = ('p1', 'p2')
INPUT_COLUMNS = tuple([f'numeric_features[{i}]' for i in range(132)] +
    [f'categorical_ids[{i}]' for i in range(41)] +
    ['token_type_ids', 'attention_mask', 'legal_action_mask'])


def refusal(error, stage, lines=()):
    """Keep indices and error type, never raw requests, arrays or error text."""
    text = str(error)
    kind, offset, line_sha = 'UNKNOWN', None, None
    match = re.search(r'kind=([A-Za-z0-9_+\-]+) offset=(\d+)', text)
    if match:
        kind, offset = match[1], int(match[2])
        if offset < len(lines):
            actual_kind = lines[offset].split('|')[1] if lines[offset].startswith('|') else 'UNKNOWN'
            kind = kind if kind == actual_kind else 'UNKNOWN'
            line_sha = hashlib.sha256(lines[offset].encode()).hexdigest()
        else:
            kind = 'UNKNOWN'
    return dict(stage=stage, line_kind=kind, line_offset=offset,
                line_sha256=line_sha, error_type=type(error).__name__)


class DiagnosticTally:
    def __init__(self, expected_views, *, parity=None, fatal_errors=()):
        self.parity = parity if parity is not None else ProjectedParityTally(expected_views=expected_views)
        if self.parity.expected_views != expected_views:
            raise ValueError('matching fixed parity denominator required')
        self.fatal_errors = fatal_errors
        self.seen = set()
        self.input_refusals = self.forward_refusals = 0
        self.by_kind, self.by_stage, self.by_column = Counter(), Counter(), Counter()
        self.records = []
        self.value_abs_sum = 0.

    def visit(self, view):
        if view.decision_id in self.seen or len(self.seen) >= self.parity.expected_views:
            raise ValueError('duplicate or excess fixed-roster decision')
        self.seen.add(view.decision_id)

    def refuse(self, view, cause, *, forward=False, propagated=False):
        self.forward_refusals += int(forward)
        self.input_refusals += int(not forward)
        self.by_kind[cause['line_kind']] += 1
        self.by_stage[cause['stage']] += 1
        columns = ('policy_logits', 'value') if forward else INPUT_COLUMNS
        self.by_column.update(columns)
        self.records.append(dict(decision_id=view.decision_id, battle_id=view.battle_id,
            seat=view.seat, decision_round_index=view.decision_round_index,
            turn_number=view.turn_number, **cause, propagated_prefix_refusal=propagated,
            refusal_scope='forward_outputs' if forward else 'whole_observation_unknown',
            # A whole-observation refusal is not evidence that every writer is defective.
            affected_columns=list(columns) if forward else 'ALL_INPUT_COLUMNS_UNKNOWN'))

    def pair(self, pair, adapter, with_forward):
        actual, reference = self.parity.pair_arrays(pair)
        kwargs = {}
        forward_error = None
        if with_forward:
            try:
                a = adapter.forward(_forward_input(actual))
                r = adapter.forward(_forward_input(reference))
                # Validate the complete paired forward before mutating its counters.
                from .recorded_encoder_parity import _forward
                _, av = _forward(a)
                _, rv = _forward(r)
                kwargs = dict(actual_forward=a, reference_forward=r)
            except Exception as error:
                if isinstance(error, self.fatal_errors):
                    raise
                forward_error = refusal(error, 'fp32_forward')
        self.parity.add(pair, **kwargs)
        if forward_error is not None:
            self.refuse(pair, forward_error, forward=True)
        if kwargs:
            self.value_abs_sum += float(abs(np.float64(av) - np.float64(rv)))

    def report(self, *, require_complete=True):
        metrics = self.parity.report()
        expected, seen = self.parity.expected_views, len(self.seen)
        if ((require_complete and seen != expected)
                or self.parity.rows + self.input_refusals != seen):
            raise ValueError('every fixed-roster decision must be compared or refused exactly once')
        if self.parity.forward_rows + self.forward_refusals not in (0, self.parity.rows):
            raise ValueError('paired forward denominator is incomplete')
        for family, count in [('numeric_columns', 132), ('categorical_columns', 41)]:
            for row in metrics[family]:
                row.update(expected_cells=expected*23, compared_cells=row['cells'],
                    refused_decisions=self.input_refusals, refused_cells=self.input_refusals*23,
                    unprocessed_decisions=expected-seen, unprocessed_cells=(expected-seen)*23,
                    refusal_scope='unknown_whole_observation_not_writer_specific',
                    compared_cell_mismatch_rate=(row['mismatched_cells']/row['cells']
                                                 if row['cells'] else None))
                if family == 'numeric_columns' and not row['cells']:
                    row['maximum_absolute_error'] = None
            if len(metrics[family]) != count:
                raise ValueError('all 173 input columns required')
        n = self.parity.forward_rows
        metrics.update(processed_views=seen, refused_input_views=self.input_refusals,
            refused_forward_views=self.forward_refusals, complete_roster_processed=seen == expected,
            unprocessed_views=expected-seen,
            model_forward_complete=n == expected,
            policy_argmax_agreement_on_compared_views=metrics['policy_argmax_agreement'],
            policy_argmax_full_roster_identification_bounds=[self.parity.argmax_equal/expected,
                (self.parity.argmax_equal+expected-n)/expected],
            legal_policy_argmax_full_roster_identification_bounds=[self.parity.legal_argmax_equal/expected,
                (self.parity.legal_argmax_equal+expected-n)/expected],
            other_tensor_expected_cells={key: expected*SHAPES[key][0]
                for key in self.parity.other_bad},
            other_tensor_refused_cells={key: self.input_refusals*SHAPES[key][0]
                for key in self.parity.other_bad},
            mean_absolute_value_difference=self.value_abs_sum/n if n else None,
            refusals_by_line_kind=dict(sorted(self.by_kind.items())),
            refusals_by_stage=dict(sorted(self.by_stage.items())),
            refusals_by_column=dict(sorted(self.by_column.items())),
            refusal_records=self.records,
            identification_bounds_are_confidence_intervals=False)
        return metrics


def compare_diagnostic_battle(rows, *, battle, stream_id, adapters, environments,
                              tally, with_forward, projector=None,
                              reference_feed=feed_player_reference_boundary):
    cursor = RecordedEncoderCursor(battle_id=battle['battle_id'], seed=battle['seed'])
    projector = projector or SavedSourcePlayerProjector(battle_id=battle['battle_id'])
    for i, seat in enumerate(SEATS):
        adapters[seat].reset(stream_id=2*stream_id+i, battle_id=battle['battle_id'])
    device_taint = dict.fromkeys(SEATS)
    reference_taint = projection_taint = None
    boundaries = views = asymmetric = source_end = 0
    request_counts = dict.fromkeys(SEATS, 0)
    for row in rows:
        if not isinstance(row, dict) or cursor.closed:
            raise ValueError('invalid or post-terminal record')
        if row.get('event') == 'end':
            cursor.close(row)
            continue
        original = cursor.advance(row)
        try:
            requests = _own_request_deltas(row)
        except ValueError as error:
            # The cursor has already validated source hashes and owning request
            # identities. Valid interleaving is unsupported by this split driver;
            # it is not corrupt source and must not end the fixed roster.
            if str(error) != 'interleaved public/request boundary requires ordered projection':
                raise
            requests = {seat: [] for seat in SEATS}
            private_seen = False
            cause = None
            for offset, line in enumerate(row['protocol_delta']):
                if line.startswith('|request|'):
                    private_seen = True
                    packet = json.loads(line[len('|request|'):])
                    requests[packet['side']['id']].append(line)
                elif private_seen and cause is None:
                    cause = dict(stage='request_order', line_kind=line.split('|')[1] or 'SEPARATOR',
                        line_offset=offset, line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        error_type=type(error).__name__,
                        origin_decision_round_index=row['decision_round_index'])
            if cause is None:
                raise ValueError('request-order refusal lacks its offending source line')
            requests = {seat: tuple(lines) for seat, lines in requests.items()}
            reference_taint = reference_taint or cause
            for seat in SEATS:
                device_taint[seat] = device_taint[seat] or cause
        seats = tuple(view.seat for view in original.views)
        references, batch = {}, None
        if projection_taint is None:
            try:
                projected = projector.project_nonrequest_delta(original.trusted_nonrequest_delta)
            except Exception as error:
                projection_taint = refusal(error, 'projection', original.trusted_nonrequest_delta)
                projection_taint['origin_decision_round_index'] = row['decision_round_index']
            else:
                # Refusals of a protocol family are measurements. A foreign
                # identity, chronology or projection contract is not a measurement.
                if (projected.scope != PROJECTION_SCOPE
                        or projected.struggle_warning_policy != STRUGGLE_WARNING_POLICY):
                    raise ValueError('exact projected source scope and policy required')
                batch = _snapshot_batch(projected, original=original, start=source_end)
        if batch is not None:
            if reference_taint is None:
                try:
                    references = reference_feed(environments, batch, own_request_deltas=requests,
                        requested_seats=seats, request_tail_verified=True)
                    if set(references) != set(seats):
                        raise ValueError('reference view roster differs')
                except Exception as error:
                    # The two production histories may have advanced partially.
                    reference_taint = refusal(error, 'reference_feed')
                    reference_taint['origin_decision_round_index'] = row['decision_round_index']
            for seat in SEATS:
                if device_taint[seat] is None:
                    lines = batch.for_seat(seat).lines
                    try:
                        adapters[seat].feed(lines)
                    except Exception as error:
                        device_taint[seat] = refusal(error, 'jax_feed', lines)
                        device_taint[seat]['origin_decision_round_index'] = row['decision_round_index']
        source_end += len(original.trusted_nonrequest_delta)
        boundaries += 1
        asymmetric += len(seats) != 2
        for seat in SEATS:
            request_counts[seat] += len(requests[seat])
        for view in original.views:
            tally.visit(view)
            views += 1
            cause = projection_taint or reference_taint or device_taint[view.seat]
            if cause is not None:
                tally.refuse(view, cause, propagated=(cause['origin_decision_round_index'] !=
                    view.decision_round_index))
                continue
            try:
                reference = _reference(references[view.seat], view)
                encoded = adapters[view.seat].encode(json.loads(view.own_request_json),
                                                    owner_side=SEATS.index(view.seat))
                actual = {}
                if set(encoded) != set(SHAPES):
                    raise ValueError('all five actual tensors required')
                for name, shape in SHAPES.items():
                    value = np.array(encoded[name], copy=True)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError('complete finite actual tensor required: '+name)
                    value.setflags(write=False)
                    actual[name] = value
                pair = ProjectedEncoderPair(view.decision_id, view.battle_id, view.seat,
                    view.decision_round_index, view.turn_number, view.source_prefix_sha256,
                    view.own_request_json, reference, digest(json.loads(view.own_reference_json)), actual)
                tally.pair(pair, adapters[view.seat], with_forward)
            except Exception as error:
                tally.refuse(view, refusal(error, 'decision_encode'))
    if not cursor.closed:
        raise ValueError('complete terminal record required')
    return dict(battle_id=battle['battle_id'], boundaries=boundaries, views=views,
        asymmetric_boundaries=asymmetric, owning_request_deltas=request_counts,
        source_prefix_sha256=cursor.source_prefix_sha256, terminal_validated_not_encoded=True)


def compare_diagnostic_collection(collection, *, expected_collection_sha256,
                                  expected_cohort_sha256, adapters, reference_config,
                                  with_forward, progress=None, checkpoint=None):
    if type(with_forward) is not bool or set(adapters) != set(SEATS) or adapters['p1'] is adapters['p2']:
        raise ValueError('two independent adapters and explicit forward mode required')
    root = Path(collection).resolve(strict=True)
    inventory = inventory_collection(root, expected_collection_sha256=expected_collection_sha256,
        expected_cohort_sha256=expected_cohort_sha256)
    manifest = _json(_inside(root, 'collection.json'), expected_collection_sha256)
    tally = DiagnosticTally(inventory['totals']['requested_views'])
    reports = []
    for i, battle in enumerate(manifest['battles']):
        receipt = _json(_inside(root, f"battle-{battle['index']:04d}/battle.json"),
            manifest['battle_receipt_hashes'][f"battle-{battle['index']:04d}/battle.json"])
        expected = receipt['files']['boundaries.jsonl.gz']
        path = _inside(root, f"battle-{battle['index']:04d}/boundaries.jsonl.gz")
        environments = new_player_reference_environments(reference_config, battle_id=battle['battle_id'])
        try:
            if set(environments) != set(SEATS) or environments['p1'] is environments['p2']:
                raise ValueError('fresh independent reference histories required')
            with path.open('rb') as raw:
                hashed = _HashedInput(raw)
                with gzip.GzipFile(fileobj=hashed, mode='rb') as stream:
                    reports.append(compare_diagnostic_battle((json.loads(line) for line in stream),
                        battle=battle, stream_id=i+1, adapters=adapters, environments=environments,
                        tally=tally, with_forward=with_forward))
                for _ in iter(lambda: hashed.read(1024*1024), b''):
                    pass
                if hashed.hasher.hexdigest() != expected:
                    raise ValueError('boundary artifact hash mismatch')
        finally:
            _close_environments(environments)
        if checkpoint is not None:
            checkpoint(dict(schema='pokezero.B2-projected-diagnostic-checkpoint.v1',
                status='PARTIAL_FIXED_ROSTER_NOT_A_COMPLETE_RESULT',
                collection_sha256=expected_collection_sha256,
                cohort_sha256=expected_cohort_sha256,
                battles_processed=i+1, expected_battles=inventory['battle_count'],
                metrics=tally.report(require_complete=False),
                actual_captured_wire=False, full_encoder_admitted=False))
        if progress is not None:
            progress(dict(battles_processed=i+1, expected_battles=inventory['battle_count'],
                processed_views=len(tally.seen), compared_views=tally.parity.rows,
                refused_input_views=tally.input_refusals,
                paired_forward_views=tally.parity.forward_rows,
                refused_forward_views=tally.forward_refusals, qualification=False))
    if (_sha(_inside(root, 'collection.json')) != expected_collection_sha256
            or _sha(_inside(root, 'cohort.json')) != expected_cohort_sha256
            or sum(r['boundaries'] for r in reports) != inventory['totals']['boundaries']
            or sum(r['asymmetric_boundaries'] for r in reports) != inventory['totals']['asymmetric_boundaries']
            or sum(sum(r['owning_request_deltas'].values()) for r in reports)
               != inventory['totals']['discarded_private_request_lines']
            or len(reports) != inventory['battle_count']):
        raise ValueError('fixed collection identity/count changed')
    metrics = tally.report()
    return dict(schema='pokezero.B2-projected-diagnostic-comparison.v1',
        status='COMPLETE_FIXED_ROSTER_WITH_REFUSALS', inventory=inventory, metrics=metrics,
        visibility=VISIBILITY, scope=PROJECTION_SCOPE, battles=reports,
        new_games=0, simulator_steps=0, actual_captured_wire=False, full_encoder_admitted=False,
        limitation='Source-projected owning references, not captured player wire. '
            'Refused prefixes are unknown, never stale-state forwards; report all denominators.')
