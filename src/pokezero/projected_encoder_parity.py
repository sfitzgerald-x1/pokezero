"""All-cell metrics for regenerated owning-player references.

Original trusted references supply lineage hashes only. These metrics neither
attest captured player bytes nor qualify the runtime/source or full encoder.
"""
import json
import re

import numpy as np

from .mirror_value_probe import canonical, digest
from .recorded_encoder_corpus import _Prefix
from .recorded_encoder_parity import SHAPES, _arrays, _forward
from .recorded_mirror_roots import recorded_requested_players
from .recorded_player_encoder_runner import ProjectedEncoderPair, VISIBILITY
from .saved_player_projection import PROJECTION_SCOPE, STRUGGLE_WARNING_POLICY, PINNED_SHOWDOWN_REVISION


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _packet(blob):
    """Canonical bytes reject duplicate JSON members as well as ambiguity."""
    if not isinstance(blob, bytes):
        raise ValueError('canonical JSON bytes required')
    try:
        value = json.loads(blob)
        valid = isinstance(value, dict) and canonical(value) == blob
    except (ValueError, TypeError, UnicodeDecodeError):
        valid = False
    if not valid:
        raise ValueError('canonical JSON object required')
    return value


def projected_pair_arrays(pair):
    """Validate the projected envelope before exposing exactly five tensors."""
    if (not isinstance(pair, ProjectedEncoderPair) or pair.visibility != VISIBILITY
            or pair.actual_captured_wire is not False or pair.full_encoder_admitted is not False
            or not _sha(pair.decision_id) or not _sha(pair.source_prefix_sha256)
            or not _sha(pair.original_reference_sha256)
            or not isinstance(pair.battle_id, str) or not pair.battle_id.strip()
            or '\n' in pair.battle_id or '\r' in pair.battle_id or pair.seat not in ('p1', 'p2')
            or type(pair.decision_round_index) is not int or pair.decision_round_index < 0
            or type(pair.turn_number) is not int or pair.turn_number < 0):
        raise ValueError('valid unqualified projected owning pair required')
    request = _packet(pair.own_request_json)
    if (not isinstance(request.get('side'), dict) or request['side'].get('id') != pair.seat
            or recorded_requested_players({pair.seat: request}) != (pair.seat,)):
        raise ValueError('actionable owning request required')
    reference = _packet(pair.regenerated_reference_json)
    if (set(reference) != set(SHAPES) | {'turn_number'}
            or type(reference['turn_number']) is not int or reference['turn_number'] != pair.turn_number):
        raise ValueError('complete regenerated owning reference turn required')
    return (_arrays(pair.actual, device=True),
            _arrays({key: reference[key] for key in SHAPES}, device=False))


class ProjectedParityTally:
    """Atomic per-view accumulation, separate from trusted-source ParityTally."""
    def __init__(self, *, expected_views, pair_arrays=None, reference_json=None,
                 reference_hash_key='regenerated_reference_sha256'):
        if type(expected_views) is not int or expected_views <= 0:
            raise ValueError('fixed positive expected view count required')
        self.expected_views = expected_views
        # Saved actual captures use their own strict envelope validator and
        # reference accessor; the projected default remains unchanged.
        self.pair_arrays = pair_arrays or projected_pair_arrays
        self.reference_json = reference_json or (lambda pair: pair.regenerated_reference_json)
        self.reference_hash_key = reference_hash_key
        self.ids, self.roles, self.rounds, self.latest = set(), set(), {}, {}
        self.rows = 0
        self.numeric_bad = np.zeros(132, np.int64)
        self.numeric_max = np.zeros(132, np.float64)
        self.categorical_bad = np.zeros(41, np.int64)
        self.other_bad = {key: 0 for key in ('token_type_ids', 'attention_mask', 'legal_action_mask')}
        self.forward_rows = self.argmax_equal = self.legal_argmax_equal = 0
        self.value_max = 0.
        self.owner_views = {'p1': 0, 'p2': 0}
        self.lineage = _Prefix()

    def add(self, pair, *, actual_forward=None, reference_forward=None):
        got, ref = self.pair_arrays(pair)
        role = (pair.battle_id, pair.seat, pair.decision_round_index)
        root = (pair.battle_id, pair.decision_round_index)
        owner = (pair.battle_id, pair.seat)
        signature = (pair.source_prefix_sha256, pair.turn_number)
        if (self.rows >= self.expected_views or pair.decision_id in self.ids or role in self.roles
                or pair.decision_round_index <= self.latest.get(owner, -1)
                or (root in self.rounds and self.rounds[root] != signature)):
            raise ValueError('duplicate, out-of-order or mismatched paired owning identity')
        difference = np.abs(got['numeric_features'].astype(np.float64) - ref['numeric_features'].astype(np.float64))
        numeric_bad = np.sum(difference > 1e-6, axis=0)
        numeric_max = np.max(difference, axis=0)
        categorical_bad = np.sum(got['categorical_ids'] != ref['categorical_ids'], axis=0)
        other_bad = {key: int(np.count_nonzero(got[key] != ref[key])) for key in self.other_bad}
        if (actual_forward is None) != (reference_forward is None):
            raise ValueError('both or neither FP32 forwards required')
        forward = None
        if actual_forward is not None:
            a, av = _forward(actual_forward)
            r, rv = _forward(reference_forward)
            ai = int(np.argmax(np.where(got['legal_action_mask'], a, -np.inf)))
            ri = int(np.argmax(np.where(ref['legal_action_mask'], r, -np.inf)))
            forward = (int(np.argmax(a)) == int(np.argmax(r)), ai == ri,
                       float(abs(np.float64(av) - np.float64(rv))))
        lineage = self.lineage.append([dict(decision_id=pair.decision_id, battle_id=pair.battle_id,
            seat=pair.seat, decision_round_index=pair.decision_round_index,
            source_prefix_sha256=pair.source_prefix_sha256, own_request_sha256=digest(_packet(pair.own_request_json)),
            **{self.reference_hash_key: digest(_packet(self.reference_json(pair)))},
            original_reference_sha256=pair.original_reference_sha256)])
        # All refusal paths precede accumulator mutation.
        self.ids.add(pair.decision_id)
        self.roles.add(role)
        self.rounds[root] = signature
        self.latest[owner] = pair.decision_round_index
        self.rows += 1
        self.owner_views[pair.seat] += 1
        self.lineage = lineage
        self.numeric_bad += numeric_bad
        self.numeric_max = np.maximum(self.numeric_max, numeric_max)
        self.categorical_bad += categorical_bad
        for key, value in other_bad.items():
            self.other_bad[key] += value
        if forward is not None:
            self.forward_rows += 1
            self.argmax_equal += forward[0]
            self.legal_argmax_equal += forward[1]
            self.value_max = max(self.value_max, forward[2])

    def report(self):
        complete = self.rows == self.expected_views
        enough = self.rows >= 10000
        inputs_equal = not (self.numeric_bad.any() or self.categorical_bad.any() or any(self.other_bad.values()))
        forward_complete = self.forward_rows == self.rows and self.rows > 0
        agreement = self.argmax_equal / self.forward_rows if self.forward_rows else None
        legal_agreement = self.legal_argmax_equal / self.forward_rows if self.forward_rows else None
        input_met = complete and enough and inputs_equal
        forward_met = (complete and enough and forward_complete and agreement >= .999
                       and legal_agreement >= .999 and self.value_max < 1e-4)
        return dict(schema='pokezero.B2-projected-player-comparison-metrics.v1', visibility=VISIBILITY,
            status='PROJECTED_PLAYER_METRICS_ONLY_REQUIRES_RUNTIME_ATTESTATION',
            scope=PROJECTION_SCOPE, struggle_warning_policy=STRUGGLE_WARNING_POLICY,
            showdown_revision=PINNED_SHOWDOWN_REVISION,
            expected_views=self.expected_views, views=self.rows, owning_views=dict(self.owner_views),
            lineage_sha256=self.lineage.sha256(), original_reference_values_compared=False,
            complete_view_count=complete, minimum10000_views_met=enough,
            numeric_columns=[dict(index=i, cells=self.rows*23, mismatched_cells=int(self.numeric_bad[i]),
                maximum_absolute_error=float(self.numeric_max[i]), tolerance=1e-6) for i in range(132)],
            categorical_columns=[dict(index=i, cells=self.rows*23, mismatched_cells=int(self.categorical_bad[i]),
                comparison='EXACT') for i in range(41)], other_tensor_mismatched_cells=dict(self.other_bad),
            other_tensor_cells={key: self.rows * SHAPES[key][0] for key in self.other_bad},
            projected_input_numeric_criteria_met=input_met, model_forward_views=self.forward_rows,
            model_forward_complete=forward_complete, policy_argmax_agreement=agreement,
            legal_policy_argmax_agreement=legal_agreement, value_tolerance=1e-4,
            maximum_value_difference=self.value_max if self.forward_rows else None,
            measured_value_criterion_met=self.value_max < 1e-4 if self.forward_rows else None,
            projected_forward_numeric_criteria_met=forward_met,
            projected_comparison_numeric_criteria_met=input_met and forward_met,
            actual_captured_wire=False, captured_player_wire=False, owning_stream_bytes_reconstructed=False,
            public_wire_qualified=False, fullencoder=False, full_encoder_admitted=False)
