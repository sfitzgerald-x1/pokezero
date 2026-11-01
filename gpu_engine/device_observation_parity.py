"""Nonfatal public29 comparison on independently selected state-agreement views.

This reducer never decides engine/Showdown agreement from encoded tensors. Its
caller measures represented state/chance agreement first and supplies actual
B2-protocol-fed public29 reference tensors. Every attempted decision stays in
the ledger; an unsupported line or malformed view cannot abort the cohort.
Numerical checks alone do not attest collection provenance or pass all B4 gates.
"""
from collections import Counter, defaultdict
import numpy as np


SHAPES = dict(numeric_features=((29, 132), 'fiu'), categorical_ids=((29, 41), 'iu'),
    token_type_ids=((29,), 'iu'), attention_mask=((29,), 'b'),
    legal_action_mask=((9,), 'b'), switch_targets=((5,), 'iu'))


class AgreementParity:
    def __init__(self, *, source_bindings):
        if not isinstance(source_bindings, dict) or not source_bindings:
            raise ValueError('source identities required before any collection')
        self.bindings = dict(source_bindings)
        self.ids = set(); self.total = Counter(); self.turns = defaultdict(Counter)
        self.refusals = Counter(); self.kinds = defaultdict(Counter)
        self.numeric_bad = np.zeros((29, 132), np.int64)
        self.category_bad = np.zeros((29, 41), np.int64)
        self.numeric_max = np.zeros((29, 132), np.float64)
        self.numeric_nonfinite = np.zeros((29, 132), np.int64)
        self.structure_bad = Counter()
        self.column_kind = Counter()

    def _refuse(self, source, reason, kinds, column='ALL'):
        for kind in kinds:self.refusals[(source, kind, column, reason)] += 1

    def add(self, *, decision_id, turn, line_kinds, agreement,
            reference=None, candidate=None, refusal=None):
        """One source-recorded view. False/None agreement is never token-scored.

        ``line_kinds`` comes from this actual Showdown boundary's new lines;
        multi-kind counts overlap. Refusals retain source/kind/column/reason.
        Top-level cohort/source configuration errors remain fatal at startup.
        """
        kinds = tuple(sorted({k for k in line_kinds if isinstance(k, str) and k})) if isinstance(line_kinds, (tuple, list)) else ()
        kinds = kinds or ('NO_NEW_LINE_KIND',)
        self.total['attempted'] += 1
        if (not isinstance(decision_id, str) or not decision_id or decision_id in self.ids
                or type(turn) is not int or turn < 0):
            self.total['metadata_refused'] += 1
            self._refuse('collection', 'MISSING_OR_DUPLICATE_DECISION_METADATA', kinds)
            return
        self.ids.add(decision_id); ledger = self.turns[turn]; ledger['attempted'] += 1
        if agreement is not True:
            key = 'state_disagreement' if agreement is False else 'state_agreement_unknown'
            self.total[key] += 1; ledger[key] += 1
            return
        self.total['agreement'] += 1; ledger['agreement'] += 1
        for kind in kinds:self.kinds[kind]['agreement'] += 1
        if refusal is not None:
            self.total['comparison_refused'] += 1; ledger['comparison_refused'] += 1
            source = str(refusal.get('source', 'reference')) if isinstance(refusal, dict) else 'reference'
            reason = str(refusal.get('reason', refusal)) if isinstance(refusal, dict) else str(refusal)
            column = str(refusal.get('column', 'ALL')) if isinstance(refusal, dict) else 'ALL'
            self._refuse(source, reason, kinds, column)
            return
        decoded = []
        for source, record in (('reference', reference), ('candidate', candidate)):
            arrays = {}; problems = []
            if not isinstance(record, dict):
                problems.append(('ALL', 'MISSING_VIEW'))
            else:
                for name, (shape, dtype) in SHAPES.items():
                    try:
                        value = np.asarray(record[name])
                        if value.shape != shape or value.dtype.kind not in dtype:
                            raise ValueError('shape or dtype')
                        arrays[name] = value
                    except (KeyError, ValueError, TypeError, OverflowError):
                        problems.append((name, 'MALFORMED_VIEW_FIELD'))
                if 'admitted' in record:
                    try:
                        admitted = np.asarray(record['admitted'])
                        if admitted.shape != () or admitted.dtype.kind != 'b':
                            problems.append(('ALL', 'MALFORMED_OR_UNKNOWN_ADMISSION'))
                        elif not bool(admitted):
                            problems.append(('ALL', str(record.get('reason', 'PRODUCER_REFUSED'))))
                    except (TypeError, ValueError, OverflowError):
                        problems.append(('ALL', 'MALFORMED_OR_UNKNOWN_ADMISSION'))
            if problems:
                self.total['comparison_refused'] += 1; ledger['comparison_refused'] += 1
                for column, reason in problems:self._refuse(source, reason, kinds, column)
                return
            decoded.append(arrays)
        ref, got = decoded
        left = ref['numeric_features'].astype(np.float64)
        right = got['numeric_features'].astype(np.float64)
        finite = np.isfinite(left) & np.isfinite(right)
        with np.errstate(invalid='ignore', over='ignore'):
            delta = np.where(finite, np.abs(left-right), 0.)
        nb = ~finite | (delta > 1e-6)
        cb = (ref['categorical_ids'] != got['categorical_ids']) | (got['categorical_ids'] < 0) | (ref['categorical_ids'] < 0)
        self.numeric_bad += nb; self.category_bad += cb
        self.numeric_nonfinite += ~finite | ~np.isfinite(delta)
        self.numeric_max = np.maximum(self.numeric_max, np.where(np.isfinite(delta), delta, 0.))
        structure = {key: not np.array_equal(ref[key], got[key])
            for key in SHAPES if key not in ('numeric_features', 'categorical_ids')}
        self.structure_bad.update({key: int(value) for key, value in structure.items()})
        for kind, bad in (('numeric', nb), ('categorical', cb)):
            for column in np.flatnonzero(np.any(bad, axis=0)):
                for line in kinds:self.column_kind[(kind, int(column), line)] += 1
        joint = not (nb.any() or cb.any() or any(structure.values()))
        self.total['compared'] += 1; ledger['compared'] += 1
        self.total['joint_match'] += int(joint); ledger['joint_match'] += int(joint)
        for kind in kinds:
            self.kinds[kind]['compared'] += 1; self.kinds[kind]['joint_match'] += int(joint)

    def report(self):
        columns = []
        for kind, bad in (('numeric', self.numeric_bad), ('categorical', self.category_bad)):
            for column in range(bad.shape[1]):
                value = dict(kind=kind, column=column, compared_views=self.total['compared'],
                    compared_cells=29*self.total['compared'], mismatched_cells=int(bad[:, column].sum()),
                    token_mismatched_cells=bad[:, column].tolist())
                if kind == 'numeric':value.update(tolerance=1e-6,
                    max_finite_absolute_difference=float(self.numeric_max[:, column].max()),
                    nonfinite_cells=int(self.numeric_nonfinite[:, column].sum()))
                else:value['tolerance'] = 0
                columns.append(value)
        denominator = self.total['agreement']
        ratio = self.total['joint_match']/denominator if denominator else None
        late = sum(row['compared'] for turn, row in self.turns.items() if turn >= 10)
        enough = self.total['compared'] >= 10000
        late_enough = late >= 1000
        return dict(schema='pokezero.engine-device-public29-agreement-parity.v1',
            source_bindings=self.bindings, counts=dict(self.total), columns=columns,
            joint_match_fraction_on_ALL_agreement_views=ratio,
            joint_denominator_scope='All independently state-agreeing attempted views; comparison refusals stay in denominator. Zero/deferred columns are included.',
            views_by_turn=[dict(turn=k, **dict(v)) for k, v in sorted(self.turns.items())],
            compared_views_turn10_or_later=late,
            minimum_compared_views_met=enough, minimum_late_views_met=late_enough,
            numerical_gate_only=bool(enough and late_enough and ratio is not None and ratio >= .99
                and not self.total['metadata_refused']),
            collection_provenance_independently_verified=False, full_B4_pass=False,
            structure_mismatched_views=dict(self.structure_bad),
            line_kind_tables=[dict(kind=k, **dict(v)) for k, v in sorted(self.kinds.items())],
            refusals_by_kind_column=[dict(source=s, kind=k, column=c, reason=r, views=n)
                for (s, k, c, r), n in sorted(self.refusals.items())],
            mismatched_views_by_column_kind=[dict(column_kind=k, column=c, line_kind=l, views=n)
                for (k, c, l), n in sorted(self.column_kind.items())],
            kind_count_scope='Line kinds and column mismatch kinds are nonexclusive; do not sum as disjoint views.',
            qualification='NUMERICAL_ONLY; live engine/Showdown physical agreement, actual owning protocol references, fixed cohort provenance and independent result review required separately.')
