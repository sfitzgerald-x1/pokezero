"""Depth-2 exactness gate: Depth2Pricer(digest=True) against the independent oracle and D's FramePricer.

Roots come from either the 40 saved D3 cost roots (--saved-roots, built as d3_saved_root_gate
--batched-prep builds them: device_d_cost.run's frames, worlds.detach_roots, the original Recipe; the
serial Provider is NOT run, its label_chunk only captures the requests) or one D3 bank chunk (--d3-root
--chunk N, built as d3_chunk_equal builds it, then Provider._prepare_batched, grouped by frame). Every
group of requests is priced by gpu_engine.device_depth2_pricer.Depth2Pricer(digest=True), by
FramePricer(vectorized=True, host_prep=True) and by gpu_engine.depth2_oracle on the SAME requests,
materializer, recipe and params. --worlds-blocks K adds K-1 recipe blocks (U1's world_block_seeds,
quarter selection bypassed as in U1); each block is a separate call to all three.

Checks (gate.json; exit code 0 iff all pass):
 (a) exact: every oracle-enumerated turn-1 child and every checked turn-2 leaf has the same status,
     error code (including the depth-2-only child codes 5/6), terminal value and leaf digest; children
     also the same own/peer digests, legal masks and request flags; roots the same pre code (-1 for a
     world without a state) and opponent request/legal. The first 20 mismatches are listed.
 (b) D (not bitwise for floats): every discrete field exactly D's (pre and refused worlds, root opponent
     request/legal, status/error wherever the pricer's code is D's 0-4, the turn-1 pair set), and
     |t1_value - D| and |root prior - D| <= D_VALUE_TOLERANCE on every D-priced cell (codes 5/6 included:
     they keep D's value); the bit-equal fraction, max |diff| and first inexact cells are reported.
 (c) heads, against the oracle's own bf16 heads (a different batch geometry): t2_value and t1_own_logits
     within HEAD_TOLERANCE, D3's declared cross-geometry target tolerance (HEAD_TOLERANCE_SOURCE);
     t1_own_logits is gated here because FramePricer exposes no child logits. Reported per quantity:
     max |diff|, exact fraction, the depth-1 noise floor max |D t1_value - oracle t1 head| and the ratio
     t2_value max / floor.
 (d) enumeration: the pricer's enumerated turn-1 and turn-2 sets equal the oracle's.
--sample-leaves N checks all turn-1 children plus N uniformly sampled turn-2 leaves per world (fixed
SAMPLE_SEED); (d) always covers every leaf. --child-block N is passed to Depth2Pricer. --block-invariance
prices every call twice, with child_block 27 and 81 (the 81 frame is the one checked), and requires
every output array to be bit-equal (on GPU the turn-2 forward batch changes with child_block). Reported, not gated: the depth-2 root refusal share under the
reference rule (depth2_backups.refusal_summary when importable, else the gate's own count of the frame)
next to D3 FramePricer's on the same requests, and turn-1/turn-2 error counts by code. All paths are command-line arguments; --out must be new.
"""
import argparse
import collections
from contextlib import contextmanager, nullcontext
import dataclasses
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCHEMA = 'pokezero.depth2-gate.v1'
D_VALUE_TOLERANCE = 1e-5   # (b): |t1_value - D| and |root prior - D|; discrete fields and digests stay exact
HEAD_TOLERANCE = .01       # (c): D3's declared cross-geometry target tolerance
HEAD_TOLERANCE_SOURCE = ('gpu_engine/device_d_frame_parity.py:94-95 (vectorized_comparison: D3 frame-batched vs serial '
    'policy/value targets must agree within .01 absolute)')
PROGRESS_INTERVAL = 10.    # seconds between oracle chunk progress lines (first and last chunk always logged)
FIRST = 20
SAMPLE_SEED = 20261105
BLOCK_DOMAIN = 'pokezero.d3-u1.world-block.v1'   # U1's (d3_u1_completed_q.world_block_seeds) domain, unchanged
NO_STATE = -1                  # pre of a world without a state
DEPTH2_CODES = (5, 6)          # depth-2-only child codes: opponent child view encoder error / unusable requested prior
INPUT_ORDER = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')
# D's reason strings (device_public_teacher_batch.FramePricer._result) -> codes.
D_ERRORS = {'hypothetical original action route invalid': 1, 'hypothetical terminal winner unavailable': 2,
    'current direct-encoder input error': 3, 'current public actor output refused/nonfinite': 4}
D_PRE = {'current direct-encoder input error': (1, 8), 'same actual root legal-action denominator required': (2,),
    'current public actor output refused/nonfinite': (9,), 'actual learner root input fields differ': (10,)}
SHAPES = dict(pre=(2,), root_peer_requested=(2,), root_peer_legal=(2, 9), root_peer_logits=(2, 9),
    t1_status=(2, 9, 9), t1_value=(2, 9, 9), t1_error=(2, 9, 9), t1_own_requested=(2, 9, 9), t1_own_legal=(2, 9, 9, 9),
    t1_own_logits=(2, 9, 9, 9), t1_peer_requested=(2, 9, 9), t1_peer_legal=(2, 9, 9, 9), t1_peer_logits=(2, 9, 9, 9),
    t2_status=(2, 9, 9, 9, 9), t2_value=(2, 9, 9, 9, 9), t2_error=(2, 9, 9, 9, 9),
    t2_digest=(2, 9, 9, 9, 9, 2), t1_digest_own=(2, 9, 9, 2), t1_digest_peer=(2, 9, 9, 2))
CONVENTIONS = (
    'Root: the opponent is requested iff any(its view legal_action_mask), as D; the owner always acts at turn 1.',
    'Child: a seat is requested iff any(executor.legal(child)[0, seat]); its legal actions are its encoder mask (observe(...).observation.legal_action_mask, the request-order lanes of logits and dispatch), never the engine mask; a waiting seat is dispatched -1 and enumerated as action 0 only.',
    'Turn-2 chance is executor.sample(child state); turn-1 chance is executor.sample(root state).',
    'Child codes: D\'s 1-4 first (transition, owner view, learner head); then, on a non-terminal child where the opponent is requested, 5 if its child view has an encoder error, else 6 if its prior is unusable; a waiting opponent never refuses a cell. Codes 5/6 keep D\'s learner value and are not expanded.',
    'pre is -1 for a world without a state (materialization refusal); such a world enumerates nothing.',
    'Digests are compared where the scored view is meaningful: status 2/4, or error 3-6 (own view) and 5/6 (opponent view); terminal and invalid cells compare status, error and value only.',
    'Child masks, request flags and the opponent digest are compared on expanded children and codes 5/6.',
    'Logits are compared on legal lanes of requested seats only.',
    'Exact: every discrete field and digest. Floats: (b) t1_value and the root prior within D_VALUE_TOLERANCE of D; (c) head outputs within HEAD_TOLERANCE of the oracle\'s bf16 heads, whose batch geometry differs (D3\'s cross-geometry target tolerance).',
    'Top-k ranks by the oracle\'s own bf16 child priors (logit descending, lower index on ties, non-finite last); full width needs no priors.',
    'Each world block is a separate call (Depth2Pricer.price takes one recipe); blocks after 0 bypass Recipe.selected, as U1.',
    'Depth-2 refusal (reference rule): a root refuses if any world has pre != 0, or any enumerated turn-1 or turn-2 cell has status 3 (codes 5/6 included).',
)


def log(message, **values):
    print(json.dumps(dict(message=message, time=time.strftime('%H:%M:%S'), **values), default=str), flush=True)


def world_block_seeds(seed, blocks):
    """U1's world blocks: block 0 is the declared recipe seed, later blocks fixed derivations of it."""
    if type(seed) is not int or type(blocks) is not int or blocks < 1:raise ValueError('declared seed and block count required')
    return (seed,) + tuple(int.from_bytes(hashlib.sha256(json.dumps([BLOCK_DOMAIN, seed, k]).encode()).digest()[:8], 'big')
        for k in range(1, blocks))


def field(frame, name):
    return np.asarray(frame[name] if isinstance(frame, dict) else getattr(frame, name))


def _plain(x):
    x = np.asarray(x)
    if x.dtype.kind == 'f':
        return [_plain(v) for v in x] if x.ndim else (float(x) if math.isfinite(float(x)) else str(float(x)))
    return x.tolist()


def _bits(x):
    return np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)


def _rows(equal):
    equal = np.asarray(equal, bool)
    if equal.ndim <= 1 or not len(equal):return equal.reshape(len(equal))
    return equal.reshape(len(equal), -1).all(axis=1)


class Findings:
    """Per-field compared/mismatched counts and the first FIRST mismatches (with call context)."""
    def __init__(self, context=None):
        self.context = dict(context or {});self.compared = collections.Counter();self.mismatched = collections.Counter();self.first = []

    def record(self, name, keys, equal, got, want):
        equal = _rows(equal);self.compared[name] += len(equal);self.mismatched[name] += 0
        if not len(equal):return
        keys = np.asarray(keys).reshape(len(equal), -1);bad = np.flatnonzero(~equal)
        self.fail(name, keys[bad], np.asarray(got)[bad], np.asarray(want)[bad])

    def fail(self, name, keys, got, want):
        self.mismatched[name] += len(keys)
        for i in range(min(len(keys), max(0, FIRST - len(self.first)))):
            self.first.append(dict(self.context, field=name, cell=np.asarray(keys[i]).tolist(), pricer=_plain(got[i]), oracle=_plain(want[i])))

    def report(self):
        return dict(passed=sum(self.mismatched.values()) == 0, compared=dict(self.compared), mismatched=dict(self.mismatched),
            first_mismatches=self.first)


def merge_findings(reports):
    compared, mismatched, first = collections.Counter(), collections.Counter(), []
    for r in reports:
        compared.update(r['compared']);mismatched.update(r['mismatched']);first.extend(r['first_mismatches'])
    return dict(passed=sum(mismatched.values()) == 0, compared=dict(compared), mismatched=dict(mismatched), first_mismatches=first[:FIRST])


def frame_geometry(frame, requests):
    problems = []
    for name, shape in SHAPES.items():
        try:value = field(frame, name)
        except (KeyError, AttributeError):problems.append(name + ':missing');continue
        if value.shape != (requests,) + shape:problems.append(f'{name}:shape {value.shape} != {(requests,) + shape}')
    return problems


def check_exact(frame, data, findings):
    """(a) status, error, terminal value and digests of every oracle cell/checked leaf; root pre and opponent request."""
    roots, children, leaves = data['roots'], data['children'], data['leaves']
    count = field(frame, 'pre').shape[0];keys = [(q, w) for q in range(count) for w in (0, 1)]
    if keys:
        k = np.asarray(keys).reshape(-1, 2);want = np.asarray([roots[x]['pre'] if x in roots else NO_STATE for x in keys])
        got = field(frame, 'pre')[k[:, 0], k[:, 1]];findings.record('pre', k, got.astype(np.int64) == want, got, want)
        usable = k[want == 0]
        if len(usable):
            for name, key in (('root_peer_requested', 'peer_requested'), ('root_peer_legal', 'peer_legal')):
                want = np.asarray([roots[(q, w)][key] for q, w in usable.tolist()], bool)
                got = field(frame, name)[usable[:, 0], usable[:, 1]].astype(bool);findings.record(name, usable, got == want, got, want)
    cells = children['cells']
    if len(cells):
        at = tuple(cells.T);status, error = children['status'], children['error']
        for name, want in (('t1_status', status), ('t1_error', error)):
            got = field(frame, name)[at];findings.record(name, cells, got.astype(np.int64) == want, got, want)
        terminal = status == 1;got = field(frame, 't1_value')[at]
        findings.record('t1_value_terminal', cells[terminal], (got == children['value'])[terminal], got[terminal], children['value'][terminal])
        viewed = (status == 2) | ((status == 3) & np.isin(error, (3, 4) + DEPTH2_CODES))
        got = field(frame, 't1_digest_own')[at].astype(np.uint32)
        findings.record('t1_digest_own', cells[viewed], (got == children['own_digest'])[viewed], got[viewed], children['own_digest'][viewed])
        expanded = (status == 2) | ((status == 3) & np.isin(error, DEPTH2_CODES))
        for name, key, kind in (('t1_digest_peer', 'peer_digest', np.uint32), ('t1_own_legal', 'own_legal', bool),
                ('t1_peer_legal', 'peer_legal', bool), ('t1_own_requested', 'own_requested', bool), ('t1_peer_requested', 'peer_requested', bool)):
            got = field(frame, name)[at].astype(kind);want = np.asarray(children[key]).astype(kind)
            findings.record(name, cells[expanded], (got == want)[expanded], got[expanded], want[expanded])
    specs = leaves['specs']
    if len(specs):
        at = tuple(specs.T);status, error = leaves['status'], leaves['error']
        for name, want in (('t2_status', status), ('t2_error', error)):
            got = field(frame, name)[at];findings.record(name, specs, got.astype(np.int64) == want, got, want)
        terminal = status == 1;got = field(frame, 't2_value')[at]
        findings.record('t2_value_terminal', specs[terminal], (got == leaves['value'])[terminal], got[terminal], leaves['value'][terminal])
        viewed = (status == 4) | ((status == 3) & np.isin(error, (3, 4)))
        got = field(frame, 't2_digest')[at].astype(np.uint32)
        findings.record('t2_digest', specs[viewed], (got == leaves['digest'])[viewed], got[viewed], leaves['digest'][viewed])
    return findings


def check_enumeration(frame, data, findings):
    """(d) the pricer's enumerated sets (status != 0) equal the oracle's, at both turns."""
    for turn, name, oracle in ((1, 't1_status', data['children']['cells']), (2, 't2_status', data['enumerated'])):
        pricer = set(map(tuple, np.argwhere(field(frame, name) != 0).tolist()))
        expected = set(map(tuple, np.asarray(oracle, np.int64).reshape(-1, 2 * turn + 2).tolist()))
        findings.compared[f't{turn}_enumerated'] += len(pricer | expected)
        for label, cells, got in ((f't{turn}_missing_from_pricer', sorted(expected - pricer), 'not enumerated'),
                (f't{turn}_extra_in_pricer', sorted(pricer - expected), 'enumerated')):
            other = 'enumerated' if got == 'not enumerated' else 'not enumerated'
            findings.fail(label, cells, [got] * len(cells), [other] * len(cells))
    return findings


def _pre_codes(reason, request):
    reason = reason.removeprefix('ValueError: ')
    if reason in D_PRE:return D_PRE[reason]
    prefix = 'actual learner root input differs: '
    if reason.startswith(prefix):
        order = tuple(k for k in INPUT_ORDER if k in request.root_inputs);key = reason[len(prefix):]
        if key in order:return (3 + order.index(key),)
    return ()


def check_d(frame, d_results, requests, findings, *, full_turn1=True):
    """(b) the pricer's turn 1 against FramePricer's own Results: bit-equal values, D's errors and root prior."""
    pre, t1s, t1e, t1v = (field(frame, n) for n in ('pre', 't1_status', 't1_error', 't1_value'))
    requested, legal, logits = (field(frame, n) for n in ('root_peer_requested', 'root_peer_legal', 'root_peer_logits'))
    if len(d_results) != len(requests):
        findings.record('d_results', [[len(requests)]], [False], [len(d_results)], [len(requests)]);return findings
    rows, numeric = collections.defaultdict(list), collections.defaultdict(list)
    for q, (request, result) in enumerate(zip(requests, d_results)):
        for wi, report in enumerate(result.world_reports):
            grid = report.get('joint_values');enumerated = t1s[q, wi] != 0
            if grid is None:
                rows['d_refused_world_unenumerated'].append(((q, wi), not enumerated.any(), int(enumerated.sum()), 0))
                reason = report.get('reason') or ''
                if reason.startswith('physical-materialization'):
                    rows['pre_vs_d'].append(((q, wi), int(pre[q, wi]) == NO_STATE, int(pre[q, wi]), NO_STATE))
                else:
                    codes = _pre_codes(reason, request)
                    rows['pre_vs_d'].append(((q, wi), int(pre[q, wi]) in codes, int(pre[q, wi]), list(codes) or reason))
                continue
            rows['pre_vs_d'].append(((q, wi), int(pre[q, wi]) == 0, int(pre[q, wi]), 0))
            rows['root_peer_requested_vs_d'].append(((q, wi), bool(requested[q, wi]) == bool(report['opponent_requested']),
                bool(requested[q, wi]), bool(report['opponent_requested'])))
            want = np.asarray(report['reply_legal'], bool)
            rows['root_peer_legal_vs_d'].append(((q, wi), np.array_equal(legal[q, wi].astype(bool), want), legal[q, wi].astype(bool), want))
            want = np.asarray(report['reply_logits'], np.float32)
            numeric['root_peer_logits'].extend(((q, wi, lane), logits[q, wi, lane], want[lane]) for lane in range(9))
            errors = {(a, r): D_ERRORS.get(reason.removeprefix('ValueError: '), -1) for a, r, reason in report.get('errors', ())}
            pairs = sorted(set(errors) | {(a, r) for a in range(9) for r in range(9) if grid[a][r] is not None})
            if full_turn1:
                got = sorted(map(tuple, np.argwhere(enumerated).tolist()))
                rows['t1_enumeration_vs_d'].append(((q, wi), got == pairs, len(got), len(pairs)))
            for a, r in pairs:
                if not enumerated[a, r]:continue   # pruned by k1 widths; the enumeration itself is check (d)
                status, code, cell = int(t1s[q, wi, a, r]), int(t1e[q, wi, a, r]), (q, wi, a, r)
                d_code = code not in DEPTH2_CODES   # status/error are D's only where the pricer's code is D's (0-4)
                if (a, r) in errors:
                    if d_code:rows['t1_error_vs_d'].append((cell, status == 3 and code == errors[(a, r)], [status, code], [3, errors[(a, r)]]))
                    continue
                numeric['t1_value'].append((cell, t1v[q, wi, a, r], grid[a][r]))   # D priced it: D's value, under codes 5/6 too
                if d_code:rows['t1_status_vs_d'].append((cell, status in (1, 2) and code == 0, [status, code], '1|2 with code 0'))
                else:rows['t1_depth2_code_status'].append((cell, status == 3, [status, code], '3 under code 5/6'))
    for name, items in rows.items():
        keys, equal, got, want = zip(*items)
        findings.record(name, [list(k) for k in keys], np.asarray(equal, bool), _objects(got), _objects(want))
    findings.values = {}
    for name in ('t1_value', 'root_peer_logits'):
        cells, got, want = zip(*numeric[name]) if numeric[name] else ((), (), ())
        summary, beyond = _numeric(cells, got, want, D_VALUE_TOLERANCE, findings.context, list_inexact=True)
        findings.values[name] = summary
        if len(beyond):findings.record(name + '_vs_d_beyond_tolerance', list(cells), ~beyond, np.asarray(got, np.float32), np.asarray(want, np.float32))
    return findings


def _numeric(cells, got, want, tolerance, context=None, *, list_inexact=False):
    """Float agreement on matching cells: (summary, beyond-tolerance mask). Bit-equal cells (NaN payloads
    included) always agree; otherwise both must be finite and within tolerance."""
    got, want = np.asarray(got, np.float32).reshape(-1), np.asarray(want, np.float32).reshape(-1)
    cells = np.asarray(cells, np.int64).reshape(len(got), -1) if len(got) else np.zeros((0, 1), np.int64)
    exact = _bits(got) == _bits(want);finite = np.isfinite(got) & np.isfinite(want)
    delta = np.where(finite, np.abs(got.astype(np.float64) - want.astype(np.float64)), np.inf)
    beyond = ~exact & ~(delta <= tolerance)
    row = lambda i: dict(context or {}, cell=cells[i].tolist(), pricer=_plain(got[i]), reference=_plain(want[i]))
    summary = dict(compared=int(len(got)), exact_equal=int(np.count_nonzero(exact)), nonfinite=int(np.count_nonzero(~exact & ~finite)),
        max_abs_difference=float(delta[~exact & finite].max()) if np.any(~exact & finite) else 0., beyond_tolerance=int(np.count_nonzero(beyond)),
        tolerance=tolerance, first_beyond=[row(i) for i in np.flatnonzero(beyond)[:FIRST]])
    if list_inexact:summary['first_inexact'] = [row(i) for i in np.flatnonzero(~exact)[:FIRST]]
    return summary, beyond


def merge_numeric(reports, tolerance):
    """Merge {quantity: _numeric summary} dicts across calls; adds exact_equal_fraction and within_tolerance."""
    merged = {}
    for report in reports:
        for name, d in report.items():
            m = merged.setdefault(name, dict(compared=0, exact_equal=0, nonfinite=0, beyond_tolerance=0, max_abs_difference=0., first_beyond=[]))
            for key in ('compared', 'exact_equal', 'nonfinite', 'beyond_tolerance'):m[key] += d[key]
            m['max_abs_difference'] = max(m['max_abs_difference'], d['max_abs_difference'])
            m['first_beyond'] = (m['first_beyond'] + d['first_beyond'])[:FIRST]
            if 'first_inexact' in d:m['first_inexact'] = (m.get('first_inexact', []) + d['first_inexact'])[:FIRST]
    for m in merged.values():
        m['exact_equal_fraction'] = None if not m['compared'] else m['exact_equal'] / m['compared']
        m['tolerance'] = tolerance;m['within_tolerance'] = m['beyond_tolerance'] == 0
    return merged


def _objects(values):
    result = np.empty(len(values), object)
    for i, value in enumerate(values):result[i] = value
    return result


def _masked(keys, got, want, mask):
    """(cells, got, want) where mask is set; a mask with one more axis than keys adds the lane to each cell."""
    keys, got, want, mask = np.asarray(keys), np.asarray(got), np.asarray(want), np.asarray(mask, bool)
    where = np.argwhere(mask);cells = keys[where[:, 0]]
    if mask.ndim == 2:cells = np.concatenate([cells, where[:, 1:]], axis=1)
    return cells, got[mask], want[mask]


GATED = ('t2_value', 't1_own_logits')   # (c) quantities gated at HEAD_TOLERANCE; the rest are reported


def check_heads(frame, data, d_results=None):
    """(c) head outputs against the oracle's heads, plus depth 1's own noise floor against the same oracle."""
    children, leaves, roots = data['children'], data['leaves'], data['roots'];parts = {}
    specs = leaves['specs']
    if len(specs):
        at = tuple(specs.T);scored = (leaves['status'] == 4) & (field(frame, 't2_status')[at] == 4)
        parts['t2_value'] = _masked(specs, field(frame, 't2_value')[at], leaves['head_value'], scored)
    cells = children['cells']
    if len(cells):
        at = tuple(cells.T);status, error = children['status'], children['error']
        same = (field(frame, 't1_status')[at] == status) & (field(frame, 't1_error')[at] == error)
        expanded = same & (status == 2);valued = same & ((status == 2) | np.isin(error, DEPTH2_CODES))
        parts['t1_value_head'] = _masked(cells, field(frame, 't1_value')[at], children['head_value'], valued)
        for seat, scope in (('own', valued), ('peer', expanded)):
            lanes = scope[:, None] & np.asarray(children[seat + '_requested'], bool)[:, None] & np.asarray(children[seat + '_legal'], bool)
            parts[f't1_{seat}_logits'] = _masked(cells, field(frame, f't1_{seat}_logits')[at], children[seat + '_logits'], lanes)
        if d_results is not None:
            # D's own learner value at the child vs the oracle head: the cross-geometry noise of depth 1.
            grid = {(q, wi): r.get('joint_values') for q, result in enumerate(d_results) for wi, r in enumerate(result.world_reports)}
            learner = (status == 2) | np.isin(error, DEPTH2_CODES)
            values = [grid.get((q, w)) for q, w in cells[:, :2].tolist()]
            d = np.asarray([np.nan if v is None or v[a][r] is None else v[a][r] for v, (a, r) in zip(values, cells[:, 2:].tolist())], np.float32)
            parts['depth1_noise_floor'] = _masked(cells, d, children['head_value'], learner & ~np.isnan(d))
    keys = [k for k in sorted(roots) if roots[k]['pre'] == 0 and roots[k]['peer_requested']]
    if keys:
        k = np.asarray(keys);lanes = np.asarray([roots[x]['peer_legal'] for x in keys], bool)
        parts['root_peer_logits'] = _masked(k, field(frame, 'root_peer_logits')[k[:, 0], k[:, 1]],
            np.asarray([roots[x]['peer_logits_raw'] for x in keys]), lanes)
    return {name: _numeric(*part, HEAD_TOLERANCE)[0] for name, part in parts.items()}


def check_oracle(data, findings):
    """The oracle's own leaves re-ran each child identically to its child pass (determinism of the reference)."""
    children, leaves = data['children'], data['leaves']
    if not len(leaves['specs']):return findings
    index = {tuple(c): i for i, c in enumerate(children['cells'].tolist())}
    rows = np.asarray([index.get(tuple(s[:4]), -1) for s in leaves['specs'].tolist()])
    known = rows >= 0;findings.record('leaf_child_known', leaves['specs'], known, known, np.ones_like(known))
    rows = rows[known];specs = leaves['specs'][known]
    findings.record('leaf_child_expanded', specs, children['status'][rows] == 2, children['status'][rows], np.full(len(rows), 2))
    for name in ('invalid', 'done', 'own_digest', 'peer_digest'):
        got = np.asarray(leaves['child_' + name])[known];want = np.asarray(children[name])[rows]
        findings.record('leaf_child_' + name, specs, got == want, got, want)
    return findings


@contextmanager
def stage(progress, name, **context):
    """STAGE_START / STAGE_END progress lines with elapsed seconds; yields a dict merged into the end line."""
    info, clock = {}, time.perf_counter()
    if progress is not None:progress('STAGE_START', stage=name, **context)
    try:yield info
    finally:
        if progress is not None:progress('STAGE_END', stage=name, seconds=time.perf_counter() - clock, **context, **info)


def evaluate(frame, d_results, requests, data, *, context=None, full_turn1=True, progress=None):
    """All four checks for ONE pricing call; data: the oracle's roots/children/enumerated/leaves for the same requests."""
    count = len(requests);geometry = frame_geometry(frame, count);context = context or {}
    if geometry:
        bad = dict(passed=False, compared={}, mismatched=dict(frame_geometry=len(geometry)),
            first_mismatches=[dict(context, field='frame_geometry', problems=geometry[:FIRST])])
        return dict(a=bad, b=dict(bad, values={}), c={}, d=bad, oracle=check_oracle(data, Findings(context)).report(), geometry=geometry)
    with stage(progress, 'check_a', **context) as info:a = check_exact(frame, data, Findings(context)).report();info['passed'] = a['passed']
    with stage(progress, 'check_b', **context) as info:
        found = check_d(frame, d_results, requests, Findings(context), full_turn1=full_turn1);b = dict(found.report(), values=found.values)
        info['passed'] = b['passed']
    with stage(progress, 'check_c', **context) as info:
        c = check_heads(frame, data, d_results);info['max_abs_difference'] = {k: v['max_abs_difference'] for k, v in c.items()}
    with stage(progress, 'check_d', **context) as info:d = check_enumeration(frame, data, Findings(context)).report();info['passed'] = d['passed']
    with stage(progress, 'check_oracle_consistency', **context) as info:
        oracle = check_oracle(data, Findings(context)).report();info['passed'] = oracle['passed']
    return dict(a=a, b=b, c=c, d=d, oracle=oracle, geometry=[])


def summarize(calls, *, tolerance=HEAD_TOLERANCE):
    """Merge per-call evaluations into the gate's checks: (b) discrete exact + values within D_VALUE_TOLERANCE,
    (c) GATED head quantities within HEAD_TOLERANCE."""
    a, b, d, oracle = (merge_findings([c[k] for c in calls]) for k in ('a', 'b', 'd', 'oracle'))
    values = merge_numeric([x['b'].get('values', {}) for x in calls], D_VALUE_TOLERANCE)
    b.update(reference='D3_FRAMEPRICER_DISCRETE_EXACT_FLOATS_WITHIN_TOLERANCE (not bitwise)', value_tolerance=D_VALUE_TOLERANCE, values=values)
    b['passed'] = b['passed'] and all(v['within_tolerance'] for v in values.values())
    quantities = merge_numeric([x['c'] for x in calls], tolerance)
    for name, m in quantities.items():m['gates'] = 'c' if name in GATED else None
    floor = quantities.get('depth1_noise_floor', {}).get('max_abs_difference');t2 = quantities.get('t2_value', {}).get('max_abs_difference')
    c = dict(passed=all(quantities[n]['within_tolerance'] for n in GATED if n in quantities), tolerance=tolerance,
        tolerance_source=HEAD_TOLERANCE_SOURCE, reference='ORACLE_BF16_HEADS_DIFFERENT_BATCH_GEOMETRY', gated=list(GATED),
        depth1_noise_floor=floor, t2_value_to_floor_ratio=None if not floor or t2 is None else t2 / floor, quantities=quantities)
    if any(x.get('geometry') for x in calls):c['passed'] = False
    checks = dict(a=a, b=b, c=c, d=d, oracle_consistency=oracle)
    if any('invariance' in x for x in calls):checks['block_invariance'] = merge_invariance([r for x in calls for r in x.get('invariance', ())])
    return dict(passed=all(x['passed'] for x in checks.values()), checks=checks)


def error_counts(status, error, codes):
    """{code: cells with status 3 and that code}, every listed code present, any unexpected code added."""
    status, error = np.asarray(status), np.asarray(error);failed = error[status == 3]
    return {str(c): int(np.count_nonzero(failed == c)) for c in sorted(set(codes) | set(map(int, np.unique(failed))))}


def depth2_refused(frame):
    """The gate's own reference-rule count: a root refuses if a world has pre != 0 or any cell has status 3."""
    pre, t1, t2 = (field(frame, n) for n in ('pre', 't1_status', 't2_status'));count = len(pre)
    return np.any(pre != 0, axis=1) | np.any(t1.reshape(count, -1) == 3, axis=1) | np.any(t2.reshape(count, -1) == 3, axis=1)


def depth2_summary(frame):
    """depth2_backups.refusal_summary(frame) when importable (it lands with the pricer), else (None, reason)."""
    try:from gpu_engine.depth2_backups import refusal_summary
    except ImportError as error:return None, 'not importable: ' + str(error)
    try:return refusal_summary(SimpleNamespace(**frame) if isinstance(frame, dict) else frame), None
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:return None, f'{type(error).__name__}: {error}'


def refusal_report(frame, d_results, data):
    """One call's root refusals at depth 2 (reference rule) and depth 1 (D3 FramePricer), and error counts by code."""
    refused = depth2_refused(frame)
    own = dict(source='gate count of the frame', root_count=len(refused), refused_count=int(refused.sum()), refused_roots=np.flatnonzero(refused).tolist())
    report = dict(depth2=own, depth1=dict(source='FramePricer(vectorized=True, host_prep=True)', root_count=len(d_results),
            refused_count=sum(r.status == 'REFUSED' for r in d_results), refused_roots=[i for i, r in enumerate(d_results) if r.status == 'REFUSED']),
        turn1_errors_by_code=error_counts(field(frame, 't1_status'), field(frame, 't1_error'), range(1, 7)),
        turn2_errors_by_code=error_counts(field(frame, 't2_status'), field(frame, 't2_error'), range(1, 5)))
    children, leaves = data['children'], data['leaves']
    report['oracle_turn1_errors_by_code'] = error_counts(children.get('status', []), children.get('error', []), range(1, 7))
    report['oracle_turn2_errors_by_code_checked_leaves'] = error_counts(leaves.get('status', []), leaves.get('error', []), range(1, 5))
    summary, reason = depth2_summary(frame)
    if summary is None:report['refusal_summary'] = reason
    else:
        keys = ('root_count', 'refused_count', 'depth1_refused_count', 'depth2_only_refused_count', 'refused_by_cause',
            'turn1_errors_by_code', 'turn2_errors_by_code')
        picked = json.loads(json.dumps({k: summary[k] for k in keys if k in summary}, default=str))
        report['depth2'] = dict(picked, source='depth2_backups.refusal_summary', refused_roots=[int(x['root']) for x in summary.get('roots', ()) if x.get('refused')])
        report['depth2_gate_count'] = own;report['depth2_sources_agree'] = picked.get('refused_count') == own['refused_count']
    return report


def merge_refusals(reports):
    """Sum per-call refusal reports; shares are refused / root requests (each world block counts its own requests)."""
    def add(total, part):
        for key, value in part.items():
            if isinstance(value, dict):add(total.setdefault(key, {}), value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):total[key] = total.get(key, 0) + value
            elif isinstance(value, str):total.setdefault(key, value)
            elif isinstance(value, bool):total[key] = total.get(key, True) and value
    merged = {}
    for report in reports:
        add(merged, {k: ({kk: vv for kk, vv in v.items() if kk != 'refused_roots'} if isinstance(v, dict) else v) for k, v in report.items()})
    for depth in ('depth2', 'depth1', 'depth2_gate_count'):
        if depth in merged:
            m = merged[depth];m['refused_share'] = None if not m.get('root_count') else m['refused_count'] / m['root_count']
    return merged


def sample_leaves(leaves, count, addresses, *, block):
    """All leaves (count None), or count uniform leaves per world from a fixed per-world seed."""
    if count is None:return list(leaves)
    if type(count) is not int or count < 0:raise ValueError('non-negative leaf sample size required')
    worlds = collections.defaultdict(list)
    for leaf in leaves:worlds[leaf[:2]].append(leaf)
    chosen = []
    for (q, wi), items in worlds.items():
        a = addresses[q];rng = np.random.default_rng([SAMPLE_SEED, block, a.stream, a.frame, a.owner, wi])
        chosen.extend(items[i] for i in np.sort(rng.choice(len(items), size=min(count, len(items)), replace=False)))
    return chosen


def pricer_costs(timing):
    """compile/device seconds from a pricer timing dict: FramePricer.last_timing's names, which Depth2Pricer's
    `timing` is read by too (None where a pricer does not report one)."""
    timing = timing if isinstance(timing, dict) else {}
    number = lambda key: timing.get(key) if isinstance(timing.get(key), (int, float)) and not isinstance(timing.get(key), bool) else None
    lower, compiled, device = number('lower_seconds'), number('compile_seconds'), number('device_seconds')
    result = dict(lower_seconds=lower, compile_seconds=compiled, device_seconds=device,
        lower_and_compile_seconds=None if lower is None and compiled is None else (lower or 0.) + (compiled or 0.))
    if None in (compiled, device):result['reported_seconds_keys'] = sorted(k for k in timing if 'second' in str(k))
    return result


def _frame_timing(frame):
    return frame['timing'] if isinstance(frame, dict) and 'timing' in frame else getattr(frame, 'timing', None)


def price_call(*, requests, recipe, materializer, params, pricer, frame_pricer, oracle, widths, sample, block, invariance=None,
               progress=None, context=None):
    """Price ONE request list with the pricer, FramePricer and the oracle; returns (frame, D results, oracle data, timing).

    invariance: pricers to run first on the same requests (other child blocks); their frames go in
    data['invariance_frames'], and `pricer` gives the checked frame. progress(message, **values) gets
    STAGE_START/STAGE_END lines per stage."""
    from gpu_engine import depth2_oracle as reference, device_public_teacher_batch as batch
    requests = tuple(requests);timing = {};others = [];context = dict(context or {}, roots=len(requests))
    costs = dict(invariance_pricers=[])
    for i, other in enumerate(invariance or ()):
        with stage(progress, f'depth2_pricer_invariance_{i}', **context) as info:
            clock = time.perf_counter();others.append(other.price(requests, materializer, recipe, params))
            timing[f'invariance_pricer_{i}_seconds'] = time.perf_counter() - clock
            costs['invariance_pricers'].append(dict(wall_seconds=timing[f'invariance_pricer_{i}_seconds'], **pricer_costs(_frame_timing(others[-1]))))
            info.update(costs['invariance_pricers'][-1])
    with stage(progress, 'depth2_pricer', **context) as info:
        clock = time.perf_counter();frame = pricer.price(requests, materializer, recipe, params);timing['pricer_seconds'] = time.perf_counter() - clock
        costs['depth2_pricer'] = dict(wall_seconds=timing['pricer_seconds'], **pricer_costs(_frame_timing(frame)));info.update(costs['depth2_pricer'])
    with stage(progress, 'frame_pricer', **context) as info:
        clock = time.perf_counter();d_results = frame_pricer.price(requests, materializer, recipe, params)
        timing['frame_pricer_seconds'] = time.perf_counter() - clock
        costs['frame_pricer'] = dict(wall_seconds=timing['frame_pricer_seconds'], **pricer_costs(getattr(frame_pricer, 'last_timing', None)))
        info.update(costs['frame_pricer'])
    clock = time.perf_counter();before = dict(oracle.seconds);chunks = len(oracle.chunks)
    with stage(progress, 'oracle_worlds', **context) as info:
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe));info['worlds'] = len(stack.rows)
    with stage(progress, 'oracle_roots', **context) as info:
        roots = oracle.roots(stack, requests, params);info['usable_worlds'] = sum(r['pre'] == 0 for r in roots.values())
    with stage(progress, 'oracle_children', **context) as info:
        cells = reference.turn1_cells(requests, recipe, roots, k1_own=widths['k1_own'], k1_opp=widths['k1_opp'])
        children = oracle.children(stack, cells, params);info['children'] = len(cells)
    enumerated = reference.turn2_leaves(children, k_own=widths['k_own'], k_opp=widths['k_opp']) if len(cells) else []
    checked = sample_leaves(enumerated, sample, [r.plans.root.address for r in requests], block=block)
    with stage(progress, 'oracle_leaves', **context) as info:
        leaves = oracle.leaves(stack, checked, params);info.update(enumerated=len(enumerated), checked=len(checked))
    timing['oracle_seconds'] = time.perf_counter() - clock
    costs['oracle'] = dict(wall_seconds=timing['oracle_seconds'], chunks=len(oracle.chunks) - chunks,
        seconds={k: oracle.seconds[k] - before[k] for k in oracle.seconds},
        chunk_device_seconds=sum(c['step_seconds'] + c['head_seconds'] for c in oracle.chunks[chunks:]))
    return frame, d_results, dict(roots=roots, children=children, enumerated=enumerated, leaves=leaves, worlds=len(stack.rows),
        invariance_frames=others, costs=costs), timing


def run_call(*, requests, recipe, materializer, params, pricer, frame_pricer, oracle, widths, sample, block, context, invariance=None,
             progress=None):
    """Price ONE request list three ways and evaluate it; returns (evaluation, call summary)."""
    requests = tuple(requests)
    frame, d_results, data, timing = price_call(requests=requests, recipe=recipe, materializer=materializer, params=params,
        pricer=pricer, frame_pricer=frame_pricer, oracle=oracle, widths=widths, sample=sample, block=block, invariance=invariance,
        progress=progress, context=context)
    evaluation = evaluate(frame, d_results, requests, data, context=context, full_turn1=widths['k1_own'] == widths['k1_opp'] == 9,
        progress=progress)
    if invariance:
        with stage(progress, 'check_block_invariance', **context) as info:
            evaluation['invariance'] = [block_invariance(other, frame, context=context) for other in data['invariance_frames']]
            info['passed'] = all(r['passed'] for r in evaluation['invariance'])
    pricer_timing = _frame_timing(frame)
    roots = data['roots']
    with stage(progress, 'refusals', **context):refusals = None if evaluation['geometry'] else refusal_report(frame, d_results, data)
    summary = dict(context, roots=len(requests), worlds=data['worlds'], usable_worlds=sum(r['pre'] == 0 for r in roots.values()), refusals=refusals,
        turn1_cells=len(data['children']['cells']), turn2_leaves=len(data['enumerated']), checked_leaves=len(data['leaves']['specs']), timing=timing,
        costs=data['costs'], pricer_timing=json.loads(json.dumps(pricer_timing, default=str)) if pricer_timing is not None else None,
        passed=dict(a=evaluation['a']['passed'], b=evaluation['b']['passed'], d=evaluation['d']['passed'], oracle=evaluation['oracle']['passed'],
            **({'block_invariance': all(r['passed'] for r in evaluation['invariance'])} if invariance else {})))
    return evaluation, summary


def merge_costs(calls):
    """Per-pricer totals over calls: wall, lower+compile and device seconds (None where never reported)."""
    total = lambda values: None if all(v is None for v in values) else sum(v for v in values if v is not None)
    result = {}
    for name in ('depth2_pricer', 'frame_pricer'):
        rows = [c['costs'][name] for c in calls]
        result[name] = {key: total([r.get(key) for r in rows]) for key in ('wall_seconds', 'lower_seconds', 'compile_seconds',
            'lower_and_compile_seconds', 'device_seconds')}
    rows = [x for c in calls for x in c['costs']['invariance_pricers']]
    if rows:result['invariance_pricers'] = {key: total([r.get(key) for r in rows]) for key in ('wall_seconds', 'lower_and_compile_seconds', 'device_seconds')}
    oracle = [c['costs']['oracle'] for c in calls]
    result['oracle'] = dict(wall_seconds=sum(o['wall_seconds'] for o in oracle), chunks=sum(o['chunks'] for o in oracle),
        chunk_device_seconds=sum(o['chunk_device_seconds'] for o in oracle))
    return result


def oracle_chunk_progress(log_line, *, interval=PROGRESS_INTERVAL, clock=time.perf_counter):
    """Oracle progress callback: the first and last chunk of each pass, and otherwise at most one line per interval."""
    state = dict(last=-math.inf)
    def progress(**record):
        now = clock()
        if record['chunk'] == 0 or record['chunk'] == record['chunks'] - 1 or now - state['last'] >= interval:
            state['last'] = now;log_line('ORACLE_CHUNK', **record)
    return progress


def make_pricer(executor, model, *, widths, actor_capacity, child_block=None):
    """Depth2Pricer(executor, model, digest=True, widths..., child_block) - imported lazily: it lands on another branch.

    Options the constructor does not take are dropped, unless they would change what is priced: a width
    other than 9, or an explicit child_block, then refuse."""
    from gpu_engine.device_depth2_pricer import Depth2Pricer
    options = dict(digest=True, actor_capacity=actor_capacity, **widths)
    required = {k for k, v in widths.items() if v != 9}
    if child_block is not None:options['child_block'] = child_block;required.add('child_block')
    parameters = inspect.signature(Depth2Pricer).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        if 'digest' not in parameters:raise SystemExit('Depth2Pricer has no digest option')
        unsupported = sorted(k for k in required if k not in parameters)
        if unsupported:raise SystemExit('Depth2Pricer does not accept option(s): ' + ','.join(unsupported))
        options = {k: v for k, v in options.items() if k in parameters}
    return Depth2Pricer(executor, model, **options), options


INVARIANCE_BLOCKS = (27, 81)   # --block-invariance: child_block values priced; the last frame is the checked one


def frame_fields(frame):
    if isinstance(frame, dict):return dict(frame)
    if dataclasses.is_dataclass(frame):return {f.name: getattr(frame, f.name) for f in dataclasses.fields(frame)}
    if hasattr(frame, '_asdict'):return dict(frame._asdict())
    return dict(vars(frame))


def block_invariance(a, b, *, context=None):
    """Bit-equality of every output array of two frames of the same requests (timing excluded).

    Arrays must match in dtype, shape and bytes (NaN payloads included). Non-array fields (world_reports)
    are compared as canonical JSON and reported, not gated."""
    fa, fb = frame_fields(a), frame_fields(b);arrays, other, differing, first = 0, {}, {}, []
    for name in sorted((set(fa) | set(fb)) - {'timing'}):
        if name not in fa or name not in fb:differing[name] = dict(problem='missing in ' + ('first' if name not in fa else 'second'));continue
        x, y = np.asarray(fa[name]), np.asarray(fb[name])
        if x.dtype.kind not in 'biufc' and y.dtype.kind not in 'biufc':
            canonical = lambda v: json.dumps(v, sort_keys=True, default=str)
            other[name] = canonical(fa[name]) == canonical(fb[name]);continue
        arrays += 1
        if x.dtype != y.dtype or x.shape != y.shape:
            differing[name] = dict(problem=f'dtype/shape {x.dtype}{x.shape} != {y.dtype}{y.shape}');continue
        size = x.dtype.itemsize;flat = lambda v: np.ascontiguousarray(v).reshape(-1).view(np.uint8).reshape(-1, size)
        bad = np.flatnonzero(np.any(flat(x) != flat(y), axis=1)) if x.size else np.zeros(0, int)
        if len(bad):
            differing[name] = dict(count=int(len(bad)), size=int(x.size))
            for i in bad[:max(0, FIRST - len(first))]:
                cell = np.unravel_index(i, x.shape)
                first.append(dict(context or {}, field=name, cell=[int(c) for c in cell], first=_plain(x[cell]), second=_plain(y[cell])))
    return dict(passed=not differing, arrays_compared=arrays, differing=differing, first_differences=first, nonarray_equal=other)


def merge_invariance(reports, blocks=INVARIANCE_BLOCKS):
    differing, first, other = {}, [], {}
    for r in reports:
        for name, d in r['differing'].items():
            m = differing.setdefault(name, dict(count=0, size=0))
            if 'problem' in d:m.setdefault('problems', []).append(d['problem'])
            else:m['count'] += d['count'];m['size'] += d['size']
        first.extend(r['first_differences'])
        for name, equal in r['nonarray_equal'].items():other[name] = other.get(name, True) and equal
    return dict(passed=all(r['passed'] for r in reports), child_blocks=list(blocks), calls=len(reports),
        arrays_compared=max((r['arrays_compared'] for r in reports), default=0), differing=differing,
        first_differences=first[:FIRST], nonarray_equal=other)


def _probe(worker, runtime, jax):
    try:worker._probe(runtime)
    except ValueError as error:
        # The probe pins the device count/kind of the original worker; keep its numeric settings (as the D3 gates).
        log('PROBE_DIFFERS', error=str(error));jax.config.update('jax_enable_x64', False);jax.config.update('jax_default_matmul_precision', 'highest')


def saved_root_requests(args):
    """The 40 saved D3 cost roots' Requests, as d3_saved_root_gate --batched-prep builds them (no serial pricing)."""
    root = Path(args.saved_roots)
    from pokezero import v5_worker as worker, v5_run as runs, v5_device_evaluation_bank as schema, v5_contract as contract
    runtime = json.loads((root / 'runtime.json').read_text());pool_sha = runtime['constructor_pool']['sha256']
    profile, vocabulary, _, category = worker._runtime_metadata(runtime, expected_pool_sha256=pool_sha)
    worker._memory(runtime['memory_policy'], apply=True)   # must precede the first JAX import
    import jax
    _probe(worker, runtime, jax)
    prepared, _, _, _ = worker.prepare_device_runtime(SimpleNamespace(runtime=runtime, vocabulary=vocabulary, plan=SimpleNamespace(pool_sha256=pool_sha)))
    from gpu_engine import device_d_cost as cost, v5_device_teacher as teacher, device_public_worlds as worlds
    from gpu_engine.device_public_teacher_batch import FramePricer, Request
    index = worker._record(dict(path=str(root / 'bank/cost-root-index.json'), sha256=args.index_sha256))
    declaration, batches, _, labels = schema.declaration(index['declaration'])
    prespec = json.loads((root / 'prespec.json').read_text())
    plan = cost.plan_from_bank(profile=profile, labels=labels, category=category, batches=batches, pool_sha256=pool_sha,
        selection_seed=declaration['cost_recipe_seed'], run_id=prespec['cost_run_id'], max_requests=declaration['max_requests'])
    snapshot = runs.Snapshot(**json.loads((root / 'cost-snapshot.json').read_text()))
    provider = teacher.Provider(prepared, plan, batch_joint=True);captured = {};actors = []
    def capture(roots, frames, bank, params):
        # device_d_frame_parity.run's capture, minus the serial label: one REFUSED record per selected slot.
        records = []
        for step, row, owner in teacher._requests(frames, bank):
            step, row, owner = map(int, (step, row, owner));base = teacher.binding(frames, bank, step, row, owner)
            record = dict(base, recipe_seed=provider.recipe.seed, status='UNSELECTED', refusal=None, improvement=None, diagnostics=None)
            records.append(record);address = contract.DecisionAddress(**base['address'])
            if not provider.recipe.selected(address):continue
            record.update(status='REFUSED', refusal='depth2-gate: request captured, not priced by the serial provider')
            if not frames.trainable[step, row, owner]:continue
            context = jax.tree.map(lambda x: x[step], roots)
            detached = worlds.detach_roots(context, provider.source, rows=[row], owners=[owner], enable_device_worlds=True)[0]
            inputs = {name: getattr(frames, name)[step, row, owner][None, None] for name in teacher.INPUTS}
            inputs['history_mask'] = np.ones((1, 1), bool)
            captured[detached.address] = Request(worlds.plan_starts(detached, provider.source, provider.recipe),
                tuple(map(float, frames.raw_policy_logits[step, row, owner])), tuple(map(bool, frames.legal_action_mask[step, row, owner])),
                inputs, int(frames.actor_indices[step, row, owner]), int(frames.actor_indices[step, row, 1 - owner]))
        if actors and any(p is not q for p, q in zip(actors[0], params)):raise ValueError('saved actor parameter identity differs')
        if not actors:actors.append(tuple(params))
        return records
    provider.label_chunk = capture
    reference = cost.run(provider, snapshot, snapshot.archive, index, enable_cost=True)
    if not reference['producer_integrity']:raise SystemExit('original fixed saved-root producer failed')
    requests = [captured[a] for a in cost.select(provider.plan) if a in captured]
    log('SAVED_ROOTS', captured=len(captured), dispositions=reference['summary']['dispositions'])
    frame_pricer = FramePricer(provider.executor, plan.model, actor_capacity=plan.historical_window + 1, capacity=4096,
        vectorized=True, host_prep=True)
    return provider, plan, actors[0], [requests], frame_pricer, dict(saved_roots=str(root), index_sha256=args.index_sha256)


def bank_chunk_requests(args):
    """One D3 bank chunk's Requests, as d3_chunk_equal builds the chunk and Provider._prepare_batched the requests."""
    root = Path(args.d3_root)
    from pokezero import v5_worker as worker, v5_run as runs
    runtime_path = root / 'runtime.json'
    runtime = worker._record(dict(path=str(runtime_path), sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest()))
    if args.memory_policy:worker._memory(runtime['memory_policy'], apply=True)   # must precede the first JAX import
    import jax
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_loss import V5LossConfig
    from pokezero import v5_training as training
    from gpu_engine import v5_device_rollout as rollout, v5_model
    from gpu_engine.v5_device_backend import RuntimeBackend
    from gpu_engine.v5_device_teacher import Provider
    profile, vocabulary, gender, _ = worker._runtime_metadata(runtime, expected_pool_sha256=runtime['constructor_pool']['sha256'])
    d = json.loads((root / 'run' / 'declaration.json').read_text())
    loss_fields = {k: v for k, v in d['loss'].items() if k in V5LossConfig.__dataclass_fields__}
    batches = tuple(tuple(runs.Batch(tuple(runs.Game(**g) for g in x['games']), x['constructor_seed'], x['sampling_seed'],
        x.get('gender_seed'), x.get('gender_domain'), tuple(x.get('gender_root_ids', ()))) for x in it) for it in d['iterations'])
    plan = runs.Plan(d['run_id'], V5ModelConfig.from_metadata(d['model']), V5LossConfig(**loss_fields), training.TrainConfig(**d['train']),
        batches, d['pool_sha256'], d['selection_seed'], d['historical_fraction'], d['historical_window'], d['max_requests'],
        runs.Watchdog(**d['watchdog']), checkpoint_updates=d['checkpoint_updates'], discount=d['discount'])
    declared = json.loads((root / 'run' / 'journal' / '000000000.json').read_text())['payload']['declaration_sha256']
    if runs._sha(plan.manifest()) != declared:raise SystemExit('rebuilt plan differs from the D3 declaration')
    _probe(worker, runtime, jax)
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    provider = Provider(prepared, plan, frame_batch=True, chunk_batch=True, vectorized=True, batched_prep=True)
    backend = RuntimeBackend(prepared, pool, gender_policy=gender, teacher=provider);backend.check_plan(plan)
    events = (json.loads(p.read_text()) for p in sorted((root / 'run' / 'journal').glob('*.json')))
    snapshot = next(e['payload'] for e in events if e['kind'] == 'SNAPSHOT' and e['payload']['iteration'] == -1)
    params, config, _ = v5_model.load(root / 'run' / snapshot['archive'], expected_archive_sha256=snapshot['archive_sha256'],
        expected_checkpoint_sha256=snapshot['checkpoint_sha256'], expected_category_vocabulary_sha256=plan.model.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=plan.model.label_vocabulary.sha256, enable_v5=True)
    if config != plan.model:raise SystemExit('actor configuration differs')
    batch = plan.iterations[0][0];ordinal = {g.stream_id: i for i, g in enumerate(plan.games)};snap = runs.Snapshot(**snapshot)
    assignments = json.loads(runs._json(tuple(runs.assignment(plan, g, ordinal[g.stream_id], snap, ()) for g in batch.games)))
    archive = snapshot['archive_sha256']
    bindings = rollout.Bindings(plan.model.category_vocabulary_sha256, plan.model.label_vocabulary, prepared.source_sha256,
        (archive,) * 2, (True, True), backend.declaration_sha256)
    actor_bank = rollout.actor_bank_from_assignments(bindings, (archive,), current_archive=archive, assignments=assignments)
    collector = rollout.make_collector(prepared.engine_tables, prepared.observation_tables, plan.model, backend.labels, bindings,
        sampling_seed=batch.sampling_seed, chunk_steps=8, max_requests=plan.max_requests, actor_bank=actor_bank, retain_teacher_roots=True)
    carry = jax.block_until_ready(backend._construct(batch));log('CONSTRUCTED', games=len(batch.games))
    for _ in range(args.chunk + 1):
        device = jax.block_until_ready(collector(carry, (params,)));carry = device.carry
        host = rollout.boundary(device._replace(carry=carry._replace(state=None, public=None, truth=None), roots=None))
    provider.bind(actor_bank, (params,));provider.reset_metrics()
    records, grouped = provider._prepare_batched(device.roots, host.frames, actor_bank)
    groups = [[request for _, request in items] for _, items in grouped.items()]
    log('BANK_CHUNK', chunk=args.chunk, records=len(records), selected=sum(r['status'] != 'UNSELECTED' for r in records),
        frames=sorted(grouped), requests=sum(map(len, groups)))
    return provider, plan, (params,), groups, provider.frame_pricer, dict(d3_root=str(root), chunk=args.chunk, declaration_sha256=declared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--saved-roots', help='saved D3 cost bank attempt directory (runtime.json, bank/, cost-snapshot.json, prespec.json)')
    parser.add_argument('--index-sha256', help='sha256 of bank/cost-root-index.json (with --saved-roots)')
    parser.add_argument('--d3-root', help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, help='bank chunk index (with --d3-root); chunks 0..N are collected')
    parser.add_argument('--memory-policy', action='store_true', help="with --d3-root: apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory')
    for name in ('k-own', 'k-opp', 'k1-own', 'k1-opp'):
        parser.add_argument('--' + name, type=int, default=9, help='width 1..9 (9 = full width)')
    parser.add_argument('--worlds-blocks', type=int, default=1, help='K recipe blocks, W = 2K worlds per root')
    parser.add_argument('--max-roots', type=int, help='price only the first N requests')
    parser.add_argument('--sample-leaves', type=int, help='check N sampled turn-2 leaves per world (default: all)')
    parser.add_argument('--oracle-chunk', type=int, default=512, help='oracle specs per vmap invocation')
    parser.add_argument('--child-block', type=int, help='Depth2Pricer child_block (passed through; refused if not accepted)')
    parser.add_argument('--block-invariance', action='store_true',
        help='price every call with child_block 27 and 81 and require bit-equal output arrays (the 81 frame is checked)')
    args = parser.parse_args(argv)
    if bool(args.saved_roots) == bool(args.d3_root):parser.error('exactly one of --saved-roots or --d3-root required')
    if args.saved_roots and not args.index_sha256:parser.error('--saved-roots requires --index-sha256')
    if args.d3_root and (args.chunk is None or args.chunk < 0):parser.error('--d3-root requires --chunk N >= 0')
    widths = dict(k_own=args.k_own, k_opp=args.k_opp, k1_own=args.k1_own, k1_opp=args.k1_opp)
    if any(not 1 <= v <= 9 for v in widths.values()):parser.error('widths must be 1..9')
    if args.worlds_blocks < 1 or (args.max_roots is not None and args.max_roots < 1):parser.error('positive --worlds-blocks/--max-roots required')
    if args.oracle_chunk < 1 or (args.sample_leaves is not None and args.sample_leaves < 0):parser.error('positive --oracle-chunk, non-negative --sample-leaves')
    if args.child_block is not None and args.child_block < 1:parser.error('positive --child-block required')
    if args.block_invariance and args.child_block is not None:parser.error('--block-invariance fixes child_block to 27 and 81; drop --child-block')
    import importlib.util   # fail before loading roots; find_spec does not import JAX
    if importlib.util.find_spec('gpu_engine.device_depth2_pricer') is None:raise SystemExit('gpu_engine.device_depth2_pricer is not on this tree')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    provider, plan, params, groups, frame_pricer, source = (saved_root_requests if args.saved_roots else bank_chunk_requests)(args)
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds, depth2_oracle as reference
    if args.max_roots is not None:
        kept, left = [], args.max_roots
        for group in groups:
            if left > 0:kept.append(group[:left]);left -= len(kept[-1])
        groups = kept
    capacity = frame_pricer.capacity
    groups = [g[i:i + capacity] for g in groups for i in range(0, len(g), capacity) if g[i:i + capacity]]
    blocks = INVARIANCE_BLOCKS if args.block_invariance else (args.child_block,)
    pricers = [make_pricer(provider.executor, plan.model, widths=widths, actor_capacity=plan.historical_window + 1, child_block=b) for b in blocks]
    (pricer, options), invariance = pricers[-1], [p for p, _ in pricers[:-1]]
    oracle = reference.Oracle(provider.executor, plan.model, chunk=args.oracle_chunk, progress=oracle_chunk_progress(log))
    seeds = world_block_seeds(provider.recipe.seed, args.worlds_blocks)
    log('GATE_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), widths=widths, sample_leaves=args.sample_leaves, seeds=seeds,
        pricer_options=options)
    evaluations, calls = [], []
    for block, seed in enumerate(seeds):
        recipe = provider.recipe if block == 0 else original.Recipe(seed)
        # As U1: later blocks price every root, so only the quarter selection is bypassed.
        with patch.object(original.Recipe, 'selected', lambda self, address: True) if block else nullcontext():
            for call, group in enumerate(groups):
                requests = group if block == 0 else [replace(r, plans=worlds.plan_starts(r.plans.root, provider.source, recipe)) for r in group]
                evaluation, summary = run_call(requests=requests, recipe=recipe, materializer=provider.materializer, params=params,
                    pricer=pricer, frame_pricer=frame_pricer, oracle=oracle, widths=widths, sample=args.sample_leaves, block=block,
                    context=dict(block=block, call=call), invariance=invariance, progress=log)
                evaluations.append(evaluation);calls.append(summary);log('CALL_CHECKED', **summary)
    result = summarize(evaluations)
    reported = [c for c in calls if c['refusals'] is not None]
    refusals = dict(merge_refusals([c['refusals'] for c in reported]), calls_reported=len(reported), calls=len(calls),
        by_block={str(b): merge_refusals([c['refusals'] for c in reported if c['block'] == b]) for b in range(len(seeds))},
        rule='reference (D): any world pre != 0 or any status-3 cell at either turn refuses the root; codes 5/6 refuse too')
    leaves = oracle.rows['leaves'];children = oracle.rows['children']
    report = dict(schema=SCHEMA, status='PASS' if result['passed'] else 'FAIL', passed=result['passed'], checks=result['checks'],
        mode='SAVED_ROOTS' if args.saved_roots else 'BANK_CHUNK', source=source, widths=widths, worlds_blocks=args.worlds_blocks,
        block_seeds=list(seeds), max_roots=args.max_roots, sample_leaves=args.sample_leaves, sample_seed=SAMPLE_SEED,
        head_tolerance=HEAD_TOLERANCE, head_tolerance_source=HEAD_TOLERANCE_SOURCE, d_value_tolerance=D_VALUE_TOLERANCE, pricer_options=options, block_invariance=args.block_invariance,
        invariance_pricer_options=[o for _, o in pricers[:-1]], calls=calls, conventions=list(CONVENTIONS), refusals=refusals,
        costs=merge_costs(calls),
        oracle_cost=dict(seconds=oracle.seconds, rows=oracle.rows, chunk=args.oracle_chunk, chunks=oracle.chunks,
            first_chunk_seconds={kind: next((c['seconds'] for c in oracle.chunks if c['kind'] == kind), None) for kind in ('roots', 'children', 'leaves')},
            chunk_scope='per chunk: step_seconds = the vmapped executor program to block_until_ready, head_seconds = the bf16 heads; '
                'the first chunk of a program and world bucket includes its compile',
            seconds_per_leaf=None if not leaves else oracle.seconds['leaves'] / leaves,
            seconds_per_child=None if not children else oracle.seconds['children'] / children))
    (out / 'gate.json').write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')
    log('DEPTH2_GATE_COMPLETE', status=report['status'], out=str(out / 'gate.json'),
        checks={k: v['passed'] for k, v in result['checks'].items()},
        refused_share=dict(depth2=refusals.get('depth2', {}).get('refused_share'), depth1=refusals.get('depth1', {}).get('refused_share')))
    return 0 if result['passed'] else 1


if __name__ == '__main__':raise SystemExit(main())
