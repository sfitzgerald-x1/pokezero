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
     request/legal, status/error wherever the pricer's code is D's 0-4, the turn-1 pair set);
     |t1_value - D| <= VALUE_TOLERANCE on every D-priced cell (codes 5/6 included: they keep D's value) and
     |root prior - D| <= LOGIT_TOLERANCE; the bit-equal fraction, max |diff| and first inexact cells are reported.
 (c) heads, against the oracle's own bf16 heads (a different batch geometry): t2_value and t1 head values
     within VALUE_TOLERANCE, both seats' child logits and the root peer logits within LOGIT_TOLERANCE
     (TOLERANCE_SOURCE). Reported per quantity: max |diff| and exact fraction, next to the depth-1 noise
     floor max |D t1_value - oracle t1 head| and the ratio t2_value max / floor.
 (d) enumeration: the pricer's enumerated turn-1 and turn-2 sets equal the oracle's.
 (e) decision, when every leaf is checked (no --sample-leaves): depth2_backups.learner_max_root on the
     pricer frame and on a copy carrying the oracle's turn-2 head values and child-prior logits (statuses,
     terminal values and enumeration unchanged); passes iff no root's completed-Q argmax or improved-policy
     target argmax differs (lowest index on ties). Cell-level learner_max_choice differences are reported.
     depth2_backups must be importable (fail closed). --save-arrays writes each call's frame arrays and the
     oracle's per-leaf head values and child logits to arrays-block{B}-call{C}.npz for offline re-runs.
--sample-leaves N checks all turn-1 children plus N uniformly sampled turn-2 leaves per world (fixed
SAMPLE_SEED); (d) always covers every leaf. --child-block N is passed to Depth2Pricer. --block-invariance
prices every call twice, with child_block 27 and 81 (the 81 frame is the one checked), and requires
every discrete output array to be bit-equal and every float array within its tolerance (on GPU the turn-2
forward batch changes with child_block). Reported, not gated: the depth-2 root refusal share under the
reference rule (depth2_backups.refusal_summary when importable, else the gate's own count of the frame)
next to D3 FramePricer's on the same requests, turn-1/turn-2 error counts by code, the values section (magnitudes,
learner-max cell and root spreads and the near-tie share, per root-frame band, from the pricer's frame) and the sibling
spread of head-valued leaves. Offline: --recheck DIR re-evaluates every check its saved arrays allow and names what is
missing; --pruning-report DIR ranks learner-max's turn-2 action against the top-k child prior on full-width arrays.
--topk-errors FULL_DIR TOPK_DIR gives, under top-k, the share of roots whose only full-width errors sit in cells top-k
prunes away. --pricer-only runs only Depth2Pricer (status PRICER_ONLY): values, refusals and arrays.

Evaluator: by default the bank snapshot's actor prices (the D3 cost banks' snapshot is iteration -1, UNTRAINED). The
head override --head-archive/--head-archive-sha256/--head-checkpoint-sha256 (optional --head-category-sha256 and
--head-model-sha256) loads a v5 head exactly as U1 does: the bank declaration's category vocabulary and the plan's labels
bind it, its geometry must be the bank D model, and the D-arm and head-arm forwards must agree exactly on the first root.
Every request then takes the head's own root logits and actors 0/0 (one actor) in every mode, for the pricer,
FramePricer and the oracle alike. The first JSON line and the top of gate.json carry the evaluator and value_scale (root
and expanded-cell |v| p50/p90, and --value-reference echoed). --raw-head-check gates, in (c), the pricer's leaf and child
values against the head's own arm on the oracle's inputs (U1's predict path). All paths are command-line arguments;
--out must be new.
"""
import argparse
import collections
from contextlib import contextmanager, nullcontext
import dataclasses
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCHEMA = 'pokezero.depth2-gate.v1'
VALUE_TOLERANCE = 1e-4     # every cross-program / cross-geometry value comparison (absolute)
LOGIT_TOLERANCE = 1e-2     # every cross-program / cross-geometry logit comparison (absolute)
TOLERANCE_SOURCE = ('lead decision after the 40-root GPU run at 08d1395b: values within 1e-4 absolute (head values are ~1e-3; '
    'measured cross-geometry value noise <= 3.7e-5), logits within 1e-2; discrete fields, digests and check (e) stay exact')
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
    'Exact: every discrete field and digest, and check (e). Floats, uniformly for every cross-program or cross-geometry comparison (vs D3, vs the oracle, across child blocks): values within VALUE_TOLERANCE, logits within LOGIT_TOLERANCE.',
    'Top-k: the oracle selects with its own top_k code but ranks by the PRICER\'s child (and k1_opp root) priors, which (c) checks against the oracle\'s within LOGIT_TOLERANCE, so a bf16 near-tie cannot fail (d); topk_boundary reports each selection\'s rank-k vs rank-(k+1) gap and how often the oracle\'s own prior would differ.',
    'Status: FAIL if any check that ran failed; INCOMPLETE if one is unavailable or vacuous (no turn-2 leaf compared, or (e) compared no root); PASS_SAMPLED with --sample-leaves ((e) SKIPPED_SAMPLED); else PASS. A sample always includes every terminal and error leaf the pricer reports.',
    '(e) swaps in the oracle\'s heads only on the lanes (c) checks (requested seats, legal lanes; the root opponent prior too); a waiting seat\'s unused prior keeps the pricer\'s value.',
    'Each world block is a separate call (Depth2Pricer.price takes one recipe); blocks after 0 bypass Recipe.selected, as U1.',
    'Evaluator: the bank snapshot actor unless --head-archive; with the override every request takes the head\'s root logits and actors 0/0 (U1).',
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
                if (a, r) in errors:   # D's codes 1-4 take precedence over the depth-2-only 5/6, so a 5/6 here is wrong too
                    rows['t1_error_vs_d'].append((cell, status == 3 and code == errors[(a, r)], [status, code], [3, errors[(a, r)]]))
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
        summary, beyond = _numeric(cells, got, want, tolerance_for(name), findings.context, list_inexact=True)
        findings.values[name] = summary
        if len(beyond):findings.record(name + '_vs_d_beyond_tolerance', list(cells), ~beyond, np.asarray(got, np.float32), np.asarray(want, np.float32))
    return findings


def _numeric(cells, got, want, tolerance, context=None, *, list_inexact=False, shape=None):
    """Float agreement on matching cells: (summary, beyond-tolerance mask). Bit-equal cells (NaN payloads
    included) always agree; otherwise both must be finite and within tolerance. cells=None with `shape`
    names a listed cell by unravelling its flat index (no coordinate array for whole frames)."""
    got, want = np.asarray(got, np.float32).reshape(-1), np.asarray(want, np.float32).reshape(-1)
    if cells is None:cells = _Unravel(shape)
    else:cells = np.asarray(cells, np.int64).reshape(len(got), -1) if len(got) else np.zeros((0, 1), np.int64)
    exact = _bits(got) == _bits(want);finite = np.isfinite(got) & np.isfinite(want)
    delta = np.where(finite, np.abs(got.astype(np.float64) - want.astype(np.float64)), np.inf)
    beyond = ~exact & ~(delta <= tolerance)
    row = lambda i: dict(context or {}, cell=cells[i].tolist(), pricer=_plain(got[i]), reference=_plain(want[i]))
    summary = dict(compared=int(len(got)), exact_equal=int(np.count_nonzero(exact)), nonfinite=int(np.count_nonzero(~exact & ~finite)),
        max_abs_difference=float(delta[~exact & finite].max()) if np.any(~exact & finite) else 0., beyond_tolerance=int(np.count_nonzero(beyond)),
        tolerance=tolerance, first_beyond=[row(i) for i in np.flatnonzero(beyond)[:FIRST]])
    if list_inexact:summary['first_inexact'] = [row(i) for i in np.flatnonzero(~exact)[:FIRST]]
    return summary, beyond


def tolerance_for(name):
    """The uniform float tolerance of a compared quantity or frame field: logits LOGIT_TOLERANCE, values VALUE_TOLERANCE."""
    return LOGIT_TOLERANCE if 'logits' in name else VALUE_TOLERANCE


class _Unravel:
    def __init__(self, shape):self.shape = shape

    def __getitem__(self, i):return np.asarray(np.unravel_index(int(i), self.shape), np.int64)


def merge_numeric(reports):
    """Merge {quantity: _numeric summary} dicts across calls; adds exact_equal_fraction and within_tolerance."""
    merged = {}
    for report in reports:
        for name, d in report.items():
            m = merged.setdefault(name, dict(compared=0, exact_equal=0, nonfinite=0, beyond_tolerance=0, max_abs_difference=0., first_beyond=[]))
            for key in ('compared', 'exact_equal', 'nonfinite', 'beyond_tolerance'):m[key] += d[key]
            m['max_abs_difference'] = max(m['max_abs_difference'], d['max_abs_difference'])
            m['first_beyond'] = (m['first_beyond'] + d['first_beyond'])[:FIRST]
            if 'first_inexact' in d:m['first_inexact'] = (m.get('first_inexact', []) + d['first_inexact'])[:FIRST]
    for name, m in merged.items():
        m['exact_equal_fraction'] = None if not m['compared'] else m['exact_equal'] / m['compared']
        m['tolerance'] = tolerance_for(name);m['within_tolerance'] = m['beyond_tolerance'] == 0
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


GATED = ('t2_value', 't1_value_head', 't1_own_logits', 't1_peer_logits', 'root_peer_logits',
    'raw_head_t2_value', 'raw_head_t1_value')   # (c), each when present; the depth-1 floor is reported


def check_heads(frame, data, d_results=None):
    """(c) head outputs against the oracle's heads, plus depth 1's own noise floor against the same oracle."""
    children, leaves, roots = data['children'], data['leaves'], data['roots'];parts = {}
    specs = leaves['specs']
    if len(specs):
        at = tuple(specs.T);scored = (leaves['status'] == 4) & (field(frame, 't2_status')[at] == 4)
        parts['t2_value'] = _masked(specs, field(frame, 't2_value')[at], leaves['head_value'], scored)
        if 'raw_head_value' in leaves:   # --raw-head-check: the pricer's D-arm value vs the head's own arm on the oracle's leaf inputs
            parts['raw_head_t2_value'] = _masked(specs, field(frame, 't2_value')[at], leaves['raw_head_value'], scored)
    cells = children['cells']
    if len(cells):
        at = tuple(cells.T);status, error = children['status'], children['error']
        same = (field(frame, 't1_status')[at] == status) & (field(frame, 't1_error')[at] == error)
        expanded = same & (status == 2);valued = same & ((status == 2) | np.isin(error, DEPTH2_CODES))
        parts['t1_value_head'] = _masked(cells, field(frame, 't1_value')[at], children['head_value'], valued)
        if 'raw_head_value' in children:
            parts['raw_head_t1_value'] = _masked(cells, field(frame, 't1_value')[at], children['raw_head_value'], valued)
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
    return {name: _numeric(*part, tolerance_for(name))[0] for name, part in parts.items()}


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


def gate_status(result, *, sampled=False, unavailable=None, skipped=None):
    """FAIL if any check that ran failed; INCOMPLETE if a check is unavailable or vacuous (no turn-2 leaf compared, or (e)
    compared no root); PASS_SAMPLED when the leaves were sampled ((e) skipped); else PASS. Returns (status, reasons)."""
    checks = result['checks']
    if any(not c['passed'] for c in checks.values()):return 'FAIL', []
    reasons = []
    if not checks:reasons.append('no check ran')
    if 'a' in checks and not checks['a']['compared'].get('t2_status'):reasons.append('vacuous: no turn-2 leaf was compared by (a)')
    if 'e' in checks and not checks['e'].get('roots_compared'):reasons.append('vacuous: (e) compared no root (every root refused or unlabeled)')
    reasons.extend(f'unavailable: {k}' for k in sorted(unavailable or ()))
    if reasons:return 'INCOMPLETE', reasons
    return ('PASS_SAMPLED' if sampled else 'PASS'), [f'skipped: {k} ({v})' for k, v in sorted((skipped or {}).items())]


def summarize(calls):
    """Merge per-call evaluations into the gate's checks: (b) discrete exact + floats within tolerance against D,
    (c) GATED head quantities within tolerance against the oracle."""
    have = lambda key: bool(calls) and all(key in x for x in calls)   # offline rechecks may lack a check's inputs
    checks = {}
    for key, name in (('a', 'a'), ('d', 'd'), ('oracle', 'oracle_consistency')):
        if have(key):checks[name] = merge_findings([c[key] for c in calls])
    if have('b'):
        b = merge_findings([c['b'] for c in calls]);values = merge_numeric([x['b'].get('values', {}) for x in calls])
        b.update(reference='D3_FRAMEPRICER_DISCRETE_EXACT_FLOATS_WITHIN_TOLERANCE (not bitwise)', value_tolerance=VALUE_TOLERANCE,
            logit_tolerance=LOGIT_TOLERANCE, values=values)
        b['passed'] = b['passed'] and all(v['within_tolerance'] for v in values.values());checks['b'] = b
    if have('c'):checks['c'] = _heads_check(calls)
    if any('e' in x for x in calls):checks['e'] = merge_decisions([x['e'] for x in calls if 'e' in x])
    if any('invariance' in x for x in calls):checks['block_invariance'] = merge_invariance([r for x in calls for r in x.get('invariance', ())])
    order = ('a', 'b', 'c', 'd', 'oracle_consistency', 'e', 'block_invariance')
    checks = {k: checks[k] for k in order if k in checks}
    return dict(passed=bool(checks) and all(x['passed'] for x in checks.values()), checks=checks)


def _heads_check(calls):
    quantities = merge_numeric([x['c'] for x in calls])
    for name, m in quantities.items():m['gates'] = 'c' if name in GATED else None
    floor = quantities.get('depth1_noise_floor', {}).get('max_abs_difference');t2 = quantities.get('t2_value', {}).get('max_abs_difference')
    c = dict(passed=all(quantities[n]['within_tolerance'] for n in GATED if n in quantities), value_tolerance=VALUE_TOLERANCE,
        logit_tolerance=LOGIT_TOLERANCE, tolerance_source=TOLERANCE_SOURCE, reference='ORACLE_BF16_HEADS_DIFFERENT_BATCH_GEOMETRY', gated=list(GATED),
        depth1_noise_floor=floor, t2_value_to_floor_ratio=None if not floor or t2 is None else t2 / floor, quantities=quantities)
    if any(x.get('geometry') for x in calls):c['passed'] = False
    return c


DECISION_FIRST = 10


def import_backups():
    """gpu_engine.depth2_backups through the import system (a None in sys.modules blocks it even once the package
    attribute exists, which `from gpu_engine import depth2_backups` would not honour). Raises ImportError if absent."""
    return importlib.import_module('gpu_engine.depth2_backups')


def oracle_heads_frame(frame, data):
    """A copy of the frame with the oracle's head outputs exactly where (c) compares them: t2_value at head-valued
    leaves (status 4 on both); at expanded children (status 2 on both) each REQUESTED seat's legal lanes of
    t1_own/t1_peer_logits; at usable worlds with a requested opponent the legal lanes of root_peer_logits. Every other
    lane keeps the pricer's value (a waiting seat's unused, possibly invalid, prior never enters the copy).
    Statuses, errors, terminal values and enumeration are the pricer's. Returns (copy, counts)."""
    fields = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in frame_fields(frame).items()}
    leaves, children = data['leaves'], data['children'];counts = dict(t2_value=0, t1_logits=0, unmatched_head_leaves=0)
    t2 = np.asarray(fields['t2_status']);done = np.zeros(t2.shape, bool)
    if len(leaves['specs']):
        at = tuple(leaves['specs'].T);head = (leaves['status'] == 4) & (t2[at] == 4)
        value = np.array(fields['t2_value'], np.float32, copy=True);value[tuple(leaves['specs'][head].T)] = leaves['head_value'][head]
        fields['t2_value'] = value;done[tuple(leaves['specs'][head].T)] = True;counts['t2_value'] = int(head.sum())
    counts['unmatched_head_leaves'] = int(np.count_nonzero((t2 == 4) & ~done))
    counts['root_logits'] = 0
    if len(children['cells']):
        at = tuple(children['cells'].T);expanded = (children['status'] == 2) & (np.asarray(fields['t1_status'])[at] == 2)
        for seat in ('own', 'peer'):
            lanes = expanded[:, None] & np.asarray(children[seat + '_requested'], bool)[:, None] & np.asarray(children[seat + '_legal'], bool)
            rows, lane = np.nonzero(lanes);cells = children['cells'][rows]
            logits = np.array(fields[f't1_{seat}_logits'], np.float32, copy=True)
            logits[tuple(cells.T) + (lane,)] = np.asarray(children[seat + '_logits'], np.float32)[rows, lane]
            fields[f't1_{seat}_logits'] = logits
        counts['t1_logits'] = int(expanded.sum())
    roots = data.get('roots') or {}
    keys = [k for k in sorted(roots) if roots[k]['pre'] == 0 and roots[k]['peer_requested'] and 'peer_logits_raw' in roots[k]]
    if keys:
        logits = np.array(fields['root_peer_logits'], np.float32, copy=True)
        for q, w in keys:
            legal = np.asarray(roots[(q, w)]['peer_legal'], bool);logits[q, w, legal] = np.asarray(roots[(q, w)]['peer_logits_raw'], np.float32)[legal]
        fields['root_peer_logits'] = logits;counts['root_logits'] = len(keys)
    return SimpleNamespace(**fields), counts


def _argmax(values, legal):
    best = None
    for i, (v, ok) in enumerate(zip(values, legal)):
        if ok and v is not None and (best is None or v > values[best]):best = i   # strict: ties keep the lower index
    return best


def decision_check(frame, data, requests, *, context=None, backups=None):
    """(e) for ONE call: learner-max root decisions on the pricer frame vs the oracle-heads copy."""
    if backups is None:
        try:backups = import_backups()
        except ImportError as error:
            return dict(passed=False, reason='gpu_engine.depth2_backups is not importable (fail closed): ' + str(error), roots=len(requests))
    original = SimpleNamespace(**frame_fields(frame));swapped, counts = oracle_heads_frame(frame, data)
    result = dict(roots=len(requests), roots_compared=0, both_refused=0, status_differs=0, nonfinite_roots=0, argmax_q_differs=0, argmax_target_differs=0,
        cells_compared=0, cell_choice_differs=0, max_abs_delta_q=0., max_abs_delta_target=0., replaced=counts, first_differing_roots=[],
        differing_roots=[])
    for q, request in enumerate(requests):
        logits, legal = tuple(map(float, request.root_logits)), tuple(map(bool, request.root_legal))
        a, b = (backups.learner_max_root(f, q, logits, legal) for f in (original, swapped))
        ia, ib = a.improvement, b.improvement
        if ia is None and ib is None:result['both_refused'] += 1
        qa, qb = (None if i is None else i.completed_q for i in (ia, ib));ta, tb = (None if i is None else i.policy_target for i in (ia, ib))
        pick = lambda v: None if v is None else _argmax(v, legal)
        q_differs, t_differs = pick(qa) != pick(qb), pick(ta) != pick(tb)
        if (ia is None) != (ib is None):result['status_differs'] += 1
        finite = all(v is None or math.isfinite(float(v)) for vector in (qa, qb, ta, tb) if vector is not None for v in vector)
        result['nonfinite_roots'] += int(not finite)
        if ia is not None and ib is not None and finite:
            result['roots_compared'] += 1
            deltas = [abs(x - y) for x, y, ok in zip(qa, qb, legal) if ok and x is not None and y is not None]
            result['max_abs_delta_q'] = max([result['max_abs_delta_q']] + deltas)
            result['max_abs_delta_target'] = max([result['max_abs_delta_target']] + [abs(x - y) for x, y in zip(ta, tb)])
        result['argmax_q_differs'] += int(q_differs);result['argmax_target_differs'] += int(t_differs)
        if q_differs or t_differs or not finite or (ia is None) != (ib is None):result['differing_roots'].append(q)
        if (q_differs or t_differs or not finite or (ia is None) != (ib is None)) and len(result['first_differing_roots']) < DECISION_FIRST:
            address = request.plans.root.address
            result['first_differing_roots'].append(dict(context or {}, root=q, address=[address.stream, address.frame, address.owner],
                status=[a.status, b.status], completed_q_pricer=_plain_q(qa), completed_q_oracle_heads=_plain_q(qb),
                target_pricer=_plain_q(ta), target_oracle_heads=_plain_q(tb), argmax_q=[pick(qa), pick(qb)], argmax_target=[pick(ta), pick(tb)]))
        for world in (0, 1):
            if int(original.pre[q, world]) != 0:continue
            ca, cb = backups.learner_max_choice(original, q, world), backups.learner_max_choice(swapped, q, world)
            both = (ca >= 0) | (cb >= 0);result['cells_compared'] += int(both.sum())
            result['cell_choice_differs'] += int(np.count_nonzero(both & (ca != cb)))
    result['passed'] = (result['argmax_q_differs'] == 0 and result['argmax_target_differs'] == 0 and counts['unmatched_head_leaves'] == 0
        and result['nonfinite_roots'] == 0)
    if counts['unmatched_head_leaves']:result['reason'] = 'head-valued pricer leaves without an oracle head value: the copy is incomplete (fail closed)'
    if result['nonfinite_roots']:result['reason'] = 'non-finite completed-Q or target: decisions are not comparable (fail closed)'
    return result


def _plain_q(values):
    return None if values is None else [None if v is None else float(v) for v in values]


def merge_decisions(reports):
    """Sum per-call decision reports; maxima of deltas; the first DECISION_FIRST differing roots."""
    keys = ('roots', 'roots_compared', 'both_refused', 'status_differs', 'nonfinite_roots', 'argmax_q_differs', 'argmax_target_differs',
        'cells_compared', 'cell_choice_differs')
    merged = {k: sum(r.get(k, 0) for r in reports) for k in keys}
    merged.update(max_abs_delta_q=max((r.get('max_abs_delta_q', 0.) for r in reports), default=0.),
        max_abs_delta_target=max((r.get('max_abs_delta_target', 0.) for r in reports), default=0.),
        first_differing_roots=[x for r in reports for x in r.get('first_differing_roots', ())][:DECISION_FIRST],
        replaced={k: sum((r.get('replaced') or {}).get(k, 0) for r in reports) for k in ('t2_value', 't1_logits', 'root_logits', 'unmatched_head_leaves')},
        reasons=sorted({r['reason'] for r in reports if r.get('reason')}), passed=bool(reports) and all(r['passed'] for r in reports),
        gating='passes iff argmax_q_differs == 0 and argmax_target_differs == 0 (a status flip counts in both); fails closed on '
            'non-finite Q/targets or an incomplete oracle-heads copy; cell_choice_differs is reported, not gated',
        reference='depth2_backups.learner_max_root: pricer frame vs the same frame with the oracle\'s head values, child priors and root '
            'opponent prior on the lanes (c) checks (requested seats, legal lanes)')
    return merged


BANDS = ((0, 4), (5, 9), (10, 19), (20, 39), (40, None))   # root address frame bands of the values section
PERCENTILES = (1, 10, 50, 90, 99)
PRUNING_KS = (2, 3, 4)
ARRAYS_SCHEMA = 'pokezero.depth2-gate-arrays.v2'


def band_of(frame_index):
    for low, high in BANDS:
        if high is None or frame_index <= high:return f'{low}-{high}' if high is not None else f'{low}+'


def percentiles(values):
    values = np.asarray(values, np.float64).reshape(-1);values = values[np.isfinite(values)]
    if not values.size:return dict(count=0)
    return dict(count=int(values.size), **{f'p{p}': float(v) for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))})


def cell_rows(t2_status, t2_value, peer_logits):
    """One expanded cell's learner-max a2 values sum_r2 pi_opp(r2) v(a2, r2) ([9], NaN where a2 is not enumerated),
    as the reference backup computes them: pi_opp is the masked softmax of the opponent's child prior over the
    enumerated replies. None when an enumerated leaf errored (the reference refuses the cell)."""
    t2_status = np.asarray(t2_status);t2_value = np.asarray(t2_value, np.float64)
    if (t2_status == 3).any():return None
    own, opp = (t2_status != 0).any(axis=1), (t2_status != 0).any(axis=0)
    if not own.any():return None
    logits = np.asarray(peer_logits, np.float64)[opp];prior = np.exp(logits - logits.max());prior /= prior.sum()
    rows = np.full(9, np.nan)
    for a2 in np.flatnonzero(own):rows[a2] = float(np.sum(prior * t2_value[a2, opp]))
    return rows


def value_samples(frame, requests, *, backups=None, context=None):
    """Raw per-root samples of the values section, from the PRICER's frame: |t2| at head-valued leaves, |t1| at
    expanded cells, each expanded cell's learner-max a2 spread (>= 2 enumerated a2), and per labeled root the
    learner-max completed-Q spread and top-two gap (depth2_backups.learner_max_root)."""
    f = SimpleNamespace(**frame_fields(frame));samples = collections.defaultdict(list);roots = [];reason = None
    if backups is None:
        try:backups = import_backups()
        except ImportError as error:reason = 'root spread and near ties need gpu_engine.depth2_backups: ' + str(error)
    for q, request in enumerate(requests):
        address = request.plans.root.address;band = band_of(int(address.frame))
        samples[('abs_t2_value', band)].append(np.abs(np.asarray(f.t2_value[q], np.float64)[np.asarray(f.t2_status[q]) == 4]))
        samples[('abs_t1_value', band)].append(np.abs(np.asarray(f.t1_value[q], np.float64)[np.asarray(f.t1_status[q]) == 2]))
        spreads = []
        for w, a1, r1 in np.argwhere(np.asarray(f.t1_status[q]) == 2).tolist():
            rows = cell_rows(f.t2_status[q, w, a1, r1], f.t2_value[q, w, a1, r1], f.t1_peer_logits[q, w, a1, r1])
            if rows is not None and np.count_nonzero(~np.isnan(rows)) >= 2:spreads.append(np.nanmax(rows) - np.nanmin(rows))
        samples[('cell_a2_spread', band)].append(np.asarray(spreads))
        if backups is None:continue
        legal = tuple(map(bool, request.root_legal));reduction = backups.learner_max_root(f, q, tuple(map(float, request.root_logits)), legal)
        if reduction.improvement is None:continue
        values = sorted((float(v) for v, ok in zip(reduction.improvement.completed_q, legal) if ok and v is not None), reverse=True)
        if not values:continue
        samples[('root_q_spread', band)].append(np.asarray([values[0] - values[-1]]))
        roots.append(dict(context or {}, root=q, address=[address.stream, address.frame, address.owner], band=band,
            top_two_gap=None if len(values) < 2 else values[0] - values[1]))
    return dict(samples=dict(samples), roots=roots, unavailable=reason)


def values_section(collected, *, oracle_checked):
    """collected: [(value_samples(...), decision result of the same call or None)]. Percentiles per frame band and in total,
    and the near-tie share (top-two root Q within VALUE_TOLERANCE) with (e)'s verdict on each near-tie root."""
    pooled = collections.defaultdict(list);roots = [];reasons = set()
    for result, decision in collected:
        for key, parts in result['samples'].items():pooled[key].extend(parts)
        if result['unavailable']:reasons.add(result['unavailable'])
        differs = None if decision is None or 'differing_roots' not in decision else set(decision['differing_roots'])
        roots.extend(dict(r, e_evaluated=differs is not None, e_differs=None if differs is None else r['root'] in differs) for r in result['roots'])
    quantities = ('abs_t2_value', 'abs_t1_value', 'cell_a2_spread', 'root_q_spread');bands = [band_of(low) for low, _ in BANDS]
    join = lambda parts: np.concatenate([np.asarray(x, np.float64).reshape(-1) for x in parts]) if parts else np.zeros(0)
    by_band = {b: {name: percentiles(join(pooled.get((name, b), []))) for name in quantities} for b in bands}
    totals = {name: percentiles(join([x for b in bands for x in pooled.get((name, b), [])])) for name in quantities}
    near = [r for r in roots if r['top_two_gap'] is not None and r['top_two_gap'] <= VALUE_TOLERANCE]
    labeled = len(roots)
    return dict(source='PRICER_FRAME', oracle_checked=oracle_checked,
        scope='the pricer frame\'s own values' + (' (every leaf checked by the oracle: (a)-(e))' if oracle_checked else
            ' - NOT oracle-checked (sampled leaves or --pricer-only)'),
        frame_bands=bands, percentiles=list(PERCENTILES), by_band=by_band, totals=totals,
        definitions=dict(abs_t2_value='|t2_value| at head-valued leaves (status 4)', abs_t1_value='|t1_value| at expanded cells (status 2)',
            cell_a2_spread='max - min over enumerated a2 of sum_r2 pi_opp(r2) v(a2, r2), expanded cells with >= 2 enumerated a2',
            root_q_spread='max - min of learner-max completed-Q over legal root actions, labeled roots'),
        near_ties=dict(threshold=VALUE_TOLERANCE, labeled_roots=labeled, near_tie_roots=len(near), share=None if not labeled else len(near) / labeled,
            by_band={b: sum(r['band'] == b for r in near) for b in bands}, e_evaluated=sum(r['e_evaluated'] for r in near),
            e_exact_on_all=all(r['e_differs'] is False for r in near) if near and all(r['e_evaluated'] for r in near) else None,
            e_differs_on=[r for r in near if r['e_differs']], roots=near[:FIRST]),
        unavailable=sorted(reasons))


def sibling_spread(frame, *, context=None):
    """Per usable world: median and 10th percentile of |v(a2,r2) - v(a2',r2')| over pairs of head-valued sibling leaves
    of the same expanded child (pricer t2_value), and each child's own 10th percentile."""
    f = SimpleNamespace(**frame_fields(frame));worlds = [];children = []
    for q, w in np.argwhere(np.asarray(f.pre) == 0).tolist():
        pooled = []
        for a1, r1 in np.argwhere(np.asarray(f.t1_status[q, w]) == 2).tolist():
            values = np.asarray(f.t2_value[q, w, a1, r1], np.float64)[np.asarray(f.t2_status[q, w, a1, r1]) == 4]
            if len(values) < 2:continue
            pairs = np.abs(values[:, None] - values[None, :])[np.triu_indices(len(values), 1)]
            pooled.append(pairs);children.append(float(np.percentile(pairs, 10)))
        if pooled:
            pairs = np.concatenate(pooled)
            worlds.append(dict(context or {}, root=q, world=w, children=len(pooled), pairs=int(pairs.size),
                median=float(np.median(pairs)), p10=float(np.percentile(pairs, 10))))
    return dict(worlds=worlds, child_p10=children)


def merge_sibling_spread(reports, *, oracle_checked):
    worlds = [w for r in reports for w in r['worlds']];children = np.asarray([c for r in reports for c in r['child_p10']], np.float64)
    below = int(np.count_nonzero(children < VALUE_TOLERANCE))
    return dict(source='PRICER_FRAME', oracle_checked=oracle_checked, threshold=VALUE_TOLERANCE, worlds=worlds,
        children=int(children.size), children_p10_below_threshold=below, share_children_p10_below_threshold=None if not children.size else below / children.size,
        world_median=percentiles([w['median'] for w in worlds]), world_p10=percentiles([w['p10'] for w in worlds]), child_p10=percentiles(children),
        definition='per world: |v(a2,r2) - v(a2\',r2\')| over pairs of head-valued sibling leaves of one expanded child (pricer t2_value)')


def pruning_report(items, backups):
    """For each expanded cell of a usable world: is learner-max's a2 (learner_max_choice) outside the learner's top-k
    by child prior (logit descending, lower index on ties; non-finite last)? Per cell, and over the cells under each
    labeled root's chosen turn-1 action (completed-Q argmax, lower index on ties). items: [(frame, requests, context)]
    of FULL-WIDTH frames: a cell whose enumerated learner actions differ from its legal set refuses the report."""
    cells = dict(total=0, **{f'outside_top{k}': 0 for k in PRUNING_KS});chosen = dict(cells)
    roots = dict(labeled=0, refused=0, **{f'any_outside_top{k}': 0 for k in PRUNING_KS});ranks = [0] * 9;narrow = []
    for frame, requests, context in items:
        f = SimpleNamespace(**frame_fields(frame))
        for q, request in enumerate(requests):
            legal = tuple(map(bool, request.root_legal));reduction = backups.learner_max_root(f, q, tuple(map(float, request.root_logits)), legal)
            best = None if reduction.improvement is None else _argmax(reduction.improvement.completed_q, legal)
            roots['labeled' if best is not None else 'refused'] += 1;outside = {k: False for k in PRUNING_KS}
            for w in (0, 1):
                if int(f.pre[q, w]) != 0:continue
                choice = backups.learner_max_choice(f, q, w)
                for a1, r1 in np.argwhere(np.asarray(choice) >= 0).tolist():
                    enumerated = np.asarray(f.t2_status[q, w, a1, r1] != 0).any(axis=1)
                    requested = bool(f.t1_own_requested[q, w, a1, r1])
                    expected = np.asarray(f.t1_own_legal[q, w, a1, r1], bool) if requested else np.arange(9) == 0
                    if not np.array_equal(enumerated, expected):narrow.append(dict(context or {}, cell=[q, w, a1, r1]));continue
                    logits = np.asarray(f.t1_own_logits[q, w, a1, r1], np.float64)
                    score = lambda i: logits[i] if np.isfinite(logits[i]) else -np.inf
                    order = sorted(np.flatnonzero(enumerated).tolist(), key=lambda i: (-score(i), i));rank = order.index(int(choice[a1, r1]))
                    ranks[rank] += 1;cells['total'] += 1
                    under = best is not None and a1 == best
                    if under:chosen['total'] += 1
                    for k in PRUNING_KS:
                        cells[f'outside_top{k}'] += int(rank >= k)
                        if under:chosen[f'outside_top{k}'] += int(rank >= k);outside[k] = outside[k] or rank >= k
            if best is not None:
                for k in PRUNING_KS:roots[f'any_outside_top{k}'] += int(outside[k])
    share = lambda part, total: None if not total else part / total
    return dict(full_width=not narrow, not_full_width_cells=narrow[:FIRST], not_full_width_count=len(narrow),
        per_cell={**cells, **{f'share_outside_top{k}': share(cells[f'outside_top{k}'], cells['total']) for k in PRUNING_KS}},
        under_chosen_root_action={**chosen, **{f'share_outside_top{k}': share(chosen[f'outside_top{k}'], chosen['total']) for k in PRUNING_KS}},
        roots={**roots, **{f'share_any_outside_top{k}': share(roots[f'any_outside_top{k}'], roots['labeled']) for k in PRUNING_KS}},
        chosen_a2_rank_histogram=ranks,
        definition='rank of learner_max_choice\'s a2 among the cell\'s enumerated learner actions by t1_own_logits (descending, lower index on ties); '
            'outside top-k means rank >= k; the chosen root action is the completed-Q argmax of learner_max_root (lower index on ties)')


def _d_arrays(d_results):
    """FramePricer Results as arrays: what check (b) and the depth-1 refusal share read."""
    count = len(d_results)
    out = dict(d_status=np.asarray([r.status for r in d_results], '<U16'), d_priced=np.zeros((count, 2), bool),
        d_reason=np.full((count, 2), '', '<U200'), d_opponent_requested=np.zeros((count, 2), bool), d_reply_legal=np.zeros((count, 2, 9), bool),
        d_reply_logits=np.zeros((count, 2, 9), np.float32), d_value=np.full((count, 2, 9, 9), np.nan, np.float32),
        d_known=np.zeros((count, 2, 9, 9), bool), d_error=np.zeros((count, 2, 9, 9), np.int8))
    for q, result in enumerate(d_results):
        for wi, report in enumerate(result.world_reports):
            grid = report.get('joint_values')
            if grid is None:out['d_reason'][q, wi] = (report.get('reason') or '')[:200];continue
            out['d_priced'][q, wi] = True;out['d_opponent_requested'][q, wi] = bool(report['opponent_requested'])
            out['d_reply_legal'][q, wi] = report['reply_legal'];out['d_reply_logits'][q, wi] = report['reply_logits']
            for a in range(9):
                for r in range(9):
                    if grid[a][r] is not None:out['d_value'][q, wi, a, r] = grid[a][r];out['d_known'][q, wi, a, r] = True
            for a, r, reason in report.get('errors', ()):out['d_error'][q, wi, a, r] = D_ERRORS.get(reason.removeprefix('ValueError: '), -1)
    return out


def _d_results(arrays):
    """FramePricer-shaped Results rebuilt from _d_arrays (reasons keep D's strings; unknown codes stay unknown)."""
    names = {code: reason for reason, code in D_ERRORS.items()};results = []
    for q, status in enumerate(arrays['d_status'].tolist()):
        reports = []
        for wi in (0, 1):
            if not arrays['d_priced'][q, wi]:reports.append(dict(joint_values=None, reason=str(arrays['d_reason'][q, wi])));continue
            known, value = arrays['d_known'][q, wi], arrays['d_value'][q, wi]
            grid = [[float(value[a, r]) if known[a, r] else None for r in range(9)] for a in range(9)]
            errors = tuple((a, r, 'ValueError: ' + names.get(int(arrays['d_error'][q, wi, a, r]), 'unrecognized D reason'))
                for a, r in np.argwhere(arrays['d_error'][q, wi] != 0).tolist())
            reports.append(dict(joint_values=grid, opponent_requested=bool(arrays['d_opponent_requested'][q, wi]),
                reply_legal=tuple(map(bool, arrays['d_reply_legal'][q, wi])), reply_logits=tuple(map(float, arrays['d_reply_logits'][q, wi])), errors=errors))
        results.append(SimpleNamespace(status=status, world_reports=tuple(reports)))
    return results


ROOT_FIELDS = ('pre', 'own_error', 'peer_error', 'peer_requested', 'peer_usable', 'own_legal', 'peer_legal', 'peer_logits',
    'peer_logits_raw', 'own_digest', 'peer_digest')


def save_arrays(path, frame, requests, *, data=None, d_results=None, invariance_frames=(), meta=None):
    """Everything --recheck needs, as .npz: the pricer frame (frame_*), any other child-block frame (invariance{i}_frame_*),
    the requests' root logits/legal/address/input keys, the oracle's roots/children/leaves/enumeration (oracle_*), D3
    FramePricer's (b) outputs (d_*) and the call's meta (JSON). Returns the saved keys."""
    arrays = {}
    def put_frame(prefix, value):
        for name, x in frame_fields(value).items():
            x = np.asarray(x) if not isinstance(x, dict) else None
            if x is not None and x.dtype.kind in 'biufc':arrays[prefix + name] = x
    put_frame('frame_', frame)
    for i, other in enumerate(invariance_frames):put_frame(f'invariance{i}_frame_', other)
    arrays.update(request_root_logits=np.asarray([r.root_logits for r in requests], np.float32).reshape(-1, 9),
        request_root_legal=np.asarray([r.root_legal for r in requests], bool).reshape(-1, 9),
        request_address=np.asarray([[r.plans.root.address.stream, r.plans.root.address.frame, r.plans.root.address.owner] for r in requests], np.int64).reshape(-1, 3),
        request_input_keys=np.asarray([[k in r.root_inputs for k in INPUT_ORDER] for r in requests], bool).reshape(-1, len(INPUT_ORDER)))
    if data is not None:
        keys = sorted(data['roots']);arrays['oracle_root_keys'] = np.asarray(keys, np.int64).reshape(-1, 2)
        for name in ROOT_FIELDS:
            values = [np.asarray(data['roots'][k][name]) for k in keys]
            if values:arrays['oracle_root_' + name] = np.stack(values)
        for name, value in data['children'].items():arrays['oracle_child_' + name] = np.asarray(value)
        for name, value in data['leaves'].items():
            if name != 'logits':arrays['oracle_leaf_' + name] = np.asarray(value)
        arrays['oracle_enumerated'] = np.asarray(data['enumerated'], np.int64).reshape(-1, 6)
    if d_results is not None:arrays.update(_d_arrays(d_results))
    timing = _frame_timing(frame)   # refusal_summary reads frame.timing (k1 widths)
    arrays['meta'] = np.asarray(json.dumps(dict(meta or {}, schema=ARRAYS_SCHEMA, frame_timing=timing if isinstance(timing, dict) else None), default=str))
    np.savez_compressed(path, **arrays)
    return sorted(arrays)


def load_arrays(path):
    """A saved call, rebuilt: frame, invariance frames, request shims, oracle data, D results (None where absent) and notes."""
    with np.load(path, allow_pickle=False) as saved:arrays = {k: saved[k] for k in saved.files}
    notes = [];v1 = 'meta' not in arrays
    meta = dict(schema='v1 (5ea0552b --save-arrays)') if v1 else json.loads(str(arrays.pop('meta')))
    if v1:notes.append('v1 arrays (no meta): full widths and every leaf checked are assumed')
    frame = {k[6:]: v for k, v in arrays.items() if k.startswith('frame_')}
    if isinstance(meta.get('frame_timing'), dict):frame['timing'] = meta['frame_timing']
    elif v1:frame['timing'] = {}   # refusal_summary reads timing's k1 widths: v1 runs are full width (see the v1 note)
    invariance = []
    while any(k.startswith(f'invariance{len(invariance)}_frame_') for k in arrays):
        prefix = f'invariance{len(invariance)}_frame_';invariance.append({k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)})
    keys = arrays.get('request_input_keys')
    if keys is None:notes.append('request input keys not saved: assumed all of ' + ','.join(INPUT_ORDER))
    requests = []
    for q, (logits, legal, address) in enumerate(zip(arrays['request_root_logits'], arrays['request_root_legal'], arrays['request_address'])):
        present = INPUT_ORDER if keys is None else tuple(k for k, use in zip(INPUT_ORDER, keys[q]) if use)
        requests.append(SimpleNamespace(root_logits=tuple(map(float, logits)), root_legal=tuple(map(bool, legal)), root_inputs={k: None for k in present},
            plans=SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=int(address[0]), frame=int(address[1]), owner=int(address[2]))))))
    children = {k[len('oracle_child_'):]: v for k, v in arrays.items() if k.startswith('oracle_child_')}
    leaves = {k[len('oracle_leaf_'):]: v for k, v in arrays.items() if k.startswith('oracle_leaf_')}
    roots = None
    if 'oracle_root_keys' in arrays:
        roots = {}
        for i, (q, w) in enumerate(arrays['oracle_root_keys'].tolist()):
            root = {name: arrays['oracle_root_' + name][i] for name in ROOT_FIELDS if 'oracle_root_' + name in arrays}
            for name in ('own_legal', 'peer_legal'):root[name] = tuple(map(bool, root[name]))
            for name in ('pre', 'own_error', 'peer_error'):root[name] = int(root[name])
            for name in ('peer_requested', 'peer_usable'):root[name] = bool(root[name])
            roots[(q, w)] = root
    enumerated = arrays['oracle_enumerated'].tolist() if 'oracle_enumerated' in arrays else None
    if enumerated is None and 'specs' in leaves:
        enumerated = leaves['specs'].tolist();notes.append('oracle enumeration not saved: taken from the checked leaves (every leaf assumed checked)')
    d_results = _d_results(arrays) if 'd_status' in arrays else None
    return dict(meta=meta, frame=frame, invariance=invariance, requests=requests, roots=roots, children=children, leaves=leaves,
        enumerated=[tuple(x) for x in enumerated] if enumerated is not None else None, d_results=d_results, notes=notes, keys=sorted(arrays))


RECHECK_NEEDS = dict(
    a=dict(roots=(), children=('cells', 'status', 'error', 'value', 'own_digest', 'peer_digest', 'own_legal', 'peer_legal', 'own_requested', 'peer_requested'),
        leaves=('specs', 'status', 'error', 'value', 'digest')),
    b=dict(d=()), c=dict(children=('cells', 'status', 'head_value', 'own_logits', 'peer_logits'), leaves=('specs', 'status', 'head_value')),
    d=dict(children=('cells',), enumerated=()), oracle=dict(children=('cells', 'status', 'invalid', 'done', 'own_digest', 'peer_digest'),
        leaves=('specs', 'child_invalid', 'child_done', 'child_own_digest', 'child_peer_digest')),
    e=dict(children=('cells', 'status', 'own_logits', 'peer_logits'), leaves=('specs', 'status', 'head_value')))


def _missing(saved, check):
    """The saved array keys check `check` lacks (empty when it can run)."""
    missing = []
    for part, fields in RECHECK_NEEDS[check].items():
        if part == 'roots' and saved['roots'] is None:missing.append('oracle_root_*')
        elif part == 'd' and saved['d_results'] is None:missing.append('d_* (FramePricer outputs)')
        elif part == 'enumerated' and saved['enumerated'] is None:missing.append('oracle_enumerated')
        elif part in ('children', 'leaves'):
            prefix = 'oracle_child_' if part == 'children' else 'oracle_leaf_'
            missing.extend(prefix + name for name in fields if name not in saved[part])
    return missing


def recheck_call(saved, *, context, backups=None):
    """Every check the saved arrays allow, as the live gate computes it; returns (evaluation, unavailable, substitutions)."""
    frame, requests, d_results = saved['frame'], saved['requests'], saved['d_results'];unavailable = {};substitutions = []
    children, leaves = dict(saved['children']), dict(saved['leaves'])
    data = dict(roots=saved['roots'] or {}, children=children, leaves=leaves, enumerated=saved['enumerated'] or [])
    geometry = frame_geometry(frame, len(requests));evaluation = dict(geometry=geometry)
    if geometry:return dict(geometry=geometry), dict(all=['frame geometry: ' + '; '.join(geometry)]), substitutions
    widths = saved['meta'].get('widths', dict(k1_own=9, k1_opp=9));full_turn1 = widths.get('k1_own', 9) == widths.get('k1_opp', 9) == 9
    for check, run in (('a', lambda: check_exact(frame, data, Findings(context)).report()),
            ('b', lambda: (lambda found: dict(found.report(), values=found.values))(check_d(frame, d_results, requests, Findings(context), full_turn1=full_turn1))),
            ('d', lambda: check_enumeration(frame, data, Findings(context)).report()),
            ('oracle', lambda: check_oracle(data, Findings(context)).report())):
        missing = _missing(saved, check)
        if missing:unavailable[check] = missing
        else:evaluation[check] = run()
    if len(children.get('cells', ())):   # arrays saved before 41e45122 lack the oracle's child error and masks
        at = tuple(np.asarray(children['cells']).T)
        for name, source in (('error', 't1_error'), ('own_requested', 't1_own_requested'), ('own_legal', 't1_own_legal'),
                ('peer_requested', 't1_peer_requested'), ('peer_legal', 't1_peer_legal')):
            if name not in children:children[name] = np.asarray(frame[source])[at];substitutions.append(f'(c)/(e) oracle child {name} taken from the pricer frame')
    missing = _missing(saved, 'c')
    if missing:unavailable['c'] = missing
    else:
        if saved['roots'] is None:substitutions.append('(c) root_peer_logits not compared: oracle roots not saved')
        if d_results is None:substitutions.append('(c) depth-1 noise floor not computed: FramePricer outputs not saved')
        evaluation['c'] = check_heads(frame, data, d_results)
    if saved['invariance']:evaluation['invariance'] = [block_invariance(other, frame, context=context) for other in saved['invariance']]
    elif saved['meta'].get('invariance') is False:evaluation['skipped_block_invariance'] = 'the run did not request --block-invariance'
    else:unavailable['block_invariance'] = ['invariance0_frame_* (the other child-block frame)']
    missing = _missing(saved, 'e')
    if saved['meta'].get('sample_leaves') is not None:evaluation['skipped_e'] = 'SKIPPED_SAMPLED: (e) needs every leaf checked'
    elif missing:unavailable['e'] = missing
    elif backups is None:unavailable['e'] = ['gpu_engine.depth2_backups (not importable)']
    else:evaluation['e'] = decision_check(frame, data, requests, context=context, backups=backups)
    if d_results is None:unavailable['refusals_depth1'] = ['d_* (FramePricer outputs)']
    evaluation['refusals'] = refusal_report(frame, d_results, data, root_legal=[r.root_legal for r in requests])
    if 'topk' in saved['meta']:evaluation['topk'] = saved['meta']['topk']
    return evaluation, unavailable, substitutions


def saved_calls(directory):
    files = sorted(Path(directory).glob('arrays-block*-call*.npz'))
    if not files:raise SystemExit(f'no arrays-block*-call*.npz in {directory}')
    for path in files:
        block, call = (int(x) for x in path.stem.removeprefix('arrays-block').split('-call'))
        yield path, dict(block=block, call=call), load_arrays(path)


def _backups():
    try:
        return import_backups()
    except ImportError:return None


def recheck(directory, *, backups=None, progress=None):
    """--recheck: every check, the values section and the sibling spread from saved arrays, without the GPU or the engine."""
    backups = backups or _backups();evaluations, unavailable, substitutions, notes, collected, sibling = [], {}, set(), set(), [], []
    checked = True;evaluator = None
    for path, context, saved in saved_calls(directory):
        checked = checked and not saved['meta'].get('sample_leaves');evaluator = evaluator or saved['meta'].get('evaluator')
        with stage(progress, 'recheck_call', file=path.name, **context) as info:
            evaluation, missing, subs = recheck_call(saved, context=context, backups=backups)
            evaluations.append(evaluation);substitutions.update(subs);notes.update(saved['notes'])
            for check, keys in missing.items():unavailable.setdefault(check, set()).update(keys)
            collected.append((value_samples(saved['frame'], saved['requests'], backups=backups, context=context), evaluation.get('e')))
            sibling.append(sibling_spread(saved['frame'], context=context));info['checks'] = sorted(k for k in evaluation if k != 'geometry')
    result = summarize(evaluations)
    refusals = merge_refusals([e['refusals'] for e in evaluations if 'refusals' in e])
    skipped = {key: e[f'skipped_{key}'] for e in evaluations for key in ('e', 'block_invariance') if f'skipped_{key}' in e}
    status, reasons = gate_status(result, sampled=not checked, unavailable=unavailable, skipped=skipped)
    if any(e.get('geometry') for e in evaluations):status = 'FAIL'
    return dict(evaluator=evaluator, schema=SCHEMA, mode='RECHECK', status=status, passed=status in ('PASS', 'PASS_SAMPLED'), status_reasons=reasons,
        skipped=skipped, source=str(directory), calls=len(evaluations),
        checks=result['checks'], unavailable={k: sorted(v) for k, v in unavailable.items()}, substitutions=sorted(substitutions), notes=sorted(notes),
        values=values_section(collected, oracle_checked=checked and 'e' in result['checks']),
        sibling_spread=merge_sibling_spread(sibling, oracle_checked=checked), refusals=refusals,
        topk_boundary=merge_topk_boundary([e.get('topk') for e in evaluations]),
        value_tolerance=VALUE_TOLERANCE, logit_tolerance=LOGIT_TOLERANCE, tolerance_source=TOLERANCE_SOURCE, conventions=list(CONVENTIONS))


def _roots_by_address(directory):
    roots = {}
    for _, context, saved in saved_calls(directory):
        for q, request in enumerate(saved['requests']):
            a = request.plans.root.address;roots[(context['block'], a.stream, a.frame, a.owner)] = (saved, q)
    return roots


def topk_error_share(full_dir, topk_dir):
    """--topk-errors: under top-k, the share of roots whose only full-width depth-2 errors sit in cells top-k prunes away.

    Roots are matched by (block, address) across a full-width and a top-k array set of the same requests. A root's
    full-width errors are its status-3 turn-1 cells and turn-2 leaves; one is pruned away when the top-k frame does not
    enumerate it. Roots refused by pre/materialization have no cell errors and are counted apart."""
    full, topk = _roots_by_address(full_dir), _roots_by_address(topk_dir)
    widths = lambda roots: {json.dumps(s['meta'].get('widths'), sort_keys=True) for s, _ in roots.values()}
    full_widths = widths(full)
    if any(w != 'null' and any(v != 9 for v in json.loads(w).values()) for w in full_widths):
        raise SystemExit(f'--topk-errors: {full_dir} is not a full-width array set (widths {sorted(full_widths)})')
    counts = collections.Counter();examples = []
    for key in sorted(set(full) & set(topk)):
        (fs, fq), (ts, tq) = full[key], topk[key];f, t = fs['frame'], ts['frame'];counts['matched'] += 1
        if (np.asarray(f['pre'][fq]) != 0).any():counts['pre_or_materialization'] += 1;continue
        e1 = np.argwhere(np.asarray(f['t1_status'][fq]) == 3).tolist();e2 = np.argwhere(np.asarray(f['t2_status'][fq]) == 3).tolist()
        k1 = np.argwhere(np.asarray(t['t1_status'][tq]) == 3).tolist();k2 = np.argwhere(np.asarray(t['t2_status'][tq]) == 3).tolist()
        counts['topk_errors_not_at_full_width'] += len({tuple(x) for x in k1} - {tuple(x) for x in e1}) + len({tuple(x) for x in k2} - {tuple(x) for x in e2})
        if not e1 and not e2:counts['no_full_width_errors'] += 1;continue
        counts['with_full_width_errors'] += 1
        kept = [x for x in e1 if t['t1_status'][(tq,) + tuple(x)] != 0] + [x for x in e2 if t['t2_status'][(tq,) + tuple(x)] != 0]
        if kept:counts['errors_kept_by_topk'] += 1
        else:
            counts['only_pruned_errors'] += 1
            if len(examples) < FIRST:examples.append(dict(block=key[0], address=list(key[1:]), turn1_errors=len(e1), turn2_errors=len(e2)))
    share = lambda part, total: None if not total else part / total
    return dict(schema=SCHEMA, mode='TOPK_ERRORS', full=str(full_dir), topk=str(topk_dir), full_widths=sorted(full_widths), topk_widths=sorted(widths(topk)),
        unmatched=dict(full_only=len(set(full) - set(topk)), topk_only=len(set(topk) - set(full))), counts=dict(counts),
        share_of_matched_roots=share(counts['only_pruned_errors'], counts['matched']),
        share_of_roots_with_full_width_errors=share(counts['only_pruned_errors'], counts['with_full_width_errors']), examples=examples,
        definition='a root counts when it has full-width status-3 cells/leaves and the top-k frame enumerates none of them')


def offline_pruning(directory, *, backups=None):
    backups = backups or _backups()
    if backups is None:raise SystemExit('--pruning-report needs gpu_engine.depth2_backups (learner_max_root, learner_max_choice): failing closed')
    items = [(saved['frame'], saved['requests'], context) for _, context, saved in saved_calls(directory)]
    return dict(schema=SCHEMA, mode='PRUNING_REPORT', source=str(directory), calls=len(items), ks=list(PRUNING_KS), **pruning_report(items, backups))


def error_counts(status, error, codes):
    """{code: cells with status 3 and that code}, every listed code present, any unexpected code added."""
    status, error = np.asarray(status), np.asarray(error);failed = error[status == 3]
    return {str(c): int(np.count_nonzero(failed == c)) for c in sorted(set(codes) | set(map(int, np.unique(failed))))}


def depth2_refused(frame, root_legal=None):
    """The gate's own reference-rule count: a root refuses if a world has pre != 0 or any cell has status 3, or (given
    root_legal [R,9]) a legal root action has no enumerated turn-1 row in a usable world (k1_own pruning: the reduction
    has no value for it)."""
    pre, t1, t2 = (field(frame, n) for n in ('pre', 't1_status', 't2_status'));count = len(pre)
    refused = np.any(pre != 0, axis=1) | np.any(t1.reshape(count, -1) == 3, axis=1) | np.any(t2.reshape(count, -1) == 3, axis=1)
    if root_legal is not None:
        rows = (t1 != 0).any(axis=-1)   # [R,2,9]: an enumerated turn-1 row per root action
        pruned = np.asarray(root_legal, bool)[:, None, :] & ~rows & (pre == 0)[:, :, None]
        refused |= pruned.reshape(count, -1).any(axis=1)
    return refused


def depth2_summary(frame, root_legal=None):
    """depth2_backups.refusal_summary(frame, root_legal=) when importable (it lands with the pricer), else (None, reason).
    root_legal makes its k1_own pruning exact; without it every usable root of a k1_own<9 frame is flagged."""
    try:refusal_summary = import_backups().refusal_summary
    except (ImportError, AttributeError) as error:return None, 'not importable: ' + str(error)
    view = SimpleNamespace(**frame) if isinstance(frame, dict) else frame
    exact = root_legal is not None and 'root_legal' in inspect.signature(refusal_summary).parameters
    try:return (refusal_summary(view, root_legal=np.asarray(root_legal, bool)) if exact else refusal_summary(view)), None
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:return None, f'{type(error).__name__}: {error}'


def refusal_report(frame, d_results=None, data=None, root_legal=None):
    """One call's root refusals at depth 2 (reference rule) and depth 1 (D3 FramePricer, when it ran), and error counts by
    code. root_legal ([R,9], the requests' root legal masks) makes k1_own pruning refusals exact."""
    refused = depth2_refused(frame, root_legal)
    own = dict(source='gate count of the frame', root_count=len(refused), refused_count=int(refused.sum()), refused_roots=np.flatnonzero(refused).tolist())
    report = dict(depth2=own, turn1_errors_by_code=error_counts(field(frame, 't1_status'), field(frame, 't1_error'), range(1, 7)),
        turn2_errors_by_code=error_counts(field(frame, 't2_status'), field(frame, 't2_error'), range(1, 5)))
    if d_results is not None:
        report['depth1'] = dict(source='FramePricer(vectorized=True, host_prep=True)', root_count=len(d_results),
            refused_count=sum(r.status == 'REFUSED' for r in d_results), refused_roots=[i for i, r in enumerate(d_results) if r.status == 'REFUSED'])
    if data is not None:
        children, leaves = data['children'], data['leaves']
        if 'error' in children:report['oracle_turn1_errors_by_code'] = error_counts(children.get('status', []), children.get('error', []), range(1, 7))
        if 'error' in leaves:report['oracle_turn2_errors_by_code_checked_leaves'] = error_counts(leaves.get('status', []), leaves.get('error', []), range(1, 5))
    summary, reason = depth2_summary(frame, root_legal)
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


def topk_boundary(children, roots, widths, priors, root_priors, reference):
    """Under top-k: each pruned selection's boundary gap (rank-k minus rank-(k+1) logit of the prior that ranked it) and
    how many selections the oracle's own prior would have made differently. None at full width."""
    if all(widths[k] == 9 for k in ('k_own', 'k_opp', 'k1_opp')):return None
    out = dict(gaps=dict(own=[], peer=[], root_peer=[]), oracle_prior_set_differs=dict(own=0, peer=0, root_peer=0),
        ranked_by='PRICER_PRIORS' if priors is not None or root_priors is not None else 'ORACLE_PRIORS')
    if len(children.get('cells', ())) and (widths['k_own'] < 9 or widths['k_opp'] < 9):
        for i in np.flatnonzero(np.asarray(children['status']) == 2).tolist():
            for seat, k in (('own', widths['k_own']), ('peer', widths['k_opp'])):
                if k == 9 or not children[seat + '_requested'][i]:continue
                legal = children[seat + '_legal'][i];used = children[seat + '_logits'][i] if priors is None else priors[seat][i]
                gap = reference.boundary_gap(legal, used, k)
                if gap is None:continue
                out['gaps'][seat].append(gap)
                out['oracle_prior_set_differs'][seat] += int(reference.top_k(legal, used, k) != reference.top_k(legal, children[seat + '_logits'][i], k))
    if widths['k1_opp'] < 9:
        for key, root in sorted(roots.items()):
            if root['pre'] != 0 or not root['peer_requested']:continue
            used = root['peer_logits'] if root_priors is None else root_priors[key]
            gap = reference.boundary_gap(root['peer_legal'], used, widths['k1_opp'])
            if gap is None:continue
            out['gaps']['root_peer'].append(gap)
            out['oracle_prior_set_differs']['root_peer'] += int(reference.top_k(root['peer_legal'], used, widths['k1_opp'])
                != reference.top_k(root['peer_legal'], root['peer_logits'], widths['k1_opp']))
    return out


def merge_topk_boundary(reports):
    reports = [r for r in reports if r is not None]
    if not reports:return None
    seats = ('own', 'peer', 'root_peer');gaps = {s: [g for r in reports for g in r['gaps'][s]] for s in seats}
    return dict(ranked_by=sorted({r['ranked_by'] for r in reports}), logit_tolerance=LOGIT_TOLERANCE,
        selections={s: len(gaps[s]) for s in seats}, gap={s: percentiles(gaps[s]) for s in seats},
        gap_below_logit_tolerance={s: int(np.count_nonzero(np.asarray(gaps[s], np.float64) < LOGIT_TOLERANCE)) for s in seats},
        oracle_prior_set_differs={s: sum(r['oracle_prior_set_differs'][s] for r in reports) for s in seats},
        definition='gap = rank-k minus rank-(k+1) logit of the prior that ranked the selection (the pricer\'s, checked by (c)); '
            'oracle_prior_set_differs = selections the oracle\'s own bf16 prior would have made differently (a near-tie the tolerance absorbs)')


def sample_leaves(leaves, count, addresses, *, block, forced=()):
    """All leaves (count None), or per world `forced` leaves plus count uniform others from a fixed per-world seed."""
    if count is None:return list(leaves)
    if type(count) is not int or count < 0:raise ValueError('non-negative leaf sample size required')
    forced = set(forced);worlds = collections.defaultdict(list)
    for leaf in leaves:worlds[leaf[:2]].append(leaf)
    chosen = []
    for (q, wi), items in worlds.items():
        a = addresses[q];rng = np.random.default_rng([SAMPLE_SEED, block, a.stream, a.frame, a.owner, wi])
        rest = [x for x in items if x not in forced];picked = set(rest[i] for i in rng.choice(len(rest), size=min(count, len(rest)), replace=False))
        chosen.extend(x for x in items if x in forced or x in picked)
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
    # Top-k is ranked by the PRICER's priors ((c) checks them against the oracle's within LOGIT_TOLERANCE) with the
    # oracle's own top_k, so a bf16 near-tie between two heads' geometries cannot fail (d); (d) still checks the ranking.
    usable = not frame_geometry(frame, len(requests))
    root_priors = {k: field(frame, 'root_peer_logits')[k] for k in roots} if usable and widths['k1_opp'] < 9 else None
    with stage(progress, 'oracle_children', **context) as info:
        cells = reference.turn1_cells(requests, recipe, roots, k1_own=widths['k1_own'], k1_opp=widths['k1_opp'], root_priors=root_priors)
        children = oracle.children(stack, cells, params);info['children'] = len(cells)
    priors = None
    if usable and len(cells) and (widths['k_own'] < 9 or widths['k_opp'] < 9):
        at = tuple(children['cells'].T);priors = dict(own=field(frame, 't1_own_logits')[at], peer=field(frame, 't1_peer_logits')[at])
    enumerated = reference.turn2_leaves(children, k_own=widths['k_own'], k_opp=widths['k_opp'], priors=priors) if len(cells) else []
    topk = topk_boundary(children, roots, widths, priors, root_priors, reference)
    # A sample always carries every terminal and error leaf the pricer reports (and the oracle enumerates).
    forced = set()
    if usable and sample is not None and enumerated:
        known = set(enumerated);t2 = field(frame, 't2_status')
        forced = {cell for cell in map(tuple, np.argwhere((t2 == 1) | (t2 == 3)).tolist()) if cell in known}
    checked = sample_leaves(enumerated, sample, [r.plans.root.address for r in requests], block=block, forced=forced)
    with stage(progress, 'oracle_leaves', **context) as info:
        leaves = oracle.leaves(stack, checked, params);info.update(enumerated=len(enumerated), checked=len(checked))
    timing['oracle_seconds'] = time.perf_counter() - clock
    costs['oracle'] = dict(wall_seconds=timing['oracle_seconds'], chunks=len(oracle.chunks) - chunks,
        seconds={k: oracle.seconds[k] - before[k] for k in oracle.seconds},
        chunk_device_seconds=sum(c['step_seconds'] + c['head_seconds'] for c in oracle.chunks[chunks:]))
    return frame, d_results, dict(roots=roots, children=children, enumerated=enumerated, leaves=leaves, worlds=len(stack.rows),
        invariance_frames=others, costs=costs, topk=topk, forced_leaves=len(forced)), timing


def run_call(*, requests, recipe, materializer, params, pricer, frame_pricer, oracle, widths, sample, block, context, invariance=None,
             progress=None, decision=None, save=None, evaluator=None):
    """Price ONE request list three ways and evaluate it; returns (evaluation, call summary).

    decision: run check (e) (default: when every leaf is checked); save: an .npz path for --save-arrays. The
    evaluation also carries the call's values samples and sibling spread, from the pricer frame."""
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
    roots = data['roots'];evaluation['topk'] = data['topk']
    with stage(progress, 'refusals', **context):
        refusals = None if evaluation['geometry'] else refusal_report(frame, d_results, data, root_legal=[r.root_legal for r in requests])
    if (sample is None if decision is None else decision) and not evaluation['geometry']:
        with stage(progress, 'check_e_decision', **context) as info:
            evaluation['e'] = decision_check(frame, data, requests, context=context)
            info.update({k: evaluation['e'].get(k) for k in ('passed', 'roots_compared', 'argmax_q_differs', 'argmax_target_differs', 'cell_choice_differs')})
    if not evaluation['geometry']:
        with stage(progress, 'values', **context):
            evaluation['values'] = value_samples(frame, requests, context=context);evaluation['sibling'] = sibling_spread(frame, context=context)
    if save is not None:
        with stage(progress, 'save_arrays', **context) as info:
            info['arrays'] = len(save_arrays(save, frame, requests, data=data, d_results=d_results, invariance_frames=data['invariance_frames'],
                meta=dict(context, widths=widths, sample_leaves=sample, invariance=bool(invariance), evaluator=evaluator, topk=data['topk'])))
            info['path'] = str(save)
    summary = dict(context, roots=len(requests), worlds=data['worlds'], usable_worlds=sum(r['pre'] == 0 for r in roots.values()), refusals=refusals,
        turn1_cells=len(data['children']['cells']), turn2_leaves=len(data['enumerated']), checked_leaves=len(data['leaves']['specs']),
        forced_leaves=data['forced_leaves'], timing=timing,
        costs=data['costs'], pricer_timing=json.loads(json.dumps(pricer_timing, default=str)) if pricer_timing is not None else None,
        passed=dict(a=evaluation['a']['passed'], b=evaluation['b']['passed'], d=evaluation['d']['passed'], oracle=evaluation['oracle']['passed'],
            **({'block_invariance': all(r['passed'] for r in evaluation['invariance'])} if invariance else {}),
            **({'e': evaluation['e']['passed']} if 'e' in evaluation else {})), saved_arrays=None if save is None else str(save))
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


def load_head(args, *, plan, labels, category_sha256):
    """--head-archive: U1's load and arm check. The bank declaration's category vocabulary and the plan's labels bind the
    archive; its geometry must be the bank's D model up to arm/replicate. Returns (params, head_config, metadata, model_sha256)."""
    from pokezero import v5_run as runs
    from gpu_engine import v5_model
    if args.head_category_sha256 is not None and args.head_category_sha256 != category_sha256:
        raise ValueError(f'head category vocabulary {args.head_category_sha256} differs from the bank declaration\'s {category_sha256}')
    params, head_config, meta = v5_model.load(args.head_archive, expected_archive_sha256=args.head_archive_sha256,
        expected_checkpoint_sha256=args.head_checkpoint_sha256, expected_category_vocabulary_sha256=category_sha256,
        expected_label_vocabulary_sha256=labels.sha256, enable_v5=True)
    if replace(head_config, arm='D', replicate=0) != plan.model:raise ValueError('head geometry differs from the bank D model')
    model_sha256 = runs._sha(head_config.metadata())   # what Snapshot.model_sha256 records for a model
    if args.head_model_sha256 is not None and model_sha256 != args.head_model_sha256:
        raise ValueError(f'head model sha256 {model_sha256} differs from --head-model-sha256 {args.head_model_sha256}')
    return params, head_config, meta, model_sha256


def bind_roots(groups, params, plan, *, head_config=None):
    """Each request's learner value head on its own root inputs (D arm, bf16, batch 1).

    With a head override (head_config, params = (head,)): the arm check on the first request, where the D-arm and head-arm
    forwards must give exactly equal value and policy logits, as U1. Each request then takes the head's root logits (U1's
    root prior) and actors 0/0, and a root whose head output is invalid or non-finite is dropped (U1 refuses it).
    Returns (groups, root |v| list, dropped addresses, arm_checked)."""
    import jax
    from gpu_engine import v5_model
    forward = jax.jit(lambda p, x: v5_model.forward(p, x, plan.model, compute_dtype='bfloat16'))
    head = None if head_config is None else jax.jit(lambda p, x: v5_model.forward(p, x, head_config, compute_dtype='bfloat16'))
    out, values, dropped, checked = [], [], [], False
    for group in groups:
        kept = []
        for request in group:
            p = params[0] if head is not None else params[request.learner_actor]
            inputs = {k: np.asarray(v) for k, v in request.root_inputs.items()}
            output = jax.device_get(forward(p, inputs))
            if head is not None and not checked:
                other = jax.device_get(head(p, inputs))
                if not (np.array_equal(np.asarray(other.value), np.asarray(output.value))
                        and np.array_equal(np.asarray(other.policy_logits), np.asarray(output.policy_logits))):
                    raise ValueError('D-arm and head-arm forwards differ')
                checked = True
            value, logits = float(np.asarray(output.value).reshape(-1)[0]), np.asarray(output.policy_logits, np.float32).reshape(-1)
            usable = bool(np.asarray(output.valid).reshape(-1)[0]) and np.isfinite(logits).all() and np.isfinite(value) and abs(value) <= 1
            if usable:values.append(abs(value))
            if head is not None:
                if not usable:
                    a = request.plans.root.address;dropped.append([a.stream, a.frame, a.owner]);continue
                request = replace(request, root_logits=tuple(map(float, logits)), learner_actor=0, opponent_actor=0)
            kept.append(request)
        if kept:out.append(kept)
    return out, values, dropped, checked


def evaluator_header(args, loaded, head):
    """Who priced: the override head, or the bank snapshot's actor (iteration -1 is the bank's untrained actor)."""
    plan = loaded.plan
    if head is not None:
        _, config, meta, model_sha256 = head
        return dict(override=True, archive=args.head_archive, archive_sha256=args.head_archive_sha256,
            checkpoint=meta.get('checkpoint_path') or meta.get('checkpoint'), checkpoint_sha256=args.head_checkpoint_sha256, iteration=None,
            model_sha256=model_sha256, head_arm=config.arm, pricer_arm=plan.model.arm, parameter_count=meta.get('parameter_count'),
            category_vocabulary_sha256=loaded.category_sha256, label_vocabulary_sha256=loaded.labels.sha256)
    s = loaded.snapshot
    header = dict(override=False, archive=s.get('archive'), archive_sha256=s.get('archive_sha256'), checkpoint=s.get('checkpoint'),
        checkpoint_sha256=s.get('checkpoint_sha256'), iteration=s.get('iteration'), model_sha256=s.get('model_sha256'), head_arm=plan.model.arm,
        pricer_arm=plan.model.arm, category_vocabulary_sha256=loaded.category_sha256, label_vocabulary_sha256=loaded.labels.sha256)
    if s.get('iteration') == -1:header['warning'] = 'iteration -1 is the bank\'s own UNTRAINED actor: values are not a trained head\'s'
    return header


def value_scale(root_values, t1_totals, reference):
    pick = lambda d: {k: d.get(k) for k in ('count', 'p50', 'p90')}
    return dict(root_abs_value=pick(percentiles(root_values)), t1_abs_value_expanded=pick(t1_totals or dict(count=0)), reference=reference,
        root_scope='the learner value head on each request\'s own root inputs (D arm, bf16)', t1_scope='|t1_value| at expanded cells of the priced frames')


def add_head_arguments(parser):
    parser.add_argument('--head-archive', help='price with this v5 actor archive instead of the bank snapshot\'s actor (U1\'s head override)')
    parser.add_argument('--head-archive-sha256', help='sha256 of --head-archive (required with it)')
    parser.add_argument('--head-checkpoint-sha256', help='the archive\'s checkpoint sha256 (required with --head-archive)')
    parser.add_argument('--head-category-sha256', help='optional: must equal the bank declaration\'s category vocabulary sha256')
    parser.add_argument('--head-model-sha256', help='optional: must equal sha of the head config metadata (Snapshot.model_sha256)')
    parser.add_argument('--value-reference', default=None, help='free text echoed in value_scale.reference (e.g. a training value-estimate scale)')


def check_head_arguments(parser, args):
    given = [args.head_archive, args.head_archive_sha256, args.head_checkpoint_sha256]
    if any(given) and not all(given):parser.error('--head-archive, --head-archive-sha256 and --head-checkpoint-sha256 go together')
    if (args.head_category_sha256 or args.head_model_sha256) and not args.head_archive:parser.error('--head-category/model-sha256 need --head-archive')


def call_capacity(frame_pricer, pricers):
    """One call must fit every pricer it goes to: the smallest declared root capacity (FramePricer's and each Depth2Pricer's)."""
    return min([frame_pricer.capacity] + [c for c in (getattr(p, 'capacity', None) for p in pricers) if isinstance(c, int) and c > 0])


def split_groups(groups, capacity):
    return [g[i:i + capacity] for g in groups for i in range(0, len(g), capacity) if g[i:i + capacity]]


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


def invariance_tolerance(name):
    """Float frame fields: logits LOGIT_TOLERANCE, values VALUE_TOLERANCE, any other float exact."""
    return LOGIT_TOLERANCE if 'logits' in name else VALUE_TOLERANCE if name.endswith('value') else 0.


def block_invariance(a, b, *, context=None):
    """Two frames of the same requests (timing excluded): discrete arrays bit-equal (dtype, shape, bytes),
    float arrays within invariance_tolerance (bit-equal NaNs agree; any other non-finite mismatch fails).

    Every float array also reports its exact fraction and max |diff|. Non-array fields (world_reports) are
    compared as canonical JSON and reported, not gated."""
    fa, fb = frame_fields(a), frame_fields(b);arrays, other, differing, first, floats = 0, {}, {}, [], {}
    for name in sorted((set(fa) | set(fb)) - {'timing'}):
        if name not in fa or name not in fb:differing[name] = dict(problem='missing in ' + ('first' if name not in fa else 'second'));continue
        x, y = np.asarray(fa[name]), np.asarray(fb[name])
        if x.dtype.kind not in 'biufc' and y.dtype.kind not in 'biufc':
            canonical = lambda v: json.dumps(v, sort_keys=True, default=str)
            other[name] = canonical(fa[name]) == canonical(fb[name]);continue
        arrays += 1
        if x.dtype != y.dtype or x.shape != y.shape:
            differing[name] = dict(problem=f'dtype/shape {x.dtype}{x.shape} != {y.dtype}{y.shape}');continue
        if x.dtype.kind in 'fc':
            summary, beyond = _numeric(None, x.reshape(-1), y.reshape(-1), invariance_tolerance(name), context, shape=x.shape)
            floats[name] = summary
            if summary['beyond_tolerance']:
                differing[name] = dict(count=summary['beyond_tolerance'], size=int(x.size), tolerance=summary['tolerance'])
                first.extend(dict(row, field=name, first=row['pricer'], second=row['reference']) for row in summary['first_beyond'][:max(0, FIRST - len(first))])
            continue
        size = x.dtype.itemsize;flat = lambda v: np.ascontiguousarray(v).reshape(-1).view(np.uint8).reshape(-1, size)
        bad = np.flatnonzero(np.any(flat(x) != flat(y), axis=1)) if x.size else np.zeros(0, int)
        if len(bad):
            differing[name] = dict(count=int(len(bad)), size=int(x.size))
            for i in bad[:max(0, FIRST - len(first))]:
                cell = np.unravel_index(i, x.shape)
                first.append(dict(context or {}, field=name, cell=[int(c) for c in cell], first=_plain(x[cell]), second=_plain(y[cell])))
    return dict(passed=not differing, arrays_compared=arrays, differing=differing, first_differences=first, nonarray_equal=other, floats=floats)


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
        first_differences=first[:FIRST], nonarray_equal=other, floats=merge_numeric([r.get('floats', {}) for r in reports]),
        rule='discrete arrays bit-equal; float arrays within tolerance (logits LOGIT_TOLERANCE, *value VALUE_TOLERANCE, other floats exact)')


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
    return SimpleNamespace(provider=provider, plan=plan, params=actors[0], groups=[requests], frame_pricer=frame_pricer,
        source=dict(saved_roots=str(root), index_sha256=args.index_sha256), labels=labels, category_sha256=declaration['category_vocabulary_sha256'],
        snapshot=dict(iteration=snapshot.iteration, checkpoint=snapshot.checkpoint, checkpoint_sha256=snapshot.checkpoint_sha256,
            archive=snapshot.archive, archive_sha256=snapshot.archive_sha256, model_sha256=snapshot.model_sha256))


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
    return SimpleNamespace(provider=provider, plan=plan, params=(params,), groups=groups, frame_pricer=provider.frame_pricer,
        source=dict(d3_root=str(root), chunk=args.chunk, declaration_sha256=declared), labels=plan.model.label_vocabulary,
        category_sha256=plan.model.category_vocabulary_sha256,
        snapshot={k: snapshot.get(k) for k in ('iteration', 'checkpoint', 'checkpoint_sha256', 'archive', 'archive_sha256', 'model_sha256')})


DIGEST_NOTE = ('--pricer-only keeps digest=True: Depth2Pricer passes digest into make_depth2_kernel, which adds the leaf digests '
    'as kernel outputs, so digest=False would compile a different program and miss the gate\'s warm compile cache. The digests are not used.')


def pricer_only(args, *, provider, params, groups, pricer, options, seeds, out, source, widths, evaluator=None, root_values=()):
    """--pricer-only: the gate's normal Depth2Pricer call on every request group, nothing else; values and refusals from its frame."""
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds
    log('PRICER_ONLY_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), widths=widths, pricer_options=options)
    collected, sibling, refusals, calls = [], [], [], []
    for block, seed in enumerate(seeds):
        recipe = provider.recipe if block == 0 else original.Recipe(seed)
        with patch.object(original.Recipe, 'selected', lambda self, address: True) if block else nullcontext():
            for call, group in enumerate(groups):
                context = dict(block=block, call=call)
                requests = tuple(group if block == 0 else [replace(r, plans=worlds.plan_starts(r.plans.root, provider.source, recipe)) for r in group])
                with stage(log, 'depth2_pricer', roots=len(requests), **context) as info:
                    clock = time.perf_counter();frame = pricer.price(requests, provider.materializer, recipe, params)
                    costs = dict(wall_seconds=time.perf_counter() - clock, **pricer_costs(_frame_timing(frame)));info.update(costs)
                geometry = frame_geometry(frame, len(requests))
                summary = dict(context, roots=len(requests), costs=dict(depth2_pricer=costs), geometry=geometry)
                if not geometry:
                    with stage(log, 'values', **context):
                        collected.append((value_samples(frame, requests, context=context), None));sibling.append(sibling_spread(frame, context=context))
                        refusals.append(refusal_report(frame, root_legal=[r.root_legal for r in requests]));summary['refusals'] = refusals[-1]
                if args.save_arrays:
                    path = out / f'arrays-block{block}-call{call}.npz'
                    with stage(log, 'save_arrays', path=str(path), **context):
                        save_arrays(path, frame, requests, meta=dict(context, widths=widths, pricer_only=True, evaluator=evaluator));summary['saved_arrays'] = str(path)
                calls.append(summary);log('PRICER_CALL', **summary)
    rows = [c['costs']['depth2_pricer'] for c in calls]
    total = lambda key: None if all(r.get(key) is None for r in rows) else sum(r.get(key) or 0. for r in rows)
    values = values_section(collected, oracle_checked=False)
    report = dict(evaluator=evaluator, value_scale=value_scale(root_values, values['totals'].get('abs_t1_value'), getattr(args, 'value_reference', None)),
        schema=SCHEMA, status='PRICER_ONLY', passed=False, mode='PRICER_ONLY', source=source, widths=widths, worlds_blocks=args.worlds_blocks,
        block_seeds=list(seeds), max_roots=args.max_roots, pricer_options=options, digest_note=DIGEST_NOTE, calls=calls,
        geometry_failures=[c for c in calls if c['geometry']],
        values=values, sibling_spread=merge_sibling_spread(sibling, oracle_checked=False),
        refusals=merge_refusals(refusals), costs=dict(depth2_pricer={k: total(k) for k in ('wall_seconds', 'lower_and_compile_seconds', 'device_seconds')}),
        scope='Depth2Pricer only: no FramePricer, oracle or checks; every value is the pricer\'s own, not oracle-checked')
    (out / 'gate.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')   # evaluator first
    log('PRICER_ONLY_COMPLETE', out=str(out / 'gate.json'), near_ties=report['values']['near_ties']['share'], refused=report['refusals'].get('depth2'))
    return 0


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
    parser.add_argument('--save-arrays', action='store_true',
        help='write each call\'s frame, other child-block frame, oracle outputs and FramePricer outputs to --out as .npz (for --recheck)')
    parser.add_argument('--recheck', metavar='DIR', help='offline: re-evaluate every check from DIR\'s saved arrays (no GPU, no engine) into --out/recheck.json')
    parser.add_argument('--pruning-report', metavar='DIR', help='offline: learner-max a2 rank vs top-k from DIR\'s full-width saved arrays into --out/pruning.json')
    parser.add_argument('--pricer-only', action='store_true', help='run only Depth2Pricer (no FramePricer, oracle or checks): values, refusals, arrays')
    parser.add_argument('--topk-errors', nargs=2, metavar=('FULL_DIR', 'TOPK_DIR'),
        help='offline: share of roots whose only full-width errors sit in cells top-k prunes away, into --out/topk-errors.json')
    parser.add_argument('--raw-head-check', action='store_true',
        help='with --head-archive: gate the pricer\'s leaf and child values against the head\'s own arm on the oracle\'s inputs (in (c))')
    add_head_arguments(parser)
    args = parser.parse_args(argv)
    check_head_arguments(parser, args)
    if args.raw_head_check and not args.head_archive:parser.error('--raw-head-check needs --head-archive')
    if args.raw_head_check and args.pricer_only:parser.error('--raw-head-check needs the oracle: drop --pricer-only')
    offline = bool(args.recheck or args.pruning_report or args.topk_errors)
    if offline:
        if args.saved_roots or args.d3_root or args.pricer_only or args.head_archive:
            parser.error('--recheck/--pruning-report/--topk-errors run offline: no --saved-roots/--d3-root/--pricer-only/--head-archive')
        out = Path(args.out);out.mkdir(parents=True, exist_ok=False);status = 0
        if args.recheck:
            report = recheck(args.recheck, progress=log)
            if report['evaluator'] is None:report['evaluator'] = dict(unknown='arrays saved before the evaluator header (5ea0552b): the bank snapshot\'s actor')
            (out / 'recheck.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')
            log('RECHECK_COMPLETE', status=report['status'], out=str(out / 'recheck.json'), checks={k: v['passed'] for k, v in report['checks'].items()},
                unavailable=report['unavailable']);status = 0 if report['passed'] else 1
        if args.topk_errors:
            report = topk_error_share(*args.topk_errors)
            (out / 'topk-errors.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')
            log('TOPK_ERRORS_COMPLETE', out=str(out / 'topk-errors.json'), counts=report['counts'], share=report['share_of_roots_with_full_width_errors'])
        if args.pruning_report:
            report = offline_pruning(args.pruning_report)
            (out / 'pruning.json').write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')
            log('PRUNING_REPORT_COMPLETE', out=str(out / 'pruning.json'), full_width=report['full_width'], per_cell=report['per_cell'])
            status = status or (0 if report['full_width'] else 1)
        return status
    if bool(args.saved_roots) == bool(args.d3_root):parser.error('exactly one of --saved-roots or --d3-root required')
    if args.saved_roots and not args.index_sha256:parser.error('--saved-roots requires --index-sha256')
    if args.d3_root and (args.chunk is None or args.chunk < 0):parser.error('--d3-root requires --chunk N >= 0')
    widths = dict(k_own=args.k_own, k_opp=args.k_opp, k1_own=args.k1_own, k1_opp=args.k1_opp)
    if any(not 1 <= v <= 9 for v in widths.values()):parser.error('widths must be 1..9')
    if args.worlds_blocks < 1 or (args.max_roots is not None and args.max_roots < 1):parser.error('positive --worlds-blocks/--max-roots required')
    if args.oracle_chunk < 1 or (args.sample_leaves is not None and args.sample_leaves < 1):
        parser.error('positive --oracle-chunk and --sample-leaves N >= 1 (no --sample-leaves checks every leaf)')
    if args.child_block is not None and args.child_block < 1:parser.error('positive --child-block required')
    if args.block_invariance and args.child_block is not None:parser.error('--block-invariance fixes child_block to 27 and 81; drop --child-block')
    if args.pricer_only and args.block_invariance:parser.error('--pricer-only runs one pricer: drop --block-invariance')
    import importlib.util   # fail before loading roots; find_spec does not import JAX
    if importlib.util.find_spec('gpu_engine.device_depth2_pricer') is None:raise SystemExit('gpu_engine.device_depth2_pricer is not on this tree')
    if args.sample_leaves is None and not args.pricer_only and importlib.util.find_spec('gpu_engine.depth2_backups') is None:
        raise SystemExit('check (e) needs gpu_engine.depth2_backups (learner_max_root), which is not on this tree: failing closed')
    log('EVALUATOR', override=bool(args.head_archive), archive=args.head_archive, archive_sha256=args.head_archive_sha256,
        checkpoint_sha256=args.head_checkpoint_sha256, value_reference=args.value_reference,
        resolved='EVALUATOR_RESOLVED follows once the bank is loaded' + ('' if args.head_archive else ' (the bank snapshot\'s actor)'))
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    loaded = (saved_root_requests if args.saved_roots else bank_chunk_requests)(args)
    provider, plan, params, frame_pricer, source = loaded.provider, loaded.plan, loaded.params, loaded.frame_pricer, loaded.source
    head = load_head(args, plan=plan, labels=loaded.labels, category_sha256=loaded.category_sha256) if args.head_archive else None
    if head is not None:params = (head[0],)
    groups, root_values, dropped, arm_checked = bind_roots(loaded.groups, params, plan, head_config=None if head is None else head[1])
    evaluator = dict(evaluator_header(args, loaded, head), arm_checked=arm_checked, dropped_roots=dropped)
    log('EVALUATOR_RESOLVED', evaluator=evaluator, root_value_scale=percentiles(root_values))
    actor_capacity = len(params) if head is not None else plan.historical_window + 1
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds, depth2_oracle as reference
    if args.max_roots is not None:
        kept, left = [], args.max_roots
        for group in groups:
            if left > 0:kept.append(group[:left]);left -= len(kept[-1])
        groups = kept
    blocks = INVARIANCE_BLOCKS if args.block_invariance else (args.child_block,)
    pricers = [make_pricer(provider.executor, plan.model, widths=widths, actor_capacity=actor_capacity, child_block=b) for b in blocks]
    (pricer, options), invariance = pricers[-1], [p for p, _ in pricers[:-1]]
    capacity = call_capacity(frame_pricer, [p for p, _ in pricers]);groups = split_groups(groups, capacity)
    seeds = world_block_seeds(provider.recipe.seed, args.worlds_blocks)
    if args.pricer_only:
        return pricer_only(args, provider=provider, params=params, groups=groups, pricer=pricer, options=options, seeds=seeds, out=out, source=source,
            widths=widths, evaluator=evaluator, root_values=root_values)
    oracle = reference.Oracle(provider.executor, plan.model, chunk=args.oracle_chunk, progress=oracle_chunk_progress(log),
        raw_model=head[1] if args.raw_head_check else None)
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
                    context=dict(block=block, call=call), invariance=invariance, progress=log,
                    save=out / f'arrays-block{block}-call{call}.npz' if args.save_arrays else None, evaluator=evaluator)
                evaluations.append(evaluation);calls.append(summary);log('CALL_CHECKED', **summary)
    result = summarize(evaluations)
    reported = [c for c in calls if c['refusals'] is not None]
    refusals = dict(merge_refusals([c['refusals'] for c in reported]), calls_reported=len(reported), calls=len(calls),
        by_block={str(b): merge_refusals([c['refusals'] for c in reported if c['block'] == b]) for b in range(len(seeds))},
        rule='reference (D): any world pre != 0 or any status-3 cell at either turn refuses the root; codes 5/6 refuse too')
    leaves = oracle.rows['leaves'];children = oracle.rows['children']
    values = values_section([(e['values'], e.get('e')) for e in evaluations if 'values' in e], oracle_checked=args.sample_leaves is None)
    status, reasons = gate_status(result, sampled=args.sample_leaves is not None)
    report = dict(evaluator=evaluator, value_scale=value_scale(root_values, values['totals'].get('abs_t1_value'), args.value_reference),
        schema=SCHEMA, status=status, passed=status in ('PASS', 'PASS_SAMPLED'), status_reasons=reasons, checks=result['checks'],
        raw_head_check=args.raw_head_check, split_capacity=capacity,
        decision=result['checks'].get('e', dict(status='SKIPPED_SAMPLED', reason='check (e) needs every leaf checked (no --sample-leaves)')),
        topk_boundary=merge_topk_boundary([e.get('topk') for e in evaluations]),
        mode='SAVED_ROOTS' if args.saved_roots else 'BANK_CHUNK', source=source, widths=widths, worlds_blocks=args.worlds_blocks,
        block_seeds=list(seeds), max_roots=args.max_roots, sample_leaves=args.sample_leaves, sample_seed=SAMPLE_SEED,
        value_tolerance=VALUE_TOLERANCE, logit_tolerance=LOGIT_TOLERANCE, tolerance_source=TOLERANCE_SOURCE, pricer_options=options, block_invariance=args.block_invariance,
        invariance_pricer_options=[o for _, o in pricers[:-1]], calls=calls, conventions=list(CONVENTIONS), refusals=refusals,
        values=values, sibling_spread=merge_sibling_spread([e['sibling'] for e in evaluations if 'sibling' in e], oracle_checked=args.sample_leaves is None),
        costs=merge_costs(calls),
        oracle_cost=dict(seconds=oracle.seconds, rows=oracle.rows, chunk=args.oracle_chunk, chunks=oracle.chunks,
            first_chunk_seconds={kind: next((c['seconds'] for c in oracle.chunks if c['kind'] == kind), None) for kind in ('roots', 'children', 'leaves')},
            chunk_scope='per chunk: step_seconds = the vmapped executor program to block_until_ready, head_seconds = the bf16 heads; '
                'the first chunk of a program and world bucket includes its compile',
            seconds_per_leaf=None if not leaves else oracle.seconds['leaves'] / leaves,
            seconds_per_child=None if not children else oracle.seconds['children'] / children))
    (out / 'gate.json').write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')   # evaluator first
    log('DEPTH2_GATE_COMPLETE', status=report['status'], out=str(out / 'gate.json'), status_reasons=reasons,
        checks={k: v['passed'] for k, v in result['checks'].items()},
        refused_share=dict(depth2=refusals.get('depth2', {}).get('refused_share'), depth1=refusals.get('depth1', {}).get('refused_share')))
    return 0 if report['passed'] else 1


if __name__ == '__main__':raise SystemExit(main())
