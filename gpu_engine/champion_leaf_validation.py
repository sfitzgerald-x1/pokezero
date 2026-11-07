"""Gap 2: champion 9375 validated as the search pricers' evaluator on real D3 roots (bf16 vs f32, the adapter against an independent raw
forward, sign and perspective, value scale). Numpy-only import; JAX loads after the bank's memory policy, as the gates.

Setup is the depth-N gate's, reused, not reimplemented: --saved-roots DIR --index-sha256 SHA (the 40 saved roots) or --d3-root DIR --chunk N
(a bank chunk) through d3_depth2_gate.saved_root_requests / bank_chunk_requests (the bank's provider: materializer, executor, recipe), the
champion through g2.load_champion (archive and checkpoint sha256, the real 9375, the bank catalog compiled for it), the root prior through
g2.bind_evaluator_roots (its exact first-root check included), DepthNPricer (the gate's depth2_pass_pricer and prior DepthNPricers) and the
world stack of depth2_oracle. Two ChampionEvaluators share the archive, f32 (HIGHEST; the reference) and bf16 (the ladder's and the tools'
default; the one under test), each bound to the roots on its own and pricing with its own DepthNPricer instances and program tables: depth 2
at full width (k = 9) and, unless --depth3-k 0, depth 3 prior top-k. Roots kept under both evaluators are compared.

Checks, each with an explicit verdict against THRESHOLDS (declared below, logged before any loading, copied into the report). The two
evaluators must be one archive at two precisions (check_pair: same type, parameters, config and archive; f32 and bf16), and the bf16 one
under test is gated on its own: its slice, placement, owner's view and gross divergence from f32 FAIL like the f32 path's.
1. precision (WARN; FAIL on gross divergence): the bf16 frames against the f32 frames, joined by action path (a bf16 top-k selection that
   differs at depth 3 compares only what both priced): max, p99, p99.9, p99.99 |diff| of root logits and values, turn-1 values, ply-2
   node values and leaf values, both seats' priors; argmax changes per node; status and terminal differences; sign disagreements on
   confident f32 values; the root decision (completed-Q argmax of depthn_backups.learner_max_root, d3_depthn_gate.root_decisions)
   agreement per frame family.
2. adapter (gated): a second input-construction path that never touches the 29 -> 23 slice. Every root (one row per root: its first world
   the pricers price; both worlds' views equal the root inputs) and a seeded sample of turn-1 cells, ply-2 nodes and leaves (--sample per
   level and call) is re-walked from its materialized root world with the depth-N oracle's step (depthn_oracle._step: the node's own
   chance, D's turn-1 joint rule) and observed as Codex's ladder observes a leaf (phase_d_search.Search.observe: device_public_leaf.
   from_public(..., legacy_champion=True), engine_device_observation.champion_inputs), owner and opponent alike; champion.forward runs on
   that layout directly, in f32, under its own jit. Against it: the layout itself vs EACH evaluator's view of the executor's V2 view
   (bit-exact), the walk's placement (its digest vs each frame's; terminal and error facts), each pricer's value and both seats' priors at
   that position, and each adapter on the same rows in the same batch. The observed maxima are reported beside the bounds.
3. perspective (gated): the PRICERS' values (what the search consumes) against the V2 B1-i99 head (--head-archive, required; its own arm,
   bf16) on the same learner views: Pearson, Spearman, same-sign rates and the regression slope, pooled, per level and per learner seat;
   the champion's own observer-relativity (owner's view vs opponent's view) and that each pricer value is the owner's view's; and an
   outcome-linked check: parents one step before a terminal whose decisive terminal children (+1/-1 from the learner's side, as the pricer
   assigns them) agree in sign, pooled and per learner seat. No realized game outcome is available per root: the bank's chunk Frames
   record none, the collector carry's winner exists only once a game completes (past the chunk the gate's loader stops at), and the
   saved-root Requests carry none; so the terminals are the search's own.
4. scale (WARN): |v| of the champion and of the V2 head at roots, turn-1 cells and leaves (p50, p90, p99, share above 0.95), and the
   champion's over every valued frame position (a uniform sample).
Out of scope: device-encoder vs B2 parity (B2 folds a Showdown protocol on the host; D3 roots have none). JSON (validation.json) and a
markdown summary (validation.md) in --out. Exit codes (EXIT): 0 PASS or WARN, 1 FAIL, 2 INCONCLUSIVE (the sign or the adapter was not
established: no verdict from either the V2 check or the near-terminal check, or no f32-valued position compared, or no root at all).
"""
import argparse
from dataclasses import replace
import importlib
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import champion_evaluator as ce, d3_depth2_gate as g2, d3_depthn_gate as gate

SCHEMA = 'pokezero.champion-leaf-validation.v2'
REFERENCE, TESTED = 'f32', 'bf16'          # the evaluators by name: f32 is the reference, bf16 the one under test
NAMES = (REFERENCE, TESTED)
PRECISION_OF = dict(f32='float32', bf16='bfloat16')
SEED = 20261107
SAMPLE = 4096                              # walked rows per sampled level and call (every root is walked)
CHUNK = 512                                # walk rows per jitted vmap invocation
FRAME_SAMPLE = 1 << 18                     # per call and level: the uniform sample of frame values the scale section reads
TAILS = (99, 99.9, 99.99)
RAW_KEYS = ('numeric', 'categorical', 'token_types', 'attention')   # engine_device_observation.champion_inputs, champion.forward's order
UNIFORM = ('root', 'depth2.turn1', 'depth2.leaf', 'depth3.ply2', 'depth3.leaf')   # depth3.turn1 re-reads depth2.turn1's rows
ORDER = dict(PASS=0, SKIPPED=1, INCONCLUSIVE=2, WARN=3, FAIL=4)
EXIT = dict(PASS=0, WARN=0, FAIL=1, INCONCLUSIVE=2)
log, stage = g2.log, g2.stage

# Thresholds: declared before any run (logged first, copied into the report). -------------------------------------------------------
ADAPTER_VALUE = 1e-5          # f32 pricer (or adapter) value vs the raw f32 forward, every compared position
ADAPTER_LOGIT = 1e-4          # f32 priors (both seats, legal lanes) vs the raw f32 forward
PRECISION_VALUE_WARN = .05    # bf16 vs f32 (and bf16 vs the raw f32 forward) max |value diff| at any compared position
PRECISION_DECISION_WARN = .95  # bf16 vs f32 completed-Q argmax agreement over roots both label, per frame family
BF16_GROSS_VALUE = .25        # bf16 FAIL: max |value diff| vs f32 (frames) or vs the raw f32 forward above this
BF16_SIGN_SHARE = .01         # bf16 FAIL: sign disagreements with f32 on confident f32 values (|v| >= CONFIDENT) above this share
PEARSON_PASS, SIGN_PASS = .5, .8
PEARSON_FAIL, SIGN_FAIL = 0., .5
CONFIDENT = .2                # same-sign is read where both critics have |v| >= this (when there are MIN_ROWS such rows)
Z99 = 2.3263478740408408      # one-sided 99% normal quantile: subgroup FAILs need the 99% upper bound past the FAIL point
SEAT_PASS, SEAT_FAIL = -.5, 0.  # Pearson(champion on the owner's view, champion on the opponent's view)
SEAT_MARGIN = dict(f32=1e-3, bf16=2 * BF16_GROSS_VALUE)   # rows whose two views differ by more than this discriminate owner from opponent
NEAR_TERMINAL_PASS, NEAR_TERMINAL_FAIL = .9, .5
MIN_ROWS = 30                 # a group judged on fewer rows is reported, not judged
SCALE_RATIO = (.5, 2.)        # champion p50 |v| / V2 p50 |v|
SATURATED, SATURATION_WARN = .95, .25
SUBGROUP_RULE = ('a subgroup (a level, a learner seat, a level x seat, a source) with >= 30 rows FAILs only when its one-sided 99% upper bound '
    'is at or past the FAIL point: the Fisher-z bound tanh(atanh(r) + 2.326 / sqrt(n - 3)) <= the Pearson FAIL point, or the Wilson score '
    'bound on a share <= its FAIL point; the pooled group keeps the point thresholds. Subgroups are small (a seat of 40 roots is 20 rows), '
    'so point thresholds there FAIL a correct critic by chance (the reviewer simulated 15-19% at true r = 0.5 on 40 roots), while an '
    'inversion (r near -1, same-sign near 0) clears the bound at 20 rows')
THRESHOLDS = dict(
    adapter=dict(value=ADAPTER_VALUE, logit=ADAPTER_LOGIT, bf16_gross=BF16_GROSS_VALUE, bf16_warn=PRECISION_VALUE_WARN, rule='FAIL unless, on every '
        'walked row a frame values: the champion\'s own layout (the ladder\'s legacy_champion leaf) equals EACH evaluator\'s view of the '
        'executor\'s V2 view bit for bit; the walk lands on each pricer\'s position (digest equal, no terminal/invalid/error the frame did not '
        'see); the f32 pricer\'s value is within 1e-5 and its priors (both seats, legal lanes) within 1e-4 of champion.forward (f32, HIGHEST) '
        'on the independent layout, the same bound for the f32 adapter on the same rows; and the bf16 pricer and adapter are within 0.25 of '
        'it (WARN above 0.05). 1e-5 is the suggested value bound: tanh bounds the value and two f32 HIGHEST programs differ only in summation '
        'order (about 1e-6); different GPU batch shapes may approach it, so the observed max is reported beside it. Logits are pre-activation '
        'scores of magnitude about 10, so the same relative error is about 1e-4. A priced level with no valued position sampled WARNs; '
        'INCONCLUSIVE (exit 2) with no compared f32 row.'),
    precision=dict(value_warn=PRECISION_VALUE_WARN, decision_warn=PRECISION_DECISION_WARN, gross_value=BF16_GROSS_VALUE, sign_share=BF16_SIGN_SHARE,
        confident=CONFIDENT, rule='WARN when the bf16 pricer\'s max |value - f32 value| at any compared position exceeds 0.05 (the suggested '
        'bar: about 3x the 0.0173 A3 measured, and above typical top-two completed-Q gaps, so beyond it decisions move), or when the bf16 '
        'and f32 completed-Q argmax agree on under 95% of the roots both label in a frame family (A3\'s 99.78% is a per-row policy rate; '
        'under 95% the bf16 search is a different decision-maker). FAIL (gross divergence: a bf16-only slice, sign or perspective error, '
        'never rounding) when the max |diff| exceeds 0.25, or when bf16 has the other sign on more than 1% of the f32 values with |v| >= '
        '0.2 at any level (>= 30 such values) or pooled. bf16 operands with f32 accumulation move a value by about 1e-2 (A3: 0.0173 max), so '
        '0.25 is >10x rounding yet below the |v| a mirrored or mis-sliced value moves by (typically 0.3-1); at |v| >= 0.2 rounding cannot '
        'flip a sign, so a correct bf16 flips none, and 1% tolerates a handful of pathological rows where a systematic error flips most. '
        'INCONCLUSIVE when the two frames share no position. PASS otherwise.'),
    perspective=dict(pearson_pass=PEARSON_PASS, sign_pass=SIGN_PASS, pearson_fail=PEARSON_FAIL, sign_fail=SIGN_FAIL, confident=CONFIDENT, z99=Z99,
        seat_pass=SEAT_PASS, seat_fail=SEAT_FAIL, seat_margin=SEAT_MARGIN, near_terminal_pass=NEAR_TERMINAL_PASS, near_terminal_fail=NEAR_TERMINAL_FAIL,
        subgroup_rule=SUBGROUP_RULE,
        v2_rule='the pricers\' values (each precision, each pricer\'s rows kept) vs the V2 head on the same learner views; one row per root. '
            'PASS: pooled Pearson >= 0.5 and same-sign >= 0.8 (the suggested bars), with both learner seats present (>= 30 rows each; '
            'INCONCLUSIVE otherwise). Same-sign is read on confident rows (both |v| >= 0.2) when there are at least 30, where a sign means '
            'something, else on every row. FAIL "perspective suspect" when the pooled group has Pearson <= 0 or same-sign <= 0.5, or a '
            'subgroup clears the subgroup rule: a sign or perspective error mirrors the agreement, and a seat- or ply-specific error shows only '
            'in its group. WARN in between: two critics of different families (a tanh PPO critic with GAE lambda 0.754 vs an HL-Gauss head) '
            'may agree weakly without being inverted. --head-archive is required; a run without a V2 verdict needs a near-terminal one.',
        seats_rule='the champion is observer-relative: Pearson(its value on the owner\'s view, on the opponent\'s view; one row per walked '
            'state) <= -0.5 PASS, >= 0 FAIL (a critic from a fixed side correlates positively, a zero-sum observer-relative one near -1; '
            'sampled hidden information and the opponent view\'s three zeroed history columns weaken it), WARN between; and EACH pricer\'s '
            'value is nearer the owner\'s-view value than the opponent\'s on every row where the two differ by more than its margin (f32 1e-3; '
            'bf16 0.5 = 2 x the gross bound, so a bf16 within 0.25 of the owner\'s view always passes), FAIL otherwise: the learner value is '
            'the owner\'s view at every ply. INCONCLUSIVE under 30 rows.',
        near_terminal_rule='parents one step before a terminal (roots over turn-1 cells, both worlds pooled per root; turn-1 cells over '
            'depth-2 leaves; ply-2 nodes over depth-3 leaves) whose decisive terminal children (+1/-1 from the learner\'s side, as the pricer '
            'assigns them; draws excluded) all share one sign: the share whose pricer value has that sign. Pooled: PASS >= 0.9 (suggested), '
            'FAIL <= 0.5 (an inverted value is wrong more often than chance), WARN between, INCONCLUSIVE under 30 parents. Per learner seat '
            '(>= 30 parents): FAIL under the subgroup rule, WARN under 0.9. The V2 head\'s share on the same (walked) parents is the '
            'reference.'),
    scale=dict(ratio=SCALE_RATIO, saturated=SATURATED, saturation_warn=SATURATION_WARN, rule='WARN when the champion\'s p50 |v| is outside '
        '[0.5, 2] x the V2 head\'s at any level (>= 30 rows each), or when more than 25% of its values have |v| > 0.95 at any level (a '
        'saturated critic compresses completed-Q gaps): v5-tuned fixed constants (NEAR_TIE, SPREAD_THRESHOLD) would mean something else. '
        'The exactness gates are evaluator-relative and unaffected. PASS otherwise.'),
    min_rows=MIN_ROWS, exit=EXIT, rollup='FAIL (exit 1) if any check fails. INCONCLUSIVE (exit 2) if the sign is not established (neither the '
        'V2 check nor the near-terminal check reached PASS, WARN or FAIL), the adapter compared no f32 row, or no root was validated. '
        'Otherwise WARN if any check warns, is INCONCLUSIVE or SKIPPED, else PASS (exit 0).')


# Frame positions by action path -------------------------------------------------------------------------------------------------------
T1_FIELDS = dict(own_logits='t1_own_logits', own_legal='t1_own_legal', own_requested='t1_own_requested', opp_logits='t1_peer_logits',
    opp_legal='t1_peer_legal', opp_requested='t1_peer_requested', digest='t1_digest_own')
NODE_FIELDS = dict(own_logits='own_logits', own_legal='own_legal', own_requested='own_requested', opp_logits='opp_logits', opp_legal='opp_legal',
    opp_requested='opp_requested', digest='digest_own')


def _take(source, at, names):
    return {key: np.asarray(getattr(source, name))[at] for key, name in names.items() if getattr(source, name, None) is not None}


def paths(frame):
    """Every enumerated position of a DepthNFrame by ply: {p: dict(spec [n, 2 + 2p] int64 = (root, world, a1, r1, ..., a_p, r_p), status,
    value, digest (when the pricer digests), and at interior plies `node` (its dense id) and both seats' logits, legal masks and request
    flags)}. Ply 1 holds the turn-1 cells, plies 2..N-1 the interior nodes, ply N the leaves; status-0 positions are absent. The actions
    are each slot's mapped action, so two frames whose selections order or choose slots differently still join by path."""
    N = int(frame.depth);status = np.asarray(frame.t1_status);q, w, a, r = np.nonzero(status != 0);at = (q, w, a, r)
    out = {1: dict(spec=np.stack(at, 1).astype(np.int64).reshape(-1, 4), node=(a * 9 + r).astype(np.int64), status=status[at],
        value=np.asarray(frame.t1_value)[at], **_take(frame, at, T1_FIELDS))}
    for p in range(2, N + 1):
        prev = out[p - 1];parent = np.flatnonzero(prev['status'] == 2);pspec, pid = prev['spec'][parent], prev['node'][parent];pq, pw = pspec[:, 0], pspec[:, 1]
        if p == 2:own, opp = (np.asarray(m)[pq, pw, pid // 9, pid % 9] for m in (frame.t1_own_actions, frame.t1_opp_actions))
        else:lv = frame.nodes[p - 3];own, opp = (np.asarray(m)[pq, pw, pid] for m in (lv.own_actions, lv.opp_actions))
        k_opp = opp.shape[1];K = own.shape[1] * k_opp;i, s, x = np.nonzero((own[:, :, None] >= 0) & (opp[:, None, :] >= 0))
        spec = np.concatenate([pspec[i], own[i, s][:, None], opp[i, x][:, None]], axis=1).astype(np.int64);cq, cw = pq[i], pw[i]
        if p < N:
            lv = frame.nodes[p - 2];node = (pid[i] * K + s * k_opp + x).astype(np.int64);at = (cq, cw, node)
            level = dict(spec=spec, node=node, status=np.asarray(lv.status)[at], value=np.asarray(lv.value)[at], **_take(lv, at, NODE_FIELDS))
        else:
            at = (cq, cw, pid[i], s, x)
            level = dict(spec=spec, status=np.asarray(frame.leaf_status)[at], value=np.asarray(frame.leaf_value)[at], **_take(frame, at, dict(digest='leaf_digest')))
        keep = level['status'] != 0;out[p] = {k: v[keep] for k, v in level.items()}
    return out


def level_name(ply, depth):
    return 'root' if ply == 0 else 'turn1' if ply == 1 else 'leaf' if ply == depth else f'ply{ply}'


def valued_status(ply, depth):
    """The status of a position the champion values: an expanded node (2) or a head-valued leaf (4)."""
    return 4 if ply == depth else 2


def keys(spec):
    """One int64 per action path (root, world, actions in -1..8): unique within a ply."""
    spec = np.asarray(spec, np.int64)
    if spec.ndim != 2:raise ValueError('[n, 2 + 2p] action-path specs required')
    key = spec[:, 0] * 2 + spec[:, 1]
    for c in range(2, spec.shape[1]):key = key * 10 + spec[:, c] + 1
    return key


def lookup(table, spec):
    """(index into `table`'s rows, found) for each row of `spec`, by action path; index 0 where not found."""
    a, b = keys(table), keys(spec)
    if not len(a):return np.zeros(len(b), np.int64), np.zeros(len(b), bool)
    order = np.argsort(a, kind='stable');index = order[np.clip(np.searchsorted(a[order], b), 0, len(a) - 1)]
    return index, a[index] == b


def _pick(x, index, found, fill):
    x = np.asarray(x)
    if not len(x):return np.full((len(index),) + x.shape[1:], fill, np.float32 if isinstance(fill, float) else x.dtype)
    out = x[index].astype(np.float32 if isinstance(fill, float) and x.dtype.kind != 'f' else x.dtype, copy=True);out[~found] = fill
    return out


def _choose(rng, index, size):
    index = np.asarray(index, np.int64)
    return index[np.sort(rng.choice(len(index), size=min(size, len(index)), replace=False))] if len(index) else index


def _unique(spec):
    _, first = np.unique(keys(spec), return_index=True);return spec[np.sort(first)]


# Statistics -------------------------------------------------------------------------------------------------------------------------
def spread(diff):
    """|diff|: count, non-finite count, max, p99, p99.9, p99.99, mean and the exactly-equal share."""
    d = np.abs(np.asarray(diff, np.float64)).reshape(-1);finite = np.isfinite(d);out = dict(count=int(d.size), nonfinite=int(np.count_nonzero(~finite)))
    d = d[finite]
    if not d.size:return out
    return dict(out, max=float(d.max()), **{f'p{t:g}': float(v) for t, v in zip(TAILS, np.percentile(d, TAILS))}, mean=float(d.mean()),
        exact=float(np.mean(d == 0)))


def magnitude(values):
    """|v|: count, p50, p90, p99 and the share above SATURATED."""
    v = np.abs(np.asarray(values, np.float64)).reshape(-1);v = v[np.isfinite(v)]
    if not v.size:return dict(count=0)
    return dict(count=int(v.size), **{f'p{p}': float(x) for p, x in zip((50, 90, 99), np.percentile(v, (50, 90, 99)))},
        saturated=float(np.mean(v > SATURATED)))


def _pearson(x, y):
    sx, sy = x.std(), y.std()
    return None if sx == 0 or sy == 0 else float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _ranks(x):
    """Average ranks (ties share their mean rank), as Spearman's correlation takes them."""
    order = np.argsort(x, kind='mergesort');ranks = np.empty(len(x));ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def agreement(x, y):
    """The champion x against the reference y: Pearson, Spearman, same-sign over every row and over confident rows (both |v| >= CONFIDENT),
    and the OLS slope and intercept of x on y."""
    x, y = np.asarray(x, np.float64).reshape(-1), np.asarray(y, np.float64).reshape(-1);ok = np.isfinite(x) & np.isfinite(y);x, y = x[ok], y[ok]
    out = dict(count=int(len(x)))
    if len(x) < 3:return out
    confident = (np.abs(x) >= CONFIDENT) & (np.abs(y) >= CONFIDENT);var = y.var();slope = None if var == 0 else float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    return dict(out, pearson=_pearson(x, y), spearman=_pearson(_ranks(x), _ranks(y)), same_sign=float(np.mean(np.sign(x) == np.sign(y))),
        confident=int(confident.sum()), same_sign_confident=float(np.mean(np.sign(x[confident]) == np.sign(y[confident]))) if confident.any() else None,
        slope=slope, intercept=None if slope is None else float(x.mean() - slope * y.mean()))


def fisher_upper(r, n):
    """One-sided 99% upper bound on a correlation r from n rows (Fisher z); None without one."""
    if r is None or n < 4:return None
    return float(np.tanh(np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12)) + Z99 / math.sqrt(n - 3)))


def wilson_upper(share, n):
    """One-sided 99% upper bound on a share from n rows (Wilson score); None without one."""
    if share is None or n < 1:return None
    z2 = Z99 * Z99;half = Z99 * math.sqrt(share * (1 - share) / n + z2 / (4 * n * n))
    return float((share + z2 / (2 * n) + half) / (1 + z2 / n))


def worst(statuses):
    statuses = [s for s in statuses if s is not None];return max(statuses, key=ORDER.__getitem__) if statuses else 'SKIPPED'


# 1. Precision ---------------------------------------------------------------------------------------------------------------------
def _argmax(logits, legal):
    return np.argmax(np.where(legal, logits, -np.inf), axis=-1)


def precision_call(frames, pf, requests, roots, families, backups):
    """bf16 against f32 on one call (frames and their paths pf {(family, name)}, requests {name}, roots {name}: root_outputs): signed differences per label
    (positions joined by action path; values where both frames value the position, priors on the legal lanes where both seats are
    requested), sign disagreements on confident f32 values ([confident, disagreeing]), argmax changes, status and terminal differences,
    enumeration overlap and per-root decisions."""
    f, b = REFERENCE, TESTED;values, logits, argmax, status, terminal, enumeration, decisions, signs = {}, {}, {}, {}, {}, {}, {}, {}
    def flips(x, y):
        confident = np.abs(x) >= CONFIDENT;return [int(confident.sum()), int(np.count_nonzero(np.sign(x[confident]) != np.sign(y[confident])))]
    legal = np.asarray([r.root_legal for r in requests[f]], bool).reshape(-1, 9)
    la, lb = (np.asarray([r.root_logits for r in requests[n]], np.float32).reshape(-1, 9) for n in NAMES)
    logits['root.own'] = (la - lb)[legal];argmax['root.own'] = [len(la), int(np.count_nonzero(_argmax(la, legal) != _argmax(lb, legal)))]
    values['root'] = roots[f]['value'].astype(np.float64) - roots[b]['value'];signs['root'] = flips(roots[f]['value'], roots[b]['value'])
    A, B = frames[(families[0], f)], frames[(families[0], b)]
    req = (np.asarray(A.pre) == 0) & (np.asarray(B.pre) == 0) & np.asarray(A.root_peer_requested, bool) & np.asarray(B.root_peer_requested, bool)
    x, y, lanes = np.asarray(A.root_peer_logits)[req], np.asarray(B.root_peer_logits)[req], np.asarray(A.root_peer_legal, bool)[req]
    logits['root.opp'] = (x - y)[lanes];argmax['root.opp'] = [int(req.sum()), int(np.count_nonzero(_argmax(x, lanes) != _argmax(y, lanes)))]
    for family in families:
        A, B = frames[(family, f)], frames[(family, b)];depth = int(A.depth);pa, pb = pf[(family, f)], pf[(family, b)]
        for p in sorted(pa):
            label = f'{family}.{level_name(p, depth)}';X, Y = pa[p], pb[p];ib, found = lookup(Y['spec'], X['spec']);ia = np.flatnonzero(found);ib = ib[found]
            enumeration[label] = [len(X['spec']), len(Y['spec']), len(ia)]
            sa, sb = X['status'][ia], Y['status'][ib];same = sa == sb;status[label] = int(np.count_nonzero(~same))
            valued = same & (sa == valued_status(p, depth));ends = same & (sa == 1)
            x, y = X['value'][ia][valued], Y['value'][ib][valued];values[label] = x.astype(np.float64) - y;signs[label] = flips(x, y)
            terminal[label] = int(np.count_nonzero(X['value'][ia][ends] != Y['value'][ib][ends]))
            if p == depth:continue
            for seat in ('own', 'opp'):
                req = valued & X[f'{seat}_requested'][ia] & Y[f'{seat}_requested'][ib];x, y = X[f'{seat}_logits'][ia][req], Y[f'{seat}_logits'][ib][req]
                lanes = X[f'{seat}_legal'][ia][req];logits[f'{label}.{seat}'] = (x - y)[lanes]
                argmax[f'{label}.{seat}'] = [int(req.sum()), int(np.count_nonzero(_argmax(x, lanes) != _argmax(y, lanes)))]
        da, db = gate.root_decisions(A, requests[f], backups), gate.root_decisions(B, requests[b], backups)
        decisions[family] = [(None, None) if u is None else u for u in da], [(None, None) if v is None else v for v in db]
    return dict(values=values, logits=logits, argmax=argmax, status_differs=status, terminal_differs=terminal, enumeration=enumeration, decisions=decisions,
        signs=signs)


def precision_check(calls):
    merged = {}
    for c in calls:
        for part in ('values', 'logits'):
            for k, v in c['precision'][part].items():merged.setdefault(part, {}).setdefault(k, []).append(np.asarray(v, np.float64))
    values = {k: spread(np.concatenate(v)) for k, v in merged.get('values', {}).items()}
    logits = {k: spread(np.concatenate(v)) for k, v in merged.get('logits', {}).items()}
    total = lambda part, k: [sum(c['precision'][part][k][i] for c in calls if k in c['precision'][part]) for i in (0, 1)]
    names = sorted({k for c in calls for k in c['precision']['argmax']})
    argmax = {k: dict(nodes=n, changed=m, share=None if not n else m / n) for k in names for n, m in [total('argmax', k)]}
    counts = lambda part: {k: sum(c['precision'][part].get(k, 0) for c in calls) for k in sorted({k for c in calls for k in c['precision'][part]})}
    enumeration = {}
    for c in calls:
        for k, (x, y, z) in c['precision']['enumeration'].items():
            e = enumeration.setdefault(k, dict(f32=0, bf16=0, shared=0));e['f32'] += x;e['bf16'] += y;e['shared'] += z
    decisions = {}
    for c in calls:
        for family, (da, db) in c['precision']['decisions'].items():
            d = decisions.setdefault(family, dict(roots=0, labeled_both=0, agree=0, refused_f32_only=0, refused_bf16_only=0, flips=[]))
            for root, (u, v) in enumerate(zip(da, db)):
                d['roots'] += 1;d['refused_f32_only'] += u[0] is None and v[0] is not None;d['refused_bf16_only'] += u[0] is not None and v[0] is None
                if u[0] is None or v[0] is None:continue
                d['labeled_both'] += 1;d['agree'] += u[0] == v[0]
                if u[0] != v[0] and len(d['flips']) < g2.FIRST:d['flips'].append(dict(call=c['call'], root=root, f32=u[0], bf16=v[0], f32_gap=u[1], bf16_gap=v[1]))
    for d in decisions.values():d['share'] = d['agree'] / d['labeled_both'] if d['labeled_both'] else None
    signs = {}
    for c in calls:
        for k, (n, m) in c['precision'].get('signs', {}).items():e = signs.setdefault(k, dict(confident=0, disagree=0));e['confident'] += n;e['disagree'] += m
    for e in signs.values():e['share'] = e['disagree'] / e['confident'] if e['confident'] else None
    pooled = dict(confident=sum(e['confident'] for e in signs.values()), disagree=sum(e['disagree'] for e in signs.values()))
    pooled['share'] = pooled['disagree'] / pooled['confident'] if pooled['confident'] else None
    biggest = max(((v['max'], k) for k, v in values.items() if 'max' in v), default=(None, None));failed, warned = [], []
    if biggest[0] is not None and biggest[0] > BF16_GROSS_VALUE:failed.append(f'gross bf16 divergence: max |value diff| vs f32 {biggest[0]:.4g} at {biggest[1]} > {BF16_GROSS_VALUE}')
    elif biggest[0] is not None and biggest[0] > PRECISION_VALUE_WARN:warned.append(f'bf16 vs f32 max |value diff| {biggest[0]:.4g} at {biggest[1]} > {PRECISION_VALUE_WARN}')
    for k, e in dict(signs, pooled=pooled).items():
        if e['confident'] >= MIN_ROWS and e['share'] > BF16_SIGN_SHARE:
            failed.append(f'gross bf16 divergence: the other sign than f32 on {e["disagree"]} of {e["confident"]} confident values ({e["share"]:.3f} > {BF16_SIGN_SHARE}) at {k}')
    nonfinite = [k for k, v in values.items() if v.get('nonfinite')]
    if nonfinite:failed.append('non-finite bf16 vs f32 value differences at ' + ', '.join(nonfinite))
    warned += [f'{family}: completed-Q argmax agreement {d["share"]:.4f} < {PRECISION_DECISION_WARN}' for family, d in decisions.items()
        if d['share'] is not None and d['share'] < PRECISION_DECISION_WARN]
    shared = sum(e['shared'] for e in enumeration.values())
    status = 'FAIL' if failed else 'INCONCLUSIVE' if not shared else 'WARN' if warned else 'PASS'
    if not failed and not shared:failed.append('the bf16 and f32 frames share no priced position')
    return dict(status=status, reasons=failed + warned, max_value_diff=dict(value=biggest[0], at=biggest[1], warn=PRECISION_VALUE_WARN, gross=BF16_GROSS_VALUE),
        sign_disagreements=dict(signs, pooled=pooled), values=values, logits=logits, argmax_changes=argmax, status_differs=counts('status_differs'),
        terminal_differs=counts('terminal_differs'), enumeration=enumeration, decisions=decisions, rule=THRESHOLDS['precision']['rule'])


# 2. The independent path ---------------------------------------------------------------------------------------------------------
def _path(executor, observe, states, publics, owners, spec, plies, view=None):
    """One spec [row, a1, r1, ..., a_plies, r_plies] walked from its root world as depthn_oracle._walk walks it (the oracle's own step:
    D's turn-1 joint rule, each later seat requested iff its engine mask has a legal action, the node's own chance), then observed at the
    final state twice: the executor's V2 view of the owner (what the pricer's evaluator reads; its digest through `view`) and the
    champion's own layout of the owner and of the opponent through `observe` (the ladder's legacy_champion leaf)."""
    import jax.numpy as j
    from . import depth2_oracle as o2, depthn_oracle as on
    row = spec[0];state, public, owner = o2._gather(states, row), o2._gather(publics, row), owners[row]
    masks, seat, own, peer = o2._views(executor, state, public, owner)
    own_requested, peer_requested = j.asarray(True), j.any(peer.observation.legal_action_mask[0])
    invalid = broken = j.asarray(False)
    for p in range(plies):
        after = on._step(executor, state, public, owner, seat, own, peer, spec[1 + 2 * p], spec[2 + 2 * p], own_requested, peer_requested)
        if p < plies - 1:broken = broken | after.step.invalid_actions[0] | after.step.state.done[0]
        invalid = after.step.invalid_actions[0];state, public = after.step.state, after.trackers
        masks, _, own, peer = o2._views(executor, state, public, owner);own_requested, peer_requested = j.any(masks[0, owner]), j.any(masks[0, 1 - owner])
    mine, theirs = observe(state, public, seat, masks[:, owner]), observe(state, public, 1 - seat, masks[:, 1 - owner])
    facts = dict(invalid=invalid, done=state.done[0], winner=state.winner[0], broken=broken, own_error=own.observation.error[0],
        champion_own_error=mine.observation.error[0], champion_peer_error=theirs.observation.error[0], digest=o2._digest(own.inputs, view))
    return facts, dict(v2=own.inputs, own=mine.inputs, peer=theirs.inputs)


class Walker:
    """The independent walks: one jitted vmap of _path per ply count over fixed chunks (a short last chunk repeats its first spec; padded
    rows are dropped), each chunk handed to `heads` before the next is walked."""
    def __init__(self, executor, observe, *, chunk=CHUNK, view=None):
        if type(chunk) is not int or chunk <= 0:raise ValueError('positive walk chunk required')
        self.executor, self.observe, self.chunk, self.view, self.fns, self.seconds, self.rows = executor, observe, chunk, view, {}, 0., 0

    def fn(self, plies):
        if plies not in self.fns:
            import jax
            ex, observe, view = self.executor, self.observe, self.view
            self.fns[plies] = jax.jit(jax.vmap(lambda s, p, o, spec: _path(ex, observe, s, p, o, spec, plies, view), in_axes=(None, None, None, 0)))
        return self.fns[plies]

    def run(self, stack, specs, plies, heads):
        """specs [n, 2 + 2 plies] = (root, world, actions...): the walk facts, `heads`' outputs, the specs and each row's learner seat."""
        import jax
        import jax.numpy as j
        specs = np.asarray(specs, np.int64).reshape(-1, 2 + 2 * plies)
        if not len(specs):return {}
        clock = time.perf_counter();rows = np.asarray([stack.rows[(int(q), int(w))] for q, w in specs[:, :2]], np.int32)
        flat = np.concatenate([rows[:, None], specs[:, 2:]], axis=1).astype(np.int32);fn = self.fn(plies);parts = []
        for start in range(0, len(flat), self.chunk):
            part = flat[start:start + self.chunk];count = len(part);padded = np.concatenate([part, np.repeat(part[:1], self.chunk - count, axis=0)])
            facts, inputs = fn(stack.states, stack.publics, stack.owners, j.asarray(padded))
            out = {k: np.asarray(v)[:count] for k, v in jax.device_get(facts).items()};out.update({k: np.asarray(v)[:count] for k, v in heads(inputs).items()})
            parts.append(out)
        self.seconds += time.perf_counter() - clock;self.rows += len(flat)
        return dict({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, spec=specs, seat=np.asarray(stack.owner)[rows].astype(np.int64))


def _rows_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:return np.zeros(len(a), bool)
    same = a == b
    if a.dtype.kind == 'f' and b.dtype.kind == 'f':same |= np.isnan(a) & np.isnan(b)
    return same.reshape(len(a), -1).all(axis=1)


class Heads:
    """A walked chunk's evaluations: champion.forward (f32, its own jit, never through ChampionEvaluator) on the independent layouts of
    the owner and of the opponent; each evaluator's adapter on the V2 owner view (same rows, same batch); the V2 reference head; and
    whether the independent layout is bit-equal to what EACH evaluator reads (its `view`) of the V2 view."""
    def __init__(self, evaluators, reference=None):
        import jax
        from . import champion
        ev = evaluators[REFERENCE];config = ev.config;self.params, self.reference = ev.params, reference;self.views = {name: e.view for name, e in evaluators.items()}
        self.raw = jax.jit(lambda p, *x: champion.forward(p, *x, config, compute_dtype='float32'))
        self.adapters = {name: (e.params, (lambda e: jax.jit(lambda p, x: e(p, x)))(e)) for name, e in evaluators.items()}

    def __call__(self, inputs):
        import jax
        v2 = jax.tree.map(lambda x: x.reshape((x.shape[0],) + x.shape[2:]), inputs['v2']);out = {}
        for side, prefix in (('own', 'raw'), ('peer', 'raw_peer')):
            o = jax.device_get(self.raw(self.params, *[inputs[side][k][:, 0] for k in RAW_KEYS]))
            out[prefix + '_value'], out[prefix + '_logits'] = np.asarray(o.value, np.float32), np.asarray(o.policy_logits, np.float32)
        for name, (params, fn) in self.adapters.items():
            o = jax.device_get(fn(params, v2));out[f'adapter_{name}_value'], out[f'adapter_{name}_logits'] = np.asarray(o.value, np.float32), np.asarray(o.policy_logits, np.float32)
        if self.reference is not None:
            o = jax.device_get(self.reference.forward(self.reference.params, v2));value = np.asarray(o.value, np.float32).reshape(-1)
            out['v2_value'], out['v2_ok'] = value, np.asarray(o.valid, bool).reshape(-1) & np.isfinite(value) & (np.abs(value) <= 1)
        host, v2_host = jax.device_get(inputs['own']), jax.device_get(v2)
        for name, view in self.views.items():
            seen = view(v2_host);out[f'inputs_equal_{name}'] = np.all([_rows_equal(np.asarray(seen[k])[:, 0], np.asarray(host[r])[:, 0]) for k, r in zip(ce.INPUT_KEYS, RAW_KEYS)], axis=0)
        return out


def root_outputs(requests, evaluators):
    """Each evaluator's adapter on every request's V2 root inputs, batch 1 and jitted as g2.bind_evaluator_roots runs it: the root value
    (the pricers store none), the logits, the bound root prior and whether the two are bit-equal."""
    import jax
    out = {}
    for name, ev in evaluators.items():
        fn = (lambda e: jax.jit(lambda p, x: e(p, x)))(ev)
        rows = [jax.device_get(fn(ev.params, {k: np.asarray(v) for k, v in r.root_inputs.items()})) for r in requests[name]]
        logits = np.asarray([np.asarray(o.policy_logits, np.float32).reshape(-1) for o in rows], np.float32).reshape(-1, 9)
        bound = np.asarray([r.root_logits for r in requests[name]], np.float32).reshape(-1, 9)
        out[name] = dict(value=np.asarray([float(np.asarray(o.value).reshape(-1)[0]) for o in rows], np.float32), logits=logits, bound=bound,
            rebound_equal=bool(np.array_equal(logits, bound)))
    return out


def pricer_fields(family, ply, spec, name, pf, roots, frames, requests, depth):
    """One evaluator's pricer surface at `spec` rows: found, valued, value, both seats' logits, legal masks and request flags, and the
    digest (absent at roots). A root's value is the adapter's on its root inputs (root_outputs) and its learner prior the bound root
    logits; a position the frame does not hold is NaN / False."""
    n = len(spec)
    if ply == 0:
        q, w = spec[:, 0], spec[:, 1];frame = frames[('depth2', name)]
        return dict(found=np.ones(n, bool), valued=np.asarray(frame.pre)[q, w] == 0, value=roots[name]['value'][q], own_logits=roots[name]['bound'][q],
            own_legal=np.asarray([requests[name][i].root_legal for i in q], bool).reshape(n, 9), own_requested=np.ones(n, bool),
            opp_logits=np.asarray(frame.root_peer_logits, np.float32)[q, w], opp_legal=np.asarray(frame.root_peer_legal, bool)[q, w],
            opp_requested=np.asarray(frame.root_peer_requested, bool)[q, w])
    level = pf[(family, name)][ply];index, found = lookup(level['spec'], spec);d = depth[family]
    out = dict(found=found, valued=found & (_pick(level['status'], index, found, 0) == valued_status(ply, d)), value=_pick(level['value'], index, found, np.nan))
    for seat in ('own', 'opp'):
        out[f'{seat}_logits'] = _pick(level[f'{seat}_logits'], index, found, np.nan) if f'{seat}_logits' in level else np.full((n, 9), np.nan, np.float32)
        for k, fill in (('legal', np.zeros(9, bool)), ('requested', False)):
            out[f'{seat}_{k}'] = _pick(level[f'{seat}_{k}'], index, found, False) if f'{seat}_{k}' in level else np.zeros((n,) + np.shape(fill), bool)
    if 'digest' in level:out['digest'] = _pick(level['digest'], index, found, 0)
    return out


def terminal_parents(pf, families, rng, sample, root_world):
    """Parents one step before a terminal, by source: {label: dict(ply, spec, plus, minus, walk)}: per parent the decisive terminal children
    (status 1, value +1 / -1 from the learner's side) won and lost; `walk` the seeded sample of unanimous parents the V2 head is read on.
    A root is ONE parent (its value is the root's, the same for both worlds): both worlds' decisive turn-1 children pooled, at the world its
    root row uses (root_world {root: world})."""
    sources = [('root', 0, pf[('depth2', REFERENCE)][1]), ('depth2.turn1', 1, pf[('depth2', REFERENCE)][2])]
    if 'depth3' in families:sources.append(('depth3.ply2', 2, pf[('depth3', REFERENCE)][3]))
    out = {}
    for label, ply, children in sources:
        decisive = (children['status'] == 1) & (np.abs(children['value']) == 1);spec, v = children['spec'][decisive][:, :-2], children['value'][decisive]
        if ply == 0:
            keep = np.isin(spec[:, 0], np.asarray(sorted(root_world), np.int64));spec, v = spec[keep], v[keep]
            spec = np.stack([spec[:, 0], np.asarray([root_world[int(q)] for q in spec[:, 0]], np.int64).reshape(-1)], axis=1).reshape(-1, 2)
        _, first, inverse = np.unique(keys(spec), return_index=True, return_inverse=True);inverse = inverse.reshape(-1)
        plus = np.bincount(inverse, weights=(v > 0), minlength=len(first)).astype(np.int64);minus = np.bincount(inverse, weights=(v < 0), minlength=len(first)).astype(np.int64)
        spec = spec[first].reshape(-1, 2 + 2 * ply);out[label] = dict(ply=ply, spec=spec, plus=plus, minus=minus, walk=_choose(rng, np.flatnonzero((plus == 0) | (minus == 0)), sample))
    return out


def call_records(call, frames, pf, requests, roots, stack, walker, heads, *, families, sample, rng):
    """The walked rows of one call by group (root, {family}.{level}), each aligned with both pricers' surfaces (pf: each frame's paths);
    the near-terminal parents by source; the labels of priced levels with no valued position to sample (`empty`); and the uniform
    frame-value samples the scale section reads. A root is one row: the first of its materialized worlds with pre 0 (its value and the V2
    head's are per root: a pre-0 world's view IS the root inputs, so both worlds would duplicate the row)."""
    depth = {family: int(frames[(family, REFERENCE)].depth) for family in families}
    pre = np.asarray(frames[('depth2', REFERENCE)].pre);root_world = {}
    for q, w in sorted(stack.rows):
        if pre[q, w] == 0:root_world.setdefault(int(q), int(w))
    groups = {'root': (None, 0, np.asarray(sorted(root_world.items()), np.int64).reshape(-1, 2))}
    for family in families:
        for p in range(1, depth[family] + 1):
            label = f'{family}.{level_name(p, depth[family])}'
            if p == 1 and family != 'depth2':groups[label] = (family, 1, groups['depth2.turn1'][2]);continue
            level = pf[(family, REFERENCE)][p];valued = np.flatnonzero(level['status'] == valued_status(p, depth[family]))
            groups[label] = (family, p, level['spec'][_choose(rng, valued, sample)])
    parents = terminal_parents(pf, families, rng, sample, root_world);walked = {}
    for p in sorted({g[1] for g in groups.values()} | {x['ply'] for x in parents.values()}):
        specs = [g[2] for g in groups.values() if g[1] == p] + [x['spec'][x['walk']] for x in parents.values() if x['ply'] == p]
        specs = _unique(np.concatenate([s.reshape(-1, 2 + 2 * p) for s in specs]))
        if len(specs):walked[p] = walker.run(stack, specs, p, heads)
    records, empty = {}, []
    for label, (family, p, spec) in groups.items():
        if not len(spec) or p not in walked:empty.append(label);continue
        index, found = lookup(walked[p]['spec'], spec);assert found.all(), 'every sampled row is walked'
        w = {k: v[index] for k, v in walked[p].items()};n = len(spec)
        rec = dict(call=np.full(n, call, np.int64), ply=np.full(n, p, np.int64), key=keys(spec), seat=w['seat'], **{f'inputs_equal_{m}': w[f'inputs_equal_{m}'] for m in NAMES},
            walk_ok=~w['invalid'] & ~w['done'] & ~w['broken'] & (w['own_error'] == 0) & (w['champion_own_error'] == 0), peer_ok=w['champion_peer_error'] == 0,
            walk_digest=w['digest'].astype(np.uint32), raw_value=w['raw_value'], raw_logits=w['raw_logits'], raw_peer_value=w['raw_peer_value'],
            raw_peer_logits=w['raw_peer_logits'], v2_value=w.get('v2_value', np.full(n, np.nan, np.float32)), v2_ok=w.get('v2_ok', np.zeros(n, bool)))
        for name in NAMES:
            rec[f'same_batch_value_{name}'], rec[f'same_batch_logits_{name}'] = w[f'adapter_{name}_value'], w[f'adapter_{name}_logits']
            rec.update({f'{k}_{name}': v for k, v in pricer_fields(family, p, spec, name, pf, roots, frames, requests, depth).items()})
        records[label] = rec
    owners = np.asarray([int(r.plans.root.address.owner) for r in requests[REFERENCE]], np.int64);terminal = {}
    for label, x in parents.items():
        spec = x['spec'];family = None if label == 'root' else label.split('.')[0];n = len(spec)
        rec = dict(call=np.full(n, call, np.int64), seat=owners[spec[:, 0]] if n else np.zeros(0, np.int64), plus=x['plus'], minus=x['minus'])
        for name in NAMES:
            fields = pricer_fields(family, x['ply'], spec, name, pf, roots, frames, requests, depth) if n else dict(valued=np.zeros(0, bool), value=np.zeros(0, np.float32))
            rec[f'value_{name}'] = np.where(fields['valued'], fields['value'], np.nan).astype(np.float32)
        v2 = np.full(n, np.nan, np.float32)
        if n and x['ply'] in walked and 'v2_value' in walked[x['ply']]:
            index, found = lookup(walked[x['ply']]['spec'], spec);ok = found & walked[x['ply']]['v2_ok'][index];v2[ok] = walked[x['ply']]['v2_value'][index][ok]
        rec['v2_value'] = v2;terminal[label] = rec
    population = {}
    for family in families:
        for p, level in pf[(family, REFERENCE)].items():
            v = level['value'][level['status'] == valued_status(p, depth[family])];population[f'{family}.{level_name(p, depth[family])}'] = v[_choose(rng, np.arange(len(v)), FRAME_SAMPLE)]
    population['root'] = roots[REFERENCE]['value']
    return records, terminal, population, empty


# 2-4. Judging ---------------------------------------------------------------------------------------------------------------------
def merge(calls, field):
    out = {}
    for c in calls:
        for label, rec in c[field].items():out.setdefault(label, []).append(rec)
    return {label: {k: np.concatenate([r[k] for r in recs]) for k in recs[0]} for label, recs in out.items()}


def adapter_check(records, expected=()):
    """Check 2 over every walked group (THRESHOLDS['adapter']); `expected` the labels of every priced level (a priced level with no row WARNs).
    `observed` puts each maximum beside its bound."""
    groups, failed, warned, compared = {}, [], [], 0;top = lambda name, keys: max((groups[k][name][x].get('max', 0.) for k in groups for x in keys), default=None)
    warned += [f'{label}: a priced level with no valued position sampled (nothing compared there)' for label in expected if label not in records]
    for label, r in records.items():
        ok = r['walk_ok'];entry = dict(walked=len(ok), walk_ok=int(ok.sum()))
        for name in NAMES:
            valued = r[f'valued_{name}'];use = valued & ok
            e = dict(compared=int(use.sum()), inputs_unequal=int(np.count_nonzero(ok & ~r[f'inputs_equal_{name}'])), walk_disagrees=int(np.count_nonzero(valued & ~ok)),
                value=spread(r[f'value_{name}'][use] - r['raw_value'][use]), same_batch_value=spread(r[f'same_batch_value_{name}'][ok] - r['raw_value'][ok]),
                same_batch_logits=spread((r[f'same_batch_logits_{name}'] - r['raw_logits'])[ok]))
            for seat, raw, extra in (('own', 'raw_logits', ok), ('opp', 'raw_peer_logits', ok & r['peer_ok'])):
                rows = use & extra & r[f'{seat}_requested_{name}'];e[f'{seat}_logits'] = spread((r[f'{seat}_logits_{name}'][rows] - r[raw][rows])[r[f'{seat}_legal_{name}'][rows]])
            if f'digest_{name}' in r:e['digest_mismatch'] = int(np.count_nonzero(use & np.any(r[f'digest_{name}'] != r['walk_digest'], axis=-1)))
            entry[name] = e
            if e['inputs_unequal']:failed.append(f'{label}: the independent champion layout differs from the {name} evaluator\'s view on {e["inputs_unequal"]} rows')
            if e['walk_disagrees']:failed.append(f'{label}: {e["walk_disagrees"]} positions the {name} frame values end terminal, invalid or in error on the independent walk')
            if e.get('digest_mismatch'):failed.append(f'{label}: {e["digest_mismatch"]} walked positions have another digest than the {name} frame\'s (placement)')
            if name == REFERENCE:
                compared += e['compared']
                for k, bound in (('value', ADAPTER_VALUE), ('same_batch_value', ADAPTER_VALUE), ('own_logits', ADAPTER_LOGIT), ('opp_logits', ADAPTER_LOGIT), ('same_batch_logits', ADAPTER_LOGIT)):
                    if e[k].get('nonfinite'):failed.append(f'{label}: {e[k]["nonfinite"]} non-finite f32 {k} differences')
                    if e[k].get('max', 0.) > bound:failed.append(f'{label}: f32 {k} max |diff| {e[k]["max"]:.3g} > {bound}')
                continue
            for k in ('value', 'same_batch_value'):
                if e[k].get('nonfinite'):failed.append(f'{label}: {e[k]["nonfinite"]} non-finite {name} {k} differences')
                if e[k].get('max', 0.) > BF16_GROSS_VALUE:failed.append(f'{label}: gross {name} divergence: {k} max |diff| vs the f32 raw forward {e[k]["max"]:.4g} > {BF16_GROSS_VALUE}')
                elif e[k].get('max', 0.) > PRECISION_VALUE_WARN:warned.append(f'{label}: {name} {k} max |diff| vs the f32 raw forward {e[k]["max"]:.4g} > {PRECISION_VALUE_WARN}')
        groups[label] = entry
    observed = dict(f32_value_max=top(REFERENCE, ('value', 'same_batch_value')), f32_value_bound=ADAPTER_VALUE,
        f32_logit_max=top(REFERENCE, ('own_logits', 'opp_logits', 'same_batch_logits')), f32_logit_bound=ADAPTER_LOGIT,
        bf16_value_max=top(TESTED, ('value', 'same_batch_value')), bf16_value_warn=PRECISION_VALUE_WARN, bf16_value_gross=BF16_GROSS_VALUE)
    status = 'FAIL' if failed else 'INCONCLUSIVE' if not compared else 'WARN' if warned else 'PASS'
    if not compared and not failed:failed.append('no f32-valued position was compared')
    return dict(status=status, reasons=failed + warned, compared=compared, observed=observed, groups=groups, rule=THRESHOLDS['adapter']['rule'])


def _pooled(records, labels, fields, *, by):
    """`fields` of the uniform groups concatenated, with their labels. by='state': each walked state once (by call, ply and path), for
    quantities of the state itself (the raw forward's, the V2 head's); by='pricer': each pricer's row kept (by call, pricer family, ply
    and path): depth2.leaf and depth3.ply2 hold the same states priced by two pricers, two measurements of the search's values."""
    present = [k for k in labels if k in records]
    if not present:return {f: np.zeros(0, bool) for f in fields + ('seat', 'label', 'call', 'ply', 'key')}
    cat = {f: np.concatenate([records[k][f] for k in present]) for f in fields + ('call', 'ply', 'key', 'seat')}
    cat['label'] = np.concatenate([np.full(len(records[k]['key']), k) for k in present])
    family = np.concatenate([np.full(len(records[k]['key']), ('root', 'depth2', 'depth3').index(k.split('.')[0])) for k in present])
    columns = [cat['call'], cat['ply'], cat['key']] + ([family] if by == 'pricer' else [])
    _, first = np.unique(np.stack(columns, axis=1), axis=0, return_index=True);first = np.sort(first)
    return {k: v[first] for k, v in cat.items()}


def _same(g):
    """(same-sign share, its rows): on confident rows when there are MIN_ROWS of them, else on every row."""
    return (g['same_sign_confident'], g['confident']) if (g.get('confident') or 0) >= MIN_ROWS else (g['same_sign'], g['count'])


def sign_verdict(groups):
    """groups {name: agreement}; 'pooled' is the whole population. FAIL "perspective suspect" when the pooled group is inverted at the point
    thresholds or any subgroup under SUBGROUP_RULE; INCONCLUSIVE without a judged pooled group; PASS at both pooled PASS bars; WARN otherwise."""
    judged = {k: g for k, g in groups.items() if g.get('count', 0) >= MIN_ROWS and g.get('pearson') is not None};inverted = []
    for k, g in judged.items():
        share, n = _same(g)
        if k == 'pooled':
            if g['pearson'] <= PEARSON_FAIL or share <= SIGN_FAIL:inverted.append(f'pooled (Pearson {g["pearson"]:.3f}, same-sign {share:.3f}, n {g["count"]})')
            continue
        r_up, s_up = fisher_upper(g['pearson'], g['count']), wilson_upper(share, n)
        if r_up <= PEARSON_FAIL or s_up <= SIGN_FAIL:
            inverted.append(f'{k} (Pearson {g["pearson"]:.3f}, 99% upper {r_up:.3f}; same-sign {share:.3f}, 99% upper {s_up:.3f}; n {g["count"]})')
    if inverted:return 'FAIL', ['perspective suspect: ' + '; '.join(inverted)]
    if 'pooled' not in judged:return 'INCONCLUSIVE', [f'pooled: {groups.get("pooled", {}).get("count", 0)} rows (< {MIN_ROWS}) or a constant critic']
    p = judged['pooled'];share, _ = _same(p)
    if p['pearson'] >= PEARSON_PASS and share >= SIGN_PASS:return 'PASS', []
    return 'WARN', [f'weak agreement with the V2 head, not inverted: pooled Pearson {p["pearson"]:.3f}, same-sign {share:.3f}']


def v2_check(records):
    """Check 3 (a): the pricers' values (and, reported, the raw forward's) against the V2 head on the same learner views. A PASS or WARN needs
    both learner seats (>= MIN_ROWS rows each); short of that, INCONCLUSIVE (a FAIL stands)."""
    if not any(r['v2_ok'].any() for r in records.values()):return dict(status='SKIPPED', reasons=['no V2 reference head (--head-archive) or no usable V2 row'])
    out, statuses, reasons = {}, [], []
    for name in NAMES + ('raw',):
        value = 'raw_value' if name == 'raw' else f'value_{name}'
        rows = _pooled(records, UNIFORM, (value, 'v2_value', 'v2_ok', 'walk_ok') + (() if name == 'raw' else (f'valued_{name}',)), by='state' if name == 'raw' else 'pricer')
        use = rows['v2_ok'] & (rows['walk_ok'] if name == 'raw' else rows[f'valued_{name}']);x, y, label, seat = rows[value][use], rows['v2_value'][use], rows['label'][use], rows['seat'][use]
        groups = dict(pooled=agreement(x, y), **{k: agreement(x[label == k], y[label == k]) for k in UNIFORM if k in records},
            **{f'seat{s}': agreement(x[seat == s], y[seat == s]) for s in (0, 1)},
            **{f'{k}.seat{s}': agreement(x[(label == k) & (seat == s)], y[(label == k) & (seat == s)]) for k in UNIFORM if k in records for s in (0, 1)})
        status, why = sign_verdict(groups);short = [f'seat{s}' for s in (0, 1) if groups[f'seat{s}'].get('count', 0) < MIN_ROWS]
        if short and status in ('PASS', 'WARN'):status, why = 'INCONCLUSIVE', why + [f'both learner seats required: {", ".join(short)} under {MIN_ROWS} rows']
        out[name] = dict(status=status if name != 'raw' else f'{status} (reported)', reasons=why, groups=groups)
        if name != 'raw':statuses.append(status);reasons += [f'{name}: {x}' for x in why]
    return dict(status=worst(statuses), reasons=reasons, by_evaluator=out, rule=THRESHOLDS['perspective']['v2_rule'])


def seats_check(records):
    """Check 3 (b): the champion's observer-relativity (one row per walked state) and the owner's view behind every pricer value (each
    pricer's rows), for both evaluators."""
    rows = _pooled(records, UNIFORM, ('raw_value', 'raw_peer_value', 'walk_ok', 'peer_ok'), by='state')
    both = rows['walk_ok'] & rows['peer_ok'] & np.isfinite(rows['raw_value']) & np.isfinite(rows['raw_peer_value'])
    own, peer = rows['raw_value'][both].astype(np.float64), rows['raw_peer_value'][both].astype(np.float64)
    out = dict(pooled=dict(count=int(both.sum()), pearson=_pearson(own, peer) if len(own) >= 3 else None, zero_sum_same_sign=float(np.mean(np.sign(own) == -np.sign(peer))) if len(own) else None))
    out['by_level'] = {k: dict(count=int(m.sum()), pearson=_pearson(own[m], peer[m]) if m.sum() >= 3 else None) for k in UNIFORM if k in records for m in [rows['label'][both] == k]}
    failed, warned, inconclusive = [], [], []
    priced = _pooled(records, UNIFORM, ('raw_value', 'raw_peer_value', 'walk_ok', 'peer_ok') + tuple(f'{k}_{n}' for n in NAMES for k in ('value', 'valued')), by='pricer')
    seen = priced['walk_ok'] & priced['peer_ok'] & np.isfinite(priced['raw_value']) & np.isfinite(priced['raw_peer_value'])
    for name in NAMES:
        use = seen & priced[f'valued_{name}'] & (np.abs(priced['raw_value'] - priced['raw_peer_value']) > SEAT_MARGIN[name]);v = priced[f'value_{name}'][use]
        closer = np.abs(v - priced['raw_value'][use]) < np.abs(v - priced['raw_peer_value'][use])
        out[f'owner_view_{name}'] = dict(rows=int(use.sum()), margin=SEAT_MARGIN[name], owner_closer=float(closer.mean()) if use.any() else None,
            by_level={k: float(closer[m].mean()) for k in UNIFORM if k in records for m in [priced['label'][use] == k] if m.any()})
        if use.any() and not closer.all():failed.append(f'the {name} pricer value is nearer the opponent\'s view than the owner\'s on {int((~closer).sum())} of {int(use.sum())} rows')
    p = out['pooled']['pearson']
    if out['pooled']['count'] < MIN_ROWS or p is None:inconclusive.append(f'{out["pooled"]["count"]} rows with both views valid (< {MIN_ROWS}) or a constant critic')
    elif p >= SEAT_FAIL:failed.append(f'Pearson(owner view, opponent view) {p:.3f} >= {SEAT_FAIL}: the champion\'s value is not observer-relative')
    elif p > SEAT_PASS:warned.append(f'Pearson(owner view, opponent view) {p:.3f} > {SEAT_PASS}: weakly observer-relative')
    status = 'FAIL' if failed else 'INCONCLUSIVE' if inconclusive else 'WARN' if warned else 'PASS'
    return dict(status=status, reasons=failed + inconclusive + warned, **out, rule=THRESHOLDS['perspective']['seats_rule'])


def near_terminal_check(terminal):
    """Check 3 (c): unanimous near-terminal parents, the pricers' values (and, reported, the V2 head's) against the terminal sign, pooled
    (point thresholds) and per learner seat (SUBGROUP_RULE)."""
    if not terminal:return dict(status='INCONCLUSIVE', reasons=['no frame family priced'])
    sources, statuses, reasons = {}, [], []
    def share(value, r, unanimous, decided):
        """agree: over unanimous parents, sign(value) == the children's sign; pair_agree: over every parent with decisive children (mixed
        ones too), the decisive children whose sign the parent's value has."""
        ok = np.isfinite(value);u = unanimous & ok;v, t = value[u], np.where(r['plus'][u] > 0, 1., -1.);d = decided & ok
        pairs = int((r['plus'][d] + r['minus'][d]).sum());won = float(np.where(value[d] > 0, r['plus'][d], np.where(value[d] < 0, r['minus'][d], 0)).sum())
        return dict(parents=int(len(v)), agree=float(np.mean(np.sign(v) == t)) if len(v) else None, pair_agree=won / pairs if pairs else None)
    pooled = {k: np.concatenate([r[k] for r in terminal.values()]) for k in next(iter(terminal.values()))}
    for label, r in dict(terminal, pooled=pooled).items():
        decided = (r['plus'] + r['minus']) > 0;unanimous = ((r['plus'] == 0) | (r['minus'] == 0)) & decided
        sources[label] = dict(parents_with_decisive_children=int(decided.sum()), unanimous=int(unanimous.sum()),
            **{name: share(r[f'value_{name}'], r, unanimous, decided) for name in NAMES}, v2=share(r['v2_value'], r, unanimous, decided),
            seats={f'seat{s}': {name: share(r[f'value_{name}'], r, unanimous & (r['seat'] == s), decided & (r['seat'] == s)) for name in NAMES} for s in (0, 1)})
    for name in NAMES:
        s = sources['pooled'][name]
        if s['parents'] < MIN_ROWS:statuses.append('INCONCLUSIVE');reasons.append(f'{name}: {s["parents"]} unanimous near-terminal parents (< {MIN_ROWS})')
        elif s['agree'] <= NEAR_TERMINAL_FAIL:statuses.append('FAIL');reasons.append(f'{name}: perspective suspect: near-terminal sign agreement {s["agree"]:.3f} <= {NEAR_TERMINAL_FAIL}')
        elif s['agree'] < NEAR_TERMINAL_PASS:statuses.append('WARN');reasons.append(f'{name}: near-terminal sign agreement {s["agree"]:.3f} < {NEAR_TERMINAL_PASS}')
        else:statuses.append('PASS')
        for seat in (0, 1):
            t = sources['pooled']['seats'][f'seat{seat}'][name]
            if t['parents'] < MIN_ROWS:continue
            upper = wilson_upper(t['agree'], t['parents'])
            if upper <= NEAR_TERMINAL_FAIL:statuses.append('FAIL');reasons.append(f'{name}: perspective suspect at seat{seat}: near-terminal sign agreement {t["agree"]:.3f} '
                f'(99% upper {upper:.3f} <= {NEAR_TERMINAL_FAIL}, {t["parents"]} parents)')
            elif t['agree'] < NEAR_TERMINAL_PASS:statuses.append('WARN');reasons.append(f'{name}: seat{seat} near-terminal sign agreement {t["agree"]:.3f} < {NEAR_TERMINAL_PASS}')
    return dict(status=worst(statuses), reasons=reasons, sources=sources, rule=THRESHOLDS['perspective']['near_terminal_rule'])


def scale_check(records, population):
    """Check 4: |v| at roots, turn-1 cells and leaves for the champion (each pricer, the raw forward) and the V2 head on the same rows,
    and the champion over every valued f32 frame position (a uniform sample per call)."""
    levels, reasons = {}, []
    for label in UNIFORM:
        if label not in records:continue
        r = records[label];entry = {name: magnitude(r[f'value_{name}'][r[f'valued_{name}']]) for name in NAMES}
        entry['raw'] = magnitude(r['raw_value'][r['walk_ok']]);entry['v2'] = magnitude(r['v2_value'][r['v2_ok']])
        c, v = entry[REFERENCE], entry['v2']
        if c.get('count', 0) >= MIN_ROWS and v.get('count', 0) >= MIN_ROWS and v['p50'] > 0:
            entry['p50_ratio'] = c['p50'] / v['p50']
            if not SCALE_RATIO[0] <= entry['p50_ratio'] <= SCALE_RATIO[1]:reasons.append(f'{label}: champion/V2 p50 |v| ratio {entry["p50_ratio"]:.3f} outside {SCALE_RATIO}')
        if c.get('count', 0) >= MIN_ROWS and c['saturated'] > SATURATION_WARN:reasons.append(f'{label}: {c["saturated"]:.1%} of champion values have |v| > {SATURATED}')
        levels[label] = entry
    frames = {k: magnitude(v) for k, v in population.items()}
    return dict(status='WARN' if reasons else 'PASS', reasons=reasons, levels=levels, frame_population=frames, rule=THRESHOLDS['scale']['rule'])


def rollup(checks):
    """(status, reasons) of a run (THRESHOLDS['rollup']): FAIL, INCONCLUSIVE when the sign or the adapter was not established, WARN, PASS."""
    p = checks['perspective'];statuses = {k: checks[k]['status'] for k in ('precision', 'adapter', 'scale')}
    statuses.update({f'perspective.{k}': p[k]['status'] for k in ('v2', 'seats', 'near_terminal')})
    if 'FAIL' in statuses.values():return 'FAIL', [f'{k} FAIL' for k, v in statuses.items() if v == 'FAIL']
    unestablished = []
    if {p['v2']['status'], p['near_terminal']['status']} <= {'SKIPPED', 'INCONCLUSIVE'}:
        unestablished.append('sign not established: neither the V2 check nor the near-terminal check reached a verdict')
    if checks['adapter']['status'] == 'INCONCLUSIVE':unestablished.append('adapter not established: no f32-valued position compared')
    if unestablished:return 'INCONCLUSIVE', unestablished
    rest = [f'{k} {v}' for k, v in statuses.items() if v != 'PASS']
    return ('WARN', rest) if rest else ('PASS', [])


def judge(calls, expected=()):
    """The four checks over every call, and the run's status, passed and exit code (THRESHOLDS['rollup']); `expected` the priced levels."""
    records, terminal = merge(calls, 'records'), merge(calls, 'terminal')
    if not records:
        empty = lambda: dict(status='INCONCLUSIVE', reasons=['no root was validated'])
        checks = dict(precision=empty(), adapter=empty(), perspective=dict(status='INCONCLUSIVE', v2=empty(), seats=empty(), near_terminal=empty()), scale=empty())
        return dict(status='INCONCLUSIVE', passed=False, exit=EXIT['INCONCLUSIVE'], status_reasons=['no root was validated'], checks=checks,
            calls=[dict(call=c['call'], roots=c['roots']) for c in calls])
    population = {k: np.concatenate([c['population'][k] for c in calls if k in c['population']]) for k in sorted({k for c in calls for k in c['population']})}
    perspective = dict(v2=v2_check(records), seats=seats_check(records), near_terminal=near_terminal_check(terminal))
    perspective['status'] = worst(x['status'] for x in perspective.values())
    checks = dict(precision=precision_check(calls), adapter=adapter_check(records, expected), perspective=perspective, scale=scale_check(records, population))
    status, reasons = rollup(checks)
    summary = [dict(call=c['call'], roots=c['roots'], rows={k: len(v['key']) for k, v in c['records'].items()}, empty_levels=c['empty'], rebound_equal=c['rebound_equal'],
        costs=c['costs']) for c in calls]
    return dict(status=status, passed=status in ('PASS', 'WARN'), exit=EXIT[status], status_reasons=reasons, checks=checks, calls=summary)


# Driving one run ------------------------------------------------------------------------------------------------------------------
def address(request):
    a = request.plans.root.address;return (int(a.stream), int(a.frame), int(a.owner))


def bind(groups, evaluators):
    """g2.bind_evaluator_roots under each evaluator (its exact first-root check, its root prior, actors 0/0), then the roots every evaluator
    kept, paired in the loaded group structure. Returns ({name: [[Request]]}, {name: dropped addresses}, {name: root |v| list})."""
    bound, dropped, values = {}, {}, {}
    for name, ev in evaluators.items():
        kept, v, d, _ = g2.bind_evaluator_roots(groups, (ev.params,), ev);bound[name] = {address(r): r for g in kept for r in g};dropped[name], values[name] = d, v
    out = {name: [] for name in evaluators}
    for group in groups:
        shared = [address(r) for r in group if all(address(r) in bound[n] for n in evaluators)]
        if shared:
            for name in evaluators:out[name].append([bound[name][a] for a in shared])
    return out, dropped, values


def build_pricers(module, executor, model, evaluators, *, capacity, block_lanes=2187, depth3_k=3, fixed_shapes=False):
    """{(family, name): DepthNPricer}: depth 2 at full width (the gate's depth2_pass_pricer) and depth 3 prior top-depth3_k, digests on,
    one instance (and so one program table) per evaluator."""
    out = {}
    for name, ev in evaluators.items():
        out[('depth2', name)] = gate.depth2_pass_pricer(module, executor, model, capacity=capacity, actor_capacity=1, block_lanes=block_lanes, digest=True,
            fixed_capacity=fixed_shapes, forward=ev)
    if depth3_k:
        for name, ev in evaluators.items():
            out[('depth3', name)] = module.DepthNPricer(executor, model, capacity=capacity, depth=3, k_own=depth3_k, k_opp=depth3_k, block_lanes=block_lanes,
                selection='prior', digest=True, actor_capacity=1, **gate.fixed_kwargs(fixed_shapes), forward=ev)
    return out


def validate_call(call, requests, *, pricers, evaluators, materializer, recipe, walker, heads, backups, sample, seed, stack_rows=None, progress=None):
    """One request list: every pricer, the roots' adapter outputs, the precision comparison, the stack of materialized worlds and the walks."""
    from . import depth2_oracle as o2, device_public_teacher_batch as batch
    context = dict(call=call, roots=len(requests[REFERENCE]));frames, costs = {}, {}
    for (family, name), pricer in pricers.items():
        with stage(progress, 'price', family=family, precision=name, **context) as info:
            clock = time.perf_counter();frames[(family, name)] = pricer.price(requests[name], materializer, recipe, (evaluators[name].params,))
            costs[f'{family}.{name}'] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(frames[(family, name)], 'timing', None)))
            info.update(costs[f'{family}.{name}'])
    families = tuple(dict.fromkeys(f for f, _ in pricers))
    with stage(progress, 'root_outputs', **context):roots = root_outputs(requests, evaluators)
    with stage(progress, 'paths', **context):pf = {(family, name): paths(frames[(family, name)]) for family in families for name in NAMES}
    with stage(progress, 'precision', **context):precision = precision_call(frames, pf, requests, roots, families, backups)
    with stage(progress, 'walk', **context) as info:
        stack = o2.stack(requests[REFERENCE], batch._numpy_worlds(requests[REFERENCE], materializer, recipe), size=stack_rows)
        records, terminal, population, empty = call_records(call, frames, pf, requests, roots, stack, walker, heads, families=families, sample=sample,
            rng=np.random.default_rng([seed, call]))
        info.update(rows={k: len(v['key']) for k, v in records.items()}, seconds_walking=walker.seconds)
    return dict(call=call, roots=len(requests[REFERENCE]), precision=precision, records=records, terminal=terminal, population=population, empty=empty, costs=costs,
        rebound_equal={n: roots[n]['rebound_equal'] for n in NAMES})


def check_pair(evaluators, *, tested='bfloat16'):
    """The reference and the evaluator under test are ONE champion at two precisions: the same evaluator type, config, archive and checkpoint
    sha256 and parameters (the same tree, or bit-equal arrays), precisions float32 and `tested` (bfloat16; float32 only for the f32-vs-f32
    self-test). Otherwise a bf16-only change of slice, sign or view would be judged against an f32 reference it does not share: ValueError."""
    a, b = evaluators[REFERENCE], evaluators[TESTED]
    if type(a) is not type(b):raise ValueError(f'one evaluator type at two precisions required (got {type(a).__name__} and {type(b).__name__})')
    if (a.precision, b.precision) != ('float32', tested):raise ValueError(f'precisions float32 (reference) and {tested} (under test) required, got {a.precision} and {b.precision}')
    if a.config != b.config or a.archive_sha256 != b.archive_sha256 or a.checkpoint_sha256 != b.checkpoint_sha256:
        raise ValueError('the two evaluators must share one archive (config, archive and checkpoint sha256)')
    if a.params is not b.params and (set(a.params) != set(b.params) or any(not np.array_equal(np.asarray(a.params[k]), np.asarray(b.params[k])) for k in a.params)):
        raise ValueError('the two evaluators must share one parameter set')
    return True


def expected_levels(pricers):
    """Every priced level's label: the root and each frame family's plies (turn1, ply2.., leaf)."""
    return ['root'] + [f'{family}.{level_name(q, p.depth)}' for (family, name), p in pricers.items() if name == REFERENCE for q in range(1, p.depth + 1)]


def validate(requests, *, evaluators, executor, model, materializer, recipe, observe, reference=None, module=None, backups=None, capacity=512,
        block_lanes=2187, depth3_k=3, sample=SAMPLE, seed=SEED, chunk=CHUNK, fixed_shapes=False, progress=None, tested_precision='bfloat16'):
    """requests {name: [[Request]]} (bind's paired groups); evaluators {'f32': reference, 'bf16': under test}, one champion at two precisions
    (check_pair; tested_precision='float32' only for the f32-vs-f32 self-test); observe(state, public, seat, mask) the champion-layout leaf
    (champion_observer); reference the V2 head (load_reference) or None. Prices, walks and judges every call (groups split at `capacity`);
    returns judge's report body."""
    if set(evaluators) != set(NAMES) or set(requests) != set(NAMES):raise ValueError(f'evaluators and requests named {NAMES} required')
    check_pair(evaluators, tested=tested_precision)
    module = module or gate.pricer_module();backups = backups or importlib.import_module('gpu_engine.depthn_backups')
    pricers = build_pricers(module, executor, model, evaluators, capacity=capacity, block_lanes=block_lanes, depth3_k=depth3_k, fixed_shapes=fixed_shapes)
    split = {name: g2.split_groups(groups, capacity) for name, groups in requests.items()}
    walker = Walker(executor, observe, chunk=chunk, view=ce.digest_view(evaluators[REFERENCE]));heads = Heads(evaluators, reference)
    stack_rows = importlib.import_module('gpu_engine.depth2_oracle').bucket(2 * capacity) if fixed_shapes else None
    calls = [validate_call(i, {name: tuple(split[name][i]) for name in NAMES}, pricers=pricers, evaluators=evaluators, materializer=materializer, recipe=recipe,
        walker=walker, heads=heads, backups=backups, sample=sample, seed=seed, stack_rows=stack_rows, progress=progress) for i in range(len(split[REFERENCE]))]
    report = judge(calls, expected_levels(pricers))
    report['pricers'] = {f'{family}.{name}': dict(depth=p.depth, k_own=list(p.k_own), k_opp=list(p.k_opp), capacity=p.capacity, block_lanes=p.block_lanes,
        fixed_capacity=p.fixed_capacity, evaluator=ce.describe(p.forward)) for (family, name), p in pricers.items()}
    report['walk'] = dict(rows=walker.rows, seconds=walker.seconds, chunk=chunk, stack_rows=stack_rows)
    return report


def champion_observer(loaded):
    """The ladder's leaf observation (phase_d_search.Search.observe; phase_d_ladder_search): device_public_leaf.from_public(...,
    legacy_champion=True) over the bank's observation tables, the champion's 23-token layout straight from engine_device_observation.
    champion_inputs, never through the V2 actor_inputs tensors or ChampionEvaluator.view."""
    from . import device_public_leaf as leaf
    tables = loaded.provider.prepared.observation_tables
    return lambda state, public, seat, mask: leaf.from_public(state, public, tables, observer_sides=seat, legal_action_mask=mask, enable_device_leaf=True,
        legacy_champion=True)


def load_reference(args, loaded):
    """--head-archive: the V2 head through g2.load_head (U1's load and checks against the bank declaration), evaluated on its own arm in
    bf16 (the raw-head path); None without it."""
    if not args.head_archive:return None
    import jax
    from . import v5_model
    params, config, meta, model_sha256 = g2.load_head(args, plan=loaded.plan, labels=loaded.labels, category_sha256=loaded.category_sha256)
    return SimpleNamespace(params=params, forward=jax.jit(lambda p, x: v5_model.forward(p, x, config, compute_dtype='bfloat16')),
        header=dict(archive=args.head_archive, archive_sha256=args.head_archive_sha256, checkpoint_sha256=args.head_checkpoint_sha256, model_sha256=model_sha256,
            arm=config.arm, precision='bfloat16', parameter_count=meta.get('parameter_count'), role='reference critic for check 3 and 4 (never prices)'))


# Command line ---------------------------------------------------------------------------------------------------------------------
def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--saved-roots', help='saved D3 cost bank attempt directory (as d3_depthn_gate)')
    parser.add_argument('--index-sha256', help='sha256 of bank/cost-root-index.json (with --saved-roots)')
    parser.add_argument('--d3-root', help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, help='bank chunk index (with --d3-root)')
    parser.add_argument('--memory-policy', action='store_true', help="with --d3-root: apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory (validation.json, validation.md)')
    parser.add_argument('--capacity', type=int, default=512, help='DepthNPricer root capacity (calls split at it)')
    parser.add_argument('--block-lanes', type=int, default=2187, help='DepthNPricer block_lanes')
    parser.add_argument('--depth3-k', type=int, default=3, help='depth-3 prior top-k per seat and ply (0: depth 2 only)')
    parser.add_argument('--sample', type=int, default=SAMPLE, help='walked rows per sampled level and call (every root world is walked)')
    parser.add_argument('--walk-chunk', type=int, default=CHUNK, help='walk rows per jitted vmap invocation')
    parser.add_argument('--seed', type=int, default=SEED, help='sampling seed')
    parser.add_argument('--max-roots', type=int, help='validate only the first N roots')
    parser.add_argument('--fixed-shapes', action='store_true', help='fixed_capacity pricers and a fixed stack size: compile once per process')
    ce.add_arguments(parser);g2.add_head_arguments(parser)
    args = parser.parse_args(argv)
    if bool(args.saved_roots) == bool(args.d3_root):parser.error('exactly one of --saved-roots or --d3-root required')
    if args.saved_roots and not args.index_sha256:parser.error('--saved-roots requires --index-sha256')
    if args.d3_root and (args.chunk is None or args.chunk < 0):parser.error('--d3-root requires --chunk N >= 0')
    if not all(getattr(args, name) for name in ce.FLAGS):parser.error('--champion-archive, --champion-archive-sha256 and --champion-checkpoint-sha256 are required')
    for name in ce.FLAGS[1:]:
        if not ce._hex64(getattr(args, name)):parser.error(f'--{name.replace("_", "-")} must be a 64-hex sha256')
    if args.champion_precision is not None:parser.error('--champion-precision does not apply: this tool prices with both precisions (bf16 and f32)')
    if not args.head_archive:parser.error('--head-archive (the V2 reference for the sign check) is required, with --head-archive-sha256 and --head-checkpoint-sha256')
    g2.check_head_arguments(parser, args)
    for name in ('head_archive_sha256', 'head_checkpoint_sha256'):
        if not ce._hex64(getattr(args, name) or ''):parser.error(f'--{name.replace("_", "-")} must be a 64-hex sha256')
    if any(x < 1 for x in (args.capacity, args.block_lanes, args.sample, args.walk_chunk)):parser.error('positive sizes required')
    if not 0 <= args.depth3_k <= 9:parser.error('--depth3-k in 0..9')
    if args.max_roots is not None and args.max_roots < 1:parser.error('positive --max-roots required')
    return args


def plain(x):
    """JSON-safe: numpy scalars and arrays to Python, non-finite floats to None."""
    if isinstance(x, dict):return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):return [plain(v) for v in x]
    if isinstance(x, np.ndarray):return plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):return bool(x)
    if isinstance(x, np.integer):return int(x)
    if isinstance(x, (float, np.floating)):return float(x) if math.isfinite(float(x)) else None
    return x


def _num(x, fmt='.3g'):
    return '-' if x is None else format(x, fmt) if isinstance(x, (int, float)) else str(x)


def markdown(report):
    """The run's summary: the verdict table, then each check's key numbers (validation.json holds everything). Every section reads what the
    report has, so an empty or INCONCLUSIVE run still renders."""
    c, h = report.get('checks') or {}, report.get('header') or {};p = c.get('perspective') or {};get = lambda d, k: (d or {}).get(k) or {}
    champ = get(h.get('evaluators'), REFERENCE);ref = h.get('reference')
    lines = [f'# Champion 9375 leaf validation: {report.get("status")} (exit {report.get("exit")})', '']
    lines += [f'- Status reasons: {"; ".join(report.get("status_reasons") or []) or "-"}',
        f'- Source: `{json.dumps(h.get("source"), default=str)}`; roots validated {h.get("roots")} in {len(report.get("calls") or [])} call(s); dropped at '
        f'binding { {k: len(v) for k, v in (h.get("dropped") or {}).items()} }', f'- Champion archive `{champ.get("archive_sha256")}` (checkpoint '
        f'`{champ.get("checkpoint_sha256")}`); reference evaluator f32, under test bf16',
        '- V2 reference: ' + (f'`{ref.get("archive")}` ({ref.get("arm")} arm, bf16)' if ref else 'none'), '']
    lines += ['| Check | Status | Reasons |', '| --- | --- | --- |']
    rows = [('1 bf16 vs f32', c.get('precision')), ('2 adapter vs raw', c.get('adapter')), ('3a sign vs V2', p.get('v2')), ('3b seats', p.get('seats')),
        ('3c near-terminal', p.get('near_terminal')), ('4 scale', c.get('scale'))]
    lines += [f'| {name} | {(x or {}).get("status")} | {"; ".join((x or {}).get("reasons") or []) or "-"} |' for name, x in rows] + ['']
    pr = c.get('precision') or {}
    lines += ['## 1. bf16 vs f32', '', '| Quantity | n | max | p99 | p99.9 | p99.99 |', '| --- | --- | --- | --- | --- | --- |']
    for part in ('values', 'logits'):
        for k, v in (pr.get(part) or {}).items():
            lines.append(f'| {part[:-1]} {k} | {v.get("count")} | {_num(v.get("max"))} | {_num(v.get("p99"))} | {_num(v.get("p99.9"))} | {_num(v.get("p99.99"))} |')
    lines += ['', '| Sign disagreements (f32 \\|v\\| >= 0.2) | confident | disagree | share |', '| --- | --- | --- | --- |']
    lines += [f'| {k} | {v.get("confident")} | {v.get("disagree")} | {_num(v.get("share"))} |' for k, v in (pr.get('sign_disagreements') or {}).items()]
    lines += ['', '| Argmax | nodes | changed | share |', '| --- | --- | --- | --- |']
    lines += [f'| {k} | {v.get("nodes")} | {v.get("changed")} | {_num(v.get("share"))} |' for k, v in (pr.get('argmax_changes') or {}).items()]
    lines += ['', '| Root decision | roots | labeled both | agree | share |', '| --- | --- | --- | --- | --- |']
    lines += [f'| {k} | {v.get("roots")} | {v.get("labeled_both")} | {v.get("agree")} | {_num(v.get("share"), ".4f")} |' for k, v in (pr.get('decisions') or {}).items()]
    ad = c.get('adapter') or {};o = ad.get('observed') or {}
    lines += ['', '## 2. Adapter vs the independent raw forward (f32 HIGHEST)', '',
        f'- Observed f32 max |dv| {_num(o.get("f32_value_max"))} (bound {_num(o.get("f32_value_bound"))}); f32 max |dlogit| {_num(o.get("f32_logit_max"))} '
        f'(bound {_num(o.get("f32_logit_bound"))}); bf16 max |dv| vs the f32 raw forward {_num(o.get("bf16_value_max"))} (WARN {_num(o.get("bf16_value_warn"))}, '
        f'FAIL {_num(o.get("bf16_value_gross"))})', '',
        '| Group | walked | evaluator | inputs unequal | compared | value max | own logits max | opp logits max | digest mismatch | walk disagrees |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for k, g in (ad.get('groups') or {}).items():
        for name in NAMES:
            e = g.get(name) or {};lines.append(f'| {k} | {g.get("walked")} | {name} | {e.get("inputs_unequal")} | {e.get("compared")} | {_num(get(e, "value").get("max"))} | '
                f'{_num(get(e, "own_logits").get("max"))} | {_num(get(e, "opp_logits").get("max"))} | {_num(e.get("digest_mismatch"))} | {e.get("walk_disagrees")} |')
    lines += ['', '## 3. Sign and perspective', '']
    if get(p, 'v2').get('by_evaluator'):
        lines += ['| Evaluator | group | n | Pearson | Spearman | same-sign | confident same-sign (n) | slope |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for name, e in p['v2']['by_evaluator'].items():
            for k, g in e['groups'].items():
                if g.get('count', 0) and (k.count('.') == 0 or k.count('.') == 1 and not k.endswith(('seat0', 'seat1'))):
                    lines.append(f'| {name} | {k} | {g["count"]} | {_num(g.get("pearson"))} | {_num(g.get("spearman"))} | {_num(g.get("same_sign"))} | '
                        f'{_num(g.get("same_sign_confident"))} ({g.get("confident", 0)}) | {_num(g.get("slope"))} |')
    seats = get(p, 'seats');lines += ['', f'- Owner vs opponent view (champion): Pearson {_num(get(seats, "pooled").get("pearson"))} over {get(seats, "pooled").get("count")} '
        'rows; ' + '; '.join(f'{n} pricer value nearer the owner\'s view than the opponent\'s: share {_num(get(seats, f"owner_view_{n}").get("owner_closer"))} of '
            f'{get(seats, f"owner_view_{n}").get("rows")} rows differing by > {_num(get(seats, f"owner_view_{n}").get("margin"))}' for n in NAMES)]
    pooled = get(get(p, 'near_terminal').get('sources'), 'pooled')
    lines += [f'- Near-terminal unanimous parents {pooled.get("unanimous")}: ' + '; '.join(f'{n} agree {_num(get(pooled, n).get("agree"))} (pairs '
        f'{_num(get(pooled, n).get("pair_agree"))})' for n in NAMES + ('v2',))
        + ''.join(f'; {seat} ' + ', '.join(f'{n} {_num(get(x, n).get("agree"))} of {get(x, n).get("parents")}' for n in NAMES) for seat, x in (pooled.get('seats') or {}).items())]
    sc = c.get('scale') or {}
    lines += ['', '## 4. Value scale (|v|)', '', '| Level | critic | n | p50 | p90 | p99 | share > 0.95 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for k, e in (sc.get('levels') or {}).items():
        for name in NAMES + ('raw', 'v2'):
            m = e.get(name) or {};lines.append(f'| {k} | {name} | {m.get("count")} | {_num(m.get("p50"))} | {_num(m.get("p90"))} | {_num(m.get("p99"))} | {_num(m.get("saturated"))} |')
    for k, m in (sc.get('frame_population') or {}).items():
        lines.append(f'| {k} (frame) | f32 | {m.get("count")} | {_num(m.get("p50"))} | {_num(m.get("p90"))} | {_num(m.get("p99"))} | {_num(m.get("saturated"))} |')
    lines += ['', 'Thresholds and rules: `validation.json` header.thresholds (declared in gpu_engine/champion_leaf_validation.py before the run).', '']
    return '\n'.join(lines)


def write(out, report):
    out = Path(out);(out / 'validation.json').write_text(json.dumps(plain(report), indent=1, allow_nan=False) + '\n');(out / 'validation.md').write_text(markdown(plain(report)))


def main(argv=None):
    args = parse(argv);out = Path(args.out);out.mkdir(parents=True, exist_ok=False);started = time.perf_counter()
    log('THRESHOLDS', thresholds=THRESHOLDS)
    needed = ['gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups']
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:raise SystemExit('not on this tree (failing closed): ' + ', '.join(missing))
    loaded = (g2.saved_root_requests if args.saved_roots else g2.bank_chunk_requests)(args)
    args.champion_precision = 'bf16';bf16 = g2.load_champion(args, loaded);evaluators = {REFERENCE: replace(bf16, precision='float32'), TESTED: bf16}
    reference = load_reference(args, loaded)
    log('EVALUATORS', **{name: ev.header()['identity'] for name, ev in evaluators.items()}, reference=None if reference is None else reference.header)
    requests, dropped, root_values = bind(loaded.groups, evaluators)
    if args.max_roots is not None:
        for name, groups in requests.items():
            kept, left = [], args.max_roots
            for group in groups:
                if left > 0:kept.append(group[:left]);left -= len(kept[-1])
            requests[name] = kept
    roots = sum(map(len, requests[REFERENCE]));log('ROOTS', roots=roots, dropped={k: len(v) for k, v in dropped.items()})
    report = validate(requests, evaluators=evaluators, executor=loaded.provider.executor, model=loaded.plan.model, materializer=loaded.provider.materializer,
        recipe=loaded.provider.recipe, observe=champion_observer(loaded), reference=reference, capacity=args.capacity, block_lanes=args.block_lanes,
        depth3_k=args.depth3_k, sample=args.sample, seed=args.seed, chunk=args.walk_chunk, fixed_shapes=args.fixed_shapes, progress=log)
    import jax
    report = dict(schema=SCHEMA, status=report.pop('status'), passed=report.pop('passed'), exit=report.pop('exit'), status_reasons=report.pop('status_reasons'), header=dict(thresholds=THRESHOLDS, source=loaded.source, roots=roots,
        dropped=dropped, root_abs_value={name: g2.percentiles(v) for name, v in root_values.items()},
        evaluators={name: dict(ev.header(), archive=args.champion_archive) for name, ev in evaluators.items()},
        reference=None if reference is None else reference.header, options=dict(capacity=args.capacity, block_lanes=args.block_lanes, depth3_k=args.depth3_k,
            sample=args.sample, walk_chunk=args.walk_chunk, seed=args.seed, max_roots=args.max_roots, fixed_shapes=args.fixed_shapes),
        devices=[str(d) for d in jax.devices()], seconds=time.perf_counter() - started), **report)
    write(out, report)
    log('CHAMPION_LEAF_VALIDATION_COMPLETE', status=report['status'], exit=report['exit'], status_reasons=report['status_reasons'], out=str(out / 'validation.json'),
        checks=dict(precision=report['checks']['precision']['status'], adapter=report['checks']['adapter']['status'],
            **{f'perspective.{k}': report['checks']['perspective'][k]['status'] for k in ('v2', 'seats', 'near_terminal')}, scale=report['checks']['scale']['status']))
    return report['exit']


if __name__ == '__main__':raise SystemExit(main())
