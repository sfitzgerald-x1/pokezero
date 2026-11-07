"""Gap 2, the last measurement round: every float surface the champion gate's three runs gate, bf16 against f32 on identical requests at
exactly the positions each check compares, the per-site table and the declared-bounds proposal (informational, never a verdict).
Numpy-only import; JAX loads after the bank (as the gates).

The runs judged (RUN_ARGV, parsed by d3_depthn_gate's own parser; every pricer here is built from the parsed arguments exactly as run_gate /
run_n2 build it, and every departure from the pinned shapes (PINNED) is flagged):
 R1, the gate: --depth 2 --k-own 9 --k-opp 9 --k1-own 9 --k1-opp 9 --fixed-shapes --capacity 512 --block-lanes 2187 --oracle-chunk 512, with or
    without --block-invariance (block_lanes INVARIANCE_LANES, 729 against the checked 2187; decided here: R1_INVARIANCE_RULE);
 R2: --n2-equivalence --block-invariance at its defaults (--fixed-shapes is refused there: every pricer prices at per-call buckets): per config
    (full width, top --n2-k) DepthNPricer N=2 against the merged Depth2Pricer at child_block 27 and 81 (81 the compared one);
 R3, the kill test: --depth 3 --selection prior --k-own 3 --k-opp 3 --mutation-test --fixed-shapes --capacity 512 --block-lanes 2187 on the 40
    saved roots, measured here on chunk 0 (its fixed-shape pricers and oracle price any roots with the same programs; D3, whose (b) programs
    are per-call buckets, is priced at R3's own bucket: calls of --r3-call-roots 40, bucket 64).
Setup as champion_d3_geometry: --d3-root DIR --chunk N through d3_depth2_gate.bank_chunk_requests, the champion through g2.load_champion at
bf16 (under test) and f32 (the reference; champion_leaf_validation.check_pair), the roots bound ONCE under the bf16 champion as the gate binds
them (g2.bind_roots), both precisions pricing these identical requests, calls split at 512 as every run splits them.

Measurement groups (--measure; GROUPS): r1 (R1's frame, which is also R1's and R3's full-width depth-2 pass), r1-invariance (its 729-lane twin),
r3 (the depth-3 frame), r3-d3 (D3 at R3's bucket), r2-full and r2-top (each R2 config's DepthNPricer N=2 and Depth2Pricer at 27 and 81). Each
geometry is priced at both precisions per call and measured at the gate's own masks (POSITIONS_RULE), whole arrays where block invariance reads
them, completed Q through the gate's own (e) (Q_RULE) and the guard's spreads (GUARD_RULE); the bf16 site itself is measured directly where both
of its geometries are priced here (DIRECT_RULE), and the f32 premises of the triangle checked (PREMISE_RULE). Separate runs (pieces) fold
together offline: --fold-from DIR... (FOLD_RULE). The fold (SITES_RULE, CONSTRUCTIONS) builds one row per float site the three runs gate,
enumerated from the gate's code (site_registry; each row's bound field from the gate's own functions: gate_bound) with each side's tail and
its source (validation.json, d3_geometry.json or this run), and the proposal (PROPOSAL_RULE): each field 1.25 x its largest site tail, rounded
up to two significant figures. gate_surfaces.json and gate_surfaces.md in --out. Exit 0 when anything was measured, 2 when nothing was.
"""
import argparse
import collections
import contextlib
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np

from . import champion_d3_geometry as d3g, champion_evaluator as ce, champion_leaf_validation as clv, d3_depth2_gate as g2, d3_depthn_gate as gate

SCHEMA = 'pokezero.champion-gate-surfaces.v1'
REFERENCE, TESTED, NAMES = clv.REFERENCE, clv.TESTED, clv.NAMES
FIRST = 5
log, stage = g2.log, g2.stage
EXIT = dict(MEASURED=0, NOTHING_COMPARED=2)
MARGIN, FIGURES = 1.25, 2
FIELDS = ('value_bound', 'turn1_logit_bound', 'root_logit_bound', 'deeper_logit_bound', 'near_tie_q_bound')
DRAFT = dict(value_bound=.032, turn1_logit_bound=.63, root_logit_bound=.26, deeper_logit_bound=.87, near_tie_q_bound=.032)
SENTINEL = dict(value_bound=1., turn1_logit_bound=2., root_logit_bound=3., deeper_logit_bound=4., near_tie_q_bound=5.)   # field discovery only
GROUPS = ('r1', 'r1-invariance', 'r3', 'r3-d3', 'r2-full', 'r2-top')
NEEDS = {'r1-invariance': 'r1'}   # a group priced only beside another (its direct site comparison)
R3_ROOTS = 40          # R3's saved roots: one call of 40, D3 bucket 64
SPLIT = 512            # every run's call split: g2.call_capacity(D3's 4096, the pricers' --capacity 512)
ADAPTER = dict(value=clv.ADAPTER_VALUE, logit=clv.ADAPTER_LOGIT)
# Minutes on one GB300 for chunk 0 (4,073 roots, 11 calls), both precisions, compiles included (ESTIMATE_RULE)
ESTIMATE_MINUTES = {'r1': 34, 'r1-invariance': 31, 'r3': 28, 'r3-d3': 12, 'r2-full': 115, 'r2-top': 61, 'load': 4}
TAIL_EDGES = np.geomspace(1e-12, 1e3, 34600)   # |diff| histogram upper edges: ratio 1.001 per bin

# ---------------------------------------------------------------------------------------------------------------- the runs
BANK_ARGV = ['--d3-root', '/nonexistent/d3-root', '--chunk', '0', '--out', '/nonexistent/out']
SAVED_ARGV = ['--saved-roots', '/nonexistent/saved-roots', '--index-sha256', '0' * 64, '--out', '/nonexistent/out']
PARSE_CHAMPION = ['--champion-archive', '/nonexistent/champion.npz', '--champion-archive-sha256', '0' * 64, '--champion-checkpoint-sha256', '0' * 64]
RUN_ARGV = dict(
    R1=['--depth', '2', '--k-own', '9', '--k-opp', '9', '--k1-own', '9', '--k1-opp', '9', '--fixed-shapes', '--capacity', '512', '--block-lanes', '2187',
        '--oracle-chunk', '512'],
    R1_invariance=['--depth', '2', '--k-own', '9', '--k-opp', '9', '--k1-own', '9', '--k1-opp', '9', '--fixed-shapes', '--capacity', '512', '--block-lanes',
        '2187', '--oracle-chunk', '512', '--block-invariance'],
    R2=['--n2-equivalence', '--block-invariance'],
    R3=['--depth', '3', '--selection', 'prior', '--k-own', '3', '--k-opp', '3', '--mutation-test', '--fixed-shapes', '--capacity', '512', '--block-lanes', '2187'])
RUN_ROOTS = dict(R1=BANK_ARGV, R1_invariance=BANK_ARGV, R2=BANK_ARGV, R3=SAVED_ARGV)
PINNED = dict(
    R1=dict(depth=2, k_own_ply=(9,), k_opp_ply=(9,), k1_own=9, k1_opp=9, selection='prior', capacity=512, block_lanes=2187, fixed_shapes=True, oracle_chunk=512,
        block_invariance=False),
    R1_invariance=dict(depth=2, k_own_ply=(9,), k_opp_ply=(9,), k1_own=9, k1_opp=9, selection='prior', capacity=512, block_lanes=2187, fixed_shapes=True,
        oracle_chunk=512, block_invariance=True),
    R2=dict(n2_equivalence=True, block_invariance=True, capacity=512, block_lanes=2187, k1_own=9, k1_opp=9, n2_k=3, fixed_shapes=False),
    R3=dict(depth=3, k_own_ply=(3, 3), k_opp_ply=(3, 3), k1_own=9, k1_opp=9, selection='prior', capacity=512, block_lanes=2187, fixed_shapes=True,
        mutation_test=True, block_invariance=False, sample_leaves=None))
RECORDED_ARGS = ('depth', 'k_own_ply', 'k_opp_ply', 'k1_own', 'k1_opp', 'selection', 'capacity', 'block_lanes', 'fixed_shapes', 'block_invariance',
    'n2_equivalence', 'n2_k', 'oracle_chunk', 'depth2_pass_sample', 'mutation_test', 'sample_leaves', 'worlds_blocks', 'max_roots')

# ---------------------------------------------------------------------------------------------------------------- the rules
POSITIONS_RULE = ('Each geometry is priced at bf16 and f32 on the same requests and |bf16 - f32| taken at the gate\'s own masks, built on the bf16 '
    'frame (the run\'s) where the f32 frame agrees on every discrete array that defines the position (a disagreement is a structural difference: '
    'counted by kind, never dropped, and it flags every site it touches). Turn 1 (g2.check_heads, depth2_pass_check, n2_compare): t1_value at '
    'valued cells (status 2, or 3 with codes 5/6), t1_own_logits on the requested learner\'s legal lanes of valued cells, t1_peer_logits on the '
    'requested opponent\'s legal lanes of expanded cells, root_peer_logits on the requested opponent\'s legal lanes of usable worlds (pre 0), and '
    'root_peer_logits.all_lanes on all nine lanes of usable worlds ((b) reads all nine lanes of every world D3 priced; a D3-priced world is usable). '
    'Plies 2..N-1 are slot-indexed through each parent\'s selection maps, which `prior` orders by logit, so the f32 frame is first aligned to '
    'the bf16 frame\'s slots by action path (align: a near-tie orders the same children differently in the two precisions; at top-k it may '
    'select other children, which are counted as *_selection_differs and never compared). '
    'Plies 2..N-1 (check_heads_n, _node_valued): ply{p}_value at valued nodes (status 2 or codes 5/6), ply{p}_own_logits on the requested learner\'s '
    'legal lanes of valued nodes, ply{p}_opp_logits on the requested opponent\'s legal lanes of expanded nodes. Leaves: leaf_value / t2_value at '
    'status 4 (head-valued). (c) compares where the ORACLE agrees on status, which (a) makes the pricer\'s own: these masks are the gate\'s positions '
    'when (a) passes, a superset otherwise. Whole arrays (block invariance): every float array g2.block_invariance compares (gate.frame_arrays for '
    'the depth-N frame, the Depth2Frame\'s fields), every element; a position non-finite in both precisions is padding (counted, excluded from the '
    'tail; the gate\'s own bf16 comparison, DIRECT_RULE, decides whether its bits match there). |diff| follows g2._numeric: bit-equal 0, both finite '
    '|x - y|, else non-finite')
Q_RULE = ('(e)\'s completed Q: gate.decide_roots(bf16 frame, f32 frame, requests, backups) with the backups the site uses (depthn_backups for R1 '
    'and R3, depth2_backups on the Depth2Frame / as_depth2_frame for R2), its learner_max_root calls recorded: |Q_bf16 - Q_f32| over the legal '
    'actions of every root labeled in both and finite (decide_roots\' own max |dQ| cross-checked); the bf16 top-two gaps are decide_roots\' '
    'gap_pricer. Every root counts, including roots a structural difference touches (a top-k selection that differs between the precisions '
    'changes the children backed up: conservative, and the site is flagged); q.same_selection repeats the tail over the other roots. '
    'The Q adapter bound (|Q_f32 - Q_f32 raw|) is Q_ADAPTER(depth) = 1e-5 + depth x (exp(2e-4) - 1): every backup is 1-Lipschitz '
    'in the values (max, convex averages, clip) and each of the depth prior layers (the root reply prior, then each ply\'s opponent prior; '
    'learner-max reads no learner prior) is a softmax of logits within 1e-4, which moves a weighted average of values in [-1, 1] by at most '
    'exp(2 x 1e-4) - 1')
GUARD_RULE = ('The misplacement guard as the gate computes it: gate.guard_spread over gate.node_spreads of every call\'s checked bf16 frame (the '
    'p10 over NON-ZERO last-ply own-slot spreads; R2 per config over its N=2 frames, as summarize_n2), guard = g2.tolerance_rule(None, p10) '
    'guard_bound = 0.25 x p10. Informational (the lead\'s decision: under --declared-bounds the guard is digest-certified where every certifying '
    'digest matches, gated where any does not): the guard, each guarded site\'s tail and its ratio to the guard')
DIRECT_RULE = ('Where both geometries of a site are priced here (R1 block invariance, R2\'s n2 floats, (e) and child-block invariance, R3\'s turn 1 '
    'against the depth-2 frame) the site itself is measured on chunk 0 with the gate\'s own function on the two bf16 frames (g2.block_invariance, '
    'gate.n2_compare, gate.decide_roots): reported beside the construction, and a site tail below it is raised to it (flagged)')
PREMISE_RULE = ('The triangle |A_bf16 - B_bf16| <= (|A_bf16 - A_f32| + adapter) + (|B_bf16 - B_f32| + adapter) assumes each f32 is within the '
    'adapter bound of the f32 raw forward at every compared position; where both f32 frames are priced here their own comparison (the same '
    'function on the two f32 frames) must be within 2 x the adapter bound, else every site built on it is flagged')
CONSTRUCTIONS = dict(
    sum='both sides measured: tail = side A + side B (each its max |bf16 - f32| + the adapter bound, or an earlier report\'s all-positions P)',
    oracle_sample=('the gate oracle measured on a sample only (validation.json O_sample, |oracle bf16 - f32 raw|): tail = max(P + O_sample, 2 x P), '
        'champion_leaf_validation\'s all_positions construction'),
    mirror=('the oracle-heads side of (e) unmeasured (it would need the oracle at both precisions): taken equal to the pricer side, tail = 2 x P, '
        'the all_positions construction\'s 2 x P_all term'),
    derived='(f): every grid entry and row q is a 1-Lipschitz function (max, convex averages under the same priors, clip) of the leaf values (c) '
        'compares at the same leaves, and terminals are exact: the tail is that (c) leaf_value site\'s')
SITES_RULE = ('One row per float site the three runs gate, enumerated from the gate\'s code (site_registry): (b) g2.check_d\'s values; (c) '
    'g2.check_heads at turn 1 (raw-head quantities do not apply to the champion), check_heads_n\'s plies and leaves and depth2_pass_check\'s floats; '
    '(e) decision_check_n / decide_roots; (f) choice_check_n\'s grid and row q; block invariance (gate.frame_arrays\' float arrays); under '
    '--n2-equivalence n2_compare\'s floats, its (e), the child-block invariance and MERGED_DEPTH2_DECISION. The bound field of each is what the '
    'gate applies (gate_bound: declared_bound_for for (c), g2.bound_for under the summarize hook\'s declared_rule plus declared_root_sites for (b) '
    'and block invariance, finalize_n for the n2 floats, the q_noise_bound for every (e), the value bound for (f)), found by running those '
    'functions under a sentinel declaration. Reported-only sites (R2\'s turn-1 heads, the floors, topk_boundary) are listed apart')
FOLD_RULE = ('Each side\'s tail is the largest of its sources (this run: max |bf16 - f32| + the adapter bound; validation.json: P_all, already '
    'with it, or O_sample; d3_geometry.json: P or D_all, with it), the source that sets it named. A source with non-finite differences WITHHOLDS '
    'the side, the site and every field it covers (flagged); a structural difference or a failed premise FLAGS the sites it touches. Pieces '
    '(--fold-from) must share the source, roots, champion and split; a geometry measured in two pieces is refused')
PROPOSAL_RULE = ('PROPOSAL, not a verdict: each field = 1.25 x the largest site tail it covers, rounded UP to two significant figures, the '
    'setting site named; near_tie_q_bound = 1.25 x max(the value tail, the (e) dQ tail). Reported-only and unmeasured sites set nothing (they '
    'become report-only in the declaration). R1\'s block invariance sites enter only when R1_INVARIANCE_RULE keeps --block-invariance')
R1_INVARIANCE_RULE = ('R1 runs with --block-invariance iff, measured here, its discrete arrays are bit-equal between 729 and 2187 lanes at bf16, '
    'its sites are measured, finite and unflagged, and they raise no field of the proposal without them; otherwise without it')
ESTIMATE_RULE = ('chunk 0 on one GB300: per group the pricers\' measured chunk-0 costs (champion_leaf_validation: depth 2 f32 ~990 s, bf16 ~780 s, '
    'depth 3 f32 ~1080 s, bf16 ~470 s, compiles included; the 729-lane twin taken 20% slower), ~3.5 min per new D3 / Depth2Pricer bucket and '
    'precision (champion_d3_geometry: D3 671 s f32 and 539 s bf16 over buckets 8, 32, 512; chunk 0\'s calls make R2 compile buckets 512, 32 '
    'and 8), and the host backups for (e)\'s Q (about 2 us per leaf: ~1.5 min per full-width frame family and precision); the Depth2Pricer has '
    'no GB300 measurement yet: its device time taken as the DepthNPricer\'s at the same leaf count')
SCOPE_RULE = ('R1 and R2 are measured on this chunk\'s call split, as runs on this chunk would price it; R2 prices at per-call buckets, so a '
    'run on other roots compiles other buckets (chunk 0: 512, 32 and 8; the 40 saved roots: 64), unmeasured here. R3 runs on the 40 saved roots: '
    'its fixed-shape pricers (and oracle) price any roots with the same programs, measured here on this chunk\'s roots; its D3 (per-call '
    'buckets) is priced at R3\'s bucket in calls of --r3-call-roots on this chunk\'s roots (r3-d3)')
RULES = dict(scope=SCOPE_RULE, positions=POSITIONS_RULE, q=Q_RULE, guard=GUARD_RULE, direct=DIRECT_RULE, premise=PREMISE_RULE, constructions=CONSTRUCTIONS, sites=SITES_RULE,
    fold=FOLD_RULE, proposal=PROPOSAL_RULE, r1_invariance=R1_INVARIANCE_RULE, estimate=ESTIMATE_RULE, adapter=ADAPTER, margin=MARGIN, figures=FIGURES,
    draft=DRAFT, exit=EXIT)


def q_adapter(depth):
    """The completed-Q adapter bound at a search depth (Q_RULE)."""
    return clv.ADAPTER_VALUE + int(depth) * math.expm1(2 * clv.ADAPTER_LOGIT)


def kind_of(quantity):
    return 'q' if quantity == 'q' else 'logit' if 'logits' in quantity else 'value'


def adapter_for(quantity, depth=2):
    k = kind_of(quantity);return q_adapter(depth) if k == 'q' else ADAPTER[k]


def round_up(x, figures=FIGURES):
    """x rounded UP to `figures` significant figures (decimal, from its shortest repr)."""
    if x is None or not math.isfinite(x):return None
    if x <= 0:return 0.
    d = Decimal(repr(float(x)));return float(d.quantize(Decimal(1).scaleb(d.adjusted() - figures + 1), rounding=ROUND_CEILING))


# ---------------------------------------------------------------------------------------------------------------- |diff| accumulation
def deltas(x, y):
    """|x - y| as g2._numeric takes it (both cast to float32): 0 where bit-equal, |x - y| (float64) where both are finite, inf otherwise."""
    x, y = np.asarray(x, np.float32).reshape(-1), np.asarray(y, np.float32).reshape(-1)
    exact = g2._bits(x) == g2._bits(y);finite = np.isfinite(x) & np.isfinite(y)
    return np.where(exact, 0., np.where(finite, np.abs(x.astype(np.float64) - y.astype(np.float64)), np.inf))


def whole_deltas(x, y):
    """(deltas over the positions not non-finite in both, keep mask, count non-finite in both with unequal bits): whole-array padding."""
    x, y = np.asarray(x, np.float32).reshape(-1), np.asarray(y, np.float32).reshape(-1)
    exact = g2._bits(x) == g2._bits(y);padding = ~exact & ~np.isfinite(x) & ~np.isfinite(y)
    return deltas(x, y)[~padding], ~padding, int(np.count_nonzero(padding))


class Tail:
    """|bf16 - f32| over positions, merged over calls: count, non-finite count, exact max, mean, the exactly-equal share and the tail
    quantiles p99, p99.9, p99.99 from a log histogram (TAIL_EDGES: each is the upper edge of the bin holding the nearest-rank order
    statistic over the finite values, never below it and within 0.1% of it, capped at the max); the largest few with their call and cell."""
    def __init__(self):
        self.count = self.nonfinite = self.exact = 0;self.max = None;self.total = 0.;self.hist = np.zeros(len(TAIL_EDGES) + 1, np.int64);self.largest = []

    def add(self, d, *, call=None, locate=None):
        d = np.asarray(d, np.float64).reshape(-1);finite = np.isfinite(d);f = d[finite]
        self.count += int(d.size);self.nonfinite += int(d.size - f.size)
        if f.size:
            self.exact += int(np.count_nonzero(f == 0));self.total += float(f.sum());top = float(f.max());self.max = top if self.max is None else max(self.max, top)
            self.hist += np.bincount(np.searchsorted(TAIL_EDGES, f), minlength=len(self.hist))
        if d.size:
            k = min(FIRST, d.size);key = np.where(np.isnan(d), np.inf, d);idx = np.argpartition(-key, k - 1)[:k]
            cells = locate(idx) if locate is not None else [[int(i)] for i in idx]
            rows = [dict(call=call, cell=[int(c) for c in cell], abs_difference=float(d[i]) if np.isfinite(d[i]) else 'non-finite') for i, cell in zip(idx, cells)]
            score = lambda r: math.inf if r['abs_difference'] == 'non-finite' else r['abs_difference']
            self.largest = sorted(self.largest + rows, key=lambda r: -score(r))[:FIRST]
        return self

    def quantile(self, p):
        n = self.count - self.nonfinite
        if not n:return None
        rank = max(1, math.ceil(p / 100 * n));i = int(np.searchsorted(np.cumsum(self.hist), rank))
        if self.max == 0 or rank <= self.exact:return 0.   # exact zeros are the smallest values
        return self.max if i >= len(TAIL_EDGES) else min(float(TAIL_EDGES[i]), self.max)

    def summary(self):
        out = dict(count=self.count, nonfinite=self.nonfinite)
        n = self.count - self.nonfinite
        if not n:return dict(out, largest=self.largest)
        return dict(out, max=self.max, **{f'p{t:g}': self.quantile(t) for t in clv.TAILS}, mean=self.total / n, exact=self.exact / n, largest=self.largest)


def _locator(mask):
    """locate(indices) for the masked elements of `mask` (their cells)."""
    def locate(idx):
        flat = np.flatnonzero(mask)[np.asarray(idx, np.int64)];return [list(np.unravel_index(int(i), mask.shape)) for i in flat]
    return locate


# ---------------------------------------------------------------------------------------------------------------- the gate's masks
def _get(frame, path):
    obj = frame
    for key in (path if isinstance(path, tuple) else (path,)):obj = obj[key] if isinstance(key, int) or isinstance(obj, dict) else getattr(obj, key)
    return np.asarray(obj)


def turn1_masks(f, b):
    """({quantity: (field, mask)}, {kind: (mask, affects)}) at turn 1 (POSITIONS_RULE): masks on the bf16 frame b where the f32 frame f agrees."""
    g = lambda frame, name: _get(frame, name);both = lambda name: g(b, name).astype(bool) & g(f, name).astype(bool)
    differ = lambda name: g(b, name).astype(bool) != g(f, name).astype(bool)
    t1s, t1e, fs, fe = g(b, 't1_status'), g(b, 't1_error'), g(f, 't1_status'), g(f, 't1_error');same = (fs == t1s) & (fe == t1e)
    valued = same & ((t1s == 2) | ((t1s == 3) & np.isin(t1e, g2.DEPTH2_CODES)));expanded = same & (t1s == 2)
    masks = dict(t1_value=('t1_value', valued));structural = dict(t1_status_or_error=(((fs != 0) | (t1s != 0)) & ~same, 'all'))
    for seat, scope in (('own', valued), ('peer', expanded)):
        requested, legal = f't1_{seat}_requested', f't1_{seat}_legal'
        masks[f't1_{seat}_logits'] = (f't1_{seat}_logits', (scope & both(requested))[..., None] & both(legal))
        structural[requested] = (scope & differ(requested), (f't1_{seat}_logits', 'q'))
        structural[legal] = ((scope & both(requested))[..., None] & differ(legal), (f't1_{seat}_logits', 'q'))
    pb, pf = g(b, 'pre'), g(f, 'pre');usable = (pb == 0) & (pf == 0)
    masks['root_peer_logits'] = ('root_peer_logits', (usable & both('root_peer_requested'))[..., None] & both('root_peer_legal'))
    masks['root_peer_logits.all_lanes'] = ('root_peer_logits', np.repeat(usable[..., None], 9, axis=-1))
    structural.update(pre=(pb != pf, 'all'), root_peer_requested=(usable & differ('root_peer_requested'), ('root_peer_logits', 'q')),
        root_peer_legal=((usable & both('root_peer_requested'))[..., None] & differ('root_peer_legal'), ('root_peer_logits', 'q')))
    return masks, structural


def t2_masks(f, b):
    """The depth-2 surface's leaves (Depth2Frame or as_depth2_frame): t2_value at status 4 in both; enumeration or status differences."""
    sb, sf, eb, ef = _get(b, 't2_status'), _get(f, 't2_status'), _get(b, 't2_error'), _get(f, 't2_error')
    return dict(t2_value=('t2_value', (sb == 4) & (sf == 4))), dict(t2_status_or_error=(((sb != 0) | (sf != 0)) & ((sb != sf) | (eb != ef)), ('t2_value', 'q')))


NODE_FIELDS = ('status', 'value', 'error', 'own_requested', 'opp_requested', 'own_legal', 'opp_legal', 'own_logits', 'opp_logits', 'own_actions', 'opp_actions',
    'digest_own', 'digest_opp')
LEAF_FIELDS = ('leaf_status', 'leaf_value', 'leaf_error', 'leaf_digest')
UNHELD = -9   # an aligned map slot whose child the f32 frame did not select


def slot_perm(mb, mf):
    """For each bf16 slot of a slot -> action map, the f32 slot holding the same action (-1 if the f32 map has none); padding slots (-1 in
    both) pair with themselves."""
    mb, mf = np.asarray(mb), np.asarray(mf);eq = (mb[..., :, None] == mf[..., None, :]) & (mb[..., :, None] >= 0)
    return np.where(eq.any(axis=-1), eq.argmax(axis=-1), np.where((mb < 0) & (mf < 0), np.arange(mb.shape[-1]), -1))


def align(f, b):
    """(the f32 depth-N frame gathered into the bf16 frame's slot layout, {level: found}). Plies past turn 1 are slot-indexed through each
    parent's selection maps, and `prior` orders slots by logit (a near-tie orders them differently in the two precisions; at top-k it can
    select other children): every node and leaf of the bf16 frame is matched to the f32 frame's same action path. A position the f32 frame
    does not hold (found False) keeps the bf16 frame's arrays in the copy (never compared as a float: POSITIONS_RULE). Turn 1 is
    action-indexed and unchanged; the copy's maps are the f32 maps re-ordered (UNHELD where not found)."""
    R = len(_get(b, 'pre'));shape = _get(b, 't1_own_actions').shape
    own, opp = (_get(b, n).reshape(R, 2, 81, -1) for n in ('t1_own_actions', 't1_opp_actions'));fown, fopp = (_get(f, n).reshape(R, 2, 81, -1) for n in ('t1_own_actions', 't1_opp_actions'))
    po, px = slot_perm(own, fown), slot_perm(opp, fopp);index, ok = np.broadcast_to(np.arange(81), (R, 2, 81)), np.ones((R, 2, 81), bool)
    held = lambda maps, perm, mine: np.where(perm >= 0, np.take_along_axis(maps, np.clip(perm, 0, None), axis=-1), np.where(mine < 0, mine, UNHELD)).astype(np.int32)
    changes = dict(t1_own_actions=held(fown, po, own).reshape(shape), t1_opp_actions=held(fopp, px, opp).reshape(_get(b, 't1_opp_actions').shape));found, nodes = {}, []
    rows, worlds = np.arange(R)[:, None, None], np.arange(2)[None, :, None]
    for p, (lb, lf) in enumerate(zip(b.nodes, f.nodes), start=2):
        kx = px.shape[-1];K = po.shape[-1] * kx
        idx = (index[..., None, None] * K + np.clip(po, 0, None)[..., :, None] * kx + np.clip(px, 0, None)[..., None, :]).reshape(R, 2, -1)
        ok = (ok[..., None, None] & (po >= 0)[..., :, None] & (px >= 0)[..., None, :]).reshape(R, 2, -1);fields = {}
        for name in NODE_FIELDS:
            x, mine = getattr(lf, name, None), getattr(lb, name, None)
            if x is None or mine is None:continue
            got = np.asarray(x)[rows, worlds, idx];keep = ok.reshape(ok.shape + (1,) * (got.ndim - 3))
            fields[name] = np.where(keep, got, np.asarray(mine))
        bo, bx = np.asarray(lb.own_actions), np.asarray(lb.opp_actions);po, px = slot_perm(bo, fields['own_actions']), slot_perm(bx, fields['opp_actions'])
        fields.update(own_actions=held(fields['own_actions'], po, bo), opp_actions=held(fields['opp_actions'], px, bx))
        nodes.append(gate._copy(lf, **fields));found[f'ply{p}'] = ok;index = idx
    n = index[..., None, None];oi, xi = np.clip(po, 0, None)[..., :, None], np.clip(px, 0, None)[..., None, :]
    leaf = ok[..., None, None] & (po >= 0)[..., :, None] & (px >= 0)[..., None, :];found['leaf'] = leaf
    for name in LEAF_FIELDS:
        x, mine = getattr(f, name, None), getattr(b, name, None)
        if x is None or mine is None:continue
        got = np.asarray(x)[rows[..., None, None], worlds[..., None, None], n, oi, xi];changes[name] = np.where(leaf.reshape(leaf.shape + (1,) * (got.ndim - 5)), got, np.asarray(mine))
    return gate._copy(f, nodes=tuple(nodes), **changes), found


def found_arrays(found, arrays):
    """{gate.frame_arrays name: found mask broadcast to its shape} for the aligned levels (leaf_*, ply{p}.*); other arrays are action-indexed."""
    out = {}
    for name, x in arrays.items():
        level = 'leaf' if name.startswith('leaf_') else name.split('.')[0] if name.startswith('ply') and '.' in name else None
        if level in found and np.asarray(x).ndim >= found[level].ndim:
            m = found[level];out[name] = np.broadcast_to(m.reshape(m.shape + (1,) * (np.asarray(x).ndim - m.ndim)), np.asarray(x).shape)
    return out


def depthn_masks(f, b, found=None):
    """A depth-N frame's turn 1 (turn1_masks), every ply 2..N-1 node level and its leaves (check_heads_n's masks), the f32 frame `f` aligned
    to the bf16 frame's slots (align; found its levels' found masks, None: all found): a node or leaf the f32 frame does not hold is counted
    as `ply{p}_selection_differs` / `leaf_selection_differs`, never compared."""
    masks, structural = turn1_masks(f, b);found = found or {}
    held = lambda name: (_get(f, name) != UNHELD).all(axis=-1);t1 = (_get(b, 't1_status') == 2) & (_get(f, 't1_status') == 2)
    structural['t1_selection_differs'] = (t1 & ~(held('t1_own_actions') & held('t1_opp_actions')), 'all')
    for p, (lb, lf) in enumerate(zip(b.nodes, f.nodes), start=2):
        st, er, fs = np.asarray(lb.status), np.asarray(lb.error), np.asarray(lf.status);ok = found.get(f'ply{p}', np.ones(st.shape, bool))
        same = (fs == st) & (np.asarray(lf.error) == er) & ok;valued = same & ((st == 2) | np.isin(er, (5, 6)));expanded = same & (st == 2)
        both = lambda name: np.asarray(getattr(lb, name)).astype(bool) & np.asarray(getattr(lf, name)).astype(bool)
        differ = lambda name: np.asarray(getattr(lb, name)).astype(bool) != np.asarray(getattr(lf, name)).astype(bool)
        masks[f'ply{p}_value'] = (('nodes', p - 2, 'value'), valued);structural[f'ply{p}_selection_differs'] = ((st != 0) & ~ok, 'all')
        structural[f'ply{p}_status_or_error'] = (((st != 0) | (fs != 0)) & ok & ~((fs == st) & (np.asarray(lf.error) == er)), 'all')
        for seat, scope in (('own', valued), ('opp', expanded)):
            masks[f'ply{p}_{seat}_logits'] = (('nodes', p - 2, f'{seat}_logits'), (scope & both(f'{seat}_requested'))[..., None] & both(f'{seat}_legal'))
            structural[f'ply{p}_{seat}_requested'] = (scope & differ(f'{seat}_requested'), (f'ply{p}_{seat}_logits', 'q'))
            structural[f'ply{p}_{seat}_legal'] = ((scope & both(f'{seat}_requested'))[..., None] & differ(f'{seat}_legal'), (f'ply{p}_{seat}_logits', 'q'))
    sb, sf = _get(b, 'leaf_status'), _get(f, 'leaf_status');eb, ef = _get(b, 'leaf_error'), _get(f, 'leaf_error');ok = found.get('leaf', np.ones(sb.shape, bool))
    masks['leaf_value'] = ('leaf_value', (sb == 4) & (sf == 4) & ok);structural['leaf_selection_differs'] = ((sb != 0) & ~ok, ('leaf_value', 'q'))
    structural['leaf_status_or_error'] = (((sb != 0) | (sf != 0)) & ok & ((sb != sf) | (eb != ef)), ('leaf_value', 'q'))
    return masks, structural


# ---------------------------------------------------------------------------------------------------------------- one geometry's record
class Geometry:
    """One geometry's run-level record (both precisions on the same calls): Tails per quantity and per whole array, structural kinds, (e)'s Q,
    the guard's spreads, costs and the calls priced."""
    def __init__(self, gid, *, group, kind, depth, label):
        self.gid, self.group, self.kind, self.depth, self.label = gid, group, kind, depth, label
        self.quantities, self.whole, self.padding, self.unaligned, self.structural = {}, {}, collections.Counter(), collections.Counter(), {}
        self.q_same = None
        self.q, self.decide, self.gaps, self.q_crosscheck = None, collections.Counter(), dict(bf16=[], f32=[]), []
        self.spreads, self.call_p10 = {name: [] for name in NAMES}, [];self.costs = {name: collections.defaultdict(float) for name in NAMES};self.calls, self.roots = [], 0
        self.extra = {}

    def note(self, kind, mask, affects, call):
        mask = np.asarray(mask, bool);count = int(mask.sum())
        e = self.structural.setdefault(kind, dict(count=0, affects=affects, first=[]))
        if count:
            e['count'] += count;e['first'] += [dict(call=call, cell=[int(x) for x in c]) for c in np.argwhere(mask)[:max(0, FIRST - len(e['first']))]]

    def add_masked(self, f, b, masks, structural, call):
        """Every quantity's |bf16 - f32| at its mask and every structural kind; returns the roots any structural difference touches."""
        roots = np.zeros(len(_get(b, 'pre')), bool)
        for kind, (mask, affects) in structural.items():
            self.note(kind, mask, affects, call);mask = np.asarray(mask, bool)
            if mask.ndim and mask.shape[0] == len(roots):roots |= mask.reshape(len(roots), -1).any(axis=1)
        for name, (path, mask) in masks.items():
            x, y = _get(b, path), _get(f, path);self.quantities.setdefault(name, Tail()).add(deltas(x[mask], y[mask]), call=call, locate=_locator(mask))
        return roots

    def add_whole(self, fa, ba, call, context=None, found=None):
        """bf16 against f32 over every array g2.block_invariance compares (fa, ba: frame_arrays dicts or frames; found: {array: mask} of the
        aligned positions, the rest counted in `unaligned` and never compared)."""
        inv = g2.block_invariance(fa, ba, context=context);x, y = g2.frame_fields(fa), g2.frame_fields(ba)
        for name in inv['floats']:
            d, keep, padding = whole_deltas(y[name], x[name]);shape = np.asarray(x[name]).shape;self.padding[name] += padding
            if (found or {}).get(name) is not None:
                aligned = np.asarray(found[name]).reshape(-1);self.unaligned[name] += int(np.count_nonzero(~aligned));d = d[aligned[keep]];keep = keep & aligned
            self.whole.setdefault(name, Tail()).add(d, call=call, locate=lambda idx, keep=keep, shape=shape: [list(np.unravel_index(int(i), shape)) for i in np.flatnonzero(keep)[idx]])
        for name, d in inv['differing'].items():
            e = self.structural.setdefault(f'whole.{name}', dict(count=0, affects=('whole',), first=[]));e['count'] += int(d.get('count', 1))
            if len(e['first']) < FIRST:e['first'].append(dict(call=call, detail=d))
        return inv

    def add_q(self, bf16, f32, requests, backups, call, context=None, differs=None):
        """Q_RULE: gate.decide_roots on the bf16 and f32 frames, its learner_max_root calls recorded; `differs` [R] marks roots a structural
        difference touches (their |dQ| is kept, and also left out of q.same_selection)."""
        rows = [];same = []
        class Recorder:
            def learner_max_root(self, frame, q, logits, legal):
                r = backups.learner_max_root(frame, q, logits, legal);rows.append(r);return r
        result = gate.decide_roots(bf16, f32, requests, Recorder(), context=context);d, where = [], []
        for q, request in enumerate(requests):
            ia, ib = rows[2 * q].improvement, rows[2 * q + 1].improvement
            if ia is None or ib is None:continue
            qa, qb, legal = ia.completed_q, ib.completed_q, tuple(map(bool, request.root_legal))
            if not all(v is None or math.isfinite(float(v)) for vec in (qa, qb, ia.policy_target, ib.policy_target) for v in vec):continue
            for a, (x, y, ok) in enumerate(zip(qa, qb, legal)):
                if ok and x is not None and y is not None:
                    d.append(abs(float(x) - float(y)));where.append([q, a])
                    if differs is None or not differs[q]:same.append(d[-1])
        self.q = self.q or Tail();self.q.add(np.asarray(d, np.float64), call=call, locate=lambda idx: [where[int(i)] for i in idx])
        self.q_same = self.q_same or Tail();self.q_same.add(np.asarray(same, np.float64), call=call)
        self.decide['roots_structural'] += 0 if differs is None else int(np.count_nonzero(differs))
        self.q.nonfinite += int(result['nonfinite_roots']);self.q.count += int(result['nonfinite_roots'])
        for key in ('roots', 'roots_compared', 'both_refused', 'status_differs', 'nonfinite_roots', 'argmax_q_differs', 'argmax_target_differs'):self.decide[key] += int(result[key])
        self.decide['max_abs_delta_q'] = max(self.decide['max_abs_delta_q'], float(result['max_abs_delta_q']))
        self.q_crosscheck.append(dict(call=call, decide_roots=float(result['max_abs_delta_q']), recorded=max(d, default=0.)))
        self.gaps['bf16'] += [g['gap_pricer'] for g in result['gaps']];self.gaps['f32'] += [g['gap_oracle'] for g in result['gaps']]
        if result['status_differs']:self.note('q_status_differs', np.ones(int(result['status_differs']), bool), ('q',), call)
        return result

    def add_spreads(self, frames, call):
        for name in NAMES:
            s = gate.node_spreads(frames[name]);self.spreads[name] += s
            if name == TESTED:self.call_p10.append(dict(call=call, p10=gate.guard_spread(s)[0]))

    def report(self):
        guard = {}
        for name in NAMES:
            p10, description = gate.guard_spread(self.spreads[name]);guard[name] = dict(description, p10=p10, guard=g2.tolerance_rule(None, p10)['guard_bound'])
        gaps = {name: [g for g in v if g is not None] for name, v in self.gaps.items()}
        crosscheck = [x for x in self.q_crosscheck if abs(x['decide_roots'] - x['recorded']) > 0]
        return dict(gid=self.gid, group=self.group, kind=self.kind, depth=self.depth, label=self.label, calls=self.calls, roots=self.roots,
            quantities={k: t.summary() for k, t in sorted(self.quantities.items())},
            whole={k: dict(t.summary(), padding_nonfinite=self.padding[k], unaligned=self.unaligned[k]) for k, t in sorted(self.whole.items())},
            structural={k: e for k, e in sorted(self.structural.items()) if e['count']}, q=None if self.q is None else dict(self.q.summary(), adapter=q_adapter(self.depth),
                decide_roots=dict(self.decide), crosscheck_mismatches=crosscheck, same_selection=dict(self.q_same.summary(), rule='roots no structural difference touches')),
            gaps=gaps if self.q is not None else None,
            spreads=None if not any(self.spreads.values()) else dict(guard, per_call_bf16=self.call_p10),
            costs={name: dict(v) for name, v in self.costs.items()}, **self.extra)


class Directs:
    """The bf16 site measured directly (DIRECT_RULE) and the f32 premise (PREMISE_RULE), merged over calls."""
    def __init__(self):self.items = {}

    def numeric(self, key, precision, floats):
        e = self.items.setdefault(key, {}).setdefault(precision, dict(floats={}, discrete=dict(passed=True, mismatched=collections.Counter()), calls=0))
        for name, m in floats.items():
            x = e['floats'].setdefault(name, dict(compared=0, exact_equal=0, nonfinite=0, max_abs_difference=0.))
            for k in ('compared', 'exact_equal', 'nonfinite'):x[k] += int(m[k])
            x['max_abs_difference'] = max(x['max_abs_difference'], float(m['max_abs_difference']))
        return e

    def invariance(self, key, precision, inv):
        e = self.numeric(key, precision, inv['floats']);e['calls'] += 1
        e['discrete']['passed'] = e['discrete']['passed'] and inv['passed'];e['discrete']['mismatched'].update({k: int(d.get('count', 1)) for k, d in inv['differing'].items()})
        e.setdefault('nonarray_equal', {}).update({k: e.get('nonarray_equal', {}).get(k, True) and v for k, v in inv['nonarray_equal'].items()})

    def n2(self, key, precision, found, floats):
        e = self.numeric(key, precision, floats);e['calls'] += 1
        e['discrete']['passed'] = e['discrete']['passed'] and found['passed'] and found.get('world_reports_equal', True)
        e['discrete']['mismatched'].update({k: v for k, v in found['mismatched'].items() if v});e['discrete'].setdefault('world_reports_equal', True)
        e['discrete']['world_reports_equal'] = e['discrete']['world_reports_equal'] and found.get('world_reports_equal', True)

    def decision(self, key, result):
        e = self.items.setdefault(key, {}).setdefault(TESTED, dict(decide=collections.Counter(), max_abs_delta_q=0., gaps=[]))
        for k in ('roots', 'roots_compared', 'status_differs', 'nonfinite_roots', 'argmax_q_differs', 'argmax_target_differs'):e['decide'][k] += int(result[k])
        e['max_abs_delta_q'] = max(e['max_abs_delta_q'], float(result['max_abs_delta_q']));e['gaps'] += [min(x for x in (g['gap_pricer'], g['gap_oracle']) if x is not None)
            for g in result['gaps'] if g['gap_pricer'] is not None or g['gap_oracle'] is not None]

    def identity(self, key, precision, a, b, names):
        """R3's turn 1 against the depth-2 frame's, field by field over whole arrays: floats max |diff| (0: bit-identical), discrete mismatches."""
        e = self.items.setdefault(key, {}).setdefault(precision, dict(fields={}, calls=0));e['calls'] += 1
        for name in names:
            x, y = getattr(a, name, None), getattr(b, name, None)
            r = e['fields'].setdefault(name, dict(compared=0, unequal=0, max_abs_difference=0., nonfinite=0, shape_differs=False))
            if x is None or y is None or np.asarray(x).shape != np.asarray(y).shape:r['shape_differs'] = True;continue
            x, y = np.asarray(x), np.asarray(y);r['compared'] += int(x.size)
            if x.dtype.kind == 'f':
                d = deltas(x, y);r['unequal'] += int(np.count_nonzero(d));r['nonfinite'] += int(np.count_nonzero(~np.isfinite(d)))
                if np.isfinite(d).any():r['max_abs_difference'] = max(r['max_abs_difference'], float(d[np.isfinite(d)].max()))
            else:r['unequal'] += int(np.count_nonzero(x != y))

    def report(self):
        return clv.plain(self.items)


TURN1_FIELDS = ('pre', 'root_peer_requested', 'root_peer_legal', 'root_peer_logits', 't1_status', 't1_value', 't1_error', 't1_own_requested', 't1_own_legal',
    't1_own_logits', 't1_peer_requested', 't1_peer_legal', 't1_peer_logits', 't1_digest_own', 't1_digest_peer')


# ---------------------------------------------------------------------------------------------------------------- the runs' arguments and pricers
def parse_run(name):
    """d3_depthn_gate.parse on RUN_ARGV[name] (with roots and the champion flags it needs; nothing is loaded): (args, None) or (None, error)."""
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):_, args = gate.parse(RUN_ARGV[name] + RUN_ROOTS[name] + PARSE_CHAMPION)
    except SystemExit:return None, stderr.getvalue().strip().splitlines()[-1:] or ['refused']
    return args, None


def n2_fixed_refused():
    """Whether the gate's parser refuses --fixed-shapes with --n2-equivalence (R2 then prices at per-call buckets)."""
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):gate.parse(RUN_ARGV['R2'] + ['--fixed-shapes'] + BANK_ARGV + PARSE_CHAMPION)
    except SystemExit:return True, stderr.getvalue().strip().splitlines()[-1:]
    return False, []


def run_arguments(overrides=None):
    """{run: parsed args} for RUN_ARGV (the gate's own parser), each with `overrides` {run: {arg: value}} applied after parsing (tests and
    fixtures; every override is a departure), and the record: each run's command line, the parsed values, PINNED and the departures."""
    runs, record = {}, {}
    for name in RUN_ARGV:
        args, error = parse_run(name)
        if args is None:record[name] = dict(argv=RUN_ARGV[name], refused=error);continue
        for key, value in ((overrides or {}).get(name) or {}).items():setattr(args, key, value)
        runs[name] = args;parsed = {k: _tuple(getattr(args, k, None)) for k in RECORDED_ARGS}
        departures = [f'{name}: {k} {parsed.get(k, getattr(args, k, None))!r} != pinned {v!r}' for k, v in PINNED[name].items() if _tuple(getattr(args, k, None)) != v]
        record[name] = dict(argv=RUN_ARGV[name], parsed=parsed, pinned=PINNED[name], departures=departures, overrides=(overrides or {}).get(name) or {})
    refused, why = n2_fixed_refused();record['R2']['fixed_shapes_refused'] = dict(refused=refused, message=why)
    bare = gate.parse(BANK_ARGV + PARSE_CHAMPION)[1];record['gate_cli_bare_defaults'] = {k: _tuple(getattr(bare, k, None)) for k in RECORDED_ARGS}
    return runs, record


def _tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else x


def pricer_options(p):
    """A pricer's recorded options (DepthNPricer, Depth2Pricer or D3's FramePricer), its evaluator by identity."""
    keys = ('depth', 'k_own', 'k_opp', 'k1_own', 'k1_opp', 'capacity', 'block_lanes', 'blocks_per_call', 'selection', 'digest', 'actor_capacity', 'fixed_capacity',
        'child_block', 'block_rows', 'vectorized', 'host_prep')
    out = {k: (list(v) if isinstance(v, tuple) else v) for k in keys if (v := getattr(p, k, None)) is not None}
    return dict(out, type=type(p).__name__, evaluator=ce.describe(getattr(p, 'forward', None)))


def r2_configs(args):
    """R2's configs as run_n2 names them: (full_width, 9) and (top{n2_k}, n2_k)."""
    return (('full_width', 9), (f'top{args.n2_k}', args.n2_k))


def build_pricers(module, executor, model, evaluators, runs, selected, *, frame_pricer=None):
    """{gid: {precision: pricer}} for the selected groups, each built as the gate builds it (from the parsed run arguments), and
    {gid: dict(options per precision, departures)}: R1's frame through gate.DepthNPricers (with INVARIANCE_LANES when r1-invariance is
    selected: the twin is the 729 instance) and its depth-2 pass (gate.depth2_pass_pricer) compared option by option; R3's through
    gate.DepthNPricers and its pass likewise; R2's DepthNPricer N=2 as run_n2 constructs it and g2.make_pricer per child block; D3 as
    g2.with_forward builds it for (b) (champion_d3_geometry.build_d3)."""
    pricers, options = {}, {}
    def put(gid, made, departures=()):
        pricers[gid] = made;options[gid] = dict(pricers={n: pricer_options(p) for n, p in made.items()}, departures=list(departures))
    def same_pass(args, frame):
        out = []
        for name in NAMES:
            passed = pricer_options(gate.depth2_pass_pricer(module, executor, model, capacity=args.capacity, actor_capacity=1, k1_own=args.k1_own, k1_opp=args.k1_opp,
                block_lanes=args.block_lanes, digest=True, fixed_capacity=args.fixed_shapes, forward=evaluators[name]))
            if passed != pricer_options(frame[name]):out.append(f'{name}: the depth-2 pass {passed} differs from R1\'s frame {pricer_options(frame[name])}: measured once as R1\'s frame')
        return out
    if 'r1' in selected:
        args = runs['R1_invariance' if 'r1-invariance' in selected else 'R1'];lanes = gate.INVARIANCE_LANES if 'r1-invariance' in selected else (args.block_lanes,)
        made = {name: gate.DepthNPricers(module, executor, model, depth=args.depth, k_own=args.k_own_ply, k_opp=args.k_opp_ply, selection=args.selection,
            capacity=args.capacity, actor_capacity=1, lanes=lanes, k1_own=args.k1_own, k1_opp=args.k1_opp, fixed_capacity=args.fixed_shapes,
            forward=evaluators[name]).get()[1] for name in NAMES}
        put('R1.frame', {n: m[-1] for n, m in made.items()})
        options['R1.frame']['departures'] += same_pass(runs['R1'], pricers['R1.frame'])
        if 'r3' in selected:options['R1.frame']['departures'] += [f'R3 {x}' for x in same_pass(runs['R3'], pricers['R1.frame'])]
        if 'r1-invariance' in selected:put('R1.invariance_twin', {n: m[0] for n, m in made.items()})
    if 'r3' in selected:
        args = runs['R3']
        put('R3.frame', {name: gate.DepthNPricers(module, executor, model, depth=args.depth, k_own=args.k_own_ply, k_opp=args.k_opp_ply, selection=args.selection,
            capacity=args.capacity, actor_capacity=1, lanes=(args.block_lanes,), k1_own=args.k1_own, k1_opp=args.k1_opp, fixed_capacity=args.fixed_shapes,
            forward=evaluators[name]).get()[1][-1] for name in NAMES})
    if 'r3-d3' in selected:
        if frame_pricer is None:raise ValueError('r3-d3 needs the bank provider\'s FramePricer')
        put('R3.d3', d3g.build_d3(frame_pricer, evaluators))
    args = runs['R2']
    for group, (config, k) in zip(('r2-full', 'r2-top'), r2_configs(args)):
        if group not in selected:continue
        put(f'R2.{config}.depthn', {name: module.DepthNPricer(executor, model, capacity=args.capacity, depth=2, k_own=k, k_opp=k, k1_own=args.k1_own, k1_opp=args.k1_opp,
            block_lanes=args.block_lanes, selection='prior', digest=True, actor_capacity=1, **gate.forward_kwargs(evaluators[name])) for name in NAMES})
        for b in g2.INVARIANCE_BLOCKS:
            put(f'R2.{config}.depth2_cb{b}', {name: g2.make_pricer(executor, model, widths=dict(k_own=k, k_opp=k, k1_own=args.k1_own, k1_opp=args.k1_opp), actor_capacity=1,
                capacity=args.capacity, child_block=b, **gate.forward_kwargs(evaluators[name]))[0] for name in NAMES})
    for gid, record in options.items():
        record['departures'] += [f'{name}: evaluator {o["evaluator"]["identity"]} is not the champion at {clv.PRECISION_OF[name]}' for name, o in record['pricers'].items()
            if o['evaluator']['identity'][:1] != ['champion'] or o['evaluator']['identity'][-1] != clv.PRECISION_OF[name]]
    return pricers, options


def expected_options(gid, runs):
    """The pinned options of a geometry (PINNED through the parsed run), the gate's construction."""
    r1, r2, r3 = PINNED['R1'], PINNED['R2'], PINNED['R3'];base = dict(digest=True, actor_capacity=1)
    if gid in ('R1.frame', 'R1.invariance_twin'):
        return dict(base, depth=2, k_own=[9], k_opp=[9], k1_own=9, k1_opp=9, capacity=r1['capacity'], block_lanes=r1['block_lanes'] if gid == 'R1.frame' else gate.INVARIANCE_LANES[0],
            selection='prior', fixed_capacity=True)
    if gid == 'R3.frame':
        return dict(base, depth=3, k_own=[3, 3], k_opp=[3, 3], k1_own=9, k1_opp=9, capacity=r3['capacity'], block_lanes=r3['block_lanes'], selection='prior', fixed_capacity=True)
    if gid == 'R3.d3':return dict(capacity=4096, block_rows=32, vectorized=True, host_prep=True)
    config, kind = gid.split('.')[1:];k = 9 if config == 'full_width' else r2['n2_k']
    if kind == 'depthn':return dict(base, depth=2, k_own=[k], k_opp=[k], k1_own=9, k1_opp=9, capacity=r2['capacity'], block_lanes=r2['block_lanes'], selection='prior', fixed_capacity=False)
    return dict(base, k_own=k, k_opp=k, k1_own=9, k1_opp=9, capacity=r2['capacity'], child_block=int(kind.removeprefix('depth2_cb')), fixed_capacity=False)


def option_departures(options, runs):
    out = []
    for gid, record in options.items():
        want = expected_options(gid, runs);out += [f'{gid}: {x}' for x in record['departures']]
        for name, got in record['pricers'].items():out += [f'{gid} {name}: {k} {got.get(k)!r} != pinned {v!r}' for k, v in want.items() if got.get(k) != v]
    return out


# ---------------------------------------------------------------------------------------------------------------- driving the measurement
GEOMETRY_LABELS = {'R1.frame': 'R1\'s depth-2 frame (DepthNPricer k 9, 2187 lanes, fixed; also R1\'s and R3\'s full-width depth-2 pass)',
    'R1.invariance_twin': 'R1\'s block-invariance twin (729 lanes)', 'R3.frame': 'R3\'s depth-3 frame (DepthNPricer prior k 3/3, 2187 lanes, fixed)',
    'R3.d3': 'D3\'s FramePricer at R3\'s bucket (calls of 40 roots)'}


def label_of(gid):
    if gid in GEOMETRY_LABELS:return GEOMETRY_LABELS[gid]
    _, config, kind = gid.split('.')
    return f'R2 {config}: ' + ('DepthNPricer N=2 (per-call buckets)' if kind == 'depthn' else f'merged Depth2Pricer child_block {kind.removeprefix("depth2_cb")}')


def _price(pricer, name, requests, materializer, recipe, evaluators, progress, context, gid, record):
    with stage(progress, 'price', geometry=gid, precision=name, **context) as info:
        clock = time.perf_counter();out = pricer.price(requests, materializer, recipe, (evaluators[name].params,))
        timing = getattr(pricer, 'last_timing', None) if isinstance(out, (list, tuple)) else getattr(out, 'timing', None)
        cost = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(timing));info.update(cost)
        for k, v in cost.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):record.costs[name][k] += v
    return out


def price_both(record, pricers, requests, materializer, recipe, evaluators, progress, context):
    frames = {name: _price(pricers[name], name, requests, materializer, recipe, evaluators, progress, context, record.gid, record) for name in NAMES}
    record.calls.append(context['call']);record.roots += len(requests);return frames


def measure(groups, *, evaluators, executor, model, materializer, recipe, frame_pricer=None, module=None, runs=None, selected=GROUPS, calls=None,
        r3_call_roots=R3_ROOTS, split=SPLIT, progress=None, backups=None, backups2=None):
    """groups [[Request]] bound once under the bf16 champion; evaluators {'f32', 'bf16'} one champion at two precisions. Per call (groups split
    at `split`; `calls` a set of call indices or None for all): every selected geometry at both precisions, its masks, whole arrays, Q and
    spreads, and the direct sites; then R3's D3 at calls of `r3_call_roots`. Returns dict(geometries, direct, options, departures, calls)."""
    selected = tuple(g for g in GROUPS if g in selected);missing = [f'{g} needs {NEEDS[g]}' for g in selected if g in NEEDS and NEEDS[g] not in selected]
    if missing:raise ValueError('; '.join(missing))
    if set(evaluators) != set(NAMES):raise ValueError(f'evaluators named {NAMES} required')
    module = module or gate.pricer_module();backups = backups or importlib.import_module('gpu_engine.depthn_backups')
    backups2 = backups2 or importlib.import_module('gpu_engine.depth2_backups');runs = runs or run_arguments()[0]
    pricers, options = build_pricers(module, executor, model, evaluators, runs, selected, frame_pricer=frame_pricer)
    records = {gid: Geometry(gid, group=_group(gid), kind=_kind(gid), depth=3 if gid == 'R3.frame' else 2, label=label_of(gid)) for gid in pricers}
    direct = Directs();split_calls = g2.split_groups(groups, split);done = []
    for call, reqs in enumerate(split_calls):
        if calls is not None and call not in calls:continue
        reqs = tuple(reqs);context = dict(call=call, roots=len(reqs));done.append(call);frames = {}
        with stage(progress, 'call', **context) as info:
            if 'r1' in selected:
                r = records['R1.frame'];frames['R1'] = f = price_both(r, pricers['R1.frame'], reqs, materializer, recipe, evaluators, progress, context)
                _depthn_parts(r, f, reqs, backups, call, context, whole='r1-invariance' in selected)
                if 'r1-invariance' in selected:
                    t = records['R1.invariance_twin'];twin = price_both(t, pricers['R1.invariance_twin'], reqs, materializer, recipe, evaluators, progress, context)
                    _depthn_parts(t, twin, reqs, backups, call, context, whole=True, q=False, spreads=False)
                    for name in NAMES:direct.invariance('R1.block_invariance', name, g2.block_invariance(gate.frame_arrays(twin[name]), gate.frame_arrays(f[name]), context=context))
                    twin = None
            if 'r3' in selected:
                r = records['R3.frame'];d3 = price_both(r, pricers['R3.frame'], reqs, materializer, recipe, evaluators, progress, context)
                _depthn_parts(r, d3, reqs, backups, call, context)
                if 'R1' in frames:
                    for name in NAMES:direct.identity('R3.turn1_identity', name, d3[name], frames['R1'][name], TURN1_FIELDS)
                d3 = None
            frames = None
            for group, (config, _) in zip(('r2-full', 'r2-top'), r2_configs(runs['R2'])):
                if group in selected:_r2_call(records, pricers, direct, config, reqs, materializer, recipe, evaluators, module, backups2, call, context, progress)
            info.update(groups=list(selected))
    if 'r3-d3' in selected:_d3_calls(records['R3.d3'], pricers['R3.d3'], groups, r3_call_roots, materializer, recipe, evaluators, progress)
    departures = option_departures(options, runs) + ([] if split == SPLIT else [f'call split {split} != every run\'s {SPLIT}'])
    return dict(geometries={gid: r.report() for gid, r in records.items()}, direct=direct.report(), options=options, departures=departures,
        calls=dict(split=split, sizes=[len(c) for c in split_calls], measured=done, selected=None if calls is None else sorted(calls)), groups=list(selected))


def _group(gid):
    return dict(zip(('R1.frame', 'R1.invariance_twin', 'R3.frame', 'R3.d3'), ('r1', 'r1-invariance', 'r3', 'r3-d3'))).get(gid) or (
        'r2-full' if gid.startswith('R2.full_width.') else 'r2-top')


def _kind(gid):
    return 'd3' if gid == 'R3.d3' else 'depthn' if gid.startswith(('R1.', 'R3.')) else 'n2_depthn' if gid.endswith('.depthn') else 'depth2'


def _depthn_parts(record, frames, requests, backups, call, context, *, whole=False, q=True, spreads=True):
    f, b = frames[REFERENCE], frames[TESTED];aligned, found = align(f, b);differs = record.add_masked(aligned, b, *depthn_masks(aligned, b, found), call)
    if whole:
        fa = gate.frame_arrays(aligned);record.add_whole(fa, gate.frame_arrays(b), call, context, found=found_arrays(found, fa))
    if q:record.add_q(b, f, requests, backups, call, context, differs=differs)
    if spreads:record.add_spreads(frames, call)


def _r2_call(records, pricers, direct, config, reqs, materializer, recipe, evaluators, module, backups2, call, context, progress):
    """One R2 config on one call: N=2 (as_depth2_frame), the Depth2Pricer at each child block, their measurements and the direct sites."""
    where = dict(context, config=config);gid = f'R2.{config}.depthn';r = records[gid]
    raw = price_both(r, pricers[gid], reqs, materializer, recipe, evaluators, progress, where);n2 = {name: module.as_depth2_frame(raw[name]) for name in NAMES}
    masks, structural = turn1_masks(n2[REFERENCE], n2[TESTED]);t2, s2 = t2_masks(n2[REFERENCE], n2[TESTED])
    differs = r.add_masked(n2[REFERENCE], n2[TESTED], {**masks, **t2}, {**structural, **s2}, call);r.add_q(n2[TESTED], n2[REFERENCE], reqs, backups2, call, where, differs=differs)
    r.add_spreads(raw, call)
    raw = None;d2 = {}
    for b in g2.INVARIANCE_BLOCKS:   # as n2_equivalence_call: the other child block first, then the compared one
        gid = f'R2.{config}.depth2_cb{b}';r = records[gid];d2[b] = frames = price_both(r, pricers[gid], reqs, materializer, recipe, evaluators, progress, where)
        masks, structural = turn1_masks(frames[REFERENCE], frames[TESTED]);t2, s2 = t2_masks(frames[REFERENCE], frames[TESTED])
        differs = r.add_masked(frames[REFERENCE], frames[TESTED], {**masks, **t2}, {**structural, **s2}, call);r.add_whole(frames[REFERENCE], frames[TESTED], call, where)
        if b == g2.INVARIANCE_BLOCKS[-1]:r.add_q(frames[TESTED], frames[REFERENCE], reqs, backups2, call, where, differs=differs)
    compared, other = d2[g2.INVARIANCE_BLOCKS[-1]], d2[g2.INVARIANCE_BLOCKS[0]]
    for name in NAMES:
        found, floats = gate.n2_compare(compared[name], n2[name], where);direct.n2(f'R2.{config}.n2', name, found, floats)
        direct.invariance(f'R2.{config}.child_block_invariance', name, g2.block_invariance(other[name], compared[name], context=where))
    direct.decision(f'R2.{config}.e', gate.decide_roots(compared[TESTED], n2[TESTED], reqs, backups2, context=where))


def _d3_calls(record, pricers, groups, size, materializer, recipe, evaluators, progress):
    """R3's D3 side of (b): calls of `size` roots (R3's saved-root count), only those at the bucket a `size`-root call compiles (shorter tails
    are skipped and counted); champion_d3_geometry.compare_call per call."""
    want = d3g.bucket(pricers[TESTED], size);record.extra.update(bucket=want, call_roots=size, skipped_roots=0)
    for call, reqs in enumerate(g2.split_groups(groups, size)):
        if d3g.bucket(pricers[TESTED], len(reqs)) != want:record.extra['skipped_roots'] += len(reqs);continue
        reqs = tuple(reqs);context = dict(call=call, roots=len(reqs), bucket=want);results = price_both(record, pricers, reqs, materializer, recipe, evaluators, progress, context)
        c = d3g.compare_call(results, reqs, call=call)
        for name, key in (('d3.t1_value', 'values'), ('d3.root_peer_logits', 'logits'), ('d3.root_peer_logits.all_lanes', 'all_lanes')):
            d = np.abs(np.asarray(c[key], np.float64));d[~np.isfinite(d)] = np.inf;record.quantities.setdefault(name, Tail()).add(d, call=call)
        for kind, e in c['findings'].items():
            x = record.structural.setdefault(kind, dict(count=0, affects='all' if kind not in d3g.NEUTRAL_KINDS else (), first=[]));x['count'] += e['count'];x['first'] = (x['first'] + e['first'])[:FIRST]


# ---------------------------------------------------------------------------------------------------------------- earlier reports
def read_json(path):
    raw = Path(path).read_bytes();return json.loads(raw), hashlib.sha256(raw).hexdigest()


def earlier_entry(earlier, spec):
    """An earlier report's measurement as a side source: ('validation', level, column, quantity) (geometry.levels: pricer_all already holds
    the adapter bound; oracle_sample is |oracle bf16 - f32 raw|) or ('d3_geometry', section, quantity) (p_run / d_all, with the bound)."""
    source, *keys = spec;report = (earlier or {}).get(source)
    if source == 'validation':
        level, column, quantity = keys;m = ((((report or {}).get('geometry') or {}).get('levels') or {}).get(level) or {}).get(column, {}).get(quantity) or {}
        key = f'geometry.levels.{level}.{column}.{quantity}'
    else:
        section, quantity = keys;m = ((report or {}).get(section) or {}).get(quantity) or {};key = f'{section}.{quantity}'
    return dict(source=f'{source}.json', key=key, max=m.get('max') if m.get('count') else None, count=m.get('count', 0), nonfinite=m.get('nonfinite', 0),
        precision_max=m.get('precision_max'))


def check_earlier(earlier, header):
    """The earlier reports against this run (source, roots, champion and shapes); {'validation': flags, 'd3_geometry': flags}."""
    out = {}
    source, roots, champion = header.get('source') or {}, header.get('roots'), header.get('champion') or {}
    v = (earlier or {}).get('validation')
    if v is None:out['validation'] = ['no validation.json']
    else:
        h, geo = v.get('header') or {}, v.get('geometry') or {};shapes = geo.get('shapes') or {};flags = []
        if v.get('schema') != clv.SCHEMA:flags.append(f'schema {v.get("schema")!r} != {clv.SCHEMA!r}')
        flags += [f'source {k} {(h.get("source") or {}).get(k)!r} != this run\'s {source.get(k)!r}' for k in ('d3_root', 'chunk', 'declaration_sha256') if (h.get('source') or {}).get(k) != source.get(k)]
        if h.get('roots') != roots:flags.append(f'roots {h.get("roots")!r} != this run\'s {roots!r}')
        for name in NAMES:
            e = (h.get('evaluators') or {}).get(name) or {}
            flags += [f'evaluator {name} {k} {e.get(k)!r} != this run\'s {champion.get(k)!r}' for k in ('archive_sha256', 'checkpoint_sha256') if e.get(k) != champion.get(k)]
            if e.get('precision') != clv.PRECISION_OF[name]:flags.append(f'evaluator {name} precision {e.get("precision")!r}')
        want = dict(fixed_shapes=True, capacity=512, block_lanes=2187, depth3_k=3, oracle_chunk=512, matches_gate=True)
        flags += [f'shapes {k} {shapes.get(k)!r} != {x!r}' for k, x in want.items() if shapes.get(k) != x]
        if (h.get('options') or {}).get('max_roots') is not None:flags.append('the earlier run priced only --max-roots')
        out['validation'] = flags
    d = (earlier or {}).get('d3_geometry')
    if d is None:out['d3_geometry'] = ['no d3_geometry.json']
    else:
        h = d.get('header') or {};flags = []
        if d.get('schema') != d3g.SCHEMA:flags.append(f'schema {d.get("schema")!r} != {d3g.SCHEMA!r}')
        flags += [f'source {k} {(h.get("source") or {}).get(k)!r} != this run\'s {source.get(k)!r}' for k in ('d3_root', 'chunk', 'declaration_sha256') if (h.get('source') or {}).get(k) != source.get(k)]
        if h.get('roots') != roots:flags.append(f'roots {h.get("roots")!r} != this run\'s {roots!r}')
        flags += [f'champion {k} {(h.get("champion") or {}).get(k)!r} != this run\'s {champion.get(k)!r}' for k in ('archive_sha256', 'checkpoint_sha256') if (h.get('champion') or {}).get(k) != champion.get(k)]
        for section in ('d3_options', 'pass_options'):
            if not (h.get(section) or {}).get('matches_gate'):flags.append(f'{section} does not match the gate: {(h.get(section) or {}).get("departures")}')
        if (h.get('options') or {}).get('fixed_shapes') is not True:flags.append('the earlier run did not use --fixed-shapes')
        if d.get('status') != 'MEASURED':flags.append(f'status {d.get("status")!r}')
        out['d3_geometry'] = flags
    return out


# ---------------------------------------------------------------------------------------------------------------- the sites and their fields
def sentinel_declared():
    return dict(content=dict(bounds=dict(SENTINEL)), sha256='sentinel', path='sentinel')


def gate_bound(check, name, depth):
    """The bound the gate applies to one site under a declaration, by running the gate's own functions on a sentinel declaration:
    (c) declared_bound_for; (b) the summarize hook's declared_rule through g2.bound_for, then declared_root_sites (root_bound); block invariance
    g2.invariance_bound, the root re-finalized at root_logit_bound (declared_root_sites / summarize_n2); n2 floats and heads finalize_n's
    root_bound or g2.bound_for; every (e) the rule's q_noise_bound; (f) the value bound (merge_choices)."""
    declared = sentinel_declared();rule = gate.declared_rule(g2.tolerance_rule(None, None), declared)
    if check == 'c':return gate.declared_bound_for(name, depth, gate.declared_numbers(declared), gate.declared_deeper(declared, None, 'not measured: n/a')[0])[0]
    if check in ('b', 'n2.floats', 'n2.heads'):
        root = gate.root_bound(name, rule);return root if root is not None else g2.bound_for(name, rule)
    if check in ('block_invariance', 'n2.block_invariance'):return rule['root_logit_bound'] if name in gate.ROOT_LOGITS else g2.invariance_bound(name, rule)
    if check in ('e', 'n2.e', 'n2.e_merged_depth2'):return rule.get('q_noise_bound', rule['value_bound'])
    if check == 'f':return rule['value_bound']
    raise ValueError(f'no gate bound for check {check!r}')


def field_of(bound):
    if bound == 0:return 'exact'
    return next((k for k, v in SENTINEL.items() if v == bound), None)


def guarded(check, name):
    """Whether the gate applies the misplacement guard to this site (g2.guard_for / merge_invariance's *value rule)."""
    if check in ('b', 'c', 'n2.floats'):return name not in g2.FLOORS and not g2.is_logit(name)
    if check in ('block_invariance', 'n2.block_invariance'):return name.endswith('value')
    return False


GATE_FUNCTION = dict(b='g2.check_d values; g2.summarize (bound_for under the declared_rule hook) + declared_root_sites', c='declared_bound_for (apply_rules_depthn)',
    block_invariance='g2.merge_invariance (invariance_bound) + declared_root_sites', e='decision_check_n -> decide_roots; q_noise_bound',
    f='choice_check_n; merge_choices(value_bound)', **{'n2.floats': 'n2_compare; summarize_n2 finalize_n', 'n2.e': 'decide_roots (depth2_backups); q_noise_bound',
    'n2.block_invariance': 'CHILD_BLOCK_INVARIANCE: g2.merge_invariance + refinalize_root', 'n2.e_merged_depth2': 'MERGED_DEPTH2_DECISION: g2.decision_check; q_noise_bound',
    'n2.heads': 'summarize_n2 turn1_heads_vs_oracle (reported, not gated)'})
EXACT = dict(b='(b)\'s discrete fields exact against D3 (pre, request and legal masks, status and error, the turn-1 pair set)',
    c='(a) exact at the same positions: digests of the view each value or prior was computed from (t1_digest_own/peer, ply digests, leaf_digest), '
        'status, error, request flags and legal masks; (f)\'s exact prior check covers the priors the backup is handed',
    block_invariance='every discrete array bit-equal between the two block geometries', e='nothing exact on Q itself; (f)\'s read identity and exact '
        'priors certify which children and priors the backup reads', f='read identity (exact: which child every grid entry reads) and the exact prior check',
    **{'n2.floats': 'n2\'s discrete arrays exact, digests on scored views, world_reports equal', 'n2.e': 'n2\'s discrete arrays exact (both surfaces enumerate the same leaves)',
    'n2.block_invariance': 'every discrete array bit-equal between child blocks', 'n2.e_merged_depth2': 'the merged Depth2Pricer\'s exact checks in the depth-2 gate',
    'n2.heads': 'n/a (reported)'})
B_POSITIONS = dict(t1_value='every (q, world, a, r) D3 priced (joint value non-None) that the frame enumerates (g2.check_d)',
    root_peer_logits='all nine lanes of every world D3 priced, requested or not (g2.check_d)')
C_POSITIONS = dict(t1_value_head='turn-1 cells valued (status 2 or codes 5/6) with the oracle\'s status and error', t1_own_logits='the requested learner\'s legal lanes of valued cells',
    t1_peer_logits='the requested opponent\'s legal lanes of expanded cells', root_peer_logits='the requested opponent\'s legal lanes of usable worlds',
    leaf_value='head-valued leaves (status 4 in the oracle and the frame)', depth2_pass_t1_value='the depth-N frame\'s valued turn-1 cells (pass against frame)',
    depth2_pass_t1_own_logits='the frame\'s valued cells, requested learner\'s legal lanes', depth2_pass_t1_peer_logits='the frame\'s expanded cells, requested opponent\'s legal lanes',
    depth2_pass_root_peer_logits='usable worlds with the opponent requested, legal lanes', depth2_pass_t2_value='the pass\'s re-run leaves (every terminal/error leaf plus '
        '--depth2-pass-sample 8 per world) head-valued in both (pass against the oracle)')


def site_registry(runs, *, r1_whole=None, r2_whole=None):
    """Every float site R1, R2 and R3 gate (and the reported-only ones), from the gate's code (SITES_RULE): one dict per site with its run,
    check, the gate's quantity name, positions, the bound field the gate applies (gate_bound), the guard, the sides and the construction."""
    rows = [];r1, r3 = int(runs['R1'].depth), int(runs['R3'].depth)
    def add(run, check, quantity, positions, sides, construction='sum', depth=2, report_only=False, derived=None, note=None, direct=None, premise=None, config=None):
        rows.append(dict(id=f'{run}{"." + config if config else ""}.{check}.{quantity}', run=run, config=config, check=check, quantity=quantity, positions=positions,
            field=field_of(gate_bound(check, quantity, depth)), gate_function=GATE_FUNCTION.get(check), guarded=guarded(check, quantity), sides=sides,
            construction=construction, derived=derived, report_only=report_only, exact_coverage=EXACT.get(check), note=note, direct=direct, premise=premise, depth=depth))
    side = lambda label, gid=None, q=None, earlier=(), depth=2: dict(label=label, geometry=gid, quantity=q, earlier=list(earlier), depth=depth)
    def depthn_run(run, gid, depth, level):
        t1, frame, r1_run = f'depth{depth}.turn1', f'{run} frame', run == 'R1'
        d3 = lambda q: side('D3\'s FramePricer' + ('' if r1_run else ' (R3\'s bucket here, chunk 0\'s buckets from d3_geometry.json)'), None if r1_run else 'R3.d3', q,
            [('d3_geometry', 'd_all', q)])
        add(run, 'b', 't1_value', B_POSITIONS['t1_value'], [side(f'{frame} turn 1', gid, 't1_value', [('validation', t1, 'pricer_all', 'value')] +
            ([('d3_geometry', 'p_run', 'pricer.t1_value')] if r1_run else [])), d3('d3.t1_value')], depth=depth)
        add(run, 'b', 'root_peer_logits', B_POSITIONS['root_peer_logits'], [side(f'{frame} turn 1', gid, 'root_peer_logits.all_lanes',
            [('d3_geometry', 'p_run', 'pricer.root_peer_logits.all_lanes')] if r1_run else []), d3('d3.root_peer_logits.all_lanes')], depth=depth)
        for name, q, lv, oq in (('t1_value_head', 't1_value', t1, 'value'), ('t1_own_logits', 't1_own_logits', t1, 'own_logits'), ('t1_peer_logits', 't1_peer_logits', t1, 'opp_logits'),
                ('root_peer_logits', 'root_peer_logits', 'root', 'opp_logits')):
            add(run, 'c', name, C_POSITIONS[name], [side(f'{frame}', gid, q, [('validation', lv, 'pricer_all', oq)]), side('the gate oracle', earlier=[('validation', lv, 'oracle_sample', oq)])],
                'oracle_sample', depth=depth)
        for p in range(2, depth):
            for name, q, oq in ((f'ply{p}_value_head', f'ply{p}_value', 'value'), (f'ply{p}_own_logits', f'ply{p}_own_logits', 'own_logits'), (f'ply{p}_opp_logits', f'ply{p}_opp_logits', 'opp_logits')):
                add(run, 'c', name, f'ply-{p} nodes ({q}: check_heads_n, _node_valued with the oracle\'s status)', [side(frame, gid, q, [('validation', f'depth{depth}.ply{p}', 'pricer_all', oq)]),
                    side('the gate oracle', earlier=[('validation', f'depth{depth}.ply{p}', 'oracle_sample', oq)])], 'oracle_sample', depth=depth)
        add(run, 'c', 'leaf_value', C_POSITIONS['leaf_value'], [side(frame, gid, 'leaf_value', [('validation', level, 'pricer_all', 'value')]),
            side('the gate oracle', earlier=[('validation', level, 'oracle_sample', 'value')])], 'oracle_sample', depth=depth)
        for name, q, levels, oq in (('depth2_pass_t1_value', 't1_value', ('depth2.turn1', t1), 'value'), ('depth2_pass_t1_own_logits', 't1_own_logits', ('depth2.turn1', t1),
                'own_logits'), ('depth2_pass_t1_peer_logits', 't1_peer_logits', ('depth2.turn1', t1), 'opp_logits'), ('depth2_pass_root_peer_logits', 'root_peer_logits', ('root', 'root'), 'opp_logits')):
            add(run, 'c', name, C_POSITIONS[name], [side('the full-width depth-2 pass (R1\'s frame)', 'R1.frame', q, [('validation', levels[0], 'pricer_all', oq)]),
                side(f'{frame} turn 1', gid, q, [('validation', levels[1], 'pricer_all', oq)])], depth=depth, direct=('R3.turn1_identity', q) if run == 'R3' else None)
        add(run, 'c', 'depth2_pass_t2_value', C_POSITIONS['depth2_pass_t2_value'], [side('the full-width depth-2 pass (R1\'s frame)', 'R1.frame', 'leaf_value',
            [('validation', 'depth2.leaf', 'pricer_all', 'value')]), side('the gate oracle', earlier=[('validation', 'depth2.leaf', 'oracle_sample', 'value')])], 'oracle_sample', depth=depth)
        add(run, 'e', 'max_abs_delta_q', 'completed Q over the legal actions of every root labeled in both frames (decision_check_n: the frame against its '
            'oracle-heads copy, depthn_backups); also the near-tie threshold on the top-two gaps', [side(f'{frame} (depthn_backups)', gid, 'q', depth=depth)], 'mirror', depth=depth)
        for name in ('max_abs_delta_child_value', 'max_abs_delta_row_q'):
            add(run, 'f', name, 'every interior node priced on both sides (turn-1 cells' + (', ply-2..N-1 nodes' if depth > 2 else '') + '): grid entries and row q under the '
                'pricer\'s priors on both sides', [], 'derived', depth=depth, derived=f'{run}.c.leaf_value')
    depthn_run('R1', 'R1.frame', r1, f'depth{r1}.leaf')
    for name in r1_whole or DEPTHN_FLOATS:
        add('R1', 'block_invariance', name, 'the whole array (every element), block_lanes 729 against 2187 (--block-invariance only)',
            [side('the 729-lane twin', 'R1.invariance_twin', f'whole.{name}'), side('R1 frame (2187)', 'R1.frame', f'whole.{name}')], direct=('R1.block_invariance', name),
            premise=('R1.block_invariance', name))
    depthn_run('R3', 'R3.frame', r3, f'depth{r3}.leaf')
    for config, k in r2_configs(runs['R2']):
        n, d81, d27 = f'R2.{config}.depthn', f'R2.{config}.depth2_cb{g2.INVARIANCE_BLOCKS[-1]}', f'R2.{config}.depth2_cb{g2.INVARIANCE_BLOCKS[0]}'
        for q, where in (('t1_value', 'valued turn-1 cells of the Depth2Frame'), ('t2_value', 'status-4 turn-2 leaves of the Depth2Frame'), ('root_peer_logits',
                'usable worlds with the opponent requested, legal lanes'), ('t1_own_logits', 'valued cells, requested learner, legal lanes'), ('t1_peer_logits', 'expanded cells, requested opponent, legal lanes')):
            add('R2', 'n2.floats', q, f'{where} (n2_compare, scored cells of the Depth2Pricer frame)', [side('merged Depth2Pricer (child_block 81)', d81, q), side('DepthNPricer N=2', n, q)],
                config=config, direct=(f'R2.{config}.n2', q), premise=(f'R2.{config}.n2', q))
        add('R2', 'n2.e', 'max_abs_delta_q', 'completed Q, every root labeled on both surfaces (decide_roots, depth2_backups); also the near-tie threshold',
            [side('merged Depth2Pricer (child_block 81)', d81, 'q'), side('DepthNPricer N=2 (as_depth2_frame)', n, 'q')], config=config, direct=(f'R2.{config}.e', 'max_abs_delta_q'))
        for name in r2_whole or DEPTH2_FLOATS:
            add('R2', 'n2.block_invariance', name, 'the whole Depth2Frame array (every element), child_block 27 against 81', [side('Depth2Pricer child_block 27', d27, f'whole.{name}'),
                side('Depth2Pricer child_block 81', d81, f'whole.{name}')], config=config, direct=(f'R2.{config}.child_block_invariance', name), premise=(f'R2.{config}.child_block_invariance', name))
        add('R2', 'n2.e_merged_depth2', 'max_abs_delta_q', 'completed Q, every root labeled (g2.decision_check on the merged Depth2Pricer against its oracle-heads copy); '
            'also the near-tie threshold', [side('merged Depth2Pricer (child_block 81)', d81, 'q')], 'mirror', config=config)
        for name, q, lv, oq in (('t1_value_head', 't1_value', 'depth2.turn1', 'value'), ('t1_own_logits', 't1_own_logits', 'depth2.turn1', 'own_logits'),
                ('t1_peer_logits', 't1_peer_logits', 'depth2.turn1', 'opp_logits'), ('root_peer_logits', 'root_peer_logits', 'root', 'opp_logits')):
            add('R2', 'n2.heads', name, 'turn-1 heads of the DepthNPricer N=2 against the oracle (reported, not gated)', [side('DepthNPricer N=2', n, q),
                side('the gate oracle', earlier=[('validation', lv, 'oracle_sample', oq)])], 'oracle_sample', report_only=True, config=config)
    return rows


DEPTHN_FLOATS = ('leaf_value', 'root_peer_logits', 't1_own_logits', 't1_peer_logits', 't1_value')   # gate.frame_arrays' floats of a depth-2 DepthNFrame
DEPTH2_FLOATS = ('root_peer_logits', 't1_own_logits', 't1_peer_logits', 't1_value', 't2_value')   # a Depth2Frame's float fields (g2.block_invariance)
REPORTED_ONLY = (dict(id='R1|R2|R3.c.floors', what='depth1_noise_floor and root_logit_floor (g2.FLOORS): measured, reported beside the bounds, never gated; '
        'under --declared-bounds they set no bound'),
    dict(id='R3.topk_boundary', what='per ply and seat the rank-k vs rank-(k+1) prior gap of each pruning selection and how many fall below the logit bound '
        '(merge_topk_n): reported, not gated (R1 at k 9 and k1 9 prunes nothing; --n2-equivalence computes none)'),
    dict(id='R1|R3.f.near_ties / R1|R2|R3.e.near_tie_roots', what='near-tie counts and flips among them: reported; the threshold is the site\'s bound (value / '
        'near_tie_q), whose gaps section is below'),
    dict(id='R3.mutation_test', what='MUTATION_RULE judges every call and mutant at the run\'s (declared) value bound: no noise site of its own (the clean call '
        'is the run\'s sites call by call, each call with its own guard spread); a larger value bound lowers its kill power (a sweep survivor is '
        'material beyond 2 x the value bound)'))


# ---------------------------------------------------------------------------------------------------------------- the fold
def resolve_side(spec, measurements, earlier):
    """One side's tail: the largest of its sources (this run's max + the adapter bound; the earlier reports' entries), withheld on any
    non-finite difference, unmeasured without a source."""
    candidates = [];gid, q = spec.get('geometry'), spec.get('quantity')
    g = (measurements or {}).get(gid) if gid else None
    if g is not None:
        m = g.get('q') if q == 'q' else (g.get('whole') or {}).get(q[len('whole.'):]) if q.startswith('whole.') else (g.get('quantities') or {}).get(q)
        if m and m.get('count'):
            a = m.get('adapter') if q == 'q' else adapter_for(q.removeprefix('whole.'))
            candidates.append(dict(source='this run', key=f'{gid}.{q}', max=None if m.get('max') is None else m['max'] + a, raw_max=m.get('max'), adapter=a,
                count=m['count'], nonfinite=m.get('nonfinite', 0)))
    for e in spec.get('earlier') or ():
        x = earlier_entry(earlier, e)
        if x['count']:candidates.append(x)
    out = dict(spec, candidates=candidates, geometry_measured=g is not None)
    if not candidates:return dict(out, status='unmeasured', tail=None, reason=f'no measurement: {gid or "an earlier report"} {q or ""} not in this run or its sources'.strip())
    bad = [c for c in candidates if c['nonfinite'] or c['max'] is None]
    if bad:return dict(out, status='withheld', tail=None, reason='; '.join(f'{c["source"]} {c["key"]}: {c["nonfinite"]} non-finite of {c["count"]}' for c in bad))
    top = max(candidates, key=lambda c: c['max']);return dict(out, status='measured', tail=top['max'], source=top['source'], key=top['key'])


def site_flags(row, measurements, direct):
    """Structural differences of the side geometries touching the site's quantities, failed premises and the direct comparison's own flags."""
    flags = []
    for s in row['sides']:
        g = (measurements or {}).get(s.get('geometry')) or {};q = s.get('quantity') or ''
        for kind, e in (g.get('structural') or {}).items():
            affects = e.get('affects');base = q.removeprefix('whole.').split('.')[0]
            hit = affects == 'all' or (isinstance(affects, (list, tuple)) and (q in affects or base in affects or ('whole' in affects and q.startswith('whole.'))))
            if hit and e.get('count'):flags.append(f'structural {s["geometry"]} {kind} {e["count"]}')
        if q.startswith('whole.') and (g.get('whole') or {}).get(q[6:], {}).get('padding_nonfinite'):
            flags.append(f'{s["geometry"]} {q}: {g["whole"][q[6:]]["padding_nonfinite"]} positions non-finite in both precisions with unequal bits (padding; excluded)')
    if row.get('premise'):
        key, name = row['premise'];e = ((direct or {}).get(key) or {}).get(REFERENCE) or {};m = (e.get('floats') or {}).get(name)
        bound = 2 * adapter_for(name)
        if m is not None and (m['nonfinite'] or m['max_abs_difference'] > bound):
            flags.append(f'premise {key} {name}: |A_f32 - B_f32| max {m["max_abs_difference"]:.3g} (non-finite {m["nonfinite"]}) > 2 x the adapter bound {bound:g}')
    return flags


def direct_value(row, direct):
    """The site measured on the bf16 frames here (DIRECT_RULE): dict(max, nonfinite, discrete_equal) or None."""
    if not row.get('direct'):return None
    key, name = row['direct'];e = ((direct or {}).get(key) or {}).get(TESTED)
    if not e:return None
    if 'max_abs_delta_q' in e:return dict(key=key, max=e['max_abs_delta_q'], nonfinite=e['decide']['nonfinite_roots'], status_differs=e['decide']['status_differs'],
        argmax_q_differs=e['decide']['argmax_q_differs'])
    if 'fields' in e:
        m = e['fields'].get(name) or {};return dict(key=key, max=m.get('max_abs_difference'), nonfinite=m.get('nonfinite'), unequal=m.get('unequal'), compared=m.get('compared'))
    m = (e.get('floats') or {}).get(name)
    if m is None:return None
    return dict(key=key, max=m['max_abs_difference'], nonfinite=m['nonfinite'], compared=m['compared'], discrete_equal=e['discrete']['passed'], discrete_mismatched=dict(e['discrete']['mismatched']))


def fold_sites(rows, measurements, direct, earlier):
    """Every site's tail (CONSTRUCTIONS), its status (measured, constructed, derived, withheld, unmeasured), flags and draft coverage."""
    out, by_id = [], {}
    for row in sorted(rows, key=lambda r: r['construction'] == 'derived'):
        sides = [resolve_side(s, measurements, earlier) for s in row['sides']];flags = site_flags(row, measurements, direct);tail, status = None, 'measured'
        c = row['construction'];tails = [s['tail'] for s in sides]
        if c == 'derived':
            src = by_id.get(row['derived']) or {};tail, status = src.get('tail'), 'derived' if src.get('tail') is not None else src.get('status', 'unmeasured')
        elif any(s['status'] == 'withheld' for s in sides):status = 'withheld'
        elif c == 'sum':
            if any(t is None for t in tails):status = 'unmeasured'
            else:tail = sum(tails)
        elif c == 'oracle_sample':
            p, o = tails
            if p is None:status = 'unmeasured'
            elif o is None:tail, status = 2 * p, 'constructed';flags.append('the oracle side is unmeasured: 2 x P')
            else:tail, status = max(p + o, 2 * p), 'constructed'
        elif c == 'mirror':
            tail, status = (None, 'unmeasured') if tails[0] is None else (2 * tails[0], 'constructed')
        d = direct_value(row, direct)
        if d is not None and tail is not None:
            if d.get('nonfinite'):status, tail = 'withheld', None;flags.append(f'direct {d["key"]}: {d["nonfinite"]} non-finite')
            elif d.get('max') is not None and d['max'] > tail:flags.append(f'direct {d["key"]} max {d["max"]:.6g} above the construction {tail:.6g}: raised to it');tail = d['max']
        if row['run'] == 'R3' and row['check'] == 'b' and not ((measurements or {}).get('R3.d3')):flags.append('D3 side not at R3\'s bucket (r3-d3 not measured)')
        unmeasured = [f'{s["label"]}: {s.get("reason")}' for s in sides if s['status'] == 'unmeasured']
        withheld = [f'{s["label"]}: {s.get("reason")}' for s in sides if s['status'] == 'withheld']
        draft = DRAFT.get(row['field']);r = dict(row, sides=sides, tail=tail, status=status, flags=flags, direct_measured=d, draft=draft,
            covered_by_draft=None if tail is None or draft is None else tail <= draft, unmeasured_reasons=unmeasured, withheld_reasons=withheld)
        out.append(r);by_id[row['id']] = r
    order = {row['id']: i for i, row in enumerate(rows)};return sorted(out, key=lambda r: order[r['id']])


def propose(sites, *, exclude=()):
    """PROPOSAL_RULE over the gated sites (exclude: ids left out): each field 1.25 x its largest tail rounded up, the setter named."""
    gated = [s for s in sites if not s['report_only'] and s['id'] not in exclude];out = {}
    for field in FIELDS:
        mine = [s for s in gated if s['field'] == field];withheld = [s['id'] for s in mine if s['status'] == 'withheld']
        measured = [s for s in mine if s['tail'] is not None];unmeasured = [s['id'] for s in mine if s['tail'] is None and s['status'] != 'withheld']
        top = max(measured, key=lambda s: s['tail']) if measured else None
        out[field] = dict(tail=None if top is None else top['tail'], set_by=None if top is None else top['id'], sites=len(mine), measured=len(measured),
            withheld=withheld, unmeasured=unmeasured, flagged=[s['id'] for s in mine if s['flags']], draft=DRAFT[field])
    value, dq = out['value_bound'], out['near_tie_q_bound']
    terms = [(x['tail'], f'{name}: {x["set_by"]}') for name, x in (('value tail', value), ('dQ tail', dq)) if x['tail'] is not None]
    if terms:t, by = max(terms);dq.update(tail=t, set_by=by, rule='1.25 x max(the value tail, the (e) dQ tail)')
    if value['withheld']:dq['withheld'] = dq['withheld'] + [f'value_bound: {x}' for x in value['withheld']]
    for field, x in out.items():
        x['proposed'] = None if x['withheld'] or x['tail'] is None else round_up(MARGIN * x['tail'])
        x['draft_covers'] = None if x['tail'] is None else x['tail'] <= x['draft']
    return out


def r1_invariance_decision(sites, direct):
    """R1_INVARIANCE_RULE: the proposal with and without R1's block-invariance sites; with it only when it is clean and raises nothing."""
    ids = [s['id'] for s in sites if s['run'] == 'R1' and s['check'] == 'block_invariance']
    inv = ((direct or {}).get('R1.block_invariance') or {})
    discrete = {name: (inv.get(name) or {}).get('discrete', {}).get('passed') for name in NAMES}
    with_, without = propose(sites), propose(sites, exclude=ids)
    raises = [f for f in FIELDS if with_[f]['proposed'] != without[f]['proposed']]
    clean = all(s['status'] in ('measured',) and not s['flags'] for s in sites if s['id'] in ids) and bool(ids)
    keep = discrete.get(TESTED) is True and clean and not raises
    reasons = [] if keep else ([] if discrete.get(TESTED) is True else [f'bf16 discrete arrays bit-equal 729 vs 2187: {discrete.get(TESTED)}']) + \
        ([] if clean else ['its sites are not all measured and unflagged']) + ([f'it raises {", ".join(raises)}'] if raises else [])
    return dict(block_invariance=keep, sites=ids, discrete_equal=discrete, raises=raises, reasons=reasons, proposal_with=with_, proposal_without=without, rule=R1_INVARIANCE_RULE)


def guard_section(sites, measurements):
    """GUARD_RULE: per run (R2 per config) the guard from the bf16 frames' spreads, and each guarded site's tail against it."""
    sources = dict(R1='R1.frame', R3='R3.frame');out = {}
    for run, gid in list(sources.items()) + [(f'R2.{gid.split(".")[1]}', gid) for gid in (measurements or {}) if gid.startswith('R2.') and gid.endswith('.depthn')]:
        g = (measurements or {}).get(gid) or {};spreads = g.get('spreads')
        if spreads is None:out[run] = dict(source=gid, measured=False);continue
        guard = spreads[TESTED]['guard'];rows = []
        for s in sites:
            if not s['guarded'] or (f'{s["run"]}.{s["config"]}' if s['config'] else s['run']) != run:continue
            rows.append(dict(site=s['id'], tail=s['tail'], direct=(s.get('direct_measured') or {}).get('max'), ratio=None if s['tail'] is None or not guard else s['tail'] / guard,
                within=None if s['tail'] is None or guard is None else s['tail'] < guard))
        out[run] = dict(source=gid, measured=True, p10_bf16=spreads[TESTED]['p10'], guard_bf16=guard, p10_f32=spreads[REFERENCE]['p10'], guard_f32=spreads[REFERENCE]['guard'],
            parents=spreads[TESTED].get('parents'), nonzero=spreads[TESTED].get('nonzero'), zero_share=spreads[TESTED].get('zero_share'),
            per_call_p10=spreads.get('per_call_bf16'), sites=rows, clean_run_passes=None if not rows or guard is None else all(r['within'] for r in rows if r['within'] is not None)
                and all(r['tail'] is not None for r in rows), note='informational (GUARD_RULE)')
    return out


def gaps_section(sites, measurements, proposal):
    """Item 4's gaps: per (e) site, the bf16 frame's top-two completed-Q gaps (p1, p5, p10, p50) and how many roots fall within each candidate."""
    out = {};near = (proposal.get('near_tie_q_bound') or {}).get('proposed')
    for s in sites:
        if s['check'] not in ('e', 'n2.e', 'n2.e_merged_depth2'):continue
        gid = s['sides'][0].get('geometry');g = (measurements or {}).get(gid) or {};gaps = np.asarray((g.get('gaps') or {}).get(TESTED) or [], np.float64)
        if not gaps.size:out[s['id']] = dict(geometry=gid, roots=0);continue
        candidates = dict(draft_near_tie=DRAFT['near_tie_q_bound'], proposed_near_tie=near, site_tail=s['tail'])
        out[s['id']] = dict(geometry=gid, roots=int(gaps.size), **{f'p{p}': float(np.percentile(gaps, p)) for p in (1, 5, 10, 50)}, min=float(gaps.min()),
            within={k: dict(bound=v, roots=int(np.count_nonzero(gaps <= v)), share=float(np.mean(gaps <= v))) for k, v in candidates.items() if v is not None})
    return out


def fold(measured, earlier, *, runs, header=None):
    """The table, the proposal, R1's block-invariance decision and the guard and gaps sections from the measurements (this run's and the
    pieces'), the earlier reports and the runs' arguments."""
    measurements, direct = measured.get('geometries') or {}, measured.get('direct') or {}
    r1_whole = sorted((measurements.get('R1.frame') or {}).get('whole') or ()) or None
    r2_whole = sorted(set().union(*[(g.get('whole') or {}).keys() for gid, g in measurements.items() if gid.startswith('R2.')])) or None
    sites = fold_sites(site_registry(runs, r1_whole=r1_whole, r2_whole=r2_whole), measurements, direct, earlier)
    decision = r1_invariance_decision(sites, direct)
    proposal = decision['proposal_with'] if decision['block_invariance'] else decision['proposal_without']
    unmeasured = [dict(site=s['id'], status=s['status'], reasons=s['unmeasured_reasons'] + s['withheld_reasons'], exact_coverage=s['exact_coverage'])
        for s in sites if s['tail'] is None]
    return dict(sites=sites, proposal=dict(fields=proposal, r1_block_invariance=decision['block_invariance'], label='PROPOSAL (informational, not a verdict)',
        margin=MARGIN, figures=FIGURES, rule=PROPOSAL_RULE), r1_invariance=decision, guard=guard_section(sites, measurements), gaps=gaps_section(sites, measurements, proposal),
        unmeasured=unmeasured, reported_only=list(REPORTED_ONLY), turn1_identity=direct.get('R3.turn1_identity'))


def merge_pieces(measured, pieces):
    """This run's measurements plus each piece's (--fold-from), every geometry once; returns (merged, flags)."""
    merged = dict(geometries=dict((measured or {}).get('geometries') or {}), direct=dict((measured or {}).get('direct') or {}));flags = []
    for path, piece in pieces:
        m = piece.get('measurements') or {}
        for section in ('geometries', 'direct'):
            for key, value in (m.get(section) or {}).items():
                if key in merged[section]:raise SystemExit(f'--fold-from: {key} is measured twice ({path} and another piece or this run)')
                merged[section][key] = value
    return merged, flags


def piece_checks(header, pieces):
    """Every piece against the run's header: source, roots, champion, split and calls; flags."""
    flags = []
    for path, piece in pieces:
        h = piece.get('header') or {}
        for key in ('source', 'roots', 'champion', 'split'):
            if h.get(key) != header.get(key):flags.append(f'piece {path}: {key} {h.get(key)!r} != {header.get(key)!r}')
        if (h.get('tree') or {}).get('commit') != (header.get('tree') or {}).get('commit'):flags.append(f'piece {path}: tree {(h.get("tree") or {}).get("commit")} != {(header.get("tree") or {}).get("commit")}')
        if piece.get('schema') != SCHEMA:flags.append(f'piece {path}: schema {piece.get("schema")!r}')
    return flags


def estimate(selected):
    return dict(minutes={g: ESTIMATE_MINUTES[g] for g in selected}, total_minutes=sum(ESTIMATE_MINUTES[g] for g in selected) + ESTIMATE_MINUTES['load'], rule=ESTIMATE_RULE)


# ---------------------------------------------------------------------------------------------------------------- report
def _num(x, fmt='.4g'):
    return clv._num(x, fmt)


def markdown(report):
    h = report.get('header') or {};lines = [f'# Champion 9375: the gate\'s float surfaces, bf16 vs f32 ({report.get("status")}, exit {report.get("exit")}; informational)', '',
        f'- Source `{json.dumps(h.get("source"), default=str)}`, roots {h.get("roots")}, tree `{(h.get("tree") or {}).get("commit")}`; groups {h.get("groups")}; pieces {len(h.get("pieces") or [])}',
        f'- Departures: {"; ".join(report.get("departures") or []) or "none"}', '']
    lines += ['## Measurements (this run and pieces): max |bf16 - f32|', '', '| geometry | quantity | n | non-finite | max | p99 | p99.9 | p99.99 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for gid, g in sorted(((report.get('measurements') or {}).get('geometries') or {}).items()):
        for name, m in list((g.get('quantities') or {}).items()) + [(f'whole.{k}', m) for k, m in (g.get('whole') or {}).items()] + ([('q', g['q'])] if g.get('q') else []):
            lines.append(f'| {gid} | {name} | {m.get("count")} | {m.get("nonfinite")} | {_num(m.get("max"))} | {_num(m.get("p99"))} | {_num(m.get("p99.9"))} | {_num(m.get("p99.99"))} |')
        if g.get('structural'):lines.append(f'| {gid} | structural | ' + '; '.join(f'{k} {e["count"]}' for k, e in g['structural'].items()) + ' | | | | | |')
    f = report.get('fold') or {};lines += ['', '## Every gated float site', '', '| site | field | positions | sides (tail, source) | construction | tail | direct | draft | covered | flags |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for s in f.get('sites') or []:
        if s['report_only']:continue
        sides = '; '.join(f'{x["label"]}: {_num(x.get("tail"))} (' + ' '.join(str(y) for y in (x.get('source', x.get('status')), x.get('key')) if y) + ')' for x in s['sides']) or f'= {s.get("derived")}'
        lines.append(f'| {s["id"]} | {s["field"]} | {s["positions"]} | {sides} | {s["construction"]} | {_num(s["tail"])} ({s["status"]}) | '
            f'{_num((s.get("direct_measured") or {}).get("max"))} | {s.get("draft")} | {s.get("covered_by_draft")} | {"; ".join(s["flags"]) or "-"} |')
    p = (f.get('proposal') or {}).get('fields') or {};lines += ['', f'## Proposal (PROPOSAL, not a verdict; R1 --block-invariance: {(f.get("proposal") or {}).get("r1_block_invariance")})', '',
        '| field | largest tail | set by | x 1.25, rounded up | draft | draft covers | withheld | unmeasured |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    lines += [f'| {k} | {_num(x["tail"])} | {x["set_by"]} | {_num(x["proposed"])} | {x["draft"]} | {x["draft_covers"]} | {len(x["withheld"])} | {len(x["unmeasured"])} |' for k, x in p.items()]
    r = f.get('r1_invariance') or {};lines += ['', f'- R1 block invariance: {"with" if r.get("block_invariance") else "without"} ({"; ".join(r.get("reasons") or []) or "clean, raises nothing"})']
    lines += ['', '## Guard (informational)', ''] + [f'- {run}: p10 {_num(x.get("p10_bf16"))}, guard {_num(x.get("guard_bf16"))} (f32 {_num(x.get("guard_f32"))}); clean run passes: '
        f'{x.get("clean_run_passes")}; ' + ', '.join(f'{y["site"]} {_num(y["ratio"], ".3g")}x' for y in x.get('sites') or []) for run, x in (f.get('guard') or {}).items() if x.get('measured')]
    lines += ['', '## (e) top-two gaps (bf16)', ''] + [f'- {k}: roots {x.get("roots")}, p1 {_num(x.get("p1"))}, p5 {_num(x.get("p5"))}, p10 {_num(x.get("p10"))}; within: ' +
        ', '.join(f'{n} {_num(y["bound"])}: {y["roots"]}' for n, y in (x.get('within') or {}).items()) for k, x in (f.get('gaps') or {}).items()]
    t = f.get('turn1_identity') or {};lines += ['', '## R3 turn 1 against the depth-2 frame', ''] + [f'- {name}: ' + ', '.join(f'{k} max {_num(v.get("max_abs_difference"))} unequal {v.get("unequal")}'
        for k, v in (x.get('fields') or {}).items()) for name, x in t.items()]
    lines += ['', '## Unmeasured or withheld sites', ''] + [f'- {x["site"]} ({x["status"]}): {"; ".join(x["reasons"]) or "-"}; exact: {x["exact_coverage"]}' for x in f.get('unmeasured') or []]
    lines += ['', f'- Flags: {"; ".join(report.get("flags") or []) or "none"}', '', 'Rules: gate_surfaces.json header.rules.', '']
    return '\n'.join(lines)


def write(out, report):
    out = Path(out);body = clv.plain(report)
    (out / 'gate_surfaces.json').write_text(json.dumps(body, indent=1, allow_nan=False) + '\n');(out / 'gate_surfaces.md').write_text(markdown(body))


def assemble(measured, *, earlier, runs, run_record, header, pieces=()):
    """The report body: the merged measurements, the fold, every flag, the status and exit."""
    merged, _ = merge_pieces(measured, pieces);checks = check_earlier(earlier, header)
    flags = [f'{k}: {x}' for k, v in checks.items() for x in v] + piece_checks(header, pieces)
    runs_only = {name: run_record[name] for name in RUN_ARGV if name in run_record}
    departures = list((measured or {}).get('departures') or []) + [x for r in runs_only.values() for x in r.get('departures') or []]
    departures += [f'{name}: refused by the gate\'s parser: {r["refused"]}' for name, r in runs_only.items() if r.get('refused')]
    for path, piece in pieces:departures += [f'piece {path}: {x}' for x in piece.get('departures') or []]
    calls = (measured or {}).get('calls') or {}
    if calls.get('selected') is not None:departures.append(f'calls subset {calls["selected"]} of {len(calls.get("sizes") or [])}: tails over those calls only')
    body = fold(merged, earlier, runs=runs, header=header)
    compared = sum((m.get('count') or 0) for g in merged['geometries'].values() for m in list((g.get('quantities') or {}).values()) + list((g.get('whole') or {}).values()))
    status = 'MEASURED' if compared else 'NOTHING_COMPARED'
    for s in body['sites']:
        if s['flags'] and not s['report_only']:flags.append(f'{s["id"]}: {"; ".join(s["flags"])}')
    for gid, g in merged['geometries'].items():
        if (g.get('q') or {}).get('crosscheck_mismatches'):flags.append(f'{gid}: recorded Q differs from decide_roots\' max |dQ|: {g["q"]["crosscheck_mismatches"]}')
    return dict(status=status, exit=EXIT[status], status_reasons=[] if compared else ['nothing was compared (no geometry priced, or no position in either precision)'],
        departures=departures, flags=flags, earlier_checks=checks, measurements=merged, fold=body)


def tree_commit():
    return d3g.tree_commit()


# ---------------------------------------------------------------------------------------------------------------- command line
def parse_calls(text):
    if text is None:return None
    out = set()
    for part in str(text).split(','):
        part = part.strip()
        if not part:continue
        a, _, b = part.partition('-');out.update(range(int(a), int(b or a) + 1))
    return out


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', help='the D3 bank run root (live mode; omit with --fold-from to fold pieces only)')
    parser.add_argument('--chunk', type=int, help='bank chunk index (with --d3-root)')
    parser.add_argument('--out', required=True, help='new output directory (gate_surfaces.json, gate_surfaces.md)')
    parser.add_argument('--validation-json', required=True, help='champion_leaf_validation\'s report (P_all, O_sample)')
    parser.add_argument('--d3-geometry-json', required=True, help='champion_d3_geometry\'s report (D3 and the pass at (b)\'s positions)')
    parser.add_argument('--fixed-shapes', action='store_true', help='as R1 and R3 run (their DepthNPricers at fixed_capacity); R2 never takes it')
    parser.add_argument('--measure', default=','.join(GROUPS), help=f'comma-separated groups of {GROUPS} (default all; r1-invariance needs r1)')
    parser.add_argument('--calls', help='only these call indices of the 512 split, e.g. 0,2,4-8 (a departure: tails over those calls only)')
    parser.add_argument('--r3-call-roots', type=int, default=R3_ROOTS, help='r3-d3: D3 call size (R3\'s saved roots: 40, bucket 64)')
    parser.add_argument('--fold-from', nargs='+', metavar='DIR', help='earlier pieces (their gate_surfaces.json) folded with this run, or alone without --d3-root')
    ce.add_arguments(parser);parser.set_defaults(memory_policy=False)
    args = parser.parse_args(argv)
    args.groups = tuple(g.strip() for g in args.measure.split(',') if g.strip())
    unknown = [g for g in args.groups if g not in GROUPS]
    if unknown:parser.error(f'--measure: unknown group(s) {unknown}; choose from {GROUPS}')
    missing = [f'{g} needs {NEEDS[g]}' for g in args.groups if g in NEEDS and NEEDS[g] not in args.groups]
    if missing:parser.error('--measure: ' + '; '.join(missing))
    try:args.call_set = parse_calls(args.calls)
    except ValueError:parser.error('--calls: indices and ranges, e.g. 0,2,4-8')
    if args.r3_call_roots < 1:parser.error('positive --r3-call-roots required')
    if args.d3_root is None:
        if not args.fold_from:parser.error('--d3-root (live) or --fold-from (offline fold) required')
        given = [n for n in ce.FLAGS + ('champion_precision',) if getattr(args, n)] + (['--calls'] if args.calls else []) + (['--chunk'] if args.chunk is not None else [])
        if given:parser.error(f'an offline fold prices nothing: drop {given}')
        return args
    if args.chunk is None or args.chunk < 0:parser.error('--chunk N >= 0 required with --d3-root')
    if not all(getattr(args, name) for name in ce.FLAGS):parser.error('--champion-archive, --champion-archive-sha256 and --champion-checkpoint-sha256 are required')
    for name in ce.FLAGS[1:]:
        if not ce._hex64(getattr(args, name)):parser.error(f'--{name.replace("_", "-")} must be a 64-hex sha256')
    if args.champion_precision is not None:parser.error('--champion-precision does not apply: this tool prices with both precisions (bf16 and f32)')
    return args


def _pieces(dirs):
    out = []
    for d in dirs or ():
        path = Path(d) / 'gate_surfaces.json';out.append((str(path), json.loads(path.read_text())))
    return out


def main(argv=None):
    args = parse(argv);started = time.perf_counter()
    earlier, hashes = {}, {}
    for name, path in (('validation', args.validation_json), ('d3_geometry', args.d3_geometry_json)):earlier[name], hashes[name] = read_json(path)   # fail before loading
    pieces = _pieces(args.fold_from);out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    runs, run_record = run_arguments(overrides=None if args.d3_root is None or args.fixed_shapes else dict(R1=dict(fixed_shapes=False), R1_invariance=dict(fixed_shapes=False),
        R3=dict(fixed_shapes=False)))
    log('GATE_SURFACES_RULES', rules=RULES, runs=run_record, groups=list(args.groups) if args.d3_root else [], calls=args.calls, estimate=estimate(args.groups) if args.d3_root else None,
        validation_json=args.validation_json, d3_geometry_json=args.d3_geometry_json, fold_from=args.fold_from)
    earlier_record = {name: dict(path=getattr(args, f'{name}_json'), sha256=hashes[name]) for name in earlier}
    if args.d3_root is None:   # offline fold: the first piece's header stands for the run, every piece's groups and pricer options merged
        header = dict((pieces[0][1].get('header') or {}), pieces=[p for p, _ in pieces], earlier=earlier_record, mode='FOLD_ONLY', tree=tree_commit(), runs=run_record,
            groups=[g for g in GROUPS if any(g in ((x.get('header') or {}).get('groups') or ()) for _, x in pieces)],
            pricer_options={k: o for _, x in pieces for k, o in ((x.get('header') or {}).get('pricer_options') or {}).items()},
            piece_trees={p: (x.get('header') or {}).get('tree') for p, x in pieces}, piece_calls={p: (x.get('header') or {}).get('calls') for p, x in pieces})
        measured = dict(geometries={}, direct={}, departures=[], calls={})
    else:
        with stage(log, 'load_chunk', d3_root=args.d3_root, chunk=args.chunk):loaded = g2.bank_chunk_requests(args)
        args.champion_precision = 'bf16';bf16 = g2.load_champion(args, loaded);evaluators = {REFERENCE: replace(bf16, precision='float32'), TESTED: bf16}
        clv.check_pair(evaluators);log('EVALUATORS', **{name: ev.header()['identity'] for name, ev in evaluators.items()})
        with stage(log, 'bind') as info:
            groups, root_values, dropped, checked = g2.bind_roots(loaded.groups, (bf16.params,), loaded.plan, evaluator=bf16);roots = sum(map(len, groups))
            info.update(roots=roots, dropped=len(dropped), arm_checked=checked)
        measured = measure(groups, evaluators=evaluators, executor=loaded.provider.executor, model=loaded.plan.model, materializer=loaded.provider.materializer,
            recipe=loaded.provider.recipe, frame_pricer=loaded.frame_pricer, runs=runs, selected=args.groups, calls=args.call_set, r3_call_roots=args.r3_call_roots, progress=log)
        import jax
        header = dict(mode='LIVE', source=loaded.source, roots=roots, split=SPLIT, binding='d3_depthn_gate.load_roots: g2.bind_roots under the bf16 champion; both precisions '
            'price these identical requests', dropped=dropped, root_abs_value=g2.percentiles(root_values),
            champion=dict(archive=args.champion_archive, archive_sha256=args.champion_archive_sha256, checkpoint_sha256=args.champion_checkpoint_sha256),
            evaluators={name: dict(ev.header(), archive=args.champion_archive) for name, ev in evaluators.items()}, tree=tree_commit(), runs=run_record,
            gate_cli_defaults=dict(bare=run_record.get('gate_cli_bare_defaults'), **{name: run_record[name].get('parsed') for name in RUN_ARGV if name in run_record}),
            pricer_options=measured['options'], groups=list(measured['groups']),
            calls=measured['calls'], fixed_shapes=args.fixed_shapes, r3_call_roots=args.r3_call_roots, estimate=estimate(measured['groups']), earlier=earlier_record,
            pieces=[p for p, _ in pieces], devices=[str(x) for x in jax.devices()])
        if pieces:header.update(piece_groups={p: (x.get('header') or {}).get('groups') for p, x in pieces},
            pricer_options=dict({k: o for _, x in pieces for k, o in ((x.get('header') or {}).get('pricer_options') or {}).items()}, **header['pricer_options']))
    body = assemble(measured, earlier=earlier, runs=runs, run_record=run_record, header=header, pieces=pieces)
    header['seconds'] = time.perf_counter() - started
    report = dict(schema=SCHEMA, status=body.pop('status'), exit=body.pop('exit'), status_reasons=body.pop('status_reasons'), informational=True,
        header=dict(header, rules=RULES), **body)
    write(out, report);p = report['fold']['proposal']['fields']
    log('CHAMPION_GATE_SURFACES_COMPLETE', status=report['status'], exit=report['exit'], out=str(out / 'gate_surfaces.json'), groups=header.get('groups'),
        proposal={k: dict(proposed=x['proposed'], tail=x['tail'], set_by=x['set_by'], draft=x['draft'], draft_covers=x['draft_covers'], withheld=len(x['withheld'])) for k, x in p.items()},
        r1_block_invariance=report['fold']['proposal']['r1_block_invariance'], unmeasured=len(report['fold']['unmeasured']), flags=len(report['flags']),
        departures=len(report['departures']), seconds=header['seconds'])
    return report['exit']


if __name__ == '__main__':raise SystemExit(main())
