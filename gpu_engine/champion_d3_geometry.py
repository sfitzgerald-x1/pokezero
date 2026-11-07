"""Gap 2, the third bf16 geometry: D3's FramePricer (the reference of the depth-N gate's check (b)) under champion 9375, bf16 against f32,
at every position (b) compares, plus a three-geometry bounds proposal (informational, never a verdict). Numpy-only import; JAX loads
after the bank (as the gates).

champion_leaf_validation measured the bf16 noise of the pricers (P_all, every position) and of the gate oracle (O_sample); (b) compares
the pricer's turn 1 against a third bf16 program, D3's FramePricer, whose batch geometry (2 x bucket root-world rows for the opponent's
prior, 81 x that for the joint leaves) is neither. This tool measures it.

Setup is champion_leaf_validation's, reused: --d3-root DIR --chunk N through d3_depth2_gate.bank_chunk_requests (the bank's provider and
its FramePricer), the champion through g2.load_champion (archive and checkpoint sha256, the real 9375, the bank catalog), one archive at
two precisions (bf16 under test, f32 the reference; champion_leaf_validation.check_pair), the roots bound by champion_leaf_validation.bind
(each evaluator binds them itself through g2.bind_evaluator_roots: its root prior, actors 0/0; the roots both keep, paired), calls split
at 512 (D3_RULE). D3 is built EXACTLY as d3_depthn_gate builds it for (b): g2.with_forward(the bank provider's FramePricer, champion), one
pricer per precision; its options, each call's compile bucket and any departure from the gate's are recorded (d3_options).

1. comparison (POSITIONS_RULE): per call, D3 at bf16 on the bf16-bound requests (exactly the gate's (b) under a bf16 champion) and at f32
   on the f32-bound ones (as the validation tool's f32 pricers; a root's prior moves only D3's pricing ORDER, i.e. which joint-batch row
   holds a cell, never which cells are priced or their inputs; roots whose order differs are counted). d3.t1_value: every (q, world, a, r)
   with a non-None joint value in BOTH runs; d3.root_peer_logits: every legal lane (legal in both) of reply_logits on worlds both runs
   priced with the opponent requested in both. Each: n, non-finite count, max, p99, p99.9, p99.99 of |bf16 - f32|. Any structural
   difference (root statuses, refused worlds and their reasons, opponent request, legal masks, None patterns, D's error codes, request
   pairing) is listed by kind with its count and first cells, never dropped. Informational: d3.root_peer_logits.all_lanes, every lane of
   every world both runs priced, which is what g2.check_d actually reads for root_peer_logits (all nine lanes, requested or not).
2. D_all (D_ALL_RULE): the all-positions max |D3 bf16 - D3 f32| + the declared f32 adapter bound (1e-5 values, 1e-4 logits), through
   champion_leaf_validation.pricer_all, the construction of P_all.
3. f32 anchor (ANCHOR_RULE): the declared adapter bound checked for D3's f32 against the f32 raw forward champion_leaf_validation uses
   (champion.forward, HIGHEST, on the ladder's own champion layout of an independent walk: champion_leaf_validation.Walker and Heads, the
   depth-N oracle's step), on a seeded sample per call (--sample turn-1 cells and root worlds); observed maxima beside the bounds, flagged
   when exceeded, with terminal, walk and layout disagreements listed.
4. proposal (FOLD_RULE): the (b) site tails from the earlier report (--validation-json) and D_all, folded into the all-positions value
   and root-logit proposals; the earlier report's source, evaluators, precision and shapes are checked against this run, any mismatch
   flagged. `margin` is left empty: the declaration adds it.
d3_geometry.json and d3_geometry.md in --out. Exit 0 when anything was compared, 2 when nothing was (no verdict either way).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from . import champion_evaluator as ce, champion_leaf_validation as clv, d3_depth2_gate as g2

SCHEMA = 'pokezero.champion-d3-geometry.v1'
REFERENCE, TESTED, NAMES = clv.REFERENCE, clv.TESTED, clv.NAMES
VALUE, LOGITS, ALL_LANES = 'd3.t1_value', 'd3.root_peer_logits', 'd3.root_peer_logits.all_lanes'
QUANTITIES = (VALUE, LOGITS)
SAMPLE = 512           # anchor rows per call and kind (turn-1 cells; root worlds): one walk chunk each
SEED, CHUNK, FIRST = clv.SEED, clv.CHUNK, g2.FIRST
CAPACITY = 512         # the gate's call split: g2.call_capacity(FramePricer 4096, the depth-2 pass at d3_depthn_gate's default --capacity 512)
GATE_D3 = dict(capacity=4096, block_rows=32, vectorized=True, host_prep=True)   # the bank provider's FramePricer, kept by g2.with_forward
EXIT = dict(MEASURED=0, NOTHING_COMPARED=2)
log, stage = g2.log, g2.stage

D3_RULE = ('D3\'s FramePricer as d3_depthn_gate builds it for (b): g2.with_forward(loaded.frame_pricer, champion), loaded.frame_pricer '
    'being the bank provider\'s (v5_device_teacher.Provider(vectorized=True, batched_prep=True): capacity 4096, block_rows 32, actor_capacity '
    'historical_window + 1, vectorized, host_prep) and with_forward keeping every option and adding forward=. Calls split at 512 '
    '(g2.call_capacity(FramePricer 4096, the depth-2 pass at --capacity 512); d3_depthn_gate.run_gate); each call compiles at its own '
    'bucket min(4096, the next power of two of its roots) (vectorized FramePricer.price) with 2 x bucket opponent-prior rows and 162 x bucket '
    'joint-leaf rows; --fixed-shapes leaves D3 unchanged (d3_depthn_gate.FIXED_SHAPES: D3 keeps its per-call buckets) and here fixes the '
    'anchor walk\'s stack only. One pricer per precision: bf16 (the gate\'s) and f32 (the reference, the same options).')
POSITIONS_RULE = ('d3.t1_value: every (q, world, a, r) whose joint value is non-None in BOTH runs (g2.check_d reads each such cell); '
    'd3.root_peer_logits: the lanes legal in both runs of reply_logits, on worlds both runs priced with the opponent requested in both. '
    '|bf16 - f32| per position: n, non-finite count, max, p99, p99.9, p99.99. Structural differences are listed by kind, never dropped. '
    'd3.root_peer_logits.all_lanes (informational, not folded): every lane of every world both priced, as g2.check_d reads root_peer_logits '
    '(all nine lanes); P_all covers legal lanes of requested worlds only, so no all-lane (b) tail can be composed from the earlier report.')
D_ALL_RULE = ('D_all = the all-positions max |D3 bf16 - D3 f32| + the declared f32 adapter bound (1e-5 values, 1e-4 logits), built by '
    'champion_leaf_validation.pricer_all exactly as P_all: |D3 bf16 - f32 raw| <= |D3 bf16 - D3 f32| + |D3 f32 - f32 raw|.')
ANCHOR_RULE = ('The adapter bound\'s premise for D3 (|D3 f32 - f32 raw| <= 1e-5 values, 1e-4 logits) checked against the f32 raw forward '
    'champion_leaf_validation uses: champion.forward (f32, HIGHEST, its own jit) on the ladder\'s champion layout of an independent walk '
    '(champion_leaf_validation.Walker: the depth-N oracle\'s step, D\'s turn-1 joint rule, the node\'s own chance; Heads). A seeded sample per '
    'call: --sample turn-1 cells both runs value and --sample root worlds both runs priced with the opponent requested. Values compare where '
    'the walk is a head-valued child (not terminal, invalid or in error on either layout); reply logits on D3\'s legal lanes where both of the '
    'walk\'s root views are error-free. Terminal cells compare D3\'s value with the walk\'s terminal (+1 / -1 / 0 from the owner) exactly. '
    'Flagged: a max above its bound, non-finite differences, no compared row, a valued cell the walk disagrees on, a terminal mismatch, a '
    'champion layout unequal to an evaluator\'s view, and a bf16 sample max above D_all (the all-positions bound failing at the sample). '
    'Cheapest faithful anchor: the walk re-steps only sampled cells; the pricers\' f32 turn 1 would cost a full depth-2 f32 pass per call.')
FOLD_RULE = ('PROPOSAL, not a verdict; `margin` empty (the declaration adds it). (b) site tails, all positions: b.t1_value = P_all(depth2.turn1 '
    'value) + D_all(d3.t1_value); b.root_peer_logits = P_all(root opp_logits) + D_all(d3.root_peer_logits), by the triangle inequality through '
    'the f32 raw forward. value_bound_all_geometries = max(the earlier all_positions value_bound, b.t1_value) (and without the root level: '
    'max(value_bound_without_root, b.t1_value)); near_tie_q_bound_all_geometries = value_bound_all_geometries (the earlier rule: near_tie_q = '
    'value); root_logit_all_geometries = max(the root (c) tail max(P_all(root opp) + O_sample(root opp), 2 x P_all(root opp)), '
    'b.root_peer_logits). turn1_logit_bound and deeper_logit_bound are the earlier report\'s, unchanged. A tail with an unmeasured or '
    'non-finite input has no bound (None, flagged). The earlier report\'s source (d3_root, chunk, declaration), evaluators (archive and '
    'checkpoint sha256, precision), roots and shapes are checked against this run; every mismatch is flagged.')
RULES = dict(d3=D3_RULE, positions=POSITIONS_RULE, d_all=D_ALL_RULE, anchor=ANCHOR_RULE, fold=FOLD_RULE,
    bounds=dict(f32_adapter_value=clv.ADAPTER_VALUE, f32_adapter_logit=clv.ADAPTER_LOGIT), exit=EXIT)


# D3 as the gate builds it ---------------------------------------------------------------------------------------------------------
def build_d3(frame_pricer, evaluators):
    """{name: FramePricer}: the gate's construction for (b), g2.with_forward(the bank provider's FramePricer, evaluator), per precision."""
    return {name: g2.with_forward(frame_pricer, evaluators[name]) for name in NAMES}


def bucket(pricer, roots):
    """The root capacity a vectorized FramePricer.price compiles one call of `roots` requests at (its own rule)."""
    return min(pricer.capacity, 1 << (roots - 1).bit_length()) if pricer.vectorized else pricer.capacity


def d3_options(pricers, *, actor_capacity=None, split_capacity=CAPACITY, fixed_shapes=True):
    """Each pricer's options against the gate's (GATE_D3, actor_capacity = the plan's historical_window + 1 when known, the champion at the
    pricer's precision, the 512 split, --fixed-shapes); `departures` names every difference, `notes` what could not be checked."""
    record = {name: dict(capacity=p.capacity, block_rows=p.block_rows, actor_capacity=p.actor_capacity, vectorized=p.vectorized, host_prep=p.host_prep,
        evaluator=ce.describe(getattr(p, 'forward', None))) for name, p in pricers.items()}
    departures, notes = [], []
    for name, r in record.items():
        departures += [f'{name}: {k} {r[k]!r} != the gate\'s {v!r}' for k, v in GATE_D3.items() if r[k] != v]
        if actor_capacity is None:notes.append(f'{name}: actor_capacity {r["actor_capacity"]} not checked (the plan\'s historical_window is unknown)')
        elif r['actor_capacity'] != actor_capacity:departures.append(f'{name}: actor_capacity {r["actor_capacity"]!r} != the gate\'s {actor_capacity!r}')
        identity = r['evaluator']['identity']
        if r['evaluator']['kind'] != 'champion' or identity[-1] != clv.PRECISION_OF.get(name):
            departures.append(f'{name}: evaluator {identity} is not the champion at {clv.PRECISION_OF.get(name)}')
    if split_capacity != CAPACITY:departures.append(f'call split {split_capacity} != the gate\'s {CAPACITY}')
    if not fixed_shapes:departures.append('--fixed-shapes not given: the gate\'s (b) runs under it (D3 itself is unchanged by it; here it fixes the anchor walk\'s stack only)')
    return dict(pricers=record, gate=dict(GATE_D3, actor_capacity=actor_capacity, split_capacity=CAPACITY, fixed_shapes=True), split_capacity=split_capacity,
        fixed_shapes=fixed_shapes, departures=departures, matches_gate=not departures, notes=notes, rule=D3_RULE)


# 1. The comparison -----------------------------------------------------------------------------------------------------------------
def _note(findings, name, cells, f32, bf16, *, call, addresses):
    """One structural difference kind: its count and the first FIRST cells (call, cell, root address, each run's value)."""
    cells = np.asarray(cells, np.int64).reshape(len(cells), -1) if len(cells) else np.zeros((0, 1), np.int64)
    if not len(cells):return
    e = findings.setdefault(name, dict(count=0, first=[]));e['count'] += len(cells)
    for i in range(min(len(cells), FIRST - len(e['first']))):
        cell = cells[i].tolist();e['first'].append(dict(call=call, cell=cell, address=list(addresses[cell[0]]), f32=f32[i], bf16=bf16[i]))


def compare_call(results, requests, *, recipe=None, call=0):
    """bf16 against f32 on one call: results {name: D3 Results}, requests {name: Requests} (bind's pairing). Returns the signed differences
    (values, logits, all_lanes), the structural findings {kind: dict(count, first)}, counts, the roots whose pricing order differs and the
    Results as g2._d_arrays (the gate's array form of what check (b) reads)."""
    f, b = REFERENCE, TESTED
    if not len(requests[f]) == len(requests[b]) == len(results[f]) == len(results[b]):
        raise ValueError('one D3 Result per request, the two request lists paired')
    A, B = g2._d_arrays(results[f]), g2._d_arrays(results[b]);n = len(requests[f]);findings = {}
    addresses = [clv.address(r) for r in requests[f]];other = [clv.address(r) for r in requests[b]]
    note = lambda name, cells, x, y: _note(findings, name, cells, x, y, call=call, addresses=addresses)
    bad = [q for q in range(n) if addresses[q] != other[q]];note('request_address', [[q] for q in bad], [list(addresses[q]) for q in bad], [list(other[q]) for q in bad])
    lf, lb = (np.asarray([r.root_legal for r in requests[k]], bool).reshape(-1, 9) for k in (f, b))
    bad = np.flatnonzero(np.any(lf != lb, axis=1));note('request_root_legal', bad[:, None], lf[bad], lb[bad])
    bad = np.flatnonzero(A['d_status'] != B['d_status']);note('root_status', bad[:, None], A['d_status'][bad], B['d_status'][bad])
    at = np.argwhere(A['d_priced'] != B['d_priced']);x, y = (tuple(D['d_reason'][q, w] if not D['d_priced'][q, w] else 'priced' for q, w in at) for D in (A, B))
    note('world_refused', at, x, y)
    at = np.argwhere(~A['d_priced'] & ~B['d_priced'] & (A['d_reason'] != B['d_reason']));note('refusal_reason', at, A['d_reason'][tuple(at.T)], B['d_reason'][tuple(at.T)])
    both = A['d_priced'] & B['d_priced']
    at = np.argwhere(both & (A['d_opponent_requested'] != B['d_opponent_requested']))
    note('opponent_requested', at, A['d_opponent_requested'][tuple(at.T)], B['d_opponent_requested'][tuple(at.T)])
    at = np.argwhere(both & np.any(A['d_reply_legal'] != B['d_reply_legal'], axis=-1));note('reply_legal', at, A['d_reply_legal'][tuple(at.T)], B['d_reply_legal'][tuple(at.T)])
    grid = both[..., None, None] & np.ones((1, 1, 9, 9), bool)
    at = np.argwhere(grid & (A['d_known'] != B['d_known']));note('joint_value_none_pattern', at, A['d_known'][tuple(at.T)], B['d_known'][tuple(at.T)])
    at = np.argwhere(grid & (A['d_error'] != B['d_error']));note('joint_error', at, A['d_error'][tuple(at.T)], B['d_error'][tuple(at.T)])
    cells = grid & A['d_known'] & B['d_known'];values = B['d_value'][cells].astype(np.float64) - A['d_value'][cells]
    requested = both & A['d_opponent_requested'] & B['d_opponent_requested'];lanes = requested[..., None] & A['d_reply_legal'] & B['d_reply_legal']
    logits = B['d_reply_logits'][lanes].astype(np.float64) - A['d_reply_logits'][lanes]
    all_lanes = (B['d_reply_logits'][both].astype(np.float64) - A['d_reply_logits'][both]).reshape(-1)
    order = []
    if recipe is not None:
        order = [q for q, (u, v) in enumerate(zip(requests[f], requests[b])) if recipe.pricing_order(u.plans.root.address, u.root_logits, u.root_legal)
            != recipe.pricing_order(v.plans.root.address, v.root_logits, v.root_legal)]
    counts = dict(roots=n, worlds=2 * n, priced_f32=int(A['d_priced'].sum()), priced_bf16=int(B['d_priced'].sum()), priced_both=int(both.sum()),
        requested_both=int(requested.sum()), cells_f32=int(A['d_known'].sum()), cells_bf16=int(B['d_known'].sum()), cells_compared=int(cells.sum()),
        lanes_compared=int(lanes.sum()), all_lanes_compared=int(all_lanes.size))
    return dict(values=values, logits=logits, all_lanes=all_lanes, findings=findings, counts=counts, pricing_order_differs=order,
        addresses=[list(addresses[q]) for q in order[:FIRST]], arrays={f: A, b: B})


def merge_findings(parts):
    out = {}
    for findings in parts:
        for name, e in findings.items():
            m = out.setdefault(name, dict(count=0, first=[]));m['count'] += e['count'];m['first'] = (m['first'] + e['first'])[:FIRST]
    return out


def merge_comparison(calls):
    """The run's comparison over every call: the two quantities' spreads (and the informational all-lane one), the structural findings,
    the counts and the pricing-order note."""
    cat = lambda key: np.concatenate([c['comparison'][key] for c in calls]) if calls else np.zeros(0)
    findings = merge_findings([c['comparison']['findings'] for c in calls])
    counts = {k: sum(c['comparison']['counts'][k] for c in calls) for k in (calls[0]['comparison']['counts'] if calls else ())}
    order = [dict(call=c['call'], roots=len(c['comparison']['pricing_order_differs']), first=c['comparison']['addresses']) for c in calls
        if c['comparison']['pricing_order_differs']]
    return dict(quantities={VALUE: clv.spread(cat('values')), LOGITS: clv.spread(cat('logits'))}, informational={ALL_LANES: clv.spread(cat('all_lanes'))},
        structural=dict(count=sum(e['count'] for e in findings.values()), kinds=findings, rule='every structural difference between the bf16 and '
            'f32 runs, by kind (count and first cells: [q, world(, a, r)] within the call); none is dropped'), counts=counts,
        pricing_order=dict(roots=sum(x['roots'] for x in order), by_call=order, note='informational, not a structural difference: the two runs bind '
            'their own root prior, which orders D3\'s joint-batch rows; the grid is (action, reply)-indexed and the priced cells are the same'),
        rule=POSITIONS_RULE)


# 2. D_all ----------------------------------------------------------------------------------------------------------------------------
def d_all(quantities):
    """D_all per quantity: champion_leaf_validation.pricer_all, P_all's construction (all-positions max + the declared f32 adapter bound)."""
    precision = dict(values={'d3': quantities.get(VALUE) or {}}, logits={'d3.opp': quantities.get(LOGITS) or {}})
    return {VALUE: clv.pricer_all(precision, 'd3', 'value'), LOGITS: clv.pricer_all(precision, 'd3', 'opp_logits')}


# 3. The f32 anchor ----------------------------------------------------------------------------------------------------------------
CELL_FIELDS = dict(raw='raw_value', invalid='invalid', done='done', broken='broken', winner='winner', own_error='own_error',
    champion_own_error='champion_own_error', seat='seat')
ROOT_FIELDS = dict(raw='raw_peer_logits', invalid='invalid', done='done', own_error='own_error', champion_own_error='champion_own_error',
    champion_peer_error='champion_peer_error')


def _walked(walker, stack, spec, plies, heads, fields):
    """The walk's fields at `spec` (rows in spec order), empty arrays without a spec."""
    w = walker.run(stack, spec, plies, heads) if len(spec) else {}
    out = {k: np.asarray(w[v]) for k, v in fields.items()} if w else {}
    for name in NAMES:out[f'inputs_equal_{name}'] = np.asarray(w[f'inputs_equal_{name}'], bool) if w else np.zeros(0, bool)
    return out


def anchor_call(call, arrays, stack, walker, heads, *, sample=SAMPLE, rng=None):
    """One call's anchor rows: a seeded sample of turn-1 cells both runs value (walked one ply) and of root worlds both runs priced with the
    opponent requested (walked zero plies), each with both runs' D3 values / reply logits and the walk's raw f32 forward and facts."""
    rng = np.random.default_rng([SEED, call]) if rng is None else rng
    A, B = arrays[REFERENCE], arrays[TESTED];rows = set(stack.rows)
    kept = lambda at: at[np.asarray([(int(q), int(w)) in rows for q, w in at[:, :2]], bool)] if len(at) else at
    cells = kept(np.argwhere(A['d_known'] & B['d_known']));worlds = kept(np.argwhere(A['d_priced'] & B['d_priced'] & A['d_opponent_requested'] & B['d_opponent_requested']))
    cells = cells[clv._choose(rng, np.arange(len(cells)), sample)].astype(np.int64).reshape(-1, 4)
    worlds = worlds[clv._choose(rng, np.arange(len(worlds)), sample)].astype(np.int64).reshape(-1, 2)
    at, wt = tuple(cells.T), tuple(worlds.T)
    c = dict(spec=cells, f32=A['d_value'][at], bf16=B['d_value'][at], **_walked(walker, stack, cells, 1, heads, CELL_FIELDS))
    r = dict(spec=worlds, f32=A['d_reply_logits'][wt].reshape(-1, 9), bf16=B['d_reply_logits'][wt].reshape(-1, 9),
        legal=(A['d_reply_legal'][wt] & B['d_reply_legal'][wt]).reshape(-1, 9), **_walked(walker, stack, worlds, 0, heads, ROOT_FIELDS))
    return dict(call=call, cells=c, roots=r)


def _concat(parts, key):
    parts = [p[key] for p in parts if p[key] and len(p[key]['spec'])]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def anchor_section(records, d=None):
    """ANCHOR_RULE over every call's anchor rows; d: D_all (the bf16 sample max is checked against it)."""
    cells, roots = _concat(records, 'cells'), _concat(records, 'roots');flags, findings = [], {}
    out = dict(reference='f32 raw forward: champion.forward (float32, HIGHEST, its own jit) on the ladder\'s champion layout of an independent walk '
        '(champion_leaf_validation.Walker / Heads)', sampled=True, rule=ANCHOR_RULE)
    quantity = {}
    if cells is not None:
        terminal = cells['done'] & ~cells['invalid'] & ~cells['broken']
        ok = ~cells['done'] & ~cells['invalid'] & ~cells['broken'] & (cells['own_error'] == 0) & (cells['champion_own_error'] == 0)
        expected = np.where(cells['winner'] == 2, 0., np.where(cells['winner'] == cells['seat'], 1., -1.)).astype(np.float32)
        findings['cells_walk_disagrees'] = int(np.count_nonzero(~ok & ~terminal))
        for name in NAMES:
            findings[f'terminal_mismatch_{name}'] = int(np.count_nonzero(terminal & (cells[name] != expected)))
            findings[f'cells_inputs_unequal_{name}'] = int(np.count_nonzero(ok & ~cells[f'inputs_equal_{name}']))
        quantity[VALUE] = dict(rows=int(len(ok)), compared=int(ok.sum()), terminal=int(terminal.sum()), f32=clv.spread(cells['f32'][ok].astype(np.float64) - cells['raw'][ok]),
            bf16=clv.spread(cells['bf16'][ok].astype(np.float64) - cells['raw'][ok]))
    else:quantity[VALUE] = dict(rows=0, compared=0, terminal=0, f32=clv.spread(np.zeros(0)), bf16=clv.spread(np.zeros(0)))
    if roots is not None:
        ok = ~roots['done'] & ~roots['invalid'] & (roots['own_error'] == 0) & (roots['champion_own_error'] == 0) & (roots['champion_peer_error'] == 0)
        findings['roots_walk_disagrees'] = int(np.count_nonzero(~ok))
        for name in NAMES:findings[f'roots_inputs_unequal_{name}'] = int(np.count_nonzero(ok & ~roots[f'inputs_equal_{name}']))
        lanes = roots['legal'] & ok[:, None]
        quantity[LOGITS] = dict(rows=int(len(ok)), compared=int(ok.sum()), f32=clv.spread((roots['f32'].astype(np.float64) - roots['raw'])[lanes]),
            bf16=clv.spread((roots['bf16'].astype(np.float64) - roots['raw'])[lanes]))
    else:quantity[LOGITS] = dict(rows=0, compared=0, f32=clv.spread(np.zeros(0)), bf16=clv.spread(np.zeros(0)))
    verified = {}
    for key, bound in ((VALUE, clv.ADAPTER_VALUE), (LOGITS, clv.ADAPTER_LOGIT)):
        q = quantity[key];m = q['f32'];q.update(bound=bound, max=m.get('max'), within=None if not m.get('count') else (m.get('max', 0.) <= bound and not m.get('nonfinite')))
        if not m.get('count'):flags.append(f'{key}: not verified: no anchor row compared')
        if m.get('nonfinite'):flags.append(f'{key}: {m["nonfinite"]} non-finite D3 f32 vs raw differences')
        if m.get('count') and m.get('max', 0.) > bound:flags.append(f'{key}: EXCEEDED: D3 f32 vs the f32 raw forward max {m["max"]:.3g} > the declared bound {bound}')
        top = (d or {}).get(key, {}).get('max')
        if top is not None and q['bf16'].get('max') is not None and q['bf16']['max'] > top:
            flags.append(f'{key}: the sample max |D3 bf16 - f32 raw| {q["bf16"]["max"]:.4g} exceeds D_all {top:.4g}')
        verified[key] = bool(q['within'])
    flags += [f'{k}: {v}' for k, v in findings.items() if v]
    return dict(out, quantities=quantity, findings=findings, verified=verified, flags=flags,
        observed=dict(f32_value_max=quantity[VALUE]['max'], f32_value_bound=clv.ADAPTER_VALUE, f32_logit_max=quantity[LOGITS]['max'], f32_logit_bound=clv.ADAPTER_LOGIT))


# 4. The proposal -----------------------------------------------------------------------------------------------------------------------
def source_check(earlier, *, source, evaluators, roots, capacity=CAPACITY, fixed_shapes=True, path=None):
    """The earlier validation report against this run: schema, source (d3_root, chunk, declaration), evaluators (archive and checkpoint sha256,
    precision), the geometry's evaluator, roots, max_roots and shapes (fixed_shapes, capacity, the gate's shapes). `flags` lists each mismatch."""
    if not isinstance(earlier, dict):return dict(path=path, matches=False, flags=['no earlier validation report'])
    h, geo = earlier.get('header') or {}, earlier.get('geometry') or {};s, shapes = h.get('source') or {}, geo.get('shapes') or {};flags = []
    if earlier.get('schema') != clv.SCHEMA:flags.append(f'schema {earlier.get("schema")!r} != {clv.SCHEMA!r}')
    for k in ('d3_root', 'chunk'):
        if s.get(k) != (source or {}).get(k):flags.append(f'source {k} {s.get(k)!r} != this run\'s {(source or {}).get(k)!r}')
    if s.get('declaration_sha256') != (source or {}).get('declaration_sha256'):
        flags.append(f'source declaration_sha256 {s.get("declaration_sha256")!r} != this run\'s {(source or {}).get("declaration_sha256")!r}')
    for name, ev in evaluators.items():
        e, mine = (h.get('evaluators') or {}).get(name) or {}, ev.header()
        flags += [f'evaluator {name} {k} {e.get(k)!r} != this run\'s {mine[k]!r}' for k in ('archive_sha256', 'checkpoint_sha256', 'precision') if e.get(k) != mine[k]]
    if geo.get('evaluator') != TESTED:flags.append(f'geometry evaluator {geo.get("evaluator")!r} != {TESTED!r}')
    if h.get('roots') != roots:flags.append(f'roots {h.get("roots")!r} != this run\'s {roots!r}')
    if (h.get('options') or {}).get('max_roots') is not None:flags.append(f'the earlier run priced only --max-roots {h["options"]["max_roots"]}')
    if shapes.get('fixed_shapes') != fixed_shapes:flags.append(f'fixed_shapes {shapes.get("fixed_shapes")!r} != this run\'s {fixed_shapes!r}')
    if shapes.get('capacity') != capacity:flags.append(f'capacity {shapes.get("capacity")!r} != this run\'s {capacity!r}')
    if not shapes.get('matches_gate'):flags.append('the earlier pricer column does not describe the gate\'s shapes: ' + ('; '.join(shapes.get('flags') or []) or 'not recorded'))
    if not geo.get('levels'):flags.append('no geometry levels (P_all, O_sample) in the earlier report')
    if not (geo.get('proposals') or {}).get('all_positions'):flags.append('no all_positions proposal in the earlier report')
    return dict(path=path, matches=not flags, flags=flags, earlier=dict(schema=earlier.get('schema'), source=s, roots=h.get('roots'), shapes={k: shapes.get(k)
        for k in ('fixed_shapes', 'capacity', 'block_lanes', 'depth3_k', 'oracle_chunk', 'matches_gate')}))


def _entry(levels, label, column, quantity):
    m = (((levels.get(label) or {}).get(column) or {}).get(quantity)) or {}
    return dict(level=label, column=column, quantity=quantity, max=m.get('max') if m.get('count') else None, count=m.get('count', 0), nonfinite=m.get('nonfinite', 0))


def _tail(name, p, d):
    """A (b) site tail P_all + D_all; None (with the reasons) when either is unmeasured or has non-finite differences."""
    why = [f'{name}: {label} unmeasured' for label, x in (('P_all', p), ('D_all', d)) if not x.get('count') or x.get('max') is None]
    why += [f'{name}: {label} has {x["nonfinite"]} non-finite differences' for label, x in (('P_all', p), ('D_all', d)) if x.get('nonfinite')]
    return dict(tail=None if why else p['max'] + d['max'], p_all=p, d_all=dict(quantity=d.get('quantity'), max=d.get('max'), count=d.get('count', 0),
        nonfinite=d.get('nonfinite', 0), precision_max=d.get('precision_max'), f32_adapter_bound=d.get('f32_adapter_bound'))), why


def fold(earlier, d, check=None):
    """FOLD_RULE: the (b) site tails and the bounds with them folded in, from the earlier report's geometry section and D_all."""
    geo = (earlier or {}).get('geometry') or {} if isinstance(earlier, dict) else {};levels = geo.get('levels') or {}
    allpos = (geo.get('proposals') or {}).get('all_positions') or {};flags = [f'source: {x}' for x in (check or {}).get('flags', [])]
    flags += [f'earlier all_positions: {x}' for x in allpos.get('flags') or []]
    t1, f1 = _tail('b.t1_value', _entry(levels, 'depth2.turn1', 'pricer_all', 'value'), dict(d.get(VALUE) or {}, quantity=VALUE))
    root, f2 = _tail('b.root_peer_logits', _entry(levels, 'root', 'pricer_all', 'opp_logits'), dict(d.get(LOGITS) or {}, quantity=LOGITS));flags += f1 + f2
    t3 = _entry(levels, 'depth3.turn1', 'pricer_all', 'value')
    if t3['max'] is not None and t1['p_all']['max'] is not None and t3['max'] > t1['p_all']['max']:
        flags.append(f'P_all(depth3.turn1 value) {t3["max"]:.4g} > P_all(depth2.turn1 value) {t1["p_all"]["max"]:.4g}: the depth-N gate\'s (b) reads the depth-N frame\'s turn 1')
    if 'root' in levels:
        view = {'root': {c: (levels['root'].get(c) or {}) for c in ('pricer_all', 'oracle_sample')}}
        c_bound, c_flags, c_inputs = clv._bound(view, (('root', 'opp_logits'),), 'pricer_all', clv.COMBINE['all_positions'][1])
    else:c_bound, c_flags, c_inputs = None, ['no root level in the earlier report'], []
    flags += [f'root (c) tail: {x}' for x in c_flags]
    def top(name, *xs):
        if any(x is None for x in xs):flags.append(f'{name}: no bound (an input is unresolved)');return None
        return max(xs)
    value = top('value_bound_all_geometries', allpos.get('value_bound'), t1['tail'])
    gated = top('value_bound_all_geometries_without_root', allpos.get('value_bound_without_root'), t1['tail'])
    root_all = top('root_logit_all_geometries', c_bound, root['tail'])
    return dict(label='PROPOSAL (informational, not a verdict): three geometries (the pricers, the gate oracle, D3\'s FramePricer); the declaration '
        'adds the margin', kind='all_positions_three_geometries', rule=FOLD_RULE, sites={'b.t1_value': t1, 'b.root_peer_logits': root},
        root_c=dict(tail=c_bound, flags=c_flags, inputs=c_inputs), value_bound_all_geometries=value, value_bound_all_geometries_without_root=gated,
        near_tie_q_bound_all_geometries=value, root_logit_all_geometries=root_all, turn1_logit_bound=allpos.get('turn1_logit_bound'),
        deeper_logit_bound=allpos.get('deeper_logit_bound'), restated='turn1_logit_bound and deeper_logit_bound: the earlier all_positions proposal\'s, unchanged',
        earlier_all_positions={k: allpos.get(k) for k in ('value_bound', 'value_bound_level', 'value_bound_without_root', 'value_bound_without_root_level',
            'turn1_logit_bound', 'deeper_logit_bound', 'matches_gate_shapes')}, matches_earlier=bool((check or {}).get('matches')), margin=None, flags=flags)


# Driving one run ---------------------------------------------------------------------------------------------------------------------
def measure(requests, *, evaluators, frame_pricer, executor, materializer, recipe, observe, capacity=CAPACITY, sample=SAMPLE, seed=SEED, chunk=CHUNK,
        fixed_shapes=False, actor_capacity=None, progress=None):
    """requests {name: [[Request]]} (bind's paired groups); evaluators {'f32', 'bf16'}: one champion at two precisions; frame_pricer the bank
    provider's (the gate's base); observe the ladder's champion-layout leaf (champion_observer). Per call (groups split at `capacity`): D3 at
    both precisions, the comparison and the anchor walk. Returns dict(calls, d3_options, walk)."""
    import importlib
    if set(evaluators) != set(NAMES) or set(requests) != set(NAMES):raise ValueError(f'evaluators and requests named {NAMES} required')
    o2, batch = (importlib.import_module(f'gpu_engine.{m}') for m in ('depth2_oracle', 'device_public_teacher_batch'))
    pricers = build_d3(frame_pricer, evaluators);options = d3_options(pricers, actor_capacity=actor_capacity, split_capacity=capacity, fixed_shapes=fixed_shapes)
    split = {name: g2.split_groups(groups, capacity) for name, groups in requests.items()}
    if [len(g) for g in split[REFERENCE]] != [len(g) for g in split[TESTED]]:raise ValueError('the two precisions\' request groups must pair')
    walker = clv.Walker(executor, observe, chunk=chunk, view=ce.digest_view(evaluators[REFERENCE]));heads = clv.Heads(evaluators)
    stack_rows = o2.bucket(2 * capacity) if fixed_shapes else None;calls = []
    for call in range(len(split[REFERENCE])):
        reqs = {name: tuple(split[name][call]) for name in NAMES};context = dict(call=call, roots=len(reqs[REFERENCE]));results, costs = {}, {}
        for name in NAMES:
            with stage(progress, 'd3_frame_pricer', precision=name, **context) as info:
                clock = time.perf_counter();results[name] = pricers[name].price(reqs[name], materializer, recipe, (evaluators[name].params,))
                timing = getattr(pricers[name], 'last_timing', None) or {};want = bucket(pricers[name], len(reqs[name]))
                costs[name] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(timing), world_prepare_seconds=timing.get('world_prepare_seconds'),
                    root_capacity=timing.get('root_capacity'), expected_root_capacity=want);info.update(costs[name])
                if timing.get('root_capacity', want) != want:options['departures'].append(f'call {call} {name}: compiled at {timing["root_capacity"]} roots, not the gate\'s bucket {want}')
        with stage(progress, 'compare', **context) as info:
            comparison = compare_call(results, reqs, recipe=recipe, call=call)
            info.update(counts=comparison['counts'], structural={k: e['count'] for k, e in comparison['findings'].items()},
                max={k: float(np.abs(comparison[k]).max()) if comparison[k].size else None for k in ('values', 'logits')})
        with stage(progress, 'anchor_walk', **context) as info:
            stack = o2.stack(reqs[REFERENCE], batch._numpy_worlds(reqs[REFERENCE], materializer, recipe), size=stack_rows)
            anchor = anchor_call(call, comparison.pop('arrays'), stack, walker, heads, sample=sample, rng=np.random.default_rng([seed, call]))
            info.update(anchor_cells=len(anchor['cells']['spec']), anchor_worlds=len(anchor['roots']['spec']), seconds_walking=walker.seconds)
        calls.append(dict(call=call, roots=context['roots'], costs=costs, comparison=comparison, anchor=anchor))
    options['matches_gate'] = not options['departures']
    return dict(calls=calls, d3_options=options, walk=dict(rows=walker.rows, seconds=walker.seconds, chunk=chunk, stack_rows=stack_rows, sample=sample, seed=seed))


def assemble(measured, *, earlier=None, check=None):
    """The report body from measure's output: comparison, D_all, the anchor, the proposal, the calls, costs and the exit code."""
    calls = measured['calls'];comparison = merge_comparison(calls);d = d_all(comparison['quantities'])
    anchor = anchor_section([c['anchor'] for c in calls], d);proposal = fold(earlier, d, check)
    compared = sum(comparison['quantities'][k].get('count', 0) for k in QUANTITIES);status = 'MEASURED' if compared else 'NOTHING_COMPARED'
    reasons = [] if compared else ['no D3 value or reply logit was compared (no root, or no world priced by both runs)']
    total = lambda name, key: sum(c['costs'][name].get(key) or 0. for c in calls)
    costs = {name: dict(wall_seconds=total(name, 'wall_seconds'), lower_and_compile_seconds=total(name, 'lower_and_compile_seconds'),
        device_seconds=total(name, 'device_seconds'), buckets=sorted({c['costs'][name].get('root_capacity') for c in calls} - {None})) for name in NAMES}
    summary = [dict(call=c['call'], roots=c['roots'], counts=c['comparison']['counts'], structural={k: e['count'] for k, e in c['comparison']['findings'].items()},
        pricing_order_differs=len(c['comparison']['pricing_order_differs']), anchor=dict(cells=len(c['anchor']['cells']['spec']), roots=len(c['anchor']['roots']['spec'])),
        costs=c['costs']) for c in calls]
    return dict(status=status, exit=EXIT[status], status_reasons=reasons, comparison=comparison, d_all=dict(d, rule=D_ALL_RULE), anchor=anchor,
        source_check=check, proposal=proposal, calls=summary, costs=dict(costs, walk=measured.get('walk')), d3_options=measured['d3_options'])


def tree_commit(root=None):
    """The tree's commit: TREE_COMMIT at the tree root (the pod's convention), else `git rev-parse HEAD`; None when neither is available."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent;marker = root / 'TREE_COMMIT'
    if marker.is_file():return dict(commit=marker.read_text().strip() or None, source=str(marker))
    try:out = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():return dict(commit=out.stdout.strip(), source='git rev-parse HEAD')
    return dict(commit=None, source='unavailable (no TREE_COMMIT, no git)')


# Command line ---------------------------------------------------------------------------------------------------------------------
def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--d3-root', required=True, help='the D3 bank run root (prespec.json, runtime.json, run/), as champion_leaf_validation')
    parser.add_argument('--chunk', type=int, required=True, help='bank chunk index')
    parser.add_argument('--out', required=True, help='new output directory (d3_geometry.json, d3_geometry.md)')
    parser.add_argument('--validation-json', required=True, help='the earlier champion_leaf_validation report (validation.json) whose geometry '
        'section (P_all, O_sample, the all_positions proposal) the proposal folds D_all into')
    parser.add_argument('--sample', type=int, default=SAMPLE, help='anchor rows per call and kind (turn-1 cells; root worlds)')
    parser.add_argument('--fixed-shapes', action='store_true', help='as the gate\'s (b) runs: D3 is unchanged by it (per-call buckets); fixes the anchor walk\'s stack')
    ce.add_arguments(parser);parser.set_defaults(memory_policy=False)   # g2.bank_chunk_requests reads it; never applied (as the validation run)
    args = parser.parse_args(argv)
    if args.chunk < 0:parser.error('--chunk N >= 0 required')
    if not all(getattr(args, name) for name in ce.FLAGS):parser.error('--champion-archive, --champion-archive-sha256 and --champion-checkpoint-sha256 are required')
    for name in ce.FLAGS[1:]:
        if not ce._hex64(getattr(args, name)):parser.error(f'--{name.replace("_", "-")} must be a 64-hex sha256')
    if args.champion_precision is not None:parser.error('--champion-precision does not apply: this tool prices with both precisions (bf16 and f32)')
    if args.sample < 1:parser.error('positive --sample required')
    return args


def _num(x, fmt='.4g'):
    return clv._num(x, fmt)


def markdown(report):
    """A short summary: the comparison table with D_all, structural findings, the anchor and the proposal (d3_geometry.json holds everything)."""
    h, c = report.get('header') or {}, report.get('comparison') or {};get = lambda d, k: (d or {}).get(k) or {}
    champ, tree, opts = get(h, 'champion'), get(h, 'tree'), get(h, 'd3_options');d = report.get('d_all') or {}
    lines = [f'# Champion 9375: D3 FramePricer bf16 vs f32 at the gate\'s (b) positions ({report.get("status")}, exit {report.get("exit")}; informational)', '',
        f'- Source: `{json.dumps(h.get("source"), default=str)}`; roots {h.get("roots")} in {len(report.get("calls") or [])} call(s); dropped at binding '
        f'{ {k: len(v) for k, v in (h.get("dropped") or {}).items()} }', f'- Champion archive `{champ.get("archive_sha256")}` (checkpoint `{champ.get("checkpoint_sha256")}`); '
        f'tree `{tree.get("commit")}` ({tree.get("source")})']
    for name, p in (opts.get('pricers') or {}).items():
        lines.append(f'- D3 {name}: capacity {p.get("capacity")}, block_rows {p.get("block_rows")}, actor_capacity {p.get("actor_capacity")}, vectorized {p.get("vectorized")}, '
            f'host_prep {p.get("host_prep")}, evaluator {p.get("evaluator", {}).get("identity")}; buckets {get(get(report, "costs"), name).get("buckets")}')
    lines += [f'- D3 options match the gate\'s: {opts.get("matches_gate")}' + (f' (FLAGGED: {"; ".join(opts.get("departures") or [])})' if opts.get('departures') else ''), '']
    lines += ['| Quantity | n | non-finite | max | p99 | p99.9 | p99.99 | D_all |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    rows = [(k, v, _num(get(d, k).get('max'))) for k, v in (c.get('quantities') or {}).items()] + [(f'{k} (informational)', v, '-') for k, v in (c.get('informational') or {}).items()]
    lines += [f'| {k} | {v.get("count")} | {v.get("nonfinite")} | {_num(v.get("max"))} | {_num(v.get("p99"))} | {_num(v.get("p99.9"))} | {_num(v.get("p99.99"))} | {x} |' for k, v, x in rows]
    s = c.get('structural') or {}
    lines += ['', '- Structural differences: ' + ('; '.join(f'{k} {e.get("count")}' for k, e in (s.get('kinds') or {}).items()) or 'none'),
        f'- Pricing order differs (informational) on {get(c, "pricing_order").get("roots", 0)} root(s)']
    a = report.get('anchor') or {}
    lines += ['', '## f32 anchor (sample)', '', a.get('reference', ''), '', '| Quantity | rows | compared | D3 f32 max | bound | within | D3 bf16 max (sample) |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for k, q in (a.get('quantities') or {}).items():
        lines.append(f'| {k} | {q.get("rows")} | {q.get("compared")} | {_num(q.get("max"), ".3g")} | {_num(q.get("bound"), "g")} | {q.get("within")} | {_num(get(q, "bf16").get("max"))} |')
    lines += ['', f'- Anchor flags: {"; ".join(a.get("flags") or []) or "none"}']
    p, sc = report.get('proposal') or {}, report.get('source_check') or {}
    lines += ['', '## Three-geometry proposal (PROPOSAL, not a verdict; the declaration adds the margin)', '',
        f'- Earlier report `{sc.get("path")}`: matches this run {sc.get("matches")}' + (f' (FLAGGED: {"; ".join(sc.get("flags") or [])})' if sc.get('flags') else ''), '',
        '| Bound | value | from | margin |', '| --- | --- | --- | --- |']
    for name, x in (p.get('sites') or {}).items():
        lines.append(f'| {name} | {_num(x.get("tail"))} | P_all {_num(get(x, "p_all").get("max"))} + D_all {_num(get(x, "d_all").get("max"))} | - |')
    e = get(p, 'earlier_all_positions')
    lines += [f'| root (c) tail | {_num(get(p, "root_c").get("tail"))} | max(P_all + O_sample, 2 x P_all) at root opp_logits | - |',
        f'| value_bound_all_geometries | {_num(p.get("value_bound_all_geometries"))} | max(earlier {_num(e.get("value_bound"))}, b.t1_value) | (declaration) |',
        f'| value_bound_all_geometries_without_root | {_num(p.get("value_bound_all_geometries_without_root"))} | max(earlier {_num(e.get("value_bound_without_root"))}, b.t1_value) | (declaration) |',
        f'| near_tie_q_bound_all_geometries | {_num(p.get("near_tie_q_bound_all_geometries"))} | = value_bound_all_geometries | (declaration) |',
        f'| root_logit_all_geometries | {_num(p.get("root_logit_all_geometries"))} | max(root (c) tail, b.root_peer_logits) | (declaration) |',
        f'| turn1_logit_bound | {_num(p.get("turn1_logit_bound"))} | earlier, unchanged | (declaration) |',
        f'| deeper_logit_bound | {_num(p.get("deeper_logit_bound"))} | earlier, unchanged | (declaration) |', '',
        f'- Proposal flags: {"; ".join(p.get("flags") or []) or "none"}', '', 'Rules: `d3_geometry.json` header.rules (declared in gpu_engine/champion_d3_geometry.py).', '']
    return '\n'.join(lines)


def write(out, report):
    out = Path(out);body = clv.plain(report)
    (out / 'd3_geometry.json').write_text(json.dumps(body, indent=1, allow_nan=False) + '\n');(out / 'd3_geometry.md').write_text(markdown(body))


def main(argv=None):
    args = parse(argv);started = time.perf_counter();source_path = Path(args.validation_json)
    earlier = json.loads(source_path.read_text())   # read (and fail) before any loading
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    log('D3_GEOMETRY_RULES', rules=RULES, gate_d3=GATE_D3, split_capacity=CAPACITY, validation_json=str(source_path), sample=args.sample, fixed_shapes=args.fixed_shapes)
    with stage(log, 'load_chunk', d3_root=args.d3_root, chunk=args.chunk):loaded = g2.bank_chunk_requests(args)
    args.champion_precision = 'bf16';bf16 = g2.load_champion(args, loaded);evaluators = {REFERENCE: replace(bf16, precision='float32'), TESTED: bf16}
    clv.check_pair(evaluators);log('EVALUATORS', **{name: ev.header()['identity'] for name, ev in evaluators.items()})
    with stage(log, 'bind') as info:
        requests, dropped, root_values = clv.bind(loaded.groups, evaluators);roots = sum(map(len, requests[REFERENCE]))
        info.update(roots=roots, dropped={k: len(v) for k, v in dropped.items()})
    window = getattr(loaded.plan, 'historical_window', None)
    measured = measure(requests, evaluators=evaluators, frame_pricer=loaded.frame_pricer, executor=loaded.provider.executor, materializer=loaded.provider.materializer,
        recipe=loaded.provider.recipe, observe=clv.champion_observer(loaded), capacity=CAPACITY, sample=args.sample, chunk=CHUNK, fixed_shapes=args.fixed_shapes,
        actor_capacity=None if window is None else window + 1, progress=log)
    check = source_check(earlier, source=loaded.source, evaluators=evaluators, roots=roots, fixed_shapes=args.fixed_shapes, path=str(source_path))
    body = assemble(measured, earlier=earlier, check=check)
    import jax
    report = dict(schema=SCHEMA, status=body.pop('status'), exit=body.pop('exit'), status_reasons=body.pop('status_reasons'), informational=True,
        header=dict(rules=RULES, source=loaded.source, roots=roots, dropped=dropped, root_abs_value={name: g2.percentiles(v) for name, v in root_values.items()},
            champion=dict(archive=args.champion_archive, archive_sha256=args.champion_archive_sha256, checkpoint_sha256=args.champion_checkpoint_sha256),
            evaluators={name: dict(ev.header(), archive=args.champion_archive) for name, ev in evaluators.items()}, tree=tree_commit(), d3_options=body.pop('d3_options'),
            options=dict(split_capacity=CAPACITY, fixed_shapes=args.fixed_shapes, sample=args.sample, seed=SEED, walk_chunk=CHUNK), validation_json=str(source_path),
            devices=[str(x) for x in jax.devices()], seconds=time.perf_counter() - started), **body)
    write(out, report);c, a, p = report['comparison'], report['anchor'], report['proposal']
    log('CHAMPION_D3_GEOMETRY_COMPLETE', status=report['status'], exit=report['exit'], out=str(out / 'd3_geometry.json'),
        compared={k: c['quantities'][k].get('count') for k in QUANTITIES}, max={k: c['quantities'][k].get('max') for k in QUANTITIES},
        d_all={k: report['d_all'][k].get('max') for k in QUANTITIES}, structural=c['structural']['count'], anchor=a['observed'], anchor_flags=len(a['flags']),
        d3_departures=report['header']['d3_options']['departures'], source_flags=len(check['flags']),
        proposal={k: p.get(k) for k in ('value_bound_all_geometries', 'root_logit_all_geometries', 'turn1_logit_bound', 'deeper_logit_bound')})
    return report['exit']


if __name__ == '__main__':raise SystemExit(main())
