"""Gap 2, the third bf16 geometry: D3's FramePricer (the reference of the depth-N gate's check (b)) under champion 9375, bf16 against f32,
at every position (b) compares, the gate's own depth-2 pass at the same positions (all nine root-logit lanes included), and a
three-geometry bounds proposal (informational, never a verdict). Numpy-only import; JAX loads after the bank (as the gates).

champion_leaf_validation measured the bf16 noise of the pricers (P_all, every position) and of the gate oracle (O_sample); (b) compares
the pricer's turn 1 against a third bf16 program, D3's FramePricer, whose batch geometry (2 x bucket root-world rows for the opponent's
prior, 81 x that for the joint leaves) is neither; and g2.check_d reads root_peer_logits on ALL nine lanes of every world D3 priced,
requested or not, where P_all covered legal lanes of requested worlds only. This tool measures both.

Setup: --d3-root DIR --chunk N through d3_depth2_gate.bank_chunk_requests (the bank's provider and its FramePricer), the champion through
g2.load_champion (archive and checkpoint sha256, the real 9375, the bank catalog), one archive at two precisions (bf16 under test, f32 the
reference; champion_leaf_validation.check_pair). The roots are bound ONCE, as the gate binds them (d3_depthn_gate.load_roots:
g2.bind_roots under the bf16 champion: its root prior, actors 0/0), and BOTH precisions price these identical requests; calls split at
512. No f32-bound variant is kept: the anchor walks the materialized worlds, which do not depend on the root prior. Built EXACTLY as
d3_depthn_gate builds them (options, each call's compile bucket and any departure recorded): D3 (D3_RULE: g2.with_forward(the bank
provider's FramePricer, champion)) and the full-width depth-2 pass (PASS_RULE: d3_depthn_gate.depth2_pass_pricer), one each per precision.

1. comparison (POSITIONS_RULE), |bf16 - f32| per position (n, non-finite, max, p99, p99.9, p99.99):
   D3: d3.t1_value (every (q, world, a, r) with a joint value in both runs), d3.root_peer_logits (lanes legal in both, worlds requested in
   both), d3.root_peer_logits.all_lanes (all nine lanes of every world both runs priced: (b)'s root positions).
   The depth-2 pass, at (b)'s positions (the bf16 D3, the gate's): pricer.t1_value (cells D3 priced and the frames enumerate),
   pricer.root_peer_logits.all_lanes (all nine lanes of every world D3 priced) and, as a cross-check of the earlier report's P_all(root
   opp), pricer.root_peer_logits (champion_leaf_validation.precision_call's positions: pre 0 and requested in both frames, legal lanes).
   Every structural difference between the two runs of either program is listed by kind (count, first cells), never dropped.
2. D_all and this run's P (D_ALL_RULE): all-positions max + the declared f32 adapter bound (1e-5 values, 1e-4 logits), through
   champion_leaf_validation.pricer_all, P_all's construction.
3. f32 anchor (ANCHOR_RULE): D3's f32 against the f32 raw forward champion_leaf_validation uses (champion.forward, HIGHEST, on the ladder's
   champion layout of an independent walk: its Walker and Heads), on a seeded sample per call (--sample turn-1 cells and root worlds);
   values, legal reply-logit lanes and illegal lanes reported separately; any lane over its bound flagged.
4. proposal (FOLD_RULE): the (b) site tails from the earlier report (--validation-json), this run's P and D_all, folded into the
   all-positions value and root-logit bounds, each labelled with the term that sets it; the earlier report's source, evaluators, roots
   and shapes checked against this run, any mismatch flagged. `margin` is left empty: the declaration adds it.
d3_geometry.json and d3_geometry.md in --out. Exit 0 when anything was compared, 2 when nothing was (no verdict either way).
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from . import champion_evaluator as ce, champion_leaf_validation as clv, d3_depth2_gate as g2, d3_depthn_gate as gate

SCHEMA = 'pokezero.champion-d3-geometry.v2'
REFERENCE, TESTED, NAMES = clv.REFERENCE, clv.TESTED, clv.NAMES
VALUE, LOGITS, ALL_LANES = 'd3.t1_value', 'd3.root_peer_logits', 'd3.root_peer_logits.all_lanes'
ILLEGAL = 'd3.root_peer_logits.illegal_lanes'   # the anchor's illegal lanes of requested worlds
P_VALUE, P_LOGITS, P_ALL_LANES = 'pricer.t1_value', 'pricer.root_peer_logits', 'pricer.root_peer_logits.all_lanes'
QUANTITIES = (VALUE, LOGITS, ALL_LANES)
PRICER_QUANTITIES = (P_VALUE, P_LOGITS, P_ALL_LANES)
KIND = {VALUE: 'value', P_VALUE: 'value', LOGITS: 'logits', ALL_LANES: 'logits', P_LOGITS: 'logits', P_ALL_LANES: 'logits'}
SAMPLE = 512           # anchor rows per call and kind (turn-1 cells; root worlds): one walk chunk each
SEED, CHUNK, FIRST = clv.SEED, clv.CHUNK, g2.FIRST
GATE_CAPACITY = 512    # d3_depthn_gate's default --capacity: the depth-2 pass's capacity and the call split (g2.call_capacity(FramePricer 4096, 512))
CAPACITY = GATE_CAPACITY   # this run's
CROSS_CHECK = 1e-6     # this run's legal-lane pricer max against the earlier report's P_all(root opp)
GATE_D3 = dict(capacity=4096, block_rows=32, vectorized=True, host_prep=True)   # the bank provider's FramePricer, kept by g2.with_forward
GATE_PASS = dict(depth=2, k_own=[9], k_opp=[9], k1_own=9, k1_opp=9, capacity=GATE_CAPACITY, block_lanes=2187, blocks_per_call=16, selection='prior', digest=True,
    actor_capacity=1, fixed_capacity=True)   # d3_depthn_gate.run_gate's depth2_pass_pricer under --fixed-shapes and a champion (actors: len(params) = 1)
EXIT = dict(MEASURED=0, NOTHING_COMPARED=2)
log, stage = g2.log, g2.stage

D3_RULE = ('D3\'s FramePricer as d3_depthn_gate builds it for (b): g2.with_forward(loaded.frame_pricer, champion), loaded.frame_pricer '
    'being the bank provider\'s (v5_device_teacher.Provider(vectorized=True, batched_prep=True): capacity 4096, block_rows 32, actor_capacity '
    'historical_window + 1, vectorized, host_prep) and with_forward keeping every option and adding forward=. Calls split at 512 '
    '(g2.call_capacity(FramePricer 4096, the depth-2 pass at --capacity 512); d3_depthn_gate.run_gate); each call compiles at its own '
    'bucket min(4096, the next power of two of its roots) (vectorized FramePricer.price) with 2 x bucket opponent-prior rows and 162 x bucket '
    'joint-leaf rows; --fixed-shapes leaves D3 unchanged (d3_depthn_gate.FIXED_SHAPES: D3 keeps its per-call buckets). One pricer per '
    'precision: bf16 (the gate\'s) and f32 (the reference, the same options).')
PASS_RULE = ('The full-width depth-2 pass as d3_depthn_gate.run_gate builds it: depth2_pass_pricer(pricer_module(), executor, model, '
    'capacity=--capacity 512, actor_capacity=len(params) = 1 under the champion, k1_own=k1_opp=9, block_lanes=2187, digest=True, '
    'fixed_capacity=--fixed-shapes, forward=champion): DepthNPricer(depth=2, k_own=k_opp=9, selection prior, blocks_per_call 16). One per '
    'precision. Its turn 1 (root_peer_logits, t1_value) is what the depth-N gate\'s (b) compares with D3; the gate\'s (b) reads the depth-N '
    'frame\'s turn 1, which the validation tool measured equal to depth 2\'s (depth3.turn1 = depth2.turn1 in validation.json).')
POSITIONS_RULE = ('D3: d3.t1_value at every (q, world, a, r) whose joint value is non-None in BOTH runs (g2.check_d reads each such cell); '
    'd3.root_peer_logits on the lanes legal in both runs of worlds both priced with the opponent requested in both; d3.root_peer_logits.all_lanes '
    'on all nine lanes of every world both runs priced (g2.check_d reads all nine lanes of every world D priced, requested or not). The '
    'depth-2 pass at (b)\'s positions as the gate defines them (the bf16 D3): pricer.t1_value at cells D3 priced that both frames enumerate '
    '(g2.check_d skips a cell the pricer did not enumerate; D3\'s error cells hold no value); pricer.root_peer_logits.all_lanes on all nine '
    'lanes of every world D3 priced; pricer.root_peer_logits (cross-check) at champion_leaf_validation.precision_call\'s positions (pre 0 and '
    'the opponent requested in both frames, the f32 frame\'s legal lanes). |bf16 - f32| per position: n, non-finite count, max, p99, p99.9, '
    'p99.99. Structural differences of either program are listed by kind, never dropped; unrequested worlds must carry zero reply logits.')
D_ALL_RULE = ('D_all (and this run\'s P) = the all-positions max |bf16 - f32| + the declared f32 adapter bound (1e-5 values, 1e-4 logits), '
    'built by champion_leaf_validation.pricer_all exactly as P_all: |bf16 - f32 raw| <= |bf16 - f32| + |f32 - f32 raw|.')
ANCHOR_RULE = ('The adapter bound\'s premise for D3 (|D3 f32 - f32 raw| <= 1e-5 values, 1e-4 logits on EVERY lane) checked against the f32 raw '
    'forward champion_leaf_validation uses: champion.forward (f32, HIGHEST, its own jit) on the ladder\'s champion layout of an independent walk '
    '(champion_leaf_validation.Walker: the depth-N oracle\'s step, D\'s turn-1 joint rule, the node\'s own chance; Heads). A seeded sample per '
    'call: --sample turn-1 cells both runs value and --sample root worlds both runs priced with the opponent requested. Values compare where '
    'the walk is a head-valued child (not terminal, invalid or in error on either layout); reply logits on all nine lanes where both of the '
    'walk\'s root views are error-free, legal and illegal lanes reported separately (an unrequested world\'s reply logits are D3\'s zeros, '
    'checked exactly in the comparison, not against the forward). Terminal cells compare D3\'s value with the walk\'s terminal (+1 / -1 / 0 '
    'from the owner) exactly. Flagged: any value or lane over its bound, non-finite differences, no compared row, a valued cell the walk '
    'disagrees on, a terminal mismatch, a champion layout unequal to an evaluator\'s view, and a bf16 sample max above its D_all. Cheapest '
    'faithful anchor: the walk re-steps only sampled cells; the depth-2 pass priced here is another bf16/f32 program, not an independent one.')
FOLD_RULE = ('PROPOSAL, not a verdict; `margin` empty (the declaration adds it). (b) site tails, all positions, by the triangle inequality '
    'through the f32 raw forward: b.t1_value = P(t1 value) + D_all(d3.t1_value), P(t1 value) the larger of the earlier P_all(depth2.turn1 '
    'value) and this run\'s P(pricer.t1_value) at (b)\'s cells (flagged when this run\'s is larger); b.root_peer_logits = P_all(root opp_logits) '
    '+ D_all(d3.root_peer_logits) (legal lanes; this run\'s legal-lane pricer max is a cross-check of P_all(root opp), flagged beyond 1e-6); '
    'b.root_peer_logits_all_lanes = P(pricer.root_peer_logits.all_lanes) + D_all(d3.root_peer_logits.all_lanes) (what (b) reads). '
    'value_bound_all_geometries = max(the earlier all_positions value_bound, b.t1_value) (and without the root level: '
    'max(value_bound_without_root, b.t1_value)); near_tie_q_bound_all_geometries = value_bound_all_geometries (the earlier rule); '
    'root_logit_all_geometries = max(the root (c) tail max(P_all(root opp) + O_sample(root opp), 2 x P_all(root opp)), b.root_peer_logits, '
    'b.root_peer_logits_all_lanes). Each bound names the term that sets it. turn1_logit_bound and deeper_logit_bound are the earlier '
    'report\'s, unchanged. A tail with an unmeasured or non-finite input has no bound (None, flagged), and neither has a bound it enters. '
    'The earlier report\'s source (d3_root, chunk, declaration), evaluators (archive and checkpoint sha256, precision), roots and shapes are '
    'checked against this run; every mismatch is flagged.')
RULES = dict(d3=D3_RULE, depth2_pass=PASS_RULE, positions=POSITIONS_RULE, d_all=D_ALL_RULE, anchor=ANCHOR_RULE, fold=FOLD_RULE,
    bounds=dict(f32_adapter_value=clv.ADAPTER_VALUE, f32_adapter_logit=clv.ADAPTER_LOGIT, cross_check=CROSS_CHECK), exit=EXIT)


# The gate's two programs ---------------------------------------------------------------------------------------------------------------
def build_d3(frame_pricer, evaluators):
    """{name: FramePricer}: the gate's construction for (b), g2.with_forward(the bank provider's FramePricer, evaluator), per precision."""
    return {name: g2.with_forward(frame_pricer, evaluators[name]) for name in NAMES}


def build_passes(module, executor, model, evaluators, *, capacity=GATE_CAPACITY, fixed_shapes=True):
    """{name: DepthNPricer}: d3_depthn_gate.run_gate's full-width depth-2 pass (depth2_pass_pricer with the gate's defaults and digest=True)
    under the champion (actor_capacity = len(params) = 1), per precision."""
    return {name: gate.depth2_pass_pricer(module, executor, model, capacity=capacity, actor_capacity=1, k1_own=9, k1_opp=9, block_lanes=2187, digest=True,
        fixed_capacity=fixed_shapes, forward=evaluators[name]) for name in NAMES}


def bucket(pricer, roots):
    """The root capacity a vectorized FramePricer.price compiles one call of `roots` requests at (its own rule)."""
    return min(pricer.capacity, 1 << (roots - 1).bit_length()) if pricer.vectorized else pricer.capacity


def _evaluator_departures(name, evaluator):
    identity = evaluator['identity']
    return [] if evaluator['kind'] == 'champion' and identity[-1] == clv.PRECISION_OF.get(name) else [f'{name}: evaluator {identity} is not the champion at {clv.PRECISION_OF.get(name)}']


def d3_options(pricers, *, actor_capacity=None, split_capacity=GATE_CAPACITY, fixed_shapes=True):
    """Each FramePricer's options against the gate's (GATE_D3, actor_capacity = the plan's historical_window + 1 when known, the champion at
    the pricer's precision, the 512 split, --fixed-shapes); `departures` names every difference, `notes` what could not be checked."""
    record = {name: dict(capacity=p.capacity, block_rows=p.block_rows, actor_capacity=p.actor_capacity, vectorized=p.vectorized, host_prep=p.host_prep,
        evaluator=ce.describe(getattr(p, 'forward', None))) for name, p in pricers.items()}
    departures, notes = [], []
    for name, r in record.items():
        departures += [f'{name}: {k} {r[k]!r} != the gate\'s {v!r}' for k, v in GATE_D3.items() if r[k] != v]
        if actor_capacity is None:notes.append(f'{name}: actor_capacity {r["actor_capacity"]} not checked (the plan\'s historical_window is unknown)')
        elif r['actor_capacity'] != actor_capacity:departures.append(f'{name}: actor_capacity {r["actor_capacity"]!r} != the gate\'s {actor_capacity!r}')
        departures += _evaluator_departures(name, r['evaluator'])
    if split_capacity != GATE_CAPACITY:departures.append(f'call split {split_capacity} != the gate\'s {GATE_CAPACITY}')
    if not fixed_shapes:departures.append('--fixed-shapes not given: the gate\'s (b) runs under it (D3 itself is unchanged by it)')
    return dict(pricers=record, gate=dict(GATE_D3, actor_capacity=actor_capacity, split_capacity=GATE_CAPACITY, fixed_shapes=True), split_capacity=split_capacity,
        fixed_shapes=fixed_shapes, departures=departures, matches_gate=not departures, notes=notes, rule=D3_RULE)


def pass_options(passes):
    """Each depth-2 pass's options against the gate's (GATE_PASS, the champion at the pricer's precision); `departures` names every difference."""
    record = {name: dict(depth=p.depth, k_own=list(p.k_own), k_opp=list(p.k_opp), k1_own=p.k1_own, k1_opp=p.k1_opp, capacity=p.capacity, block_lanes=p.block_lanes,
        blocks_per_call=p.blocks_per_call, selection=p.selection, digest=p.digest, actor_capacity=p.actor_capacity, fixed_capacity=p.fixed_capacity,
        evaluator=ce.describe(getattr(p, 'forward', None))) for name, p in passes.items()}
    departures = []
    for name, r in record.items():
        departures += [f'{name}: {k} {r[k]!r} != the gate\'s {v!r}' for k, v in GATE_PASS.items() if r[k] != v] + _evaluator_departures(name, r['evaluator'])
    return dict(pricers=record, gate=GATE_PASS, departures=departures, matches_gate=not departures, rule=PASS_RULE)


# 1. The comparisons -----------------------------------------------------------------------------------------------------------------
def _note(findings, name, cells, f32, bf16, *, call, addresses):
    """One structural difference kind: its count and the first FIRST cells (call, cell, root address, each run's value)."""
    cells = np.asarray(cells, np.int64).reshape(len(cells), -1) if len(cells) else np.zeros((0, 1), np.int64)
    if not len(cells):return
    e = findings.setdefault(name, dict(count=0, first=[]));e['count'] += len(cells)
    for i in range(min(len(cells), FIRST - len(e['first']))):
        cell = cells[i].tolist();e['first'].append(dict(call=call, cell=cell, address=list(addresses[cell[0]]), f32=f32[i], bf16=bf16[i]))


def compare_call(results, requests, *, call=0):
    """D3 bf16 against D3 f32 on one call: results {name: D3 Results} on the same requests. Returns the signed differences (values, logits,
    all_lanes), the structural findings {kind: dict(count, first)}, counts and the Results as g2._d_arrays (check (b)'s array form)."""
    f, b = REFERENCE, TESTED;requests = tuple(requests)
    if not len(requests) == len(results[f]) == len(results[b]):raise ValueError('one D3 Result per request in each run')
    A, B = g2._d_arrays(results[f]), g2._d_arrays(results[b]);n = len(requests);findings = {};addresses = [clv.address(r) for r in requests]
    note = lambda name, cells, x, y: _note(findings, name, cells, x, y, call=call, addresses=addresses)
    bad = np.flatnonzero(A['d_status'] != B['d_status']);note('root_status', bad[:, None], A['d_status'][bad], B['d_status'][bad])
    at = np.argwhere(A['d_priced'] != B['d_priced']);x, y = (tuple(D['d_reason'][q, w] if not D['d_priced'][q, w] else 'priced' for q, w in at) for D in (A, B))
    note('world_refused', at, x, y)
    at = np.argwhere(~A['d_priced'] & ~B['d_priced'] & (A['d_reason'] != B['d_reason']));note('refusal_reason', at, A['d_reason'][tuple(at.T)], B['d_reason'][tuple(at.T)])
    both = A['d_priced'] & B['d_priced']
    at = np.argwhere(both & (A['d_opponent_requested'] != B['d_opponent_requested']))
    note('opponent_requested', at, A['d_opponent_requested'][tuple(at.T)], B['d_opponent_requested'][tuple(at.T)])
    at = np.argwhere(both & np.any(A['d_reply_legal'] != B['d_reply_legal'], axis=-1));note('reply_legal', at, A['d_reply_legal'][tuple(at.T)], B['d_reply_legal'][tuple(at.T)])
    for name, D in ((f, A), (b, B)):   # D3 zeroes an unrequested opponent's prior; (b) reads those lanes too
        at = np.argwhere(D['d_priced'] & ~D['d_opponent_requested'] & np.any(D['d_reply_logits'] != 0, axis=-1))
        note(f'unrequested_reply_logits_nonzero_{name}', at, A['d_reply_logits'][tuple(at.T)], B['d_reply_logits'][tuple(at.T)])
    grid = both[..., None, None] & np.ones((1, 1, 9, 9), bool)
    at = np.argwhere(grid & (A['d_known'] != B['d_known']));note('joint_value_none_pattern', at, A['d_known'][tuple(at.T)], B['d_known'][tuple(at.T)])
    at = np.argwhere(grid & (A['d_error'] != B['d_error']));note('joint_error', at, A['d_error'][tuple(at.T)], B['d_error'][tuple(at.T)])
    cells = grid & A['d_known'] & B['d_known'];values = B['d_value'][cells].astype(np.float64) - A['d_value'][cells]
    requested = both & A['d_opponent_requested'] & B['d_opponent_requested'];lanes = requested[..., None] & A['d_reply_legal'] & B['d_reply_legal']
    logits = B['d_reply_logits'][lanes].astype(np.float64) - A['d_reply_logits'][lanes]
    all_lanes = (B['d_reply_logits'][both].astype(np.float64) - A['d_reply_logits'][both]).reshape(-1)
    counts = dict(roots=n, worlds=2 * n, priced_f32=int(A['d_priced'].sum()), priced_bf16=int(B['d_priced'].sum()), priced_both=int(both.sum()),
        requested_both=int(requested.sum()), cells_f32=int(A['d_known'].sum()), cells_bf16=int(B['d_known'].sum()), cells_compared=int(cells.sum()),
        lanes_compared=int(lanes.sum()), all_lanes_compared=int(all_lanes.size))
    return dict(values=values, logits=logits, all_lanes=all_lanes, findings=findings, counts=counts, arrays={f: A, b: B})


FRAME_FIELDS = ('pre', 'root_peer_requested', 'root_peer_legal', 'root_peer_logits', 't1_status', 't1_value')


def pricer_call(frames, gate_d3, requests, *, call=0):
    """The depth-2 pass bf16 against f32 on one call at (b)'s positions: frames {name: frame}; gate_d3 the bf16 D3's g2._d_arrays (the gate's
    (b) reference). Returns the signed differences (values, logits (precision_call's legal-lane positions), all_lanes), findings and counts."""
    requests = tuple(requests);n = len(requests);addresses = [clv.address(r) for r in requests];findings = {}
    F = {name: {k: np.asarray(getattr(frames[name], k))[:n] for k in FRAME_FIELDS} for name in NAMES};f, b = F[REFERENCE], F[TESTED]
    note = lambda name, cells, x, y: _note(findings, name, cells, x, y, call=call, addresses=addresses)
    worlds = np.asarray(gate_d3['d_priced'], bool)[:n];known = np.asarray(gate_d3['d_known'], bool)[:n]
    for k in ('pre', 'root_peer_requested'):
        at = np.argwhere(worlds & (f[k] != b[k]));note(f'pricer_{k}', at, f[k][tuple(at.T)], b[k][tuple(at.T)])
    at = np.argwhere(worlds & np.any(f['root_peer_legal'] != b['root_peer_legal'], axis=-1))
    note('pricer_root_peer_legal', at, f['root_peer_legal'][tuple(at.T)], b['root_peer_legal'][tuple(at.T)])
    at = np.argwhere(known & ((f['t1_status'] != 0) != (b['t1_status'] != 0)));note('pricer_t1_enumeration', at, f['t1_status'][tuple(at.T)], b['t1_status'][tuple(at.T)])
    cells = known & (f['t1_status'] != 0) & (b['t1_status'] != 0);values = b['t1_value'][cells].astype(np.float64) - f['t1_value'][cells]
    all_lanes = (b['root_peer_logits'][worlds].astype(np.float64) - f['root_peer_logits'][worlds]).reshape(-1)
    req = (f['pre'] == 0) & (b['pre'] == 0) & f['root_peer_requested'].astype(bool) & b['root_peer_requested'].astype(bool)
    logits = (b['root_peer_logits'].astype(np.float64) - f['root_peer_logits'])[req][f['root_peer_legal'].astype(bool)[req]]
    counts = dict(worlds_d3_priced=int(worlds.sum()), cells_d3_priced=int(known.sum()), cells_compared=int(cells.sum()),
        cells_unenumerated_bf16=int(np.count_nonzero(known & (b['t1_status'] == 0))), all_lanes_compared=int(all_lanes.size), legal_worlds=int(req.sum()),
        legal_lanes_compared=int(logits.size))
    return dict(values=values, logits=logits, all_lanes=all_lanes, findings=findings, counts=counts)


def merge_findings(parts):
    out = {}
    for findings in parts:
        for name, e in findings.items():
            m = out.setdefault(name, dict(count=0, first=[]));m['count'] += e['count'];m['first'] = (m['first'] + e['first'])[:FIRST]
    return out


def _merged(parts, names):
    """{quantity: spread} over every call's signed differences (names: quantity -> the part's key), the findings and summed counts."""
    cat = lambda key: np.concatenate([p[key] for p in parts]) if parts else np.zeros(0)
    findings = merge_findings([p['findings'] for p in parts]);counts = {k: sum(p['counts'][k] for p in parts) for k in (parts[0]['counts'] if parts else ())}
    return dict(quantities={q: clv.spread(cat(key)) for q, key in names.items()}, structural=dict(count=sum(e['count'] for e in findings.values()),
        kinds=findings), counts=counts)


def merge_comparison(calls):
    """The run's comparison over every call: D3's three quantities and the depth-2 pass's (at (b)'s positions), each program's structural
    findings and counts."""
    d3 = _merged([c['comparison'] for c in calls], {VALUE: 'values', LOGITS: 'logits', ALL_LANES: 'all_lanes'})
    pricer = _merged([c['pricer'] for c in calls if c.get('pricer') is not None], {P_VALUE: 'values', P_LOGITS: 'logits', P_ALL_LANES: 'all_lanes'})
    d3['structural']['rule'] = pricer['structural']['rule'] = ('every structural difference between the bf16 and f32 runs, by kind (count and first '
        'cells: [q, world(, a, r)] within the call); none is dropped')
    return dict(d3, pricer=pricer, rule=POSITIONS_RULE)


# 2. D_all and this run's P ---------------------------------------------------------------------------------------------------------
def bounded(quantities):
    """{quantity: all-positions max + the declared f32 adapter bound}, through champion_leaf_validation.pricer_all (P_all's construction)."""
    out = {}
    for q, m in quantities.items():
        out[q] = clv.pricer_all(dict(values={'x': m or {}}), 'x', 'value') if KIND[q] == 'value' else clv.pricer_all(dict(logits={'x.opp': m or {}}), 'x', 'opp_logits')
    return out


def d_all(quantities):
    """D_all per D3 quantity (D_ALL_RULE)."""
    return bounded({q: (quantities or {}).get(q) for q in QUANTITIES})


def p_run(quantities):
    """This run's P per depth-2 pass quantity (D_ALL_RULE)."""
    return bounded({q: (quantities or {}).get(q) for q in PRICER_QUANTITIES})


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
    opponent requested (walked zero plies), each with both runs' D3 values / reply logits (all nine lanes) and the walk's raw f32 forward."""
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
    """ANCHOR_RULE over every call's anchor rows; d: D_all (each bf16 sample max is checked against it)."""
    cells, roots = _concat(records, 'cells'), _concat(records, 'roots');flags, findings = [], {};empty = lambda: clv.spread(np.zeros(0))
    quantity = {VALUE: dict(rows=0, compared=0, terminal=0, f32=empty(), bf16=empty()), LOGITS: dict(rows=0, compared=0, f32=empty(), bf16=empty()),
        ILLEGAL: dict(rows=0, compared=0, f32=empty(), bf16=empty())}
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
    if roots is not None:
        ok = ~roots['done'] & ~roots['invalid'] & (roots['own_error'] == 0) & (roots['champion_own_error'] == 0) & (roots['champion_peer_error'] == 0)
        findings['roots_walk_disagrees'] = int(np.count_nonzero(~ok))
        for name in NAMES:findings[f'roots_inputs_unequal_{name}'] = int(np.count_nonzero(ok & ~roots[f'inputs_equal_{name}']))
        for key, lanes in ((LOGITS, roots['legal'] & ok[:, None]), (ILLEGAL, ~roots['legal'] & ok[:, None])):
            quantity[key] = dict(rows=int(len(ok)), compared=int(ok.sum()), lanes=int(lanes.sum()), f32=clv.spread((roots['f32'].astype(np.float64) - roots['raw'])[lanes]),
                bf16=clv.spread((roots['bf16'].astype(np.float64) - roots['raw'])[lanes]))
    verified = {}
    for key, bound in ((VALUE, clv.ADAPTER_VALUE), (LOGITS, clv.ADAPTER_LOGIT), (ILLEGAL, clv.ADAPTER_LOGIT)):
        q = quantity[key];m = q['f32'];q.update(bound=bound, max=m.get('max'), within=None if not m.get('count') else (m.get('max', 0.) <= bound and not m.get('nonfinite')))
        if not m.get('count'):flags.append(f'{key}: not verified: no anchor row compared')
        if m.get('nonfinite'):flags.append(f'{key}: {m["nonfinite"]} non-finite D3 f32 vs raw differences')
        if m.get('count') and m.get('max', 0.) > bound:flags.append(f'{key}: EXCEEDED: D3 f32 vs the f32 raw forward max {m["max"]:.3g} > the declared bound {bound}')
        verified[key] = bool(q['within'])
    lanes_bf16 = max((quantity[k]['bf16'].get('max') for k in (LOGITS, ILLEGAL) if quantity[k]['bf16'].get('max') is not None), default=None)
    for key, sample_max in ((VALUE, quantity[VALUE]['bf16'].get('max')), (LOGITS, quantity[LOGITS]['bf16'].get('max')), (ALL_LANES, lanes_bf16)):
        top = (d or {}).get(key, {}).get('max')
        if top is not None and sample_max is not None and sample_max > top:flags.append(f'{key}: the sample max |D3 bf16 - f32 raw| {sample_max:.4g} exceeds D_all {top:.4g}')
    flags += [f'{k}: {v}' for k, v in findings.items() if v]
    return dict(reference='f32 raw forward: champion.forward (float32, HIGHEST, its own jit) on the ladder\'s champion layout of an independent walk '
        '(champion_leaf_validation.Walker / Heads)', sampled=True, rule=ANCHOR_RULE, quantities=quantity, findings=findings, verified=verified, flags=flags,
        observed=dict(f32_value_max=quantity[VALUE]['max'], f32_value_bound=clv.ADAPTER_VALUE, f32_legal_logit_max=quantity[LOGITS]['max'],
            f32_illegal_logit_max=quantity[ILLEGAL]['max'], f32_logit_bound=clv.ADAPTER_LOGIT))


# 4. The proposal -----------------------------------------------------------------------------------------------------------------------
def source_check(earlier, *, source, evaluators, roots, capacity=GATE_CAPACITY, fixed_shapes=True, path=None):
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
    return dict(source='validation.json', level=label, column=column, quantity=quantity, max=m.get('max') if m.get('count') else None, count=m.get('count', 0),
        nonfinite=m.get('nonfinite', 0), precision_max=m.get('precision_max'))


def _run_entry(p, quantity):
    m = (p or {}).get(quantity) or {}
    return dict(source='this run', quantity=quantity, max=m.get('max') if m.get('count') else None, count=m.get('count', 0), nonfinite=m.get('nonfinite', 0),
        precision_max=m.get('precision_max'))


def _tail(name, p, d):
    """A (b) site tail P + D_all; None (with the reasons) when either is unmeasured or has non-finite differences."""
    why = [f'{name}: {label} unmeasured' for label, x in (('P', p), ('D_all', d)) if not x.get('count') or x.get('max') is None]
    why += [f'{name}: {label} has {x["nonfinite"]} non-finite differences' for label, x in (('P', p), ('D_all', d)) if x.get('nonfinite')]
    return dict(tail=None if why else p['max'] + d['max'], p=p, d_all=dict(quantity=d.get('quantity'), max=d.get('max'), count=d.get('count', 0),
        nonfinite=d.get('nonfinite', 0), precision_max=d.get('precision_max'), f32_adapter_bound=d.get('f32_adapter_bound'))), why


def fold(earlier, d, check=None, p=None):
    """FOLD_RULE: the (b) site tails and the bounds with them folded in, from the earlier report's geometry section, this run's P (p) and
    D_all (d). Every bound names the term that sets it (`*_set_by`)."""
    geo = (earlier or {}).get('geometry') or {} if isinstance(earlier, dict) else {};levels = geo.get('levels') or {}
    allpos = (geo.get('proposals') or {}).get('all_positions') or {};flags = [f'source: {x}' for x in (check or {}).get('flags', [])]
    flags += [f'earlier all_positions: {x}' for x in allpos.get('flags') or []];cross = {}
    # P(t1 value): the larger of the earlier P_all(depth2.turn1 value) and this run's frames at (b)'s cells
    earlier_t1, run_t1 = _entry(levels, 'depth2.turn1', 'pricer_all', 'value'), _run_entry(p, P_VALUE)
    use = run_t1 if run_t1['max'] is not None and (earlier_t1['max'] is None or run_t1['max'] > earlier_t1['max']) else earlier_t1
    if use is run_t1:flags.append(f'P(t1 value): this run\'s {P_VALUE} {run_t1["max"]:.6g} > the earlier P_all(depth2.turn1 value) {_fmt(earlier_t1["max"])}: '
        'b.t1_value uses this run\'s')
    elif run_t1['max'] is None:flags.append(f'P(t1 value): this run\'s {P_VALUE} unmeasured: the earlier P_all only')
    cross['t1_value'] = dict(earlier=earlier_t1, this_run=run_t1, used=use['source'])
    t3 = _entry(levels, 'depth3.turn1', 'pricer_all', 'value')
    if t3['max'] is not None and use['max'] is not None and t3['max'] > use['max']:
        flags.append(f'P_all(depth3.turn1 value) {t3["max"]:.4g} > the P(t1 value) used {use["max"]:.4g}: the depth-N gate\'s (b) reads the depth-N frame\'s turn 1')
    # the legal-lane root prior: this run's frames against the earlier P_all(root opp)
    earlier_root, run_root = _entry(levels, 'root', 'pricer_all', 'opp_logits'), _run_entry(p, P_LOGITS)
    a, b = earlier_root.get('precision_max'), run_root.get('precision_max');gap = None if a is None or b is None else abs(a - b)
    cross['root_peer_logits'] = dict(earlier=earlier_root, this_run=run_root, difference=gap, tolerance=CROSS_CHECK)
    if gap is None:flags.append(f'cross-check root opp_logits: unavailable (earlier {_fmt(a)}, this run {_fmt(b)})')
    elif gap > CROSS_CHECK:flags.append(f'cross-check root opp_logits: this run\'s legal-lane pricer max {b:.6g} differs from the earlier P_all {a:.6g} by {gap:.3g} > {CROSS_CHECK}')
    t1, f1 = _tail('b.t1_value', use, dict(d.get(VALUE) or {}, quantity=VALUE))
    root, f2 = _tail('b.root_peer_logits', earlier_root, dict(d.get(LOGITS) or {}, quantity=LOGITS))
    lanes, f3 = _tail('b.root_peer_logits_all_lanes', _run_entry(p, P_ALL_LANES), dict(d.get(ALL_LANES) or {}, quantity=ALL_LANES));flags += f1 + f2 + f3
    if 'root' in levels:
        view = {'root': {c: (levels['root'].get(c) or {}) for c in ('pricer_all', 'oracle_sample')}}
        c_bound, c_flags, c_inputs = clv._bound(view, (('root', 'opp_logits'),), 'pricer_all', clv.COMBINE['all_positions'][1])
    else:c_bound, c_flags, c_inputs = None, ['no root level in the earlier report'], []
    flags += [f'root (c) tail: {x}' for x in c_flags]
    def top(name, terms):
        missing = [label for label, x in terms if x is None]
        if missing:flags.append(f'{name}: no bound (unresolved: {", ".join(missing)})');return None, None
        label, value = max(terms, key=lambda t: t[1]);return value, label
    value, value_by = top('value_bound_all_geometries', [('earlier all_positions value_bound', allpos.get('value_bound')), ('b.t1_value', t1['tail'])])
    gated, gated_by = top('value_bound_all_geometries_without_root', [('earlier all_positions value_bound_without_root', allpos.get('value_bound_without_root')),
        ('b.t1_value', t1['tail'])])
    root_all, root_by = top('root_logit_all_geometries', [('root (c) tail', c_bound), ('b.root_peer_logits', root['tail']), ('b.root_peer_logits_all_lanes', lanes['tail'])])
    return dict(label='PROPOSAL (informational, not a verdict): three geometries (the pricers, the gate oracle, D3\'s FramePricer); the declaration '
        'adds the margin', kind='all_positions_three_geometries', rule=FOLD_RULE, sites={'b.t1_value': t1, 'b.root_peer_logits': root,
        'b.root_peer_logits_all_lanes': lanes}, root_c=dict(tail=c_bound, flags=c_flags, inputs=c_inputs), cross_checks=cross,
        value_bound_all_geometries=value, value_bound_all_geometries_set_by=value_by, value_bound_all_geometries_without_root=gated,
        value_bound_all_geometries_without_root_set_by=gated_by, near_tie_q_bound_all_geometries=value, root_logit_all_geometries=root_all,
        root_logit_all_geometries_set_by=root_by, turn1_logit_bound=allpos.get('turn1_logit_bound'), deeper_logit_bound=allpos.get('deeper_logit_bound'),
        restated='turn1_logit_bound and deeper_logit_bound: the earlier all_positions proposal\'s, unchanged',
        earlier_all_positions={k: allpos.get(k) for k in ('value_bound', 'value_bound_level', 'value_bound_without_root', 'value_bound_without_root_level',
            'turn1_logit_bound', 'deeper_logit_bound', 'matches_gate_shapes')}, matches_earlier=bool((check or {}).get('matches')), margin=None, flags=flags)


def _fmt(x):
    return '-' if x is None else f'{x:.6g}'


# Driving one run ---------------------------------------------------------------------------------------------------------------------
def measure(groups, *, evaluators, frame_pricer, executor, model, materializer, recipe, observe, module=None, capacity=CAPACITY, sample=SAMPLE, seed=SEED,
        chunk=CHUNK, fixed_shapes=False, actor_capacity=None, progress=None):
    """groups [[Request]]: the requests bound once under the bf16 champion (the gate's), priced at BOTH precisions; evaluators {'f32', 'bf16'}:
    one champion at two precisions; frame_pricer the bank provider's (the gate's base); observe the ladder's champion-layout leaf
    (champion_observer). Per call (groups split at `capacity`): the depth-2 pass and D3 at both precisions, both comparisons and the anchor
    walk. Returns dict(calls, d3_options, pass_options, walk)."""
    import importlib
    if set(evaluators) != set(NAMES):raise ValueError(f'evaluators named {NAMES} required')
    o2, batch = (importlib.import_module(f'gpu_engine.{m}') for m in ('depth2_oracle', 'device_public_teacher_batch'))
    module = module or gate.pricer_module();pricers = build_d3(frame_pricer, evaluators)
    passes = build_passes(module, executor, model, evaluators, capacity=capacity, fixed_shapes=fixed_shapes)
    options = d3_options(pricers, actor_capacity=actor_capacity, split_capacity=capacity, fixed_shapes=fixed_shapes);pass_record = pass_options(passes)
    split = g2.split_groups(groups, capacity)
    walker = clv.Walker(executor, observe, chunk=chunk, view=ce.digest_view(evaluators[REFERENCE]));heads = clv.Heads(evaluators)
    stack_rows = o2.bucket(2 * capacity) if fixed_shapes else None;calls = []
    for call, reqs in enumerate(split):
        reqs = tuple(reqs);context = dict(call=call, roots=len(reqs));frames, results, costs = {}, {}, {}
        for name in NAMES:
            with stage(progress, 'depth2_pass', precision=name, **context) as info:
                clock = time.perf_counter();frames[name] = passes[name].price(reqs, materializer, recipe, (evaluators[name].params,))
                costs[f'depth2_pass.{name}'] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(frames[name], 'timing', None)))
                info.update(costs[f'depth2_pass.{name}'])
        for name in NAMES:
            with stage(progress, 'd3_frame_pricer', precision=name, **context) as info:
                clock = time.perf_counter();results[name] = pricers[name].price(reqs, materializer, recipe, (evaluators[name].params,))
                timing = getattr(pricers[name], 'last_timing', None) or {};want = bucket(pricers[name], len(reqs))
                costs[f'd3.{name}'] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(timing), world_prepare_seconds=timing.get('world_prepare_seconds'),
                    root_capacity=timing.get('root_capacity'), expected_root_capacity=want);info.update(costs[f'd3.{name}'])
                if timing.get('root_capacity', want) != want:options['departures'].append(f'call {call} {name}: compiled at {timing["root_capacity"]} roots, not the gate\'s bucket {want}')
        with stage(progress, 'compare', **context) as info:
            comparison = compare_call(results, reqs, call=call);pricer = pricer_call(frames, comparison['arrays'][TESTED], reqs, call=call);frames = None
            info.update(d3=comparison['counts'], pricer=pricer['counts'], structural={k: e['count'] for k, e in {**comparison['findings'], **pricer['findings']}.items()},
                max={f'{label}.{k}': float(np.abs(x[k]).max()) if x[k].size else None for label, x in (('d3', comparison), ('pricer', pricer)) for k in ('values', 'logits', 'all_lanes')})
        with stage(progress, 'anchor_walk', **context) as info:
            stack = o2.stack(reqs, batch._numpy_worlds(reqs, materializer, recipe), size=stack_rows)
            anchor = anchor_call(call, comparison.pop('arrays'), stack, walker, heads, sample=sample, rng=np.random.default_rng([seed, call]))
            info.update(anchor_cells=len(anchor['cells']['spec']), anchor_worlds=len(anchor['roots']['spec']), seconds_walking=walker.seconds)
        calls.append(dict(call=call, roots=context['roots'], costs=costs, comparison=comparison, pricer=pricer, anchor=anchor))
    options['matches_gate'] = not options['departures']
    return dict(calls=calls, d3_options=options, pass_options=pass_record, walk=dict(rows=walker.rows, seconds=walker.seconds, chunk=chunk, stack_rows=stack_rows,
        sample=sample, seed=seed))


def assemble(measured, *, earlier=None, check=None):
    """The report body from measure's output: both comparisons, D_all and this run's P, the anchor, the proposal, the calls, costs and the exit."""
    calls = measured['calls'];comparison = merge_comparison(calls);d = d_all(comparison['quantities']);p = p_run(comparison['pricer']['quantities'])
    anchor = anchor_section([c['anchor'] for c in calls], d);proposal = fold(earlier, d, check, p)
    compared = sum(comparison['quantities'][k].get('count', 0) for k in QUANTITIES);status = 'MEASURED' if compared else 'NOTHING_COMPARED'
    reasons = [] if compared else ['no D3 value or reply logit was compared (no root, or no world priced by both runs)']
    total = lambda name, key: sum((c['costs'].get(name) or {}).get(key) or 0. for c in calls)
    costs = {name: dict(wall_seconds=total(name, 'wall_seconds'), lower_and_compile_seconds=total(name, 'lower_and_compile_seconds'), device_seconds=total(name, 'device_seconds'))
        for name in [f'{program}.{n}' for program in ('depth2_pass', 'd3') for n in NAMES]}
    for n in NAMES:costs[f'd3.{n}']['buckets'] = sorted({c['costs'][f'd3.{n}'].get('root_capacity') for c in calls} - {None})
    summary = [dict(call=c['call'], roots=c['roots'], d3=c['comparison']['counts'], pricer=c['pricer']['counts'], structural={k: e['count'] for k, e in
        {**c['comparison']['findings'], **c['pricer']['findings']}.items()}, anchor=dict(cells=len(c['anchor']['cells']['spec']), worlds=len(c['anchor']['roots']['spec'])),
        costs=c['costs']) for c in calls]
    return dict(status=status, exit=EXIT[status], status_reasons=reasons, comparison=comparison, d_all=dict(d, rule=D_ALL_RULE), p_run=dict(p, rule=D_ALL_RULE),
        anchor=anchor, source_check=check, proposal=proposal, calls=summary, costs=dict(costs, walk=measured.get('walk')), d3_options=measured['d3_options'],
        pass_options=measured.get('pass_options'))


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
    parser.add_argument('--fixed-shapes', action='store_true', help='as the gate\'s (b) runs: the depth-2 pass at fixed_capacity; D3 is unchanged '
        '(per-call buckets); the anchor walk\'s stack fixed')
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
    """A short summary: the comparison tables with D_all and P, structural findings, the anchor and the proposal (d3_geometry.json holds all)."""
    h, c = report.get('header') or {}, report.get('comparison') or {};get = lambda d, k: (d or {}).get(k) or {}
    champ, tree, opts, popts = get(h, 'champion'), get(h, 'tree'), get(h, 'd3_options'), get(h, 'pass_options');d, p = report.get('d_all') or {}, report.get('p_run') or {}
    lines = [f'# Champion 9375: D3 FramePricer and the depth-2 pass, bf16 vs f32 at the gate\'s (b) positions ({report.get("status")}, exit {report.get("exit")}; '
        'informational)', '', f'- Source: `{json.dumps(h.get("source"), default=str)}`; roots {h.get("roots")} in {len(report.get("calls") or [])} call(s), bound once '
        f'under the bf16 champion (both precisions price them); dropped at binding {len(h.get("dropped") or [])}',
        f'- Champion archive `{champ.get("archive_sha256")}` (checkpoint `{champ.get("checkpoint_sha256")}`); tree `{tree.get("commit")}` ({tree.get("source")})']
    for name, x in (opts.get('pricers') or {}).items():
        lines.append(f'- D3 {name}: capacity {x.get("capacity")}, block_rows {x.get("block_rows")}, actor_capacity {x.get("actor_capacity")}, vectorized {x.get("vectorized")}, '
            f'host_prep {x.get("host_prep")}, evaluator {x.get("evaluator", {}).get("identity")}; buckets {get(get(report, "costs"), f"d3.{name}").get("buckets")}')
    for name, x in (popts.get('pricers') or {}).items():
        lines.append(f'- depth-2 pass {name}: depth {x.get("depth")}, k {x.get("k_own")}/{x.get("k_opp")}, k1 {x.get("k1_own")}/{x.get("k1_opp")}, capacity {x.get("capacity")}, '
            f'block_lanes {x.get("block_lanes")}, fixed_capacity {x.get("fixed_capacity")}, actor_capacity {x.get("actor_capacity")}, evaluator {x.get("evaluator", {}).get("identity")}')
    departures = (opts.get('departures') or []) + (popts.get('departures') or [])
    lines += [f'- Options match the gate\'s: {not departures}' + (f' (FLAGGED: {"; ".join(departures)})' if departures else ''), '']
    lines += ['| Quantity | n | non-finite | max | p99 | p99.9 | p99.99 | + f32 bound |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    rows = [(k, v, get(d, k).get('max')) for k, v in (c.get('quantities') or {}).items()] + [(k, v, get(p, k).get('max')) for k, v in (get(c, 'pricer').get('quantities') or {}).items()]
    lines += [f'| {k} | {v.get("count")} | {v.get("nonfinite")} | {_num(v.get("max"))} | {_num(v.get("p99"))} | {_num(v.get("p99.9"))} | {_num(v.get("p99.99"))} | {_num(x)} |'
        for k, v, x in rows]
    for label, s in (('D3', c.get('structural') or {}), ('depth-2 pass', get(c, 'pricer').get('structural') or {})):
        lines.append(f'- {label} structural differences: ' + ('; '.join(f'{k} {e.get("count")}' for k, e in (s.get('kinds') or {}).items()) or 'none'))
    a = report.get('anchor') or {}
    lines += ['', '## f32 anchor (sample)', '', a.get('reference', ''), '', '| Quantity | rows | compared | D3 f32 max | bound | within | D3 bf16 max (sample) |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for k, q in (a.get('quantities') or {}).items():
        lines.append(f'| {k} | {q.get("rows")} | {q.get("lanes", q.get("compared"))} | {_num(q.get("max"), ".3g")} | {_num(q.get("bound"), "g")} | {q.get("within")} | '
            f'{_num(get(q, "bf16").get("max"))} |')
    lines += ['', f'- Anchor flags: {"; ".join(a.get("flags") or []) or "none"}']
    pr, sc = report.get('proposal') or {}, report.get('source_check') or {}
    lines += ['', '## Three-geometry proposal (PROPOSAL, not a verdict; the declaration adds the margin)', '',
        f'- Earlier report `{sc.get("path")}`: matches this run {sc.get("matches")}' + (f' (FLAGGED: {"; ".join(sc.get("flags") or [])})' if sc.get('flags') else ''), '',
        '| Bound | value | from | margin |', '| --- | --- | --- | --- |']
    for name, x in (pr.get('sites') or {}).items():
        lines.append(f'| {name} | {_num(x.get("tail"))} | P ({get(x, "p").get("source")}) {_num(get(x, "p").get("max"))} + D_all {_num(get(x, "d_all").get("max"))} | - |')
    e = get(pr, 'earlier_all_positions')
    lines += [f'| root (c) tail | {_num(get(pr, "root_c").get("tail"))} | max(P_all + O_sample, 2 x P_all) at root opp_logits | - |',
        f'| value_bound_all_geometries | {_num(pr.get("value_bound_all_geometries"))} | set by {pr.get("value_bound_all_geometries_set_by")} (earlier {_num(e.get("value_bound"))}) | (declaration) |',
        f'| value_bound_all_geometries_without_root | {_num(pr.get("value_bound_all_geometries_without_root"))} | set by {pr.get("value_bound_all_geometries_without_root_set_by")} | (declaration) |',
        f'| near_tie_q_bound_all_geometries | {_num(pr.get("near_tie_q_bound_all_geometries"))} | = value_bound_all_geometries | (declaration) |',
        f'| root_logit_all_geometries | {_num(pr.get("root_logit_all_geometries"))} | set by {pr.get("root_logit_all_geometries_set_by")} | (declaration) |',
        f'| turn1_logit_bound | {_num(pr.get("turn1_logit_bound"))} | earlier, unchanged | (declaration) |',
        f'| deeper_logit_bound | {_num(pr.get("deeper_logit_bound"))} | earlier, unchanged | (declaration) |', '',
        f'- Proposal flags: {"; ".join(pr.get("flags") or []) or "none"}', '', 'Rules: `d3_geometry.json` header.rules (declared in gpu_engine/champion_d3_geometry.py).', '']
    return '\n'.join(lines)


def write(out, report):
    out = Path(out);body = clv.plain(report)
    (out / 'd3_geometry.json').write_text(json.dumps(body, indent=1, allow_nan=False) + '\n');(out / 'd3_geometry.md').write_text(markdown(body))


def main(argv=None):
    args = parse(argv);started = time.perf_counter();source_path = Path(args.validation_json)
    earlier = json.loads(source_path.read_text())   # read (and fail) before any loading
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    log('D3_GEOMETRY_RULES', rules=RULES, gate_d3=GATE_D3, gate_depth2_pass=GATE_PASS, split_capacity=CAPACITY, validation_json=str(source_path), sample=args.sample,
        fixed_shapes=args.fixed_shapes)
    with stage(log, 'load_chunk', d3_root=args.d3_root, chunk=args.chunk):loaded = g2.bank_chunk_requests(args)
    args.champion_precision = 'bf16';bf16 = g2.load_champion(args, loaded);evaluators = {REFERENCE: replace(bf16, precision='float32'), TESTED: bf16}
    clv.check_pair(evaluators);log('EVALUATORS', **{name: ev.header()['identity'] for name, ev in evaluators.items()})
    with stage(log, 'bind') as info:   # d3_depthn_gate.load_roots: the roots bound once under the run's (bf16) champion
        groups, root_values, dropped, checked = g2.bind_roots(loaded.groups, (bf16.params,), loaded.plan, evaluator=bf16);roots = sum(map(len, groups))
        info.update(roots=roots, dropped=len(dropped), arm_checked=checked)
    window = getattr(loaded.plan, 'historical_window', None)
    measured = measure(groups, evaluators=evaluators, frame_pricer=loaded.frame_pricer, executor=loaded.provider.executor, model=loaded.plan.model,
        materializer=loaded.provider.materializer, recipe=loaded.provider.recipe, observe=clv.champion_observer(loaded), capacity=CAPACITY, sample=args.sample,
        chunk=CHUNK, fixed_shapes=args.fixed_shapes, actor_capacity=None if window is None else window + 1, progress=log)
    check = source_check(earlier, source=loaded.source, evaluators=evaluators, roots=roots, capacity=CAPACITY, fixed_shapes=args.fixed_shapes, path=str(source_path))
    body = assemble(measured, earlier=earlier, check=check)
    import jax
    report = dict(schema=SCHEMA, status=body.pop('status'), exit=body.pop('exit'), status_reasons=body.pop('status_reasons'), informational=True,
        header=dict(rules=RULES, source=loaded.source, roots=roots, binding='d3_depthn_gate.load_roots: g2.bind_roots under the bf16 champion; both precisions '
            'price these identical requests', dropped=dropped, root_abs_value=g2.percentiles(root_values),
            champion=dict(archive=args.champion_archive, archive_sha256=args.champion_archive_sha256, checkpoint_sha256=args.champion_checkpoint_sha256),
            evaluators={name: dict(ev.header(), archive=args.champion_archive) for name, ev in evaluators.items()}, tree=tree_commit(), d3_options=body.pop('d3_options'),
            pass_options=body.pop('pass_options'), options=dict(split_capacity=CAPACITY, fixed_shapes=args.fixed_shapes, sample=args.sample, seed=SEED, walk_chunk=CHUNK),
            validation_json=str(source_path), devices=[str(x) for x in jax.devices()], seconds=time.perf_counter() - started), **body)
    write(out, report);c, a, p = report['comparison'], report['anchor'], report['proposal'];h = report['header']
    log('CHAMPION_D3_GEOMETRY_COMPLETE', status=report['status'], exit=report['exit'], out=str(out / 'd3_geometry.json'),
        compared={k: v.get('count') for k, v in {**c['quantities'], **c['pricer']['quantities']}.items()},
        max={k: v.get('max') for k, v in {**c['quantities'], **c['pricer']['quantities']}.items()},
        d_all={k: report['d_all'][k].get('max') for k in QUANTITIES}, p_run={k: report['p_run'][k].get('max') for k in PRICER_QUANTITIES},
        structural=dict(d3=c['structural']['count'], pricer=c['pricer']['structural']['count']), anchor=a['observed'], anchor_flags=len(a['flags']),
        departures=h['d3_options']['departures'] + h['pass_options']['departures'], source_flags=len(check['flags']), proposal_flags=len(p['flags']),
        proposal={k: p.get(k) for k in ('value_bound_all_geometries', 'value_bound_all_geometries_set_by', 'root_logit_all_geometries', 'root_logit_all_geometries_set_by',
            'turn1_logit_bound', 'deeper_logit_bound')})
    return report['exit']


if __name__ == '__main__':raise SystemExit(main())
