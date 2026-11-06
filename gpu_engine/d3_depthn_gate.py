"""Depth-N exactness gate: DepthNPricer(digest=True) against the independent N-step oracle (gpu_engine.depthn_oracle),
D3's FramePricer and, with --n2-equivalence, the depth-2 kernel.

Roots and evaluator handling are the depth-2 gate's (gpu_engine.d3_depth2_gate: --saved-roots or a --d3-root bank chunk,
the head override, the evaluator header, the evaluator-relative float rule and the statuses). Every call also runs the
full-width depth-2 pass (DepthNPricer(depth=2, k=9) through as_depth2_frame) on the same requests. Checks at every ply:
 (a) exact: turn 1 as depth 2 (pre, root opponent request/legal, statuses, errors, terminal values, digests, encoder
     masks, request flags) plus the ply-2 maps stored at each turn-1 cell; every intermediate node's status, error,
     terminal value, both digests, encoder masks, request flags, next-ply maps and parent links; every checked leaf's
     status, error, terminal value and digest; each engine's self-reported widths equal the REQUESTED ones (the plan
     never takes widths from the engine: prior = the request, searched ply 2 = this gate's searched_widths_reference).
     Under `searched`, the supplied ply-2 maps are checked well formed and EXACTLY equal to this gate's own host-only
     re-derivation of the documented rule (searched_maps_reference) from the full-width depth-2 pass, slot order
     included, with the per-root depth2_refused flag; the oracle then expands the maps as given.
     The full-width depth-2 pass itself is trusted via --n2-equivalence and the depth-2 gate; in the run its turn 1 is
     checked against the depth-N frame's, its enumeration against the oracle's full-width set, and every terminal/error
     leaf plus --depth2-pass-sample uniform leaves per world are re-run by the oracle (PASS_TRUST).
 (b) turn 1 against D3 (depth 2's check: discrete exact, values and the root prior within the rule).
 (c) head values (turn-1 children, every ply's nodes, the leaves) and both seats' priors at every ply against the
     oracle's own heads: values within 2 x the run's depth-1 noise floor, logits within 0.1, value comparisons below
     0.25 x the p10 learner-max own-slot spread of the last ply's parents (the misplacement guard). The p10 is over
     NON-ZERO spreads: this departs from the depth-2 gate (p10 over every cell_a2_spread) and applies to
     --n2-equivalence too; a run whose spreads are all zero skips the guard and says so.
 (d) the enumerated set at every ply (turn-1 cells, node ids, leaves) equals the oracle's.
 (e) recursive learner-max (gpu_engine.depthn_backups.learner_max_root) on the pricer frame vs the same frame with the
     oracle's leaf head values and priors on the lanes (c) checks; near ties relative to the run (depth 2's rule).
 block invariance (--block-invariance): block_lanes 729 and 2187, discrete arrays bit-equal, floats within the rule.
 refusals (reported): depthn_backups.refusal_summary (and the gate's own count) beside D3's depth-1 refusals and the
 full-width depth-2 pass's. topk_boundary (reported): per ply and seat, each pruning prior selection's rank-k vs rank-(k+1)
 gap and how often the oracle's own prior would select differently.
The oracle enumerates with its own ranking code (depthn_oracle.prior_map) over the PRICER's priors (checked by (c)).

--n2-equivalence: DepthNPricer at N=2 (full width and top --n2-k) and Depth2Pricer on the same requests;
as_depth2_frame(depth-N) vs the Depth2Frame: discrete arrays exact, world_reports equal, values and priors within the
run's rule ONLY on scored cells (elsewhere both frames hold unspecified padding), the same (e) decision per root up to
near ties; INCOMPLETE when no turn-2 leaf or no (e) root was compared. --depth3-diagnostics: full-width depth 2 and depth 3 per selection and k (live,
no oracle); --diagnostics-from DIR... recomputes them from saved frames (--save-arrays writes
frames-block{B}-call{C}-{depth2|prior-k{k}|searched-k{k}}.npz). All paths are command-line arguments; --out must be new.
"""
import argparse
import collections
from contextlib import nullcontext
import dataclasses
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import d3_depth2_gate as g2

SCHEMA = 'pokezero.depthn-gate.v1'
FIRST = g2.FIRST
MODES = ('prior', 'searched')
DIAGNOSTIC_KS = (2, 3, 4)
INVARIANCE_LANES = (729, 2187)   # --block-invariance: block_lanes priced; the last frame is the checked one
SKIP = ('nodes', 'world_reports', 'timing', 'depth', 'k_own', 'k_opp')
log, stage, Findings = g2.log, g2.stage, g2.Findings
EXPANDED = 2


# ---------------------------------------------------------------- frame access (dataclass or duck-typed)

def _fields(obj):
    if dataclasses.is_dataclass(obj):return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return dict(vars(obj)) if not isinstance(obj, dict) else dict(obj)


def _copy(obj, **changes):
    if dataclasses.is_dataclass(obj):return replace(obj, **changes)
    return SimpleNamespace(**dict(_fields(obj), **changes))


def frame_arrays(frame):
    """{name: array}: every top-level array and every ply's node array as ply{p}.{name} (the pricer's `arrays` naming)."""
    out = {}
    for name, value in _fields(frame).items():
        if name in SKIP or value is None:continue
        out[name] = np.asarray(value)
    for node in getattr(frame, 'nodes', None) or ():
        for name, value in _fields(node).items():
            if name == 'ply' or value is None:continue
            out[f'ply{node.ply}.{name}'] = np.asarray(value)
    return out


def plan_of(depth, k_own, k_opp, selection='prior', *, k1_own=9, k1_opp=9):
    """The run's declared shape: the pricer's slot widths for plies 2..N (an int repeats; under `searched` ply 2's are the
    supplied maps' widths), node counts M_p per ply (M_1 = 81) and the turn-1 widths."""
    widths = lambda k: tuple(int(x) for x in k) if isinstance(k, (tuple, list)) else (int(k),) * (depth - 1)
    k_own, k_opp = widths(k_own), widths(k_opp)
    if len(k_own) != depth - 1 or len(k_opp) != depth - 1:raise ValueError('one width per ply 2..N required')
    counts = [81]
    for p in range(2, depth):counts.append(counts[-1] * k_own[p - 2] * k_opp[p - 2])
    return dict(depth=depth, k_own=k_own, k_opp=k_opp, counts=counts, selection=selection, k1_own=k1_own, k1_opp=k1_opp)


def logits_at(frame, ply, q, w, node):
    """The pricer's (own, opp) priors at a ply-`ply` node (ply 1: turn-1 cell a1*9+r1)."""
    if ply == 1:
        a1, r1 = divmod(int(node), 9);return frame.t1_own_logits[q, w, a1, r1], frame.t1_peer_logits[q, w, a1, r1]
    level = frame.nodes[ply - 2];return level.own_logits[q, w, node], level.opp_logits[q, w, node]


def maps_at(frame, ply, q, w, node):
    """The pricer's (own, opp) maps stored at a ply-`ply` node for ply+1."""
    if ply == 1:
        a1, r1 = divmod(int(node), 9);return frame.t1_own_actions[q, w, a1, r1], frame.t1_opp_actions[q, w, a1, r1]
    level = frame.nodes[ply - 2];return level.own_actions[q, w, node], level.opp_actions[q, w, node]


def frame_geometry_n(frame, requests, plan):
    problems = [];R = requests;N = plan['depth']
    try:
        if int(frame.depth) != N:problems.append(f'depth {frame.depth} != declared {N}')
        if tuple(frame.k_own) != plan['k_own'] or tuple(frame.k_opp) != plan['k_opp']:
            problems.append(f'widths {tuple(frame.k_own)}/{tuple(frame.k_opp)} != declared {plan["k_own"]}/{plan["k_opp"]}')
    except (AttributeError, TypeError) as error:return [f'frame shape metadata: {error}']
    expect = {name: shape for name, shape in g2.SHAPES.items() if not name.startswith('t2_')}
    expect.update(t1_own_actions=(2, 9, 9, plan['k_own'][0]), t1_opp_actions=(2, 9, 9, plan['k_opp'][0]),
        leaf_status=(2, plan['counts'][-1], plan['k_own'][-1], plan['k_opp'][-1]), leaf_value=(2, plan['counts'][-1], plan['k_own'][-1], plan['k_opp'][-1]),
        leaf_error=(2, plan['counts'][-1], plan['k_own'][-1], plan['k_opp'][-1]), leaf_digest=(2, plan['counts'][-1], plan['k_own'][-1], plan['k_opp'][-1], 2))
    for name, shape in expect.items():
        value = getattr(frame, name, None)
        if value is None:problems.append(name + ':missing');continue
        if np.asarray(value).shape != (R,) + shape:problems.append(f'{name}:shape {np.asarray(value).shape} != {(R,) + shape}')
    nodes = list(getattr(frame, 'nodes', ()) or ())
    if len(nodes) != N - 2:problems.append(f'{len(nodes)} intermediate plies != {N - 2}');return problems
    for p, level in zip(range(2, N), nodes):
        M = plan['counts'][p - 1]
        for name, shape in dict(status=(2, M), value=(2, M), error=(2, M), own_requested=(2, M), opp_requested=(2, M), own_legal=(2, M, 9),
                opp_legal=(2, M, 9), own_logits=(2, M, 9), opp_logits=(2, M, 9), own_actions=(2, M, plan['k_own'][p - 1]),
                opp_actions=(2, M, plan['k_opp'][p - 1]), digest_own=(2, M, 2), digest_opp=(2, M, 2)).items():
            value = getattr(level, name, None)
            if value is None:problems.append(f'ply{p}.{name}:missing');continue
            if np.asarray(value).shape != (R,) + shape:problems.append(f'ply{p}.{name}:shape {np.asarray(value).shape} != {(R,) + shape}')
        for name in ('parent', 'parent_own_slot', 'parent_opp_slot'):
            value = getattr(level, name, None)
            if value is None or np.asarray(value).shape != (M,):problems.append(f'ply{p}.{name}:shape')
    return problems


# ---------------------------------------------------------------- oracle enumeration along the pricer's selection

def oracle_levels(*, oracle, stack, requests, recipe, params, frame, plan, selection_maps=None, sample=None, block=0, reference=None):
    """Roots, turn-1 cells and every ply's nodes and leaves, enumerated by the selection rule (prior maps from the pricer's
    checked priors through the oracle's own ranking; `searched` ply-2 maps as given). frame=None (or a frame of the wrong
    geometry) ranks by the oracle's own priors. Returns data for evaluate_n."""
    reference = reference or importlib.import_module('gpu_engine.depthn_oracle')
    N = plan['depth'];usable = frame is not None and not frame_geometry_n(frame, len(requests), plan)
    roots = oracle.roots(stack, requests, params)
    root_priors = {k: frame.root_peer_logits[k] for k in roots} if usable and plan.get('k1_opp', 9) < 9 else None
    cells = reference.turn1_cells(requests, recipe, roots, k1_own=plan.get('k1_own', 9), k1_opp=plan.get('k1_opp', 9), root_priors=root_priors)
    children = oracle.children(stack, cells, params);cells = np.asarray(cells, np.int64).reshape(-1, 4)
    level_records = {1: children};ids = {1: cells[:, 2] * 9 + cells[:, 3]};links = {};maps, leaves, enumerated, topk = {}, None, {}, {}
    if usable and plan.get('k1_opp', 9) < 9:   # the root opponent's k1_opp selection, ranked by the pricer's root prior
        for key, root in sorted(roots.items()):
            if root['pre'] == 0 and root['peer_requested']:
                boundary(topk, 'root', 'opp', root['peer_legal'], frame.root_peer_logits[key], root['peer_logits'], plan['k1_opp'], reference)
    expanded = np.flatnonzero(np.asarray(children.get('status', np.zeros(0))) == EXPANDED)
    parent_specs, parent_ids, parent_rows, ply = cells[expanded], ids[1][expanded], expanded, 1
    for p in range(2, N + 1):
        k_own, k_opp = plan['k_own'][p - 2], plan['k_opp'][p - 2];rec = level_records[ply]
        own_maps, opp_maps = np.full((len(parent_ids), k_own), -1, np.int64), np.full((len(parent_ids), k_opp), -1, np.int64)
        for t, (spec, node, i) in enumerate(zip(parent_specs.tolist(), parent_ids.tolist(), parent_rows.tolist())):
            q, w = spec[:2]
            if p == 2 and plan['selection'] == 'searched':
                a1, r1 = spec[2:4];own, opp = pad_map(selection_maps[2][0][q, w, a1, r1], k_own), pad_map(selection_maps[2][1][q, w, a1, r1], k_opp)
            else:
                own_logits, opp_logits = logits_at(frame, ply, q, w, node) if usable else (rec['own_logits'][i], rec['peer_logits'][i])
                own = reference.prior_map(rec['own_legal'][i], own_logits, bool(rec['own_requested'][i]), k_own)
                opp = reference.prior_map(rec['peer_legal'][i], opp_logits, bool(rec['peer_requested'][i]), k_opp)
                if usable:
                    for seat, legal, used, mine, requested, k in (('own', rec['own_legal'][i], own_logits, rec['own_logits'][i], rec['own_requested'][i], k_own),
                            ('opp', rec['peer_legal'][i], opp_logits, rec['peer_logits'][i], rec['peer_requested'][i], k_opp)):
                        if requested:boundary(topk, f'ply{p}', seat, legal, used, mine, k, reference)
            maps[(p, q, w, node)] = (own, opp);own_maps[t], opp_maps[t] = own, opp
        specs, child_ids, parents, own_slot, opp_slot = reference.expand_arrays(parent_specs, parent_ids, own_maps, opp_maps)
        if p < N:
            records = oracle.nodes(stack, specs, p, params);level_records[p] = records
            ids[p] = child_ids;links[p] = (parents, own_slot, opp_slot);enumerated[p] = np.stack([specs[:, 0], specs[:, 1], child_ids], axis=1)
            expanded = np.flatnonzero(np.asarray(records.get('status', np.zeros(0))) == EXPANDED)
            parent_specs, parent_ids, parent_rows, ply = specs[expanded], child_ids[expanded], expanded, p
        else:
            index = np.stack([specs[:, 0], specs[:, 1], parents, own_slot, opp_slot], axis=1);enumerated['leaf'] = index
            chosen, forced = _sample(index, sample, requests, block, frame if usable else None)
            leaves = oracle.leaves_n(stack, specs[chosen], N, params);leaves['index'] = index[chosen]
    return dict(roots=roots, children=children, cells=cells, levels=level_records, ids=ids, links=links, maps=maps, leaves=leaves,
        enumerated=enumerated, worlds=len(stack.rows), plan=plan, forced_leaves=forced, topk=topk)


def boundary(topk, ply, seat, legal, used, mine, k, reference):
    """One prior top-k selection that prunes (more legal actions than k): its rank-k minus rank-(k+1) gap under the prior that
    ranked it (the pricer's, checked by (c)) and whether the oracle's own prior would keep a different set."""
    order = reference.ranked(legal, used)
    if len(order) <= k:return
    score = lambda a: float(used[a]) if math.isfinite(float(used[a])) else -math.inf
    entry = topk.setdefault(ply, {}).setdefault(seat, dict(gaps=[], oracle_prior_set_differs=0))
    entry['gaps'].append(score(order[k - 1]) - score(order[k]));entry['oracle_prior_set_differs'] += int(set(order[:k]) != set(reference.ranked(legal, mine)[:k]))


def merge_topk_n(reports):
    """Per ply and seat over calls: pruning selections, their boundary gaps, how many fall below the logit bound and how often
    the oracle's own prior would select differently (a near tie the logit tolerance absorbs). None without pruning."""
    merged = {}
    for report in reports:
        for ply, seats in (report or {}).items():
            for seat, entry in seats.items():
                m = merged.setdefault(ply, {}).setdefault(seat, dict(gaps=[], oracle_prior_set_differs=0))
                m['gaps'].extend(entry['gaps']);m['oracle_prior_set_differs'] += entry['oracle_prior_set_differs']
    if not merged:return None
    out = {ply: {seat: dict(selections=len(m['gaps']), gap=g2.percentiles(m['gaps']), gap_below_logit_bound=int(np.count_nonzero(np.asarray(m['gaps']) < g2.LOGIT_BOUND)),
        oracle_prior_set_differs=m['oracle_prior_set_differs']) for seat, m in seats.items()} for ply, seats in sorted(merged.items())}
    return dict(out, definition='per ply (root = the k1_opp root selection) and seat, over prior selections that prune: gap = rank-k minus rank-(k+1) '
        'logit of the pricer\'s prior (the one ranked; (c) checks it within the logit bound); oracle_prior_set_differs = selections the oracle\'s '
        'own prior would make differently')


def pad_map(m, width):
    """A supplied slot -> action map padded with -1 to the pricer's slot width (a wider map is refused, as the pricer does)."""
    m = np.asarray(m, np.int64)
    if len(m) > width:raise ValueError(f'a supplied map has {len(m)} slots, wider than the pricer\'s {width}')
    return np.concatenate([m, np.full(width - len(m), -1, np.int64)])


def _sample(index, sample, requests, block, frame):
    """(chosen leaf positions, forced count): all leaves, or per world every pricer-reported terminal/error leaf plus
    `sample` uniform others (fixed seed per world)."""
    index = np.asarray(index, np.int64).reshape(-1, 5)
    if sample is None:return np.arange(len(index)), 0
    forced = np.zeros(len(index), bool)
    if frame is not None and len(index):forced = np.isin(np.asarray(frame.leaf_status)[tuple(index.T)], (1, 3))
    chosen = forced.copy();world = index[:, 0] * 2 + index[:, 1]
    for key in np.unique(world).tolist():
        q, w = divmod(key, 2);a = requests[q].plans.root.address;rng = np.random.default_rng([g2.SAMPLE_SEED, block, a.stream, a.frame, a.owner, w])
        rest = np.flatnonzero((world == key) & ~forced);chosen[rest[rng.choice(len(rest), size=min(sample, len(rest)), replace=False)]] = True
    return np.flatnonzero(chosen), int(forced.sum())


# ---------------------------------------------------------------- checks

def _viewed_own(status, error):
    return (status == 2) | ((status == 3) & np.isin(error, (3, 4, 5, 6)))


def _views_peer(status, error):
    return (status == 2) | ((status == 3) & np.isin(error, (5, 6)))


def check_exact_n(frame, data, findings):
    """(a) beyond turn 1's depth-2 fields: ply-2 maps at the cells, every ply's nodes and maps, parent links and the leaves."""
    plan = data['plan'];children = data['children']
    g2.check_exact(frame, dict(roots=data['roots'], children=children, leaves=dict(specs=np.zeros((0, 6), np.int64))), findings)
    for p_parent in range(1, plan['depth']):   # maps stored at ply p_parent nodes, for ply p_parent + 1
        records = data['levels'][p_parent];specs = records['cells'] if p_parent == 1 else records.get('specs', ())
        if not len(specs):continue
        specs = np.asarray(specs, np.int64);node = np.asarray(data['ids'][p_parent], np.int64);q, w = specs[:, 0], specs[:, 1]
        if p_parent == 1:got_own, got_opp = np.asarray(frame.t1_own_actions)[tuple(specs[:, :4].T)], np.asarray(frame.t1_opp_actions)[tuple(specs[:, :4].T)]
        else:level = frame.nodes[p_parent - 2];got_own, got_opp = np.asarray(level.own_actions)[q, w, node], np.asarray(level.opp_actions)[q, w, node]
        want_own, want_opp = np.full(got_own.shape, -1, np.int64), np.full(got_opp.shape, -1, np.int64)
        for i in np.flatnonzero(np.asarray(records['status']) == EXPANDED).tolist():
            own, opp = data['maps'][(p_parent + 1, int(q[i]), int(w[i]), int(node[i]))];want_own[i], want_opp[i] = own, opp
        got, want = np.concatenate([got_own, got_opp], axis=1), np.concatenate([want_own, want_opp], axis=1)
        findings.record(f'ply{p_parent + 1}_maps', np.stack([q, w, node], axis=1), got == want, got, want)
    for p in range(2, plan['depth']):
        records = data['levels'][p]
        if not len(records.get('specs', ())):continue
        level = frame.nodes[p - 2];q, w = records['specs'][:, 0], records['specs'][:, 1];node = np.asarray(data['ids'][p])
        at = (q, w, node);keys = np.stack([q, w, node], axis=1);status, error = records['status'], records['error']
        for name, want in (('status', status), ('error', error)):
            got = np.asarray(getattr(level, name))[at];findings.record(f'ply{p}_{name}', keys, got.astype(np.int64) == want, got, want)
        terminal = status == 1;got = np.asarray(level.value)[at]
        findings.record(f'ply{p}_value_terminal', keys[terminal], (got == records['value'])[terminal], got[terminal], records['value'][terminal])
        viewed, peer = _viewed_own(status, error), _views_peer(status, error)
        for name, rec, mask, kind in (('digest_own', 'own_digest', viewed, np.uint32), ('digest_opp', 'peer_digest', peer, np.uint32),
                ('own_legal', 'own_legal', peer, bool), ('opp_legal', 'peer_legal', peer, bool), ('own_requested', 'own_requested', peer, bool),
                ('opp_requested', 'peer_requested', peer, bool)):
            got = np.asarray(getattr(level, name))[at].astype(kind);want = np.asarray(records[rec]).astype(kind)
            findings.record(f'ply{p}_{name}', keys[mask], (got == want)[mask], got[mask], want[mask])
        parent, own_slot, opp_slot = data['links'][p]
        for name, want in (('parent', parent), ('parent_own_slot', own_slot), ('parent_opp_slot', opp_slot)):
            got = np.asarray(getattr(level, name))[node];findings.record(f'ply{p}_{name}', keys, got == want, got, want)
    leaves = data['leaves']
    if leaves is not None and len(leaves['specs']):
        at = tuple(leaves['index'].T);keys = leaves['index'];status, error = leaves['status'], leaves['error']
        for name, want in (('leaf_status', status), ('leaf_error', error)):
            got = np.asarray(getattr(frame, name))[at];findings.record(name, keys, got.astype(np.int64) == want, got, want)
        terminal = status == 1;got = np.asarray(frame.leaf_value)[at]
        findings.record('leaf_value_terminal', keys[terminal], (got == leaves['value'])[terminal], got[terminal], leaves['value'][terminal])
        viewed = (status == 4) | ((status == 3) & np.isin(error, (3, 4)));got = np.asarray(frame.leaf_digest)[at].astype(np.uint32)
        findings.record('leaf_digest', keys[viewed], (got == leaves['digest'])[viewed], got[viewed], leaves['digest'][viewed])
    return findings


def check_heads_n(frame, data, d_results=None):
    """(c): turn 1 as depth 2 (with both floors), every ply's node values and both seats' priors, and the leaf values."""
    out = g2.check_heads(frame, dict(roots=data['roots'], children=data['children'], leaves=dict(specs=np.zeros((0, 6), np.int64))), d_results)
    for p in range(2, data['plan']['depth']):
        records = data['levels'][p]
        if not len(records.get('specs', ())):continue
        level = frame.nodes[p - 2];specs = records['specs'];node = np.asarray(data['ids'][p]);at = (specs[:, 0], specs[:, 1], node)
        keys = np.stack(at, axis=1);status, error = records['status'], records['error']
        same = (np.asarray(level.status)[at] == status) & (np.asarray(level.error)[at] == error)
        valued = same & ((status == 2) | np.isin(error, (5, 6)));expanded = same & (status == 2)
        parts = {f'ply{p}_value_head': g2._masked(keys, np.asarray(level.value)[at], records['head_value'], valued)}
        if 'raw_head_value' in records:parts[f'raw_head_ply{p}_value'] = g2._masked(keys, np.asarray(level.value)[at], records['raw_head_value'], valued)
        for seat, rec, scope in (('own', 'own', valued), ('opp', 'peer', expanded)):
            lanes = scope[:, None] & np.asarray(records[rec + '_requested'], bool)[:, None] & np.asarray(records[rec + '_legal'], bool)
            parts[f'ply{p}_{seat}_logits'] = g2._masked(keys, np.asarray(getattr(level, f'{seat}_logits'))[at], records[rec + '_logits'], lanes)
        out.update({name: g2._numeric(*part) for name, part in parts.items()})
    leaves = data['leaves']
    if leaves is not None and len(leaves['specs']):
        at = tuple(leaves['index'].T);scored = (leaves['status'] == 4) & (np.asarray(frame.leaf_status)[at] == 4)
        out['leaf_value'] = g2._numeric(*g2._masked(leaves['index'], np.asarray(frame.leaf_value)[at], leaves['head_value'], scored))
        if 'raw_head_value' in leaves:
            out['raw_head_leaf_value'] = g2._numeric(*g2._masked(leaves['index'], np.asarray(frame.leaf_value)[at], leaves['raw_head_value'], scored))
    return out


def check_enumeration_n(frame, data, findings):
    """(d) at every ply: turn-1 cells, node ids and leaves (status != 0) equal the oracle's."""
    plan = data['plan'];sets = [('t1', np.asarray(frame.t1_status), data['cells'])]
    sets += [(f'ply{p}', np.asarray(frame.nodes[p - 2].status), data['enumerated'].get(p, np.zeros((0, 3), np.int64))) for p in range(2, plan['depth'])]
    sets.append(('leaf', np.asarray(frame.leaf_status), data['enumerated'].get('leaf', np.zeros((0, 5), np.int64))))
    for name, status, expected in sets:
        expected = np.asarray(expected, np.int64).reshape(-1, status.ndim)
        got, want = np.flatnonzero(status != 0), np.unique(np.ravel_multi_index(expected.T, status.shape) if len(expected) else np.zeros(0, np.int64))
        findings.compared[f'{name}_enumerated'] += len(np.union1d(got, want))
        for label, flat, value in ((f'{name}_missing_from_pricer', np.setdiff1d(want, got), 'not enumerated'), (f'{name}_extra_in_pricer', np.setdiff1d(got, want), 'enumerated')):
            keys = np.stack(np.unravel_index(flat, status.shape), axis=1) if len(flat) else np.zeros((0, status.ndim), np.int64)
            findings.fail(label, keys, [value] * len(keys), ['enumerated' if value == 'not enumerated' else 'not enumerated'] * len(keys))
    return findings


def check_oracle_n(data, findings):
    """The oracle's own paths: every node and leaf it computed passed only through expanded nodes (none broken)."""
    for p, records in data['levels'].items():
        if p == 1 or not len(records.get('specs', ())):continue
        broken = np.asarray(records['broken_path'], bool);findings.record(f'ply{p}_path', records['specs'], ~broken, broken, np.zeros_like(broken))
    leaves = data['leaves']
    if leaves is not None and len(leaves['specs']):
        broken = np.asarray(leaves['broken_path'], bool);findings.record('leaf_path', leaves['specs'], ~broken, broken, np.zeros_like(broken))
    return findings


def node_spreads(frame):
    """Per expanded parent of the last ply with two enumerated own slots and no errored leaf: max - min over those slots of
    sum_j pi_opp(j) v(i, j) (leaf values; pi_opp the masked softmax of the parent's opponent prior over its enumerated
    slots). The guard's spread (depth 2's cell_a2_spread one ply down)."""
    status, value = np.asarray(frame.leaf_status), np.asarray(frame.leaf_value, np.float64)
    R, _, M, k_own, k_opp = status.shape
    if int(frame.depth) == 2:
        opp_map = np.asarray(frame.t1_opp_actions).reshape(R, 2, 81, k_opp);logits = np.asarray(frame.t1_peer_logits, np.float64).reshape(R, 2, 81, 9)
    else:
        level = frame.nodes[-1];opp_map, logits = np.asarray(level.opp_actions), np.asarray(level.opp_logits, np.float64)
    enumerated = status != 0;own, opp = enumerated.any(axis=-1), enumerated.any(axis=-2)
    keep = (own.sum(axis=-1) >= 2) & ~(status == 3).any(axis=(-1, -2))
    if not keep.any():return []
    lanes = np.take_along_axis(logits, np.clip(opp_map, 0, 8).astype(np.int64), axis=-1)
    lanes = np.where(opp & (opp_map >= 0), lanes, -np.inf)[keep];prior = np.exp(lanes - lanes.max(axis=-1, keepdims=True))
    prior /= prior.sum(axis=-1, keepdims=True)
    rows = np.sum(prior[:, None, :] * np.where(enumerated[keep], value[keep], 0.), axis=-1)
    rows_own = own[keep]
    return (np.where(rows_own, rows, -np.inf).max(axis=-1) - np.where(rows_own, rows, np.inf).min(axis=-1)).tolist()


def oracle_heads_frame_n(frame, data):
    """A copy of the frame with the oracle's heads exactly where (c) compares them: leaf values at head-valued leaves;
    each requested seat's legal lanes of the priors at expanded nodes of every ply and the turn-1 cells; the root opponent
    prior. Statuses, errors, terminal values, maps and enumeration are the pricer's."""
    d2_copy, counts = g2.oracle_heads_frame(SimpleNamespace(**{k: v for k, v in frame_arrays(frame).items() if not k.startswith('ply')},
        t2_status=np.zeros((1,), np.int8), t2_value=np.zeros((1,), np.float32)), dict(roots=data['roots'], children=data['children'],
        leaves=dict(specs=np.zeros((0, 6), np.int64))))
    changes = {name: getattr(d2_copy, name) for name in ('t1_own_logits', 't1_peer_logits', 'root_peer_logits')}
    leaves = data['leaves'];value = np.array(frame.leaf_value, np.float32, copy=True);replaced = 0
    if leaves is not None and len(leaves['specs']):
        at = tuple(leaves['index'].T);head = (leaves['status'] == 4) & (np.asarray(frame.leaf_status)[at] == 4)
        value[tuple(leaves['index'][head].T)] = leaves['head_value'][head];replaced = int(head.sum())
    changes['leaf_value'] = value;unmatched = int(np.count_nonzero(np.asarray(frame.leaf_status) == 4)) - replaced
    nodes = []
    for p, level in zip(range(2, data['plan']['depth']), frame.nodes):
        records = data['levels'][p];fields = {}
        if len(records.get('specs', ())):
            specs = records['specs'];node = np.asarray(data['ids'][p]);at = (specs[:, 0], specs[:, 1], node)
            expanded = (records['status'] == 2) & (np.asarray(level.status)[at] == 2)
            for seat, rec in (('own', 'own'), ('opp', 'peer')):
                lanes = expanded[:, None] & np.asarray(records[rec + '_requested'], bool)[:, None] & np.asarray(records[rec + '_legal'], bool)
                rows, lane = np.nonzero(lanes);logits = np.array(getattr(level, f'{seat}_logits'), np.float32, copy=True)
                logits[specs[rows, 0], specs[rows, 1], node[rows], lane] = np.asarray(records[rec + '_logits'], np.float32)[rows, lane]
                fields[f'{seat}_logits'] = logits
        nodes.append(_copy(level, **fields))
    return _copy(frame, nodes=tuple(nodes), **changes), dict(counts, leaf_value=replaced, unmatched_head_leaves=unmatched)


def decide_roots(original, swapped, requests, backups, *, context=None, q_noise_bound=None, counts=None):
    """(e)'s per-root comparison (depth 2's recording and verdict) between two frames under `backups.learner_max_root`."""
    result = dict(roots=len(requests), roots_compared=0, both_refused=0, status_differs=0, nonfinite_roots=0, argmax_q_differs=0,
        argmax_target_differs=0, cells_compared=0, cell_choice_differs=0, max_abs_delta_q=0., max_abs_delta_target=0., replaced=counts or {},
        differing=[], gaps=[], differing_roots=[])
    for q, request in enumerate(requests):
        logits, legal = tuple(map(float, request.root_logits)), tuple(map(bool, request.root_legal))
        a, b = (backups.learner_max_root(f, q, logits, legal) for f in (original, swapped))
        ia, ib = a.improvement, b.improvement
        if ia is None and ib is None:result['both_refused'] += 1
        qa, qb = (None if i is None else i.completed_q for i in (ia, ib));ta, tb = (None if i is None else i.policy_target for i in (ia, ib))
        pick = lambda v: None if v is None else g2._argmax(v, legal)
        q_differs, t_differs = pick(qa) != pick(qb), pick(ta) != pick(tb);status_differs = (ia is None) != (ib is None)
        result['status_differs'] += int(status_differs)
        finite = all(v is None or math.isfinite(float(v)) for vector in (qa, qb, ta, tb) if vector is not None for v in vector)
        result['nonfinite_roots'] += int(not finite)
        address = request.plans.root.address;where = dict(context or {}, root=q, address=[address.stream, address.frame, address.owner])
        gaps = dict(gap_pricer=None, gap_oracle=None)
        if ia is not None and ib is not None and finite:
            result['roots_compared'] += 1;gaps = dict(gap_pricer=g2._top_two_gap(qa, legal), gap_oracle=g2._top_two_gap(qb, legal))
            result['max_abs_delta_q'] = max([result['max_abs_delta_q']] + [abs(x - y) for x, y, ok in zip(qa, qb, legal) if ok and x is not None and y is not None])
            result['max_abs_delta_target'] = max([result['max_abs_delta_target']] + [abs(x - y) for x, y in zip(ta, tb)])
            result['gaps'].append(dict(where, **gaps))
        result['argmax_q_differs'] += int(q_differs);result['argmax_target_differs'] += int(t_differs)
        if q_differs or t_differs or not finite or status_differs:
            result['differing_roots'].append(q)
            result['differing'].append(dict(where, status=[a.status, b.status], status_differs=status_differs, finite=finite,
                argmax_q_differs=q_differs, argmax_target_differs=t_differs, completed_q_pricer=g2._plain_q(qa), completed_q_oracle_heads=g2._plain_q(qb),
                target_pricer=g2._plain_q(ta), target_oracle_heads=g2._plain_q(tb), argmax_q=[pick(qa), pick(qb)], argmax_target=[pick(ta), pick(tb)], **gaps))
    return g2.decision_verdict(result, q_noise_bound)


def decision_check_n(frame, data, requests, *, context=None, backups=None, q_noise_bound=None):
    """(e) at depth N: recursive learner-max on the pricer frame vs its oracle-heads copy."""
    if backups is None:
        try:backups = importlib.import_module('gpu_engine.depthn_backups')
        except ImportError as error:
            return dict(passed=False, reason='gpu_engine.depthn_backups is not importable (fail closed): ' + str(error), roots=len(requests))
    swapped, counts = oracle_heads_frame_n(frame, data)
    return decide_roots(frame, swapped, requests, backups, context=context, q_noise_bound=q_noise_bound, counts=counts)


def evaluate_n(frame, d_results, requests, data, *, context=None, full_turn1=True, progress=None):
    """All checks for ONE depth-N pricing call (floats are bounded at the run level by g2.summarize)."""
    context = context or {};geometry = frame_geometry_n(frame, len(requests), data['plan'])
    if geometry:
        bad = dict(passed=False, compared={}, mismatched=dict(frame_geometry=len(geometry)),
            first_mismatches=[dict(context, field='frame_geometry', problems=geometry[:FIRST])])
        return dict(a=bad, b=dict(bad, values={}), c={}, d=bad, oracle=check_oracle_n(data, Findings(context)).report(), geometry=geometry)
    with stage(progress, 'check_a', **context) as info:a = check_exact_n(frame, data, Findings(context)).report();info['passed'] = a['passed']
    with stage(progress, 'check_b', **context) as info:
        found = g2.check_d(frame, d_results, requests, Findings(context), full_turn1=full_turn1);b = dict(found.report(), values=found.values)
        info['passed'] = b['passed']
    with stage(progress, 'check_c', **context):c = check_heads_n(frame, data, d_results)
    with stage(progress, 'check_d', **context) as info:d = check_enumeration_n(frame, data, Findings(context)).report();info['passed'] = d['passed']
    with stage(progress, 'check_oracle_consistency', **context):oracle = check_oracle_n(data, Findings(context)).report()
    return dict(a=a, b=b, c=c, d=d, oracle=oracle, geometry=[], spreads=node_spreads(frame))


def guard_spread(spreads):
    """(p10, description) of the misplacement guard's spread: over parents with a NON-ZERO own-slot spread (identical rows
    cannot reveal a misplaced value, and a zero p10 would make every comparison fail the strict guard); None if none."""
    spreads = np.asarray(list(spreads), np.float64);positive = spreads[spreads > 0]
    return (float(np.percentile(positive, 10)) if len(positive) else None), dict(parents=int(len(spreads)), nonzero=int(len(positive)),
        zero_share=None if not len(spreads) else float(np.mean(spreads == 0)),
        source='p10 over the last ply\'s expanded parents (two or more own slots, no errored leaf) of max - min over own slots of '
            'sum_j pi_opp(j) v(i, j), parents with a zero spread excluded')


GUARD_SPREAD_RULE = ('the misplacement guard is 0.25 x the p10 over NON-ZERO last-ply own-slot spreads; this departs from the depth-2 gate '
    '(p10 over every cell_a2_spread) and applies to --n2-equivalence too')


def guard_note(rule, spread):
    """The rule with the guard's spread section and, when the guard is skipped, the actual reason."""
    rule = dict(rule, spread=spread, guard_spread_rule=GUARD_SPREAD_RULE)
    if rule.get('guard_bound') is None:
        rule['guard'] = (f'skipped: every one of the {spread["parents"]} last-ply parent spreads is zero (identical rows), so the p10 over '
            'non-zero spreads is undefined') if spread['parents'] else 'skipped: no expanded last-ply parent with two own slots and no errored leaf'
    return rule


def summarize_n(evaluations):
    """g2.summarize with the guard's spread taken from the last ply's learner-max own-slot spreads of the same frames."""
    p10, spread = guard_spread(s for e in evaluations for s in e.get('spreads', ()))
    result = g2.summarize(evaluations, spread_p10=p10)
    result['tolerance'] = guard_note(result['tolerance'], spread)
    return result


def split_searched(result):
    """(own, opp, extras) from searched_maps' return: (own, opp) or (own, opp, *extras)."""
    if isinstance(result, dict):own, opp = result[2] if 2 in result else (result['own'], result['opp']);return np.asarray(own), np.asarray(opp), ()
    result = tuple(result);return np.asarray(result[0]), np.asarray(result[1]), result[2:]


def searched_maps_wellformed(depth2_frame, own_maps, opp_maps, findings):
    """(a) for supplied `searched` ply-2 maps, rule-independent: per enumerated turn-1 cell of the full-width depth-2 pass, a
    non-expanded cell keeps every slot -1; at an expanded cell a waiting seat keeps [0, -1, ...] and a requested seat's live
    slots are distinct actions of its encoder-legal mask, then -1 padding. Prior top-1 membership is reported, not gated
    (the selection rule itself is the pricer's: the oracle takes the maps as given)."""
    f = depth2_frame;cells = np.argwhere(np.asarray(f.t1_status) != 0);top1 = dict(own=[0, 0], opp=[0, 0])
    for seat, maps, requested, legal, logits in (('own', own_maps, f.t1_own_requested, f.t1_own_legal, f.t1_own_logits),
            ('opp', opp_maps, f.t1_peer_requested, f.t1_peer_legal, f.t1_peer_logits)):
        rows = []
        for cell in map(tuple, cells.tolist()):
            m = np.asarray(maps[cell]);live = m[m >= 0]
            if f.t1_status[cell] != 2:ok, want = bool((m == -1).all()), 'all -1 (not expanded)'
            elif not requested[cell]:ok, want = bool(m[0] == 0 and (m[1:] == -1).all()), '[0, -1, ...] (waiting)'
            else:
                lanes = np.asarray(legal[cell], bool);padded = (m[len(live):] == -1).all() and (m[:len(live)] >= 0).all()
                ok = bool(len(live) and padded and len(set(live.tolist())) == len(live) and (live <= 8).all() and lanes[np.clip(live, 0, 8)].all())
                want = 'distinct encoder-legal actions, then -1'
                ranked = sorted((a for a in range(9) if lanes[a]), key=lambda a: (-(float(logits[cell][a]) if math.isfinite(float(logits[cell][a])) else -math.inf), a))
                if ranked:top1[seat][0] += 1;top1[seat][1] += int(ranked[0] in live.tolist())
            rows.append((cell, ok, m.tolist(), want))
        if rows:
            keys, equal, got, want = zip(*rows)
            findings.record(f'searched_maps_{seat}', [list(k) for k in keys], np.asarray(equal), g2._objects(got), g2._objects(want))
    return dict(prior_top1_kept={seat: dict(cells=n, kept=k) for seat, (n, k) in top1.items()},
        widths=dict(own=int(np.asarray(own_maps).shape[-1]), opp=int(np.asarray(opp_maps).shape[-1])))


def searched_widths_reference(k_own, k_opp):
    """The documented `searched` slot widths: the learner's k_own plus its prior top-1; the opponent's prior top-1 plus one
    refutation per kept learner move, or its k_opp + 1 fill, whichever is wider; nine legal actions bound both at 9."""
    return min(k_own + 1, 9), min(max(k_opp + 1, k_own + 2), 9)


def _top1(legal, logits):
    """The highest-logit legal action, lower action on ties (None when nothing is legal)."""
    best = None
    for a in range(9):
        if legal[a] and (best is None or float(logits[a]) > float(logits[best])):best = a
    return best


def _dedupe(items):
    out = []
    for x in items:
        if x is not None and int(x) not in out:out.append(int(x))
    return out


def searched_maps_reference(frame, k_own, k_opp):
    """This gate's own host-only re-derivation of the `searched` ply-2 maps from a FULL-WIDTH depth-2 frame, from the
    documented rule (never the pricer's code; the pricer's rule is the specification). Per expanded turn-1 cell, valid
    leaves are status 1 or 4:
    - the learner's KEPT set (requested): the k_own a2 with the highest q(a2) = sum over the row's valid r2 of
      pi_opp(r2) v(a2, r2) / sum over the same r2 of pi_opp(r2), pi_opp the masked softmax of the cell's opponent prior
      over ALL enumerated replies (q descending, lower action on ties; a row without a valid leaf has no q), then its prior
      top-1 (highest child logit over its encoder-legal actions, lower action on ties) appended if absent;
    - the opponent's ranked set (requested): its prior top-1, the refutation of each KEPT learner move in kept order (argmin
      over the row's valid r2 of v, lower action on ties), then, while it has fewer than k_opp + 1 entries, the lowest
      threat t(r2) = min over the KEPT a2 with a valid leaf in r2 of v(a2, r2) (ascending, lower action on ties);
    - errored (status 3) pairs, sorted by (a2, r2), never feed q, refutations or the fill: their rows (first-occurrence
      order) are PREPENDED to the learner's kept set and their columns (first-occurrence order over the sorted pairs) to
      the opponent's ranked set, later duplicates dropped; each list is then TRUNCATED from the end to its fixed width, so
      errored entries occupy slots and displace the tail (the lexicographically first errored pair stays reachable);
    - a waiting seat keeps [0, -1, ...]; a cell that is not expanded keeps all -1.
    Returns int32 [R,2,9,9,w_own], [R,2,9,9,w_opp] with (w_own, w_opp) = searched_widths_reference(k_own, k_opp)."""
    w_own, w_opp = searched_widths_reference(k_own, k_opp);f = frame;R = np.asarray(f.pre).shape[0]
    own = np.full((R, 2, 9, 9, w_own), -1, np.int32);opp = np.full((R, 2, 9, 9, w_opp), -1, np.int32)
    t1, t2, value = np.asarray(f.t1_status), np.asarray(f.t2_status), np.asarray(f.t2_value, np.float64)
    for cell in map(tuple, np.argwhere(t1 == 2).tolist()):
        s, v = t2[cell], value[cell];valid = (s == 1) | (s == 4);errors = [tuple(x) for x in np.argwhere(s == 3).tolist()]
        rows = [a for a in range(9) if (s[a] != 0).any()];cols = [r for r in range(9) if (s[:, r] != 0).any()]
        live = {a: [r for r in cols if valid[a, r]] for a in rows}
        if f.t1_own_requested[cell]:
            peer = np.asarray(f.t1_peer_logits[cell], np.float64);q = {}
            top = max(peer[r] for r in cols);weights = {r: math.exp(peer[r] - top) for r in cols};total = math.fsum(weights.values())
            pi = {r: x / total for r, x in weights.items()}   # the opponent's masked softmax over ALL enumerated replies
            for a in rows:
                if live[a]:q[a] = math.fsum(pi[r] * v[a, r] for r in live[a]) / math.fsum(pi[r] for r in live[a])   # renormalized over valid r2
            kept = _dedupe(sorted(q, key=lambda a: (-q[a], a))[:k_own] + [_top1(f.t1_own_legal[cell], f.t1_own_logits[cell])])
            learner = _dedupe([a for a, _ in errors] + kept)[:w_own]
        else:kept = learner = [0]
        if f.t1_peer_requested[cell]:   # refutations and the fill are over the value-ranked KEPT moves, errored rows apart
            ranked = _dedupe([_top1(f.t1_peer_legal[cell], f.t1_peer_logits[cell])] + [min(live[a], key=lambda r: (v[a, r], r)) for a in kept if live.get(a)])
            threat = {r: min(v[a, r] for a in kept if a in live and valid[a, r]) for r in cols if any(a in live and valid[a, r] for a in kept)}
            for r in sorted(threat, key=lambda r: (threat[r], r)):
                if len(ranked) >= k_opp + 1:break
                if r not in ranked:ranked.append(r)
            opponent = _dedupe([r for _, r in errors] + ranked)[:w_opp]
        else:opponent = [0]
        own[cell][:len(learner)] = learner;opp[cell][:len(opponent)] = opponent
    return own, opp


def check_searched_rule(depth2_frame, own_maps, opp_maps, k_own, k_opp, findings, extras=()):
    """(a) for `searched`: the supplied ply-2 maps EXACTLY equal searched_maps_reference on every enumerated turn-1 cell of the
    full-width depth-2 pass (slots in the documented order), at the documented widths; a per-root [R] bool extra (the
    pricer's depth2_refused) equals the depth-2 reference refusal of the same frame."""
    mine_own, mine_opp = searched_maps_reference(depth2_frame, k_own, k_opp);cells = np.argwhere(np.asarray(depth2_frame.t1_status) != 0)
    widths = (int(np.asarray(own_maps).shape[-1]), int(np.asarray(opp_maps).shape[-1]));want = (mine_own.shape[-1], mine_opp.shape[-1])
    findings.record('searched_maps_widths', [[0]], [widths == want], g2._objects([list(widths)]), g2._objects([list(want)]))
    for seat, given, mine in (('own', own_maps, mine_own), ('opp', opp_maps, mine_opp)):
        width = max(np.asarray(given).shape[-1], mine.shape[-1]);at = tuple(cells.T)
        pad = lambda m: np.concatenate([np.asarray(m)[at], np.full(np.asarray(m)[at].shape[:-1] + (width - np.asarray(m).shape[-1],), -1)], axis=-1)
        got, expect = pad(given), pad(mine);findings.record(f'searched_maps_{seat}_rule', cells, (got == expect).all(axis=-1), got, expect)
    R = np.asarray(depth2_frame.pre).shape[0]
    for x in extras:
        flag = np.asarray(x)
        if flag.shape == (R,) and flag.dtype == bool:
            narrow = isinstance(depth2_frame.timing, dict) and int(depth2_frame.timing.get('k1_own', 9)) < 9
            expect = g2.depth2_refused(depth2_frame) | (narrow & (np.asarray(depth2_frame.pre) == 0).any(axis=1))
            findings.record('searched_depth2_refused', np.arange(R)[:, None], flag == expect, flag, expect)
    return findings


# ---------------------------------------------------------------- one gate call, refusals, saved frames

def pricer_module():
    return importlib.import_module('gpu_engine.device_depthn_pricer')


def depthn_refused(frame, root_legal=None):
    """The gate's own reference-rule count at depth N: a world with pre != 0, any status-3 cell at any ply or leaf, or (given
    root_legal) a legal root action with no enumerated turn-1 row in a usable world."""
    pre = np.asarray(frame.pre);R = len(pre)
    refused = np.any(pre != 0, axis=1) | np.any(np.asarray(frame.t1_status).reshape(R, -1) == 3, axis=1)
    refused |= np.any(np.asarray(frame.leaf_status).reshape(R, -1) == 3, axis=1)
    for level in frame.nodes:refused |= np.any(np.asarray(level.status).reshape(R, -1) == 3, axis=1)
    if root_legal is not None:
        rows = (np.asarray(frame.t1_status) != 0).any(axis=-1)
        refused |= (np.asarray(root_legal, bool)[:, None, :] & ~rows & (pre == 0)[:, :, None]).reshape(R, -1).any(axis=1)
    return refused


def refusal_report_n(frame, d_results=None, depth2_frame=None, root_legal=None):
    """Depth-N refusals (depthn_backups.refusal_summary when importable, the gate's own count beside it), D3's depth-1
    refusals and the full-width depth-2 pass's on the same requests."""
    refused = depthn_refused(frame, root_legal)
    own = dict(source='gate count of the frame', root_count=len(refused), refused_count=int(refused.sum()), refused_roots=np.flatnonzero(refused).tolist())
    report = dict(depthn=own, depth=int(frame.depth))
    try:
        summary = importlib.import_module('gpu_engine.depthn_backups').refusal_summary(frame, root_legal=None if root_legal is None else np.asarray(root_legal, bool))
        keys = ('root_count', 'refused_count', 'depth1_refused_count', 'deeper_refused_count', 'refused_by_cause', 'errors_by_ply', 'errors_by_code')
        picked = json.loads(json.dumps({k: summary[k] for k in keys if k in summary}, default=str))
        report['depthn'] = dict(picked, source='depthn_backups.refusal_summary', refused_roots=[int(x['root']) for x in summary.get('roots', ()) if x.get('refused')])
        report['depthn_gate_count'] = own;report['depthn_sources_agree'] = picked.get('refused_count') == own['refused_count']
    except (ImportError, AttributeError, KeyError, TypeError, ValueError) as error:report['refusal_summary'] = f'{type(error).__name__}: {error}'
    if d_results is not None:
        report['depth1'] = dict(source='FramePricer(vectorized=True, host_prep=True)', root_count=len(d_results),
            refused_count=sum(r.status == 'REFUSED' for r in d_results), refused_roots=[i for i, r in enumerate(d_results) if r.status == 'REFUSED'])
    if depth2_frame is not None:
        report['depth2_full_width'] = g2.refusal_report(depth2_frame, root_legal=root_legal)['depth2']
    return report


def merge_refusals_n(reports):
    merged = g2.merge_refusals(reports)
    for depth in ('depthn', 'depthn_gate_count', 'depth2_full_width'):
        if depth in merged:m = merged[depth];m['refused_share'] = None if not m.get('root_count') else m['refused_count'] / m['root_count']
    return merged


def save_frame(path, frame, requests, *, meta=None):
    """A depth-N (or depth-2 surface) frame's arrays (frame_*), the requests' root logits/legal/address and meta, as .npz."""
    arrays = {'frame_' + name: value for name, value in frame_arrays(frame).items() if value.dtype.kind in 'biufc'}
    arrays.update(request_root_logits=np.asarray([r.root_logits for r in requests], np.float32).reshape(-1, 9),
        request_root_legal=np.asarray([r.root_legal for r in requests], bool).reshape(-1, 9),
        request_address=np.asarray([[r.plans.root.address.stream, r.plans.root.address.frame, r.plans.root.address.owner] for r in requests], np.int64).reshape(-1, 3))
    shape = dict(depth=int(getattr(frame, 'depth', 2)), k_own=list(getattr(frame, 'k_own', ()) or ()), k_opp=list(getattr(frame, 'k_opp', ()) or ()))
    arrays['meta'] = np.asarray(json.dumps(dict(meta or {}, **shape, schema=SCHEMA), default=str))
    np.savez_compressed(path, **arrays)


def load_frame(path):
    """(frame, requests, meta) from save_frame: a duck-typed frame (nodes rebuilt per ply) that depthn_backups and
    as_depth2_frame read like the pricer's own."""
    with np.load(path, allow_pickle=False) as saved:arrays = {k: saved[k] for k in saved.files}
    meta = json.loads(str(arrays.pop('meta')));top, plies = {}, collections.defaultdict(dict)
    for key, value in arrays.items():
        if not key.startswith('frame_'):continue
        name = key[len('frame_'):]
        if name.startswith('ply') and '.' in name:p, field = name[3:].split('.', 1);plies[int(p)][field] = value
        else:top[name] = value
    nodes = tuple(SimpleNamespace(ply=p, **fields) for p, fields in sorted(plies.items()))
    frame = SimpleNamespace(**top, nodes=nodes, depth=meta['depth'], k_own=tuple(meta['k_own']), k_opp=tuple(meta['k_opp']),
        world_reports=(), timing=meta.get('timing') or {})
    for name in ('t1_digest_own', 't1_digest_peer', 'leaf_digest', 't2_digest'):
        if not hasattr(frame, name):setattr(frame, name, None)
    requests = [SimpleNamespace(root_logits=tuple(map(float, l)), root_legal=tuple(map(bool, g)), root_inputs={},
        plans=SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=int(a[0]), frame=int(a[1]), owner=int(a[2])))))
        for l, g, a in zip(arrays['request_root_logits'], arrays['request_root_legal'], arrays['request_address'])]
    return frame, requests, meta


class DepthNPricers:
    """The run's DepthNPricer instances, one per block_lanes value (the last prices the checked frame). k_own/k_opp are the
    constructor's per-ply k's: under `searched` the ply-2 entries are searched_maps' k's and the pricer derives its ply-2 slot
    widths (searched_widths), so the plan's widths are read back from the engine (engine.k_own/k_opp hold widths,
    engine.searched_k the k's); plies past 2 are prior widths in both modes."""
    def __init__(self, module, executor, model, *, depth, k_own, k_opp, selection, capacity, actor_capacity, lanes=(2187,), k1_own=9,
                 k1_opp=9, digest=True):
        self.module, self.executor, self.model, self.depth, self.selection = module, executor, model, depth, selection
        shape = plan_of(depth, k_own, k_opp);self.k_own, self.k_opp = shape['k_own'], shape['k_opp']
        self.capacity, self.actor_capacity, self.lanes, self.k1_own, self.k1_opp, self.digest = capacity, actor_capacity, tuple(lanes), k1_own, k1_opp, digest
        self.instances = None

    def expected_widths(self):
        """The slot widths the REQUEST implies, never the engine's self-report: under prior the requested k's; under searched
        ply 2 is this gate's own searched_widths_reference(k_own[0], k_opp[0]) and plies 3..N the requested k's."""
        if self.selection != 'searched':return self.k_own, self.k_opp
        w_own, w_opp = searched_widths_reference(self.k_own[0], self.k_opp[0]);return (w_own,) + self.k_own[1:], (w_opp,) + self.k_opp[1:]

    def engine_widths(self):
        """Each instance's self-reported (k_own, k_opp, searched_k), for the report and the engine_widths check."""
        return [dict(k_own=list(map(int, getattr(e, 'k_own', ()) or ())), k_opp=list(map(int, getattr(e, 'k_opp', ()) or ())),
            searched_k=None if getattr(e, 'searched_k', None) is None else list(map(int, e.searched_k))) for e in (self.instances or ())]

    def get(self):
        """(plan, [pricer per block_lanes value]). The plan's widths are the REQUESTED ones (expected_widths): the frame's
        geometry is checked against them, so an engine that prices narrower than asked fails whatever it reports."""
        if self.instances is None:
            self.instances = [self.module.DepthNPricer(self.executor, self.model, capacity=self.capacity, depth=self.depth, k_own=self.k_own,
                k_opp=self.k_opp, k1_own=self.k1_own, k1_opp=self.k1_opp, block_lanes=lanes, selection=self.selection, digest=self.digest,
                actor_capacity=self.actor_capacity) for lanes in self.lanes]
        k_own, k_opp = self.expected_widths()
        return plan_of(self.depth, k_own, k_opp, self.selection, k1_own=self.k1_own, k1_opp=self.k1_opp), self.instances

    def maps(self, depth2_frame):
        """(selection_maps, the maps' widths, extras) from the public searched_maps(frame, k_own[0], k_opp[0]) under `searched`
        (the same k's the pricer was built with); (None, None, ()) under prior."""
        if self.selection != 'searched':return None, None, ()
        own, opp, extras = split_searched(self.module.searched_maps(depth2_frame, self.k_own[0], self.k_opp[0]))
        return {2: (own, opp)}, (int(own.shape[-1]), int(opp.shape[-1])), extras

    def options(self):
        engine = (self.instances or [None])[-1];k_own, k_opp = self.expected_widths()
        return dict(depth=self.depth, k_own=list(self.k_own), k_opp=list(self.k_opp), selection=self.selection, capacity=self.capacity,
            actor_capacity=self.actor_capacity, block_lanes=list(self.lanes), k1_own=self.k1_own, k1_opp=self.k1_opp, digest=self.digest,
            slot_widths=dict(k_own=list(k_own), k_opp=list(k_opp), source='the request (searched ply 2: searched_widths_reference)'),
            engine_widths=self.engine_widths() or None,
            searched_k=None if engine is None or getattr(engine, 'searched_k', None) is None else list(engine.searched_k),
            widths_note='under searched, k_own[0]/k_opp[0] are searched_maps\' k\'s; the engine derives ply 2\'s slot widths' if self.selection == 'searched' else None)


def depth2_pass_pricer(module, executor, model, *, capacity, actor_capacity, k1_own=9, k1_opp=9, block_lanes=2187, digest=False):
    """The full-width depth-2 pass (refusal accounting; `searched` maps): DepthNPricer(depth=2, k=9), read through as_depth2_frame.
    The gate prices it with digest=True (its leaves are re-checked against the oracle); diagnostics need no digests."""
    return module.DepthNPricer(executor, model, capacity=capacity, depth=2, k_own=9, k_opp=9, k1_own=k1_own, k1_opp=k1_opp, block_lanes=block_lanes,
        selection='prior', digest=digest, actor_capacity=actor_capacity)


def describe(extras):
    """searched_maps' extra outputs, summarized (e.g. a per-root flag: its shape and set count)."""
    out = []
    for x in extras:
        a = np.asarray(x)
        out.append(dict(shape=list(a.shape), dtype=str(a.dtype), **({'set': int(np.count_nonzero(a))} if a.dtype.kind in 'biu' else {})))
    return out


def frame_tag(selection, k):
    return f'{selection}-k{k}'


def abs_t1(frame):
    return np.abs(np.asarray(frame.t1_value, np.float64)[np.asarray(frame.t1_status) == 2])


PASS_T1 = dict(pre=2, root_peer_requested=2, root_peer_legal=2, t1_status=4, t1_error=4, t1_own_requested=4, t1_own_legal=4, t1_peer_requested=4,
    t1_peer_legal=4)
PASS_TRUST = ('The full-width depth-2 pass (the searched maps\' input and the depth-2 refusals) is DepthNPricer(depth=2, k=9). It is trusted via '
    '--n2-equivalence (DepthNPricer N=2 equals the Depth2Pricer kernel) and the depth-2 gate (Depth2Pricer equals the depth-2 oracle). In this run '
    'its turn 1 is checked against the checked depth-N frame\'s (discrete exact, floats within the rule), its enumerated leaves against the '
    'oracle\'s full-width set, and every terminal/error leaf plus --depth2-pass-sample uniform leaves per world are re-run by the oracle (status, '
    'error, terminal value and digest exact, head value within the rule). Re-running every leaf would cost a full depth-2 gate, so the rest of '
    'its leaves rest on the two gates above.')


def depth2_pass_check(depth2_frame, frame, data, *, oracle, stack, params, sample, requests, block, context, reference=None):
    """The full-width depth-2 pass against this gate's oracle where it is cheap (PASS_TRUST). Returns (findings report,
    {quantity: _numeric} for (c), counts)."""
    d2ref = importlib.import_module('gpu_engine.depth2_oracle');found = Findings(dict(context, frame='depth2_pass'));floats = {}
    reference = reference or importlib.import_module('gpu_engine.depthn_oracle');R = len(requests);a, b = depth2_frame, frame
    problems = [x for x in g2.frame_geometry(a, R) if not (x.split(':')[0] in ('t1_digest_own', 't1_digest_peer', 't2_digest') and getattr(a, x.split(':')[0], None) is None)]   # digests optional
    if problems:
        found.fail('depth2_pass_geometry', [[0]], g2._objects([problems[:FIRST]]), g2._objects(['depth-2 surface']));return found.report(), floats, {}
    for name, axes in PASS_T1.items():
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name));keys = np.argwhere(np.ones(x.shape[:axes], bool))
        found.record(f'depth2_pass_{name}', keys, (x == y).reshape(len(keys), -1).all(axis=1), x.reshape(len(keys), -1), y.reshape(len(keys), -1))
    t1s, t1e = np.asarray(b.t1_status), np.asarray(b.t1_error);valued = (t1s == 2) | ((t1s == 3) & np.isin(t1e, (5, 6)))
    floats['depth2_pass_t1_value'] = g2._numeric(np.argwhere(valued), np.asarray(a.t1_value)[valued], np.asarray(b.t1_value)[valued], context)
    for seat, scope in (('own', valued), ('peer', t1s == 2)):
        lanes = (scope & np.asarray(getattr(b, f't1_{seat}_requested'), bool))[..., None] & np.asarray(getattr(b, f't1_{seat}_legal'), bool)
        floats[f'depth2_pass_t1_{seat}_logits'] = g2._numeric(np.argwhere(lanes), np.asarray(getattr(a, f't1_{seat}_logits'))[lanes], np.asarray(getattr(b, f't1_{seat}_logits'))[lanes], context)
    lanes = ((np.asarray(b.pre) == 0) & np.asarray(b.root_peer_requested, bool))[..., None] & np.asarray(b.root_peer_legal, bool)
    floats['depth2_pass_root_peer_logits'] = g2._numeric(np.argwhere(lanes), np.asarray(a.root_peer_logits)[lanes], np.asarray(b.root_peer_logits)[lanes], context)
    children = data['children'];expected = np.asarray(d2ref.turn2_leaves(children, k_own=9, k_opp=9) if len(data['cells']) else [], np.int64).reshape(-1, 6)
    status = np.asarray(a.t2_status);got, want = np.flatnonzero(status != 0), np.unique(np.ravel_multi_index(expected.T, status.shape)) if len(expected) else np.zeros(0, np.int64)
    found.compared['depth2_pass_leaf_enumerated'] += len(np.union1d(got, want))
    for label, flat in (('depth2_pass_leaf_missing', np.setdiff1d(want, got)), ('depth2_pass_leaf_extra', np.setdiff1d(got, want))):
        keys = np.stack(np.unravel_index(flat, status.shape), axis=1) if len(flat) else np.zeros((0, 6), np.int64)
        found.fail(label, keys, ['pass'] * len(keys), ['oracle'] * len(keys))
    counts = dict(enumerated=int(len(expected)), rerun=0, forced=0)
    if sample and len(expected):
        st = status[tuple(expected.T)];forced = np.isin(st, (1, 3));chosen = forced.copy();world = expected[:, 0] * 2 + expected[:, 1]
        for key in np.unique(world).tolist():
            q, w = divmod(key, 2);address = requests[q].plans.root.address
            rng = np.random.default_rng([g2.SAMPLE_SEED, block, address.stream, address.frame, address.owner, w, 2])
            rest = np.flatnonzero((world == key) & ~forced);chosen[rest[rng.choice(len(rest), size=min(sample, len(rest)), replace=False)]] = True
        specs = expected[chosen];leaves = oracle.leaves_n(stack, specs, 2, params);at = tuple(specs.T);counts.update(rerun=int(len(specs)), forced=int(forced.sum()))
        for name in ('status', 'error'):
            x = np.asarray(getattr(a, 't2_' + name))[at];found.record(f'depth2_pass_t2_{name}', specs, x.astype(np.int64) == leaves[name], x, leaves[name])
        terminal = leaves['status'] == 1;x = np.asarray(a.t2_value)[at]
        found.record('depth2_pass_t2_value_terminal', specs[terminal], (x == leaves['value'])[terminal], x[terminal], leaves['value'][terminal])
        if getattr(a, 't2_digest', None) is not None:
            viewed = (leaves['status'] == 4) | ((leaves['status'] == 3) & np.isin(leaves['error'], (3, 4)));x = np.asarray(a.t2_digest)[at].astype(np.uint32)
            found.record('depth2_pass_t2_digest', specs[viewed], (x == leaves['digest']).all(axis=-1)[viewed], x[viewed], leaves['digest'][viewed])
        scored = (leaves['status'] == 4) & (np.asarray(a.t2_status)[at] == 4)
        floats['depth2_pass_t2_value'] = g2._numeric(*g2._masked(specs, np.asarray(a.t2_value)[at], leaves['head_value'], scored), context)
    return found.report(), floats, counts


def run_call_n(*, requests, recipe, materializer, params, pricers, depth2_pricer, frame_pricer, oracle, sample, block, context,
               progress=None, decision=None, save=None, evaluator=None, backups=None, pass_sample=8):
    """One request list: the full-width depth-2 pass (refusals; `searched` maps), the depth-N pricer (and its block-invariance
    twins), D3, the oracle at every ply and the checks. save: (directory, prefix) for --save-arrays. Returns (evaluation, summary)."""
    from gpu_engine import device_public_teacher_batch as batch
    reference = importlib.import_module('gpu_engine.depthn_oracle');module = pricers.module
    requests = tuple(requests);context = dict(context or {});timing, costs = {}, {};depth2_frame = None
    if depth2_pricer is not None:
        with stage(progress, 'depth2_full_width_pass', roots=len(requests), **context) as info:
            clock = time.perf_counter();raw = depth2_pricer.price(requests, materializer, recipe, params);depth2_frame = module.as_depth2_frame(raw)
            costs['depth2_pass'] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(raw, 'timing', None)));info.update(costs['depth2_pass'])
    selection_maps, first, extras = None, None, ()
    if pricers.selection == 'searched':
        if depth2_frame is None:raise ValueError('searched selection needs the full-width depth-2 pass')
        with stage(progress, 'searched_maps', **context) as info:
            clock = time.perf_counter();selection_maps, first, extras = pricers.maps(depth2_frame)
            timing['searched_maps_seconds'] = time.perf_counter() - clock;info.update(widths=first, extras=describe(extras))
    plan, instances = pricers.get();pricer, others = instances[-1], instances[:-1]
    if first is not None and (first[0] > plan['k_own'][0] or first[1] > plan['k_opp'][0]):
        raise ValueError(f'searched_maps returned widths {first}, wider than the pricer\'s ply-2 slot widths {(plan["k_own"][0], plan["k_opp"][0])}')
    kwargs = {} if selection_maps is None else dict(selection_maps=selection_maps);other_frames = []
    for i, other in enumerate(others):
        with stage(progress, f'depthn_pricer_invariance_{i}', block_lanes=other.block_lanes, **context) as info:
            clock = time.perf_counter();other_frames.append(other.price(requests, materializer, recipe, params, **kwargs))
            info.update(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(other_frames[-1], 'timing', None)))
    with stage(progress, 'depthn_pricer', roots=len(requests), block_lanes=pricer.block_lanes, **context) as info:
        clock = time.perf_counter();frame = pricer.price(requests, materializer, recipe, params, **kwargs)
        costs['depthn_pricer'] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(frame, 'timing', None)));info.update(costs['depthn_pricer'])
    with stage(progress, 'frame_pricer', **context) as info:
        clock = time.perf_counter();d_results = frame_pricer.price(requests, materializer, recipe, params)
        costs['frame_pricer'] = dict(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(frame_pricer, 'last_timing', None)))
        info.update(costs['frame_pricer'])
    clock = time.perf_counter();before = dict(oracle.seconds)
    with stage(progress, 'oracle', **context) as info:
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe))
        data = oracle_levels(oracle=oracle, stack=stack, requests=requests, recipe=recipe, params=params, frame=frame, plan=plan,
            selection_maps=selection_maps, sample=sample, block=block, reference=reference)
        info.update(nodes={str(p): len(r.get('specs', r.get('cells', ()))) for p, r in data['levels'].items()},
            leaves_enumerated=len(data['enumerated'].get('leaf', ())), leaves_checked=0 if data['leaves'] is None else len(data['leaves']['specs']))
    costs['oracle'] = dict(wall_seconds=time.perf_counter() - clock, seconds={k: oracle.seconds[k] - before.get(k, 0.) for k in oracle.seconds})
    evaluation = evaluate_n(frame, d_results, requests, data, context=context, full_turn1=plan['k1_own'] == plan['k1_opp'] == 9, progress=progress)
    found = Findings(context);want = [dict(k_own=list(plan['k_own']), k_opp=list(plan['k_opp']))] * len(instances)   # each engine's self-report
    got = [dict(k_own=e['k_own'], k_opp=e['k_opp']) for e in pricers.engine_widths()]
    found.record('engine_widths', np.arange(len(got))[:, None], np.asarray([g == x for g, x in zip(got, want)]), g2._objects(got), g2._objects(want))
    evaluation['a'] = g2.merge_findings([evaluation['a'], found.report()])
    if selection_maps is not None and not evaluation['geometry']:
        found = Findings(context);evaluation['searched_maps'] = dict(searched_maps_wellformed(depth2_frame, *selection_maps[2], found), extras=describe(extras))
        check_searched_rule(depth2_frame, *selection_maps[2], pricers.k_own[0], pricers.k_opp[0], found, extras)
        evaluation['searched_maps']['rule_mismatches'] = {k: v for k, v in found.mismatched.items() if k.endswith('_rule') or k.endswith('_widths') or k == 'searched_depth2_refused'}
        evaluation['a'] = g2.merge_findings([evaluation['a'], found.report()])
    if depth2_frame is not None and not evaluation['geometry']:
        with stage(progress, 'check_depth2_pass', **context) as info:
            found, floats, counts = depth2_pass_check(depth2_frame, frame, data, oracle=oracle, stack=stack, params=params, sample=pass_sample,
                requests=requests, block=block, context=context, reference=reference)
            evaluation['a'] = g2.merge_findings([evaluation['a'], found]);evaluation['c'].update(floats);evaluation['depth2_pass'] = counts
            info.update(counts, passed=found['passed'])
    if others and not evaluation['geometry']:
        with stage(progress, 'check_block_invariance', **context) as info:
            evaluation['invariance'] = [g2.block_invariance(frame_arrays(o), frame_arrays(frame), context=context) for o in other_frames]
            info['discrete_equal'] = all(r['passed'] for r in evaluation['invariance'])
    root_legal = [r.root_legal for r in requests]
    with stage(progress, 'refusals', **context):
        refusals = None if evaluation['geometry'] else refusal_report_n(frame, d_results, depth2_frame, root_legal)
    if (sample is None if decision is None else decision) and not evaluation['geometry']:
        with stage(progress, 'check_e_decision', **context) as info:
            evaluation['e'] = decision_check_n(frame, data, requests, context=context, backups=backups)
            info.update({k: evaluation['e'].get(k) for k in ('passed', 'roots_compared', 'argmax_q_differs', 'argmax_target_differs')})
    if not evaluation['geometry']:evaluation['abs_t1'] = abs_t1(frame)
    saved = []
    if save is not None:
        directory, prefix = save;meta = dict(context, plan=plan, sample_leaves=sample, evaluator=evaluator, timing=getattr(frame, 'timing', None))
        path = Path(directory) / f'{prefix}-{frame_tag(plan["selection"], pricers.k_own[0])}.npz';save_frame(path, frame, requests, meta=meta);saved.append(str(path))
        if depth2_frame is not None:
            path = Path(directory) / f'{prefix}-depth2.npz';save_frame(path, depth2_frame, requests, meta=dict(context, full_width_depth2_pass=True,
                evaluator=evaluator, timing=getattr(depth2_frame, 'timing', None)));saved.append(str(path))
    summary = dict(context, roots=len(requests), worlds=data['worlds'], plan=plan, refusals=refusals, timing=timing, costs=costs,
        nodes_checked={str(p): len(r.get('specs', r.get('cells', ()))) for p, r in data['levels'].items()},
        leaves_enumerated=len(data['enumerated'].get('leaf', ())), leaves_checked=0 if data['leaves'] is None else len(data['leaves']['specs']),
        forced_leaves=data['forced_leaves'], plies_work=json.loads(json.dumps((getattr(frame, 'timing', None) or {}).get('plies'), default=str)),
        searched_maps=evaluation.get('searched_maps'), saved_frames=saved, depth2_pass=evaluation.get('depth2_pass'), topk=data.get('topk'),
        searched={k: json.loads(json.dumps((getattr(frame, 'timing', None) or {}).get(k), default=str)) for k in ('searched_k', 'searched_widths')}
            if plan['selection'] == 'searched' else None,
        passed=dict(a=evaluation['a']['passed'], b_discrete=evaluation['b']['passed'], d=evaluation['d']['passed'], oracle=evaluation['oracle']['passed'],
            **({'block_invariance_discrete': all(r['passed'] for r in evaluation['invariance'])} if evaluation.get('invariance') else {}),
            **({'e': evaluation['e']['passed']} if 'e' in evaluation else {})))
    return evaluation, summary


def gate_status_n(result, *, sampled=False):
    """FAIL if any check failed; INCOMPLETE if none ran or one is vacuous (no leaf compared by (a), or (e) compared no root);
    PASS_SAMPLED with sampled leaves ((e) skipped); else PASS."""
    checks = result['checks']
    if any(not c['passed'] for c in checks.values()):return 'FAIL', []
    reasons = [] if checks else ['no check ran']
    if 'a' in checks and not checks['a']['compared'].get('leaf_status'):reasons.append('vacuous: no leaf was compared by (a)')
    if 'e' in checks and not checks['e'].get('roots_compared'):reasons.append('vacuous: (e) compared no root (every root refused or unlabeled)')
    if reasons:return 'INCOMPLETE', reasons
    return ('PASS_SAMPLED' if sampled else 'PASS'), []


def summarize_run(evaluations, lanes):
    """summarize_n plus the block-invariance section labeled by block_lanes."""
    result = summarize_n(evaluations)
    inv = result['checks'].get('block_invariance')
    if inv is not None:inv.pop('child_blocks', None);inv['block_lanes'] = list(lanes)
    return result


# ---------------------------------------------------------------- N=2 equivalence

N2_DISCRETE = dict(pre=2, root_peer_requested=2, root_peer_legal=2, t1_status=4, t1_error=4, t1_own_requested=4, t1_own_legal=4,
    t1_peer_requested=4, t1_peer_legal=4, t2_status=6, t2_error=6)   # field -> cell axes (any further axis is a lane)


def n2_compare(theirs, mine, context=None):
    """The depth-2 kernel's Depth2Frame vs as_depth2_frame(DepthNPricer at N=2): every discrete array exact everywhere;
    terminal values exact; digests where the view is scored (depth 2's rule); world_reports as canonical JSON (gated). Floats
    are compared ONLY on scored cells (t1 values where expanded or codes 5/6, t2 values at status 4, priors on requested
    seats' legal lanes, the root prior on usable worlds' legal lanes); elsewhere both frames hold unspecified padding.
    Returns (findings report, {quantity: _numeric})."""
    a, b = g2.frame_fields(theirs), g2.frame_fields(mine);found = Findings(context);floats = {}
    for name, axes in N2_DISCRETE.items():
        x, y = np.asarray(a[name]), np.asarray(b[name])
        if x.shape != y.shape:found.fail(name + ':shape', [[0]], [list(y.shape)], [list(x.shape)]);continue
        keys = np.argwhere(np.ones(x.shape[:axes], bool));equal = (x == y).reshape(len(keys), -1).all(axis=1)
        found.record(name, keys, equal, y.reshape(len(keys), -1), x.reshape(len(keys), -1))
    t1s, t1e, t2s, t2e = (np.asarray(a[n]) for n in ('t1_status', 't1_error', 't2_status', 't2_error'))
    if t1s.shape != np.asarray(b['t1_status']).shape or t2s.shape != np.asarray(b['t2_status']).shape:return found.report(), floats
    masks = dict(t1_digest_own=(t1s == 2) | ((t1s == 3) & np.isin(t1e, (3, 4, 5, 6))), t1_digest_peer=(t1s == 2) | ((t1s == 3) & np.isin(t1e, (5, 6))),
        t2_digest=(t2s == 4) | ((t2s == 3) & np.isin(t2e, (3, 4))))
    for name, mask in masks.items():
        if a.get(name) is None or b.get(name) is None:found.fail(name + ':missing', [[0]], ['missing'], ['present']);continue
        keys = np.argwhere(mask);x, y = np.asarray(a[name])[mask], np.asarray(b[name])[mask]
        found.record(name, keys, (x == y).all(axis=-1), y, x)
    for name, status in (('t1_value', t1s), ('t2_value', t2s)):
        keys = np.argwhere(status == 1);x, y = np.asarray(a[name])[status == 1], np.asarray(b[name])[status == 1]
        found.record(name + '_terminal', keys, x == y, y, x)
    valued = (t1s == 2) | ((t1s == 3) & np.isin(t1e, (5, 6)))
    floats['t1_value'] = g2._numeric(np.argwhere(valued), np.asarray(b['t1_value'])[valued], np.asarray(a['t1_value'])[valued], context)
    floats['t2_value'] = g2._numeric(np.argwhere(t2s == 4), np.asarray(b['t2_value'])[t2s == 4], np.asarray(a['t2_value'])[t2s == 4], context)
    usable = (np.asarray(a['pre']) == 0) & np.asarray(a['root_peer_requested'], bool)
    lanes = usable[..., None] & np.asarray(a['root_peer_legal'], bool)
    floats['root_peer_logits'] = g2._numeric(np.argwhere(lanes), np.asarray(b['root_peer_logits'])[lanes], np.asarray(a['root_peer_logits'])[lanes], context)
    for seat, scope in (('own', valued), ('peer', t1s == 2)):
        lanes = scope[..., None] & np.asarray(a[f't1_{seat}_requested'], bool)[..., None] & np.asarray(a[f't1_{seat}_legal'], bool)
        floats[f't1_{seat}_logits'] = g2._numeric(np.argwhere(lanes), np.asarray(b[f't1_{seat}_logits'])[lanes], np.asarray(a[f't1_{seat}_logits'])[lanes], context)
    canonical = lambda x: json.dumps(x, sort_keys=True, default=str);ra, rb = tuple(a.get('world_reports') or ()), tuple(b.get('world_reports') or ())
    report = found.report();report['world_reports_equal'] = canonical(ra) == canonical(rb)   # gated by summarize_n2
    if not report['world_reports_equal']:
        root = next((q for q in range(max(len(ra), len(rb))) if q >= len(ra) or q >= len(rb) or canonical(ra[q]) != canonical(rb[q])), None)
        report['world_reports_first_difference'] = dict(context or {}, root=root, depth2=None if root is None or root >= len(ra) else json.loads(canonical(ra[root])),
            depthn=None if root is None or root >= len(rb) else json.loads(canonical(rb[root])))
    return report, floats


def n2_equivalence_call(*, requests, recipe, materializer, params, configs, frame_pricer, oracle, context, progress=None, backups2=None, module=None):
    """One request list under --n2-equivalence: configs [(name, DepthNPricer N=2, Depth2Pricer)], the same requests to both,
    D3 and the oracle's roots and turn-1 children once (the run's depth-1 noise floor and turn-1 heads). Per config:
    n2_compare, (e) on both surfaces (depth2_backups.learner_max_root) and the guard's spreads."""
    from gpu_engine import device_public_teacher_batch as batch
    reference = importlib.import_module('gpu_engine.depthn_oracle');module = module or pricer_module()
    backups2 = backups2 or importlib.import_module('gpu_engine.depth2_backups');requests = tuple(requests);out = {}
    with stage(progress, 'frame_pricer', **context):d_results = frame_pricer.price(requests, materializer, recipe, params)
    with stage(progress, 'oracle_turn1', **context):
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe));roots = oracle.roots(stack, requests, params)
        children = oracle.children(stack, reference.turn1_cells(requests, recipe, roots), params)
    turn1 = dict(roots=roots, children=children, leaves=dict(specs=np.zeros((0, 6), np.int64)))
    for name, depthn, depth2 in configs:
        where = dict(context, config=name)
        with stage(progress, 'depthn_pricer_n2', **where) as info:
            clock = time.perf_counter();raw = depthn.price(requests, materializer, recipe, params);mine = module.as_depth2_frame(raw)
            info.update(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(raw, 'timing', None)))
        with stage(progress, 'depth2_pricer', **where) as info:
            clock = time.perf_counter();theirs = depth2.price(requests, materializer, recipe, params)
            info.update(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(theirs, 'timing', None)))
        found, floats = n2_compare(theirs, mine, where)
        c = g2.check_heads(mine, turn1, d_results)   # depth-N turn 1 against the oracle, and the run's depth-1 floor
        e = decide_roots(theirs, mine, requests, backups2, context=where)
        out[name] = dict(a=found, floats=floats, c=c, e=e, spreads=node_spreads(raw))
    return out


def summarize_n2(per_config):
    """{config: [call results]} -> per config: discrete (exact), floats within the run's rule, (e) under the near-tie rule;
    the rule from the depth-1 floor (D3 vs the oracle's turn-1 heads) and the p10 spread of the N=2 frames."""
    report = {}
    for name, calls in per_config.items():
        quantities = g2.merge_numeric([x['c'] for x in calls]);p10, spread = guard_spread(s for x in calls for s in x['spreads'])
        rule = guard_note(g2.tolerance_rule(quantities.get('depth1_noise_floor'), p10), spread)
        floats = g2.merge_numeric([x['floats'] for x in calls])
        for q, m in floats.items():g2.finalize_numeric(m, g2.bound_for(q, rule), g2.guard_for(q, rule))
        heads = {q: g2.finalize_numeric(m, g2.bound_for(q, rule), None if q in g2.FLOORS else g2.guard_for(q, rule)) for q, m in quantities.items()}
        discrete = g2.merge_findings([x['a'] for x in calls]);e = g2.merge_decisions([x['e'] for x in calls], q_noise_bound=rule['value_bound'])
        e['reference'] = 'depth2_backups.learner_max_root on the Depth2Frame vs on as_depth2_frame(DepthNPricer N=2)'
        reports_equal = all(x['a'].get('world_reports_equal', False) for x in calls)
        passed = discrete['passed'] and all(m['passed'] for m in floats.values()) and e['passed'] and reports_equal
        vacuous = [reason for reason, empty in (('no turn-2 leaf compared', not discrete['compared'].get('t2_status')),
            ('(e) compared no root', not e.get('roots_compared'))) if empty]
        report[name] = dict(status='FAIL' if not passed else 'INCOMPLETE' if vacuous else 'PASS', status_reasons=vacuous, passed=passed and not vacuous,
            discrete=discrete, floats=floats, e=e, tolerance=dict(rule, q_noise_bound=rule['value_bound']),
            turn1_heads_vs_oracle=dict(reported_not_gated=True, quantities=heads), world_reports_equal=reports_equal,
            world_reports_first_difference=next((x['a']['world_reports_first_difference'] for x in calls if x['a'].get('world_reports_first_difference')), None))
    return report


# ---------------------------------------------------------------- depth-3 diagnostics (live frames or saved ones)

def root_decisions(frame, requests, backups):
    """Per root: (argmax completed-Q over legal actions, lower index on ties; top-two gap) or None when refused."""
    out = []
    for q, request in enumerate(requests):
        legal = tuple(map(bool, request.root_legal));reduction = backups.learner_max_root(frame, q, tuple(map(float, request.root_logits)), legal)
        if reduction.improvement is None:out.append(None);continue
        values = reduction.improvement.completed_q;out.append((g2._argmax(values, legal), g2._top_two_gap(values, legal)))
    return out


def change_counts(base, other, context=None):
    """(i) counts: roots labeled at both depths, and those whose completed-Q argmax differs."""
    both = [(q, a, b) for q, (a, b) in enumerate(zip(base, other)) if a is not None and b is not None]
    changed = [dict(context or {}, root=q, depth2=a[0], depth3=b[0], gap_depth2=a[1], gap_depth3=b[1]) for q, a, b in both if a[0] != b[0]]
    return dict(roots=len(base), labeled_both=len(both), changed=len(changed), changed_roots=changed[:FIRST],
        refused_only_at_depth2=sum(a is None and b is not None for a, b in zip(base, other)),
        refused_only_at_depth3=sum(a is not None and b is None for a, b in zip(base, other)), refused_both=sum(a is None and b is None for a, b in zip(base, other)))


def _softmax(logits, actions):
    top = max(float(logits[a]) for a in actions);weights = [math.exp(float(logits[a]) - top) for a in actions];total = math.fsum(weights)
    return [x / total for x in weights]


def turn2_choices(frame, backups):
    """{(q, w, a1, r1): a2}: the depth-3 learner-max turn-2 move of each expanded turn-1 cell: rows are the backed-up ply-2
    node values averaged under the cell's opponent prior renormalized over its enumerated slots (fsum); ties go to the lower
    ACTION (as depth2_backups.learner_max_choice and the root argmax); a cell with an unpriced child is omitted."""
    out = {};k_opp = int(np.asarray(frame.t1_opp_actions).shape[-1]);K = int(np.asarray(frame.t1_own_actions).shape[-1]) * k_opp
    for q, w, a1, r1 in np.argwhere(np.asarray(frame.t1_status) == 2).tolist():
        own, opp = np.asarray(frame.t1_own_actions[q, w, a1, r1]), np.asarray(frame.t1_opp_actions[q, w, a1, r1]);cell = a1 * 9 + r1
        own_slots, opp_slots = np.flatnonzero(own >= 0).tolist(), np.flatnonzero(opp >= 0).tolist()
        if not own_slots or not opp_slots:continue
        pi = _softmax(frame.t1_peer_logits[q, w, a1, r1], [int(opp[x]) for x in opp_slots]);best = None
        for i in own_slots:
            values = [backups.node_value(frame, q, w, 2, cell * K + i * k_opp + x) for x in opp_slots]
            if any(v is None for v in values):best = None;break
            row = math.fsum(p * v for p, v in zip(pi, values))
            if best is None or row > best[0] or (row == best[0] and int(own[i]) < best[1]):best = (row, int(own[i]))
        if best is not None:out[(q, w, a1, r1)] = best[1]
    return out


def kept_set(frame, cell):
    return {int(x) for x in np.asarray(frame.t1_own_actions[cell]) if x >= 0}


def prior_vs_searched(prior, searched, depth2_frame, backups, backups2, context=None):
    """(ii) for `prior` against `searched` at the same k, over expanded turn-1 cells both price at depth 3:
    - set_differs: the learner's kept turn-2 set (the cell's ply-2 own map) differs;
    - prior_set_not_subset_of_searched: prior's kept set is not contained in searched's (searched keeps k + 1, so a superset
      is the expected case; this counts cells where searched drops a move prior keeps);
    - choice_differs: the depth-3 learner-max turn-2 move differs (the primary measure);
    - searched_choice_outside_prior_set: searched's depth-3 learner-max move is not in prior's kept set;
    - depth2_choice_outside_prior_set: full-width depth 2's learner-max move is not in prior's kept set."""
    a, b = turn2_choices(prior, backups), turn2_choices(searched, backups);common = sorted(set(a) & set(b))
    counts = collections.Counter(cells=len(common));examples = []
    for c in common:
        mine, theirs = kept_set(prior, c), kept_set(searched, c)
        counts['set_differs'] += int(mine != theirs);counts['prior_set_not_subset_of_searched'] += int(not mine <= theirs);counts['choice_differs'] += int(a[c] != b[c])
        counts['searched_choice_outside_prior_set'] += int(b[c] not in mine)
        choice = int(backups2.learner_max_choice(depth2_frame, c[0], c[1])[c[2], c[3]])
        counts['depth2_labeled'] += int(choice >= 0);counts['depth2_choice_outside_prior_set'] += int(choice >= 0 and choice not in mine)
        if a[c] != b[c] and len(examples) < FIRST:examples.append(dict(context or {}, cell=list(c), prior_choice=a[c], searched_choice=b[c],
            prior_set=sorted(mine), searched_set=sorted(theirs), depth2_choice=choice))
    return dict(counts, examples=examples)


def pruned_error_counts(depth2_frame, frame, context=None):
    """(iii) roots whose full-width depth-2 errors all sit in turn-2 pairs the depth-3 ply-2 selection does not enumerate
    (a pair is kept iff a2 is in the cell's own map and r2 in its opponent map; a turn-1 error is never pruned: turn 1 is
    full width). Pre/materialization refusals are counted apart."""
    counts = collections.Counter();examples = []
    own_maps, opp_maps = np.asarray(frame.t1_own_actions), np.asarray(frame.t1_opp_actions)
    for q in range(np.asarray(depth2_frame.pre).shape[0]):
        counts['roots'] += 1
        if (np.asarray(depth2_frame.pre[q]) != 0).any():counts['pre_or_materialization'] += 1;continue
        t1 = np.argwhere(np.asarray(depth2_frame.t1_status[q]) == 3);t2 = np.argwhere(np.asarray(depth2_frame.t2_status[q]) == 3)
        if not len(t1) and not len(t2):counts['no_depth2_errors'] += 1;continue
        counts['with_depth2_errors'] += 1
        kept = lambda w, a1, r1, a2, r2: a2 in own_maps[q, w, a1, r1] and r2 in opp_maps[q, w, a1, r1]
        if len(t1) or any(kept(*map(int, x)) for x in t2):counts['errors_kept'] += 1
        else:
            counts['only_pruned_errors'] += 1
            if len(examples) < FIRST:examples.append(dict(context or {}, root=q, turn2_errors=len(t2)))
    return dict(counts, examples=examples)


def diagnostics_call(depth2_frame, frames, requests, *, backups=None, backups2=None, context=None):
    """One request list: frames {(mode, k): depth-3 frame} on the same requests as the full-width depth-2 frame."""
    backups = backups or importlib.import_module('gpu_engine.depthn_backups');backups2 = backups2 or importlib.import_module('gpu_engine.depth2_backups')
    base = root_decisions(depth2_frame, requests, backups2);out = dict(change={}, prior_vs_searched={}, pruned={}, widths={})
    for (mode, k), frame in sorted(frames.items()):
        tag = frame_tag(mode, k);where = dict(context or {}, config=tag)
        out['change'][tag] = change_counts(base, root_decisions(frame, requests, backups), where)
        out['pruned'][tag] = pruned_error_counts(depth2_frame, frame, where)
        out['widths'][tag] = dict(k_own=list(map(int, frame.k_own)), k_opp=list(map(int, frame.k_opp)))
    for k in sorted({k for _, k in frames}):
        if ('prior', k) in frames and ('searched', k) in frames:
            out['prior_vs_searched'][f'k{k}'] = prior_vs_searched(frames[('prior', k)], frames[('searched', k)], depth2_frame, backups, backups2,
                dict(context or {}, k=k))
    return out


def merge_diagnostics(calls):
    """Sum per-call diagnostics and give the shares."""
    share = lambda part, total: None if not total else part / total
    def add(sections):
        total = collections.Counter();examples = []
        for x in sections:
            total.update({k: v for k, v in x.items() if isinstance(v, int)});examples.extend(x.get('examples', x.get('changed_roots', ())))
        return total, examples[:FIRST]
    report = dict(change_rate={}, prior_vs_searched={}, pruned_errors={}, widths={})
    for tag in sorted({t for c in calls for t in c['change']}):
        total, examples = add([c['change'][tag] for c in calls if tag in c['change']])
        report['change_rate'][tag] = dict(total, rate=share(total['changed'], total['labeled_both']), changed_roots=examples)
        total, examples = add([c['pruned'][tag] for c in calls if tag in c['pruned']])
        report['pruned_errors'][tag] = dict(total, share_of_roots=share(total['only_pruned_errors'], total['roots']),
            share_of_roots_with_depth2_errors=share(total['only_pruned_errors'], total['with_depth2_errors']), examples=examples)
        report['widths'][tag] = sorted({json.dumps(c['widths'][tag]) for c in calls if tag in c['widths']})
    for tag in sorted({t for c in calls for t in c['prior_vs_searched']}):
        total, examples = add([c['prior_vs_searched'][tag] for c in calls if tag in c['prior_vs_searched']])
        report['prior_vs_searched'][tag] = dict(total, examples=examples, **{f'{k}_share': share(total[k], total['cells']) for k in
            ('set_differs', 'prior_set_not_subset_of_searched', 'choice_differs', 'searched_choice_outside_prior_set')},
            depth2_choice_outside_prior_set_share=share(total['depth2_choice_outside_prior_set'], total['depth2_labeled']))
    report['definitions'] = dict(change_rate='(i) among roots labeled at both depths: the share whose completed-Q argmax (lower index on ties; '
        'depth2_backups on the full-width depth-2 frame, depthn_backups on the depth-3 frame) differs', prior_vs_searched=prior_vs_searched.__doc__.strip(),
        pruned_errors=pruned_error_counts.__doc__.strip())
    return report


FRAME_FILE = re.compile(r'^(?P<prefix>frames-block(?P<block>\d+)-call(?P<call>\d+))-(?P<tag>depth2|(?P<mode>prior|searched)-k(?P<k>\d+))\.npz$')


def saved_diagnostic_sets(directories):
    """{(block, call): {'depth2': path, (mode, k): path}} over saved frame files (gate --save-arrays or --depth3-diagnostics
    --save-arrays); the first file found for a key wins, later duplicates are listed."""
    sets, duplicates = collections.defaultdict(dict), []
    for directory in directories:
        for path in sorted(Path(directory).glob('frames-block*-call*.npz')):
            m = FRAME_FILE.match(path.name)
            if m is None:continue
            key = (int(m['block']), int(m['call']));item = 'depth2' if m['tag'] == 'depth2' else (m['mode'], int(m['k']))
            if item in sets[key]:duplicates.append(str(path));continue
            sets[key][item] = path
    return dict(sets), duplicates


def offline_diagnostics(directories, *, backups=None, backups2=None, progress=None):
    """Diagnostics from saved frames: each (block, call) with a full-width depth-2 frame and depth-3 frames on the same
    request addresses."""
    sets, duplicates = saved_diagnostic_sets(directories);calls, skipped = [], []
    for key, items in sorted(sets.items()):
        if 'depth2' not in items:skipped.append(dict(block=key[0], call=key[1], reason='no full-width depth-2 frame'));continue
        depth2_frame, requests, _ = load_frame(items['depth2']);addresses = [(r.plans.root.address.stream, r.plans.root.address.frame,
            r.plans.root.address.owner) for r in requests];frames = {}
        for item, path in items.items():
            if item == 'depth2':continue
            frame, others, meta = load_frame(path)
            if int(meta['depth']) != 3:skipped.append(dict(block=key[0], call=key[1], file=str(path), reason=f'depth {meta["depth"]} != 3'));continue
            if [(r.plans.root.address.stream, r.plans.root.address.frame, r.plans.root.address.owner) for r in others] != addresses:
                skipped.append(dict(block=key[0], call=key[1], file=str(path), reason='request addresses differ from the depth-2 frame\'s'));continue
            frames[item] = frame
        if not frames:skipped.append(dict(block=key[0], call=key[1], reason='no depth-3 frame'));continue
        with stage(progress, 'diagnostics_call', block=key[0], call=key[1], configs=[frame_tag(*k) for k in sorted(frames)]):
            calls.append(diagnostics_call(depth2_frame, frames, requests, backups=backups, backups2=backups2, context=dict(block=key[0], call=key[1])))
    return dict(merge_diagnostics(calls), calls=len(calls), skipped=skipped, duplicates=duplicates, directories=[str(d) for d in directories])


# ---------------------------------------------------------------- command line

def per_ply(text, depth, name):
    """'3' or '3,2' -> one width per ply 2..N."""
    values = tuple(int(x) for x in str(text).split(','))
    if len(values) == 1:values = values * (depth - 1)
    if len(values) != depth - 1:raise ValueError(f'{name}: one width or one per ply 2..{depth}')
    if any(not 1 <= v <= 9 for v in values):raise ValueError(f'{name}: widths 1..9')
    return values


def load_roots(args):
    """Requests (saved roots or a bank chunk), the evaluator (head override or the bank snapshot's actor) and --max-roots."""
    loaded = (g2.saved_root_requests if args.saved_roots else g2.bank_chunk_requests)(args)
    head = g2.load_head(args, plan=loaded.plan, labels=loaded.labels, category_sha256=loaded.category_sha256) if args.head_archive else None
    params = (head[0],) if head is not None else loaded.params
    groups, root_values, dropped, arm_checked = g2.bind_roots(loaded.groups, params, loaded.plan, head_config=None if head is None else head[1])
    evaluator = dict(g2.evaluator_header(args, loaded, head), arm_checked=arm_checked, dropped_roots=dropped)
    log('EVALUATOR_RESOLVED', evaluator=evaluator, root_value_scale=g2.percentiles(root_values))
    if args.max_roots is not None:
        kept, left = [], args.max_roots
        for group in groups:
            if left > 0:kept.append(group[:left]);left -= len(kept[-1])
        groups = kept
    return SimpleNamespace(loaded=loaded, head=head, params=params, groups=groups, root_values=root_values, evaluator=evaluator,
        actor_capacity=len(params) if head is not None else loaded.plan.historical_window + 1)


def each_call(provider, groups, seeds):
    """(block, call, recipe, requests) over the K world blocks (U1's seeds; blocks after 0 bypass Recipe.selected, as U1)."""
    from pokezero import v5_gumbel as original
    from gpu_engine import device_public_worlds as worlds
    for block, seed in enumerate(seeds):
        recipe = provider.recipe if block == 0 else original.Recipe(seed)
        with patch.object(original.Recipe, 'selected', lambda self, address: True) if block else nullcontext():
            for call, group in enumerate(groups):
                yield block, call, recipe, group if block == 0 else [replace(r, plans=worlds.plan_starts(r.plans.root, provider.source, recipe)) for r in group]


def write(path, report):
    path.write_text(json.dumps(report, indent=1, allow_nan=False, default=str) + '\n')


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--saved-roots', help='saved D3 cost bank attempt directory (as d3_depth2_gate)')
    parser.add_argument('--index-sha256', help='sha256 of bank/cost-root-index.json (with --saved-roots)')
    parser.add_argument('--d3-root', help='the D3 bank run root (prespec.json, runtime.json, run/)')
    parser.add_argument('--chunk', type=int, help='bank chunk index (with --d3-root); chunks 0..N are collected')
    parser.add_argument('--memory-policy', action='store_true', help="with --d3-root: apply the bank runtime's JAX memory policy")
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--depth', type=int, default=3, help='search depth N >= 2 (default 3)')
    parser.add_argument('--selection', choices=MODES, default='prior', help='ply-2 selection (plies 3..N always prior)')
    parser.add_argument('--k-own', default='3', help='learner width per ply 2..N, "k" or "k2,k3,..." (under searched, ply 2\'s is searched_maps\' k)')
    parser.add_argument('--k-opp', default='3', help='opponent width per ply 2..N (as --k-own)')
    parser.add_argument('--k1-own', type=int, default=9, help='turn-1 learner width (9 = full)')
    parser.add_argument('--k1-opp', type=int, default=9, help='turn-1 opponent width (9 = full)')
    parser.add_argument('--capacity', type=int, default=512, help='DepthNPricer root capacity (calls are split at the smallest pricer capacity)')
    parser.add_argument('--block-lanes', type=int, default=2187, help='DepthNPricer block_lanes')
    parser.add_argument('--block-invariance', action='store_true', help=f'price every call with block_lanes {INVARIANCE_LANES} (the last is checked)')
    parser.add_argument('--worlds-blocks', type=int, default=1, help='K recipe blocks, W = 2K worlds per root')
    parser.add_argument('--max-roots', type=int, help='price only the first N requests')
    parser.add_argument('--sample-leaves', type=int, help='check every node and N sampled leaves per world (plus every pricer terminal/error leaf)')
    parser.add_argument('--oracle-chunk', type=int, default=512, help='oracle specs per vmap invocation')
    parser.add_argument('--depth2-pass-sample', type=int, default=8, help='uniform full-width depth-2 pass leaves per world the oracle re-runs '
        '(plus every terminal/error leaf); 0 checks only its turn 1 and enumeration')
    parser.add_argument('--save-arrays', action='store_true', help='write each call\'s depth-N and full-width depth-2 frames (for --diagnostics-from)')
    parser.add_argument('--raw-head-check', action='store_true', help='with --head-archive: gate node and leaf values against the head\'s own arm')
    parser.add_argument('--n2-equivalence', action='store_true', help='DepthNPricer N=2 (full width and top --n2-k) vs Depth2Pricer on the same requests')
    parser.add_argument('--n2-k', type=int, default=3, help='the top-k config of --n2-equivalence (k_own = k_opp = k)')
    parser.add_argument('--depth3-diagnostics', action='store_true', help='price full-width depth 2 and depth 3 per mode and --diagnostic-ks; no oracle')
    parser.add_argument('--diagnostic-ks', default=','.join(map(str, DIAGNOSTIC_KS)), help='k values of --depth3-diagnostics')
    parser.add_argument('--diagnostics-from', nargs='+', metavar='DIR', help='offline: depth-3 diagnostics from saved frames (one or more directories)')
    g2.add_head_arguments(parser)
    args = parser.parse_args(argv)
    g2.check_head_arguments(parser, args)
    if args.raw_head_check and not args.head_archive:parser.error('--raw-head-check needs --head-archive')
    if args.diagnostics_from:
        if args.saved_roots or args.d3_root or args.head_archive or args.n2_equivalence or args.depth3_diagnostics:
            parser.error('--diagnostics-from runs offline: no roots, head, --n2-equivalence or --depth3-diagnostics')
        return parser, args
    if sum((args.n2_equivalence, args.depth3_diagnostics)) > 1:parser.error('one of --n2-equivalence or --depth3-diagnostics')
    if bool(args.saved_roots) == bool(args.d3_root):parser.error('exactly one of --saved-roots or --d3-root required')
    if args.saved_roots and not args.index_sha256:parser.error('--saved-roots requires --index-sha256')
    if args.d3_root and (args.chunk is None or args.chunk < 0):parser.error('--d3-root requires --chunk N >= 0')
    if args.depth < 2:parser.error('--depth >= 2 required')
    try:args.k_own_ply, args.k_opp_ply = per_ply(args.k_own, args.depth, '--k-own'), per_ply(args.k_opp, args.depth, '--k-opp')
    except ValueError as error:parser.error(str(error))
    try:args.diagnostic_k_values = tuple(int(x) for x in args.diagnostic_ks.split(','))
    except ValueError:parser.error('--diagnostic-ks: comma-separated integers')
    if any(not 1 <= v <= 9 for v in (args.k1_own, args.k1_opp, args.n2_k) + args.diagnostic_k_values):parser.error('widths must be 1..9')
    if args.capacity < 1 or args.block_lanes < 1 or args.worlds_blocks < 1 or args.oracle_chunk < 1:parser.error('positive sizes required')
    if args.max_roots is not None and args.max_roots < 1:parser.error('positive --max-roots required')
    if args.sample_leaves is not None and args.sample_leaves < 1:parser.error('--sample-leaves N >= 1 (omit it to check every leaf)')
    if args.depth2_pass_sample < 0:parser.error('--depth2-pass-sample N >= 0')
    if args.block_invariance and (args.n2_equivalence or args.depth3_diagnostics):parser.error('--block-invariance belongs to the gate')
    if args.raw_head_check and args.depth3_diagnostics:parser.error('--raw-head-check needs the oracle')
    if args.selection == 'searched' and args.depth < 3:parser.error('searched selection needs --depth >= 3 (it selects ply 2 from a depth-2 pass)')
    return parser, args


def main(argv=None):
    parser, args = parse(argv)
    if args.diagnostics_from:
        out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
        report = dict(schema=SCHEMA, mode='DIAGNOSTICS_OFFLINE', **offline_diagnostics(args.diagnostics_from, progress=log))
        write(out / 'diagnostics.json', report)
        log('DIAGNOSTICS_COMPLETE', out=str(out / 'diagnostics.json'), calls=report['calls'], change_rate={k: v['rate'] for k, v in report['change_rate'].items()})
        return 0 if report['calls'] else 1
    import importlib.util   # fail before loading roots; find_spec does not import JAX
    needed = ['gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups', 'gpu_engine.depth2_backups'] + (['gpu_engine.device_depth2_pricer'] if args.n2_equivalence else [])
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:raise SystemExit('not on this tree (failing closed): ' + ', '.join(missing))
    log('TOLERANCE_RULE', rule=g2.TOLERANCE_RULE, logit_bound=g2.LOGIT_BOUND, guard_factor=g2.GUARD_FACTOR, guard_spread='p10 learner-max own-slot spread of the last ply\'s parents')
    log('EVALUATOR', override=bool(args.head_archive), archive=args.head_archive, archive_sha256=args.head_archive_sha256,
        checkpoint_sha256=args.head_checkpoint_sha256, value_reference=args.value_reference, resolved='EVALUATOR_RESOLVED follows once the roots are loaded')
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    roots = load_roots(args);provider, plan = roots.loaded.provider, roots.loaded.plan;module = pricer_module()
    reference = importlib.import_module('gpu_engine.depthn_oracle');frame_pricer = roots.loaded.frame_pricer
    seeds = g2.world_block_seeds(provider.recipe.seed, args.worlds_blocks)
    common = dict(roots=roots, provider=provider, plan=plan, module=module, frame_pricer=frame_pricer, seeds=seeds, out=out, reference=reference)
    if args.n2_equivalence:return run_n2(args, **common)
    if args.depth3_diagnostics:return run_diagnostics(args, **common)
    return run_gate(args, **common)


def header(args, roots, abs_values, source):
    return dict(evaluator=roots.evaluator, value_scale=g2.value_scale(roots.root_values, g2.percentiles(abs_values), args.value_reference), source=source)


def run_gate(args, *, roots, provider, plan, module, frame_pricer, seeds, out, reference):
    lanes = INVARIANCE_LANES if args.block_invariance else (args.block_lanes,)
    pricers = DepthNPricers(module, provider.executor, plan.model, depth=args.depth, k_own=args.k_own_ply, k_opp=args.k_opp_ply, selection=args.selection,
        capacity=args.capacity, actor_capacity=roots.actor_capacity, lanes=lanes, k1_own=args.k1_own, k1_opp=args.k1_opp)
    depth2_pricer = depth2_pass_pricer(module, provider.executor, plan.model, capacity=args.capacity, actor_capacity=roots.actor_capacity,
        k1_own=args.k1_own, k1_opp=args.k1_opp, block_lanes=args.block_lanes, digest=True)
    capacity = g2.call_capacity(frame_pricer, [depth2_pricer]);groups = g2.split_groups(roots.groups, capacity)
    oracle = reference.DepthNOracle(provider.executor, plan.model, chunk=args.oracle_chunk, progress=g2.oracle_chunk_progress(log),
        raw_model=roots.head[1] if args.raw_head_check else None)
    log('GATE_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), pricer_options=pricers.options(), sample_leaves=args.sample_leaves, seeds=seeds)
    evaluations, calls = [], []
    for block, call, recipe, requests in each_call(provider, groups, seeds):
        evaluation, summary = run_call_n(requests=requests, recipe=recipe, materializer=provider.materializer, params=roots.params, pricers=pricers,
            depth2_pricer=depth2_pricer, frame_pricer=frame_pricer, oracle=oracle, sample=args.sample_leaves, block=block,
            context=dict(block=block, call=call), progress=log, save=(out, f'frames-block{block}-call{call}') if args.save_arrays else None,
            evaluator=roots.evaluator, pass_sample=args.depth2_pass_sample)
        evaluations.append(evaluation);calls.append(summary);log('CALL_CHECKED', **summary)
    result = summarize_run(evaluations, lanes);log('TOLERANCE', **g2.tolerance_line(result['tolerance']))
    status, reasons = gate_status_n(result, sampled=args.sample_leaves is not None)
    reported = [c for c in calls if c['refusals'] is not None]
    refusals = dict(g2.with_dropped(merge_refusals_n([c['refusals'] for c in reported]), roots.evaluator.get('dropped_roots', ())), calls_reported=len(reported),
        rule='reference (D) at every ply: any world pre != 0 or any status-3 node or leaf refuses the root; depth2_full_width is the full-width depth-2 pass')
    abs_values = np.concatenate([e['abs_t1'] for e in evaluations if 'abs_t1' in e]) if any('abs_t1' in e for e in evaluations) else []
    report = dict(header(args, roots, abs_values, roots.loaded.source), tolerance=result['tolerance'], schema=SCHEMA, status=status,
        passed=status in ('PASS', 'PASS_SAMPLED'), status_reasons=reasons, checks=result['checks'],
        decision=result['checks'].get('e', dict(status='SKIPPED_SAMPLED', reason='check (e) needs every leaf checked (no --sample-leaves)')),
        mode='SAVED_ROOTS' if args.saved_roots else 'BANK_CHUNK', pricer_options=pricers.options(), block_invariance=args.block_invariance,
        raw_head_check=args.raw_head_check, split_capacity=capacity, worlds_blocks=args.worlds_blocks, block_seeds=list(seeds), max_roots=args.max_roots,
        sample_leaves=args.sample_leaves, sample_seed=g2.SAMPLE_SEED,
        leaves=dict(enumerated=sum(c['leaves_enumerated'] for c in calls), checked=sum(c['leaves_checked'] for c in calls),
            forced_terminal_or_error=sum(c['forced_leaves'] for c in calls)),
        searched_maps=[dict(block=c['block'], call=c['call'], **c['searched_maps']) for c in calls if c.get('searched_maps')] or None,
        searched_rule=None if args.selection != 'searched' else ('the supplied maps are gated EXACT in (a) (searched_maps_*_rule, searched_maps_widths, '
            'searched_depth2_refused) against this gate\'s own re-derivation (searched_maps_reference) from the full-width depth-2 pass; see depth2_pass'),
        depth2_pass=dict(trust=PASS_TRUST, sample_per_world=args.depth2_pass_sample, **{k: sum((c.get('depth2_pass') or {}).get(k, 0) for c in calls)
            for k in ('enumerated', 'rerun', 'forced')}),
        topk_boundary=merge_topk_n([c.get('topk') for c in calls]),
        refusals=refusals, calls=calls, oracle_cost=dict(seconds=oracle.seconds, rows=oracle.rows, chunk=args.oracle_chunk))
    write(out / 'gate.json', report)
    log('DEPTHN_GATE_COMPLETE', status=status, out=str(out / 'gate.json'), status_reasons=reasons, checks={k: v['passed'] for k, v in result['checks'].items()},
        refused_share={k: refusals.get(k, {}).get('refused_share') for k in ('depthn', 'depth2_full_width', 'depth1')})
    return 0 if report['passed'] else 1


def run_n2(args, *, roots, provider, plan, module, frame_pricer, seeds, out, reference):
    configs, options = [], {}
    for name, k in (('full_width', 9), (f'top{args.n2_k}', args.n2_k)):
        depthn = module.DepthNPricer(provider.executor, plan.model, capacity=args.capacity, depth=2, k_own=k, k_opp=k, k1_own=args.k1_own, k1_opp=args.k1_opp,
            block_lanes=args.block_lanes, selection='prior', digest=True, actor_capacity=roots.actor_capacity)
        depth2, depth2_options = g2.make_pricer(provider.executor, plan.model, widths=dict(k_own=k, k_opp=k, k1_own=args.k1_own, k1_opp=args.k1_opp),
            actor_capacity=roots.actor_capacity, capacity=args.capacity)
        configs.append((name, depthn, depth2));options[name] = dict(k=k, depth2_options=depth2_options)
    capacity = g2.call_capacity(frame_pricer, [c[1] for c in configs] + [c[2] for c in configs]);groups = g2.split_groups(roots.groups, capacity)
    oracle = reference.DepthNOracle(provider.executor, plan.model, chunk=args.oracle_chunk, progress=g2.oracle_chunk_progress(log))
    log('N2_EQUIVALENCE_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), configs=options)
    per_config = collections.defaultdict(list);abs_values = []
    for block, call, recipe, requests in each_call(provider, groups, seeds):
        results = n2_equivalence_call(requests=requests, recipe=recipe, materializer=provider.materializer, params=roots.params, configs=configs,
            frame_pricer=frame_pricer, oracle=oracle, context=dict(block=block, call=call), progress=log, module=module)
        for name, result in results.items():per_config[name].append(result)
        log('N2_CALL', block=block, call=call, discrete={n: r['a']['passed'] for n, r in results.items()}, e={n: r['e'].get('argmax_q_differs') for n, r in results.items()})
    report = summarize_n2(per_config);statuses = [r['status'] for r in report.values()]
    status = 'FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if 'INCOMPLETE' in statuses or not report else 'PASS'
    write(out / 'n2-equivalence.json', dict(header(args, roots, abs_values, roots.loaded.source), schema=SCHEMA, mode='N2_EQUIVALENCE',
        status=status, passed=status == 'PASS', configs=report, options=options, split_capacity=capacity, block_seeds=list(seeds),
        rule='per config: every discrete array exact and world_reports equal; floats only on scored cells (see n2_compare): values within the value '
            'bound and guard, priors within the logit bound (the run\'s evaluator-relative rule); (e) depth2_backups argmax per root equal except near '
            'ties; INCOMPLETE when no turn-2 leaf or no (e) root was compared'))
    log('N2_EQUIVALENCE_COMPLETE', status=status, out=str(out / 'n2-equivalence.json'), configs={n: r['status'] for n, r in report.items()})
    return 0 if status == 'PASS' else 1


def run_diagnostics(args, *, roots, provider, plan, module, frame_pricer, seeds, out, reference):
    depth2_pricer = depth2_pass_pricer(module, provider.executor, plan.model, capacity=args.capacity, actor_capacity=roots.actor_capacity,
        k1_own=args.k1_own, k1_opp=args.k1_opp, block_lanes=args.block_lanes)
    sets = {(mode, k): DepthNPricers(module, provider.executor, plan.model, depth=3, k_own=k, k_opp=k, selection=mode, capacity=args.capacity,
        actor_capacity=roots.actor_capacity, lanes=(args.block_lanes,), k1_own=args.k1_own, k1_opp=args.k1_opp, digest=False)
        for mode in MODES for k in args.diagnostic_k_values}
    groups = g2.split_groups(roots.groups, g2.call_capacity(frame_pricer, [depth2_pricer]))
    log('DIAGNOSTICS_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), configs=[frame_tag(*k) for k in sets])
    calls, abs_values = [], []
    for block, call, recipe, requests in each_call(provider, groups, seeds):
        context = dict(block=block, call=call);requests = tuple(requests)
        with stage(log, 'depth2_full_width_pass', **context):depth2_frame = module.as_depth2_frame(depth2_pricer.price(requests, provider.materializer, recipe, roots.params))
        frames = {}
        for key, pricers in sets.items():
            with stage(log, 'depth3', config=frame_tag(*key), **context) as info:
                maps, first, extras = pricers.maps(depth2_frame);_, (pricer,) = pricers.get()
                frames[key] = pricer.price(requests, provider.materializer, recipe, roots.params, **({} if maps is None else dict(selection_maps=maps)))
                info.update(widths=first, extras=describe(extras), **g2.pricer_costs(frames[key].timing))
            if args.save_arrays:save_frame(out / f'frames-block{block}-call{call}-{frame_tag(*key)}.npz', frames[key], requests, meta=dict(context, evaluator=roots.evaluator))
        if args.save_arrays:save_frame(out / f'frames-block{block}-call{call}-depth2.npz', depth2_frame, requests, meta=dict(context, full_width_depth2_pass=True))
        abs_values.append(abs_t1(depth2_frame))
        with stage(log, 'diagnostics_call', **context):calls.append(diagnostics_call(depth2_frame, frames, requests, context=context))
    report = dict(header(args, roots, np.concatenate(abs_values) if abs_values else [], roots.loaded.source), schema=SCHEMA, mode='DIAGNOSTICS_LIVE',
        **merge_diagnostics(calls), calls=len(calls), block_seeds=list(seeds), ks=list(args.diagnostic_k_values), saved_arrays=args.save_arrays,
        pricer_options={frame_tag(*k): p.options() for k, p in sets.items()})
    write(out / 'diagnostics.json', report)
    log('DIAGNOSTICS_COMPLETE', out=str(out / 'diagnostics.json'), calls=len(calls), change_rate={k: v['rate'] for k, v in report['change_rate'].items()})
    return 0 if calls else 1


if __name__ == '__main__':raise SystemExit(main())
