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
     oracle's own heads: values within 2 x the run's depth-1 noise floor (gated everywhere), logits within 0.1 at turn 1
     and in depth2_pass, and at plies >= 2 of a depth >= 3 frame within DEEPER_LOGIT_RULE's max(0.1, 2 x turn1_logit_floor)
     (the run's turn-1 logit max |diff| against the same oracle, as the value rule's floor), value comparisons
     below 0.25 x the p10 learner-max own-slot spread of the last ply's parents (the misplacement guard). The
     p10 is over NON-ZERO spreads: this departs from the depth-2 gate (p10 over every cell_a2_spread) and applies to
     --n2-equivalence too; a run whose spreads are all zero skips the guard and says so. GUARD_DEPTHN_RULE: at plies >= 2
     of a depth >= 3 frame (ply{p}_value_head, raw_head_ply{p}_value, leaf_value, raw_head_leaf_value) the guard is
     DIGEST-CERTIFIED (reported with guard_mode, beyond_guard and the digest and actor evidence; not gated) when the run's
     values came from ONE evaluator parameter set (a digest certifies the inputs, not which parameters priced them) and (a)
     digest-compared every value-compared position of that ply (leaf_digest / ply{p}_digest_own, checked per position) and
     all matched: the pricer writes a lane's value and digest from the same inputs, so an exact digest certifies placement
     and any misplacement beyond noise still fails the value bound. Otherwise, and for turn-1 and depth2_pass quantities,
     the guard stays gated; --n2-equivalence is unchanged.
 (d) the enumerated set at every ply (turn-1 cells, node ids, leaves) equals the oracle's.
 (e) recursive learner-max (gpu_engine.depthn_backups.learner_max_root) on the pricer frame vs the same frame with the
     oracle's leaf head values and priors on the lanes (c) checks; near ties relative to the run (depth 2's rule).
 (f) the backup's grid and its choices (CHOICE_RULE): at every turn-1 cell and ply 2..N-1 node, the grid depthn_backups.node_value
     hands its per-node rule (captured as passed by a recording rule) against this gate's own recursion over the oracle's data
     (OracleTree), both under the pricer's priors: read identity (READ_IDENTITY: on a tagged copy, which child every grid entry
     reads, exact for any rule), every grid entry and every row q within the value bound (the per-child grid gate), and the
     learner's selected own action and per own row the opponent's argmin action EXACT by action except near ties (rule 1 per
     node: reported, not gated).
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
frames-block{B}-call{C}-{depth2|prior-k{k}|searched-k{k}}.npz). --reverdict GATE_JSON (offline: no GPU, oracle or
pricer) re-applies the current verdict rules to a saved gate.json (each (c) quantity's passed from its saved bound/guard
numbers and guard_mode, (c), (f) through choice_verdict, the status through gate_status_n) and writes gate-reverdict.json
beside it, never over an existing one; exit 0 iff PASS/PASS_SAMPLED. Every report carries a `header` (header_block): (e)'s
and (f)'s near-tie counts and flips among near ties, and RULES_CHANGED_AFTER_FAILURE with the run that triggered each.
--mutation-test (MUTATION_RULE): once every call is priced, at the run's value bound, ten mutants (eight misplacements ({backup
path, scatter path} x {sibling, cousin} x {single, systematic}), a turn-1 misread and wrong priors handed to the rule) and a seeded
swap sweep on both paths, on each call's frame and oracle
data (held until then); mutation-test.json in --out; exit non-zero if the clean run fails, any mutant survives or either sweep
has a material survivor. --save-oracle writes each call's oracle data and D3 outputs
beside its frames (frames-block{B}-call{C}-oracle.npz, implies --save-arrays); --mutation-from DIR... re-runs the mutation
test offline from them. All paths are command-line arguments; --out must be new.
--fixed-shapes (FIXED_SHAPES; opt-in, a harness setting, not a rule; refused with --n2-equivalence): every DepthNPricer the run builds
takes fixed_capacity=True and every oracle stack is padded to a fixed row count (2 x the call-split capacity), so a multi-call run
compiles each program once per process; recorded as the report's and the start line's `harness` section. Without it, nothing is built,
stacked, logged or written differently; a report without `harness` ran without it.
--champion-archive/--champion-archive-sha256/--champion-checkpoint-sha256 [--champion-precision bf16|f32] (exclusive with
--head-archive): champion 9375 is the run's ONE evaluator (gpu_engine.champion_evaluator through the forward= seam): every
DepthNPricer, the full-width depth-2 pass, --n2-equivalence's Depth2Pricer, D3's FramePricer (with_forward) and the oracle price
with it, digests hash its 23-token view, and every request takes its root logits and actors 0/0 (bind_evaluator_roots, after its
exact adapter-vs-champion.forward check on the first root); the evaluator header records it. Without the flags nothing changes.
--declared-bounds PATH --declared-bounds-sha256 SHA --declared-bounds-commit COMMIT (DECLARED_RULE; opt-in, champion only: with
--champion-archive, refused with --head-archive, --depth3-diagnostics and the offline modes): a JSON declaration (DECLARED_SCHEMA) committed
under gpu_engine/declared_bounds/ through a reviewed PR before the run, its bytes pinned by SHA, its git blob sha1 recorded with COMMIT
(DECLARED_NOTE: what is verified and what is only recorded for an audit) and its evaluator identity (kind, archive sha256 and precision)
the run's, whose value, turn-1 logit, deeper-ply logit and optional
near-tie Q bounds replace the tolerance NUMBERS of the B1-calibrated rule for that evaluator; every check and rule is unchanged, the measured
floors are still reported beside the declared bounds, and every report's header carries a `declared_bounds` section (the file, its sha256,
the note that it is a pre-registered evaluator-specific bound, not a rule change after a failure). --reverdict re-applies a report's recorded
declaration after re-verifying its sha256 and git blob sha1 (and the commit and sha256 given with it), refuses a report that keeps traces of a
declaration without its sections, and re-judges any other report only if its numbers are the frozen rule's. Without the flags nothing is
built, judged, logged or written differently.
The champion's depth-2 surfaces against the merged depth-2 gate (d3_depth2_gate, which has no champion flags): on the champion path
(--champion-archive) this gate at --depth 2 and --n2-equivalence stand in for it, with two additions found only there, so a run on the
bank snapshot's actors or the head override (B1) is built, judged and written exactly as before:
 - gate --depth 2 (any widths): D2_ORACLE_CONSISTENCY, the merged gate's leaf_child_* oracle consistency on the inherited depth-2 leaf walk,
   plus that walk equal to the depth-N leaf walk (leaf_walk_*), merged into checks.oracle_consistency;
 - --n2-equivalence --block-invariance (refused off the champion path, as before): CHILD_BLOCK_INVARIANCE, the merged Depth2Pricer at
   child_block 27 and 81 judged as the merged gate does, gated per config (configs.*.block_invariance);
 - --n2-equivalence: MERGED_DEPTH2_DECISION, the merged gate's (e) with depth2_backups on the merged Depth2Pricer's frame against the oracle
   heads (the oracle walks every turn-2 leaf it enumerates), gated per config (configs.*.e_merged_depth2).
Every such report's header carries `merged_depth2_parity` (merged_depth2_parity): the merged gate's checks and where they are made, whether
each addition ran, and the remaining differences (MERGED_DEPTH2_DIFFERENCES): the merged gate's reported-only oracle error counts, by-block
refusal breakdown, values section and sibling spread are absent; its reported-only (e) cell_choice_differs is replaced by (f)'s gated choices
(stricter) in gate mode and reported under --n2-equivalence; the guard's p10 is over non-zero spreads, not every spread (looser or equal); the raw-head check does not apply to the champion
(nothing is substituted; the first-root exact check and the digests over its view stand in); gate --block-invariance varies the DepthNPricer's
block_lanes, not a Depth2Pricer child block; (e)'s backups, sampling seeds and offline modes differ; (a), (f) and leaf_walk_* are extra.
"""
import argparse
import collections
from contextlib import nullcontext
import dataclasses
from dataclasses import replace
import datetime
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import time
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import champion_evaluator as ce, d3_depth2_gate as g2

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


def merge_topk_n(reports, *, logit_bound=g2.LOGIT_BOUND):
    """Per ply and seat over calls: pruning selections, their boundary gaps, how many fall below the logit bound (LOGIT_BOUND; a
    declared turn1_logit_bound under --declared-bounds) and how often the oracle's own prior would select differently (a near tie
    the logit tolerance absorbs). None without pruning."""
    merged = {}
    for report in reports:
        for ply, seats in (report or {}).items():
            for seat, entry in seats.items():
                m = merged.setdefault(ply, {}).setdefault(seat, dict(gaps=[], oracle_prior_set_differs=0))
                m['gaps'].extend(entry['gaps']);m['oracle_prior_set_differs'] += entry['oracle_prior_set_differs']
    if not merged:return None
    out = {ply: {seat: dict(selections=len(m['gaps']), gap=g2.percentiles(m['gaps']), gap_below_logit_bound=int(np.count_nonzero(np.asarray(m['gaps']) < logit_bound)),
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


def _leaf_viewed(status, error):
    """(a)'s leaf digest mask: head-valued leaves and the learner view's encoder/head errors (codes 3/4)."""
    return (status == 4) | ((status == 3) & np.isin(error, (3, 4)))


def _node_valued(level, records, at):
    """(c)'s ply-p masks (valued, expanded): pricer and oracle agree on status and error, and the node is expanded or carries
    code 5/6 (the learner's value survives); expanded alone for the opponent's prior."""
    status, error = records['status'], records['error']
    same = (np.asarray(level.status)[at] == status) & (np.asarray(level.error)[at] == error)
    return same & ((status == 2) | np.isin(error, (5, 6))), same & (status == 2)


def _leaf_scored(frame, leaves):
    """(c)'s leaf mask: head-valued in both the oracle and the pricer."""
    return (leaves['status'] == 4) & (np.asarray(frame.leaf_status)[tuple(leaves['index'].T)] == 4)


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
        viewed = _leaf_viewed(status, error);got = np.asarray(frame.leaf_digest)[at].astype(np.uint32)
        findings.record('leaf_digest', keys[viewed], (got == leaves['digest'])[viewed], got[viewed], leaves['digest'][viewed])
    return findings


def check_heads_n(frame, data, d_results=None):
    """(c): turn 1 as depth 2 (with both floors), every ply's node values and both seats' priors, and the leaf values."""
    out = g2.check_heads(frame, dict(roots=data['roots'], children=data['children'], leaves=dict(specs=np.zeros((0, 6), np.int64))), d_results)
    for p in range(2, data['plan']['depth']):
        records = data['levels'][p]
        if not len(records.get('specs', ())):continue
        level = frame.nodes[p - 2];specs = records['specs'];node = np.asarray(data['ids'][p]);at = (specs[:, 0], specs[:, 1], node)
        keys = np.stack(at, axis=1);valued, expanded = _node_valued(level, records, at)
        parts = {f'ply{p}_value_head': g2._masked(keys, np.asarray(level.value)[at], records['head_value'], valued)}
        if 'raw_head_value' in records:parts[f'raw_head_ply{p}_value'] = g2._masked(keys, np.asarray(level.value)[at], records['raw_head_value'], valued)
        for seat, rec, scope in (('own', 'own', valued), ('opp', 'peer', expanded)):
            lanes = scope[:, None] & np.asarray(records[rec + '_requested'], bool)[:, None] & np.asarray(records[rec + '_legal'], bool)
            parts[f'ply{p}_{seat}_logits'] = g2._masked(keys, np.asarray(getattr(level, f'{seat}_logits'))[at], records[rec + '_logits'], lanes)
        out.update({name: g2._numeric(*part) for name, part in parts.items()})
    leaves = data['leaves']
    if leaves is not None and len(leaves['specs']):
        at = tuple(leaves['index'].T);scored = _leaf_scored(frame, leaves)
        out['leaf_value'] = g2._numeric(*g2._masked(leaves['index'], np.asarray(frame.leaf_value)[at], leaves['head_value'], scored))
        if 'raw_head_value' in leaves:
            out['raw_head_leaf_value'] = g2._numeric(*g2._masked(leaves['index'], np.asarray(frame.leaf_value)[at], leaves['raw_head_value'], scored))
    return out


def digest_cover_n(frame, data):
    """{(c) value quantity past turn 1: its value-compared positions OUTSIDE (a)'s digest mask}, over the same records and
    masks both checks use (GUARD_DEPTHN_RULE's per-position subset check). Nodes: valued (_node_valued) vs _viewed_own; the
    subset holds only through the oracle's codes (5/6 come with status 3), so it is counted, not assumed. Leaves: scored
    (_leaf_scored) vs _leaf_viewed; status 4 lies inside the mask by construction (counted all the same)."""
    cover = {}
    for p in range(2, data['plan']['depth']):
        records = data['levels'][p]
        if not len(records.get('specs', ())):continue
        specs = records['specs'];at = (specs[:, 0], specs[:, 1], np.asarray(data['ids'][p]))
        outside = int(np.count_nonzero(_node_valued(frame.nodes[p - 2], records, at)[0] & ~_viewed_own(records['status'], records['error'])))
        cover[f'ply{p}_value_head'] = outside
        if 'raw_head_value' in records:cover[f'raw_head_ply{p}_value'] = outside
    leaves = data['leaves']
    if leaves is not None and len(leaves['specs']):
        outside = int(np.count_nonzero(_leaf_scored(frame, leaves) & ~_leaf_viewed(leaves['status'], leaves['error'])))
        cover['leaf_value'] = outside
        if 'raw_head_value' in leaves:cover['raw_head_leaf_value'] = outside
    return cover


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


# ---------------------------------------------------------------- (f) the backup's grid and its choices

CHOICE_RULE = ('(f) the backup\'s grid, entrywise, and its choices: at every turn-1 cell and every ply 2..N-1 node expanded on either side, the '
    'pricer side is the grid depthn_backups.node_value hands its per-node rule for THAT node, snapshotted as passed by a recording rule before the '
    'rule runs (never re-gathered from the frame) and mapped to actions through the pricer\'s maps there; the oracle side is this gate\'s own '
    'recursion over the oracle\'s statuses, terminal values, leaf head values, links and maps (OracleTree) under the same per-node rule (learner-max '
    'live) and the PRICER\'s priors (gated by (c)\'s logit bounds). Gated per node: (0) READ IDENTITY (exact, rule-independent): on a tagged copy of '
    'the pricer frame the backup\'s grid must name exactly the children the oracle\'s indexing names, at every ply (see READ_IDENTITY); (0\') the '
    'PRIORS (exact): the pi_own and pi_opp handed to the rule equal, by action and bit for bit, the gate\'s own masked softmax of the pricer\'s logits '
    'at the oracle-derived node id over the oracle\'s maps (see PRIOR_CHECK); (1) the '
    'per-child value gate: every grid entry within the value bound of the oracle\'s child at the same (own action, opponent action), noise-safe by '
    'induction (each entry is a head leaf inside (c)\'s bound, an exact terminal under (a), or a backed-up value inside the bound; max, average and '
    'clip are 1-Lipschitz), the numeric check under learner-max; (2) each row\'s opponent-averaged q within the bound (rule 1\'s max |dQ| clause per '
    'node); (3) the learner\'s selected own action (strict max, lower slot on ties) and per own row the opponent\'s argmin action (lower slot on '
    'ties), exact by action except near ties (rule 1 per comparison: the top-two, or lowest-two, gap <= the bound in either frame: reported, not '
    'gated; with shared priors a flip whose gaps exceed the bound in both frames cannot be noise). Read identity, leaf values within (c)\'s bound '
    'and exact terminals together mean every rule\'s grid entries are the right children\'s values under that rule, and the prior check that the '
    'priors it is handed are the pricer\'s checked priors over the oracle\'s maps (the hedge\'s matrix game included): the value gates certify '
    'learner-max\'s backed-up entries numerically, read identity which child every entry is and the prior check what weighs them, for any rule. '
    'Derived quantities (grid entries, row q, gaps) meet the bound up to REL_EPS relative (float rounding only). FAIL also on a node priced '
    '(not None) on one side only, a node not comparable (rows or columns on one side only, a non-finite entry, a grid off its maps) or, unsampled, '
    'a node the oracle could not value; INCOMPLETE when an unsampled run compares nothing. --sample-leaves: only nodes whose every enumerated '
    'descendant leaf was checked (an errored descendant decides None whatever a sampled-out leaf holds; SKIPPED_SAMPLED when none)')
CHOICE_SAVED = ('comparisons', 'differing', 'one_sided', 'problems', 'not_comparable', 'max_abs_delta_row_q', 'max_abs_delta_child_value', 'value_bound',
    'sampled', 'read_identity', 'priors')
MAX_FLIPS_SAVED = 10000   # gate.json keeps at most this many differing comparisons (differing_complete False beyond)
MISSING = type('Missing', (), dict(__repr__=lambda self: 'MISSING', __slots__=()))()   # an oracle value a sampled-out leaf hides
REL_EPS = 1e-12   # derived quantities vs the value bound: absorbs float rounding (fsum of products, clip) only, never a tolerance


def _within(x, bound):
    """x <= bound up to REL_EPS relative (float rounding only): (f)'s grid entries, row q and near-tie gaps."""
    return x <= bound * (1. + REL_EPS)


def _gap(values, lowest=False):
    """The top-two gap (lowest: the lowest-two gap) of a list; None with fewer than two (no tie is possible)."""
    ranked = sorted(values, reverse=not lowest);return None if len(ranked) < 2 else abs(ranked[0] - ranked[1])


def prior_softmax(logits, actions):
    """(f)'s expected prior: the masked softmax of action-indexed logits over `actions` in that order, this gate's OWN code (never imported from
    depthn_backups) in the documented order of operations (max-subtract, exp, fsum, divide), so a correct backup's priors match it bit for bit.
    Intended coupling: the check is exact, so it ties the frozen gate to depthn_backups' softmax order of operations; a mathematically equal
    refactor that rounds differently (e.g. multiplying by 1/total instead of dividing) fails it, and such a failure is a backups change, not a
    pricer defect."""
    top = max(float(logits[a]) for a in actions)
    weights = [math.exp(float(logits[a]) - top) for a in actions]
    total = math.fsum(weights)
    return [x / total for x in weights]


def _live(m):
    """(slots, actions) of a slot -> action map: the enumerated slots (action >= 0) in slot order, as depthn_backups enumerates them."""
    slots = [i for i in range(len(m)) if int(m[i]) >= 0];return slots, [int(m[i]) for i in slots]


def _solve(grid, pi_opp, own_slots, opp_slots, own, opp):
    """One node from its grid as given (rows: enumerated own slots, columns: enumerated opponent slots, slot order; kept as the object passed),
    the opponent prior over those slots and the slots' actions: each row's opponent-averaged q (fsum, as depthn_backups.learner_max), the
    learner's selected row (strict max: the lower slot on ties) with the top-two q gap, and per row the opponent's argmin column (lower slot on
    ties) with the lowest-two child-value gap."""
    q = [math.fsum(p * v for p, v in zip(pi_opp, row)) for row in grid];best = 0
    for i in range(1, len(q)):
        if q[i] > q[best]:best = i
    argmin = []
    for row in grid:
        low = 0
        for x in range(1, len(row)):
            if row[x] < row[low]:low = x
        argmin.append(low)
    finite = all(math.isfinite(v) for row in grid for v in row) and all(math.isfinite(x) for x in q)
    return dict(own=own, opp=opp, own_slots=own_slots, opp_slots=opp_slots, grid=grid, pi_opp=pi_opp, q=q, learner=best, gap=_gap(q),
        argmin=argmin, row_gaps=[_gap(row, lowest=True) for row in grid], finite=finite)


class _Recorder:
    """The per-node rule node_value is called with: it snapshots the grid node_value hands its rule, row by row, and both priors, BEFORE calling the wrapped rule
    (the content as passed: never a re-gather from the frame, and a rule that edits its grid in place can neither false-fail a clean backup nor
    blind the gate), keeps both priors, and returns the wrapped rule's value unchanged (depthn_backups.learner_max by default; any
    rule(grid, pi_own, pi_opp) passes through, e.g. the hedge's matrix_game). node_value solves in post-order (every child before its parent),
    so after node_value(node) the last call holds the node's own grid."""
    def __init__(self, rule):self.rule, self.last = rule, None

    def __call__(self, grid, pi_own, pi_opp):
        self.last = ([list(row) for row in grid], list(pi_own), list(pi_opp));return self.rule(grid, pi_own, pi_opp)


def _status(frame, ply, q, w, node):
    if ply == 1:a1, r1 = divmod(int(node), 9);return int(frame.t1_status[q, w, a1, r1])
    return int(frame.nodes[ply - 2].status[q, w, node])


def pricer_record(frame, backups, q, w, ply, node, *, rule=None):
    """(f)'s pricer side at one interior node: backups.node_value(frame, q, w, ply, node, rule=_Recorder(rule)) on the pricer frame (rule:
    backups.learner_max by default). The record's grid is the recorder's captured grid ALONE, the object node_value passed its rule for THIS
    node, mapped to actions through the pricer's maps at the node exactly as node_value enumerates its slots (_live); nothing is re-read from
    the frame. None when the pricer does not price the node (not expanded, or node_value None: an errored descendant); a `shape` entry when
    the grid does not match the maps' enumerated slots."""
    if _status(frame, ply, q, w, node) != 2:return None
    recorder = _Recorder(rule or backups.learner_max);value = backups.node_value(frame, q, w, ply, node, rule=recorder)
    if value is None or recorder.last is None:return None
    (grid, pi_own, pi_opp), (own_map, opp_map) = recorder.last, maps_at(frame, ply, q, w, node);(own_slots, own), (opp_slots, opp) = _live(own_map), _live(opp_map)
    if len(grid) != len(own) or len(pi_opp) != len(opp) or len(pi_own) != len(own) or any(len(row) != len(opp) for row in grid):
        return dict(shape=f'grid {[len(row) for row in grid]} (priors {len(pi_own)}, {len(pi_opp)}) vs maps {len(own)} x {len(opp)}', value=float(value))
    return dict(_solve(grid, pi_opp, own_slots, opp_slots, own, opp), pi_own=pi_own, value=float(value))   # both priors as handed to the rule


class OracleTree:
    """(f)'s oracle side: this gate's own recursion over the oracle's data alone, never depthn_backups and never the pricer frame's values or
    statuses. Node statuses and terminal values come from the oracle's records (turn-1 cells: data['children'] over data['cells']; plies
    2..N-1: data['levels'] with ids data['ids']), children through data['links'] (parent, own slot, opponent slot), maps from data['maps'],
    leaf values keyed by the oracle-derived leaf index (expand_arrays' (q, w, parent, own slot, opponent slot)): the head value at status
    4, the terminal value at status 1, None at an error, MISSING when sampled out. Priors are the PRICER's (the frame's logits at the
    ORACLE-derived node id, over the oracle's map; (c) gates them): masked softmax over the enumerated slots. A node's value is clip(rule)
    to [-1, 1]: rule None = learner-max written here (fsum rows, strict max), else the given rule(grid, pi_own, pi_opp), the same one the
    pricer side wraps. An errored descendant gives None (it decides the backup whatever a sampled-out leaf holds, as node_value returns
    None on any error); otherwise a sampled-out leaf gives MISSING (the node cannot be compared)."""
    def __init__(self, frame, data, *, rule=None):
        self.frame, self.maps, self.depth, self.rule = frame, data['maps'], int(data['plan']['depth']), rule
        self.status, self.child, self.leaf, self.memo = {}, {}, {}, {};children = data['children']
        for i, (q, w, a1, r1) in enumerate(np.asarray(data['cells'], np.int64).reshape(-1, 4).tolist()):
            self.status[(q, w, 1, a1 * 9 + r1)] = (int(children['status'][i]), float(children['value'][i]))
        for p in range(2, self.depth):
            records = data['levels'][p]
            if not len(records.get('specs', ())):continue
            parent, own_slot, opp_slot = (np.asarray(x).tolist() for x in data['links'][p])
            for k, ((q, w), node) in enumerate(zip(np.asarray(records['specs'])[:, :2].tolist(), np.asarray(data['ids'][p]).tolist())):
                self.child[(q, w, p, parent[k], own_slot[k], opp_slot[k])] = node;self.status[(q, w, p, node)] = (int(records['status'][k]), float(records['value'][k]))
        leaves = data['leaves']
        if leaves is not None and len(leaves['specs']):
            for key, s, value, head in zip(np.asarray(leaves['index']).tolist(), np.asarray(leaves['status']).tolist(), np.asarray(leaves['value']).tolist(),
                    np.asarray(leaves['head_value']).tolist()):
                self.leaf[tuple(key)] = float(value) if s == 1 else float(head) if s == 4 else None

    def solve(self, q, w, ply, node):
        """(value, record): the terminal value, None (not priced), MISSING, or the backed-up value with the node's _solve record (expanded
        nodes only); memoized."""
        key = (q, w, ply, node)
        if key not in self.memo:self.memo[key] = self._solve(q, w, ply, node)
        return self.memo[key]

    def _solve(self, q, w, ply, node):
        status, terminal = self.status.get((q, w, ply, node), (0, None))
        if status == 1:return terminal, None
        if status != 2 or (ply + 1, q, w, node) not in self.maps:return None, None
        own_map, opp_map = self.maps[(ply + 1, q, w, node)];(own_slots, own), (opp_slots, opp) = _live(own_map), _live(opp_map)
        if not own or not opp:return None, None
        grid, missing, errored = [], False, False
        for i in own_slots:
            row = []
            for x in opp_slots:
                if ply + 1 == self.depth:v = self.leaf.get((q, w, node, i, x), MISSING)
                else:child = self.child.get((q, w, ply + 1, node, i, x));v = None if child is None else self.solve(q, w, ply + 1, child)[0]
                missing |= v is MISSING;errored |= v is None;row.append(v)
            grid.append(row)
        if errored:return None, None
        if missing:return MISSING, None
        own_logits, opp_logits = logits_at(self.frame, ply, q, w, node);record = _solve(grid, prior_softmax(opp_logits, opp), own_slots, opp_slots, own, opp)
        record['pi_own'] = prior_softmax(own_logits, own)
        solved = record['q'][record['learner']] if self.rule is None else self.rule([list(row) for row in grid], list(record['pi_own']), list(record['pi_opp']))   # copies
        return min(1., max(-1., solved)), record


def interior_nodes(frame, data):
    """(q, w, ply, node) of every turn-1 cell and ply 2..N-1 node expanded on either side (the pricer's status or the oracle's records), in
    (root, world, ply, node) order."""
    keys = {(q, w, 1, a1 * 9 + r1) for q, w, a1, r1 in np.argwhere(np.asarray(frame.t1_status) == 2).tolist()}
    cells = np.asarray(data['cells'], np.int64).reshape(-1, 4);status = np.asarray(data['children'].get('status', np.zeros(len(cells))))
    keys |= {(q, w, 1, a1 * 9 + r1) for q, w, a1, r1 in cells[status == 2].tolist()}
    for p in range(2, int(data['plan']['depth'])):
        keys |= {(q, w, p, n) for q, w, n in np.argwhere(np.asarray(frame.nodes[p - 2].status) == 2).tolist()}
        records = data['levels'][p]
        if len(records.get('specs', ())):
            expanded = np.asarray(records['status']) == 2
            keys |= {(q, w, p, n) for (q, w), n in zip(np.asarray(records['specs'])[expanded][:, :2].tolist(), np.asarray(data['ids'][p])[expanded].tolist())}
    return sorted(keys)


def compare_records(mine, theirs):
    """(f) at one node priced on both sides: (comparisons [(kind, own action or None, pricer action, oracle action, gap_pricer, gap_oracle)],
    problems [str], max |row q diff|, max |grid entry diff|), matched by ACTION through each side's own map, never by slot. The grid entry
    difference reads the pricer record's grid only (the recorder's capture) against the oracle's child at the same (own action, opponent
    action)."""
    if 'shape' in mine:return [], ['pricer grid does not match its maps: ' + mine['shape']], 0., 0.
    if not (mine['finite'] and theirs['finite']):return [], ['non-finite row q or child value (' + ', '.join(s for s, r in (('pricer', mine), ('oracle', theirs))
        if not r['finite']) + ')'], 0., 0.   # not comparable: no gap or difference to report
    problems = []
    for seat in ('own', 'opp'):
        if sorted(mine[seat]) != sorted(theirs[seat]):problems.append(f'{seat} actions differ: pricer {mine[seat]}, oracle {theirs[seat]}')
    out = [('learner', None, mine['own'][mine['learner']], theirs['own'][theirs['learner']], mine['gap'], theirs['gap'])]
    rows, cols = {a: k for k, a in enumerate(theirs['own'])}, {r: x for x, r in enumerate(theirs['opp'])};dq = dv = 0.
    for k, a in enumerate(mine['own']):
        if a not in rows:continue
        t = rows[a];out.append(('opponent', a, mine['opp'][mine['argmin'][k]], theirs['opp'][theirs['argmin'][t]], mine['row_gaps'][k], theirs['row_gaps'][t]))
        dq = max(dq, abs(mine['q'][k] - theirs['q'][t]))
        for x, r in enumerate(mine['opp']):
            if r in cols:dv = max(dv, abs(mine['grid'][k][x] - theirs['grid'][t][cols[r]]))
    return out, problems, dq, dv


def _near_tie(bound):
    return lambda gap_pricer, gap_oracle: any(g is not None and _within(g, bound) for g in (gap_pricer, gap_oracle))


def node_fails(mine, theirs, bound):
    """(f)'s verdict at one node under `bound` (the sweeps' local re-evaluation): priced on one side only, not comparable, a grid entry or a
    row q beyond the bound, or a non-exempt flip."""
    if mine is None and theirs is None:return False
    if (mine is None) != (theirs is None):return True
    comparisons, problems, dq, dv = compare_records(mine, theirs);tie = _near_tie(bound)
    return bool(problems) or not _within(dv, bound) or not _within(dq, bound) or any(got != want and not tie(gp, go) for _, _, got, want, gp, go in comparisons)


PRIOR_CHECK = ('priors (gated, EXACT): at every interior node priced on both sides, the pi_own and pi_opp depthn_backups.node_value hands its '
    'per-node rule (captured as passed by the recording rule) equal, by action and bit for bit, this gate\'s own masked softmax (prior_softmax, never '
    'imported) of the PRICER\'s logits read at the ORACLE-derived node id over the ORACLE\'s maps; a mismatch, a missing prior or a node priced on one '
    'side only FAILs. With read identity and the per-child value gate, the grid AND the priors any rule is handed are certified')


def _same(x, y):
    """Bitwise-equal floats (NaN equal to NaN: both sides read the same pricer logits)."""
    return x == y or (x != x and y != y)


def compare_priors(mine, theirs):
    """The seats whose prior, by action, is not EXACTLY the expected one: [dict(seat, pricer {action: p} or None (missing), expected)]."""
    out = []
    for seat in ('own', 'opp'):
        got, want = mine.get('pi_' + seat), dict(zip(theirs[seat], theirs['pi_' + seat]))
        got = None if got is None or len(got) != len(mine[seat]) else dict(zip(mine[seat], got))
        if got is None or set(got) != set(want) or not all(_same(got[a], want[a]) for a in want):
            out.append(dict(seat=seat, pricer=None if got is None else {str(a): p for a, p in got.items()}, expected={str(a): p for a, p in want.items()}))
    return out


def choice_check_n(frame, data, requests, *, context=None, backups=None, sampled=False, tree=None, nodes=None, rule=None):
    """(f) for ONE call (CHOICE_RULE): every interior node's pricer_record (through `backups`, depthn_backups by default: fail closed when
    absent; `rule` the per-node rule both sides solve with, learner-max by default) against the OracleTree. Records each node's max |grid
    entry diff| and max |row q diff| (_node_dv, _node_dq; the largest beside their nodes), each comparison's smaller gap (_gaps: the run's
    near-tie count), every differing comparison with both gaps, nodes priced on one side only, nodes not comparable and nodes the oracle
    cannot value (a sampled-out leaf); choice_verdict judges under the run's value bound (summarize_n)."""
    if backups is None:
        try:backups = importlib.import_module('gpu_engine.depthn_backups')
        except ImportError as error:return dict(passed=False, reason='gpu_engine.depthn_backups is not importable (fail closed): ' + str(error), sampled=sampled)
    tree = tree or OracleTree(frame, data, rule=rule);context = context or {};gaps, dvs, dqs, largest = [], [], [], []
    result = dict(nodes=0, nodes_by_ply={}, comparisons=0, learner_comparisons=0, opponent_comparisons=0, not_comparable=0, both_unpriced=0, one_sided=0,
        one_sided_examples=[], problems=0, problem_examples=[], differing=[], max_abs_delta_row_q=0., max_abs_delta_child_value=0., sampled=sampled)
    priors = dict(nodes=0, mismatched_nodes=0, one_sided=0, first_mismatches=[], rule=PRIOR_CHECK)
    for q, w, ply, node in (interior_nodes(frame, data) if nodes is None else nodes):
        value, theirs = tree.solve(q, w, ply, node)
        if value is MISSING:result['not_comparable'] += 1;continue
        mine = pricer_record(frame, backups, q, w, ply, node, rule=rule)
        if mine is None and theirs is None:result['both_unpriced'] += 1;continue
        address = requests[q].plans.root.address;where = dict(context, root=q, world=w, ply=ply, node=node, address=[address.stream, address.frame, address.owner])
        if (mine is None) != (theirs is None):
            result['one_sided'] += 1;priors['one_sided'] += 1
            if len(result['one_sided_examples']) < FIRST:result['one_sided_examples'].append(dict(where, priced_by='oracle' if mine is None else 'pricer',
                pricer_value=None if mine is None else mine['value'], oracle_value=None if value is None else value))
            continue
        bad = [dict(seat='both', pricer=None, expected=None, problem='grid off its maps: priors not comparable')] if 'shape' in mine else compare_priors(mine, theirs)
        priors['nodes'] += 1;priors['mismatched_nodes'] += bool(bad)
        if bad and len(priors['first_mismatches']) < FIRST:priors['first_mismatches'].append(dict(where, seats=bad))
        comparisons, problems, dq, dv = compare_records(mine, theirs)
        result['nodes'] += 1;result['nodes_by_ply'][str(ply)] = result['nodes_by_ply'].get(str(ply), 0) + 1;dvs.append(dv);dqs.append(dq)
        result['max_abs_delta_row_q'] = max(result['max_abs_delta_row_q'], dq);result['max_abs_delta_child_value'] = max(result['max_abs_delta_child_value'], dv)
        if dv > 0:
            largest.append(dict(where, abs_grid_difference=dv, abs_row_q_difference=dq));largest = sorted(largest, key=lambda r: -r['abs_grid_difference'])[:FIRST]
        if problems:
            result['problems'] += 1
            if len(result['problem_examples']) < FIRST:result['problem_examples'].append(dict(where, problems=problems))
        for kind, row, got, want, gap_pricer, gap_oracle in comparisons:
            result['comparisons'] += 1;result[kind + '_comparisons'] += 1;gaps.append(min(g for g in (gap_pricer, gap_oracle, math.inf) if g is not None))
            if got != want:result['differing'].append(dict(where, kind=kind, own_action=row, actions=[got, want], gap_pricer=gap_pricer, gap_oracle=gap_oracle))
    result.update(_gaps=np.asarray(gaps, np.float64), _node_dv=np.asarray(dvs, np.float64), _node_dq=np.asarray(dqs, np.float64), largest_grid_differences=largest,
        read_identity=read_identity_n(frame, data, requests, backups=backups, tree=tree, context=context, nodes=nodes, sampled=sampled),
        priors=dict(priors, passed=priors['mismatched_nodes'] == 0 and priors['one_sided'] == 0))
    return result


def choice_verdict(f, value_bound, *, sampled=None):
    """(f)'s verdict under the run's value bound, in place (the live run and --reverdict alike). Gated: every grid entry and every row q
    within the bound (_within), and no non-exempt flip (rule 1 per comparison: a near tie when its gap in EITHER frame is within the bound:
    flips reported, not gated); FAIL also on a node priced on one side only, a node not comparable or, unsampled, a node the oracle could not
    value; SKIPPED_SAMPLED when a sampled run compared nothing. Counts beyond the bound need the per-call arrays (live) or the bound saved; a
    saved flip list that was truncated keeps its counts under the same bound and fails closed under another."""
    if 'differing' not in f:return dict(f, passed=False, status='NOT_RUN')   # could not run (no depthn_backups): its reason is kept
    bound = float(value_bound);sampled = bool(f.get('sampled')) if sampled is None else sampled;same = f.get('value_bound') == bound;tie = _near_tie(bound)
    gaps, dvs, dqs = (f.pop(k, None) for k in ('_gaps', '_node_dv', '_node_dq'));reasons = []
    for key, values in (('near_ties', gaps), ('nodes_grid_beyond_bound', dvs), ('nodes_row_q_beyond_bound', dqs)):
        if values is not None:f[key] = int(np.count_nonzero([_within(x, bound) == (key == 'near_ties') for x in np.asarray(values, np.float64)]))   # ties within, the others beyond
        elif not same:f[key] = None
    if f.get('differing_complete', True):
        near, hard = [r for r in f['differing'] if tie(r['gap_pricer'], r['gap_oracle'])], [r for r in f['differing'] if not tie(r['gap_pricer'], r['gap_oracle'])]
        f.update(near_tie_flips=len(near), non_exempt_mismatches=len(hard), near_tie_flip_examples=near[:FIRST], first_mismatches=hard[:FIRST])
    elif not same:f.update(near_tie_flips=None, non_exempt_mismatches=None);reasons.append('the saved flip list is truncated and the value bound changed: not re-judged (fail closed)')
    f.update(value_bound=bound, sampled=sampled, rule=CHOICE_RULE, grid_within_bound=_within(f['max_abs_delta_child_value'], bound),
        row_q_within_bound=_within(f['max_abs_delta_row_q'], bound), first_grid_mismatches=[r for r in f.get('largest_grid_differences', ()) if not _within(r['abs_grid_difference'], bound)],
        status='SKIPPED_SAMPLED' if sampled and not (f['comparisons'] or f['one_sided'] or f['problems']) else 'COMPARED')
    identity = f.get('read_identity') if isinstance(f.get('read_identity'), dict) else dict(passed=False, reason='no read identity recorded')
    identity_ok = identity.get('passed') is True and identity.get('mismatched_nodes') == 0 and identity.get('interned_collisions') == 0 and identity.get('injective') is True
    if not identity_ok:reasons.extend(([identity['reason']] if identity.get('reason') else []) + list(identity.get('reasons') or ()) + ([f'read identity: '
        f'{identity.get("mismatched_nodes")} interior node(s) whose grid reads other children than the oracle\'s indexing (exact, any rule)']
        if identity.get('mismatched_nodes') else []) or ['read identity not passed (passed, mismatched_nodes, interned_collisions, injective: '
        f'{[identity.get(k) for k in ("passed", "mismatched_nodes", "interned_collisions", "injective")]})'])
    prior = f.get('priors') if isinstance(f.get('priors'), dict) else dict(passed=False)
    if not (prior.get('passed') is True and prior.get('mismatched_nodes') == 0 and prior.get('one_sided') == 0):
        reasons.append(f'priors: {prior.get("mismatched_nodes")} interior node(s) whose rule was handed pi_own/pi_opp other than the pricer\'s logits over '
            f'the oracle\'s maps (exact)' if prior.get('mismatched_nodes') else f'priors not certified (passed, mismatched_nodes, one_sided: '
            f'{[prior.get(k) for k in ("passed", "mismatched_nodes", "one_sided")]})')
    if not f['grid_within_bound']:reasons.append(f'max |grid entry diff| {f["max_abs_delta_child_value"]:.3g} exceeds the value bound {bound:.3g}: the backup handed its '
        f'rule a child value the oracle does not have there (per-child grid gate)')
    if not f['row_q_within_bound']:reasons.append(f'max |row q diff| {f["max_abs_delta_row_q"]:.3g} exceeds the value bound {bound:.3g}')
    if f.get('non_exempt_mismatches'):reasons.append(f'{f["non_exempt_mismatches"]} choice(s) flip with a gap above the value bound in both frames')
    if f['one_sided']:reasons.append(f'{f["one_sided"]} interior node(s) priced on one side only')
    if f['problems']:reasons.append(f'{f["problems"]} interior node(s) not comparable (rows/columns on one side only, non-finite, or a grid off its maps)')
    if f['not_comparable'] and not sampled:reasons.append(f'{f["not_comparable"]} node(s) the oracle could not value in an unsampled run (fail closed)')
    f.update(passed=not reasons, reasons=reasons)
    return f


def merge_choices(reports, *, value_bound):
    """Merge per-call (f) reports and judge under the run's value bound (choice_verdict); gate.json keeps at most MAX_FLIPS_SAVED differing
    comparisons (differing_complete)."""
    if any('differing' not in r for r in reports):
        return dict(passed=False, status='NOT_RUN', reasons=sorted({r.get('reason') or 'not run' for r in reports}), rule=CHOICE_RULE)
    keys = ('nodes', 'comparisons', 'learner_comparisons', 'opponent_comparisons', 'not_comparable', 'both_unpriced', 'one_sided', 'problems')
    merged = {k: sum(r[k] for r in reports) for k in keys};by_ply = collections.Counter()
    for r in reports:by_ply.update(r['nodes_by_ply'])
    joined = lambda key: np.concatenate([np.asarray(r.get(key, ()), np.float64) for r in reports]) if reports else np.zeros(0)
    merged.update(nodes_by_ply=dict(sorted(by_ply.items())), differing=[x for r in reports for x in r['differing']], sampled=any(r['sampled'] for r in reports),
        one_sided_examples=[x for r in reports for x in r['one_sided_examples']][:FIRST], problem_examples=[x for r in reports for x in r['problem_examples']][:FIRST],
        max_abs_delta_row_q=max(r['max_abs_delta_row_q'] for r in reports), max_abs_delta_child_value=max(r['max_abs_delta_child_value'] for r in reports),
        largest_grid_differences=sorted((x for r in reports for x in r.get('largest_grid_differences', ())), key=lambda x: -x['abs_grid_difference'])[:FIRST],
        _gaps=joined('_gaps'), _node_dv=joined('_node_dv'), _node_dq=joined('_node_dq'), read_identity=merge_read_identity([r.get('read_identity') or {} for r in reports]),
        priors=dict({k: sum((r.get('priors') or {}).get(k, 0) for r in reports) for k in ('nodes', 'mismatched_nodes', 'one_sided')}, rule=PRIOR_CHECK,
            passed=bool(reports) and all((r.get('priors') or {}).get('passed') is True for r in reports),
            first_mismatches=[x for r in reports for x in (r.get('priors') or {}).get('first_mismatches', ())][:FIRST]))
    merged = choice_verdict(merged, value_bound);merged['differing_complete'] = len(merged['differing']) <= MAX_FLIPS_SAVED
    merged['differing'] = merged['differing'][:MAX_FLIPS_SAVED]
    merged['reference'] = ('the grid depthn_backups.node_value hands its per-node rule on the pricer frame (captured by a recording rule) vs OracleTree on '
        'the oracle\'s statuses, terminal values, leaf head values, links and maps; both under the pricer\'s priors')
    return merged

# ---------------------------------------------------------------- (f) read identity: which child every grid entry reads, any rule

TAG_STEP = 2. ** -42   # tags are integer multiples of 2**-42 strictly inside (-1, 1): exact in float64, untouched by node_value's clamp
TAG_CAPACITY = 2 ** 42 - 1   # per range: position tags -1 + (k + 1) * TAG_STEP in (-1, 0), interned grid tags (m + 1) * TAG_STEP in (0, 1)


class TagOverflow(RuntimeError):
    """More positions or distinct grids than a tag range holds exactly (read identity fails closed)."""


def position_tag(k):
    """The k-th position tag, -1 + (k + 1) * 2**-42: exact, distinct, strictly inside (-1, 0) for 0 <= k < TAG_CAPACITY."""
    return -1. + (k + 1) * TAG_STEP


def interned_tag(m):
    """The m-th interned grid tag, (m + 1) * 2**-42: exact, distinct, strictly inside (0, 1) for 0 <= m < TAG_CAPACITY (disjoint from positions)."""
    return (m + 1) * TAG_STEP


class Intern:
    """The interning rule: a grid's content as passed (a tuple of row tuples) -> a unique interned tag, equal content -> equal tag whatever the
    call order; the tag is never computed arithmetically from the grid's values (no sum, no mean), only looked up or issued. A grid's content
    is made of unique position tags and, recursively, unique child tags, so content interning is node-unique by construction (read_identity_n
    asserts it). One table is SHARED by the pricer's backup and the oracle side, so a child's tag means the same subtree on both. Any rule
    signature: rule(grid, pi_own, pi_opp) -> tag; TagOverflow beyond `capacity` distinct grids."""
    def __init__(self, capacity=TAG_CAPACITY):self.table, self.capacity = {}, capacity

    def tag(self, grid):
        key = tuple(tuple(row) for row in grid)
        if key not in self.table:
            if len(self.table) >= self.capacity:raise TagOverflow(f'more than {self.capacity} distinct grids for the interned tag range')
            self.table[key] = interned_tag(len(self.table))
        return self.table[key]

    def __call__(self, grid, pi_own, pi_opp):return self.tag(grid)


def tagged_frame(frame, *, capacity=TAG_CAPACITY):
    """(copy, layout): a host copy of the pricer frame whose leaf_value at every position and ply 2..N-1 node `value` at every position hold
    their position tags (float64; head-valued and terminal leaves and terminal nodes are the values node_value reads; other positions are
    never read as values); statuses, errors, maps and priors untouched. layout: [(name, first position, shape)] to decode a tag. TagOverflow
    when the positions outnumber the range (fail closed)."""
    names = [('leaf', np.asarray(frame.leaf_value).shape)] + [(f'ply{p}', np.asarray(level.value).shape) for p, level in zip(range(2, int(frame.depth)), frame.nodes)]
    total = sum(int(np.prod(shape)) for _, shape in names)
    if total > capacity:raise TagOverflow(f'{total} value positions exceed the {capacity} exact position tags')
    values, layout, offset = {}, [], 0
    for name, shape in names:
        n = int(np.prod(shape));values[name] = (-1. + (np.arange(offset, offset + n, dtype=np.float64) + 1.) * TAG_STEP).reshape(shape);layout.append((name, offset, shape))
        offset += n
    nodes = tuple(_copy(level, value=values[f'ply{p}']) for p, level in zip(range(2, int(frame.depth)), frame.nodes))
    return _copy(frame, leaf_value=values['leaf'], nodes=nodes), layout


def decode_tag(tag, layout, objects=None):
    """A tag as what it names: ['leaf', q, w, node, own slot, opp slot], ['ply{p}', q, w, node], an interned grid as the oracle node whose expected
    grid it is (['node', q, w, ply, node], through IdentityTree.objects) or, when no oracle node has that content, ['grid', m, 'no oracle node'];
    None as is."""
    if tag is None:return None
    if tag > 0:
        named = (objects or {}).get(tag)
        return list(named) if named is not None else ['grid', int(round(tag / TAG_STEP)) - 1] + (['no oracle node'] if objects is not None else [])
    k = int(round((tag + 1.) / TAG_STEP)) - 1
    for name, offset, shape in layout:
        if offset <= k < offset + int(np.prod(shape)):return [name] + [int(i) for i in np.unravel_index(k - offset, shape)]
    return ['unknown', tag]


class IdentityTree:
    """Read identity's oracle side: each node's EXPECTED tag from the oracle's own indexing alone (OracleTree's statuses, links, maps and leaf
    presence): a leaf at the oracle-derived index (q, w, parent, own slot, opponent slot) -> that position's tag; a terminal node at ply >= 2
    -> its position's tag at the oracle's node id; an expanded node -> the interned tag of its expected grid (rows: the oracle's own slots,
    columns: its opponent slots), through the table shared with the pricer side; an errored descendant -> None; a sampled-out leaf (no error)
    -> MISSING."""
    def __init__(self, tree, layout, intern):
        self.tree, self.intern, self.memo, self.objects = tree, intern, {}, {};self.layout = {name: (offset, shape) for name, offset, shape in layout}

    def position(self, name, index):
        """The position's tag (None outside the pricer's arrays: the pricer cannot read it; frame geometry fails too); objects[tag] names it."""
        offset, shape = self.layout[name]
        try:tag = position_tag(offset + int(np.ravel_multi_index(index, shape)))
        except ValueError:return None
        self.objects[tag] = (name,) + tuple(int(i) for i in index);return tag

    def solve(self, q, w, ply, node):
        """(expected tag, expected grid {(own action, opponent action): tag} or None); memoized."""
        key = (q, w, ply, node)
        if key not in self.memo:self.memo[key] = self._solve(q, w, ply, node)
        return self.memo[key]

    def _solve(self, q, w, ply, node):
        t = self.tree;status, _ = t.status.get((q, w, ply, node), (0, None))
        if status == 1:return (self.position(f'ply{ply}', (q, w, node)) if ply >= 2 else None), None
        if status != 2 or (ply + 1, q, w, node) not in t.maps:return None, None
        own_map, opp_map = t.maps[(ply + 1, q, w, node)];(own_slots, own), (opp_slots, opp) = _live(own_map), _live(opp_map)
        if not own or not opp:return None, None
        grid, missing, errored = [], False, False
        for i in own_slots:
            row = []
            for x in opp_slots:
                if ply + 1 == t.depth:
                    v = t.leaf.get((q, w, node, i, x), MISSING);v = v if v is MISSING or v is None else self.position('leaf', (q, w, node, i, x))
                else:child = t.child.get((q, w, ply + 1, node, i, x));v = None if child is None else self.solve(q, w, ply + 1, child)[0]
                missing |= v is MISSING;errored |= v is None;row.append(v)
            grid.append(row)
        if errored:return None, None
        if missing:return MISSING, None
        tag = self.intern.tag(grid);self.objects[tag] = ('node', q, w, ply, node)
        return tag, {(a, r): grid[k][m] for k, a in enumerate(own) for m, r in enumerate(opp)}


def _read_grid(tagged, backups, intern, q, w, ply, node):
    """The pricer side of read identity at one node: backups.node_value on the tagged copy with the interning rule wrapped by _Recorder:
    (captured grid {(own action, opponent action): tag} through the pricer's maps, as node_value enumerates its slots, or 'shape' when the grid
    does not match them; the node's own tag, node_value's return), or None when not priced."""
    if _status(tagged, ply, q, w, node) != 2:return None
    recorder = _Recorder(intern);own_tag = backups.node_value(tagged, q, w, ply, node, rule=recorder)
    if own_tag is None or recorder.last is None:return None
    grid, (own_map, opp_map) = recorder.last[0], maps_at(tagged, ply, q, w, node);own, opp = _live(own_map)[1], _live(opp_map)[1]
    if len(grid) != len(own) or any(len(row) != len(opp) for row in grid):return 'shape', own_tag
    return {(a, r): grid[k][m] for k, a in enumerate(own) for m, r in enumerate(opp)}, own_tag


READ_IDENTITY = ('read identity (gated, exact, rule-independent): a host copy of the pricer frame holds a unique exact tag at every value position '
    '(leaves, ply 2..N-1 nodes; float64 multiples of 2**-42 strictly inside (-1, 0)); the REAL depthn_backups.node_value (the injected backups) runs '
    'on it with an interning rule (a grid\'s content -> a unique tag in (0, 1), one table shared with the oracle side), and each interior node\'s grid '
    'as passed is captured; the oracle side builds the expected grid from its own indexing (the oracle-derived leaf index\'s tag, the terminal\'s '
    'tag, the interned tag of the child\'s expected grid); every interior node\'s captured grid must EQUAL the expected one by (own action, opponent '
    'action), else FAIL (no bound, no near-tie exemption); a node priced on one side only FAILs; more positions or grids than a tag range holds fails '
    'closed. Two guards, gated: INJECTIVITY, the distinct tags in the captured grids equal the distinct leaves, terminals and interior nodes the '
    'oracle\'s indexing names for the same nodes (a cast, a float32 or bf16 path or a clamp that collapses tags fails here); INTERIOR IDENTITY, the '
    'interned tag -> node map over priced interior nodes is one-to-one (no two nodes share a tag: content interning is node-unique by construction, '
    'never arithmetic on values). It certifies WHICH child every grid entry reads, at every ply, whatever the per-node rule')


def read_identity_n(frame, data, requests, *, backups, tree=None, context=None, nodes=None, capacity=TAG_CAPACITY, sampled=False):
    """READ_IDENTITY for ONE call: the tagged pricer frame through `backups` against IdentityTree over `tree` (the oracle's data). Unsampled, a
    node the oracle cannot name (a leaf it did not compute) fails closed, as in (f). Returns dict(passed, reasons, nodes, entries,
    mismatched_nodes, one_sided, not_comparable, both_unpriced, origins, first_mismatches (decoded reads, interned tags named as the oracle's
    nodes; the deepest mismatches, those with no mismatched descendant, first and marked origin), the guards' counts, tags)."""
    out = dict(nodes=0, entries=0, mismatched_nodes=0, one_sided=0, not_comparable=0, both_unpriced=0, first_mismatches=[], rule=READ_IDENTITY)
    captured, named, owners, mismatched = set(), set(), collections.defaultdict(set), []
    try:
        tagged, layout = tagged_frame(frame, capacity=capacity);intern = Intern(capacity);expected = IdentityTree(tree or OracleTree(frame, data), layout, intern)
        for q, w, ply, node in (interior_nodes(frame, data) if nodes is None else nodes):
            want, theirs = expected.solve(q, w, ply, node)
            if want is MISSING:out['not_comparable'] += 1;continue
            read = _read_grid(tagged, backups, intern, q, w, ply, node);mine = None if read is None else read[0]
            if mine is None and theirs is None:out['both_unpriced'] += 1;continue
            if read is not None and theirs is not None:   # the guards' tallies over nodes priced on both sides
                owners[read[1]].add((q, w, ply, node));named.update(expected.objects[t] for t in theirs.values())
                if mine != 'shape':captured.update(mine.values())
            if (mine is None) != (theirs is None):out['one_sided'] += 1
            else:
                out['nodes'] += 1;out['entries'] += len(theirs)
                if mine == theirs:continue
            mismatched.append(((q, w, ply, node), mine, theirs))
        out.update(tags=dict(positions=int(sum(int(np.prod(s)) for _, _, s in layout)), interned=len(intern.table), capacity=capacity))
        below = {a for key, _, _ in mismatched for a in _ancestors(frame, key)[1:]}   # the deepest mismatches (no mismatched descendant) are origins
        def detail(key, mine, theirs):
            q, w, ply, node = key;address = requests[q].plans.root.address
            where = dict(context or {}, root=q, world=w, ply=ply, node=node, address=[address.stream, address.frame, address.owner], origin=key not in below)
            if (mine is None) != (theirs is None):return dict(where, priced_by='oracle' if mine is None else 'pricer')
            return dict(where, entries=[dict(actions=list(k), read=decode_tag(mine.get(k), layout, expected.objects) if mine != 'shape' else 'grid off its maps',
                expected=decode_tag(theirs.get(k), layout, expected.objects)) for k in sorted(set(theirs) | (set(mine) if mine != 'shape' else set()))
                if mine == 'shape' or mine.get(k) != theirs.get(k)][:9])
        ranked = [m for m in mismatched if m[0] not in below] + [m for m in mismatched if m[0] in below]
        out.update(mismatched_nodes=len(mismatched), origins=sum(m[0] not in below for m in mismatched), first_mismatches=[detail(*m) for m in ranked[:FIRST]])
    except TagOverflow as error:return dict(out, passed=False, reason=f'read identity fails closed: {error}')
    shared = {tag: sorted(keys) for tag, keys in owners.items() if len(keys) > 1}
    out.update(distinct_captured_tags=len(captured), distinct_named_reads=len(named), injective=len(captured) == len(named), interned_collisions=len(shared),
        first_collisions=[dict(tag=tag, nodes=[list(k) for k in keys[:4]]) for tag, keys in list(shared.items())[:FIRST]])
    reasons = ([] if out['injective'] else [f'read identity injectivity: {len(captured)} distinct tags captured for {len(named)} distinct leaves, terminals and '
        'interior nodes read (a cast or clamp collapsed tags, or reads repeat)']) + ([] if not shared else [f'read identity: {len(shared)} interned tag(s) '
        'shared by distinct interior nodes (the tag -> node map is not one-to-one)']) + ([f'read identity: {out["not_comparable"]} node(s) the oracle '
        'could not name in an unsampled run (fail closed)'] if out['not_comparable'] and not sampled else [])
    out.update(passed=out['mismatched_nodes'] == 0 and not reasons, reasons=reasons)   # one-sided nodes are counted among the mismatched
    return out


def merge_read_identity(reports):
    """Per-call read identity summed (the guards per call: injective and one-to-one in every call); passed iff every call's passed (a call that
    failed closed keeps its reason)."""
    keys = ('nodes', 'entries', 'mismatched_nodes', 'one_sided', 'not_comparable', 'both_unpriced', 'interned_collisions')
    return dict({k: sum(r.get(k, 0) for r in reports) for k in keys}, passed=bool(reports) and all(r.get('passed') for r in reports), rule=READ_IDENTITY,
        injective=bool(reports) and all(r.get('injective') for r in reports), distinct_captured_tags=[r.get('distinct_captured_tags') for r in reports],
        distinct_named_reads=[r.get('distinct_named_reads') for r in reports], first_collisions=[x for r in reports for x in r.get('first_collisions', ())][:FIRST],
        first_mismatches=sorted((x for r in reports for x in r.get('first_mismatches', ())), key=lambda x: not x.get('origin'))[:FIRST],
        origins=sum(r.get('origins', 0) for r in reports), reasons=sorted({x for r in reports for x in ([r['reason']] if r.get('reason') else []) + list(r.get('reasons') or ())}),
        tags=dict(positions=sum((r.get('tags') or {}).get('positions', 0) for r in reports), interned=sum((r.get('tags') or {}).get('interned', 0) for r in reports)))


def evaluate_n(frame, d_results, requests, data, *, context=None, full_turn1=True, progress=None, backups=None, decision=False, choices=False,
               sampled=False, tree=None):
    """All checks for ONE depth-N pricing call (floats are bounded at the run level by g2.summarize); with decision/choices, (e)/(f)
    through `backups` (backup_checks; run_call_n runs them after the depth-2 pass and refusals)."""
    out = _evaluate_n(frame, d_results, requests, data, context=context, full_turn1=full_turn1, progress=progress)
    return backup_checks(out, frame, data, requests, context=context, backups=backups, decision=decision, choices=choices, sampled=sampled,
        progress=progress, tree=tree)


def backup_checks(evaluation, frame, data, requests, *, context=None, backups=None, decision=True, choices=True, sampled=False, progress=None,
                  tree=None):
    """(e) and (f) on one call's evaluation, in place, both through `backups` (depthn_backups by default; each fails closed without it);
    nothing on a frame-geometry failure."""
    if evaluation.get('geometry'):return evaluation
    context = context or {}
    if decision:
        with stage(progress, 'check_e_decision', **context) as info:
            evaluation['e'] = decision_check_n(frame, data, requests, context=context, backups=backups)
            info.update({k: evaluation['e'].get(k) for k in ('passed', 'roots_compared', 'argmax_q_differs', 'argmax_target_differs')})
    if choices:
        with stage(progress, 'check_f_choices', **context) as info:
            evaluation['f'] = choice_check_n(frame, data, requests, context=context, backups=backups, sampled=sampled, tree=tree)
            info.update({k: evaluation['f'].get(k) for k in ('nodes', 'comparisons', 'one_sided', 'problems', 'not_comparable', 'reason')},
                differing=len(evaluation['f'].get('differing', ())), max_abs_delta_row_q=evaluation['f'].get('max_abs_delta_row_q'),
                max_abs_delta_child_value=evaluation['f'].get('max_abs_delta_child_value'))
    return evaluation


def _evaluate_n(frame, d_results, requests, data, *, context=None, full_turn1=True, progress=None):
    context = context or {};geometry = frame_geometry_n(frame, len(requests), data['plan']);depth = data['plan']['depth']
    if geometry:
        bad = dict(passed=False, compared={}, mismatched=dict(frame_geometry=len(geometry)),
            first_mismatches=[dict(context, field='frame_geometry', problems=geometry[:FIRST])])
        return dict(a=bad, b=dict(bad, values={}), c={}, d=bad, oracle=check_oracle_n(data, Findings(context)).report(), geometry=geometry, depth=depth)
    with stage(progress, 'check_a', **context) as info:a = check_exact_n(frame, data, Findings(context)).report();info['passed'] = a['passed']
    with stage(progress, 'check_b', **context) as info:
        found = g2.check_d(frame, d_results, requests, Findings(context), full_turn1=full_turn1);b = dict(found.report(), values=found.values)
        info['passed'] = b['passed']
    with stage(progress, 'check_c', **context):c = check_heads_n(frame, data, d_results);cover = digest_cover_n(frame, data)
    with stage(progress, 'check_d', **context) as info:d = check_enumeration_n(frame, data, Findings(context)).report();info['passed'] = d['passed']
    with stage(progress, 'check_oracle_consistency', **context):oracle = check_oracle_n(data, Findings(context)).report()
    return dict(a=a, b=b, c=c, d=d, oracle=oracle, geometry=[], spreads=node_spreads(frame), depth=depth, digest_cover=cover)


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


GUARD_DEPTHN_RULE = ('(c) value quantities at plies >= 2 of a depth >= 3 frame (ply{p}_value_head, raw_head_ply{p}_value, leaf_value, '
    'raw_head_leaf_value): the misplacement guard is DIGEST-CERTIFIED (guard_bound, within_guard, ratio_to_guard and beyond_guard reported, '
    'not gated) when the run\'s values came from a single evaluator parameter set (evaluator_actors == 1: the live run counts the parameter sets '
    'it priced with; a saved report establishes it by that record, the head override or actor_capacity 1, else it stays gated), because a digest '
    'certifies the inputs, not which parameters priced them, and (a) digest-compared every value-compared position of that ply and all matched: '
    'the digest (leaf_digest, ply{p}_digest_own) compared >= the quantity\'s compared with zero mismatches, and zero value-compared positions outside '
    '(a)\'s digest mask (counted per position; for leaves inside by construction). The pricer writes each lane\'s value and digest from the same '
    'inputs, so an exact digest certifies placement and a misplacement whose value effect exceeds noise still fails the value bound; a misread in '
    'the backup (the right value in the frame, the wrong child read) is (f)\'s: read identity (exact, any rule) certifies which child every grid '
    'entry the backup hands its per-node rule reads, at every ply, and the per-child value gate holds every learner-max entry within the value '
    'bound of the oracle\'s child; with (c)\'s leaf bound and (a)\'s exact terminals every rule\'s grid entries are the right children\'s values, '
    'and the exact prior check certifies the pi_own and pi_opp every rule is handed (the pricer\'s logits, checked by (c), over the oracle\'s maps); '
    'that replaces the guard there. Otherwise the guard stays gated, as '
    'it does for turn-1 and depth2_pass quantities; the value bound is gated everywhere; --n2-equivalence is unchanged')


def certifying_digest(name, depth):
    """(a)'s digest field that can certify the guard of (c) quantity `name` under GUARD_DEPTHN_RULE (None: always gated)."""
    if depth is None or int(depth) < 3:return None
    if name in ('leaf_value', 'raw_head_leaf_value'):return 'leaf_digest'
    m = re.fullmatch(r'ply(\d+)_value_head|raw_head_ply(\d+)_value', name);p = None if m is None else int(m.group(1) or m.group(2))
    return f'ply{p}_digest_own' if p is not None and 2 <= p < int(depth) else None


def guard_mode(name, m, a, depth, outside, actors):
    """('digest_certified' | 'gated', evidence or None) for one guarded (c) value quantity `m` against (a)'s merged report.
    outside: its value-compared positions outside (a)'s digest mask (digest_cover_n); None when not recorded, which certifies
    leaves (scored lies inside the mask by construction) and leaves a node quantity gated. actors: dict(count, source) of the
    evaluator parameter sets the values came from; only count 1 certifies."""
    digest = certifying_digest(name, depth)
    if digest is None:return 'gated', None
    compared, mismatched, leaf = a.get('compared', {}).get(digest), a.get('mismatched', {}).get(digest), digest == 'leaf_digest'
    actors = actors or dict(count=None, source='not established')
    evidence = dict(digest=digest, digest_compared=compared, digest_mismatched=mismatched, value_compared=m['compared'],
        value_positions_outside_digest=outside, subset='per position (digest_cover_n)' if outside is not None else
            'by construction: scored leaves (oracle status 4) lie inside the leaf digest mask' if leaf else 'unknown: no per-position count recorded',
        evaluator_actors=actors['count'], evaluator_actors_source=actors['source'])
    certified = actors['count'] == 1 and compared is not None and mismatched == 0 and compared >= m['compared'] and (outside == 0 or (outside is None and leaf))
    return ('digest_certified' if certified else 'gated'), evidence


ACTORS_LIVE = 'len(params): the evaluator parameter sets this run priced with, pricer and oracle alike (the head override: the head alone)'


def saved_actors(report):
    """dict(count, source) of the evaluator parameter sets a saved report's values came from: its tolerance.evaluator_actors
    (recorded by the live run since the single-actor condition), else count 1 from evaluator.override (the head prices alone,
    actors 0/0) or pricer_options.actor_capacity == 1; count None (not established: the guard stays gated) otherwise."""
    recorded = (report.get('tolerance') or {}).get('evaluator_actors')
    if isinstance(recorded, dict) and 'count' in recorded:return dict(count=recorded['count'], source=recorded.get('source'))
    found = (['saved evaluator.override (the head override prices with the head alone, actors 0/0)'] if (report.get('evaluator') or {}).get('override') is True else []) \
        + (['saved pricer_options.actor_capacity == 1'] if (report.get('pricer_options') or {}).get('actor_capacity') == 1 else [])
    if found:return dict(count=1, source='; '.join(found))
    return dict(count=None, source='not established: no tolerance.evaluator_actors, evaluator.override not true, pricer_options.actor_capacity '
        f'{(report.get("pricer_options") or {}).get("actor_capacity")!r}')


DEEPER_LOGIT_RULE = ('(c) priors at plies >= 2 of a depth >= 3 frame (ply{p}_own_logits, ply{p}_opp_logits): max |diff| <= deeper_logit_bound = '
    'max(LOGIT_BOUND, 2 x turn1_logit_floor), turn1_logit_floor = this run\'s max |diff| over t1_own_logits and t1_peer_logits (the same merged run, '
    'against the same oracle); LOGIT_BOUND when neither is measured or one is non-finite. It mirrors the value rule (2 x a turn-1 floor measured '
    'against the same oracle): deeper plies\' priors come from other compiled programs than the oracle\'s, as turn 1\'s do, while turn-1 logits stay '
    'anchored at LOGIT_BOUND by the depth-2 gates and the N=2 equivalence (bitwise at turn 1). Turn-1 (t1_*_logits, root_peer_logits) and '
    'depth2_pass_* logits keep LOGIT_BOUND; depth 2, --n2-equivalence, every value rule and block invariance are unchanged')
TURN1_LOGITS = ('t1_own_logits', 't1_peer_logits')


def deeper_logit(name, depth):
    """Whether (c) quantity `name` is a prior at a ply >= 2 of a depth >= 3 frame (DEEPER_LOGIT_RULE's bound)."""
    m = None if depth is None or int(depth) < 3 else re.fullmatch(r'ply(\d+)_(?:own|opp)_logits', name)
    return m is not None and 2 <= int(m.group(1)) < int(depth)


def turn1_logit_floor(quantities):
    """(turn1_logit_floor or None, source): the max |diff| over the measured TURN1_LOGITS quantities of (c); None (the
    LOGIT_BOUND fallback) when neither is measured or a measured one is non-finite."""
    measured = {n: quantities[n] for n in TURN1_LOGITS if (quantities.get(n) or {}).get('compared')}
    if not measured:return None, 'fallback LOGIT_BOUND: no turn-1 logit was compared'
    bad = [n for n, m in measured.items() if m.get('nonfinite')]
    if bad:return None, 'fallback LOGIT_BOUND: non-finite turn-1 logit differences (' + ', '.join(bad) + ')'
    name = max(measured, key=lambda n: measured[n]['max_abs_difference']);return float(measured[name]['max_abs_difference']), f'max |diff| of {name}'


def rebound(m, bound, deltas=None):
    """A finalized quantity re-judged against `bound`, in place (finalize_numeric's rule): within_bound from max_abs_difference and
    nonfinite; beyond_bound exact from `deltas` {|diffs|}, else 0 when within, else counted on the saved first_beyond when that
    list provably holds every difference above `bound`, else None; first_beyond and ratio_to_bound to match."""
    if deltas is None and m['bound'] == bound:return m
    within = m['nonfinite'] == 0 and m['max_abs_difference'] <= bound;saved = m.get('first_beyond') or []
    rows = [r for r in saved if float(r['abs_difference']) > bound]   # float(): a non-finite difference is saved as 'inf'
    if deltas is not None:beyond = int(np.count_nonzero(np.asarray(deltas, np.float64) > bound))
    elif within:beyond = 0
    else:beyond = len(rows) if bound >= m['bound'] and (len(rows) < len(saved) or m.get('beyond_bound') == len(saved)) else None
    m.update(bound=bound, beyond_bound=beyond, within_bound=within, first_beyond=rows, ratio_to_bound=None if m['nonfinite'] else m['max_abs_difference'] / bound)
    return m


def apply_rules_depthn(result, depth, *, actors=None, deltas=None, outside=None, geometry=None, declared=None):
    """The current (c) verdict on a summarized result or a saved gate report, in place (the live run and --reverdict alike):
    DEEPER_LOGIT_RULE's bound on the deeper priors (rebound; `bound_rule` on every logit quantity); each non-floor
    quantity's passed = within_bound and, where it has a guard, (within_guard unless GUARD_DEPTHN_RULE certifies it:
    guard_mode, guard_evidence, beyond_guard from `deltas` {quantity: |diffs|}, else as saved); then (c)'s passed (a
    frame-geometry failure keeps it failed; None reads a.mismatched.frame_geometry), the tolerance table's (c) rows and the
    tolerance section's rules and numbers. actors: dict(count, source) of the evaluator parameter sets (None: not established);
    outside: {quantity: digest_cover_n count}; a quantity without one reuses its saved guard_evidence count. declared: a
    --declared-bounds declaration (DECLARED_RULE): every non-floor quantity is rebound to its declared number (values, turn-1
    logits, deeper logits by declared_deeper) and tolerance.declared_bounds records them beside the measured floors and the
    frozen rule's numbers; None: exactly the frozen rule."""
    checks = result['checks'];c, a = checks.get('c'), checks.get('a') or {};modes = {}
    if c is None:return result
    floor, floor_source = turn1_logit_floor(c['quantities']);deeper = g2.LOGIT_BOUND if floor is None else max(g2.LOGIT_BOUND, 2 * floor)
    source = floor_source if floor is None else (f'2 x turn1_logit_floor ({floor_source})' if 2 * floor > g2.LOGIT_BOUND else f'LOGIT_BOUND (2 x turn1_logit_floor, {floor_source}, is below it)')
    frozen_deeper, numbers = deeper, None if declared is None else declared_numbers(declared)
    if declared is not None:deeper, source = declared_deeper(declared, floor, floor_source)
    for name, m in c['quantities'].items():
        if name in g2.FLOORS:continue
        if declared is not None:
            bound, label = declared_bound_for(name, depth, numbers, deeper);rebound(m, bound, None if deltas is None else deltas.get(name));m['bound_rule'] = label
        elif deeper_logit(name, depth):rebound(m, deeper, None if deltas is None else deltas.get(name));m['bound_rule'] = 'deeper_logit_bound'
        elif g2.is_logit(name):m['bound_rule'] = 'logit_bound'
        if m.get('guard_bound') is None:m['passed'] = bool(m['within_bound']);continue
        count = (outside or {}).get(name, (m.get('guard_evidence') or {}).get('value_positions_outside_digest'))
        mode, evidence = guard_mode(name, m, a, depth, count, actors);modes[name] = mode
        if deltas is not None and name in deltas:m['beyond_guard'] = int(np.count_nonzero(np.asarray(deltas[name], np.float64) >= m['guard_bound']))
        m.update(guard_mode=mode, guard_evidence=evidence, beyond_guard=m.get('beyond_guard'))
        m['passed'] = bool(m['within_bound']) and (mode == 'digest_certified' or bool(m['within_guard']))
    if geometry is None:geometry = bool(a.get('mismatched', {}).get('frame_geometry'))
    c['passed'] = all(m['passed'] for n, m in c['quantities'].items() if n not in g2.FLOORS) and not geometry
    for row in result['tolerance'].get('table', ()):
        if row.get('check') == 'c' and row.get('quantity') in c['quantities']:
            m = c['quantities'][row['quantity']]
            row.update(bound=m['bound'], ratio_to_bound=m['ratio_to_bound'], passed=m['passed'], guard_mode=m.get('guard_mode'), beyond_guard=m.get('beyond_guard'))
    result['tolerance'].update(guard_depthn_rule=GUARD_DEPTHN_RULE, guard_modes=modes, evaluator_actors=actors or dict(count=None, source='not established'),
        turn1_logit_floor=floor, deeper_logit_bound=deeper,
        deeper_logit_bound_source=source, deeper_logit_rule=DEEPER_LOGIT_RULE if declared is None else DECLARED_DEEPER_RULE,
        deeper_logit_quantities=[n for n in c['quantities'] if deeper_logit(n, depth)])
    if declared is not None:result['tolerance']['declared_bounds'] = declared_tolerance(result['tolerance'], declared, c['quantities'], frozen_deeper)
    return result


def digest_cover_merged(evaluations):
    """{quantity: summed digest_cover_n count} over calls; None for a quantity some call compared without a count."""
    out = {}
    for e in evaluations:
        cover = e.get('digest_cover') or {}
        for name in e.get('c') or {}:out[name] = None if name not in cover or out.get(name, 0) is None else out.get(name, 0) + cover[name]
    return out


def summarize_n(evaluations, *, declared=None):
    """g2.summarize with the guard's spread taken from the last ply's learner-max own-slot spreads of the same frames, then
    apply_rules_depthn (DEEPER_LOGIT_RULE, GUARD_DEPTHN_RULE) on (c) at the calls' depth (their plan's; mixed or unknown
    depths keep LOGIT_BOUND and every guard gated), then (f) (merge_choices) after (e) under the same value bound. declared:
    --declared-bounds (DECLARED_RULE: the declared numbers in g2.summarize's rule and apply_rules_depthn); None: the frozen rule."""
    p10, spread = guard_spread(s for e in evaluations for s in e.get('spreads', ()))
    result = g2.summarize(evaluations, spread_p10=p10, **declared_kwargs(declared))
    result['tolerance'] = guard_note(result['tolerance'], spread);depths = {e.get('depth') for e in evaluations};deltas = collections.defaultdict(list)
    for e in evaluations:
        for name, m in (e.get('c') or {}).items():deltas[name].append(np.asarray(m['_deltas'], np.float64))
    counts = {e.get('actors') for e in evaluations};count = next(iter(counts)) if len(counts) == 1 else None
    actors = dict(count=count, source=ACTORS_LIVE) if type(count) is int else dict(count=None, source=f'not established: the calls recorded {sorted(map(str, counts))}')
    apply_rules_depthn(result, depths.pop() if len(depths) == 1 else None, actors=actors, deltas={k: np.concatenate(v) for k, v in deltas.items()},
        outside=digest_cover_merged(evaluations), geometry=any(e.get('geometry') for e in evaluations), declared=declared)
    choices = [e['f'] for e in evaluations if 'f' in e]
    if choices:   # (f) after (e), under the same value bound as (e)'s q_noise_bound
        f = merge_choices(choices, value_bound=result['tolerance']['value_bound']);checks = result['checks']
        result['checks'] = dict([(k, v) for k, v in checks.items() if k != 'block_invariance'] + [('f', f)] + [(k, v) for k, v in checks.items() if k == 'block_invariance'])
        result['tolerance'].update(choice_rule=CHOICE_RULE, f_near_ties=f.get('near_ties'), f_near_tie_flips=f.get('near_tie_flips'),
            f_max_abs_delta_row_q=f.get('max_abs_delta_row_q'), f_max_abs_delta_child_value=f.get('max_abs_delta_child_value'))
    result['passed'] = bool(result['checks']) and all(x['passed'] for x in result['checks'].values())
    return result


def tolerance_line_n(tolerance):
    """g2.tolerance_line with the deeper logit bound and its turn-1 floor, (f)'s near ties, flips among them, max |row q diff| and max |grid diff|,
    and each row's bound, guard_mode (rows with a guard outside (c) are gated) and beyond_guard."""
    line = dict(g2.tolerance_line(tolerance), turn1_logit_floor=tolerance.get('turn1_logit_floor'), deeper_logit_bound=tolerance.get('deeper_logit_bound'),
        **{k: tolerance.get(k) for k in ('f_near_ties', 'f_near_tie_flips', 'f_max_abs_delta_row_q', 'f_max_abs_delta_child_value')},
        **({} if 'declared_bounds' not in tolerance else dict(declared_bounds_sha256=tolerance['declared_bounds'].get('sha256'))))
    for entry, row in zip(line['checks'], tolerance.get('table', ())):
        entry.update(bound=row.get('bound'), guard_mode=row.get('guard_mode', 'gated' if row.get('guard_bound') is not None else None), beyond_guard=row.get('beyond_guard'))
    return line


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
    engine.searched_k the k's); plies past 2 are prior widths in both modes. fixed_capacity (--fixed-shapes): every instance is built
    with fixed_capacity=True and options() says so; False builds and reports exactly as without the option. forward: the run's
    evaluator (--champion-archive), passed to every instance and named in options(); None builds and reports as before."""
    def __init__(self, module, executor, model, *, depth, k_own, k_opp, selection, capacity, actor_capacity, lanes=(2187,), k1_own=9,
                 k1_opp=9, digest=True, fixed_capacity=False, forward=None):
        self.module, self.executor, self.model, self.depth, self.selection = module, executor, model, depth, selection
        shape = plan_of(depth, k_own, k_opp);self.k_own, self.k_opp = shape['k_own'], shape['k_opp']
        self.capacity, self.actor_capacity, self.lanes, self.k1_own, self.k1_opp, self.digest = capacity, actor_capacity, tuple(lanes), k1_own, k1_opp, digest
        self.fixed_capacity = fixed_capacity;self.forward = forward;self.instances = None

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
                actor_capacity=self.actor_capacity, **fixed_kwargs(self.fixed_capacity), **forward_kwargs(self.forward)) for lanes in self.lanes]
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
            widths_note='under searched, k_own[0]/k_opp[0] are searched_maps\' k\'s; the engine derives ply 2\'s slot widths' if self.selection == 'searched' else None,
            **fixed_kwargs(self.fixed_capacity), **({} if self.forward is None else dict(evaluator=ce.describe(self.forward))))


def fixed_kwargs(fixed_capacity):
    """DepthNPricer's fixed_capacity=True under --fixed-shapes; nothing otherwise, so the default build is the unchanged call."""
    return dict(fixed_capacity=True) if fixed_capacity else {}


def forward_kwargs(forward):
    """forward= for a pricer or oracle under the run's evaluator (--champion-archive); nothing for the V2 default, so its build is the
    unchanged call."""
    return {} if forward is None else dict(forward=forward)


def depth2_pass_pricer(module, executor, model, *, capacity, actor_capacity, k1_own=9, k1_opp=9, block_lanes=2187, digest=False, fixed_capacity=False,
        forward=None):
    """The full-width depth-2 pass (refusal accounting; `searched` maps): DepthNPricer(depth=2, k=9), read through as_depth2_frame.
    The gate prices it with digest=True (its leaves are re-checked against the oracle); diagnostics need no digests. fixed_capacity:
    --fixed-shapes (fixed_kwargs); forward: the run's evaluator (forward_kwargs)."""
    return module.DepthNPricer(executor, model, capacity=capacity, depth=2, k_own=9, k_opp=9, k1_own=k1_own, k1_opp=k1_opp, block_lanes=block_lanes,
        selection='prior', digest=digest, actor_capacity=actor_capacity, **fixed_kwargs(fixed_capacity), **forward_kwargs(forward))


FIXED_SHAPES = ('--fixed-shapes is a harness setting, not a rule: no check, bound, tolerance or verdict reads it. Every DepthNPricer the run '
    'builds takes fixed_capacity=True (turn 1 compiled at exactly --capacity root slots, whatever a call holds) and every oracle stack is '
    'padded to oracle_stack_rows = 2 x the call-split capacity (split_groups puts at most that many requests, two worlds each, in one call), '
    'rounded up to a power of two, so a multi-call run compiles each pricer and oracle program once per process. Padding rows and root '
    'slots are never read. D3\'s FramePricer keeps its per-call buckets. --n2-equivalence refuses the flag: its Depth2Pricer has no fixed '
    'option here, so the two compared pricers would price turn 1 at different batch sizes, and on GPU a bf16 near tie could flip a top-k '
    'set the exact checks compare. On GPU a fixed capacity changes the depth-N pricer\'s batch geometry, so its float values may move '
    'slightly, within the run\'s bounds against the oracle.')


def fixed_shapes_harness(capacity, *, oracle=True):
    """--fixed-shapes' `harness` section (report and start line): the oracle stack's row count is depth2_oracle.bucket(2 x the call-split
    `capacity`), exactly 2 x capacity at a power-of-two capacity >= 4 (None where the mode runs no oracle)."""
    rows = importlib.import_module('gpu_engine.depth2_oracle').bucket(2 * capacity) if oracle else None
    return dict(fixed_shapes=True, pricer_fixed_capacity=True, oracle_stack_rows=rows, split_capacity=capacity, rule=FIXED_SHAPES)


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
               progress=None, decision=None, save=None, evaluator=None, backups=None, pass_sample=8, choices=True, save_oracle=False, mutation=None,
               stack_rows=None, d2_consistency=False):
    """One request list: the full-width depth-2 pass (refusals; `searched` maps), the depth-N pricer (and its block-invariance
    twins), D3, the oracle at every ply and the checks ((e) unsampled unless `decision` says otherwise, (f) unless choices=False).
    save: (directory, prefix) for --save-arrays (save_oracle: the oracle data too, save_oracle_data); mutation: a dict (--mutation-test;
    may carry `source` and `sweep`) keeping the call's frame, oracle data and D3 results in evaluation[MUTATION_INPUTS] for mutation_tests,
    which runs once the run's value bound is known. stack_rows: --fixed-shapes' oracle stack size (depth2_oracle.stack(size=); it refuses
    a call with more world rows), None for the per-call bucket. d2_consistency: D2_ORACLE_CONSISTENCY merged into the oracle-consistency
    check (run_gate: the champion path at depth 2); False changes nothing. Returns (evaluation, summary)."""
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
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe), size=stack_rows)
        data = oracle_levels(oracle=oracle, stack=stack, requests=requests, recipe=recipe, params=params, frame=frame, plan=plan,
            selection_maps=selection_maps, sample=sample, block=block, reference=reference)
        info.update(nodes={str(p): len(r.get('specs', r.get('cells', ()))) for p, r in data['levels'].items()},
            leaves_enumerated=len(data['enumerated'].get('leaf', ())), leaves_checked=0 if data['leaves'] is None else len(data['leaves']['specs']))
    costs['oracle'] = dict(wall_seconds=time.perf_counter() - clock, seconds={k: oracle.seconds[k] - before.get(k, 0.) for k in oracle.seconds})
    evaluation = evaluate_n(frame, d_results, requests, data, context=context, full_turn1=plan['k1_own'] == plan['k1_opp'] == 9, progress=progress)
    evaluation['actors'] = len(params)   # the evaluator parameter sets every value came from (GUARD_DEPTHN_RULE's single-actor condition)
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
    if d2_consistency and not evaluation['geometry']:
        with stage(progress, 'check_d2_oracle_consistency', **context) as info:
            found = d2_oracle_consistency(data, oracle=oracle, stack=stack, params=params, context=context)
            evaluation['oracle'] = g2.merge_findings([evaluation['oracle'], found]);info.update(passed=found['passed'], compared=found['compared'])
    if others and not evaluation['geometry']:
        with stage(progress, 'check_block_invariance', **context) as info:
            evaluation['invariance'] = [g2.block_invariance(frame_arrays(o), frame_arrays(frame), context=context) for o in other_frames]
            info['discrete_equal'] = all(r['passed'] for r in evaluation['invariance'])
    root_legal = [r.root_legal for r in requests]
    with stage(progress, 'refusals', **context):
        refusals = None if evaluation['geometry'] else refusal_report_n(frame, d_results, depth2_frame, root_legal)
    backup_checks(evaluation, frame, data, requests, context=context, backups=backups, decision=sample is None if decision is None else decision,
        choices=choices, sampled=sample is not None, progress=progress)
    if not evaluation['geometry']:evaluation['abs_t1'] = abs_t1(frame)
    saved = []
    if save is not None:
        directory, prefix = save;meta = dict(context, plan=plan, sample_leaves=sample, evaluator=evaluator, timing=getattr(frame, 'timing', None))
        path = Path(directory) / f'{prefix}-{frame_tag(plan["selection"], pricers.k_own[0])}.npz';save_frame(path, frame, requests, meta=meta);saved.append(str(path))
        if depth2_frame is not None:
            path = Path(directory) / f'{prefix}-depth2.npz';save_frame(path, depth2_frame, requests, meta=dict(context, full_width_depth2_pass=True,
                evaluator=evaluator, timing=getattr(depth2_frame, 'timing', None)));saved.append(str(path))
        if save_oracle:
            path = Path(directory) / f'{prefix}-oracle.npz';save_oracle_data(path, data, d_results, requests, meta=dict(context, sample_leaves=sample, actors=len(params),
                frame=f'{prefix}-{frame_tag(plan["selection"], pricers.k_own[0])}.npz', full_turn1=plan['k1_own'] == plan['k1_opp'] == 9));saved.append(str(path))
    if mutation is not None and not evaluation['geometry']:   # judged later, at the run's value bound (mutation_tests)
        evaluation[MUTATION_INPUTS] = dict(mutation, frame=frame, data=data, d_results=d_results, requests=requests, context=context,
            full_turn1=plan['k1_own'] == plan['k1_opp'] == 9)
    summary = dict(context, roots=len(requests), worlds=data['worlds'], plan=plan, refusals=refusals, timing=timing, costs=costs,
        nodes_checked={str(p): len(r.get('specs', r.get('cells', ()))) for p, r in data['levels'].items()},
        leaves_enumerated=len(data['enumerated'].get('leaf', ())), leaves_checked=0 if data['leaves'] is None else len(data['leaves']['specs']),
        forced_leaves=data['forced_leaves'], plies_work=json.loads(json.dumps((getattr(frame, 'timing', None) or {}).get('plies'), default=str)),
        searched_maps=evaluation.get('searched_maps'), saved_frames=saved, depth2_pass=evaluation.get('depth2_pass'), topk=data.get('topk'),
        searched={k: json.loads(json.dumps((getattr(frame, 'timing', None) or {}).get(k), default=str)) for k in ('searched_k', 'searched_widths')}
            if plan['selection'] == 'searched' else None,
        passed=dict(a=evaluation['a']['passed'], b_discrete=evaluation['b']['passed'], d=evaluation['d']['passed'], oracle=evaluation['oracle']['passed'],
            **({'block_invariance_discrete': all(r['passed'] for r in evaluation['invariance'])} if evaluation.get('invariance') else {}),
            **({'e': evaluation['e']['passed']} if 'e' in evaluation else {})),
        choices=None if 'f' not in evaluation else {k: (len(v) if k == 'differing' else v) for k, v in evaluation['f'].items()
            if k in ('nodes', 'comparisons', 'differing', 'one_sided', 'problems', 'not_comparable', 'max_abs_delta_row_q', 'max_abs_delta_child_value', 'reason')})
    return evaluation, summary


def gate_status_n(result, *, sampled=False):
    """FAIL if any check failed; INCOMPLETE if none ran or one is vacuous (no leaf compared by (a), (e) compared no root, or (f) compared
    no interior node outside SKIPPED_SAMPLED); PASS_SAMPLED with sampled leaves ((e) skipped, (f) on fully checked nodes); else PASS."""
    checks = result['checks']
    if any(not c['passed'] for c in checks.values()):return 'FAIL', []
    reasons = [] if checks else ['no check ran']
    if 'a' in checks and not checks['a']['compared'].get('leaf_status'):reasons.append('vacuous: no leaf was compared by (a)')
    if 'e' in checks and not checks['e'].get('roots_compared'):reasons.append('vacuous: (e) compared no root (every root refused or unlabeled)')
    if 'f' in checks and checks['f'].get('status') != 'SKIPPED_SAMPLED' and not checks['f'].get('comparisons'):
        reasons.append('vacuous: (f) compared no interior node')
    if reasons:return 'INCOMPLETE', reasons
    return ('PASS_SAMPLED' if sampled else 'PASS'), []


def summarize_run(evaluations, lanes, *, declared=None):
    """summarize_n (under `declared`, --declared-bounds, when given) plus the block-invariance section labeled by block_lanes."""
    result = summarize_n(evaluations, declared=declared)
    inv = result['checks'].get('block_invariance')
    if inv is not None:inv.pop('child_blocks', None);inv['block_lanes'] = list(lanes)
    return result


# ---------------------------------------------------------------- offline re-verdict of a saved gate.json

REVERDICT_FILE = 'gate-reverdict.json'
REVERDICT_SHOWN = ('compared', 'max_abs_difference', 'bound', 'bound_rule', 'ratio_to_bound', 'within_bound', 'beyond_bound', 'guard_bound',
    'ratio_to_guard', 'within_guard', 'beyond_guard', 'guard_mode')


def reverdict_missing(report):
    """What --reverdict needs that a saved report lacks (or why it is not a depth-N gate report); [] when complete."""
    if not isinstance(report, dict):return ['a JSON object']
    missing = [] if report.get('schema') == SCHEMA else [f'schema {SCHEMA} (got {report.get("schema")!r})']
    if report.get('mode') not in ('SAVED_ROOTS', 'BANK_CHUNK'):missing.append(f'mode SAVED_ROOTS|BANK_CHUNK (got {report.get("mode")!r})')
    missing += [key for key in ('status', 'checks', 'sample_leaves', 'tolerance', 'pricer_options') if key not in report]
    if missing:return missing
    if type((report['pricer_options'] or {}).get('depth')) is not int:missing.append('pricer_options.depth')
    if not isinstance(report['tolerance'], dict):missing.append('tolerance')
    elif type(report['tolerance'].get('value_bound')) not in (int, float):missing.append('tolerance.value_bound')   # (f) is re-judged under it
    if isinstance(report['tolerance'], dict):   # the rule's own record: frozen_problems compares the numbers with it
        missing += [f'tolerance.{key}' for key in ('rule', 'logit_bound', 'depth1_noise_floor') if key not in report['tolerance']]
    checks = report['checks'] if isinstance(report['checks'], dict) else {}
    missing += [f'checks.{name}.passed' for name in ('a', 'c', 'f') if name not in checks]
    missing += [f'checks.{name}.passed' for name, check in checks.items() if not isinstance(check, dict) or 'passed' not in check]
    missing += [f'checks.a.{key}' for key in ('compared', 'mismatched') if not isinstance(checks.get('a', {}).get(key), dict)]
    if isinstance(checks.get('f'), dict) and checks['f'].get('status') != 'NOT_RUN':missing += [f'checks.f.{key}' for key in CHOICE_SAVED if key not in checks['f']]
    quantities = (checks.get('c') or {}).get('quantities')
    if not isinstance(quantities, dict):return missing + ['checks.c.quantities']
    for name, m in quantities.items():
        if name in g2.FLOORS:continue
        need = ('compared', 'within_bound', 'max_abs_difference', 'nonfinite', 'bound') \
            + (('within_guard',) if isinstance(m, dict) and m.get('guard_bound') is not None else ())   # re-bounds (deeper priors, a declaration) read these
        missing += [f'checks.c.quantities.{name}.{key}' for key in need if not isinstance(m, dict) or key not in m]
    return missing


def reverdict(path, *, expected=None):
    """--reverdict: the current verdict rules re-applied offline to a saved gate.json (apply_rules_depthn on (c) at the run's
    depth, the status through gate_status_n; the other checks keep their saved verdicts: their rules are unchanged), written
    to gate-reverdict.json beside it with a `reverdict` section (old status and reasons, changed quantities). Never
    overwrites; refuses (SystemExit, non-zero) a report missing a field the verdict needs. A report with a declared-bounds
    section is re-judged under the declaration it recorded, after saved_declared re-verifies it (refused otherwise); a report
    without one exactly as before. Returns 0 iff PASS/PASS_SAMPLED."""
    path = Path(path);out = path.with_name(REVERDICT_FILE)
    if out.exists():raise SystemExit(f'--reverdict: {out} exists; it is never overwritten')
    try:report = json.loads(path.read_text())
    except (OSError, ValueError) as error:raise SystemExit(f'--reverdict: cannot read {path}: {error}')
    missing = reverdict_missing(report)
    if missing:raise SystemExit(f'--reverdict: {path} lacks what the verdict needs: ' + ', '.join(missing[:FIRST]))
    declared, section = saved_declared(report, expected=expected)
    quantities = report['checks']['c']['quantities'];before = {name: m.get('passed') for name, m in quantities.items()}
    old = dict(status=report['status'], status_reasons=report.get('status_reasons'), failing=[k for k, v in report['checks'].items() if not v['passed']])
    apply_rules_depthn(report, report['pricer_options']['depth'], actors=saved_actors(report), declared=declared)
    f_before = report['checks']['f']['passed'];sampled = report['sample_leaves'] is not None
    report['checks']['f'] = choice_verdict(report['checks']['f'], report['tolerance']['value_bound'], sampled=sampled)   # (f) through the live function
    status, reasons = gate_status_n(report, sampled=sampled)
    changed = [dict(quantity=name, passed_before=before[name], passed=m['passed'], guard_mode=m.get('guard_mode')) for name, m in quantities.items()
        if name not in g2.FLOORS and m['passed'] != before[name]]
    if report['checks']['f']['passed'] != f_before:changed.append(dict(quantity='f', passed_before=f_before, passed=report['checks']['f']['passed']))
    failing = [k for k, v in report['checks'].items() if not v['passed']]
    report.update(status=status, passed=status in ('PASS', 'PASS_SAMPLED'), status_reasons=reasons, reverdict=dict(source=str(path),
        source_status=old['status'], source_status_reasons=old['status_reasons'], source_failing_checks=old['failing'], failing_checks=failing,
        changed_quantities=changed, depth=report['pricer_options']['depth'], rule=GUARD_DEPTHN_RULE,
        deeper_logit_rule=DEEPER_LOGIT_RULE if declared is None else DECLARED_DEEPER_RULE,
        turn1_logit_floor=report['tolerance']['turn1_logit_floor'], deeper_logit_bound=report['tolerance']['deeper_logit_bound'], choice_rule=CHOICE_RULE,
        note='offline: the saved numbers re-judged by the current verdict rules; nothing was re-priced or re-compared',
        **({} if declared is None else dict(declared_bounds_sha256=declared['sha256'], declared_bounds_commit=declared['commit'],
            declared_bounds_reverified=declared['reverified'], declared_bounds='the bounds the report recorded (header.declared_bounds: its sha256 and '
            'git blob sha1 re-verified, the local git history and the source measurement re-checked where available), re-applied'))))
    report = _with_header(report, header_block(report['checks'], declared=section, parity=(report['header'] if isinstance(report.get('header'), dict) else {}).get('merged_depth2_parity')))
    text = json.dumps(report, indent=1, allow_nan=False, default=str) + '\n'   # before the file exists: a failure here leaves none behind
    try:
        with out.open('x') as handle:handle.write(text)
    except FileExistsError:raise SystemExit(f'--reverdict: {out} exists; it is never overwritten')
    for name, m in quantities.items():
        if name not in g2.FLOORS and (m.get('guard_bound') is not None or m.get('bound_rule') in ('deeper_logit_bound', 'declared_deeper_logit_bound')):
            log('REVERDICT_QUANTITY', quantity=name, **{k: m.get(k) for k in REVERDICT_SHOWN}, guard_evidence=m.get('guard_evidence'),
                passed_before=before[name], passed=m['passed'])
    log('REVERDICT', gate=str(path), out=str(out), old_status=old['status'], new_status=status, old_status_reasons=old['status_reasons'],
        status_reasons=reasons, old_failing_checks=old['failing'], failing_checks=failing, changed_quantities=changed,
        checks={k: v['passed'] for k, v in report['checks'].items()}, evaluator_actors=report['tolerance']['evaluator_actors'],
        turn1_logit_floor=report['tolerance']['turn1_logit_floor'],
        deeper_logit_bound=report['tolerance']['deeper_logit_bound'], header=report['header'])
    return 0 if report['passed'] else 1


# ---------------------------------------------------------------- report header (every report the gate writes)

RULES_CHANGED_AFTER_FAILURE = (
    dict(name='rule 1: (e) near-tie rule', summary='(e): a root whose top-two completed-Q gap is <= the run\'s value bound in either frame is a near '
        'tie; its flips are reported, not gated (max |dQ| <= the bound stays gated); (f) applies it per interior-node comparison',
        triggering_run='d2-gate40-b0-r1', trigger='the live B0-i99 depth-2 run d2-gate40-b0-r1 (gate 60a52442) failed (e) at root 15: its top two '
        '1.2e-3 apart in the pricer frame, 2.3e-5 apart in the oracle-heads frame; fixed in 4f88727e', approval='confirmed by the lead 2026-11-06'),
    dict(name='rule 2: searched threat rule revision', summary='(a) under searched: the opponent\'s refutations and lowest-threat fill range over the '
        'learner\'s KEPT moves only (top-k_own by q, then its prior top-1); errored pairs are prepended afterwards and each list truncated to its width '
        '(searched_maps_reference, the pricer\'s 8ade7e21 construction order)', triggering_run=None,
        trigger='an Opus review counterexample, not a run', approval='confirmed by the lead 2026-11-06'),
    dict(name='rule 3: GUARD_DEPTHN_RULE', summary='(c): at plies >= 2 of a depth >= 3 frame the misplacement guard is digest-certified (reported, not '
        'gated) on a single evaluator actor with every value-compared position digest-matched; (f) gates the backup\'s grid: read identity (which '
        'child every entry reads, exact, any rule), the priors handed to the rule (exact, by action, against the gate\'s own softmax of the '
        'pricer\'s logits at the oracle\'s node) and every learner-max entry within the value bound of the oracle\'s child, beside its row q and '
        'its choices, and that per-child grid gate replaces the guard there; the value bound stays gated everywhere', triggering_run='dn-gate40-prior3-r3',
        trigger='the live 40-root depth-3 prior run dn-gate40-prior3-r3 (the guard rule); the per-child grid gate and read identity that replace '
        'the guard, and the exact prior check beside them, were triggered by review findings, not a run: the kill-test sweep on the backup path '
        '(review of 7ab91711), interior entries under another rule (review of a730e321) and wrong priors handed to the rule (review of 27adf5a7)',
        approval='per-child grid gate replaces the guard at plies >= 2: signed off by the lead 2026-11-06'),
    dict(name='rule 4: DEEPER_LOGIT_RULE', summary='(c): priors at plies >= 2 of a depth >= 3 frame are bounded by max(0.1, 2 x the run\'s turn-1 '
        'logit max |diff| against the oracle); turn-1, root and depth2_pass priors keep 0.1', triggering_run='dn-gate40-searched3-r3',
        trigger='the live 40-root depth-3 searched run dn-gate40-searched3-r3', approval='confirmed by the lead 2026-11-06'))
FROZEN_NOTE = 'the gate code at the PR #1833 merge is the frozen rule set for every later run; a rule changed after that names its triggering run here'


def header_block(checks=None, *, configs=None, declared=None, parity=None):
    """Every report's header: (e)'s and (f)'s near-tie counts and flips among near ties (per config under --n2-equivalence; None where the
    check did not run) and the rules changed after a failure, each with its triggering run and approval; under --declared-bounds a
    separate `declared_bounds` section (declared_section: the declaration, its sha256 and DECLARED_NOTE), never mixed into those rules;
    on the champion path's depth-2 surfaces a `merged_depth2_parity` section (merged_depth2_parity)."""
    def e_ties(e):
        return None if not isinstance(e, dict) else dict(near_tie_roots=e.get('near_tie_roots'), near_tie_flips=e.get('near_tie_flips'), q_noise_bound=e.get('q_noise_bound'))
    def f_ties(f):
        return None if not isinstance(f, dict) else dict(near_ties=f.get('near_ties'), near_tie_flips=f.get('near_tie_flips'), status=f.get('status'),
            value_bound=f.get('value_bound'))
    checks = checks or {}
    near = {name: dict(e=e_ties(c.get('e'))) for name, c in configs.items()} if configs is not None else dict(e=e_ties(checks.get('e')), f=f_ties(checks.get('f')))
    block = dict(near_ties=near, rules_changed_after_failure=[dict(r) for r in RULES_CHANGED_AFTER_FAILURE], frozen_rule_set=FROZEN_NOTE)
    if declared is not None:block['declared_bounds'] = declared
    if parity is not None:block['merged_depth2_parity'] = parity
    return block


def _with_header(report, block):
    """report with `header` placed after its evaluator/value_scale/source keys (the first keys of every gate report)."""
    first = [k for k in ('evaluator', 'value_scale', 'source') if k in report]
    return dict([(k, report[k]) for k in first] + [('header', block)] + [(k, v) for k, v in report.items() if k not in first and k != 'header'])


# ---------------------------------------------------------------- declared bounds (--declared-bounds; opt-in, one evaluator)

DECLARED_SCHEMA = 'pokezero.depthn-gate.declared-bounds.v1'
DECLARED_FIELDS = ('schema', 'evaluator', 'bounds', 'derivation', 'declared_at', 'source_measurement', 'declared_by')
DECLARED_KINDS = ('champion',)   # champion only: a V2 head (B1's, under --head-archive) and the bank snapshot's actors keep the frozen rule
DECLARED_PRECISIONS = tuple(ce.PRECISIONS.values())   # the evaluator identity's names ('bfloat16', 'float32'), never the flags' aliases
DECLARED_RULE = ('--declared-bounds (opt-in; champion only: with --champion-archive, refused with --head-archive and the bank snapshot\'s actors): a JSON declaration '
    '(DECLARED_SCHEMA) committed under gpu_engine/declared_bounds/ through a reviewed PR before the run (--declared-bounds-commit; see '
    'DECLARED_NOTE for what is verified and what is only recorded), its bytes pinned by --declared-bounds-sha256 '
    'and bound to ONE evaluator identity (kind, archive sha256, precision; checkpoint sha256 when declared), which must be the run\'s, replaces '
    'the tolerance NUMBERS for that evaluator; no check, mask, rule or status changes. value_bound replaces 2 x depth1_noise_floor (and its 1e-4 '
    'fallback) wherever the value bound applies: (b) t1_value, every (c) value quantity (turn 1, ply nodes, leaves, depth2_pass, raw head), '
    'block-invariance values, (e)\'s q_noise_bound unless near_tie_q_bound is declared, (f)\'s grid entries, row q and near ties, the '
    '--n2-equivalence values and the mutation test. turn1_logit_bound replaces LOGIT_BOUND wherever it applies: turn-1 and root priors in (b) '
    'and (c), depth2_pass priors, block-invariance logits, the --n2-equivalence priors and topk_boundary\'s gap count. Priors at plies >= 2 of '
    'a depth >= 3 frame: deeper_logit_bound as declared, or deeper_logit_rule = max(turn1_logit_bound, factor x turn1_logit_floor) '
    '(DEEPER_LOGIT_RULE\'s construction with the declared base and factor; turn1_logit_bound when the floor is unmeasured or non-finite). '
    'near_tie_q_bound (default value_bound) is (e)\'s q_noise_bound, one number in the frozen rule: the root near-tie threshold of rule 1 and '
    '(e)\'s max |dQ| clause (completed Q mixes the swapped priors too); (f) keeps the value bound (both of its sides solve under the pricer\'s '
    'priors). Unchanged: every exact check ((a), (d), digests, discrete fields, oracle consistency), the misplacement guard (0.25 x the p10 '
    'spread) and GUARD_DEPTHN_RULE, (f)\'s read identity and priors, the statuses, and the kill test (run at the declared value bound). The '
    'measured floors (depth1_noise_floor, root_logit_floor, turn1_logit_floor) are still measured and reported beside the declared bounds, '
    'with the numbers the frozen rule would have given (tolerance.declared_bounds.frozen_rule_would_give; not gated)')
DECLARED_NOTE = ('evaluator-specific bounds for the champion identity in `evaluator`, applied in place of the B1-calibrated tolerance numbers for '
    'that evaluator only. What this gate verified: the declaration\'s bytes hash to --declared-bounds-sha256 (sha256) and pass its schema; the file '
    'lies under gpu_engine/declared_bounds/ of the running checkout (repo_path); its git blob sha1 (git_blob_sha1) and --declared-bounds-commit '
    '(commit) are recorded and, where the local git history holds that commit, the commit holds exactly this blob at repo_path (git_verification; '
    '"not verifiable here" otherwise); the source measurement\'s sha256 where its file is readable from the run (source_measurement_verification); '
    'declared_at is no later than the run\'s start. Not verified here: declared_at and run_started_at are self-reported; that the commit went '
    'through a reviewed PR and was on the integration branch before run_started_at is for an auditor to confirm with git (the commit, holding '
    'git_blob_sha1 at repo_path, an ancestor of the integration branch at a time before run_started_at). It is not a rule change after a failure: '
    'rules_changed_after_failure and the frozen rule set are untouched, and a run without --declared-bounds is judged exactly as before')
DECLARED_SUBDIR = 'gpu_engine/declared_bounds'   # where a declaration is committed, relative to the repository root
REPO_ROOT = Path(__file__).resolve().parents[1]   # the running checkout (tests point it at a scratch repository)
DECLARED_TOLERANCE_RULE = ('declared bounds (DECLARED_RULE): values max |diff| <= the declared value_bound; turn-1, root and depth2_pass '
    'logits <= the declared turn1_logit_bound; deeper priors as declared; the guard, digests, discrete fields and (e)/(f) rules unchanged')
DECLARED_DEEPER_RULE = ('(c) priors at plies >= 2 of a depth >= 3 frame under --declared-bounds: the declared deeper_logit_bound, or the declared '
    'deeper_logit_rule max(turn1_logit_bound, factor x turn1_logit_floor) (turn1_logit_floor measured as in DEEPER_LOGIT_RULE; turn1_logit_bound '
    'when it is not); turn-1, root and depth2_pass priors keep the declared turn1_logit_bound')


class DeclaredBoundsError(ValueError):
    """A declared-bounds file, or its use in this run, refused (fail closed)."""


def _positive(value):
    """A finite JSON number > 0 (never a bool)."""
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _utc(text):
    """The aware UTC datetime of an ISO 8601 text with an explicit zero offset ('Z' or '+00:00'), else None."""
    if not isinstance(text, str):return None
    try:when = datetime.datetime.fromisoformat(text)
    except ValueError:return None
    return when if when.tzinfo is not None and when.utcoffset() == datetime.timedelta(0) else None


def _object(problems, where, value, required, optional=()):
    """Whether `value` is a JSON object; its missing required and unknown keys appended to problems (a misspelt key is refused,
    never ignored)."""
    if not isinstance(value, dict):problems.append(f'{where}: a JSON object');return False
    problems += [f'{where}.{k}: missing' for k in required if k not in value] + [f'{where}.{k}: unknown key' for k in value if k not in required + tuple(optional)]
    return True


def declared_problems(content):
    """Every reason a parsed declaration is refused ([] when valid): exactly DECLARED_FIELDS at the top and no unknown key at any level;
    schema DECLARED_SCHEMA; evaluator {kind in DECLARED_KINDS, archive_sha256 64-hex, precision in DECLARED_PRECISIONS, optional
    checkpoint_sha256 64-hex}; bounds {value_bound, turn1_logit_bound, exactly one of deeper_logit_bound or deeper_logit_rule {factor},
    optional near_tie_q_bound}, each a finite number > 0; derivation and declared_by non-empty text; declared_at ISO 8601 in UTC;
    source_measurement {path non-empty, sha256 64-hex}."""
    problems = []
    if not _object(problems, 'declaration', content, DECLARED_FIELDS):return problems
    if 'schema' in content and content['schema'] != DECLARED_SCHEMA:problems.append(f'schema: {DECLARED_SCHEMA!r} required (got {content["schema"]!r})')
    evaluator = content.get('evaluator')
    if 'evaluator' in content and _object(problems, 'evaluator', evaluator, ('kind', 'archive_sha256', 'precision'), ('checkpoint_sha256',)):
        if 'kind' in evaluator and evaluator['kind'] not in DECLARED_KINDS:problems.append(f'evaluator.kind: one of {DECLARED_KINDS} (got {evaluator["kind"]!r})')
        if 'precision' in evaluator and evaluator['precision'] not in DECLARED_PRECISIONS:
            problems.append(f'evaluator.precision: one of {DECLARED_PRECISIONS}, the identity\'s names (got {evaluator["precision"]!r})')
        problems += [f'evaluator.{k}: a 64-hex sha256' for k in ('archive_sha256', 'checkpoint_sha256') if k in evaluator and not ce._hex64(evaluator[k])]
    bounds = content.get('bounds')
    if 'bounds' in content and _object(problems, 'bounds', bounds, ('value_bound', 'turn1_logit_bound'), ('deeper_logit_bound', 'deeper_logit_rule', 'near_tie_q_bound')):
        problems += [f'bounds.{k}: a finite number > 0' for k in ('value_bound', 'turn1_logit_bound', 'deeper_logit_bound', 'near_tie_q_bound')
            if k in bounds and not _positive(bounds[k])]
        if ('deeper_logit_bound' in bounds) == ('deeper_logit_rule' in bounds):problems.append('bounds: exactly one of deeper_logit_bound or deeper_logit_rule')
        rule = bounds.get('deeper_logit_rule')
        if 'deeper_logit_rule' in bounds and _object(problems, 'bounds.deeper_logit_rule', rule, ('factor',)) and 'factor' in rule and not _positive(rule['factor']):
            problems.append('bounds.deeper_logit_rule.factor: a finite number > 0')
    problems += [f'{k}: non-empty text' for k in ('derivation', 'declared_by') if k in content and not (isinstance(content[k], str) and content[k].strip())]
    if 'declared_at' in content and _utc(content['declared_at']) is None:
        problems.append(f'declared_at: ISO 8601 in UTC with an explicit offset, e.g. 2026-11-07T12:00:00Z (got {content["declared_at"]!r})')
    measurement = content.get('source_measurement')
    if 'source_measurement' in content and _object(problems, 'source_measurement', measurement, ('path', 'sha256')):
        if 'path' in measurement and not (isinstance(measurement['path'], str) and measurement['path'].strip()):problems.append('source_measurement.path: non-empty text')
        if 'sha256' in measurement and not ce._hex64(measurement['sha256']):problems.append('source_measurement.sha256: a 64-hex sha256')
    return problems


def _strict_json(text):
    """json.loads refusing duplicate keys and NaN / Infinity (DeclaredBoundsError)."""
    def pairs(items):
        keys = [k for k, _ in items];duplicated = sorted({k for k in keys if keys.count(k) > 1})
        if duplicated:raise DeclaredBoundsError(f'duplicate key(s) {duplicated}')
        return dict(items)
    def constant(name):raise DeclaredBoundsError(f'{name} is not a JSON number')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def declared_from_bytes(raw, sha256, *, path, now=None):
    """A declaration from its exact bytes: their sha256 must be `sha256`, the text UTF-8, the JSON strict (_strict_json) and valid
    (declared_problems), and declared_at no later than `now` (the run's start, UTC: the bounds are declared BEFORE the run). Returns
    dict(path, sha256, content, content_text, run_started_at); DeclaredBoundsError otherwise."""
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:raise DeclaredBoundsError(f'{path}: its sha256 {got} differs from --declared-bounds-sha256 {sha256}')
    try:text = raw.decode('utf-8')
    except UnicodeDecodeError as error:raise DeclaredBoundsError(f'{path}: not UTF-8 ({error})')
    try:content = _strict_json(text)
    except ValueError as error:raise DeclaredBoundsError(f'{path}: not a JSON declaration ({error})')
    problems = declared_problems(content)
    if problems:raise DeclaredBoundsError(f'{path}: ' + '; '.join(problems))
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if _utc(content['declared_at']) > now:
        raise DeclaredBoundsError(f'{path}: declared_at {content["declared_at"]} is after the run\'s start {now.isoformat()} (bounds are declared before the run)')
    return dict(path=str(path), sha256=got, content=content, content_text=text, run_started_at=now.isoformat())


def git_blob_sha1(raw):
    """git's object id of a file with these bytes (sha1 of "blob <len>\\0" + bytes; `git hash-object`)."""
    return hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()


def _commit_id(text):
    return isinstance(text, str) and re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', text) is not None


def _git(root, *args):
    try:return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):return None


def git_verification(commit, repo_path, blob, *, root=None):
    """What the local git history at `root` (REPO_ROOT) says of the declaration: dict(status, detail) with status 'verified' (the commit is
    here and holds exactly this blob at repo_path), 'mismatch' (another object there), 'absent' (nothing there) or 'not verifiable here' (no
    git, no repository, or the commit is not in the local history, e.g. a shallow or exported checkout). load_declared and saved_declared
    refuse 'mismatch' and 'absent'."""
    root = Path(root or REPO_ROOT);found = _git(root, 'rev-parse', '--verify', '--quiet', f'{commit}^{{commit}}')
    if found is None:return dict(status='not verifiable here', detail='git is not available')
    if found.returncode != 0:return dict(status='not verifiable here', detail=f'commit {commit} is not in a local repository at {root}')
    at = _git(root, 'rev-parse', '--verify', '--quiet', f'{commit}:{repo_path}')
    if at is None or at.returncode != 0:return dict(status='absent', detail=f'commit {commit} has no {repo_path}')
    held = at.stdout.strip()
    return dict(status='verified' if held == blob else 'mismatch', detail=f'commit {commit} holds {held} at {repo_path}', blob_at_commit=held)


def measurement_verification(measurement):
    """The declaration's source measurement: dict(status, ...) 'verified' (its file is readable from here and hashes to the declared sha256),
    'mismatch' (it hashes to another; refused) or 'not verifiable here' (not readable from this run)."""
    path = Path(measurement['path'])
    try:raw = path.read_bytes()
    except OSError as error:return dict(status='not verifiable here', detail=f'{path} is not readable from this run ({type(error).__name__})')
    got = hashlib.sha256(raw).hexdigest()
    return dict(status='verified' if got == measurement['sha256'] else 'mismatch', sha256=got, detail=f'{path} hashes to {got}')


def repo_path_of(path, *, root=None):
    """The declaration's path relative to the repository root, which must lie under DECLARED_SUBDIR (DeclaredBoundsError otherwise)."""
    root = Path(root or REPO_ROOT).resolve();resolved = Path(path).resolve()
    if not resolved.is_relative_to(root / DECLARED_SUBDIR):
        raise DeclaredBoundsError(f'{path}: a declaration is committed under {DECLARED_SUBDIR}/ of the running checkout ({root / DECLARED_SUBDIR}) through a '
            'reviewed PR before the run')
    return resolved.relative_to(root).as_posix()


def load_declared(path, sha256, commit, *, now=None, root=None):
    """--declared-bounds PATH --declared-bounds-sha256 SHA --declared-bounds-commit COMMIT: the file under DECLARED_SUBDIR of the checkout at
    `root` (REPO_ROOT), declared_from_bytes on its bytes, its git blob sha1 and the commit recorded with git_verification (refused on 'mismatch'
    or 'absent'), and the source measurement's measurement_verification (refused on 'mismatch'). Returns the declaration dict."""
    if not _commit_id(commit):raise DeclaredBoundsError(f'--declared-bounds-commit {commit!r}: a full lowercase hex commit id')
    repo_path = repo_path_of(path, root=root)
    try:raw = Path(path).read_bytes()
    except OSError as error:raise DeclaredBoundsError(f'cannot read {path}: {error}')
    declared = declared_from_bytes(raw, sha256, path=path, now=now);blob = git_blob_sha1(raw)
    git = git_verification(commit, repo_path, blob, root=root)
    if git['status'] in ('absent', 'mismatch'):raise DeclaredBoundsError(f'{path}: not the file committed at {commit}: {git["detail"]} (this file is {blob})')
    measurement = measurement_verification(declared['content']['source_measurement'])
    if measurement['status'] == 'mismatch':raise DeclaredBoundsError(f'source_measurement: {measurement["detail"]}, not the declared {declared["content"]["source_measurement"]["sha256"]}')
    return dict(declared, repo_path=repo_path, git_blob_sha1=blob, commit=commit, git_verification=git, source_measurement_verification=measurement)


def evaluator_identity(evaluator):
    """What a declaration is matched against, from an evaluator header (g2.evaluator_header; a report's `evaluator`)."""
    e = evaluator if isinstance(evaluator, dict) else {}
    return dict(override=e.get('override') is True, kind=e.get('kind'), archive_sha256=e.get('archive_sha256'), precision=e.get('precision'),
        checkpoint_sha256=e.get('checkpoint_sha256'))


def args_identity(args):
    """The evaluator identity the champion flags name, before anything is loaded (kind champion, its archive and checkpoint sha256, its
    precision), or none (no --champion-archive: a declaration never applies). load_roots resolves the same identity into the evaluator
    header, which main checks again."""
    if getattr(args, 'champion_archive', None):
        return dict(override=True, kind='champion', archive_sha256=args.champion_archive_sha256, precision=ce.PRECISIONS[args.champion_precision or 'bf16'],
            checkpoint_sha256=args.champion_checkpoint_sha256)
    return dict(override=False, kind=None, archive_sha256=None, precision=None, checkpoint_sha256=None)


def evaluator_mismatches(content, identity):
    """Why a valid declaration does not apply to a run's evaluator identity ([] when it does): an override must price (one parameter set
    with one identity), and kind, archive sha256 and precision (and checkpoint sha256, when declared) must be the run's."""
    declared = content['evaluator']
    out = [] if identity.get('override') else ['the run has no champion evaluator: a declaration binds one champion identity']
    for key in ('kind', 'archive_sha256', 'precision') + (('checkpoint_sha256',) if 'checkpoint_sha256' in declared else ()):
        if declared[key] != identity.get(key):out.append(f'evaluator.{key}: declared {declared[key]!r}, the run\'s {identity.get(key)!r}')
    return out


def declared_numbers(declared):
    """The declaration's numbers as floats: value_bound, turn1_logit_bound, deeper_logit_bound or deeper_logit_factor (one is None) and
    near_tie_q_bound (None: the value bound)."""
    b = declared['content']['bounds'];number = lambda key: None if key not in b else float(b[key])
    return dict(value_bound=number('value_bound'), turn1_logit_bound=number('turn1_logit_bound'), deeper_logit_bound=number('deeper_logit_bound'),
        deeper_logit_factor=None if 'deeper_logit_rule' not in b else float(b['deeper_logit_rule']['factor']), near_tie_q_bound=number('near_tie_q_bound'))


def declared_rule(rule, declared):
    """A tolerance rule (g2.tolerance_rule's) under a declaration: value_bound and logit_bound replaced by the declared value and turn-1 logit
    bounds, q_noise_bound set when near_tie_q_bound is declared, the frozen rule's own entries kept under frozen_rule (reported, not gated);
    every other entry (the measured depth-1 floor, the guard) unchanged. declared None: the rule itself."""
    if declared is None:return rule
    n = declared_numbers(declared)
    out = dict(rule, rule=DECLARED_TOLERANCE_RULE, value_bound=n['value_bound'], value_bound_source=f'declared value_bound (--declared-bounds sha256 {declared["sha256"]})',
        logit_bound=n['turn1_logit_bound'], frozen_rule={k: rule.get(k) for k in ('rule', 'value_bound', 'value_bound_source', 'logit_bound')},
        declared_bounds=dict(sha256=declared['sha256'], rule=DECLARED_RULE))
    if n['near_tie_q_bound'] is not None:out['q_noise_bound'] = n['near_tie_q_bound']
    return out


def declared_kwargs(declared):
    """g2.summarize's bounds hook under a declaration; nothing otherwise, so the default call is the unchanged one."""
    return {} if declared is None else dict(bounds=lambda rule: declared_rule(rule, declared))


def declared_deeper(declared, floor, floor_source):
    """(deeper_logit_bound, source) under a declaration: deeper_logit_bound as declared, or deeper_logit_rule's max(turn1_logit_bound,
    factor x turn1_logit_floor), turn1_logit_bound when the floor is None (unmeasured or non-finite: turn1_logit_floor's reason)."""
    n = declared_numbers(declared)
    if n['deeper_logit_bound'] is not None:return n['deeper_logit_bound'], 'declared deeper_logit_bound'
    base, factor = n['turn1_logit_bound'], n['deeper_logit_factor']
    if floor is None:return base, 'declared turn1_logit_bound: turn1_logit_floor ' + floor_source.split(': ', 1)[-1]
    return max(base, factor * floor), (f'declared deeper_logit_rule: {factor:g} x turn1_logit_floor ({floor_source})' if factor * floor > base
        else f'declared deeper_logit_rule: turn1_logit_bound ({factor:g} x turn1_logit_floor, {floor_source}, is below it)')


def declared_bound_for(name, depth, numbers, deeper):
    """(bound, bound_rule) of (c) quantity `name` under a declaration: deeper priors the deeper bound, other priors the turn-1 logit bound,
    values the value bound."""
    if deeper_logit(name, depth):return deeper, 'declared_deeper_logit_bound'
    if g2.is_logit(name):return numbers['turn1_logit_bound'], 'declared_turn1_logit_bound'
    return numbers['value_bound'], 'declared_value_bound'


def declared_tolerance(tolerance, declared, quantities, frozen_deeper):
    """tolerance.declared_bounds: the numbers applied, the floors this run measured and what the frozen rule would have given (not gated)."""
    n = declared_numbers(declared);frozen = tolerance.get('frozen_rule') or {}
    return dict(sha256=declared['sha256'], path=declared['path'], rule=DECLARED_RULE,
        applied=dict(value_bound=n['value_bound'], turn1_logit_bound=n['turn1_logit_bound'], deeper_logit_bound=tolerance['deeper_logit_bound'],
            deeper_logit_bound_source=tolerance['deeper_logit_bound_source'], q_noise_bound=n['value_bound'] if n['near_tie_q_bound'] is None else n['near_tie_q_bound'],
            guard_bound=tolerance.get('guard_bound')),
        measured=dict(depth1_noise_floor=tolerance.get('depth1_noise_floor'), root_logit_floor=(quantities.get('root_logit_floor') or {}).get('max_abs_difference'),
            turn1_logit_floor=tolerance.get('turn1_logit_floor')),
        frozen_rule_would_give=dict(value_bound=frozen.get('value_bound'), value_bound_source=frozen.get('value_bound_source'), logit_bound=frozen.get('logit_bound'),
            deeper_logit_bound=frozen_deeper, note='the B1-calibrated rule\'s numbers on this run\'s floors, reported beside the declared bounds, not gated'))


def declared_section(declared, evaluator):
    """The header's `declared_bounds` section: the declaration (parsed and as its exact text, whose sha256 --reverdict re-verifies), its
    sha256 and path, the run evaluator it was matched against, the run's start and DECLARED_NOTE."""
    return dict(sha256=declared['sha256'], path=declared['path'], repo_path=declared['repo_path'], git_blob_sha1=declared['git_blob_sha1'],
        commit=declared['commit'], git_verification=declared['git_verification'], source_measurement_verification=declared['source_measurement_verification'],
        content=declared['content'], content_text=declared['content_text'], evaluator={k: v for k, v in evaluator_identity(evaluator).items() if k != 'override'},
        run_started_at=declared['run_started_at'], note=DECLARED_NOTE, rule=DECLARED_RULE)


def declared_traces(report):
    """Where a report still shows a declaration (header or tolerance sections, the declared tolerance texts and sources, a (c) quantity's
    declared bound_rule, a reverdict of one); [] for a report judged under the frozen rule."""
    tolerance = report.get('tolerance') if isinstance(report.get('tolerance'), dict) else {};header = report.get('header') if isinstance(report.get('header'), dict) else {}
    found = [f'{where}.{key}' for where, part, key in (('header', header, 'declared_bounds'), ('tolerance', tolerance, 'declared_bounds'), ('tolerance', tolerance, 'frozen_rule'))
        if key in part]
    found += [f'tolerance.{key}' for key, text in (('rule', DECLARED_TOLERANCE_RULE), ('deeper_logit_rule', DECLARED_DEEPER_RULE)) if tolerance.get(key) == text]
    found += [f'tolerance.{key}' for key in ('value_bound_source', 'deeper_logit_bound_source') if str(tolerance.get(key) or '').startswith('declared')]
    quantities = ((report.get('checks') or {}).get('c') or {}).get('quantities') if isinstance(report.get('checks'), dict) else None
    found += [f'checks.c.quantities.{n}.bound_rule' for n, m in (quantities or {}).items() if isinstance(m, dict) and str(m.get('bound_rule') or '').startswith('declared_')]
    if isinstance(report.get('reverdict'), dict) and 'declared_bounds_sha256' in report['reverdict']:found.append('reverdict.declared_bounds_sha256')
    return found


def frozen_problems(report):
    """Why a report without a declaration was not judged under the frozen rule's numbers ([] when it was): tolerance.rule is TOLERANCE_RULE,
    logit_bound LOGIT_BOUND, value_bound 2 x depth1_noise_floor (VALUE_FALLBACK without a floor), and every saved bound of (b)'s values and of
    the (c) quantities --reverdict keeps is its class's (values the value bound, priors LOGIT_BOUND; the deeper priors are re-bound anyway)."""
    tolerance = report['tolerance'];floor = tolerance.get('depth1_noise_floor');problems = []
    value = g2.VALUE_FALLBACK if floor is None else g2.FLOOR_FACTOR * floor if type(floor) in (int, float) else None
    if tolerance.get('rule') != g2.TOLERANCE_RULE:problems.append('tolerance.rule is not the frozen TOLERANCE_RULE')
    if tolerance.get('logit_bound') != g2.LOGIT_BOUND:problems.append(f'tolerance.logit_bound {tolerance.get("logit_bound")!r} is not LOGIT_BOUND {g2.LOGIT_BOUND}')
    if value is None or tolerance.get('value_bound') != value:
        problems.append(f'tolerance.value_bound {tolerance.get("value_bound")!r} is not the frozen rule\'s {value!r} (2 x depth1_noise_floor {floor!r})')
    depth = report['pricer_options']['depth'];class_bound = lambda name: g2.LOGIT_BOUND if g2.is_logit(name) else tolerance.get('value_bound')
    sections = (('c', report['checks']['c'].get('quantities') or {}), ('b', (report['checks'].get('b') or {}).get('values') or {}))
    problems += [f'checks.{check}.{name}.bound {m.get("bound")!r} is not {class_bound(name)!r}' for check, part in sections for name, m in part.items()
        if name not in g2.FLOORS and not (check == 'c' and deeper_logit(name, depth)) and isinstance(m, dict) and m.get('bound') != class_bound(name)]
    return problems


def saved_declared(report, *, expected=None):
    """(declared, header section) a report recorded under --declared-bounds, re-verified for --reverdict, or (None, None) for a report judged
    under the frozen rule. Fail closed (SystemExit) unless: with a declaration, the header section's content_text hashes to its sha256 and to
    its git blob sha1, parses (strictly) to its content, a valid declaration, matching the report's evaluator header, declared_at no later than
    the recorded run start, its repo_path under DECLARED_SUBDIR and its commit a full commit id, git_verification and the source
    measurement re-run here contradicting nothing ('mismatch' / 'absent'), and the tolerance section naming the same sha256 with the declared
    value and turn-1 logit bounds; without one, no trace of a declaration (declared_traces) and the frozen numbers (frozen_problems).
    expected: dict(commit, sha256) given on the command line (the pre-registration record): each given must be the recorded one."""
    expected = {k: v for k, v in (expected or {}).items() if v is not None}
    header = report.get('header') if isinstance(report.get('header'), dict) else {}
    section, recorded = header.get('declared_bounds'), (report.get('tolerance') or {}).get('declared_bounds')
    if section is None and recorded is None:
        traces, frozen = declared_traces(report), frozen_problems(report)
        if traces:raise SystemExit('--reverdict: a declaration left traces but no declared_bounds sections (fail closed): ' + ', '.join(traces[:FIRST]))
        if frozen:raise SystemExit('--reverdict: the report is not judged under the frozen rule\'s numbers (fail closed): ' + '; '.join(frozen[:FIRST]))
        if expected:raise SystemExit('--reverdict: --declared-bounds-commit/--declared-bounds-sha256 given, but the report records no declaration (fail closed)')
        return None, None
    problems, content, text, sha, again = [], None, None, None, {}
    if not isinstance(section, dict):problems.append('tolerance.declared_bounds without header.declared_bounds')
    else:
        text, sha = section.get('content_text'), section.get('sha256')
        if not isinstance(text, str) or hashlib.sha256(text.encode('utf-8')).hexdigest() != sha:
            problems.append('header.declared_bounds.content_text does not hash to its sha256')
        else:
            try:content = _strict_json(text)
            except ValueError as error:problems.append(f'header.declared_bounds.content_text: {error}')
            if git_blob_sha1(text.encode('utf-8')) != section.get('git_blob_sha1'):problems.append('header.declared_bounds.content_text does not hash to its git_blob_sha1')
    if content is not None:
        problems += declared_problems(content) + ([] if content == section.get('content') else ['header.declared_bounds.content differs from its content_text'])
    if content is not None and not problems:
        problems += evaluator_mismatches(content, evaluator_identity(report.get('evaluator')))
        started = _utc(section.get('run_started_at'))
        if started is None or _utc(content['declared_at']) > started:problems.append('declared_at is not before the recorded run start')
        if not _commit_id(section.get('commit')):problems.append('header.declared_bounds.commit is not a full commit id')
        if not str(section.get('repo_path') or '').startswith(DECLARED_SUBDIR + '/'):problems.append(f'header.declared_bounds.repo_path is not under {DECLARED_SUBDIR}/')
        problems += [f'--declared-bounds-{key.replace("_", "-")} {value} is not the recorded {section.get(key)}' for key, value in expected.items() if section.get(key) != value]
        if not problems:
            again = dict(git=git_verification(section['commit'], section['repo_path'], section['git_blob_sha1']), source_measurement=measurement_verification(content['source_measurement']))
            problems += [f'{name}: {result["detail"]}' for name, result in again.items() if result['status'] in ('mismatch', 'absent')]
        n, tolerance = declared_numbers(dict(content=content)), report.get('tolerance') or {}
        if not isinstance(recorded, dict) or recorded.get('sha256') != sha:problems.append('tolerance.declared_bounds does not name the same sha256')
        if (tolerance.get('value_bound'), tolerance.get('logit_bound')) != (n['value_bound'], n['turn1_logit_bound']):
            problems.append(f'tolerance value_bound / logit_bound {[tolerance.get("value_bound"), tolerance.get("logit_bound")]} are not the declared '
                f'{[n["value_bound"], n["turn1_logit_bound"]]}')
    if problems:raise SystemExit('--reverdict: the report\'s declared bounds fail their checks (fail closed): ' + '; '.join(problems[:FIRST]))
    keep = ('path', 'repo_path', 'git_blob_sha1', 'commit', 'git_verification', 'source_measurement_verification', 'run_started_at')
    return dict({k: section.get(k) for k in keep}, sha256=sha, content=content, content_text=text, reverified=dict(again, expected=expected)), section


# ---------------------------------------------------------------- the champion's depth-2 surfaces against the merged depth-2 gate

D2_ORACLE_CONSISTENCY = ('oracle consistency at depth 2 on the champion path (gated, exact): the depth-2 oracle\'s own leaf walk '
    '(depth2_oracle.Oracle.leaves, inherited by DepthNOracle) is re-run on the checked leaves and judged by d3_depth2_gate.check_oracle against the '
    'turn-1 children pass exactly as the merged depth-2 gate does (leaf_child_known, leaf_child_expanded, leaf_child_invalid, leaf_child_done, '
    'leaf_child_own_digest, leaf_child_peer_digest: every leaf\'s walk re-derives its parent child identically); its status, error, terminal '
    'value and leaf digest (on the (a) digest mask) must also equal the depth-N leaf walk\'s (leaf_walk_*), so the determinism it certifies is '
    'that of the leaves (a)-(c) compare. The extra oracle leaf walk is the cost')
CHILD_BLOCK_INVARIANCE = ('--n2-equivalence --block-invariance on the champion path (gated): each config\'s merged Depth2Pricer is priced at '
    'child_block 27 and 81 (d3_depth2_gate.INVARIANCE_BLOCKS; the 81 frame is the one compared with DepthNPricer N=2, as the merged gate\'s '
    'checked frame), and d3_depth2_gate.block_invariance / merge_invariance judge them as the merged gate does: discrete arrays bit-equal, '
    'floats within the run\'s rule (logits the logit bound, *value the value bound and guard, other floats exact)')
MERGED_DEPTH2_DECISION = ('--n2-equivalence on the champion path (gated): the merged gate\'s (e) on each config\'s merged Depth2Pricer frame. The '
    'oracle data are built as d3_depth2_gate.price_call builds them (turn-1 cells at the pricer\'s k1 widths, ranked by its root prior; the children; '
    'every turn-2 leaf it enumerates, ranked by its own top-k priors; depth2_oracle.Oracle.leaves on each), and d3_depth2_gate.decision_check judges '
    'depth2_backups.learner_max_root on the frame against its oracle-heads copy under the run\'s q_noise_bound, as the merged gate\'s (e) '
    '(learner_max_choice cell_choice_differs reported, not gated). The cost is the oracle\'s walk over every enumerated turn-2 leaf of both configs')
MERGED_DEPTH2_MATCHED = ('the merged depth-2 gate\'s checks and where this gate makes them on the champion path: (a) check_exact (turn 1 by the '
    'same function, leaves on the same digest mask), (b) check_d (the same function), (c) check_heads (the same function; leaf_value is t2_value, '
    'raw heads aside), (d) the enumeration, oracle consistency (D2_ORACLE_CONSISTENCY, gate --depth 2), (e) the decision rule (decision_verdict, '
    'the same near-tie rule) in gate mode with depthn_backups on the DepthNPricer and, under --n2-equivalence, with depth2_backups on the merged '
    'Depth2Pricer (MERGED_DEPTH2_DECISION), block invariance of the Depth2Pricer (CHILD_BLOCK_INVARIANCE, --n2-equivalence --block-invariance), the '
    'misplacement guard (gated, its p10 as below) and the refusal accounting (the reference rule and depth1, as below)')
MERGED_DEPTH2_DIFFERENCES = (
    dict(name='reported-only sections absent', effect='none on the verdict', detail='reported, not gated, by the merged gate and absent here: '
        'oracle_turn1_errors_by_code and oracle_turn2_errors_by_code_checked_leaves; the by_block refusal breakdown and the calls count; the '
        'full-width depth-2 pass\'s frame-level turn1/turn2 errors_by_code, depth2_gate_count and depth2_sources_agree (only its depth2 section is '
        'kept); the values section (|v| per frame band, cell_a2_spread, root_q_spread, the near-tie share); the sibling spread'),
    dict(name='(e) cell_choice_differs', effect='stricter here', detail='the merged gate counts learner_max_choice per-cell differences under (e), '
        'reported, not gated; in gate mode it is absent, and (f) gates every turn-1 cell\'s learner choice and each own row\'s opponent choice exactly '
        'except near ties; under --n2-equivalence MERGED_DEPTH2_DECISION reports it as the merged gate does'),
    dict(name='misplacement guard p10', effect='looser or equal here', detail='the p10 is over NON-ZERO last-ply own-slot spreads (GUARD_SPREAD_RULE) '
        'where the merged gate takes it over every cell_a2_spread, so the guard is at least as wide; a run whose spreads are all zero skips the '
        'guard here, where the merged gate\'s guard would be 0 and fail every value comparison; gated in both'),
    dict(name='raw-head check', effect='not applicable to the champion', detail='--raw-head-check is refused with --champion-archive: the merged '
        'gate\'s check guards a V2 head\'s D-arm geometry substitute, and nothing is substituted for the champion (every pricer and the oracle '
        'run the evaluator itself). Its slice and wiring are checked exactly on the first root (ChampionEvaluator.check) and every digest hashes '
        'its 23-token view; there is no per-leaf adapter-vs-champion.forward check'),
    dict(name='block invariance in gate mode', effect='a different pricer', detail='gate --block-invariance varies the DepthNPricer\'s block_lanes '
        '(729 vs 2187); the merged Depth2Pricer\'s child-block invariance is gated only under --n2-equivalence --block-invariance, whose compared '
        'Depth2Pricer is then child_block 81 (the default is 27)'),
    dict(name='backups', effect='none on the rule', detail='(e) runs depthn_backups.learner_max_root on the DepthNPricer in gate mode; under '
        '--n2-equivalence depth2_backups runs on both N=2 surfaces (pricer against pricer) and, on the merged Depth2Pricer against the oracle heads, '
        'as the merged gate\'s (e) (MERGED_DEPTH2_DECISION)'),
    dict(name='sampling and offline modes', effect='none on the rule', detail='--sample-leaves keeps the same forced terminal/error leaves with other '
        'per-world seeds (other sampled leaves); the merged gate\'s offline --recheck has no counterpart (--reverdict re-judges the verdict only)'),
    dict(name='extra here', effect='stricter here', detail='(a) also gates the ply-2 maps, the engines\' widths and the full-width depth-2 pass; (f) '
        'gates the backup\'s grid, read identity and priors; oracle consistency also gates the two leaf walks equal (leaf_walk_*)'),
    dict(name='bounds', effect='as declared', detail='at depth 2 the frozen rule applies as in the merged gate unless --declared-bounds replaces its '
        'numbers (header.declared_bounds)'))


def d2_oracle_consistency(data, *, oracle, stack, params, context=None):
    """D2_ORACLE_CONSISTENCY on one depth-2 call: the inherited depth-2 leaf walk on data's checked leaves, g2.check_oracle against the
    turn-1 children pass, and the two leaf walks' status, error, terminal value and viewed digest equal. Returns a findings report."""
    found = Findings(context);leaves = data['leaves']
    if int(data['plan']['depth']) != 2:raise ValueError('the depth-2 oracle consistency check needs a depth-2 plan')
    if leaves is None or not len(leaves['specs']):return found.report()
    specs = np.asarray(leaves['specs'], np.int64).reshape(-1, 6);walked = oracle.leaves(stack, specs, params)
    g2.check_oracle(dict(children=data['children'], leaves=walked), found)
    for name in ('status', 'error'):
        found.record(f'leaf_walk_{name}', specs, np.asarray(walked[name]) == np.asarray(leaves[name]), walked[name], leaves[name])
    terminal = (np.asarray(walked['status']) == 1) & (np.asarray(leaves['status']) == 1)
    found.record('leaf_walk_value_terminal', specs[terminal], (np.asarray(walked['value']) == np.asarray(leaves['value']))[terminal],
        np.asarray(walked['value'])[terminal], np.asarray(leaves['value'])[terminal])
    viewed = _leaf_viewed(np.asarray(leaves['status']), np.asarray(leaves['error']))
    a, b = np.asarray(walked['digest'], np.uint32), np.asarray(leaves['digest'], np.uint32)
    found.record('leaf_walk_digest', specs[viewed], (a == b).all(axis=-1)[viewed], a[viewed], b[viewed])
    return found.report()


def merged_depth2_decision(frame, pricer, *, requests, recipe, oracle, stack, params, roots, children, backups, context=None):
    """MERGED_DEPTH2_DECISION on one config's merged Depth2Pricer frame (pricer: the Depth2Pricer that priced it, read for its widths): the
    merged gate's oracle data for that frame, then g2.decision_check with `backups` (depth2_backups). Returns the per-call (e) report."""
    d2ref = importlib.import_module('gpu_engine.depth2_oracle')
    widths = {k: int(getattr(pricer, k)) for k in ('k_own', 'k_opp', 'k1_own', 'k1_opp')};usable = not g2.frame_geometry(frame, len(requests))
    if widths['k1_own'] < 9 or widths['k1_opp'] < 9:   # the merged gate's cells: k1 widths, the k1_opp selection ranked by the pricer's root prior
        root_priors = {k: g2.field(frame, 'root_peer_logits')[k] for k in roots} if usable and widths['k1_opp'] < 9 else None
        children = oracle.children(stack, d2ref.turn1_cells(requests, recipe, roots, k1_own=widths['k1_own'], k1_opp=widths['k1_opp'], root_priors=root_priors), params)
    priors = None
    if usable and len(children['cells']) and (widths['k_own'] < 9 or widths['k_opp'] < 9):
        at = tuple(np.asarray(children['cells']).T);priors = dict(own=g2.field(frame, 't1_own_logits')[at], peer=g2.field(frame, 't1_peer_logits')[at])
    specs = d2ref.turn2_leaves(children, k_own=widths['k_own'], k_opp=widths['k_opp'], priors=priors) if len(children['cells']) else []
    data = dict(roots=roots, children=children, leaves=oracle.leaves(stack, specs, params), enumerated=specs)
    return g2.decision_check(frame, data, requests, context=context, backups=backups)


def merged_depth2_parity(mode, *, oracle_consistency=False, child_block_invariance=False):
    """The header's merged_depth2_parity section (champion path, depth-2 surfaces: gate --depth 2 or --n2-equivalence): what this run
    gates of the merged depth-2 gate's checks, whether the two champion-path additions ran here, and every remaining difference, so the
    report reads neither stricter nor looser than it is."""
    ran = dict(gate=('ran (D2_ORACLE_CONSISTENCY), gated in checks.oracle_consistency' if oracle_consistency else 'not run'),
        n2_equivalence='not in this mode: gate --depth 2 runs it')
    blocks = dict(gate='not in this mode: --n2-equivalence --block-invariance checks the merged Depth2Pricer across child blocks',
        n2_equivalence=('ran (CHILD_BLOCK_INVARIANCE, child_block 27 vs 81), gated per config in configs.*.block_invariance' if child_block_invariance
            else 'not run: pass --block-invariance with --n2-equivalence'))
    decision = dict(gate='not in this mode: --n2-equivalence runs it on the merged Depth2Pricer',
        n2_equivalence='ran (MERGED_DEPTH2_DECISION), gated per config in configs.*.e_merged_depth2')
    return dict(mode=mode, matched=MERGED_DEPTH2_MATCHED, champion_path_additions=dict(d2_oracle_consistency=ran[mode], child_block_invariance=blocks[mode],
        merged_depth2_decision=decision[mode], rules=dict(d2_oracle_consistency=D2_ORACLE_CONSISTENCY, child_block_invariance=CHILD_BLOCK_INVARIANCE,
            merged_depth2_decision=MERGED_DEPTH2_DECISION)),
        remaining_differences=[dict(d) for d in MERGED_DEPTH2_DIFFERENCES],
        note='the merged depth-2 gate (d3_depth2_gate) has no champion flags; on the champion path this gate\'s depth-2 surfaces stand in for it, '
            'with these additions and differences')


def check_declared_arguments(parser, args):
    """--declared-bounds PATH and --declared-bounds-sha256 SHA go together (a 64-hex sha256), champion only (--champion-archive: one
    parameter set with one identity for the declaration to bind; refused with --head-archive, whose V2 heads include B1's, and on the bank
    snapshot's actors, the B1-calibrated default path), and only where a verdict is priced live: refused with
    --depth3-diagnostics (no oracle, no bound) and the offline modes (--diagnostics-from, --mutation-from; --reverdict takes no option and
    re-applies the declaration its report recorded)."""
    given = [getattr(args, name, None) for name in ('declared_bounds', 'declared_bounds_sha256', 'declared_bounds_commit')]
    if not any(given):return args
    if not all(given):parser.error('--declared-bounds, --declared-bounds-sha256 and --declared-bounds-commit go together')
    if not ce._hex64(args.declared_bounds_sha256):parser.error('--declared-bounds-sha256 must be a 64-hex sha256')
    if not _commit_id(args.declared_bounds_commit):parser.error('--declared-bounds-commit must be a full lowercase hex commit id')
    if args.diagnostics_from or args.mutation_from:parser.error('--declared-bounds applies to a live gate run: --diagnostics-from / --mutation-from judge nothing live')
    if args.head_archive:parser.error('--declared-bounds applies to the champion only (--champion-archive): a V2 head under --head-archive (B1\'s '
        'included) keeps the frozen rule')
    if not args.champion_archive:parser.error('--declared-bounds needs --champion-archive: a declaration binds one champion identity, and the bank '
        'snapshot\'s actors keep the frozen rule')
    if args.depth3_diagnostics:parser.error('--declared-bounds does not apply to --depth3-diagnostics (no oracle, no bound)')
    return args


# ---------------------------------------------------------------- mutation kill test (--mutation-test, --mutation-from)

MUTATION_SEED = 20261106
SWEEP_SWAPS = 5000
MUTANTS = tuple((path, kind, variant) for path in ('backup', 'scatter') for kind in ('sibling', 'cousin') for variant in ('single', 'systematic')) \
    + (('backup', 'turn1', 'systematic'),   # turn-1 cells reading their interior (ply-2) children swapped
       ('backup', 'priors', 'systematic'))   # every node handing its rule a uniform pi_own (values under learner-max unchanged)
UNIFORM_PI_OWN = lambda pi, *where: [1. / len(pi)] * len(pi)
SWEEP_PATHS = ('scatter', 'backup')
MUTATION_FILE = 'mutation-test.json'
# depthn_backups.node_value patched in memory (each anchor must occur exactly once in the source): its child reads rerouted through _SWAP
# (keyed by the parent's ply: the eight mutants' pairs sit at ply >= 2, the turn-1 mutant's at ply 1), the grid it hands its rule passed
# through _AT_RULE(grid, root, world, ply, node) and its pi_own through _PRIORS(pi_own, root, world, ply, node) (identity unless a mutant or a
# test sets them).
BACKUP_PATCH = (
    ('            if leaf:\n', '            _r,_w,_n,_i,_x=_SWAP.get((root,world,ply,node,i,x),(root,world,node,i,x))\n'
        '            if leaf:\n'),
    ('frame.leaf_status[root,world,node,i,x]', 'frame.leaf_status[_r,_w,_n,_i,_x]'),
    ('frame.leaf_value[root,world,node,i,x]', 'frame.leaf_value[_r,_w,_n,_i,_x]'),
    ('node_value(frame,root,world,ply+1,node*K+i*k_opp+x,rule)', 'node_value(frame,_r,_w,ply+1,_n*K+_i*k_opp+_x,rule)'),
    ('rule(grid,pi_own,pi_opp)', 'rule(_AT_RULE(grid,root,world,ply,node),pi_own,pi_opp)'),
    ('pi_own=_softmax(own_logits,[int(own_actions[i])for i in own_slots])', 'pi_own=_PRIORS(_softmax(own_logits,[int(own_actions[i])for i in own_slots]),root,world,ply,node)'))
MUTATION_RULE = ('--mutation-test, after every call is priced and the run\'s value bound is known, on each call\'s frame and oracle data (never the '
    'pricer\'s files): eight misplacements at parents of ply >= 2 (children at ply >= 3), {backup path, scatter path} x {sibling, cousin} x '
    '{single, systematic}, a ninth at the turn-1 cells (backup path, systematic), a tenth handing the rule wrong priors (backup path), and a '
    'seeded sweep on both paths. Backup path: depthn_backups\' SOURCE executed into a separate module with node_value\'s '
    'child reads rerouted through a swap map INSIDE the source (BACKUP_PATCH), passed as the backups of (e) and (f); the frame is untouched, so '
    '(a)-(d) see clean values and (f)\'s per-child grid gate has to catch it. Scatter path: the pair\'s stored VALUES exchanged in a copy of the '
    'frame (leaf_value at the last ply, the child ply\'s node value otherwise), digests and statuses untouched, then every check. Single: P = the '
    'first parent at ply >= 2 in (root, world, ply, node) order whose clean learner choice is outside the near ties of both frames; sibling = the '
    'pair of P\'s children whose swap most reverses P\'s pricer-side choice (margin = best other row q minus the clean choice\'s, after the swap); '
    'cousin = P\' a different priced parent at the same ply in the SAME root and world (the same grandparent first; the next root and world only '
    'when none, recorded as a fallback), the (c1 under P, c2 under P\') pair that most reverses it; why each pair is material is recorded. '
    'Systematic: sibling = own slots 0 and 1 swapped for every opponent slot at every parent; cousin = consecutive parents under each grandparent '
    'paired, their (0, 0) children swapped. The clean call and each mutant are judged alone through summarize_run and gate_status_n at the RUN\'s '
    'value bound (the run\'s depth-1 floor substituted for the call\'s) with the evaluator actors set to 1, so GUARD_DEPTHN_RULE certifies the '
    'guard past turn 1 (report-only) wherever the digests match; KILLED = FAIL with a gated check failing other than through a guard alone. Sweep '
    '(seeded): up to SWEEP_SWAPS single sibling and SWEEP_SWAPS single cousin (same root and world, different parents) swaps of (c)-scored leaves, '
    'drawn uniformly without replacement, each applied on BOTH paths: scatter (caught by (c) when either swapped leaf moves beyond the value bound '
    'from its oracle head) and backup (the mutant\'s swap map set per swap; (c) cannot see it); either caught by (f) when a parent or ancestor fails '
    'node_fails; (e) is not re-run per swap. Every survivor\'s |v1 - v2| and its parents\' backed-up value change are reported; a survivor with '
    '|v1 - v2| > 2 x the value bound and at least one affected parent priced is a MATERIAL misplacement (the rule\'s withdrawal case); survivors '
    'whose parents are all refused are counted apart (immaterial). With the per-child grid gate the sweeps are implementation checks (a material '
    'survivor with a priced parent is impossible by construction: its grid entry sits more than the bound from the oracle\'s child); interior '
    'reads (children that are backed-up values) are exercised by the single and systematic mutants at ply 2.., a ninth mutant (backup path: '
    'every turn-1 cell reading its own slots 0 and 1 swapped, interior children at depth >= 3) and read identity in every verdict; a tenth (backup '
    'path) hands every node\'s rule a uniform pi_own, which no value gate sees under learner-max and the exact prior check does. PASS iff the '
    'clean call passes and stays unchanged, every mutant is exercised and killed, and neither sweep has a material survivor')


class MutantPatchError(RuntimeError):
    """depthn_backups' source is absent or no longer carries the anchors the backup-path mutants patch (fail closed)."""


def backups_source():
    """depthn_backups' SOURCE text (its file is only read, never written)."""
    spec = importlib.util.find_spec('gpu_engine.depthn_backups');origin = getattr(spec, 'origin', None)
    if not isinstance(origin, str) or not Path(origin).is_file():raise MutantPatchError('gpu_engine.depthn_backups has no source file on this tree')
    return Path(origin).read_text()


def _swap_map(pairs):
    swap = {}
    for a, b in pairs:swap[tuple(a)] = (b[0], b[1], b[3], b[4], b[5]);swap[tuple(b)] = (a[0], a[1], a[3], a[4], a[5])
    return swap


def mutant_backups(pairs, *, source=None, name='depthn_backups_mutant'):
    """A separate module object executed from depthn_backups' SOURCE (backups_source(), or `source`) patched by BACKUP_PATCH: node_value's
    child reads at ply >= 2 rerouted through its _SWAP map (each pair (a, b) of children (q, w, ply, parent, own slot, opponent slot) at the
    same ply makes a's parent read b and b's parent read a: a leaf's status and value, or a node's whole subtree; set module._SWAP to change
    it), and the grid handed to the rule passed through module._AT_RULE (identity). MutantPatchError unless every anchor occurs exactly once."""
    text = backups_source() if source is None else source
    for anchor, replacement in BACKUP_PATCH:
        if text.count(anchor) != 1:raise MutantPatchError(f'depthn_backups source: anchor {anchor!r} occurs {text.count(anchor)} times, not once')
        text = text.replace(anchor, replacement)
    module = types.ModuleType(name);module._SWAP = _swap_map(pairs);module._AT_RULE = lambda grid, *where: grid;module._PRIORS = lambda pi, *where: pi
    exec(compile(text, f'<{name}: depthn_backups source patched in memory>', 'exec'), module.__dict__)
    return module


def scatter_frame(frame, pairs):
    """A copy of the frame with each pair's stored VALUES exchanged (leaf_value at the last ply, the child ply's node `value` otherwise);
    statuses, digests, priors and maps untouched: the misplacement a digest cannot certify."""
    depth = int(frame.depth);leaf = np.array(frame.leaf_value, copy=True);values = {}
    for a, b in pairs:
        ply = int(a[2])
        if ply + 1 == depth:x, y = (a[0], a[1], a[3], a[4], a[5]), (b[0], b[1], b[3], b[4], b[5]);target = leaf
        else:
            target = values.setdefault(ply + 1, np.array(frame.nodes[ply - 1].value, copy=True));k_opp = int(frame.k_opp[ply - 1]);K = int(frame.k_own[ply - 1]) * k_opp
            x, y = (a[0], a[1], a[3] * K + a[4] * k_opp + a[5]), (b[0], b[1], b[3] * K + b[4] * k_opp + b[5])
        target[x], target[y] = target[y], target[x]
    nodes = tuple(_copy(level, value=values[p]) if p in values else level for p, level in zip(range(2, depth), frame.nodes))
    return _copy(frame, leaf_value=leaf, nodes=nodes)


def _parents(frame):
    """(q, w, ply, node) of every expanded pricer node at plies 2..N-1 (children at ply >= 3), in (root, world, ply, node) order."""
    out = []
    for p, level in zip(range(2, int(frame.depth)), frame.nodes):out += [(q, w, p, n) for q, w, n in np.argwhere(np.asarray(level.status) == 2).tolist()]
    return sorted(out)


def _up(frame, key):
    """The parent (q, w, ply - 1, id) of a ply >= 2 node through the pricer's parent links ((a) checks them against the oracle's)."""
    q, w, ply, node = key;return (q, w, ply - 1, int(np.asarray(frame.nodes[ply - 2].parent)[node]))


def _ancestors(frame, key):
    """A node and every ancestor up to its turn-1 cell."""
    out = [tuple(key)]
    while out[-1][2] > 1:out.append(_up(frame, out[-1]))
    return out


def _slot_values(record):
    return {(i, x): record['grid'][k][m] for k, i in enumerate(record['own_slots']) for m, x in enumerate(record['opp_slots'])}


def _reverse(record, changes):
    """P's pricer-side rows with children replaced ({(own slot, opp slot): value}): (margin, new learner action, new top-two gap); margin =
    the best other row's q minus the clean choice's (> 0: the choice reverses)."""
    grid = [[changes.get((i, x), v) for x, v in zip(record['opp_slots'], row)] for i, row in zip(record['own_slots'], record['grid'])]
    solved = _solve(grid, record['pi_opp'], record['own_slots'], record['opp_slots'], record['own'], record['opp'])
    c = record['learner'];others = [x for k, x in enumerate(solved['q']) if k != c]
    return (max(others) - solved['q'][c] if others else -math.inf), solved['own'][solved['learner']], solved['gap']


def single_pairs(frame, backups, tree, bound):
    """The single variants' pairs (MUTATION_RULE): ({'sibling'|'cousin': (pair, why)}, reason when none)."""
    parents = _parents(frame);records = {}
    def record(key):
        if key not in records:records[key] = pricer_record(frame, backups, *key)
        return records[key]
    usable = lambda r: r is not None and 'shape' not in r and r['finite']
    chosen = None
    for key in parents:
        mine = record(key)
        if not usable(mine):continue
        theirs = tree.solve(*key)[1]
        if usable(theirs) and all(g is not None and not _within(g, bound) for g in (mine['gap'], theirs['gap'])):chosen = (key, mine, theirs);break
    if chosen is None:return {}, 'no parent at ply >= 2 whose clean learner choice is outside the near ties of both frames'
    key, mine, theirs = chosen;values = _slot_values(mine);out = {}
    def why(v1, v2, margin, new, gap, **more):
        d = abs(v1 - v2)
        return dict(more, parent=list(key), v1=v1, v2=v2, abs_value_difference=d, value_bound=bound, ratio_to_2_bound=None if not bound else d / (2 * bound),
            clean_learner=mine['own'][mine['learner']], clean_gap_pricer=mine['gap'], clean_gap_oracle=theirs['gap'], swapped_learner=new, reversal_margin=margin,
            swapped_gap_pricer=gap, choice_reversed=margin > 0, why=f'P\'s clean choice gaps {mine["gap"]:.3g} (pricer) / {theirs["gap"]:.3g} (oracle) exceed the value '
            f'bound {bound:.3g}; the swap moves values {d:.3g} apart and {"reverses" if margin > 0 else "does not reverse"} P\'s pricer-side choice '
            f'(margin {margin:.3g}, new top-two gap {gap if gap is None else round(gap, 6)})')
    best = None
    for s, t in itertools.combinations(sorted(values), 2):
        margin, new, gap = _reverse(mine, {s: values[t], t: values[s]})
        if best is None or margin > best[0]:best = (margin, s, t, new, gap)
    if best is not None:
        margin, s, t, new, gap = best
        out['sibling'] = ((key + s, key + t), why(values[s], values[t], margin, new, gap, c1=list(s), c2=list(t)))
    up = _up(frame, key);level = [k for k in parents if k != key and k[2] == key[2]];here = [k for k in level if k[:2] == key[:2]]
    same = lambda k: _up(frame, k) == up   # P' candidates in preference order (same root and world, same grandparent first), priced lazily
    order = [k for k in here if same(k) and k > key] + [k for k in here if same(k) and k < key] + [k for k in here if not same(k) and k > key] \
        + [k for k in here if not same(k) and k < key]
    other = next((k for k in order if usable(record(k))), None);fell_back = other is None
    if fell_back:other = next((k for k in [k for k in level if k[:2] > key[:2]] + [k for k in level if k[:2] < key[:2]] if usable(record(k))), None)
    if other is not None:
        theirs_values = _slot_values(record(other));best = None
        for s in sorted(values):
            for t in sorted(theirs_values):
                margin, new, gap = _reverse(mine, {s: theirs_values[t]})
                if best is None or margin > best[0]:best = (margin, s, t, new, gap)
        margin, s, t, new, gap = best
        out['cousin'] = ((key + s, other + t), why(values[s], theirs_values[t], margin, new, gap, parent2=list(other), same_root_world=other[:2] == key[:2],
            same_grandparent=other[:2] == key[:2] and same(other), fell_back_to_another_root_world=fell_back, c1=list(s), c2=list(t)))
    return out, None if out else 'P has no swappable pair'


def systematic_pairs(frame, kind):
    """sibling: (P, 0, x) <-> (P, 1, x) for every enumerated x at every parent with own slots 0 and 1; turn1: the same at every expanded
    turn-1 cell (its children are ply-2 nodes, interior at depth >= 3; backup path only); cousin: consecutive parents under each grandparent
    paired, (P0, 0, 0) <-> (P1, 0, 0) where both are enumerated."""
    parents = _parents(frame) if kind != 'turn1' else [(q, w, 1, a1 * 9 + r1) for q, w, a1, r1 in np.argwhere(np.asarray(frame.t1_status) == 2).tolist()];pairs = []
    live = lambda key, seat, slot: (lambda m: slot < len(m) and int(m[slot]) >= 0)(maps_at(frame, key[2], key[0], key[1], key[3])[seat])
    if kind in ('sibling', 'turn1'):
        for key in parents:
            if live(key, 0, 0) and live(key, 0, 1):
                opp = maps_at(frame, key[2], key[0], key[1], key[3])[1];pairs += [(key + (0, x), key + (1, x)) for x in range(len(opp)) if int(opp[x]) >= 0]
        return pairs
    groups = collections.defaultdict(list)
    for key in parents:groups[_up(frame, key)].append(key)
    for _, group in sorted(groups.items()):
        for p0, p1 in zip(group[0::2], group[1::2]):
            if all(live(k, seat, 0) for k in (p0, p1) for seat in (0, 1)):pairs.append((p0 + (0, 0), p1 + (0, 0)))
    return pairs


def at_bound(evaluation, value_bound):
    """The evaluation with its depth-1 floor replaced so tolerance_rule gives exactly `value_bound` (FLOOR_FACTOR x the floor): a call judged
    at the RUN's value bound rather than its own floor's; unchanged when value_bound is None."""
    if value_bound is None:return evaluation
    half = float(value_bound) / g2.FLOOR_FACTOR
    floor = dict(compared=1, exact_equal=0, nonfinite=0, max_abs_difference=half, largest=[], _deltas=np.asarray([half], np.float64))
    return dict(evaluation, c=dict(evaluation.get('c') or {}, depth1_noise_floor=floor))


def mutant_verdict(evaluation, *, lanes, sampled=False, value_bound=None, declared=None):
    """(verdict, summarized result) of ONE call's evaluation judged alone (at `value_bound`, the run's, when given; under `declared`,
    --declared-bounds, when given): status, the checks that caught it (a (c) quantity only beyond its value bound; guard-only failures listed
    apart, never a kill) and KILLED = FAIL with a gated catch."""
    result = summarize_run([at_bound(evaluation, value_bound)], lanes, declared=declared);status, reasons = gate_status_n(result, sampled=sampled);caught, guard_only = {}, []
    if value_bound is not None and result['tolerance']['value_bound'] != float(value_bound):
        raise ValueError(f'judged at {result["tolerance"]["value_bound"]!r}, not the run\'s value bound {value_bound!r}')
    depth = evaluation.get('depth')
    for name, check in result['checks'].items():
        if check['passed']:continue
        if name == 'c':
            failing = {q: m for q, m in check['quantities'].items() if q not in g2.FLOORS and not m['passed']}
            beyond = {q: dict(max_abs_difference=m['max_abs_difference'], bound=m['bound'], beyond_bound=m.get('beyond_bound'), guard_mode=m.get('guard_mode'))
                for q, m in failing.items() if not m['within_bound']}
            guard_only += [q for q, m in failing.items() if m['within_bound']]
            if beyond or not failing:caught['c'] = beyond or dict(reason='frame geometry')
        elif name == 'f':caught['f'] = {k: check.get(k) for k in ('grid_within_bound', 'max_abs_delta_child_value', 'nodes_grid_beyond_bound', 'row_q_within_bound',
            'max_abs_delta_row_q', 'non_exempt_mismatches', 'near_tie_flips', 'one_sided', 'problems', 'reasons')} | dict(first_mismatches=(check.get('first_mismatches') or [])[:3],
            first_grid_mismatches=(check.get('first_grid_mismatches') or [])[:3], read_identity={k: (check.get('read_identity') or {}).get(k) for k in ('passed',
            'mismatched_nodes', 'nodes', 'injective', 'interned_collisions', 'distinct_captured_tags', 'distinct_named_reads')},
            read_identity_first=((check.get('read_identity') or {}).get('first_mismatches') or [])[:2], priors={k: (check.get('priors') or {}).get(k) for k in ('passed',
            'mismatched_nodes', 'nodes', 'one_sided')}, priors_first=((check.get('priors') or {}).get('first_mismatches') or [])[:1])
        elif name == 'e':caught['e'] = {k: check.get(k) for k in ('argmax_q_differs_non_tie', 'argmax_target_differs_non_tie', 'status_differs', 'q_noise_within_bound',
            'max_abs_delta_q', 'reasons')}
        else:caught[name] = {k: v for k, v in (check.get('mismatched') or {}).items() if v} or check.get('reason') or 'failed'
    modes = {q: m.get('guard_mode') for q, m in (result['checks'].get('c') or {}).get('quantities', {}).items() if certifying_digest(q, depth)}
    return dict(status=status, status_reasons=reasons, killed=status == 'FAIL' and bool(caught), caught_by=sorted(caught), caught=caught, guard_only=guard_only,
        guard_modes_past_turn1=modes, value_bound=result['tolerance']['value_bound']), result


def mutant_evaluation(path, pairs, *, frame, data, requests, d_results, clean, backups, source, tree, full_turn1, context, priors=None):
    """One mutant's evaluation: backup path = the clean evaluation with (e) and (f) through mutant_backups(pairs) (priors: its _PRIORS hook);
    scatter path = every check on scatter_frame(frame, pairs) with the clean backups. Evaluator actors 1 (GUARD_DEPTHN_RULE certifies where
    digests match)."""
    decision = 'e' in clean
    if path == 'backup':
        evaluation = {k: v for k, v in clean.items() if k not in ('e', 'f', 'mutation', MUTATION_INPUTS)};module = mutant_backups(pairs, source=source)
        if priors is not None:module._PRIORS = priors
        backup_checks(evaluation, frame, data, requests, context=context, backups=module, decision=decision, tree=tree)
    else:evaluation = evaluate_n(scatter_frame(frame, pairs), d_results, requests, data, context=context, full_turn1=full_turn1, backups=backups,
        decision=decision, choices=True, tree=tree)
    evaluation['actors'] = 1
    return evaluation


def _uniform_pairs(groups, rng, count):
    """(up to `count` distinct pairs (a, b), a before b, uniform without replacement over every pair within `groups`, population)."""
    groups = [g for g in groups if len(g) > 1];sizes = np.asarray([len(g) * (len(g) - 1) // 2 for g in groups], np.int64);total = int(sizes.sum())
    picks = np.arange(total) if total <= count else np.sort(rng.choice(total, size=count, replace=False));starts = np.cumsum(sizes) - sizes;out = []
    for pick in picks.tolist():
        g = int(np.searchsorted(starts, pick, side='right')) - 1;k = pick - int(starts[g]);n = len(groups[g]);i = 0
        while k >= n - 1 - i:k -= n - 1 - i;i += 1
        out.append((groups[g][i], groups[g][i + 1 + k]))
    return out, total


def _cousin_pairs(worlds, parent, rng, count):
    """(up to `count` distinct pairs uniform without replacement over every pair with DIFFERENT parents inside one (root, world), population):
    a world drawn by its pair count, then two members by rejection (same parent redrawn)."""
    worlds = [g for g in worlds if len(g) > 1];sizes = []
    for g in worlds:
        same = collections.Counter(parent[i] for i in g);sizes.append(len(g) * (len(g) - 1) // 2 - sum(n * (n - 1) // 2 for n in same.values()))
    sizes = np.asarray(sizes, np.int64);total = int(sizes.sum())
    if total <= 4 * count:   # small populations are enumerated (the rejection loop below stays fast: at most a quarter of the pairs is drawn)
        every = [(a, b) for g in worlds for a, b in itertools.combinations(g, 2) if parent[a] != parent[b]]
        return (every if total <= count else [every[i] for i in np.sort(rng.choice(total, size=count, replace=False)).tolist()]), total
    chosen, weights = set(), sizes / total
    while len(chosen) < count:
        g = worlds[int(rng.choice(len(worlds), p=weights))];a, b = sorted(rng.choice(len(g), size=2, replace=False).tolist());a, b = g[a], g[b]
        if parent[a] != parent[b]:chosen.add((a, b))
    return sorted(chosen), total


def swap_sweep(frame, data, backups, tree, bound, *, count=SWEEP_SWAPS, seed=MUTATION_SEED, block=0, source=None, paths=SWEEP_PATHS):
    """MUTATION_RULE's sweep on one call: single sibling and cousin swaps of (c)-scored leaves (count: per kind, or {kind: count}), drawn once
    and applied on each of `paths`: scatter (a working copy of the leaves, restored after every swap; (c) at the two swapped positions) and
    backup (mutant_backups' swap map set per swap, from `source`); (f) (node_fails) at the parents and their ancestors on both. Returns
    {'pairs': population and drawn counts, path: its accounting}."""
    clock = time.perf_counter();leaves = data['leaves'];out = dict(pairs=dict(seed=[seed, block], population={}, drawn={}, value_bound=bound))
    if leaves is None or not len(leaves['specs']):return dict(out, **{p: dict(skipped='no checked leaf', material_survivors=0) for p in paths})
    index = np.asarray(leaves['index'], np.int64);at = tuple(index.T);values = np.asarray(frame.leaf_value, np.float32)[at].astype(np.float64)
    heads = np.asarray(leaves['head_value'], np.float32).astype(np.float64)
    scored = (np.asarray(leaves['status']) == 4) & (np.asarray(frame.leaf_status)[at] == 4) & np.isfinite(values) & np.isfinite(heads)
    cand = np.flatnonzero(scored).tolist();rng = np.random.default_rng([seed, block]);parent = {i: tuple(index[i, :3].tolist()) for i in cand}
    by_parent, by_world = collections.defaultdict(list), collections.defaultdict(list)
    for i in cand:by_parent[parent[i]].append(i);by_world[parent[i][:2]].append(i)
    limit = lambda kind: int(count.get(kind, 0)) if isinstance(count, dict) else int(count)
    drawn = dict(zip(('sibling', 'cousin'), (_uniform_pairs([by_parent[k] for k in sorted(by_parent)], rng, limit('sibling')),
        _cousin_pairs([by_world[k] for k in sorted(by_world)], parent, rng, limit('cousin')))))
    out['pairs'].update(population={k: v[1] for k, v in drawn.items()}, drawn={k: len(v[0]) for k, v in drawn.items()}, seconds=time.perf_counter() - clock)
    ply = int(frame.depth) - 1;clean = {}
    def clean_record(key):
        if key not in clean:clean[key] = pricer_record(frame, backups, *key)
        return clean[key]
    for path in paths:
        clock = time.perf_counter();counts = collections.Counter();rows = [];worst = dict(abs_value_difference=0., parent_value_change=0.);refused = 0
        if path == 'scatter':work = np.array(frame.leaf_value, copy=True);judged, reader = _copy(frame, leaf_value=work), backups
        else:judged, reader = frame, mutant_backups([], source=source)
        for kind, (pairs, _) in drawn.items():
            for a, b in pairs:
                x, y = tuple(index[a].tolist()), tuple(index[b].tolist());v1, v2 = values[a], values[b];d = abs(v1 - v2)
                by_c = path == 'scatter' and (abs(v2 - heads[a]) > bound or abs(v1 - heads[b]) > bound)   # (c)'s own rule; a backup misread leaves the frame clean
                if path == 'scatter':work[x], work[y] = work[y], work[x]
                else:reader._SWAP = _swap_map([((x[0], x[1], ply, x[2], x[3], x[4]), (y[0], y[1], ply, y[2], y[3], y[4]))])
                tops = list(dict.fromkeys([(x[0], x[1], ply, x[2]), (y[0], y[1], ply, y[2])]))
                affected = sorted(set(_ancestors(frame, tops[0])) | set(_ancestors(frame, tops[-1])));records = {key: pricer_record(judged, reader, *key) for key in affected}
                by_f = any(node_fails(records[key], tree.solve(*key)[1], bound) for key in affected if tree.solve(*key)[0] is not MISSING)
                if path == 'scatter':work[x], work[y] = work[y], work[x]
                else:reader._SWAP = {}
                counts[kind + (':c' if by_c else '') + (':f' if by_f else '')] += 1
                if by_c or by_f:continue
                change = [None if clean_record(key) is None or records[key] is None else records[key]['value'] - clean_record(key)['value'] for key in tops]
                priced = any(clean_record(key) is not None for key in tops);refused += not priced
                if priced:worst['abs_value_difference'] = max(worst['abs_value_difference'], d)
                worst['parent_value_change'] = max([worst['parent_value_change']] + [abs(c) for c in change if c is not None])
                rows.append(dict(kind=kind, a=list(x), b=list(y), abs_value_difference=d, parent_value_change=change, parent_priced=priced, material=priced and d > 2 * bound))
        material = [r for r in rows if r['material']]
        out[path] = dict(swaps={kind: len(pairs) for kind, (pairs, _) in drawn.items()}, caught={k: v for k, v in sorted(counts.items()) if ':' in k},
            survived={k: v for k, v in sorted(counts.items()) if ':' not in k}, survivors=len(rows), survivors_parents_refused=refused,
            survivor_max_abs_value_difference=worst['abs_value_difference'], survivor_max_parent_value_change=worst['parent_value_change'],
            survivor_ratio_to_2_bound=None if not bound else worst['abs_value_difference'] / (2 * bound), material_survivors=len(material),
            first_material_survivors=material[:FIRST], claim_holds=not material, survivor_rows=rows, seconds=time.perf_counter() - clock,
            e='not re-run per swap: (e) needs a full learner_max_root per root; (c) and (f) decide each swap',
            claim='every survivor with a priced parent is within noise: |v1 - v2| <= 2 x value bound (the survivor_max_abs_value_difference over them) '
                'and no gated (f) failure at its parents or ancestors; survivors whose parents are all refused are immaterial')
    return out


MUTATION_INPUTS = '_mutation_inputs'   # run_call_n keeps a call's frame, oracle data and D3 results here until mutation_tests runs


def mutation_test_call(frame, data, requests, d_results, clean, *, context=None, backups=None, source=None, lanes=(2187,), full_turn1=True,
                       sweep=SWEEP_SWAPS, seed=MUTATION_SEED, progress=None, value_bound=None, declared=None):
    """MUTATION_RULE on ONE call: the clean call and the ten mutants judged alone at `value_bound` (the RUN's; None: the call's own), then
    the sweep on both paths; `clean` is the call's evaluation (never modified). backups: depthn_backups (default) for the clean side and the
    scatter path; source: depthn_backups' source text for the backup path (backups_source() by default); declared: the run's --declared-bounds
    (every verdict under it; the run's value bound is then the declared one). Returns the call's report."""
    clock = time.perf_counter();context = dict(context or {});log_line = progress or (lambda *a, **k: None)
    out = dict(context, rule='MUTATION_RULE (gate.json tolerance and this file\'s rule)', mutants={}, passed=False)
    if backups is None:
        try:backups = importlib.import_module('gpu_engine.depthn_backups')
        except ImportError as error:return dict(out, reason=f'gpu_engine.depthn_backups is not importable: {error}', seconds=time.perf_counter() - clock)
    base = {k: v for k, v in clean.items() if k not in ('mutation', MUTATION_INPUTS)};base['actors'] = 1
    verdict, result = mutant_verdict(base, lanes=lanes, value_bound=value_bound, declared=declared);bound = result['tolerance']['value_bound'];block = int(context.get('block', 0))
    out.update(clean=dict(status=verdict['status'], status_reasons=verdict['status_reasons'], failing=verdict['caught_by'], guard_only=verdict['guard_only']),
        value_bound=bound, value_bound_source=('the run\'s declared value_bound (--declared-bounds)' if declared is not None else 'the run\'s (its depth-1 floor '
            'substituted for the call\'s)' if value_bound is not None else 'the call\'s own depth-1 floor'),
        evaluator_actors_live=clean.get('actors'))
    if int(frame.depth) < 3:return dict(out, reason='depth < 3: no parent at ply >= 2', seconds=time.perf_counter() - clock)
    tree = OracleTree(frame, data);singles, none = single_pairs(frame, backups, tree, bound)
    for path, kind, variant in MUTANTS:
        name = f'{path}-{kind}-{variant}';started = time.perf_counter()
        pairs, pair = ([], None) if kind == 'priors' else ([singles[kind][0]], singles[kind][1]) if variant == 'single' and kind in singles \
            else (systematic_pairs(frame, kind), None) if variant == 'systematic' else ([], None)   # the priors mutant swaps no child
        entry = dict(path=path, kind=kind, variant=variant, pairs=len(pairs), pair=pair if kind != 'priors' else dict(priors='pi_own uniform over the enumerated own slots at every node'))
        if not pairs and kind != 'priors':out['mutants'][name] = dict(entry, exercised=False, killed=False, reason=(none or f'no {kind} pair at P') if variant == 'single' else 'no pair to swap');continue
        try:evaluation = mutant_evaluation(path, pairs, frame=frame, data=data, requests=requests, d_results=d_results, clean=base, backups=backups,
            source=source, tree=tree, full_turn1=full_turn1, context=dict(context, mutant=name), priors=UNIFORM_PI_OWN if kind == 'priors' else None)
        except MutantPatchError as error:out['mutants'][name] = dict(entry, exercised=False, killed=False, reason=str(error));continue
        verdict, _ = mutant_verdict(evaluation, lanes=lanes, value_bound=bound, declared=declared)
        out['mutants'][name] = dict(entry, exercised=True, **verdict, seconds=time.perf_counter() - started)
        log_line('MUTATION_TEST_MUTANT', **context, mutant=name, status=verdict['status'], killed=verdict['killed'], caught_by=verdict['caught_by'],
            guard_only=verdict['guard_only'], pairs=len(pairs), value_bound=bound, seconds=out['mutants'][name]['seconds'])
    try:out['sweep'] = swap_sweep(frame, data, backups, tree, bound, count=sweep, seed=seed, block=block, source=source)
    except MutantPatchError as error:out['sweep'] = dict(skipped=str(error))
    for path in SWEEP_PATHS:
        s = out['sweep'].get(path) or {}
        log_line('MUTATION_TEST_SWEEP', **context, path=path, **{k: s.get(k) for k in ('swaps', 'caught', 'survived', 'survivors', 'survivors_parents_refused',
            'material_survivors', 'survivor_max_abs_value_difference', 'survivor_ratio_to_2_bound', 'survivor_max_parent_value_change', 'claim_holds', 'seconds', 'skipped')})
    again, _ = mutant_verdict(base, lanes=lanes, value_bound=value_bound, declared=declared);out['clean_unchanged'] = again['status'] == out['clean']['status'] and again['caught_by'] == out['clean']['failing']
    survivors = [n for n, m in out['mutants'].items() if not m['killed']];material = {p: (out['sweep'].get(p) or {}).get('material_survivors') for p in SWEEP_PATHS}
    out.update(survivors=survivors, material_survivors=material, passed=out['clean']['status'] == 'PASS' and out['clean_unchanged'] and not survivors
        and all(n == 0 for n in material.values()) and not any('skipped' in (out['sweep'].get(p) or {'skipped': 1}) for p in SWEEP_PATHS), seconds=time.perf_counter() - clock)
    log_line('MUTATION_TEST', **context, passed=out['passed'], clean=out['clean']['status'], value_bound=bound, killed={n: m['killed'] for n, m in out['mutants'].items()},
        caught_by={n: m.get('caught_by') for n, m in out['mutants'].items()}, survivors=survivors, material_survivors=material, seconds=out['seconds'])
    return out


def mutation_tests(evaluations, *, value_bound, lanes=(2187,), backups=None, source=None, progress=None, sweep=SWEEP_SWAPS, declared=None):
    """--mutation-test once the run's value bound is known: mutation_test_call on every call that kept its inputs (run_call_n with
    mutation=...), at the RUN's value bound (under the run's --declared-bounds, when given); the inputs are dropped from each evaluation
    afterwards. Returns the per-call reports."""
    out = []
    for evaluation in evaluations:
        inputs = evaluation.pop(MUTATION_INPUTS, None)
        if inputs is None:continue
        out.append(mutation_test_call(inputs['frame'], inputs['data'], inputs['requests'], inputs['d_results'], evaluation, context=inputs['context'],
            backups=backups, source=inputs.get('source', source), lanes=lanes, full_turn1=inputs['full_turn1'], progress=progress, value_bound=value_bound,
            sweep=inputs.get('sweep', sweep), declared=declared))
    return out


def mutation_report(calls, *, seconds=None, source=None, checks=None, value_bound=None, declared=None):
    """mutation-test.json: per-call reports, PASS iff every call's passes (and at least one ran); header: the gate's (checks: the run's;
    declared: its declared_bounds section)."""
    passed = bool(calls) and all(c['passed'] for c in calls)
    return dict(schema=SCHEMA, mode='MUTATION_TEST', header=header_block(checks, declared=declared), passed=passed, status='PASS' if passed else 'FAIL', rule=MUTATION_RULE,
        choice_rule=CHOICE_RULE, mutants=[f'{p}-{k}-{v}' for p, k, v in MUTANTS], sweep_swaps=SWEEP_SWAPS, sweep_paths=list(SWEEP_PATHS), seed=MUTATION_SEED,
        source=source, value_bound=value_bound, survivors={f'block{c.get("block")}-call{c.get("call")}': c.get('survivors') for c in calls},
        material_survivors={f'block{c.get("block")}-call{c.get("call")}': c.get('material_survivors') for c in calls}, seconds=seconds, calls=calls)


ORACLE_ROOT_FIELDS = g2.ROOT_FIELDS


def save_oracle_data(path, data, d_results, requests, *, meta=None):
    """--save-oracle: one call's oracle data (what evaluate_n, (e) and (f) read: roots, turn-1 children and cells, every ply's records, ids,
    links and maps, the checked leaves and the enumeration; never the leaves' logits), D3's outputs (g2._d_arrays) and the requests' root
    input keys ((b) maps D's input-differs reasons through them; save_frame keeps the rest of the requests), as .npz."""
    arrays = {};keys = sorted(data['roots']);arrays['roots.keys'] = np.asarray(keys, np.int64).reshape(-1, 2)
    arrays['request_input_keys'] = np.asarray([[k in r.root_inputs for k in g2.INPUT_ORDER] for r in requests], bool).reshape(-1, len(g2.INPUT_ORDER))
    for name in ORACLE_ROOT_FIELDS:
        values = [np.asarray(data['roots'][k][name]) for k in keys if name in data['roots'][k]]
        if values and len(values) == len(keys):arrays['roots.' + name] = np.stack(values)
    parts = [('children', data['children']), ('leaves', data['leaves'] or {})] + [(f'levels.{p}', r) for p, r in data['levels'].items() if p >= 2]
    for part, records in parts:
        for name, value in records.items():
            if name != 'logits' and np.asarray(value).dtype.kind in 'biuf':arrays[f'{part}.{name}'] = np.asarray(value)
    arrays['cells'] = np.asarray(data['cells'], np.int64).reshape(-1, 4)
    for p, ids in data['ids'].items():arrays[f'ids.{p}'] = np.asarray(ids, np.int64)
    for p, link in data['links'].items():arrays[f'links.{p}'] = np.stack([np.asarray(x, np.int64) for x in link])
    by_ply = collections.defaultdict(list)
    for key, (own, opp) in sorted(data['maps'].items()):by_ply[key[0]].append((key[1:], own, opp))
    for p, items in by_ply.items():
        arrays[f'maps.{p}.keys'] = np.asarray([k for k, _, _ in items], np.int64);arrays[f'maps.{p}.own'] = np.stack([np.asarray(o, np.int64) for _, o, _ in items])
        arrays[f'maps.{p}.opp'] = np.stack([np.asarray(o, np.int64) for _, _, o in items])
    for p, value in data['enumerated'].items():arrays[f'enumerated.{p}'] = np.asarray(value, np.int64)
    arrays.update(g2._d_arrays(d_results))
    arrays['meta'] = np.asarray(json.dumps(dict(meta or {}, plan=data['plan'], worlds=data['worlds'], forced_leaves=data['forced_leaves'], schema=SCHEMA), default=str))
    np.savez_compressed(path, **arrays)


def load_oracle_data(path):
    """(data, d_results, meta) from save_oracle_data, in oracle_levels' layout; meta['input_keys']: each request's root input keys."""
    with np.load(path, allow_pickle=False) as saved:arrays = {k: saved[k] for k in saved.files}
    meta = json.loads(str(arrays.pop('meta')));plan = dict(meta['plan']);plan.update(k_own=tuple(plan['k_own']), k_opp=tuple(plan['k_opp']))
    meta['input_keys'] = [[k for k, use in zip(g2.INPUT_ORDER, row) if use] for row in arrays.pop('request_input_keys').tolist()]
    group = lambda prefix: {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}
    roots = {}
    for i, (q, w) in enumerate(arrays['roots.keys'].tolist()):
        root = {name: arrays['roots.' + name][i] for name in ORACLE_ROOT_FIELDS if 'roots.' + name in arrays}
        for name in ('own_legal', 'peer_legal'):root[name] = tuple(map(bool, root[name]))
        for name in ('pre', 'own_error', 'peer_error'):root[name] = int(root[name])
        for name in ('peer_requested', 'peer_usable'):root[name] = bool(root[name])
        roots[(q, w)] = root
    levels = {1: group('children.')}
    for name in {k.split('.')[1] for k in arrays if k.startswith('levels.')}:levels[int(name)] = group(f'levels.{name}.')
    maps = {}
    for p in {int(k.split('.')[1]) for k in arrays if k.startswith('maps.')}:
        for key, own, opp in zip(arrays[f'maps.{p}.keys'].tolist(), arrays[f'maps.{p}.own'], arrays[f'maps.{p}.opp']):maps[(p,) + tuple(key)] = (own, opp)
    leaves = group('leaves.') or None
    enumerated = {(k if k == 'leaf' else int(k)): v for k, v in group('enumerated.').items()}
    data = dict(roots=roots, children=levels[1], cells=arrays['cells'], levels=levels, ids={int(k): v for k, v in group('ids.').items()},
        links={int(k): tuple(v) for k, v in group('links.').items()}, maps=maps, leaves=leaves, enumerated=enumerated, worlds=meta['worlds'], plan=plan,
        forced_leaves=meta['forced_leaves'], topk={})
    return data, g2._d_results(arrays), meta


ORACLE_FILE = re.compile(r'^(?P<prefix>frames-block(?P<block>\d+)-call(?P<call>\d+))-oracle\.npz$')


def offline_mutation(directories, *, backups=None, source=None, progress=None, sweep=SWEEP_SWAPS, lanes=(2187,)):
    """--mutation-from: the clean checks ((a)-(f) through evaluate_n) on every saved call (frames-block{B}-call{C}-oracle.npz beside its depth-N
    frame from --save-arrays --save-oracle), the run's value bound and checks from them all (summarize_run), then mutation_test_call on each
    at that bound. Returns dict(calls, checks, value_bound)."""
    evaluations = []
    for directory in directories:
        for path in sorted(Path(directory).glob('frames-block*-call*-oracle.npz')):
            m = ORACLE_FILE.match(path.name)
            if m is None:continue
            data, d_results, meta = load_oracle_data(path);frame, requests, frame_meta = load_frame(path.with_name(meta['frame']))
            if (frame_meta.get('evaluator') or {}).get('declared_bounds_sha256'):
                raise SystemExit(f'--mutation-from: {path.name} comes from a --declared-bounds run, judged under its declaration; offline it would be judged '
                    'under the frozen rule (refused: re-run --mutation-test live with the declaration)')
            for request, keys in zip(requests, meta['input_keys']):request.root_inputs = {k: None for k in keys}
            context = dict(block=int(m['block']), call=int(m['call']), source=str(path))
            clean = evaluate_n(frame, d_results, requests, data, context=context, full_turn1=meta.get('full_turn1', True), backups=backups, decision=True, choices=True)
            clean.update(actors=meta.get('actors'), **{MUTATION_INPUTS: dict(frame=frame, data=data, d_results=d_results, requests=requests, context=context,
                full_turn1=meta.get('full_turn1', True))})
            evaluations.append(clean)
    if not evaluations:return dict(calls=[], checks={}, value_bound=None)
    result = summarize_run([{k: v for k, v in e.items() if k != MUTATION_INPUTS} for e in evaluations], lanes);bound = result['tolerance']['value_bound']
    calls = mutation_tests(evaluations, value_bound=bound, lanes=lanes, backups=backups, source=source, progress=progress, sweep=sweep)
    return dict(calls=calls, checks=result['checks'], value_bound=bound)


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


def n2_equivalence_call(*, requests, recipe, materializer, params, configs, frame_pricer, oracle, context, progress=None, backups2=None, module=None,
                        stack_rows=None, merged_decision=False):
    """One request list under --n2-equivalence: configs [(name, DepthNPricer N=2, Depth2Pricer)], the same requests to both,
    D3 and the oracle's roots and turn-1 children once (the run's depth-1 noise floor and turn-1 heads). Per config:
    n2_compare, (e) on both surfaces (depth2_backups.learner_max_root) and the guard's spreads. stack_rows: as run_call_n's. A config
    (name, DepthNPricer, Depth2Pricer, [other Depth2Pricers]) also prices the others first on the same requests (CHILD_BLOCK_INVARIANCE:
    other child blocks) and records g2.block_invariance of each against the compared Depth2Pricer's frame. merged_decision (run_n2: the
    champion path): MERGED_DEPTH2_DECISION per config (merged_depth2_decision) as e_merged; False changes nothing."""
    from gpu_engine import device_public_teacher_batch as batch
    reference = importlib.import_module('gpu_engine.depthn_oracle');module = module or pricer_module()
    backups2 = backups2 or importlib.import_module('gpu_engine.depth2_backups');requests = tuple(requests);out = {}
    with stage(progress, 'frame_pricer', **context):d_results = frame_pricer.price(requests, materializer, recipe, params)
    with stage(progress, 'oracle_turn1', **context):
        stack = reference.stack(requests, batch._numpy_worlds(requests, materializer, recipe), size=stack_rows);roots = oracle.roots(stack, requests, params)
        children = oracle.children(stack, reference.turn1_cells(requests, recipe, roots), params)
    turn1 = dict(roots=roots, children=children, leaves=dict(specs=np.zeros((0, 6), np.int64)))
    for name, depthn, depth2, *rest in configs:
        where = dict(context, config=name);others = rest[0] if rest else ();other_frames = []
        with stage(progress, 'depthn_pricer_n2', **where) as info:
            clock = time.perf_counter();raw = depthn.price(requests, materializer, recipe, params);mine = module.as_depth2_frame(raw)
            info.update(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(raw, 'timing', None)))
        for i, other in enumerate(others):   # as the merged gate: the other child blocks first, then the compared frame
            with stage(progress, f'depth2_pricer_invariance_{i}', child_block=getattr(other, 'child_block', None), **where) as info:
                clock = time.perf_counter();other_frames.append(other.price(requests, materializer, recipe, params))
                info.update(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(other_frames[-1], 'timing', None)))
        with stage(progress, 'depth2_pricer', **where) as info:
            clock = time.perf_counter();theirs = depth2.price(requests, materializer, recipe, params)
            info.update(wall_seconds=time.perf_counter() - clock, **g2.pricer_costs(getattr(theirs, 'timing', None)))
        found, floats = n2_compare(theirs, mine, where)
        c = g2.check_heads(mine, turn1, d_results)   # depth-N turn 1 against the oracle, and the run's depth-1 floor
        e = decide_roots(theirs, mine, requests, backups2, context=where)
        out[name] = dict(a=found, floats=floats, c=c, e=e, spreads=node_spreads(raw))
        if merged_decision:
            with stage(progress, 'check_e_merged_depth2', **where) as info:
                out[name]['e_merged'] = merged_depth2_decision(theirs, depth2, requests=requests, recipe=recipe, oracle=oracle, stack=stack, params=params, roots=roots,
                    children=children, backups=backups2, context=where)
                info.update({k: out[name]['e_merged'].get(k) for k in ('passed', 'roots_compared', 'argmax_q_differs', 'cell_choice_differs')})
        if others:
            with stage(progress, 'check_child_block_invariance', **where) as info:
                out[name]['invariance'] = [g2.block_invariance(o, theirs, context=where) for o in other_frames]
                info['discrete_equal'] = all(r['passed'] for r in out[name]['invariance'])
    return out


def summarize_n2(per_config, *, declared=None):
    """{config: [call results]} -> per config: discrete (exact), floats within the run's rule, (e) under the near-tie rule;
    the rule from the depth-1 floor (D3 vs the oracle's turn-1 heads) and the p10 spread of the N=2 frames. declared: the run's
    --declared-bounds (declared_rule: its value and turn-1 logit bounds, and its near-tie Q bound as (e)'s q_noise_bound). Calls that
    carry child-block invariance (CHILD_BLOCK_INVARIANCE) add a gated block_invariance section (g2.merge_invariance under the same rule), and
    calls that carry MERGED_DEPTH2_DECISION a gated e_merged_depth2 section (g2.merge_decisions under the same q_noise_bound)."""
    report = {}
    for name, calls in per_config.items():
        quantities = g2.merge_numeric([x['c'] for x in calls]);p10, spread = guard_spread(s for x in calls for s in x['spreads'])
        rule = guard_note(declared_rule(g2.tolerance_rule(quantities.get('depth1_noise_floor'), p10), declared), spread)
        floats = g2.merge_numeric([x['floats'] for x in calls]);q_noise = rule.get('q_noise_bound', rule['value_bound'])   # tolerance_rule sets none
        for q, m in floats.items():g2.finalize_numeric(m, g2.bound_for(q, rule), g2.guard_for(q, rule))
        heads = {q: g2.finalize_numeric(m, g2.bound_for(q, rule), None if q in g2.FLOORS else g2.guard_for(q, rule)) for q, m in quantities.items()}
        discrete = g2.merge_findings([x['a'] for x in calls]);e = g2.merge_decisions([x['e'] for x in calls], q_noise_bound=q_noise)
        e['reference'] = 'depth2_backups.learner_max_root on the Depth2Frame vs on as_depth2_frame(DepthNPricer N=2)'
        reports_equal = all(x['a'].get('world_reports_equal', False) for x in calls)
        invariance = None if not any('invariance' in x for x in calls) else g2.merge_invariance([r for x in calls for r in x.get('invariance', ())], rule=rule)
        merged = None if not any('e_merged' in x for x in calls) else dict(g2.merge_decisions([x['e_merged'] for x in calls], q_noise_bound=q_noise), rule=MERGED_DEPTH2_DECISION)
        passed = discrete['passed'] and all(m['passed'] for m in floats.values()) and e['passed'] and reports_equal and (invariance is None or invariance['passed']) \
            and (merged is None or merged['passed'])
        vacuous = [reason for reason, empty in (('no turn-2 leaf compared', not discrete['compared'].get('t2_status')),
            ('(e) compared no root', not e.get('roots_compared')), ('(e) on the merged Depth2Pricer compared no root', merged is not None and not merged.get('roots_compared')))
            if empty]
        report[name] = dict(status='FAIL' if not passed else 'INCOMPLETE' if vacuous else 'PASS', status_reasons=vacuous, passed=passed and not vacuous,
            discrete=discrete, floats=floats, e=e, tolerance=dict(rule, q_noise_bound=q_noise),
            turn1_heads_vs_oracle=dict(reported_not_gated=True, quantities=heads), world_reports_equal=reports_equal,
            world_reports_first_difference=next((x['a']['world_reports_first_difference'] for x in calls if x['a'].get('world_reports_first_difference')), None),
            **({} if invariance is None else dict(block_invariance=invariance)), **({} if merged is None else dict(e_merged_depth2=merged)))
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
    """Requests (saved roots or a bank chunk), the evaluator (head override, champion 9375 or the bank snapshot's actor) and
    --max-roots. `forward` is the champion evaluator (None for V2: the pricers' default forward)."""
    loaded = (g2.saved_root_requests if args.saved_roots else g2.bank_chunk_requests)(args)
    head = g2.load_head(args, plan=loaded.plan, labels=loaded.labels, category_sha256=loaded.category_sha256) if args.head_archive else None
    champion = g2.load_champion(args, loaded) if getattr(args, 'champion_archive', None) else None
    params = (head[0],) if head is not None else (champion.params,) if champion is not None else loaded.params
    groups, root_values, dropped, arm_checked = g2.bind_roots(loaded.groups, params, loaded.plan, head_config=None if head is None else head[1],
        evaluator=champion)
    evaluator = dict(g2.evaluator_header(args, loaded, head, champion=champion), arm_checked=arm_checked, dropped_roots=dropped)
    log('EVALUATOR_RESOLVED', evaluator=evaluator, root_value_scale=g2.percentiles(root_values))
    if args.max_roots is not None:
        kept, left = [], args.max_roots
        for group in groups:
            if left > 0:kept.append(group[:left]);left -= len(kept[-1])
        groups = kept
    return SimpleNamespace(loaded=loaded, head=head, params=params, groups=groups, root_values=root_values, evaluator=evaluator, forward=champion,
        actor_capacity=len(params) if head is not None or champion is not None else loaded.plan.historical_window + 1)


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
    parser.add_argument('--out', help='new output directory (every mode but --reverdict)')
    parser.add_argument('--reverdict', metavar='GATE_JSON', help='offline: re-apply the current verdict rules to a saved gate.json; writes '
        'gate-reverdict.json beside it (never overwritten); exit 0 iff the new status is PASS/PASS_SAMPLED')
    parser.add_argument('--depth', type=int, default=3, help='search depth N >= 2 (default 3)')
    parser.add_argument('--selection', choices=MODES, default='prior', help='ply-2 selection (plies 3..N always prior)')
    parser.add_argument('--k-own', default='3', help='learner width per ply 2..N, "k" or "k2,k3,..." (under searched, ply 2\'s is searched_maps\' k)')
    parser.add_argument('--k-opp', default='3', help='opponent width per ply 2..N (as --k-own)')
    parser.add_argument('--k1-own', type=int, default=9, help='turn-1 learner width (9 = full)')
    parser.add_argument('--k1-opp', type=int, default=9, help='turn-1 opponent width (9 = full)')
    parser.add_argument('--capacity', type=int, default=512, help='DepthNPricer root capacity (calls are split at the smallest pricer capacity)')
    parser.add_argument('--block-lanes', type=int, default=2187, help='DepthNPricer block_lanes')
    parser.add_argument('--block-invariance', action='store_true', help=f'price every call with block_lanes {INVARIANCE_LANES} (the last is checked); with '
        f'--n2-equivalence (champion path only): each Depth2Pricer with child_block {g2.INVARIANCE_BLOCKS} (CHILD_BLOCK_INVARIANCE)')
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
    parser.add_argument('--mutation-test', action='store_true', help='once every call is priced, at the run\'s value bound: the ten mutants (eight '
        'misplacements, a turn-1 misread, wrong priors) and the swap sweep on both paths (MUTATION_RULE; each call\'s frame and oracle data are '
        'held until then); writes mutation-test.json; exit non-zero if the clean run fails, a mutant survives or a sweep has a material survivor')
    parser.add_argument('--save-oracle', action='store_true', help='also write each call\'s oracle data and D3 outputs beside its frames '
        '(frames-block{B}-call{C}-oracle.npz; implies --save-arrays) for --mutation-from')
    parser.add_argument('--mutation-from', nargs='+', metavar='DIR', help='offline: the mutation test from saved frames and oracle data '
        '(--save-arrays --save-oracle); writes mutation-test.json in --out')
    parser.add_argument('--fixed-shapes', action='store_true', help='compile once per process: every DepthNPricer with fixed_capacity=True and '
        'every oracle stack padded to 2 x the call-split capacity (FIXED_SHAPES; a harness setting, recorded as `harness`; no rule reads it)')
    parser.add_argument('--declared-bounds', metavar='PATH', help='opt-in, champion only (with --champion-archive): a JSON declaration '
        f'({DECLARED_SCHEMA}) of this evaluator\'s value, turn-1 logit and deeper-ply logit bounds, written before the run; it replaces the '
        'tolerance numbers (DECLARED_RULE) and is recorded in every report\'s header (declared_bounds)')
    parser.add_argument('--declared-bounds-sha256', metavar='SHA256', help='sha256 of --declared-bounds\' bytes (required with it; with --reverdict: must be '
        'the report\'s recorded one)')
    parser.add_argument('--declared-bounds-commit', metavar='COMMIT', help=f'the commit that added --declared-bounds under {DECLARED_SUBDIR}/ through a reviewed '
        'PR (required with it; recorded with the file\'s git blob sha1; with --reverdict: must be the report\'s recorded one)')
    g2.add_head_arguments(parser);ce.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.reverdict:
        record = ('declared_bounds_commit', 'declared_bounds_sha256')   # the pre-registration record a declared report is compared with
        given = sorted(a.option_strings[0] for a in parser._actions if a.dest not in ('help', 'reverdict') + record and getattr(args, a.dest) != a.default)
        if given:parser.error('--reverdict runs offline on a saved gate.json and takes no other option but --declared-bounds-commit / '
            '--declared-bounds-sha256 (given: ' + ', '.join(given) + ')')
        if args.declared_bounds_sha256 is not None and not ce._hex64(args.declared_bounds_sha256):parser.error('--declared-bounds-sha256 must be a 64-hex sha256')
        if args.declared_bounds_commit is not None and not _commit_id(args.declared_bounds_commit):parser.error('--declared-bounds-commit must be a full lowercase hex commit id')
        return parser, args
    g2.check_head_arguments(parser, args);ce.check_arguments(parser, args);check_declared_arguments(parser, args)
    if args.raw_head_check and not args.head_archive:parser.error('--raw-head-check needs --head-archive')
    if not args.out:parser.error('--out is required (every mode but --reverdict)')
    if args.diagnostics_from or args.mutation_from:
        if (args.saved_roots or args.d3_root or args.head_archive or args.champion_archive or args.n2_equivalence or args.depth3_diagnostics or args.mutation_test
                or args.save_oracle):
            parser.error('--diagnostics-from / --mutation-from run offline: no roots, head, champion, --n2-equivalence, --depth3-diagnostics, --mutation-test or --save-oracle')
        if args.diagnostics_from and args.mutation_from:parser.error('one of --diagnostics-from or --mutation-from')
        if args.fixed_shapes:parser.error('--fixed-shapes shapes the live pricers and oracle; --diagnostics-from / --mutation-from price nothing')
        return parser, args
    if args.fixed_shapes and args.n2_equivalence:parser.error('--fixed-shapes does not apply to --n2-equivalence: its Depth2Pricer keeps per-call '
        'buckets, so the two compared pricers would price turn 1 at different batch sizes')
    if (args.mutation_test or args.save_oracle) and (args.n2_equivalence or args.depth3_diagnostics):parser.error('--mutation-test and --save-oracle belong to the gate')
    if args.mutation_test and args.sample_leaves is not None:parser.error('--mutation-test needs every leaf checked (no --sample-leaves)')
    if args.mutation_test and args.depth < 3:parser.error('--mutation-test needs --depth >= 3 (parents at ply >= 2)')
    if args.save_oracle:args.save_arrays = True   # the oracle data is saved beside the frames it checks
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
    if args.block_invariance and (args.depth3_diagnostics or (args.n2_equivalence and not args.champion_archive)):   # champion + n2: CHILD_BLOCK_INVARIANCE
        parser.error('--block-invariance belongs to the gate')
    if args.raw_head_check and args.depth3_diagnostics:parser.error('--raw-head-check needs the oracle')
    if args.selection == 'searched' and args.depth < 3:parser.error('searched selection needs --depth >= 3 (it selects ply 2 from a depth-2 pass)')
    return parser, args


def main(argv=None):
    parser, args = parse(argv)
    if args.reverdict:return reverdict(args.reverdict, expected=dict(commit=args.declared_bounds_commit, sha256=args.declared_bounds_sha256))
    if args.diagnostics_from:
        out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
        report = dict(header=header_block(), schema=SCHEMA, mode='DIAGNOSTICS_OFFLINE', **offline_diagnostics(args.diagnostics_from, progress=log))
        write(out / 'diagnostics.json', report)
        log('DIAGNOSTICS_COMPLETE', out=str(out / 'diagnostics.json'), calls=report['calls'], change_rate={k: v['rate'] for k, v in report['change_rate'].items()})
        return 0 if report['calls'] else 1
    if args.mutation_from:
        out = Path(args.out);out.mkdir(parents=True, exist_ok=False);clock = time.perf_counter()
        found = offline_mutation(args.mutation_from, progress=log)
        report = mutation_report(found['calls'], source=dict(offline=[str(d) for d in args.mutation_from]), checks=found['checks'], value_bound=found['value_bound'])
        report['seconds'] = time.perf_counter() - clock;write(out / MUTATION_FILE, report)
        log('MUTATION_TEST_COMPLETE', status=report['status'], out=str(out / MUTATION_FILE), calls=len(report['calls']), survivors=report['survivors'], seconds=report['seconds'])
        return 0 if report['passed'] else 1
    declared = None
    if args.declared_bounds:   # before anything is created or loaded: the file, its sha256, its schema and the flags' evaluator identity
        try:declared = load_declared(args.declared_bounds, args.declared_bounds_sha256, args.declared_bounds_commit)
        except DeclaredBoundsError as error:raise SystemExit(f'--declared-bounds (fail closed): {error}')
        mismatched = evaluator_mismatches(declared['content'], args_identity(args))
        if mismatched:raise SystemExit('--declared-bounds (fail closed): the declaration is not for this run\'s evaluator: ' + '; '.join(mismatched))
    needed = ['gpu_engine.device_depthn_pricer', 'gpu_engine.depthn_backups', 'gpu_engine.depth2_backups'] + (['gpu_engine.device_depth2_pricer'] if args.n2_equivalence else [])
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    if missing:raise SystemExit('not on this tree (failing closed): ' + ', '.join(missing))
    log('TOLERANCE_RULE', rule=g2.TOLERANCE_RULE, logit_bound=g2.LOGIT_BOUND, guard_factor=g2.GUARD_FACTOR, guard_spread='p10 learner-max own-slot spread of the last ply\'s parents',
        guard_depthn_rule=GUARD_DEPTHN_RULE, deeper_logit_rule=DEEPER_LOGIT_RULE,
        **({} if declared is None else dict(numbers_replaced_by=f'DECLARED_BOUNDS (--declared-bounds sha256 {declared["sha256"]})')))
    log('EVALUATOR', override=bool(args.head_archive or args.champion_archive), archive=args.head_archive, archive_sha256=args.head_archive_sha256,
        checkpoint_sha256=args.head_checkpoint_sha256, value_reference=args.value_reference, **ce.log_fields(args),
        resolved='EVALUATOR_RESOLVED follows once the roots are loaded')
    if declared is not None:
        log('DECLARED_BOUNDS', path=declared['path'], sha256=declared['sha256'], **{k: declared['content'][k] for k in DECLARED_FIELDS if k != 'schema'},
            **{k: declared[k] for k in ('repo_path', 'git_blob_sha1', 'commit', 'git_verification', 'source_measurement_verification', 'run_started_at')},
            rule=DECLARED_RULE, note=DECLARED_NOTE)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    roots = load_roots(args);provider, plan = roots.loaded.provider, roots.loaded.plan;module = pricer_module()
    if declared is not None:
        mismatched = evaluator_mismatches(declared['content'], evaluator_identity(roots.evaluator))
        if mismatched:raise SystemExit('--declared-bounds (fail closed): the declaration is not for the resolved evaluator: ' + '; '.join(mismatched))
    reference = importlib.import_module('gpu_engine.depthn_oracle');frame_pricer = g2.with_forward(roots.loaded.frame_pricer, getattr(roots, 'forward', None))
    seeds = g2.world_block_seeds(provider.recipe.seed, args.worlds_blocks)
    common = dict(roots=roots, provider=provider, plan=plan, module=module, frame_pricer=frame_pricer, seeds=seeds, out=out, reference=reference,
        **({} if declared is None else dict(declared=declared)))
    if args.n2_equivalence:return run_n2(args, **common)
    if args.depth3_diagnostics:return run_diagnostics(args, **common)
    return run_gate(args, **common)


def header(args, roots, abs_values, source):
    return dict(evaluator=roots.evaluator, value_scale=g2.value_scale(roots.root_values, g2.percentiles(abs_values), args.value_reference), source=source)


def run_gate(args, *, roots, provider, plan, module, frame_pricer, seeds, out, reference, declared=None):
    lanes = INVARIANCE_LANES if args.block_invariance else (args.block_lanes,);forward = getattr(roots, 'forward', None)
    section = None if declared is None else declared_section(declared, roots.evaluator)
    evaluator = roots.evaluator if declared is None else dict(roots.evaluator, declared_bounds_sha256=declared['sha256'])   # saved frames' meta (--mutation-from refuses them)
    declared_only = {} if declared is None else dict(declared_bounds_sha256=declared['sha256'])
    d2_parity = forward is not None and args.depth == 2   # the champion's depth-2 gate: D2_ORACLE_CONSISTENCY, the merged_depth2_parity header
    parity = merged_depth2_parity('gate', oracle_consistency=True) if d2_parity else None
    pricers = DepthNPricers(module, provider.executor, plan.model, depth=args.depth, k_own=args.k_own_ply, k_opp=args.k_opp_ply, selection=args.selection,
        capacity=args.capacity, actor_capacity=roots.actor_capacity, lanes=lanes, k1_own=args.k1_own, k1_opp=args.k1_opp, fixed_capacity=args.fixed_shapes,
        forward=forward)
    depth2_pricer = depth2_pass_pricer(module, provider.executor, plan.model, capacity=args.capacity, actor_capacity=roots.actor_capacity,
        k1_own=args.k1_own, k1_opp=args.k1_opp, block_lanes=args.block_lanes, digest=True, fixed_capacity=args.fixed_shapes, forward=forward)
    capacity = g2.call_capacity(frame_pricer, [depth2_pricer]);groups = g2.split_groups(roots.groups, capacity)
    harness = fixed_shapes_harness(capacity) if args.fixed_shapes else None;shaped = {} if harness is None else dict(harness=harness)
    oracle = reference.DepthNOracle(provider.executor, plan.model, chunk=args.oracle_chunk, progress=g2.oracle_chunk_progress(log),
        raw_model=roots.head[1] if args.raw_head_check else None, **forward_kwargs(forward))
    log('GATE_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), pricer_options=pricers.options(), sample_leaves=args.sample_leaves, seeds=seeds,
        mutation_test=args.mutation_test, save_oracle=args.save_oracle, **shaped, **declared_only)
    evaluations, calls = [], []
    for block, call, recipe, requests in each_call(provider, groups, seeds):
        evaluation, summary = run_call_n(requests=requests, recipe=recipe, materializer=provider.materializer, params=roots.params, pricers=pricers,
            depth2_pricer=depth2_pricer, frame_pricer=frame_pricer, oracle=oracle, sample=args.sample_leaves, block=block,
            context=dict(block=block, call=call), progress=log, save=(out, f'frames-block{block}-call{call}') if args.save_arrays else None,
            evaluator=evaluator, pass_sample=args.depth2_pass_sample, save_oracle=args.save_oracle,
            mutation={} if args.mutation_test else None, stack_rows=None if harness is None else harness['oracle_stack_rows'], d2_consistency=d2_parity)
        evaluations.append(evaluation);calls.append(summary);log('CALL_CHECKED', **summary)
    result = summarize_run(evaluations, lanes, declared=declared);log('TOLERANCE', **tolerance_line_n(result['tolerance']))
    status, reasons = gate_status_n(result, sampled=args.sample_leaves is not None)
    reported = [c for c in calls if c['refusals'] is not None]
    refusals = dict(g2.with_dropped(merge_refusals_n([c['refusals'] for c in reported]), roots.evaluator.get('dropped_roots', ()), roots.evaluator),
        calls_reported=len(reported),
        rule='reference (D) at every ply: any world pre != 0 or any status-3 node or leaf refuses the root; depth2_full_width is the full-width depth-2 pass')
    abs_values = np.concatenate([e['abs_t1'] for e in evaluations if 'abs_t1' in e]) if any('abs_t1' in e for e in evaluations) else []
    report = dict(header(args, roots, abs_values, roots.loaded.source), header=header_block(result['checks'], declared=section, parity=parity), tolerance=result['tolerance'], schema=SCHEMA, status=status,
        passed=status in ('PASS', 'PASS_SAMPLED'), status_reasons=reasons, checks=result['checks'],
        decision=result['checks'].get('e', dict(status='SKIPPED_SAMPLED', reason='check (e) needs every leaf checked (no --sample-leaves)')),
        mode='SAVED_ROOTS' if args.saved_roots else 'BANK_CHUNK', pricer_options=pricers.options(), block_invariance=args.block_invariance,
        raw_head_check=args.raw_head_check, split_capacity=capacity, **shaped, worlds_blocks=args.worlds_blocks, block_seeds=list(seeds), max_roots=args.max_roots,
        sample_leaves=args.sample_leaves, sample_seed=g2.SAMPLE_SEED,
        leaves=dict(enumerated=sum(c['leaves_enumerated'] for c in calls), checked=sum(c['leaves_checked'] for c in calls),
            forced_terminal_or_error=sum(c['forced_leaves'] for c in calls)),
        searched_maps=[dict(block=c['block'], call=c['call'], **c['searched_maps']) for c in calls if c.get('searched_maps')] or None,
        searched_rule=None if args.selection != 'searched' else ('the supplied maps are gated EXACT in (a) (searched_maps_*_rule, searched_maps_widths, '
            'searched_depth2_refused) against this gate\'s own re-derivation (searched_maps_reference) from the full-width depth-2 pass; see depth2_pass'),
        depth2_pass=dict(trust=PASS_TRUST, sample_per_world=args.depth2_pass_sample, **{k: sum((c.get('depth2_pass') or {}).get(k, 0) for c in calls)
            for k in ('enumerated', 'rerun', 'forced')}),
        topk_boundary=merge_topk_n([c.get('topk') for c in calls], **({} if declared is None else dict(logit_bound=declared_numbers(declared)['turn1_logit_bound']))),
        refusals=refusals, calls=calls, oracle_cost=dict(seconds=oracle.seconds, rows=oracle.rows, chunk=args.oracle_chunk))
    write(out / 'gate.json', report)
    log('DEPTHN_GATE_COMPLETE', status=status, out=str(out / 'gate.json'), status_reasons=reasons, checks={k: v['passed'] for k, v in result['checks'].items()},
        refused_share={k: refusals.get(k, {}).get('refused_share') for k in ('depthn', 'depth2_full_width', 'depth1')}, header=report['header'])
    if not args.mutation_test:return 0 if report['passed'] else 1
    bound = result['tolerance']['value_bound']
    with stage(log, 'mutation_test', calls=len(evaluations), value_bound=bound) as info:
        mutations = mutation_tests(evaluations, value_bound=bound, lanes=lanes, progress=log, declared=declared);info.update(passed=[m['passed'] for m in mutations])
    mutation = mutation_report(mutations, seconds=sum(m.get('seconds', 0.) for m in mutations), source='live', checks=result['checks'], value_bound=bound, declared=section)
    write(out / MUTATION_FILE, mutation)
    log('MUTATION_TEST_COMPLETE', status=mutation['status'], out=str(out / MUTATION_FILE), gate_status=status, value_bound=bound, survivors=mutation['survivors'],
        material_survivors=mutation['material_survivors'], seconds=mutation['seconds'])
    return 0 if report['passed'] and mutation['passed'] else 1


def run_n2(args, *, roots, provider, plan, module, frame_pricer, seeds, out, reference, declared=None):
    configs, options = [], {};forward = getattr(roots, 'forward', None);section = None if declared is None else declared_section(declared, roots.evaluator)
    blocks = g2.INVARIANCE_BLOCKS if forward is not None and getattr(args, 'block_invariance', False) else (None,)   # parse refuses --block-invariance off the champion path
    for name, k in (('full_width', 9), (f'top{args.n2_k}', args.n2_k)):
        depthn = module.DepthNPricer(provider.executor, plan.model, capacity=args.capacity, depth=2, k_own=k, k_opp=k, k1_own=args.k1_own, k1_opp=args.k1_opp,
            block_lanes=args.block_lanes, selection='prior', digest=True, actor_capacity=roots.actor_capacity, **forward_kwargs(forward))
        made = [g2.make_pricer(provider.executor, plan.model, widths=dict(k_own=k, k_opp=k, k1_own=args.k1_own, k1_opp=args.k1_opp),
            actor_capacity=roots.actor_capacity, capacity=args.capacity, **({} if b is None else dict(child_block=b)), **forward_kwargs(forward)) for b in blocks]
        (depth2, depth2_options), others = made[-1], [p for p, _ in made[:-1]]
        configs.append((name, depthn, depth2) + ((others,) if others else ()))
        options[name] = dict(k=k, depth2_options=depth2_options, **({} if not others else dict(child_blocks=list(blocks), invariance_depth2_options=[o for _, o in made[:-1]])))
    capacity = g2.call_capacity(frame_pricer, [c[1] for c in configs] + [c[2] for c in configs] + [o for c in configs for o in (c[3] if len(c) > 3 else ())])
    groups = g2.split_groups(roots.groups, capacity)
    parity = None if forward is None else merged_depth2_parity('n2_equivalence', child_block_invariance=blocks != (None,))
    oracle = reference.DepthNOracle(provider.executor, plan.model, chunk=args.oracle_chunk, progress=g2.oracle_chunk_progress(log), **forward_kwargs(forward))
    log('N2_EQUIVALENCE_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), configs=options)
    per_config = collections.defaultdict(list);abs_values = []
    for block, call, recipe, requests in each_call(provider, groups, seeds):
        results = n2_equivalence_call(requests=requests, recipe=recipe, materializer=provider.materializer, params=roots.params, configs=configs,
            frame_pricer=frame_pricer, oracle=oracle, context=dict(block=block, call=call), progress=log, module=module, merged_decision=forward is not None)
        for name, result in results.items():per_config[name].append(result)
        log('N2_CALL', block=block, call=call, discrete={n: r['a']['passed'] for n, r in results.items()}, e={n: r['e'].get('argmax_q_differs') for n, r in results.items()})
    report = summarize_n2(per_config, declared=declared);statuses = [r['status'] for r in report.values()]
    status = 'FAIL' if 'FAIL' in statuses else 'INCOMPLETE' if 'INCOMPLETE' in statuses or not report else 'PASS'
    write(out / 'n2-equivalence.json', dict(header(args, roots, abs_values, roots.loaded.source), header=header_block(configs=report, declared=section, parity=parity), schema=SCHEMA, mode='N2_EQUIVALENCE',
        status=status, passed=status == 'PASS', configs=report, options=options, split_capacity=capacity, block_seeds=list(seeds),
        rule='per config: every discrete array exact and world_reports equal; floats only on scored cells (see n2_compare): values within the value '
            'bound and guard, priors within the logit bound (the run\'s evaluator-relative rule); (e) depth2_backups argmax per root equal except near '
            'ties; INCOMPLETE when no turn-2 leaf or no (e) root was compared'))
    log('N2_EQUIVALENCE_COMPLETE', status=status, out=str(out / 'n2-equivalence.json'), configs={n: r['status'] for n, r in report.items()})
    return 0 if status == 'PASS' else 1


def run_diagnostics(args, *, roots, provider, plan, module, frame_pricer, seeds, out, reference):
    forward = getattr(roots, 'forward', None)
    depth2_pricer = depth2_pass_pricer(module, provider.executor, plan.model, capacity=args.capacity, actor_capacity=roots.actor_capacity,
        k1_own=args.k1_own, k1_opp=args.k1_opp, block_lanes=args.block_lanes, fixed_capacity=args.fixed_shapes, forward=forward)
    sets = {(mode, k): DepthNPricers(module, provider.executor, plan.model, depth=3, k_own=k, k_opp=k, selection=mode, capacity=args.capacity,
        actor_capacity=roots.actor_capacity, lanes=(args.block_lanes,), k1_own=args.k1_own, k1_opp=args.k1_opp, digest=False, fixed_capacity=args.fixed_shapes,
        forward=forward) for mode in MODES for k in args.diagnostic_k_values}
    capacity = g2.call_capacity(frame_pricer, [depth2_pricer]);groups = g2.split_groups(roots.groups, capacity)
    shaped = {} if not args.fixed_shapes else dict(harness=fixed_shapes_harness(capacity, oracle=False))   # no oracle in this mode
    log('DIAGNOSTICS_START', roots=sum(map(len, groups)), calls=len(groups) * len(seeds), configs=[frame_tag(*k) for k in sets], **shaped)
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
    report = dict(header(args, roots, np.concatenate(abs_values) if abs_values else [], roots.loaded.source), header=header_block(), schema=SCHEMA, mode='DIAGNOSTICS_LIVE',
        **merge_diagnostics(calls), calls=len(calls), block_seeds=list(seeds), ks=list(args.diagnostic_k_values), saved_arrays=args.save_arrays,
        pricer_options={frame_tag(*k): p.options() for k, p in sets.items()}, **shaped)
    write(out / 'diagnostics.json', report)
    log('DIAGNOSTICS_COMPLETE', out=str(out / 'diagnostics.json'), calls=len(calls), change_rate={k: v['rate'] for k, v in report['change_rate'].items()})
    return 0 if calls else 1


if __name__ == '__main__':raise SystemExit(main())
