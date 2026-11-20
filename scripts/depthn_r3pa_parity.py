"""R3pa parity (CPU only): gpu_engine.depthn_r3pa.choose (prior substitution into WorldPrices) on the re-pricer's rows against an
expected rung's actions, with the direct prior override (the offline gate's computation) beside it for diagnostics.

Roots: the rows (scripts/depthn_rr_reprice.py JSONL) loaded as mc_ladder_backline loads its sampled run (mc_ladder_worlds.load:
mc_ladder_rungs.identity, the manifest for self roots, run E for self and C for Foul Play). Per root, as the offline gate that
produced the R3pa / R3pag rungs prices it:
- worlds: with --depth2 the row's top-level worlds (each world's depth-2 backed-up grid and root prior, the live search's:
  depthn_rr_score.world_prices on depthn_rr_score.normalize(dict(worlds=...))), else the depth-1 grids
  (mc_ladder_worlds.depth1_worlds / prices_of). Foul Play roots take the reply model's logits as each requested world's prior
  (depthn_rr_score.model_worlds / reply_logits_of); self roots keep the champion's. R3pa averages these priors.
- labels: mc_ladder_backline.labels(world, seen_species(worlds)), refused by mc_ladder_backline.check_labels (lane_unlabelled).
- the root: mc_ladder_rungs.root_prior's legal set (on mc_ladder_worlds.normalized) and the row's root.root_logits.
Then depthn_r3pa.choose(prices, legal, root_logits, labels, --mode) plays: off = today's NOISE_GATED_K2 (rung R3g), gated = R3pag,
replace_gate = R3pa. An undecided root (any refusal above, an unknown cell, a ValueError from choose) has no action.
Direct (diagnostics only): the same root priors put straight into search_hedge's grid prior arrays (no logits), averaging's
summation, noise_gated on per_world_q: the offline gate's own computation. Substitution passes the priors through log / softmax,
so q may differ in the last bits and an exact near-tie may flip.
--expect: a rung file (one row per root keyed by mc_ladder_rungs.KEY_FIELDS; action null when undecided).
Prints one JSON line: roots, compared (both decided), agree (exact action agreement), disagreements (both decided, different
actions), decided_mismatch (one side undecided), both_undecided, missing roots on either side, undecided reasons, gate counts,
prior fallback worlds; direct: its agreement with the expected rung and with the substituted decision, and the substituted q's
distance from it (bitwise-equal roots, max |dq|); every disagreement with the top-two q gap (best minus second legal action)
under both computations and, gated modes, the gate margin (mean gap - k standard errors; None when candidate = greedy) under both;
up to 20 decided_mismatch / missing examples. Exit 1 on any disagreement, decided_mismatch or missing root, or when nothing is
compared.

usage: python -m scripts.depthn_r3pa_parity --kind self|foulplay [--manifest MANIFEST.jsonl] [--depth2]
           --mode off|gated|replace_gate --expect RUNG.jsonl --rows ROWS...
"""
import argparse
import collections
import dataclasses
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from gpu_engine import depthn_r3pa as r3pa  # noqa: E402
from gpu_engine import mc_ladder_backline as B  # noqa: E402
from gpu_engine import search_hedge as hedge  # noqa: E402

L, M, rr = B.L, B.M, B.rr
EXAMPLES = 20


def root_inputs(raw, kind, depth2):
    """(prices, root_legal, root_logits, labels) of one root; raises depthn_rr_score.Unscorable (module docstring)."""
    legal, _ = L.root_prior(M.normalized(raw));logits = (raw.get('root') or {}).get('root_logits')
    if logits is None:raise rr.Unscorable('root_logits_missing')
    if depth2:
        worlds = raw.get('worlds') if type(raw.get('worlds')) is list else []
        if not worlds:raise rr.Unscorable('worlds_missing')
        prices = rr.world_prices(rr.normalize(dict(worlds=worlds)))
        if M.REPLY[kind]:prices = rr.model_worlds(prices, rr.reply_logits_of(rr.normalize(dict(worlds=worlds)), prices, None))
    else:
        worlds = M.depth1_worlds(raw);prices = M.prices_of(worlds, M.REPLY[kind])
    seen = B.seen_species(worlds)
    if seen is None:raise rr.Unscorable('world_sets_missing')
    labs = [B.labels(w, seen) for w in worlds];B.check_labels(worlds, prices, labs)
    return prices, legal, logits, labs


def decide(raw, kind, depth2, mode):
    """(action or None, undecided reason or None, depthn_r3pa.Decision or None, root_inputs or None) of one root."""
    inputs = None
    try:
        inputs = root_inputs(raw, kind, depth2);decision = r3pa.choose(*inputs, mode)
    except rr.Unscorable as error:return None, error.reason, None, inputs
    except r3pa.LaneUnlabelled:return None, 'lane_unlabelled', None, inputs
    except ValueError as error:return None, f'choose_error:{error}', None, inputs
    if decision is None:return None, 'unpriced_cell', None, inputs
    return decision.action, None, decision, inputs


def margin(qw, candidate, greedy, k):
    """noise_gated's margin, mean gap - k standard errors (passes when > 0); None when candidate = greedy."""
    if candidate == greedy:return None
    gaps = [q[candidate] - q[greedy] for q in qw]
    return statistics.fmean(gaps) - k * statistics.stdev(gaps) / math.sqrt(len(gaps))


def direct(prices, legal, logits, priors, k=r3pa.NOISE_GATE_K):
    """The offline gate's computation: priors (None: the worlds' own) straight into the grids' prior arrays, averaging's summation,
    noise_gated on per_world_q: dict(q (nine entries), candidate, played, gate_margin)."""
    gs = hedge.grids(prices, legal)
    if priors is not None:
        gs = [g if pr is None else dataclasses.replace(g, prior=np.asarray([pr.get(b, 0.) for b in g.replies], np.float64))
              for g, pr in zip(gs, priors)]
    count = len(gs);actions = gs[0].actions
    q = [math.fsum(g.prior[j] * g.values[i, j] / count for g in gs for j in range(len(g.replies))) for i in range(len(actions))]
    candidate = actions[max(range(len(actions)), key=lambda i: (q[i], -i))];full = [None] * 9
    for a, v in zip(actions, q):full[a] = v
    if len(gs) < 2:return dict(q=full, candidate=candidate, played=None, gate_margin=None)   # no gate on one world
    qw = hedge.per_world_q(gs);greedy = hedge.logits_greedy(logits, legal)
    return dict(q=full, candidate=candidate, played=hedge.noise_gated(qw, candidate, greedy, k), gate_margin=margin(qw, candidate, greedy, k))


def top_two_gap(q):
    """Best minus second-best q over the legal (non-None) actions; None with one legal action."""
    vals = sorted((v for v in q if v is not None), reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else None


def compare(rows, kind, mode, depth2, expected_rows, manifest=None):
    """The report (module docstring)."""
    loads = collections.Counter();roots = M.load(rows, kind, manifest or {}, M.SAMPLED_RUN[kind], loads)
    expected = {}
    for r in expected_rows:
        index = tuple(r[f] for f in L.KEY_FIELDS[kind])
        if index in expected:raise ValueError(f'two expected rows for root {index}')
        expected[index] = r
    c = collections.Counter();reasons = collections.Counter();examples = [];details = [];max_dq = 0.
    for index in sorted(roots):
        _, raw, _ = roots[index];action, reason, decision, inputs = decide(raw, kind, depth2, mode);ref = None
        if action is None:reasons[reason] += 1
        else:
            c['decided'] += 1;c['prior_fallback_worlds'] += decision.fallback_worlds
            if decision.gate is not None:c['gate_passed' if decision.gate.passed else 'gate_fell_back'] += 1
            prices, legal, logits, _ = inputs;ref = direct(prices, legal, logits, decision.priors)
            ref['action'] = ref['candidate'] if mode == 'replace_gate' else ref['played']
            dq = [abs(x - y) for x, y in zip(decision.q, ref['q']) if x is not None];max_dq = max(max_dq, *dq)
            c['direct_q_bitwise_equal'] += all(x == y for x, y in zip(decision.q, ref['q']))
            c['direct_action_differs'] += ref['action'] != action;c['direct_candidate_differs'] += ref['candidate'] != decision.candidate
        if index not in expected:
            c['missing_expected'] += 1
            if len(examples) < EXAMPLES:examples.append(dict(root=list(index), kind='missing_expected', mine=action, mine_reason=reason))
            continue
        theirs = expected[index].get('action')
        if ref is not None and theirs is not None:c['direct_agree'] += ref['action'] == theirs
        if action is None and theirs is None:c['both_undecided'] += 1;continue
        if action is None or theirs is None:
            c['decided_mismatch'] += 1
            if len(examples) < EXAMPLES:
                examples.append(dict(root=list(index), kind='decided_mismatch', mine=action, mine_reason=reason, expected=theirs,
                                     expected_reason=expected[index].get('reason')))
            continue
        c['compared'] += 1
        if action == theirs:c['agree'] += 1;continue
        c['disagreements'] += 1;gate = decision.gate
        details.append(dict(root=list(index), mine=action, expected=theirs, direct=ref['action'], candidate=decision.candidate,
                            direct_candidate=ref['candidate'], greedy=decision.greedy,
                            q_gap_substituted=top_two_gap(decision.q), q_gap_direct=top_two_gap(ref['q']),
                            gate_margin_substituted=None if gate is None or gate.mean_gap is None else gate.mean_gap - r3pa.NOISE_GATE_K * gate.stderr,
                            gate_margin_direct=None if gate is None else ref['gate_margin']))
    missing = [index for index in expected if index not in roots];c['missing_mine'] = len(missing)
    examples += [dict(root=list(index), kind='missing_mine', expected=expected[index].get('action')) for index in missing[:max(0, EXAMPLES - len(examples))]]
    fields = ('decided', 'compared', 'agree', 'disagreements', 'decided_mismatch', 'both_undecided', 'missing_expected', 'missing_mine',
              'gate_passed', 'gate_fell_back', 'prior_fallback_worlds')
    return dict(kind=kind, mode=mode, depth2=depth2, roots=len(roots), expected_rows=len(expected), **{f: c[f] for f in fields},
                undecided=dict(sorted(reasons.items())), loads=dict(sorted(loads.items())),
                direct=dict(agree_with_expected=c['direct_agree'], action_differs=c['direct_action_differs'],
                            candidate_differs=c['direct_candidate_differs'], q_bitwise_equal_roots=c['direct_q_bitwise_equal'], max_abs_q_diff=max_dq),
                disagreement_details=details, examples=examples)


def failed(report):
    return bool(report['disagreements'] or report['decided_mismatch'] or report['missing_expected'] or report['missing_mine']
                or not report['compared'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=sorted(M.REPLY), required=True);parser.add_argument('--manifest')
    parser.add_argument('--depth2', action='store_true', help="the rows' top-level (depth-2) worlds, else the depth-1 grids")
    parser.add_argument('--mode', choices=r3pa.MODES, required=True);parser.add_argument('--expect', required=True)
    parser.add_argument('--rows', nargs='+', required=True)
    args = parser.parse_args(argv)
    if (args.kind == 'self') != (args.manifest is not None):raise SystemExit('--manifest is required for self roots and refused for Foul Play roots')
    with open(args.expect) as stream:expected = [json.loads(line) for line in stream if line.strip()]
    report = compare(L.read_rows(args.rows), args.kind, args.mode, args.depth2, expected, L.load_manifest(args.manifest) if args.manifest else None)
    print(json.dumps(dict(report, expect=args.expect), sort_keys=True))
    return 1 if failed(report) else 0


if __name__ == '__main__':
    raise SystemExit(main())
