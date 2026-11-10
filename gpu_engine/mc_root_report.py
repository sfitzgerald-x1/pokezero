"""M0 report over mc_root_labeller's roots: cost per rollout and per root, the common-random-numbers correlation, standard errors,
the share of separable sibling pairs at K = 64 and the full K, and a core-hour projection.

usage: python -m gpu_engine.mc_root_report ROOTS.jsonl [ROOTS.jsonl ...] [--train-roots 20000] [--gate-roots 2000] [--gate-K 256]
"""
import argparse
import itertools
import json
import math

import numpy as np


def pair_stats(o, K):
    """Over every sibling pair of one root's outcomes o [A, K'] (first K rollouts): the CRN correlations, the paired difference
    SEs, whether each pair is separable (|mean difference| > 2 SE of the paired difference), and the top pair's separation."""
    o = o[:, :K];out = dict(rho=[], reduction=[], se_pair=[], separable=[], top_separable=None)
    for a, b in itertools.combinations(range(len(o)), 2):
        d = o[a] - o[b];se = d.std(ddof=1) / math.sqrt(K) if K > 1 else float('nan')
        va, vb = o[a].var(ddof=1), o[b].var(ddof=1)
        if va > 0 and vb > 0:out['rho'].append(float(np.corrcoef(o[a], o[b])[0, 1]))
        if va + vb > 0:out['reduction'].append(float(d.var(ddof=1) / (va + vb)))   # 1 - rho when the variances are equal
        out['se_pair'].append(float(se));out['separable'].append(bool(abs(d.mean()) > 2 * se) if se > 0 else bool(d.mean() != 0))
    means = o.mean(1);order = np.argsort(-means)
    if len(o) >= 2:
        d = o[order[0]] - o[order[1]];se = d.std(ddof=1) / math.sqrt(K)
        out['top_separable'] = bool(abs(d.mean()) > 2 * se) if se > 0 else bool(d.mean() != 0)
    return out


def summarize(rows, K):
    rho, red, sep, top, se_pair, se_action, degenerate, roots_any = [], [], [], [], [], [], 0, 0
    for r in rows:
        o = np.asarray(r['outcomes'], float)
        if o.shape[1] < K:continue
        s = pair_stats(o, K);rho += s['rho'];red += s['reduction'];sep += s['separable'];se_pair += s['se_pair']
        se_action += [float(x) for x in o[:, :K].std(1, ddof=1) / math.sqrt(K)]
        if s['top_separable'] is not None:top.append(s['top_separable'])
        degenerate += float(o[:, :K].std()) == 0.;roots_any += any(s['separable'])
    n = sum(np.asarray(r['outcomes']).shape[1] >= K for r in rows)
    mean = lambda x: None if not x else round(float(np.mean(x)), 4)
    return dict(K=K, roots=n, pairs=len(sep), rho_mean=mean(rho), rho_median=None if not rho else round(float(np.median(rho)), 4),
                variance_ratio_mean=mean(red), se_action_mean=mean(se_action), se_pair_mean=mean(se_pair),
                separable_pair_share=mean(sep), roots_with_a_separable_pair=None if not n else round(roots_any / n, 4),
                top_pair_separable_share=mean(top), roots_without_outcome_variance=None if not n else round(degenerate / n, 4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='+');parser.add_argument('--train-roots', type=int, default=20000)
    parser.add_argument('--gate-roots', type=int, default=2000);parser.add_argument('--gate-K', type=int, default=256)
    parser.add_argument('--train-K', type=int, default=64)
    args = parser.parse_args(argv)
    rows = [json.loads(l) for p in args.roots for l in open(p) if l.strip()]
    report = {}
    for (kind, mode, envs), group in itertools.groupby(sorted(rows, key=lambda r: (r['kind'], r['mode'], r['envs'])), key=lambda r: (r['kind'], r['mode'], r['envs'])):
        g = list(group);rollouts = sum(r['rollouts'] for r in g);decisions = sum(sum(map(sum, r['decisions'])) for r in g)
        cpu = sum(r.get('cpu_seconds', 0) for r in g);wall = sum(r['wall_seconds'] for r in g)
        actions = float(np.mean([len(r['actions']) for r in g]))
        cost = dict(roots=len(g), rollouts=rollouts, mean_legal_actions=round(actions, 2), decisions_per_rollout=round(decisions / rollouts, 1),
                    wall_seconds_per_rollout=round(wall / rollouts, 4), cpu_seconds_per_rollout=round(cpu / rollouts, 4),
                    cpu_python_share=round(sum(r.get('cpu_python_seconds', 0) for r in g) / cpu, 3) if cpu else None,
                    cpu_ms_per_decision=round(1000 * cpu / decisions, 2))
        cost['core_seconds_per_root_at_train_K'] = round(cost['cpu_seconds_per_rollout'] * actions * args.train_K, 1)
        cost['core_seconds_per_root_at_gate_K'] = round(cost['cpu_seconds_per_rollout'] * actions * args.gate_K, 1)
        cost['projected_core_hours'] = round((args.train_roots * cost['core_seconds_per_root_at_train_K']
                                              + args.gate_roots * cost['core_seconds_per_root_at_gate_K']) / 3600, 1)
        noise = {f'K{K}': summarize(g, K) for K in sorted({64, max(np.asarray(r['outcomes']).shape[1] for r in g)} | {16})}
        report[f'{kind}/{mode}/envs{envs}'] = dict(cost=cost, noise=noise)
    print(json.dumps(dict(schema='mc-root-report.v1', train_roots=args.train_roots, train_K=args.train_K, gate_roots=args.gate_roots,
                          gate_K=args.gate_K, groups=report), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
