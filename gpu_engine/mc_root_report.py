"""Report over mc_root_labeller's roots: cost per rollout and per root, the common-random-numbers correlation, standard errors,
the share of separable sibling pairs at K = 64 and the full K, a core-hour projection, and (roots labelled with
--baseline-draws) the real-Showdown U1 baseline: the Monte Carlo value of the value head's depth-1 completed-Q choice against the
policy's argmax, paired per root, with a 90% interval clustered by battle seed. A capped rollout (mc_root_labeller.CAPPED) is
censored: left out of every mean, standard error and pair, never counted as an outcome.

usage: python -m gpu_engine.mc_root_report ROOTS.jsonl [ROOTS.jsonl ...] [--train-roots 20000] [--gate-roots 2000] [--gate-K 256]
"""
import argparse
import itertools
import json
import math

import numpy as np

CAPPED = -0.25


def censored(o):
    """Outcomes [A, K] with capped rollouts as NaN."""
    o = np.asarray(o, float).copy();o[o == CAPPED] = np.nan;return o


def pair_stats(o, K):
    """Over every sibling pair of one root's outcomes o [A, K'] (first K rollouts): the CRN correlations, the paired difference
    SEs, whether each pair is separable (|mean difference| > 2 SE of the paired difference), and the top pair's separation."""
    o = censored(o)[:, :K];out = dict(rho=[], reduction=[], se_pair=[], separable=[], top_separable=None)
    def pair(a, b):   # the rollouts where neither sibling was censored
        m = ~np.isnan(o[a]) & ~np.isnan(o[b]);return o[a][m], o[b][m]
    for a, b in itertools.combinations(range(len(o)), 2):
        x, y = pair(a, b);n = len(x)
        if n < 2:continue
        d = x - y;se = d.std(ddof=1) / math.sqrt(n);va, vb = x.var(ddof=1), y.var(ddof=1)
        if va > 0 and vb > 0:out['rho'].append(float(np.corrcoef(x, y)[0, 1]))
        if va + vb > 0:out['reduction'].append(float(d.var(ddof=1) / (va + vb)))   # 1 - rho when the variances are equal
        out['se_pair'].append(float(se));out['separable'].append(bool(abs(d.mean()) > 2 * se) if se > 0 else bool(d.mean() != 0))
    means = np.nanmean(o, 1);order = np.argsort(-means)
    if len(o) >= 2:
        x, y = pair(order[0], order[1])
        if len(x) >= 2:
            d = x - y;se = d.std(ddof=1) / math.sqrt(len(d))
            out['top_separable'] = bool(abs(d.mean()) > 2 * se) if se > 0 else bool(d.mean() != 0)
    return out


def summarize(rows, K):
    rho, red, sep, top, se_pair, se_action, degenerate, roots_any = [], [], [], [], [], [], 0, 0
    for r in rows:
        o = np.asarray(r['outcomes'], float)
        if o.shape[1] < K:continue
        s = pair_stats(o, K);rho += s['rho'];red += s['reduction'];sep += s['separable'];se_pair += s['se_pair']
        c = censored(o)[:, :K];se_action += [float(np.nanstd(row, ddof=1) / math.sqrt(np.sum(~np.isnan(row)))) for row in c if np.sum(~np.isnan(row)) > 1]
        if s['top_separable'] is not None:top.append(s['top_separable'])
        degenerate += float(np.nanstd(c)) == 0.;roots_any += any(s['separable'])
    n = sum(np.asarray(r['outcomes']).shape[1] >= K for r in rows)
    mean = lambda x: None if not x else round(float(np.mean(x)), 4)
    return dict(K=K, roots=n, pairs=len(sep), rho_mean=mean(rho), rho_median=None if not rho else round(float(np.median(rho)), 4),
                variance_ratio_mean=mean(red), se_action_mean=mean(se_action), se_pair_mean=mean(se_pair),
                separable_pair_share=mean(sep), roots_with_a_separable_pair=None if not n else round(roots_any / n, 4),
                top_pair_separable_share=mean(top), roots_without_outcome_variance=None if not n else round(degenerate / n, 4))


def cluster_ci(values, clusters, draws=10000, seed=0):
    """Mean of `values` and its 90% bootstrap interval resampling `clusters` (battle seeds) with replacement."""
    values = np.asarray(values, float);u, inv = np.unique(np.asarray(clusters), return_inverse=True)
    if len(values) == 0:return dict(mean=None, ci90=None, roots=0, clusters=0)
    sums = np.bincount(inv, values, len(u));counts = np.bincount(inv, None, len(u));rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (draws, len(u)));stats = sums[idx].sum(1) / counts[idx].sum(1)
    return dict(mean=round(float(values.mean()), 5), ci90=[round(float(q), 5) for q in np.quantile(stats, [.05, .95])], roots=int(len(values)), clusters=int(len(u)))


def baseline_report(rows):
    """The real-Showdown U1 baseline over roots with a baseline: per root the Monte Carlo value (the action's mean outcome, censored
    rollouts left out) of the value head's depth-1 completed-Q argmax and of the policy's argmax; their paired difference over all
    roots and over the roots where the two choices differ, clustered by battle seed."""
    diffs, clusters, differ, dclusters, agree = [], [], [], [], 0
    for r in rows:
        b = r.get('baseline')
        if not b or not r.get('policy'):continue
        actions = r['actions'];q_mc = np.nanmean(censored(r['outcomes']), 1)
        a_v = int(np.argmax(b['completed_q']));a_pi = int(np.argmax([r['policy'][a] for a in actions]))
        d = float(q_mc[a_v] - q_mc[a_pi]);diffs.append(d);clusters.append(r['seed'])
        if a_v != a_pi:differ.append(d);dclusters.append(r['seed'])
        else:agree += 1
    return dict(roots=len(diffs), same_choice=agree, value_minus_policy=cluster_ci(diffs, clusters),
                value_minus_policy_where_different=cluster_ci(differ, dclusters))


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
        report[f'{kind}/{mode}/envs{envs}'] = dict(cost=cost, noise=noise, u1_baseline=baseline_report(g))
    print(json.dumps(dict(schema='mc-root-report.v1', train_roots=args.train_roots, train_K=args.train_K, gate_roots=args.gate_roots,
                          gate_K=args.gate_K, groups=report), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
