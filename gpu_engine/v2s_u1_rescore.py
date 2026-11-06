"""Re-score U1's predictors against the raw POLICY instead of the raw value head (CPU only, no model).

U1 (`gpu_engine/d3_u1_completed_q.py`) paired every predictor against the raw value head scoring the real
child, which already reflects the opponent's recorded reply. A search target replaces the raw policy in the
loop, so this re-pairs the same fixed top-two pairs against the raw policy's outcome on each pair.

Input: U1's `metrics.json` (the reviewed scorer's per-pair rows for every predictor). The fixed pairs are the
raw-policy rows (all 503 eligible pairs); a predictor's pair outcome is its scorer row's `correct`, missing
when D refused the root. Modes, as in U1: fallback (a refused pair keeps the raw policy's ranking, difference
0) and conditional (only the pairs the predictor resolves). Intervals: 90% game-cluster bootstrap (selector
stream), 10,000 draws, seed 0, ratio of resampled difference sum to resampled pair count -- U1's statistic.
The bank's decision frames (0, 4, 12, 24) are its turn bands.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

KEY = ('stream', 'frame', 'owner')
DRAWS = 10000
SEED = 0
LEVEL = 0.90
BASELINE = 'raw_policy'


def pair_key(row):
    return tuple(row['selector'][k] for k in KEY) + (tuple(row['actions']),)


def interval(sums, counts, draws=DRAWS, seed=SEED):
    """U1's game-cluster ratio bootstrap: resample games, ratio of summed differences to summed pairs."""
    rng = np.random.default_rng(seed)
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    index = rng.integers(0, len(sums), size=(draws, len(sums)))
    stats = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    low, high = np.quantile(stats, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
    return [float(low), float(high)]


def paired(pairs, base, other, *, mode):
    """(other correct - base correct) / pairs on the fixed pairs; other may be None (refused)."""
    if mode == 'fallback':rows = [(p, b, b if o is None else o) for p, b, o in zip(pairs, base, other)]
    elif mode == 'conditional':rows = [(p, b, o) for p, b, o in zip(pairs, base, other) if o is not None]
    else:raise ValueError('declared pairing mode required')
    if not rows:return dict(mode=mode, pairs=0, difference=None)
    clusters = collections.OrderedDict()
    for p, b, o in rows:
        c = clusters.setdefault(p[0], [0, 0]);c[0] += o - b;c[1] += 1
    return dict(mode=mode, pairs=len(rows), games=len(clusters), base_correct=sum(b for _, b, _ in rows),
        other_correct=sum(o for _, _, o in rows), difference=sum(o - b for _, b, o in rows) / len(rows),
        ci90_game_cluster=interval([c[0] for c in clusters.values()], [c[1] for c in clusters.values()]))


def rescore(metrics):
    base_rows = metrics[BASELINE]['top_two']['rows']
    pairs = [pair_key(r) for r in base_rows]
    if len(set(pairs)) != len(pairs):raise ValueError('duplicate fixed pair')
    base = [int(r['correct']) for r in base_rows]
    frames = sorted({p[1] for p in pairs})
    out = {}
    for name, m in sorted(metrics.items()):
        if name == BASELINE:continue
        rows = {pair_key(r): int(r['correct']) for r in m['top_two']['rows']}
        if not set(rows) <= set(pairs):raise ValueError(name + ': pair outside the raw-policy fixed pairs')
        other = [rows.get(p) for p in pairs]
        result = {mode: paired(pairs, base, other, mode=mode) for mode in ('fallback', 'conditional')}
        result['by_frame'] = {str(f): {mode: paired([p for p in pairs if p[1] == f], [b for p, b in zip(pairs, base) if p[1] == f],
            [o for p, o in zip(pairs, other) if p[1] == f], mode=mode) for mode in ('fallback', 'conditional')} for f in frames}
        result['resolved_pairs'] = sum(o is not None for o in other)
        out[name] = result
    return dict(baseline=BASELINE, fixed_pairs=len(pairs), baseline_correct=sum(base),
        pairs_by_frame={str(f): sum(p[1] == f for p in pairs) for f in frames}, versus_raw_policy=out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--metrics', required=True, help="U1's metrics.json")
    parser.add_argument('--out', required=True, help='output json path (must not exist)')
    args = parser.parse_args(argv)
    blob = Path(args.metrics).read_bytes()
    report = dict(schema='v2s-u1-rescore.v1', metrics=dict(path=args.metrics, sha256=hashlib.sha256(blob).hexdigest()),
        bootstrap=dict(draws=DRAWS, seed=SEED, level=LEVEL, cluster='game (selector stream)'), **rescore(json.loads(blob)))
    out = Path(args.out)
    with out.open('x') as stream:stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    for name, r in report['versus_raw_policy'].items():
        f, c = r['fallback'], r['conditional']
        print(f"{name:24s} fallback {100*f['difference']:+6.2f} ({100*f['ci90_game_cluster'][0]:+.1f}, {100*f['ci90_game_cluster'][1]:+.1f}) n={f['pairs']}"
            f" | conditional {100*c['difference']:+6.2f} ({100*c['ci90_game_cluster'][0]:+.1f}, {100*c['ci90_game_cluster'][1]:+.1f}) n={c['pairs']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
