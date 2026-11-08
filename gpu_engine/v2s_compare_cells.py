"""Compare two search head-to-head cells played on the same pairs (CPU only).

`--x` and `--y` are the output prefixes `<out>/<label>` of two `v2_search_head_to_head` runs. The tool
reports:
- the paired difference X - Y of pair scores, on the pairs valid in both, with a 90% normal interval (in
  points of score);
- with `--identical`, the plumbing check: the same game records (status, winner, terminal frame per stream),
  and the same raw and chosen action at every decision of seat A;
- with `--near-terminal`, the champion mirror's pre-registered value check on X's own log: on searched decisions
  within 3 frames of a COMPLETE terminal, the sign agreement of A's root value with A's outcome (win +1, loss -1;
  draws apart). PASS: agreement >= 0.8 on |v| >= 0.5 with at least 50 such decisions; fewer is INCONCLUSIVE.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def pairs(prefix):
    return {r['stream']: r['score'] for r in map(json.loads, Path(str(prefix) + '-pairs.jsonl').read_text().splitlines())}


def paired_difference(x, y):
    if set(x) != set(y):raise ValueError('cells were not played on the same pairs')
    d = np.asarray([x[k] - y[k] for k in sorted(x) if x[k] is not None and y[k] is not None], np.float64)
    mean = float(d.mean());se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
    return dict(pairs=len(d), excluded=len(x) - len(d), difference_points=100 * mean,
        interval_90_points=[100 * (mean - 1.645 * se), 100 * (mean + 1.645 * se)], standard_error_points=100 * se,
        pairs_x_better=int((d > 0).sum()), pairs_y_better=int((d < 0).sum()))


def identical(x, y):
    gx, gy = np.load(str(x) + '-games.npz'), np.load(str(y) + '-games.npz')
    games = {k: bool(np.array_equal(gx[k], gy[k])) for k in ('streams', 'status', 'winner', 'terminal_frame')}
    dx = [json.loads(line) for line in Path(str(x) + '-decisions.jsonl').read_text().splitlines()]
    dy = [json.loads(line) for line in Path(str(y) + '-decisions.jsonl').read_text().splitlines()]
    key = lambda rows: {(r['stream'], d['frame']): (d['raw'], d['chosen']) for r in rows for d in r['decisions']}
    kx, ky = key(dx), key(dy)
    return dict(games=games, decisions_x=len(kx), decisions_y=len(ky), same_decision_set=set(kx) == set(ky),
        differing_decisions=sum(kx[k] != ky.get(k) for k in kx), all_identical=all(games.values()) and kx == ky)


NEAR_FRAMES, NEAR_BAR, NEAR_FLOOR, NEAR_CONFIDENT = 3, .8, 50, .5


def near_terminal(prefix):
    """Root value vs outcome on searched decisions within NEAR_FRAMES of a COMPLETE terminal (pre-registered, amendment 13)."""
    games = np.load(str(prefix) + '-games.npz')
    complete = {int(s): (int(w), int(t)) for s, w, t, st in zip(games['streams'], games['winner'], games['terminal_frame'], games['status'])
        if int(st) == 1}   # v5_device_rollout.COMPLETE
    groups = {'confident': [], 'unconfident': []};draws = 0
    for row in map(json.loads, Path(str(prefix) + '-decisions.jsonl').read_text().splitlines()):
        if row['stream'] not in complete:continue
        winner, terminal = complete[row['stream']]
        for d in row['decisions']:
            if not d.get('searched') or d.get('root_value') is None or not 0 < terminal - d['frame'] <= NEAR_FRAMES:continue
            if winner == 2:draws += 1;continue
            outcome = 1. if winner == row['a_side'] else -1.;v = float(d['root_value'])
            groups['confident' if abs(v) >= NEAR_CONFIDENT else 'unconfident'].append((v, outcome))
    def summary(items):
        if not items:return dict(decisions=0, agreement=None, mean_abs_value=None)
        return dict(decisions=len(items), agreement=float(np.mean([np.sign(v) == o for v, o in items])),
            mean_abs_value=float(np.mean([abs(v) for v, _ in items])))
    confident = summary(groups['confident'])
    verdict = ('INCONCLUSIVE' if confident['decisions'] < NEAR_FLOOR else 'PASS' if confident['agreement'] >= NEAR_BAR else 'FAIL')
    return dict(frames=NEAR_FRAMES, bar=NEAR_BAR, floor=NEAR_FLOOR, confident_abs_value=NEAR_CONFIDENT, verdict=verdict,
        confident=confident, unconfident=summary(groups['unconfident']), draws=draws)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--x', required=True);parser.add_argument('--y', required=True)
    parser.add_argument('--identical', action='store_true');parser.add_argument('--out')
    parser.add_argument('--near-terminal', action='store_true', help="X's root-value vs outcome check (champion mirror)")
    args = parser.parse_args(argv)
    report = dict(x=args.x, y=args.y, paired=paired_difference(pairs(args.x), pairs(args.y)))
    if args.identical:report['identical'] = identical(args.x, args.y)
    if args.near_terminal:report['near_terminal'] = near_terminal(args.x)
    text = json.dumps(report, indent=1)
    if args.out:Path(args.out).write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
