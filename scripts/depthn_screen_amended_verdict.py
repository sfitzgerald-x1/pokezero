#!/usr/bin/env python3
"""Apply the unknown-cells amendment (evals/depthn-foulplay-dn2w16-amendment-unknown-cells-20261108.json) to a panel run's official
readout. The official readout is read, sha256-bound, and never changed; the run's result files supply the per-arm UNKNOWN cells.

The amendment (the lead's ruling, committed before any paired or outcome statistic of the run was computed):
- breakage: 'unknown_cells' is dropped; 'unknown_share' is added when either arm's UNKNOWN cells exceed unknown_share_max_per_arm (5%)
  of the readout's paired cells; every other breakage item of the official readout stands;
- the verdict's lower bound is conservative: every UNKNOWN or missing search cell scores 0 (a loss), every UNKNOWN or missing RAW cell
  1 (a win); the two seats of a seed form a cluster; the same seed-clustered t interval as the declared estimator, two-sided 90%;
- the screen's harm check (the addendum, evals/depthn-foulplay-dn2w16-amendment-harm-mirror-20261108.json) is the mirror image: every
  UNKNOWN or missing search cell scores 1 and every UNKNOWN or missing RAW cell 0, the same clustered interval; harm only if its upper
  bound is < 0 (the known-pairs upper bound beside it, descriptive);
- decision: a void, a held fairness flag or an incomplete run keeps the official decision; at the screen, STOP_HARM if mirror harm, else
  STOP_BREAKAGE if any amended breakage, else CONTINUE_TO_2000; at the bar, BAR_MET when the conservative lower bound is > 0 with no
  amended breakage, else BAR_NOT_MET;
- every UNKNOWN cell is classified by cause (round_cap, foulplay_challenge_timeout, foulplay_choice_timeout, other) and arm.
Only the standard library and scipy (the declared estimator's own t quantile). Writes the verdict once with --output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

AMENDMENT_SCHEMA = 'depthn-foulplay-panel.amendment.v1'
VERDICT_SCHEMA = 'depthn-foulplay-panel.amended-verdict.v1'
T_QUANTILE = .95   # two-sided 90%, as the declared estimator
KEPT = ('VOID_FAIRNESS', 'VOID_LATENCY', 'FLAG_ARM_FAIRNESS', 'INCOMPLETE')
CAUSES = (('round cap', 'round_cap'), ('foul-play challenge', 'foulplay_challenge_timeout'), ('foul-play choice', 'foulplay_choice_timeout'))


def read_bound(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:raise ValueError(f'{path}: sha256 differs from the one given')
    return json.loads(raw)


def cause(error):
    text = str(error or '')
    return next((name for needle, name in CAUSES if needle in text), 'other')


def cells(run_directory, arm, limit):
    """{pair_ordinal: (status, score, error)} for the arm's result files within the first `limit` pairs."""
    out = {}
    for path in sorted(Path(run_directory, 'results').glob('*.json')):
        value = json.loads(path.read_bytes());cell = value.get('cell') or {}
        if cell.get('arm') == arm and type(cell.get('pair_ordinal')) is int and 0 <= cell['pair_ordinal'] < limit:
            if cell['pair_ordinal'] in out:raise ValueError(f'two results for {arm} pair {cell["pair_ordinal"]}')
            out[cell['pair_ordinal']] = (value.get('status'), value.get('score'), value.get('error'))
    return out


def clustered_interval(raw, search, limit, *, search_unknown, raw_unknown):
    """The seed-clustered t interval (two-sided 90%) with every UNKNOWN or missing search cell scored search_unknown and RAW cell
    raw_unknown (the verdict: 0 and 1; the harm mirror: 1 and 0)."""
    from scipy.stats import t
    score = lambda cell, imputed: cell[1] if cell is not None and cell[0] == 'COMPLETE' else imputed
    deltas = [score(search.get(p), search_unknown) - score(raw.get(p), raw_unknown) for p in range(limit)]
    clusters = [(deltas[s] + deltas[s + 1]) / 2 for s in range(0, limit - 1, 2)]
    mean = statistics.fmean(clusters)
    if len(clusters) < 2 or statistics.variance(clusters) == 0:return dict(mean=mean, interval90=None, clusters=len(clusters))
    radius = float(t.ppf(T_QUANTILE, len(clusters) - 1)) * math.sqrt(statistics.variance(clusters) / len(clusters))
    return dict(mean=mean, interval90=[max(-1., mean - radius), min(1., mean + radius)], clusters=len(clusters))


def conservative_interval(raw, search, limit):return clustered_interval(raw, search, limit, search_unknown=0., raw_unknown=1.)


def mirror_interval(raw, search, limit):return clustered_interval(raw, search, limit, search_unknown=1., raw_unknown=0.)


def verdict(readout, amendment, run_directory, *, readout_sha256, amendment_sha256, addendum, addendum_sha256):
    if amendment.get('schema') != AMENDMENT_SCHEMA or readout.get('declaration_sha256') != amendment.get('declaration_sha256'):
        raise ValueError('an amendment of this readout\'s declaration required')
    if (addendum.get('schema') != AMENDMENT_SCHEMA or addendum.get('declaration_sha256') != amendment['declaration_sha256']
            or (addendum.get('addendum_to') or {}).get('sha256') != amendment_sha256):
        raise ValueError('the harm addendum to this amendment required')
    arm, limit = readout['arm'], readout['paired_cells'];screen = readout['bar'] is None
    rule = amendment['rule'];cap = rule['unknown_share_max_per_arm']
    arms = {name: cells(run_directory, name, limit) for name in ('RAW', arm)}
    unknown = {}
    for name, found in arms.items():
        bad = {p: c for p, c in found.items() if c[0] != 'COMPLETE'}
        by_cause = {}
        for _, (_, _, error) in sorted(bad.items()):by_cause[cause(error)] = by_cause.get(cause(error), 0) + 1
        unknown[name] = dict(cells=len(bad), share=len(bad) / limit, by_cause=dict(sorted(by_cause.items())), pairs=sorted(bad),
            results=len(found))
    breakage = [b for b in readout['breakage'] if b != 'unknown_cells']
    if any(u['share'] > cap for u in unknown.values()):breakage.append('unknown_share')
    bound = conservative_interval(arms['RAW'], arms[arm], limit)
    lower = None if bound['interval90'] is None else bound['interval90'][0]
    mirror = mirror_interval(arms['RAW'], arms[arm], limit)
    upper = None if mirror['interval90'] is None else mirror['interval90'][1]
    harm = upper is not None and upper < 0   # the addendum: a harm stop survives every unknown counted for search
    if readout['decision'] in KEPT:decision = readout['decision']
    elif screen:decision = 'STOP_HARM' if harm else 'STOP_BREAKAGE' if breakage else 'CONTINUE_TO_2000'
    else:decision = 'BAR_MET' if lower is not None and lower > 0 and not breakage else 'BAR_NOT_MET'
    paired = readout.get('paired') or {}
    return dict(schema=VERDICT_SCHEMA, amendment_sha256=amendment_sha256, addendum_sha256=addendum_sha256, readout_sha256=readout_sha256,
        arm=arm, paired_cells=limit, official_decision=readout['decision'], official_breakage=readout['breakage'], official_harm=readout['harm'],
        decision=decision, breakage=breakage, harm=harm,
        conservative=dict(rule='UNKNOWN or missing: search 0, RAW 1; seed-clustered t, two-sided 90%', **bound, lower_bound=lower),
        harm_mirror=dict(rule='UNKNOWN or missing: search 1, RAW 0; seed-clustered t, two-sided 90%; harm only if the upper bound < 0', **mirror,
            upper_bound=upper, known_pairs_upper_bound=None if not (paired.get('interval90')) else paired['interval90'][1]),
        known_pairs=dict(scope='descriptive (the declared estimator)', interval90=paired.get('interval90'), known_pairs=paired.get('known_pairs'),
            complete_seed_difference=paired.get('complete_seed_difference'),
            full_roster_identification_bounds=paired.get('full_roster_identification_bounds')),
        unknown=dict(max_share_per_arm=cap, arms=unknown, share_difference=unknown[arm]['share'] - unknown['RAW']['share']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('readout', 'readout-sha256', 'amendment', 'amendment-sha256', 'addendum', 'addendum-sha256', 'run-directory'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--output', help='write the verdict once here')
    args = p.parse_args(argv)
    value = verdict(read_bound(args.readout, args.readout_sha256), read_bound(args.amendment, args.amendment_sha256), args.run_directory,
        readout_sha256=args.readout_sha256, amendment_sha256=args.amendment_sha256, addendum=read_bound(args.addendum, args.addendum_sha256),
        addendum_sha256=args.addendum_sha256)
    text = json.dumps(value, indent=1, sort_keys=True) + '\n'
    if args.output:
        with open(args.output, 'x') as f:f.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':sys.exit(main())
