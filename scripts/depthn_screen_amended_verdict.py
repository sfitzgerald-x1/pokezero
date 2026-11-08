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
- both bounds (the addendum evals/depthn-foulplay-dn2w16-amendment-both-bounds-20261108.json): harm needs the known-pairs upper bound < 0
  AND the mirror upper bound < 0; the bar needs the known-pairs lower bound > 0 AND the conservative lower bound > 0, so the amendment
  only adds robustness to unknowns and never makes a claim easier than the declared estimator;
- decision: a void, a held fairness flag or an incomplete run keeps the official decision; at the screen, STOP_HARM if harm (both), else
  STOP_BREAKAGE if any amended breakage, else CONTINUE_TO_2000; at the bar, BAR_MET when both lower bounds are > 0 with no amended
  breakage, else BAR_NOT_MET;
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
SCREEN_CELLS, BAR_CELLS = 500, 2000   # the declaration's screen_paired_cells and max_paired_cells
PINNED = dict(amendment='c4ff30a25a573f90fdd02957b40fb8307d8b87b56dfbfe60d85ea662b18e8b3b',   # the committed files (the CLI refuses others)
              addendum='dbce9a54223969a213a3e584fd2e01bbe259ebad7530678fbba7cd38ecd9539f',
              both='98e6120f30565c0b40d423504aa0d07a2929b2207551a5c901adb7579d361e13')
CAUSES = (('round cap', 'round_cap'), ('foul-play challenge', 'foulplay_challenge_timeout'), ('foul-play choice', 'foulplay_choice_timeout'))


def read_bound(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:raise ValueError(f'{path}: sha256 differs from the one given')
    return json.loads(raw)


def cause(error):
    text = str(error or '')
    return next((name for needle, name in CAUSES if needle in text), 'other')


CELL_SCHEMA = 'depthn-foulplay-panel.cell.v1'
LOST = 'claimed_without_result'


def cells(run_directory, arm, limit, declaration_sha256=None):
    """{pair_ordinal: (status, score, error)} for the arm's cells within the first `limit` pairs: each result file (of this panel and
    declaration, when declaration_sha256 is given), and each claim without a result as (LOST, None, None), as the official arm summary
    counts it."""
    out = {}
    for path in sorted(Path(run_directory, 'results').glob('*.json')):
        value = json.loads(path.read_bytes());cell = value.get('cell') or {}
        if cell.get('arm') == arm and type(cell.get('pair_ordinal')) is int and 0 <= cell['pair_ordinal'] < limit:
            if declaration_sha256 is not None and (value.get('schema') != CELL_SCHEMA or value.get('declaration_sha256') != declaration_sha256
                    or path.stem != str(cell.get('ordinal'))):raise ValueError(f'{path}: a result of another panel, declaration or cell')
            if cell['pair_ordinal'] in out:raise ValueError(f'two results for {arm} pair {cell["pair_ordinal"]}')
            out[cell['pair_ordinal']] = (value.get('status'), value.get('score'), value.get('error'))
    for path in sorted(Path(run_directory, 'claims').glob('*.json')):
        cell = (json.loads(path.read_bytes()).get('cell') or {})
        if cell.get('arm') == arm and type(cell.get('pair_ordinal')) is int and 0 <= cell['pair_ordinal'] < limit and cell['pair_ordinal'] not in out:
            out[cell['pair_ordinal']] = (LOST, None, None)
    return out


def clustered_interval(raw, search, limit, *, search_unknown, raw_unknown):
    """The seed-clustered t interval (two-sided 90%) with every UNKNOWN or missing search cell scored search_unknown and RAW cell
    raw_unknown (the verdict: 0 and 1; the harm mirror: 1 and 0)."""
    from scipy.stats import t
    score = lambda cell, imputed: cell[1] if cell is not None and cell[0] == 'COMPLETE' else imputed
    deltas = [score(search.get(p), search_unknown) - score(raw.get(p), raw_unknown) for p in range(limit)]
    clusters = [(deltas[s] + deltas[s + 1]) / 2 for s in range(0, limit - 1, 2)]
    if len(clusters) < 2 or statistics.variance(clusters) == 0:
        return dict(mean=statistics.fmean(clusters) if clusters else None, interval90=None, clusters=len(clusters))
    mean = statistics.fmean(clusters)
    radius = float(t.ppf(T_QUANTILE, len(clusters) - 1)) * math.sqrt(statistics.variance(clusters) / len(clusters))
    return dict(mean=mean, interval90=[max(-1., mean - radius), min(1., mean + radius)], clusters=len(clusters))


def conservative_interval(raw, search, limit):return clustered_interval(raw, search, limit, search_unknown=0., raw_unknown=1.)


def mirror_interval(raw, search, limit):return clustered_interval(raw, search, limit, search_unknown=1., raw_unknown=0.)


def verdict(readout, amendment, run_directory, *, readout_sha256, amendment_sha256, addendum, addendum_sha256, both, both_sha256):
    if amendment.get('schema') != AMENDMENT_SCHEMA or readout.get('declaration_sha256') != amendment.get('declaration_sha256'):
        raise ValueError('an amendment of this readout\'s declaration required')
    if (addendum.get('schema') != AMENDMENT_SCHEMA or addendum.get('declaration_sha256') != amendment['declaration_sha256']
            or (addendum.get('addendum_to') or {}).get('sha256') != amendment_sha256):
        raise ValueError('the harm addendum to this amendment required')
    if (both.get('schema') != AMENDMENT_SCHEMA or both.get('declaration_sha256') != amendment['declaration_sha256']
            or sorted(x.get('sha256') for x in both.get('addendum_to') or ()) != sorted([amendment_sha256, addendum_sha256])):
        raise ValueError('the both-bounds addendum to this amendment and its harm addendum required')
    if (readout.get('tree') or {}).get('commit') != amendment['run']['tree_commit']:raise ValueError('a readout of the amended run\'s tree required')
    arm, limit = readout['arm'], readout['paired_cells']
    if limit not in (SCREEN_CELLS, BAR_CELLS) or readout['decision'] not in KEPT and (readout['bar'] is None) != (limit == SCREEN_CELLS):
        raise ValueError('a screen or bar readout required')   # a kept decision (void, held flag) carries no bar at either size
    screen = limit == SCREEN_CELLS;rule = amendment['rule'];cap = rule['unknown_share_max_per_arm']
    arms = {name: cells(run_directory, name, limit, amendment['declaration_sha256']) for name in ('RAW', arm)}
    unknown = {}
    for name, found in arms.items():
        official = readout['arms'][name]
        complete = sum(c[0] == 'COMPLETE' for c in found.values());lost = sum(c[0] == LOST for c in found.values())
        results = len(found) - lost;score_mean = (statistics.fmean([c[1] for c in found.values() if c[0] == 'COMPLETE']) if complete else None)
        mine = dict(results=results, complete=complete, unknown=len(found) - complete, claimed_without_result=lost,
            not_attempted=limit - len(found), score_mean=score_mean)
        theirs = {k: official.get(k) for k in mine}
        if mine != theirs:raise ValueError(f'{name}: the run directory\'s cells differ from the readout\'s {theirs} (found {mine})')
        bad = {p: c for p, c in found.items() if c[0] != 'COMPLETE'}
        by_cause = {}
        for _, (status, _, error) in sorted(bad.items()):
            kind = LOST if status == LOST else cause(error);by_cause[kind] = by_cause.get(kind, 0) + 1
        unknown[name] = dict(cells=len(bad), share=len(bad) / limit, by_cause=dict(sorted(by_cause.items())), pairs=sorted(bad),
            results=results, not_attempted=limit - len(found))
    paired = readout.get('paired')
    if paired is not None:   # the run's own known pairs and full-roster bounds, recomputed: the files are the readout's
        score = lambda name, p: arms[name][p][1] if p in arms[name] and arms[name][p][0] == 'COMPLETE' else None
        known = [p for p in range(limit) if score('RAW', p) is not None and score(arm, p) is not None]
        low = statistics.fmean([(0. if score(arm, p) is None else score(arm, p)) - (1. if score('RAW', p) is None else score('RAW', p)) for p in range(limit)])
        high = statistics.fmean([(1. if score(arm, p) is None else score(arm, p)) - (0. if score('RAW', p) is None else score('RAW', p)) for p in range(limit)])
        if paired.get('known_pairs') != len(known) or paired.get('full_roster_identification_bounds') != [low, high]:
            raise ValueError('the run directory\'s pairs differ from the readout\'s')
    breakage = [b for b in readout['breakage'] if b != 'unknown_cells']
    if any(u['share'] > cap for u in unknown.values()):breakage.append('unknown_share')
    void = paired is None   # VOID_LATENCY computes no screen statistic: neither do we
    bound = None if void else conservative_interval(arms['RAW'], arms[arm], limit)
    lower = None if void or bound['interval90'] is None else bound['interval90'][0]
    mirror = None if void else mirror_interval(arms['RAW'], arms[arm], limit)
    upper = None if void or mirror['interval90'] is None else mirror['interval90'][1]
    known = None if void else paired.get('interval90')   # the declared estimator's known-pairs interval
    harm = known is not None and known[1] < 0 and upper is not None and upper < 0   # both: declared and mirror
    met = known is not None and known[0] > 0 and lower is not None and lower > 0   # both: declared and conservative
    if readout['decision'] in KEPT:decision = readout['decision']
    elif screen:decision = 'STOP_HARM' if harm else 'STOP_BREAKAGE' if breakage else 'CONTINUE_TO_2000'
    else:decision = 'BAR_MET' if met and not breakage else 'BAR_NOT_MET'
    paired = paired or {}
    return dict(schema=VERDICT_SCHEMA, amendment_sha256=amendment_sha256, addendum_sha256=addendum_sha256, both_sha256=both_sha256,
        readout_sha256=readout_sha256, tree_commit=readout['tree']['commit'], bar_met=None if screen else met,
        arm=arm, paired_cells=limit, official_decision=readout['decision'], official_breakage=readout['breakage'], official_harm=readout['harm'],
        decision=decision, breakage=breakage, harm=harm,
        conservative=None if void else dict(rule='UNKNOWN or missing: search 0, RAW 1; seed-clustered t, two-sided 90%', **bound, lower_bound=lower),
        harm_mirror=None if void else dict(rule='UNKNOWN or missing: search 1, RAW 0; seed-clustered t, two-sided 90%; harm only if the upper bound < 0', **mirror,
            upper_bound=upper, known_pairs_upper_bound=None if not (paired.get('interval90')) else paired['interval90'][1]),
        known_pairs=dict(scope='descriptive (the declared estimator)', interval90=paired.get('interval90'), known_pairs=paired.get('known_pairs'),
            complete_seed_difference=paired.get('complete_seed_difference'),
            full_roster_identification_bounds=paired.get('full_roster_identification_bounds')),
        unknown=dict(max_share_per_arm=cap, arms=unknown, share_difference=unknown[arm]['share'] - unknown['RAW']['share']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('readout', 'readout-sha256', 'amendment', 'amendment-sha256', 'addendum', 'addendum-sha256', 'both', 'both-sha256', 'run-directory'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--output', help='write the verdict once here')
    args = p.parse_args(argv)
    for name, given in (('amendment', args.amendment_sha256), ('addendum', args.addendum_sha256), ('both', args.both_sha256)):
        if given != PINNED[name]:raise SystemExit(f'--{name}-sha256 must be the committed file\'s {PINNED[name]}')
    amendment = read_bound(args.amendment, args.amendment_sha256)
    if str(Path(args.run_directory).resolve()) != amendment['run']['run_directory']:raise SystemExit('--run-directory must be the amended run\'s')
    value = verdict(read_bound(args.readout, args.readout_sha256), read_bound(args.amendment, args.amendment_sha256), args.run_directory,
        readout_sha256=args.readout_sha256, amendment_sha256=args.amendment_sha256, addendum=read_bound(args.addendum, args.addendum_sha256),
        addendum_sha256=args.addendum_sha256, both=read_bound(args.both, args.both_sha256), both_sha256=args.both_sha256)
    text = json.dumps(value, indent=1, sort_keys=True) + '\n'
    if args.output:
        with open(args.output, 'x') as f:f.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':sys.exit(main())
