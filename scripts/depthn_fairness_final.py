"""The final fairness gate and the process-age check (evals/depthn-foulplay-dn2w16-amendment-final-fairness-20261108.json): Foul Play
rates only, never outcomes. Needs the run's tree on the path (pokezero.depthn_live_fairness, its reading and sparse-cell merge).

- Finer cells: budget stratum x turn band x our alive bucket x their alive bucket (1-2, 3-4, 5-6), from the owner's owning-wire capture
  (round r is the owner stream's r-th request line; the owner is its first request's side.id; alive = 6 - faints of that side so far).
  The stratum label carries (budget stratum, our bucket, their bucket) and fairness.reading merges sparse turn bands within each label, so
  the merge and the weights are the declared ones.
- Final gate: that reading over every COMPLETE cell of both arms within the first `limit` pairs; within when |difference| <= 5%.
- Process age (descriptive): a Foul Play move's age is the seconds since its game started, approximated by the summed wait + overlap
  seconds of that game's earlier Foul Play rounds (each round's request-forwarded-to-choice window); buckets 0-120, 120-300, 300-600, 600+.
  Reported: the rate by age bucket and arm, and the arm ratio with the age bucket added to the finer label.
"""
import collections
import json
from pathlib import Path
from statistics import fmean

from pokezero import depthn_live_fairness as fairness
from pokezero.phase_d_saved_roots import band as turn_band

TOLERANCE = .05
AGE_BUCKETS = ((0, 120), (120, 300), (300, 600), (600, None))


def alive_bucket(n):return '1-2' if n <= 2 else '3-4' if n <= 4 else '5-6'


def age_bucket(seconds):
    for lo, hi in AGE_BUCKETS:
        if hi is None or seconds < hi:return f'{lo}+' if hi is None else f'{lo}-{hi}'


def rounds(capture):
    """Per round of the owner's capture: (turn, our alive, their alive); None without a capture."""
    if capture is None or not Path(capture).exists():return None
    out, turn, faints, owner = [], 0, collections.Counter(), None
    for raw in Path(capture).read_bytes().split(b'\n'):
        try:row = json.loads(raw)
        except ValueError:continue
        if type(row) is not dict or row.get('event') != 'WIRE':continue
        for line in row.get('lines') or ():
            if line.startswith('|turn|'):
                value = line.split('|')[2];turn = int(value) if value.isascii() and value.isdigit() else turn
            elif line.startswith('|faint|'):faints[line.split('|')[2][:2]] += 1
            elif line.startswith('|request|'):
                if owner is None:
                    try:owner = (json.loads(line[len('|request|'):]).get('side') or {}).get('id')
                    except ValueError:owner = None
                other = 'p2' if owner == 'p1' else 'p1'
                out.append((turn, 6 - faints[owner], 6 - faints[other]))
    return out


def pooled(run_directory, search_arm, limit, *, with_age=False):
    """Both arms' rated Foul Play moves as fairness.pool's shape, with the finer label (and the age bucket when with_age)."""
    run = Path(run_directory);arms = {'RAW': dict(rated=[], moves=0, band_from_round=0), search_arm: dict(rated=[], moves=0, band_from_round=0)}
    by_age = collections.defaultdict(list)
    for path in sorted((run / 'results').glob('*.json')):
        result = json.loads(path.read_bytes());cell = result.get('cell') or {}
        if (result.get('status') != 'COMPLETE' or cell.get('arm') not in arms or type(cell.get('pair_ordinal')) is not int
                or not 0 <= cell['pair_ordinal'] < limit):continue
        think = (result.get('source_game') or {}).get('opponent_think') or ();arms[cell['arm']]['moves'] += len(think)
        per_round = rounds(result.get('owning_wire') or run / 'owning-wire' / f"{cell['ordinal']}.jsonl");age = 0.
        for row in think:
            rate, stratum, number = row.get('iterations_per_budget_second'), row.get('stratum'), row.get('round')
            if type(rate) in (int, float) and not isinstance(rate, bool) and type(stratum) is str:
                if per_round is not None and type(number) is int and 0 <= number < len(per_round):turn, ours, theirs = per_round[number]
                else:turn, ours, theirs = (number or 0) + 1, 6, 6;arms[cell['arm']]['band_from_round'] += 1
                label = f'{stratum}|ours {alive_bucket(ours)}|theirs {alive_bucket(theirs)}'
                if with_age:label += f'|age {age_bucket(age)}'
                arms[cell['arm']]['rated'].append((label, turn_band(max(1, turn)), float(rate)))
                by_age[(age_bucket(age), cell['arm'])].append(float(rate))
            for key in ('wait_seconds', 'overlap_seconds'):
                if type(row.get(key)) in (int, float):age += row[key]
    return arms, by_age


def summary(reading):
    return dict(weighted_ratio=reading['weighted_ratio'], difference=reading['difference'], unstratified_ratio=reading['unstratified_ratio'],
        moves=reading['moves'], cells=len(reading['cells']), excluded_cells=reading['excluded_cells'])


def final_gate(run_directory, search_arm, limit):
    """The whole-run finer reading, its verdict (within: |difference| <= 5%) and the descriptive process-age check."""
    arms, by_age = pooled(run_directory, search_arm, limit)
    finer = summary(fairness.reading(arms['RAW'], arms[search_arm]))
    within = finer['difference'] is not None and abs(finer['difference']) <= TOLERANCE
    aged = summary(fairness.reading(*(lambda a: (a['RAW'], a[search_arm]))(pooled(run_directory, search_arm, limit, with_age=True)[0])))
    rates = {f'{bucket} s': {arm: dict(moves=len(by_age[(f'{lo}+' if hi is None else f'{lo}-{hi}', arm)]),
             mean=fmean(by_age[(bucket, arm)]) if by_age[(bucket, arm)] else None) for arm in ('RAW', search_arm)}
             for lo, hi in AGE_BUCKETS for bucket in [f'{lo}+' if hi is None else f'{lo}-{hi}']}
    age_flag = (aged['difference'] is not None and abs(aged['difference']) > TOLERANCE) and within
    return dict(rule='finer-stratified weighted ratio over the whole run (budget stratum x turn band x our and their alive buckets); '
        f'|difference| > {TOLERANCE:.0%}: VOID_FAIRNESS', limit=limit, tolerance=TOLERANCE, finer=finer, within=within,
        process_age=dict(scope='descriptive; age approximated by the summed wait + overlap seconds of the game\'s earlier Foul Play rounds',
            age_controlled=aged, rate_by_age_bucket=rates, report_to_lead=age_flag))
