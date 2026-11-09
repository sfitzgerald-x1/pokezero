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
- Alive bucketing (ALIVE_BANDS): 'coarse' (the default, the final-fairness amendment's 1-2 / 3-4 / 5-6) or 'exact' ({1}, {2}, {3-4},
  {5-6}: screen-r2's alive-bands amendment, evals/depthn-foulplay-dn2w16-amendment-screen-r2-alive-bands-20261109.json), per side; the
  stratum label, the sparse-cell merge and the tolerance are otherwise the same. The default's output is unchanged.
- Windowed diagnosis (window_diagnosis; the CLI's `window`): the flag's own population, every cell the panel logged in
  state/completions.jsonl from index `start` (a fairness window starts at its acknowledgement's log length) up to `end` (or the first
  prefix whose completed pairs reach `pairs`, as the panel's checkpoint was evaluated; or a recorded flag's window and pairs), screen
  pairs only: the declared reading (budget stratum x turn band, the panel's own cell_moves and arm_fairness rule; it reproduces the
  flag's weighted ratio) and the finer reading under the chosen bucketing. Fairness data only: no score is read.
- Screen-r3's rule (`--rule r3`; depthn_live_fairness.RULE_R3: budget stratum x turn band x our and their exact alive bands, the sparse
  merge at 120 moves per arm within each (budget, ours, theirs) group, and minimum support: a merged cell enters the weighted ratio only
  with 120 rated moves in each arm). final_gate_r3: fairness.reading under RULE_R3 over every COMPLETE cell of both arms within the first
  `limit` pairs, the moves as the panel reads them (depthn_live_panel.cell_moves under the rule); within when |difference| <= 5%; it
  reports each arm's supported share. `window --rule r3` reads the declared reading (and the finer one) under RULE_R3. The alive counts
  come from depthn_live_fairness.round_states (rounds here is that function), so every reader shares one definition.
"""
import argparse
import sys
import collections
import json
from pathlib import Path
from statistics import fmean

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from pokezero import depthn_live_fairness as fairness  # noqa: E402
from pokezero.phase_d_saved_roots import band as turn_band  # noqa: E402

TOLERANCE = .05
AGE_BUCKETS = ((0, 120), (120, 300), (300, 600), (600, None))
ALIVE_BANDS = dict(coarse=('1-2', '3-4', '5-6'), exact=fairness.ALIVE_BANDS)   # per side; 'coarse' is the default
RULES = dict(default=None, r3=fairness.RULE_R3)   # --rule: the default (RULE: the declared reading; the coarse/exact final gate) or screen-r3's


def alive_bucket(n, bands='coarse'):
    """A side's alive count's bucket: coarse 1-2 / 3-4 / 5-6 (the default), exact 1 / 2 / 3-4 / 5-6 (0 joins the lowest bucket)."""
    if bands == 'coarse':return '1-2' if n <= 2 else '3-4' if n <= 4 else '5-6'
    if bands == 'exact':return fairness.alive_band(n)
    raise ValueError(f'alive bands in {sorted(ALIVE_BANDS)} required, got {bands!r}')


def age_bucket(seconds):
    for lo, hi in AGE_BUCKETS:
        if hi is None or seconds < hi:return f'{lo}+' if hi is None else f'{lo}-{hi}'


rounds = fairness.round_states   # per round of the owner's capture: (turn, our alive, their alive); None without a capture


def add_cell(arms, by_age, run, result, *, with_age=False, bands='coarse'):
    """One cell result's rated Foul Play moves into its arm of `arms` (fairness.pool's shape) with the finer label (and the age
    bucket when with_age), and their rates into `by_age`."""
    cell = result.get('cell') or {}
    think = (result.get('source_game') or {}).get('opponent_think') or ();arms[cell['arm']]['moves'] += len(think)
    per_round = rounds(result.get('owning_wire') or run / 'owning-wire' / f"{cell['ordinal']}.jsonl");age = 0.
    for row in think:
        rate, stratum, number = row.get('iterations_per_budget_second'), row.get('stratum'), row.get('round')
        if type(rate) in (int, float) and not isinstance(rate, bool) and type(stratum) is str:
            if per_round is not None and type(number) is int and 0 <= number < len(per_round):turn, ours, theirs = per_round[number]
            else:turn, ours, theirs = (number or 0) + 1, 6, 6;arms[cell['arm']]['band_from_round'] += 1
            label = f'{stratum}|ours {alive_bucket(ours, bands)}|theirs {alive_bucket(theirs, bands)}'
            if with_age:label += f'|age {age_bucket(age)}'
            arms[cell['arm']]['rated'].append((label, turn_band(max(1, turn)), float(rate)))
            by_age[(age_bucket(age), cell['arm'])].append(float(rate))
        for key in ('wait_seconds', 'overlap_seconds'):
            if type(row.get(key)) in (int, float):age += row[key]


def pooled(run_directory, search_arm, limit, *, with_age=False, bands='coarse'):
    """Both arms' rated Foul Play moves as fairness.pool's shape, with the finer label (and the age bucket when with_age)."""
    run = Path(run_directory);arms = {'RAW': dict(rated=[], moves=0, band_from_round=0), search_arm: dict(rated=[], moves=0, band_from_round=0)}
    by_age = collections.defaultdict(list)
    for path in sorted((run / 'results').glob('*.json')):
        result = json.loads(path.read_bytes());cell = result.get('cell') or {}
        if (result.get('status') != 'COMPLETE' or cell.get('arm') not in arms or type(cell.get('pair_ordinal')) is not int
                or not 0 <= cell['pair_ordinal'] < limit):continue
        add_cell(arms, by_age, run, result, with_age=with_age, bands=bands)
    return arms, by_age


SUPPORT_KEYS = ('supported_cells', 'unsupported_cells', 'supported_moves', 'supported_share')   # a reading under minimum support (RULE_R3)


def summary(reading):
    return dict(weighted_ratio=reading['weighted_ratio'], difference=reading['difference'], unstratified_ratio=reading['unstratified_ratio'],
        moves=reading['moves'], cells=len(reading['cells']), excluded_cells=reading['excluded_cells'],
        **{k: reading[k] for k in SUPPORT_KEYS if k in reading})


def final_gate(run_directory, search_arm, limit, *, bands='coarse'):
    """The whole-run finer reading, its verdict (within: |difference| <= 5%) and the descriptive process-age check. `bands`: the alive
    bucketing ('coarse', the default, whose output is unchanged; 'exact' adds alive_bands to the result and names it in the rule)."""
    if bands not in ALIVE_BANDS:raise ValueError(f'alive bands in {sorted(ALIVE_BANDS)} required, got {bands!r}')
    arms, by_age = pooled(run_directory, search_arm, limit, bands=bands)
    finer = summary(fairness.reading(arms['RAW'], arms[search_arm]))
    within = finer['difference'] is not None and abs(finer['difference']) <= TOLERANCE
    aged = summary(fairness.reading(*(lambda a: (a['RAW'], a[search_arm]))(pooled(run_directory, search_arm, limit, with_age=True, bands=bands)[0])))
    rates = {f'{bucket} s': {arm: dict(moves=len(by_age[(f'{lo}+' if hi is None else f'{lo}-{hi}', arm)]),
             mean=fmean(by_age[(bucket, arm)]) if by_age[(bucket, arm)] else None) for arm in ('RAW', search_arm)}
             for lo, hi in AGE_BUCKETS for bucket in [f'{lo}+' if hi is None else f'{lo}-{hi}']}
    age_flag = (aged['difference'] is not None and abs(aged['difference']) > TOLERANCE) and within
    buckets = '' if bands == 'coarse' else ' (exact alive bands 1 / 2 / 3-4 / 5-6 per side)'
    out = dict(rule=f'finer-stratified weighted ratio over the whole run (budget stratum x turn band x our and their alive buckets{buckets}); '
        f'|difference| > {TOLERANCE:.0%}: VOID_FAIRNESS', limit=limit, tolerance=TOLERANCE, finer=finer, within=within,
        process_age=dict(scope='descriptive; age approximated by the summed wait + overlap seconds of the game\'s earlier Foul Play rounds',
            age_controlled=aged, rate_by_age_bucket=rates, report_to_lead=age_flag))
    if bands != 'coarse':out['alive_bands'] = bands
    return out


def final_gate_r3(run_directory, search_arm, limit):
    """Screen-r3's final gate (module docstring): fairness.reading under RULE_R3 over every COMPLETE cell of both arms within the first
    `limit` pairs, each cell's moves as the panel reads them under that rule; within when |difference| <= 5% (none: not within)."""
    from pokezero import depthn_live_panel as panel
    run, rule = Path(run_directory), fairness.RULE_R3;cells = {'RAW': [], search_arm: []}
    for path in sorted((run / 'results').glob('*.json')):
        result = json.loads(path.read_bytes());cell = result.get('cell') or {}
        if (result.get('status') != 'COMPLETE' or cell.get('arm') not in cells or type(cell.get('pair_ordinal')) is not int
                or not 0 <= cell['pair_ordinal'] < limit):continue
        cells[cell['arm']].append(panel.cell_moves(run, cell['ordinal'], result, rule))
    value = fairness.reading(fairness.pool(cells['RAW']), fairness.pool(cells[search_arm]), rule)
    within = value['difference'] is not None and abs(value['difference']) <= TOLERANCE
    return dict(rule=f'screen-r3 arm fairness (RULE_R3: budget stratum x turn band x our and their exact alive bands 1 / 2 / 3-4 / 5-6; '
        f'sparse turn bands merged at {rule["min_cell_moves"]} moves per arm; a cell enters the weighted ratio only with '
        f'{rule["min_support_moves"]} rated moves in each arm) over the whole run; |difference| > {TOLERANCE:.0%}: VOID_FAIRNESS',
        arm_fairness=json.loads(json.dumps(rule)), limit=limit, tolerance=TOLERANCE, within=within, difference=value['difference'],
        supported_share=value['supported_share'], reading=value)


def window_entries(run_directory, start, *, end=None, pairs=None, search_arm, limit):
    """The panel's completion log entries [start:end) with pair_ordinal < limit (the screen), as a fairness window holds them; with
    `pairs` instead of `end`, the shortest such prefix whose completed pairs (both arms) reach `pairs` (where the panel evaluated that
    checkpoint). ValueError when the log is shorter."""
    from pokezero import depthn_live_panel as panel
    log = panel.completions(Path(run_directory))
    if type(start) is not int or not 0 <= start <= len(log):raise ValueError(f'start {start!r} outside the completion log (0..{len(log)})')
    if end is not None and pairs is not None:raise ValueError('end or pairs, not both')
    if end is not None:
        if type(end) is not int or not start <= end <= len(log):raise ValueError(f'end {end!r} outside {start}..{len(log)}')
        return [e for e in log[start:end] if e['pair_ordinal'] < limit], end
    entries = []
    for index in range(start, len(log)):
        if log[index]['pair_ordinal'] < limit:entries.append(log[index])
        if pairs is not None and panel.completed_pairs(entries, search_arm) >= pairs:return entries, index + 1
    if pairs is not None:raise ValueError(f'the window from {start} never reaches {pairs} completed pairs')
    return entries, len(log)


def window_diagnosis(run_directory, search_arm, start, *, end=None, pairs=None, limit=500, bands='coarse', rule=None):
    """The fairness reading of one window of the completion log (module docstring): the declared reading (the panel's cell_moves,
    `rule` the declared arm_fairness, default depthn_live_fairness.RULE) and the finer reading under `bands`, both read under `rule`
    (another rule than RULE is named in the result's arm_fairness). Fairness data only."""
    from pokezero import depthn_live_panel as panel
    if bands not in ALIVE_BANDS:raise ValueError(f'alive bands in {sorted(ALIVE_BANDS)} required, got {bands!r}')
    run = Path(run_directory);entries, stop = window_entries(run, start, end=end, pairs=pairs, search_arm=search_arm, limit=limit)
    rule = fairness.RULE if rule is None else rule
    declared = {arm: fairness.pool([panel.cell_moves(run, e['ordinal'], rule=rule) for e in entries if e['arm'] == arm]) for arm in ('RAW', search_arm)}
    arms = {'RAW': dict(rated=[], moves=0, band_from_round=0), search_arm: dict(rated=[], moves=0, band_from_round=0)}
    by_age = collections.defaultdict(list)
    for e in entries:
        if e['arm'] in arms:add_cell(arms, by_age, run, panel.read_result(run, e['ordinal']), bands=bands)
    coarse = fairness.reading(declared['RAW'], declared[search_arm], rule);finer = fairness.reading(arms['RAW'], arms[search_arm], rule)
    out = dict(scope='fairness data only (Foul Play iteration rates, alive counts and turns); no score is read',
        window=dict(start=start, end=stop, cells=len(entries), completed_pairs=panel.completed_pairs(entries, search_arm), limit=limit),
        alive_bands=bands, tolerance=TOLERANCE,
        declared=dict(summary(coarse), coverage_difference=coarse['coverage_difference']),
        finer=dict(summary(finer), within=finer['difference'] is not None and abs(finer['difference']) <= TOLERANCE, cell_ratios=[
            dict(stratum=c['stratum'], bands=c['bands'], raw_moves=c['raw_moves'], search_moves=c['search_moves'], ratio=c['ratio'],
                 **({'supported': c['supported']} if 'supported' in c else {})) for c in finer['cells']]))
    if rule != fairness.RULE:out['arm_fairness'] = json.loads(json.dumps(rule))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    gate = sub.add_parser('final', help='the whole-run final gate')
    window = sub.add_parser('window', help='a windowed drift diagnosis over the completion log')
    for q in (gate, window):
        q.add_argument('--run-directory', required=True);q.add_argument('--arm', required=True, help='the search arm (e.g. DN2W16)')
        q.add_argument('--alive-bands', choices=sorted(ALIVE_BANDS), help='alive bucketing (default coarse: 1-2 / 3-4 / 5-6)')
        q.add_argument('--rule', choices=sorted(RULES), default='default', help='the arm fairness rule (default: RULE; r3: screen-r3\'s RULE_R3)')
        q.add_argument('--output', help='also write the result here (a new file)')
    gate.add_argument('--limit', type=int, default=500, help='the readout\'s paired cells (default 500, the screen)')
    window.add_argument('--start', type=int, help='the window\'s first completion-log index (a flag\'s window: its acknowledgement\'s log length)')
    window.add_argument('--end', type=int, help='the window\'s end index (exclusive; default the log\'s end)')
    window.add_argument('--pairs', type=int, help='end the window where its completed pairs reach this (the flag\'s checkpoint)')
    window.add_argument('--flag', type=int, help='take start and pairs from state/fairness/screen/flag-N.json')
    window.add_argument('--limit', type=int, default=500, help='screen pairs only: pair_ordinal < limit (default 500)')
    args = p.parse_args(argv)
    if args.command == 'final' and args.rule == 'r3':
        if args.alive_bands not in (None, 'exact'):p.error('final --rule r3 reads RULE_R3\'s exact alive bands')
        value = final_gate_r3(args.run_directory, args.arm, args.limit)
    elif args.command == 'final':value = final_gate(args.run_directory, args.arm, args.limit, bands=args.alive_bands or 'coarse')
    else:
        start, pairs, flag = args.start, args.pairs, None
        if args.flag is not None:
            flag = json.loads((Path(args.run_directory) / 'state' / 'fairness' / 'screen' / f'flag-{args.flag}.json').read_bytes())
            if start is not None and start != flag['window']:p.error(f"--start {start} is not flag {args.flag}'s window {flag['window']}")
            start = flag['window'];pairs = flag['pairs'] if pairs is None and args.end is None else pairs
        if start is None:p.error('--start or --flag required')
        value = window_diagnosis(args.run_directory, args.arm, start, end=args.end, pairs=pairs, limit=args.limit,
                                 bands=args.alive_bands or 'coarse', rule=RULES[args.rule])
        if flag is not None:
            recorded = (flag.get('reading') or {}).get('weighted_ratio')
            value['flag'] = dict(number=args.flag, recorded_weighted_ratio=recorded, reproduced=recorded == value['declared']['weighted_ratio'])
    text = json.dumps(value, indent=1, sort_keys=True) + '\n'
    if args.output:
        with open(args.output, 'x') as stream:stream.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':sys.exit(main())
