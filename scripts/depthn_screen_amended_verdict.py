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
- the final fairness gate (the addendum evals/depthn-foulplay-dn2w16-amendment-final-fairness-20261108.json, computed by
  depthn_fairness_final.final_gate over the whole run): when its finer-stratified |difference| > 5% the decision is VOID_FAIRNESS, checked
  before any interval (none is computed then);
- decision: a void, a held fairness flag or an incomplete run keeps the official decision; at the screen, STOP_HARM if harm (both), else
  STOP_BREAKAGE if any amended breakage, else CONTINUE_TO_2000; at the bar, BAR_MET when both lower bounds are > 0 with no amended
  breakage, else BAR_NOT_MET;
- every UNKNOWN cell is classified by cause (round_cap, foulplay_challenge_timeout, foulplay_choice_timeout, other) and arm.
The rule files are screen-r1's (their run fields name /out/screen-r1, its declaration and tree). Another run adopts them through a
run binding (--run-binding PATH SHA256, pinned in RUN_BINDINGS: the adoption record, e.g.
evals/depthn-foulplay-dn2w16-amendment-screen-r2-adoption-20261109.json): it adopts exactly the four pinned rule files by sha256 and
supplies the run directory, the declaration sha256 the readout and every result file must carry, and the readout's tree commit, in
place of the rule files' run fields; the rule logic is unchanged and still read from the rule files, whose own chain (each addendum's
declaration and addendum_to) is checked as before. Without --run-binding the script is screen-r1's, byte for byte.
A binding may require run amendments (RUN_AMENDMENTS, each given by --run-amendment PATH SHA256): screen-r2's requires its alive-bands
amendment (evals/depthn-foulplay-dn2w16-amendment-screen-r2-alive-bands-20261109.json), under which the final fairness gate
(depthn_fairness_final.final_gate) stratifies by exact alive bands 1 / 2 / 3-4 / 5-6 per side; the verdict checks the gate's result
says so. The tolerance and every other rule are unchanged.
A binding may instead declare its final fairness gate (final_fairness_rule, pinned in RUN_FINAL_RULES by the binding's sha256):
'RULE_R3', screen-r3's declared arm fairness rule (depthn_live_fairness.RULE_R3, pinned here as FINAL_RULES['RULE_R3']; the binding
carries it as arm_fairness), whose whole-run gate is depthn_fairness_final.final_gate_r3 (`final --rule r3`). The verdict then requires
that gate's result (its arm_fairness RULE_R3, at the readout's paired cells), refuses the coarse and exact-band gates' results and any run
amendment (the alive-bands one is screen-r2's), and voids (VOID_FAIRNESS) when it is not within, as the final-fairness addendum's gate does.
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
              both='98e6120f30565c0b40d423504aa0d07a2929b2207551a5c901adb7579d361e13',
              final='b2fccc253136b2bce3125cc773973c7186a0dff6a5ff8e6825a5618d6268050e')
ADOPTION_SCHEMA = 'depthn-foulplay-panel.amendment-adoption.v1'
RUN_BINDINGS = dict(screen_r2='5e53654231abc940461decde4cdfa6b85ce89f22e20ff0ea7f9e626ea1855a75')   # committed adoption records (the CLI refuses others)
# the run amendments each binding requires, {kind: committed sha256} (the CLI refuses a missing, extra or other one)
RUN_AMENDMENTS = {RUN_BINDINGS['screen_r2']: dict(alive_bands='fa7c32a51db058526e436a925ef2156b52ab9debd54881fae6829febcfaced53')}
EXACT_ALIVE_BANDS = ['1', '2', '3-4', '5-6']
# the final fairness gates a binding may declare (final_fairness_rule): screen-r3's arm fairness rule, depthn_live_fairness.RULE_R3 as committed
FINAL_RULES = dict(RULE_R3=dict(metric='ITERATIONS_PER_BUDGET_SECOND', strata='FOULPLAY_BUDGET_STRATUM_X_TURN_BAND_X_ALIVE_BANDS',
    combine='POOLED_COUNT_WEIGHTED_RATIO_OF_MEANS', reference='RAW', ratio_tolerance=.05, coverage_difference_max=.05, min_moves_per_arm=1000,
    min_pairs=50, checkpoint_pairs=50, rearm_margin=.025, min_cell_moves=120, alive_bands=['1', '2', '3-4', '5-6'], min_support_moves=120))
RUN_FINAL_RULES = {}   # {committed binding sha256: its final_fairness_rule} (the CLI refuses a binding declaring one that is not pinned here)
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


def run_fields(binding, binding_sha256, rule_shas):
    """(declaration_sha256, run_directory, tree_commit) of a run binding that adopts exactly the four rule files `rule_shas`;
    ValueError otherwise."""
    adopts = binding.get('adopts') if type(binding.get('adopts')) is list else []
    run = binding.get('run') if type(binding.get('run')) is dict else {}
    hexes = lambda value, n: type(value) is str and len(value) == n and all(c in '0123456789abcdef' for c in value)
    if (binding.get('schema') != ADOPTION_SCHEMA or len(adopts) != 4 or any(type(x) is not dict or x.get('change') != 'none' for x in adopts)
            or sorted(str(x.get('sha256')) for x in adopts) != sorted(rule_shas)):
        raise ValueError(f'a run binding adopting exactly the four rule files unchanged required ({binding_sha256})')
    if not (hexes(binding.get('declaration_sha256'), 64) and type(run.get('run_directory')) is str and hexes(run.get('tree_commit'), 40)):
        raise ValueError('a run binding with its declaration sha256, run directory and tree commit required')
    return binding['declaration_sha256'], run['run_directory'], run['tree_commit']


def final_rule(binding, binding_sha256):
    """The final fairness gate a run binding declares: None (none declared: the rule files' gate, or screen-r2's through its run
    amendment) or the name of a FINAL_RULES rule, which the binding also carries as arm_fairness; ValueError otherwise."""
    if 'final_fairness_rule' not in binding and 'arm_fairness' not in binding:return None
    name = binding.get('final_fairness_rule')
    if name not in FINAL_RULES or binding.get('arm_fairness') != FINAL_RULES[name]:
        raise ValueError(f'a run binding declaring final_fairness_rule {sorted(FINAL_RULES)} with that rule as its arm_fairness required '
                         f'({binding_sha256})')
    return name


def check_alive_bands(record, record_sha256, *, binding_sha256, declaration_sha256, run_directory, tree_commit):
    """ValueError unless `record` is the alive-bands amendment of this bound run: an amendment of its declaration, run directory and
    tree, an addendum to its binding, with the exact bands."""
    run = record.get('run') if type(record.get('run')) is dict else {}
    if (record.get('schema') != AMENDMENT_SCHEMA or record.get('declaration_sha256') != declaration_sha256
            or run.get('run_directory') != run_directory or run.get('tree_commit') != tree_commit
            or binding_sha256 not in {x.get('sha256') for x in record.get('addendum_to') or () if type(x) is dict}
            or (record.get('rule') or {}).get('alive_bands') != EXACT_ALIVE_BANDS):
        raise ValueError(f'the alive-bands amendment of the bound run required ({record_sha256})')


def verdict(readout, amendment, run_directory, *, readout_sha256, amendment_sha256, addendum, addendum_sha256, both, both_sha256,
            final, final_sha256, final_fairness, binding=None, binding_sha256=None, run_amendments=None):
    """The amended verdict (module docstring). `binding`: a run binding (the adoption record) whose run fields replace the rule
    files' (screen-r1's); None: screen-r1, unchanged. `run_amendments`: {kind: (record, sha256)}, the run amendments the binding
    requires (alive_bands: the final fairness gate's result must be the exact-bands one). A binding declaring final_fairness_rule
    (RULE_R3) takes no run amendments and needs that rule's gate result."""
    run_amendments = run_amendments or {};gate_rule = None
    if run_amendments and binding is None:raise ValueError('run amendments need their run binding')
    if binding is not None:
        declaration_sha256, bound_directory, tree_commit = run_fields(binding, binding_sha256, [amendment_sha256, addendum_sha256, both_sha256, final_sha256])
        if str(Path(run_directory).resolve()) != bound_directory:raise ValueError('the bound run\'s directory required')
        gate_rule = final_rule(binding, binding_sha256)
        if gate_rule is not None and run_amendments:
            raise ValueError(f'a binding with final_fairness_rule {gate_rule} takes no run amendments (the alive-bands one is screen-r2\'s)')
        if gate_rule is not None and (type(final_fairness) is not dict or final_fairness.get('arm_fairness') != FINAL_RULES[gate_rule]
                                      or 'alive_bands' in final_fairness or 'finer' in final_fairness):
            raise ValueError(f'the binding\'s final fairness gate ({gate_rule}: depthn_fairness_final.final_gate_r3) required, not the coarse '
                             'or exact-band gate')
        if set(run_amendments) - {'alive_bands'}:raise ValueError(f'unknown run amendments {sorted(set(run_amendments) - {"alive_bands"})}')
        if 'alive_bands' in run_amendments:
            check_alive_bands(*run_amendments['alive_bands'], binding_sha256=binding_sha256, declaration_sha256=declaration_sha256,
                              run_directory=bound_directory, tree_commit=tree_commit)
            if type(final_fairness) is not dict or final_fairness.get('alive_bands') != 'exact':
                raise ValueError('the alive-bands amendment needs the final fairness gate over exact alive bands')
    else:declaration_sha256, tree_commit = amendment.get('declaration_sha256'), (amendment.get('run') or {}).get('tree_commit')
    if amendment.get('schema') != AMENDMENT_SCHEMA or readout.get('declaration_sha256') != declaration_sha256:
        raise ValueError('an amendment of this readout\'s declaration required')
    if (addendum.get('schema') != AMENDMENT_SCHEMA or addendum.get('declaration_sha256') != amendment['declaration_sha256']
            or (addendum.get('addendum_to') or {}).get('sha256') != amendment_sha256):
        raise ValueError('the harm addendum to this amendment required')
    if (both.get('schema') != AMENDMENT_SCHEMA or both.get('declaration_sha256') != amendment['declaration_sha256']
            or sorted(x.get('sha256') for x in both.get('addendum_to') or ()) != sorted([amendment_sha256, addendum_sha256])):
        raise ValueError('the both-bounds addendum to this amendment and its harm addendum required')
    if (final.get('schema') != AMENDMENT_SCHEMA or final.get('declaration_sha256') != amendment['declaration_sha256']
            or not {amendment_sha256, addendum_sha256, both_sha256} <= {x.get('sha256') for x in final.get('addendum_to') or ()}):
        raise ValueError('the final-fairness addendum to this amendment chain required')
    if type(final_fairness) is not dict or type(final_fairness.get('within')) is not bool:raise ValueError('the final fairness gate\'s result required')
    if (readout.get('tree') or {}).get('commit') != tree_commit:raise ValueError('a readout of the amended run\'s tree required')
    arm, limit = readout['arm'], readout['paired_cells']
    if gate_rule is not None and final_fairness.get('limit') != limit:raise ValueError('the final fairness gate over the readout\'s paired cells required')
    if limit not in (SCREEN_CELLS, BAR_CELLS) or readout['decision'] not in KEPT and (readout['bar'] is None) != (limit == SCREEN_CELLS):
        raise ValueError('a screen or bar readout required')   # a kept decision (void, held flag) carries no bar at either size
    screen = limit == SCREEN_CELLS;rule = amendment['rule'];cap = rule['unknown_share_max_per_arm']
    arms = {name: cells(run_directory, name, limit, declaration_sha256) for name in ('RAW', arm)}
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
    unfair = final_fairness['within'] is False   # the final gate, before any interval
    void = paired is None or unfair   # VOID_LATENCY computes no screen statistic, nor does a run that fails the final gate
    bound = None if void else conservative_interval(arms['RAW'], arms[arm], limit)
    lower = None if void or bound['interval90'] is None else bound['interval90'][0]
    mirror = None if void else mirror_interval(arms['RAW'], arms[arm], limit)
    upper = None if void or mirror['interval90'] is None else mirror['interval90'][1]
    known = None if void else paired.get('interval90')   # the declared estimator's known-pairs interval
    harm = known is not None and known[1] < 0 and upper is not None and upper < 0   # both: declared and mirror
    met = known is not None and known[0] > 0 and lower is not None and lower > 0   # both: declared and conservative
    if unfair:decision = 'VOID_FAIRNESS'
    elif readout['decision'] in KEPT:decision = readout['decision']
    elif screen:decision = 'STOP_HARM' if harm else 'STOP_BREAKAGE' if breakage else 'CONTINUE_TO_2000'
    else:decision = 'BAR_MET' if met and not breakage else 'BAR_NOT_MET'
    paired = paired or {}
    bound_run = {} if binding is None else dict(run_binding_sha256=binding_sha256,
        run_binding=dict(declaration_sha256=declaration_sha256, run_directory=bound_directory, tree_commit=tree_commit),
        run_amendments={kind: sha for kind, (_, sha) in sorted(run_amendments.items())},
        **({} if gate_rule is None else dict(final_fairness_rule=gate_rule)))
    return dict(**bound_run, schema=VERDICT_SCHEMA, amendment_sha256=amendment_sha256, addendum_sha256=addendum_sha256, both_sha256=both_sha256,
        final_sha256=final_sha256, final_fairness=final_fairness,
        readout_sha256=readout_sha256, tree_commit=readout['tree']['commit'], bar_met=None if screen or void or readout['decision'] in KEPT else met,
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
    for name in ('readout', 'readout-sha256', 'amendment', 'amendment-sha256', 'addendum', 'addendum-sha256', 'both', 'both-sha256', 'final',
                 'final-sha256', 'run-directory'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--output', help='write the verdict once here')
    p.add_argument('--run-binding', nargs=2, metavar=('PATH', 'SHA256'), help='a run binding (adoption record) pinned in RUN_BINDINGS: '
        'its run directory, declaration sha256 and tree commit replace the rule files\' (screen-r1\'s)')
    p.add_argument('--run-amendment', nargs=2, action='append', default=[], metavar=('PATH', 'SHA256'),
        help='a run amendment the binding requires (RUN_AMENDMENTS; screen-r2: its alive-bands amendment)')
    args = p.parse_args(argv)
    for name, given in (('amendment', args.amendment_sha256), ('addendum', args.addendum_sha256), ('both', args.both_sha256), ('final', args.final_sha256)):
        if given != PINNED[name]:raise SystemExit(f'--{name}-sha256 must be the committed file\'s {PINNED[name]}')
    amendment = read_bound(args.amendment, args.amendment_sha256);binding = binding_sha256 = gate_rule = None;run_amendments = {}
    if args.run_binding is not None:
        path, binding_sha256 = args.run_binding
        if binding_sha256 not in RUN_BINDINGS.values():raise SystemExit(f'--run-binding must be a committed run binding: {sorted(RUN_BINDINGS.values())}')
        binding = read_bound(path, binding_sha256)
        try:directory = run_fields(binding, binding_sha256, list(PINNED.values()))[1];gate_rule = final_rule(binding, binding_sha256)
        except ValueError as error:raise SystemExit(str(error))
        if RUN_FINAL_RULES.get(binding_sha256) != gate_rule:
            raise SystemExit(f'--run-binding: its final_fairness_rule ({gate_rule}) must be the one pinned in RUN_FINAL_RULES '
                             f'({RUN_FINAL_RULES.get(binding_sha256)})')
        if str(Path(args.run_directory).resolve()) != directory:raise SystemExit('--run-directory must be the bound run\'s')
        if gate_rule is not None and args.run_amendment:
            raise SystemExit(f'--run-amendment: a binding with final_fairness_rule {gate_rule} takes none (the alive-bands amendment is screen-r2\'s)')
        required = RUN_AMENDMENTS.get(binding_sha256, {});kinds = {sha: kind for kind, sha in required.items()}
        given = [sha for _, sha in args.run_amendment]
        if sorted(given) != sorted(required.values()):
            raise SystemExit(f'--run-amendment: this binding requires exactly {sorted(required.items())}, got {sorted(given)}')
        run_amendments = {kinds[sha]: (read_bound(path, sha), sha) for path, sha in args.run_amendment}
    elif args.run_amendment:raise SystemExit('--run-amendment needs its --run-binding')
    elif str(Path(args.run_directory).resolve()) != amendment['run']['run_directory']:raise SystemExit('--run-directory must be the amended run\'s')
    from scripts import depthn_fairness_final   # the run's tree (pokezero) on the path
    readout = read_bound(args.readout, args.readout_sha256)
    if gate_rule == 'RULE_R3':   # the binding's declared gate (depthn_fairness_final final --rule r3)
        final_fairness = depthn_fairness_final.final_gate_r3(args.run_directory, readout['arm'], readout['paired_cells'])
    elif 'alive_bands' in run_amendments:   # the amended gate (exact alive bands)
        final_fairness = depthn_fairness_final.final_gate(args.run_directory, readout['arm'], readout['paired_cells'], bands='exact')
    else:final_fairness = depthn_fairness_final.final_gate(args.run_directory, readout['arm'], readout['paired_cells'])
    value = verdict(readout, read_bound(args.amendment, args.amendment_sha256), args.run_directory,
        readout_sha256=args.readout_sha256, amendment_sha256=args.amendment_sha256, addendum=read_bound(args.addendum, args.addendum_sha256),
        addendum_sha256=args.addendum_sha256, both=read_bound(args.both, args.both_sha256), both_sha256=args.both_sha256,
        final=read_bound(args.final, args.final_sha256), final_sha256=args.final_sha256, final_fairness=final_fairness,
        **({} if binding is None else dict(binding=binding, binding_sha256=binding_sha256, run_amendments=run_amendments)))
    text = json.dumps(value, indent=1, sort_keys=True) + '\n'
    if args.output:
        with open(args.output, 'x') as f:f.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':sys.exit(main())
