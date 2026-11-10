#!/usr/bin/env python3
"""NOISE_GATED_K2 replay check: the live search's choice under --root-operator NOISE_GATED_K2 against the RR scorer's. CPU only.

Input: --rows, re-pricer rows (scripts/depthn_rr_reprice.py rows-*.jsonl) produced with --root-operator NOISE_GATED_K2 (every
row's replay names the operator: replay.root_operator NOISE_GATED_K2, with noise_gate). Each row is one decision sent to the
live search; a root is compared when its replay is COMPLETE, after the live client's own payload validation
(depthn_target_argmax_replay_check.live_response: a malformed COMPLETE payload is an input problem).

Per compared root, from the row's own (re-priced) fields, through the scorer's own code (scripts/depthn_rr_score.py: normalize,
recompute, per_world_q, noise_gated, guard_choices; the first three as depthn_rr_score.operators calls them) on one side and the
live stack's on the other:
- scorer: a* = averaging's choice over the row's worlds (recompute(row).average), qw = per_world_q(its grids), g = the argmax of
  root.root_logits over root.root_legal, lowest index on ties (the scorer's greedy_argmax: the greedy the live search has);
  raw = noise_gated(qw, a*, g, 2), gate = dict(passed=raw != g, greedy=g, candidate=a*); played = guard_choices' choice for
  noise_gated:k=2 against the live greedy (live.greedy_action_index, the actor's, as the scorer and the actor guard): that
  greedy where the own-recharge guard (root.own_moves) or UNREPRESENTED_EFFECT (the row's effect_guard: (b), or (c) on raw)
  trips, else raw.
- live: raw = replay.action_index, gate = replay.noise_gate, played = what the live actor plays on that response
  (pokezero.depthn_live_actor.guarded_action against its greedy, live.greedy_action_index; that greedy when any guard trips).
A root mismatches on any of: raw, gate, played, guard_causes (the scorer's causes against the actor's), q (recomputed q
against replay.q_values after a JSON round trip, the scorer's parity rule) and guard_choice (the record's `choice`, the
action its (c) was evaluated on, against the live raw choice). Mismatches must be 0; up to EXAMPLES are listed.
The RR scorer's own noise_gated:k=2 uses the live actor's greedy (live.greedy_action_index) as g, which the search never sees:
`rr_greedy` reports, apart from the mismatches and the exit code, the roots where that greedy differs from the root logits'
argmax, and where the scorer's choice under it (raw; played, under the same guards) differs from the live one, with examples.
Reported: roots compared, mismatches by kind with examples, the gate (passed, a* == g, failed), the override rate against the
live greedy (raw and played; and the gate pass rate, raw against g) over every compared root, over the unguarded roots (the
record's (b) false: the definitive RR's primary population) and over the roots the live actor's guards did not trip, guard
trips by arm, rr_greedy, the rows by replay status and the battles covered.
Rows not compared: a replay that is not COMPLETE (the live stack plays greedy there; counted by status and cause) and a COMPLETE
root without the inputs (no live greedy action, a recompute the scorer refuses: counted by reason, a coverage problem).
Exit codes: 0 no mismatch; 1 any mismatch; 2 an input problem (unreadable or malformed rows, a duplicated (ordinal,
decision_index), a row not re-priced under NOISE_GATED_K2, a COMPLETE payload the live client refuses); 3 a coverage problem
without a mismatch (no root compared, a COMPLETE root left uncompared, or --expect-battles not met). The report is JSON on
stdout (and --out, a new file).

  python scripts/depthn_noise_gated_replay_check.py --rows OUT/shard-*/rows-*.jsonl --expect-battles 160 [--out REPORT.json]
"""
import argparse
import collections
import json
from pathlib import Path
import sys
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / 'src')):
    if _path not in sys.path:sys.path.insert(0, _path)

from scripts import depthn_rr_score as scorer  # noqa: E402
from scripts.depthn_target_argmax_replay_check import EXAMPLES, ROW_SCHEMA, InputProblem, live_response, read_rows  # noqa: E402
from pokezero import depthn_effect_guard as effect_guard  # noqa: E402
from pokezero import depthn_live_actor as actor  # noqa: E402
from pokezero.depthn_root_operator import NOISE_GATED_K2, NOISE_K2  # noqa: E402

SCHEMA = 'pokezero.depthn-noise-gated-replay-check.v1'
KINDS = ('raw', 'gate', 'played', 'guard_causes', 'q', 'guard_choice')
OPERATOR = f'noise_gated:k={NOISE_K2}'   # the scorer's SETTINGS name
assert OPERATOR in scorer.SETTINGS


def _int(x):return type(x) is int


def scorer_choice(decision, base, greedy, live_greedy, record):
    """(raw, played, guard causes) of the scorer's noise_gated:k=2 on a recomputed row with `greedy` as g (guards against the
    live greedy, as guard_choices plays them)."""
    chosen = scorer.hedge.top_action(base.average.strategy)
    raw = scorer.noise_gated(scorer.per_world_q(base.grids), chosen, greedy, NOISE_K2)
    played, causes = scorer.guard_choices({OPERATOR: raw}, live_greedy, decision['own_moves'], record)
    return raw, played[OPERATOR], causes[OPERATOR] or []


def compare_root(row):
    """dict(scorer=..., live=..., kinds=[mismatching KINDS], rr=..., guard=...) of one COMPLETE row; scorer.Unscorable when the
    scorer leaves it out (its recompute refuses, or no legal live greedy action: 'greedy_missing', as score_decision)."""
    replay = row['replay'];response = live_response(row)
    decision = scorer.normalize(row)
    base = scorer.recompute(decision);live_greedy = decision['greedy_action'];legal = base.legal
    if not _int(live_greedy) or not 0 <= live_greedy < 9 or not legal[live_greedy]:raise scorer.Unscorable('greedy_missing')
    record = decision['effect_guard'] if type(decision['effect_guard']) is dict else None
    actions = [a for a in range(9) if legal[a]]
    greedy = scorer.argmax(decision['root_logits'], actions)   # the scorer's greedy_argmax
    chosen = scorer.hedge.top_action(base.average.strategy)
    raw, played, causes = scorer_choice(decision, base, greedy, live_greedy, record)
    mine = dict(raw=raw, gate=dict(passed=raw != greedy, greedy=greedy, candidate=chosen), played=played, guard_causes=causes)
    action, cause, guards, _ = actor.guarded_action(response, SimpleNamespace(legal_action_mask=legal), decision['own_moves'], live_greedy)
    live = dict(raw=replay.get('action_index'), gate=replay.get('noise_gate'), played=live_greedy if action is None else action,
                guard_causes=list(guards), cause=cause)
    q_equal = scorer.q_json_equal(base.average.q, decision['replay_q'])
    guard_choice = None if record is None else record.get('choice')
    differs = {k: mine[k] != live[k] for k in ('raw', 'gate', 'played', 'guard_causes')}
    differs.update(q=not q_equal, guard_choice=record is not None and guard_choice != live['raw'])
    kinds = [k for k in KINDS if differs[k]]
    rr_raw, rr_played, rr_causes = scorer_choice(decision, base, live_greedy, live_greedy, record)   # the RR scorer's own greedy
    rr = dict(greedy=live_greedy, raw=rr_raw, played=rr_played, guard_causes=rr_causes, greedy_differs=greedy != live_greedy,
              raw_differs=rr_raw != live['raw'], played_differs=rr_played != live['played'])
    trips = None if record is None else effect_guard.trips(record, raw)
    guard = dict(record=record is not None, b=None if trips is None else trips['b'], c=None if trips is None else trips['c'],
                 arms=[] if record is None else list(record.get('arms') or ()), causes=causes)
    return dict(scorer=mine, live=live, kinds=kinds, live_greedy=live_greedy, q_equal=q_equal, guard_choice=guard_choice, guard=guard, rr=rr)


class Rates:
    """Override counts over one population of compared roots."""
    def __init__(self):self.n = 0;self.c = collections.Counter()

    def add(self, result):
        s, greedy = result['scorer'], result['live_greedy'];self.n += 1
        self.c['raw_vs_greedy'] += s['raw'] != greedy;self.c['played_vs_greedy'] += s['played'] != greedy
        self.c['gate_passed'] += s['gate']['passed'];self.c['candidate_is_greedy'] += s['gate']['candidate'] == s['gate']['greedy']
        self.c['rr_raw_vs_greedy'] += result['rr']['raw'] != greedy;self.c['rr_played_vs_greedy'] += result['rr']['played'] != greedy

    def report(self):
        rate = lambda k: self.c[k] / self.n if self.n else None
        return dict(roots=self.n, counts=dict(self.c), raw_vs_greedy=rate('raw_vs_greedy'), played_vs_greedy=rate('played_vs_greedy'),
                    gate_pass=rate('gate_passed'), rr_raw_vs_greedy=rate('rr_raw_vs_greedy'), rr_played_vs_greedy=rate('rr_played_vs_greedy'))


def check(rows, *, expect_battles=None, examples=EXAMPLES):
    """The report over (path, line, row) triples (module docstring); InputProblem on an input problem."""
    seen = set();statuses = collections.Counter();causes = collections.Counter();uncompared = collections.Counter();uncompared_rows = []
    battles, compared_battles = set(), set();mismatch_kinds = collections.Counter();mismatches = [];mismatched = n = 0
    gate = collections.Counter();guards = collections.Counter();arms = collections.Counter()
    rates = dict(all=Rates(), unguarded=Rates(), not_tripped=Rates());rr = collections.Counter();rr_examples = []
    for path, number, row in rows:
        where = f'{path}:{number}'
        if row.get('schema') != ROW_SCHEMA:raise InputProblem(f'{where}: a re-pricer row ({ROW_SCHEMA}) required, got schema {row.get("schema")!r}')
        key = (row.get('ordinal'), row.get('decision_index'))
        if not all(map(_int, key)):raise InputProblem(f'{where}: integer ordinal and decision_index required')
        if key in seen:raise InputProblem(f'{where}: duplicated (ordinal, decision_index) {key}')
        seen.add(key);battles.add(key[0])
        replay = row.get('replay')
        if type(replay) is not dict:raise InputProblem(f'{where}: no replay')
        if replay.get('root_operator') != NOISE_GATED_K2:
            raise InputProblem(f'{where}: replay.root_operator is {replay.get("root_operator")!r}: re-price with --root-operator {NOISE_GATED_K2}')
        statuses[str(replay.get('status'))] += 1
        address = dict(ordinal=key[0], decision_index=key[1])
        if replay.get('status') != 'COMPLETE':causes[str(replay.get('cause'))] += 1;continue
        try:result = compare_root(row);reason = None
        except scorer.Unscorable as error:reason = error.reason
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            raise InputProblem(f'{where}: malformed row ({type(error).__name__}: {error})') from None
        if reason is not None:
            uncompared[reason] += 1
            if len(uncompared_rows) < examples:uncompared_rows.append(dict(address, reason=reason))
            continue
        n += 1;compared_battles.add(key[0]);s = result['scorer']
        gate['passed'] += s['gate']['passed'];gate['candidate_is_greedy'] += s['gate']['candidate'] == s['gate']['greedy']
        gate['failed'] += s['gate']['candidate'] != s['gate']['greedy'] and not s['gate']['passed']
        gate['live_passed'] += type(result['live']['gate']) is dict and result['live']['gate'].get('passed') is True
        g = result['guard'];guards['record_missing'] += not g['record']
        guards['own_recharge'] += scorer.OWN_RECHARGE in g['causes'];guards['unrepresented_effect'] += scorer.UNREPRESENTED in g['causes']
        guards['b'] += bool(g['b']);guards['c'] += bool(g['c']);guards['c_without_b'] += bool(g['c']) and not g['b']
        guards['any'] += bool(g['causes'])
        for arm in g['arms']:arms[arm] += 1
        rates['all'].add(result)
        if g['record'] and not g['b']:rates['unguarded'].add(result)
        if not result['live']['guard_causes']:rates['not_tripped'].add(result)
        r = result['rr'];rr['greedy_differs'] += r['greedy_differs'];rr['raw_differs'] += r['raw_differs'];rr['played_differs'] += r['played_differs']
        if g['record'] and not g['b']:rr['unguarded_played_differs'] += r['played_differs']
        if (r['greedy_differs'] or r['raw_differs'] or r['played_differs']) and len(rr_examples) < examples:
            rr_examples.append(dict(address, rr=r, live=dict(raw=result['live']['raw'], played=result['live']['played']), operator_greedy=s['gate']['greedy']))
        if result['kinds']:
            mismatched += 1
            for kind in result['kinds']:mismatch_kinds[kind] += 1
            if len(mismatches) < examples:
                mismatches.append(dict(address, kinds=result['kinds'], scorer=s, live=result['live'], live_greedy=result['live_greedy'],
                                       q_equal=result['q_equal'], guard_choice=result['guard_choice'], guard=g))
    coverage_problems = []
    if not n:coverage_problems.append('no root compared')
    if uncompared:coverage_problems.append(f'{sum(uncompared.values())} COMPLETE root(s) not compared')
    if expect_battles is not None and len(battles) != expect_battles:coverage_problems.append(f'{len(battles)} battles, {expect_battles} expected')
    return dict(schema=SCHEMA, operator=NOISE_GATED_K2, rows=len(seen), battles=len(battles), compared_battles=len(compared_battles),
                expect_battles=expect_battles, statuses=dict(statuses), not_complete_causes=dict(causes), roots_compared=n,
                uncompared=dict(uncompared), uncompared_examples=uncompared_rows, mismatches=mismatched, mismatch_kinds=dict(mismatch_kinds),
                mismatch_examples=mismatches, gate=dict(gate, pass_rate=gate['passed'] / n if n else None),
                guard_trips=dict(guards, arms=dict(arms)),
                override={name: r.report() for name, r in rates.items()},
                rr_greedy=dict(note='the RR scorer\'s own noise_gated:k=2 uses the live actor\'s greedy (live.greedy_action_index), which '
                               'the search never sees; reported, never a mismatch', **{k: rr[k] for k in
                               ('greedy_differs', 'raw_differs', 'played_differs', 'unguarded_played_differs')}, examples=rr_examples),
                coverage_problems=coverage_problems)


def exit_code(report):
    """1 on any mismatch, else 3 on a coverage problem, else 0 (an input problem is 2, before any report)."""
    return 1 if report['mismatches'] else 3 if report['coverage_problems'] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--rows', nargs='+', required=True, help='the re-pricer rows-*.jsonl of a --root-operator NOISE_GATED_K2 replay')
    parser.add_argument('--expect-battles', type=int, help='the battles (distinct ordinals) the rows must cover (the RR sample: 160)')
    parser.add_argument('--examples', type=int, default=EXAMPLES, help=f'mismatches (and uncompared roots) listed (default {EXAMPLES})')
    parser.add_argument('--out', help='also write the report here (a new file)')
    args = parser.parse_args(argv)
    try:report = check(read_rows(args.rows), expect_battles=args.expect_battles, examples=args.examples)
    except InputProblem as error:
        print(json.dumps(dict(schema=SCHEMA, error='input', detail=str(error)), indent=1, sort_keys=True))
        print(f'INPUT PROBLEM: {error}', file=sys.stderr);return 2
    code = exit_code(report);report['exit_code'] = code
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, 'x') as stream:stream.write(text + '\n')
    print(text)
    if report['mismatches']:print(f"MISMATCHES: {report['mismatches']} of {report['roots_compared']} roots ({report['mismatch_kinds']})", file=sys.stderr)
    if report['coverage_problems']:print('COVERAGE: ' + '; '.join(report['coverage_problems']), file=sys.stderr)
    o = report['override']['unguarded'];rr = report['rr_greedy']
    print(f"NOISE_GATED_K2: gate pass {report['gate']['pass_rate']}; unguarded roots {o['roots']}: override played {o['played_vs_greedy']} "
          f"(raw {o['raw_vs_greedy']}); RR greedy differs on {rr['greedy_differs']} roots, played choice differs on {rr['played_differs']}",
          file=sys.stderr)
    return code


if __name__ == '__main__':sys.exit(main())
