#!/usr/bin/env python3
"""TARGET_ARGMAX replay check: the live search's choice under --root-operator TARGET_ARGMAX against the RR scorer's. CPU only.

Input: --rows, re-pricer rows (scripts/depthn_rr_reprice.py rows-*.jsonl) produced with --root-operator TARGET_ARGMAX (every
row's replay names the operator: replay.root_operator TARGET_ARGMAX, with target_fallback). Each row is one decision sent to
the live search; a root is compared when its replay is COMPLETE.

Per compared root, from the row's own fields, through the scorer's own code (scripts/depthn_rr_score.py: normalize, recompute,
target_argmax, guard_choices) on one side and the live stack's on the other:
- scorer: q = recompute(row).average.q (search_hedge.averaging over the row's re-priced worlds, as the scorer prices every
  operator); raw, fallback = target_argmax(replay.policy_target, root.root_legal, root.root_logits, q); played = guard_choices'
  choice for that operator: greedy (live.greedy_action_index) where the own-recharge guard (root.own_moves) or
  UNREPRESENTED_EFFECT (the row's effect_guard record: its (b), or (c) on raw: depthn_effect_guard.trips) trips, else raw.
- live: raw = replay.action_index (the re-priced live search's choice), fallback = replay.target_fallback, played = what the
  live actor plays on that response (pokezero.depthn_live_actor.guarded_action: its searched-action checks, the own-recharge
  guard on root.own_moves, the record's tripped; greedy when any trips).
A root mismatches on any of: raw (the choices), fallback, played, guard_causes (the scorer's causes against the actor's), q
(recomputed q against replay.q_values after a JSON round trip, the scorer's parity rule) and guard_choice (the record's
`choice`, the action its (c) was evaluated on, against the live raw choice). Mismatches must be 0; up to EXAMPLES are listed.
Reported: roots compared, mismatches by kind with examples, the fallback count, guard trips by arm (on the scorer's side:
own_recharge, unrepresented_effect and within it b and c; the records' own arms), the override rate against greedy (raw and
played; raw against averaging's own argmax too), the rows by replay status and the battles covered.
Rows not compared: a replay that is not COMPLETE (the live stack plays greedy there; counted by status and cause) and a COMPLETE
root without the inputs (no live greedy action, a recompute the scorer refuses: counted by reason, a coverage problem).
Exit codes: 0 no mismatch; 1 any mismatch; 2 an input problem (unreadable or malformed rows, a duplicated (ordinal,
decision_index), a row not re-priced under TARGET_ARGMAX); 3 a coverage problem without a mismatch (no root compared, a
COMPLETE root left uncompared, or --expect-battles not met). The report is JSON on stdout (and --out, a new file).

  python scripts/depthn_target_argmax_replay_check.py --rows OUT/shard-*/rows-*.jsonl --expect-battles 160 [--out REPORT.json]
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
from pokezero import depthn_effect_guard as effect_guard  # noqa: E402
from pokezero import depthn_live_actor as actor  # noqa: E402
from pokezero.depthn_root_operator import TARGET_ARGMAX  # noqa: E402

SCHEMA = 'pokezero.depthn-target-argmax-replay-check.v1'
ROW_SCHEMA = 'pokezero.depthn-rr-reprice.row.v1'   # scripts/depthn_rr_reprice.ROW_SCHEMA (not imported: that module loads JAX)
EXAMPLES = 20
KINDS = ('raw', 'fallback', 'played', 'guard_causes', 'q', 'guard_choice')
OPERATOR = 'target_argmax'   # the scorer's SETTINGS name


class InputProblem(ValueError):
    """Rows that cannot be checked as TARGET_ARGMAX replays (exit 2)."""


def read_rows(paths):
    """(path, line number, row) of every non-empty line; InputProblem on a file or line that does not read as a JSON object."""
    for path in paths:
        try:stream = open(path, 'rb')
        except OSError as error:raise InputProblem(f'{path}: {error}') from None
        with stream:
            for number, raw in enumerate(stream, 1):
                if not raw.strip():continue
                try:row = json.loads(raw)
                except ValueError as error:raise InputProblem(f'{path}:{number}: not JSON ({error})') from None
                if type(row) is not dict:raise InputProblem(f'{path}:{number}: a JSON object required')
                yield path, number, row


def _int(x):return type(x) is int


def compare_root(row):
    """dict(scorer=..., live=..., kinds=[mismatching KINDS], guard=...) of one COMPLETE row; scorer.Unscorable when the scorer
    leaves it out (its recompute refuses, or no legal live greedy action: 'greedy_missing', as score_decision)."""
    decision = scorer.normalize(row);replay = row['replay']
    base = scorer.recompute(decision);greedy = decision['greedy_action'];legal = base.legal
    if not _int(greedy) or not 0 <= greedy < 9 or not legal[greedy]:raise scorer.Unscorable('greedy_missing')
    record = decision['effect_guard'] if type(decision['effect_guard']) is dict else None
    target = replay.get('policy_target')
    raw, fallback = scorer.target_argmax(target, legal, decision['root_logits'], base.average.q)
    played, causes = scorer.guard_choices({OPERATOR: raw}, greedy, decision['own_moves'], record)
    mine = dict(raw=raw, fallback=fallback, played=played[OPERATOR], guard_causes=causes[OPERATOR] or [])
    response = dict(status=replay.get('status'), cause=replay.get('cause'), action_index=replay.get('action_index'), q_values=replay.get('q_values'),
                    effect_guard=row.get('effect_guard'))
    action, cause, guards, _ = actor.guarded_action(response, SimpleNamespace(legal_action_mask=legal), decision['own_moves'], greedy)
    live = dict(raw=replay.get('action_index'), fallback=replay.get('target_fallback'), played=greedy if action is None else action,
                guard_causes=list(guards), cause=cause)
    q_equal = scorer.q_json_equal(base.average.q, decision['replay_q'])
    guard_choice = None if record is None else record.get('choice')
    differs = {k: mine[k] != live[k] for k in ('raw', 'fallback', 'played', 'guard_causes')}
    differs.update(q=not q_equal, guard_choice=record is not None and guard_choice != live['raw'])
    kinds = [k for k in KINDS if differs[k]]
    averaging = scorer.argmax(base.average.q, [a for a in range(9) if legal[a]])
    trips = None if record is None else effect_guard.trips(record, raw)
    guard = dict(record=record is not None, b=None if trips is None else trips['b'], c=None if trips is None else trips['c'],
                 arms=[] if record is None else list(record.get('arms') or ()), causes=mine['guard_causes'])
    return dict(scorer=mine, live=live, kinds=kinds, greedy=greedy, averaging=averaging, q_equal=q_equal, guard_choice=guard_choice, guard=guard)


def check(rows, *, expect_battles=None, examples=EXAMPLES):
    """The report over (path, line, row) triples (module docstring); InputProblem on an input problem."""
    seen = set();statuses = collections.Counter();causes = collections.Counter();uncompared = collections.Counter();uncompared_rows = []
    battles, compared_battles = set(), set();mismatch_kinds = collections.Counter();mismatches = [];mismatched = n = 0
    fallback = collections.Counter();overrides = collections.Counter();guards = collections.Counter();arms = collections.Counter()
    for path, number, row in rows:
        where = f'{path}:{number}'
        if row.get('schema') != ROW_SCHEMA:raise InputProblem(f'{where}: a re-pricer row ({ROW_SCHEMA}) required, got schema {row.get("schema")!r}')
        key = (row.get('ordinal'), row.get('decision_index'))
        if not all(map(_int, key)):raise InputProblem(f'{where}: integer ordinal and decision_index required')
        if key in seen:raise InputProblem(f'{where}: duplicated (ordinal, decision_index) {key}')
        seen.add(key);battles.add(key[0])
        replay = row.get('replay')
        if type(replay) is not dict:raise InputProblem(f'{where}: no replay')
        if replay.get('root_operator') != TARGET_ARGMAX:
            raise InputProblem(f'{where}: replay.root_operator is {replay.get("root_operator")!r}: re-price with --root-operator {TARGET_ARGMAX}')
        statuses[str(replay.get('status'))] += 1
        address = dict(ordinal=key[0], decision_index=key[1])
        if replay.get('status') != 'COMPLETE':causes[str(replay.get('cause'))] += 1;continue
        try:result = compare_root(row);reason = None
        except scorer.Unscorable as error:reason = error.reason
        except (KeyError, TypeError, ValueError, IndexError) as error:raise InputProblem(f'{where}: malformed row ({type(error).__name__}: {error})') from None
        if reason is not None:
            uncompared[reason] += 1
            if len(uncompared_rows) < examples:uncompared_rows.append(dict(address, reason=reason))
            continue
        n += 1;compared_battles.add(key[0]);s, live = result['scorer'], result['live']
        fallback['scorer'] += s['fallback'] is True;fallback['live'] += live['fallback'] is True
        overrides['raw_vs_greedy'] += s['raw'] != result['greedy'];overrides['played_vs_greedy'] += s['played'] != result['greedy']
        overrides['raw_vs_averaging'] += s['raw'] != result['averaging'];overrides['played_vs_raw'] += s['played'] != s['raw']
        g = result['guard'];guards['record_missing'] += not g['record']
        guards['own_recharge'] += scorer.OWN_RECHARGE in g['causes'];guards['unrepresented_effect'] += scorer.UNREPRESENTED in g['causes']
        guards['b'] += bool(g['b']);guards['c'] += bool(g['c']);guards['c_without_b'] += bool(g['c']) and not g['b']
        guards['any'] += bool(g['causes'])
        for arm in g['arms']:arms[arm] += 1
        if result['kinds']:
            mismatched += 1
            for kind in result['kinds']:mismatch_kinds[kind] += 1
            if len(mismatches) < examples:
                mismatches.append(dict(address, kinds=result['kinds'], scorer=s, live=live, greedy=result['greedy'], averaging=result['averaging'],
                                       q_equal=result['q_equal'], guard_choice=result['guard_choice'], guard=g))
    rate = lambda k: overrides[k] / n if n else None
    coverage_problems = []
    if not n:coverage_problems.append('no root compared')
    if uncompared:coverage_problems.append(f'{sum(uncompared.values())} COMPLETE root(s) not compared')
    if expect_battles is not None and len(battles) != expect_battles:coverage_problems.append(f'{len(battles)} battles, {expect_battles} expected')
    return dict(schema=SCHEMA, operator=TARGET_ARGMAX, rows=len(seen), battles=len(battles), compared_battles=len(compared_battles),
                expect_battles=expect_battles, statuses=dict(statuses), not_complete_causes=dict(causes), roots_compared=n,
                uncompared=dict(uncompared), uncompared_examples=uncompared_rows, mismatches=mismatched, mismatch_kinds=dict(mismatch_kinds),
                mismatch_examples=mismatches, fallback=dict(scorer=fallback['scorer'], live=fallback['live']),
                guard_trips=dict(guards, arms=dict(arms)),
                override=dict(counts=dict(overrides), raw_vs_greedy=rate('raw_vs_greedy'), played_vs_greedy=rate('played_vs_greedy'),
                              raw_vs_averaging=rate('raw_vs_averaging'), guarded_share=rate('played_vs_raw')),
                coverage_problems=coverage_problems)


def exit_code(report):
    """1 on any mismatch, else 3 on a coverage problem, else 0 (an input problem is 2, before any report)."""
    return 1 if report['mismatches'] else 3 if report['coverage_problems'] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--rows', nargs='+', required=True, help='the re-pricer rows-*.jsonl of a --root-operator TARGET_ARGMAX replay')
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
    return code


if __name__ == '__main__':sys.exit(main())
