"""scripts/depthn_guard_equivalence.py on synthetic rows and decisions: the joins, the checks and the exit codes."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from pokezero import depthn_effect_guard as G
from scripts import depthn_guard_equivalence as E


def row(ordinal, index, m_own, m_opp, *, b_turn1=False, unavailable=False, rule=G.RULE, tripped=None, battle_id=None, owner='p1'):
    """A reach-live re-pricer row; `tripped` overrides the record's reach decision (a record inconsistent with its own M)."""
    trips = None if unavailable else max(m_own, m_opp) >= G.THRESHOLD
    if tripped is not None:trips = tripped
    record = dict(rule=rule, threshold=G.THRESHOLD, m_own=None if unavailable else m_own, m_opp=None if unavailable else m_opp,
                  b=b_turn1 or trips is not False, turn2_available=not unavailable,
                  reach=dict(m_own=m_own, m_opp=m_opp, tripped=True if unavailable else trips, unavailable=unavailable))
    return dict(ordinal=ordinal, decision_index=index, battle_id=battle_id or f'battle-{ordinal}', owner=owner, effect_guard=record)


def decision(ordinal, index, own, opp, *, b_turn1=False, battle_id=None, owner='p1'):
    """An old score's decision: own and opp are (lower, upper) bounds."""
    lower, upper = max(own[0], opp[0]) >= G.THRESHOLD, max(own[1], opp[1]) >= G.THRESHOLD
    return dict(ordinal=ordinal, decision_index=index, battle_id=battle_id or f'battle-{ordinal}', owner=owner,
                reach_turn2=dict(m_own_lower=own[0], m_own_upper=own[1], m_opp_lower=opp[0], m_opp_upper=opp[1],
                                 tripped=True if lower else False if not upper else None),
                reach_scope_tripped_lower=b_turn1 or lower, reach_scope_tripped_upper=b_turn1 or upper)


def run(rows, decisions, *extra):
    """main over the rows and decisions written to files: (exit code, the stdout report, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        r, d = Path(tmp) / 'rows-0.jsonl', Path(tmp) / 'decisions.jsonl'
        r.write_text(''.join(json.dumps(x) + '\n' for x in rows));d.write_text(''.join(json.dumps(x) + '\n' for x in decisions))
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            code = E.main(['--rows', str(r), '--decisions', str(d), *extra])
    return code, json.loads(stdout.getvalue()), stderr.getvalue()


class EquivalenceTests(unittest.TestCase):
    def cases(self):
        rows = [row(1, 0, .08, 0.), row(1, 1, .01, .02), row(1, 2, .03, .06), row(1, 3, .02, 0.), row(2, 0, 0., 0., unavailable=True), row(2, 1, .5, 0.),
                row(3, 0, 0., 0.)]
        decisions = [decision(1, 0, (.08, .08), (0., 0.)),   # lower trips: live trips
                     decision(1, 1, (.01, .01), (.02, .02)),   # upper passes: live passes
                     decision(1, 2, (.01, .04), (.0, .07)),   # ambiguous: the live trips inside the bounds
                     decision(1, 3, (0., .06), (0., 0.)),   # ambiguous: the live passes inside
                     decision(2, 0, (0., 0.), (0., 0.)),   # the live record is unavailable: not compared, not counted against coverage
                     decision(9, 9, (0., 0.), (0., 0.))]   # no new row
        return rows, decisions

    def test_a_consistent_run_passes(self):
        rows, decisions = self.cases()
        r = E.compare(rows, decisions)
        self.assertEqual((r['passed'], r['failed']), (True, []))
        self.assertEqual((r['lower']['rows'], r['lower']['agree'], r['upper']['rows'], r['upper']['agree']), (1, 1, 1, 1))
        self.assertEqual((r['bounds']['rows'], r['bounds']['within'], r['bounds']['outside'], r['bounds']['rows_outside']), (4, 4, 0, []))
        self.assertEqual((r['ambiguous']['rows'], r['ambiguous']['live_tripped'], r['ambiguous']['live_passed']), (2, 1, 1))
        self.assertEqual((r['coverage']['compared'], r['coverage']['available_both'], r['coverage']['share'], r['coverage']['passed']), (4, 4, 1., True))
        c = r['counts']
        self.assertEqual((c['rows'], c['decisions'], c['joined'], c['compared'], c['live_unavailable'], c['rows_without_decision'], c['decisions_without_row']),
            (7, 6, 5, 4, 1, 2, 1))
        self.assertEqual((r['scope']['lower_rows'], r['scope']['lower_agree']), (4, 3))   # the ambiguous row the live trips: lower scope clear

    def test_each_check_fails_on_its_own(self):
        rows, decisions = self.cases()
        def failed(changed):
            r = E.compare([changed.get((x['ordinal'], x['decision_index']), x) for x in rows], decisions);self.assertFalse(r['passed']);return r
        r = failed({(1, 0): row(1, 0, .08, 0., tripped=False)})   # the lower bound trips, the live record passes
        self.assertEqual((r['failed'], r['lower']['agree'], r['lower']['failures'][0]['decision_index']), (['lower'], 0, 0))
        r = failed({(1, 1): row(1, 1, .01, .02, tripped=True)})   # the upper bound passes, the live record trips
        self.assertEqual((r['failed'], r['upper']['agree']), (['upper'], 0))
        r = failed({(1, 2): row(1, 2, .05, .06)})   # ambiguous: m_own above its upper bound .04
        self.assertEqual((r['failed'], r['bounds']['outside'], r['bounds']['rows_outside'][0]['decision_index']), (['bounds'], 1, 2))
        self.assertAlmostEqual(r['bounds']['rows_outside'][0]['deltas']['own'], .01)

    def test_m_lies_within_the_bounds_on_every_compared_row(self):
        """Not only the ambiguous rows: a lower-trip and an upper-pass row whose M leaves its bounds by more than 1e-12 fail,
        listed with signed deltas, largest first; within 1e-12 passes."""
        rows, decisions = self.cases()
        r = E.compare([row(1, 0, .08 + 5e-13, 0.)] + rows[1:], decisions)
        self.assertEqual((r['passed'], r['bounds']['within']), (True, 4))
        moved = [row(1, 0, .08 - 1e-11, 0.), row(1, 1, .01, .02 + 1e-9)] + rows[2:]   # still trips / still passes: lower and upper agree
        r = E.compare(moved, decisions)
        self.assertEqual((r['failed'], r['lower']['agree'], r['upper']['agree'], r['bounds']['outside']), (['bounds'], 1, 1, 2))
        first, second = r['bounds']['rows_outside']
        self.assertEqual(((first['decision_index'], list(first['deltas'])), (second['decision_index'], list(second['deltas']))), ((1, ['opp']), (0, ['own'])))
        self.assertAlmostEqual(first['deltas']['opp'], 1e-9, delta=1e-15);self.assertAlmostEqual(second['deltas']['own'], -1e-11, delta=1e-15)
        self.assertEqual((r['bounds']['max_abs_delta'], first['battle_id'], first['owner']), (first['max_abs_delta'], 'battle-1', 'p1'))
        code, report, stderr = run(moved, decisions)
        self.assertEqual((code, report['failed']), (1, ['bounds']))
        self.assertIn('bounds: 2 row(s) outside [lower, upper]', stderr);self.assertIn('ordinal 1 decision 1', stderr)

    def test_the_coverage_floor(self):
        """compared must reach 99% of the joined rows available on both sides; an unavailable side does not count against it."""
        decisions = [decision(k, 0, (0., 0.), (0., 0.)) for k in range(100)]
        rows = [row(k, 0, 0., 0.) for k in range(100)]
        old_rule = lambda x: dict(x, effect_guard=dict(x['effect_guard'], rule='any_cell'))
        r = E.compare([old_rule(rows[0])] + rows[1:], decisions)   # 99 of 100
        self.assertEqual((r['passed'], r['coverage']['share'], r['coverage']['uncompared']), (True, .99, {'live_rule_not_reach': 1}))
        r = E.compare([old_rule(x) for x in rows[:2]] + rows[2:], decisions)   # 98 of 100: every other check passes
        self.assertEqual((r['failed'], r['coverage']['share'], r['coverage']['compared'], r['coverage']['available_both']), (['coverage'], .98, 98, 100))
        r = E.compare([row(k, 0, 0., 0., unavailable=True) for k in range(2)] + rows[2:], decisions)   # unavailable live captures
        self.assertEqual((r['passed'], r['coverage']['available_both'], r['counts']['live_unavailable']), (True, 98, 2))
        lost = [dict(d, reach_turn2=dict(d['reach_turn2'], m_own_lower=None)) for d in decisions[:3]] + decisions[3:]   # unavailable bounds
        r = E.compare(rows, lost);self.assertEqual((r['passed'], r['coverage']['available_both'], r['counts']['bounds_unavailable']), (True, 97, 3))

    def test_rows_joined_but_nothing_compared_exit_1(self):
        rows, decisions = self.cases()
        nothing = [row(1, k, 0., 0., unavailable=True) for k in range(4)]
        code, report, stderr = run(nothing, decisions)
        self.assertEqual((code, report['counts']['joined'], report['counts'].get('compared', 0)), (1, 4, 0))
        self.assertEqual(report['failed'], ['compared', 'coverage']);self.assertIn('FAILED: compared, coverage', stderr)
        old_rule = [dict(x, effect_guard=dict(x['effect_guard'], rule='any_cell')) for x in rows]   # a run on the wrong tree
        code, report, _ = run(old_rule, decisions)
        self.assertEqual((code, report['failed'], report['coverage']['uncompared']), (1, ['compared', 'coverage'], {'live_rule_not_reach': 4}))

    def test_the_join_checks_battle_and_owner(self):
        rows, decisions = self.cases()
        for changed, field in ((row(1, 2, .03, .06, battle_id='battle-7'), 'battle_id'), (row(1, 2, .03, .06, owner='p2'), 'owner'),
                               (dict(row(1, 2, .03, .06), battle_id=None), 'battle_id')):
            bad = rows[:2] + [changed] + rows[3:]
            with self.assertRaises(E.JoinMismatch) as caught:E.compare(bad, decisions)
            self.assertEqual([(m['ordinal'], m['decision_index'], sorted(set(m) - {'ordinal', 'decision_index'})) for m in caught.exception.mismatches],
                             [(1, 2, [field])], field)
            code, report, stderr = run(bad, decisions)
            self.assertEqual((code, report['error'], report['mismatches'][0][field][1]), (3, 'join_mismatch', decisions[2][field]))
            self.assertIn('JOIN MISMATCH', stderr)
        seated = lambda x, seat: dict({k: v for k, v in x.items() if k != 'owner'}, seat=seat)   # without the owner: the seat
        self.assertTrue(E.compare([seated(x, 0) for x in rows], [seated(x, 0) for x in decisions])['passed'])
        with self.assertRaises(E.JoinMismatch):E.compare([seated(x, 0) for x in rows], [seated(x, 1) for x in decisions])
        with self.assertRaises(E.JoinMismatch):E.compare([seated(x, 0) for x in rows], decisions)   # no owner and no seat to check

    def test_main_and_exit_codes(self):
        rows, decisions = self.cases()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report.json'
            code, report, stderr = run(rows, decisions, '--out', str(out))
            self.assertEqual((code, report['passed'], stderr), (0, True, ''));self.assertTrue(json.loads(out.read_text())['passed'])
        self.assertEqual(run([row(1, 1, .3, 0.)], decisions)[0], 1)   # outside its bounds and the upper bound passes
        self.assertEqual(run([row(7, 7, 0., 0.)], decisions)[0], 2)   # nothing joined


if __name__ == '__main__':
    unittest.main()
