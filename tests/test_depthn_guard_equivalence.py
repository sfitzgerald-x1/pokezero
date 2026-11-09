"""scripts/depthn_guard_equivalence.py on synthetic rows and decisions: the joins, the three checks and the exit codes."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from pokezero import depthn_effect_guard as G
from scripts import depthn_guard_equivalence as E


def row(ordinal, index, m_own, m_opp, *, b_turn1=False, unavailable=False, rule=G.RULE):
    trips = None if unavailable else max(m_own, m_opp) >= G.THRESHOLD
    record = dict(rule=rule, threshold=G.THRESHOLD, m_own=None if unavailable else m_own, m_opp=None if unavailable else m_opp,
                  b=b_turn1 or trips is not False, reach=dict(m_own=m_own, m_opp=m_opp, tripped=True if unavailable else trips, unavailable=unavailable))
    return dict(ordinal=ordinal, decision_index=index, effect_guard=record)


def decision(ordinal, index, own, opp, *, b_turn1=False):
    """An old score's decision: own and opp are (lower, upper) bounds."""
    lower, upper = max(own[0], opp[0]) >= G.THRESHOLD, max(own[1], opp[1]) >= G.THRESHOLD
    return dict(ordinal=ordinal, decision_index=index, reach_turn2=dict(m_own_lower=own[0], m_own_upper=own[1], m_opp_lower=opp[0], m_opp_upper=opp[1],
                tripped=True if lower else False if not upper else None), reach_scope_tripped_lower=b_turn1 or lower, reach_scope_tripped_upper=b_turn1 or upper)


class EquivalenceTests(unittest.TestCase):
    def cases(self):
        rows = [row(1, 0, .08, 0.), row(1, 1, .01, .02), row(1, 2, .03, .06), row(1, 3, .02, 0.), row(2, 0, 0., 0., unavailable=True), row(2, 1, .5, 0.),
                row(3, 0, 0., 0.)]
        decisions = [decision(1, 0, (.08, .08), (0., 0.)),   # lower trips: live trips
                     decision(1, 1, (.01, .01), (.02, .02)),   # upper passes: live passes
                     decision(1, 2, (.01, .04), (.0, .07)),   # ambiguous: the live trips inside the bounds
                     decision(1, 3, (0., .06), (0., 0.)),   # ambiguous: the live passes inside
                     decision(2, 0, (0., 0.), (0., 0.)),   # the live record is unavailable: not compared
                     decision(9, 9, (0., 0.), (0., 0.))]   # no new row
        return rows, decisions

    def test_a_consistent_run_passes(self):
        rows, decisions = self.cases()
        r = E.compare(rows, decisions)
        self.assertTrue(r['passed'])
        self.assertEqual((r['lower']['rows'], r['lower']['agree'], r['upper']['rows'], r['upper']['agree']), (1, 1, 1, 1))
        self.assertEqual((r['ambiguous']['rows'], r['ambiguous']['within'], r['ambiguous']['live_tripped'], r['ambiguous']['live_passed']), (2, 2, 1, 1))
        c = r['counts']
        self.assertEqual((c['rows'], c['decisions'], c['joined'], c['compared'], c['live_unavailable'], c['rows_without_decision'], c['decisions_without_row']),
            (7, 6, 5, 4, 1, 2, 1))
        self.assertEqual((r['scope']['lower_rows'], r['scope']['lower_agree']), (4, 3))   # the ambiguous row the live trips: lower scope clear

    def test_each_check_fails_on_its_own(self):
        rows, decisions = self.cases()
        bad_lower = [row(1, 0, .01, 0.)] + rows[1:]   # the lower bound trips, the live passes
        r = E.compare(bad_lower, decisions);self.assertFalse(r['passed']);self.assertEqual((r['lower']['agree'], len(r['lower']['failures'])), (0, 1))
        bad_upper = rows[:1] + [row(1, 1, .3, 0.)] + rows[2:]   # the upper bound passes, the live trips
        r = E.compare(bad_upper, decisions);self.assertFalse(r['passed']);self.assertEqual(r['upper']['agree'], 0)
        outside = rows[:2] + [row(1, 2, .05, .06)] + rows[3:]   # m_own above its upper bound .04
        r = E.compare(outside, decisions);self.assertFalse(r['passed']);self.assertEqual((r['ambiguous']['within'], r['ambiguous']['failures'][0]['decision_index']), (1, 2))
        old_rule = [dict(rows[0], effect_guard=dict(rows[0]['effect_guard'], rule=None))] + rows[1:]
        self.assertEqual(E.compare(old_rule, decisions)['counts']['without_live_record'], 1)

    def test_main_and_exit_codes(self):
        rows, decisions = self.cases()
        with tempfile.TemporaryDirectory() as tmp:
            r, d, out = Path(tmp) / 'rows-0.jsonl', Path(tmp) / 'decisions.jsonl', Path(tmp) / 'report.json'
            r.write_text(''.join(json.dumps(x) + '\n' for x in rows));d.write_text(''.join(json.dumps(x) + '\n' for x in decisions))
            with redirect_stdout(io.StringIO()) as stdout:self.assertEqual(E.main(['--rows', str(r), '--decisions', str(d), '--out', str(out)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())['passed'], True);self.assertTrue(json.loads(out.read_text())['passed'])
            r.write_text(json.dumps(row(1, 1, .3, 0.)) + '\n')
            with redirect_stdout(io.StringIO()):self.assertEqual(E.main(['--rows', str(r), '--decisions', str(d)]), 1)
            r.write_text(json.dumps(row(7, 7, 0., 0.)) + '\n')
            with redirect_stdout(io.StringIO()):self.assertEqual(E.main(['--rows', str(r), '--decisions', str(d)]), 2)


if __name__ == '__main__':
    unittest.main()
