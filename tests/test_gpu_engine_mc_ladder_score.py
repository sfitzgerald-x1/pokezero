"""mc_ladder_score: rung choices scored by the gate bank's Monte Carlo truth (numpy only; no engine)."""
import json
import tempfile
import unittest
from pathlib import Path

from gpu_engine import mc_ladder_score as L

CAPPED = -0.25


def bank_row(kind, seed, actions, means, policy_best, value_best, **identity):
    policy = [0.] * 9;policy[policy_best] = 1.;q = [0.] * len(actions);q[actions.index(value_best)] = 1.
    return dict(kind=kind, seed=seed, actions=actions, outcomes=[[m] * 4 for m in means], policy=policy,
                baseline=dict(completed_q=q), **identity)


class LadderScoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory();d = Path(self.dir.name)
        rows = [bank_row('foulplay', 5, [0, 3, 6], [0., .5, -.5], 0, 3, ordinal=1, target=4),
                bank_row('foulplay', 5, [0, 1], [.2, .9], 1, 1, ordinal=2, target=7),
                bank_row('foulplay', 8, [2, 4], [.1, .3], 2, 4, ordinal=3, target=2),
                bank_row('self', 900, [0, 1], [.4, .1], 0, 1, ours='p2', target=3)]
        (d / 'roots-foulplay-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:3]))
        (d / 'roots-self-0.jsonl').write_text(json.dumps(rows[3]) + '\n')
        self.bank = L.load_bank(d)

    def tearDown(self):self.dir.cleanup()

    def rung(self, rows):
        path = Path(self.dir.name) / 'rung.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in rows));return path

    def test_r0_is_the_value_head_choice_and_scores_against_the_policy_by_battle(self):
        r0 = L.r0_choices(self.bank)
        self.assertEqual(r0[('foulplay', 1, 4)], 3);self.assertEqual(r0[('self', 900, 'p2', 3)], 1)
        s = L.score(self.bank, L.as_records(r0), r0, 'foulplay')
        self.assertEqual((s['scored'], s['coverage'], s['override_rate']), (3, 1.0, round(2 / 3, 4)))
        self.assertAlmostEqual(s['minus_policy']['mean'], (.5 + 0. + .2) / 3, places=5)
        self.assertEqual(s['minus_policy']['clusters'], 2)   # roots 1 and 2 share battle seed 5
        self.assertEqual(s['minus_policy_where_different']['roots'], 2);self.assertEqual(s['minus_r0']['mean'], 0.)
        self.assertAlmostEqual(L.score(self.bank, L.as_records(r0), r0, 'self')['minus_policy']['mean'], -.3, places=5)
        self.assertEqual(s['minus_policy_intent_to_treat'], s['minus_policy'])   # every root decided

    def test_rung_choices_guards_and_skips(self):
        rung = L.load_rung(self.rung([dict(kind='foulplay', ordinal=1, target=4, action=6, guard_tripped=True, greedy_action=0),
                                      dict(kind='foulplay', ordinal=2, target=7, action=5),
                                      dict(kind='self', seed=900, ours='p2', target=3, action=1)]))
        self.assertEqual(rung[('foulplay', 1, 4)]['action'], 0)   # guards on: the tripped root plays greedy
        tripped = dict(kind='foulplay', ordinal=1, target=4, action=6, guard_tripped=True, greedy_action=3)
        self.assertEqual(L.load_rung(self.rung([tripped]), 'off')[('foulplay', 1, 4)]['action'], 6)
        s = L.score(self.bank, L.load_rung(self.rung([tripped]), 'policy'), {}, 'foulplay')   # policy: D = 0 at the trip
        self.assertEqual((s['scored'], s['minus_policy']['mean'], s['guard_trips'], s['greedy_ne_policy']), (1, 0., 1, 1))
        s = L.score(self.bank, rung, L.r0_choices(self.bank), 'foulplay')
        self.assertEqual(s['skipped'], {'choice_not_legal': 1, 'not_decided': 1});self.assertEqual(s['scored'], 1)
        self.assertEqual((s['minus_r0']['mean'], s['guard_trips'], s['greedy_ne_policy']), (-.5, 1, 0))
        self.assertEqual(s['minus_policy_intent_to_treat']['roots'], 3)   # the decided root, plus the illegal and undecided ones at D = 0
        with self.assertRaises(ValueError):L.load_rung(self.rung([dict(kind='foulplay', ordinal=1, target=4, action=0)] * 2))
        with self.assertRaises(ValueError):L.load_rung(self.rung([dict(kind='foulplay', ordinal=1, target=4, action=0, guard_tripped=True)]))

    def test_echoed_identity_must_match_the_bank(self):
        rows = [dict(kind='foulplay', ordinal=1, target=4, action=3, actions=[6, 0, 3], seed=5),   # the echo is order-free
                dict(kind='foulplay', ordinal=2, target=7, action=1, actions=[0, 2]),
                dict(kind='foulplay', ordinal=3, target=2, action=4, seed=9)]
        s = L.score(self.bank, L.load_rung(self.rung(rows)), {}, 'foulplay')
        self.assertEqual((s['scored'], s['skipped']), (1, {'identity_mismatch': 2}))
        self.assertEqual(s['minus_policy_intent_to_treat']['roots'], 3)
        off = L.load_rung(self.rung([dict(kind='foulplay', ordinal=1, target=4, guard_tripped=True, greedy_action=0)]), 'off')
        self.assertEqual(L.score(self.bank, off, {}, 'foulplay')['guard_trips'], 1)   # counted when the row has no action too

    def test_a_bank_kind_must_not_mix_continuations(self):
        d = Path(self.dir.name)
        (d / 'roots-foulplay-1.jsonl').write_text(json.dumps(bank_row('foulplay', 6, [0, 1], [0., 0.], 0, 1, ordinal=9, target=1,
                                                                        continuation='champion_both_seats')) + '\n')
        with self.assertRaisesRegex(ValueError, 'mixes'):L.load_bank(d)

    def test_censored_choices_are_skipped_not_scored(self):
        row = self.bank[('foulplay', 3, 2)];row['outcomes'][1] = [CAPPED] * 4
        s = L.score(self.bank, L.as_records({('foulplay', 3, 2): 4}), {}, 'foulplay')
        self.assertEqual((s['scored'], s['skipped']['censored_choice']), (0, 1))

    def test_main_writes_a_report_per_rung_and_kind(self):
        out = Path(self.dir.name) / 'report.json'
        rung = self.rung([dict(kind='self', seed=900, ours='p2', target=3, action=0), dict(kind='self', seed=1, ours='p1', target=0, action=0)])
        L.main([self.dir.name, '--rung', f'R1={rung}', '--out', str(out)])
        report = json.loads(out.read_text());self.assertEqual(report['unmatched_rung_rows'], {'R1': 1})
        self.assertEqual(sorted(report['rungs']), ['R0', 'R1']);self.assertEqual(report['roots'], {'foulplay': 3, 'self': 1})
        self.assertEqual(report['rungs']['R1']['self']['override_rate'], 0.)
        with self.assertRaises(SystemExit):L.main([self.dir.name, '--rung', 'R0=x'])


if __name__ == '__main__':
    unittest.main()
