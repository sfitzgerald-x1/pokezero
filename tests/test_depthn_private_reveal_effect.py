"""scripts/depthn_private_reveal_effect.py on synthetic re-pricer rows: the joins, the reply-prior KL and flips, averaging recomputed on
the baseline grids with the redacted run's priors, the applied / control split and the determinism check."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from pokezero import v5_gumbel as original
from scripts import depthn_private_reveal_effect as E

KEPT = [True, True] + [False] * 7
LEGAL = [True, True] + [False] * 7
GRID = [[1., -1.] + [0.] * 7, [-1., 1.] + [0.] * 7] + [[0.] * 9] * 7   # our lane 0 beats reply 0, lane 1 beats reply 1


def logits(lean):
    """Reply logits leaning to reply 0 (lean > 0) or reply 1 (lean < 0)."""
    return [lean, -lean] + [0.] * 7


def world(i, lean, *, requested=True):
    return dict(world=i, priced=True, opponent_requested=requested, kept=KEPT if requested else [False] * 9,
                root_peer_logits=logits(lean), grid=GRID)


def row(ordinal, index, leans, *, redacted=(), applicable=None, b=False, guarded=None, battle_id=None, waiting=(), greedy=0, own_moves=None):
    """A decided re-pricer row: the live greedy action `greedy`, our lanes' moves `own_moves`, the guard record's (b) and guarded
    lanes (c)."""
    record = dict(rule='reach', b=b, own_guarded_lanes=guarded or [None] * 4)
    members = [dict(species=s, ability='pressure', slot=k) for k, s in enumerate(redacted)]
    return dict(ordinal=ordinal, decision_index=index, battle_id=battle_id or f'battle-{ordinal}', owner='p1',
                replay=dict(status='COMPLETE', cause=None), live=dict(greedy_action_index=greedy),
                root=dict(root_logits=[0.] * 9, root_legal=LEGAL, own_moves=own_moves), effect_guard=record,
                worlds=[world(i, x, requested=i not in waiting) for i, x in enumerate(leans)],
                private_reveals=dict(enabled=True, found=[], applicable=members if applicable is None else applicable, redacted=members, unmatched=[]))


def baseline_of(r):
    """The same root without the flag: nothing redacted, the same applicable members."""
    out = json.loads(json.dumps(r));p = out['private_reveals'];p.update(enabled=False, redacted=[]);return out


def run(baseline, redacted, *extra):
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / 'base.jsonl', Path(tmp) / 'flag.jsonl'
        a.write_text(''.join(json.dumps(x) + '\n' for x in baseline));b.write_text(''.join(json.dumps(x) + '\n' for x in redacted))
        out = Path(tmp) / 'report.json'
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            code = E.main(['--baseline', str(a), '--redacted', str(b), '--out', str(out), *extra])
        return code, stdout.getvalue(), stderr.getvalue(), json.loads(out.read_text())


class EffectTests(unittest.TestCase):
    def cases(self):
        """Every applied root's redacted reply priors move to reply 1, so averaging goes 0 -> 1 (greedy 0) unless stated."""
        beam = ['bodyslam', 'hyperbeam', None, None]
        flag = [row(1, 0, [2., 2.], redacted=['zapdos']),   # applied: averaging and the played action flip 0 -> 1
                row(1, 1, [1., 1.]),   # control
                row(2, 0, [2., 2.], waiting=(1,)),   # control with a waiting world
                row(3, 0, [2., 2.], redacted=['zapdos'], b=True),   # applied, the baseline's guard trips: not searched
                row(4, 0, [2., 2.], redacted=['zapdos'], own_moves=beam),   # applied: 1 is Hyper Beam, greedy 0: the own-recharge guard
                row(5, 0, [2., 2.], redacted=['zapdos'])]   # applied: the redacted record guards lane 1 ((c)): greedy plays
        base = [baseline_of(x) for x in flag]
        for i in (0, 4, 5):flag[i]['worlds'] = [world(0, -2.), world(1, -2.)]
        flag[3]['worlds'] = [world(0, -.5), world(1, 2.)];flag[5]['effect_guard']['own_guarded_lanes'] = [None, ['A', 'x'], None, None]
        return base, flag

    def test_a_redaction_moves_the_reply_prior_and_the_played_action_and_the_control_does_not(self):
        base, flag = self.cases()
        r = E.compare(base, flag)
        p, q = original._probabilities(tuple(logits(2.)), tuple(KEPT)), original._probabilities(tuple(logits(-2.)), tuple(KEPT))
        expected = math.fsum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a > 0)
        a = r['groups']['applied'];c = r['groups']['control'];every = r['groups']['all']
        self.assertEqual((a['roots'], a['worlds'], a['reply_top1_flips'], a['logits_bits_differ']), (4, 8, 7, 7))   # (3, 0)'s world 1 kept its logits
        self.assertAlmostEqual(a['kl_max'], expected, places=12)
        m = a['averaging']   # searched: (1, 0), (4, 0), (5, 0); averaging flips on all three, the played action on (1, 0) only;
        # (3, 0)'s mixed priors (one world moved a little) keep averaging's 0
        self.assertEqual((m['scored_roots'], m['searched_roots'], m['played_flips'], m['raw_flips'], m['raw_flips_all_scored'], m['b_changed']),
                         (4, 3, 1, 3, 3, 0))
        self.assertAlmostEqual(m['played_flip_rate'], 1 / 3);self.assertEqual(m['raw_flip_rate'], 1.)
        self.assertEqual((c['roots'], c['worlds'], c['kl_max'], c['kl_nonzero'], c['reply_top1_flips'], c['averaging']['played_flips'],
                          c['averaging']['raw_flips'], c['averaging']['searched_roots']), (2, 3, 0., 0, 0, 0, 0, 2))
        self.assertEqual((every['worlds'], every['averaging']['searched_roots']), (11, 5))
        self.assertEqual((r['worlds']['compared'], r['worlds']['waiting']), (11, 1))
        self.assertEqual((r['determinism']['passed'], r['determinism']['rows']), (True, []));self.assertEqual(r['redacted_species'], {'zapdos': 4})
        self.assertEqual((r['identity_mismatches']['count'], r['record_mismatches']['count'], r['counts']['joined']), (0, 0, 6))
        self.assertEqual(E.quantile([1., 2., 3., 4.], .5), 2.5);self.assertAlmostEqual(E.quantile([0., 10.], .95), 9.5)
        played = {k: E.averaging_effect(base[k], flag[k])[1] for k in (0, 4, 5)}
        self.assertEqual([(d['baseline'], d['mixed'], d['played_baseline'], d['played_redacted']) for d in played.values()],
                         [(0, 1, 0, 1), (0, 1, 0, 0), (0, 1, 0, 0)])

    def test_any_control_movement_is_a_determinism_failure(self):
        base, flag = self.cases()
        def failures(change):
            moved = json.loads(json.dumps(flag));change(moved[1]);r = E.compare(base, moved)
            self.assertFalse(r['determinism']['passed']);return r['determinism']['rows']
        def nudge(x):x['worlds'][0]['root_peer_logits'][0] = math.nextafter(x['worlds'][0]['root_peer_logits'][0], 2.)   # one ulp
        self.assertEqual([(x['world'], x['bits_equal']) for x in failures(nudge)], [(0, False)])
        def kept(x):x['worlds'][0]['kept'] = [True] + [False] * 8
        self.assertEqual([x.get('status') or x.get('averaging_status') for x in failures(kept)], ['kept_mismatch', 'world_mismatch'])
        def missing(x):x['worlds'] = x['worlds'][:1]
        self.assertEqual([x.get('status') or x.get('averaging_status') for x in failures(missing)], ['redacted_world_missing', 'redacted_worlds_missing'])
        def requested(x):x['worlds'][1]['opponent_requested'] = False;x['worlds'][1]['kept'] = [False] * 9
        self.assertEqual([x.get('status') or x.get('averaging_status') for x in failures(requested)], ['requested_mismatch', 'world_mismatch'])
        def guard(x):x['effect_guard']['b'] = True   # (b) changes: the played action does not (greedy is averaging's 0)
        self.assertEqual([(x['b_changed'], x['played']) for x in failures(guard)], [(True, [0, 0])])
        def flipped(x):x['worlds'] = [world(0, -1.), world(1, -1.)]
        self.assertEqual([x.get('averaging') for x in failures(flipped) if x['world'] is None], [[0, 1]])
        def refused(x):x['replay'] = dict(status='REFUSED', cause='depthn_refused')
        self.assertEqual([x.get('averaging_status') for x in failures(refused)], ['decision_mismatch'])
        alike = [json.loads(json.dumps(x)) for x in (base, flag)]
        for rows in alike:rows[1]['replay'] = dict(status='REFUSED', cause='depthn_refused')   # refused alike: no failure
        r = E.compare(*alike);self.assertTrue(r['determinism']['passed']);self.assertEqual(r['counts']['averaging_undecided'], 1)
        moved = json.loads(json.dumps(flag));nudge(moved[1])
        code, stdout, stderr, report = run(base, moved)
        self.assertEqual((code, report['determinism']['passed'], report['determinism']['failures']), (3, False, 1))
        self.assertIn('DETERMINISM FAILURE', stderr);self.assertIn('"decision_index": 1', stderr);self.assertIn('determinism: FAILED', stdout)

    def test_joins_mismatches_and_main(self):
        base, flag = self.cases()
        code, stdout, stderr, report = run(base, flag)
        self.assertEqual((code, stderr), (0, ''));self.assertIn('applied: roots 4', stdout);self.assertIn('PLAYED flips 1 (0.333)', stdout)
        self.assertIn('determinism: ok', stdout)
        other = json.loads(json.dumps(flag));other[1]['battle_id'] = 'elsewhere';other.append(row(9, 9, [1., 1.]))
        report = E.compare(base, other)
        self.assertEqual((report['identity_mismatches']['count'], report['counts']['redacted_without_baseline'], report['determinism']['passed']), (1, 1, True))
        wrong = json.loads(json.dumps(base));wrong[0]['private_reveals']['applicable'] = []   # the runs disagree about the selection
        self.assertEqual(E.compare(wrong, flag)['record_mismatches']['count'], 1)
        code, _, stderr, report = run(base, other)
        self.assertEqual(code, 0);self.assertIn('identity mismatches', stderr)


if __name__ == '__main__':
    unittest.main()
