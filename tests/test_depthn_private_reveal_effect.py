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


def row(ordinal, index, leans, *, redacted=(), applicable=None, b=False, guarded=None, battle_id=None, waiting=()):
    record = dict(rule='reach', b=b, own_guarded_lanes=guarded or [None] * 4)
    members = [dict(species=s, ability='pressure', slot=k) for k, s in enumerate(redacted)]
    return dict(ordinal=ordinal, decision_index=index, battle_id=battle_id or f'battle-{ordinal}', owner='p1',
                replay=dict(status='COMPLETE'), root=dict(root_logits=[0.] * 9, root_legal=LEGAL), effect_guard=record,
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
        flag = [row(1, 0, [2., 2.], redacted=['zapdos']),   # applied: both worlds' reply prior moves to reply 1: averaging flips 0 -> 1
                row(1, 1, [1., 1.]),   # control
                row(2, 0, [2., 2.], waiting=(1,)),   # control with a waiting world
                row(3, 0, [2., 2.], redacted=['zapdos'], b=True)]   # applied, but the guard trips: not searched
        base = [baseline_of(x) for x in flag]
        flag[0]['worlds'] = [world(0, -2.), world(1, -2.)];flag[3]['worlds'] = [world(0, -.5), world(1, 2.)]
        return base, flag

    def test_a_redaction_moves_the_reply_prior_and_averaging_and_the_control_does_not(self):
        base, flag = self.cases()
        r = E.compare(base, flag)
        p, q = original._probabilities(tuple(logits(2.)), tuple(KEPT)), original._probabilities(tuple(logits(-2.)), tuple(KEPT))
        expected = math.fsum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a > 0)
        a = r['groups']['applied'];c = r['groups']['control'];every = r['groups']['all']
        self.assertEqual((a['roots'], a['worlds'], a['reply_top1_flips'], a['logits_bits_differ']), (2, 4, 3, 3))   # (3, 0)'s world 1 kept its logits
        self.assertAlmostEqual(a['kl_max'], expected, places=12)
        self.assertEqual((a['averaging']['scored_roots'], a['averaging']['searched_roots'], a['averaging']['flips'], a['averaging']['flip_rate']),
                         (2, 1, 1, 1.))   # the guarded root is scored but not searched
        self.assertEqual((c['roots'], c['worlds'], c['kl_max'], c['kl_nonzero'], c['reply_top1_flips'], c['averaging']['flips'], c['averaging']['searched_roots']),
                         (2, 3, 0., 0, 0, 0, 2))
        self.assertEqual((every['worlds'], every['averaging']['searched_roots']), (7, 3))
        self.assertEqual((r['worlds']['compared'], r['worlds']['waiting']), (7, 1))
        self.assertTrue(r['determinism']['passed']);self.assertEqual(r['redacted_species'], {'zapdos': 2})
        self.assertEqual((r['identity_mismatches']['count'], r['record_mismatches']['count'], r['counts']['joined']), (0, 0, 4))
        self.assertEqual(E.quantile([1., 2., 3., 4.], .5), 2.5);self.assertAlmostEqual(E.quantile([0., 10.], .95), 9.5)

    def test_any_control_movement_is_a_determinism_failure(self):
        base, flag = self.cases()
        nudged = list(flag[1]['worlds'][0]['root_peer_logits']);nudged[0] = math.nextafter(nudged[0], 2.)   # one ulp: no flip, a tiny KL
        flag[1]['worlds'][0]['root_peer_logits'] = nudged
        code, stdout, stderr, report = run(base, flag)
        self.assertEqual((code, report['determinism']['passed'], report['determinism']['failures']), (0, False, 1))
        self.assertEqual((report['determinism']['examples'][0]['ordinal'], report['determinism']['examples'][0]['bits_equal']), (1, False))
        self.assertIn('DETERMINISM FAILURE', stderr);self.assertIn('determinism: FAILED', stdout)
        flipped = json.loads(json.dumps(flag));flipped[1]['worlds'] = [world(0, -1.), world(1, -1.)]   # a control root whose averaging flips
        report = E.compare(base, flipped)
        self.assertEqual(report['groups']['control']['averaging']['flips'], 1)
        self.assertTrue(any(x.get('averaging') == [0, 1] for x in report['determinism']['examples']))

    def test_joins_mismatches_and_main(self):
        base, flag = self.cases()
        code, stdout, stderr, report = run(base, flag)
        self.assertEqual((code, stderr), (0, ''));self.assertIn('applied: roots 2', stdout);self.assertIn('determinism: ok', stdout)
        other = json.loads(json.dumps(flag));other[1]['battle_id'] = 'elsewhere';other.append(row(9, 9, [1., 1.]))
        missing = json.loads(json.dumps(other));missing[2]['worlds'][0]['kept'] = [True] * 2 + [False] * 7   # unchanged: still joins
        missing[2]['worlds'][0]['kept'][1] = False   # kept differs: that world is not compared
        report = E.compare(base, missing)
        self.assertEqual((report['identity_mismatches']['count'], report['counts']['redacted_without_baseline'], report['worlds']['kept_mismatch']), (1, 1, 1))
        self.assertEqual(report['counts']['averaging_world_mismatch'], 1)
        wrong = json.loads(json.dumps(base));wrong[0]['private_reveals']['applicable'] = []   # the runs disagree about the selection
        self.assertEqual(E.compare(wrong, flag)['record_mismatches']['count'], 1)
        code, _, stderr, report = run(base, other)
        self.assertEqual(code, 0);self.assertIn('identity mismatches', stderr)


if __name__ == '__main__':
    unittest.main()
