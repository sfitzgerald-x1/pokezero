import hashlib
import json
from pathlib import Path
import random
import unittest
from tempfile import TemporaryDirectory

from pokezero import depthn_live_panel as panel
from scripts import depthn_screen_amended_verdict as V

AMENDMENT = Path(__file__).resolve().parents[1]/'evals'/'depthn-foulplay-dn2w16-amendment-unknown-cells-20261108.json'
DECLARATION_SHA256 = '1d6f227d74555b8c98f313b7cf50cebb05c2261de6a7ef9b1b25464486237065'


def write_run(root, raw, search, arm='DN2W16'):
    """raw/search: {pair: score or an error string (UNKNOWN)} written as the panel's result files."""
    (root/'results').mkdir(parents=True)
    for name, values, axis in (('RAW', raw, 0), (arm, search, 1)):
        for pair, value in values.items():
            known = not isinstance(value, str)
            cell = dict(arm=name, pair_ordinal=pair, ordinal=3*pair+axis)
            (root/'results'/f"{cell['ordinal']}.json").write_text(json.dumps(dict(cell=cell, status='COMPLETE' if known else 'UNKNOWN',
                score=value if known else None, error=None if known else value)))


def readout(limit, *, decision='CONTINUE_TO_2000', breakage=(), harm=False, bar=None, paired=None):
    return dict(declaration_sha256=DECLARATION_SHA256, arm='DN2W16', paired_cells=limit, decision=decision, breakage=list(breakage),
                harm=harm, bar=bar, paired=paired or {})


class AmendedVerdictTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)
        self.amendment = json.loads(AMENDMENT.read_bytes()); self.sha = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()

    def tearDown(self): self.directory.cleanup()

    def run_verdict(self, value): return V.verdict(value, self.amendment, self.root, readout_sha256='r'*64, amendment_sha256=self.sha)

    def test_the_committed_amendment_is_the_rule_applied(self):
        self.assertEqual((self.amendment['schema'], self.amendment['declaration_sha256'], self.amendment['rule']['unknown_share_max_per_arm']),
                         (V.AMENDMENT_SCHEMA, DECLARATION_SHA256, .05))

    def test_a_few_unknowns_are_no_longer_breakage_and_are_classified(self):
        raw = {p: 1. if p % 3 == 0 else 0. for p in range(20)}; search = {p: 1. if p % 2 == 0 else .5 for p in range(20)}
        raw[7] = 'RuntimeError: owning game round cap without actual owner terminal'
        write_run(self.root, raw, search)
        value = self.run_verdict(readout(20, decision='STOP_BREAKAGE', breakage=['unknown_cells']))
        self.assertEqual((value['decision'], value['breakage'], value['official_decision']), ('CONTINUE_TO_2000', [], 'STOP_BREAKAGE'))
        self.assertEqual(value['unknown']['arms']['RAW'], dict(cells=1, share=.05, by_cause={'round_cap': 1}, pairs=[7], results=20))
        self.assertEqual(value['unknown']['share_difference'], -.05)

    def test_more_than_five_percent_in_either_arm_is_breakage(self):
        raw = {p: 0. for p in range(20)}; search = {p: 1. for p in range(20)}
        search[3] = 'TimeoutError: timed out waiting for foul-play challenge.'; search[4] = 'TimeoutError: timed out waiting for foul-play choice.'
        write_run(self.root, raw, search)
        value = self.run_verdict(readout(20, decision='STOP_BREAKAGE', breakage=['unknown_cells', 'latency_p95']))
        self.assertEqual((value['decision'], value['breakage']), ('STOP_BREAKAGE', ['latency_p95', 'unknown_share']))   # 2/20 = 10%; others stand
        self.assertEqual(value['unknown']['arms']['DN2W16']['by_cause'], {'foulplay_challenge_timeout': 1, 'foulplay_choice_timeout': 1})

    def test_the_conservative_bound_counts_unknowns_against_search(self):
        rng = random.Random(5); limit = 40
        raw = {p: rng.choice((0., .5, 1.)) for p in range(limit)}; search = {p: rng.choice((0., .5, 1.)) for p in range(limit)}
        write_run(self.root, raw, search)
        rows = lambda values, name: [dict(status='COMPLETE', score=v, cell=dict(pair_ordinal=p, arm=name)) for p, v in values.items()]
        declared = panel.paired(rows(raw, 'RAW'), rows(search, 'DN2W16'), limit=limit)
        clean = V.conservative_interval(V.cells(self.root, 'RAW', limit), V.cells(self.root, 'DN2W16', limit), limit)
        for got, want in zip(clean['interval90'], declared['interval90']): self.assertAlmostEqual(got, want, places=12)   # no unknowns: the declared estimator
        self.directory.cleanup(); self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)
        raw[0] = 'x'; search[1] = 'y'; del search[5]   # an unknown RAW cell counts 1, an unknown or missing search cell 0
        write_run(self.root, raw, search)
        worst = V.conservative_interval(V.cells(self.root, 'RAW', limit), V.cells(self.root, 'DN2W16', limit), limit)
        deltas = [(0. if p in (1, 5) else search[p]) - (1. if p == 0 else raw[p]) for p in range(limit)]
        self.assertAlmostEqual(worst['mean'], sum((deltas[s]+deltas[s+1])/2 for s in range(0, limit, 2))/(limit//2), places=12)
        self.assertLess(worst['interval90'][0], clean['interval90'][0])

    def test_harm_kept_decisions_and_the_bar(self):
        write_run(self.root, {p: 0. for p in range(10)}, {p: 1. if p != 4 else 0. for p in range(10)})
        self.assertEqual(self.run_verdict(readout(10, decision='STOP_HARM', harm=True))['decision'], 'STOP_HARM')
        for kept in ('VOID_LATENCY', 'VOID_FAIRNESS', 'FLAG_ARM_FAIRNESS', 'INCOMPLETE'):
            self.assertEqual(self.run_verdict(readout(10, decision=kept))['decision'], kept)
        bar = self.run_verdict(readout(10, decision='BAR_NOT_MET', bar=dict(met=True)))
        self.assertEqual(bar['decision'], 'BAR_MET'); self.assertGreater(bar['conservative']['lower_bound'], 0)
        write_run_unknowns = {p: 'x' for p in range(10)}
        self.directory.cleanup(); self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)
        write_run(self.root, {p: 0. for p in range(10)}, write_run_unknowns)
        self.assertEqual(self.run_verdict(readout(10, decision='BAR_MET', bar=dict(met=True)))['decision'], 'BAR_NOT_MET')

    def test_refuses_another_declaration_a_wrong_sha_and_writes_once(self):
        write_run(self.root, {0: 0., 1: 1.}, {0: 1., 1: 1.})
        with self.assertRaisesRegex(ValueError, 'amendment of this readout'):
            self.run_verdict(dict(readout(2), declaration_sha256='0'*64))
        path = self.root/'readout.json'; path.write_text(json.dumps(readout(2)))
        with self.assertRaisesRegex(ValueError, 'sha256 differs'): V.read_bound(path, '0'*64)
        out = self.root/'verdict.json'; sha = hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ['--readout', str(path), '--readout-sha256', sha, '--amendment', str(AMENDMENT), '--amendment-sha256', self.sha,
                '--run-directory', str(self.root), '--output', str(out)]
        self.assertEqual(V.main(argv), 0); self.assertEqual(json.loads(out.read_text())['readout_sha256'], sha)
        with self.assertRaises(FileExistsError): V.main(argv)


if __name__ == '__main__':
    unittest.main()
