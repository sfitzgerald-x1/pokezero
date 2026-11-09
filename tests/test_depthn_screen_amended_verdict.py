import hashlib
import json
from pathlib import Path
import random
import unittest
from tempfile import TemporaryDirectory

from pokezero import depthn_live_panel as panel
from scripts import depthn_screen_amended_verdict as V

EVALS = Path(__file__).resolve().parents[1]/'evals'
AMENDMENT = EVALS/'depthn-foulplay-dn2w16-amendment-unknown-cells-20261108.json'
ADDENDUM = EVALS/'depthn-foulplay-dn2w16-amendment-harm-mirror-20261108.json'
BOTH = EVALS/'depthn-foulplay-dn2w16-amendment-both-bounds-20261108.json'
FINAL = EVALS/'depthn-foulplay-dn2w16-amendment-final-fairness-20261108.json'
FAIR = dict(within=True, finer=dict(difference=-.01))   # a final fairness gate that passes (depthn_fairness_final.final_gate's shape)
DECLARATION_SHA256 = '1d6f227d74555b8c98f313b7cf50cebb05c2261de6a7ef9b1b25464486237065'
TREE = '1aaa75f94d41b538f1bca22f5b4afc0627e627f4'
ROUND_CAP = 'RuntimeError: owning game round cap without actual owner terminal'
CHALLENGE = 'TimeoutError: timed out waiting for foul-play challenge.'
CHOICE = 'TimeoutError: timed out waiting for foul-play choice.'


def write_run(root, raw, search, *, lost=(), arm='DN2W16', declaration=DECLARATION_SHA256):
    """raw/search: {pair: score, or an error string for an UNKNOWN cell}; lost: (arm, pair) claimed with no result."""
    for name in ('results', 'claims'):(root/name).mkdir(parents=True, exist_ok=True)
    cell = lambda name, pair: dict(arm=name, pair_ordinal=pair, ordinal=3*pair + (0 if name == 'RAW' else 1))
    for name, values in (('RAW', raw), (arm, search)):
        for pair, value in values.items():
            known = not isinstance(value, str);c = cell(name, pair)
            (root/'results'/f"{c['ordinal']}.json").write_text(json.dumps(dict(schema=V.CELL_SCHEMA, declaration_sha256=declaration, cell=c,
                status='COMPLETE' if known else 'UNKNOWN', score=value if known else None, error=None if known else value)))
            (root/'claims'/f"{c['ordinal']}.json").write_text(json.dumps(dict(cell=c)))
    for name, pair in lost:
        c = cell(name, pair);(root/'claims'/f"{c['ordinal']}.json").write_text(json.dumps(dict(cell=c)))


def official(root, limit, *, decision='CONTINUE_TO_2000', breakage=(), harm=False, arm='DN2W16', declaration=DECLARATION_SHA256, tree=TREE):
    """The readout fields the script binds to, as the panel's arm_summary and paired() compute them from the same files."""
    arms, rows = {}, {}
    for name in ('RAW', arm):
        found = V.cells(root, name, limit, declaration)
        complete = [c for c in found.values() if c[0] == 'COMPLETE'];lost = sum(c[0] == V.LOST for c in found.values())
        arms[name] = dict(results=len(found) - lost, complete=len(complete), unknown=len(found) - len(complete), claimed_without_result=lost,
            not_attempted=limit - len(found), score_mean=None if not complete else sum(c[1] for c in complete)/len(complete))
        rows[name] = [dict(status=c[0], score=c[1], cell=dict(pair_ordinal=p)) for p, c in found.items() if c[0] != V.LOST]
    paired = panel.paired(rows['RAW'], rows[arm], limit=limit)
    return dict(declaration_sha256=declaration, tree=dict(commit=tree), arm=arm, paired_cells=limit, decision=decision,
                breakage=list(breakage), harm=harm, bar=None if limit == V.SCREEN_CELLS else dict(met=False), paired=paired, arms=arms)


def scores(limit, rng, mean=.5):
    """Literal scores; mean .5 is uniform over (0, .5, 1), other means push toward 1 or 0."""
    return {p: rng.choice((0., .5, 1.)) if rng.random() > abs(mean - .5) * 2 else (1. if mean > .5 else 0.) for p in range(limit)}


class AmendedVerdictTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)
        self.amendment = json.loads(AMENDMENT.read_bytes()); self.sha = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
        self.addendum = json.loads(ADDENDUM.read_bytes()); self.addendum_sha = hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()
        self.both = json.loads(BOTH.read_bytes()); self.both_sha = hashlib.sha256(BOTH.read_bytes()).hexdigest()
        self.final = json.loads(FINAL.read_bytes()); self.final_sha = hashlib.sha256(FINAL.read_bytes()).hexdigest()

    def tearDown(self): self.directory.cleanup()

    def fresh(self): self.directory.cleanup(); self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)

    def run_verdict(self, value, addendum=None, both=None, final=None, final_fairness=FAIR):
        return V.verdict(value, self.amendment, self.root, readout_sha256='r'*64, amendment_sha256=self.sha,
                         addendum=self.addendum if addendum is None else addendum, addendum_sha256=self.addendum_sha,
                         both=self.both if both is None else both, both_sha256=self.both_sha,
                         final=self.final if final is None else final, final_sha256=self.final_sha, final_fairness=final_fairness)

    def test_the_committed_files_are_the_rules_applied(self):
        self.assertEqual((self.amendment['schema'], self.amendment['declaration_sha256'], self.amendment['rule']['unknown_share_max_per_arm'],
                          self.amendment['run']['tree_commit']), (V.AMENDMENT_SCHEMA, DECLARATION_SHA256, .05, TREE))
        self.assertEqual((self.addendum['addendum_to']['sha256'], self.addendum['declaration_sha256']), (self.sha, DECLARATION_SHA256))
        self.assertEqual(sorted(x['sha256'] for x in self.both['addendum_to']), sorted([self.sha, self.addendum_sha]))
        self.assertLessEqual({self.sha, self.addendum_sha, self.both_sha}, {x['sha256'] for x in self.final['addendum_to']})
        self.assertEqual((self.final['declaration_sha256'], self.final['run']['tree_commit']), (DECLARATION_SHA256, TREE))

    def test_the_final_fairness_gate_voids_before_any_interval(self):
        rng = random.Random(7); raw, search = scores(500, rng), scores(500, rng, mean=.1)   # search clearly worse: harm, were intervals computed
        write_run(self.root, raw, search); value = official(self.root, 500, decision='STOP_HARM', harm=True)
        unfair = dict(within=False, finer=dict(difference=-.08))
        out = self.run_verdict(value, final_fairness=unfair)
        self.assertEqual((out['decision'], out['harm'], out['conservative'], out['harm_mirror'], out['bar_met'], out['final_fairness']),
                         ('VOID_FAIRNESS', False, None, None, None, unfair))
        self.assertEqual((out['official_decision'], out['final_sha256']), ('STOP_HARM', self.final_sha))
        self.assertEqual(self.run_verdict(value)['decision'], 'STOP_HARM')   # the same run within the gate
        self.fresh(); raw, search = scores(2000, rng, mean=.3), scores(2000, rng, mean=.8)   # at the bar too: no BAR_MET
        write_run(self.root, raw, search); value = official(self.root, 2000, decision='BAR_MET')
        self.assertEqual((self.run_verdict(value, final_fairness=unfair)['decision'], self.run_verdict(value)['decision']), ('VOID_FAIRNESS', 'BAR_MET'))
        for kept in ('FLAG_ARM_FAIRNESS', 'VOID_LATENCY'):   # the gate comes before the kept decisions too
            self.assertEqual(self.run_verdict(dict(value, decision=kept), final_fairness=unfair)['decision'], 'VOID_FAIRNESS')
        with self.assertRaisesRegex(ValueError, "final fairness gate's result"): self.run_verdict(value, final_fairness=dict(finer={}))
        with self.assertRaisesRegex(ValueError, 'final-fairness addendum'):
            self.run_verdict(value, final=dict(self.final, addendum_to=self.final['addendum_to'][1:]))

    def test_harm_and_the_bar_need_both_the_declared_and_the_imputed_bound(self):
        rng = random.Random(11); raw, search = scores(500, rng), scores(500, rng, mean=.1)   # search clearly worse: mirror upper < 0
        write_run(self.root, raw, search); value = official(self.root, 500, decision='STOP_HARM', harm=True)
        self.assertEqual(self.run_verdict(value)['decision'], 'STOP_HARM')
        loose = dict(value, paired=dict(value['paired'], interval90=[-.06, .001]))   # the declared bound does not show harm: neither do we
        self.assertEqual((self.run_verdict(loose)['harm'], self.run_verdict(loose)['decision']), (False, 'CONTINUE_TO_2000'))
        self.fresh(); raw, search = scores(2000, rng, mean=.3), scores(2000, rng, mean=.8)   # search clearly better: conservative lower > 0
        write_run(self.root, raw, search); value = official(self.root, 2000, decision='BAR_MET')
        self.assertEqual((self.run_verdict(value)['bar_met'], self.run_verdict(value)['decision']), (True, 'BAR_MET'))
        short = dict(value, paired=dict(value['paired'], interval90=[-.01, .2]))   # the declared lower bound misses: so does the bar
        self.assertEqual((self.run_verdict(short)['bar_met'], self.run_verdict(short)['decision']), (False, 'BAR_NOT_MET'))
        self.fresh()
        for p in range(2000):
            if p % 4 != 3: search[p] = CHALLENGE   # three of four search cells unknown, counted as losses: the conservative lower bound misses
        write_run(self.root, raw, search); value = official(self.root, 2000, decision='BAR_MET')
        self.assertEqual(self.run_verdict(dict(value, paired=dict(value['paired'], interval90=[.05, .3])))['bar_met'], False)

    def test_a_few_unknowns_are_no_longer_breakage_and_are_classified(self):
        rng = random.Random(1); raw, search = scores(500, rng), scores(500, rng)
        raw[7] = ROUND_CAP; search[9] = CHALLENGE; del search[11]
        write_run(self.root, raw, search, lost=[('DN2W16', 11)])
        value = self.run_verdict(official(self.root, 500, decision='STOP_BREAKAGE', breakage=['unknown_cells']))
        self.assertEqual((value['decision'], value['breakage'], value['official_decision']), ('CONTINUE_TO_2000', [], 'STOP_BREAKAGE'))
        self.assertEqual(value['unknown']['arms']['RAW']['by_cause'], {'round_cap': 1})
        self.assertEqual(value['unknown']['arms']['DN2W16']['by_cause'], {V.LOST: 1, 'foulplay_challenge_timeout': 1})   # a claim with no result counts
        self.assertEqual(value['unknown']['arms']['DN2W16']['cells'], 2)
        self.assertAlmostEqual(value['unknown']['share_difference'], 2/500 - 1/500, places=12)

    def test_more_than_five_percent_in_either_arm_counting_lost_claims_is_breakage(self):
        rng = random.Random(2); raw, search = scores(500, rng), scores(500, rng)
        for p in range(20): search[p] = CHOICE   # 20 errors (4%) and 6 claims without a result (1.2%): 5.2%, the review's case
        lost = [('DN2W16', p) for p in range(20, 26)]
        for _, p in lost: del search[p]
        write_run(self.root, raw, search, lost=lost)
        value = self.run_verdict(official(self.root, 500, decision='STOP_BREAKAGE', breakage=['unknown_cells', 'latency_p95']))
        self.assertEqual((value['decision'], value['breakage']), ('STOP_BREAKAGE', ['latency_p95', 'unknown_share']))
        self.assertEqual(value['unknown']['arms']['DN2W16']['by_cause'], {V.LOST: 6, 'foulplay_choice_timeout': 20})

    def test_the_conservative_bound_counts_unknowns_against_search_and_equals_the_declared_without_them(self):
        rng = random.Random(5); raw, search = scores(500, rng), scores(500, rng)
        write_run(self.root, raw, search)
        clean = V.conservative_interval(V.cells(self.root, 'RAW', 500), V.cells(self.root, 'DN2W16', 500), 500)
        declared = official(self.root, 500)['paired']['interval90']
        for got, want in zip(clean['interval90'], declared): self.assertAlmostEqual(got, want, places=12)
        r = next(p for p in range(500) if raw[p] < 1.); q = next(p for p in range(500) if search[p] > 0. and p != r)   # imputation must matter
        known = dict(raw), dict(search); self.fresh(); raw[r] = ROUND_CAP; search[q] = CHALLENGE
        write_run(self.root, raw, search)
        worst = V.conservative_interval(V.cells(self.root, 'RAW', 500), V.cells(self.root, 'DN2W16', 500), 500)
        deltas = [(0. if p == q else known[1][p]) - (1. if p == r else known[0][p]) for p in range(500)]
        self.assertAlmostEqual(worst['mean'], sum((deltas[s]+deltas[s+1])/2 for s in range(0, 500, 2))/250, places=12)
        self.assertLess(worst['interval90'][0], clean['interval90'][0])

    def test_harm_is_the_mirror_bound_robust_to_unknowns_for_search(self):
        rng = random.Random(7); raw, search = scores(500, rng), scores(500, rng, mean=.1)   # search clearly worse
        write_run(self.root, raw, search)
        value = self.run_verdict(official(self.root, 500, decision='STOP_HARM', harm=True))
        self.assertEqual((value['decision'], value['harm']), ('STOP_HARM', True)); self.assertLess(value['harm_mirror']['upper_bound'], 0)
        self.fresh()
        for p in range(0, 300): search[p] = CHALLENGE   # unknowns that could all be search wins
        write_run(self.root, raw, search)
        value = self.run_verdict(official(self.root, 500, decision='STOP_HARM', harm=True, breakage=['unknown_cells']))
        self.assertFalse(value['harm']); self.assertGreaterEqual(value['harm_mirror']['upper_bound'], 0)
        self.assertEqual(value['decision'], 'STOP_BREAKAGE')

    def test_kept_decisions_void_latency_and_the_bar(self):
        rng = random.Random(9); raw, search = scores(2000, rng, mean=.3), scores(2000, rng, mean=.8)
        write_run(self.root, raw, search)
        for kept in ('VOID_FAIRNESS', 'FLAG_ARM_FAIRNESS', 'INCOMPLETE'):
            self.assertEqual(self.run_verdict(official(self.root, 2000, decision=kept))['decision'], kept)
        bar = self.run_verdict(official(self.root, 2000, decision='BAR_NOT_MET'))
        self.assertEqual(bar['decision'], 'BAR_MET'); self.assertGreater(bar['conservative']['lower_bound'], 0)
        for kept in ('VOID_FAIRNESS', 'FLAG_ARM_FAIRNESS'):   # the panel sets no bar on a void or held flag, at 2,000 too: kept, never refused
            kept_value = self.run_verdict(dict(official(self.root, 2000, decision=kept), bar=None))
            self.assertEqual((kept_value['decision'], kept_value['bar_met']), (kept, None))
        value = self.run_verdict(dict(official(self.root, 2000, decision='VOID_LATENCY'), paired=None, bar=None))   # no screen statistic: no interval
        self.assertEqual((value['decision'], value['conservative'], value['harm_mirror']), ('VOID_LATENCY', None, None))

    def test_the_files_are_bound_to_the_readout(self):
        rng = random.Random(3); write_run(self.root, scores(500, rng), scores(500, rng)); value = official(self.root, 500)
        with self.assertRaisesRegex(ValueError, "amended run's tree"): self.run_verdict(dict(value, tree=dict(commit='0'*40)))
        with self.assertRaisesRegex(ValueError, 'screen or bar readout'): self.run_verdict(dict(value, paired_cells=40))
        moved = dict(value, arms=dict(value['arms'], RAW=dict(value['arms']['RAW'], complete=value['arms']['RAW']['complete'] - 1)))
        with self.assertRaisesRegex(ValueError, 'cells differ from the readout'): self.run_verdict(moved)
        with self.assertRaisesRegex(ValueError, 'pairs differ from the readout'):
            self.run_verdict(dict(value, paired=dict(value['paired'], known_pairs=value['paired']['known_pairs'] - 1)))
        path = next((self.root/'results').glob('*.json')); bad = json.loads(path.read_text()); bad['declaration_sha256'] = '0'*64
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'another panel, declaration or cell'): self.run_verdict(value)

    def test_refuses_another_declaration_or_addendum_a_wrong_sha_and_writes_once(self):
        rng = random.Random(4); write_run(self.root, scores(500, rng), scores(500, rng)); value = official(self.root, 500)
        with self.assertRaisesRegex(ValueError, 'amendment of this readout'): self.run_verdict(dict(value, declaration_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'harm addendum'):
            self.run_verdict(value, addendum=dict(self.addendum, addendum_to=dict(self.addendum['addendum_to'], sha256='0'*64)))
        with self.assertRaisesRegex(ValueError, 'both-bounds addendum'):
            self.run_verdict(value, both=dict(self.both, addendum_to=self.both['addendum_to'][:1]))
        path = self.root/'readout.json'; path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'sha256 differs'): V.read_bound(path, '0'*64)
        out = self.root/'verdict.json'; sha = hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ['--readout', str(path), '--readout-sha256', sha, '--amendment', str(AMENDMENT), '--amendment-sha256', self.sha,
                '--addendum', str(ADDENDUM), '--addendum-sha256', self.addendum_sha, '--both', str(BOTH), '--both-sha256', self.both_sha,
                '--final', str(FINAL), '--final-sha256', self.final_sha, '--run-directory', str(self.root), '--output', str(out)]
        self.assertEqual(V.PINNED, dict(amendment=self.sha, addendum=self.addendum_sha, both=self.both_sha, final=self.final_sha))   # the committed files are the pinned ones
        with self.assertRaisesRegex(SystemExit, "amended run's"): V.main(argv)   # this temporary directory is not /out/screen-r1
        with self.assertRaisesRegex(SystemExit, 'committed file'): V.main([x if x != self.both_sha else '0'*64 for x in argv])
        with self.assertRaisesRegex(SystemExit, 'committed file'): V.main([x if x != self.final_sha else '0'*64 for x in argv])
        real = V.read_bound
        def here(p, digest):   # the committed amendment, read for a run at this temporary directory (its bytes and sha unchanged)
            value = real(p, digest)
            return dict(value, run=dict(value['run'], run_directory=str(self.root.resolve()))) if Path(p) == AMENDMENT else value
        from unittest import mock
        from scripts import depthn_fairness_final
        with mock.patch.object(V, 'read_bound', side_effect=here), mock.patch.object(depthn_fairness_final, 'final_gate', return_value=FAIR) as gate:
            self.assertEqual(V.main(argv), 0); written = json.loads(out.read_text())
            self.assertEqual((written['readout_sha256'], written['final_fairness'], written['final_sha256']), (sha, FAIR, self.final_sha))
            self.assertEqual(gate.call_args.args, (str(self.root), 'DN2W16', 500))   # the whole run, at the readout's limit
            with self.assertRaises(FileExistsError): V.main(argv)


ADOPTION = EVALS/'depthn-foulplay-dn2w16-amendment-screen-r2-adoption-20261109.json'
R2_DECLARATION = EVALS/'depthn-foulplay-dn2w16-declaration-r2-20261109.json'
R2_DECLARATION_SHA256 = 'f065fee350d9c9a27eb0c29595bec2292ca669ec47ff1e9defad71b97e9141a6'
R2_TREE = '5ca04a4b6eb0ba62f619fa5944d4737b52b5ad96'
# screen-r1's verdicts through the CLI on the seeded fixtures below, written by the script before --run-binding existed (5ca04a4b)
R1_GOLDEN = ('cbd3d50968409d4377c655b728fa21fc7066864b67f0520df13c491c708ef7ff', 'ec66ae8fefde34105d34b5ae60abd36d67a5e141fcab307cca7a1c89bd4df9fd',
             '2cfa4290e607ecae972d6d4f421c91837c7d6aefa2e19dfb6deb3863198d723f', '79efc9bf948b415393d879b7fc6f0e1ceb367afdbd7e69e4d0d2322549c7fc8e')
SCENARIOS = ((4, 500, .5, .5, None, 'CONTINUE_TO_2000', (), False), (1, 500, .5, .5, 'unknown', 'STOP_BREAKAGE', ('unknown_cells',), False),
             (7, 500, .5, .1, None, 'STOP_HARM', (), True), (9, 2000, .3, .8, None, 'BAR_MET', (), False))


def scenario(root, seed, limit, mean_raw, mean_search, edit, decision, breakage, harm, **kw):
    """One seeded run written at `root` and its official readout (the screen-r1 fixtures, or another run's with `declaration`/`tree`)."""
    rng = random.Random(seed);raw, search = scores(limit, rng, mean_raw), scores(limit, rng, mean_search);lost = ()
    if edit == 'unknown':raw[7] = ROUND_CAP;search[9] = CHALLENGE;del search[11];lost = [('DN2W16', 11)]
    write_run(root, raw, search, lost=lost, declaration=kw.get('declaration', DECLARATION_SHA256))
    return official(root, limit, decision=decision, breakage=breakage, harm=harm, **kw)


class RunBindingTests(unittest.TestCase):
    """--run-binding: the adoption record supplies the run directory, declaration sha256 and tree commit; the rules are unchanged."""
    def setUp(self):
        self.directory = TemporaryDirectory();self.root = Path(self.directory.name)
        self.files = {k: (p, json.loads(p.read_bytes()), hashlib.sha256(p.read_bytes()).hexdigest())
                      for k, p in (('amendment', AMENDMENT), ('addendum', ADDENDUM), ('both', BOTH), ('final', FINAL))}
        self.adoption = json.loads(ADOPTION.read_bytes());self.adoption_sha = hashlib.sha256(ADOPTION.read_bytes()).hexdigest()

    def tearDown(self):self.directory.cleanup()

    def binding(self, **run):
        """The committed adoption record, its run directory this temporary directory (or `run`'s overrides)."""
        return dict(self.adoption, run={**self.adoption['run'], 'run_directory': str(self.root.resolve()), **run})

    def run_verdict(self, value, binding=None, *, unbound=False):
        f = self.files
        return V.verdict(value, f['amendment'][1], self.root, readout_sha256='r'*64, amendment_sha256=f['amendment'][2],
            addendum=f['addendum'][1], addendum_sha256=f['addendum'][2], both=f['both'][1], both_sha256=f['both'][2],
            final=f['final'][1], final_sha256=f['final'][2], final_fairness=FAIR,
            **({} if unbound else dict(binding=self.binding() if binding is None else binding, binding_sha256=self.adoption_sha)))

    def cli(self, readout, *extra, binding=True):
        path = self.root/'readout.json';path.write_text(json.dumps(readout));sha = hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ['--readout', str(path), '--readout-sha256', sha]
        for k, (p, _, digest) in self.files.items():argv += [f'--{k}', str(p), f'--{k}-sha256', digest]
        argv += ['--run-directory', str(self.root), '--output', str(self.root/'verdict.json')]
        return argv + (['--run-binding', str(ADOPTION), self.adoption_sha] if binding else []) + list(extra)

    def main(self, argv, rewrite):
        """V.main with read_bound rewriting the run directory of `rewrite` (the committed amendment or adoption record) to this one."""
        from unittest import mock
        from scripts import depthn_fairness_final
        real = V.read_bound
        def here(p, digest):
            value = real(p, digest)
            return dict(value, run=dict(value['run'], run_directory=str(self.root.resolve()))) if Path(p) == rewrite else value
        with mock.patch.object(V, 'read_bound', side_effect=here), mock.patch.object(depthn_fairness_final, 'final_gate', return_value=FAIR), \
                mock.patch('sys.stdout'):
            return V.main(argv)

    def test_without_a_binding_screen_r1_is_byte_identical(self):
        """screen-r1's verdict path (the CLI, no --run-binding) on four seeded runs writes the bytes the script wrote before the binding."""
        got = []
        for case in SCENARIOS:
            self.tearDown();self.setUp()
            self.assertEqual(self.main(self.cli(scenario(self.root, *case), binding=False), AMENDMENT), 0)
            got.append(hashlib.sha256((self.root/'verdict.json').read_bytes()).hexdigest())
        self.assertEqual(tuple(got), R1_GOLDEN)

    def test_the_adoption_record_adopts_the_four_pinned_rules_for_screen_r2(self):
        a = self.adoption
        self.assertEqual((a['schema'], V.RUN_BINDINGS['screen_r2']), (V.ADOPTION_SCHEMA, self.adoption_sha))
        self.assertEqual(sorted(x['sha256'] for x in a['adopts']), sorted(V.PINNED.values()))
        self.assertEqual({x['change'] for x in a['adopts']}, {'none'})
        for x in a['adopts']:self.assertEqual(hashlib.sha256((EVALS.parent/x['path']).read_bytes()).hexdigest(), x['sha256'])
        self.assertEqual((a['declaration_sha256'], hashlib.sha256(R2_DECLARATION.read_bytes()).hexdigest()), (R2_DECLARATION_SHA256, R2_DECLARATION_SHA256))
        self.assertEqual((a['run']['run_directory'], a['run']['tree_commit'], a['declaration']), ('/out/screen-r2', R2_TREE, 'evals/' + R2_DECLARATION.name))
        self.assertEqual([x['sha256'] for x in a['not_adopted']], [hashlib.sha256((EVALS/'depthn-foulplay-dn2w16-amendment-extension-20261108.json').read_bytes()).hexdigest()])

    def test_the_bound_run_gets_screen_r1s_rules_unchanged(self):
        """The same seeded cells under r2's declaration (bound) and r1's (unbound): the same decision, intervals and unknowns."""
        for case in SCENARIOS:
            with self.subTest(seed=case[0]):
                self.tearDown();self.setUp()
                bound = self.run_verdict(scenario(self.root, *case, declaration=R2_DECLARATION_SHA256, tree=R2_TREE));here = str(self.root.resolve())
                self.tearDown();self.setUp()
                plain = self.run_verdict(scenario(self.root, *case), unbound=True)
                self.assertEqual((bound.pop('run_binding_sha256'), bound.pop('run_binding')), (self.adoption_sha, dict(declaration_sha256=R2_DECLARATION_SHA256,
                    run_directory=here, tree_commit=R2_TREE)))
                self.assertEqual((bound.pop('tree_commit'), plain.pop('tree_commit')), (R2_TREE, TREE));self.assertEqual(bound, plain)

    def test_the_binding_refuses_another_declaration_directory_or_tree(self):
        value = scenario(self.root, *SCENARIOS[0], declaration=R2_DECLARATION_SHA256, tree=R2_TREE)
        self.assertEqual(self.run_verdict(value)['decision'], 'CONTINUE_TO_2000')
        with self.assertRaisesRegex(ValueError, "readout's declaration"):self.run_verdict(dict(value, declaration_sha256=DECLARATION_SHA256))
        with self.assertRaisesRegex(ValueError, "amended run's tree"):self.run_verdict(dict(value, tree=dict(commit=TREE)))
        with self.assertRaisesRegex(ValueError, "bound run's directory"):self.run_verdict(value, self.binding(run_directory='/out/screen-r2'))
        with self.assertRaisesRegex(ValueError, "readout's declaration"):self.run_verdict(value, dict(self.binding(), declaration_sha256='0'*64))
        with self.assertRaisesRegex(ValueError, "amended run's tree"):self.run_verdict(value, self.binding(tree_commit='0'*40))
        for bad in (dict(self.binding(), adopts=self.adoption['adopts'][:3]), dict(self.binding(), schema=V.AMENDMENT_SCHEMA),
                    dict(self.binding(), adopts=[dict(x, change='edited') for x in self.adoption['adopts']]),
                    dict(self.binding(), adopts=self.adoption['adopts'][:3] + [dict(self.adoption['adopts'][3], sha256='0'*64)])):
            with self.assertRaisesRegex(ValueError, 'exactly the four rule files'):self.run_verdict(value, bad)
        with self.assertRaisesRegex(ValueError, 'tree commit required'):self.run_verdict(value, self.binding(tree_commit='5ca04a4b'))
        path = next((self.root/'results').glob('*.json'));cell = json.loads(path.read_text());cell['declaration_sha256'] = DECLARATION_SHA256
        path.write_text(json.dumps(cell))   # a screen-r1 result file in the bound run
        with self.assertRaisesRegex(ValueError, 'another panel, declaration or cell'):self.run_verdict(value)

    def test_the_cli_binds_screen_r2_through_the_committed_record_only(self):
        value = scenario(self.root, *SCENARIOS[0], declaration=R2_DECLARATION_SHA256, tree=R2_TREE)
        argv = self.cli(value)
        with self.assertRaisesRegex(SystemExit, "bound run's"):V.main(argv)   # this temporary directory is not /out/screen-r2
        with self.assertRaisesRegex(SystemExit, 'committed run binding'):V.main([x if x != self.adoption_sha else '0'*64 for x in argv])
        self.assertEqual(self.main(argv, ADOPTION), 0)
        written = json.loads((self.root/'verdict.json').read_text())
        self.assertEqual((written['run_binding_sha256'], written['run_binding']['declaration_sha256'], written['tree_commit'], written['decision']),
                         (self.adoption_sha, R2_DECLARATION_SHA256, R2_TREE, 'CONTINUE_TO_2000'))
        unbound = self.cli(value, binding=False);unbound[unbound.index('--output') + 1] = str(self.root/'r1.json')
        with self.assertRaisesRegex(ValueError, "readout's declaration"):self.main(unbound, AMENDMENT)   # screen-r1's rules alone refuse r2's readout


if __name__ == '__main__':
    unittest.main()
