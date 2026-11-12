import json
from pathlib import Path
import random
import unittest
from tempfile import TemporaryDirectory

from scripts import depthn_fairness_final as F

RATE = {6: 400000., 5: 450000., 4: 520000., 3: 600000., 2: 800000., 1: 900000.}   # Foul Play iterates faster with fewer of its own Pokemon left


def write_game(root, ordinal, arm, pair, *, ours_left, theirs_left, scale=1., stratum='2x1000ms', rounds=40, rng=None):
    """One COMPLETE cell: its owner capture (the owner is p1, Foul Play p2; by mid-game faints bring Foul Play's side (p2) to ours_left and
    the owner's side (p1) to theirs_left: names from Foul Play's view) and Foul Play think rows whose rate follows Foul Play's alive count
    (RATE), times scale. The gate labels by the owner's view: its 'ours' is p1, its 'theirs' p2."""
    rng = rng or random.Random(ordinal);(root/'results').mkdir(parents=True, exist_ok=True);(root/'owning-wire').mkdir(exist_ok=True)
    lines, think, ours, theirs = [], [], 6, 6
    for r in range(rounds):
        turn_lines = [f'|turn|{r + 1}']
        if r >= rounds // 2 and ours > ours_left:turn_lines.append('|faint|p2a: X');ours -= 1   # the owner is p1; Foul Play's side is p2
        if r >= rounds // 2 and theirs > theirs_left:turn_lines.append('|faint|p1a: Y');theirs -= 1
        turn_lines.append('|request|' + json.dumps(dict(side=dict(id='p1'))))
        lines.append(dict(event='WIRE', lines=turn_lines))
        think.append(dict(round=r, stratum=stratum, iterations_per_budget_second=RATE[ours] * scale * (1 + rng.uniform(-.01, .01)),
                          wait_seconds=1., overlap_seconds=1.))
    (root/'owning-wire'/f'{ordinal}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines))
    (root/'results'/f'{ordinal}.json').write_text(json.dumps(dict(status='COMPLETE', cell=dict(arm=arm, pair_ordinal=pair, ordinal=ordinal),
        source_game=dict(opponent_think=think))))


class FinalGateTests(unittest.TestCase):
    def setUp(self): self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)

    def tearDown(self): self.directory.cleanup()

    def run_games(self, *, search_scale=1., search_left=(4, 4), raw_left=(4, 4), pairs=40):
        for pair in range(pairs):
            write_game(self.root, 3*pair, 'RAW', pair, ours_left=raw_left[0], theirs_left=raw_left[1])
            write_game(self.root, 3*pair + 1, 'DN2W16', pair, ours_left=search_left[0], theirs_left=search_left[1], scale=search_scale)
        return F.final_gate(self.root, 'DN2W16', 500)

    def test_equal_rates_pass(self):
        value = self.run_games()
        self.assertTrue(value['within']); self.assertAlmostEqual(value['finer']['difference'], 0., delta=.01)
        self.assertFalse(value['process_age']['report_to_lead'])

    def test_a_uniform_search_arm_deficit_voids(self):
        value = self.run_games(search_scale=.9)
        self.assertFalse(value['within']); self.assertAlmostEqual(value['finer']['difference'], -.1, delta=.01)

    def test_position_mix_alone_passes_though_the_coarse_ratio_is_off(self):
        # search games leave Foul Play with more of its Pokemon (ours = Foul Play's side here) in the late game: slower positions
        value = self.run_games(search_left=(6, 2), raw_left=(2, 6))
        self.assertTrue(value['within']); self.assertAlmostEqual(value['finer']['difference'], 0., delta=.02)
        self.assertLess(value['finer']['unstratified_ratio'], .9)   # the coarse reading alone would have flagged it

    def test_cells_beyond_the_limit_and_unknown_cells_are_left_out(self):
        self.run_games(pairs=10)
        for pair in (600, 601, 602):   # outside a 500-pair screen
            write_game(self.root, 3*pair + 1, 'DN2W16', pair, ours_left=4, theirs_left=4, scale=.5)
        (self.root/'results'/'9999.json').write_text(json.dumps(dict(status='UNKNOWN', cell=dict(arm='DN2W16', pair_ordinal=3, ordinal=9999))))
        value = F.final_gate(self.root, 'DN2W16', 500)
        self.assertTrue(value['within']); self.assertEqual(value['finer']['moves'], dict(RAW=400, search=400))
        self.assertFalse(F.final_gate(self.root, 'DN2W16', 2000)['within'])   # at 2,000 they count: a 50% deficit on 3 of 13 search games

    def test_no_rated_moves_is_not_within(self):   # fail-closed: a run whose Foul Play rates cannot be read is not certified
        write_game(self.root, 0, 'RAW', 0, ours_left=4, theirs_left=4)
        value = F.final_gate(self.root, 'DN2W16', 500)
        self.assertIsNone(value['finer']['difference']); self.assertFalse(value['within'])

    def test_alive_counts_and_age_buckets(self):
        write_game(self.root, 0, 'RAW', 0, ours_left=3, theirs_left=5)
        per_round = F.rounds(self.root/'owning-wire'/'0.jsonl')
        self.assertEqual((per_round[0], per_round[-1]), ((1, 6, 6), (40, 5, 3)))   # the owner (p1) keeps 5, Foul Play (p2) 3
        self.assertEqual([F.age_bucket(x) for x in (0, 119.9, 120, 599, 600, 5000)], ['0-120', '0-120', '120-300', '300-600', '600+', '600+'])
        self.assertIsNone(F.rounds(self.root/'missing.jsonl'))


# final_gate's output on the seeded fixtures of coarse_cases(), written by the script before the bucketing was parameterized
COARSE_GOLDEN = ('88508341fb0e9f3648098ee8879931afceedd799b5d104a4fed36cc9a2c3bfaa', '99cfdd0988c5eef0a7e57ea296d20253ae7de27a0f39cb5309d66fe96d00a41b',
                 '03520032caddb4ac5c683f37fdb2b215433cce8dd2b3976428503fbbfea8a673')


def coarse_cases():
    import hashlib
    out = []
    for search_left, raw_left, scale in (((4, 4), (4, 4), 1.), ((6, 1), (1, 6), .97), ((2, 2), (5, 1), 1.05)):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pair in range(30):
                stratum = '2x1000ms' if pair % 3 else '4x500ms'
                write_game(root, 3*pair, 'RAW', pair, ours_left=raw_left[0], theirs_left=raw_left[1], stratum=stratum)
                write_game(root, 3*pair + 1, 'DN2W16', pair, ours_left=search_left[0], theirs_left=search_left[1], scale=scale, stratum=stratum)
            out.append(hashlib.sha256(json.dumps(F.final_gate(root, 'DN2W16', 500), sort_keys=True).encode()).hexdigest())
    return tuple(out)


def endgame_run(root, pairs=40):
    """RAW leaves Foul Play 1 or 2 Pokemon alike; the search arm mostly 1 (Foul Play iterates faster with 1 than 2): the coarse 1-2
    bucket pools them, the exact bands keep them apart. Rates are otherwise equal."""
    for pair in range(pairs):
        write_game(root, 3*pair, 'RAW', pair, ours_left=1 + pair % 2, theirs_left=4)
        write_game(root, 3*pair + 1, 'DN2W16', pair, ours_left=2 if pair % 4 == 0 else 1, theirs_left=4)


def log(root, entries):
    (root/'state').mkdir(parents=True, exist_ok=True)
    (root/'state'/'completions.jsonl').write_text(''.join(json.dumps(dict(ordinal=o, arm=a, pair_ordinal=p)) + '\n' for o, a, p in entries))


class AliveBandsTests(unittest.TestCase):
    def setUp(self): self.directory = TemporaryDirectory(); self.root = Path(self.directory.name)

    def tearDown(self): self.directory.cleanup()

    def test_the_bucketing(self):
        self.assertEqual([F.alive_bucket(n) for n in range(7)], ['1-2', '1-2', '1-2', '3-4', '3-4', '5-6', '5-6'])
        self.assertEqual([F.alive_bucket(n, 'exact') for n in range(7)], ['1', '1', '2', '3-4', '3-4', '5-6', '5-6'])
        self.assertEqual(F.ALIVE_BANDS, dict(coarse=('1-2', '3-4', '5-6'), exact=('1', '2', '3-4', '5-6')))
        with self.assertRaisesRegex(ValueError, 'alive bands'): F.alive_bucket(3, 'fine')
        with self.assertRaisesRegex(ValueError, 'alive bands'): F.final_gate(self.root, 'DN2W16', 500, bands='fine')

    def test_the_default_gate_is_unchanged(self):
        self.assertEqual(coarse_cases(), COARSE_GOLDEN)

    def test_exact_bands_separate_one_and_two_alive(self):
        endgame_run(self.root)
        coarse, exact = F.final_gate(self.root, 'DN2W16', 500), F.final_gate(self.root, 'DN2W16', 500, bands='exact')
        self.assertNotIn('alive_bands', coarse);self.assertEqual(exact['alive_bands'], 'exact');self.assertIn('exact alive bands', exact['rule'])
        self.assertGreater(coarse['finer']['difference'], .01)   # the pooled 1-2 bucket favours the arm that leaves 1 more often
        self.assertAlmostEqual(exact['finer']['difference'], 0., delta=.005)
        self.assertGreater(exact['finer']['cells'], coarse['finer']['cells'])

    def test_a_windowed_diagnosis_under_both_bucketings(self):
        endgame_run(self.root)
        entries = [(3*p + k, ('RAW', 'DN2W16')[k], p) for p in range(40) for k in (0, 1)]
        log(self.root, entries + [(3*700, 'RAW', 700)])   # a cell beyond the screen: never in a screen window
        from pokezero import depthn_live_fairness as fairness, depthn_live_panel as panel
        start = 20   # the window of a flag acknowledged at log length 20 (pairs 10 onward)
        coarse = F.window_diagnosis(self.root, 'DN2W16', start, pairs=20)
        self.assertEqual(coarse['window'], dict(start=20, end=60, cells=40, completed_pairs=20, limit=500))
        window = entries[20:60]
        declared = {arm: fairness.pool([panel.cell_moves(self.root, o) for o, a, _ in window if a == arm]) for arm in ('RAW', 'DN2W16')}
        self.assertEqual(coarse['declared']['weighted_ratio'], fairness.reading(declared['RAW'], declared['DN2W16'])['weighted_ratio'])
        exact = F.window_diagnosis(self.root, 'DN2W16', start, pairs=20, bands='exact')
        self.assertEqual((coarse['alive_bands'], exact['alive_bands']), ('coarse', 'exact'))
        self.assertEqual(exact['declared'], coarse['declared'])   # the declared reading does not depend on the bucketing
        self.assertGreater(coarse['finer']['difference'], .01);self.assertAlmostEqual(exact['finer']['difference'], 0., delta=.005)
        self.assertTrue(exact['finer']['within'])
        self.assertEqual(F.window_diagnosis(self.root, 'DN2W16', start, end=30)['window'], dict(start=20, end=30, cells=10, completed_pairs=5, limit=500))
        everything = F.window_diagnosis(self.root, 'DN2W16', 0)   # to the log's end: the cell beyond the screen is left out
        self.assertEqual((everything['window']['end'], everything['window']['cells']), (81, 80))
        with self.assertRaisesRegex(ValueError, 'never reaches'): F.window_diagnosis(self.root, 'DN2W16', 60, pairs=50)
        with self.assertRaisesRegex(ValueError, 'outside the completion log'): F.window_diagnosis(self.root, 'DN2W16', 500)

    def test_the_cli_reproduces_a_flag_and_runs_the_final_gate(self):
        import io
        from contextlib import redirect_stdout
        endgame_run(self.root);log(self.root, [(3*p + k, ('RAW', 'DN2W16')[k], p) for p in range(40) for k in (0, 1)])
        recorded = F.window_diagnosis(self.root, 'DN2W16', 20, pairs=20)['declared']['weighted_ratio']
        flags = self.root/'state'/'fairness'/'screen';flags.mkdir(parents=True)
        (flags/'flag-2.json').write_text(json.dumps(dict(phase='screen', reason='drift', window=20, pairs=20, reading=dict(weighted_ratio=recorded))))
        out = self.root/'diag.json'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(F.main(['window', '--run-directory', str(self.root), '--arm', 'DN2W16', '--flag', '2', '--alive-bands', 'exact',
                                     '--output', str(out)]), 0)
        value = json.loads(out.read_text())
        self.assertEqual((value['flag'], value['window']['start'], value['window']['end'], value['alive_bands']),
                         (dict(number=2, recorded_weighted_ratio=recorded, reproduced=True), 20, 60, 'exact'))
        with redirect_stdout(io.StringIO()) as stdout:
            F.main(['window', '--run-directory', str(self.root), '--arm', 'DN2W16', '--start', '20', '--pairs', '20'])
        self.assertEqual(json.loads(stdout.getvalue())['declared']['weighted_ratio'], recorded)
        with redirect_stdout(io.StringIO()) as stdout:
            F.main(['final', '--run-directory', str(self.root), '--arm', 'DN2W16', '--alive-bands', 'exact'])
        self.assertEqual(json.loads(stdout.getvalue()), json.loads(json.dumps(F.final_gate(self.root, 'DN2W16', 500, bands='exact'))))
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), __import__('contextlib').redirect_stderr(io.StringIO()):
            F.main(['window', '--run-directory', str(self.root), '--arm', 'DN2W16', '--flag', '2', '--start', '0'])


if __name__ == '__main__':
    unittest.main()
