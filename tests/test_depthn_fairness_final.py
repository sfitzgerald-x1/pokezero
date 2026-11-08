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


if __name__ == '__main__':
    unittest.main()
