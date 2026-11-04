import unittest
from pokezero import canary_attribution as c


def rows(score=1.):
    return [dict(pair=i, candidate_seat=side, status='COMPLETE', score=score)
            for i in range(1000) for side in (0, 1)]


class AttributionTests(unittest.TestCase):
    def test_whole_seed_pair_bootstrap_and_win(self):
        value = c.summarize(rows(), pairs=1000, bootstrap_seed=7, replicates=10)
        self.assertEqual(value['direction'], 'WIN')
        self.assertEqual(value['paired_seed_90_missing_cell_envelope'], [1., 1.])
        self.assertEqual(value['complete_pairs'], 1000)

    def test_unknowns_are_extrema_not_draws_or_survivor_refill(self):
        data = rows(0.)
        for row in data[:1000]:row.update(status='CAPPED', score=None)
        value = c.summarize(data, pairs=1000, bootstrap_seed=8, replicates=10)
        self.assertEqual(value['identification_bounds'], [0., .5])
        self.assertEqual(value['unknown_games'], 1000)
        self.assertEqual(value['unknown_pairs'], 500)
        self.assertEqual(value['completed_only_score'], 0.)
        self.assertGreater(value['paired_seed_90_missing_cell_envelope'][1], .45)

    def test_missing_and_reordered_cells_refuse(self):
        with self.assertRaises(ValueError):c.summarize(rows()[:-1], pairs=1000, bootstrap_seed=1)
        data = rows();data[0],data[1]=data[1],data[0]
        with self.assertRaises(ValueError):c.summarize(data, pairs=1000, bootstrap_seed=1)

    def test_unknown_score_refuses(self):
        data=rows();data[0].update(status='REFUSED',score=.5)
        with self.assertRaises(ValueError):c.summarize(data,pairs=1000,bootstrap_seed=1)

    def test_three_declared_causes(self):
        for a,b,cause,route in (('WIN','LOSS','ENGINE_OR_OBSERVATION_SKEW','PYTHON_A_B'),
                               ('LOSS','LOSS','FINETUNE_DEGRADATION','V2'),
                               ('WIN','WIN','SELFPLAY_DRIFT','V2')):
            self.assertEqual(c.attribute(dict(direction=a),dict(direction=b),foulplay_failed=True),
                             dict(cause=cause,training_route=route))

    def test_inconclusive_does_not_force_python(self):
        self.assertIsNone(c.attribute(dict(direction='UNRESOLVED'),dict(direction='LOSS'),foulplay_failed=True)['training_route'])
        self.assertEqual(c.attribute(dict(direction='LOSS'),dict(direction='LOSS'),foulplay_failed=False)['training_route'],'V2')
        with self.assertRaises(ValueError):c.attribute(dict(direction='WIN'),dict(direction='WIN'),foulplay_failed=None)


if __name__ == '__main__':unittest.main()
