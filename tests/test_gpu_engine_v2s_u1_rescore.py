"""U1 re-score pairing on synthetic scorer rows; no bank, model or device."""
import unittest

from gpu_engine import v2s_u1_rescore as rescore


def rows(outcomes):
    """Scorer top-two rows: (stream, frame, correct) -> row with fixed actions."""
    return dict(top_two=dict(rows=[dict(selector=dict(stream=s, frame=f, owner=0), actions=[0, 1], correct=bool(c)) for s, f, c in outcomes]))


class RescoreTests(unittest.TestCase):
    def test_fallback_and_conditional_modes(self):
        base = [(1, 0, 1), (1, 4, 0), (2, 0, 0), (2, 4, 1), (3, 0, 0)]
        # The predictor resolves four pairs (refuses stream 3) and flips (1,4) to right and (2,4) to wrong.
        other = [(1, 0, 1), (1, 4, 1), (2, 0, 0), (2, 4, 0)]
        report = rescore.rescore(dict(raw_policy=rows(base), search=rows(other)))['versus_raw_policy']['search']
        self.assertEqual(report['fallback']['pairs'], 5);self.assertEqual(report['fallback']['difference'], 0.)
        self.assertEqual(report['conditional']['pairs'], 4);self.assertEqual(report['conditional']['games'], 2)
        self.assertEqual(report['resolved_pairs'], 4)
        self.assertEqual(report['by_frame']['4']['conditional']['difference'], 0.)
        low, high = report['fallback']['ci90_game_cluster'];self.assertLessEqual(low, 0.);self.assertGreaterEqual(high, 0.)

    def test_baseline_choice_and_empty_predictor(self):
        base = [(1, 0, 1), (2, 0, 0)]
        metrics = dict(raw_policy=rows(base), raw_value=rows([(1, 0, 0), (2, 0, 0)]), refused=rows([]))
        report = rescore.rescore(metrics, 'raw_value')
        self.assertEqual(report['baseline'], 'raw_value');self.assertEqual(report['versus_raw_policy']['raw_policy']['fallback']['difference'], .5)
        self.assertIsNone(rescore.rescore(metrics)['versus_raw_policy']['refused']['conditional']['difference'])

    def test_pairs_outside_the_fixed_set_are_rejected(self):
        with self.assertRaises(ValueError):rescore.rescore(dict(raw_policy=rows([(1, 0, 1)]), x=rows([(9, 0, 1)])))


if __name__ == '__main__':
    unittest.main()
