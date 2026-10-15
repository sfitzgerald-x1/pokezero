import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("candidate_calibration_test",
    Path(__file__).resolve().parents[1] / "scripts/calibrate_search_over_raw_candidates.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class CandidateCalibrationTests(unittest.TestCase):
    def test_exact_fft_matches_exhaustive_studentized_randomization(self):
        for values in ([0, 0, 0, 0], [-1, .25, .25, .25], [.25, .5, -.75, 1.], [.25] * 4):
            x = np.asarray(values)
            signed = np.asarray(list(itertools.product((-1, 1), repeat=len(x)))) * x
            def t(rows):
                sums = rows.sum(axis=-1)
                den = np.sqrt(np.maximum(len(x) * (rows * rows).sum(axis=-1) - sums * sums, 0))
                return np.divide(sums, den, out=np.where(sums > 0, np.inf,
                    np.where(sums < 0, -np.inf, 0.)), where=den > 0)
            expected = np.mean(t(signed) >= t(x))
            self.assertAlmostEqual(cal.studentized_signflip_pvalue(x), expected, places=12)

    def test_signflip_studentization_does_not_change_mean_ordering(self):
        x = np.array([-1.] + [.25] * 7)
        signed = np.asarray(list(itertools.product((-1, 1), repeat=len(x)))) * x
        self.assertAlmostEqual(cal.studentized_signflip_pvalue(x),
            np.mean(signed.mean(axis=1) >= x.mean()), places=12)

    def test_bernstein_range_two_bound_and_missing_abstention(self):
        x = np.full((2, 256), .25)
        self.assertTrue(np.all(cal.empirical_bernstein_reject(x)))
        unknown = np.zeros(x.shape, bool); unknown[0, 0] = True
        result = cal.evaluate(x, unknown, .025)
        self.assertEqual(result["empirical_bernstein"]["count"], 1)
        self.assertEqual(result["empirical_bernstein_abstentions"], 1)

    def test_original_generator_and_no_hidden_adoption(self):
        values = cal.four_games(np.random.default_rng(4), .1, 5, 64)
        self.assertEqual(values.shape, (5, 64))
        self.assertTrue(np.all(np.abs(values) <= 1))
        self.assertTrue(np.all(values * 4 == np.rint(values * 4)))
        result = cal.run(replicates=2, sizes=(64,))
        self.assertFalse(result["method_adopted"])
        self.assertEqual(result["scientific_calls"], 0)


if __name__ == "__main__":
    unittest.main()
