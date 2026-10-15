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

    def test_joint_third_refusal_stops_both_without_terminal_claim(self):
        lower = np.full((2, 256, 2), .5)
        upper = lower.copy()
        refused = np.zeros((2, 256), bool)
        refused[0, [2, 5, 9]] = True
        for method in ("empirical_bernstein", "studentized_signflip"):
            result = cal.joint_candidate(lower, upper, refused, method)
            np.testing.assert_array_equal(result["stop"], [10, 128])
            np.testing.assert_array_equal(result["crossed"], [[False, False], [True, True]])

    def test_joint_unknown_bounds_retained_and_no_suffix_inference(self):
        lower = np.full((2, 256, 2), .5)
        upper = lower.copy()
        # Shared unknown raw score: known candidate .5 implies [-.5, .5].
        lower[0, 0] = -.5
        # Uncertainty only after the fixed look is not observed.
        lower[1, 200:203] = -1
        upper[1, 200:203] = 1
        result = cal.joint_candidate(lower, upper, np.zeros((2, 256), bool), "empirical_bernstein")
        np.testing.assert_array_equal(result["abstains"], [[True, True], [False, False]])
        np.testing.assert_array_equal(result["stop"], [128, 128])
        self.assertTrue(np.all(result["crossed"][1]))

    def test_joint_bernstein_matches_fixed_look_formula(self):
        values = np.random.default_rng(7).choice((-.25, 0., .25), (4, 256, 2))
        result = cal.joint_candidate(values, values, np.zeros((4, 256), bool), "empirical_bernstein")
        expected = cal.empirical_bernstein_reject(values[:, :128].transpose(0, 2, 1))
        np.testing.assert_array_equal(result["crossed"], expected)


if __name__ == "__main__":
    unittest.main()
