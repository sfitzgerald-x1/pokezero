"""Authored full-game fixtures; no models, games, native search or collection."""
import math
from statistics import fmean, stdev
import unittest

from pokezero.mcts_eval.fullgame_strength import Outcome, paired_t, summarize_pairs


class FullgameStrengthTests(unittest.TestCase):
    def row(self, seed, seat, score, status="COMPLETE", validated=True):
        return Outcome(seed, seat, status, score, validated)

    def test_both_seats_are_one_cluster_and_missing_control_is_adverse(self):
        result = summarize_pairs([1, 2], [self.row(1, "p1", 1), self.row(1, "p2", 0)],
                                 [self.row(1, "p1", 0), self.row(1, "p2", 0)])
        self.assertEqual(result["identification_bounds"], [-0.25, 0.75])
        self.assertEqual(result["per_seed"][0], {"seed": 1, "lower": 0.5, "upper": 0.5})
        self.assertEqual(result["seed_clusters"], 2)
        self.assertIsNone(result["identified_point_estimate"])

    def test_all_missingness_patterns_keep_registered_denominator(self):
        for status in ("CAPPED", "REFUSED", "FAILED", "UNSTARTED", "COMPLETE"):
            with self.subTest(status=status):
                candidate = [self.row(1, "p1", 0.5, status, False), self.row(1, "p2", 1)]
                control = [self.row(1, seat, 0) for seat in ("p1", "p2")]
                result = summarize_pairs([1], candidate, control)
                self.assertEqual(result["identification_bounds"], [0.5, 1])
                self.assertEqual(result["candidate_validated_games"], 1)
                self.assertEqual(result["planned_games_per_arm"], 2)

    def test_genuine_draw_is_observed_but_capped_half_score_is_not(self):
        self.assertEqual(self.row(1, "p1", 0.5).bounds(), (0.5, 0.5))
        self.assertEqual(self.row(1, "p1", 0.5, "CAPPED", False).bounds(), (0, 1))

    def test_control_missingness_uses_upper_endpoint_for_lower_contrast(self):
        result = summarize_pairs([1], [self.row(1, seat, 0.5) for seat in ("p1", "p2")], [])
        self.assertEqual(result["identification_bounds"], [-0.5, 0.5])

    def test_duplicate_foreign_and_invalid_rosters_are_rejected(self):
        row = self.row(1, "p1", 1)
        for seeds, candidate in (([1, 1], []), ([True], []), ([1], [row, row]), ([2], [row])):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                summarize_pairs(seeds, candidate, [])
        for score in (True, math.nan, math.inf, 0.25):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.row(1, "p1", score)
        with self.assertRaises(ValueError):
            self.row(1, "p1", 0.5, "CAPPED", True)

    def test_t_test_uses_128_clusters_and_one_sided_tail(self):
        from scipy.stats import ttest_1samp, t

        values = [0.5, -0.25, 0.25, 0] * 32
        result = paired_t(values)
        independent = ttest_1samp(values, 0, alternative="greater")
        self.assertAlmostEqual(result["nominal_one_sided_p"], independent.pvalue, places=13)
        expected = fmean(values) - t.ppf(0.975, 127) * stdev(values) / math.sqrt(128)
        self.assertAlmostEqual(result["lower"], expected, places=13)
        reverse = paired_t([-v for v in values])
        self.assertAlmostEqual(reverse["nominal_one_sided_p"], 1 - independent.pvalue, places=13)
        self.assertFalse(result["finite_sample_certified"])
        with self.assertRaises(ValueError):
            paired_t(values[:-1])

    def test_zero_variance_never_manufactures_significance(self):
        for value in (-1, 0, 1):
            result = paired_t([value] * 128)
            self.assertEqual(result["status"], "DEGENERATE_VARIANCE")
            self.assertIsNone(result["nominal_one_sided_p"])
            self.assertFalse(result["screening_rule_passed"])

    def test_live_run_never_advances_and_shared_control_is_reusable(self):
        seeds = list(range(128))
        control = [self.row(seed, seat, 0) for seed in seeds for seat in ("p1", "p2")]
        candidate = [self.row(seed, seat, 1 if seed % 2 else 0.5)
                     for seed in seeds for seat in ("p1", "p2")]
        live = summarize_pairs(seeds, candidate, control)
        self.assertIsNone(live["primary_test"])
        self.assertFalse(live["screening_rule_passed"])
        first = summarize_pairs(seeds, candidate, control, terminal=True)
        second = summarize_pairs(seeds, list(candidate), control, terminal=True)
        self.assertEqual(first, second)
        self.assertTrue(first["screening_rule_passed"])
        self.assertTrue(first["deployment_point_rule_passed"])
        self.assertFalse(first["deployability_qualified"])

    def test_positive_point_estimate_below_five_points_is_not_deployment_rule(self):
        seeds = list(range(128))
        control = [self.row(seed, seat, 0.5) for seed in seeds for seat in ("p1", "p2")]
        candidate = [self.row(seed, seat, 1 if seed < 10 else 0.5)
                     for seed in seeds for seat in ("p1", "p2")]
        result = summarize_pairs(seeds, candidate, control, terminal=True)
        self.assertTrue(result["screening_rule_passed"])
        self.assertLess(result["identified_point_estimate"], 0.05)
        self.assertFalse(result["deployment_point_rule_passed"])

    def test_stopped_or_incomplete_panel_never_advances_even_if_favorable(self):
        seeds = list(range(128))
        control = [self.row(seed, seat, 0) for seed in seeds for seat in ("p1", "p2")]
        candidate = [self.row(seed, seat, 1 if seed % 2 else 0.5)
                     for seed in seeds for seat in ("p1", "p2")]
        for partial_candidate, partial_control in ((candidate[:-1], control),
                                                    (candidate, control[:-1])):
            result = summarize_pairs(seeds, partial_candidate, partial_control, terminal=True)
            self.assertGreater(result["identification_bounds"][0], 0.5)
            self.assertFalse(result["complete_roster"])
            self.assertIsNone(result["primary_test"])
            self.assertFalse(result["screening_rule_passed"])
