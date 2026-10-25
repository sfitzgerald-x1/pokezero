"""Authored paired panels only; no checkpoints, collection or scientific outcomes."""
from itertools import product
from math import sqrt
from statistics import fmean, stdev
import unittest

from scipy.stats import t

from pokezero.mcts_eval.fullgame_strength import Outcome, paired_t_completion_box
from pokezero.mcts_eval.training_strength import (
    registered_paired_interval, summarize_training_pairs,
)


class TrainingStrengthTests(unittest.TestCase):
    def panel(self):
        seeds = list(range(1000))
        treatment = [Outcome(seed, seat, "COMPLETE", int(seed < 20), True)
                     for seed in seeds for seat in ("p1", "p2")]
        control = [Outcome(seed, seat, "COMPLETE", 0, True)
                   for seed in seeds for seat in ("p1", "p2")]
        return seeds, treatment, control

    def test_known_1000_cluster_interval_matches_independent_reference(self):
        values = [-0.5, 0, 0.25, 0.75] * 250
        result = registered_paired_interval(values, values, registered_seed_count=1000)
        margin = float(t.ppf(0.975, 999)) * stdev(values) / sqrt(1000)
        self.assertAlmostEqual(result["lower"], fmean(values) - margin, places=13)
        self.assertAlmostEqual(result["upper"], fmean(values) + margin, places=13)
        self.assertEqual(result["degrees_of_freedom"], 999)
        self.assertFalse(result["finite_sample_certified"])

    def test_box_envelope_matches_exhaustive_independent_intervals(self):
        choices = [(-1, 0), (0, 0.5), (-0.25, 0.75), (0.25, 1)]
        lower = [lo for lo, hi in choices] + [-1, 1]
        upper = [hi for lo, hi in choices] + [-1, 1]
        intervals = []
        for completion in product(*choices):
            values = list(completion) + [-1, 1]
            margin = float(t.ppf(0.975, 5)) * stdev(values) / sqrt(6)
            intervals.append((fmean(values) - margin, fmean(values) + margin))
        result = registered_paired_interval(lower, upper, registered_seed_count=6)
        self.assertAlmostEqual(result["lower"], min(lo for lo, hi in intervals), places=13)
        self.assertAlmostEqual(result["upper"], max(hi for lo, hi in intervals), places=13)
        reverse = registered_paired_interval([-v for v in upper], [-v for v in lower],
                                             registered_seed_count=6)
        self.assertEqual(result["lower"], -reverse["upper"])

    def test_128_seed_extension_preserves_frozen_lower_endpoint(self):
        lower = [-0.5, 0, 0.25, 0.75] * 32
        upper = list(lower)
        upper[0] = 0.5
        expected = paired_t_completion_box(lower, upper)
        result = registered_paired_interval(lower, upper, registered_seed_count=128)
        self.assertEqual(result["lower"], expected["lower"])

    def test_registration_invalid_grid_and_nonboolean_attestation_refused(self):
        for count in (True, 0, 1, 2.0, 1000):
            with self.subTest(count=count), self.assertRaises(ValueError):
                registered_paired_interval([-1, 1], [-1, 1], registered_seed_count=count)
        for lo, hi in ((0.1, 0.5), (True, 0.5), (float("nan"), 1),
                       (0, float("inf")), (0.5, 0.25), (-1.25, 0)):
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):
                registered_paired_interval([lo, 1], [hi, 1], registered_seed_count=2)
        with self.assertRaises(ValueError):
            summarize_training_pairs([1, 2], [], [], registered_seed_count=2,
                                     normal_complete_attempt=1)
        with self.assertRaises(ValueError):
            summarize_training_pairs([1, 2], [], [], registered_seed_count=1000)

    def test_both_seats_are_clustered_and_no_five_point_gate_is_inherited(self):
        seeds, treatment, control = self.panel()
        result = summarize_training_pairs(seeds, treatment, control,
            registered_seed_count=1000, normal_complete_attempt=True)
        self.assertEqual(result["identification_bounds"], [0.02, 0.02])
        self.assertTrue(result["improvement_established"])
        self.assertEqual(result["primary_interval"]["degrees_of_freedom"], 999)
        self.assertEqual(result["paired_scored_cells"], 2000)
        self.assertNotIn("deployment_point_rule_passed", result)
        self.assertEqual(result["per_seed"][0], {"seed": 0, "lower": 1, "upper": 1})

    def test_caps_are_unknown_and_can_still_allow_robust_interval(self):
        seeds, treatment, control = self.panel()
        treatment[0] = Outcome(0, "p1", "CAPPED", 0.5)
        result = summarize_training_pairs(seeds, iter(treatment), iter(control),
            registered_seed_count=1000, normal_complete_attempt=True)
        self.assertTrue(result["improvement_established"])
        self.assertEqual(result["identification_bounds"], [0.0195, 0.02])
        self.assertEqual(result["primary_interval"]["unknown_seed_clusters"], 1)
        self.assertEqual(result["candidate_validated_games"], 1999)
        self.assertEqual(result["paired_scored_cells"], 1999)

    def test_live_failed_missing_and_unvalidated_panels_never_infer(self):
        seeds, treatment, control = self.panel()
        live = summarize_training_pairs(seeds, treatment, control, registered_seed_count=1000)
        self.assertIsNone(live["primary_interval"])
        for rows in (treatment[:-1], *([Outcome(0, "p1", status)] + treatment[1:]
                     for status in ("FAILED", "REFUSED", "UNSTARTED", "COMPLETE"))):
            with self.subTest(status=rows[0].status, count=len(rows)):
                result = summarize_training_pairs(seeds, rows, control,
                    registered_seed_count=1000, normal_complete_attempt=True)
                self.assertIsNone(result["primary_interval"])
                self.assertFalse(result["improvement_established"])

    def test_degenerate_and_all_uncertain_panels_do_not_manufacture_interval(self):
        for lower, upper in (([1] * 1000, [1] * 1000), ([-1] * 1000, [1] * 1000)):
            result = registered_paired_interval(lower, upper, registered_seed_count=1000)
            self.assertIsNone(result["lower"])
            self.assertIsNone(result["upper"])
            self.assertFalse(result["improvement_established"])


if __name__ == "__main__":
    unittest.main()
