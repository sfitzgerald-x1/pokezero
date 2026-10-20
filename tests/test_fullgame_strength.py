"""Authored full-game fixtures; no models, games, native search or collection."""
import math
from itertools import product
from statistics import fmean, stdev
import unittest

from pokezero.mcts_eval.fullgame_strength import (
    Outcome, paired_t, paired_t_completion_box, summarize_capped_pairs, summarize_pairs,
)


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


class CompletionSensitivityTests(unittest.TestCase):
    def panel(self):
        seeds = list(range(128))
        candidate = [Outcome(seed, seat, "COMPLETE", int(seed < 64), True)
                     for seed in seeds for seat in ("p1", "p2")]
        control = [Outcome(seed, seat, "COMPLETE", 0, True)
                   for seed in seeds for seat in ("p1", "p2")]
        return seeds, candidate, control

    def test_known_scalar_intervals_preserve_existing_test(self):
        values = [0.5, -0.25, 0.25, 0] * 32
        expected = paired_t(values)
        result = paired_t_completion_box(values, values)
        for key, value in expected.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["unknown_seed_clusters"], 0)

    def test_off_grid_invalid_and_wrong_sized_boxes_are_rejected(self):
        lower, upper = [0.0] * 128, [0.5] * 128
        for lo, hi in ((0.1, 0.5), (True, 0.5), (math.nan, 0.5),
                       (0, math.inf), (-1.25, 0), (0.75, 0.5)):
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):
                paired_t_completion_box([lo] + lower[1:], [hi] + upper[1:])
        for lo, hi in ((lower[:-1], upper), (lower, upper[:-1])):
            with self.assertRaises(ValueError):
                paired_t_completion_box(lo, hi)

    def test_possible_constant_completion_keeps_variance_unavailable(self):
        result = paired_t_completion_box([0] * 127 + [0.25], [1] * 127 + [0.75])
        self.assertEqual(result["status"], "DEGENERATE_COMPLETION_POSSIBLE")
        self.assertIsNone(result["lower"])
        self.assertFalse(result["screening_rule_passed"])
        self.assertIsNone(result["nominal_one_sided_p"])

    def test_single_unknown_is_not_automatic_disqualification(self):
        lower, upper = [1.0] * 64 + [0.0] * 64, [1.0] * 64 + [0.0] * 64
        lower[0] = 0.5
        result = paired_t_completion_box(lower, upper)
        self.assertTrue(result["screening_rule_passed"])
        self.assertEqual(result["unknown_seed_clusters"], 1)
        self.assertFalse(result["finite_sample_certified"])

    def test_all_lower_imputation_can_pass_while_robust_test_fails(self):
        lower = [0.25] * 4 + [0.0] * 124
        upper = list(lower)
        upper[-1] = 1
        self.assertTrue(paired_t(lower)["screening_rule_passed"])
        result = paired_t_completion_box(lower, upper)
        self.assertFalse(result["screening_rule_passed"])
        self.assertLess(result["lower"], 0)
        self.assertLess(paired_t(result["least_favorable_completion"])["lower"], 0)

    def test_dynamic_program_matches_exhaustive_independent_arithmetic(self):
        from scipy.stats import t

        critical = float(t.ppf(0.975, 127))
        choices = [(-1, 1), (0, 0.5), (-0.25, 0.75), (0.25, 1),
                   (-1, -0.5), (-0.5, 0), (0.5, 1)]
        for count in range(1, 8):
            lower, upper = [0.25, 0, 0.5, -0.25] * 32, [0.25, 0, 0.5, -0.25] * 32
            for index, (lo, hi) in enumerate(choices[:count]):
                lower[index], upper[index] = lo, hi
            endpoints = []
            for completion in product(*choices[:count]):
                values = list(completion) + lower[count:]
                endpoints.append(fmean(values) - critical * stdev(values) / math.sqrt(128))
            result = paired_t_completion_box(lower, upper)
            with self.subTest(count=count):
                self.assertAlmostEqual(result["lower"], min(endpoints), places=13)
                witness = result["least_favorable_completion"]
                self.assertTrue(all(v in (lo, hi) for v, lo, hi in zip(witness, lower, upper)))
                self.assertAlmostEqual(paired_t(witness)["lower"], result["lower"], places=13)

    def test_all_128_uncertain_clusters_have_no_unknown_count_cutoff(self):
        lower, upper = [-1] * 127 + [0.75], [-0.75] + [1] * 127
        result = paired_t_completion_box(lower, upper)
        self.assertEqual(result["unknown_seed_clusters"], 128)
        self.assertEqual(len(result["least_favorable_completion"]), 128)
        self.assertLessEqual(result["retained_sum_states"], 1025)
        self.assertFalse(result["screening_rule_passed"])

    def test_new_reader_never_changes_legacy_active_entry_contract(self):
        seeds, candidate, control = self.panel()
        candidate[0] = Outcome(0, "p1", "CAPPED", 0.5, False)
        legacy = summarize_pairs(seeds, candidate, control, terminal=True)
        live = summarize_capped_pairs(seeds, candidate, control)
        future = summarize_capped_pairs(seeds, candidate, control, normal_complete_attempt=True)
        self.assertIsNone(legacy["primary_test"])
        self.assertFalse(legacy["screening_rule_passed"])
        self.assertIsNone(live["primary_test"])
        self.assertTrue(future["screening_rule_passed"])
        self.assertEqual(future["identification_bounds"], [127 / 256, 0.5])
        self.assertIsNone(future["identified_point_estimate"])
        self.assertFalse(future["complete_roster"])
        self.assertFalse(future["deployability_qualified"])

    def test_normal_flag_never_admits_non_cap_missingness(self):
        seeds, candidate, control = self.panel()
        variants = [candidate[:-1]]
        for status in ("REFUSED", "FAILED", "UNSTARTED", "COMPLETE"):
            rows = list(candidate)
            rows[0] = Outcome(0, "p1", status, 0.5, False)
            variants.append(rows)
        for rows in variants:
            with self.subTest(rows=len(rows), status=rows[0].status):
                result = summarize_capped_pairs(seeds, rows, control, normal_complete_attempt=True)
                self.assertIsNone(result["primary_test"])
                self.assertFalse(result["screening_rule_passed"])
                self.assertEqual(result["inference_unavailable_reason"], "UNVALIDATED_OR_NON_CAP_CELL")

    def test_nonprimary_roster_and_nonboolean_attestation_do_not_infer(self):
        result = summarize_capped_pairs([1], [], [], normal_complete_attempt=True)
        self.assertIsNone(result["primary_test"])
        self.assertEqual(result["inference_unavailable_reason"], "NON_PRIMARY_ROSTER")
        with self.assertRaises(ValueError):
            summarize_capped_pairs([1], [], [], normal_complete_attempt=1)

    def test_iterable_arms_are_not_consumed_before_admission_check(self):
        seeds, candidate, control = self.panel()
        result = summarize_capped_pairs(seeds, iter(candidate), iter(control),
                                        normal_complete_attempt=True)
        self.assertTrue(result["screening_rule_passed"])
