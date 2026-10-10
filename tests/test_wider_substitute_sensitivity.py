"""Nine historical clusters are outcome-blind, not a sixth convenient seed."""
import math
import itertools
from collections import Counter
from functools import lru_cache
import unittest

from pokezero.mcts_eval.wider_native_recovery import worst_case_statistics


class SubstituteSensitivityTests(unittest.TestCase):
    def test_full_64_cluster_p_matches_independent_binomial_grouped_enumeration(self):
        # Synthetic scores only: never peek at the running study's outcomes.
        known = [0, 1, 2, 3, 4] * 11
        values = [0.] * 9 + [v / 4 for v in known]
        production = worst_case_statistics(values, range(9), max_uncertain=9)

        @lru_cache(None)
        def grouped_tail(groups, threshold):
            # Independent construction: grouped binomial coefficients rather
            # than production's sign-per-value recurrence / baseline reuse.
            distribution = {0: 1}
            for magnitude, count in groups:
                changed = Counter()
                for positives in range(count + 1):
                    increment = magnitude * (2 * positives - count)
                    ways = math.comb(count, positives)
                    for subtotal, subtotal_ways in distribution.items():
                        changed[subtotal + increment] += subtotal_ways * ways
                distribution = changed
            return sum(ways for subtotal, ways in distribution.items()
                       if abs(subtotal) >= threshold) / (2 ** sum(n for _, n in groups))

        # Enumerate a different representation: compositions of nine labeled
        # score-class counts; multinomial multiplicity covers ordered vectors.
        maximum, covered = 0., 0
        for cuts in itertools.combinations(range(17), 8):
            boundaries = (-1, *cuts, 17)
            counts = [boundaries[i + 1] - boundaries[i] - 1 for i in range(9)]
            self.assertEqual(sum(counts), 9)
            groups = Counter(v for v in known if v)
            for quarter, count in zip(range(-4, 5), counts):
                if quarter:
                    groups[abs(quarter)] += count
            threshold = abs(sum(known) + sum(q * n for q, n in zip(range(-4, 5), counts)))
            maximum = max(maximum, grouped_tail(tuple(sorted(groups.items())), threshold))
            multiplicity = math.factorial(9)
            for count in counts:
                multiplicity //= math.factorial(count)
            covered += multiplicity
        self.assertEqual(covered, 9 ** 9)
        self.assertEqual(production['maximum_exact_p'], maximum)

    def test_all_nine_score_assignments_are_covered_and_prior_outcomes_irrelevant(self):
        uncertain = list(range(9))
        left = worst_case_statistics([1.] * 9 + [.5] * 55, uncertain, max_uncertain=9)
        right = worst_case_statistics([-1.] * 9 + [.5] * 55, uncertain, max_uncertain=9)
        self.assertEqual(left, right)
        self.assertEqual(left['distinct_counterfactual_multisets'], math.comb(17, 9))
        self.assertEqual(left['ordered_counterfactuals_covered'], 9 ** 9)
        self.assertEqual(left['minimum_mean_win_score_delta'], (55 * .5 - 9) / 64)
        self.assertEqual(left['minimum_bounded_mean_lower'],
            (55 * .5 - 9) / 64 - math.sqrt(2 * math.log(40) / 64))
        self.assertFalse(left['all_scores_support_advantage'])

    def test_zero_untouched_scores_allow_an_exact_p_of_one_without_equivalence(self):
        value = worst_case_statistics([1.] * 9 + [0.] * 55, range(9), max_uncertain=9)
        self.assertEqual(value['maximum_exact_p'], 1.)
        self.assertEqual(value['minimum_mean_win_score_delta'], -9 / 64)
        self.assertFalse(value['all_scores_support_advantage'])

    def test_nine_needs_explicit_registered_bound_and_no_duplicates_or_tenth_cluster(self):
        for indices, cap in ((range(9), 5), (range(10), 9), ([0] * 9, 9)):
            with self.subTest(indices=list(indices), cap=cap):
                with self.assertRaisesRegex(RuntimeError, 'invalid sensitivity roster'):
                    worst_case_statistics([0.] * 64, indices, max_uncertain=cap)


if __name__ == '__main__':
    unittest.main()
