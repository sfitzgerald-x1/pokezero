"""Authored report fixtures only: no native search, models or source games."""

from __future__ import annotations

from copy import deepcopy
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokezero.engine_search import _aggregate_root_arms


NAMES = ("alpha", "beta", "gamma")
PRIORS = dict(zip(NAMES, (.1, .2, .7)))


def record(order=NAMES, *, side="side_one", priors=PRIORS, available=True):
    """Native reports sort arms by visits, not by their initialized prior."""
    def entries(names):
        return [dict(move=name, visits=len(names) - i, q=.25,
                     prior=priors[name]) for i, name in enumerate(names)]

    other = "side_two" if side == "side_one" else "side_one"
    return dict(side_key=side, report={
        side: entries(order), other: entries(tuple(reversed(order))),
        "root_priors": list(priors.values()) if available else None,
    })


class RootPriorAggregationTests(unittest.TestCase):
    def test_normalization_does_not_depend_on_builtin_sum_algorithm(self):
        # The supported Python 3.11 floor predates the more accurate builtin
        # float summation introduced in 3.12. Exercise sequential accumulation
        # explicitly rather than letting the local interpreter hide the case.
        def sequential_sum(values, start=0):
            total = start
            for value in values:
                total += value
            return total

        self.assertNotEqual(sequential_sum((.1, .2, .7)),
                            sequential_sum((.7, .2, .1)))
        with patch("pokezero.engine_search.sum", sequential_sum, create=True):
            expected = _aggregate_root_arms([record()])
            for order in itertools.permutations(NAMES):
                actual = _aggregate_root_arms([record(order)])
                self.assertEqual(actual.prior_share, expected.prior_share)
                self.assertEqual(actual.reported_prior_share,
                                 expected.reported_prior_share)
                self.assertEqual(actual.opponent_prior_share,
                                 expected.opponent_prior_share)

    def test_acting_priors_do_not_depend_on_visit_report_order(self):
        for side in ("side_one", "side_two"):
            expected = _aggregate_root_arms([record(side=side)])
            for order in itertools.permutations(NAMES):
                with self.subTest(side=side, order=order):
                    actual = _aggregate_root_arms([record(order, side=side)])
                    self.assertEqual(actual.prior_share, expected.prior_share)
                    self.assertEqual(actual.prior_cause, expected.prior_cause)
                    # These reports intentionally have different visit leaders.
                    self.assertGreater(actual.visit_share[order[0]],
                                       actual.visit_share[order[-1]])

    def test_reported_fallback_priors_are_order_stable_but_not_model_priors(self):
        expected = _aggregate_root_arms([record(available=False)])
        for order in itertools.permutations(NAMES):
            actual = _aggregate_root_arms([record(order, available=False)])
            self.assertEqual(actual.reported_prior_share,
                             expected.reported_prior_share)
            self.assertFalse(actual.prior_share)
            self.assertIsNotNone(actual.prior_cause)

    def test_opponent_priors_do_not_depend_on_visit_report_order(self):
        for side in ("side_one", "side_two"):
            expected = _aggregate_root_arms([record(side=side)])
            for order in itertools.permutations(NAMES):
                actual = _aggregate_root_arms([record(order, side=side)])
                self.assertEqual(actual.opponent_prior_share,
                                 expected.opponent_prior_share)

    def test_duplicate_world_multiplicity_is_preserved(self):
        second = dict(zip(NAMES, (.2, .3, .5)))
        worlds = [record(), record(priors=second), record(), record()]
        changed_order = [record(tuple(reversed(NAMES))),
                         record(tuple(reversed(NAMES)), priors=second),
                         record(tuple(reversed(NAMES))),
                         record(tuple(reversed(NAMES)))]
        expected = _aggregate_root_arms(worlds)
        actual = _aggregate_root_arms(changed_order)
        self.assertEqual(actual.prior_share, expected.prior_share)
        self.assertEqual(actual.reported_prior_share, expected.reported_prior_share)
        for name in NAMES:
            self.assertAlmostEqual(actual.prior_share[name],
                                   3 * PRIORS[name] + second[name])
            self.assertEqual(actual.reported_prior_worlds[name], 4)

    def test_partial_missing_and_misaligned_authority_remain_unmeasurable(self):
        partial = _aggregate_root_arms([record(), record(available=False)])
        self.assertFalse(partial.prior_share)
        self.assertIsNotNone(partial.prior_cause)

        missing = record()
        missing["report"]["side_one"][0]["prior"] = None
        mismatched = record()
        mismatched["report"]["root_priors"] = [.2, .2, .6]
        for item in (missing, mismatched):
            aggregate = _aggregate_root_arms([item])
            self.assertFalse(aggregate.prior_share)
            self.assertIsNotNone(aggregate.prior_cause)

    def test_singleton_stays_exact_and_reports_are_not_mutated(self):
        item = record(("only",), priors={"only": 1.})
        before = deepcopy(item)
        aggregate = _aggregate_root_arms([item])
        self.assertEqual(aggregate.prior_share, {"only": 1.})
        self.assertEqual(aggregate.reported_prior_share, {"only": 1.})
        self.assertIsNone(aggregate.prior_cause)
        self.assertEqual(item, before)


if __name__ == "__main__":
    unittest.main()
