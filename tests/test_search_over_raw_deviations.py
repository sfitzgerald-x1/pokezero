"""Synthetic arithmetic only: no model imports, source games or holdout reads."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/pokezero/mcts_eval"
PACKAGE = "deviation_kernel_test_modules"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".search_over_raw_deviations",
                                             SOURCE / "search_over_raw_deviations.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class DeviationAccountingTest(unittest.TestCase):
    def root(self, model=0, hp=0, seed=1, index=1):
        return m.ScannedRoot(seed, index, 0, model, hp)

    def test_all_agreement_skips_every_game_and_is_exact_zero(self):
        root = self.root()
        self.assertEqual(root.continuation_actions, ())
        self.assertEqual(m.root_intervals(root, {}), {leaf: (0., 0.) for leaf in m.LEAVES})

    def test_one_or_two_deviations_share_raw_and_identical_candidate(self):
        self.assertEqual(self.root(2, 0).continuation_actions, (0, 2))
        self.assertEqual(self.root(2, 2).continuation_actions, (0, 2))
        self.assertEqual(self.root(2, 3).continuation_actions, (0, 2, 3))

    def test_unavailable_is_neither_agreement_nor_a_made_up_action(self):
        root = self.root(None, 0)
        self.assertEqual(root.continuation_actions, ())
        self.assertIsNone(root.deviation("model"))
        self.assertEqual(m.root_intervals(root, {})["model"], (-1., 1.))
        self.assertEqual(self.root(None, 2).continuation_actions, (0, 2))

    def test_shared_raw_refusal_keeps_both_contrasts_uncertain(self):
        cells = {(a, r): dict(status="COMPLETE", validated=True, score=1.)
                 for a in (1, 2) for r in range(8)}
        cells.update({(0, r): dict(status="REFUSED", validated=False, score=None) for r in range(8)})
        self.assertEqual(m.root_intervals(self.root(1, 2), cells),
                         {leaf: (0., 1.) for leaf in m.LEAVES})

    def test_unvalidated_capped_and_unstarted_cells_remain_unknown(self):
        for status in ("COMPLETE", "CAPPED", "REFUSED", "UNSTARTED"):
            cells = {(a, r): dict(status=status, validated=False,
                                 score=1. if status == "COMPLETE" else None)
                     for a in (0, 1) for r in range(8)}
            self.assertEqual(m.root_intervals(self.root(1, 1), cells),
                             {leaf: (-1., 1.) for leaf in m.LEAVES})

    def test_alias_does_not_excuse_malformed_supplied_receipts(self):
        with self.assertRaises(ValueError):
            m.root_intervals(self.root(), {(0, 0): dict(status="REFUSED", validated=True, score=1.)})

    def test_stop_requires_both_leaves_and_never_overruns_caps_for_coverage(self):
        roots = [self.root(1, 0, index=i + 1) for i in range(100)]
        self.assertIsNone(m.stop_reason(roots, 1))
        roots = [self.root(1, 1, index=i + 1) for i in range(100)]
        self.assertEqual(m.stop_reason(roots, 1), "DEVIATION_TARGET")
        self.assertEqual(m.stop_reason(roots, 28800), "WALL_CAP")
        roots = [self.root(None, None, index=i + 1) for i in range(2000)]
        self.assertEqual(m.stop_reason(roots, 1), "ROOT_CAP")

    def test_unknowns_do_not_count_towards_target(self):
        roots = [self.root(None, 1, index=i + 1) for i in range(100)]
        self.assertIsNone(m.stop_reason(roots, 1))

    def test_duplicates_openings_invalid_actions_and_order_rejected(self):
        for roots in ([self.root(), self.root()], [self.root(index=2), self.root(index=1)]):
            with self.assertRaises(ValueError):
                m.stop_reason(roots, 0)
        for kwargs in (dict(index=0), dict(seed=True), dict(model=True), dict(hp=-1)):
            with self.assertRaises(ValueError):
                self.root(**kwargs)


class DeviationSummaryTest(unittest.TestCase):
    def report(self, roots, intervals, **kwargs):
        return m.summarize(roots, intervals, bootstrap_seed=108, bootstrap_draws=100, **kwargs)

    def test_196_aliases_four_deviations_exact_factorization(self):
        roots = [m.ScannedRoot(i // 7, i % 7 + 1, 0, int(i < 4), int(i < 4)) for i in range(200)]
        intervals = [{leaf: (.125, .125) if i < 4 else (0., 0.) for leaf in m.LEAVES}
                     for i in range(200)]
        report = self.report(roots, intervals)
        for leaf in m.LEAVES:
            row = report["leaves"][leaf]
            values = row["identification_intervals"]
            self.assertAlmostEqual(values["deviation_rate"][0], .02)
            self.assertAlmostEqual(values["conditional_contrast"][0], .125)
            self.assertAlmostEqual(values["overall_effect"][0], .0025)
            self.assertFalse(row["minimum_count_satisfied"])
            self.assertFalse(row["screening_authorized"])

    def test_zero_deviations_conditional_is_undefined_not_zero(self):
        roots = [m.ScannedRoot(s, 1, 0, 0, 0) for s in range(32)]
        report = self.report(roots, [{leaf: (0., 0.) for leaf in m.LEAVES}] * 32)
        values = report["leaves"]["model"]["identification_intervals"]
        self.assertFalse(values["conditional_mean_defined"])
        self.assertIsNone(values["conditional_contrast"])
        self.assertEqual(values["overall_effect"], (0., 0.))
        self.assertFalse(report["leaves"]["model"]["strictly_positive_lower_endpoint"])

    def test_weighting_is_explicit_and_product_consistent_for_both(self):
        roots = [m.ScannedRoot(0, i + 1, 0, 1, 1) for i in range(3)] + [m.ScannedRoot(1, 1, 0, 0, 0)]
        intervals = [{leaf: (.5, .5) for leaf in m.LEAVES}] * 3 + [{leaf: (0., 0.) for leaf in m.LEAVES}]
        for weighting, expected in (("pooled_roots", .375), ("equal_seed", .25)):
            row = self.report(roots, intervals, weighting=weighting)["leaves"]["model"]
            values = row["identification_intervals"]
            self.assertAlmostEqual(values["overall_effect"][0], expected)
            self.assertAlmostEqual(values["deviation_rate"][0] * values["conditional_contrast"][0], expected)

    def test_unknown_selection_keeps_denominator_and_uncertainty(self):
        roots = [m.ScannedRoot(0, 1, 0, None, 0), m.ScannedRoot(1, 1, 0, 1, 0)]
        intervals = [dict(model=(-1., 1.), hp_fraction=(0., 0.)),
                     dict(model=(.5, .5), hp_fraction=(0., 0.))]
        report = self.report(roots, intervals)
        row = report["leaves"]["model"]
        self.assertEqual(report["scanned_roots"], 2)
        self.assertEqual(row["identification_intervals"]["deviation_rate"], (.5, 1.))
        self.assertEqual(row["identification_intervals"]["overall_effect"], (-.25, .75))
        self.assertIsNone(row["identification_intervals"]["conditional_contrast"])

    def test_sparse_deviation_bootstrap_reports_undefined_draws_not_fake_bounds(self):
        roots = [m.ScannedRoot(s, 1, 0, int(s == 0), int(s == 0)) for s in range(32)]
        intervals = [{leaf: ((.5, .5) if s == 0 else (0., 0.)) for leaf in m.LEAVES} for s in range(32)]
        row = self.report(roots, intervals)["leaves"]["model"]
        self.assertEqual(row["identification_intervals"]["conditional_contrast"], (.5, .5))
        statuses = row["conditional_bootstrap_draw_statuses"]
        self.assertGreater(statuses["UNDEFINED_NO_DEVIATIONS"], 0)
        self.assertEqual(sum(statuses.values()), 100)
        self.assertIsNone(row["descriptive_seed_bootstrap_95"]["conditional_contrast"])
        self.assertIsNotNone(row["descriptive_seed_bootstrap_95"]["overall_effect"])
        self.assertIsNotNone(row["descriptive_seed_bootstrap_95"]["deviation_rate"])

    def test_20_positive_deviations_do_not_open_uncalibrated_screen(self):
        roots = [m.ScannedRoot(s, 1, 0, 1, 1) for s in range(32)]
        report = self.report(roots, [{leaf: (.5, .5) for leaf in m.LEAVES}] * 32)
        row = report["leaves"]["model"]
        self.assertTrue(row["minimum_count_satisfied"])
        self.assertTrue(row["source_minimum_satisfied"])
        self.assertTrue(row["strictly_positive_lower_endpoint"])
        self.assertFalse(row["screening_authorized"])
        for flag in ("adaptive_stopping_validity_certified", "calibration_certified", "collection_authorized",
                     "phase_b_authorized", "scientific_strength_claim"):
            self.assertFalse(report[flag])

    def test_same_resamples_preserve_shared_leaf_contrasts(self):
        roots = [m.ScannedRoot(s, 1, 0, 1, 1) for s in range(32)]
        intervals = [{leaf: ((.5, .5) if s < 24 else (-.5, -.5)) for leaf in m.LEAVES} for s in range(32)]
        report = self.report(roots, intervals)
        self.assertEqual(report["leaves"]["model"], report["leaves"]["hp_fraction"])
        self.assertEqual(report, self.report(roots, intervals))

    def test_false_zero_for_unavailable_or_nonzero_alias_rejected(self):
        for root, value in ((m.ScannedRoot(0, 1, 0, None, 0), (0., 0.)),
                            (m.ScannedRoot(0, 1, 0, 0, 0), (.1, .1))):
            with self.assertRaises(ValueError):
                self.report([root], [dict(model=value, hp_fraction=(0., 0.))])


class InformationAccountingTest(unittest.TestCase):
    def test_candidate_identity_is_not_a_leaf_label_alias(self):
        root=m.ScannedInformationRoot(7,1,0,1,2)
        self.assertEqual(root.candidates,("public","oracle"))
        self.assertEqual(root.continuation_actions,(0,1,2))
        self.assertEqual(set(m.root_intervals(root,{})),{"public","oracle"})
        with self.assertRaises(ValueError):root.action("hp_fraction")
        with self.assertRaises(ValueError):root.action("model")

    def test_agreement_and_unknown_action_have_different_meanings(self):
        root=m.ScannedInformationRoot(7,1,0,None,0)
        self.assertEqual(root.continuation_actions,())
        self.assertEqual(m.root_intervals(root,{}),dict(public=(-1.,1.),oracle=(0.,0.)))
        self.assertIsNone(root.deviation("public"))
        self.assertFalse(root.deviation("oracle"))

    def test_alias_still_validates_supplied_cells(self):
        with self.assertRaises(ValueError):
            m.root_intervals(m.ScannedInformationRoot(7,1,0,0,0),
                {(0,0):dict(status="REFUSED",validated=True,score=1.)})

    def test_invalid_boundaries_and_mixed_scientific_comparisons_refuse(self):
        for args in ((True,1,0,1,1),(7,0,0,1,1),(7,1,0,True,1),(7,1,0,1,-1)):
            with self.assertRaises(ValueError):m.ScannedInformationRoot(*args)
        with self.assertRaisesRegex(ValueError,"mix scientific"):
            m.validate_scan([m.ScannedInformationRoot(7,1,0,1,1),m.ScannedRoot(8,1,0,1,1)])

    def test_first_target_and_caps_use_both_actual_candidates(self):
        roots=[m.ScannedInformationRoot(7,i+1,0,None,1) for i in range(100)]
        self.assertIsNone(m.stop_reason(roots,0))
        both=[m.ScannedInformationRoot(7,i+1,0,1,1) for i in range(100)]
        self.assertEqual(m.stop_reason(both,0),"DEVIATION_TARGET")
        self.assertEqual(m.stop_reason(both,28800),"WALL_CAP")
        self.assertEqual(m.stop_reason([m.ScannedInformationRoot(7,i+1,0,None,None)
            for i in range(2000)],0),"ROOT_CAP")

    def test_information_summary_names_weighting_and_closed_screening(self):
        roots=[m.ScannedInformationRoot(7,i+1,0,1,1) for i in range(3)]+[
            m.ScannedInformationRoot(8,1,0,0,0)]
        bounds=[dict(public=(.5,.5),oracle=(.5,.5))]*3+[dict(public=(0.,0.),oracle=(0.,0.))]
        for weighting,expected in (("pooled_roots",.375),("equal_seed",.25)):
            report=m.summarize(roots,bounds,weighting=weighting,bootstrap_seed=7,bootstrap_draws=100)
            self.assertNotIn("leaves",report)
            self.assertEqual(set(report["candidates"]),{"public","oracle"})
            self.assertEqual(report["candidates"]["public"],report["candidates"]["oracle"])
            row=report["candidates"]["oracle"]
            self.assertEqual(row["identification_intervals"]["overall_effect"],(expected,expected))
            self.assertFalse(row["screening_authorized"])
            self.assertFalse(report["calibration_certified"])
            self.assertFalse(row["source_minimum_satisfied"])

    def test_unknown_information_arm_keeps_the_scanned_denominator(self):
        roots=[m.ScannedInformationRoot(7,1,0,None,0),m.ScannedInformationRoot(8,1,0,1,0)]
        bounds=[dict(public=(-1.,1.),oracle=(0.,0.)),dict(public=(.5,.5),oracle=(0.,0.))]
        report=m.summarize(roots,bounds,bootstrap_seed=7,bootstrap_draws=100)
        self.assertEqual(report["scanned_roots"],2)
        self.assertEqual(report["candidates"]["public"]["identification_intervals"]["overall_effect"],(-.25,.75))
        self.assertEqual(report["candidates"]["oracle"]["identification_intervals"]["overall_effect"],(0.,0.))


class PairedInformationSummaryTest(unittest.TestCase):
    def report(self, roots, intervals, **kwargs):
        return m.paired_information_summary(roots, intervals, bootstrap_seed=7, draws=100, **kwargs)

    def test_direct_alias_cancels_without_subtracting_marginal_bounds(self):
        roots = [m.ScannedInformationRoot(7, 1, 0, 2, 2),
                 m.ScannedInformationRoot(8, 1, 0, None, 2)]
        report = self.report(roots, [(0., 0.), (-1., 1.)])
        self.assertEqual(report["estimates"]["pooled_roots"]["identification_bounds"], [-.5, .5])
        self.assertEqual(report["identified_roots"], 1)
        self.assertEqual(report["unknown_action_pairs"], 1)
        self.assertEqual(report["known_action_disagreements"], 0)
        for pairs in ([(0., 0.), (0., 0.)], [(-1., 1.), (-1., 1.)]):
            with self.assertRaises(ValueError):
                self.report(roots, pairs)

    def test_pooled_roots_and_equal_sources_remain_explicit(self):
        roots = [m.ScannedInformationRoot(7, i, 0, 1, 2) for i in (1, 2, 3)] + [
            m.ScannedInformationRoot(8, 1, 0, 0, 0)]
        pairs = [(.5, .5)] * 3 + [(0., 0.)]
        report = self.report(roots, pairs)
        self.assertEqual(report["scanned_roots"], 4)
        self.assertEqual(report["source_clusters"], 2)
        self.assertEqual(report["known_action_disagreements"], 3)
        self.assertEqual(report["estimates"]["pooled_roots"]["identification_bounds"], [.375, .375])
        self.assertEqual(report["estimates"]["equal_seed"]["identification_bounds"], [.25, .25])
        self.assertEqual(report, self.report(roots, pairs))
        for key in ("screening_authorized", "calibration_certified", "scientific_strength_claim"):
            self.assertIs(report[key], False)

    def test_complete_disagreement_keeps_its_direct_sign(self):
        root = m.ScannedInformationRoot(7, 1, 0, 1, 2)
        report = self.report([root], [(-.25, -.25)])
        for estimate in report["estimates"].values():
            self.assertEqual(estimate["identification_bounds"], [-.25, -.25])
            self.assertEqual(estimate["descriptive_seed_bootstrap_95"], [-.25, -.25])

    def test_unknown_pairs_cannot_be_complete_case_zeros(self):
        for root in (m.ScannedInformationRoot(7, 1, 0, None, 2),
                     m.ScannedInformationRoot(7, 1, 0, 2, None),
                     m.ScannedInformationRoot(7, 1, 0, None, None)):
            self.assertEqual(self.report([root], [(-1., 1.)])["unknown_action_pairs"], 1)
            with self.assertRaisesRegex(ValueError, "unknown information"):
                self.report([root], [(0., 0.)])

    def test_invalid_rosters_and_intervals_reject(self):
        root = m.ScannedInformationRoot(7, 1, 0, 1, 2)
        for roots, pairs in (([], []), ([root], []), ([root, root], [(0., 0.)] * 2),
                             ([m.ScannedRoot(7, 1, 0, 1, 2)], [(0., 0.)])):
            with self.assertRaises(ValueError):
                self.report(roots, pairs)
        for pair in (None, "00", [0], [0, 0, 0], [True, 1], [-1, float("nan")],
                     [-1, float("inf")], [1, -1], [-2, 0]):
            with self.assertRaises(ValueError):
                self.report([root], [pair])

    def test_bootstrap_parameters_are_bound(self):
        root = m.ScannedInformationRoot(7, 1, 0, 1, 2)
        for seed, draws in ((True, 100), ("7", 100), (7, 19), (7, True), (7, 20.)):
            with self.assertRaises(ValueError):
                m.paired_information_summary([root], [(0., 0.)], bootstrap_seed=seed, draws=draws)


if __name__ == "__main__":
    unittest.main()
