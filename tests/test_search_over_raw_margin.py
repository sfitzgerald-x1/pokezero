"""Authored stopped margin arithmetic; no games, labels or calibration runs."""
import unittest
import importlib
from pathlib import Path
import sys
import types

PACKAGE="margin_arithmetic_test_modules"
if PACKAGE not in sys.modules:
    package=types.ModuleType(PACKAGE)
    package.__path__=[str(Path(__file__).resolve().parents[1]/"src/pokezero/mcts_eval")]
    sys.modules[PACKAGE]=package
kernel=importlib.import_module(PACKAGE+".search_over_raw_margin")
in_band,member,raw_margin,ratio_bounds,summarize=(getattr(kernel,name) for name in
    ("in_band","member","raw_margin","ratio_bounds","summarize"))


class MarginArithmeticTests(unittest.TestCase):
    def row(self,seed,index,membership,pair):
        return dict(seed=seed,request_index=index,member=membership,contrast=pair)

    def test_margin_uses_two_largest_legal_probabilities(self):
        self.assertAlmostEqual(raw_margin([.6,0.,.4],[0,2]),.2)
        self.assertIsNone(raw_margin([0.,1.],[1]))
        self.assertTrue(in_band(.1));self.assertTrue(in_band(.3))
        self.assertFalse(in_band(.099999));self.assertFalse(in_band(.300001))
        for prior,legal in (([.6,.4],[0]),([True,0.],[0,1]),([float("nan"),0.],[0,1])):
            with self.assertRaises(ValueError):raw_margin(prior,legal)

    def test_non_deviation_alias_does_not_enter_conditional_denominator(self):
        self.assertFalse(member(.2,2,2));self.assertTrue(member(.2,2,3))
        self.assertIsNone(member(.2,2,None));self.assertFalse(member(.8,2,None))
        groups=[[self.row(1,1,False,[0.,0.]),self.row(1,2,True,[.5,.5])]]
        self.assertEqual(ratio_bounds(groups),[.5,.5])

    def test_unknown_membership_can_enter_or_leave_ratio(self):
        groups=[[self.row(1,1,True,[.5,.5]),self.row(1,2,None,[-1.,1.])]]
        self.assertEqual(ratio_bounds(groups),[-.25,.75])

    def test_missing_labels_remain_full_range_on_known_deviation(self):
        self.assertEqual(ratio_bounds([[self.row(1,1,True,[-1.,1.])]]),[-1.,1.])

    def test_zero_contribution_seeds_are_retained_in_bootstrap(self):
        rows=[self.row(1,1,True,[1.,1.]),self.row(2,1,False,[0.,0.])]
        result=summarize(rows,bootstrap_seed=4,draws=100)
        self.assertEqual(result["scanned_source_clusters"],2)
        self.assertEqual(result["deviation_source_clusters"],1)
        self.assertEqual(result["estimates"]["pooled_roots"]["identification_bounds"],[1.,1.])
        self.assertEqual(result["estimates"]["pooled_roots"]["descriptive_seed_bootstrap_95"],[-1.,1.])
        self.assertGreater(result["estimates"]["pooled_roots"]["empty_denominator_resamples"],0)

    def test_no_deviations_is_undefined_not_zero(self):
        result=summarize([self.row(1,1,False,[0.,0.])],bootstrap_seed=4,draws=100)
        self.assertEqual(result["conditional_mean_status"],"UNDEFINED_NO_DEVIATIONS")
        self.assertEqual(result["estimates"]["pooled_roots"]["identification_bounds"],[-1.,1.])

    def test_source_balanced_ratio_is_not_average_of_deviation_means(self):
        groups=[[self.row(1,1,True,[1.,1.]),self.row(1,2,False,[0.,0.])],
            [self.row(2,1,True,[-1.,-1.])]]
        self.assertEqual(ratio_bounds(groups),[0.,0.])
        self.assertAlmostEqual(ratio_bounds(groups,"equal_seed")[0],-1/3)

    def test_duplicate_identity_and_unknown_as_zero_refuse(self):
        row=self.row(1,1,True,[.5,.5])
        with self.assertRaises(ValueError):summarize([row,row],bootstrap_seed=4,draws=100)
        with self.assertRaises(ValueError):ratio_bounds([[self.row(1,1,None,[0.,0.])]])
