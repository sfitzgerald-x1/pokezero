"""Authored budget arithmetic; no collection or scientific launch."""
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
summarize_budget=importlib.import_module(PACKAGE+".search_over_raw_budget16384").summarize_budget


class BudgetArithmeticTests(unittest.TestCase):
    def row(self,seed,index,member,pair,band="in_band",valid=True):
        return dict(seed=seed,request_index=index,member=member,contrast=pair,band=band,source_validated=valid)

    def test_overall_is_not_conditional_or_complete_case(self):
        rows=[self.row(1,1,True,[1.,1.]),self.row(1,2,False,[0.,0.]),
            self.row(2,1,None,[-1.,1.],valid=False)]
        result=summarize_budget(rows,bootstrap_seed=4,draws=100)
        self.assertEqual(result["estimates"]["pooled_roots"]["identification_bounds"],[0.,2/3])
        self.assertEqual(result["conditional_deviations"]["estimates"]["pooled_roots"]["identification_bounds"],[0.,1.])
        self.assertEqual(result["identified_roots"],2)
        self.assertEqual(result["scanned_roots"],3)
        self.assertFalse(result["candidate_nominated"])

    def test_margin_split_does_not_drop_zero_contribution_sources(self):
        rows=[self.row(1,1,True,[1.,1.]),self.row(2,1,True,[-1.,-1.],"above")]
        result=summarize_budget(rows,bootstrap_seed=4,draws=100)
        target=result["margin_strata"]["in_band"]["conditional_deviations"]
        self.assertEqual(target["known_members"],1)
        self.assertGreater(target["estimates"]["pooled_roots"]["empty_denominator_resamples"],0)
        self.assertEqual(result["estimates"]["pooled_roots"]["identification_bounds"],[0.,0.])

    def test_failed_source_can_change_stratum_membership(self):
        result=summarize_budget([self.row(1,1,None,[-1.,1.],"above",False)],bootstrap_seed=4,draws=100)
        for band in result["margin_strata"].values():
            self.assertEqual(band["scanned_contrast"]["unknown_membership"],1)
            self.assertEqual(band["conditional_deviations"]["unknown_membership"],1)

    def test_all_aliases_have_overall_zero_and_undefined_conditional(self):
        result=summarize_budget([self.row(1,1,False,[0.,0.])],bootstrap_seed=4,draws=100)
        self.assertEqual(result["estimates"]["pooled_roots"]["identification_bounds"],[0.,0.])
        self.assertEqual(result["conditional_deviations"]["ratio_status"],"UNDEFINED_NO_DEVIATIONS")

    def test_invalid_alias_or_source_membership_is_rejected(self):
        for row in (self.row(1,1,False,[.5,.5]),self.row(1,1,True,[.5,.5],valid=False),
                self.row(1,1,None,[0.,0.])):
            with self.subTest(row=row),self.assertRaises(ValueError):
                summarize_budget([row],bootstrap_seed=4,draws=100)
