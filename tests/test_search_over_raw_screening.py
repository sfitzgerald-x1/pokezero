"""Synthetic-only reconciled kernels; avoid package model/runtime imports."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import sys
import unittest

MODULE = Path(__file__).resolve().parents[1] / "src/pokezero/mcts_eval/search_over_raw_screening.py"
spec = importlib.util.spec_from_file_location("screening_kernel_under_test", MODULE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ScreeningTest(unittest.TestCase):
    def setUp(self):
        self.design = m.ScreeningDesign("e20d0e8d-8d43-4536-b8c1-a5e60f4cbd34", tuple(range(32)))

    def run_screen(self, reference, incumbent=None):
        if incumbent is None:
            incumbent = reference
        return m.screen(self.design, dict(incumbent=incumbent, reference=reference))

    def test_noiseless_ten_points_screens_at_32_not_betting(self):
        result = self.run_screen([(.1, .1)] * 32)
        self.assertTrue(result["both_arms_screen_positive"])
        self.assertEqual(result["arms"]["reference"]["nominal_alpha"], .05)
        for key in ("phase_b_authorized", "collection_authorized", "confirmatory_strength",
                    "family_error_control_claim", "calibration_certified"):
            self.assertFalse(result[key])

    def test_zero_ties_and_negative_do_not_screen(self):
        for value in (0., -.1):
            self.assertFalse(self.run_screen([(value, value)] * 32)["any_arm_screen_positive"])

    def test_independent_arm_decisions_no_joint_success_prerequisite(self):
        result = self.run_screen([(.1, .1)] * 32, [(0., 0.)] * 32)
        self.assertTrue(result["any_arm_screen_positive"])
        self.assertFalse(result["both_arms_screen_positive"])

    def test_all_missing_retains_uncertainty_and_full_seed_count(self):
        result = self.run_screen([(-1., 1.)] * 32)
        row = result["arms"]["reference"]
        self.assertEqual(row["source_seeds"], 32)
        self.assertEqual(row["uncertain_seeds"], 32)
        self.assertEqual(row["identification_interval"], [-1., 1.])
        self.assertFalse(row["screen_positive"])

    def test_unavailable_arm_is_explicit_not_dropped(self):
        result = m.screen(self.design, {"incumbent": None, "reference": [(.1, .1)] * 32})
        self.assertEqual(result["arms"]["incumbent"]["status"], "UNAVAILABLE")
        self.assertTrue(result["any_arm_screen_positive"])
        self.assertFalse(result["both_arms_screen_positive"])

    def test_shared_indices_and_exact_order_statistic(self):
        values = tuple(.2 if i < 24 else -.5 for i in range(32))
        result = self.run_screen([(x, x) for x in values], [(-x, -x) for x in values])
        rng = random.Random(self.design.random_seed)
        means = sorted(sum(values[i] for i in rng.choices(range(32), k=32)) / 32 for _ in range(10000))
        self.assertAlmostEqual(result["arms"]["reference"]["lower_percentile_endpoint"], means[500])
        self.assertAlmostEqual(result["arms"]["incumbent"]["lower_percentile_endpoint"], -means[9499])

    def test_outcome_dependent_uncertainty_only_lowers_bootstrap(self):
        values = [(.4, .4)] * 24 + [(-.2, -.2)] * 8
        changed = [(-1., 1.)] * 4 + values[4:]
        a, b = self.run_screen(values), self.run_screen(changed)
        self.assertLessEqual(b["arms"]["reference"]["lower_percentile_endpoint"],
                             a["arms"]["reference"]["lower_percentile_endpoint"])

    def test_invalid_design_and_short_roster_rejected(self):
        for seeds in (tuple(range(31)), (True,) + tuple(range(1, 32)), (0,) * 32):
            with self.assertRaises(ValueError):
                m.ScreeningDesign(self.design.namespace, seeds)
        with self.assertRaises(ValueError):
            m.ScreeningDesign("not-a-namespace", self.design.seeds)
        for rows in ([(-1., 1.)] * 31, [(float("nan"), 1.)] * 32, [(True, 1.)] * 32):
            with self.assertRaises(ValueError):
                self.run_screen(rows)
        with self.assertRaises(ValueError):
            m.screen(self.design, {"reference": [(0., 0.)] * 32})


class PairedAccountingTest(unittest.TestCase):
    def complete(self, score):
        return dict(status="COMPLETE", validated=True, score=score)

    def test_shared_raw_refusal_enters_both_contrasts(self):
        rows = {(action, rep): self.complete(1.) for action in (1, 2) for rep in range(8)}
        rows.update({(0, rep): dict(status="REFUSED", validated=False, score=None) for rep in range(8)})
        self.assertEqual(m.paired_root_interval(1, 0, rows), (0., 1.))
        self.assertEqual(m.paired_root_interval(2, 0, rows), (0., 1.))

    def test_identical_action_alias_is_zero_with_unknown_outcomes(self):
        self.assertEqual(m.paired_root_interval(0, 0, {}), (0., 0.))
        with self.assertRaises(ValueError):
            m.paired_root_interval(0, 0, {(0, 0): dict(status="REFUSED", validated=True, score=1.)})

    def test_unvalidated_complete_and_missing_are_uncertain(self):
        rows = {(a, r): dict(status="COMPLETE", validated=False, score=1.) for a in (0, 1) for r in range(8)}
        self.assertEqual(m.paired_root_interval(1, 0, rows), (-1., 1.))
        self.assertEqual(m.paired_root_interval(1, 0, {}), (-1., 1.))

    def test_paired_replicates_and_partial_refusal(self):
        rows = {(a, r): self.complete(float(a)) for a in (0, 1) for r in range(8)}
        self.assertEqual(m.paired_root_interval(1, 0, rows), (1., 1.))
        rows.pop((1, 0))
        self.assertEqual(m.paired_root_interval(1, 0, rows), (7/8, 1.))

    def test_fixed_seed_weight_not_root_weight_and_missing_slots(self):
        roster = tuple((s, tuple(f"{s}:{r}" for r in range(2 if s == 0 else 1))) for s in range(32))
        roots = {root: (1., 1.) if seed == 0 else (0., 0.) for seed, quota in roster for root in quota}
        rows = m.cluster_intervals(roster, roots)
        self.assertEqual(sum(row[0] for row in rows) / 32, 1/32)
        roots.pop("0:0")
        changed = m.cluster_intervals(roster, roots)
        self.assertEqual(changed[0], (0., 1.))
        self.assertEqual(len(changed), 32)

    def test_duplicate_or_unregistered_roots_are_rejected(self):
        roster = tuple((s, (str(s),)) for s in range(32))
        with self.assertRaises(ValueError):
            m.cluster_intervals(roster, {"extra": (0., 0.)})
        with self.assertRaises(ValueError):
            m.cluster_intervals(roster[:-1] + ((31, ("0",)),), {})


class StageSelectionTest(unittest.TestCase):
    def a2(self):
        return [m.StageSummary("A2", a, "public", l, 1., .1, 64, "QUALIFIED")
                for a in m.ARMS for l in m.DEPLOYABLE_LEAVES]

    def test_a2_independent_of_a1_and_failed_configuration(self):
        rows = self.a2()
        rows[0] = replace(rows[0], accepted_units=0, status="UNQUALIFIED_AT_CEILING")
        self.assertEqual(m.choose_evaluators(rows), (("incumbent", "hp_fraction"), ("reference", "model")))

    def test_diagnostic_oracle_and_terminal_cannot_win_deployable_selection(self):
        rows = self.a2() + [m.StageSummary("A2", "incumbent", "oracle", "model", 1., 1., 64, "QUALIFIED",
            lane="wall_clock_diagnostic"),
            m.StageSummary("A2", "reference", "public", "raw_rollout", None, 1., 64, "QUALIFIED",
            lane="fixed_work_diagnostic")]
        self.assertEqual(m.choose_evaluators(rows), (("incumbent", "model"), ("reference", "model")))

    def test_fixed_work_model_and_hp_cannot_change_deployment_selection(self):
        rows = self.a2()
        rows[0] = replace(rows[0], lower_mean=.2)
        rows += [m.StageSummary("A2", a, "public", leaf, None, 1., 64, "QUALIFIED",
                 lane="fixed_work_diagnostic") for a in m.ARMS for leaf in m.DEPLOYABLE_LEAVES]
        self.assertEqual(m.choose_evaluators(rows), (("incumbent", "model"), ("reference", "model")))

    def test_diagnostic_row_cannot_replace_missing_deployment_row(self):
        rows = self.a2()
        rows[0] = replace(rows[0], seconds=None, lane="fixed_work_diagnostic")
        with self.assertRaisesRegex(ValueError, "exact public A2 roster"):
            m.choose_evaluators(rows)

    def test_fixed_work_has_no_ceiling_and_full_accepted_target(self):
        row = m.StageSummary("A2", "incumbent", "public", "raw_rollout", None,
            .1, 64, "QUALIFIED", lane="fixed_work_diagnostic")
        for changed in (dict(seconds=1.), dict(accepted_units=63), dict(stage="A3")):
            with self.assertRaises(ValueError):
                replace(row, **changed)
        with self.assertRaises(ValueError):
            replace(row, seconds=1., lane="deployment")

    def test_incomplete_fixed_work_diagnostic_is_nonblocking_not_ceiling_failure(self):
        rows = self.a2()
        rows.append(m.StageSummary("A2", "incumbent", "public", "raw_rollout", None,
            -1., 12, "DIAGNOSTIC_INCOMPLETE", lane="fixed_work_diagnostic"))
        self.assertEqual(m.choose_evaluators(rows), (("incumbent", "model"), ("reference", "model")))
        with self.assertRaises(ValueError):
            replace(rows[-1], status="UNQUALIFIED_AT_CEILING")
        with self.assertRaises(ValueError):
            replace(rows[0], status="DIAGNOSTIC_INCOMPLETE")
        with self.assertRaises(ValueError):
            replace(rows[0], status="UNSTARTED")

    def test_unavailable_one_arm_does_not_block_other(self):
        rows = [replace(r, accepted_units=0, status="UNQUALIFIED_AT_CEILING")
                if r.arm == "incumbent" else r for r in self.a2()]
        choices = m.choose_evaluators(rows)
        self.assertEqual(choices, (("incumbent", None), ("reference", "model")))
        a3 = [m.StageSummary("A3", "reference", "public", "model", c, .1, 64, "QUALIFIED") for c in (1., 3., 10.)]
        self.assertEqual(m.choose_ceilings(choices, a3), (("incumbent", None, None), ("reference", "model", 1.)))

    def test_a3_selects_best_qualified_ceiling_smaller_breaks_tie(self):
        choices = m.choose_evaluators(self.a2())
        rows = [m.StageSummary("A3", a, "public", "model", c, .2 if c > 1 else .1, 64, "QUALIFIED")
                for a in m.ARMS for c in (1., 3., 10.)]
        rows[-1] = replace(rows[-1], accepted_units=0, status="UNQUALIFIED_AT_CEILING")
        self.assertEqual(m.choose_ceilings(choices, rows), (("incumbent", "model", 3.), ("reference", "model", 3.)))

    def test_integrity_failure_is_not_local_unqualification(self):
        rows = self.a2()
        rows[0] = replace(rows[0], status="OPERATIONAL_STOP")
        with self.assertRaisesRegex(ValueError, "integrity"):
            m.choose_evaluators(rows)

    def test_zero_work_cannot_claim_qualified(self):
        with self.assertRaisesRegex(ValueError, "accepted work"):
            replace(self.a2()[0], accepted_units=0)

    def test_duplicate_or_missing_qualifications_rejected(self):
        for rows in (self.a2() + self.a2()[:1], self.a2()[:-1]):
            with self.assertRaises(ValueError):
                m.choose_evaluators(rows)

    def test_a3_cannot_reselect_leaf_promote_oracle_or_omit_ceiling(self):
        choices = m.choose_evaluators(self.a2())
        rows = [m.StageSummary("A3", a, "public", "model", c, .1, 64, "QUALIFIED") for a in m.ARMS for c in (1., 3., 10.)]
        for changed in (rows[:-1], [replace(rows[0], leaf="hp_fraction")] + rows[1:]):
            with self.assertRaises(ValueError):
                m.choose_ceilings(choices, changed)
        with self.assertRaises(ValueError):
            replace(rows[0], belief="oracle")


if __name__ == "__main__":
    unittest.main()
