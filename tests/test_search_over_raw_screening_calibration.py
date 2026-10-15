"""Synthetic fixture checks; no screening-rate calibration or scientific calls."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("staged_calibration_under_test",
    ROOT / "scripts/calibrate_search_over_raw_screening.py")
cal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cal
spec.loader.exec_module(cal)


class SharedCellTests(unittest.TestCase):
    def test_missing_raw_propagates_to_every_candidate(self):
        raw = np.array([[[[.5, 0.]]]])
        candidate = np.array([[[[1., .25]], [[.25, 1.]]]])
        lo, hi = cal.contrast_intervals(raw, candidate, np.ones(raw.shape, bool),
            np.zeros(candidate.shape, bool), np.zeros(candidate.shape, bool))
        np.testing.assert_array_equal(lo, candidate-1)
        np.testing.assert_array_equal(hi, candidate)

    def test_candidate_only_refusal_does_not_erase_known_raw(self):
        raw = np.full((1, 1, 1, 2), .5)
        candidate = np.ones((1, 2, 1, 2))
        lo, hi = cal.contrast_intervals(raw, candidate, np.zeros(raw.shape, bool),
            np.ones(candidate.shape, bool), np.zeros(candidate.shape, bool))
        np.testing.assert_array_equal(lo, np.full(candidate.shape, -.5))
        np.testing.assert_array_equal(hi, np.full(candidate.shape, .5))

    def test_alias_unknown_outcome_cancels_exactly(self):
        raw = np.full((1, 1, 1, 2), .5)
        candidate = np.ones((1, 2, 1, 2))
        lo, hi = cal.contrast_intervals(raw, candidate, np.ones(raw.shape, bool),
            np.ones(candidate.shape, bool), np.ones(candidate.shape, bool))
        self.assertTrue(np.all(lo == 0) and np.all(hi == 0))

    def test_quarter_cells_match_actual_eight_continuation_kernel(self):
        for candidate in (0., .25, .5, .75, 1.):
            for raw in (0., .25, .5, .75, 1.):
                for cm in (False, True):
                    for rm in (False, True):
                        rows = {}
                        for action, score, missing in ((0, raw, rm), (1, candidate, cm)):
                            for r in range(8):
                                rows[action, r] = dict(status="REFUSED" if missing else "COMPLETE",
                                    validated=not missing, score=None if missing else int(r < 8*score))
                        exact = cal.kernel.paired_root_interval(1, 0, rows)
                        low, high = cal.contrast_intervals(np.array(raw), np.array(candidate),
                            np.array(rm), np.array(cm), np.array(False))
                        self.assertEqual(exact, (float(low), float(high)))

    def test_extreme_conditional_probabilities_realize_marginal_effect(self):
        for base in (.05, .10, .15, .20):
            effects = np.array([base, base-.05, 0., -.05])
            p, qpos, qneg = cal.extreme_probabilities(base, effects)
            np.testing.assert_allclose(p*qpos-(1-p)*qneg, effects, atol=1e-15)
            self.assertEqual(qpos[0], 1.)
            self.assertEqual(qneg[0], 1.)

    def test_unrealizable_extreme_effect_rejected(self):
        with self.assertRaises(ValueError):
            cal.extreme_probabilities(.1, np.array([.9]))

    def test_score_cells_bounded_and_on_quarter_grid(self):
        families = [dict(generator="shared_control", mean=.1),
                    dict(generator="low_variance", mean=.1),
                    dict(generator="extreme", mean=.1),
                    dict(generator="nonidentical", mean=0.),
                    dict(mean=0., values=[-1., .25], probs=[.2, .8])]
        for family in families:
            effects = cal.configuration_effects(family, 3).reshape(3, 2, 6)
            raw, candidate = cal.draw_cells(family, np.random.default_rng(12), effects, 32)
            self.assertEqual(raw.shape, (3, 1, 1, 32))
            self.assertEqual(candidate.shape, (3, 2, 6, 32))
            for cells in (raw, candidate):
                self.assertTrue(np.all((cells >= 0) & (cells <= 1)))
                self.assertTrue(np.all(cells*4 == np.rint(cells*4)))

    def test_pipeline_stop_has_common_suffix_and_no_complete_case_drop(self):
        family = dict(generator="shared_control", mean=.1, stop_probability=1.)
        effects = cal.configuration_effects(family, 2).reshape(2, 2, 6)
        lo, hi, stopped = cal.draw_panel(family, np.random.default_rng(5), effects, 32)
        self.assertEqual(lo.shape, (2, 2, 6, 32))
        self.assertTrue(np.all(lo == -1) and np.all(hi == 1) and np.all(stopped))


class StageTests(unittest.TestCase):
    def setUp(self):
        self.means = np.array([[[.1, .4, .8], [.2, .3, .5]],
                               [[.2, .1, .3], [.1, .7, .9]]])

    def test_real_stages_freeze_a2_leaf_even_when_other_a3_is_better(self):
        choices = cal.select_trial(self.means)
        self.assertEqual(choices, (("incumbent", "hp_fraction", 10.), ("reference", "model", 10.)))

    def test_a1_failure_and_fixed_diagnostic_failure_cannot_block(self):
        expected = cal.select_trial(self.means)
        for overlay in ("a1_unqualified", "fixed_diagnostic_incomplete"):
            self.assertEqual(cal.select_trial(self.means, overlay), expected)

    def test_a2_zero_work_is_local(self):
        means = self.means.copy()
        means[0, 0, 0] = .9
        self.assertEqual(cal.select_trial(means)[0][1], "model")
        self.assertEqual(cal.select_trial(means, "a2_incumbent_model_unqualified")[0][1], "hp_fraction")
        self.assertEqual(cal.select_trial(means, "a2_incumbent_model_unqualified")[1],
                         cal.select_trial(means)[1])

    def test_a3_zero_work_is_local(self):
        choices = cal.select_trial(self.means, "a3_reference_10_unqualified")
        self.assertEqual(choices[1], ("reference", "model", 1.))
        self.assertEqual(choices[0], cal.select_trial(self.means)[0])

    def test_unavailable_arm_does_not_block_other(self):
        choices = cal.select_trial(self.means, "incumbent_unavailable")
        self.assertEqual(choices[0], ("incumbent", None, None))
        self.assertEqual(choices[1], cal.select_trial(self.means)[1])

    def test_operational_stop_prevents_both_selections(self):
        self.assertEqual(cal.select_trial(self.means, stopped=True),
                         (("incumbent", None, None), ("reference", None, None)))

    def test_exact_tie_policy(self):
        self.assertEqual(cal.select_trial(np.zeros((2, 2, 3))),
                         (("incumbent", "model", 1.), ("reference", "model", 1.)))

    def test_calls_actual_selection_kernels(self):
        with patch.object(cal.kernel, "choose_evaluators", wraps=cal.kernel.choose_evaluators) as a2:
            with patch.object(cal.kernel, "choose_ceilings", wraps=cal.kernel.choose_ceilings) as a3:
                cal.select_trial(self.means)
        a2.assert_called_once()
        a3.assert_called_once()

    def test_effect_map_and_only_reference_positive(self):
        family = dict(mean=.1, incumbent_mean=0., inferior_config_gap=.05)
        effects = cal.configuration_effects(family, 1)
        self.assertEqual(effects[0, 0, 1, 2], 0.)
        self.assertEqual(effects[0, 0, 0, 0], -.05)
        self.assertEqual(effects[0, 1, 1, 2], .1)
        self.assertEqual(effects[0, 1, 0, 0], .05)


class BootstrapTests(unittest.TestCase):
    def test_all_declared_sizes_match_exact_kernel_on_admitted_grid(self):
        for n in cal.SIZES:
            bank = cal.bootstrap_bank(n)
            np.testing.assert_array_equal(bank.sum(axis=0), np.full(10000, n))
            self.assertTrue(cal.parity_check(bank)["exact_endpoint_and_decision"])

    def test_nonquarter_grid_cannot_claim_accelerated_parity(self):
        bank = np.ones((32, 10000), np.float32)
        with self.assertRaises(ValueError):
            cal.bootstrap_endpoints(np.full((2, 32), 1/7), bank)

    def test_wilson_zero_count_and_empty_denominator(self):
        self.assertGreater(cal.wilson(0, 5000)["mc_interval95"][1], 0)
        self.assertIsNone(cal.wilson(0, 0)["rate"])


class WholeProcedureFixtureTests(unittest.TestCase):
    def run_fixture(self, overlay="none", validation_stop=False, exploration_stop=False):
        family = dict(name="fixture", mean=.1, inferior_config_gap=.05)
        # Every exploration score ties, making model/1 the frozen choice.
        exploration = (np.zeros((2, 2, 6, 32)), np.zeros((2, 2, 6, 32)),
                       np.full(2, exploration_stop))
        validation = (np.full((2, 2, 1, 32), .25), np.full((2, 2, 1, 32), .25),
                      np.full(2, validation_stop))
        bank = np.ones((32, 10000), np.float32)
        with patch.object(cal, "draw_panel", side_effect=[exploration, validation]) as panel:
            result = cal.run_condition(family, 32, bank, np.random.default_rng(1), overlay, replicates=2)
        self.assertEqual(panel.call_args_list[0].args[2].shape, (2, 2, 6))
        self.assertEqual(panel.call_args_list[1].args[2].shape, (2, 2, 1))
        return result

    def test_frozen_a2_a3_selected_effect_used_for_independent_panel(self):
        result = self.run_fixture()
        for arm in cal.kernel.ARMS:
            self.assertEqual(result["arms"][arm]["selected_configs"], {"model/1": 2})
            self.assertEqual(result["arms"][arm]["selected_effect_counts"], [{"effect": .05, "count": 2}])
            self.assertEqual(result["arms"][arm]["true_screening"]["count"], 2)

    def test_validation_operational_stop_blocks_even_positive_endpoint(self):
        result = self.run_fixture(validation_stop=True)
        self.assertEqual(result["any_true_screening"]["count"], 0)
        self.assertEqual(result["validation_stop"]["count"], 2)
        self.assertEqual(result["any_true_screening"]["denominator"], 2)

    def test_exploration_operational_stop_preserves_unavailable_denominator(self):
        result = self.run_fixture(exploration_stop=True, validation_stop=True)
        self.assertEqual(result["validation_attempted"]["count"], 0)
        self.assertEqual(result["validation_stop"]["count"], 0)
        self.assertIsNone(result["validation_stop_given_attempt"]["rate"])
        for arm in cal.kernel.ARMS:
            row = result["arms"][arm]
            self.assertEqual(row["unavailable"]["count"], 2)
            self.assertEqual(row["true_screening"]["count"], 0)
            self.assertEqual(row["selected_effect_counts"], [])
            self.assertEqual(row["mean_identification_width"], 2.)

    def test_unavailable_arm_not_false_null_observation_or_denominator_drop(self):
        result = self.run_fixture("incumbent_unavailable")
        inc, ref = (result["arms"][arm] for arm in cal.kernel.ARMS)
        self.assertEqual(inc["unavailable"]["count"], 2)
        self.assertEqual(inc["false_screening"]["count"], 0)
        self.assertEqual(inc["false_screening"]["denominator"], 2)
        self.assertEqual(ref["true_screening"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
