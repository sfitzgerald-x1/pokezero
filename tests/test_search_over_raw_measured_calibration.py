import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measured_calibration_test",
    ROOT / "scripts/calibrate_search_over_raw_measured.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class MeasuredCalibrationTests(unittest.TestCase):
    def test_constant_panel_has_zero_variance_without_pseudoreplicating(self):
        groups = [np.full((7 if i < 8 else 6, 2), .125) for i in range(32)]
        result = cal.variance_components(groups)
        self.assertEqual(sum(result["seed_root_counts"]), 200)
        np.testing.assert_array_equal(result["pooled_within_seed_covariance"], np.zeros((2, 2)))
        np.testing.assert_array_equal(result["between_seed_covariance_mom_psd"], np.zeros((2, 2)))
        self.assertEqual(result["mean"], [.125, .125])

    def test_perfect_within_seed_dependence_stays_between_seeds(self):
        groups = [np.full((7 if i < 8 else 6, 1), .25 if i % 2 else -.25) for i in range(32)]
        result = cal.variance_components(groups)
        self.assertEqual(result["pooled_within_seed_covariance"], [[0.]])
        self.assertAlmostEqual(result["observed_seed_mean_covariance"][0][0], .0625 * 32 / 31)
        self.assertAlmostEqual(result["equal_seed_mean_sampling_covariance"][0][0], .0625 / 31)

    def test_missing_roots_do_not_yield_complete_case_variance(self):
        cfg = dict(arm="incumbent", belief="public", leaf="model", seconds=1., workers=1)
        slots = [dict(root_id=f"r{i}", source_seed=i % 32) for i in range(200)]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / "completed-roots").mkdir()
            (p / "entry.json").write_text(json.dumps(dict(contract=dict(configurations=[cfg],
                panels=dict(exploration=dict(seeds=list(range(32)), root_slots=slots))))))
            (p / "readout.json").write_text(json.dumps(dict(recorded_roots=0, rejected_root_receipts=[],
                leaves=dict(model=dict(fixed_seed_weighted_mean_bounds=[-1, 1])))))
            metadata, groups = cal.load_panel(p / "readout.json")
            self.assertIsNone(groups)
            self.assertEqual(metadata["uncertain_config_root_contrasts"], 200)
            self.assertTrue(metadata["fixed_denominator"])

    def test_selection_uses_independent_holdout_and_declares_missing_reference(self):
        groups = [np.zeros((7 if i < 8 else 6, 2)) for i in range(32)]
        configs = [dict(arm="incumbent", leaf=leaf) for leaf in ("model", "hp_fraction")]
        result = cal.simulate(cal.variance_components(groups), configs, replicates=2, draws=20)
        self.assertEqual(result["results"][0]["per_arm"]["incumbent"]["rate"], 0.)
        self.assertEqual(result["results"][1]["per_arm"]["incumbent"]["rate"], 1.)
        self.assertFalse(result["both_plan_arms_measured"])
        self.assertFalse(result["holdout_opened"])
        self.assertTrue(result["independent_exploration_and_holdout"])


if __name__ == "__main__":
    unittest.main()
