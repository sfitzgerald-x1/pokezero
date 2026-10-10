import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibrate_search_over_raw_joint import (
    CAPS, METHODS, SCENARIOS, check_scalar_parity, generate, rate, run,
    vector_adjudicate, write_new,
)

from pokezero.mcts_eval.search_over_raw_statistics import (
    ARMS, JointDesign, adjudicate, log_wealth, lower_confidence_bound,
    primary_cluster_intervals,
)


class JointStatisticsTest(unittest.TestCase):
    def design(self, **kwargs):
        return JointDesign(seeds=tuple(range(64)), looks=(16, 32, 48, 64), **kwargs)

    def rows(self, n=64, d=.1):
        return [dict(seed=i, contrasts={arm: (d, d) for arm in ARMS},
            refused=False, provenance_ok=True) for i in range(n)]

    def cell(self, score=1, status="COMPLETE", validated=True):
        return dict(score=score, status=status, validated=validated)

    def test_shared_raw_missing_harms_both_lower_endpoints(self):
        cells = {(arm, seat): self.cell(1 if arm != "raw" else 0)
            for arm in ("raw", *ARMS) for seat in ("p1", "p2")}
        self.assertEqual(primary_cluster_intervals(cells), {arm: (1, 1) for arm in ARMS})
        cells[("raw", "p1")] = self.cell(None, "REFUSED", False)
        self.assertEqual(primary_cluster_intervals(cells), {arm: (.5, 1) for arm in ARMS})

    def test_unvalidated_complete_is_uncertain_not_scored(self):
        cells = {("reference", "p1"): self.cell(1, validated=False)}
        self.assertEqual(primary_cluster_intervals(cells), {arm: (-1, 1) for arm in ARMS})
        self.assertEqual(primary_cluster_intervals({}), {arm: (-1, 1) for arm in ARMS})

    def test_invalid_cells_never_silently_score(self):
        for cell in (self.cell(True), self.cell(.25), self.cell(float("nan")),
                     self.cell(None, "REFUSED", True), dict(status="COMPLETE", score=1)):
            with self.assertRaises(ValueError):
                primary_cluster_intervals({("raw", "p1"): cell})
        with self.assertRaises(ValueError):
            primary_cluster_intervals({("raw", "p3"): self.cell()})

    def test_design_requires_fixed_unique_order_and_registered_looks(self):
        cases = [dict(seeds=tuple(range(63))), dict(seeds=(0,) * 64),
            dict(seeds=(True, *range(1, 64))), dict(looks=(64, 16)), dict(looks=(16, 32)),
            dict(looks=(16, 16, 64)), dict(looks=(True, 64))]
        for change in cases:
            kwargs = dict(seeds=tuple(range(64)), looks=(64,))
            kwargs.update(change)
            with self.assertRaises(ValueError):
                JointDesign(**kwargs)

    def test_uncalibrated_ttest_and_invalid_stakes_are_rejected(self):
        for change in (dict(method="t_test"), dict(stakes=(0., .4)), dict(stakes=(1., .4)),
            dict(stakes=(float("nan"), .4)), dict(alpha=True), dict(alpha=.051),
            dict(refusal_cap=True)):
            with self.assertRaises(ValueError):
                self.design(**change)

    def test_betting_uses_forty_not_twenty_for_two_claims(self):
        design = self.design()
        result = adjudicate(design, self.rows(32, .25))
        self.assertGreater(log_wealth([(.25, .25)] * 32, .4), math.log(20))
        self.assertLess(log_wealth([(.25, .25)] * 32, .4), math.log(40))
        self.assertEqual(result["primary_crossings"], {arm: None for arm in ARMS})
        result = adjudicate(design, self.rows(48, .25))
        self.assertEqual(result["primary_crossings"], {arm: 48 for arm in ARMS})
        self.assertEqual(result["status"], "BOTH_PRIMARY_THRESHOLDS_CROSSED")

    def test_unknown_endpoints_cannot_increase_betting_wealth(self):
        known = [(-.25, -.25), (.25, .25), (1, 1), (0, 0)]
        for index in range(len(known)):
            unknown = list(known)
            unknown[index] = (-1, 1)
            self.assertLessEqual(log_wealth(unknown, .4), log_wealth(known, .4))

    def test_bernstein_range_two_and_uncertain_abstention(self):
        n, alpha = 64, .025
        expected = .5 - 14 * math.log(2 / alpha) / (3 * (n - 1))
        self.assertAlmostEqual(lower_confidence_bound([(.5, .5)] * n, alpha,
            "empirical_bernstein"), expected)
        self.assertIsNone(lower_confidence_bound([(.5, .5), (-1, 1)], alpha,
            "empirical_bernstein"))

    def test_hoeffding_uses_lower_endpoints_and_range_two(self):
        rows = [(.5, 1.)] * 64
        self.assertAlmostEqual(lower_confidence_bound(rows, .025, "hoeffding"),
            .5 - math.sqrt(2 * math.log(40) / 64))

    def test_fixed_time_bounds_split_alpha_over_all_looks(self):
        result = adjudicate(self.design(method="hoeffding"), self.rows(32, .5))
        expected = .5 - math.sqrt(2 * math.log(160) / 32)
        self.assertAlmostEqual(result["latest_lower_bounds"]["reference"], expected)
        self.assertIsNone(result["primary_crossings"]["reference"])

    def test_third_refused_cluster_stops_before_look(self):
        rows = self.rows(3)
        for row in rows:
            row["refused"] = True
            row["contrasts"] = {arm: (-1, 1) for arm in ARMS}
        result = adjudicate(self.design(), rows)
        self.assertEqual(result["status"], "REFUSAL_CAP_EXCEEDED")
        self.assertEqual(result["refused_clusters"], 3)
        with self.assertRaisesRegex(ValueError, "terminal"):
            adjudicate(self.design(), rows + self.rows(4)[3:])

    def test_provenance_failure_is_terminal_not_an_isolated_refusal(self):
        rows = self.rows(1)
        rows[0]["provenance_ok"] = False
        self.assertEqual(adjudicate(self.design(), rows)["status"],
            "INVALID_PROVENANCE_OR_WORKER_STATE")
        with self.assertRaisesRegex(ValueError, "terminal"):
            adjudicate(self.design(), rows + self.rows(2)[1:])

    def test_uncertainty_cannot_evade_refusal_cap_but_suffix_is_not_refused(self):
        rows = self.rows(1)
        rows[0]["contrasts"]["reference"] = (-1, 1)
        with self.assertRaisesRegex(ValueError, "uncertain attempted"):
            adjudicate(self.design(), rows)
        rows[0]["refused"] = True
        self.assertEqual(adjudicate(self.design(), rows)["refused_clusters"], 1)
        self.assertEqual(adjudicate(self.design(), [])["refused_clusters"], 0)
        # Secondary-opponent failure still counts with complete primary cells.
        rows = self.rows(1)
        rows[0]["refused"] = True
        self.assertEqual(adjudicate(self.design(), rows)["refused_clusters"], 1)

    def test_prefix_preserves_full_roster_denominator(self):
        result = adjudicate(self.design(), self.rows(1, 1))
        self.assertEqual(result["full_roster_identification_intervals"]["reference"],
            [-62 / 64, 1])
        result = adjudicate(self.design(), [])
        self.assertEqual(result["full_roster_identification_intervals"],
            {arm: [-1, 1] for arm in ARMS})

    def test_reordered_missing_or_extra_records_fail(self):
        for rows in (list(reversed(self.rows(2))), self.rows(65)):
            with self.assertRaises(ValueError):
                adjudicate(self.design(), rows)
        rows = self.rows(1)
        rows[0]["contrasts"].pop("raw", None)
        rows[0]["contrasts"].pop("reference")
        with self.assertRaises(ValueError):
            adjudicate(self.design(), rows)

    def test_no_admission_retry_or_holm_claim(self):
        rows = self.rows(64, 0)
        before = copy.deepcopy(rows)
        result = adjudicate(self.design(), rows)
        self.assertEqual(result["status"], "CAP_REACHED")
        self.assertEqual(rows, before)
        self.assertFalse(result["scientific_admission"])
        self.assertFalse(result["phase_b_authorized"])
        self.assertFalse(result["retries_permitted"])
        self.assertEqual(result["alpha_procedure"], "fixed Bonferroni, NOT Holm")


class JointCalibrationTest(unittest.TestCase):
    def test_parity_rejects_forged_early_stop_on_all_ties(self):
        design = JointDesign(tuple(range(64)), (16, 32, 48, 64))
        lower = np.zeros((3, 256, 2))
        refused = np.zeros((3, 256), dtype=bool)
        forged = (np.zeros((3, 2), dtype=int), np.full(3, 16, dtype=int),
            np.zeros(3, dtype=bool))
        with self.assertRaisesRegex(AssertionError, "mismatch"):
            check_scalar_parity(lower, lower, refused, design, forged)

    def test_batch_scalar_parity_for_every_scenario_cap_and_method(self):
        for scenario in SCENARIOS:
            lower, upper, refused = generate(np.random.default_rng(7), scenario, 3)
            for cap in CAPS:
                for method, stake in METHODS:
                    design = JointDesign(tuple(range(cap)), tuple(range(16, cap + 1, 16)),
                        method, (stake, stake))
                    check_scalar_parity(lower, upper, refused, design,
                        vector_adjudicate(lower, upper, refused, design))

    def test_batch_scalar_parity_on_success_and_refusal_after_one_crossing(self):
        for method, stake in METHODS:
            design = JointDesign(tuple(range(64)), (16, 32, 48, 64), method, (stake, stake))
            lower = np.ones((3, 256, 2))
            lower[1, :, 1] = 0
            lower[2] = .25
            upper = lower.copy()
            refused = np.zeros((3, 256), dtype=bool)
            refused[1, 16:19] = True
            lower[1, 16:19] = -1
            upper[1, 16:19] = 1
            vector = vector_adjudicate(lower, upper, refused, design)
            check_scalar_parity(lower, upper, refused, design, vector)
            self.assertEqual(vector[1][1], 19)
            self.assertTrue(vector[2][1])

    def test_joint_games_have_one_actual_shared_raw_control(self):
        scenario = next(s for s in SCENARIOS if s[0] == "partial_null_reference")
        lower, upper, refused = generate(np.random.default_rng(9), scenario, 3)
        u = np.random.default_rng(9).random((3, 256, 3, 2))
        p = np.asarray((.45, .60, .45))[None, None, :, None]
        scores = np.where(u < p, 1., np.where(u < p + .1, .5, 0.)).mean(axis=3)
        np.testing.assert_array_equal(lower, scores[:, :, 1:] - scores[:, :, :1])
        np.testing.assert_array_equal(lower, upper)
        self.assertFalse(refused.any())

    def test_missing_raw_affects_both_and_keeps_upper_endpoint(self):
        scenario = next(s for s in SCENARIOS if s[0] == "null_shared_raw_missing05")
        lower, upper, refused = generate(np.random.default_rng(9), scenario, 3)
        self.assertTrue(refused.any())
        self.assertTrue(np.all(upper[refused] - lower[refused] == 1))
        u = np.random.default_rng(9).random((3, 256, 3, 2))
        scores = np.where(u < .45, 1., np.where(u < .55, .5, 0.)).mean(axis=3)
        np.testing.assert_array_equal(upper[refused], scores[:, :, 1:][refused])
        np.testing.assert_array_equal(lower[refused], scores[:, :, 1:][refused] - 1)

    def test_invalid_dependence_control_is_not_iid(self):
        lower, upper, refused = generate(np.random.default_rng(9), SCENARIOS[-1], 3)
        self.assertTrue(np.all(lower == lower[:, :1]))
        self.assertEqual(SCENARIOS[-1][1], "dependent")

    def test_rate_retains_counts_and_monte_carlo_uncertainty(self):
        result = rate(5, 100)
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["replicates"], 100)
        self.assertLess(result["monte_carlo_95pct_wilson"][0], .05)
        self.assertGreater(result["monte_carlo_95pct_wilson"][1], .05)

    def test_create_only_evidence_and_invalid_run_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            write_new(path, {"status": "original"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, {"status": "replacement"})
            self.assertEqual(path.read_bytes(), original)
            destination = Path(tmp) / "new"
            for kwargs in (dict(replicates=999), dict(assumed_seconds_per_cluster=float("nan"))):
                with self.assertRaises(ValueError):
                    run(destination, **kwargs)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
