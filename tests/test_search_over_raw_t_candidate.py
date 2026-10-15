from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibrate_search_over_raw_joint import SCENARIOS, generate
from calibrate_search_over_raw_t_candidate import parity, run, vector_candidate
from pokezero.mcts_eval.search_over_raw_statistics import ARMS, JointDesign, primary_cluster_intervals
from pokezero.mcts_eval.search_over_raw_t_candidate import (
    TCandidateDesign, adjudicate_t_candidate, adjudicate_t_primary_cells, candidate_arm,
    student_critical, student_survival,
)


def rows(values, *, refused=(), bad_provenance=()):
    return [dict(seed=i, refused=i in refused, provenance_ok=i not in bad_provenance,
        contrasts={arm: pair for arm in ARMS}) for i, pair in enumerate(values)]


class StudentNumericsTests(unittest.TestCase):
    def test_cauchy_tail(self):
        for t in (-12., -2., -.1, 0., .1, 2., 12.):
            self.assertAlmostEqual(student_survival(t, 1), .5 - math.atan(t) / math.pi, places=12)

    def test_nist_rounded_critical_table(self):
        # Independent values from NIST's 0.975 column, not our beta inversion.
        for df, expected in ((1, 12.706), (2, 4.303), (5, 2.571), (10, 2.228),
                (30, 2.042), (64, 1.998), (100, 1.984)):
            self.assertLess(abs(student_critical(df) - expected), .00051)

    def test_independent_density_quadrature(self):
        # Simpson integration of normalized density, no incomplete-beta call.
        for df in (2, 5, 31, 127, 255):
            for t in (.15, 1., 2., 4.):
                steps = 4096
                h = t / steps
                norm = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
                f = lambda x: norm * (1 + x * x / df) ** (-(df + 1) / 2)
                integral = h / 3 * (f(0) + f(t)
                    + 4 * math.fsum(f(h * i) for i in range(1, steps, 2))
                    + 2 * math.fsum(f(h * i) for i in range(2, steps, 2)))
                self.assertAlmostEqual(student_survival(t, df), .5 - integral, places=11)

    def test_symmetry_monotonicity_and_inverse(self):
        for df in (1, 2, 10, 127, 255):
            for alpha in (.01, .025, .1):
                t = student_critical(df, alpha)
                self.assertAlmostEqual(student_survival(t, df), alpha, places=12)
                self.assertAlmostEqual(student_survival(-t, df), 1 - alpha, places=12)
                self.assertGreater(student_survival(t - .01, df), alpha)

    def test_invalid_numerics(self):
        student_critical(127, .025)
        for df in (0, 256, True, 127.):
            with self.assertRaises(ValueError):
                student_survival(1., df)
            with self.assertRaises(ValueError):
                student_critical(df, .025)
        for t in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                student_survival(t, 127)
        for alpha in (True, 0., .5, float("nan")):
            with self.assertRaises(ValueError):
                student_critical(127, alpha)


class TCandidateTests(unittest.TestCase):
    def setUp(self):
        self.plain = TCandidateDesign(tuple(range(128)), None)
        self.gated = TCandidateDesign(tuple(range(128)), .5)
        self.positive = [(.25, .25), (.5, .5)] * 64

    def test_design_roster_and_fixed_gate(self):
        for seeds in (tuple(range(64)), (1,) * 128, [*range(128)], (True, *range(1, 128)),
                (-1, *range(1, 128))):
            with self.assertRaises(ValueError):
                TCandidateDesign(seeds)
        for gate in (.4, 0, True, float("nan")):
            with self.assertRaises(ValueError):
                TCandidateDesign(tuple(range(128)), gate)
        with self.assertRaises(ValueError):
            JointDesign(tuple(range(128)), (128,), "skew_gated_t")

    def test_no_interim_no_scientific_admission(self):
        prefix = adjudicate_t_candidate(self.plain, rows(self.positive[:127]))
        self.assertEqual(prefix["status"], "NONTERMINAL_PREFIX")
        self.assertFalse(any(prefix["nominal_primary_crossings"].values()))
        self.assertTrue(all(v is None for v in prefix["candidate_evaluations"].values()))
        full = adjudicate_t_candidate(self.gated, rows(self.positive))
        self.assertTrue(all(full["nominal_primary_crossings"].values()))
        self.assertEqual(full["status"], "FIXED_CAP_REACHED")
        for key in ("scientific_admission", "phase_b_authorized", "method_adopted", "finite_sample_validity_established"):
            self.assertFalse(full[key])

    def test_skew_formula_and_gate(self):
        values = [(-1., -1.)] * 16 + [(.25, .25)] * 112
        p, q = candidate_arm(values, None), candidate_arm(values, .5)
        self.assertTrue(p["nominal_crossing"])
        self.assertFalse(q["nominal_crossing"])
        self.assertTrue(q["skew_gate_abstains"])
        self.assertAlmostEqual(q["adjusted_sample_skewness"],
            math.sqrt(128 * 127) / 126 * (2 * .125 - 1) / math.sqrt(.125 * .875), places=12)

    def test_zero_variance_abstention(self):
        for value in (0., .5, -.5):
            result = candidate_arm([(value, value)] * 128, None)
            self.assertTrue(result["zero_variance_abstains"])
            self.assertFalse(result["nominal_crossing"])

    def test_uncertain_constant_completion_abstention(self):
        values = [(.5, .5)] * 126 + [(0., 1.)] * 2
        result = candidate_arm(values, None)
        self.assertGreater(result["completion_statistic_lower"], student_critical(127))
        self.assertTrue(result["constant_completion_possible"])
        self.assertFalse(result["nominal_crossing"])

    def test_uncertain_skew_abstains_not_complete_case(self):
        values = list(self.positive)
        values[0] = (.25, 1.)
        result = candidate_arm(values, .5)
        self.assertIsNone(result["adjusted_sample_skewness"])
        self.assertTrue(result["uncertain"])
        self.assertTrue(result["skew_gate_abstains"])
        self.assertFalse(result["nominal_crossing"])

    def test_completion_envelope_against_exhaustive_completions(self):
        values = [(.25, .25), (.5, .5)] * 63 + [(0., 1.), (-.25, .75)]
        bounded = candidate_arm(values, None)
        self.assertTrue(bounded["nominal_crossing"])
        for x, y in itertools.product(np.linspace(0, 1, 21), np.linspace(-.25, .75, 21)):
            full = candidate_arm(values[:126] + [(float(x), float(x)), (float(y), float(y))], None)
            self.assertGreaterEqual(full["completion_statistic_lower"] + 1e-12,
                bounded["completion_statistic_lower"])
            self.assertTrue(full["nominal_crossing"])

    def test_full_denominator_missing_suffix(self):
        result = adjudicate_t_candidate(self.plain, rows([(.5, .5)] * 2))
        self.assertEqual(result["full_roster_identification"]["incumbent"], [-125 / 128, 127 / 128])

    def test_third_refusal_and_last_seed_failure_override(self):
        for failures in ((0, 1, 127), (0, 1, 2)):
            data = rows(self.positive, refused=failures)
            result = adjudicate_t_candidate(self.plain, data[:failures[-1] + 1])
            self.assertEqual(result["status"], "REFUSAL_CAP_EXCEEDED")
            self.assertFalse(any(result["nominal_primary_crossings"].values()))
            self.assertTrue(all(v is None for v in result["candidate_evaluations"].values()))
        result = adjudicate_t_candidate(self.plain, rows(self.positive, bad_provenance=(127,)))
        self.assertEqual(result["status"], "PROVENANCE_FAILURE")
        self.assertFalse(any(result["nominal_primary_crossings"].values()))

    def test_uncertainty_infers_refusal(self):
        data = rows([(-1., 1.)] * 3)
        result = adjudicate_t_candidate(self.plain, data)
        self.assertEqual(result["refused_clusters"], 3)
        self.assertEqual(result["status"], "REFUSAL_CAP_EXCEEDED")
        with self.assertRaises(ValueError):
            adjudicate_t_candidate(self.plain, data + rows([(0., 0.)]))

    def test_closed_rows_seed_order_and_cell_override(self):
        for mutate in (lambda r: r.update(seed=True), lambda r: r.update(seed=1),
                lambda r: r.update(extra=1), lambda r: r.update(refused=1),
                lambda r: r.update(contrasts={"incumbent": (0., 0.)})):
            data = rows([(0., 0.)])
            mutate(data[0])
            with self.assertRaises(ValueError):
                adjudicate_t_candidate(self.plain, data)
        with self.assertRaises(ValueError):
            adjudicate_t_primary_cells(self.plain, [dict(seed=0, primary_cells={}, refused=False,
                provenance_ok=True, contrasts={})])

    def test_shared_raw_and_unvalidated_cells_uncertain(self):
        cells = {(arm, seat): dict(status="COMPLETE", validated=True, score=.5)
            for arm in ("raw", *ARMS) for seat in ("p1", "p2")}
        cells[("raw", "p1")] = dict(status="COMPLETE", validated=False, score=1.)
        intervals = primary_cluster_intervals(cells)
        self.assertEqual(intervals, {arm: (-.25, .25) for arm in ARMS})
        result = adjudicate_t_primary_cells(self.plain, [dict(seed=0, primary_cells=cells,
            refused=False, provenance_ok=True)])
        self.assertEqual(result["refused_clusters"], 1)


class SyntheticTCandidateTests(unittest.TestCase):
    def test_all_families_parity_and_guard_subset_identical_cost(self):
        for i, scenario in enumerate(SCENARIOS):
            faithful = scenario[1] == "games" and scenario[3] in {"none", "random"}
            data = generate(np.random.default_rng(104 + i), scenario, 12, original_cells=faithful)
            vectors = []
            for gate in (None, .5):
                design = TCandidateDesign(tuple(range(128)), gate)
                vector = vector_candidate(*data[:3], design)
                parity(*data[:3], design, vector, *(data[3:] if faithful else ()))
                vectors.append(vector)
            self.assertTrue(np.all(~vectors[1]["crossed"] | vectors[0]["crossed"]))
            np.testing.assert_array_equal(vectors[0]["stop"], vectors[1]["stop"])
            np.testing.assert_array_equal(vectors[0]["failed"], vectors[1]["failed"])

    def test_no_failed_suffix_gate_counts(self):
        lo = np.zeros((3, 256, 2))
        hi = lo.copy()
        hi[:, :3] = 1
        vector = vector_candidate(lo, hi, np.zeros((3, 256), dtype=bool), TCandidateDesign(tuple(range(128))))
        self.assertTrue(vector["failed"].all())
        self.assertFalse(vector["skew_abstains"].any())
        self.assertFalse(vector["crossed"].any())
        self.assertTrue((vector["stop"] == 3).all())

    def test_driver_create_only_guards_and_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for reps, seed in ((999, 1), (True, 1), (1000, -1), (1000, True)):
                with self.assertRaises(ValueError):
                    run(root / "bad", replicates=reps, seed=seed)
                self.assertFalse((root / "bad").exists())
            result = run(root / "good", replicates=1000, seed=101)
            self.assertEqual(len(result["results"]), 38)
            self.assertEqual(result["scientific_calls"], 0)
            self.assertFalse(result["calibration_certified"])
            self.assertFalse(result["method_adopted"])
            self.assertFalse(result["phase_b_authorized"])
            self.assertTrue(result["source_hashes_unchanged"])
            contract = json.loads((root / "good/contract.json").read_text())
            self.assertEqual(contract["terminal_look"], 128)
            self.assertEqual(contract["refusal_cap"], 2)
            for record in result["results"]:
                self.assertIsNone(record["projected_sequential_hours"])
                self.assertAlmostEqual(record["expected_full_cluster_game_budget"], 12 * record["expected_clusters"])
            with self.assertRaises(FileExistsError):
                run(root / "good", replicates=1000, seed=101)


if __name__ == "__main__":
    unittest.main()
