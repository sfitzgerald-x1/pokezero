"""Synthetic fixtures only, not a calibration or an experiment admission."""
import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sparse_sensitivity", ROOT / "scripts/calibrate_search_over_raw_sparse.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
package = types.ModuleType("sparse_kernel_fixture")
package.__path__ = [str(ROOT / "src/pokezero/mcts_eval")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + ".search_over_raw_deviations",
    ROOT / "src/pokezero/mcts_eval/search_over_raw_deviations.py")
k = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = k
spec.loader.exec_module(k)


class SparseSensitivityTest(unittest.TestCase):
    def test_effect_scale_and_ties_are_exact_on_balanced_grid(self):
        u = (np.arange(10000) + .5) / 10000
        raw = np.repeat([0., 1.], len(u))
        for delta in (0., .05, .10, .20, -.4):
            candidate = m.flip_scores(raw, delta, np.tile(u, 2))
            self.assertAlmostEqual(float((candidate-raw).mean()), delta)
            self.assertAlmostEqual(float((candidate == raw).mean()), .6)
        with self.assertRaises(ValueError):
            m.flip_scores(raw, .41, np.tile(u, 2))

    def test_actual_both_target_first_hit_and_kernel_parity(self):
        actions = np.zeros((2000, 2), dtype=int)
        actions[:100, 0] = 1; actions[50:150, 1] = 2
        self.assertEqual(m.first_stop(actions), (150, "DEVIATION_TARGET"))
        rows = [k.ScannedRoot(1, i+1, 0, int(a), int(b)) for i, (a, b) in enumerate(actions[:150])]
        self.assertIsNone(k.stop_reason(rows[:-1], 1))
        self.assertEqual(k.stop_reason(rows, 1), "DEVIATION_TARGET")

    def test_root_cap_and_unknown_stop_never_count_unknown_as_deviation(self):
        actions = np.zeros((2000, 2), dtype=int)
        self.assertEqual(m.first_stop(actions), (2000, "ROOT_CAP"))
        actions[10:13, 0] = -1
        self.assertEqual(m.first_stop(actions), (13, "THREE_CONSECUTIVE_UNAVAILABLE"))
        actions[11, 0] = 0
        self.assertEqual(m.first_stop(actions), (2000, "ROOT_CAP"))
        actions[1997:2000, 1] = -1
        self.assertEqual(m.first_stop(actions), (2000, "THREE_CONSECUTIVE_UNAVAILABLE"))

    def test_interval_parity_shared_raw_alias_missing_unavailable(self):
        actions = np.array([[1, 2], [1, 1], [0, 0], [-1, 0]])
        rng = np.random.default_rng(21)
        scores = rng.integers(2, size=(4, 3, 8)).astype(float)
        valid = np.ones(scores.shape, dtype=bool)
        valid[0, 0, 0] = False; valid[1, 1, 2] = False
        actual = m.intervals(actions, scores, valid)
        for i, (a, b) in enumerate(actions):
            cells = {(action, r): dict(status="COMPLETE" if valid[i, action, r] else "REFUSED",
                validated=bool(valid[i, action, r]),
                score=float(scores[i, action, r]) if valid[i, action, r] else None)
                for action in range(3) for r in range(8)}
            root = k.ScannedRoot(1, i+1, 0, None if a < 0 else int(a), int(b))
            expected = k.root_intervals(root, cells)
            np.testing.assert_array_equal(actual[i], [expected[leaf] for leaf in k.LEAVES])

    def test_bootstrap_endpoint_parity_with_actual_kernel_same_draws(self):
        rows = [k.ScannedRoot(s, j+1, 0, 1, 2) for s in range(32) for j in range(1+s % 3)]
        seeds = np.array([r.seed for r in rows])
        values = np.array([((i % 5)-2)/8 for i in range(len(rows))])
        bounds = np.stack([np.stack([values, values], axis=1),
                           np.stack([-values, -values], axis=1)], axis=1)
        intervals = [{leaf: tuple(map(float, bounds[i, j])) for j, leaf in enumerate(k.LEAVES)}
                     for i in range(len(rows))]
        report = k.summarize(rows, intervals, bootstrap_seed=21, bootstrap_draws=100)
        rng = random.Random(21)
        indices = np.array([rng.choices(range(32), k=32) for _ in range(100)])
        result = m.lower_endpoints(bounds, seeds, indices)
        for j, leaf in enumerate(k.LEAVES):
            self.assertAlmostEqual(result[j], report["leaves"][leaf]["one_sided_lower_endpoint"])

    def test_shared_candidate_has_identical_realized_banks(self):
        a, _, bounds, _ = m.panel(np.random.default_rng(42), m.Family("alias", shared_candidate=True), 1., 6)
        np.testing.assert_array_equal(a[:, 0], a[:, 1])
        np.testing.assert_array_equal(bounds[:, 0], bounds[:, 1])
        with self.assertRaises(ValueError):
            m.Family("invalid_alias", effects=(0., .1), shared_candidate=True)

    def test_declared_skew_null_mean_and_shared_membership(self):
        self.assertAlmostEqual(.8*.1 + .2*(-.4), 0.)
        a, _, _, _ = m.panel(np.random.default_rng(42), m.Family("shared", shared_membership=True), .1, 48)
        np.testing.assert_array_equal(a[:, 0] > 0, a[:, 1] > 0)

    def test_smoke_output_preserves_limits_and_all_stop_counts(self):
        with patch.object(m, "FAMILIES", (m.Family("fixture_null"),)):
            result = m.run(replicates=1, draws=20, seed=42)
        for flag in ("full_phase_a_calibration", "screening_authorized", "holdout_authorized",
                     "phase_b_authorized", "wall_runtime_modeled", "configuration_selection_modeled"):
            self.assertIs(result[flag], False)
        self.assertEqual(result["scientific_calls"], 0)
        self.assertEqual(len(result["results"]), 6)
        for row in result["results"]:
            self.assertEqual(sum(v["hits"] for v in row["stop_reasons"].values()), 1)
            self.assertEqual(set(row["count_and_source_eligibility_by_leaf"]), {"model", "hp_fraction"})

    def test_targets_reached_exactly_at_cap_not_lost_from_target_count(self):
        actions = np.zeros((2000, 2), dtype=int)
        actions[:100, 0] = 1; actions[-100:, 1] = 2
        n, reason = m.first_stop(actions)
        self.assertEqual((n, reason), (2000, "ROOT_CAP"))
        fixture = actions, np.arange(2000) // 6, np.zeros((2000, 2, 2)), reason
        with patch.object(m, "FAMILIES", (m.Family("fixture_null"),)), patch.object(m, "panel", return_value=fixture):
            result = m.run(replicates=1, draws=20, seed=42)
        for row in result["results"]:
            self.assertEqual(row["both_targets_reached"]["hits"], 1)
            self.assertEqual(row["stop_reasons"]["ROOT_CAP"]["hits"], 1)
            self.assertEqual(row["stop_reasons"]["DEVIATION_TARGET"]["hits"], 0)


if __name__ == "__main__":
    unittest.main()
