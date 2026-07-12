"""Focused tests for the Stage A diversity-fingerprint metric math.

The analysis lives in scripts/diversity_fingerprint_analyze.py (a standalone tool, not
a package module); we importlib-load it and exercise the pure functions that decide the
verdicts, so a regression in the distance math fails loudly.
"""
import importlib.util
import math
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diversity_fingerprint_analyze.py"


def _load():
    spec = importlib.util.spec_from_file_location("divfp_analyze", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DiversityFingerprintMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_js_identical_is_zero(self):
        p = {0: 0.5, 1: 0.3, 2: 0.2}
        self.assertAlmostEqual(self.m.js_divergence(p, p), 0.0, places=12)

    def test_js_disjoint_is_one_bit(self):
        # base-2 JS of two non-overlapping distributions is exactly 1.0
        self.assertAlmostEqual(self.m.js_divergence({0: 1.0}, {1: 1.0}), 1.0, places=9)

    def test_js_symmetric(self):
        a = {0: 0.7, 1: 0.2, 2: 0.1}
        b = {0: 0.2, 1: 0.2, 2: 0.6}
        self.assertAlmostEqual(self.m.js_divergence(a, b), self.m.js_divergence(b, a), places=12)

    def test_norm_entropy_extremes(self):
        self.assertAlmostEqual(self.m.norm_entropy({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}), 1.0, places=9)
        self.assertAlmostEqual(self.m.norm_entropy({0: 1.0, 1: 0.0}), 0.0, places=9)

    def _fp(self, top1=0, value_scale=1.0):
        rows = {}
        for i in range(50):
            probs = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
            rows[f"d{i}"] = {"top1": top1, "probs": probs, "value": value_scale * math.sin(i)}
        return rows

    def test_self_pair_sanity(self):
        fp = self._fp()
        ids = list(fp)
        contested = {d: True for d in ids}
        r = self.m.pair_metrics(fp, fp, ids, contested)
        self.assertEqual(r["top1_disagreement"], 0.0)
        self.assertAlmostEqual(r["js_divergence"], 0.0, places=12)
        self.assertAlmostEqual(r["value_1_minus_pearson"], 0.0, places=9)
        self.assertAlmostEqual(r["value_p95_abs"], 0.0, places=12)

    def test_disagreement_detected(self):
        a = self._fp(top1=0)
        b = self._fp(top1=1)  # every top-1 differs
        ids = list(a)
        contested = {d: True for d in ids}
        r = self.m.pair_metrics(a, b, ids, contested)
        self.assertGreater(r["top1_disagreement"], 0.9)


_MATCHUP = Path(__file__).resolve().parents[1] / "scripts" / "diversity_matchup_analyze.py"


def _load_matchup():
    spec = importlib.util.spec_from_file_location("divfp_matchup", _MATCHUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DiversityMatchupMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_matchup()

    def test_bradley_terry_recovers_ladder(self):
        import numpy as np
        labels = ["a", "b", "c"]
        # a beats b beats c transitively
        W = np.array([[np.nan, 0.75, 0.9], [0.25, np.nan, 0.75], [0.1, 0.25, np.nan]])
        N = np.where(np.isnan(W), 0.0, 300.0)
        s, pred = self.m.bradley_terry(labels, W, N)
        self.assertGreater(s[0], s[1])
        self.assertGreater(s[1], s[2])

    def test_intransitivity_zero_when_transitive(self):
        import numpy as np
        # perfectly transitive win matrix has no consistent cycle
        W = np.array([[0.5, 0.8, 0.9], [0.2, 0.5, 0.8], [0.1, 0.2, 0.5]])
        self.assertAlmostEqual(self.m.intransitivity(W), 0.0, places=9)

    def test_intransitivity_positive_when_cyclic(self):
        import numpy as np
        # a>b>c>a rock-paper-scissors
        W = np.array([[0.5, 0.7, 0.3], [0.3, 0.5, 0.7], [0.7, 0.3, 0.5]])
        self.assertGreater(self.m.intransitivity(W), 0.0)


if __name__ == "__main__":
    unittest.main()
