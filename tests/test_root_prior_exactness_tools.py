"""Pure-helper tests for the root-prior exactness tools (no Showdown, model or native module).

scripts/regen_root_corpus.py and scripts/live_root_prior_check.py replay seeded source games;
those paths need a built Showdown and a checkpoint and are exercised by hand (see
docs/search_over_raw_root_prior_exactness_20261016.md). The helpers that turn their outputs
into verdicts are deterministic and tested here.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live = _load("live_root_prior_check")
regen = _load("regen_root_corpus")


class CompareTest(unittest.TestCase):
    def test_identical_priors_are_exact(self) -> None:
        py = {0: 0.9, 1: 0.1}
        out = live.compare(py, {0: 0.9, 1: 0.1})
        self.assertTrue(out["agree"])
        self.assertEqual(out["python_argmax"], 0)
        self.assertEqual(out["native_argmax"], 0)
        self.assertEqual(out["tv"], 0.0)
        self.assertEqual(out["kl"], 0.0)

    def test_native_prior_is_normalized_before_comparison(self) -> None:
        out = live.compare({0: 0.9, 1: 0.1}, {0: 9.0, 1: 1.0})
        self.assertEqual(out["tv"], 0.0)
        self.assertEqual(out["native_prior"], {"0": 0.9, "1": 0.1})

    def test_disagreement_is_reported_with_distances(self) -> None:
        out = live.compare({0: 0.98, 1: 0.02}, {0: 0.03, 1: 0.97})
        self.assertFalse(out["agree"])
        self.assertEqual((out["python_argmax"], out["native_argmax"]), (0, 1))
        self.assertAlmostEqual(out["tv"], 0.95, places=4)
        self.assertGreater(out["kl"], 3.0)
        self.assertTrue(math.isfinite(out["kl"]))

    def test_zero_native_mass_on_python_support_stays_finite(self) -> None:
        out = live.compare({0: 0.5, 1: 0.5}, {0: 1.0, 1: 0.0})
        self.assertTrue(math.isfinite(out["kl"]))
        self.assertEqual(out["tv"], 0.5)


class FindKeyTest(unittest.TestCase):
    def test_nested_lookup_through_dicts_and_lists(self) -> None:
        payload = {"engine_mcts": {"override": {"root_allocation": {"arms": [{"action_index": 2, "model_prior": 0.7}]}}}}
        self.assertEqual(live.find_key(payload, "root_allocation")["arms"][0]["action_index"], 2)
        self.assertEqual(live.find_key({"a": [{"b": {"model_argmax": 4}}]}, "model_argmax"), 4)
        self.assertIsNone(live.find_key({"a": 1}, "missing"))


class GateVerdictTest(unittest.TestCase):
    def _row(self, **variants):
        return {"variants": variants}

    def test_passes_only_when_every_variant_is_within_gate_and_error_free(self) -> None:
        rows = [self._row(a={"agree": True, "tv": 0.0}), self._row(a={"agree": True, "tv": 0.0005})]
        verdict = live.gate_verdict(rows, ["a"], 0.001)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["variants"]["a"]["over_gate"], 0)
        verdict = live.gate_verdict(rows + [self._row(a={"agree": True, "tv": 0.02})], ["a"], 0.001)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["variants"]["a"]["over_gate"], 1)
        self.assertAlmostEqual(verdict["variants"]["a"]["max_tv"], 0.02)

    def test_errors_fail_the_gate_and_are_counted(self) -> None:
        rows = [self._row(a={"agree": True, "tv": 0.0}, b={"error": "boom"})]
        verdict = live.gate_verdict(rows, ["a", "b"], 0.001)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["variants"]["b"], {"measured": 0, "agree": 0, "over_gate": 0, "errors": 1, "max_tv": 0.0})

    def test_argmax_flip_within_tv_bound_and_unmeasured_variant_both_fail(self) -> None:
        flip = [self._row(a={"agree": False, "tv": 0.05})]
        self.assertFalse(live.gate_verdict(flip, ["a"], 0.1)["passed"])
        self.assertFalse(live.gate_verdict([self._row(a={"agree": True, "tv": 0.0})], ["a", "never_written"], 0.1)["passed"])


class ArchivedRootsTest(unittest.TestCase):
    def test_reads_roots_from_source_json_and_tolerates_missing_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            seed_dir = run / "sources" / "123"
            seed_dir.mkdir(parents=True)
            (seed_dir / "source.json").write_text(json.dumps({"roots": [{"root_id": "exploration:123:0", "source_request_index": 4}]}))
            self.assertEqual(regen.archived_roots(run, 123)[0]["root_id"], "exploration:123:0")
            self.assertEqual(regen.archived_roots(run, 999), [])


scan = _load("search_deviation_scan")


class DeviationScanSummaryTest(unittest.TestCase):
    def test_summary_counts_deviations_overrides_and_prior_mismatches(self) -> None:
        rows = [
            {"raw_action": 2, "budgets": {"80": {"action": 2, "model_argmax": 2, "search_argmax": 2, "root_q_gap": 0.02, "seconds": 0.9}}},
            {"raw_action": 2, "budgets": {"80": {"action": 5, "model_argmax": 2, "search_argmax": 5, "root_q_gap": 0.04, "seconds": 1.1}}},  # tree override
            {"raw_action": 2, "budgets": {"80": {"action": 6, "model_argmax": 2, "search_argmax": 6, "root_q_gap": 0.10, "seconds": 1.2}}},  # tree override
            {"raw_action": 1, "budgets": {"80": {"action": 3, "model_argmax": 3, "search_argmax": 3, "root_q_gap": 0.30, "seconds": 1.0}}},  # prior mismatch, no override
            {"raw_action": 1, "budgets": {"80": {"error": "boom"}}},
        ]
        out = scan.summarize(rows, [80])["80"]
        self.assertEqual(out["decisions"], 4)
        self.assertEqual(out["deviations"], 3)
        self.assertAlmostEqual(out["deviation_rate"], 3 / 4)
        self.assertEqual(out["tree_overrides"], 2)       # search_argmax != model_argmax
        self.assertEqual(out["prior_argmax_mismatch"], 1)  # model_argmax != raw_action
        self.assertEqual(out["errors"], 1)
        self.assertAlmostEqual(out["deviation_root_q_gap_median"], 0.10)  # median of 0.04, 0.10, 0.30

    def test_empty_budget_is_reported_without_division(self) -> None:
        out = scan.summarize([], [256])["256"]
        self.assertEqual(out["decisions"], 0)
        self.assertIsNone(out["deviation_rate"])
        self.assertIsNone(out["seconds_median"])


if __name__ == "__main__":
    unittest.main()
