"""Pure-helper tests for the root-prior exactness tools (no Showdown, model or native module).

scripts/regen_root_corpus.py and scripts/live_root_prior_check.py replay seeded source games;
those paths need a built Showdown and a checkpoint and are exercised by hand (see
docs/search_over_raw_root_prior_exactness_20261016.md). The helpers that turn their outputs
into verdicts are deterministic and tested here.
"""
from __future__ import annotations

import importlib.util
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_gate_uses_unrounded_distance_and_keeps_full_precision_vectors(self) -> None:
        out = live.compare({0: .75, 1: .25}, {0: .750000001, 1: .249999999})
        self.assertGreater(out["tv"], 0.)
        self.assertEqual(round(out["tv"], 4), 0.)
        self.assertEqual(out["native_prior_full_precision"]["0"], .750000001)
        self.assertFalse(live.gate_verdict([{"variants": {"sampled4": out}}], ["sampled4"], 0.)["passed"])


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


class BindingDriftTest(unittest.TestCase):
    def test_drift_before_reload_or_final_publication_keeps_report_incomplete(self):
        for seeds in ([12,13],[12]):
            with self.subTest(seeds=seeds),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                root=Path(folder);artifact=root/"authored-input"
                artifact.write_bytes(b"authored")
                sha=hashlib.sha256(b"authored").hexdigest()
                original={name:sha for name in ("checkpoint_sha256","model_sha256","tables_sha256")}
                changed={**original,"model_sha256":"d"*64}
                bind=stack.enter_context(patch.object(live,"input_binding",side_effect=[
                    deepcopy(original),deepcopy(original),deepcopy(original),changed]))
                config=SimpleNamespace(resolved_showdown_root=lambda:root)
                stack.enter_context(patch.object(live,"env_config_with_policy_spec_masks",return_value=config))
                env=Mock();env.snapshot.return_value.bridge_snapshot={}
                stack.enter_context(patch.object(live,"LocalShowdownEnv",return_value=env))
                stack.enter_context(patch.object(live,"load_showdown_dex_cached"))
                stack.enter_context(patch.object(live,"load_gen3_randbat_source_cached"))
                stack.enter_context(patch.object(live,"_true_teams_from_bridge_snapshot",return_value={
                    "p1":{"packed":"own"},"p2":{"packed":"opponent"}}))
                engine=stack.enter_context(patch.object(live,"EngineMctsPolicy"))
                stack.enter_context(patch.object(live,"load_transformer_policy"))
                stack.enter_context(patch.object(live,"continue_rollout_from_current_state",
                    return_value=SimpleNamespace(terminal=SimpleNamespace(turn_count=1))))
                out=root/"out.json"
                args=["--checkpoint",str(artifact),"--model-ts",str(artifact),"--tables",str(artifact),
                    "--showdown-root",str(root),"--seeds",*[str(s) for s in seeds],
                    "--variants","sampled4","--out",str(out)]
                with self.assertRaisesRegex(ValueError,"inputs changed"):live.main(args)
                self.assertEqual(bind.call_count,4)
                self.assertEqual(engine.call_count,1)  # Never reload after drift.
                self.assertIs(json.loads(out.read_text())["complete"],False)
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
