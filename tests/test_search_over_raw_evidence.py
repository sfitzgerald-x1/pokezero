"""Synthetic supplied receipts only; never source allocation or game execution."""
from copy import deepcopy
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import types
import unittest

PACKAGE = "reconciled_evidence_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/pokezero/mcts_eval")]
sys.modules[PACKAGE] = package
m = importlib.import_module(PACKAGE + ".search_over_raw_evidence")
NAMESPACE = "86fbb43c-dd13-47f5-8c91-82c1758765eb"
SHA = "a" * 64


class ReconciledEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.slot = dict(root_id="fixture:0", source_seed=123, root_slot=0)
        self.raw = m.SearchConfiguration("raw", seconds=1.)
        self.inc = m.SearchConfiguration("incumbent", seconds=1.)
        self.ref = m.SearchConfiguration("reference", seconds=1., workers=20)
        self.configs = (self.raw, self.inc, self.ref)
        binding = dict(root_id=self.slot["root_id"], source_seed=123,
            public_record_sha256=SHA, information_key="fixture-public-info", legal_action_mask=[True, True])
        self.bundle = dict(root_binding=binding, continuation_contract=dict(root_binding=deepcopy(binding),
            root_id=self.slot["root_id"], namespace=NAMESPACE, subject="p1", max_boundaries=250,
            replicates=8, selection_seed=123, policy="raw_argmax_after_initial_sampled_opponent_reply"),
            selections={}, outcomes=[])
        self.runtime_hashes = {}
        for cfg in self.configs:
            key = m.selector_key(cfg)
            evidence = (dict(selector="deterministic_masked_argmax", model_forwards=1,
                legal_actions=[0, 1], priors=[.75, .25]) if cfg.arm == "raw" else
                dict(total_iterations=64) if cfg.arm == "incumbent" else dict(result=dict(trajectories=80)))
            self.bundle["selections"][key] = dict(root_id=self.slot["root_id"] + ":" + key,
                statistics_root_id=self.slot["root_id"], selection_seed=123,
                configuration_sha256=cfg.identity, runtime_sha256=SHA, independent_root_statistics=True,
                action=0 if cfg.arm == "raw" else 1, status="SELECTED", elapsed_seconds=.5,
                nominal_search_seconds=1., evidence=evidence)
            self.runtime_hashes[key] = SHA
        self.bundle["outcomes"] = [dict(root_id=self.slot["root_id"], action=action, replicate=rep,
            status="COMPLETE", signed_outcome=-1 if action == 0 else 1, boundaries=2)
            for action in (0, 1) for rep in range(8)]

    def reconcile(self, *, outcome_validator=None, selector_validator=None, allow_unqualified_at_ceiling=False):
        return m.reconcile_root_evidence(self.bundle, namespace=NAMESPACE, slot=self.slot,
            configurations=self.configs, public_record_sha256=SHA, runtime_hashes=self.runtime_hashes,
            validate_selector=selector_validator or (lambda cfg, row, binding: m.digest(row)),
            allow_unqualified_at_ceiling=allow_unqualified_at_ceiling,
            validate_outcome=outcome_validator or (lambda row, contract:
                m.digest(row) if row["status"] == "COMPLETE" else None))

    def test_actual_validated_receipts_derive_shared_control_contrasts(self):
        result = self.reconcile()
        self.assertEqual(result["root_intervals"], {self.inc.identity: [1., 1.], self.ref.identity: [1., 1.]})
        self.assertEqual(result["unique_action_cells"], 16)
        self.assertEqual(result["validated_action_cells"], 16)
        self.assertEqual(result["accepted_units"], {"raw": 0, self.inc.identity: 64, self.ref.identity: 80})
        self.assertFalse(result["full_world_matching_established"])
        self.assertFalse(result["runtime_qualified"])
        self.assertFalse(result["collection_authorized"])

    def test_missing_raw_cell_affects_both_arms(self):
        self.bundle["outcomes"].pop(0)
        result = self.reconcile()
        self.assertEqual(result["root_intervals"][self.inc.identity], [.875, 1.])
        self.assertEqual(result["root_intervals"][self.ref.identity], [.875, 1.])
        self.assertEqual(result["uncertain_action_cells"], 1)

    def test_explicit_v2_zero_work_has_no_action_cells_and_never_scores(self):
        selected=self.bundle["selections"][self.inc.identity]
        selected.update(status="UNQUALIFIED_AT_CEILING",action=None,attempt_consumed=True,
            retry_authorized=False,continuation_authorized=False,evidence=dict(total_iterations=0))
        with self.assertRaises(ValueError):self.reconcile()
        result=self.reconcile(allow_unqualified_at_ceiling=True)
        self.assertEqual(result["schema"],"pokezero.search-over-raw.reconciled-root-evidence.v2")
        self.assertEqual(result["root_intervals"][self.inc.identity],[-1.,1.])
        self.assertEqual(result["root_intervals"][self.ref.identity],[1.,1.])
        self.assertNotIn(self.inc.identity,result["actions"])
        self.assertEqual(result["accepted_units"][self.inc.identity],0)
        for key,value in (("action",0),("continuation_authorized",True),("retry_authorized",True)):
            old=selected[key];selected[key]=value
            with self.assertRaises(ValueError):self.reconcile(allow_unqualified_at_ceiling=True)
            selected[key]=old
        with self.assertRaises(ValueError):self.reconcile(allow_unqualified_at_ceiling=True,
            selector_validator=lambda cfg,row,binding:None if cfg.arm=="incumbent" else m.digest(row))

    def test_all_searches_unqualified_only_raw_continuations_remain(self):
        for cfg in (self.inc,self.ref):
            self.bundle["selections"][cfg.identity].update(status="UNQUALIFIED_AT_CEILING",action=None,
                attempt_consumed=True,retry_authorized=False,continuation_authorized=False,
                evidence=dict(total_iterations=0) if cfg.arm=="incumbent" else dict(result=dict(trajectories=0)))
        self.bundle["outcomes"]=[r for r in self.bundle["outcomes"] if r["action"]==0]
        result=self.reconcile(allow_unqualified_at_ceiling=True)
        self.assertEqual(result["actions"],{"raw":0})
        self.assertEqual(result["unique_action_cells"],8)
        self.assertEqual(result["root_intervals"],{self.inc.identity:[-1.,1.],self.ref.identity:[-1.,1.]})
        self.bundle["outcomes"].append(dict(root_id=self.slot["root_id"],action=1,replicate=0,
            status="COMPLETE",signed_outcome=1,boundaries=2))
        with self.assertRaises(ValueError):self.reconcile(allow_unqualified_at_ceiling=True)

    def test_unvalidated_complete_is_not_scored(self):
        result = self.reconcile(outcome_validator=lambda row, contract: None)
        self.assertEqual(result["root_intervals"][self.inc.identity], [-1., 1.])
        self.assertEqual(result["validated_action_cells"], 0)

    def test_capped_refused_and_unstarted_remain_uncertain(self):
        for status in ("CAPPED", "REFUSED", "UNSTARTED"):
            with self.subTest(status=status):
                row = self.bundle["outcomes"][0]
                row.update(status=status, signed_outcome=None)
                result = self.reconcile()
                self.assertEqual(result["root_intervals"][self.ref.identity], [.875, 1.])

    def test_identical_action_unknown_outcomes_exactly_zero_not_dropped(self):
        for row in self.bundle["selections"].values():
            row["action"] = 0
        self.bundle["outcomes"] = []
        result = self.reconcile()
        self.assertEqual(result["root_intervals"], {self.inc.identity: [0., 0.], self.ref.identity: [0., 0.]})
        self.assertEqual(result["unique_action_cells"], 8)
        self.assertEqual(result["uncertain_action_cells"], 8)

    def test_duplicate_and_unused_outcome_cell_rejected(self):
        original = deepcopy(self.bundle)
        for extra in (deepcopy(self.bundle["outcomes"][0]),
                      dict(self.bundle["outcomes"][0], action=2)):
            self.bundle = deepcopy(original)
            self.bundle["outcomes"].append(extra)
            with self.assertRaises(ValueError):
                self.reconcile()

    def test_action_alias_does_not_excuse_invalid_terminal_receipt(self):
        self.bundle["outcomes"][0]["signed_outcome"] = True
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_false_or_wrong_digest_validation_never_scores(self):
        for value in (False, True, "b" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reconcile(outcome_validator=lambda row, contract: value)
        with self.assertRaises(ValueError):
            self.reconcile(selector_validator=lambda cfg, row, binding: None)

    def test_incomplete_cannot_receive_terminal_validation(self):
        self.bundle["outcomes"][0].update(status="REFUSED", signed_outcome=None)
        with self.assertRaises(ValueError):
            self.reconcile(outcome_validator=lambda row, contract: m.digest(row))

    def test_source_seed_root_namespace_and_continuation_law_drift(self):
        original = deepcopy(self.bundle)
        mutations = (("source_seed", 124), ("root_id", "other"), ("namespace", "other"),
            ("replicates", 4), ("policy", "raw_argmax"), ("max_boundaries", 200))
        for key, value in mutations:
            self.bundle = deepcopy(original)
            self.bundle["continuation_contract"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.reconcile()

    def test_runtime_and_selector_statistics_drift(self):
        selected = self.bundle["selections"][self.inc.identity]
        original = deepcopy(selected)
        for key, value in (("runtime_sha256", "b"*64), ("root_id", "other"),
                ("statistics_root_id", "other"), ("selection_seed", True),
                ("independent_root_statistics", False), ("nominal_search_seconds", True),
                ("elapsed_seconds", float("nan")), ("action", True)):
            selected.clear()
            selected.update(original)
            selected[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.reconcile()

    def test_raw_argmax_proof_required_and_deterministic_tie(self):
        raw = self.bundle["selections"]["raw"]
        raw["action"] = 1
        with self.assertRaises(ValueError):
            self.reconcile()
        raw["action"] = 0
        raw["evidence"]["priors"] = [.5, .5]
        self.reconcile()

    def test_stage_local_one_arm_does_not_require_other_arm_or_factorial(self):
        self.configs = (self.raw, self.inc)
        self.bundle["selections"].pop(self.ref.identity)
        self.runtime_hashes.pop(self.ref.identity)
        self.assertEqual(set(self.reconcile()["root_intervals"]), {self.inc.identity})

    def test_zero_work_nonforced_cannot_score(self):
        self.bundle["selections"][self.inc.identity]["evidence"]["total_iterations"] = 0
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_fixed_work_diagnostic_cannot_promote_into_deployment(self):
        self.bundle["selections"][self.inc.identity]["fixed_work_diagnostic"] = {}
        with self.assertRaises(ValueError):
            self.reconcile()
        self.bundle["selections"][self.inc.identity].pop("fixed_work_diagnostic")
        self.configs = (self.raw, replace(self.inc, leaf="raw_rollout"))
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_validator_sees_detached_copies_not_input_or_shared_bank(self):
        before = deepcopy(self.bundle)
        def validate(cfg, row, binding):
            sha = m.digest(row)
            row["action"] = 999
            binding["legal_action_mask"] = []
            return sha
        self.reconcile(selector_validator=validate)
        self.assertEqual(self.bundle, before)

    def test_outcome_validation_once_per_unique_action_cell(self):
        seen = []
        def validate(row, contract):
            seen.append((row["action"], row["replicate"]))
            return m.digest(row)
        self.reconcile(outcome_validator=validate)
        self.assertEqual(len(seen), 16)
        self.assertEqual(len(set(seen)), 16)


if __name__ == "__main__":
    unittest.main()
