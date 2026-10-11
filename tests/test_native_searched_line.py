"""Excluded actual native search -> actual Showdown deterministic ancestry.

Passing these tests proves capture/replay and honest mismatch accounting, not
general searched-line fidelity or scientific calibration.
"""
import hashlib
import json
import os
import time
import unittest

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, pokezero_search
import test_raw_policy_leaf_native as raw_fixture
from pokezero.mcts_eval.search_over_raw_searched_line import (
    DeterministicSearchedLineFixture, run_deterministic_fixture, validate_fixture_evidence)


@unittest.skipUnless(_crate_ready and hasattr(pokezero_search.NativeVisitedValueBank, "check_searched_lines"),
    "requires isolated prospective searched-line native build")
class NativeSearchedLineTest(_EncodedSearchFixture, unittest.TestCase):
    certain = staticmethod(raw_fixture.RawPolicyLeafNativeTest.certain)
    run_search = raw_fixture.RawPolicyLeafNativeTest.run_search

    def handle(self, **options):
        return pokezero_search.NativeVisitedValueBank(sample_seed=17, worker=0,
            root_information_key="aa", original_deadline_at=time.perf_counter()+60,
            capture_searched_lines=True, **options)

    def test_actual_searched_ancestry_replays_and_preserves_pp_mismatch(self):
        fixture = DeterministicSearchedLineFixture(showdown_root=os.environ["POKEZERO_SHOWDOWN_ROOT"])
        previous = self.position
        self.position = fixture.position
        handle = self.handle(leaves_per_native_invocation=8)
        try:
            raw_report = self.run_search(rollout_leaf_mode=None, visited_value_bank=handle)
            evidence = json.loads(handle.check_searched_lines(fixture))
            self.assertGreater(raw_report["iterations"], 0)
            self.assertEqual(evidence["model_evaluations_seen"], raw_report["model_evals"]-1)
            self.assertEqual(evidence["root_native_state_sha256"],
                hashlib.sha256(self.position["state_str"].encode()).hexdigest())
            self.assertGreater(len(evidence["leaves"]), 0)
            self.assertTrue(any(row["path_boundaries"] >= 2 for row in evidence["leaves"]),
                "must exercise an ancestor, not just a fresh one-turn check")
            for row in evidence["leaves"]:
                self.assertEqual(row["result"]["checked_transitions"], row["path_boundaries"])
                self.assertEqual(row["result"]["status"], "MISMATCH")
                self.assertEqual(row["result"]["mismatch_components"], ["native_pp"])
            for flag in ("full_state_correspondence", "searched_line_showdown_fidelity",
                    "scientific_calibration_evidence", "scientific_strength_evidence", "labels_used_for_backup"):
                self.assertIs(evidence[flag], False)
            self.assertNotIn(self.position["state_str"], json.dumps(evidence))
            self.assertLessEqual(evidence["retained_private_payload_bytes"], evidence["maximum_payload_bytes"])
            with self.assertRaisesRegex(ValueError, "selection required"):
                handle.check_searched_lines(fixture)
            with self.assertRaisesRegex(ValueError, "selection required"):
                handle.label()
        finally:
            self.position = previous
            handle.close()
            fixture.close()

    def test_default_capture_cannot_be_promoted_to_path_evidence(self):
        handle = pokezero_search.NativeVisitedValueBank(sample_seed=17, worker=0,
            root_information_key="aa", original_deadline_at=time.perf_counter()+60)
        self.run_search(rollout_leaf_mode=None, visited_value_bank=handle)
        with self.assertRaisesRegex(ValueError, "not enabled"):
            handle.check_searched_lines(lambda _: "{}")
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()

    def test_private_controller_error_consumes_handle_and_never_calls_policy(self):
        policy_calls = []
        def policy(raw):
            policy_calls.append(json.loads(raw))
            return self.certain(raw)
        handle = self.handle()
        self.run_search(callbacks=[policy, policy], rollout_leaf_mode=None, visited_value_bank=handle)
        self.assertFalse(policy_calls)
        def failed(private):
            self.assertIn("steps", json.loads(private))
            raise ValueError("excluded controller failure")
        with self.assertRaisesRegex(ValueError, "controller failure"):
            handle.check_searched_lines(failed)
        self.assertFalse(policy_calls, "private fidelity state must not enter policy callbacks")
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.check_searched_lines(failed)

    def test_private_reply_or_false_promotion_is_not_exported(self):
        for reply in ({"status": "FULL_FIDELITY", "checked_transitions": 1, "mismatch_components": []},
                {"status": "MISMATCH", "checked_transitions": 0, "mismatch_components": ["secret"]},
                {"status": "MISMATCH", "checked_transitions": 0, "mismatch_components": [], "state": "private"}):
            with self.subTest(reply=reply):
                handle = self.handle()
                self.run_search(rollout_leaf_mode=None, visited_value_bank=handle)
                with self.assertRaisesRegex(ValueError, "non-sanitized"):
                    handle.check_searched_lines(lambda _: json.dumps(reply))

    def test_combined_ancestry_payload_cap_refuses_instead_of_dropping_rows(self):
        handle = self.handle(maximum_payload_bytes=1)
        with self.assertRaisesRegex(ValueError, "payload cap"):
            self.run_search(rollout_leaf_mode=None, visited_value_bank=handle)
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.check_searched_lines(lambda _: "{}")

    def test_shared_driver_reports_actual_multi_boundary_check_without_scientific_admission(self):
        result = run_deterministic_fixture(native_model=self.native, tables_json=self.tables_json,
            showdown_root=os.environ["POKEZERO_SHOWDOWN_ROOT"])
        self.assertEqual(result["actual_search"]["iterations"], 8)
        self.assertEqual(result["evidence"]["model_evaluations_seen"], 2)
        self.assertEqual(len(result["controller_readout"]), 2)
        self.assertFalse(result["calibration_admission"])
        self.assertFalse(result["full_native_state_correspondence"])
        self.assertIn("OOV", result["fixture_vocabulary_limitation"])
        self.assertTrue(all(row["result"]["mismatch_components"] == ["native_pp"]
            for row in result["evidence"]["leaves"]))

    def test_fixture_admission_refuses_empty_one_turn_or_partial_replays(self):
        valid = dict(leaves=[dict(path_boundaries=2, result=dict(checked_transitions=2))])
        validate_fixture_evidence(valid)
        for value in (dict(leaves=[]), dict(leaves=[dict(path_boundaries=1,
                result=dict(checked_transitions=1))]), dict(leaves=[dict(path_boundaries=2,
                result=dict(checked_transitions=1))])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_fixture_evidence(value)


if __name__ == "__main__":
    unittest.main()
