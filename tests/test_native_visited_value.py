"""Actual native invocation fixtures, excluded from scientific evidence.

The labels are poke-engine results, not Showdown ground truth. This closes the
native prediction/frontier/lifetime seam only, not fidelity or producer admission.
"""
import hashlib
import json
import time
import unittest
from types import SimpleNamespace

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, pokezero_search
import test_raw_policy_leaf_native as raw_fixture
from test_rollout_model_priors import TIMING_FIELDS
from pokezero.mcts_eval.search_over_raw import rng_seed


@unittest.skipUnless(_crate_ready and hasattr(pokezero_search, "NativeVisitedValueBank"),
    "requires isolated prospective native visited-value build")
class NativeVisitedValueTest(_EncodedSearchFixture, unittest.TestCase):
    certain = staticmethod(raw_fixture.RawPolicyLeafNativeTest.certain)
    run_search = raw_fixture.RawPolicyLeafNativeTest.run_search

    def handle(self, *, deadline=None, **changes):
        return pokezero_search.NativeVisitedValueBank(**{
            "sample_seed": 17, "worker": 0, "root_information_key": "aa",
            "original_deadline_at": deadline if deadline is not None else time.perf_counter() + 60,
            **changes})

    def capture(self, handle, **changes):
        return self.run_search(rollout_leaf_mode=None, visited_value_bank=handle, **changes)

    def test_actual_model_tree_capture_retains_nonroot_states_without_altering_fixed_work(self):
        baseline = self.run_search(rollout_leaf_mode=None, raw_policy_callbacks=None,
            raw_policy_request_orders=None)
        handle = self.handle()
        report = self.capture(handle)
        self.assertEqual({k: v for k, v in report.items() if k not in TIMING_FIELDS},
            {k: v for k, v in baseline.items() if k not in TIMING_FIELDS})
        # Operational label failure below is deliberate; it proves actual
        # capture and bridge handoff without claiming arbitrary native lines
        # are faithful calibration ground truth.
        handle.close()
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()

    def test_post_selection_deadline_retains_full_sample_and_eight_null_labels(self):
        deadline = time.perf_counter() + 0.5
        handle = self.handle(deadline=deadline)
        report = self.capture(handle)
        self.assertGreater(report["iterations"], 0)
        time.sleep(max(0.0, deadline - time.perf_counter()) + .01)
        values = json.loads(handle.label())
        self.assertEqual(values["schema"], "pokezero.native-visited-value.engineering.v1")
        self.assertEqual(values["root_native_state_sha256"],
            hashlib.sha256(self.position["state_str"].encode()).hexdigest())
        self.assertEqual(values["model_evaluations_seen"], report["model_evals"] - 1)
        expected = sorted((rng_seed("visited-model-sample.v1", 17, 0, "aa", ordinal), ordinal)
            for ordinal in range(values["model_evaluations_seen"]))[:2]
        self.assertEqual([(r["priority"], r["evaluation_ordinal"]) for r in values["leaves"]], expected)
        self.assertTrue(any(r["native_state_sha256"] != values["root_native_state_sha256"]
            for r in values["leaves"]), "a root label cannot replace a visited-state label")
        for row in values["leaves"]:
            self.assertTrue(-1 <= row["model_signed_value"] <= 1)
            self.assertEqual(len(row["labels"]), 8)
            for label in row["labels"]:
                self.assertEqual(label["status"], "DEADLINE_UNCERTAIN")
                self.assertIsNone(label["signed_outcome"])
                self.assertEqual(label["boundaries"], 0)
        self.assertFalse(values["scientific_calibration_evidence"])
        self.assertFalse(values["searched_line_showdown_fidelity"])
        self.assertFalse(values["labels_used_for_backup"])
        self.assertNotIn(self.position["state_str"], json.dumps(values))
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.capture(handle)

    def test_post_selection_callback_receives_private_safe_frontier_and_failure_consumes_labels(self):
        calls = []
        def broken(raw):
            calls.append(json.loads(raw))
            raise ValueError("excluded diagnostic label failure")
        handle = self.handle()
        report = self.capture(handle, callbacks=[broken, self.certain])
        self.assertFalse(calls, "post-selection diagnostics must never price search leaves")
        self.assertGreater(report["iterations"], 0)
        with self.assertRaisesRegex(ValueError, "excluded diagnostic label failure"):
            handle.label()
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), {"native_request_bundle", "public_branch_lines", "opponent_slot"})
        self.assertTrue(calls[0]["public_branch_lines"])
        self.assertEqual(calls[0]["opponent_slot"], "p1")
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()

    def test_actual_hp_capture_preserves_model_forwards_and_fixed_work(self):
        kwargs = dict(rollout_leaf_mode="hp_fraction", rollout_policy="uniform")
        baseline = self.run_search(raw_policy_callbacks=None, raw_policy_request_orders=None, **kwargs)
        handle = self.handle()
        report = self.run_search(visited_value_bank=handle, **kwargs)
        self.assertEqual({k: v for k, v in report.items() if k not in TIMING_FIELDS},
            {k: v for k, v in baseline.items() if k not in TIMING_FIELDS})
        self.assertTrue(report["hp_leaf_model_forwards_retained"])
        handle.close()

    def test_failed_raw_search_poisoned_handle_cannot_label_or_repeat_capture(self):
        handle = self.handle()
        with self.assertRaisesRegex(ValueError, "nonterminal ply cap"):
            self.run_search(visited_value_bank=handle)
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.capture(handle)

    def test_invalid_sampling_context_deadline_or_payload_contract_refuses(self):
        for changes in (dict(root_information_key=""), dict(root_information_key="xyz"),
                dict(leaves_per_native_invocation=0), dict(leaves_per_native_invocation=9),
                dict(maximum_payload_bytes=0), dict(maximum_payload_bytes=8388609),
                dict(original_deadline_at=float("nan")), dict(original_deadline_at=time.perf_counter()+14401)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.handle(**changes)
        handle = self.handle(deadline=time.perf_counter()-1)
        with self.assertRaisesRegex(ValueError, "deadline expired"):
            self.capture(handle)
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()

    def test_original_deadline_or_payload_failure_never_invents_a_selection(self):
        handle = self.handle(maximum_payload_bytes=1)
        with self.assertRaisesRegex(ValueError, "payload cap"):
            self.capture(handle)
        with self.assertRaisesRegex(ValueError, "selection required"):
            handle.label()

    def test_actual_native_session_through_adapter_finish_preserves_report_hashes_and_null_roster(self):
        from pokezero.mcts_eval.search_over_raw_value_diagnostics import (
            NativeValueDiagnosticSession, ValueDiagnosticContract, reconcile_collected_values)
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw import digest
        deadline = time.perf_counter() + .5
        contract = ValueDiagnosticContract(17, deadline, incumbent_sampling="per_native_invocation.v1")
        session = NativeValueDiagnosticSession(contract, root_key="aa", native_module=pokezero_search)
        actual_native = self.native
        class RecordingNative:
            def search_batched_multi_encoded(_, *args, **kwargs):
                raw = actual_native.search_batched_multi_encoded(*args, **kwargs)
                session.complete(kwargs["visited_value_bank"], raw)
                return raw
        self.native = RecordingNative()
        try:
            for seed in (5, 7):
                handle = session.begin(dict(state_str=self.position["state_str"], seed=seed))
                self.capture(handle, seed=seed)
        finally:
            self.native = actual_native
        selected = dict(root_id="excluded-native-session", configuration_sha256="a"*64,
            runtime_sha256="b"*64, evidence=dict(visited_value_native_invocations=session.bindings(),
                model_evals=sum(row["model_evaluations_seen"]+1 for row in session.bindings()),
                visited_value_root_legal_actions=2, engine_mcts=dict(time_budget=dict(native_invocations=[
                    dict(status="completed", world_seed=row["world_seed"]) for row in session.bindings()]))))
        adapter = object.__new__(PublicModelSearchAdapter)
        adapter._closed = adapter._poisoned = False
        adapter._pool = None
        adapter.configuration = SimpleNamespace(identity="a"*64)
        adapter.runtime_sha256 = "b"*64
        adapter._pending_value_diagnostics = (selected, digest(selected), session)
        time.sleep(max(0., deadline-time.perf_counter())+.01)
        values = adapter.finish_value_diagnostics(selected)
        summary = reconcile_collected_values(values, selected=selected, contract=contract,
            workers=1, information_key="aa", arm="incumbent")
        self.assertEqual(summary["native_invocations"], 2)
        self.assertEqual(summary["labels"], 32)
        self.assertEqual(summary["unknown_labels"], 32)
        self.assertFalse(summary["scientific_calibration_evidence"])
        self.assertFalse(session.invocations)
        with self.assertRaisesRegex(ValueError, "no live"):
            adapter.finish_value_diagnostics(selected)


if __name__ == "__main__":
    unittest.main()
