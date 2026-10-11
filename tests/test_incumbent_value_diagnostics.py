"""Excluded engineering handoff fixtures; no scientific or fidelity evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.engine_search import EngineMctsPolicy, EngineSearchWitnessError, OpponentRequestOrderResolution
from pokezero.mcts_eval.search_over_raw import digest, rng_seed, SearchConfiguration
from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
from pokezero.mcts_eval.search_over_raw_value_diagnostics import (
    ValueDiagnosticContract, NativeValueDiagnosticSession, reconcile_collected_values)
from test_model_tree_hp_leaf_contract import config
from test_policy_opponent_engine import Fixtures


class FakeNativeBank:
    """Declared synthetic FFI, never a native evaluation or Showdown result."""
    def __init__(self, **kwargs):
        self.kwargs, self.closed, self.ready, self.labels_called = kwargs, False, False, 0

    def record(self, state, raw):
        self.state, self.raw, self.ready = state, raw, True

    def close(self):
        self.closed = True

    def label(self):
        if self.closed or not self.ready:
            raise ValueError("synthetic consumed handle")
        self.labels_called += 1
        self.closed = True
        k = self.kwargs
        n = json.loads(self.raw)["model_evals"]-1
        priorities = sorted((rng_seed("visited-model-sample.v1", k["sample_seed"], k["worker"],
            k["root_information_key"], ordinal), ordinal) for ordinal in range(n))[:k["leaves_per_native_invocation"]]
        leaves = [dict(evaluation_ordinal=ordinal, priority=priority, native_state_sha256="c"*64,
            snapshot_sha256="d"*64, model_signed_value=.2, labels=[dict(replicate=rep,
                status="DEADLINE_UNCERTAIN", signed_outcome=None, boundaries=0,
                continuation_policy="raw_argmax_both_seats", seed=rng_seed("native-visited-label.v1",
                    k["sample_seed"], k["worker"], k["root_information_key"], ordinal, rep))
                for rep in range(8)]) for priority, ordinal in priorities]
        return json.dumps(dict(schema="pokezero.native-visited-value.engineering.v1", worker=k["worker"],
            root_information_key=k["root_information_key"], sample_seed=k["sample_seed"],
            original_deadline_at=k["original_deadline_at"],
            root_native_state_sha256=hashlib.sha256(self.state.encode()).hexdigest(),
            selection_report_sha256=hashlib.sha256(self.raw.encode()).hexdigest(),
            leaves_per_native_invocation=k["leaves_per_native_invocation"], replicates=8, maximum_boundaries=250,
            model_evaluations_seen=n, sampling_unit="successful_model_evaluation_event",
            sampling="smallest_seeded_priorities_per_native_invocation", value_frame="self_relative",
            retained_payload_bytes=1, maximum_payload_bytes=k["maximum_payload_bytes"],
            payload_cap_is_not_rss_qualification=True, leaves=leaves, label_lossy_subcases={},
            label_engine="poke_engine_unqualified_against_showdown", searched_line_showdown_fidelity=False,
            scientific_calibration_evidence=False, scientific_strength_evidence=False, labels_used_for_backup=False,
            private_frontiers_exported=False, instrumentation_can_change_selection=True))


FAKE_MODULE = SimpleNamespace(NativeVisitedValueBank=FakeNativeBank,
    FoldState=SimpleNamespace(from_payload=lambda payload: object()))


def synthetic_session(contract, root_key="0102"):
    session = NativeValueDiagnosticSession(contract, root_key=root_key, native_module=FAKE_MODULE)
    record = dict(state_str="excluded-synthetic-native-state", seed=17)
    handle = session.begin(record)
    raw = json.dumps(dict(model_evals=2))
    handle.record(record["state_str"], raw)
    session.complete(handle, raw)
    return session


class IncumbentValueIntegrationTests(unittest.TestCase):
    def contract(self):
        return ValueDiagnosticContract(7, 100., incumbent_sampling="per_native_invocation.v1")

    def selected(self, session):
        bindings = session.bindings()
        return dict(root_id="excluded", configuration_sha256="a"*64, runtime_sha256="b"*64,
            evidence=dict(visited_value_native_invocations=bindings, visited_value_root_legal_actions=2,
                model_evals=sum(row["model_evaluations_seen"]+1 for row in bindings),
                engine_mcts=dict(time_budget=dict(native_invocations=[dict(status="completed",
                    world_seed=row["world_seed"]) for row in bindings]))))

    def adapter(self, session):
        adapter = object.__new__(PublicModelSearchAdapter)
        adapter._closed = adapter._poisoned = False
        adapter._pool = None
        adapter.configuration = SimpleNamespace(identity="a"*64)
        adapter.runtime_sha256 = "b"*64
        selected = self.selected(session)
        adapter._pending_value_diagnostics = (selected, digest(selected), session)
        return adapter, selected

    def test_exact_adapter_handoff_reconciles_engineering_labels_and_consumes_private_handles(self):
        session = synthetic_session(self.contract())
        handle = session.invocations[0]["handle"]
        adapter, selected = self.adapter(session)
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            adapter.finish_value_diagnostics(dict(selected))
        values = adapter.finish_value_diagnostics(selected)
        summary = reconcile_collected_values(values, selected=selected, contract=self.contract(),
            workers=1, information_key="0102", arm="incumbent")
        self.assertEqual((summary["native_invocations"], summary["labels"], summary["unknown_labels"]), (1, 8, 8))
        self.assertFalse(summary["scientific_calibration_evidence"])
        self.assertTrue(handle.closed)
        self.assertEqual(handle.labels_called, 1)
        self.assertFalse(session.invocations)
        with self.assertRaises(ValueError):
            adapter.finish_value_diagnostics(selected)

    def test_tampered_native_rosters_predictions_reports_and_sampling_refuse(self):
        session = synthetic_session(self.contract())
        adapter, selected = self.adapter(session)
        values = adapter.finish_value_diagnostics(selected)
        for mutate in (
            lambda v: v["native_invocations"].clear(),
            lambda v: v["native_invocations"][0]["evidence"].update(selection_report_sha256="e"*64),
            lambda v: v["native_invocations"][0]["evidence"].update(scientific_calibration_evidence=True),
            lambda v: v["native_invocations"][0]["evidence"]["leaves"][0].update(model_signed_value=float("nan")),
            lambda v: v["native_invocations"][0]["evidence"]["leaves"][0].update(model_signed_value=True),
            lambda v: v["native_invocations"][0]["evidence"]["leaves"][0].update(priority=1),
            lambda v: v["native_invocations"][0]["evidence"]["leaves"][0]["labels"].pop(),
            lambda v: v["native_invocations"][0]["evidence"].update(private_frontiers_exported=True)):
            changed = deepcopy(values)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                reconcile_collected_values(changed, selected=selected, contract=self.contract(),
                    workers=1, information_key="0102", arm="incumbent")
        with self.assertRaisesRegex(ValueError, "cannot be reference"):
            reconcile_collected_values(values, selected=selected, contract=self.contract(), workers=1, information_key="0102")

    def test_native_failure_consumes_invocation_and_cannot_be_promoted(self):
        session = NativeValueDiagnosticSession(self.contract(), root_key="0102", native_module=FAKE_MODULE)
        handle = session.begin(dict(seed=7, state_str="excluded"))
        with self.assertRaisesRegex(ValueError, "cannot become a selection"):
            session.bindings()
        session.close()
        self.assertTrue(handle.closed)
        with self.assertRaises(ValueError):
            session.begin(dict(seed=7, state_str="excluded"))

    def test_omitted_native_calls_cannot_be_zero_leaf_evidence_even_with_rebound_hash(self):
        adapter, selected = self.adapter(synthetic_session(self.contract()))
        values = adapter.finish_value_diagnostics(selected)
        missing = deepcopy(selected)
        missing["evidence"].pop("engine_mcts")
        changed = deepcopy(values)
        changed["selected_receipt_sha256"] = digest(missing)
        with self.assertRaisesRegex(ValueError, "ledger missing"):
            reconcile_collected_values(changed, selected=missing, contract=self.contract(),
                workers=1, information_key="0102", arm="incumbent")
        selected["evidence"]["visited_value_native_invocations"] = []
        values["native_invocations"] = []
        values["selected_receipt_sha256"] = digest(selected)
        with self.assertRaisesRegex(ValueError, "omission"):
            reconcile_collected_values(values, selected=selected, contract=self.contract(),
                workers=1, information_key="0102", arm="incumbent")
        selected["evidence"].pop("engine_mcts")
        values["selected_receipt_sha256"] = digest(selected)
        with self.assertRaisesRegex(ValueError, "forced/no-native"):
            reconcile_collected_values(values, selected=selected, contract=self.contract(),
                workers=1, information_key="0102", arm="incumbent")

    def run_engine(self, *, leaf=None, duplicate=False, failed=False, cancelled=False):
        policy = Fixtures._policy(workers=1)
        policy._config = config(worlds=4, model_leaf_override=leaf, search_sims=100, search_batch=10,
            **(dict(rollout_policy="raw_argmax", rollout_count=1, rollout_threads=1, rollout_max_plies=250,
                rollout_seed=7, rollout_branch_on_damage=True) if leaf == "raw_policy_terminal" else {}),
            **(dict(model_decision_time_ms=10000) if cancelled else {}))
        policy._value_diagnostic_session = session = NativeValueDiagnosticSession(self.contract(),
            root_key="0102", native_module=FAKE_MODULE)
        ctx = Fixtures._context()
        ctx.public_materialization_state.self_request = {"side": {"pokemon": [{"details": "Rattata, L100"}]}}
        calls = []
        class Native:
            def search_batched_multi_encoded(self, *args, **kwargs):
                calls.append((args, kwargs))
                if failed:
                    raise ValueError("excluded native failure")
                report = Fixtures._timed_report(60, 40, time_budget_ms=args[-1]) if cancelled else Fixtures._report(60, 40)
                if leaf == "hp_fraction":
                    report.update(model_leaf_override="hp_fraction", hp_leaf_model_forwards_retained=True,
                        hp_leaf_value_frame="side_one_absolute", hp_leaf_rows_priced=report["model_evals"]-1)
                if leaf == "raw_policy_terminal":
                    import test_model_tree_raw_leaf_contract as raw_fixture
                    report.update(raw_fixture.report(args[9]))
                    from pokezero.engine_search import raw_policy_leaf_seed
                    report["raw_leaf_seed"] = raw_policy_leaf_seed(policy._config, {"seed": args[9]})
                if cancelled and len(calls) == 1:
                    report.update(iterations=0, requested_iterations=100, remaining_iterations=100,
                        model_evals=3, side_one=[], side_two=[], root_priors=[],
                        time_budget_exhausted=True, time_budget_elapsed_ms=float(args[-1])+1,
                        time_budget_batch_overshoot_ms=1.)
                raw = json.dumps(report)
                kwargs["visited_value_bank"].record(args[0], raw)
                return raw
        with patch.dict(sys.modules, {"pokezero_search": FAKE_MODULE}), \
                patch.object(policy, "_native", return_value=Native()), \
                patch.object(policy, "_validate_model_root_observation"), \
                patch.object(policy, "_root_inputs_json", return_value="{}"), \
                patch.object(policy, "_checkpoint_policy_callback", return_value=lambda raw: (1.,)), \
                patch("pokezero.engine_search.opponent_request_order_resolution", return_value=
                    OpponentRequestOrderResolution(("chansey",), "resolved")):
            try:
                result = policy._search_model(ctx, [Fixtures._world("a"), Fixtures._world("a" if duplicate else "b")],
                    SimpleNamespace(to_payload=lambda: {}), random.Random(7))
            except BaseException:
                session.close()
                self.failed_session, self.failed_calls = session, calls
                raise
        return result, calls, session, policy

    def test_real_engine_call_seam_receives_owned_handle_orders_and_independent_callbacks(self):
        _, calls, session, _ = self.run_engine()
        self.assertEqual(len(session.bindings()), len(calls))
        self.assertEqual(len(calls), 2)
        rng = random.Random(7)
        self.assertEqual([a[9] for a, _ in calls], [rng.getrandbits(63), rng.getrandbits(63)])
        for _, kwargs in calls:
            self.assertEqual(kwargs["raw_policy_request_orders"], [["rattata"], ["chansey"]])
            self.assertEqual(len(kwargs["raw_policy_callbacks"]), 2)
        session.close()

    def test_actual_call_path_all_leaf_modes_and_collapsed_worlds_keep_one_label_credit(self):
        for leaf in (None, "hp_fraction", "raw_policy_terminal"):
            with self.subTest(leaf=leaf):
                _, calls, session, _ = self.run_engine(leaf=leaf)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(session.label()), 2)
        _, calls, session, _ = self.run_engine(duplicate=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(session.label()), 1)

    def test_cancelled_native_invocation_remains_in_roster_and_failed_call_discards_handles(self):
        decision, calls, session, _ = self.run_engine(cancelled=True)
        actual = decision.metadata["engine_mcts"]["time_budget"]["native_invocations"]
        self.assertEqual(len(calls), len(actual))
        self.assertEqual(actual[0]["completed_iterations"], 0)
        self.assertEqual(session.bindings()[0]["model_evaluations_seen"], 2)
        self.assertEqual(len(session.label()), 2)
        with self.assertRaises(EngineSearchWitnessError):
            self.run_engine(failed=True)
        self.assertTrue(self.failed_session.closed)
        self.assertFalse(self.failed_session.invocations)
        self.assertTrue(self.failed_calls[0][1]["visited_value_bank"].closed)

    def test_adapter_optin_prepares_native_runtime_and_default_remains_reference_only(self):
        checkpoint = SimpleNamespace(checkpoint_path="weights", checkpoint_sha256="a"*64, showdown_source_sha256="b"*64)
        cfg = SearchConfiguration("incumbent")
        native = Mock()
        with patch("pokezero.mcts_eval.search_over_raw_adapters._incumbent_runtime", return_value=(Mock(), native, Mock())):
            adapter = PublicModelSearchAdapter(cfg, checkpoint_contract=checkpoint, showdown_root="none",
                value_diagnostics=self.contract())
            native.prepare_visited_value_runtime.assert_called_once()
            self.assertFalse(adapter.runtime_configuration["visited_value_diagnostics"]["equals_reference_sampling"])
            adapter.close()
            with self.assertRaisesRegex(ValueError, "supported arm"):
                PublicModelSearchAdapter(cfg, checkpoint_contract=checkpoint, showdown_root="none",
                    value_diagnostics=replace(self.contract(), incumbent_sampling=None))

    def test_runtime_gate_rejects_parallel_or_nonincumbent_allocations(self):
        policy = object.__new__(EngineMctsPolicy)
        baseline = config(worlds=4, model_leaf_override=None)
        policy._config = baseline
        with patch.object(policy, "_load_own_policy_runtime") as loader:
            policy.prepare_visited_value_runtime()
            loader.assert_called_once()
            for change in (dict(worlds=2), dict(model_world_workers=2), dict(use_opponent_priors=True),
                    dict(depth_min=2), dict(worlds_min=1), dict(rollout_leaf_eval=True),
                    dict(rollout_leaf_shadow=True), dict(strict_fallbacks=False),
                    dict(early_stop=True, early_stop_min_sims=1)):
                with self.subTest(change=change):
                    policy._config = replace(baseline, **change)
                    with self.assertRaises(EngineSearchWitnessError):
                        policy.prepare_visited_value_runtime()
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
