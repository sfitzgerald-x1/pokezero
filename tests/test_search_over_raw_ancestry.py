"""Authored sampled paths plus actual disposable ownership; no native model."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import pickle
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest
from pokezero.mcts_eval.search_over_raw_ancestry import (
    SCHEMA, POPULATION, NativeAncestryContract, NativeAncestrySession, _priority,
    read_native_ancestry_export, validate_owned_native_ancestry,
)
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator
from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
import test_search_over_raw_fixed_work_run as owning_fixture
from test_search_over_raw_supervision import BINDING, INPUT_BINDING, selection_inputs

CONTRACT = NativeAncestryContract(17)


def export_fixture(call, worker=0):
    # Two actual toy model rows, plus the root-prior forward. Not a real model.
    report = call["native_report"]
    report["model_evals"] = 3
    report["collision_self_side"] = "side_one"
    raw = json.dumps(report, indent=1)
    call.update(native_report_utf8=raw, report_utf8_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        report_sha256=digest(report))
    leaves = []
    for ordinal in range(2):
        root, end = "complete-state", f"branch-{ordinal}"
        step = dict(key=[0, ordinal], parent=None, depth=0, before=root, after=end,
            branch_on_damage=True, side_one_choice=dict(kind="move", engine_id="move"),
            side_two_choice=dict(kind="move", engine_id="move"))
        capture = dict(root_native_state=root, steps=[step], native_frontier=end,
            model_signed_value=(-.5 if ordinal == 0 else .5), evaluation_ordinal=ordinal,
            private_controller_only=True)
        leaves.append(dict(priority=_priority(17, worker, INPUT_BINDING["information_key"], ordinal),
            model_self_value01_bits=struct.unpack("<I", struct.pack("<f", .25 if ordinal == 0 else .75))[0],
            branch_key=[0, ordinal], native_state_sha256=hashlib.sha256(end.encode()).hexdigest(),
            private_frontier_sha256="b"*64, capture=capture))
    leaves.sort(key=lambda item:(item["priority"], item["capture"]["evaluation_ordinal"]))
    payload = dict(schema=SCHEMA, root_native_state_sha256=call["native_call_parameters"]["state_sha256"],
        selection_report_utf8_sha256=call["report_utf8_sha256"], root_information_key=INPUT_BINDING["information_key"],
        sample_seed=17, worker=worker, model_evaluations_seen=2, leaves_per_native_invocation=2,
        maximum_payload_bytes=8388608, retained_private_payload_bytes=4096, evaluations=leaves,
        event_population=POPULATION, private_controller_only=True, labels_used_for_backup=False,
        full_world_equivalence=False, runtime_qualified=False, fixed_work_capture=True, model_self_side_one=True)
    return row_from_payload(payload, call, worker)


def row_from_payload(payload, call, worker=0):
    raw = json.dumps(payload)
    return dict(invocation=worker, world_seed=call["native_call_parameters"]["chance_seed"],
        export_utf8=raw, export_utf8_sha256=hashlib.sha256(raw.encode()).hexdigest())


@dataclass(frozen=True)
class AuthoredAncestryJob(owning_fixture.AuthoredFixedWorkJob):
    def __call__(self, ownership, phase):
        selected = super().__call__(ownership, phase)
        calls = selected["evidence"]["consumed_world_inputs"]["calls"]
        selected["evidence"]["native_searched_ancestry"] = [export_fixture(c, i) for i, c in enumerate(calls)]
        return selected


class AncestryTests(unittest.TestCase):
    def fixture(self):
        call = selection_inputs("incumbent")["consumed_world_inputs"]["calls"][0]
        return export_fixture(call), call

    def read(self, row, call, **kwargs):
        return read_native_ancestry_export(row, call, **dict(dict(contract=CONTRACT,
            root_key=INPUT_BINDING["information_key"], worker=0, root_binding=BINDING, searching_side_one=True), **kwargs))

    def test_exact_sample_atomic_membership_and_full_paths(self):
        row, call = self.fixture()
        result = self.read(row, call)
        self.assertEqual(result["actual_model_evaluations"], 2)
        self.assertEqual(len(result["retained_evaluations"]), 2)
        self.assertEqual([r["committed_round"] for r in result["retained_evaluations"]], [0,0])
        self.assertTrue(all(len(r["boundaries"]) == 2 for r in result["retained_evaluations"]))
        self.assertFalse(result["full_world_equivalence"])

    def test_schema_denominator_sampler_root_and_membership_drift(self):
        for mode in ("old_schema", "population", "seen", "priority", "missing", "duplicate", "report",
                "state", "seed", "worker", "key", "signed", "path", "numeric", "fixed", "cap"):
            row, call = self.fixture(); payload = json.loads(row["export_utf8"])
            leaf = payload["evaluations"][0]
            if mode == "old_schema": payload["schema"] = "pokezero.native-searched-line.excluded-engineering.v1"
            if mode == "population": payload["event_population"] = "accepted_backups"
            if mode == "seen": payload["model_evaluations_seen"] = 3
            if mode == "priority": leaf["priority"] += 1
            if mode == "missing": payload["evaluations"].pop()
            if mode == "duplicate": payload["evaluations"][1] = deepcopy(leaf)
            if mode == "report": payload["selection_report_utf8_sha256"] = "c"*64
            if mode == "state": payload["root_native_state_sha256"] = "c"*64
            if mode == "seed": payload["sample_seed"] = 18
            if mode == "worker": payload["worker"] = False
            if mode == "key": leaf["branch_key"][1] = 8
            if mode == "signed": leaf["capture"]["model_signed_value"] = .125
            if mode == "path": leaf["capture"]["steps"][0]["before"] = "different"
            if mode == "numeric": leaf["capture"]["steps"][0]["key"][0] = False
            if mode == "fixed": payload["fixed_work_capture"] = False
            if mode == "cap": payload["retained_private_payload_bytes"] = 8388609
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.read(row_from_payload(payload, call), call)

    def test_exact_raw_bytes_and_contract_required(self):
        row, call = self.fixture()
        with self.assertRaises(ValueError): self.read(row, call, contract=NativeAncestryContract(18))
        row["export_utf8"] += " "
        with self.assertRaises(ValueError): self.read(row, call)

    def test_p2_original_float32_is_not_reconstructed_from_reflected_value(self):
        row, call = self.fixture()
        call["native_report"]["collision_self_side"] = "side_two"
        raw = json.dumps(call["native_report"])
        call.update(native_report_utf8=raw, report_utf8_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            report_sha256=digest(call["native_report"]))
        payload = json.loads(row["export_utf8"])
        payload.update(model_self_side_one=False, selection_report_utf8_sha256=call["report_utf8_sha256"])
        f32 = lambda value:struct.unpack("<f", struct.pack("<f", value))[0]
        for item in payload["evaluations"]:
            ordinal = item["capture"]["evaluation_ordinal"]
            # Reflection of this next float after .25 rounds to .75. Inverting
            # that atomic .75 would wrongly erase the original model's bits.
            original = .75 if ordinal == 0 else f32(.2500000298023224)
            item["model_self_value01_bits"] = struct.unpack("<I", struct.pack("<f", original))[0]
            item["capture"]["model_signed_value"] = f32(f32(2*original)-1)
        self.assertEqual(f32(1-f32(.2500000298023224)), .75)
        result = self.read(row_from_payload(payload, call), call, searching_side_one=False)
        self.assertEqual(result["actual_model_evaluations"], 2)
        with self.assertRaises(ValueError): self.read(row_from_payload(payload, call), call)
        row, call = self.fixture(); call["native_report_utf8"] += " "
        with self.assertRaises(ValueError): self.read(row, call)

    def test_session_is_one_shot_no_labels_and_fixed_deadline_mode(self):
        row, call = self.fixture(); handle = Mock()
        handle.export_searched_ancestry.return_value = row["export_utf8"]
        module = SimpleNamespace(NativeVisitedValueBank=Mock(return_value=handle))
        session = NativeAncestrySession(CONTRACT, root_key=INPUT_BINDING["information_key"], native_module=module)
        record = dict(state_str="complete-state", seed=7)
        self.assertIs(session.begin(record), handle)
        self.assertTrue(module.NativeVisitedValueBank.call_args.kwargs["fixed_work_capture"])
        self.assertEqual(module.NativeVisitedValueBank.call_args.kwargs["original_deadline_at"], 0.)
        with self.assertRaises(ValueError): session.begin(record)
        session.complete(handle, call["native_report_utf8"])
        self.assertEqual(session.exports(), [row])
        with self.assertRaises(ValueError): session.complete(handle, call["native_report_utf8"])
        with self.assertRaises(ValueError): session.exports()
        handle.label.assert_not_called(); handle.check_searched_lines.assert_not_called()
        handle.close.assert_called_once()

    def test_failed_export_is_not_retried(self):
        handle = Mock(); handle.export_searched_ancestry.side_effect = RuntimeError("authored export failure")
        session = NativeAncestrySession(CONTRACT, root_key="aa",
            native_module=SimpleNamespace(NativeVisitedValueBank=Mock(return_value=handle)))
        session.begin(dict(state_str="complete-state", seed=7))
        with self.assertRaises(RuntimeError): session.complete(handle, "{}")
        with self.assertRaises(ValueError): session.complete(handle, "{}")
        with self.assertRaises(ValueError): session.exports()
        handle.export_searched_ancestry.assert_called_once(); handle.close.assert_called_once()

    def test_real_owned_synthetic_run_remains_nonclearing_by_default(self):
        fixture = owning_fixture.FixedWorkRunReaderTests(methodName="runTest")
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        cfg = SearchConfiguration("incumbent", seconds=1.)
        job = AuthoredAncestryJob(cfg); contract = FixedWorkDiagnosticContract()
        directory = fixture.root / "ancestry"
        terminal = _supervise_owned_job(job, directory=directory, binding_sha256=BINDING,
            verify_admission=lambda:None, contract=contract, kind="fixed-work", work_phase="selection",
            validate_result=lambda selected:derive_fixed_work_accounting(selected["evidence"],
                arm="incumbent", binding=INPUT_BINDING, statistics_root_id="same-root", leaf="model"))
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        inputs = dict(directory=directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=BINDING, configuration=cfg, root_id="same-root:"+cfg.identity,
            statistics_root_id="same-root", selection_seed=7, runtime_sha256=BINDING,
            consumed_binding=INPUT_BINDING, file_hashes=fixture.files,
            checkpoint_path=str(fixture.checkpoint), checkpoint_sha256=fixture.files[str(fixture.checkpoint)],
            showdown_root=str(fixture.showdown))
        selected = terminal["result"]
        result = validate_owned_native_ancestry(selected, run_validator=FixedWorkRunValidator(**inputs), contract=CONTRACT)
        self.assertFalse(result["owning_run_validated"])
        result = validate_owned_native_ancestry(selected,
            run_validator=FixedWorkRunValidator(**inputs, allow_synthetic=True), contract=CONTRACT)
        self.assertTrue(result["owning_run_validated"])
        self.assertEqual((result["construction_ordinals"], result["actual_native_calls"]), (4,1))
        for flag in ("full_world_equivalence", "independent_terminal_labels", "party_mapping_validated",
                "transition_correctness_validated", "resource_cost_complete", "runtime_qualified", "scientific_collection_authorized"):
            self.assertFalse(result[flag])

    def test_controller_binds_ancestry_sources(self):
        from pokezero.mcts_eval.search_over_raw_reconciled import controller_input_hashes
        names = {p.rsplit("/",1)[-1] for p in controller_input_hashes()}
        self.assertTrue({"search_over_raw_ancestry.py", "visited_value.rs", "searched_ancestry.rs"} <= names)

    def test_adapter_opt_in_prepares_and_restores_private_session_on_failure(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputFixedWorkAdapter, NativeInputHandle
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        cfg = SearchConfiguration("incumbent", seconds=1.)
        adapter = object.__new__(WorldInputFixedWorkAdapter)
        adapter._expected_public_request_sha256 = None  # Isolated fixture bypasses adapter __init__.
        adapter._timed_native_atomic_work = False
        adapter.configuration, adapter.fixed_work = cfg, FixedWorkDiagnosticContract()
        adapter._ancestry_contract, adapter.last_failure = CONTRACT, None
        policy = SimpleNamespace(_world_attempt_observer=None, _world_observer=None,
            _world_construction_observer=None, _native_construction_observer=None,
            _value_diagnostic_session=None, _native=lambda:"unchanged-model",
            _config=SimpleNamespace(model_world_workers=1, early_stop=False, search_sims=16,
                model_decision_time_ms=None),
            stats=SimpleNamespace(worlds_attempted=0, worlds_constructed=0, worlds_searched=0),
            prepare_visited_value_runtime=Mock())
        adapter._native = policy
        native_module = SimpleNamespace(NativeVisitedValueBank=Mock())
        seen = []
        def select(*args, **kwargs):
            adapter._prepare_incumbent("authored-request")
            seen.append(policy._value_diagnostic_session)
            self.assertIsInstance(policy._native(), NativeInputHandle)
            raise RuntimeError("authored stopped selector")
        with patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=INPUT_BINDING), \
             patch.object(PublicModelSearchAdapter, "_prepare_incumbent", return_value=None), \
             patch.object(PublicModelSearchAdapter, "select", side_effect=select), \
             patch.dict("sys.modules", {"pokezero_search":native_module}):
            with self.assertRaises(RuntimeError): adapter.select(None)
        policy.prepare_visited_value_runtime.assert_called_once()
        self.assertTrue(seen[0].closed)
        self.assertIsNone(policy._value_diagnostic_session)
        self.assertIsNone(policy._world_construction_observer)
        self.assertEqual(policy._native(), "unchanged-model")

    def test_owning_job_forwards_explicit_ancestry_contract(self):
        import test_search_over_raw_supervision as fixtures
        from dataclasses import replace
        helper = fixtures.FixedWorkJobWiringTests(methodName="runTest")
        base = helper.job()
        # Re-enter the same mocked public admission used by the existing job
        # wiring fixture; no catalog or source environment is instantiated.
        request = SimpleNamespace(state=base.context.public_materialization_state,
            observation=base.context.observation, set_source_hash=BINDING)
        with patch("pokezero.mcts_eval.head_to_head.public_only_context", side_effect=lambda c:c), \
             patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture", return_value=request), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=INPUT_BINDING):
            job = replace(base, native_ancestry_contract=CONTRACT)
        adapter = Mock(runtime_sha256=BINDING)
        adapter.select.return_value = dict(runtime_sha256=BINDING, evidence=selection_inputs("incumbent"))
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))
        with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
             patch("pokezero.mcts_eval.search_over_raw_world_inputs.WorldInputFixedWorkAdapter", return_value=adapter) as factory:
            job({}, lambda name:None)
        self.assertIs(factory.call_args.kwargs["native_ancestry_contract"], CONTRACT)

    def test_preexisting_private_session_is_never_closed_by_rejected_setup(self):
        from pokezero.mcts_eval.search_over_raw_world_inputs import WorldInputFixedWorkAdapter
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        adapter = object.__new__(WorldInputFixedWorkAdapter)
        adapter.configuration = SearchConfiguration("incumbent", seconds=1.)
        adapter._ancestry_contract, adapter.last_failure = CONTRACT, None
        adapter._timed_native_atomic_work = False  # This rejection fixture bypasses __init__.
        existing = Mock()
        model = lambda:"untouched-native"
        policy = SimpleNamespace(_value_diagnostic_session=existing, _native=model,
            _world_attempt_observer=None, _world_observer=None,
            _world_construction_observer=None, _native_construction_observer=None)
        adapter._native = policy
        with patch.object(PublicModelSearchAdapter, "_prepare_incumbent", return_value=None), \
             patch.object(PublicModelSearchAdapter, "select", side_effect=lambda *a,**k:adapter._prepare_incumbent(None)):
            with self.assertRaises(ValueError): adapter.select(None)
        existing.close.assert_not_called()
        self.assertIs(policy._value_diagnostic_session, existing)
        self.assertIs(policy._native, model)
        self.assertIsNone(policy._world_attempt_observer)
