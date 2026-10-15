"""Authored exports and disposable workers, never scientific source games."""
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest, rng_seed
from pokezero.mcts_eval.search_over_raw_fixed_work import FixedWorkDiagnosticContract
from pokezero.mcts_eval.search_over_raw_fixed_work_run import fixed_work_reader_source_hashes
from pokezero.mcts_eval.search_over_raw_native_fixed_values import (
    SCHEMA, HISTORY_LAW, NativeFixedWorkValueContract, NativeFixedWorkValueSession,
)
from pokezero.mcts_eval.search_over_raw_native_fixed_value_run import (
    RESULT_SCHEMA, NativeFixedWorkValueAdapter, NativeFixedWorkValueJob, NativeFixedWorkValueRunValidator,
    read_native_fixed_value_result, runtime_receipt, _ProviderPreparation,
    supervise_native_fixed_values,
)
from pokezero.mcts_eval.search_over_raw_resources import derive_fixed_work_accounting
from pokezero.mcts_eval.search_over_raw_supervision import _supervise_owned_job
from pokezero.mcts_eval.search_over_raw_world_inputs import NativeInputHandle, NativeWorldInputs
import test_search_over_raw_world_inputs as input_fixture
from test_search_over_raw_supervision import BINDING, INPUT_BINDING, selection_inputs

CONTRACT = NativeFixedWorkValueContract(17)
PUBLIC_SHA = "c"*64


def authored_result():
    cfg = SearchConfiguration("incumbent", seconds=1.)
    evidence = selection_inputs("incumbent")
    manifest = evidence["consumed_world_inputs"]
    manifest["public_request_observations"] = [dict(ordinal=i, witness=dict(
        schema="pokezero.search-over-raw.public-request-consumption.v1", public_request_sha256=PUBLIC_SHA,
        consumed_binding=INPUT_BINDING, full_supported_public_state_and_pending=True,
        private_source_snapshot_included=False, runtime_qualified=False)) for i in range(4)]
    call, = manifest["calls"]
    call["native_report"].update(model_evals=3, collision_self_side="side_one")
    call["native_report_utf8"] = json.dumps(call["native_report"])
    call["report_utf8_sha256"] = hashlib.sha256(call["native_report_utf8"].encode()).hexdigest()
    call["report_sha256"] = digest(call["native_report"])
    binding = {k:deepcopy(call[k]) for k in ("native_call_parameters", "construction_ordinals",
        "native_report", "native_report_utf8", "report_utf8_sha256")}
    leaves = []
    for priority, ordinal in sorted((rng_seed("visited-model-sample.v1", 17, 0,
            INPUT_BINDING["information_key"], i), i) for i in range(2)):
        prediction = .25 if ordinal == 0 else .75
        from pokezero.mcts_eval.search_over_raw_native_atomic_work import _bits
        leaves.append(dict(evaluation_ordinal=ordinal, priority=priority, native_state_sha256=BINDING,
            snapshot_sha256=BINDING, model_signed_value=2*prediction-1,
            model_self_value01_bits=_bits(prediction), labels=[dict(replicate=r, status="CAPPED_UNCERTAIN",
                signed_outcome=None, boundaries=250, continuation_policy="raw_argmax_both_seats",
                seed=rng_seed("native-visited-label.v1", 17, 0, INPUT_BINDING["information_key"], ordinal, r))
                for r in range(8)]))
    export = dict(schema=SCHEMA, worker=0, root_information_key=INPUT_BINDING["information_key"], sample_seed=17,
        root_native_state_sha256=call["native_call_parameters"]["state_sha256"],
        selection_report_sha256=call["report_utf8_sha256"], leaves_per_native_invocation=2,
        replicates=8, maximum_boundaries=250, model_evaluations_seen=2,
        sampling_unit="successful_model_evaluation_event", sampling="smallest_seeded_priorities_per_native_invocation",
        value_frame="self_relative", retained_payload_bytes=1, maximum_payload_bytes=8388608,
        payload_cap_is_not_rss_qualification=True, leaves=leaves, label_lossy_subcases={},
        label_engine="poke_engine_unqualified_against_showdown", instrumentation_can_change_selection=True,
        search_decision_ceiling=None, label_deadline=None, external_emergency_watchdog_required=True,
        native_actor_history_law=HISTORY_LAW, model_self_side_one=True)
    for key in ("searched_line_showdown_fidelity", "scientific_calibration_evidence", "scientific_strength_evidence",
        "labels_used_for_backup", "private_frontiers_exported", "independent_showdown_labels",
        "deployment_eligible", "resource_cost_complete"):
        export[key] = False
    from pokezero.mcts_eval.search_over_raw_native_fixed_values import read_fixed_native_values
    raw = json.dumps(export)
    readout = read_fixed_native_values(raw, binding=binding, contract=CONTRACT,
        root_key=INPUT_BINDING["information_key"], worker=0)
    selected = dict(root_id="same-root:"+cfg.identity, statistics_root_id="same-root",
        configuration_sha256=cfg.identity, selection_seed=7, runtime_sha256=BINDING,
        status="SELECTED", scientific_strength_evidence=False, elapsed_seconds=0.,
        fixed_work_diagnostic=dict(contract=FixedWorkDiagnosticContract().receipt(), accepted_actual=64), evidence=evidence)
    return dict(schema=RESULT_SCHEMA, selected=selected, native_values=dict(schema=SCHEMA,
        selected_receipt_sha256=digest(selected), native_invocations=[dict(invocation=0,
            binding=binding, raw_native_export=raw, readout=readout)]),
        label_wall_seconds=0., runtime_scope=runtime_receipt(CONTRACT))


@dataclass(frozen=True)
class AuthoredValueJob:
    fail: bool = False
    hang: bool = False
    synthetic_process_fixture = True

    def __call__(self, ownership, phase):
        phase("startup")
        phase("selection")
        if self.hang: time.sleep(30)
        if self.fail: raise ValueError("authored post-selection label failure")
        result = authored_result()
        phase("cleanup")
        return result


@dataclass
class AuthoredBelief:
    set_source: object = None
    def clone(self): return AuthoredBelief()


@dataclass(frozen=True)
class AuthoredPublicState:
    belief_engine: object


@dataclass(frozen=True)
class AuthoredContext:
    player_id: str
    observation: object
    requested_observations: dict
    public_materialization_state: object


@dataclass(frozen=True)
class AuthoredObservation:
    legal_action_mask: tuple = (True, True)


@dataclass(frozen=True)
class AuthoredCheckpoint:
    checkpoint_path: str = "authored-unused-checkpoint"
    checkpoint_sha256: str = BINDING
    showdown_source_sha256: str = BINDING
    feature_masks: dict = field(default_factory=lambda:{"tier2_residuals":False})


class NativeValueJobWiringTests(unittest.TestCase):
    def job(self):
        cfg = SearchConfiguration("incumbent", seconds=1.)
        state = AuthoredPublicState(AuthoredBelief())
        obs = AuthoredObservation()
        context = AuthoredContext("p1", obs, {"p1":obs, "p2":"excluded"}, state)
        request = SimpleNamespace(state=state, observation=obs, set_source_hash=BINDING)
        with patch("pokezero.mcts_eval.head_to_head.public_only_context", side_effect=lambda c:c), \
            patch("pokezero.mcts_eval.paper_reference_runtime.PublicRootRequest.capture", return_value=request), \
            patch("pokezero.mcts_eval.search_over_raw_world_inputs._binding", return_value=INPUT_BINDING), \
            patch("pokezero.mcts_eval.search_over_raw_world_inputs.public_request_witness", return_value={"public_request_sha256":PUBLIC_SHA}):
            return NativeFixedWorkValueJob(cfg, AuthoredCheckpoint(), "authored-unused-showdown", context,
                "same-root:"+cfg.identity, 7, "same-root", BINDING,
                expected_public_request_sha256=PUBLIC_SHA, value_contract=CONTRACT)

    def test_actual_job_method_labels_before_cleanup_and_transports_no_source_or_p2_observation(self):
        job = self.job()
        result = authored_result()
        adapter = Mock(runtime_sha256=BINDING)
        adapter.select.return_value = result["selected"]
        adapter.finish_native_fixed_values.return_value = result
        phases = []
        source = SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))
        with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=source), \
            patch("pokezero.mcts_eval.search_over_raw_native_fixed_value_run.NativeFixedWorkValueAdapter", return_value=adapter) as make:
            returned = job({}, phases.append)
        self.assertIs(returned, result)
        self.assertEqual(phases, ["startup", "selection", "cleanup"])
        self.assertIs(make.call_args.kwargs["value_contract"], CONTRACT)
        self.assertEqual(make.call_args.kwargs["expected_public_request_sha256"], PUBLIC_SHA)
        actor = adapter.select.call_args.args[0]
        self.assertEqual(set(actor.requested_observations), {"p1"})
        self.assertIs(actor.public_materialization_state.belief_engine.set_source, source)
        self.assertIsNone(job.context.public_materialization_state.belief_engine.set_source)
        adapter.finish_native_fixed_values.assert_called_once_with(result["selected"])
        adapter.close.assert_called_once()
        # Pickle reconstructs the immutable primitive fields without adopting
        # them as new mutable references; contracts/context preserve aliases.
        pickle.loads(pickle.dumps(job))._check()

    def test_job_mutation_during_labels_or_cleanup_refuses_and_still_closes(self):
        for stage in ("labels", "cleanup"):
            job = self.job()
            result = authored_result()
            adapter = Mock(runtime_sha256=BINDING)
            adapter.select.return_value = result["selected"]
            def labels(_selected):
                if stage == "labels": object.__setattr__(job, "value_contract", replace(CONTRACT))
                return result
            adapter.finish_native_fixed_values.side_effect = labels
            def phase(name):
                if stage == "cleanup" and name == "cleanup":
                    object.__setattr__(job, "_consumed_binding", deepcopy(job._consumed_binding))
            with patch("pokezero.randbat.load_gen3_randbat_source_cached", return_value=SimpleNamespace(metadata=SimpleNamespace(source_hash=BINDING))), \
                patch("pokezero.mcts_eval.search_over_raw_native_fixed_value_run.NativeFixedWorkValueAdapter", return_value=adapter):
                with self.subTest(stage=stage), self.assertRaises(ValueError): job({}, phase)
            adapter.close.assert_called_once()

    def test_supervisor_rechecks_job_after_result_callbacks_and_final_return(self):
        for stage in ("validator", "return"):
            job = self.job()
            def supervise(worker, **kwargs):
                self.assertIs(worker, job)
                kwargs["verify_admission"]()
                if stage == "validator":
                    original = derive_fixed_work_accounting
                    def mutate(*a, **k):
                        accounting = original(*a, **k)
                        object.__setattr__(job, "value_contract", replace(CONTRACT))
                        return accounting
                    with patch("pokezero.mcts_eval.search_over_raw_resources.derive_fixed_work_accounting", side_effect=mutate):
                        kwargs["validate_result"](authored_result())
                else:
                    object.__setattr__(job, "value_contract", replace(CONTRACT))
                return {}
            with patch("pokezero.mcts_eval.search_over_raw_native_fixed_value_run._supervise_owned_job", side_effect=supervise):
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    supervise_native_fixed_values(job, directory="unused", binding_sha256=BINDING, verify_admission=lambda:None)


class NativeValueRunTests(unittest.TestCase):
    def test_all_original_slots_and_child_readouts_are_reconstructed(self):
        result = authored_result()
        checked = read_native_fixed_value_result(result, contract=CONTRACT, root_key=INPUT_BINDING["information_key"])
        self.assertEqual((checked["required_labels"], checked["unknown_labels"]), (16, 16))
        for kind in ("drop", "readout", "alias", "selected", "scope", "number", "worker"):
            changed = deepcopy(result)
            row = changed["native_values"]["native_invocations"][0]
            if kind == "drop": changed["native_values"]["native_invocations"].clear()
            if kind == "readout": row["readout"]["unknown_labels"] = 0
            if kind == "alias": row["binding"]["construction_ordinals"] = [0]
            if kind == "selected": changed["selected"]["selection_seed"] = 8
            if kind == "scope": changed["runtime_scope"]["independent_showdown_labels"] = True
            if kind == "number": changed["label_wall_seconds"] = True
            if kind == "worker": row["invocation"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                read_native_fixed_value_result(changed, contract=CONTRACT, root_key=INPUT_BINDING["information_key"])

    def test_failed_dispatch_or_competing_bank_never_reaches_ffi(self):
        for kind in ("dispatch", "bank", "allocation"):
            with self.subTest(kind=kind):
                recorder = NativeWorldInputs(INPUT_BINDING, 16, deadline_limited=False, record_atomic_work=True)
                if kind != "dispatch": input_fixture.construct(recorder)
                session = NativeFixedWorkValueSession(CONTRACT, root_key=INPUT_BINDING["information_key"],
                    native_module=SimpleNamespace(NativeVisitedValueBank=lambda **kw:SimpleNamespace(close=Mock())))
                base = SimpleNamespace(search_batched_multi_encoded=Mock())
                kwargs = dict(visited_value_bank="competing") if kind == "bank" else {}
                args = input_fixture.arguments()
                if kind == "allocation": args[2] = 8
                with self.assertRaises(ValueError):
                    NativeInputHandle(base, recorder, value_session=session).search_batched_multi_encoded(*args, **kwargs)
                base.search_batched_multi_encoded.assert_not_called()
                self.assertTrue(session.closed)

    def test_incompatible_adapter_scope_rejects_before_runtime(self):
        for cfg, kwargs in ((SearchConfiguration("reference", workers=20), {}),
            (SearchConfiguration("incumbent", leaf="hp_fraction"), {}),
            (SearchConfiguration("incumbent"), dict(native_ancestry_contract=object())),
            (SearchConfiguration("incumbent"), dict(value_diagnostics=object())),
            (SearchConfiguration("incumbent"), dict(capture_native_public_fold=True))):
            with self.subTest(cfg=cfg, kwargs=kwargs), self.assertRaises(ValueError):
                NativeFixedWorkValueAdapter(cfg, value_contract=CONTRACT,
                    expected_public_request_sha256=PUBLIC_SHA, **kwargs)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / "authored-not-a-model"
        self.checkpoint.write_bytes(b"authored-unused-model")
        self.showdown = self.root / "authored-not-a-simulator"
        dist = self.showdown / "dist/sim"
        dist.mkdir(parents=True)
        for name in ("index.js", "state.js"): (dist / name).write_text("// authored unused simulator")
        self.files = fixed_work_reader_source_hashes()
        for path in (self.checkpoint, Path(sys.executable).resolve(), *dist.glob("*.js")):
            self.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def run_owned(self, job, *, short=None):
        directory = self.root / ("failed" if job.fail else "hung" if job.hang else "complete")
        terminal = _supervise_owned_job(job, directory=directory, binding_sha256=BINDING,
            verify_admission=lambda:None, contract=FixedWorkDiagnosticContract(), kind="fixed-work", work_phase="selection",
            validate_result=lambda result: derive_fixed_work_accounting(result["selected"]["evidence"],
                arm="incumbent", binding=INPUT_BINDING, statistics_root_id="same-root", leaf="model",
                expected_public_request_sha256=PUBLIC_SHA), _synthetic_watchdog_seconds=short)
        cfg = SearchConfiguration("incumbent", seconds=1.)
        inputs = dict(directory=directory, job_sha256=hashlib.sha256(pickle.dumps(job)).hexdigest(),
            binding_sha256=BINDING, configuration=cfg, root_id="same-root:"+cfg.identity,
            statistics_root_id="same-root", selection_seed=7, runtime_sha256=BINDING,
            consumed_binding=INPUT_BINDING, file_hashes=self.files,
            checkpoint_path=str(self.checkpoint), checkpoint_sha256=self.files[str(self.checkpoint)],
            showdown_root=str(self.showdown), public_request_sha256=PUBLIC_SHA, value_contract=CONTRACT)
        return terminal, inputs

    def test_normal_owned_probe_is_independent_and_synthetic_nonclearing(self):
        terminal, inputs = self.run_owned(AuthoredValueJob())
        self.assertEqual(terminal["supervisor_exit_code"], 0, terminal)
        result = terminal["result"]
        ordinary = NativeFixedWorkValueRunValidator(**inputs)
        self.assertIsNone(ordinary(result))
        reader = NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)
        self.assertEqual(reader(result), digest(result))
        checked = reader.last_validation
        self.assertTrue(checked["process_receipts_validated"])
        self.assertEqual(checked["native_values"]["unknown_labels"], 16)
        self.assertFalse(checked["resource_cost_complete"])
        self.assertFalse(checked["scientific_calibration_evidence"])
        from pokezero.mcts_eval.search_over_raw_fixed_work_run import FixedWorkRunValidator
        legacy_inputs = {k:v for k,v in inputs.items() if k != "value_contract"}
        with self.assertRaises((KeyError, ValueError)): FixedWorkRunValidator(**legacy_inputs)(result)

    def test_label_failure_and_external_cap_are_not_successful_or_retriable(self):
        for job, short in ((AuthoredValueJob(fail=True), None), (AuthoredValueJob(hang=True), .4)):
            terminal, inputs = self.run_owned(job, short=short)
            self.assertNotEqual(terminal["supervisor_exit_code"], 0)
            self.assertFalse(terminal["resource_cost_complete"])
            self.assertIsNone(terminal["result"])
            with self.assertRaises((ValueError, TypeError, KeyError)):
                NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)({})
            with self.assertRaises((ValueError, FileExistsError)): self.run_owned(job, short=short)

    def test_dispatch_callback_cannot_disable_capture_or_redirect_original_ffi(self):
        for kind in ("session", "base", "recorder"):
            recorder = NativeWorldInputs(INPUT_BINDING, 16, deadline_limited=False, record_atomic_work=True)
            input_fixture.construct(recorder)
            session = NativeFixedWorkValueSession(CONTRACT, root_key=INPUT_BINDING["information_key"],
                native_module=SimpleNamespace(NativeVisitedValueBank=Mock()))
            base = SimpleNamespace(search_batched_multi_encoded=Mock())
            handle = NativeInputHandle(base, recorder, value_session=session)
            dispatch = recorder.dispatch
            def mutate(*args):
                witness = dispatch(*args)
                if kind == "session": handle.value_session = None
                if kind == "base": handle.base = SimpleNamespace(search_batched_multi_encoded=Mock())
                if kind == "recorder": handle.recorder = None
                return witness
            recorder.dispatch = mutate
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                handle.search_batched_multi_encoded(*input_fixture.arguments())
            self.assertTrue(session.closed)
            base.search_batched_multi_encoded.assert_not_called()

    def test_mutation_after_base_resource_read_cannot_change_labels_or_receipts(self):
        terminal, inputs = self.run_owned(AuthoredValueJob())
        from pokezero.mcts_eval.search_over_raw_native_fixed_value_run import read_native_fixed_value_result as original
        for kind in ("result", "receipt", "contract"):
            reader = NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)
            result = deepcopy(terminal["result"])
            path = inputs["directory"] / "worker-result.json"
            old = path.read_bytes()
            def mutate(*args, **kwargs):
                checked = original(*args, **kwargs)
                if kind == "result": result["native_values"]["native_invocations"][0]["readout"]["unknown_labels"] = 0
                if kind == "receipt": path.write_bytes(old+b" ")
                if kind == "contract": reader.value_contract = replace(CONTRACT)
                return checked
            with patch("pokezero.mcts_eval.search_over_raw_native_fixed_value_run.read_native_fixed_value_result", side_effect=mutate):
                with self.subTest(kind=kind), self.assertRaises(ValueError): reader(result)
            self.assertIsNone(reader.last_validation)
            path.write_bytes(old)

    def test_original_reader_identity_and_cost_bound_survive_final_file_callback(self):
        terminal, inputs = self.run_owned(AuthoredValueJob())
        result = terminal["result"]
        for kind in ("synthetic", "directory", "binding", "contract", "configuration"):
            reader = NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)
            files = reader._files
            calls = []
            def change():
                files()
                calls.append(True)
                # Two base calls precede the distinct final native readback.
                if len(calls) == 3:
                    if kind == "synthetic": reader.allow_synthetic = False
                    if kind == "directory": reader.directory = Path(str(reader.directory))
                    if kind == "binding": reader.binding = "f"*64
                    if kind == "contract": reader.contract = FixedWorkDiagnosticContract()
                    if kind == "configuration": reader.configuration = replace(reader.configuration)
            with patch.object(NativeFixedWorkValueRunValidator, "_files", lambda owner:change()):
                with self.subTest(kind=kind), self.assertRaises(ValueError): reader(result)
            self.assertEqual(len(calls), 3)
            self.assertIsNone(reader.last_validation)
        reader = NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)
        reader(result)
        reader.allow_synthetic = False
        with self.assertRaises(ValueError): reader(result)
        self.assertIsNone(reader.last_validation)
        changed = deepcopy(result)
        changed["label_wall_seconds"] = 1801.
        for name in ("worker-result.json", "terminal.json"):
            path = inputs["directory"] / name
            payload = json.loads(path.read_bytes()); payload["result"] = changed
            path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "containing work phase"):
            NativeFixedWorkValueRunValidator(**inputs, allow_synthetic=True)(changed)


if __name__ == "__main__":
    unittest.main()
