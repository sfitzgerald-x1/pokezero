"""Distinct supervised native value probe; no collection or Showdown truth.

The work phase includes selection and post-selection labels. Wait4 cost covers
both and cleanup; sampled RSS and controller/source work remain incomplete.
Historical AFTER, ancestry and timed-label laws are unchanged.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
import math
import hashlib
import time

from .search_over_raw import digest, require
from .search_over_raw_native_fixed_values import (
    SCHEMA, HISTORY_LAW, NativeFixedWorkValueContract, NativeFixedWorkValueSession,
    read_fixed_native_values,
    value_schema, EMPTY_DELTA_LAW,
)
from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
from .search_over_raw_fixed_work_run import FixedWorkRunValidator
from .search_over_raw_supervision import FixedWorkAdapterJob, _supervise_owned_job
from .search_over_raw_world_inputs import (
    NativeInputHandle, WorldInputFixedWorkAdapter, _state_payload, validate_fixed_work_inputs,
)
from .search_over_raw_annotation_boundary import ActorAnnotationBoundary

RESULT_SCHEMA = "pokezero.search-over-raw.owned-native-fixed-values.v1"


def runtime_receipt(contract):
    require(type(contract) is NativeFixedWorkValueContract, "prospective native value contract required")
    receipt = dict(schema=value_schema(contract), contract=asdict(contract), native_actor_history_law=HISTORY_LAW,
        labels_after_selection=True, labels_used_for_backup=False, decision_deadline=None,
        actor_annotation_handoff="explicit-original-boundary-or-checkpoint-inactive",
        label_deadline=None, external_watchdog_seconds=1800,
        work_phase="selection_including_post_selection_native_labels",
        independent_showdown_labels=False, scientific_calibration_evidence=False,
        resource_cost_complete=False, runtime_qualified=False, deployment_eligible=False)
    if contract.certified_empty_delta_progress:
        receipt["empty_delta_progress_law"] = EMPTY_DELTA_LAW
    return receipt


def native_fixed_value_runtime_recipe(configuration, checkpoint_contract, *, value_contract):
    """Pure prospective runtime assembly, no adapter or model initialization.

    This does not add this job to historical six-recipe manifests. The caller
    must separately review its complete inputs and new exploration entry.
    """
    from .search_over_raw_diagnostic_producer import fixed_work_runtime_recipe
    return fixed_work_runtime_recipe(configuration, checkpoint_contract,
        native_fixed_value_contract=value_contract)


class _ProviderPreparation:
    """Private sentinel: prepare original providers, never allocate a second bank."""
    def __init__(self, session):
        self.session = session
        self.original = session

    def begin(self, record):
        require(self.session is self.original and not self.session.closed,
            "original native provider sentinel changed")
        self.session._check()
        return None

    def close(self):
        self.original.close()

    def __reduce_ex__(self, protocol):
        raise TypeError("private native provider sentinel cannot enter transport")


def _binding_roster(bindings, calls):
    require(type(bindings) is list and type(calls) is list and len(bindings) == len(calls)
        and 1 <= len(calls) <= 4, "all original native calls required")
    covered = []
    for binding, call in zip(bindings, calls):
        require(type(binding) is dict and set(binding) == {"native_call_parameters", "construction_ordinals",
            "native_report", "native_report_utf8", "report_utf8_sha256"}
            and digest(binding) == digest({k:call[k] for k in binding}),
            "native value binding differs from original observed selection call")
        covered.extend(binding["construction_ordinals"])
    require(sorted(covered) == list(range(4))
        and sum(row["native_report"]["iterations"] for row in bindings) == 64,
        "native value probe requires all four original construction slots and accepted allocation")


def read_native_fixed_value_result(result, *, contract, root_key):
    """Rebuild every child readout from its raw export and original call roster."""
    require(type(result) is dict and type(contract) is NativeFixedWorkValueContract,
        "original native value result required")
    saved = deepcopy(result)
    selected = saved["selected"]
    values = saved["native_values"]
    require(set(saved) == {"schema", "selected", "native_values", "label_wall_seconds", "runtime_scope"}
        and saved["schema"] == RESULT_SCHEMA and digest(saved["runtime_scope"]) == digest(runtime_receipt(contract))
        and type(saved["label_wall_seconds"]) in (int, float)
        and math.isfinite(saved["label_wall_seconds"]) and saved["label_wall_seconds"] >= 0
        and type(values) is dict and set(values) == {"schema", "selected_receipt_sha256", "native_invocations"}
        and values["schema"] == value_schema(contract) and values["selected_receipt_sha256"] == digest(selected)
        and selected["status"] == "SELECTED" and selected["scientific_strength_evidence"] is False
        and type(selected["elapsed_seconds"]) in (int, float)
        and math.isfinite(selected["elapsed_seconds"]) and selected["elapsed_seconds"] >= 0,
        "native value result scope/selected binding drift")
    manifest = selected["evidence"]["consumed_world_inputs"]
    require(manifest["information_key"] == root_key, "native value original information root drift")
    rows = values["native_invocations"]
    require(type(rows) is list and all(type(row) is dict and set(row) == {
        "invocation", "binding", "raw_native_export", "readout"} for row in rows),
        "complete original native value invocation exports required")
    _binding_roster([row["binding"] for row in rows], manifest["calls"])
    rebuilt = []
    for i, row in enumerate(rows):
        require(type(row["invocation"]) is int and row["invocation"] == i, "native value invocation order drift")
        readout = read_fixed_native_values(row["raw_native_export"], binding=row["binding"],
            contract=contract, root_key=root_key, worker=i)
        require(digest(readout) == digest(row["readout"]), "child native value summary differs from rebuilt export")
        rebuilt.append(readout)
    require(digest(result) == digest(saved), "native value return changed during reconstruction")
    return dict(schema=RESULT_SCHEMA, selected_sha256=digest(selected), invocations=rebuilt,
        actual_model_evaluations=sum(row["actual_model_evaluations"] for row in rebuilt),
        required_labels=sum(row["required_labels"] for row in rebuilt),
        unknown_labels=sum(row["unknown_labels"] for row in rebuilt),
        independent_showdown_labels=False, scientific_calibration_evidence=False,
        runtime_qualified=False, resource_cost_complete=False, owning_process_validated=False)


class NativeFixedWorkValueAdapter(WorldInputFixedWorkAdapter):
    """Separate runtime; actual dispatch owns capture, selection precedes labels."""
    def __init__(self, configuration, *, value_contract, annotation_boundary=None, **kwargs):
        require(type(value_contract) is NativeFixedWorkValueContract and configuration.arm == "incumbent"
            and configuration.leaf == "model" and configuration.belief == "public"
            and configuration.seconds == 1. and configuration.workers == 1
            and kwargs.get("expected_public_request_sha256") is not None
            and all(kwargs.get(k) is None for k in ("native_ancestry_contract", "value_diagnostics",
                "reference_factory", "policy_opponent_diagnostics"))
            and not kwargs.get("capture_native_public_fold", False),
            "native value probe needs its own public fixed-model/full-request runtime")
        self.value_contract = value_contract
        require(annotation_boundary is None or type(annotation_boundary) is ActorAnnotationBoundary,
            "explicit typed actor annotation handoff required")
        self.annotation_boundary = annotation_boundary
        self._annotation_original = annotation_boundary
        self._annotation_restore = None
        self._value_contract_original = (value_contract, digest(asdict(value_contract)))
        self._native_values_pending = None
        self._fixed_value_session = self._provider_sentinel = None
        super().__init__(configuration, **kwargs)
        self.runtime_configuration["native_fixed_values"] = runtime_receipt(value_contract)
        self.runtime_sha256 = digest(self.runtime_configuration)
        self._fixed_runtime_original = (self.runtime_configuration, digest(self.runtime_configuration), self.runtime_sha256)

    def _value_inputs(self):
        contract, sha = self._value_contract_original
        runtime, runtime_sha, sha_value = self._fixed_runtime_original
        require(self.annotation_boundary is self._annotation_original,
            "original annotation handoff replaced")
        if self.annotation_boundary is not None:
            self.annotation_boundary._check()
        require(self.value_contract is contract and digest(asdict(contract)) == sha
            and self.runtime_configuration is runtime and digest(runtime) == runtime_sha
            and self.runtime_sha256 == sha_value, "original native value adapter contract/runtime changed")

    def _prepare_incumbent(self, request):
        self._value_inputs()
        boundary = self.annotation_boundary
        require(boundary is not None or not self.contract.feature_masks["tier2_residuals"],
            "active residual annotations require original actor boundary handoff")
        if boundary is not None:
            boundary.validate(request, self.contract)
            self._annotation_restore = self._native._annotation_source
            self._native._annotation_source = boundary
        require(self._native._value_diagnostic_session is None,
            "existing native diagnostic session cannot be replaced by value probe")
        super()._prepare_incumbent(request)
        self._native.prepare_visited_value_runtime()
        session = NativeFixedWorkValueSession(self.value_contract,
            root_key=self._input_recorder.binding["information_key"])
        sentinel = _ProviderPreparation(session)
        self._fixed_value_session, self._provider_sentinel = session, sentinel
        self._value_bridge_original = (session, sentinel)
        self._native._value_diagnostic_session = sentinel
        original = self._input_restore[1]
        recorder = self._input_recorder
        self._native._native = lambda: NativeInputHandle(original(), recorder, value_session=session)

    def select(self, context, **kwargs):
        require(self._native_values_pending is None and self._fixed_value_session is None,
            "native value adapter is single-decision; no overwrite or retry")
        self._value_inputs()
        try:
            selected = super().select(context, **kwargs)
            require(self.annotation_boundary is None or
                self._native._annotation_source is self.annotation_boundary,
                "original actor annotation source changed during native selection")
            session = self._fixed_value_session
            require(type(session) is NativeFixedWorkValueSession
                and session is self._value_bridge_original[0]
                and self._provider_sentinel is self._value_bridge_original[1]
                and self._native._value_diagnostic_session is self._provider_sentinel
                and self._provider_sentinel.session is session,
                "original native provider/session handoff changed")
            _binding_roster(session.bindings(), selected["evidence"]["consumed_world_inputs"]["calls"])
            self._value_inputs()
            self._native_values_pending = (selected, digest(selected), session, self._input_recorder)
            return selected
        except BaseException:
            self._poisoned = True
            if self._fixed_value_session is not None:
                self._fixed_value_session.close()
            raise
        finally:
            if self._provider_sentinel is not None:
                self._native._value_diagnostic_session = None
            if self._annotation_restore is not None:
                self._native._annotation_source = self._annotation_restore
                self._annotation_restore = None

    def close(self):
        if self._fixed_value_session is not None:
            self._fixed_value_session.close()
        self._native_values_pending = None
        super().close()

    def finish_native_fixed_values(self, selected):
        require(not self._closed and not self._poisoned and self._native_values_pending is not None,
            "no original selected native value attempt")
        original, sha, session, recorder = self._native_values_pending
        self._native_values_pending = None
        started = time.perf_counter()
        try:
            self._value_inputs()
            require(selected is original and digest(selected) == sha and session is self._fixed_value_session
                and recorder is self._input_recorder, "original selected native probe changed")
            call_sha = digest(recorder.calls)
            values = session.label()
            self._value_inputs()
            require(digest(selected) == sha and digest(recorder.calls) == call_sha
                and session is self._fixed_value_session and recorder is self._input_recorder,
                "original native selection/call roster changed during labels")
            result = dict(schema=RESULT_SCHEMA, selected=deepcopy(selected),
                native_values=dict(schema=value_schema(self.value_contract), selected_receipt_sha256=sha, native_invocations=values),
                label_wall_seconds=time.perf_counter()-started, runtime_scope=runtime_receipt(self.value_contract))
            read_native_fixed_value_result(result, contract=self.value_contract, root_key=recorder.binding["information_key"])
            return result
        except BaseException:
            self._poisoned = True
            session.close()
            raise


@dataclass(frozen=True)
class NativeFixedWorkValueJob(FixedWorkAdapterJob):
    value_contract: NativeFixedWorkValueContract = field(kw_only=True)
    annotation_boundary: ActorAnnotationBoundary | None = field(default=None, kw_only=True)
    synthetic_process_fixture: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        require(type(self.synthetic_process_fixture) is bool
            and type(self.value_contract) is NativeFixedWorkValueContract
            and self.configuration.arm == "incumbent" and self.configuration.leaf == "model"
            and self.native_ancestry_contract is None and self.reference_factory is None
            and self.capture_native_public_fold is False and self.expected_public_request_sha256 is not None,
            "separate native value owning job scope required")
        super().__post_init__()
        require(self.annotation_boundary is not None or
            not self.checkpoint_contract.feature_masks["tier2_residuals"],
            "owning job needs original active actor annotations")
        if self.annotation_boundary is not None:
            require(type(self.annotation_boundary) is ActorAnnotationBoundary,
                "original typed job annotation boundary required")
            from .paper_reference_runtime import PublicRootRequest
            self.annotation_boundary.validate(PublicRootRequest(self.context.public_materialization_state,
                self.context.observation, self._set_source_hash, self.pending_transition), self.checkpoint_contract)
        object.__setattr__(self, "_value_binding_original", self._consumed_binding)
        object.__setattr__(self, "_value_job_refs", tuple(getattr(self, f.name) for f in fields(self)))
        object.__setattr__(self, "_value_job_sha", self._value_inputs())

    def _value_inputs(self):
        return digest(dict(fields={f.name:_state_payload(getattr(self, f.name)) for f in fields(self)},
            consumed=self._consumed_binding, set_source_hash=self._set_source_hash))

    def _check(self):
        require(all(type(getattr(self, f.name)) is type(original)
            and (getattr(self, f.name) == original if type(original) in (str, int, float, bool, type(None))
                else getattr(self, f.name) is original) for f, original in zip(fields(self), self._value_job_refs))
            and self._consumed_binding is self._value_binding_original
            and self._value_inputs() == self._value_job_sha, "original native value job changed")

    def __call__(self, ownership, phase):
        adapter = None
        phase("startup")
        try:
            self._check()
            from ..randbat import load_gen3_randbat_source_cached
            source = load_gen3_randbat_source_cached(self.showdown_root)
            require(source.metadata.source_hash == self._set_source_hash, "native value original catalog changed")
            state = self.context.public_materialization_state
            engine = state.belief_engine.clone()
            engine.set_source = source
            context = replace(self.context, public_materialization_state=replace(state, belief_engine=engine))
            adapter = NativeFixedWorkValueAdapter(self.configuration, value_contract=self.value_contract,
                annotation_boundary=self.annotation_boundary,
                expected_public_request_sha256=self.expected_public_request_sha256,
                fixed_work=FixedWorkDiagnosticContract(), checkpoint_contract=self.checkpoint_contract,
                showdown_root=self.showdown_root)
            require(adapter.runtime_sha256 == self.expected_runtime_sha256, "native value reviewed runtime drift")
            phase("selection")  # Explicitly covers labels too, inside the same external watchdog.
            selected = adapter.select(context, root_id=self.root_id, selection_seed=self.selection_seed,
                statistics_root_id=self.statistics_root_id, pending_transition=self.pending_transition)
            validate_fixed_work_inputs(selected["evidence"], arm="incumbent", binding=self._consumed_binding,
                statistics_root_id=self.statistics_root_id, construction_required=True,
                expected_public_request_sha256=self.expected_public_request_sha256)
            result = adapter.finish_native_fixed_values(selected)
            self._check()
            return result
        finally:
            phase("cleanup")
            if adapter is not None:
                adapter.close()
            self._check()


def supervise_native_fixed_values(job, *, directory, binding_sha256, verify_admission,
        _synthetic_watchdog_seconds=None):
    """One immutable job; requires separately authorized caller admission."""
    require(type(job) is NativeFixedWorkValueJob, "original native value job required")
    frozen = job._value_inputs()
    def admission():
        job._check()
        require(job._value_inputs() == frozen, "original native value supervisor job drift")
        value = verify_admission()
        job._check()
        return value
    def validate_result(result):
        job._check()
        from .search_over_raw_resources import derive_fixed_work_accounting
        selected = result["selected"]
        require(selected["root_id"] == job.root_id and selected["statistics_root_id"] == job.statistics_root_id
            and selected["configuration_sha256"] == job.configuration.identity
            and type(selected["selection_seed"]) is int and selected["selection_seed"] == job.selection_seed
            and selected["runtime_sha256"] == job.expected_runtime_sha256,
            "original native value supervised selection drift")
        read_native_fixed_value_result(result, contract=job.value_contract,
            root_key=job._consumed_binding["information_key"])
        accounting = derive_fixed_work_accounting(selected["evidence"], arm="incumbent", binding=job._consumed_binding,
            statistics_root_id=job.statistics_root_id, leaf="model",
            expected_public_request_sha256=job.expected_public_request_sha256)
        job._check()
        return accounting
    terminal = _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=admission, contract=FixedWorkDiagnosticContract(), kind="fixed-work",
        work_phase="selection", validate_result=validate_result,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)
    job._check()
    return terminal


class NativeFixedWorkValueRunValidator(FixedWorkRunValidator):
    """Normal reap plus independently reconstructed native-only label roster."""
    def __init__(self, *, value_contract, **kwargs):
        require(type(value_contract) is NativeFixedWorkValueContract
            and kwargs["configuration"].arm == "incumbent" and kwargs["configuration"].leaf == "model"
            and kwargs.get("public_request_sha256") is not None
            and not kwargs.get("native_public_fold_required", False),
            "separate native value owning-reader scope required")
        self.value_contract = value_contract
        self._value_contract_original = value_contract
        super().__init__(**kwargs)
        self._reader_fields_original = {key:(value, type(value)) for key, value in vars(self).items()
            if key != "last_validation"}
        self._reader_values_original = digest(dict(directory=str(self.directory), job_sha=self.job_sha,
            binding=self.binding, configuration=asdict(self.configuration), root_id=self.root_id,
            statistics_root_id=self.statistics_root_id, seed=self.seed, runtime_sha=self.runtime_sha,
            synthetic=self.allow_synthetic, contract=asdict(self.contract), consumed=self.consumed,
            inputs=self._inputs()))

    def _original_reader(self):
        require(set(vars(self)) == set(self._reader_fields_original) | {
            "last_validation", "_reader_fields_original", "_reader_values_original"}
            and all(type(getattr(self, key)) is kind and
                (getattr(self, key) == value if kind in (str, int, float, bool, type(None))
                    else getattr(self, key) is value) for key, (value, kind) in self._reader_fields_original.items())
            and digest(dict(directory=str(self.directory), job_sha=self.job_sha, binding=self.binding,
                configuration=asdict(self.configuration), root_id=self.root_id,
                statistics_root_id=self.statistics_root_id, seed=self.seed, runtime_sha=self.runtime_sha,
                synthetic=self.allow_synthetic, contract=asdict(self.contract), consumed=self.consumed,
                inputs=self._inputs())) == self._reader_values_original,
            "original native value owning reader identity/scope changed")

    def _inputs(self):
        require(self.value_contract is self._value_contract_original,
            "original native value reader contract replaced")
        return digest(dict(base=super()._inputs(), value_contract=asdict(self.value_contract)))

    def _selected_result(self, result):
        return result["selected"]

    def __call__(self, result):
        self.last_validation = None
        self._original_reader()
        before = digest(result)
        self.last_validation = None
        accepted = super().__call__(result)
        ownership = deepcopy(self.last_validation)
        self.last_validation = None
        rebuilt = read_native_fixed_value_result(result, contract=self.value_contract,
            root_key=self.consumed["information_key"])
        self._files()
        self._original_reader()
        require(result["selected"]["elapsed_seconds"] + result["label_wall_seconds"] <=
            ownership["costs"]["phase_costs"]["selection_wall_seconds"],
            "native selection/label intervals exceed independently observed containing work phase")
        require(self._inputs() == self.frozen and digest(result) == before
            and all(hashlib.sha256((self.directory / name).read_bytes()).hexdigest() == sha
                for name, sha in ownership["receipt_file_sha256"].items())
            and hashlib.sha256((self.directory / "job.pickle").read_bytes()).hexdigest() == self.job_sha
            and not list((self.directory / "owned-groups/active").glob("group-*.json"))
            and not list((self.directory / "owned-groups").glob("group-*.json")),
            "native label or owning evidence changed during final reconstruction")
        self.last_validation = dict(schema=RESULT_SCHEMA, result_sha256=before,
            selection_owner=ownership, native_values=rebuilt,
            process_receipts_validated=True, labels_inside_external_watchdog=True,
            work_phase="selection_including_post_selection_native_labels",
            resource_cost_complete=False, runtime_qualified=False,
            scientific_calibration_evidence=False, independent_showdown_labels=False)
        return before if accepted is not None else None
