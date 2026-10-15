"""Fixed-work native A2 probe, separate from independent Showdown calibration.

Opaque original native private bridges define the continuation policy inputs.
No source-opponent Python history is accepted or reconstructed. This private
unit does not admit collection, provide an external watchdog, measure full
costs, qualify physics or turn native outcomes into Showdown ground truth.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import struct

from .search_over_raw import digest, require, rng_seed
from .search_over_raw_native_atomic_work import validate_native_report
from .search_over_raw_value_diagnostics import calibration_bounds

SCHEMA = "pokezero.native-visited-value.fixed-work-native.v2"
EMPTY_DELTA_SCHEMA = "pokezero.native-visited-value.fixed-work-native.v3"
EMPTY_DELTA_LAW = "attributed_native_two_actor_turn_no_redraw.v1"
HISTORY_LAW = "original_retained_sampled_native_private_bridges"


@dataclass(frozen=True)
class NativeFixedWorkValueContract:
    sample_seed: int
    leaves_per_invocation: int = 2
    maximum_payload_bytes: int = 8388608
    iterations_per_world: int = 16
    batch_size: int = 16
    maximum_invocations: int = 4
    replicates: int = 8
    maximum_boundaries: int = 250
    external_watchdog_seconds: int = 1800
    certified_empty_delta_progress: bool = False

    def __post_init__(self):
        require(type(self.certified_empty_delta_progress) is bool
            and all(type(v) is int for k, v in asdict(self).items() if k != "certified_empty_delta_progress")
            and 0 <= self.sample_seed < 2**64 and 1 <= self.leaves_per_invocation <= 8
            and 0 < self.maximum_payload_bytes <= 8388608
            and (self.iterations_per_world, self.batch_size, self.maximum_invocations,
                 self.replicates, self.maximum_boundaries, self.external_watchdog_seconds)
                == (16, 16, 4, 8, 250, 1800),
            "prospective fixed-work native value contract required")


def value_schema(contract):
    require(type(contract) is NativeFixedWorkValueContract
        and type(contract.certified_empty_delta_progress) is bool, "original native value contract required")
    return EMPTY_DELTA_SCHEMA if contract.certified_empty_delta_progress else SCHEMA


def _sha(text):
    require(type(text) is str, "original native UTF-8 bytes required")
    return hashlib.sha256(text.encode()).hexdigest()


def read_fixed_native_values(raw, *, binding, contract, root_key, worker):
    """Bind sanitized outcomes to the original actual atomic evaluation roster."""
    require(type(contract) is NativeFixedWorkValueContract and type(worker) is int
        and 0 <= worker < contract.maximum_invocations and type(raw) is str
        and len(raw.encode()) <= contract.maximum_payload_bytes and type(binding) is dict,
        "bounded native value export and original invocation binding required")
    def unique(pairs):
        require(len({k for k, _ in pairs}) == len(pairs), "duplicate native value fields")
        return dict(pairs)
    value = json.loads(raw, object_pairs_hook=unique)
    require(type(value) is dict, "typed native value export required")
    flags = {"searched_line_showdown_fidelity", "scientific_calibration_evidence",
        "scientific_strength_evidence", "labels_used_for_backup", "private_frontiers_exported",
        "independent_showdown_labels", "deployment_eligible", "resource_cost_complete"}
    fields = {"schema", "worker", "root_information_key", "sample_seed", "root_native_state_sha256",
        "selection_report_sha256", "leaves_per_native_invocation", "replicates", "maximum_boundaries",
        "model_evaluations_seen", "sampling_unit", "sampling", "value_frame", "retained_payload_bytes",
        "maximum_payload_bytes", "payload_cap_is_not_rss_qualification", "leaves", "label_lossy_subcases",
        "label_engine", "instrumentation_can_change_selection", "search_decision_ceiling", "label_deadline",
        "external_emergency_watchdog_required", "native_actor_history_law", "model_self_side_one"} | flags
    if contract.certified_empty_delta_progress:
        fields.add("empty_delta_progress_law")
        require(value.get("empty_delta_progress_law") == EMPTY_DELTA_LAW,
            "explicit native empty-delta chronological progress law required")
    require(type(value) is dict and set(value) == fields and value["schema"] == value_schema(contract)
        and all(value[k] is False for k in flags)
        and value["search_decision_ceiling"] is None and value["label_deadline"] is None
        and value["external_emergency_watchdog_required"] is True
        and value["native_actor_history_law"] == HISTORY_LAW
        and value["label_engine"] == "poke_engine_unqualified_against_showdown"
        and value["sampling_unit"] == "successful_model_evaluation_event"
        and value["sampling"] == "smallest_seeded_priorities_per_native_invocation"
        and value["value_frame"] == "self_relative"
        and value["payload_cap_is_not_rss_qualification"] is True
        and value["instrumentation_can_change_selection"] is True,
        "native fixed-work scope/schema drift; legacy timed labels are not upgraded")
    report = binding["native_report"]
    aliases = binding["construction_ordinals"]
    require(type(aliases) is list and 1 <= len(aliases) <= 4
        and all(type(v) is int and 0 <= v < 4 for v in aliases)
        and aliases == sorted(set(aliases)), "explicit original collapsed construction roster required")
    requested = contract.iterations_per_world * len(aliases)
    require(binding["native_call_parameters"]["requested_iterations"] == requested
        and binding["native_call_parameters"]["batch_size"] == contract.batch_size,
        "native fixed-work actual allocation differs from prospective contract")
    validate_native_report(report, **binding["native_call_parameters"])
    require(_sha(binding["native_report_utf8"]) == binding["report_utf8_sha256"]
        and digest(json.loads(binding["native_report_utf8"])) == digest(report)
        and value["selection_report_sha256"] == binding["report_utf8_sha256"]
        and value["root_native_state_sha256"] == binding["native_call_parameters"]["state_sha256"]
        and report["iterations"] == requested
        and report.get("time_budget_exhausted", False) is False,
        "native fixed-work original selection/atomic binding drift")
    for key, expected in dict(worker=worker, root_information_key=root_key, sample_seed=contract.sample_seed,
            leaves_per_native_invocation=contract.leaves_per_invocation, replicates=8, maximum_boundaries=250,
            maximum_payload_bytes=contract.maximum_payload_bytes).items():
        require(type(value[key]) is type(expected) and value[key] == expected,
            "native fixed-work original sampler/worker binding drift")
    n = value["model_evaluations_seen"]
    lossy = value["label_lossy_subcases"]
    require(type(lossy) is dict and all(type(key) is str and type(count) is int and count >= 0
        for key, count in lossy.items()), "native label lossy-subcase ledger drift")
    require(type(n) is int and n >= 0 and n == report["model_evals"] - 1
        and type(value["retained_payload_bytes"]) is int
        and 0 <= value["retained_payload_bytes"] <= contract.maximum_payload_bytes,
        "native actual evaluation denominator or private retention drift")
    side = report["collision_self_side"]
    require(side in {"side_one", "side_two"}
        and (value["model_self_side_one"] is (side == "side_one") if n else
            value["model_self_side_one"] is None)
        and report.get("model_leaf_override") in (None, "model_value"),
        "original model prediction subject/evaluator drift")
    events, offset = {}, 0
    for batch in report["atomic_work"]["rounds"]:
        require(batch["status"] == "COMMITTED", "native fixed-work value probe cannot adopt cancelled rows")
        for chance in batch["after_collection"]["chances"]:
            for branch in chance:
                local = branch["pending_row"]
                if local is not None:
                    events[offset + local] = batch["row_value_bits"][local]
        offset += len(batch["row_value_bits"])
    require(offset == n and sorted(events) == list(range(n)), "native actual model-row roster drift")
    f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
    expected = heapq.nsmallest(contract.leaves_per_invocation,
        ((rng_seed("visited-model-sample.v1", contract.sample_seed, worker, root_key, ordinal), ordinal)
            for ordinal in range(n)))
    leaves = value["leaves"]
    require(type(leaves) is list and len(leaves) == len(expected), "complete native sampled-event roster required")
    for leaf, (priority, ordinal) in zip(leaves, expected):
        require(type(leaf) is dict and set(leaf) == {"evaluation_ordinal", "priority", "native_state_sha256",
            "snapshot_sha256", "model_signed_value", "model_self_value01_bits", "labels"}
            and type(leaf["priority"]) is int and leaf["priority"] == priority
            and type(leaf["evaluation_ordinal"]) is int and leaf["evaluation_ordinal"] == ordinal,
            "native fixed-work actual sampled event drift")
        original = leaf["model_self_value01_bits"]
        require(type(original) is int and 0 <= original < 2**32, "actual original model float32 bits required")
        prediction = struct.unpack("<f", struct.pack("<I", original))[0]
        require(math.isfinite(prediction) and 0 <= prediction <= 1
            and type(leaf["model_signed_value"]) in (int, float)
            and leaf["model_signed_value"] == f32(f32(2 * prediction) - 1)
            and bits(prediction if side == "side_one" else f32(1 - prediction)) == events[ordinal],
            "original SELF prediction differs from actual frame-converted atomic price")
        for name in ("native_state_sha256", "snapshot_sha256"):
            sha = leaf[name]
            require(type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                "native private frontier digest required")
        require(type(leaf["labels"]) is list and len(leaf["labels"]) == 8,
            "all eight native continuation streams required")
        for rep, label in enumerate(leaf["labels"]):
            label_fields = {"replicate", "status", "signed_outcome", "boundaries", "continuation_policy", "seed"}
            if contract.certified_empty_delta_progress:
                label_fields.add("certified_empty_delta_boundaries")
            require(type(label) is dict and set(label) == label_fields
                and type(label["replicate"]) is int and label["replicate"] == rep
                and label["continuation_policy"] == "raw_argmax_both_seats"
                and label["status"] in {"COMPLETE", "CAPPED_UNCERTAIN"}
                and type(label["boundaries"]) is int and 0 <= label["boundaries"] <= 250
                and type(label["seed"]) is int and label["seed"] == rng_seed("native-visited-label.v1",
                    contract.sample_seed, worker, root_key, ordinal, rep),
                "native fixed-work continuation stream/cap law drift")
            if contract.certified_empty_delta_progress:
                require(type(label["certified_empty_delta_boundaries"]) is int
                    and 0 <= label["certified_empty_delta_boundaries"] <= label["boundaries"],
                    "native certified empty-delta counter drift")
            require((type(label["signed_outcome"]) in (int, float) and label["signed_outcome"] in (-1, 0, 1))
                if label["status"] == "COMPLETE" else
                label["signed_outcome"] is None and label["boundaries"] == 250,
                "native unresolved cap is not a scored outcome")
    digest(value)  # Reject nonfinite/uncanonical ancillary JSON too.
    bins = calibration_bounds(leaves)  # Keeps every unknown replicate in bounds.
    checked = dict(schema=value_schema(contract), native_export_sha256=_sha(raw), selection_report_sha256=binding["report_utf8_sha256"],
        actual_model_evaluations=n, retained_events=len(leaves), required_labels=8*len(leaves),
        completed_iterations=report["iterations"], construction_ordinals=deepcopy(aliases),
        construction_ownership_validated=False, aliases_counted_as_samples=False,
        label_lossy_subcases=deepcopy(lossy),
        unknown_labels=sum(row["unknown_labels"] for row in bins), calibration=bins,
        native_actor_history_law=HISTORY_LAW, label_engine="poke_engine_unqualified_against_showdown",
        independent_showdown_labels=False, scientific_calibration_evidence=False, mechanism_qualified=False,
        resource_cost_complete=False, runtime_qualified=False, scientific_collection_authorized=False,
        deployment_eligible=False, owning_process_validated=False)
    if contract.certified_empty_delta_progress:
        checked.update(empty_delta_progress_law=EMPTY_DELTA_LAW,
            certified_empty_delta_boundaries=sum(label["certified_empty_delta_boundaries"]
                for leaf in leaves for label in leaf["labels"]))
    return checked


class NativeFixedWorkValueSession:
    """Separate opt-in call controller, not yet a producer/adapter entry.

    External ownership/watchdog/cost validation is mandatory upstream. Binding
    these FFI outputs alone never proves it; historical timed sessions remain
    unchanged and cannot be silently promoted into this new law.
    """
    def __init__(self, contract, *, root_key, native_module=None):
        require(type(contract) is NativeFixedWorkValueContract and type(root_key) is str
            and bool(root_key) and len(root_key) % 2 == 0
            and all(c in "0123456789abcdef" for c in root_key), "explicit native fixed-work binding required")
        if native_module is None:
            import pokezero_search as native_module
        self.contract, self.root_key, self.module = contract, root_key, native_module
        self._original = (contract, root_key, native_module, native_module.NativeVisitedValueBank)
        self._contract_sha = digest(asdict(contract))
        self.rows, self.closed = [], False
        self._rows_original = []
        self._bindings_original = []

    def begin(self, record):
        require(not self.closed and len(self.rows) < 4
            and len(self._bindings_original) == len(self.rows),
            "native fixed-work unfinished invocation/cap; no retry")
        self._check()
        require(type(record) is dict and set(record) == {"state_str", "seed", "construction_ordinals"}
            and type(record["state_str"]) is str and bool(record["state_str"])
            and type(record["seed"]) is int and 0 <= record["seed"] < 2**64,
            "original explicit native fixed-work call required")
        aliases = record["construction_ordinals"]
        used = {v for row in self.rows for v in row["construction_ordinals"]}
        require(type(aliases) is list and 1 <= len(aliases) <= 4
            and all(type(v) is int and 0 <= v < 4 for v in aliases)
            and aliases == sorted(set(aliases)) and not used.intersection(aliases),
            "original disjoint construction aliases required; no replacement or duplicate samples")
        try:
            handle = self.module.NativeVisitedValueBank(sample_seed=self.contract.sample_seed,
                worker=len(self.rows), root_information_key=self.root_key, original_deadline_at=0.,
                leaves_per_native_invocation=self.contract.leaves_per_invocation,
                maximum_payload_bytes=self.contract.maximum_payload_bytes, fixed_work_native_labels=True,
                **({"certified_empty_delta_labels":True} if self.contract.certified_empty_delta_progress else {}))
        except BaseException:
            self.close()
            raise
        row = dict(handle=handle, construction_ordinals=deepcopy(aliases),
            parameters=dict(state_sha256=_sha(record["state_str"]),
            requested_iterations=16*len(aliases), batch_size=16, chance_seed=record["seed"]))
        self.rows.append(row)
        self._rows_original.append((row, handle, row["parameters"],
            digest(dict(parameters=row["parameters"], aliases=row["construction_ordinals"]))))
        return handle

    def complete(self, handle, raw_report):
        require(not self.closed and self.rows and self.rows[-1]["handle"] is handle
            and "completion_attempted" not in self.rows[-1], "native fixed-work completion/retry drift")
        self.rows[-1]["completion_attempted"] = True
        try:
            self._check()
            report = json.loads(raw_report)
            validate_native_report(report, **self.rows[-1]["parameters"])
            require(report["iterations"] == self.rows[-1]["parameters"]["requested_iterations"]
                and report.get("time_budget_exhausted", False) is False,
                "complete native fixed-work allocation required")
            self.rows[-1]["binding"] = dict(native_call_parameters=deepcopy(self.rows[-1]["parameters"]),
                construction_ordinals=deepcopy(self.rows[-1]["construction_ordinals"]),
                native_report=report, native_report_utf8=raw_report, report_utf8_sha256=_sha(raw_report))
            self._bindings_original.append((self.rows[-1]["binding"], digest(self.rows[-1]["binding"])))
        except BaseException:
            self.close()
            raise

    def bindings(self):
        self._check()
        require(not self.closed and self.rows
            and len(self._bindings_original) == len(self.rows),
            "original completed native selections required")
        return deepcopy([row["binding"] for row in self.rows])

    def label(self):
        require(not self.closed, "native fixed-work session already consumed")
        self.closed = True
        original_roster = tuple(self._rows_original)
        original_bindings = tuple(self._bindings_original)
        original_rows = tuple(row[0] for row in original_roster)
        original_handles = tuple(row[1] for row in original_roster)
        original_inputs = (self._original, self._contract_sha)
        def guard():
            self._check()
            require(self.closed is True and self._original is original_inputs[0]
                and self._contract_sha == original_inputs[1]
                and len(self.rows) == len(original_rows)
                and len(self._rows_original) == len(original_roster)
                and all(a is b for a, b in zip(self._rows_original, original_roster))
                and len(self._bindings_original) == len(original_bindings)
                and all(a is b for a, b in zip(self._bindings_original, original_bindings))
                and all(a is b for a, b in zip(self.rows, original_rows))
                and all(row["handle"] is handle for row, handle in zip(original_rows, original_handles))
                and all(row["parameters"] is params
                    and digest(dict(parameters=params, aliases=row["construction_ordinals"])) == sha
                    for row, _, params, sha in original_roster)
                and all(row["binding"] is expected and digest(expected) == sha
                    for row, (expected, sha) in zip(original_rows, original_bindings)),
                "original native label roster/bindings changed during callback")
        try:
            require(original_rows and len(original_bindings) == len(original_rows)
                and all(row.get("completion_attempted") is True for row in original_rows),
                "actual original completed selections required before native labels")
            results = []
            for i, row in enumerate(original_rows):
                guard()
                raw = original_handles[i].label()
                guard()
                result = dict(invocation=i, binding=deepcopy(row["binding"]), raw_native_export=raw,
                    readout=read_fixed_native_values(raw, binding=row["binding"], contract=self.contract,
                        root_key=self.root_key, worker=i))
                guard()
                results.append(result)
            guard()
            return results
        finally:
            for handle in original_handles:
                handle.close()
            self.rows.clear()
            self._rows_original.clear()
            self._bindings_original.clear()

    def _check(self):
        contract, key, module, factory = self._original
        require(type(self.contract) is NativeFixedWorkValueContract and self.contract is contract
            and self.root_key == key and self.module is module
            and self.module.NativeVisitedValueBank is factory
            and digest(asdict(self.contract)) == self._contract_sha,
            "original native diagnostic contract/module/root changed")
        require(len(self.rows) == len(self._rows_original)
            and len(self._bindings_original) <= len(self._rows_original)
            and all(row is original and row["handle"] is handle and row["parameters"] is params
                and digest(dict(parameters=params, aliases=row["construction_ordinals"])) == sha
                for row, (original, handle, params, sha)
                    in zip(self.rows, self._rows_original))
            and all(self.rows[i].get("binding") is binding and digest(binding) == sha
                for i, (binding, sha) in enumerate(self._bindings_original)),
            "original native invocation row/handle/parameters/binding changed")

    def __reduce_ex__(self, protocol):
        raise TypeError("private native fixed-work session cannot enter transport")

    def close(self):
        self.closed = True
        for _, handle, _, _ in self._rows_original:
            handle.close()
        self.rows.clear()
        self._rows_original.clear()
        self._bindings_original.clear()
