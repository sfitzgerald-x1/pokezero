"""Prospective private native ancestry, not labels or general engine fidelity.

Exports actual evaluation-event reservoirs after selection. A separate reader
joins them to the owning fixed-work result and the atomic pending-row journal.
No model, sampler, environment, replay or process is started by the reader.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import heapq
import json
import struct

from .search_over_raw import digest, require
from .search_over_raw_correspondence import boundary_bindings

SCHEMA = "pokezero.native-searched-ancestry.private-owned.v1"
EDGE_SCHEMA = "pokezero.native-searched-ancestry.private-owned.v2"
POPULATION = "actual_model_evaluations_including_cancelled_rounds"


def _sha(text):
    require(type(text) is str, "actual UTF-8 text required")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _priority(seed, worker, root, ordinal):
    raw = json.dumps(["visited-model-sample.v1", seed, worker, root, ordinal],
        separators=(",", ":"), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


@dataclass(frozen=True)
class NativeAncestryContract:
    sample_seed: int
    leaves_per_native_invocation: int = 2
    maximum_payload_bytes: int = 8388608
    capture_native_edges: bool = False

    def __post_init__(self):
        require(type(self.sample_seed) is int and 0 <= self.sample_seed < 2**64
            and type(self.leaves_per_native_invocation) is int and 1 <= self.leaves_per_native_invocation <= 8
            and type(self.maximum_payload_bytes) is int and 0 < self.maximum_payload_bytes <= 8388608,
            "explicit bounded prospective ancestry contract required")
        require(type(self.capture_native_edges) is bool,"explicit prospective native-edge capture mode required")

    @property
    def schema(self):
        return EDGE_SCHEMA if self.capture_native_edges else SCHEMA


class NativeAncestrySession:
    """One serial fixed-work decision; consume before every FFI export.

    Uses the existing private frontier bank but NEVER calls its label or old
    excluded-fixture callback. Each actual invocation gets its own reservoir.
    """
    def __init__(self, contract, *, root_key, native_module=None):
        require(type(contract) is NativeAncestryContract and type(root_key) is str
            and bool(root_key) and len(root_key) % 2 == 0
            and all(c in "0123456789abcdef" for c in root_key), "native ancestry root binding required")
        if native_module is None:
            import pokezero_search as native_module
        require(hasattr(native_module, "NativeVisitedValueBank"), "prospective native ancestry build required")
        self.contract, self.root_key, self.module = contract, root_key, native_module
        self.rows, self.closed = [], False

    def begin(self, record):
        require(not self.closed and len(self.rows) < 4
            and all("export_utf8" in r for r in self.rows), "ancestry invocation unfinished/cap; no retry")
        options = dict(capture_native_edges=True) if self.contract.capture_native_edges else {}
        handle = self.module.NativeVisitedValueBank(sample_seed=self.contract.sample_seed,
            worker=len(self.rows), root_information_key=self.root_key,
            original_deadline_at=0., fixed_work_capture=True,
            leaves_per_native_invocation=self.contract.leaves_per_native_invocation,
            maximum_payload_bytes=self.contract.maximum_payload_bytes, capture_searched_lines=True,**options)
        self.rows.append(dict(handle=handle, state_sha256=_sha(record["state_str"]), world_seed=record["seed"]))
        return handle

    def complete(self, handle, raw_report):
        require(not self.closed and self.rows and self.rows[-1]["handle"] is handle
            and "export_attempted" not in self.rows[-1], "ancestry completion/retry drift")
        row = self.rows[-1]
        row["export_attempted"] = True
        try:
            raw = handle.export_searched_ancestry()
            require(type(raw) is str and len(raw.encode()) <= self.contract.maximum_payload_bytes,
                "private native ancestry export cap/type drift")
            payload = json.loads(raw)
            require(payload.get("schema") == self.contract.schema
                and payload.get("root_native_state_sha256") == row["state_sha256"]
                and payload.get("selection_report_utf8_sha256") == _sha(raw_report),
                "actual native ancestry root/selection drift")
            row["export_utf8"] = raw
        finally:
            handle.close()

    def exports(self):
        require(not self.closed and all("export_utf8" in row for row in self.rows),
            "failed ancestry invocation cannot become selection evidence")
        self.closed = True
        return [dict(invocation=i, world_seed=row["world_seed"],
            export_utf8=row["export_utf8"], export_utf8_sha256=_sha(row["export_utf8"]))
            for i, row in enumerate(self.rows)]

    def close(self):
        self.closed = True
        for row in self.rows:
            row["handle"].close()


def _unique_object(pairs):
    require(len({k for k, _ in pairs}) == len(pairs), "duplicate ancestry JSON fields")
    return dict(pairs)


def read_native_ancestry_export(row, call, *, contract, root_key, worker, root_binding, searching_side_one):
    """Validate the entire retained sample against independently read atomics.

    Fixed-work model-only input; cancelled or unpriced events cannot be silently
    assigned a committed membership. Empty populations remain empty evidence.
    """
    require(type(contract) is NativeAncestryContract and type(row) is dict
        and set(row) == {"invocation", "world_seed", "export_utf8", "export_utf8_sha256"}
        and type(row["invocation"]) is int and row["invocation"] == worker
        and type(row["world_seed"]) is int
        and row["world_seed"] == call["native_call_parameters"]["chance_seed"]
        and row["export_utf8_sha256"] == _sha(row["export_utf8"])
        and len(row["export_utf8"].encode()) <= contract.maximum_payload_bytes,
        "ancestry actual invocation/export binding drift")
    payload = json.loads(row["export_utf8"], object_pairs_hook=_unique_object)
    fields = {"schema", "root_native_state_sha256", "selection_report_utf8_sha256", "root_information_key",
        "sample_seed", "worker", "model_evaluations_seen", "leaves_per_native_invocation",
        "maximum_payload_bytes", "retained_private_payload_bytes", "evaluations", "event_population",
        "private_controller_only", "labels_used_for_backup", "full_world_equivalence", "runtime_qualified", "fixed_work_capture",
        "model_self_side_one"}
    require(type(payload) is dict and set(payload) == fields and payload["schema"] == contract.schema
        and payload["root_native_state_sha256"] == call["native_call_parameters"]["state_sha256"]
        and payload["selection_report_utf8_sha256"] == call["report_utf8_sha256"]
        and payload["root_information_key"] == root_key and payload["event_population"] == POPULATION
        and payload["private_controller_only"] is True and payload["labels_used_for_backup"] is False
        and payload["fixed_work_capture"] is True
        and payload["full_world_equivalence"] is False and payload["runtime_qualified"] is False,
        "prospective private ancestry schema/root/scope drift")
    for name, expected in (("sample_seed", contract.sample_seed), ("worker", worker),
            ("leaves_per_native_invocation", contract.leaves_per_native_invocation),
            ("maximum_payload_bytes", contract.maximum_payload_bytes)):
        require(type(payload[name]) is int and payload[name] == expected, "ancestry sampling contract drift")
    require(type(payload["retained_private_payload_bytes"]) is int
        and 0 < payload["retained_private_payload_bytes"] <= contract.maximum_payload_bytes,
        "ancestry private retention accounting drift")
    report = call["native_report"]
    from .search_over_raw_native_atomic_work import validate_native_report
    validate_native_report(report, **call["native_call_parameters"])
    require(_sha(call["native_report_utf8"]) == call["report_utf8_sha256"]
        and digest(json.loads(call["native_report_utf8"], object_pairs_hook=_unique_object)) == digest(report),
        "ancestry actual atomic report bytes drift")
    seen = payload["model_evaluations_seen"]
    require(type(seen) is int and seen >= 0 and type(report.get("model_evals")) is int
        and seen == report["model_evals"] - 1, "ancestry evaluation population drift")
    require(type(searching_side_one) is bool
        and report.get("collision_self_side") == ("side_one" if searching_side_one else "side_two")
        and (payload["model_self_side_one"] is searching_side_one if seen else payload["model_self_side_one"] is None),
        "ancestry model/atomic/consumed subject frame drift")
    # Atomic journals were independently validated by the owning run reader.
    # Retained paths must also belong to the precise actual pending model rows.
    events, offset = {}, 0
    for round in report["atomic_work"]["rounds"]:
        require(round["status"] == "COMMITTED", "cancelled ancestry membership remains unqualified")
        snapshot = round["after_collection"]
        for chance, branches in enumerate(snapshot["chances"]):
            for branch, value in enumerate(branches):
                local = value["pending_row"]
                if local is not None:
                    ordinal = offset + local
                    require(ordinal not in events, "duplicate ancestry pending-row membership")
                    events[ordinal] = ((chance, branch), round["row_value_bits"][local], snapshot, round["ordinal"])
        offset += len(round["row_value_bits"])
    require(offset == seen and sorted(events) == list(range(seen)), "ancestry atomic event denominator drift")
    expected = heapq.nsmallest(contract.leaves_per_native_invocation,
        ((_priority(contract.sample_seed, worker, root_key, ordinal), ordinal) for ordinal in range(seen)))
    require(type(payload["evaluations"]) is list and len(payload["evaluations"]) == len(expected),
        "ancestry retained sample omitted or duplicated")
    validated = []
    for item, (priority, ordinal) in zip(payload["evaluations"], expected):
        require(type(item) is dict and set(item) == {"priority", "branch_key", "native_state_sha256",
            "private_frontier_sha256", "model_self_value01_bits", "capture"} and type(item["priority"]) is int
            and item["priority"] == priority, "ancestry sampled event identity drift")
        capture = item["capture"]
        require(type(capture) is dict and type(capture.get("steps")) is list
            and all(("native_edge" in step) is contract.capture_native_edges for step in capture["steps"]),
            "prospective native-edge roster/schema drift; legacy evidence is not upgraded")
        bindings = boundary_bindings(capture, root_binding=root_binding,
            selection_report_sha256=call["report_utf8_sha256"])
        key, atomic_bits, snapshot, round_id = events[ordinal]
        bits = item["model_self_value01_bits"]
        require(type(bits) is int and 0 <= bits < 2**32, "actual model float32 bits required")
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        require(0 <= value <= 1, "invalid original model value")
        f32 = lambda number:struct.unpack("<f", struct.pack("<f", number))[0]
        signed = f32(f32(2.0 * value) - 1.0)
        side_one_value = value if searching_side_one else f32(1.0 - value)
        require(struct.unpack("<I", struct.pack("<f", side_one_value))[0] == atomic_bits,
            "ancestry model value does not equal the exact frame-converted atomic row")
        require(capture["evaluation_ordinal"] == ordinal and capture["model_signed_value"] == signed
            and _sha(capture["root_native_state"]) == payload["root_native_state_sha256"]
            and _sha(capture["native_frontier"]) == item["native_state_sha256"]
            and digest(item["branch_key"]) == digest(list(key))
            and digest(capture["steps"][-1]["key"]) == digest(list(key))
            and type(item["private_frontier_sha256"]) is str and len(item["private_frontier_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in item["private_frontier_sha256"]),
            "ancestry actual evaluated frontier/membership drift")
        parent = None
        for step in capture["steps"]:
            chance, branch = step["key"]
            owners = [index for index, node in enumerate(snapshot["decisions"])
                if any(child[2] == chance for child in node["children"])]
            require(len(owners) == 1 and chance < len(snapshot["chances"])
                and branch < len(snapshot["chances"][chance])
                and (owners[0] == 0 if parent is None else
                    snapshot["chances"][parent[0]][parent[1]]["child"] == owners[0]),
                "ancestry path is not a member of the actual atomic tree")
            parent = (chance, branch)
        validated.append(dict(evaluation_ordinal=ordinal, priority=priority, committed_round=round_id,
            capture_sha256=digest(capture), boundaries=bindings,
            membership="actual_committed_model_evaluation_not_independent_label"))
    return dict(export_utf8_sha256=row["export_utf8_sha256"], actual_model_evaluations=seen,
        retained_evaluations=validated, full_world_equivalence=False, runtime_qualified=False)


def validate_owned_native_ancestry(selected, *, run_validator, contract):
    from .search_over_raw_fixed_work_run import FixedWorkRunValidator
    require(type(run_validator) is FixedWorkRunValidator and run_validator.configuration.arm == "incumbent"
        and run_validator.configuration.leaf == "model" and type(contract) is NativeAncestryContract,
        "independent owning fixed-model reader and prospective contract required")
    before = digest(selected)
    witness = run_validator(selected)
    saved = deepcopy(selected)
    rows = saved["evidence"].get("native_searched_ancestry")
    calls = saved["evidence"]["consumed_world_inputs"]["calls"]
    require(type(rows) is list and len(rows) == len(calls), "ancestry actual invocation roster omission")
    results = []
    manifest = saved["evidence"]["consumed_world_inputs"]
    for i, (row, call) in enumerate(zip(rows, calls)):
        sides = {r["slot_sides"][run_validator.consumed["subject"]] for r in manifest["rows"]
            if r["ordinal"] in call["construction_ordinals"]}
        require(len(sides) == 1 and sides <= {"side_one", "side_two"}, "actual constructed subject frame drift")
        results.append(read_native_ancestry_export(row, call, contract=contract,
            root_key=run_validator.consumed["information_key"], worker=i,
            root_binding=run_validator.consumed["root_binding"], searching_side_one=next(iter(sides)) == "side_one"))
    require(digest(selected) == before, "ancestry selected evidence mutated during validation")
    return dict(schema="pokezero.search-over-raw.owned-native-ancestry-readout.v1", selected_sha256=before,
        owning_run_validated=witness == before, invocations=results,
        construction_ordinals=4, actual_native_calls=len(calls),
        full_world_equivalence=False, independent_terminal_labels=False,
        party_mapping_validated=False, transition_correctness_validated=False,
        resource_cost_complete=False, runtime_qualified=False, scientific_collection_authorized=False)
