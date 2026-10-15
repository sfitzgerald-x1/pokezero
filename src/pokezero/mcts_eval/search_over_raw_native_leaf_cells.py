"""Fixed cells for retained native AFTER events, never continuation outcomes.

Construction aliases are correspondence records, not extra model events. The
canonical alias is frozen from original invocation membership before any
constructor, replay or label result. Neither a plan nor a caller's completed
cell can supply independent terminal evidence. Actual AFTER raw/raw execution
and original endpoint/history admission are separate owning protocols.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from .search_over_raw import digest, require, rng_seed
from .search_over_raw_ancestry import NativeAncestryContract, read_native_ancestry_export
from .search_over_raw_outcomes import _sha

SCHEMA = "pokezero.search-over-raw.native-after-leaf-cell-plan.v1"


def leaf_cell_law():
    """Prospective law: no eligibility/outcome-informed alias selection."""
    return dict(schema=SCHEMA, population="unique_retained_native_model_evaluation_events",
        canonical_alias="minimum_original_call_construction_ordinal_no_fallback",
        replicates=8, maximum_boundaries=250, policy="raw_argmax_both_seats_from_after_step_zero",
        randomness="native-after-label.v1/sample_seed/event_id/replicate/step",
        unknown_outcome_interval=[-1, 1], retries_permitted=False,
        labels_used_for_backup=False, scientific_collection_authorized=False)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class NativeLeafCellPlan:
    payload_utf8: str

    def receipt(self):
        value = json.loads(self.payload_utf8)
        require(_canonical(value) == self.payload_utf8 and value.get("schema") in {SCHEMA,SCHEMA.replace(".v1",".v2")},
            "canonical original leaf-cell plan required")
        return value

    @property
    def sha256(self):
        return hashlib.sha256(self.payload_utf8.encode()).hexdigest()


def native_leaf_run_binding(reader):
    from .search_over_raw_fixed_work_run import FixedWorkRunValidator
    require(type(reader) is FixedWorkRunValidator, "exact original native owning reader required")
    return dict(directory=str(reader.directory), job_sha256=reader.job_sha, binding_sha256=reader.binding,
        statistics_root_id=reader.statistics_root_id, selection_root_id=reader.root_id,
        selection_seed=reader.seed, runtime_sha256=reader.runtime_sha)


def plan_native_leaf_cells(selected, *, run_binding, consumed_binding, ancestry_contract,paired_labels=False):
    """Structural builder: validate exports, retain every sampled event.

No owning-process clearance is inferred from run_binding supplied by a caller.
The owning controller and completed reader separately validate that run. Equal
frontiers and predictions never merge different evaluation events/invocations.
"""
    require(type(paired_labels) is bool and type(ancestry_contract) is NativeAncestryContract and type(run_binding) is dict
        and set(run_binding) == {"directory", "job_sha256", "binding_sha256", "statistics_root_id",
            "selection_root_id", "selection_seed", "runtime_sha256"}
        and all(_sha(run_binding[k]) for k in ("job_sha256", "binding_sha256", "runtime_sha256"))
        and type(run_binding["selection_seed"]) is int
        and all(type(run_binding[k]) is str and run_binding[k] for k in
            ("directory", "statistics_root_id", "selection_root_id"))
        and type(consumed_binding) is dict and consumed_binding.get("subject") in {"p1", "p2"},
        "explicit original run/consumed/ancestry bindings required")
    before = digest(dict(selected=selected, run=run_binding, consumed=consumed_binding))
    manifest = selected["evidence"]["consumed_world_inputs"]
    calls, worlds = manifest["calls"], manifest["rows"]
    exports = selected["evidence"]["native_searched_ancestry"]
    require(type(calls) is list and 1 <= len(calls) <= 4 and type(worlds) is list
        and sorted(r["ordinal"] for r in worlds) == list(range(4))
        and all(type(r["ordinal"]) is int and type(r["call_ordinal"]) is int for r in worlds)
        and type(exports) is list and len(exports) == len(calls),
        "complete original invocation/construction/export roster required")
    selected_sha = digest(selected)
    events, expanded, seen = [], 0, 0
    partition = []
    for invocation, (call, export) in enumerate(zip(calls, exports)):
        aliases = call["construction_ordinals"]
        require(type(call["call_ordinal"]) is int and call["call_ordinal"] == invocation
            and type(aliases) is list and aliases and all(type(n) is int and 0 <= n < 4 for n in aliases)
            and len(set(aliases)) == len(aliases)
            and sorted(aliases) == sorted(r["ordinal"] for r in worlds if r["call_ordinal"] == invocation),
            "original call alias membership required; no eligible-subset selection")
        partition.extend(aliases)
        sides = {r["slot_sides"][consumed_binding["subject"]] for r in worlds if r["ordinal"] in aliases}
        require(len(sides) == 1 and sides <= {"side_one", "side_two"}, "original model subject frame required")
        checked = read_native_ancestry_export(export, call, contract=ancestry_contract,
            root_key=consumed_binding["information_key"], worker=invocation,
            root_binding=consumed_binding["root_binding"], searching_side_one=next(iter(sides)) == "side_one")
        payload = json.loads(export["export_utf8"])
        private = {item["capture"]["evaluation_ordinal"]: item for item in payload["evaluations"]}
        seen += checked["actual_model_evaluations"]
        for retained in sorted(checked["retained_evaluations"], key=lambda e: e["evaluation_ordinal"]):
            ordinal = retained["evaluation_ordinal"]
            item = private[ordinal]
            capture = item["capture"]
            identity = dict(run_binding=deepcopy(run_binding), native_selected_sha256=selected_sha,
                invocation=invocation, evaluation_ordinal=ordinal, native_capture_sha256=retained["capture_sha256"],
                frontier_utf8_sha256=item["native_state_sha256"])
            event_id = digest(identity)
            events.append(dict(event_id=event_id, **identity, construction_aliases=sorted(aliases),
                canonical_construction_ordinal=min(aliases), model_subject=consumed_binding["subject"],
                model_self_side_one=payload["model_self_side_one"],
                model_self_value01_bits=item["model_self_value01_bits"], model_signed_value=capture["model_signed_value"],
                required_after_snapshot_index=len(capture["steps"]), labels=[dict(
                    cell_id=digest(dict(event_id=event_id, replicate=rep)), replicate=rep,
                    chance_stream_key=rng_seed("native-after-label.v1", ancestry_contract.sample_seed, event_id, rep),
                    status="UNSTARTED_UNCERTAIN", signed_outcome=None, boundaries=0) for rep in range(8)]))
            expanded += len(aliases)
    require(sorted(partition) == list(range(4)) and len({e["event_id"] for e in events}) == len(events)
        and before == digest(dict(selected=selected, run=run_binding, consumed=consumed_binding)),
        "original leaf event roster or bindings changed")
    law=leaf_cell_law();schema=SCHEMA
    if paired_labels:
        from .search_over_raw_paired_chain import paired_leaf_law,paired_label_coordinates
        law=paired_leaf_law();schema=SCHEMA.replace(".v1",".v2")
        for event in events:
            root_id,namespace=paired_label_coordinates(event,ancestry_contract.sample_seed)
            for cell in event["labels"]:
                cell["chance_stream_key"]=rng_seed(namespace,root_id,cell["replicate"])
    return NativeLeafCellPlan(_canonical(dict(schema=schema, law=law,
        native_selected_sha256=selected_sha, run_binding=deepcopy(run_binding), consumed_binding_sha256=digest(consumed_binding),
        sample_seed=ancestry_contract.sample_seed, events=events, actual_native_calls=len(calls),
        actual_model_evaluations=seen, retained_unique_evaluation_events=len(events),
        construction_alias_event_records=expanded, required_label_cells=len(events)*8,
        completed_label_cells=0, label_process_launches=0, independent_terminal_labels=False,
        native_evaluation_input_parity=False, runtime_qualified=False, scientific_collection_authorized=False)))


def freeze_owned_native_leaf_cells(native_owner, *, ledger,paired_labels=False):
    """Validate original live owner; freeze roster before constructor outcomes."""
    from .search_over_raw_diagnostic_producer import OwnedFixedWorkDiagnostic
    from .search_over_raw_consumed_materialization import _original_diagnostic_owner_guard
    require(type(native_owner) is OwnedFixedWorkDiagnostic, "original private native diagnostic owner required")
    objects = (native_owner.source_owner, native_owner.job, native_owner.reader, native_owner.selected)
    guard = _original_diagnostic_owner_guard(native_owner.selected, [], [], native_owner.reader)
    native_owner.validate(ledger)
    readouts = {str(native_owner.reader.directory): deepcopy(native_owner.reader.last_validation)}
    plan = plan_native_leaf_cells(native_owner.selected, run_binding=native_leaf_run_binding(native_owner.reader),
        consumed_binding=native_owner.reader.consumed, ancestry_contract=native_owner.job.native_ancestry_contract,
        paired_labels=paired_labels)
    frozen = plan.payload_utf8
    native_owner.validate(ledger)
    native_owner._original_inputs()
    guard(readouts)
    require(all(a is b for a, b in zip((native_owner.source_owner, native_owner.job,
        native_owner.reader, native_owner.selected), objects)) and plan.payload_utf8 == frozen,
        "original native leaf plan/owner changed through callbacks")
    return plan
