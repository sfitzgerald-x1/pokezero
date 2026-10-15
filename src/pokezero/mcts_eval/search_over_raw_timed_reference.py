"""Prospective reference timed-prefix reader; not a process/runtime qualifier.

No collection authority, fixed quota, raw fallback, retry or historical recovery.
Every worker's OWN exports survive even when it attempted no draws. Clocks are
backend-reported consistency witnesses, not independently established on-time
deployability. Normally closed owning process generations remain a separate gate.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import math
from types import SimpleNamespace

from .search_over_raw import digest, require
from .search_over_raw_atomic_work import _validate_reference_atomic_work
from .search_over_raw_world_inputs import reference_manifest, compare_world_inputs, TIMED_REFERENCE_PUBLIC_CAPTURE

SCHEMA = "pokezero.search-over-raw.timed-reference-prefix.v1"


def _record(value):
    return asdict(value) if is_dataclass(value) else value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_reference_public_request(evidence, *, binding, expected_sha256):
    """Exact observed transport at capture and all twenty worker preparations.

    No inference about pending-transition semantics or sampled-world correctness.
    A worker with no draws still prepared a request and must export its witness.
    """
    require(type(expected_sha256) is str and len(expected_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_sha256),
        "external full reference public request digest required")
    expected = dict(schema="pokezero.search-over-raw.public-request-consumption.v1",
        public_request_sha256=expected_sha256, consumed_binding=deepcopy(binding),
        full_supported_public_state_and_pending=True, private_source_snapshot_included=False,
        runtime_qualified=False)
    require(digest(evidence.get("adapter_public_request_witness")) == digest(expected),
        "reference actual adapter public capture differs from external request")
    workers, exports = set(), 0
    for receipt in evidence["worker_receipts"]:
        worker = receipt["worker"]
        observed = receipt["evidence"].get("worker_public_request_preparation")
        require(type(worker) is int and 0 <= worker < 20 and type(observed) is dict
            and set(observed) == {"worker", "witness"} and type(observed["worker"]) is int
            and observed["worker"] == worker and digest(observed["witness"]) == digest(expected),
            "reference worker preparation/export differs from full public request/index")
        workers.add(worker); exports += 1
    require(workers == set(range(20)), "all twenty actual public request preparations required")
    return dict(public_request_sha256=expected_sha256, observed_worker_preparations=20,
        observed_worker_exports=exports, adapter_capture_validated=True,
        full_public_request_consumption_validated=True, runtime_qualified=False,
        collection_authorized=False)


def validate_reference_timed_inputs(evidence, *, binding, battle_id, selection_seed,
                                    nominal_seconds, public_request_sha256=None):
    """Recompute full twenty-worker prefix, including normal zero cancellation."""
    saved = deepcopy(evidence)
    require(type(saved) is dict and type(binding) is dict and set(binding) == {
        "root_binding", "subject", "set_source_hash", "information_key"}
        and type(selection_seed) is int and 0 <= selection_seed < 2**64
        and type(nominal_seconds) in (int, float) and nominal_seconds in (1., 3., 10.),
        "external timed reference input/seed/ceiling required")
    prefix = saved.get("reference_timed_prefix")
    require(type(prefix) is dict and set(prefix) == {
        "schema", "battle_id", "selection_seed", "decision_id", "nominal_search_seconds",
        "decision_started_monotonic", "absolute_deadline_monotonic", "decision_finished_monotonic",
        "adapter_started_monotonic", "stop_acknowledgements",
        "initial_dispatch_workers", "acknowledged_done_workers", "status", "retry_authorized",
        "runtime_qualified", "on_time_deployability_established"}
        and prefix["schema"] == SCHEMA and prefix["battle_id"] == battle_id
        and type(prefix["selection_seed"]) is int and prefix["selection_seed"] == selection_seed
        and type(prefix["decision_id"]) is int and prefix["decision_id"] == 1
        and type(saved.get("decision_id")) is int and saved["decision_id"] == 1
        and type(prefix["nominal_search_seconds"]) in (int, float)
        and prefix["nominal_search_seconds"] == nominal_seconds
        and type(prefix["initial_dispatch_workers"]) is int and prefix["initial_dispatch_workers"] == 6
        and digest(prefix["acknowledged_done_workers"]) == digest(list(range(20)))
        and all(prefix[key] is False for key in ("retry_authorized", "runtime_qualified",
            "on_time_deployability_established")), "reference timed protocol/root drift")
    start, deadline, finish = (prefix[key] for key in (
        "decision_started_monotonic", "absolute_deadline_monotonic", "decision_finished_monotonic"))
    origin = prefix["adapter_started_monotonic"]
    require(all(_finite(value) for value in (origin, start, deadline, finish)) and deadline > 0
        and origin <= start <= finish and finish >= deadline and deadline == origin + nominal_seconds,
        "reference timed absolute decision clock drift")
    pids = saved.get("worker_pids")
    require(type(pids) in (list, tuple) and len(pids) == 20 and len(set(pids)) == 20
        and all(type(pid) is int and pid > 0 for pid in pids)
        and _finite(saved.get("startup_seconds")), "reference timed actual worker roster required")
    work = _validate_reference_atomic_work(saved, binding=binding, battle_id=battle_id,
                                          allow_zero_deadline=True)
    receipts, result = saved["worker_receipts"], _record(saved["result"])
    require(receipts[0]["evidence"]["atomic_work"]["root"]["key"] == binding["information_key"],
        "reference timed root differs from externally consumed information key")
    require(type(saved.get("exchange_count")) is int and saved["exchange_count"] == len(receipts)
        and _finite(result.get("elapsed_seconds")) and type(result.get("deadline_exhausted")) is bool
        and _finite(result.get("deadline_overrun_seconds"))
        and math.isclose(result["elapsed_seconds"], finish-start, rel_tol=0, abs_tol=1e-8)
        and result["deadline_exhausted"] == (finish >= deadline)
        and math.isclose(result["deadline_overrun_seconds"], max(0., finish-deadline),
            rel_tol=0, abs_tol=1e-8), "reference timed aggregate clock/work drift")
    base_version = int(receipts[0]["evidence"]["atomic_work"]["root"]["faint_count"] > 0)
    first_updates, final_receipts = {}, {}
    for index, receipt in enumerate(receipts, base_version+1):
        batch = _record(receipt["batch"])
        require(type(receipt.get("master_version")) is int and receipt["master_version"] == index
            and _finite(receipt.get("preparation_seconds"))
            and (receipt["sequence"] == 1 or receipt["preparation_seconds"] == 0)
            and _finite(batch.get("elapsed_seconds")) and type(batch.get("deadline_exhausted")) is bool
            and _finite(batch.get("deadline_overrun_seconds"))
            and (batch["deadline_exhausted"] or batch["deadline_overrun_seconds"] == 0)
            and type(batch.get("trajectories")) is int and batch["trajectories"] <= 10,
            "reference timed exchange/worker clock drift")
        if receipt["sequence"] == 1:
            first_updates[receipt["worker"]] = index
        if receipt["worker"] in final_receipts:
            require(final_receipts[receipt["worker"]]["timed_acknowledgement"]["limit"] == 10,
                "reference timed worker work continued after its stop acknowledgement")
        ack = receipt.get("timed_acknowledgement")
        require(type(ack) is dict and set(ack) == {"worker", "sequence", "master_version", "limit",
            "decided_monotonic", "acknowledged_monotonic", "deadline_expired", "worker_deadline_exhausted"}
            and digest({key:ack[key] for key in ("worker","sequence","master_version")}) == digest(dict(
                worker=receipt["worker"], sequence=receipt["sequence"], master_version=index))
            and type(ack["limit"]) is int and ack["limit"] in (0,10)
            and all(_finite(ack[key]) for key in ("decided_monotonic","acknowledged_monotonic"))
            and start <= ack["decided_monotonic"] <= ack["acknowledged_monotonic"] <= finish
            and type(ack["deadline_expired"]) is bool
            and ack["deadline_expired"] == (ack["decided_monotonic"] >= deadline)
            and type(ack["worker_deadline_exhausted"]) is bool
            and ack["worker_deadline_exhausted"] is batch["deadline_exhausted"]
            and ack["limit"] == (0 if ack["deadline_expired"] or batch["deadline_exhausted"] else 10),
            "reference timed actual acknowledgement/limit/deadline law drift")
        final_receipts[receipt["worker"]] = receipt
        for draw in receipt["evidence"]["draws"]:
            if draw["status"] == "DEADLINE_CANCELLED":
                diagnostic = draw.get("sampling_diagnostic", {})
                checked, draw_deadline = diagnostic.get("checked_at"), diagnostic.get("deadline_at")
                require(diagnostic.get("schema") == "pokezero.world-sampling-deadline.v1"
                    and _finite(checked) and _finite(draw_deadline) and checked >= draw_deadline
                    and draw_deadline == deadline and diagnostic.get("accepted_world") is False
                    and diagnostic.get("backed_up") is False and batch["deadline_exhausted"] is True,
                    "reference timed sampling cancellation lacks this absolute deadline")
    stops = prefix["stop_acknowledgements"]
    require(type(stops) is list and len(stops) == 20, "reference timed stop acknowledgement roster missing")
    for worker, stop in enumerate(stops):
        last = final_receipts[worker]
        batch = _record(last["batch"])
        require(type(stop) is dict and set(stop) == {"worker", "sequence", "master_version", "limit",
            "decided_monotonic", "acknowledged_monotonic", "deadline_expired", "worker_deadline_exhausted"}
            and digest({key:stop[key] for key in ("worker","sequence","master_version","limit")}) == digest(dict(
                worker=worker, sequence=last["sequence"], master_version=last["master_version"], limit=0))
            and _finite(stop["acknowledged_monotonic"])
            and max(start, deadline) <= stop["acknowledged_monotonic"] <= finish
            and type(stop["deadline_expired"]) is bool
            and type(stop["worker_deadline_exhausted"]) is bool
            and stop["worker_deadline_exhausted"] is batch["deadline_exhausted"]
            and (stop["deadline_expired"] or stop["worker_deadline_exhausted"])
            and (not stop["deadline_expired"] or stop["acknowledged_monotonic"] >= deadline)
            and digest(stop) == digest(last["timed_acknowledgement"]),
            "reference timed done is not bound to its final actual zero-limit acknowledgement")
    schedule = saved.get("initial_dispatch_schedule")
    require(type(schedule) in (tuple, list) and len(schedule) == 20, "reference timed dispatch roster missing")
    released, prior_time = set(), 0.
    for worker, dispatch in enumerate(schedule):
        elapsed = dispatch.get("seconds_from_decision_start")
        require(type(dispatch) is dict and set(dispatch) == {"worker", "seconds_from_decision_start",
            "released_by_first_update", "master_version", "deadline_expired"}
            and type(dispatch["worker"]) is int and dispatch["worker"] == worker
            and _finite(elapsed) and prior_time <= elapsed <= result["elapsed_seconds"]
            and type(dispatch["deadline_expired"]) is bool
            and dispatch["deadline_expired"] == (start+elapsed >= deadline)
            and type(dispatch["master_version"]) is int, "reference timed FIFO/deadline dispatch drift")
        prior_time = elapsed
        source = dispatch["released_by_first_update"]
        if worker < 6:
            require(source is None and dispatch["master_version"] == base_version,
                "reference timed first dispatch changed")
        else:
            require(type(source) is int and 0 <= source < worker and source not in released
                and dispatch["master_version"] == first_updates[source]
                and source == sorted(first_updates, key=first_updates.get)[worker-6],
                "reference timed activation lacks its actual first OWN update")
            released.add(source)
    manifest = saved.get("consumed_world_inputs")
    require(type(manifest) is dict and digest(reference_manifest(saved, binding)) == digest(manifest),
        "reference timed consumed population differs from actual draw receipts")
    compare_world_inputs(manifest, manifest)  # Schema/scope validation, not a completeness gate on uncertain draws.
    status = "SELECTED" if work["accepted_trajectories"] else "UNQUALIFIED_AT_CEILING"
    require(prefix["status"] == status, "reference timed zero/status drift")
    public = None if public_request_sha256 is None else validate_reference_public_request(
        saved, binding=binding, expected_sha256=public_request_sha256)
    require(digest(evidence) == digest(saved), "reference timed evidence changed during validation")
    readout = dict(schema=SCHEMA, status=status, accepted_units=work["accepted_trajectories"],
        attempted_draws=work["attempted_draws"], transitions=work["transitions"],
        witness_sha256=work["witness_sha256"], all_twenty_worker_exports_validated=True,
        process_ownership_validated=False, runtime_qualified=False,
        on_time_deployability_established=False, independent_terminal_labels=False,
        scientific_strength_evidence=False, retry_authorized=False)
    if public is not None:
        readout.update(public_request_consumption=public, full_public_request_consumption_validated=True)
    return readout


def read_reference_timed_selection(selected, *, configuration, runtime_configuration, runtime_sha256,
                                   root_id, statistics_root_id, selection_seed, consumed_binding,
                                   legal_action_mask, public_request_sha256=None):
    """External decision bindings and selector law; does not attest process exits."""
    require(not any(k in runtime_configuration for k in ("team_oracle","reference_team_oracle_observation")),
        "public timed reference cannot carry oracle observation")
    return _read_reference_timed_selection(selected,configuration=configuration,
        runtime_configuration=runtime_configuration,runtime_sha256=runtime_sha256,
        root_id=root_id,statistics_root_id=statistics_root_id,selection_seed=selection_seed,
        consumed_binding=consumed_binding,legal_action_mask=legal_action_mask,
        public_request_sha256=public_request_sha256,configuration_check=_public_configuration)


def _public_configuration(configuration):
    from .search_over_raw import SearchConfiguration
    require(type(configuration) is SearchConfiguration and configuration.arm == "reference"
        and configuration.belief == "public" and configuration.leaf in {"model", "hp_fraction"}
        and configuration.workers == 20 and configuration.seconds in (1., 3., 10.),
        "distinct public reference timed model/HP configuration required")


def _read_reference_timed_selection(selected, *, configuration, runtime_configuration, runtime_sha256,
        root_id, statistics_root_id, selection_seed, consumed_binding,legal_action_mask,
        public_request_sha256,configuration_check):
    from .search_over_raw_leaves import validate_leaf_work, ROLLOUT_CAP
    configuration_check(configuration)
    require(type(runtime_configuration) is dict and digest(runtime_configuration) == runtime_sha256
        and digest(runtime_configuration.get("configuration")) == digest(asdict(configuration))
        and runtime_configuration.get("root_statistics") == "independent_per_root"
        and runtime_configuration.get("incumbent") is None
        and type(runtime_configuration.get("initial_dispatch_workers")) is int
        and runtime_configuration["initial_dispatch_workers"] == 6
        and "fixed_work_diagnostic" not in runtime_configuration
        and "visited_value_diagnostics" not in runtime_configuration
        and digest(runtime_configuration.get("timed_reference_atomic_work")) == digest(dict(schema=SCHEMA,
            zero_work_not_scored=True, fresh_single_attempt=True,
            on_time_deployability_established=False, runtime_qualified=False)),
        "reference timed distinct runtime drift")
    require(digest(runtime_configuration.get("consumed_world_inputs")) == digest(dict(
        schema="pokezero.search-over-raw.consumed-world-inputs.v1", initial_inputs_only=True,
        snapshot_fields_excluded=[], ownership_metadata_excluded=["battle_token", "battle_id"],
        instrumentation_can_change_deadlines=True, qualifies_uninstrumented_runtime=False))
        and digest(runtime_configuration.get("reference_atomic_work")) == digest(dict(
            schema="pokezero.search-over-raw.reference-atomic-work.v1",
            scope="actual_own_backup_and_cleanup; not independent simulator labels",
            instrumentation_can_change_deadlines=True, runtime_qualified=False)),
        "reference timed observation scope drift")
    require((public_request_sha256 is not None) == ("full_public_request_observation" in runtime_configuration)
        and (public_request_sha256 is None or digest(runtime_configuration["full_public_request_observation"])
            == digest(TIMED_REFERENCE_PUBLIC_CAPTURE)), "reference full request scope/digest drift")
    if configuration.leaf == "hp_fraction":
        require(digest(runtime_configuration.get("reference_leaf")) == digest(dict(leaf="hp_fraction",
            rollout_cap=ROLLOUT_CAP, rollout_policy="raw_argmax_both_seats",
            tree="unchanged_trajectory_reference", priors="unchanged_champion")),
            "reference timed HP leaf runtime drift")
    else:
        require("reference_leaf" not in runtime_configuration, "reference timed model leaf override")
    mask = tuple(legal_action_mask)
    require(mask and all(type(v) is bool for v in mask) and sum(mask) >= 2
        and type(statistics_root_id) is str and bool(statistics_root_id)
        and root_id == statistics_root_id+":"+configuration.identity
        and digest({k: selected.get(k) for k in ("root_id", "statistics_root_id",
            "selection_seed", "configuration_sha256", "runtime_sha256")}) == digest(dict(
                root_id=root_id, statistics_root_id=statistics_root_id, selection_seed=selection_seed,
                configuration_sha256=configuration.identity, runtime_sha256=runtime_sha256))
        and selected.get("independent_root_statistics") is True
        and selected.get("scientific_strength_evidence") is False
        and _finite(selected.get("elapsed_seconds"))
        and type(selected.get("nominal_search_seconds")) in (int, float)
        and selected["nominal_search_seconds"] == configuration.seconds
        and selected.get("exceeded_nominal_seconds") is (selected["elapsed_seconds"] > configuration.seconds),
        "reference timed selected external root/clock/mask drift")
    evidence = selected.get("evidence")
    require(type(evidence) is dict, "reference timed raw evidence required")
    readout = validate_reference_timed_inputs(evidence, binding=consumed_binding,
        battle_id="search-over-raw-root:"+statistics_root_id, selection_seed=selection_seed,
        nominal_seconds=configuration.seconds, public_request_sha256=public_request_sha256)
    root = evidence["worker_receipts"][0]["evidence"]["atomic_work"]["root"]
    require(digest(root["actions"]) == digest(["action:"+str(i) for i,v in enumerate(mask) if v]),
        "reference timed root actions differ from external legal mask")
    require(digest(evidence.get("timed_reference_readout")) == digest(readout)
        and selected.get("status") == readout["status"] and "action" in selected
        and selected["elapsed_seconds"] + 1e-8 >= evidence["reference_timed_prefix"]["decision_finished_monotonic"]
            - evidence["reference_timed_prefix"]["adapter_started_monotonic"],
        "reference timed cached accounting/status drift")
    if readout["accepted_units"]:
        action = selected["action"]
        require(type(action) is int and 0 <= action < len(mask) and mask[action]
            and _record(evidence["result"])["action"] == "action:"+str(action),
            "reference timed action differs from fresh max-visit work")
    else:
        require(selected["action"] is None and selected.get("attempt_consumed") is True
            and all(selected.get(k) is False for k in ("retry_authorized", "continuation_authorized",
                "runtime_qualified")), "reference timed zero result must be unscored and consumed")
    if configuration.leaf == "hp_fraction":
        receipts = [dict(receipt, batch=SimpleNamespace(**_record(receipt["batch"])))
            for receipt in evidence["worker_receipts"]]
        validate_leaf_work(SimpleNamespace(worker_receipts=receipts), configuration.leaf)
    return readout
