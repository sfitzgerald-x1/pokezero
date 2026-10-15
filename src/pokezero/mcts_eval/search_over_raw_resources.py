"""Observed owned-process RSS, never an exact or physical tree-memory peak.

One bounded process-table query per sample; no command lines are retained.
The supervisor alone owns its job wait4. Sampler subprocess CPU is separately
reaped and must never be added to that job's descendant-inclusive CPU again.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from .search_over_raw import digest, require

SCHEMA = "pokezero.search-over-raw.sampled-rss.v1"
# Retain actual unreaped Popen objects; their destructor must not steal wait
# ownership. A pending query prevents another query in this local supervisor.
_PENDING_SAMPLERS = {}


def measurement_protocol():
    return dict(schema=SCHEMA, cadence_seconds=.25, query_timeout_seconds=.2,
        metric="maximum_observed_process_snapshot_rss_sum_bytes",
        population="admitted_groups_and_observed_descendants_by_pid_and_birth",
        process_table_atomic=False, shared_pages_counted_per_process=True,
        exact_peak_measured=False, physical_memory_measured=False,
        true_peak_lower_bound_claim=False, resource_cost_complete=False)


def resource_source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_completed_process_cost(terminal, worker, *, work_phase):
    """Read costs only AFTER a distinct owning reader validates the run.

    This function checks arithmetic/scope, not provenance. A successful wait4
    covers the job and reaped descendants; supervisor and sampler costs are
    separate. Sampled RSS and a job high-water mark are never a tree peak.
    """
    from .search_over_raw_supervision import _phase_costs
    finite = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    require(terminal["cpu_wall_cost_complete"] is True and terminal["resource_cost_complete"] is False
        and terminal["peak_tree_memory_measured"] is False and terminal["runtime_qualified"] is False
        and terminal["cpu_scope"] == "isolated job wait4; reaped descendants only"
        and terminal["supervisor_cpu_scope"] == "calling thread through receipt construction; separate from job and ps CPU"
        and terminal["monitoring_wall_overlaps_job"] is True
        and terminal["maxrss_scope"] == "wait4 job high-water mark; not aggregate tree peak"
        and terminal["maxrss_units"] in {"bytes", "KiB"}
        and all(finite(terminal[key]) for key in ("elapsed_wall_seconds",
            "cpu_seconds_including_reaped_children", "supervisor_thread_cpu_seconds", "maxrss")),
        "normal owning-process CPU/wall scope and finite costs required")
    phases = _phase_costs(worker["phases"], worker["terminal_monotonic_seconds"], work_phase)
    reported = terminal["phase_costs"]
    require(type(reported) is dict and set(reported) == set(phases) | {
        "process_bootstrap_wall_seconds", "receipt_reap_and_verification_wall_seconds"}
        and all(reported[key] == value for key, value in phases.items())
        and all(finite(value) for value in reported.values())
        and sum(reported.values()) <= terminal["elapsed_wall_seconds"]
        and sum(phases.values()) + reported["process_bootstrap_wall_seconds"] <= terminal["watchdog_seconds"],
        "owning-process phase cost accounting drift")
    memory = terminal["memory_measurement"]
    sampler = None if memory is None else memory["sampling_child_cpu_seconds"]
    require(sampler is None or finite(sampler), "finite separately reaped sampler CPU required")
    return dict(elapsed_wall_seconds=terminal["elapsed_wall_seconds"], phase_costs=deepcopy(reported),
        job_cpu_seconds_including_reaped_children=terminal["cpu_seconds_including_reaped_children"],
        supervisor_thread_cpu_seconds=terminal["supervisor_thread_cpu_seconds"],
        sampling_child_cpu_seconds=sampler, job_cpu_scope=terminal["cpu_scope"],
        supervisor_cpu_scope=terminal["supervisor_cpu_scope"],
        sampler_cpu_scope="ps wait4 only; separate from job CPU",
        job_maxrss=terminal["maxrss"], job_maxrss_units=terminal["maxrss_units"],
        job_maxrss_scope=terminal["maxrss_scope"], memory_measurement=deepcopy(memory),
        resource_measurement_error_type=terminal["resource_measurement_error_type"],
        exact_tree_peak_measured=False, complete_workflow_cost=False, resource_cost_complete=False)


class SamplingFailure(RuntimeError):
    def __init__(self, reason, cpu_seconds=None):
        super().__init__(reason)
        self.reason, self.cpu_seconds = reason, cpu_seconds


def _pending_samplers():
    from .search_over_raw_observer_cost import drain_pending_queries,pending_rss_query_pids,record_legacy_pending_reap
    drain_pending_queries()
    for pid,process in list(_PENDING_SAMPLERS.items()):
        try: child,status,usage = os.wait4(pid,os.WNOHANG)
        except ChildProcessError:
            # Wait ownership is lost: never signal a possibly reused PID.
            continue
        if child:
            process.returncode = os.waitstatus_to_exitcode(status)
            record_legacy_pending_reap(pid,status,usage)
            del _PENDING_SAMPLERS[pid]
    return sorted(set(_PENDING_SAMPLERS)|set(pending_rss_query_pids()))


def _process_table(timeout):
    """Bound and reap only our ps child, retaining no unrelated process rows."""
    require(type(timeout) in (int, float) and 0 < timeout <= .2, "bounded RSS query required")
    if _pending_samplers(): raise SamplingFailure("pending_sampler_cleanup")
    from .search_over_raw_observer_cost import current_observer_window,run_owned_ps
    if current_observer_window() is not None:
        try:
            result,cpu=run_owned_ps(["ps", "-axo", "pid=,ppid=,pgid=,rss=,stat=,lstart="],timeout=timeout)
        except subprocess.TimeoutExpired as error:raise SamplingFailure(error.query_reason,error.cpu_seconds) from error
        except RuntimeError as error:
            if hasattr(error,"query_reason"):raise SamplingFailure(error.query_reason,error.cpu_seconds) from error
            raise
        if result.returncode!=0 or result.stderr:raise SamplingFailure("process_table_unavailable",cpu)
        return _parse_process_table(result.stdout.encode(),cpu),cpu
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["ps", "-axo", "pid=,ppid=,pgid=,rss=,stat=,lstart="],
            stdout=output, stderr=errors)
        limit = time.perf_counter() + timeout
        timed_out = False
        while True:
            child, status, usage = os.wait4(process.pid, os.WNOHANG)
            if child:
                break
            if time.perf_counter() >= limit:
                timed_out = True
                # Exact unreaped child, never a discovered or reused PID.
                try: os.kill(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                # SIGKILL does not guarantee prompt reap (e.g. uninterruptible
                # I/O). Never block the scientific job's emergency supervisor.
                until = time.perf_counter()+.02
                while True:
                    child,status,usage = os.wait4(process.pid,os.WNOHANG)
                    if child: break
                    if time.perf_counter() >= until:
                        _PENDING_SAMPLERS[process.pid] = process
                        raise SamplingFailure("query_timeout_cleanup_pending")
                    time.sleep(.002)
                break
            time.sleep(.002)
        process.returncode = os.waitstatus_to_exitcode(status)
        cpu = usage.ru_utime + usage.ru_stime
        if timed_out: raise SamplingFailure("query_timeout", cpu)
        errors.seek(0)
        if process.returncode != 0 or errors.read(1):
            raise SamplingFailure("process_table_unavailable", cpu)
        output.seek(0); payload = output.read(16*1024*1024+1)
        if len(payload) > 16*1024*1024:
            raise SamplingFailure("process_table_too_large", cpu)
        return _parse_process_table(payload,cpu),cpu


def _parse_process_table(payload,cpu):
    rows=[]
    try:
        for line in payload.decode().splitlines():
            parts=line.split()
            require(len(parts)==10,"RSS process row drift")
            pid,parent,group,rss=map(int,parts[:4])
            rows.append(dict(pid=pid,parent_pid=parent,group=group,rss_bytes=rss*1024,state=parts[4],birth=" ".join(parts[5:])))
        _rows(rows)
    except (ValueError,UnicodeError):raise SamplingFailure("process_table_schema_drift",cpu) from None
    return rows


def _rows(rows):
    require(type(rows) is list, "RSS process rows required")
    pids = set()
    for row in rows:
        require(type(row) is dict and set(row) == {"pid", "parent_pid", "group", "rss_bytes", "state", "birth"}
            and all(type(row[k]) is int and row[k] >= 0 for k in ("pid", "parent_pid", "group", "rss_bytes"))
            and row["pid"] > 0 and row["pid"] not in pids and row["group"] > 0
            and type(row["state"]) is str and row["state"]
            and type(row["birth"]) is str and row["birth"] == " ".join(row["birth"].split())
            and row["birth"], "RSS identity, units or duplicate PID drift")
        pids.add(row["pid"])


def _generations(directory, main, binding):
    """All immutable launches, including closed short-lived generations."""
    generations = {"job":dict(pid=main["pid"], group=main["group"],
        birth=" ".join(main["birth_identity"].split()), launch_sha256=digest(main))}
    owned = directory / "owned-groups"
    paths = sorted(owned.glob("group-*.json"))
    require(len(paths) <= 20, "RSS owned generation roster drift")
    for path in paths:
        row = json.loads(path.read_bytes())
        require(row.get("protocol") == "active-generations-v1"
            and row.get("controller_pid") == main["pid"] and row.get("registration_sha256") == binding
            and type(row.get("pid")) is int and row["pid"] > 0 and row.get("group") == row["pid"]
            and type(row.get("generation")) is str and row["generation"]
            and type(row.get("birth_identity")) is str and row["birth_identity"]
            and path.name == f"group-{row['pid']}-{row['generation']}.json", "RSS launch binding drift")
        active = owned / "active" / path.name
        if active.exists(): require(os.path.samefile(active,path), "RSS active launch index drift")
        generations[path.name] = dict(pid=row["pid"], group=row["group"],
            birth=" ".join(row["birth_identity"].split()), launch_sha256=digest(row))
    return generations


def _select(rows, generations, observed):
    """Admitted leaders, their groups/children, and previously observed members.

    Birth mismatches for a reaped generation simply retire that anchor. This
    reader never signals processes. A new process reusing its PID is excluded.
    """
    _rows(rows)
    live = {r["pid"]:r for r in rows if not r["state"].startswith("Z")}
    anchors = {g["pid"] for g in generations.values() if g["pid"] in live
        and live[g["pid"]]["birth"] == g["birth"] and live[g["pid"]]["group"] == g["group"]}
    groups = {live[pid]["group"] for pid in anchors}
    chosen = {pid for pid,row in live.items() if row["group"] in groups
        or observed.get(pid) == row["birth"]}
    while True:
        new = {pid for pid,row in live.items() if row["parent_pid"] in chosen}
        if new <= chosen: break
        chosen.update(new)
    return [deepcopy(live[pid]) for pid in sorted(chosen)]


def _generation_table(generations, main):
    require(type(generations) is dict and "job" in generations and len(generations) <= 21,
        "RSS generation inventory required")
    for key,g in generations.items():
        require(type(key) is str and key and type(g) is dict
            and set(g) == {"pid", "group", "birth", "launch_sha256"}
            and type(g["pid"]) is int and g["pid"] > 0 and type(g["group"]) is int and g["group"] == g["pid"]
            and type(g["birth"]) is str and g["birth"] and g["birth"] == " ".join(g["birth"].split())
            and type(g["launch_sha256"]) is str and len(g["launch_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in g["launch_sha256"]), "RSS generation schema drift")
    require(generations["job"] == dict(pid=main["pid"],group=main["group"],
        birth=" ".join(main["birth_identity"].split()),launch_sha256=digest(main)), "RSS main generation drift")


class OwnedRSSObserver:
    def __init__(self, directory, main_launch, binding, began):
        self.directory, self.main, self.binding = Path(directory), deepcopy(main_launch), binding
        self.began, self.samples, self.observed = began, [], {}
        self.next_sample = began
        self.sampling_child_cpu = 0.
        self.sampling_child_cpu_complete = True

    def sample(self, *, timeout=.2):
        start = time.perf_counter()
        sample = dict(ordinal=len(self.samples), started=start, ended=None,
            generations=None, rows=None, error=None, sampling_child_cpu_seconds=None)
        try:
            sample["generations"] = _generations(self.directory,self.main,self.binding)
            rows, cpu = _process_table(timeout)
            sample["sampling_child_cpu_seconds"] = cpu
            sample["rows"] = _select(rows,sample["generations"],self.observed)
            self.observed.update({r["pid"]:r["birth"] for r in sample["rows"]})
        except SamplingFailure as error:
            sample["error"] = error.reason
            sample["sampling_child_cpu_seconds"] = error.cpu_seconds
        except Exception as error:
            # Sampling does not waive the job watchdog or disrupt owned cleanup.
            # Export only error class, never process details or private inputs.
            sample["error"] = type(error).__name__
        sample["ended"] = time.perf_counter()
        if sample["sampling_child_cpu_seconds"] is None:
            self.sampling_child_cpu_complete = False
        else: self.sampling_child_cpu += sample["sampling_child_cpu_seconds"]
        self.samples.append(sample)
        self.next_sample = sample["ended"] + measurement_protocol()["cadence_seconds"]

    def finish(self, ended):
        try:
            generations = _generations(self.directory,self.main,self.binding)
            inventory_error = None
        except Exception as error:
            generations, inventory_error = None, type(error).__name__
        trace = dict(schema=SCHEMA, protocol=measurement_protocol(), binding_sha256=self.binding,
            main_launch=deepcopy(self.main), began=self.began, ended=ended,
            samples=deepcopy(self.samples), generations=generations, inventory_error=inventory_error,
            pending_sampler_cleanup_pids=_pending_samplers())
        return trace, summarize_memory_trace(trace)


def summarize_memory_trace(trace):
    """Separate reader: recompute all metrics; trace does not certify provenance."""
    saved = deepcopy(trace)
    require(type(trace) is dict and set(trace) == {"schema", "protocol", "binding_sha256", "main_launch",
        "began", "ended", "samples", "generations", "inventory_error", "pending_sampler_cleanup_pids"}
        and trace["schema"] == SCHEMA and trace["protocol"] == measurement_protocol(), "RSS protocol drift")
    require(type(trace["samples"]) is list and all(type(trace[k]) in (int,float) and math.isfinite(trace[k])
        for k in ("began","ended")) and trace["ended"] >= trace["began"], "RSS clock drift")
    main = trace["main_launch"]
    require(type(main) is dict and set(main) == {"pid", "group", "birth_identity", "supervisor_pid", "binding_sha256"}
        and type(main["pid"]) is int and main["pid"] > 0 and type(main["group"]) is int and main["group"] == main["pid"]
        and type(main["supervisor_pid"]) is int and main["supervisor_pid"] > 0
        and type(main["birth_identity"]) is str and main["birth_identity"]
        and type(trace["binding_sha256"]) is str and len(trace["binding_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in trace["binding_sha256"])
        and main["binding_sha256"] == trace["binding_sha256"], "RSS main launch binding drift")
    generations = trace["generations"]
    require((generations is None and type(trace["inventory_error"]) is str)
        or (type(generations) is dict and "job" in generations and trace["inventory_error"] is None),
        "RSS final inventory required")
    if generations is not None: _generation_table(generations,main)
    pending = trace["pending_sampler_cleanup_pids"]
    require(type(pending) is list and all(type(pid) is int and pid > 0 for pid in pending)
        and pending == sorted(set(pending)), "RSS pending sampler cleanup roster drift")
    seen, previous, totals, errors, cpu = set(), {}, [], [], 0.
    cpu_complete = True; last = trace["began"]; gaps = []; durations = []
    for i,sample in enumerate(trace["samples"]):
        require(type(sample) is dict and set(sample) == {"ordinal", "started", "ended", "generations",
            "rows", "error", "sampling_child_cpu_seconds"} and type(sample["ordinal"]) is int
            and sample["ordinal"] == i and all(type(sample[k]) in (int,float) and math.isfinite(sample[k])
                for k in ("started","ended")) and last <= sample["started"] <= sample["ended"] <= trace["ended"],
            "RSS sample order/clock drift")
        gaps.append(sample["started"]-last); durations.append(sample["ended"]-sample["started"])
        last = sample["ended"]
        if sample["error"] is not None:
            require(type(sample["error"]) is str and sample["error"] and sample["rows"] is None,
                "RSS failed sample cannot report zero memory")
            errors.append(dict(ordinal=i,error=sample["error"]))
        else:
            _generation_table(sample["generations"],main)
            require(type(sample["generations"]) is dict and sample["generations"]
                and (generations is None or all(generations.get(k) == v for k,v in sample["generations"].items())),
                "RSS sample launch inventory drift")
            rows = sample["rows"]
            require(rows == _select(rows,sample["generations"],previous), "RSS unowned row or row-order drift")
            previous.update({r["pid"]:r["birth"] for r in rows})
            if rows: totals.append(sum(r["rss_bytes"] for r in rows))
            for key,g in sample["generations"].items():
                if any(r["pid"] == g["pid"] and r["birth"] == g["birth"] and r["group"] == g["group"] for r in rows):
                    seen.add(key)
        value = sample["sampling_child_cpu_seconds"]
        if value is None: cpu_complete = False
        else:
            require(type(value) in (int,float) and math.isfinite(value) and value >= 0,
                "RSS sampler CPU drift")
            cpu += value
    gaps.append(trace["ended"]-last)
    missed = None if generations is None else sorted(set(generations)-seen)
    require(digest(trace) == digest(saved), "RSS trace mutated during read")
    return dict(protocol=measurement_protocol(), trace_sha256=digest(trace),
        maximum_observed_rss_sum_bytes=max(totals) if totals else None,
        sample_count=len(trace["samples"]), successful_sample_count=len(trace["samples"])-len(errors),
        maximum_unsampled_gap_seconds=max(gaps), maximum_query_wall_seconds=max(durations,default=0.),
        observed_launch_generations=sorted(seen), missed_launch_generations=missed,
        errors=errors, inventory_error=trace["inventory_error"],
        sampling_child_cpu_seconds=cpu if cpu_complete else None,
        sampling_child_cpu_scope="ps wait4 only; separate from job CPU",
        pending_sampler_cleanup_pids=pending,
        all_declared_launch_generations_observed_without_query_errors=bool(totals) and not errors and missed == []
            and trace["inventory_error"] is None and not pending,
        independently_validated=False, resource_cost_complete=False)


def validate_memory_receipt(directory, *, binding, main_launch, expected_summary):
    """Re-read immutable launch inventory and raw samples, then compare metrics."""
    directory = Path(directory)
    payload = (directory / "memory-trace.json").read_bytes()
    trace = json.loads(payload)
    require(trace["binding_sha256"] == binding and trace["main_launch"] == main_launch
        and trace["generations"] == _generations(directory,main_launch,binding), "RSS owning launch join drift")
    require(summarize_memory_trace(trace) == expected_summary
        and (directory / "memory-trace.json").read_bytes() == payload, "RSS derived summary or trace drift")
    return digest(trace)


def read_sampling_cpu_components(trace):
    """Recover known reaped CPU without changing historical RSS summary shape.

    Unknown samples stay unknown; pending/later reaps and emergency ownership
    queries are not inferred from this RSS-only trace. No process is polled.
    """
    summary=summarize_memory_trace(trace)
    known=[row["sampling_child_cpu_seconds"] for row in trace["samples"]
        if row["sampling_child_cpu_seconds"] is not None]
    unknown=sum(row["sampling_child_cpu_seconds"] is None for row in trace["samples"])
    return dict(trace_sha256=summary["trace_sha256"],known_reaped_sampling_cpu_seconds=sum(known),
        unknown_sample_cpu_records=unknown,pending_sampler_cleanup_pids=deepcopy(trace["pending_sampler_cleanup_pids"]),
        sampling_cpu_seconds=summary["sampling_child_cpu_seconds"],
        rss_query_cost_window_complete=unknown==0 and not trace["pending_sampler_cleanup_pids"],
        emergency_ownership_query_cpu_seconds=None,late_sampler_reap_cpu_seconds=None,
        complete_workflow_cost=False,resource_cost_complete=False)


def derive_fixed_work_accounting(evidence, *, arm, binding, statistics_root_id, leaf,
        native_public_fold_required=False, expected_public_request_sha256=None):
    """Atomic counts from separate readers; leaf outcomes keep their scope.

    Missing evaluator receipts are unknown, never zero. A successful report
    covers the final exported search, not an interrupted worker with lost tail.
    Native multiplicity is already present in requested iterations: count each
    actual call once, not once per construction ordinal.
    """
    from .search_over_raw_world_inputs import validate_fixed_work_inputs
    validate_fixed_work_inputs(evidence,arm=arm,binding=binding,statistics_root_id=statistics_root_id,
        construction_required=native_public_fold_required,
        native_public_fold_required=native_public_fold_required,
        expected_public_request_sha256=expected_public_request_sha256)
    require(leaf in {"model", "hp_fraction", "raw_rollout"}, "known fixed-work evaluator required")
    attempted = accepted = cancelled = transitions = leaf_prices = tree_terminals = 0
    terminal_leaves = capped_leaves = refused_leaves = None
    cap_value_fallbacks = None
    if arm == "incumbent":
        from .search_over_raw_native_atomic_work import validate_native_report
        raw_reports = []
        for call in evidence["consumed_world_inputs"]["calls"]:
            report = call["native_report"]
            validated = validate_native_report(report,**call["native_call_parameters"])
            accepted += validated["completed_iterations"]
            for batch in report["atomic_work"]["rounds"]:
                paths = batch["attempted_paths"] if batch["status"] == "CANCELLED" else [c["path"] for c in batch["commits"]]
                attempted += len(paths); transitions += sum(len(p) for p in paths)
                if batch["status"] == "CANCELLED": cancelled += len(paths)
                else:
                    leaf_prices += len(batch["row_value_bits"])
                    for commit in batch["commits"]:
                        step = commit["path"][-1]
                        if commit["end"]["kind"] == "ready" and step["branch"] is not None:
                            branch = batch["after_collection"]["chances"][step["chance"]][step["branch"]]
                            tree_terminals += branch["terminal_bits"] is not None
            raw_reports.append(report)
        if leaf == "raw_rollout" and all(all(type(r.get(k)) is int and r[k] >= 0 for k in
                ("raw_leaf_terminal","raw_leaf_cap_fallbacks","raw_leaf_cancelled_rows")) for r in raw_reports):
            # Engine-reported work counters, NOT independently replayed labels.
            terminal_leaves = sum(r["raw_leaf_terminal"] for r in raw_reports)
            cap_value_fallbacks = sum(r["raw_leaf_cap_fallbacks"] for r in raw_reports)
        unit = "complete_native_iteration_with_committed_backup"
        transition_scope = "native selected path edges, not simulator plies"
    else:
        from .search_over_raw_atomic_work import validate_reference_atomic_work
        validated = validate_reference_atomic_work(evidence,binding=binding,
            battle_id="search-over-raw-root:"+statistics_root_id)
        accepted, attempted, transitions = (validated[k] for k in
            ("accepted_trajectories","attempted_draws","transitions"))
        evaluations = []
        full_leaf_ledger = True
        for worker in evidence["worker_receipts"]:
            for row in worker["evidence"]["atomic_work"]["rows"]:
                cancelled += row["status"] == "CANCELLED"
                if row["status"] == "COMMITTED":
                    leaf_prices += row["discovered_leaf"] is not None
                    tree_terminals += row["discovered_leaf"] is None
            ledger = worker["evidence"].get("leaf_ablation")
            if not isinstance(ledger,dict) or ledger.get("leaf") != leaf or not isinstance(ledger.get("evaluations"),list):
                full_leaf_ledger = False
            else: evaluations.extend(ledger["evaluations"])
        if leaf == "raw_rollout" and full_leaf_ledger:
            require(all(type(row) is dict and row.get("leaf") == leaf
                and row.get("status") in {"COMPLETE","DEADLINE_CANCELLED","REFUSED"}
                for row in evaluations), "reference reported leaf work drift")
            # Only committed discovered leaves are covered by these exports.
            # Extra root/failed evaluations cannot be silently promoted.
            if len(evaluations) == leaf_prices and all(r["status"] == "COMPLETE" for r in evaluations):
                terminal_leaves = len(evaluations)
                capped_leaves = refused_leaves = 0
        unit = "complete_reference_trajectory_with_committed_backup"
        transition_scope = "reference trajectory transitions, not native edges or rollout plies"
    require(attempted == accepted+cancelled, "atomic attempted/accepted/cancelled work drift")
    return dict(schema="pokezero.search-over-raw.observed-work-cost.v1",arm=arm,leaf=leaf,accepted_unit=unit,
        evidence_sha256=digest(evidence),attempted_units=attempted,accepted_units=accepted,cancelled_units=cancelled,
        observed_path_transitions=transitions,transition_scope=transition_scope,
        committed_leaf_prices=leaf_prices,tree_terminal_endings=tree_terminals,
        reported_terminal_rollout_leaves=terminal_leaves,reported_capped_rollout_leaves=capped_leaves,
        reported_refused_rollout_leaves=refused_leaves,
        reported_cap_value_fallbacks=cap_value_fallbacks,
        terminal_count_scope="backend-reported evaluator work; not independent terminal truth",
        cancelled_or_unreturned_leaf_prices_unknown=cancelled > 0,
        final_atomic_work_reconciled=True,independent_terminal_labels=False,
        interrupted_or_lost_final_receipt_covered=False,equal_cpu_or_work_claim=False,
        resource_cost_complete=False,runtime_qualified=False)
