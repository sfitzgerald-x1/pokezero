"""Read-only fixed attempt-roster cost reconciliation, never work validation.

Only bound original claims/launches/terminal receipts are read. No pickle is
loaded, process queried, attempt created, failure retried or outcome scored.
Measured descendant-inclusive CPU is counted at a nonoverlapping frontier;
failed/absent/malformed entries and uncovered components remain explicit.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import signal
from pathlib import Path

from .search_over_raw import digest,require
from .search_over_raw_continuation_admission import _path
from .search_over_raw_outcomes import _sha
from .search_over_raw_resources import (measurement_protocol,read_completed_process_cost,
    validate_memory_receipt,read_sampling_cpu_components)
from .search_over_raw_nested_ownership import (_read, _detached, MAX_CHILDREN, MAX_DEPTH,
    verify_delegated_closure, _clean_diagnostic_incomplete, delegated_close_value)
from .search_over_raw_supervision import _verify_closed_generations

KINDS={"fixed-work":"DIAGNOSTIC", "outcome-run":"OUTCOME_RUN", "boundary-run":"BOUNDARY_RUN",
    "materialization-run":"MATERIALIZATION_RUN", "native-timed-run":"NATIVE_TIMED_RUN",
    "actor-annotated-native-timed-run":"ACTOR_ANNOTATED_NATIVE_TIMED_RUN",
    "native-oracle-timed-run":"NATIVE_ORACLE_TIMED_RUN", "reference-timed-run":"REFERENCE_TIMED_RUN",
    "reference-oracle-timed-run":"REFERENCE_ORACLE_TIMED_RUN", "raw-selection-run":"RAW_SELECTION_RUN",
    "staged-exploration-run":"STAGED_EXPLORATION_RUN"}
# Bounded read-only inventory, not a new producer/scientific-work allocation.
# A declared forest can contain several existing per-owner inventories.
MAX_COST_ATTEMPTS=MAX_CHILDREN*(MAX_DEPTH+1)


@dataclass(frozen=True)
class AttemptCostSpec:
    directory: str
    job_sha256: str
    binding_sha256: str
    kind: str
    work_phase: str
    contract: dict
    resource_source_sha256: str
    effective_watchdog_seconds: float
    parent_directory: str | None=None

    def __post_init__(self):
        require(type(self.directory) is str and str(_path(self.directory))==self.directory
            and self.kind in KINDS and self.work_phase in {"selection","materialization","continuations","boundaries"}
            and all(_sha(v) for v in (self.job_sha256,self.binding_sha256,self.resource_source_sha256))
            and type(self.contract) is dict,"original typed cost-attempt bindings required")
        require(_finite(self.effective_watchdog_seconds) and self.effective_watchdog_seconds>0
            and _finite(self.contract.get("emergency_watchdog_seconds"))
            and self.effective_watchdog_seconds<=self.contract["emergency_watchdog_seconds"],
            "original effective watchdog required")
        require(self.parent_directory is None or type(self.parent_directory) is str
            and str(_path(self.parent_directory))==self.parent_directory and self.parent_directory!=self.directory,
            "canonical distinct declared parent required")
        object.__setattr__(self,"contract",deepcopy(self.contract))
        digest(asdict(self))


def _json(path):
    path=_path(path)
    require(path.is_file() and path.stat().st_size<=16*1024*1024,"bounded cost evidence required")
    def pairs(rows):
        require(len({k for k,_ in rows})==len(rows),"duplicate cost evidence field")
        return dict(rows)
    def invalid(value):raise ValueError("nonfinite cost evidence")
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=invalid)


def _inventory(paths):
    snapshot={}
    for directory in paths:
        directory=_path(directory)
        snapshot[str(directory)]=(directory.is_dir(),directory.is_file())
        if directory.exists():
            require(directory.is_dir(),"attempt namespace is not a directory")
            for path in directory.rglob("*"):
                path=_path(path)
                require(path.is_file() or path.is_dir(),"unsupported cost evidence node")
                snapshot[str(path)]="directory" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def _attempt(spec):
    directory=_path(spec.directory)
    row=dict(directory=spec.directory,parent_directory=spec.parent_directory,state="UNSTARTED_UNOBSERVED",
        original_supervisor_status=None,original_supervisor_exit_code=None,actual_worker_exit_code=None,
        worker_reported_exit_code=None,worker_declared_cleanup_pending=None,
        worker_pid=None,job_cpu_seconds_including_reaped_children=None,supervisor_thread_cpu_seconds=None,
        sampling_cpu_components=None,elapsed_job_wall_seconds=None,phase_costs=None,
        normal_descendant_closure_validated=False,control_descendant_closure_validated=False,
        measurement_window_cpu_wall_complete=False,
        evidence_error_type=None,work_validated=False,runtime_qualified=False,resource_cost_complete=False)
    if not directory.exists():return row,None,None
    claim_path=directory/"claim.json"
    require(claim_path.is_file(),"occupied attempt lacks original claim")
    claim=_json(claim_path)
    require(claim["job_sha256"]==spec.job_sha256==hashlib.sha256((directory/"job.pickle").read_bytes()).hexdigest()
        and claim["binding_sha256"]==spec.binding_sha256 and digest(claim["contract"])==digest(spec.contract)
        and claim["retry_authorized"] is False and type(claim["synthetic_process_fixture"]) is bool
        and digest(claim["resource_measurement_protocol"])==digest(measurement_protocol())
        and claim["resource_measurement_source_sha256"]==spec.resource_source_sha256,"original attempt claim/job/source drift")
    row["state"]="CLAIMED_NO_LAUNCH_OBSERVED"
    launch=None
    if (directory/"launch.json").exists():
        launch=_json(directory/"launch.json")
        require(type(launch) is dict and set(launch)=={"pid","group","birth_identity","supervisor_pid","binding_sha256"}
        and type(launch["pid"]) is int and launch["pid"]>0 and type(launch["group"]) is int and launch["group"]==launch["pid"]
        and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"]>0 and launch["supervisor_pid"]!=launch["pid"]
        and type(launch["birth_identity"]) is str and launch["birth_identity"]
            and launch["binding_sha256"]==spec.binding_sha256,"bound actual cost launch required")
        row.update(state="LAUNCHED_TERMINAL_UNOBSERVED",worker_pid=launch["pid"])
    if not (directory/"terminal.json").exists():return row,launch,None
    terminal=_json(directory/"terminal.json")
    require(terminal["schema"]==f"pokezero.search-over-raw.{spec.kind}-supervision.v1"
        and terminal["binding_sha256"]==spec.binding_sha256
        and (terminal["worker_pid"] is None if launch is None and terminal["worker_pid"] is None else
            type(terminal["worker_pid"]) is int and terminal["worker_pid"]>0
            and (launch is None or terminal["worker_pid"]==launch["pid"]))
        and type(terminal["supervisor_exit_code"]) is int and terminal["supervisor_exit_code"] in {0,1,124}
        and terminal["cpu_scope"]=="isolated job wait4; reaped descendants only"
        and terminal["supervisor_cpu_scope"]=="calling thread through receipt construction; separate from job and ps CPU"
        and terminal["monitoring_wall_overlaps_job"] is True and terminal["resource_cost_complete"] is False
        and terminal["maxrss_scope"]=="wait4 job high-water mark; not aggregate tree peak"
        and terminal["maxrss_units"] in {"bytes","KiB"}
        and terminal["peak_tree_memory_measured"] is False and terminal["runtime_qualified"] is False
        and terminal["deployment_eligible"] is False and terminal["retries_permitted"] is False
        and terminal["holdout_authorized"] is False
        and _finite(terminal["watchdog_seconds"]) and terminal["watchdog_seconds"]==spec.effective_watchdog_seconds
        and (spec.effective_watchdog_seconds==spec.contract["emergency_watchdog_seconds"]
            or claim["synthetic_process_fixture"] is True)
        and _finite(terminal["elapsed_wall_seconds"]) and _finite(terminal["supervisor_thread_cpu_seconds"]),
        "original supervision cost scope drift")
    code=terminal["worker_exit_code"];cpu=terminal["cpu_seconds_including_reaped_children"]
    require(code is None and cpu is None and terminal["maxrss"] is None or
        type(code) is int and _finite(cpu) and _finite(terminal["maxrss"]),"wait4 exit/usage cannot be invented independently")
    success=terminal["supervisor_exit_code"]==0
    require(type(terminal["watchdog_reached"]) is bool and type(terminal["cpu_wall_cost_complete"]) is bool,
        "exact original supervision flags required")
    worker=None
    if success:
        require(launch is not None and code==0 and terminal["status"]==KINDS[spec.kind]+"_COMPLETE"
            and terminal["error_type"] is None and terminal["cleanup_error_type"] is None
            and terminal["watchdog_reached"] is False and terminal["cpu_wall_cost_complete"] is True,
            "normal original success scope required")
        worker=_json(directory/"worker-result.json")
        require(worker["binding_sha256"]==spec.binding_sha256 and type(worker["worker_pid"]) is int
            and worker["worker_pid"]==launch["pid"] and type(worker["exit_code"]) is int
            and worker["exit_code"]==0 and worker["error_type"] is None,"normal original worker receipt required")
        outcomes=[]
        verify_delegated_closure(directory,launch["pid"],spec.binding_sha256,_verify_closed_generations,
            depth=0 if spec.parent_directory is None else _json(directory/"nested-detach.json")["depth"],
            outcomes=outcomes)
        row["control_descendant_closure_validated"]=True
        row["normal_descendant_closure_validated"]=not outcomes
        row["phase_costs"]=read_completed_process_cost(terminal,worker,work_phase=spec.work_phase)["phase_costs"]
    else:
        require(terminal["cpu_wall_cost_complete"] is False and terminal["phase_costs"] is None
            and terminal["observed_work_accounting"] is None and terminal["result"] is None
            and type(terminal["error_type"]) is str and terminal["error_type"],"failure cannot become successful measured work")
        if terminal["supervisor_exit_code"]==124:
            require(terminal["watchdog_reached"] is True and terminal["cleanup_error_type"] is None
                and terminal["status"]==KINDS[spec.kind]+"_INCOMPLETE","original watchdog failure drift")
        else:require(terminal["status"]=="OPERATIONAL_STOP","original operational failure drift")
        if launch is None:
            require(terminal["supervisor_exit_code"]==1 and terminal["watchdog_reached"] is False
                and terminal["memory_measurement"] is None,
                "unpublished launch cannot become normal completion or watchdog work")
            if terminal["worker_pid"] is None:
                require(code is None and terminal["cleanup_error_type"] is None
                    and terminal["owned_groups"]==[] and not (directory/"worker-result.json").exists(),
                    "prelaunch failure cannot fabricate worker usage")
        if (directory/"worker-result.json").exists():
            worker=_json(directory/"worker-result.json")
            pending_cleanup=(code == -signal.SIGKILL and type(worker.get("exit_code")) is int
                and worker["exit_code"] == 1 and worker.get("cleanup_pending") is True
                and type(worker.get("error_type")) is str and bool(worker["error_type"])
                and "result" in worker and worker["result"] is None)
            require(launch is not None and worker["binding_sha256"]==spec.binding_sha256
                and type(worker["worker_pid"]) is int and worker["worker_pid"]==launch["pid"]
                and type(worker["exit_code"]) is int and worker["exit_code"] in {0,1}
                and (worker["exit_code"]==code or pending_cleanup),
                "failed worker result lost original launch/wait binding")
    sampling=None
    if terminal["memory_measurement"] is not None:
        require(launch is not None and terminal["resource_measurement_error_type"] is None,
            "conflicting original sampling result/error")
        validate_memory_receipt(directory,binding=spec.binding_sha256,main_launch=launch,expected_summary=terminal["memory_measurement"])
        sampling=read_sampling_cpu_components(_json(directory/"memory-trace.json"))
    else:require(not success and terminal["resource_measurement_error_type"] is None or
        type(terminal["resource_measurement_error_type"]) is str and terminal["resource_measurement_error_type"],
        "missing sampler measurement must remain explicit")
    row.update(state=("PRELAUNCH_FAILURE_COST_OBSERVED" if terminal["worker_pid"] is None else
        "UNPUBLISHED_LAUNCH_FAILURE_COST_OBSERVED") if launch is None else
        "REAPED_SUCCESS_COST_OBSERVED" if success else "REAPED_FAILURE_COST_OBSERVED" if code is not None
        else "TERMINAL_WITHOUT_WAIT4_USAGE",original_supervisor_status=terminal["status"],
        original_supervisor_exit_code=terminal["supervisor_exit_code"],actual_worker_exit_code=code,
        worker_reported_exit_code=None if worker is None else worker["exit_code"],
        worker_declared_cleanup_pending=None if worker is None else worker.get("cleanup_pending",False),
        job_cpu_seconds_including_reaped_children=cpu,supervisor_thread_cpu_seconds=terminal["supervisor_thread_cpu_seconds"],
        sampling_cpu_components=sampling,elapsed_job_wall_seconds=terminal["elapsed_wall_seconds"],
        worker_pid=terminal["worker_pid"],wait4_usage_join_validated=launch is not None and code is not None,
        measurement_window_cpu_wall_complete=terminal["cpu_wall_cost_complete"])
    return row,launch,terminal


def reconcile_attempt_cost_roster(attempts,*,attempt_root,roster_sha256):
    """Every externally frozen slot remains; no receipt-directory inference.

    The hash binds supplied roster bytes, not independent review or scientific
    authority. Delegation discovery validates its completeness; it cannot add
    omitted jobs. Cost invalidity never rewrites original scientific evidence.
    """
    require(type(attempts) is tuple and 0<len(attempts)<=MAX_COST_ATTEMPTS and all(type(s) is AttemptCostSpec for s in attempts),
        "nonempty original typed attempt roster required")
    root=_path(attempt_root)
    require(_sha(roster_sha256) and roster_sha256==digest([asdict(s) for s in attempts]),"original fixed cost roster drift")
    directories={s.directory:s for s in attempts}
    require(len(directories)==len(attempts) and all(_path(p).is_relative_to(root) and _path(p)!=root for p in directories),
        "distinct original attempts within fixed canonical namespace required")
    require(all(s.parent_directory is None or s.parent_directory in directories for s in attempts),"declared cost parent omitted")
    # Validate the prospective forest before any owner callback or file read.
    depths={}
    for spec in attempts:
        seen=set();current=spec
        while current.parent_directory is not None:
            require(current.directory not in seen and len(seen)<MAX_DEPTH,"declared attempt cost cycle/depth drift")
            seen.add(current.directory);current=directories[current.parent_directory]
        depths[spec.directory]=len(seen)
    before_specs=digest([asdict(s) for s in attempts]);before=_inventory(directories)
    rows={};launches={};terminals={};errors=[]
    for spec in attempts:
        try:row,launch,terminal=_attempt(spec)
        except (ValueError,OSError,KeyError,TypeError) as error:
            row=dict(directory=spec.directory,parent_directory=spec.parent_directory,state="MALFORMED_EVIDENCE",
                job_cpu_seconds_including_reaped_children=None,supervisor_thread_cpu_seconds=None,
                sampling_cpu_components=None,elapsed_job_wall_seconds=None,evidence_error_type=type(error).__name__)
            launch=terminal=None;errors.append(dict(directory=spec.directory,error_type=type(error).__name__))
        rows[spec.directory]=row;launches[spec.directory]=launch;terminals[spec.directory]=terminal
    generations={}
    for path,launch in launches.items():
        if launch is not None:
            generation=(launch["pid"],launch["birth_identity"],launch["supervisor_pid"])
            require(generation not in generations,"duplicate original cost launch generation")
            generations[generation]=path
    witnessed=set();graph_errors=[]
    for parent in attempts:
        registry=_path(Path(parent.directory)/"delegated-groups")
        for path in sorted(registry.glob("child-*.json")):
            try:
                require(launches[parent.directory] is not None,"delegation lacks original parent launch")
                row,child=_read(path,_path(parent.directory),launches[parent.directory]["pid"],parent.binding_sha256,depths[parent.directory])
                require(str(child) in directories and directories[str(child)].parent_directory==parent.directory,
                    "actual delegated attempt omitted or assigned another parent")
                require(launches[str(child)] is not None and row["binding_sha256"]==directories[str(child)].binding_sha256
                    and str(child) not in witnessed and _detached(row,child,path),"cost delegation launch/detach drift")
                witnessed.add(str(child))
                closed=path.with_name("closed-"+path.name)
                if terminals[str(child)] is not None:
                    value=_json(closed)
                    clean=value.get("clean_diagnostic_incomplete") is True
                    if clean: require(_clean_diagnostic_incomplete(row,child,terminals[str(child)]),
                        "original clean diagnostic cost join drift")
                    require(digest(value)==digest(delegated_close_value(path,row,child,terminals[str(child)],clean_incomplete=clean)),
                        "actual parent wait/closure cost join drift")
            except (ValueError,OSError,KeyError,TypeError) as error:
                graph_errors.append(dict(parent_directory=parent.directory,error_type=type(error).__name__))
    for spec in attempts:
        if spec.parent_directory is not None and launches[spec.directory] is not None and spec.directory not in witnessed:
            graph_errors.append(dict(parent_directory=spec.parent_directory,error_type="MissingDelegationWitness"))
    children={p:[] for p in directories}
    for spec in attempts:
        if spec.parent_directory is not None:children[spec.parent_directory].append(spec.directory)
    frontier=[]
    def visit(path):
        if launches[path] is not None and rows[path]["job_cpu_seconds_including_reaped_children"] is not None:frontier.append(path)
        else:
            for child in children[path]:visit(child)
    for spec in attempts:
        if spec.parent_directory is None:visit(spec.directory)
    job_cpu=None if graph_errors or not frontier else sum(rows[path]["job_cpu_seconds_including_reaped_children"] for path in frontier)
    require(before_specs==digest([asdict(s) for s in attempts]) and before==_inventory(directories),
        "original attempt cost bytes, roster or namespace changed during reconciliation")
    observed=[r for r in rows.values() if r["elapsed_job_wall_seconds"] is not None]
    return dict(schema="pokezero.search-over-raw.attempt-roster-cost.v1",roster_sha256=roster_sha256,
        attempt_root=str(root),required_attempt_slots=len(attempts),attempts=[rows[s.directory] for s in attempts],
        evidence_errors=errors,delegation_errors=graph_errors,measured_cpu_frontier=frontier if not graph_errors else [],
        known_nonoverlapping_job_cpu_seconds=job_cpu,nonfrontier_job_cpu_added_again=False,
        nested_supervisor_or_sampler_cpu_added_again=False,
        outer_observer_components=[rows[s.directory] for s in attempts if s.parent_directory is None],
        job_wall_exposure_seconds=sum(r["elapsed_job_wall_seconds"] for r in observed),
        job_wall_exposure_is_campaign_elapsed=False,campaign_elapsed_wall_seconds=None,total_workflow_cpu_seconds=None,
        uncovered_cost_scope=["unobserved attempt usage and unreaped/unknown descendant CPU",
            "emergency ownership queries and late sampler reaps absent from historical RSS traces",
            "preclaim planning, final serialization and external observer overlap not measured by job wait4",
            "instantaneous or physical whole-tree memory peak"],
        supplied_roster_only=True,attempt_population_exhaustiveness_validated=False,
        roster_review_validated=False,work_validated=False,scientific_collection_authorized=False,
        holdout_authorized=False,retries_permitted=False,resource_cost_complete=False,complete_workflow_cost=False)
