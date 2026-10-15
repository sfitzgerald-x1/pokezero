"""Opt-in calling-thread and exclusively reaped own-ps costs.

No scientific work, process-table rows or command output is retained here.
Pending queries keep their original window identity. Closed windows are never
rewritten by late reaps; a later window only carries a linked observation.
"""
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import threading
import time

from .search_over_raw import digest,require

SCHEMA="pokezero.search-over-raw.external-observer-window.v1"
PUBLICATION_SCHEMA="pokezero.search-over-raw.observer-publication-cost.v1"
PURPOSES={"rss","ownership_freeze","ownership_identity","ownership_active_identities","ownership_group_liveness"}
_CURRENT=ContextVar("search_over_raw_observer_window",default=None)
_PURPOSE=ContextVar("search_over_raw_process_query_purpose",default="rss")
_PENDING={}
_LOST={}  # Retain Popen objects after ECHILD; never signal or repoll that PID.
_LATE={}  # Linked observations retained even if actual reap occurs outside a window.


def _finite(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def current_observer_window():
    window=_CURRENT.get()
    if window is not None:window.verify_owner()
    return window


class ObserverWindow:
    def __init__(self,label):
        require(label=="exploration_entry" or label=="authored_disposable_test","fixed observer window label required")
        self.label=label;self.pid=os.getpid();self.thread=threading.get_ident()
        self.window_id=hashlib.sha256(os.urandom(32)).hexdigest()
        self.queries=[];self.late_reaps=[];self.bindings={};self._binding_sha=digest({})
        self._closed=None;self._token=None
    def verify_owner(self):
        require(type(self) is ObserverWindow and self.pid==os.getpid() and self.thread==threading.get_ident()
            and self._closed is None and self._binding_sha==digest(self.bindings),"original observer PID/thread/binding drift")
    def bind(self,name,value):
        self.verify_owner()
        require(name in {"entry_job","journal_handle","outer_terminal"} and name not in self.bindings,
            "observer original input binding is one-shot")
        value=json.loads(json.dumps(value,allow_nan=False))
        self.bindings[name]=value;self._binding_sha=digest(self.bindings)
    def __enter__(self):
        require(_CURRENT.get() is None and self._token is None,"observer window already installed")
        self.verify_owner();self.started=time.perf_counter();self.cpu_started=time.thread_time()
        self._token=_CURRENT.set(self)
        return self
    def __exit__(self,kind,error,traceback):
        try:
            self.verify_owner();drain_pending_queries()
            ended=time.perf_counter();cpu_ended=time.thread_time()
            self._closed=dict(schema=SCHEMA,window_id=self.window_id,label=self.label,
                observer_pid=self.pid,observer_thread_id=self.thread,
                started_monotonic_seconds=self.started,ended_monotonic_seconds=ended,
                calling_thread_cpu_started_seconds=self.cpu_started,calling_thread_cpu_ended_seconds=cpu_ended,
                elapsed_wall_seconds=ended-self.started,calling_thread_cpu_seconds=cpu_ended-self.cpu_started,
                bindings=deepcopy(self.bindings),bindings_sha256=self._binding_sha,
                own_process_queries=deepcopy(self.queries),foreign_late_reap_observations=deepcopy(self.late_reaps),
                error_type=None if error is None else type(error).__name__,
                scope="entry calling thread after instrumentation import through final readback/failure; not nested job CPU",
                query_scope="own root-process ps children with actual wait4; RSS is an overlapping breakdown",
                cleanup_bound_seconds=.02,process_table_output_retained=False,
                final_measurement_serialization_included=False,nested_counters_added_again=False,
                full_process_cpu_measured=False,complete_workflow_cost=False,resource_cost_complete=False,
                runtime_qualified=False,scientific_collection_authorized=False,holdout_authorized=False)
            read_observer_window(self._closed)
        except BaseException as failure:
            if error is None:raise
            error.add_note("observer window close failed: "+type(failure).__name__)
        finally:_CURRENT.reset(self._token)
        return False
    def snapshot(self):
        require(self._closed is not None,"observer window has not closed")
        return deepcopy(self._closed)


class QueryPurpose:
    def __init__(self,purpose):require(purpose in PURPOSES,"known query purpose required");self.purpose=purpose
    def __enter__(self):self.token=_PURPOSE.set(self.purpose)
    def __exit__(self,*args):_PURPOSE.reset(self.token)


@dataclass
class _PendingQuery:
    process: object
    owner_pid: int
    owner_thread: int
    window_id: str
    query_id: str
    purpose: str
    original_record: dict


def _usage(usage):
    cpu=usage.ru_utime+usage.ru_stime
    require(_finite(cpu),"finite actual query wait4 CPU required")
    return cpu


def _reaped(pending,status,usage):
    pending.process.returncode=os.waitstatus_to_exitcode(status)
    observation=dict(origin_window_id=pending.window_id,query_id=pending.query_id,
        query_pid=pending.process.pid,owner_pid=pending.owner_pid,purpose=pending.purpose,
        status="REAPED_LATE",actual_exit_code=pending.process.returncode,
        actual_cpu_seconds=_usage(usage),observed_monotonic_seconds=time.perf_counter(),
        adopted_into_current_window_cpu=False)
    window=current_observer_window()
    # Only a still-open ORIGINAL window can complete its original record.
    if window is not None and window.window_id==pending.window_id:
        pending.original_record.update(state="REAPED",actual_exit_code=observation["actual_exit_code"],
            actual_cpu_seconds=observation["actual_cpu_seconds"],ended_monotonic_seconds=observation["observed_monotonic_seconds"])
    else:_LATE[pending.query_id]=(pending.owner_thread,observation)
    return observation


def drain_pending_queries():
    """Exact retained children only, one nonblocking wait; no historic PID scan."""
    observations=[]
    for query_id,pending in list(_PENDING.items()):
        if pending.owner_pid!=os.getpid() or pending.owner_thread!=threading.get_ident():continue
        try:child,status,usage=os.wait4(pending.process.pid,os.WNOHANG)
        except ChildProcessError:
            _LOST[query_id]=pending;del _PENDING[query_id]
            window=current_observer_window()
            if window is not None and window.window_id==pending.window_id:
                pending.original_record.update(state="WAIT_OWNERSHIP_LOST",error_type="ChildProcessError")
            else:
                _LATE[query_id]=(pending.owner_thread,dict(origin_window_id=pending.window_id,query_id=query_id,
                    query_pid=pending.process.pid,owner_pid=pending.owner_pid,purpose=pending.purpose,
                    status="WAIT_OWNERSHIP_LOST",actual_exit_code=None,actual_cpu_seconds=None,
                    observed_monotonic_seconds=time.perf_counter(),adopted_into_current_window_cpu=False))
            continue
        if child:
            require(child==pending.process.pid,"query wait returned another child")
            observations.append(_reaped(pending,status,usage));del _PENDING[query_id]
    window=current_observer_window()
    if window is not None:
        for query_id,(owner_thread,observation) in list(_LATE.items()):
            if observation["owner_pid"]==window.pid and owner_thread==window.thread:
                observation=dict(observation,reported_monotonic_seconds=time.perf_counter())
                window.late_reaps.append(observation);del _LATE[query_id]
    return observations


def pending_rss_query_pids():
    return sorted(p.process.pid for p in (*_PENDING.values(),*_LOST.values())
        if p.owner_pid==os.getpid() and p.purpose=="rss")


def record_legacy_pending_reap(pid,status,usage):
    """Known usage without an original window is never adopted as window cost."""
    window=current_observer_window()
    observation=dict(origin_window_id=None,query_id=None,query_pid=pid,owner_pid=os.getpid(),purpose="rss",
            status="LEGACY_UNBOUND_REAP",actual_exit_code=os.waitstatus_to_exitcode(status),actual_cpu_seconds=_usage(usage),
            observed_monotonic_seconds=time.perf_counter(),adopted_into_current_window_cpu=False)
    if window is not None:window.late_reaps.append(dict(observation,reported_monotonic_seconds=time.perf_counter()))
    else:_LATE[("legacy",pid,observation["observed_monotonic_seconds"])]=(threading.get_ident(),observation)


def run_owned_ps(command,*,timeout,purpose=None):
    """Same ps argv and timeout; bounded exclusive wait4/cleanup when opted in.

    Returns (CompletedProcess[str], CPU) or raises the original timeout/spawn
    class. A timeout may leave an exact unreaped child in the pending registry;
    unlike subprocess.run, cleanup never waits unboundedly for SIGKILL.
    """
    window=current_observer_window();purpose=_PURPOSE.get() if purpose is None else purpose
    require(window is not None and purpose in PURPOSES and type(command) is list and command and command[0]=="ps"
        and all(type(x) is str for x in command) and _finite(timeout) and 0<timeout<=2,"bounded own-ps query in active window required")
    drain_pending_queries()
    if purpose=="rss" and pending_rss_query_pids():
        error=RuntimeError("pending own RSS query cleanup");error.query_reason="pending_sampler_cleanup";error.cpu_seconds=None;raise error
    query_id=window.window_id+":"+str(len(window.queries))
    row=dict(query_id=query_id,origin_window_id=window.window_id,owner_pid=window.pid,purpose=purpose,
        query_pid=None,timeout_seconds=timeout,started_monotonic_seconds=time.perf_counter(),ended_monotonic_seconds=None,
        state="SPAWN_UNOBSERVED",actual_exit_code=None,actual_cpu_seconds=None,error_type=None,timed_out=False)
    window.queries.append(row);process=None;wait_owned=True
    try:
        with tempfile.TemporaryFile() as output,tempfile.TemporaryFile() as errors:
            deadline=row["started_monotonic_seconds"]+timeout
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=errors)
            row["query_pid"]=process.pid
            receipt=None
            while receipt is None:
                child,status,usage=os.wait4(process.pid,os.WNOHANG)
                if child:
                    require(child==process.pid,"query wait returned another child");receipt=(status,usage);break
                if time.perf_counter()>=deadline:
                    row["timed_out"]=True
                    try:os.kill(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    until=time.perf_counter()+.02
                    while receipt is None:
                        child,status,usage=os.wait4(process.pid,os.WNOHANG)
                        if child:
                            require(child==process.pid,"query cleanup wait returned another child");receipt=(status,usage);break
                        if time.perf_counter()>=until:break
                        time.sleep(.002)
                    break
                time.sleep(.002)
            if receipt is None:
                row["state"]="PENDING_UNREAPED"
                _PENDING[query_id]=_PendingQuery(process,window.pid,window.thread,window.window_id,query_id,purpose,row)
            else:
                process.returncode=os.waitstatus_to_exitcode(receipt[0])
                row.update(state="REAPED",actual_exit_code=process.returncode,actual_cpu_seconds=_usage(receipt[1]))
            if row["timed_out"]:
                error=subprocess.TimeoutExpired(command,timeout)
                error.cpu_seconds=row["actual_cpu_seconds"]
                error.query_reason="query_timeout_cleanup_pending" if receipt is None else "query_timeout"
                raise error
            output.seek(0);errors.seek(0)
            stdout=output.read(16*1024*1024+1);stderr=errors.read(16*1024*1024+1)
            require(len(stdout)<=16*1024*1024 and len(stderr)<=16*1024*1024,"bounded process-query output required")
            return subprocess.CompletedProcess(command,process.returncode,stdout.decode(),stderr.decode()),row["actual_cpu_seconds"]
    except ChildProcessError:
        wait_owned=False
        if process is not None:
            row.update(state="WAIT_OWNERSHIP_LOST",error_type="ChildProcessError")
            _LOST[query_id]=_PendingQuery(process,window.pid,window.thread,window.window_id,query_id,purpose,row)
        raise
    except BaseException as error:
        row["error_type"]=type(error).__name__
        if process is None:row["state"]="SPAWN_FAILED_USAGE_UNKNOWN"
        # If a parser/I/O failure occurs before reaping, retain exclusive wait
        # ownership. Never call Popen.poll/wait/communicate or signal after ECHILD.
        elif wait_owned and row["state"]=="SPAWN_UNOBSERVED":
            row["state"]="PENDING_UNREAPED"
            _PENDING[query_id]=_PendingQuery(process,window.pid,window.thread,window.window_id,query_id,purpose,row)
        raise
    finally:row["ended_monotonic_seconds"]=time.perf_counter()


def read_observer_window(trace):
    """Pure finite arithmetic and attribution checks, never evidence admission."""
    require(type(trace) is dict and trace["schema"]==SCHEMA
        and type(trace["observer_pid"]) is int and trace["observer_pid"]>0
        and type(trace["observer_thread_id"]) is int and trace["observer_thread_id"]>0
        and type(trace["window_id"]) is str and len(trace["window_id"])==64
        and trace["bindings_sha256"]==digest(trace["bindings"]),"original observer window identity/binding required")
    finite_keys=("started_monotonic_seconds","ended_monotonic_seconds","calling_thread_cpu_started_seconds",
        "calling_thread_cpu_ended_seconds","elapsed_wall_seconds","calling_thread_cpu_seconds")
    require(all(_finite(trace[k]) for k in finite_keys)
        and trace["elapsed_wall_seconds"]==trace["ended_monotonic_seconds"]-trace["started_monotonic_seconds"]
        and trace["calling_thread_cpu_seconds"]==trace["calling_thread_cpu_ended_seconds"]-trace["calling_thread_cpu_started_seconds"]
        and trace["elapsed_wall_seconds"]>=0 and trace["calling_thread_cpu_seconds"]>=0
        # Separate clock samples can differ by a small rounding/sampling skew;
        # a single calling thread cannot otherwise consume more CPU than wall.
        and trace["calling_thread_cpu_seconds"]<=trace["elapsed_wall_seconds"]+.001+trace["elapsed_wall_seconds"]*1e-6
        and all(trace[k] is False for k in ("process_table_output_retained","final_measurement_serialization_included",
            "nested_counters_added_again","full_process_cpu_measured","complete_workflow_cost","resource_cost_complete",
            "runtime_qualified","scientific_collection_authorized","holdout_authorized")),"observer arithmetic/scope drift")
    rows=trace["own_process_queries"]
    require(type(rows) is list and len({r["query_id"] for r in rows})==len(rows),"unique original query IDs required")
    for index,row in enumerate(rows):
        require(type(row) is dict and set(row)=={"query_id","origin_window_id","owner_pid","purpose","query_pid","timeout_seconds",
            "started_monotonic_seconds","ended_monotonic_seconds","state","actual_exit_code","actual_cpu_seconds","error_type","timed_out"}
            and row["query_id"]==trace["window_id"]+":"+str(index) and row["origin_window_id"]==trace["window_id"]
            and type(row["owner_pid"]) is int and row["owner_pid"]==trace["observer_pid"] and row["purpose"] in PURPOSES
            and _finite(row["timeout_seconds"]) and 0<row["timeout_seconds"]<=2 and type(row["timed_out"]) is bool
            and _finite(row["started_monotonic_seconds"]) and _finite(row["ended_monotonic_seconds"])
            and trace["started_monotonic_seconds"]<=row["started_monotonic_seconds"]<=row["ended_monotonic_seconds"]<=trace["ended_monotonic_seconds"]
            and (row["query_pid"] is None if row["state"]=="SPAWN_FAILED_USAGE_UNKNOWN"
                else type(row["query_pid"]) is int and row["query_pid"]>0 and row["query_pid"]!=trace["observer_pid"])
            and row["state"] in {"REAPED","PENDING_UNREAPED","WAIT_OWNERSHIP_LOST","SPAWN_FAILED_USAGE_UNKNOWN"},"original bounded query attribution required")
        require(row["state"]=="REAPED" and type(row["actual_exit_code"]) is int and _finite(row["actual_cpu_seconds"])
            or row["state"]!="REAPED" and row["actual_exit_code"] is None and row["actual_cpu_seconds"] is None,
            "query CPU/exit requires an actual owning wait4")
    foreign=trace["foreign_late_reap_observations"]
    require(type(foreign) is list and len({(r["origin_window_id"],r["query_id"],r["query_pid"],r["observed_monotonic_seconds"]) for r in foreign})==len(foreign),
        "unique linked late observations required")
    for row in foreign:
        require(type(row) is dict and set(row)=={"origin_window_id","query_id","query_pid","owner_pid","purpose","status",
            "actual_exit_code","actual_cpu_seconds","observed_monotonic_seconds","reported_monotonic_seconds","adopted_into_current_window_cpu"}
            and row["origin_window_id"]!=trace["window_id"] and type(row["query_pid"]) is int and row["query_pid"]>0
            and row["owner_pid"]==trace["observer_pid"] and row["purpose"] in PURPOSES
            and row["status"] in {"REAPED_LATE","LEGACY_UNBOUND_REAP","WAIT_OWNERSHIP_LOST"}
            and row["adopted_into_current_window_cpu"] is False and _finite(row["observed_monotonic_seconds"])
            and _finite(row["reported_monotonic_seconds"]) and row["observed_monotonic_seconds"]<=row["reported_monotonic_seconds"]
            and trace["started_monotonic_seconds"]<=row["reported_monotonic_seconds"]<=trace["ended_monotonic_seconds"],"late query costs cannot be adopted by another window")
        require(row["status"]=="WAIT_OWNERSHIP_LOST" and row["actual_exit_code"] is None and row["actual_cpu_seconds"] is None
            or row["status"]!="WAIT_OWNERSHIP_LOST" and type(row["actual_exit_code"]) is int and _finite(row["actual_cpu_seconds"]),
            "linked late query usage must be actual or unknown")
    return dict(trace_sha256=digest(trace),known_own_query_cpu_seconds=sum(r["actual_cpu_seconds"] for r in rows if r["actual_cpu_seconds"] is not None),
        own_query_usage_unknown=sum(r["actual_cpu_seconds"] is None for r in rows),
        calling_thread_cpu_seconds=trace["calling_thread_cpu_seconds"],elapsed_wall_seconds=trace["elapsed_wall_seconds"],
        late_query_usage_adopted=False,legacy_sampler_cpu_added_again=False,nested_counters_added_again=False,
        measurement_final_serialization_included=False,complete_workflow_cost=False,resource_cost_complete=False)


def publication_cost(trace,*,ended_wall,ended_cpu,trace_path,trace_bytes_sha256,error_type=None):
    """Freeze the disjoint original-thread tail; its own receipt is excluded."""
    require(trace["observer_pid"]==os.getpid() and trace["observer_thread_id"]==threading.get_ident(),
        "publication must retain original observer PID/thread")
    receipt=dict(schema=PUBLICATION_SCHEMA,window_id=trace["window_id"],trace_sha256=digest(trace),
        trace_path=trace_path,trace_bytes_sha256=trace_bytes_sha256,bindings_sha256=trace["bindings_sha256"],
        observer_pid=trace["observer_pid"],observer_thread_id=trace["observer_thread_id"],
        started_monotonic_seconds=trace["ended_monotonic_seconds"],ended_monotonic_seconds=ended_wall,
        calling_thread_cpu_started_seconds=trace["calling_thread_cpu_ended_seconds"],calling_thread_cpu_ended_seconds=ended_cpu,
        elapsed_wall_seconds=ended_wall-trace["ended_monotonic_seconds"],
        calling_thread_cpu_seconds=ended_cpu-trace["calling_thread_cpu_ended_seconds"],
        status="TRACE_PUBLISHED" if error_type is None else "PUBLICATION_FAILED",error_type=error_type,
        original_window_close_and_validation_included=True,first_trace_write_closed=error_type is None,
        supplemental_receipt_validation_serialization_write_included=False,unmeasured_child_cpu_included=False,
        complete_workflow_cost=False,resource_cost_complete=False,runtime_qualified=False,
        scientific_collection_authorized=False,holdout_authorized=False)
    read_publication_cost(receipt,trace,trace_path=trace_path,trace_bytes_sha256=trace_bytes_sha256)
    return receipt


def read_publication_cost(receipt,trace,*,trace_path,trace_bytes_sha256):
    """Pure original-window join, not historical admission or process polling."""
    read_observer_window(trace)
    fields={"schema","window_id","trace_sha256","trace_path","trace_bytes_sha256","bindings_sha256",
        "observer_pid","observer_thread_id","started_monotonic_seconds","ended_monotonic_seconds",
        "calling_thread_cpu_started_seconds","calling_thread_cpu_ended_seconds","elapsed_wall_seconds",
        "calling_thread_cpu_seconds","status","error_type","original_window_close_and_validation_included",
        "first_trace_write_closed","supplemental_receipt_validation_serialization_write_included",
        "unmeasured_child_cpu_included","complete_workflow_cost","resource_cost_complete","runtime_qualified",
        "scientific_collection_authorized","holdout_authorized"}
    require(type(receipt) is dict and set(receipt)==fields and receipt["schema"]==PUBLICATION_SCHEMA
        and receipt["window_id"]==trace["window_id"] and receipt["trace_sha256"]==digest(trace)
        and receipt["trace_path"]==trace_path and type(trace_path) is str
        and receipt["trace_bytes_sha256"]==trace_bytes_sha256
        and (trace_bytes_sha256 is None or type(trace_bytes_sha256) is str and len(trace_bytes_sha256)==64
            and all(c in "0123456789abcdef" for c in trace_bytes_sha256))
        and receipt["bindings_sha256"]==trace["bindings_sha256"]
        and type(receipt["observer_pid"]) is int and receipt["observer_pid"]==trace["observer_pid"]
        and type(receipt["observer_thread_id"]) is int and receipt["observer_thread_id"]==trace["observer_thread_id"],
        "publication lost original trace/path/bytes/owner binding")
    clock_keys=("started_monotonic_seconds","ended_monotonic_seconds","calling_thread_cpu_started_seconds",
        "calling_thread_cpu_ended_seconds","elapsed_wall_seconds","calling_thread_cpu_seconds")
    require(all(_finite(receipt[k]) for k in clock_keys)
        and receipt["started_monotonic_seconds"]==trace["ended_monotonic_seconds"]
        and receipt["calling_thread_cpu_started_seconds"]==trace["calling_thread_cpu_ended_seconds"]
        and receipt["elapsed_wall_seconds"]==receipt["ended_monotonic_seconds"]-receipt["started_monotonic_seconds"]
        and receipt["calling_thread_cpu_seconds"]==receipt["calling_thread_cpu_ended_seconds"]-receipt["calling_thread_cpu_started_seconds"]
        and receipt["calling_thread_cpu_seconds"]<=receipt["elapsed_wall_seconds"]+.001+receipt["elapsed_wall_seconds"]*1e-6
        and receipt["original_window_close_and_validation_included"] is True
        and all(receipt[k] is False for k in ("supplemental_receipt_validation_serialization_write_included",
            "unmeasured_child_cpu_included","complete_workflow_cost","resource_cost_complete","runtime_qualified",
            "scientific_collection_authorized","holdout_authorized")),"publication interval/scope drift")
    require(receipt["status"]=="TRACE_PUBLISHED" and receipt["first_trace_write_closed"] is True
        and receipt["error_type"] is None and trace_bytes_sha256 is not None
        or receipt["status"]=="PUBLICATION_FAILED" and receipt["first_trace_write_closed"] is False
        and type(receipt["error_type"]) is str and bool(receipt["error_type"]),"publication failure cannot become successful trace write")
    return dict(receipt_sha256=digest(receipt),status=receipt["status"],
        calling_thread_cpu_seconds=receipt["calling_thread_cpu_seconds"],elapsed_wall_seconds=receipt["elapsed_wall_seconds"],
        ended_monotonic_seconds=receipt["ended_monotonic_seconds"],supplemental_receipt_write_included=False,
        complete_workflow_cost=False,resource_cost_complete=False)
