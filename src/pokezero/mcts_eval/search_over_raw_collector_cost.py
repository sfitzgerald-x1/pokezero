"""Opt-in exclusive wait4 owner of the existing collector CLI invocation.

Not a new job/ownership depth, admission, retry, or resource qualification.
The only additive CPU frontier is the exact CLI process including descendants
it actually reaped. Original entry/attempt/observer costs remain breakdowns.
External launch preparation and this reader are outside the measured command.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time

from .search_over_raw import digest, require
from .search_over_raw_continuation_admission import _path
from .search_over_raw_ledger import _write_new
from .search_over_raw_attempt_cost import _json
from .search_over_raw_supervision import (_reap, _signal_unreaped_child,
    _groups, observe_original_supervisor_launch)
from .search_over_raw_nested_ownership import freeze_launcher, freeze_delegated_groups
from .paper_reference_parallel import owned_process_identity, signal_owned_group, live_owned_groups

SCHEMA = "pokezero.search-over-raw.collector-invocation-cost.v1"
_MAX_OUTPUT = 65536
_UNRESOLVED_CHILD_HANDLES = {}  # No Popen destructor poll after missing/lost wait ownership.
_WALL_SCOPE = "owning launch through actual reap and bounded output capture; external readback excluded"
_CPU_SCOPE = "exact CLI wait4 including descendants it actually reaped; nested counters not added"
_RSS_SCOPE = "wait4 CLI high-water mark; not aggregate tree peak"
_TERMINAL_KEYS = {"schema", "claim_sha256", "cli_pid", "actual_exit_code", "actual_user_cpu_seconds",
    "actual_system_cpu_seconds", "actual_cpu_seconds_including_reaped_children", "started_monotonic_seconds",
    "ended_monotonic_seconds", "elapsed_wall_seconds", "wall_scope", "cpu_scope", "maxrss", "maxrss_units",
    "maxrss_scope", "watchdog_reached", "wait_ownership_lost", "error_type", "cleanup_errors",
    "verified_cleanup_groups", "stdout", "retries_permitted", "admission_override", "resource_cost_complete",
    "peak_tree_memory_measured", "scientific_collection_authorized", "holdout_authorized"}


def _sha(path):
    return hashlib.sha256(_path(path).read_bytes()).hexdigest()


def _executable_sha():
    # Virtualenv interpreters are routinely symlinks. Hash their resolved
    # binary without replacing argv[0] and thereby abandoning the venv.
    return hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _attempt_phase_breakdown(census):
    """Project the original validated census, never sum overlapping windows.

    A parent's work phase includes its nested jobs. Job CPU is a whole-job
    wait4 measurement, not a phase CPU allocation. Original readers establish
    ownership and clocks before this projection is used by read_collector_cost.
    """
    require(census["declared_supervised_attempt_census_complete"] is True
        and census["all_journal_writers_normal_closed"] is True,
        "closed original attempt census required for phase breakdown")
    rows = []
    for attempt in census["attempts"]:
        phases = attempt["phase_costs"]
        work = {name + "_wall_seconds" for name in
            ("selection", "continuations", "materialization", "boundaries")}
        common = {"process_bootstrap_wall_seconds", "startup_wall_seconds",
            "cleanup_wall_seconds", "receipt_reap_and_verification_wall_seconds"}
        require(type(phases) is dict and len(set(phases) & work) == 1
            and set(phases) - work == common and all(_finite(v) for v in phases.values())
            and attempt["measurement_window_cpu_wall_complete"] is True
            and _finite(attempt["elapsed_job_wall_seconds"])
            and _finite(attempt["job_cpu_seconds_including_reaped_children"])
            and sum(phases.values()) <= attempt["elapsed_job_wall_seconds"],
            "original validated per-attempt phase costs required")
        rows.append(dict(directory=attempt["directory"], parent_directory=attempt["parent_directory"],
            phase_wall_seconds=deepcopy(phases),
            elapsed_job_wall_seconds=attempt["elapsed_job_wall_seconds"],
            job_cpu_seconds_including_reaped_children=attempt["job_cpu_seconds_including_reaped_children"],
            phase_cpu_seconds=None))
    return dict(schema="pokezero.search-over-raw.collector-attempt-phase-breakdown.v1",
        attempts=rows, phase_wall_totals=None, phase_cpu_totals=None,
        nested_windows_may_overlap=True, nested_counters_added_to_invocation=False,
        scope="original supervised attempts only; source actors and external readback are not separate phases",
        phase_cpu_measured=False, complete_workflow_cost=False, resource_cost_complete=False)


def collector_script():
    return _path(Path(__file__).resolve().parents[3] / "scripts/collect_search_over_raw_reconciled_exploration.py")


def _bindings():
    from .search_over_raw_attempt_journal import _framework
    files = [collector_script(), Path(__file__), Path(__file__).with_name("search_over_raw_exploration_entry.py")]
    return {**_framework(), **{str(_path(p)): _sha(p) for p in files}}


def _check_claim(directory):
    directory = _path(directory)
    row = _json(directory / "claim.json")
    require(type(row) is dict and row.get("schema") == SCHEMA + ".claim"
        and set(row) == {"schema", "directory", "ledger_directory", "spec_path", "input_hashes", "framework_hashes",
            "entry_job", "argv", "executable_sha256", "owner_pid", "owner_birth", "watchdog_seconds",
            "synthetic_process_fixture", "admission_override", "retries_permitted", "holdout_authorized"}
        and row["directory"] == str(directory) and type(row["owner_pid"]) is int and row["owner_pid"] > 0
        and type(row["owner_birth"]) is str and bool(row["owner_birth"])
        and type(row["synthetic_process_fixture"]) is bool
        and all(row[k] is False for k in ("admission_override", "retries_permitted", "holdout_authorized"))
        and _finite(row["watchdog_seconds"]) and 0 < row["watchdog_seconds"] <= 7 * 86400,
        "original one-shot collector cost claim required")
    ledger = _path(row["ledger_directory"])
    require(not directory.is_relative_to(ledger) and not ledger.is_relative_to(directory),
        "cost namespace must not change original ledger evidence")
    require(type(row["input_hashes"]) is dict and type(row["framework_hashes"]) is dict
        and all(str(_path(p)) == p and _sha(p) == sha
            for p, sha in {**row["input_hashes"], **row["framework_hashes"]}.items()),
        "original collector inputs or executable sources changed")
    require(type(row["argv"]) is list and row["argv"] and row["argv"][0] == sys.executable
        and _executable_sha() == row["executable_sha256"], "original collector executable drift")
    if not row["synthetic_process_fixture"]:
        require(row["framework_hashes"] == _bindings()
            and row["argv"] == [sys.executable, "-B", str(collector_script()), "--ledger", str(ledger),
                "--entry-spec", row["spec_path"], "--cost-child", str(directory)],
            "only the existing exact collector command may receive invocation costs")
        from .search_over_raw_exploration_entry import ExplorationAuditorJob
        job = ExplorationAuditorJob(**row["entry_job"])
        spec = _json(row["spec_path"])
        envelope = _json(ledger / "contract.json")
        require(row["input_hashes"] == {str(p): _sha(p) for p in
            (_path(row["spec_path"]), ledger / "contract.json", _path(spec["review_path"]))},
            "exact original invocation input inventory required")
        require(job.ledger_directory == str(ledger) and job.spec_path == row["spec_path"]
            and job.spec_sha256 == _sha(job.spec_path) and job.review_sha256 == _sha(spec["review_path"])
            and job.contract_sha256 == envelope["contract_sha256"] == digest(envelope["contract"]),
            "original entry job/input correspondence drift")
    return row


def _original_root(claim, cli_pid):
    """No unpickling, discovery, historical PID signalling, or new admission."""
    directory = _path(Path(claim["ledger_directory"]) / "exploration-entry-run")
    launch_path = directory / "launch.json"
    if not launch_path.exists():
        return None
    launch = _json(launch_path)
    original = _json(directory / "claim.json")
    require(set(launch) == {"pid", "group", "birth_identity", "supervisor_pid", "binding_sha256"}
        and type(launch["pid"]) is int and launch["pid"] > 0
        and type(launch["group"]) is int and launch["group"] == launch["pid"]
        and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"] == cli_pid
        and type(launch["birth_identity"]) is str and bool(launch["birth_identity"])
        and launch["binding_sha256"] == original["binding_sha256"]
        and original["job_sha256"] == _sha(directory / "job.pickle")
        and original["retry_authorized"] is False, "original direct CLI-owned auditor required")
    if not claim["synthetic_process_fixture"]:
        from .search_over_raw_exploration_entry import ExplorationAuditorJob
        job = ExplorationAuditorJob(**claim["entry_job"])
        require(original["job_sha256"] == hashlib.sha256(pickle.dumps(job)).hexdigest()
            and launch["binding_sha256"] == digest(asdict(job)), "original registered auditor job drift")
    return dict(directory=str(directory), kind="staged-exploration-run",
        job_sha256=original["job_sha256"], launch=launch)


def _launch(claim):
    directory = _path(claim["directory"])
    launch = _json(directory / "launch.json")
    require(set(launch) == {"pid", "group", "birth_identity", "supervisor_pid", "claim_sha256"}
        and type(launch["pid"]) is int and launch["pid"] > 0
        and type(launch["group"]) is int and launch["group"] == launch["pid"]
        and type(launch["birth_identity"]) is str and bool(launch["birth_identity"])
        and type(launch["supervisor_pid"]) is int and launch["supervisor_pid"] == claim["owner_pid"]
        and launch["claim_sha256"] == _sha(directory / "claim.json"), "original exact CLI launch required")
    return launch


def _query_census():
    # Only exact retained query handles may be reaped. No process-table scan.
    from . import search_over_raw_observer_cost as queries
    from . import search_over_raw_resources as resources
    resources._pending_samplers()
    pending = [p for p in queries._PENDING.values() if p.owner_pid == os.getpid()]
    lost = [p for p in queries._LOST.values() if p.owner_pid == os.getpid()]
    return dict(pending_queries=len(pending), lost_wait_ownership_queries=len(lost),
        pending_legacy_samplers=len(resources._PENDING_SAMPLERS))


@contextmanager
def collector_cost_child(directory, ledger_directory, spec_path):
    """Private child handshake; original collector admission remains mandatory."""
    claim = _check_claim(directory)
    require(not claim["synthetic_process_fixture"]
        and claim["ledger_directory"] == str(_path(ledger_directory))
        and claim["spec_path"] == str(_path(spec_path))
        and claim["owner_pid"] == os.getppid(), "cost child is not the original collector invocation")
    # Parent publishes launch.json after Popen, then releases this header.
    # Reading launch first races a fast fresh interpreter against publication.
    header = json.loads(sys.stdin.buffer.read())
    require(header == dict(pid=os.getpid(), claim_sha256=_sha(Path(directory) / "claim.json")),
        "cost child requires original owning permission header")
    launch = _launch(claim)
    require(launch["pid"] == os.getpid() == os.getpgrp()
        and launch["birth_identity"] == owned_process_identity(os.getpid())
        and claim["owner_birth"] == owned_process_identity(os.getppid()), "live original CLI handshake drift")
    def register(descriptor):
        require(descriptor == _original_root(claim, os.getpid())
            and descriptor["kind"] == "staged-exploration-run", "original pre-header registration drift")
        _write_new(Path(directory) / "auditor-registration.json", dict(schema=SCHEMA + ".registration",
            claim_sha256=launch["claim_sha256"], cli_launch_sha256=_sha(Path(directory) / "launch.json"),
            descriptor=descriptor))
    error = None
    primary = None
    with observe_original_supervisor_launch(register):
        _write_new(Path(directory) / "child-ready.json", dict(schema=SCHEMA + ".ready",
            cli_pid=os.getpid(), claim_sha256=launch["claim_sha256"],
            cli_launch_sha256=_sha(Path(directory) / "launch.json"), hook_installed_before_entry=True))
        try:
            yield
        except BaseException as failure:
            error = type(failure).__name__
            primary = failure
            raise
        finally:
            # This is after the original collector readback AND stdout write.
            # No process query is issued after this census in the child path.
            try:
                _write_new(Path(directory) / "child-close.json", dict(schema=SCHEMA + ".close",
                    cli_pid=os.getpid(), claim_sha256=launch["claim_sha256"], error_type=error,
                    query_census=_query_census()))
            except BaseException as failure:
                if primary is None:
                    raise
                primary.add_note("cost-child close evidence failed: " + type(failure).__name__)


def _stop_cli(claim, process, birth):
    """Freeze exact unreaped CLI, then actual registered depth-zero auditor."""
    event = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if event is not None:
        return _reap(process.pid), [], []  # Never use saved generations after exit.
    issues, groups, order = [], set(), []
    frozen = False
    try:
        require(freeze_launcher(process.pid, birth, {process.pid}, expected_parent=os.getpid()) == process.pid,
            "live exact CLI must be stopped before auditor inventory")
        frozen = True
        directory = Path(claim["directory"])
        ready = _json(directory / "child-ready.json")
        launch = _launch(claim)
        require(ready == dict(schema=SCHEMA + ".ready", cli_pid=process.pid,
            claim_sha256=launch["claim_sha256"], cli_launch_sha256=_sha(directory / "launch.json"),
            hook_installed_before_entry=True), "missing original mandatory pre-header hook proof")
        root = _original_root(claim, process.pid)
        if root is not None:
            # A missing registration is safe only because this exact bound
            # child installed the mandatory hook before entry: launch.json
            # precedes hook publication and execution-header release.
            registration = directory / "auditor-registration.json"
            if registration.exists():
                try:
                    registered = _json(registration)
                    require(registered == dict(schema=SCHEMA + ".registration",
                        claim_sha256=launch["claim_sha256"], cli_launch_sha256=_sha(directory / "launch.json"),
                        descriptor=root), "original cost registration drift")
                except (ValueError, TypeError, KeyError, OSError) as failure:
                    # A bad supplemental registration cannot erase original
                    # live canonical launch/claim/mandatory-hook authority.
                    issues.append("registration_" + type(failure).__name__)
            leader = root["launch"]
            if freeze_launcher(leader["pid"], leader["birth_identity"], {leader["pid"]},
                    expected_parent=process.pid) == leader["pid"]:
                groups.add(leader["pid"])
                ordinary = lambda d, p, b: _groups(d, p, b, _issues=issues, _live_authority=True)
                groups |= freeze_delegated_groups(Path(root["directory"]), leader["pid"],
                    leader["binding_sha256"], ordinary, depth=0, issues=issues, known=groups, kill_order=order)
            else:
                issues.append("auditor_live_authority_unavailable")
    except BaseException as failure:
        issues.append(type(failure).__name__)
    # Retain partial verified cleanup even if a later inventory failed.
    for group in dict.fromkeys(order + sorted(groups - set(order))):
        if group not in groups:
            continue
        try:
            signal_owned_group(group, signal.SIGKILL)
        except BaseException as failure:
            issues.append(type(failure).__name__)
    try:
        if frozen:
            signal_owned_group(process.pid, signal.SIGKILL)
        else:
            _signal_unreaped_child(process.pid, signal.SIGKILL)
    except BaseException as failure:
        issues.append(type(failure).__name__)
    until = time.perf_counter() + 2.
    receipt = None
    while receipt is None and time.perf_counter() < until:
        receipt = _reap(process.pid)
        if receipt is None:
            time.sleep(.002)
    if receipt is None:
        issues.append("exact_cli_reap_unobserved")
    else:
        try:
            if live_owned_groups(groups | {process.pid}):
                issues.append("verified_groups_remain_live")
        except BaseException as failure:
            issues.append("post_kill_liveness_" + type(failure).__name__)
    return receipt, sorted(groups), issues


def _run_claimed_invocation(directory):
    """Internal producer; synthetic claims exist only in disposable tests."""
    directory = _path(directory)
    claim = _check_claim(directory)
    if {p.name for p in directory.iterdir()} != {"claim.json"}:
        raise FileExistsError("collector invocation already occupied; no retry or resume")
    require(claim["owner_pid"] == os.getpid() and claim["owner_birth"] == owned_process_identity(os.getpid()),
        "only the original invocation owner may dispatch")
    began = time.perf_counter()
    process = None
    birth = None
    reaped = None
    error = None
    cleanup_errors, groups = [], []
    timed_out = False
    wait_lost = False
    stdout = None
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(claim["argv"], stdin=subprocess.PIPE, stdout=output, stderr=errors,
                start_new_session=True)
            birth = owned_process_identity(process.pid)
            require(birth, "original CLI birth unavailable")
            _write_new(directory / "launch.json", dict(pid=process.pid, group=process.pid, birth_identity=birth,
                supervisor_pid=os.getpid(), claim_sha256=_sha(directory / "claim.json")))
            process.stdin.write(json.dumps(dict(pid=process.pid, claim_sha256=_sha(directory / "claim.json"))).encode())
            process.stdin.close()
            while reaped is None:
                reaped = _reap(process.pid)
                if reaped is not None:
                    break
                if time.perf_counter() - began >= claim["watchdog_seconds"]:
                    timed_out = True
                    reaped, groups, cleanup_errors = _stop_cli(claim, process, birth)
                    break
                time.sleep(.02)
            if reaped is not None:
                process.returncode = reaped[0]
            output.seek(0)
            raw = output.read(_MAX_OUTPUT + 1)
            require(len(raw) <= _MAX_OUTPUT, "collector status output exceeds cost receipt bound")
            stdout = raw.decode()
        except BaseException as failure:
            error = type(failure).__name__
            wait_lost = isinstance(failure, ChildProcessError)
            if process is not None and reaped is None and not wait_lost:
                try:
                    reaped, groups, cleanup_errors = _stop_cli(claim, process, birth)
                except BaseException as cleanup:
                    cleanup_errors.append(type(cleanup).__name__)
                    wait_lost = isinstance(cleanup, ChildProcessError)
        finally:
            if process is not None:
                if reaped is not None:
                    process.returncode = reaped[0]
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if reaped is None:
                    _UNRESOLVED_CHILD_HANDLES[str(directory)] = process
    ended = time.perf_counter()
    usage = None if reaped is None else reaped[1]
    value = dict(schema=SCHEMA + ".terminal", claim_sha256=_sha(directory / "claim.json"),
        cli_pid=None if process is None else process.pid, actual_exit_code=None if reaped is None else reaped[0],
        actual_user_cpu_seconds=None if usage is None else usage.ru_utime,
        actual_system_cpu_seconds=None if usage is None else usage.ru_stime,
        actual_cpu_seconds_including_reaped_children=None if usage is None else usage.ru_utime + usage.ru_stime,
        started_monotonic_seconds=began, ended_monotonic_seconds=ended, elapsed_wall_seconds=ended-began,
        wall_scope=_WALL_SCOPE, cpu_scope=_CPU_SCOPE,
        maxrss=None if usage is None else usage.ru_maxrss, maxrss_units="bytes" if sys.platform == "darwin" else "KiB",
        maxrss_scope=_RSS_SCOPE,
        watchdog_reached=timed_out, wait_ownership_lost=wait_lost,
        error_type=error, cleanup_errors=cleanup_errors, verified_cleanup_groups=groups,
        stdout=stdout, retries_permitted=False, admission_override=False, resource_cost_complete=False,
        peak_tree_memory_measured=False, scientific_collection_authorized=False, holdout_authorized=False)
    _write_new(directory / "terminal.json", value)
    return value


def run_collector_cost(ledger_directory, spec_path, *, directory, watchdog_seconds=86400.):
    """Existing CLI, one fresh invocation, existing admission only. No preflight bypass."""
    ledger, spec_path, directory = map(_path, (ledger_directory, spec_path, directory))
    require(not directory.is_relative_to(ledger) and not ledger.is_relative_to(directory), "separate cost namespace required")
    require(_finite(watchdog_seconds) and 0 < watchdog_seconds <= 7 * 86400, "bounded external cost watchdog required")
    spec = _json(spec_path)
    envelope = _json(ledger / "contract.json")
    from .search_over_raw_exploration_entry import ExplorationAuditorJob
    job = ExplorationAuditorJob(str(ledger), str(spec_path), _sha(spec_path), envelope["contract_sha256"], _sha(spec["review_path"]))
    inputs = [spec_path, ledger / "contract.json", _path(spec["review_path"])]
    argv = [sys.executable, "-B", str(collector_script()), "--ledger", str(ledger),
        "--entry-spec", str(spec_path), "--cost-child", str(directory)]
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_new(directory / "claim.json", dict(schema=SCHEMA + ".claim", directory=str(directory), ledger_directory=str(ledger),
        spec_path=str(spec_path), input_hashes={str(p): _sha(p) for p in inputs}, framework_hashes=_bindings(),
        entry_job=asdict(job), argv=argv, executable_sha256=_executable_sha(), owner_pid=os.getpid(),
        owner_birth=owned_process_identity(os.getpid()), watchdog_seconds=watchdog_seconds,
        synthetic_process_fixture=False, admission_override=False, retries_permitted=False, holdout_authorized=False))
    return _run_claimed_invocation(directory)


def read_collector_cost(directory):
    """Read original evidence only; never poll old PIDs or manufacture exits."""
    directory = _path(directory)
    before = {p.name: _sha(p) for p in directory.iterdir() if p.is_file()}
    claim = _check_claim(directory)
    terminal = _json(directory / "terminal.json")
    require(type(terminal) is dict and set(terminal) == _TERMINAL_KEYS
        and terminal["schema"] == SCHEMA + ".terminal" and terminal["claim_sha256"] == _sha(directory / "claim.json")
        and all(terminal[k] is False for k in ("retries_permitted", "admission_override", "resource_cost_complete",
            "peak_tree_memory_measured", "scientific_collection_authorized", "holdout_authorized"))
        and terminal["wall_scope"] == _WALL_SCOPE and terminal["cpu_scope"] == _CPU_SCOPE
        and terminal["maxrss_scope"] == _RSS_SCOPE and terminal["maxrss_units"] in {"bytes", "KiB"}
        and type(terminal["wait_ownership_lost"]) is bool
        and (terminal["error_type"] is None or type(terminal["error_type"]) is str and bool(terminal["error_type"]))
        and type(terminal["cleanup_errors"]) is list and all(type(v) is str and v for v in terminal["cleanup_errors"])
        and type(terminal["verified_cleanup_groups"]) is list
        and all(type(v) is int and v > 0 for v in terminal["verified_cleanup_groups"])
        and terminal["verified_cleanup_groups"] == sorted(set(terminal["verified_cleanup_groups"]))
        and (terminal["stdout"] is None or type(terminal["stdout"]) is str
            and len(terminal["stdout"].encode()) <= _MAX_OUTPUT),
        "original incomplete resource/authority scope required")
    cpu = terminal["actual_cpu_seconds_including_reaped_children"]
    code = terminal["actual_exit_code"]
    if terminal["cli_pid"] is not None:
        require(type(terminal["cli_pid"]) is int and terminal["cli_pid"] > 0
            and _launch(claim)["pid"] == terminal["cli_pid"], "terminal lost original CLI identity")
    else:
        require(code is None and not (directory / "launch.json").exists(), "prelaunch cost cannot invent a CLI exit")
    require((code is None and cpu is None and terminal["actual_user_cpu_seconds"] is None
            and terminal["actual_system_cpu_seconds"] is None and terminal["maxrss"] is None)
        or (type(code) is int and -64 <= code <= 255 and _finite(cpu) and _finite(terminal["maxrss"])
            and terminal["wait_ownership_lost"] is False and _finite(terminal["actual_user_cpu_seconds"])
            and _finite(terminal["actual_system_cpu_seconds"])
            and cpu == terminal["actual_user_cpu_seconds"] + terminal["actual_system_cpu_seconds"]),
        "CPU and exit require original actual wait4")
    require(all(_finite(terminal[k]) for k in ("started_monotonic_seconds", "ended_monotonic_seconds", "elapsed_wall_seconds"))
        and terminal["elapsed_wall_seconds"] == terminal["ended_monotonic_seconds"] - terminal["started_monotonic_seconds"]
        and type(terminal["watchdog_reached"]) is bool and type(terminal["cleanup_errors"]) is list,
        "original invocation wall/cleanup evidence drift")
    complete = False
    phase_breakdown = None
    incomplete = []
    if claim["synthetic_process_fixture"]:
        incomplete.append("authored process fixture, not a validated collector")
    elif code != 0 or terminal["watchdog_reached"] or terminal["wait_ownership_lost"] or terminal["error_type"] is not None or terminal["cleanup_errors"]:
        incomplete.append("collector failed, forced, or owning cleanup incomplete")
    else:
        join_step = "original_entry_binding"
        try:
            from .search_over_raw_exploration_entry import ExplorationAuditorJob, read_exploration_entry, read_entry_observer_cost
            from .search_over_raw_attempt_journal import AttemptJournalHandle, read_attempt_journal_cost
            from .search_over_raw_resources import read_sampling_cpu_components
            job = ExplorationAuditorJob(**claim["entry_job"])
            root = _original_root(claim, terminal["cli_pid"])
            require(root is not None, "original auditor launch absent")
            root_terminal = _json(Path(root["directory"]) / "terminal.json")
            join_step = "original_auditor_registration"
            registration = _json(directory / "auditor-registration.json")
            require(registration == dict(schema=SCHEMA + ".registration", claim_sha256=terminal["claim_sha256"],
                cli_launch_sha256=_sha(directory / "launch.json"), descriptor=root), "original auditor registration lost")
            close = _json(directory / "child-close.json")
            join_step = "collector_query_census"
            require(type(close.get("query_census")) is dict
                and all(type(v) is int and v == 0 for v in close["query_census"].values()),
                "exact closed query counts required")
            require(close == dict(schema=SCHEMA + ".close", cli_pid=terminal["cli_pid"],
                claim_sha256=terminal["claim_sha256"], error_type=None,
                query_census=dict(pending_queries=0, lost_wait_ownership_queries=0, pending_legacy_samplers=0)),
                "collector query closure absent or incomplete")
            join_step = "completed_entry_reader"
            read_exploration_entry(job, root_terminal)
            join_step = "entry_observer_reader"
            observer = read_entry_observer_cost(job, root_terminal)
            journal = Path(job.ledger_directory) / "attempt-cost-journal"
            join_step = "attempt_census_reader"
            census = read_attempt_journal_cost(AttemptJournalHandle(str(journal), _sha(journal / "metadata.json")),
                expected_outer_terminal=root_terminal)
            join_step = "closed_observer_and_attempt_census"
            require(observer["observer_window"]["own_query_usage_unknown"] == 0
                and observer["publication_usage_unknown"] is False
                and census["declared_supervised_attempt_census_complete"] is True
                and census["all_journal_writers_normal_closed"] is True, "original entry/query/attempt census incomplete")
            for attempt in census["attempts"]:
                join_step = "normal_attempt_cpu"
                require(attempt.get("measurement_window_cpu_wall_complete") is True,
                    "forced or unknown attempt CPU prevents complete invocation cost")
                t = _json(Path(attempt["directory"]) / "terminal.json")
                join_step = "attempt_query_trace"
                require(t["resource_measurement_error_type"] is None and t["memory_measurement"] is not None,
                    "original query cost trace absent")
                trace = _json(Path(attempt["directory"]) / "memory-trace.json")
                join_step = "closed_attempt_rss_queries"
                require(read_sampling_cpu_components(trace)["rss_query_cost_window_complete"] is True,
                    "original pending/lost RSS query usage")
            # CLI output is the original status, never the parent reader result.
            join_step = "collector_final_status"
            require(json.loads(terminal["stdout"]) == dict(status=root_terminal["status"], holdout_authorized=False,
                phase_b_authorized=False, resource_cost_complete=False), "original final collector status drift")
            join_step = "original_attempt_phase_breakdown"
            phase_breakdown = _attempt_phase_breakdown(census)
            complete = True
        except (ValueError, TypeError, KeyError, OSError) as failure:
            # Fixed code-owned phase only: never export a callback's exception
            # text, traceback, model observation or source snapshot.
            incomplete.append("original collector join refused at " + join_step + ": " + type(failure).__name__)
    require(before == {p.name: _sha(p) for p in directory.iterdir() if p.is_file()},
        "original invocation evidence changed during readback")
    require(digest(claim) == digest(_check_claim(directory)),
        "original invocation inputs or executable changed during final readback")
    return dict(schema=SCHEMA, actual_exit_code=code, actual_cli_cpu_seconds_including_reaped_children=cpu,
        elapsed_invocation_wall_seconds=terminal["elapsed_wall_seconds"], invocation_cpu_wall_complete=complete,
        attempt_phase_breakdown=phase_breakdown,
        total_invocation_cpu_seconds=cpu if complete else None, incomplete_components=incomplete,
        nested_counters_added_again=False, external_launcher_and_reader_included=False,
        whole_cli_lifetime_wait4_observed=code is not None,
        collector_startup_other_threads_final_output_included=complete,
        peak_tree_memory_measured=False, resource_cost_complete=False, complete_workflow_cost=False,
        runtime_qualified=False, scientific_collection_authorized=False, holdout_authorized=False, retries_permitted=False)
