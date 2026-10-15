"""Ancestor-visible ownership of nested one-shot supervisors.

Only launch/terminal metadata travels here, never source archives or jobs. A
nested bootstrap inherits its launcher's group until durable delegation admits
its detach. Each direct parent alone owns wait4. Ancestors can freeze and stop
bound generations, but cannot manufacture child exit or complete-tree costs.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from .search_over_raw import digest, require
from .search_over_raw_ledger import _write_new
from .paper_reference_parallel import owned_process_identity, signal_owned_group


MAX_DEPTH = 4
MAX_CHILDREN = 8192  # Bounded engineering inventory, not a scientific allocation.
MAX_ACTIVE = 20
_SCOPE = None
_WORKER_PID = None


@dataclass(frozen=True)
class DiagnosticIncompletePolicy:
    """Prospective fixed-work-only control exception, never work acceptance.

    The manifest and slot identify the caller's independently reviewed roster;
    hashes do not themselves confer review or collection authority.
    """
    manifest_sha256: str
    slot_sha256: str
    job_sha256: str
    binding_sha256: str

    def __post_init__(self):
        require(all(type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
            for v in vars(self).values()), "prospective diagnostic policy hashes required")

    def receipt(self):
        from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
        return dict(schema="pokezero.search-over-raw.clean-diagnostic-incomplete.v1",
            **vars(self), contract=FixedWorkDiagnosticContract().receipt(),
            delegated_children_permitted=False, scored_work_permitted=False)


def _diagnostic_policy(value, claim):
    require(type(value) is dict, "exact prospective diagnostic policy required")
    policy = DiagnosticIncompletePolicy(*(value[k] for k in
        ("manifest_sha256", "slot_sha256", "job_sha256", "binding_sha256")))
    require(digest(value) == digest(policy.receipt()) and value["job_sha256"] == claim["job_sha256"]
        and value["binding_sha256"] == claim["binding_sha256"]
        and digest(value["contract"]) == digest(claim["contract"]), "diagnostic policy/claim drift")
    return value


def _clean_diagnostic_incomplete(row, child, terminal):
    """Read original cleanup/wait evidence without querying or inventing costs."""
    policy = row.get("diagnostic_incomplete_policy")
    if policy is None:
        return False
    claim = json.loads((child / "claim.json").read_bytes())
    require(_hash(child / "claim.json") == row["claim_sha256"]
        and digest(claim.get("diagnostic_incomplete_policy")) == digest(policy)
        and _hash(child / "job.pickle") == claim["job_sha256"], "original diagnostic exception claim drift")
    _diagnostic_policy(policy, claim)
    # No supervised subtree may be hidden by an outer timeout. Ordinary pool
    # generations may have been forcibly stopped and remain explicitly abnormal.
    registry = child / "delegated-groups"
    require(not registry.exists() or not list(registry.iterdir()),
        "local incomplete exception cannot conceal delegated supervised children")
    finite = lambda v: type(v) in (int, float) and math.isfinite(v) and v >= 0
    require(terminal["schema"] == "pokezero.search-over-raw.fixed-work-supervision.v1"
        and terminal["binding_sha256"] == policy["binding_sha256"]
        and type(terminal["worker_pid"]) is int and terminal["worker_pid"] == row["pid"]
        and type(terminal["supervisor_exit_code"]) is int and terminal["supervisor_exit_code"] == 124
        and terminal["status"] == "DIAGNOSTIC_INCOMPLETE" and terminal["watchdog_reached"] is True
        and terminal["cleanup_error_type"] is None and terminal["cpu_wall_cost_complete"] is False
        and type(terminal["worker_exit_code"]) is int
        and finite(terminal["cpu_seconds_including_reaped_children"]) and finite(terminal["maxrss"])
        and finite(terminal["elapsed_wall_seconds"]) and finite(terminal["supervisor_thread_cpu_seconds"])
        and terminal["watchdog_seconds"] == row["effective_watchdog_seconds"]
        and (terminal["watchdog_seconds"] == policy["contract"]["emergency_watchdog_seconds"]
            or claim["synthetic_process_fixture"] is True)
        and terminal["result"] is None and terminal["observed_work_accounting"] is None
        and terminal["phase_costs"] is None and terminal["runtime_qualified"] is False
        and terminal["deployment_eligible"] is False and terminal["retries_permitted"] is False,
        "clean local diagnostic termination lacks original wait/cleanup evidence")
    return True


def delegated_close_value(path, row, child, terminal, *, clean_incomplete=False):
    value = dict(delegation_sha256=_hash(path), terminal_sha256=_hash(child / "terminal.json"),
        parent_pid=row["parent_pid"], child_pid=row["pid"],
        actual_worker_exit_code=terminal["worker_exit_code"], normal=terminal["supervisor_exit_code"] == 0)
    if clean_incomplete:
        value.update(normal=False, clean_diagnostic_incomplete=True)
    return value


def _canonical(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(), "delegated path must be canonical")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "delegated symlink refused")
    return path


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class OwnedScope:
    directory: Path
    pid: int
    binding: str
    depth: int
    thread_id: int


def current_scope():
    scope = _SCOPE
    require(scope is None or scope.pid == os.getpid(), "inherited or stale ownership scope")
    require(scope is not None or _WORKER_PID is None, "ownership scope already closed or inherited")
    return scope


@contextmanager
def worker_scope(directory, binding, depth):
    global _SCOPE, _WORKER_PID
    require(current_scope() is None and type(depth) is int and 0 <= depth <= MAX_DEPTH,
        "invalid nested worker scope")
    _WORKER_PID = os.getpid()
    _SCOPE = OwnedScope(_canonical(directory), os.getpid(), binding, depth, threading.get_ident())
    try:
        yield
    finally:
        _SCOPE = None  # Permanent worker marker forbids late background launches.


def check_capacity(scope):
    require(scope is current_scope() and scope.thread_id == threading.get_ident(),
        "threaded nested launches require a separate reviewed ownership protocol")
    require(scope.depth < MAX_DEPTH, "delegated depth exhausted")
    registry = scope.directory / "delegated-groups"
    require(len(list(registry.glob("child-*.json"))) < MAX_CHILDREN
        and len(list((registry / "active").glob("child-*.json"))) < MAX_ACTIVE,
        "delegated inventory exhausted")


def publish_delegation(scope, directory, launch, *, diagnostic_policy=None, effective_watchdog_seconds=None):
    check_capacity(scope)
    directory = _canonical(directory)
    require(directory != scope.directory and launch["supervisor_pid"] == scope.pid,
        "delegated launcher drift")
    registry = scope.directory / "delegated-groups"
    registry.mkdir(exist_ok=True)
    active = registry / "active"
    active.mkdir(exist_ok=True)
    path = registry / f"child-{launch['pid']}-{launch['binding_sha256']}.json"
    row = dict(protocol="nested-owned-v1", state="PENDING_GROUP", parent_pid=scope.pid,
        parent_directory=str(scope.directory), parent_binding=scope.binding,
        child_directory=str(directory), depth=scope.depth + 1,
        launch_sha256=_hash(directory / "launch.json"), **launch)
    if diagnostic_policy is not None:
        claim = json.loads((directory / "claim.json").read_bytes())
        row.update(diagnostic_incomplete_policy=_diagnostic_policy(diagnostic_policy, claim),
            claim_sha256=_hash(directory / "claim.json"), effective_watchdog_seconds=effective_watchdog_seconds)
    _write_new(path, row)
    os.link(path, active / path.name)
    _sync(registry); _sync(active)
    return dict(path=str(path), sha256=_hash(path), depth=row["depth"])


def _read(path, directory, pid, binding, depth):
    path = _canonical(path)
    row = json.loads(path.read_bytes())
    require(row.get("protocol") == "nested-owned-v1" and row.get("state") == "PENDING_GROUP"
        and type(row.get("parent_pid")) is int and row["parent_pid"] == pid and row.get("parent_directory") == str(directory)
        and row.get("parent_binding") == binding and row.get("supervisor_pid") == pid
        and type(row.get("pid")) is int and row["pid"] > 0 and row.get("group") == row["pid"]
        and type(row.get("birth_identity")) is str and bool(row["birth_identity"])
        and type(row.get("depth")) is int and row["depth"] == depth + 1 and row["depth"] <= MAX_DEPTH
        and path.name == f"child-{row['pid']}-{row['binding_sha256']}.json",
        "delegation identity or depth drift")
    child = _canonical(row["child_directory"])
    require(child != directory and _hash(child / "launch.json") == row["launch_sha256"],
        "delegated launch bytes drift")
    launch = json.loads((child / "launch.json").read_bytes())
    require(all(launch.get(k) == row.get(k) for k in
        ("pid", "group", "birth_identity", "supervisor_pid", "binding_sha256")),
        "delegated launch fields drift")
    if "diagnostic_incomplete_policy" in row:
        claim = json.loads((child / "claim.json").read_bytes())
        require(_hash(child / "claim.json") == row["claim_sha256"]
            and digest(claim.get("diagnostic_incomplete_policy")) == digest(row["diagnostic_incomplete_policy"]),
            "prospective delegated diagnostic policy drift")
        _diagnostic_policy(row["diagnostic_incomplete_policy"], claim)
    return row, child


def accept_delegation(permission, directory, binding):
    """Called in the waiting bootstrap, before unpickling or running a job."""
    path = _canonical(permission["path"])
    parent = path.parent.parent
    row, child = _read(path, parent, os.getppid(),
        json.loads((parent / "launch.json").read_bytes())["binding_sha256"], permission["depth"] - 1)
    require(_hash(path) == permission["sha256"] and child == directory
        and row["pid"] == os.getpid() and row["binding_sha256"] == binding
        and os.path.samefile(path, path.parent / "active" / path.name)
        and os.getpgrp() == row["parent_pid"], "nested detach permission drift")
    birth = owned_process_identity(os.getpid())
    require(birth == row["birth_identity"], "nested child birth identity drift")
    os.setsid()
    require(os.getpgrp() == os.getpid(), "nested detach failed")
    _write_new(directory / "nested-detach.json", dict(state="DETACHED_READY",
        pid=os.getpid(), group=os.getpgrp(), depth=row["depth"], delegation_sha256=_hash(path)))
    return row["depth"]


def _detached(row, child, path):
    ack = child / "nested-detach.json"
    if not ack.exists():
        return False
    _canonical(ack)
    value = json.loads(ack.read_bytes())
    require(digest(value) == digest(dict(state="DETACHED_READY", pid=row["pid"], group=row["pid"],
        depth=row["depth"], delegation_sha256=_hash(path))), "nested detach acknowledgement drift")
    return True


def close_delegation(permission, receipt):
    """Actual direct-parent terminal only; failed entries remain active."""
    scope = current_scope()
    require(scope is not None, "delegated close lacks direct-parent scope")
    path = _canonical(permission["path"])
    row, child = _read(path, scope.directory, scope.pid, scope.binding, scope.depth)
    require(_hash(path) == permission["sha256"]
        and receipt == json.loads((child / "terminal.json").read_bytes())
        and receipt["worker_pid"] == row["pid"] and receipt["binding_sha256"] == row["binding_sha256"],
        "delegated terminal binding drift")
    normal = (type(receipt["supervisor_exit_code"]) is int and receipt["supervisor_exit_code"] == 0
        and type(receipt["worker_exit_code"]) is int and receipt["worker_exit_code"] == 0
        and receipt["cpu_wall_cost_complete"] is True and receipt["cleanup_error_type"] is None
        and _detached(row, child, path))
    clean_incomplete = False
    if not normal and row.get("diagnostic_incomplete_policy") is not None and receipt["supervisor_exit_code"] == 124:
        clean_incomplete = _clean_diagnostic_incomplete(row, child, receipt) and _detached(row, child, path)
    value = delegated_close_value(path, row, child, receipt, clean_incomplete=clean_incomplete)
    value["normal"] = normal
    _write_new(path.with_name("closed-" + path.name), value)
    if normal or clean_incomplete:
        (path.parent / "active" / path.name).unlink()
        _sync(path.parent / "active")


def verify_delegated_closure(directory, pid, binding, verify_ordinary, depth=0, seen=None, *, outcomes=None):
    """One final bounded tree audit. Swallowed nested failure is not success."""
    directory = _canonical(directory)
    seen = set() if seen is None else seen
    require(str(directory) not in seen and len(seen) <= MAX_CHILDREN, "delegated cycle or count drift")
    seen.add(str(directory))
    registry = directory / "delegated-groups"
    if registry.exists():
        _canonical(registry)
        require(all((p.name == "active" and p.is_dir()) or
            (p.is_file() and (p.name.startswith("child-") or p.name.startswith("closed-child-"))
                and p.suffix == ".json") for p in registry.iterdir()), "unexpected delegation registry entry")
        require(all(p.is_file() and p.name.startswith("child-") and p.suffix == ".json"
            for p in (registry / "active").iterdir()), "unexpected active delegation entry")
    require(not list((registry / "active").glob("child-*.json")), "nested generations still active")
    paths = list(registry.glob("child-*.json"))
    require(len(paths) <= MAX_CHILDREN, "delegated roster drift")
    if registry.exists():
        require({p.name for p in registry.iterdir()} == {"active"} | {p.name for p in paths}
            | {"closed-" + p.name for p in paths}, "missing or orphan delegated closure")
    witnesses = []
    for path in sorted(paths):
        row, child = _read(path, directory, pid, binding, depth)
        closed_path = _canonical(path.with_name("closed-" + path.name))
        closed = json.loads(closed_path.read_bytes())
        terminal = json.loads((child / "terminal.json").read_bytes())
        if closed.get("clean_diagnostic_incomplete") is True:
            require(_clean_diagnostic_incomplete(row, child, terminal) and _detached(row, child, path)
                and digest(closed) == digest(delegated_close_value(path, row, child, terminal, clean_incomplete=True)),
                "original clean diagnostic closure drift")
            if outcomes is not None: outcomes.append("CLEAN_DIAGNOSTIC_INCOMPLETE")
            # Keep forced ordinary generations in the fingerprint, but do NOT
            # claim their normal wait/CPU closure or score any partial work.
            ordinary = [(str(p.relative_to(child)), _hash(_canonical(p)))
                for p in sorted((child / "owned-groups").rglob("*.json"))]
            require(len(ordinary) <= 60, "bounded diagnostic ordinary inventory required")
            witnesses.append(dict(launch_sha256=_hash(path), closure_sha256=_hash(closed_path),
                terminal_sha256=_hash(child / "terminal.json"), detach_sha256=_hash(child / "nested-detach.json"),
                ordinary_abnormal=ordinary, normal=False))
            continue
        require(digest(closed) == digest(dict(delegation_sha256=_hash(path), terminal_sha256=_hash(child / "terminal.json"),
            parent_pid=pid, child_pid=row["pid"], actual_worker_exit_code=0, normal=True))
            and terminal["worker_pid"] == row["pid"] and terminal["binding_sha256"] == row["binding_sha256"]
            and type(terminal["supervisor_exit_code"]) is int and terminal["supervisor_exit_code"] == 0
            and type(terminal["worker_exit_code"]) is int and terminal["worker_exit_code"] == 0
            and terminal["cpu_wall_cost_complete"] is True and _detached(row, child, path),
            "normal delegated wait/close witness missing")
        verify_ordinary(child, row["pid"], row["binding_sha256"])
        nested = verify_delegated_closure(child, row["pid"], row["binding_sha256"], verify_ordinary,
            row["depth"], seen, outcomes=outcomes)
        witnesses.append(dict(launch_sha256=_hash(path), closure_sha256=_hash(closed_path),
            terminal_sha256=_hash(child / "terminal.json"), detach_sha256=_hash(child / "nested-detach.json"),
            nested_sha256=nested))
    ordinary = [(p.name, _hash(_canonical(p))) for p in sorted((directory / "owned-groups").glob("*.json"))]
    return digest(dict(directory=str(directory), witnesses=witnesses, ordinary=ordinary))


def freeze_launcher(pid, birth, allowed_groups, *, expected_parent=None):
    """Bounded stopped-state confirmation, never a wait on another parent's child."""
    deadline = time.perf_counter() + 1.
    stopped = False
    from .search_over_raw_resources import _process_table
    from .search_over_raw_observer_cost import QueryPurpose
    while True:
        remaining = deadline - time.perf_counter()
        require(remaining > 0, "owned launcher stop not confirmed")
        # Reuse exclusive wait4, bounded kill/reap and retained pending handles.
        # subprocess.run(timeout=...) can block indefinitely in its final wait.
        with QueryPurpose("ownership_freeze"):
            rows, _ = _process_table(min(.2, remaining))
        row = next((r for r in rows if r["pid"] == pid), None)
        if row is None:
            return None
        state, group, actual_birth = row["state"], row["group"], row["birth"]
        require(actual_birth == " ".join(birth.split())
            and group in allowed_groups, "owned stopped launcher identity drift")
        require(expected_parent is None or type(row.get("parent_pid")) is int
            and row["parent_pid"] == expected_parent, "owned launcher was reparented or is unrelated")
        if state.startswith("Z"):
            return None  # A terminal leader is not live group authority.
        if "T" in state:
            return int(group)
        if not stopped:
            try: os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError: return None
            stopped = True
        time.sleep(.002)


def freeze_delegated_groups(directory, pid, binding, ordinary_groups, depth=0, seen=None, issues=None, known=None,
                            kill_order=None):
    """Launcher is already frozen; freeze each child before its registry read.

    A child may be stopped between setsid and its acknowledgement publication.
    Its exact live PGID resolves that window; no job starts before the ack. A
    missing child without ack remains pending, never an inferred successful exit.
    """
    directory = _canonical(directory)
    seen = set() if seen is None else seen
    issues = [] if issues is None else issues
    known = set() if known is None else known
    kill_order = [] if kill_order is None else kill_order
    groups = {pid}  # The caller has already frozen and bound this launcher.
    try:
        require(str(directory) not in seen and len(seen) <= MAX_CHILDREN, "delegated cycle or count drift")
        seen.add(str(directory))
        try:
            groups = ordinary_groups(directory, pid, binding)
        except BaseException as failure:
            issues.append(type(failure).__name__)
        known.update(groups)
        # Each admitted leader is already frozen. Repeated group STOP would
        # introduce another failure boundary before sibling traversal.
        active = directory / "delegated-groups" / "active"
        try:
            if active.exists():
                _canonical(active)
                if any(not p.is_file() or not p.name.startswith("child-") or p.suffix != ".json" for p in active.iterdir()):
                    issues.append("unexpected_active_delegation_entry")
            paths = list(active.glob("child-*.json"))
        except BaseException as failure:
            issues.append(type(failure).__name__)
            return groups
        require(len(paths) <= MAX_ACTIVE, "active delegated roster drift")
        for path in paths:
            try:
                groups |= _freeze_child(path, directory, pid, binding, ordinary_groups, depth, seen, issues, known, kill_order)
            except BaseException as failure:
                # Retain every previously verified scope, including children
                # whose later inventory read failed. Unbound PIDs stay absent.
                issues.append(type(failure).__name__)
        return groups
    finally:
        # Postorder must survive every exit, not only successful traversal.
        # Recursive children append themselves before this owner's turn.
        kill_order.extend(sorted(groups - {pid} - set(kill_order)))
        if pid not in kill_order:
            kill_order.append(pid)


def _freeze_child(path, directory, pid, binding, ordinary_groups, depth, seen, issues, known, kill_order=None):
        original = path.parent.parent / path.name
        require(os.path.samefile(_canonical(path), _canonical(original)), "active delegation index drift")
        row, child = _read(original, directory, pid, binding, depth)
        pgid = freeze_launcher(row["pid"], row["birth_identity"], {pid, row["pid"]}, expected_parent=pid)
        if pgid == row["pid"]:
            known.add(row["pid"])
        try:
            try:
                ready = _detached(row, child, original)
            except (ValueError, OSError):
                # Partial ack cannot erase verified live cleanup reachability.
                ready = pgid == row["pid"]
                issues.append("incomplete_nested_detach_acknowledgement")
            if pgid == row["pid"]:
                return freeze_delegated_groups(child, row["pid"], row["binding_sha256"],
                    ordinary_groups, row["depth"], seen, issues, known, kill_order)
            elif pgid is None:
                issues.append("delegated_live_authority_lost")
                return set()
            else:
                # Bootstrap remains in the already-frozen parent group.
                require(not list((child / "owned-groups" / "active").glob("group-*.json"))
                    and not list((child / "delegated-groups" / "active").glob("child-*.json")),
                    "pending bootstrap unexpectedly launched descendants")
            return set()
        finally:
            # Canonical-path/ack/cycle failures can precede recursive setup.
            # The live child admitted above still goes before its parent.
            if pgid == row["pid"] and kill_order is not None and pgid not in kill_order:
                kill_order.append(pgid)
