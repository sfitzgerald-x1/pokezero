"""One-shot POSIX supervision for a reviewed fixed-work diagnostic job.

No source allocation, registration or scientific execution authority. The
caller must verify its admission before invoking this helper. An attempt's
directory is create-only. CPU is the actual wait4 receipt of the isolated job,
including children it reaped; abnormal cleanup never claims complete tree cost.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import threading
import time

from .search_over_raw import require
from .search_over_raw_fixed_work import FixedWorkDiagnosticContract
from .search_over_raw_resources import OwnedRSSObserver, measurement_protocol, resource_source_hash
from .paper_reference_parallel import (
    live_owned_groups, owned_process_identity, owned_process_identities, signal_owned_group,
)
from .search_over_raw_nested_ownership import (
    worker_scope, current_scope, check_capacity, publish_delegation, accept_delegation,
    close_delegation, verify_delegated_closure, freeze_delegated_groups, freeze_launcher,
)


_ORIGINAL_LAUNCH_OBSERVER = ContextVar("search_over_raw_original_launch_observer", default=None)


@contextmanager
def observe_original_supervisor_launch(observer):
    """Opt-in same-process launch registration BEFORE the execution header.

    The observer receives original public launch/claim coordinates, never a
    Popen handle, stdin, job object or private task payload. Its exception
    prevents dispatch and follows the existing owning cleanup path. Fresh
    worker interpreters do not inherit this observer, so original scientific
    ownership depths and attempt-journal roots remain unchanged.

    This hook supplies no collection authority or complete-cost assertion.
    """
    require(callable(observer) and _ORIGINAL_LAUNCH_OBSERVER.get() is None,
        "one original launch observer may be installed")
    token = _ORIGINAL_LAUNCH_OBSERVER.set((os.getpid(), threading.get_ident(), observer))
    try:
        yield
    finally:
        _ORIGINAL_LAUNCH_OBSERVER.reset(token)


def _observe_original_supervisor_launch(directory, launch, payload_sha256, kind):
    installed = _ORIGINAL_LAUNCH_OBSERVER.get()
    if installed is None:
        return
    pid, thread, observer = installed
    require(pid == os.getpid() and thread == threading.get_ident(),
        "inherited or cross-thread launch observer refused")
    launch_path = directory / "launch.json"
    original_bytes = launch_path.read_bytes()
    original_claim = (directory / "claim.json").read_bytes()
    original_launch = deepcopy(launch)
    require(json.loads(original_bytes) == original_launch
        and hashlib.sha256((directory / "job.pickle").read_bytes()).hexdigest() == payload_sha256,
        "original launch/job bytes required before external registration")
    observer(dict(directory=str(directory), kind=kind, job_sha256=payload_sha256,
        launch=deepcopy(original_launch)))
    require(launch == original_launch and launch_path.read_bytes() == original_bytes
        and (directory / "claim.json").read_bytes() == original_claim
        and hashlib.sha256((directory / "job.pickle").read_bytes()).hexdigest() == payload_sha256,
        "original launch/job changed during external registration")


def _save_new(path, value):
    from .search_over_raw_ledger import _write_new
    _write_new(path, value)


@dataclass(frozen=True)
class FixedWorkAdapterJob:
    """A public, already allocated root; no source environment crosses transport."""
    configuration: object
    checkpoint_contract: object
    showdown_root: str
    context: object
    root_id: str
    selection_seed: int
    statistics_root_id: str
    expected_runtime_sha256: str
    reference_factory: object = None
    pending_transition: object = None
    native_ancestry_contract: object = None
    expected_public_request_sha256: str | None = None
    capture_native_public_fold: bool = False
    owned_reference_value_recipe: object = None

    def __post_init__(self):
        if self.owned_reference_value_recipe is not None:
            from .search_over_raw_reference_values import ReferenceValueLabelRecipe
            require(type(self.owned_reference_value_recipe) is ReferenceValueLabelRecipe
                and self.configuration.arm == "reference" and self.configuration.leaf in {"model", "hp_fraction"},
                "owned reference values require their distinct fixed-work job recipe")
        require(type(self.capture_native_public_fold) is bool and (not self.capture_native_public_fold
            or self.native_ancestry_contract is not None and self.expected_public_request_sha256 is not None),
            "prospective public-fold owning job requires ancestry and full request")
        if self.native_ancestry_contract is not None:
            from .search_over_raw_ancestry import NativeAncestryContract
            require(type(self.native_ancestry_contract) is NativeAncestryContract
                and self.configuration.arm == "incumbent" and self.configuration.leaf == "model",
                "ancestry owning job requires prospective incumbent model contract")
        from .head_to_head import public_only_context
        from .paper_reference_runtime import PublicRootRequest
        from dataclasses import replace
        from .search_over_raw_world_inputs import _binding
        public = public_only_context(self.context)
        request = PublicRootRequest.capture(public.public_materialization_state, public.observation,
            pending_transition=self.pending_transition)
        public = replace(public, observation=request.observation,
            requested_observations={public.player_id: request.observation},
            public_materialization_state=request.state)
        object.__setattr__(self, "context", public)
        object.__setattr__(self, "_set_source_hash", request.set_source_hash)
        object.__setattr__(self, "_consumed_binding", _binding(request))
        if self.expected_public_request_sha256 is not None:
            from .search_over_raw_world_inputs import public_request_witness
            require(public_request_witness(request)["public_request_sha256"] == self.expected_public_request_sha256,
                "owning diagnostic full public request drift")
        require(request.set_source_hash == self.checkpoint_contract.showdown_source_sha256,
            "public root catalog differs from pinned checkpoint contract")
        require(self.configuration.arm in {"incumbent", "reference"}
            and self.configuration.belief == "public" and self.configuration.seconds == 1.,
            "only the canonical public fixed-work diagnostic may be supervised")
        require(type(self.statistics_root_id) is str and bool(self.statistics_root_id)
            and self.root_id == self.statistics_root_id + ":" + self.configuration.identity,
            "diagnostic receipt and matched statistics root must be explicit")
        require(type(self.expected_runtime_sha256) is str and len(self.expected_runtime_sha256) == 64
            and all(c in "0123456789abcdef" for c in self.expected_runtime_sha256),
            "prospectively reviewed instrumented runtime binding required")

    def __call__(self, ownership, phase):
        adapter = None
        phase("startup")
        try:
            from dataclasses import replace
            from ..randbat import load_gen3_randbat_source_cached
            from .search_over_raw_world_inputs import WorldInputFixedWorkAdapter, validate_fixed_work_inputs
            source = load_gen3_randbat_source_cached(self.showdown_root)
            require(source.metadata.source_hash == self._set_source_hash, "worker public catalog binding drift")
            state = self.context.public_materialization_state
            engine = state.belief_engine.clone()
            engine.set_source = source  # Reattach independently pinned catalog, never a source env.
            context = replace(self.context, public_materialization_state=replace(state, belief_engine=engine))
            adapter = WorldInputFixedWorkAdapter(self.configuration,
                native_ancestry_contract=self.native_ancestry_contract,
                capture_native_public_fold=self.capture_native_public_fold,
                **({"expected_public_request_sha256": self.expected_public_request_sha256}
                    if self.expected_public_request_sha256 is not None else {}),
                fixed_work=FixedWorkDiagnosticContract(), checkpoint_contract=self.checkpoint_contract,
                showdown_root=self.showdown_root, reference_factory=self.reference_factory,
                **({"owned_value_recipe":self.owned_reference_value_recipe}
                    if self.owned_reference_value_recipe is not None else {}),
                **({"owned_process_receipts": ownership} if self.configuration.arm == "reference" else {}))
            require(adapter.runtime_sha256 == self.expected_runtime_sha256,
                "diagnostic instrumented runtime differs from reviewed binding")
            phase("selection")
            selected = adapter.select(context, root_id=self.root_id, selection_seed=self.selection_seed,
                statistics_root_id=self.statistics_root_id,
                pending_transition=self.pending_transition)
            require(selected.get("runtime_sha256") == self.expected_runtime_sha256,
                "diagnostic selected runtime binding drift")
            validate_fixed_work_inputs(selected.get("evidence"), arm=self.configuration.arm,
                binding=self._consumed_binding, statistics_root_id=self.statistics_root_id,
                construction_required=self.configuration.arm == "incumbent",
                native_public_fold_required=self.capture_native_public_fold,
                expected_public_request_sha256=self.expected_public_request_sha256)
            if self.owned_reference_value_recipe is not None:
                selected["owned_reference_value_diagnostic"] = adapter.finish_owned_reference_values(selected)
            return selected
        finally:
            phase("cleanup")
            if adapter is not None:
                adapter.close()


def _safe_error_frames(failure):
    """Repository code sites only; no messages, source lines, locals or paths."""
    root = Path(__file__).resolve().parents[1]
    frames, seen = [], set()
    # A strict cleanup can raise while another exception is active. Preserve
    # that original code site too, without persisting messages or payloads.
    while failure is not None and id(failure) not in seen and len(seen) < 4:
        seen.add(id(failure))
        trace = failure.__traceback__
        while trace is not None:
            try:
                relative = Path(trace.tb_frame.f_code.co_filename).resolve().relative_to(root)
            except (OSError, ValueError, RuntimeError):
                pass
            else:
                if relative.suffix == ".py":
                    frames.append(dict(module="pokezero." + ".".join(relative.with_suffix("").parts),
                        function=trace.tb_frame.f_code.co_name, line=trace.tb_lineno))
            trace = trace.tb_next
        failure = failure.__cause__ or failure.__context__
    return frames[-16:]


def _worker(directory, binding):
    """Starts scientific work only after parent publishes this exact ownership."""
    directory = Path(directory)
    phases = []
    def phase(name):
        require(name in {"startup", "selection", "continuations", "boundaries", "materialization", "cleanup"}, "unknown owned job phase")
        phases.append(dict(phase=name, monotonic_seconds=time.perf_counter()))
    began = time.perf_counter()
    result, error, error_frames = None, None, None
    journal_writer = None
    try:
        header = json.loads(sys.stdin.buffer.read())
        payload = (directory / "job.pickle").read_bytes()
        require(header["pid"] == os.getpid() and header["binding"] == binding
            and hashlib.sha256(payload).hexdigest() == header["payload_sha256"],
            "owned launch handshake drift")
        depth = accept_delegation(header["delegation"], directory, binding) if header.get("delegation") else 0
        require(os.getpgrp() == os.getpid(), "owned worker must detach before job load")
        ownership = dict(directory=str(directory / "owned-groups"), controller_pid=os.getpid(),
            registration_sha256=binding, protocol="active-generations-v1")
        with worker_scope(directory, binding, depth):
            from .search_over_raw_attempt_journal import worker_attempt_journal_scope
            with worker_attempt_journal_scope(header.get("attempt_declaration"),directory,binding,
                    header["payload_sha256"]) as journal_writer:
                job = pickle.loads(payload)  # Trusted local implementation, never external data.
                result = job(ownership, phase)
    except BaseException as failure:
        error = type(failure).__name__
        try:
            error_frames = _safe_error_frames(failure)
        except BaseException:
            # Auxiliary diagnostics must never bypass the original failure
            # receipt or the live owner hold needed for descendant cleanup.
            error_frames = []
    pending_cleanup = (bool(list((directory / "owned-groups" / "active").glob("group-*.json")))
        or bool(list((directory / "delegated-groups" / "active").glob("child-*.json"))))
    if pending_cleanup:
        error = error or "UnclosedOwnedGenerations"
    terminal_at = time.perf_counter()
    receipt = dict(result=result if error is None else None,
        error_type=error, phases=phases, worker_wall_seconds=time.perf_counter()-began,
        terminal_monotonic_seconds=terminal_at,
        binding_sha256=binding, worker_pid=os.getpid(), exit_code=0 if error is None else 1)
    if error_frames is not None:
        receipt["error_frames"] = error_frames
    if journal_writer is not None:
        receipt["attempt_journal_writer_close_sha256"] = journal_writer.close_sha256
    if pending_cleanup:
        receipt["cleanup_pending"] = True
    _save_new(directory / "worker-result.json", receipt)
    if pending_cleanup:
        # Keep the original launcher live and unreaped until its supervisor
        # freezes it and resolves its registered scopes. Returning now would
        # orphan those scopes and destroy live signalling authority.
        while True:
            time.sleep(.1)
    return 0 if error is None else 1


def _groups(directory, pid, binding, *, _issues=None, _live_authority=False):
    rows = []
    for path in sorted((directory / "owned-groups" / "active").glob("group-*.json")):
        try:
            original = directory / "owned-groups" / path.name
            require(original.is_file() and os.path.samefile(path, original),
                "active ownership index differs from immutable launch")
            row = json.loads(path.read_bytes())
            require(type(row.get("controller_pid")) is int and row["controller_pid"] == pid
                and row.get("registration_sha256") == binding
                and type(row.get("pid")) is int and row["pid"] > 0 and type(row.get("group")) is int
                and row["group"] == row["pid"] and type(row.get("birth_identity")) is str and row["birth_identity"]
                and type(row.get("generation")) is str and row["generation"]
                and path.name == f"group-{row['pid']}-{row['generation']}.json"
                and row.get("protocol") == "active-generations-v1", "owned subgroup receipt drift")
            rows.append(row)
        except BaseException as failure:
            if _issues is None: raise
            _issues.append(type(failure).__name__)
    require(len(rows) <= 20 and len({r["pid"] for r in rows}) == len(rows), "owned subgroup roster drift")
    if rows:
        identities = owned_process_identities([r["pid"] for r in rows])
        safe = [r for r in rows if (not _live_authority and r["pid"] not in identities)
            or identities.get(r["pid"]) == " ".join(r["birth_identity"].split())]
        if len(safe) != len(rows):
            require(_issues is not None, "owned subgroup PID reuse; do not signal")
            _issues.append("owned_subgroup_pid_reuse")
            rows = safe
        if _live_authority:
            frozen = []
            for row in rows:
                try:
                    if freeze_launcher(row["pid"], row["birth_identity"], {row["pid"]}, expected_parent=pid) == row["pid"]:
                        frozen.append(row)
                    else:
                        require(_issues is not None, "owned subgroup is not a live stopped leader")
                        _issues.append("owned_subgroup_live_authority_lost")
                except BaseException as error:
                    # A later failed inspection must not discard a valid,
                    # already-stopped independent sibling from cleanup.
                    if _issues is None:
                        raise
                    _issues.append(type(error).__name__)
            rows = frozen
    return {pid, *(r["group"] for r in rows)}


def _verify_closed_generations(directory, pid, binding):
    """Normal descendant wait/close receipts underpin complete CPU attribution."""
    owned = directory / "owned-groups"
    require(not list((owned / "active").glob("group-*.json")), "diagnostic still has active worker generations")
    paths = list(owned.glob("group-*.json"))
    require(len(paths) <= 20, "diagnostic worker roster drift")
    for path in paths:
        launch = json.loads(path.read_bytes())
        closed = json.loads(path.with_name("closed-" + path.name).read_bytes())
        require(launch["controller_pid"] == pid and launch["registration_sha256"] == binding
            and launch["protocol"] == "active-generations-v1"
            and closed["launch_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            and closed["generation"] == launch["generation"] and closed["pid"] == launch["pid"]
            and closed["controller_pid"] == pid
            and type(closed["worker_exit_code"]) is int and closed["worker_exit_code"] == 0
            and (closed.get("cleanup_protocol") == "live_leader_ack_before_reap.v2"
                and closed.get("clean_acknowledgment") is True and closed.get("forced_cleanup") is False
                and closed.get("runtime_close_failed") is False
                and type(closed.get("cleanup_exception_count")) is int and closed["cleanup_exception_count"] == 0
                if "worker_index" in launch else closed.get("cleanup_protocol") == "exact_subprocess_reap.v2")
            and closed["group_live"] is False, "normal reaped worker cost witness missing")


def _reap(pid):
    child, status, usage = os.wait4(pid, os.WNOHANG)
    return None if child == 0 else (os.waitstatus_to_exitcode(status), usage)


def _signal_unreaped_child(pid, signum):
    # Popen.send_signal polls and may consume waitpid first. Until our wait4
    # reaps this exact child, its PID cannot be reused; keep wait ownership here.
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass


def _stop_and_reap(directory, process, birth, binding, already_reaped=None, *, depth=0):
    """Only recorded groups; a corrupt/reused ownership witness stops signalling."""
    if already_reaped is not None:
        # Terminal receipts support historical readback, not signalling saved
        # groups after the launcher's wait ownership has been released.
        return already_reaped, set()
    event = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if event is not None and event.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED):
        return _reap(process.pid), set()
    current = owned_process_identity(process.pid)
    require(current is None or " ".join(current.split()) == " ".join(birth.split()),
        "diagnostic controller PID reuse; do not signal")
    # Freeze the launcher before reading descendant ownership. Children cannot
    # detach before their immutable ownership is published by the launcher.
    groups, issues, inventory_error, order = set(), [], None, []
    try:
        require(freeze_launcher(process.pid, birth, {process.pid}, expected_parent=os.getpid()) == process.pid,
            "live original launcher must remain stopped before group cleanup")
        groups.add(process.pid)
        ordinary = lambda d, p, b: _groups(d, p, b, _issues=issues, _live_authority=True)
        groups |= freeze_delegated_groups(directory, process.pid, binding, ordinary,
            depth=depth, issues=issues, known=groups, kill_order=order)
    except BaseException as error:
        # Cleanup every already-verified group even if a later witness is bad.
        # Unbound/reused identities never enter this set and remain untouched.
        inventory_error = error
    def done(receipt):
        if inventory_error is not None or issues:
            failure = inventory_error or RuntimeError("owned nested cleanup evidence incomplete")
            failure.reaped_receipt = receipt
            failure.verified_groups = groups
            raise failure
        return receipt, groups
    # Descendant leaders are frozen too. Finish the deepest verified scopes
    # before killing their owner; no TERM/CONT/reap-then-KILL sequence.
    sequence = list(dict.fromkeys([g for g in order if g != process.pid]
        + sorted(groups - set(order) - {process.pid}) + [process.pid]))
    for group in sequence:
        if group not in groups:
            continue
        try:
            signal_owned_group(group, signal.SIGKILL)
        except BaseException as failure:
            issues.append(type(failure).__name__)
            inventory_error = inventory_error or failure
            # No bypass/retry of the denied descendant group. Still attempt
            # independently verified siblings and the original owner below.
    if process.pid not in groups:
        try:
            _signal_unreaped_child(process.pid, signal.SIGKILL)
        except BaseException as failure:
            issues.append(type(failure).__name__)
            inventory_error = inventory_error or failure
    receipt = None
    until = time.perf_counter() + 2.
    while time.perf_counter() < until:
        try:
            receipt = receipt or _reap(process.pid)
        except BaseException as failure:
            failure.reaped_receipt = receipt
            failure.verified_groups = groups
            raise
        if receipt is not None:
            try:
                live = live_owned_groups(groups)
            except BaseException as failure:
                inventory_error = inventory_error or failure
                return done(receipt)  # Preserve the consumed wait4 receipt.
            if not live:
                return done(receipt)
        time.sleep(.02)
    failure = inventory_error or RuntimeError("owned diagnostic cleanup incomplete")
    failure.reaped_receipt = receipt
    failure.verified_groups = groups
    raise failure


def _stop_unadmitted_child(process):
    """Before handshake no job/descendant may start; reap this exact child."""
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    _signal_unreaped_child(process.pid, signal.SIGTERM)
    until = time.perf_counter() + 2.
    while time.perf_counter() < until:
        receipt = _reap(process.pid)
        if receipt is not None:
            return receipt
        time.sleep(.02)
    _signal_unreaped_child(process.pid, signal.SIGKILL)
    until = time.perf_counter() + 2.
    while time.perf_counter() < until:
        receipt = _reap(process.pid)
        if receipt is not None:
            return receipt
        time.sleep(.02)
    raise RuntimeError("unadmitted diagnostic child not reaped")


def _phase_costs(phases, terminal_at, work_phase="selection"):
    require(work_phase in ("selection", "continuations", "boundaries", "materialization")
        and type(phases) is list and [r["phase"] for r in phases] == ["startup", work_phase, "cleanup"],
        "complete diagnostic requires startup/selection/cleanup evidence")
    times = [r["monotonic_seconds"] for r in phases] + [terminal_at]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in times)
        and times == sorted(times), "diagnostic phase clock drift")
    return {name + "_wall_seconds": times[i+1]-times[i]
        for i, name in enumerate(("startup", work_phase, "cleanup"))}


def supervise_fixed_work(job, *, directory, binding_sha256, verify_admission,
                         contract=None, diagnostic_incomplete_policy=None, _synthetic_watchdog_seconds=None):
    """External emergency watchdog, not a decision ceiling or collection grant.

    Production limit is always1800 seconds. The private shorter limit is solely
    for a job explicitly labelled synthetic_process_fixture with no adapters.
    Caller verification must also bind job/root/runtime inputs and authority.
    """
    contract = FixedWorkDiagnosticContract() if contract is None else contract
    require(type(contract) is FixedWorkDiagnosticContract and callable(verify_admission),
        "reviewed contract and caller admission verifier required")
    def validate_result(selected):
        if isinstance(job, FixedWorkAdapterJob):
            from .search_over_raw_world_inputs import validate_fixed_work_inputs
            require(selected["root_id"] == job.root_id and selected["configuration_sha256"] == job.configuration.identity
                and selected["statistics_root_id"] == job.statistics_root_id
                and selected["selection_seed"] == job.selection_seed
                and selected["runtime_sha256"] == job.expected_runtime_sha256
                and selected["fixed_work_diagnostic"]["contract"] == contract.receipt()
                and selected["fixed_work_diagnostic"]["accepted_actual"] == contract.accepted_boundary(job.configuration.arm),
                "fixed-work selected receipt drift")
            validate_fixed_work_inputs(selected["evidence"], arm=job.configuration.arm,
                binding=job._consumed_binding, statistics_root_id=job.statistics_root_id,
                construction_required=job.configuration.arm == "incumbent",
                native_public_fold_required=job.capture_native_public_fold,
                expected_public_request_sha256=job.expected_public_request_sha256)
            from .search_over_raw_resources import derive_fixed_work_accounting
            return derive_fixed_work_accounting(selected["evidence"],arm=job.configuration.arm,
                binding=job._consumed_binding,statistics_root_id=job.statistics_root_id,leaf=job.configuration.leaf,
                native_public_fold_required=job.capture_native_public_fold,
                expected_public_request_sha256=job.expected_public_request_sha256)
    return _supervise_owned_job(job, directory=directory, binding_sha256=binding_sha256,
        verify_admission=verify_admission, contract=contract, kind="fixed-work", work_phase="selection",
        validate_result=validate_result, diagnostic_incomplete_policy=diagnostic_incomplete_policy,
        _synthetic_watchdog_seconds=_synthetic_watchdog_seconds)


def _supervise_owned_job(job, *, directory, binding_sha256, verify_admission, contract,
                         kind, work_phase, validate_result, diagnostic_incomplete_policy=None,
                         _synthetic_watchdog_seconds=None):
    """Shared one-shot wait ownership; wrappers retain their distinct work law."""
    require(os.name == "posix" and hasattr(os, "wait4"), "POSIX wait4 supervision required")
    require(kind in ("fixed-work", "outcome-run", "boundary-run", "materialization-run", "native-timed-run", "actor-annotated-native-timed-run", "native-oracle-timed-run", "reference-timed-run", "reference-oracle-timed-run", "raw-selection-run", "staged-exploration-run") and callable(verify_admission)
        and callable(validate_result), "owned job scope and independent result validator required")
    require(type(binding_sha256) is str and len(binding_sha256) == 64
        and all(c in "0123456789abcdef" for c in binding_sha256), "bound attempt hash required")
    require(verify_admission() is not False, "caller refused diagnostic admission")
    scope = current_scope()
    if scope is not None:
        check_capacity(scope)  # Refuse exhausted depth/inventory before any spawn.
    watchdog = contract.emergency_watchdog_seconds
    if _synthetic_watchdog_seconds is not None:
        require(getattr(job, "synthetic_process_fixture", False) is True
            and not isinstance(job, FixedWorkAdapterJob)
            and type(_synthetic_watchdog_seconds) in (int, float)
            and math.isfinite(_synthetic_watchdog_seconds) and 0 < _synthetic_watchdog_seconds < watchdog,
            "short watchdog is allowed only for synthetic process tests")
        watchdog = _synthetic_watchdog_seconds
    payload = pickle.dumps(job)
    policy = None
    if diagnostic_incomplete_policy is not None:
        from .search_over_raw_nested_ownership import DiagnosticIncompletePolicy
        require(type(diagnostic_incomplete_policy) is DiagnosticIncompletePolicy
            and kind == "fixed-work" and type(contract) is FixedWorkDiagnosticContract
            and (type(job) is FixedWorkAdapterJob or
                getattr(job, "synthetic_process_fixture", False) is True and _synthetic_watchdog_seconds is not None)
            and diagnostic_incomplete_policy.job_sha256 == hashlib.sha256(payload).hexdigest()
            and diagnostic_incomplete_policy.binding_sha256 == binding_sha256,
            "prospective fixed-work-only incomplete policy must bind original job and contract")
        policy = diagnostic_incomplete_policy.receipt()
    directory = Path(directory).resolve()
    from .search_over_raw_attempt_journal import current_attempt_journal,declare_supervised_attempt
    from .search_over_raw_attempt_cost import AttemptCostSpec
    journal_writer=current_attempt_journal()
    declaration=None
    if journal_writer is not None:
        declaration=declare_supervised_attempt(AttemptCostSpec(str(directory),hashlib.sha256(payload).hexdigest(),
            binding_sha256,kind,work_phase,contract.receipt(),resource_source_hash(),watchdog,
            None if scope is None else str(scope.directory)))
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    (directory / "owned-groups").mkdir()
    claim = dict(binding_sha256=binding_sha256,
        job_sha256=hashlib.sha256(payload).hexdigest(), contract=contract.receipt(),
        resource_measurement_protocol=measurement_protocol(), resource_measurement_source_sha256=resource_source_hash(),
        retry_authorized=False, synthetic_process_fixture=getattr(job, "synthetic_process_fixture", False) is True)
    if declaration is not None:claim["attempt_declaration"] = declaration
    if policy is not None: claim["diagnostic_incomplete_policy"] = policy
    _save_new(directory / "claim.json", claim)
    with (directory / "job.pickle").open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)  # Outcome jobs may carry private frozen roots.
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    began = time.perf_counter()
    supervisor_cpu_began = time.thread_time()
    process = None
    reaped = None
    timed_out, error, cleanup_error = False, None, None
    birth, groups = None, set()
    worker = None
    memory = None
    memory_summary = None
    measurement_error = None
    work_accounting = None
    delegation = None
    try:
        process = subprocess.Popen([sys.executable, "-B", "-m", __name__, "worker", str(directory), binding_sha256],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=scope is None)
        birth = owned_process_identity(process.pid)
        require(birth, "diagnostic process ownership unavailable")
        launch = dict(pid=process.pid, group=process.pid, birth_identity=birth,
            supervisor_pid=os.getpid(), binding_sha256=binding_sha256)
        _save_new(directory / "launch.json", launch)
        _observe_original_supervisor_launch(directory, launch, hashlib.sha256(payload).hexdigest(), kind)
        if scope is not None:
            delegation = (publish_delegation(scope, directory, launch) if policy is None else
                publish_delegation(scope, directory, launch, diagnostic_policy=policy,
                    effective_watchdog_seconds=watchdog))
        memory = OwnedRSSObserver(directory, launch, binding_sha256, began)
        memory.sample(timeout=min(measurement_protocol()["query_timeout_seconds"],watchdog))
        header = dict(pid=process.pid, binding=binding_sha256, payload_sha256=hashlib.sha256(payload).hexdigest())
        if delegation is not None:
            header["delegation"] = delegation
        if declaration is not None:
            header["attempt_declaration"] = declaration
        # Only a tiny permission packet crosses this pipe. The potentially
        # large public task was durably staged before launch, so pipe backpressure
        # cannot hide an unbounded task transfer outside the watchdog.
        process.stdin.write(json.dumps(header).encode())
        process.stdin.close()
        deadline = began + watchdog
        while reaped is None:
            reaped = _reap(process.pid)
            if reaped is not None:
                break
            worker_path = directory / "worker-result.json"
            if worker_path.exists():
                try:
                    pending = json.loads(worker_path.read_bytes())
                except json.JSONDecodeError:
                    pending = {}  # An incomplete write is not terminal proof.
                if pending.get("cleanup_pending") is True:
                    require(pending.get("worker_pid") == process.pid
                        and pending.get("binding_sha256") == binding_sha256,
                        "original cleanup request identity drift")
                    reaped, groups = _stop_and_reap(directory, process, birth, binding_sha256,
                        depth=0 if scope is None else scope.depth + 1)
                    break
            if time.perf_counter() >= deadline:
                timed_out = True
                reaped, groups = _stop_and_reap(directory, process, birth, binding_sha256,
                    depth=0 if scope is None else scope.depth + 1)
                break
            if time.perf_counter() >= memory.next_sample:
                remaining = deadline-time.perf_counter()
                if remaining > 0:
                    memory.sample(timeout=min(measurement_protocol()["query_timeout_seconds"],remaining))
            time.sleep(.02)
        process.returncode = reaped[0]  # wait4—not disappearance—owns exit status.
        require(not timed_out, "diagnostic emergency watchdog reached")
        require(reaped[0] == 0, "diagnostic worker nonzero exit")
        groups = _groups(directory, process.pid, binding_sha256)
        require(not live_owned_groups(groups), "diagnostic descendants remain live")
        _verify_closed_generations(directory, process.pid, binding_sha256)
        closure_sha = verify_delegated_closure(directory, process.pid, binding_sha256, _verify_closed_generations,
            depth=0 if scope is None else scope.depth + 1)
        worker = json.loads((directory / "worker-result.json").read_bytes())
        require(worker["binding_sha256"] == binding_sha256 and worker["worker_pid"] == process.pid
            and worker["exit_code"] == 0 and worker["error_type"] is None, "diagnostic terminal evidence drift")
        if declaration is not None:
            from .search_over_raw_attempt_journal import read_job_writer_close
            read_job_writer_close(declaration,directory,binding_sha256,worker)
            journal_writer.verify()
        if worker["terminal_monotonic_seconds"] > deadline:
            timed_out = True
            require(False, "diagnostic finished after emergency deadline")
        require(verify_admission() is not False, "caller revoked diagnostic admission")
        costs = _phase_costs(worker["phases"], worker["terminal_monotonic_seconds"], work_phase)
        costs["process_bootstrap_wall_seconds"] = worker["phases"][0]["monotonic_seconds"] - began
        costs["receipt_reap_and_verification_wall_seconds"] = time.perf_counter() - worker["terminal_monotonic_seconds"]
        work_accounting = validate_result(worker["result"])
        require(verify_delegated_closure(directory, process.pid, binding_sha256, _verify_closed_generations,
            depth=0 if scope is None else scope.depth + 1) == closure_sha,
            "original delegated closure evidence changed across final callbacks")
        if declaration is not None:journal_writer.verify()
    except BaseException as failure:
        error = type(failure).__name__
        reaped = getattr(failure, "reaped_receipt", reaped)
        groups |= getattr(failure, "verified_groups", set())
        if hasattr(failure, "verified_groups"):
            cleanup_error = type(failure).__name__
        if process is not None:
            try:
                if birth is None:
                    reaped = _stop_unadmitted_child(process)
                elif scope is not None and not (directory / "nested-detach.json").exists():
                    # It may still share the launcher's group. Never killpg the
                    # parent or assume a child group exists before detach.
                    if reaped is None:
                        reaped = _stop_unadmitted_child(process)
                else:
                    reaped, closed_groups = _stop_and_reap(directory, process, birth, binding_sha256, reaped,
                        depth=0 if scope is None else scope.depth + 1)
                    groups |= closed_groups
                process.returncode = reaped[0]
            except BaseException as cleanup:
                cleanup_error = type(cleanup).__name__
                reaped = getattr(cleanup, "reaped_receipt", reaped)
                groups |= getattr(cleanup, "verified_groups", set())
                if reaped is not None:
                    process.returncode = reaped[0]
    # A pre-header failure (including external registration refusal) leaves
    # the permission pipe open. Close our descriptor after owning cleanup;
    # no complete execution header has been sent and no retry is permitted.
    if process is not None and process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except BaseException as failure:
            cleanup_error = cleanup_error or type(failure).__name__
    if memory is not None:
        # Sampling errors remain cost limitations and never obscure the job's
        # primary error or interrupt owned cleanup. Post-reap inventory retains
        # short-lived generations even when no sample ever saw their memory.
        try:
            memory.sample()
            trace, memory_summary = memory.finish(time.perf_counter())
            _save_new(directory / "memory-trace.json", trace)
        except BaseException as failure:
            # Retain the primary job verdict and always attempt its terminal
            # receipt, even when auxiliary measurement/serialization fails.
            measurement_error = type(failure).__name__
            memory_summary = None
    usage = None if reaped is None else reaped[1]
    complete = error is None and cleanup_error is None
    receipt = dict(schema=f"pokezero.search-over-raw.{kind}-supervision.v1", binding_sha256=binding_sha256,
        status={"fixed-work":"DIAGNOSTIC_COMPLETE", "outcome-run":"OUTCOME_RUN_COMPLETE",
            "boundary-run":"BOUNDARY_RUN_COMPLETE", "materialization-run":"MATERIALIZATION_RUN_COMPLETE",
            "native-timed-run":"NATIVE_TIMED_RUN_COMPLETE", "reference-timed-run":"REFERENCE_TIMED_RUN_COMPLETE",
            "actor-annotated-native-timed-run":"ACTOR_ANNOTATED_NATIVE_TIMED_RUN_COMPLETE",
            "native-oracle-timed-run":"NATIVE_ORACLE_TIMED_RUN_COMPLETE",
            "reference-oracle-timed-run":"REFERENCE_ORACLE_TIMED_RUN_COMPLETE",
            "raw-selection-run":"RAW_SELECTION_RUN_COMPLETE", "staged-exploration-run":"STAGED_EXPLORATION_RUN_COMPLETE"}[kind] if complete
            else {"fixed-work":"DIAGNOSTIC_INCOMPLETE", "outcome-run":"OUTCOME_RUN_INCOMPLETE",
                "boundary-run":"BOUNDARY_RUN_INCOMPLETE", "materialization-run":"MATERIALIZATION_RUN_INCOMPLETE",
                "native-timed-run":"NATIVE_TIMED_RUN_INCOMPLETE", "reference-timed-run":"REFERENCE_TIMED_RUN_INCOMPLETE",
                "actor-annotated-native-timed-run":"ACTOR_ANNOTATED_NATIVE_TIMED_RUN_INCOMPLETE",
                "native-oracle-timed-run":"NATIVE_ORACLE_TIMED_RUN_INCOMPLETE",
                "reference-oracle-timed-run":"REFERENCE_ORACLE_TIMED_RUN_INCOMPLETE",
                "raw-selection-run":"RAW_SELECTION_RUN_INCOMPLETE", "staged-exploration-run":"STAGED_EXPLORATION_RUN_INCOMPLETE"}[kind]
            if timed_out and cleanup_error is None else "OPERATIONAL_STOP",
        supervisor_exit_code=0 if complete else 124 if timed_out and cleanup_error is None else 1,
        worker_exit_code=None if reaped is None else reaped[0], error_type=error, cleanup_error_type=cleanup_error,
        worker_pid=None if process is None else process.pid, owned_groups=sorted(groups),
        watchdog_seconds=watchdog, watchdog_reached=timed_out,
        elapsed_wall_seconds=time.perf_counter()-began, phase_costs=costs if complete else None,
        cpu_seconds_including_reaped_children=None if usage is None else usage.ru_utime + usage.ru_stime,
        cpu_scope="isolated job wait4; reaped descendants only", cpu_wall_cost_complete=complete,
        resource_cost_complete=False,
        maxrss=None if usage is None else usage.ru_maxrss,
        maxrss_units="bytes" if sys.platform == "darwin" else "KiB",
        maxrss_scope="wait4 job high-water mark; not aggregate tree peak",
        memory_measurement=memory_summary,
        resource_measurement_error_type=measurement_error,
        observed_work_accounting=work_accounting if complete else None,
        supervisor_thread_cpu_seconds=time.thread_time()-supervisor_cpu_began,
        supervisor_cpu_scope="calling thread through receipt construction; separate from job and ps CPU",
        monitoring_wall_overlaps_job=True,
        peak_tree_memory_measured=False, deployment_eligible=False, runtime_qualified=False,
        retries_permitted=False, holdout_authorized=False,
        result=worker["result"] if complete else None)
    if declaration is not None:
        writer_close_sha,journal_error=None,None
        if process is not None and reaped is not None:
            try:
                from .search_over_raw_attempt_journal import read_job_writer_close
                observed_worker=json.loads((directory/"worker-result.json").read_bytes())
                require(type(observed_worker["worker_pid"]) is int and observed_worker["worker_pid"]==process.pid
                    and type(observed_worker["exit_code"]) is int and observed_worker["exit_code"]==reaped[0],
                    "actual reaped journal writer required")
                writer_close_sha,_=read_job_writer_close(declaration,directory,binding_sha256,observed_worker)
            except (ValueError,TypeError,KeyError,OSError) as failure:journal_error=type(failure).__name__
        receipt.update(attempt_declaration_sha256=declaration["declaration_sha256"],
            attempt_journal_writer_close_sha256=writer_close_sha,attempt_journal_error_type=journal_error,
            declared_attempt_census_complete=False)
    _save_new(directory / "terminal.json", receipt)
    if delegation is not None:
        close_delegation(delegation, receipt)
    return receipt


if __name__ == "__main__":
    require(len(sys.argv) == 4 and sys.argv[1] == "worker", "private owned worker entry only")
    raise SystemExit(_worker(sys.argv[2], sys.argv[3]))
