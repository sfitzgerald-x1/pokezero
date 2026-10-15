"""Persistent, opt-in paper worker processes with measured Q/N/M/F transport.

Default: twenty spawned processes, ten trajectories per exchange. Each process
owns its model, simulator, RNG and P; only cumulative OWN statistics travel to
the master. Public-root preparation, inference, simulation, serialization,
communication, acknowledgement and final selection are inside the decision
clock. Initial pool/model/simulator startup is reported separately, not hidden.
There is no production integration, retry, worker replacement or raw fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
import select
import struct
import signal
import subprocess
from pathlib import Path
import time
import traceback
import uuid
from typing import Any, Callable, Protocol

from .paper_reference import (
    BatchResult, DecisionState, Evaluation, ReferenceConfig, ReferenceRefusal,
    SearchResult, TrajectorySearch, World,
)
from .paper_reference_exchange import MasterSnapshot, StatisticsMaster, WorkerUpdate


@dataclass
class PreparedDecision:
    root: DecisionState
    evaluate_root: Callable[[DecisionState], Evaluation]
    sample_world: Callable[..., World]
    # Called after each batch; returns incremental evidence, not all old draws.
    evidence: Callable[[], Any] = lambda: None
    # Worker-local clock binding only; no callable crosses the public transport.
    set_sampling_deadline: Callable[[float | None], None] = lambda deadline: None
    # Opt-in worker-local labels AFTER selection; never exchanged as Q/N/M/F.
    finish_diagnostics: Callable[[Callable[[Any], None]], Any] | None = None
    # Prospective worker-local commit observation; absent on historical runtimes.
    atomic_work: Any | None = None


class WorkerRuntime(Protocol):
    def prepare(self, public_request: Any) -> PreparedDecision: ...
    def close(self) -> None: ...


class ParallelRefusal(ReferenceRefusal):
    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True)
class ParallelResult:
    result: SearchResult
    worker_pids: tuple[int, ...]
    worker_receipts: tuple[dict[str, Any], ...]
    exchange_count: int
    startup_seconds: float
    decision_id: int
    initial_dispatch_schedule: tuple[dict[str, Any], ...] = ()
    reference_timed_prefix: dict[str, Any] | None = None


def _worker_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(b"paper.reference.worker.v1\0" + seed.to_bytes(8, "big")
                            + index.to_bytes(4, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def owned_process_identity(pid):
    result = _owned_ps_result(["ps", "-p", str(pid), "-o", "lstart="],"ownership_identity")
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise ReferenceRefusal("owned process identity check failed")
    return result.stdout.strip() if result.returncode == 0 else None


def owned_process_identities(pids):
    """One bounded identity query for the exact active generations, not history."""
    if not pids or len(pids) > 20 or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ReferenceRefusal("invalid active owned process roster")
    result = _owned_ps_result(["ps", "-p", ",".join(str(pid) for pid in sorted(set(pids))),
        "-o", "pid=,lstart="],"ownership_active_identities")
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise ReferenceRefusal("active owned identity query failed")
    identities = {}
    for line in result.stdout.splitlines():
        pid, birth = line.strip().split(maxsplit=1)
        pid = int(pid)
        if pid not in pids or pid in identities:
            raise ReferenceRefusal("active owned identity response drift")
        identities[pid] = " ".join(birth.split())
    return identities


def live_owned_groups(groups):
    """Bounded process-group liveness check; no commands or unrelated data kept."""
    result = _owned_ps_result(["ps", "-axo", "pid=,pgid=,stat="],"ownership_group_liveness")
    if result.returncode != 0 or result.stderr.strip():
        raise ReferenceRefusal("owned group liveness query failed")
    live = set()
    for line in result.stdout.splitlines():
        pid, group, state = line.split()
        if int(group) in groups and not state.startswith("Z"):
            live.add(int(group))
    return live


def _owned_ps_result(command,purpose):
    from .search_over_raw_observer_cost import current_observer_window,run_owned_ps
    if current_observer_window() is not None:return run_owned_ps(command,timeout=2,purpose=purpose)[0]
    return subprocess.run(command,capture_output=True,text=True,timeout=2)


def signal_owned_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can return EPERM for a group that disappeared between TERM
        # and CONT/KILL. Suppress only when no live member remains; a genuine
        # access denial still propagates and stops the attempt.
        if live_owned_groups({group}):
            raise


def record_owned_process(pid, ownership, *, worker_index=None):
    """Parent publishes ownership BEFORE permitting the child to detach/start."""
    if ownership["controller_pid"] != os.getpid():
        raise ReferenceRefusal("owned process receipt controller drift")
    identity = owned_process_identity(pid)
    if not identity:
        raise ReferenceRefusal("owned process identity unavailable")
    directory = Path(ownership["directory"])
    generation = str(uuid.uuid4()) if ownership.get("protocol") == "active-generations-v1" else None
    path = directory / (f"group-{pid}-{generation}.json" if generation else f"group-{pid}.json")
    row = dict(pid=pid, group=pid, birth_identity=identity,
        controller_pid=os.getpid(), registration_sha256=ownership["registration_sha256"])
    if worker_index is not None:
        if type(worker_index) is not int or worker_index < 0:
            raise ReferenceRefusal("owned worker index malformed")
        row["worker_index"] = worker_index
    if generation:
        row.update(generation=generation, protocol="active-generations-v1")
    with path.open("x") as stream:
        json.dump(row, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if generation:
        # An active hardlink is an expendable index, not historical evidence.
        # It must be durable before the waiting worker may detach/runtime.
        active = directory / "active"
        active.mkdir(exist_ok=True)
        os.link(path, active / path.name)
        for parent in (directory, active):
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return path
    return None


def close_owned_generations(receipts, processes, cleanup_records=None):
    """Publish actual terminal exits and retire only the corresponding index."""
    if not receipts:
        return
    rows = {index: json.loads(path.read_text()) for index, path in receipts.items()}
    cleanup = {row["worker"]: row for row in cleanup_records or []}
    if any(processes[index].exitcode is None or processes[index].pid != row["pid"]
           or row["controller_pid"] != os.getpid() for index, row in rows.items()):
        raise ReferenceRefusal("owned generation not actually reaped")
    if live_owned_groups({row["group"] for row in rows.values()}):
        raise ReferenceRefusal("owned generation still has live descendants")
    for index, row in rows.items():
        path = receipts[index]
        receipt_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        result = dict(generation=row["generation"], launch_sha256=receipt_hash,
                pid=row["pid"], worker_exit_code=processes[index].exitcode,
                group_live=False, controller_pid=os.getpid(),
                cleanup_protocol="exact_subprocess_reap.v2")
        if cleanup_records is not None:
            witness = cleanup[index]
            result.update(cleanup_protocol="live_leader_ack_before_reap.v2",
                clean_acknowledgment=(witness["before_reap"]["state"] == "LIVE"
                    and witness["terminal_reaped"] is True and witness["actual_exitcode"] == 0
                    and witness["forced"] is False and witness["runtime_close_error"] is None
                    and not witness["cleanup_exceptions"]),
                forced_cleanup=witness["forced"], runtime_close_failed=witness["runtime_close_error"] is not None,
                cleanup_exception_count=len(witness["cleanup_exceptions"]))
        with path.with_name("closed-" + path.name).open("x") as stream:
            json.dump(result, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Only this owned generation's active index is removed. Immutable
        # launch and terminal receipts remain available for every generation.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)  # Persist the terminal directory entry first.
        finally:
            os.close(descriptor)
        (path.parent / "active" / path.name).unlink()
        descriptor = os.open(path.parent / "active", os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for parent in {path.parent for path in receipts.values()}:
        for directory in (parent, parent / "active"):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _owned_value_error_category(error):
    # Never use arbitrary subclass names, messages, notes or chained reprs.
    for kind in (ReferenceRefusal, TimeoutError, PermissionError, FileNotFoundError,
                 ValueError, TypeError, LookupError, OSError, RuntimeError, MemoryError):
        if isinstance(error, kind):
            return kind.__name__
    return "BaseException"


def _worker(connection, index, runtime_factory, config, ownership_required=False):
    runtime, prepared = None, None
    phase, decision = "startup", None
    started = time.perf_counter()
    observed_children = []
    original_popen = subprocess.Popen
    needs_generation = False

    class ObservedPopen(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed_children.append(self)

    try:
        generation = None
        needs_generation = getattr(runtime_factory, "requires_owned_value_generation", False) is True
        if ownership_required:
            permission = connection.recv()
            if needs_generation:
                if len(permission) != 2 or permission[0] != "owned_value_launch.v1":
                    raise ReferenceRefusal("owned diagnostic generation permission missing")
                generation = permission[1]
            elif permission != ("owned_launch",):
                raise ReferenceRefusal("owned launch permission missing")
        elif needs_generation:
            raise ReferenceRefusal("owned label runtime requires durable parent generation")
        # Own an isolated group, including this worker's simulator children.
        # A stopped/hung bridge cannot execute EOF cleanup after Python exits.
        if os.name == "posix":
            os.setsid()
        connection.send(("isolation", index, os.getpid(), os.getpgrp()))
        if connection.recv() != ("isolation_ack", index):
            raise ReferenceRefusal("actual worker isolation acknowledgment required before runtime")
        if needs_generation:
            from .search_over_raw_reference_values import generation_inputs
            generation_inputs(generation, worker=index)
        subprocess.Popen = ObservedPopen
        runtime = runtime_factory(index)
        if needs_generation:
            runtime.bind_owned_value_generation(generation)
        search = TrajectorySearch(config)
        battle_id, sequence = None, 0
        diagnostics_consumed = True
        connection.send(("boot", index, os.getpid(), time.perf_counter() - started,
                         os.getpgrp() if os.name == "posix" else None))
        while True:
            command = connection.recv()
            if command[0] == "close":
                break
            if command[0] == "diagnostics":
                if (command != ("diagnostics", decision) or phase != "idle"
                        or diagnostics_consumed or prepared.finish_diagnostics is None):
                    raise ReferenceRefusal("diagnostic decision is absent, stale or already consumed")
                diagnostics_consumed = True
                phase = "post_selection_diagnostics"
                evidence = prepared.finish_diagnostics(lambda row: connection.send(
                    ("diagnostics_progress", decision, index, row)))
                connection.send(("diagnostics_done", decision, index, evidence))
                phase = "idle"
                continue
            if command[0] == "restore":
                if battle_id is not None or sequence != 0:
                    raise ReferenceRefusal('worker recovery requires unused processes')
                _, snapshot, ordinal = command
                battle_id = snapshot.battle_id
                search.reset_battle(battle_id)
                search.merge_statistics(snapshot)
                search.restore_draw_ordinal(ordinal)
                connection.send(('restored', index, ordinal))
                continue
            if command[0] != "start":
                raise ReferenceRefusal("worker received an out-of-order decision command")
            _, decision, request, root, snapshot, seed, limit, deadline = command
            phase = "public_root_preparation"
            diagnostics_consumed = False
            prepared = None
            if snapshot.battle_id != battle_id:
                battle_id, sequence = snapshot.battle_id, 0
                search.reset_battle(battle_id)
            search.merge_statistics(snapshot)
            prepared_at = time.perf_counter()
            prepared = runtime.prepare(request)
            preparation_seconds = time.perf_counter() - prepared_at
            if not isinstance(prepared, PreparedDecision) or prepared.root != root:
                raise ReferenceRefusal("worker prepared a different public/actor-known root")
            prepared.set_sampling_deadline(deadline)
            while True:
                phase = "trajectory_batch"
                batch = search.search_batch(root, battle_id=battle_id,
                    evaluate_root=prepared.evaluate_root, sample_world=prepared.sample_world,
                    seed=seed, trajectories=limit, deadline_at=deadline,
                    atomic_work=prepared.atomic_work)
                sequence += 1
                update = search.export_statistics(worker_id=str(index), sequence=sequence)
                phase = "statistics_exchange"
                evidence = prepared.evidence()
                if prepared.atomic_work is not None:
                    from .search_over_raw_atomic_work import validate_reference_batch
                    validate_reference_batch(evidence, batch=batch, update=update, root=root,
                        battle_id=battle_id, worker_id=str(index), sequence=sequence)
                connection.send(("update", decision, index, update, batch,
                                 preparation_seconds, evidence))
                acknowledgement = connection.recv()
                if acknowledgement[0] != "ack" or acknowledgement[1] != decision:
                    raise ReferenceRefusal("worker received an out-of-order acknowledgement")
                _, _, snapshot, limit = acknowledgement
                search.merge_statistics(snapshot)
                preparation_seconds = 0.0
                if limit == 0:
                    connection.send(("done", decision, index))
                    break
            phase = "idle"
    except BaseException as error:
        if needs_generation:
            # Owned value failures can carry private clone events, requests or
            # snapshots. No arbitrary diagnostic or chained exception crosses
            # this opt-in boundary. The whole attempt remains nonbankable.
            category = _owned_value_error_category(error)
            failure_text = category + ": owned reference value attempt refused"
            failure_trace = ""
            failure_evidence = dict(scope="owned_reference_values.v1", error_category=category,
                private_failure_data_redacted=True, partial_batch_evidence=None, sampling_diagnostic=None)
        else:
            failure_text, failure_trace = f"{type(error).__name__}: {error}", traceback.format_exc()
            failure_evidence = {"sampling_diagnostic": getattr(error, "sampling_diagnostic", None)}
            try:
                failure_evidence["partial_batch_evidence"] = (
                    prepared.evidence() if isinstance(prepared, PreparedDecision) else None)
            except BaseException as evidence_error:
                # A broken diagnostic sink must not hide the original refusal.
                failure_evidence["partial_batch_evidence_error"] = str(evidence_error)
            try:
                pickle.dumps(failure_evidence)
            except BaseException as evidence_error:
                failure_evidence = {"diagnostic_serialization_error":
                    f"{type(evidence_error).__name__}: {evidence_error}"}
        try:
            connection.send(("error", index, decision, phase,
                failure_text, failure_trace, failure_evidence))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        close_error = None
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as error:
                close_error = (_owned_value_error_category(error) + ": owned reference value close refused"
                    if needs_generation else f"{type(error).__name__}: {error}")
        subprocess.Popen = original_popen
        live = [child.pid for child in observed_children if child.poll() is None]
        try:
            # Remain the live group leader until the owner either releases a
            # clean runtime or freezes/signals this exact unreaped scope.
            connection.send(("cleanup_ready", index, os.getpid(), close_error, live))
            while connection.recv() != ("cleanup_ack", index):
                pass  # A queued close command is not permission to exit.
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


def _nonreaping_state(process):
    """Never call multiprocessing poll/is_alive/exitcode before group cleanup."""
    if process.pid is None:
        return {"state": "NOT_STARTED", "exitcode": None}
    cached = process._popen.returncode
    if cached is not None:
        return {"state": "ALREADY_REAPED", "exitcode": cached}
    try:
        event = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return {"state": "OWNERSHIP_LOST", "exitcode": None}
    if event is not None and event.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED):
        return {"state": "TERMINAL_UNREAPED", "exitcode":
            event.si_status if event.si_code == os.CLD_EXITED else -event.si_status}
    return {"state": "LIVE", "exitcode": None}


def _freeze_original_leader(process, group):
    if group != process.pid or _nonreaping_state(process)["state"] != "LIVE":
        return False
    try:
        os.kill(process.pid, signal.SIGSTOP)
        until = time.perf_counter() + 1.
        while time.perf_counter() < until:
            event = os.waitid(os.P_PID, process.pid,
                os.WSTOPPED | os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if event is not None:
                return (event.si_code == os.CLD_STOPPED
                    and os.getpgid(process.pid) == process.pid
                    and os.getsid(process.pid) == process.pid)
            time.sleep(.01)
    except (ProcessLookupError, ChildProcessError):
        return False
    return False


def _signal_reported_children(owner, children):
    """Only original direct handles while their worker's reaping is frozen.

    This does not discover descendants or certify escaped grandchild closure.
    """
    signalled = []
    for pid in children:
        try:
            parent = subprocess.run(["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True, text=True, timeout=1., check=False)
            if parent.returncode == 0 and int(parent.stdout.strip()) == owner:
                os.kill(pid, signal.SIGKILL)
                signalled.append(pid)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass  # Missing identity is uncertainty, never a guessed signal.
    return signalled


def _bounded_message(connection, deadline):
    """Complete CPython POSIX Pipe framing, not just initial readability.

    An incomplete frame poisons transport: no retry/reassembly from a new
    worker. Exclusive ownership means no concurrent reads of this connection.
    """
    fd = connection.fileno()
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)

    def read_exact(size):
        parts = bytearray()
        while len(parts) < size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise ReferenceRefusal("worker transport frame completion timed out")
            if not select.select([fd], [], [], min(.05, remaining))[0]:
                continue
            try:
                chunk = os.read(fd, min(65536, size - len(parts)))
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("worker closed during transport frame")
            parts.extend(chunk)
        return parts

    try:
        size, = struct.unpack("!i", read_exact(4))
        if size == -1:
            size, = struct.unpack("!Q", read_exact(8))
        if not 0 < size <= 64 * 1024 * 1024:
            raise ReferenceRefusal("worker transport frame outside 64MiB limit")
        return pickle.loads(read_exact(size))
    finally:
        os.set_blocking(fd, blocking)


class ParallelTrajectorySearch:
    """Explicitly owned persistent processes; one decision at a time.

    A runtime factory must be spawn-pickleable. Never pass a live environment,
    its private serializer, an opponent request or a historical opponent action
    as the public request. The domain adapter must validate that contract.
    Failure poisons this pool. Closing terminates only processes created here,
    never a devbox, browser, production service or another caller's process.
    """

    def __init__(self, config: ReferenceConfig, runtime_factory: Callable[[int], WorkerRuntime],
                 *, workers: int = 20, batch_size: int = 10, transport_timeout: float = 120.,
                 initial_dispatch_workers: int | None = None, owned_process_receipts=None):
        if (type(workers) is not int or not 1 <= workers <= 20
                or type(batch_size) is not int or batch_size != 10
                or type(transport_timeout) not in (int, float)
                or not math.isfinite(transport_timeout) or transport_timeout <= 0):
            raise ReferenceRefusal("invalid parallel worker/exchange/timeout contract")
        if (initial_dispatch_workers is not None and (type(initial_dispatch_workers) is not int
                or not 1 <= initial_dispatch_workers <= workers)):
            raise ReferenceRefusal("initial dispatch width requires an integer within the owned worker count")
        if os.name != "posix" or not all(hasattr(os, name) for name in ("waitid", "WNOWAIT")):
            raise ReferenceRefusal("non-reaping POSIX worker ownership required")
        self.workers, self.batch_size = workers, batch_size
        self._requires_owned_value_generation = getattr(runtime_factory, "requires_owned_value_generation", False) is True
        if self._requires_owned_value_generation and (owned_process_receipts is None
                or owned_process_receipts.get("protocol") != "active-generations-v1"):
            raise ReferenceRefusal("owned labels require active immutable reference worker generations")
        # Default preserves immediate broadcast. Opt-in staggering changes only
        # when the first batch starts, never workers, particles, seeds or budget.
        self.initial_dispatch_workers = (workers if initial_dispatch_workers is None
                                         else initial_dispatch_workers)
        self.transport_timeout = transport_timeout
        self._closed = self._poisoned = False
        self._master = None
        self._decision_id = 0
        self._connections, self._processes = [], []
        self._owned_groups = {}
        self._ownership_receipts = {}
        self.worker_pids = ()
        self.startup_seconds = 0.0
        self.last_evidence = {}
        self.cleanup_evidence = {"status": "NOT_STARTED", "workers": [],
            "complete_descendant_accounting": False, "cpu_accounting_complete": False}
        started = time.perf_counter()
        context = mp.get_context("spawn")
        try:
            for index in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, index, runtime_factory, config, owned_process_receipts is not None),
                    name=f"paper-reference-{index}")
                self._connections.append(parent)
                self._processes.append(process)
                process.start()
                for prior in self._processes[:-1]:
                    if _nonreaping_state(prior)["state"] != "LIVE":
                        self._fail("worker ownership lost during pool startup", [])
                if owned_process_receipts is not None:
                    receipt = record_owned_process(process.pid, owned_process_receipts, worker_index=index)
                    if receipt is not None:
                        self._ownership_receipts[index] = receipt
                    if self._requires_owned_value_generation:
                        parent.send(("owned_value_launch.v1", dict(path=str(receipt.resolve()),
                            sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())))
                    else:
                        parent.send(("owned_launch",))
                child.close()
            pending = set(range(workers))
            pids = {}
            while pending:
                for index, message in self._receive(pending):
                    if message[0] == "isolation":
                        self._record_isolation(index, message)
                        continue
                    if message[0] != "boot" or message[1] != index:
                        self._fail("reference worker failed during startup", [message])
                    pids[index] = message[2]
                    if message[2] != self._processes[index].pid:
                        self._fail("worker boot differs from actual child", [message])
                    if os.name == "posix":
                        if message[4] != message[2] or self._owned_groups.get(index) != message[2]:
                            self._fail("worker process group is not isolated/owned", [message])
                        self._owned_groups[index] = message[4]
                    pending.remove(index)
            self.worker_pids = tuple(pids[i] for i in range(workers))
            if len(set(self.worker_pids)) != workers:
                self._fail("reference workers are not distinct processes", [])
            self.startup_seconds = time.perf_counter() - started
        except BaseException:
            try:
                self.close()
            except ParallelRefusal:
                pass  # Preserve startup failure with final cleanup evidence.
            raise

    def _record_isolation(self, index, message):
        if (len(message) != 4 or message[1] != index or type(message[2]) is not int
                or message[2] != self._processes[index].pid or message[3] != message[2]
                or index in self._owned_groups
                or _nonreaping_state(self._processes[index])["state"] != "LIVE"
                or os.getpgid(message[2]) != message[2] or os.getsid(message[2]) != message[2]):
            self._fail("worker isolation receipt differs from actual child", [message])
        self._owned_groups[index] = message[3]
        self._connections[index].send(("isolation_ack", index))

    def _fail(self, message, errors):
        self._poisoned = True
        self.last_evidence["errors"] = errors
        raise ParallelRefusal(message, self.last_evidence)

    def restore_statistics(self, snapshot: MasterSnapshot, *,
                           worker_ordinals: tuple[int, ...] | None = None,
                           decision_id: int = 0) -> None:
        """Fresh-pool aggregate recovery, imported exactly once, never OWN work.

        Old processes/P caches are NOT resurrected. Explicit draw ordinals can
        preserve the accepted-prefix RNG domains in fresh processes. A contribution
        holds the accepted checkpoint while fresh workers export only new work.
        Failed/unplayed partial batches must not be present in the checkpoint.
        """
        if (self._closed or self._poisoned or self._master is not None or self._decision_id != 0
                or not isinstance(snapshot, MasterSnapshot)):
            raise ReferenceRefusal('statistics recovery requires a fresh, unused pool')
        if (type(decision_id) is not int or decision_id < 0
                or worker_ordinals is not None and (
                    not isinstance(worker_ordinals, tuple) or len(worker_ordinals) != self.workers
                    or any(type(value) is not int or value < 0 for value in worker_ordinals))
                or worker_ordinals is None and decision_id != 0):
            raise ReferenceRefusal('invalid accepted-prefix worker/decision recovery positions')
        master = StatisticsMaster(snapshot.battle_id)
        master.advance_real_faints(snapshot.faint_floor)
        master.accept(WorkerUpdate(snapshot.battle_id, 'retained-accepted-history', 1, snapshot.rows))
        self._master = master
        if worker_ordinals is not None:
            try:
                for index, connection in enumerate(self._connections):
                    connection.send(('restore', master.snapshot(), worker_ordinals[index]))
                pending = set(range(self.workers))
                while pending:
                    for index, message in self._receive(pending):
                        if message != ('restored', index, worker_ordinals[index]):
                            self._fail('worker accepted-prefix recovery failed', [message])
                        pending.remove(index)
                self._decision_id = decision_id
            except BaseException:
                self._poisoned = True
                raise

    def _receive(self, pending):
        deadline = time.perf_counter() + self.transport_timeout
        ready = wait([self._connections[i] for i in sorted(pending)], timeout=self.transport_timeout)
        if not ready:
            self._fail("reference worker transport stalled; no silent retry", [
                {"worker": i, **_nonreaping_state(self._processes[i])} for i in sorted(pending)])
        indices = {self._connections[i]: i for i in pending}
        for connection in ready:
            index = indices[connection]
            try:
                yield index, _bounded_message(connection, deadline)
            except (EOFError, OSError, ReferenceRefusal, pickle.UnpicklingError) as error:
                self._fail("reference worker connection was lost", [
                    {"worker": index, "error": str(error), **_nonreaping_state(self._processes[index])}])

    def search(self, public_request: Any, root: DecisionState, *, battle_id: str, seed: int,
               trajectories_per_worker: int | None = None,
               deadline_seconds: float | None = None, timed_zero_work: bool = False,
               deadline_at: float | None = None, adapter_started_at: float | None = None) -> ParallelResult:
        started = time.perf_counter()  # Before public-root serialization/broadcast.
        if (self._closed or self._poisoned or not isinstance(root, DecisionState)
                or not isinstance(battle_id, str) or not battle_id
                or type(seed) is not int or not 0 <= seed < 2**64
                or (trajectories_per_worker is not None and (
                    type(trajectories_per_worker) is not int or trajectories_per_worker <= 0))
                or (deadline_seconds is not None and (type(deadline_seconds) not in (int, float)
                    or not math.isfinite(deadline_seconds) or deadline_seconds <= 0))
                or (trajectories_per_worker is None and deadline_seconds is None)):
            raise ReferenceRefusal("invalid parallel decision/work/deadline contract")
        if (type(timed_zero_work) is not bool
                or (not timed_zero_work and (deadline_at is not None or adapter_started_at is not None))
                or (timed_zero_work and (self.workers != 20 or self.initial_dispatch_workers != 6
                    or len(root.actions) < 2 or trajectories_per_worker is not None
                    or deadline_seconds not in (1., 3., 10.)
                    or type(deadline_at) not in (int, float) or not math.isfinite(deadline_at)
                    or type(adapter_started_at) not in (int, float) or not math.isfinite(adapter_started_at)
                    or not 0 <= adapter_started_at <= started
                    or deadline_at != adapter_started_at + deadline_seconds
                    or self._decision_id != 0 or self._master is not None))):
            raise ReferenceRefusal("timed zero work requires the explicit fresh twenty-worker deadline contract")
        self._decision_id += 1
        decision = self._decision_id
        deadline = (deadline_at if timed_zero_work else
                    None if deadline_seconds is None else started + deadline_seconds)
        receipts, progress = [], [0] * self.workers
        self.last_evidence = {"decision_id": decision, "battle_id": battle_id,
            "worker_pids": self.worker_pids, "receipts": receipts, "errors": []}
        dispatch_schedule = []
        self.last_evidence['initial_dispatch_workers'] = self.initial_dispatch_workers
        self.last_evidence['initial_dispatch_schedule'] = dispatch_schedule
        try:
            if self._master is None or self._master.battle_id != battle_id:
                self._master = StatisticsMaster(battle_id)
            self._master.advance_real_faints(root.faint_count)
            snapshot = self._master.snapshot()
            before = next((row.count for row in snapshot.rows if row.state == root), 0)
            first = min(self.batch_size, trajectories_per_worker or self.batch_size)
            queued = list(range(self.initial_dispatch_workers, self.workers))
            first_updates, done_workers, stop_acks = set(), set(), {}
            def dispatch(index, snapshot, released_by=None):
                # Public capture, serialization and queue delay are all inside
                # the original absolute deadline. A late worker receives that
                # expired deadline and reports zero work, not a fresh budget.
                at = time.perf_counter()
                dispatch_schedule.append(dict(worker=index, seconds_from_decision_start=at-started,
                    released_by_first_update=released_by, master_version=snapshot.version,
                    deadline_expired=deadline is not None and at >= deadline))
                self._connections[index].send(("start", decision, public_request, root, snapshot,
                                               _worker_seed(seed, index), first, deadline))
            for index in range(self.initial_dispatch_workers):
                dispatch(index, snapshot)
            pending = set(range(self.workers))
            while pending:
                for index, message in self._receive(pending):
                    if message[0] == "error":
                        self._fail("reference worker refused; decision is nonbankable", [message])
                    if message[0] not in ("update", "done") or message[1:3] != (decision, index):
                        self._fail("reference process/decision protocol drift", [message])
                    if message[0] == "done":
                        if timed_zero_work and (index not in first_updates or index in done_workers
                                or index not in stop_acks):
                            self._fail("reference done lacks its actual stop acknowledgement", [message])
                        done_workers.add(index)
                        pending.remove(index)
                        continue
                    if timed_zero_work and index in stop_acks:
                        self._fail("reference update arrived after its stop acknowledgement", [message])
                    _, _, _, update, batch, preparation, evidence = message
                    if (not isinstance(update, WorkerUpdate) or update.worker_id != str(index)
                            or not isinstance(batch, BatchResult)
                            or not 0 <= batch.trajectories <= self.batch_size):
                        self._fail("invalid worker contribution/work receipt", [message])
                    if isinstance(evidence, dict) and "atomic_work" in evidence:
                        from .search_over_raw_atomic_work import validate_reference_batch, update_record
                        validate_reference_batch(evidence, batch=batch, update=update, root=root,
                            battle_id=battle_id, worker_id=str(index), sequence=update.sequence)
                    snapshot = self._master.accept(update)
                    progress[index] += batch.trajectories
                    receipts.append({"worker": index, "sequence": update.sequence,
                        "batch": batch, "preparation_seconds": preparation,
                        "evidence": evidence, "master_version": snapshot.version,
                        **({"own_statistics": update_record(update)} if isinstance(evidence, dict)
                           and "atomic_work" in evidence else {})})
                    if index not in first_updates:
                        first_updates.add(index)
                        if queued:
                            # FIFO activation, irrespective of outcomes, rewards,
                            # acceptance or throughput. No worker/draw is retried.
                            dispatch(queued.pop(0), snapshot, released_by=index)
                    remaining = (self.batch_size if trajectories_per_worker is None else
                                 trajectories_per_worker - progress[index])
                    ack_decided_at = time.perf_counter()
                    expired = deadline is not None and ack_decided_at >= deadline
                    limit = 0 if expired or batch.deadline_exhausted or remaining <= 0 else min(
                        self.batch_size, remaining)
                    # Every last update gets acknowledged, including empty expiry.
                    self._connections[index].send(("ack", decision, snapshot, limit))
                    if timed_zero_work:
                        acknowledgement = dict(worker=index, sequence=update.sequence,
                            master_version=snapshot.version, limit=limit,
                            decided_monotonic=ack_decided_at,
                            acknowledged_monotonic=time.perf_counter(), deadline_expired=expired,
                            worker_deadline_exhausted=batch.deadline_exhausted)
                        receipts[-1]["timed_acknowledgement"] = acknowledgement
                        if limit == 0:
                            stop_acks[index] = acknowledgement
            completed = sum(progress)
            snapshot = self._master.snapshot()
            row = next((row for row in snapshot.rows if row.state == root), None)
            if completed == 0:
                if not timed_zero_work:
                    self._fail("zero NEW complete trajectories; no stale-action or raw fallback", [])
                if (before != 0 or time.perf_counter() < deadline
                        or done_workers != set(range(20)) or first_updates != done_workers
                        or any(not r["batch"].deadline_exhausted for r in receipts)):
                    self._fail("zero work lacks the complete normal fresh-root deadline protocol", [])
            # A legitimate trajectory can revisit the same information state;
            # N/M count visits, not distinct completed trajectories. Every new
            # trajectory must back up its starting root at least once.
            if completed and (row is None or row.count - before < completed):
                self._fail("master root counts omit completed new worker work", [])
            visits = (0,) * len(root.actions) if row is None else row.visits
            totals = (0.,) * len(root.actions) if row is None else row.totals
            action = (root.actions[max(range(len(root.actions)), key=lambda i: (visits[i], -i))]
                      if completed else None)
            elapsed = time.perf_counter() - started
            result = SearchResult(action, completed,
                sum(r["batch"].transitions for r in receipts),
                sum(r["batch"].world_draws for r in receipts), visits,
                tuple(q / n if n else None for q, n in zip(totals, visits)), elapsed,
                deadline is not None and started + elapsed >= deadline,
                0.0 if deadline is None else max(0., started + elapsed - deadline))
            prefix = None
            if timed_zero_work:
                prefix = dict(schema="pokezero.search-over-raw.timed-reference-prefix.v1",
                    battle_id=battle_id, selection_seed=seed, decision_id=decision,
                    nominal_search_seconds=deadline_seconds, decision_started_monotonic=started,
                    adapter_started_monotonic=adapter_started_at,
                    absolute_deadline_monotonic=deadline, decision_finished_monotonic=started+elapsed,
                    initial_dispatch_workers=6, acknowledged_done_workers=sorted(done_workers),
                    stop_acknowledgements=[stop_acks[index] for index in sorted(stop_acks)],
                    status="SELECTED" if completed else "UNQUALIFIED_AT_CEILING",
                    retry_authorized=False, runtime_qualified=False,
                    on_time_deployability_established=False)
                # This prospective pool is one-shot, even when its result is positive.
                self._poisoned = True
            self.last_evidence.update(status="COMPLETE" if completed else "UNQUALIFIED_AT_CEILING", result=result)
            self._last_result = ParallelResult(result, self.worker_pids, tuple(receipts), len(receipts),
                                  self.startup_seconds, decision, tuple(dispatch_schedule), prefix)
            self._diagnostics_consumed = False
            return self._last_result
        except BaseException:
            self._poisoned = True
            raise

    def finish_diagnostics(self, selected: ParallelResult):
        """Consume this exact completed decision once, without changing statistics.

        Label clocks are owned by the opt-in runtime's original producer contract.
        No new deadline, snapshot or callable is supplied over public transport.
        """
        if (self._closed or self._poisoned or selected is not getattr(self, "_last_result", None)
                or getattr(self, "_diagnostics_consumed", True)):
            raise ReferenceRefusal("diagnostics require the exact latest unconsumed selection")
        self._diagnostics_consumed = True
        rows = []
        progress = {}
        self.last_evidence["diagnostic_progress"] = progress
        try:
            for connection in self._connections:
                connection.send(("diagnostics", selected.decision_id))
            pending = set(range(self.workers))
            while pending:
                for index, message in self._receive(pending):
                    if message[:3] == ("diagnostics_progress", selected.decision_id, index) and len(message) == 4:
                        row = message[3]
                        previous = progress.get(index, dict(boundaries=0, labels_accounted=0))
                        if (type(row) is not dict or row.get("schema") != "pokezero.visited-value-progress.v1"
                                or any(type(row.get(field)) is not int or row[field] < previous[field]
                                    for field in ("boundaries", "labels_accounted"))
                                or (row["boundaries"], row["labels_accounted"]) ==
                                    (previous["boundaries"], previous["labels_accounted"])):
                            self._fail("invalid reference diagnostic progress", [message])
                        progress[index] = row
                        continue
                    if message[:3] != ("diagnostics_done", selected.decision_id, index) or len(message) != 4:
                        self._fail("reference post-selection diagnostic failed", [message])
                    rows.append(dict(worker=index, evidence=message[3]))
                    pending.remove(index)
            return tuple(sorted(rows, key=lambda row: row["worker"]))
        except BaseException:
            self._poisoned = True
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        started = time.perf_counter()
        replies, errors = {}, []
        for index, connection in enumerate(self._connections):
            try:
                connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                pass
            except BaseException as error:
                errors.append({"worker": index, "error": type(error).__name__})
        # One shared grace interval, not workers * runtime-close grace. Leaders
        # wait for acknowledgement and are never reaped during this interval.
        until = time.perf_counter() + 6.
        pending = {i for i, p in enumerate(self._processes) if p.pid is not None}
        while pending and time.perf_counter() < until:
            try:
                ready = wait([self._connections[i] for i in pending], timeout=min(.05, max(0., until - time.perf_counter())))
            except BaseException as error:
                errors.append({"transport_wait_error": type(error).__name__})
                break  # Still attempt every original worker below.
            for i in tuple(pending):
                try:
                    if self._connections[i] not in ready:
                        if _nonreaping_state(self._processes[i])["state"] != "LIVE":
                            errors.append({"worker": i, "error": "leader terminated before cleanup handshake"})
                            pending.remove(i)
                        continue
                    message = _bounded_message(self._connections[i], until)
                    if not isinstance(message, (tuple, list)) or not message:
                        raise ReferenceRefusal("malformed cleanup transport envelope")
                    if message[0] == "isolation":
                        self._record_isolation(i, message)
                    elif message[0] == "cleanup_ready":
                        if (len(message) != 5 or message[1] != i or message[2] != self._processes[i].pid
                                or message[3] is not None and not isinstance(message[3], str)
                                or not isinstance(message[4], list)
                                or any(type(pid) is not int or pid <= 1 for pid in message[4])):
                            raise ReferenceRefusal("malformed cleanup handshake")
                        replies[i] = message
                        pending.remove(i)
                    elif message[0] not in ("boot", "error", "done", "update", "restored", "diagnostics_progress", "diagnostics_done"):
                        raise ReferenceRefusal("unexpected cleanup transport")
                except BaseException as error:
                    errors.append({"worker": i, "error": f"{type(error).__name__}: {error}"})
                    pending.remove(i)
        records = []
        for index, process in enumerate(self._processes):
            if process.pid is None:
                continue
            record = self._close_original_worker(index, process, replies.get(index))
            records.append(record)
            if (record["forced"] or record["before_reap"]["state"] != "LIVE"
                    or record["actual_exitcode"] != 0 or record["cleanup_exceptions"]):
                errors.append(record)
        for connection in self._connections:
            try:
                connection.close()
            except BaseException as error:
                errors.append({"connection_close_error": type(error).__name__})
        try:
            close_owned_generations(self._ownership_receipts, self._processes, records)
        except BaseException as error:
            errors.append({"owned_generation_close_error": type(error).__name__})
        self.cleanup_evidence = {"status": "REFUSED" if errors else "CLEAN_OBSERVED_SCOPE",
            "workers": records, "errors": errors, "wall_seconds": time.perf_counter() - started,
            "complete_descendant_accounting": False, "cpu_accounting_complete": False}
        self.last_evidence["cleanup"] = self.cleanup_evidence
        if errors:
            self._poisoned = True
            raise ParallelRefusal("reference worker cleanup refused", self.last_evidence)

    def _close_original_worker(self, index, process, reply):
        before = {"state": "INSPECTION_FAILED", "exitcode": None}
        actual = before
        forced = not reply or reply[3] is not None or bool(reply[4])
        group_signalled = False
        children_signalled, exceptions = [], []
        try:
            before = _nonreaping_state(process)
            if before["state"] == "LIVE" and not forced:
                try:
                    self._connections[index].send(("cleanup_ack", index))
                except (BrokenPipeError, EOFError, OSError):
                    forced = True
            if forced and before["state"] == "LIVE":
                if _freeze_original_leader(process, self._owned_groups.get(index)):
                    children_signalled = _signal_reported_children(process.pid, reply[4] if reply else [])
                    if _nonreaping_state(process)["state"] == "LIVE":
                        os.killpg(process.pid, signal.SIGKILL)
                        group_signalled = True
                elif _nonreaping_state(process)["state"] == "LIVE":
                    # Only this owned unreaped process, not an unvalidated group.
                    os.kill(process.pid, signal.SIGKILL)
        except BaseException as error:
            forced = True
            exceptions.append(f"{type(error).__name__}: {error}")
            try:
                if _nonreaping_state(process)["state"] == "LIVE":
                    os.kill(process.pid, signal.SIGKILL)  # Exact unreaped child only.
            except BaseException as fallback_error:
                exceptions.append(f"{type(fallback_error).__name__}: {fallback_error}")
        finally:
            # FIRST reaping operation; no group signal follows it, including
            # fallback paths. An exception cannot strand later owned workers.
            try:
                process.join(timeout=2.)
                actual = _nonreaping_state(process)
                if actual["state"] == "LIVE":
                    forced = True
                    os.kill(process.pid, signal.SIGKILL)
                    process.join(timeout=2.)
                    actual = _nonreaping_state(process)
            except BaseException as error:
                exceptions.append(f"{type(error).__name__}: {error}")
                try:
                    if _nonreaping_state(process)["state"] == "LIVE":
                        os.kill(process.pid, signal.SIGKILL)
                    process.join(timeout=2.)
                    actual = _nonreaping_state(process)
                except BaseException as final_error:
                    exceptions.append(f"{type(final_error).__name__}: {final_error}")
        return {"worker": index, "pid": process.pid, "before_reap": before,
            "runtime_close_error": reply[3] if reply else None,
            "live_children_reported": reply[4] if reply else None,
            "group_signalled_before_reap": group_signalled,
            "reported_children_signalled": children_signalled,
            "forced": forced, "actual_exitcode": actual["exitcode"],
            "terminal_reaped": actual["state"] == "ALREADY_REAPED",
            "cleanup_exceptions": exceptions}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            self.close()
        except ParallelRefusal:
            if exc_type is None:
                raise
            # Retain cleanup refusal without masking an existing search failure.
