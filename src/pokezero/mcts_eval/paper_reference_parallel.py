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
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import pickle
import select
import signal
import struct
import subprocess
import time
import traceback
from typing import Any, Callable, Protocol

from ..bridge_error_sites import safe_bridge_error_origin, safe_bridge_error_reason, safe_bridge_error_sites
from .paper_reference import (
    BatchResult, DecisionState, Evaluation, ReferenceConfig, ReferenceRefusal,
    SearchResult, TrajectorySearch, World,
)
from .paper_reference_exchange import StatisticsMaster, WorkerUpdate


@dataclass
class PreparedDecision:
    root: DecisionState
    evaluate_root: Callable[[DecisionState], Evaluation]
    sample_world: Callable[..., World]
    # Called after each batch; returns incremental evidence, not all old draws.
    evidence: Callable[[], Any] = lambda: None
    # Prospective worker-local commit observation; absent on historical runtimes.
    atomic_work: Any | None = None
    # Opt-in sampler gets the SAME absolute deadline as trajectory search.
    bind_sampling_deadline: Callable[[float | None], None] | None = None


class WorkerRuntime(Protocol):
    def prepare(self, public_request: Any) -> PreparedDecision: ...
    def close(self) -> None: ...


class ParallelRefusal(ReferenceRefusal):
    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_FAILURE_PHASES = frozenset({"startup", "public_root_preparation", "trajectory_batch",
                           "statistics_exchange", "idle"})
_ERROR_CATEGORIES = frozenset({"ReferenceRefusal", "ValueError", "TypeError", "KeyError",
                              "RuntimeError", "OSError", "OtherException"})


def _code_file(value):
    """Accept only actual package source paths, never absolute/private paths."""
    if (not isinstance(value, str) or not value.endswith(".py") or len(value) > 160
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/." for c in value)):
        return None
    try:
        path = (_PACKAGE_ROOT / value).resolve()
        relative = path.relative_to(_PACKAGE_ROOT).as_posix()
        return relative if relative == value and path.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None


def _exception_code_sites(error):
    """Bounded causal code locations only: no messages, locals or source text."""
    chain, seen = [], set()
    for _ in range(4):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        category = ("ReferenceRefusal" if isinstance(error, ReferenceRefusal) else
                    next((name for cls, name in ((ValueError, "ValueError"), (TypeError, "TypeError"),
                         (KeyError, "KeyError"), (RuntimeError, "RuntimeError"), (OSError, "OSError"))
                         if type(error) is cls), "OtherException"))
        frames, trace = [], error.__traceback__
        for _ in range(64):
            if trace is None:
                break
            try:
                relative = Path(trace.tb_frame.f_code.co_filename).resolve().relative_to(_PACKAGE_ROOT).as_posix()
            except ValueError:
                relative = None
            if relative is not None and _code_file(relative) is not None:
                frames.append({"file": relative, "line": trace.tb_lineno})
            trace = trace.tb_next
        entry = {"category": category, "frames": frames[-8:]}
        try:
            bridge = safe_bridge_error_sites(getattr(error, "bridge_code_sites", None))
        except Exception:
            bridge = []  # Diagnostics must not replace the original refusal.
        if bridge:
            entry["bridge_frames"] = bridge
        try:
            origin = safe_bridge_error_origin(getattr(error, "bridge_error_origin", None))
        except Exception:
            origin = None
        if origin is not None:
            entry["bridge_origin"] = origin
        try:
            reason = safe_bridge_error_reason(getattr(error, "bridge_error_reason", None))
        except Exception:
            reason = None
        if reason is not None:
            entry["bridge_reason"] = reason
        chain.append(entry)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return chain


def sanitized_worker_failures(errors):
    """Project internal error transports onto a closed, privacy-safe schema."""
    result = []
    if not isinstance(errors, (list, tuple)):
        return result
    for message in errors[:20]:
        if (not isinstance(message, (list, tuple)) or len(message) != 7 or message[0] != "error"
                or type(message[1]) is not int or not 0 <= message[1] < 20
                or (message[2] is not None and (type(message[2]) is not int or not 0 <= message[2] < 2**64))
                or message[3] not in tuple(_FAILURE_PHASES) or not isinstance(message[6], dict)):
            continue
        sites = message[6].get("error_code_sites")
        if not isinstance(sites, list) or not 1 <= len(sites) <= 4:
            continue
        safe = []
        for entry in sites:
            if (not isinstance(entry, dict) or not {"category", "frames"} <= set(entry)
                    or not set(entry) <= {"category", "frames", "bridge_frames", "bridge_origin", "bridge_reason"}
                    or entry["category"] not in tuple(_ERROR_CATEGORIES)
                    or not isinstance(entry["frames"], list) or len(entry["frames"]) > 8):
                break
            frames = entry["frames"]
            if any(not isinstance(row, dict) or set(row) != {"file", "line"}
                   or _code_file(row["file"]) is None or type(row["line"]) is not int
                   or not 0 < row["line"] <= 10**7 for row in frames):
                break
            projected = {"category": entry["category"], "frames": [dict(row) for row in frames]}
            if "bridge_frames" in entry:
                bridge = safe_bridge_error_sites(entry["bridge_frames"])
                if not bridge:
                    break
                projected["bridge_frames"] = bridge
            if "bridge_origin" in entry:
                origin = safe_bridge_error_origin(entry["bridge_origin"])
                if origin is None:
                    break
                projected["bridge_origin"] = origin
            if "bridge_reason" in entry:
                reason = safe_bridge_error_reason(entry["bridge_reason"])
                if reason is None:
                    break
                projected["bridge_reason"] = reason
            safe.append(projected)
        else:
            result.append({"worker": message[1], "decision_id": message[2], "phase": message[3],
                           "error_code_sites": safe})
    return result


@dataclass(frozen=True)
class ParallelResult:
    result: SearchResult
    worker_pids: tuple[int, ...]
    worker_receipts: tuple[dict[str, Any], ...]
    exchange_count: int
    startup_seconds: float
    decision_id: int


def _worker_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(b"paper.reference.worker.v1\0" + seed.to_bytes(8, "big")
                            + index.to_bytes(4, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def _worker(connection, index, runtime_factory, config):
    runtime, prepared = None, None
    phase, decision = "startup", None
    started = time.perf_counter()
    original_popen = subprocess.Popen
    observed_children = []

    class OwnedPopen(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed_children.append(self)

    try:
        # Own an isolated group, including this worker's simulator children.
        # A stopped/hung bridge cannot execute EOF cleanup after Python exits.
        if os.name == "posix":
            os.setsid()
        connection.send(("isolation", index, os.getpid(), os.getpgrp()))
        if connection.recv() != ("isolation_ack", index):
            raise ReferenceRefusal("runtime construction lacks original isolation acknowledgement")
        subprocess.Popen = OwnedPopen
        runtime = runtime_factory(index)
        search = TrajectorySearch(config)
        battle_id, sequence = None, 0
        connection.send(("boot", index, os.getpid(), time.perf_counter() - started,
                         os.getpgrp() if os.name == "posix" else None))
        while True:
            command = connection.recv()
            if command[0] == "close":
                break
            if command[0] != "start":
                raise ReferenceRefusal("worker received an out-of-order decision command")
            _, decision, request, root, snapshot, seed, limit, deadline = command
            phase = "public_root_preparation"
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
            if prepared.bind_sampling_deadline is not None:
                prepared.bind_sampling_deadline(deadline)
            while True:
                phase = "trajectory_batch"
                batch = search.search_batch(root, battle_id=battle_id,
                    evaluate_root=prepared.evaluate_root, sample_world=prepared.sample_world,
                    seed=seed, trajectories=limit, deadline_at=deadline,
                    atomic_work=prepared.atomic_work,
                    allow_sampling_deadline=prepared.bind_sampling_deadline is not None)
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
        # Kept even when the optional evidence sink failed or was unpicklable.
        # Never let diagnostic construction replace the actual refusal.
        try:
            failure_evidence["error_code_sites"] = _exception_code_sites(error)
        except Exception:
            pass
        try:
            connection.send(("error", index, decision, phase,
                f"{type(error).__name__}: {error}", traceback.format_exc(),
                failure_evidence))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        close_error = None
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as error:
                close_error = f"{type(error).__name__}: {error}"
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
                 *, workers: int = 20, batch_size: int = 10, transport_timeout: float = 120.):
        if (type(workers) is not int or not 1 <= workers <= 20
                or type(batch_size) is not int or batch_size != 10
                or type(transport_timeout) not in (int, float)
                or not math.isfinite(transport_timeout) or transport_timeout <= 0):
            raise ReferenceRefusal("invalid parallel worker/exchange/timeout contract")
        if os.name != "posix" or not all(hasattr(os, name) for name in ("waitid", "WNOWAIT")):
            raise ReferenceRefusal("non-reaping POSIX worker ownership required")
        self.workers, self.batch_size = workers, batch_size
        self.transport_timeout = transport_timeout
        self._closed = self._poisoned = False
        self._master = None
        self._decision_id = 0
        self._connections, self._processes = [], []
        self._owned_groups = {}
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
                    args=(child, index, runtime_factory, config), name=f"paper-reference-{index}")
                self._connections.append(parent)
                self._processes.append(process)
                process.start()
                # Process.start() polls older children. A terminal leader that
                # it reaped loses signalling authority; never infer otherwise.
                for prior in self._processes[:-1]:
                    if _nonreaping_state(prior)["state"] != "LIVE":
                        self._fail("worker ownership lost during pool startup", [])
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
                pass  # Preserve startup error; cleanup_evidence remains bound.
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
               deadline_seconds: float | None = None) -> ParallelResult:
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
        self._decision_id += 1
        decision = self._decision_id
        deadline = None if deadline_seconds is None else started + deadline_seconds
        receipts, progress = [], [0] * self.workers
        self.last_evidence = {"decision_id": decision, "battle_id": battle_id,
            "worker_pids": self.worker_pids, "receipts": receipts, "errors": []}
        try:
            if self._master is None or self._master.battle_id != battle_id:
                self._master = StatisticsMaster(battle_id)
            self._master.advance_real_faints(root.faint_count)
            snapshot = self._master.snapshot()
            before = next((row.count for row in snapshot.rows if row.state == root), 0)
            first = min(self.batch_size, trajectories_per_worker or self.batch_size)
            for index, connection in enumerate(self._connections):
                connection.send(("start", decision, public_request, root, snapshot,
                                 _worker_seed(seed, index), first, deadline))
            pending = set(range(self.workers))
            while pending:
                for index, message in self._receive(pending):
                    if message[0] == "error":
                        self._fail("reference worker refused; decision is nonbankable", [message])
                    if message[0] not in ("update", "done") or message[1:3] != (decision, index):
                        self._fail("reference process/decision protocol drift", [message])
                    if message[0] == "done":
                        pending.remove(index)
                        continue
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
                    remaining = (self.batch_size if trajectories_per_worker is None else
                                 trajectories_per_worker - progress[index])
                    expired = deadline is not None and time.perf_counter() >= deadline
                    limit = 0 if expired or batch.deadline_exhausted or remaining <= 0 else min(
                        self.batch_size, remaining)
                    # Every last update gets acknowledged, including empty expiry.
                    self._connections[index].send(("ack", decision, snapshot, limit))
            completed = sum(progress)
            snapshot = self._master.snapshot()
            row = next((row for row in snapshot.rows if row.state == root), None)
            if completed == 0:
                self._fail("zero NEW complete trajectories; no stale-action or raw fallback", [])
            # A legitimate trajectory can revisit the same information state;
            # N/M count visits, not distinct completed trajectories. Every new
            # trajectory must back up its starting root at least once.
            if row is None or row.count - before < completed:
                self._fail("master root counts omit completed new worker work", [])
            action_index = max(range(len(root.actions)), key=lambda i: (row.visits[i], -i))
            elapsed = time.perf_counter() - started
            result = SearchResult(root.actions[action_index], completed,
                sum(r["batch"].transitions for r in receipts),
                sum(r["batch"].world_draws for r in receipts), row.visits,
                tuple(q / n if n else None for q, n in zip(row.totals, row.visits)), elapsed,
                deadline is not None and started + elapsed >= deadline,
                0.0 if deadline_seconds is None else max(0., elapsed - deadline_seconds))
            self.last_evidence.update(status="COMPLETE", result=result)
            return ParallelResult(result, self.worker_pids, tuple(receipts), len(receipts),
                                  self.startup_seconds, decision)
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
                    elif message[0] not in ("boot", "error", "done", "update"):
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
