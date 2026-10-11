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


def _worker_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(b"paper.reference.worker.v1\0" + seed.to_bytes(8, "big")
                            + index.to_bytes(4, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def owned_process_identity(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True, text=True, timeout=2)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise ReferenceRefusal("owned process identity check failed")
    return result.stdout.strip() if result.returncode == 0 else None


def owned_process_identities(pids):
    """One bounded identity query for the exact active generations, not history."""
    if not pids or len(pids) > 20 or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ReferenceRefusal("invalid active owned process roster")
    result = subprocess.run(["ps", "-p", ",".join(str(pid) for pid in sorted(set(pids))),
        "-o", "pid=,lstart="], capture_output=True, text=True, timeout=2)
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
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="],
        capture_output=True, text=True, timeout=2)
    if result.returncode != 0 or result.stderr.strip():
        raise ReferenceRefusal("owned group liveness query failed")
    live = set()
    for line in result.stdout.splitlines():
        pid, group, state = line.split()
        if int(group) in groups and not state.startswith("Z"):
            live.add(int(group))
    return live


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


def record_owned_process(pid, ownership):
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


def close_owned_generations(receipts, processes):
    """Publish actual terminal exits and retire only the corresponding index."""
    if not receipts:
        return
    rows = {index: json.loads(path.read_text()) for index, path in receipts.items()}
    if any(processes[index].exitcode is None or processes[index].pid != row["pid"]
           or row["controller_pid"] != os.getpid() for index, row in rows.items()):
        raise ReferenceRefusal("owned generation not actually reaped")
    if live_owned_groups({row["group"] for row in rows.values()}):
        raise ReferenceRefusal("owned generation still has live descendants")
    for index, row in rows.items():
        path = receipts[index]
        receipt_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.with_name("closed-" + path.name).open("x") as stream:
            json.dump(dict(generation=row["generation"], launch_sha256=receipt_hash,
                pid=row["pid"], worker_exit_code=processes[index].exitcode,
                group_live=False, controller_pid=os.getpid()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Only this owned generation's active index is removed. Immutable
        # launch and terminal receipts remain available for every generation.
        (path.parent / "active" / path.name).unlink()
    for parent in {path.parent for path in receipts.values()}:
        for directory in (parent, parent / "active"):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _worker(connection, index, runtime_factory, config, ownership_required=False):
    runtime, prepared = None, None
    phase, decision = "startup", None
    started = time.perf_counter()
    try:
        if ownership_required and connection.recv() != ("owned_launch",):
            raise ReferenceRefusal("owned launch permission missing")
        # Own an isolated group, including this worker's simulator children.
        # A stopped/hung bridge cannot execute EOF cleanup after Python exits.
        if os.name == "posix":
            os.setsid()
        runtime = runtime_factory(index)
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
                    seed=seed, trajectories=limit, deadline_at=deadline)
                sequence += 1
                update = search.export_statistics(worker_id=str(index), sequence=sequence)
                phase = "statistics_exchange"
                connection.send(("update", decision, index, update, batch,
                                 preparation_seconds, prepared.evidence()))
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
        try:
            connection.send(("error", index, decision, phase,
                f"{type(error).__name__}: {error}", traceback.format_exc(),
                failure_evidence))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as error:
                try:
                    connection.send(("close_error", index, str(error)))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        connection.close()


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
        self.workers, self.batch_size = workers, batch_size
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
                if owned_process_receipts is not None:
                    receipt = record_owned_process(process.pid, owned_process_receipts)
                    if receipt is not None:
                        self._ownership_receipts[index] = receipt
                    parent.send(("owned_launch",))
                child.close()
            pending = set(range(workers))
            pids = {}
            while pending:
                for index, message in self._receive(pending):
                    if message[0] != "boot" or message[1] != index:
                        self._fail("reference worker failed during startup", [message])
                    pids[index] = message[2]
                    if os.name == "posix":
                        if message[4] != message[2]:
                            self._fail("worker process group is not isolated/owned", [message])
                        self._owned_groups[index] = message[4]
                    pending.remove(index)
            self.worker_pids = tuple(pids[i] for i in range(workers))
            if len(set(self.worker_pids)) != workers:
                self._fail("reference workers are not distinct processes", [])
            self.startup_seconds = time.perf_counter() - started
        except BaseException:
            self.close()
            raise

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
        ready = wait([self._connections[i] for i in sorted(pending)], timeout=self.transport_timeout)
        if not ready:
            self._fail("reference worker transport stalled; no silent retry", [
                {"worker": i, "alive": self._processes[i].is_alive(),
                 "exitcode": self._processes[i].exitcode} for i in sorted(pending)])
        indices = {self._connections[i]: i for i in pending}
        for connection in ready:
            index = indices[connection]
            try:
                yield index, connection.recv()
            except (EOFError, OSError) as error:
                self._fail("reference worker connection was lost", [
                    {"worker": index, "error": str(error), "exitcode": self._processes[index].exitcode}])

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
            first_updates = set()
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
                        pending.remove(index)
                        continue
                    _, _, _, update, batch, preparation, evidence = message
                    if (not isinstance(update, WorkerUpdate) or update.worker_id != str(index)
                            or not isinstance(batch, BatchResult)
                            or not 0 <= batch.trajectories <= self.batch_size):
                        self._fail("invalid worker contribution/work receipt", [message])
                    snapshot = self._master.accept(update)
                    progress[index] += batch.trajectories
                    receipts.append({"worker": index, "sequence": update.sequence,
                        "batch": batch, "preparation_seconds": preparation,
                        "evidence": evidence, "master_version": snapshot.version})
                    if index not in first_updates:
                        first_updates.add(index)
                        if queued:
                            # FIFO activation, irrespective of outcomes, rewards,
                            # acceptance or throughput. No worker/draw is retried.
                            dispatch(queued.pop(0), snapshot, released_by=index)
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
            self._last_result = ParallelResult(result, self.worker_pids, tuple(receipts), len(receipts),
                                  self.startup_seconds, decision, tuple(dispatch_schedule))
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
        for connection in self._connections:
            try:
                connection.send(("close",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        # Bounded cleanup of these OWN processes. No process discovery/killing.
        until = time.perf_counter() + 2.
        for index, process in enumerate(self._processes):
            if process.pid is not None:
                process.join(timeout=max(0., until - time.perf_counter()))
                if process.is_alive():
                    if os.name == "posix" and index not in self._owned_groups:
                        try:
                            if os.getpgid(process.pid) == process.pid:
                                self._owned_groups[index] = process.pid
                        except ProcessLookupError:
                            pass
                    process.terminate()
                    process.join(timeout=2.)
        if os.name == "posix":
            remaining = []
            permitted = set(self._owned_groups.values())
            if self._ownership_receipts:
                rows = [json.loads(path.read_text()) for path in self._ownership_receipts.values()]
                identities = owned_process_identities([row["pid"] for row in rows])
                permitted = {row["group"] for row in rows if row["pid"] not in identities
                    or identities[row["pid"]] == " ".join(row["birth_identity"].split())}
            for group in self._owned_groups.values():
                if group not in permitted:
                    continue  # A reused PID is never an owned signalling target.
                try:
                    signal_owned_group(group, signal.SIGTERM)
                    remaining.append(group)
                except ProcessLookupError:
                    pass
            if remaining:
                time.sleep(.2)
                for group in remaining:
                    try:
                        signal_owned_group(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        # A stubborn worker is owned too; don't leave it running after close.
        for process in self._processes:
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join(timeout=2.)
        for connection in self._connections:
            connection.close()
        close_owned_generations(self._ownership_receipts, self._processes)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
