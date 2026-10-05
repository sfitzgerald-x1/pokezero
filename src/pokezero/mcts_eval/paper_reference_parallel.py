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
import pickle
import signal
import time
import traceback
from typing import Any, Callable, Protocol

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


def _worker_seed(seed: int, index: int) -> int:
    digest = hashlib.sha256(b"paper.reference.worker.v1\0" + seed.to_bytes(8, "big")
                            + index.to_bytes(4, "big")).digest()
    return int.from_bytes(digest[:8], "big")


def _worker(connection, index, runtime_factory, config):
    runtime, prepared = None, None
    phase, decision = "startup", None
    started = time.perf_counter()
    try:
        # Own an isolated group, including this worker's simulator children.
        # A stopped/hung bridge cannot execute EOF cleanup after Python exits.
        if os.name == "posix":
            os.setsid()
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
                 *, workers: int = 20, batch_size: int = 10, transport_timeout: float = 120.):
        if (type(workers) is not int or not 1 <= workers <= 20
                or type(batch_size) is not int or batch_size != 10
                or type(transport_timeout) not in (int, float)
                or not math.isfinite(transport_timeout) or transport_timeout <= 0):
            raise ReferenceRefusal("invalid parallel worker/exchange/timeout contract")
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
                    snapshot = self._master.accept(update)
                    progress[index] += batch.trajectories
                    receipts.append({"worker": index, "sequence": update.sequence,
                        "batch": batch, "preparation_seconds": preparation,
                        "evidence": evidence, "master_version": snapshot.version})
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
            for group in self._owned_groups.values():
                try:
                    os.killpg(group, signal.SIGTERM)
                    remaining.append(group)
                except ProcessLookupError:
                    pass
            if remaining:
                time.sleep(.2)
                for group in remaining:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        # A stubborn worker is owned too; don't leave it running after close.
        for process in self._processes:
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join(timeout=2.)
        for connection in self._connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
