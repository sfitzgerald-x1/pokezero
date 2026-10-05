"""Cumulative OWN contributions for the paper's periodic Q/N/M/F exchange.

Imported counts are never re-exported as a worker's new samples. The master
replaces a worker's prior cumulative contribution, rejecting inconsistent or
regressing updates. Re-delivery is idempotent; stale delivery cannot undo work.
P is intentionally absent. Transport/process scheduling is a separate layer.
"""
from dataclasses import dataclass
import math

from .paper_reference import DecisionState, ReferenceRefusal


@dataclass(frozen=True)
class Statistics:
    state: DecisionState
    visits: tuple[int, ...]
    totals: tuple[float, ...]
    count: int

    def __post_init__(self):
        if (not isinstance(self.state, DecisionState) or type(self.count) is not int
                or not isinstance(self.visits, tuple) or not isinstance(self.totals, tuple)
                or len(self.visits) != len(self.state.actions) or len(self.totals) != len(self.visits)
                or any(type(n) is not int or n < 0 for n in self.visits)
                or sum(self.visits) != self.count
                or any(type(q) not in (int, float) or not math.isfinite(q)
                    or abs(q) > n or (n == 0 and q != 0)
                    for q, n in zip(self.totals, self.visits))):
            raise ReferenceRefusal("invalid signed Q/N/M/F contribution")


def _rows(rows):
    if not isinstance(rows, tuple) or any(not isinstance(r, Statistics) for r in rows):
        raise ReferenceRefusal("statistics exchange requires immutable rows")
    result = {r.state.key: r for r in rows}
    if len(result) != len(rows):
        raise ReferenceRefusal("duplicate information-state key in statistics exchange")
    return result


@dataclass(frozen=True)
class WorkerUpdate:
    battle_id: str
    worker_id: str
    sequence: int
    rows: tuple[Statistics, ...]

    def __post_init__(self):
        if (not isinstance(self.battle_id, str) or not self.battle_id
                or not isinstance(self.worker_id, str) or not self.worker_id
                or type(self.sequence) is not int or self.sequence < 1):
            raise ReferenceRefusal("invalid worker update identity or sequence")
        _rows(self.rows)


@dataclass(frozen=True)
class MasterSnapshot:
    battle_id: str
    version: int
    faint_floor: int
    rows: tuple[Statistics, ...]
    acknowledged: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        _rows(self.rows)
        if (not isinstance(self.battle_id, str) or not self.battle_id
                or type(self.version) is not int or self.version < 0
                or type(self.faint_floor) is not int or not 0 <= self.faint_floor <= 12
                or any(r.state.faint_count < self.faint_floor for r in self.rows)):
            raise ReferenceRefusal("invalid master snapshot identity or faint floor")
        if (not isinstance(self.acknowledged, tuple)
                or any(not isinstance(r, tuple) or len(r) != 2
                    or not isinstance(r[0], str) or not r[0] or type(r[1]) is not int or r[1] < 1
                    for r in self.acknowledged)
                or len(dict(self.acknowledged)) != len(self.acknowledged)):
            raise ReferenceRefusal("invalid worker acknowledgement watermark")


def _nonregressing(before: Statistics, after: Statistics):
    if before.state != after.state:
        raise ReferenceRefusal("worker aliased an information-state identity")
    for old_n, n, old_q, q in zip(before.visits, after.visits, before.totals, after.totals):
        if n < old_n or (n == old_n and q != old_q) or abs(q - old_q) > n - old_n + 1e-9:
            raise ReferenceRefusal("worker contribution regressed or changed already-counted values")


class StatisticsMaster:
    def __init__(self, battle_id: str):
        if not isinstance(battle_id, str) or not battle_id:
            raise ReferenceRefusal("statistics master requires battle identity")
        self.battle_id = battle_id
        self.version = self.faint_floor = 0
        self._workers: dict[str, WorkerUpdate] = {}

    def advance_real_faints(self, faint_floor: int):
        if type(faint_floor) is not int or not self.faint_floor <= faint_floor <= 12:
            raise ReferenceRefusal("master real faint count decreased or is invalid")
        if faint_floor != self.faint_floor:
            self.faint_floor = faint_floor
            self._workers = {key: WorkerUpdate(w.battle_id, w.worker_id, w.sequence,
                tuple(r for r in w.rows if r.state.faint_count >= faint_floor))
                for key, w in self._workers.items()}
            self.version += 1

    def accept(self, update: WorkerUpdate) -> MasterSnapshot:
        if not isinstance(update, WorkerUpdate) or update.battle_id != self.battle_id:
            raise ReferenceRefusal("worker statistics belong to a different battle")
        update = WorkerUpdate(update.battle_id, update.worker_id, update.sequence,
            tuple(r for r in update.rows if r.state.faint_count >= self.faint_floor))
        previous = self._workers.get(update.worker_id)
        if previous is not None:
            if update.sequence == previous.sequence:
                if update != previous:
                    raise ReferenceRefusal("same worker sequence has different contributions")
                return self.snapshot()
            if update.sequence < previous.sequence:
                return self.snapshot()  # Delayed receipt, not new work.
            new = _rows(update.rows)
            for old in previous.rows:
                if old.state.faint_count >= self.faint_floor:
                    if old.state.key not in new:
                        raise ReferenceRefusal("worker discarded unpruned cumulative contributions")
                    _nonregressing(old, new[old.state.key])
        # Validate cross-worker state identity BEFORE committing the update.
        known = {r.state.key: r.state for w in self._workers.values() for r in w.rows
                 if r.state.faint_count >= self.faint_floor}
        for row in update.rows:
            if row.state.faint_count >= self.faint_floor and row.state.key in known:
                if known[row.state.key] != row.state:
                    raise ReferenceRefusal("workers disagree on public identity/legal support/faints")
        candidate = {**self._workers, update.worker_id: update}
        result = self._snapshot(candidate, self.version + 1)
        self._workers = candidate
        self.version += 1
        return result

    def snapshot(self) -> MasterSnapshot:
        return self._snapshot(self._workers, self.version)

    def _snapshot(self, workers: dict[str, WorkerUpdate], version: int) -> MasterSnapshot:
        grouped: dict[bytes, list[Statistics]] = {}
        for worker in workers.values():
            for row in worker.rows:
                if row.state.faint_count >= self.faint_floor:
                    grouped.setdefault(row.state.key, []).append(row)
        rows = tuple(Statistics(group[0].state,
            tuple(sum(r.visits[i] for r in group) for i in range(len(group[0].visits))),
            tuple(math.fsum(r.totals[i] for r in group) for i in range(len(group[0].totals))),
            sum(r.count for r in group)) for _, group in sorted(grouped.items()))
        return MasterSnapshot(self.battle_id, version, self.faint_floor, rows,
            tuple(sorted((key, w.sequence) for key, w in workers.items())))
