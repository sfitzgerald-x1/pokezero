"""Persistent collector-fleet worker.

Claims task manifests from the filesystem work queue and runs each through the
standard ``collect-selfplay-training-cache`` path IN-PROCESS, so interpreter +
torch import startup is paid once per worker process instead of once per task
(measured ~22 s of a ~46 s slice wall). The queue TRANSPORT is unchanged and
byte-compatible with the shell fleet worker:

    pending/i<N>-s<K>.env  --atomic rename-->  claimed/<base>.<worker>
        success + claim still present  -> out committed, claim -> done/<base>
        claim revoked mid-task         -> work discarded (revocation-discard)
        failure                        -> claim -> failed/<base>.<worker>.failed

Shard fan-in mode (``shard_fanin=True``): tasks stay micro (1-2 games) but the
worker owns ONE train shard per window — ``<iter cache dir>/shard-w<worker>`` —
so shard count tracks worker count (<=fleet size), not task count. Each
committed task's cache and a schema-versioned task manifest are assembled in a
new ``-v<k+1>`` version and atomically renamed as one unit. The manifest is the
durable commit record: a retry consults selected highest versions before
collecting, recognizes an already committed task, and only recovers its done
marker. The strict reader exposes selected versions and a deterministic global
inventory, so training can require an exact task, game, offset, and seed
contract. Fan-in is exactly-once for accepted training input, including a
crash after version publication and before the done marker.

Manifest keys (shell-sourceable ``a_key=value`` lines): ``a_iter``,
``a_offset``, ``a_count``, ``a_seed``, ``a_out``, ``a_policy``.

Death/OOM posture (the two operational risks of persistence):
- The pod command wraps this in a small respawn loop, so a crashed or recycled
  worker comes back in ~2 s with a fresh interpreter; the queue's stale-claim
  reaper requeues whatever a dead worker had claimed, and the revocation check
  keeps a resurrected claim from double-committing.
- The worker SELF-RECYCLES (clean exit 0) once resident memory exceeds
  ``max_rss_mb`` or after ``max_tasks`` tasks, bounding leak accumulation over
  long runs — the respawn loop turns that into a fresh process at a task
  boundary, never mid-game. Set ``max_rss_mb`` BELOW the container memory
  limit, or the kernel OOM-killer fires first and the recycle degrades into a
  mid-task kill (wasted slice + stale claim until the reaper requeues it).

Every claim/commit/failure line is timestamped, which doubles as the
collect-queue plan's Step-0 instrumentation (task-duration histogram and the
startup-vs-compute split come straight from these logs).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class TaskManifest:
    base: str
    claim_path: Path
    iteration: int
    offset: int
    count: int
    seed: int
    out: Path
    policy: str


FANIN_MANIFEST_NAME = "fanin-manifest.json"
FANIN_MANIFEST_SCHEMA_VERSION = 1
_FANIN_MANIFEST_KIND = "pokezero-fanin-shard"


class FanInValidationError(ValueError):
    """A fan-in version or its global task inventory is not safe to train from."""


class _ClaimRevokedError(RuntimeError):
    """The controller revoked a claim before it crossed the publication boundary."""


@dataclass(frozen=True)
class FanInTask:
    """One queue task durably included in a fan-in shard version."""

    task_id: str
    iteration: int
    offset: int
    count: int
    seed: int

    @property
    def offset_stop(self) -> int:
        return self.offset + self.count

    @property
    def seed_stop(self) -> int:
        return self.seed + self.count


@dataclass(frozen=True)
class FanInShard:
    """The selected highest version for one worker shard."""

    worker: str
    version: int
    path: Path
    tasks: tuple[FanInTask, ...]


@dataclass(frozen=True)
class FanInQueueContract:
    """The complete queue coverage required before a private launcher trains."""

    iteration: int
    expected_task_count: int
    expected_game_count: int
    offset_start: int = 0
    seed_start: int = 0


@dataclass(frozen=True)
class FanInInventory:
    """Strict selected-version view suitable for handing directly to a trainer."""

    shards: tuple[FanInShard, ...]
    tasks: tuple[FanInTask, ...]
    total_games: int


def _log(worker_id: str, message: str, *, log_handle: Any | None = None) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{stamp} fleet-worker {worker_id}: {message}"
    print(line, flush=True)
    if log_handle is not None:
        try:
            log_handle.write(line + "\n")
            log_handle.flush()
        except OSError:
            pass  # durable logging is best-effort; never fail a task over it


def _parse_manifest(path: Path, base: str) -> TaskManifest:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        fields[key.strip()] = value
    try:
        return TaskManifest(
            base=base,
            claim_path=path,
            iteration=int(fields["a_iter"]),
            offset=int(fields.get("a_offset", "0")),
            count=int(fields["a_count"]),
            seed=int(fields["a_seed"]),
            out=Path(fields["a_out"]),
            policy=fields["a_policy"],
        )
    except KeyError as exc:
        raise ValueError(f"task manifest {base} missing key {exc}") from exc


def claim_next_task(queue: Path, worker_id: str) -> TaskManifest | None:
    """Claim the first available pending manifest via atomic rename (or None)."""
    pending = queue / "pending"
    try:
        candidates = sorted(pending.glob("*.env"))
    except OSError:
        return None
    for candidate in candidates:
        claim = queue / "claimed" / f"{candidate.name}.{worker_id}"
        try:
            os.rename(candidate, claim)
        except OSError:
            continue  # lost the race; try the next manifest
        try:
            return _parse_manifest(claim, candidate.name)
        except ValueError:
            # Malformed manifest: park it in failed/ so the controller's attempt
            # bound decides, rather than looping on it forever.
            failed = queue / "failed" / f"{candidate.name}.{worker_id}.failed"
            try:
                os.rename(claim, failed)
            except OSError:
                pass
            continue
    return None


def _sanitize_worker_id(worker: str) -> str:
    return "".join(ch for ch in worker.lower() if ch.isalnum() or ch == "-") or "w"


def _shard_versions(base: Path) -> list[tuple[int, Path]]:
    """All complete versions of a worker shard, ascending: ``<base>-v<k>``."""
    versions: list[tuple[int, Path]] = []
    prefix = f"{base.name}-v"
    for candidate in base.parent.glob(f"{base.name}-v*"):
        suffix = candidate.name[len(prefix):]
        if suffix.isdigit():
            versions.append((int(suffix), candidate))
    return sorted(versions)


def _adopt_shard(base: Path) -> tuple[Path | None, int]:
    """Return the highest shard version without mutating recovery evidence."""
    versions = _shard_versions(base)
    if not versions:
        return None, 0
    top_version, top_path = versions[-1]
    return top_path, top_version


def select_fanin_shards(cache_dir: Path) -> list[Path]:
    """Highest version per worker shard under an iteration cache dir.

    This compatibility selector deliberately only selects paths. Call
    :func:`read_fanin_inventory` before training; it strictly validates the
    manifests and queue coverage of these selected paths.
    """
    best: dict[str, tuple[int, Path]] = {}
    for candidate in Path(cache_dir).glob("shard-w*-v*"):
        name, _, suffix = candidate.name.rpartition("-v")
        if not suffix.isdigit():
            continue
        version = int(suffix)
        if name not in best or version > best[name][0]:
            best[name] = (version, candidate)
    return [path for _, (_, path) in sorted(best.items())]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fanin_task_from_task_manifest(task: TaskManifest) -> FanInTask:
    candidate = FanInTask(task.base, task.iteration, task.offset, task.count, task.seed)
    if (
        not candidate.task_id
        or candidate.iteration < 0
        or candidate.offset < 0
        or candidate.count <= 0
        or candidate.seed < 0
    ):
        raise FanInValidationError(f"task {task.base!r} has invalid fan-in queue metadata")
    return candidate


def _fanin_task_payload(task: FanInTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "iteration": task.iteration,
        "offset": task.offset,
        "count": task.count,
        "seed": task.seed,
    }


def _read_fanin_manifest(path: Path) -> tuple[FanInTask, ...]:
    manifest_path = path / FANIN_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FanInValidationError(f"fan-in shard {path} has no valid {FANIN_MANIFEST_NAME}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "kind", "tasks"}:
        raise FanInValidationError(f"fan-in shard {path} has a malformed manifest envelope")
    if payload["schema_version"] != FANIN_MANIFEST_SCHEMA_VERSION or payload["kind"] != _FANIN_MANIFEST_KIND:
        raise FanInValidationError(f"fan-in shard {path} has an unsupported manifest schema")
    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise FanInValidationError(f"fan-in shard {path} has an empty or malformed task manifest")
    tasks: list[FanInTask] = []
    seen: set[str] = set()
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict) or set(raw_task) != {"task_id", "iteration", "offset", "count", "seed"}:
            raise FanInValidationError(f"fan-in shard {path} has a malformed task entry")
        task_id = raw_task["task_id"]
        numeric = (raw_task["iteration"], raw_task["offset"], raw_task["count"], raw_task["seed"])
        if not isinstance(task_id, str) or not task_id or not all(_is_int(value) for value in numeric):
            raise FanInValidationError(f"fan-in shard {path} has invalid task metadata")
        iteration, offset, count, seed = numeric
        if iteration < 0 or offset < 0 or count <= 0 or seed < 0:
            raise FanInValidationError(f"fan-in shard {path} has out-of-range task metadata")
        if task_id in seen:
            raise FanInValidationError(f"fan-in shard {path} repeats task id {task_id!r}")
        seen.add(task_id)
        tasks.append(FanInTask(task_id, iteration, offset, count, seed))
    return tuple(tasks)


def _cache_record_count(path: Path) -> int:
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FanInValidationError(f"fan-in shard {path} has no valid cache metadata") from exc
    record_count = metadata.get("record_count") if isinstance(metadata, dict) else None
    if not _is_int(record_count) or record_count < 0:
        raise FanInValidationError(f"fan-in shard {path} has invalid cache record_count")
    return record_count


def _read_selected_fanin_shards(
    cache_dir: Path,
    *,
    expected_iteration: int | None = None,
) -> tuple[tuple[FanInShard, ...], dict[str, FanInTask]]:
    shards: list[FanInShard] = []
    by_task_id: dict[str, FanInTask] = {}
    for path in select_fanin_shards(cache_dir):
        name, _, version_text = path.name.rpartition("-v")
        tasks = _read_fanin_manifest(path)
        if _cache_record_count(path) != sum(task.count for task in tasks):
            raise FanInValidationError(f"fan-in shard {path} cache games disagree with its task manifest")
        if expected_iteration is not None and any(task.iteration != expected_iteration for task in tasks):
            raise FanInValidationError(f"fan-in shard {path} has a task from the wrong iteration")
        for task in tasks:
            previous = by_task_id.get(task.task_id)
            if previous is not None:
                if previous != task:
                    raise FanInValidationError(f"fan-in inventory has conflicting metadata for {task.task_id!r}")
                raise FanInValidationError(f"fan-in inventory repeats task id {task.task_id!r}")
            by_task_id[task.task_id] = task
        shards.append(FanInShard(name.removeprefix("shard-w"), int(version_text), path, tasks))
    return tuple(shards), by_task_id


def _validate_fanin_contract(tasks: Sequence[FanInTask], contract: FanInQueueContract) -> None:
    if (
        not _is_int(contract.iteration)
        or not _is_int(contract.expected_task_count)
        or not _is_int(contract.expected_game_count)
        or not _is_int(contract.offset_start)
        or not _is_int(contract.seed_start)
        or contract.iteration < 0
        or contract.expected_task_count <= 0
        or contract.expected_game_count <= 0
        or contract.offset_start < 0
        or contract.seed_start < 0
    ):
        raise FanInValidationError("fan-in queue contract has invalid bounds")
    if len(tasks) != contract.expected_task_count:
        raise FanInValidationError(
            f"fan-in inventory has {len(tasks)} tasks; expected {contract.expected_task_count}"
        )
    total_games = sum(task.count for task in tasks)
    if total_games != contract.expected_game_count:
        raise FanInValidationError(
            f"fan-in inventory has {total_games} games; expected {contract.expected_game_count}"
        )
    offset = contract.offset_start
    for task in sorted(tasks, key=lambda item: (item.offset, item.task_id)):
        if task.offset != offset:
            raise FanInValidationError("fan-in inventory has a gap or overlap in queue offsets")
        offset = task.offset_stop
    seed = contract.seed_start
    for task in sorted(tasks, key=lambda item: (item.seed, item.task_id)):
        if task.seed != seed:
            raise FanInValidationError("fan-in inventory has a gap or overlap in queue seed ranges")
        seed = task.seed_stop
    for task in tasks:
        if task.seed - contract.seed_start != task.offset - contract.offset_start:
            raise FanInValidationError("fan-in task seed and offset ranges do not describe the same queue position")


def read_fanin_inventory(cache_dir: Path, contract: FanInQueueContract) -> FanInInventory:
    """Return strict selected shards and their deterministic, complete task inventory.

    ``contract`` lets a launcher require, for example, 800 distinct tasks and
    1,600 games before it trains. Selected legacy shards without
    ``fanin-manifest.json`` are intentionally rejected.
    """
    shards, by_task_id = _read_selected_fanin_shards(cache_dir, expected_iteration=contract.iteration)
    tasks = tuple(sorted(by_task_id.values(), key=lambda item: (item.offset, item.seed, item.task_id)))
    _validate_fanin_contract(tasks, contract)
    return FanInInventory(shards=shards, tasks=tasks, total_games=sum(task.count for task in tasks))


def _write_fanin_manifest(path: Path, tasks: Sequence[FanInTask]) -> None:
    payload = {
        "schema_version": FANIN_MANIFEST_SCHEMA_VERSION,
        "kind": _FANIN_MANIFEST_KIND,
        "tasks": [_fanin_task_payload(task) for task in tasks],
    }
    manifest_path = path / FANIN_MANIFEST_NAME
    if manifest_path.exists():
        raise FanInValidationError(f"refusing to overwrite existing fan-in manifest in {path}")
    temporary_path = path / f".{FANIN_MANIFEST_NAME}.tmp.{os.getpid()}"
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
        _fsync_directory(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Persist a manifest replacement or version rename before acknowledging it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _complete_claim(task: TaskManifest, queue: Path) -> None:
    """Publish a done marker without overwriting another worker's marker."""
    done = queue / "done" / task.base
    try:
        os.link(task.claim_path, done)
    except FileExistsError as exc:
        raise FanInValidationError(f"done marker already exists for {task.base}") from exc
    except OSError as exc:
        raise _ClaimRevokedError(f"claim was revoked before done marker for {task.base}") from exc
    try:
        task.claim_path.unlink()
    except OSError:
        # The durable done marker wins; a later stale-claim cleanup is harmless.
        pass


def _fail_claim(task: TaskManifest, queue: Path, worker: str) -> None:
    try:
        os.rename(task.claim_path, queue / "failed" / f"{task.base}.{worker}.failed")
    except OSError:
        pass


def _inject_crash(
    crash_inject: Callable[[str, TaskManifest], None] | None,
    boundary: str,
    task: TaskManifest,
) -> None:
    if crash_inject is not None:
        crash_inject(boundary, task)


def _recover_fanin_task(
    task: TaskManifest,
    queue: Path,
    *,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> bool:
    """Finalize a claim whose task is already durably selected, without collecting."""
    candidate = _fanin_task_from_task_manifest(task)
    _, committed = _read_selected_fanin_shards(task.out.parent, expected_iteration=task.iteration)
    existing = committed.get(task.base)
    if existing is None:
        return False
    if existing != candidate:
        raise FanInValidationError(f"committed task {task.base!r} conflicts with its retried manifest")
    _complete_claim(task, queue)
    _inject_crash(crash_inject, "fanin-after-recovery", task)
    return True


def _publish_fanin_task(
    task: TaskManifest,
    temporary_cache: Path,
    queue: Path,
    worker: str,
    *,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> tuple[Path | None, bool]:
    """Atomically publish one manifest-bearing version, or recover a concurrent commit."""
    cache_dir = task.out.parent
    fanin_task = _fanin_task_from_task_manifest(task)
    if _cache_record_count(temporary_cache) != fanin_task.count:
        raise FanInValidationError(
            f"task {task.base} cache games do not match queue count {fanin_task.count}"
        )
    _, committed = _read_selected_fanin_shards(cache_dir, expected_iteration=task.iteration)
    existing = committed.get(task.base)
    if existing is not None:
        if existing != fanin_task:
            raise FanInValidationError(f"committed task {task.base!r} conflicts with its retried manifest")
        _complete_claim(task, queue)
        shutil.rmtree(temporary_cache, ignore_errors=True)
        _inject_crash(crash_inject, "fanin-after-recovery", task)
        return None, True
    if not task.claim_path.exists():
        raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")

    base = cache_dir / f"shard-w{_sanitize_worker_id(worker)}"
    current, version = _adopt_shard(base)
    current_tasks: tuple[FanInTask, ...] = ()
    if current is not None:
        current_tasks = _read_fanin_manifest(current)
    target = base.parent / f"{base.name}-v{version + 1}"
    staging = base.parent / f".{target.name}.tmp.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if current is None:
            os.rename(temporary_cache, staging)
        else:
            from .dataset import concat_training_caches

            concat_training_caches((current, temporary_cache), staging)
            shutil.rmtree(temporary_cache, ignore_errors=True)
        version_tasks = (*current_tasks, fanin_task)
        if _cache_record_count(staging) != sum(entry.count for entry in version_tasks):
            raise FanInValidationError(f"fan-in staging cache does not match task manifest for {task.base}")
        _write_fanin_manifest(staging, version_tasks)
        _inject_crash(crash_inject, "fanin-before-target-publication", task)
        if not task.claim_path.exists():
            raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
        os.rename(staging, target)
        _fsync_directory(target.parent)
        _inject_crash(crash_inject, "fanin-after-target-publication", task)
        _complete_claim(task, queue)
        # A selected higher version is cumulative. Delete only after its done
        # marker exists, so a crash always leaves recoverable evidence.
        for _, stale in _shard_versions(base):
            if stale != target:
                shutil.rmtree(stale, ignore_errors=True)
        return target, False
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource

        # ru_maxrss: KiB on Linux, bytes on macOS. High-water, not current — an
        # acceptable, conservative fallback for the recycle bound.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024.0 if os.uname().sysname == "Linux" else peak / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def run_worker(
    queue: Path,
    *,
    worker_id: str | None = None,
    static_argv: Sequence[str],
    collect_fn: Callable[[list[str]], int],
    max_rss_mb: float | None = 3300.0,
    max_tasks: int | None = None,
    idle_exit_seconds: float | None = None,
    sleep_seconds: float = 2.0,
    shard_fanin: bool = False,
    log_dir: Path | None = None,
    crash_inject: Callable[[str, TaskManifest], None] | None = None,
) -> int:
    """Drain the queue forever (daemon) or until a recycle/idle bound trips.

    ``collect_fn`` receives the full per-task argv (static flags first, then the
    per-task ``--games/--seed-start/--out/--current-policy`` overrides — last
    wins under argparse, so per-task values always take precedence) and returns
    a process-style exit code.
    """
    worker = worker_id or socket.gethostname()
    log_handle = None
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_dir / f"{_sanitize_worker_id(worker)}.log", "a", encoding="utf-8")
        except OSError:
            log_handle = None  # never block collection on the durable log

    def log(message: str) -> None:
        _log(worker, message, log_handle=log_handle)

    log(
        f"persistent worker up; queue={queue} rss_limit_mb={max_rss_mb} "
        f"max_tasks={max_tasks} fanin={int(shard_fanin)}"
    )
    tasks_done = 0
    idle_since: float | None = None

    def recycle_due() -> bool:
        rss = _rss_mb()
        if max_rss_mb is not None and rss > max_rss_mb:
            log(f"rss {rss:.0f}MB over {max_rss_mb:.0f}MB; recycling after {tasks_done} tasks")
            return True
        if max_tasks is not None and tasks_done >= max_tasks:
            log(f"max_tasks {max_tasks} reached; recycling")
            return True
        return False

    while True:
        task = claim_next_task(queue, worker)
        if task is None:
            now = time.monotonic()
            if idle_since is None:
                idle_since = now
            if idle_exit_seconds is not None and now - idle_since >= idle_exit_seconds:
                log(f"idle for {idle_exit_seconds:.0f}s; exiting after {tasks_done} tasks")
                return 0
            time.sleep(sleep_seconds)
            continue
        idle_since = None
        if shard_fanin:
            try:
                if _recover_fanin_task(task, queue, crash_inject=crash_inject):
                    log(f"recover-fanin {task.base}; durable version already selected")
                    tasks_done += 1
                    if recycle_due():
                        return 0
                    continue
            except _ClaimRevokedError:
                log(f"revoked {task.base}; no fan-in recovery marker emitted")
                tasks_done += 1
                if recycle_due():
                    return 0
                continue
            except FanInValidationError as exc:
                log(f"fan-in recovery rejected {task.base}: {exc}")
                _fail_claim(task, queue, worker)
                tasks_done += 1
                if recycle_due():
                    return 0
                continue
        if not task.claim_path.exists():
            log(f"revoked {task.base}; discarding before collection")
            tasks_done += 1
            if recycle_due():
                return 0
            continue
        tmp = Path(f"{task.out}.tmp.{worker}")
        shutil.rmtree(tmp, ignore_errors=True)
        task.out.parent.mkdir(parents=True, exist_ok=True)
        log(f"claim {task.base} iter={task.iteration} games={task.count} seed={task.seed}")
        started = time.monotonic()
        try:
            returncode = collect_fn(
                [
                    *static_argv,
                    "--games", str(task.count),
                    "--seed-start", str(task.seed),
                    "--out", str(tmp),
                    "--current-policy", task.policy,
                ]
            )
            succeeded = returncode == 0
        except Exception:
            log(f"task {task.base} raised:\n{traceback.format_exc()}")
            succeeded = False
        elapsed = time.monotonic() - started
        if succeeded and shard_fanin:
            concat_started = time.monotonic()
            try:
                target, recovered = _publish_fanin_task(
                    task, tmp, queue, worker, crash_inject=crash_inject,
                )
            except _ClaimRevokedError:
                log(f"revoked {task.base}; discarding {elapsed:.1f}s of work")
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                log(f"fan-in commit for {task.base} raised:\n{traceback.format_exc()}")
                shutil.rmtree(tmp, ignore_errors=True)
                _fail_claim(task, queue, worker)
            else:
                concat_elapsed = time.monotonic() - concat_started
                if recovered:
                    log(f"recover-fanin {task.base}; concurrent durable version already selected")
                else:
                    assert target is not None
                    # Wall attribution: collect= is game compute for this task,
                    # concat= is fan-in's added critical-path cost (must stay small).
                    log(
                        f"commit-fanin {task.base} -> {target.name} games={task.count} "
                        f"collect={elapsed:.1f}s concat={concat_elapsed:.2f}s rss={_rss_mb():.0f}MB",
                    )
        elif succeeded:
            if task.claim_path.exists():  # revocation-discard, exactly as the shell worker
                shutil.rmtree(task.out, ignore_errors=True)
                os.rename(tmp, task.out)
                os.rename(task.claim_path, queue / "done" / task.base)
                log(f"commit {task.base} games={task.count} elapsed={elapsed:.1f}s rss={_rss_mb():.0f}MB")
            else:
                log(f"revoked {task.base}; discarding {elapsed:.1f}s of work")
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            log(f"FAILED {task.base} elapsed={elapsed:.1f}s")
            shutil.rmtree(tmp, ignore_errors=True)
            _fail_claim(task, queue, worker)
        tasks_done += 1
        if recycle_due():
            return 0
