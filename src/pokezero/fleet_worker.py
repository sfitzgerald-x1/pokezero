"""Persistent collector-fleet worker.

Claims task manifests from the filesystem work queue and runs each through the
standard ``collect-selfplay-training-cache`` path IN-PROCESS, so interpreter +
torch import startup is paid once per worker process instead of once per task
(measured ~22 s of a ~46 s slice wall). The queue TRANSPORT is unchanged and
byte-compatible with the shell fleet worker:

    pending/i<N>-s<K>.env  --atomic rename-->  claimed/<base>.<worker>
        success + claim still present  -> out committed, claim -> done/<base>
        claim revoked before fan-in publish -> work discarded (revocation-discard)
        failure                        -> claim -> failed/<base>.<worker>.failed

Shard fan-in mode (``shard_fanin=True``): tasks stay micro (1-2 games) but one
worker-id shard per window — ``<iter cache dir>/shard-w<worker>`` — keeps shard
count near fleet size rather than task count. Publishers sharing a worker id
serialize on a durable lock; unique worker ids remain the normal deployment
configuration to avoid unnecessary contention. Each committed task's cache and
a schema-versioned task manifest are assembled in a new ``-v<k+1>`` version and
atomically renamed as one unit. The manifest is the persistent commit record
across process/pod crashes: a retry consults selected highest versions before
collecting, recognizes an already committed task, and only recovers its done
marker. This is not a power-loss durability claim; cache payload files are not
individually fsynced. The strict reader exposes selected versions and a
deterministic global inventory, so training can require an exact task, game,
offset, and seed contract. Fan-in is exactly-once for accepted training input,
including a process/pod crash after version publication and before the done
marker.

Strict fan-in rollout precondition: start from a fresh cache directory, or a
fully drained directory whose selected versions all have fan-in manifests.
Legacy manifest-less selected versions are not safe to extend and make the
worker fail nonzero with its current claim intact; the final strict validator
rejects them as well. Strict inventory is a private launcher/trainer handoff
boundary and requires queue quiescence: workers use a task-specific lookup
instead, so an unrelated worker publishing does not make every live task retry.
The seed contract validates manifest declarations; it does not inspect
``seeds.npy``.

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

Fan-in failure classification is intentionally narrow: invalid current-task
metadata and a short collected cache move only that claim to ``failed/``;
vanishing selected versions are transient and trigger a clean recycle with the
claim retained; selected legacy/malformed/duplicate inventory and accepted
input versus queue-acknowledgement conflicts are terminal (nonzero) with the
claim retained. Power-loss durability beyond the explicit fsyncs below remains
out of scope.
"""

from __future__ import annotations

import json
import os
import shlex
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
_FANIN_STAGING_OWNER_SUFFIX = ".owner.json"
_FANIN_PUBLISH_LOCK_SUFFIX = ".publish-lock.json"
_SELECTED_FANIN_READ_ATTEMPTS = 3
_SELECTED_FANIN_RETRY_SECONDS = 0.01
_FANIN_PUBLISH_LOCK_ATTEMPTS = 3


class FanInValidationError(ValueError):
    """A fan-in version or its global task inventory is not safe to train from."""


class FanInTaskValidationError(FanInValidationError):
    """The current queue task or its collected cache is invalid and can fail alone."""


class FanInInventoryValidationError(FanInValidationError):
    """A selected published shard is permanently unsafe; preserve the claim and stop."""


class FanInTaskConflictError(FanInTaskValidationError):
    """The claimed task contradicts metadata already committed for that task ID."""


class _SelectedFanInVersionVanishedError(RuntimeError):
    """A selected immutable version vanished while its files were being read."""


class _FanInTransientError(RuntimeError):
    """Selected versions changed during a task-local lookup; recycle and retry later."""


class _ClaimRevokedError(RuntimeError):
    """The controller revoked a claim before it crossed fan-in's commit boundary."""


@dataclass(frozen=True)
class FanInTask:
    """Queue-content identity durably included in a fan-in shard version.

    Output and policy routes are intentionally not part of the cumulative cache
    identity; canonical done markers validate those producer routes separately.
    """

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
    """The complete queue coverage required before a private launcher trains.

    Each game consumes exactly one seed: ``count`` is the width of both a
    task's offset range and seed range, and those ranges must start at the same
    relative position within the contract.
    """

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
        try:
            values = shlex.split(value.strip(), comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(f"task manifest {base} has invalid shell value for {key.strip()!r}") from exc
        if len(values) != 1:
            raise ValueError(f"task manifest {base} has non-scalar value for {key.strip()!r}")
        fields[key.strip()] = values[0]
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
        if suffix.isascii() and suffix.isdecimal():
            versions.append((int(suffix), candidate))
    return sorted(versions)


def _fanin_staging_owner_path(staging: Path) -> Path:
    return staging.parent / f"{staging.name}{_FANIN_STAGING_OWNER_SUFFIX}"


def _write_fanin_staging_owner(staging: Path, task: FanInTask) -> Path:
    """Durably bind an exact staging path to its task before materialization.

    The sidecar is adjacent to ``staging`` rather than inside it, so a pod kill
    while ``concat_training_caches`` is still building the directory cannot
    create an ownerless leak. It remains until accepted publication has also
    been acknowledged, or a later worker proves the claim is no longer live.
    """
    owner = _fanin_staging_owner_path(staging)
    temporary = owner.parent / f".{owner.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    payload = {
        "schema_version": 2,
        "staging": staging.name,
        "task": _fanin_task_payload(task),
    }
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, owner)
        _fsync_directory(owner.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return owner


def _read_fanin_staging_owner(staging: Path) -> FanInTask | None:
    """Return a sidecar owner only when it names this exact staging path."""
    try:
        payload = json.loads(_fanin_staging_owner_path(staging).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "staging", "task"}:
        return None
    if payload["schema_version"] != 2 or payload["staging"] != staging.name or not isinstance(payload["task"], dict):
        return None
    task = payload["task"]
    if set(task) != {"task_id", "iteration", "offset", "count", "seed"}:
        return None
    values = (task["iteration"], task["offset"], task["count"], task["seed"])
    if not isinstance(task["task_id"], str) or not task["task_id"] or not all(_is_int(value) for value in values):
        return None
    candidate = FanInTask(task["task_id"], *values)
    if candidate.iteration < 0 or candidate.offset < 0 or candidate.count <= 0 or candidate.seed < 0:
        return None
    return candidate


def _has_live_claim(queue: Path, task_id: str) -> bool:
    claimed = queue / "claimed"
    try:
        return any(path.name.startswith(f"{task_id}.") for path in claimed.iterdir())
    except OSError:
        # An unreadable queue is not proof that a staging owner is dead.
        return True


def _sweep_abandoned_fanin_staging(cache_dir: Path, queue: Path) -> None:
    """Reclaim only staging whose durable owner no longer has a live claim.

    Staging with no valid owner record is deliberately retained: a different
    worker must never guess that an in-progress producer is dead.
    """
    for sidecar in Path(cache_dir).glob(f".shard-w*-v*.tmp.*{_FANIN_STAGING_OWNER_SUFFIX}"):
        staging_name = sidecar.name.removesuffix(_FANIN_STAGING_OWNER_SUFFIX)
        candidate = sidecar.with_name(staging_name)
        owner = _read_fanin_staging_owner(candidate)
        if owner is None or _has_live_claim(queue, owner.task_id):
            continue
        # The missing claim proves this is not a live producer. A new retry
        # creates a distinct staging path before doing any materialization.
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        sidecar.unlink(missing_ok=True)


def _fanin_publish_lock_path(base: Path) -> Path:
    return base.parent / f".{base.name}{_FANIN_PUBLISH_LOCK_SUFFIX}"


def _read_fanin_publish_lock(base: Path) -> FanInTask | None:
    """Return the task owning a well-formed per-worker publish lock."""
    lock = _fanin_publish_lock_path(base)
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "base", "task"}:
        return None
    if payload["schema_version"] != 1 or payload["base"] != base.name or not isinstance(payload["task"], dict):
        return None
    task = payload["task"]
    if set(task) != {"task_id", "iteration", "offset", "count", "seed"}:
        return None
    numeric = (task["iteration"], task["offset"], task["count"], task["seed"])
    if not isinstance(task["task_id"], str) or not task["task_id"] or not all(_is_int(value) for value in numeric):
        return None
    candidate = FanInTask(task["task_id"], *numeric)
    if candidate.iteration < 0 or candidate.offset < 0 or candidate.count <= 0 or candidate.seed < 0:
        return None
    return candidate


def _acquire_fanin_publish_lock(base: Path, queue: Path, task: FanInTask) -> tuple[Path, tuple[int, int]]:
    """Serialize one worker shard and recover only a proven-dead prior owner."""
    lock = _fanin_publish_lock_path(base)
    payload = {"schema_version": 1, "base": base.name, "task": _fanin_task_payload(task)}
    for attempt in range(_FANIN_PUBLISH_LOCK_ATTEMPTS):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            owner = _read_fanin_publish_lock(base)
            if owner is None:
                raise FanInInventoryValidationError(f"fan-in publish lock is malformed: {lock}")
            if _has_live_claim(queue, owner.task_id):
                raise _FanInTransientError(
                    f"fan-in worker shard {base.name} is actively publishing {owner.task_id!r}"
                )
            # Only a missing owner claim proves this lock is abandoned. A new
            # retry will receive a distinct claim/staging path before publish.
            lock.unlink(missing_ok=True)
            _fsync_directory(lock.parent)
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(lock.parent)
            stat = lock.stat()
            return lock, (stat.st_dev, stat.st_ino)
        except Exception:
            lock.unlink(missing_ok=True)
            raise
    raise _FanInTransientError(f"fan-in publish lock for {base.name} did not stabilize after retries")


def _release_fanin_publish_lock(lock: Path, identity: tuple[int, int]) -> None:
    """Release only the exact lock inode this publisher acquired."""
    try:
        stat = lock.stat()
    except FileNotFoundError:
        return
    if (stat.st_dev, stat.st_ino) != identity:
        return
    lock.unlink(missing_ok=True)
    _fsync_directory(lock.parent)


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
        if not suffix.isascii() or not suffix.isdecimal():
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
        raise FanInTaskValidationError(f"task {task.base!r} has invalid fan-in queue metadata")
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
    except FileNotFoundError as exc:
        if not path.exists():
            raise _SelectedFanInVersionVanishedError(f"selected fan-in shard vanished: {path}") from exc
        raise FanInValidationError(f"fan-in shard {path} has no valid {FANIN_MANIFEST_NAME}") from exc
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
    except FileNotFoundError as exc:
        if not path.exists():
            raise _SelectedFanInVersionVanishedError(f"selected fan-in shard vanished: {path}") from exc
        raise FanInValidationError(f"fan-in shard {path} has no valid cache metadata") from exc
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
    vanished: _SelectedFanInVersionVanishedError | None = None
    for _attempt in range(_SELECTED_FANIN_READ_ATTEMPTS):
        shards: list[FanInShard] = []
        by_task_id: dict[str, FanInTask] = {}
        selected_paths = select_fanin_shards(cache_dir)
        try:
            for path in selected_paths:
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
                            raise FanInValidationError(
                                f"fan-in inventory has conflicting metadata for {task.task_id!r}"
                            )
                        raise FanInValidationError(f"fan-in inventory repeats task id {task.task_id!r}")
                    by_task_id[task.task_id] = task
                shards.append(FanInShard(name.removeprefix("shard-w"), int(version_text), path, tasks))
        except _SelectedFanInVersionVanishedError as exc:
            vanished = exc
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        if select_fanin_shards(cache_dir) != selected_paths:
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        return tuple(shards), by_task_id
    raise FanInValidationError(
        f"selected fan-in versions did not stabilize after {_SELECTED_FANIN_READ_ATTEMPTS} attempts"
    ) from vanished


def _find_committed_fanin_task(cache_dir: Path, candidate: FanInTask) -> FanInTask | None:
    """Find one task across selected versions without demanding global quiescence.

    A worker needs only one answer: whether *its* task was already accepted.
    It still validates every selected version it observes and rejects duplicate
    or conflicting entries for that task, but intentionally does not require a
    second identical all-worker snapshot. The trainer-facing strict reader is
    the place that requires that stronger quiescent invariant.
    """
    vanished: _SelectedFanInVersionVanishedError | None = None
    for _attempt in range(_SELECTED_FANIN_READ_ATTEMPTS):
        matches: list[FanInTask] = []
        try:
            for path in select_fanin_shards(cache_dir):
                tasks = _read_fanin_manifest(path)
                if _cache_record_count(path) != sum(task.count for task in tasks):
                    raise FanInInventoryValidationError(
                        f"fan-in shard {path} cache games disagree with its task manifest"
                    )
                for entry in tasks:
                    if entry.task_id == candidate.task_id:
                        matches.append(entry)
        except _SelectedFanInVersionVanishedError as exc:
            vanished = exc
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        except FanInInventoryValidationError:
            raise
        except FanInValidationError as exc:
            raise FanInInventoryValidationError(
                f"selected fan-in inventory is corrupt: {exc}"
            ) from exc
        if not matches:
            return None
        if len(matches) != 1:
            if any(entry != matches[0] for entry in matches[1:]):
                raise FanInInventoryValidationError(
                    f"fan-in inventory has conflicting metadata for {candidate.task_id!r}"
                )
            raise FanInInventoryValidationError(
                f"fan-in inventory repeats task id {candidate.task_id!r}"
            )
        return matches[0]
    raise _FanInTransientError(
        f"selected fan-in version for task {candidate.task_id!r} kept vanishing"
    ) from vanished


def _read_current_worker_shard(base: Path, expected_iteration: int) -> tuple[Path | None, int, tuple[FanInTask, ...]]:
    """Read only this worker's cumulative shard before publishing its next version."""
    current, version = _adopt_shard(base)
    if current is None:
        return None, version, ()
    try:
        tasks = _read_fanin_manifest(current)
        if _cache_record_count(current) != sum(task.count for task in tasks):
            raise FanInInventoryValidationError(
                f"fan-in shard {current} cache games disagree with its task manifest"
            )
        if any(task.iteration != expected_iteration for task in tasks):
            raise FanInInventoryValidationError(f"fan-in shard {current} has a task from the wrong iteration")
    except _SelectedFanInVersionVanishedError as exc:
        raise _FanInTransientError(f"current worker shard vanished: {current}") from exc
    except FanInInventoryValidationError:
        raise
    except FanInValidationError as exc:
        raise FanInInventoryValidationError(
            f"selected worker shard is corrupt: {exc}"
        ) from exc
    return current, version, tasks


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
    ``fanin-manifest.json`` are intentionally rejected. Call only after queue
    quiescence: unlike task-local recovery, this reader deliberately requires
    a stable all-worker selected-version snapshot. It validates declared seed
    ranges in manifests, not the contents of ``seeds.npy``.
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


def _task_manifest_text(task: TaskManifest) -> str:
    """Canonical queue metadata for a done marker that a reaper can recreate."""
    return (
        f"a_iter={task.iteration}\n"
        f"a_offset={task.offset}\n"
        f"a_count={task.count}\n"
        f"a_seed={task.seed}\n"
        f"a_out={shlex.quote(str(task.out))}\n"
        f"a_policy={shlex.quote(task.policy)}\n"
    )


def _done_marker_matches_task(done: Path, task: TaskManifest) -> bool:
    try:
        parsed = _parse_manifest(done, task.base)
    except (OSError, ValueError, FanInValidationError):
        return False
    # Fan-in manifests identify accepted rollout contents by queue metadata;
    # acknowledgements also bind the output and policy routes that produced it.
    return (
        parsed.base == task.base
        and parsed.iteration == task.iteration
        and parsed.offset == task.offset
        and parsed.count == task.count
        and parsed.seed == task.seed
        and parsed.out == task.out
        and parsed.policy == task.policy
    )


def _write_done_marker(task: TaskManifest, done: Path) -> bool:
    """Create a canonical done marker without overwriting a concurrent marker.

    Returns ``True`` if this call created the marker. A completed fan-in shard
    is already accepted input, so this fallback also handles a reaper that
    removed the original claim between version publication and acknowledgement.
    """
    temporary = done.parent / f".{task.base}.done.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_task_manifest_text(task))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, done)
        except FileExistsError:
            if not _done_marker_matches_task(done, task):
                raise FanInTaskConflictError(f"done marker conflicts with accepted task {task.base}")
            return False
        _fsync_directory(done.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _reject_prepublication_done_marker(task: TaskManifest, queue: Path) -> None:
    """Refuse to publish when queue state already claims this task is terminal."""
    done = queue / "done" / task.base
    if not done.exists():
        return
    if _done_marker_matches_task(done, task):
        raise FanInInventoryValidationError(
            f"done marker already exists without accepted fan-in input for {task.base}"
        )
    raise FanInInventoryValidationError(f"done marker conflicts with current task {task.base}")


def _complete_claim(
    task: TaskManifest,
    queue: Path,
    *,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> None:
    """Publish a canonical done marker after an accepted fan-in task.

    Publication is fan-in's input commit point. Therefore revocation after the
    version rename cannot discard output: if the old claim vanished, recreate a
    matching done marker from the parsed task metadata instead.
    """
    done = queue / "done" / task.base
    created = False
    try:
        os.link(task.claim_path, done)
    except FileExistsError:
        if not _done_marker_matches_task(done, task):
            raise FanInTaskConflictError(f"done marker conflicts with accepted task {task.base}")
    except FileNotFoundError:
        created = _write_done_marker(task, done)
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"could not acknowledge accepted fan-in task {task.base}: {exc}"
        ) from exc
    else:
        created = True
        _fsync_directory(done.parent)
    if created:
        _inject_crash(crash_inject, "fanin-after-done-link", task)
    try:
        task.claim_path.unlink()
    except OSError:
        # The durable done marker wins; a stale-claim cleanup is harmless.
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
    existing = _find_committed_fanin_task(task.out.parent, candidate)
    if existing is None:
        return False
    if existing != candidate:
        raise FanInTaskConflictError(f"committed task {task.base!r} conflicts with its retried manifest")
    try:
        _complete_claim(task, queue, crash_inject=crash_inject)
    except FanInTaskConflictError as exc:
        raise FanInInventoryValidationError(
            f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
        ) from exc
    _inject_crash(crash_inject, "fanin-after-recovery", task)
    return True


def _temporary_cache_record_count(temporary_cache: Path, task: TaskManifest) -> int:
    """Validate the current collection output without poisoning selected inventory."""
    try:
        return _cache_record_count(temporary_cache)
    except (FanInValidationError, OSError) as exc:
        raise FanInTaskValidationError(
            f"task {task.base} collected cache has no valid metadata: {exc}"
        ) from exc


def _recover_current_worker_task(
    current_tasks: Sequence[FanInTask],
    candidate: FanInTask,
) -> bool:
    """Detect an accepted task before appending it to a cumulative manifest."""
    for existing in current_tasks:
        if existing.task_id != candidate.task_id:
            continue
        if existing != candidate:
            raise FanInInventoryValidationError(
                f"current worker shard has conflicting metadata for {candidate.task_id!r}"
            )
        return True
    return False


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
    if _temporary_cache_record_count(temporary_cache, task) != fanin_task.count:
        raise FanInTaskValidationError(
            f"task {task.base} cache games do not match queue count {fanin_task.count}"
        )
    existing = _find_committed_fanin_task(cache_dir, fanin_task)
    if existing is not None:
        if existing != fanin_task:
            raise FanInTaskConflictError(f"committed task {task.base!r} conflicts with its retried manifest")
        try:
            _complete_claim(task, queue, crash_inject=crash_inject)
        except FanInTaskConflictError as exc:
            raise FanInInventoryValidationError(
                f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
            ) from exc
        shutil.rmtree(temporary_cache, ignore_errors=True)
        _inject_crash(crash_inject, "fanin-after-recovery", task)
        return None, True
    if not task.claim_path.exists():
        raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
    _reject_prepublication_done_marker(task, queue)

    base = cache_dir / f"shard-w{_sanitize_worker_id(worker)}"
    lock, lock_identity = _acquire_fanin_publish_lock(base, queue, fanin_task)
    staging: Path | None = None
    owner_sidecar: Path | None = None
    published = False
    try:
        # Re-check under the shard lock so two processes with the same worker
        # id cannot append the same task to independent stale snapshots.
        existing = _find_committed_fanin_task(cache_dir, fanin_task)
        if existing is not None:
            if existing != fanin_task:
                raise FanInInventoryValidationError(
                    f"committed task {task.base!r} conflicts with its retried manifest"
                )
            _complete_claim(task, queue, crash_inject=crash_inject)
            shutil.rmtree(temporary_cache, ignore_errors=True)
            _inject_crash(crash_inject, "fanin-after-recovery", task)
            return None, True
        _reject_prepublication_done_marker(task, queue)
        current, version, current_tasks = _read_current_worker_shard(base, task.iteration)
        if _recover_current_worker_task(current_tasks, fanin_task):
            _complete_claim(task, queue, crash_inject=crash_inject)
            shutil.rmtree(temporary_cache, ignore_errors=True)
            _inject_crash(crash_inject, "fanin-after-recovery", task)
            return None, True
        target = base.parent / f"{base.name}-v{version + 1}"
        staging = base.parent / f".{target.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
        # This sidecar is written before either rename or concat can create
        # staging, closing the kill-during-concat ownerless-leak window.
        owner_sidecar = _write_fanin_staging_owner(staging, fanin_task)
        if current is None:
            os.rename(temporary_cache, staging)
        else:
            from .dataset import concat_training_caches

            concat_training_caches((current, temporary_cache), staging)
            shutil.rmtree(temporary_cache, ignore_errors=True)
        version_tasks = (*current_tasks, fanin_task)
        try:
            staging_record_count = _cache_record_count(staging)
        except (FanInValidationError, OSError) as exc:
            raise FanInTaskValidationError(
                f"task {task.base} fan-in staging cache has no valid metadata: {exc}"
            ) from exc
        if staging_record_count != sum(entry.count for entry in version_tasks):
            raise FanInTaskValidationError(f"fan-in staging cache does not match task manifest for {task.base}")
        _write_fanin_manifest(staging, version_tasks)
        _inject_crash(crash_inject, "fanin-before-target-publication", task)
        if not task.claim_path.exists():
            raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
        try:
            os.rename(staging, target)
        except FileExistsError as exc:
            # A legacy or pre-lock publisher may have won this version. It is
            # safe only if the selected inventory already contains this task.
            existing = _find_committed_fanin_task(cache_dir, fanin_task)
            if existing == fanin_task:
                _complete_claim(task, queue, crash_inject=crash_inject)
                shutil.rmtree(staging, ignore_errors=True)
                owner_sidecar.unlink(missing_ok=True)
                shutil.rmtree(temporary_cache, ignore_errors=True)
                return None, True
            if existing is not None:
                raise FanInInventoryValidationError(
                    f"committed task {task.base!r} conflicts with its retried manifest"
                ) from exc
            raise _FanInTransientError(
                f"fan-in worker shard target raced before publishing {task.base}"
            ) from exc
        published = True
        _fsync_directory(target.parent)
        _inject_crash(crash_inject, "fanin-after-target-publication", task)
        try:
            _complete_claim(task, queue, crash_inject=crash_inject)
        except FanInTaskConflictError as exc:
            raise FanInInventoryValidationError(
                f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
            ) from exc
        # A selected higher version is cumulative. Never delete a concurrently
        # published higher version; only lower stale snapshots are safe.
        _inject_crash(crash_inject, "fanin-before-stale-cleanup", task)
        for stale_version, stale in _shard_versions(base):
            if stale_version < version + 1:
                shutil.rmtree(stale, ignore_errors=True)
        owner_sidecar.unlink(missing_ok=True)
        _fsync_directory(owner_sidecar.parent)
        return target, False
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if owner_sidecar is not None and not published:
            owner_sidecar.unlink(missing_ok=True)
        raise
    finally:
        _release_fanin_publish_lock(lock, lock_identity)


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

    def finish(code: int = 0) -> int:
        if log_handle is not None:
            log_handle.close()
        return code

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
                return finish()
            time.sleep(sleep_seconds)
            continue
        idle_since = None
        if shard_fanin:
            _sweep_abandoned_fanin_staging(task.out.parent, queue)
            try:
                if _recover_fanin_task(task, queue, crash_inject=crash_inject):
                    log(f"recover-fanin {task.base}; durable version already selected")
                    tasks_done += 1
                    if recycle_due():
                        return finish()
                    continue
            except FanInTaskValidationError as exc:
                log(f"fan-in recovery rejected only {task.base}: {exc}")
                _fail_claim(task, queue, worker)
                tasks_done += 1
                if recycle_due():
                    return finish()
                continue
            except _FanInTransientError as exc:
                log(f"fan-in task lookup transient for {task.base}; preserving claim and recycling: {exc}")
                return finish()
            except (FanInInventoryValidationError, OSError) as exc:
                log(f"TERMINAL fan-in selected inventory corruption for {task.base}; preserving claim: {exc}")
                return finish(2)
            try:
                # A done marker without selected input is a queue/inventory
                # contradiction, not a bad collection result. Detect it before
                # wasting a collection slot and retain the claim for repair.
                _reject_prepublication_done_marker(task, queue)
            except FanInInventoryValidationError as exc:
                log(f"TERMINAL fan-in queue acknowledgement disagreement for {task.base}; preserving claim: {exc}")
                return finish(2)
        if not task.claim_path.exists():
            log(f"revoked {task.base}; discarding before collection")
            tasks_done += 1
            if recycle_due():
                return finish()
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
            except FanInTaskValidationError:
                log(f"fan-in commit rejected only {task.base}:\n{traceback.format_exc()}")
                shutil.rmtree(tmp, ignore_errors=True)
                _fail_claim(task, queue, worker)
            except _FanInTransientError as exc:
                log(f"fan-in task lookup transient for {task.base}; preserving claim and recycling: {exc}")
                shutil.rmtree(tmp, ignore_errors=True)
                return finish()
            except (FanInInventoryValidationError, OSError) as exc:
                log(f"TERMINAL fan-in selected inventory corruption for {task.base}; preserving claim: {exc}")
                shutil.rmtree(tmp, ignore_errors=True)
                return finish(2)
            except Exception:
                log(
                    f"TERMINAL fan-in commit failure for {task.base}; preserving claim:\n"
                    f"{traceback.format_exc()}"
                )
                shutil.rmtree(tmp, ignore_errors=True)
                return finish(2)
            else:
                concat_elapsed = time.monotonic() - concat_started
                if recovered:
                    log(f"recover-fanin {task.base}; concurrent durable version already selected")
                else:
                    if target is None:
                        raise RuntimeError("fan-in publication returned neither a target nor recovery")
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
        if shard_fanin:
            # A retry may have completed a task whose earlier crashed staging
            # used the same task id. Once this claim is done/failed, no live
            # owner remains and the ownership sweep can reclaim it safely.
            _sweep_abandoned_fanin_staging(task.out.parent, queue)
        tasks_done += 1
        if recycle_due():
            return finish()
