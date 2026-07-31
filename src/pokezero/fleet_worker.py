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
paired with an immutable per-task acceptance outcome across process/pod
crashes. A retry validates target provenance before accepting a target-first
crash or recovering its done marker. This is not a power-loss durability claim;
cache payload files are not individually fsynced. The strict reader exposes
only acceptance-authenticated versions and a deterministic global inventory,
so training can require an exact task, game, offset, and seed contract. Fan-in
is exactly-once for accepted training input, including a process/pod crash
after target publication and before acceptance or the done marker.

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
import hashlib
import errno
import os
import shlex
import shutil
import socket
import threading
import time
import traceback
import uuid
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
    claim_token: str = ""


FANIN_MANIFEST_NAME = "fanin-manifest.json"
FANIN_PUBLICATION_NAME = "fanin-publication.json"
FANIN_MANIFEST_SCHEMA_VERSION = 2
_FANIN_MANIFEST_KIND = "pokezero-fanin-shard"
_FANIN_ACCEPTANCE_NAME = "acceptance.json"
_FANIN_STAGING_OWNER_SUFFIX = ".owner.json"
_FANIN_STAGING_LEASE_SUFFIX = ".producer-lease.json"
_FANIN_PUBLISH_LOCK_SUFFIX = ".publish-lock.json"
_FANIN_PUBLISH_GUARD_SUFFIX = ".publish-guard"
_FANIN_TASK_LOCK_DIRECTORY = ".fanin-task-locks"
_FANIN_ROUTE_DIRECTORY = ".fanin-routes"
_SELECTED_FANIN_READ_ATTEMPTS = 3
_SELECTED_FANIN_RETRY_SECONDS = 0.01
_FANIN_PUBLISH_LOCK_ATTEMPTS = 3
_FANIN_PRODUCER_LEASE_SECONDS = 60.0
_FANIN_HEARTBEAT_INTERVAL_SECONDS = 10.0
_FANIN_GUARD_CHAIN_MAX_GENERATIONS = 4096
_FANIN_GUARD_ACQUIRE_ATTEMPTS = 8


class FanInValidationError(ValueError):
    """A fan-in version or its global task inventory is not safe to train from."""


class FanInTaskValidationError(FanInValidationError):
    """The current queue task or its collected cache is invalid and can fail alone."""


class FanInInventoryValidationError(FanInValidationError):
    """A selected published shard is permanently unsafe; preserve the claim and stop."""


class FanInTaskConflictError(FanInTaskValidationError):
    """The claimed task contradicts metadata already committed for that task ID."""


class FanInRouteConflictError(FanInInventoryValidationError):
    """Accepted input names a different producer route than the current claim."""


class _SelectedFanInVersionVanishedError(RuntimeError):
    """A selected immutable version vanished while its files were being read."""


class _FanInTransientError(RuntimeError):
    """Selected versions changed during a task-local lookup; recycle and retry later."""


class _ClaimRevokedError(RuntimeError):
    """The controller revoked a claim before it crossed fan-in's commit boundary."""


@dataclass(frozen=True)
class FanInTask:
    """Queue identity and producer route durably included in a fan-in shard."""

    task_id: str
    iteration: int
    offset: int
    count: int
    seed: int
    out: str
    policy: str

    @property
    def offset_stop(self) -> int:
        return self.offset + self.count

    @property
    def seed_stop(self) -> int:
        return self.seed + self.count


@dataclass(frozen=True)
class _FanInAcceptance:
    """Immutable proof that one guard generation accepted one target append."""

    task: FanInTask
    guard_root: str
    guard_generation: str
    claim_name: str
    claim_token: str
    lineage: str
    target: str
    version: int
    task_index: int
    prefix_sha256: str
    manifest_sha256: str
    metadata_sha256: str
    content_sha256: str
    record_count: int


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
        # A claim pathname is a generation capability, not a reusable worker
        # slot. Terminal actions can therefore never target a later retry.
        claim = queue / "claimed" / f"{candidate.name}.{worker_id}.{uuid.uuid4().hex}"
        try:
            os.rename(candidate, claim)
        except OSError:
            continue  # lost the race; try the next manifest
        try:
            task = _parse_manifest(claim, candidate.name)
            _sweep_orphaned_claim_tokens(claim.parent, candidate.name)
            token = _write_claim_token(claim)
            return TaskManifest(**{**task.__dict__, "claim_token": token})
        except ValueError:
            # Malformed manifest: park it in failed/ so the controller's attempt
            # bound decides, rather than looping on it forever.
            failed = queue / "failed" / f"{claim.name}.failed"
            try:
                os.rename(claim, failed)
            except OSError:
                pass
            continue
    return None


def _claim_token_path(claim: Path, token: str) -> Path:
    """Return a generation-specific sidecar; claim names are intentionally reusable."""
    return claim.parent / f".{claim.name}.lease.{token}.json"


def _sweep_orphaned_claim_tokens(claimed: Path, task_id: str) -> None:
    for lease in claimed.glob(f".{task_id}.*.lease.*.json"):
        try:
            payload = json.loads(lease.read_text(encoding="utf-8"))
            claim_name = payload["claim"]
            identity = (payload["claim_dev"], payload["claim_ino"], payload["claim_ctime_ns"])
            stat = (claimed / claim_name).stat()
        except FileNotFoundError:
            lease.unlink(missing_ok=True)
            continue
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            # A malformed token is not evidence that a live claimant is dead.
            continue
        if (stat.st_dev, stat.st_ino, stat.st_ctime_ns) != identity:
            lease.unlink(missing_ok=True)


def _write_claim_token(claim: Path) -> str:
    """Record a fresh claimant generation so path reuse cannot revive a publisher."""
    token = uuid.uuid4().hex
    stat = claim.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
    lease = _claim_token_path(claim, token)
    temporary = lease.parent / f".{lease.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    payload = {
        "schema_version": 2,
        "claim": claim.name,
        "claim_dev": identity[0],
        "claim_ino": identity[1],
        "claim_ctime_ns": identity[2],
        "token": token,
    }
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, lease)
        _fsync_directory(lease.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)
    try:
        current = claim.stat()
    except FileNotFoundError as exc:
        raise ValueError(f"claim disappeared before its lease was recorded: {claim}")
    if (current.st_dev, current.st_ino, current.st_ctime_ns) != identity:
        raise ValueError(f"claim was reused before its lease was recorded: {claim}")
    return token


def _claim_token_is_current(task: TaskManifest) -> bool:
    return _claim_token_is_live(task.claim_path, task.claim_token)


def _claim_token_is_live(claim: Path, token: str) -> bool:
    if not token:
        return False
    try:
        stat = claim.stat()
        payload = json.loads(_claim_token_path(claim, token).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload == {
            "schema_version": 2,
            "claim": claim.name,
            "claim_dev": stat.st_dev,
            "claim_ino": stat.st_ino,
            "claim_ctime_ns": stat.st_ctime_ns,
            "token": token,
        }
    )


def _remove_claim_token(task: TaskManifest) -> None:
    """Remove only this claimant generation's sidecar after terminal handling."""
    lease = _claim_token_path(task.claim_path, task.claim_token)
    try:
        payload = json.loads(lease.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("claim") == task.claim_path.name
        and payload.get("token") == task.claim_token
    ):
        lease.unlink(missing_ok=True)


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


def _fanin_staging_lease_path(staging: Path) -> Path:
    return staging.parent / f"{staging.name}{_FANIN_STAGING_LEASE_SUFFIX}"


def _write_fanin_staging_owner(
    staging: Path,
    task: FanInTask,
    *,
    producer_token: str = "",
) -> Path:
    """Durably bind an exact staging path to its task before materialization.

    The sidecar is adjacent to ``staging`` rather than inside it, so a pod kill
    while ``concat_training_caches`` is still building the directory cannot
    create an ownerless leak. It remains until accepted publication has also
    been acknowledged, or a later worker proves the claim is no longer live.
    """
    owner = _fanin_staging_owner_path(staging)
    temporary = owner.parent / f".{owner.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    payload = {
        "schema_version": 4,
        "staging": staging.name,
        "task": _fanin_task_payload(task),
        "producer_token": producer_token,
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


def _read_fanin_staging_owner_record(staging: Path) -> tuple[FanInTask, str] | None:
    """Return sidecar ownership only when it names this exact staging path."""
    try:
        payload = json.loads(_fanin_staging_owner_path(staging).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "staging", "task", "producer_token",
    }:
        return None
    if (
        payload["schema_version"] != 4
        or payload["staging"] != staging.name
        or not isinstance(payload["task"], dict)
        or not isinstance(payload.get("producer_token"), str)
    ):
        return None
    task = payload["task"]
    if set(task) != {"task_id", "iteration", "offset", "count", "seed", "out", "policy"}:
        return None
    values = (task["iteration"], task["offset"], task["count"], task["seed"])
    if (
        not isinstance(task["task_id"], str)
        or not task["task_id"]
        or not all(_is_int(value) for value in values)
        or not isinstance(task["out"], str)
        or not task["out"]
        or not isinstance(task["policy"], str)
        or not task["policy"]
    ):
        return None
    candidate = FanInTask(task["task_id"], *values, task["out"], task["policy"])
    if candidate.iteration < 0 or candidate.offset < 0 or candidate.count <= 0 or candidate.seed < 0:
        return None
    return candidate, payload["producer_token"]


def _read_fanin_staging_owner(staging: Path) -> FanInTask | None:
    record = _read_fanin_staging_owner_record(staging)
    return record[0] if record is not None else None


def _refresh_fanin_staging_lease(staging: Path, producer_token: str) -> None:
    if not producer_token:
        return
    lease = _fanin_staging_lease_path(staging)
    temporary = lease.parent / f".{lease.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    payload = {
        "schema_version": 1,
        "staging": staging.name,
        "producer_token": producer_token,
        "renewed_at": time.time(),
    }
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, lease)
        _fsync_directory(lease.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fanin_staging_lease_is_active(staging: Path, producer_token: str) -> bool:
    if not producer_token:
        return False
    try:
        payload = json.loads(_fanin_staging_lease_path(staging).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "staging", "producer_token", "renewed_at"}:
        return False
    renewed_at = payload["renewed_at"]
    return (
        payload["schema_version"] == 1
        and payload["staging"] == staging.name
        and payload["producer_token"] == producer_token
        and isinstance(renewed_at, (int, float))
        and not isinstance(renewed_at, bool)
        and time.time() - renewed_at <= _FANIN_PRODUCER_LEASE_SECONDS
    )


def _remove_fanin_staging_lease(staging: Path, producer_token: str) -> None:
    if not producer_token:
        return
    if _fanin_staging_lease_is_active(staging, producer_token):
        _fanin_staging_lease_path(staging).unlink(missing_ok=True)


def _has_live_claim(queue: Path, task_id: str) -> bool:
    claimed = queue / "claimed"
    try:
        return any(path.name.startswith(f"{task_id}.") for path in claimed.iterdir())
    except OSError:
        # An unreadable queue is not proof that a staging owner is dead.
        return True


def _sweep_abandoned_fanin_staging(cache_dir: Path, queue: Path) -> None:
    """Reclaim only staging whose producer fence and claim are both dead.

    Staging with no valid owner record is deliberately retained: a different
    worker must never guess that an in-progress producer is dead.
    """
    for sidecar in Path(cache_dir).glob(f".shard-w*-v*.tmp.*{_FANIN_STAGING_OWNER_SUFFIX}"):
        staging_name = sidecar.name.removesuffix(_FANIN_STAGING_OWNER_SUFFIX)
        candidate = sidecar.with_name(staging_name)
        record = _read_fanin_staging_owner_record(candidate)
        if record is None:
            continue
        owner, producer_token = record
        fence = _fanin_task_fence_path(Path(cache_dir), owner.task_id)
        if (
            _fanin_fence_is_active(fence, queue / "claimed")
            or _fanin_staging_lease_is_active(candidate, producer_token)
            or _has_live_claim(queue, owner.task_id)
        ):
            continue
        # Move the sidecar out of the shared name first. Concurrent sweepers
        # then observe a missing sidecar instead of deleting a new producer's.
        tombstone = sidecar.parent / f".{sidecar.name}.gc.{os.getpid()}.{time.monotonic_ns()}"
        try:
            os.rename(sidecar, tombstone)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        _fanin_staging_lease_path(candidate).unlink(missing_ok=True)
        tombstone.unlink(missing_ok=True)
        _fsync_directory(candidate.parent)


def _fanin_publish_lock_path(base: Path) -> Path:
    return base.parent / f".{base.name}{_FANIN_PUBLISH_LOCK_SUFFIX}"


def _fanin_publish_guard_path(base: Path) -> Path:
    return base.parent / f".{base.name}{_FANIN_PUBLISH_GUARD_SUFFIX}"


def _fanin_fence_owner_path(path: Path) -> Path:
    return path / "owner.json"


def _fanin_fence_release_marker(path: Path, claim_token: str) -> Path:
    """Return an immutable marker for one never-reused guard generation."""
    return path.parent / f".{path.name}.released.{claim_token}"


@dataclass(frozen=True)
class _FanInFenceGeneration:
    path: Path
    identity: tuple[int, int]
    record: tuple[FanInTask, str, str, float]
    acceptance: _FanInAcceptance | None = None


@dataclass
class _FanInFilesystemFence:
    """An mkdir-owned fence whose owner record is renewed while work is live.

    The shared cache volume must provide POSIX atomic directory creation and
    rename within one directory. Unlike advisory ``flock``, those operations
    are part of the cross-node filesystem protocol. The local mount capability
    is preflighted below; deployment must test contention from multiple nodes.
    """

    root: Path
    path: Path
    identity: tuple[int, int]
    task: FanInTask
    claim_name: str
    claim_token: str
    stop: threading.Event
    heartbeat: threading.Thread


def _fanin_fence_payload(task: FanInTask, claim_name: str, claim_token: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": _fanin_task_payload(task),
        "claim_name": claim_name,
        "claim_token": claim_token,
        "renewed_at": time.time(),
    }


def _read_fanin_fence(path: Path) -> tuple[FanInTask, str, str, float] | None:
    try:
        payload = json.loads(_fanin_fence_owner_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "task", "claim_name", "claim_token", "renewed_at",
    }:
        return None
    if (
        payload["schema_version"] != 1
        or not isinstance(payload["claim_name"], str)
        or not payload["claim_name"]
        or not isinstance(payload["claim_token"], str)
        or not payload["claim_token"]
        or not isinstance(payload["renewed_at"], (int, float))
        or isinstance(payload["renewed_at"], bool)
    ):
        return None
    raw_task = payload["task"]
    if not isinstance(raw_task, dict) or set(raw_task) != {
        "task_id", "iteration", "offset", "count", "seed", "out", "policy",
    }:
        return None
    numeric = (raw_task["iteration"], raw_task["offset"], raw_task["count"], raw_task["seed"])
    if (
        not isinstance(raw_task["task_id"], str)
        or not raw_task["task_id"]
        or not all(_is_int(value) for value in numeric)
        or not isinstance(raw_task["out"], str)
        or not raw_task["out"]
        or not isinstance(raw_task["policy"], str)
        or not raw_task["policy"]
    ):
        return None
    task = FanInTask(raw_task["task_id"], *numeric, raw_task["out"], raw_task["policy"])
    if task.iteration < 0 or task.offset < 0 or task.count <= 0 or task.seed < 0:
        return None
    return task, payload["claim_name"], payload["claim_token"], float(payload["renewed_at"])


def _write_fanin_fence(path: Path, task: FanInTask, claim_name: str, claim_token: str) -> None:
    owner = _fanin_fence_owner_path(path)
    temporary = path / f".owner.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(_fanin_fence_payload(task, claim_name, claim_token), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, owner)
        _fsync_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fanin_fence_successor_path(root: Path, generation: _FanInFenceGeneration) -> Path:
    """Derive the generation's one successor-or-acceptance outcome pathname."""
    claim_token = generation.record[2]
    digest = hashlib.sha256(
        f"{root.name}\0{generation.path.name}\0{claim_token}".encode("utf-8")
    ).hexdigest()
    return root.parent / f".{root.name}.generation.{digest}"


def _fanin_acceptance_path(path: Path) -> Path:
    return path / _FANIN_ACCEPTANCE_NAME


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.isascii()
        and all(character in "0123456789abcdef" for character in value)
    )


def _fanin_acceptance_payload(acceptance: _FanInAcceptance) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": _fanin_task_payload(acceptance.task),
        "guard_root": acceptance.guard_root,
        "guard_generation": acceptance.guard_generation,
        "claim_name": acceptance.claim_name,
        "claim_token": acceptance.claim_token,
        "lineage": acceptance.lineage,
        "target": acceptance.target,
        "version": acceptance.version,
        "task_index": acceptance.task_index,
        "prefix_sha256": acceptance.prefix_sha256,
        "manifest_sha256": acceptance.manifest_sha256,
        "metadata_sha256": acceptance.metadata_sha256,
        "content_sha256": acceptance.content_sha256,
        "record_count": acceptance.record_count,
    }


def _fanin_acceptance_from_payload(payload: Any) -> _FanInAcceptance | None:
    expected = {
        "schema_version", "task", "guard_root", "guard_generation", "claim_name",
        "claim_token", "lineage", "target", "version", "task_index", "prefix_sha256",
        "manifest_sha256", "metadata_sha256", "content_sha256", "record_count",
    }
    if not isinstance(payload, dict) or set(payload) != expected or payload["schema_version"] != 1:
        return None
    task = _fanin_task_from_payload(payload["task"])
    string_fields = (
        "guard_root", "guard_generation", "claim_name", "claim_token", "lineage", "target",
    )
    if (
        task is None
        or any(not isinstance(payload[field], str) or not payload[field] for field in string_fields)
        or any(Path(payload[field]).name != payload[field] for field in ("guard_root", "guard_generation", "lineage", "target"))
        or not _is_int(payload["version"])
        or payload["version"] <= 0
        or not _is_int(payload["task_index"])
        or payload["task_index"] < 0
        or not _is_int(payload["record_count"])
        or payload["record_count"] < 0
        or any(
            not _is_sha256(payload[field])
            for field in ("prefix_sha256", "manifest_sha256", "metadata_sha256", "content_sha256")
        )
        or payload["target"] != f"{payload['lineage']}-v{payload['version']}"
    ):
        return None
    return _FanInAcceptance(
        task=task,
        guard_root=payload["guard_root"],
        guard_generation=payload["guard_generation"],
        claim_name=payload["claim_name"],
        claim_token=payload["claim_token"],
        lineage=payload["lineage"],
        target=payload["target"],
        version=payload["version"],
        task_index=payload["task_index"],
        prefix_sha256=payload["prefix_sha256"],
        manifest_sha256=payload["manifest_sha256"],
        metadata_sha256=payload["metadata_sha256"],
        content_sha256=payload["content_sha256"],
        record_count=payload["record_count"],
    )


def _read_fanin_acceptance(path: Path) -> _FanInAcceptance | None:
    try:
        payload = json.loads(_fanin_acceptance_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _fanin_acceptance_from_payload(payload)


def _read_fanin_publication(path: Path) -> _FanInAcceptance | None:
    try:
        payload = json.loads((path / FANIN_PUBLICATION_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _fanin_acceptance_from_payload(payload)


def _installed_fanin_generation_paths(root: Path) -> set[Path]:
    """Enumerate only exact immutable outcome names for one guard root."""
    prefix = f".{root.name}.generation."
    installed: set[Path] = set()
    try:
        for entry in root.parent.iterdir():
            if not entry.name.startswith(prefix):
                continue
            digest = entry.name.removeprefix(prefix)
            if _is_sha256(digest):
                installed.add(entry)
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in publication fence directory is unreadable: {root.parent}"
        ) from exc
    return installed


def _reject_unreachable_fanin_generations(root: Path, reachable: set[Path]) -> None:
    unreachable = _installed_fanin_generation_paths(root) - reachable
    if unreachable:
        names = ", ".join(sorted(path.name for path in unreachable))
        raise FanInInventoryValidationError(
            f"fan-in publication fence has unreachable generations: {names}"
        )


def _read_current_fanin_fence(root: Path) -> _FanInFenceGeneration | None:
    """Resolve an immutable successor/acceptance chain and reject orphan state."""
    path = root
    path_was_observed = False
    reachable: set[Path] = set()
    for _depth in range(_FANIN_GUARD_CHAIN_MAX_GENERATIONS):
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            if not path_was_observed:
                _reject_unreachable_fanin_generations(root, reachable)
                return None
            raise FanInInventoryValidationError(f"fan-in publication fence chain is broken: {path}") from exc
        except OSError as exc:
            raise FanInInventoryValidationError(f"fan-in publication fence is unreadable: {path}") from exc
        record = _read_fanin_fence(path)
        if record is None:
            raise FanInInventoryValidationError(f"fan-in publication fence is malformed: {path}")
        generation = _FanInFenceGeneration(path, (stat.st_dev, stat.st_ino), record)
        outcome = _fanin_fence_successor_path(root, generation)
        try:
            outcome.stat()
        except FileNotFoundError:
            _reject_unreachable_fanin_generations(root, reachable)
            return generation
        except OSError as exc:
            raise FanInInventoryValidationError(
                f"fan-in publication fence outcome is unreadable: {outcome}"
            ) from exc
        reachable.add(outcome)
        successor = _read_fanin_fence(outcome)
        acceptance = _read_fanin_acceptance(outcome)
        if (successor is None) == (acceptance is None):
            raise FanInInventoryValidationError(
                f"fan-in publication fence outcome is malformed or ambiguous: {outcome}"
            )
        if acceptance is not None:
            if (
                acceptance.guard_root != root.name
                or acceptance.guard_generation != generation.path.name
                or acceptance.task != generation.record[0]
                or acceptance.claim_name != generation.record[1]
                or acceptance.claim_token != generation.record[2]
            ):
                raise FanInInventoryValidationError(
                    f"fan-in publication acceptance conflicts with its guard: {outcome}"
                )
            _reject_unreachable_fanin_generations(root, reachable)
            return _FanInFenceGeneration(
                generation.path, generation.identity, generation.record, acceptance,
            )
        path = outcome
        path_was_observed = True
    raise FanInInventoryValidationError(
        f"fan-in publication fence chain exceeds {_FANIN_GUARD_CHAIN_MAX_GENERATIONS} generations: {root}"
    )


def _fanin_generation_is_active(generation: _FanInFenceGeneration, claimed: Path) -> bool:
    if generation.acceptance is not None:
        return False
    _task, claim_name, claim_token, renewed_at = generation.record
    if _fanin_fence_release_marker(generation.path, claim_token).exists():
        return False
    return (
        time.time() - renewed_at <= _FANIN_PRODUCER_LEASE_SECONDS
        or _claim_token_is_live(claimed / claim_name, claim_token)
    )


def _fanin_fence_is_current(fence: _FanInFilesystemFence) -> bool:
    try:
        stat = fence.path.stat()
    except OSError:
        return False
    record = _read_fanin_fence(fence.path)
    if not (
        (stat.st_dev, stat.st_ino) == fence.identity
        and not _fanin_fence_release_marker(fence.path, fence.claim_token).exists()
        and record is not None
        and record[0] == fence.task
        and record[1] == fence.claim_name
        and record[2] == fence.claim_token
    ):
        return False
    generation = _FanInFenceGeneration(fence.path, fence.identity, record)
    try:
        _fanin_fence_successor_path(fence.root, generation).stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _fanin_fence_is_active(path: Path, claimed: Path) -> bool:
    generation = _read_current_fanin_fence(path)
    if generation is None:
        return False
    return _fanin_generation_is_active(generation, claimed)


def _publish_initialized_fanin_generation(
    target: Path,
    task: TaskManifest,
    fanin_task: FanInTask,
) -> bool:
    """Atomically publish one initialized, never-reused generation directory."""
    temporary = target.parent / f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    published = False
    try:
        temporary.mkdir()
        _write_fanin_fence(temporary, fanin_task, task.claim_path.name, task.claim_token)
        try:
            os.rename(temporary, target)
        except OSError as exc:
            if not _is_fanin_target_collision(exc):
                raise
            return False
        published = True
        _fsync_directory(target.parent)
        return True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _publish_initialized_fanin_acceptance(
    target: Path,
    acceptance: _FanInAcceptance,
) -> bool:
    """Atomically install a terminal acceptance into a generation outcome slot."""
    temporary = target.parent / f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    published = False
    try:
        temporary.mkdir()
        acceptance_path = _fanin_acceptance_path(temporary)
        with acceptance_path.open("x", encoding="utf-8") as handle:
            json.dump(
                _fanin_acceptance_payload(acceptance), handle,
                sort_keys=True, separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        try:
            os.rename(temporary, target)
        except OSError as exc:
            if not _is_fanin_target_collision(exc):
                raise
            return False
        published = True
        _fsync_directory(target.parent)
        return True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _start_fanin_guard_heartbeat(
    root: Path,
    path: Path,
    task: TaskManifest,
    fanin_task: FanInTask,
) -> _FanInFilesystemFence:
    """Validate ownership of a newly published generation and start renewal."""
    stat = path.stat()
    record = _read_fanin_fence(path)
    expected = (fanin_task, task.claim_path.name, task.claim_token)
    if record is None or record[:3] != expected:
        raise FanInInventoryValidationError(f"new fan-in publication fence is malformed: {path}")
    stop = threading.Event()
    fence = _FanInFilesystemFence(
        root, path, (stat.st_dev, stat.st_ino), fanin_task, task.claim_path.name,
        task.claim_token, stop, threading.Thread(),
    )

    def renew() -> None:
        interval = min(_FANIN_HEARTBEAT_INTERVAL_SECONDS, _FANIN_PRODUCER_LEASE_SECONDS / 3)
        while not stop.wait(max(interval, 0.001)):
            if not _fanin_fence_is_current(fence):
                return
            try:
                _write_fanin_fence(fence.path, fanin_task, task.claim_path.name, task.claim_token)
            except OSError:
                return

    fence.heartbeat = threading.Thread(target=renew, name="fanin-fence-heartbeat", daemon=True)
    fence.heartbeat.start()
    return fence


def _acquire_fanin_guard(
    path: Path,
    task: TaskManifest,
    fanin_task: FanInTask,
    *,
    nonblocking: bool = False,
) -> _FanInFilesystemFence:
    """Append an initialized guard generation after the current inactive owner."""
    del nonblocking  # Every acquisition uses the same bounded, non-waiting loop.
    for _attempt in range(_FANIN_GUARD_ACQUIRE_ATTEMPTS):
        current = _read_current_fanin_fence(path)
        if current is None:
            target = path
        else:
            if current.acceptance is not None:
                raise _FanInTransientError(
                    f"fan-in publication fence is already accepted: {path.name}"
                )
            if _fanin_generation_is_active(current, task.claim_path.parent):
                raise _FanInTransientError(f"fan-in publication fence is held: {current.path.name}")
            target = _fanin_fence_successor_path(path, current)
        if _publish_initialized_fanin_generation(target, task, fanin_task):
            return _start_fanin_guard_heartbeat(path, target, task, fanin_task)
    raise _FanInTransientError(f"fan-in publication fence kept advancing: {path.name}")


def _release_fanin_guard(fence: _FanInFilesystemFence) -> None:
    """Publish a generation-specific release marker without mutating its directory."""
    fence.stop.set()
    fence.heartbeat.join(timeout=max(_FANIN_HEARTBEAT_INTERVAL_SECONDS, 0.1) + 0.1)
    marker = _fanin_fence_release_marker(fence.path, fence.claim_token)
    try:
        with marker.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        pass
    _fsync_directory(fence.path.parent)


def _read_fanin_publish_lock(base: Path) -> tuple[FanInTask, str, str] | None:
    """Return the task owning a well-formed per-worker publish lock."""
    lock = _fanin_publish_lock_path(base)
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "base", "task", "claim_name", "claim_token",
    }:
        return None
    if (
        payload["schema_version"] != 3
        or payload["base"] != base.name
        or not isinstance(payload["task"], dict)
        or not isinstance(payload["claim_name"], str)
        or not payload["claim_name"]
        or not isinstance(payload["claim_token"], str)
        or not payload["claim_token"]
    ):
        return None
    task = payload["task"]
    if set(task) != {"task_id", "iteration", "offset", "count", "seed", "out", "policy"}:
        return None
    numeric = (task["iteration"], task["offset"], task["count"], task["seed"])
    if (
        not isinstance(task["task_id"], str)
        or not task["task_id"]
        or not all(_is_int(value) for value in numeric)
        or not isinstance(task["out"], str)
        or not task["out"]
        or not isinstance(task["policy"], str)
        or not task["policy"]
    ):
        return None
    candidate = FanInTask(task["task_id"], *numeric, task["out"], task["policy"])
    if candidate.iteration < 0 or candidate.offset < 0 or candidate.count <= 0 or candidate.seed < 0:
        return None
    return candidate, payload["claim_name"], payload["claim_token"]


def _acquire_fanin_publish_lock(
    base: Path,
    task: TaskManifest,
    fanin_task: FanInTask,
) -> tuple[Path, tuple[int, int]]:
    """Serialize one worker shard and reclaim only an inactive claimant's lock."""
    lock = _fanin_publish_lock_path(base)
    payload = {
        "schema_version": 3,
        "base": base.name,
        "task": _fanin_task_payload(fanin_task),
        "claim_name": task.claim_path.name,
        "claim_token": task.claim_token,
    }
    guard = _acquire_fanin_guard(_fanin_publish_guard_path(base), task, fanin_task)
    try:
        owner = _read_fanin_publish_lock(base) if lock.exists() else None
        if lock.exists() and owner is None:
            raise FanInInventoryValidationError(f"fan-in publish lock is malformed: {lock}")
        if owner is not None:
            owner_task, owner_claim_name, owner_token = owner
            if owner_token != task.claim_token and _claim_token_is_live(
                task.claim_path.parent / owner_claim_name, owner_token,
            ):
                raise _FanInTransientError(
                    f"fan-in worker shard {base.name} is actively publishing {owner_task.task_id!r}"
                )
        temporary = lock.parent / f".{lock.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, lock)
            _fsync_directory(lock.parent)
        finally:
            temporary.unlink(missing_ok=True)
        stat = lock.stat()
        return lock, (stat.st_dev, stat.st_ino)
    finally:
        _release_fanin_guard(guard)


def _release_fanin_publish_lock(lock: Path, identity: tuple[int, int]) -> None:
    """Release only the exact lock inode this publisher acquired.

    Publication normally removes the claim before this cleanup, so an orphaned
    lock is intentionally left for the next acquirer to reclaim under its
    filesystem fence. This avoids a stale publisher's check-then-unlink race.
    """
    del lock, identity


@dataclass
class _FanInTaskPublicationLease:
    guard: _FanInFilesystemFence
    record: Path
    claim_token: str


def _fanin_task_lock_record(cache_dir: Path, task_id: str) -> Path:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    directory = cache_dir / _FANIN_TASK_LOCK_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.json"


def _fanin_task_fence_path(cache_dir: Path, task_id: str) -> Path:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return cache_dir / _FANIN_TASK_LOCK_DIRECTORY / f"{digest}.guard"


def _read_fanin_task_lock(record: Path) -> tuple[FanInTask, str, str] | None:
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "task", "claim_name", "claim_token"}:
        return None
    if (
        payload["schema_version"] != 1
        or not isinstance(payload["task"], dict)
        or not isinstance(payload["claim_name"], str)
        or not payload["claim_name"]
        or not isinstance(payload["claim_token"], str)
        or not payload["claim_token"]
    ):
        return None
    raw_task = payload["task"]
    if set(raw_task) != {"task_id", "iteration", "offset", "count", "seed", "out", "policy"}:
        return None
    numeric = (raw_task["iteration"], raw_task["offset"], raw_task["count"], raw_task["seed"])
    if (
        not isinstance(raw_task["task_id"], str)
        or not raw_task["task_id"]
        or not all(_is_int(value) for value in numeric)
        or not isinstance(raw_task["out"], str)
        or not raw_task["out"]
        or not isinstance(raw_task["policy"], str)
        or not raw_task["policy"]
    ):
        return None
    return (
        FanInTask(raw_task["task_id"], *numeric, raw_task["out"], raw_task["policy"]),
        payload["claim_name"],
        payload["claim_token"],
    )


def _write_fanin_task_lock(record: Path, task: TaskManifest, fanin_task: FanInTask) -> None:
    payload = {
        "schema_version": 1,
        "task": _fanin_task_payload(fanin_task),
        "claim_name": task.claim_path.name,
        "claim_token": task.claim_token,
    }
    temporary = record.parent / f".{record.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, record)
        _fsync_directory(record.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _acquire_fanin_task_publication_lease(
    cache_dir: Path,
    task: TaskManifest,
    fanin_task: FanInTask,
) -> _FanInTaskPublicationLease:
    """Fence one task across every worker base until its acceptance is decided."""
    record = _fanin_task_lock_record(cache_dir, fanin_task.task_id)
    guard = _acquire_fanin_guard(_fanin_task_fence_path(cache_dir, fanin_task.task_id), task, fanin_task, nonblocking=True)
    try:
        previous = _read_fanin_task_lock(record) if record.exists() else None
        if record.exists() and previous is None:
            raise FanInInventoryValidationError(f"fan-in task publication lock is malformed: {record}")
        if previous is not None:
            previous_task, _previous_claim_name, _previous_token = previous
            if previous_task != fanin_task:
                if _fanin_route_conflicts(previous_task, fanin_task):
                    raise FanInRouteConflictError(
                        f"fan-in task {task.base!r} has different output or policy route"
                    )
                raise FanInInventoryValidationError(
                    f"fan-in task publication lock conflicts with {task.base!r}"
                )
        _write_fanin_task_lock(record, task, fanin_task)
    except Exception:
        _release_fanin_guard(guard)
        raise
    return _FanInTaskPublicationLease(guard, record, task.claim_token)


def _release_fanin_task_publication_lease(lease: _FanInTaskPublicationLease) -> None:
    """Retire the guard without deleting a reusable shared lock record.

    The next generation owns replacement under the guard handoff. Leaving the
    record behind is intentional: a stale owner has no portable conditional
    unlink primitive for a pathname a successor may already have replaced.
    """
    _release_fanin_guard(lease.guard)


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
    candidate = FanInTask(
        task.base,
        task.iteration,
        task.offset,
        task.count,
        task.seed,
        str(task.out),
        task.policy,
    )
    if (
        not candidate.task_id
        or candidate.iteration < 0
        or candidate.offset < 0
        or candidate.count <= 0
        or candidate.seed < 0
        or not candidate.out
        or not candidate.policy
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
        "out": task.out,
        "policy": task.policy,
    }


def _fanin_route_record(queue: Path, task_id: str) -> Path:
    """Return this task's append-only, queue-local route provenance record."""
    directory = queue / _FANIN_ROUTE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _fanin_task_from_payload(payload: Any) -> FanInTask | None:
    if not isinstance(payload, dict) or set(payload) != {
        "task_id", "iteration", "offset", "count", "seed", "out", "policy",
    }:
        return None
    task_id = payload["task_id"]
    numeric = (payload["iteration"], payload["offset"], payload["count"], payload["seed"])
    out = payload["out"]
    policy = payload["policy"]
    if (
        not isinstance(task_id, str)
        or not task_id
        or not all(_is_int(value) for value in numeric)
        or not isinstance(out, str)
        or not out
        or not isinstance(policy, str)
        or not policy
    ):
        return None
    task = FanInTask(task_id, *numeric, out, policy)
    if task.iteration < 0 or task.offset < 0 or task.count <= 0 or task.seed < 0:
        return None
    return task


def _read_fanin_route(record: Path) -> tuple[Path, FanInTask] | None:
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "cache_dir", "task"}:
        return None
    if payload["schema_version"] != 1 or not isinstance(payload["cache_dir"], str) or not payload["cache_dir"]:
        return None
    task = _fanin_task_from_payload(payload["task"])
    if task is None:
        return None
    return Path(payload["cache_dir"]), task


def _resolve_fanin_route(queue: Path, task: TaskManifest) -> Path:
    """Bind one task ID to its first fan-in root and route, or reject divergence.

    The record is created with ``link`` and never replaced. Recovery therefore
    searches the original root even when a retry's caller-supplied destination
    changes, and malformed provenance is terminal rather than guessed.
    """
    candidate = _fanin_task_from_task_manifest(task)
    cache_dir = task.out.parent
    record = _fanin_route_record(queue, candidate.task_id)
    payload = {
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "task": _fanin_task_payload(candidate),
    }
    temporary = record.parent / f".{record.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, record)
        except FileExistsError:
            pass
        else:
            _fsync_directory(record.parent)
    finally:
        temporary.unlink(missing_ok=True)
    existing = _read_fanin_route(record)
    if existing is None:
        raise FanInInventoryValidationError(f"fan-in route provenance is malformed: {record}")
    established_root, established_task = existing
    if established_task != candidate or established_root != cache_dir:
        if _fanin_route_conflicts(established_task, candidate) or established_root != cache_dir:
            raise FanInRouteConflictError(
                f"fan-in task {task.base!r} has different output, policy, or cache root route"
            )
        raise FanInTaskConflictError(f"fan-in route provenance conflicts with {task.base!r}")
    return established_root


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
        if not isinstance(raw_task, dict) or set(raw_task) != {
            "task_id", "iteration", "offset", "count", "seed", "out", "policy",
        }:
            raise FanInValidationError(f"fan-in shard {path} has a malformed task entry")
        task_id = raw_task["task_id"]
        numeric = (raw_task["iteration"], raw_task["offset"], raw_task["count"], raw_task["seed"])
        out = raw_task["out"]
        policy = raw_task["policy"]
        if (
            not isinstance(task_id, str)
            or not task_id
            or not all(_is_int(value) for value in numeric)
            or not isinstance(out, str)
            or not out
            or not isinstance(policy, str)
            or not policy
        ):
            raise FanInValidationError(f"fan-in shard {path} has invalid task metadata")
        iteration, offset, count, seed = numeric
        if iteration < 0 or offset < 0 or count <= 0 or seed < 0:
            raise FanInValidationError(f"fan-in shard {path} has out-of-range task metadata")
        if task_id in seen:
            raise FanInValidationError(f"fan-in shard {path} repeats task id {task_id!r}")
        seen.add(task_id)
        tasks.append(FanInTask(task_id, iteration, offset, count, seed, out, policy))
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


def _read_authoritative_fanin_acceptance(
    cache_dir: Path,
    task: FanInTask,
) -> _FanInAcceptance | None:
    root = _fanin_task_fence_path(cache_dir, task.task_id)
    current = _read_current_fanin_fence(root)
    if current is None or current.acceptance is None:
        return None
    if current.acceptance.task.task_id != task.task_id:
        raise FanInInventoryValidationError(
            f"fan-in acceptance guard conflicts with task id {task.task_id!r}"
        )
    return current.acceptance


def _fanin_candidate_is_accepted(path: Path, tasks: Sequence[FanInTask]) -> bool:
    """Whether one cumulative version is fully authenticated and trainable."""
    lineage, separator, version_text = path.name.rpartition("-v")
    if not separator or not version_text.isascii() or not version_text.isdecimal():
        raise FanInInventoryValidationError(f"fan-in shard has invalid version name: {path}")
    version = int(version_text)
    publication = _read_fanin_publication(path)
    if publication is None:
        raise FanInInventoryValidationError(
            f"fan-in shard {path} has missing or malformed publication provenance"
        )
    acceptances: list[_FanInAcceptance] = []
    for task_index, task in enumerate(tasks):
        acceptance = _read_authoritative_fanin_acceptance(path.parent, task)
        if acceptance is None:
            if task_index == len(tasks) - 1 and publication.task == task:
                current = _read_current_fanin_fence(
                    _fanin_task_fence_path(path.parent, task.task_id)
                )
                if current is not None and current.path.name != publication.guard_generation:
                    _validate_fanin_target_acceptance(path, tasks, publication)
                    return False
            raise FanInInventoryValidationError(
                f"fan-in shard {path} has no immutable acceptance for {task.task_id!r}"
            )
        if acceptance.task != task:
            raise FanInInventoryValidationError(
                f"fan-in shard {path} has conflicting metadata for {task.task_id!r}"
            )
        if acceptance.lineage != lineage:
            _validate_fanin_target_acceptance(path, tasks, publication)
            return False
        if (
            acceptance.task_index != task_index
            or acceptance.version > version
            or acceptance.prefix_sha256 != _fanin_manifest_prefix_sha256(tasks, task_index)
        ):
            raise FanInInventoryValidationError(
                f"fan-in shard {path} has ambiguous lineage provenance for {task.task_id!r}"
            )
        acceptances.append(acceptance)
    final = acceptances[-1]
    if final.target != path.name or final.version != version:
        _validate_fanin_target_acceptance(path, tasks, publication)
        return False
    _validate_fanin_target_acceptance(path, tasks, final)
    return True


def _select_accepted_fanin_shards(cache_dir: Path) -> list[Path]:
    lineages: set[str] = set()
    for candidate in Path(cache_dir).glob("shard-w*-v*"):
        lineage, separator, suffix = candidate.name.rpartition("-v")
        if separator and suffix.isascii() and suffix.isdecimal():
            lineages.add(lineage)
    selected: list[Path] = []
    for lineage in sorted(lineages):
        base = Path(cache_dir) / lineage
        for _version, path in reversed(_shard_versions(base)):
            tasks = _read_fanin_manifest(path)
            if _cache_record_count(path) != sum(task.count for task in tasks):
                raise FanInInventoryValidationError(
                    f"fan-in shard {path} cache games disagree with its task manifest"
                )
            if _fanin_candidate_is_accepted(path, tasks):
                selected.append(path)
                break
    return selected


def _read_selected_fanin_shards(
    cache_dir: Path,
    *,
    expected_iteration: int | None = None,
) -> tuple[tuple[FanInShard, ...], dict[str, FanInTask]]:
    vanished: _SelectedFanInVersionVanishedError | None = None
    for _attempt in range(_SELECTED_FANIN_READ_ATTEMPTS):
        shards: list[FanInShard] = []
        by_task_id: dict[str, FanInTask] = {}
        try:
            selected_paths = _select_accepted_fanin_shards(cache_dir)
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
        try:
            stable_paths = _select_accepted_fanin_shards(cache_dir)
        except _SelectedFanInVersionVanishedError as exc:
            vanished = exc
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        if stable_paths != selected_paths:
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
            for path in _select_accepted_fanin_shards(cache_dir):
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
    versions = _shard_versions(base)
    version = versions[-1][0] if versions else 0
    current: Path | None = None
    current_tasks: tuple[FanInTask, ...] = ()
    try:
        for _candidate_version, candidate in reversed(versions):
            tasks = _read_fanin_manifest(candidate)
            if _cache_record_count(candidate) != sum(task.count for task in tasks):
                raise FanInInventoryValidationError(
                    f"fan-in shard {candidate} cache games disagree with its task manifest"
                )
            if _fanin_candidate_is_accepted(candidate, tasks):
                current = candidate
                current_tasks = tasks
                break
        if current is not None and any(task.iteration != expected_iteration for task in current_tasks):
            raise FanInInventoryValidationError(f"fan-in shard {current} has a task from the wrong iteration")
    except _SelectedFanInVersionVanishedError as exc:
        raise _FanInTransientError(f"current worker shard vanished: {current}") from exc
    except FanInInventoryValidationError:
        raise
    except FanInValidationError as exc:
        raise FanInInventoryValidationError(
            f"selected worker shard is corrupt: {exc}"
        ) from exc
    return current, version, current_tasks


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
    for task in tasks:
        if not task.out or not task.policy:
            raise FanInValidationError("fan-in task manifest requires non-empty output and policy routes")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fanin_manifest_prefix_sha256(tasks: Sequence[FanInTask], task_index: int) -> str:
    payload = [_fanin_task_payload(task) for task in tasks[:task_index + 1]]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fanin_content_sha256(path: Path) -> str:
    """Hash all published cache files except the self-describing publication proof."""
    digest = hashlib.sha256()
    try:
        entries = sorted(path.rglob("*"), key=lambda entry: entry.relative_to(path).as_posix())
        for entry in entries:
            relative = entry.relative_to(path).as_posix()
            if relative == FANIN_PUBLICATION_NAME or entry.is_dir():
                continue
            if entry.is_symlink() or not entry.is_file():
                raise FanInValidationError(f"fan-in shard {path} has unsupported cache entry {relative!r}")
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(entry.stat().st_size.to_bytes(8, "big"))
            with entry.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    except FileNotFoundError as exc:
        raise _SelectedFanInVersionVanishedError(f"selected fan-in shard vanished: {path}") from exc
    except OSError as exc:
        raise FanInValidationError(f"fan-in shard content is unreadable: {path}") from exc
    return digest.hexdigest()


def _build_fanin_acceptance(
    path: Path,
    target: Path,
    lineage: Path,
    version: int,
    tasks: Sequence[FanInTask],
    task_index: int,
    fence: _FanInFilesystemFence,
) -> _FanInAcceptance:
    return _FanInAcceptance(
        task=tasks[task_index],
        guard_root=fence.root.name,
        guard_generation=fence.path.name,
        claim_name=fence.claim_name,
        claim_token=fence.claim_token,
        lineage=lineage.name,
        target=target.name,
        version=version,
        task_index=task_index,
        prefix_sha256=_fanin_manifest_prefix_sha256(tasks, task_index),
        manifest_sha256=_sha256_file(path / FANIN_MANIFEST_NAME),
        metadata_sha256=_sha256_file(path / "metadata.json"),
        content_sha256=_fanin_content_sha256(path),
        record_count=_cache_record_count(path),
    )


def _write_fanin_publication(path: Path, acceptance: _FanInAcceptance) -> None:
    publication = path / FANIN_PUBLICATION_NAME
    temporary = path / f".{FANIN_PUBLICATION_NAME}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                _fanin_acceptance_payload(acceptance), handle,
                sort_keys=True, separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, publication)
        _fsync_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_fanin_target_acceptance(
    path: Path,
    tasks: Sequence[FanInTask],
    acceptance: _FanInAcceptance,
) -> None:
    publication = _read_fanin_publication(path)
    if publication is None:
        raise FanInInventoryValidationError(
            f"fan-in shard {path} has missing or malformed publication provenance"
        )
    if publication != acceptance:
        raise FanInInventoryValidationError(
            f"fan-in shard {path} publication provenance conflicts with acceptance"
        )
    if (
        acceptance.target != path.name
        or acceptance.task_index != len(tasks) - 1
        or tasks[acceptance.task_index] != acceptance.task
        or acceptance.prefix_sha256 != _fanin_manifest_prefix_sha256(tasks, acceptance.task_index)
        or acceptance.manifest_sha256 != _sha256_file(path / FANIN_MANIFEST_NAME)
        or acceptance.metadata_sha256 != _sha256_file(path / "metadata.json")
        or acceptance.content_sha256 != _fanin_content_sha256(path)
        or acceptance.record_count != _cache_record_count(path)
    ):
        raise FanInInventoryValidationError(
            f"fan-in shard {path} content or lineage disagrees with its acceptance"
        )


def _accept_fanin_publication(root: Path, publication: _FanInAcceptance) -> _FanInAcceptance:
    """Race terminal acceptance against successor installation at one atomic slot."""
    current = _read_current_fanin_fence(root)
    if current is None:
        raise FanInInventoryValidationError(
            f"fan-in publication acceptance has no guard root: {root}"
        )
    if current.acceptance is not None:
        return current.acceptance
    if (
        publication.guard_root != root.name
        or publication.guard_generation != current.path.name
        or publication.task != current.record[0]
        or publication.claim_name != current.record[1]
        or publication.claim_token != current.record[2]
    ):
        raise _FanInTransientError(
            f"fan-in publication generation was revoked before acceptance: {publication.task.task_id}"
        )
    outcome = _fanin_fence_successor_path(root, current)
    _publish_initialized_fanin_acceptance(outcome, publication)
    resolved = _read_current_fanin_fence(root)
    if resolved is None or resolved.acceptance is None:
        raise _FanInTransientError(
            f"fan-in publication generation lost its acceptance race: {publication.task.task_id}"
        )
    return resolved.acceptance


def _fsync_directory(path: Path) -> None:
    """Persist a manifest replacement or version rename before acknowledging it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preflight_fanin_filesystem(cache_dir: Path) -> None:
    """Verify local support for directory operations required by fan-in.

    Fan-in deliberately does not use ``flock``: all workers that share this
    cache directory must instead see POSIX-atomic ``mkdir`` and same-directory
    ``rename`` operations. Deployments must mount the cache on a filesystem
    with those cross-node semantics (for example a POSIX NFS/CephFS mount), not
    an object-store FUSE layer. This local probe fails closed when those
    primitives are unavailable, but cannot establish cross-node atomicity. The
    private deployment smoke must make workers on multiple nodes contend for
    one guard on the shared mount before launch.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / f".fanin-filesystem-probe.{os.getpid()}.{uuid.uuid4().hex}"
    moved = probe.with_name(f".{probe.name}.moved")
    try:
        probe.mkdir()
        os.rename(probe, moved)
        probe.mkdir()
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in cache filesystem lacks required atomic mkdir/rename support: {cache_dir}: {exc}"
        ) from exc
    finally:
        shutil.rmtree(probe, ignore_errors=True)
        shutil.rmtree(moved, ignore_errors=True)


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


def _fanin_route_conflicts(existing: FanInTask, candidate: FanInTask) -> bool:
    """Whether matching queue contents were already accepted on another route."""
    return (
        existing.task_id == candidate.task_id
        and existing.iteration == candidate.iteration
        and existing.offset == candidate.offset
        and existing.count == candidate.count
        and existing.seed == candidate.seed
        and (existing.out != candidate.out or existing.policy != candidate.policy)
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

    Immutable acceptance is fan-in's input commit point. Therefore revocation
    after acceptance cannot discard output: if the old claim vanished, recreate
    a matching done marker from the parsed task metadata instead.
    """
    done = queue / "done" / task.base
    created = False
    owns_claim = _claim_token_is_current(task)
    if not owns_claim:
        created = _write_done_marker(task, done)
    else:
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
    if owns_claim:
        try:
            task.claim_path.unlink()
        except OSError:
            # The durable done marker wins; a stale-claim cleanup is harmless.
            pass
    _remove_claim_token(task)


def _fail_claim(task: TaskManifest, queue: Path, worker: str) -> None:
    del worker  # The claim generation already carries the worker identity.
    try:
        os.rename(task.claim_path, queue / "failed" / f"{task.claim_path.name}.failed")
    except OSError:
        pass
    _remove_claim_token(task)


def _remove_stale_fanin_version(stale: Path) -> None:
    """Unpublish a lower cumulative version before deleting it for race-safe reads."""
    tombstone = stale.parent / f".{stale.name}.gc.{os.getpid()}.{time.monotonic_ns()}"
    try:
        os.rename(stale, tombstone)
    except FileNotFoundError:
        return
    shutil.rmtree(tombstone, ignore_errors=True)


def _inject_crash(
    crash_inject: Callable[[str, TaskManifest], None] | None,
    boundary: str,
    task: TaskManifest,
) -> None:
    if crash_inject is not None:
        crash_inject(boundary, task)


def _recover_pending_fanin_publication(
    cache_dir: Path,
    candidate: FanInTask,
) -> _FanInAcceptance | None:
    """Finish target-before-acceptance recovery without recollecting the task."""
    root = _fanin_task_fence_path(cache_dir, candidate.task_id)
    current = _read_current_fanin_fence(root)
    if current is None:
        return None
    if current.acceptance is not None:
        if current.acceptance.task != candidate:
            if _fanin_route_conflicts(current.acceptance.task, candidate):
                raise FanInRouteConflictError(
                    f"accepted fan-in task {candidate.task_id!r} has different output or policy route"
                )
            raise FanInTaskConflictError(
                f"accepted fan-in task {candidate.task_id!r} conflicts with retried metadata"
            )
        return current.acceptance
    pending: list[tuple[Path, tuple[FanInTask, ...], _FanInAcceptance]] = []
    for target in Path(cache_dir).glob("shard-w*-v*"):
        publication_path = target / FANIN_PUBLICATION_NAME
        if not publication_path.exists():
            continue
        publication = _read_fanin_publication(target)
        if publication is None:
            raise FanInInventoryValidationError(
                f"fan-in shard {target} has malformed publication provenance"
            )
        if publication.task.task_id != candidate.task_id:
            continue
        if publication.task != candidate:
            if _fanin_route_conflicts(publication.task, candidate):
                raise FanInRouteConflictError(
                    f"pending fan-in task {candidate.task_id!r} has different output or policy route"
                )
            raise FanInTaskConflictError(
                f"pending fan-in task {candidate.task_id!r} conflicts with retried metadata"
            )
        if (
            publication.guard_root != root.name
            or publication.guard_generation != current.path.name
        ):
            continue
        tasks = _read_fanin_manifest(target)
        _validate_fanin_target_acceptance(target, tasks, publication)
        pending.append((target, tasks, publication))
    if not pending:
        return None
    if len(pending) != 1:
        names = ", ".join(sorted(target.name for target, _tasks, _publication in pending))
        raise FanInInventoryValidationError(
            f"fan-in task {candidate.task_id!r} has ambiguous pending targets: {names}"
        )
    return _accept_fanin_publication(root, pending[0][2])


def _recover_fanin_task(
    task: TaskManifest,
    queue: Path,
    *,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> bool:
    """Finalize a claim whose task is already durably selected, without collecting."""
    candidate = _fanin_task_from_task_manifest(task)
    cache_dir = _resolve_fanin_route(queue, task)
    _recover_pending_fanin_publication(cache_dir, candidate)
    existing = _find_committed_fanin_task(cache_dir, candidate)
    if existing is None:
        return False
    if existing != candidate:
        if _fanin_route_conflicts(existing, candidate):
            raise FanInRouteConflictError(
                f"accepted fan-in task {task.base!r} has different output or policy route"
            )
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


def _is_fanin_target_collision(exc: OSError) -> bool:
    """Whether a shared filesystem reported an already-published target."""
    return isinstance(exc, FileExistsError) or exc.errno in (errno.EEXIST, errno.ENOTEMPTY)


def _publish_fanin_task(
    task: TaskManifest,
    temporary_cache: Path,
    queue: Path,
    worker: str,
    *,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> tuple[Path | None, bool]:
    """Atomically publish one manifest-bearing version, fenced across workers."""
    cache_dir = _resolve_fanin_route(queue, task)
    fanin_task = _fanin_task_from_task_manifest(task)
    if _temporary_cache_record_count(temporary_cache, task) != fanin_task.count:
        raise FanInTaskValidationError(
            f"task {task.base} cache games do not match queue count {fanin_task.count}"
        )
    _recover_pending_fanin_publication(cache_dir, fanin_task)
    existing = _find_committed_fanin_task(cache_dir, fanin_task)
    if existing is not None:
        if existing != fanin_task:
            if _fanin_route_conflicts(existing, fanin_task):
                raise FanInRouteConflictError(
                    f"accepted fan-in task {task.base!r} has different output or policy route"
                )
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
    if not _claim_token_is_current(task):
        raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
    _reject_prepublication_done_marker(task, queue)

    task_lease = _acquire_fanin_task_publication_lease(cache_dir, task, fanin_task)
    try:
        # A stalled predecessor holds this lease. Once it exits or is fenced by
        # its claim token, a retry rechecks selected input before any append.
        existing = _find_committed_fanin_task(cache_dir, fanin_task)
        if existing is not None:
            if existing != fanin_task:
                if _fanin_route_conflicts(existing, fanin_task):
                    raise FanInRouteConflictError(
                        f"accepted fan-in task {task.base!r} has different output or policy route"
                    )
                raise FanInInventoryValidationError(
                    f"committed task {task.base!r} conflicts with its retried manifest"
                )
            _complete_claim(task, queue, crash_inject=crash_inject)
            shutil.rmtree(temporary_cache, ignore_errors=True)
            _inject_crash(crash_inject, "fanin-after-recovery", task)
            return None, True
        if not _claim_token_is_current(task):
            raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
        _reject_prepublication_done_marker(task, queue)
        base = cache_dir / f"shard-w{_sanitize_worker_id(worker)}"
        lock, lock_identity = _acquire_fanin_publish_lock(base, task, fanin_task)
        staging: Path | None = None
        owner_sidecar: Path | None = None
        published = False
        try:
            current, version, current_tasks = _read_current_worker_shard(base, task.iteration)
            if _recover_current_worker_task(current_tasks, fanin_task):
                _complete_claim(task, queue, crash_inject=crash_inject)
                shutil.rmtree(temporary_cache, ignore_errors=True)
                _inject_crash(crash_inject, "fanin-after-recovery", task)
                return None, True
            target = base.parent / f"{base.name}-v{version + 1}"
            staging = base.parent / f".{target.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
            owner_sidecar = _write_fanin_staging_owner(
                staging, fanin_task, producer_token=task.claim_token,
            )
            _refresh_fanin_staging_lease(staging, task.claim_token)
            if current is None:
                os.rename(temporary_cache, staging)
            else:
                from .dataset import concat_training_caches

                concat_training_caches((current, temporary_cache), staging)
                shutil.rmtree(temporary_cache, ignore_errors=True)
            _refresh_fanin_staging_lease(staging, task.claim_token)
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
            acceptance = _build_fanin_acceptance(
                staging, target, base, version + 1, version_tasks,
                len(version_tasks) - 1, task_lease.guard,
            )
            _write_fanin_publication(staging, acceptance)
            _inject_crash(crash_inject, "fanin-before-target-publication", task)
            if not _claim_token_is_current(task) or not _fanin_fence_is_current(task_lease.guard):
                raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
            if _read_fanin_staging_owner_record(staging) != (fanin_task, task.claim_token):
                raise _FanInTransientError(f"fan-in staging ownership changed before publishing {task.base}")
            _inject_crash(crash_inject, "fanin-after-publication-fence-check", task)
            try:
                os.rename(staging, target)
            except OSError as exc:
                if not _is_fanin_target_collision(exc):
                    raise
                existing = _find_committed_fanin_task(cache_dir, fanin_task)
                if existing == fanin_task:
                    _complete_claim(task, queue, crash_inject=crash_inject)
                    shutil.rmtree(staging, ignore_errors=True)
                    owner_sidecar.unlink(missing_ok=True)
                    shutil.rmtree(temporary_cache, ignore_errors=True)
                    return None, True
                if existing is not None:
                    if _fanin_route_conflicts(existing, fanin_task):
                        raise FanInRouteConflictError(
                            f"accepted fan-in task {task.base!r} has different output or policy route"
                        )
                    raise FanInInventoryValidationError(
                        f"committed task {task.base!r} conflicts with its retried manifest"
                    ) from exc
                raise _FanInTransientError(
                    f"fan-in worker shard target raced before publishing {task.base}"
                ) from exc
            published = True
            _fsync_directory(target.parent)
            _inject_crash(crash_inject, "fanin-after-target-publication", task)
            accepted = _accept_fanin_publication(task_lease.guard.root, acceptance)
            if accepted != acceptance:
                if accepted.task != fanin_task:
                    raise FanInInventoryValidationError(
                        f"fan-in acceptance conflicts with published task {task.base!r}"
                    )
                _complete_claim(task, queue, crash_inject=crash_inject)
                owner_sidecar.unlink(missing_ok=True)
                _remove_fanin_staging_lease(staging, task.claim_token)
                _fsync_directory(owner_sidecar.parent)
                return None, True
            try:
                _complete_claim(task, queue, crash_inject=crash_inject)
            except FanInTaskConflictError as exc:
                raise FanInInventoryValidationError(
                    f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
                ) from exc
            _inject_crash(crash_inject, "fanin-before-stale-cleanup", task)
            for stale_version, stale in _shard_versions(base):
                if stale_version < version + 1:
                    _remove_stale_fanin_version(stale)
            owner_sidecar.unlink(missing_ok=True)
            _remove_fanin_staging_lease(staging, task.claim_token)
            _fsync_directory(owner_sidecar.parent)
            return target, False
        except Exception:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            if owner_sidecar is not None and not published:
                owner_sidecar.unlink(missing_ok=True)
            if staging is not None and not published:
                _remove_fanin_staging_lease(staging, task.claim_token)
            raise
        finally:
            _release_fanin_publish_lock(lock, lock_identity)
    finally:
        _release_fanin_task_publication_lease(task_lease)


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
            try:
                try:
                    cache_dir = _resolve_fanin_route(queue, task)
                except FanInTaskValidationError:
                    # Preserve the established per-task failure classification;
                    # recovery below will reject malformed queue metadata.
                    cache_dir = task.out.parent
                _preflight_fanin_filesystem(cache_dir)
                _sweep_abandoned_fanin_staging(cache_dir, queue)
            except (FanInInventoryValidationError, OSError) as exc:
                log(f"TERMINAL fan-in filesystem preflight failed for {task.base}; preserving claim: {exc}")
                return finish(2)
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
                _remove_claim_token(task)
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
