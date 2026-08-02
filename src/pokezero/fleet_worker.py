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
import stat
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
    claim_identity: tuple[int, int, int, int, int, int] | None = None
    claim_parent_identity: tuple[int, int] | None = None
    claim_contents_sha256: str | None = None


@dataclass(frozen=True)
class _DoneMarkerWrite:
    """The exact marker generation installed by one acknowledgement attempt."""

    created: bool
    identity: tuple[int, int, int, int, int] | None = None

    def __bool__(self) -> bool:
        return self.created


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
    payload_files: tuple["_FanInPayloadFile", ...]
    route: "_FanInRouteResolution"


@dataclass(frozen=True)
class FanInShard:
    """The selected highest version for one worker shard."""

    worker: str
    version: int
    path: Path
    tasks: tuple[FanInTask, ...]


@dataclass(frozen=True)
class _FanInPayloadFile:
    """One selected cache file's generation and bytes before pathname reuse."""

    relative: tuple[str, ...]
    identity: tuple[int, int, int, int, int, int]
    sha256: str


@dataclass(frozen=True)
class _SelectedFanInShard:
    """An accepted shard and the immutable manifest generation that selected it."""

    path: Path
    cache_root_identity: tuple[int, int]
    root_snapshot: tuple[int, int, int, int, int, int]
    manifest_snapshot: tuple[int, int, int, int, int, int]
    publication_snapshot: tuple[int, int, int, int, int, int]
    acceptances: tuple["_FanInFenceGeneration", ...]
    tasks: tuple[FanInTask, ...]
    payload_files: tuple[_FanInPayloadFile, ...]


@dataclass(frozen=True)
class _FanInRouteResolution:
    """The append-only route record generation that authorizes one commit."""

    root: Path
    root_identity: tuple[int, int] | None
    record: Path
    record_parent_identity: tuple[int, int]
    record_identity: tuple[int, int, int, int, int, int]
    record_sha256: str
    task: FanInTask


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


def _canonical_fanin_physical_path(value: Path | str, label: str) -> Path:
    """Require a fan-in protocol path to be absolute and symlink-resolved."""
    route = Path(value)
    if not route.is_absolute():
        raise FanInInventoryValidationError(
            f"fan-in {label} must be an absolute canonical physical path: {route}"
        )
    try:
        physical = route.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} cannot be resolved physically: {route}"
        ) from exc
    if route != physical:
        raise FanInInventoryValidationError(
            f"fan-in {label} is not a canonical physical path: {route}"
        )
    return physical


def _canonical_fanin_output_path(value: Path | str) -> Path:
    """Require the queue route to name one absolute, symlink-resolved output."""
    return _canonical_fanin_physical_path(value, "output route")


def _canonical_fanin_task_manifest(task: TaskManifest) -> TaskManifest:
    """Canonicalize the route before it can become fan-in provenance."""
    return TaskManifest(**{**task.__dict__, "out": _canonical_fanin_output_path(task.out)})


def _parse_manifest(path: Path, base: str, *, fanin: bool = False) -> TaskManifest:
    return _parse_manifest_text(path.read_text(encoding="utf-8"), path, base, fanin=fanin)


def _parse_manifest_text(
    contents: str,
    path: Path,
    base: str,
    *,
    fanin: bool = False,
) -> TaskManifest:
    fields: dict[str, str] = {}
    for line in contents.splitlines():
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
        task = TaskManifest(
            base=base,
            claim_path=path,
            iteration=int(fields["a_iter"]),
            offset=int(fields.get("a_offset", "0")),
            count=int(fields["a_count"]),
            seed=int(fields["a_seed"]),
            out=Path(fields["a_out"]),
            policy=fields["a_policy"],
        )
        return _canonical_fanin_task_manifest(task) if fanin else task
    except KeyError as exc:
        raise ValueError(f"task manifest {base} missing key {exc}") from exc


def _fanin_directory_identity(observed: os.stat_result) -> tuple[int, int]:
    """The stable portion of a directory identity despite normal child writes."""
    return observed.st_dev, observed.st_ino


def _verify_fanin_directory_identity(
    path: Path,
    expected: tuple[int, int],
    label: str,
) -> None:
    observed = _fanin_authoritative_directory_stat(path, label)
    if _fanin_directory_identity(observed) != expected:
        raise FanInInventoryValidationError(
            f"fan-in {label} identity changed during validation: {path}"
        )


def _read_fanin_file_with_parent_snapshot(
    path: Path,
    label: str,
    *,
    expected_parent: tuple[int, int] | None = None,
) -> tuple[bytes | None, os.stat_result | None, tuple[int, int]]:
    """Read a no-follow record while retaining its parent directory identity."""
    descriptor, parent = _open_fanin_authoritative_directory(path.parent, f"{label} directory")
    parent_identity = _fanin_directory_identity(parent)
    try:
        if expected_parent is not None and parent_identity != expected_parent:
            raise FanInInventoryValidationError(
                f"fan-in {label} directory identity changed during validation: {path.parent}"
            )
        contents, identity = _read_fanin_authoritative_regular_file_snapshot(
            descriptor, path.name, label,
        )
    finally:
        os.close(descriptor)
    _verify_fanin_directory_identity(path.parent, parent_identity, f"{label} directory")
    return contents, identity, parent_identity


def _claim_manifest_snapshot(
    path: Path,
    *,
    expected_parent: tuple[int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return the exact regular-file generation a fan-in claimant parsed."""
    _contents, observed, _parent = _read_fanin_file_with_parent_snapshot(
        path, "claimed manifest", expected_parent=expected_parent,
    )
    if observed is None:
        raise FileNotFoundError(path)
    return _fanin_stat_snapshot(observed)


def _parse_claimed_fanin_manifest(
    path: Path,
    base: str,
    *,
    expected_parent: tuple[int, int] | None = None,
) -> TaskManifest:
    """Parse a claimed manifest only if its file and parent stay stable."""
    contents, observed, parent_identity = _read_fanin_file_with_parent_snapshot(
        path, "claimed manifest", expected_parent=expected_parent,
    )
    if contents is None or observed is None:
        raise FileNotFoundError(path)
    expected = _fanin_stat_snapshot(observed)
    try:
        task = _parse_manifest_text(contents.decode("utf-8"), path, base, fanin=True)
    except UnicodeDecodeError as exc:
        raise ValueError(f"task manifest {base} is not UTF-8") from exc
    if _claim_manifest_snapshot(path, expected_parent=parent_identity) != expected:
        raise FanInInventoryValidationError(
            f"fan-in claimed manifest changed while parsing: {path}"
        )
    return TaskManifest(
        **{
            **task.__dict__,
            "claim_identity": expected,
            "claim_parent_identity": parent_identity,
            "claim_contents_sha256": hashlib.sha256(contents).hexdigest(),
        }
    )


def _same_task_manifest_route(left: TaskManifest, right: TaskManifest) -> bool:
    """Whether pre-claim metadata still describes the claimed manifest."""
    return (
        left.base == right.base
        and left.iteration == right.iteration
        and left.offset == right.offset
        and left.count == right.count
        and left.seed == right.seed
        and left.out == right.out
        and left.policy == right.policy
    )


def _move_fanin_manifest_into_claim(
    candidate: Path,
    claim: Path,
    preview: TaskManifest,
) -> TaskManifest:
    """Rename only the previewed inode between the descriptor-bound queue parents."""
    if preview.claim_identity is None or preview.claim_parent_identity is None:
        raise FanInInventoryValidationError(
            f"fan-in task {candidate.name!r} has no stable preflight identity"
        )
    pending_descriptor, pending = _open_fanin_authoritative_directory(
        candidate.parent, "pending queue directory",
    )
    claimed_descriptor, claimed = _open_fanin_authoritative_directory(
        claim.parent, "claimed queue directory",
    )
    pending_identity = _fanin_directory_identity(pending)
    claimed_identity = _fanin_directory_identity(claimed)
    try:
        if pending_identity != preview.claim_parent_identity:
            raise FanInInventoryValidationError(
                f"fan-in pending queue directory changed after preflight: {candidate.parent}"
            )
        os.rename(
            candidate.name, claim.name,
            src_dir_fd=pending_descriptor, dst_dir_fd=claimed_descriptor,
        )
        contents, claimed_manifest = _read_fanin_authoritative_regular_file_snapshot(
            claimed_descriptor, claim.name, "claimed manifest",
        )
        if contents is None or claimed_manifest is None:
            raise FanInInventoryValidationError(
                f"fan-in claimed manifest disappeared during lease transition: {claim}"
            )
        try:
            task = _parse_manifest_text(contents.decode("utf-8"), claim, candidate.name, fanin=True)
        except UnicodeDecodeError as exc:
            raise ValueError(f"task manifest {candidate.name} is not UTF-8") from exc
        if (
            _fanin_stat_snapshot(claimed_manifest)[:2] != preview.claim_identity[:2]
            or
            hashlib.sha256(contents).hexdigest() != preview.claim_contents_sha256
            or not _same_task_manifest_route(preview, task)
        ):
            raise FanInInventoryValidationError(
                f"fan-in task {candidate.name!r} changed during lease transition"
            )
        _verify_fanin_directory_identity(
            candidate.parent, pending_identity, "pending queue directory",
        )
        _verify_fanin_directory_identity(
            claim.parent, claimed_identity, "claimed queue directory",
        )
        _verify_fanin_authoritative_regular_file_identity(
            claimed_descriptor, claim.name, claimed_manifest, "claimed manifest",
        )
        return TaskManifest(
            **{
                **task.__dict__,
                "claim_identity": _fanin_stat_snapshot(claimed_manifest),
                "claim_parent_identity": claimed_identity,
                "claim_contents_sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    finally:
        os.close(claimed_descriptor)
        os.close(pending_descriptor)


def claim_next_task(
    queue: Path,
    worker_id: str,
    *,
    before_claim: Callable[[Path, TaskManifest | None], None] | None = None,
    fanin: bool = False,
) -> TaskManifest | None:
    """Claim the first available pending manifest via atomic rename (or None)."""
    pending = queue / "pending"
    try:
        candidates = sorted(pending.glob("*.env"))
    except OSError:
        return None
    for candidate in candidates:
        try:
            preview = (
                _parse_claimed_fanin_manifest(candidate, candidate.name)
                if fanin else _parse_manifest(candidate, candidate.name)
            )
        except FileNotFoundError:
            continue
        except FanInInventoryValidationError:
            # Fan-in route provenance is a trust-boundary failure. Do not move
            # the manifest into claimed/ merely to quarantine it as a routine
            # task error.
            raise
        except ValueError:
            preview = None
        if before_claim is not None:
            before_claim(candidate, preview)
        # A claim pathname is a generation capability, not a reusable worker
        # slot. Terminal actions can therefore never target a later retry.
        claim = queue / "claimed" / f"{candidate.name}.{worker_id}.{uuid.uuid4().hex}"
        try:
            if fanin:
                if preview is None:
                    raise FanInInventoryValidationError(
                        f"fan-in task {candidate.name!r} has no valid preflight manifest"
                    )
                try:
                    task = _move_fanin_manifest_into_claim(candidate, claim, preview)
                except OSError:
                    continue  # lost the claim race; inspect the next pending manifest
            else:
                try:
                    os.rename(candidate, claim)
                except OSError:
                    continue  # lost the race; try the next manifest
                task = _parse_manifest(claim, candidate.name)
            _sweep_orphaned_claim_tokens(claim.parent, candidate.name)
            token = (
                _write_claim_token(
                    claim,
                    expected_claim=task.claim_identity,
                    expected_parent=task.claim_parent_identity,
                )
                if fanin else _write_claim_token(claim)
            )
            task = TaskManifest(**{**task.__dict__, "claim_token": token})
            if fanin and not _fanin_claim_manifest_is_current(task):
                raise FanInInventoryValidationError(
                    f"fan-in claimed manifest disappeared before lease binding: {claim}"
                )
            return task
        except FanInInventoryValidationError:
            # A manifest changed after preflight. Retain this exact claimed
            # generation for diagnosis instead of silently quarantining it.
            raise
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
            claim = claimed / claim_name
            identity = _claim_token_identity(payload)
            snapshot = _claim_manifest_snapshot(claim)
        except FileNotFoundError:
            lease.unlink(missing_ok=True)
            continue
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            # A malformed token is not evidence that a live claimant is dead.
            continue
        if identity is None or not _claim_snapshot_matches_identity(snapshot, identity):
            lease.unlink(missing_ok=True)


def _claim_token_identity(payload: Any) -> tuple[int, ...] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") == 2:
        values = (payload.get("claim_dev"), payload.get("claim_ino"), payload.get("claim_ctime_ns"))
        return values if all(_is_int(value) for value in values) else None
    if payload.get("schema_version") in (3, 4):
        values = (
            payload.get("claim_dev"), payload.get("claim_ino"), payload.get("claim_mode"),
            payload.get("claim_size"), payload.get("claim_mtime_ns"), payload.get("claim_ctime_ns"),
        )
        return values if all(_is_int(value) for value in values) else None
    return None


def _claim_snapshot_matches_identity(
    snapshot: tuple[int, int, int, int, int, int],
    identity: tuple[int, ...],
) -> bool:
    if len(identity) == 3:
        return (snapshot[0], snapshot[1], snapshot[5]) == identity
    return snapshot == identity


def _write_claim_token(
    claim: Path,
    *,
    expected_claim: tuple[int, int, int, int, int, int] | None = None,
    expected_parent: tuple[int, int] | None = None,
) -> str:
    """Record a fresh claimant generation so path reuse cannot revive a publisher."""
    token = uuid.uuid4().hex
    descriptor, parent = _open_fanin_authoritative_directory(claim.parent, "claimed queue directory")
    parent_identity = _fanin_directory_identity(parent)
    try:
        if expected_parent is not None and parent_identity != expected_parent:
            raise FanInInventoryValidationError(
                f"fan-in claimed queue directory changed before lease binding: {claim.parent}"
            )
        _contents, observed = _read_fanin_authoritative_regular_file_snapshot(
            descriptor, claim.name, "claimed manifest",
        )
        if observed is None:
            raise FileNotFoundError(claim)
        identity = _fanin_stat_snapshot(observed)
        if expected_claim is not None and identity != expected_claim:
            raise FanInInventoryValidationError(
                f"fan-in claimed manifest changed before lease binding: {claim}"
            )
        lease_name = _claim_token_path(claim, token).name
        temporary_name = f".{lease_name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
        payload = {
            "schema_version": 4,
            "claim": claim.name,
            "claim_dev": identity[0],
            "claim_ino": identity[1],
            "claim_mode": identity[2],
            "claim_size": identity[3],
            "claim_mtime_ns": identity[4],
            "claim_ctime_ns": identity[5],
            "claimed_parent_dev": parent_identity[0],
            "claimed_parent_ino": parent_identity[1],
            "token": token,
        }
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise FanInInventoryValidationError("fan-in claim lease creation requires O_NOFOLLOW support")
        temporary_descriptor = os.open(
            temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow, 0o600,
            dir_fd=descriptor,
        )
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            written = 0
            while written < len(encoded):
                written += os.write(temporary_descriptor, encoded[written:])
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        try:
            os.link(temporary_name, lease_name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=descriptor)
            except FileNotFoundError:
                pass
        _verify_fanin_authoritative_regular_file_identity(
            descriptor, lease_name,
            os.stat(lease_name, dir_fd=descriptor, follow_symlinks=False),
            "claim lease",
        )
    finally:
        os.close(descriptor)
    _verify_fanin_directory_identity(claim.parent, parent_identity, "claimed queue directory")
    current = _claim_manifest_snapshot(claim, expected_parent=parent_identity)
    if current != identity:
        raise ValueError(f"claim was reused before its lease was recorded: {claim}")
    if expected_claim is not None and current != expected_claim:
        raise FanInInventoryValidationError(
            f"fan-in claimed manifest changed while lease binding completed: {claim}"
        )
    return token


def _claim_token_parent_identity(payload: Any) -> tuple[int, int] | None:
    if not isinstance(payload, dict) or payload.get("schema_version") != 4:
        return None
    values = payload.get("claimed_parent_dev"), payload.get("claimed_parent_ino")
    return values if all(_is_int(value) for value in values) else None


def _claim_token_is_current(task: TaskManifest) -> bool:
    return _claim_token_is_live(
        task.claim_path, task.claim_token,
        expected_claim=task.claim_identity,
        expected_parent=task.claim_parent_identity,
    )


def _claim_token_is_live(
    claim: Path,
    token: str,
    *,
    expected_claim: tuple[int, int, int, int, int, int] | None = None,
    expected_parent: tuple[int, int] | None = None,
) -> bool:
    if not token:
        return False
    try:
        contents, observed, parent_identity = _read_fanin_file_with_parent_snapshot(
            claim, "claimed manifest", expected_parent=expected_parent,
        )
        lease_contents, _lease, lease_parent_identity = _read_fanin_file_with_parent_snapshot(
            _claim_token_path(claim, token), "claim lease", expected_parent=parent_identity,
        )
        if (
            contents is None
            or observed is None
            or lease_contents is None
            or lease_parent_identity != parent_identity
        ):
            return False
        snapshot = _fanin_stat_snapshot(observed)
        if expected_claim is not None and snapshot != expected_claim:
            return False
        payload = json.loads(lease_contents.decode("utf-8"))
    except (OSError, FanInValidationError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("claim") == claim.name
        and payload.get("token") == token
        and _claim_snapshot_matches_identity(snapshot, _claim_token_identity(payload) or ())
        and (
            payload.get("schema_version") != 4
            or _claim_token_parent_identity(payload) == parent_identity
        )
    )


def _fanin_claim_manifest_is_current(task: TaskManifest) -> bool:
    """Reject a live claim whose parsed manifest no longer matches its lease."""
    if task.claim_identity is None:
        return _claim_token_is_current(task)
    if task.claim_parent_identity is None:
        raise FanInInventoryValidationError(
            f"fan-in claimed manifest has no parent identity: {task.claim_path}"
        )
    try:
        snapshot = _claim_manifest_snapshot(
            task.claim_path, expected_parent=task.claim_parent_identity,
        )
    except FileNotFoundError:
        return False
    if snapshot != task.claim_identity:
        raise FanInInventoryValidationError(
            f"fan-in claimed manifest changed after lease binding: {task.claim_path}"
        )
    if not _claim_token_is_current(task):
        raise FanInInventoryValidationError(
            f"fan-in claim lease no longer binds its parsed manifest: {task.claim_path}"
        )
    return True


def _remove_claim_token(task: TaskManifest) -> None:
    """Remove only this claimant generation's sidecar after terminal handling."""
    lease = _claim_token_path(task.claim_path, task.claim_token)
    try:
        payload = json.loads(lease.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") in (2, 3, 4)
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
    payload = {
        "schema_version": 4,
        "staging": staging.name,
        "task": _fanin_task_payload(task),
        "producer_token": producer_token,
    }
    _write_fanin_authoritative_json(owner, payload, "staging owner record")
    return owner


def _read_fanin_staging_owner_record(staging: Path) -> tuple[FanInTask, str] | None:
    """Return sidecar ownership only when it names this exact staging path."""
    payload = _read_fanin_authoritative_json(
        _fanin_staging_owner_path(staging), "staging owner record",
    )
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
    candidate = _fanin_task_from_payload(task)
    if candidate is None:
        return None
    return candidate, payload["producer_token"]


def _read_fanin_staging_owner(staging: Path) -> FanInTask | None:
    record = _read_fanin_staging_owner_record(staging)
    return record[0] if record is not None else None


def _refresh_fanin_staging_lease(staging: Path, producer_token: str) -> None:
    if not producer_token:
        return
    lease = _fanin_staging_lease_path(staging)
    payload = {
        "schema_version": 1,
        "staging": staging.name,
        "producer_token": producer_token,
        "renewed_at": time.time(),
    }
    _write_fanin_authoritative_json(lease, payload, "staging producer lease")


def _fanin_staging_lease_is_active(staging: Path, producer_token: str) -> bool:
    if not producer_token:
        return False
    payload = _read_fanin_authoritative_json(
        _fanin_staging_lease_path(staging), "staging producer lease",
    )
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
    parent_identity: tuple[int, int] = ()
    chain: tuple[tuple[Path, tuple[int, int]], ...] = ()
    acceptance_path: Path | None = None
    acceptance_parent_identity: tuple[int, int] | None = None
    acceptance_identity: tuple[int, int, int, int, int, int] | None = None
    chain_records: tuple[tuple[Path, tuple[FanInTask, str, str]], ...] = ()


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
    owner_lock: Any
    heartbeat: threading.Thread


def _fanin_fence_payload(task: FanInTask, claim_name: str, claim_token: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": _fanin_task_payload(task),
        "claim_name": claim_name,
        "claim_token": claim_token,
        "renewed_at": time.time(),
    }


def _fanin_authoritative_directory_stat(path: Path, label: str) -> os.stat_result:
    """lstat one protocol directory so symlinks never become capabilities."""
    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} is unreadable: {path}"
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise FanInInventoryValidationError(
            f"fan-in {label} is not a real directory: {path}"
        )
    return observed


def _open_fanin_authoritative_directory(path: Path, label: str) -> tuple[int, os.stat_result]:
    """Open a real protocol directory and bind it to its lstat identity."""
    expected = _fanin_authoritative_directory_stat(path, label)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise FanInInventoryValidationError(
            f"fan-in {label} validation requires O_NOFOLLOW and O_DIRECTORY support"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow | directory_flag)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} is unreadable: {path}"
        ) from exc
    observed = os.fstat(descriptor)
    # Child creation/replacement legitimately updates a directory's mtime and
    # ctime. Callers bind authoritative child records separately, so opening a
    # mutable guard directory only needs the stable directory object identity.
    if _fanin_stat_identity(observed) != _fanin_stat_identity(expected):
        os.close(descriptor)
        raise FanInInventoryValidationError(
            f"fan-in {label} changed while opening: {path}"
        )
    return descriptor, expected


def _verify_fanin_authoritative_directory_identity(
    path: Path,
    expected: os.stat_result,
    label: str,
) -> None:
    observed = _fanin_authoritative_directory_stat(path, label)
    if _fanin_stat_identity(observed) != _fanin_stat_identity(expected):
        raise FanInInventoryValidationError(
            f"fan-in {label} identity changed during validation: {path}"
        )


def _read_fanin_authoritative_regular_file(
    descriptor: int,
    name: str,
    label: str,
) -> bytes | None:
    """Read one authoritative record without following or racing replacement."""
    contents, _expected = _read_fanin_authoritative_regular_file_snapshot(
        descriptor, name, label,
    )
    return contents


def _read_fanin_authoritative_regular_file_snapshot(
    descriptor: int,
    name: str,
    label: str,
) -> tuple[bytes | None, os.stat_result | None]:
    """Read one authoritative record and retain the generation it came from."""
    try:
        expected = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} is unreadable: {name}"
        ) from exc
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
        raise FanInInventoryValidationError(
            f"fan-in {label} is not a regular non-symlink file: {name}"
        )
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise FanInInventoryValidationError(
            f"fan-in {label} validation requires O_NOFOLLOW support"
        )
    try:
        file_descriptor = os.open(name, os.O_RDONLY | no_follow, dir_fd=descriptor)
    except FileNotFoundError:
        raise FanInInventoryValidationError(f"fan-in {label} vanished during validation: {name}") from None
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} is unreadable: {name}"
        ) from exc
    try:
        if _fanin_stat_snapshot(os.fstat(file_descriptor)) != _fanin_stat_snapshot(expected):
            raise FanInInventoryValidationError(
                f"fan-in {label} changed while opening: {name}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(file_descriptor, 1024 * 1024):
            chunks.append(chunk)
        if _fanin_stat_snapshot(os.fstat(file_descriptor)) != _fanin_stat_snapshot(expected):
            raise FanInInventoryValidationError(
                f"fan-in {label} changed during read: {name}"
            )
    finally:
        os.close(file_descriptor)
    try:
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FanInInventoryValidationError(f"fan-in {label} vanished during validation: {name}") from exc
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} is unreadable: {name}"
        ) from exc
    if _fanin_stat_snapshot(observed) != _fanin_stat_snapshot(expected):
        raise FanInInventoryValidationError(
            f"fan-in {label} changed during read: {name}"
        )
    return b"".join(chunks), expected


def _verify_fanin_authoritative_regular_file_identity(
    descriptor: int,
    name: str,
    expected: os.stat_result,
    label: str,
) -> None:
    """Require one no-follow record pathname to retain a prior snapshot."""
    try:
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} vanished during validation: {name}"
        ) from exc
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {label} is unreadable: {name}"
        ) from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or _fanin_stat_snapshot(observed) != _fanin_stat_snapshot(expected)
    ):
        raise FanInInventoryValidationError(
            f"fan-in {label} identity changed during validation: {name}"
        )


def _read_fanin_authoritative_file(path: Path, label: str) -> bytes | None:
    """Read a protocol file through a no-follow, stable parent descriptor."""
    contents, _expected = _read_fanin_authoritative_file_snapshot(path, label)
    return contents


def _read_fanin_authoritative_file_snapshot(
    path: Path,
    label: str,
) -> tuple[bytes | None, os.stat_result | None]:
    """Read a protocol file and retain the stable file generation it used."""
    descriptor, expected = _open_fanin_authoritative_directory(path.parent, f"{label} directory")
    try:
        contents, file_identity = _read_fanin_authoritative_regular_file_snapshot(
            descriptor, path.name, label,
        )
    finally:
        os.close(descriptor)
    _verify_fanin_authoritative_directory_identity(path.parent, expected, f"{label} directory")
    return contents, file_identity


def _read_fanin_authoritative_json(path: Path, label: str) -> Any | None:
    contents = _read_fanin_authoritative_file(path, label)
    if contents is None:
        return None
    try:
        return json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _write_fanin_authoritative_json(path: Path, payload: Any, label: str) -> None:
    """Replace one record through a stable, no-follow parent descriptor."""
    descriptor, expected = _open_fanin_authoritative_directory(path.parent, f"{label} directory")
    temporary_name = f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    try:
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise FanInInventoryValidationError(
                f"fan-in {label} creation requires O_NOFOLLOW support"
            )
        file_descriptor = os.open(
            temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow, 0o600,
            dir_fd=descriptor,
        )
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            written = 0
            while written < len(encoded):
                written += os.write(file_descriptor, encoded[written:])
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary_name, path.name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(descriptor)
    _verify_fanin_authoritative_directory_identity(path.parent, expected, f"{label} directory")


def _fanin_fence_release_marker_is_present(marker: Path) -> bool:
    """A release marker is authoritative only as a stable regular file."""
    return _read_fanin_authoritative_file(marker, "publication fence release marker") is not None


def _fanin_fence_from_payload(payload: Any) -> tuple[FanInTask, str, str, float] | None:
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
    task = _fanin_task_from_payload(raw_task)
    if task is None:
        return None
    return task, payload["claim_name"], payload["claim_token"], float(payload["renewed_at"])


def _read_fanin_fence(path: Path) -> tuple[FanInTask, str, str, float] | None:
    payload = _read_fanin_authoritative_json(
        _fanin_fence_owner_path(path), "publication fence owner record",
    )
    return _fanin_fence_from_payload(payload)


def _write_fanin_fence(path: Path, task: FanInTask, claim_name: str, claim_token: str) -> None:
    _write_fanin_authoritative_json(
        _fanin_fence_owner_path(path), _fanin_fence_payload(task, claim_name, claim_token),
        "publication fence owner record",
    )


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
        "schema_version": 2,
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
        "payload_files": [
            {
                "relative": list(payload.relative),
                "identity": list(payload.identity),
                "sha256": payload.sha256,
            }
            for payload in acceptance.payload_files
        ],
        "route": _fanin_route_resolution_payload(acceptance.route),
    }


def _fanin_acceptance_from_payload(payload: Any) -> _FanInAcceptance | None:
    expected = {
        "schema_version", "task", "guard_root", "guard_generation", "claim_name",
        "claim_token", "lineage", "target", "version", "task_index", "prefix_sha256",
        "manifest_sha256", "metadata_sha256", "content_sha256", "record_count", "payload_files",
        "route",
    }
    if not isinstance(payload, dict) or set(payload) != expected or payload["schema_version"] != 2:
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
    payload_files = _fanin_payload_files_from_payload(payload["payload_files"])
    route = _fanin_route_resolution_from_payload(payload["route"], task)
    if payload_files is None or route is None:
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
        payload_files=payload_files,
        route=route,
    )


def _read_fanin_acceptance(path: Path) -> _FanInAcceptance | None:
    payload = _read_fanin_authoritative_json(
        _fanin_acceptance_path(path), "publication fence acceptance record",
    )
    return _fanin_acceptance_from_payload(payload)


def _read_fanin_publication_snapshot(
    path: Path,
) -> tuple[_FanInAcceptance | None, tuple[int, int, int, int, int, int] | None]:
    root = _fanin_real_directory_stat(path)
    contents, observed, _parent = _read_fanin_file_with_parent_snapshot(
        path / FANIN_PUBLICATION_NAME, "shard publication provenance",
    )
    _verify_fanin_root_unchanged(path, root)
    if contents is None or observed is None:
        return None, None
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, _fanin_stat_snapshot(observed)
    return _fanin_acceptance_from_payload(payload), _fanin_stat_snapshot(observed)


def _read_fanin_publication(path: Path) -> _FanInAcceptance | None:
    return _read_fanin_publication_snapshot(path)[0]


def _read_fanin_guard_directory(
    path: Path,
) -> tuple[
    os.stat_result,
    bytes | None,
    tuple[FanInTask, str, str, float] | None,
    bytes | None,
    _FanInAcceptance | None,
]:
    """Read one guard/outcome directory as a coherent no-follow snapshot."""
    descriptor, expected = _open_fanin_authoritative_directory(path, "publication fence directory")
    try:
        owner_contents = _read_fanin_authoritative_regular_file(
            descriptor, "owner.json", "publication fence owner record",
        )
        acceptance_contents = _read_fanin_authoritative_regular_file(
            descriptor, _FANIN_ACCEPTANCE_NAME, "publication fence acceptance record",
        )
    finally:
        os.close(descriptor)
    _verify_fanin_authoritative_directory_identity(path, expected, "publication fence directory")
    owner_payload: Any | None = None
    acceptance_payload: Any | None = None
    try:
        if owner_contents is not None:
            owner_payload = json.loads(owner_contents.decode("utf-8"))
        if acceptance_contents is not None:
            acceptance_payload = json.loads(acceptance_contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Keep the raw-presence bits so an otherwise valid sibling cannot make
        # malformed authoritative state disappear.
        pass
    return (
        expected,
        owner_contents,
        _fanin_fence_from_payload(owner_payload),
        acceptance_contents,
        _fanin_acceptance_from_payload(acceptance_payload),
    )


def _installed_fanin_generation_paths(root: Path) -> set[Path]:
    """Enumerate only exact immutable outcome names for one guard root."""
    prefix = f".{root.name}.generation."
    installed: set[Path] = set()
    try:
        descriptor, expected = _open_fanin_authoritative_directory(
            root.parent, "publication fence parent directory",
        )
    except FileNotFoundError:
        return set()
    try:
        with os.scandir(os.dup(descriptor)) as entries:
            for entry in entries:
                if not entry.name.startswith(prefix):
                    continue
                digest = entry.name.removeprefix(prefix)
                if _is_sha256(digest):
                    installed.add(root.parent / entry.name)
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in publication fence directory is unreadable: {root.parent}"
        ) from exc
    finally:
        os.close(descriptor)
    _verify_fanin_authoritative_directory_identity(
        root.parent, expected, "publication fence parent directory",
    )
    return installed


def _reject_unreachable_fanin_generations(root: Path, reachable: set[Path]) -> None:
    unreachable = _installed_fanin_generation_paths(root) - reachable
    if unreachable:
        names = ", ".join(sorted(path.name for path in unreachable))
        raise FanInInventoryValidationError(
            f"fan-in publication fence has unreachable generations: {names}"
        )


def _verify_fanin_fence_generation_snapshots(
    snapshots: Sequence[tuple[Path, os.stat_result]],
) -> None:
    """Keep every generation traversed for one fence result on the same inode."""
    for path, expected in snapshots:
        try:
            _verify_fanin_authoritative_directory_identity(
                path, expected, "publication fence directory",
            )
        except FileNotFoundError as exc:
            raise FanInInventoryValidationError(
                f"fan-in publication fence generation vanished during traversal: {path}"
            ) from exc


def _verify_fanin_fence_generation_identity(generation: _FanInFenceGeneration) -> None:
    """Keep every predecessor, successor, and terminal record behind a result live."""
    if not generation.parent_identity:
        raise FanInInventoryValidationError(
            f"fan-in publication fence has no parent identity: {generation.path}"
        )
    _verify_fanin_directory_identity(
        generation.path.parent, generation.parent_identity, "publication fence parent directory",
    )
    for path, identity in generation.chain:
        _verify_fanin_directory_identity(path, identity, "publication fence directory")
    for path, expected in generation.chain_records:
        record = _read_fanin_fence(path)
        if record is None or record[:3] != expected:
            raise FanInInventoryValidationError(
                f"fan-in publication fence owner changed during validation: {path}"
            )
        _verify_fanin_directory_identity(path, dict(generation.chain)[path], "publication fence directory")
    if generation.acceptance is not None:
        if (
            generation.acceptance_path is None
            or generation.acceptance_parent_identity is None
            or generation.acceptance_identity is None
        ):
            raise FanInInventoryValidationError(
                f"fan-in publication acceptance has no record identity: {generation.path}"
            )
        contents, observed, parent_identity = _read_fanin_file_with_parent_snapshot(
            generation.acceptance_path,
            "publication fence acceptance record",
            expected_parent=generation.acceptance_parent_identity,
        )
        if (
            contents is None
            or observed is None
            or parent_identity != generation.acceptance_parent_identity
            or _fanin_stat_snapshot(observed) != generation.acceptance_identity
        ):
            raise FanInInventoryValidationError(
                f"fan-in publication acceptance identity changed during validation: {generation.acceptance_path}"
            )
        try:
            payload = json.loads(contents.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FanInInventoryValidationError(
                f"fan-in publication acceptance became malformed: {generation.acceptance_path}"
            ) from exc
        if _fanin_acceptance_from_payload(payload) != generation.acceptance:
            raise FanInInventoryValidationError(
                f"fan-in publication acceptance changed during validation: {generation.acceptance_path}"
            )


def _read_current_fanin_fence(root: Path) -> _FanInFenceGeneration | None:
    """Resolve an immutable successor/acceptance chain and reject orphan state."""
    try:
        root_parent = _fanin_authoritative_directory_stat(
            root.parent, "publication fence parent directory",
        )
    except FileNotFoundError:
        # No task-lock directory means no fence has ever been published.
        return None
    parent_identity = _fanin_directory_identity(root_parent)
    path = root
    path_was_observed = False
    reachable: set[Path] = set()
    snapshots: list[tuple[Path, os.stat_result]] = []
    records: list[tuple[Path, tuple[FanInTask, str, str]]] = []
    expected_path_identity: tuple[int, int] | None = None
    for _depth in range(_FANIN_GUARD_CHAIN_MAX_GENERATIONS):
        _verify_fanin_directory_identity(
            root.parent, parent_identity, "publication fence parent directory",
        )
        try:
            observed, owner_contents, record, acceptance_contents, acceptance = _read_fanin_guard_directory(path)
        except FileNotFoundError as exc:
            if not path_was_observed:
                _reject_unreachable_fanin_generations(root, reachable)
                _verify_fanin_directory_identity(
                    root.parent, parent_identity, "publication fence parent directory",
                )
                return None
            raise FanInInventoryValidationError(f"fan-in publication fence chain is broken: {path}") from exc
        observed_identity = _fanin_directory_identity(observed)
        if expected_path_identity is not None and observed_identity != expected_path_identity:
            raise FanInInventoryValidationError(
                f"fan-in publication fence successor identity changed during traversal: {path}"
            )
        if owner_contents is None or record is None or acceptance_contents is not None:
            raise FanInInventoryValidationError(f"fan-in publication fence is malformed: {path}")
        snapshots.append((path, observed))
        records.append((path, record[:3]))
        generation = _FanInFenceGeneration(
            path, observed_identity, record, parent_identity=parent_identity,
        )
        outcome = _fanin_fence_successor_path(root, generation)
        try:
            _outcome_stat, successor_contents, successor, acceptance_contents, acceptance = (
                _read_fanin_guard_directory(outcome)
            )
        except FileNotFoundError:
            _reject_unreachable_fanin_generations(root, reachable)
            _verify_fanin_fence_generation_snapshots(snapshots)
            _verify_fanin_directory_identity(
                root.parent, parent_identity, "publication fence parent directory",
            )
            return _FanInFenceGeneration(
                generation.path, generation.identity, generation.record,
                parent_identity=parent_identity,
                chain=tuple(
                    (snapshot_path, _fanin_directory_identity(snapshot))
                    for snapshot_path, snapshot in snapshots
                ),
                chain_records=tuple(records),
            )
        reachable.add(outcome)
        if (
            (successor_contents is None and acceptance_contents is None)
            or (successor_contents is not None and acceptance_contents is not None)
            or (successor_contents is not None and successor is None)
            or (acceptance_contents is not None and acceptance is None)
        ):
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
            _verify_fanin_fence_generation_snapshots([*snapshots, (outcome, _outcome_stat)])
            _verify_fanin_directory_identity(
                root.parent, parent_identity, "publication fence parent directory",
            )
            contents, acceptance_record, acceptance_parent = _read_fanin_file_with_parent_snapshot(
                _fanin_acceptance_path(outcome), "publication fence acceptance record",
                expected_parent=_fanin_directory_identity(_outcome_stat),
            )
            if (
                contents is None
                or acceptance_record is None
                or acceptance_parent != _fanin_directory_identity(_outcome_stat)
            ):
                raise FanInInventoryValidationError(
                    f"fan-in publication acceptance vanished during traversal: {outcome}"
                )
            try:
                payload = json.loads(contents.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise FanInInventoryValidationError(
                    f"fan-in publication acceptance became malformed: {outcome}"
                ) from exc
            if _fanin_acceptance_from_payload(payload) != acceptance:
                raise FanInInventoryValidationError(
                    f"fan-in publication acceptance changed during traversal: {outcome}"
                )
            chain = tuple(
                (snapshot_path, _fanin_directory_identity(snapshot))
                for snapshot_path, snapshot in [*snapshots, (outcome, _outcome_stat)]
            )
            return _FanInFenceGeneration(
                generation.path, generation.identity, generation.record, acceptance,
                parent_identity, chain, _fanin_acceptance_path(outcome),
                _fanin_directory_identity(_outcome_stat),
                _fanin_stat_snapshot(acceptance_record),
                tuple(records),
            )
        path = outcome
        path_was_observed = True
        expected_path_identity = _fanin_directory_identity(_outcome_stat)
    raise FanInInventoryValidationError(
        f"fan-in publication fence chain exceeds {_FANIN_GUARD_CHAIN_MAX_GENERATIONS} generations: {root}"
    )


def _fanin_generation_is_active(generation: _FanInFenceGeneration, claimed: Path) -> bool:
    _verify_fanin_fence_generation_identity(generation)
    if generation.acceptance is not None:
        return False
    _task, claim_name, claim_token, renewed_at = generation.record
    if _fanin_fence_release_marker_is_present(
        _fanin_fence_release_marker(generation.path, claim_token)
    ):
        return False
    return (
        time.time() - renewed_at <= _FANIN_PRODUCER_LEASE_SECONDS
        or _claim_token_is_live(claimed / claim_name, claim_token)
    )


def _fanin_fence_is_current(fence: _FanInFilesystemFence) -> bool:
    with fence.owner_lock:
        return _fanin_fence_is_current_locked(fence)


def _fanin_fence_is_current_locked(fence: _FanInFilesystemFence) -> bool:
    """Read a local fence while its caller holds its owner renewal lock."""
    try:
        observed, owner_contents, record, acceptance_contents, _acceptance = _read_fanin_guard_directory(
            fence.path
        )
        released = _fanin_fence_release_marker_is_present(
            _fanin_fence_release_marker(fence.path, fence.claim_token)
        )
    except (OSError, FanInInventoryValidationError):
        return False
    if not (
        (observed.st_dev, observed.st_ino) == fence.identity
        and owner_contents is not None
        and acceptance_contents is None
        and not released
        and record is not None
        and record[0] == fence.task
        and record[1] == fence.claim_name
        and record[2] == fence.claim_token
    ):
        return False
    generation = _FanInFenceGeneration(fence.path, fence.identity, record)
    try:
        _fanin_authoritative_directory_stat(
            _fanin_fence_successor_path(fence.root, generation),
            "publication fence outcome directory",
        )
    except FileNotFoundError:
        return True
    except (OSError, FanInInventoryValidationError):
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
        _write_fanin_authoritative_json(
            acceptance_path, _fanin_acceptance_payload(acceptance),
            "publication fence acceptance record",
        )
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
    observed, owner_contents, record, acceptance_contents, _acceptance = _read_fanin_guard_directory(path)
    expected = (fanin_task, task.claim_path.name, task.claim_token)
    if (
        owner_contents is None
        or acceptance_contents is not None
        or record is None
        or record[:3] != expected
    ):
        raise FanInInventoryValidationError(f"new fan-in publication fence is malformed: {path}")
    stop = threading.Event()
    fence = _FanInFilesystemFence(
        root, path, (observed.st_dev, observed.st_ino), fanin_task, task.claim_path.name,
        task.claim_token, stop, threading.Lock(), threading.Thread(),
    )

    def renew() -> None:
        interval = min(_FANIN_HEARTBEAT_INTERVAL_SECONDS, _FANIN_PRODUCER_LEASE_SECONDS / 3)
        while not stop.wait(max(interval, 0.001)):
            try:
                # Local liveness and acceptance reads share this lock, so an
                # expected renewal cannot race their owner-file snapshot.
                with fence.owner_lock:
                    if stop.is_set() or not _fanin_fence_is_current_locked(fence):
                        return
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
            _verify_fanin_fence_generation_identity(current)
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
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise FanInInventoryValidationError("fan-in release marker creation requires O_NOFOLLOW support")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow, 0o600)
        os.close(descriptor)
    except FileExistsError:
        _fanin_fence_release_marker_is_present(marker)
    _fsync_directory(fence.path.parent)


def _read_fanin_publish_lock(base: Path) -> tuple[FanInTask, str, str] | None:
    """Return the task owning a well-formed per-worker publish lock."""
    lock = _fanin_publish_lock_path(base)
    payload = _read_fanin_authoritative_json(lock, "worker publish lock")
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
    candidate = _fanin_task_from_payload(task)
    if candidate is None:
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
        try:
            os.lstat(lock)
        except FileNotFoundError:
            lock_was_present = False
            owner = None
        else:
            lock_was_present = True
            owner = _read_fanin_publish_lock(base)
        if owner is None and lock_was_present:
            raise FanInInventoryValidationError(f"fan-in publish lock is malformed: {lock}")
        if owner is not None:
            owner_task, owner_claim_name, owner_token = owner
            if owner_token != task.claim_token and _claim_token_is_live(
                task.claim_path.parent / owner_claim_name, owner_token,
            ):
                raise _FanInTransientError(
                    f"fan-in worker shard {base.name} is actively publishing {owner_task.task_id!r}"
                )
        _write_fanin_authoritative_json(lock, payload, "worker publish lock")
        observed = os.lstat(lock)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise FanInInventoryValidationError(f"fan-in publish lock is malformed: {lock}")
        return lock, (observed.st_dev, observed.st_ino)
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
    payload = _read_fanin_authoritative_json(record, "task publication lock")
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
    task = _fanin_task_from_payload(raw_task)
    if task is None:
        return None
    return task, payload["claim_name"], payload["claim_token"]


def _write_fanin_task_lock(record: Path, task: TaskManifest, fanin_task: FanInTask) -> None:
    payload = {
        "schema_version": 1,
        "task": _fanin_task_payload(fanin_task),
        "claim_name": task.claim_path.name,
        "claim_token": task.claim_token,
    }
    _write_fanin_authoritative_json(record, payload, "task publication lock")


def _acquire_fanin_task_publication_lease(
    cache_dir: Path,
    task: TaskManifest,
    fanin_task: FanInTask,
) -> _FanInTaskPublicationLease:
    """Fence one task across every worker base until its acceptance is decided."""
    record = _fanin_task_lock_record(cache_dir, fanin_task.task_id)
    guard = _acquire_fanin_guard(_fanin_task_fence_path(cache_dir, fanin_task.task_id), task, fanin_task, nonblocking=True)
    try:
        try:
            os.lstat(record)
        except FileNotFoundError:
            record_was_present = False
            previous = None
        else:
            record_was_present = True
            previous = _read_fanin_task_lock(record)
        if previous is None and record_was_present:
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
    output = _canonical_fanin_output_path(task.out)
    candidate = FanInTask(
        task.base,
        task.iteration,
        task.offset,
        task.count,
        task.seed,
        str(output),
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
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in route provenance directory cannot be created: {directory}"
        ) from exc
    _fanin_authoritative_directory_stat(directory, "route provenance directory")
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
    try:
        output = _canonical_fanin_output_path(out)
    except FanInInventoryValidationError:
        return None
    if out != str(output):
        return None
    task = FanInTask(task_id, *numeric, str(output), policy)
    if task.iteration < 0 or task.offset < 0 or task.count <= 0 or task.seed < 0:
        return None
    return task


def _fanin_route_from_payload(payload: Any) -> tuple[Path, FanInTask] | None:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "cache_dir", "task"}:
        return None
    if payload["schema_version"] != 1 or not isinstance(payload["cache_dir"], str) or not payload["cache_dir"]:
        return None
    try:
        cache_dir = _canonical_fanin_physical_path(payload["cache_dir"], "cache route")
    except FanInInventoryValidationError:
        return None
    if payload["cache_dir"] != str(cache_dir):
        return None
    task = _fanin_task_from_payload(payload["task"])
    if task is None or Path(task.out).parent != cache_dir:
        return None
    return cache_dir, task


def _resolve_fanin_route(queue: Path, task: TaskManifest) -> _FanInRouteResolution:
    """Bind one task ID to its first fan-in root and route, or reject divergence.

    The record is created with ``link`` and never replaced. Recovery therefore
    searches the original root even when a retry's caller-supplied destination
    changes, and malformed provenance is terminal rather than guessed.
    """
    candidate = _fanin_task_from_task_manifest(task)
    cache_dir = _canonical_fanin_physical_path(candidate.out, "output route").parent
    record = _fanin_route_record(queue, candidate.task_id)
    payload = {
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "task": _fanin_task_payload(candidate),
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    descriptor, directory_identity = _open_fanin_authoritative_directory(
        record.parent, "route provenance directory",
    )
    temporary_name = f".{record.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    record_contents: bytes | None = None
    record_identity: os.stat_result | None = None
    try:
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise FanInInventoryValidationError(
                "fan-in route provenance creation requires O_NOFOLLOW support"
            )
        temporary_descriptor = os.open(
            temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow, 0o600,
            dir_fd=descriptor,
        )
        try:
            os.write(temporary_descriptor, encoded)
            os.fsync(temporary_descriptor)
            temporary_identity = _fanin_stat_identity(os.fstat(temporary_descriptor))
        finally:
            os.close(temporary_descriptor)
        try:
            os.link(
                temporary_name, record.name, src_dir_fd=descriptor, dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            record_contents, record_identity = _read_fanin_authoritative_regular_file_snapshot(
                descriptor, record.name, "route provenance record",
            )
            if record_contents != encoded:
                try:
                    existing_payload = json.loads((record_contents or b"").decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    existing_payload = None
                existing = _fanin_route_from_payload(existing_payload)
                if existing is None:
                    raise FanInInventoryValidationError(
                        f"fan-in route provenance is malformed: {record}"
                    )
                established_root, established_task = existing
                if _fanin_route_conflicts(established_task, candidate) or established_root != cache_dir:
                    raise FanInRouteConflictError(
                        f"fan-in task {task.base!r} has different output, policy, or cache root route"
                    )
                if established_task != candidate:
                    raise FanInTaskConflictError(
                        f"fan-in route provenance conflicts with {task.base!r}"
                    )
                raise FanInInventoryValidationError(
                    f"fan-in route provenance has non-canonical record content: {record}"
                )
        else:
            # The CAS record initially shares the temporary inode. Drop that
            # link before recording ctime so the retained snapshot is stable.
            os.unlink(temporary_name, dir_fd=descriptor)
            record_contents, record_identity = _read_fanin_authoritative_regular_file_snapshot(
                descriptor, record.name, "route provenance record",
            )
            if (
                record_contents != encoded
                or record_identity is None
                or _fanin_stat_identity(record_identity) != temporary_identity
            ):
                raise FanInInventoryValidationError(
                    f"fan-in route provenance CAS did not create its expected record: {record}"
                )
            os.fsync(descriptor)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(descriptor)
    _verify_fanin_authoritative_directory_identity(
        record.parent, directory_identity, "route provenance directory",
    )
    if record_contents is None or record_identity is None:
        raise FanInInventoryValidationError(f"fan-in route provenance is malformed: {record}")
    descriptor, observed_directory = _open_fanin_authoritative_directory(
        record.parent, "route provenance directory",
    )
    try:
        if _fanin_stat_identity(observed_directory) != _fanin_stat_identity(directory_identity):
            raise FanInInventoryValidationError(
                f"fan-in route provenance directory identity changed during validation: {record.parent}"
            )
        _verify_fanin_authoritative_regular_file_identity(
            descriptor, record.name, record_identity, "route provenance record",
        )
    finally:
        os.close(descriptor)
    _verify_fanin_authoritative_directory_identity(
        record.parent, directory_identity, "route provenance directory",
    )
    try:
        existing_payload = json.loads(record_contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        existing_payload = None
    existing = _fanin_route_from_payload(existing_payload)
    if existing is None:
        raise FanInInventoryValidationError(f"fan-in route provenance is malformed: {record}")
    established_root, established_task = existing
    if established_task != candidate or established_root != cache_dir:
        if _fanin_route_conflicts(established_task, candidate) or established_root != cache_dir:
            raise FanInRouteConflictError(
                f"fan-in task {task.base!r} has different output, policy, or cache root route"
            )
        raise FanInTaskConflictError(f"fan-in route provenance conflicts with {task.base!r}")
    # Parsing used a retained byte string. Recheck both names after parsing so
    # a replacement in the final parse-to-return window cannot become a route.
    final_contents, final_identity, final_parent = _read_fanin_file_with_parent_snapshot(
        record, "route provenance record", expected_parent=_fanin_directory_identity(directory_identity),
    )
    if (
        final_contents is None
        or final_identity is None
        or _fanin_stat_snapshot(final_identity) != _fanin_stat_snapshot(record_identity)
    ):
        raise FanInInventoryValidationError(
            f"fan-in route provenance record identity changed during validation: {record}"
        )
    _verify_fanin_directory_identity(
        record.parent, final_parent, "route provenance directory",
    )
    try:
        root_identity: tuple[int, int] | None = _fanin_directory_identity(
            _fanin_authoritative_directory_stat(established_root, "fan-in cache root")
        )
    except FileNotFoundError:
        # Direct route-resolution callers may bind provenance before a worker
        # preflight creates the cache root. Publication requires it later.
        root_identity = None
    return _FanInRouteResolution(
        root=established_root,
        root_identity=root_identity,
        record=record,
        record_parent_identity=final_parent,
        record_identity=_fanin_stat_snapshot(final_identity),
        record_sha256=hashlib.sha256(final_contents).hexdigest(),
        task=established_task,
    )


def _verify_fanin_route_resolution(
    resolution: _FanInRouteResolution,
    task: TaskManifest | FanInTask,
) -> None:
    """Keep one route record generation live through every commit side effect."""
    candidate = task if isinstance(task, FanInTask) else _fanin_task_from_task_manifest(task)
    if candidate != resolution.task:
        raise FanInInventoryValidationError(
            f"fan-in route provenance conflicts with {candidate.task_id!r}"
        )
    contents, observed, parent_identity = _read_fanin_file_with_parent_snapshot(
        resolution.record,
        "route provenance record",
        expected_parent=resolution.record_parent_identity,
    )
    if (
        contents is None
        or observed is None
        or parent_identity != resolution.record_parent_identity
        or _fanin_stat_snapshot(observed) != resolution.record_identity
        or hashlib.sha256(contents).hexdigest() != resolution.record_sha256
    ):
        raise FanInInventoryValidationError(
            f"fan-in route provenance record identity changed during validation: {resolution.record}"
        )
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FanInInventoryValidationError(
            f"fan-in route provenance became malformed: {resolution.record}"
        ) from exc
    if _fanin_route_from_payload(payload) != (resolution.root, resolution.task):
        raise FanInInventoryValidationError(
            f"fan-in route provenance changed during validation: {resolution.record}"
        )
    _verify_fanin_directory_identity(
        resolution.record.parent, resolution.record_parent_identity, "route provenance directory",
    )
    if resolution.root_identity is not None:
        _verify_fanin_directory_identity(
            resolution.root, resolution.root_identity, "fan-in cache root",
        )


def _fanin_route_resolution_payload(resolution: _FanInRouteResolution) -> dict[str, Any]:
    """Serialize the exact append-only route record generation in an acceptance."""
    return {
        "root": str(resolution.root),
        "root_identity": None if resolution.root_identity is None else list(resolution.root_identity),
        "record": str(resolution.record),
        "record_parent_identity": list(resolution.record_parent_identity),
        "record_identity": list(resolution.record_identity),
        "record_sha256": resolution.record_sha256,
    }


def _fanin_identity_tuple(value: Any, size: int) -> tuple[int, ...] | None:
    if not isinstance(value, list) or len(value) != size or not all(_is_int(item) for item in value):
        return None
    return tuple(value)


def _fanin_route_resolution_from_payload(
    payload: Any,
    task: FanInTask | None,
) -> _FanInRouteResolution | None:
    """Parse a durable route witness without resolving it again by pathname."""
    expected = {
        "root", "root_identity", "record", "record_parent_identity", "record_identity",
        "record_sha256",
    }
    if (
        task is None
        or not isinstance(payload, dict)
        or set(payload) != expected
        or not isinstance(payload["root"], str)
        or not isinstance(payload["record"], str)
        or not _is_sha256(payload["record_sha256"])
    ):
        return None
    try:
        root = _canonical_fanin_physical_path(payload["root"], "accepted cache route")
        record = _canonical_fanin_physical_path(payload["record"], "accepted route record")
    except FanInInventoryValidationError:
        return None
    root_identity = payload["root_identity"]
    if root_identity is not None:
        root_identity = _fanin_identity_tuple(root_identity, 2)
        if root_identity is None:
            return None
    else:
        # A terminal acceptance must carry the cache-root generation too.
        return None
    record_parent_identity = _fanin_identity_tuple(payload["record_parent_identity"], 2)
    record_identity = _fanin_identity_tuple(payload["record_identity"], 6)
    if record_parent_identity is None or record_identity is None:
        return None
    return _FanInRouteResolution(
        root=root,
        root_identity=root_identity,
        record=record,
        record_parent_identity=record_parent_identity,
        record_identity=record_identity,
        record_sha256=payload["record_sha256"],
        task=task,
    )


def _fanin_tasks_from_manifest_contents(contents: bytes, path: Path) -> tuple[FanInTask, ...]:
    """Parse one already-validated immutable manifest generation."""
    try:
        payload = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
        try:
            output = _canonical_fanin_output_path(out)
        except FanInInventoryValidationError as exc:
            raise FanInValidationError(
                f"fan-in shard {path} has a non-canonical output route"
            ) from exc
        if out != str(output):
            raise FanInValidationError(f"fan-in shard {path} has a non-canonical output route")
        if task_id in seen:
            raise FanInValidationError(f"fan-in shard {path} repeats task id {task_id!r}")
        seen.add(task_id)
        tasks.append(FanInTask(task_id, iteration, offset, count, seed, str(output), policy))
    return tuple(tasks)


def _read_fanin_manifest_snapshot(
    path: Path,
) -> tuple[tuple[int, int, int, int, int, int], tuple[int, int, int, int, int, int], tuple[FanInTask, ...]]:
    """Read a manifest and retain both its shard and record generations."""
    root = _fanin_real_directory_stat(path)
    manifest_path = path / FANIN_MANIFEST_NAME
    try:
        contents, manifest = _read_fanin_authoritative_file_snapshot(
            manifest_path, "shard task manifest",
        )
    except FileNotFoundError as exc:
        try:
            _fanin_real_directory_stat(path)
        except _SelectedFanInVersionVanishedError:
            raise
        raise FanInValidationError(f"fan-in shard {path} has no valid {FANIN_MANIFEST_NAME}") from exc
    if contents is None or manifest is None:
        raise FanInValidationError(f"fan-in shard {path} has no valid {FANIN_MANIFEST_NAME}")
    tasks = _fanin_tasks_from_manifest_contents(contents, path)
    _verify_fanin_root_unchanged(path, root)
    return _fanin_stat_snapshot(root), _fanin_stat_snapshot(manifest), tasks


def _read_fanin_manifest(path: Path) -> tuple[FanInTask, ...]:
    return _read_fanin_manifest_snapshot(path)[2]


def _verify_selected_fanin_shard(selected: _SelectedFanInShard) -> None:
    """Reject a selected shard if any accepted-input handoff was replaced later."""
    _verify_fanin_directory_identity(
        selected.path.parent, selected.cache_root_identity, "fan-in cache root",
    )
    root = _fanin_real_directory_stat(selected.path)
    if _fanin_stat_snapshot(root) != selected.root_snapshot:
        raise FanInInventoryValidationError(
            f"selected fan-in shard root identity changed after selection: {selected.path}"
        )
    try:
        _contents, manifest = _read_fanin_authoritative_file_snapshot(
            selected.path / FANIN_MANIFEST_NAME, "shard task manifest",
        )
    except FileNotFoundError as exc:
        raise FanInInventoryValidationError(
            f"selected fan-in shard manifest vanished after selection: {selected.path}"
        ) from exc
    if manifest is None or _fanin_stat_snapshot(manifest) != selected.manifest_snapshot:
        raise FanInInventoryValidationError(
            f"selected fan-in shard manifest identity changed after selection: {selected.path}"
        )
    _verify_fanin_root_unchanged(selected.path, root)
    _contents, final_manifest = _read_fanin_authoritative_file_snapshot(
        selected.path / FANIN_MANIFEST_NAME, "shard task manifest",
    )
    if final_manifest is None or _fanin_stat_snapshot(final_manifest) != selected.manifest_snapshot:
        raise FanInInventoryValidationError(
            f"selected fan-in shard manifest identity changed after selection: {selected.path}"
        )
    _verify_fanin_root_unchanged(selected.path, root)
    _contents, publication, publication_parent = _read_fanin_file_with_parent_snapshot(
        selected.path / FANIN_PUBLICATION_NAME, "shard publication provenance",
        expected_parent=_fanin_directory_identity(root),
    )
    if (
        publication is None
        or publication_parent != _fanin_directory_identity(root)
        or _fanin_stat_snapshot(publication) != selected.publication_snapshot
    ):
        raise FanInInventoryValidationError(
            f"selected fan-in shard publication identity changed after selection: {selected.path}"
        )
    _verify_fanin_payload_files(selected.path, selected.payload_files)
    if not selected.acceptances or selected.acceptances[-1].acceptance is None:
        raise FanInInventoryValidationError(
            f"selected fan-in shard has no terminal acceptance: {selected.path}"
        )
    terminal = selected.acceptances[-1].acceptance
    if selected.payload_files != terminal.payload_files:
        raise FanInInventoryValidationError(
            f"selected fan-in shard payload witness disagrees with terminal acceptance: {selected.path}"
        )
    for acceptance in selected.acceptances:
        _verify_fanin_fence_generation_identity(acceptance)
        if acceptance.acceptance is None:
            raise FanInInventoryValidationError(
                f"selected fan-in shard has incomplete acceptance evidence: {selected.path}"
            )
        _verify_fanin_route_resolution(acceptance.acceptance.route, acceptance.acceptance.task)
    _verify_fanin_directory_identity(
        selected.path.parent, selected.cache_root_identity, "fan-in cache root",
    )


def _verify_selected_fanin_task_route(
    selected: _SelectedFanInShard,
    candidate: FanInTask,
    route: _FanInRouteResolution | None,
) -> None:
    """Bind acknowledgement to the exact route witness that accepted this task."""
    matches = [
        generation.acceptance
        for generation in selected.acceptances
        if generation.acceptance is not None and generation.acceptance.task == candidate
    ]
    if len(matches) != 1:
        raise FanInInventoryValidationError(
            f"selected fan-in shard has ambiguous route acceptance for {candidate.task_id!r}"
        )
    accepted = matches[0]
    _verify_fanin_route_resolution(accepted.route, candidate)
    if route is not None and accepted.route != route:
        raise FanInInventoryValidationError(
            f"fan-in route witness changed after acceptance for {candidate.task_id!r}"
        )


def _cache_record_count(path: Path) -> int:
    root = _fanin_real_directory_stat(path)
    try:
        contents = _read_fanin_authoritative_file(path / "metadata.json", "shard cache metadata")
    except FileNotFoundError as exc:
        try:
            _fanin_real_directory_stat(path)
        except _SelectedFanInVersionVanishedError:
            raise
        raise FanInValidationError(f"fan-in shard {path} has no valid cache metadata") from exc
    if contents is None:
        raise FanInValidationError(f"fan-in shard {path} has no valid cache metadata")
    try:
        metadata = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FanInValidationError(f"fan-in shard {path} has no valid cache metadata") from exc
    record_count = metadata.get("record_count") if isinstance(metadata, dict) else None
    if not _is_int(record_count) or record_count < 0:
        raise FanInValidationError(f"fan-in shard {path} has invalid cache record_count")
    _verify_fanin_root_unchanged(path, root)
    return record_count


def _read_authoritative_fanin_acceptance(
    cache_dir: Path,
    task: FanInTask,
) -> _FanInFenceGeneration | None:
    root = _fanin_task_fence_path(cache_dir, task.task_id)
    current = _read_current_fanin_fence(root)
    if current is None or current.acceptance is None:
        return None
    _verify_fanin_fence_generation_identity(current)
    if current.acceptance.task.task_id != task.task_id:
        raise FanInInventoryValidationError(
            f"fan-in acceptance guard conflicts with task id {task.task_id!r}"
        )
    return current


def _fanin_candidate_acceptance_evidence(
    path: Path,
    tasks: Sequence[FanInTask],
) -> tuple[tuple[int, int, int, int, int, int], tuple[_FanInFenceGeneration, ...]] | None:
    """Return the retained acceptance chain for one authenticated version."""
    cache_root = _fanin_authoritative_directory_stat(path.parent, "fan-in cache root")
    cache_root_identity = _fanin_directory_identity(cache_root)
    shard_root = _fanin_real_directory_stat(path)
    shard_root_identity = _fanin_directory_identity(shard_root)
    lineage, separator, version_text = path.name.rpartition("-v")
    if not separator or not version_text.isascii() or not version_text.isdecimal():
        raise FanInInventoryValidationError(f"fan-in shard has invalid version name: {path}")
    version = int(version_text)
    publication, publication_snapshot = _read_fanin_publication_snapshot(path)
    if publication is None or publication_snapshot is None:
        raise FanInInventoryValidationError(
            f"fan-in shard {path} has missing or malformed publication provenance"
        )
    acceptances: list[_FanInFenceGeneration] = []
    for task_index, task in enumerate(tasks):
        acceptance = _read_authoritative_fanin_acceptance(path.parent, task)
        if acceptance is None:
            if task_index == len(tasks) - 1 and publication.task == task:
                current = _read_current_fanin_fence(
                    _fanin_task_fence_path(path.parent, task.task_id)
                )
                if current is not None and current.path.name != publication.guard_generation:
                    _validate_fanin_target_acceptance(path, tasks, publication)
                    _verify_fanin_directory_identity(path.parent, cache_root_identity, "fan-in cache root")
                    return None
            raise FanInInventoryValidationError(
                f"fan-in shard {path} has no immutable acceptance for {task.task_id!r}"
            )
        if acceptance.acceptance.task != task:
            raise FanInInventoryValidationError(
                f"fan-in shard {path} has conflicting metadata for {task.task_id!r}"
            )
        if acceptance.acceptance.lineage != lineage:
            _validate_fanin_target_acceptance(path, tasks, publication)
            _verify_fanin_directory_identity(path.parent, cache_root_identity, "fan-in cache root")
            return None
        if (
            acceptance.acceptance.task_index != task_index
            or acceptance.acceptance.version > version
            or acceptance.acceptance.prefix_sha256 != _fanin_manifest_prefix_sha256(tasks, task_index)
        ):
            raise FanInInventoryValidationError(
                f"fan-in shard {path} has ambiguous lineage provenance for {task.task_id!r}"
            )
        acceptances.append(acceptance)
    final = acceptances[-1].acceptance
    assert final is not None
    if final.target != path.name or final.version != version:
        _validate_fanin_target_acceptance(path, tasks, publication)
        _verify_fanin_directory_identity(path.parent, cache_root_identity, "fan-in cache root")
        return None
    _validate_fanin_target_acceptance(path, tasks, final)
    for acceptance in acceptances:
        _verify_fanin_fence_generation_identity(acceptance)
    _contents, final_publication, publication_parent = _read_fanin_file_with_parent_snapshot(
        path / FANIN_PUBLICATION_NAME, "shard publication provenance",
        expected_parent=shard_root_identity,
    )
    if (
        final_publication is None
        or publication_parent != shard_root_identity
        or _fanin_stat_snapshot(final_publication) != publication_snapshot
    ):
        raise FanInInventoryValidationError(
            f"fan-in shard publication provenance identity changed after acceptance: {path}"
        )
    _verify_fanin_directory_identity(path.parent, cache_root_identity, "fan-in cache root")
    _verify_fanin_root_unchanged(path, shard_root)
    return publication_snapshot, tuple(acceptances)


def _fanin_candidate_is_accepted(path: Path, tasks: Sequence[FanInTask]) -> bool:
    """Whether one cumulative version is fully authenticated and trainable."""
    return _fanin_candidate_acceptance_evidence(path, tasks) is not None


def _selected_fanin_shard_key(
    selected: _SelectedFanInShard,
) -> tuple[
    Path,
    tuple[int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[tuple[Path, tuple[int, int]], ...],
    tuple[_FanInPayloadFile, ...],
]:
    return (
        selected.path,
        selected.cache_root_identity,
        selected.root_snapshot,
        selected.manifest_snapshot,
        selected.publication_snapshot,
        tuple(
            (acceptance.path, acceptance.identity)
            for acceptance in selected.acceptances
        ),
        selected.payload_files,
    )


def _select_accepted_fanin_shards(cache_dir: Path) -> list[_SelectedFanInShard]:
    cache_root = _fanin_authoritative_directory_stat(cache_dir, "fan-in cache root")
    cache_root_identity = _fanin_directory_identity(cache_root)
    lineages: set[str] = set()
    for candidate in Path(cache_dir).glob("shard-w*-v*"):
        lineage, separator, suffix = candidate.name.rpartition("-v")
        if separator and suffix.isascii() and suffix.isdecimal():
            lineages.add(lineage)
    selected: list[_SelectedFanInShard] = []
    for lineage in sorted(lineages):
        base = Path(cache_dir) / lineage
        for _version, path in reversed(_shard_versions(base)):
            root_snapshot, manifest_snapshot, tasks = _read_fanin_manifest_snapshot(path)
            if _cache_record_count(path) != sum(task.count for task in tasks):
                raise FanInInventoryValidationError(
                    f"fan-in shard {path} cache games disagree with its task manifest"
                )
            acceptance_evidence = _fanin_candidate_acceptance_evidence(path, tasks)
            if acceptance_evidence is not None:
                publication_snapshot, acceptances = acceptance_evidence
                _verify_fanin_directory_identity(
                    cache_dir, cache_root_identity, "fan-in cache root",
                )
                payload_files = _snapshot_fanin_payload_files(path)
                final_acceptance = acceptances[-1].acceptance
                assert final_acceptance is not None
                _validate_fanin_target_acceptance(path, tasks, final_acceptance)
                shard = _SelectedFanInShard(
                    path, cache_root_identity, root_snapshot, manifest_snapshot,
                    publication_snapshot, acceptances, tasks, payload_files,
                )
                _verify_selected_fanin_shard(shard)
                selected.append(shard)
                break
    _verify_fanin_directory_identity(cache_dir, cache_root_identity, "fan-in cache root")
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
            selected_shards = _select_accepted_fanin_shards(cache_dir)
            for selected in selected_shards:
                _verify_selected_fanin_shard(selected)
                path = selected.path
                name, _, version_text = path.name.rpartition("-v")
                tasks = selected.tasks
                if _cache_record_count(path) != sum(task.count for task in tasks):
                    raise FanInValidationError(f"fan-in shard {path} cache games disagree with its task manifest")
                _verify_selected_fanin_shard(selected)
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
            stable_shards = _select_accepted_fanin_shards(cache_dir)
        except _SelectedFanInVersionVanishedError as exc:
            vanished = exc
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        if [_selected_fanin_shard_key(item) for item in stable_shards] != [
            _selected_fanin_shard_key(item) for item in selected_shards
        ]:
            time.sleep(_SELECTED_FANIN_RETRY_SECONDS)
            continue
        return tuple(shards), by_task_id
    raise FanInValidationError(
        f"selected fan-in versions did not stabilize after {_SELECTED_FANIN_READ_ATTEMPTS} attempts"
    ) from vanished


def _find_committed_fanin_task_evidence(
    cache_dir: Path,
    candidate: FanInTask,
) -> tuple[FanInTask, _SelectedFanInShard] | None:
    """Find one task across selected versions without demanding global quiescence.

    A worker needs only one answer: whether *its* task was already accepted.
    It still validates every selected version it observes and rejects duplicate
    or conflicting entries for that task, but intentionally does not require a
    second identical all-worker snapshot. The trainer-facing strict reader is
    the place that requires that stronger quiescent invariant.
    """
    vanished: _SelectedFanInVersionVanishedError | None = None
    for _attempt in range(_SELECTED_FANIN_READ_ATTEMPTS):
        matches: list[tuple[FanInTask, _SelectedFanInShard]] = []
        try:
            selected_shards = _select_accepted_fanin_shards(cache_dir)
            for selected in selected_shards:
                _verify_selected_fanin_shard(selected)
                path = selected.path
                tasks = selected.tasks
                if _cache_record_count(path) != sum(task.count for task in tasks):
                    raise FanInInventoryValidationError(
                        f"fan-in shard {path} cache games disagree with its task manifest"
                    )
                _verify_selected_fanin_shard(selected)
                for entry in tasks:
                    if entry.task_id == candidate.task_id:
                        matches.append((entry, selected))
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
        for selected in selected_shards:
            _verify_selected_fanin_shard(selected)
        if not matches:
            return None
        if len(matches) != 1:
            if any(entry != matches[0][0] for entry, _selected in matches[1:]):
                raise FanInInventoryValidationError(
                    f"fan-in inventory has conflicting metadata for {candidate.task_id!r}"
                )
            raise FanInInventoryValidationError(
                f"fan-in inventory repeats task id {candidate.task_id!r}"
            )
        _verify_selected_fanin_shard(matches[0][1])
        return matches[0]
    raise _FanInTransientError(
        f"selected fan-in version for task {candidate.task_id!r} kept vanishing"
    ) from vanished


def _find_committed_fanin_task(cache_dir: Path, candidate: FanInTask) -> FanInTask | None:
    """Compatibility view of an accepted task without exposing internal evidence."""
    result = _find_committed_fanin_task_evidence(cache_dir, candidate)
    return None if result is None else result[0]


def _read_current_worker_shard(
    base: Path,
    expected_iteration: int,
) -> tuple[_SelectedFanInShard | None, int, tuple[FanInTask, ...]]:
    """Read only this worker's cumulative shard before publishing its next version."""
    cache_root = _fanin_authoritative_directory_stat(base.parent, "fan-in cache root")
    cache_root_identity = _fanin_directory_identity(cache_root)
    versions = _shard_versions(base)
    version = versions[-1][0] if versions else 0
    current: _SelectedFanInShard | None = None
    current_tasks: tuple[FanInTask, ...] = ()
    try:
        for _candidate_version, candidate in reversed(versions):
            root_snapshot, manifest_snapshot, tasks = _read_fanin_manifest_snapshot(candidate)
            if _cache_record_count(candidate) != sum(task.count for task in tasks):
                raise FanInInventoryValidationError(
                    f"fan-in shard {candidate} cache games disagree with its task manifest"
                )
            acceptance_evidence = _fanin_candidate_acceptance_evidence(candidate, tasks)
            if acceptance_evidence is not None:
                publication_snapshot, acceptances = acceptance_evidence
                payload_files = _snapshot_fanin_payload_files(candidate)
                final_acceptance = acceptances[-1].acceptance
                assert final_acceptance is not None
                _validate_fanin_target_acceptance(candidate, tasks, final_acceptance)
                selected = _SelectedFanInShard(
                    candidate, cache_root_identity, root_snapshot, manifest_snapshot,
                    publication_snapshot, acceptances, tasks, payload_files,
                )
                _verify_selected_fanin_shard(selected)
                current = selected
                current_tasks = tasks
                break
        if current is not None and any(task.iteration != expected_iteration for task in current_tasks):
            raise FanInInventoryValidationError(f"fan-in shard {current.path} has a task from the wrong iteration")
        _verify_fanin_directory_identity(base.parent, cache_root_identity, "fan-in cache root")
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
        try:
            output = _canonical_fanin_output_path(task.out)
        except FanInInventoryValidationError as exc:
            raise FanInValidationError(
                "fan-in task manifest requires canonical absolute physical output routes"
            ) from exc
        if task.out != str(output):
            raise FanInValidationError(
                "fan-in task manifest requires canonical absolute physical output routes"
            )
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
    contents = _read_fanin_authoritative_file(path, "shard provenance file")
    if contents is None:
        raise FanInValidationError(f"fan-in shard provenance file is missing: {path}")
    digest.update(contents)
    return digest.hexdigest()


def _fanin_manifest_prefix_sha256(tasks: Sequence[FanInTask], task_index: int) -> str:
    payload = [_fanin_task_payload(task) for task in tasks[:task_index + 1]]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fanin_stat_identity(observed: os.stat_result) -> tuple[int, int, int]:
    return observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)


def _fanin_stat_snapshot(observed: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _fanin_real_directory_stat(path: Path) -> os.stat_result:
    """Return an lstat-validated shard root, never following a replacement link."""
    try:
        observed = os.lstat(path)
    except FileNotFoundError as exc:
        raise _SelectedFanInVersionVanishedError(f"selected fan-in shard vanished: {path}") from exc
    except OSError as exc:
        raise FanInValidationError(f"fan-in shard root is unreadable: {path}") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise FanInValidationError(f"fan-in shard root is not a real directory: {path}")
    return observed


def _verify_fanin_root_unchanged(path: Path, expected: os.stat_result) -> None:
    observed = _fanin_real_directory_stat(path)
    if _fanin_stat_identity(observed) != _fanin_stat_identity(expected):
        raise FanInValidationError(f"fan-in shard root identity changed during validation: {path}")
    if _fanin_stat_snapshot(observed) != _fanin_stat_snapshot(expected):
        raise FanInValidationError(f"fan-in shard root changed during validation: {path}")


def _fanin_hash_entry(
    digest: Any,
    relative: str,
    kind: bytes,
    observed: os.stat_result,
) -> None:
    try:
        encoded = relative.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FanInValidationError(f"fan-in shard has non-UTF-8 cache entry {relative!r}") from exc
    if observed.st_size < 0 or observed.st_size >= 2 ** 64:
        raise FanInValidationError(f"fan-in shard has invalid cache entry size for {relative!r}")
    digest.update(kind)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(observed.st_size.to_bytes(8, "big"))


def _fanin_no_follow_open_flags(*, directory: bool) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise FanInValidationError("fan-in content validation requires O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is None:
            raise FanInValidationError("fan-in content validation requires O_DIRECTORY support")
        flags |= directory_flag
    return flags


def _open_fanin_entry(
    name: str | Path,
    expected: os.stat_result,
    *,
    directory: bool,
    dir_fd: int | None = None,
) -> int:
    try:
        descriptor = os.open(name, _fanin_no_follow_open_flags(directory=directory), dir_fd=dir_fd)
    except FileNotFoundError as exc:
        raise _SelectedFanInVersionVanishedError(f"selected fan-in shard entry vanished: {name}") from exc
    except OSError as exc:
        raise FanInValidationError(f"fan-in shard entry is unreadable: {name}") from exc
    observed = os.fstat(descriptor)
    if (
        _fanin_stat_snapshot(observed) != _fanin_stat_snapshot(expected)
        or stat.S_ISDIR(observed.st_mode) != directory
    ):
        os.close(descriptor)
        raise FanInValidationError(f"fan-in shard entry changed during validation: {name}")
    return descriptor


def _verify_fanin_entry_unchanged(
    name: str,
    expected: os.stat_result,
    *,
    dir_fd: int,
) -> None:
    try:
        observed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise _SelectedFanInVersionVanishedError(f"selected fan-in shard entry vanished: {name}") from exc
    except OSError as exc:
        raise FanInValidationError(f"fan-in shard entry is unreadable: {name}") from exc
    if _fanin_stat_snapshot(observed) != _fanin_stat_snapshot(expected):
        raise FanInValidationError(f"fan-in shard entry changed during validation: {name}")


def _snapshot_fanin_payload_files(path: Path) -> tuple[_FanInPayloadFile, ...]:
    """Capture every regular file a pathname-based cache consumer can reopen."""
    root = _fanin_real_directory_stat(path)
    descriptor = _open_fanin_entry(path, root, directory=True)
    files: list[_FanInPayloadFile] = []

    def snapshot_directory(
        directory: int,
        expected: os.stat_result,
        relative_parent: tuple[str, ...],
    ) -> None:
        try:
            with os.scandir(os.dup(directory)) as entries:
                children = sorted(
                    ((entry.name, entry.stat(follow_symlinks=False)) for entry in entries),
                    key=lambda item: item[0],
                )
        except OSError as exc:
            raise FanInValidationError("fan-in shard directory is unreadable during payload snapshot") from exc
        for name, observed in children:
            relative = (*relative_parent, name)
            if stat.S_ISLNK(observed.st_mode):
                raise FanInValidationError(f"fan-in shard has symlinked cache entry {'/'.join(relative)!r}")
            if stat.S_ISDIR(observed.st_mode):
                child = _open_fanin_entry(name, observed, directory=True, dir_fd=directory)
                try:
                    snapshot_directory(child, observed, relative)
                finally:
                    os.close(child)
                _verify_fanin_entry_unchanged(name, observed, dir_fd=directory)
                continue
            if not stat.S_ISREG(observed.st_mode):
                raise FanInValidationError(f"fan-in shard has unsupported cache entry {'/'.join(relative)!r}")
            # The acceptance is written after this snapshot. It is protocol
            # metadata, not a cache payload a concat/training consumer opens.
            if relative == (FANIN_PUBLICATION_NAME,):
                _verify_fanin_entry_unchanged(name, observed, dir_fd=directory)
                continue
            file_descriptor = _open_fanin_entry(name, observed, directory=False, dir_fd=directory)
            digest = hashlib.sha256()
            try:
                while chunk := os.read(file_descriptor, 1024 * 1024):
                    digest.update(chunk)
                if _fanin_stat_snapshot(os.fstat(file_descriptor)) != _fanin_stat_snapshot(observed):
                    raise FanInValidationError(
                        f"fan-in shard entry changed during payload snapshot: {'/'.join(relative)}"
                    )
            finally:
                os.close(file_descriptor)
            _verify_fanin_entry_unchanged(name, observed, dir_fd=directory)
            files.append(_FanInPayloadFile(relative, _fanin_stat_snapshot(observed), digest.hexdigest()))
        if _fanin_stat_snapshot(os.fstat(directory)) != _fanin_stat_snapshot(expected):
            raise FanInValidationError("fan-in shard directory changed during payload snapshot")

    try:
        snapshot_directory(descriptor, root, ())
    finally:
        os.close(descriptor)
    _verify_fanin_root_unchanged(path, root)
    return tuple(files)


def _verify_fanin_payload_files(
    path: Path,
    expected: tuple[_FanInPayloadFile, ...],
) -> None:
    """Reject inode or in-place byte changes before a cache path is consumed."""
    if _snapshot_fanin_payload_files(path) != expected:
        raise FanInInventoryValidationError(
            f"selected fan-in shard payload changed after selection: {path}"
        )


def _fanin_payload_files_from_payload(value: Any) -> tuple[_FanInPayloadFile, ...] | None:
    """Parse one acceptance's exact regular-file payload generations."""
    if not isinstance(value, list) or not value:
        return None
    files: list[_FanInPayloadFile] = []
    previous: tuple[str, ...] | None = None
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"relative", "identity", "sha256"}:
            return None
        relative = entry["relative"]
        identity = _fanin_identity_tuple(entry["identity"], 6)
        if (
            not isinstance(relative, list)
            or not relative
            or any(
                not isinstance(part, str) or not part or part in {".", ".."} or "/" in part
                for part in relative
            )
            or identity is None
            or not _is_sha256(entry["sha256"])
        ):
            return None
        parsed = _FanInPayloadFile(tuple(relative), identity, entry["sha256"])
        if previous is not None and parsed.relative <= previous:
            return None
        previous = parsed.relative
        files.append(parsed)
    return tuple(files)


def _fanin_hash_directory(
    digest: Any,
    descriptor: int,
    expected: os.stat_result,
    relative_parent: str,
) -> None:
    try:
        with os.scandir(descriptor) as entries:
            children = sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in entries),
                key=lambda item: item[0],
            )
    except OSError as exc:
        raise FanInValidationError("fan-in shard directory is unreadable during validation") from exc
    for name, observed in children:
        relative = f"{relative_parent}/{name}" if relative_parent else name
        if stat.S_ISLNK(observed.st_mode):
            raise FanInValidationError(f"fan-in shard has symlinked cache entry {relative!r}")
        if stat.S_ISDIR(observed.st_mode):
            _fanin_hash_entry(digest, relative, b"D", observed)
            child = _open_fanin_entry(name, observed, directory=True, dir_fd=descriptor)
            try:
                _fanin_hash_directory(digest, child, observed, relative)
            finally:
                os.close(child)
            _verify_fanin_entry_unchanged(name, observed, dir_fd=descriptor)
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise FanInValidationError(f"fan-in shard has unsupported cache entry {relative!r}")
        if relative == FANIN_PUBLICATION_NAME:
            _verify_fanin_entry_unchanged(name, observed, dir_fd=descriptor)
            continue
        _fanin_hash_entry(digest, relative, b"F", observed)
        file_descriptor = _open_fanin_entry(name, observed, directory=False, dir_fd=descriptor)
        try:
            while chunk := os.read(file_descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(file_descriptor)
        _verify_fanin_entry_unchanged(name, observed, dir_fd=descriptor)
    if _fanin_stat_snapshot(os.fstat(descriptor)) != _fanin_stat_snapshot(expected):
        raise FanInValidationError("fan-in shard directory changed during validation")


def _fanin_content_sha256(path: Path) -> str:
    """Hash an immutable real-directory tree without following symlinked paths."""
    root = _fanin_real_directory_stat(path)
    digest = hashlib.sha256()
    digest.update(b"pokezero-fanin-content-v2\0")
    descriptor = _open_fanin_entry(path, root, directory=True)
    try:
        try:
            _fanin_hash_directory(digest, descriptor, root, "")
        except Exception:
            # Check the root even on an earlier entry failure, but keep that
            # more specific failure as the reason this validation was rejected.
            try:
                _verify_fanin_root_unchanged(path, root)
            except FanInValidationError:
                pass
            raise
    finally:
        os.close(descriptor)
    _verify_fanin_root_unchanged(path, root)
    return digest.hexdigest()


def _build_fanin_acceptance(
    path: Path,
    target: Path,
    lineage: Path,
    version: int,
    tasks: Sequence[FanInTask],
    task_index: int,
    fence: _FanInFilesystemFence,
    route: _FanInRouteResolution,
) -> _FanInAcceptance:
    _verify_fanin_route_resolution(route, tasks[task_index])
    if route.root_identity is None or route.root != target.parent or path.parent != target.parent:
        raise FanInInventoryValidationError(
            f"fan-in acceptance route disagrees with target parent: {target}"
        )
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
        payload_files=_snapshot_fanin_payload_files(path),
        route=route,
    )


def _write_fanin_publication(path: Path, acceptance: _FanInAcceptance) -> None:
    _write_fanin_authoritative_json(
        path / FANIN_PUBLICATION_NAME, _fanin_acceptance_payload(acceptance),
        "shard publication provenance",
    )


def _validate_fanin_staging_acceptance(
    path: Path,
    tasks: Sequence[FanInTask],
    acceptance: _FanInAcceptance,
) -> None:
    """Recheck an acceptance proof while its payload still resides in staging."""
    root = _fanin_real_directory_stat(path)
    try:
        if acceptance.route.root != path.parent:
            raise FanInInventoryValidationError(
                f"fan-in shard {path} route disagrees with its cache root"
            )
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
            acceptance.task_index != len(tasks) - 1
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
        _verify_fanin_root_unchanged(path, root)
        _verify_fanin_payload_files(path, acceptance.payload_files)
        _verify_fanin_route_resolution(acceptance.route, acceptance.task)
    except Exception:
        # An entry-level rejection remains more actionable than the parent
        # directory timestamp it also changes, while both checks still run.
        try:
            _verify_fanin_root_unchanged(path, root)
        except FanInValidationError:
            pass
        raise
    _verify_fanin_root_unchanged(path, root)


def _validate_fanin_target_acceptance(
    path: Path,
    tasks: Sequence[FanInTask],
    acceptance: _FanInAcceptance,
) -> None:
    if acceptance.target != path.name:
        raise FanInInventoryValidationError(
            f"fan-in shard {path} content or lineage disagrees with its acceptance"
        )
    _validate_fanin_staging_acceptance(path, tasks, acceptance)


def _accept_fanin_publication(
    root: Path,
    publication: _FanInAcceptance,
    *,
    fence: _FanInFilesystemFence | None = None,
) -> _FanInAcceptance:
    """Race terminal acceptance against successor installation at one atomic slot."""
    if fence is not None:
        if fence.root != root:
            raise FanInInventoryValidationError(
                f"fan-in publication fence root disagrees with acceptance: {root}"
            )
        with fence.owner_lock:
            return _accept_fanin_publication(root, publication)
    _verify_fanin_route_resolution(publication.route, publication.task)
    current = _read_current_fanin_fence(root)
    if current is None:
        raise FanInInventoryValidationError(
            f"fan-in publication acceptance has no guard root: {root}"
        )
    _verify_fanin_fence_generation_identity(current)
    if current.acceptance is not None:
        _verify_fanin_route_resolution(current.acceptance.route, current.acceptance.task)
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
    _verify_fanin_fence_generation_identity(resolved)
    _verify_fanin_route_resolution(resolved.acceptance.route, resolved.acceptance.task)
    return resolved.acceptance


def _fsync_directory(path: Path) -> None:
    """Persist a manifest replacement or version rename before acknowledging it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preflight_fanin_location(path: Path, location: str) -> None:
    """Probe one mount for the local atomic operations fan-in relies on."""
    path.mkdir(parents=True, exist_ok=True)
    _fanin_authoritative_directory_stat(path, f"{location} filesystem probe directory")
    token = f".fanin-filesystem-probe.{os.getpid()}.{uuid.uuid4().hex}"
    source_a = path / f"{token}.source-a"
    source_b = path / f"{token}.source-b"
    collision = path / f"{token}.collision"
    directory = path / f"{token}.directory"
    moved = path / f"{token}.moved"
    try:
        for source, contents in ((source_a, b"a\n"), (source_b, b"b\n")):
            with source.open("xb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())

        gate = threading.Barrier(3)
        results: list[tuple[Path, OSError | None]] = []
        results_lock = threading.Lock()

        def race_link(source: Path) -> None:
            try:
                gate.wait(timeout=1.0)
                os.link(source, collision)
            except OSError as exc:
                outcome: OSError | None = exc
            except threading.BrokenBarrierError:
                outcome = OSError(errno.ETIMEDOUT, "hard-link probe did not start")
            else:
                outcome = None
            with results_lock:
                results.append((source, outcome))

        threads = [threading.Thread(target=race_link, args=(source,)) for source in (source_a, source_b)]
        for thread in threads:
            thread.start()
        try:
            gate.wait(timeout=1.0)
        except threading.BrokenBarrierError as exc:
            raise FanInInventoryValidationError(
                f"fan-in {location} filesystem hard-link probe did not start: {path}"
            ) from exc
        for thread in threads:
            thread.join(timeout=1.0)
        if any(thread.is_alive() for thread in threads):
            raise FanInInventoryValidationError(
                f"fan-in {location} filesystem hard-link probe did not finish: {path}"
            )
        winners = [source for source, outcome in results if outcome is None]
        collisions = [
            outcome for _source, outcome in results
            if outcome is not None and outcome.errno == errno.EEXIST
        ]
        if len(results) != 2 or len(winners) != 1 or len(collisions) != 1:
            raise FanInInventoryValidationError(
                f"fan-in {location} filesystem lacks atomic hard-link CAS/collision support: {path}"
            )
        observed = os.lstat(collision)
        winner = os.lstat(winners[0])
        if (
            stat.S_ISLNK(observed.st_mode)
            or _fanin_stat_identity(observed) != _fanin_stat_identity(winner)
        ):
            raise FanInInventoryValidationError(
                f"fan-in {location} filesystem has invalid hard-link visibility: {path}"
            )

        directory.mkdir()
        os.rename(directory, moved)
        if not stat.S_ISDIR(os.lstat(moved).st_mode):
            raise FanInInventoryValidationError(
                f"fan-in {location} filesystem lost a same-directory rename: {path}"
            )
        directory.mkdir()
    except FanInInventoryValidationError:
        raise
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in {location} filesystem lacks required hard-link, mkdir, or same-directory rename support: {path}: {exc}"
        ) from exc
    finally:
        for candidate in (collision, source_a, source_b):
            candidate.unlink(missing_ok=True)
        shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(moved, ignore_errors=True)


def _preflight_fanin_route_directory(queue: Path) -> None:
    """Probe the exact directory that will host the route-record link CAS."""
    directory = queue / _FANIN_ROUTE_DIRECTORY
    created = False
    try:
        directory.mkdir()
        created = True
    except FileExistsError:
        pass
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in route provenance directory cannot be created: {directory}"
        ) from exc
    try:
        _preflight_fanin_location(directory, "route-CAS")
    except BaseException:
        if created:
            try:
                directory.rmdir()
            except OSError as exc:
                raise FanInInventoryValidationError(
                    f"fan-in route-CAS probe cleanup failed: {directory}"
                ) from exc
        raise
    if created:
        try:
            directory.rmdir()
            _fsync_directory(queue)
        except OSError as exc:
            raise FanInInventoryValidationError(
                f"fan-in route-CAS probe cleanup failed: {directory}"
            ) from exc


def _preflight_fanin_filesystems(queue: Path, cache_dir: Path) -> None:
    """Verify queue and cache mounts before a fan-in route CAS or queue claim.

    Fan-in requires POSIX-visible, atomic ``link`` collision, ``mkdir``, and
    same-directory ``rename`` behavior at both locations. This local race only
    establishes one-node behavior; deployment validation must additionally
    contend across all worker nodes sharing each mount.
    """
    _preflight_fanin_location(queue, "queue")
    _preflight_fanin_location(cache_dir, "cache")
    _preflight_fanin_route_directory(queue)


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


def _write_done_marker(task: TaskManifest, done: Path) -> _DoneMarkerWrite:
    """Create a canonical done marker without overwriting a concurrent marker.

    A completed fan-in shard is already accepted input, so this fallback also
    handles a reaper that removed the original claim between version
    publication and acknowledgement.
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
                return _DoneMarkerWrite(False)
            # ``temporary`` is still our open hard link, so this snapshot is
            # the installed inode even if the done pathname is raced later.
            identity = _fanin_done_marker_identity(os.fstat(handle.fileno()))
        _fsync_directory(done.parent)
        return _DoneMarkerWrite(True, identity)
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


def _fanin_done_marker_identity(observed: os.stat_result) -> tuple[int, int, int, int, int]:
    """Identify a done marker despite the expected temporary-link ctime change."""
    snapshot = _fanin_stat_snapshot(observed)
    return snapshot[:-1]


def _fanin_done_marker_snapshot(path: Path) -> tuple[int, int, int, int, int]:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise FanInInventoryValidationError(
            f"fan-in done marker vanished after acknowledgement: {path}"
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise FanInInventoryValidationError(f"fan-in done marker is not a regular file: {path}")
    return _fanin_done_marker_identity(observed)


def _remove_fanin_done_marker_generation(
    path: Path,
    expected: tuple[int, int, int, int, int],
) -> None:
    """Undo only the marker this acknowledgement wrote after its proof failed."""
    try:
        if _fanin_done_marker_snapshot(path) != expected:
            return
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _complete_claim(
    task: TaskManifest,
    queue: Path,
    *,
    crash_inject: Callable[[str, TaskManifest], None] | None,
    accepted_shard: _SelectedFanInShard | None = None,
    route: _FanInRouteResolution | None = None,
) -> None:
    """Publish a canonical done marker after an accepted fan-in task.

    Immutable acceptance is fan-in's input commit point. Therefore revocation
    after acceptance cannot discard output: if the old claim vanished, recreate
    a matching done marker from the parsed task metadata instead.
    """
    candidate = _fanin_task_from_task_manifest(task) if accepted_shard is not None else None

    def verify_completion_proof() -> None:
        if route is not None:
            _verify_fanin_route_resolution(route, task)
        if accepted_shard is not None:
            _verify_selected_fanin_shard(accepted_shard)
            assert candidate is not None
            _verify_selected_fanin_task_route(accepted_shard, candidate, route)

    verify_completion_proof()
    done = queue / "done" / task.base
    write = _DoneMarkerWrite(False)
    owns_claim = (
        _fanin_claim_manifest_is_current(task)
        if task.claim_identity is not None else _claim_token_is_current(task)
    )
    # Keep the pre-write proof adjacent to the durable acknowledgement.
    verify_completion_proof()
    if not owns_claim:
        write = _write_done_marker(task, done)
    elif task.claim_identity is not None:
        # Never hard-link a mutable pathname into done/. The parsed manifest
        # bound into the lease is the only acknowledgement payload we accept.
        write = _write_done_marker(task, done)
    else:
        try:
            os.link(task.claim_path, done)
        except FileExistsError:
            if not _done_marker_matches_task(done, task):
                raise FanInTaskConflictError(f"done marker conflicts with accepted task {task.base}")
        except FileNotFoundError:
            write = _write_done_marker(task, done)
        except OSError as exc:
            raise FanInInventoryValidationError(
                f"could not acknowledge accepted fan-in task {task.base}: {exc}"
            ) from exc
        else:
            write = _DoneMarkerWrite(True, _fanin_done_marker_snapshot(done))
            _fsync_directory(done.parent)
    done_identity = write.identity if write else None
    try:
        # A mutation in the narrow write-to-acknowledgement interval must not
        # leave our success marker behind. The generation snapshot prevents a
        # failed worker from unlinking a concurrent repair's replacement.
        verify_completion_proof()
    except Exception:
        if done_identity is not None:
            _remove_fanin_done_marker_generation(done, done_identity)
        raise
    if write:
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
    route: _FanInRouteResolution,
) -> _FanInAcceptance | None:
    """Finish target-before-acceptance recovery without recollecting the task."""
    if cache_dir != route.root:
        raise FanInInventoryValidationError(
            f"fan-in recovery cache root disagrees with route provenance: {cache_dir}"
        )
    _verify_fanin_route_resolution(route, candidate)
    cache_root = _fanin_authoritative_directory_stat(cache_dir, "fan-in cache root")
    cache_root_identity = _fanin_directory_identity(cache_root)
    root = _fanin_task_fence_path(cache_dir, candidate.task_id)
    current = _read_current_fanin_fence(root)
    if current is None:
        return None
    _verify_fanin_fence_generation_identity(current)
    if current.acceptance is not None:
        if current.acceptance.task != candidate:
            if _fanin_route_conflicts(current.acceptance.task, candidate):
                raise FanInRouteConflictError(
                    f"accepted fan-in task {candidate.task_id!r} has different output or policy route"
                )
            raise FanInTaskConflictError(
                f"accepted fan-in task {candidate.task_id!r} conflicts with retried metadata"
            )
        _verify_fanin_route_resolution(current.acceptance.route, candidate)
        if current.acceptance.route != route:
            raise FanInInventoryValidationError(
                f"fan-in route witness changed after acceptance for {candidate.task_id!r}"
            )
        return current.acceptance
    pending: list[
        tuple[Path, tuple[FanInTask, ...], _FanInAcceptance, tuple[_FanInPayloadFile, ...]]
    ] = []
    for target in Path(cache_dir).glob("shard-w*-v*"):
        _verify_fanin_route_resolution(route, candidate)
        _verify_fanin_directory_identity(cache_dir, cache_root_identity, "fan-in cache root")
        publication, publication_snapshot = _read_fanin_publication_snapshot(target)
        if publication is None and publication_snapshot is None:
            continue
        if publication is None or publication_snapshot is None:
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
        _verify_fanin_route_resolution(publication.route, candidate)
        if publication.route != route:
            raise FanInInventoryValidationError(
                f"fan-in route witness changed after target publication for {candidate.task_id!r}"
            )
        if (
            publication.guard_root != root.name
            or publication.guard_generation != current.path.name
        ):
            continue
        target_snapshot, manifest_snapshot, tasks = _read_fanin_manifest_snapshot(target)
        _validate_fanin_target_acceptance(target, tasks, publication)
        _verify_fanin_directory_identity(cache_dir, cache_root_identity, "fan-in cache root")
        current_target = _fanin_real_directory_stat(target)
        if _fanin_stat_snapshot(current_target) != target_snapshot:
            raise FanInInventoryValidationError(
                f"fan-in recovery target identity changed during validation: {target}"
            )
        _contents, current_manifest = _read_fanin_authoritative_file_snapshot(
            target / FANIN_MANIFEST_NAME, "shard task manifest",
        )
        if current_manifest is None or _fanin_stat_snapshot(current_manifest) != manifest_snapshot:
            raise FanInInventoryValidationError(
                f"fan-in recovery manifest identity changed during validation: {target}"
            )
        _contents, current_publication, publication_parent = _read_fanin_file_with_parent_snapshot(
            target / FANIN_PUBLICATION_NAME, "shard publication provenance",
            expected_parent=_fanin_directory_identity(current_target),
        )
        if (
            current_publication is None
            or publication_parent != _fanin_directory_identity(current_target)
            or _fanin_stat_snapshot(current_publication) != publication_snapshot
        ):
            raise FanInInventoryValidationError(
                f"fan-in recovery publication identity changed during validation: {target}"
            )
        payload_files = _snapshot_fanin_payload_files(target)
        _validate_fanin_target_acceptance(target, tasks, publication)
        pending.append((target, tasks, publication, payload_files))
    if not pending:
        return None
    if len(pending) != 1:
        names = ", ".join(sorted(target.name for target, _tasks, _publication, _payload in pending))
        raise FanInInventoryValidationError(
            f"fan-in task {candidate.task_id!r} has ambiguous pending targets: {names}"
        )
    _verify_fanin_fence_generation_identity(current)
    _verify_fanin_directory_identity(cache_dir, cache_root_identity, "fan-in cache root")
    target, _tasks, publication, payload_files = pending[0]
    _verify_fanin_payload_files(target, payload_files)
    _verify_fanin_route_resolution(route, candidate)
    return _accept_fanin_publication(root, publication)


def _recover_fanin_task(
    task: TaskManifest,
    queue: Path,
    *,
    route: _FanInRouteResolution | None = None,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> bool:
    """Finalize a claim whose task is already durably selected, without collecting."""
    candidate = _fanin_task_from_task_manifest(task)
    route = route or _resolve_fanin_route(queue, task)
    _verify_fanin_route_resolution(route, candidate)
    cache_dir = route.root
    _recover_pending_fanin_publication(cache_dir, candidate, route)
    committed = _find_committed_fanin_task_evidence(cache_dir, candidate)
    if committed is None:
        return False
    existing, accepted_shard = committed
    if existing != candidate:
        if _fanin_route_conflicts(existing, candidate):
            raise FanInRouteConflictError(
                f"accepted fan-in task {task.base!r} has different output or policy route"
            )
        raise FanInTaskConflictError(f"committed task {task.base!r} conflicts with its retried manifest")
    try:
        _complete_claim(
            task, queue, crash_inject=crash_inject, accepted_shard=accepted_shard, route=route,
        )
    except FanInTaskConflictError as exc:
        raise FanInInventoryValidationError(
            f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
        ) from exc
    _inject_crash(crash_inject, "fanin-after-recovery", task)
    _verify_fanin_route_resolution(route, candidate)
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
    route: _FanInRouteResolution | None = None,
    crash_inject: Callable[[str, TaskManifest], None] | None,
) -> tuple[Path | None, bool]:
    """Atomically publish one manifest-bearing version, fenced across workers."""
    if task.claim_identity is not None and not _fanin_claim_manifest_is_current(task):
        raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
    route = route or _resolve_fanin_route(queue, task)
    fanin_task = _fanin_task_from_task_manifest(task)
    _verify_fanin_route_resolution(route, fanin_task)
    cache_dir = route.root
    if _temporary_cache_record_count(temporary_cache, task) != fanin_task.count:
        raise FanInTaskValidationError(
            f"task {task.base} cache games do not match queue count {fanin_task.count}"
        )
    _recover_pending_fanin_publication(cache_dir, fanin_task, route)
    committed = _find_committed_fanin_task_evidence(cache_dir, fanin_task)
    if committed is not None:
        existing, accepted_shard = committed
        if existing != fanin_task:
            if _fanin_route_conflicts(existing, fanin_task):
                raise FanInRouteConflictError(
                    f"accepted fan-in task {task.base!r} has different output or policy route"
                )
            raise FanInTaskConflictError(f"committed task {task.base!r} conflicts with its retried manifest")
        try:
            _complete_claim(
                task, queue, crash_inject=crash_inject, accepted_shard=accepted_shard, route=route,
            )
        except FanInTaskConflictError as exc:
            raise FanInInventoryValidationError(
                f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
            ) from exc
        shutil.rmtree(temporary_cache, ignore_errors=True)
        _inject_crash(crash_inject, "fanin-after-recovery", task)
        _verify_fanin_route_resolution(route, fanin_task)
        return None, True
    if not _fanin_claim_manifest_is_current(task):
        raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
    _reject_prepublication_done_marker(task, queue)

    _verify_fanin_route_resolution(route, fanin_task)
    task_lease = _acquire_fanin_task_publication_lease(cache_dir, task, fanin_task)
    try:
        # A stalled predecessor holds this lease. Once it exits or is fenced by
        # its claim token, a retry rechecks selected input before any append.
        committed = _find_committed_fanin_task_evidence(cache_dir, fanin_task)
        if committed is not None:
            existing, accepted_shard = committed
            if existing != fanin_task:
                if _fanin_route_conflicts(existing, fanin_task):
                    raise FanInRouteConflictError(
                        f"accepted fan-in task {task.base!r} has different output or policy route"
                    )
                raise FanInInventoryValidationError(
                    f"committed task {task.base!r} conflicts with its retried manifest"
                )
            _complete_claim(
                task, queue, crash_inject=crash_inject, accepted_shard=accepted_shard, route=route,
            )
            shutil.rmtree(temporary_cache, ignore_errors=True)
            _inject_crash(crash_inject, "fanin-after-recovery", task)
            _verify_fanin_route_resolution(route, fanin_task)
            return None, True
        if not _fanin_claim_manifest_is_current(task):
            raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
        _reject_prepublication_done_marker(task, queue)
        base = cache_dir / f"shard-w{_sanitize_worker_id(worker)}"
        _verify_fanin_route_resolution(route, fanin_task)
        lock, lock_identity = _acquire_fanin_publish_lock(base, task, fanin_task)
        staging: Path | None = None
        owner_sidecar: Path | None = None
        published = False
        try:
            current, version, current_tasks = _read_current_worker_shard(base, task.iteration)
            if _recover_current_worker_task(current_tasks, fanin_task):
                _complete_claim(
                    task, queue, crash_inject=crash_inject, accepted_shard=current, route=route,
                )
                shutil.rmtree(temporary_cache, ignore_errors=True)
                _inject_crash(crash_inject, "fanin-after-recovery", task)
                _verify_fanin_route_resolution(route, fanin_task)
                return None, True
            target = base.parent / f"{base.name}-v{version + 1}"
            staging = base.parent / f".{target.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
            _verify_fanin_route_resolution(route, fanin_task)
            owner_sidecar = _write_fanin_staging_owner(
                staging, fanin_task, producer_token=task.claim_token,
            )
            _refresh_fanin_staging_lease(staging, task.claim_token)
            if current is None:
                os.rename(temporary_cache, staging)
            else:
                from .dataset import concat_training_caches

                # ``concat_training_caches`` reopens its inputs by pathname.
                # Keep the selected shard's full acceptance chain live on both
                # sides so a swapped cache tree cannot become a new target.
                _verify_selected_fanin_shard(current)
                concat_training_caches((current.path, temporary_cache), staging)
                _verify_selected_fanin_shard(current)
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
            _verify_fanin_route_resolution(route, fanin_task)
            _write_fanin_manifest(staging, version_tasks)
            acceptance = _build_fanin_acceptance(
                staging, target, base, version + 1, version_tasks,
                len(version_tasks) - 1, task_lease.guard, route,
            )
            _write_fanin_publication(staging, acceptance)
            _inject_crash(crash_inject, "fanin-before-target-publication", task)
            if not _fanin_claim_manifest_is_current(task) or not _fanin_fence_is_current(task_lease.guard):
                raise _ClaimRevokedError(f"claim was revoked before fan-in publication for {task.base}")
            if _read_fanin_staging_owner_record(staging) != (fanin_task, task.claim_token):
                raise _FanInTransientError(f"fan-in staging ownership changed before publishing {task.base}")
            _inject_crash(crash_inject, "fanin-after-publication-fence-check", task)
            _verify_fanin_route_resolution(route, fanin_task)
            if acceptance.route != route:
                raise FanInInventoryValidationError(
                    f"fan-in acceptance route witness changed before publishing {task.base}"
                )
            # The proof was captured before writing the publication record;
            # re-open every payload generation after the final fence check and
            # immediately before moving the staging tree into public view.
            _validate_fanin_staging_acceptance(staging, version_tasks, acceptance)
            try:
                os.rename(staging, target)
            except OSError as exc:
                if not _is_fanin_target_collision(exc):
                    raise
                committed = _find_committed_fanin_task_evidence(cache_dir, fanin_task)
                existing = None if committed is None else committed[0]
                if existing == fanin_task:
                    _complete_claim(
                        task, queue, crash_inject=crash_inject,
                        accepted_shard=committed[1], route=route,
                    )
                    shutil.rmtree(staging, ignore_errors=True)
                    owner_sidecar.unlink(missing_ok=True)
                    shutil.rmtree(temporary_cache, ignore_errors=True)
                    _verify_fanin_route_resolution(route, fanin_task)
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
            _verify_fanin_route_resolution(route, fanin_task)
            if acceptance.route != route:
                raise FanInInventoryValidationError(
                    f"fan-in acceptance route witness changed before accepting {task.base}"
                )
            _validate_fanin_target_acceptance(target, version_tasks, acceptance)
            accepted = _accept_fanin_publication(
                task_lease.guard.root, acceptance, fence=task_lease.guard,
            )
            if accepted != acceptance:
                if accepted.task != fanin_task:
                    raise FanInInventoryValidationError(
                        f"fan-in acceptance conflicts with published task {task.base!r}"
                    )
                committed = _find_committed_fanin_task_evidence(cache_dir, fanin_task)
                if committed is None or committed[0] != fanin_task:
                    raise FanInInventoryValidationError(
                        f"fan-in accepted task disappeared before acknowledgement: {task.base}"
                    )
                _complete_claim(
                    task, queue, crash_inject=crash_inject, accepted_shard=committed[1], route=route,
                )
                owner_sidecar.unlink(missing_ok=True)
                _remove_fanin_staging_lease(staging, task.claim_token)
                _fsync_directory(owner_sidecar.parent)
                _verify_fanin_route_resolution(route, fanin_task)
                return None, True
            try:
                committed = _find_committed_fanin_task_evidence(cache_dir, fanin_task)
                if committed is None or committed[0] != fanin_task:
                    raise FanInInventoryValidationError(
                        f"fan-in accepted task disappeared before acknowledgement: {task.base}"
                    )
                _complete_claim(
                    task, queue, crash_inject=crash_inject, accepted_shard=committed[1], route=route,
                )
            except FanInTaskConflictError as exc:
                raise FanInInventoryValidationError(
                    f"accepted fan-in task {task.base} conflicts with queue acknowledgement"
                ) from exc
            _inject_crash(crash_inject, "fanin-before-stale-cleanup", task)
            _verify_fanin_route_resolution(route, fanin_task)
            for stale_version, stale in _shard_versions(base):
                if stale_version < version + 1:
                    _remove_stale_fanin_version(stale)
            owner_sidecar.unlink(missing_ok=True)
            _remove_fanin_staging_lease(staging, task.claim_token)
            _fsync_directory(owner_sidecar.parent)
            _verify_fanin_route_resolution(route, fanin_task)
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

    def preflight_fanin_claim(_candidate: Path, preview: TaskManifest | None) -> None:
        if preview is None:
            # A malformed manifest has no cache route to probe, but its queue
            # claim must still not precede validation of the queue mount.
            _preflight_fanin_location(queue, "queue")
            return
        _preflight_fanin_filesystems(queue, preview.out.parent)

    while True:
        try:
            task = claim_next_task(
                queue,
                worker,
                before_claim=preflight_fanin_claim if shard_fanin else None,
                fanin=shard_fanin,
            )
        except (FanInInventoryValidationError, OSError) as exc:
            log(f"TERMINAL fan-in filesystem preflight failed before claim or route binding: {exc}")
            return finish(2)
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
        route: _FanInRouteResolution | None = None
        if shard_fanin:
            try:
                try:
                    route = _resolve_fanin_route(queue, task)
                    _verify_fanin_route_resolution(route, task)
                    cache_dir = route.root
                except FanInTaskValidationError:
                    # Preserve the established per-task failure classification;
                    # recovery below will reject malformed queue metadata.
                    cache_dir = task.out.parent
                _sweep_abandoned_fanin_staging(cache_dir, queue)
            except (FanInInventoryValidationError, OSError) as exc:
                log(f"TERMINAL fan-in selected inventory corruption for {task.base}; preserving claim: {exc}")
                return finish(2)
            try:
                if _recover_fanin_task(task, queue, route=route, crash_inject=crash_inject):
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
        try:
            owns_claim = (
                _fanin_claim_manifest_is_current(task)
                if shard_fanin else task.claim_path.exists()
            )
        except FanInInventoryValidationError as exc:
            log(f"TERMINAL fan-in claim manifest changed for {task.base}; preserving claim: {exc}")
            return finish(2)
        if not owns_claim:
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
                    task, tmp, queue, worker, route=route, crash_inject=crash_inject,
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
