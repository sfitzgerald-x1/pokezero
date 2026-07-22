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
  boundary, never mid-game.

Every claim/commit/failure line is timestamped, which doubles as the
collect-queue plan's Step-0 instrumentation (task-duration histogram and the
startup-vs-compute split come straight from these logs).
"""

from __future__ import annotations

import os
import shutil
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


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


def _log(worker_id: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} fleet-worker {worker_id}: {message}", flush=True)


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
    max_rss_mb: float | None = 12000.0,
    max_tasks: int | None = None,
    idle_exit_seconds: float | None = None,
    sleep_seconds: float = 2.0,
) -> int:
    """Drain the queue forever (daemon) or until a recycle/idle bound trips.

    ``collect_fn`` receives the full per-task argv (static flags first, then the
    per-task ``--games/--seed-start/--out/--current-policy`` overrides — last
    wins under argparse, so per-task values always take precedence) and returns
    a process-style exit code.
    """
    worker = worker_id or socket.gethostname()
    _log(worker, f"persistent worker up; queue={queue} rss_limit_mb={max_rss_mb} max_tasks={max_tasks}")
    tasks_done = 0
    idle_since: float | None = None
    while True:
        task = claim_next_task(queue, worker)
        if task is None:
            now = time.monotonic()
            if idle_since is None:
                idle_since = now
            if idle_exit_seconds is not None and now - idle_since >= idle_exit_seconds:
                _log(worker, f"idle for {idle_exit_seconds:.0f}s; exiting after {tasks_done} tasks")
                return 0
            time.sleep(sleep_seconds)
            continue
        idle_since = None
        tmp = Path(f"{task.out}.tmp.{worker}")
        shutil.rmtree(tmp, ignore_errors=True)
        task.out.parent.mkdir(parents=True, exist_ok=True)
        _log(worker, f"claim {task.base} iter={task.iteration} games={task.count} seed={task.seed}")
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
            _log(worker, f"task {task.base} raised:\n{traceback.format_exc()}")
            succeeded = False
        elapsed = time.monotonic() - started
        if succeeded:
            if task.claim_path.exists():  # revocation-discard, exactly as the shell worker
                shutil.rmtree(task.out, ignore_errors=True)
                os.rename(tmp, task.out)
                os.rename(task.claim_path, queue / "done" / task.base)
                _log(worker, f"commit {task.base} elapsed={elapsed:.1f}s rss={_rss_mb():.0f}MB")
            else:
                _log(worker, f"revoked {task.base}; discarding {elapsed:.1f}s of work")
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            _log(worker, f"FAILED {task.base} elapsed={elapsed:.1f}s")
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                os.rename(task.claim_path, queue / "failed" / f"{task.base}.{worker}.failed")
            except OSError:
                pass
        tasks_done += 1
        rss = _rss_mb()
        if max_rss_mb is not None and rss > max_rss_mb:
            _log(worker, f"rss {rss:.0f}MB over {max_rss_mb:.0f}MB; recycling after {tasks_done} tasks")
            return 0
        if max_tasks is not None and tasks_done >= max_tasks:
            _log(worker, f"max_tasks {max_tasks} reached; recycling")
            return 0
