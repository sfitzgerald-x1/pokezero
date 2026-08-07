"""Run foundation training inside the pod that already reserves the serving GPU.

WHY THIS EXISTS. A foundation run used to hold TWO GPUs: a long-lived inference-server Deployment
and a per-iteration training Job. The controller loop is sequential -- collect and train never
overlap -- so at any instant one of those two GPUs is idle, and the serving one is idle for the
whole train segment, roughly two thirds of every cycle. Co-locating training in the serving pod
returns one GPU per run. It also keeps the pod alive, which matters for a second reason: that pod
is the podAffinity anchor for the packed collector fleet, so scaling it down (the other way to free
the GPU) would unpin 24 collectors.

WHAT IT MUST NOT LOSE. The per-iteration Job gave training real reliability: `backoffLimit: 2`
retried a transient failure, `activeDeadlineSeconds` bounded it, and a crash was contained -- the
Job failed, the serving Deployment did not care. Reproducing that inside a long-lived pod is the
whole design problem, and it is why training runs as a FORKED SUBPROCESS here rather than in this
process:

  * CUDA state. An OOM or a device assert poisons the CUDA context of the process it happens in.
    In fork-per-attempt the poisoned context dies with the child; this agent and the serve
    processes (separate processes, separate contexts) are untouched. Training in-process would turn
    every train crash into a pod restart -- strictly worse than the Job it replaces.
  * Deadlines. `activeDeadlineSeconds` becomes wait-with-timeout then kill the child. Nothing else
    in the pod is disturbed.
  * Memory. The child's optimizer and activation allocations are returned to the GPU when it exits,
    so the resident serve copy never competes with a stale allocator pool.

RESTARTABILITY. Nothing about the run lives only in this pod. The request, the claim and the result
are files on shared storage, and the checkpoint plus its `train-complete.json` terminal record are
written by the training command itself exactly as they were under the Job. So:

  * this agent crashing -> its supervisor restarts it; it re-adopts or releases its own claim
  * the pod being rescheduled -> the Deployment brings it back, the claim is reaped as stale, the
    controller's attempt bound decides
  * the controller restarting -> it reads `train-complete.json`, the same durable proof it already
    used to recover a Job-based iteration

The controller's completion check (`verify_train_completion`) is UNCHANGED by this file. That is
deliberate: the proof that an iteration trained correctly must not depend on which execution shape
produced it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The claim carries the identity of the process that owns it, so a restarted agent can tell its own
# abandoned claim (re-adoptable or releasable) from one held by a live sibling. pid alone is not
# enough -- pids are reused, and a reused pid on a restarted pod would let two agents believe they
# own the same iteration. The boot id changes on every pod start, which is what makes the pair
# unambiguous within the only scope that matters here.
# Resolved once, at import, rather than inside every forked child: the fork -> setsid -> dlopen ->
# prctl window is time during which the agent can die and orphan the child, and there is no reason
# to make it wider than it has to be.
try:  # pragma: no cover - Linux only
    import ctypes as _ctypes

    _LIBC = _ctypes.CDLL("libc.so.6", use_errno=True)
    _PR_SET_PDEATHSIG = 1
except Exception:  # pragma: no cover
    _LIBC = None
    _PR_SET_PDEATHSIG = 1

_CLAIM_SCHEMA = "pokezero.foundation-train-claim.v1"
_RESULT_SCHEMA = "pokezero.foundation-train-result.v1"


def _boot_identity() -> str:
    """A value that is stable for this pod and changes when it restarts."""
    for candidate in ("/proc/sys/kernel/random/boot_id", "/etc/machine-id"):
        try:
            return Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
    # No /proc (a developer machine, a test): fall back to this process's start time, which is
    # still stable for the life of the agent and distinct across restarts.
    return f"pid{os.getpid()}-{int(time.time())}"


@dataclass(frozen=True)
class TrainRequest:
    """One iteration's training work, as written by the controller."""

    path: Path
    iteration: int
    script: str
    deadline_seconds: int
    max_attempts: int

    @property
    def name(self) -> str:
        return self.path.name


def _read_request(path: Path) -> TrainRequest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema != "pokezero.foundation-train-request.v1":
        raise ValueError(f"unsupported train request schema: {schema!r}")
    for field in ("iteration", "script", "deadline_seconds"):
        if payload.get(field) in (None, ""):
            raise ValueError(f"train request {path.name!r} is missing {field}")
    return TrainRequest(
        path=path,
        iteration=int(payload["iteration"]),
        script=str(payload["script"]),
        deadline_seconds=int(payload["deadline_seconds"]),
        # Mirrors the Job's backoffLimit: 2, i.e. three attempts in total.
        max_attempts=int(payload.get("max_attempts", 3)),
    )


def _queue_dirs(queue: Path) -> tuple[Path, Path, Path, Path]:
    pending, claimed, done, failed = (
        queue / "pending", queue / "claimed", queue / "done", queue / "failed",
    )
    for directory in (pending, claimed, done, failed):
        directory.mkdir(parents=True, exist_ok=True)
    return pending, claimed, done, failed


def _write_json_atomic(target: Path, payload: dict) -> None:
    """Publish a record indivisibly.

    A half-written result is worse than none: the controller polls for these files and would read a
    truncated one as authoritative. Write to a sibling temp and rename, which is atomic within a
    directory on both local and NFS-backed storage.
    """
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def claim_next_request(queue: Path, owner: str) -> TrainRequest | None:
    """Claim the lowest-numbered pending request via atomic rename, or return None.

    Atomic rename is the whole concurrency story: if two agents ever race (a pod mid-reschedule,
    say), exactly one rename succeeds and the loser sees ENOENT and moves on. No lock file, no
    lease renewal, nothing that can be held by a process that has already died.
    """
    pending, claimed, _, failed = _queue_dirs(queue)
    try:
        candidates = sorted(pending.glob("*.json"))
    except OSError:
        return None
    for candidate in candidates:
        claim = claimed / candidate.name
        try:
            os.rename(candidate, claim)
        except OSError:
            continue  # lost the race, or it vanished; try the next one
        try:
            request = _read_request(claim)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            # Malformed: park it so the controller's own bound decides, rather than looping on it
            # forever. Same policy the collect queue applies to a malformed manifest.
            try:
                os.rename(claim, failed / f"{claim.name}.malformed")
                _write_json_atomic(
                    failed / f"{claim.stem}.json",
                    {
                        "schema_version": _RESULT_SCHEMA,
                        "status": "malformed",
                        "detail": str(error),
                    },
                )
            except OSError:
                pass
            continue
        _write_json_atomic(
            claimed / f"{claim.stem}.claim.json",
            {
                "schema_version": _CLAIM_SCHEMA,
                "owner": owner,
                "iteration": request.iteration,
                "claimed_at": int(time.time()),
            },
        )
        return request
    return None


def _claim_token_path(queue: Path, stem: str) -> Path:
    return queue / "claimed" / f"{stem}.claim.json"


def record_running_trainer(queue: Path, stem: str, pgid: int | None) -> None:
    """Note in the claim whether a trainer process is currently alive for it.

    The second layer under PR_SET_PDEATHSIG, and the one that works where prctl does not (no Linux,
    or a child that outran the prctl call). Without it, an agent restart releases its own claim
    while an orphaned trainer is still writing the iteration, and the next agent starts a second
    trainer on the same checkpoint path.
    """
    token = _claim_token_path(queue, stem)
    try:
        payload = json.loads(token.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload["trainer_pgid"] = pgid
    try:
        _write_json_atomic(token, payload)
    except OSError:
        pass


def _process_group_is_alive(pgid: int | None) -> bool:
    """Is ANY process left in the trainer's group?

    The group, not the direct child, and that distinction is the whole fix. The rendered train
    script defines a shell function before invoking python, so bash cannot exec-replace itself: the
    real tree is agent -> `bash -c` -> python trainer. Tracking the direct child tracked the
    WRAPPER. Kill the wrapper and the trainer is reparented and keeps writing the checkpoint, while
    a liveness check on the wrapper pid reports dead -- so the claim was released and a second
    trainer started. That was reproduced.

    After setsid the wrapper's pid IS the group id of the whole tree, so killpg(pgid, 0) answers
    the question that actually matters: is anything still training.
    """
    if not pgid:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _kill_group(pgid: int | None) -> None:
    """SIGKILL a whole trainer group and wait briefly for it to clear."""
    if not pgid:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        return
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _process_group_is_alive(pgid):
            return
        time.sleep(0.2)


def release_stale_claims(queue: Path, owner: str) -> list[str]:
    """Return claims this pod previously owned to pending, and report what was released.

    Only OUR OWN abandoned claims. A claim whose owner is a different boot identity belongs to a
    pod that may still be alive -- reaping it here would let two agents train the same iteration
    into the same output directory. Cross-pod staleness is the controller's call, because only it
    knows whether the other pod is gone; this is the half an agent can decide by itself.
    """
    pending, claimed, _, _ = _queue_dirs(queue)
    released: list[str] = []
    for token in sorted(claimed.glob("*.claim.json")):
        try:
            payload = json.loads(token.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("owner") != owner:
            continue
        # KILL our own orphaned trainer before releasing, rather than merely refusing.
        #
        # This claim belongs to a PREVIOUS life of this pod, so whatever is still running under it
        # is unowned: the agent that would have recorded its result is gone. Leaving it alive means
        # it writes a checkpoint nobody tracks, racing whoever picks the claim up next -- the exact
        # two-writer case. Refusing instead would be safe but strands the iteration until the
        # controller's bound expires, on a run that could restart in seconds.
        #
        # The GROUP, not the direct child: the trainer is a grandchild (bash wrapper -> python), so
        # signalling the wrapper leaves the trainer running.
        pgid = payload.get("trainer_pgid")
        if _process_group_is_alive(pgid):
            print(f"train-agent: killing orphaned trainer group {pgid} before releasing", flush=True)
            _kill_group(pgid)
            if _process_group_is_alive(pgid):
                # Could not clear it; refuse rather than create a second writer.
                print(
                    f"train-agent: trainer group {pgid} survived SIGKILL; leaving the claim in "
                    "place rather than risking a second trainer",
                    flush=True,
                )
                continue
        request_path = claimed / f"{token.name[: -len('.claim.json')]}.json"
        try:
            if request_path.exists():
                os.rename(request_path, pending / request_path.name)
                released.append(request_path.name)
            token.unlink()
        except OSError:
            continue
    return released


def _child_preexec() -> None:  # pragma: no cover - runs only in the forked child
    """Detach into a new session AND die with the agent.

    Two things, and the second exists because the first creates the hazard. `setsid` is what lets
    the deadline signal the whole tree, but it also means the child is NOT killed when the agent
    dies -- it is reparented to init and keeps training. The supervisor then respawns the agent,
    which releases its own claim, and a SECOND trainer starts on the same iteration, writing the
    same `iteration-NNNN/transformer-policy.pt` while the first is still writing it. That was
    reproduced, not theorised.

    PR_SET_PDEATHSIG closes it at the source: the kernel SIGKILLs this child the moment its parent
    thread dies, so an orphaned trainer cannot outlive the agent that owns its claim. The agent is
    single-threaded, which is what makes the parent-THREAD semantics safe here.

    Linux only, and best-effort: on any platform without prctl this is a no-op and the claim's
    liveness record (see `claim_next_request`) is what prevents the double-trainer.
    """
    os.setsid()
    if _LIBC is not None:
        try:
            _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
        except Exception:
            pass


def run_training_attempt(
    script: str,
    deadline_seconds: int,
    *,
    log: Path | None = None,
    on_start: "Callable[[int], None] | None" = None,
) -> int:
    """Run one training attempt as a child process; return its exit code.

    The child gets its own process GROUP so the deadline kill reaches the whole tree. Training
    spawns helpers (torchrun in the multi-GPU case, dataloader workers in every case); signalling
    only the direct child would leave those orphaned, holding GPU memory that the next attempt then
    fails to allocate -- turning one timeout into a run-ending cascade.
    """
    handle = None
    try:
        stream = None
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("ab")
            stream = handle
        process = subprocess.Popen(
            ["bash", "-c", script],
            stdout=stream, stderr=subprocess.STDOUT if stream else None,
            # setsid + PR_SET_PDEATHSIG together; see _child_preexec. Not start_new_session=True,
            # which gives the session without the death-linkage that makes it safe.
            preexec_fn=_child_preexec,
        )
        if on_start is not None:
            on_start(process.pid)
        try:
            return process.wait(timeout=deadline_seconds)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            return 124  # conventional timeout status, and distinct from any train exit code
    finally:
        if handle is not None:
            handle.close()


def _terminate_group(process: subprocess.Popen) -> None:
    """SIGTERM the child's group, then SIGKILL what survives.

    TERM first so torch can release the GPU on its way out; a straight KILL can leave the device in
    a state the next attempt has to wait on. The grace period is deliberately short -- this path is
    already a timeout, and the iteration's own deadline has been spent.
    """
    try:
        group = os.getpgid(process.pid)
    except OSError:
        return
    # TERM the group, wait for the DIRECT child, then KILL the group unconditionally.
    #
    # The escalation is not conditional on the child surviving, which is the bug this replaced: a
    # child that dies politely on SIGTERM used to return early, so grandchildren that ignore
    # SIGTERM were never killed at all -- they survived holding GPU memory, which is the exact
    # cascade this function exists to prevent. Killing an already-empty group is a harmless ESRCH.
    try:
        os.killpg(group, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.2)
    try:
        os.killpg(group, signal.SIGKILL)
    except OSError:
        pass
    # Reap the direct child so it cannot linger as a zombie while the next attempt starts.
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def serve_queue(
    queue: Path,
    *,
    owner: str | None = None,
    poll_seconds: float = 2.0,
    once: bool = False,
    log_dir: Path | None = None,
) -> int:
    """The agent loop: claim a request, run it with retries, publish a result, repeat.

    Returns the number of requests completed, which is what makes the loop testable with once=True.
    """
    owner = owner or _boot_identity()
    _, claimed, done, failed = _queue_dirs(queue)
    released = release_stale_claims(queue, owner)
    for name in released:
        print(f"train-agent: released own stale claim {name}", flush=True)

    completed = 0
    while True:
        request = claim_next_request(queue, owner)
        if request is None:
            if once:
                return completed
            time.sleep(poll_seconds)
            continue

        attempts: list[dict] = []
        status = "failed"
        for attempt in range(1, request.max_attempts + 1):
            started = time.time()
            log = None if log_dir is None else log_dir / f"train-{request.iteration:04d}.log"
            code = run_training_attempt(
                request.script, request.deadline_seconds, log=log,
                on_start=lambda pid: record_running_trainer(queue, request.path.stem, pid),
            )
            record_running_trainer(queue, request.path.stem, None)
            attempts.append(
                {
                    "attempt": attempt,
                    "exit_code": code,
                    "elapsed_seconds": int(time.time() - started),
                }
            )
            print(
                f"train-agent: iteration={request.iteration} attempt={attempt}"
                f"/{request.max_attempts} exit={code}",
                flush=True,
            )
            if code == 0:
                status = "succeeded"
                break

        result = {
            "schema_version": _RESULT_SCHEMA,
            "status": status,
            "iteration": request.iteration,
            "owner": owner,
            "attempts": attempts,
        }
        destination = done if status == "succeeded" else failed
        # The archived request keeps a DISTINCT name from the result. They shared one when this was
        # first written, so the archive rename silently overwrote the result the controller polls
        # for -- a completed iteration reported as never finished.
        try:
            os.rename(request.path, destination / f"{request.path.stem}.request.json")
        except OSError:
            pass
        # The RESULT is published before the claim is cleared, never after. The controller polls
        # for the result; clearing the claim first would open a window where the request looks
        # neither claimed nor finished, and a restarted agent would re-run an iteration that had
        # already succeeded -- overwriting a checkpoint the controller may already have consumed.
        _write_json_atomic(destination / f"{request.path.stem}.json", result)
        try:
            (claimed / f"{request.path.stem}.claim.json").unlink()
        except OSError:
            pass

        completed += 1
        if once:
            return completed


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="pokezero.train_agent",
        description="Run foundation training inside the serving pod, from a shared queue.",
    )
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--once", action="store_true",
        help="claim and run at most one request, then exit (testing and one-shot recovery)",
    )
    args = parser.parse_args(argv)
    serve_queue(
        args.queue, poll_seconds=args.poll_seconds, once=args.once, log_dir=args.log_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
