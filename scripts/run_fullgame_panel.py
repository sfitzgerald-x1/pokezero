#!/usr/bin/env python3
"""One reviewed, non-resumable panel using the existing game runner.

Collection only: prefixes are observations awaiting independent readout. A
nonzero exit is never replaced by a clean prefix's execution_integrity block.

Entries default to sequential shards. A fresh reviewed standalone entry may set
``shard_execution: parallel_seats_fixed_work`` for the exact candidate-p1/p2
raw-head-to-head pair. It requires explicit CPU engine-mcts fixed-work arguments;
timed/dynamic search and per-seat policy overrides are rejected. Both workers
remain under this supervisor's existing budget lock and absolute deadline.
Each seat publishes its own *-progress.json, and either seat's failure or
completed-game watchdog drains both. Contention can still slow wall time or
trigger watchdogs: this option does not promise a speedup, qualify an experiment,
or authorize changing an active entry, replaying cells or extending its cap.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, document):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def tree_digest(root):
    root = Path(root)
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.js")):
        h.update(str(path.relative_to(root)).encode() + b"\0")
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def verify(entry):
    for name, expected in entry["input_hashes"].items():
        if digest(name) != expected:
            raise ValueError(f"input changed: {name}")
    for name, expected in entry["tree_hashes"].items():
        if tree_digest(name) != expected:
            raise ValueError(f"compiled simulator changed: {name}")
    for root, expected in entry["repositories"].items():
        def git(*args):
            return subprocess.check_output(["git", "-C", root, *args], text=True).strip()
        if git("rev-parse", "HEAD") != expected["commit"]:
            raise ValueError(f"commit changed: {root}")
        if git("status", "--porcelain", "--untracked-files=normal") != expected["status"]:
            raise ValueError(f"checkout changed: {root}")


def observed_count(path, seed_start, planned):
    if not path.exists():
        return 0
    doc = json.loads(path.read_text())
    rows = doc["game_results"]
    count = len(rows)
    if count > planned or doc["completed_games"] != count:
        raise ValueError("unexpected game count")
    if [row["seed"] for row in rows] != list(range(seed_start, seed_start + count)):
        raise ValueError("summary is not the registered chronological prefix")
    return count


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_owned(process):
    # start_new_session makes this PID the owned process-group identity.
    if group_exists(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        code = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if group_exists(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        code = process.wait(timeout=10)
    if group_exists(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
        limit = time.time() + 10
        while group_exists(process.pid) and time.time() < limit:
            time.sleep(0.2)
    return code, not group_exists(process.pid)


def coordination_reason(entry, entry_hash, deadline, now):
    """Fail closed on an absent coordinator without changing registered work."""
    coordination = entry.get("coordination")
    if coordination is None:
        return None
    if not isinstance(coordination, dict):
        return "COORDINATION_INVALID"
    lease = coordination.get("lease_path")
    abort = coordination.get("abort_path")
    timeout = coordination.get("lease_timeout_seconds")
    if (not isinstance(lease, str) or not lease
            or not isinstance(abort, str) or not abort or lease == abort
            or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0 < timeout <= 60):
        return "COORDINATION_INVALID"
    if Path(abort).exists():
        return "COORDINATOR_ABORT"
    try:
        lease_path = Path(lease)
        # Renewal is an atomic replacement on this worker's filesystem. The
        # receiver's own mtime, not a caller-supplied future timestamp, is the
        # liveness clock. A lease can never renew the scientific deadline.
        with lease_path.open() as stream:
            document = json.load(stream)
            age = now - os.fstat(stream.fileno()).st_mtime
        if (not isinstance(document, dict)
                or document.get("entry_sha256") != entry_hash
                or type(document.get("deadline_unix")) not in (int, float)
                or document["deadline_unix"] != deadline or age < -1):
            return "COORDINATION_INVALID"
        return "COORDINATOR_LOST" if age > timeout else None
    except FileNotFoundError:
        return "COORDINATOR_LOST"
    except (OSError, ValueError, TypeError):
        return "COORDINATION_INVALID"


def validate_execution(entry):
    """Parallelism is entry-bound and limited to fixed-work CPU/raw seat pairs.

    Use an exact option allowlist, not a search for forbidden spellings:
    argparse abbreviations and per-shard overrides must not enable timed work.
    The ordinary sequential path retains its existing argument contract.
    """
    mode = entry.get("shard_execution", "sequential")
    if mode == "sequential":
        return mode
    if mode != "parallel_seats_fixed_work":
        raise ValueError("unknown shard_execution")
    shards = entry["shards"]
    if (entry.get("comparison_kind") != "raw_head_to_head"
            or entry.get("coordination") is not None
            or [(s["name"], s["seat"]) for s in shards]
            != [("candidate-p1", "p1"), ("candidate-p2", "p2")]):
        raise ValueError("parallel seats require a standalone raw head-to-head pair")
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False, exit_on_error=False)
    for name, value in (("policy-mode", "engine-mcts"), ("opponent-policy-mode", "raw"),
                        ("device", "cpu")):
        parser.add_argument("--" + name, required=True, choices=[value])
    for name in ("checkpoint", "showdown-root", "node-binary", "engine-model-path",
                 "engine-tables-path", "belief-set-source"):
        parser.add_argument("--" + name)
    for name in ("engine-worlds", "engine-sims", "engine-depth", "engine-batch",
                 "engine-model-world-workers", "max-decision-rounds",
                 "engine-policy-opponent-seed"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--engine-c-puct", type=float)
    for name in ("no-search-fallback", "engine-override-telemetry", "engine-policy-opponent",
                 "engine-opponent-priors", "no-engine-model-priors"):
        parser.add_argument("--" + name, action="store_true")
    try:
        args, unknown = parser.parse_known_args(entry["common_arguments"])
    except (argparse.ArgumentError, SystemExit) as error:
        raise ValueError("invalid parallel fixed-work arguments") from error
    if unknown or not args.no_search_fallback or any(
        getattr(args, name) is None or getattr(args, name) < 1
        for name in ("engine_worlds", "engine_sims", "engine_depth", "engine_batch",
                     "engine_model_world_workers")
    ):
        raise ValueError("parallel seats require explicit fixed work without timed/dynamic options")
    for shard in shards:
        arguments = shard["arguments"]
        expected = str(Path(entry["run_directory"]) / (shard["name"] + "-phase.json"))
        if arguments not in ([], ["--phase-progress-out", expected]):
            raise ValueError("parallel seat arguments may only name their own phase receipt")
    return mode


def worker_argv(entry, shard, summary):
    return [entry["python"], "-B", "-m", "pokezero.foulplay_bridge",
            *entry["common_arguments"], *shard["arguments"],
            "--games", str(entry["seed_count"]), "--seed-start", str(entry["seed_start"]),
            "--foulplay-random-seed", str(entry["foulplay_seed_start"]),
            "--pokezero-player", shard["seat"], "--summary-out", str(summary)]


def run_parallel_seats(entry, entry_path, entry_hash, output, env, terminal):
    """One owner/lock, one deadline, two independent completed-game watchdogs.

    Any failure drains every launched group, including failure to launch seat 2.
    No retries, resumed prefixes or worker-local renewable calendar clocks.
    """
    active = []
    reason = None
    deadline = terminal["deadline_unix"]

    def guard():
        if digest(entry_path) != entry_hash:
            raise ValueError("registration changed during collection")
        now = time.time()
        return "CALENDAR_CAP" if now >= deadline else coordination_reason(
            entry, entry_hash, deadline, now)

    def observe(state):
        receipt = state["receipt"]
        count = observed_count(state["summary"], entry["seed_start"], entry["seed_count"])
        if count < receipt["observed_games"]:
            raise ValueError("completed prefix shrank")
        if count > receipt["observed_games"]:
            receipt.update(observed_games=count, last_completion_unix=time.time())
            state["last_progress"] = receipt["last_completion_unix"]
            write(output / (receipt["name"] + "-progress.json"), receipt)
            print(json.dumps(receipt, sort_keys=True), flush=True)

    def finish(state, stop_reason):
        # Drain and preserve actual exit before parsing a potentially bad prefix.
        code, clean = stop_owned(state["process"])
        state["done"] = True
        receipt = state["receipt"]
        receipt.update(exit_code=code, cleanup_ok=clean, stop_reason=stop_reason,
                       ended_unix=time.time())
        terminal["shards"].append(receipt)
        path = output / (receipt["name"] + "-terminal.json")
        write(path, receipt)
        try:
            observe(state)
        except Exception as error:
            receipt["prefix_error"] = dict(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            write(path, receipt)
        return code, clean

    try:
        for shard in entry["shards"]:
            verify(entry)
            reason = guard()
            if reason:
                break
            summary = output / (shard["name"] + ".json")
            argv = worker_argv(entry, shard, summary)
            shard_start = time.time()
            with (output / (shard["name"] + ".log")).open("xb") as log:
                process = subprocess.Popen(argv, cwd=entry["runtime"], env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            receipt = dict(name=shard["name"], pid=process.pid, argv=argv,
                           started_unix=shard_start, deadline_unix=deadline,
                           observed_games=0, last_completion_unix=None)
            active.append(dict(process=process, receipt=receipt, summary=summary,
                               last_progress=shard_start, done=False))
            write(output / (shard["name"] + "-launch.json"), receipt)
        while reason is None and any(not state["done"] for state in active):
            reason = guard()
            if reason:
                break
            for state in active:
                if state["done"]:
                    continue
                observe(state)
                code = state["process"].poll()
                if code is not None:
                    code, clean = finish(state, None)
                    if code != 0 or not clean:
                        reason = "PROCESS_FAILURE"
                    else:
                        verify(entry)
                        reason = guard()  # Exit zero cannot bypass the cap.
                        if reason is None and state["receipt"]["observed_games"] != entry["seed_count"]:
                            raise ValueError("exit zero without the planned completed prefix")
                    state["receipt"]["stop_reason"] = reason
                    write(output / (state["receipt"]["name"] + "-terminal.json"), state["receipt"])
                elif time.time() - state["last_progress"] >= entry["no_completion_seconds"]:
                    reason = "NO_COMPLETION_TIMEOUT"
                if reason:
                    terminal["stopping_shard"] = state["receipt"]["name"]
                    break
            if reason is None and any(not state["done"] for state in active):
                time.sleep(5)
        terminal["status"] = reason or "COLLECTION_ATTEMPT_COMPLETE"
    finally:
        # A bad receipt for one worker must never prevent draining its peer.
        errors = []
        for state in active:
            if not state["done"]:
                try:
                    finish(state, reason or "SUPERVISOR_FAILURE")
                except Exception as error:
                    errors.append(dict(name=state["receipt"]["name"],
                                       error_type=type(error).__name__, error=str(error)))
        if errors:
            terminal["drain_errors"] = errors
            raise RuntimeError("parallel seat cleanup/receipt failure")
        verify(entry)
        if digest(entry_path) != entry_hash:
            raise ValueError("registration changed during collection")
        if terminal["status"] == "COLLECTION_ATTEMPT_COMPLETE":
            # The final input verification can itself cross the absolute cap.
            terminal["status"] = guard() or "COLLECTION_ATTEMPT_COMPLETE"


def run(entry_path, review_path, *, deadline_unix=None):
    entry_path = Path(entry_path)
    entry_hash = digest(entry_path)
    entry = json.loads(entry_path.read_text())
    review = json.loads(Path(review_path).read_text())
    if review.get("status") != "CLEAR" or review.get("entry_sha256") != entry_hash:
        raise ValueError("exact entry lacks independent CLEAR review")
    mode = validate_execution(entry)
    verify(entry)
    started = time.time()
    if entry.get("coordination") is not None and deadline_unix is None:
        raise ValueError("coordinated collection requires its common absolute deadline")
    if deadline_unix is not None and (
        type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix)
        or deadline_unix > started + entry["calendar_seconds"]
    ):
        raise ValueError("shared deadline must be finite and cannot extend the entry cap")
    deadline = started + entry["calendar_seconds"] if deadline_unix is None else deadline_unix
    output = Path(entry["run_directory"])
    output.mkdir()  # Exclusive: never resume, overwrite or redraw this entry.
    env = {k: os.environ[k] for k in ("PATH", "TMPDIR", "LANG", "HOME") if k in os.environ}
    env.update(entry["environment"])
    worker = None
    terminal = dict(entry_sha256=entry_hash, supervisor_pid=os.getpid(),
                    started_unix=started, deadline_unix=deadline,
                    deadline_source="entry_relative" if deadline_unix is None else "shared_absolute",
                    status="RUNNING", shards=[])
    write(output / "launch.json", terminal)
    # The laptop needs its owned idle-sleep assertion; Linux workers do not
    # provide caffeinate. This does not alter the common calendar deadline.
    assertion = (
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        if sys.platform == "darwin" else None
    )
    try:
        if mode == "parallel_seats_fixed_work":
            terminal["shard_execution"] = mode
            run_parallel_seats(entry, entry_path, entry_hash, output, env, terminal)
            return 0 if terminal["status"] == "COLLECTION_ATTEMPT_COMPLETE" else 1
        for shard in entry["shards"]:
            if digest(entry_path) != entry_hash:
                raise ValueError("registration changed during collection")
            verify(entry)
            now = time.time()
            if now >= deadline:
                terminal["status"] = "CALENDAR_CAP"
                break
            reason = coordination_reason(entry, entry_hash, deadline, now)
            if reason is not None:
                terminal["status"] = reason
                break
            summary = output / (shard["name"] + ".json")
            argv = worker_argv(entry, shard, summary)
            shard_start = time.time()
            with (output / (shard["name"] + ".log")).open("xb") as log:
                worker = subprocess.Popen(argv, cwd=entry["runtime"], env=env,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                receipt = dict(name=shard["name"], pid=worker.pid, argv=argv,
                               started_unix=shard_start, observed_games=0,
                               last_completion_unix=None)
                write(output / (shard["name"] + "-launch.json"), receipt)
                last_progress = shard_start
                reason = None
                while worker.poll() is None:
                    count = observed_count(summary, entry["seed_start"], entry["seed_count"])
                    if count < receipt["observed_games"]:
                        raise ValueError("completed prefix shrank")
                    if count > receipt["observed_games"]:
                        receipt.update(observed_games=count, last_completion_unix=time.time())
                        last_progress = receipt["last_completion_unix"]
                        write(output / "progress.json", receipt)
                        print(json.dumps(receipt, sort_keys=True), flush=True)
                    now = time.time()
                    if now >= deadline:
                        reason = "CALENDAR_CAP"
                        break
                    reason = coordination_reason(entry, entry_hash, deadline, now)
                    if reason is not None:
                        break
                    if now - last_progress >= entry["no_completion_seconds"]:
                        reason = "NO_COMPLETION_TIMEOUT"
                        break
                    time.sleep(5)
                code, cleanup_ok = stop_owned(worker)
                worker = None
                receipt.update(exit_code=code, cleanup_ok=cleanup_ok,
                               observed_games=observed_count(summary, entry["seed_start"], entry["seed_count"]))
                terminal["shards"].append(receipt)
                # Preserve the actual child outcome before a later provenance
                # check can raise. A failed input check must not erase a
                # nonzero child exit or prevent its terminal receipt existing.
                receipt["stop_reason"] = reason
                write(output / (shard["name"] + "-terminal.json"), receipt)
                try:
                    verify(entry)
                except Exception as error:
                    failure = dict(error_type=type(error).__name__, error=str(error))
                    terminal["post_exit_verification_error"] = failure
                    receipt.update(post_exit_verification_error=failure, ended_unix=time.time())
                    write(output / (shard["name"] + "-terminal.json"), receipt)
                    if code != 0 or not cleanup_ok or reason is not None:
                        terminal["status"] = reason or "PROCESS_FAILURE"
                        break
                    raise  # A clean child never excuses provenance drift.
                # poll() may first observe an already-exited worker. Recheck at
                # acceptance, after cleanup/provenance checks, so an exit zero
                # cannot bypass a cap, peer abort or lost coordinator lease.
                now = time.time()
                if reason is None and code == 0 and cleanup_ok:
                    reason = ("CALENDAR_CAP" if now >= deadline else
                              coordination_reason(entry, entry_hash, deadline, now))
                receipt.update(ended_unix=now, stop_reason=reason)
                write(output / (shard["name"] + "-terminal.json"), receipt)
                # Conservative first launch: unexplained failure stops everything.
                # No automatic refusal classification or candidate restart.
                if code != 0 or not cleanup_ok or reason is not None:
                    terminal["status"] = reason or "PROCESS_FAILURE"
                    break
                if receipt["observed_games"] != entry["seed_count"]:
                    raise ValueError("exit zero without the planned completed prefix")
        else:
            terminal["status"] = "COLLECTION_ATTEMPT_COMPLETE"
    except BaseException as error:
        terminal.update(status="SUPERVISOR_FAILURE", error_type=type(error).__name__,
                        error=str(error))
        if worker is not None:
            code, clean = stop_owned(worker)
            terminal.update(interrupted_worker_exit=code, cleanup_ok=clean)
        raise
    finally:
        terminal["ended_unix"] = time.time()
        terminal["scientific_qualification"] = "UNREVIEWED"
        write(output / "terminal.json", terminal)
        if assertion is not None:
            if assertion.poll() is None:
                assertion.terminate()
            assertion.wait(timeout=10)
    return 0 if terminal["status"] == "COLLECTION_ATTEMPT_COMPLETE" else 1


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--deadline-unix", type=float,
                        help="One coordinator-owned absolute deadline; never extends the entry cap.")
    args = parser.parse_args()
    # Shared local lock prevents this supervisor from overlapping another panel.
    with open("/tmp/pokezero-phase0-budget.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(args.entry, args.review, deadline_unix=args.deadline_unix))
