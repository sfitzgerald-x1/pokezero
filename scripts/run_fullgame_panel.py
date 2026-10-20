#!/usr/bin/env python3
"""One reviewed, non-resumable four-shard panel using the existing game runner.

Collection only: prefixes are observations awaiting independent readout. A
nonzero exit is never replaced by a clean prefix's execution_integrity block.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
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


def run(entry_path, review_path):
    entry_path = Path(entry_path)
    entry_hash = digest(entry_path)
    entry = json.loads(entry_path.read_text())
    review = json.loads(Path(review_path).read_text())
    if review.get("status") != "CLEAR" or review.get("entry_sha256") != entry_hash:
        raise ValueError("exact entry lacks independent CLEAR review")
    verify(entry)
    output = Path(entry["run_directory"])
    output.mkdir()  # Exclusive: never resume, overwrite or redraw this entry.
    env = {k: os.environ[k] for k in ("PATH", "TMPDIR", "LANG", "HOME") if k in os.environ}
    env.update(entry["environment"])
    started = time.time()
    deadline = started + entry["calendar_seconds"]
    worker = None
    terminal = dict(entry_sha256=entry_hash, supervisor_pid=os.getpid(),
                    started_unix=started, deadline_unix=deadline, status="RUNNING", shards=[])
    write(output / "launch.json", terminal)
    assertion = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        for shard in entry["shards"]:
            if digest(entry_path) != entry_hash:
                raise ValueError("registration changed during collection")
            verify(entry)
            if time.time() >= deadline:
                terminal["status"] = "CALENDAR_CAP"
                break
            summary = output / (shard["name"] + ".json")
            argv = [entry["python"], "-B", "-m", "pokezero.foulplay_bridge",
                    *entry["common_arguments"], *shard["arguments"],
                    "--games", str(entry["seed_count"]),
                    "--seed-start", str(entry["seed_start"]),
                    "--foulplay-random-seed", str(entry["foulplay_seed_start"]),
                    "--pokezero-player", shard["seat"], "--summary-out", str(summary)]
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
                    if time.time() >= deadline:
                        reason = "CALENDAR_CAP"
                        break
                    if time.time() - last_progress >= entry["no_completion_seconds"]:
                        reason = "NO_COMPLETION_TIMEOUT"
                        break
                    time.sleep(5)
                code, cleanup_ok = stop_owned(worker)
                worker = None
                receipt.update(exit_code=code, cleanup_ok=cleanup_ok, ended_unix=time.time(),
                               observed_games=observed_count(summary, entry["seed_start"], entry["seed_count"]),
                               stop_reason=reason)
                write(output / (shard["name"] + "-terminal.json"), receipt)
                terminal["shards"].append(receipt)
                verify(entry)
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
    args = parser.parse_args()
    # Shared local lock prevents this supervisor from overlapping another panel.
    with open("/tmp/pokezero-phase0-budget.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(args.entry, args.review))
