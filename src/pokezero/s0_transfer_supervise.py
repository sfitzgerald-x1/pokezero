"""Run a transfer worker with a hard job deadline and a single terminal status."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .s0_transfer_harness import wait_for
from .s0_transfer_roster import Shard, panel_lock, write_json


SCRATCH_DEADLINE = {"engine": 5400, "showdown": 14400}


def run_worker(root, side, rest, *, deadline):
    env = dict(os.environ)
    if side == "showdown":
        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    command = [sys.executable, "-m", "pokezero.s0_transfer_harness", side, "--out", str(root), *rest]
    with subprocess.Popen(command, env=env, start_new_session=True) as child:
        try:
            return child.wait(timeout=deadline)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124 if isinstance(error, subprocess.TimeoutExpired) else 130


def score_after_workers(root, peers, finish, score_seconds, publish_stop, attempt):
    paths = [root / f"{peer}.done.json" for peer in peers]
    for path in paths:
        try:
            wait_for(path, f"{path.name} unavailable", seconds=max(0, finish - time.monotonic()),
                     min_attempt=attempt)
        except TimeoutError:
            break  # Unstarted and evicted peers contribute missing cells.
        except (OSError, ValueError, TypeError, AttributeError) as error:
            print(f"{path.name}: {error}", file=sys.stderr)
    command = [sys.executable, "-m", "pokezero.s0_transfer_recompute", str(root),
               "--lock-seconds", str(score_seconds)]
    if publish_stop:
        command.append("--publish-stop")
    try:
        score_code = subprocess.run(command, check=False, timeout=score_seconds).returncode
    except subprocess.TimeoutExpired:
        score_code = 124
    except OSError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        score_code = 1
    peer_code = 0
    for path in paths:
        try:
            receipt = json.loads(path.read_bytes())
            status = receipt["exit_code"] if receipt.get("attempt", 0) >= attempt else 124
        except FileNotFoundError:
            status = 124
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            print(f"{path.name}: {error}", file=sys.stderr)
            status = 1
        peer_code = peer_code or status
    return score_code, peer_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=tuple(SCRATCH_DEADLINE))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--deadline-seconds", type=int)
    parser.add_argument("--score-seconds", type=int, default=300, help="additional wall for quiescence and scoring")
    parser.add_argument("--shard", type=Shard.parse, default=Shard())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt", type=int, help="shared increasing attempt number for coordinated resumes")
    parser.add_argument("--publish-stop", action="store_true")
    args, rest = parser.parse_known_args(argv)
    deadline = args.deadline_seconds if args.deadline_seconds is not None else SCRATCH_DEADLINE[args.side]
    if min(deadline, args.score_seconds) <= 0:
        parser.error("a positive job deadline is required")
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = args.shard.stem(args.side)
    done = root / f"{name}.done.json"
    finish = time.monotonic() + deadline
    sharded = args.shard.count > 1
    if args.side == "engine" and sharded:
        parser.error("only CPU workers are sharded by this supervisor")
    worker_args = [*rest, "--shard", f"{args.shard.index}/{args.shard.count}",
                   "--join-seconds", str(deadline)]
    if args.resume:
        worker_args.append("--resume")

    peak = Path("/sys/fs/cgroup/memory.peak")
    current = Path("/sys/fs/cgroup/memory.current")
    maximum = None
    stop = threading.Event()

    def sample_memory():
        nonlocal maximum
        while not stop.is_set():
            size = int(current.read_text())
            maximum = size if maximum is None else max(maximum, size)
            stop.wait(1)

    with (root / f"{name}.attempt.lock").open("a") as attempt:
        fcntl.flock(attempt, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = root / f"{name}.attempt.json"
        if (started.exists() or done.exists()) and not args.resume:
            parser.error("previous attempt exists. Use --resume to continue the same shard")
        previous = json.loads(started.read_bytes())["attempt"] if started.exists() else (0 if done.exists() else -1)
        number = args.attempt if args.attempt is not None else max(int(args.resume), previous + 1)
        if number <= previous or number < 0:
            parser.error("attempt must increase for each shard restart")
        worker_args.extend(["--attempt", str(number)])
        monitor = None
        if not peak.is_file() and current.is_file():
            monitor = threading.Thread(target=sample_memory, daemon=True)
            monitor.start()
        receipt = {}
        try:
            with panel_lock(root, wait_seconds=deadline):
                write_json(started, {"attempt": number})
                done.unlink(missing_ok=True)
                try:
                    remaining = finish - time.monotonic()
                    code = run_worker(root, args.side, worker_args, deadline=remaining) if remaining > 0 else 124
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    print(f"{type(error).__name__}: {error}", file=sys.stderr)
                    code = 1
                receipt.update(attempt=number, worker_exit_code=code, exit_code=code)
                coordinator = args.side == "showdown" and args.shard.index == 0 and (root / "manifest.json").is_file()
                if not coordinator:
                    receipt["memory_peak_bytes"] = int(peak.read_text()) if peak.is_file() else maximum
                    write_json(done, receipt)
            if coordinator:
                peers = ["engine"] + [Shard(i, args.shard.count).stem("showdown")
                                      for i in range(1, args.shard.count)]
                score_code, peer_code = score_after_workers(root, peers, finish, args.score_seconds, args.publish_stop, number)
                code = code or peer_code or score_code
                receipt.update(exit_code=code, score_exit_code=score_code,
                               memory_peak_bytes=int(peak.read_text()) if peak.is_file() else maximum)
                write_json(done, receipt)
        finally:
            stop.set()
            if monitor is not None:
                monitor.join()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
