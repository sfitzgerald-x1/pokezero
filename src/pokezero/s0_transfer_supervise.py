"""Run a transfer worker with a hard job deadline and a single terminal status."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .s0_transfer_harness import wait_for
from .s0_transfer_roster import Shard, write_json


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=tuple(SCRATCH_DEADLINE))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--deadline-seconds", type=int)
    parser.add_argument("--shard", type=Shard.parse, default=Shard())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish-stop", action="store_true")
    args, rest = parser.parse_known_args()
    deadline = args.deadline_seconds if args.deadline_seconds is not None else SCRATCH_DEADLINE[args.side]
    if deadline <= 0:
        parser.error("a positive job deadline is required")
    root = args.directory.resolve()
    name = args.shard.stem(args.side)
    done = root / f"{name}.done.json"
    finish = time.monotonic() + deadline
    sharded = args.shard.count > 1
    if args.side == "engine" and sharded:
        parser.error("only CPU workers are sharded by this supervisor")
    if done.exists() and not args.resume:
        parser.error("previous attempt exists. Use --resume to continue the same shard")
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

    monitor = None
    if not peak.is_file() and current.is_file():
        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
    code = 1
    try:
        code = run_worker(root, args.side, worker_args, deadline=deadline)
        if code == 0 and args.side == "showdown" and args.shard.index == 0 and (root / "manifest.json").is_file():
            peers = ["engine"] + [Shard(i, args.shard.count).stem("showdown")
                                  for i in range(1, args.shard.count)]
            for peer in peers:
                terminal = wait_for(root / f"{peer}.done.json", f"{peer} terminal status unavailable",
                                    seconds=max(0, finish - time.monotonic()))
                if terminal["exit_code"]:
                    code = terminal["exit_code"]
                    return code
            command = [sys.executable, "-m", "pokezero.s0_transfer_recompute", str(root)]
            if args.publish_stop:
                command.append("--publish-stop")
            code = subprocess.run(command, check=False, timeout=max(0, finish - time.monotonic())).returncode
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        code = code or 1
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
        write_json(done, dict(
            exit_code=code, memory_peak_bytes=int(peak.read_text()) if peak.is_file() else maximum,
        ))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
