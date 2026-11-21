"""Run a transfer worker with a hard job deadline and a single terminal status."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from .s0_transfer_harness import wait_for
from .s0_transfer_roster import write_json


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
    args, rest = parser.parse_known_args()
    deadline = args.deadline_seconds if args.deadline_seconds is not None else SCRATCH_DEADLINE[args.side]
    if deadline <= 0:
        parser.error("a positive job deadline is required")
    root = args.directory.resolve()
    if (root / f"{args.side}.done.json").exists() or (root / f"{args.side}.jsonl").exists():
        parser.error("this side has already run in the output directory")

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
        code = run_worker(root, args.side, rest, deadline=deadline)
        if args.side == "showdown" and (root / "manifest.json").is_file():
            engine = wait_for(root / "engine.done.json", "engine terminal status unavailable")
            code = code or engine["exit_code"]
            scorer = subprocess.run([sys.executable, "-m", "pokezero.s0_transfer_recompute", str(root)],
                                    check=False, timeout=deadline)
            code = code or scorer.returncode
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        code = code or 1
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
        write_json(root / f"{args.side}.done.json", dict(
            exit_code=code, memory_peak_bytes=int(peak.read_text()) if peak.is_file() else maximum,
        ))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
