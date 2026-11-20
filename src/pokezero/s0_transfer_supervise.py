"""Bound a scratch partner and independently verify its archived cells."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from .s0_transfer_harness import wait_for
from .s0_transfer_roster import write_once


DEADLINE_SECONDS = {"engine": 5400, "showdown": 5400}


def finalize(root):
    manifest = root / "manifest.json"
    shards = {}
    producer_counts = Counter()
    for side in ("engine", "showdown"):
        terminal = wait_for(root / f"{side}.terminal.json", f"{side} terminal receipt absent")
        if terminal.get("status") != "COMPLETE" or terminal.get("dry_run") is not True:
            raise ValueError(f"{side} failed before sealing its scratch cells")
        name = f"{side}.cells.json"
        if terminal["name"] != name:
            raise ValueError("producer did not seal its expected shard")
        shards[name] = terminal["sha256"]
        producer_counts.update(terminal["status_counts"])
    seals = root / "seals.json"
    write_once(seals, dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), shards=shards))
    recomputed = root / "recomputed.json"
    subprocess.run([sys.executable, "-m", "pokezero.s0_transfer_recompute", str(manifest),
                    str(seals), str(recomputed)], check=True, timeout=120)
    result = json.loads(recomputed.read_bytes())
    if (result["status"] != "UNSCORED" or result["verdict"] is not None
            or result["primitive_cells"] != sum(producer_counts.values())
            or result["complete_cells"] != producer_counts["COMPLETE"]):
        raise ValueError("offline terminal scoring and producer counts differ")
    write_once(root / "comparison.json", dict(
        status="UNSCORED", raw_cells=result["primitive_cells"], terminal_counts=dict(producer_counts),
        offline_recompute_sha256=hashlib.sha256(recomputed.read_bytes()).hexdigest(),
    ))


def supervise(root, side, rest):
    env = dict(os.environ)
    if side == "showdown":
        env["JAX_PLATFORMS"] = "cpu"
    command = [sys.executable, "-m", "pokezero.s0_transfer_harness", side,
               "--out", str(root), *rest]
    with subprocess.Popen(command, env=env, start_new_session=True) as child:
        try:
            code = child.wait(timeout=DEADLINE_SECONDS[side])
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
    if code == 0 and side == "showdown":
        finalize(root)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=tuple(DEADLINE_SECONDS))
    parser.add_argument("directory", type=Path)
    args, rest = parser.parse_known_args()
    root = args.directory.resolve()
    peak = Path("/sys/fs/cgroup/memory.peak")
    current = Path("/sys/fs/cgroup/memory.current")
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            samples.append(int(current.read_text()))
            stop.wait(1)

    monitor = None
    if not peak.is_file() and current.is_file():
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
    try:
        code = supervise(root, args.side, rest)
    except (TimeoutError, ValueError, OSError, subprocess.SubprocessError) as error:
        code = 1
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
    write_once(root / f"{args.side}.supervisor.json", dict(
        dry_run=True, exit_code=code, deadline_seconds=DEADLINE_SECONDS[args.side],
        memory_peak_bytes=int(peak.read_text()) if peak.is_file() else max(samples, default=None),
    ))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
