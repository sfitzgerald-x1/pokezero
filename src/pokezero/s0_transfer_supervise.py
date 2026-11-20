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
import time

from .s0_transfer_harness import wait_for
from .s0_transfer_roster import SCHEMA, cell_record, digest, expected_cells, journal_path, write_once


DEADLINE_SECONDS = {"engine": 5400, "showdown": 5400}
SHOWDOWN_CONTINUATION_SECONDS = 10800


def seal_side(root, side, exit_code, elapsed_seconds):
    terminal_path = root / f"{side}.terminal.json"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        write_once(terminal_path, dict(dry_run=True, status="FAILED", worker_exit_code=exit_code,
                                      reason="worker exited before a scratch manifest was available"))
        return False
    manifest = json.loads(manifest_path.read_bytes())
    expected = expected_cells(manifest["banks"], dry_run=True)
    if (manifest["schema"] != SCHEMA or manifest["dry_run"] is not True
            or manifest["expected_cells_sha256"] != digest(list(expected.values()))):
        raise ValueError("cannot seal cells from a changed scratch manifest")
    records = []
    missing = 0
    for cell in expected.values():
        if cell["environment"] != side:
            continue
        path = journal_path(root, cell)
        if path.is_file():
            records.append(json.loads(path.read_bytes()))
        else:
            missing += 1
            reason = f"worker exit {exit_code} before a durable cell result"
            records.append(cell_record(manifest, cell, "FAILED", None, {"reason": reason}))
    name = f"{side}.cells.json"
    shard = write_once(root / name, records)
    complete = exit_code == 0 and missing == 0
    write_once(terminal_path, dict(
        dry_run=True, status="COMPLETE" if complete else "PARTIAL", name=name, sha256=shard,
        cells=len(records), missing_journal=missing, worker_exit_code=exit_code,
        elapsed_seconds=elapsed_seconds, status_counts=dict(Counter(r["status"] for r in records)),
    ))
    return complete


def finalize(root):
    manifest = root / "manifest.json"
    shards = {}
    producer_counts = Counter()
    for side in ("engine", "showdown"):
        terminal = wait_for(root / f"{side}.terminal.json", f"{side} terminal receipt absent")
        if terminal.get("status") not in ("COMPLETE", "PARTIAL") or terminal.get("dry_run") is not True:
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


def supervise(root, side, rest, *, deadline):
    env = dict(os.environ)
    if side == "showdown":
        env["JAX_PLATFORMS"] = "cpu"
    command = [sys.executable, "-m", "pokezero.s0_transfer_harness", side,
               "--out", str(root), *rest]
    started = time.monotonic()
    with subprocess.Popen(command, env=env, start_new_session=True) as child:
        try:
            code = child.wait(timeout=deadline)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
    complete = seal_side(root, side, code, time.monotonic() - started)
    if side == "showdown" and (root / "manifest.json").is_file():
        try:
            finalize(root)
        except (OSError, TimeoutError, ValueError, subprocess.SubprocessError) as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return code or 1
    return code if complete else (code or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=tuple(DEADLINE_SECONDS))
    parser.add_argument("directory", type=Path)
    args, rest = parser.parse_known_args()
    root = args.directory.resolve()
    deadline = (SHOWDOWN_CONTINUATION_SECONDS if args.side == "showdown" and "--resume-from" in rest
                else DEADLINE_SECONDS[args.side])
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
        code = supervise(root, args.side, rest, deadline=deadline)
    except (TimeoutError, ValueError, OSError, subprocess.SubprocessError) as error:
        code = 1
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
    write_once(root / f"{args.side}.supervisor.json", dict(
        dry_run=True, exit_code=code, deadline_seconds=deadline,
        memory_peak_bytes=int(peak.read_text()) if peak.is_file() else max(samples, default=None),
    ))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
