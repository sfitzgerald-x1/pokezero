#!/usr/bin/env python3
"""One bounded authored decision, never a game or scientific collection entry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _OwnerSignal(BaseException):
    def __init__(self, signum):
        self.signum = signum


def run_owned(command, *, env, hard_seconds, reap_seconds=5):
    """Always drain this newly owned process group, including on TERM/INT."""
    process = None
    timed_out, owner_signal, pending_signal = False, None, None
    launching, child_reaped = True, False
    stdout, stderr = "", ""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        nonlocal pending_signal
        # Popen can spawn the child before returning its handle. Raising while
        # that assignment is incomplete would lose ownership of a live child.
        pending_signal = signum
        if not launching:
            raise _OwnerSignal(signum)
    for sig in previous:
        signal.signal(sig, interrupted)
    try:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        launching = False
        if pending_signal is not None:
            raise _OwnerSignal(pending_signal)
        try:
            process.communicate(timeout=hard_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
    except _OwnerSignal as error:
        owner_signal = error.signum
    finally:
        # A second interrupt must not interrupt the cleanup of the first.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    stdout, stderr = process.communicate(timeout=reap_seconds)
                    child_reaped = process.returncode is not None
                except subprocess.TimeoutExpired as error:
                    def decoded(value):
                        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                    stdout, stderr = decoded(error.output), decoded(error.stderr)
                    # Preserve an unknown actual exit rather than hanging or
                    # promoting process disappearance into a successful exit.
                    for stream in (process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return {"actual_exit_code": process.returncode if process else None,
            "owned_child_pid": process.pid if process else None,
            "stdout": stdout if process else "", "stderr": stderr if process else "",
            "timed_out": timed_out, "owner_signal": owner_signal,
            "owned_group_drain_attempted": process is not None,
            "owned_child_reaped": child_reaped}


def owned_exit_status(owned):
    if owned["owner_signal"]:
        return 128 + owned["owner_signal"]
    if owned["timed_out"] or not owned["owned_child_reaped"]:
        return 1
    code = owned["actual_exit_code"]
    return code if code and code > 0 else 0 if code == 0 else 1


def worker(args):
    # The owner supplies this at process creation, before importing native code.
    if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
        raise RuntimeError("missing fresh-process exact-roll setting")
    import pokezero_search as native
    from pokezero.shallow_exact import exact_depth_two
    from pokezero.shallow_exact_native import NativeExactBackend, NativePosition, validate_encoder_frame

    fixture = json.loads(Path(args.fixture).read_text())
    if fixture.get("authored_fixture") is not True or fixture.get("scientific_collection") is not False:
        raise RuntimeError("this driver accepts explicitly authored non-collection fixtures only")
    if sha(args.model) != fixture["model_sha256"] or sha(args.tables) != fixture["tables_sha256"]:
        raise RuntimeError("model/tables bytes changed from the authored fixture binding")
    for row in fixture["worlds"]:
        validate_encoder_frame(row["root_inputs"], row["self_side"])
    # Behavioral check of THIS native engine's process-local roll mode. A flag
    # alone cannot attest a OnceLock that another engine call initialized.
    probe = fixture["enumeration_probe"]
    branches = json.loads(native.branch_events(
        probe["state_str"], probe["row_action"], probe["column_action"],
        json.dumps(probe["ctx"]), branch_on_damage=True, include_post_state=False,
    ))["branches"]
    damages = sorted({probe["defender_hp"] - branch["post"][probe["defender_side"]]["active_hp"]
                      for branch in branches})
    if damages != probe["expected_damage_values"] or len(damages) <= 2:
        raise RuntimeError("native exact-damage behavioral attestation failed")
    tables = Path(args.tables).read_text()
    spec = fixture["observation_spec"]
    model = native.NativeLeafModel(args.model, device=args.device, **spec)
    worlds = []
    for row in fixture["worlds"]:
        ctx = row["ctx"]
        encoder = native.LeafEncoder(tables, row["root_inputs"], json.dumps(ctx), row["state_str"])
        worlds.append(NativePosition(
            row["state_str"], {side: tuple(ctx[side]) for side in ("p1", "p2")},
            ctx["turn"], native.FoldState.from_payload(row["fold_state"]),
            encoder, row["self_side"],
        ))
    backend = NativeExactBackend(native, model, window=spec["window"],
                                 max_outcomes_per_call=args.max_branches)
    started = time.perf_counter()
    try:
        result = exact_depth_two(
            worlds, backend, batch_size=args.batch_size, max_leaves=args.max_leaves,
            max_joint_actions=args.max_joint_actions, max_chance_branches=args.max_branches,
            max_seconds=args.soft_seconds,
            streaming=getattr(args, "streaming", False),
        )
        report = {"complete_decision": True, "result": asdict(result)}
        code = 0
    except Exception as error:
        report = {"complete_decision": False, "error_type": type(error).__name__, "error": str(error)}
        code = 1
    report.update(elapsed_seconds=time.perf_counter() - started,
                  encode_calls=backend.encode_calls, forward_batches=backend.forward_batches,
                  render_refusals=dict(backend.render_refusals), exact_roll_probe_values=damages,
                  render_subcases=dict(backend.render_subcases), stage_seconds=dict(backend.stage_seconds),
                  device=args.device, scientific_collection=False, scientific_progress=False,
                  limitation="Authored fixture only; prepared roots and model load excluded; not battle throughput or strength.")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "model", "tables", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--streaming", action="store_true",
                        help="complete exact child matrices with bounded pending leaf batches")
    parser.add_argument("--max-leaves", type=int, default=8192)
    parser.add_argument("--max-joint-actions", type=int, default=4096)
    parser.add_argument("--max-branches", type=int, default=16384)
    parser.add_argument("--soft-seconds", type=float, default=45)
    parser.add_argument("--hard-seconds", type=float, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("refusing to overwrite a prior benchmark receipt")
    if (not math.isfinite(args.hard_seconds)
            or not 0 < args.soft_seconds < args.hard_seconds):
        raise RuntimeError("positive soft deadline must be shorter than hard deadline")
    started = time.time()
    owned = run_owned(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
        env={**os.environ, "POKEZERO_ENUMERATE_ROLLS": "1"},
        hard_seconds=args.hard_seconds,
    )
    report = {"schema": "pokezero.shallow-exact-authored-benchmark.v1",
              "scientific_collection": False, "scientific_progress": False,
              **owned, "started_unix": started,
              "ended_unix": time.time(), "fixture_sha256": sha(args.fixture),
              "driver_sha256": sha(__file__)}
    with output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"actual_exit_code": owned["actual_exit_code"], "timed_out": owned["timed_out"],
                      "output": str(output)}))
    return owned_exit_status(owned)


if __name__ == "__main__":
    raise SystemExit(main())
