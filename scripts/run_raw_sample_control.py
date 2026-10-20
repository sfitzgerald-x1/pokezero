#!/usr/bin/env python3
"""Fixed 32-seed, both-seat policy control; no search, retry or resumption.

The sibling supervisor supplies provenance and owned-process cleanup primitives.
Scientific imports happen only in the pinned runtime's worker, not the parent.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import warnings

import run_fullgame_panel as panel


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(start_ticks=fields[19], state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def parent_guard_reason(launch, now):
    identity = process_identity(launch["supervisor_pid"])
    if (identity is None or identity["state"] == "Z"
            or identity["start_ticks"] != launch["supervisor_start_ticks"]):
        return "SUPERVISOR_LOST"
    return "CALENDAR_CAP" if now >= launch["deadline_unix"] else None


def guard_parent(launch, output):
    # A worker owns its new process group, including every Showdown child.
    # If the supervisor is SIGKILLed, this independent watchdog stops collection.
    while True:
        reason = parent_guard_reason(launch, time.time())
        if reason:
            try:
                panel.write(Path(output) / "worker-stop.json",
                            dict(reason=reason, worker_pid=os.getpid(), observed_unix=time.time()))
            finally:
                os.killpg(os.getpid(), signal.SIGKILL)
        time.sleep(1)


def slots(entry):
    seeds = entry["seeds"]
    if (len(seeds) != 32 or len(set(seeds)) != 32
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("control requires exactly 32 distinct uint32 seeds")
    return [dict(seed=seed, raw_seat=seat) for seed in seeds for seat in ("p1", "p2")]


def check(entry_path, review_path):
    entry_path = Path(entry_path)
    entry = json.loads(entry_path.read_text())
    review = json.loads(Path(review_path).read_text())
    if review.get("status") != "CLEAR" or review.get("entry_sha256") != panel.digest(entry_path):
        raise ValueError("exact entry lacks independent CLEAR review")
    if entry.get("slots") != slots(entry):
        raise ValueError("slots must be the fixed seed-major, both-seat roster")
    if (entry.get("calendar_seconds") != 86400
            or entry.get("no_completion_seconds") != 1800
            or entry.get("first_completion_seconds") != 900
            or entry.get("max_decision_rounds") != 250):
        raise ValueError("control caps changed")
    if entry.get("host") != dict(
        hostname=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    ):
        raise ValueError("registered cloud host changed")
    cpu = entry.get("cpu")
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0):
        raise ValueError("registered CPU unavailable")
    for path in (Path(__file__).resolve(), Path(panel.__file__).resolve(),
                 Path(entry["checkpoint"]), Path(entry["node_binary"]), Path(entry["python"])):
        if str(path) not in entry["input_hashes"]:
            raise ValueError(f"unbound executable or input: {path}")
    if entry["checkpoint_sha256"] != entry["input_hashes"][entry["checkpoint"]]:
        raise ValueError("checkpoint metadata differs from bound weights")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if entry["environment"].get(name) != "1":
            raise ValueError("control requires single-thread policy execution")
    if Path(entry["environment"]["PATH"].split(os.pathsep)[0]) != Path(entry["node_binary"]).parent:
        raise ValueError("catalog loader must resolve the pinned node first")
    panel.verify(entry)
    return entry


def prefix(output, entry):
    path = Path(output) / "progress.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())["games"]
    if len(rows) > 64 or [dict(seed=r["seed"], raw_seat=r["raw_seat"]) for r in rows] != entry["slots"][:len(rows)]:
        raise ValueError("progress is not the fixed chronological prefix")
    for row in rows:
        if row["status"] not in ("COMPLETE", "CAPPED"):
            raise ValueError("invalid observation status")
        expected = None if row["status"] == "CAPPED" else (
            0.5 if row["winner"] is None else float(row["winner"] == row["raw_seat"]))
        if row["raw_score"] != expected:
            raise ValueError("unknown or terminal score changed")
    return rows


def terminal_row(slot, terminal):
    if terminal.winner not in (None, "p1", "p2"):
        raise ValueError("invalid terminal winner")
    return dict(**slot, status="CAPPED" if terminal.capped else "COMPLETE",
                winner=terminal.winner, raw_score=None if terminal.capped else (
                    0.5 if terminal.winner is None else float(terminal.winner == slot["raw_seat"])))


def play_game(entry, slot, model, result, env_config):
    from pokezero.local_showdown import LocalShowdownEnv
    from pokezero.neural_policy import TransformerSoftmaxPolicy
    from pokezero.rollout import RolloutConfig, RolloutDriver
    from pokezero.trajectory import trajectory_to_dict

    # Fresh adapters per game: no shared mutable history or opponent observation.
    policies = {seat: TransformerSoftmaxPolicy(
        model=model, result=result, deterministic=seat == slot["raw_seat"],
        exploration_epsilon=0.0, sampling_temperature=1.0,
        family_gated_selection=False, history_mask_k=None, device="cpu",
        policy_id="champion-raw" if seat == slot["raw_seat"] else "champion-sampled",
        checkpoint_path=entry["checkpoint"], weights_sha256=entry["checkpoint_sha256"],
    ) for seat in ("p1", "p2")}
    with LocalShowdownEnv(env_config) as env:
        result = RolloutDriver(env=env, policies=policies, config=RolloutConfig(
            max_decision_rounds=250, format_id="gen3randombattle",
            hide_opponent_legal_action_masks=True,
        )).run(seed=slot["seed"], battle_id=f"raw-sampled-{slot['seed']}-{slot['raw_seat']}")
        document = dict(trajectory=trajectory_to_dict(result.trajectory),
                        protocol_lines=list(env.protocol_lines))
        row = terminal_row(slot, result.terminal)
        row["decision_round_count"] = result.decision_round_count
    # An exception during bridge close leaves this slot unknown, never admitted.
    return row, document


def worker(entry_path, review_path):
    entry = check(entry_path, review_path)
    launch_path = Path(entry["run_directory"]) / "launch.json"
    # Parent publishes the actual PID just after Popen. No scientific work is
    # performed while waiting for this bounded, exact launch handshake.
    limit = time.monotonic() + 5
    launch = json.loads(launch_path.read_text())
    while "worker_pid" not in launch and time.monotonic() < limit:
        time.sleep(0.05)
        launch = json.loads(launch_path.read_text())
    if (launch.get("worker_pid") != os.getpid()
            or launch.get("entry_sha256") != panel.digest(entry_path)
            or os.getppid() != launch["supervisor_pid"]
            or os.getpgrp() != os.getpid()):
        raise ValueError("worker is not this entry's actual owned launch")
    if parent_guard_reason(launch, time.time()) is not None:
        raise ValueError("supervisor lost or calendar cap before scientific work")
    threading.Thread(target=guard_parent, args=(launch, entry["run_directory"]),
                     daemon=True).start()
    os.sched_setaffinity(0, {entry["cpu"]})
    import torch
    from pokezero.local_showdown import LocalShowdownConfig, env_config_from_checkpoint_provenance
    from pokezero.neural_policy import (
        FreshValueHeadWarning, category_vocab_from_model_config,
        feature_masks_from_model_config, load_transformer_checkpoint,
        observation_spec_from_model_config,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warnings.simplefilter("error", FreshValueHeadWarning)
    source = load_gen3_randbat_source_cached(entry["showdown_root"])
    if source.metadata.source_hash != entry["catalog_source_hash"]:
        raise ValueError("resolved belief catalog differs from the entry")
    model, trained = load_transformer_checkpoint(entry["checkpoint"], map_location="cpu")
    model.eval()
    vocab = category_vocab_from_model_config(trained.model_config, entry["showdown_root"])
    config = env_config_from_checkpoint_provenance(
        LocalShowdownConfig(showdown_root=entry["showdown_root"],
                            node_binary=entry["node_binary"], set_belief_source=True,
                            category_vocab=vocab),
        feature_masks_from_model_config(trained.model_config),
        required_specs=observation_spec_from_model_config(trained.model_config),
        required_vocabs=vocab, context="raw-sampled-control",
    )
    output = Path(entry["run_directory"])
    if (output / "progress.json").exists():
        raise ValueError("worker cannot resume an existing prefix")
    rows = []
    for index, slot in enumerate(entry["slots"]):
        row, document = play_game(entry, slot, model, trained, config)
        journal = output / f"game-{index:02d}.json.gz"
        # Only publish the prefix after the full journal and clean env close.
        with journal.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(json.dumps(document, sort_keys=True).encode())
        row.update(journal=journal.name, journal_sha256=panel.digest(journal),
                   completed_unix=time.time())
        rows.append(row)
        panel.write(output / "progress.json", dict(games=rows, planned_games=64,
                    scientific_qualification="UNREVIEWED"))
        print(json.dumps(dict(observed_games=len(rows), **row)), flush=True)
    return 0


def run(entry_path, review_path):
    entry = check(entry_path, review_path)
    output = Path(entry["run_directory"])
    output.mkdir()  # Exclusive; stopped rosters are never resumed or replaced.
    started = time.time()
    terminal = dict(entry_sha256=panel.digest(entry_path), status="RUNNING",
                    started_unix=started, deadline_unix=started + 86400,
                    supervisor_pid=os.getpid(), observed_games=0,
                    supervisor_start_ticks=process_identity(os.getpid())["start_ticks"],
                    scientific_qualification="UNREVIEWED")
    panel.write(output / "launch.json", terminal)
    process = None
    try:
        env = {k: os.environ[k] for k in ("HOME", "LANG", "PATH", "TMPDIR") if k in os.environ}
        env.update(entry["environment"])
        argv = [entry["python"], "-B", str(Path(__file__).resolve()),
                "--entry", str(Path(entry_path).resolve()),
                "--review", str(Path(review_path).resolve()), "--worker"]
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(argv, cwd=entry["runtime"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            terminal.update(worker_pid=process.pid, argv=argv)
            panel.write(output / "launch.json", terminal)
            previous = 0
            last_progress = started
            reason = None
            while process.poll() is None:
                rows = prefix(output, entry)
                if len(rows) < previous:
                    raise ValueError("completed prefix shrank")
                if len(rows) > previous:
                    previous = len(rows)
                    last_progress = time.time()
                now = time.time()
                if now >= terminal["deadline_unix"]:
                    reason = "CALENDAR_CAP"
                    break
                limit = 1800 if previous else 900
                if now - last_progress >= limit:
                    reason = "NO_COMPLETION_TIMEOUT"
                    break
                time.sleep(5)
            code, cleanup = panel.stop_owned(process)
            process = None
            terminal.update(exit_code=code, cleanup_ok=cleanup)
            rows = prefix(output, entry)
            terminal["observed_games"] = len(rows)
            check(entry_path, review_path)
            if time.time() >= terminal["deadline_unix"]:
                reason = reason or "CALENDAR_CAP"
            terminal["status"] = reason or ("COLLECTION_FINISHED" if code == 0 and cleanup and len(rows) == 64 else "PROCESS_FAILURE")
    except BaseException as exc:
        terminal.update(status="SUPERVISOR_FAILURE", error_type=type(exc).__name__,
                        error=str(exc))
        traceback.print_exc()
    finally:
        if process is not None:
            code, cleanup = panel.stop_owned(process)
            terminal.update(exit_code=code, cleanup_ok=cleanup)
        terminal["ended_unix"] = time.time()
        panel.write(output / "terminal.json", terminal)
    return 0 if terminal["status"] == "COLLECTION_FINISHED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    def terminate(_signum, _frame):
        raise KeyboardInterrupt("supervisor terminated")
    if not args.worker:
        signal.signal(signal.SIGTERM, terminate)
    return (worker if args.worker else run)(args.entry, args.review)


if __name__ == "__main__":
    raise SystemExit(main())
