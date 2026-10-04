#!/usr/bin/env python3
"""Create-only bounded frozen-root profile, never a game-strength PASS.

Requires the actual attested ARM64 image runtime and a fresh output identity.
Writes the registration before selecting any action. Each owned root worker has
a hard process-group timeout; completed arm units survive a later timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

if __name__ == "__main__":
    # Root exec inherits the parent's brief ownership-publication signal mask.
    # Release it before project imports or any model initialization.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})

PROCESS_STARTED_UNIX = time.time()
PROCESS_STARTED_MONOTONIC = time.monotonic()
ACTIVE_ROOT_PROCESS = None
ACTIVE_STUDY_PROCESS = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mcts_mcts_h2h import _source_provenance, _showdown_source_provenance
from engine_build_fingerprint import assert_fresh, compute_fingerprint
from pokezero.collection import env_config_with_policy_spec_masks
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.lattice import materialize_search_artifacts
from pokezero.mcts_eval.manifest import SearchConfig
from pokezero.mcts_eval.policy_opponent_live_replay import qualify_live_replay
from pokezero.mcts_eval.policy_opponent_profile import ARMS, MODES, profile_root
from pokezero.mcts_eval.policy_opponent_registration import (
    BUDGET, CHAMPION_SHA256, ROSTER_SHA256, SHOWDOWN_SHA256, freeze_registration,
    canonical_sha256, load_registration, remaining_study_seconds, wall_statistics,
)
from pokezero.mcts_eval.policy_opponent_roster import load_frozen_roster, verify_source_files
from pokezero.mcts_eval.policy_opponent_source_prefixes import load_source_records
from pokezero.mcts_eval.resolver import ContractError, resolve_checkpoint_contract, sha256_file


def immutable_json(path: Path, payload: object) -> str:
    """Complete/fsync an owned temporary then link without replacing anything."""
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)  # Only this call's owned staging file.
    return hashlib.sha256(raw).hexdigest()


def load_image_receipt(args):
    """Refuse malformed bindings before importing/loading runtime consumers."""
    raw = Path(args.image_receipt).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.image_receipt_sha256:
        raise ContractError("image receipt byte hash drift")
    receipt = json.loads(raw)
    if not isinstance(receipt, dict) or not isinstance(receipt.get("model_runtime"), dict):
        raise ContractError("source image/model runtime receipt must be objects")
    model = receipt.get("model_runtime", {})
    image = receipt.get("immutable_image", "")
    if (receipt.get("schema_version") != "pokezero.b2-source-image-receipt.v8"
            or receipt.get("complete") is not True
            or not isinstance(image, str) or re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) is None
            or receipt.get("image_digest") != "sha256:" + image.rsplit("@sha256:", 1)[-1]
            or model.get("schema_version") != "pokezero.b2-model-runtime.v4"
            or model.get("copied_runtime_probe") is not True
            or model.get("system_site_packages") is not True):
        raise ContractError("source image/model runtime receipt is incomplete")
    source = model.get("source")
    if (not isinstance(source, dict)
            or set(source) != {"commit", "tree_sha256", "tree_status"}
            or re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit"))) is None
            or receipt.get("source_commit") != source["commit"]
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("tree_sha256"))) is None
            or source["tree_status"] != "clean_tracked_checkout"
            or model.get("runtime_path") != "/opt/pokezero-b2-model-runtime"
            or not isinstance(model.get("python_version"), str)
            or not isinstance(model.get("torch"), dict)
            or not isinstance(model["torch"].get("version"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(model["torch"].get("module_sha256"))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(model.get("engine_fingerprint"))) is None):
        raise ContractError("image receipt source/interpreter/Torch binding is malformed")
    extensions = model.get("native_extensions")
    if (not isinstance(extensions, list) or len(extensions) != 2
            or any(not isinstance(row, dict) for row in extensions)
            or any(not isinstance(row.get("module"), str) for row in extensions)
            or {row["module"] for row in extensions} != {"poke_engine", "pokezero_search"}
            or any(not isinstance(row.get("relative_path"), str)
                   or Path(row["relative_path"]).is_absolute()
                   or ".." in Path(row["relative_path"]).parts
                   or not row["relative_path"].endswith(".so")
                   or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None for row in extensions)):
        raise ContractError("image receipt lacks valid bindings for both native consumers")
    return receipt


def verify_image_runtime(args, *, configure_threads=True):
    receipt = load_image_receipt(args)
    model = receipt["model_runtime"]
    source = _source_provenance()
    if (source != model.get("source") or source["commit"] != receipt.get("source_commit")
            or platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64")
            or str(Path(sys.prefix).resolve()) != model.get("runtime_path")
            or platform.python_version() != model.get("python_version")):
        raise ContractError("active source/platform/interpreter differs from image receipt")
    import torch
    if configure_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ContractError("active Torch thread allocation differs from registration")
    if model.get("torch") != {"version": torch.__version__, "module_sha256": sha256_file(torch.__file__)}:
        raise ContractError("active Torch runtime differs from image receipt")
    extensions = model["native_extensions"]
    prefix = Path(sys.prefix).resolve()
    for row in extensions:
        module = importlib.import_module(row["module"])
        artifacts = list(Path(module.__file__).parent.glob("*.so"))
        expected = (prefix / row["relative_path"]).resolve()
        if (len(artifacts) != 1 or not expected.is_relative_to(prefix)
                or artifacts[0].resolve() != expected or sha256_file(expected) != row["sha256"]):
            raise ContractError("active native extension differs from image receipt")
    assert_fresh()
    fingerprint = compute_fingerprint()["fingerprint"]
    if fingerprint != model.get("engine_fingerprint"):
        raise ContractError("active native source fingerprint differs from image receipt")
    quota = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    if (len(quota) != 2 or not all(value.isdigit() for value in quota)
            or int(quota[1]) <= 0 or int(quota[0]) != int(quota[1]) * BUDGET["cpu_per_runner"]):
        raise ContractError("profile requires the registered two-CPU cgroup quota")
    showdown = _showdown_source_provenance(args.showdown_root)
    if showdown["content_sha256"] != SHOWDOWN_SHA256:
        raise ContractError("active battle oracle differs from frozen historical runtime")
    return receipt, source, fingerprint, showdown


def load_execution_evidence(out, args, roster):
    """External pins bind registration, study clock and the complete replay panel."""
    registration = load_registration(out / "REGISTRATION.json",
        expected_sha256=args.registration_sha256, roster=roster)
    raw = (out / "STUDY_START.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.study_start_sha256:
        raise ContractError("study-start receipt byte hash drift")
    start = json.loads(raw)
    if (not isinstance(start, dict) or start.get("registration_sha256") != args.registration_sha256
            or type(start.get("reserved_cpus")) is not int
            or start["reserved_cpus"] != BUDGET["cpu_per_runner"]
            or start.get("includes_initialization") is not True):
        raise ContractError("study-start registration/resource binding drift")
    remaining_study_seconds(study_started_unix_s=start.get("study_started_unix_s"),
        now_unix_s=time.time(), prior_cpu_hours=0., reserved_cpus=start["reserved_cpus"])
    raw = (out / "LIVE_REPLAY.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != start.get("live_replay_sha256"):
        raise ContractError("worker replay qualification bytes drift")
    preflight = json.loads(raw)
    if (not isinstance(preflight, dict)
            or preflight.get("schema_version") != "pokezero.paper-policy-opponent-live-replay.v1"
            or type(preflight.get("root_denominator")) is not int or preflight["root_denominator"] != 32
            or preflight.get("replacement_roots") != []
            or not isinstance(preflight.get("roots"), list) or len(preflight["roots"]) != 32):
        raise ContractError("live replay panel denominator/identity drift")
    for public, source in zip(preflight["roots"], roster["profile_roots"]):
        if not isinstance(public, dict) or public.get("decision_id") != source["decision_id"]:
            raise ContractError("worker source/preflight root identity drift")
        if public.get("state") == "LIVE_PUBLIC_REPLAY_VALID":
            if public.get("requested_players") not in ([source["seat"]], ["p1", "p2"]):
                raise ContractError("live replay request boundary drift")
        elif public.get("state") != "REFUSED" or not public.get("phase") or not public.get("refusal"):
            raise ContractError("live replay disposition is malformed")
    return registration, preflight, start


def validate_terminal(unit, *, source, registration_sha):
    if (not isinstance(unit, dict) or unit.get("registration_sha256") != registration_sha
            or canonical_sha256(unit.get("source")) != canonical_sha256(source)):
        raise ContractError("root worker terminal identity differs from registered source")
    result = unit.get("result")
    if (not isinstance(result, dict) or result.get("decision_id") != source["decision_id"]
            or result.get("state") not in {"COMPLETE", "REFUSED", "CAP_NOT_RUN", "CAP_UNRESOLVED"}
            or not isinstance(result.get("modes"), dict)):
        raise ContractError("root worker terminal disposition/decision identity is malformed")
    modes = result["modes"]
    if modes:
        if set(modes) != set(MODES) or any(not isinstance(rows, dict) or set(rows) != set(ARMS)
                                         for rows in modes.values()):
            raise ContractError("root worker terminal arm/mode panel is incomplete")
        if any(not isinstance(row, dict) or row.get("state") not in {"COMPLETE", "REFUSED"}
               for rows in modes.values() for row in rows.values()):
            raise ContractError("root worker terminal arm disposition is malformed")
    complete = bool(modes) and all(row["state"] == "COMPLETE" for rows in modes.values() for row in rows.values())
    if (result["state"] == "COMPLETE") != complete:
        raise ContractError("root worker COMPLETE contradicts its arm evidence")
    return result


def communicate_bounded(process, timeout):
    """Only call for this launcher's new-session child, never an existing process."""
    try:
        return _communicate_bounded(process, timeout)
    except BaseException:
        # Covers interruptions during initial observation AND termination grace.
        stop_owned_process(process)
        raise


def _communicate_bounded(process, timeout):
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        for sig, grace in ((signal.SIGTERM, BUDGET["cleanup_grace_seconds"]), (signal.SIGKILL, None)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass  # The owned process group exited between observation and signal.
            try:
                stdout, stderr = process.communicate(timeout=grace)
                return stdout, stderr, True
            except subprocess.TimeoutExpired:
                continue
    raise ContractError("owned worker did not terminate after kill")


def stop_owned_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=BUDGET["cleanup_grace_seconds"])
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def terminate_owned_root(signum, _frame):
    # The root child has its own session for per-root deadlines. Cascade the
    # outer study timeout into that session, so it cannot escape the hard cap.
    if ACTIVE_ROOT_PROCESS is not None and ACTIVE_ROOT_PROCESS.poll() is None:
        try:
            os.killpg(ACTIVE_ROOT_PROCESS.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            ACTIVE_ROOT_PROCESS.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(128 + signum)


def launch_root_process(command):
    global ACTIVE_ROOT_PROCESS
    # Do not let SIGTERM land between creating a separate process group and
    # publishing its ownership to the outer-deadline cleanup handler.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        ACTIVE_ROOT_PROCESS = subprocess.Popen(command, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    return ACTIVE_ROOT_PROCESS


def terminate_owned_study(signum, _frame):
    if ACTIVE_STUDY_PROCESS is not None:
        # Graceful cascade first: the study owns the separately supervised root.
        stop_owned_process(ACTIVE_STUDY_PROCESS)
    raise SystemExit(128 + signum)


def record_timeout(out, *, ordinal, source, registration_sha):
    result = {"state": "CAP_UNRESOLVED", "decision_id": source["decision_id"], "modes": {}}
    path = out / f"roots/{ordinal:02d}.json"
    if not path.exists():
        try:
            immutable_json(path, {"registration_sha256": registration_sha, "source": source, "result": result})
        except FileExistsError:
            pass  # A late terminal may race the timeout; it must not be replaced.
    immutable_json(out / f"caps/{ordinal:02d}.json", {"state": "CAP_UNRESOLVED",
        "registration_sha256": registration_sha, "decision_id": source["decision_id"],
        "partial_arm_receipts_retained": True})


def bound_inputs(args, checkpoint):
    receipt, source, engine, showdown = verify_image_runtime(args)
    roster = load_frozen_roster(args.roster, expected_sha256=ROSTER_SHA256)
    verify_source_files(args.source_root, roster)
    contract = resolve_checkpoint_contract(checkpoint, expected_sha256=CHAMPION_SHA256,
        showdown_root=args.showdown_root, showdown_source_sha256=SHOWDOWN_SHA256)
    return roster, contract, dict(source_commit=source["commit"], source_tree_sha256=source["tree_sha256"],
        engine_fingerprint=engine, immutable_image=receipt["immutable_image"],
        source_receipt_sha256=args.image_receipt_sha256, checkpoint_sha256=CHAMPION_SHA256,
        observation_contract_sha256=contract.observation_contract_sha256,
        showdown_runtime_sha256=showdown["content_sha256"])


def worker(args):
    out = Path(args.out_root)
    roster, contract, runtime = bound_inputs(args, out / "inputs/champion.pt")
    registration, preflight, _ = load_execution_evidence(out, args, roster)
    artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
    runtime.update(model_sha256=sha256_file(artifacts["model_path"]), tables_sha256=sha256_file(artifacts["tables_path"]))
    if runtime != registration["expected_runtime"]:
        raise ContractError("worker runtime differs from frozen registration")
    ordinal = args.worker_ordinal
    if not 0 <= ordinal < 32:
        raise ContractError("worker ordinal outside frozen denominator")
    source = roster["profile_roots"][ordinal]
    public = preflight["roots"][ordinal]
    if public["decision_id"] != source["decision_id"]:
        raise ContractError("worker source/preflight root identity drift")
    if public["state"] != "LIVE_PUBLIC_REPLAY_VALID":
        result = {"state": "REFUSED", "decision_id": source["decision_id"], "phase": public["phase"],
                  "reason": public["refusal"], "modes": {}}
    else:
        records = load_source_records(args.source_root, roster)
        def persist(mode, arm, row):
            immutable_json(out / f"measurements/{ordinal:02d}/{mode}-{arm}.json",
                {"registration_sha256": args.registration_sha256, "decision_id": source["decision_id"],
                 "mode": mode, "arm": arm, "measurement": row})
        result = profile_root(records[source["source_relative_path"]], source_records=tuple(records.values()),
            contract=contract, showdown_root=args.showdown_root, config=SearchConfig(**registration["config"]),
            seed=registration["seed"], root_ordinal=ordinal,
            source_requested_players=public["requested_players"],
            deadline_ms=registration["timing"]["deadline_ms"],
            native_batch_guard_ms=registration["timing"]["native_batch_guard_ms"], on_row=persist)
    verify_source_files(args.source_root, roster)
    if (_source_provenance()["tree_sha256"] != runtime["source_tree_sha256"]
            or _showdown_source_provenance(args.showdown_root)["content_sha256"] != SHOWDOWN_SHA256
            or sha256_file(out / "inputs/champion.pt") != CHAMPION_SHA256
            or sha256_file(artifacts["model_path"]) != runtime["model_sha256"]
            or sha256_file(artifacts["tables_path"]) != runtime["tables_sha256"]):
        raise ContractError("worker inputs drift during profiling")
    load_execution_evidence(out, args, roster)
    unit = {"registration_sha256": args.registration_sha256, "source": source, "result": result}
    validate_terminal(unit, source=source, registration_sha=args.registration_sha256)
    immutable_json(out / f"roots/{ordinal:02d}.json", unit)


def run(args):
    global ACTIVE_ROOT_PROCESS
    out = Path(args.out_root)
    if out.exists() or out.is_symlink():
        raise ContractError("fresh create-only profile output root already exists")
    roster, _, runtime = bound_inputs(args, args.checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    immutable_json(out / "OWNERSHIP.json", {"token": args.ownership_token})
    (out / "inputs").mkdir()
    shutil.copyfile(args.checkpoint, out / "inputs/champion.pt")
    contract = resolve_checkpoint_contract(out / "inputs/champion.pt", expected_sha256=CHAMPION_SHA256,
        showdown_root=args.showdown_root, showdown_source_sha256=SHOWDOWN_SHA256)
    artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
    runtime.update(model_sha256=sha256_file(artifacts["model_path"]), tables_sha256=sha256_file(artifacts["tables_path"]))
    config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=args.showdown_root,
        set_belief_source=True), [f"neural:{contract.checkpoint_path}"], context="registered paper root profile")
    with LocalShowdownEnv(config) as env:
        preflight = qualify_live_replay(args.roster, expected_roster_sha256=ROSTER_SHA256,
                                       source_root=args.source_root, env=env)
    preflight_sha = immutable_json(out / "LIVE_REPLAY.json", preflight)
    registration = freeze_registration(roster, roster_sha256=ROSTER_SHA256, expected_runtime=runtime)
    registration_sha = immutable_json(out / "REGISTRATION.json", registration)
    start_sha = immutable_json(out / "STUDY_START.json", {"study_started_unix_s": PROCESS_STARTED_UNIX,
        "registration_sha256": registration_sha, "live_replay_sha256": preflight_sha,
        "reserved_cpus": BUDGET["cpu_per_runner"], "includes_initialization": True})
    rows = []
    for ordinal, source in enumerate(roster["profile_roots"]):
        elapsed = time.monotonic() - PROCESS_STARTED_MONOTONIC
        remaining = min(BUDGET["study_elapsed_seconds"] - elapsed,
            remaining_study_seconds(study_started_unix_s=PROCESS_STARTED_UNIX, now_unix_s=time.time(),
                prior_cpu_hours=elapsed * BUDGET["cpu_per_runner"] / 3600,
                reserved_cpus=BUDGET["cpu_per_runner"]))
        timeout = min(BUDGET["per_root_worker_seconds"], remaining) - BUDGET["cleanup_grace_seconds"]
        if timeout <= 0:
            result = {"state": "CAP_NOT_RUN", "decision_id": source["decision_id"], "modes": {}}
            immutable_json(out / f"roots/{ordinal:02d}.json", {"registration_sha256": registration_sha,
                           "source": source, "result": result})
        else:
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                       "--worker-ordinal", str(ordinal), "--registration-sha256", registration_sha]
            command += ["--study-start-sha256", start_sha]
            process = launch_root_process(command)
            try:
                stdout, stderr, timed_out = communicate_bounded(process, timeout)
            finally:
                ACTIVE_ROOT_PROCESS = None
            if timed_out:
                record_timeout(out, ordinal=ordinal, source=source, registration_sha=registration_sha)
            immutable_json(out / f"logs/{ordinal:02d}.json", {"exit_code": process.returncode,
                "stdout_last_200": stdout.decode(errors="replace").splitlines()[-200:],
                "stderr_last_200": stderr.decode(errors="replace").splitlines()[-200:]})
            path = out / f"roots/{ordinal:02d}.json"
            if not path.is_file():
                raise ContractError(f"root worker {ordinal} exited {process.returncode} without terminal evidence")
            unit = json.loads(path.read_bytes())
            result = validate_terminal(unit, source=source, registration_sha=registration_sha)
            if timed_out:
                result = {"state": "CAP_UNRESOLVED", "decision_id": source["decision_id"], "modes": {}}
            if process.returncode != 0 and not timed_out:
                raise ContractError(f"root worker {ordinal} exited nonzero: {process.returncode}")
        rows.append(result)
    args.registration_sha256, args.study_start_sha256 = registration_sha, start_sha
    load_execution_evidence(out, args, roster)
    receipt, source, engine, showdown = verify_image_runtime(args, configure_threads=False)
    if (source["commit"] != runtime["source_commit"] or source["tree_sha256"] != runtime["source_tree_sha256"]
            or engine != runtime["engine_fingerprint"] or receipt["immutable_image"] != runtime["immutable_image"]
            or sha256_file(out / "inputs/champion.pt") != CHAMPION_SHA256
            or sha256_file(artifacts["model_path"]) != runtime["model_sha256"]
            or sha256_file(artifacts["tables_path"]) != runtime["tables_sha256"]):
        raise ContractError("parent runtime/input binding drift before readout")
    verify_source_files(args.source_root, roster)
    cells = {}
    for mode in MODES:
        cells[mode] = {}
        for arm in ARMS:
            measured = [r.get("modes", {}).get(mode, {}).get(arm) for r in rows]
            complete = [r for r in measured if r is not None and r["state"] == "COMPLETE"]
            cells[mode][arm] = {**wall_statistics([r["decision_wall_seconds"] for r in complete]),
                "root_denominator": 32, "uncompleted": 32 - len(complete),
                "productive_iterations": sum(r["telemetry"]["total_iterations"] for r in complete)}
    elapsed = time.monotonic() - PROCESS_STARTED_MONOTONIC
    immutable_json(out / "PROFILE_READOUT.json", {"status": "DIAGNOSTIC_COMPLETE_NOT_STRENGTH_PASS",
        "registration_sha256": registration_sha, "root_denominator": 32,
        "root_dispositions": [{"decision_id": r["decision_id"], "state": r["state"]} for r in rows],
        "cells": cells, "study_started_unix_s": PROCESS_STARTED_UNIX,
        "profile_elapsed_seconds": elapsed, "reserved_profile_cpu_hours": elapsed * 2 / 3600,
        "budget_exceeded": elapsed > BUDGET["study_elapsed_seconds"] or elapsed * 2 / 3600 > BUDGET["study_cpu_hours"],
        "remaining_shared_seconds": max(0., 7200 - elapsed), "replacement_roots": [],
        "timed_out_arm_receipts": sorted(str(path.relative_to(out)) for path in (out / "measurements").glob("*/*.json")
            if rows[int(path.parent.name)]["state"] == "CAP_UNRESOLVED"),
        "partial_receipts_are_diagnostic_only": True, "advance_authorized": False,
        "strength_qualified": False, "complete_paper_reproduction": False})


def supervise(args):
    """Bound initialization/preflight as well as roots; no stalled export escape."""
    global ACTIVE_STUDY_PROCESS
    out = Path(args.out_root)
    if out.exists() or out.is_symlink():
        raise ContractError("fresh create-only profile output root already exists")
    token = uuid.uuid4().hex
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
        "--study-worker", "--ownership-token", token, "--study-started-unix", str(PROCESS_STARTED_UNIX)]
    timeout = BUDGET["study_elapsed_seconds"] - (time.monotonic() - PROCESS_STARTED_MONOTONIC) - BUDGET["cleanup_grace_seconds"]
    if timeout <= 0:
        raise ContractError("study cap expired before initialization")
    signal.signal(signal.SIGTERM, terminate_owned_study)
    signal.signal(signal.SIGINT, terminate_owned_study)
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        ACTIVE_STUDY_PROCESS = subprocess.Popen(command, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    process = ACTIVE_STUDY_PROCESS
    try:
        stdout, stderr, timed_out = communicate_bounded(process, timeout)
    finally:
        ACTIVE_STUDY_PROCESS = None
    # Only an output root positively owned by this child may receive its terminal.
    owned = out / "OWNERSHIP.json"
    if not owned.is_file() or json.loads(owned.read_bytes()) != {"token": token}:
        raise ContractError(f"study exited {process.returncode} without an owned output root; "
                            f"last error: {stderr.decode(errors='replace').splitlines()[-3:]}")
    immutable_json(out / "PROCESS_TERMINAL.json", {"exit_code": process.returncode,
        "state": "CAP_UNRESOLVED" if timed_out else "EXIT_ZERO" if process.returncode == 0 else "FAILED",
        "includes_initialization": True, "strength_qualified": False,
        "stdout_last_200": stdout.decode(errors="replace").splitlines()[-200:],
        "stderr_last_200": stderr.decode(errors="replace").splitlines()[-200:]})
    if timed_out or process.returncode != 0:
        raise ContractError(f"study {'reached hard cap' if timed_out else 'failed'}; exit code {process.returncode}")
    if not (out / "PROFILE_READOUT.json").is_file():
        raise ContractError("study exit zero is missing complete diagnostic readout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("roster", "source-root", "checkpoint", "showdown-root", "image-receipt",
                  "image-receipt-sha256", "out-root"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--worker-ordinal", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--registration-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--study-start-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--study-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ownership-token", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--study-started-unix", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.study_started_unix is not None:
            global PROCESS_STARTED_UNIX, PROCESS_STARTED_MONOTONIC
            PROCESS_STARTED_UNIX = args.study_started_unix
            elapsed = time.time() - PROCESS_STARTED_UNIX
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ContractError("study clock moved backwards")
            PROCESS_STARTED_MONOTONIC = time.monotonic() - elapsed
        if args.worker_ordinal is not None:
            worker(args)
        elif args.study_worker:
            if not args.ownership_token:
                raise ContractError("study worker requires supervisor ownership token")
            signal.signal(signal.SIGTERM, terminate_owned_root)
            signal.signal(signal.SIGINT, terminate_owned_root)
            run(args)
        else:
            supervise(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"PROFILE REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
