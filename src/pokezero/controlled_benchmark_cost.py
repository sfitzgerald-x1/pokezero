"""Own one controlled Foul Play attempt and read its measured, limited cost.

This is an execution consumer, not a scientific admission or registration tool.
wait4 measures the exact child plus descendants that child actually reaped. It
does NOT certify all descendants, the native extension, or complete workflow
cost. No retries, shell commands, raw fallbacks, or arbitrary-command surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


KIND = "pokezero.controlled-benchmark-cost.v1"
RECIPE = "pokezero.controlled-benchmark-recipe.v1"
_ROOT = Path(__file__).resolve().parents[2]
_FIELDS = {"kind", "python", "checkpoint", "showdown_root", "foulplay_root",
           "foulplay_python", "node", "policy_mode", "seat", "games", "seed_start",
           "max_decision_rounds", "timeout_seconds", "native", "input_sha256"}
_NATIVE_FIELDS = {"model", "tables", "depth", "sims", "batch", "worlds",
                  "workers", "cpuct", "opponent_priors"}


class CostRefusal(ValueError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise CostRefusal("duplicate JSON field")
            value[key] = item
        return value
    def constant(_):
        raise CostRefusal("nonfinite JSON value")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant)


def _create(path: Path, value: Any) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _path(value: Any) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise CostRefusal("recipe paths must be absolute")
    return Path(value)


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def validate_recipe(recipe: Any) -> dict:
    if not isinstance(recipe, dict) or set(recipe) != _FIELDS or recipe["kind"] != RECIPE:
        raise CostRefusal("closed benchmark recipe required")
    if recipe["policy_mode"] not in ("raw", "engine-mcts"):
        raise CostRefusal("reference remains unavailable without an owned cleanup handshake")
    if (recipe["seat"] not in ("p1", "p2")
            or not _integer(recipe["games"], 1, 100000)
            or not _integer(recipe["seed_start"], 0, 2**32 - recipe["games"])
            or not _integer(recipe["max_decision_rounds"], 1, 100000)
            or type(recipe["timeout_seconds"]) not in (int, float)
            or not math.isfinite(recipe["timeout_seconds"])
            or not 0 < recipe["timeout_seconds"] <= 7 * 86400):
        raise CostRefusal("invalid seat, fixed seed band, cap or timeout")
    required = {_path(recipe[name]).resolve() for name in
                ("python", "checkpoint", "foulplay_python", "node")}
    for name in ("python", "foulplay_python", "node"):
        if not os.access(_path(recipe[name]), os.X_OK):
            raise CostRefusal("recipe executable is not executable")
    required.update((_path(recipe["showdown_root"]) / "dist/sim/battle-stream.js",
                     _path(recipe["foulplay_root"]) / "run.py"))
    native = recipe["native"]
    if recipe["policy_mode"] == "raw":
        if native is not None:
            raise CostRefusal("raw recipe cannot retain native candidate settings")
    else:
        if not isinstance(native, dict) or set(native) != _NATIVE_FIELDS:
            raise CostRefusal("explicit native fixed-work recipe required")
        if (any(not _integer(native[k], 1, 1000000)
                for k in ("depth", "sims", "batch", "worlds", "workers"))
                or native["workers"] != 1
                or type(native["cpuct"]) not in (int, float)
                or not math.isfinite(native["cpuct"]) or native["cpuct"] <= 0
                or type(native["opponent_priors"]) is not bool):
            raise CostRefusal("invalid native fixed-work settings")
        required.update(_path(native[k]).resolve() for k in ("model", "tables"))
    pins = recipe["input_sha256"]
    if not isinstance(pins, dict) or not pins:
        raise CostRefusal("prospective input file bindings required")
    for name, digest in pins.items():
        path = _path(name)
        if (str(path.resolve()) != name or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or not path.is_file() or _hash(path) != digest):
            raise CostRefusal("input binding missing or changed")
    if not {str(p.resolve()) for p in required} <= set(pins):
        raise CostRefusal("required checkpoint, executable or entry file is not bound")
    return recipe


def bridge_arguments(recipe: dict, summary: Path) -> list[str]:
    """Only the fixed controlled benchmark; no comparison retry driver."""
    args = ["--checkpoint", recipe["checkpoint"], "--showdown-root", recipe["showdown_root"],
            "--foulplay-root", recipe["foulplay_root"], "--foulplay-python", recipe["foulplay_python"],
            "--node-binary", recipe["node"], "--policy-mode", recipe["policy_mode"],
            "--opponent-policy-mode", "foul-play", "--device", "cpu", "--temperature", "1",
            "--games", str(recipe["games"]), "--seed-start", str(recipe["seed_start"]),
            "--foulplay-random-seed", str(recipe["seed_start"]), "--search-time-ms", "1000",
            "--max-decision-rounds", str(recipe["max_decision_rounds"]),
            "--format", "gen3randombattle", "--pokezero-player", recipe["seat"],
            "--belief-set-source", "on", "--no-search-fallback", "--summary-out", str(summary)]
    if recipe["native"] is not None:
        native = recipe["native"]
        for key, flag in (("model", "--engine-model-path"), ("tables", "--engine-tables-path"),
                          ("depth", "--engine-depth"), ("sims", "--engine-sims"),
                          ("batch", "--engine-batch"), ("worlds", "--engine-worlds"),
                          ("workers", "--engine-model-world-workers"), ("cpuct", "--engine-c-puct")):
            args.extend((flag, str(native[key])))
        if native["opponent_priors"]:
            args.append("--engine-opponent-priors")
    return args


def _source_bindings() -> dict[str, str]:
    # Explicit source scope, not a loaded-extension/package qualification claim.
    paths = sorted((_ROOT / "src/pokezero").rglob("*.py"))
    paths += sorted((_ROOT / "scripts").glob("battle_bridge*.mjs"))
    return {str(p.resolve()): _hash(p) for p in paths}


def _same_bindings(bindings: dict[str, str]) -> bool:
    try:
        return all(_hash(Path(p)) == digest for p, digest in bindings.items())
    except OSError:
        return False


def _file_binding(path: Path) -> dict | None:
    return {"name": path.name, "sha256": _hash(path)} if path.is_file() else None


def _parent_pid(pid: int) -> int | None:
    # Targeted read-only identity check, not process-tree discovery.
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "ppid="],
                                capture_output=True, text=True, timeout=1, check=False)
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _clean_registered_groups(directory: Path, owner: int, sig: int) -> list[int]:
    """Signal only direct, still-owned session leaders while owner remains pinned.

    Caller MUST first confirm the owner is SIGSTOPped. That freezes its reaping
    threads, pins unreaped direct leaders and closes the identity-to-signal race.
    Escaped/unobserved scopes remain explicitly unqualified.
    """
    signalled = []
    path = directory / "children.jsonl"
    if not path.exists():
        return signalled
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            pid = record["pid"]
            if (type(pid) is int and pid > 1 and record["owner"] == owner
                    and record["new_session"] is True and os.getpgid(pid) == pid
                    and _parent_pid(pid) == owner):
                os.killpg(pid, sig)
                signalled.append(pid)
        except (OSError, KeyError, ValueError, TypeError):
            # A truncated last write never authorizes a guessed target.
            continue
    return signalled


def _stop_owner(owner: int) -> bool:
    os.kill(owner, signal.SIGSTOP)
    until = time.perf_counter() + 1
    while time.perf_counter() < until:
        event = os.waitid(os.P_PID, owner, os.WSTOPPED | os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if event is not None:
            return event.si_code == os.CLD_STOPPED
        time.sleep(.01)
    return False


def _terminal_event(event: Any) -> bool:
    # Some platforms return an outstanding STOP event even on a WEXITED query.
    # Non-None alone is not proof that wait4 can reap without blocking.
    return event is not None and event.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)


def _terminate_owned(directory: Path, owner: int, sig: int) -> list[int]:
    # The caller owns and has not reaped owner. Freeze ALL its threads before
    # inspecting child identity; do not signal a guessed or reparented group.
    stopped = _stop_owner(owner)
    try:
        if _terminal_event(os.waitid(os.P_PID, owner, os.WEXITED | os.WNOHANG | os.WNOWAIT)):
            return []  # No direct-child group signal after owner exit/reparenting.
        groups = _clean_registered_groups(directory, owner, sig) if stopped else []
        try:
            os.killpg(owner, sig)
        except (ProcessLookupError, PermissionError):
            if not _terminal_event(os.waitid(os.P_PID, owner, os.WEXITED | os.WNOHANG | os.WNOWAIT)):
                raise
        return groups
    finally:
        if stopped and sig != signal.SIGKILL:
            try:
                os.kill(owner, signal.SIGCONT)
            except ProcessLookupError:
                pass


def _launch_matches(directory: Path, recipe: dict, launch: dict) -> bool:
    return (launch.get("recipe_sha256") == _hash(directory / "recipe.json")
            and launch.get("argv") == bridge_arguments(recipe, directory / "summary.json"))


def _command(recipe: dict, directory: Path) -> list[str]:
    return [recipe["python"], "-m", "pokezero.controlled_benchmark_cost", "_worker", str(directory)]


def _supervise(recipe: dict, directory: Path) -> dict:
    for name in ("wait4", "waitid", "WNOWAIT"):
        if not hasattr(os, name):
            raise CostRefusal("exclusive wait4 with non-reaping terminal observation required")
    env = {k: v for k, v in os.environ.items() if not k.startswith("POKEZERO_")}
    env.update(PYTHONPATH=str(_ROOT / "src"), PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.perf_counter()
    interrupted = timed_out = forced_cleanup = False
    process = None
    groups = []
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        try:
            # A dedicated group in our session remains non-orphaned while we
            # own the child; freezing it must not trigger orphaned-group HUP.
            process = subprocess.Popen(_command(recipe, directory), cwd=_ROOT, env=env,
                                       stdout=stdout, stderr=stderr, process_group=0)
        except OSError as error:
            return {"pid": None, "returncode": None, "wait4_reaped": False,
                    "launch_error": type(error).__name__, "wall_seconds": time.perf_counter() - started,
                    "cpu_seconds": None, "timed_out": False, "interrupted": False}
        deadline = started + recipe["timeout_seconds"]
        termination_at = None
        try:
            while True:
                # Keep the child unreaped so its PID/group cannot be recycled.
                terminal = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if _terminal_event(terminal):
                    break
                now = time.perf_counter()
                if (directory / "cleanup-request.json").exists():
                    forced_cleanup = True
                    groups.extend(_terminate_owned(directory, process.pid, signal.SIGKILL))
                    continue
                if now >= deadline and termination_at is None:
                    timed_out = True
                    termination_at = now
                    groups.extend(_terminate_owned(directory, process.pid, signal.SIGTERM))
                if termination_at is not None and now >= termination_at + 3:
                    groups.extend(_terminate_owned(directory, process.pid, signal.SIGKILL))
                time.sleep(.02)
        except BaseException:
            # Interrupt/error cleanup does not release ownership before signalling.
            interrupted = True
            groups.extend(_terminate_owned(directory, process.pid, signal.SIGKILL))
        pid, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    return {"pid": pid, "returncode": process.returncode, "wait4_reaped": True,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "reported_maxrss": usage.ru_maxrss,
            "maxrss_units": "bytes" if sys.platform == "darwin" else "KiB",
            "maxrss_scope": "kernel child rusage; not physical process-tree peak",
            "timed_out": timed_out, "interrupted": interrupted,
            "forced_cleanup": forced_cleanup,
            "signalled_owned_session_leaders": sorted(set(groups))}


def worker_main(directory: Path) -> int:
    """Fixed child entry; wrappers observe subprocess ownership, never game data."""
    started = time.perf_counter()
    phase = "preflight"
    code, error_type = 1, None
    benchmark_seconds = None
    original = subprocess.Popen
    observed = []
    registry = (directory / "children.jsonl").open("x", encoding="utf-8")

    class ObservedPopen(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append((self, kwargs.get("start_new_session", False) is True))
            record = {"owner": os.getpid(), "pid": self.pid,
                      "new_session": kwargs.get("start_new_session", False) is True}
            registry.write(json.dumps(record, sort_keys=True) + "\n")
            registry.flush()
            os.fsync(registry.fileno())

    def interrupt(_sig, _frame):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    subprocess.Popen = ObservedPopen
    try:
        recipe = validate_recipe(_json(directory / "recipe.json"))
        manifest = _json(directory / "launch.json")
        if not _launch_matches(directory, recipe, manifest):
            raise CostRefusal("prospective launch recipe or arguments changed")
        if (not _same_bindings(manifest["sources"])
                or _source_bindings() != manifest["sources"]):
            raise CostRefusal("child import source differs from the frozen source inventory")
        from .foulplay_bridge import build_arg_parser, main as bridge_main
        args = bridge_arguments(recipe, directory / "summary.json")
        build_arg_parser().parse_args(args)  # Reject unsupported before construction.
        preflight_seconds = time.perf_counter() - started
        phase = "benchmark_call"
        benchmark_started = time.perf_counter()
        try:
            code = bridge_main(args)
            if type(code) is not int:
                raise CostRefusal("bridge returned no integer terminal code")
        finally:
            benchmark_seconds = time.perf_counter() - benchmark_started
        phase = "returned"
    except BaseException as error:
        code = error.code if isinstance(error, SystemExit) and type(error.code) is int else 1
        error_type = type(error).__name__
        preflight_seconds = locals().get("preflight_seconds", time.perf_counter() - started)
    finally:
        subprocess.Popen = original
        signal.signal(signal.SIGTERM, old_handler)
        registry.close()
        live = [process.pid for process, _ in observed if process.poll() is None]
        if live:
            phase, code, error_type = "cleanup", 1, "OwnedChildStillLive"
    _create(directory / "worker-terminal.json", {
        "kind": KIND, "pid": os.getpid(), "phase": phase, "returncode": code,
        "error_type": error_type, "preflight_seconds": preflight_seconds,
        "benchmark_call_seconds": benchmark_seconds,
        "benchmark_call_scope": "bridge call including its startup and teardown; not isolated phases",
        "startup_cpu_seconds": None, "cleanup_cpu_seconds": None,
        "descendant_cleanup_complete": False})
    if live:
        # Parent freezes this process (including any asyncio reaping threads)
        # before verifying/signalling direct session leaders. This process must
        # not exit/reparent them before that cleanup handshake.
        _create(directory / "cleanup-request.json", {"kind": KIND, "owner": os.getpid(), "pids": live})
        while True:
            time.sleep(.1)  # Parent owns a bounded timeout and kills this scope.
    return code


def _summary_reading(summary: Any, recipe: dict) -> dict:
    """Read actual bridge game rows without independently validating outcomes."""
    if not isinstance(summary, dict):
        raise CostRefusal("benchmark summary is not a mapping")
    expected = {"checkpoint_sha256": recipe["input_sha256"][str(_path(recipe["checkpoint"]).resolve())],
                "policy_mode": recipe["policy_mode"], "pokezero_player": recipe["seat"],
                "games": recipe["games"], "seed_start": recipe["seed_start"],
                "max_decision_rounds": recipe["max_decision_rounds"], "format_id": "gen3randombattle",
                "opponent_policy_id": "foul-play"}
    if any(type(summary.get(k)) is not type(v) or summary.get(k) != v for k, v in expected.items()):
        raise CostRefusal("summary recipe or checkpoint mismatch")
    root = summary.get("root_puct")
    if (not isinstance(root, dict) or root.get("foulplay_search_time_ms") != 1000
            or root.get("allow_search_fallback") is not False
            or summary.get("belief_set_source") is not True
            or summary.get("foulplay_random_seed") != recipe["seed_start"]):
        raise CostRefusal("summary opponent budget, fallback or belief drift")
    if recipe["native"] is not None:
        native = recipe["native"]
        block = summary.get("engine_mcts")
        expected_native = {"depth": native["depth"], "sims": native["sims"],
                           "batch": native["batch"], "worlds": native["worlds"],
                           "model_world_workers": native["workers"],
                           "opponent_priors": native["opponent_priors"]}
        if not isinstance(block, dict) or any(type(block.get(k)) is not type(v) or block.get(k) != v
                                               for k, v in expected_native.items()):
            raise CostRefusal("native summary differs from fixed work recipe")
    rows = summary.get("game_results")
    if (not isinstance(rows, list) or len(rows) > recipe["games"]
            or type(summary.get("completed_games")) is not int
            or summary["completed_games"] != len(rows)):
        raise CostRefusal("summary game count mismatch")
    cells = []
    for index, row in enumerate(rows):
        seed = recipe["seed_start"] + index
        if (not isinstance(row, dict) or type(row.get("seed")) is not int or row["seed"] != seed
                or any(type(row.get(k)) is not bool for k in ("capped", "tied", "pokezero_won"))
                or (row["tied"] and row["pokezero_won"])):
            raise CostRefusal("summary row not in exact original seed prefix")
        score = 1.0 if row["pokezero_won"] else .5 if row["tied"] else 0.0
        if (type(row.get("pokezero_score")) not in (int, float)
                or not math.isfinite(row["pokezero_score"])
                or (not row["capped"] and row["pokezero_score"] != score)):
            raise CostRefusal("summary row score mismatch")
        cells.append({"seed": seed, "seat": recipe["seat"],
                      "reported_score": None if row["capped"] else score,
                      "capped": row["capped"], "independently_validated": False,
                      "scientific_score_interval": [0., 1.]})
    cells.extend({"seed": recipe["seed_start"] + index, "seat": recipe["seat"],
                  "reported_score": None, "capped": None, "independently_validated": False,
                  "scientific_score_interval": [0., 1.]}
                 for index in range(len(rows), recipe["games"]))
    complete = (len(rows) == recipe["games"] and summary.get("complete") is True
                and summary.get("status") == "complete" and not any(r["capped"] for r in rows))
    return {"reported_games": len(rows), "uncapped_complete_summary": complete,
            "cells": cells, "policy_timing": summary.get("policy_timing"),
            "search_statistics": summary.get("engine_mcts"),
            "native_cpuct_binding": "launch arguments only; no native summary field",
            "scientific_outcomes_accepted": 0}


def read_attempt(directory: Path) -> dict:
    """Exact durable readback, no polling, scoring admission, mutation or retry."""
    terminal = _json(directory / "terminal.json")
    if not isinstance(terminal, dict) or terminal.get("kind") != KIND:
        raise CostRefusal("missing original terminal receipt")
    for name, digest in terminal["artifact_sha256"].items():
        if Path(name).name != name or _hash(directory / name) != digest:
            raise CostRefusal("bound attempt file changed or missing")
    launch = _json(directory / "launch.json")
    recipe = _json(directory / "recipe.json")
    validate_recipe(recipe)
    if not _launch_matches(directory, recipe, launch):
        raise CostRefusal("prospective launch recipe or arguments changed")
    sources_ok = _same_bindings(launch["sources"]) and _source_bindings() == launch["sources"]
    if not sources_ok:
        raise CostRefusal("benchmark source drift after launch")
    child = terminal["child"]
    worker = _json(directory / "worker-terminal.json") if terminal["worker_terminal"] else None
    if worker is not None and (worker.get("pid") != child.get("pid")
                              or (worker.get("returncode") != child.get("returncode")
                                  and not (child.get("forced_cleanup") or child.get("timed_out")
                                           or child.get("interrupted")))):
        raise CostRefusal("worker and owned terminal receipt disagree")
    reading, summary_error = None, None
    if terminal["summary"] is not None:
        try:
            reading = _summary_reading(_json(directory / "summary.json"), recipe)
        except (CostRefusal, ValueError, TypeError, KeyError) as error:
            summary_error = type(error).__name__
    child_complete = (child.get("wait4_reaped") is True and child.get("returncode") == 0
                      and not child.get("timed_out") and not child.get("interrupted")
                      and not child.get("forced_cleanup")
                      and worker is not None and worker.get("phase") == "returned"
                      and worker.get("error_type") is None
                      and reading is not None and reading["uncapped_complete_summary"])
    return {"kind": KIND, "status": "MEASURED_CHILD" if child_complete else "REFUSED",
            "child": child, "worker": worker, "summary": reading,
            "summary_error": summary_error, "input_file_bindings_match": True,
            "source_bindings_match": True, "cpu_scope": "exact child plus descendants it actually reaped",
            "complete_workflow_cost": False, "descendant_cleanup_complete": False,
            "loaded_native_extension_qualified": False, "scientific_qualified": False,
            "scientific_outcomes_accepted": 0, "phase_b_authorized": False,
            "retries": 0, "raw_fallback": False}


def run_attempt(recipe: dict, directory: Path) -> dict:
    validate_recipe(recipe)
    # Preflight refusal never allocates/reuses a producer directory.
    if not all(hasattr(os, name) for name in ("wait4", "waitid", "WNOWAIT")):
        raise CostRefusal("POSIX exclusive child ownership required")
    sources = _source_bindings()
    directory = directory.resolve()
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    _create(directory / "recipe.json", recipe)
    _create(directory / "launch.json", {"kind": KIND, "sources": sources,
                                        "recipe_sha256": _hash(directory / "recipe.json"),
                                        "argv": bridge_arguments(recipe, directory / "summary.json"),
                                        "environment_scope": "exact checkout PYTHONPATH, no POKEZERO overrides, CPU thread limits 1"})
    child = _supervise(recipe, directory)
    files = ("recipe.json", "launch.json", "worker-terminal.json", "summary.json",
             "children.jsonl", "cleanup-request.json", "stdout.log", "stderr.log")
    _create(directory / "terminal.json", {
        "kind": KIND, "child": child,
        "worker_terminal": _file_binding(directory / "worker-terminal.json"),
        "summary": _file_binding(directory / "summary.json"),
        "artifact_sha256": {name: _hash(directory / name) for name in files
                            if (directory / name).is_file()}})
    return read_attempt(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute one separately authorized fixed benchmark")
    run.add_argument("--recipe", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    read = commands.add_parser("read", help="Read an existing original terminal attempt")
    read.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return worker_main(args.directory)
    try:
        result = (run_attempt(_json(args.recipe), args.output) if args.command == "run"
                  else read_attempt(args.output))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"benchmark cost refused: {type(error).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "MEASURED_CHILD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
