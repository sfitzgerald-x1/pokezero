#!/usr/bin/env python3
"""Coordinate two reviewed, isolated full-game partitions without a restart path.

The controller runs where kubectl is already authenticated. Workers receive a
short lease and the same absolute deadline. Losing the controller stops work;
it never authorizes a replacement controller, clock, pod, or stopped cell.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SPEC = importlib.util.spec_from_file_location(
    "fullgame_panel", Path(__file__).with_name("run_fullgame_panel.py"))
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)

PRIMARY_KIND = "foulplay_matched_control"
HEAD_TO_HEAD_KIND = "raw_head_to_head"
PRIMARY_SHARDS = [["candidate-p1", "p1"], ["raw-p1", "p1"],
                  ["candidate-p2", "p2"], ["raw-p2", "p2"]]
HEAD_TO_HEAD_SHARDS = [["candidate-p1", "p1"], ["candidate-p2", "p2"]]


def head_to_head_contract(entry):
    """Derive the explicit secondary geometry from actual worker arguments.

    Legacy primary entries keep their original four-shard contract. A label
    alone cannot turn a Foul Play/control experiment into a head-to-head.
    """
    kind = entry.get("comparison_kind", PRIMARY_KIND)
    if kind == PRIMARY_KIND:
        return {}
    if kind != HEAD_TO_HEAD_KIND:
        raise ValueError("unknown comparison kind")
    if [[s["name"], s["seat"]] for s in entry["shards"]] != HEAD_TO_HEAD_SHARDS:
        raise ValueError("head-to-head requires both candidate seats and no raw-control shards")

    def argument(argv, flag):
        if not isinstance(argv, list) or any(not isinstance(token, str) for token in argv):
            raise ValueError("worker arguments must be string arrays")
        # argparse accepts abbreviated long flags and --flag=value. Refuse
        # those aliases here rather than certify one value while it runs another.
        if any(token.startswith("--") and flag.startswith(token.split("=", 1)[0])
               and token != flag for token in argv):
            raise ValueError("head-to-head critical arguments must use exact separate flags")
        if argv.count(flag) != 1:
            raise ValueError("head-to-head critical argument missing or repeated")
        index = argv.index(flag)
        if index + 1 == len(argv) or argv[index + 1].startswith("--"):
            raise ValueError("head-to-head critical argument lacks a value")
        return argv[index + 1]

    modes = []
    for shard in entry["shards"]:
        common, specific = entry["common_arguments"], shard["arguments"]
        if not isinstance(common, list) or not isinstance(specific, list):
            raise ValueError("worker arguments must be string arrays")
        argv = common + specific
        if argument(argv, "--opponent-policy-mode") != "raw":
            raise ValueError("head-to-head opponent must actually be raw")
        mode = argument(argv, "--policy-mode")
        if mode not in {"engine-mcts", "trajectory-reference"}:
            raise ValueError("head-to-head candidate must be a registered search mode")
        modes.append(mode)
    if len(set(modes)) != 1:
        raise ValueError("head-to-head candidate mode differs across seats")
    return dict(comparison_kind=kind, candidate_policy_mode=modes[0], opponent_policy_mode="raw")


def read(path):
    return json.loads(Path(path).read_text())


def reviewed(entry_path, review_path, expected=None):
    entry_hash = panel.digest(entry_path)
    review = read(review_path)
    if (review.get("status") != "CLEAR" or review.get("entry_sha256") != entry_hash
            or (expected is not None and expected != entry_hash)):
        raise ValueError("exact entry lacks independent CLEAR review")
    return read(entry_path), entry_hash


def process_identity(pid):
    """Linux birth identity, not a PID-only liveness claim."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])
    except FileNotFoundError:
        return None


def alive(identity):
    current = process_identity(identity["pid"])
    return bool(current and current["start_ticks"] == identity["start_ticks"]
                and current["state"] not in ("Z", "X"))


def snapshot(entry, parent, cursors):
    output = Path(entry["run_directory"])
    documents = {}
    for name in ("launch.json", "progress.json", "terminal.json"):
        path = output / name
        if path.exists():
            documents[name] = read(path)
    terminal = documents.get("terminal.json")
    counts = dict(cursors)
    for shard in entry["shards"]:
        name = shard["name"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError("unsafe shard name")
        for suffix in ("-launch.json", "-progress.json", "-terminal.json"):
            path = output / (name + suffix)
            if path.exists():
                documents[path.name] = read(path)
        path = output / (name + ".json")
        if path.exists():
            count = panel.observed_count(path, entry["seed_start"], entry["seed_count"])
            if count < cursors.get(name, 0):
                raise ValueError("completed prefix shrank")
            if count > cursors.get(name, 0) or terminal is not None:
                document = read(path)
                if document["completed_games"] < count:
                    raise ValueError("completed prefix shrank during export")
                documents[path.name] = document
                counts[name] = document["completed_games"]
    launcher = parent / "coordinator-launch.json"
    outer_exit = parent / "coordinator-exit.json"
    launch = read(launcher) if launcher.exists() else None
    exit_receipt = read(outer_exit) if outer_exit.exists() else None
    live = alive(launch["identity"]) if launch and "identity" in launch else False
    return dict(documents=documents, counts=counts, launcher=launch,
                supervisor_exit=exit_receipt, launcher_alive=live)


def agent(action, entry_path, review_path, expected, deadline, cpus, cursors,
          pod_uid=None, node_name=None):
    # Downward-API identities close the get-pod/exec name-reuse race, including
    # aborts: a replacement pod must refuse before touching any owned file.
    if (not pod_uid or not node_name
            or os.environ.get("POKEZERO_WORKER_POD_UID") != pod_uid
            or os.environ.get("POKEZERO_WORKER_NODE_NAME") != node_name):
        raise ValueError("reviewed worker pod UID/node does not match downward API")
    entry, entry_hash = reviewed(entry_path, review_path, expected)
    parent = Path(entry_path).parent
    for source in (Path(__file__).resolve(), Path(panel.__file__).resolve()):
        if entry.get("input_hashes", {}).get(str(source)) != panel.digest(source):
            raise ValueError("coordinator/supervisor source is not bound by worker entry")
    if action == "preflight":
        panel.verify(entry)
        if entry.get("coordination") is None:
            raise ValueError("worker entry must require a coordinator")
        if Path(entry["run_directory"]).exists() or (parent / "coordinator-launch.json").exists():
            raise ValueError("worker partition already launched; no restart")
        return dict(entry_sha256=entry_hash, seed_start=entry["seed_start"],
                    seed_count=entry["seed_count"], calendar_seconds=entry["calendar_seconds"],
                    shards=[[s["name"], s["seat"]] for s in entry["shards"]],
                    **head_to_head_contract(entry))
    if action == "abort":
        panel.write(entry["coordination"]["abort_path"],
                    dict(entry_sha256=entry_hash, reason="GLOBAL_STOP", time_unix=time.time()))
        return dict(abort_recorded=True)
    if action == "observe":
        # Deliberately does not renew a lease after a global stop.
        return snapshot(entry, parent, cursors)
    if action == "refresh":
        if deadline is None:
            raise ValueError("missing common deadline")
        panel.write(entry["coordination"]["lease_path"],
                    dict(entry_sha256=entry_hash, deadline_unix=deadline))
        reason = panel.coordination_reason(entry, entry_hash, deadline, time.time())
        if reason is not None:
            raise ValueError(reason)
        return snapshot(entry, parent, cursors)
    if action == "start":
        reason = panel.coordination_reason(entry, entry_hash, deadline, time.time())
        if reason is not None or time.time() >= deadline:
            raise ValueError(reason or "CALENDAR_CAP")
        # The exclusive marker survives transport loss. Never retry this action.
        with (parent / "coordinator-launch.json").open("x") as marker:
            json.dump(dict(status="LAUNCHING", entry_sha256=entry_hash), marker)
        argv = ["/usr/bin/taskset", "-c", cpus, sys.executable, "-B", str(Path(__file__).resolve()),
                "--agent", "wait", "--entry", str(entry_path), "--review", str(review_path),
                "--expected-sha256", entry_hash, "--deadline-unix", str(deadline),
                "--pod-uid", pod_uid, "--node-name", node_name]
        with (parent / "coordinator-launch.log").open("xb") as log:
            worker = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, stdin=subprocess.DEVNULL)
        identity = process_identity(worker.pid)
        if identity is None:
            raise RuntimeError("launcher exited before birth identity was recorded")
        receipt = dict(identity=identity, argv=argv, entry_sha256=entry_hash,
                       deadline_unix=deadline, started_unix=time.time())
        panel.write(parent / "coordinator-launch.json", receipt)
        return receipt
    if action == "wait":
        argv = [sys.executable, "-B", str(Path(__file__).with_name("run_fullgame_panel.py")),
                "--entry", str(entry_path), "--review", str(review_path),
                "--deadline-unix", str(deadline)]
        result = subprocess.run(argv, check=False)
        panel.write(parent / "coordinator-exit.json", dict(
            entry_sha256=entry_hash, exit_code=result.returncode, ended_unix=time.time(),
            deadline_unix=deadline))
        return dict(exit_code=result.returncode)
    raise ValueError("unknown agent action")


class Transport:
    def __init__(self):
        self.denied = {}

    def prefix(self, target):
        return ["kubectl", "--context", target["context"], "--namespace", target["namespace"]]

    def execute(self, target, argv):
        if target["name"] in self.denied:
            raise PermissionError(self.denied[target["name"]])
        try:
            return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=15)
        except subprocess.CalledProcessError as error:
            detail = error.stderr or ""
            if "Forbidden" in detail or "Unauthorized" in detail:
                message = (f"kubectl access denied for {target['context']}/{target['namespace']}/"
                           f"{target['pod']}: {detail[:1000]}")
                self.denied[target["name"]] = message
                raise PermissionError(message) from error
            raise

    def preflight_pod(self, target):
        result = self.execute(target, [*self.prefix(target), "get", "pod", target["pod"], "-o", "json"])
        pod = json.loads(result.stdout)
        if (pod["metadata"]["uid"] != target["pod_uid"]
                or pod["spec"]["nodeName"] != target["node_name"]
                or pod["status"]["phase"] != "Running"):
            raise ValueError("reviewed pod identity/node/state changed")
        container = next(c for c in pod["spec"]["containers"] if c["name"] == target["container"])
        if (container["image"] != target["image"] or pod["spec"]["restartPolicy"] != "Never"
                or container["resources"]["limits"]["cpu"] != target["cpu_limit"]
                or container["resources"]["limits"]["memory"] != target["memory_limit"]):
            raise ValueError("reviewed worker image/resources/restart contract changed")

    def call(self, target, action, deadline=None, cursors=None):
        self.preflight_pod(target)
        argv = [*self.prefix(target), "exec", target["pod"], "-c", target["container"], "--",
                target["python"], "-B", target["agent"], "--agent", action,
                "--entry", target["entry"], "--review", target["review"],
                "--expected-sha256", target["entry_sha256"],
                "--cpus", target["cpus"], "--cursors", json.dumps(cursors or {}),
                "--pod-uid", target["pod_uid"], "--node-name", target["node_name"]]
        if deadline is not None:
            argv += ["--deadline-unix", str(deadline)]
        result = self.execute(target, argv)
        return json.loads(result.stdout)


def validate_partitions(entry, preflights):
    targets = entry["targets"]
    if len(targets) != 2 or len(preflights) != 2 or len({t["name"] for t in targets}) != 2:
        raise ValueError("exactly two named partitions required")
    if len({t["node_name"] for t in targets}) != 2 or len({t["pod_uid"] for t in targets}) != 2:
        raise ValueError("partitions require different reviewed pods and nodes")
    if any(not re.fullmatch(r"[a-z][a-z0-9-]*", t["name"]) for t in targets):
        raise ValueError("unsafe partition name")
    if entry["calendar_seconds"] != 86400:
        raise ValueError("one 24-hour common clock required")
    kind = entry.get("comparison_kind", PRIMARY_KIND)
    if kind not in {PRIMARY_KIND, HEAD_TO_HEAD_KIND}:
        raise ValueError("unknown comparison kind")
    expected_shards = HEAD_TO_HEAD_SHARDS if kind == HEAD_TO_HEAD_KIND else PRIMARY_SHARDS
    candidate_modes = set()
    seeds = []
    for target, facts in zip(targets, preflights):
        if facts["entry_sha256"] != target["entry_sha256"] or facts["calendar_seconds"] != 86400:
            raise ValueError("partition binding/cap changed")
        if facts.get("comparison_kind", PRIMARY_KIND) != kind or facts["shards"] != expected_shards:
            raise ValueError("partition must retain its registered comparison and both seats")
        if kind == HEAD_TO_HEAD_KIND:
            if (facts.get("opponent_policy_mode") != "raw"
                    or facts.get("candidate_policy_mode") not in {"engine-mcts", "trajectory-reference"}):
                raise ValueError("head-to-head worker modes differ from the registered comparison")
            candidate_modes.add(facts["candidate_policy_mode"])
        seeds += list(range(facts["seed_start"], facts["seed_start"] + facts["seed_count"]))
    planned = list(range(entry["seed_start"], entry["seed_start"] + entry["seed_count"]))
    if sorted(seeds) != planned:
        raise ValueError("partitions do not exactly cover the fixed roster")
    if len(candidate_modes) > 1:
        raise ValueError("head-to-head candidate mode differs across partitions")


def save_snapshot(output, target, state):
    directory = output / target["name"]
    directory.mkdir(exist_ok=True)
    for name, document in state["documents"].items():
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("unsafe exported filename")
        panel.write(directory / name, document)
    panel.write(directory / "coordinator-snapshot.json", snapshot_metadata(state))


def snapshot_metadata(state):
    return {**{key: value for key, value in state.items() if key != "documents"},
            "exported_files": sorted(state["documents"])}


def cleanup_confirmed(state):
    terminal = state["documents"].get("terminal.json")
    return bool(state["supervisor_exit"] is not None and terminal is not None
                and not state["launcher_alive"] and terminal.get("cleanup_ok", True)
                and all(shard.get("cleanup_ok") is True for shard in terminal.get("shards", [])))


def drain(transport, targets, deadline, cursors, output, receipt):
    """Four bounded, non-renewing observations; unreachable remains uncertain."""
    pending = list(targets)
    for attempt in range(4):
        if not pending:
            break
        remaining = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = [(target, pool.submit(transport.call, target, "observe", deadline,
                                          cursors[target["name"]])) for target in pending]
            for target, future in calls:
                name = target["name"]
                try:
                    state = future.result()
                    save_snapshot(output, target, state)
                    receipt["states"][name] = snapshot_metadata(state)
                    cursors[name] = state["counts"]
                    if not cleanup_confirmed(state):
                        remaining.append(target)
                except Exception as error:
                    receipt.setdefault("observation_errors", {})[name] = type(error).__name__
                    remaining.append(target)
        pending = remaining
        panel.write(output / "progress.json", receipt)
        if pending and attempt < 3:
            # Cover lease expiry (<=60s), the worker's 5s poll and owned cleanup
            # without renewing scientific work. Unreachable still stays unknown.
            time.sleep(35)
    receipt["cleanup_verified"] = not pending
    receipt["cleanup_unverified_partitions"] = [target["name"] for target in pending]


def run(entry_path, review_path, transport=None):
    entry, entry_hash = reviewed(entry_path, review_path)
    if entry.get("coordinator_sha256") != panel.digest(__file__):
        raise ValueError("controller source is not bound by its reviewed entry")
    transport = transport or Transport()
    targets = entry["targets"]
    # All setup/provenance happens before the common clock or any launch.
    preflights = []
    for target in targets:
        transport.preflight_pod(target)
        preflights.append(transport.call(target, "preflight"))
    validate_partitions(entry, preflights)
    output = Path(entry["run_directory"])
    output.mkdir()  # Exclusive; no replacement coordinator/resume.
    started = time.time()
    deadline = started + entry["calendar_seconds"]
    receipt = dict(status="RUNNING", entry_sha256=entry_hash,
                   started_unix=started, deadline_unix=deadline, states={})
    panel.write(output / "launch.json", receipt)
    cursors = {t["name"]: {} for t in targets}
    complete = set()
    assertion = (subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
                 if sys.platform == "darwin" else None)
    try:
        for target in targets:
            transport.call(target, "refresh", deadline)
        for target in targets:
            receipt.setdefault("launches", {})[target["name"]] = transport.call(target, "start", deadline)
            panel.write(output / "launch.json", receipt)
        while len(complete) != len(targets):
            if panel.digest(entry_path) != entry_hash:
                raise RuntimeError("coordinator entry changed")
            if time.time() >= deadline:
                raise RuntimeError("CALENDAR_CAP")
            active = [t for t in targets if t["name"] not in complete]
            # Parallel only across isolated nodes; no overlapping games per node.
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [(target, pool.submit(transport.call, target, "refresh", deadline,
                                               cursors[target["name"]])) for target in active]
                for target, future in futures:
                    state = future.result()
                    save_snapshot(output, target, state)
                    receipt["states"][target["name"]] = snapshot_metadata(state)
                    cursors[target["name"]] = state["counts"]
                    exit_receipt = state["supervisor_exit"]
                    terminal = state["documents"].get("terminal.json")
                    if terminal is not None and terminal["status"] != "COLLECTION_ATTEMPT_COMPLETE":
                        raise RuntimeError("PARTITION_FAILURE")
                    if exit_receipt is not None:
                        if (exit_receipt["exit_code"] != 0 or terminal is None
                                or exit_receipt["deadline_unix"] != deadline
                                or terminal["deadline_unix"] != deadline
                                or terminal["ended_unix"] > deadline):
                            raise RuntimeError("PARTITION_FAILURE")
                        complete.add(target["name"])
                    elif not state["launcher_alive"]:
                        raise RuntimeError("PARTITION_PROCESS_LOST")
            panel.write(output / "progress.json", receipt)
            if len(complete) != len(targets):
                time.sleep(10)
        if panel.digest(entry_path) != entry_hash:
            raise RuntimeError("coordinator entry changed")
        receipt["status"] = "COLLECTION_ATTEMPT_COMPLETE"
    except BaseException as error:
        receipt.update(status="COORDINATOR_STOP", error_type=type(error).__name__, error=str(error))
        # A failed transport never licenses retry/start. Lease expiry remains the
        # fallback if an abort RPC cannot get through; cleanup is not presumed.
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = [(target, pool.submit(transport.call, target, "abort", deadline))
                     for target in targets if target["name"] not in complete]
            for target, future in calls:
                try:
                    receipt.setdefault("abort_receipts", {})[target["name"]] = future.result()
                except Exception as abort_error:
                    receipt.setdefault("abort_errors", {})[target["name"]] = type(abort_error).__name__
        drain(transport, [t for t in targets if t["name"] not in complete],
              deadline, cursors, output, receipt)
    finally:
        receipt.update(ended_unix=time.time(), scientific_qualification="UNREVIEWED")
        panel.write(output / "terminal.json", receipt)
        if assertion is not None:
            assertion.terminate()
            assertion.wait(timeout=10)
    return 0 if receipt["status"] == "COLLECTION_ATTEMPT_COMPLETE" else 1


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"coordinator received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--agent", choices=("preflight", "refresh", "observe", "start", "wait", "abort"))
    parser.add_argument("--expected-sha256")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--cpus", default="0-13")
    parser.add_argument("--cursors", default="{}")
    parser.add_argument("--pod-uid")
    parser.add_argument("--node-name")
    args = parser.parse_args()
    if args.agent:
        result = agent(args.agent, args.entry, args.review, args.expected_sha256,
                       args.deadline_unix, args.cpus, json.loads(args.cursors),
                       pod_uid=args.pod_uid, node_name=args.node_name)
        print(json.dumps(result), flush=True)
    else:
        with open("/tmp/pokezero-phase0-budget.lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(run(args.entry, args.review))
