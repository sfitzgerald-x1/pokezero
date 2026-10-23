#!/usr/bin/env python3
"""One-shot public-root constructor diagnostic, never a strength collector.

Freeze eight raw-champion source prefixes before any constructor attempt. Keep
both perspectives at the first ordinary boundaries of turns4/12 (32 slots).
Expected construction refusals are observations, not replacement opportunities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_fullgame_panel as panel
from coordinate_fullgame_panels import process_identity


def slots(entry):
    start = entry["seed_start"]
    if (type(start) is not int or not 0 <= start <= 2**32 - 8
            or entry["seed_count"] != 8 or type(entry["seed_count"]) is not int
            or entry["turns"] != [4, 12] or any(type(t) is not int for t in entry["turns"])
            or any(type(entry[k]) is not int for k in ("calendar_seconds", "source_turn_cap",
                                                      "source_boundary_cap", "sampling_seconds"))
            or entry["calendar_seconds"] != 3600
            or entry["source_turn_cap"] != 20 or entry["source_boundary_cap"] != 256
            or entry["sampling_seconds"] != 10
            or type(entry["binding"]["joint_sleep_proposal_cap"]) is not int
            or entry["binding"]["joint_sleep_proposal_cap"] != 32
            or entry["binding"]["allow_earlier_compatible_template"] is not True):
        raise ValueError("diagnostic geometry differs from the fixed32-slot contract")
    return [dict(slot=i, seed=seed, turn=turn, seat=seat, status="UNAVAILABLE",
                 unavailable_reason="NOT_CAPTURED")
            for i, (seed, turn, seat) in enumerate(
                (s, t, p) for s in range(start, start + 8) for t in (4, 12) for p in ("p1", "p2"))]


def checked_entry(args):
    entry = json.loads(Path(args.entry).read_text())
    review = json.loads(Path(args.review).read_text())
    if (panel.digest(args.entry) != args.sha256 or review.get("status") != "CLEAR"
            or review.get("entry_sha256") != args.sha256
            or entry["input_hashes"].get(str(Path(__file__).resolve())) != panel.digest(__file__)
            or Path(entry["python"]).resolve() != Path(sys.executable).resolve()
            or entry["binding"]["checkpoint_sha256"] != entry["input_hashes"].get(entry["binding"]["checkpoint"])):
        raise ValueError("entry, review, instrument, interpreter or checkpoint binding mismatch")
    slots(entry)
    for helper in (panel.__file__, ROOT / "scripts" / "coordinate_fullgame_panels.py"):
        if entry["input_hashes"].get(str(Path(helper).resolve())) != panel.digest(helper):
            raise ValueError("owned lifecycle helper is not bound")
    panel.verify(entry)
    return entry


def root_seed(row):
    tag = f"reference-construction-v1:{row['seed']}:{row['turn']}:{row['seat']}".encode()
    return int.from_bytes(hashlib.sha256(tag).digest()[:16], "big")


def same_birth(expected, actual):
    return bool(expected and actual and actual["state"] not in ("Z", "X")
                and all(expected[k] == actual[k] for k in ("pid", "start_ticks")))


def owner_signal(signum, _frame):
    # Parent only: a requested stop must drain its actual owned process group.
    raise InterruptedError(f"owner received signal {signum}")


def validate_child_launch(args, launch):
    if (type(args.deadline) not in (int, float) or not math.isfinite(args.deadline)
            or launch["entry_sha256"] != args.sha256 or launch["deadline_unix"] != args.deadline
            or not same_birth(launch["identity"], process_identity(os.getpid()))
            or not same_birth(launch["owner_identity"], process_identity(os.getppid()))):
        raise ValueError("child is not the originally launched owned process")


def capture_sources(env, policy, rows, output, *, deadline):
    """No sampler/evaluator callback is accepted by this source-only phase."""
    from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
    from pokezero.mcts_eval.paper_reference_showdown import decision_state
    sources = []
    for seed in sorted({r["seed"] for r in rows}):
        if time.time() >= deadline:
            raise TimeoutError("global calendar cap")
        policy.reset()
        env.reset(seed=seed)
        rng = random.Random(seed)
        selected, trace = set(), []
        reason = "BOUNDARY_CAP"
        for boundary in range(256):
            if time.time() >= deadline:
                raise TimeoutError("global calendar cap")
            if env.terminal() is not None:
                reason = "SOURCE_TERMINAL"
                break
            requested = tuple(env.requested_players())
            if not requested or len(set(requested)) != len(requested):
                raise ValueError("invalid source request boundary")
            states = {p: env.public_materialization_state(p) for p in requested}
            turns = {state.replay.turn_number for state in states.values()}
            if len(turns) != 1:
                raise ValueError("source actor turn disagreement")
            turn = turns.pop()
            if turn > 20:
                reason = "TURN_CAP"
                break
            ordinary = (set(requested) == {"p1", "p2"} and all(
                state.self_request.get("active") and not state.self_request.get("forceSwitch")
                for state in states.values()))
            if ordinary and turn in (4, 12) and turn not in selected:
                for row in (r for r in rows if r["seed"] == seed and r["turn"] == turn):
                    request = PublicRootRequest.capture(states[row["seat"]], env.observe(row["seat"]))
                    data = pickle.dumps(request, protocol=5)
                    name = f"root-{row['slot']:02d}.pickle"
                    with (output / name).open("xb") as stream:
                        stream.write(data)
                    row.update(status="CAPTURED", root_file=name,
                               captured_unix=time.time(),
                               root_file_sha256=hashlib.sha256(data).hexdigest(),
                               root_sha256=decision_state(request.observation, player=row["seat"]).key.hex())
                    row.pop("unavailable_reason")
                panel.write(output / "progress.json", dict(phase="SOURCE_CAPTURE", slots=rows, sources=sources))
                selected.add(turn)
            actions = {p: policy.select_action(env.observe(p), rng=rng).action_index for p in requested}
            trace.append([turn, actions])
            env.step(actions)
        for row in rows:
            if row["seed"] == seed and row["status"] == "UNAVAILABLE":
                row["unavailable_reason"] = reason
        sources.append(dict(seed=seed, boundaries=len(trace), prefix_end=reason,
                            action_trace_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()))
        panel.write(output / "progress.json", dict(phase="SOURCE_CAPTURE", slots=rows, sources=sources))
    return sources


def sampling_counts(evidence):
    """Costs only: never copy raw error strings, sampled parties or requests."""
    if (type(evidence) is not dict or type(evidence.get("draws")) is not list
            or any(type(d) is not dict or type(d.get("joint_sleep_proposals")) is not list
                   for d in evidence["draws"])):
        raise ValueError("missing or malformed construction cost evidence")
    draws = evidence["draws"]
    proposals = [p for d in draws for p in d["joint_sleep_proposals"]]
    if any(type(p) is not dict or p.get("status") not in {"ACCEPTED", "REJECTED", "STARTED"}
           for p in proposals):
        raise ValueError("malformed proposal cost evidence")
    return dict(draws_started=len(draws), proposals_started=len(proposals),
                proposals_accepted=sum(p.get("status") == "ACCEPTED" for p in proposals),
                proposals_rejected=sum(p.get("status") == "REJECTED" for p in proposals),
                proposals_incomplete=sum(p.get("status") == "STARTED" for p in proposals))


def raised_by(error, module, function):
    """Exact innermost authored origin, not exception text supplied by a peer."""
    trace = error.__traceback__
    if trace is None:
        return False
    while trace.tb_next is not None:
        trace = trace.tb_next
    return (Path(trace.tb_frame.f_code.co_filename).resolve() == Path(module.__file__).resolve()
            and trace.tb_frame.f_code.co_name == function)


def failure_status(error, counts, *, deadline_at=None):
    from pokezero.bridge_error_sites import safe_bridge_error_reason
    from pokezero.local_showdown import LocalShowdownError
    from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
    from pokezero.mcts_eval import paper_reference_factory, reference_sleep_history, reference_sleep_joint
    if isinstance(error, SamplingDeadlineExceeded):
        if (deadline_at is not None and time.perf_counter() >= deadline_at and
                (raised_by(error, paper_reference_factory, "_check_deadline")
                or (error.__traceback__ is not None and
                    error.__traceback__.tb_frame.f_code is probe_root.__code__
                    and error.__traceback__.tb_next is None))):
            return "DEADLINE_INTERRUPTION"
    if isinstance(error, ReferenceRefusal):
        if (str(error) == "joint team/sleep proposal cap exhausted"
                and raised_by(error, paper_reference_factory, "__call__")
                and counts["proposals_rejected"] == 32 and counts["proposals_accepted"] == 0):
            return "PROPOSAL_CAP_EXHAUSTED"
        if (raised_by(error, reference_sleep_joint, "joint_sleep_proposal")
                and str(error).startswith(("joint sleep trait history unresolved:",
                                           "joint sleep evidence unresolved:"))):
            return "UNRESOLVED_EVIDENCE"
        if (raised_by(error, reference_sleep_joint, "joint_sleep_proposal")
                and str(error) == "joint sleep current ability is not represented by the materializer"):
            return "REFERENCE_REFUSAL"
        if (raised_by(error, reference_sleep_history, "public_sleep_history") and str(error) in {
                "sleeping switch lacks its public onset", "Yawn source identity is not public",
                "sleep onset lacks its public source identity", "sleep attempt has no direct sleep-usable move proof"}):
            return "UNRESOLVED_EVIDENCE"
    if isinstance(error, LocalShowdownError) and safe_bridge_error_reason(error.bridge_error_reason):
        return "TYPED_CONSTRUCTION_REFUSAL"
    # Unknown bridge failures may be I/O/provenance/instrument failures. Stop,
    # not a guessed expected refusal followed by another root.
    raise error


def probe_root(row, request, binding, *, runtime_factory=None):
    from pokezero.mcts_eval.paper_reference import SamplingDeadlineExceeded
    from pokezero.mcts_eval.paper_reference_parallel import _exception_code_sites
    from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
    from pokezero.mcts_eval.paper_reference_showdown import decision_state
    started, cpu, children = time.perf_counter(), time.process_time(), os.times()
    runtime, prepared, world = None, None, None
    evidence, status, sites = {}, None, []
    startup = None
    try:
        runtime = (runtime_factory or ShowdownWorkerFactory(**binding))(row["slot"])
        startup = time.perf_counter() - started
        deadline = time.perf_counter() + 10
        try:
            prepared = runtime.prepare(request)
            prepared.bind_sampling_deadline(deadline)
            world = prepared.sample_world(random.Random(root_seed(row)))
            # frame() also evaluates the opponent: explicitly do NOT call it
            # in this constructor-only diagnostic.
            if (world.subject != row["seat"] or
                    decision_state(world.env.observe(row["seat"]), player=row["seat"]) != prepared.root
                    or prepared.root.key.hex() != row["root_sha256"]):
                raise ValueError("constructed root differs from saved actor identity")
            if time.perf_counter() >= deadline:
                raise SamplingDeadlineExceeded("diagnostic construction deadline")
            status = "ROOT_PRESERVED"
        except Exception as error:
            # prepare has no sampler/RNG call. Only a closed, trusted-origin
            # preparation refusal can return; every other error is re-raised.
            evidence = prepared.evidence() if prepared is not None else {"draws": []}
            status = failure_status(error, sampling_counts(evidence), deadline_at=deadline)
            sites = _exception_code_sites(error)
        else:
            evidence = prepared.evidence()
        sampled = time.perf_counter()
    finally:
        try:
            if world is not None:
                world.close()
        finally:
            if runtime is not None:
                runtime.close()
    after = os.times()
    return dict(status=status, error_code_sites=sites, counts=sampling_counts(evidence),
                elapsed_wall_seconds=time.perf_counter() - started, startup_wall_seconds=startup,
                construction_wall_seconds=sampled - deadline + 10,
                construction_overrun_seconds=max(0., sampled - deadline),
                python_cpu_seconds=time.process_time() - cpu,
                reaped_direct_child_cpu_seconds=(after.children_user + after.children_system
                    - children.children_user - children.children_system))


def collect(args, entry):
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import load_transformer_policy
    output, rows = Path(entry["run_directory"]), slots(entry)
    # The private child mode cannot adopt another run or bypass its actual
    # still-live original owner. Wait only for the parent's launch receipt.
    limit = time.monotonic() + 5
    while not (output / "launch.json").exists() and time.monotonic() < limit:
        time.sleep(.02)
    launch = json.loads((output / "launch.json").read_text())
    validate_child_launch(args, launch)
    with (output / "child-start.json").open("x") as marker:
        json.dump(dict(entry_sha256=args.sha256, identity=launch["identity"]), marker)
    binding = entry["binding"]
    config = env_config_with_policy_spec_masks(LocalShowdownConfig(
        showdown_root=Path(binding["showdown_root"]), node_binary=binding["node_binary"],
        set_belief_source=True), [f"neural:{binding['checkpoint']}"], context="fixed public-root construction diagnostic")
    policy = load_transformer_policy(binding["checkpoint"], device="cpu", deterministic=True,
        exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
    with LocalShowdownEnv(config) as env:
        sources = capture_sources(env, policy, rows, output, deadline=args.deadline)
    # All source RNG/actions/prefixes are immutable before the first probe.
    panel.write(output / "source-capture.json", dict(slots=rows, sources=sources))
    checked_entry(args)
    for row in rows:
        if row["status"] != "CAPTURED":
            continue
        if time.time() >= args.deadline:
            raise TimeoutError("global calendar cap")
        if panel.digest(args.entry) != args.sha256:
            raise ValueError("entry changed")
        data = (output / row["root_file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["root_file_sha256"]:
            raise ValueError("captured root changed")
        # Only files this same one-shot process just captured, never untrusted
        # pickles or historical failed states supplied by another run.
        begun = time.perf_counter()
        row.update(status="CONSTRUCTION_STARTED", probe_started_unix=time.time())
        panel.write(output / "progress.json", dict(phase="CONSTRUCTION", slots=rows))
        try:
            row.update(probe_root(row, pickle.loads(data), binding))
        except BaseException as error:
            from pokezero.mcts_eval.paper_reference_parallel import _exception_code_sites
            row.update(status="INSTRUMENT_FAILURE_STOP", elapsed_wall_seconds=time.perf_counter() - begun,
                       error_code_sites=_exception_code_sites(error), failed_work_accounting_complete=False)
            panel.write(output / "progress.json", dict(phase="INSTRUMENT_FAILURE_STOP", slots=rows))
            raise
        panel.write(output / "progress.json", dict(phase="CONSTRUCTION", slots=rows))
    checked_entry(args)
    if time.time() >= args.deadline:
        raise TimeoutError("global calendar cap")
    panel.write(output / "result.json", dict(status="DIAGNOSTIC_COMPLETE", slots=rows,
        counts=dict(Counter(r["status"] for r in rows)), sources=sources,
        scope="32 fixed root slots; source prefixes only. No outcomes, strength or trajectory-fidelity inference."))


def main(args):
    entry = checked_entry(args)
    if args.child:
        collect(args, entry)
        return 0
    import fcntl
    with open("/tmp/pokezero-phase0-budget.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(entry["run_directory"])
        output.mkdir()  # Exclusive: no resume/retry or adoption of old roots.
        started = time.time()
        deadline = started + 3600
        panel.write(output / "manifest.json", dict(entry_sha256=args.sha256, slots=slots(entry)))
        argv = [sys.executable, "-B", __file__, "--entry", args.entry, "--review", args.review,
                "--sha256", args.sha256, "--child", "--deadline", str(deadline)]
        process, reason = None, "PROCESS_EXIT"
        received = []
        # Defer stop signals until Popen returns its real handle: raising from
        # inside Popen can lose a child created before assignment completes.
        previous = {sig: signal.signal(sig, lambda signum, _frame: received.append(signum))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (output / "worker.log").open("xb") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True, cwd=entry["runtime"],
                    env={**os.environ, **entry["environment"]})
            if received:
                owner_signal(received[0], None)
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError("child exited before actual birth identity capture")
            panel.write(output / "launch.json", dict(entry_sha256=args.sha256, identity=identity,
                owner_identity=process_identity(os.getpid()),
                argv=argv, started_unix=started, deadline_unix=deadline))
            while process.poll() is None:
                if received:
                    owner_signal(received[0], None)
                if time.time() >= deadline:
                    reason = "CALENDAR_CAP"
                    break
                time.sleep(1)
            if received:
                owner_signal(received[0], None)
            if time.time() >= deadline:
                reason = "CALENDAR_CAP"
        except BaseException:
            reason = "OWNER_FAILURE"
            raise
        finally:
            # Do not interrupt the bounded TERM/wait/KILL/reap sequence halfway.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if process is not None:
                    code, clean = panel.stop_owned(process)
                    panel.write(output / "terminal.json", dict(entry_sha256=args.sha256,
                        actual_child_exit=code, cleanup_ok=clean, reason=reason, ended_unix=time.time()))
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        return 0 if code == 0 and clean and reason == "PROCESS_EXIT" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    try:
        exit_code = main(arguments)
    except Exception as error:
        # Failures may include hypothetical private parties in their original
        # messages. Preserve only authored code sites, never a raw traceback.
        from pokezero.mcts_eval.paper_reference_parallel import _exception_code_sites
        print(json.dumps(dict(status="INSTRUMENT_FAILURE_STOP",
                              error_code_sites=_exception_code_sites(error))), file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)
