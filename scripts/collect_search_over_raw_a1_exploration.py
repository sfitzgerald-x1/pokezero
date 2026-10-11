"""One reviewed A1 exploration collection on a fixed 400-root Phase A design.

Only the 200 exploration slots / 32 source seeds are executable here. Both
search arms retain public and diagnostic-oracle configurations, one-second
nominal selection budgets and unchanged worker/world allocations. Actual
startup, selection overshoot and cleanup costs are not an equal-compute claim.
The 200 held-out slots remain unopened; A2/A3 and inference admission remain
separate unfinished requirements. No failed root/seed is replaced or retried.
"""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import qualify_search_over_raw_a1 as a1
from qualify_search_over_raw_opening import prepare_binding, verify_inputs
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest, phase_a_contract, require)
from pokezero.mcts_eval.search_over_raw_ledger import PhaseALedger
from pokezero.mcts_eval.resolver import sha256_file


ROOT = a1.ROOT
SCHEMA = "pokezero.search-over-raw.a1-exploration.v1"
NAMESPACE = "7e105338-51b1-44fd-8e0d-9fd526493b4c"
ATTEMPT_WALL_SECONDS = 14400
CLEANUP_GRACE_SECONDS = 10
COMPLETE = "COMPLETE_A1_EXPLORATION_COLLECTION_ONLY"
REVIEW = "CLEAR_ONE_FIXED_A1_EXPLORATION_ATTEMPT"


def exposure_inventory(paths):
    require(paths, "explicit prior exposure registrations required")
    excluded = set(ENGINEERING_EXCLUDED_SEEDS) | set(range(4711, 4736)) | set(a1.SEEDS)
    bindings = {}
    for supplied in paths:
        path = Path(supplied).resolve()
        value = json.loads(path.read_text())
        seeds = value.get("seeds")
        require(type(seeds) is list and seeds and all(type(s) is int and 0 <= s < 2**32
            for s in seeds), "exposure requires explicit uint32 seeds")
        excluded.update(seeds)
        bindings[str(path)] = sha256_file(path)
    return sorted(excluded), bindings


def cohort(excluded):
    configs = [SearchConfiguration("raw", seconds=1.)] + [
        SearchConfiguration(arm, belief, leaf, seconds, 20 if arm == "reference" else 1)
        for arm in ("incumbent", "reference") for belief in ("public", "oracle")
        for leaf in ("model", "hp_fraction", "raw_rollout") for seconds in (1., 3., 10.)]
    c = phase_a_contract(NAMESPACE, excluded_seeds=excluded, configurations=configs,
        roots=400, seeds_per_panel=32)
    c.update(exclude_opening_requests=True, execution_ready=False,
        remaining_gates=["finish A1/A2/A3 exploration and direct checks",
            "freeze both deployable selections and valid useful inference before holdout admission"],
        validation_inference_not_yet_admitted=True,
        analytical_extension="Existing fixed-panel betting specification is not admitted for holdout; sensitivity remains unresolved. Exploration bootstrap is descriptive only.")
    return c


def source_contract(c):
    # No validation seeds or validation execution path enters the collector.
    return dict(namespace=c["namespace"], candidate_seat="p1", exclude_opening_requests=True,
        panels={"exploration": c["panels"]["exploration"]}, continuation_replicates=8,
        source_policy="raw_argmax_both_seats", source_root_rule=c["source_root_rule"],
        max_source_boundaries=250, max_continuation_boundaries=250,
        continuation_policy=c["continuation_policy"], replacements_permitted=False,
        retry_authorized=False, attempt_wall_seconds=ATTEMPT_WALL_SECONDS,
        cleanup_grace_seconds=CLEANUP_GRACE_SECONDS,
        stop_policy="first failed/capped source, selection, continuation, provenance or worker stops attempt; completed short sources leave explicit missing slots",
        timing_policy="one second nominal selection only; all measured startup, selection overshoot and cleanup retained; not strict one-second or equal CPU",
        owned_receipts_protocol="active-generations-v1; immutable generation launch/terminal; at most20active identities queried once within2s",
        holdout_opened=False, phase_a_admission=False, phase_b_authorized=False)


def bound_files():
    return list(dict.fromkeys([Path(__file__).resolve(), *a1.bound_files(),
        ROOT / "src/pokezero/mcts_eval/search_over_raw_ledger.py",
        ROOT / "src/pokezero/mcts_eval/search_over_raw_statistics.py"]))


def register(*, output, exposure_registrations, plan, **kwargs):
    excluded, bindings = exposure_inventory(exposure_registrations)
    c = cohort(excluded)
    plan = Path(plan).resolve()
    c["input_hashes"] = dict(bindings, **{str(plan): sha256_file(plan)})
    r = prepare_binding(**kwargs, seconds=1.)
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT), "evidence output must be outside executing checkout")
    r.update(schema=SCHEMA, status="REGISTERED_A1_EXPLORATION_NOT_COLLECTED", namespace=NAMESPACE,
        seeds=c["panels"]["exploration"]["seeds"], fixture_seed=c["panels"]["exploration"]["seeds"][0],
        configurations=[asdict(cfg) for cfg in a1.configurations()], phase_a_cohort=c,
        phase_a_cohort_sha256=digest(c), source_contract=source_contract(c),
        attempt_directory=str(output), exposure_registrations=bindings,
        excluded_seeds=excluded, excluded_seed_inventory_sha256=digest(excluded), plan=str(plan),
        required_independent_review=True, qualifies_uninstrumented_runtime=False,
        owned_receipts_protocol="active-generations-v1",
        representative_runtime_evidence=False, historical_attempt_reused=False,
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False,
        holdout_opened=False, exposure_scope="supplied registrations plus explicit minimum exclusions; independently verify completeness",
        scope="200 fixed A1 exploration slots; full 400-root two-panel design retained; no holdout, A2/A3 completion or strength admission",
        required_diagnostics="both arms actual attempted original-team traits; truth joined only in controller")
    r["input_sha256"].update(c["input_hashes"])
    r["input_sha256"].update({str(p): sha256_file(p) for p in bound_files()})
    validate_contract(r, output)
    verify_inputs(r)
    output.mkdir(parents=True, exist_ok=False)
    a1.save_new(output / "registration.json", r)
    a1.save_new(output / "registration-binding.json", dict(registration_sha256=digest(r)))
    PhaseALedger.create(output / "phase-a-ledger", c)
    return r


def validate_contract(r, output):
    excluded, bindings = exposure_inventory(list(r["exposure_registrations"]))
    c = cohort(excluded)
    c["input_hashes"] = dict(bindings, **{r["plan"]: sha256_file(r["plan"])})
    require(r["schema"] == SCHEMA and r["namespace"] == NAMESPACE
        and r["phase_a_cohort"] == c and r["phase_a_cohort_sha256"] == digest(c)
        and r["source_contract"] == source_contract(c)
        and r["seeds"] == c["panels"]["exploration"]["seeds"]
        and r["fixture_seed"] == r["seeds"][0]
        and r["configurations"] == [asdict(cfg) for cfg in a1.configurations()]
        and r["candidate_seat"] == "p1" and type(r["initial_dispatch_workers"]) is int
        and r["initial_dispatch_workers"] == 6 and r["required_independent_review"] is True
        and r["owned_receipts_protocol"] == "active-generations-v1"
        and all(r[field] is False for field in ("retry_authorized", "phase_a_admission",
            "phase_b_authorized", "scientific_strength_evidence", "holdout_opened",
            "qualifies_uninstrumented_runtime", "representative_runtime_evidence", "historical_attempt_reused")),
        "exploration core/cohort drift")
    require(r["excluded_seeds"] == excluded and r["excluded_seed_inventory_sha256"] == digest(excluded)
        and r["exposure_registrations"] == bindings
        and all(r["input_sha256"].get(p) == s for p, s in c["input_hashes"].items()), "exploration exposure/plan drift")
    require(r["source_root"] == str(ROOT) and r["attempt_directory"] == str(Path(output).resolve())
        and not Path(output).resolve().is_relative_to(ROOT)
        and all(r["input_sha256"].get(str(p)) == sha256_file(p) for p in bound_files()),
        "exploration driver/output drift")


def new_progress(r):
    contract = r["source_contract"]
    return dict(stage="runtime_import", source_games_completed=0, roots_completed=0,
        selections_completed=0, continuations_completed=0,
        fixed_roster=a1.planned_cells(contract, "exploration"),
        continuation_roster=a1.planned_continuations(contract, "exploration"),
        source_roster={str(seed): dict(status="UNSTARTED_UNCERTAIN") for seed in r["seeds"]})


def admit(output, review):
    output, review = Path(output).resolve(), Path(review).resolve()
    r = json.loads((output / "registration.json").read_text())
    require(json.loads((output / "registration-binding.json").read_text()) == dict(registration_sha256=digest(r)),
        "exploration registration binding drift")
    validate_contract(r, output)
    verify_inputs(r)
    ledger = PhaseALedger(output / "phase-a-ledger")
    ledger.verify()
    require(ledger.contract_sha256 == r["phase_a_cohort_sha256"]
        and not any((ledger.directory / name).exists() for name in
            ("selection.json", "validation-claim.json", "validation-result.json", "validation-failure.json")),
        "holdout/selection already claimed")
    clearance = json.loads(review.read_text())
    require(clearance.get("disposition") == REVIEW and clearance.get("registration_sha256") == digest(r)
        and clearance.get("source_commit") == r["source_commit"] and clearance.get("independent_reviewer")
        and clearance.get("scientific_admission") is False and clearance.get("holdout_authorized") is False,
        "bound independent exploration review required")
    require(not (output / "attempt.json").exists() and not (output / "terminal.json").exists(),
        "claimed/closed exploration attempt; no retry")
    return output, review, r


def run(output, review):
    output, review, r = admit(output, review)
    supervision = json.loads((output / "supervision.json").read_text())
    require(supervision["supervisor_pid"] == os.getppid() and supervision["registration_sha256"] == digest(r)
        and supervision["review_sha256"] == sha256_file(review), "owned exploration supervisor required")
    a1.save_new(output / "attempt.json", dict(pid=os.getpid(), registration_sha256=digest(r),
        review_sha256=sha256_file(review), retry_authorized=False, status="CLAIMED_BEFORE_RUNTIME"))
    started = time.perf_counter()
    p = new_progress(r)
    snapshots = 0
    def progress_sink(value):
        nonlocal snapshots
        snapshots += 1
        a1.save_new(output / f"progress-{snapshots:04d}.json", dict(
            **{key: value[key] for key in ("stage", "source_games_completed", "roots_completed",
                "selections_completed", "continuations_completed")}, recorded_unix_seconds=time.time(),
            elapsed_seconds=time.perf_counter()-started, registration_sha256=digest(r),
            validation="collection progress; independent evidence validation not yet complete"))
    code, error = 0, None
    try:
        a1.measure(r, output, p, source_contract_value=r["source_contract"], panel="exploration",
            completion_status="COMPLETE_EXPLORATION", progress_sink=progress_sink)
        verify_inputs(r)
        a1.verify_completion(output, p, contract=r["source_contract"], panel="exploration",
            completion_status="COMPLETE_EXPLORATION")
        ledger = PhaseALedger(output / "phase-a-ledger")
        for cfg in a1.configurations()[1:]:
            intervals = {cell["root_id"]: cell["contrast_interval"] for cell in p["fixed_roster"]
                if cell["configuration"] == cfg.identity and cell["status"] == "COMPLETE_EXPLORATION"}
            ledger.record_exploration(cfg.identity, intervals)
    except BaseException as failure:
        code = 1
        error = dict(error_type=type(failure).__name__, failure_frames=[dict(file=f.filename,
            line=f.lineno, function=f.name) for f in traceback.extract_tb(failure.__traceback__)])
    a1.save_new(output / "terminal.json", dict(**p, exit_code=code, error=error,
        status=COMPLETE if code == 0 else "FAILED_NO_RETRY", elapsed_seconds=time.perf_counter()-started,
        registration_sha256=digest(r), retry_authorized=False, holdout_opened=False,
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False,
        partial_failure_admission="no partial cells admitted automatically; independent audit required"))
    print(json.dumps(dict(exit_code=code, roots_completed=p["roots_completed"], stage=p["stage"])), flush=True)
    return code


def supervise(output, review):
    output, review, r = admit(output, review)
    require(not (output / "supervision.json").exists(), "claimed exploration supervisor; no retry")
    began = time.perf_counter()
    a1.save_new(output / "supervision.json", dict(supervisor_pid=os.getpid(), registration_sha256=digest(r),
        review_sha256=sha256_file(review), attempt_wall_seconds=ATTEMPT_WALL_SECONDS,
        cleanup_grace_seconds=CLEANUP_GRACE_SECONDS, retry_authorized=False))
    (output / "owned-groups").mkdir()
    ownership = dict(directory=str(output / "owned-groups"), registration_sha256=digest(r),
        protocol=r["owned_receipts_protocol"])
    process = None
    worker_pid = return_code = failure = cleanup_error = None
    deadline = dict(reached=False)
    timed_out = False
    try:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "worker",
            "--output", str(output), "--review", str(review)], start_new_session=True)
        worker_pid = process.pid
        a1.save_new(output / "worker-launch.json", dict(pid=worker_pid, process_group=worker_pid,
            supervisor_pid=os.getpid(), registration_sha256=digest(r)))
        return_code, timed_out = a1.wait_bounded(process, max(.001, ATTEMPT_WALL_SECONDS -
            (time.perf_counter()-began)), CLEANUP_GRACE_SECONDS, ownership, deadline)
        require(not timed_out, "A1_EXPLORATION_WALL_CAP")
        require(return_code == 0, "exploration worker nonzero exit")
        terminal = json.loads((output / "terminal.json").read_text())
        require(terminal["exit_code"] == 0 and terminal["status"] == COMPLETE
            and terminal["registration_sha256"] == digest(r), "missing/inconsistent exploration completion")
    except BaseException as error:
        timed_out = timed_out or deadline["reached"]
        failure, cleanup_error = type(error).__name__, deadline.get("cleanup_error")
        if process is not None and not deadline.get("cleanup_completed"):
            try:
                return_code = a1.stop_owned(process, CLEANUP_GRACE_SECONDS, ownership)
            except BaseException as cleanup:
                cleanup_error = type(cleanup).__name__
    code = 0 if failure is None else 124 if timed_out else 1
    if code and not (output / "terminal.json").exists():
        a1.save_new(output / "terminal.json", dict(**new_progress(r), status="FAILED_NO_RETRY",
            exit_code=code, error_type="A1_EXPLORATION_WALL_CAP" if timed_out else failure,
            registration_sha256=digest(r), retry_authorized=False, holdout_opened=False,
            scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False,
            accounting="all cells unvalidated; audit preserved durable receipts before admitting any"))
    a1.save_new(output / "supervision-terminal.json", dict(exit_code=code, worker_return_code=return_code,
        worker_pid=worker_pid, wall_cap_reached=timed_out, error_type=failure, cleanup_error_type=cleanup_error,
        elapsed_seconds=time.perf_counter()-began, registration_sha256=digest(r),
        status=COMPLETE if code == 0 else "FAILED_NO_RETRY", retry_authorized=False,
        holdout_opened=False, scientific_strength_evidence=False))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "worker"):
        launch = sub.add_parser(command)
        launch.add_argument("--output", type=Path, required=True)
        launch.add_argument("--review", type=Path, required=True)
    prep = sub.add_parser("register")
    prep.add_argument("--exposure-registration", dest="exposure_registrations", type=Path, action="append", required=True)
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables", "plan"):
        prep.add_argument("--"+name, type=Path, required=True)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256"):
        prep.add_argument("--"+name, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command in ("run", "worker"):
        return (supervise if command == "run" else run)(**args)
    r = register(**args)
    print(json.dumps(dict(registration_sha256=digest(r), status=r["status"],
        exploration_roots=200, exploration_seeds=32, holdout_opened=False)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
