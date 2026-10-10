"""Fresh real-game engineering benchmark, never a scientific panel or retry.

Two predeclared raw-vs-raw source games supply two priority-sampled nonopening
roots each. Both public search arms retain their full allocations at 1/3/10s
with model/HP/raw-terminal leaves. Eight paired continuations measure the full
harness after action selection. First failure stops the entire attempt; the
fixed remaining roster stays unmeasured. No easier roots, redrawn seeds, fallback
actions, reduced work or replay of any prior qualification is permitted.

Version 2 is a separately registered post-metadata-optimization diagnostic.
Version 1's closed attempt, namespace and seeds remain historical evidence;
no old root is replayed and neither version supplies scientific admission.
"""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from qualify_search_over_raw_opening import prepare_binding, verify_inputs
from pokezero.mcts_eval.resolver import sha256_file
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest, paired_continuations,
    require, root_contrast, select_source_requests,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pokezero.search-over-raw.real-source-feasibility.v2"
NAMESPACE = "6ff7b2be-433a-4961-95b7-d0d47e0604ce"
SEEDS = (2026101018, 2026101019)
ROOTS_PER_SEED = 2


def configurations():
    # Fixed order before any source game: budget, leaf, then arm. No outcome-
    # dependent ordering or continuation after a poisoned pool.
    return [SearchConfiguration("raw")] + [
        SearchConfiguration(arm, "public", leaf, seconds, 20 if arm == "reference" else 1)
        for seconds in (1., 3., 10.)
        for leaf in ("model", "hp_fraction", "raw_rollout")
        for arm in ("incumbent", "reference")]


def source_contract():
    return dict(namespace=NAMESPACE, candidate_seat="p1", exclude_opening_requests=True,
        panels={"excluded": dict(seeds=list(SEEDS), root_slots=[
            dict(root_id=f"excluded:{seed}:{slot}", source_seed=seed, root_slot=slot)
            for seed in SEEDS for slot in range(ROOTS_PER_SEED)])},
        continuation_replicates=8, source_policy="raw_argmax_both_seats",
        source_root_rule="fixed seed-derived priorities over each completed full nonopening raw request catalog",
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        max_source_boundaries=250, max_continuation_boundaries=250,
        stop_policy="first source, selection, continuation, provenance or worker failure stops whole attempt",
        replacements_permitted=False, retry_authorized=False,
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False)


def save_new(path, value):
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exposure_inventory(paths):
    """Explicit supplied exposure only; no claim about an undisclosed roster."""
    require(paths, "explicit exposure registrations required")
    # These two reserved new engineering seeds are already excluded from future
    # scientific panels, but are not prior exposures until this attempt.
    seeds = set(ENGINEERING_EXCLUDED_SEEDS) - set(SEEDS)
    bindings = {}
    for supplied in paths:
        path = Path(supplied).resolve()
        value = json.loads(path.read_text())
        roster = value.get("seeds")
        require(isinstance(roster, list) and roster
            and all(type(s) is int and 0 <= s < 2**32 for s in roster),
            "exposure registration needs explicit uint32 seeds")
        seeds.update(roster)
        bindings[str(path)] = sha256_file(path)
    require(not set(SEEDS) & seeds, "fresh feasibility seeds overlap supplied exposure; no retry/redraw")
    return sorted(seeds), bindings


def register(*, output, exposure_registrations, **kwargs):
    excluded, exposure = exposure_inventory(exposure_registrations)
    binding = prepare_binding(**kwargs, seconds=10.)
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT), "attempt output must be outside executing checkout")
    binding.update(schema=SCHEMA, namespace=NAMESPACE, seeds=list(SEEDS),
        fixture_seed=SEEDS[0], configurations=[asdict(c) for c in configurations()],
        source_contract=source_contract(), attempt_directory=str(output),
        exposure_registrations=exposure, excluded_seeds=excluded,
        excluded_seed_inventory_sha256=digest(excluded),
        exposure_scope="supplied registrations plus minimum exclusions; not a universal exposure census",
        scope="four fixed real-game roots; complete public search allocations and paired continuation timing; excluded engineering only",
        representative_runtime_evidence=False, historical_attempt_reused=False)
    binding["input_sha256"].update(exposure)
    binding["input_sha256"][str(Path(__file__).resolve())] = sha256_file(__file__)
    validate_contract(binding, output)
    verify_inputs(binding)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", binding)
    save_new(output / "registration-binding.json", dict(registration_sha256=digest(binding)))
    return binding


def validate_contract(registration, output):
    require(registration["schema"] == SCHEMA
        and registration["namespace"] == NAMESPACE
        and digest(registration["source_contract"]) == digest(source_contract())
        and digest(registration["configurations"]) == digest([asdict(c) for c in configurations()])
        and registration["seeds"] == list(SEEDS)
        and all(type(s) is int for s in registration["seeds"])
        and type(registration["fixture_seed"]) is int and registration["fixture_seed"] == SEEDS[0]
        and registration["candidate_seat"] == "p1"
        and type(registration["initial_dispatch_workers"]) is int and registration["initial_dispatch_workers"] == 6
        and all(registration[key] is False for key in (
            "retry_authorized", "phase_a_admission", "phase_b_authorized",
            "scientific_strength_evidence", "representative_runtime_evidence", "historical_attempt_reused")),
        "feasibility core changed or scientific admission/retry attempted")
    excluded, bindings = exposure_inventory(list(registration["exposure_registrations"]))
    require(bindings == registration["exposure_registrations"]
        and excluded == registration["excluded_seeds"]
        and digest(excluded) == registration["excluded_seed_inventory_sha256"]
        and all(registration["input_sha256"].get(path) == sha for path, sha in bindings.items()),
        "exposure inventory drift")
    script = Path(__file__).resolve()
    require(Path(registration["source_root"]).resolve() == ROOT
        and registration["input_sha256"].get(str(script)) == sha256_file(script),
        "executing source or mandatory driver hash differs")
    require(registration["attempt_directory"] == str(Path(output).resolve())
        and not Path(output).resolve().is_relative_to(ROOT),
        "attempt directory changed or lies inside source; no copied retry")


def verify_completion(output, progress):
    """Reconcile complete execution from durable sources/selections/outcomes."""
    def load(path):
        return json.loads(path.read_text())
    selectors = continuations = 0
    expected_cells, expected_outcomes = {}, {}
    for seed in SEEDS:
        source_dir = output / f"source-{seed}"
        source = load(source_dir / "source.json")
        validate_source_receipt(source, seed)
        require(progress["source_roster"][str(seed)]["status"] == "COMPLETE",
            "completed source accounting differs")
        for root in source["roots"]:
            root_dir = source_dir / f"root-{root['root_id'].rsplit(':', 1)[1]}"
            audit = load(root_dir / "audit.json")
            require(audit["root_id"] == root["root_id"] and audit["status"] == "COMPLETE"
                and set(audit["actions"]) == {"raw", *(c.identity for c in configurations()[1:])},
                "complete audit action roster differs")
            for cfg in configurations():
                key = "raw" if cfg.arm == "raw" else cfg.identity
                selected = load(root_dir / f"{key}-selected.json")
                runtime = load(root_dir / f"{key}-runtime.json")["runtime_configuration"]
                close = load(root_dir / f"{key}-close.json")
                require(selected["status"] == "SELECTED" and selected["root_id"] == root["root_id"] + ":" + key
                    and selected["configuration_sha256"] == cfg.identity
                    and selected["runtime_sha256"] == digest(runtime)
                    and digest(runtime["configuration"]) == digest(asdict(cfg))
                    and type(close["seconds"]) in (int, float) and close["seconds"] >= 0
                    and selected["action"] == audit["actions"][key], "durable selected/runtime/close binding differs")
                interval = list(root_contrast(audit, key))
                require(interval == audit["contrasts"][key], "durable contrast differs from terminal outcomes")
                expected_cells[(root["root_id"], cfg.identity)] = (selected["action"], interval)
                for row in audit["outcomes"]:
                    if row["action"] == selected["action"]:
                        expected_outcomes[(root["root_id"], key, row["replicate"])] = row
                selectors += 1
            for row in audit["outcomes"]:
                outcome = load(root_dir / f"outcome-{row['action']}-{row['replicate']}.json")
                attempt = load(root_dir / f"continuation-{row['action']}-{row['replicate']}-attempt.json")
                require(all(outcome.get(k) == v for k, v in row.items())
                    and outcome["root_id"] == root["root_id"]
                    and attempt == dict(root_id=root["root_id"], action=row["action"], replicate=row["replicate"]),
                    "durable continuation attempt/outcome differs")
                continuations += 1
    require(progress["source_games_completed"] == len(SEEDS)
        and progress["roots_completed"] == len(SEEDS) * ROOTS_PER_SEED
        and progress["selections_completed"] == selectors == 76
        and progress["continuations_completed"] == continuations,
        "full engineering work counters differ")
    require(len(progress["fixed_roster"]) == len(expected_cells) == 76
        and len({(c["root_id"], c["configuration"]) for c in progress["fixed_roster"]}) == 76,
        "fixed selector roster differs")
    for cell in progress["fixed_roster"]:
        action, interval = expected_cells[(cell["root_id"], cell["configuration"])]
        require(cell["status"] == "MEASURED_ENGINEERING_ONLY" and cell["selection_status"] == "SELECTED"
            and cell["action"] == action and cell["contrast_interval"] == interval,
            "selector accounting differs from durable evidence")
    require(len(progress["continuation_roster"]) == len(expected_outcomes) == 608
        and len({(c["root_id"], c["configuration"], c["replicate"]) for c in progress["continuation_roster"]}) == 608,
        "continuation alias roster differs")
    for cell in progress["continuation_roster"]:
        row = expected_outcomes[(cell["root_id"], cell["configuration"], cell["replicate"])]
        require(cell["status"] == "COMPLETE" and cell["action"] == row["action"]
            and cell["signed_outcome"] == row["signed_outcome"], "continuation accounting differs from durable outcome")


def planned_cells():
    return [dict(root_id=row["root_id"], configuration=c.identity,
                 status="UNMEASURED_UNCERTAIN", contrast_interval=[-1., 1.],
                 selection_status="UNSTARTED", action=None)
        for row in source_contract()["panels"]["excluded"]["root_slots"]
        for c in configurations()]


def validate_source_receipt(source, seed):
    contract = source_contract()
    rows = source["eligible_public_records"]
    indices = [row["source_request_index"] for row in rows]
    require(source["contract_sha256"] == digest(contract)
        and source["source_seed"] == seed and source["panel"] == "excluded"
        and source["status"] == "COMPLETE" and source["source_terminal_complete"] is True
        and source["source_policy"] == "raw_argmax_both_seats"
        and source["eligible_requests"] == len(rows)
        and digest(rows) == source["eligible_catalog_sha256"]
        and all(type(i) is int and i > 0 for i in indices)
        and len(set(indices)) == len(indices), "source or frozen public catalog drift")
    selected = select_source_requests(NAMESPACE, seed, indices, ROOTS_PER_SEED)
    slots = [r for r in contract["panels"]["excluded"]["root_slots"] if r["source_seed"] == seed]
    catalog = {row["source_request_index"]: row["public_record_sha256"] for row in rows}
    require(source["missing_root_ids"] == [] and source["requested_root_slots"] == ROOTS_PER_SEED
        and len(source["roots"]) == len(selected) == ROOTS_PER_SEED,
        "incomplete source; no redraw/partial feasibility")
    for root, slot, index in zip(source["roots"], slots, selected):
        require(root["root_id"] == slot["root_id"] and root["source_request_index"] == index
            and digest(root["public_record"]) == root["public_record_sha256"] == catalog[index],
            "selected root does not match full-catalog priority sampling")


def planned_continuations():
    return [dict(root_id=row["root_id"], configuration="raw" if c.arm == "raw" else c.identity,
                 replicate=replicate, status="UNSTARTED_UNCERTAIN", action=None, signed_outcome=None)
        for row in source_contract()["panels"]["excluded"]["root_slots"]
        for c in configurations() for replicate in range(8)]


def record_outcome(output, progress, actions, row, elapsed):
    # Persist the actual unique-action row before stopping. Aliases inherit the
    # SAME outcome; never invent independent observations for equal actions.
    save_new(output / f"outcome-{row['action']}-{row['replicate']}.json",
        dict(row, cumulative_seconds=elapsed))
    for cell in progress["continuation_roster"]:
        if (cell["root_id"] == row["root_id"] and cell["replicate"] == row["replicate"]
                and actions.get(cell["configuration"]) == row["action"]):
            cell.update(status=row["status"], action=row["action"], signed_outcome=row["signed_outcome"])
    progress["continuations_completed"] += int(row["status"] == "COMPLETE")
    print(json.dumps(dict(stage="continuation", **row)), flush=True)
    require(row["status"] == "COMPLETE", "capped continuation; stop whole attempt")


def record_continuation_attempt(output, progress, actions, row):
    save_new(output / f"continuation-{row['action']}-{row['replicate']}-attempt.json", row)
    progress["current_continuation"] = dict(row)
    for cell in progress["continuation_roster"]:
        if (cell["root_id"] == row["root_id"] and cell["replicate"] == row["replicate"]
                and actions.get(cell["configuration"]) == row["action"]):
            cell.update(status="ATTEMPTED_UNCERTAIN", action=row["action"])


def selection_work(configuration, selected, legal_choices):
    evidence = selected["evidence"]
    if configuration.arm == "raw":
        completed, unit = 0, "no_search_raw_forward"
    elif configuration.arm == "incumbent":
        completed, unit = evidence["total_iterations"], "native_atomic_iterations"
    else:
        completed, unit = evidence["result"]["trajectories"], "reference_completed_trajectories"
    require(type(completed) is int and completed >= 0, "invalid completed search work")
    return dict(legal_choices=legal_choices, forced_action=legal_choices == 1,
        completed_search_units=completed, search_unit=unit,
        substantive_search_exercised=configuration.arm != "raw" and legal_choices > 1 and completed > 0,
        leaf_exercise_qualified=False,
        leaf_note="Full adapter work witnesses retained; selection/throughput alone is not leaf-effect qualification")


def close_resources(resources, progress):
    """Attempt all owned cleanup, retaining the FIRST operational failure."""
    primary_active = sys.exc_info()[0] is not None
    first_cleanup = None
    for name, resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            progress.setdefault("cleanup_failures", []).append(dict(resource=name,
                error_type=type(error).__name__, failure_frames=[dict(
                    file=f.filename, line=f.lineno, function=f.name)
                    for f in traceback.extract_tb(error.__traceback__)]))
            if first_cleanup is None:
                first_cleanup = error
    if first_cleanup is not None and not primary_active:
        progress["current_operation"] = dict(kind="cleanup")
        progress["stage"] += ":cleanup"
        raise first_cleanup


def _runtime(registration):
    import sys
    import torch
    import poke_engine
    import pokezero_search
    require(sys.executable == registration["python_executable"]
        and torch.__version__ == registration["torch_version"]
        and torch.__file__ == registration["torch_path"], "Python/PyTorch identity drift")
    require(getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False) is True,
        "native model feature absent")
    for module, name in ((poke_engine, "poke_engine"), (pokezero_search, "pokezero_search")):
        require(Path(module.__file__).resolve() == Path(registration["packages_root"]) / name / "__init__.py",
            "native package resolves outside isolated build")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def measure(registration, output, progress):
    _runtime(registration)
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
    from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
    from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
    from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy, collect_raw_source

    checkpoint, showdown = registration["checkpoint"], registration["showdown_root"]
    contract = resolve_checkpoint_contract(checkpoint, expected_sha256=registration["checkpoint_sha256"],
        showdown_root=showdown, showdown_source_sha256=registration["set_source_hash"],
        model_path=registration["model_path"], tables_path=registration["encoder_tables"])
    require(contract.to_manifest() == registration["checkpoint_contract"], "checkpoint contract drift")
    factory = ShowdownWorkerFactory(checkpoint, registration["checkpoint_sha256"], showdown,
        registration["set_source_hash"], **registration["factory_options"])
    for seed in SEEDS:
        env = archive = adapter = None
        source_dir = output / f"source-{seed}"
        source_dir.mkdir()
        progress["stage"] = f"source:{seed}:construct"
        progress["current_operation"] = dict(kind="source", seed=seed, phase="construct")
        try:
            config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(showdown),
                set_belief_source=True), [f"neural:{checkpoint}"], context="fresh source feasibility")
            env = LocalShowdownEnv(config)
            archive = SealedSourceArchive()
            policies = {}
            for seat in ("p1", "p2"):
                policy = load_transformer_policy(Path(checkpoint), device="cpu", deterministic=True,
                    exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
                require(policy.result.belief_set_source_hash == registration["set_source_hash"], "champion catalog drift")
                policies[seat] = AuditedRawPolicy(policy, checkpoint_sha256=registration["checkpoint_sha256"],
                    public_context_sink=archive.capture_public if seat == "p1" else None)

            def source_progress(event):
                if event.decision_round_count == 1 or event.decision_round_count % 10 == 0 or event.terminal:
                    row = dict(stage="source", seed=seed, boundary_count=event.decision_round_count,
                        source_terminal=event.terminal)
                    save_new(source_dir / f"progress-{event.decision_round_count:03d}.json", row)
                    print(json.dumps(row), flush=True)

            verify_inputs(registration)
            save_new(source_dir / "attempt.json", dict(seed=seed, retry_authorized=False))
            progress["stage"] = f"source:{seed}:collect"
            progress["current_operation"]["phase"] = "collect"
            progress["source_roster"][str(seed)]["status"] = "ATTEMPTED_UNCERTAIN"
            source = collect_raw_source(registration["source_contract"], panel="excluded", source_seed=seed,
                env=env, policies=policies, max_decision_rounds=250, decision_sink=source_progress,
                sealed_pre_step_sink=archive.capture_private)
            save_new(source_dir / "source.json", source)
            verify_inputs(registration)
            progress["source_roster"][str(seed)].update(status=source["status"],
                eligible_requests=source["eligible_requests"], missing_root_ids=source["missing_root_ids"],
                elapsed_seconds=source["elapsed_seconds"], decision_boundaries=source["decision_boundaries"])
            validate_source_receipt(source, seed)
            progress["source_games_completed"] += 1
            evaluator = ChampionEvaluator(policies["p1"].policy)
            for root in source["roots"]:
                context, pending, snapshot = archive.selected(root)
                root_id = root["root_id"]
                root_dir = source_dir / f"root-{root_id.rsplit(':', 1)[1]}"
                root_dir.mkdir()
                actions = {}
                for cfg in configurations():
                    key = "raw" if cfg.arm == "raw" else cfg.identity
                    progress["stage"] = root_id + ":" + key + ":construct"
                    progress["current_operation"] = dict(kind="selector", root_id=root_id,
                        configuration=cfg.identity, phase="construct")
                    verify_inputs(registration)
                    save_new(root_dir / f"{key}-attempt.json", dict(configuration=asdict(cfg),
                        source_public_record_sha256=root["public_record_sha256"], retry_authorized=False))
                    cell = next(c for c in progress["fixed_roster"] if c["root_id"] == root_id
                        and c["configuration"] == cfg.identity)
                    cell["selection_status"] = "CONSTRUCT_ATTEMPTED_UNCERTAIN"
                    constructed = time.perf_counter()
                    adapter = PublicModelSearchAdapter(cfg, checkpoint_contract=contract,
                        showdown_root=showdown, evaluator=evaluator if cfg.arm == "raw" else None,
                        reference_factory=factory if cfg.arm == "reference" else None,
                        initial_dispatch_workers=registration["initial_dispatch_workers"])
                    construction_seconds = time.perf_counter() - constructed
                    save_new(root_dir / f"{key}-runtime.json", dict(construction_seconds=construction_seconds,
                        runtime_configuration=adapter.runtime_configuration))
                    progress["stage"] = root_id + ":" + key + ":select"
                    progress["current_operation"]["phase"] = "select"
                    cell["selection_status"] = "SELECTION_ATTEMPTED_UNCERTAIN"
                    try:
                        selected = adapter.select(context, root_id=root_id + ":" + key,
                            selection_seed=seed, pending_transition=pending)
                    except BaseException:
                        progress["adapter_failure"] = adapter.last_failure
                        raise
                    verify_inputs(registration)
                    save_new(root_dir / f"{key}-selected.json", selected)
                    work = selection_work(cfg, selected, sum(context.observation.legal_action_mask))
                    save_new(root_dir / f"{key}-work.json", work)
                    actions[key] = selected["action"]
                    for cell in progress["fixed_roster"]:
                        if cell["root_id"] == root_id and cell["configuration"] == cfg.identity:
                            cell.update(selection_status="SELECTED", action=selected["action"],
                                selection_seconds=selected["elapsed_seconds"],
                                exceeded_nominal_seconds=selected["exceeded_nominal_seconds"],
                                construction_seconds=construction_seconds, work=work)
                    progress["selections_completed"] += 1
                    print(json.dumps(dict(stage=progress["stage"], seconds=selected["elapsed_seconds"],
                        exceeded_nominal_seconds=selected["exceeded_nominal_seconds"])), flush=True)
                    # One genuine pool per configuration; no overlapping20-worker pools.
                    progress["stage"] = root_id + ":" + key + ":close"
                    progress["current_operation"]["phase"] = "close"
                    closing = time.perf_counter()
                    adapter.close()
                    close_seconds = time.perf_counter() - closing
                    adapter = None
                    save_new(root_dir / f"{key}-close.json", dict(seconds=close_seconds,
                        end_to_end_seconds=construction_seconds + selected["elapsed_seconds"] + close_seconds,
                        nominal_ceiling_scope="selection only; observed overshoot is reported, not silently reclassified"))
                require(actions["raw"] == root["public_record"]["recorded_action_index"],
                    "raw source/evaluator action drift")
                progress["stage"] = root_id + ":continuations"
                progress["current_operation"] = dict(kind="continuation", root_id=root_id)
                save_new(root_dir / "continuation-attempt.json", dict(actions=actions, replicates=8,
                    maximum_boundaries=250, retry_authorized=False))
                began = time.perf_counter()

                def outcome_sink(row):
                    record_outcome(root_dir, progress, actions, row, time.perf_counter() - began)

                def raw_evaluator(observation):
                    legal, evaluation = evaluator(observation)
                    return legal, evaluation.priors

                def attempt_sink(row):
                    record_continuation_attempt(root_dir, progress, actions, row)

                audit = paired_continuations(env=env, snapshot=snapshot, subject="p1", actions=actions,
                    evaluator=raw_evaluator, namespace=NAMESPACE, root_id=root_id, max_boundaries=250,
                    outcome_sink=outcome_sink, attempt_sink=attempt_sink)
                audit.update(elapsed_seconds=time.perf_counter() - began,
                    scientific_strength_evidence=False,
                    contrasts={key: root_contrast(audit, key) for key in actions})
                verify_inputs(registration)
                save_new(root_dir / "audit.json", audit)
                for cell in progress["fixed_roster"]:
                    if cell["root_id"] == root_id:
                        cell.update(status="MEASURED_ENGINEERING_ONLY",
                            contrast_interval=list(audit["contrasts"]["raw" if cell["configuration"] ==
                                configurations()[0].identity else cell["configuration"]]))
                progress["roots_completed"] += 1
        finally:
            close_resources((("adapter", adapter), ("archive", archive), ("source_env", env)), progress)


def run(output):
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    binding = json.loads((output / "registration-binding.json").read_text())
    require(binding == dict(registration_sha256=digest(registration)), "registered input-map or contract drift")
    validate_contract(registration, output)
    verify_inputs(registration)
    for name in ("attempt.json", "terminal.json"):
        if (output / name).exists():
            raise FileExistsError("closed or claimed feasibility attempt; no retry")
    save_new(output / "attempt.json", dict(pid=os.getpid(), status="CLAIMED_BEFORE_RUNTIME",
        registration_sha256=digest(registration), retry_authorized=False))
    started = time.perf_counter()
    progress = dict(stage="runtime_import", source_games_completed=0, selections_completed=0,
        continuations_completed=0, roots_completed=0, fixed_roster=planned_cells(),
        continuation_roster=planned_continuations(), source_roster={str(seed): dict(
            status="UNSTARTED_UNCERTAIN", missing_root_ids=[r["root_id"] for r in
                source_contract()["panels"]["excluded"]["root_slots"] if r["source_seed"] == seed])
            for seed in SEEDS})
    result = dict(registration_sha256=digest(registration), retry_authorized=False,
        phase_a_admission=False, phase_b_authorized=False, scientific_strength_evidence=False,
        representative_runtime_evidence=False, historical_attempt_reused=False)
    try:
        measure(registration, output, progress)
        verify_inputs(registration)
        verify_completion(output, progress)
        result.update(status="COMPLETE_REAL_SOURCE_ENGINEERING_ONLY")
        code = 0
    except BaseException as error:
        from pokezero.mcts_eval.policy_opponent_profile import fallback_refusal_diagnostic, refusal_diagnostic
        result.update(status="FAILED_NO_RETRY", error_type=type(error).__name__,
            native_diagnostic=refusal_diagnostic(error), engine_fallback_diagnostic=fallback_refusal_diagnostic(error),
            failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)])
        code = 1
        if progress.get("current_continuation"):
            current = progress["current_continuation"]
            for cell in progress["continuation_roster"]:
                if (cell["status"] == "ATTEMPTED_UNCERTAIN" and cell["root_id"] == current["root_id"]
                        and cell["action"] == current["action"] and cell["replicate"] == current["replicate"]):
                    cell["status"] = "REFUSED_UNCERTAIN"
        operation = progress.get("current_operation", {})
        if operation.get("kind") == "source":
            source = progress["source_roster"][str(operation["seed"])]
            if source["status"] != "COMPLETE":
                source["status"] = "REFUSED_UNCERTAIN" if operation["phase"] == "collect" else "CONSTRUCTION_FAILED_UNCERTAIN"
        elif operation.get("kind") == "selector":
            cell = next(c for c in progress["fixed_roster"] if c["root_id"] == operation["root_id"]
                and c["configuration"] == operation["configuration"])
            if operation["phase"] == "close":
                cell["cleanup_status"] = "FAILED"
            elif cell["selection_status"] != "SELECTED":
                cell["selection_status"] = "REFUSED_UNCERTAIN" if operation["phase"] == "select" else "CONSTRUCTION_FAILED_UNCERTAIN"
    result.update(**progress, elapsed_seconds=time.perf_counter() - started, exit_code=code)
    save_new(output / "terminal.json", result)
    print(json.dumps(dict(status=result["status"], exit_code=code,
        stage=progress["stage"], roots_completed=progress["roots_completed"])), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("register")
    prep.add_argument("--exposure-registration", dest="exposure_registrations", type=Path,
        action="append", required=True)
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256"):
        prep.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "run":
        return run(**args)
    registration = register(**args)
    print(json.dumps(dict(status=registration["status"], registration_sha256=digest(registration))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
