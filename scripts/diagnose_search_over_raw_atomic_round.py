"""Fresh excluded native raw-terminal cost probe, not a scientific panel.

One completed raw-vs-raw source game supplies one seed-priority sampled
nonopening root. Two independent full-allocation 10s adapters use that SAME
prospectively registered root and selection seed, instrumented then disabled.
This ordered pair is descriptive, not a causal overhead or runtime estimate.
No old root, redraw, reduced batch/world count, or post-refusal control runs.
"""
from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from uuid import UUID

from benchmark_search_over_raw_feasibility import _runtime, close_resources, save_new
from qualify_search_over_raw_opening import prepare_binding, verify_inputs
from pokezero.mcts_eval.resolver import sha256_file
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest, require, select_source_requests,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pokezero.search-over-raw.atomic-round-diagnostic.v1"
NAMESPACE = "a29f9e42-63e5-4d0b-9124-7c5a8d306bf1"
SEED = 2026101020
MODES = ("instrumented", "disabled")
CONFIGURATION = SearchConfiguration("incumbent", "public", "raw_rollout", 10., 1)


def validate_identity(seed, namespace):
    require(type(seed) is int and 0 <= seed < 2**32, "probe seed must be exact uint32")
    require(type(namespace) is str, "probe namespace must be canonical UUID")
    try:
        canonical = str(UUID(namespace))
    except (ValueError, AttributeError):
        raise ValueError("probe namespace must be canonical UUID") from None
    require(namespace == canonical, "probe namespace must be canonical UUID")
    require((seed == SEED) == (namespace == NAMESPACE), "historical probe identity cannot be rebound")


def source_contract(seed=SEED, namespace=NAMESPACE):
    validate_identity(seed, namespace)
    return dict(namespace=namespace, candidate_seat="p1", exclude_opening_requests=True,
        panels={"excluded": dict(seeds=[seed], root_slots=[dict(
            root_id=f"excluded:{seed}:0", source_seed=seed, root_slot=0)])},
        source_policy="raw_argmax_both_seats", max_source_boundaries=250,
        replacements_permitted=False, retry_authorized=False,
        scientific_strength_evidence=False, phase_a_admission=False)


def core(seed=SEED, namespace=NAMESPACE):
    validate_identity(seed, namespace)
    return dict(schema=SCHEMA, namespace=namespace, seeds=[seed], fixture_seed=seed,
        candidate_seat="p1", configurations=[asdict(CONFIGURATION)],
        source_contract=source_contract(seed, namespace), mode_order=list(MODES),
        selection_seed=int(digest([namespace, seed, "selection"])[:16], 16),
        paired_root_reuse="Same fresh root in independently constructed adapters, declared before source collection",
        stop_policy="First source, construction, selection, diagnostic, provenance or cleanup failure stops entire attempt",
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False,
        retry_authorized=False, representative_runtime_evidence=False,
        historical_attempt_reused=False, causal_overhead_evidence=False,
        native_allocation=dict(depth=6, sims=4096, batch=16, worlds=4, threads=1,
            guard_ms=64, rollout_count=1, rollout_threads=1, rollout_max_plies=250,
            branch_on_damage=True, model_forwards="retained", priors="unchanged_champion",
            deadline="whole_round_cancel_without_backup"))


def exposure_inventory(paths, *, seed=SEED):
    require(paths, "explicit exposure registrations required")
    require(type(seed) is int and 0 <= seed < 2**32, "probe seed must be exact uint32")
    # Preserve the historical default contract for audit/tests. Fresh attempts
    # retain ALL minimum exclusions, including the permanently terminal seed20.
    seeds = set(ENGINEERING_EXCLUDED_SEEDS) - ({SEED} if seed == SEED else set())
    bindings = {}
    for supplied in paths:
        path = Path(supplied).resolve()
        value = json.loads(path.read_text())
        roster = value.get("seeds")
        require(isinstance(roster, list) and roster
            and all(type(s) is int and 0 <= s < 2**32 for s in roster),
            "exposure registration requires explicit uint32 seeds")
        seeds.update(roster)
        bindings[str(path)] = sha256_file(path)
    require(seed not in seeds, "fresh diagnostic seed already exposed; no retry/redraw")
    return sorted(seeds), bindings


def bound_driver_files():
    return [Path(__file__).resolve(), ROOT / "scripts/benchmark_search_over_raw_feasibility.py",
        ROOT / "scripts/qualify_search_over_raw_opening.py", *[
            ROOT / "src/pokezero" / path for path in (
                "engine_search.py", "policy_opponent.py", "policy_opponent_diagnostics.py",
                "policy_opponent_view.py",
                "mcts_eval/lattice.py", "mcts_eval/policy_opponent_profile.py",
                "mcts_eval/search_over_raw.py", "mcts_eval/search_over_raw_adapters.py",
                "mcts_eval/search_over_raw_source.py", "mcts_eval/search_over_raw_archive.py")]]


def validate_contract(registration, output):
    expected = core(registration["fixture_seed"], registration["namespace"])
    require(digest({k: registration[k] for k in expected}) == digest(expected),
        "diagnostic core changed or scientific/retry claim attempted")
    excluded, exposure = exposure_inventory(list(registration["exposure_registrations"]),
        seed=registration["fixture_seed"])
    require(excluded == registration["excluded_seeds"]
        and digest(excluded) == registration["excluded_seed_inventory_sha256"]
        and exposure == registration["exposure_registrations"], "exposure inventory drift")
    require(all(registration["input_sha256"].get(path) == sha for path, sha in exposure.items())
        and all(registration["input_sha256"].get(str(p)) == sha256_file(p)
                for p in bound_driver_files()), "mandatory driver/input binding drift")
    require(registration["source_root"] == str(ROOT)
        and registration["attempt_directory"] == str(Path(output).resolve())
        and not Path(output).resolve().is_relative_to(ROOT), "source/output identity drift")


def register(*, output, exposure_registrations, seed=SEED, namespace=NAMESPACE, **kwargs):
    prospective = core(seed, namespace)
    require(seed not in ENGINEERING_EXCLUDED_SEEDS, "terminal or excluded probe seed cannot be registered")
    excluded, exposure = exposure_inventory(exposure_registrations, seed=seed)
    binding = prepare_binding(**kwargs, seconds=10.)
    output = Path(output).resolve()
    binding.update(**prospective, attempt_directory=str(output), exposure_registrations=exposure,
        excluded_seeds=excluded, excluded_seed_inventory_sha256=digest(excluded),
        exposure_scope="supplied registrations plus minimum exclusions, not a universal census",
        scope="one fresh real-source root, instrumented then disabled native atomic-round diagnostic only")
    binding["input_sha256"].update(exposure)
    binding["input_sha256"].update({str(p): sha256_file(p) for p in bound_driver_files()})
    validate_contract(binding, output)
    verify_inputs(binding)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", binding)
    save_new(output / "registration-binding.json", dict(registration_sha256=digest(binding)))
    return binding


def validate_source(source, *, checkpoint_sha256=None, seed=SEED, namespace=NAMESPACE):
    validate_identity(seed, namespace)
    from pokezero.public_decision_corpus import PublicDecisionRecord
    rows = source["eligible_public_records"]
    indices = [r["source_request_index"] for r in rows]
    require(source["schema"] == "pokezero.search-over-raw.source.v1"
        and source["panel"] == "excluded" and type(source["source_seed"]) is int
        and source["source_seed"] == seed and source["contract_sha256"] == digest(source_contract(seed, namespace))
        and source["status"] == "COMPLETE" and source["source_terminal_complete"] is True
        and source["source_policy"] == "raw_argmax_both_seats"
        and type(source["eligible_requests"]) is int and source["eligible_requests"] == len(rows)
        and digest(rows) == source["eligible_catalog_sha256"]
        and all(type(i) is int and i > 0 for i in indices)
        and len(set(indices)) == len(indices), "source/catalog drift")
    require(type(source["decision_boundaries"]) is int and 0 < source["decision_boundaries"] <= 250
        and all(i < source["decision_boundaries"] for i in indices)
        and finite_seconds(source["elapsed_seconds"])
        and type(source["verified_raw_decisions"]) is int and source["verified_raw_decisions"] > 0
        and (checkpoint_sha256 is None or source["checkpoint_sha256"] == checkpoint_sha256),
        "source checkpoint/cap/elapsed evidence drift")
    chosen = select_source_requests(namespace, seed, indices, 1)
    require(source["missing_root_ids"] == [] and source["requested_root_slots"] == 1
        and len(source["roots"]) == len(chosen) == 1, "source incomplete; no redraw")
    root = source["roots"][0]
    public = PublicDecisionRecord.from_dict(root["public_record"])
    require(root["root_id"] == f"excluded:{seed}:0" and root["source_request_index"] == chosen[0]
        and type(root["public_record"]["seed"]) is int and public.seed == seed
        and public.acting_player == "p1" and public.format_id == "gen3randombattle"
        and public.turn_index == chosen[0]
        and digest(public.to_dict())
            == root["public_record_sha256"]
        and root["public_record_sha256"] == next(
            r["public_record_sha256"] for r in rows if r["source_request_index"] == chosen[0]),
        "root differs from frozen priority sampling")
    return root


def measure(registration, output, progress):
    seed, namespace = registration["fixture_seed"], registration["namespace"]
    _runtime(registration)
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
    from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
    from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy, collect_raw_source
    from pokezero.policy_opponent_diagnostics import PolicyOpponentDiagnostics

    checkpoint, showdown = registration["checkpoint"], registration["showdown_root"]
    contract = resolve_checkpoint_contract(checkpoint, expected_sha256=registration["checkpoint_sha256"],
        showdown_root=showdown, showdown_source_sha256=registration["set_source_hash"],
        model_path=registration["model_path"], tables_path=registration["encoder_tables"])
    require(contract.to_manifest() == registration["checkpoint_contract"], "checkpoint contract drift")
    env = archive = adapter = None
    try:
        progress["stage"] = "source:construct"
        env = LocalShowdownEnv(env_config_with_policy_spec_masks(LocalShowdownConfig(
            showdown_root=Path(showdown), set_belief_source=True), [f"neural:{checkpoint}"],
            context="fresh excluded atomic-round cost diagnostic"))
        archive = SealedSourceArchive()
        policies = {}
        for seat in ("p1", "p2"):
            policy = load_transformer_policy(Path(checkpoint), device="cpu", deterministic=True,
                exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
            require(policy.result.belief_set_source_hash == registration["set_source_hash"], "champion catalog drift")
            policies[seat] = AuditedRawPolicy(policy, checkpoint_sha256=registration["checkpoint_sha256"],
                public_context_sink=archive.capture_public if seat == "p1" else None)
        verify_inputs(registration)
        save_new(output / "source-attempt.json", dict(seed=seed, retry_authorized=False))
        progress.update(stage="source:collect", source_status="ATTEMPTED_UNCERTAIN")
        source = collect_raw_source(source_contract(seed, namespace), panel="excluded", source_seed=seed,
            env=env, policies=policies, max_decision_rounds=250,
            sealed_pre_step_sink=archive.capture_private)
        save_new(output / "source.json", source)
        root = validate_source(source, checkpoint_sha256=registration["checkpoint_sha256"],
            seed=seed, namespace=namespace)
        progress.update(source_status="COMPLETE", source_boundaries=source["decision_boundaries"],
            source_elapsed_seconds=source["elapsed_seconds"], source_request_index=root["source_request_index"])
        context, pending, _snapshot = archive.selected(root)
        # A forced root is not replaced by a more interesting one.
        require(sum(context.observation.legal_action_mask) > 1, "priority-sampled root forced; no redraw")
        for mode in MODES:
            sink = PolicyOpponentDiagnostics() if mode == "instrumented" else None
            progress.update(stage=mode + ":construct", current_mode=mode)
            verify_inputs(registration)
            save_new(output / f"{mode}-attempt.json", dict(mode=mode, configuration=asdict(CONFIGURATION),
                source_public_record_sha256=root["public_record_sha256"],
                root_id=root["root_id"], selection_seed=registration["selection_seed"], retry_authorized=False))
            progress["modes"][mode]["status"] = "ATTEMPTED_UNCERTAIN"
            began = time.perf_counter()
            try:
                try:
                    adapter = PublicModelSearchAdapter(CONFIGURATION, checkpoint_contract=contract,
                        showdown_root=showdown, **({"policy_opponent_diagnostics": sink} if sink is not None else {}))
                finally:
                    construction_seconds = time.perf_counter() - began
                    progress["modes"][mode]["construction_seconds"] = construction_seconds
                save_new(output / f"{mode}-runtime.json", dict(
                    runtime_configuration=adapter.runtime_configuration, runtime_sha256=adapter.runtime_sha256,
                    construction_seconds=construction_seconds))
                progress["stage"] = mode + ":select"
                selection_began = time.perf_counter()
                try:
                    selected = adapter.select(context, root_id=root["root_id"],
                        selection_seed=registration["selection_seed"], pending_transition=pending)
                finally:
                    progress["modes"][mode]["selection_wall_seconds"] = time.perf_counter() - selection_began
                verify_inputs(registration)
                save_new(output / f"{mode}-selected.json", selected)
                progress["modes"][mode].update(status="SELECTED_ENGINEERING_ONLY", action=selected["action"],
                    completed_atomic_iterations=selected["evidence"]["total_iterations"],
                    elapsed_seconds=selected["elapsed_seconds"])
            finally:
                # Secondary diagnostic failures must not mask the native refusal.
                primary_active = sys.exc_info()[0] is not None
                try:
                    snapshot = sink.snapshot() if sink is not None else dict(enabled=False,
                        schema="pokezero.policy-opponent.callback-diagnostics.disabled.v1")
                    progress["modes"][mode]["diagnostics"] = snapshot
                    save_new(output / f"{mode}-diagnostics.json", snapshot)
                except BaseException as error:
                    progress.setdefault("diagnostic_failures", []).append(dict(mode=mode,
                        error_type=type(error).__name__, failure_frames=[dict(
                            file=f.filename, line=f.lineno, function=f.name)
                            for f in traceback.extract_tb(error.__traceback__)]))
                    if not primary_active:
                        raise
            require(sink is None or (snapshot["valid"] and snapshot["phases"]["callback_total"]["calls"] > 0),
                "missing or invalid callback diagnostic")
            progress["stage"] = mode + ":close"
            began = time.perf_counter()
            owned, adapter = adapter, None
            close_resources([(mode, owned)], progress)
            save_new(output / f"{mode}-close.json", dict(seconds=time.perf_counter() - began))
            print(json.dumps(dict(stage=mode, **progress["modes"][mode])), flush=True)
    finally:
        close_resources([("adapter", adapter), ("archive", archive), ("env", env)], progress)


def expected_runtime(registration, *, enabled):
    runtime = dict(configuration=asdict(CONFIGURATION),
        checkpoint_sha256=registration["checkpoint_sha256"], set_source_hash=registration["set_source_hash"],
        root_statistics="independent_per_root",
        incumbent=dict(depth=6, sims=4096, batch=16, worlds=4, native_batch_guard_ms=64),
        reference_factory=None, initial_dispatch_workers=None,
        incumbent_leaf=dict(leaf="raw_rollout", tree="unchanged_encoded_model_tree",
            priors="unchanged_champion", model_forwards="retained", value_frame="side_one_absolute",
            policy="both_seats_own_raw_masked_argmax", rollout_cap=250,
            cap="refusal_without_value_fallback", deadline="whole_round_cancel_without_backup",
            rollout_count=1, rollout_threads=1, branch_on_damage=True,
            seed="selection_seed_then_sha256_world_domain_v1"))
    if enabled:
        from pokezero.policy_opponent_diagnostics import SCHEMA as DIAGNOSTIC_SCHEMA
        runtime["callback_diagnostics"] = dict(schema=DIAGNOSTIC_SCHEMA,
            enabled=True, aggregate_only=True, instrumentation_can_change_deadlines=True,
            qualifies_uninstrumented_runtime=False)
    return runtime


def finite_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_diagnostic(diagnostic):
    from pokezero.policy_opponent_diagnostics import PHASES, PolicyOpponentDiagnostics
    template = PolicyOpponentDiagnostics().snapshot()
    require(set(diagnostic) == set(template)
        and digest({k: v for k, v in diagnostic.items() if k != "phases"})
            == digest({k: v for k, v in template.items() if k != "phases"})
        and set(diagnostic["phases"]) == set(PHASES), "diagnostic schema or validity drift")
    for row in diagnostic["phases"].values():
        require(set(row) == {"calls", "timed_calls", "failed_calls", "elapsed_seconds"}
            and all(type(row[k]) is int and row[k] >= 0 for k in ("calls", "timed_calls", "failed_calls"))
            and row["calls"] == row["timed_calls"] and row["failed_calls"] == 0
            and finite_seconds(row["elapsed_seconds"]), "invalid diagnostic phase counters/time")
    require(diagnostic["phases"]["callback_total"]["calls"] > 0, "callback not exercised")
    calls = {phase: row["calls"] for phase, row in diagnostic["phases"].items()}
    require(len({calls[p] for p in ("callback_total", "payload_binding", "view_reconstruction",
        "distribution_binding", "observation_and_surface")}) == 1
        and calls["model_evaluation"] == calls["output_certification"] <= calls["callback_total"],
        "successful callback phase counts do not conserve")
    require(all(calls[p] == calls["view_reconstruction"] for p in (
        "view_public_projection", "view_own_request_certification", "view_replay_parse",
        "view_belief_rebuild", "view_normalization", "view_materialization"))
        and calls["view_branch_clone"] == calls["view_suffix_parse"] == calls["view_replay_snapshot"]
            <= calls["view_replay_parse"]
        and calls["view_prefix_prepare"] <= calls["view_branch_clone"],
        "successful view component counts do not conserve")


def verify_completion(registration, output, progress):
    from pokezero.mcts_eval.manifest import SearchConfig
    from pokezero.mcts_eval.policy_opponent_profile import validate_selection
    def load(name):
        return json.loads((output / name).read_text())
    root = validate_source(load("source.json"), checkpoint_sha256=registration["checkpoint_sha256"],
        seed=registration["fixture_seed"], namespace=registration["namespace"])
    identities = []
    for mode in MODES:
        attempt, runtime = load(f"{mode}-attempt.json"), load(f"{mode}-runtime.json")
        selected, diagnostic = load(f"{mode}-selected.json"), load(f"{mode}-diagnostics.json")
        rc = runtime["runtime_configuration"]
        require(attempt == dict(mode=mode, configuration=asdict(CONFIGURATION),
            source_public_record_sha256=root["public_record_sha256"], root_id=root["root_id"],
            selection_seed=registration["selection_seed"], retry_authorized=False), "durable attempt drift")
        require(digest(rc) == digest(expected_runtime(registration, enabled=mode == "instrumented")),
            "durable runtime allocation/checkpoint/catalog differs from frozen core")
        require(selected["status"] == "SELECTED" and selected["root_id"] == root["root_id"]
            and selected["configuration_sha256"] == CONFIGURATION.identity
            and selected["runtime_sha256"] == runtime["runtime_sha256"] == digest(rc)
            and digest(rc["configuration"]) == digest(asdict(CONFIGURATION))
            and type(selected["evidence"]["total_iterations"]) is int
            and selected["evidence"]["total_iterations"] > 0
            and finite_seconds(runtime["construction_seconds"])
            and finite_seconds(selected["elapsed_seconds"]),
            "durable search/runtime identity or completed work drift")
        mask = root["public_record"]["current_legal_action_mask"]
        require(sum(mask) > 1 and type(selected["action"]) is int
            and 0 <= selected["action"] < len(mask) and mask[selected["action"]]
            and selected["evidence"]["root_action"] == f"action:{selected['action']}", "selected action/mask drift")
        validate_selection(selected["evidence"], arm="incumbent_mcts", mode="matched_deadline",
            config=SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode="local"),
            mask=mask, opponent_seed=registration["selection_seed"], deadline_ms=10000,
            native_batch_guard_ms=64, model_leaf_override="raw_policy_terminal")
        require(progress["modes"][mode]["status"] == "SELECTED_ENGINEERING_ONLY"
            and progress["modes"][mode]["action"] == selected["action"]
            and progress["modes"][mode]["completed_atomic_iterations"] == selected["evidence"]["total_iterations"]
            and progress["modes"][mode]["construction_seconds"] == runtime["construction_seconds"]
            and finite_seconds(progress["modes"][mode]["selection_wall_seconds"])
            and progress["modes"][mode]["diagnostics"] == diagnostic, "progress/evidence drift")
        closed = load(f"{mode}-close.json")["seconds"]
        require(finite_seconds(closed), "cleanup evidence missing")
        if mode == "instrumented":
            validate_diagnostic(diagnostic)
        else:
            require(diagnostic == dict(enabled=False,
                schema="pokezero.policy-opponent.callback-diagnostics.disabled.v1")
                and "callback_diagnostics" not in rc, "disabled control contains timing")
        identities.append(runtime["runtime_sha256"])
    require(len(set(identities)) == 2, "enabled/disabled runtime identities alias")


def run(output):
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    require(load_binding(output) == digest(registration), "registration binding drift")
    validate_contract(registration, output)
    require(registration["fixture_seed"] not in ENGINEERING_EXCLUDED_SEEDS,
        "terminal or excluded probe seed cannot run")
    verify_inputs(registration)
    require(not any((output / name).exists() for name in (
        "attempt.json", "terminal.json", "source-attempt.json", "source.json",
        *(f"{mode}-{suffix}.json" for mode in MODES for suffix in
            ("attempt", "runtime", "selected", "diagnostics", "close")))),
        "attempt or evidence already exists; no retry")
    save_new(output / "attempt.json", dict(status="CLAIMED_BEFORE_RUNTIME", registration_sha256=digest(registration)))
    started = time.perf_counter()
    progress = dict(stage="runtime_import", source_status="UNSTARTED_UNCERTAIN",
        modes={mode: dict(status="UNSTARTED_UNCERTAIN") for mode in MODES})
    result = dict(registration_sha256=digest(registration), **{
        k: False for k in ("retry_authorized", "scientific_strength_evidence", "phase_a_admission",
            "phase_b_authorized", "representative_runtime_evidence", "causal_overhead_evidence")})
    try:
        measure(registration, output, progress)
        verify_inputs(registration)
        verify_completion(registration, output, progress)
        result["status"], code = "COMPLETE_ORDERED_PAIR_ENGINEERING_ONLY", 0
    except BaseException as error:
        from pokezero.mcts_eval.policy_opponent_profile import fallback_refusal_diagnostic, refusal_diagnostic
        result.update(status="FAILED_NO_RETRY", error_type=type(error).__name__,
            native_diagnostic=refusal_diagnostic(error), engine_fallback_diagnostic=fallback_refusal_diagnostic(error),
            failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)])
        mode = progress.get("current_mode")
        if progress["source_status"] == "ATTEMPTED_UNCERTAIN":
            progress["source_status"] = "REFUSED_UNCERTAIN"
        if mode and progress["modes"][mode]["status"] == "ATTEMPTED_UNCERTAIN":
            progress["modes"][mode]["status"] = "REFUSED_UNCERTAIN"
        code = 1
    result.update(**progress, exit_code=code, elapsed_seconds=time.perf_counter() - started)
    save_new(output / "terminal.json", result)
    print(json.dumps(dict(status=result["status"], stage=progress["stage"], exit_code=code)), flush=True)
    return code


def load_binding(output):
    return json.loads((output / "registration-binding.json").read_text())["registration_sha256"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("register")
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--namespace", required=True)
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
