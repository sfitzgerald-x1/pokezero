"""One prospectively reviewed staged A1 engineering attempt; never a retry.

Four nonopening roots from two fixed fresh raw games. On each root run raw,
incumbent/public, incumbent/oracle, reference/public, reference/oracle with
model leaves at 1s and unchanged full allocations, then eight paired terminal
continuations. Later evaluators/budgets do not gate this information diagnostic.
This does not replace any scientific registration or open the holdout. Failed
attempts remain closed; zero complete comparison is failed qualification.
"""
from dataclasses import asdict
import argparse
import json
import os
import sys
from pathlib import Path
import time
import traceback

from qualify_search_over_raw_opening import prepare_binding, verify_inputs
from benchmark_search_over_raw_feasibility import save_new, close_resources, _runtime, selection_work
from pokezero.mcts_eval.resolver import sha256_file
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest, paired_continuations,
    require, root_contrast, select_source_requests,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pokezero.search-over-raw.staged-a1-engineering.v1"
NAMESPACE = "93e6f244-397b-47c6-a7b9-841d17efc73b"
SEEDS = (2026101024, 2026101025)
ROOTS_PER_SEED = 2
SECONDS = 1.


def configurations():
    return [SearchConfiguration("raw", seconds=SECONDS)] + [
        SearchConfiguration(arm, belief, "model", SECONDS, 20 if arm == "reference" else 1)
        for arm in ("incumbent", "reference") for belief in ("public", "oracle")]


def key(cfg):
    return "raw" if cfg.arm == "raw" else cfg.identity


def source_contract():
    return dict(namespace=NAMESPACE, candidate_seat="p1", exclude_opening_requests=True,
        panels={"excluded": dict(seeds=list(SEEDS), root_slots=[
            dict(root_id=f"excluded:{seed}:{slot}", source_seed=seed, root_slot=slot)
            for seed in SEEDS for slot in range(ROOTS_PER_SEED)])},
        continuation_replicates=8, source_policy="raw_argmax_both_seats",
        source_root_rule="fixed seed-derived priorities over completed full nonopening catalog",
        max_source_boundaries=250, max_continuation_boundaries=250,
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        replacements_permitted=False, retry_authorized=False,
        stop_policy="first source, selection, continuation, provenance or worker failure stops attempt",
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False)


def bound_files():
    return [Path(__file__).resolve(), ROOT / "scripts/qualify_search_over_raw_opening.py",
        ROOT / "scripts/benchmark_search_over_raw_feasibility.py", *[
        ROOT / "src/pokezero" / name for name in (
            "engine_search.py", "engine_world.py", "mcts_eval/search_over_raw.py",
            "mcts_eval/search_over_raw_adapters.py", "mcts_eval/search_over_raw_archive.py",
            "mcts_eval/search_over_raw_source.py", "mcts_eval/search_over_raw_oracle.py",
            "mcts_eval/search_over_raw_belief_diagnostics.py")]]


def exposure_inventory(paths):
    require(paths, "explicit prior exposure registrations required")
    seeds, bindings = set(ENGINEERING_EXCLUDED_SEEDS) | set(range(4711, 4736)), {}
    for supplied in paths:
        path = Path(supplied).resolve()
        value = json.loads(path.read_text())
        roster = value.get("seeds")
        require(type(roster) is list and roster and all(type(s) is int and 0 <= s < 2**32
            for s in roster), "exposure registration requires explicit uint32 seeds")
        seeds.update(roster)
        bindings[str(path)] = sha256_file(path)
    require(not set(SEEDS) & seeds, "staged A1 seed exposure overlap; no redraw or retry")
    return sorted(seeds), bindings


def register(*, output, exposure_registrations, **kwargs):
    excluded, bindings = exposure_inventory(exposure_registrations)
    r = prepare_binding(**kwargs, seconds=SECONDS)
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT), "attempt output must be outside executing checkout")
    r.update(schema=SCHEMA, namespace=NAMESPACE, seeds=list(SEEDS), fixture_seed=SEEDS[0],
        configurations=[asdict(c) for c in configurations()], source_contract=source_contract(),
        attempt_directory=str(output), exposure_registrations=bindings, excluded_seeds=excluded,
        excluded_seed_inventory_sha256=digest(excluded),
        exposure_scope="supplied registrations plus minimum exclusions; not a universal census",
        required_diagnostics="both arms actual attempted original-team traits; controller-only truth join",
        qualifies_uninstrumented_runtime=False, representative_runtime_evidence=False,
        historical_attempt_reused=False, required_independent_review=True,
        sequencing="A1 engineering before further inference calibration/A2/A3; no scientific scope reduction",
        implementation_deadline="one working-day tranche; stop and identify dependency if incomplete",
        scope="staged A1 complete paired comparisons, excluded engineering only; no strength admission")
    r["input_sha256"].update(bindings)
    r["input_sha256"].update({str(p): sha256_file(p) for p in bound_files()})
    validate_contract(r, output)
    verify_inputs(r)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", r)
    save_new(output / "registration-binding.json", dict(registration_sha256=digest(r)))
    return r


def validate_contract(r, output):
    require(r["schema"] == SCHEMA and r["namespace"] == NAMESPACE
        and digest(r["source_contract"]) == digest(source_contract())
        and digest(r["configurations"]) == digest([asdict(c) for c in configurations()])
        and r["seeds"] == list(SEEDS) and all(type(s) is int for s in r["seeds"])
        and type(r["fixture_seed"]) is int and r["fixture_seed"] == SEEDS[0]
        and r["candidate_seat"] == "p1" and type(r["initial_dispatch_workers"]) is int
        and r["initial_dispatch_workers"] == 6 and r["required_independent_review"] is True
        and all(r[field] is False for field in ("retry_authorized", "phase_a_admission",
            "phase_b_authorized", "scientific_strength_evidence", "qualifies_uninstrumented_runtime",
            "representative_runtime_evidence", "historical_attempt_reused")), "staged A1 core drift")
    excluded, exposure = exposure_inventory(list(r["exposure_registrations"]))
    require(exposure == r["exposure_registrations"] and excluded == r["excluded_seeds"]
        and digest(excluded) == r["excluded_seed_inventory_sha256"]
        and all(r["input_sha256"].get(p) == s for p, s in exposure.items()), "exposure binding drift")
    require(Path(r["source_root"]).resolve() == ROOT
        and all(r["input_sha256"].get(str(p)) == sha256_file(p) for p in bound_files())
        and r["attempt_directory"] == str(Path(output).resolve())
        and not Path(output).resolve().is_relative_to(ROOT), "staged driver/attempt binding drift")


def planned_cells():
    return [dict(root_id=root["root_id"], configuration=key(cfg),
        status="UNSTARTED_UNCERTAIN", contrast_interval=[-1., 1.], action=None)
        for root in source_contract()["panels"]["excluded"]["root_slots"] for cfg in configurations()]


def planned_continuations():
    return [dict(root_id=root["root_id"], configuration=key(cfg), replicate=replicate,
        status="UNSTARTED_UNCERTAIN", action=None, signed_outcome=None)
        for root in source_contract()["panels"]["excluded"]["root_slots"]
        for cfg in configurations() for replicate in range(8)]


def validate_source(source, seed):
    rows = source["eligible_public_records"]
    indices = [r["source_request_index"] for r in rows]
    require(source["contract_sha256"] == digest(source_contract()) and source["source_seed"] == seed
        and source["panel"] == "excluded" and source["status"] == "COMPLETE"
        and source["source_terminal_complete"] is True and source["source_policy"] == "raw_argmax_both_seats"
        and source["eligible_requests"] == len(rows) and digest(rows) == source["eligible_catalog_sha256"]
        and all(type(i) is int and i > 0 for i in indices) and len(set(indices)) == len(indices)
        and source["requested_root_slots"] == ROOTS_PER_SEED and source["missing_root_ids"] == [],
        "source catalog incomplete or drifted")
    selected = select_source_requests(NAMESPACE, seed, indices, ROOTS_PER_SEED)
    catalog = {r["source_request_index"]: r["public_record_sha256"] for r in rows}
    require(len(source["roots"]) == len(selected) == ROOTS_PER_SEED, "missing roots; no replacement")
    for slot, (root, index) in enumerate(zip(source["roots"], selected)):
        require(root["root_id"] == f"excluded:{seed}:{slot}" and root["source_request_index"] == index
            and digest(root["public_record"]) == root["public_record_sha256"] == catalog[index],
            "source priority selection differs")


def make_adapter(cfg, *, oracle, **kwargs):
    from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
    from pokezero.mcts_eval.search_over_raw_belief_diagnostics import (
        PublicBeliefDiagnosticSearchAdapter, OracleBeliefDiagnosticSearchAdapter)
    if cfg.arm == "raw":
        return PublicModelSearchAdapter(cfg, **kwargs)
    if cfg.belief == "oracle":
        return OracleBeliefDiagnosticSearchAdapter(cfg, oracle=oracle, **kwargs)
    return PublicBeliefDiagnosticSearchAdapter(cfg, **kwargs)


def agreement(selected, truth, request, cfg):
    from pokezero.mcts_eval.paper_reference_showdown import decision_state
    from pokezero.mcts_eval.search_over_raw_belief_diagnostics import compare_sampled_teams
    information = decision_state(request.observation, player=request.state.player_id).key.hex()
    evidence = selected["evidence"]
    ledgers = [evidence["belief_draws"]] if cfg.arm == "incumbent" else [
        row["evidence"]["draws"] for row in evidence["worker_receipts"]]
    require(bool(ledgers), "missing actual-world diagnostic ledger")
    return [compare_sampled_teams(draws, truth=truth, information_key=information) for draws in ledgers]


def collect_root(*, root, context, pending, snapshot, output, progress, env, evaluator,
                 factory, contract, source, showdown, verify):
    from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
    from pokezero.mcts_eval.search_over_raw_oracle import TeamOracle
    from pokezero.mcts_eval.search_over_raw_belief_diagnostics import truth_record
    request = PublicRootRequest.capture(context.public_materialization_state, context.observation,
        pending_transition=pending)
    oracle = TeamOracle.capture(snapshot, request, source)
    truth = truth_record(oracle, request=request, source=source)
    save_new(output / "controller-truth.json", truth)
    from pokezero.mcts_eval.paper_reference_showdown import decision_state
    save_new(output / "root-binding.json", dict(root_id=root["root_id"],
        public_record_sha256=root["public_record_sha256"], root_binding=truth["root_binding"],
        legal_choices=sum(context.observation.legal_action_mask),
        information_key=decision_state(request.observation, player=request.state.player_id).key.hex()))
    actions = {}
    for cfg in configurations():
        name = key(cfg)
        cell = next(c for c in progress["fixed_roster"] if c["root_id"] == root["root_id"]
            and c["configuration"] == name)
        adapter = None
        progress["stage"] = root["root_id"] + ":" + name + ":construct"
        cell["status"] = "CONSTRUCTION_ATTEMPTED_UNCERTAIN"
        save_new(output / f"{name}-attempt.json", dict(root_id=root["root_id"],
            configuration=asdict(cfg), retry_authorized=False))
        verify()
        began = time.perf_counter()
        try:
            # Public actors receive NO oracle, source env or snapshot. The
            # constructor helper does not forward oracle except on oracle arm.
            adapter = make_adapter(cfg, oracle=oracle, checkpoint_contract=contract,
                showdown_root=showdown, evaluator=evaluator if cfg.arm == "raw" else None,
                reference_factory=factory if cfg.arm == "reference" else None, initial_dispatch_workers=6)
            construction = time.perf_counter() - began
            save_new(output / f"{name}-runtime.json", dict(runtime_configuration=adapter.runtime_configuration,
                construction_seconds=construction))
            progress["stage"] = root["root_id"] + ":" + name + ":select"
            cell["status"] = "SELECTION_ATTEMPTED_UNCERTAIN"
            selected = adapter.select(context, root_id=root["root_id"] + ":" + name,
                selection_seed=root["public_record"]["seed"], pending_transition=pending)
            verify()
            save_new(output / f"{name}-selected.json", selected)
            if cfg.arm != "raw":
                save_new(output / f"{name}-agreement.json", agreement(selected, truth, request, cfg))
            work = selection_work(cfg, selected, sum(context.observation.legal_action_mask))
            save_new(output / f"{name}-work.json", work)
            actions[name] = selected["action"]
            cell.update(status="SELECTED_UNMEASURED", action=selected["action"])
            progress["selections_completed"] += 1
            progress["stage"] = root["root_id"] + ":" + name + ":close"
            close_start = time.perf_counter()
            adapter.close()
            adapter = None
            save_new(output / f"{name}-close.json", dict(seconds=time.perf_counter()-close_start,
                end_to_end_seconds=time.perf_counter()-began,
                nominal_ceiling_scope="selection only; startup/cleanup and overshoot reported separately"))
            print(json.dumps(dict(stage=progress["stage"], seconds=selected["elapsed_seconds"])), flush=True)
        finally:
            primary_active = sys.exc_info()[0] is not None
            try:
                if adapter is not None and adapter.last_failure is not None:
                    save_new(output / f"{name}-failure.json", adapter.last_failure)
            except BaseException as error:
                progress.setdefault("evidence_write_failures", []).append(type(error).__name__)
                if not primary_active:
                    raise
            finally:
                close_resources((("adapter", adapter),), progress)
    require(actions["raw"] == root["public_record"]["recorded_action_index"], "raw source action drift")
    progress["stage"] = root["root_id"] + ":continuations"
    save_new(output / "continuation-attempt.json", dict(actions=actions, replicates=8, maximum_boundaries=250))

    def attempt(row):
        save_new(output / f"continuation-{row['action']}-{row['replicate']}-attempt.json", row)
        for cell in progress["continuation_roster"]:
            if (cell["root_id"] == root["root_id"] and cell["replicate"] == row["replicate"]
                    and actions[cell["configuration"]] == row["action"]):
                cell.update(status="ATTEMPTED_UNCERTAIN", action=row["action"])

    def outcome(row):
        save_new(output / f"outcome-{row['action']}-{row['replicate']}.json", row)
        for cell in progress["continuation_roster"]:
            if (cell["root_id"] == root["root_id"] and cell["replicate"] == row["replicate"]
                    and actions[cell["configuration"]] == row["action"]):
                cell.update(status=row["status"], action=row["action"], signed_outcome=row["signed_outcome"])
        progress["continuations_completed"] += int(row["status"] == "COMPLETE")
        require(row["status"] == "COMPLETE", "capped/refused continuation stops attempt")

    def raw_evaluator(observation):
        legal, evaluation = evaluator(observation)
        return legal, evaluation.priors

    audit = paired_continuations(env=env, snapshot=snapshot, subject="p1", actions=actions,
        evaluator=raw_evaluator, namespace=NAMESPACE, root_id=root["root_id"], max_boundaries=250,
        outcome_sink=outcome, attempt_sink=attempt)
    audit.update(contrasts={name: list(root_contrast(audit, name)) for name in actions},
        scientific_strength_evidence=False)
    verify()
    save_new(output / "audit.json", audit)
    for cell in progress["fixed_roster"]:
        if cell["root_id"] == root["root_id"]:
            cell.update(status="COMPLETE_ENGINEERING_ONLY", contrast_interval=audit["contrasts"][cell["configuration"]])
    progress["roots_completed"] += 1


def measure(r, output, progress):
    _runtime(r)
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
    from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
    from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy, collect_raw_source
    checkpoint, showdown = r["checkpoint"], r["showdown_root"]
    contract = resolve_checkpoint_contract(checkpoint, expected_sha256=r["checkpoint_sha256"],
        showdown_root=showdown, showdown_source_sha256=r["set_source_hash"],
        model_path=r["model_path"], tables_path=r["encoder_tables"])
    require(contract.to_manifest() == r["checkpoint_contract"], "checkpoint contract drift")
    source_catalog = load_gen3_randbat_source_cached(showdown)
    factory = ShowdownWorkerFactory(checkpoint, r["checkpoint_sha256"], showdown,
        r["set_source_hash"], **r["factory_options"])
    for seed in SEEDS:
        env = archive = None
        directory = output / f"source-{seed}"
        directory.mkdir()
        progress["stage"] = f"source:{seed}:construct"
        progress["source_roster"][str(seed)]["status"] = "CONSTRUCTION_ATTEMPTED_UNCERTAIN"
        try:
            config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(showdown),
                set_belief_source=True), [f"neural:{checkpoint}"], context="staged A1 excluded engineering")
            env, archive = LocalShowdownEnv(config), SealedSourceArchive()
            policies = {}
            for seat in ("p1", "p2"):
                policy = load_transformer_policy(Path(checkpoint), device="cpu", deterministic=True,
                    exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
                require(policy.result.belief_set_source_hash == r["set_source_hash"], "champion catalog drift")
                policies[seat] = AuditedRawPolicy(policy, checkpoint_sha256=r["checkpoint_sha256"],
                    public_context_sink=archive.capture_public if seat == "p1" else None)
            verify_inputs(r)
            save_new(directory / "attempt.json", dict(seed=seed, retry_authorized=False))
            progress["stage"] = f"source:{seed}:collect"
            progress["source_roster"][str(seed)]["status"] = "ATTEMPTED_UNCERTAIN"
            def source_progress(event):
                if event.decision_round_count == 1 or event.decision_round_count % 10 == 0 or event.terminal:
                    row = dict(seed=seed, boundaries=event.decision_round_count, terminal=event.terminal)
                    save_new(directory / f"progress-{event.decision_round_count:03d}.json", row)
                    print(json.dumps(row), flush=True)
            receipt = collect_raw_source(source_contract(), panel="excluded", source_seed=seed,
                env=env, policies=policies, max_decision_rounds=250, decision_sink=source_progress,
                sealed_pre_step_sink=archive.capture_private)
            save_new(directory / "source.json", receipt)
            validate_source(receipt, seed)
            progress["source_roster"][str(seed)]["status"] = "COMPLETE"
            progress["source_games_completed"] += 1
            evaluator = ChampionEvaluator(policies["p1"].policy)
            for root in receipt["roots"]:
                context, pending, snapshot = archive.selected(root)
                root_dir = directory / f"root-{root['root_id'].rsplit(':', 1)[1]}"
                root_dir.mkdir()
                collect_root(root=root, context=context, pending=pending, snapshot=snapshot,
                    output=root_dir, progress=progress, env=env, evaluator=evaluator, factory=factory,
                    contract=contract, source=source_catalog, showdown=showdown, verify=lambda: verify_inputs(r))
        finally:
            close_resources((("archive", archive), ("source_env", env)), progress)


def verify_completion(output, progress):
    selectors = outcomes = roots = aliases = 0
    substantive = set()
    for seed in SEEDS:
        directory = output / f"source-{seed}"
        source = json.loads((directory / "source.json").read_text())
        validate_source(source, seed)
        require(progress["source_roster"][str(seed)]["status"] == "COMPLETE", "source accounting mismatch")
        for root in source["roots"]:
            d = directory / f"root-{root['root_id'].rsplit(':', 1)[1]}"
            load = lambda name: json.loads((d / name).read_text())
            audit = load("audit.json")
            require(audit["root_id"] == root["root_id"] and audit["status"] == "COMPLETE"
                and set(audit["actions"]) == {key(c) for c in configurations()}, "incomplete staged action roster")
            truth = load("controller-truth.json")
            binding = load("root-binding.json")
            require(binding["root_id"] == root["root_id"]
                and binding["public_record_sha256"] == root["public_record_sha256"]
                and binding["root_binding"] == truth["root_binding"], "root/truth binding drift")
            for cfg in configurations():
                name = key(cfg)
                selected, runtime, close = load(f"{name}-selected.json"), load(f"{name}-runtime.json"), load(f"{name}-close.json")
                require(selected["status"] == "SELECTED" and selected["root_id"] == root["root_id"] + ":" + name
                    and selected["configuration_sha256"] == cfg.identity
                    and selected["runtime_sha256"] == digest(runtime["runtime_configuration"])
                    and digest(runtime["runtime_configuration"]["configuration"]) == digest(asdict(cfg))
                    and selected["action"] == audit["actions"][name] and close["seconds"] >= 0,
                    "durable selection/runtime/cleanup mismatch")
                interval = list(root_contrast(audit, name))
                work = selection_work(cfg, selected, binding["legal_choices"])
                require(work == load(f"{name}-work.json"), "durable actual-work accounting mismatch")
                if work["substantive_search_exercised"]:
                    substantive.add(name)
                require(audit["contrasts"][name] == interval, "durable paired contrast mismatch")
                cell = next(c for c in progress["fixed_roster"] if c["root_id"] == root["root_id"]
                    and c["configuration"] == name)
                require(cell["status"] == "COMPLETE_ENGINEERING_ONLY" and cell["action"] == selected["action"]
                    and cell["contrast_interval"] == interval, "fixed roster disagrees with durable outcomes")
                if cfg.arm != "raw":
                    from pokezero.mcts_eval.search_over_raw_belief_diagnostics import compare_sampled_teams
                    e = selected["evidence"]
                    ledgers = [e["belief_draws"]] if cfg.arm == "incumbent" else [
                        row["evidence"]["draws"] for row in e["worker_receipts"]]
                    require(bool(ledgers), "missing actual-world ledgers")
                    expected = [compare_sampled_teams(rows, truth=truth,
                        information_key=binding["information_key"]) for rows in ledgers]
                    require(expected == load(f"{name}-agreement.json"), "durable team agreement mismatch")
                for replicate in range(8):
                    row = next(row for row in audit["outcomes"] if row["action"] == selected["action"]
                        and row["replicate"] == replicate)
                    alias = next(c for c in progress["continuation_roster"] if c["root_id"] == root["root_id"]
                        and c["configuration"] == name and c["replicate"] == replicate)
                    require(alias["status"] == "COMPLETE" and alias["action"] == selected["action"]
                        and alias["signed_outcome"] == row["signed_outcome"], "paired alias accounting mismatch")
                    aliases += 1
                selectors += 1
            for row in audit["outcomes"]:
                require(row["status"] == "COMPLETE" and load(f"outcome-{row['action']}-{row['replicate']}.json")
                    == dict(root_id=root["root_id"], **row)
                    and load(f"continuation-{row['action']}-{row['replicate']}-attempt.json") == dict(
                        root_id=root["root_id"], action=row["action"], replicate=row["replicate"]),
                    "durable terminal outcome/attempt mismatch")
                outcomes += 1
            roots += 1
    require(substantive == {key(c) for c in configurations() if c.arm != "raw"},
        "NO_SUBSTANTIVE_A1_EXERCISE; both public/oracle arms required, forced roots retained, no redraw")
    require(progress["source_games_completed"] == len(SEEDS) and progress["roots_completed"] == roots == 4
        and progress["selections_completed"] == selectors == 20 and progress["continuations_completed"] == outcomes
        and len(progress["fixed_roster"]) == 20 and len({(c["root_id"], c["configuration"])
            for c in progress["fixed_roster"]}) == 20 and len(progress["continuation_roster"]) == aliases == 160
        and len({(c["root_id"], c["configuration"], c["replicate"])
            for c in progress["continuation_roster"]}) == 160, "full staged work accounting mismatch")


def run(output, review):
    output = Path(output).resolve()
    r = json.loads((output / "registration.json").read_text())
    require(json.loads((output / "registration-binding.json").read_text()) == dict(registration_sha256=digest(r)),
        "registration binding drift")
    validate_contract(r, output)
    verify_inputs(r)
    review = Path(review).resolve()
    clearance = json.loads(review.read_text())
    require(clearance.get("disposition") == "CLEAR_ONE_STAGED_A1_ENGINEERING_ATTEMPT"
        and clearance.get("registration_sha256") == digest(r) and clearance.get("source_commit") == r["source_commit"]
        and clearance.get("independent_reviewer") and clearance.get("scientific_admission") is False,
        "bound independent staged review required")
    require(not (output / "attempt.json").exists() and not (output / "terminal.json").exists(),
        "claimed or closed staged attempt; no retry")
    save_new(output / "attempt.json", dict(pid=os.getpid(), registration_sha256=digest(r),
        review_path=str(review), review_sha256=sha256_file(review), status="CLAIMED_BEFORE_RUNTIME", retry_authorized=False))
    began = time.perf_counter()
    progress = dict(stage="runtime_import", source_games_completed=0, roots_completed=0,
        selections_completed=0, continuations_completed=0, fixed_roster=planned_cells(),
        continuation_roster=planned_continuations(), source_roster={str(seed): dict(
            status="UNSTARTED_UNCERTAIN") for seed in SEEDS})
    code, error = 0, None
    try:
        measure(r, output, progress)
        verify_inputs(r)
        verify_completion(output, progress)
    except BaseException as failure:
        code = 1
        error = dict(error_type=type(failure).__name__, failure_frames=[dict(file=f.filename,
            line=f.lineno, function=f.name) for f in traceback.extract_tb(failure.__traceback__)])
    save_new(output / "terminal.json", dict(**progress, exit_code=code, error=error,
        status="COMPLETE_STAGED_A1_ENGINEERING_ONLY" if code == 0 else "FAILED_NO_RETRY",
        elapsed_seconds=time.perf_counter()-began, registration_sha256=digest(r), retry_authorized=False,
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False,
        holdout_opened=False, qualifies_uninstrumented_runtime=False))
    print(json.dumps(dict(exit_code=code, roots_completed=progress["roots_completed"], stage=progress["stage"])), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--review", type=Path, required=True)
    prep = sub.add_parser("register")
    prep.add_argument("--exposure-registration", dest="exposure_registrations", type=Path, action="append", required=True)
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables"):
        prep.add_argument("--"+name, type=Path, required=True)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256"):
        prep.add_argument("--"+name, required=True)
    args = vars(parser.parse_args(argv))
    if args.pop("command") == "run":
        return run(**args)
    r = register(**args)
    print(json.dumps(dict(registration_sha256=digest(r), status=r["status"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
