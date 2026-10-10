"""One excluded raw source game and two non-opening paired-continuation roots.

Register against clean source/native bindings first, then run once. This is
engineering qualification, never exploration, held-out admission or strength.
No historical pool, accepted prefix or model is resumed or changed.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time
import traceback

from qualify_search_over_raw_opening import prepare_binding, save_new, verify_inputs, ARMS, FIXTURE_SEED
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest, require, paired_continuations, root_contrast)

NAMESPACE = "b969773c-de77-42ca-a67a-1ce83b60f7a9"
SCHEMA = "pokezero.search-over-raw.source-qualification.v1"


def engineering_contract():
    return dict(namespace=NAMESPACE, candidate_seat="p1", exclude_opening_requests=True,
        panels={"excluded": dict(seeds=[FIXTURE_SEED], root_slots=[
            dict(root_id=f"excluded:{FIXTURE_SEED}:{i}", source_seed=FIXTURE_SEED, root_slot=i)
            for i in range(2)])}, continuation_replicates=8,
        source_policy="raw_argmax_both_seats",
        root_rule="priority sampling over all non-opening requests of one complete excluded raw source game",
        continuation_policy="raw_argmax_after_initial_sampled_opponent_reply",
        max_source_boundaries=250, max_continuation_boundaries=250,
        scientific_strength_evidence=False, phase_a_admission=False, retry_authorized=False)


def register(*, output, **kwargs):
    registration = prepare_binding(**kwargs)
    registration.update(schema=SCHEMA, source_contract=engineering_contract(),
        scope="one excluded raw source game; two non-opening roots; eight continuations per unique selected action")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", registration)
    return registration


def claim_attempt(output, registration):
    require(registration["schema"] == SCHEMA and registration["source_contract"] == engineering_contract()
        and registration["fixture_seed"] == FIXTURE_SEED
        and registration["seeds"] == list(ENGINEERING_EXCLUDED_SEEDS)
        and registration["candidate_seat"] == "p1"
        and registration["retry_authorized"] is False and registration["phase_a_admission"] is False
        and registration["scientific_strength_evidence"] is False,
        "qualification changed or admits scientific outcomes")
    configs = [SearchConfiguration(**c) for c in registration["configurations"]]
    require(tuple(c.arm for c in configs) == ARMS and all(c.belief == "public" and c.leaf == "model"
        and c.workers == (20 if c.arm == "reference" else 1) for c in configs)
        and len({c.seconds for c in configs}) == 1 and configs[0].seconds in (1., 3., 10.),
        "qualification arm/resource drift")
    save_new(output / "attempt.json", dict(status="CLAIMED_BEFORE_RUNTIME",
        registration_sha256=digest(registration), retry_authorized=False))
    return configs


def run(output):
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    verify_inputs(registration)
    configs = claim_attempt(output, registration)
    started = time.perf_counter()
    env = archive = None
    adapters = {}
    completed_roots = []
    stage = "runtime_import"
    try:
        import sys
        import torch
        import poke_engine
        import pokezero_search
        require(sys.executable == registration["python_executable"] and torch.__version__ == registration["torch_version"]
            and torch.__file__ == registration["torch_path"], "Python/PyTorch identity drift")
        require(getattr(pokezero_search, "MODEL_FEATURE_ENABLED", False) is True, "native model feature absent")
        for module, name in ((poke_engine, "poke_engine"), (pokezero_search, "pokezero_search")):
            require(Path(module.__file__).resolve() == Path(registration["packages_root"]) / name / "__init__.py",
                "native package resolves outside isolated build")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from pokezero.collection import env_config_with_policy_spec_masks
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.neural_policy import load_transformer_policy
        from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.mcts_eval.search_over_raw_archive import SealedSourceArchive
        from pokezero.mcts_eval.search_over_raw_source import AuditedRawPolicy, collect_raw_source
        stage = "source_construct"
        checkpoint, showdown = registration["checkpoint"], registration["showdown_root"]
        contract = resolve_checkpoint_contract(checkpoint, expected_sha256=registration["checkpoint_sha256"],
            showdown_root=showdown, showdown_source_sha256=registration["set_source_hash"],
            model_path=registration["model_path"], tables_path=registration["encoder_tables"])
        require(contract.to_manifest() == registration["checkpoint_contract"], "checkpoint contract drift")
        config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(showdown),
            set_belief_source=True), [f"neural:{checkpoint}"], context="excluded raw source qualification")
        env = LocalShowdownEnv(config)
        archive = SealedSourceArchive()
        policies = {}
        for seat in ("p1", "p2"):
            policy = load_transformer_policy(Path(checkpoint), device="cpu", deterministic=True,
                exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
            require(policy.result.belief_set_source_hash == registration["set_source_hash"], "champion catalog drift")
            policies[seat] = AuditedRawPolicy(policy, checkpoint_sha256=registration["checkpoint_sha256"],
                public_context_sink=archive.capture_public if seat == "p1" else None)

        def progress(event):
            if event.decision_round_count == 1 or event.decision_round_count % 10 == 0 or event.terminal:
                row = dict(stage="source", boundary_count=event.decision_round_count,
                    elapsed_seconds=time.perf_counter()-started, source_terminal=event.terminal)
                save_new(output / f"source-progress-{event.decision_round_count:03d}.json", row)
                print(json.dumps(row), flush=True)

        stage = "source_collect"
        source = collect_raw_source(registration["source_contract"], panel="excluded", source_seed=FIXTURE_SEED,
            env=env, policies=policies, max_decision_rounds=250, decision_sink=progress,
            sealed_pre_step_sink=archive.capture_private)
        verify_inputs(registration)
        save_new(output / "source.json", source)
        require(source["status"] == "COMPLETE" and not source["missing_root_ids"]
            and len(source["roots"]) == 2, "source incomplete; no redraw or partial qualification")
        evaluator = ChampionEvaluator(policies["p1"].policy)
        factory = ShowdownWorkerFactory(checkpoint, registration["checkpoint_sha256"], showdown,
            registration["set_source_hash"], **registration["factory_options"])
        for root in source["roots"]:
            context, pending, snapshot = archive.selected(root)
            root_id = root["root_id"]
            root_dir = output / f"root-{root_id.rsplit(':', 1)[1]}"
            root_dir.mkdir()
            actions = {}
            for cfg in configs:
                verify_inputs(registration)
                stage = root_id + ":" + cfg.arm
                save_new(root_dir / (cfg.arm + "-attempt.json"), dict(status="CLAIMED_BEFORE_SELECTION",
                    configuration=asdict(cfg), source_request_index=root["source_request_index"], retry_authorized=False))
                if cfg.arm not in adapters:
                    began = time.perf_counter()
                    adapters[cfg.arm] = PublicModelSearchAdapter(cfg, checkpoint_contract=contract,
                        showdown_root=showdown, evaluator=evaluator if cfg.arm == "raw" else None,
                        reference_factory=factory if cfg.arm == "reference" else None,
                        initial_dispatch_workers=registration["initial_dispatch_workers"])
                    save_new(output / (cfg.arm + "-construction.json"),
                        dict(seconds=time.perf_counter()-began, runtime_configuration=adapters[cfg.arm].runtime_configuration))
                selected = adapters[cfg.arm].select(context, root_id=root_id + ":" + cfg.arm,
                    selection_seed=FIXTURE_SEED, pending_transition=pending)
                actions[cfg.arm] = selected["action"]
                save_new(root_dir / (cfg.arm + "-selected.json"), selected)
                print(json.dumps(dict(stage=stage, action=selected["action"], seconds=selected["elapsed_seconds"])), flush=True)
            require(actions["raw"] == root["public_record"]["recorded_action_index"], "raw source/evaluator action drift")
            # The true source boundary enters only the auditor, AFTER all arms
            # select from public contexts. No source opponent choice is replayed.
            stage = root_id + ":continuations"
            save_new(root_dir / "continuation-attempt.json", dict(actions=actions, replicates=8,
                maximum_boundaries=250, retry_authorized=False))

            continuation_started = time.perf_counter()

            def outcome_sink(row):
                row = dict(row, cumulative_seconds=time.perf_counter()-continuation_started)
                save_new(root_dir / f"outcome-{row['action']}-{row['replicate']}.json", row)
                print(json.dumps(dict(stage="continuation", **row)), flush=True)

            def raw_evaluator(observation):
                legal, evaluation = evaluator(observation)
                return legal, evaluation.priors

            audit = paired_continuations(env=env, snapshot=snapshot, subject="p1", actions=actions,
                evaluator=raw_evaluator, namespace=NAMESPACE, root_id=root_id, max_boundaries=250,
                outcome_sink=outcome_sink)
            audit.update(contrasts={arm: root_contrast(audit, arm) for arm in ARMS[1:]},
                scientific_strength_evidence=False, source_public_record_sha256=root["public_record_sha256"])
            verify_inputs(registration)
            save_new(root_dir / "audit.json", audit)
            completed_roots.append(dict(root_id=root_id, source_request_index=root["source_request_index"],
                actions=actions, status=audit["status"], contrasts=audit["contrasts"],
                continuation_count=len(audit["outcomes"])))
            require(audit["status"] == "COMPLETE", "capped continuation; no complete-case qualification")
        stage = "close"
        for adapter in adapters.values():
            adapter.close()
        archive.close()
        env.close()
        env = None
        verify_inputs(registration)
        save_new(output / "terminal.json", dict(status="COMPLETE_SOURCE_CONTINUATION_TECHNICAL_QUALIFICATION_ONLY",
            registration_sha256=digest(registration), roots=completed_roots,
            elapsed_seconds=time.perf_counter()-started, scientific_strength_evidence=False, phase_a_admission=False))
        return 0
    except BaseException as error:
        save_new(output / "terminal.json", dict(status="FAILED_NO_RETRY", stage=stage,
            error_type=type(error).__name__, elapsed_seconds=time.perf_counter()-started,
            completed_roots=completed_roots, failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)], registration_sha256=digest(registration),
            retry_authorized=False, scientific_strength_evidence=False, phase_a_admission=False))
        print(json.dumps(dict(status="FAILED_NO_RETRY", stage=stage, error_type=type(error).__name__)), flush=True)
        return 1
    finally:
        try:
            for adapter in adapters.values():
                adapter.close()
        finally:
            if archive is not None:
                archive.close()
            if env is not None:
                env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("register")
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--seconds", type=float, choices=(1., 3., 10.), default=1.)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "run":
        return run(**args)
    result = register(**args)
    print(json.dumps(dict(status=result["status"], registration_sha256=digest(result))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
