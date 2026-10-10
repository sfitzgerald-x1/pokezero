"""Excluded opening benchmark of real raw/incumbent/reference model adapters.

Register first, then run exactly once from clean, pinned sources. This measures
one public opening per arm, not terminal continuations or playing strength.
The retained scientific collection/runtime is never resumed or installed into.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from pokezero.mcts_eval.resolver import sha256_file
from pokezero.mcts_eval.search_over_raw import (
    ENGINEERING_EXCLUDED_SEEDS, SearchConfiguration, digest, require)


FIXTURE_SEED = ENGINEERING_EXCLUDED_SEEDS[0]
ARMS = ("raw", "incumbent", "reference")
ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def clean_identity(root, expected):
    require(git(root, "rev-parse", "HEAD") == expected, "source commit drift")
    require(not git(root, "status", "--porcelain"), "source worktree must be clean")


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def verify_inputs(registration):
    clean_identity(Path(registration["source_root"]), registration["source_commit"])
    clean_identity(Path(registration["showdown_root"]), registration["showdown_commit"])
    for path, expected in registration["input_sha256"].items():
        require(sha256_file(path) == expected, "benchmark input drift")


def register(*, output, checkpoint, checkpoint_sha256, showdown_root,
             showdown_commit, source_commit, native_receipt, factory_options,
             factory_options_sha256, encoder_tables, seconds=1.):
    require(type(seconds) in (int, float) and seconds in (1., 3., 10.),
        "opening benchmark requires a planned 1/3/10 second budget")
    clean_identity(ROOT, source_commit)
    showdown_root = Path(showdown_root).resolve()
    clean_identity(showdown_root, showdown_commit)
    checkpoint, native_receipt, factory_options = [Path(p).resolve() for p in
        (checkpoint, native_receipt, factory_options)]
    require(sha256_file(checkpoint) == checkpoint_sha256, "champion hash drift")
    require(sha256_file(factory_options) == factory_options_sha256, "factory options hash drift")
    receipt = json.loads(native_receipt.read_text())
    require(receipt["status"] == "BUILT_AND_CHAMPION_PARITY_VERIFIED_NOT_SEARCH_ADMISSION"
        and receipt["champion_sha256"] == checkpoint_sha256
        and receipt["libtorch_version_bypass_present"] is False,
        "unqualified build receipt or wrong champion")
    # A new Python driver may use the native build only if EVERY checked build
    # input is unchanged; the old build commit is retained, not relabeled.
    from engine_build_fingerprint import compute_fingerprint
    require(compute_fingerprint() == receipt["fingerprint"], "native build fingerprint drift")
    inputs = {str(ROOT / path): expected for path, expected in receipt["input_sha256"].items()}
    inputs.update(receipt["artifact_sha256"])
    inputs.update({str(p): sha256_file(p) for p in (checkpoint, native_receipt, factory_options)})
    model_paths = [p for p in receipt["artifact_sha256"] if p.endswith("/exports/model_ts.pt")]
    require(len(model_paths) == 1, "one isolated model export required")
    encoder_tables = Path(encoder_tables).resolve()
    inputs[str(encoder_tables)] = sha256_file(encoder_tables)
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    source_hash = load_gen3_randbat_source_cached(showdown_root).metadata.source_hash
    contract = resolve_checkpoint_contract(checkpoint, expected_sha256=checkpoint_sha256,
        showdown_root=showdown_root, showdown_source_sha256=source_hash,
        model_path=model_paths[0], tables_path=encoder_tables)
    configs = [SearchConfiguration(arm, seconds=seconds, workers=20 if arm == "reference" else 1)
        for arm in ARMS]
    registration = dict(schema="pokezero.search-over-raw.opening-benchmark.v1",
        status="REGISTERED_EXCLUDED_ENGINEERING_ONLY", source_root=str(ROOT),
        source_commit=source_commit, showdown_root=str(showdown_root), showdown_commit=showdown_commit,
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256,
        packages_root=receipt["packages_root"], native_build_source_commit=receipt["source_commit"],
        set_source_hash=source_hash, checkpoint_contract=contract.to_manifest(),
        model_path=model_paths[0], encoder_tables=str(encoder_tables),
        python_executable=receipt["python_executable"], torch_version=receipt["torch_version"],
        torch_path=receipt["torch_path"], input_sha256=inputs,
        factory_options=json.loads(factory_options.read_text()),
        configurations=[asdict(c) for c in configs], seeds=list(ENGINEERING_EXCLUDED_SEEDS),
        fixture_seed=FIXTURE_SEED, candidate_seat="p1", initial_dispatch_workers=6,
        root_statistics="independent", retry_authorized=False,
        scope="one opening request per arm; no continuation, strength score or Phase A panel",
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False)
    verify_inputs(registration)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", registration)
    return registration


def claim_attempt(output, registration):
    require(registration["schema"] == "pokezero.search-over-raw.opening-benchmark.v1"
        and registration["fixture_seed"] == FIXTURE_SEED
        and registration["seeds"] == list(ENGINEERING_EXCLUDED_SEEDS)
        and registration["candidate_seat"] == "p1"
        and registration["retry_authorized"] is False
        and registration["phase_a_admission"] is False
        and registration["scientific_strength_evidence"] is False,
        "benchmark contract changed or admits scientific outcomes")
    configs = [SearchConfiguration(**c) for c in registration["configurations"]]
    require(tuple(c.arm for c in configs) == ARMS and all(c.belief == "public" and c.leaf == "model"
        and c.workers == (20 if c.arm == "reference" else 1) for c in configs)
        and len({c.seconds for c in configs}) == 1 and configs[0].seconds in (1., 3., 10.),
        "benchmark arm/resource drift")
    save_new(output / "attempt.json", dict(status="CLAIMED_BEFORE_RUNTIME",
        registration_sha256=digest(registration), retry_authorized=False))
    return configs


def run(output):
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    verify_inputs(registration)
    configs = claim_attempt(output, registration)
    started = time.perf_counter()
    env = adapter = None
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
                "native package import resolves outside fresh build")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from pokezero.collection import env_config_with_policy_spec_masks
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.neural_policy import load_transformer_policy
        from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator
        from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
        from pokezero.mcts_eval.search_over_raw_adapters import PublicModelSearchAdapter
        from pokezero.policy import PolicyContext
        from pokezero.trajectory import BattleTrajectory
        stage = "opening_capture"
        checkpoint, showdown = registration["checkpoint"], registration["showdown_root"]
        config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(showdown),
            set_belief_source=True), [f"neural:{checkpoint}"], context="excluded Phase A opening qualification")
        env = LocalShowdownEnv(config)
        env.reset(seed=FIXTURE_SEED)
        public = env.public_materialization_state("p1")
        source_hash = public.belief_engine.set_source.metadata.source_hash
        require(source_hash == registration["set_source_hash"], "opening catalog drift")
        contract = resolve_checkpoint_contract(checkpoint, expected_sha256=registration["checkpoint_sha256"],
            showdown_root=showdown, showdown_source_sha256=source_hash,
            model_path=registration["model_path"], tables_path=registration["encoder_tables"])
        require(contract.to_manifest() == registration["checkpoint_contract"], "checkpoint contract drift")
        own = env.observe("p1")
        context = PolicyContext("p1", 0, "excluded-opening", "gen3randombattle", FIXTURE_SEED,
            own, tuple(env.requested_players()), BattleTrajectory("excluded-opening", "gen3randombattle", FIXTURE_SEED),
            requested_legal_action_masks={"p1": own.legal_action_mask},
            requested_observations={"p1": own}, public_materialization_state=public)
        evaluator = ChampionEvaluator(load_transformer_policy(Path(checkpoint), device="cpu", deterministic=True,
            exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False))
        require(evaluator.policy.result.belief_set_source_hash == source_hash,
            "champion belief source differs from opening catalog")
        factory = ShowdownWorkerFactory(checkpoint, registration["checkpoint_sha256"], showdown,
            source_hash, **registration["factory_options"])
        save_new(output / "root-binding.json", dict(checkpoint_contract=contract.to_manifest(),
            set_source_hash=source_hash, legal_actions=[i for i, x in enumerate(own.legal_action_mask) if x],
            scientific_strength_evidence=False))
        receipts = []
        for cfg in configs:
            verify_inputs(registration)
            stage = cfg.arm + ":construct"
            save_new(output / (cfg.arm + "-attempt.json"), dict(status="CLAIMED_BEFORE_CONSTRUCTION",
                configuration=asdict(cfg), retry_authorized=False))
            construction_started = time.perf_counter()
            adapter = PublicModelSearchAdapter(cfg, checkpoint_contract=contract, showdown_root=showdown,
                evaluator=evaluator if cfg.arm == "raw" else None,
                reference_factory=factory if cfg.arm == "reference" else None,
                initial_dispatch_workers=registration["initial_dispatch_workers"])
            construction_seconds = time.perf_counter() - construction_started
            stage = cfg.arm + ":select"
            row = adapter.select(context, root_id="excluded-opening:" + cfg.arm, selection_seed=FIXTURE_SEED)
            row.update(construction_seconds=construction_seconds,
                runtime_configuration=adapter.runtime_configuration, registration_sha256=digest(registration))
            verify_inputs(registration)
            save_new(output / (cfg.arm + "-selected.json"), row)
            receipts.append(dict(arm=cfg.arm, action=row["action"], elapsed_seconds=row["elapsed_seconds"],
                construction_seconds=construction_seconds, exceeded_nominal_seconds=row["exceeded_nominal_seconds"]))
            print(json.dumps(dict(status="OPENING_SELECTED", **receipts[-1])), flush=True)
            stage = cfg.arm + ":close"
            adapter.close()
            adapter = None
        save_new(output / "terminal.json", dict(status="COMPLETE_OPENING_TECHNICAL_BENCHMARK_ONLY",
            registration_sha256=digest(registration), selections=receipts,
            elapsed_seconds=time.perf_counter()-started, terminal_games=0,
            scientific_strength_evidence=False, phase_a_admission=False))
        return 0
    except BaseException as error:
        save_new(output / "terminal.json", dict(status="FAILED_NO_RETRY", stage=stage,
            error_type=type(error).__name__, elapsed_seconds=time.perf_counter()-started,
            failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)],
            registration_sha256=digest(registration), retry_authorized=False,
            scientific_strength_evidence=False, phase_a_admission=False))
        print(json.dumps(dict(status="FAILED_NO_RETRY", stage=stage, error_type=type(error).__name__)), flush=True)
        return 1
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
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
