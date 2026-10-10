"""One create-only champion timing attempt on an authored six-versus-six root.

This is a NEW synthetic team-oracle fixture, not a replay of a failed source
qualification. Full historical incumbent allocation and raw-terminal semantics
are retained. It cannot establish playing strength, representative resource
cost, deployability or Phase A admission. No source games or continuations run.
"""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import uuid

from qualify_search_over_raw_opening import prepare_binding, save_new, verify_inputs
from pokezero.mcts_eval.search_over_raw import SearchConfiguration, digest, require


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SEED = 2026101013
NAMESPACE = "7636a4d9-8214-4a31-b806-4338170ddde8"
SCHEMA = "pokezero.search-over-raw.synthetic-champion-timing.v1"
CONFIGURATION = SearchConfiguration("incumbent", "oracle", "raw_rollout", 10., 1)
ALLOCATION = dict(depth=6, sims=4096, batch=16, worlds=4, native_batch_guard_ms=64)
NATIVE_CONFIGURATION = dict(worlds=4, threads=1, leaf_eval="model", search_sims=4096,
    search_batch=16, search_depth=6, model_decision_time_ms=10000,
    model_native_batch_guard_ms=64, model_world_workers=1, model_priors=True,
    use_opponent_priors=False, model_leaf_override="raw_policy_terminal", early_stop=False,
    rollout_count=1, rollout_max_plies=250, rollout_policy="raw_argmax",
    rollout_threads=1, rollout_branch_on_damage=True, strict_fallbacks=True)
# Authored independently of search outcomes. Spread uses the pinned request
# reconstruction recipe; these are NOT draws from the randbat team generator.
SETS = (
    ("Alakazam", ("psychic", "calmmind", "recover", "firepunch"), "synchronize", "F"),
    ("Gengar", ("thunderbolt", "icepunch", "gigadrain", "explosion"), "levitate", "F"),
    ("Snorlax", ("bodyslam", "earthquake", "rest", "sleeptalk"), "thickfat", "M"),
    ("Skarmory", ("drillpeck", "spikes", "roar", "rest"), "keeneye", "M"),
    ("Swampert", ("surf", "earthquake", "icebeam", "protect"), "torrent", "M"),
    ("Blissey", ("seismictoss", "softboiled", "toxic", "icebeam"), "naturalcure", "F"),
)


def fixture(source):
    from pokezero.determinization import _gen3_randbat_fixture_spread
    from pokezero.showdown_fixture import FixturePokemon, pack_team
    team = []
    for species, moves, ability, gender in SETS:
        spread = _gen3_randbat_fixture_spread({}, species=species, moves=moves,
            item="leftovers", level=100, set_source=source)
        require(spread is not None, "synthetic fixture spread cannot be reconstructed")
        team.append(FixturePokemon(species, moves, ability, "leftovers", gender=gender,
            evs=spread["evs"], ivs=spread["ivs"]))
    # Different authored leads without adding any observed outcome-based choice.
    return dict(format_id="gen3customgame", observation_format_id="gen3randombattle",
        player_teams={"p1": pack_team(team), "p2": pack_team((*team[4:], *team[:4]))},
        authored_not_random_team=True, expected_living_members_per_seat=6,
        expected_moves_per_member=4, initial_hp="full", source_hash=source.metadata.source_hash)


def register(*, output, namespace, **kwargs):
    require(str(uuid.UUID(namespace)) == namespace == NAMESPACE, "fresh synthetic namespace required")
    binding = prepare_binding(**kwargs, seconds=10.)  # Input binding ONLY; never old run/register.
    from pokezero.randbat import load_gen3_randbat_source_cached
    authored = fixture(load_gen3_randbat_source_cached(binding["showdown_root"]))
    binding.update(schema=SCHEMA, namespace=namespace, fixture=authored,
        fixture_sha256=digest(authored), configurations=[asdict(CONFIGURATION)],
        allocation=ALLOCATION.copy(), native_configuration=NATIVE_CONFIGURATION.copy(),
        seeds=[FIXTURE_SEED], fixture_seed=FIXTURE_SEED,
        representative_runtime_evidence=False, deployable=False,
        scope="one authored six-v-six oracle root; champion timing only; no source games or continuations")
    binding["input_sha256"][str(Path(__file__).resolve())] = digest_file(Path(__file__))
    output = Path(output).resolve()
    binding["attempt_directory"] = str(output)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", binding)
    return binding


def digest_file(path):
    from pokezero.mcts_eval.resolver import sha256_file
    return sha256_file(path)


def validate_contract(registration):
    require(registration["schema"] == SCHEMA
        and str(uuid.UUID(registration["namespace"])) == registration["namespace"] == NAMESPACE
        and registration["configurations"] == [asdict(CONFIGURATION)]
        and registration["allocation"] == ALLOCATION
        and registration["native_configuration"] == NATIVE_CONFIGURATION
        and registration["seeds"] == [FIXTURE_SEED]
        and registration["fixture_seed"] == FIXTURE_SEED
        and registration["candidate_seat"] == "p1"
        and all(registration[key] is False for key in ("retry_authorized",
            "phase_a_admission", "phase_b_authorized", "scientific_strength_evidence",
            "representative_runtime_evidence", "deployable"))
        and digest(registration["fixture"]) == registration["fixture_sha256"],
        "synthetic contract changed, reduced allocation or scientific admission")
    script = Path(__file__).resolve()
    require(Path(registration["source_root"]).resolve() == ROOT
        and registration["input_sha256"].get(str(script)) == digest_file(script),
        "executing source or mandatory driver hash differs")


def measure(registration, output, progress):
    """Actual adapter path; warm-up separated, nothing overrides its search settings."""
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
            "native package import resolves outside isolated build")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.env import BattleStartOverride
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.mcts_eval.search_over_raw_oracle import (
        DiagnosticTeamOracleSearchAdapter, TeamOracle, opening_team)
    from pokezero.policy import PolicyContext
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.showdown_fixture import pack_team
    from pokezero.trajectory import BattleTrajectory
    progress["stage"] = "fixture_capture"
    checkpoint, showdown = registration["checkpoint"], registration["showdown_root"]
    source = load_gen3_randbat_source_cached(showdown)
    require(fixture(source) == registration["fixture"], "authored fixture or catalog drift")
    authored = registration["fixture"]
    env = adapter = None
    try:
        config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(showdown),
            set_belief_source=True), [f"neural:{checkpoint}"], context="synthetic champion timing")
        env = LocalShowdownEnv(config)
        env.reset_with_start_override(seed=FIXTURE_SEED, start_override=BattleStartOverride(
            player_teams=authored["player_teams"], observation_format_id="gen3randombattle"))
        snapshot = env.snapshot_actionable_boundary()
        for seat in ("p1", "p2"):
            rows = snapshot.first_requests[seat]["side"]["pokemon"]
            require(len(rows) == 6 and all("/" in row["condition"]
                and row["condition"].split()[0].split("/")[0] == row["condition"].split()[0].split("/")[1]
                and len(row["moves"]) == 4 for row in rows), "fixture is not full six-v-six")
            require(pack_team(opening_team(snapshot.first_requests[seat], seat, source))
                == authored["player_teams"][seat], "oracle reconstruction changes authored team")
        own = env.observe("p1")
        public = env.public_materialization_state("p1")
        request = PublicRootRequest.capture(public, own)
        oracle = TeamOracle.capture(snapshot, request, source)
        contract = resolve_checkpoint_contract(checkpoint,
            expected_sha256=registration["checkpoint_sha256"], showdown_root=showdown,
            showdown_source_sha256=source.metadata.source_hash,
            model_path=registration["model_path"], tables_path=registration["encoder_tables"])
        require(contract.to_manifest() == registration["checkpoint_contract"], "checkpoint contract drift")
        context = PolicyContext("p1", 0, "synthetic-champion-timing", "gen3randombattle", FIXTURE_SEED,
            own, tuple(env.requested_players()), BattleTrajectory("synthetic-champion-timing", "gen3randombattle", FIXTURE_SEED),
            requested_legal_action_masks={"p1": own.legal_action_mask},
            requested_observations={"p1": own}, public_materialization_state=public)
        save_new(output / "root-binding.json", dict(fixture_sha256=registration["fixture_sha256"],
            team_oracle=oracle.receipt(), legal_actions=[i for i, x in enumerate(own.legal_action_mask) if x],
            living_members_per_seat=6, scientific_strength_evidence=False))
        progress["stage"] = "adapter_construct_and_warm"
        constructed = time.perf_counter()
        adapter = DiagnosticTeamOracleSearchAdapter(CONFIGURATION, oracle=oracle,
            checkpoint_contract=contract, showdown_root=showdown)
        construction_seconds = time.perf_counter() - constructed
        require(adapter.runtime_configuration["incumbent"] == ALLOCATION, "runtime allocation drift")
        actual_native = {key: getattr(adapter._native._config, key) for key in NATIVE_CONFIGURATION}
        require(actual_native == NATIVE_CONFIGURATION, "native allocation or semantics drift")
        save_new(output / "runtime-binding.json", dict(configuration=adapter.runtime_configuration,
            native_configuration=actual_native, construction_seconds=construction_seconds,
            timing_excludes_warmup=True))
        verify_inputs(registration)
        progress["stage"] = "select_once"
        selection_started = time.perf_counter()
        try:
            row = adapter.select(context, root_id="synthetic:" + registration["namespace"],
                selection_seed=FIXTURE_SEED)
        except BaseException:
            progress["adapter_failure"] = adapter.last_failure
            progress["selection_elapsed_seconds"] = time.perf_counter() - selection_started
            raise
        row.update(construction_seconds=construction_seconds,
            registration_sha256=digest(registration), representative_runtime_evidence=False,
            phase_a_admission=False, deployable=False)
        verify_inputs(registration)
        save_new(output / "selected.json", row)
        progress["stage"] = "close"
        return row
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            if env is not None:
                env.close()


def run(output):
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    validate_contract(registration)
    require(registration["attempt_directory"] == str(output), "attempt directory changed; no copied retry")
    verify_inputs(registration)
    save_new(output / "attempt.json", dict(status="CLAIMED_BEFORE_RUNTIME", pid=os.getpid(),
        registration_sha256=digest(registration), retry_authorized=False))
    started = time.perf_counter()
    progress = dict(stage="runtime_import")
    result = dict(registration_sha256=digest(registration), retry_authorized=False,
        scientific_strength_evidence=False, phase_a_admission=False,
        phase_b_authorized=False, representative_runtime_evidence=False, deployable=False,
        terminal_games=0)
    try:
        row = measure(registration, output, progress)
        result.update(status="COMPLETE_SYNTHETIC_TIMING_ONLY", selection_elapsed_seconds=row["elapsed_seconds"],
            exceeded_nominal_seconds=row["exceeded_nominal_seconds"])
        code = 0
    except BaseException as error:
        from pokezero.mcts_eval.policy_opponent_profile import fallback_refusal_diagnostic, refusal_diagnostic
        result.update(status="FAILED_NO_RETRY", error_type=type(error).__name__, **progress,
            native_diagnostic=refusal_diagnostic(error), engine_fallback_diagnostic=fallback_refusal_diagnostic(error),
            failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)])
        code = 1
    result.update(elapsed_seconds=time.perf_counter() - started, exit_code=code)
    save_new(output / "terminal.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("register")
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256", "namespace"):
        prep.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "run":
        return run(**args)
    result = register(**args)
    print(json.dumps(dict(status=result["status"], registration_sha256=digest(result))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
