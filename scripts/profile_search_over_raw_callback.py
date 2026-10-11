"""Create-only component profile of unchanged canonical raw-policy callbacks.

A separately authored opening and fixed repetition count, not a search replay,
qualification retry, continuation, admission or representative runtime claim.
cProfile perturbs timings: profiled and unprofiled wall costs are both retained.
No canonical callback/model function is patched, cached or replaced.
"""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
import time
import traceback

from qualify_search_over_raw_opening import prepare_binding, save_new, verify_inputs
from pokezero.mcts_eval.search_over_raw import digest, require
from pokezero.mcts_eval.resolver import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pokezero.search-over-raw.callback-component-profile.v1"
NAMESPACE = "866c49c6-5694-47e7-b93d-7e9cb7ea32aa"
FIXTURE_SEED = 2026101014
REPETITIONS = 25
SETS = (
    ("Starmie", ("surf", "psychic", "thunderbolt", "recover"), "naturalcure", ""),
    ("Jolteon", ("thunderbolt", "thunderwave", "substitute", "batonpass"), "voltabsorb", "F"),
    ("Metagross", ("meteormash", "earthquake", "rockslide", "explosion"), "clearbody", ""),
    ("Flygon", ("earthquake", "rockslide", "fireblast", "dragonclaw"), "levitate", "M"),
    ("Milotic", ("surf", "icebeam", "recover", "toxic"), "marvelscale", "F"),
    ("Heracross", ("megahorn", "brickbreak", "rockslide", "swordsdance"), "guts", "M"),
)


def fixture(source):
    from pokezero.determinization import _gen3_randbat_fixture_spread
    from pokezero.showdown_fixture import FixturePokemon, pack_team
    team = []
    for species, moves, ability, gender in SETS:
        spread = _gen3_randbat_fixture_spread({}, species=species, moves=moves,
            item="leftovers", level=100, set_source=source)
        require(spread is not None, "authored spread cannot be reconstructed")
        team.append(FixturePokemon(species, moves, ability, "leftovers", gender=gender,
            evs=spread["evs"], ivs=spread["ivs"]))
    return dict(player_teams={"p1": pack_team(team), "p2": pack_team((*team[3:], *team[:3]))},
        observation_format_id="gen3randombattle", authored_not_random_team=True,
        source_hash=source.metadata.source_hash)


def register(*, output, **kwargs):
    binding = prepare_binding(**kwargs)  # Immutable input binding only; no old attempt.
    from pokezero.randbat import load_gen3_randbat_source_cached
    authored = fixture(load_gen3_randbat_source_cached(binding["showdown_root"]))
    output = Path(output).resolve()
    binding.update(schema=SCHEMA, namespace=NAMESPACE, fixture_seed=FIXTURE_SEED,
        seeds=[FIXTURE_SEED], configurations=[], fixture=authored,
        fixture_sha256=digest(authored), repetitions=REPETITIONS, seats=["p1", "p2"],
        attempt_directory=str(output), representative_runtime_evidence=False,
        deployable=False, scope="isolated serial component microprofile; no search or game steps")
    binding["input_sha256"][str(Path(__file__).resolve())] = sha256_file(__file__)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", binding)
    return binding


def validate_contract(r, output):
    require(r["schema"] == SCHEMA and r["namespace"] == NAMESPACE
        and r["fixture_seed"] == FIXTURE_SEED and r["seeds"] == [FIXTURE_SEED]
        and r["repetitions"] == REPETITIONS and r["seats"] == ["p1", "p2"]
        and r["configurations"] == [] and r["attempt_directory"] == str(output.resolve())
        and all(r[key] is False for key in ("retry_authorized", "phase_a_admission",
            "phase_b_authorized", "scientific_strength_evidence", "representative_runtime_evidence", "deployable"))
        and digest(r["fixture"]) == r["fixture_sha256"], "component profile contract drift")
    require(Path(r["source_root"]).resolve() == ROOT
        and r["input_sha256"].get(str(Path(__file__).resolve())) == sha256_file(__file__),
        "executing source or mandatory driver hash drift")


def code_key(function):
    code = function.__code__
    return code.co_filename, code.co_firstlineno, code.co_name


def function_cost(stats, function, *, caller=None):
    """Cumulative costs overlap; never present their sum as exclusive time."""
    row = stats.get(code_key(function))
    if row is None:
        return dict(calls=0, exclusive_seconds=0., cumulative_seconds=0.)
    if caller is not None:
        row = row[4].get(code_key(caller))
        if row is None:
            return dict(calls=0, exclusive_seconds=0., cumulative_seconds=0.)
    return dict(calls=row[1], exclusive_seconds=row[2], cumulative_seconds=row[3])


def repeated(callback, payload, count, expected=None):
    started = time.perf_counter()
    for _ in range(count):
        actual = callback(payload)
        if expected is None:
            expected = actual
        require(actual == expected, "canonical callback output changed across identical calls")
    return expected, time.perf_counter() - started


def component_profile(callback, timed_callback, payload, count, timing):
    from pokezero.neural_policy import observation_window_to_torch, evaluate_transformer_action_priors
    from pokezero.policy_opponent import policy_opponent_distribution
    from pokezero.policy_opponent_view import PolicyOpponentView, build_policy_opponent_view_from_native_bundle
    expected, warm_seconds = repeated(callback, payload, 1)
    _, plain_seconds = repeated(callback, payload, count, expected)
    before = digest(json.loads(payload))
    profiler = cProfile.Profile()
    _, profiled_seconds = profiler.runcall(repeated, timed_callback, payload, count, expected)
    require(digest(json.loads(payload)) == before, "profile input mutated")
    stats = pstats.Stats(profiler).stats
    metrics = {
        "payload_json_loads_direct": function_cost(stats, json.loads, caller=timed_callback),
        "view_reconstruction": function_cost(stats, build_policy_opponent_view_from_native_bundle),
        "canonical_observation": function_cost(stats, PolicyOpponentView.observation),
        "tensor_encoding": function_cost(stats, observation_window_to_torch),
        "action_prior_evaluation": function_cost(stats, evaluate_transformer_action_priors),
        "policy_distribution_inclusive": function_cost(stats, policy_opponent_distribution),
    }
    require(metrics["view_reconstruction"]["calls"] == count
        and metrics["canonical_observation"]["calls"] == count,
        "profile did not exercise complete canonical callback")
    return dict(warmup_calls=1, repetitions_per_pass=count, measured_passes=2,
        canonical_output=list(expected), warmup_seconds=warm_seconds,
        unprofiled_seconds=plain_seconds, profiled_seconds=profiled_seconds,
        combined_instrumentation_overhead_ratio=profiled_seconds / plain_seconds if plain_seconds else None,
        cumulative_metrics_overlap_do_not_sum=True, functions=metrics,
        neural_timing_profiled=timing.snapshot().to_dict(), payload_sha256=before,
        profiler_rows=[dict(file=key[0], line=key[1], function=key[2], primitive_calls=row[0],
            calls=row[1], exclusive_seconds=row[2], cumulative_seconds=row[3],
            callers=[dict(file=caller[0], line=caller[1], function=caller[2], values=list(cost))
                for caller, cost in sorted(row[4].items())]) for key, row in sorted(stats.items())],
        limitations=["one authored opening; not representative searched branches",
            "serialized CPU callbacks; not contention or end-to-end search",
            "native helper includes state deserialization and request serialization; not the live search bridge decomposition",
            "neural_forward timer includes masking and output extraction, not pure network latency",
            "cProfile and timing accumulator overhead are included in profiled costs; ratio also includes pass-to-pass variation"])


def measure(r, output, progress):
    import torch
    import poke_engine
    import pokezero_search
    require(sys.executable == r["python_executable"] and torch.__version__ == r["torch_version"]
        and torch.__file__ == r["torch_path"], "Python/PyTorch binding drift")
    for module, name in ((poke_engine, "poke_engine"), (pokezero_search, "pokezero_search")):
        require(Path(module.__file__).resolve() == Path(r["packages_root"]) / name / "__init__.py",
            "native package binding drift")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_world import build_engine_world
    from pokezero.env import BattleStartOverride
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.neural_policy import load_transformer_policy, TransformerInferenceTimingAccumulator
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.policy_opponent import make_policy_opponent_callback
    from pokezero.policy_opponent_view import public_policy_lines
    from pokezero.randbat import load_gen3_randbat_source_cached
    source = load_gen3_randbat_source_cached(r["showdown_root"])
    require(fixture(source) == r["fixture"], "authored fixture drift")
    policy = load_transformer_policy(Path(r["checkpoint"]), device="cpu", deterministic=True,
        exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
    require(policy.result.belief_set_source_hash == r["set_source_hash"], "policy source drift")
    dex = load_showdown_dex_cached(r["showdown_root"])
    tables = json.loads(Path(r["encoder_tables"]).read_text())["vocab"]
    tokens = tuple(policy.result.model_config.category_vocab)
    require(tuple(tables["tokens"]) == tokens and tables["oov_buckets"] == policy.result.model_config.category_oov_buckets
        and all(tables["index"].get(token.strip().lower()) == index for index, token in enumerate(tokens, 1)),
        "native vocabulary differs from checkpoint")
    require(all(type(index) is int and 1 <= index <= len(tokens) for index in tables["index"].values()),
        "native vocabulary alias range drift")
    vocab = CategoryVocabulary(tokens=tokens, aliases={alias: tokens[index-1] for alias, index in tables["index"].items()},
        oov_buckets=tables["oov_buckets"])
    config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(r["showdown_root"]),
        set_belief_source=True), [f"neural:{r['checkpoint']}"], context="excluded component microprofile")
    env = LocalShowdownEnv(config)
    try:
        override = BattleStartOverride(player_teams=r["fixture"]["player_teams"], observation_format_id="gen3randombattle")
        env.reset_with_start_override(seed=FIXTURE_SEED, start_override=override)
        maximum = {name: info.max_pp for name, info in dex.moves.items()}
        base = {name: info.pp for name, info in dex.moves.items()}
        results = []
        for slot in r["seats"]:
            progress["stage"] = slot + ":component_profile"
            materialization = env.public_materialization_state(slot)
            world, native = build_engine_world(materialization, override, dex=dex, module=poke_engine)
            native_state = native.to_string()
            names = [dex.species_info(name).name for name in world.party_species[slot]]
            def request(_):
                return pokezero_search.sampled_policy_request(native_state, slot, names, names, maximum, base_pp=base)
            bundle_json, request_seconds = repeated(request, None, REPETITIONS)
            payload = json.dumps(dict(native_request_bundle=json.loads(bundle_json),
                public_branch_lines=[], opponent_slot=slot), sort_keys=True)
            replay = materialization.replay
            args = dict(public_lines=replay.public_lines, hp_visibility=replay.hp_visibility,
                opponent_slot=slot, battle_id="authored-component-profile", battle_seed=FIXTURE_SEED,
                format_id="gen3randombattle", set_source=source, model=policy.model, result=policy.result,
                category_vocab=vocab, dex=dex, device="cpu", raw_argmax=True)
            callback = make_policy_opponent_callback(**args)
            timing = TransformerInferenceTimingAccumulator()
            timed_callback = make_policy_opponent_callback(**args, timing=timing)
            measured = component_profile(callback, timed_callback, payload, REPETITIONS, timing)
            require(measured["functions"]["action_prior_evaluation"]["calls"] == REPETITIONS
                and measured["functions"]["tensor_encoding"]["calls"] == REPETITIONS
                and measured["neural_timing_profiled"]["neural_forward_count"] == REPETITIONS
                and measured["neural_timing_profiled"]["observation_encoding_count"] == REPETITIONS,
                "microprofile did not exercise every pinned-champion forward")
            measured.update(seat=slot, native_request_calls=REPETITIONS,
                native_request_seconds=request_seconds, native_state_sha256=digest(native_state),
                public_prefix_sha256=digest(public_policy_lines(replay.public_lines, hp_visibility=replay.hp_visibility)),
                public_prefix_lines=len(replay.public_lines), native_action_indices=json.loads(bundle_json)["native_action_indices"])
            verify_inputs(r)
            save_new(output / (slot + "-profile.json"), measured)
            results.append(measured)
        return results
    finally:
        env.close()


def run(output):
    output = Path(output).resolve()
    r = json.loads((output / "registration.json").read_text())
    validate_contract(r, output)
    verify_inputs(r)
    save_new(output / "attempt.json", dict(pid=os.getpid(), status="CLAIMED_BEFORE_RUNTIME",
        registration_sha256=digest(r), retry_authorized=False))
    started, progress = time.perf_counter(), dict(stage="runtime_import")
    terminal = dict(registration_sha256=digest(r), terminal_games=0, searched_decisions=0,
        scientific_strength_evidence=False, phase_a_admission=False, phase_b_authorized=False,
        representative_runtime_evidence=False, retry_authorized=False, deployable=False)
    try:
        profiles = measure(r, output, progress)
        verify_inputs(r)
        terminal.update(status="COMPLETE_COMPONENT_PROFILE_ONLY", seats=[row["seat"] for row in profiles])
        code = 0
    except BaseException as error:
        terminal.update(status="FAILED_NO_RETRY", **progress, error_type=type(error).__name__,
            error=str(error), failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)])
        code = 1
    terminal.update(exit_code=code, elapsed_seconds=time.perf_counter() - started)
    save_new(output / "terminal.json", terminal)
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run").add_argument("--output", required=True, type=Path)
    prep = sub.add_parser("register")
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables"):
        prep.add_argument("--" + name, required=True, type=Path)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256"):
        prep.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    if args.pop("command") == "run":
        return run(**args)
    registered = register(**args)
    print(json.dumps(dict(status=registered["status"], registration_sha256=digest(registered))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
