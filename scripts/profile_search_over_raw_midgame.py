"""Create-only authored midgame component diagnostic; no search or strength claim.

Valid Showdown histories, rather than duplicated protocol lines, supply the
fixed history strata. Unchanged canonical policy callbacks are measured. The
native helper is NOT the live Rust bridge; Python cumulative costs overlap.
No old failed root, qualification, scientific panel or producer is replayed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import profile_search_over_raw_callback as opening
from benchmark_search_over_raw_feasibility import close_resources, save_new
from qualify_search_over_raw_opening import prepare_binding, verify_inputs
from pokezero.mcts_eval.resolver import sha256_file
from pokezero.mcts_eval.search_over_raw import ENGINEERING_EXCLUDED_SEEDS, digest, require

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pokezero.search-over-raw.authored-midgame-component-profile.v1"
NAMESPACE = "cb3064df-6034-49fd-aa91-41a87b23dc39"
FIXTURE_SEED = 2026101017
STRATA = (2, 8, 24)
SEATS = ("p1", "p2")
REPETITIONS = 25
FLAGS = ("retry_authorized", "phase_a_admission", "phase_b_authorized",
    "scientific_strength_evidence", "representative_runtime_evidence", "deployable")


def core():
    return dict(schema=SCHEMA, namespace=NAMESPACE, seeds=[FIXTURE_SEED],
        fixture_seed=FIXTURE_SEED, strata=list(STRATA), seats=list(SEATS),
        repetitions=REPETITIONS, configurations=[],
        script="both Surf on turn1, then Starmie/Milotic Recover through turn24",
        stop_policy="first fixture, callback, provenance or cleanup failure; no retry/redraw",
        **{key: False for key in FLAGS})


def fixture(source):
    # Reuse the established spread reconstruction, not the old active pairing.
    value = opening.fixture(source)
    team = value["player_teams"]["p1"].split("]")
    require(len(team) == 6, "authored packed roster differs")
    value["player_teams"]["p2"] = "]".join(team[4:] + team[:4])
    return value


def cells():
    return [dict(cell_id=f"turn{turn}-{seat}", completed_turns=turn, seat=seat,
        status="UNSTARTED") for turn in STRATA for seat in SEATS]


def exposure_inventory(paths):
    require(paths, "explicit exposure registrations required")
    seeds, bindings = set(ENGINEERING_EXCLUDED_SEEDS) - {FIXTURE_SEED}, {}
    for supplied in paths:
        path = Path(supplied).resolve()
        roster = json.loads(path.read_text()).get("seeds")
        require(isinstance(roster, list) and roster
            and all(type(s) is int and 0 <= s < 2**32 for s in roster),
            "exposure registration needs explicit uint32 seeds")
        seeds.update(roster)
        bindings[str(path)] = sha256_file(path)
    require(FIXTURE_SEED not in seeds, "authored seed overlaps supplied exposure; no retry/redraw")
    return sorted(seeds), bindings


def register(*, output, exposure_registrations, **kwargs):
    excluded, exposure = exposure_inventory(exposure_registrations)
    r = prepare_binding(**kwargs)
    from pokezero.randbat import load_gen3_randbat_source_cached
    authored = fixture(load_gen3_randbat_source_cached(r["showdown_root"]))
    output = Path(output).resolve()
    r.update(core(), fixture=authored, fixture_sha256=digest(authored),
        attempt_directory=str(output), exposure_registrations=exposure,
        excluded_seeds=excluded, excluded_seed_inventory_sha256=digest(excluded),
        exposure_scope="supplied registrations and minimum exclusions, not universal census",
        scope="six fixed authored midgame callback cells; engineering only")
    r["input_sha256"].update(exposure)
    for script in (__file__, opening.__file__):
        r["input_sha256"][str(Path(script).resolve())] = sha256_file(script)
    validate_contract(r, output)
    verify_inputs(r)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "registration.json", r)
    save_new(output / "registration-binding.json", dict(registration_sha256=digest(r)))
    return r


def validate_contract(r, output):
    require(digest({key: r[key] for key in core()}) == digest(core()),
        "midgame core drift or scientific admission/retry attempted")
    require(digest(r["fixture"]) == r["fixture_sha256"], "authored fixture drift")
    excluded, bindings = exposure_inventory(list(r["exposure_registrations"]))
    require(bindings == r["exposure_registrations"] and excluded == r["excluded_seeds"]
        and digest(excluded) == r["excluded_seed_inventory_sha256"]
        and all(r["input_sha256"].get(path) == sha for path, sha in bindings.items()),
        "exposure inventory drift")
    require(Path(r["source_root"]).resolve() == ROOT
        and r["attempt_directory"] == str(output.resolve())
        and not output.resolve().is_relative_to(ROOT), "source or attempt directory drift")
    require(all(r["input_sha256"].get(str(Path(script).resolve())) == sha256_file(script)
        for script in (__file__, opening.__file__)), "mandatory driver/helper hash drift")


def advance(env, completed, target, trace):
    require(type(completed) is int and type(target) is int and 0 <= completed < target <= max(STRATA),
        "invalid authored boundary progression")
    for turn in range(completed + 1, target + 1):
        require(env.terminal() is None and tuple(env.requested_players()) == SEATS,
            "authored fixture terminal or not simultaneous; no alternate script")
        actions = {"p1": 0, "p2": 0} if turn == 1 else {"p1": 3, "p2": 2}
        expected = "surf" if turn == 1 else "recover"
        for seat, action in actions.items():
            request = env.public_materialization_state(seat).self_request
            move = request["active"][0]["moves"][action]
            require(move["id"] == expected and move["pp"] > 0 and not move.get("disabled", False),
                "authored move unavailable; no fallback")
        env.step(actions)
        require(env.terminal() is None, "authored script terminated; no replacement")
        trace.append(dict(turn=turn, actions=actions, move=expected))


def certify_request(bundle, request):
    """Check current own PP, HP, active order and legal request, not full mask parity."""
    native = bundle["request"]
    def moves(value):
        return [(m["id"], m["pp"], m["maxpp"], bool(m.get("disabled", False)))
            for m in value["active"][0]["moves"]]
    require(moves(native) == moves(request), "native own move/PP differs from current Showdown request")
    def details(text):
        # Showdown omits the default L100; the native helper writes it. Do not
        # erase species, nondefault levels, gender or other semantic fields.
        parts = text.split(", ")
        return (parts[0], next((p for p in parts[1:] if p.startswith("L")), "L100"),
            tuple(p for p in parts[1:] if not p.startswith("L")))
    def party(value):
        return [(details(p["details"]), p["condition"], p["active"]) for p in value["side"]["pokemon"]]
    require(party(native) == party(request), "native own party/HP/order differs from current Showdown request")
    require(bundle["native_action_indices"] == list(range(9)), "authored native action roster differs")


def certify_saved_boundary(request, lines, seat, completed_turns):
    from pokezero.policy_opponent_view import public_policy_lines
    active = [p for p in request["side"]["pokemon"] if p["active"]]
    team = [s[0] for s in opening.SETS]
    team = team if seat == "p1" else team[4:] + team[:4]
    species = "Starmie" if seat == "p1" else "Milotic"
    require([p["details"].split(",")[0] for p in request["side"]["pokemon"]] == team
        and len(active) == 1 and active[0]["details"].split(",")[0] == species
        and active[0]["condition"] != "0 fnt" and request.get("wait") is not True
        and not any(request.get("forceSwitch", [])), "authored active identity/actionability differs")
    turns = [int(line.split("|")[2]) for line in lines if line.startswith("|turn|")]
    require(turns == list(range(1, completed_turns + 2)), "authored completed-turn boundary differs")
    require(tuple(lines) == public_policy_lines(lines, hp_visibility={"p1": "percentage", "p2": "percentage"})
        and "|start" in lines, "saved prefix is not canonical public history")
    moves = request["active"][0]["moves"]
    expected = opening.SETS[0 if seat == "p1" else 4][1]
    require(tuple(m["id"] for m in moves) == expected
        and all(m["pp"] > 0 and not m.get("disabled", False) for m in moves)
        and next(m for m in moves if m["id"] == "recover")["pp"] == 33 - completed_turns
        and next(m for m in moves if m["id"] == "recover")["maxpp"] == 32
        and moves[0]["pp"] == 23 and moves[0]["maxpp"] == 24,
        "saved authored legal moves/current Gen3 PP differ")
    return request


def certify_boundary(materialization, seat, completed_turns):
    from pokezero.policy_opponent_view import public_policy_lines
    lines = public_policy_lines(materialization.replay.public_lines,
        hp_visibility=materialization.replay.hp_visibility)
    return certify_saved_boundary(materialization.self_request, lines, seat, completed_turns)


def rebuild_bundle(r, inputs, seat):
    import pokezero_search
    from pokezero.dex import load_showdown_dex_cached
    dex = load_showdown_dex_cached(r["showdown_root"])
    names = inputs["native_species"]
    expected = [p["details"].split(",")[0] for p in inputs["own_request"]["side"]["pokemon"]]
    require(names == expected, "saved native own species/order differs")
    return json.loads(pokezero_search.sampled_policy_request(inputs["native_state"], seat, names, names,
        {name: info.max_pp for name, info in dex.moves.items()},
        base_pp={name: info.pp for name, info in dex.moves.items()}))


def profile_functions():
    from pokezero.belief import PublicBattleBeliefEngine
    from pokezero.showdown import normalize_for_player, parse_showdown_replay
    from pokezero.neural_policy import observation_window_to_torch, evaluate_transformer_action_priors
    from pokezero.policy_opponent import policy_opponent_distribution
    from pokezero.policy_opponent_view import PolicyOpponentView, build_policy_opponent_view_from_native_bundle
    return dict(public_replay_parse=parse_showdown_replay,
        public_belief_from_events=PublicBattleBeliefEngine.from_events.__func__,
        player_normalization=normalize_for_player,
        view_reconstruction=build_policy_opponent_view_from_native_bundle,
        canonical_observation=PolicyOpponentView.observation,
        tensor_encoding=observation_window_to_torch,
        action_prior_evaluation=evaluate_transformer_action_priors,
        policy_distribution_inclusive=policy_opponent_distribution)


def validate_profile(row):
    from pokezero.neural_policy import TransformerInferenceTiming
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    require(row["warmup_calls"] == 1 and type(row["warmup_calls"]) is int
        and row["measured_passes"] == 2 and type(row["measured_passes"]) is int
        and digest([row["callback_total_calls"], row["native_helper_profile_calls"]]) == digest([51, 26]),
        "fixed callback/helper call accounting differs")
    for name in ("warmup_seconds", "unprofiled_seconds", "profiled_seconds",
            "native_request_first_call_seconds", "native_request_seconds"):
        require(finite(row[name]), "missing/nonfinite/negative component timing")
    require(row["unprofiled_seconds"] > 0 and row["profiled_seconds"] > 0
        and math.isclose(row["combined_instrumentation_overhead_ratio"],
            row["profiled_seconds"] / row["unprofiled_seconds"], rel_tol=1e-12)
        and row["cumulative_metrics_overlap_do_not_sum"] is True,
        "timing ratio or overlap certificate differs")
    output = row["canonical_output"]
    require(len(output) == 9 and all(type(v) in (int, float) and v in (0, 1) for v in output)
        and sum(output) == 1, "raw callback output is not a nine-slot one-hot law")
    raw = row["profiler_rows"]
    require(isinstance(raw, list) and raw, "raw cProfile rows missing")
    rows = {}
    for entry in raw:
        key = (entry["file"], entry["line"], entry["function"])
        require(key not in rows and type(entry["calls"]) is int
            and type(entry["primitive_calls"]) is int and 0 <= entry["primitive_calls"] <= entry["calls"]
            and finite(entry["exclusive_seconds"]) and finite(entry["cumulative_seconds"])
            and entry["cumulative_seconds"] >= entry["exclusive_seconds"]
            and isinstance(entry["callers"], list), "invalid or duplicate raw profiler row")
        rows[key] = entry
    for name, function in profile_functions().items():
        raw_row = rows[opening.code_key(function)]
        metric = row["functions"][name]
        require(type(metric["calls"]) is int and metric["calls"] == REPETITIONS
            and all(metric[key] == raw_row[key] for key in
                ("calls", "exclusive_seconds", "cumulative_seconds")),
            "component summary does not match raw profile")
    timing = row["neural_timing_profiled"]
    require(set(timing) == set(TransformerInferenceTiming().to_dict())
        and all(finite(value) for key, value in timing.items() if key.endswith("seconds"))
        and all(type(value) is int and value >= 0 for key, value in timing.items() if key.endswith("count"))
        and all(timing[key] == REPETITIONS for key in
            ("neural_forward_count", "observation_encoding_count", "action_prior_neural_forward_count"))
        and all(timing[key] == 0 for key in ("policy_neural_forward_count",
            "value_neural_forward_count", "opponent_action_prior_neural_forward_count"))
        and all(timing[key] == 0 for key in ("policy_neural_forward_seconds",
            "value_neural_forward_seconds", "opponent_action_prior_neural_forward_seconds"))
        and timing["neural_forward_seconds"] == timing["action_prior_neural_forward_seconds"],
        "durable neural timing/counts differ")


def profile(callback, timed, payload, timing):
    measured = opening.component_profile(callback, timed, payload, REPETITIONS, timing)
    rows = {(r["file"], r["line"], r["function"]): r for r in measured["profiler_rows"]}
    for name, function in profile_functions().items():
        row = rows.get(opening.code_key(function))
        require(row is not None and row["calls"] == REPETITIONS,
            "profile omitted canonical replay/belief/normalization work")
        measured["functions"][name] = {key: row[key]
            for key in ("calls", "exclusive_seconds", "cumulative_seconds")}
    measured["limitations"][0] = "authored Surf/Recover histories only; history length covaries with PP/state"
    measured["limitations"].extend([
        "warmup_seconds is first call of this callback, not cold process/model/cache latency",
        "live Rust prefix/PP/order assembly is not isolated by this native helper",
        "fixed same-state repeated calls do not establish searched-branch throughput or feasibility"])
    require(all(measured["functions"][name]["calls"] == REPETITIONS for name in
        ("action_prior_evaluation", "tensor_encoding"))
        and measured["neural_timing_profiled"]["neural_forward_count"] == REPETITIONS
        and measured["neural_timing_profiled"]["observation_encoding_count"] == REPETITIONS,
        "profile did not exercise pinned-champion forward/encoding")
    return measured


def record_profile(output, progress, cell, row, inputs):
    save_new(output / (cell["cell_id"] + "-inputs.json"), inputs)
    row.update(cell_id=cell["cell_id"], seat=cell["seat"], completed_turns=cell["completed_turns"],
        inputs_sha256=digest(inputs))
    save_new(output / (cell["cell_id"] + "-profile.json"), row)
    cell["status"] = "COMPLETE_COMPONENT_ONLY"
    progress["profiles_completed"] += 1


def verify_completion(output, progress, r):
    require(progress["profiles_completed"] == len(cells())
        and [{k: c[k] for k in ("cell_id", "completed_turns", "seat")} for c in progress["roster"]]
            == [{k: c[k] for k in ("cell_id", "completed_turns", "seat")} for c in cells()],
        "full fixed profile roster differs")
    for cell in progress["roster"]:
        inputs = json.loads((output / (cell["cell_id"] + "-inputs.json")).read_text())
        row = json.loads((output / (cell["cell_id"] + "-profile.json")).read_text())
        claim = json.loads((output / (cell["cell_id"] + "-attempt.json")).read_text())
        require(cell["status"] == "COMPLETE_COMPONENT_ONLY"
            and claim == dict(cell_id=cell["cell_id"], registration_sha256=digest(r))
            and all(row[key] == cell[key] for key in ("cell_id", "seat", "completed_turns"))
            and row["inputs_sha256"] == digest(inputs)
            and row["payload_sha256"] == digest(inputs["payload"])
            and row["public_prefix_sha256"] == digest(inputs["public_lines"])
            and row["native_state_sha256"] == digest(inputs["native_state"])
            and inputs["payload"]["opponent_slot"] == cell["seat"]
            and inputs["payload"]["public_branch_lines"] == []
            and len(inputs["script_trace"]) == cell["completed_turns"]
            and inputs["script_trace"] == [dict(turn=t, actions={"p1": 0, "p2": 0}
                if t == 1 else {"p1": 3, "p2": 2}, move="surf" if t == 1 else "recover")
                for t in range(1, cell["completed_turns"] + 1)]
            and type(row["repetitions_per_pass"]) is int and row["repetitions_per_pass"] == REPETITIONS
            and type(row["native_request_calls"]) is int and row["native_request_calls"] == REPETITIONS,
            "durable profile binding or count differs")
        certify_request(inputs["payload"]["native_request_bundle"], inputs["own_request"])
        certify_saved_boundary(inputs["own_request"], inputs["public_lines"], cell["seat"], cell["completed_turns"])
        require(digest(rebuild_bundle(r, inputs, cell["seat"])) == digest(inputs["payload"]["native_request_bundle"]),
            "saved native bundle differs from retained native state")
        validate_profile(row)
        require(all(row["functions"][name]["calls"] == REPETITIONS for name in (
            "view_reconstruction", "canonical_observation", "public_replay_parse",
            "public_belief_from_events", "player_normalization", "tensor_encoding", "action_prior_evaluation"))
            and row["neural_timing_profiled"]["neural_forward_count"] == REPETITIONS
            and row["neural_timing_profiled"]["observation_encoding_count"] == REPETITIONS,
            "durable canonical callback work differs")


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
    require(fixture(source) == r["fixture"], "authored fixture differs")
    policy = load_transformer_policy(Path(r["checkpoint"]), device="cpu", deterministic=True,
        exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
    require(policy.result.belief_set_source_hash == r["set_source_hash"], "policy source drift")
    dex = load_showdown_dex_cached(r["showdown_root"])
    tables = json.loads(Path(r["encoder_tables"]).read_text())["vocab"]
    tokens = tuple(policy.result.model_config.category_vocab)
    require(tuple(tables["tokens"]) == tokens and tables["oov_buckets"] == policy.result.model_config.category_oov_buckets
        and all(tables["index"].get(token.strip().lower()) == i for i, token in enumerate(tokens, 1))
        and all(type(i) is int and 1 <= i <= len(tokens) for i in tables["index"].values()),
        "native vocabulary differs from checkpoint")
    vocab = CategoryVocabulary(tokens=tokens, aliases={a: tokens[i-1] for a, i in tables["index"].items()},
        oov_buckets=tables["oov_buckets"])
    config = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(r["showdown_root"]),
        set_belief_source=True), [f"neural:{r['checkpoint']}"], context="excluded authored midgame profile")
    env = LocalShowdownEnv(config)
    try:
        override = BattleStartOverride(player_teams=r["fixture"]["player_teams"], observation_format_id="gen3randombattle")
        env.reset_with_start_override(seed=FIXTURE_SEED, start_override=override)
        maximum = {name: info.max_pp for name, info in dex.moves.items()}
        base = {name: info.pp for name, info in dex.moves.items()}
        completed, trace = 0, []
        for target in STRATA:
            progress["stage"] = f"authored_advance:{target}"
            advance(env, completed, target, trace)
            completed = target
            for cell in [c for c in progress["roster"] if c["completed_turns"] == target]:
                progress["stage"] = cell["cell_id"]
                save_new(output / (cell["cell_id"] + "-attempt.json"),
                    dict(cell_id=cell["cell_id"], registration_sha256=digest(r)))
                cell["status"] = "ATTEMPTED_UNCERTAIN"
                slot = cell["seat"]
                materialization = env.public_materialization_state(slot)
                certify_boundary(materialization, slot, target)
                world, native = build_engine_world(materialization, override, dex=dex, module=poke_engine)
                state = native.to_string()
                names = [dex.species_info(name).name for name in world.party_species[slot]]
                def request(_):
                    return pokezero_search.sampled_policy_request(state, slot, names, names, maximum, base_pp=base)
                bundle_json, first_request = opening.repeated(request, None, 1)
                _, request_seconds = opening.repeated(request, None, REPETITIONS, bundle_json)
                bundle = json.loads(bundle_json)
                certify_request(bundle, materialization.self_request)
                lines = public_policy_lines(materialization.replay.public_lines,
                    hp_visibility=materialization.replay.hp_visibility)
                payload = dict(native_request_bundle=bundle, public_branch_lines=[], opponent_slot=slot)
                args = dict(public_lines=lines, hp_visibility={"p1": "percentage", "p2": "percentage"},
                    opponent_slot=slot, battle_id="authored-midgame-component-profile", battle_seed=FIXTURE_SEED,
                    format_id="gen3randombattle", set_source=source, model=policy.model, result=policy.result,
                    category_vocab=vocab, dex=dex, device="cpu", raw_argmax=True)
                callback = make_policy_opponent_callback(**args)
                timing = TransformerInferenceTimingAccumulator()
                timed = make_policy_opponent_callback(**args, timing=timing)
                measured = profile(callback, timed, json.dumps(payload, sort_keys=True), timing)
                measured.update(native_request_calls=REPETITIONS, native_request_first_call_seconds=first_request,
                    callback_total_calls=51, native_helper_profile_calls=26,
                    native_request_seconds=request_seconds, native_state_sha256=digest(state),
                    public_prefix_sha256=digest(lines), public_prefix_lines=len(lines))
                verify_inputs(r)
                record_profile(output, progress, cell, measured, dict(payload=payload,
                    public_lines=lines, own_request=materialization.self_request, native_state=state,
                    native_species=names,
                    script_trace=list(trace)))
                print(json.dumps(dict(cell_id=cell["cell_id"], status=cell["status"])), flush=True)
    finally:
        close_resources([("authored_env", env)], progress)


def run(output):
    output = Path(output).resolve()
    r = json.loads((output / "registration.json").read_text())
    validate_contract(r, output)
    require(json.loads((output / "registration-binding.json").read_text()) == dict(registration_sha256=digest(r)),
        "whole-registration digest drift")
    require(not any((output / name).exists() for name in ("attempt.json", "terminal.json")),
        "attempt or terminal exists; no repeat")
    require(not any((output / (cell["cell_id"] + suffix)).exists() for cell in cells()
        for suffix in ("-attempt.json", "-inputs.json", "-profile.json")),
        "partial cell evidence exists; no repeat")
    verify_inputs(r)
    save_new(output / "attempt.json", dict(pid=os.getpid(), registration_sha256=digest(r), retry_authorized=False))
    progress = dict(stage="runtime_import", roster=cells(), profiles_completed=0)
    started = time.perf_counter()
    terminal = dict(registration_sha256=digest(r), searched_decisions=0, terminal_games=0,
        **{key: False for key in FLAGS})
    try:
        measure(r, output, progress)
        verify_inputs(r)
        verify_completion(output, progress, r)
        terminal.update(status="COMPLETE_AUTHORED_MIDGAME_COMPONENT_ONLY", exit_code=0,
            native_helper_verification_calls=len(cells()))
    except BaseException as error:
        terminal.update(status="FAILED_NO_RETRY", exit_code=1, error_type=type(error).__name__, error=str(error),
            failure_frames=[dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_tb(error.__traceback__)])
    terminal.update(progress, elapsed_seconds=time.perf_counter() - started)
    save_new(output / "terminal.json", terminal)
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return terminal["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run").add_argument("--output", required=True, type=Path)
    prep = sub.add_parser("register")
    for name in ("output", "checkpoint", "showdown-root", "native-receipt", "factory-options", "encoder-tables"):
        prep.add_argument("--" + name, required=True, type=Path)
    for name in ("checkpoint-sha256", "showdown-commit", "source-commit", "factory-options-sha256"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--exposure-registration", required=True, action="append", type=Path,
        dest="exposure_registrations")
    args = vars(parser.parse_args(argv))
    if args.pop("command") == "run":
        return run(**args)
    r = register(**args)
    print(json.dumps(dict(status=r["status"], registration_sha256=digest(r))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
