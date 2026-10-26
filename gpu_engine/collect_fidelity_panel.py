"""Fixed real-team JAX-sourced exact-outcome panel; not simulator truth.

All roots and actions are generated, saved and hashed before the first oracle
comparison. The external owned supervisor must enforce the entry's deadline
during operations and preserve an interrupted slot as UNKNOWN. No resume/redraw.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def encoded(value):
    def exact(item):
        if isinstance(item, Fraction):
            return str(item)
        raise TypeError(f"unserialized panel value: {type(item).__name__}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=exact).encode()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def request_labels(raw, actions, requested, usable, vocabulary):
    """Freeze actual requests, without attributing ignored zeros or callees."""
    rows = []
    for side, action in enumerate(actions):
        if not requested[side]:
            rows.append(dict(kind="IGNORED", selected_move=None, effective_move=None))
        elif action < 0:
            rows.append(dict(kind="MISSING_LEGAL_ACTION", selected_move=None, effective_move=None))
        elif action >= 4:
            rows.append(dict(kind="SWITCH", selected_move=None, effective_move=None))
        else:
            p = raw["active"][side]
            selected = vocabulary[raw["moves"][side][p][action]]
            charge = raw["charge_move"][side][p]
            if charge:
                rows.append(dict(kind="CHARGE_RELEASE", selected_move=selected,
                                 effective_move=vocabulary[charge]))
            elif not any(usable[side]):
                rows.append(dict(kind="SYNTHETIC_STRUGGLE", selected_move=None,
                                 effective_move="struggle"))
            else:
                rows.append(dict(kind="MOVE", selected_move=selected, effective_move=selected))
    return rows


def source_rows(count, team_start, prefix_cycle):
    if (type(count) is not int or count < 1000 or type(team_start) is not int or team_start < 0
            or type(prefix_cycle) is not int or prefix_cycle < 1):
        raise ValueError("fixed thousands-slot roster, team range and request-step cycle required")
    return tuple(dict(slot=i, teams=[team_start + 2*i, team_start + 2*i + 1],
                      prefix_requests=i % prefix_cycle) for i in range(count))


def requested_actions(state, table, *, salt):
    import jax
    import jax.numpy as jnp
    from .benchmark_games import sample_actions
    from .transition import action_mask
    legal = action_mask(state, table)
    keys = jax.vmap(lambda key: jax.random.split(jax.random.fold_in(key, salt), 2))(state.keys)
    drawn = sample_actions(jnp.zeros(legal.shape), legal, keys)
    hp = jnp.take_along_axis(state.hp, state.active[..., None], axis=2)[..., 0]
    forced = state.forced_switch | (hp <= 0)
    requested = jnp.where(jnp.any(forced, axis=1)[:, None], forced, True) & ~state.done[:, None]
    # Ignored seats have canonical zero; they never acquire a sampled decision.
    actions = jnp.where(requested, drawn, 0).astype(jnp.int32)
    missing = jnp.any(requested & (drawn < 0), axis=1)
    return actions, missing, requested


def generate_inputs(pool, table, entry, *, choose=requested_actions, species_vocabulary=None,
                    constructors=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from . import transition
    rows = source_rows(entry["slots"], entry["team_start"], entry["prefix_cycle"])
    indices = np.asarray([r["teams"] for r in rows], np.int32)
    if int(indices.max()) >= len(pool["species"]):
        raise ValueError("fixed roster exceeds immutable pool; never wrap indices")
    gender = entry.get("gender_initialization")
    constructor_errors=np.zeros(len(rows),bool)
    if gender is None:
        state = transition.state_from_pool(pool, indices, seed=entry["seed"],
            table=table if constructors is None else None)
        if constructors is not None:
            state=state._replace(gender=jnp.asarray(pool["gender"][indices],jnp.int32))
    else:
        from .gender import initialize_genders
        if (set(gender) != {"seed", "domain", "root_start", "policy"}
                or type(gender["root_start"]) is not int or gender["root_start"] < 0
                or gender["root_start"]+len(rows) > 2**32):
            raise ValueError("explicit bounded gender initialization domain required")
        state = transition.state_from_pool(pool, indices, seed=entry["seed"])
        if constructors is None:
            state = initialize_genders(state, gender["policy"], species_vocabulary,
                seed=gender["seed"], domain=gender["domain"],
                root_ids=[r["slot"]+gender["root_start"] for r in rows],
                explicit_genders=pool["gender"][indices] if "gender" in pool else None)
        else:
            from .gender import validate_policy
            codes=validate_policy(gender["policy"],species_vocabulary)
            explicit=(pool["gender"][indices] if "gender" in pool else np.full(state.gender.shape,-1,np.int32))
            constructor_errors=np.any((codes[np.asarray(state.species)]==-1)&(explicit==-1),axis=(1,2))
            valid=np.flatnonzero(~constructor_errors)
            if len(valid):
                proof=jax.tree.map(lambda x:x[valid],state)
                proof=initialize_genders(proof,gender["policy"],species_vocabulary,
                    seed=gender["seed"],domain=gender["domain"],
                    root_ids=[rows[i]["slot"]+gender["root_start"] for i in valid],
                    explicit_genders=explicit[valid])
                state=state._replace(gender=state.gender.at[valid].set(proof.gender))
    if constructors is not None:
        originals=tuple(constructors.capture(state,i,row["teams"],entry) for i,row in enumerate(rows))
    if gender is not None or constructors is not None:
        state = transition.initialize(state, table)
    if constructors is not None:
        post_entry=jax.device_get(state)
    lengths = jnp.asarray([r["prefix_requests"] for r in rows])
    errors = constructor_errors.copy()
    traces = [[] for _ in rows]
    compiled = jax.jit(transition.step)
    for request in range(entry["prefix_cycle"] - 1):
        actions, missing, _ = choose(state, table, salt=entry["rng_domain"] + request)
        result = jax.block_until_ready(compiled(state, actions, table))
        advance = (lengths > request) & ~constructor_errors
        errors |= np.asarray(advance & (missing | result.invalid_actions))
        host_actions = np.asarray(actions)
        for i, row in enumerate(rows):
            if row["prefix_requests"] > request and not constructor_errors[i]:
                traces[i].append(host_actions[i].tolist())
        state = jax.tree.map(lambda old, new: jnp.where(
            advance.reshape((len(rows),) + (1,)*(old.ndim-1)), new, old), state, result.state)
    actions, missing, requested = choose(state, table, salt=entry["rng_domain"] + entry["prefix_cycle"])
    errors |= np.asarray(missing)
    from .state import usable_moves
    result=(jax.device_get(state), np.asarray(actions), rows, traces, errors,
            np.asarray(requested), np.asarray(usable_moves(state)))
    return result if constructors is None else (*result,originals,post_entry,constructor_errors)


def runtime_admission(entry):
    # Check BEFORE importing poke_engine, whose exact-roll flag is a OnceLock.
    if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
        raise ValueError("exact native rolls must be enabled before this process starts")
    root = Path(__file__).resolve().parents[1]
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "gpu_engine").glob("*.py"))}
    if actual != entry["gpu_sources_sha256"]:
        raise ValueError("full GPU source inventory differs from reviewed entry")
    gender = entry.get("gender_initialization")
    if gender is not None:
        sources = gender["policy"]["provenance"]["generator_sources"]
        if set(sources) != {"gender_policy.cjs", "generate_teams.cjs"} or any(
                hashlib.sha256((root/"gpu_engine"/name).read_bytes()).hexdigest() != value
                for name, value in sources.items()):
            raise ValueError("gender lookup generator source differs from reviewed policy")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != entry["source_commit"]:
        raise ValueError("runtime Git commit differs from entry")
    native = Path(importlib.import_module("poke_engine.poke_engine").__file__)
    if hashlib.sha256(native.read_bytes()).hexdigest() != entry["native_sha256"]:
        raise ValueError("native binding differs from behaviorally attested profile")
    package = {str(p.relative_to(native.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(native.parent.rglob("*.py"))}
    versions = dict(python=platform.python_version(), **{name:importlib.metadata.version(name)
                    for name in ("jax", "jaxlib", "numpy", "poke-engine")})
    if package != entry["native_python_sha256"] or versions != entry["runtime_versions"]:
        raise ValueError("native wrapper/dependency versions differ from entry")
    if entry.get("constructor_provenance") is not None:
        from .source_provenance import RuntimeAttestation,verify_native_runtime
        verify_native_runtime(RuntimeAttestation(**entry["constructor_provenance"]["runtime"]))
    return dict(source_commit=commit, native=str(native), source_inventory=actual,
                native_python_sha256=package, runtime_versions=versions)


def run(entry, pool_path, out):
    import jax
    import jax.numpy as jnp
    from . import future_state as mapping
    from .benchmark import load_bound_pool, sole_gpu
    from .chance_consumers import plan_for_state
    from .fidelity_panel import JaxCaseExecutor, compare_slot, native_executor
    from .transition import tables_from_metadata

    if out.exists():
        raise FileExistsError("exclusive new panel directory required; no resume")
    admission = runtime_admission(entry)
    device = sole_gpu()
    constructors=None
    if entry.get("constructor_provenance") is None:
        pool, metadata, digest = load_bound_pool(pool_path)
    else:
        from .prospective_panel import admit
        constructors=admit(entry,pool_path)
        pool=constructors.pool.arrays;metadata=constructors.context.metadata()
        digest=constructors.context.pool_sha256
    if digest != entry["pool_sha256"]:
        raise ValueError("immutable pool digest mismatch")
    if entry.get("gender_initialization") is not None:
        policy = entry["gender_initialization"]["policy"]
        if (policy["provenance"]["compiled_tree"] != metadata["compiled_tree"]
                or policy["provenance"]["pool_sha256"] != digest):
            raise ValueError("gender policy compiled Dex differs from immutable pool")
    table = tables_from_metadata(metadata if constructors is None else constructors.pool.metadata())
    out.mkdir()
    started = time.monotonic()
    with (out / "admission.json").open("xb") as stream:
        stream.write(encoded(dict(**admission, device=device.device_kind, entry=entry)))
    generated = generate_inputs(pool, table, entry,
        species_vocabulary=metadata["vocabulary"]["species"],constructors=constructors)
    state, actions, roster, traces, source_errors, requested, usable = generated[:7]
    if constructors is not None:
        originals,post_entry,constructor_errors=generated[7:]
        with (out/"constructor-context.json").open("xb") as stream:
            stream.write(encoded(constructors.frozen_context()))
    records = []
    record_hashes=[]
    # Entire source input artifact closes before the first native/planner call.
    input_digest = hashlib.sha256()
    with gzip.open(out / "inputs.jsonl.gz", "xb") as stream:
        for i, slot in enumerate(roster):
            raw = mapping._sample(state, i)
            record = dict(**slot, state=raw, actions=actions[i].tolist(),
                prefix_actions=traces[i], source_error=bool(source_errors[i]),
                terminal=bool(raw["done"]), requested=requested[i].tolist(),
                request_labels=request_labels(raw, actions[i].tolist(), requested[i].tolist(),
                    usable[i].tolist(), metadata["vocabulary"]["moves"]))
            if constructors is not None:
                record.update(constructor_provenance=asdict(originals[i]),
                    post_entry_state=mapping._sample(post_entry,i),
                    constructor_context_sha256=constructors.context.sha256,
                    constructor_source_error=bool(constructor_errors[i]))
            records.append(record)
            if constructors is not None:
                from .source_provenance import _sha
                record_hashes.append(_sha(record))
            line = encoded(record) + b"\n"
            input_digest.update(line)
            stream.write(line)
    frozen = dict(slots=len(records), input_jsonl_sha256=input_digest.hexdigest(),
                  compressed_sha256=hashlib.sha256((out / "inputs.jsonl.gz").read_bytes()).hexdigest())
    if constructors is not None:
        frozen["constructor_context_file_sha256"]=hashlib.sha256((out/"constructor-context.json").read_bytes()).hexdigest()
    with (out / "inputs-frozen.json").open("xb") as stream:
        stream.write(encoded(frozen))
    print(json.dumps(dict(event="INPUTS_FROZEN", **frozen)), flush=True)

    counts, categories, by_move = Counter(), Counter(), {}
    for record in records:
        index = record["slot"]
        # Durable START identifies an interrupted slot to the owned supervisor.
        atomic_json(out / "current-slot.json", dict(slot=index, status="STARTED", inputs=frozen))
        # Preserve exact source dtypes, notably uint32 PRNG addresses.
        root = jax.tree.map(lambda x: jnp.asarray(x[index:index+1]), state)
        if record["source_error"] or record["terminal"]:
            result = dict(status="UNKNOWN", category="SOURCE_CONSTRUCTOR_FAILURE" if record.get("constructor_source_error")
                          else "SOURCE_REQUEST_FAILURE" if record["source_error"]
                          else "TERMINAL_SOURCE_ROOT", identified_mass="0", unknown_mass="1")
        else:
            if constructors is None:
                plan = plan_for_state(root, 0, tuple(record["actions"]), table, metadata,
                                      cap=entry["max_cases"])
            else:
                from .prospective_panel import plan_for_record
                initial=jax.tree.map(lambda x:jnp.asarray(x[index:index+1]),post_entry)
                plan=plan_for_record(root,initial,record,originals[index],constructors,table,entry,
                    record_sha256=record_hashes[index])
            if isinstance(plan, mapping.Refused):
                codes = sorted({gap.code for gap in plan.gaps})
                result = dict(status="UNKNOWN", category="CASE_CAP" if "CHANCE_CARDINALITY_CAP" in codes
                              else "INPUT_OR_PLAN_REFUSED", gap_codes=codes, detail=asdict(plan),
                              identified_mass="0", unknown_mass="1")
            else:
                executor = JaxCaseExecutor(root, tuple(record["actions"]), table, plan,
                                          batch_size=entry["case_batch"])
                execute_native=(plan.native_results if constructors is not None and record["state"]["pending_residual"]
                                else lambda:native_executor(plan))
                result = compare_slot(plan, executor, execute_native,
                    batch_size=entry["case_batch"], mass_tolerance=entry["mass_tolerance"],
                    probability_tolerance=entry["probability_tolerance"])
                result["consumer_certificate"] = dict(source=plan.source_certificate,
                    table=plan.table_certificate, axes=[asdict(axis) for axis in plan.axes],
                    cardinality=plan.cardinality, native_input=plan.native_input.raw_native,
                    native_actions=(asdict(plan.native_actions) if constructors is not None else plan.native_actions.choices))
                if constructors is not None and record["state"]["pending_residual"]:
                    result["consumer_certificate"]["native_cases"]=plan.native_evidence()
        result.update(slot=index, input_jsonl_sha256=frozen["input_jsonl_sha256"])
        # Exclusive per-slot compressed full law; no accepted-prefix rewriting.
        with gzip.open(out / f"slot-{index:04d}.json.gz", "xb") as stream:
            stream.write(encoded(result))
        counts[result["status"]] += 1
        if result["status"] == "UNKNOWN":
            categories[result["category"]] += 1
        for label in record["request_labels"]:
            if label["effective_move"] is not None:
                move = label["effective_move"]
                counter = by_move.setdefault(move, Counter())
                counter[result["status"]] += 1
        progress = dict(completed_slots=sum(counts.values()), fixed_slots=len(records),
            unattempted=len(records)-sum(counts.values()), counts=dict(counts),
            unknown_categories=dict(categories), by_requested_dispatch={k:dict(v) for k,v in by_move.items()},
            elapsed_seconds=time.monotonic()-started,
            correct_effect_credit="NOT_ASSESSED; selected presence/no-op agreement is not execution credit")
        atomic_json(out / "progress.json", progress)
        print(json.dumps(dict(event="SLOT_COMPLETE", slot=index, status=result["status"], **progress)), flush=True)
    with (out / "terminal.json").open("xb") as stream:
        stream.write(encoded(dict(status="FIXED_PANEL_ATTEMPT_COMPLETE", **progress,
            fidelity_qualified=False, coverage_85_qualified=False, production_go=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entry", type=Path, required=True)
    p.add_argument("--expected-entry-sha", required=True)
    p.add_argument("--team-pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    blob = args.entry.read_bytes()
    if hashlib.sha256(blob).hexdigest() != args.expected_entry_sha:
        raise ValueError("prospective entry hash mismatch")
    run(json.loads(blob), args.team_pool, args.out)


if __name__ == "__main__":
    main()
