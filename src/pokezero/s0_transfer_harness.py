"""Run the finite, unscored scratch transfer harness. No training or S0 dispatch."""

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import zipfile

from .s0_transfer_roster import (
    DRY_RUN_ROOT, FP, H2H, SCHEMA, cell_key, cell_record, digest, dry_run_clusters,
    expected_cells, journal_path, write_once,
)


MAX_REQUESTS = 250
JOIN_SECONDS = 900
CELL_SECONDS = 150
WARMUP_ROOT = DRY_RUN_ROOT + 5000


def load_settings(runtime_path, expected_sha, labels_path, labels_sha):
    import jax
    from . import v5_contract, v5_worker

    runtime = v5_worker._record(dict(path=str(runtime_path), sha256=expected_sha))
    labels = v5_worker._record(dict(path=str(labels_path), sha256=labels_sha))
    if "model" in labels:
        # Reuse the pinned label geometry only. No recorded game or seed is executed.
        from .v5_model import V5ModelConfig
        labels = V5ModelConfig.from_metadata(labels["model"]).label_vocabulary
    else:
        labels = v5_contract.LabelVocabulary(**labels)
    pool_sha = runtime["constructor_pool"]["sha256"]
    profile, vocabulary, gender, category = v5_worker._runtime_metadata(runtime, expected_pool_sha256=pool_sha)
    config = profile.configurations(labels, category_vocabulary=category, arm="B", replicate=0)
    plan = SimpleNamespace(pool_sha256=pool_sha, self_moves=False)
    inputs = v5_worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), "")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    if (jax.__version__ != runtime["versions"]["jax"] or jax.config.jax_default_prng_impl != "threefry2x32"
            or not jax.config.jax_threefry_partitionable or jax.config.jax_high_dynamic_range_gumbel):
        raise ValueError("scratch run must use the pinned JAX version and categorical defaults")
    return inputs, config


def load_actor(actor, config):
    from gpu_engine import v5_model

    params, actual, metadata = v5_model.load(
        actor["archive"], expected_archive_sha256=actor["archive_sha256"],
        expected_checkpoint_sha256=actor["checkpoint_sha256"], enable_v5=True,
        expected_category_vocabulary_sha256=config.category_vocabulary_sha256,
        expected_label_vocabulary_sha256=config.label_vocabulary.sha256,
    )
    if actual != config or metadata.get("self_moves", False) != actor["self_moves"]:
        raise ValueError("checkpoint model or archive observation convention differs")
    return params


def fixed_fixture(actor, config):
    """Verify same-weight model, value buffers, masked law and deterministic draw."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpu_engine import v5_model
    from gpu_engine.v5_evaluation_sampling import EvaluationDraw

    params = load_actor(actor, config)
    inputs = dict(
        categorical_ids=jnp.zeros((1, 1, 29, 41), jnp.int32),
        numeric_features=jnp.zeros((1, 1, 29, 132), jnp.float32).at[0, 0, 0, 0].set(.25),
        token_type_ids=jnp.asarray(v5_model.TOKEN_ROLES, jnp.int32).reshape((1, 1, 29)),
        attention_mask=jnp.ones((1, 1, 29), bool), history_mask=jnp.ones((1, 1), bool),
    )
    # These are fixed synthetic model inputs. They exercise checkpoint export;
    # cross-encoder parity on real roots is a separate fidelity gate.
    policy = jax.device_get(v5_model.forward(params, inputs, config, compute_dtype="bfloat16"))
    if not bool(np.asarray(policy.valid)[0]):
        raise ValueError("synthetic checkpoint fixture refused")
    mask = [True, False, True, False, True, True, True, False, False]
    choice, probabilities = EvaluationDraw(DRY_RUN_ROOT + 3000, DRY_RUN_ROOT, 0, 0).choose(policy.policy_logits[0], mask)
    return digest(dict(logits=policy.policy_logits.tolist(), value=policy.value.tolist(),
                       value_logits=None if policy.value_logits is None else policy.value_logits.tolist(),
                       probabilities=probabilities, choice=choice, mask=mask))


def prepare(args, inputs, config, root):
    from . import v5_python_backend, v5_training
    from .s0_transfer_control import shadow, write_actor
    from .s0_transfer_starts import prepare_starts

    prepared = v5_python_backend.prepare(inputs)
    banks = {"scratch": dry_run_clusters()}
    cells = expected_cells(banks, dry_run=True)
    clusters = sorted({cell["cluster"] for cell in cells.values()} | {WARMUP_ROOT})
    starts = prepare_starts(prepared, clusters, gender_seed=DRY_RUN_ROOT + 2000, gender_domain=17)
    actors = {}
    for offset, role in enumerate(("B", "T")):
        train = v5_training.TrainConfig(seed=DRY_RUN_ROOT + 2001 + offset)
        runtime = v5_training.create(config["model"], config["loss"], train, enable_v5=True)
        actors[role] = write_actor(root / role, runtime)
    actors["C"], control = shadow(root / "C", actors["B"], [], dry_run=True, fixture=fixed_fixture)
    if actors["B"]["tensor_sha256"] == actors["T"]["tensor_sha256"]:
        raise ValueError("scratch T and B must exercise distinct weights")
    manifest = dict(
        schema=SCHEMA, dry_run=True, banks=banks, expected_cells_sha256=digest(list(cells.values())),
        runtime_sha256=args.runtime_sha, labels_sha256=args.labels_sha, max_requests=MAX_REQUESTS,
        starts=starts, actors={"scratch": actors}, control={"scratch": control}, bootstrap_seed=DRY_RUN_ROOT + 2003,
        source_commit=source_commit(), static_runtime_source_commit=inputs.runtime["source_commit"],
        bridge_sha256=args.bridge_sha, fp_root=str(args.foulplay_root), fp_python=str(args.foulplay_python),
        fp_revision=args.foulplay_revision, fp_budget_ms=1000,
        policy=dict(sampled_forward="bf16", logits="float32", greedy_forward="float32",
                    law="threefry2x32/fold-cluster-frame-physical-seat/categorical-low",
                    legacy_actor_greedy_tie="lowest_legal_index", jax=inputs.runtime["versions"]["jax"]),
    )
    write_once(root / "manifest.json", manifest)
    return prepared, manifest


def source_commit():
    return os.environ.get("V2_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def wait_for(path, message, *, peer_terminal=None):
    start = time.monotonic()
    while not path.is_file():
        if peer_terminal is not None and peer_terminal.is_file():
            terminal = json.loads(peer_terminal.read_bytes())
            if terminal["status"] in ("FAILED", "PARTIAL"):
                detail = terminal.get("reason", f'worker exit {terminal["worker_exit_code"]}')
                raise ValueError(f"{peer_terminal.name}: {detail}")
        if time.monotonic() - start >= JOIN_SECONDS:
            raise TimeoutError(message)
        time.sleep(2)
    return json.loads(path.read_bytes())


def install_bridge(path, expected_sha):
    """Restore the small hash-bound Node sources omitted by the launcher archive."""
    names = {"battle_bridge.mjs", "battle_bridge_reference_rest.mjs", "battle_bridge_reference_sleep.mjs",
             "battle_bridge_reference_trace.mjs", "battle_bridge_boundary_requests.mjs"}
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("staged bridge source changed")
    dest = Path(__file__).resolve().parents[2] / "scripts"
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
            raise ValueError("unexpected bridge source inventory")
        for name in names:
            blob = archive.read(name)
            p = dest / name
            if p.exists():
                if p.read_bytes() != blob:
                    raise ValueError("existing bridge source differs")
            else:
                with p.open("xb") as stream:
                    stream.write(blob)


def engine(inputs, config, manifest, cells, *, publish, replica="scratch"):
    from gpu_engine.s0_transfer_engine import evaluate
    from . import v5_worker

    runtime = dict(inputs.runtime, source_commit=manifest["source_commit"])
    prepared, pool, _, _ = v5_worker.prepare_device_runtime(replace(inputs, runtime_json=json.dumps(runtime)))
    actors = {role: (actor["archive_sha256"], load_actor(actor, config["model"]))
              for role, actor in manifest["actors"][replica].items() if role != "H"}
    groups = [[c for c in cells if c["role"] == role and c["seat"] == seat]
              for role in ("T", "C") for seat in (0, 1)]
    return evaluate(prepared, pool, config["model"], groups, manifest["starts"], actors,
                    run_sha=digest(manifest), lanes=20, max_requests=MAX_REQUESTS, wall_seconds=CELL_SECONDS,
                    publish=publish)


def factories(prepared, model, actors, *, sampled):
    from gpu_engine import v4_encoder
    from gpu_engine.v5_live_protocol import load_factory

    result = {}
    for role, actor in actors.items():
        contract = replace(prepared.contract, checkpoint_sha256=actor["checkpoint_sha256"],
                           model_config_sha256=actor["model_sha256"])
        catalog = v4_encoder.compile_encoder(
            prepared.compiled_source, prepared.dex, prepared.vocabulary,
            checkpoint_sha256=contract.checkpoint_sha256, feature_masks=contract.encoder_masks(),
        )
        result[role] = load_factory(
            actor["archive"], catalog=catalog, contract=contract, expected_archive_sha256=actor["archive_sha256"],
            expected_checkpoint_sha256=actor["checkpoint_sha256"], enable_v5=True,
            expected_category_vocabulary_sha256=model.category_vocabulary_sha256,
            expected_label_vocabulary_sha256=model.label_vocabulary.sha256, deterministic=not sampled,
            compute_dtype="bfloat16" if sampled else "float32",
        )
    return result


def prime_factories(public_config, packed):
    """Capture one separate real owning opening for independent factory warmups."""
    from .env import BattleStartOverride
    from .v5_python_collection import OwningEnv

    env = OwningEnv(public_config)
    try:
        env.reset_with_start_override(
            seed=WARMUP_ROOT, start_override=BattleStartOverride(packed, observation_format_id="gen3randombattle"),
        )
        return {seat: tuple(lines) for seat, lines in env.owner_lines.items()}
    finally:
        env.close()


def prime_actors(actors, manifest, public_config):
    from gpu_engine.v5_live_protocol import prime_evaluation

    lines = prime_factories(public_config, manifest["starts"][str(WARMUP_ROOT)]["packed"])
    for role, factory in actors.items():
        prime_evaluation(factory, battle_id=f"warmup-{role}", seed=WARMUP_ROOT, owner_lines=lines)


def showdown_h2h(prepared, model, manifest, cells, *, replica="scratch"):
    from .env import BattleStartOverride
    from .s0_transfer_showdown import evaluate_game
    from .s0_transfer_starts import showdown_birth

    actors = factories(prepared, model, {role: actor for role, actor in manifest["actors"][replica].items()
                                         if role != "H"}, sampled=True)
    prime_actors(actors, manifest, prepared.public_config)
    for cell in cells:
        roles = (cell["role"], "B") if cell["seat"] == 0 else ("B", cell["role"])
        start = manifest["starts"][str(cell["cluster"])]
        status, score, witness = evaluate_game(
            prepared.public_config, cell, factories=tuple(actors[role] for role in roles),
            packed_start=BattleStartOverride(start["packed"], observation_format_id="gen3randombattle"),
            check_start=lambda snapshot: showdown_birth(snapshot, start["showdown_birth"]),
            max_requests=MAX_REQUESTS, wall_seconds=CELL_SECONDS,
        )
        yield cell, status, score, witness


async def showdown_fp(prepared, model, manifest, cells, *, publish, replica="scratch"):
    from .env import BattleStartOverride
    from .foulplay_bridge import ControlledFoulPlayConfig
    from .s0_transfer_starts import showdown_birth
    from .v5_foulplay import run_panel

    fw = manifest["fp_root"]
    revision = subprocess.check_output(["git", "-C", fw, "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["fp_revision"]:
        raise ValueError("actual Foul Play revision differs")
    roster_actors = manifest["actors"][replica]
    actors = factories(prepared, model, {role: actor for role, actor in roster_actors.items() if role != "H"},
                       sampled=False)
    prime_actors(actors, manifest, prepared.public_config)
    limiter = asyncio.Semaphore(4)

    async def play(cell):
        role, seat = cell["role"], cell["seat"]
        start = manifest["starts"][str(cell["cluster"])]
        birth = []
        config = ControlledFoulPlayConfig(
            checkpoint=Path(roster_actors[role]["checkpoint"]),
            showdown_root=Path(prepared.public_config.showdown_root),
            node_binary=prepared.public_config.node_binary, foulplay_root=Path(fw),
            foulplay_python=Path(manifest["fp_python"]), games=1, seed_start=cell["chance"],
            foulplay_random_seed=cell["action"], search_time_ms=manifest["fp_budget_ms"],
            max_decision_rounds=MAX_REQUESTS, policy_mode="raw", encoder_kind="jax-owning",
            device="cpu", belief_set_source=True, owning_decision_timeout_seconds=60.,
            opponent_crash_retries=0, allow_search_fallback=False, pokezero_player=f"p{seat + 1}",
            record_refusals=False,
        )
        async with limiter:
            panel = await run_panel(
                config, actors[role], enable_v5=True, packed_starts={
                    0: BattleStartOverride(start["packed"], observation_format_id="gen3randombattle"),
                },
                packed_birth_callback=lambda snapshot: birth.append(showdown_birth(snapshot, start["showdown_birth"])),
            )
        raw = panel["rows"][0]
        if raw["status"] != "COMPLETE":
            status = raw["status"]
            if status == "FAILED" and raw["reason"] == "RuntimeError: owning game round cap without actual owner terminal":
                status = "CAPPED"
            return cell, status, None, dict(reason=raw["reason"], source=raw["source_game"],
                                            birth=birth[0] if len(birth) == 1 else None)
        witness = raw["source_game"]["terminal_witness"]
        if len(birth) != 1:
            raise ValueError("completed Foul Play cell lacks one verified native birth")
        player = witness["winner_player"]
        return cell, "COMPLETE", raw["score"], dict(
            marker=witness["public_terminal_line"], players=[witness["players"][p] for p in ("p1", "p2")],
            winner=2 if player is None else int(player[1]) - 1, source=raw["source_game"],
            birth=birth[0], seed=raw["seed"], fp_seed=raw["fp_seed"],
        )

    async def publish_finished(cell):
        publish(await play(cell))

    await asyncio.gather(*(publish_finished(cell) for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=("engine", "showdown"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--labels-sha", required=True)
    parser.add_argument("--foulplay-root", type=Path, required=True)
    parser.add_argument("--foulplay-python", type=Path, required=True)
    parser.add_argument("--foulplay-revision", required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-sha")
    args = parser.parse_args()
    root = args.out.resolve()
    inputs, config = load_settings(args.runtime, args.runtime_sha, args.labels, args.labels_sha)
    if args.side == "showdown":
        if args.bridge is not None:
            install_bridge(args.bridge, args.bridge_sha)
        prepared, manifest = prepare(args, inputs, config, root)
        joined = wait_for(root / "engine.join.json", "engine partner did not join",
                          peer_terminal=root / "engine.terminal.json")
        if joined != dict(manifest_sha256=digest(manifest)):
            raise ValueError("engine joined a different scratch manifest")
    else:
        manifest = wait_for(root / "manifest.json", "Showdown partner did not publish a scratch manifest",
                            peer_terminal=root / "showdown.terminal.json")
    if (manifest["source_commit"] != source_commit()
            or manifest["runtime_sha256"] != args.runtime_sha
            or manifest["labels_sha256"] != args.labels_sha or manifest["bridge_sha256"] != args.bridge_sha
            or manifest["dry_run"] is not True):
        raise ValueError("partner source or scratch input manifest differs")
    if args.side == "engine":
        write_once(root / "engine.join.json", dict(manifest_sha256=digest(manifest)))
    expected = {key: cell for key, cell in expected_cells(manifest["banks"], dry_run=True).items()
                if cell["environment"] == args.side}
    cells = list(expected.values())

    def publish(result):
        cell, status, score, evidence = result
        if expected.get(cell_key(cell)) != cell:
            raise ValueError("result belongs to an unexpected scratch cell")
        write_once(journal_path(root, cell), cell_record(manifest, cell, status, score, evidence))

    if args.side == "engine":
        engine(inputs, config, manifest, cells, publish=publish)
    else:
        for result in showdown_h2h(prepared, config["model"], manifest, [c for c in cells if c["assay"] == H2H]):
            publish(result)
        asyncio.run(showdown_fp(prepared, config["model"], manifest, [c for c in cells if c["assay"] == FP],
                               publish=publish))


if __name__ == "__main__":
    main()
