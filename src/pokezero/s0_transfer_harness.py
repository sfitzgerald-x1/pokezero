"""Play one side of a matched S0 transfer panel and write raw terminal rows."""

import argparse
import asyncio
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

from .s0_transfer_roster import (
    DRY_RUN_ROOT, FP, H2H, SCHEMA, canonical, cell_key, digest, dry_run_clusters,
    expected_cells, write_json, Shard, terminal_writer,
)


MAX_REQUESTS = 250
JOIN_SECONDS = 900
WARMUP_OFFSET = 5000
BRIDGE_SOURCES = (
    "battle_bridge.mjs",
    "battle_bridge_boundary_requests.mjs",
    "battle_bridge_reference_rest.mjs",
    "battle_bridge_reference_sleep.mjs",
    "battle_bridge_reference_trace.mjs",
    "battle_bridge_witness_port.mjs",
)
FOUL_PLAY_CAP_REASON = "RuntimeError: owning game round cap without actual owner terminal"


def scratch_label_vocabulary(source, expected_source_hash):
    from .dex import normalize_id
    from .v5_contract import LabelVocabulary

    moves = {normalize_id(move) for universe in source.universes.values()
             for variant in universe.variants for move in variant.moves}
    moves = {"hiddenpower" if m.startswith("hiddenpower") else m for m in moves}
    return LabelVocabulary.from_public_source(source, move_ids=tuple(sorted(moves | {"struggle"})),
                                             expected_source_hash=expected_source_hash)


def load_settings(runtime_path, expected_sha, *, dry_run, self_moves):
    import jax
    from . import v5_worker
    from .live_protocol_actor import ActorContract
    from .randbat import load_gen3_randbat_source_cached

    runtime = v5_worker._record(dict(path=str(runtime_path), sha256=expected_sha))
    pool_sha = runtime["constructor_pool"]["sha256"]
    profile, vocabulary, gender, category = v5_worker._runtime_metadata(runtime, expected_pool_sha256=pool_sha)
    config = None
    if dry_run:
        source = load_gen3_randbat_source_cached(runtime["showdown"]["root"])
        labels = scratch_label_vocabulary(source, ActorContract.champion().belief_set_source_hash)
        config = profile.configurations(labels, category_vocabulary=category, arm="B", replicate=0)
    plan = SimpleNamespace(pool_sha256=pool_sha, self_moves=self_moves)
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


def scratch_actors(root, config):
    from . import v5_training
    from .s0_transfer_control import shadow, write_actor

    actors = {}
    for offset, role in enumerate(("B", "T")):
        train = v5_training.TrainConfig(seed=DRY_RUN_ROOT + 2001 + offset)
        runtime = v5_training.create(config["model"], config["loss"], train, enable_v5=True)
        actors[role] = write_actor(root / role, runtime)
        actors[role]["model"] = config["model"].metadata()
    if actors["B"]["tensor_sha256"] == actors["T"]["tensor_sha256"]:
        raise ValueError("scratch T and B must exercise distinct weights")
    control = shadow(root / "C", actors["B"], 2, fixture=fixed_fixture)
    actors["C"] = dict(control, model=actors["B"]["model"])
    return {"scratch": actors}


def fused_snapshots(entry, cycles, route_sha):
    """Bind the published milestone to the committed fused state and original B."""
    from . import v5_run
    from .v5_deferred_evaluation import read_owned_milestone
    from .v5_evaluation import Artifact

    ref = Artifact(**entry["dispatch"])
    request = read_owned_milestone(ref)
    base = request.run_directory
    state = json.loads((base / "state.json").read_bytes())
    declaration = json.loads((base / "declaration.json").read_bytes())
    snapshots = state["snapshots"]
    reports = state["reports"][:cycles]
    if (base != Path(entry["run"]).resolve() or request.declared_games != cycles * 20_000
            or request.snapshot.iteration != cycles - 1
            or declaration["evaluation"]["route_declaration_sha256"] != route_sha or not route_sha
            or digest(declaration) != request.declaration_sha256
            or state["declaration_sha256"] != request.declaration_sha256
            or [s["iteration"] for s in snapshots[:cycles + 1]] != list(range(-1, cycles))
            or [r["iteration"] for r in reports] != list(range(cycles))
            or any(r["games"] != 20_000 for r in reports)
            or any(sum(len(batch["games"]) for batch in it) != 20_000
                   for it in declaration["iterations"][:cycles])
            or v5_run.Snapshot(**snapshots[cycles]) != request.snapshot):
        raise ValueError("dispatch does not match committed fused milestone")
    return (base, declaration, v5_run.Snapshot(**snapshots[0]), request.snapshot,
            state["checkpoint_encoding_receipts"])


def production_actors(root, spec):
    from . import checkpoint_self_moves, v5_training
    from .live_protocol_actor import ActorContract
    from .s0_transfer_control import shadow, tensor_digest
    from .v5_loss import V5LossConfig
    from .v5_model import V5ModelConfig

    cycles = spec["cadence_iterations"]
    if cycles not in (50, 100):
        raise ValueError("expected a declared 1M or 2M milestone")
    anchor = spec["champion"]
    if anchor["checkpoint_sha256"] != ActorContract.champion().checkpoint_sha256:
        raise ValueError("the Foul Play anchor must be the untouched champion")

    result = {}
    runs, replicate_ids = set(), set()
    for replica, entry in spec["replicas"].items():
        base, declaration, first, trained, receipts = fused_snapshots(
            entry, cycles, spec["route_declaration_sha256"])
        reference = Path(entry["reference_run"]).resolve()
        replicate = entry["replicate"]
        if (type(replicate) is not int or declaration["model"]["replicate"] != replicate
                or replicate in replicate_ids or base.resolve() in runs):
            raise ValueError("registered replicas require distinct, matching training runs")
        runs.add(base.resolve())
        replicate_ids.add(replicate)
        if declaration.get("self_moves", False) != spec["self_moves"]:
            raise ValueError("fused encoder configuration differs")
        actors = {}
        for role, snap, owner in (("B", first, reference), ("T", trained, base)):
            if role == "B":
                checked, self_moves, _ = checkpoint_self_moves.run_checkpoint(
                    owner, snap.checkpoint, snap.checkpoint_sha256, iteration=-1,
                    expected_snapshot=snap, require_snapshot=True)
            else:
                checked = checkpoint_self_moves.checked_checkpoint(
                    owner / snap.checkpoint, expected_checkpoint_sha256=snap.checkpoint_sha256)
                proof = None if spec["self_moves"] else checkpoint_self_moves.off_from_receipt(
                    receipts[snap.checkpoint_sha256])
                self_moves = checkpoint_self_moves.resolve_self_moves(
                    checked, expected_self_moves=spec["self_moves"], off_provenance=proof)
            if self_moves != spec["self_moves"]:
                raise ValueError("baseline/target input convention differs")
            checkpoint = (owner / snap.checkpoint).resolve()
            archive = (owner / snap.archive).resolve()
            if not archive.is_relative_to(owner):
                raise ValueError("actor archive must belong to the training run")
            model = checked.metadata["model"]
            if digest(model) != snap.model_sha256:
                raise ValueError("checkpoint model differs from its published snapshot")
            actor = dict(
                checkpoint=str(checkpoint), checkpoint_sha256=snap.checkpoint_sha256,
                archive=str(archive), archive_sha256=snap.archive_sha256,
                self_moves=self_moves, model=model, model_sha256=snap.model_sha256,
            )
            load_actor(actor, V5ModelConfig.from_metadata(model))
            if role == "B":
                metadata = checked.metadata
                if (metadata["steps"] or metadata["rows"] or metadata["progress"] is not None
                        or checked.payload["optimizer_state"]["state"]):
                    raise ValueError("B must be the original game-zero checkpoint")
                loss = V5LossConfig(**{field.name: metadata["loss"][field.name] for field in fields(V5LossConfig)})
                train = replace(v5_training.TrainConfig(**metadata["training"]), device="cpu")
                baseline = v5_training.create(V5ModelConfig.from_metadata(model), loss, train,
                                               enable_v5=True, self_moves=self_moves)
                baseline.model.load_state_dict(checked.payload["state_dict"], strict=True)
                actor["tensor_sha256"] = tensor_digest(baseline)
            actors[role] = actor
        if actors["B"]["model"] != actors["T"]["model"]:
            raise ValueError("T and B must have the same model configuration within each replica")
        control = shadow(root / replica / "C", actors["B"], cycles, fixture=fixed_fixture,
                         baseline_runtime=baseline)
        actors["C"] = dict(control, model=actors["B"]["model"])
        actors["H"] = anchor
        result[replica] = actors
    return result


def prepare_showdown(inputs):
    from . import v5_python_backend

    prepared = v5_python_backend.prepare(replace(inputs, plan=SimpleNamespace(pool_sha256=inputs.plan.pool_sha256)))
    contract = replace(prepared.contract, self_moves=inputs.plan.self_moves)
    masks = replace(prepared.public_config.feature_masks, self_moves=contract.self_moves)
    return replace(prepared, contract=contract, public_config=replace(prepared.public_config, feature_masks=masks))


def prepare(args, inputs, config, root, spec):
    from .s0_transfer_starts import prepare_starts

    dry_run = spec is None
    banks = {"scratch": dry_run_clusters()} if dry_run else spec["banks"]
    roster_seed = DRY_RUN_ROOT + 2000 if dry_run else spec["roster_seed"]
    if type(roster_seed) is not int or not 0 <= roster_seed < 2**32:
        raise ValueError("uint32 roster seed required")
    if not dry_run and (len(banks) != 2 or banks.keys() != spec["replicas"].keys()
                        or any(len(bank) != 5000 for b in banks.values() for bank in b.values())):
        raise ValueError("declared panel needs two replicas and 5,000 clusters per assay")
    cells = expected_cells(banks, dry_run=dry_run)
    warmup = DRY_RUN_ROOT + WARMUP_OFFSET
    clusters = sorted({cell["cluster"] for cell in cells.values()} | {warmup})
    prepared = prepare_showdown(inputs)
    starts = prepare_starts(prepared, clusters, gender_seed=roster_seed, gender_domain=17)
    actors = scratch_actors(root, config) if dry_run else production_actors(root, spec)
    for roles in actors.values():
        for actor in roles.values():
            actor.pop("encoding_receipt", None)
            actor.pop("tensor_sha256", None)
    manifest = dict(
        schema=SCHEMA, dry_run=dry_run, banks=banks, roster_seed=roster_seed,
        source_commit=source_commit(), starts=starts, actors=actors,
        bootstrap_seed=DRY_RUN_ROOT + 2003 if dry_run else spec["bootstrap_seed"],
        config=dict(
            self_moves=inputs.plan.self_moves, warmup=warmup, max_requests=MAX_REQUESTS,
            control_cycles=2 if dry_run else spec["cadence_iterations"],
            runtime_sha256=args.runtime_sha, bridge_sha256=args.bridge_sha,
            fp_root=str(args.foulplay_root), fp_python=str(args.foulplay_python),
            fp_revision=args.foulplay_revision, fp_budget_ms=1000,
            sampled_forward="bf16", logits="float32", greedy_forward="float32",
            law="threefry2x32/fold-cluster-frame-physical-seat/categorical-low",
            jax=inputs.runtime["versions"]["jax"],
        ),
    )
    if spec is not None:
        manifest.update(route_declaration_sha256=spec["route_declaration_sha256"],
                        dispatches={r: entry["dispatch"] for r, entry in spec["replicas"].items()})
    write_json(root / "manifest.json", manifest)
    return prepared, manifest


def source_commit():
    return os.environ.get("V2_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def wait_for(path, message, *, peer_terminal=None, seconds=JOIN_SECONDS, min_attempt=None,
             peer_attempt=0):
    start = time.monotonic()
    while True:
        try:
            value = json.loads(path.read_bytes())
        except FileNotFoundError:
            pass
        else:
            if min_attempt is None or value.get("attempt", 0) >= min_attempt:
                return value
        if peer_terminal is not None:
            try:
                terminal = json.loads(peer_terminal.read_bytes())
            except FileNotFoundError:
                pass
            else:
                if terminal.get("attempt", 0) >= peer_attempt:
                    raise ValueError(f'peer exited {terminal["exit_code"]} before publishing {path.name}')
        remaining = start + seconds - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(message)
        time.sleep(min(2, remaining))


def pack_bridge(path):
    """Bundle the sources needed to start the staged Showdown bridge."""
    source = Path(__file__).resolve().parents[2] / "scripts"
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in BRIDGE_SOURCES:
            archive.writestr(name, (source / name).read_bytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_bridge(path, expected_sha):
    """Restore the small hash-bound Node sources omitted by the launcher archive."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("staged bridge source changed")
    dest = Path(__file__).resolve().parents[2] / "scripts"
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != list(BRIDGE_SOURCES):
            raise ValueError("unexpected bridge source inventory")
        for name in BRIDGE_SOURCES:
            blob = archive.read(name)
            p = dest / name
            if p.exists():
                if p.read_bytes() != blob:
                    raise ValueError("existing bridge source differs")
            else:
                with p.open("xb") as stream:
                    stream.write(blob)


def engine(inputs, model, manifest, cells, *, publish, replica):
    from gpu_engine.s0_transfer_engine import evaluate
    from . import v5_worker

    runtime = dict(inputs.runtime, source_commit=manifest["source_commit"])
    prepared, pool, _, _ = v5_worker.prepare_device_runtime(replace(inputs, runtime_json=json.dumps(runtime)))
    actors = {role: (actor["archive_sha256"], load_actor(actor, model))
              for role, actor in manifest["actors"][replica].items() if role != "H"}
    groups = [[c for c in cells if c["role"] == role and c["seat"] == seat]
              for role in ("T", "C") for seat in (0, 1)]
    batches = (group[i:i + 20] for group in groups for i in range(0, len(group), 20))
    evaluate(prepared, pool, model, batches, manifest["starts"], actors,
             run_sha=digest([manifest["source_commit"], manifest["roster_seed"], replica]),
             lanes=20, max_requests=manifest["config"]["max_requests"], publish=publish)


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


def prime_factories(public_config, packed, seed):
    """Capture one separate real owning opening for independent factory warmups."""
    from .env import BattleStartOverride
    from .v5_python_collection import OwningEnv

    env = OwningEnv(public_config)
    try:
        env.reset_with_start_override(
            seed=seed, start_override=BattleStartOverride(packed, observation_format_id="gen3randombattle"),
        )
        return {seat: tuple(lines) for seat, lines in env.owner_lines.items()}
    finally:
        env.close()


def prime_actors(actors, manifest, public_config):
    from gpu_engine.v5_live_protocol import prime_evaluation

    seed = manifest["config"]["warmup"]
    lines = prime_factories(public_config, manifest["starts"][str(seed)]["packed"], seed)
    for role, factory in actors.items():
        prime_evaluation(factory, battle_id=f"warmup-{role}", seed=seed, owner_lines=lines)


def showdown_h2h(prepared, model, manifest, cells, *, replica):
    from .env import BattleStartOverride
    from .s0_transfer_showdown import evaluate_game
    from .s0_transfer_starts import showdown_birth

    actors = factories(prepared, model, {role: actor for role, actor in manifest["actors"][replica].items()
                                         if role != "H"}, sampled=True)
    prime_actors(actors, manifest, prepared.public_config)
    for cell in cells:
        roles = (cell["role"], "B") if cell["seat"] == 0 else ("B", cell["role"])
        start = manifest["starts"][str(cell["cluster"])]
        status, witness = evaluate_game(
            prepared.public_config, cell, factories=tuple(actors[role] for role in roles),
            packed_start=BattleStartOverride(start["packed"], observation_format_id="gen3randombattle"),
            check_start=lambda snapshot: showdown_birth(snapshot, start["showdown_birth"]),
            max_requests=manifest["config"]["max_requests"],
        )
        yield cell, status, witness


def foulplay_failure(cell, status, reason):
    if reason == FOUL_PLAY_CAP_REASON:
        status = "CAPPED"
    return cell, status, {"reason": reason}


def champion_game_runner(config):
    """Load H once per worker. Each battle gets fresh owning seats and histories."""
    from . import foulplay_bridge as fp
    from .live_protocol_runtime import champion_actor_factory
    from .v5_foulplay import _TransportOnly

    fp._validate_external_paths(config)
    factory = champion_actor_factory(config)
    model, result = fp.load_transformer_checkpoint(config.checkpoint, map_location="cpu")
    model.eval()
    fp._warn_on_belief_provenance_mismatch(config, result)
    model_config = result.model_config
    env = fp.LocalShowdownConfig(
        showdown_root=config.showdown_root, node_binary=config.node_binary,
        category_vocab=fp.category_vocab_from_model_config(model_config, config.showdown_root),
        observation_spec=fp.observation_spec_from_model_config(model_config),
        feature_masks=fp.feature_masks_from_model_config(model_config))
    dex = fp.load_showdown_dex_cached(config.showdown_root)

    async def game(cell_config, packed, verify_birth):
        return await fp._run_controlled_foulplay_games(
            cell_config, policy=_TransportOnly(), policy_id=model_config.policy_id,
            vocab=env.category_vocab, dex=dex, observation_spec=env.observation_spec,
            feature_masks=env.feature_masks, checkpoint_sha256=factory.contract.checkpoint_sha256,
            owning_actor_factory=factory, actor_runtime=(model, result, env),
            packed_starts={0: packed}, packed_birth_callback=verify_birth)
    return game


async def showdown_fp(prepared, model, manifest, cells, *, publish, replica):
    from .env import BattleStartOverride
    from .foulplay_bridge import ControlledFoulPlayConfig
    from .s0_transfer_starts import showdown_birth
    from .v5_foulplay import run_panel

    run_config = manifest["config"]
    fw = run_config["fp_root"]
    revision = subprocess.check_output(["git", "-C", fw, "rev-parse", "HEAD"], text=True).strip()
    if revision != run_config["fp_revision"]:
        raise ValueError("actual Foul Play revision differs")
    roster_actors = manifest["actors"][replica]
    actors = factories(prepared, model, {role: actor for role, actor in roster_actors.items() if role != "H"},
                       sampled=False)
    prime_actors(actors, manifest, prepared.public_config)
    if "H" in roster_actors:
        import torch

        torch.set_float32_matmul_precision("highest")
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def cell_config(cell):
        role, seat = cell["role"], cell["seat"]
        return ControlledFoulPlayConfig(
            checkpoint=Path(roster_actors[role]["checkpoint"]),
            showdown_root=Path(prepared.public_config.showdown_root),
            node_binary=prepared.public_config.node_binary, foulplay_root=Path(fw),
            foulplay_python=Path(run_config["fp_python"]), games=1, seed_start=cell["chance"],
            foulplay_random_seed=cell["action"], search_time_ms=run_config["fp_budget_ms"],
            max_decision_rounds=run_config["max_requests"], policy_mode="raw", encoder_kind="jax-owning",
            device="cpu", belief_set_source=True, owning_decision_timeout_seconds=60.,
            opponent_crash_retries=0, allow_search_fallback=False, pokezero_player=f"p{seat + 1}",
            record_refusals=False,
        )
    champion_cell = next((cell for cell in cells if cell["role"] == "H"), None)
    champion_game = champion_game_runner(cell_config(champion_cell)) if champion_cell is not None else None

    async def play(cell):
        role = cell["role"]
        config = cell_config(cell)
        start = manifest["starts"][str(cell["cluster"])]
        birth = []
        packed = BattleStartOverride(start["packed"], observation_format_id="gen3randombattle")

        def verify_birth(snapshot):
            birth.append(showdown_birth(snapshot, start["showdown_birth"]))

        if role == "H":
            panel = await champion_game(config, packed, verify_birth)
            game, = panel.games
            fp_seed, = panel.foulplay_random_seed_schedule
            raw = dict(seed=game.seed, fp_seed=fp_seed, source_game=game.to_dict())
        else:
            panel = await run_panel(config, actors[role], enable_v5=True, packed_starts={0: packed},
                                    packed_birth_callback=verify_birth)
            raw = panel["rows"][0]
            if raw["status"] != "COMPLETE":
                return foulplay_failure(cell, raw["status"], raw["reason"])
        witness = raw["source_game"]["terminal_witness"]
        if len(birth) != 1:
            raise ValueError("completed Foul Play cell lacks one verified native birth")
        return cell, "COMPLETE", dict(
            marker=witness["public_terminal_line"], players=[witness["players"][p] for p in ("p1", "p2")],
            birth=birth[0], seed=raw["seed"], fp_seed=raw["fp_seed"],
        )

    pending = iter(cells)

    async def play_next():
        for cell in pending:
            try:
                result = await play(cell)
            except PermissionError:
                raise
            except (ValueError, KeyError, IndexError, TypeError, RuntimeError, OSError) as error:
                # Only the game call is isolated. Ledger and infrastructure failures propagate.
                status = "REFUSED" if isinstance(error, (ValueError, KeyError, IndexError, TypeError)) else "FAILED"
                result = foulplay_failure(cell, status, f"{type(error).__name__}: {error}")
            publish(result)

    await asyncio.gather(*(play_next() for _ in range(4)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["pack-bridge"]:
        parser = argparse.ArgumentParser(
            prog="s0_transfer_harness pack-bridge",
            description="Pack the current Showdown bridge and print worker arguments as JSON.",
        )
        parser.add_argument("output", type=Path)
        parser.add_argument("--worker-path", type=Path, help="destination of the staged archive")
        args = parser.parse_args(argv[1:])
        bridge_sha = pack_bridge(args.output)
        worker_path = args.worker_path or args.output.resolve()
        print(json.dumps(["--bridge", str(worker_path), "--bridge-sha", bridge_sha]))
        return

    from .v5_model import V5ModelConfig

    parser = argparse.ArgumentParser(description=__doc__, epilog="Use pack-bridge to build the staged archive.")
    parser.add_argument("side", choices=("prepare", "engine", "showdown"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha", required=True)
    parser.add_argument("--foulplay-root", type=Path, required=True)
    parser.add_argument("--foulplay-python", type=Path, required=True)
    parser.add_argument("--foulplay-revision", required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-sha")
    parser.add_argument("--shard", type=Shard.parse, default=Shard(), metavar="I/N")
    parser.add_argument("--resume", action="store_true", help="continue only unattempted cells of the same shard")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--join-seconds", type=float, default=JOIN_SECONDS)
    parser.add_argument("--spec", type=Path, help="registered production inputs; omit for the unscored scratch roster")
    args = parser.parse_args(argv)
    if (args.bridge is None) != (args.bridge_sha is None):
        parser.error("--bridge and --bridge-sha must be supplied together")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if args.spec is not None and args.side != "prepare":
        parser.error("prepare the registered run on CPU before submitting either worker")
    if args.shard.count > 1 and args.side != "prepare" and not manifest_path.is_file():
        raise ValueError("prepare a shared manifest before admitting shards")
    if args.side != "engine" and args.bridge is not None:
        install_bridge(args.bridge, args.bridge_sha)
    if args.side == "prepare" or (args.side == "showdown" and not manifest_path.is_file()):
        if manifest_path.exists():
            raise ValueError("use a fresh output directory for a new panel")
        spec = json.loads(args.spec.read_bytes()) if args.spec is not None else None
        self_moves = False if spec is None else spec["self_moves"]
        if type(self_moves) is not bool:
            raise ValueError("explicit self-move encoder setting required")
        inputs, config = load_settings(args.runtime, args.runtime_sha, dry_run=spec is None, self_moves=self_moves)
        prepared, manifest = prepare(args, inputs, config, root, spec)
        if args.side == "prepare":
            return
    else:
        manifest = wait_for(manifest_path, "Showdown did not prepare a run manifest",
                            peer_terminal=root / "showdown.done.json" if args.side == "engine" else None,
                            seconds=args.join_seconds, peer_attempt=args.attempt)
        if (manifest["schema"] != SCHEMA or manifest["source_commit"] != source_commit()
                or manifest["config"]["runtime_sha256"] != args.runtime_sha):
            raise ValueError("worker source or runtime differs from the run manifest")
        inputs, _ = load_settings(args.runtime, args.runtime_sha, dry_run=False,
                                  self_moves=manifest["config"]["self_moves"])
        if args.side == "showdown":
            if manifest["config"]["bridge_sha256"] != args.bridge_sha:
                raise ValueError("Showdown bridge differs from the prepared run")
            prepared = prepare_showdown(inputs)

    expected = {key: cell for key, cell in expected_cells(manifest["banks"], dry_run=manifest["dry_run"]).items()
                if cell["environment"] == args.side and args.shard.owns(cell)}
    with terminal_writer(root, args.side, args.shard, expected, digest(manifest), resume=args.resume) as worker:
        remaining, publish = worker
        for replica, actors in manifest["actors"].items():
            model = V5ModelConfig.from_metadata(actors["B"]["model"])
            cells = [cell for cell in remaining if cell["replica"] == replica]
            if not cells:
                continue
            if args.side == "engine":
                engine(inputs, model, manifest, cells, publish=publish, replica=replica)
            else:
                for result in showdown_h2h(prepared, model, manifest,
                                           [cell for cell in cells if cell["assay"] == H2H], replica=replica):
                    publish(result)
                asyncio.run(showdown_fp(prepared, model, manifest,
                                        [cell for cell in cells if cell["assay"] == FP],
                                        publish=publish, replica=replica))


if __name__ == "__main__":
    main()
