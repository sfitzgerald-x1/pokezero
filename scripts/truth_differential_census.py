#!/usr/bin/env python3
"""Truth-injection differential census (fallback-burndown plan 4, sequencing item 1).

Three modes.

``--mode plan``
    Build the stratified game plan: every gen3 randbat VARIANT appears in at
    least ``--min-games-per-variant`` games and on BOTH seats, as packed Custom
    Game teams observed as ``gen3randombattle``. The plan is a file, so the seed
    block and the team composition are fixed and the census is re-runnable after
    every merge.

``--mode run``
    Play one shard of the plan and, at every decision, run the FULL consumer
    chain on the TRUE world (see :mod:`pokezero.truth_differential`).

``--mode report``
    Merge shard outputs into the inventory.

Build requirement, non-negotiable: the truth arm must run ``leaf_eval=model``
against a crate built ``--features model``. Without it the abort gate in
``rust/pokezero-search/src/tree.rs`` is ``allow(dead_code)`` and ``worlds
searched == worlds constructed`` identically -- the abort channel is
structurally invisible, and any "the abort channel is small" reading taken on
such a build is unfalsifiable rather than evidence (report 4 section 4.1). This
script REFUSES to run the truth arm on a crate without the feature.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

OBSERVATION_FORMAT_ID = "gen3randombattle"


def _default_showdown_root() -> str:
    """Resolve the checkout the way the library does, never as a literal.

    A hardcoded default here put a maintainer home directory into a tracked file,
    which `tests.test_public_invariant` rejects -- and `pokezero.local_showdown.
    default_showdown_root` exists precisely so the library and its harnesses
    cannot drift on this. Resolution order: `POKEZERO_SHOWDOWN_ROOT`, then
    conventional locations expressed via `Path.home()` and the repo root.
    """

    from pokezero.local_showdown import default_showdown_root

    return str(default_showdown_root())


# --- plan --------------------------------------------------------------------


def _variant_rows(set_source: Any, base_by_id: Mapping[str, str]) -> list[tuple[str, Any]]:
    """Every concrete variant in the pool, in a deterministic order.

    Re-derived from the loaded source rather than assumed: the count is a
    function of the enumeration code AND the resolved dex, and stale caches with
    different counts exist on this machine. The run manifest records both the
    source hash and the count actually seen.
    """

    rows: list[tuple[str, Any]] = []
    for species_key in sorted(set_source.universes):
        universe = set_source.universes[species_key]
        clause_key = _clause_key(species_key, base_by_id)
        for variant in universe.variants:
            rows.append((clause_key, variant))
    return rows


def base_species_ids(showdown_root: str) -> dict[str, str]:
    """``{species id: base species id}`` straight from Showdown's own Pokedex.

    Species Clause compares BASE species, and a randbat "species" is a forme:
    `deoxys`, `deoxysattack`, `deoxysdefense` and `deoxysspeed` are four distinct
    pool entries and ONE base species. Composing sides by pool key put all four on
    one team, Showdown collapsed their idents, and the run's top inventory row
    became `self_moveset_mismatch: 'Deoxys' … move 'calmmind' absent from root
    self_team` -- a harness artifact reading as a defect, which is precisely the
    wrong-on-contact shape report 4 section 2.1 is about.

    Read from the dex rather than derived by splitting on ``-``: that split is
    right for `Deoxys-Speed` and wrong for `Ho-Oh`, `Porygon-Z` and `Nidoran-F`,
    and a prefix rule additionally merges `Porygon` into `Porygon2`, which are
    different species that Species Clause allows together.
    """

    root = Path(showdown_root).expanduser().resolve()
    script = """
const root = process.argv[1];
const {Pokedex} = require(root + '/dist/data/pokedex.js');
const out = {};
for (const [id, entry] of Object.entries(Pokedex)) {
  out[id] = entry.baseSpecies || entry.name || id;
}
process.stdout.write(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "-e", script, str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    raw = json.loads(result.stdout)
    return {
        key: "".join(ch for ch in str(value).lower() if ch.isalnum())
        for key, value in raw.items()
        if value
    }


def _clause_key(species_key: str, base_by_id: Mapping[str, str]) -> str:
    """The key Species Clause actually compares, with a loud fallback.

    An unmapped species falls back to its own id, which is the SAFE direction:
    it can only make the harness treat two formes as distinct, and that is the
    bug this function exists to prevent -- so unmapped ids are reported by
    `build_plan` rather than silently absorbed.
    """

    return base_by_id.get(species_key, species_key)


def _round_robin_by_species(rows: Sequence[tuple[str, Any]], rotation: int) -> list[tuple[str, Any]]:
    """Interleave across species so any window of 6 is species-distinct.

    Species Clause is NOT enforced by ``gen3customgame`` (its ruleset is
    ``Max Team Size``/``Max Move Count``/``Max Level``/``Default Level`` only),
    and a duplicated species produces duplicate idents that the request parser
    only survives by unioning movesets. Composing species-distinct sides is
    therefore the harness's job, and round-robin makes it structural instead of
    a rejection loop.
    """

    by_species: dict[str, list[Any]] = defaultdict(list)
    for species_key, variant in rows:
        by_species[species_key].append(variant)
    species_order = sorted(by_species)
    if rotation:
        offset = rotation % len(species_order)
        species_order = species_order[offset:] + species_order[:offset]
    depth = max(len(values) for values in by_species.values())
    ordered: list[tuple[str, Any]] = []
    for index in range(depth):
        for species_key in species_order:
            values = by_species[species_key]
            if index < len(values):
                ordered.append((species_key, values[index]))
    return ordered


def _side_from(
    pool: list[tuple[str, Any]], size: int, refill: Sequence[tuple[str, Any]]
) -> list[tuple[str, Any]]:
    """Consume ``size`` species-distinct variants from the head of ``pool``.

    ``pool`` is mutated. A species collision DEFERS the entry to the head of the
    pool for the next side; it is never dropped -- an earlier version advanced
    past collisions and silently lost 45 of 1,682 variants from a plan whose
    whole point is that no variant is missing. When the pool cannot fill a side,
    ``refill`` tops it up with already-covered variants, which only ADDS coverage.
    """

    picked: list[tuple[str, Any]] = []
    seen: set[str] = set()
    deferred: list[tuple[str, Any]] = []
    refill_cursor = 0
    while len(picked) < size:
        if not pool:
            if refill_cursor >= len(refill):
                break
            pool.append(refill[refill_cursor])
            refill_cursor += 1
        species_key, variant = pool.pop(0)
        if species_key in seen:
            deferred.append((species_key, variant))
            continue
        seen.add(species_key)
        picked.append((species_key, variant))
    pool[:0] = deferred
    if len(picked) != size:  # pragma: no cover - 220 species against 6 slots
        raise RuntimeError(f"cannot compose a species-distinct side of {size}")
    return picked


def build_plan(
    *,
    set_source: Any,
    passes: int,
    seed_base: int,
    showdown_root: str,
    team_size: int = 6,
) -> dict[str, Any]:
    from pokezero.determinization import _fixture_from_variant
    from pokezero.showdown_fixture import pack_team

    base_by_id = base_species_ids(showdown_root)
    unmapped = sorted(key for key in set_source.universes if key not in base_by_id)
    rows = _variant_rows(set_source, base_by_id)
    total = len(rows)
    games: list[dict[str, Any]] = []
    coverage: dict[str, list[list[Any]]] = defaultdict(list)
    seat_seen: dict[str, set[str]] = defaultdict(set)
    for pass_index in range(passes):
        pool = _round_robin_by_species(rows, rotation=pass_index * 7)
        # One PASS consumes the pool exactly once across both sides, so a pass is
        # ceil(total / 2*team_size) games and `passes` passes give every variant
        # `passes` games.
        padding = list(_round_robin_by_species(rows, rotation=pass_index * 7 + 3))
        game_in_pass = 0
        while pool:
            side_a = _side_from(pool, team_size, padding)
            side_b = _side_from(pool, team_size, padding[team_size:])
            # Seat assignment is chosen GREEDILY to maximise newly-covered
            # (variant, seat) pairs, not by a parity rule. A parity rule looks
            # sufficient and is not: with a deterministic rotation a variant's
            # side and the pass parity stay correlated, and the first version left
            # 429 of 1,682 variants on one seat only. Seat asymmetry is real here
            # (report 4's Q2 is an opponent-seat-only class), so a single-seat
            # variant is half a census of that variant.
            gain_direct = sum("p1" not in seat_seen[v.variant_id] for _s, v in side_a) + sum(
                "p2" not in seat_seen[v.variant_id] for _s, v in side_b
            )
            gain_swapped = sum("p2" not in seat_seen[v.variant_id] for _s, v in side_a) + sum(
                "p1" not in seat_seen[v.variant_id] for _s, v in side_b
            )
            if gain_swapped > gain_direct or (
                gain_swapped == gain_direct and pass_index % 2 == 1
            ):
                side_a_seat, side_b_seat = "p2", "p1"
            else:
                side_a_seat, side_b_seat = "p1", "p2"
            seed = seed_base + pass_index * 100000 + game_in_pass
            rng = random.Random(f"truth-census|{pass_index}|{game_in_pass}")
            packed = {}
            for seat, side in ((side_a_seat, side_a), (side_b_seat, side_b)):
                fixtures = [
                    _fixture_from_variant(variant, set_source=set_source, rng=rng)
                    for _species, variant in side
                ]
                packed[seat] = pack_team(tuple(fixtures))
            game = {
                "game_index": len(games),
                "pass": pass_index,
                "seed": seed,
                "packed": packed,
                "variants": {
                    side_a_seat: [variant.variant_id for _s, variant in side_a],
                    side_b_seat: [variant.variant_id for _s, variant in side_b],
                },
            }
            for seat, side in ((side_a_seat, side_a), (side_b_seat, side_b)):
                for _species, variant in side:
                    coverage[variant.variant_id].append([game["game_index"], seat])
                    seat_seen[variant.variant_id].add(seat)
            games.append(game)
            game_in_pass += 1

    uncovered = [row[1].variant_id for row in rows if not coverage.get(row[1].variant_id)]
    per_variant = {vid: len(entries) for vid, entries in coverage.items()}
    seats_per_variant = {vid: sorted({e[1] for e in entries}) for vid, entries in coverage.items()}
    return {
        "schema": "truth-differential-census-plan/v1",
        "source_hash": set_source.metadata.source_hash,
        "format_id": set_source.metadata.format_id,
        "variant_count": total,
        "species_count": len(set_source.universes),
        "base_species_count": len({_clause_key(k, base_by_id) for k in set_source.universes}),
        "species_ids_without_a_base_species": unmapped,
        "passes": passes,
        "seed_base": seed_base,
        "team_size": team_size,
        "games": games,
        "coverage": {
            "min_games_per_variant": min(per_variant.values()) if per_variant else 0,
            "max_games_per_variant": max(per_variant.values()) if per_variant else 0,
            "variants_covered": len(per_variant),
            "variants_uncovered": uncovered,
            "variants_on_one_seat_only": sorted(
                vid for vid, seats in seats_per_variant.items() if len(seats) < 2
            ),
        },
        "variant_games": {vid: entries for vid, entries in sorted(coverage.items())},
    }


# --- run ---------------------------------------------------------------------


def _neutral_cwd_witness(env: Mapping[str, str]) -> dict[str, Any]:
    """Re-resolve the identity witness in a CHILD spawned from a neutral cwd.

    `sys.path` does not cross a subprocess boundary and a RELATIVE
    ``PYTHONPATH=src`` resolves against the CHILD's cwd, so a run can measure one
    tree in-process and a different one in every subprocess it spawns (report 4
    section 4.2 case 4). Running the witness from ``/`` with the same environment
    is the only way to see that from inside the run.
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "pokezero.truth_differential"],
            cwd="/",
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "error": f"exit {completed.returncode}",
            "stderr": completed.stderr[-2000:],
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {"error": f"unparseable witness: {error}", "stdout": completed.stdout[-2000:]}


class _BrokenNative:
    """Wrap the NativeLeafModel INSTANCE, never the module.

    ``search_batched_multi_encoded`` is a METHOD on the instance returned by
    ``EngineMctsPolicy._native()``. Patching the module attribute forces nothing
    and the run then reports a clean zero refusals and looks like a passing
    instrument test -- which is exactly how a capture harness returned 0 records
    and read as healthy.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def search_batched_multi_encoded(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("forced_instrument_test: crate search refused")


def install_forcing(truth_policy: Any, mode: str) -> None:
    """Force a truth rejection, scoped to the TRUTH arm only.

    An instrument that cannot report failure will report success. Before any
    "0 truth-rejections" reading is trusted, one of these modes must be shown to
    produce a counted, ATTRIBUTED rejection -- and the production arm's counters
    must be byte-identical to the unforced run, which is what proves the forcing
    did not leak into the thing being measured.

    Scoping is achieved by wrapping the truth policy's own bound method and
    patching only for the duration of that call. The probe is synchronous and
    single-threaded, so no other arm can observe the patch.
    """

    if mode == "none":
        return

    from pokezero import engine_search as ES
    from pokezero.engine_world import EngineWorldUnsupported

    original = truth_policy.select_action_with_context

    if mode == "construct":

        def broken_world(*_a: Any, **_k: Any) -> Any:
            raise EngineWorldUnsupported(
                "forced_instrument_test", "construction refused by the instrument test"
            )

        def wrapped(context: Any, *, rng: Any) -> Any:
            saved = ES.world_battle_spec
            ES.world_battle_spec = broken_world
            try:
                return original(context, rng=rng)
            finally:
                ES.world_battle_spec = saved

    elif mode == "abort":
        original_native = type(truth_policy)._native

        def wrapped(context: Any, *, rng: Any) -> Any:
            saved = truth_policy.__dict__.get("_native")
            truth_policy._native = lambda: _BrokenNative(original_native(truth_policy))
            try:
                return original(context, rng=rng)
            finally:
                if saved is None:
                    truth_policy.__dict__.pop("_native", None)
                else:
                    truth_policy._native = saved

    elif mode == "unmapped":
        original_map = truth_policy._map_choices

        def wrapped(context: Any, *, rng: Any) -> Any:
            truth_policy._map_choices = lambda *a, **k: None
            try:
                return original(context, rng=rng)
            finally:
                truth_policy._map_choices = original_map

    else:
        raise SystemExit(f"unknown --force mode {mode!r}")

    truth_policy.select_action_with_context = wrapped


def run_shard(args: argparse.Namespace) -> int:
    from pokezero.collection import policy_from_spec  # noqa: F401 - parity with prod harnesses
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_search import (
        EngineMctsConfig,
        EngineMctsPolicy,
        EngineSearchFallbackWarning,
        EnvTier2AnnotationSource,
    )
    from pokezero.env import BattleStartOverride
    from pokezero.local_showdown import (
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_from_checkpoint_provenance,
    )
    from pokezero.neural_policy import (
        category_vocab_from_model_config,
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state
    from pokezero.truth_differential import (
        TruthDifferentialProbe,
        TruthWorldBuilder,
        aggregate_records,
        identity_witness,
        probe_policy_config,
    )

    warnings.simplefilter("ignore", EngineSearchFallbackWarning)

    witness = identity_witness()
    child_witness = _neutral_cwd_witness(os.environ)
    if not witness.get("pokezero_search_model_feature"):
        print(json.dumps(witness, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(
            "REFUSING TO RUN: pokezero_search was built WITHOUT --features model. "
            "The tree.rs attribution-unsafe abort gate is allow(dead_code) on such a "
            "build, so worlds_searched == worlds_constructed identically and the abort "
            "channel is invisible. Rebuild with scripts/build_search_crate_model.sh."
        )
    mismatches = {
        key: [witness.get(key), child_witness.get(key)]
        for key in ("pokezero_file", "engine_search_file", "truth_differential_file",
                    "pokezero_search_so_sha256")
        if witness.get(key) != child_witness.get(key)
    }

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    dex = load_showdown_dex_cached(args.showdown_root)
    set_source = load_gen3_randbat_source_cached(args.showdown_root)
    if plan is not None and plan["source_hash"] != set_source.metadata.source_hash:
        raise SystemExit(
            f"plan source_hash {plan['source_hash']} != loaded {set_source.metadata.source_hash}"
        )

    env_config = LocalShowdownConfig(
        showdown_root=Path(args.showdown_root), set_belief_source=True
    )
    model_config = load_transformer_model_config(str(args.checkpoint))
    env_config = env_config_from_checkpoint_provenance(
        env_config,
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=category_vocab_from_model_config(
            model_config, env_config.resolved_showdown_root()
        ),
        context="truth-injection differential census",
    )
    env = LocalShowdownEnv(env_config)
    annotation_source = EnvTier2AnnotationSource(env)

    driver_config = EngineMctsConfig(
        leaf_eval=args.driver_leaf_eval,
        model_path=args.model_path if args.driver_leaf_eval == "model" else None,
        checkpoint_path=args.checkpoint if args.driver_leaf_eval == "model" else None,
        tables_path=args.tables if args.driver_leaf_eval == "model" else None,
        model_device="cpu",
        worlds=args.driver_worlds,
        sample_retry_factor=args.sample_retry_factor,
        search_sims=args.driver_sims,
        search_batch=args.driver_batch,
        search_depth=args.driver_depth,
    )
    truth_config = probe_policy_config(
        dataclasses.replace(
            driver_config,
            leaf_eval="model",
            model_path=args.model_path,
            checkpoint_path=args.checkpoint,
            tables_path=args.tables,
            search_depth=args.truth_depth,
            search_batch=args.truth_batch,
        ),
        sims=args.truth_sims,
    )

    def make(config: Any, name: str) -> Any:
        return EngineMctsPolicy(
            dex=dex,
            set_source=set_source,
            annotation_source=annotation_source,
            config=config,
            policy_id=name,
        )

    records: list[Any] = []
    exemplars: dict[str, Any] = {}
    builder = TruthWorldBuilder(env, set_source=set_source)
    probes: dict[str, Any] = {}
    truth_policies: dict[str, Any] = {}
    driver_policies: dict[str, Any] = {}
    for seat in ("p1", "p2"):
        driver = make(driver_config, f"tdc-driver-{seat}")
        truth = make(truth_config, f"tdc-truth-{seat}")
        install_forcing(truth, args.force)
        driver_policies[seat] = driver
        truth_policies[seat] = truth
        probes[seat] = TruthDifferentialProbe(
            primary=driver,
            truth_policy=truth,
            truth_builder=builder,
            records=records,
            seed=0,
            repeats=args.truth_repeats,
            exemplar_store=exemplars,
        )

    games = _shard_games(plan, args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    per_game: list[dict[str, Any]] = []

    def dump() -> None:
        payload = {
            "schema": "truth-differential-census-shard/v1",
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "driver_config": dataclasses.asdict(driver_config),
            "truth_config": dataclasses.asdict(truth_config),
            "identity_witness": witness,
            "identity_witness_child_neutral_cwd": child_witness,
            "identity_witness_mismatches": mismatches,
            "plan_source_hash": plan["source_hash"] if plan else None,
            "wall_seconds": time.perf_counter() - started,
            "per_game": per_game,
            "instrument_errors": sorted(
                {error for probe in probes.values() for error in probe.errors}
            ),
            "instrument_error_count": sum(len(probe.errors) for probe in probes.values()),
            "driver_stats": {
                seat: policy.stats.to_dict() for seat, policy in driver_policies.items()
            },
            "truth_stats": {
                seat: policy.stats.to_dict() for seat, policy in truth_policies.items()
            },
            "summary": aggregate_records(records),
            "records": [record.to_dict() for record in records],
        }
        out_path.write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=str), encoding="utf-8"
        )

    try:
        for index, game in enumerate(games):
            seed = int(game["seed"])
            battle_id = f"tdc-{args.tag}-{seed}"
            for probe in probes.values():
                probe.seed = seed
            builder.reset()
            before = len(records)
            game_started = time.perf_counter()
            try:
                if game.get("packed"):
                    env.reset_with_start_override(
                        seed=seed,
                        start_override=BattleStartOverride(
                            player_teams=dict(game["packed"]),
                            observation_format_id=OBSERVATION_FORMAT_ID,
                        ),
                    )
                else:
                    env.reset(seed=seed, format_id=OBSERVATION_FORMAT_ID)
                continue_rollout_from_current_state(
                    env=env,
                    policies={"p1": probes["p1"], "p2": probes["p2"]},
                    config=RolloutConfig(
                        max_decision_rounds=args.max_rounds,
                        format_id=OBSERVATION_FORMAT_ID,
                    ),
                    seed=seed,
                    battle_id=battle_id,
                    reset_policies=False,
                )
                ok, error = True, None
            except Exception as exc:  # noqa: BLE001 - one bad game must not lose the shard
                ok, error = False, f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - game_started
            per_game.append(
                {
                    "game_index": game.get("game_index", index),
                    "seed": seed,
                    "battle_id": battle_id,
                    "ok": ok,
                    "error": error,
                    "decisions": len(records) - before,
                    "wall_seconds": round(elapsed, 3),
                }
            )
            print(
                f"[{args.tag}] game {index + 1}/{len(games)} seed={seed} ok={ok} "
                f"decisions+={len(records) - before} {elapsed:.1f}s "
                f"total={time.perf_counter() - started:.0f}s",
                flush=True,
            )
            if index % args.dump_every == 0 or index == len(games) - 1:
                dump()
            if args.limit_decisions and len(records) >= args.limit_decisions:
                print(f"[{args.tag}] decision limit reached", flush=True)
                break
    finally:
        dump()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = aggregate_records(records)
    print("=" * 78)
    print(f"wrote {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "predicates"}, indent=2)[:4000])
    return 0


def _shard_games(plan: Mapping[str, Any] | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    if plan is None:
        base = int(args.control_seed_base)
        games = [
            {"game_index": index, "seed": base + index, "packed": None}
            for index in range(args.control_games)
        ]
    else:
        games = list(plan["games"])
    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        games = [game for position, game in enumerate(games) if position % total == index]
    if args.max_games:
        games = games[: args.max_games]
    return games


# --- report ------------------------------------------------------------------


def merge_shards(paths: Iterable[Path]) -> dict[str, Any]:
    from pokezero.truth_differential import aggregate_records

    records: list[Any] = []
    shards: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("records") or [])
        shards.append(
            {
                "path": str(path),
                "wall_seconds": payload.get("wall_seconds"),
                "instrument_error_count": payload.get("instrument_error_count"),
                "instrument_errors": (payload.get("instrument_errors") or [])[:10],
                "identity_witness_mismatches": payload.get("identity_witness_mismatches"),
                "identity_witness": payload.get("identity_witness"),
                "games_ok": sum(1 for g in payload.get("per_game") or [] if g.get("ok")),
                "games_failed": sum(1 for g in payload.get("per_game") or [] if not g.get("ok")),
                "driver_config": payload.get("driver_config"),
                "truth_config": payload.get("truth_config"),
            }
        )
    summary = aggregate_records(records)
    summary["shards"] = shards
    summary["shard_count"] = len(shards)
    summary["instrument_error_total"] = sum(
        int(shard.get("instrument_error_count") or 0) for shard in shards
    )
    summary["games_ok"] = sum(int(shard["games_ok"]) for shard in shards)
    summary["games_failed"] = sum(int(shard["games_failed"]) for shard in shards)
    return summary


# --- cli ---------------------------------------------------------------------


def render_queue(summary: Mapping[str, Any], *, commands: Sequence[str], title: str) -> str:
    """Render the inventory as the new QUEUE, one table per UNIT.

    ``world_failure_reasons`` counts WORLDS, ``fallback_reasons`` counts
    DECISIONS and ``lossy_subcase_renders`` counts BRANCH RENDERS. They are never
    co-ranked in one table, and every figure is printed beside the command that
    produced it.
    """

    lines: list[str] = [f"# {title}", ""]
    lines.append("## Commands that produced every figure below")
    lines.append("")
    lines.append("```sh")
    lines.extend(commands)
    lines.append("```")
    lines.append("")
    probed = summary.get("truth_probed_decisions") or 0
    rate = summary.get("truth_rejection_rate")
    lines.append("## The number")
    lines.append("")
    lines.append("| quantity | unit | value |")
    lines.append("|---|---|---|")
    lines.append(f"| truth-rejection rate | DECISIONS | {_pct(rate)} |")
    lines.append(
        f"| truth-rejected decisions | DECISIONS | {summary.get('truth_rejected_decisions')} |"
    )
    lines.append(f"| truth-probed decisions | DECISIONS | {probed} |")
    lines.append(
        f"| distinct open predicates | PREDICATES | {summary.get('distinct_open_predicates')} |"
    )
    lines.append(
        "| sampler-search-failure rate | DECISIONS | "
        f"{_pct(summary.get('sampler_search_failure_rate'))} |"
    )
    lines.append(
        "| production fallback rate (same block) | DECISIONS | "
        f"{_pct(summary.get('production_fallback_rate'))} |"
    )
    lines.append(
        f"| decisions seen | DECISIONS | {summary.get('decisions_seen')} |"
    )
    lines.append(f"| battles | BATTLES | {summary.get('battles')} |")
    lines.append(
        "| truth unavailable (instrument gap) | DECISIONS | "
        f"{summary.get('truth_unavailable_decisions')} |"
    )
    lines.append(
        f"| instrument errors | EVENTS | {summary.get('instrument_error_total')} |"
    )
    lines.append("")
    lines.append("## The queue: every predicate that rejected the TRUE world")
    lines.append("")
    lines.append(
        "Frequency is UNCAPPED. `decisions` is the number of decisions on which the "
        "predicate rejected the truth; `counter units` is the raw source-counter delta, "
        "whose unit is that counter's unit (WORLDS for construction and crate keys, "
        "DECISIONS for fallback literals). Never compare the two columns across stages."
    )
    lines.append("")
    lines.append("| # | predicate | stage | channel family | decisions | counter units |")
    lines.append("|---|---|---|---|---|---|")
    for index, row in enumerate(summary.get("predicates") or [], start=1):
        lines.append(
            f"| {index} | `{row['predicate']}` | {row['stage']} | {row['family']} | "
            f"{row['decisions']} | {row['counter_units']} |"
        )
    if not summary.get("predicates"):
        lines.append("| - | *(empty)* | - | - | 0 | 0 |")
    lines.append("")
    lines.append("## Exemplars (one per predicate, first occurrence)")
    lines.append("")
    for row in summary.get("predicates") or []:
        exemplar = row.get("exemplar") or {}
        lines.append(f"### `{row['predicate']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(exemplar, indent=1, sort_keys=True))
        lines.append("```")
        lines.append("")
    lines.append("## Branch renders (a SEPARATE unit; not refusals)")
    lines.append("")
    lines.append("| lossy subcase | BRANCH RENDERS |")
    lines.append("|---|---|")
    for name, count in (summary.get("lossy_subcase_renders") or {}).items():
        lines.append(f"| `{name}` | {count} |")
    if not summary.get("lossy_subcase_renders"):
        lines.append("| *(none observed)* | 0 |")
    lines.append("")
    lines.append("## Decision-level fallback literals")
    lines.append("")
    lines.append("| literal | truth arm (DECISIONS) | production arm (DECISIONS) |")
    lines.append("|---|---|---|")
    literals = sorted(
        set(summary.get("truth_fallback_literals") or {})
        | set(summary.get("production_fallback_literals") or {})
    )
    for literal in literals:
        lines.append(
            f"| `{literal}` | {(summary.get('truth_fallback_literals') or {}).get(literal, 0)} "
            f"| {(summary.get('production_fallback_literals') or {}).get(literal, 0)} |"
        )
    if not literals:
        lines.append("| *(none)* | 0 | 0 |")
    lines.append("")
    lines.append("## Sampler-search failure (the isolated residual)")
    lines.append("")
    lines.append(
        "The truth constructs, but belief sampling found NO consistent completion in "
        "`worlds * sample_retry_factor` tries. A conditioning problem, not a guard "
        "problem. Never folded into the truth-rejection rate."
    )
    lines.append("")
    lines.append("| sampling failure reason | WORLDS |")
    lines.append("|---|---|")
    for name, count in sorted(
        (summary.get("sampler_search_failure_reasons") or {}).items(),
        key=lambda item: -item[1],
    ):
        lines.append(f"| `{name}` | {count} |")
    if not summary.get("sampler_search_failure_reasons"):
        lines.append("| *(none)* | 0 |")
    lines.append("")
    lines.append("## Identity witness (per shard, from the LOADED module)")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            [
                {
                    "path": shard["path"],
                    "witness": shard.get("identity_witness"),
                    "mismatches_vs_neutral_cwd_child": shard.get(
                        "identity_witness_mismatches"
                    ),
                }
                for shard in (summary.get("shards") or [])[:2]
            ],
            indent=1,
            sort_keys=True,
        )
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _pct(value: Any) -> str:
    if value is None:
        return "UNMEASURED (no denominator)"
    return f"{value:.4%}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default="run", choices=("plan", "run", "report", "queue", "witness")
    )
    parser.add_argument("--showdown-root", default=None)
    parser.add_argument("--out", default="-")
    # plan
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=9_800_000)
    # run
    parser.add_argument("--plan")
    parser.add_argument("--shard", help="i/N")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--limit-decisions", type=int, default=0)
    parser.add_argument("--control-games", type=int, default=0,
                        help="plain reset(seed=) randbat games, no start override")
    parser.add_argument("--control-seed-base", type=int, default=9_900_000)
    parser.add_argument("--tag", default="tdc")
    parser.add_argument("--max-rounds", type=int, default=250)
    parser.add_argument("--dump-every", type=int, default=5)
    parser.add_argument("--model-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tables")
    parser.add_argument("--driver-leaf-eval", default="hp_fraction_crate",
                        choices=("hp_fraction_crate", "model"))
    parser.add_argument("--driver-worlds", type=int, default=8)
    parser.add_argument("--driver-sims", type=int, default=256)
    parser.add_argument("--driver-batch", type=int, default=16)
    parser.add_argument("--driver-depth", type=int, default=4)
    parser.add_argument("--sample-retry-factor", type=int, default=4)
    parser.add_argument("--truth-sims", type=int, default=64)
    parser.add_argument("--truth-batch", type=int, default=16)
    parser.add_argument("--truth-depth", type=int, default=4)
    parser.add_argument("--truth-repeats", type=int, default=1)
    parser.add_argument("--force", default="none",
                        choices=("none", "construct", "abort", "unmapped"))
    # report / queue
    parser.add_argument("--shards", nargs="*", default=[])
    parser.add_argument("--summary", help="a --mode report output, for --mode queue")
    parser.add_argument("--queue-title", default="Truth-injection differential census")
    parser.add_argument("--command", action="append", default=[],
                        help="a command line to publish beside the figures; repeatable")
    args = parser.parse_args(argv)
    if args.showdown_root is None:
        args.showdown_root = _default_showdown_root()

    if args.mode == "witness":
        from pokezero.truth_differential import identity_witness

        payload = {
            "in_process": identity_witness(),
            "child_neutral_cwd": _neutral_cwd_witness(os.environ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.mode == "plan":
        from pokezero.randbat import load_gen3_randbat_source_cached

        set_source = load_gen3_randbat_source_cached(args.showdown_root)
        plan = build_plan(
            set_source=set_source,
            passes=args.passes,
            seed_base=args.seed_base,
            showdown_root=args.showdown_root,
        )
        text = json.dumps(plan, indent=1, sort_keys=True)
        if args.out == "-":
            print(json.dumps({k: v for k, v in plan.items()
                              if k not in ("games", "variant_games")}, indent=2))
        else:
            Path(args.out).write_text(text, encoding="utf-8")
            print(json.dumps({k: v for k, v in plan.items()
                              if k not in ("games", "variant_games")}, indent=2))
            print(f"wrote {args.out} ({len(plan['games'])} games)")
        return 0

    if args.mode == "report":
        summary = merge_shards(Path(path) for path in args.shards)
        text = json.dumps(summary, indent=1, sort_keys=True)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        return 0

    if args.mode == "queue":
        if args.summary:
            summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        else:
            summary = merge_shards(Path(path) for path in args.shards)
        text = render_queue(summary, commands=args.command, title=args.queue_title)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        return 0

    for required in ("model_path", "checkpoint", "tables"):
        if not getattr(args, required):
            raise SystemExit(f"--{required.replace('_', '-')} is required in run mode")
    if args.out == "-":
        raise SystemExit("--out is required in run mode")
    return run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
