#!/usr/bin/env python3
"""§8 acceptance re-bench: engine MCTS vs the SAME checkpoint's raw policy.

This is the run that settles §10–§11 of ``docs/mcts_degradation_findings.md``.
It differs from the ad-hoc ``power_h2h.py`` that produced the §4/§9 grids in one
structural way, and that difference is the whole point:

    **Pairing is WITHIN-seed.** Each battle seed is played TWICE — once with
    search seated p1 and once with search seated p2 — so the two seats face the
    *same two teams*. ``power_h2h.py`` derived the seat from seed parity
    (``seed % 2``), which meant the seats held disjoint seed sets and every
    seat comparison was also a team comparison (§11.7's standing weakness).

Everything else follows the in-house conventions rather than reinventing them:
results are emitted as ``pokezero.mcts_eval.scoring.GameResult`` rows, so the
fail-closed merge (a pair missing a seat is an error, never a silently
half-scored row) and the deterministic paired bootstrap apply unchanged.

The engine build gate runs BEFORE the first game and is a hard stop: a stale
build does not error, it produces a plausible number.

Usage (one shard)::

    python scripts/mcts_acceptance_h2h.py \\
        --checkpoint <path> --showdown-root <path> \\
        --arm search --pair-start 7800000 --pairs 10 \\
        --depth 4 --sims 1024 --out shard.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_provenance(checkpoint: str, config_id: str, arm: str, *,
                     vocab_sha256: str, commit: str) -> str:
    """Bind every row to the exact build + cell that produced it.

    ``merge_game_results`` treats a canonical-outcome conflict as terminal, so
    two shards that disagree because they ran different engine builds cannot be
    silently averaged together.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from engine_build_fingerprint import compute_fingerprint  # noqa: PLC0415

    payload = {
        "engine_fingerprint": compute_fingerprint()["fingerprint"],
        "checkpoint": checkpoint,
        "config_id": config_id,
        "arm": arm,
        # The enumeration the encode actually used. Rows produced against a
        # different vocabulary describe different observations, so they must not
        # merge with these -- `merge_game_results` treats a provenance conflict as
        # terminal, which is exactly the protection wanted here.
        "category_vocab_sha256": vocab_sha256,
        "commit": commit,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _policy_stats_payload(stats) -> dict:
    """Policy stats with the depth keys GUARANTEED present.

    Every shard row must carry `search_ran`, `depth_reached_samples` and
    `depth_reached_histogram`, so that a missing key is a real defect rather than
    an ordinary "nothing to report". Without this, an absent histogram and a cap
    that never bound are the same bytes on disk.
    """
    payload = stats.to_payload() if hasattr(stats, "to_payload") else {}
    payload.setdefault("search_ran", True)
    payload.setdefault("depth_reached_samples", 0)
    payload.setdefault("depth_reached_max", 0)
    payload.setdefault("depth_reached_histogram", {})
    return payload


def outcome_for(winner: str | None, seat: str, capped: bool) -> str:
    if capped:
        return "cap"
    if winner is None:
        return "tie"
    return "win" if winner == seat else "loss"


def build_policies(checkpoint, showdown_root, device, cfg, annotation_source):
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
    from pokezero.mcts_eval.lattice import materialize_search_artifacts
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.randbat import load_gen3_randbat_source_cached

    contract = resolve_checkpoint_contract(
        checkpoint, model_device=device, showdown_root=showdown_root
    )
    artifacts = materialize_search_artifacts(contract, showdown_root=showdown_root)
    print(f"artifacts: {artifacts}", flush=True)
    raw = load_transformer_policy(checkpoint, device=device, deterministic=True)
    search = EngineMctsPolicy(
        dex=load_showdown_dex_cached(showdown_root),
        set_source=load_gen3_randbat_source_cached(showdown_root),
        config=EngineMctsConfig(
            leaf_eval="model",
            checkpoint_path=checkpoint,
            model_path=artifacts["model_path"],
            tables_path=artifacts["tables_path"],
            model_device=device,
            **cfg,
        ),
        policy_id="search",
        annotation_source=annotation_source,
    )
    return search, raw



def _source_commit() -> str:
    """The pokezero commit this container was built from.

    Images do not carry .git, so the build stamps POKEZERO_COMMIT; fall back to a
    local checkout when running outside a container, and never silently report
    "unknown" as if it were a SHA.
    """
    import os
    import subprocess

    stamped = os.environ.get("POKEZERO_COMMIT")
    if stamped:
        return stamped
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        raise SystemExit(
            "cannot determine the source commit: set POKEZERO_COMMIT in the job spec"
        )


def checkpoint_category_vocabulary(model_config, showdown_root: str):
    """The vocabulary the ROOT encode must use: the checkpoint's own.

    `LocalShowdownConfig.category_vocab` defaults to None and the env then builds the
    enumeration from the showdown root (local_showdown.py:505), which is the build's,
    not the checkpoint's. `env_config_with_checkpoint_masks` latches the mask and spec
    axes from checkpoint provenance but not this one, so nothing supplies it unless a
    caller does. Post-#948 the LEAF tables speak the checkpoint's enumeration, so
    leaving the root on the build's would put the two sides of one tree 13 volatile
    rows apart. Anchor it here until the root-binding lane lands.

    Alias and OOV handling come from the build's vocabulary object and only the token
    list is swapped, which is exactly what the exporter does for the leaf tables, so
    both sides resolve strings the same way.
    """
    from dataclasses import replace as _replace

    from pokezero.observation import TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS
    from pokezero.neural_policy import observation_spec_from_model_config
    from pokezero.randbat_vocab import gen3_category_vocabulary

    spec = observation_spec_from_model_config(model_config)
    base = gen3_category_vocabulary(
        showdown_root,
        include_turn_merged=spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    )
    trained = tuple(str(t) for t in (getattr(model_config, "category_vocab", ()) or ()))
    if not trained:
        raise SystemExit(
            "checkpoint carries no category_vocab; refusing to guess the enumeration"
        )
    return _replace(base, tokens=trained)


def assert_vocab_alignment(model_config, env_config, tables_path) -> str:
    """Hard stop before the first game: root, checkpoint and leaf must agree.

    Same standing as the engine fingerprint gate, and for the same reason -- a
    misaligned enumeration does not error, it produces a plausible number. Returns
    the vocabulary digest so it can be stamped into every row's provenance.
    """
    import hashlib
    import json as _json

    checkpoint_tokens = tuple(str(t) for t in (model_config.category_vocab or ()))
    root_vocab = env_config.category_vocab
    if root_vocab is None:
        raise SystemExit(
            "root encode would build its own vocabulary from the showdown root; "
            "the env config must carry the checkpoint's."
        )
    root_tokens = tuple(str(t) for t in root_vocab.tokens)
    if root_tokens != checkpoint_tokens:
        # Report the first divergence, exactly as the leaf branch below does.
        # Equal lengths with a different enumeration ("1216 vs 1216") is the
        # hardest case to diagnose from a pod log and the one most likely to be
        # waved past; the index names the token that moved.
        first = next(
            (i for i, (a, b) in enumerate(zip(root_tokens, checkpoint_tokens)) if a != b),
            None,
        )
        detail = (
            f", first divergence at {first} "
            f"(root {root_tokens[first]!r} vs checkpoint {checkpoint_tokens[first]!r})"
            if first is not None
            else " (one is a prefix of the other)"
        )
        raise SystemExit(
            f"root vocabulary != checkpoint: {len(root_tokens)} vs "
            f"{len(checkpoint_tokens)} tokens{detail}"
        )
    if tables_path is not None:
        leaf_tokens = tuple(
            str(t) for t in _json.loads(
                Path(tables_path).read_text(encoding="utf-8")
            )["vocab"]["tokens"]
        )
        if leaf_tokens != checkpoint_tokens:
            first = next(
                (i for i, (a, b) in enumerate(zip(leaf_tokens, checkpoint_tokens)) if a != b),
                None,
            )
            raise SystemExit(
                f"leaf tables vocabulary != checkpoint: {len(leaf_tokens)} vs "
                f"{len(checkpoint_tokens)} tokens, first divergence at {first}"
            )
    digest = hashlib.sha256("\n".join(checkpoint_tokens).encode("utf-8")).hexdigest()
    print(
        f"vocab gate OK: root == checkpoint"
        f"{' == leaf tables' if tables_path is not None else ''} "
        f"({len(checkpoint_tokens)} tokens, sha256 {digest[:16]})",
        flush=True,
    )
    return digest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", choices=("search", "control"), default="search")
    ap.add_argument("--pair-start", type=int, required=True,
                    help="first BATTLE seed; each seed is played from both seats")
    ap.add_argument("--pairs", type=int, required=True,
                    help="number of battle seeds (games = 2 x pairs)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sims", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-build-check", action="store_true",
                    help="offline/dry use only; never for a scored shard")
    args = ap.parse_args(argv)

    # HARD STOP before any engine call.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from engine_build_fingerprint import assert_fresh, compute_fingerprint  # noqa: PLC0415

    assert_fresh(skip=args.skip_build_check)
    engine_fingerprint = compute_fingerprint()["fingerprint"]
    print(f"engine fingerprint: {engine_fingerprint}", flush=True)

    from pokezero.engine_search import EnvTier2AnnotationSource
    from pokezero.local_showdown import (
        LocalShowdownConfig,
        LocalShowdownEnv,
        env_config_with_checkpoint_masks,
    )
    from pokezero.neural_policy import (
        feature_masks_from_model_config,
        load_transformer_model_config,
        observation_spec_from_model_config,
    )
    from pokezero.rollout import RolloutConfig, RolloutDriver

    config_id = f"d{args.depth}-s{args.sims}-b{args.batch}-w{args.worlds}"
    if args.arm == "control":
        config_id = "control-raw-v-raw"

    cfg = dict(
        search_depth=args.depth,
        search_sims=args.sims,
        search_batch=args.batch,
        worlds=args.worlds,
    )
    model_config = load_transformer_model_config(args.checkpoint)
    env_config = env_config_with_checkpoint_masks(
        LocalShowdownConfig(
            showdown_root=args.showdown_root,
            set_belief_source=True,
            # Both arms observe through this one env, so anchoring it here covers
            # the raw opponent and the search root together.
            category_vocab=checkpoint_category_vocabulary(model_config, args.showdown_root),
        ),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        context="mcts acceptance re-bench",
    )

    # Resolve the leaf artifacts up front so the gate can see them. Materialization
    # is keyed and idempotent, so build_policies below reuses this exact export.
    leaf_tables = None
    if args.arm != "control":
        from pokezero.mcts_eval.lattice import materialize_search_artifacts
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract

        leaf_tables = materialize_search_artifacts(
            resolve_checkpoint_contract(
                args.checkpoint, model_device=args.device, showdown_root=args.showdown_root
            ),
            showdown_root=args.showdown_root,
        )["tables_path"]

    vocab_sha256 = assert_vocab_alignment(model_config, env_config, leaf_tables)
    provenance = build_provenance(
        args.checkpoint, config_id, args.arm,
        vocab_sha256=vocab_sha256, commit=_source_commit(),
    )
    env = LocalShowdownEnv(env_config)
    search, raw = build_policies(
        args.checkpoint, args.showdown_root, args.device, cfg,
        EnvTier2AnnotationSource(env),
    )
    if args.arm == "control":
        # Raw vs raw on the SAME seeds. Any deviation from 0.500 here is harness
        # bias and would masquerade as a search effect. A second deterministic
        # instance of the same checkpoint, so the arms differ only by search.
        from pokezero.neural_policy import load_transformer_policy as _load

        search = _load(args.checkpoint, device=args.device, deterministic=True)

        class _NoStats:
            decisions = fallback_decisions = 0
            fallback_reasons: dict = {}
            world_failure_reasons: dict = {}

            def to_payload(self) -> dict:
                """Explicit zeros, never an empty dict.

                A bare ``{}`` here would make three different situations
                serialize identically: "this arm ran no search", "search ran but
                reported no depth", and "the field was never emitted at all". A
                depth ladder is read off exactly these counters, so `no data` and
                `the cap never bound` must not collapse into one shape. The
                control arm therefore states its own emptiness.
                """
                return {
                    "search_ran": False,
                    "depth_reached_samples": 0,
                    "depth_reached_max": 0,
                    "depth_reached_histogram": {},
                }

        search.stats = _NoStats()

    results = []
    per_game = []
    search_decisions = 0
    search_wall = 0.0
    started = time.perf_counter()
    try:
        for index in range(args.pairs):
            pair_seed = args.pair_start + index
            # Both seats of the SAME battle seed: identical teams, swapped sides.
            for seat in ("p1", "p2"):
                other = "p2" if seat == "p1" else "p1"
                d0 = search.stats.decisions
                f0 = search.stats.fallback_decisions
                wf0 = Counter(search.stats.world_failure_reasons)
                t0 = time.perf_counter()
                driver = RolloutDriver(
                    env=env,
                    policies={seat: search, other: raw},
                    config=RolloutConfig(format_id="gen3randombattle"),
                )
                result = driver.run(
                    seed=pair_seed, battle_id=f"accept-{args.arm}-{pair_seed}-{seat}"
                )
                elapsed = time.perf_counter() - t0
                decisions = search.stats.decisions - d0
                fallbacks = search.stats.fallback_decisions - f0
                failures = Counter(search.stats.world_failure_reasons) - wf0
                search_decisions += decisions
                search_wall += elapsed
                terminal = result.terminal
                outcome = outcome_for(terminal.winner, seat, bool(terminal.capped))
                results.append(
                    {
                        "config_id": config_id,
                        "seed": pair_seed,
                        "seat": seat,
                        "outcome": outcome,
                        "turns": int(getattr(terminal, "turn_count", 0) or 0),
                        "provenance_sha256": provenance,
                        "opponent_crashed": False,
                    }
                )
                per_game.append(
                    {
                        "seed": pair_seed,
                        "seat": seat,
                        "winner": terminal.winner,
                        "outcome": outcome,
                        "decisions": decisions,
                        "fallbacks": fallbacks,
                        "world_failures": dict(failures),
                        "wall_s": round(elapsed, 3),
                    }
                )
                print(
                    f"[pair {index + 1}/{args.pairs}] seed={pair_seed} seat={seat} "
                    f"outcome={outcome} dec={decisions} fb={fallbacks}",
                    flush=True,
                )
    finally:
        env.close()

    report = {
        "schema_version": "pokezero.mcts-acceptance-shard.v1",
        "arm": args.arm,
        "config_id": config_id,
        "checkpoint": args.checkpoint,
        "engine_fingerprint": engine_fingerprint,
        "provenance_sha256": provenance,
        "pair_start": args.pair_start,
        "pairs": args.pairs,
        "games": len(results),
        "total_decisions": search_decisions,
        "fallback_decisions": search.stats.fallback_decisions,
        "fallback_rate": round(
            search.stats.fallback_decisions / max(1, search.stats.decisions), 6
        ),
        "fallback_reasons": dict(search.stats.fallback_reasons),
        "world_failure_reasons": dict(search.stats.world_failure_reasons),
        "search_wall_per_decision": round(search_wall / max(1, search_decisions), 4),
        # Full policy stats payload, which carries the reached-depth aggregates
        # (depth_reached_samples / _sum / _max / _histogram). A depth ladder is
        # uninterpretable without them: whether the CAP was binding is the
        # difference between "depth does not help" and "the sims budget never let
        # the tree reach the cap".
        "policy_stats": _policy_stats_payload(search.stats),
        "wall_s": round(time.perf_counter() - started, 1),
        "results": results,
        "per_game": per_game,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SHARD COMPLETE ===")
    print(json.dumps(
        {k: v for k, v in report.items() if k not in ("results", "per_game")}, indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
