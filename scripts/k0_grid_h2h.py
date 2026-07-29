"""k0 depth-grid harness: engine MCTS vs the SAME checkpoint's raw policy.

A variant of the search-power harness that takes the leaf encoder tables as an
EXPLICIT argument instead of materializing them implicitly.

Why: the shipped exporter derived the encoder tables' ``default_feature_masks``
from dataclass defaults rather than from the checkpoint, so the crate's leaf
encode and Python's root encode disagreed about which features exist. Testing
whether that disagreement is what makes search lose to its own prior requires
running the SAME checkpoint against tables that differ only in those mask
fields, which the implicit path cannot express (it is keyed by the contract, so
one checkpoint has exactly one tables file).

Every run prints, and records in its artifact, the masks the tables actually
carry next to the masks the checkpoint actually wants. A disagreement is
recorded as ``mask_drift`` and requires --allow-mask-drift, so a contaminated
arm can never be mistaken for a clean one after the fact.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter


def _table_masks(tables_path):
    layout = json.loads(open(tables_path, encoding="utf-8").read())["layout"]
    return layout.get("default_feature_masks") or {}, layout


def _checkpoint_masks(contract):
    m = dict(contract.feature_masks)
    return {
        "exact_state": m.get("exact_state"),
        "stats_block": m.get("opponent_tendency_stats_block"),
        "tier2_residuals": m.get("tier2_residuals"),
        "tier2_investment": m.get("tier2_investment"),
        "transition_token_budget": min(
            int(m.get("transition_token_budget") or 0), contract.transition_token_count
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--showdown-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=600000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sims", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default="", help="label recorded in the artifact")
    ap.add_argument("--tables-path", default=None,
                    help="explicit encoder tables; default = materialize from the contract")
    ap.add_argument("--model-path", default=None, help="explicit TorchScript artifact")
    ap.add_argument("--allow-mask-drift", action="store_true",
                    help="proceed when the tables' masks disagree with the checkpoint's "
                         "(deliberately-contaminated arms only)")
    ap.add_argument("--control", action="store_true",
                    help="raw vs raw on the same seeds: validates the 0.500 null")
    args = ap.parse_args(argv)

    from pokezero.local_showdown import (
        LocalShowdownConfig, LocalShowdownEnv, env_config_with_checkpoint_masks,
    )
    from pokezero.neural_policy import (
        category_vocab_from_model_config,
        feature_masks_from_model_config, load_transformer_model_config,
        observation_spec_from_model_config, load_transformer_policy,
    )
    from pokezero.rollout import RolloutConfig, RolloutDriver
    from pokezero.engine_search import (
        EngineMctsConfig, EngineMctsPolicy, EnvTier2AnnotationSource,
    )
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.mcts_eval.lattice import materialize_search_artifacts

    model_config = load_transformer_model_config(args.checkpoint)
    env_config = env_config_with_checkpoint_masks(
        LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True),
        feature_masks_from_model_config(model_config),
        required_specs=observation_spec_from_model_config(model_config),
        required_vocabs=category_vocab_from_model_config(model_config, args.showdown_root),
        context="k0 depth grid",
    )
    env = LocalShowdownEnv(env_config)

    contract = resolve_checkpoint_contract(
        args.checkpoint, model_device=args.device, showdown_root=args.showdown_root
    )
    want = _checkpoint_masks(contract)

    provenance = {"arm": args.arm}
    if args.control:
        tables_path = model_path = None
        mask_drift = {}
    else:
        if args.tables_path:
            tables_path = args.tables_path
            model_path = args.model_path
            if model_path is None:
                raise SystemExit("--tables-path requires --model-path")
        else:
            artifacts = materialize_search_artifacts(contract, showdown_root=args.showdown_root)
            tables_path, model_path = artifacts["tables_path"], artifacts["model_path"]
        got, layout = _table_masks(tables_path)
        # Shape must always agree; only the mask fields are allowed to be varied,
        # and only deliberately.
        shape = {
            k: (layout.get(k), v) for k, v in (
                ("schema_version", contract.schema_version),
                ("token_count", contract.token_count),
                ("categorical_feature_count", contract.categorical_feature_count),
                ("numeric_feature_count", contract.numeric_feature_count),
            ) if layout.get(k) != v
        }
        if shape:
            raise SystemExit(f"tables/checkpoint SHAPE disagreement: {shape}")
        mask_drift = {k: {"tables": got.get(k), "checkpoint": v}
                      for k, v in want.items() if got.get(k) != v}
        print("=== LEAF ENCODER CONTRACT ===", flush=True)
        print(f"  tables    : {tables_path}")
        print(f"  masks(tables)    : {got}")
        print(f"  masks(checkpoint): {want}")
        print(f"  mask_drift       : {mask_drift or 'NONE (clean)'}", flush=True)
        if mask_drift and not args.allow_mask_drift:
            raise SystemExit(
                "refusing to run: the leaf encoder describes a different observation "
                "than the checkpoint. Pass --allow-mask-drift only for arms whose "
                "POINT is the contamination."
            )
        provenance.update(
            tables_path=tables_path, model_path=model_path,
            tables_masks=got, checkpoint_masks=want, mask_drift=mask_drift,
        )

    raw = load_transformer_policy(args.checkpoint, device=args.device, deterministic=True)
    if args.control:
        search = load_transformer_policy(args.checkpoint, device=args.device, deterministic=True)
        class _Z:
            decisions = fallback_decisions = 0
            fallback_reasons = {}
            world_failure_reasons = {}
        search.stats = _Z()
    else:
        search = EngineMctsPolicy(
            dex=load_showdown_dex_cached(args.showdown_root),
            set_source=load_gen3_randbat_source_cached(args.showdown_root),
            config=EngineMctsConfig(
                leaf_eval="model",
                checkpoint_path=args.checkpoint,
                model_path=model_path,
                tables_path=tables_path,
                model_device=args.device,
                search_depth=args.depth, search_sims=args.sims,
                search_batch=args.batch, worlds=args.worlds,
            ),
            policy_id="search",
            annotation_source=EnvTier2AnnotationSource(env),
        )

    wins = draws = clean_games = 0
    search_decisions = 0
    search_wall = 0.0
    per_game = []
    try:
        for i in range(args.games):
            seed = args.seed_start + i
            # Seat is a function of the SEED, never the loop index, so shards of
            # different sizes still pair across configs.
            search_seat = "p1" if seed % 2 == 0 else "p2"
            other = "p2" if search_seat == "p1" else "p1"

            d0 = search.stats.decisions
            f0 = search.stats.fallback_decisions
            wf0 = Counter(search.stats.world_failure_reasons)
            t0 = time.perf_counter()

            driver = RolloutDriver(
                env=env,
                policies={search_seat: search, other: raw},
                config=RolloutConfig(format_id="gen3randombattle"),
            )
            result = driver.run(seed=seed, battle_id=f"k0grid-{seed}")
            elapsed = time.perf_counter() - t0

            dec = search.stats.decisions - d0
            fb = search.stats.fallback_decisions - f0
            wf = Counter(search.stats.world_failure_reasons) - wf0
            search_decisions += dec
            search_wall += elapsed
            if fb == 0:
                clean_games += 1

            winner = result.terminal.winner
            if winner == search_seat:
                wins += 1
            elif winner is None:
                draws += 1
            per_game.append({
                "seed": seed, "search_seat": search_seat, "winner": winner,
                "decisions": dec, "fallbacks": fb, "world_failures": dict(wf),
            })
            print(f"[{i+1}/{args.games}] seed={seed} seat={search_seat} "
                  f"winner={winner} dec={dec} fb={fb}", flush=True)
    finally:
        env.close()

    played = args.games
    score = (wins + 0.5 * draws) / played if played else 0.0
    report = {
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "config": f"d{args.depth}-s{args.sims}-b{args.batch}-w{args.worlds}",
        "depth": args.depth, "sims": args.sims, "worlds": args.worlds,
        "seed_start": args.seed_start,
        "games": played,
        "search_wins": wins,
        "draws": draws,
        "score_vs_raw": round(score, 4),
        "clean_games": clean_games,
        "total_decisions": search_decisions,
        "fallback_decisions": search.stats.fallback_decisions,
        "fallback_rate": round(
            search.stats.fallback_decisions / max(1, search.stats.decisions), 6
        ),
        "fallback_reasons": dict(search.stats.fallback_reasons),
        "world_failure_reasons": dict(search.stats.world_failure_reasons),
        "search_wall_per_decision": round(search_wall / max(1, search_decisions), 4),
        "provenance": provenance,
        "per_game": per_game,
    }
    print("\n=== K0 GRID CELL ===")
    print(json.dumps({k: v for k, v in report.items() if k != "per_game"}, indent=2))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
