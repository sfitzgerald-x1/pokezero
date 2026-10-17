"""Search deviation scan: how often, and how confidently, does the model-leaf search leave raw's action as the budget grows?

Replays seeded raw-versus-raw source games (same construction as scripts/regen_root_corpus.py) and at every p1
boundary runs main's production EngineMctsPolicy (model priors, model leaf, four belief worlds) at several
per-world iteration budgets, recording for each: raw's action, the search's action, the native prior argmax,
the search argmax, the root Q gap and visit gap, and the worlds searched. No continuations are played;
on roots where the search keeps raw's action the paired contrast is identically zero, so this scan is the
cheap half of a budget sweep and the sizing input for a deviation-triggered panel.

Usage:
    PYTHONPATH=src python scripts/search_deviation_scan.py --checkpoint <ckpt.pt> --model-ts <model_ts.pt> \
        --tables <encoder_tables.json> --showdown-root <showdown> --seeds S1 S2 ... --out report.json \
        [--sims-per-world 80 256 1024] [--worlds 4] [--max-decisions-per-game N]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.collection import env_config_with_policy_spec_masks  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy, EnvTier2AnnotationSource  # noqa: E402
from pokezero.golden_corpus import _CapturingPolicy  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.neural_policy import load_transformer_policy  # noqa: E402
from pokezero.randbat import load_gen3_randbat_source_cached  # noqa: E402
from pokezero.rollout import RolloutConfig, continue_rollout_from_current_state  # noqa: E402


def find_key(obj: Any, key: str, depth: int = 0):
    if depth > 8:
        return None
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = find_key(v, key, depth + 1)
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = find_key(v, key, depth + 1)
            if hit is not None:
                return hit
    return None


def summarize(rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in budgets:
        key = str(b)
        have = [r for r in rows if "action" in r["budgets"].get(key, {})]
        dev = [r for r in have if r["budgets"][key]["action"] != r["raw_action"]]
        prior_mismatch = [r for r in have if r["budgets"][key].get("model_argmax") not in (None, r["raw_action"])]
        override = [r for r in have if r["budgets"][key].get("search_argmax") is not None
                    and r["budgets"][key].get("model_argmax") is not None
                    and r["budgets"][key]["search_argmax"] != r["budgets"][key]["model_argmax"]]
        qg = [r["budgets"][key]["root_q_gap"] for r in dev if isinstance(r["budgets"][key].get("root_q_gap"), (int, float))]
        secs = [r["budgets"][key]["seconds"] for r in have]
        out[key] = {"decisions": len(have), "deviations": len(dev), "deviation_rate": (len(dev) / len(have)) if have else None,
                    "prior_argmax_mismatch": len(prior_mismatch), "tree_overrides": len(override),
                    "deviation_root_q_gap_median": (statistics.median(qg) if qg else None),
                    "seconds_median": (statistics.median(secs) if secs else None), "errors": sum(1 for r in rows if "error" in r["budgets"].get(key, {}))}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-ts", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format-id", default="gen3randombattle")
    parser.add_argument("--max-decision-rounds", type=int, default=250)
    parser.add_argument("--sims-per-world", type=int, nargs="+", default=[80, 256, 1024])
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-decisions-per-game", type=int, default=None, help="probe mode: stop recording after N p1 decisions")
    args = parser.parse_args(argv)

    base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    config = env_config_with_policy_spec_masks(base, ["neural:" + str(args.checkpoint)], context="search_deviation_scan")
    env = LocalShowdownEnv(config)
    showdown_root = config.resolved_showdown_root()
    dex = load_showdown_dex_cached(showdown_root)
    set_source = load_gen3_randbat_source_cached(showdown_root)
    annotation_source = EnvTier2AnnotationSource(env)

    def engine(sims: int) -> EngineMctsPolicy:
        policy = EngineMctsPolicy(
            dex=dex, set_source=set_source, policy_id=f"scan-{sims}", annotation_source=annotation_source,
            config=EngineMctsConfig(
                worlds=args.worlds, search_time_ms=600000, threads=1, leaf_eval="model",
                model_path=str(args.model_ts), checkpoint_path=str(args.checkpoint), model_device="cpu",
                tables_path=str(args.tables), search_sims=sims, search_batch=args.batch, search_depth=args.depth,
                c_puct=1.4, deep_ko_split=True, model_priors=True, use_opponent_priors=False,
                override_telemetry=True, early_stop=False, model_world_workers=1,
            ),
        )
        policy.warm_model_runtime()
        return policy

    rows: list[dict[str, Any]] = []
    try:
        for seed in args.seeds:
            battle_id = f"deviation-scan:{args.format_id}:{seed}"
            env.reset(seed=seed, format_id=args.format_id)
            engines = {sims: engine(sims) for sims in args.sims_per_world}
            raw = {seat: load_transformer_policy(args.checkpoint, device="cpu", deterministic=True, exploration_epsilon=0.0,
                                                 sampling_temperature=1.0, family_gated_selection=False) for seat in ("p1", "p2")}
            for p in raw.values():
                p.record_policy_distribution = True
            counted = {"n": 0}

            def _sink(context, decision, seed=seed, engines=engines, counted=counted):
                if context.player_id != "p1":
                    return
                if args.max_decisions_per_game is not None and counted["n"] >= args.max_decisions_per_game:
                    return
                counted["n"] += 1
                legal = [i for i, ok in enumerate(context.observation.legal_action_mask) if ok]
                dist = decision.metadata.get("policy_distribution") or []
                top2 = sorted((float(dist[a]) for a in legal), reverse=True)[:2] if dist else []
                row = dict(seed=seed, round=context.decision_round_index, turn=context.observation.metadata.get("turn_number"),
                           raw_action=decision.action_index, legal=legal, raw_margin=(top2[0] - top2[1]) if len(top2) == 2 else None, budgets={})
                for sims, pol in engines.items():
                    started = time.perf_counter()
                    try:
                        d = pol.select_action_with_context(context, rng=random.Random(seed))
                    except Exception as error:  # noqa: BLE001
                        row["budgets"][str(sims)] = dict(error=f"{type(error).__name__}: {str(error)[:160]}")
                        continue
                    md = dict(d.metadata); ov = find_key(md, "override") or {}
                    row["budgets"][str(sims)] = dict(
                        action=d.action_index, model_argmax=ov.get("model_argmax"), search_argmax=ov.get("search_argmax"),
                        root_q_gap=ov.get("root_q_gap"), root_visit_gap=ov.get("root_visit_gap"),
                        worlds_searched=find_key(md, "worlds_searched"),
                        seconds=round(time.perf_counter() - started, 3))
                rows.append(row)

            policies = {seat: _CapturingPolicy(raw[seat], _sink) for seat in ("p1", "p2")}
            result = continue_rollout_from_current_state(
                env=env, policies=policies,
                config=RolloutConfig(max_decision_rounds=args.max_decision_rounds, format_id=args.format_id, hide_opponent_legal_action_masks=True),
                seed=seed, battle_id=battle_id, reset_policies=True,
            )
            game_rows = [r for r in rows if r["seed"] == seed]
            print(json.dumps(dict(seed=seed, p1_decisions=len(game_rows), turns=result.terminal.turn_count,
                                  summary={k: {kk: v[kk] for kk in ("decisions", "deviations", "seconds_median")} for k, v in summarize(game_rows, args.sims_per_world).items()})), flush=True)
            args.out.write_text(json.dumps(dict(rows=rows, complete=False, summary=summarize(rows, args.sims_per_world)), indent=1) + "\n")
    finally:
        env.close()
    summary = summarize(rows, args.sims_per_world)
    args.out.write_text(json.dumps(dict(rows=rows, complete=True, summary=summary, worlds=args.worlds, sims_per_world=args.sims_per_world), indent=1) + "\n")
    print("== summary:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
