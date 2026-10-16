"""Live root-prior check: main's incumbent engine vs the raw policy on regenerated roots.

Replays seeded raw-vs-raw source games (same construction as scripts/regen_root_corpus.py) and,
at every p1 decision, also runs the production EngineMctsPolicy (model priors, model leaf) under
three world variants without touching the game:

  sampled4  belief-sampled worlds, worlds=4 (the exploration collector's configuration)
  sampled1  belief-sampled worlds, worlds=1
  oracle    fixed override with the game's TRUE packed teams (no belief sampling)

For each variant it records the native root prior over legal actions (root_allocation.arms) and
compares it with the raw policy's masked distribution: argmax agreement, total variation, KL.
If `oracle` agrees with the raw policy but `sampled*` do not, the root encoding depends on the
sampled hidden state. If `oracle` also disagrees, the live path differs from the parity gate.

Exit status is the gate: 0 only if every measured decision agrees on the argmax and has total
variation at or below `--gate-tv` under every variant, every variant measured at least one
decision, and no variant errored. The default of 0.1 sits above main's own numeric residual
(at most 0.09, same argmax) and well below the collector pattern this gate exists to catch
(median 0.42 on disagreement roots). On main the reference configuration fails the default
gate only on the three game-1612591941 rows of the move-surface class. A root with a single
legal action has no choice and counts as exact. `--world-rng per-decision` (default) reseeds the
belief sampler with the game seed at every boundary, so a run is reproducible; `per-game` draws
fresh worlds at every boundary from one per-game stream, closer to a live collector.

Usage:
    PYTHONPATH=src python scripts/live_root_prior_check.py --checkpoint <ckpt.pt> --model-ts <model_ts.pt> \
        --tables <encoder_tables.json> --showdown-root <showdown> --seeds S1 S2 ... --out report.json [--sims 64]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokezero.collection import env_config_with_policy_spec_masks  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy, EnvTier2AnnotationSource  # noqa: E402
from pokezero.env import BattleStartOverride  # noqa: E402
from pokezero.golden_corpus import _CapturingPolicy, _true_teams_from_bridge_snapshot  # noqa: E402
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


def gate_verdict(rows: list[dict[str, Any]], variants: list[str], gate_tv: float) -> dict[str, Any]:
    """Summarize rows against the gate.

    `passed` is False on any error, any argmax disagreement, any TV above gate_tv, or a variant
    that measured nothing (a vacuous pass is not a pass).
    """
    out: dict[str, Any] = {"gate_tv": gate_tv, "rows": len(rows), "variants": {}, "passed": True}
    for v in variants:
        measured = [r for r in rows if "tv" in r["variants"].get(v, {})]
        errors = [r for r in rows if "error" in r["variants"].get(v, {})]
        over = [r for r in measured if r["variants"][v]["tv"] > gate_tv]
        disagree = [r for r in measured if not r["variants"][v]["agree"]]
        out["variants"][v] = {"measured": len(measured), "agree": len(measured) - len(disagree),
                              "over_gate": len(over), "errors": len(errors),
                              "max_tv": max((r["variants"][v]["tv"] for r in measured), default=0.0)}
        if over or errors or disagree or not measured:
            out["passed"] = False
    return out


def compare(py: dict[int, float], nat: dict[int, float]) -> dict[str, Any]:
    z = sum(nat.values()) or 1.0
    nat = {a: v / z for a, v in nat.items()}
    pa = max(py, key=py.get)
    na = max(nat, key=nat.get) if nat else None
    tv = 0.5 * sum(abs(py.get(a, 0.0) - nat.get(a, 0.0)) for a in set(py) | set(nat))
    kl = sum(p * math.log(p / max(nat.get(a, 0.0), 1e-9)) for a, p in py.items() if p > 1e-12)
    return dict(python_argmax=pa, native_argmax=na, agree=(pa == na), tv=round(tv, 4), kl=round(kl, 4),
                native_prior={str(a): round(v, 5) for a, v in sorted(nat.items())},
                python_prior={str(a): round(v, 5) for a, v in sorted(py.items())})


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
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--variants", nargs="+", default=["sampled4", "sampled1", "oracle"])
    parser.add_argument("--dump-first-metadata", action="store_true")
    parser.add_argument("--gate-tv", type=float, default=0.1, help="exit 1 if any decision exceeds this total variation or flips the argmax")
    parser.add_argument("--world-rng", choices=("per-decision", "per-game"), default="per-decision")
    args = parser.parse_args(argv)

    base = LocalShowdownConfig(showdown_root=args.showdown_root, set_belief_source=True)
    config = env_config_with_policy_spec_masks(base, ["neural:" + str(args.checkpoint)], context="live_root_prior_check")
    env = LocalShowdownEnv(config)
    showdown_root = config.resolved_showdown_root()
    dex = load_showdown_dex_cached(showdown_root)
    set_source = load_gen3_randbat_source_cached(showdown_root)
    annotation_source = EnvTier2AnnotationSource(env)

    def engine(worlds: int, fixed_override=None, tag: str = "") -> EngineMctsPolicy:
        policy = EngineMctsPolicy(
            dex=dex, set_source=set_source, policy_id=f"live-check-{tag}",
            fixed_override=fixed_override, annotation_source=annotation_source,
            config=EngineMctsConfig(
                worlds=worlds, search_time_ms=5000, threads=1, leaf_eval="model",
                model_path=str(args.model_ts), checkpoint_path=str(args.checkpoint), model_device="cpu",
                tables_path=str(args.tables), search_sims=args.sims, search_batch=16, search_depth=6,
                c_puct=1.4, deep_ko_split=True, model_priors=True, use_opponent_priors=False,
                override_telemetry=True, early_stop=False, model_world_workers=1,
            ),
        )
        policy.warm_model_runtime()
        return policy

    rows: list[dict[str, Any]] = []
    dumped = {"done": False}
    try:
        for seed in args.seeds:
            battle_id = f"live-check:{args.format_id}:{seed}"
            env.reset(seed=seed, format_id=args.format_id)
            true_teams = _true_teams_from_bridge_snapshot(env.snapshot().bridge_snapshot)
            packed = {slot: true_teams[slot]["packed"] for slot in ("p1", "p2")}
            engines: dict[str, EngineMctsPolicy] = {}
            if "sampled4" in args.variants:
                engines["sampled4"] = engine(4, tag="s4")
            if "sampled1" in args.variants:
                engines["sampled1"] = engine(1, tag="s1")
            if "oracle" in args.variants:
                engines["oracle"] = engine(1, fixed_override=BattleStartOverride(player_teams=packed), tag="oracle")
            raw = {seat: load_transformer_policy(args.checkpoint, device="cpu", deterministic=True, exploration_epsilon=0.0,
                                                 sampling_temperature=1.0, family_gated_selection=False) for seat in ("p1", "p2")}
            for p in raw.values():
                p.record_policy_distribution = True

            game_rngs = {name: random.Random(seed) for name in engines}

            def _sink(context, decision, seed=seed, engines=engines, game_rngs=game_rngs):
                if context.player_id != "p1":
                    return
                dist = decision.metadata.get("policy_distribution")
                legal = [i for i, ok in enumerate(context.observation.legal_action_mask) if ok]
                py = {a: float(dist[a]) for a in legal}
                z = sum(py.values()) or 1.0
                py = {a: v / z for a, v in py.items()}
                row = dict(seed=seed, round=context.decision_round_index, turn=context.observation.metadata.get("turn_number"),
                           raw_action=decision.action_index, legal=legal, variants={})
                for name, pol in engines.items():
                    started = time.perf_counter()
                    try:
                        rng = game_rngs[name] if args.world_rng == "per-game" else random.Random(seed)
                        d = pol.select_action_with_context(context, rng=rng)
                    except Exception as error:  # noqa: BLE001
                        row["variants"][name] = dict(error=f"{type(error).__name__}: {str(error)[:200]}")
                        continue
                    md = dict(d.metadata)
                    if args.dump_first_metadata and not dumped["done"]:
                        dumped["done"] = True
                        print("FIRST ENGINE METADATA KEYS:", list(md.keys()), file=sys.stderr)
                        em = md.get("engine_mcts") or {}
                        print("engine_mcts keys:", list(em.keys()) if isinstance(em, dict) else type(em), file=sys.stderr)
                        ov = find_key(md, "override")
                        print("override keys:", list(ov.keys()) if isinstance(ov, dict) else ov, file=sys.stderr)
                    arms = find_key(md, "root_allocation")
                    arms = (arms or {}).get("arms") if isinstance(arms, dict) else None
                    if not arms:
                        row["variants"][name] = dict(error="no root_allocation arms in metadata", keys=list(md.keys())[:12])
                        continue
                    unmapped = sum(1 for a in arms if a.get("action_index") is None)
                    nat = {int(a["action_index"]): float(a.get("model_prior", 0.0)) for a in arms if a.get("action_index") is not None}
                    if not nat:
                        if len(legal) == 1:  # no choice to make; the native side reports no mappable arm
                            row["variants"][name] = dict(python_argmax=legal[0], native_argmax=legal[0], agree=True, tv=0.0, kl=0.0,
                                                         forced_single_action=True, unmapped_arms=unmapped)
                        else:
                            row["variants"][name] = dict(error="every arm lacks an action_index", unmapped_arms=unmapped)
                        continue
                    out = compare(py, nat)
                    if unmapped:
                        out["unmapped_arms"] = unmapped
                    out.update(engine_action=d.action_index, model_argmax=find_key(md, "model_argmax"),
                               search_argmax=find_key(md, "search_argmax"), seconds=round(time.perf_counter() - started, 3))
                    row["variants"][name] = out
                rows.append(row)

            policies = {seat: _CapturingPolicy(raw[seat], _sink) for seat in ("p1", "p2")}
            result = continue_rollout_from_current_state(
                env=env, policies=policies,
                config=RolloutConfig(max_decision_rounds=args.max_decision_rounds, format_id=args.format_id,
                                     hide_opponent_legal_action_masks=True),
                seed=seed, battle_id=battle_id, reset_policies=True,
            )
            n = sum(1 for r in rows if r["seed"] == seed)
            summary = {v: sum(1 for r in rows if r["seed"] == seed and r["variants"].get(v, {}).get("agree")) for v in engines}
            errors = {v: sum(1 for r in rows if r["seed"] == seed and "error" in r["variants"].get(v, {})) for v in engines}
            print(json.dumps(dict(seed=seed, p1_decisions=n, turns=result.terminal.turn_count, agree=summary, errors=errors)), flush=True)
            args.out.write_text(json.dumps(dict(rows=rows, complete=False), indent=1) + "\n")  # incremental, survives a crash
    finally:
        env.close()
    verdict = gate_verdict(rows, list(args.variants), args.gate_tv)
    args.out.write_text(json.dumps(dict(rows=rows, complete=True, gate=verdict, world_rng=args.world_rng), indent=1) + "\n")
    print("== gate:", json.dumps(verdict))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
