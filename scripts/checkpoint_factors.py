"""Track how a checkpoint's strategy evolves — a suite of behavioral factors scored against a
shared, fixed corpus of real decision states. Re-run on each new checkpoint; watch the factors.

  python scripts/checkpoint_factors.py \
      --checkpoint checkpoints/curated/current-v2-500k.pt=v2-500k \
      --checkpoint checkpoints/curated/current-v2-600k.pt=v2-600k \
      --showdown-root /Users/scott/workspace/pokerena/vendor/pokemon-showdown \
      --num-games 60 --out runs/probes/factors-<date>.json

Factors: switch_propensity, toxic_switch (counterfactual lift), setup_usage (Dragon Dance /
Calm Mind / Swords Dance / Bulk Up, split by board pressure). See pokezero.checkpoint_factors.
Keep --num-games / --seed-start fixed across runs for comparable longitudinal numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokezero.checkpoint_factors import (
    DEFAULT_SETUP_MOVES,
    build_corpus,
    evaluate_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH[=LABEL]")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--num-games", type=int, default=60, help="games to generate the corpus")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=800)
    parser.add_argument("--setup-moves", default=",".join(DEFAULT_SETUP_MOVES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    specs = []
    for raw in args.checkpoint:
        path, _, label = raw.partition("=")
        specs.append((label or Path(path).stem, path))

    setup_moves = tuple(s.strip() for s in args.setup_moves.split(",") if s.strip())
    print(f"[corpus] generating from {args.num_games} games (seed {args.seed_start}+), fixed heuristic drivers…")
    corpus = build_corpus(
        args.showdown_root,
        num_games=args.num_games,
        seed_start=args.seed_start,
        max_states=args.max_states,
        setup_moves=setup_moves,
    )
    n_switch = sum(1 for e in corpus if e.legal_switch)
    n_setup = sum(1 for e in corpus if e.setup_slots)
    n_toxic = sum(1 for e in corpus if e.poison_susceptible and e.legal_switch and e.active_status == "none")
    print(
        f"[corpus] {len(corpus)} decision states — {n_switch} with a legal switch, "
        f"{n_setup} with a legal setup move, {n_toxic} eligible for the toxic counterfactual"
    )
    if n_setup == 0:
        print("[corpus] WARNING: no setup-move states captured; raise --num-games for setup coverage")

    results = [evaluate_checkpoint(label, path, args.showdown_root, corpus) for label, path in specs]

    def row(label, *cols):
        print(f"  {label:<22}" + "".join(f"{c:>14}" for c in cols))

    print("\n=== switch propensity (over legal-switch states) ===")
    row("checkpoint", "mean P(sw)", "argmax rate")
    for r in results:
        s = r["switch_propensity"]
        row(r["label"], f"{s['mean_p_switch']:.3f}", f"{s['argmax_switch_rate']:.3f}")

    print("\n=== toxic switching (P(switch) lift when badly poisoned) ===")
    row("checkpoint", "mean lift", "n")
    for r in results:
        t = r["toxic_switch"]
        row(r["label"], f"{t['mean_lift']:+.3f}", str(t["n_states"]))

    print("\n=== setup usage (prob mass on Dragon Dance / Calm Mind / Swords Dance / Bulk Up) ===")
    row("checkpoint", "healthy", "pressured", "argmax rate")
    for r in results:
        u = r["setup_usage"]
        row(r["label"], f"{u['mean_mass_healthy']:.3f}", f"{u['mean_mass_pressured']:.3f}", f"{u['argmax_setup_rate']:.3f}")
    print("  (a well-timed policy sets up MORE when healthy than when pressured)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_games": args.num_games,
            "seed_start": args.seed_start,
            "setup_moves": list(setup_moves),
            "corpus_states": len(corpus),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\n[out] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
