"""Behavioral collapse-signal probe for foundation checkpoints.

Self-play can degrade into a few characteristic collapse modes (see the design notes on the
foundation runs). This probe measures the *behavioral* signatures that are computable from a
checkpoint alone — no self-play games needed — over the same deterministic corpus the factor
tracker uses (``checkpoint_factors.build_corpus``):

  - policy_entropy        — mean Shannon entropy of the action distribution over LEGAL actions.
                            A steady collapse toward 0 is mode collapse (the policy going
                            deterministic / peaked). Reported with action_perplexity = exp(entropy)
                            (the effective number of actions considered) and near_deterministic_rate
                            (fraction of states where the top action has > 0.9 mass).
  - switch_propensity     — mean P(switch) over switch-legal states (raw behavioral fingerprint;
                            collapse can show as it crashing to ~0 or saturating near 1).
  - setup_usage           — mean P(setup move) over setup-legal HEALTHY states. Setup vanishing
                            over checkpoints is the myopic-collapse signature (the Dragon Dance
                            collapse writ large).

These are the checkpoint-side detectors. The run-side detectors (training entropy, game length /
tie rate, win-rate vs historical checkpoints) come from the run summaries + a self-play-vs-past
benchmark and are recorded by the monitor. Together they cover mode / myopic / stall / cycling
collapse.

Usage:
    python scripts/collapse_probe.py \
      --checkpoint checkpoints/curated/current-v2-500k.pt=v2-500k \
      --checkpoint checkpoints/curated/current-v2-600k.pt=v2-600k \
      --showdown-root /path/to/pokemon-showdown \
      --out evals/collapse_signals.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from pokezero.actions import ACTION_COUNT, MOVE_ACTION_COUNT
from pokezero.checkpoint_factors import HEALTHY_HP, build_corpus
from pokezero.neural_policy import evaluate_transformer_action_priors
from pokezero.online_client import build_agent
from pokezero.showdown import observation_from_player_state


def _legal_entropy(probs, legal_mask):
    """Shannon entropy (nats) of the action distribution restricted+renormalized to legal actions,
    plus the top action's mass. Restricting to legal actions keeps the signal comparable across
    states with different legal-action counts (illegal actions carry ~0 mass anyway)."""
    legal = [i for i, m in enumerate(legal_mask) if m]
    if not legal:
        return 0.0, 1.0
    mass = sum(probs[i] for i in legal) or 1.0
    ps = [probs[i] / mass for i in legal]
    entropy = -sum(p * math.log(p) for p in ps if p > 0.0)
    return entropy, max(ps)


def probe_checkpoint(label: str, checkpoint: str, showdown_root: str, corpus) -> dict:
    agent = build_agent(checkpoint, showdown_root, our_name="collapse", deterministic=True)

    def priors(state):
        obs = observation_from_player_state(
            state, category_vocab=agent.vocab, spec=agent.spec, dex=agent.dex,
            **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
        )
        return evaluate_transformer_action_priors(
            model=agent.policy.model, result=agent.policy.result, observations=[obs]
        )

    entropies, perplexities, near_det = [], [], 0
    switch_probs, setup_healthy = [], []
    for entry in corpus:
        probs = priors(entry.state)
        entropy, top = _legal_entropy(probs, entry.state.legal_action_mask)
        entropies.append(entropy)
        perplexities.append(math.exp(entropy))
        if top > 0.9:
            near_det += 1
        if entry.legal_switch:
            switch_probs.append(sum(probs[MOVE_ACTION_COUNT:ACTION_COUNT]))
        if entry.setup_slots and entry.active_hp > HEALTHY_HP:
            setup_healthy.append(sum(probs[slot] for slot, _ in entry.setup_slots))

    n = len(corpus)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    return {
        "label": label,
        "checkpoint": checkpoint,
        "states": n,
        "policy_entropy": mean(entropies),
        "action_perplexity": mean(perplexities),
        "near_deterministic_rate": round(near_det / n, 4) if n else 0.0,
        "switch_propensity": mean(switch_probs),
        "setup_usage_healthy": mean(setup_healthy),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="PATH[=LABEL]",
                        help="checkpoint to probe; repeatable. Optional =LABEL for display.")
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--corpus-games", type=int, default=60, help="unique games sampled for the corpus")
    parser.add_argument("--out", default=None, help="write the full result JSON here")
    args = parser.parse_args()

    print(f"[collapse] building shared corpus ({args.corpus_games} games)…", file=sys.stderr)
    corpus = build_corpus(args.showdown_root, num_games=args.corpus_games)
    print(f"[collapse] corpus: {len(corpus)} decision states", file=sys.stderr)

    rows = []
    for spec in args.checkpoint:
        path, _, label = spec.partition("=")
        label = label or Path(path).stem
        print(f"[collapse] probing {label}…", file=sys.stderr)
        row = probe_checkpoint(label, path, args.showdown_root, corpus)
        rows.append(row)
        print(f"  entropy={row['policy_entropy']} perplexity={row['action_perplexity']} "
              f"near_det={row['near_deterministic_rate']} switch={row['switch_propensity']} "
              f"setup={row['setup_usage_healthy']}", file=sys.stderr)

    payload = {"corpus_games": args.corpus_games, "corpus_states": len(corpus), "checkpoints": rows}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[collapse] wrote {out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
