"""Probe pairwise policy JS-divergence over a fixed decision-state corpus.

This is a read-only D0 diversity measurement. It builds the same checkpoint-
independent corpus used by ``collapse_probe.py``, evaluates each checkpoint's
legal-action prior on every state, and writes a compact artifact that
``diversity_population_dashboard.py`` can summarize.

Usage:
    python scripts/policy_js_divergence_probe.py \
      --checkpoint checkpoints/foundation-a.pt=a \
      --checkpoint checkpoints/foundation-b.pt=b \
      --showdown-root /path/to/pokemon-showdown \
      --out evals/policy-js-divergence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokezero.checkpoint_factors import build_corpus
from pokezero.neural_policy import evaluate_transformer_action_priors
from pokezero.online_client import build_agent
from pokezero.opponents import require_current_family_checkpoint_paths
from pokezero.showdown import observation_from_player_state


SCHEMA_VERSION = "pokezero.policy_js_divergence_probe.v1"


def _state_id(index: int, entry) -> str:
    return f"{index:05d}:seed={entry.seed}:player={entry.player}:turn={entry.turn}"


def probe_checkpoint(
    *,
    label: str,
    checkpoint: str,
    showdown_root: str,
    corpus,
    temperature: float,
    device: str | None,
) -> dict:
    agent = build_agent(checkpoint, showdown_root, our_name="policy-js", deterministic=True)
    model = getattr(agent.policy, "model", None)
    result = getattr(agent.policy, "result", None)
    if model is None or result is None:
        raise ValueError(f"{label} is not a transformer checkpoint policy")

    states = []
    for index, entry in enumerate(corpus):
        obs = observation_from_player_state(
            entry.state,
            category_vocab=agent.vocab,
            spec=agent.spec,
            dex=agent.dex,
            **({"feature_masks": agent.feature_masks} if agent.feature_masks is not None else {}),
        )
        probs = evaluate_transformer_action_priors(
            model=model,
            result=result,
            observations=[obs],
            temperature=temperature,
            device=device,
        )
        states.append(
            {
                "state_id": _state_id(index, entry),
                "seed": entry.seed,
                "player": entry.player,
                "turn": entry.turn,
                "active_species": entry.active_species,
                "legal_action_mask": [bool(value) for value in entry.state.legal_action_mask],
                "action_probabilities": [round(float(value), 8) for value in probs],
            }
        )
    return {
        "label": label,
        "checkpoint": checkpoint,
        "states": states,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="PATH[=LABEL]",
        help="checkpoint to probe; repeatable. Optional =LABEL for display.",
    )
    parser.add_argument("--showdown-root", required=True)
    parser.add_argument("--corpus-games", type=int, default=60, help="unique games sampled for the fixed corpus")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the full result JSON here")
    parser.add_argument(
        "--allow-legacy-checkpoints",
        action="store_true",
        help=(
            "allow no-belief/pre-v2 checkpoints for explicit historical reproduction; "
            "do not use for current diversity evals"
        ),
    )
    args = parser.parse_args()

    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    specs = []
    for spec in args.checkpoint:
        path, _, label = spec.partition("=")
        specs.append((label or Path(path).stem, path))
    if not args.allow_legacy_checkpoints:
        require_current_family_checkpoint_paths(
            (path for _, path in specs),
            context="policy-JS divergence probe",
        )
    print(f"[policy-js] building shared corpus ({args.corpus_games} games)…", file=sys.stderr)
    corpus = build_corpus(
        args.showdown_root,
        num_games=args.corpus_games,
        seed_start=args.seed_start,
        max_states=args.max_states,
    )
    print(f"[policy-js] corpus: {len(corpus)} decision states", file=sys.stderr)

    policies = []
    for label, path in specs:
        print(f"[policy-js] probing {label}…", file=sys.stderr)
        policies.append(
            probe_checkpoint(
                label=label,
                checkpoint=path,
                showdown_root=args.showdown_root,
                corpus=corpus,
                temperature=args.temperature,
                device=args.device,
            )
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus": {
            "games": args.corpus_games,
            "seed_start": args.seed_start,
            "max_states": args.max_states,
            "state_count": len(corpus),
        },
        "temperature": args.temperature,
        "policies": policies,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[policy-js] wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
