"""Render the turn-N choice-probability sample as a grouped horizontal bar chart — one panel
per decision state, one bar per checkpoint for every legal choice.

  python scripts/plot_choice_sample.py \
      --in evals/turn10_choice_sample.json \
      --out evals/turn10_choice_sample.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", default="evals/turn10_choice_sample.json")
    parser.add_argument("--out", default="evals/turn10_choice_sample.png")
    parser.add_argument("--min-prob", type=float, default=0.01, help="hide choices below this in every checkpoint")
    args = parser.parse_args()

    data = json.loads(Path(args.inp).read_text())
    checkpoints = data["checkpoints"]
    states = data["states"]

    ncols = 5
    nrows = -(-len(states) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.viridis(np.linspace(0.15, 0.75, len(checkpoints)))

    for ax, state in zip(axes, states):
        dists = state["checkpoints"]
        # keep choices where any checkpoint clears the threshold; order by last checkpoint desc
        labels = [c["label"] for c in state["legal_choices"]]
        labels = [l for l in labels if max(dists[cp].get(l, 0.0) for cp in checkpoints) >= args.min_prob]
        labels.sort(key=lambda l: dists[checkpoints[-1]].get(l, 0.0))  # ascending -> top of chart is largest

        y = np.arange(len(labels))
        h = 0.8 / len(checkpoints)
        for i, cp in enumerate(checkpoints):
            vals = [dists[cp].get(l, 0.0) * 100 for l in labels]
            offset = (i - (len(checkpoints) - 1) / 2) * h
            ax.barh(y + offset, vals, height=h, color=colors[i], label=cp)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(
            f"seed {state['seed']} · t{state['turn']}: {state['active']} vs {state['opponent_active']}",
            fontsize=8.5,
        )
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(axis="x", alpha=0.25)

    for ax in axes[len(states):]:
        ax.axis("off")

    fig.suptitle(
        f"Turn-{data['turn']} choice probabilities — {' vs '.join(checkpoints)} (bar = P(choice), %)",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[out] wrote {out_path} ({len(states)} panels, {len(checkpoints)} checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
