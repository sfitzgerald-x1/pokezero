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
from matplotlib.patches import Rectangle

STATUS_COLOR = {
    "tox": "#9c40b0", "psn": "#9c40b0", "brn": "#d0492a",
    "par": "#c8a000", "slp": "#6a7a8a", "frz": "#3390c8",
}


def _hp_color(hp: float) -> str:
    return "#37a35a" if hp > 0.5 else ("#e0a020" if hp > 0.2 else "#d0402a")


def _pretty(value) -> str:
    return str(value).replace("-", " ").title() if value else "—"


def _boost_text(boosts) -> str:
    if not boosts:
        return "—"
    return ", ".join(f"{'+' if v > 0 else ''}{v} {k.capitalize()[:3]}" for k, v in boosts.items())


def _draw_active_detail(ax, x0, width, detail):
    """Left column: the active mon's moveset, item, ability, status and stat changes."""
    if not detail:
        return
    ax.text(x0, 0.96, "ACTIVE MON", fontsize=10, fontweight="bold", va="top")
    y, lh = 0.82, 0.082
    ax.text(x0, y, detail.get("species") or "?", fontsize=10, fontweight="bold", va="center")
    y -= lh
    ax.text(x0, y, f"Item: {_pretty(detail.get('item'))}", fontsize=8.5, va="center")
    y -= lh
    ax.text(x0, y, f"Ability: {_pretty(detail.get('ability'))}", fontsize=8.5, va="center")
    y -= lh
    status = detail.get("status", "none")
    ax.text(x0, y, "Status:", fontsize=8.5, va="center")
    if status and status != "none":
        ax.text(x0 + width * 0.42, y, status.upper(), fontsize=7.5, color="white", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=STATUS_COLOR.get(status, "#666"), ec="none"))
        if status == "tox":
            stage = detail.get("toxic_stage") or 0
            ax.text(x0 + width * 0.58, y, f"×{stage}  (≈{stage}/16 HP/turn)",
                    fontsize=8, color="#9c40b0", va="center")
    else:
        ax.text(x0 + width * 0.42, y, "healthy", fontsize=8.5, color="#777", va="center")
    y -= lh
    ax.text(x0, y, f"Boosts: {_boost_text(detail.get('boosts'))}", fontsize=8.5, va="center")
    y -= lh * 1.2
    ax.text(x0, y, "Moves:", fontsize=8.5, fontweight="bold", va="center")
    y -= lh
    for move in detail.get("moves", []):
        disabled = move.get("disabled")
        color = "#bbbbbb" if disabled else "#222"
        pp = f"  ({move['pp']}/{move['maxpp']})" if move.get("maxpp") is not None else ""
        tag = "  [disabled]" if disabled else ""
        ax.text(x0 + 0.01, y, f"• {move.get('name') or '?'}{pp}{tag}", fontsize=8, color=color, va="center")
        y -= lh * 0.92


def _draw_team(ax, x0, width, header, mons):
    """Draw one side's team as a stack of name + HP-bar + status rows within [x0, x0+width]."""
    n = len(mons)
    ax.text(x0, 0.96, header, fontsize=10, fontweight="bold", va="top")
    top, rowh = 0.82, min(0.16, 0.82 / max(n, 1))
    bar_x = x0 + width * 0.44
    bar_w = width * 0.30
    for i, mon in enumerate(mons):
        y = top - i * rowh
        active, fainted, hp = mon["active"], mon["fainted"], mon["hp"]
        color = "#111111" if active else ("#b0b0b0" if fainted else "#555555")
        marker = "▶" if active else ("✗" if fainted else "•")
        weight = "bold" if active else "normal"
        ax.text(x0, y, marker, fontsize=9, color=color, va="center", fontweight=weight)
        ax.text(x0 + 0.02, y, mon["species"], fontsize=8.5, color=color, va="center", fontweight=weight)
        ax.add_patch(Rectangle((bar_x, y - 0.28 * rowh), bar_w, 0.42 * rowh, fill=False, ec="#999", lw=0.6))
        if not fainted and hp > 0:
            ax.add_patch(Rectangle((bar_x, y - 0.28 * rowh), bar_w * hp, 0.42 * rowh, color=_hp_color(hp), lw=0))
        ax.text(bar_x + bar_w + 0.008, y, "fnt" if fainted else f"{hp*100:.0f}%", fontsize=7.5, color=color, va="center")
        status = mon["status"]
        if status and status != "none":
            tag = status.upper()
            if status == "tox" and mon.get("toxic_stage"):
                tag += f" ×{mon['toxic_stage']}"
            ax.text(
                bar_x + bar_w + 0.05, y, tag, fontsize=7, color="white", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=STATUS_COLOR.get(status, "#666"), ec="none"),
            )


def _draw_state(ax, snapshot):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_active_detail(ax, 0.0, 0.26, snapshot.get("active_detail"))
    _draw_team(ax, 0.30, 0.33, "YOU (p1)", snapshot["self_team"])
    _draw_team(ax, 0.66, 0.33, "OPPONENT (revealed)", snapshot["opponent_team"])
    field = []
    if snapshot.get("weather"):
        field.append(f"weather:{snapshot['weather']}")
    if snapshot.get("self_side_conditions"):
        field.append("you:" + ",".join(snapshot["self_side_conditions"]))
    if snapshot.get("opponent_side_conditions"):
        field.append("opp:" + ",".join(snapshot["opponent_side_conditions"]))
    ax.text(0.30, -0.02, "field:  " + ("  ·  ".join(field) if field else "none"),
            fontsize=8, color="#555", va="top")


def _select_labels(state, checkpoints, min_prob):
    dists = state["checkpoints"]
    labels = [c["label"] for c in state["legal_choices"]]
    labels = [l for l in labels if max(dists[cp].get(l, 0.0) for cp in checkpoints) >= min_prob]
    labels.sort(key=lambda l: dists[checkpoints[-1]].get(l, 0.0))  # ascending -> largest on top
    return labels


def _plot_one(state, checkpoints, colors, min_prob):
    """Draw one decision: a game-state board panel on top, grouped choice-probability bars below."""
    dists = state["checkpoints"]
    labels = _select_labels(state, checkpoints, min_prob)
    snapshot = state.get("state")
    bars_h = max(2.2, 0.55 * len(labels) + 1.2)

    if snapshot:
        team_rows = max(len(snapshot["self_team"]), len(snapshot["opponent_team"]))
        detail_rows = 6 + len(snapshot.get("active_detail", {}).get("moves", []))
        state_h = 0.30 * max(team_rows, detail_rows) + 0.8
        fig = plt.figure(figsize=(12.0, state_h + bars_h))
        gs = fig.add_gridspec(2, 1, height_ratios=[state_h, bars_h], hspace=0.30)
        _draw_state(fig.add_subplot(gs[0]), snapshot)
        ax = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(8.5, bars_h))
    y = np.arange(len(labels))
    h = 0.8 / len(checkpoints)
    for i, cp in enumerate(checkpoints):
        vals = [dists[cp].get(l, 0.0) * 100 for l in labels]
        offset = (i - (len(checkpoints) - 1) / 2) * h
        bars = ax.barh(y + offset, vals, height=h, color=colors[i], label=cp)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("P(choice) %", fontsize=9)
    ax.set_title(
        f"seed {state['seed']} · turn {state['turn']}: "
        f"{state['active']} ({state['active_condition']}) vs {state['opponent_active']}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    if not state.get("state"):
        fig.tight_layout()
    return fig, ax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", default="evals/turn10_choice_sample.json")
    parser.add_argument("--out", default="evals/turn10_choice_sample.png")
    parser.add_argument(
        "--individual",
        dest="outdir",
        default=None,
        help="also write one readable PNG per decision into this directory",
    )
    parser.add_argument("--min-prob", type=float, default=0.01, help="hide choices below this in every checkpoint")
    args = parser.parse_args()

    data = json.loads(Path(args.inp).read_text())
    checkpoints = data["checkpoints"]
    states = data["states"]
    colors_all = plt.cm.viridis(np.linspace(0.15, 0.75, len(checkpoints)))

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for state in states:
            _plot_one(state, checkpoints, colors_all, args.min_prob)
            path = outdir / f"seed{state['seed']:02d}.png"
            plt.savefig(path, dpi=130, bbox_inches="tight")
            plt.close()
        print(f"[out] wrote {len(states)} per-decision plots to {outdir}/")

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
