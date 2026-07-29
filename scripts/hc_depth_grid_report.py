#!/usr/bin/env python3
"""Render the handcrafted-leaf depth grid into the tables the findings doc needs.

Reads the per-cell JSON written by ``scripts/hc_depth_grid.py`` and emits:

1. the slope table (score + Wilson 95% per cell), at both the findings doc's
   comparability window (the first 100 seeds) and the full run;
2. paired seed-level comparisons between cells - every cell plays the SAME seeds
   against the SAME raw opponent, so the discordant-pair (McNemar/sign) test is
   the powerful read on "does depth change anything", not the overlap of two
   Wilson intervals;
3. the depth-ACTUALLY-reached table, which decides whether the depth knob binds.
   The crate counts node depth with the root at 0 and the cap bounds child
   CREATION, so a cap of ``d`` can show at most ``d - 1``.

Usage:
    python scripts/hc_depth_grid_report.py --dir docs/audit_artifacts/hc-depth-grid-20260729
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from hc_depth_grid import wilson_interval  # noqa: E402  (sibling script)

_CELL_ORDER = ["control", "hc-d1", "hc-d2", "hc-d4", "hc-d6", "hc-d8"]


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant pairs."""

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        required=True,
        action="append",
        help="Cell directory. Repeatable: pass a sims-axis directory to fold it "
        "into the same tables (cells are labelled from their own recorded sims).",
    )
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args(argv)

    cells: dict[str, dict] = {}
    for directory in args.dir:
        root = Path(directory)
        for name in _CELL_ORDER:
            path = root / f"{name}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            # A cell's identity is its configuration, so the label carries the
            # sims budget whenever more than one directory is folded in.
            label = name
            if len(args.dir) > 1 and payload.get("sims"):
                label = f"{name}-s{payload['sims']}"
            cells[label] = payload
    cells = dict(
        sorted(cells.items(), key=lambda item: (item[0] != "control", item[0]))
    )

    by_seed = {
        name: {row["seed"]: row for row in payload["results"]}
        for name, payload in cells.items()
    }
    common = sorted(set.intersection(*(set(rows) for rows in by_seed.values())))
    window = common[: args.window]

    def table(seeds: list[int], title: str) -> list[str]:
        lines = [
            "",
            f"### {title} (n = {len(seeds)})",
            "",
            "| cell | score | Wilson 95% | s/decision | fallback |",
            "|---|---|---|---|---|",
        ]
        for name, payload in cells.items():
            wins = sum(by_seed[name][seed]["score"] for seed in seeds)
            low, high = wilson_interval(wins, len(seeds))
            stats = payload.get("engine_stats") or {}
            per_decision = stats.get("wall_per_decision")
            fallback = stats.get("fallback_rate")
            lines.append(
                f"| {name} | {wins / len(seeds):.3f} | [{low:.3f}, {high:.3f}] | "
                f"{'' if per_decision is None else format(per_decision, '.3f')} | "
                f"{'' if fallback is None else format(fallback, '.4f')} |"
            )
        return lines

    def seat_table(seeds: list[int], title: str) -> list[str]:
        """Per-seat split.

        Mirroring cancels seat advantage in the pooled score, which also hides
        it. Seat asymmetry is a live hypothesis for the depth decay (the parity
        lane's seat-constant model-value inversion), so every cell reports both
        halves: the candidate on p1 (even seeds) and on p2 (odd seeds).
        """

        lines = [
            "",
            f"### {title} — per seat (n = {len(seeds)})",
            "",
            "| cell | p1 score | p1 Wilson 95% | n | p2 score | p2 Wilson 95% | n "
            "| p1 − p2 | z | p (uncorr.) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in cells:
            halves = {}
            for seat in ("p1", "p2"):
                rows = [
                    by_seed[name][seed]
                    for seed in seeds
                    if by_seed[name][seed]["candidate_seat"] == seat
                ]
                wins = sum(row["score"] for row in rows)
                halves[seat] = (
                    wins / len(rows) if rows else 0.0,
                    wilson_interval(wins, len(rows)),
                    len(rows),
                )
            p1, p2 = halves["p1"], halves["p2"]
            # Unpaired two-proportion z: the seats are DISJOINT seed sets
            # (even vs odd), so there is nothing to pair on.
            pooled = (p1[0] * p1[2] + p2[0] * p2[2]) / (p1[2] + p2[2])
            se = math.sqrt(pooled * (1 - pooled) * (1 / p1[2] + 1 / p2[2]))
            z = (p1[0] - p2[0]) / se if se > 0 else 0.0
            pval = math.erfc(abs(z) / math.sqrt(2))
            lines.append(
                f"| {name} | {p1[0]:.3f} | [{p1[1][0]:.3f}, {p1[1][1]:.3f}] | {p1[2]} | "
                f"{p2[0]:.3f} | [{p2[1][0]:.3f}, {p2[1][1]:.3f}] | {p2[2]} | "
                f"{p1[0] - p2[0]:+.3f} | {z:+.2f} | {pval:.3f} |"
            )
        return lines

    out = ["## Slope table"]
    out += table(window, "Findings-doc comparability window")
    out += table(common, "Full run")
    out += seat_table(common, "Full run")

    out += [
        "",
        "## Paired seed-level comparison (same seeds, same opponent)",
        "",
        "| A | B | A wins only | B wins only | agree | two-sided sign p |",
        "|---|---|---|---|---|---|",
    ]
    names = list(cells)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            b = c = agree = 0
            for seed in common:
                a_score = by_seed[left][seed]["score"]
                b_score = by_seed[right][seed]["score"]
                if a_score == b_score:
                    agree += 1
                elif a_score > b_score:
                    b += 1
                else:
                    c += 1
            out.append(
                f"| {left} | {right} | {b} | {c} | {agree} | "
                f"{_binom_two_sided(b, c):.4f} |"
            )

    out += [
        "",
        "## Depth actually reached",
        "",
        "Crate node depth, root = 0. The cap bounds child creation "
        "(`depth + 1 >= max_depth`), so a BINDING cap `d` tops out at `d - 1`.",
        "",
        "| cell | cap | ceiling if binding | max reached | mean reached | histogram |",
        "|---|---|---|---|---|---|",
    ]
    for name, payload in cells.items():
        stats = payload.get("engine_stats")
        if not stats or payload.get("depth") is None:
            continue
        cap = payload["depth"]
        out.append(
            f"| {name} | {cap} | {cap - 1} | {stats['depth_reached_max']} | "
            f"{stats.get('depth_reached_mean', 0.0):.3f} | "
            f"{json.dumps(stats['depth_reached_histogram'])} |"
        )

    if "hc-d6" in cells and "hc-d8" in cells:
        same = sum(
            1
            for seed in common
            if by_seed["hc-d6"][seed]["winner"] == by_seed["hc-d8"][seed]["winner"]
        )
        out += [
            "",
            f"d6 vs d8 identical per-seed winner on {same}/{len(common)} seeds.",
        ]

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
