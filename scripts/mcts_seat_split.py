#!/usr/bin/env python3
"""Split recorded MCTS head-to-head cells by the seat the search agent played.

The h2h harness writes one JSON shard per batch of games, each carrying a
``per_game`` list of ``{seed, search_seat, winner}``. Pooled cell scores average
the two seats, which hides any defect that fires on one seat only -- exactly the
shape of the value-orientation bug in
``docs/mcts_degradation_findings.md`` §10. This script recovers the split.

Cell identity comes from the shard FILENAME (``<arm>-<NN>.json`` -> ``<arm>``),
not from the ``config`` field: the control shards carry a leftover ``config``
string from the runner's defaults and would otherwise merge into a search cell.

Scoring matches ``pokezero.mcts_eval.scoring``: win = 1, tie/no-winner = 0.5,
loss = 0. Results are deduped on (arm, seat, seed) so overlapping shards cannot
double-count a game.

Usage::

    python3 scripts/mcts_seat_split.py <results-dir> [<results-dir> ...]

Runs on a bare python3 (no repo imports) so it can be dropped into a controller
pod and pointed at ``/shared``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; `wins` may be fractional (draws count 0.5)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(w1: float, n1: int, w2: float, n2: int) -> float:
    if not n1 or not n2:
        return float("nan")
    p1, p2 = w1 / n1, w2 / n2
    pooled = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else float("nan")


def game_score(winner: str | None, seat: str) -> float:
    if not winner or winner == "tie":
        return 0.5
    return 1.0 if winner == seat else 0.0


def arm_of(filename: str) -> str:
    return re.sub(r"-\d+$", "", filename[:-5])


def load(roots: list[str]) -> dict[tuple[str, str], dict[int, tuple[float, str | None]]]:
    cells: dict[tuple[str, str], dict[int, tuple[float, str | None]]] = defaultdict(dict)
    for root in roots:
        for name in sorted(os.listdir(root)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                continue
            per_game = payload.get("per_game")
            if not isinstance(per_game, list) or not per_game:
                continue
            arm = arm_of(name)
            for game in per_game:
                seat, seed = game.get("search_seat"), game.get("seed")
                if seat in ("p1", "p2") and seed is not None:
                    winner = game.get("winner")
                    cells[(arm, seat)][seed] = (game_score(winner, seat), winner)
    return cells


def main(roots: list[str]) -> int:
    if not roots:
        print(__doc__)
        return 2
    cells = load(roots)
    if not cells:
        print("no shards with a per_game list found", file=sys.stderr)
        return 1
    arms = sorted({arm for arm, _ in cells})

    header = (
        f"{'arm':<24} {'n1':>4} {'p1':>6} {'p1 Wilson95':>16}  "
        f"{'n2':>4} {'p2':>6} {'p2 Wilson95':>16}  {'p1-p2':>7} {'z':>6} "
        f"{'pooled':>7} {'p2 share':>9}"
    )
    print(header)
    print("-" * len(header))
    for arm in arms:
        p1 = cells.get((arm, "p1"), {})
        p2 = cells.get((arm, "p2"), {})
        n1, n2 = len(p1), len(p2)
        s1 = sum(value for value, _ in p1.values())
        s2 = sum(value for value, _ in p2.values())
        if not n1 or not n2:
            continue
        m1, m2 = s1 / n1, s2 / n2
        lo1, hi1 = wilson(s1, n1)
        lo2, hi2 = wilson(s2, n2)
        pooled = (s1 + s2) / (n1 + n2)
        deficit = pooled - 0.5
        # Each seat's contribution to the pooled deviation from the 0.5 null.
        part2 = (m2 - 0.5) * n2 / (n1 + n2)
        share = (part2 / deficit * 100.0) if abs(deficit) > 1e-9 else float("nan")
        print(
            f"{arm:<24} {n1:>4} {m1:>6.3f} [{lo1:.3f}, {hi1:.3f}]  "
            f"{n2:>4} {m2:>6.3f} [{lo2:.3f}, {hi2:.3f}]  "
            f"{m1 - m2:>+7.3f} {two_proportion_z(s1, n1, s2, n2):>+6.2f} "
            f"{pooled:>7.3f} {share:>8.0f}%"
        )

    print("\n# seed sets per seat (cells sharing a set are paired within a seat)")
    signatures: dict[tuple[str, str], frozenset[int]] = {
        key: frozenset(value) for key, value in cells.items()
    }
    groups: dict[frozenset[int], list[str]] = defaultdict(list)
    for (arm, seat), seeds in signatures.items():
        groups[seeds].append(f"{arm}/{seat}")
    for index, (seeds, members) in enumerate(
        sorted(groups.items(), key=lambda item: -len(item[1]))
    ):
        ordered = sorted(seeds)
        print(
            f"  set {index}: n={len(ordered)} range={ordered[0]}..{ordered[-1]} "
            f"parity={'even' if all(s % 2 == 0 for s in ordered) else 'odd' if all(s % 2 for s in ordered) else 'mixed'} "
            f"-> {len(members)} cells"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
